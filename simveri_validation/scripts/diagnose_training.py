# scripts/diagnose_training.py
"""
Diagnose why C1 training failed: cls_accuracy=0.0 and loss_cls stuck near ln(530) ~= 6.27
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.dataset.simveri_fastreid
import src.dataset.veri776_fastreid
from src.path_utils import get_default_simveri_root, get_validation_output_dir

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.data import build_reid_train_loader


def check_labels(cfg):
    """Check whether training labels are contiguous in the 0..N-1 range."""
    print("=" * 60)
    print("CHECK 1: Label Contiguity")
    print("=" * 60)

    loader = build_reid_train_loader(cfg)
    dataset = loader.dataset

    # Collect all PIDs from the dataset
    all_pids = set()
    for item in dataset:
        if isinstance(item, dict):
            pid = item.get("targets", item.get("pids", None))
        elif isinstance(item, (list, tuple)):
            pid = item[1]  # (img_path, pid, camid)
        else:
            print(f"  Unknown item type: {type(item)}")
            return
        if isinstance(pid, torch.Tensor):
            pid = pid.item()
        all_pids.add(int(pid))

    sorted_pids = sorted(all_pids)
    num_pids = len(sorted_pids)
    expected = list(range(num_pids))

    print(f"  Unique PIDs: {num_pids}")
    print(f"  PID range: {sorted_pids[0]} .. {sorted_pids[-1]}")
    print(f"  Expected range: 0 .. {num_pids - 1}")
    print(f"  NUM_CLASSES in config: {cfg.MODEL.HEADS.NUM_CLASSES}")

    if sorted_pids == expected:
        print("  [OK] Labels are contiguous 0..N-1")
    else:
        missing = set(expected) - all_pids
        extra = all_pids - set(expected)
        print(f"  [FAIL] Labels are not contiguous")
        if missing:
            print(f"    Missing from 0..{num_pids-1}: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            print(f"    Extra IDs outside 0..{num_pids-1}: {sorted(extra)[:20]}{'...' if len(extra) > 20 else ''}")

    if num_pids != cfg.MODEL.HEADS.NUM_CLASSES:
        print(f"  [FAIL] MISMATCH: {num_pids} unique PIDs vs NUM_CLASSES={cfg.MODEL.HEADS.NUM_CLASSES}")
    else:
        print(f"  [OK] NUM_CLASSES matches the number of unique PIDs")


def check_batch_composition(cfg):
    """Check whether one batch contains positive pairs."""
    print("\n" + "=" * 60)
    print("CHECK 2: Batch Composition (Positive Pairs)")
    print("=" * 60)

    loader = build_reid_train_loader(cfg)
    it = iter(loader)
    batch = next(it)

    if isinstance(batch, dict):
        pids = batch.get("targets", batch.get("pids", None))
    elif isinstance(batch, (list, tuple)):
        pids = batch[1]
    else:
        print(f"  Unknown batch type: {type(batch)}")
        return

    if isinstance(pids, torch.Tensor):
        pids_list = pids.tolist()
    else:
        pids_list = list(pids)

    print(f"  Batch size: {len(pids_list)}")
    print(f"  PIDs in batch: {pids_list}")

    from collections import Counter
    pid_counts = Counter(pids_list)
    num_unique = len(pid_counts)
    has_positive_pairs = any(c >= 2 for c in pid_counts.values())

    print(f"  Unique PIDs: {num_unique}")
    print(f"  PID distribution: {dict(pid_counts)}")

    if has_positive_pairs:
        print("  [OK] Batch has positive pairs (same-ID samples)")
    else:
        print("  [FAIL] No positive pairs in the batch; TripletLoss will fail.")


def check_model_gradients(cfg):
    """Check whether classifier parameters receive gradients."""
    print("\n" + "=" * 60)
    print("CHECK 3: Classifier Gradients")
    print("=" * 60)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    model = trainer.model

    # Print all parameter groups
    print("\n  Model parameter groups:")
    for name, param in model.named_parameters():
        if "classifier" in name or "heads" in name:
            print(f"    {name}: shape={list(param.shape)}, requires_grad={param.requires_grad}")

    # Do one forward pass to check gradients
    loader = build_reid_train_loader(cfg)
    it = iter(loader)
    batch = next(it)

    model.train()
    loss_dict = model(batch)
    total_loss = sum(loss_dict.values())
    total_loss.backward()

    print("\n  After one backward pass:")
    for name, param in model.named_parameters():
        if "classifier" in name or "heads" in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"    {name}: grad_norm={grad_norm:.6f}")
            else:
                print(f"    {name}: grad=None [FAIL]")

    print(f"\n  Loss dict: {{{', '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items())}}}")


def main():
    sys.argv = sys.argv[:1]  # Clear args

    from scripts.train_baseline import setup_cfg, parse_args

    # Simulate args
    class Args:
        dataset_root = get_default_simveri_root()
        dataset_name = "SimVeRi"
        exp_id = "C1"
        output_dir = get_validation_output_dir("models", "C1_diagnose")
        weights = "none"
        input_size = 320
        rea_prob = 0.2
        cj_prob = 0.2
        batch_size = 24
        base_lr = None
        lr_ref = 0.00035
        lr_ref_batch = 32

    args = Args()
    cfg = setup_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("SimVeRi C1 Training Diagnostics")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"NUM_CLASSES: {cfg.MODEL.HEADS.NUM_CLASSES}")
    print(f"Losses: {cfg.MODEL.LOSSES.NAME}")

    check_labels(cfg)
    check_batch_composition(cfg)
    check_model_gradients(cfg)

    print("\n" + "=" * 60)
    print("Diagnostics complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
