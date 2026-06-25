# scripts/train_baseline.py
"""
SimVeRi baseline training script v2.0
Optimized release: corrected class count, Circle Loss, and stronger augmentation
"""

import os
import sys
import argparse
from typing import Optional, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


import src.dataset.simveri_fastreid
import src.dataset.veri776_fastreid
from src.path_utils import (
    get_default_simveri_root,
    get_validation_output_dir,
    get_validation_pretrained_path,
)

# FastReID
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer

# ---------------------------------------------------------------------------
# Windows stability patch. FastReID's DataLoaderX prefetches batches to the GPU
# from a background thread on a side CUDA stream; on Windows this async cross-
# thread CUDA interaction can intermittently crash/stall training (observed here
# as the process dying mid-epoch with the model evicted from VRAM). We replace
# the TRAIN DataLoaderX with a standard torch DataLoader (no CUDA prefetch) and
# move each batch to the model device in the main training step instead.
# ---------------------------------------------------------------------------
from collections.abc import Mapping as _Mapping
from torch.utils.data import DataLoader as _TorchDataLoader
import fastreid.data.build as _fastreid_data_build
from fastreid.engine.train_loop import SimpleTrainer as _SimpleTrainer
from fastreid.utils.params import ContiguousParams as _ContiguousParams


class _TrainDataLoaderNoCudaPrefetch(_TorchDataLoader):
    """Drop-in for fastreid DataLoaderX on the train loader: ignores local_rank
    and keeps ordinary torch DataLoader semantics (no background CUDA prefetch)."""

    def __init__(self, local_rank=None, **kwargs):
        super().__init__(**kwargs)


_fastreid_data_build.DataLoaderX = _TrainDataLoaderNoCudaPrefetch


def _move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, _Mapping):
        return type(batch)((k, _move_batch_to_device(v, device)) for k, v in batch.items())
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(_move_batch_to_device(v, device) for v in batch))
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    return batch


def _run_step_main_thread_to_device(self):
    import time as _time
    assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
    start = _time.perf_counter()
    data = next(self._data_loader_iter)
    _m = self.model.module if hasattr(self.model, "module") else self.model
    if hasattr(_m, "device"):
        data = _move_batch_to_device(data, _m.device)
    data_time = _time.perf_counter() - start
    loss_dict = self.model(data)
    losses = sum(loss_dict.values())
    self.optimizer.zero_grad(set_to_none=False)
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    self.optimizer.step()
    if isinstance(self.param_wrapper, _ContiguousParams):
        self.param_wrapper.assert_buffer_is_valid()


_SimpleTrainer.run_step = _run_step_main_thread_to_device


# SimVeRi "twins" ID range used by the optional dataset variants in src/dataset/simveri_fastreid.py.
# Note: in the released layout the core images/ contains only base + occlusion vehicles
# (mapped IDs 0001-0650); Twins are distributed separately under extras/twins/ (mapped IDs
# 0431-0555). Because these per-component identifier spaces are not globally unique, Twins are
# best identified by original_id prefix ('H') / fleet_id or by source folder rather than by a
# mapped-ID range. These constants are used only by the optional SimVeRiBase / SimVeRiTwins
# training variants; the released core benchmark and the reference baseline do not use them.
SIMVERI_TWINS_ID_START = 431
SIMVERI_TWINS_ID_END = 530


def _parse_pid_from_filename(img_name: str) -> Optional[int]:
    parts = img_name.split("_")
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _pid_in_twins_range(pid: int) -> bool:
    return SIMVERI_TWINS_ID_START <= pid <= SIMVERI_TWINS_ID_END


def count_train_ids(dataset_root: str, dataset_name: str = "SimVeRi",
                    fewshot_ratio: float = 1.0, fewshot_seed: int = 42,
                    simveri_root: str = "") -> int:
    """Count unique vehicle IDs from training images.

    When *fewshot_ratio* < 1.0 and *dataset_name* is a VeRi-776 variant,
    the count reflects the few-shot subset so that NUM_CLASSES is correct.
    """
    if dataset_name == "VeRi776SimVeRiMixed":
        # Mixed: SimVeRi IDs + VeRi few-shot IDs
        from src.dataset.veri776_fastreid import collect_veri776_pids, sample_fewshot_pids
        veri_dir = os.path.join(dataset_root, "image_train")
        sim_dir = os.path.join(simveri_root, "images", "train") if simveri_root else ""
        veri_count = 0
        if os.path.isdir(veri_dir):
            veri_pids = collect_veri776_pids(veri_dir)
            if fewshot_ratio < 1.0:
                veri_pids = sample_fewshot_pids(veri_pids, fewshot_ratio, fewshot_seed)
            veri_count = len(veri_pids)
        sim_count = 0
        if sim_dir and os.path.isdir(sim_dir):
            sim_pids = set()
            for img_name in os.listdir(sim_dir):
                if img_name.endswith(".jpg"):
                    pid = _parse_pid_from_filename(img_name)
                    if pid is not None:
                        sim_pids.add(pid)
            sim_count = len(sim_pids)
        return sim_count + veri_count

    if dataset_name in ("VeRi776", "VeRi776FewShot"):
        # Use shared collect function (same filter as VeRi776 dataset class)
        from src.dataset.veri776_fastreid import collect_veri776_pids, sample_fewshot_pids
        train_dir = os.path.join(dataset_root, "image_train")
        if not os.path.isdir(train_dir):
            return 0
        sorted_pids = collect_veri776_pids(train_dir)
        if fewshot_ratio < 1.0:
            sorted_pids = sample_fewshot_pids(sorted_pids, fewshot_ratio, fewshot_seed)
        return len(sorted_pids)

    # SimVeRi variants
    train_dir = os.path.join(dataset_root, "images", "train")
    if not os.path.isdir(train_dir):
        return 0
    pids = set()
    for img_name in os.listdir(train_dir):
        if not img_name.endswith(".jpg"):
            continue
        pid = _parse_pid_from_filename(img_name)
        if pid is None:
            continue

        if dataset_name == "SimVeRiBase" and _pid_in_twins_range(pid):
            continue
        if dataset_name == "SimVeRiTwins" and (not _pid_in_twins_range(pid)):
            continue

        pids.add(pid)
    return len(pids)


def parse_args():
    parser = argparse.ArgumentParser(description="SimVeRi baseline training (FastReID)")
    parser.add_argument("--dataset-root", type=str, default=get_default_simveri_root())
    parser.add_argument("--simveri-root", type=str, default=get_default_simveri_root(),
                        help="SimVeRi dataset root (for mixed training C4).")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SimVeRi",
        choices=["SimVeRi", "SimVeRiBase", "SimVeRiTwins", "VeRi776", "VeRi776SimVeRiMixed"],
        help="Dataset to train on (must be registered in FastReID).",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default="C1",
        help="Experiment preset that controls the loss recipe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=get_validation_output_dir("models", "{exp}_{dataset}"),
        help="Output directory. Supports placeholders: {exp}, {dataset}, {ratio}, {seed}.",
    )
    parser.add_argument("--fewshot-ratio", type=float, default=1.0,
                        help="Fraction of training IDs to use (0 < ratio <= 1.0). Default 1.0 = full dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for few-shot ID sampling.")
    parser.add_argument("--weights", type=str, default=get_validation_pretrained_path(),
                        help="Path to pretrained weights. Use 'none' for ImageNet-only initialization.")

    # Input / augmentation
    parser.add_argument("--input-size", type=int, default=320, help="Train/test input size (square).")
    parser.add_argument("--rea-prob", type=float, default=0.2, help="Random Erasing probability.")
    parser.add_argument("--cj-prob", type=float, default=0.2, help="ColorJitter probability.")

    # Solver
    parser.add_argument("--batch-size", type=int, default=24, help="Images per batch (total).")
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs. LR milestones auto-scale when != 150.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes (0 = single-thread, can stall on Windows).")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in --output-dir (restores optimizer/scheduler/epoch).")
    parser.add_argument("--base-lr", type=float, default=None, help="If set, override BASE_LR. Otherwise auto-scale from lr-ref.")
    parser.add_argument("--lr-ref", type=float, default=0.00035, help="Reference LR for lr-ref-batch.")
    parser.add_argument("--lr-ref-batch", type=int, default=32, help="Reference batch size for lr scaling.")

    # Transfer-aware fine-tuning (P2 experiment)
    parser.add_argument("--heads-lr-factor", type=float, default=1.0,
                        help="LR multiplier for head layers relative to backbone. "
                             "E.g. 10.0 means head LR = 10x backbone LR.")
    parser.add_argument("--freeze-backbone-iters", type=int, default=0,
                        help="Freeze backbone for this many iterations at start of training.")
    parser.add_argument("--reset-bn", action="store_true",
                        help="Reset BN running stats after loading pretrained weights.")

    return parser.parse_args()


def apply_loss_config(cfg) -> Tuple[str, ...]:
    """Apply stable loss recipe: CE + TripletLoss (hard mining).

    All experiments (C1/C2/C3) use the SAME loss config so the only
    variable is training data / init weights - fair comparison.
    """
    cfg.MODEL.LOSSES.NAME = ("CrossEntropyLoss", "TripletLoss")

    cfg.MODEL.LOSSES.CE.EPSILON = 0.1
    cfg.MODEL.LOSSES.CE.SCALE = 1.0

    cfg.MODEL.LOSSES.TRI.MARGIN = 0.3
    cfg.MODEL.LOSSES.TRI.HARD_MINING = True
    cfg.MODEL.LOSSES.TRI.NORM_FEAT = False
    cfg.MODEL.LOSSES.TRI.SCALE = 1.0
    return cfg.MODEL.LOSSES.NAME


def setup_cfg(args):
    """Create the optimized training configuration."""
    cfg = get_cfg()
    dataset_root = args.dataset_root
    dataset_name = args.dataset_name
    num_classes = count_train_ids(dataset_root, dataset_name,
                                  fewshot_ratio=args.fewshot_ratio,
                                  fewshot_seed=args.seed,
                                  simveri_root=args.simveri_root)

    # NaiveIdentitySampler needs at least IMS_PER_BATCH / NUM_INSTANCE identities
    # per batch (default 24/4=6). Fewer IDs will cause the sampler to hang.
    min_ids = int(args.batch_size) // 4  # NUM_INSTANCE = 4
    if 0 < num_classes < min_ids:
        raise ValueError(
            f"Few-shot sampling produced only {num_classes} IDs, but "
            f"NaiveIdentitySampler needs at least {min_ids} IDs "
            f"(IMS_PER_BATCH={args.batch_size} / NUM_INSTANCE=4). "
            f"Increase --fewshot-ratio or decrease --batch-size."
        )
    
    # ----------------------------------------------------------------------------
    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    
    # Backbone: ResNet-50-IBN
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.PRETRAIN = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""  # use the default ImageNet pretraining

    # Use pretrained weights for initialization ("none" or empty = ImageNet only)
    cfg.MODEL.WEIGHTS = "" if (not args.weights or args.weights.lower() == "none") else args.weights
    
    # Head
    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.NUM_CLASSES = num_classes if num_classes > 0 else 250
    cfg.MODEL.HEADS.EMBEDDING_DIM = 0   # keep the original 2048-dimensional head
    cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.MODEL.HEADS.NECK_FEAT = "after"
    cfg.MODEL.HEADS.WITH_BNNECK = True
    cfg.MODEL.HEADS.CLS_LAYER = "Linear"
    
    # ----------------------------------------------------------------------------
    # IMPORTANT: CE + Circle (gamma=128, scale=1.0) is easy to stall (loss_cls ~ ln(C) for many epochs).
    # Use a stable loss preset per experiment id.
    apply_loss_config(cfg)
    
    # ----------------------------------------------------------------------------
    cfg.INPUT.SIZE_TRAIN = [int(args.input_size), int(args.input_size)]
    cfg.INPUT.SIZE_TEST = [int(args.input_size), int(args.input_size)]
    
    cfg.INPUT.FLIP.ENABLED = True
    cfg.INPUT.FLIP.PROB = 0.5
    cfg.INPUT.PADDING.ENABLED = True
    cfg.INPUT.PADDING.SIZE = 10
    cfg.INPUT.PADDING.MODE = "constant"
    
    cfg.INPUT.REA.ENABLED = True
    cfg.INPUT.REA.PROB = float(args.rea_prob)
    cfg.INPUT.REA.VALUE = [123.675, 116.28, 103.53]  # ImageNet mean (0-255 RGB)
    
    cfg.INPUT.CJ.ENABLED = True
    cfg.INPUT.CJ.PROB = float(args.cj_prob)
    cfg.INPUT.CJ.BRIGHTNESS = 0.2
    cfg.INPUT.CJ.CONTRAST = 0.15
    cfg.INPUT.CJ.SATURATION = 0.1
    cfg.INPUT.CJ.HUE = 0.1
    
    # ----------------------------------------------------------------------------
    cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    cfg.DATALOADER.NUM_INSTANCE = 4  # sample 4 images per identity
    cfg.DATALOADER.SAMPLER_TRAIN = "NaiveIdentitySampler"
    
    # ----------------------------------------------------------------------------
    cfg.SOLVER.MAX_EPOCH = int(args.epochs)
    if args.base_lr is not None:
        cfg.SOLVER.BASE_LR = float(args.base_lr)
    else:
        cfg.SOLVER.BASE_LR = float(args.lr_ref) * float(args.batch_size) / float(args.lr_ref_batch)
    cfg.SOLVER.BIAS_LR_FACTOR = 2.0
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
    cfg.SOLVER.IMS_PER_BATCH = int(args.batch_size)
    
    cfg.SOLVER.SCHED = "MultiStepLR"
    # LR milestones: default keeps the validated 150-epoch schedule; other epoch
    # budgets scale milestones to ~0.5 and ~0.83 of the run so decay stays inside
    # training (60 epochs -> [30, 50], matching configs/simveri_resnet50_ibn.yml).
    if int(args.epochs) == 150:
        cfg.SOLVER.STEPS = [60, 100, 130]
    else:
        _E = int(args.epochs)
        cfg.SOLVER.STEPS = [m for m in (round(0.5 * _E), round(0.83 * _E)) if 0 < m < _E]
    cfg.SOLVER.GAMMA = 0.1
    
    # Warmup
    cfg.SOLVER.WARMUP_FACTOR = 0.01
    # Scale warmup iters for few-shot: estimate iters/epoch from num_classes,
    # then cap warmup at ~10 epochs to avoid dominating the training schedule.
    num_instance = cfg.DATALOADER.NUM_INSTANCE  # 4
    est_imgs = num_classes * num_instance  # conservative lower bound
    iters_per_epoch = max(1, est_imgs // cfg.SOLVER.IMS_PER_BATCH)
    warmup_iters = min(1000, iters_per_epoch * 10)
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    cfg.SOLVER.OPT = "Adam"
    cfg.SOLVER.MOMENTUM = 0.9

    # Transfer-aware fine-tuning: differential LR and optional backbone freeze
    cfg.SOLVER.HEADS_LR_FACTOR = float(args.heads_lr_factor)
    if args.freeze_backbone_iters > 0:
        cfg.MODEL.FREEZE_LAYERS = ["backbone"]
        cfg.SOLVER.FREEZE_ITERS = args.freeze_backbone_iters
    
    # Checkpoint
    cfg.SOLVER.CHECKPOINT_PERIOD = 1  # save every epoch (Windows resilience: clean resume after a crash)
    
    # ----------------------------------------------------------------------------
    cfg.DATASETS.NAMES = (dataset_name,)
    cfg.DATASETS.TESTS = (dataset_name,)
    cfg.DATASETS.ROOT = dataset_root
    
    # ----------------------------------------------------------------------------
    # Separate experiment output to avoid mixing with previous runs.
    cfg.OUTPUT_DIR = args.output_dir
    
    # ----------------------------------------------------------------------------
    # DefaultTrainer always expects at least one evaluation result; EvalHook uses epoch-based period.
    # Set to MAX_EPOCH+1 so it runs exactly once at the end (after_train).
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_EPOCH + 1
    cfg.TEST.IMS_PER_BATCH = 128
    
    # ----------------------------------------------------------------------------
    cfg.MODEL.DEVICE = "cuda"
    cfg.CUDNN_BENCHMARK = True
    
    return cfg


def main():
    args = parse_args()

    # Validate fewshot_ratio
    if args.fewshot_ratio <= 0.0 or args.fewshot_ratio > 1.0:
        raise ValueError(f"--fewshot-ratio must be in (0, 1.0], got {args.fewshot_ratio}")

    # Set global training seed for reproducibility
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Few-shot: switch dataset to VeRi776FewShot and configure sampling
    if args.fewshot_ratio < 1.0:
        from src.dataset.veri776_fastreid import set_fewshot_config
        set_fewshot_config(ratio=args.fewshot_ratio, seed=args.seed)
        if args.dataset_name == "VeRi776":
            args.dataset_name = "VeRi776FewShot"

    # Mixed training (C4): configure SimVeRi root
    if args.dataset_name == "VeRi776SimVeRiMixed":
        from src.dataset.veri776_fastreid import set_mixed_simveri_root
        set_mixed_simveri_root(args.simveri_root)

    # Ensure FastReID's data loader uses the correct dataset root.
    # FastReID reads os.environ["FASTREID_DATASETS"] (default "datasets") as
    os.environ["FASTREID_DATASETS"] = args.dataset_root

    args.output_dir = args.output_dir.format(
        exp=args.exp_id, dataset=args.dataset_name.lower(),
        ratio=args.fewshot_ratio, seed=args.seed,
    )
    cfg = setup_cfg(args)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Dump config for reproducibility (custom entry scripts don't always get a config.yaml by default).
    cfg_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(cfg.dump())
    except Exception as e:
        print(f"[WARN] Failed to write config to: {cfg_path} ({e})")

    # Save selected PIDs for few-shot experiments (reproducibility)
    if args.fewshot_ratio < 1.0:
        from src.dataset.veri776_fastreid import collect_veri776_pids, sample_fewshot_pids
        train_dir = os.path.join(args.dataset_root, "image_train")
        sorted_pids = collect_veri776_pids(train_dir)
        selected = sample_fewshot_pids(sorted_pids, args.fewshot_ratio, args.seed)
        pids_path = os.path.join(cfg.OUTPUT_DIR, "selected_pids.txt")
        with open(pids_path, "w", encoding="utf-8") as f:
            f.write(f"# fewshot_ratio={args.fewshot_ratio}, seed={args.seed}\n")
            f.write(f"# {len(selected)}/{len(sorted_pids)} IDs selected\n")
            for pid in selected:
                f.write(f"{pid}\n")
        print(f"[FewShot] Saved {len(selected)} selected PIDs to {pids_path}")

    print("=" * 70)
    print(f"Baseline Training v2.0 [{args.dataset_name}] ({args.exp_id})")
    print("=" * 70)
    print(f"\nConfiguration summary:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Model: ResNet-50-IBN")
    print(f"  Number of classes: {cfg.MODEL.HEADS.NUM_CLASSES}")
    print(f"  Losses: {', '.join(cfg.MODEL.LOSSES.NAME)}")
    print(f"  Epochs: {cfg.SOLVER.MAX_EPOCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  LR decay steps: {cfg.SOLVER.STEPS}")
    print(f"  Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Augmentations: Flip + REA + ColorJitter")
    print(f"  Initial weights: {cfg.MODEL.WEIGHTS if cfg.MODEL.WEIGHTS else 'ImageNet'}")
    if args.fewshot_ratio < 1.0:
        print(f"  Few-Shot: ratio={args.fewshot_ratio}, seed={args.seed}")
    if args.heads_lr_factor != 1.0:
        print(f"  Heads LR Factor: {args.heads_lr_factor}x")
    if args.freeze_backbone_iters > 0:
        print(f"  Freeze Backbone: {args.freeze_backbone_iters} iters")
    if args.reset_bn:
        print(f"  Reset BN: enabled")
    print(f"  Warmup: {cfg.SOLVER.WARMUP_ITERS} iters")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print("=" * 70)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print(f"[resume] resume={args.resume}, start_epoch={trainer.start_epoch}, max_epoch={trainer.max_epoch}")
    if args.resume and trainer.checkpointer.has_checkpoint() and trainer.start_epoch <= 0:
        raise RuntimeError("Resume requested but start_epoch is 0; checkpoint may lack an 'epoch' field.")

    # Reset BN running stats after loading pretrained weights
    # This removes domain-specific statistics from SimVeRi pretraining
    if args.reset_bn and cfg.MODEL.WEIGHTS:
        print("[Transfer-Aware] Resetting BN running stats...")
        for m in trainer.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.num_batches_tracked.zero_()
    
    print("\nStart training...")
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training finished")
    print(f"Model saved to: {cfg.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
