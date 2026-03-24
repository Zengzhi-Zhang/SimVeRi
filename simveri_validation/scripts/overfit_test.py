# scripts/overfit_test.py
"""
Minimal overfit test: train on 10 IDs (~40 images) for 100 iterations.
Bypasses most of FastReID machinery to isolate whether the model+loss can learn.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "fast-reid-master"))

from src.path_utils import get_default_simveri_root

from fastreid.config import get_cfg
from fastreid.modeling import build_model


def main():
    # === Config (same as train_baseline.py C1) ===
    cfg = get_cfg()
    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.PRETRAIN = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.NUM_CLASSES = 10  # Only 10 IDs for overfit test
    cfg.MODEL.HEADS.EMBEDDING_DIM = 0
    cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.MODEL.HEADS.NECK_FEAT = "after"
    cfg.MODEL.HEADS.WITH_BNNECK = True
    cfg.MODEL.HEADS.CLS_LAYER = "Linear"
    cfg.MODEL.LOSSES.NAME = ("CrossEntropyLoss", "TripletLoss")
    cfg.MODEL.LOSSES.CE.EPSILON = 0.1
    cfg.MODEL.LOSSES.CE.SCALE = 1.0
    cfg.MODEL.LOSSES.TRI.MARGIN = 0.3
    cfg.MODEL.LOSSES.TRI.HARD_MINING = True
    cfg.MODEL.LOSSES.TRI.NORM_FEAT = False
    cfg.MODEL.LOSSES.TRI.SCALE = 1.0
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.SIZE_TRAIN = [320, 320]

    # === Build model ===
    model = build_model(cfg)
    model.train()
    model.cuda()
    print(f"Model built. NUM_CLASSES={cfg.MODEL.HEADS.NUM_CLASSES}")

    # === Load 10 IDs from SimVeRi train ===
    data_root = get_default_simveri_root()
    train_dir = os.path.join(data_root, "images", "train")

    # Collect images by PID
    pid_to_files = {}
    for fname in os.listdir(train_dir):
        if not fname.endswith(".jpg"):
            continue
        parts = fname.split("_")
        pid = int(parts[0])
        if pid not in pid_to_files:
            pid_to_files[pid] = []
        pid_to_files[pid].append(os.path.join(train_dir, fname))

    # Take first 10 PIDs, 4 images each
    selected_pids = sorted(pid_to_files.keys())[:10]
    pid_map = {pid: idx for idx, pid in enumerate(selected_pids)}

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    labels = []
    for pid in selected_pids:
        files = pid_to_files[pid][:4]
        for fpath in files:
            img = Image.open(fpath).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(pid_map[pid])

    images = torch.stack(images).cuda()
    labels = torch.tensor(labels, dtype=torch.long).cuda()
    print(f"Loaded {len(images)} images, {len(selected_pids)} IDs")
    print(f"Labels: {labels.tolist()}")

    # === Optimizer (same as train_baseline.py) ===
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)

    # === Training loop ===
    print("\n" + "=" * 60)
    print("Overfit Test: 200 iterations on 10 IDs")
    print("=" * 60)

    from fastreid.utils.events import EventStorage

    with EventStorage(0) as storage:
        for step in range(200):
            batch_idx = []
            sampled_pids = np.random.choice(len(selected_pids), size=6, replace=False)
            for p in sampled_pids:
                pid_indices = [i for i, l in enumerate(labels.tolist()) if l == p]
                if len(pid_indices) >= 4:
                    batch_idx.extend(np.random.choice(pid_indices, size=4, replace=False).tolist())
                else:
                    batch_idx.extend(pid_indices)
                    # pad with repeats
                    while len(batch_idx) % 4 != 0:
                        batch_idx.append(np.random.choice(pid_indices).item())

            batch_images = images[batch_idx]
            batch_labels = labels[batch_idx]

            # Forward
            batched_inputs = {
                "images": batch_images,
                "targets": batch_labels,
            }
            loss_dict = model(batched_inputs)
            total_loss = sum(loss_dict.values())

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log
            if step % 10 == 0 or step < 5:
                loss_cls = loss_dict.get("loss_cls", torch.tensor(0)).item()
                loss_tri = loss_dict.get("loss_triplet", torch.tensor(0)).item()

                # Compute accuracy manually
                with torch.no_grad():
                    outputs = model.heads(model.backbone(model.preprocess_image(batched_inputs)), batch_labels)
                    logits = outputs["pred_class_logits"]
                    preds = logits.argmax(dim=1)
                    acc = (preds == batch_labels).float().mean().item()

                print(f"  Step {step:3d}: loss_cls={loss_cls:.4f}  loss_tri={loss_tri:.4f}  "
                      f"total={total_loss.item():.4f}  acc={acc:.4f}")

            storage.step()

    print("\n" + "=" * 60)
    if acc > 0.5:
        print("[OK] Overfit test passed - the model can learn.")
        print("  Issue is likely hyperparameter/scale related, not a pipeline bug.")
    else:
        print("[FAIL] Overfit test failed - the model cannot learn even 10 IDs.")
        print("  There is a fundamental pipeline/architecture bug.")
    print("=" * 60)


if __name__ == "__main__":
    main()
