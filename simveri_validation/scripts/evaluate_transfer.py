# scripts/evaluate_transfer.py
"""
Stage-C syn-to-real transfer evaluation script

Evaluate the mAP/Rank-1 performance of the C1/C2/C3 model groups on VeRi-776.
Usage:
    python scripts/evaluate_transfer.py --model-dir ./outputs/models/C1 --exp-id C1
    python scripts/evaluate_transfer.py --model-dir ./outputs/models/C2 --exp-id C2
    python scripts/evaluate_transfer.py --model-dir ./outputs/models/C3 --exp-id C3
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_veri776_root

# Register datasets
import src.dataset.simveri_fastreid
import src.dataset.veri776_fastreid

import torch

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.evaluation import ReidEvaluator, inference_on_dataset
from fastreid.utils.checkpoint import Checkpointer

# Skip Cython compilation on Windows (pure Python fallback is used automatically)
ReidEvaluator._compile_dependencies = lambda self: None


def setup_cfg(args):
    """Build config for evaluation on VeRi-776."""
    cfg = get_cfg()

    # Model architecture (must match training)
    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.PRETRAIN = False  # No need to download ImageNet weights
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""

    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.EMBEDDING_DIM = 0
    cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.MODEL.HEADS.NECK_FEAT = "after"
    cfg.MODEL.HEADS.WITH_BNNECK = True
    cfg.MODEL.HEADS.CLS_LAYER = "Linear"

    # Auto-infer NUM_CLASSES from checkpoint to support any experiment
    # (C1/C2/C3 full or few-shot with varying class counts).
    weights_path = os.path.join(args.model_dir, "model_final.pth")
    if os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        # FastReID uses "heads.classifier.weight" or "heads.weight" depending on CLS_LAYER
        cls_key = None
        for candidate in ("heads.weight", "heads.classifier.weight"):
            if candidate in state:
                cls_key = candidate
                break
        if cls_key:
            cfg.MODEL.HEADS.NUM_CLASSES = state[cls_key].shape[0]
            print(f"[Auto] NUM_CLASSES inferred from checkpoint: {cfg.MODEL.HEADS.NUM_CLASSES}")
        else:
            cfg.MODEL.HEADS.NUM_CLASSES = 576
            print(f"[WARN] Classifier key not found in checkpoint, defaulting to 576")
    else:
        cfg.MODEL.HEADS.NUM_CLASSES = 576
        print(f"[WARN] Checkpoint not found at {weights_path}, defaulting to 576")

    # Weights
    cfg.MODEL.WEIGHTS = os.path.join(args.model_dir, "model_final.pth")
    cfg.MODEL.DEVICE = "cuda" if not args.cpu else "cpu"

    # Input (must match training)
    cfg.INPUT.SIZE_TEST = [320, 320]

    # Dataset: always evaluate on VeRi-776
    cfg.DATASETS.NAMES = ("VeRi776",)
    cfg.DATASETS.TESTS = ("VeRi776",)
    cfg.DATASETS.ROOT = args.veri_root

    # Test config
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.TEST.METRIC = "cosine"
    cfg.TEST.RERANK.ENABLED = False
    cfg.TEST.AQE.ENABLED = False

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.OUTPUT_DIR = args.model_dir

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate transfer experiment on VeRi-776")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model_final.pth")
    parser.add_argument("--exp-id", type=str, required=True,
                        help="Experiment ID (e.g. C1, C2, C3, C2-5pct, etc.)")
    parser.add_argument("--veri-root", type=str, default=get_default_veri776_root(),
                        help="VeRi-776 dataset root (containing image_train, image_test, image_query)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Test batch size")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Ensure FastReID's data loader uses the correct dataset root.
    os.environ["FASTREID_DATASETS"] = args.veri_root

    cfg = setup_cfg(args)

    print("=" * 70)
    print(f"Transfer Evaluation: {args.exp_id} on VeRi-776")
    print("=" * 70)
    print(f"  Model: {cfg.MODEL.WEIGHTS}")
    print(f"  NUM_CLASSES: {cfg.MODEL.HEADS.NUM_CLASSES}")
    print(f"  Device: {cfg.MODEL.DEVICE}")
    print("=" * 70)

    # Build model
    model = DefaultTrainer.build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Build test loader and evaluator
    test_loader, num_query = DefaultTrainer.build_test_loader(cfg, dataset_name="VeRi776")
    evaluator = ReidEvaluator(cfg, num_query)

    # Run evaluation
    results = inference_on_dataset(model, test_loader, evaluator, flip_test=False)

    print("\n" + "=" * 70)
    print(f"Results for {args.exp_id}:")
    print("=" * 70)
    for key, val in results.items():
        print(f"  {key}: {val}")
    print("=" * 70)


if __name__ == "__main__":
    main()
