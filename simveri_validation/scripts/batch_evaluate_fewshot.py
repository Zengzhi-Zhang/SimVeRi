# scripts/batch_evaluate_fewshot.py
"""
Batch-evaluate all Stage-C few-shot models.

Iterate over all C2_r*/C3_r* models under the comparison-output directory,
run VeRi-776 evaluation one by one, and aggregate the results into one CSV file.

Usage (run locally from the simveri_validation directory):
    python scripts/batch_evaluate_fewshot.py               # GPU run (recommended)
    python scripts/batch_evaluate_fewshot.py --cpu          # CPU fallback
    python scripts/batch_evaluate_fewshot.py --batch-size 64  # reduce GPU memory usage
"""

import os
import sys
import gc
import csv
import re
import argparse
import statistics
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_veri776_root, get_validation_output_dir

# CRITICAL: Set FASTREID_DATASETS BEFORE importing fastreid modules.
# FastReID caches _root = os.getenv("FASTREID_DATASETS") at import time
# (fastreid/data/build.py line 34), so setting it later has no effect.
_DEFAULT_VERI_ROOT = get_default_veri776_root()
# Pre-scan sys.argv for --veri-root
for i, arg in enumerate(sys.argv):
    if arg == "--veri-root" and i + 1 < len(sys.argv):
        _DEFAULT_VERI_ROOT = sys.argv[i + 1]
        break
os.environ["FASTREID_DATASETS"] = _DEFAULT_VERI_ROOT

import src.dataset.simveri_fastreid
import src.dataset.veri776_fastreid

import torch

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.evaluation import ReidEvaluator, inference_on_dataset
from fastreid.utils.checkpoint import Checkpointer

# Skip Cython compilation on Windows
ReidEvaluator._compile_dependencies = lambda self: None


def parse_exp_name(dirname):
    """Parse experiment directory name like 'C2_r0.05_s42' -> (exp, ratio, seed).
    Also handles 'C1', 'C2', 'C3' (100% baselines).
    """
    # Few-shot: C2_r0.05_s42, C3t_r0.1_s43, etc.
    m = re.match(r'^(C[1234]tgroups)_r(\d+\.groups\d*)_s(\d+)$', dirname)
    if m:
        return m.group(1), float(m.group(2)), int(m.group(3))
    # Full: C1, C2, C3
    m = re.match(r'^(C[1234]tgroups)$', dirname)
    if m:
        return m.group(1), 1.0, 0
    return None, None, None


def build_cfg(model_dir, veri_root, use_cpu=False, batch_size=128):
    """Build FastReID config for evaluation."""
    cfg = get_cfg()

    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""

    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.EMBEDDING_DIM = 0
    cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.MODEL.HEADS.NECK_FEAT = "after"
    cfg.MODEL.HEADS.WITH_BNNECK = True
    cfg.MODEL.HEADS.CLS_LAYER = "Linear"

    # Auto-infer NUM_CLASSES from classifier head weight
    weights_path = os.path.join(model_dir, "model_final.pth")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    # FastReID EmbeddingHead uses "heads.weight" (nn.Parameter); some forks use "heads.classifier.weight"
    cls_key = None
    for candidate in ("heads.weight", "heads.classifier.weight"):
        if candidate in state:
            cls_key = candidate
            break
    if cls_key:
        cfg.MODEL.HEADS.NUM_CLASSES = state[cls_key].shape[0]
    else:
        cfg.MODEL.HEADS.NUM_CLASSES = 576

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu" if use_cpu else "cuda"

    cfg.INPUT.SIZE_TEST = [320, 320]
    cfg.DATASETS.NAMES = ("VeRi776",)
    cfg.DATASETS.TESTS = ("VeRi776",)
    cfg.DATASETS.ROOT = veri_root
    cfg.TEST.IMS_PER_BATCH = batch_size
    cfg.TEST.METRIC = "cosine"
    cfg.TEST.RERANK.ENABLED = False
    cfg.TEST.AQE.ENABLED = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = model_dir

    return cfg


def extract_metrics(results):
    """Extract mAP/Rank-1/5/10 from FastReID results dict.

    Handles both flat dicts ({"mAP": ..., "Rank-1": ...}) and nested
    dicts ({"VeRi776": {"mAP": ..., ...}}).

    FastReID ReidEvaluator returns values in 0-100 scale (percentages).
    """
    # Handle nested results (single top-level key whose value is a dict)
    if len(results) == 1:
        key = list(results.keys())[0]
        if isinstance(results[key], dict):
            results = results[key]

    mAP = float(results.get("mAP", 0))
    r1 = float(results.get("Rank-1", 0))
    r5 = float(results.get("Rank-5", 0))
    r10 = float(results.get("Rank-10", 0))

    return mAP, r1, r5, r10


def evaluate_one(model_dir, veri_root, use_cpu=False, batch_size=128):
    """Evaluate a single model and return (mAP, R1, R5, R10)."""
    cfg = build_cfg(model_dir, veri_root, use_cpu, batch_size)

    model = DefaultTrainer.build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    test_loader, num_query = DefaultTrainer.build_test_loader(cfg, dataset_name="VeRi776")
    evaluator = ReidEvaluator(cfg, num_query)
    results = inference_on_dataset(model, test_loader, evaluator, flip_test=False)

    mAP, r1, r5, r10 = extract_metrics(results)

    # Free GPU memory explicitly
    del model, test_loader, evaluator
    gc.collect()
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.empty_cache()

    return mAP, r1, r5, r10


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all few-shot models")
    parser.add_argument("--models-dir", type=str,
                        default=get_validation_output_dir("models"),
                        help="Directory containing all experiment subdirectories")
    parser.add_argument("--veri-root", type=str, default=get_default_veri776_root(),
                        help="VeRi-776 dataset root")
    parser.add_argument("--output-csv", type=str,
                        default=get_validation_output_dir("fewshot_results.csv"),
                        help="Output CSV file for results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    os.environ["FASTREID_DATASETS"] = args.veri_root
    output_parent = os.path.dirname(args.output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    # Discover all experiment directories
    all_dirs = sorted(os.listdir(args.models_dir))
    experiments = []
    for d in all_dirs:
        model_dir = os.path.join(args.models_dir, d)
        if not os.path.isfile(os.path.join(model_dir, "model_final.pth")):
            continue
        exp, ratio, seed = parse_exp_name(d)
        if exp is None:
            continue
        experiments.append((d, exp, ratio, seed, model_dir))

    print(f"Found {len(experiments)} experiments to evaluate")
    print(f"Device: {'CPU' if args.cpu else 'CUDA'}")
    print("=" * 70)

    # Run evaluations
    rows = []
    for i, (dirname, exp, ratio, seed, model_dir) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Evaluating {dirname} ...")
        try:
            mAP, r1, r5, r10 = evaluate_one(
                model_dir, args.veri_root, args.cpu, args.batch_size)

            row = {
                "exp": exp, "ratio": ratio, "seed": seed,
                "mAP": round(mAP, 2), "Rank-1": round(r1, 2),
                "Rank-5": round(r5, 2), "Rank-10": round(r10, 2),
                "dirname": dirname,
            }
            rows.append(row)
            print(f"  => mAP={row['mAP']:.2f}, Rank-1={row['Rank-1']:.2f}")

            # Save intermediate results after each evaluation
            _save_csv(rows, args.output_csv)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            # Clean up GPU memory after failure
            gc.collect()
            if torch.cuda.is_available() and not args.cpu:
                torch.cuda.empty_cache()
            rows.append({
                "exp": exp, "ratio": ratio, "seed": seed,
                "mAP": -1, "Rank-1": -1, "Rank-5": -1, "Rank-10": -1,
                "dirname": dirname,
            })
            _save_csv(rows, args.output_csv)

    # Final save
    _save_csv(rows, args.output_csv)

    # Print summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY: Few-Shot Learning Curve")
    print("=" * 70)
    _print_summary(rows)


def _save_csv(rows, path):
    """Save results to CSV."""
    if not rows:
        return
    fieldnames = ["exp", "ratio", "seed", "mAP", "Rank-1", "Rank-5", "Rank-10", "dirname"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Saved] {path} ({len(rows)} rows)")


def _print_summary(rows):
    """Print mean+-std table grouped by (exp, ratio)."""
    grouped = defaultdict(list)
    for r in rows:
        if r["mAP"] < 0:
            continue
        grouped[(r["exp"], r["ratio"])].append(r)

    print(f"\n{'Exp':>4s} {'Ratio':>6s} {'#Seeds':>6s} {'mAP':>14s} {'Rank-1':>14s}")
    print("-" * 50)

    for (exp, ratio) in sorted(grouped.keys()):
        items = grouped[(exp, ratio)]
        maps = [x["mAP"] for x in items]
        r1s = [x["Rank-1"] for x in items]
        n = len(maps)

        if n == 1:
            print(f"{exp:>4s} {ratio:>6.0%} {n:>6d} {maps[0]:>14.2f} {r1s[0]:>14.2f}")
        else:
            m_mean, m_std = statistics.mean(maps), statistics.stdev(maps)
            r_mean, r_std = statistics.mean(r1s), statistics.stdev(r1s)
            print(f"{exp:>4s} {ratio:>6.0%} {n:>6d} {m_mean:>7.2f}+-{m_std:<5.2f} {r_mean:>7.2f}+-{r_std:<5.2f}")

    print()


if __name__ == "__main__":
    main()
