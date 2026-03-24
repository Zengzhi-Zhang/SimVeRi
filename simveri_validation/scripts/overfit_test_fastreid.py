# scripts/overfit_test_fastreid.py
"""
Overfit test using FastReID's DefaultTrainer on a 10-ID subset.
This isolates whether the issue is in DefaultTrainer vs raw PyTorch training.
"""
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root, get_validation_output_dir

def create_subset_dataset(src_root, dst_root, num_ids=10):
    """Create a subset with only num_ids vehicle IDs."""
    src_dirs = {
        "train": os.path.join(src_root, "images", "train"),
        "query": os.path.join(src_root, "images", "query"),
        "gallery": os.path.join(src_root, "images", "gallery"),
    }

    # Find the first num_ids PIDs from train
    pid_set = set()
    for fname in sorted(os.listdir(src_dirs["train"])):
        if not fname.endswith(".jpg"):
            continue
        pid = int(fname.split("_")[0])
        pid_set.add(pid)
        if len(pid_set) >= num_ids:
            break

    selected_pids = pid_set
    print(f"Selected {len(selected_pids)} PIDs: {sorted(selected_pids)}")

    # Copy images for selected PIDs
    for split, src_dir in src_dirs.items():
        dst_dir = os.path.join(dst_root, "images", split)
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for fname in os.listdir(src_dir):
            if not fname.endswith(".jpg"):
                continue
            pid = int(fname.split("_")[0])
            if pid in selected_pids:
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
                count += 1
        print(f"  {split}: copied {count} images")

    return len(selected_pids)


def main():
    data_root = get_default_simveri_root()
    subset_root = get_validation_output_dir("subset_10ids")
    output_dir = get_validation_output_dir("models", "overfit_test")

    # Create subset
    print("Creating 10-ID subset...")
    if os.path.exists(subset_root):
        shutil.rmtree(subset_root)
    num_ids = create_subset_dataset(data_root, subset_root, num_ids=10)

    # CRITICAL: Set env variable so FastReID's data loader uses our subset root
    # (FastReID uses os.getenv("FASTREID_DATASETS", "datasets"), NOT cfg.DATASETS.ROOT)
    os.environ["FASTREID_DATASETS"] = subset_root

    # Now import fastreid modules AFTER setting env and AFTER creating subset
    import src.dataset.simveri_fastreid
    import src.dataset.veri776_fastreid
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer

    # Also need to override the hardcoded SIMVERI_ROOT in simveri_fastreid.py
    src.dataset.simveri_fastreid.SIMVERI_ROOT = subset_root

    # Build config (same as train_baseline.py C1)
    cfg = get_cfg()
    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.PRETRAIN = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.NUM_CLASSES = 0  # Let auto_scale_hyperparams compute from dataset
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
    cfg.MODEL.LOSSES.TRI.NORM_FEAT = True
    cfg.MODEL.LOSSES.TRI.SCALE = 1.0
    cfg.INPUT.SIZE_TRAIN = [320, 320]
    cfg.INPUT.SIZE_TEST = [320, 320]
    cfg.INPUT.FLIP.ENABLED = True
    cfg.INPUT.FLIP.PROB = 0.5
    cfg.INPUT.PADDING.ENABLED = True
    cfg.INPUT.PADDING.SIZE = 10
    cfg.INPUT.PADDING.MODE = "constant"
    cfg.INPUT.REA.ENABLED = True
    cfg.INPUT.REA.PROB = 0.2
    cfg.INPUT.REA.VALUE = [123.675, 116.28, 103.53]
    cfg.INPUT.CJ.ENABLED = True
    cfg.INPUT.CJ.PROB = 0.2
    cfg.INPUT.CJ.BRIGHTNESS = 0.2
    cfg.INPUT.CJ.CONTRAST = 0.15
    cfg.INPUT.CJ.SATURATION = 0.1
    cfg.INPUT.CJ.HUE = 0.1
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.NUM_INSTANCE = 4
    cfg.DATALOADER.SAMPLER_TRAIN = "NaiveIdentitySampler"
    cfg.SOLVER.MAX_EPOCH = 10  # 10 epochs for overfit test
    cfg.SOLVER.BASE_LR = 0.00035
    cfg.SOLVER.BIAS_LR_FACTOR = 2.0
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.SCHED = "MultiStepLR"
    cfg.SOLVER.STEPS = [7, 9]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 10  # Very short warmup for small dataset
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.OPT = "Adam"
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.DATASETS.NAMES = ("SimVeRi",)
    cfg.DATASETS.TESTS = ("SimVeRi",)
    cfg.DATASETS.ROOT = subset_root
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.EVAL_PERIOD = 0  # Disable eval entirely (no query/gallery in subset)
    cfg.TEST.IMS_PER_BATCH = 24
    cfg.MODEL.DEVICE = "cuda"
    cfg.CUDNN_BENCHMARK = True

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Overfit Test with DefaultTrainer ({num_ids} IDs, 5 epochs)")
    print("=" * 60)
    print(f"If loss_cls drops below {0.5 * 6.12:.2f} and cls_accuracy > 0.3,")
    print(f"the DefaultTrainer pipeline is working correctly.")
    print("=" * 60)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n" + "=" * 60)
    print("Overfit test complete. Check the logs above for loss_cls and cls_accuracy trends.")
    print("=" * 60)


if __name__ == "__main__":
    main()
