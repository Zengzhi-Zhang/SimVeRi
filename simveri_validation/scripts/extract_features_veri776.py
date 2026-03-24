# scripts/extract_features_veri776.py
"""
VeRi-776 feature extraction script.
Based on extract_features.py but uses VeRi776Dataset loader.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_veri776_root


def main():
    parser = argparse.ArgumentParser(description="Extract features from VeRi-776 dataset")
    parser.add_argument("--data-root", type=str, default=get_default_veri776_root())
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (e.g. model_final.pth)")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input-size", type=int, default=320,
                        help="Input size (should match training cfg.INPUT.SIZE_TEST).")
    args = parser.parse_args()

    print("=" * 70)
    print("VeRi-776 Feature Extraction")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    from src.dataset.veri776_loader import VeRi776Dataset
    from src.models.feature_extractor import FeatureExtractor

    # Output directory
    if not args.output_dir:
        args.output_dir = os.path.join(PROJECT_ROOT, "outputs", "features_veri776")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = args.model_path
    print(f"\nModel: {model_path}")

    # Load dataset
    print("\n" + "-" * 50)
    print("Loading VeRi-776 Dataset")
    print("-" * 50)
    dataset = VeRi776Dataset(args.data_root, verbose=True)

    # Feature extractor
    print("\n" + "-" * 50)
    print("Initializing Feature Extractor")
    print("-" * 50)
    extractor = FeatureExtractor(model_path=model_path, device=args.device, input_size=args.input_size)
    print(f"  Feature dimension: {extractor.get_feature_dim()}")

    # Extract gallery features
    print("\n" + "=" * 50)
    print("Extracting Gallery Features")
    print("=" * 50)
    gallery_paths = [s.image_path for s in dataset.gallery_samples]
    gallery_features = extractor.extract_batch(gallery_paths, batch_size=args.batch_size)
    np.save(os.path.join(args.output_dir, 'gallery_features.npy'), gallery_features)
    print(f"  Saved: gallery_features.npy ({gallery_features.shape})")

    gallery_info = [{
        'index': i,
        'image_name': s.image_name,
        'vehicle_id': s.vehicle_id,
        'camera_id': s.camera_id,
        'is_twins': s.is_twins,
    } for i, s in enumerate(dataset.gallery_samples)]
    with open(os.path.join(args.output_dir, 'gallery_info.json'), 'w') as f:
        json.dump(gallery_info, f, indent=2)
    print(f"  Saved: gallery_info.json ({len(gallery_info)} samples)")

    # Extract query features
    print("\n" + "=" * 50)
    print("Extracting Query Features")
    print("=" * 50)
    query_paths = [s.image_path for s in dataset.query_samples]
    query_features = extractor.extract_batch(query_paths, batch_size=args.batch_size)
    np.save(os.path.join(args.output_dir, 'query_features.npy'), query_features)
    print(f"  Saved: query_features.npy ({query_features.shape})")

    query_info = [{
        'index': i,
        'image_name': s.image_name,
        'vehicle_id': s.vehicle_id,
        'camera_id': s.camera_id,
        'is_twins': s.is_twins,
    } for i, s in enumerate(dataset.query_samples)]
    with open(os.path.join(args.output_dir, 'query_info.json'), 'w') as f:
        json.dump(query_info, f, indent=2)
    print(f"  Saved: query_info.json ({len(query_info)} samples)")

    # Metadata
    meta = {
        "data_root": os.path.abspath(args.data_root),
        "model_path": model_path,
        "input_size": int(args.input_size),
        "gallery_count": len(gallery_info),
        "query_count": len(query_info),
        "generated_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: features_meta.json")

    # Statistics
    print("\n" + "=" * 50)
    print("Feature Statistics")
    print("=" * 50)
    print(f"  Feature dimension: {gallery_features.shape[1]}")
    print(f"  Gallery: {len(gallery_info)}")
    print(f"  Query: {len(query_info)}")

    print("\n" + "=" * 70)
    print("Feature extraction completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
