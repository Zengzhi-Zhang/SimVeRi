# scripts/extract_features.py
"""
Feature extraction script
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root, get_validation_pretrained_path


def main():
    parser = argparse.ArgumentParser(description="Extract features from SimVeRi dataset")
    parser.add_argument("--data-root", type=str, default=get_default_simveri_root())
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Resize input images to this square size before feature extraction (should match training cfg.INPUT.SIZE_TEST).",
    )
    parser.add_argument("--use-veri", action="store_true", help="Force using VeRi pretrained model")
    args = parser.parse_args()
    
    print("="*70)
    print("SimVeRi Feature Extraction")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    from src.dataset.simveri_loader import SimVeRiDataset
    from src.models.feature_extractor import FeatureExtractor
    
    if not args.output_dir:
        args.output_dir = os.path.join(PROJECT_ROOT, "outputs", "features")
    os.makedirs(args.output_dir, exist_ok=True)
    
    trained_model_candidates = [
        os.path.join(PROJECT_ROOT, "outputs", "models", "simveri_v2_320", "model_final.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "models", "simveri_v2", "model_final.pth"),
    ]
    trained_model_path = next((p for p in trained_model_candidates if os.path.exists(p)), trained_model_candidates[0])
    veri_model_path = get_validation_pretrained_path()
    
    if args.model_path:
        model_path = args.model_path
        print(f"\nUsing specified model: {model_path}")
    elif args.use_veri and os.path.exists(veri_model_path):
        model_path = veri_model_path
        print(f"\nUsing VeRi pretrained model: {model_path}")
    elif os.path.exists(trained_model_path):
        model_path = trained_model_path
        print(f"\nUsing our trained model: {model_path}")
    elif os.path.exists(veri_model_path):
        model_path = veri_model_path
        print(f"\nUsing VeRi pretrained model: {model_path}")
    else:
        model_path = None
        print("\nNo model found, using torchvision ResNet-50")
    
    print("\n" + "-"*50)
    print("Loading Dataset")
    print("-"*50)
    dataset = SimVeRiDataset(args.data_root, verbose=True)
    
    print("\n" + "-"*50)
    print("Initializing Feature Extractor")
    print("-"*50)
    extractor = FeatureExtractor(model_path=model_path, device=args.device, input_size=args.input_size)
    print(f"  Feature dimension: {extractor.get_feature_dim()}")
    
    print("\n" + "="*50)
    print("Extracting Gallery Features")
    print("="*50)
    gallery_paths = [s.image_path for s in dataset.gallery_samples]
    gallery_features = extractor.extract_batch(gallery_paths, batch_size=args.batch_size)
    np.save(os.path.join(args.output_dir, 'gallery_features.npy'), gallery_features)
    print(f"  Saved: gallery_features.npy ({gallery_features.shape})")
    
    gallery_info = [{
        'index': i,
        'image_name': s.image_name,
        'vehicle_id': s.vehicle_id,
        'camera_id': s.camera_id,
        'timestamp': s.timestamp,
        'position': s.position.tolist(),
        'heading': s.heading,
        'speed': s.speed,
        'occlusion': s.occlusion,
        'is_twins': s.is_twins,
        'twins_group': s.twins_group
    } for i, s in enumerate(dataset.gallery_samples)]
    with open(os.path.join(args.output_dir, 'gallery_info.json'), 'w') as f:
        json.dump(gallery_info, f, indent=2)
    print(f"  Saved: gallery_info.json ({len(gallery_info)} samples)")
    
    print("\n" + "="*50)
    print("Extracting Query Features")
    print("="*50)
    query_paths = [s.image_path for s in dataset.query_samples]
    query_features = extractor.extract_batch(query_paths, batch_size=args.batch_size)
    np.save(os.path.join(args.output_dir, 'query_features.npy'), query_features)
    print(f"  Saved: query_features.npy ({query_features.shape})")
    
    query_info = [{
        'index': i,
        'image_name': s.image_name,
        'vehicle_id': s.vehicle_id,
        'camera_id': s.camera_id,
        'timestamp': s.timestamp,
        'position': s.position.tolist(),
        'heading': s.heading,
        'speed': s.speed,
        'occlusion': s.occlusion,
        'is_twins': s.is_twins,
        'twins_group': s.twins_group
    } for i, s in enumerate(dataset.query_samples)]
    with open(os.path.join(args.output_dir, 'query_info.json'), 'w') as f:
        json.dump(query_info, f, indent=2)
    print(f"  Saved: query_info.json ({len(query_info)} samples)")

    # Save metadata for consistency checks in evaluation
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
    
    print("\n" + "="*50)
    print("Feature Statistics")
    print("="*50)
    print(f"  Feature dimension: {gallery_features.shape[1]}")
    print(f"  Gallery: {len(gallery_info)} (Base: {sum(1 for s in gallery_info if not s['is_twins'])}, Twins: {sum(1 for s in gallery_info if s['is_twins'])})")
    print(f"  Query: {len(query_info)} (Base: {sum(1 for s in query_info if not s['is_twins'])}, Twins: {sum(1 for s in query_info if s['is_twins'])})")
    
    print("\n" + "="*70)
    print("Feature extraction completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
