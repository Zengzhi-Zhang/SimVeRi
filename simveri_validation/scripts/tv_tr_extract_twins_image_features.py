#!/usr/bin/env python
"""
Extract per-image features for the Twins extras package.

Twins extras is not a standard SimVeRi dataset root (no images/{train,gallery,query}),
so we provide this dedicated extractor to produce:
  - twins_image_features.npy (N, 2048)  L2-normalized float32
  - twins_image_index.json   {image_name: row_index}
  - twins_features_meta.json

GPU/torch is required (same environment as other feature extractors).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Twins extras image features")
    p.add_argument("--twins-root", type=str, required=True, help="Path to <release>/extras/twins")
    p.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N images (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    twins_root = args.twins_root
    meta_dir = os.path.join(twins_root, "metadata")
    st_path = os.path.join(meta_dir, "spatiotemporal_twins.json")
    images_dir = os.path.join(twins_root, "images")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"spatiotemporal_twins.json not found: {st_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Twins images dir not found: {images_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    import json

    with open(st_path, "r", encoding="utf-8") as f:
        st = json.load(f)
    ann = st.get("annotations", {})
    if not isinstance(ann, dict):
        raise ValueError(f"Invalid spatiotemporal_twins.json format: {st_path}")

    image_names = sorted(list(ann.keys()))
    if args.limit and int(args.limit) > 0:
        image_names = image_names[: int(args.limit)]

    image_paths = [os.path.join(images_dir, n) for n in image_names]
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Twins images (example: {missing[0]})")

    print("=" * 70)
    print("SimVeRi Twins Image Feature Extraction")
    print("=" * 70)
    print(f"Start time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Twins root:  {twins_root}")
    print(f"Images:      {len(image_paths)}")
    print(f"Model:       {args.model_path}")
    print(f"Device:      {args.device}")
    print(f"Input size:  {int(args.input_size)}")
    print(f"Batch size:  {int(args.batch_size)}")
    print(f"Output dir:  {args.output_dir}")

    from src.models.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(model_path=args.model_path, device=args.device, input_size=int(args.input_size))
    feats = extractor.extract_batch(image_paths, batch_size=int(args.batch_size), show_progress=True).astype(np.float32)

    out_npy = os.path.join(args.output_dir, "twins_image_features.npy")
    np.save(out_npy, feats)

    index: Dict[str, int] = {n: i for i, n in enumerate(image_names)}
    out_index = os.path.join(args.output_dir, "twins_image_index.json")
    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    meta = {
        "twins_root": os.path.abspath(twins_root),
        "spatiotemporal": os.path.abspath(st_path),
        "images_dir": os.path.abspath(images_dir),
        "model_path": os.path.abspath(args.model_path),
        "device": str(args.device),
        "input_size": int(args.input_size),
        "batch_size": int(args.batch_size),
        "image_count": int(len(image_names)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_meta = os.path.join(args.output_dir, "twins_features_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("-" * 70)
    print("Done")
    print(f"Saved: {out_npy} {feats.shape}")
    print(f"Saved: {out_index}")
    print(f"Saved: {out_meta}")
    print("=" * 70)


if __name__ == "__main__":
    main()
