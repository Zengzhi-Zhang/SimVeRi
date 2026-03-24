#!/usr/bin/env python
"""
Aggregate per-image features into per-tracklet features for TR.

Base mode:
  - reads SimVeRi GG features: gallery_features.npy/query_features.npy + *_info.json
  - aggregates features for images listed in tracklets_base.json

Twins mode:
  - reads twins_image_features.npy + twins_image_index.json
  - aggregates features for images listed in tracklets_twins.json

Output:
  - tracklet_features_<tag>.npy
  - tracklet_index_<tag>.json
  - tracklet_features_meta_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import l2_normalize, load_tracklets, uniform_subsample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-image features into per-tracklet features")
    p.add_argument("--tracklets-json", type=str, required=True, help="tracklets_base.json or tracklets_twins.json")
    p.add_argument("--tag", type=str, required=True, choices=("base", "twins"), help="Output tag")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Base sources
    p.add_argument("--gg-features-dir", type=str, default="", help="GG features dir (base mode)")

    # Twins sources
    p.add_argument("--twins-features-dir", type=str, default="", help="Twins image features dir (twins mode)")

    p.add_argument("--max-images-per-tracklet", type=int, default=0, help="If >0, uniformly subsample per tracklet")
    p.add_argument("--strict", action="store_true", help="Fail if any image feature is missing")
    return p.parse_args()


def _load_base_image_features(gg_dir: str) -> Dict[str, np.ndarray]:
    gallery_feat = np.load(os.path.join(gg_dir, "gallery_features.npy"))
    query_feat = np.load(os.path.join(gg_dir, "query_features.npy"))
    if gallery_feat.ndim != 2 or query_feat.ndim != 2:
        raise ValueError("Expected 2D feature arrays for gallery/query")
    if gallery_feat.shape[1] != query_feat.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: gallery={gallery_feat.shape} query={query_feat.shape} (must share dim)"
        )

    with open(os.path.join(gg_dir, "gallery_info.json"), "r", encoding="utf-8") as f:
        gallery_info = json.load(f)
    with open(os.path.join(gg_dir, "query_info.json"), "r", encoding="utf-8") as f:
        query_info = json.load(f)

    if len(gallery_info) != int(gallery_feat.shape[0]):
        raise ValueError("gallery_info.json length != gallery_features rows")
    if len(query_info) != int(query_feat.shape[0]):
        raise ValueError("query_info.json length != query_features rows")

    mapping: Dict[str, np.ndarray] = {}
    for i, rec in enumerate(gallery_info):
        name = rec.get("image_name")
        if isinstance(name, str):
            mapping[name] = gallery_feat[i]
    for i, rec in enumerate(query_info):
        name = rec.get("image_name")
        if isinstance(name, str):
            mapping[name] = query_feat[i]

    return mapping


def _load_twins_image_features(twins_dir: str) -> Tuple[Dict[str, int], np.ndarray]:
    feats = np.load(os.path.join(twins_dir, "twins_image_features.npy"))
    with open(os.path.join(twins_dir, "twins_image_index.json"), "r", encoding="utf-8") as f:
        idx = json.load(f)
    if not isinstance(idx, dict):
        raise ValueError("twins_image_index.json must be a dict {image_name: idx}")
    return {str(k): int(v) for k, v in idx.items()}, feats


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tracklets, meta_in = load_tracklets(args.tracklets_json)
    tag = args.tag

    if tag == "base":
        if not args.gg_features_dir:
            raise ValueError("--gg-features-dir is required for --tag base")
        # Map image_name -> feature vector.
        img2feat = _load_base_image_features(args.gg_features_dir)
        if not img2feat:
            raise RuntimeError(f"No base features loaded from: {args.gg_features_dir}")
        feat_dim = int(next(iter(img2feat.values())).shape[0])
        twins_idx = None
        twins_feats = None
    else:
        if not args.twins_features_dir:
            raise ValueError("--twins-features-dir is required for --tag twins")
        twins_idx, twins_feats = _load_twins_image_features(args.twins_features_dir)
        if twins_feats.ndim != 2:
            raise ValueError(f"Expected 2D twins features, got shape={twins_feats.shape}")
        feat_dim = int(twins_feats.shape[1])
        img2feat = {}

    features = np.zeros((len(tracklets), feat_dim), dtype=np.float32)
    index_out: List[dict] = []

    missing_images = 0
    missing_tracklets = 0

    for i, t in enumerate(tracklets):
        tid = t.get("track_id")
        images = list(t.get("image_names") or [])
        images = uniform_subsample(images, int(args.max_images_per_tracklet))

        acc = np.zeros((feat_dim,), dtype=np.float64)
        cnt = 0

        for name in images:
            if tag == "base":
                v = img2feat.get(name)
                if v is None:
                    missing_images += 1
                    if args.strict:
                        raise KeyError(f"Missing base image feature: {name}")
                    continue
                # Normalize per-image feature explicitly for safety.
                v64 = v.astype(np.float64, copy=False)
                n = float(np.linalg.norm(v64) + 1e-12)
                acc += (v64 / n)
                cnt += 1
            else:
                assert twins_idx is not None and twins_feats is not None
                j = twins_idx.get(name)
                if j is None:
                    missing_images += 1
                    if args.strict:
                        raise KeyError(f"Missing twins image feature: {name}")
                    continue
                v64 = twins_feats[int(j)].astype(np.float64, copy=False)
                n = float(np.linalg.norm(v64) + 1e-12)
                acc += (v64 / n)
                cnt += 1

        if cnt == 0:
            missing_tracklets += 1
            vec = np.zeros((feat_dim,), dtype=np.float32)
        else:
            vec = (acc / float(cnt)).astype(np.float32)
            vec = l2_normalize(vec).astype(np.float32)

        features[i] = vec
        index_out.append(
            {
                "index": int(i),
                "track_id": tid,
                "camera_id": t.get("camera_id"),
                "vehicle_id_original": t.get("vehicle_id_original"),
                "vehicle_id_mapped": t.get("vehicle_id_mapped"),
                "twins_group": t.get("twins_group"),
                "image_count_used": int(cnt),
                "image_count_listed": int(len(images)),
            }
        )

    out_feat = os.path.join(args.output_dir, f"tracklet_features_{tag}.npy")
    out_idx = os.path.join(args.output_dir, f"tracklet_index_{tag}.json")
    out_meta = os.path.join(args.output_dir, f"tracklet_features_meta_{tag}.json")

    np.save(out_feat, features)
    with open(out_idx, "w", encoding="utf-8") as f:
        json.dump(index_out, f, indent=2, ensure_ascii=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "tracklets_json": os.path.abspath(args.tracklets_json),
        "tracklet_count": int(len(tracklets)),
        "feature_dim": int(feat_dim),
        "missing_images": int(missing_images),
        "missing_tracklets": int(missing_tracklets),
        "max_images_per_tracklet": int(args.max_images_per_tracklet),
        "gg_features_dir": os.path.abspath(args.gg_features_dir) if args.gg_features_dir else None,
        "twins_features_dir": os.path.abspath(args.twins_features_dir) if args.twins_features_dir else None,
        "tracklets_meta": meta_in,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("TR Tracklet Feature Aggregation")
    print("=" * 70)
    print(f"Tag:            {tag}")
    print(f"Tracklets:      {len(tracklets)}")
    print(f"Missing images: {missing_images}")
    print(f"Missing tracklets (cnt=0): {missing_tracklets}")
    print(f"Saved: {out_feat} {features.shape}")
    print(f"Saved: {out_idx}")
    print(f"Saved: {out_meta}")
    print("=" * 70)


if __name__ == "__main__":
    main()
