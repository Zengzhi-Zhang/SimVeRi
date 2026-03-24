#!/usr/bin/env python
"""
Numeric sanity check for heading_deg vs displacement direction.

This script is meant for *Technical Validation* (Scientific Data):
it verifies that the spatiotemporal heading field is physically consistent.

Computation (camera-level, to avoid cross-camera jumps):
  For each (vehicle, camera), sort frames by timestamp.
  For consecutive frames with displacement >= min_disp_m:
    actual_dir = atan2(dy, dx) in degrees
    diff = angdiff(actual_dir, annotated_heading_deg_at_frame_t)

Output is a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from statistics import mean, median
from typing import DefaultDict, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import angdiff_deg, as_float, now_iso, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check: heading_deg vs displacement direction")
    p.add_argument("--spatiotemporal-json", type=str, required=True, help="Path to spatiotemporal.json (or *_twins.json)")
    p.add_argument("--out", type=str, required=True, help="Output JSON path")
    p.add_argument("--vehicle-field", type=str, default="original_id", choices=("original_id", "vehicle_id"))
    p.add_argument("--min-disp-m", type=float, default=1.0, help="Skip segments with displacement < this (default: 1.0m)")
    p.add_argument("--max-vehicles", type=int, default=0, help="If >0, randomly sample this many vehicles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk-outliers", type=int, default=20, help="Save top-K largest diffs (default: 20)")
    return p.parse_args()


def _atan2_deg(dy: float, dx: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def main() -> None:
    args = parse_args()
    st_path = args.spatiotemporal_json
    with open(st_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ann = obj.get("annotations", {})
    if not isinstance(ann, dict):
        raise ValueError("spatiotemporal json must contain dict field: annotations")

    # Group frames by (vehicle, camera).
    buckets: DefaultDict[Tuple[str, str], List[dict]] = defaultdict(list)
    for image_name, rec in ann.items():
        if not isinstance(rec, dict):
            continue
        vid = rec.get(args.vehicle_field)
        cam = rec.get("camera_id")
        if not isinstance(vid, str) or not isinstance(cam, str):
            continue
        ts = as_float(rec.get("timestamp"))
        pos = rec.get("position") or {}
        mot = rec.get("motion") or {}
        x = as_float(pos.get("x"))
        y = as_float(pos.get("y"))
        h = as_float(mot.get("heading_deg"))
        buckets[(vid, cam)].append({"ts": ts, "x": x, "y": y, "h": h, "image": image_name})

    # Vehicle sampling (optional) - keep all cameras for chosen vehicles.
    vehicles = sorted({k[0] for k in buckets.keys()})
    if args.max_vehicles and int(args.max_vehicles) > 0 and len(vehicles) > int(args.max_vehicles):
        random.seed(int(args.seed))
        vehicles = sorted(random.sample(vehicles, int(args.max_vehicles)))
        buckets = defaultdict(list, {k: v for k, v in buckets.items() if k[0] in set(vehicles)})

    diffs: List[float] = []
    outliers: List[dict] = []
    used_segments = 0
    skipped_small = 0

    min_disp = float(args.min_disp_m)

    for (vid, cam), frames in buckets.items():
        frames = sorted(frames, key=lambda r: r["ts"])
        for a, b in zip(frames, frames[1:]):
            dx = float(b["x"] - a["x"])
            dy = float(b["y"] - a["y"])
            d = math.hypot(dx, dy)
            if d < min_disp:
                skipped_small += 1
                continue
            actual = _atan2_deg(dy, dx)
            annotated = float(a["h"])
            diff = angdiff_deg(actual, annotated)
            if not math.isfinite(diff):
                continue
            diffs.append(float(diff))
            used_segments += 1
            outliers.append(
                {
                    "diff_deg": float(diff),
                    "vehicle": vid,
                    "camera": cam,
                    "image_a": a["image"],
                    "image_b": b["image"],
                    "actual_deg": float(actual),
                    "annotated_deg": float(annotated),
                    "disp_m": float(d),
                }
            )

    if not diffs:
        raise RuntimeError("No valid segments to evaluate (try lowering --min-disp-m).")

    diffs_sorted = sorted(diffs)
    outliers_sorted = sorted(outliers, key=lambda r: r["diff_deg"], reverse=True)[: int(args.topk_outliers)]

    def pct(p: float) -> float:
        idx = int(round((p / 100.0) * (len(diffs_sorted) - 1)))
        idx = max(0, min(len(diffs_sorted) - 1, idx))
        return float(diffs_sorted[idx])

    report = {
        "meta": {
            "generated_at": now_iso(),
            "spatiotemporal_json": os.path.abspath(st_path),
            "vehicle_field": args.vehicle_field,
            "min_disp_m": float(min_disp),
            "max_vehicles": int(args.max_vehicles),
            "seed": int(args.seed),
            "bucketed_vehicle_count": int(len(vehicles)),
            "bucket_count_vehicle_camera": int(len(buckets)),
            "used_segments": int(used_segments),
            "skipped_small_disp": int(skipped_small),
        },
        "summary": {
            "mean_diff_deg": float(mean(diffs)),
            "median_diff_deg": float(median(diffs)),
            "p90_diff_deg": pct(90),
            "p95_diff_deg": pct(95),
            "p99_diff_deg": pct(99),
        },
        "outliers_topk": outliers_sorted,
    }

    save_json(args.out, report)
    print(f"Saved: {args.out}")
    print(
        f"Heading diff (deg): mean={report['summary']['mean_diff_deg']:.2f}, "
        f"median={report['summary']['median_diff_deg']:.2f}, "
        f"p95={report['summary']['p95_diff_deg']:.2f} (segments={used_segments})"
    )


if __name__ == "__main__":
    main()
