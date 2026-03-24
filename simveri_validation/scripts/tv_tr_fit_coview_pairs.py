#!/usr/bin/env python
"""
Fit co-view camera pairs from Base *train* vehicles (data-driven detection).

Co-view definition (directed):
  For each train vehicle, sort its tracklets by t_event = start_time.
  For each adjacent pair (cam_i -> cam_{i+1}), compute entry-to-entry dt:
      dt_exit_entry = start_time(next) - end_time(curr)
  Aggregate dt_entry per directed camera pair and mark as co-view if:
      median(dt_exit_entry) < threshold_s

Notes:
  - Using exit-to-entry dt directly captures *overlap* (dt<=0), which is a strong
    signature of co-view / shared road segment observation.
  - We still sort by start_time (stable ordering). The dt definition here is
    only for detecting co-view pairs, not for ordering the chain.

This file is intentionally independent (no torch) and should be runnable
in the CPU evaluation environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from statistics import median
from typing import DefaultDict, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_float, load_json, now_iso, read_csv_dicts, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit co-view camera pairs (train split only)")
    p.add_argument("--release-root", type=str, required=True, help="SimVeRi release root (contains metadata/)")
    p.add_argument("--out", type=str, required=True, help="Output coview_pairs.json path")
    p.add_argument("--threshold-s", type=float, default=2.0, help="Median dt threshold in seconds (default: 2.0)")
    p.add_argument("--min-samples", type=int, default=5, help="Min samples per pair to consider coview (default: 5)")
    return p.parse_args()


def _load_train_vehicle_ids(release_root: str) -> List[str]:
    splits_path = os.path.join(release_root, "metadata", "splits.json")
    splits = load_json(splits_path)
    ids = splits.get("train_vehicle_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"Invalid train_vehicle_ids in {splits_path}")
    return [str(x) for x in ids]


def main() -> None:
    args = parse_args()
    rel = args.release_root
    traj_path = os.path.join(rel, "metadata", "trajectory_info.csv")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"trajectory_info.csv not found: {traj_path}")

    train_ids = set(_load_train_vehicle_ids(rel))
    rows = read_csv_dicts(traj_path)

    # Collect per-vehicle tracklets for train split.
    per_vehicle: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in rows:
        vid = str(r.get("vehicle_id", ""))
        if vid in train_ids:
            per_vehicle[vid].append(r)

    # Pair -> list of dt (exit->entry) and dt_entry (start->start) for diagnostics
    dts_exit: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    dts_entry: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)

    for vid, tracks in per_vehicle.items():
        tracks_sorted = sorted(tracks, key=lambda x: as_float(x.get("start_time", 0.0)))
        for a, b in zip(tracks_sorted, tracks_sorted[1:]):
            cam_a = str(a.get("camera_id", "")).strip()
            cam_b = str(b.get("camera_id", "")).strip()
            if not cam_a or not cam_b or cam_a == cam_b:
                continue
            dt_exit_entry = as_float(b.get("start_time")) - as_float(a.get("end_time"))
            dt_entry = as_float(b.get("start_time")) - as_float(a.get("start_time"))
            dts_exit[(cam_a, cam_b)].append(float(dt_exit_entry))
            dts_entry[(cam_a, cam_b)].append(float(dt_entry))

    pairs_out: Dict[str, dict] = {}
    coview_pairs = 0
    for (cam_a, cam_b), values in sorted(dts_exit.items()):
        n = len(values)
        med_exit = float(median(values)) if values else float("nan")
        med_entry = float(median(dts_entry.get((cam_a, cam_b), []))) if dts_entry.get((cam_a, cam_b)) else float("nan")
        overlap_rate = float(sum(1 for x in values if x <= 0.0) / max(n, 1))
        is_coview = bool(n >= int(args.min_samples) and med_exit < float(args.threshold_s))
        if is_coview:
            coview_pairs += 1
        key = f"{cam_a}->{cam_b}"
        pairs_out[key] = {
            "from": cam_a,
            "to": cam_b,
            "n": int(n),
            # For compatibility with the TR workflow doc: median_dt_s refers to exit->entry dt.
            "median_dt_s": med_exit,
            # Additional diagnostics
            "median_dt_entry_s": med_entry,
            "overlap_rate": overlap_rate,
            "coview": is_coview,
        }

    out = {
        "meta": {
            "release_root": os.path.abspath(rel),
            "trajectory_info": os.path.abspath(traj_path),
            "generated_at": now_iso(),
            "threshold_s": float(args.threshold_s),
            "min_samples": int(args.min_samples),
            "train_vehicle_count": int(len(train_ids)),
            "train_vehicle_with_tracklets": int(len(per_vehicle)),
            "directed_pair_count": int(len(pairs_out)),
            "coview_pair_count": int(coview_pairs),
        },
        "pairs": pairs_out,
    }

    save_json(args.out, out)
    print(f"Saved: {args.out}")
    print(
        f"Pairs: {out['meta']['directed_pair_count']} directed, "
        f"coview={out['meta']['coview_pair_count']} (threshold={args.threshold_s}s, min_n={args.min_samples})"
    )


if __name__ == "__main__":
    main()
