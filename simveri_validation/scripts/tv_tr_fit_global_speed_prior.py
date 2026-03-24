#!/usr/bin/env python
"""
Fit a global speed prior for TR (train split only), co-view aware.

We DO NOT use camera_network.distance_matrix here. Speed estimation uses
Euclidean displacement between adjacent tracklets (exit->entry):

  dt = start_time(next) - end_time(curr)   (must be > 0)
  d  = hypot(start_x(next)-end_x(curr), start_y(next)-end_y(curr))
  v_est = 3.6 * d / dt   (km/h)

Pairs marked as co-view are excluded from the prior fitting set.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev
from typing import DefaultDict, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_float, load_coview_pairs, load_json, now_iso, read_csv_dicts, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit global speed prior (train split only, coview excluded)")
    p.add_argument("--release-root", type=str, required=True, help="SimVeRi release root")
    p.add_argument("--coview-json", type=str, required=True, help="coview_pairs.json from tv_tr_fit_coview_pairs.py")
    p.add_argument("--out", type=str, required=True, help="Output global_speed_prior.json path")
    p.add_argument("--v-max-kmh", type=float, default=120.0, help="Conservative gate for prior samples (default: 120)")
    p.add_argument("--v-std-floor", type=float, default=2.0, help="Minimum std to avoid collapse (default: 2.0)")
    p.add_argument("--trim-pct", type=float, default=0.0, help="Optional symmetric trim percent (e.g., 1.0 trims 1%% tails)")
    p.add_argument("--save-samples-csv", type=str, default="", help="Optional: save per-sample v_est to CSV for debugging")
    return p.parse_args()


def _load_train_vehicle_ids(release_root: str) -> List[str]:
    splits_path = os.path.join(release_root, "metadata", "splits.json")
    splits = load_json(splits_path)
    ids = splits.get("train_vehicle_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"Invalid train_vehicle_ids in {splits_path}")
    return [str(x) for x in ids]


def _is_coview(coview_map: Dict[Tuple[str, str], dict], cam_a: str, cam_b: str) -> bool:
    rec = coview_map.get((cam_a, cam_b))
    if not isinstance(rec, dict):
        return False
    return bool(rec.get("coview", False))


def _trim(values: List[float], trim_pct: float) -> List[float]:
    if trim_pct <= 0.0 or not values:
        return values
    xs = sorted(values)
    k = int(math.floor(len(xs) * (trim_pct / 100.0)))
    if 2 * k >= len(xs):
        return xs
    return xs[k : len(xs) - k]


def main() -> None:
    args = parse_args()
    rel = args.release_root
    traj_path = os.path.join(rel, "metadata", "trajectory_info.csv")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"trajectory_info.csv not found: {traj_path}")

    train_ids = set(_load_train_vehicle_ids(rel))
    coview = load_coview_pairs(args.coview_json)

    rows = read_csv_dicts(traj_path)
    per_vehicle: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in rows:
        vid = str(r.get("vehicle_id", ""))
        if vid in train_ids:
            per_vehicle[vid].append(r)

    samples: List[float] = []
    sample_rows: List[Dict[str, object]] = []

    v_max = float(args.v_max_kmh)

    for vid, tracks in per_vehicle.items():
        tracks_sorted = sorted(tracks, key=lambda x: as_float(x.get("start_time", 0.0)))
        for cur, nxt in zip(tracks_sorted, tracks_sorted[1:]):
            cam_a = str(cur.get("camera_id", "")).strip()
            cam_b = str(nxt.get("camera_id", "")).strip()
            if not cam_a or not cam_b or cam_a == cam_b:
                continue

            if _is_coview(coview, cam_a, cam_b):
                continue

            t_end = as_float(cur.get("end_time"))
            t_next = as_float(nxt.get("start_time"))
            dt = float(t_next - t_end)
            if dt <= 0:
                continue

            x_exit = as_float(cur.get("end_x"))
            y_exit = as_float(cur.get("end_y"))
            x_ent = as_float(nxt.get("start_x"))
            y_ent = as_float(nxt.get("start_y"))

            d = math.hypot(x_ent - x_exit, y_ent - y_exit)
            v_est = 3.6 * d / dt
            if not math.isfinite(v_est):
                continue
            if v_est > v_max:
                continue

            samples.append(float(v_est))

            if args.save_samples_csv:
                sample_rows.append(
                    {
                        "vehicle_id": vid,
                        "cam_from": cam_a,
                        "cam_to": cam_b,
                        "dt_s": dt,
                        "d_m": d,
                        "v_kmh": v_est,
                        "cur_track_id": cur.get("track_id", ""),
                        "next_track_id": nxt.get("track_id", ""),
                    }
                )

    if not samples:
        raise RuntimeError("No valid samples for speed prior. Check coview threshold and v_max.")

    used = _trim(samples, float(args.trim_pct))
    v_mean = float(mean(used))
    v_std = float(pstdev(used))  # population std; stable for reporting
    v_std = float(max(v_std, float(args.v_std_floor)))

    used_sorted = sorted(used)
    def pct(p: float) -> float:
        if not used_sorted:
            return float("nan")
        idx = int(round((p / 100.0) * (len(used_sorted) - 1)))
        idx = max(0, min(len(used_sorted) - 1, idx))
        return float(used_sorted[idx])

    out = {
        "meta": {
            "release_root": os.path.abspath(rel),
            "trajectory_info": os.path.abspath(traj_path),
            "coview_json": os.path.abspath(args.coview_json),
            "generated_at": now_iso(),
            "v_max_kmh_gate": float(v_max),
            "trim_pct": float(args.trim_pct),
            "train_vehicle_count": int(len(train_ids)),
            "train_vehicle_with_tracklets": int(len(per_vehicle)),
        },
        "v_mean_kmh": v_mean,
        "v_std_kmh": v_std,
        "v_max_kmh": float(v_max),
        "sample_count": int(len(used)),
        "sample_count_raw": int(len(samples)),
        "p50_kmh": pct(50),
        "p90_kmh": pct(90),
        "p95_kmh": pct(95),
        "p99_kmh": pct(99),
    }

    save_json(args.out, out)
    print(f"Saved: {args.out}")
    print(f"Speed prior: mean={v_mean:.2f} km/h, std={v_std:.2f} km/h, n={len(used)} (raw={len(samples)})")

    if args.save_samples_csv:
        import csv

        os.makedirs(os.path.dirname(args.save_samples_csv) or ".", exist_ok=True)
        with open(args.save_samples_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            w.writeheader()
            w.writerows(sample_rows)
        print(f"Saved samples: {args.save_samples_csv}")


if __name__ == "__main__":
    main()
