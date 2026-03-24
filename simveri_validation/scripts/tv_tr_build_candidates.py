#!/usr/bin/env python
"""
Build TR-Closed candidate sets for trajectory reconstruction.

Given a tracklets json file (base or twins), this script produces a candidates json:
  - Ground-truth chain per vehicle (sorted by t_event = start_time)
  - At each step t>=2, a candidate set C_t of tracklets in the same camera as gt(t)
    within a time window around gt(t).

Protocols:
  - base        : uses time window W_base (default 30s)
  - twins_full  : uses time window W_twins (default 8s)
  - twins_group : restrict candidates to the same twins_group at each camera (no time window)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_float, load_tracklets, now_iso, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build candidate sets for TR-Closed reconstruction")
    p.add_argument("--tracklets-json", type=str, required=True, help="tracklets_base.json or tracklets_twins.json")
    p.add_argument("--protocol", type=str, required=True, choices=("base", "twins_full", "twins_group"))
    p.add_argument("--out", type=str, required=True, help="Output candidates_*.json path")
    p.add_argument("--W", type=float, default=0.0, help="Time window (seconds). If 0, uses protocol default.")
    p.add_argument("--min-chain-len", type=int, default=3, help="Only keep vehicles with chain length >= this")
    return p.parse_args()


def _default_W(protocol: str) -> float:
    if protocol == "base":
        return 30.0
    if protocol == "twins_full":
        return 8.0
    return 0.0  # twins_group: no window


def main() -> None:
    args = parse_args()
    tracklets, meta_in = load_tracklets(args.tracklets_json)

    protocol = args.protocol
    W = float(args.W) if float(args.W) > 0 else _default_W(protocol)
    min_L = int(args.min_chain_len)

    # Determine vehicle key.
    if protocol == "base":
        vehicle_key = "vehicle_id_original"
    else:
        vehicle_key = "vehicle_id_mapped"

    # Index all tracklets by camera for fast candidate query.
    by_cam: DefaultDict[str, List[dict]] = defaultdict(list)
    for t in tracklets:
        cam = str(t.get("camera_id", "")).strip()
        if cam:
            by_cam[cam].append(t)
    for cam in list(by_cam.keys()):
        by_cam[cam].sort(key=lambda r: as_float(r.get("t_event", r.get("t_start", 0.0))))

    # Group tracklets by vehicle.
    by_vehicle: DefaultDict[str, List[dict]] = defaultdict(list)
    for t in tracklets:
        vid = t.get(vehicle_key)
        if isinstance(vid, str) and vid:
            by_vehicle[vid].append(t)

    vehicles_out: List[dict] = []
    total_steps = 0
    total_candidates = 0
    kept_vehicles = 0

    for vid, ts in sorted(by_vehicle.items()):
        chain = sorted(ts, key=lambda r: as_float(r.get("t_event", r.get("t_start", 0.0))))
        if len(chain) < min_L:
            continue

        kept_vehicles += 1
        chain_ids = [str(x.get("track_id", "")) for x in chain]
        chain_cams = [str(x.get("camera_id", "")) for x in chain]
        chain_te = [as_float(x.get("t_event", x.get("t_start", 0.0))) for x in chain]

        group_id: Optional[str] = None
        if protocol in ("twins_full", "twins_group"):
            # Infer group id from any tracklet record (should be consistent).
            group_id = chain[0].get("twins_group")
            if group_id is not None:
                group_id = str(group_id)

        steps: List[dict] = []
        for t_idx in range(1, len(chain)):  # steps 2..L (1-indexed)
            gt = chain[t_idx]
            cam_gt = str(gt.get("camera_id", ""))
            te_gt = as_float(gt.get("t_event", gt.get("t_start", 0.0)))

            if protocol == "twins_group":
                # Exactly the same group at this camera. No time window.
                cands = [
                    str(x.get("track_id", ""))
                    for x in by_cam.get(cam_gt, [])
                    if str(x.get("twins_group", "")) == str(group_id or "")
                ]
            else:
                # Windowed candidates.
                cands = [
                    str(x.get("track_id", ""))
                    for x in by_cam.get(cam_gt, [])
                    if abs(as_float(x.get("t_event", x.get("t_start", 0.0))) - te_gt) <= W
                ]

            gt_id = str(gt.get("track_id", ""))
            if gt_id and gt_id not in cands:
                cands.append(gt_id)

            # Stable order: sort by t_event then track_id.
            id2t = {str(x.get("track_id", "")): x for x in by_cam.get(cam_gt, [])}
            cands_sorted = sorted(
                set([c for c in cands if c]),
                key=lambda cid: (
                    as_float(id2t.get(cid, {}).get("t_event", id2t.get(cid, {}).get("t_start", 0.0))),
                    cid,
                ),
            )

            steps.append(
                {
                    "step": int(t_idx + 1),
                    "gt_track_id": gt_id,
                    "camera_id": cam_gt,
                    "gt_t_event": float(te_gt),
                    "candidates": cands_sorted,
                    "candidate_count": int(len(cands_sorted)),
                }
            )
            total_steps += 1
            total_candidates += int(len(cands_sorted))

        vehicles_out.append(
            {
                "vehicle_id": vid,
                "twins_group": group_id,
                "chain_track_ids": chain_ids,
                "chain_camera_ids": chain_cams,
                "chain_t_event": [float(x) for x in chain_te],
                "steps": steps,
                "chain_len": int(len(chain)),
            }
        )

    out = {
        "meta": {
            "tracklets_json": os.path.abspath(args.tracklets_json),
            "tracklets_meta": meta_in,
            "generated_at": now_iso(),
            "protocol": protocol,
            "W_s": float(W),
            "min_chain_len": int(min_L),
            "vehicle_key": vehicle_key,
            "vehicle_count": int(kept_vehicles),
            "total_steps": int(total_steps),
            "avg_candidates_per_step": float(total_candidates / max(total_steps, 1)),
        },
        "vehicles": vehicles_out,
    }

    save_json(args.out, out)
    print(f"Saved: {args.out}")
    print(
        f"Protocol={protocol} vehicles={kept_vehicles} steps={total_steps} "
        f"avg|C|={out['meta']['avg_candidates_per_step']:.2f} W={W}s"
    )


if __name__ == "__main__":
    main()

