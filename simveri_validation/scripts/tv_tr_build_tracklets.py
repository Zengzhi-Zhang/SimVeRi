#!/usr/bin/env python
"""
Build tracklet node files for Technical Validation: Trajectory Reconstruction (TR).

Outputs:
  - tracklets_base.json  (Base test subset from a release root)
  - tracklets_twins.json (Twins extras subset from extras/twins)

This script *does not* extract any visual features. It only prepares the
tracklet graph nodes and labels needed by downstream TR scripts.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import DefaultDict, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import (
    as_bool,
    as_float,
    as_int,
    load_json,
    now_iso,
    read_csv_dicts,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TR tracklet node files (base + twins)")

    # Base
    p.add_argument("--release-root", type=str, default="", help="Release root (contains images/ annotations/ metadata/)")
    p.add_argument(
        "--base-subset",
        type=str,
        default="test",
        choices=("test", "all"),
        help="Which base vehicles to include when building tracklets_base.json (default: test)",
    )

    # Twins
    p.add_argument("--twins-root", type=str, default="", help="Twins extras root (<release>/extras/twins)")

    # Output
    p.add_argument("--out-dir", type=str, required=True, help="Directory to write tracklets_*.json")

    # Matching tolerance
    p.add_argument("--time-eps", type=float, default=1e-3, help="Timestamp epsilon for frame assignment (default: 1e-3)")
    p.add_argument("--max-eps", type=float, default=0.1, help="Max epsilon escalation if no frames found (default: 0.1)")
    p.add_argument("--verbose", action="store_true", help="Print detailed warnings")

    return p.parse_args()


def _build_image_relpath_index(release_root: str) -> Dict[str, str]:
    """
    Map image_name -> relative path under release root, e.g. images/gallery/xxx.jpg.
    """
    out: Dict[str, str] = {}
    for split in ("train", "gallery", "query"):
        d = os.path.join(release_root, "images", split)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.lower().endswith(".jpg"):
                continue
            if name in out:
                # Should not happen in a valid release.
                continue
            out[name] = os.path.join("images", split, name).replace("\\", "/")
    return out


def _load_base_vehicle_ids(release_root: str, subset: str) -> Optional[set]:
    if subset == "all":
        return None
    splits_path = os.path.join(release_root, "metadata", "splits.json")
    splits = load_json(splits_path)
    ids = splits.get("test_vehicle_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"Invalid test_vehicle_ids in {splits_path}")
    return set(str(x) for x in ids)


def _index_frames_from_spatiotemporal(
    annotations: Dict[str, dict],
    *,
    vehicle_id_filter: Optional[set],
    image_relpaths: Optional[Dict[str, str]] = None,
    use_original_id: bool = True,
) -> DefaultDict[Tuple[str, str], List[dict]]:
    """
    Build index: (vehicle_id_original, camera_id) -> sorted list of frame dicts.

    If use_original_id=True: vehicle key = annotations[*]["original_id"]
    else: vehicle key = annotations[*]["vehicle_id"]
    """
    idx: DefaultDict[Tuple[str, str], List[dict]] = defaultdict(list)
    for image_name, ann in annotations.items():
        if not isinstance(ann, dict):
            continue
        vid = ann.get("original_id") if use_original_id else ann.get("vehicle_id")
        cam = ann.get("camera_id")
        if not isinstance(vid, str) or not isinstance(cam, str):
            continue
        if vehicle_id_filter is not None and vid not in vehicle_id_filter:
            continue
        ts = as_float(ann.get("timestamp"))
        heading = as_float((ann.get("motion") or {}).get("heading_deg"))
        occ = as_float((ann.get("quality") or {}).get("occlusion_ratio"))
        rel = None
        if image_relpaths is not None:
            rel = image_relpaths.get(image_name)
        idx[(vid, cam)].append(
            {
                "image_name": image_name,
                "timestamp": ts,
                "heading_deg": heading,
                "occlusion_ratio": occ,
                "image_relpath": rel,
            }
        )
    # Sort each bucket by timestamp for stable start/end heading and image order.
    for k in list(idx.keys()):
        idx[k].sort(key=lambda r: r["timestamp"])
    return idx


def _slice_tracklet_frames(frames: List[dict], t_start: float, t_end: float, eps: float) -> List[dict]:
    lo = float(t_start) - float(eps)
    hi = float(t_end) + float(eps)
    out = [r for r in frames if lo <= float(r["timestamp"]) <= hi]
    return out


def _assign_frames_with_escalation(
    frames: List[dict], t_start: float, t_end: float, eps0: float, eps_max: float
) -> Tuple[List[dict], float]:
    eps = float(eps0)
    while True:
        chosen = _slice_tracklet_frames(frames, t_start, t_end, eps)
        if chosen or eps >= float(eps_max):
            return chosen, eps
        eps = min(float(eps_max), eps * 10.0)


def _build_tracklets_base(args: argparse.Namespace) -> Optional[str]:
    rel = args.release_root
    if not rel:
        return None
    if not os.path.isdir(rel):
        raise FileNotFoundError(f"Release root not found: {rel}")

    traj_path = os.path.join(rel, "metadata", "trajectory_info.csv")
    st_path = os.path.join(rel, "metadata", "spatiotemporal.json")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"trajectory_info.csv not found: {traj_path}")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"spatiotemporal.json not found: {st_path}")

    vehicle_filter = _load_base_vehicle_ids(rel, args.base_subset)
    image_relpaths = _build_image_relpath_index(rel)

    rows = read_csv_dicts(traj_path)
    # Filter rows (base subset).
    rows_f = [r for r in rows if (vehicle_filter is None or str(r.get("vehicle_id", "")) in vehicle_filter)]

    st = load_json(st_path)
    annotations = st.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError(f"Invalid spatiotemporal.json format: {st_path}")

    frame_index = _index_frames_from_spatiotemporal(
        annotations, vehicle_id_filter=vehicle_filter, image_relpaths=image_relpaths, use_original_id=True
    )

    tracklets: List[dict] = []
    missing_frames = 0
    used_eps: List[float] = []

    for r in rows_f:
        vid = str(r.get("vehicle_id", ""))
        mid = str(r.get("mapped_id", ""))
        cam = str(r.get("camera_id", ""))
        tid = str(r.get("track_id", ""))
        t_start = as_float(r.get("start_time"))
        t_end = as_float(r.get("end_time"))

        frames_bucket = frame_index.get((vid, cam), [])
        chosen, eps_used = _assign_frames_with_escalation(frames_bucket, t_start, t_end, args.time_eps, args.max_eps)
        used_eps.append(eps_used)

        if not chosen:
            missing_frames += 1
            if args.verbose:
                print(f"[base] Warning: no frames matched tracklet {tid} ({vid}, {cam}) t=[{t_start},{t_end}]")
            heading_start = float("nan")
            heading_end = float("nan")
            occ_values: List[float] = []
            images: List[str] = []
            relpaths: List[Optional[str]] = []
        else:
            heading_start = float(chosen[0]["heading_deg"])
            heading_end = float(chosen[-1]["heading_deg"])
            occ_values = [float(x["occlusion_ratio"]) for x in chosen]
            images = [str(x["image_name"]) for x in chosen]
            relpaths = [x.get("image_relpath") for x in chosen]

        occ_mean = float(mean(occ_values)) if occ_values else float("nan")
        occ_max = float(max(occ_values)) if occ_values else float("nan")

        tracklets.append(
            {
                "track_id": tid,
                "camera_id": cam,
                "vehicle_id_original": vid,
                "vehicle_id_mapped": mid,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "t_event": float(t_start),
                "x_event": as_float(r.get("start_x")),
                "y_event": as_float(r.get("start_y")),
                "x_exit": as_float(r.get("end_x")),
                "y_exit": as_float(r.get("end_y")),
                "heading_start": heading_start,
                "heading_end": heading_end,
                "occlusion_mean": occ_mean,
                "occlusion_max": occ_max,
                "image_names": images,
                "image_relpaths": relpaths,
                "image_count_expected": as_int(r.get("image_count")),
                "image_count_found": int(len(images)),
                "is_twins": as_bool(r.get("is_twins", False)),
            }
        )

    out_path = os.path.join(args.out_dir, "tracklets_base.json")
    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "meta": {
            "subset": f"base_{args.base_subset}",
            "release_root": os.path.abspath(rel),
            "trajectory_info": os.path.abspath(traj_path),
            "spatiotemporal": os.path.abspath(st_path),
            "generated_at": now_iso(),
            "time_eps": float(args.time_eps),
            "max_eps": float(args.max_eps),
            "tracklet_count": int(len(tracklets)),
            "vehicle_count": int(len(set(t["vehicle_id_original"] for t in tracklets))),
            "camera_count": int(len(set(t["camera_id"] for t in tracklets))),
            "missing_tracklets_no_frames": int(missing_frames),
            "eps_used_p50": float(sorted(used_eps)[len(used_eps) // 2]) if used_eps else float("nan"),
        },
        "tracklets": tracklets,
    }
    save_json(out_path, out)
    print(f"Saved: {out_path} (tracklets={len(tracklets)}, vehicles={out['meta']['vehicle_count']})")
    if missing_frames:
        print(f"[base] Warning: {missing_frames} tracklets have no matched frames (check time_eps/max_eps).")
    return out_path


def _build_twins_vehicle_to_group(twins_groups: dict) -> Dict[str, str]:
    groups = twins_groups.get("groups", {}) if isinstance(twins_groups, dict) else {}
    if not isinstance(groups, dict):
        return {}
    out: Dict[str, str] = {}
    for gid, rec in groups.items():
        if not isinstance(rec, dict):
            continue
        vehicles = rec.get("vehicles", [])
        if isinstance(vehicles, list):
            for v in vehicles:
                if isinstance(v, str):
                    out[v] = str(gid)
    return out


def _build_tracklets_twins(args: argparse.Namespace) -> Optional[str]:
    twins_root = args.twins_root
    if not twins_root:
        return None
    if not os.path.isdir(twins_root):
        raise FileNotFoundError(f"Twins root not found: {twins_root}")

    traj_path = os.path.join(twins_root, "metadata", "trajectory_info_twins.csv")
    st_path = os.path.join(twins_root, "metadata", "spatiotemporal_twins.json")
    groups_path = os.path.join(twins_root, "metadata", "twins_groups.json")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"trajectory_info_twins.csv not found: {traj_path}")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"spatiotemporal_twins.json not found: {st_path}")
    if not os.path.exists(groups_path):
        raise FileNotFoundError(f"twins_groups.json not found: {groups_path}")

    rows = read_csv_dicts(traj_path)
    st = load_json(st_path)
    annotations = st.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError(f"Invalid spatiotemporal_twins.json format: {st_path}")

    twins_groups = load_json(groups_path)
    v2g = _build_twins_vehicle_to_group(twins_groups)

    frame_index = _index_frames_from_spatiotemporal(annotations, vehicle_id_filter=None, image_relpaths=None, use_original_id=True)

    tracklets: List[dict] = []
    missing_frames = 0
    for r in rows:
        vid_orig = str(r.get("vehicle_id", ""))  # Hxx_xx
        mid = str(r.get("mapped_id", ""))        # 4-digit
        cam = str(r.get("camera_id", ""))
        tid = str(r.get("track_id", ""))
        t_start = as_float(r.get("start_time"))
        t_end = as_float(r.get("end_time"))

        frames_bucket = frame_index.get((vid_orig, cam), [])
        chosen, _ = _assign_frames_with_escalation(frames_bucket, t_start, t_end, args.time_eps, args.max_eps)

        if not chosen:
            missing_frames += 1
            if args.verbose:
                print(f"[twins] Warning: no frames matched tracklet {tid} ({vid_orig}, {cam}) t=[{t_start},{t_end}]")
            heading_start = float("nan")
            heading_end = float("nan")
            occ_values: List[float] = []
            images: List[str] = []
        else:
            heading_start = float(chosen[0]["heading_deg"])
            heading_end = float(chosen[-1]["heading_deg"])
            occ_values = [float(x["occlusion_ratio"]) for x in chosen]
            images = [str(x["image_name"]) for x in chosen]

        occ_mean = float(mean(occ_values)) if occ_values else float("nan")
        occ_max = float(max(occ_values)) if occ_values else float("nan")

        tracklets.append(
            {
                "track_id": tid,
                "camera_id": cam,
                # For Twins evaluation, vehicle-level label is the mapped 4-digit id.
                "vehicle_id": mid,
                "vehicle_id_original": vid_orig,
                "vehicle_id_mapped": mid,
                "twins_group": v2g.get(vid_orig, None),
                "t_start": float(t_start),
                "t_end": float(t_end),
                "t_event": float(t_start),
                "x_event": as_float(r.get("start_x")),
                "y_event": as_float(r.get("start_y")),
                "x_exit": as_float(r.get("end_x")),
                "y_exit": as_float(r.get("end_y")),
                "heading_start": heading_start,
                "heading_end": heading_end,
                "occlusion_mean": occ_mean,
                "occlusion_max": occ_max,
                "image_names": images,
                "image_count_expected": as_int(r.get("image_count")),
                "image_count_found": int(len(images)),
                "is_twins": True,
            }
        )

    out_path = os.path.join(args.out_dir, "tracklets_twins.json")
    out = {
        "meta": {
            "subset": "twins_extras",
            "twins_root": os.path.abspath(twins_root),
            "trajectory_info": os.path.abspath(traj_path),
            "spatiotemporal": os.path.abspath(st_path),
            "twins_groups": os.path.abspath(groups_path),
            "generated_at": now_iso(),
            "time_eps": float(args.time_eps),
            "max_eps": float(args.max_eps),
            "tracklet_count": int(len(tracklets)),
            "vehicle_count": int(len(set(t["vehicle_id_mapped"] for t in tracklets))),
            "camera_count": int(len(set(t["camera_id"] for t in tracklets))),
            "group_count": int(len(set(t["twins_group"] for t in tracklets if t.get("twins_group")))),
            "missing_tracklets_no_frames": int(missing_frames),
        },
        "tracklets": tracklets,
    }
    save_json(out_path, out)
    print(f"Saved: {out_path} (tracklets={len(tracklets)}, vehicles={out['meta']['vehicle_count']})")
    if missing_frames:
        print(f"[twins] Warning: {missing_frames} tracklets have no matched frames (check time_eps/max_eps).")
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_out = _build_tracklets_base(args)
    twins_out = _build_tracklets_twins(args)

    if not base_out and not twins_out:
        raise ValueError("Nothing to do: provide --release-root and/or --twins-root.")


if __name__ == "__main__":
    main()

