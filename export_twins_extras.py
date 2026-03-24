"""
Export the Twins (H*) subset as a standalone "extras" package inside an existing release.

Why:
  - The main benchmark release typically excludes Twins (H*) to avoid "identical appearance" bias.
  - For public dataset releases / papers, Twins are valuable and should be shipped as a separate, clearly
    labeled supplement to enable dedicated evaluation/protocols.

Inputs:
  --clean-dir:  output_cleaned_<RUN_ID> (must contain metadata/captures_cleaned.json and image_train/)
  --release-dir: release_<RUN_ID> (the main release; we will write under <release>/extras/twins/)

Outputs (under <release>/extras/twins/):
  images/*.jpg
  metadata/
    captures_twins.json
    spatiotemporal_twins.json
    trajectory_info_twins.csv
    twins_groups.json
    vehicle_id_mapping_twins.json
  statistics/
    twins_summary.json

Notes:
  - Images are copied and renamed into the standard SimVeRi filename format:
      <mapped_id>_c<cam>_<frame:06d>.jpg
    where mapped_id is in [0431..] for Twins vehicles (consistent with SimVeRi conventions).
  - This script does NOT change the main release train/gallery/query splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_README_TEMPLATE = """# SimVeRi Twins Extras (H* / Identical-Appearance Vehicles)

This folder contains the Twins subset (vehicles with intentionally identical appearance), exported as a supplement to the main SimVeRi benchmark release.

Why separategroups
- The main benchmark split (train/gallery/query) often excludes Twins (H*) to keep the standard visual ReID evaluation comparable and stable.
- Twins are still important for public releases and research (hard negatives, identical appearance, trajectory-aware reID), so they are published here as an extras package.

## Folder Structure

```
extras/twins/
  images/
    0431_c001_000123.jpg
    ...
  metadata/
    vehicle_id_mapping_twins.json
    captures_twins.json
    spatiotemporal_twins.json
    trajectory_info_twins.csv
    twins_groups.json
  statistics/
    twins_summary.json
```

Key files:
- images/*.jpg: renamed to the standard SimVeRi filename format: <mapped_id>_c<cam>_<frame:06d>.jpg
- metadata/vehicle_id_mapping_twins.json: original Twins IDs (H*) -> mapped IDs (typically 0431-0530)
- metadata/captures_twins.json: per-image capture records; uses relative image_path for portability
- metadata/spatiotemporal_twins.json: frame-level spatiotemporal annotations keyed by renamed filename
- metadata/trajectory_info_twins.csv: track summaries per (vehicle_id, camera_id)
- metadata/twins_groups.json: Twins group definitions (typically derived from fleet_id)
- statistics/twins_summary.json: export summary (counts, missing_images, camera distribution, etc.)

## Recommended Twins Challenge Protocols (Suggestions)

Twins are hard by design. Visual-only models will often confuse vehicles within the same Twins group; this is expected.

T1) Twins-only Vehicle ReID (Closed-Set)
- Query: one image per (vehicle_id, camera_id) (pick the middle frame by frame_id for determinism)
- Gallery: all remaining Twins images (exclude the exact query image)
- Ignore during ranking: same vehicle_id + same camera_id

T2) Twins Queries + Distractor Gallery (Open-World)
- Query: Twins queries from T1
- Gallery: (Twins gallery) + (base/occlusion gallery from the main benchmark release)

T3) Twins Group Retrieval (Fleet-Level)
- Label is group_id from metadata/twins_groups.json, rather than per-vehicle ID

## Notes for This Repository

The baseline scripts in simveri_validation expect a full SimVeRi-style dataset root with images/{gallery,query}, annotations, and metadata. This extras folder alone is a supplement.

If you want to evaluate Twins with the baseline pipeline, build a small Twins-only dataset root (with images/gallery+query, annotations/query_list.txt and gallery_annotations.xml, metadata/spatiotemporal.json and twins_groups.json), then run extract_features.py + evaluate_baseline.py.
"""


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_readme(out_root: Path, overwrite: bool) -> None:
    readme_path = out_root / "README.md"
    if readme_path.exists() and not overwrite:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(_README_TEMPLATE, encoding="utf-8")


def _is_twins_cap(cap: dict) -> bool:
    vid = cap.get("vehicle_id")
    if isinstance(vid, str) and vid.startswith("H"):
        return True
    return bool(cap.get("is_fleet", False))


def _sorted_caps(caps: Iterable[dict]) -> List[dict]:
    def key(c: dict) -> Tuple[str, str, int, float]:
        return (
            str(c.get("vehicle_id", "")),
            str(c.get("camera_id", "")),
            int(c.get("frame_id", 0) or 0),
            float(c.get("timestamp", 0.0) or 0.0),
        )

    return sorted(list(caps), key=key)


def _create_twins_id_mapping(twins_vehicle_ids: List[str]) -> Dict[str, str]:
    """Map H* vehicle IDs into 4-digit IDs starting from 0431 (SimVeRi convention)."""
    mapping: Dict[str, str] = {}
    counter = 431
    for vid in sorted(set(twins_vehicle_ids)):
        mapping[vid] = f"{counter:04d}"
        counter += 1
    return mapping


def _generate_filename(mapped_id: str, camera_id: str, frame_id: int) -> str:
    cam_num = str(camera_id).replace("c", "").replace("C", "")
    return f"{mapped_id}_c{cam_num}_{int(frame_id):06d}.jpg"


def _resolve_src_image_path(clean_dir: Path, cap: dict) -> Path:
    """
    Resolve the source image path robustly.

    Cleaned captures usually store absolute paths in cap['image_path'].
    """
    p = cap.get("image_path") or cap.get("image_name") or ""
    if not isinstance(p, str):
        p = str(p)
    src = Path(p)
    if src.exists():
        return src

    # Fallback: try to locate under <clean_dir>/image_train/<basename>
    base = src.name or os.path.basename(p)
    if base:
        cand = clean_dir / "image_train" / base
        if cand.exists():
            return cand

    return src  # may not exist; caller will count missing


def _build_spatiotemporal_annotations(twins_caps: List[dict], vid_map: Dict[str, str]) -> dict:
    annotations: Dict[str, dict] = {}
    for cap in twins_caps:
        orig_vid = cap.get("vehicle_id")
        if orig_vid not in vid_map:
            continue
        mapped = vid_map[orig_vid]
        cam = str(cap.get("camera_id", ""))
        frame = int(cap.get("frame_id", 0) or 0)
        fname = _generate_filename(mapped, cam, frame)
        annotations[fname] = {
            "vehicle_id": mapped,
            "original_id": orig_vid,
            "camera_id": cam,
            "frame_id": frame,
            "timestamp": round(float(cap.get("timestamp", 0.0) or 0.0), 2),
            "position": {
                "x": round(float(cap.get("global_x", 0.0) or 0.0), 2),
                "y": round(float(cap.get("global_y", 0.0) or 0.0), 2),
                "z": round(float(cap.get("global_z", 0.0) or 0.0), 2),
            },
            "motion": {
                "speed_kmh": round(float(cap.get("speed", 0.0) or 0.0), 2),
                "heading_deg": round(float(cap.get("heading", 0.0) or 0.0), 2),
            },
            "quality": {
                "occlusion_ratio": round(float(cap.get("occlusion_ratio", 0.0) or 0.0), 3),
                "distance_m": round(float(cap.get("distance", 0.0) or 0.0), 2),
            },
        }

    return {
        "description": "Twins-only spatiotemporal annotations (supplement to the main SimVeRi benchmark release)",
        "version": "twins_extras_v1",
        "coordinate_system": "CARLA world coordinates (meters)",
        "total_records": len(annotations),
        "annotations": annotations,
    }


def _write_trajectory_info_csv(twins_caps: List[dict], vid_map: Dict[str, str], out_csv: Path) -> dict:
    tracks: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for cap in twins_caps:
        vid = str(cap.get("vehicle_id", ""))
        cam = str(cap.get("camera_id", ""))
        if not vid or not cam:
            continue
        tracks[(vid, cam)].append(cap)

    for k in tracks:
        tracks[k].sort(key=lambda c: float(c.get("timestamp", 0.0) or 0.0))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    track_id = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "track_id",
                "vehicle_id",
                "mapped_id",
                "camera_id",
                "start_time",
                "end_time",
                "duration",
                "image_count",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "avg_speed",
                "is_twins",
            ]
        )

        for (vid, cam), caps in sorted(tracks.items(), key=lambda x: (x[0][0], x[0][1])):
            mapped = vid_map.get(vid, "")
            if not mapped:
                continue
            start = caps[0]
            end = caps[-1]
            t0 = float(start.get("timestamp", 0.0) or 0.0)
            t1 = float(end.get("timestamp", 0.0) or 0.0)
            dur = t1 - t0
            speeds = [float(c.get("speed", 0.0) or 0.0) for c in caps]
            avg_speed = sum(speeds) / max(1, len(speeds))

            w.writerow(
                [
                    f"track_{track_id:04d}",
                    vid,
                    mapped,
                    cam,
                    f"{t0:.2f}",
                    f"{t1:.2f}",
                    f"{dur:.2f}",
                    len(caps),
                    f"{float(start.get('global_x', 0.0) or 0.0):.2f}",
                    f"{float(start.get('global_y', 0.0) or 0.0):.2f}",
                    f"{float(end.get('global_x', 0.0) or 0.0):.2f}",
                    f"{float(end.get('global_y', 0.0) or 0.0):.2f}",
                    f"{avg_speed:.2f}",
                    "true",
                ]
            )
            track_id += 1

    return {"tracks": track_id}


def _build_twins_groups(twins_caps: List[dict], vid_map: Dict[str, str]) -> dict:
    groups: Dict[str, dict] = {}
    for cap in twins_caps:
        if not _is_twins_cap(cap):
            continue
        fleet_id = str(cap.get("fleet_id") or "").strip()
        if not fleet_id:
            # Keep a fallback bucket so we don't silently drop Twins without fleet_id.
            fleet_id = "fleet_unknown"
        group_id = fleet_id.replace("fleet_", "group_")
        vid = str(cap.get("vehicle_id") or "")
        if not vid:
            continue
        g = groups.setdefault(
            group_id,
            {
                "fleet_id": fleet_id,
                "vehicles": [],
                "mapped_ids": [],
                "blueprint": cap.get("blueprint", ""),
                "color_name": cap.get("color_name", ""),
                "image_count": 0,
            },
        )
        if vid not in g["vehicles"]:
            g["vehicles"].append(vid)
        g["image_count"] += 1

    for gid, g in groups.items():
        g["mapped_ids"] = [vid_map.get(v, v) for v in g.get("vehicles", [])]
        g["vehicle_count"] = len(g.get("vehicles", []))

    # Stable ordering for public releases
    groups_sorted = {k: groups[k] for k in sorted(groups.keys())}
    total_images = sum(g["image_count"] for g in groups_sorted.values())
    total_vehicles = sum(g["vehicle_count"] for g in groups_sorted.values())

    return {
        "description": "Twins subset - groups of vehicles with identical appearance (supplement)",
        "version": "twins_extras_v1",
        "purpose": "Challenge visual-only re-identification with identical-looking vehicles",
        "total_groups": len(groups_sorted),
        "total_vehicles": total_vehicles,
        "total_images": total_images,
        "groups": groups_sorted,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Twins (H*) extras into an existing release")
    p.add_argument("--clean-dir", type=str, required=True, help="Cleaned output dir (output_cleaned_...)")
    p.add_argument("--release-dir", type=str, required=True, help="Main release dir (release_...)")
    p.add_argument("--out-subdir", type=str, default="extras/twins", help="Relative path under --release-dir")
    p.add_argument("--dry-run", action="store_true", help="Do not write files; only print counts/checks")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata/images if they already exist (safe resume).",
    )
    p.add_argument("--max-images", type=int, default=0, help="If >0, limit how many images to copy (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    release_dir = Path(args.release_dir)
    out_root = release_dir / Path(args.out_subdir)
    out_images = out_root / "images"
    out_meta = out_root / "metadata"
    out_stats = out_root / "statistics"

    input_json = clean_dir / "metadata" / "captures_cleaned.json"
    if not input_json.exists():
        raise FileNotFoundError(f"captures_cleaned.json not found: {input_json}")

    raw = _load_json(input_json)
    if not isinstance(raw, list):
        raise ValueError("captures_cleaned.json should be a list of capture dicts")

    twins_caps = [c for c in raw if isinstance(c, dict) and _is_twins_cap(c)]
    twins_caps = _sorted_caps(twins_caps)

    twins_vids = [str(c.get("vehicle_id")) for c in twins_caps if isinstance(c.get("vehicle_id"), str)]
    vid_map = _create_twins_id_mapping(twins_vids)

    # Sanity checks for public releases
    by_prefix = sum(1 for v in twins_vids if v.startswith("H"))
    by_flag = sum(1 for c in twins_caps if bool(c.get("is_fleet", False)))
    if by_prefix != len(twins_vids) or by_flag != len(twins_caps):
        print(f"[WARN] Twins identification mismatch: prefix(H*)={by_prefix} is_fleet={by_flag} total={len(twins_caps)}")

    cam_ids = sorted({str(c.get("camera_id", "")) for c in twins_caps if c.get("camera_id")})
    layer_dist = Counter(str(c.get("camera_layer", "")).lower().strip() for c in twins_caps)

    print("=" * 70)
    print("SimVeRi Twins Extras Export")
    print("=" * 70)
    print(f"Clean dir:   {clean_dir}")
    print(f"Release dir: {release_dir}")
    print(f"Output:      {out_root}")
    print("-" * 70)
    print(f"Twins images:   {len(twins_caps)}")
    print(f"Twins vehicles: {len(vid_map)}")
    print(f"Cameras (Twins): {len(cam_ids)}  {cam_ids[:10]}{'...' if len(cam_ids) > 10 else ''}")
    print(f"Camera layers:  {dict(layer_dist)}")

    if args.dry_run:
        print("[DRY RUN] No files will be written.")
        print("=" * 70)
        return

    out_images.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)
    _write_readme(out_root, overwrite=bool(args.overwrite))

    missing_images = 0
    copied = 0
    skipped_exists = 0
    exported_caps: List[dict] = []

    max_images = int(args.max_images or 0)

    for i, cap in enumerate(twins_caps):
        if max_images > 0 and i >= max_images:
            break

        orig_vid = cap.get("vehicle_id")
        if orig_vid not in vid_map:
            continue
        mapped = vid_map[orig_vid]
        cam = str(cap.get("camera_id", ""))
        frame = int(cap.get("frame_id", 0) or 0)
        new_name = _generate_filename(mapped, cam, frame)

        src = _resolve_src_image_path(clean_dir, cap)
        dst = out_images / new_name

        if not src.exists():
            missing_images += 1
            continue

        if dst.exists() and not args.overwrite:
            skipped_exists += 1
        else:
            shutil.copy2(str(src), str(dst))
            copied += 1

        # Build an export-friendly capture record (relative image path; keep original fields)
        rec = dict(cap)
        rec["mapped_id"] = mapped
        rec["image_name"] = new_name
        rec["image_path"] = str(Path("images") / new_name).replace("\\", "/")
        rec["source_image_path"] = str(cap.get("image_path") or "")
        exported_caps.append(rec)

    # Metadata exports
    _save_json(out_meta / "vehicle_id_mapping_twins.json", vid_map)
    _save_json(out_meta / "captures_twins.json", exported_caps)
    _save_json(out_meta / "spatiotemporal_twins.json", _build_spatiotemporal_annotations(twins_caps, vid_map))
    _save_json(out_meta / "twins_groups.json", _build_twins_groups(twins_caps, vid_map))

    traj_meta = _write_trajectory_info_csv(twins_caps, vid_map, out_meta / "trajectory_info_twins.csv")

    # Stats summary
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "clean_dir": os.path.abspath(str(clean_dir)),
        "release_dir": os.path.abspath(str(release_dir)),
        "output_root": os.path.abspath(str(out_root)),
        "twins": {
            "images": len(twins_caps),
            "vehicles": len(vid_map),
            "cameras": cam_ids,
            "camera_layer_distribution": dict(layer_dist),
        },
        "export": {
            "copied_images": copied,
            "skipped_existing_images": skipped_exists,
            "missing_images": missing_images,
        },
        "trajectory_info": traj_meta,
    }
    _save_json(out_stats / "twins_summary.json", summary)

    print("-" * 70)
    print("Done")
    print(f"Copied images: {copied} (skipped existing: {skipped_exists}, missing: {missing_images})")
    print(f"Saved: {out_meta / 'vehicle_id_mapping_twins.json'}")
    print(f"Saved: {out_meta / 'captures_twins.json'}")
    print(f"Saved: {out_meta / 'spatiotemporal_twins.json'}")
    print(f"Saved: {out_meta / 'trajectory_info_twins.csv'}")
    print(f"Saved: {out_meta / 'twins_groups.json'}")
    print(f"Saved: {out_stats / 'twins_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
