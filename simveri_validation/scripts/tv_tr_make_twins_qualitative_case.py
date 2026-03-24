#!/usr/bin/env python
"""
Generate a qualitative Twins TR-Closed case visualization (paper figure).

Goal:
  - Show a 7-step chain (6 decisions) where each step has 5 within-group
    candidates, and visualize which candidate each method selects (B0/B1/B2).

Inputs:
  - tracklets_twins.json (from tv_tr_build_tracklets.py)
  - candidates_twins_group.json (from tv_tr_build_candidates.py --protocol twins_group)
  - per_step_twins_group.csv (from tv_tr_evaluate.py --tag twins_group)
  - Twins extras root (contains images/)

Output:
  - twins_case_vehicle_<vid>.png + twins_case_vehicle_<vid>.json under --out-dir
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_int, index_by, load_json, load_tracklets, now_iso, read_csv_dicts, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a qualitative Twins within-group TR case figure")
    p.add_argument("--twins-root", type=str, required=True, help="Path to <release>/extras/twins (contains images/)")
    p.add_argument("--tracklets-json", type=str, required=True, help="Path to tracklets_twins.json")
    p.add_argument("--candidates-json", type=str, required=True, help="Path to candidates_twins_group.json")
    p.add_argument("--per-step-csv", type=str, required=True, help="Path to per_step_twins_group.csv")
    p.add_argument("--out-dir", type=str, required=True, help="Output dir for the figure")
    p.add_argument("--vehicle-id", type=str, default="", help="Optional: force a specific vehicle_id (mapped 4-digit)")
    p.add_argument(
        "--prefer",
        type=str,
        default="B0_wrong_B2_right",
        choices=("B0_wrong_B2_right", "max_improve_B2_minus_B0"),
        help="Auto-selection strategy if --vehicle-id is not set",
    )
    p.add_argument("--dpi", type=int, default=160, help="Figure dpi (default: 160)")
    return p.parse_args()


def _rep_image_name(tracklet: dict) -> Optional[str]:
    names = tracklet.get("image_names") or []
    if not isinstance(names, list) or not names:
        return None
    return str(names[len(names) // 2])


def _vehicle_chain_exact(per_step_rows: List[dict], method: str) -> int:
    # Chain exact: all steps correct for a vehicle.
    for r in per_step_rows:
        if as_int(r.get(f"correct_{method}", 0)) != 1:
            return 0
    return 1 if per_step_rows else 0


def _vehicle_total_correct(per_step_rows: List[dict], method: str) -> int:
    return sum(as_int(r.get(f"correct_{method}", 0)) for r in per_step_rows)


def _pick_vehicle(per_step_by_vid: Dict[str, List[dict]], prefer: str) -> Tuple[str, str]:
    """
    Return (vehicle_id, reason).
    """
    vids = sorted(per_step_by_vid.keys())
    if not vids:
        raise ValueError("per_step csv has no vehicle rows")

    if prefer == "B0_wrong_B2_right":
        good = []
        for vid in vids:
            rows = sorted(per_step_by_vid[vid], key=lambda r: as_int(r.get("step", 0)))
            if _vehicle_chain_exact(rows, "B0") == 0 and _vehicle_chain_exact(rows, "B2") == 1:
                good.append(vid)
        if good:
            return good[0], "first vid with chain_exact_B0=0 and chain_exact_B2=1"

    # Fallback: choose max improvement in total correct steps, tie-break smallest vid.
    best_vid = vids[0]
    best_imp = -10**9
    for vid in vids:
        rows = sorted(per_step_by_vid[vid], key=lambda r: as_int(r.get("step", 0)))
        imp = _vehicle_total_correct(rows, "B2") - _vehicle_total_correct(rows, "B0")
        if imp > best_imp or (imp == best_imp and vid < best_vid):
            best_imp = imp
            best_vid = vid
    return best_vid, f"fallback: max improvement (B2_correct - B0_correct) = {best_imp}"


def _markers_for(track_id: str, gt: str, preds: Dict[str, str]) -> List[str]:
    out: List[str] = []
    if track_id == gt:
        out.append("GT")
    for m in ("B0", "B1", "B2"):
        if track_id == preds.get(m):
            out.append(m)
    return out


def main() -> None:
    args = parse_args()
    twins_images = os.path.join(args.twins_root, "images")
    if not os.path.isdir(twins_images):
        raise FileNotFoundError(f"Twins images dir not found: {twins_images}")

    tracklets, meta_t = load_tracklets(args.tracklets_json)
    t_by_id: Dict[str, dict] = index_by(tracklets, key="track_id")

    cand_obj = load_json(args.candidates_json)
    vehicles = cand_obj.get("vehicles", [])
    if not isinstance(vehicles, list):
        raise ValueError("candidates json must have a list field 'vehicles'")
    cand_by_vid = {str(v.get("vehicle_id")): v for v in vehicles if isinstance(v, dict)}

    per_step_rows = read_csv_dicts(args.per_step_csv)
    per_step_by_vid: Dict[str, List[dict]] = defaultdict(list)
    for r in per_step_rows:
        vid = str(r.get("vehicle_id", "")).strip()
        if not vid:
            continue
        per_step_by_vid[vid].append(r)

    if args.vehicle_id.strip():
        vid = args.vehicle_id.strip()
        if vid not in per_step_by_vid:
            raise KeyError(f"--vehicle-id {vid} not found in per_step csv")
        reason = "user-specified"
    else:
        vid, reason = _pick_vehicle(per_step_by_vid, args.prefer)

    if vid not in cand_by_vid:
        raise KeyError(f"vehicle_id {vid} not found in candidates json")

    # Prepare step records
    vrec = cand_by_vid[vid]
    steps = vrec.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"No steps for vehicle {vid} in candidates json")

    rows_vid = sorted(per_step_by_vid[vid], key=lambda r: as_int(r.get("step", 0)))
    pred_by_step: Dict[int, Dict[str, str]] = {}
    for r in rows_vid:
        si = as_int(r.get("step", 0))
        pred_by_step[si] = {
            "B0": str(r.get("pred_B0", "")).strip(),
            "B1": str(r.get("pred_B1", "")).strip(),
            "B2": str(r.get("pred_B2", "")).strip(),
        }

    # Load images and plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required for image IO") from e

    os.makedirs(args.out_dir, exist_ok=True)

    # Assume within-group protocol: 5 candidates per step. Fall back to per-step candidate_count.
    n_steps = len(steps)
    n_cols = max(as_int(s.get("candidate_count", 0)) for s in steps) if steps else 5
    n_cols = max(1, int(n_cols))

    fig = plt.figure(figsize=(2.4 * n_cols, 2.4 * n_steps))

    for r_i, s in enumerate(sorted(steps, key=lambda x: as_int(x.get("step", 0)))):
        si = as_int(s.get("step", 0))
        cam = str(s.get("camera_id", "")).strip()
        gt = str(s.get("gt_track_id", "")).strip()
        cands = s.get("candidates", [])
        if not isinstance(cands, list):
            cands = []

        preds = pred_by_step.get(si, {"B0": "", "B1": "", "B2": ""})
        for c_i, tid in enumerate(cands):
            tid = str(tid).strip()
            ax = fig.add_subplot(n_steps, n_cols, r_i * n_cols + c_i + 1)
            ax.axis("off")

            t = t_by_id.get(tid, {})
            rep = _rep_image_name(t) if isinstance(t, dict) else None
            if rep:
                img_path = os.path.join(twins_images, rep)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "missing\nimage", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "no\nimages", ha="center", va="center")

            markers = _markers_for(tid, gt, preds)
            vid_cand = str(t.get("vehicle_id_mapped", t.get("vehicle_id", ""))) if isinstance(t, dict) else ""
            title = f"{vid_cand}"
            if markers:
                title += "\n" + ",".join(markers)
            ax.set_title(title, fontsize=8)

            # Row label (step/camera) on first column only.
            if c_i == 0:
                ax.set_ylabel(f"step {si}\n{cam}", fontsize=9)

    fig.suptitle(f"Twins qualitative case (vehicle_id={vid}; {reason})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = os.path.join(args.out_dir, f"twins_case_vehicle_{vid}.png")
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    # Save a small case descriptor for strict reproducibility.
    out_json = os.path.join(args.out_dir, f"twins_case_vehicle_{vid}.json")
    save_json(
        out_json,
        {
            "generated_at": now_iso(),
            "vehicle_id": vid,
            "twins_group": vrec.get("twins_group"),
            "selection_reason": reason,
            "tracklets_json": os.path.abspath(args.tracklets_json),
            "candidates_json": os.path.abspath(args.candidates_json),
            "per_step_csv": os.path.abspath(args.per_step_csv),
            "twins_root": os.path.abspath(args.twins_root),
            "chain_track_ids": vrec.get("chain_track_ids"),
            "steps": [
                {
                    "step": as_int(s.get("step", 0)),
                    "camera_id": s.get("camera_id"),
                    "gt_track_id": s.get("gt_track_id"),
                    "candidates": s.get("candidates"),
                    "pred": pred_by_step.get(as_int(s.get("step", 0)), {}),
                }
                for s in sorted(steps, key=lambda x: as_int(x.get("step", 0)))
            ],
            "tracklets_meta": meta_t,
            "candidates_meta": cand_obj.get("meta", {}),
        },
        indent=2,
    )

    print("Saved:")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()

