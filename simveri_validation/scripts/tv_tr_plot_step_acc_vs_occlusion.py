#!/usr/bin/env python
"""
Plot Step Accuracy vs occlusion bins (paper figure).

This uses the step-level debug CSV produced by tv_tr_evaluate.py:
  - per_step_<tag>.csv

and joins it with tracklet metadata (tracklets_*.json) to obtain occlusion.

Typical use (Base):
  python tv_tr_plot_step_acc_vs_occlusion.py \
    --tracklets-json .../tracklets_base.json \
    --per-step-csv .../per_step_base.csv \
    --out .../figures/step_acc_vs_occlusion_base.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_float, as_int, index_by, load_tracklets, read_csv_dicts, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Step Accuracy vs occlusion bins from per_step csv")
    p.add_argument("--tracklets-json", type=str, required=True, help="tracklets_base.json or tracklets_twins.json")
    p.add_argument("--per-step-csv", type=str, required=True, help="per_step_<tag>.csv from tv_tr_evaluate.py")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument(
        "--occ-field",
        type=str,
        default="occlusion_max",
        choices=("occlusion_max", "occlusion_mean"),
        help="Which tracklet occlusion field to use (default: occlusion_max)",
    )
    p.add_argument(
        "--bins",
        type=str,
        default="0,0.1,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated bin edges in [0,1] (default: 0,0.1,0.2,0.4,0.6,0.8,1.0)",
    )
    p.add_argument("--title", type=str, default="", help="Optional plot title")
    p.add_argument("--dpi", type=int, default=160, help="Figure dpi (default: 160)")
    p.add_argument("--save-json", type=str, default="", help="Optional: save aggregated stats JSON to this path")
    return p.parse_args()


def _parse_bins(spec: str) -> List[float]:
    xs: List[float] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        xs.append(float(part))
    xs = sorted(set(xs))
    if len(xs) < 2:
        raise ValueError("bins must contain at least 2 edges")
    return xs


def _bin_index(edges: List[float], x: float) -> int:
    # Return i where edges[i] <= x < edges[i+1] (last edge is inclusive).
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if a <= x <= b:
                return i
        if a <= x < b:
            return i
    return -1


def main() -> None:
    args = parse_args()
    edges = _parse_bins(args.bins)

    tracklets, _ = load_tracklets(args.tracklets_json)
    t_by_id: Dict[str, dict] = index_by(tracklets, key="track_id")

    rows = read_csv_dicts(args.per_step_csv)
    methods = ("B0", "B1", "B2")

    bin_count = [0 for _ in range(len(edges) - 1)]
    bin_correct = {m: [0 for _ in range(len(edges) - 1)] for m in methods}

    missing_tracklets = 0
    missing_occ = 0

    for r in rows:
        tid = str(r.get("gt_track_id", "")).strip()
        if not tid:
            continue
        t = t_by_id.get(tid)
        if t is None:
            missing_tracklets += 1
            continue
        occ = as_float(t.get(args.occ_field), float("nan"))
        if occ != occ:
            missing_occ += 1
            continue
        bi = _bin_index(edges, occ)
        if bi < 0:
            continue
        bin_count[bi] += 1
        for m in methods:
            bin_correct[m][bi] += as_int(r.get(f"correct_{m}", 0))

    # Build a table for plotting and optional saving.
    table: List[dict] = []
    for i in range(len(edges) - 1):
        n = bin_count[i]
        rec = {
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(n),
        }
        for m in methods:
            rec[f"acc_{m}"] = float(bin_correct[m][i] / n) if n > 0 else float("nan")
        table.append(rec)

    if args.save_json:
        save_json(
            args.save_json,
            {
                "tracklets_json": os.path.abspath(args.tracklets_json),
                "per_step_csv": os.path.abspath(args.per_step_csv),
                "occ_field": args.occ_field,
                "bins": edges,
                "missing_tracklets": int(missing_tracklets),
                "missing_occ": int(missing_occ),
                "table": table,
            },
            indent=2,
        )

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            labels.append(f"[{a:.1f},{b:.1f}]")
        else:
            labels.append(f"[{a:.1f},{b:.1f})")

    xs = list(range(len(labels)))
    width = 0.25
    offsets = {"B0": -width, "B1": 0.0, "B2": width}
    colors = {"B0": "#E45756", "B1": "#4C78A8", "B2": "#54A24B"}

    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)

    for m in methods:
        ys = [table[i][f"acc_{m}"] for i in range(len(labels))]
        ax.bar([x + offsets[m] for x in xs], ys, width=width, label=m, color=colors[m])

    # Annotate counts on top of the B1 bars (center) for readability.
    for i, x in enumerate(xs):
        n = bin_count[i]
        ax.text(x, 1.02, f"n={n}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Step Accuracy")
    ax.set_xlabel(f"Occlusion bins ({args.occ_field})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", ncol=3, frameon=True)

    title = args.title.strip()
    if not title:
        title = "Step Accuracy vs Occlusion"
    ax.set_title(title)

    fig.tight_layout()
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print("Saved:")
    print(f"  {out}")
    if missing_tracklets or missing_occ:
        print(f"Warnings: missing_tracklets={missing_tracklets}, missing_occ={missing_occ}")


if __name__ == "__main__":
    main()

