#!/usr/bin/env python
"""
Plot co-view (co-observation) artifacts from coview_pairs.json.

This script is intentionally independent from the main pipeline and can be run
after tv_tr_fit_coview_pairs.py to produce paper-ready figures.

Inputs:
  - coview_pairs.json (from tv_tr_fit_coview_pairs.py)

Outputs (under --out-dir):
  - coview_pairs_median_dt_hist.pdf
  - coview_pairs_top.-pdf

Notes:
  - "median_dt_s" is computed from exit-to-entry time:
        dt = start_time(next) - end_time(curr)
  - co-view pairs typically have dt close to 0 or negative (overlap).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import as_bool, as_float, load_json, now_iso, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot co-view pair statistics from coview_pairs.json")
    p.add_argument("--coview-json", type=str, required=True, help="Path to coview_pairs.json")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for figures")
    p.add_argument("--topk", type=int, default=20, help="Top-K smallest-median pairs to visualize (default: 20)")
    p.add_argument(
        "--hist-max-dt-s",
        type=float,
        default=60.0,
        help="Clip dt for histogram x-range (seconds). Use <=0 to disable clipping.",
    )
    p.add_argument("--dpi", type=int, default=160, help="Figure dpi (default: 160)")
    return p.parse_args()


def _load_pairs(coview_json: str) -> Dict[str, Any]:
    obj = load_json(coview_json)
    if not isinstance(obj, dict) or "pairs" not in obj:
        raise ValueError(f"Invalid coview json: {coview_json}")
    return obj


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    obj = _load_pairs(args.coview_json)
    meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
    pairs = obj.get("pairs", {})
    if not isinstance(pairs, dict):
        raise ValueError("coview json must have a dict field 'pairs'")

    threshold_s = as_float(meta.get("threshold_s"), 2.0)

    rows: List[Dict[str, Any]] = []
    for _, rec in pairs.items():
        if not isinstance(rec, dict):
            continue
        a = str(rec.get("from", "")).strip()
        b = str(rec.get("to", "")).strip()
        if not a or not b:
            continue
        rows.append(
            {
                "pair": f"{a}->{b}",
                "from": a,
                "to": b,
                "n": int(as_float(rec.get("n"), 0)),
                "median_dt_s": as_float(rec.get("median_dt_s"), float("nan")),
                "median_dt_entry_s": as_float(rec.get("median_dt_entry_s"), float("nan")),
                "overlap_rate": as_float(rec.get("overlap_rate"), float("nan")),
                "coview": as_bool(rec.get("coview"), False),
            }
        )

    rows = [r for r in rows if r["median_dt_s"] == r["median_dt_s"]]  # filter NaN
    rows_sorted = sorted(rows, key=lambda r: float(r["median_dt_s"]))

    # Save a lightweight sorted table for debugging / appendix.
    save_json(
        os.path.join(args.out_dir, "coview_pairs_sorted.json"),
        {"generated_at": now_iso(), "threshold_s": threshold_s, "pairs": rows_sorted},
        indent=2,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in the evaluation env."
        ) from e

    # 1) Histogram of median_dt_s
    all_dt = [float(r["median_dt_s"]) for r in rows_sorted]
    coview_dt = [float(r["median_dt_s"]) for r in rows_sorted if r["coview"]]
    non_dt = [float(r["median_dt_s"]) for r in rows_sorted if not r["coview"]]

    if args.hist_max_dt_s and args.hist_max_dt_s > 0:
        xmax = float(args.hist_max_dt_s)
        all_dt_plot = [min(x, xmax) for x in all_dt]
        coview_dt_plot = [min(x, xmax) for x in coview_dt]
        non_dt_plot = [min(x, xmax) for x in non_dt]
    else:
        xmax = max(all_dt) if all_dt else 1.0
        all_dt_plot, coview_dt_plot, non_dt_plot = all_dt, coview_dt, non_dt

    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    bins = 40
    ax.hist(non_dt_plot, bins=bins, alpha=0.75, color="#4C78A8", label="non-coview")
    if coview_dt_plot:
        ax.hist(coview_dt_plot, bins=bins, alpha=0.75, color="#F58518", label="coview")
    ax.axvline(threshold_s, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold_s:.1f}s")
    ax.set_xlabel("median exit-to-entry dt (s)")
    ax.set_ylabel("directed camera pairs")
    ax.set_title("Co-view artifact summary (median dt per directed camera pair)")
    ax.set_xlim(min(-1.0, min(all_dt_plot, default=0.0)), xmax)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out_hist = os.path.join(args.out_dir, "coview_pairs_median_dt_hist.pdf")
    fig.savefig(out_hist, dpi=args.dpi)
    plt.close(fig)

    # 2) Top-K smallest median_dt_s pairs
    topk = max(1, int(args.topk))
    top = rows_sorted[:topk]
    labels = [r["pair"] for r in top]
    values = [float(r["median_dt_s"]) for r in top]
    colors = ["#F58518" if r["coview"] else "#9ecae9" for r in top]

    fig = plt.figure(figsize=(8.0, max(3.0, 0.28 * len(top) + 1.0)))
    ax = fig.add_subplot(1, 1, 1)
    y = list(range(len(top)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold_s, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("median dt (s)")
    ax.set_title(f"Top-{len(top)} smallest-median camera pairs (orange=coview)")
    fig.tight_layout()
    out_top = os.path.join(args.out_dir, "coview_pairs_top.pdf")
    fig.savefig(out_top, dpi=args.dpi)
    plt.close(fig)

    print("Saved:")
    print(f"  {out_hist}")
    print(f"  {out_top}")


if __name__ == "__main__":
    main()

