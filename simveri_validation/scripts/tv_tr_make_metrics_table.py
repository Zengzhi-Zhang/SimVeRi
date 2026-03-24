#!/usr/bin/env python
"""
Make a table-ready CSV from one or more TR metrics_*.json files.

This is intentionally a small standalone utility so the evaluation script
doesn't need to manage appending/merging across multiple subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TR metrics.csv from metrics_*.json files")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--metrics-json", type=str, nargs="+", required=True, help="One or more metrics_*.json files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []

    for path in args.metrics_json:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        tag = (obj.get("meta") or {}).get("tag", os.path.splitext(os.path.basename(path))[0])
        methods = obj.get("methods") or {}
        methods_group = obj.get("methods_group") or {}

        for m, rec in methods.items():
            if not isinstance(rec, dict):
                continue
            row: Dict[str, object] = {
                "subset": tag,
                "method": m,
                "step_acc_macro": rec.get("step_acc_macro"),
                "transition_acc_macro": rec.get("transition_acc_macro"),
                "chain_exact": rec.get("chain_exact"),
            }
            if m in methods_group and isinstance(methods_group.get(m), dict):
                grec = methods_group[m]
                row.update(
                    {
                        "step_acc_group_macro": grec.get("step_acc_macro"),
                        "transition_acc_group_macro": grec.get("transition_acc_macro"),
                        "chain_exact_group": grec.get("chain_exact"),
                    }
                )
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows generated.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    # Stable header union (some subsets may not have group metrics).
    header: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in header:
                header.append(k)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {args.out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()

