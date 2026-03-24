#!/usr/bin/env python
"""
Evaluate TR trajectory reconstruction baselines (B0/B1/B2) on a TR-Closed candidate set.

Inputs:
  - tracklets_*.json
  - tracklet_features_*.npy + tracklet_index_*.json
  - candidates_*.json
  - coview_pairs.json
  - global_speed_prior.json

Outputs:
  - metrics_<tag>.json
  - per_case_<tag>.csv
  - per_step_<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tech_validation_tr.common import (
    angdiff_deg,
    as_float,
    index_by,
    load_coview_pairs,
    load_speed_prior,
    load_tracklets,
    now_iso,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TR reconstruction (B0/B1/B2)")
    p.add_argument("--tracklets-json", type=str, required=True)
    p.add_argument("--tracklet-features-npy", type=str, required=True)
    p.add_argument("--tracklet-index-json", type=str, required=True)
    p.add_argument("--candidates-json", type=str, required=True)
    p.add_argument("--coview-json", type=str, required=True)
    p.add_argument("--speed-prior-json", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="Output tag used in filenames")

    # Hyperparams (fixed; no tuning for Scientific Data technical validation)
    p.add_argument("--sigma-h-deg", type=float, default=30.0)
    p.add_argument("--w-vis", type=float, default=0.8)
    p.add_argument("--eps", type=float, default=1e-12)
    return p.parse_args()


def _load_tracklet_feature_map(features_npy: str, index_json: str) -> Tuple[np.ndarray, Dict[str, int]]:
    feats = np.load(features_npy)
    with open(index_json, "r", encoding="utf-8") as f:
        idx_list = json.load(f)
    if not isinstance(idx_list, list):
        raise ValueError("tracklet_index json must be a list of records")
    tid2i: Dict[str, int] = {}
    for rec in idx_list:
        if not isinstance(rec, dict):
            continue
        tid = rec.get("track_id")
        i = rec.get("index")
        if isinstance(tid, str) and isinstance(i, int):
            tid2i[tid] = i
    if feats.shape[0] != len(tid2i):
        # Not necessarily fatal (index may contain extra fields), but should match.
        pass
    return feats, tid2i


@dataclass
class EdgeScore:
    valid: bool
    s_vis: float
    s_dir: float
    s_time: float
    s_st: float
    s_total: float


class Scorer:
    def __init__(
        self,
        tracklets_by_id: Dict[str, dict],
        feats: np.ndarray,
        tid2i: Dict[str, int],
        coview_map: Dict[Tuple[str, str], dict],
        v_mean: float,
        v_std: float,
        v_max: float,
        sigma_h: float,
        w_vis: float,
    ):
        self.t = tracklets_by_id
        self.feats = feats
        self.tid2i = tid2i
        self.coview_map = coview_map
        self.v_mean = float(v_mean)
        self.v_std = float(v_std)
        self.v_max = float(v_max)
        self.sigma_h = float(sigma_h)
        self.w_vis = float(w_vis)

    def _is_coview(self, cam_a: str, cam_b: str) -> bool:
        rec = self.coview_map.get((cam_a, cam_b))
        return bool(isinstance(rec, dict) and rec.get("coview", False))

    def _feat(self, tid: str) -> np.ndarray:
        idx = self.tid2i.get(tid)
        if idx is None:
            raise KeyError(
                f"track_id not found in feature index: {tid}. "
                f"Have {len(self.tid2i)} track_ids. "
                f"Check that you are using matching (tracklets_json, tracklet_features_*.npy, tracklet_index_*.json) "
                f"and that candidates were built from the same tracklets file."
            )
        return self.feats[int(idx)]

    def edge(self, prev_id: str, cand_id: str) -> EdgeScore:
        a = self.t[prev_id]
        b = self.t[cand_id]

        cam_a = str(a.get("camera_id", ""))
        cam_b = str(b.get("camera_id", ""))

        # Visual score in [0, 1]
        va = self._feat(prev_id)
        vb = self._feat(cand_id)
        cos = float(np.dot(va, vb))
        cos = max(-1.0, min(1.0, cos))
        s_vis = 0.5 * (cos + 1.0)

        # Direction continuity (heading-based)
        ha = as_float(a.get("heading_end"))
        hb = as_float(b.get("heading_start"))
        if math.isfinite(ha) and math.isfinite(hb) and self.sigma_h > 0:
            d = angdiff_deg(ha, hb)
            s_dir = math.exp(-(d * d) / (2.0 * self.sigma_h * self.sigma_h))
        else:
            s_dir = 1.0

        # Time gating + score (exit->entry)
        t_end = as_float(a.get("t_end"))
        t_next = as_float(b.get("t_start", b.get("t_event", 0.0)))
        dt = float(t_next - t_end)

        coview = self._is_coview(cam_a, cam_b)

        # dt<=0 is only plausible for co-view camera pairs (overlapping FOV / near-simultaneous observations).
        # For non-co-view pairs, dt<=0 means the candidate happened before the previous tracklet ended, so the edge
        # must be invalid; otherwise the spatiotemporal term would incorrectly favor time-reversed candidates.
        if dt <= 0:
            if coview:
                valid = True
                s_time = 1.0
            else:
                valid = False
                s_time = 0.0
        else:
            x_exit = as_float(a.get("x_exit"))
            y_exit = as_float(a.get("y_exit"))
            x_ent = as_float(b.get("x_event"))
            y_ent = as_float(b.get("y_event"))
            dxy = math.hypot(x_ent - x_exit, y_ent - y_exit)
            v_est = 3.6 * dxy / dt

            if (not coview) and (v_est > self.v_max):
                valid = False
                s_time = 0.0
            else:
                valid = True
                if coview:
                    s_time = 1.0
                else:
                    s_time = math.exp(-((v_est - self.v_mean) ** 2) / (2.0 * (self.v_std**2)))

        s_st = float(s_time * s_dir)
        s_total = float(self.w_vis * s_vis + (1.0 - self.w_vis) * s_st)

        return EdgeScore(valid=valid, s_vis=float(s_vis), s_dir=float(s_dir), s_time=float(s_time), s_st=s_st, s_total=s_total)


def _greedy_chain(
    gt_chain: List[str],
    steps: List[dict],
    scorer: Scorer,
    use_st: bool,
) -> Tuple[List[str], int]:
    """
    Return predicted chain (length L) with gt_chain[0] as probe fixed.
    Also returns how many times we had to fall back due to gating (B1 only).
    """
    pred = [gt_chain[0]]
    fallback = 0
    prev = gt_chain[0]

    for step in steps:
        cands = list(step.get("candidates") or [])
        if not cands:
            pred.append(prev)
            continue

        best_id = None
        best_score = -1e9
        any_valid = False

        if use_st:
            # Use S_total with gating; if all invalid, fall back to visual-only.
            for cid in cands:
                es = scorer.edge(prev, cid)
                if not es.valid:
                    continue
                any_valid = True
                if es.s_total > best_score:
                    best_score = es.s_total
                    best_id = cid

            if not any_valid:
                fallback += 1
                # Fall back to B0 visual-only.
                best_score = -1e9
                for cid in cands:
                    es = scorer.edge(prev, cid)
                    if es.s_vis > best_score:
                        best_score = es.s_vis
                        best_id = cid
        else:
            for cid in cands:
                es = scorer.edge(prev, cid)
                if es.s_vis > best_score:
                    best_score = es.s_vis
                    best_id = cid

        assert best_id is not None
        pred.append(best_id)
        prev = best_id

    return pred, fallback


def _viterbi_chain(
    gt_chain: List[str],
    steps: List[dict],
    scorer: Scorer,
    eps: float,
) -> List[str]:
    """
    Viterbi on layered candidates; start node is the probe tracklet gt_chain[0].
    """
    probe = gt_chain[0]
    layers = [list(s.get("candidates") or []) for s in steps]  # steps correspond to t=2..L

    # dp values for previous layer
    prev_ids = [probe]
    prev_dp = np.array([0.0], dtype=np.float64)
    backptrs: List[List[int]] = []
    layer_ids: List[List[str]] = []

    for cand_ids in layers:
        if not cand_ids:
            # Degenerate; keep probe.
            layer_ids.append([probe])
            backptrs.append([0])
            prev_ids = [probe]
            prev_dp = np.array([0.0], dtype=np.float64)
            continue

        dp_next = np.full((len(cand_ids),), -np.inf, dtype=np.float64)
        bp_next = [-1] * len(cand_ids)

        for j, cid in enumerate(cand_ids):
            best = -np.inf
            best_i = -1
            for i, pid in enumerate(prev_ids):
                es = scorer.edge(pid, cid)
                if not es.valid:
                    continue
                w = math.log(max(float(es.s_total), float(eps)))
                v = float(prev_dp[i] + w)
                if v > best:
                    best = v
                    best_i = i
            dp_next[j] = best
            bp_next[j] = best_i

        # If all -inf due to gating, fall back to visual-only edges (no gating).
        if not np.isfinite(dp_next).any():
            for j, cid in enumerate(cand_ids):
                best = -np.inf
                best_i = -1
                for i, pid in enumerate(prev_ids):
                    es = scorer.edge(pid, cid)
                    w = math.log(max(float(es.s_vis), float(eps)))
                    v = float(prev_dp[i] + w)
                    if v > best:
                        best = v
                        best_i = i
                dp_next[j] = best
                bp_next[j] = best_i

        backptrs.append(bp_next)
        layer_ids.append(cand_ids)
        prev_ids = cand_ids
        prev_dp = dp_next

    # Select best in final layer
    if not layer_ids:
        return [probe]
    last = layer_ids[-1]
    if len(prev_dp) == 0:
        j_best = 0
    elif not np.isfinite(prev_dp).any():
        # Degenerate: all paths invalid. Pick an arbitrary candidate (index 0) and continue.
        # This should be extremely rare because we fall back to visual-only edges when gating kills all edges.
        j_best = 0
    else:
        j_best = int(np.argmax(prev_dp))

    # Backtrack
    path = [last[j_best]]
    for t in range(len(layer_ids) - 1, 0, -1):
        j_best = backptrs[t][j_best]
        if j_best < 0:
            j_best = 0
        path.append(layer_ids[t - 1][j_best])
    path.reverse()

    return [probe] + path


def _vehicle_metrics(gt: List[str], pred: List[str]) -> Tuple[float, float, float]:
    L = len(gt)
    if L <= 1:
        return 1.0, 1.0, 1.0
    step_correct = sum(1 for t in range(1, L) if pred[t] == gt[t])
    step_acc = step_correct / float(L - 1)
    if L <= 2:
        trans_acc = 1.0
    else:
        trans_correct = sum(1 for t in range(2, L) if (pred[t - 1] == gt[t - 1] and pred[t] == gt[t]))
        trans_acc = trans_correct / float(L - 2)
    chain_exact = 1.0 if all(pred[t] == gt[t] for t in range(1, L)) else 0.0
    return step_acc, trans_acc, chain_exact


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tracklets, meta_track = load_tracklets(args.tracklets_json)
    t_by_id = index_by(tracklets, key="track_id")

    feats, tid2i = _load_tracklet_feature_map(args.tracklet_features_npy, args.tracklet_index_json)

    coview = load_coview_pairs(args.coview_json)
    prior = load_speed_prior(args.speed_prior_json)

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        cand_obj = json.load(f)
    vehicles = cand_obj.get("vehicles", [])
    meta_cand = cand_obj.get("meta", {})
    if not isinstance(vehicles, list):
        raise ValueError("candidates json must have a list field 'vehicles'")

    scorer = Scorer(
        tracklets_by_id=t_by_id,
        feats=feats,
        tid2i=tid2i,
        coview_map=coview,
        v_mean=prior.v_mean_kmh,
        v_std=prior.v_std_kmh,
        v_max=prior.v_max_kmh,
        sigma_h=float(args.sigma_h_deg),
        w_vis=float(args.w_vis),
    )

    has_groups = any(t.get("twins_group") is not None for t in tracklets)

    # Global accumulators
    per_case_rows: List[Dict[str, object]] = []
    per_step_rows: List[Dict[str, object]] = []

    step_acc_B0: List[float] = []
    step_acc_B1: List[float] = []
    step_acc_B2: List[float] = []
    trans_acc_B0: List[float] = []
    trans_acc_B1: List[float] = []
    trans_acc_B2: List[float] = []
    chain_B0: List[float] = []
    chain_B1: List[float] = []
    chain_B2: List[float] = []

    # Twins group-level accumulators (only if has_groups)
    step_acc_g_B0: List[float] = []
    step_acc_g_B1: List[float] = []
    step_acc_g_B2: List[float] = []
    trans_acc_g_B0: List[float] = []
    trans_acc_g_B1: List[float] = []
    trans_acc_g_B2: List[float] = []
    chain_g_B0: List[float] = []
    chain_g_B1: List[float] = []
    chain_g_B2: List[float] = []
    group_vehicle_count = 0
    group_skipped_missing_label = 0

    # By chain length bins (macro per L)
    by_L: Dict[int, Dict[str, List[float]]] = {}

    # Step-level buckets by candidate count
    buckets = ["<=1", "2", ">=3"]
    step_bucket_correct = {m: {b: 0 for b in buckets} for m in ("B0", "B1", "B2")}
    step_bucket_total = {b: 0 for b in buckets}

    b1_fallback_steps = 0

    def bucket(k: int) -> str:
        if k <= 1:
            return "<=1"
        if k == 2:
            return "2"
        return ">=3"

    for vrec in vehicles:
        if not isinstance(vrec, dict):
            continue
        vid = str(vrec.get("vehicle_id", ""))
        gt_chain = list(vrec.get("chain_track_ids") or [])
        steps = list(vrec.get("steps") or [])
        L = int(vrec.get("chain_len") or len(gt_chain))
        if len(gt_chain) != L:
            L = len(gt_chain)
        if L < 2:
            continue

        # Predict chains
        pred_B0, _ = _greedy_chain(gt_chain, steps, scorer, use_st=False)
        pred_B1, fb = _greedy_chain(gt_chain, steps, scorer, use_st=True)
        pred_B2 = _viterbi_chain(gt_chain, steps, scorer, eps=float(args.eps))

        b1_fallback_steps += fb

        # Metrics per vehicle (vehicle-level)
        s0, t0, c0 = _vehicle_metrics(gt_chain, pred_B0)
        s1, t1, c1 = _vehicle_metrics(gt_chain, pred_B1)
        s2, t2, c2 = _vehicle_metrics(gt_chain, pred_B2)

        step_acc_B0.append(s0)
        step_acc_B1.append(s1)
        step_acc_B2.append(s2)
        trans_acc_B0.append(t0)
        trans_acc_B1.append(t1)
        trans_acc_B2.append(t2)
        chain_B0.append(c0)
        chain_B1.append(c1)
        chain_B2.append(c2)

        # Chain-length buckets
        by_L.setdefault(int(L), {"B0": [], "B1": [], "B2": []})
        by_L[int(L)]["B0"].append(s0)
        by_L[int(L)]["B1"].append(s1)
        by_L[int(L)]["B2"].append(s2)

        # Twins group-level metrics (optional; only computed when group labels exist for the whole GT chain).
        group_metrics: Optional[dict] = None
        if has_groups:
            def _group_label(track_id: str) -> Optional[str]:
                g = t_by_id.get(track_id, {}).get("twins_group")
                if isinstance(g, str) and g:
                    return g
                return None

            gt_groups = [_group_label(tid) for tid in gt_chain]
            if any(g is None for g in gt_groups):
                group_skipped_missing_label += 1
            else:
                group_vehicle_count += 1
                # Missing predicted group labels are treated as incorrect (never match GT).
                p0_groups = [_group_label(tid) or "__MISSING__" for tid in pred_B0]
                p1_groups = [_group_label(tid) or "__MISSING__" for tid in pred_B1]
                p2_groups = [_group_label(tid) or "__MISSING__" for tid in pred_B2]

                gt_groups_str = [str(g) for g in gt_groups]  # all non-None
                gs0, gt0, gc0 = _vehicle_metrics(gt_groups_str, p0_groups)
                gs1, gt1, gc1 = _vehicle_metrics(gt_groups_str, p1_groups)
                gs2, gt2, gc2 = _vehicle_metrics(gt_groups_str, p2_groups)

                step_acc_g_B0.append(gs0)
                step_acc_g_B1.append(gs1)
                step_acc_g_B2.append(gs2)
                trans_acc_g_B0.append(gt0)
                trans_acc_g_B1.append(gt1)
                trans_acc_g_B2.append(gt2)
                chain_g_B0.append(gc0)
                chain_g_B1.append(gc1)
                chain_g_B2.append(gc2)

                group_metrics = {
                    "step_acc_group_B0": gs0,
                    "step_acc_group_B1": gs1,
                    "step_acc_group_B2": gs2,
                    "transition_acc_group_B0": gt0,
                    "transition_acc_group_B1": gt1,
                    "transition_acc_group_B2": gt2,
                    "chain_exact_group_B0": gc0,
                    "chain_exact_group_B1": gc1,
                    "chain_exact_group_B2": gc2,
                }

        # Per-step correctness for buckets
        for si, step in enumerate(steps, start=2):
            gt_id = str(step.get("gt_track_id", ""))
            cand_count = int(step.get("candidate_count") or len(step.get("candidates") or []))
            b = bucket(cand_count)
            step_bucket_total[b] += 1

            p0 = pred_B0[si - 1]
            p1 = pred_B1[si - 1]
            p2 = pred_B2[si - 1]
            c0s = int(p0 == gt_id)
            c1s = int(p1 == gt_id)
            c2s = int(p2 == gt_id)
            step_bucket_correct["B0"][b] += c0s
            step_bucket_correct["B1"][b] += c1s
            step_bucket_correct["B2"][b] += c2s

            # Step-level row (debug)
            row = {
                "vehicle_id": vid,
                "step": int(si),
                "camera_id": step.get("camera_id"),
                "candidate_count": cand_count,
                "gt_track_id": gt_id,
                "pred_B0": p0,
                "pred_B1": p1,
                "pred_B2": p2,
                "correct_B0": c0s,
                "correct_B1": c1s,
                "correct_B2": c2s,
            }
            # Twins group-level correctness (if present)
            gt_g = t_by_id.get(gt_id, {}).get("twins_group")
            if gt_g is not None:
                row["gt_group"] = gt_g
                row["pred_group_B0"] = t_by_id.get(p0, {}).get("twins_group")
                row["pred_group_B1"] = t_by_id.get(p1, {}).get("twins_group")
                row["pred_group_B2"] = t_by_id.get(p2, {}).get("twins_group")
                row["correct_group_B0"] = int(row["pred_group_B0"] == gt_g)
                row["correct_group_B1"] = int(row["pred_group_B1"] == gt_g)
                row["correct_group_B2"] = int(row["pred_group_B2"] == gt_g)
            per_step_rows.append(row)

        # Per-case row
        avg_cand = float(sum(int(s.get("candidate_count") or len(s.get("candidates") or [])) for s in steps) / max(len(steps), 1))
        case_row = {
            "vehicle_id": vid,
            "twins_group": vrec.get("twins_group"),
            "chain_len": int(L),
            "avg_candidates": avg_cand,
            "step_acc_B0": s0,
            "step_acc_B1": s1,
            "step_acc_B2": s2,
            "transition_acc_B0": t0,
            "transition_acc_B1": t1,
            "transition_acc_B2": t2,
            "chain_exact_B0": c0,
            "chain_exact_B1": c1,
            "chain_exact_B2": c2,
            "b1_fallback_steps": int(fb),
        }
        if group_metrics is not None:
            case_row.update(group_metrics)
        per_case_rows.append(case_row)

    def avg(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    metrics = {
        "meta": {
            "generated_at": now_iso(),
            "tag": args.tag,
            "tracklets_json": os.path.abspath(args.tracklets_json),
            "tracklet_features_npy": os.path.abspath(args.tracklet_features_npy),
            "tracklet_index_json": os.path.abspath(args.tracklet_index_json),
            "candidates_json": os.path.abspath(args.candidates_json),
            "coview_json": os.path.abspath(args.coview_json),
            "speed_prior_json": os.path.abspath(args.speed_prior_json),
            "tracklets_meta": meta_track,
            "candidates_meta": meta_cand,
            "sigma_h_deg": float(args.sigma_h_deg),
            "w_vis": float(args.w_vis),
            "speed_prior": {
                "v_mean_kmh": prior.v_mean_kmh,
                "v_std_kmh": prior.v_std_kmh,
                "v_max_kmh": prior.v_max_kmh,
            },
            "b1_fallback_steps_total": int(b1_fallback_steps),
            "has_groups": bool(has_groups),
            "group_vehicle_count": int(group_vehicle_count),
            "group_skipped_missing_label": int(group_skipped_missing_label),
        },
        "counts": {
            "vehicles": int(len(step_acc_B0)),
        },
        "methods": {
            "B0": {
                "step_acc_macro": avg(step_acc_B0),
                "transition_acc_macro": avg(trans_acc_B0),
                "chain_exact": avg(chain_B0),
            },
            "B1": {
                "step_acc_macro": avg(step_acc_B1),
                "transition_acc_macro": avg(trans_acc_B1),
                "chain_exact": avg(chain_B1),
            },
            "B2": {
                "step_acc_macro": avg(step_acc_B2),
                "transition_acc_macro": avg(trans_acc_B2),
                "chain_exact": avg(chain_B2),
            },
        },
        "by_chain_len_step_acc": {
            str(L): {m: float(sum(xs) / max(len(xs), 1)) for m, xs in mm.items()} for L, mm in sorted(by_L.items())
        },
        "step_bucket_acc": {
            b: {
                "total": int(step_bucket_total[b]),
                "B0": float(step_bucket_correct["B0"][b] / max(step_bucket_total[b], 1)),
                "B1": float(step_bucket_correct["B1"][b] / max(step_bucket_total[b], 1)),
                "B2": float(step_bucket_correct["B2"][b] / max(step_bucket_total[b], 1)),
            }
            for b in buckets
        },
    }

    if group_vehicle_count > 0:
        metrics["methods_group"] = {
            "B0": {
                "step_acc_macro": avg(step_acc_g_B0),
                "transition_acc_macro": avg(trans_acc_g_B0),
                "chain_exact": avg(chain_g_B0),
            },
            "B1": {
                "step_acc_macro": avg(step_acc_g_B1),
                "transition_acc_macro": avg(trans_acc_g_B1),
                "chain_exact": avg(chain_g_B1),
            },
            "B2": {
                "step_acc_macro": avg(step_acc_g_B2),
                "transition_acc_macro": avg(trans_acc_g_B2),
                "chain_exact": avg(chain_g_B2),
            },
        }

    out_metrics = os.path.join(args.out_dir, f"metrics_{args.tag}.json")
    out_case = os.path.join(args.out_dir, f"per_case_{args.tag}.csv")
    out_step = os.path.join(args.out_dir, f"per_step_{args.tag}.csv")

    save_json(out_metrics, metrics)

    def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out_case, per_case_rows)
    write_csv(out_step, per_step_rows)

    print("=" * 70)
    print("TR Evaluation Completed")
    print("=" * 70)
    print(f"Tag:      {args.tag}")
    print(f"Vehicles: {metrics['counts']['vehicles']}")
    for m in ("B0", "B1", "B2"):
        mm = metrics["methods"][m]
        print(f"{m}: step={mm['step_acc_macro']*100:.2f}%  chain={mm['chain_exact']*100:.2f}%")
    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_case}")
    print(f"Saved: {out_step}")
    print("=" * 70)


if __name__ == "__main__":
    main()
