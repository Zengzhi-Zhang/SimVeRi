#!/usr/bin/env python3
"""
AG spatiotemporal ablation evaluator for the SimVeRi release.

This script intentionally lives outside E:\\SimVeRi_release for sandboxed runs.
It reads the released AG protocol metadata, evaluates the requested late-fusion
ablation modes, and reuses the CMC/AP metric semantics from
SimVeRi-code/simveri_validation/scripts/evaluate_ag_protocol.py.

Default self-test output location: %TEMP%/ag_ablation_selftest/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TrackletMap = Dict[str, dict]
TrajectoryMap = Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]


FUSION_MODES = {
    "visual-only": ("visual",),
    "ST-only": ("temporal", "spatial"),
    "visual+temporal": ("visual", "temporal"),
    "visual+world": ("visual", "spatial"),
    "visual+full": ("visual", "temporal", "spatial"),
}


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _coerce_tracklets_dict(obj: object) -> Dict[str, dict]:
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items() if isinstance(v, dict)}
    if isinstance(obj, list):
        out: Dict[str, dict] = {}
        for rec in obj:
            if isinstance(rec, dict) and rec.get("tracklet_id"):
                out[str(rec["tracklet_id"])] = rec
        return out
    return {}


def load_tracklets_map(tracklets_path: str) -> Tuple[TrackletMap, dict]:
    raw = _load_json(tracklets_path)
    if isinstance(raw, dict) and ("air_tracklets" in raw or "ground_tracklets" in raw):
        air = _coerce_tracklets_dict(raw.get("air_tracklets") or {})
        ground = _coerce_tracklets_dict(raw.get("ground_tracklets") or {})
        merged: TrackletMap = {}
        merged.update(air)
        merged.update(ground)
        meta = {k: v for k, v in raw.items() if k not in ("air_tracklets", "ground_tracklets")}
        meta.update({"air_tracklet_count": len(air), "ground_tracklet_count": len(ground), "tracklet_count": len(merged)})
        return merged, meta
    flat = _coerce_tracklets_dict(raw)
    return flat, {"tracklet_count": len(flat)}


def load_spatiotemporal_annotations(path: str) -> Dict[str, dict]:
    raw = _load_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("annotations"), dict):
        return raw["annotations"]
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, dict) and "timestamp" in v}
    raise ValueError(f"Unsupported spatiotemporal JSON format: {path}")


def build_trajectories(tracklets: TrackletMap, st_ann: Mapping[str, dict]) -> TrajectoryMap:
    trajectories: TrajectoryMap = {}
    for tid, tracklet in tracklets.items():
        samples: List[Tuple[float, float, float]] = []
        for image_name in tracklet.get("images") or []:
            ann = st_ann.get(image_name)
            if not ann:
                continue
            pos = ann.get("position") or {}
            if "timestamp" not in ann or "x" not in pos or "y" not in pos:
                continue
            samples.append((float(ann["timestamp"]), float(pos["x"]), float(pos["y"])))
        if not samples:
            continue
        samples.sort(key=lambda x: x[0])
        dedup: List[Tuple[float, float, float]] = []
        for sample in samples:
            if dedup and abs(sample[0] - dedup[-1][0]) < 1e-9:
                dedup[-1] = sample
            else:
                dedup.append(sample)
        arr = np.asarray(dedup, dtype=np.float64)
        trajectories[tid] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return trajectories


def perturb_trajectories(trajectories: TrajectoryMap, sigma_m: float, rng: np.random.Generator) -> TrajectoryMap:
    if sigma_m < 0:
        raise ValueError("noise sigma must be non-negative")
    perturbed: TrajectoryMap = {}
    for tid, (times, xs, ys) in trajectories.items():
        if sigma_m == 0:
            perturbed[tid] = (times.copy(), xs.copy(), ys.copy())
            continue
        noise = rng.normal(0.0, sigma_m, size=(len(xs), 2))
        perturbed[tid] = (times.copy(), xs + noise[:, 0], ys + noise[:, 1])
    return perturbed


def interval_gap(q: Mapping[str, object], g: Mapping[str, object]) -> float:
    start_q, end_q = float(q["start_time"]), float(q["end_time"])
    start_g, end_g = float(g["start_time"]), float(g["end_time"])
    return max(0.0, max(start_q, start_g) - min(end_q, end_g))


def temporal_score(q: Mapping[str, object], g: Mapping[str, object], sigma_t: float) -> float:
    if sigma_t <= 0:
        raise ValueError("sigma_t must be positive")
    return math.exp(-interval_gap(q, g) / sigma_t)


def mean_overlap_distance(
    q: Mapping[str, object],
    g: Mapping[str, object],
    q_traj: Tuple[np.ndarray, np.ndarray, np.ndarray],
    g_traj: Tuple[np.ndarray, np.ndarray, np.ndarray],
    step_s: float,
) -> float:
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    q_t, q_x, q_y = q_traj
    g_t, g_x, g_y = g_traj
    start = max(float(q["start_time"]), float(g["start_time"]), float(q_t[0]), float(g_t[0]))
    end = min(float(q["end_time"]), float(g["end_time"]), float(q_t[-1]), float(g_t[-1]))
    if end < start:
        return math.inf
    if end - start < 1e-9:
        grid = np.asarray([start], dtype=np.float64)
    else:
        grid = np.arange(start, end + min(step_s * 0.5, 1e-6), step_s, dtype=np.float64)
        if grid.size == 0 or grid[-1] < end:
            grid = np.append(grid, end)
    qxi = np.interp(grid, q_t, q_x)
    qyi = np.interp(grid, q_t, q_y)
    gxi = np.interp(grid, g_t, g_x)
    gyi = np.interp(grid, g_t, g_y)
    return float(np.mean(np.sqrt((qxi - gxi) ** 2 + (qyi - gyi) ** 2)))


def spatial_score(
    q: Mapping[str, object],
    g: Mapping[str, object],
    q_traj: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    g_traj: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sigma_d: float,
    step_s: float,
) -> float:
    if sigma_d <= 0:
        raise ValueError("sigma_d must be positive")
    if q_traj is None or g_traj is None:
        return 0.0
    dist = mean_overlap_distance(q, g, q_traj, g_traj, step_s)
    if not math.isfinite(dist):
        return 0.0
    return math.exp(-dist / sigma_d)


def _l2_normalize(feats: np.ndarray) -> np.ndarray:
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)


def load_features(features_dir: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not features_dir:
        return None
    feat_path = os.path.join(features_dir, "tracklet_features.npy")
    idx_path = os.path.join(features_dir, "tracklet_index.json")
    if not os.path.exists(feat_path) or not os.path.exists(idx_path):
        raise FileNotFoundError("Missing tracklet_features.npy or tracklet_index.json in --features")
    feats = np.load(feat_path)
    index = _load_json(idx_path)
    if feats.shape[0] != len(index):
        raise ValueError(f"Feature/index mismatch: feats={feats.shape[0]} index={len(index)}")
    feats = _l2_normalize(np.asarray(feats, dtype=np.float64))
    return {str(rec["tracklet_id"]): feats[i] for i, rec in enumerate(index)}


def minmax_per_query(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    row_min = np.min(values, axis=1, keepdims=True)
    row_max = np.max(values, axis=1, keepdims=True)
    denom = row_max - row_min
    np.divide(values - row_min, denom, out=out, where=denom > 1e-12)
    return out


def compute_component_matrices(
    q_tids: Sequence[str],
    g_tids: Sequence[str],
    tracklets: TrackletMap,
    trajectories: TrajectoryMap,
    features: Optional[Mapping[str, np.ndarray]],
    sigma_t: float,
    sigma_d: float,
    spatial_step_s: float,
) -> Dict[str, np.ndarray]:
    temporal = np.zeros((len(q_tids), len(g_tids)), dtype=np.float64)
    spatial = np.zeros_like(temporal)
    visual = None
    if features is not None:
        visual = np.full_like(temporal, np.nan)

    for i, q_tid in enumerate(q_tids):
        q = tracklets[q_tid]
        q_traj = trajectories.get(q_tid)
        for j, g_tid in enumerate(g_tids):
            g = tracklets[g_tid]
            temporal[i, j] = temporal_score(q, g, sigma_t)
            spatial[i, j] = spatial_score(q, g, q_traj, trajectories.get(g_tid), sigma_d, spatial_step_s)
            if visual is not None and q_tid in features and g_tid in features:
                cosine = float(np.dot(features[q_tid], features[g_tid]))
                visual[i, j] = (cosine + 1.0) / 2.0

    components = {"temporal": temporal, "spatial": spatial}
    if visual is not None:
        components["visual"] = visual
    return components


def fuse_scores(components: Mapping[str, np.ndarray], mode: str, normalize: bool = True) -> np.ndarray:
    if mode not in FUSION_MODES:
        raise KeyError(f"Unknown fusion mode: {mode}")
    names = FUSION_MODES[mode]
    missing = [name for name in names if name not in components]
    if missing:
        raise KeyError(f"Mode {mode} requires unavailable component(s): {', '.join(missing)}")
    mats = []
    for name in names:
        mat = np.asarray(components[name], dtype=np.float64)
        if np.isnan(mat).any():
            raise ValueError(f"Component {name} contains NaN; feature coverage is incomplete")
        mats.append(minmax_per_query(mat) if normalize else mat)
    return np.mean(np.stack(mats, axis=0), axis=0)


def _compute_ap(matches: List[bool]) -> float:
    if not any(matches):
        return 0.0
    num_rel = sum(matches)
    cum = 0
    ps = 0.0
    for i, match in enumerate(matches):
        if match:
            cum += 1
            ps += cum / (i + 1)
    return ps / num_rel


def _compute_cmc(ranks: List[int], max_rank: int = 50) -> np.ndarray:
    cmc = np.zeros(max_rank, dtype=np.float64)
    for rank in ranks:
        if 1 <= rank <= max_rank:
            cmc[rank - 1 :] += 1
    return cmc / max(1, len(ranks))


def evaluate_direction(sim: np.ndarray, q_tids: List[str], g_tids: List[str], positives: Dict[str, List[str]]) -> dict:
    g_count = len(g_tids)
    ranks: List[int] = []
    aps: List[float] = []
    pos_sets = {q: set(v) for q, v in positives.items()}
    for i, q_tid in enumerate(q_tids):
        pos = pos_sets.get(q_tid, set())
        if not pos:
            continue
        scores = sim[i]
        order = np.argsort(-scores)
        matches = [g_tids[int(j)] in pos for j in order]
        aps.append(_compute_ap(matches))
        ranks.append(matches.index(True) + 1 if any(matches) else (g_count + 1))
    cmc = _compute_cmc(ranks, max_rank=min(50, g_count)) * 100.0
    return {
        "Rank-1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "Rank-5": float(cmc[4]) if len(cmc) > 4 else float(cmc[-1]) if len(cmc) else 0.0,
        "Rank-10": float(cmc[9]) if len(cmc) > 9 else float(cmc[-1]) if len(cmc) else 0.0,
        "mAP": float(np.mean(aps) * 100.0) if aps else 0.0,
        "CMC": cmc.tolist(),
        "num_query": int(len(ranks)),
        "gallery_size": int(g_count),
    }


def positives_for_tau(q_tids: Sequence[str], g_tids: Sequence[str], tracklets: TrackletMap, tau_s: float) -> Dict[str, List[str]]:
    positives: Dict[str, List[str]] = {}
    for q_tid in q_tids:
        q = tracklets[q_tid]
        q_vehicle = q.get("vehicle_id")
        pos = []
        for g_tid in g_tids:
            g = tracklets[g_tid]
            if q_vehicle == g.get("vehicle_id") and interval_gap(q, g) <= tau_s:
                pos.append(g_tid)
        if pos:
            positives[q_tid] = pos
    return positives


def positives_for_vehicle(q_tids: Sequence[str], g_tids: Sequence[str], tracklets: TrackletMap) -> Dict[str, List[str]]:
    """Vehicle-level positives: any same-vehicle gallery tracklet, ignoring time.

    Supplemental diagnostic only -- NOT the co-temporal AG association protocol
    (Eq. 2). Includes co-observed same-vehicle gallery tracklets.
    """
    positives: Dict[str, List[str]] = {}
    for q_tid in q_tids:
        q = tracklets[q_tid]
        q_vehicle = q.get("vehicle_id")
        pos = []
        for g_tid in g_tids:
            g = tracklets[g_tid]
            if q_vehicle == g.get("vehicle_id"):
                pos.append(g_tid)
        if pos:
            positives[q_tid] = pos
    return positives


def cotemporal_distractor_separation(
    q_tids: Sequence[str],
    g_tids: Sequence[str],
    tracklets: TrackletMap,
    trajectories: TrajectoryMap,
    step_s: float,
    tau_s: float = 1.0,
) -> dict:
    """Anti-circularity statistic for the AG association protocol.

    For each query, among the gallery tracklets that are co-temporal with it
    (interval_gap <= tau_s, i.e. within tau seconds) -- the candidates that the
    temporal score alone cannot separate -- measure how far the true same-vehicle
    match sits from
    the nearest different-vehicle distractor in world coordinates. A positive
    margin shows the spatial signal (not the temporal label that defines the
    positives) is what resolves the association.
    """
    interpretation = (
        "Among co-temporal gallery candidates that the temporal score alone cannot separate, a "
        "positive margin means the true same-vehicle match is spatially closer than any finite "
        "different-vehicle distractor; the world coordinates, not the temporal label that defines "
        "the positives, resolve the association."
    )
    nearest_distractors: List[float] = []
    margins: List[float] = []

    for q_tid in q_tids:
        q = tracklets[q_tid]
        q_vehicle = q.get("vehicle_id")
        q_traj = trajectories.get(q_tid)
        if q_traj is None:
            continue

        true_dists: List[float] = []
        distractor_dists: List[float] = []
        for g_tid in g_tids:
            g = tracklets[g_tid]
            if interval_gap(q, g) > tau_s:
                continue
            g_traj = trajectories.get(g_tid)
            if g_traj is None:
                continue
            dist = mean_overlap_distance(q, g, q_traj, g_traj, step_s)
            if not math.isfinite(dist):
                continue
            if q_vehicle == g.get("vehicle_id"):
                true_dists.append(dist)
            else:
                distractor_dists.append(dist)

        if not true_dists or not distractor_dists:
            continue

        true_dist = min(true_dists)
        nearest_distractor = min(distractor_dists)
        nearest_distractors.append(float(nearest_distractor))
        margins.append(float(nearest_distractor - true_dist))

    if not nearest_distractors:
        return {
            "tau_s": float(tau_s),
            "n_queries": 0,
            "nearest_distractor_median_m": None,
            "nearest_distractor_p10_m": None,
            "nearest_distractor_min_m": None,
            "margin_median_m": None,
            "margin_min_m": None,
            "margin_gt_0_fraction": None,
            "interpretation": interpretation,
        }

    nearest_arr = np.asarray(nearest_distractors, dtype=float)
    margin_arr = np.asarray(margins, dtype=float)
    return {
        "tau_s": float(tau_s),
        "n_queries": int(len(nearest_distractors)),
        "nearest_distractor_median_m": float(np.median(nearest_arr)),
        "nearest_distractor_p10_m": float(np.percentile(nearest_arr, 10)),
        "nearest_distractor_min_m": float(np.min(nearest_arr)),
        "margin_median_m": float(np.median(margin_arr)),
        "margin_min_m": float(np.min(margin_arr)),
        "margin_gt_0_fraction": float(np.mean(margin_arr > 0.0)),
        "interpretation": interpretation,
    }


def positive_pair_count(positives: Mapping[str, Sequence[str]]) -> int:
    return int(sum(len(v) for v in positives.values()))


def _format_metrics(metrics: Mapping[str, object]) -> str:
    return (
        f"R1={float(metrics['Rank-1']):6.2f} "
        f"R5={float(metrics['Rank-5']):6.2f} "
        f"R10={float(metrics['Rank-10']):6.2f} "
        f"mAP={float(metrics['mAP']):6.2f} "
        f"Q={int(metrics['num_query'])} G={int(metrics['gallery_size'])}"
    )


def _summarize_noise_trials(rows: Sequence[Mapping[str, float]]) -> dict:
    maps = np.asarray([float(row["mAP"]) for row in rows], dtype=np.float64)
    rank1s = np.asarray([float(row["Rank-1"]) for row in rows], dtype=np.float64)
    return {
        "mAP_mean": float(np.mean(maps)),
        "mAP_std": float(np.std(maps)),
        "rank1_mean": float(np.mean(rank1s)),
        "rank1_std": float(np.std(rank1s)),
    }


def evaluate_noise_robustness(
    args: argparse.Namespace,
    protocol: Mapping[str, object],
    tracklets: TrackletMap,
    trajectories: TrajectoryMap,
) -> dict:
    seed_values = list(range(int(args.noise_seeds)))
    if not seed_values:
        raise ValueError("--noise-seeds must be positive")

    direction_map = {"A2G": "air2ground", "G2A": "ground2air"}
    out = {
        "mode": "ST-only",
        "noise_unit": "metres",
        "noise_model": "per-image iid Gaussian added independently to x and y for all air and ground trajectories",
        "seed_values": seed_values,
        "sigma_values": [float(sigma) for sigma in args.noise_std_list],
        "A2G": [],
        "G2A": [],
    }

    print("\nNoise robustness (ST-only)")
    for label, direction in direction_map.items():
        d = protocol.get(direction) or {}
        q_tids = [tid for tid in d.get("query_tracklets", []) if tid in tracklets]
        g_tids = [tid for tid in d.get("gallery_tracklets", []) if tid in tracklets]
        positives = {str(q): list(v) for q, v in (d.get("positives") or {}).items()}
        print(f"  [{label}] queries={len(q_tids)} gallery={len(g_tids)}")

        for sigma in args.noise_std_list:
            sigma = float(sigma)
            trials = []
            for seed in seed_values:
                rng = np.random.default_rng(seed)
                noisy_trajectories = perturb_trajectories(trajectories, sigma, rng)
                components = compute_component_matrices(
                    q_tids,
                    g_tids,
                    tracklets,
                    noisy_trajectories,
                    None,
                    args.sigma_t,
                    args.sigma_d,
                    args.spatial_step_s,
                )
                sim = fuse_scores(components, "ST-only", normalize=not args.no_normalize)
                metrics = evaluate_direction(sim, q_tids, g_tids, positives)
                trials.append({"seed": int(seed), "mAP": float(metrics["mAP"]), "Rank-1": float(metrics["Rank-1"])})

            summary = {"sigma": sigma, **_summarize_noise_trials(trials), "trials": trials}
            if sigma == 0 and (abs(summary["mAP_mean"] - 100.0) > 1e-9 or abs(summary["rank1_mean"] - 100.0) > 1e-9):
                raise AssertionError(
                    f"sigma=0 ST-only sanity failed for {label}: "
                    f"mAP={summary['mAP_mean']:.12f}, Rank-1={summary['rank1_mean']:.12f}"
                )
            out[label].append(summary)
            print(
                f"    sigma={sigma:g}m "
                f"mAP={summary['mAP_mean']:.2f}+/-{summary['mAP_std']:.2f} "
                f"R1={summary['rank1_mean']:.2f}+/-{summary['rank1_std']:.2f}"
            )
    return out


def run_unit_tests() -> None:
    q = {"start_time": 0.0, "end_time": 2.0}
    g_overlap = {"start_time": 1.0, "end_time": 3.0}
    g_gap = {"start_time": 4.0, "end_time": 5.0}
    assert abs(interval_gap(q, g_overlap) - 0.0) < 1e-12
    assert abs(interval_gap(q, g_gap) - 2.0) < 1e-12
    assert abs(temporal_score(q, g_overlap, 1.0) - 1.0) < 1e-12
    assert abs(temporal_score(q, g_gap, 1.0) - math.exp(-2.0)) < 1e-12

    q_traj = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    g_same = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    g_shift = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([0.0, 0.0, 0.0]))
    assert abs(mean_overlap_distance(q, g_overlap, q_traj, g_same, 0.1)) < 1e-12
    assert spatial_score(q, g_gap, q_traj, g_shift, 15.0, 0.1) == 0.0
    shifted_score = spatial_score(q, g_overlap, q_traj, g_shift, 15.0, 0.1)
    assert 0.0 < shifted_score < 1.0

    comps = {
        "temporal": np.array([[1.0, 0.0, 0.5], [2.0, 2.0, 2.0]]),
        "spatial": np.array([[0.0, 1.0, 0.5], [3.0, 4.0, 5.0]]),
        "visual": np.array([[0.2, 0.8, 0.5], [0.1, 0.1, 0.1]]),
    }
    st = fuse_scores(comps, "ST-only")
    assert np.allclose(st[0], np.array([0.5, 0.5, 0.5]))
    assert np.allclose(st[1], np.array([0.0, 0.25, 0.5]))
    vf = fuse_scores(comps, "visual+full")
    assert vf.shape == (2, 3)


def evaluate_release(args: argparse.Namespace) -> dict:
    protocol_path = os.path.join(args.ag_root, "metadata", "protocol.json")
    tracklets_path = os.path.join(args.ag_root, "metadata", "tracklets.json")
    st_path = os.path.join(args.ag_root, "metadata", "ag_spatiotemporal.json")
    protocol = _load_json(protocol_path)
    tracklets, tracklets_meta = load_tracklets_map(tracklets_path)
    trajectories = build_trajectories(tracklets, load_spatiotemporal_annotations(st_path))
    features = load_features(args.features)

    results = {
        "evaluation_time": datetime.now().isoformat(),
        "ag_root": os.path.abspath(args.ag_root),
        "features": os.path.abspath(args.features) if args.features else None,
        "visual_modes_status": "enabled" if features is not None else "skipped: no --features supplied",
        "parameters": {
            "sigma_t": args.sigma_t,
            "sigma_d": args.sigma_d,
            "spatial_step_s": args.spatial_step_s,
            "normalize_per_query": not args.no_normalize,
            "tau_values": args.tau,
            "negative_definition": "For each query, every gallery tracklet not in that query's positive set is a negative.",
        },
        "tracklets_meta": tracklets_meta,
        "trajectory_count": len(trajectories),
        "directions": {},
        "vehicle_level": {},
        "cotemporal_distractor_separation": {},
        "tau_sensitivity": {},
    }

    if args.noise_robustness:
        results["noise_robustness"] = evaluate_noise_robustness(args, protocol, tracklets, trajectories)

    print("AG ablation evaluation")
    print(f"AG root: {args.ag_root}")
    print(f"Output:  {args.output_dir}")
    print(f"Visual:  {'enabled' if features is not None else 'skipped (no --features)'}")

    for direction in ("air2ground", "ground2air"):
        d = protocol.get(direction) or {}
        q_tids = [tid for tid in d.get("query_tracklets", []) if tid in tracklets]
        g_tids = [tid for tid in d.get("gallery_tracklets", []) if tid in tracklets]
        positives = {str(q): list(v) for q, v in (d.get("positives") or {}).items()}
        if features is not None:
            q_tids = [tid for tid in q_tids if tid in features]
            g_tids = [tid for tid in g_tids if tid in features]
        components = compute_component_matrices(
            q_tids, g_tids, tracklets, trajectories, features, args.sigma_t, args.sigma_d, args.spatial_step_s
        )
        direction_results = {}
        print(f"\n[{direction}] queries={len(q_tids)} gallery={len(g_tids)}")
        for mode in FUSION_MODES:
            try:
                sim = fuse_scores(components, mode, normalize=not args.no_normalize)
            except KeyError as exc:
                direction_results[mode] = {"status": f"skipped: {exc}"}
                print(f"  {mode:16s} skipped ({exc})")
                continue
            metrics = evaluate_direction(sim, q_tids, g_tids, positives)
            direction_results[mode] = metrics
            print(f"  {mode:16s} {_format_metrics(metrics)}")
        results["directions"][direction] = direction_results

        # Vehicle-level diagnostic (supplemental; NOT the co-temporal protocol):
        # positives = any same-vehicle gallery tracklet, time ignored. Built AFTER
        # feature filtering so galleries without visual features are never positives.
        vehicle_positives = positives_for_vehicle(q_tids, g_tids, tracklets)
        vehicle_results = {
            "definition": "Vehicle-level retrieval diagnostic: any same-vehicle gallery tracklet is positive, time ignored; includes co-observed same-vehicle gallery tracklets. Not the Eq. 2 co-temporal association protocol.",
            "positive_pairs": positive_pair_count(vehicle_positives),
            "queries_with_positives": len(vehicle_positives),
        }
        print(f"  [veh] positives pairs={vehicle_results['positive_pairs']} queries={vehicle_results['queries_with_positives']}")
        for mode in FUSION_MODES:
            try:
                sim = fuse_scores(components, mode, normalize=not args.no_normalize)
            except KeyError as exc:
                vehicle_results[mode] = {"status": f"skipped: {exc}"}
                print(f"  [veh] {mode:16s} skipped ({exc})")
                continue
            metrics = evaluate_direction(sim, q_tids, g_tids, vehicle_positives)
            vehicle_results[mode] = metrics
            print(f"  [veh] {mode:16s} {_format_metrics(metrics)}")
        results["vehicle_level"][direction] = vehicle_results

        # Anti-circularity: among co-temporal candidates (timing alone cannot
        # separate them), how far is the true match from the nearest distractor?
        sep = cotemporal_distractor_separation(
            q_tids, g_tids, tracklets, trajectories, args.spatial_step_s, tau_s=1.0
        )
        results["cotemporal_distractor_separation"][direction] = sep
        if sep["n_queries"]:
            print(
                f"  [sep] n={sep['n_queries']} nearest_distractor"
                f" median={sep['nearest_distractor_median_m']:.2f}m"
                f" p10={sep['nearest_distractor_p10_m']:.2f}m"
                f" min={sep['nearest_distractor_min_m']:.2f}m"
                f" margin>0={sep['margin_gt_0_fraction']*100:.1f}%"
            )

        tau_rows = []
        st_sim = fuse_scores(components, "ST-only", normalize=not args.no_normalize)
        visual_sim = None
        if features is not None:
            visual_sim = fuse_scores(components, "visual-only", normalize=not args.no_normalize)
        print(f"  tau sensitivity ({direction})")
        for tau in args.tau:
            tau_pos = positives_for_tau(q_tids, g_tids, tracklets, float(tau))
            row = {"tau_s": float(tau), "positive_pairs": positive_pair_count(tau_pos), "queries_with_positives": len(tau_pos)}
            row["ST-only_mAP"] = evaluate_direction(st_sim, q_tids, g_tids, tau_pos)["mAP"]
            if visual_sim is not None:
                row["visual-only_mAP"] = evaluate_direction(visual_sim, q_tids, g_tids, tau_pos)["mAP"]
            tau_rows.append(row)
            visual_txt = f" visual_mAP={row['visual-only_mAP']:.2f}" if "visual-only_mAP" in row else ""
            print(
                f"    tau={float(tau):.1f}s pairs={row['positive_pairs']} "
                f"queries={row['queries_with_positives']} ST_mAP={row['ST-only_mAP']:.2f}{visual_txt}"
            )
        results["tau_sensitivity"][direction] = tau_rows
    return results


def parse_args() -> argparse.Namespace:
    default_ag_root = r"E:\SimVeRi_release\SimVeRi-dataset-v2.0\annotations_and_protocols\ag_protocol"
    default_out = os.path.join(tempfile.gettempdir(), "ag_ablation_selftest")
    parser = argparse.ArgumentParser(description="Evaluate AG visual/ST ablations on the released SimVeRi protocol")
    parser.add_argument("--ag-root", default=default_ag_root, help="Path to annotations_and_protocols/ag_protocol")
    parser.add_argument("--features", default=None, help="Optional dir with tracklet_features.npy and tracklet_index.json")
    parser.add_argument("--output-dir", default=default_out, help="Output directory; default is %%TEMP%%/ag_ablation_selftest")
    parser.add_argument("--sigma-t", type=float, default=1.0, help="Temporal exponential scale in seconds")
    parser.add_argument("--sigma-d", type=float, default=15.0, help="Spatial exponential scale in metres")
    parser.add_argument("--spatial-step-s", type=float, default=0.1, help="Common interpolation step in seconds")
    parser.add_argument("--tau", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0], help="Tau values for sensitivity")
    parser.add_argument("--no-normalize", action="store_true", help="Disable per-query min-max normalization before fusion")
    parser.add_argument("--noise-robustness", action="store_true", help="Run ST-only coordinate-noise robustness evaluation")
    parser.add_argument(
        "--noise-std-list",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Coordinate noise sigma values in metres for --noise-robustness",
    )
    parser.add_argument("--noise-seeds", type=int, default=10, help="Number of deterministic noise seeds: 0..N-1")
    parser.add_argument(
        "--noise-metric",
        nargs="+",
        choices=["mAP", "Rank-1"],
        default=["mAP", "Rank-1"],
        help="Accepted for compatibility; noise robustness always reports both mAP and Rank-1",
    )
    parser.add_argument("--run-unit-tests", action="store_true", help="Run synthetic scoring/fusion tests before evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Only run unit tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_unit_tests:
        run_unit_tests()
        print("Unit tests passed")
    if args.skip_eval:
        return
    _mkdir(args.output_dir)
    results = evaluate_release(args)
    out_json = os.path.join(args.output_dir, "ag_ablation_results.json")
    _write_json(out_json, results)
    print(f"\nSaved JSON: {out_json}")


if __name__ == "__main__":
    main()
