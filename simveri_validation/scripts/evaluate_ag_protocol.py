# scripts/evaluate_ag_protocol.py
"""
Evaluate Air-Ground protocol (tracklet-level retrieval) and generate paper-ready figures.

Inputs:
  --ag-root:
    - metadata/protocol.json
    - metadata/tracklets.json
    - images/air/*.jpg
    - images/ground/*.jpg
  --features-dir:
    - tracklet_features.npy
    - tracklet_index.json
    - features_meta.json (optional)

Outputs (under --output-dir):
  - ag_evaluation.json
  - figures:
      ag_cmc_air2ground.(png|pdf)
      ag_cmc_ground2air.(png|pdf)
      ag_similarity_distributions_air2ground.(png|pdf)
      ag_similarity_distributions_ground2air.(png|pdf)
  - retrieval_examples/<direction>/(correct|wrong)/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# Headless backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_tracklets_dict(obj: object) -> Dict[str, dict]:
    """
    Support multiple tracklets.json formats:
      - dict: {tracklet_id: tracklet_dict}
      - list: [tracklet_dict, ...] (each has tracklet_id)
    """
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, dict)}
    if isinstance(obj, list):
        out: Dict[str, dict] = {}
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            tid = rec.get("tracklet_id")
            if isinstance(tid, str) and tid:
                out[tid] = rec
        return out
    return {}


def _load_tracklets_map(tracklets_path: str) -> tuple[Dict[str, dict], dict]:
    """
    Flatten the release-generated tracklets.json structure into {tracklet_id: tracklet_dict}.

    Newer releases store:
      {
        ...meta,
        "air_tracklets": {tid: {...}},
        "ground_tracklets": {tid: {...}}
      }
    """
    raw = _load_json(tracklets_path)
    if isinstance(raw, dict) and ("air_tracklets" in raw or "ground_tracklets" in raw):
        air = _coerce_tracklets_dict(raw.get("air_tracklets") or {})
        ground = _coerce_tracklets_dict(raw.get("ground_tracklets") or {})
        merged: Dict[str, dict] = {}
        merged.update(air)
        merged.update(ground)
        meta = {k: v for k, v in raw.items() if k not in ("air_tracklets", "ground_tracklets")}
        meta["air_tracklet_count"] = len(air)
        meta["ground_tracklet_count"] = len(ground)
        meta["tracklet_count"] = len(merged)
        return merged, meta

    flat = _coerce_tracklets_dict(raw)
    return flat, {"tracklet_count": len(flat)}


def _save_fig(fig: plt.Figure, out_base: str, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _cosine_sim_matrix(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    return np.dot(qn, gn.T)


def _compute_ap(matches: List[bool]) -> float:
    if not any(matches):
        return 0.0
    num_rel = sum(matches)
    cum = 0
    ps = 0.0
    for i, m in enumerate(matches):
        if m:
            cum += 1
            ps += cum / (i + 1)
    return ps / num_rel


def _compute_cmc(ranks: List[int], max_rank: int = 50) -> np.ndarray:
    cmc = np.zeros(max_rank, dtype=np.float64)
    for r in ranks:
        if 1 <= r <= max_rank:
            cmc[r - 1 :] += 1
    cmc = cmc / max(1, len(ranks))
    return cmc


def _infer_layer(tracklet_id: str, tracklet: dict) -> str:
    layer = str(tracklet.get("layer", "")).lower().strip()
    if layer in ("air", "ground"):
        return layer
    if tracklet_id.startswith("air_"):
        return "air"
    return "ground"


def _tracklet_rep_image_path(ag_root: str, tracklets: Dict[str, dict], tid: str) -> str:
    t = tracklets[tid]
    layer = _infer_layer(tid, t)
    imgs = t.get("images") or []
    if not imgs:
        return ""
    # Representative: first frame (deterministic).
    return os.path.join(ag_root, "images", layer, imgs[0])


def _thumb(img_path: str, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    # Pillow compatibility: use Resampling if available, else legacy constant.
    Resampling = getattr(Image, "Resampling", None)
    resample = Resampling.BICUBIC if Resampling is not None else Image.BICUBIC
    img = img.copy()
    img.thumbnail(size, resample=resample)
    return img


def _tile_with_border(img: Image.Image, size: Tuple[int, int], border: int, color: Tuple[int, int, int]) -> Image.Image:
    tile = Image.new("RGB", (size[0] + 2 * border, size[1] + 2 * border), color=color)
    x = border + (size[0] - img.size[0]) // 2
    y = border + (size[1] - img.size[1]) // 2
    tile.paste(img, (x, y))
    return tile


def _save_retrieval_examples(
    ag_root: str,
    direction: str,
    sim: np.ndarray,
    q_tids: List[str],
    g_tids: List[str],
    positives: Dict[str, List[str]],
    tracklets: Dict[str, dict],
    out_dir: str,
    topk: int,
    num_correct: int,
    num_wrong: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    g_index = {tid: j for j, tid in enumerate(g_tids)}

    order = np.argsort(-sim, axis=1)[:, : max(1, topk)]

    correct, wrong = [], []
    for i, q in enumerate(q_tids):
        pos = set(positives.get(q, []))
        top1_tid = g_tids[int(order[i, 0])]
        if top1_tid in pos:
            correct.append(i)
        else:
            wrong.append(i)

    rng.shuffle(correct)
    rng.shuffle(wrong)
    correct = correct[: max(0, int(num_correct))]
    wrong = wrong[: max(0, int(num_wrong))]

    tile = (200, 200)
    border = 6
    green = (46, 160, 67)
    red = (214, 39, 40)
    gray = (180, 180, 180)

    def make_one(i: int, out_path: str) -> None:
        q_tid = q_tids[i]
        q_path = _tracklet_rep_image_path(ag_root, tracklets, q_tid)
        if not q_path or not os.path.exists(q_path):
            return
        q_img = _tile_with_border(_thumb(q_path, tile), tile, border, gray)

        row = [q_img]
        pos = set(positives.get(q_tid, []))
        for gi in order[i, :topk]:
            g_tid = g_tids[int(gi)]
            g_path = _tracklet_rep_image_path(ag_root, tracklets, g_tid)
            if not g_path or not os.path.exists(g_path):
                continue
            g_img = _thumb(g_path, tile)
            row.append(_tile_with_border(g_img, tile, border, green if g_tid in pos else red))

        w = sum(im.size[0] for im in row)
        h = max(im.size[1] for im in row) + 40
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        title = f"{direction} | query={q_tid} | top1={'OK' if g_tids[int(order[i,0])] in pos else 'WRONG'}"
        draw.text((10, 10), title, fill=(0, 0, 0))
        x = 0
        for im in row:
            canvas.paste(im, (x, 40))
            x += im.size[0]
        canvas.save(out_path)

    base = os.path.join(out_dir, "retrieval_examples", direction)
    out_c = os.path.join(base, "correct")
    out_w = os.path.join(base, "wrong")
    _mkdir(out_c)
    _mkdir(out_w)

    for i in correct:
        make_one(i, os.path.join(out_c, f"{q_tids[i]}_top{topk}.png"))
    for i in wrong:
        make_one(i, os.path.join(out_w, f"{q_tids[i]}_top{topk}.png"))


def _evaluate_direction(sim: np.ndarray, q_tids: List[str], g_tids: List[str], positives: Dict[str, List[str]]) -> dict:
    g_count = len(g_tids)
    ranks: List[int] = []
    aps: List[float] = []

    # Precompute pos sets
    pos_sets = {q: set(v) for q, v in positives.items()}

    for i, q in enumerate(q_tids):
        pos = pos_sets.get(q, set())
        if not pos:
            continue
        scores = sim[i]
        order = np.argsort(-scores)
        matches = [g_tids[int(j)] in pos for j in order]
        aps.append(_compute_ap(matches))
        ranks.append(matches.index(True) + 1 if any(matches) else (g_count + 1))

    cmc = _compute_cmc(ranks, max_rank=min(50, g_count)) * 100.0
    mAP = float(np.mean(aps) * 100.0) if aps else 0.0

    return {
        "Rank-1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "Rank-5": float(cmc[4]) if len(cmc) > 4 else float(cmc[-1]) if len(cmc) else 0.0,
        "Rank-10": float(cmc[9]) if len(cmc) > 9 else float(cmc[-1]) if len(cmc) else 0.0,
        "mAP": mAP,
        "CMC": cmc.tolist(),
        "num_query": int(len(ranks)),
        "gallery_size": int(g_count),
    }


def _plot_cmc(results: dict, title: str, out_base: str) -> None:
    cmc = results.get("CMC") or []
    if not cmc:
        return
    xs = np.arange(1, len(cmc) + 1)
    ys = np.asarray(cmc, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#4C78A8", linewidth=2)
    ax.set_xlim(1, len(cmc))
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate (%)")
    ax.grid(True, alpha=0.25)
    _save_fig(fig, out_base)


def _plot_similarity_hist(
    sim: np.ndarray,
    q_tids: List[str],
    g_tids: List[str],
    positives: Dict[str, List[str]],
    out_base: str,
    title: str,
    max_neg_samples: int = 200_000,
    seed: int = 0,
) -> None:
    rng = np.random.RandomState(seed)
    g_index = {tid: j for j, tid in enumerate(g_tids)}

    pos_vals = []
    neg_vals = []
    for i, q in enumerate(q_tids):
        pos = positives.get(q, [])
        if not pos:
            continue
        pos_idx = [g_index[p] for p in pos if p in g_index]
        if pos_idx:
            pos_vals.append(sim[i, pos_idx])
        # sample a small number of negatives per query (for speed)
        neg_pool = sim.shape[1] - len(pos_idx)
        if neg_pool <= 0:
            continue
        k = min(200, sim.shape[1])
        neg_idx = rng.choice(sim.shape[1], size=k, replace=False)
        # remove positives from sample
        neg_idx = [j for j in neg_idx if j not in set(pos_idx)]
        if neg_idx:
            neg_vals.append(sim[i, neg_idx])

    if not pos_vals or not neg_vals:
        return
    pos_all = np.concatenate(pos_vals, axis=0)
    neg_all = np.concatenate(neg_vals, axis=0)
    if neg_all.size > max_neg_samples:
        idx = rng.choice(neg_all.size, size=max_neg_samples, replace=False)
        neg_all = neg_all[idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(neg_all, bins=80, density=True, alpha=0.6, color="#999999", label="Negative")
    ax.hist(pos_all, bins=80, density=True, alpha=0.6, color="#4C78A8", label="Positive")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    _save_fig(fig, out_base)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Air-Ground protocol")
    p.add_argument("--ag-root", type=str, required=True, help="Path to ag_protocol or ag_protocol_full folder")
    p.add_argument("--features-dir", type=str, required=True, help="Output folder from extract_ag_tracklet_features.py")
    p.add_argument("--output-dir", type=str, required=True, help="Where to save evaluation JSON + figures")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--num-correct", type=int, default=12)
    p.add_argument("--num-wrong", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _mkdir(args.output_dir)

    protocol_path = os.path.join(args.ag_root, "metadata", "protocol.json")
    tracklets_path = os.path.join(args.ag_root, "metadata", "tracklets.json")
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"protocol.json not found: {protocol_path}")
    if not os.path.exists(tracklets_path):
        raise FileNotFoundError(f"tracklets.json not found: {tracklets_path}")

    protocol = _load_json(protocol_path)
    tracklets, _tracklets_meta = _load_tracklets_map(tracklets_path)

    feat_path = os.path.join(args.features_dir, "tracklet_features.npy")
    idx_path = os.path.join(args.features_dir, "tracklet_index.json")
    if not os.path.exists(feat_path) or not os.path.exists(idx_path):
        raise FileNotFoundError("Missing tracklet_features.npy or tracklet_index.json in --features-dir")

    feats = np.load(feat_path)
    index = _load_json(idx_path)
    if feats.shape[0] != len(index):
        raise ValueError(f"Feature/index mismatch: feats={feats.shape[0]} index={len(index)}")

    tid_to_feat = {rec["tracklet_id"]: feats[i] for i, rec in enumerate(index)}

    results = {
        "evaluation_time": datetime.now().isoformat(),
        "ag_root": os.path.abspath(args.ag_root),
        "features_dir": os.path.abspath(args.features_dir),
        "protocol_meta": {
            "scope": protocol.get("scope"),
            "base_vehicle_count": protocol.get("base_vehicle_count"),
            "protocol_vehicle_count": protocol.get("protocol_vehicle_count"),
            "vehicles_with_both": protocol.get("vehicles_with_both"),
            "time_tolerance_s": protocol.get("time_tolerance_s"),
        },
        "air2ground": None,
        "ground2air": None,
    }

    for direction_key in ("air2ground", "ground2air"):
        d = protocol.get(direction_key) or {}
        q_tids = list(d.get("query_tracklets") or [])
        g_tids = list(d.get("gallery_tracklets") or [])
        positives = d.get("positives") or {}

        # Filter to tracklets we have features for
        q_tids = [t for t in q_tids if t in tid_to_feat]
        g_tids = [t for t in g_tids if t in tid_to_feat]

        if not q_tids or not g_tids:
            results[direction_key] = {"error": "Empty query/gallery after filtering"}
            continue

        q_feat = np.stack([tid_to_feat[t] for t in q_tids], axis=0)
        g_feat = np.stack([tid_to_feat[t] for t in g_tids], axis=0)
        sim = _cosine_sim_matrix(q_feat, g_feat)

        res = _evaluate_direction(sim, q_tids, g_tids, positives)
        results[direction_key] = res

        # Figures
        _plot_cmc(res, f"AG CMC ({direction_key})", os.path.join(args.output_dir, f"ag_cmc_{direction_key}"))
        _plot_similarity_hist(
            sim,
            q_tids,
            g_tids,
            positives,
            os.path.join(args.output_dir, f"ag_similarity_distributions_{direction_key}"),
            title=f"AG Similarity Distributions ({direction_key})",
            seed=int(args.seed),
        )
        _save_retrieval_examples(
            ag_root=args.ag_root,
            direction=direction_key,
            sim=sim,
            q_tids=q_tids,
            g_tids=g_tids,
            positives=positives,
            tracklets=tracklets,
            out_dir=args.output_dir,
            topk=int(args.topk),
            num_correct=int(args.num_correct),
            num_wrong=int(args.num_wrong),
            seed=int(args.seed),
        )

    out_json = os.path.join(args.output_dir, "ag_evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Air-Ground protocol evaluation completed")
    print(f"AG root:     {args.ag_root}")
    print(f"Features:    {args.features_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Saved:       {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
