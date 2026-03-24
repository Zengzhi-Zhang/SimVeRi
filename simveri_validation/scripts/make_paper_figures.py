# scripts/make_paper_figures.py
"""
Generate paper-ready figures for SimVeRi experiments.

This script is intentionally dependency-light (numpy/matplotlib/Pillow) and uses
existing artifacts:
  - Dataset metadata:   <data-root>/statistics/dataset_summary.json
                        <data-root>/metadata/camera_network.json
  - Features (optional): <features-dir>/{gallery,query}_features.npy + *_info.json
  - Evaluation (optional): <results-json> (baseline_evaluation.json)

Outputs:
  - <output-dir>/*.png and *.pdf (vector) figures
  - <output-dir>/retrieval_examples/(correct|wrong)/*.png (if features provided)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# Headless backend (safe on servers / CI)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_fig(fig: plt.Figure, out_base: str, dpi: int = 300) -> None:
    """Save both PNG (raster) and PDF (vector) versions."""
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _safe_load_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_cam_ids(cameras: Dict[str, dict]) -> List[str]:
    def cam_key(cam_id: str) -> Tuple[int, str]:
        # "c001" -> 1; fall back to lexicographic
        try:
            return (int(cam_id.lstrip("c")), cam_id)
        except Exception:
            return (10**9, cam_id)

    return sorted(cameras.keys(), key=cam_key)


def plot_dataset_distributions(dataset_summary: dict, out_dir: str) -> None:
    dist = dataset_summary.get("distributions") or {}
    if not dist:
        return

    # Vehicle type distribution
    vtype = dist.get("type")
    if isinstance(vtype, dict) and vtype:
        items = sorted(vtype.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#4C78A8")
        ax.set_title("Vehicle Type Distribution")
        ax.set_ylabel("#Images")
        ax.grid(axis="y", alpha=0.25)
        _save_fig(fig, os.path.join(out_dir, "dataset_vehicle_type_distribution"))

    # Color family distribution
    cf = dist.get("color_family")
    if isinstance(cf, dict) and cf:
        items = sorted(cf.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, values, color="#F58518")
        ax.set_title("Color Family Distribution")
        ax.set_ylabel("#Images")
        ax.grid(axis="y", alpha=0.25)
        _save_fig(fig, os.path.join(out_dir, "dataset_color_family_distribution"))

    # Cross-camera distribution (#cameras per vehicle)
    cc = dist.get("cross_camera")
    if isinstance(cc, dict) and cc:
        # keys are strings like "2","3",...
        items = sorted(((int(k), v) for k, v in cc.items()), key=lambda kv: kv[0])
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#54A24B")
        ax.set_title("Cross-Camera Coverage (Vehicles)")
        ax.set_xlabel("#Cameras observed per vehicle")
        ax.set_ylabel("#Vehicles")
        ax.grid(axis="y", alpha=0.25)
        _save_fig(fig, os.path.join(out_dir, "dataset_cross_camera_distribution"))


def plot_camera_layout_and_distances(camera_network: dict, out_dir: str) -> None:
    cameras = camera_network.get("cameras") or {}
    dist_mat = camera_network.get("distance_matrix") or {}
    if not cameras:
        return

    cam_ids = _sorted_cam_ids(cameras)

    # Camera layout (top-down)
    xs, ys, zs = [], [], []
    layers = []
    for cid in cam_ids:
        info = cameras[cid]
        pos = info.get("position") or {}
        xs.append(float(pos.get("x", 0.0)))
        ys.append(float(pos.get("y", 0.0)))
        zs.append(float(pos.get("z", 0.0)))
        layers.append(str(info.get("layer", "unknown")))

    layer_set = sorted(set(layers))
    palette = {
        "ground": "#4C78A8",
        "air": "#F58518",
        "unknown": "#999999",
    }
    colors = [palette.get(l, "#999999") for l in layers]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, c=colors, s=60, edgecolors="white", linewidths=0.8, zorder=3)
    for cid, x, y in zip(cam_ids, xs, ys):
        ax.text(x, y, cid, fontsize=8, ha="left", va="bottom", alpha=0.9)
    ax.set_title("Camera Layout (Top-Down)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    # legend
    handles = []
    for l in layer_set:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=l, markerfacecolor=palette.get(l, "#999999"), markersize=8))
    if handles:
        ax.legend(handles=handles, loc="best", frameon=True)
    _save_fig(fig, os.path.join(out_dir, "camera_layout_topdown"))

    # Distance heatmap
    if dist_mat:
        n = len(cam_ids)
        mat = np.zeros((n, n), dtype=np.float32)
        for i, a in enumerate(cam_ids):
            row = dist_mat.get(a) or {}
            for j, b in enumerate(cam_ids):
                try:
                    mat[i, j] = float(row.get(b, 0.0))
                except Exception:
                    mat[i, j] = 0.0

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title("Camera Distance Matrix (Route-based, meters)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(cam_ids, rotation=90, fontsize=7)
        ax.set_yticklabels(cam_ids, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Distance (m)")
        _save_fig(fig, os.path.join(out_dir, "camera_distance_heatmap"))


def plot_cmc_curve(results_json: dict, out_dir: str) -> None:
    if not results_json:
        return
    full = results_json.get("full") or {}
    cmc = full.get("CMC")
    if not isinstance(cmc, list) or not cmc:
        return

    xs = np.arange(1, len(cmc) + 1)
    ys = np.asarray(cmc, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#4C78A8", linewidth=2)
    ax.set_xlim(1, len(cmc))
    ax.set_ylim(0, 100)
    ax.set_title("CMC Curve (Full)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate (%)")
    ax.grid(True, alpha=0.25)
    _save_fig(fig, os.path.join(out_dir, "reid_cmc_curve_full"))


def _cosine_sim_matrix(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    return np.dot(qn, gn.T)


def plot_similarity_distributions(
    q_feat: np.ndarray,
    g_feat: np.ndarray,
    q_info: List[dict],
    g_info: List[dict],
    out_dir: str,
    max_neg_samples: int = 250_000,
    seed: int = 0,
) -> None:
    rng = np.random.RandomState(seed)
    sim = _cosine_sim_matrix(q_feat, g_feat)

    g_vid = np.array([x["vehicle_id"] for x in g_info])
    q_vid = np.array([x["vehicle_id"] for x in q_info])

    # Collect positives and a sampled subset of negatives.
    pos_sims: List[np.ndarray] = []
    neg_sims: List[np.ndarray] = []
    for i in range(sim.shape[0]):
        pos_mask = (g_vid == q_vid[i])
        pos = sim[i, pos_mask]
        neg = sim[i, ~pos_mask]
        if pos.size:
            pos_sims.append(pos)
        if neg.size:
            neg_sims.append(neg)

    pos_all = np.concatenate(pos_sims, axis=0) if pos_sims else np.array([], dtype=np.float32)
    neg_all = np.concatenate(neg_sims, axis=0) if neg_sims else np.array([], dtype=np.float32)

    if neg_all.size > max_neg_samples:
        idx = rng.choice(neg_all.size, size=max_neg_samples, replace=False)
        neg_all = neg_all[idx]

    if pos_all.size == 0 or neg_all.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 80
    ax.hist(neg_all, bins=bins, density=True, alpha=0.6, color="#999999", label="Negative (diff ID)")
    ax.hist(pos_all, bins=bins, density=True, alpha=0.6, color="#4C78A8", label="Positive (same ID)")
    ax.set_title("Similarity Distributions (Cosine)")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    _save_fig(fig, os.path.join(out_dir, "reid_similarity_distributions"))


def _load_features_bundle(features_dir: str) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict]]:
    q_feat = np.load(os.path.join(features_dir, "query_features.npy"))
    g_feat = np.load(os.path.join(features_dir, "gallery_features.npy"))
    with open(os.path.join(features_dir, "query_info.json"), "r", encoding="utf-8") as f:
        q_info = json.load(f)
    with open(os.path.join(features_dir, "gallery_info.json"), "r", encoding="utf-8") as f:
        g_info = json.load(f)
    return q_feat, g_feat, q_info, g_info


def _thumb(img_path: str, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    # Fit into tile with preserved aspect ratio (compatible with older Pillow versions).
    # Avoid accessing deprecated Image.BICUBIC on newer Pillow; only use it on older versions
    # where Image.Resampling does not exist.
    Resampling = getattr(Image, "Resampling", None)
    if Resampling is not None:
        resample = Resampling.BICUBIC
    else:
        resample = Image.BICUBIC
    img = img.copy()
    img.thumbnail(size, resample=resample)
    return img


def _tile_with_border(img: Image.Image, size: Tuple[int, int], border: int, color: Tuple[int, int, int]) -> Image.Image:
    tile = Image.new("RGB", (size[0] + 2 * border, size[1] + 2 * border), color=color)
    # center paste
    x = border + (size[0] - img.size[0]) // 2
    y = border + (size[1] - img.size[1]) // 2
    tile.paste(img, (x, y))
    return tile


def save_retrieval_examples(
    data_root: str,
    q_feat: np.ndarray,
    g_feat: np.ndarray,
    q_info: List[dict],
    g_info: List[dict],
    out_dir: str,
    topk: int = 10,
    num_correct: int = 12,
    num_wrong: int = 12,
    seed: int = 0,
) -> None:
    sim = _cosine_sim_matrix(q_feat, g_feat)
    g_vid = np.array([x["vehicle_id"] for x in g_info])
    q_vid = np.array([x["vehicle_id"] for x in q_info])

    # Rank top-k for each query
    topk = int(topk)
    order = np.argsort(-sim, axis=1)[:, : max(topk, 1)]

    correct_idx: List[int] = []
    wrong_idx: List[int] = []
    for i in range(order.shape[0]):
        top1 = order[i, 0]
        if g_vid[top1] == q_vid[i]:
            correct_idx.append(i)
        else:
            wrong_idx.append(i)

    rng = random.Random(seed)
    rng.shuffle(correct_idx)
    rng.shuffle(wrong_idx)
    correct_idx = correct_idx[: max(0, int(num_correct))]
    wrong_idx = wrong_idx[: max(0, int(num_wrong))]

    base_out = os.path.join(out_dir, "retrieval_examples")
    out_correct = os.path.join(base_out, "correct")
    out_wrong = os.path.join(base_out, "wrong")
    _mkdir(out_correct)
    _mkdir(out_wrong)

    tile = (200, 200)
    border = 6
    green = (46, 160, 67)
    red = (214, 39, 40)
    gray = (180, 180, 180)

    def make_one(i: int, out_path: str) -> None:
        qname = q_info[i]["image_name"]
        qpath = os.path.join(data_root, "images", "query", qname)
        qimg = _thumb(qpath, tile)
        qtile = _tile_with_border(qimg, tile, border, gray)

        row_imgs: List[Image.Image] = [qtile]
        # add top-k gallery
        for rank, gi_idx in enumerate(order[i, :topk], start=1):
            gname = g_info[int(gi_idx)]["image_name"]
            gpath = os.path.join(data_root, "images", "gallery", gname)
            gimg = _thumb(gpath, tile)
            is_pos = (g_vid[int(gi_idx)] == q_vid[i])
            row_imgs.append(_tile_with_border(gimg, tile, border, green if is_pos else red))

        # Concatenate horizontally
        w = sum(im.size[0] for im in row_imgs)
        h = max(im.size[1] for im in row_imgs) + 40  # title bar
        canvas = Image.new("RGB", (w, h), (255, 255, 255))

        # Title
        draw = ImageDraw.Draw(canvas)
        title = f"Query {qname} | GT={q_vid[i]} | Top-1={'OK' if g_vid[order[i,0]]==q_vid[i] else 'WRONG'}"
        draw.text((10, 10), title, fill=(0, 0, 0))

        x = 0
        for im in row_imgs:
            canvas.paste(im, (x, 40))
            x += im.size[0]

        canvas.save(out_path)

    for i in correct_idx:
        qname = q_info[i]["image_name"].replace(".jpg", "")
        make_one(i, os.path.join(out_correct, f"{qname}_top{topk}.png"))

    for i in wrong_idx:
        qname = q_info[i]["image_name"].replace(".jpg", "")
        make_one(i, os.path.join(out_wrong, f"{qname}_top{topk}.png"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures for SimVeRi experiments")
    p.add_argument("--data-root", type=str, required=True, help="SimVeRi dataset root, e.g. ../SimVeRi-dataset-v2.0")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write figures")
    p.add_argument("--features-dir", type=str, default="", help="Feature directory (query/gallery features + info json)")
    p.add_argument("--results-json", type=str, default="", help="Evaluation JSON path (baseline_evaluation.json)")
    p.add_argument("--topk", type=int, default=10, help="Top-K for retrieval visualization")
    p.add_argument("--num-correct", type=int, default=12, help="#Queries to visualize where Top-1 is correct")
    p.add_argument("--num-wrong", type=int, default=12, help="#Queries to visualize where Top-1 is wrong")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    out_dir = args.output_dir
    _mkdir(out_dir)

    # Dataset-level figures (independent of model)
    ds_summary = _safe_load_json(os.path.join(data_root, "statistics", "dataset_summary.json"))
    if ds_summary:
        plot_dataset_distributions(ds_summary, out_dir)

    cam_net = _safe_load_json(os.path.join(data_root, "metadata", "camera_network.json"))
    if cam_net:
        plot_camera_layout_and_distances(cam_net, out_dir)

    # ReID evaluation figures
    results = _safe_load_json(args.results_json) if args.results_json else None
    if results:
        plot_cmc_curve(results, out_dir)

    # Feature-based figures (qualitative + similarity distributions)
    if args.features_dir:
        q_feat, g_feat, q_info, g_info = _load_features_bundle(args.features_dir)
        plot_similarity_distributions(q_feat, g_feat, q_info, g_info, out_dir, seed=int(args.seed))
        # Retrieval grids are the most valuable qualitative figures for papers.
        save_retrieval_examples(
            data_root=data_root,
            q_feat=q_feat,
            g_feat=g_feat,
            q_info=q_info,
            g_info=g_info,
            out_dir=out_dir,
            topk=int(args.topk),
            num_correct=int(args.num_correct),
            num_wrong=int(args.num_wrong),
            seed=int(args.seed),
        )

    print("=" * 70)
    print("Paper figures generated")
    print(f"Data root:   {data_root}")
    if args.features_dir:
        print(f"Features:    {args.features_dir}")
    if args.results_json:
        print(f"Eval JSON:   {args.results_json}")
    print(f"Output dir:  {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
