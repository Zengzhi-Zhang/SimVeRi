# scripts/evaluate_baseline.py
"""
Baseline evaluation script

Evaluate pure-visual ReID performance to validate image quality and model effectiveness
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root


def count_images(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.endswith('.jpg'))


def validate_feature_consistency(
    features_dir: str,
    data_root: str,
    gallery_count: int,
    query_count: int,
    skip_check: bool,
) -> bool:
    meta_path = os.path.join(features_dir, 'features_meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = None
    else:
        meta = None

    if meta:
        meta_gallery = meta.get('gallery_count')
        meta_query = meta.get('query_count')
        meta_input_size = meta.get('input_size')
        if meta_input_size is not None:
            print(f"  Info: features extracted with input_size={meta_input_size}")
        if meta_gallery is not None and meta_gallery != gallery_count:
            print(f"  Warning: gallery count mismatch vs features_meta.json ({gallery_count} != {meta_gallery})")
        if meta_query is not None and meta_query != query_count:
            print(f"  Warning: query count mismatch vs features_meta.json ({query_count} != {meta_query})")
        meta_root = meta.get('data_root')
        if meta_root and data_root:
            if os.path.normcase(os.path.abspath(meta_root)) != os.path.normcase(os.path.abspath(data_root)):
                print(f"  Warning: data_root mismatch (features_meta.json: {meta_root} vs args: {data_root})")

    if skip_check:
        return True

    if not data_root:
        print("  Warning: data_root not provided; skip dataset check.")
        return True

    gallery_dir = os.path.join(data_root, 'images', 'gallery')
    query_dir = os.path.join(data_root, 'images', 'query')
    if not os.path.isdir(gallery_dir) or not os.path.isdir(query_dir):
        print(f"  Warning: dataset folders not found under {data_root}; skip dataset check.")
        return True

    gallery_files = count_images(gallery_dir)
    query_files = count_images(query_dir)

    if gallery_files != gallery_count or query_files != query_count:
        print("  ERROR: Dataset/feature count mismatch detected.")
        print(f"    dataset gallery/query: {gallery_files}/{query_files}")
        print(f"    features gallery/query: {gallery_count}/{query_count}")
        print("  Please re-run extract_features.py for the current dataset.")
        return False

    return True


def cosine_similarity_matrix(query_feats: np.ndarray, 
                              gallery_feats: np.ndarray) -> np.ndarray:
    """Compute the cosine-similarity matrix."""
    query_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-12)
    gallery_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12)
    
    return np.dot(query_norm, gallery_norm.T)


def compute_ap(matches: List[bool]) -> float:
    """Compute Average Precision for one Query sample."""
    if not any(matches):
        return 0.0
    
    num_rel = sum(matches)
    cum_correct = 0
    precision_sum = 0.0
    
    for i, is_match in enumerate(matches):
        if is_match:
            cum_correct += 1
            precision_sum += cum_correct / (i + 1)
    
    return precision_sum / num_rel


def compute_cmc(ranks: List[int], max_rank: int = 50) -> np.ndarray:
    """Compute the CMC curve."""
    cmc = np.zeros(max_rank)
    
    for rank in ranks:
        if 1 <= rank <= max_rank:
            cmc[rank-1:] += 1
    
    cmc /= len(ranks)
    return cmc

def evaluate_reid(sim_matrix: np.ndarray,
                  query_info: List[Dict],
                  gallery_info: List[Dict],
                  ignore_same_camera: bool = True) -> Dict:
    """
    Complete ReID evaluation.
    """
    num_query = len(query_info)
    
    all_ap = []
    all_ranks = []
    
    for i in range(num_query):
        q_id = query_info[i]['vehicle_id']
        q_cam = query_info[i]['camera_id']
        q_img = query_info[i]['image_name']  # keep image names for same-image filtering
        
        scores = sim_matrix[i].copy()
        
        valid_indices = []
        for j in range(len(gallery_info)):
            g_id = gallery_info[j]['vehicle_id']
            g_cam = gallery_info[j]['camera_id']
            g_img = gallery_info[j]['image_name']  # keep image names for same-image filtering
            
            if g_img == q_img:
                continue
            
            if ignore_same_camera and q_id == g_id and q_cam == g_cam:
                continue
            
            valid_indices.append(j)
        
        if not valid_indices:
            continue
        
        valid_scores = scores[valid_indices]
        sorted_idx = np.argsort(-valid_scores)
        
        matches = []
        for idx in sorted_idx:
            original_idx = valid_indices[idx]
            is_match = (gallery_info[original_idx]['vehicle_id'] == q_id)
            matches.append(is_match)
        
        ap = compute_ap(matches)
        all_ap.append(ap)
        
        if True in matches:
            rank = matches.index(True) + 1
        else:
            rank = len(matches) + 1
        all_ranks.append(rank)
    
    mAP = np.mean(all_ap) * 100
    cmc = compute_cmc(all_ranks) * 100
    
    return {
        'mAP': mAP,
        'Rank-1': cmc[0],
        'Rank-5': cmc[4] if len(cmc) > 4 else cmc[-1],
        'Rank-10': cmc[9] if len(cmc) > 9 else cmc[-1],
        'CMC': cmc.tolist(),
        'num_query': len(all_ranks)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReID baseline")
    parser.add_argument("--features-dir", type=str, default="",
                        help="Directory containing extracted features")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Directory for evaluation results")
    parser.add_argument("--data-root", type=str, default=get_default_simveri_root(),
                        help="Dataset root for consistency check")
    parser.add_argument("--skip-dataset-check", action="store_true",
                        help="Skip dataset/feature consistency check")
    args = parser.parse_args()
    
    print("="*70)
    print("SimVeRi Baseline Evaluation")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.features_dir:
        args.features_dir = os.path.join(PROJECT_ROOT, "outputs", "features")
    
    if not args.output_dir:
        args.output_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "-"*50)
    print("Loading Features")
    print("-"*50)
    
    gallery_features = np.load(os.path.join(args.features_dir, 'gallery_features.npy'))
    query_features = np.load(os.path.join(args.features_dir, 'query_features.npy'))
    
    with open(os.path.join(args.features_dir, 'gallery_info.json'), 'r') as f:
        gallery_info = json.load(f)
    
    with open(os.path.join(args.features_dir, 'query_info.json'), 'r') as f:
        query_info = json.load(f)
    
    print(f"  Gallery: {gallery_features.shape}")
    print(f"  Query: {query_features.shape}")

    if gallery_features.shape[0] != len(gallery_info) or query_features.shape[0] != len(query_info):
        print("  ERROR: Feature array size does not match info JSON length.")
        print(f"    gallery_features: {gallery_features.shape[0]} vs gallery_info: {len(gallery_info)}")
        print(f"    query_features: {query_features.shape[0]} vs query_info: {len(query_info)}")
        return

    if not validate_feature_consistency(
        args.features_dir,
        args.data_root,
        len(gallery_info),
        len(query_info),
        args.skip_dataset_check,
    ):
        return
    
    print("\n" + "-"*50)
    print("Computing Similarity Matrix")
    print("-"*50)
    
    sim_matrix = cosine_similarity_matrix(query_features, gallery_features)
    print(f"  Similarity matrix: {sim_matrix.shape}")
    
    print("\n" + "="*50)
    print("Evaluation: Full Dataset")
    print("="*50)
    
    results_full = evaluate_reid(sim_matrix, query_info, gallery_info)
    
    print(f"  Rank-1:  {results_full['Rank-1']:.2f}%")
    print(f"  Rank-5:  {results_full['Rank-5']:.2f}%")
    print(f"  Rank-10: {results_full['Rank-10']:.2f}%")
    print(f"  mAP:     {results_full['mAP']:.2f}%")
    
    print("\n" + "="*50)
    print("Evaluation: Base Subset (Non-Twins)")
    print("="*50)
    
    base_query_idx = [i for i, info in enumerate(query_info) if not info['is_twins']]
    base_gallery_idx = [i for i, info in enumerate(gallery_info) if not info['is_twins']]
    
    if base_query_idx and base_gallery_idx:
        base_sim = sim_matrix[np.ix_(base_query_idx, base_gallery_idx)]
        base_query_info = [query_info[i] for i in base_query_idx]
        base_gallery_info = [gallery_info[i] for i in base_gallery_idx]
        
        results_base = evaluate_reid(base_sim, base_query_info, base_gallery_info)
        
        print(f"  Queries: {len(base_query_idx)}")
        print(f"  Rank-1:  {results_base['Rank-1']:.2f}%")
        print(f"  Rank-5:  {results_base['Rank-5']:.2f}%")
        print(f"  Rank-10: {results_base['Rank-10']:.2f}%")
        print(f"  mAP:     {results_base['mAP']:.2f}%")
    else:
        results_base = None
        print("  No Base samples in query/gallery")
    
    print("\n" + "="*50)
    print("Evaluation: Twins Subset")
    print("="*50)
    
    twins_query_idx = [i for i, info in enumerate(query_info) if info['is_twins']]
    twins_gallery_idx = [i for i, info in enumerate(gallery_info) if info['is_twins']]
    
    if twins_query_idx and twins_gallery_idx:
        twins_sim = sim_matrix[np.ix_(twins_query_idx, twins_gallery_idx)]
        twins_query_info = [query_info[i] for i in twins_query_idx]
        twins_gallery_info = [gallery_info[i] for i in twins_gallery_idx]
        
        results_twins = evaluate_reid(twins_sim, twins_query_info, twins_gallery_info)
        
        print(f"  Queries: {len(twins_query_idx)}")
        print(f"  Rank-1:  {results_twins['Rank-1']:.2f}%")
        print(f"  Rank-5:  {results_twins['Rank-5']:.2f}%")
        print(f"  Rank-10: {results_twins['Rank-10']:.2f}%")
        print(f"  mAP:     {results_twins['mAP']:.2f}%")
    else:
        results_twins = None
        print("  No Twins samples in query/gallery")
    
    print("\n" + "-"*50)
    print("Saving Results")
    print("-"*50)
    
    all_results = {
        'evaluation_time': datetime.now().isoformat(),
        'full': results_full,
        'base': results_base,
        'twins': results_twins
    }
    
    output_path = os.path.join(args.output_dir, 'baseline_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Subset':<15} {'Rank-1':>10} {'Rank-5':>10} {'mAP':>10}")
    print("-"*45)
    print(f"{'Full':<15} {results_full['Rank-1']:>9.2f}% {results_full['Rank-5']:>9.2f}% {results_full['mAP']:>9.2f}%")
    if results_base:
        print(f"{'Base':<15} {results_base['Rank-1']:>9.2f}% {results_base['Rank-5']:>9.2f}% {results_base['mAP']:>9.2f}%")
    if results_twins:
        print(f"{'Twins':<15} {results_twins['Rank-1']:>9.2f}% {results_twins['Rank-5']:>9.2f}% {results_twins['mAP']:>9.2f}%")
    print("="*70)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
