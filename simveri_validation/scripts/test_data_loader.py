# scripts/test_data_loader.py
"""
Comprehensive data-loader test script
Validate whether all dataset-loading functions behave as expected
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.simveri_loader import SimVeRiDataset
from src.path_utils import get_default_simveri_root


def run_comprehensive_tests(data_root: str):
    """Run the full test suite."""

    print("="*70)
    print("SimVeRi Data Loader Comprehensive Tests")
    print("="*70)
    
    print("\n[Test 0] Loading dataset...")
    try:
        dataset = SimVeRiDataset(data_root, verbose=True)
        print("[OK] Dataset loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        return False
    
    tests_passed = 0
    tests_total = 0
    
    # ----------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Basic Tests")
    print("-"*50)
    
    tests_total += 1
    expected_total = 21178  # expected from the dataset specification
    actual_total = len(dataset.samples)
    if actual_total > 0:
        print(f"[OK] Test 1: Total samples = {actual_total}")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 1: No samples loaded")
    
    tests_total += 1
    train_count = len(dataset.train_samples)
    gallery_count = len(dataset.gallery_samples)
    query_count = len(dataset.query_samples)
    if train_count > 0 and gallery_count > 0 and query_count > 0:
        print(f"[OK] Test 2: Train={train_count}, Gallery={gallery_count}, Query={query_count}")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 2: Some splits are empty")
    
    tests_total += 1
    twins_ids = dataset.get_twins_vehicle_ids()
    if len(twins_ids) == 100:
        print(f"[OK] Test 3: Twins vehicles = {len(twins_ids)} (correct)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 3: Twins vehicles = {len(twins_ids)} (expected 100)")
    
    # ----------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Spatiotemporal Data Tests")
    print("-"*50)
    
    tests_total += 1
    timestamps = [s.timestamp for s in dataset.samples.values() if s.timestamp > 0]
    if timestamps:
        min_ts, max_ts = min(timestamps), max(timestamps)
        print(f"[OK] Test 4: Timestamp range = [{min_ts:.2f}, {max_ts:.2f}] seconds")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 4: No valid timestamps")
    
    tests_total += 1
    positions = [s.position for s in dataset.samples.values()]
    valid_positions = [p for p in positions if np.any(p != 0)]
    if len(valid_positions) > 0:
        all_pos = np.array(valid_positions)
        x_range = (all_pos[:, 0].min(), all_pos[:, 0].max())
        y_range = (all_pos[:, 1].min(), all_pos[:, 1].max())
        z_range = (all_pos[:, 2].min(), all_pos[:, 2].max())
        print(f"[OK] Test 5: Position ranges")
        print(f"         X: [{x_range[0]:.1f}, {x_range[1]:.1f}] m")
        print(f"         Y: [{y_range[0]:.1f}, {y_range[1]:.1f}] m")
        print(f"         Z: [{z_range[0]:.1f}, {z_range[1]:.1f}] m")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 5: No valid positions")
    
    tests_total += 1
    speeds = [s.speed for s in dataset.samples.values() if s.speed > 0]
    if speeds:
        print(f"[OK] Test 6: Speed range = [{min(speeds):.1f}, {max(speeds):.1f}] km/h")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 6: No valid speeds")
    
    tests_total += 1
    headings = [s.heading for s in dataset.samples.values()]
    unique_headings = len(set([round(h, 1) for h in headings]))
    if unique_headings > 10:  # heading angles should show sufficient diversity
        print(f"[OK] Test 7: Heading diversity = {unique_headings} unique values")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 7: Low heading diversity ({unique_headings})")
    
    tests_total += 1
    occlusions = [s.occlusion for s in dataset.samples.values()]
    occ_range = (min(occlusions), max(occlusions))
    if occ_range[1] > occ_range[0]:
        print(f"[OK] Test 8: Occlusion range = [{occ_range[0]:.3f}, {occ_range[1]:.3f}]")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 8: No occlusion variation")
    
    # ----------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Subset Split Tests")
    print("-"*50)
    
    tests_total += 1
    for split in ['train', 'gallery', 'query']:
        base = dataset.get_base_samples(split)
        twins = dataset.get_twins_samples(split)
        total = getattr(dataset, f'{split}_samples')
        
        if len(base) + len(twins) == len(total):
            continue
        else:
            print(f"[FAIL] Test 9: {split} split inconsistent")
            break
    else:
        print(f"[OK] Test 9: Base/Twins split consistent across all subsets")
        tests_passed += 1
    
    tests_total += 1
    twins_samples = dataset.get_twins_samples('gallery')
    if twins_samples and all(s.is_twins for s in twins_samples):
        print(f"[OK] Test 10: All Twins samples correctly marked")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 10: Twins marking inconsistent")
    
    # ----------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Camera Network Tests")
    print("-"*50)
    
    tests_total += 1
    cameras = dataset.camera_network.get('cameras', {})
    if len(cameras) == 24:
        print(f"[OK] Test 11: Camera count = {len(cameras)} (correct)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 11: Camera count = {len(cameras)} (expected 24)")
    
    tests_total += 1
    dist_matrix = dataset.camera_network.get('distance_matrix', {})
    if dist_matrix:
        sample_dist = dataset.get_distance('c1', 'c2')
        print(f"[OK] Test 12: Distance matrix loaded (c1->c2 = {sample_dist:.1f}m)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 12: Distance matrix not loaded")
    
    tests_total += 1
    if len(dataset.transition_params) > 0:
        sample_key = list(dataset.transition_params.keys())[0]
        sample_param = dataset.transition_params[sample_key]
        print(f"[OK] Test 13: Transition params loaded ({len(dataset.transition_params)} pairs)")
        print(f"         Sample: {sample_key[0]}->{sample_key[1]}: "
              f"mean={sample_param['mean_time']:.1f}s, std={sample_param['std_time']:.1f}s")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 13: Transition params not loaded")
    
    # ----------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Sample Integrity Tests")
    print("-"*50)
    
    tests_total += 1
    valid_paths = 0
    for s in list(dataset.samples.values())[:100]:  # sample-check 100 records
        if s.image_path and os.path.exists(s.image_path):
            valid_paths += 1
    if valid_paths == 100:
        print(f"[OK] Test 14: Image paths valid (100/100 sampled)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 14: Some image paths invalid ({valid_paths}/100)")
    
    tests_total += 1
    incomplete = 0
    for s in dataset.train_samples[:100]:
        if s.timestamp == 0 or np.all(s.position == 0):
            incomplete += 1
    if incomplete == 0:
        print(f"[OK] Test 15: Sample data complete (100 sampled)")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 15: {incomplete}/100 samples have incomplete data")
    
    # ----------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"Test Results: {tests_passed}/{tests_total} passed")
    print("="*70)
    
    if tests_passed == tests_total:
        print("\n[OK] ALL TESTS PASSED")
        return True
    else:
        print(f"\n[FAIL] {tests_total - tests_passed} tests failed")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive SimVeRi data-loader tests.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=get_default_simveri_root(),
        help="Root directory of the released SimVeRi dataset.",
    )
    args = parser.parse_args()
    success = run_comprehensive_tests(args.data_root)
    sys.exit(0 if success else 1)
