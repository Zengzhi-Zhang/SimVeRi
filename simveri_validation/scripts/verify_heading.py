# scripts/verify_heading.py
"""
Heading-coordinate validation script

This is a key check for the heading-angle convention.
It verifies whether CARLA heading annotations match the assumed geometry.

Validation criteria:
- The red arrow should point along the vehicle motion direction (trajectory tangent).
- If the arrow points backward, add heading + 180 deg.
- If the arrow is perpendicular, swap cos/sin or add 90 deg.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.simveri_loader import SimVeRiDataset
from src.path_utils import get_default_simveri_root, get_validation_output_dir


def visualize_trajectory_with_heading(dataset: SimVeRiDataset, 
                                       vehicle_id: str, 
                                       output_dir: str,
                                       arrow_length: float = 8.0):
    """
    Visualize one vehicle trajectory together with heading angles.
    
    Args:
        dataset: SimVeRi dataset
        vehicle_id: vehicle ID
        output_dir: output directory
        arrow_length: arrow length in meters
    """
    samples = dataset.get_samples_by_vehicle(vehicle_id)
    
    if len(samples) < 3:
        print(f"  Vehicle {vehicle_id}: Too few samples ({len(samples)}), skipping")
        return None
    
    samples_sorted = sorted(samples, key=lambda s: s.timestamp)
    
    x = [s.position[0] for s in samples_sorted]
    y = [s.position[1] for s in samples_sorted]
    timestamps = [s.timestamp for s in samples_sorted]
    headings = [s.heading for s in samples_sorted]
    cameras = [s.camera_id for s in samples_sorted]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    ax.plot(x, y, 'b-', alpha=0.5, linewidth=2, label='Trajectory')
    
    scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=80, zorder=5, edgecolors='white')
    cbar = plt.colorbar(scatter, ax=ax, label='Time Order', shrink=0.8)
    
    for i, s in enumerate(samples_sorted):
        rad = np.radians(s.heading)
        dx = np.cos(rad) * arrow_length
        dy = np.sin(rad) * arrow_length
        
        ax.annotate('', 
                    xy=(s.position[0] + dx, s.position[1] + dy),
                    xytext=(s.position[0], s.position[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=15))
        
        ax.annotate(f'{s.camera_id}\n{s.timestamp:.0f}s', 
                    (s.position[0], s.position[1]),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=7, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))
    
    ax.scatter([x[0]], [y[0]], c='green', s=200, marker='o', zorder=10, label='Start')
    ax.scatter([x[-1]], [y[-1]], c='red', s=200, marker='s', zorder=10, label='End')
    
    red_arrow = mpatches.FancyArrowPatch((0, 0), (1, 0), color='red', arrowstyle='->', mutation_scale=15)
    ax.legend(handles=[
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Trajectory'),
        plt.Line2D([0], [0], marker='o', color='green', markersize=10, linestyle='', label='Start'),
        plt.Line2D([0], [0], marker='s', color='red', markersize=10, linestyle='', label='End'),
        mpatches.Patch(color='red', label='Heading Arrow')
    ])
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Vehicle {vehicle_id} Trajectory with Heading Arrows\n'
                 f'({len(samples_sorted)} samples, Red Arrow = Annotated Heading)',
                 fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'heading_verify_{vehicle_id}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")
    
    return samples_sorted


def analyze_heading_accuracy(samples: list) -> dict:
    """
    Quantitatively evaluate heading-angle accuracy.
    
    Compare the annotated heading with the actual displacement direction.
    """
    if len(samples) < 2:
        return None
    
    angle_diffs = []
    details = []
    
    for i in range(len(samples) - 1):
        s1, s2 = samples[i], samples[i+1]
        
        dx = s2.position[0] - s1.position[0]
        dy = s2.position[1] - s1.position[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 1.0:  # displacement is too small; skip this step
            continue
        
        actual_angle = np.degrees(np.arctan2(dy, dx))
        
        annotated_heading = s1.heading
        
        diff = actual_angle - annotated_heading
        diff = ((diff + 180) % 360) - 180
        
        angle_diffs.append(diff)
        details.append({
            'from_cam': s1.camera_id,
            'to_cam': s2.camera_id,
            'actual': actual_angle,
            'annotated': annotated_heading,
            'diff': diff,
            'distance': dist
        })
    
    if not angle_diffs:
        return None
    
    mean_diff = np.mean(angle_diffs)
    std_diff = np.std(angle_diffs)
    
    if abs(mean_diff) < 30:
        status = 'CORRECT'
        correction = 0
    elif abs(mean_diff - 180) < 30 or abs(mean_diff + 180) < 30:
        status = 'REVERSED'
        correction = 180
    elif abs(mean_diff - 90) < 30:
        status = 'ROTATED_90_CW'
        correction = -90
    elif abs(mean_diff + 90) < 30:
        status = 'ROTATED_90_CCW'
        correction = 90
    else:
        status = 'UNKNOWN'
        correction = round(mean_diff)
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'status': status,
        'correction': correction,
        'num_segments': len(angle_diffs),
        'details': details
    }


def print_heading_analysis(vehicle_id: str, analysis: dict):
    """Print the heading analysis result."""
    print(f"\n{'='*70}")
    print(f"Heading Analysis: Vehicle {vehicle_id}")
    print(f"{'='*70}")
    
    if analysis is None:
        print("  Insufficient data for analysis")
        return
    
    print(f"\n  Segments analyzed: {analysis['num_segments']}")
    print(f"  Mean difference:   {analysis['mean_diff']:.1f} deg")
    print(f"  Std deviation:     {analysis['std_diff']:.1f} deg")
    print(f"\n  Status: {analysis['status']}")
    print(f"  Suggested correction: {analysis['correction']} deg")
    
    print(f"\n  {'From':<8} {'To':<8} {'Actual':>10} {'Annotated':>10} {'Diff':>10} {'Dist':>8}")
    print(f"  {'-'*56}")
    
    for d in analysis['details'][:5]:
        print(f"  {d['from_cam']:<8} {d['to_cam']:<8} "
              f"{d['actual']:>10.1f} deg {d['annotated']:>10.1f} deg "
              f"{d['diff']:>10.1f} deg {d['distance']:>7.1f}m")
    
    if len(analysis['details']) > 5:
        print(f"  ... and {len(analysis['details']) - 5} more segments")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify heading annotations against trajectory direction.")
    parser.add_argument("--data-root", type=str, default=get_default_simveri_root(),
                        help="Root directory of the released SimVeRi dataset.")
    parser.add_argument("--output-dir", type=str, default=get_validation_output_dir("figures"),
                        help="Directory for verification figures and text summaries.")
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SimVeRi Heading Angle Verification")
    print("="*70)
    
    print("\nLoading dataset...")
    dataset = SimVeRiDataset(data_root, verbose=False)
    print(f"Loaded {len(dataset.samples)} samples")
    
    vehicle_counts = {}
    for s in dataset.samples.values():
        vehicle_counts[s.vehicle_id] = vehicle_counts.get(s.vehicle_id, 0) + 1
    
    sorted_vehicles = sorted(vehicle_counts.items(), key=lambda x: -x[1])
    
    print(f"\nTop 10 vehicles by sample count:")
    for vid, count in sorted_vehicles[:10]:
        is_twins = vid in dataset.twins_lookup
        print(f"  {vid}: {count:3d} samples {'[Twins]' if is_twins else ''}")
    
    
    test_vehicles = []
    
    for vid, count in sorted_vehicles:
        if vid not in dataset.twins_lookup and count >= 10:
            test_vehicles.append(vid)
            if len([v for v in test_vehicles if v not in dataset.twins_lookup]) >= 2:
                break
    
    for vid, count in sorted_vehicles:
        if vid in dataset.twins_lookup and count >= 5:
            test_vehicles.append(vid)
            break
    
    print(f"\nAnalyzing vehicles: {test_vehicles}")
    
    all_analyses = []
    
    for vid in test_vehicles:
        print(f"\n--- Processing vehicle {vid} ---")
        
        samples = visualize_trajectory_with_heading(dataset, vid, output_dir)
        
        if samples:
            analysis = analyze_heading_accuracy(samples)
            print_heading_analysis(vid, analysis)
            if analysis:
                all_analyses.append(analysis)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_analyses:
        all_diffs = [a['mean_diff'] for a in all_analyses]
        overall_mean = np.mean(all_diffs)
        
        print(f"\n  Overall mean heading difference: {overall_mean:.1f} deg")
        
        if all(a['status'] == 'CORRECT' for a in all_analyses):
            print("\n  [OK] CONCLUSION: Heading angles are correct")
            print("    No correction needed in the code.")
            correction = 0
        elif all(a['status'] == 'REVERSED' for a in all_analyses):
            print("\n  [WARN] CONCLUSION: Heading angles are reversed")
            print("    Add 180 deg to heading in direction_score.py")
            correction = 180
        else:
            statuses = [a['status'] for a in all_analyses]
            corrections = [a['correction'] for a in all_analyses]
            print(f"\n  [WARN] CONCLUSION: Mixed results")
            print(f"    Statuses: {statuses}")
            print(f"    Corrections: {corrections}")
            correction = int(np.median(corrections))
        
        result_path = os.path.join(output_dir, 'heading_verification_result.txt')
        with open(result_path, 'w') as f:
            f.write(f"Heading Verification Result\n")
            f.write(f"="*50 + "\n")
            f.write(f"Overall mean difference: {overall_mean:.1f} deg\n")
            f.write(f"Suggested correction: {correction} deg\n")
            f.write("\nVehicle-by-vehicle:")
            f.write("\n")
            for vid, a in zip(test_vehicles, all_analyses):
                f.write(f"  {vid}: {a['status']} (diff={a['mean_diff']:.1f} deg)\n")
        
        print(f"\n  Result saved to: {result_path}")
    
    print("\n" + "="*70)
    print("Please check the figures in:")
    print(f"  {output_dir}")
    print("\nVerification criteria:")
    print("  - Red arrows should point in the direction of vehicle movement")
    print("  - If arrows point backwards: heading_offset = 180")
    print("  - If arrows point sideways: heading_offset = +/-90")
    print("="*70)


if __name__ == "__main__":
    main()
