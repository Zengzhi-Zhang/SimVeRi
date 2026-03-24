# scripts/check_dataset.py
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root

def check_simveri_integrity(root_dir):
    """Check the integrity of a SimVeRi dataset release."""
    
    print(f"Checking dataset at: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"[ERROR] Root directory does not exist")
        return False

    required_structure = {
        'annotations': [
            'train_annotations.xml',
            'gallery_annotations.xml', 
            'query_list.txt',
            'ground_truth.txt',
            'ignore_list.txt'
        ],
        'metadata': [
            'spatiotemporal.json',
            'vehicle_attributes.json',
            'camera_network.json',
            'twins_groups.json',
            'trajectory_info.csv',
            'camera_transitions.csv'
        ],
        'statistics': [
            'dataset_summary.json'
        ]
    }
    
    errors = []
    
    for folder, files in required_structure.items():
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            errors.append(f"Missing folder: {folder}")
            continue
            
        for f in files:
            fpath = os.path.join(folder_path, f)
            if not os.path.exists(fpath):
                errors.append(f"Missing file: {folder}/{f}")
            else:
                if os.path.getsize(fpath) == 0:
                    errors.append(f"Empty file: {folder}/{f}")
                else:
                    print(f"[OK] Found: {folder}/{f}")
    
    total_images = 0
    for split in ['train', 'gallery', 'query']:
        img_dir = os.path.join(root_dir, 'images', split)
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            print(f"[OK] images/{split}: {count} images")
            total_images += count
        else:
            errors.append(f"Missing directory: images/{split}")
    
    print("-" * 30)
    if errors:
        print("\n[ERROR] INTEGRITY CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print(f"\n[OK] DATASET INTEGRITY CHECK PASSED")
    print(f"Total images found: {total_images}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the integrity of a SimVeRi dataset release.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=get_default_simveri_root(),
        help="Root directory of the released SimVeRi dataset.",
    )
    args = parser.parse_args()
    check_simveri_integrity(args.dataset_root)
