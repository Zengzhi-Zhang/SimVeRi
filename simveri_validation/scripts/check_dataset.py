# scripts/check_dataset.py
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root


def check_simveri_integrity(root_dir):
    """Check the integrity of a SimVeRi dataset release (v1.1).

    Verifies, on the extracted release package, the five root-level documentation
    files, the 18 required internal records listed in the Data Records
    "Principal record files" table (the 17 data records plus the v1.1
    release-integrity manifest), and the three core image directories.

    Air-ground records are enumerated once in the Data Records table, so the
    headline "required internal records" count uses the full-scope protocol
    (annotations_and_protocols/ag_protocol_full/) as the canonical location.
    The test-scope protocol (annotations_and_protocols/ag_protocol/) mirrors the
    same record types and is validated separately.
    """

    print(f"Checking dataset at: {root_dir}")

    if not os.path.exists(root_dir):
        print("[ERROR] Root directory does not exist")
        return False

    # Five root-level documentation files.
    root_files = ['README.md', 'LICENSE_DATA', 'CITATION.cff', 'VERSION', 'SHA256SUMS']

    # 18 required internal records, at their real released paths.
    required_structure = {
        'annotations_and_protocols/annotations': [
            'train_annotations.xml',
            'gallery_annotations.xml',
            'query_list.txt',
            'ground_truth.txt',
            'ignore_list.txt',
        ],
        'metadata': [
            'spatiotemporal.json',
            'splits.json',
            'camera_network.json',
            'trajectory_info.csv',
            'camera_transitions.csv',
            'vehicle_attributes.json',
        ],
        'extras/twins/metadata': [
            'twins_groups.json',
            'spatiotemporal_twins.json',
            'trajectory_info_twins.csv',
        ],
        'annotations_and_protocols/ag_protocol_full/metadata': [
            'tracklets.json',
            'protocol.json',
            'pairs.csv',
        ],
        'statistics': [
            'image_manifest.csv',
        ],
    }
    required_record_total = sum(len(v) for v in required_structure.values())

    # Test-scope air-ground protocol: same record types as the full scope, validated
    # separately so the headline count above follows the Data Records enumeration.
    mirror_structure = {
        'annotations_and_protocols/ag_protocol/metadata': [
            'tracklets.json',
            'protocol.json',
            'pairs.csv',
        ],
    }
    mirror_total = sum(len(v) for v in mirror_structure.values())

    errors = []

    def _check_group(structure):
        found = 0
        for folder, files in structure.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                errors.append(f"Missing folder: {folder}")
                continue
            for f in files:
                fpath = os.path.join(folder_path, f)
                if not os.path.exists(fpath):
                    errors.append(f"Missing file: {folder}/{f}")
                elif os.path.getsize(fpath) == 0:
                    errors.append(f"Empty file: {folder}/{f}")
                else:
                    found += 1
                    print(f"[OK] Found: {folder}/{f}")
        return found

    root_ok = 0
    for f in root_files:
        fpath = os.path.join(root_dir, f)
        if not os.path.exists(fpath):
            errors.append(f"Missing root file: {f}")
        elif os.path.getsize(fpath) == 0:
            errors.append(f"Empty root file: {f}")
        else:
            root_ok += 1
            print(f"[OK] Found: {f}")

    record_ok = _check_group(required_structure)
    mirror_ok = _check_group(mirror_structure)

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

    print("\n[OK] DATASET INTEGRITY CHECK PASSED")
    print(f"Root-level documentation files present: {root_ok}/{len(root_files)}")
    print(f"Required internal records present: {record_ok}/{required_record_total}")
    print(f"Test-scope air-ground records present: {mirror_ok}/{mirror_total}")
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
    ok = check_simveri_integrity(args.dataset_root)
    sys.exit(0 if ok else 1)
