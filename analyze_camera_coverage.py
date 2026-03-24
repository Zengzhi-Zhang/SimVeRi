import json
import os
import re
import argparse
from collections import Counter


def load_counts(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = Counter()
    for cap in data:
        cam = cap.get('camera_id') or cap.get('cameraID') or cap.get('cameraId')
        if not cam:
            continue
        cam = str(cam)
        counts[cam] += 1
    return counts, len(data)


def cam_sort_key(cam):
    match = re.search(r'(\d+)', cam)
    return (int(match.group(1)) if match else 10**9, cam)


def print_table(raw_counts, cleaned_counts):
    cams = sorted(set(raw_counts) | set(cleaned_counts), key=cam_sort_key)
    print("camera_id\traw\tcleaned\tkept_pct")
    for cam in cams:
        raw = raw_counts.get(cam, 0)
        clean = cleaned_counts.get(cam, 0)
        pct = (clean / raw * 100.0) if raw else 0.0
        print(f"{cam}\t{raw}\t{clean}\t{pct:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compare per-camera counts before/after cleaning")
    parser.add_argument("--raw", default=r"output\\metadata\\captures.json", help="Path to raw captures.json")
    parser.add_argument("--cleaned", default=r"output_cleaned_test\\metadata\\captures_cleaned.json", help="Path to cleaned captures_cleaned.json")
    args = parser.parse_args()

    raw_path = args.raw
    cleaned_path = args.cleaned

    if not os.path.exists(raw_path):
        print(f"missing raw file: {raw_path}")
        return
    if not os.path.exists(cleaned_path):
        print(f"missing cleaned file: {cleaned_path}")
        return

    raw_counts, raw_total = load_counts(raw_path)
    cleaned_counts, cleaned_total = load_counts(cleaned_path)

    raw_cams = sorted(raw_counts.keys(), key=cam_sort_key)
    cleaned_cams = sorted(cleaned_counts.keys(), key=cam_sort_key)
    missing_after_clean = sorted(set(raw_cams) - set(cleaned_cams), key=cam_sort_key)
    missing_in_raw = sorted(set(cleaned_cams) - set(raw_cams), key=cam_sort_key)

    print("summary")
    print(f"raw_total\t{raw_total}")
    print(f"cleaned_total\t{cleaned_total}")
    print(f"raw_cameras\t{len(raw_cams)}")
    print(f"cleaned_cameras\t{len(cleaned_cams)}")
    print("")
    print("missing_after_clean\t" + (",".join(missing_after_clean) if missing_after_clean else "none"))
    print("missing_in_raw\t" + (",".join(missing_in_raw) if missing_in_raw else "none"))
    print("")
    print_table(raw_counts, cleaned_counts)


if __name__ == "__main__":
    main()
