"""
SimVeRi data cleaning and split utility.

This script filters low-quality captures, subsamples tracklets, removes
single-camera vehicles, and exports cleaned image, JSON, and XML records.
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom

import numpy as np
from tqdm import tqdm


INPUT_IMAGE_DIR = "output/image_train"
INPUT_JSON = "output/metadata/captures.json"
OUTPUT_DIR = "output_cleaned_test"

IMG_WIDTH = 2560
IMG_HEIGHT = 1440

MIN_AREA = 96 * 96
MIN_WIDTH = 96
MIN_HEIGHT = 96
MIN_ASPECT = 0.7
MAX_ASPECT = 3.0
MARGIN_THRESHOLD = 10
MAX_EDGES_CUT = 1
MAX_OCCLUSION = 0.6

MAX_FRAMES_PER_TRACK = 30
MIN_CAMERAS_PER_VEHICLE = 2

SPECIAL_CAMERA_OVERRIDES = {
    "c008": {"min_width": 64, "min_height": 64, "min_area": 64 * 64, "max_edges_cut": 2},
    "c014": {"min_width": 64, "min_height": 64, "min_area": 64 * 64, "max_edges_cut": 2},
    "c022": {"min_width": 64, "min_height": 64, "min_area": 64 * 64, "max_edges_cut": 2},
    "c025": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
    "c026": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
    "c027": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
    "c028": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
    "c029": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
    "c030": {
        "min_width": 64,
        "min_height": 64,
        "min_area": 64 * 64,
        "max_edges_cut": 2,
        "min_aspect_ratio": 0.33,
    },
}


def is_valid_capture(cap):
    """Check whether a single capture record satisfies the release rules."""

    reasons = []

    cam_id = cap.get("camera_id", "")
    overrides = SPECIAL_CAMERA_OVERRIDES.get(cam_id, {})
    min_width = overrides.get("min_width", MIN_WIDTH)
    min_height = overrides.get("min_height", MIN_HEIGHT)
    min_area = overrides.get("min_area", MIN_AREA)
    max_edges_cut = overrides.get("max_edges_cut", MAX_EDGES_CUT)
    margin_threshold = overrides.get("margin_threshold", MARGIN_THRESHOLD)
    min_aspect = overrides.get("min_aspect_ratio", MIN_ASPECT)
    max_aspect = overrides.get("max_aspect_ratio", MAX_ASPECT)

    xmin, ymin, xmax, ymax = cap["bbox"]
    width = xmax - xmin
    height = ymax - ymin

    if width * height < min_area:
        reasons.append("too_small")

    if width < min_width:
        reasons.append("width_too_small")
    if height < min_height:
        reasons.append("height_too_small")

    aspect = width / height if height > 0 else 0
    if aspect < min_aspect or aspect > max_aspect:
        reasons.append("bad_aspect_ratio")

    edges_cut = 0
    if xmin <= margin_threshold:
        edges_cut += 1
    if xmax >= IMG_WIDTH - margin_threshold:
        edges_cut += 1
    if ymin <= margin_threshold:
        edges_cut += 1
    if ymax >= IMG_HEIGHT - margin_threshold:
        edges_cut += 1

    if edges_cut > max_edges_cut:
        reasons.append(f"truncated_{edges_cut}_edges")

    if cap.get("occlusion_ratio", 0) > MAX_OCCLUSION:
        reasons.append("high_occlusion")

    return len(reasons) == 0, reasons


def select_best_frames(items, k=10):
    """Keep up to ``k`` frames from one tracklet using uniform temporal sampling."""

    if len(items) <= k:
        return items

    items.sort(key=lambda item: item["frame_id"])
    indices = np.linspace(0, len(items) - 1, k, dtype=int)
    return [items[index] for index in indices]


def save_cleaned_json(items, output_path):
    """Save the cleaned JSON records."""

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(items, file_obj, indent=2, ensure_ascii=False)
    print(f"  Cleaned JSON: {output_path}")


def save_cleaned_xml(items, output_path):
    """Save cleaned VeRi-style XML metadata."""

    root = ET.Element("TrainLabel")
    root.set("Version", "SimVeRi-1.0-Cleaned")
    root.set("TotalImages", str(len(items)))

    for capture in items:
        item = ET.SubElement(root, "Item")
        item.set("vehicleID", capture["vehicle_id"])
        item.set("imageName", os.path.basename(capture["image_path"]))
        item.set("cameraID", capture["camera_id"])
        item.set("colorID", str(capture.get("color_id_veri", 0)))
        item.set("colorName", capture.get("color_name_veri", "unknown"))
        item.set("typeID", capture.get("category", ""))
        item.set("brandID", capture.get("brand", ""))
        item.set("frameID", str(capture["frame_id"]))
        item.set("timestamp", f"{capture['timestamp']:.2f}")
        item.set("distance", f"{capture['distance']:.1f}")
        item.set("occlusion", f"{capture['occlusion_ratio']:.2f}")
        item.set("occLevel", capture["occlusion_level"])
        item.set("bboxArea", str(capture["bbox_area"]))
        item.set("isFleet", str(capture.get("is_fleet", False)))
        item.set("globalX", f"{capture.get('global_x', 0):.2f}")
        item.set("globalY", f"{capture.get('global_y', 0):.2f}")
        item.set("globalZ", f"{capture.get('global_z', 0):.2f}")
        item.set("speed", f"{capture.get('speed', 0):.2f}")
        item.set("heading", f"{capture.get('heading', 0):.2f}")

        if capture.get("fleet_id"):
            item.set("fleetID", capture["fleet_id"])

    xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
        indent="  "
    )
    xml_str = "\n".join(line for line in xml_str.split("\n") if line.strip())

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(xml_str)

    print(f"  Cleaned XML: {output_path}")


def save_statistics(stats, final_items, output_path):
    """Save the final cleaning report."""

    final_vehicles = {item["vehicle_id"] for item in final_items}
    final_cameras = {item["camera_id"] for item in final_items}

    vehicle_cameras = defaultdict(set)
    for item in final_items:
        vehicle_cameras[item["vehicle_id"]].add(item["camera_id"])

    avg_cameras = (
        np.mean([len(cameras) for cameras in vehicle_cameras.values()])
        if vehicle_cameras
        else 0
    )

    hard_items = [item for item in final_items if item.get("is_fleet", False)]
    hard_vehicles = {item["vehicle_id"] for item in hard_items}

    report = {
        "cleaning_config": {
            "min_area": MIN_AREA,
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
            "min_aspect_ratio": MIN_ASPECT,
            "max_aspect_ratio": MAX_ASPECT,
            "margin_threshold": MARGIN_THRESHOLD,
            "max_edges_cut": MAX_EDGES_CUT,
            "max_occlusion": MAX_OCCLUSION,
            "max_frames_per_track": MAX_FRAMES_PER_TRACK,
            "min_cameras_per_vehicle": MIN_CAMERAS_PER_VEHICLE,
        },
        "filtering_stats": dict(stats),
        "final_dataset": {
            "total_images": len(final_items),
            "unique_vehicles": len(final_vehicles),
            "unique_cameras": len(final_cameras),
            "avg_cameras_per_vehicle": round(avg_cameras, 2),
            "hard_subset_images": len(hard_items),
            "hard_subset_vehicles": len(hard_vehicles),
        },
    }

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)

    print(f"  Statistics report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SimVeRi clean-and-split utility (quality filter + track sampling)"
    )
    parser.add_argument(
        "--input-dir",
        default="output",
        help="Raw capture output directory containing metadata/captures.json",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Destination directory for the cleaned release files",
    )
    args = parser.parse_args()

    input_json = os.path.join(args.input_dir, "metadata", "captures.json")
    output_dir = args.output_dir

    print("=" * 60)
    print("SimVeRi data cleaning utility v1.2")
    print("=" * 60)

    print(f"\n[1/5] Load raw data: {input_json}")
    if not os.path.exists(input_json):
        print(f"  Error: file not found: {input_json}")
        return

    with open(input_json, "r", encoding="utf-8") as file_obj:
        captures = json.load(file_obj)

    print(f"  Raw records: {len(captures)}")

    print("\n[2/5] Quality filtering...")
    stats = defaultdict(int)
    tracks = defaultdict(lambda: defaultdict(list))

    for capture in tqdm(captures, desc="  Filtering"):
        stats["total_raw"] += 1
        valid, reasons = is_valid_capture(capture)
        if not valid:
            for reason in reasons:
                stats[f"drop_{reason}"] += 1
            stats["drop_total"] += 1
            continue

        vehicle_id = capture["vehicle_id"]
        camera_id = capture["camera_id"]
        tracks[vehicle_id][camera_id].append(capture)
        stats["pass_quality_filter"] += 1

    print(
        f"  Passed quality filtering: {stats['pass_quality_filter']} / {stats['total_raw']}"
    )

    print("\n[3/5] Cross-camera filtering and track sampling...")
    final_items = []
    vehicles_dropped = 0

    for vehicle_id, cameras in tqdm(tracks.items(), desc="  Sampling"):
        if len(cameras) < MIN_CAMERAS_PER_VEHICLE:
            for _, items in cameras.items():
                stats["drop_single_camera"] += len(items)
            vehicles_dropped += 1
            continue

        for _, items in cameras.items():
            original_count = len(items)
            selected = select_best_frames(items, k=MAX_FRAMES_PER_TRACK)
            final_items.extend(selected)
            stats["sampled_from_tracks"] += len(selected)
            stats["dropped_by_sampling"] += original_count - len(selected)

    stats["vehicles_dropped_single_camera"] = vehicles_dropped
    print(f"  Dropped single-camera vehicles: {vehicles_dropped}")
    print(f"  Images after track sampling: {len(final_items)}")

    print(f"\n[4/5] Create output directories: {output_dir}")
    os.makedirs(f"{output_dir}/image_train", exist_ok=True)
    os.makedirs(f"{output_dir}/metadata", exist_ok=True)
    os.makedirs(f"{output_dir}/statistics", exist_ok=True)

    print("\n[5/5] Export data...")
    print("  Copy image files...")
    exported = 0
    missing = 0

    for item in tqdm(final_items, desc="  Copying"):
        src = item["image_path"]
        filename = os.path.basename(src)
        dst = os.path.join(output_dir, "image_train", filename)

        if os.path.exists(src):
            shutil.copy(src, dst)
            item["image_path"] = dst
            exported += 1
        else:
            missing += 1

    if missing > 0:
        print(f"  [WARN] Missing images: {missing}")

    stats["exported_images"] = exported

    output_json = os.path.join(output_dir, "metadata", "captures_cleaned.json")
    save_cleaned_json(final_items, output_json)

    output_xml = os.path.join(output_dir, "metadata", "train_label_cleaned.xml")
    save_cleaned_xml(final_items, output_xml)

    stats_path = os.path.join(output_dir, "statistics", "cleaning_report.json")
    save_statistics(stats, final_items, stats_path)

    print("\n" + "=" * 60)
    print("Cleaning summary")
    print("=" * 60)

    print("\nDrop reason statistics:")
    for key, value in sorted(stats.items()):
        if key.startswith("drop_"):
            print(f"  {key}: {value}")

    final_vehicles = {item["vehicle_id"] for item in final_items}
    final_cameras = {item["camera_id"] for item in final_items}
    hard_items = [item for item in final_items if item.get("is_fleet", False)]

    vehicle_cameras = defaultdict(set)
    for item in final_items:
        vehicle_cameras[item["vehicle_id"]].add(item["camera_id"])
    avg_cameras = (
        np.mean([len(cameras) for cameras in vehicle_cameras.values()])
        if vehicle_cameras
        else 0
    )

    print("\nFinal dataset:")
    print(f"  Total images: {len(final_items)}")
    print(f"  Unique vehicles: {len(final_vehicles)}")
    print(f"  Covered cameras: {len(final_cameras)}")
    print(f"  Average cameras per vehicle: {avg_cameras:.2f}")
    print(f"  Hard subset images: {len(hard_items)}")

    print(f"\nOutput location: {output_dir}")
    print("=" * 60)
    print("Cleaning finished.")


if __name__ == "__main__":
    main()
