#!/usr/bin/env python3
"""Export per-image spatiotemporal metadata for SimVeRi AG protocols."""

import argparse
import json
import os
import sys


DESCRIPTION = (
    "Per-image spatiotemporal metadata for the SimVeRi air-ground protocol "
    "(air+ground tracklet images)."
)
COORDINATE_SYSTEM = "CARLA world coordinates (meters)"
VERSION = "2.0"
EXPECTED_COUNTS = {
    "test": {"air": 5101, "ground": 12418, "total": 17519},
    "full": {"air": 15944, "ground": 15942, "total": 31886},
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_image_name(filename):
    stem, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() != ".jpg":
        raise ValueError("image filename must end with .jpg: %s" % filename)
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError("image filename must be <mapped>_<camera>_<frame>.jpg: %s" % filename)
    mapped_id, camera_id, frame_text = parts
    return mapped_id, camera_id, int(frame_text)


def build_capture_index(records):
    index = {}
    duplicate_keys = []
    for record in records:
        key = (
            str(record["vehicle_id"]),
            str(record["camera_id"]),
            int(record["frame_id"]),
        )
        if key in index:
            duplicate_keys.append(key)
        else:
            index[key] = record
    return index, duplicate_keys


def find_capture(key, primary_index, fallback_index):
    if key in primary_index:
        return primary_index[key], "primary"
    if fallback_index is not None and key in fallback_index:
        return fallback_index[key], "fallback"
    return None, None


def make_annotation(tracklet, filename, mapped_id, camera_id, frame_id, capture):
    return {
        "vehicle_id": mapped_id,
        "original_id": str(tracklet["vehicle_id"]),
        "camera_id": camera_id,
        "layer": str(tracklet.get("layer", "")),
        "frame_id": frame_id,
        "timestamp": capture.get("timestamp"),
        "position": {
            "x": capture.get("global_x"),
            "y": capture.get("global_y"),
            "z": capture.get("global_z"),
        },
        "motion": {
            "speed_kmh": capture.get("speed"),
            "heading_deg": capture.get("heading"),
        },
        "quality": {
            "occlusion_ratio": capture.get("occlusion_ratio"),
            "distance_m": capture.get("distance"),
        },
    }


def iter_tracklets(tracklets):
    for layer_key in ("air_tracklets", "ground_tracklets"):
        for tracklet_id in sorted(tracklets.get(layer_key, {})):
            yield layer_key, tracklets[layer_key][tracklet_id]


def infer_scope(ag_root, tracklets):
    scope = tracklets.get("scope")
    if scope:
        return str(scope)
    root_name = os.path.basename(os.path.normpath(ag_root))
    return "full" if root_name.endswith("_full") else "test"


def export_annotations(ag_root, captures_path, fallback_path=None):
    tracklets_path = os.path.join(ag_root, "metadata", "tracklets.json")
    tracklets = load_json(tracklets_path)
    primary_records = load_json(captures_path)
    fallback_records = load_json(fallback_path) if fallback_path else None

    primary_index, primary_duplicates = build_capture_index(primary_records)
    fallback_index = None
    fallback_duplicates = []
    if fallback_records is not None:
        fallback_index, fallback_duplicates = build_capture_index(fallback_records)

    annotations = {}
    image_to_tracklet = {}
    missing = []
    duplicate_images = []
    source_hits = {"primary": 0, "fallback": 0}
    layer_counts = {"air": 0, "ground": 0}

    for layer_key, tracklet in iter_tracklets(tracklets):
        layer = str(tracklet.get("layer") or layer_key.replace("_tracklets", ""))
        for filename in tracklet.get("images", []):
            mapped_id, camera_id, frame_id = parse_image_name(filename)
            if camera_id != str(tracklet.get("camera_id")):
                raise ValueError("camera mismatch for %s in %s" % (filename, tracklet.get("tracklet_id")))
            if mapped_id != str(tracklet.get("mapped_id")):
                raise ValueError("mapped_id mismatch for %s in %s" % (filename, tracklet.get("tracklet_id")))

            key = (str(tracklet["vehicle_id"]), camera_id, frame_id)
            capture, source = find_capture(key, primary_index, fallback_index)
            if capture is None:
                missing.append({
                    "image": filename,
                    "tracklet_id": tracklet.get("tracklet_id"),
                    "capture_key": key,
                })
                continue

            if filename in annotations:
                duplicate_images.append(filename)

            annotations[filename] = make_annotation(
                tracklet, filename, mapped_id, camera_id, frame_id, capture
            )
            image_to_tracklet[filename] = {
                "tracklet_id": tracklet.get("tracklet_id"),
                "start_time": tracklet.get("start_time"),
                "end_time": tracklet.get("end_time"),
                "layer": layer,
            }
            source_hits[source] += 1
            if layer in layer_counts:
                layer_counts[layer] += 1

    output = {
        "description": DESCRIPTION,
        "version": VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "total_records": len(annotations),
        "annotations": annotations,
    }
    diagnostics = {
        "scope": infer_scope(ag_root, tracklets),
        "tracklets_path": tracklets_path,
        "captures_path": captures_path,
        "captures_fallback_path": fallback_path,
        "air_count": layer_counts["air"],
        "ground_count": layer_counts["ground"],
        "total_count": len(annotations),
        "missing_count": len(missing),
        "missing_samples": missing[:10],
        "duplicate_image_count": len(duplicate_images),
        "duplicate_image_samples": duplicate_images[:10],
        "primary_hit_count": source_hits["primary"],
        "fallback_hit_count": source_hits["fallback"],
        "primary_duplicate_key_count": len(primary_duplicates),
        "fallback_duplicate_key_count": len(fallback_duplicates),
        "primary_duplicate_key_samples": primary_duplicates[:10],
        "fallback_duplicate_key_samples": fallback_duplicates[:10],
        "image_to_tracklet": image_to_tracklet,
    }
    return output, diagnostics


def check_counts(scope, diagnostics):
    expected = EXPECTED_COUNTS.get(scope)
    if expected is None:
        return {"expected": None, "passed": None}
    actual = {
        "air": diagnostics["air_count"],
        "ground": diagnostics["ground_count"],
        "total": diagnostics["total_count"],
    }
    return {"expected": expected, "actual": actual, "passed": actual == expected}


def check_core_ground(output, core_path, limit=200, tolerance=1e-2):
    if not core_path:
        return {"checked": 0, "matched": 0, "mismatched": 0, "agreement_rate": None, "samples": []}
    core = load_json(core_path).get("annotations", {})
    checked = 0
    matched = 0
    mismatches = []
    for filename in sorted(output["annotations"]):
        annotation = output["annotations"][filename]
        if annotation.get("layer") != "ground" or filename not in core:
            continue
        checked += 1
        position = annotation.get("position", {})
        core_position = core[filename].get("position", {})
        ok = True
        for axis in ("x", "y", "z"):
            if abs(float(position[axis]) - float(core_position[axis])) > tolerance:
                ok = False
                break
        if ok:
            matched += 1
        else:
            mismatches.append({"image": filename, "ag": position, "core": core_position})
        if checked >= limit:
            break
    rate = (matched / checked) if checked else None
    return {
        "checked": checked,
        "matched": matched,
        "mismatched": checked - matched,
        "agreement_rate": rate,
        "samples": mismatches[:10],
    }


def check_timestamps(output, image_to_tracklet, tolerance=0.05):
    out_of_range = []
    checked = 0
    for filename in sorted(output["annotations"]):
        annotation = output["annotations"][filename]
        tracklet = image_to_tracklet[filename]
        timestamp = float(annotation["timestamp"])
        start_time = float(tracklet["start_time"])
        end_time = float(tracklet["end_time"])
        checked += 1
        if timestamp < start_time - tolerance or timestamp > end_time + tolerance:
            out_of_range.append({
                "image": filename,
                "timestamp": timestamp,
                "start_time": start_time,
                "end_time": end_time,
                "tracklet_id": tracklet["tracklet_id"],
            })
    return {
        "checked": checked,
        "out_of_range_count": len(out_of_range),
        "samples": out_of_range[:10],
    }


def build_validation(output, diagnostics, core_path):
    scope = diagnostics["scope"]
    validation = {
        "scope": scope,
        "missing": {
            "missing_count": diagnostics["missing_count"],
            "missing_samples": diagnostics["missing_samples"],
            "primary_hit_count": diagnostics["primary_hit_count"],
            "fallback_hit_count": diagnostics["fallback_hit_count"],
            "duplicate_image_count": diagnostics["duplicate_image_count"],
            "primary_duplicate_key_count": diagnostics["primary_duplicate_key_count"],
            "fallback_duplicate_key_count": diagnostics["fallback_duplicate_key_count"],
        },
        "counts": check_counts(scope, diagnostics),
        "core_ground": check_core_ground(output, core_path),
        "timestamps": check_timestamps(output, diagnostics["image_to_tracklet"]),
    }
    return validation


def print_summary(validation):
    counts = validation["counts"].get("actual") or {}
    core = validation["core_ground"]
    print("scope=%s" % validation["scope"])
    print(
        "missing=%d primary_hits=%d fallback_hits=%d duplicate_images=%d"
        % (
            validation["missing"]["missing_count"],
            validation["missing"]["primary_hit_count"],
            validation["missing"]["fallback_hit_count"],
            validation["missing"]["duplicate_image_count"],
        )
    )
    print(
        "counts air=%s ground=%s total=%s passed=%s"
        % (counts.get("air"), counts.get("ground"), counts.get("total"), validation["counts"].get("passed"))
    )
    print(
        "core_ground checked=%d matched=%d mismatched=%d agreement_rate=%s"
        % (core["checked"], core["matched"], core["mismatched"], core["agreement_rate"])
    )
    print(
        "timestamps checked=%d out_of_range=%d"
        % (validation["timestamps"]["checked"], validation["timestamps"]["out_of_range_count"])
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ag-root", required=True, help="Path to ag_protocol or ag_protocol_full directory")
    parser.add_argument("--captures", required=True, help="Primary captures.json path")
    parser.add_argument("--captures-fallback", help="Fallback captures_cleaned.json path")
    parser.add_argument("--core-spatiotemporal", help="Core spatiotemporal.json path for ground consistency check")
    parser.add_argument("--out", required=True, help="Output ag_spatiotemporal.json path")
    parser.add_argument("--validation-out", help="Optional validation summary JSON path")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation summary checks")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    output, diagnostics = export_annotations(args.ag_root, args.captures, args.captures_fallback)
    write_json(args.out, output)

    if args.no_validate:
        return 0

    validation = build_validation(output, diagnostics, args.core_spatiotemporal)
    if args.validation_out:
        write_json(args.validation_out, validation)
    print_summary(validation)

    failed = False
    failed = failed or validation["missing"]["missing_count"] != 0
    failed = failed or validation["missing"]["duplicate_image_count"] != 0
    failed = failed or validation["counts"].get("passed") is False
    failed = failed or validation["core_ground"]["checked"] < 200
    failed = failed or validation["core_ground"]["mismatched"] != 0
    failed = failed or validation["timestamps"]["out_of_range_count"] != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
