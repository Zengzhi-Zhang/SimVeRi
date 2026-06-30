# scripts/generate_image_manifest.py
"""Generate the release-wide image manifest for a SimVeRi dataset release (v1.1).

The manifest is a per-file index of every released image, written to
``statistics/image_manifest.csv`` with columns:

    resource, relative_path, filename, sha256, content_id,
    is_replicated_copy, filename_collision

It also prints the release-wide byte-level content-hash audit reported in the
Data Records section, namely the number of physical JPEG files, the number of
unique image contents, the number of distinct filename strings, and the
cross-resource filename collisions (a filename that maps to more than one
distinct image content because the Twins supplement uses an independent identity
mapping).

Run on the extracted release package:

    python scripts/generate_image_manifest.py --dataset-root /path/to/SimVeRi

Columns:
    content_id           stable id shared by byte-identical images (distinct
                         contents are numbered by first appearance)
    is_replicated_copy   "true" if this content also exists in another physical
                         file (e.g. a core frame copied into an air-ground package)
    filename_collision   "true" if this basename maps to more than one distinct
                         content across resources (not globally unique)
"""

import os
import sys
import csv
import hashlib
import argparse
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.path_utils import get_default_simveri_root


# Image resources and their directories, relative to the release root. The order
# is fixed so the manifest (and the content_id numbering) is reproducible.
IMAGE_GROUPS = [
    ("core", [
        "images/train",
        "images/gallery",
        "images/query",
    ]),
    ("twins", [
        "extras/twins/images",
    ]),
    ("ag_test", [
        "annotations_and_protocols/ag_protocol/images/air",
        "annotations_and_protocols/ag_protocol/images/ground",
    ]),
    ("ag_full", [
        "annotations_and_protocols/ag_protocol_full/images/air",
        "annotations_and_protocols/ag_protocol_full/images/ground",
    ]),
]

FIELDNAMES = [
    "resource", "relative_path", "filename", "sha256",
    "content_id", "is_replicated_copy", "filename_collision",
]


def sha256_of(path, chunk_size=1 << 18):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def collect_records(root_dir):
    """Return a list of (resource, relative_path, filename, sha256) for every
    released .jpg, in a deterministic order."""
    records = []
    for resource, subdirs in IMAGE_GROUPS:
        for sub in subdirs:
            directory = os.path.join(root_dir, sub)
            if not os.path.isdir(directory):
                continue
            for filename in sorted(os.listdir(directory)):
                if not filename.lower().endswith(".jpg"):
                    continue
                full = os.path.join(directory, filename)
                rel = os.path.relpath(full, root_dir).replace(os.sep, "/")
                records.append((resource, rel, filename, sha256_of(full)))
    return records


def write_manifest(records, output_path):
    content_id = {}
    for _, _, _, digest in records:
        content_id.setdefault(digest, len(content_id) + 1)

    hash_counts = Counter(digest for *_, digest in records)
    name_to_hashes = {}
    for _, _, filename, digest in records:
        name_to_hashes.setdefault(filename, set()).add(digest)
    collision_names = {n for n, hs in name_to_hashes.items() if len(hs) > 1}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        for resource, rel, filename, digest in records:
            writer.writerow([
                resource, rel, filename, digest, content_id[digest],
                "true" if hash_counts[digest] > 1 else "false",
                "true" if filename in collision_names else "false",
            ])

    return content_id, hash_counts, collision_names


def main():
    parser = argparse.ArgumentParser(
        description="Generate the release-wide image manifest for a SimVeRi release."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=get_default_simveri_root(),
        help="Root directory of the extracted SimVeRi dataset release.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <dataset-root>/statistics/image_manifest.csv).",
    )
    args = parser.parse_args()

    root_dir = args.dataset_root
    output_path = args.output or os.path.join(root_dir, "statistics", "image_manifest.csv")

    if not os.path.isdir(root_dir):
        print(f"[ERROR] dataset root not found: {root_dir}")
        return 1

    print(f"Scanning images under: {root_dir}")
    records = collect_records(root_dir)
    if not records:
        print("[ERROR] no .jpg images found; check --dataset-root")
        return 1

    content_id, hash_counts, collision_names = write_manifest(records, output_path)

    distinct_filenames = len({r[2] for r in records})
    distinct_contents = len(content_id)
    replicated_files = sum(1 for c in hash_counts.values() for _ in range(c) if c > 1)

    print(f"[OK] Wrote manifest: {output_path}")
    print("-" * 30)
    print(f"Physical JPEG files     : {len(records)}")
    print(f"Unique image contents   : {distinct_contents}")
    print(f"Distinct filename strings: {distinct_filenames}")
    print(f"Replicated-copy files   : {replicated_files}")
    print(f"Filename collisions     : {len(collision_names)} "
          f"{sorted(collision_names) if collision_names else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
