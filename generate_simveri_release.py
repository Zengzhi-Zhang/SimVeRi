"""
SimVeRi release dataset generation tool v3.2
Patched version - multi-camera Query selection and Query exclusion from Gallery

Main fixes:
1. Select one Query image per camera (aligned with the VeRi protocol)
2. Exclude Query images from Gallery to avoid inflated Rank-1 scores
3. Add image-size filtering
"""

import os
import json
import random
import shutil
import csv
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm


# =============================================================================
# =============================================================================

INPUT_DIR = "output_cleaned_test"
INPUT_JSON = f"{INPUT_DIR}/metadata/captures_cleaned.json"
INPUT_IMAGE_DIR = f"{INPUT_DIR}/image_train"

CAMERA_TOPOLOGY_FILE = "output/metadata/camera_topology.json"

OUTPUT_DIR = "SimVeRi_v2_20260123"

CLEAN_OUTPUT_DIR = True

TRAIN_RATIO = 0.70

EXCLUDE_TWINS_IN_TEST = True

GALLERY_MAX_PER_CAMERA = 6

MIN_QUERY_WIDTH = 96
MIN_QUERY_HEIGHT = 96

# Air-Ground protocol (MVP): keep ground benchmark unchanged by default.
INCLUDE_AIR_IN_GROUND_RELEASE = False
GENERATE_AIR_GROUND_PROTOCOL = True
AIR_GROUND_TIME_TOLERANCE_S = 1.0

RANDOM_SEED = 42

DATASET_VERSION = "2.0"
DATASET_NAME = "SimVeRi"
DATASET_FULL_NAME = "Simulated Vehicle Re-Identification Dataset"


# =============================================================================
# =============================================================================

SIMVERI_COLOR_FAMILIES = {
    'white': ['white', 'warm_white', 'cool_white', 'beige'],
    'black': ['black'],
    'gray': ['gray', 'dark_gray', 'silver'],
    'red': ['red', 'dark_red', 'burgundy'],
    'blue': ['blue', 'dark_blue', 'sky_blue'],
    'yellow': ['yellow'],
    'green': ['green'],
    'brown': ['brown'],
}

COLOR_TO_FAMILY_ID = {}
for family_id, (family_name, colors) in enumerate(SIMVERI_COLOR_FAMILIES.items(), start=1):
    for color in colors:
        COLOR_TO_FAMILY_ID[color] = family_id

COLOR_TO_FAMILY_NAME = {}
for family_name, colors in SIMVERI_COLOR_FAMILIES.items():
    for color in colors:
        COLOR_TO_FAMILY_NAME[color] = family_name

SIMVERI_VEHICLE_TYPES = {
    1: 'sedan',
    2: 'suv',
    3: 'hatchback',
    4: 'van',
    5: 'coupe',
    6: 'special',
}

TYPE_TO_ID = {v: k for k, v in SIMVERI_VEHICLE_TYPES.items()}


# =============================================================================
# =============================================================================

def get_color_family_id(color_name):
    if not color_name:
        return 0
    return COLOR_TO_FAMILY_ID.get(color_name.lower().strip(), 0)


def get_color_family_name(color_name):
    if not color_name:
        return 'unknown'
    return COLOR_TO_FAMILY_NAME.get(color_name.lower().strip(), 'unknown')


def get_type_id(type_name):
    if not type_name:
        return 0
    return TYPE_TO_ID.get(type_name.lower().strip(), 0)


@dataclass
class VehicleIdMappingResult:
    """Vehicle ID mapping result (v3.5)."""
    mapping: dict          # vehicle_id -> "NNNN"
    base_range: str        # e.g. "0001-0600"
    occlusion_range: str   # e.g. "0601-0650"
    twins_range: str       # e.g. "0651-0775"


def create_vehicle_id_mapping(captures):
    """Create the vehicle-ID mapping with ordered assignment to avoid conflicts."""
    vehicle_ids = sorted(set(cap['vehicle_id'] for cap in captures))

    base_ids = [v for v in vehicle_ids if v.startswith('base_')]
    occ_ids = [v for v in vehicle_ids if v.startswith('occ_')]
    twins_ids = [v for v in vehicle_ids if v.startswith('H')]
    other_ids = [v for v in vehicle_ids
                 if not v.startswith('base_') and not v.startswith('occ_') and not v.startswith('H')]

    if other_ids:
        print(f"  [WARN] Found {len(other_ids)} vehicles with unknown prefixes; assigning them to the Base range: {other_ids[:5]}")

    mapping = {}
    counter = 1
    for vid in base_ids + other_ids:
        mapping[vid] = f"{counter:04d}"
        counter += 1
    base_end = counter - 1

    occ_start = counter
    for vid in occ_ids:
        mapping[vid] = f"{counter:04d}"
        counter += 1
    occ_end = counter - 1

    twins_start = counter
    for vid in twins_ids:
        mapping[vid] = f"{counter:04d}"
        counter += 1
    twins_end = counter - 1

    return VehicleIdMappingResult(
        mapping=mapping,
        base_range=f"{1:04d}-{base_end:04d}" if base_end >= 1 else "N/A",
        occlusion_range=f"{occ_start:04d}-{occ_end:04d}" if occ_end >= occ_start else "N/A",
        twins_range=f"{twins_start:04d}-{twins_end:04d}" if twins_end >= twins_start else "N/A",
    )


def generate_filename(vid_mapped, camera_id, frame_id):
    """Generate a SimVeRi-format filename."""
    cam_num = camera_id.replace('c', '').replace('C', '')
    return f"{vid_mapped}_c{cam_num}_{frame_id:06d}.jpg"


def get_image_id(cap):
    """Return the unique identifier for an image."""
    return cap.get('image_path') or cap.get('image_name') or f"{cap['vehicle_id']}_{cap['camera_id']}_{cap['frame_id']}"


def load_camera_topology(topology_path):
    """
    Load the camera-topology file and extract true distances and positions.
    """
    topology_data = {
        'cameras': {},
        'distance_matrix': {},
        'metadata': {}
    }
    
    if not os.path.exists(topology_path):
        print(f"  Warning: camera-topology file not found: {topology_path}")
        return topology_data
    
    try:
        with open(topology_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        topology_data['metadata'] = raw_data.get('metadata', {})
        topology_data['cameras'] = raw_data.get('cameras', {})
        topology_data['distance_matrix'] = raw_data.get('distance_matrix', {})
        topology_data['elevated_distances'] = raw_data.get('elevated_distances_predefined', {})
        
        print(f"  Loaded camera topology: {len(topology_data['cameras'])} cameras")
        print(f"  Loaded distance matrix: {len(topology_data['distance_matrix'])} x {len(topology_data['distance_matrix'])}")
        
    except Exception as e:
        print(f"  Failed to load the topology file: {e}")
    
    return topology_data


# =============================================================================
# =============================================================================

def split_dataset(captures, train_ratio=0.57):
    """
    Split the dataset into Train/Gallery/Query.
    
    Patched protocol:
    1. Select one Query image per camera to match the VeRi protocol.
    2. Exclude Query images from Gallery to avoid inflated Rank-1 scores.
    3. Add image-size filtering.
    """
    random.seed(RANDOM_SEED)
    
    vehicle_captures = defaultdict(list)
    for cap in captures:
        vehicle_captures[cap['vehicle_id']].append(cap)
    
    #
    # IMPORTANT: sort before shuffling so the split is stable even if `captures` order changes
    # (e.g. due to different cleaning/sampling passes).
    twins_vehicles = sorted([vid for vid in vehicle_captures.keys() if vid.startswith('H')])
    other_vehicles = sorted([vid for vid in vehicle_captures.keys() if not vid.startswith('H')])
    
    random.shuffle(other_vehicles)
    
    total_other = len(other_vehicles)
    n_train_other = int(total_other * train_ratio)
    
    train_vehicles = set(other_vehicles[:n_train_other])
    if EXCLUDE_TWINS_IN_TEST:
        test_vehicles = set(other_vehicles[n_train_other:])
    else:
        test_vehicles = set(other_vehicles[n_train_other:]) | set(twins_vehicles)
    
    twins_note = "without Twins" if EXCLUDE_TWINS_IN_TEST else f"with {len(twins_vehicles)} Twins"
    print(f"  Vehicle split: Train={len(train_vehicles)}, Test={len(test_vehicles)} ({twins_note})")
    
    train_caps = []
    gallery_caps = []
    query_caps = []
    
    query_stats = {
        'total_cams': 0,
        'valid_cams': 0,
        'skipped_small': 0,
        'skipped_no_valid': 0
    }
    
    # ----------------------------------------------------------------------------
    for vid in train_vehicles:
        train_caps.extend(vehicle_captures[vid])
    
    # ----------------------------------------------------------------------------
    for vid in test_vehicles:
        caps = vehicle_captures[vid]
        
        cam_groups = defaultdict(list)
        for cap in caps:
            cam_groups[cap['camera_id']].append(cap)
        
        selected_query_ids = set()
        
        # ----------------------------------------------------------------------------
        for cam_id, cam_caps in cam_groups.items():
            query_stats['total_cams'] += 1
            
            valid_caps = []
            for cap in cam_caps:
                bbox = cap.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    if bbox[2] > bbox[0]:
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                    else:
                        w = bbox[2]
                        h = bbox[3]
                    
                    if w >= MIN_QUERY_WIDTH and h >= MIN_QUERY_HEIGHT:
                        valid_caps.append(cap)
                    else:
                        query_stats['skipped_small'] += 1
                else:
                    valid_caps.append(cap)
            
            if not valid_caps:
                query_stats['skipped_no_valid'] += 1
                continue
            
            query_stats['valid_cams'] += 1
            
            valid_caps_sorted = sorted(valid_caps, key=lambda x: x.get('frame_id', 0))
            mid_idx = len(valid_caps_sorted) // 2
            query_cap = valid_caps_sorted[mid_idx]
            
            query_caps.append(query_cap)
            
            query_id = get_image_id(query_cap)
            selected_query_ids.add(query_id)
        
        # ----------------------------------------------------------------------------
        for cap in caps:
            cap_id = get_image_id(cap)
            if cap_id not in selected_query_ids:
                gallery_caps.append(cap)
    
    train_caps.sort(key=lambda x: (x['vehicle_id'], x['camera_id'], x['frame_id']))
    gallery_caps.sort(key=lambda x: (x['vehicle_id'], x['camera_id'], x['frame_id']))
    query_caps.sort(key=lambda x: (x['vehicle_id'], x['camera_id'], x['frame_id']))
    
    print(f"  Query selection summary:")
    print(f"    Total camera groups: {query_stats['total_cams']}")
    print(f"    Valid camera groups: {query_stats['valid_cams']}")
    print(f"    Skipped (image too small): {query_stats['skipped_small']}")
    print(f"    Skipped (no valid image): {query_stats['skipped_no_valid']}")
    print(f"  Final dataset:")
    print(f"    Train: {len(train_caps)}")
    print(f"    Gallery: {len(gallery_caps)}")
    print(f"    Query: {len(query_caps)}")
    print(f"    (Query not in Gallery [OK])")
    
    return train_caps, gallery_caps, query_caps


def limit_gallery_per_camera(gallery_caps, max_per_camera):
    """Limit Gallery images for each (vehicle, camera) pair."""
    if not max_per_camera or max_per_camera <= 0:
        return gallery_caps

    grouped = defaultdict(list)
    for cap in gallery_caps:
        key = (cap['vehicle_id'], cap['camera_id'])
        grouped[key].append(cap)

    sampled = []
    total_before = len(gallery_caps)
    total_after = 0

    for key, caps in grouped.items():
        caps_sorted = sorted(caps, key=lambda x: x.get('frame_id', 0))
        if len(caps_sorted) > max_per_camera:
            indices = np.linspace(0, len(caps_sorted) - 1, max_per_camera, dtype=int)
            selected = [caps_sorted[i] for i in indices]
        else:
            selected = caps_sorted
        sampled.extend(selected)
        total_after += len(selected)

    print(f"  Gallery cap: {total_before} -> {total_after} (max_per_camera={max_per_camera})")
    return sampled


# =============================================================================
# =============================================================================

def generate_query_list(captures, output_path, vid_mapping):
    """Generate query_list.txt."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for cap in captures:
            vid = vid_mapping[cap['vehicle_id']]
            cam = cap['camera_id']
            frame = cap['frame_id']
            filename = generate_filename(vid, cam, frame)
            f.write(filename + '\n')
    
    print(f"  Generated: {output_path} ({len(captures)} lines)")


def generate_annotations_xml(captures, output_path, vid_mapping, set_name):
    """Generate a SimVeRi-format XML annotation file."""
    root = ET.Element("SimVeRi")
    root.set("version", DATASET_VERSION)
    root.set("set", set_name)
    root.set("count", str(len(captures)))
    
    vehicles = ET.SubElement(root, "Vehicles")
    
    for cap in captures:
        vehicle = ET.SubElement(vehicles, "Vehicle")
        
        vid = vid_mapping[cap['vehicle_id']]
        cam = cap['camera_id']
        frame = cap['frame_id']
        filename = generate_filename(vid, cam, frame)
        
        vehicle.set("id", vid)
        vehicle.set("camera", cam)
        vehicle.set("frame", str(frame))
        
        image = ET.SubElement(vehicle, "Image")
        image.text = filename
        
        attributes = ET.SubElement(vehicle, "Attributes")
        color_name = cap.get('color_name', '')
        attributes.set("color", color_name)
        attributes.set("color_family", get_color_family_name(color_name))
        attributes.set("color_family_id", str(get_color_family_id(color_name)))
        attributes.set("type", cap.get('category', ''))
        attributes.set("type_id", str(get_type_id(cap.get('category', ''))))
        attributes.set("brand", cap.get('brand', ''))
        
        spatiotemporal = ET.SubElement(vehicle, "Spatiotemporal")
        spatiotemporal.set("timestamp", f"{cap.get('timestamp', 0):.2f}")
        spatiotemporal.set("x", f"{cap.get('global_x', 0):.2f}")
        spatiotemporal.set("y", f"{cap.get('global_y', 0):.2f}")
        spatiotemporal.set("z", f"{cap.get('global_z', 0):.2f}")
        spatiotemporal.set("speed", f"{cap.get('speed', 0):.2f}")
        spatiotemporal.set("heading", f"{cap.get('heading', 0):.2f}")
        
        quality = ET.SubElement(vehicle, "Quality")
        quality.set("occlusion", f"{cap.get('occlusion_ratio', 0):.3f}")
        quality.set("distance", f"{cap.get('distance', 0):.2f}")
        bbox = cap.get('bbox', [0, 0, 0, 0])
        quality.set("bbox", f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        
        if cap.get('is_fleet', False):
            twins = ET.SubElement(vehicle, "Twins")
            twins.set("group", cap.get('fleet_id', '').replace('fleet_', 'group_'))
    
    xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
    lines = [line for line in xml_str.split('\n') if line.strip()]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  Generated: {output_path} ({len(captures)} records)")


def generate_ground_truth(query_caps, gallery_caps, output_path, vid_mapping):
    """Generate ground_truth.txt."""
    gallery_by_vehicle = defaultdict(list)
    for idx, cap in enumerate(gallery_caps, start=1):
        gallery_by_vehicle[cap['vehicle_id']].append({
            'index': idx,
            'camera_id': cap['camera_id']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for query in query_caps:
            q_vid = query['vehicle_id']
            q_cam = query['camera_id']
            
            gt_indices = []
            for item in gallery_by_vehicle.get(q_vid, []):
                if item['camera_id'] != q_cam:
                    gt_indices.append(item['index'])
            
            gt_indices.sort()
            f.write(' '.join(map(str, gt_indices)) + '\n')
    
    print(f"  Generated: {output_path} ({len(query_caps)} lines)")


def generate_ignore_list(query_caps, gallery_caps, output_path):
    """Generate ignore_list.txt."""
    gallery_by_vehicle_camera = defaultdict(list)
    for idx, cap in enumerate(gallery_caps, start=1):
        key = (cap['vehicle_id'], cap['camera_id'])
        gallery_by_vehicle_camera[key].append(idx)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for query in query_caps:
            key = (query['vehicle_id'], query['camera_id'])
            ignore_indices = sorted(gallery_by_vehicle_camera.get(key, []))
            f.write(' '.join(map(str, ignore_indices)) + '\n')
    
    print(f"  Generated: {output_path} ({len(query_caps)} lines)")


# =============================================================================
# =============================================================================

def generate_vehicle_attributes(captures, output_path):
    """Generate vehicle_attributes.json."""
    color_dist = defaultdict(int)
    color_family_dist = defaultdict(int)
    type_dist = defaultdict(int)
    
    for cap in captures:
        color_name = cap.get('color_name', 'unknown')
        color_dist[color_name] += 1
        color_family_dist[get_color_family_name(color_name)] += 1
        type_dist[cap.get('category', 'unknown')] += 1
    
    attributes = {
        "description": f"{DATASET_NAME} vehicle attribute definitions",
        "version": DATASET_VERSION,
        "color_system": {
            "description": "SimVeRi uses 17 fine-grained colors grouped into 8 color families",
            "families": {
                family: {
                    "id": idx,
                    "members": colors,
                    "count": sum(color_dist.get(c, 0) for c in colors)
                }
                for idx, (family, colors) in enumerate(SIMVERI_COLOR_FAMILIES.items(), start=1)
            },
            "color_to_family": COLOR_TO_FAMILY_NAME,
            "distribution_by_color": dict(sorted(color_dist.items(), key=lambda x: -x[1])),
            "distribution_by_family": dict(sorted(color_family_dist.items(), key=lambda x: -x[1]))
        },
        "type_system": {
            "description": "SimVeRi vehicle type categories",
            "types": SIMVERI_VEHICLE_TYPES,
            "distribution": dict(sorted(type_dist.items(), key=lambda x: -x[1]))
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attributes, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated: {output_path}")


def generate_spatiotemporal_json(captures, output_path, vid_mapping):
    """Generate spatiotemporal.json."""
    annotations = {}
    
    for cap in captures:
        vid = vid_mapping[cap['vehicle_id']]
        cam = cap['camera_id']
        frame = cap['frame_id']
        filename = generate_filename(vid, cam, frame)
        
        annotations[filename] = {
            "vehicle_id": vid,
            "original_id": cap['vehicle_id'],
            "camera_id": cam,
            "frame_id": frame,
            "timestamp": round(cap.get('timestamp', 0), 2),
            "position": {
                "x": round(cap.get('global_x', 0), 2),
                "y": round(cap.get('global_y', 0), 2),
                "z": round(cap.get('global_z', 0), 2)
            },
            "motion": {
                "speed_kmh": round(cap.get('speed', 0), 2),
                "heading_deg": round(cap.get('heading', 0), 2)
            },
            "quality": {
                "occlusion_ratio": round(cap.get('occlusion_ratio', 0), 3),
                "distance_m": round(cap.get('distance', 0), 2)
            }
        }
    
    output_data = {
        "description": "Spatiotemporal annotations for trajectory-aware vehicle re-identification",
        "version": DATASET_VERSION,
        "coordinate_system": "CARLA world coordinates (meters)",
        "total_records": len(annotations),
        "annotations": annotations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated: {output_path} ({len(annotations)} records)")


def generate_camera_network(output_path, topology_data):
    """
    Generate camera_network.json.
    Use true camera positions and the true distance matrix.
    """
    camera_ids = sorted(topology_data.get('cameras', {}).keys(), 
                        key=lambda x: int(x.replace('c', '').replace('C', '')))
    
    cameras = {}
    for cam_id in camera_ids:
        cam_data = topology_data['cameras'].get(cam_id, {})
        position = cam_data.get('position', [0, 0, 0])
        rotation = cam_data.get('rotation', {})
        
        cameras[cam_id] = {
            "position": {
                "x": position[0] if isinstance(position, list) else position.get('x', 0),
                "y": position[1] if isinstance(position, list) else position.get('y', 0),
                "z": position[2] if isinstance(position, list) else position.get('z', 0)
            },
            "rotation": {
                "pitch": rotation.get('pitch', 0),
                "yaw": rotation.get('yaw', 0),
                "roll": rotation.get('roll', 0)
            },
            "fov": cam_data.get('fov', 90),
            "type": cam_data.get('type', 'unknown'),
            "layer": cam_data.get('layer', 'ground'),
            "description": cam_data.get('description', ''),
            "monitor_edge_id": cam_data.get('monitor_edge_id', '')
        }
    
    distance_matrix = {}
    raw_matrix = topology_data.get('distance_matrix', {})
    
    for from_cam in camera_ids:
        distance_matrix[from_cam] = {}
        for to_cam in camera_ids:
            if from_cam == to_cam:
                distance_matrix[from_cam][to_cam] = 0.0
            elif from_cam in raw_matrix and to_cam in raw_matrix.get(from_cam, {}):
                dist = raw_matrix[from_cam][to_cam]
                distance_matrix[from_cam][to_cam] = dist if dist >= 0 else -1
            else:
                distance_matrix[from_cam][to_cam] = -1
    
    network = {
        "description": "Camera network topology for SimVeRi dataset",
        "version": DATASET_VERSION,
        "coordinate_system": "CARLA world coordinates (meters)",
        "distance_unit": "meters",
        "distance_note": "Distances are route-based (following traffic rules), -1 means unreachable",
        "camera_count": len(cameras),
        "cameras": cameras,
        "distance_matrix": distance_matrix,
        "metadata": topology_data.get('metadata', {})
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(network, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated: {output_path} ({len(cameras)} cameras, true distance matrix)")


def generate_twins_groups(captures, output_path, vid_mapping):
    """Generate twins_groups.json."""
    groups = defaultdict(lambda: {
        'vehicles': [],
        'blueprint': '',
        'color': '',
        'color_family': '',
        'image_count': 0
    })
    
    for cap in captures:
        if cap.get('is_fleet', False) and cap.get('fleet_id'):
            fleet_id = cap['fleet_id']
            group_id = fleet_id.replace('fleet_', 'group_')
            vid = cap['vehicle_id']
            
            if vid not in groups[group_id]['vehicles']:
                groups[group_id]['vehicles'].append(vid)
                groups[group_id]['blueprint'] = cap.get('blueprint', '')
                groups[group_id]['color'] = cap.get('color_name', '')
                groups[group_id]['color_family'] = get_color_family_name(cap.get('color_name', ''))
            
            groups[group_id]['image_count'] += 1
    
    for group_id, group_data in groups.items():
        group_data['mapped_ids'] = [vid_mapping.get(v, v) for v in group_data['vehicles']]
    
    output_data = {
        "description": "Twins subset - groups of vehicles with identical appearance",
        "version": DATASET_VERSION,
        "purpose": "Challenge visual-only re-identification with identical-looking vehicles",
        "total_groups": len(groups),
        "vehicles_per_group": 5,
        "total_vehicles": sum(len(g['vehicles']) for g in groups.values()),
        "total_images": sum(g['image_count'] for g in groups.values()),
        "groups": dict(groups)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated: {output_path} ({len(groups)} groups)")


def generate_trajectory_info(captures, output_path, vid_mapping):
    """Generate trajectory_info.csv."""
    tracks = defaultdict(list)
    for cap in captures:
        key = (cap['vehicle_id'], cap['camera_id'])
        tracks[key].append(cap)
    
    for key in tracks:
        tracks[key].sort(key=lambda x: x.get('timestamp', 0))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id', 'vehicle_id', 'mapped_id', 'camera_id',
            'start_time', 'end_time', 'duration', 'image_count',
            'start_x', 'start_y', 'end_x', 'end_y',
            'avg_speed', 'is_twins'
        ])
        
        track_id = 0
        for (vid, cam), caps in sorted(tracks.items()):
            vid_mapped = vid_mapping.get(vid, vid)
            
            start_cap = caps[0]
            end_cap = caps[-1]
            
            start_time = start_cap.get('timestamp', 0)
            end_time = end_cap.get('timestamp', 0)
            duration = end_time - start_time
            
            avg_speed = np.mean([c.get('speed', 0) for c in caps])
            
            writer.writerow([
                f"track_{track_id:04d}",
                vid,
                vid_mapped,
                cam,
                f"{start_time:.2f}",
                f"{end_time:.2f}",
                f"{duration:.2f}",
                len(caps),
                f"{start_cap.get('global_x', 0):.2f}",
                f"{start_cap.get('global_y', 0):.2f}",
                f"{end_cap.get('global_x', 0):.2f}",
                f"{end_cap.get('global_y', 0):.2f}",
                f"{avg_speed:.2f}",
                str(vid.startswith('H')).lower()
            ])
            track_id += 1
    
    print(f"  Generated: {output_path} ({track_id} tracks)")


def generate_camera_transitions(captures, output_path, topology_data):
    """
    Generate inter-camera transition-time statistics.
    Combine the true distance matrix with the actual capture records.
    """
    vehicle_captures = defaultdict(list)
    for cap in captures:
        vehicle_captures[cap['vehicle_id']].append(cap)
    
    transitions = defaultdict(list)
    
    for vid, caps in vehicle_captures.items():
        caps_sorted = sorted(caps, key=lambda x: x.get('timestamp', 0))
        
        for i in range(len(caps_sorted) - 1):
            cam1 = caps_sorted[i]['camera_id']
            cam2 = caps_sorted[i + 1]['camera_id']
            t1 = caps_sorted[i].get('timestamp', 0)
            t2 = caps_sorted[i + 1].get('timestamp', 0)
            
            if cam1 != cam2 and t2 > t1:
                time_diff = t2 - t1
                if 0 < time_diff < 300:
                    transitions[(cam1, cam2)].append(time_diff)
    
    distance_matrix = topology_data.get('distance_matrix', {})
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'from_camera', 'to_camera', 
            'route_distance_m',
            'mean_time_s', 'std_time_s', 'min_time_s', 'max_time_s', 
            'avg_speed_kmh',
            'sample_count'
        ])
        
        for (cam1, cam2) in sorted(transitions.keys()):
            times = transitions[(cam1, cam2)]
            if len(times) >= 2:
                mean_time = np.mean(times)
                
                route_distance = -1
                if cam1 in distance_matrix and cam2 in distance_matrix.get(cam1, {}):
                    route_distance = distance_matrix[cam1].get(cam2, -1)
                
                avg_speed = -1
                if route_distance > 0 and mean_time > 0:
                    avg_speed = (route_distance / mean_time) * 3.6
                
                writer.writerow([
                    cam1, cam2,
                    f"{route_distance:.2f}" if route_distance >= 0 else "-1",
                    f"{mean_time:.2f}",
                    f"{np.std(times):.2f}",
                    f"{np.min(times):.2f}",
                    f"{np.max(times):.2f}",
                    f"{avg_speed:.2f}" if avg_speed >= 0 else "-1",
                    len(times)
                ])
    
    print(f"  Generated: {output_path} ({len(transitions)} transition records with true path distances)")


# =============================================================================
# =============================================================================

def generate_dataset_summary(captures, train_caps, gallery_caps, query_caps, output_path, vid_mapping, topology_data, vid_mapping_result=None):
    """Generate dataset_summary.json."""
    all_vehicles = set(cap['vehicle_id'] for cap in captures)
    train_vehicles = set(cap['vehicle_id'] for cap in train_caps)
    gallery_vehicles = set(cap['vehicle_id'] for cap in gallery_caps)
    query_vehicles = set(cap['vehicle_id'] for cap in query_caps)
    all_cameras = set(cap['camera_id'] for cap in captures)
    
    twins_caps = [cap for cap in captures if cap.get('is_fleet', False)]
    twins_vehicles = set(cap['vehicle_id'] for cap in twins_caps)
    
    vehicle_cameras = defaultdict(set)
    for cap in captures:
        vehicle_cameras[cap['vehicle_id']].add(cap['camera_id'])
    
    cross_camera_dist = defaultdict(int)
    for vid, cams in vehicle_cameras.items():
        cross_camera_dist[len(cams)] += 1
    
    query_per_vehicle = defaultdict(int)
    for cap in query_caps:
        query_per_vehicle[cap['vehicle_id']] += 1
    avg_query_per_vehicle = np.mean(list(query_per_vehicle.values())) if query_per_vehicle else 0
    
    color_dist = defaultdict(int)
    color_family_dist = defaultdict(int)
    type_dist = defaultdict(int)
    
    for cap in captures:
        color_name = cap.get('color_name', 'unknown')
        color_dist[color_name] += 1
        color_family_dist[get_color_family_name(color_name)] += 1
        type_dist[cap.get('category', 'unknown')] += 1
    
    distance_matrix = topology_data.get('distance_matrix', {})
    all_distances = []
    for from_cam, to_cams in distance_matrix.items():
        for to_cam, dist in to_cams.items():
            if from_cam != to_cam and dist > 0:
                all_distances.append(dist)
    
    summary = {
        "dataset_info": {
            "name": DATASET_NAME,
            "full_name": DATASET_FULL_NAME,
            "version": DATASET_VERSION,
            "generated_at": datetime.now().isoformat(),
            "description": "A synthetic vehicle re-identification dataset with rich spatiotemporal annotations"
        },
        "statistics": {
            "total_images": len(captures),
            "train_images": len(train_caps),
            "gallery_images": len(gallery_caps),
            "query_images": len(query_caps),
            "total_vehicles": len(all_vehicles),
            "train_vehicles": len(train_vehicles),
            "gallery_vehicles": len(gallery_vehicles),
            "query_vehicles": len(query_vehicles),
            "cameras": len(all_cameras),
            "avg_cameras_per_vehicle": round(np.mean([len(v) for v in vehicle_cameras.values()]), 2),
            "avg_images_per_vehicle": round(len(captures) / len(all_vehicles), 2),
            "avg_query_per_vehicle": round(avg_query_per_vehicle, 2)
        },
        "query_protocol": {
            "description": "Multi-camera query protocol (VeRi standard)",
            "method": "One query per camera per vehicle",
            "query_in_gallery": False,
            "total_query": len(query_caps),
            "query_vehicles": len(query_vehicles),
            "avg_query_per_vehicle": round(avg_query_per_vehicle, 2)
        },
        "camera_network": {
            "num_cameras": len(topology_data.get('cameras', {})),
            "distance_matrix_source": "CARLA GlobalRoutePlanner (route-based)",
            "min_distance_m": round(min(all_distances), 2) if all_distances else 0,
            "max_distance_m": round(max(all_distances), 2) if all_distances else 0,
            "avg_distance_m": round(np.mean(all_distances), 2) if all_distances else 0
        },
        "twins_subset": {
            "description": "Vehicles with identical appearance for challenging re-identification",
            "total_images": len(twins_caps),
            "total_vehicles": len(twins_vehicles),
            "num_groups": len(set(cap.get('fleet_id') for cap in twins_caps if cap.get('fleet_id'))),
            "vehicles_per_group": 5
        },
        "subsets": {
            "base": {
                "description": "Standard vehicles with diverse appearances",
                "vehicles": len([v for v in all_vehicles if v.startswith('base_')]),
                "images": len([c for c in captures if c['vehicle_id'].startswith('base_')])
            },
            "occlusion": {
                "description": "Vehicles with varying occlusion levels",
                "vehicles": len([v for v in all_vehicles if v.startswith('occ_')]),
                "images": len([c for c in captures if c['vehicle_id'].startswith('occ_')])
            },
            "twins": {
                "description": "Groups of identical-looking vehicles",
                "vehicles": len([v for v in all_vehicles if v.startswith('H')]),
                "images": len([c for c in captures if c['vehicle_id'].startswith('H')])
            }
        },
        "distributions": {
            "color": dict(sorted(color_dist.items(), key=lambda x: -x[1])),
            "color_family": dict(sorted(color_family_dist.items(), key=lambda x: -x[1])),
            "type": dict(sorted(type_dist.items(), key=lambda x: -x[1])),
            "cross_camera": dict(sorted(cross_camera_dist.items()))
        },
        "vehicle_id_mapping": {
            "base_range": vid_mapping_result.base_range if vid_mapping_result else "N/A",
            "occlusion_range": vid_mapping_result.occlusion_range if vid_mapping_result else "N/A",
            "twins_range": vid_mapping_result.twins_range if vid_mapping_result else "N/A",
            "total_mapped": len(vid_mapping)
        },
        "unique_features": [
            "Rich spatiotemporal annotations (position, speed, heading)",
            "Pixel-level occlusion ratio for each image",
            "Twins subset with identical-appearance vehicle groups",
            "Camera network topology with real route-based distances",
            "Ground truth trajectories for trajectory reconstruction research",
            "Multi-camera query protocol (VeRi standard compliant)"
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated: {output_path}")
    
    return summary


# =============================================================================
# Air-Ground protocol (track-level)
# =============================================================================

def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float, tol_s: float) -> bool:
    """Interval overlap with tolerance (treat small gaps as overlap)."""
    return (a_start <= (b_end + tol_s)) and (b_start <= (a_end + tol_s))


def build_tracklets(captures, vid_mapping, layer: str):
    """
    Build per-(vehicle_id, camera_id) tracklets with renamed image names (release filenames).

    Returns:
        tracklets: dict(tracklet_id -> info)
    """
    tracks = defaultdict(list)
    for cap in captures:
        vid = cap.get("vehicle_id")
        cam_id = cap.get("camera_id")
        if not vid or not cam_id:
            continue
        if vid not in vid_mapping:
            continue
        tracks[(vid, cam_id)].append(cap)

    tracklets = {}
    for (vid, cam_id), items in tracks.items():
        items_sorted = sorted(items, key=lambda x: x.get("timestamp", 0))
        mapped_vid = vid_mapping[vid]
        tid = f"{layer}_{mapped_vid}_{cam_id}"

        image_names = []
        for it in items_sorted:
            try:
                frame_id = int(it.get("frame_id", 0))
            except Exception:
                frame_id = 0
            image_names.append(generate_filename(mapped_vid, cam_id, frame_id))

        start_t = float(items_sorted[0].get("timestamp", 0)) if items_sorted else 0.0
        end_t = float(items_sorted[-1].get("timestamp", 0)) if items_sorted else 0.0

        tracklets[tid] = {
            "tracklet_id": tid,
            "vehicle_id": vid,
            "mapped_id": mapped_vid,
            "camera_id": cam_id,
            "layer": layer,
            "start_time": start_t,
            "end_time": end_t,
            "image_count": len(image_names),
            "images": image_names,
        }

    return tracklets


def build_air_ground_protocol(air_tracklets: dict, ground_tracklets: dict, tol_s: float):
    """Build Air->Ground and Ground->Air protocols using same-vehicle + time-overlap positives."""
    air_by_vid = defaultdict(list)
    for tid, t in air_tracklets.items():
        air_by_vid[t["vehicle_id"]].append(tid)

    ground_by_vid = defaultdict(list)
    for tid, t in ground_tracklets.items():
        ground_by_vid[t["vehicle_id"]].append(tid)

    vehicles_with_both = sorted(set(air_by_vid.keys()) & set(ground_by_vid.keys()))

    # Air -> Ground
    a2g_query = []
    a2g_pos = {}
    for a_tid, a_t in air_tracklets.items():
        vid = a_t["vehicle_id"]
        candidates = []
        for g_tid in ground_by_vid.get(vid, []):
            g_t = ground_tracklets[g_tid]
            if _intervals_overlap(a_t["start_time"], a_t["end_time"], g_t["start_time"], g_t["end_time"], tol_s):
                candidates.append(g_tid)
        if candidates:
            a2g_query.append(a_tid)
            a2g_pos[a_tid] = sorted(candidates)

    # Ground -> Air
    g2a_query = []
    g2a_pos = {}
    for g_tid, g_t in ground_tracklets.items():
        vid = g_t["vehicle_id"]
        candidates = []
        for a_tid in air_by_vid.get(vid, []):
            a_t = air_tracklets[a_tid]
            if _intervals_overlap(a_t["start_time"], a_t["end_time"], g_t["start_time"], g_t["end_time"], tol_s):
                candidates.append(a_tid)
        if candidates:
            g2a_query.append(g_tid)
            g2a_pos[g_tid] = sorted(candidates)

    protocol = {
        "description": "Track-level Air<->Ground protocol (same vehicle + time-overlap matching)",
        "time_tolerance_s": tol_s,
        "vehicles_with_both": len(vehicles_with_both),
        "air2ground": {
            "query_tracklets": sorted(a2g_query),
            "gallery_tracklets": sorted(ground_tracklets.keys()),
            "positives": a2g_pos,
        },
        "ground2air": {
            "query_tracklets": sorted(g2a_query),
            "gallery_tracklets": sorted(air_tracklets.keys()),
            "positives": g2a_pos,
        },
    }

    return protocol


def generate_air_ground_assets(
    all_captures: list,
    air_camera_ids: set,
    scope_vehicle_ids: set,
    src_image_dir: str,
    output_dir: str,
    vid_mapping: dict,
    tol_s: float,
    ag_subdir: str = "ag_protocol",
    scope_name: str = "test",
    only_both_vehicles: bool = True,
):
    """
    Generate track-level Air<->Ground protocol assets under:
      {output_dir}/{ag_subdir}/images/{air,ground}
      {output_dir}/{ag_subdir}/metadata/{tracklets.json,protocol.json,pairs.csv}
    """
    ag_root = os.path.join(output_dir, ag_subdir)
    air_img_dir = os.path.join(ag_root, "images", "air")
    ground_img_dir = os.path.join(ag_root, "images", "ground")
    meta_dir = os.path.join(ag_root, "metadata")
    os.makedirs(air_img_dir, exist_ok=True)
    os.makedirs(ground_img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Only keep vehicles that are present in the release mapping; otherwise images can't be renamed.
    base_vehicle_ids = set(scope_vehicle_ids or set())
    base_vehicle_ids = {vid for vid in base_vehicle_ids if vid in vid_mapping}
    if not base_vehicle_ids:
        print(f"[AG-{scope_name}] No vehicles in scope after mapping filter, skip.")
        return {
            "ag_root": ag_root,
            "scope": scope_name,
            "base_vehicle_count": 0,
            "protocol_vehicle_count": 0,
            "vehicles_with_both_in_base": 0,
            "air_tracklets": 0,
            "ground_tracklets": 0,
            "vehicles_with_both": 0,
        }

    air_vids = {
        c.get("vehicle_id")
        for c in all_captures
        if c.get("vehicle_id") in base_vehicle_ids and c.get("camera_id") in air_camera_ids
    }
    ground_vids = {
        c.get("vehicle_id")
        for c in all_captures
        if c.get("vehicle_id") in base_vehicle_ids and c.get("camera_id") not in air_camera_ids
    }
    vehicles_with_both_in_base = sorted((air_vids & ground_vids) - {None})

    protocol_vehicle_ids = set(vehicles_with_both_in_base) if only_both_vehicles else set(base_vehicle_ids)
    if not protocol_vehicle_ids:
        print(f"[AG-{scope_name}] No vehicles with both views in this scope, skip.")
        return {
            "ag_root": ag_root,
            "scope": scope_name,
            "base_vehicle_count": len(base_vehicle_ids),
            "protocol_vehicle_count": 0,
            "vehicles_with_both_in_base": 0,
            "air_tracklets": 0,
            "ground_tracklets": 0,
            "vehicles_with_both": 0,
        }

    air_caps = [
        c
        for c in all_captures
        if c.get("vehicle_id") in protocol_vehicle_ids and c.get("camera_id") in air_camera_ids
    ]
    ground_caps = [
        c
        for c in all_captures
        if c.get("vehicle_id") in protocol_vehicle_ids and c.get("camera_id") not in air_camera_ids
    ]

    # Copy full tracklet images for protocol (kept separate from image-level train/gallery/query).
    print(f"\n[AG-{scope_name}] Vehicle pool: base={len(base_vehicle_ids)} both={len(vehicles_with_both_in_base)} protocol={len(protocol_vehicle_ids)}")
    print(f"[AG-{scope_name}] Copying Air-Ground protocol images -> {ag_subdir}/ ...")
    copy_images(air_caps, src_image_dir, air_img_dir, vid_mapping)
    copy_images(ground_caps, src_image_dir, ground_img_dir, vid_mapping)

    air_tracklets = build_tracklets(air_caps, vid_mapping, layer="air")
    ground_tracklets = build_tracklets(ground_caps, vid_mapping, layer="ground")
    protocol = build_air_ground_protocol(air_tracklets, ground_tracklets, tol_s=tol_s)
    protocol.update(
        {
            "scope": scope_name,
            "base_vehicle_count": len(base_vehicle_ids),
            "vehicles_with_both_in_base": len(vehicles_with_both_in_base),
            "vehicle_ids_with_both_in_base": vehicles_with_both_in_base,
            "protocol_vehicle_count": len(protocol_vehicle_ids),
        }
    )

    # Write tracklets and protocol
    tracklets_out = {
        "description": "Tracklets for Air-Ground protocol",
        "scope": scope_name,
        "base_vehicle_count": len(base_vehicle_ids),
        "vehicles_with_both_in_base": len(vehicles_with_both_in_base),
        "protocol_vehicle_count": len(protocol_vehicle_ids),
        "air_tracklets": air_tracklets,
        "ground_tracklets": ground_tracklets,
    }
    tracklets_path = os.path.join(meta_dir, "tracklets.json")
    with open(tracklets_path, "w", encoding="utf-8") as f:
        json.dump(tracklets_out, f, indent=2, ensure_ascii=False)

    protocol_path = os.path.join(meta_dir, "protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    # Optional: pairs.csv for quick inspection / downstream association.
    pairs_path = os.path.join(meta_dir, "pairs.csv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["direction", "query_tracklet", "gallery_tracklet", "vehicle_id"])
        for q, pos in protocol["air2ground"]["positives"].items():
            vid = air_tracklets[q]["vehicle_id"]
            for g in pos:
                writer.writerow(["air2ground", q, g, vid])
        for q, pos in protocol["ground2air"]["positives"].items():
            vid = ground_tracklets[q]["vehicle_id"]
            for a in pos:
                writer.writerow(["ground2air", q, a, vid])

    print(f"[AG-{scope_name}] Tracklets: air={len(air_tracklets)} ground={len(ground_tracklets)}")
    print(f"[AG-{scope_name}] Vehicles with both views (in protocol): {protocol['vehicles_with_both']}")
    print(f"[AG-{scope_name}] Saved: {tracklets_path}")
    print(f"[AG-{scope_name}] Saved: {protocol_path}")
    print(f"[AG-{scope_name}] Saved: {pairs_path}")

    return {
        "ag_root": ag_root,
        "scope": scope_name,
        "base_vehicle_count": len(base_vehicle_ids),
        "protocol_vehicle_count": len(protocol_vehicle_ids),
        "vehicles_with_both_in_base": len(vehicles_with_both_in_base),
        "air_tracklets": len(air_tracklets),
        "ground_tracklets": len(ground_tracklets),
        "vehicles_with_both": protocol["vehicles_with_both"],
    }


# =============================================================================
# =============================================================================

def copy_images(captures, src_dir, dst_dir, vid_mapping):
    """Copy and rename image files."""
    os.makedirs(dst_dir, exist_ok=True)
    
    copied = 0
    failed = 0
    
    for cap in tqdm(captures, desc=f"  Copy to {os.path.basename(dst_dir)}"):
        src_filename = os.path.basename(cap.get('image_path', ''))
        src_path = os.path.join(src_dir, src_filename)
        
        vid = vid_mapping[cap['vehicle_id']]
        cam = cap['camera_id']
        frame = cap['frame_id']
        dst_filename = generate_filename(vid, cam, frame)
        dst_path = os.path.join(dst_dir, dst_filename)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                failed += 1
        else:
            failed += 1
    
    print(f"  Completed: {copied} successful, {failed} failed")
    return copied


# =============================================================================
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SimVeRi release generator (with optional Air-Ground protocol)")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Cleaned input dir (contains metadata/captures_cleaned.json)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Release output dir")
    parser.add_argument("--topology-file", default=CAMERA_TOPOLOGY_FILE, help="Camera topology JSON")
    parser.add_argument(
        "--include-air-in-ground-release",
        action="store_true",
        default=INCLUDE_AIR_IN_GROUND_RELEASE,
        help="If set, UAV (layer=air) cameras are included in the main train/gallery/query split (NOT recommended for ground benchmark).",
    )
    parser.add_argument(
        "--skip-air-ground-protocol",
        action="store_true",
        help="Disable generation of the track-level Air-Ground protocol outputs.",
    )
    parser.add_argument(
        "--ag-protocol-scope",
        choices=["test", "full", "both"],
        default="both",
        help=(
            "Which Air-Ground protocol assets to generate: "
            "test (only test vehicles: gallery+query), "
            "full (all release vehicles: train+test), "
            "both (default)."
        ),
    )
    parser.add_argument(
        "--air-ground-time-tol",
        type=float,
        default=AIR_GROUND_TIME_TOLERANCE_S,
        help="Time tolerance (seconds) for air-ground tracklet overlap matching.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    input_json = os.path.join(input_dir, "metadata", "captures_cleaned.json")
    input_image_dir = os.path.join(input_dir, "image_train")
    output_dir = args.output_dir
    topology_file = args.topology_file
    include_air_in_ground_release = bool(args.include_air_in_ground_release)
    generate_air_ground_protocol = bool(GENERATE_AIR_GROUND_PROTOCOL and (not args.skip_air_ground_protocol))
    ag_protocol_scope = str(args.ag_protocol_scope)
    ag_time_tol = float(args.air_ground_time_tol)

    print("=" * 70)
    print(f"{DATASET_NAME} - {DATASET_FULL_NAME}")
    print(f"Dataset Generator v3.2 (Multi-camera Query + Query not in Gallery)")
    print("=" * 70)
    
    # =========================================================================
    # =========================================================================
    print(f"\n[1/8] Load data...")
    
    if not os.path.exists(input_json):
        print(f"  Error: file not found: {input_json}")
        return
    
    with open(input_json, 'r', encoding='utf-8') as f:
        captures = json.load(f)
    
    print(f"  Loaded records: {len(captures)}")
    
    # =========================================================================
    # =========================================================================
    print(f"\n[2/8] Load camera topology...")
    
    topology_data = load_camera_topology(topology_file)

    air_camera_ids = {
        cam_id
        for cam_id, cam_data in topology_data.get("cameras", {}).items()
        if str(cam_data.get("layer", "")).lower() == "air"
    }
    if air_camera_ids and not include_air_in_ground_release:
        captures_for_split = [c for c in captures if c.get("camera_id") not in air_camera_ids]
    else:
        captures_for_split = captures

    # Keep the ground benchmark "clean": require >=2 ground cameras per vehicle after removing UAV cams.
    if not include_air_in_ground_release:
        vehicle_cams = defaultdict(set)
        for c in captures_for_split:
            vid = c.get("vehicle_id")
            cid = c.get("camera_id")
            if vid and cid:
                vehicle_cams[vid].add(cid)
        eligible_vehicles = {vid for vid, cams in vehicle_cams.items() if len(cams) >= 2}
        captures_for_split = [c for c in captures_for_split if c.get("vehicle_id") in eligible_vehicles]
    
    # =========================================================================
    # =========================================================================
    print(f"\n[3/8] Split the dataset...")
    
    train_caps, gallery_caps, query_caps = split_dataset(captures_for_split, TRAIN_RATIO)
    gallery_caps = limit_gallery_per_camera(gallery_caps, GALLERY_MAX_PER_CAMERA)
    selected_captures = train_caps + gallery_caps + query_caps

    vid_mapping_result = create_vehicle_id_mapping(selected_captures)
    vid_mapping = vid_mapping_result.mapping
    print(f"  Vehicle ID mapping: {len(vid_mapping)} vehicles")
    
    # =========================================================================
    # =========================================================================
    print(f"\n[4/8] Create the directory structure...")

    if CLEAN_OUTPUT_DIR and os.path.isdir(output_dir):
        output_abs = os.path.abspath(output_dir)
        project_root = os.path.dirname(os.path.abspath(__file__))
        output_norm = os.path.normcase(output_abs)
        project_norm = os.path.normcase(project_root)
        if output_norm == project_norm or not output_norm.startswith(project_norm + os.sep):
            print(f"  Warning: OUTPUT_DIR is outside the project directory; skip cleanup: {output_abs}")
        else:
            print(f"  Cleaning output directory: {output_dir}/")
            shutil.rmtree(output_abs)
    
    dirs = [
        f"{output_dir}/images/train",
        f"{output_dir}/images/gallery",
        f"{output_dir}/images/query",
        f"{output_dir}/annotations",
        f"{output_dir}/metadata",
        f"{output_dir}/statistics"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"  Created: {output_dir}/")
    
    # =========================================================================
    # =========================================================================
    print(f"\n[5/8] Copy images...")
    
    copy_images(train_caps, input_image_dir, f"{output_dir}/images/train", vid_mapping)
    copy_images(gallery_caps, input_image_dir, f"{output_dir}/images/gallery", vid_mapping)
    copy_images(query_caps, input_image_dir, f"{output_dir}/images/query", vid_mapping)
    
    # =========================================================================
    # =========================================================================
    print(f"\n[6/8] Generate annotation files...")
    
    generate_annotations_xml(train_caps, f"{output_dir}/annotations/train_annotations.xml", vid_mapping, "train")
    generate_annotations_xml(gallery_caps, f"{output_dir}/annotations/gallery_annotations.xml", vid_mapping, "gallery")
    generate_query_list(query_caps, f"{output_dir}/annotations/query_list.txt", vid_mapping)
    generate_ground_truth(query_caps, gallery_caps, f"{output_dir}/annotations/ground_truth.txt", vid_mapping)
    generate_ignore_list(query_caps, gallery_caps, f"{output_dir}/annotations/ignore_list.txt")
    
    # =========================================================================
    # =========================================================================
    print(f"\n[7/8] Generate metadata files...")
    
    generate_vehicle_attributes(selected_captures, f"{output_dir}/metadata/vehicle_attributes.json")
    generate_spatiotemporal_json(selected_captures, f"{output_dir}/metadata/spatiotemporal.json", vid_mapping)
    generate_camera_network(f"{output_dir}/metadata/camera_network.json", topology_data)
    generate_twins_groups(selected_captures, f"{output_dir}/metadata/twins_groups.json", vid_mapping)
    generate_trajectory_info(selected_captures, f"{output_dir}/metadata/trajectory_info.csv", vid_mapping)
    generate_camera_transitions(selected_captures, f"{output_dir}/metadata/camera_transitions.csv", topology_data)

    # Write the actual vehicle split lists for reproducibility/debugging.
    split_meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "include_air_in_ground_release": include_air_in_ground_release,
        "min_ground_cameras_per_vehicle": 2 if not include_air_in_ground_release else 1,
        "air_camera_ids": sorted(air_camera_ids),
        "train_vehicle_ids": sorted({cap["vehicle_id"] for cap in train_caps}),
        "test_vehicle_ids": sorted({cap["vehicle_id"] for cap in gallery_caps} | {cap["vehicle_id"] for cap in query_caps}),
    }
    split_path = os.path.join(output_dir, "metadata", "splits.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"  Generated: {split_path}")
    
    # =========================================================================
    # =========================================================================
    print(f"\n[8/8] Generate statistics files...")
    
    summary = generate_dataset_summary(
        selected_captures, train_caps, gallery_caps, query_caps,
        f"{output_dir}/statistics/dataset_summary.json",
        vid_mapping, topology_data, vid_mapping_result
    )

    # -------------------------------------------------------------------------
    # Optional: Track-level Air<->Ground protocol (kept separate from benchmark)
    # -------------------------------------------------------------------------
    if generate_air_ground_protocol and air_camera_ids:
        test_vehicle_ids = set(cap["vehicle_id"] for cap in gallery_caps) | set(cap["vehicle_id"] for cap in query_caps)
        full_vehicle_ids = set(vid_mapping.keys())  # all vehicles included in the release (train+test)

        if ag_protocol_scope in ("test", "both"):
            generate_air_ground_assets(
                all_captures=captures,
                air_camera_ids=air_camera_ids,
                scope_vehicle_ids=test_vehicle_ids,
                src_image_dir=input_image_dir,
                output_dir=output_dir,
                vid_mapping=vid_mapping,
                tol_s=ag_time_tol,
                ag_subdir="ag_protocol",
                scope_name="test",
                # Keep ground distractors for the test protocol (evaluation-like).
                only_both_vehicles=False,
            )

        if ag_protocol_scope in ("full", "both"):
            generate_air_ground_assets(
                all_captures=captures,
                air_camera_ids=air_camera_ids,
                scope_vehicle_ids=full_vehicle_ids,
                src_image_dir=input_image_dir,
                output_dir=output_dir,
                vid_mapping=vid_mapping,
                tol_s=ag_time_tol,
                ag_subdir="ag_protocol_full",
                scope_name="full",
                # Full protocol focuses on vehicles that actually have both views.
                only_both_vehicles=True,
            )
    
    # =========================================================================
    # =========================================================================
    stats = summary['statistics']
    twins_stats = summary['twins_subset']
    camera_stats = summary['camera_network']
    query_stats = summary['query_protocol']
    
    print("\n" + "=" * 70)
    print(f"{DATASET_NAME} dataset generation finished")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}/")
    print(f"\nDataset statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Train: {stats['train_images']} images / {stats['train_vehicles']} vehicles")
    print(f"  Gallery: {stats['gallery_images']} images / {stats['gallery_vehicles']} vehicles")
    print(f"  Query: {stats['query_images']} images / {query_stats['query_vehicles']} vehicles")
    print(f"  Average queries per vehicle: {query_stats['avg_query_per_vehicle']:.2f}")
    print(f"  Cameras: {stats['cameras']}")
    print(f"  Twins subset: {twins_stats['total_images']} images / {twins_stats['total_vehicles']} vehicles / {twins_stats['num_groups']} groups")
    print(f"\nQuery protocol:")
    print(f"  Method: {query_stats['method']}")
    print(f"  Query in Gallery: {query_stats['query_in_gallery']}")
    print(f"\nCamera network:")
    print(f"  Distance range: {camera_stats['min_distance_m']}m - {camera_stats['max_distance_m']}m")
    print(f"  Average distance: {camera_stats['avg_distance_m']}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
