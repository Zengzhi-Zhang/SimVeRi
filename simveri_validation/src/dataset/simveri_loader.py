# src/dataset/simveri_loader.py
"""
SimVeRi dataset loader
Parses the release annotations and metadata into one unified access interface
"""

import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.path_utils import get_default_simveri_root


@dataclass
class SimVeRiSample:
    """Complete information for a single SimVeRi sample."""
    image_name: str
    image_path: str
    vehicle_id: str
    camera_id: str
    
    timestamp: float
    position: np.ndarray  # [x, y, z]
    speed: float          # km/h
    heading: float        # degrees
    
    occlusion: float      # [0, 1]
    distance: float       # meters
    
    split: str            # 'train', 'gallery', 'query'
    is_twins: bool = False
    twins_group: Optional[str] = None
    
    pid: int = 0          # person/vehicle ID (numeric)
    camid: int = 0        # camera ID (numeric)


class SimVeRiDataset:
    """
    Complete SimVeRi dataset loader.
    
    Usage:
        dataset = SimVeRiDataset("../SimVeRi-dataset-v2.0")
        train_samples = dataset.train_samples
        twins_query = dataset.get_twins_samples('query')
    """
    
    def __init__(self, root_dir: str, verbose: bool = True):
        """
        Initialize the dataset loader.
        
        Args:
            root_dir: root directory of the SimVeRi dataset
            verbose: whether to print detailed information
        """
        self.root_dir = root_dir
        self.verbose = verbose
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.metadata_dir = os.path.join(root_dir, 'metadata')
        
        self._validate_directories()
        
        self.samples: Dict[str, SimVeRiSample] = {}
        self.train_samples: List[SimVeRiSample] = []
        self.gallery_samples: List[SimVeRiSample] = []
        self.query_samples: List[SimVeRiSample] = []
        
        self.spatiotemporal: Dict = {}
        self.camera_network: Dict = {}
        self.twins_groups: Dict = {}
        self.transition_params: Dict[Tuple[str, str], Dict] = {}
        
        self.vid_to_pid: Dict[str, int] = {}
        self.cam_to_camid: Dict[str, int] = {}
        
        self.image_paths: Dict[str, str] = {}
        
        if self.verbose:
            print("=" * 60)
            print("Initializing SimVeRi Dataset")
            print("=" * 60)
        
        self._load_all()
        
        if self.verbose:
            self._print_summary()
    
    def _validate_directories(self):
        """Validate that the required directories exist."""
        required_dirs = [
            self.images_dir,
            self.annotations_dir,
            self.metadata_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    def _load_all(self):
        """Load all dataset components."""
        if self.verbose:
            print("\n[1/5] Building image index...")
        self.image_paths = self._build_image_index()
        
        if self.verbose:
            print("[2/5] Loading metadata...")
        self._load_metadata()
        
        if self.verbose:
            print("[3/5] Building ID mappings...")
        self._build_id_mappings()
        
        if self.verbose:
            print("[4/5] Building Twins lookup...")
        self.twins_lookup = self._build_twins_lookup()
        
        if self.verbose:
            print("[5/5] Parsing annotations...")
        self._parse_annotations()
    
    def _build_image_index(self) -> Dict[str, str]:
        """Build an image-name-to-path index to avoid repeated filesystem lookups."""
        index = {}
        total_count = 0
        
        for split in ['train', 'gallery', 'query']:
            split_dir = os.path.join(self.images_dir, split)
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
                for img_name in files:
                    index[img_name] = os.path.join(split_dir, img_name)
                total_count += len(files)
                if self.verbose:
                    print(f"       {split}: {len(files)} images")
        
        if self.verbose:
            print(f"       Total: {total_count} images indexed")
        
        return index
    
    def _load_metadata(self):
        """Load all metadata files."""
        
        st_path = os.path.join(self.metadata_dir, 'spatiotemporal.json')
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"Required file not found: {st_path}")
        
        with open(st_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.spatiotemporal = data.get('annotations', data)
        
        if self.verbose:
            sample_keys = list(self.spatiotemporal.keys())[:3]
            print(f"       Spatiotemporal: {len(self.spatiotemporal)} records")
            print(f"       Sample keys: {sample_keys}")
        
        cn_path = os.path.join(self.metadata_dir, 'camera_network.json')
        if not os.path.exists(cn_path):
            raise FileNotFoundError(f"Required file not found: {cn_path}")
        
        with open(cn_path, 'r', encoding='utf-8') as f:
            self.camera_network = json.load(f)
        
        if self.verbose:
            num_cameras = len(self.camera_network.get('cameras', {}))
            print(f"       Camera network: {num_cameras} cameras")
        
        tg_path = os.path.join(self.metadata_dir, 'twins_groups.json')
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"Required file not found: {tg_path}")
        
        with open(tg_path, 'r', encoding='utf-8') as f:
            self.twins_groups = json.load(f)
        
        if self.verbose:
            num_groups = len(self.twins_groups.get('groups', {}))
            print(f"       Twins groups: {num_groups} groups")
        
        transitions_path = os.path.join(self.metadata_dir, 'camera_transitions.csv')
        if os.path.exists(transitions_path):
            df = pd.read_csv(transitions_path)
            for _, row in df.iterrows():
                key = (str(row['from_camera']), str(row['to_camera']))
                self.transition_params[key] = {
                    'mean_time': float(row['mean_time_s']),
                    'std_time': float(row['std_time_s']),
                    'min_time': float(row.get('min_time_s', 0)),
                    'max_time': float(row.get('max_time_s', 0)),
                    'distance': float(row.get('route_distance_m', -1)),
                    'sample_count': int(row['sample_count'])
                }
            
            if self.verbose:
                print(f"       Transition params: {len(self.transition_params)} camera pairs")
        else:
            if self.verbose:
                print(f"       Warning: camera_transitions.csv not found")
    
    def _build_id_mappings(self):
        """Build numeric mappings for vehicle IDs and camera IDs."""
        vehicle_ids = set()
        camera_ids = set()
        
        for info in self.spatiotemporal.values():
            vehicle_ids.add(info['vehicle_id'])
            camera_ids.add(info['camera_id'])
        
        for idx, vid in enumerate(sorted(vehicle_ids)):
            self.vid_to_pid[vid] = idx
        
        for idx, cid in enumerate(sorted(camera_ids)):
            self.cam_to_camid[cid] = idx
        
        if self.verbose:
            print(f"       Vehicle IDs: {len(self.vid_to_pid)}")
            print(f"       Camera IDs: {len(self.cam_to_camid)}")
    
    def _build_twins_lookup(self) -> Dict[str, str]:
        """Build the vehicle_id -> twins_group mapping."""
        lookup = {}
        for group_id, info in self.twins_groups.get('groups', {}).items():
            for vid in info.get('mapped_ids', []):
                lookup[vid] = group_id
        return lookup
    
    def _parse_annotations(self):
        """Parse all annotation files."""
        
        train_xml = os.path.join(self.annotations_dir, 'train_annotations.xml')
        self.train_samples = self._parse_xml(train_xml, 'train')
        
        gallery_xml = os.path.join(self.annotations_dir, 'gallery_annotations.xml')
        self.gallery_samples = self._parse_xml(gallery_xml, 'gallery')
        
        self.query_samples = self._parse_query_list()
        
        for sample in self.train_samples + self.gallery_samples + self.query_samples:
            self.samples[sample.image_name] = sample
    
    def _parse_xml(self, xml_path: str, split: str) -> List[SimVeRiSample]:
        """Parse one XML annotation file."""
        samples = []
        
        if not os.path.exists(xml_path):
            if self.verbose:
                print(f"       Warning: {xml_path} not found")
            return samples
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        vehicles = root.findall('.//Vehicle')
        
        iterator = tqdm(vehicles, desc=f"       Parsing {split}", disable=not self.verbose)
        
        for vehicle in iterator:
            img_elem = vehicle.find('Image')
            if img_elem is None:
                continue
            
            img_name = img_elem.text
            if img_name is None:
                continue
            
            vid = vehicle.get('id', '')
            cam = vehicle.get('camera', '')
            
            st_info = self.spatiotemporal.get(img_name, {})
            
            pos_info = st_info.get('position', {})
            position = np.array([
                float(pos_info.get('x', 0)),
                float(pos_info.get('y', 0)),
                float(pos_info.get('z', 0))
            ])
            
            motion_info = st_info.get('motion', {})
            speed = float(motion_info.get('speed_kmh', 0))
            heading = float(motion_info.get('heading_deg', 0))
            
            quality_info = st_info.get('quality', {})
            occlusion = float(quality_info.get('occlusion_ratio', 0))
            distance = float(quality_info.get('distance_m', 0))
            
            sample = SimVeRiSample(
                image_name=img_name,
                image_path=self.image_paths.get(img_name, ''),
                vehicle_id=vid,
                camera_id=cam,
                timestamp=float(st_info.get('timestamp', 0)),
                position=position,
                speed=speed,
                heading=heading,
                occlusion=occlusion,
                distance=distance,
                split=split,
                is_twins=(vid in self.twins_lookup),
                twins_group=self.twins_lookup.get(vid),
                pid=self.vid_to_pid.get(vid, 0),
                camid=self.cam_to_camid.get(cam, 0)
            )
            samples.append(sample)
        
        return samples
    
    def _parse_query_list(self) -> List[SimVeRiSample]:
        """Parse the Query list."""
        samples = []
        query_path = os.path.join(self.annotations_dir, 'query_list.txt')
        
        if not os.path.exists(query_path):
            if self.verbose:
                print(f"       Warning: {query_path} not found")
            return samples
        
        with open(query_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        iterator = tqdm(lines, desc="       Parsing query", disable=not self.verbose)
        
        for img_name in iterator:
            parts = img_name.replace('.jpg', '').split('_')
            if len(parts) < 2:
                continue
            
            vid = parts[0]
            cam_part = parts[1]
            if cam_part.startswith('c'):
                cam = cam_part
            else:
                cam = f"c{cam_part}"
            
            st_info = self.spatiotemporal.get(img_name, {})
            
            pos_info = st_info.get('position', {})
            position = np.array([
                float(pos_info.get('x', 0)),
                float(pos_info.get('y', 0)),
                float(pos_info.get('z', 0))
            ])
            
            motion_info = st_info.get('motion', {})
            speed = float(motion_info.get('speed_kmh', 0))
            heading = float(motion_info.get('heading_deg', 0))
            
            quality_info = st_info.get('quality', {})
            occlusion = float(quality_info.get('occlusion_ratio', 0))
            distance = float(quality_info.get('distance_m', 0))
            
            sample = SimVeRiSample(
                image_name=img_name,
                image_path=self.image_paths.get(img_name, ''),
                vehicle_id=vid,
                camera_id=cam,
                timestamp=float(st_info.get('timestamp', 0)),
                position=position,
                speed=speed,
                heading=heading,
                occlusion=occlusion,
                distance=distance,
                split='query',
                is_twins=(vid in self.twins_lookup),
                twins_group=self.twins_lookup.get(vid),
                pid=self.vid_to_pid.get(vid, 0),
                camid=self.cam_to_camid.get(cam, 0)
            )
            samples.append(sample)
        
        return samples
    
    def _print_summary(self):
        """Print a dataset summary."""
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"  Train samples:   {len(self.train_samples):,}")
        print(f"  Gallery samples: {len(self.gallery_samples):,}")
        print(f"  Query samples:   {len(self.query_samples):,}")
        print(f"  Total samples:   {len(self.samples):,}")
        print("-" * 60)
        print(f"  Twins vehicles:  {len(self.get_twins_vehicle_ids())}")
        print(f"  Base train:      {len(self.get_base_samples('train')):,}")
        print(f"  Twins train:     {len(self.get_twins_samples('train')):,}")
        print("=" * 60)
    
    # ----------------------------------------------------------------------------
    
    def get_twins_vehicle_ids(self) -> List[str]:
        """Return all Twins vehicle IDs."""
        ids = []
        for info in self.twins_groups.get('groups', {}).values():
            ids.extend(info.get('mapped_ids', []))
        return ids
    
    def get_base_samples(self, split: str) -> List[SimVeRiSample]:
        """Return Base-subset samples (excluding Twins)."""
        if split == 'train':
            return [s for s in self.train_samples if not s.is_twins]
        elif split == 'gallery':
            return [s for s in self.gallery_samples if not s.is_twins]
        elif split == 'query':
            return [s for s in self.query_samples if not s.is_twins]
        return []
    
    def get_twins_samples(self, split: str) -> List[SimVeRiSample]:
        """Return Twins-subset samples."""
        if split == 'train':
            return [s for s in self.train_samples if s.is_twins]
        elif split == 'gallery':
            return [s for s in self.gallery_samples if s.is_twins]
        elif split == 'query':
            return [s for s in self.query_samples if s.is_twins]
        return []
    
    def get_samples_by_vehicle(self, vehicle_id: str) -> List[SimVeRiSample]:
        """Return all samples for one vehicle ID."""
        return [s for s in self.samples.values() if s.vehicle_id == vehicle_id]
    
    def get_distance(self, cam_from: str, cam_to: str) -> float:
        """Return the path distance between two cameras."""
        matrix = self.camera_network.get('distance_matrix', {})
        return float(matrix.get(cam_from, {}).get(cam_to, -1))
    
    def get_transition_params(self, cam_from: str, cam_to: str) -> Optional[Dict]:
        """Return the transition-time parameters for one camera pair."""
        return self.transition_params.get((cam_from, cam_to))
    
    def get_camera_position(self, camera_id: str) -> Optional[np.ndarray]:
        """Return one camera position."""
        cameras = self.camera_network.get('cameras', {})
        cam_info = cameras.get(camera_id, {})
        pos = cam_info.get('position', {})
        if pos:
            return np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        return None


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    data_root = get_default_simveri_root()
    
    print("Loading SimVeRi Dataset...")
    dataset = SimVeRiDataset(data_root)
    
    print("\n" + "=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    tests_total += 1
    if len(dataset.train_samples) > 0:
        print(f"[OK] Test 1: Train samples exist ({len(dataset.train_samples)})")
        tests_passed += 1
    else:
        print("[FAIL] Test 1: Train samples empty")
    
    tests_total += 1
    if len(dataset.gallery_samples) > 0:
        print(f"[OK] Test 2: Gallery samples exist ({len(dataset.gallery_samples)})")
        tests_passed += 1
    else:
        print("[FAIL] Test 2: Gallery samples empty")
    
    tests_total += 1
    if len(dataset.query_samples) > 0:
        print(f"[OK] Test 3: Query samples exist ({len(dataset.query_samples)})")
        tests_passed += 1
    else:
        print("[FAIL] Test 3: Query samples empty")
    
    tests_total += 1
    twins_ids = dataset.get_twins_vehicle_ids()
    if len(twins_ids) == 100:
        print(f"[OK] Test 4: Twins vehicle count is correct ({len(twins_ids)})")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 4: Twins vehicle count mismatch (expected 100, got {len(twins_ids)})")
    
    tests_total += 1
    sample = dataset.train_samples[0] if dataset.train_samples else None
    if sample and sample.timestamp > 0:
        print(f"[OK] Test 5: Spatiotemporal data linked (timestamp={sample.timestamp:.2f})")
        tests_passed += 1
    else:
        print("[FAIL] Test 5: Spatiotemporal data not linked")
    
    tests_total += 1
    if sample and np.any(sample.position != 0):
        print(f"[OK] Test 6: Position data valid ({sample.position})")
        tests_passed += 1
    else:
        print("[FAIL] Test 6: Position data invalid")
    
    tests_total += 1
    base_query = dataset.get_base_samples('query')
    twins_query = dataset.get_twins_samples('query')
    total_query = len(base_query) + len(twins_query)
    if total_query == len(dataset.query_samples):
        print(f"[OK] Test 7: Subset split correct (Base={len(base_query)}, Twins={len(twins_query)})")
        tests_passed += 1
    else:
        print(f"[FAIL] Test 7: Subset split mismatch ({total_query} != {len(dataset.query_samples)})")
    
    tests_total += 1
    if len(dataset.transition_params) > 0:
        sample_key = list(dataset.transition_params.keys())[0]
        sample_param = dataset.transition_params[sample_key]
        print(f"[OK] Test 8: Transition parameters loaded ({len(dataset.transition_params)} pairs)")
        print(f"         Sample: {sample_key} -> mean={sample_param['mean_time']:.1f}s")
        tests_passed += 1
    else:
        print("[FAIL] Test 8: Transition parameters not loaded")
    
    print("\n" + "=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("[OK] All tests passed")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed")
        sys.exit(1)
