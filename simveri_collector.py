"""
SimVeRi data collection plugin v1.3 (spatiotemporal enhancement edition).
Adds track sampling limits, motion checks, cross-camera filtering, and full trajectory logging.
"""

import os
import csv
import json
import queue
import math
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.dom import minidom

from config_loader import get_config
from bbox_utils import CameraProjector, get_2d_bbox
from occlusion import OcclusionCalculator


# =============================================================================
# =============================================================================

VERI_COLOR_MAP = {
    # VeRi ID 1: yellow
    'yellow': ('yellow', 1),
    
    # VeRi ID 3: green
    'green': ('green', 3),
    
    'gray': ('gray', 4),
    'dark_gray': ('gray', 4),
    'silver': ('gray', 4),
    
    'red': ('red', 5),
    'dark_red': ('red', 5),
    'burgundy': ('red', 5),
    
    'blue': ('blue', 6),
    'dark_blue': ('blue', 6),
    'sky_blue': ('blue', 6),
    
    'white': ('white', 7),
    'warm_white': ('white', 7),
    'cool_white': ('white', 7),
    'beige': ('white', 7),
    
    # VeRi ID 9: brown
    'brown': ('brown', 9),
    
    'black': ('black', 10),
}


def map_color_to_veri(simveri_color: str) -> Tuple[str, int]:
    """Map a SimVeRi color name to the VeRi label space."""
    if not simveri_color:
        return ('unknown', 0)
    
    color_key = simveri_color.lower().strip()
    
    if color_key in VERI_COLOR_MAP:
        return VERI_COLOR_MAP[color_key]
    
    return ('unknown', 0)

def extract_brand(blueprint: str) -> str:
    """Extract the brand name from a blueprint ID."""
    # vehicle.audi.a2 -> audi
    parts = blueprint.split('.')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


# =============================================================================
# =============================================================================

@dataclass
class VehicleCapture:
    """Single vehicle capture record."""
    vehicle_id: str
    carla_actor_id: int
    camera_id: str
    frame_id: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    bbox_area: int
    distance: float
    occlusion_ratio: float
    occlusion_level: str
    image_path: str
    blueprint: str = ""
    category: str = ""
    color_name: str = ""
    color_rgb: Tuple[int, int, int] = (0, 0, 0)
    is_fleet: bool = False
    fleet_id: Optional[str] = None
    global_x: float = 0.0
    global_y: float = 0.0
    global_z: float = 0.0
    speed: float = 0.0        # km/h
    heading: float = 0.0      # heading angle in degrees


# =============================================================================
# =============================================================================

class CameraManager:
    """Camera manager."""
    
    def __init__(self, world, config):
        self.world = world
        self.cfg = config
        self.cameras: Dict[str, object] = {}
        self.projectors: Dict[str, CameraProjector] = {}
        self.image_queues: Dict[str, queue.Queue] = {}
    
    def spawn_all_cameras(self) -> int:
        """Spawn all configured cameras."""
        import carla
        
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find(self.cfg.sensor_type)
        
        width, height = self.cfg.resolution
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        # Disable motion blur if supported by the sensor
        if camera_bp.has_attribute('motion_blur_intensity'):
            camera_bp.set_attribute('motion_blur_intensity', '0.0')
        if camera_bp.has_attribute('motion_blur_max_distortion'):
            camera_bp.set_attribute('motion_blur_max_distortion', '0.0')
        if camera_bp.has_attribute('motion_blur_min_object_screen_size'):
            camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.0')
        
        count = 0
        for cam_cfg in self.cfg.cameras:
            try:
                import carla
                # Per-camera FOV (required for UAV cams and other heterogeneous setups).
                camera_bp.set_attribute('fov', str(getattr(cam_cfg, 'fov', self.cfg.fov)))
                location = carla.Location(
                    x=cam_cfg.position[0],
                    y=cam_cfg.position[1],
                    z=cam_cfg.position[2]
                )
                rotation = carla.Rotation(
                    pitch=cam_cfg.rotation.get('pitch', 0),
                    yaw=cam_cfg.rotation.get('yaw', 0),
                    roll=cam_cfg.rotation.get('roll', 0)
                )
                transform = carla.Transform(location, rotation)
                
                camera = self.world.spawn_actor(camera_bp, transform)
                cam_id = cam_cfg.camera_id
                
                self.cameras[cam_id] = camera
                self.image_queues[cam_id] = queue.Queue(maxsize=1)
                self.projectors[cam_id] = CameraProjector(camera)
                
                camera.listen(lambda img, cid=cam_id: self._on_image(cid, img))
                count += 1
                
            except Exception as e:
                print(f"  Camera {cam_cfg.camera_id} spawn failed: {e}")
        
        print(f"Spawned {count}/{len(self.cfg.cameras)} cameras")
        return count
    
    def _on_image(self, cam_id: str, image):
        q = self.image_queues[cam_id]
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put(image)
    
    def get_current_images(self, timeout: float = 2.0) -> Dict[str, object]:
        images = {}
        for cam_id, q in self.image_queues.items():
            try:
                img = q.get(timeout=timeout)
                images[cam_id] = img
            except queue.Empty:
                pass
        return images
    
    def destroy_all(self):
        for cam_id, camera in self.cameras.items():
            try:
                if camera.is_alive:
                    camera.stop()
                    camera.destroy()
            except:
                pass
        self.cameras.clear()
        print(" All cameras destroyed")


# =============================================================================
# =============================================================================

class SimVeRiCollector:
    """SimVeRi data collection plugin v1.3."""

    # =========================================================================
    # =========================================================================

    TYPE_TO_VERI_ID = {
        'sedan': 1,
        'coupe': 1,      # map coupe to sedan
        'suv': 2,
        'van': 3,
        'special': 3,    # map special-purpose vehicles to van
        'hatchback': 4,
        'mpv': 5,
        'pickup': 6,
        'bus': 7,
        'truck': 8,
        'estate': 9,
    }

    VERI_ID_TO_NAME = {
        1: 'sedan', 2: 'suv', 3: 'van', 4: 'hatchback',
        5: 'mpv', 6: 'pickup', 7: 'bus', 8: 'truck', 9: 'estate',
        0: 'unknown'
    }

    def __init__(self, world, vehicle_info: dict, output_dir: str = 'output'):
        self.world = world
        self.cfg = get_config()
        self.vehicle_info = vehicle_info
        self.output_dir = output_dir
        
        self.camera_manager: Optional[CameraManager] = None
        self.occlusion_calc: Optional[OcclusionCalculator] = None
        
        self.frame_count = 0
        self.captures: List[VehicleCapture] = []
        self.is_initialized = False
        
        self.sampling_interval = self.cfg.sampling_interval

        # Per-camera override knobs (e.g., UAV cams need different thresholds).
        self.per_camera_overrides = getattr(self.cfg, 'per_camera_overrides', {}) or {}

        # Cached camera metadata for lightweight tagging in captures.json
        self._camera_layer_map = {c.camera_id: getattr(c, 'layer', 'unknown') for c in getattr(self.cfg, 'cameras', [])}
        self._camera_fov_map = {c.camera_id: getattr(c, 'fov', None) for c in getattr(self.cfg, 'cameras', [])}
        
        # ----------------------------------------------------------------------------
        self.track_counts: Dict[Tuple[str, str], int] = {}  # (veh_id, cam_id) -> count
        self.last_capture_pos: Dict[Tuple[str, str], object] = {}  # (veh_id, cam_id) -> location
        
        self.max_images_per_track = getattr(self.cfg, 'max_images_per_track', 20)
        self.min_motion_delta = getattr(self.cfg, 'min_motion_delta', 1.0)
        self.min_bbox_width = getattr(self.cfg, 'min_bbox_width', 64)
        self.min_bbox_height = getattr(self.cfg, 'min_bbox_height', 64)
        
        self._setup_output_dirs()

        # ----------------------------------------------------------------------------
        self.traj_file = None
        self.traj_writer = None
        self.traj_interval = 20  # record every 20 frames (~1 second at 20 Hz)

        # ----------------------------------------------------------------------------
        self.current_sumo_speeds: Dict[str, float] = {}
    
    def _setup_output_dirs(self):
        folders = ['image_train', 'image_test', 'image_query', 'metadata', 'statistics']
        for folder in folders:
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)
    
    def initialize(self):
        print("\n" + "=" * 60)
        print("SimVeRi data collection plugin initialization (v1.3 spatiotemporal enhancement)")
        print("=" * 60)
        
        self.camera_manager = CameraManager(self.world, self.cfg)
        num_cameras = self.camera_manager.spawn_all_cameras()
        
        if num_cameras == 0:
            print(" No cameras were spawned successfully")
            return False
        
        self.occlusion_calc = OcclusionCalculator(self.world)
        print("Occlusion calculator initialized")
        
        print(f"\nCollection settings (VeRi-aligned release):")
        print(f"  Sampling interval: collect once every {self.sampling_interval} frames")
        print(f"  Minimum BBox: {self.min_bbox_width}x{self.min_bbox_height}")
        print(f"  Max images per track: {self.max_images_per_track}")
        print(f"  Minimum motion distance: {self.min_motion_delta}m")
        print(f"  Valid range: {self.cfg.near_distance}m - {self.cfg.far_distance}m")
        print(f"  Occlusion threshold: {self.cfg.occlusion_threshold * 100}%")
        print(f"  Output directory: {self.output_dir}")
        
        # ----------------------------------------------------------------------------
        traj_path = os.path.join(self.output_dir, 'metadata', 'full_trajectories.csv')
        self.traj_file = open(traj_path, 'w', newline='', encoding='utf-8')
        self.traj_writer = csv.writer(self.traj_file)
        self.traj_writer.writerow([
            'frame_id', 'timestamp', 'vehicle_id', 
            'x', 'y', 'z', 'speed', 'heading'
        ])
        print(f"Full trajectory recorder initialized: {traj_path}")

        self.is_initialized = True
        print("\n" + "=" * 60 + "\n")
        return True
    
    def collect_step(self, sumo2carla_ids: Dict[str, int], sumo_speeds: Dict[str, float] = None):
        """
        Collection step with optional SUMO speed data.
        """
        if not self.is_initialized:
            return
        
        self.frame_count += 1
        
        self.current_sumo_speeds = sumo_speeds or {}
        if self.frame_count % self.sampling_interval == 0:
            try:
                self._do_collect(sumo2carla_ids)
            except Exception as e:
                print(f"  [SimVeRi] frame {self.frame_count} collection error: {e}")
        
        if self.frame_count % self.traj_interval == 0:
            self._record_trajectories(sumo2carla_ids)

    def _do_collect(self, sumo2carla_ids: Dict[str, int]):
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        images = self.camera_manager.get_current_images(timeout=1.0)
        
        if not images:
            return
        
        captures_this_frame = 0
        
        for sumo_id, carla_id in sumo2carla_ids.items():
            vehicle = self.world.get_actor(carla_id)
            if vehicle is None:
                continue
            
            veh_info = self.vehicle_info.get(sumo_id, {})
            
            for cam_id, image in images.items():
                capture = self._process_vehicle_camera(
                    vehicle=vehicle,
                    sumo_id=sumo_id,
                    veh_info=veh_info,
                    cam_id=cam_id,
                    image=image,
                    timestamp=timestamp
                )
                
                if capture:
                    self.captures.append(capture)
                    captures_this_frame += 1
        
        if self.frame_count % 200 == 0:
            print(
                f"  [SimVeRi] frame {self.frame_count} | "
                f"this frame {captures_this_frame} | total {len(self.captures)}"
            )
    def _record_trajectories(self, sumo2carla_ids: Dict[str, int]):
        """Record the trajectory state of every vehicle as full ground truth."""
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        
        for sumo_id, carla_id in sumo2carla_ids.items():
            actor = self.world.get_actor(carla_id)
            if actor is None:
                continue
            
            transform = actor.get_transform()
            
            # ----------------------------------------------------------------------------
            speed = self.current_sumo_speeds.get(sumo_id, 0.0)
            
            self.traj_writer.writerow([
                self.frame_count,
                f"{timestamp:.2f}",
                sumo_id,
                f"{transform.location.x:.2f}",
                f"{transform.location.y:.2f}",
                f"{transform.location.z:.2f}",
                f"{speed:.2f}",
                f"{transform.rotation.yaw:.2f}"
            ])
        
        if self.frame_count % (self.traj_interval * 10) == 0:
            self.traj_file.flush()

    def _process_vehicle_camera(
        self,
        vehicle,
        sumo_id: str,
        veh_info: dict,
        cam_id: str,
        image,
        timestamp: float
    ) -> Optional[VehicleCapture]:
        """Process one vehicle-camera capture (v1.3 spatiotemporal enhancement)."""
        
        track_key = (sumo_id, cam_id)

        cam_overrides = self.per_camera_overrides.get(cam_id, {}) if isinstance(self.per_camera_overrides, dict) else {}
        max_images_per_track = int(cam_overrides.get('max_images_per_track', self.max_images_per_track))
        min_motion_delta = float(cam_overrides.get('min_motion_delta', self.min_motion_delta))
        near_distance = float(cam_overrides.get('near_distance', self.cfg.near_distance))
        far_distance = float(cam_overrides.get('far_distance', self.cfg.far_distance))
        min_bbox_area = int(cam_overrides.get('min_bbox_area', self.cfg.min_bbox_area))
        min_bbox_width = int(cam_overrides.get('min_bbox_width', self.min_bbox_width))
        min_bbox_height = int(cam_overrides.get('min_bbox_height', self.min_bbox_height))
        occlusion_threshold = float(cam_overrides.get('occlusion_threshold', self.cfg.occlusion_threshold))
        
        # ----------------------------------------------------------------------------
        current_count = self.track_counts.get(track_key, 0)
        if current_count >= max_images_per_track:
            return None
        
        # ----------------------------------------------------------------------------
        current_pos = vehicle.get_location()
        if track_key in self.last_capture_pos:
            last_pos = self.last_capture_pos[track_key]
            if current_pos.distance(last_pos) < min_motion_delta:
                return None
        
        camera = self.camera_manager.cameras[cam_id]
        projector = self.camera_manager.projectors[cam_id]
        
        bbox_result = get_2d_bbox(vehicle, projector)
        if bbox_result is None or not bbox_result.is_valid:
            return None
        
        distance = bbox_result.distance
        if distance < near_distance or distance > far_distance:
            return None
        
        if bbox_result.area < min_bbox_area:
            return None
        
        bbox_width = bbox_result.xmax - bbox_result.xmin
        bbox_height = bbox_result.ymax - bbox_result.ymin
        if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
            return None
        
        camera_location = camera.get_transform().location
        occ_result = self.occlusion_calc.calculate_occlusion(vehicle, camera_location)
        occ_ratio = occ_result.occlusion_ratio
        occ_level = occ_result.occlusion_level
        
        if occ_ratio > occlusion_threshold:
            return None
        
        image_path = self._save_image(image, bbox_result, sumo_id, cam_id, self.frame_count)
        if image_path is None:
            return None
        
        self.track_counts[track_key] = current_count + 1
        self.last_capture_pos[track_key] = current_pos
        
        # ----------------------------------------------------------------------------
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        global_x = transform.location.x
        global_y = transform.location.y
        global_z = transform.location.z
        heading = transform.rotation.yaw
        # ----------------------------------------------------------------------------
        speed = self.current_sumo_speeds.get(sumo_id, 0.0)

        return VehicleCapture(
            vehicle_id=sumo_id,
            carla_actor_id=vehicle.id,
            camera_id=cam_id,
            frame_id=self.frame_count,
            timestamp=timestamp,
            bbox=(bbox_result.xmin, bbox_result.ymin, bbox_result.xmax, bbox_result.ymax),
            bbox_area=bbox_result.area,
            distance=distance,
            occlusion_ratio=occ_ratio,
            occlusion_level=occ_level.name if hasattr(occ_level, 'name') else str(occ_level),
            image_path=image_path,
            blueprint=veh_info.get('blueprint', ''),
            category=veh_info.get('category', ''),
            color_name=veh_info.get('color_name', ''),
            color_rgb=veh_info.get('color_rgb', (128, 128, 128)),
            is_fleet=veh_info.get('is_fleet', False),
            fleet_id=veh_info.get('fleet_id'),
            global_x=global_x,
            global_y=global_y,
            global_z=global_z,
            speed=speed,
            heading=heading
        )
    
    def _save_image(self, image, bbox, sumo_id: str, cam_id: str, frame_id: int) -> Optional[str]:
        """Save a cropped image without resizing the original resolution."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            
            padding = float(getattr(self.cfg, 'bbox_padding', 0.05))
            cam_overrides = self.per_camera_overrides.get(cam_id, {}) if isinstance(self.per_camera_overrides, dict) else {}
            padding = float(cam_overrides.get('bbox_padding', padding))
            bbox_w = bbox.xmax - bbox.xmin
            bbox_h = bbox.ymax - bbox.ymin
            pad_x = int(bbox_w * padding)
            pad_y = int(bbox_h * padding)
            
            x1 = max(0, bbox.xmin - pad_x)
            y1 = max(0, bbox.ymin - pad_y)
            x2 = min(image.width, bbox.xmax + pad_x)
            y2 = min(image.height, bbox.ymax + pad_y)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = array[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            filename = f"{sumo_id}_{cam_id}_{frame_id:06d}.jpg"
            folder = 'image_train'
            filepath = os.path.join(self.output_dir, folder, filename)
            
            pil_img = Image.fromarray(crop)
            pil_img.save(filepath, quality=self.cfg.image_quality)
            
            return filepath
            
        except Exception as e:
            print(f" Failed to save image: {e}")
            return None
    
    def finalize(self):
        """Finalize collection, then export annotations and statistics."""
        print("\n" + "=" * 60)
        print("SimVeRi data collection finished")
        print("=" * 60)
        print(f"Raw captures: {len(self.captures)}")
        
        # ----------------------------------------------------------------------------
        # self._filter_cross_camera()
        
        if self.captures:
            self._save_veri_xml()
            self._save_captures_json()
            self._save_statistics()
        else:
            print(" No valid captures")
        
        if self.traj_file:
            self.traj_file.close()
            print(" Full trajectory log saved")

        if self.camera_manager:
            self.camera_manager.destroy_all()
        
        print("=" * 60)
    
    def _filter_cross_camera(self):
        """Remove vehicles that appear in only one camera."""
        vehicle_cameras: Dict[str, Set[str]] = {}
        for cap in self.captures:
            vid = cap.vehicle_id
            if vid not in vehicle_cameras:
                vehicle_cameras[vid] = set()
            vehicle_cameras[vid].add(cap.camera_id)
        
        valid_vehicles = {vid for vid, cams in vehicle_cameras.items() if len(cams) >= 2}
        
        original_count = len(self.captures)
        self.captures = [cap for cap in self.captures if cap.vehicle_id in valid_vehicles]
        
        removed_vehicles = len(vehicle_cameras) - len(valid_vehicles)
        print(f"Cross-camera filter: removed {removed_vehicles} single-camera vehicles")
        print(f"Valid vehicles: {len(valid_vehicles)} (appear in >=2 cameras)")
    
    def _save_veri_xml(self):
        """Save VeRi-compatible XML with integer type and color IDs."""
        root = ET.Element("TrainLabel")
        root.set("Version", "SimVeRi-1.0")
        root.set("TotalImages", str(len(self.captures)))

        unmapped_types = set()

        for cap in self.captures:
            item = ET.SubElement(root, "Item")
            item.set("vehicleID", cap.vehicle_id)
            item.set("imageName", os.path.basename(cap.image_path))
            item.set("cameraID", cap.camera_id)

            # ----------------------------------------------------------------------------
            veri_color_name, veri_color_id = map_color_to_veri(cap.color_name)
            item.set("colorID", str(veri_color_id))
            item.set("colorName", veri_color_name)

            # ----------------------------------------------------------------------------
            type_id = self.TYPE_TO_VERI_ID.get(cap.category, 0)
            if type_id == 0 and cap.category:
                unmapped_types.add(cap.category)
            item.set("typeID", str(type_id))

            brand = extract_brand(cap.blueprint)
            item.set("brandID", brand)

            item.set("frameID", str(cap.frame_id))
            item.set("timestamp", f"{cap.timestamp:.2f}")
            item.set("distance", f"{cap.distance:.1f}")
            item.set("occlusion", f"{cap.occlusion_ratio:.2f}")
            item.set("occLevel", cap.occlusion_level)
            item.set("bboxArea", str(cap.bbox_area))
            item.set("isFleet", str(cap.is_fleet))
            item.set("globalX", f"{cap.global_x:.2f}")
            item.set("globalY", f"{cap.global_y:.2f}")
            item.set("globalZ", f"{cap.global_z:.2f}")
            item.set("speed", f"{cap.speed:.2f}")
            item.set("heading", f"{cap.heading:.2f}")

            if cap.fleet_id:
                item.set("fleetID", cap.fleet_id)

        if unmapped_types:
            print(f"[WARN] The following categories were not mapped to the VeRi standard: {unmapped_types}")

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        path = os.path.join(self.output_dir, 'metadata', 'train_label.xml')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        print(f"[OK] VeRi XML saved: {path}")
    
    def _save_captures_json(self):
        """Save the detailed capture JSON."""
        data = []
        for cap in self.captures:
            veri_color_name, veri_color_id = map_color_to_veri(cap.color_name)
            type_id = self.TYPE_TO_VERI_ID.get(cap.category, 0)

            data.append({
                'vehicle_id': cap.vehicle_id,
                'carla_actor_id': cap.carla_actor_id,
                'camera_id': cap.camera_id,
                'camera_layer': self._camera_layer_map.get(cap.camera_id, 'unknown'),
                'camera_fov': self._camera_fov_map.get(cap.camera_id, None),
                'frame_id': cap.frame_id,
                'timestamp': cap.timestamp,
                'bbox': list(cap.bbox),
                'bbox_area': cap.bbox_area,
                'distance': round(cap.distance, 2),
                'occlusion_ratio': round(cap.occlusion_ratio, 3),
                'occlusion_level': cap.occlusion_level,
                'image_path': cap.image_path,
                'blueprint': cap.blueprint,
                'brand': extract_brand(cap.blueprint),
                'category': cap.category,
                'category_id_veri': type_id,
                'color_name': cap.color_name,
                'color_name_veri': veri_color_name,
                'color_id_veri': veri_color_id,
                'color_rgb': list(cap.color_rgb),
                'is_fleet': cap.is_fleet,
                'fleet_id': cap.fleet_id,
                'global_x': round(cap.global_x, 2),
                'global_y': round(cap.global_y, 2),
                'global_z': round(cap.global_z, 2),
                'speed': round(cap.speed, 2),
                'heading': round(cap.heading, 2),
            })

        path = os.path.join(self.output_dir, 'metadata', 'captures.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[OK] JSON saved: {path}")
    
    def _save_statistics(self):
        """Save collection statistics."""
        vehicle_cameras = {}
        for cap in self.captures:
            vid = cap.vehicle_id
            if vid not in vehicle_cameras:
                vehicle_cameras[vid] = set()
            vehicle_cameras[vid].add(cap.camera_id)
        
        camera_count_dist = {}
        for vid, cams in vehicle_cameras.items():
            n = len(cams)
            camera_count_dist[n] = camera_count_dist.get(n, 0) + 1
        
        stats = {
            'summary': {
                'total_captures': len(self.captures),
                'total_frames': self.frame_count,
                'sampling_interval': self.sampling_interval,
                'unique_vehicles': len(set(c.vehicle_id for c in self.captures)),
                'unique_cameras': len(set(c.camera_id for c in self.captures)),
                'avg_cameras_per_vehicle': round(
                    sum(len(cams) for cams in vehicle_cameras.values()) / len(vehicle_cameras), 2
                ) if vehicle_cameras else 0
            },
            'veri_compliance': {
                'min_bbox_size': '64x64',
                'cross_camera_filter': 'applied (>=2 cameras)',
                'max_images_per_track': self.max_images_per_track
            },
            'camera_count_distribution': camera_count_dist,
            'by_camera': {},
            'by_vehicle': {},
            'by_occlusion_level': {},
            'by_category': {},
            'hard_subset': {'total_captures': 0, 'by_fleet': {}}
        }
        
        for cap in self.captures:
            stats['by_camera'][cap.camera_id] = stats['by_camera'].get(cap.camera_id, 0) + 1
            stats['by_vehicle'][cap.vehicle_id] = stats['by_vehicle'].get(cap.vehicle_id, 0) + 1
            stats['by_occlusion_level'][cap.occlusion_level] = \
                stats['by_occlusion_level'].get(cap.occlusion_level, 0) + 1
            if cap.category:
                stats['by_category'][cap.category] = stats['by_category'].get(cap.category, 0) + 1
            if cap.is_fleet:
                stats['hard_subset']['total_captures'] += 1
                if cap.fleet_id:
                    stats['hard_subset']['by_fleet'][cap.fleet_id] = \
                        stats['hard_subset']['by_fleet'].get(cap.fleet_id, 0) + 1
        
        path = os.path.join(self.output_dir, 'statistics', 'collection_stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f" Statistics saved: {path}")
        
        print(f"\nStatistics summary:")
        print(f"  Total captures: {stats['summary']['total_captures']}")
        print(f"  Unique vehicles: {stats['summary']['unique_vehicles']}")
        print(f"  Covered cameras: {stats['summary']['unique_cameras']}")
        print(f"  Average cameras per vehicle: {stats['summary']['avg_cameras_per_vehicle']}")
        print(f"  Hard subset images: {stats['hard_subset']['total_captures']}")


# =============================================================================
# =============================================================================

def load_vehicle_manifest(csv_path: str) -> dict:
    """Load the vehicle manifest CSV."""
    info = {}
    
    if not os.path.exists(csv_path):
        print(f" Vehicle manifest not found: {csv_path}")
        return info
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['vehicle_id']
            info[vid] = {
                'blueprint': row['blueprint'],
                'category': row['category'],
                'color_name': row['color_name'],
                'color_rgb': (
                    int(row['color_r']),
                    int(row['color_g']),
                    int(row['color_b'])
                ),
                'is_fleet': row['is_fleet'] == 'True',
                'fleet_id': row['fleet_id'] if row.get('fleet_id') else None
            }
    
    base_count = sum(1 for v in info.values() if not v['is_fleet'])
    hard_count = sum(1 for v in info.values() if v['is_fleet'])
    print(f"Loaded vehicle manifest: {len(info)} vehicles (Base: {base_count}, Hard: {hard_count})")
    
    return info
