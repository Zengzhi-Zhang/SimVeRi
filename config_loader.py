"""
SimVeRi configuration loader.

This module centralizes configuration loading so that runtime code does not
depend on hard-coded paths or parameters.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    camera_id: str
    position: Tuple[float, float, float]
    rotation: Dict[str, float]
    fov: int
    node: str
    group: str
    camera_type: str
    layer: str
    capture_type: str
    target_vehicle_yaw: float
    description: str


@dataclass
class VehicleBlueprint:
    """Vehicle blueprint configuration."""

    blueprint: str
    weight: float
    category: str
    description: str


@dataclass
class ColorConfig:
    """Color configuration."""

    name: str
    rgb: Tuple[int, int, int]
    weight: float


@dataclass
class SimVeRiConfig:
    """Global SimVeRi configuration."""

    carla_version: str
    map_name: str
    synchronous_mode: bool
    fixed_delta_seconds: float
    weather: str

    camera_count: int
    resolution: Tuple[int, int]
    fov: int
    sensor_type: str

    physics_fps: int
    sampling_fps: int
    sampling_interval: int
    image_format: str
    image_quality: int
    occlusion_threshold: float
    min_bbox_area: int
    min_bbox_width: int
    min_bbox_height: int
    max_images_per_track: int
    min_motion_delta: float
    bbox_padding: float
    per_camera_overrides: Dict[str, Dict[str, Any]]
    near_distance: float
    far_distance: float

    dataset_name: str
    dataset_root: str

    vehicles_per_batch: int
    spawn_interval: float

    mvp_duration: int
    production_batches: int
    production_duration: int

    bbox_deviation_max: int
    min_samples_per_camera: int
    max_speed_threshold: float

    cameras: List[CameraConfig] = field(default_factory=list)
    vehicle_blueprints: List[VehicleBlueprint] = field(default_factory=list)
    colors: List[ColorConfig] = field(default_factory=list)
    distance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls, config_path: str = "config.yaml", topology_path: Optional[str] = None
    ) -> "SimVeRiConfig":
        """Load the configuration from YAML and camera-topology files."""

        with open(config_path, "r", encoding="utf-8") as file_obj:
            cfg = yaml.safe_load(file_obj)

        if topology_path is None:
            topology_path = os.path.join(
                cfg["dataset"]["root"],
                cfg["camera"]["topology_file"],
            )

        with open(topology_path, "r", encoding="utf-8") as file_obj:
            topo = json.load(file_obj)

        cameras = []
        for cam_id, cam_data in topo["cameras"].items():
            cameras.append(
                CameraConfig(
                    camera_id=cam_id,
                    position=tuple(cam_data["position"]),
                    rotation=cam_data["rotation"],
                    fov=cam_data.get("fov", 90),
                    node=cam_data.get("node", ""),
                    group=cam_data.get("group", ""),
                    camera_type=cam_data.get("type", ""),
                    layer=cam_data.get("layer", "ground"),
                    capture_type=cam_data.get("capture_type", "rear"),
                    target_vehicle_yaw=cam_data.get("target_vehicle_yaw", 0),
                    description=cam_data.get("description", ""),
                )
            )
        cameras.sort(key=lambda camera: camera.camera_id)

        blueprints = []
        for blueprint in cfg["traffic"]["vehicle_blueprints"]:
            blueprints.append(
                VehicleBlueprint(
                    blueprint=blueprint["blueprint"],
                    weight=blueprint["weight"],
                    category=blueprint.get("category", "unknown"),
                    description=blueprint.get("description", ""),
                )
            )

        colors = []
        for color in cfg["traffic"]["color_distribution"]["colors"]:
            rgb_raw = color["rgb"]
            rgb = tuple(map(int, rgb_raw.split(","))) if isinstance(rgb_raw, str) else tuple(rgb_raw)
            colors.append(
                ColorConfig(
                    name=color["name"],
                    rgb=rgb,
                    weight=color["weight"],
                )
            )

        physics_fps = cfg["acquisition"]["physics_fps"]
        sampling_fps = cfg["acquisition"]["sampling_fps"]
        sampling_interval = cfg["acquisition"].get(
            "sampling_interval", physics_fps // sampling_fps
        )

        acq_cfg = cfg.get("acquisition", {}) or {}
        min_bbox_width = int(acq_cfg.get("min_bbox_width", 64))
        min_bbox_height = int(acq_cfg.get("min_bbox_height", 64))
        max_images_per_track = int(acq_cfg.get("max_images_per_track", 20))
        min_motion_delta = float(acq_cfg.get("min_motion_delta", 1.0))
        bbox_padding = float(acq_cfg.get("bbox_padding", 0.05))
        per_camera_overrides = acq_cfg.get("per_camera_overrides", {}) or {}

        return cls(
            carla_version=cfg["simulation"]["carla_version"],
            map_name=cfg["simulation"]["map"],
            synchronous_mode=cfg["simulation"]["synchronous_mode"],
            fixed_delta_seconds=cfg["simulation"]["fixed_delta_seconds"],
            weather=cfg["simulation"]["weather"],
            camera_count=cfg["camera"]["count"],
            resolution=(
                cfg["camera"]["resolution"]["width"],
                cfg["camera"]["resolution"]["height"],
            ),
            fov=cfg["camera"]["fov"],
            sensor_type=cfg["camera"]["sensor_type"],
            physics_fps=physics_fps,
            sampling_fps=sampling_fps,
            sampling_interval=sampling_interval,
            image_format=cfg["acquisition"]["image_format"],
            image_quality=cfg["acquisition"]["image_quality"],
            occlusion_threshold=cfg["acquisition"]["occlusion_threshold"],
            min_bbox_area=cfg["acquisition"]["min_bbox_area"],
            min_bbox_width=min_bbox_width,
            min_bbox_height=min_bbox_height,
            max_images_per_track=max_images_per_track,
            min_motion_delta=min_motion_delta,
            bbox_padding=bbox_padding,
            per_camera_overrides=per_camera_overrides,
            near_distance=cfg["acquisition"]["near_distance"],
            far_distance=cfg["acquisition"]["far_distance"],
            dataset_name=cfg["dataset"]["name"],
            dataset_root=cfg["dataset"]["root"],
            vehicles_per_batch=cfg["traffic"]["spawn"]["vehicles_per_batch"],
            spawn_interval=cfg["traffic"]["spawn"]["spawn_interval"],
            mvp_duration=cfg["batch"]["mvp"]["duration_seconds"],
            production_batches=cfg["batch"]["production"]["total_batches"],
            production_duration=cfg["batch"]["production"]["duration_per_batch"],
            bbox_deviation_max=cfg["validation"]["bbox_deviation_max"],
            min_samples_per_camera=cfg["validation"]["min_samples_per_camera"],
            max_speed_threshold=cfg["validation"]["max_speed_threshold"],
            cameras=cameras,
            vehicle_blueprints=blueprints,
            colors=colors,
            distance_matrix=topo.get("distance_matrix", {}),
        )

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Return the configuration for a specific camera."""

        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

    def get_distance(self, from_cam: str, to_cam: str) -> float:
        """Return the path distance between two cameras."""

        if from_cam in self.distance_matrix:
            return self.distance_matrix[from_cam].get(to_cam, -1)
        return -1

    def get_output_path(self, *paths: str) -> str:
        """Build an output path under the dataset root."""

        return os.path.join(self.dataset_root, *paths)


_config: Optional[SimVeRiConfig] = None


def get_config(reload: bool = False) -> SimVeRiConfig:
    """Return the cached global configuration instance."""

    global _config
    if _config is None or reload:
        _config = SimVeRiConfig.from_files()
    return _config


def print_config_summary() -> None:
    """Print a short human-readable summary of the active configuration."""

    cfg = get_config()

    print("=" * 60)
    print("SimVeRi configuration summary")
    print("=" * 60)

    print("\n[Simulation]")
    print(f"  Map: {cfg.map_name}")
    print(f"  Synchronous mode: {cfg.synchronous_mode}")
    print(f"  Fixed step length: {cfg.fixed_delta_seconds}s ({cfg.physics_fps} Hz)")

    print("\n[Cameras]")
    print(f"  Count: {cfg.camera_count}")
    print(f"  Resolution: {cfg.resolution[0]}x{cfg.resolution[1]}")
    print(f"  FOV: {cfg.fov} deg")

    print("\n[Acquisition]")
    print(
        f"  Sampling rate: {cfg.sampling_fps} FPS "
        f"(save one image every {cfg.sampling_interval} frames)"
    )
    print(f"  Image format: {cfg.image_format.upper()} (quality {cfg.image_quality})")
    print(f"  Valid range: {cfg.near_distance}m - {cfg.far_distance}m")

    print("\n[Vehicles]")
    print(f"  Blueprint count: {len(cfg.vehicle_blueprints)}")
    print(f"  Color variants: {len(cfg.colors)}")
    print(f"  Theoretical combinations: {len(cfg.vehicle_blueprints) * len(cfg.colors)}")
    print(f"  Vehicles per batch: {cfg.vehicles_per_batch}")

    categories: Dict[str, int] = {}
    for blueprint in cfg.vehicle_blueprints:
        categories[blueprint.category] = categories.get(blueprint.category, 0) + 1
    print(f"  Category distribution: {categories}")

    print("\n[Output]")
    print(f"  Dataset name: {cfg.dataset_name}")
    print(f"  Dataset root: {cfg.dataset_root}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config_summary()

    cfg = get_config()

    print("\n[Self-test]")

    camera = cfg.get_camera("c001")
    if camera:
        print(f"  c001 position: {camera.position}")
        print(f"  c001 description: {camera.description}")

    distance = cfg.get_distance("c001", "c002")
    print(f"  c001->c002 distance: {distance}m")

    sample_path = cfg.get_output_path("image_train", "test.jpg")
    print(f"  Example output path: {sample_path}")

    print("\n[OK] config_loader.py self-test passed.")
