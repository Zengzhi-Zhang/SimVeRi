"""
SimVeRi bounding-box utilities.

This module converts 3D CARLA vehicle geometry into 2D image-space bounding
boxes for released crops and metadata export.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import carla
import numpy as np

from config_loader import get_config


@dataclass
class BBox2D:
    """Simple 2D bounding-box container."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    is_valid: bool = True
    is_edge_case: bool = False
    vertices_in_frame: int = 8
    distance: float = 0.0

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2)

    def to_string(self) -> str:
        """Return the box as ``xmin,ymin,w,h``."""

        return f"{self.xmin},{self.ymin},{self.width},{self.height}"


def get_matrix(transform: carla.Transform) -> np.ndarray:
    """Convert a CARLA transform into a 4x4 local-to-world matrix."""

    rotation = transform.rotation
    location = transform.location

    c_y = math.cos(math.radians(rotation.yaw))
    s_y = math.sin(math.radians(rotation.yaw))
    c_r = math.cos(math.radians(rotation.roll))
    s_r = math.sin(math.radians(rotation.roll))
    c_p = math.cos(math.radians(rotation.pitch))
    s_p = math.sin(math.radians(rotation.pitch))

    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


class CameraProjector:
    """Project world points into a camera image plane."""

    def __init__(
        self,
        camera_actor: carla.Sensor,
        image_width: int = None,
        image_height: int = None,
    ):
        self.camera = camera_actor

        try:
            cfg = get_config()
            self.image_width = image_width or cfg.resolution[0]
            self.image_height = image_height or cfg.resolution[1]
        except Exception:
            self.image_width = image_width or 1920
            self.image_height = image_height or 1080

        self.fov = float(camera_actor.attributes["fov"])
        self.K = self._build_intrinsic_matrix()
        self._cached_transform = None
        self._cached_world2camera = None

    def _build_intrinsic_matrix(self) -> np.ndarray:
        focal = self.image_width / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0

        K = np.identity(3)
        K[0, 0] = focal
        K[1, 1] = focal
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def _build_world2camera_matrix(self, transform: carla.Transform) -> np.ndarray:
        """Build a world-to-camera matrix with CARLA-to-OpenCV axis correction."""

        camera_model_matrix = get_matrix(transform)
        world_2_camera = np.linalg.inv(camera_model_matrix)

        correction = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        return np.dot(correction, world_2_camera)

    def get_world2camera_matrix(self) -> np.ndarray:
        """Return the cached world-to-camera matrix for the current camera pose."""

        current_transform = self.camera.get_transform()
        if (
            self._cached_transform is None
            or current_transform.location != self._cached_transform.location
            or current_transform.rotation != self._cached_transform.rotation
        ):
            self._cached_transform = current_transform
            self._cached_world2camera = self._build_world2camera_matrix(current_transform)

        return self._cached_world2camera

    def get_camera_location(self) -> carla.Location:
        return self.camera.get_transform().location

    def world_to_pixel(self, world_point: carla.Location) -> Tuple[int, int, bool]:
        """Project a world-space point into pixel coordinates."""

        world2cam = self.get_world2camera_matrix()
        world_vec = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        cam_vec = np.dot(world2cam, world_vec)

        if cam_vec[2] <= 0:
            return (0, 0, False)

        x_norm = cam_vec[0] / cam_vec[2]
        y_norm = cam_vec[1] / cam_vec[2]

        pixel_x = int(self.K[0, 0] * x_norm + self.K[0, 2])
        pixel_y = int(self.K[1, 1] * y_norm + self.K[1, 2])
        return (pixel_x, pixel_y, True)

    def is_point_in_frame(self, pixel_x: int, pixel_y: int) -> bool:
        return 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height


def get_vehicle_bbox_3d(vehicle: carla.Vehicle) -> List[carla.Location]:
    bbox = vehicle.bounding_box
    return list(bbox.get_world_vertices(vehicle.get_transform()))


def get_vehicle_center(vehicle: carla.Vehicle) -> carla.Location:
    return vehicle.get_transform().transform(vehicle.bounding_box.location)


def calculate_distance(point1: carla.Location, point2: carla.Location) -> float:
    return point1.distance(point2)


def get_2d_bbox(vehicle, projector, config=None) -> BBox2D:
    """Compute a clipped 2D box for a vehicle under the active camera."""

    if config is None:
        try:
            config = get_config()
        except Exception:
            class DefaultConfig:
                near_distance = 2.0
                far_distance = 150.0
                min_bbox_area = 100

            config = DefaultConfig()

    vehicle_location = get_vehicle_center(vehicle)
    camera_location = projector.get_camera_location()
    distance = calculate_distance(vehicle_location, camera_location)

    if distance > config.far_distance or distance < config.near_distance:
        return BBox2D(0, 0, 0, 0, False, False, 0, distance)

    vertices = get_vehicle_bbox_3d(vehicle)
    valid_points = []
    for vertex in vertices:
        pixel_x, pixel_y, valid = projector.world_to_pixel(vertex)
        if valid:
            valid_points.append((pixel_x, pixel_y))

    if not valid_points:
        return BBox2D(0, 0, 0, 0, False, False, 0, distance)

    xs = [point[0] for point in valid_points]
    ys = [point[1] for point in valid_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    in_frame = sum(
        1 for point in valid_points if projector.is_point_in_frame(point[0], point[1])
    )

    width = projector.image_width
    height = projector.image_height
    xmin_clipped = max(0, min(xmin, width - 1))
    xmax_clipped = max(0, min(xmax, width - 1))
    ymin_clipped = max(0, min(ymin, height - 1))
    ymax_clipped = max(0, min(ymax, height - 1))

    box_width = xmax_clipped - xmin_clipped
    box_height = ymax_clipped - ymin_clipped
    if box_width * box_height < config.min_bbox_area:
        return BBox2D(
            xmin_clipped,
            ymin_clipped,
            xmax_clipped,
            ymax_clipped,
            False,
            False,
            in_frame,
            distance,
        )

    is_edge_case = in_frame < 8
    return BBox2D(
        xmin_clipped,
        ymin_clipped,
        xmax_clipped,
        ymax_clipped,
        True,
        is_edge_case,
        in_frame,
        distance,
    )


def get_all_vehicles_bbox(vehicles, projector, config=None):
    """Return valid bounding boxes for all visible vehicles."""

    results = []
    for vehicle in vehicles:
        bbox = get_2d_bbox(vehicle, projector, config)
        if bbox.is_valid:
            results.append(bbox)
    return results
