"""
SimVeRi occlusion estimation module v1.1 (patched edition)
Purpose: estimate vehicle occlusion from a camera viewpoint
Method: ray sampling over the 3D bounding box
"""

import carla
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

from config_loader import get_config
from bbox_utils import (
    BBox2D, CameraProjector, 
    get_vehicle_bbox_3d, get_vehicle_center, calculate_distance
)


class OcclusionLevel(Enum):
    """Occlusion severity level."""
    NONE = 0        # none (0-10%)
    LOW = 1         # light (10-30%)
    MEDIUM = 2      # medium (30-50%)
    HIGH = 3        # high (50-80%)
    SEVERE = 4      # severe (>80%)


@dataclass
class OcclusionResult:
    """Occlusion estimation result."""
    occlusion_ratio: float          # ratio in [0.0, 1.0]
    occlusion_level: OcclusionLevel # discrete severity level
    total_rays: int                 # total number of sampled rays
    blocked_rays: int               # rays blocked before reaching the target
    occluding_labels: List[int]     # semantic labels of occluding objects
    is_valid: bool = True           # whether the estimate is valid
    
    def to_dict(self) -> dict:
        return {
            'occlusion_ratio': round(self.occlusion_ratio, 3),
            'occlusion_level': self.occlusion_level.name,
            'total_rays': self.total_rays,
            'blocked_rays': self.blocked_rays,
            'occluding_labels': self.occluding_labels,
            'is_valid': self.is_valid
        }


def get_occlusion_level(ratio: float) -> OcclusionLevel:
    """Map an occlusion ratio to a discrete severity level."""
    if ratio < 0.10:
        return OcclusionLevel.NONE
    elif ratio < 0.30:
        return OcclusionLevel.LOW
    elif ratio < 0.50:
        return OcclusionLevel.MEDIUM
    elif ratio < 0.80:
        return OcclusionLevel.HIGH
    else:
        return OcclusionLevel.SEVERE


class OcclusionCalculator:
    """
    Occlusion calculator.
    Uses ray casting to estimate how much of a target vehicle is occluded.
    """
    
    def __init__(self, world: carla.World, sample_density: int = 5):
        """
        Initialize the occlusion calculator.
        
        Args:
            world: CARLA world object
            sample_density: number of sample points per dimension
        """
        self.world = world
        self.sample_density = sample_density
        
        try:
            cfg = get_config()
            self.occlusion_threshold = cfg.occlusion_threshold
        except:
            self.occlusion_threshold = 0.80
    
    def _get_sample_points_on_bbox(
        self, 
        vehicle: carla.Vehicle
    ) -> List[carla.Location]:
        """
        Generate uniformly distributed sample points on the vehicle bounding box.
        """
        bbox = vehicle.bounding_box
        transform = vehicle.get_transform()
        extent = bbox.extent
        bbox_center = bbox.location
        
        sample_points = []
        n = self.sample_density
        
        for i in range(n):
            for j in range(n):
                u = -1 + 2 * i / (n - 1) if n > 1 else 0
                v = -1 + 2 * j / (n - 1) if n > 1 else 0
                
                sample_points.append(carla.Location(
                    x=extent.x + bbox_center.x,
                    y=u * extent.y + bbox_center.y,
                    z=v * extent.z + bbox_center.z
                ))
                
                sample_points.append(carla.Location(
                    x=-extent.x + bbox_center.x,
                    y=u * extent.y + bbox_center.y,
                    z=v * extent.z + bbox_center.z
                ))
                
                sample_points.append(carla.Location(
                    x=u * extent.x + bbox_center.x,
                    y=v * extent.y + bbox_center.y,
                    z=extent.z + bbox_center.z
                ))
                
                sample_points.append(carla.Location(
                    x=u * extent.x + bbox_center.x,
                    y=-extent.y + bbox_center.y,
                    z=v * extent.z + bbox_center.z
                ))
                
                sample_points.append(carla.Location(
                    x=u * extent.x + bbox_center.x,
                    y=extent.y + bbox_center.y,
                    z=v * extent.z + bbox_center.z
                ))
        
        world_points = [transform.transform(p) for p in sample_points]
        return world_points
    
    def _cast_ray(
        self,
        start: carla.Location,
        end: carla.Location,
        target_vehicle_id: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Cast a ray and detect whether it is blocked.
        In CARLA 0.9.13 a LabelledPoint exposes only location and label.
        """
        hit_list = self.world.cast_ray(start, end)
        
        if not hit_list:
            return (False, None)
        
        start_to_end_dist = start.distance(end)
        
        for hit in hit_list:
            hit_distance = start.distance(hit.location)
            
            if hit_distance < start_to_end_dist * 0.90:
                return (True, int(hit.label))
        
        return (False, None)
    
    def calculate_occlusion(
        self,
        vehicle: carla.Vehicle,
        camera_location: carla.Location
    ) -> OcclusionResult:
        """
        Estimate the occlusion ratio for one vehicle.
        """
        vehicle_id = vehicle.id
        
        sample_points = self._get_sample_points_on_bbox(vehicle)
        
        if not sample_points:
            return OcclusionResult(
                occlusion_ratio=0.0,
                occlusion_level=OcclusionLevel.NONE,
                total_rays=0,
                blocked_rays=0,
                occluding_labels=[],
                is_valid=False
            )
        
        total_rays = len(sample_points)
        blocked_rays = 0
        occluding_labels = set()
        
        for point in sample_points:
            is_blocked, label = self._cast_ray(
                camera_location, point, vehicle_id
            )
            
            if is_blocked:
                blocked_rays += 1
                if label is not None:
                    occluding_labels.add(label)
        
        occlusion_ratio = blocked_rays / total_rays if total_rays > 0 else 0.0
        occlusion_level = get_occlusion_level(occlusion_ratio)
        
        return OcclusionResult(
            occlusion_ratio=occlusion_ratio,
            occlusion_level=occlusion_level,
            total_rays=total_rays,
            blocked_rays=blocked_rays,
            occluding_labels=list(occluding_labels),
            is_valid=True
        )
    
    def calculate_batch_occlusion(
        self,
        vehicles: List[carla.Vehicle],
        camera_location: carla.Location
    ) -> Dict[int, OcclusionResult]:
        """Estimate occlusion ratios for multiple vehicles."""
        results = {}
        for vehicle in vehicles:
            result = self.calculate_occlusion(vehicle, camera_location)
            results[vehicle.id] = result
        return results
    
    def is_acceptable(self, occlusion_result: OcclusionResult) -> bool:
        """Return whether an occlusion result is acceptable."""
        return occlusion_result.occlusion_ratio <= self.occlusion_threshold


class OptimizedOcclusionCalculator(OcclusionCalculator):
    """
    Optimized occlusion calculator.
    Uses adaptive sampling and visible-face detection.
    """
    
    def __init__(self, world: carla.World, base_density: int = 3):
        super().__init__(world, base_density)
        self.base_density = base_density
    
    # def _get_visible_faces(
    #     self,
    #     vehicle: carla.Vehicle,
    #     camera_location: carla.Location
    # ) -> List[str]:
    #     vehicle_center = get_vehicle_center(vehicle)
    #     vehicle_transform = vehicle.get_transform()
        
    #     dx = camera_location.x - vehicle_center.x
    #     dy = camera_location.y - vehicle_center.y
    #     dz = camera_location.z - vehicle_center.z
        
    #     yaw = np.radians(vehicle_transform.rotation.yaw)
    #     local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
    #     local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
    #     local_z = dz
        
    #     visible_faces = []
        
    #     if local_x > 0:
    #         visible_faces.append('front')
    #     else:
    #         visible_faces.append('back')
        
    #     if local_y > 0:
    #         visible_faces.append('right')
    #     else:
    #         visible_faces.append('left')
        
    #     if local_z > 0:
    #         visible_faces.append('top')
        
    #     return visible_faces

    def _get_visible_faces(
        self,
        vehicle: carla.Vehicle,
        camera_location: carla.Location
    ) -> List[str]:
        """
        Determine which box faces are visible from the camera viewpoint.
        The method projects the camera position onto the vehicle local axes.
        """
        vehicle_center = get_vehicle_center(vehicle)
        vehicle_transform = vehicle.get_transform()
        
        # Vector = Camera - Vehicle
        diff_x = camera_location.x - vehicle_center.x
        diff_y = camera_location.y - vehicle_center.y
        diff_z = camera_location.z - vehicle_center.z
        
        fwd = vehicle_transform.get_forward_vector()  # Local X (forward)
        right = vehicle_transform.get_right_vector()  # Local Y (right)
        up = vehicle_transform.get_up_vector()        # Local Z (up)
        
        local_x = diff_x * fwd.x + diff_y * fwd.y + diff_z * fwd.z
        local_y = diff_x * right.x + diff_y * right.y + diff_z * right.z
        local_z = diff_x * up.x + diff_y * up.y + diff_z * up.z
        
        visible_faces = []
        
        
        if local_x > 0:
            visible_faces.append('front')
        else:
            visible_faces.append('back')
        
        if local_y > 0:
            visible_faces.append('right')
        else:
            visible_faces.append('left')
        
        if local_z > 0:
            visible_faces.append('top')
        
        return visible_faces


    def _get_optimized_sample_points(
        self,
        vehicle: carla.Vehicle,
        camera_location: carla.Location
    ) -> List[carla.Location]:
        """Optimized sampling that keeps only visible faces."""
        bbox = vehicle.bounding_box
        transform = vehicle.get_transform()
        extent = bbox.extent
        bbox_center = bbox.location
        
        visible_faces = self._get_visible_faces(vehicle, camera_location)
        
        vehicle_center = get_vehicle_center(vehicle)
        distance = calculate_distance(vehicle_center, camera_location)
        
        if distance < 20:
            n = self.base_density + 2
        elif distance < 50:
            n = self.base_density + 1
        else:
            n = self.base_density
        
        sample_points = []
        
        for i in range(n):
            for j in range(n):
                u = -1 + 2 * i / (n - 1) if n > 1 else 0
                v = -1 + 2 * j / (n - 1) if n > 1 else 0
                
                if 'front' in visible_faces:
                    sample_points.append(carla.Location(
                        x=extent.x + bbox_center.x,
                        y=u * extent.y + bbox_center.y,
                        z=v * extent.z + bbox_center.z
                    ))
                
                if 'back' in visible_faces:
                    sample_points.append(carla.Location(
                        x=-extent.x + bbox_center.x,
                        y=u * extent.y + bbox_center.y,
                        z=v * extent.z + bbox_center.z
                    ))
                
                if 'left' in visible_faces:
                    sample_points.append(carla.Location(
                        x=u * extent.x + bbox_center.x,
                        y=-extent.y + bbox_center.y,
                        z=v * extent.z + bbox_center.z
                    ))
                
                if 'right' in visible_faces:
                    sample_points.append(carla.Location(
                        x=u * extent.x + bbox_center.x,
                        y=extent.y + bbox_center.y,
                        z=v * extent.z + bbox_center.z
                    ))
                
                if 'top' in visible_faces:
                    sample_points.append(carla.Location(
                        x=u * extent.x + bbox_center.x,
                        y=v * extent.y + bbox_center.y,
                        z=extent.z + bbox_center.z
                    ))
        
        world_points = [transform.transform(p) for p in sample_points]
        return world_points
    
    def calculate_occlusion(
        self,
        vehicle: carla.Vehicle,
        camera_location: carla.Location
    ) -> OcclusionResult:
        """Optimized occlusion estimation."""
        vehicle_id = vehicle.id
        
        sample_points = self._get_optimized_sample_points(vehicle, camera_location)
        
        if not sample_points:
            return OcclusionResult(
                occlusion_ratio=0.0,
                occlusion_level=OcclusionLevel.NONE,
                total_rays=0,
                blocked_rays=0,
                occluding_labels=[],
                is_valid=False
            )
        
        total_rays = len(sample_points)
        blocked_rays = 0
        occluding_labels = set()
        
        for point in sample_points:
            is_blocked, label = self._cast_ray(
                camera_location, point, vehicle_id
            )
            
            if is_blocked:
                blocked_rays += 1
                if label is not None:
                    occluding_labels.add(label)
        
        occlusion_ratio = blocked_rays / total_rays if total_rays > 0 else 0.0
        occlusion_level = get_occlusion_level(occlusion_ratio)
        
        return OcclusionResult(
            occlusion_ratio=occlusion_ratio,
            occlusion_level=occlusion_level,
            total_rays=total_rays,
            blocked_rays=blocked_rays,
            occluding_labels=list(occluding_labels),
            is_valid=True
        )


# =============================================================================
# =============================================================================

def test_occlusion_level():
    """Test the occlusion-level thresholds."""
    assert get_occlusion_level(0.05) == OcclusionLevel.NONE
    assert get_occlusion_level(0.15) == OcclusionLevel.LOW
    assert get_occlusion_level(0.40) == OcclusionLevel.MEDIUM
    assert get_occlusion_level(0.65) == OcclusionLevel.HIGH
    assert get_occlusion_level(0.90) == OcclusionLevel.SEVERE
    print("[OK] test_occlusion_level passed")


def test_occlusion_result():
    """Test the occlusion-result data structure."""
    result = OcclusionResult(
        occlusion_ratio=0.35,
        occlusion_level=OcclusionLevel.MEDIUM,
        total_rays=100,
        blocked_rays=35,
        occluding_labels=[10, 4],
        is_valid=True
    )
    
    d = result.to_dict()
    assert d['occlusion_ratio'] == 0.35
    assert d['occlusion_level'] == 'MEDIUM'
    assert d['blocked_rays'] == 35
    print("[OK] test_occlusion_result passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("occlusion.py unit tests")
    print("=" * 60)
    
    test_occlusion_level()
    test_occlusion_result()
    
    print("\n" + "=" * 60)
    print("[OK] All unit tests passed")
    print("=" * 60)
    print("\nTip: run test_occlusion_carla.py for integration testing")



if __name__ == "__main__":
    run_all_tests()
