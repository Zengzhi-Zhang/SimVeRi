"""
SimVeRi traffic generator v3.0 (SUMO co-simulation + color jitter support)
Purpose: read config.yaml and generate SUMO .rou.xml files
Adds color jitter to increase visual similarity challenges
"""

import random
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config_loader import get_config


@dataclass
class GeneratedVehicle:
    """Generated vehicle description (v3.0 enhanced release)."""
    vehicle_id: str
    blueprint: str
    color_rgb: Tuple[int, int, int]        # actual RGB after jitter
    color_base_rgb: Tuple[int, int, int]   # base RGB before jitter
    color_name: str                         # color name
    color_family: str                       # color family
    category: str
    depart_time: float
    route_id: str
    is_fleet: bool = False
    fleet_id: Optional[str] = None


class SUMOTrafficGenerator:
    """
    SUMO traffic generator v3.0.
    Supports color jitter to increase visual similarity challenges.
    """
    
    def __init__(self, output_dir: str = "sumo"):
        self.cfg = get_config()
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.routes_root = ET.Element("routes")
        
        self.defined_vtypes = set()
        
        self.generated_vehicles: List[GeneratedVehicle] = []
        
        self.stats = {
            'base_count': 0,
            'hard_count': 0,
            'by_category': {},
            'by_color': {},
            'by_color_family': {}
        }
    
    # =========================================================================
    # =========================================================================
    
    def _apply_color_jitter(
        self, 
        base_rgb: Tuple[int, int, int], 
        intensity: int = 20
    ) -> Tuple[int, int, int]:
        """
        Add random noise around a base color.
        
        Args:
            base_rgb: base RGB value
            intensity: jitter amplitude (0-255), typically 15-30
            
        Returns:
            jittered RGB tuple
        """
        r, g, b = base_rgb
        
        mode = getattr(self.cfg, 'jitter', None)
        if mode and hasattr(mode, 'mode'):
            mode = mode.mode
        else:
            mode = 'independent'
        
        if mode == 'brightness':
            delta = random.randint(-intensity, intensity)
            dr, dg, db = delta, delta, delta
        else:
            dr = random.randint(-intensity, intensity)
            dg = random.randint(-intensity, intensity)
            db = random.randint(-intensity, intensity)
        
        new_r = min(255, max(0, r + dr))
        new_g = min(255, max(0, g + dg))
        new_b = min(255, max(0, b + db))
        
        return (new_r, new_g, new_b)
    
    def _get_jitter_intensity(self, for_hard: bool = False) -> int:
        """Return the jitter intensity."""
        if hasattr(self.cfg, 'jitter') and self.cfg.jitter:
            if for_hard:
                return getattr(self.cfg.jitter, 'hard_intensity', 25)
            else:
                return getattr(self.cfg.jitter, 'base_intensity', 20)
        return 20  # default value
    
    def _is_jitter_enabled(self) -> bool:
        """Return whether color jitter is enabled."""
        if hasattr(self.cfg, 'jitter_enabled'):
            return self.cfg.jitter_enabled
        if hasattr(self.cfg, 'jitter') and self.cfg.jitter:
            return getattr(self.cfg.jitter, 'enabled', True)
        return True  # enabled by default
    
    # =========================================================================
    # =========================================================================
    
    def _get_random_blueprint(self) -> Tuple[str, str]:
        """
        Select a vehicle type by weighted random sampling.
        Returns: (blueprint_id, category)
        """
        bps = self.cfg.vehicle_blueprints
        choices = [(bp.blueprint, bp.category) for bp in bps]
        weights = [bp.weight for bp in bps]
        
        return random.choices(choices, weights=weights, k=1)[0]
    
    def _get_random_color(
        self, 
        apply_jitter: bool = True, 
        jitter_intensity: int = None
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], str, str]:
        """
        Select a color by weighted random sampling and optionally apply jitter.
        
        Args:
            apply_jitter: whether to apply color jitter
            jitter_intensity: jitter strength; use the config value when None
            
        Returns:
            (actual_rgb, base_rgb, color_name, color_family)
        """
        colors = self.cfg.colors
        
        choices = []
        weights = []
        for c in colors:
            color_family = getattr(c, 'color_family', '') or ''
            choices.append((c.rgb, c.name, color_family))
            weights.append(c.weight)
        
        base_rgb, color_name, color_family = random.choices(choices, weights=weights, k=1)[0]
        
        if apply_jitter and self._is_jitter_enabled():
            if jitter_intensity is None:
                jitter_intensity = self._get_jitter_intensity(for_hard=False)
            actual_rgb = self._apply_color_jitter(base_rgb, jitter_intensity)
        else:
            actual_rgb = base_rgb
        
        return (actual_rgb, base_rgb, color_name, color_family)
    
    def _rgb_to_sumo_color(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to the SUMO color format."""
        return f"{rgb[0]},{rgb[1]},{rgb[2]}"

    
    def _register_vtype(self, blueprint_id: str, category: str = "sedan", vehicle_type: str = "base"):
        """
        Register a vehicle type in SUMO.
        """
        vtype_id = f"{blueprint_id}_{vehicle_type}"
        
        if vtype_id in self.defined_vtypes:
            return vtype_id
        
        vtype = ET.SubElement(self.routes_root, "vType")
        vtype.set("id", vtype_id)
        vtype.set("vClass", "passenger")
        
        # ----------------------------------------------------------------------------
        shape_map = {
            'sedan': 'passenger',
            'suv': 'passenger/van',
            'hatchback': 'passenger/hatchback',
            'van': 'delivery',
            'truck': 'truck',
            'bus': 'bus',
            'coupe': 'passenger'
        }
        gui_shape = shape_map.get(category, 'passenger')
        vtype.set("guiShape", gui_shape)
        
        if category in ["suv", "van"]:
            vtype.set("length", "5.0")
            vtype.set("width", "2.0")
            vtype.set("height", "1.8")
        elif category == "hatchback":
            vtype.set("length", "4.0")
            vtype.set("width", "1.8")
            vtype.set("height", "1.5")
        else:  # sedan, coupe
            vtype.set("length", "4.5")
            vtype.set("width", "1.8")
            vtype.set("height", "1.5")
        
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        
        # ----------------------------------------------------------------------------
        if vehicle_type == "hard":
            vtype.set("speedFactor", "normc(1.0,0.1,0.8,1.2)")
            vtype.set("speedDev", "0.1")
            vtype.set("sigma", "0.5")
            vtype.set("minGap", "2.5")
            vtype.set("tau", "1.0")
        elif vehicle_type == "occ":
            vtype.set("speedFactor", "normc(0.9,0.1,0.7,1.1)")
            vtype.set("speedDev", "0.1")
            vtype.set("sigma", "0.8")
            vtype.set("minGap", "1.5")
            vtype.set("tau", "1.2")
        else:
            vtype.set("speedFactor", "normc(1.0,0.1,0.8,1.2)")
            vtype.set("speedDev", "0.1")
            vtype.set("sigma", "0.5")
            vtype.set("minGap", "2.5")
            vtype.set("tau", "1.0")
        
        self.defined_vtypes.add(vtype_id)
        return vtype_id

    def add_route(self, route_id: str, edges: List[str]):
        """
        Add a route definition.
        
        Args:
            route_id: route ID
            edges: list of edge IDs
        """
        route = ET.SubElement(self.routes_root, "route")
        route.set("id", route_id)
        route.set("edges", " ".join(edges))
    
    def generate_base_vehicles(
        self,
        count: int = None,
        routes: List[str] = None,
        start_time: float = 0.0,
        interval: float = 2.0
    ) -> List[GeneratedVehicle]:
        """
        Generate Base-subset vehicles with random types and colors.
        
        Args:
            count: number of vehicles
            routes: list of available route IDs
            start_time: departure time of the first vehicle
            interval: time gap between vehicle departures
        """
        if count is None:
            count = self.cfg.vehicles_per_batch
        
        if routes is None:
            routes = ["route_default"]
        
        jitter_enabled = self._is_jitter_enabled()
        print(f"Generate base vehicles: {count} (color jitter: {'enabled' if jitter_enabled else 'disabled'})...")
        
        generated = []
        
        for i in range(count):
            blueprint, category = self._get_random_blueprint()
            
            actual_rgb, base_rgb, color_name, color_family = self._get_random_color(
                apply_jitter=True,
                jitter_intensity=self._get_jitter_intensity(for_hard=False)
            )
            
            vtype_id = self._register_vtype(blueprint, category, vehicle_type="base")  # use the registered vtype id
            
            veh_id = f"base_{i:04d}"
            depart_time = start_time + i * interval
            route_id = random.choice(routes)
            
            veh = ET.SubElement(self.routes_root, "vehicle")
            veh.set("id", veh_id)
            veh.set("type", vtype_id)  # use the returned vtype id
            veh.set("color", self._rgb_to_sumo_color(actual_rgb))
            veh.set("depart", f"{depart_time:.1f}")
            veh.set("route", route_id)
            
            record = GeneratedVehicle(
                vehicle_id=veh_id,
                blueprint=blueprint,
                color_rgb=actual_rgb,
                color_base_rgb=base_rgb,
                color_name=color_name,
                color_family=color_family,
                category=category,
                depart_time=depart_time,
                route_id=route_id
            )
            generated.append(record)
            self.generated_vehicles.append(record)
            
            self.stats['base_count'] += 1
            self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
            self.stats['by_color'][color_name] = self.stats['by_color'].get(color_name, 0) + 1
            if color_family:
                self.stats['by_color_family'][color_family] = \
                    self.stats['by_color_family'].get(color_family, 0) + 1
        
        return generated
    
    def generate_hard_fleets(
        self,
        num_fleets: int = None,
        fleet_size: int = None,
        routes: List[str] = None,
        start_time: float = 60.0,
        fleet_interval: float = 60.0,
        route_seed: int = 42
    ) -> List[GeneratedVehicle]:
        """
        Generate the Hard subset as fleets of visually identical vehicles.
        v3.2: use time spacing instead of position spacing to avoid short-edge issues.
        v3.4: use balanced route assignment with a fixed shuffle seed instead of random.choice.
        """
        if num_fleets is None or fleet_size is None:
            try:
                hard_cfg = self.cfg.cfg['dataset']['subsets']['hard']
                num_fleets = num_fleets or hard_cfg.get('fleets', 20)
                fleet_size = fleet_size or hard_cfg.get('vehicles_per_fleet', 5)
            except:
                num_fleets = num_fleets or 20
                fleet_size = fleet_size or 5

        if routes is None:
            routes = ["route_default"]

        n_routes = len(routes)
        groups_per_route = num_fleets // n_routes
        remainder = num_fleets % n_routes
        if remainder != 0:
            raise ValueError(
                f"num_fleets ({num_fleets}) must be divisible by the number of routes ({n_routes});"
                f"adjust num_fleets or the route count to keep the assignment balanced"
            )
        route_assignments = routes * groups_per_route  # each route receives the same number of groups
        rng = random.Random(route_seed)
        rng.shuffle(route_assignments)

        print(f"Generate hard fleets: {num_fleets} groups, {fleet_size} vehicles per group...")
        print(f"  Route allocation: {n_routes} routes, {groups_per_route} groups per route (seed={route_seed})")
        print("  Strategy: identical colors within each fleet; departures are spaced by 1.5 seconds")
        for r in routes:
            cnt = route_assignments.count(r)
            print(f"    {r}: {cnt} groups")

        generated = []

        for f in range(num_fleets):
            fleet_blueprint, fleet_category = self._get_random_blueprint()

            fleet_actual_rgb, fleet_base_rgb, fleet_color_name, fleet_color_family = \
                self._get_random_color(
                    apply_jitter=True,
                    jitter_intensity=self._get_jitter_intensity(for_hard=True)
                )

            vtype_id = self._register_vtype(fleet_blueprint, fleet_category, vehicle_type="hard")

            fleet_id = f"fleet_{f:02d}"
            base_depart = start_time + f * fleet_interval
            route_id = route_assignments[f]  # balanced assignment
            
            fleet_color_str = self._rgb_to_sumo_color(fleet_actual_rgb)

            for v in range(fleet_size):
                veh_id = f"H{f:02d}_{v:02d}"
                
                veh = ET.SubElement(self.routes_root, "vehicle")
                veh.set("id", veh_id)
                veh.set("type", vtype_id)
                veh.set("color", fleet_color_str)
                
                # ----------------------------------------------------------------------------
                vehicle_depart = base_depart + v * 1.5  # 1.5 seconds between vehicles
                veh.set("depart", f"{vehicle_depart:.1f}")
                veh.set("route", route_id)
                
                veh.set("departPos", "base")
                veh.set("departSpeed", "max")
                veh.set("departLane", "best")
                
                record = GeneratedVehicle(
                    vehicle_id=veh_id,
                    blueprint=fleet_blueprint,
                    color_rgb=fleet_actual_rgb,
                    color_base_rgb=fleet_base_rgb,
                    color_name=fleet_color_name,
                    color_family=fleet_color_family,
                    category=fleet_category,
                    depart_time=vehicle_depart,
                    route_id=route_id,
                    is_fleet=True,
                    fleet_id=fleet_id
                )
                generated.append(record)
                self.generated_vehicles.append(record)
                
                self.stats['hard_count'] += 1
                self.stats['by_category'][fleet_category] = \
                    self.stats['by_category'].get(fleet_category, 0) + 1
                self.stats['by_color'][fleet_color_name] = \
                    self.stats['by_color'].get(fleet_color_name, 0) + 1
                if fleet_color_family:
                    self.stats['by_color_family'][fleet_color_family] = \
                        self.stats['by_color_family'].get(fleet_color_family, 0) + 1
        
        return generated


    def generate_occ_vehicles(
        self,
        count: int = 200,
        routes: List[str] = None,
        start_time: float = 0.0,
        interval: float = 0.8  # denser departure interval
    ) -> List[GeneratedVehicle]:
        """
        Generate Occ-subset vehicles for dense congestion scenes.
        
        Characteristics:
        - denser departure spacing
        - aggressive car-following parameters to create congestion
        - traffic concentrated on selected routes (for example east-west corridors)
        """
        if routes is None:
            routes = ["route_ew"]  # default east-west corridor with many intersections
        
        print(f"Generate occ vehicles: {count} (high-density congestion mode)...")
        print(f"  Departure interval: {interval}s")
        print("  Driving mode: aggressive car-following (sigma=0.8, minGap=1.5m)")
        
        generated = []
        
        for i in range(count):
            blueprint, category = self._get_random_blueprint()
            
            actual_rgb, base_rgb, color_name, color_family = self._get_random_color(
                apply_jitter=True,
                jitter_intensity=self._get_jitter_intensity(for_hard=False)
            )
            
            vtype_id = self._register_vtype(blueprint, category, vehicle_type="occ")
            
            veh_id = f"occ_{i:04d}"
            depart_time = start_time + i * interval
            route_id = random.choice(routes)
            
            veh = ET.SubElement(self.routes_root, "vehicle")
            veh.set("id", veh_id)
            veh.set("type", vtype_id)
            veh.set("color", self._rgb_to_sumo_color(actual_rgb))
            veh.set("depart", f"{depart_time:.1f}")
            veh.set("route", route_id)
            
            record = GeneratedVehicle(
                vehicle_id=veh_id,
                blueprint=blueprint,
                color_rgb=actual_rgb,
                color_base_rgb=base_rgb,
                color_name=color_name,
                color_family=color_family,
                category=category,
                depart_time=depart_time,
                route_id=route_id
            )
            generated.append(record)
            self.generated_vehicles.append(record)
            
            self.stats['occ_count'] = self.stats.get('occ_count', 0) + 1
            self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
            self.stats['by_color'][color_name] = self.stats['by_color'].get(color_name, 0) + 1
            if color_family:
                self.stats['by_color_family'][color_family] = \
                    self.stats['by_color_family'].get(color_family, 0) + 1
        
        return generated
    def save(self, filename: str = "simveri.rou.xml") -> str:
        """
        Save the route file after sorting vehicles by depart time.

        Returns:
            saved file path
        """
        # ----------------------------------------------------------------------------

        routes = [elem for elem in self.routes_root if elem.tag == 'route']
        vtypes = [elem for elem in self.routes_root if elem.tag == 'vType']
        vehicles = [elem for elem in self.routes_root if elem.tag == 'vehicle']

        vehicles_sorted = sorted(vehicles, key=lambda v: float(v.get('depart', '0')))

        self.routes_root.clear()

        for r in routes:
            self.routes_root.append(r)

        added_vtypes = set()
        for veh in vehicles_sorted:
            vtype_id = veh.get('type')
            if vtype_id not in added_vtypes:
                for vt in vtypes:
                    if vt.get('id') == vtype_id:
                        self.routes_root.append(vt)
                        added_vtypes.add(vtype_id)
                        break
            self.routes_root.append(veh)

        for vt in vtypes:
            if vt.get('id') not in added_vtypes:
                self.routes_root.append(vt)

        print(f" Vehicles sorted by depart time ({len(vehicles_sorted)} total)")

        xml_str = minidom.parseString(
            ET.tostring(self.routes_root, encoding='unicode')
        ).toprettyxml(indent="  ")

        lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(lines)

        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        print(f" Route file saved: {path}")
        return path
    
    def save_vehicle_manifest(self, filename: str = "vehicle_info.csv") -> str:
        """
        Save the vehicle manifest (records actual RGB and base RGB).
        """
        import csv
        
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'vehicle_id', 'blueprint', 'category', 
                'color_name', 'color_family',
                'color_r', 'color_g', 'color_b',
                'base_r', 'base_g', 'base_b',
                'depart_time', 'route_id', 'is_fleet', 'fleet_id'
            ])
            
            for v in self.generated_vehicles:
                writer.writerow([
                    v.vehicle_id,
                    v.blueprint,
                    v.category,
                    v.color_name,
                    v.color_family,
                    v.color_rgb[0], v.color_rgb[1], v.color_rgb[2],
                    v.color_base_rgb[0], v.color_base_rgb[1], v.color_base_rgb[2],
                    v.depart_time,
                    v.route_id,
                    v.is_fleet,
                    v.fleet_id or ''
                ])
        
        print(f" Vehicle manifest saved: {path}")
        return path
    
    def print_statistics(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("Traffic generation summary (v3.5 scaled release)")
        print("=" * 60)
        occ_count = self.stats.get('occ_count', 0)
        print(f"Base vehicles: {self.stats['base_count']}")
        print(f"Hard vehicles: {self.stats['hard_count']}")
        print(f"Occ vehicles: {occ_count}")
        print(f"Total: {self.stats['base_count'] + self.stats['hard_count'] + occ_count}")
        
        print("\nDriving-mode parameters:")
        print("  Base: speedFactor=normc(1.0,0.1), sigma=0.5, minGap=2.5m")
        print("  Hard: speedFactor=normc(1.0,0.1), sigma=0.5, minGap=2.5m (same as Base)")
        print("  Occ:  speedFactor=normc(0.9,0.1), sigma=0.8, minGap=1.5m (creates congestion)")
        
        jitter_enabled = self._is_jitter_enabled()
        print(f"\nColor jitter: {'enabled' if jitter_enabled else 'disabled'}")
        if jitter_enabled:
            print(f"  Base jitter range: +/-{self._get_jitter_intensity(for_hard=False)}")
            print(f"  Hard jitter range: +/-{self._get_jitter_intensity(for_hard=True)}")
        
        print("\nBy category:")
        for cat, count in sorted(self.stats['by_category'].items()):
            print(f"  {cat}: {count}")
        
        print("\nBy color:")
        for color, count in sorted(self.stats['by_color'].items(), key=lambda x: -x[1]):
            print(f"  {color}: {count}")
        
        if self.stats['by_color_family']:
            print("\nBy color family:")
            for family, count in sorted(self.stats['by_color_family'].items(), key=lambda x: -x[1]):
                print(f"  {family}: {count}")
        
        print("=" * 60)

# =============================================================================
# =============================================================================

def test_color_format():
    """Test color-format conversion."""
    gen = SUMOTrafficGenerator.__new__(SUMOTrafficGenerator)
    
    result = gen._rgb_to_sumo_color((255, 128, 0))
    assert result == "255,128,0"
    print(" test_color_format passed")


def test_generated_vehicle():
    """Test the generated-vehicle data structure."""
    v = GeneratedVehicle(
        vehicle_id="test_001",
        blueprint="vehicle.audi.a2",
        color_rgb=(255, 255, 255),
        color_base_rgb=(255, 255, 255),
        color_name="white",
        color_family="white",
        category="sedan",
        depart_time=10.0,
        route_id="route_1"
    )
    
    assert v.vehicle_id == "test_001"
    assert not v.is_fleet
    assert v.color_family == "white"
    print(" test_generated_vehicle passed")


def test_color_jitter():
    """Test the color-jitter functionality."""
    gen = SUMOTrafficGenerator.__new__(SUMOTrafficGenerator)
    gen.cfg = get_config()
    
    base_rgb = (255, 255, 255)
    
    results = set()
    for _ in range(100):
        jittered = gen._apply_color_jitter(base_rgb, intensity=20)
        results.add(jittered)
        
        assert all(0 <= c <= 255 for c in jittered), "RGB value out of range"
        for i in range(3):
            assert abs(jittered[i] - base_rgb[i]) <= 20, "Jitter amplitude exceeds the limit"
    
    assert len(results) > 10, "Jitter should produce multiple distinct colors"
    
    print("test_color_jitter passed")


def run_all_tests():
    print("=" * 60)
    print("traffic_gen.py v3.0 unit tests")
    print("=" * 60)
    
    test_color_format()
    test_generated_vehicle()
    test_color_jitter()
    
    print("\n" + "=" * 60)
    print(" All unit tests passed")
    print("=" * 60)

def main():
    """
    SimVeRi release route generator (v7.0 Final).
    Based on the seven confirmed core routes from the 2025-12-24 release plan.
    """
    gen = SUMOTrafficGenerator(output_dir="sumo")
    
    print("Configuring 7 core routes (final release configuration)...")

    # ==========================================
    # ==========================================
    gen.add_route("route_hard", ["27.0.00", "37.0.00", "35.0.00"]) 

    # ==========================================
    # ==========================================
    
    gen.add_route("route_highway_rev", ["-36.0.00", "-38.0.00"])

    gen.add_route("route_ew_full", ["-39.0.00", "-0.0.00", "-1.0.00", "-2.0.00", "-3.0.00"])

    gen.add_route("route_west_entry", ["20.0.00", "48.0.00", "-39.0.00", "25.0.00"])

    gen.add_route("route_north", ["-21.0.00", "-41.0.00", "-51.0.00", "-52.0.00"])

    gen.add_route("route_south", ["-6.0.00", "-7.0.00", "4.0.00", "45.0.00", "-9.0.00"])

    gen.add_route("route_east", ["24.0.00", "23.0.00"])

    # ==========================================
    # ==========================================
    
    base_routes = [
        "route_highway_rev", 
        "route_ew_full", 
        "route_west_entry", 
        "route_north", 
        "route_south", 
        "route_east"
    ]
    
    print("Generating base vehicles (600 total)...")
    gen.generate_base_vehicles(
        count=600,
        routes=base_routes,
        start_time=0,
        interval=1.75  # 600 vehicles fill roughly a 1050s window
    )
    
    hard_routes = [
        "route_hard",        # elevated main direction (C001,C002,C005,C007,C003,C004) - 6 cams
        "route_ew_full",     # east-west arterial (C012,C013,C018,C016,C008) - 5 cams
        "route_west_entry",  # west-side entry corridor (C011,C012,C013,C014,C023) - 5 cams
        "route_north",       # northern connector (C024,C017,C010,C015) - 4 cams
        "route_south",       # southern curve corridor (C023,C022,C019,C020) - 4 cams
    ]
    print(f"Generating hard twins fleets (25 groups across {len(hard_routes)} routes)...")
    gen.generate_hard_fleets(
        num_fleets=25,
        fleet_size=5,
        routes=hard_routes,
        start_time=5.0,       # early departures so fleets finish before simulation ends
        fleet_interval=40.0,  # 25 groups fill roughly a 1000s window
        route_seed=42         # fixed seed for reproducible balanced assignment
    )
    
    print("Generating occ congestion traffic (50 vehicles)...")
    gen.generate_occ_vehicles(
        count=50,            # adds extra occlusion-heavy samples
        routes=["route_ew_full"],
        start_time=0.0,
        interval=0.8         # very high dispatch frequency
    )

    # ==========================================
    # ==========================================
    
    rou_path = gen.save("simveri.rou.xml")
    
    csv_path = gen.save_vehicle_manifest("vehicle_info.csv")
    
    gen.print_statistics()
    
    print("\n Route and traffic generation finished")
    print(f"   Route file: {rou_path}")
    print(f"   Vehicle manifest: {csv_path}")
    print("   Next step: run run_simveri.py to start data collection.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_all_tests()
    else:
        main()
