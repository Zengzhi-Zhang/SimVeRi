# SimVeRi SUMO Assets

**Created**: 2025-12-18  
**Purpose**: Store the SUMO network, route files, and configuration files used by the SimVeRi co-simulation pipeline.

## Directory layout

```text
sumo/
|-- network/             # SUMO network assets
|   |-- Town05.net.xml   # SUMO road network converted from CARLA Town05
|   `-- Town05.edg.xml   # Edge definitions
|-- routes/              # Vehicle route files
|   |-- base_batch1.rou.xml
|   |-- base_batch2.rou.xml
|   |-- base_batch3.rou.xml
|   |-- base_batch4.rou.xml
|   |-- hard_batch5.rou.xml
|   |-- occ_batch6.rou.xml
|   `-- weather_batch7.rou.xml
|-- config/              # SUMO configuration files
|   |-- sumo_base.sumocfg
|   |-- sumo_hard.sumocfg
|   `-- sumo_occ.sumocfg
`-- detectors/           # Optional detector definitions
    `-- detectors.add.xml
```

## File descriptions

### 1. Network files

#### `network/Town05.net.xml`

- Purpose: SUMO network definition.
- Source: Exported from the CARLA `Town05` map.
- Typical generation methods:

```bash
# If Town05.xodr is available
netconvert --opendrive Town05.xodr -o Town05.net.xml

# Or by using the CARLA helper tools
python carla_sumo_export.py --map Town05 --output Town05.net.xml
```

- Important characteristics:
  - Includes the relevant intersections used in the benchmark.
  - Preserves traffic-light control where available.
  - Retains elevation information for overpasses and elevated roads.

### 2. Route files

#### Naming convention

- Base subsets: `base_batch{N}.rou.xml`
- Twins / hard subsets: `hard_batch{N}.rou.xml`
- Occlusion subsets: `occ_batch{N}.rou.xml`

#### Example: base route file

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle type definitions -->
    <vType id="sedan" vClass="passenger" color="255,0,0"/>
    <vType id="suv" vClass="passenger" color="128,128,128"/>

    <!-- Base vehicles -->
    <vehicle id="0001" type="sedan" depart="0.0" color="255,0,0">
        <route edges="edge1 edge2 edge3 edge4"/>
    </vehicle>

    <vehicle id="0002" type="suv" depart="5.0" color="128,128,128">
        <route edges="edge5 edge6 edge7 edge8"/>
    </vehicle>
</routes>
```

#### Example: twins / hard route file

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle type definition -->
    <vType id="doppelganger_1" vClass="passenger" color="255,0,0"/>

    <!-- Fleet 01: five visually identical vehicles -->
    <vehicle id="H001" type="doppelganger_1" depart="0.0" departPos="0" color="255,0,0">
        <route edges="edge10 edge11 edge12"/>
    </vehicle>

    <vehicle id="H002" type="doppelganger_1" depart="2.5" departPos="0" color="255,0,0">
        <route edges="edge10 edge11 edge12"/>
    </vehicle>
</routes>
```

- Key parameters:
  - `depart`: vehicle departure time in seconds.
  - `departPos`: starting position in meters; hard subsets often use `0` for tightly aligned departures.
  - `color`: RGB color value; all members of a twins fleet should share the same color.
  - `route edges`: edge sequence; all members of a twins fleet should share the same route.

#### Example: occlusion route file

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Dense traffic configuration -->
    <vType id="occ_vehicle" vClass="passenger" speedFactor="0.5"/>

    <!-- Vehicles concentrated on the same corridor -->
    <vehicle id="O001" type="occ_vehicle" depart="0.0">
        <route edges="edge_congestion_1 edge_congestion_2"/>
    </vehicle>

    <vehicle id="O002" type="occ_vehicle" depart="1.0">
        <route edges="edge_congestion_1 edge_congestion_2"/>
    </vehicle>
</routes>
```

- Typical occlusion settings:
  - Lower `speedFactor` to reduce speed and increase crowding.
  - Use short `depart` intervals to create denser traffic.
  - Concentrate many vehicles on a small set of routes.

### 3. SUMO configuration files

#### `config/sumo_base.sumocfg`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="../network/Town05.net.xml"/>
        <route-files value="../routes/base_batch1.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="1800"/>  <!-- 30 minutes -->
        <step-length value="0.05"/>  <!-- 20 FPS -->
    </time>

    <processing>
        <time-to-teleport value="-1"/>  <!-- Disable teleporting -->
        <collision.action value="warn"/>  <!-- Warn on collisions -->
    </processing>
</configuration>
```

The hard and occlusion configurations follow the same structure but point to different route files.

## Route generation

Typical usage:

```bash
# Generate base routes
python traffic_gen_v2.py --mode base --batch 1 --num-vehicles 100 --output sumo/routes/base_batch1.rou.xml

# Generate twins / hard routes
python traffic_gen_v2.py --mode hard --batch 5 --num-fleets 20 --output sumo/routes/hard_batch5.rou.xml

# Generate occlusion routes
python traffic_gen_v2.py --mode occ --batch 6 --num-vehicles 100 --output sumo/routes/occ_batch6.rou.xml
```

## CARLA-SUMO co-simulation notes

Representative launch settings:

```python
sumo_config = {
    'host': 'localhost',
    'port': 8813,
    'sumocfg_file': 'sumo/config/sumo_base.sumocfg',
    'tls_manager': 'sumo',          # traffic lights are controlled by SUMO
    'step_length': 0.05,            # synchronized with CARLA
    'sync_vehicle_color': True,     # keep vehicle colors aligned
    'sync_vehicle_lights': False,   # skip light synchronization for efficiency
}
```

## Frequently asked questions

### How do I obtain `Town05.net.xml`?

Two common options are available:

1. Export `Town05.xodr` from CARLA and convert it with `netconvert`.
2. Use the `sumo_integration` utilities provided by CARLA.

### How do I ensure that twins vehicles are visually identical?

In the `.rou.xml` file, ensure that all vehicles in the same twins fleet share:

- the same `vType`,
- the same RGB `color`,
- the same `route edges`,
- and different `depart` times to avoid exact overlap.

### How do I create congested traffic for the occlusion subset?

Use a combination of:

- `speedFactor < 1.0`,
- short `depart` intervals,
- and route concentration on a small set of road segments.

### How do SUMO and CARLA coordinates correspond?

- SUMO uses 2D coordinates `(X, Y)`.
- CARLA uses 3D coordinates `(X, Y, Z)`.
- During co-simulation, CARLA maps SUMO coordinates into 3D space using map elevation.

## Pre-release checklist

Before running data collection, confirm that:

- `network/Town05.net.xml` exists and loads correctly.
- All required `.rou.xml` files have been generated.
- Vehicle colors in `.rou.xml` match the expected fleet definitions.
- `.sumocfg` paths are correct.
- `step-length` is set to `0.05` when using the default 20 FPS synchronization.
- Twins fleet departure intervals remain separated in time.

## References

- SUMO documentation: <https://sumo.dlr.de/docs/>
- CARLA-SUMO co-simulation: <https://carla.readthedocs.io/en/latest/adv_sumo/>
- Route file format: <https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html>

**Created**: 2025-12-18  
**Last updated**: 2025-12-18
