#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
SimVeRi CARLA-SUMO synchronization entry point with optional data collection.
This script launches the co-simulation loop and can enable the
SimVeRi collector for synchronized image and metadata export.
It also applies map and weather settings before the main loop starts.
The remaining logic follows the upstream CARLA synchronization flow.
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time
import traci
import yaml
# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

# ----------------------------------------------------------------------------
from simveri_collector import SimVeRiCollector, load_vehicle_manifest


try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    
    This version optionally integrates the SimVeRi collector during runtime.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False,
                 # ----------------------------------------------------------------------------
                 enable_collection=True,
                 vehicle_manifest_path='sumo/vehicle_info.csv',
                 output_dir='output_AG_MVP_20260131_01',
                 tm_port=8000):

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        try:
            traffic_manager = self.carla.client.get_trafficmanager(tm_port)
            traffic_manager.set_synchronous_mode(True)
        except RuntimeError as e:
            logging.warning(f"Traffic Manager init warning: {e}. (This is fine if SUMO controls all vehicles)")

        # ----------------------------------------------------------------------------
        self.simveri_collector = None
        self.enable_collection = enable_collection
        
        if enable_collection:
            logging.info('Initializing SimVeRi data collector...')
            vehicle_info = load_vehicle_manifest(vehicle_manifest_path)
            
            if vehicle_info:
                self.simveri_collector = SimVeRiCollector(
                    world=self.carla.world,
                    vehicle_info=vehicle_info,
                    output_dir=output_dir
                )
                
                if self.simveri_collector.initialize():
                    logging.info('SimVeRi collector initialized successfully.')
                else:
                    logging.warning('SimVeRi collector initialization failed.')
                    self.simveri_collector = None
            else:
                logging.warning('Vehicle manifest not found, collection disabled.')

    def tick(self):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

        # ----------------------------------------------------------------------------
        if self.simveri_collector is not None:
            try:
                sumo_speeds = {}
                for sumo_id in self.sumo2carla_ids.keys():
                    try:
                        speed_ms = traci.vehicle.getSpeed(sumo_id)
                        sumo_speeds[sumo_id] = speed_ms * 3.6  # convert to km/h
                    except:
                        sumo_speeds[sumo_id] = 0.0
                
                self.simveri_collector.collect_step(self.sumo2carla_ids, sumo_speeds)
            except Exception as e:
                logging.error(f'SimVeRi collection error: {e}')

    def close(self):
        """
        Cleans synchronization.
        """
        # ----------------------------------------------------------------------------
        if self.simveri_collector is not None:
            logging.info('Finalizing SimVeRi collector...')
            try:
                self.simveri_collector.finalize()
            except Exception as e:
                logging.error(f'SimVeRi finalization error: {e}')

        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()
    
    # ----------------------------------------------------------------------------
    def get_collection_stats(self):
        """Return a small snapshot of the live collection state."""
        if self.simveri_collector:
            return {
                'frame_count': self.simveri_collector.frame_count,
                'captures': len(self.simveri_collector.captures),
                'active_vehicles': len(self.sumo2carla_ids)
            }
        return None

def apply_weather_from_config(client, config_path='config.yaml'):
    """Apply weather settings from config.yaml (weather_params preferred)."""
    try:
        import carla
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            logging.warning('Weather config ignored: config.yaml did not parse to a dict')
            return
        sim_cfg = cfg.get('simulation', {})
        if not isinstance(sim_cfg, dict):
            logging.warning('Weather config ignored: simulation block is not a dict')
            return
        weather_params = sim_cfg.get('weather_params')
        world = client.get_world()
        if isinstance(weather_params, dict) and weather_params:
            world.set_weather(carla.WeatherParameters(**weather_params))
            logging.info('Applied CARLA weather from simulation.weather_params')
            return
        weather_name = sim_cfg.get('weather')
        if weather_name:
            preset = getattr(carla.WeatherParameters, weather_name, None)
            if preset is None:
                logging.warning('Unknown weather preset: %s', weather_name)
            else:
                world.set_weather(preset)
                logging.info('Applied CARLA weather preset: %s', weather_name)
    except Exception as exc:
        logging.warning('Failed to apply weather settings: %s', exc)

def synchronization_loop(args):
    """
    Entry point for sumo-carla co-simulation.
    """
    # ----------------------------------------------------------------------------
    import carla
    temp_client = carla.Client(args.carla_host, args.carla_port)
    temp_client.set_timeout(300.0)
    current_map = temp_client.get_world().get_map().name
    
    if 'Town05' not in current_map:
        logging.info(f'Current map: {current_map}, switching to Town05...')
        temp_client.load_world('Town05')
        time.sleep(5)  # allow the new map to finish loading
        logging.info('Map switched to Town05')
    # ----------------------------------------------------------------------------
    apply_weather_from_config(temp_client)

    sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)

    # ----------------------------------------------------------------------------
    synchronization = SimulationSynchronization(
        sumo_simulation, 
        carla_simulation, 
        args.tls_manager,
        args.sync_vehicle_color, 
        args.sync_vehicle_lights,
        enable_collection=args.enable_collection,
        vehicle_manifest_path=args.vehicle_manifest,
        output_dir=args.output_dir,
        tm_port=args.tm_port
    )
    
    # ----------------------------------------------------------------------------
    start_time = time.time()  # used for wall-clock progress logging
    
    step_length = args.step_length  # e.g. 0.05s
    if args.duration > 0:
        target_frames = int(args.duration / step_length)  # e.g. 600 / 0.05 = 12000
    else:
        target_frames = float('inf')  # run until interrupted when duration <= 0
    
    tick_count = 0
    
    logging.info(f'Starting simulation...')
    logging.info(f'  Duration: {args.duration}s (simulation time)')
    logging.info(f'  Step length: {step_length}s')
    logging.info(f'  Target frames: {target_frames}')
    if args.enable_collection:
        logging.info(f'  Data collection: enabled, output: {args.output_dir}')
    
    try:
        while True:
            tick_start = time.time()

            synchronization.tick()
            tick_count += 1
            
            sim_time = tick_count * step_length

            if tick_count % 500 == 0:
                real_elapsed = time.time() - start_time
                speed_ratio = sim_time / real_elapsed if real_elapsed > 0 else 0
                
                stats = synchronization.get_collection_stats()
                if stats:
                    logging.info(f'Progress: {sim_time:.1f}s / {args.duration}s (sim) | '
                                f'Real: {real_elapsed:.1f}s | '
                                f'Speed: {speed_ratio:.2f}x | '
                                f'Frames: {tick_count} | '
                                f'Captures: {stats["captures"]} | '
                                f'Vehicles: {stats["active_vehicles"]}')
                else:
                    logging.info(f'Progress: {sim_time:.1f}s / {args.duration}s (sim) | '
                                f'Real: {real_elapsed:.1f}s | '
                                f'Speed: {speed_ratio:.2f}x | '
                                f'Frames: {tick_count}')

            # ----------------------------------------------------------------------------
            if tick_count >= target_frames:
                logging.info(f'Simulation time limit reached ({args.duration}s), stopping...')
                break

            tick_end = time.time()
            tick_elapsed = tick_end - tick_start
            if tick_elapsed < step_length:
                time.sleep(step_length - tick_elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization...')
        synchronization.close()
        
        total_real_time = time.time() - start_time
        total_sim_time = tick_count * step_length
        logging.info(f'=== Final Statistics ===')
        logging.info(f'Simulation time: {total_sim_time:.1f}s')
        logging.info(f'Real time: {total_real_time:.1f}s')
        logging.info(f'Speed ratio: {total_sim_time / total_real_time:.2f}x')
        logging.info(f'Total frames: {tick_count}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    
    # ----------------------------------------------------------------------------
    argparser.add_argument('sumo_cfg_file', type=str, help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.0333,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')

    # ----------------------------------------------------------------------------
    argparser.add_argument('--enable-collection',
                           action='store_true',
                           default=True,
                           help='enable SimVeRi data collection (default: True)')
    argparser.add_argument('--no-collection',
                           action='store_true',
                           help='disable SimVeRi data collection')
    argparser.add_argument('--vehicle-manifest',
                           type=str,
                           default='sumo/vehicle_info.csv',
                           help='path to vehicle manifest CSV (default: sumo/vehicle_info.csv)')
    argparser.add_argument('--output-dir',
                           type=str,
                           default='output',
                           help='output directory for collected data (default: output)')
    argparser.add_argument('--duration',
                           type=float,
                           default=600,
                           help='simulation duration in seconds, 0 for unlimited (default: 600)')
    argparser.add_argument('--tm-port',
                           type=int,
                           default=8000,
                           help='Carla Traffic Manager port (default: 8000)')
    
    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True
    
    if arguments.no_collection:
        arguments.enable_collection = False

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info('=' * 60)
    logging.info('SimVeRi Co-Simulation Data Collection')
    logging.info('=' * 60)
    logging.info(f'SUMO config: {arguments.sumo_cfg_file}')
    logging.info(f'Carla server: {arguments.carla_host}:{arguments.carla_port}')
    logging.info(f'Data collection: {"Enabled" if arguments.enable_collection else "Disabled"}')
    logging.info(f'Output directory: {arguments.output_dir}')
    logging.info(f'Duration: {arguments.duration}s' if arguments.duration > 0 else 'Duration: Unlimited')
    logging.info('=' * 60)

    synchronization_loop(arguments)
