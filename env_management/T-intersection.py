import random
import math
import collections
import weakref
import pickle
from typing import Union, List, Tuple

import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.navigation import GlobalRoutePlanner
from env_management.env_manager import T_intersection_list
from perception import get_fov_polygon, get_nearby_lanes


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self) -> dict:
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    def is_collided(self) -> bool:
        return self.history != []
    
    def destroy(self):
        if self.sensor:
            self.sensor.destroy()

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

def _parse_waypoint(waypoint: Union[int, float, tuple], all_default_spawn: List[carla.Transform]) -> carla.Transform:
    if isinstance(waypoint, (int, float)):
        return all_default_spawn[waypoint]
    elif isinstance(waypoint, tuple):
        return carla.Transform(carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2]), carla.Rotation(yaw=waypoint[3], pitch=waypoint[4], roll=waypoint[5]))
    
def _bbox_to_polygon(bbox: carla.BoundingBox, location: List[float], yaw_deg: float) -> Polygon:
    yaw = np.deg2rad(yaw_deg)
    half_box_size = (bbox.extent.x, bbox.extent.y)
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    corners = np.array([
        [-half_box_size[0], -half_box_size[1]],
        [half_box_size[0], -half_box_size[1]],
        [half_box_size[0], half_box_size[1]],
        [-half_box_size[0], half_box_size[1]]
    ])
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners += np.array(location[:2])
    return Polygon(rotated_corners)

def _plot(ax, ego_polygon: Polygon = None, enemy_polygon_list: List[Polygon] = None, obstacle_polygon_list: List[Polygon] = None, fov_polygon: Polygon = None, map_borders: Tuple[LineString, Polygon] = None):
    if map_borders is not None or map_borders is not ():
        ax.set_facecolor('gray')
        x_outer = [coords[0] for coords in map_borders[0].coords]
        y_outer = [coords[1] for coords in map_borders[0].coords]
        ax.fill(x_outer, y_outer, color='white', alpha=1)

        for polygon in map_borders[1:]:
            x_inner, y_inner = polygon.exterior.xy
            ax.fill(x_inner, y_inner, color='gray', alpha=1)

    if ego_polygon is not None:
        x, y = ego_polygon.exterior.xy
        ax.fill(x, y, alpha=1, color='red', edgecolor='none')

    if enemy_polygon_list is not None:
        for enemy in enemy_polygon_list:
            x, y = enemy.exterior.xy
            ax.fill(x, y, alpha=1, color='blue', edgecolor='none')

    if obstacle_polygon_list is not None:
        for obstacle in obstacle_polygon_list:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, alpha=1, color='cyan', edgecolor='none')

    if fov_polygon is not None:
        x, y = fov_polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, color='green', edgecolor='none')

    ax.set_aspect('equal', 'box')

def sample(carla_port: int, episode_num: int = 1000, max_episode_steps: int = 1000, plot: bool = False):

    # command_text = f"{carla_server_path} -carla-rpc-port={carla_port} {'' if render else '-nullrhi'}"
    # carla_pid = subprocess.Popen(command_text)

    config = random.choice(T_intersection_list)

    client = carla.Client('localhost', carla_port)
    client.set_timeout(60.0)
    map_name = config.get('map_name')
    world = client.load_world(map_name)
    world_map = world.get_map()

    with open(f'map_cache/{map_name}_map_borders.pkl', 'rb') as f:
        map = pickle.load(f)
    map_borders = [LineString(map[0])] + [Polygon(map_line) for map_line in map[1:]]
    map_polyline_list = [unary_union(map_borders)]

    global_planner = GlobalRoutePlanner(world_map, 2.0)
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()

    origin_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    
    # 关闭交通信号灯
    traffic_lights = world.get_actors().filter('*traffic_light*')
    for tl in traffic_lights:
        tl.set_state(carla.TrafficLightState.Off)
        tl.freeze(True)
    
    ########################################################################################################################
    # 加载所有 Agent
    all_default_spawn = world_map.get_spawn_points()
    config['ego_agent_config']['spawn_point'] = _parse_waypoint(config['ego_agent_config']['spawn_point'], all_default_spawn)
    config['ego_agent_config']['destination'] = _parse_waypoint(config['ego_agent_config']['destination'], all_default_spawn)

    for enemy_agent in config['enemy_agents_config']:
        enemy_agent['spawn_point'] = _parse_waypoint(enemy_agent['spawn_point'], all_default_spawn)
        enemy_agent['destination'] = _parse_waypoint(enemy_agent['destination'], all_default_spawn)

    for obstacle_agent in config['obstacle_agents_config']:
        obstacle_agent['spawn_point'] = _parse_waypoint(obstacle_agent['spawn_point'], all_default_spawn)
        if obstacle_agent['destination'] is not None:
            continue
        else:
            obstacle_agent['destination'] = _parse_waypoint(obstacle_agent['destination'], all_default_spawn)

    ego_vehicle_bp = blueprint_library.find(config['ego_agent_config']['vehicle_type'])
    ego_vehicle_bp.set_attribute('color', config['ego_agent_config']['color'])
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, config['ego_agent_config']['spawn_point'])
    ego_collision_sensor = CollisionSensor(ego_vehicle)
    
    enemy_vehicle_list = []
    enemy_agent_list = []
    for enemy_agent in config['enemy_agents_config']:
        enemy_vehicle_bp = blueprint_library.find(enemy_agent['vehicle_type'])
        enemy_vehicle_bp.set_attribute('color', enemy_agent['color'])
        enemy_vehicle = world.spawn_actor(enemy_vehicle_bp, enemy_agent['spawn_point'])
        enemy_vehicle_list.append(enemy_vehicle)

    obstacle_vehicle_list = []
    for obstacle_agent in config['obstacle_agents_config']:
        obstacle_vehicle_bp = blueprint_library.find(obstacle_agent['vehicle_type'])
        obstacle_vehicle_bp.set_attribute('color', obstacle_agent['color'])
        obstacle_vehicle = world.spawn_actor(obstacle_vehicle_bp, obstacle_agent['spawn_point'])
        obstacle_vehicle_list.append(obstacle_vehicle)

    ########################################################################################################################
    # we need to tick the world once to let the client update the spawn position 我们需要先让世界运行一次，以便客户端更新生成位置
    world.tick()

    # generate the route    生成 agent 和 route
    ego_agent = config['ego_agent_config']['agent_type'](ego_vehicle, target_speed=config['ego_agent_config']['target_speed'], map_inst=world_map, grp_inst=global_planner)
    ego_agent.set_destination(config['ego_agent_config']['destination'].location)
    for i, enemy_vehicle in enumerate(enemy_vehicle_list):
        enemy_agent = config['enemy_agents_config'][i]['agent_type'](enemy_vehicle, target_speed=enemy_agent['target_speed'], map_inst=world_map, grp_inst=global_planner)
        enemy_agent_list.append(enemy_agent)
        enemy_agent.set_destination(config['enemy_agents_config'][i]['destination'].location)

    for i, obstacle_vehicle in enumerate(obstacle_vehicle_list):
        if config['obstacle_agents_config'][i]['agent_type'] is None:
            continue
        obstacle_agent = config['obstacle_agents_config'][i]['agent_type'](
            obstacle_vehicle,
            target_speed=obstacle_agent['target_speed'],
            map_inst=world_map,
            grp_inst=global_planner
            )
        obstacle_agent.set_destination(obstacle_vehicle.get_transform())

    if plot:
        fig, ax = plt.subplots()

    try:
        while True:
            # time.sleep(0.05)
            # agent.run_step()

            world.tick()

            ego_transform = ego_vehicle.get_transform()
            ego_location = (ego_transform.location.x, ego_transform.location.y, ego_transform.location.z)
            ego_yaw = ego_transform.rotation.yaw
            ego_polygon = _bbox_to_polygon(ego_vehicle.bounding_box, ego_location, ego_yaw)
            ego_velocity_vec3d = ego_vehicle.get_velocity()
            ego_velocity = (ego_velocity_vec3d.x, ego_velocity_vec3d.y, ego_velocity_vec3d.z)

            nearby_lanes = get_nearby_lanes(world_map.get_waypoint(ego_vehicle.get_location()), 1, 100, 20)

            enemy_polygon_list = []
            for enemy in enemy_vehicle_list:
                enemy_transform = enemy.get_transform()
                enemy_location = (enemy_transform.location.x, enemy_transform.location.y, enemy_transform.location.z)
                enemy_yaw = enemy_transform.rotation.yaw
                enemy_polygon = _bbox_to_polygon(enemy.bounding_box, enemy_location, enemy_yaw)
                enemy_polygon_list.append(enemy_polygon)
                enemy_velocity_vec3d = enemy.get_velocity()
                enemy_velocity = (enemy_velocity_vec3d.x, enemy_velocity_vec3d.y, enemy_velocity_vec3d.z)

            obstacle_polygon_list = []
            for obstacle in obstacle_vehicle_list:
                obstacle_transform = obstacle.get_transform()
                obstacle_location = (obstacle_transform.location.x, obstacle_transform.location.y, obstacle_transform.location.z)
                obstacle_yaw = obstacle_transform.rotation.yaw
                obstacle_polygon = _bbox_to_polygon(obstacle.bounding_box, obstacle_location, obstacle_yaw)
                obstacle_polygon_list.append(obstacle_polygon)
                obstacle_velocity_vec3d = obstacle.get_velocity()
                obstacle_velocity = (obstacle_velocity_vec3d.x, obstacle_velocity_vec3d.y, obstacle_velocity_vec3d.z)

            all_polygon_list = map_polyline_list + enemy_polygon_list + obstacle_polygon_list
            ego_point = Point(ego_location[0], ego_location[1])
            visible_area, visible_area_vertices_type, obstacles_visibility = get_fov_polygon(ego_point, 360, 50, all_polygon_list, ray_num=40)
            
            if plot:
                ax.clear()
                _plot(ax, ego_polygon = ego_polygon, enemy_polygon_list = enemy_polygon_list, obstacle_polygon_list = obstacle_polygon_list, fov_polygon = visible_area, map_borders = map_borders)
                plt.draw()
                plt.pause(0.1)

            if ego_agent.done():
                print('======== Success, Arrivied at Target Point!')
                break
            elif ego_collision_sensor.is_collided():
                print('======== Collision!')
                break
                
            # top view
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=60),
                                                    carla.Rotation(pitch=-90)))

            ego_vehicle.apply_control(ego_agent.run_step())
            for i, enemy_agent in enumerate(enemy_agent_list):
                enemy_vehicle_list[i].apply_control(enemy_agent.run_step())

    finally:
        world.apply_settings(origin_settings)
        ego_vehicle.destroy()
        ego_collision_sensor.destroy()
        for enemy_vehicle in enemy_vehicle_list:
            enemy_vehicle.destroy()
        for obstacle_vehicle in obstacle_vehicle_list:
            obstacle_vehicle.destroy()
        # carla_pid.kill()

sample(2000, plot=True)