import math
import random
import pickle
import weakref
import collections
from typing import Union, List, Tuple


import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from perception.get_fov_polygon import get_fov_polygon
from perception.get_map_lanes import get_map_lanes

T_intersection_list = [
    {
        "map_name": "Town01",
        "ego_agent_config": {
            "agent_type": BasicAgent,
            "spawn_point": 192,
            "destination": 118,
            "vehicle_type": "vehicle.audi.a2",
            "color": "255, 0, 0",
            "target_speed": 100,
            "target_speed_std_dev": 10
        },
        "enemy_agents_config":[
            {   
                "agent_type": BasicAgent,
                "spawn_point": 109,
                "destination": 73,
                "vehicle_type": "vehicle.audi.a2",
                "color": "0, 0, 255",
                "target_speed": 30,
                "target_speed_std_dev": 10
            }
        ],
        "obstacle_agents_config":[
            {   
                "agent_type": None,
                "spawn_point": 107,
                "destination": None,
                "vehicle_type": "vehicle.carlamotors.carlacola",
                "color": "0, 255, 255",
                "target_speed": None,
                "target_speed_std_dev": None 
            }
        ]
    }
]

class EnvManager():
    def __init__(self, env_name: str, carla_host: str, carla_port:int):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(60.0)
        self.env_name = env_name
        self.map_name = None
        self.world = None
        self.map = None
        self.map_borders = None
        self.map_borders_union = None

        self.ego_agent_config = None
        self.enemy_agents_config = []
        self.obstacle_agents_config = []

        self.ego_vehicle = None
        self.ego_agent = None
        self.ego_collision_sensor = None
        self.enemy_vehicles = []
        self.enemy_agents = []
        self.obstacle_vehicles = []
        self.obstacle_agents = []

        self.ego_info = None
        self.enemies_info = []
        self.obstacles_info = []
        self.map_lanes = []
        self.visible_area = []
        self.visible_area_vertices_type = []
        self.obstacles_visibility = []

        _, self.ax = plt.subplots()

        self.reset_config()
        

    def reset_config(self):
        if self.env_name == 'T-intersection':
            env_dict = random.choice(T_intersection_list)
            map_changed = True
            for key, value in env_dict.items():
                if key == "map_name" and value == self.map_name:
                    map_changed = False
                    continue
                setattr(self, key, value)
            
            if map_changed:
                self.world = self.client.load_world(self.map_name)
                self.map = self.world.get_map()
                self._load_map_borders(self.map_name)
                self.spawn_points = self.map.get_spawn_points()

                self.global_planner = GlobalRoutePlanner(self.map, 2.0)
                self.spectator = self.world.get_spectator()
                self.blueprint_library = self.world.get_blueprint_library()

                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.1
                self.world.apply_settings(settings)
                
                # 关闭交通信号灯
                traffic_lights = self.world.get_actors().filter('*traffic_light*')
                for tl in traffic_lights:
                    tl.set_state(carla.TrafficLightState.Off)
                    tl.freeze(True)
            
            self.ego_agent_config['target_speed'] = np.random.normal(self.ego_agent_config['target_speed'], self.ego_agent_config['target_speed_std_dev'])

            self._random_remove(self.enemy_agents_config, 0.1)
            self._random_remove(self.obstacle_agents_config, 0)

            for enemy_agent_config in self.enemy_agents_config:
                enemy_agent_config['target_speed'] = np.random.normal(enemy_agent_config['target_speed'], enemy_agent_config['target_speed_std_dev'])
            for obstacle_agent_config in self.obstacle_agents_config:
                if obstacle_agent_config['target_speed'] is None:
                    continue
                obstacle_agent_config['target_speed']

    def spawn_vehicles_from_config(self):
        if self.world is None:
            print("Please reset the environment first.")
            return
        self.ego_vehicle = self.world.spawn_actor(
            self.blueprint_library.find(self.ego_agent_config['vehicle_type']),
            self._parse_waypoint(self.ego_agent_config['spawn_point'])
            )
        self.ego_collision_sensor = CollisionSensor(self.ego_vehicle)

        for enemy_agent_config in self.enemy_agents_config:
            enemy_vehicle = self.world.spawn_actor(
                self.blueprint_library.find(enemy_agent_config['vehicle_type']),
                self.spawn_points[enemy_agent_config['spawn_point']]
                )
            self.enemy_vehicles.append(enemy_vehicle)

        for obstacle_agent_config in self.obstacle_agents_config:
            if obstacle_agent_config['spawn_point'] is None:
                continue
            obstacle_vehicle = self.world.spawn_actor(
                self.blueprint_library.find(obstacle_agent_config['vehicle_type']),
                self.spawn_points[obstacle_agent_config['spawn_point']]
                )
            self.obstacle_vehicles.append(obstacle_vehicle)
        
        self.tick()

    def assign_agents_to_vehicles(self):
        if self.ego_vehicle is not None:
            self.ego_agent = self.ego_agent_config['agent_type'](
                vehicle=self.ego_vehicle,
                target_speed=self.ego_agent_config['target_speed'],
                map_inst=self.map,
                grp_inst=self.global_planner
                )
            self.ego_agent.set_destination(self._parse_waypoint(self.ego_agent_config['destination']).location)
        for i, enemy_vehicle in enumerate(self.enemy_vehicles):
            enemy_agent = self.enemy_agents_config[i]['agent_type'](
                vehicle=enemy_vehicle,
                target_speed=self.enemy_agents_config[i]['target_speed'],
                map_inst=self.map,
                grp_inst=self.global_planner
                )
            enemy_agent.set_destination(self._parse_waypoint(self.enemy_agents_config[i]['destination']).location)
            self.enemy_agents.append(enemy_agent)
        for i, obstacle_vehicle in enumerate(self.obstacle_vehicles):
            if self.obstacle_agents_config[i]['agent_type'] is None:
                self.obstacle_agents.append(None)
                continue
            elif self.obstacle_agents_config[i]['agent_type'] == BasicAgent:
                obstacle_agent = self.obstacle_agents_config[i]['agent_type'](
                    vehicle=obstacle_vehicle,
                    target_speed=self.obstacle_agents_config[i]['target_speed'],
                    map_inst=self.map,
                    grp_inst=self.global_planner
                    )
                obstacle_agent.set_destination(self._parse_waypoint(self.obstacle_agents_config[i]['destination']).location)
                self.obstacle_agents.append(obstacle_agent)
            elif self.obstacle_agents_config[i]['agent_type'] == BehaviorAgent:
                obstacle_agent = self.obstacle_agents_config[i]['agent_type'](
                    vehicle=obstacle_vehicle,
                    behavior=self.obstacle_agents_config[i]['behavior'],
                    map_inst=self.map,
                    grp_inst=self.global_planner)
                obstacle_agent.set_destination(self._parse_waypoint(self.obstacle_agents_config[i]['destination']).location)
                self.obstacle_agents.append(obstacle_agent)

    def update_info(self, plot=False):
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = (ego_transform.location.x, ego_transform.location.y, ego_transform.location.z)
        ego_yaw = ego_transform.rotation.yaw
        ego_polygon = self._bbox_to_polygon(self.ego_vehicle.bounding_box, ego_location, ego_yaw)
        ego_velocity_vec3d = self.ego_vehicle.get_velocity()
        ego_velocity = (ego_velocity_vec3d.x, ego_velocity_vec3d.y, ego_velocity_vec3d.z)
        self.ego_info = {
            "location": ego_location,
            "yaw": ego_yaw,
            "velocity": ego_velocity,
            "polygon": ego_polygon
        }

        self.map_lanes = get_map_lanes(self.map.get_waypoint(self.ego_vehicle.get_location()), 1, 100, 20)

        self.enemies_info = []
        for enemy_vehicle in self.enemy_vehicles:
            enemy_transform = enemy_vehicle.get_transform()
            enemy_location = (enemy_transform.location.x, enemy_transform.location.y, enemy_transform.location.z)
            enemy_yaw = enemy_transform.rotation.yaw
            enemy_polygon = self._bbox_to_polygon(enemy_vehicle.bounding_box, enemy_location, enemy_yaw)
            enemy_velocity_vec3d = enemy_vehicle.get_velocity()
            enemy_velocity = (enemy_velocity_vec3d.x, enemy_velocity_vec3d.y, enemy_velocity_vec3d.z)
            self.enemies_info.append({
                "location": enemy_location,
                "yaw": enemy_yaw,
                "velocity": enemy_velocity,
                "polygon": enemy_polygon
            })
        
        self.obstacles_info = []
        for obstacle_vehicle in self.obstacle_vehicles:
            obstacle_transform = obstacle_vehicle.get_transform()
            obstacle_location = (obstacle_transform.location.x, obstacle_transform.location.y, obstacle_transform.location.z)
            obstacle_yaw = obstacle_transform.rotation.yaw
            obstacle_polygon = self._bbox_to_polygon(obstacle_vehicle.bounding_box, obstacle_location, obstacle_yaw)
            obstacle_velocity_vec3d = obstacle_vehicle.get_velocity()
            obstacle_velocity = (obstacle_velocity_vec3d.x, obstacle_velocity_vec3d.y, obstacle_velocity_vec3d.z)
            self.obstacles_info.append({
                "location": obstacle_location,
                "yaw": obstacle_yaw,
                "velocity": obstacle_velocity,
                "polygon": obstacle_polygon
            })

        enemy_polygon_list = [enemy_info['polygon'] for enemy_info in self.enemies_info]
        obstacle_polygon_list = [obstacle_info['polygon'] for obstacle_info in self.obstacles_info]
        all_polygon_list = self.map_borders_union + enemy_polygon_list + obstacle_polygon_list
        ego_point = Point(self.ego_info['location'][:2])
        self.visible_area, self.visible_area_vertices_type, self.obstacles_visibility = get_fov_polygon(ego_point, 360, 50, all_polygon_list, ray_num=40)

        if plot:
            self.ax.clear()
            self._plot(
                ego_polygon=ego_polygon,
                enemy_polygon_list=enemy_polygon_list,
                obstacle_polygon_list=obstacle_polygon_list,
                fov_polygon=self.visible_area,
                map_borders=self.map_borders
            )
            plt.draw()
            plt.pause(0.01)
    
    def apply_control(self):
        self.ego_vehicle.apply_control(self.ego_agent.run_step())
        for i, enemy_agent in enumerate(self.enemy_agents):
            self.enemy_vehicles[i].apply_control(enemy_agent.run_step())
        for i, obstacle_agent in enumerate(self.obstacle_agents):
            if obstacle_agent is None:
                continue
            self.obstacle_vehicles[i].apply_control(obstacle_agent.run_step())


    def destroy_all(self):
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
        if self.ego_collision_sensor is not None:
            self.ego_collision_sensor.destroy()
        for enemy_vehicle in self.enemy_vehicles:
            enemy_vehicle.destroy()
        for obstacle_vehicle in self.obstacle_vehicles:
            obstacle_vehicle.destroy()
        self.tick()

        self.ego_vehicle = None
        self.ego_collision_sensor = None
        self.enemy_vehicles = []
        self.obstacle_vehicles = []
        self.ego_agent = None
        self.enemy_agents = []
        self.obstacle_agents = []

    def tick(self):
        self.world.tick()

    def done(self):
        if self.ego_agent is not None and self.ego_collision_sensor is not None:
            if self.ego_agent.done():
                print('======== Success, Arrivied at Target Point!')
                return True
            elif self.ego_collision_sensor.is_collided():
                print('======== Collision!')
                return True
            else:
                return False
        else:
            print("Please spawn vehicles first.")
            return False

    def plot(self):
        self.ax.clear()
        self._plot(
            ego_polygon=self.ego_info['polygon'],
            enemy_polygon_list=[enemy_info['polygon'] for enemy_info in self.enemies_info],
            obstacle_polygon_list=[obstacle_info['polygon'] for obstacle_info in self.obstacles_info],
            fov_polygon=self.visible_area,
            map_borders=self.map_borders
        )
        plt.show()

    def _plot(self, ego_polygon: Polygon = None, enemy_polygon_list: List[Polygon] = None, obstacle_polygon_list: List[Polygon] = None, fov_polygon: Polygon = None, map_borders: Tuple[LineString, Polygon] = None):
        if map_borders is not None and map_borders != []:
            self.ax.set_facecolor('gray')
            x_outer = [coords[0] for coords in map_borders[0].coords]
            y_outer = [coords[1] for coords in map_borders[0].coords]
            self.ax.fill(x_outer, y_outer, color='white', alpha=1)

            for polygon in map_borders[1:]:
                x_inner, y_inner = polygon.exterior.xy
                self.ax.fill(x_inner, y_inner, color='gray', alpha=1)

        if ego_polygon is not None:
            x, y = ego_polygon.exterior.xy
            self.ax.fill(x, y, alpha=1, color='red', edgecolor='none')

        if enemy_polygon_list is not None and enemy_polygon_list != []:
            for enemy in enemy_polygon_list:
                x, y = enemy.exterior.xy
                self.ax.fill(x, y, alpha=1, color='blue', edgecolor='none')

        if obstacle_polygon_list is not None and obstacle_polygon_list != []:
            for obstacle in obstacle_polygon_list:
                x, y = obstacle.exterior.xy
                self.ax.fill(x, y, alpha=1, color='cyan', edgecolor='none')

        if fov_polygon is not None:
            x, y = fov_polygon.exterior.xy
            self.ax.fill(x, y, alpha=0.3, color='green', edgecolor='none')

        self.ax.set_aspect('equal', 'box')

    def _load_map_borders(self, map_name: str):
        with open(f'map_cache/{map_name}_map_borders.pkl', 'rb') as f:
            map = pickle.load(f)
        if map_name == 'Town01':
            self.map_borders = [LineString(map[0])] + [Polygon(map_line) for map_line in map[1:]]
            self.map_borders_union = [unary_union(self.map_borders)]
        else:
            raise ValueError("Map name not supported.")

    def _random_remove(self, agent_list: list, remove_prob: float):
        for item in agent_list[:]:
            if random.random() < remove_prob:
                agent_list.remove(item)

    def _parse_waypoint(self, waypoint: Union[int, float, tuple]) -> carla.Transform:
        if isinstance(waypoint, (int, float)):
            return self.spawn_points[waypoint]
        elif isinstance(waypoint, tuple):
            return carla.Transform(
                carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2]),
                carla.Rotation(yaw=waypoint[3], pitch=waypoint[4], roll=waypoint[5])
                )
        
    def _bbox_to_polygon(self, bbox: carla.BoundingBox, location: List[float], yaw_deg: float) -> Polygon:
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

if __name__ == "__main__":
    env_manager = EnvManager('T-intersection', 'localhost', 2000)
    env_manager.spawn_vehicles_from_config()
    env_manager.assign_agents_to_vehicles()
    print(env_manager.obstacle_agents)
    while not env_manager.done():
        env_manager.tick()
        env_manager.update_info(plot=True)
        env_manager.apply_control()
    env_manager.destroy_all()