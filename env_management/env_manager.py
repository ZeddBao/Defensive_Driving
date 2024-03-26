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
from visualization.plot_scene import plot_scene

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

def carla_bbox_to_polygon(bbox: carla.BoundingBox, location: List[float], yaw_deg: float) -> Polygon:
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

        self.ego_agent = None
        self.enemy_agents = []
        self.obstacle_agents = []

        self.map_lanes = []
        self.visible_area = []
        self.visible_area_vertices_type = []
        self.obstacles_visibility = []

        _, self.ax = plt.subplots()

        self.reset_config()
        

    def reset_config(self):
        if self.env_name == 'T-intersection':
            env_dict = random.choice(T_intersection_list)

            ############################################################################################################
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

            ############################################################################################################

            self._random_remove(self.enemy_agents_config, 0.1)
            self._random_remove(self.obstacle_agents_config, 0.5)

            for enemy_agent_config in self.enemy_agents_config:
                enemy_agent_config['target_speed'] = np.random.normal(enemy_agent_config['target_speed'], enemy_agent_config['target_speed_std_dev'])
            for obstacle_agent_config in self.obstacle_agents_config:
                if obstacle_agent_config['target_speed'] is None:
                    continue
                obstacle_agent_config['target_speed'] = np.random.normal(obstacle_agent_config['target_speed'], obstacle_agent_config['target_speed_std_dev'])

    def spawn_vehicles_from_config(self):
        if self.world is None:
            print("Please reset the environment first.")
            return
        
        self.ego_agent = VehicleAgent(self.world.spawn_actor(
            self.blueprint_library.find(self.ego_agent_config['vehicle_type']),
            self._parse_waypoint(self.ego_agent_config['spawn_point'])
            ))
        
        self.ego_agent.init_collision_sensor()

        for enemy_agent_config in self.enemy_agents_config:
            enemy_agent = VehicleAgent(self.world.spawn_actor(
                self.blueprint_library.find(enemy_agent_config['vehicle_type']),
                self.spawn_points[enemy_agent_config['spawn_point']]
                ))
            self.enemy_agents.append(enemy_agent)

        for obstacle_agent_config in self.obstacle_agents_config:
            if obstacle_agent_config['spawn_point'] is None:
                continue
            obstacle_agent = VehicleAgent(self.world.spawn_actor(
                self.blueprint_library.find(obstacle_agent_config['vehicle_type']),
                self.spawn_points[obstacle_agent_config['spawn_point']]
                ))
            self.obstacle_agents.append(obstacle_agent)
        
        self.tick()

    def assign_agents_from_config(self):
        if self.ego_agent is not None:
            self.ego_agent.init_agent(
                agent_type=self.ego_agent_config['agent_type'],
                target_speed=self.ego_agent_config['target_speed'],
                map_inst=self.map,
                grp_inst=self.global_planner
                )
            self.ego_agent.set_destination(self._parse_waypoint(self.ego_agent_config['destination']).location)
        for i, enemy_agent in enumerate(self.enemy_agents):
            enemy_agent.init_agent(
                agent_type=self.enemy_agents_config[i]['agent_type'],
                target_speed=self.enemy_agents_config[i]['target_speed'],
                map_inst=self.map,
                grp_inst=self.global_planner
                )
            enemy_agent.set_destination(self._parse_waypoint(self.enemy_agents_config[i]['destination']).location)
        for i, obstacle_agent in enumerate(self.obstacle_agents):
            if self.obstacle_agents_config[i]['agent_type'] is None:
                continue
            elif self.obstacle_agents_config[i]['agent_type'] == BasicAgent:
                obstacle_agent.init_agent(
                    agent_type=self.obstacle_agents_config[i]['agent_type'],
                    target_speed=self.obstacle_agents_config[i]['target_speed'],
                    map_inst=self.map,
                    grp_inst=self.global_planner
                    )
                obstacle_agent.set_destination(self._parse_waypoint(self.obstacle_agents_config[i]['destination']).location)
            elif self.obstacle_agents_config[i]['agent_type'] == BehaviorAgent:
                obstacle_agent.init_agent(
                    agent_type=self.obstacle_agents_config[i]['agent_type'],
                    behavior=self.obstacle_agents_config[i]['behavior'],
                    map_inst=self.map,
                    grp_inst=self.global_planner
                    )
                obstacle_agent.set_destination(self._parse_waypoint(self.obstacle_agents_config[i]['destination']).location)

    def update_info(self, plot=False):
        self.ego_agent.update_info()
        for enemy_agent in self.enemy_agents:
            enemy_agent.update_info()
        for obstacle_agent in self.obstacle_agents:
            obstacle_agent.update_info()

        self.map_lanes = get_map_lanes(
            curr_waypoint=self.map.get_waypoint(carla.Location(*self.ego_agent.location)),
            interval=1,
            search_depth=100,
            segment_waypoints_num=20
            )

        enemy_polygon_list = [enemy_agent.bbox for enemy_agent in self.enemy_agents]
        obstacle_polygon_list = [obstacle_agent.bbox for obstacle_agent in self.obstacle_agents]
        all_polygon_list = self.map_borders_union + enemy_polygon_list + obstacle_polygon_list
        ego_point = Point(self.ego_agent.location[:2])
        self.visible_area, self.visible_area_vertices_type, self.obstacles_visibility = get_fov_polygon(
            observer=ego_point,
            fov=360,
            max_distance=50,
            obstacles=all_polygon_list,
            ray_num=40
            )

        if plot:
            self.ax.clear()
            plot_scene(
                ax=self.ax,
                ego_polygon=self.ego_agent.bbox,
                enemy_polygon_list=enemy_polygon_list,
                obstacle_polygon_list=obstacle_polygon_list,
                fov_polygon=self.visible_area,
                map_borders=self.map_borders
            )
            plt.draw()
            plt.pause(0.01)
    
    def apply_control(self):
        self.ego_agent.apply_control()
        for enemy_agent in self.enemy_agents:
            enemy_agent.apply_control()
        for obstacle_agent in self.obstacle_agents:
            if obstacle_agent.agent is None:
                continue
            obstacle_agent.apply_control()

    def destroy_all(self):
        if self.ego_agent is not None:
            self.ego_agent.destroy()
        for enemy_agent in self.enemy_agents:
            enemy_agent.destroy()
        for obstacle_agent in self.obstacle_agents:
            obstacle_agent.destroy()
        self.tick()

        self.ego_agent = None
        self.enemy_agents = []
        self.obstacle_agents = []

    def tick(self):
        self.world.tick()

    def done(self):
        if self.ego_agent is not None:
            if self.ego_agent.done():
                print('>>> Success, Arrivied at Target Point!')
                return True
            elif self.ego_agent.collision_sensor is None:
                print('Collision Sensor not initialized.')
                return 
            elif self.ego_agent.is_collided():
                print('>>> Collision!')
                return True
            else:
                return False
        else:
            print("Please spawn vehicles first.")
            return None

    def plot(self):
        self.ax.clear()
        plot_scene(
            ax=self.ax,
            ego_polygon=self.ego_agent.bbox,
            enemy_polygon_list=[enemy_agent.bbox for enemy_agent in self.enemy_agents],
            obstacle_polygon_list=[obstacle_agent for obstacle_agent in self.obstacle_agents],
            fov_polygon=self.visible_area,
            map_borders=self.map_borders
        )
        plt.show()

    # def _plot(self, ego_polygon: Polygon = None, enemy_polygon_list: List[Polygon] = None, obstacle_polygon_list: List[Polygon] = None, fov_polygon: Polygon = None, map_borders: Tuple[LineString, Polygon] = None):
    #     if map_borders is not None and map_borders != []:
    #         self.ax.set_facecolor('gray')
    #         x_outer = [coords[0] for coords in map_borders[0].coords]
    #         y_outer = [coords[1] for coords in map_borders[0].coords]
    #         self.ax.fill(x_outer, y_outer, color='white', alpha=1)

    #         for polygon in map_borders[1:]:
    #             x_inner, y_inner = polygon.exterior.xy
    #             self.ax.fill(x_inner, y_inner, color='gray', alpha=1)

    #     if ego_polygon is not None:
    #         x, y = ego_polygon.exterior.xy
    #         self.ax.fill(x, y, alpha=1, color='red', edgecolor='none')

    #     if enemy_polygon_list is not None and enemy_polygon_list != []:
    #         for enemy in enemy_polygon_list:
    #             x, y = enemy.exterior.xy
    #             self.ax.fill(x, y, alpha=1, color='blue', edgecolor='none')

    #     if obstacle_polygon_list is not None and obstacle_polygon_list != []:
    #         for obstacle in obstacle_polygon_list:
    #             x, y = obstacle.exterior.xy
    #             self.ax.fill(x, y, alpha=1, color='cyan', edgecolor='none')

    #     if fov_polygon is not None:
    #         x, y = fov_polygon.exterior.xy
    #         self.ax.fill(x, y, alpha=0.3, color='green', edgecolor='none')

    #     self.ax.set_aspect('equal', 'box')

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


class VehicleAgent():
    def __init__(self, vehicle: carla.Actor, agent=None):
        self.vehicle = vehicle
        self.agent = agent
        self.collision_sensor = None

        self.location: Tuple[float, float, float] = None
        self.velocity: Tuple[float, float, float] = None
        self.yaw: float = None
        self.bbox: Polygon = None

    def assign_agent(self, agent):
        self.agent = agent

    def init_agent(self, agent_type, **kwargs):
        self.agent = agent_type(self.vehicle, **kwargs)

    def init_collision_sensor(self):
        self.collision_sensor = CollisionSensor(self.vehicle)

    def set_destination(self, destination):
        self.agent.set_destination(destination)

    def get_velocity(self) -> Tuple[float, float, float]:
        velocity_vec3d = self.vehicle.get_velocity()
        self.velocity = (velocity_vec3d.x, velocity_vec3d.y, velocity_vec3d.z)
        return self.velocity
    
    def get_location(self) -> Tuple[float, float, float]:
        transform = self.vehicle.get_transform()
        self.location = (transform.location.x, transform.location.y, transform.location.z)
        return self.location
    
    def get_yaw(self) -> float:
        transform = self.vehicle.get_transform()
        self.yaw = transform.rotation.yaw
        return self.yaw
    
    def get_bbox(self) -> Polygon:
        self.bbox = carla_bbox_to_polygon(self.vehicle.bounding_box, self.get_location(), self.get_yaw())
        return self.bbox
    
    def update_info(self) -> Tuple[Tuple[float, float, float], float, Tuple[float, float, float], Polygon]:
        transform = self.vehicle.get_transform()
        self.location = (transform.location.x, transform.location.y, transform.location.z)
        self.yaw = transform.rotation.yaw
        self.get_velocity()
        self.bbox = carla_bbox_to_polygon(self.vehicle.bounding_box, self.location, self.yaw)
        return self.location, self.yaw, self.velocity, self.bbox

    def run_step(self):
        return self.agent.run_step()
    
    def apply_control(self, control=None):
        if control is None:
            self.vehicle.apply_control(self.run_step())
        else:
            self.vehicle.apply_control(control)

    def is_collided(self):
        if self.collision_sensor is not None:
            return self.collision_sensor.is_collided()
        else:
            print("Collision Sensor not initialized.")
            return None
    
    def done(self):
        return self.agent.done()

    def destroy(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.agent is not None:
            self.agent = None


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
    env_manager.assign_agents_from_config()
    while not env_manager.done():
        env_manager.tick()
        env_manager.update_info(plot=True)
        env_manager.apply_control()
    env_manager.destroy_all()