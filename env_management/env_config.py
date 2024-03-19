import random
from typing import Union, List, Tuple

import numpy as np
import carla

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

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
        self.reset()

    def reset(self):
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
            
            self.ego_agent['target_speed'] = np.random.normal(self.ego_agent['target_speed'], self.ego_agent['target_speed_std_dev'])

            self._random_remove(self.enemy_agents, 0.1)
            self._random_remove(self.obstacle_agents, 0.5)

            for enemy_agent in self.enemy_agents:
                enemy_agent['target_speed'] = np.random.normal(enemy_agent['target_speed'], enemy_agent['target_speed_std_dev'])
            for obstacle_agent in self.obstacle_agents:
                if obstacle_agent['target_speed'] is None:
                    continue
                obstacle_agent['target_speed']

    def _random_remove(self, agent_list: list, remove_prob: float):
        for item in agent_list[:]:
            # 生成一个0到1之间的随机数
            if random.random() < remove_prob:
                # 如果随机数小于0.1，删除原始列表中的该元素
                agent_list.remove(item)

    def _parse_waypoint(waypoint: Union[int, float, tuple], all_default_spawn: list) -> carla.Transform:
        if isinstance(waypoint, (int, float)):
            pass
        elif isinstance(waypoint, tuple):
            return carla.Transform(carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2]), carla.Rotation(yaw=waypoint[3], pitch=waypoint[4], roll=waypoint[5]))

