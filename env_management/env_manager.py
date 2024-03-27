import random
import pickle
from typing import Union, List, Tuple


import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LinearRing, Point
from shapely.ops import unary_union

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.navigation import BehaviorAgent, BasicAgent, GlobalRoutePlanner
from agents import VehicleAgent
from perception import get_fov_polygon, get_forward_lanes
from visualization import plot_scene

FOV = 180
FOV_RANGE = 50
RAY_NUM = 40
LANE_RANGE = 40
INTERVAL = 1
SEGMENT_WAYPOINTS_NUM = 20

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
    },

    {
        "map_name": "Town03",
        "ego_agent_config": {
            "agent_type": BasicAgent,
            "spawn_point": 49,
            "destination": 132,
            "vehicle_type": "vehicle.audi.a2",
            "color": "255, 0, 0",
            "target_speed": 100,
            "target_speed_std_dev": 10
        },
        "enemy_agents_config":[
            {   
                "agent_type": BasicAgent,
                "spawn_point": 238,
                "destination": 130,
                "vehicle_type": "vehicle.audi.a2",
                "color": "0, 0, 255",
                "target_speed": 30,
                "target_speed_std_dev": 10
            }
        ],
        "obstacle_agents_config":[
            {   
                "agent_type": None,
                "spawn_point": 129,
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

        self.ego_agent = None
        self.enemy_agents = []
        self.obstacle_agents = []

        self.nearby_lanes = []
        self.visible_area_vertices = []
        self.visible_area_vertices_type = []
        self.agents_visibility = []

        self.recorder = EnvRecorder()

        _, self.ax = plt.subplots()

        self.reset_config()
        

    def reset_config(self):
        if self.env_name == 'T-intersection':
            env_dict = random.choice(T_intersection_list)

            ############################################################################################################
            # TODO: Might to be modified
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

        self.nearby_lanes = get_forward_lanes(
            curr_waypoint=self.map.get_waypoint(carla.Location(*self.ego_agent.location)),
            interval=INTERVAL,
            search_depth=LANE_RANGE,
            segment_waypoints_num=SEGMENT_WAYPOINTS_NUM
            )

        enemy_polygon_list = [enemy_agent.bbox for enemy_agent in self.enemy_agents]
        obstacle_polygon_list = [obstacle_agent.bbox for obstacle_agent in self.obstacle_agents]
        all_polygon_list = self.map_borders_union + enemy_polygon_list + obstacle_polygon_list
        ego_point = self.ego_agent.location[:2]
        self.visible_area_vertices, self.visible_area_vertices_type, self.agents_visibility = get_fov_polygon(
            observer=ego_point,
            fov=FOV,
            max_distance=FOV_RANGE,
            obstacles=all_polygon_list,
            ray_num=RAY_NUM
            )
        
        # 去除 map_borders_union 的可见性
        self.agents_visibility = self.agents_visibility[1:]

        if plot:
            self.ax.clear()
            fov_polygon = Polygon(self.visible_area_vertices + [ego_point]) if FOV < 360 and FOV != 180 else Polygon(self.visible_area_vertices)
            plot_scene(
                ax=self.ax,
                ego_polygon=self.ego_agent.bbox,
                enemy_polygon_list=enemy_polygon_list,
                obstacle_polygon_list=obstacle_polygon_list,
                fov_polygon=fov_polygon,
                map_borders=self.map_borders,
                nearby_lanes=self.nearby_lanes
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
                self.recorder.set_collision()
                return True
            else:
                return False
        else:
            print("Please spawn vehicles first.")
            return None

    def plot(self):
        self.ax.clear()
        ego_point = self.ego_agent.location[:2]
        fov_polygon = Polygon(self.visible_area_vertices + [ego_point]) if FOV < 360 and FOV != 180 else Polygon(self.visible_area_vertices)
        plot_scene(
            ax=self.ax,
            ego_polygon=self.ego_agent.bbox,
            enemy_polygon_list=[enemy_agent.bbox for enemy_agent in self.enemy_agents],
            obstacle_polygon_list=[obstacle_agent for obstacle_agent in self.obstacle_agents],
            fov_polygon=fov_polygon,
            map_borders=self.map_borders,
            nearby_lanes=self.nearby_lanes
        )
        plt.show()

    ############################################################################################################
    def init_recorder_info(self):
        if self.ego_agent is None or self.map_name is None:
            print("World or Ego agent not initialized.")
        else:
            self.recorder.init_info(
                map_name=self.map_name,
                ego_agent=self.ego_agent,
                enemy_agents=self.enemy_agents,
                obstacle_agents=self.obstacle_agents
            )

    def collect_dataframe(self):
        if self.ego_agent is None or self.map_name is None:
            print("World or Ego agent not initialized.")
        else:
            self.recorder.collect(
                ego_agent=self.ego_agent,
                enemy_agents=self.enemy_agents,
                obstacle_agents=self.obstacle_agents,
                nearby_lanes=self.nearby_lanes,
                visible_area=self.visible_area_vertices,
                visible_area_vertices_type=self.visible_area_vertices_type,
                agents_visibility=self.agents_visibility
            )

    def save_dataframe(self, path: str=None):
        self.recorder.save(path)

    def reset_recorder(self):
        self.recorder.reset()
    ############################################################################################################

    def _load_map_borders(self, map_name: str):
        with open(f'map_cache/{map_name}_map_borders.pkl', 'rb') as f:
            map = pickle.load(f)
        if map_name in ['Town01', 'Town03']:
            self.map_borders = [LinearRing(map_line) for map_line in map]
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

class EnvRecorder():
    def __init__(self):
        self.reset()

    def reset(self):
        self.map_name: str = None

        self.ego_agent_info: dict = None
        self.enemy_agents_info: List[dict] = []
        self.obstacle_agents_info: List[dict] = []

        self.ego_agent_history: List[np.ndarray] = []
        # self.ego_agent_bbox_history = []
        self.enemy_agents_history: List[List[np.ndarray]] = []
        # self.enemy_agent_bboxes_history = []
        self.obstacle_agents_history: List[List[np.ndarray]]= []
        # self.obstacle_agent_bboxes_history = []
        self.nearby_lanes_history: List[List[np.ndarray]] = []
        self.visible_area_vertices_history: List[List[Tuple[float, float]]] = []
        self.visible_area_vertices_type_history: List[List[Union[int, float]]] = []
        self.agents_visibility_history: List[List[int]] = []

        self.collision = False

    def collect(
            self,
            ego_agent: VehicleAgent,
            enemy_agents: List[VehicleAgent],
            obstacle_agents: List[VehicleAgent],
            nearby_lanes: List[np.ndarray],
            visible_area: Polygon,
            visible_area_vertices_type: List[int],
            agents_visibility: List[int]
            ):

        self.ego_agent_history.append(ego_agent.get_info_numpy())
        self.enemy_agents_history.append([enemy_agent.get_info_numpy() for enemy_agent in enemy_agents])
        self.obstacle_agents_history.append([obstacle_agent.get_info_numpy() for obstacle_agent in obstacle_agents])
        self.nearby_lanes_history.append(nearby_lanes)
        self.visible_area_vertices_history.append(visible_area)
        self.visible_area_vertices_type_history.append(visible_area_vertices_type)
        self.agents_visibility_history.append(agents_visibility)

    def init_info(
            self,
            map_name: str,
            ego_agent: VehicleAgent,
            enemy_agents: List[VehicleAgent],
            obstacle_agents: List[VehicleAgent]
            ):

        self.map_name = map_name
        self.ego_agent_info = {
            'id': ego_agent.id,
            'type_id': ego_agent.type_id,
            'extent': ego_agent.extent
        }
        self.enemy_agents_info = [{
            'id': enemy_agent.id,
            'type_id': enemy_agent.type_id,
            'extent': enemy_agent.extent
        } for enemy_agent in enemy_agents]
        self.obstacle_agents_info = [{
            'id': obstacle_agent.id,
            'type_id': obstacle_agent.type_id,
            'extent': obstacle_agent.extent
        } for obstacle_agent in obstacle_agents]

    def set_collision(self):
        self.collision = True

    def save(self, path: str=None):
        dataframe = vars(self)
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(dataframe, f)
        return dataframe


if __name__ == "__main__":
    import time

    env_manager = EnvManager('T-intersection', 'localhost', 2000)
    env_manager.spawn_vehicles_from_config()
    env_manager.assign_agents_from_config()
    env_manager.init_recorder_info()
    while not env_manager.done():
        # start_time = time.time()
        env_manager.tick()
        env_manager.update_info(plot=True)
        env_manager.collect_dataframe()
        env_manager.apply_control()
        end_time = time.time()
        # print(f"Time cost: {end_time - start_time:.3f}s")
    env_manager.destroy_all()
    dataframe = env_manager.save_dataframe('test2.pkl')
    env_manager.reset_recorder()