import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

T_intersection_list = [
    {
        "map": "Town01",
        "ego_agent": {
            "agent_type": BasicAgent,
            "spawn_point": 192,
            "destination": 118,
            "vehicle_type": "vehicle.audi.a2",
            "color": "255, 0, 0",
            "target_speed": 100
        },
        "enemy_agents":[
            {   
                "agent_type": BasicAgent,
                "spawn_point": 109,
                "destination": 73,
                "vehicle_type": "vehicle.audi.a2",
                "color": "0, 0, 255",
                "target_speed": 30
            }
        ],
        "obstacle_agents":[
            {   
                "agent_type": None,
                "spawn_point": 107,
                "destination": None,
                "vehicle_type": "vehicle.carlamotors.carlacola",
                "color": "0, 255, 255",
                "target_speed": None
            }
        ]
    }
]

# class EnvConfig():
#     def __init__(self, env_name: str):
#         if env_name == 'T-intersection':
#             env_dict = random.choice(T_intersection_list)
#             for key, value in env_dict.items():
#                 setattr(self, key, value)
