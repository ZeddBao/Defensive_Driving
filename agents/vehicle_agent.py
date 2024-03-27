from typing import Tuple

import carla
import numpy as np
from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from perception import carla_bbox_to_polygon, CollisionSensor


class VehicleAgent():
    def __init__(self, vehicle: carla.Actor, agent=None):
        self.vehicle = vehicle
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        extent = vehicle.bounding_box.extent
        self.extent = (extent.x, extent.y, extent.z)

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
    
    def get_info(self) -> Tuple[Tuple[float, float, float], float, Tuple[float, float, float], Polygon]:
        return self.location, self.yaw, self.velocity, self.bbox
    
    def get_info_numpy(self):
        return np.array([*self.location, self.yaw, *self.velocity])

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