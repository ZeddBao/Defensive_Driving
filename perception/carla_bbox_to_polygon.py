from typing import List

import numpy as np
import carla
from shapely.geometry import Polygon

def carla_bbox_to_polygon(bbox: carla.BoundingBox, location: List[float], yaw_deg: float) -> Polygon:
    yaw = np.deg2rad(yaw_deg)
    half_box_size = (bbox.extent.x, bbox.extent.y)
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)
    rotation_matrix = np.array([
        [cos_yaw, sin_yaw],
        [-sin_yaw, cos_yaw]
    ])
    corners = np.array([
        [-half_box_size[0], -half_box_size[1]],
        [half_box_size[0]-0.5, -half_box_size[1]],
        [half_box_size[0], 0],
        [half_box_size[0]-0.5, half_box_size[1]],
        [-half_box_size[0], half_box_size[1]]
    ])
    rotated_corners = corners @ rotation_matrix
    rotated_corners += np.array(location[:2])
    return Polygon(rotated_corners)

if __name__ == '__main__':
    pass