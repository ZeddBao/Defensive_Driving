import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from perception.get_fov_polygon import get_fov_polygon


def generate_trajectory_ribbon(x: list, y: list, ribbon_width: float, yaw: list) -> Polygon:
    """
    根据x, y, yaw生成轨迹带
    """

    def get_sides(x: list, y: list, yaw: list, is_radian: bool = False) -> list:
        """
        根据x, y, yaw获取轨迹带的两边
        """
        if not is_radian:
            yaw = np.deg2rad(yaw)
        # 计算轨迹带的左边
        x_left = x - ribbon_width / 2 * np.sin(yaw)
        y_left = y + ribbon_width / 2 * np.cos(yaw)

        # 计算轨迹带的右边
        x_right = x + ribbon_width / 2 * np.sin(yaw)
        y_right = y - ribbon_width / 2 * np.cos(yaw)

        left_side = list(zip(x_left, y_left))
        right_side = list(zip(x_right, y_right))

        return left_side, right_side

    left_side, right_side = get_sides(x, y, yaw)
    right_side = right_side[::-1]
    return Polygon(left_side + right_side)


def generate_agent_box(x: float, y: float, box_size: tuple, yaw: float, is_radian: bool = False) -> Polygon:
    """
    根据x, y, yaw生成代理车辆的盒子
    """
    if not is_radian:
        yaw = np.deg2rad(yaw)

    corners = np.array([
        [-box_size[0] / 2, -box_size[1] / 2],
        [box_size[0] / 2, -box_size[1] / 2],
        [box_size[0] / 2, box_size[1] / 2],
        [-box_size[0] / 2, box_size[1] / 2]
    ])

    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    rotated_corners = corners @ rotation_matrix.T
    rotated_corners += np.array([x, y])
    return Polygon(rotated_corners)


if __name__ == "__main__":
    with open('visualization/map_lane_sides.pkl', 'rb') as f:
        map = pickle.load(f)

    map = Polygon(shell=map[0], holes=map[1:])
    fig, ax = plt.subplots()
    x_outer, y_outer = map.exterior.xy
    inner_coords = [hole.xy for hole in map.interiors]
    x_inner = [coords[0] for coords in inner_coords]
    y_inner = [coords[1] for coords in inner_coords]
    ax.fill(x_outer, y_outer, color='grey', alpha=0.5)

    for i in range(len(map.interiors)):
        ax.fill(x_inner[i], y_inner[i], color='white')

    with open('visualization/record_overtake_no_yield_episode_2.pkl', 'rb') as f:
        records = pickle.load(f)

    tick = 100
    agent_box_list = []

    for i, record in enumerate(records.values()):
        # random color
        if i == 0:
            color = 'green'
        else:
            color = np.random.rand(3,)

        agent_x = record['x']
        agent_y = record['y']
        agent_yaw = record['yaw']
        agent_box = generate_agent_box(
            agent_x[tick], agent_y[tick], (4.8, 1.8), agent_yaw[tick])
        agent_box_x, agent_box_y = agent_box.exterior.xy
        ax.fill(agent_box_x, agent_box_y, color=color, alpha=1)

        agent_box_list.append(agent_box)
        ax.plot(agent_x[tick:], agent_y[tick:],
                color=color, alpha=0.8, linestyle='--')
        # 画出轨迹带
        # traj_ribbon = generate_trajectory_ribbon(x, y, 1.8, yaw)
        # ribbon_x, ribbon_y = traj_ribbon.exterior.xy
        # ax.fill(ribbon_x, ribbon_y, color='blue', alpha=0.4, edgecolor='none')

    ego_x = records['ego_agent']['x'][tick]
    ego_y = records['ego_agent']['y'][tick]

    obstacle_type = [2] * (len(agent_box_list)-1)
    structured_obstacles = list(zip(agent_box_list[1:], obstacle_type))

    fov_polygon, _, _ = get_fov_polygon(
        Point(ego_x, ego_y), 360, 100, structured_obstacles, ray_num=3600)
    x, y = fov_polygon.exterior.xy
    ax.fill(x, y, alpha=0.3, color='green', edgecolor='none')

    ax.set_aspect('equal', 'box')
    plt.show()
