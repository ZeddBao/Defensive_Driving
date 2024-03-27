from typing import List

import numpy as np
from shapely.geometry import Polygon, LineString
from matplotlib.axes import Axes

def plot_scene(
        ax: Axes,
        ego_polygon: Polygon = None,
        enemy_polygon_list: List[Polygon] = None,
        obstacle_polygon_list: List[Polygon] = None,
        fov_polygon: Polygon = None,
        map_borders: List[LineString] = None,
        nearby_lanes: List[np.ndarray] = None,
        centered: bool = True
        ) -> None:
    
    if map_borders is not None and map_borders != []:
        ax.set_facecolor('gray')
        x_outer = [coords[0] for coords in map_borders[0].coords]
        y_outer = [coords[1] for coords in map_borders[0].coords]
        ax.fill(x_outer, y_outer, color='white', alpha=1)

        for border in map_borders[1:]:
            x = [coords[0] for coords in border.coords]
            y = [coords[1] for coords in border.coords]
            ax.fill(x, y, color='gray', alpha=1)

    if nearby_lanes is not None and nearby_lanes != []:
        for group in nearby_lanes:
            for lane in group:
                ax.plot(lane[:, 0], lane[:, 1], 
                    linestyle='dotted', linewidth=2,
                    color='orange')

    if ego_polygon is not None:
        x, y = ego_polygon.exterior.xy
        ax.fill(x, y, alpha=1, color='red', edgecolor='none')

    if enemy_polygon_list is not None and enemy_polygon_list != []:
        for enemy in enemy_polygon_list:
            x, y = enemy.exterior.xy
            ax.fill(x, y, alpha=1, color='blue', edgecolor='none')

    if obstacle_polygon_list is not None and obstacle_polygon_list != []:
        for obstacle in obstacle_polygon_list:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, alpha=1, color='cyan', edgecolor='none')

    if fov_polygon is not None:
        x, y = fov_polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, color='green', edgecolor='none')

    if centered:
        extent = 64
        ego_location = (ego_polygon.centroid.x, ego_polygon.centroid.y)
        ax.set_xlim(ego_location[0] - extent, ego_location[0] + extent)
        ax.set_ylim(ego_location[1] - extent, ego_location[1] + extent)

    ax.set_aspect('equal', 'box')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    pass
