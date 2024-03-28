import pickle
from typing import List, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from risk_function import radial_basis_function, inverse_distance_weighting_function, logistic_function

MAX_SEGMENTS_NUM = 10
MAX_WAYPOINTS_NUM = 200
SEGMENTS_WAYPOINTS_NUM = 20
FOV_RANGE = 50

import torch

def get_ego_state_history(ego_agent_history: List[np.ndarray]) -> torch.Tensor:
    """
    获取自车历史位置

    :param ego_agent_history: 自车历史状态
    :return: 自车历史状态特征   [batch_size, 3] (x, y, yaw)
    """

    return torch.tensor(np.stack(ego_agent_history), dtype=torch.float32)[:, :3]


def transform_func_factory(ego_agent_state: torch.Tensor) -> Callable:
    """
    创建全局坐标系到局部坐标系的转换函数

    :param ego_state: 自车状态 (batch_size, x, y, yaw)
    :return: 转换函数
    """

    def transform_func(tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """
        将全局坐标系下的点或向量转换到局部坐标系

        :param tensor: 全局坐标系下的点 [batch_size, points_num, 2] or [batch_size, segments_num, segment_points_num, 2]
        :param mode: 转换模式，'vector'表示向量，'point'表示点
        :return: 局部坐标系下的点
        """

        yaw = ego_agent_state[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        transform_matrix = torch.zeros(yaw.size(0), 2, 2, device=yaw.device)
        transform_matrix[:, 0, 0] = cos_yaw
        transform_matrix[:, 0, 1] = -sin_yaw
        transform_matrix[:, 1, 0] = sin_yaw
        transform_matrix[:, 1, 1] = cos_yaw

        tensor_dim = tensor.dim()
        if tensor_dim == 4:
            transform_matrix = transform_matrix.unsqueeze(1)    # [batch_size, 2, 2] -> [batch_size, 1, 2, 2]
            ego_agent_location_unsqueezed = ego_agent_state[:, :2].unsqueeze(1).unsqueeze(1)    # [batch_size, 3] -> [batch_size, 1, 1, 2]
        elif tensor_dim == 3:
            ego_agent_location_unsqueezed = ego_agent_state[:, :2].unsqueeze(1) # [batch_size, 3] -> [batch_size, 1, 2]
        elif tensor_dim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)   # [2] -> [1, 1, 2]
            ego_agent_location_unsqueezed = ego_agent_state[:, :2].unsqueeze(1) # [batch_size, 3] -> [batch_size, 1, 2]
        else:
            raise ValueError('Invalid tensor dimension.')

        if mode == 'point':
            tensor = tensor - ego_agent_location_unsqueezed
                # [batch_size, points_num, 2] - [batch_size, 1, 3]
                # or [batch_size, segments_num, segment_points_num, 2] - [batch_size, 1, 1, 3]
                # or [1, 1, 2] - [batch_size, 1, 2]
        tensor = tensor @ transform_matrix
        return tensor.squeeze()

    return transform_func   

def preprocess_visible_area(
        transform_func: Callable,
        visible_area_vertices_history: List[List[Tuple[float, float]]],
        visible_area_vertices_type_history: List[List[Union[int, float]]],
        reference_mode: str='local'
        ) -> torch.Tensor:
    """
    预处理可见区域

    :param transform_func: 坐标转换函数
    :param visible_area_vertices_history: 可见区域顶点历史记录 [batch_size, visible_area_vertices_num, 2]
    :param visible_area_vertices_type_history: 可见区域顶点类型历史记录 [batch_size, visible_area_vertices_num]
    :param reference_mode: 参考坐标系，'local'表示局部坐标系，'global'表示全局坐标系
    :return: 可见区域特征
    """

    visible_area_vertices_history_tensor = torch.tensor(visible_area_vertices_history, dtype=torch.float32)
    if reference_mode == 'local':
        visible_area_vertices_history_tensor = transform_func(visible_area_vertices_history_tensor, 'point')
    elif reference_mode == 'global':
        pass
    else:
        raise ValueError('Invalid reference mode.')
    visible_area_vertices_type_history_tensor = torch.tensor(visible_area_vertices_type_history, dtype=torch.float32)
    return torch.cat((visible_area_vertices_history_tensor, visible_area_vertices_type_history_tensor.unsqueeze(-1)), dim=-1)

def preprocess_lane_segments(
        transform_func: Callable,
        nearby_lanes_history: List[List[Union[np.ndarray, torch.Tensor]]],
        reference_mode: str='local'
        ) -> torch.Tensor:
    """
    预处理车道线

    :param transform_func: 坐标转换函数
    :param nearby_lanes_history: 附近车道线段
    :param reference_mode: 参考坐标系，'local'表示局部坐标系，'global'表示全局坐标系
    :return: 车道线分段特征
    """

    inner_tensors = []
    need_transfer = isinstance(nearby_lanes_history[0][0], np.ndarray)
    for nearby_lanes in nearby_lanes_history:
        if need_transfer:
            nearby_lanes_np = np.vstack(nearby_lanes)[:, :, :6]
            nearby_lanes_tensor = torch.tensor(nearby_lanes_np, dtype=torch.float32)
        else:
            nearby_lanes_tensor = torch.cat(nearby_lanes, dim=0)[:, :, :6]
        pad_size = MAX_SEGMENTS_NUM - nearby_lanes_tensor.shape[0]
        if pad_size > 0:
            nearby_lanes_tensor = F.pad(
                input=nearby_lanes_tensor,
                pad=(0, 0, 0, 0, 0, pad_size),
                mode='constant',
                value=float('nan')
            )
        inner_tensors.append(nearby_lanes_tensor.unsqueeze(dim=0))
    final_tensor = torch.cat(inner_tensors, dim=0)
    if reference_mode == 'local':
        final_tensor[:, :, :, :2] = transform_func(final_tensor[:, :, :, :2], 'point')
        final_tensor[:, :, :, 2:4] = transform_func(final_tensor[:, :, :, 2:4], 'vector')
    elif reference_mode == 'global':
        pass
    else:
        raise ValueError('Invalid reference mode.')
    return final_tensor

def calculate_segments_label():
    pass

def preprocess_lane_waypoints(
        transform_func: Callable,
        nearby_lanes_history: List[List[Union[np.ndarray, torch.Tensor]]],
        reference_mode: str='local'
        ) -> torch.Tensor:
    """
    预处理车道线上的路点

    :param transform_func: 坐标转换函数
    :param nearby_waypoints: 附近车道线段 [batch_size, waypoints_num, 6]
    :param reference_mode: 参考坐标系，'local'表示局部坐标系，'global'表示全局坐标系
    :return: 路点特征
    """

    inner_tensors = []
    need_transfer = isinstance(nearby_lanes_history[0][0], np.ndarray)

    for nearby_lanes in nearby_lanes_history:
        if need_transfer:
            nearby_lanes_np = np.vstack(nearby_lanes)[:, :, :6]
            nearby_lanes_tensor = torch.tensor(nearby_lanes_np, dtype=torch.float32)
        else:
            nearby_lanes_tensor = torch.cat(nearby_lanes, dim=0)[:, :, :6]
        
        nearby_lanes_tensor = nearby_lanes_tensor.reshape(-1, 6)
        # print(nearby_lanes_tensor.shape)
        nearby_lanes_tensor = nearby_lanes_tensor.unique(dim=0)
        # print(nearby_lanes_tensor.shape)
        # print('-------------------')
        pad_size = MAX_WAYPOINTS_NUM - nearby_lanes_tensor.shape[0]
        if pad_size > 0:
            nearby_lanes_tensor = F.pad(
                input=nearby_lanes_tensor,
                pad=(0, 0, 0, pad_size),  # 反向填充
                mode='constant',
                value=float('nan')
            )
        inner_tensors.append(nearby_lanes_tensor.unsqueeze(dim=0))
    final_tensor = torch.cat(inner_tensors, dim=0)
    if reference_mode == 'local':
        final_tensor[:, :, :2] = transform_func(final_tensor[:, :, :2], 'point')
        final_tensor[:, :, 2:4] = transform_func(final_tensor[:, :, 2:4], 'vector')
    elif reference_mode == 'global':
        pass
    else:
        raise ValueError('Invalid reference mode.')
    return final_tensor

def calculate_waypoints_risk(
        transform_func: Callable,
        lane_waypoints_batch: torch.Tensor,
        collision_location_global: Union[Tuple[float, float], np.ndarray, torch.Tensor],
        ) -> torch.Tensor:  # [batch_size, waypoints_num]
    
    if collision_location is not None:
        collision_location_local = transform_collsion_location(transform_func, collision_location_global).unsqueeze(1)  # [batch_size, 2] -> [batch_size, 1, 2]
        lane_waypoints_locations = lane_waypoints_batch[:, :, :2]   # [batch_size, waypoints_num, 2]
        distance = torch.norm(lane_waypoints_locations - collision_location_local, dim=-1)  # [batch_size, waypoints_num]

        risk = torch.zeros_like(distance)
        mask = distance[:, 0] < FOV_RANGE
        risk[mask] = radial_basis_function(distance[mask], alpha=0.2)
        return risk
    else:
        shape = list(lane_waypoints_batch.shape)
        shape[-1] = 1
        return torch.zeros(shape, dtype=torch.float32)
    
def transform_collsion_location(
        transform_func: Callable,
        collision_location_global: Union[Tuple[float, float], np.ndarray, torch.Tensor],
        ) -> torch.Tensor:
    """
    转换碰撞位置

    :param transform_func: 坐标转换函数
    :param collision_location_global: 碰撞位置
    :return: 转换后的碰撞位置
    """

    return transform_func(collision_location_global, 'point')
    
     
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open('test.pkl', 'rb') as f:
        data = pickle.load(f)

    ego_agent_history = data['ego_agent_history']
    ego_state_history = get_ego_state_history(ego_agent_history)    

    transform_func = transform_func_factory(ego_state_history)
    visible_area_history = data['visible_area_vertices_history']
    visible_area_vertices_type_history = data['visible_area_vertices_type_history']
    visible_area_batch = preprocess_visible_area(transform_func, visible_area_history, visible_area_vertices_type_history)
    print('visible_area_batch shape:', visible_area_batch.shape)

    nearby_lanes_history = data['nearby_lanes_history']
    lane_segments_batch = preprocess_lane_segments(transform_func, nearby_lanes_history)
    print('lane_segments_batch shape:', lane_segments_batch.shape)

    lane_waypoints_batch = preprocess_lane_waypoints(transform_func, nearby_lanes_history)
    print('lane_waypoints_batch shape:', lane_waypoints_batch.shape)

    collision = data['collision']
    collision_location = ego_state_history[-1, :2] if collision else None
    collision_location_local = transform_collsion_location(transform_func, collision_location)
    print('collision_location_local shape:', collision_location_local.shape)

    waypoints_risk = calculate_waypoints_risk(transform_func, lane_waypoints_batch, collision_location)
    print('waypoints_risk shape:', waypoints_risk.shape)

    
    fig, ax = plt.subplots()
    for i, lane_waypoints in enumerate(lane_waypoints_batch):
        ax.scatter(lane_waypoints[:, 0], lane_waypoints[:, 1])
        if collision:
            colors = np.zeros((MAX_WAYPOINTS_NUM, 4))
            colors[:, 0] = 1.0
            colors[:, 3] = waypoints_risk[i, :]
            ax.scatter(lane_waypoints[:, 0], lane_waypoints[:, 1], c=colors)
            ax.plot(collision_location_local[i, 0], collision_location_local[i, 1], 'r*', markersize=10)
        ax.fill(visible_area_batch[i, :, 0], visible_area_batch[i, :, 1], alpha=0.3, color='green')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        plt.draw()
        plt.pause(0.1)
        ax.clear()

