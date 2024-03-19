from typing import List, Union, Tuple
from collections import deque

import numpy as np
import carla

def get_all_lane_segments(curr_waypoint: carla.libcarla.Waypoint, interval: int = 1, total_search_depth: int = 100, segment_waypoints_num: int = 10) -> Tuple[np.array, np.array]:
    '''
    从一点开始，获取所有当前车道双向延伸的所有车道分段

    :param curr_waypoint: 当前车道
    :param interval: 间隔
    :param total_search_depth: 搜索深度
    :param segment_waypoints_num: 车道段的waypoints数量
    :return: 向前和向后的车道分段列表
    '''

    previous_lane_segments_list = []
    next_lane_segments_list = []
    breakpoint_queue = deque()

    def get_lane_segment(curr_waypoint: carla.libcarla.Waypoint, towards: str, interval: int, search_depth: int, segment_waypoints_num: int) -> List[List[int]]:
        nonlocal breakpoint_queue
        segment_waypoints_list = []
        next_waypoint = curr_waypoint

        for i in range(search_depth):
            if towards == 'next':
                next_waypoint_list = next_waypoint.next(interval)
            elif towards == 'previous':
                next_waypoint_list = next_waypoint.previous(interval)
            
            next_waypoint_list_len = len(next_waypoint_list)
            if next_waypoint_list is None:
                break
            elif next_waypoint_list_len > 1 or i == segment_waypoints_num:
                breakpoint_queue += list(zip(next_waypoint_list, [search_depth-i]*next_waypoint_list_len))
                break
            elif next_waypoint_list_len == 1:
                # segment_waypoints_list.append([next_waypoint.transform.location.x, next_waypoint.transform.location.y, next_waypoint.transform.location.z])
                # is_turn_left_allowed = next_waypoint.lane_change == carla.libcarla.LaneChange.Left or next_waypoint.lane_change == carla.libcarla.LaneChange.Both
                # is_turn_right_allowed = next_waypoint.lane_change == carla.libcarla.LaneChange.Right or next_waypoint.lane_change == carla.libcarla.LaneChange.Both
                vec_forward = next_waypoint.transform.get_forward_vector()
                vec_right = next_waypoint.transform.get_right_vector()
                lane_width = next_waypoint.lane_width
                segment_waypoints_list.append([
                    next_waypoint.transform.location.x, next_waypoint.transform.location.y,     # 0:2 # x, y
                    vec_forward.x, vec_forward.y,                                               # 2:4 # vec_forward
                    total_search_depth - search_depth + i,                                      # 4 # depth
                    int(next_waypoint.is_junction),                                             # 5 # is_junction
                    lane_width,                                                                 # 6 # lane_width
                    vec_right.x, vec_right.y])                                                  # 7:9 # vec_right
                if towards == 'next':
                    next_waypoint = next_waypoint.next(interval)[0]
                elif towards == 'previous':
                    next_waypoint = next_waypoint.previous(interval)[0]

        # 处理收集的一小段车道的 waypoints
        len_waypoints = len(segment_waypoints_list)

        if len_waypoints < segment_waypoints_num and len_waypoints > 0:
            # segment_waypoints_nparray = np.concatenate((segment_waypoints_list, np.tile(segment_waypoints_list[-1], (segment_waypoints_num-len_waypoints, 1))), axis=0)  # 用最后一个waypoint填充
            segment_waypoints_list = segment_waypoints_list + [segment_waypoints_list[-1]] * (segment_waypoints_num-len_waypoints)  # 用最后一个waypoint填充
        elif len_waypoints == 0:
            segment_waypoints_list = None
            
        return segment_waypoints_list
    
    breakpoint_queue.append((curr_waypoint, total_search_depth))
    while breakpoint_queue:
        (start_waypoint, num) = breakpoint_queue.popleft()
        next_segment = get_lane_segment(start_waypoint, 'next', interval, num, segment_waypoints_num)
        if next_segment is not None:
            next_lane_segments_list.append(next_segment)

    breakpoint_queue.append((curr_waypoint, total_search_depth))
    while breakpoint_queue:
        (start_waypoint, num) = breakpoint_queue.popleft()
        previous_segment = get_lane_segment(start_waypoint, 'previous', interval, num, segment_waypoints_num)
        if previous_segment is not None:
            previous_lane_segments_list.append(previous_segment)

    return np.array(next_lane_segments_list), np.array(previous_lane_segments_list)

def get_map_lanes(curr_waypoint: carla.libcarla.Waypoint, interval: int, search_depth: int, segment_waypoints_num: int) -> List[np.array]:
    left_lane_waypoint = curr_waypoint.get_left_lane()
    right_lane_waypoint = curr_waypoint.get_right_lane()

    map_lanes_list = []
    map_lanes_list += get_all_lane_segments(
        curr_waypoint, interval=interval, total_search_depth=search_depth, segment_waypoints_num=segment_waypoints_num)

    if left_lane_waypoint is not None:
        if left_lane_waypoint.lane_type == carla.libcarla.LaneType.Driving:
            map_lanes_list += get_all_lane_segments(
                left_lane_waypoint, interval=interval, total_search_depth=search_depth, segment_waypoints_num=segment_waypoints_num)
            
    if right_lane_waypoint is not None:
        if right_lane_waypoint.lane_type == carla.libcarla.LaneType.Driving:
            map_lanes_list += get_all_lane_segments(
                right_lane_waypoint, interval=interval, total_search_depth=search_depth, segment_waypoints_num=segment_waypoints_num)
        
    return map_lanes_list

def lateral_shift(origin: Union[np.ndarray, List], right_vec: Union[np.ndarray, List], shift: float) -> np.ndarray:
    if isinstance(origin, list):
        origin = np.array(origin)
    if isinstance(right_vec, list):
        right_vec = np.array(right_vec)
    right_vec_normalized = right_vec / np.linalg.norm(right_vec)
    shifted_point = origin + right_vec_normalized * shift
    return shifted_point
    
if __name__ == '__main__':
    import time
    import random
    import matplotlib.pyplot as plt

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town01')
    traffic_manager = client.get_trafficmanager(8000)
    town_map = world.get_map()

    # 生成随机的初始位置
    spawn_points = town_map.get_spawn_points()
    spawn_points_len = len(spawn_points)
    seed = random.randint(0, spawn_points_len-1)
    spawn_point = spawn_points[seed]
    print('spawn_points:', spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)

    distance = 1
    search_depth = 100
    segment_waypoints_num = 20

    ego_location = spawn_point.location
    ego_waypoint = town_map.get_waypoint(ego_location)
    left_lane_waypoint = ego_waypoint.get_left_lane()
    right_lane_waypoint = ego_waypoint.get_right_lane()
    
    t = time.time()
    map_lanes_list = get_map_lanes(ego_waypoint, distance, search_depth, segment_waypoints_num)
    print('Time:', time.time()-t)

    for group in map_lanes_list:
        for lane in group:
            plt.plot(lane[:, 0], lane[:, 1], markersize=0.1)

    plt.plot(ego_waypoint.transform.location.x,ego_waypoint.transform.location.y, 'ro', markersize=5)
    plt.plot(left_lane_waypoint.transform.location.x,left_lane_waypoint.transform.location.y, 'go', markersize=5)
    plt.plot(right_lane_waypoint.transform.location.x,right_lane_waypoint.transform.location.y, 'bo', markersize=5)

    # example_lane = map_lanes_nparray[0][0]
    # lane_width = example_lane[0][6]
    # shift_lane = lateral_shift(
    #     example_lane[:, 0:2], example_lane[:, 7:9], lane_width)
    # plt.plot(shift_lane[:, 0], shift_lane[:, 1], markersize=0.1)

    plt.show()
            
