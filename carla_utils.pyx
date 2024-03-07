import carla
import numpy as np
cimport numpy as cnp
cimport cython

def init():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town01')
    traffic_manager = client.get_trafficmanager(8000)
    town_map = world.get_map()
    return client, world, traffic_manager, town_map

def random_spawn_point():
    import random
    # 生成随机的初始位置
    _, _, _, town_map = init()
    spawn_points = town_map.get_spawn_points()
    spawn_points_len = len(spawn_points)
    seed = random.randint(0, spawn_points_len-1)
    spawn_point = spawn_points[seed]
    print('spawn_points:', spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)
    return spawn_point


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray get_segment_lane(object curr_waypoint, str towards, double interval, int search_depth, int segment_waypoints_num, list breakpoint_stack = []):
    cdef list segment_waypoints_list = []
    cdef object next_waypoint = curr_waypoint
    cdef int i, next_waypoint_list_len
    cdef list next_waypoint_list
    cdef cnp.ndarray last_waypoint
    cdef cnp.ndarray tiles

    for i in range(search_depth):
        if towards == 'next':
            next_waypoint_list = next_waypoint.next(interval)
        elif towards == 'previous':
            next_waypoint_list = next_waypoint.previous(interval)

        next_waypoint_list_len = len(next_waypoint_list)
        if not next_waypoint_list:
            break
        elif next_waypoint_list_len > 1 or i == segment_waypoints_num:
            breakpoint_stack.extend([(wp, search_depth-i) for wp in next_waypoint_list])
            break
        elif next_waypoint_list_len == 1:
            vec_forward = next_waypoint.transform.get_forward_vector()
            vec_right = next_waypoint.transform.get_right_vector()
            lane_width = next_waypoint.lane_width
            segment_waypoints_list.append([
                next_waypoint.transform.location.x, next_waypoint.transform.location.y,
                vec_forward.x, vec_forward.y,
                0,
                int(next_waypoint.is_junction),
                lane_width,
                vec_right.x, vec_right.y
            ])
            next_waypoint = next_waypoint_list[0]

    cdef cnp.ndarray segment_waypoints_array = np.array(segment_waypoints_list, dtype=np.float64)
    cdef int len_waypoints = segment_waypoints_array.shape[0]

    if len_waypoints < segment_waypoints_num and len_waypoints > 0:
        last_waypoint = segment_waypoints_array[-1].reshape(1, -1)
        tiles = np.tile(last_waypoint, (segment_waypoints_num - len_waypoints, 1))
        segment_waypoints_array = np.concatenate((segment_waypoints_array, tiles), axis=0)
    elif len_waypoints == 0:
        segment_waypoints_array = None

    return segment_waypoints_array

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list get_all_segment_lanes(object curr_waypoint, double interval = 1, int total_search_depth = 100, int segment_waypoints_num = 10):
    segment_lanes_list = []
    cdef list breakpoint_stack = []
    breakpoint_stack.append((curr_waypoint, total_search_depth))
    cdef tuple wp_depth_pair
    cdef object start_waypoint
    cdef int num
    cdef cnp.ndarray lane_waypoints

    while breakpoint_stack:
        wp_depth_pair = breakpoint_stack.pop()
        start_waypoint, num = wp_depth_pair
        lane_waypoints = get_segment_lane(start_waypoint, 'next', interval, num, segment_waypoints_num, breakpoint_stack)
        if lane_waypoints is not None:
            segment_lanes_list.append(lane_waypoints)

    breakpoint_stack.append((curr_waypoint, total_search_depth))
    while breakpoint_stack:
        wp_depth_pair = breakpoint_stack.pop()
        start_waypoint, num = wp_depth_pair
        lane_waypoints = get_segment_lane(start_waypoint, 'previous', interval, num, segment_waypoints_num, breakpoint_stack)
        if lane_waypoints is not None:
            segment_lanes_list.append(lane_waypoints)

    return segment_lanes_list