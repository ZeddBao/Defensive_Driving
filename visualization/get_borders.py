import matplotlib.pyplot as plt
import pickle
import carla

def lateral_shift(waypoint: carla.libcarla.Waypoint, shift: float):
    """Makes a lateral shift of the forward vector of a transform"""
    # waypoint.transform.rotation.yaw += 90
    point = waypoint.transform.location + shift * waypoint.transform.get_right_vector()
    return point.x, point.y

def get_circuit_borders(road_ids: list, roads_dict: dict) -> list:
        borders = []
        for key in road_ids:
            if key < 0:
                segment = roads_dict[key][::-1]
            else:
                segment = roads_dict[key]
            for wp in segment: 
                right = lateral_shift(wp, wp.lane_width / 2)
                borders.append(right)
        return borders

if __name__ == '__main__':

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    carla_map = world.get_map()

    waypoints = carla_map.generate_waypoints(2)
    dict_waypoints = {}
    for waypoint in waypoints:
        right_lane = waypoint.get_right_lane()
        # left_lane = waypoint.get_left_lane()
        if right_lane is not None:
            if right_lane.lane_type == carla.LaneType.Driving:
                continue
        
        direction = 1 if waypoint.lane_id > 0 else -1
        if waypoint.road_id == 0 and waypoint.lane_id < 0:
            dict_waypoints[65536] = dict_waypoints.get(65536, [])
            dict_waypoints[65536].append(waypoint)
        else:
            dict_waypoints[direction * waypoint.road_id] = dict_waypoints.get(direction * waypoint.road_id, [])
            dict_waypoints[direction * waypoint.road_id].append(waypoint)

    fig, ax = plt.subplots()
    for lane_id, waypoints in dict_waypoints.items():
        x = [waypoint.transform.location.x for waypoint in waypoints]
        y = [waypoint.transform.location.y for waypoint in waypoints]
        x_mid = x[len(x) // 2]
        y_mid = y[len(y) // 2]
        ax.scatter(x, y, s=0.01)
        ax.text(x_mid, y_mid, str(lane_id),
                    fontsize=2, color='black', ha='center', va='center',
                    bbox=dict(facecolor='gray', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.1'))

    # Town03
    # keys = [-18, -1251, -29, -28, -1257, 18, 19, 1301, 28, 29, 1275, -19]
    circuit_1 = [2, -783, 1275, 726]
    circuit_2 = [-697, -1251, -406, 631]
    circuit_3 = [622, -365, -1697, 533]
    circuit_4 = [-925,-1608,-25,1691,1301,817]
    circuit_5 = [-1257,1684,1105,-489,449]
    circuit_6 = [399, -1809, -1453, 1713]   # optional
    circuit_7 = [-65, 551, -1181, -5,-1158]
    circuit_8 = [-1214,1772,203,76,77,78,79,80,1073,-1435,-45,-60,-81,-1431,-1374,1142, 46]
    circuit_9 = [1363,-1059,-80,-79,-78,-77,-76,222,-90,-67]
    circuit_10 = [255, -104, 1737, -1661, -1470, 1566, 967]
    circuit_11 = [-1917,980,1868]   # optional
    circuit_12 = [-1846, -918, 823, 20]
    circuit_outer = [-2, 526, 65, 67, 1960, 69, -332, -23, -20, -791]


    cx_borders = [
        get_circuit_borders(circuit_outer, dict_waypoints),
        get_circuit_borders(circuit_1, dict_waypoints),
        get_circuit_borders(circuit_2, dict_waypoints),
        get_circuit_borders(circuit_3, dict_waypoints),
        get_circuit_borders(circuit_4, dict_waypoints),
        get_circuit_borders(circuit_5, dict_waypoints),
        get_circuit_borders(circuit_6, dict_waypoints),
        get_circuit_borders(circuit_7, dict_waypoints),
        get_circuit_borders(circuit_8, dict_waypoints),
        get_circuit_borders(circuit_9, dict_waypoints),
        get_circuit_borders(circuit_10, dict_waypoints),
        get_circuit_borders(circuit_11, dict_waypoints),
        get_circuit_borders(circuit_12, dict_waypoints)
        ]
    
    for borders in cx_borders:
        x = [border[0] for border in borders]
        y = [border[1] for border in borders]
        ax.plot(x, y)

    with open('map_cache/Town03_map_borders.pkl', 'wb') as f:
        pickle.dump(cx_borders, f)

    plt.show()