import matplotlib.pyplot as plt
import pickle
import carla

def lateral_shift(waypoint: carla.libcarla.Waypoint, shift: float):
    """Makes a lateral shift of the forward vector of a transform"""
    # waypoint.transform.rotation.yaw += 90
    point = waypoint.transform.location + shift * waypoint.transform.get_right_vector()
    return point.x, point.y

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town03')
carla_map = world.get_map()

waypoints = carla_map.generate_waypoints(2)
dict_waypoints = {}
for waypoint in waypoints:
    if waypoint.get_right_lane() is not None:
        if waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
            continue
    
    direction = 1 if waypoint.lane_id > 0 else -1
    dict_waypoints[direction * waypoint.road_id] = dict_waypoints.get(direction * waypoint.road_id, [])
    dict_waypoints[direction * waypoint.road_id].append(waypoint)

# fig, ax = plt.subplots()
# for lane_id, waypoints in dict_waypoints.items():
#     x = [waypoint.transform.location.x for waypoint in waypoints]
#     y = [waypoint.transform.location.y for waypoint in waypoints]
#     x_mid = x[len(x) // 2]
#     y_mid = y[len(y) // 2]
#     ax.scatter(x, y, s=0.01)
#     ax.text(x_mid, y_mid, str(lane_id),
#                 fontsize=2, color='black', ha='center', va='center',
#                 bbox=dict(facecolor='gray', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.1'))

# keys = [-18, -1251, -29, -28, -1257, 18, 19, 1301, 28, 29, 1275, -19]
circuit_1 = [-783, -19, 1275, 29, 726, 2]
circuit_2 = [1, -697, 

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

c1_borders = get_circuit_borders(circuit_1, dict_waypoints)

x, y = zip(*c1_borders)
plt.plot(x, y, 'r')

# import pickle
# with open('map_cache/Town03_map_borders.pkl', 'wb') as f:
#     pickle.dump(borders, f)

plt.show()