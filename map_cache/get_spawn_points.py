import carla
import pickle

if __name__ == '__main__':
    MAP_NAME = 'Town01'
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world(MAP_NAME)
    town_map = world.get_map()

    spawn_points = town_map.get_spawn_points()
    spawn_points = [(x.location.x, x.location.y, x.location.z, x.rotation.yaw, x.rotation.pitch, x.rotation.roll) for x in spawn_points]

    with open(f'map_cache/{MAP_NAME}_spawn_points.pkl', 'wb') as f:
        pickle.dump(spawn_points, f)