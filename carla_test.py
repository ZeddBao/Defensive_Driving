import random
import logging
import pygame
import carla

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    # 如果过滤器只返回一个蓝图，我们假设这个蓝图是必需的，因此我们忽略生成
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        # 检查生成是否在可用生成中
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# Connect to the server
# 连接服务器
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')
traffic_manager = client.get_trafficmanager(8000)
town_map = world.get_map()

# Get all the blueprints of the actors
# 获取所有actor的蓝图
# blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')

# Spawn the vehicles
# 生成车辆
# SpawnActor = carla.command.SpawnActor
# SetAutopilot = carla.command.SetAutopilot
# FutureActor = carla.command.FutureActor

# batch = []
# vehicles_list = []
# hero = True
# synchronous_master = False
# spawn_points = world.get_map().get_spawn_points()
# for n, transform in enumerate(spawn_points):
#     if n >= 100:
#         break
#     blueprint = random.choice(blueprints)
#     if blueprint.has_attribute('color'):
#         color = random.choice(blueprint.get_attribute('color').recommended_values)
#         blueprint.set_attribute('color', color)
#     if blueprint.has_attribute('driver_id'):
#         driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
#         blueprint.set_attribute('driver_id', driver_id)
#     if hero:
#         blueprint.set_attribute('role_name', 'hero')
#         hero = False
#     else:
#         blueprint.set_attribute('role_name', 'autopilot')

#     batch.append(SpawnActor(blueprint, transform)
#                 .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
    
# for response in client.apply_batch_sync(batch, synchronous_master):
#     if response.error:
#         logging.error(response.error)
#     else:
#         vehicles_list.append(response.actor_id)

ego_location= carla.Location(x=0, y=0, z=0)
init_waypoint = town_map.get_waypoint(ego_location, project_to_road=True)

print('init_waypoint:', init_waypoint.transform.location.x, init_waypoint.transform.location.y, init_waypoint.transform.location.z)

left_waypoint = init_waypoint.get_left_lane()
right_waypoint = init_waypoint.get_right_lane()
print('left_waypoint:', left_waypoint.transform.location.x, left_waypoint.transform.location.y, left_waypoint.transform.location.z)
print('right_waypoint:', right_waypoint.transform.location.x, right_waypoint.transform.location.y, right_waypoint.transform.location.z)