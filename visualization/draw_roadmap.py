import math
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import carla

COLOR_BUTTER_0 = (252/255, 233/255, 79/255)
COLOR_BUTTER_1 = (237/255, 212/255, 0/255)
COLOR_BUTTER_2 = (196/255, 160/255, 0/255)

COLOR_ORANGE_0 = (252/255, 175/255, 62/255)
COLOR_ORANGE_1 = (245/255, 121/255, 0/255)
COLOR_ORANGE_2 = (209/255, 92/255, 0/255)

COLOR_CHOCOLATE_0 = (233/255, 185/255, 110/255)
COLOR_CHOCOLATE_1 = (193/255, 125/255, 17/255)
COLOR_CHOCOLATE_2 = (143/255, 89/255, 2/255)

COLOR_CHAMELEON_0 = (138/255, 226/255, 52/255)
COLOR_CHAMELEON_1 = (115/255, 210/255, 22/255)
COLOR_CHAMELEON_2 = (78/255, 154/255, 6/255)

COLOR_SKY_BLUE_0 = (114/255, 159/255, 207/255)
COLOR_SKY_BLUE_1 = (52/255, 101/255, 164/255)
COLOR_SKY_BLUE_2 = (32/255, 74/255, 135/255)

COLOR_PLUM_0 = (173/255, 127/255, 168/255)
COLOR_PLUM_1 = (117/255, 80/255, 123/255)
COLOR_PLUM_2 = (92/255, 53/255, 102/255)

COLOR_SCARLET_RED_0 = (239/255, 41/255, 41/255)
COLOR_SCARLET_RED_1 = (204/255, 0/255, 0/255)
COLOR_SCARLET_RED_2 = (164/255, 0/255, 0/255)

COLOR_ALUMINIUM_0 = (238/255, 238/255, 236/255)
COLOR_ALUMINIUM_1 = (211/255, 215/255, 207/255)
COLOR_ALUMINIUM_2 = (186/255, 189/255, 182/255)
COLOR_ALUMINIUM_3 = (136/255, 138/255, 133/255)
COLOR_ALUMINIUM_4 = (85/255, 87/255, 83/255)
COLOR_ALUMINIUM_4_5 = (66/255, 62/255, 64/255)
COLOR_ALUMINIUM_5 = (46/255, 52/255, 54/255)


COLOR_WHITE = (1, 1, 1)
COLOR_BLACK = (0, 0, 0)

class Util(object):

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        """Function that renders the all the source surfaces in a destination source"""
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        """Returns the length of a vector"""
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        """Gets the bounding box corners of an actor in world space"""
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners

def draw_road_map(carla_map, precision = 1, show_triggers = 1, show_spawn_points = 1, show_connections = 1):
    """Draws all the roads, including lane markings, arrows and traffic signs"""
    fig, ax = plt.subplots()
    ax.set_facecolor(COLOR_WHITE)

    def lane_marking_color_to_tango(lane_marking_color):
        """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
        tango_color = COLOR_BLACK

        if lane_marking_color == carla.LaneMarkingColor.White:
            tango_color = COLOR_ALUMINIUM_2

        elif lane_marking_color == carla.LaneMarkingColor.Blue:
            tango_color = COLOR_SKY_BLUE_0

        elif lane_marking_color == carla.LaneMarkingColor.Green:
            tango_color = COLOR_CHAMELEON_0

        elif lane_marking_color == carla.LaneMarkingColor.Red:
            tango_color = COLOR_SCARLET_RED_0

        elif lane_marking_color == carla.LaneMarkingColor.Yellow:
            tango_color = COLOR_ORANGE_0

        return tango_color

    def draw_solid_line(color: Union[Tuple[float, ...], str], closed: bool, points: List[Tuple[float, float]], linewidth: float=0.1):
        """Draws solid lines in a surface given a set of points, width and color"""
        if len(points) >= 2:
            if closed:
                points.append(points[0])
            ax.plot(*zip(*points), color=color, linewidth=linewidth)

    def draw_broken_line(color: Union[Tuple[float, ...], str], closed: bool, points: List[Tuple[float, float]], linewidth: float=0.1):
        """Draws broken lines in a surface given a set of points, width and color"""
        # Select which lines are going to be rendered from the set of lines
        if closed:
            points.append(points[0])
        ax.plot(*zip(*points), color=color, linewidth=linewidth, linestyle='dashed')

    def get_lane_markings(lane_marking_type, lane_marking_color, waypoints: List[carla.libcarla.Waypoint], sign):
        """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
            as a combination of Broken and Solid lines
            对于多个车道标线类型（SolidSolid，BrokenSolid，SolidBroken和BrokenBroken），它将它们转换为破碎和实线的组合"""
        
        margin = 0.25
        marking_1 = [lateral_shift(w, sign * w.lane_width * 0.5) for w in waypoints]
        if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
            return [(lane_marking_type, lane_marking_color, marking_1)]
        else:
            marking_2 = [lateral_shift(w, sign * (w.lane_width * 0.5 + margin * 2)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

        return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

    def draw_lane(lane, color):
        """Renders a single lane in a surface and with a specified color"""
        for side in lane:
            lane_left_side = [lateral_shift(w, -w.lane_width * 0.5) for w in side]
            lane_right_side = [lateral_shift(w, w.lane_width * 0.5) for w in side]

            polygon = lane_left_side + lane_right_side[::-1]

            if len(polygon) > 2:
                # ax.fill(*zip(*polygon), color=color, edgecolor=color)
                ax.plot(*zip(*lane_left_side), color=color, linewidth=1)
                ax.plot(*zip(*lane_right_side), color=color, linewidth=1)

    def draw_lane_marking(waypoints):
        """Draws the left and right side of lane markings"""
        # Left Side
        draw_lane_marking_single_side(waypoints[0], -1)

        # Right Side
        draw_lane_marking_single_side(waypoints[1], 1)

    def draw_lane_marking_single_side(waypoints, sign):    # TODO: draw_lane_marking_single_side
        """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
        the waypoint based on the sign parameter"""
        lane_marking = None

        marking_type = carla.LaneMarkingType.NONE
        previous_marking_type = carla.LaneMarkingType.NONE

        marking_color = carla.LaneMarkingColor.Other
        previous_marking_color = carla.LaneMarkingColor.Other

        markings_list = []
        temp_waypoints = []
        current_lane_marking = carla.LaneMarkingType.NONE
        for sample in waypoints:
            lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

            if lane_marking is None:
                continue

            marking_type = lane_marking.type
            marking_color = lane_marking.color

            if current_lane_marking != marking_type:
                # Get the list of lane markings to draw
                markings = get_lane_markings(
                    previous_marking_type,
                    lane_marking_color_to_tango(previous_marking_color),
                    temp_waypoints,
                    sign)
                current_lane_marking = marking_type

                # Append each lane marking in the list
                for marking in markings:
                    markings_list.append(marking)

                temp_waypoints = temp_waypoints[-1:]

            else:
                temp_waypoints.append((sample))
                previous_marking_type = marking_type
                previous_marking_color = marking_color

        # Add last marking
        last_markings = get_lane_markings(
            previous_marking_type,
            lane_marking_color_to_tango(previous_marking_color),
            temp_waypoints,
            sign)
        for marking in last_markings:
            markings_list.append(marking)

        # Once the lane markings have been simplified to Solid or Broken lines, we draw them
        for markings in markings_list:
            if markings[0] == carla.LaneMarkingType.Solid:
                draw_solid_line(markings[1], False, markings[2], 1)
            elif markings[0] == carla.LaneMarkingType.Broken:
                draw_broken_line(markings[1], False, markings[2], 1)

    def draw_arrow(waypoint, color=COLOR_ALUMINIUM_2):
        """ Draws an arrow with a specified color given a transform"""
        waypoint.transform.rotation.yaw += 180
        forward = waypoint.transform.get_forward_vector()
        waypoint.transform.rotation.yaw += 90
        right_dir = waypoint.transform.get_forward_vector()
        end = waypoint.transform.location
        start = end - 2.0 * forward
        right = start + 0.8 * forward + 0.4 * right_dir
        left = start + 0.8 * forward - 0.4 * right_dir

        # Draw lines
        ax.plot([start.x, end.x], [start.y, end.y], color=color, linewidth=1)
        ax.plot([left.x, start.x, end.x], [left.y, start.y, end.y], color=color, linewidth=1)

    # def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
    #     """Draw stop traffic signs and its bounding box if enabled"""
    #     transform = actor.get_transform()
    #     waypoint = carla_map.get_waypoint(transform.location)

    #     angle = -waypoint.transform.rotation.yaw - 90.0
    #     font_surface = pygame.transform.rotate(font_surface, angle)
    #     pixel_pos = waypoint.transform.location
    #     offset = font_surface.get_rect(center=(pixel_pos.x, pixel_pos.y))
    #     surface.blit(font_surface, offset)

    #     # Draw line in front of stop
    #     forward_vector = carla.Location(waypoint.transform.get_forward_vector())
    #     left_vector = carla.Location(-forward_vector.y, forward_vector.x,
    #                                     forward_vector.z) * waypoint.lane_width / 2 * 0.7

    #     line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
    #             (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

    #     line_pixel = [(l.x, l.y) for l in line]
    #     pygame.draw.lines(surface, color, True, line_pixel, 2)

    #     # Draw bounding box of the stop trigger
    #     if show_triggers:
    #         corners = Util.get_bounding_box(actor)
    #         corners = [(corner.x, corner.y) for corner in corners]
    #         pygame.draw.lines(surface, trigger_color, True, corners, 2)

    # def draw_crosswalk(surface, transform=None, color=COLOR_ALUMINIUM_2):
    #     """Given two points A and B, draw white parallel lines from A to B"""
    #     a = carla.Location(0.0, 0.0, 0.0)
    #     b = carla.Location(10.0, 10.0, 0.0)

    #     ab = b - a
    #     length_ab = math.sqrt(ab.x**2 + ab.y**2)
    #     unit_ab = ab / length_ab
    #     unit_perp_ab = carla.Location(-unit_ab.y, unit_ab.x, 0.0)

    #     # Crosswalk lines params
    #     space_between_lines = 0.5
    #     line_width = 0.7
    #     line_height = 2

    #     current_length = 0
    #     while current_length < length_ab:

    #         center = a + unit_ab * current_length

    #         width_offset = unit_ab * line_width
    #         height_offset = unit_perp_ab * line_height
    #         list_point = [center - width_offset - height_offset,
    #                       center + width_offset - height_offset,
    #                       center + width_offset + height_offset,
    #                       center - width_offset + height_offset]

    #         list_point = [world_to_pixel(p) for p in list_point]
    #         pygame.draw.polygon(surface, color, list_point)
    #         current_length += (line_width + space_between_lines) * 2

    def lateral_shift(waypoint: carla.libcarla.Waypoint, shift: float) -> Tuple[float, float]:
        """Makes a lateral shift of the forward vector of a transform"""
        waypoint.transform.rotation.yaw += 90
        point = waypoint.transform.location + shift * waypoint.transform.get_forward_vector()
        return point.x, point.y

    def draw_topology(carla_topology, index):
        """ Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
        topology = [x[index] for x in carla_topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]

            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

            # Draw Shoulders, Parkings and Sidewalks
            PARKING_COLOR = COLOR_ALUMINIUM_4_5
            SHOULDER_COLOR = COLOR_ALUMINIUM_5
            SIDEWALK_COLOR = COLOR_ALUMINIUM_3

            shoulder = [[], []]
            parking = [[], []]
            sidewalk = [[], []]

            for w in waypoints: # TODO: get_lane
                # Classify lane types until there are no waypoints by going left
                l = w.get_left_lane()
                while l and l.lane_type != carla.LaneType.Driving:

                    if l.lane_type == carla.LaneType.Shoulder:
                        shoulder[0].append(l)

                    if l.lane_type == carla.LaneType.Parking:
                        parking[0].append(l)

                    if l.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[0].append(l)

                    l = l.get_left_lane()

                # Classify lane types until there are no waypoints by going right
                r = w.get_right_lane()
                while r and r.lane_type != carla.LaneType.Driving:

                    if r.lane_type == carla.LaneType.Shoulder:
                        shoulder[1].append(r)

                    if r.lane_type == carla.LaneType.Parking:
                        parking[1].append(r)

                    if r.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[1].append(r)

                    r = r.get_right_lane()

            # Draw classified lane types
            draw_lane(shoulder, SHOULDER_COLOR)
            draw_lane(parking, PARKING_COLOR)
            draw_lane(sidewalk, SIDEWALK_COLOR)

        # Draw Roads    # TODO: draw roads
        for waypoints in set_waypoints:
            waypoint = waypoints[0]
            road_left_side = [lateral_shift(w, -w.lane_width * 0.5) for w in waypoints]
            road_right_side = [lateral_shift(w, w.lane_width * 0.5) for w in waypoints]

            polygon = road_left_side + road_right_side[::-1]

            if len(polygon) > 2:
                # ax.fill(*zip(*polygon), color=COLOR_ALUMINIUM_5, edgecolor=COLOR_ALUMINIUM_5)
                ax.plot(*zip(*road_left_side), color=COLOR_ALUMINIUM_5, linewidth=1)
                ax.plot(*zip(*road_right_side), color=COLOR_ALUMINIUM_5, linewidth=1)

            # Draw Lane Markings and Arrows
            if not waypoint.is_junction:
                draw_lane_marking([waypoints, waypoints])
                for n, wp in enumerate(waypoints):
                    if ((n + 1) % 400) == 0:
                        draw_arrow(wp)

    topology = carla_map.get_topology()
    draw_topology(topology, 0)  # TODO: draw_topology

    if show_spawn_points:
        for i, sp in enumerate(carla_map.get_spawn_points()):
            # 用数字标记生成点，填充文字背景
            ax.text(sp.location.x, sp.location.y, str(i),
                fontsize=5, color=COLOR_BLACK, ha='center', va='center',
                bbox=dict(facecolor=COLOR_ALUMINIUM_0, alpha=0.5, edgecolor=COLOR_ALUMINIUM_0, boxstyle='round,pad=0.1'))

    if show_connections:
        dist = 1.5

        def to_pixel(wp): return wp.transform.location
        for wp in carla_map.generate_waypoints(dist):
            col = (0, 1, 1) if wp.is_junction else (0, 1, 0)
            wp_start = to_pixel(wp)

            for nxt in wp.next(dist):
                wp_nxt = to_pixel(nxt)
                ax.plot([wp_start.x, wp_nxt.x], [wp_start.y, wp_nxt.y], color=col, linewidth=0.1)
            if wp.lane_change & carla.LaneChange.Right:
                r = wp.get_right_lane()
                if r is not None:
                    wp_r = to_pixel(r)
                if r and r.lane_type == carla.LaneType.Driving:
                    ax.plot([wp_start.x, wp_r.x], [wp_start.y, wp_r.y], color=col, linewidth=0.1)
            if wp.lane_change & carla.LaneChange.Left:
                l = wp.get_left_lane()
                if l is not None:
                    wp_l = to_pixel(l)
                if l and l.lane_type == carla.LaneType.Driving:
                    ax.plot([wp_start.x, wp_l.x], [wp_start.y, wp_l.y], color=col, linewidth=0.1)

    # actors = carla_world.get_actors()

    # Find and Draw Traffic Signs: Stops and Yields
    # font_size = 1
    # font = pygame.font.SysFont('Arial', font_size, True)

    # stops = [actor for actor in actors if 'stop' in actor.type_id]
    # yields = [actor for actor in actors if 'yield' in actor.type_id]

    # stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
    # stop_font_surface = pygame.transform.scale(
    #     stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

    # yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
    # yield_font_surface = pygame.transform.scale(
    #     yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

    # for ts_stop in stops:
    #     draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

    # for ts_yield in yields:
    #     draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    carla_map = world.get_map()
    draw_road_map(carla_map)