"""
Welcome to CARLA No-Rendering Mode Visualizer

    TAB          : toggle hero mode
    Mouse Wheel  : zoom in / zoom out
    Mouse Drag   : move map (map mode only)

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    F1           : toggle HUD
    I            : toggle actor ids
    H/?          : toggle help
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import signal
import subprocess
import socket

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import TrafficLightState as tls

import argparse
import logging
import datetime
import weakref
import math
import random
import hashlib

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

# Module Defines
TITLE_WORLD = 'WORLD'
TITLE_HUD = 'HUD'
TITLE_INPUT = 'INPUT'

PIXELS_PER_METER = 12

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 150

# ==============================================================================
# -- Util -----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


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
        bb = actor.trigger_volume.extent    # bb为actor的触发体积的半径
        corners = [carla.Location(x=-bb.x, y=-bb.y),    # 触发体称的8个顶点
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]  # 触发体积的8个顶点的世界坐标
        t = actor.get_transform()   # 获取actor的transform
        t.transform(corners)     # 将8个顶点的世界坐标转换为actor的坐标系下的坐标
        return corners  # 返回actor的8个顶点的坐标

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """Renders texts that fades out after some seconds that the user specifies"""

    def __init__(self, font, dim, pos):
        """Initializes variables such as text font, dimensions and position"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=COLOR_WHITE, seconds=2.0):
        """Sets the text, color and seconds until fade out"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill(COLOR_BLACK)
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        """Each frame, it shows the displayed text for some specified seconds, if any"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)   # 设置透明度

    def render(self, display):
        """ Renders the text in its surface and its position"""
        display.blit(self.surface, self.pos)    # 将surface渲染到display上


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        """Renders the help text that shows the controls for using no rendering mode"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill(COLOR_BLACK)
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, COLOR_WHITE)
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggles display of help text"""
        self._render = not self._render

    def render(self, display):
        """Renders the help text, if enabled"""
        if self._render:
            display.blit(self.surface, self.pos)    # 将surface渲染到display上，blit函数是将一个surface渲染到另一个surface上


# ==============================================================================
# -- HUD -----------------------------------------------------------------
# ==============================================================================


class HUD (object): # HUD: Head-Up Display 俯视显示
    """Class encharged of rendering the HUD that displays information about the world and the hero vehicle"""

    def __init__(self, name, width, height):
        """Initializes default HUD params and content data parameters that will be displayed"""
        self.name = name
        self.dim = (width, height)
        self._init_hud_params()
        self._init_data_params()

    def start(self):
        """Does nothing since it does not need to use other modules"""

    def _init_hud_params(self): # 初始化HUD参数
        """Initialized visual parameters such as font text and size"""
        font_name = 'courier' if os.name == 'nt' else 'mono'    # 如果是windows系统，使用courier字体，否则使用mono字体
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]  # 获取所有的字体
        default_font = 'ubuntumono' # 默认字体
        mono = default_font if default_font in fonts else fonts[0]  # 如果默认字体在fonts中，使用默认字体，否则使用fonts中的第一个字体
        mono = pygame.font.match_font(mono) # 获取字体的路径
        self._font_mono = pygame.font.Font(mono, 14)    # 设置字体大小
        self._header_font = pygame.font.SysFont('Arial', 14, True)  # 设置字体大小
        self.help = HelpText(pygame.font.Font(mono, 24), *self.dim) # 设置帮助文本的字体和大小
        self._notifications = FadingText(   # 初始化FadingText类
            pygame.font.Font(pygame.font.get_default_font(), 20),
            (self.dim[0], 40), (0, self.dim[1] - 40))

    def _init_data_params(self):    # 初始化数据参数
        """Initializes the content data structures"""
        self.show_info = True   # 是否显示信息
        self.show_actor_ids = False # 是否显示actor的id
        self._info_text = {}    # 信息文本

    def notification(self, text, seconds=2.0):
        """Shows fading texts for some specified seconds"""
        self._notifications.set_text(text, seconds=seconds)

    def tick(self, clock):
        """Updated the fading texts each frame"""
        self._notifications.tick(clock)

    def add_info(self, title, info):
        """Adds a block of information in the left HUD panel of the visualizer"""
        self._info_text[title] = info

    def render_vehicles_ids(self, vehicle_id_surface, list_actors, world_to_pixel, hero_actor, hero_transform):
        """When flag enabled, it shows the IDs of the vehicles that are spawned in the world. Depending on the vehicle type,
        it will render it in different colors"""

        vehicle_id_surface.fill(COLOR_BLACK)    # 填充surface为黑色
        if self.show_actor_ids: # 如果显示actor的id
            vehicle_id_surface.set_alpha(150)   # 设置透明度
            for actor in list_actors:   # 遍历所有的actor
                x, y = world_to_pixel(actor[1].location)    # 获取actor的位置

                angle = 0   # 设置角度为0
                if hero_actor is not None:  # 如果hero_actor不为空
                    angle = -hero_transform.rotation.yaw - 90   # 设置角度为-hero_actor的yaw角度-90

                color = COLOR_SKY_BLUE_0    # 设置颜色为天蓝色
                if int(actor[0].attributes['number_of_wheels']) == 2:   # 如果actor的轮子数为2
                    color = COLOR_CHOCOLATE_0   # 设置颜色为巧克力色
                if actor[0].attributes['role_name'] == 'hero':  # 如果actor的role_name为hero
                    color = COLOR_CHAMELEON_0   # 设置颜色为变色龙色

                font_surface = self._header_font.render(str(actor[0].id), True, color)  # 设置字体颜色
                rotated_font_surface = pygame.transform.rotate(font_surface, angle)  # 旋转字体
                rect = rotated_font_surface.get_rect(center=(x, y))  # 获取字体的矩形
                vehicle_id_surface.blit(rotated_font_surface, rect) # 将字体渲染到surface上

        return vehicle_id_surface

    def render(self, display):
        """If flag enabled, it renders all the information regarding the left panel of the visualizer"""
        if self.show_info:  # 如果显示信息
            info_surface = pygame.Surface((240, self.dim[1]))   # 设置surface的大小
            info_surface.set_alpha(100) # 设置透明度
            display.blit(info_surface, (0, 0))  # 将surface渲染到display上
            v_offset = 4    # 垂直偏移
            bar_h_offset = 100  # 水平偏移
            bar_width = 106 # 柱状图的宽度
            i = 0   # i初始化为0
            for title, info in self._info_text.items(): # 遍历所有的信息
                if not info:    # 如果info为空
                    continue    # 跳过
                surface = self._header_font.render(title, True, COLOR_ALUMINIUM_0).convert_alpha()  # 设置字体颜色
                display.blit(surface, (8 + bar_width / 2, 18 * i + v_offset))   # 将字体渲染到display上
                v_offset += 12  # 垂直偏移加12
                i += 1  # i加1
                for item in info:   # 遍历info
                    if v_offset + 18 > self.dim[1]: # 如果垂直偏移加18大于self.dim[1]
                        break   # 跳出循环
                    if isinstance(item, list):  # 如果item是list
                        if len(item) > 1:   # 如果item的长度大于1
                            points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]   # 设置点的坐标
                            pygame.draw.lines(display, (255, 136, 0), False, points, 2) # 画线
                        item = None # item设置为None
                    elif isinstance(item, tuple):   # 如果item是tuple
                        if isinstance(item[1], bool):   # 如果item[1]是bool类型
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))    # 设置矩形
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect, 0 if item[1] else 1) # 画矩形
                        else:
                            rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6)) # 设置矩形
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect_border, 1)    # 画矩形
                            f = (item[1] - item[2]) / (item[3] - item[2])   # 计算f
                            if item[2] < 0.0:   # 如果item[2]小于0
                                rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))  # 设置矩形
                            else:
                                rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))    # 设置矩形
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect)  # 画矩形
                        item = item[0]  # item设置为item[0]
                    if item:  # At this point has to be a str.  # 如果item不为空
                        surface = self._font_mono.render(item, True, COLOR_ALUMINIUM_0).convert_alpha() # 设置字体颜色
                        display.blit(surface, (8, 18 * i + v_offset))   # 将字体渲染到display上
                    v_offset += 18  # 垂直偏移加18
                v_offset += 24  # 垂直偏移加24
        self._notifications.render(display) # 渲染_fading_text
        self.help.render(display)   # 渲染help文本                                                                                                                                                                                                                                                                                                                                     


# ==============================================================================
# -- TrafficLightSurfaces ------------------------------------------------------
# ==============================================================================


class TrafficLightSurfaces(object):
    """Holds the surfaces (scaled and rotated) for painting traffic lights"""

    def __init__(self):
        def make_surface(tl):
            """Draws a traffic light, which is composed of a dark background surface with 3 circles that indicate its color depending on the state"""
            w = 40  # 信号灯的宽度
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)   # 设置surface的大小
            surface.fill(COLOR_ALUMINIUM_5 if tl != 'h' else COLOR_ORANGE_2)    # 填充surface的颜色
            if tl != 'h':   # 如果traffic light不是'h'
                hw = int(w / 2) # 计算w的一半
                off = COLOR_ALUMINIUM_4  # 设置颜色
                red = COLOR_SCARLET_RED_0   # 设置颜色
                yellow = COLOR_BUTTER_0 # 设置颜色
                green = COLOR_CHAMELEON_0   # 设置颜色

                # Draws the corresponding color if is on, otherwise it will be gray if its off
                # 如果灯亮了，就画对应的颜色，否则画灰色
                pygame.draw.circle(surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w))  # tls为carla中的TrafficLightState
                pygame.draw.circle(surface, yellow if tl == tls.Yellow else off, (hw, w + hw), int(0.4 * w))
                pygame.draw.circle(surface, green if tl == tls.Green else off, (hw, 2 * w + hw), int(0.4 * w))

            return pygame.transform.smoothscale(surface, (15, 45) if tl != 'h' else (19, 49))   # 缩放surface

        self._original_surfaces = { # 信号灯的颜色
            'h': make_surface('h'),
            tls.Red: make_surface(tls.Red), # 红色
            tls.Yellow: make_surface(tls.Yellow),   # 黄色
            tls.Green: make_surface(tls.Green), # 绿色
            tls.Off: make_surface(tls.Off), # 关闭
            tls.Unknown: make_surface(tls.Unknown)  # 未知
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):   # 旋转和缩放
        """Rotates and scales the traffic light surface"""
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class MapImage(object):
    """Class encharged of rendering a 2D image from top view of a carla world. Please note that a cache system is used, so if the OpenDrive content
    of a Carla town has not changed, it will read and use the stored image if it was rendered in a previous execution"""
    """负责从carla世界的俯视图中渲染2D图像的类。 
    请注意，使用了缓存系统，因此如果Carla镇的OpenDrive内容没有更改，则它将读取并使用存储的图像，如果它在以前的执行中被渲染"""

    def __init__(self, carla_world, carla_map, pixels_per_meter, show_triggers, show_connections, show_spawn_points):
        """ Renders the map image generated based on the world, its map and additional flags that provide extra information about the road network"""
        self._pixels_per_meter = pixels_per_meter   # 每米的像素数
        self.scale = 1.0    # 缩放
        self.show_triggers = show_triggers  # 是否显示触发器
        self.show_connections = show_connections    # 是否显示连接
        self.show_spawn_points = show_spawn_points  

        waypoints = carla_map.generate_waypoints(2) # 生成waypoints
        margin = 50 # 边距
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin  # 最大x
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin  # 最大y
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin  # 最小x
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin  # 最小y

        self.width = max(max_x - min_x, max_y - min_y)  # 宽度
        self._world_offset = (min_x, min_y)   # 世界偏移

        # Maximum size of a Pygame surface
        width_in_pixels = (1 << 14) - 1 # 2^14-1

        # Adapt Pixels per meter to make world fit in surface
        surface_pixel_per_meter = int(width_in_pixels / self.width) # surface每米的像素数
        if surface_pixel_per_meter > PIXELS_PER_METER:  # 如果surface每米的像素数大于PIXELS_PER_METER
            surface_pixel_per_meter = PIXELS_PER_METER  # 设置surface每米的像素数为PIXELS_PER_METER

        self._pixels_per_meter = surface_pixel_per_meter    # 设置每米的像素数
        width_in_pixels = int(self._pixels_per_meter * self.width)  # 宽度

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert() # 设置surface的大小

        # Load OpenDrive content
        opendrive_content = carla_map.to_opendrive()    # 获取opendrive内容

        # Get hash based on content
        hash_func = hashlib.sha1()  # 创建hash对象
        hash_func.update(opendrive_content.encode("UTF-8")) # 更新hash对象
        opendrive_hash = str(hash_func.hexdigest())  # 获取hash值

        # Build path for saving or loading the cached rendered map
        # 生成用于保存或加载缓存渲染地图的路径
        filename = carla_map.name.split('/')[-1] + "_" + opendrive_hash + ".tga"    # 文件名
        dirname = os.path.join("cache", "no_rendering_mode")    # 文件夹名
        full_path = str(os.path.join(dirname, filename))    # 完整路径

        if os.path.isfile(full_path):   # 如果文件存在
            # Load Image
            self.big_map_surface = pygame.image.load(full_path)  # 加载图片
        else:
            # Render map
            self.draw_road_map( # 渲染地图
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width)

            # If folders path does not exist, create it
            if not os.path.exists(dirname): # 如果文件夹不存在
                os.makedirs(dirname)    # 创建文件夹

            # Remove files if selected town had a previous version saved
            list_filenames = glob.glob(os.path.join(dirname, carla_map.name) + "*")  # 获取文件夹下的所有文件
            for town_filename in list_filenames:    # 遍历所有的文件
                os.remove(town_filename)    # 删除文件

            # Save rendered map for next executions of same map
            pygame.image.save(self.big_map_surface, full_path)  # 保存图片

        self.surface = self.big_map_surface # 设置surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        """Draws all the roads, including lane markings, arrows and traffic signs"""
        map_surface.fill(COLOR_ALUMINIUM_4) # 填充surface为灰色
        precision = 0.05    # 精度

        def lane_marking_color_to_tango(lane_marking_color):    # 将车道标记颜色转换为tango颜色
            """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
            tango_color = COLOR_BLACK   # 设置颜色为黑色

            if lane_marking_color == carla.LaneMarkingColor.White:  # 如果车道标记颜色为白色
                tango_color = COLOR_ALUMINIUM_2  # 设置颜色为铝色

            elif lane_marking_color == carla.LaneMarkingColor.Blue:   # 如果车道标记颜色为蓝色
                tango_color = COLOR_SKY_BLUE_0  # 设置颜色为天蓝色

            elif lane_marking_color == carla.LaneMarkingColor.Green:    # 如果车道标记颜色为绿色
                tango_color = COLOR_CHAMELEON_0   # 设置颜色为变色龙色

            elif lane_marking_color == carla.LaneMarkingColor.Red:  # 如果车道标记颜色为红色
                tango_color = COLOR_SCARLET_RED_0   # 设置颜色为猩红色

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:   # 如果车道标记颜色为黄色
                tango_color = COLOR_ORANGE_0    # 设置颜色为橙色

            return tango_color  # 返回颜色

        def draw_solid_line(surface, color, closed, points, width):   # 画实线
            """Draws solid lines in a surface given a set of points, width and color"""
            if len(points) >= 2:    # 如果点的数量大于等于2
                pygame.draw.lines(surface, color, closed, points, width)    # 画线

        def draw_broken_line(surface, color, closed, points, width):    # 画虚线
            """Draws broken lines in a surface given a set of points, width and color"""
            # Select which lines are going to be rendered from the set of lines
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]  # 选择要渲染的线

            # Draw selected lines
            for line in broken_lines:   # 遍历所有的线
                pygame.draw.lines(surface, color, closed, line, width)  # 画线

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
             as a combination of Broken and Solid lines"""
            margin = 0.25   # 边距
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]  # 获取车道标记的像素坐标
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid): # 如果车道标记类型为Broken或Solid
                return [(lane_marking_type, lane_marking_color, marking_1)] # 返回车道标记类型和颜色
            else:
                marking_2 = [world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]    # 获取车道标记的像素坐标
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:  # 如果车道标记类型为SolidBroken
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),  # 返回车道标记类型和颜色
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]   # 返回车道标记类型和颜色
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]   # 返回NONE

        def draw_lane(surface, lane, color):
            """Renders a single lane in a surface and with a specified color"""
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(surface, color, polygon, 5)
                    pygame.draw.polygon(surface, color, polygon)

        def draw_lane_marking(surface, waypoints):
            """Draws the left and right side of lane markings"""
            # Left Side
            draw_lane_marking_single_side(surface, waypoints[0], -1)

            # Right Side
            draw_lane_marking_single_side(surface, waypoints[1], 1)

        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
            the waypoint based on the sign parameter"""
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE   # 车道标记类型
            previous_marking_type = carla.LaneMarkingType.NONE  # 前一个车道标记类型

            marking_color = carla.LaneMarkingColor.Other    # 车道标记颜色
            previous_marking_color = carla.LaneMarkingColor.Other   # 前一个车道标记颜色

            markings_list = []  # 车道标记列表
            temp_waypoints = [] # 临时waypoints
            current_lane_marking = carla.LaneMarkingType.NONE   # 当前车道标记类型
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking  # 获取车道标记

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type    # 车道标记类型
                marking_color = lane_marking.color  # 车道标记颜色

                if current_lane_marking != marking_type:    # 如果当前车道标记类型不等于车道标记类型
                    # Get the list of lane markings to draw
                    markings = get_lane_markings(   # 获取车道标记
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type # 设置当前车道标记类型

                    # Append each lane marking in the list
                    for marking in markings:    # 遍历所有的车道标记
                        markings_list.append(marking)   # 添加车道标记

                    temp_waypoints = temp_waypoints[-1:]    # 获取最后一个waypoints

                else:
                    temp_waypoints.append((sample)) # 添加waypoints
                    previous_marking_type = marking_type    # 设置前一个车道标记类型
                    previous_marking_color = marking_color  # 设置前一个车道标记颜色

            # Add last marking
            last_markings = get_lane_markings(  # 获取车道标记
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                markings_list.append(marking)

            # Once the lane markings have been simplified to Solid or Broken lines, we draw them
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:  # 如果车道标记类型为Solid
                    draw_solid_line(surface, markings[1], False, markings[2], 2)    # 画实线
                elif markings[0] == carla.LaneMarkingType.Broken:   # 如果车道标记类型为Broken
                    draw_broken_line(surface, markings[1], False, markings[2], 2)   # 画虚线

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):    # 画箭头
            """ Draws an arrow with a specified color given a transform"""
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir

            # Draw lines
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Draw stop traffic signs and its bounding box if enabled"""
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

            # Draw bounding box of the stop trigger
            if self.show_triggers:
                corners = Util.get_bounding_box(actor)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, trigger_color, True, corners, 2)

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

        def lateral_shift(transform, shift):    # 横向移动
            """Makes a lateral shift of the forward vector of a transform"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):   # 画拓扑
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

                for w in waypoints:
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
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Draw Roads
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5

            def to_pixel(wp): return world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_junction else (0, 255, 0)
                for nxt in wp.next(dist):
                    pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)

        actors = carla_world.get_actors()

        # Find and Draw Traffic Signs: Stops and Yields
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        """Scales the map surface"""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


class World(object):
    """Class that contains all the information of a carla world that is running on the server side"""

    def __init__(self, name, args, timeout):
        self.client = None
        self.name = name
        self.args = args
        self.timeout = timeout
        self.server_fps = 0.0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()

        # World data
        self.world = None
        self.town_map = None
        self.actors_with_transforms = []

        self._hud = None
        self._input = None

        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0

        # Hero actor
        self.hero_actor = None
        self.spawned_hero = None
        self.hero_transform = None

        self.scale_offset = [0, 0]

        self.vehicle_id_surface = None
        self.result_surface = None

        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.affected_traffic_light = None

        # Map info
        self.map_image = None
        self.border_round_surface = None
        self.original_surface_size = None
        self.hero_surface = None
        self.actors_surface = None

    def _get_data_from_carla(self):
        """Retrieves the data from the server side"""
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.timeout)

            if self.args.map is None:
                world = self.client.get_world()
            else:
                world = self.client.load_world(self.args.map)

            town_map = world.get_map()
            return (world, town_map)

        except RuntimeError as ex:
            logging.error(ex)
            exit_game()

    def start(self, hud, input_control):
        """Build the map image, stores the needed modules and prepares rendering in Hero Mode"""
        self.world, self.town_map = self._get_data_from_carla()

        settings = self.world.get_settings()
        settings.no_rendering_mode = self.args.no_rendering
        self.world.apply_settings(settings)

        # Create Surfaces
        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.town_map,
            pixels_per_meter=PIXELS_PER_METER,
            show_triggers=self.args.show_triggers,
            show_connections=self.args.show_connections,
            show_spawn_points=self.args.show_spawn_points)

        self._hud = hud
        self._input = input_control

        self.original_surface_size = min(self._hud.dim[0], self._hud.dim[1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)

        # Render Actors
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

        self.border_round_surface = pygame.Surface(self._hud.dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(COLOR_WHITE)
        self.border_round_surface.fill(COLOR_BLACK)

        # Used for Hero Mode, draws the map contained in a circle with white border
        center_offset = (int(self._hud.dim[0] / 2), int(self._hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self._hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self._hud.dim[1] - 8) / 2))

        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)

        # Start hero mode by default
        self.select_hero_actor()
        self.hero_actor.set_autopilot(False)
        self._input.wheel_offset = HERO_DEFAULT_SCALE
        self._input.control = carla.VehicleControl()

        # Register event for receiving server tick
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: World.on_world_tick(weak_self, timestamp))

    def select_hero_actor(self):
        """Selects only one hero actor if there are more than one. If there are not any, it will spawn one."""
        hero_vehicles = [actor for actor in self.world.get_actors()
                         if 'vehicle' in actor.type_id and actor.attributes['role_name'] == 'hero']
        if len(hero_vehicles) > 0:
            self.hero_actor = random.choice(hero_vehicles)
            self.hero_transform = self.hero_actor.get_transform()
        else:
            self._spawn_hero()

    def _spawn_hero(self):
        """Spawns the hero actor when the script runs"""
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self.args.filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        while self.hero_actor is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.hero_actor = self.world.try_spawn_actor(blueprint, spawn_point)
        self.hero_transform = self.hero_actor.get_transform()

        # Save it in order to destroy it when closing program
        self.spawned_hero = self.hero_actor

    def tick(self, clock):
        """Retrieves the actors for Hero and Map modes and updates de HUD based on that"""
        actors = self.world.get_actors()

        # We store the transforms also so that we avoid having transforms of
        # previous tick and current tick when rendering them.
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()

        self.update_hud_info(clock)

    def update_hud_info(self, clock):
        """Updates the HUD info regarding simulation, hero mode and whether there is a traffic light affecting the hero actor"""

        hero_mode_text = []
        if self.hero_actor is not None:
            hero_speed = self.hero_actor.get_velocity()
            hero_speed_text = 3.6 * math.sqrt(hero_speed.x ** 2 + hero_speed.y ** 2 + hero_speed.z ** 2)

            affected_traffic_light_text = 'None'
            if self.affected_traffic_light is not None:
                state = self.affected_traffic_light.state
                if state == carla.TrafficLightState.Green:
                    affected_traffic_light_text = 'GREEN'
                elif state == carla.TrafficLightState.Yellow:
                    affected_traffic_light_text = 'YELLOW'
                else:
                    affected_traffic_light_text = 'RED'

            affected_speed_limit_text = self.hero_actor.get_speed_limit()
            if math.isnan(affected_speed_limit_text):
                affected_speed_limit_text = 0.0
            hero_mode_text = [
                'Hero Mode:                 ON',
                'Hero ID:              %7d' % self.hero_actor.id,
                'Hero Vehicle:  %14s' % get_actor_display_name(self.hero_actor, truncate=14),
                'Hero Speed:          %3d km/h' % hero_speed_text,
                'Hero Affected by:',
                '  Traffic Light: %12s' % affected_traffic_light_text,
                '  Speed Limit:       %3d km/h' % affected_speed_limit_text
            ]
        else:
            hero_mode_text = ['Hero Mode:                OFF']

        self.server_fps = self.server_clock.get_fps()
        self.server_fps = 'inf' if self.server_fps == float('inf') else round(self.server_fps)
        info_text = [
            'Server:  % 16s FPS' % self.server_fps,
            'Client:  % 16s FPS' % round(clock.get_fps()),
            'Simulation Time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            'Map Name:          %10s' % self.town_map.name,
        ]

        self._hud.add_info(self.name, info_text)
        self._hud.add_info('HERO', hero_mode_text)

    @staticmethod
    def on_world_tick(weak_self, timestamp):
        """Updates the server tick"""
        self = weak_self()
        if not self:
            return

        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

    def _show_nearby_vehicles(self, vehicles):
        """Shows nearby vehicles of the hero actor"""
        info_text = []
        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.hero_actor.id]

            def distance(v): return location.distance(v.get_location())
            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                info_text.append('% 5d %s' % (vehicle.id, vehicle_type))
        self._hud.add_info('NEARBY VEHICLES', info_text)

    def _split_actors(self):
        """Splits the retrieved actors by type id"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled"""
        self.affected_traffic_light = None

        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)

            if self.args.show_triggers:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_BUTTER_1, True, corners, 2)

            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()

                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
                if (d <= s):
                    # Highlight traffic light
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):
        """Renders the speed limits by drawing two concentric circles (outer is red and inner white) and a speed limit text"""

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Render speed limit concentric circles
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)

            if self.args.show_triggers:
                corners = Util.get_bounding_box(sl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_PLUM_2, True, corners, 2)

            # Blit
            if self.hero_actor is not None:
                # In hero mode, Rotate font surface with respect to hero vehicle front
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                # In map mode, there is no need to rotate the text of the speed limit
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        """Renders the walkers' bounding boxes"""
        for w in list_w:
            color = COLOR_PLUM_0

            # Compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))

    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        """Renders all the actors"""
        # Static actors
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel,
                                  self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        """Used to improve perfomance. Clips the surfaces in order to render only the part of the surfaces that are going to be visible"""
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor):
        """Based on the mouse wheel and mouse position, it will compute the scale and move the map so that it is zoomed in or out based on mouse position"""
        m = self._input.mouse_pos

        # Percentage of surface where mouse position is actually
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)

        # Offset will be the previously accumulated offset added with the
        # difference of mouse positions in the old and new scales
        diff_between_scales = ((float(self.prev_scaled_size) * px) - (float(self.scaled_size) * px),
                               (float(self.prev_scaled_size) * py) - (float(self.scaled_size) * py))

        self.scale_offset = (self.scale_offset[0] + diff_between_scales[0],
                             self.scale_offset[1] + diff_between_scales[1])

        # Update previous scale
        self.prev_scaled_size = self.scaled_size

        # Scale performed
        self.map_image.scale_map(scale_factor)

    def render(self, display):
        """Renders the map and all the actors in hero and map mode"""
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)

        # Split the actors by vehicle type id
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        # Zoom in and out
        scale_factor = self._input.wheel_offset
        self.scaled_size = int(self.map_image.width * scale_factor)
        if self.scaled_size != self.prev_scaled_size:
            self._compute_scale(scale_factor)

        # Render Actors
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(
            self.actors_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers)

        # Render Ids
        self._hud.render_vehicles_ids(self.vehicle_id_surface, vehicles,
                                      self.map_image.world_to_pixel, self.hero_actor, self.hero_transform)
        # Show nearby actors from hero mode
        self._show_nearby_vehicles(vehicles)

        # Blit surfaces
        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)

        center_offset = (0, 0)
        if self.hero_actor is not None:
            # Hero Mode
            hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * PIXELS_AHEAD_VEHICLE,
                                  (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * PIXELS_AHEAD_VEHICLE))

            # Apply clipping rect
            clipping_rect = pygame.Rect(translation_offset[0],
                                        translation_offset[1],
                                        self.hero_surface.get_width(),
                                        self.hero_surface.get_height())
            self.clip_surfaces(clipping_rect)

            Util.blits(self.result_surface, surfaces)

            self.border_round_surface.set_clip(clipping_rect)

            self.hero_surface.fill(COLOR_ALUMINIUM_4)
            self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                         -translation_offset[1]))

            rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)

            display.blit(self.border_round_surface, (0, 0))
        else:
            # Map Mode
            # Translation offset
            translation_offset = (self._input.mouse_offset[0] * scale_factor + self.scale_offset[0],
                                  self._input.mouse_offset[1] * scale_factor + self.scale_offset[1])
            center_offset = (abs(display.get_width() - self.surface_size) / 2 * scale_factor, 0)

            # Apply clipping rect
            clipping_rect = pygame.Rect(-translation_offset[0] - center_offset[0], -translation_offset[1],
                                        self._hud.dim[0], self._hud.dim[1])
            self.clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)

            display.blit(self.result_surface, (translation_offset[0] + center_offset[0],
                                               translation_offset[1]))

    def destroy(self):
        """Destroy the hero actor when class instance is destroyed"""
        if self.spawned_hero is not None:
            self.spawned_hero.destroy()

# ==============================================================================
# -- Input -----------------------------------------------------------
# ==============================================================================


class InputControl(object):
    """Class that handles input received such as keyboard and mouse."""

    def __init__(self, name):
        """Initializes input member variables when instance is created."""
        self.name = name    # 名称
        self.mouse_pos = (0, 0) # 鼠标位置
        self.mouse_offset = [0.0, 0.0]  # 鼠标偏移
        self.wheel_offset = 0.1 # 缩放比例
        self.wheel_amount = 0.025   # 缩放量
        self._steer_cache = 0.0 # 方向缓存
        self.control = None # 控制
        self._autopilot_enabled = False # 自动驾驶是否开启

        # Modules that input will depend on
        self._hud = None
        self._world = None

    def start(self, hud, world):
        """Assigns other initialized modules that input module needs."""
        self._hud = hud
        self._world = world

        self._hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def render(self, display):
        """Does nothing. Input module does not need render anything."""

    def tick(self, clock):
        """Executed each frame. Calls method for parsing input."""
        self.parse_input(clock) # 解析输入

    def _parse_events(self):
        """Parses input events. These events are executed only once when pressing a key."""
        self.mouse_pos = pygame.mouse.get_pos() # 获取鼠标的位置
        for event in pygame.event.get():    # 循环遍历每个事件
            if event.type == pygame.QUIT:
                exit_game()
            elif event.type == pygame.KEYUP:    # 如果按键被释放
                if self._is_quit_shortcut(event.key):   # 如果按下的是退出快捷键
                    exit_game() # 退出游戏
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT): # 如果按下的是H键或者是shift+?键
                    self._hud.help.toggle() # 显示帮助信息
                elif event.key == K_TAB:    # 如果按下的是TAB键
                    # Toggle between hero and map mode
                    if self._world.hero_actor is None:  # 如果没有英雄角色
                        self._world.select_hero_actor() # 选择英雄角色
                        self.wheel_offset = HERO_DEFAULT_SCALE  # 设置缩放比例
                        self.control = carla.VehicleControl()   # 设置控制
                        self._hud.notification('Hero Mode')   # 显示通知
                    else:
                        self.wheel_offset = MAP_DEFAULT_SCALE   # 设置缩放比例
                        self.mouse_offset = [0, 0]  # 设置鼠标偏移
                        self.mouse_pos = [0, 0] # 设置鼠标位置
                        self._world.scale_offset = [0, 0]   # 设置缩放偏移
                        self._world.hero_actor = None   # 设置英雄角色为空
                        self._hud.notification('Map Mode')  # 显示通知
                elif event.key == K_F1:
                    self._hud.show_info = not self._hud.show_info   # 显示信息
                elif event.key == K_i:
                    self._hud.show_actor_ids = not self._hud.show_actor_ids  # 显示角色ID
                elif isinstance(self.control, carla.VehicleControl):    # 如果控制是车辆控制
                    if event.key == K_q:
                        self.control.gear = 1 if self.control.reverse else -1
                    elif event.key == K_m:
                        self.control.manual_gear_shift = not self.control.manual_gear_shift # 切换手动挡
                        self.control.gear = self._world.hero_actor.get_control().gear   # 设置挡位
                        self._hud.notification('%s Transmission' % (    # 显示通知
                            'Manual' if self.control.manual_gear_shift else 'Automatic'))   # 手动挡或自动挡
                    elif self.control.manual_gear_shift and event.key == K_COMMA:   # 如果是手动挡
                        self.control.gear = max(-1, self.control.gear - 1)  # 设置挡位
                    elif self.control.manual_gear_shift and event.key == K_PERIOD:  # 如果是手动挡
                        self.control.gear = self.control.gear + 1   # 设置挡位
                    elif event.key == K_p:  # 如果按下的是P键
                        # Toggle autopilot
                        if self._world.hero_actor is not None:  # 如果有英雄角色
                            self._autopilot_enabled = not self._autopilot_enabled   # 切换自动驾驶
                            self._world.hero_actor.set_autopilot(self._autopilot_enabled)   # 设置自动驾驶
                            self._hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))   # 显示通知
            elif event.type == pygame.MOUSEBUTTONDOWN:  # 如果鼠标按下
                # Handle mouse wheel for zooming in and out
                if event.button == 4:   # 如果是鼠标滚轮向上
                    self.wheel_offset += self.wheel_amount  # 缩放比例增加
                    if self.wheel_offset >= 1.0:    # 如果缩放比例大于1
                        self.wheel_offset = 1.0 # 设置缩放比例为1
                elif event.button == 5: # 如果是鼠标滚轮向下
                    self.wheel_offset -= self.wheel_amount  # 缩放比例减小
                    if self.wheel_offset <= 0.1:    # 如果缩放比例小于0.1
                        self.wheel_offset = 0.1     # 设置缩放比例为0.1

    def _parse_keys(self, milliseconds):
        """Parses keyboard input when keys are pressed"""
        keys = pygame.key.get_pressed() # 获取按下的键
        self.control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0 # 设置油门
        steer_increment = 5e-4 * milliseconds   # 设置转向增量
        if keys[K_LEFT] or keys[K_a]:   # 如果按下的是左键或者A键
            self._steer_cache -= steer_increment    # 设置转向缓存
        elif keys[K_RIGHT] or keys[K_d]:    # 如果按下的是右键或者D键
            self._steer_cache += steer_increment    # 设置转向缓存
        else:   # 如果没有按下左右键
            self._steer_cache = 0.0   # 设置转向缓存为0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))  # 设置转向缓存的最大最小值
        self.control.steer = round(self._steer_cache, 1)    # 设置转向
        self.control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0  # 设置刹车
        self.control.hand_brake = keys[K_SPACE]  # 设置手刹


    def _parse_mouse(self):
        """Parses mouse input"""
        if pygame.mouse.get_pressed()[0]:   # 如果鼠标左键按下
            x, y = pygame.mouse.get_pos()   # 获取鼠标位置
            self.mouse_offset[0] += (1.0 / self.wheel_offset) * (x - self.mouse_pos[0]) # 设置鼠标偏移
            self.mouse_offset[1] += (1.0 / self.wheel_offset) * (y - self.mouse_pos[1]) # 设置鼠标偏移
            self.mouse_pos = (x, y) # 设置鼠标位置

    def parse_input(self, clock):   # 解析输入
        """Parses the input, which is classified in keyboard events and mouse"""
        self._parse_events()    # 解析事件
        self._parse_mouse() # 解析鼠标
        if not self._autopilot_enabled: # 如果没有开启自动驾驶
            if isinstance(self.control, carla.VehicleControl):  # 如果控制是车辆控制
                self._parse_keys(clock.get_time())  # 解析键盘输入
                self.control.reverse = self.control.gear < 0    # 检查是否为倒挡
            if (self._world.hero_actor is not None):    # 如果有英雄角色
                self._world.hero_actor.apply_control(self.control)  # 应用控制

    @staticmethod
    def _is_quit_shortcut(key):
        """Returns True if one of the specified keys are pressed"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)  # 如果按下的是ESC键或者是ctrl+q键


# ==============================================================================
# -- Game Loop ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """Initialized, Starts and runs all the needed modules for No Rendering Mode"""
    try:
        # Init Pygame
        pygame.init()
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Place a title to game window
        pygame.display.set_caption(args.description)

        # Show loading screen
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = font.render('Rendering map...', True, COLOR_WHITE)
        display.blit(text_surface, text_surface.get_rect(center=(args.width / 2, args.height / 2)))
        pygame.display.flip()

        # Init
        input_control = InputControl(TITLE_INPUT)
        hud = HUD(TITLE_HUD, args.width, args.height)
        world = World(TITLE_WORLD, args, timeout=20.0)

        # For each module, assign other modules that are going to be used inside that module
        input_control.start(hud, world)
        hud.start()
        world.start(hud, input_control)

        # Game loop
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)

            # Tick all modules
            world.tick(clock)
            hud.tick(clock)
            input_control.tick(clock)

            # Render all modules
            display.fill(COLOR_ALUMINIUM_4)
            world.render(display)
            hud.render(display)
            input_control.render(display)

            pygame.display.flip()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        if world is not None:
            world.destroy()


def exit_game():
    """Shuts down program and PyGame"""
    pygame.quit()
    sys.exit()

# ==============================================================================
# -- Main --------------------------------------------------------------------
# ==============================================================================


def main():
    """Parses the arguments received from commandline and runs the game loop"""

    # Define arguments that will be received and parsed
    argparser = argparse.ArgumentParser(
        description='CARLA No Rendering Mode Visualizer')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default='Town01',
        help='start a new episode at the given TOWN')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='switch off server rendering')
    argparser.add_argument(
        '--show-triggers',
        action='store_true',
        help='show trigger boxes of traffic signs')
    argparser.add_argument(
        '--show-connections',
        action='store_true',
        help='show waypoint connections')
    argparser.add_argument(
        '--show-spawn-points',
        action='store_true',
        help='show recommended spawn points')
    
    # 开启carla服务端
    def init_carla_server():
        global server_port
        global server_pid
        def port_is_used(host, port):
            """检查端口是否被占用"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    return False  # 如果成功绑定，则端口未被占用
                except socket.error as e:
                    return True   # 如果发生错误，则端口可能已被占用    

        
        server_port = random.randrange(2000, 7000, 1)
        # 检查端口是否被占用
        while port_is_used('127.0.0.1', server_port):
            server_port = random.randrange(2000, 7000, 1)

        # 定义程序的路径和参数
        executable_path = r"/home/cunluo/GitHub/CARLA_0.9.15/CarlaUE4.sh"
        carla_args = [f"-carla-rpc-port={server_port}", "-nullrhi"]

        # 使用subprocess.Popen来启动程序
        process = subprocess.Popen([executable_path] + carla_args, cwd=os.path.dirname(executable_path))
        server_pid = process.pid

    init_carla_server()

    # Parse arguments
    args = argparser.parse_args()
    args.description = argparser.description
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.port = server_port

    # Print server information
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    # Run game loop
    game_loop(args)


if __name__ == '__main__':
    main()
