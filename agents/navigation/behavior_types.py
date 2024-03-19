# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 25
    speed_lim_dist = 6  # 开始减速的距离
    speed_decrease = 12 # 减速的速度
    safety_time = 3 # 安全时间
    min_proximity_threshold = 12    # 最小距离
    braking_distance = 6    # 刹车距离
    tailgate_counter = 0    # 尾随计数器


class Normal(object):
    """Class for Normal agent."""
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 80
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 0
    tailgate_counter = -1
