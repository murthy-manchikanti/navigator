"""This module contains the launch description for the Carla simulator."""

from launch import LaunchDescription
from launch_ros.actions import SetParameter

from os import path
import sys

sys.path.append(path.abspath('/navigator/'))
from launches.utils import err_fatal, try_get_launch_description_from_include, IncludeError


def generate_launch_description():
    """Carla launch file description. This is currently identical to the vehicle launch file."""

    try:
        vehicle_launch_description = try_get_launch_description_from_include('launches/launch.vehicle.py')
    except IncludeError as e:
        err_fatal(e)

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        *vehicle_launch_description.describe_sub_entities(),
    ])
