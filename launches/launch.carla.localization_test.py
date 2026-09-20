"""Opt-in CARLA launch for LiDAR localization testing."""

from os import path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_lidar_localization', default_value='true'),
        DeclareLaunchArgument('localization_map_path', default_value=''),
        DeclareLaunchArgument('localization_pointcloud_topic', default_value='/lidar'),
        DeclareLaunchArgument('localization_gnss_topic', default_value='/gnss/odometry_raw'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(path.join(path.dirname(__file__), 'launch.carla.py')),
        ),
        Node(
            package='lidar_localization',
            executable='localization_gpsguess',
            name='localization_gpsguess',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_lidar_localization')),
            parameters=[
                {'map_path': LaunchConfiguration('localization_map_path')},
                {'pointcloud_topic': LaunchConfiguration('localization_pointcloud_topic')},
                {'gnss_topic': LaunchConfiguration('localization_gnss_topic')},
            ],
        ),
    ])
