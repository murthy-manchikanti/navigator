# BUILD COMMANDS
colcon build --symlink-install
Flags:
    --symlink-install
    --cmake-clean-cache
    --packages-select <package_name>

# SOURCE COMMAND
. install/setup.bash

# LAUNCHING COMMANDS
navigator launch {vehicle, carla, perception}
    Ex. navigator launch vehicle

# ROS COMMANDS
rviz2
rqt
rqt_graph
ros2 topic list/echo
etc.

## BUILD SEQUENCE
colcon build --symlink-install
. install/setup.bash

## CLEAN BUILD SEQUENCE
rm -rf build/ install/
colcon build --symlink-install --cmake-clean-cache
. install/setup.bash

### TESTING SEQUENCE
launch carla
launch carla_bridge (carla_interface repo)
launch navigator
## CARLA LiDAR Localization Test (Opt-in)
Build:
colcon build --symlink-install --packages-select lidar_localization
. install/setup.bash

Run default CARLA stack (unchanged, localization disabled):
navigator launch carla

Run CARLA with localization test enabled:
navigator launch carla_localization_test localization_map_path:=/absolute/path/to/map.pcd

Optional topic overrides:
- localization_pointcloud_topic (default: /lidar)
- localization_gnss_topic (default: /gnss/odometry_raw)

Assumptions:
- LiDAR topic publishes sensor_msgs/PointCloud2 with x/y/z fields.
- GNSS topic publishes nav_msgs/Odometry in the map frame.
- The map file is a valid .pcd/.ply readable by Open3D and in map coordinates matching CARLA world alignment.
