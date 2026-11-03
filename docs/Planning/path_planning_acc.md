---
layout: default
title: Path Planning and ACC
parent: Planning
nav_order: 8
---

# Path planning and Autonomous Cruise Control

The active vehicle launch uses `costs/route_costmap_node` →
`costs/grid_summation_node` → `path_planners/path_planner_node` →
`autonomous_cruise/autonomous_cruise_node`. The route is in `map`; local
costmaps, goals, and paths are in `base_link`. The vision route costmap
`/route_costmap/segmented_v2` is a separate comparison output; it is not an
input to grid summation.

PR #500 removed the planner's path cache. PR #503 switched to ACC and fixed
its local-frame handling and CARLA steering sign, but route costmap caching
still republished old local geometry with new timestamps. Its history also
records unstable turns after obstacle-detour tuning. This fix preserves the
existing steering sign and obstacle padding; it refreshes local geometry
and bounds goal selection to an 8 m horizon.

`navigator launch carla` now sets `use_sim_time=true` for all launched nodes,
so grid timestamps and TF buffers use the bridge's `/clock`. The physical
vehicle launch retains its default clock settings.

A failed plan publishes an empty path. ACC holds full brake for empty,
one-pose, stale, or non-`base_link` paths. `path_timeout` defaults to 0.5 s
and checks both the message stamp and time since reception. The PID state
is reset during that stop. This also covers the planner exiting silently.
The intersection manager accepts short paths and subscribes to `/speed`
using `navigator_msgs/VehicleSpeed` and sensor-data QoS, matching ACC.

ACC's object following reads centers from `BoundingBox3D.coordinates[:3]`,
matching the actual Navigator message definition (there is no `center`
field).

## Build and regression checks

Run Compose from the intended checkout: the bind mount uses `$PWD`. Use an
existing development container if one already mounts that checkout; do not
start a second control stack on the same ROS domain.

```bash
cd /home/mmanchikanti/novaDev/navigator
docker compose run --rm navigator_carla
```

Inside the container:

```bash
cd /navigator
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select autonomous_cruise costs path_planners intersection_manager segmentation
source install/setup.bash
ROS_DOMAIN_ID=119 python3 -m pytest -q test/integration/test_planning_acc_contract.py src/autonomous_cruise/test/test_autonomous_cruise.py
```

The integration tests construct ROS nodes on an isolated domain and capture
output callbacks, without transmitting commands to a vehicle. They cover
rectangular grid geometry, steering direction, blocked/missing/stale input,
ACC path expiry, short paths, route-end publication, and route refresh after
small translations and yaw changes. The broader intersection-handler suite
has separate existing failures in its stop/proceed timing scenarios; that
handler is not used by the active basic ACC launch.

For a driving check, start CARLA and its bridge first, verify `/clock`,
`/gnss/odometry`, `/speed`, and sensor inputs are present in the same domain,
then run `python3 navigator.py launch carla`. Check that `/planning/path`
updates, the intersection manager stays alive, and `/vehicle/control`
brakes when the path publisher stops. Common setuptools `tests_require`
warnings do not indicate a build failure. A missing input topic, incompatible
QoS, or a node traceback does require investigation.


## Obstacle stop hold

See [Traffic Rules](traffic_rules.md#carla-control-authority-current-default)
for the current CARLA API authority, 12 m slowdown / 5 m LiDAR stop
thresholds, bumper-clearance veto, retained hazards across lost detections,
and explicit brake-hold diagnostics. Camera mode remains available for
sensor-only experiments and requires separate validation.
