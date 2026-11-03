---
layout: default
title: Traffic Rules
parent: Planning
nav_order: 9
---

# Camera traffic lights and stop signs

The perception launch starts both local diagnostic detectors on `/cameras/camera0`.
They process each new frame once, at up to 5 Hz, with sensor-data QoS.
CPU inference uses two Torch threads per process. The `device` parameter
can select an explicit GPU after checking available memory.

Traffic lights use `/navigator_binaries/best_traffic_med_yolo_v8.pt`.
The actual weights label red=0, green=1, off=2, yellow=3. The node translates
these by name to Navigator's wire labels red=0, yellow=1, green=2 and
publishes confidence scores on `/traffic_lights/detections`; off is ignored.
In opt-in camera mode, ACC and the intersection manager stop on red/yellow and release on green.
Empty frames retain an observed red stop. CARLA's optional
`/carla/traffic_light_state` remains an independent stop override; yellow
now stops there too. A subscriber alone does not prove the bridge publishes
that optional topic.

Stop signs use the local COCO model `/navigator/data/models/yolo11n.pt`
(class 11, `stop sign`) and publish all detections, including empty results,
on `/perception/stop_signs`. They do not use the legacy external inference
service or its credentials. The model was downloaded locally; weights are
ignored by Git. On another checkout, provision it inside the Navigator
container with:

```bash
mkdir -p /navigator/data/models
cd /navigator/data/models
python3 -c 'from ultralytics import YOLO; YOLO("yolo11n.pt")'
```

ACC requires two confident sign detections within one second. It estimates
range using `focal_length_y * nominal_sign_height / detected_pixel_height`.
`CameraInfo.k[4]` supplies focal length when available. Parameters on ACC:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `camera_info_topic` | `/carla/hero/rgb_front/camera_info` | Front-camera calibration; configure for physical cameras |
| `stop_sign_focal_length_px` | 640 | Fallback focal length if calibration is absent |
| `stop_sign_height_m` | 0.75 | Nominal physical sign height |

The approach speed limit follows a 1.5 m/s² deceleration profile toward a
3 m buffer before the estimated sign position. If detections disappear on
approach, integrated speed continues the approach estimate. ACC holds full
brake and requires speed below 0.1 m/s for three seconds. It then permits
motion subject to red lights and obstacle braking. The same sign cannot
rearm until the car travels 10 m and the sign has been absent for five
seconds. Stale detections older than 0.5 s are ignored by ACC.

These are camera-based prototypes. Nominal sign size is not a measured stop
line, and the detectors do not establish which lane a light/sign governs or
who has right of way. Nearby cross-street signs/lights can cause conservative
stops. Empty detections intentionally do not establish red-light clearance.
Validate stopping position and sign/light applicability in CARLA before
relying on this behavior. No live camera stream was available during this
change; model loading/inference and synthetic control sequences were tested,
not live scene precision/recall.

Build and check inside the sourced Navigator container:

```bash
cd /navigator
colcon build --symlink-install --packages-select autonomous_cruise traffic_light_detector road_signs_classifier intersection_manager
source install/setup.bash
python3 -m pytest -q test/integration/test_planning_acc_contract.py src/autonomous_cruise/test/test_autonomous_cruise.py
```

Use an isolated container for tests. Normal driving uses `ROS_DOMAIN_ID=83`.
After restarting the stack, inspect `/traffic_lights/detections` and
`/perception/stop_signs` alongside `/vehicle/control` to verify actual
observations and resulting brake commands.


## CARLA control authority (current default)

`traffic_rule_source: carla` on ACC is now the default. Camera predictions
are diagnostic and cannot command a traffic-rule stop in this mode. The
legacy intersection manager also ignores camera light observations in its
CARLA mode. This avoids a speed-limit sign classified as `stop` causing a
simulator stop. Camera mode is explicit opt-in (`traffic_rule_source:
camera`); its stop-sign detector additionally requires a filled red polygon
and rejects white/round speed-sign crops. This is a rejection filter, not
proof of camera accuracy across real scenes.

CARLA's `TrafficLight.get_state()` and `get_stop_waypoints()` provide state
and lane applicability without depending on horizontal/vertical lamp
layout. `world.get_traffic_lights_from_waypoint()` selects candidate lights.
The observer also filters exactly `traffic.stop*` actors and matches their
oriented trigger volumes against forward lane waypoints. Speed-limit actors
and stop triggers on other lanes are excluded. At junctions the forward
lane search can include several reachable branches; route-specific branch
selection is not yet provided, so conservative stops remain possible.

The new observer publishes timestamped JSON facts on `/carla/scene`:
relevant light state/distance, stop actor ID/distance, ego front extent,
and nearby vehicle clearance. It queries the simulator only: it never
controls actors, ticks the world, or changes simulator settings. Vehicle
clearance subtracts both vehicles' bounding-box extents. ACC slows toward
a 2.5 m bumper gap and holds the brake while that gap is occupied. A lost
or invalid API stream holds the brake. LiDAR now starts slowing at 12 m,
stops at 5 m measured from base_link, retains a detected hazard across
pre-stop point loss, and uses API clearance as a veto on stop release.
This replaces the old 2 m threshold that did not account for the ego bumper.

Start the observer alongside the bridge using the Compose service:

```bash
cd /home/mmanchikanti/novaDev/navigator/docker/carla-ros-bridge
ROS_DOMAIN_ID=83 docker compose up -d scene_observer
```

For an existing ephemeral bridge container, the alternative launcher copies
and runs the same observer in that container's compatible CARLA/Python 3.8
runtime (choose one observer deployment):

```bash
cd /home/mmanchikanti/novaDev/navigator
bash scripts/run_carla_scene_observer.sh YOUR_BRIDGE_CONTAINER
```

The helper uses simulator port 2083 and ROS domain 83. Restart Navigator to
load ACC changes. ACC also selects simulation time in CARLA-source mode
when launched via `launch.vehicle.py`. Inspect `/carla/scene` and
`/autonomous_cruise/debug`: brake-hold messages now identify missing scene
facts, close vehicles, red/yellow stop lines, and stop-sign dwell.

The initial API probe had zero simulator actors. Later, a live ego actor
became available and the observer successfully published valid scene facts
on domain 83. Its API front extent was 2.395889759 m, confirming that the
old 2 m origin-based threshold lay inside the ego footprint. A live
parked-vehicle collision sequence or actual signal approach was not reproduced. Regression tests cover API geometry, both light layouts,
speed-sign exclusion, stop triggers on adjacent lanes, detection loss,
vehicle-clearance vetoes, stale API input, and camera crop rejection.


The observer Compose service uses host IPC as well as host networking.
Fast DDS participants otherwise discover one another on this host but can
fail to deliver scene messages across separate shared-memory namespaces.
The bridge service depends on the persistent observer, so normal Compose
bridge startup starts it automatically. Following this configuration fix,
live domain-83 verification showed valid scene reception, brake=0, and
vehicle speed approximately 3.51 m/s. The earlier hold was explicitly
`CARLA scene missing, stale, or invalid`, rather than a retained red light.


## Standstill gap tuning

A read-only live probe measured a 2.7167 m bumper gap while the old LiDAR
stop latch held the brake. The current `standstill_gap_m` parameter on ACC
defaults to 1.5 m. In CARLA mode both the API vehicle-gap speed limit and
LiDAR's hard-stop threshold use that same clearance. LiDAR converts it to
an origin-based distance by adding the API-reported ego front extent
(2.3959 + 1.5 = approximately 3.90 m for this vehicle). The 12 m slowdown
onset, stop-release hysteresis, traffic-light buffer, and stop-sign dwell
are unchanged. Actual achieved gaps vary with sampling and vehicle dynamics.
Restart Navigator to load this code; later adjustments can use the ROS
`standstill_gap_m` parameter. Existing camera-mode fallback distances stay
unchanged.

## Recovery after simulator restarts

The observer discards its CARLA client after a connection/read failure and
creates a new client on the next poll. Reusing a client from a terminated
simulator session could leave it timing out even after a new client could
connect, causing ACC to hold the brake on green because scene facts were
invalid. Missing ego actors do not churn an otherwise healthy connection.
Invalid facts still hold the brake until fresh valid facts arrive.

After loading this fix by restarting only the observer service, a live
domain-83 trace showed Green, brake=0, and speed rising to 2.77 m/s.
Regression tests in `test/integration/test_carla_scene_reconnect.py` cover
failed-session replacement and recovery when the ego actor appears.
