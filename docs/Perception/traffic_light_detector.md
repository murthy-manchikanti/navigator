---
layout: default
title: Traffic Light Detector
nav_order: 1
parent: Perception
---

# Traffic Light Detector

The active perception launch runs `traffic_light_detector/traffic_light_node`
on `/cameras/camera0` using the local weights
`/navigator_binaries/best_traffic_med_yolo_v8.pt`. Images use sensor-data QoS
and are converted to BGR. Each new frame is processed once at up to 5 Hz.

Output: `/traffic_lights/detections` (`navigator_msgs/TrafficLightDetection`).
Each detection includes its box, score, and Navigator label:

| Label | Meaning |
| --- | --- |
| 0 | Red |
| 1 | Yellow |
| 2 | Green |

The installed model uses different IDs; the node translates by class name
and ignores `off`. Image timestamps/frame IDs are preserved, and empty
results are published. ACC stops on red/yellow and only a subsequent fresh
green observation clears the camera stop. These observations are not yet
associated with a particular lane or stop line.

See [Traffic Rules](../Planning/traffic_rules.md) for setup, behavior,
parameters, tests, and validation limits.

In the default CARLA control mode these camera outputs are diagnostic.
The read-only CARLA scene observer supplies authoritative sign/light facts;
set `traffic_rule_source: camera` explicitly for camera-control experiments.
