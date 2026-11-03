---
layout: default
title: Road Signs Classifier
parent: Perception
---

# Stop-sign detection

`road_signs_classifier` now runs local YOLO11n COCO weights from
`/navigator/data/models/yolo11n.pt`. It detects the `stop sign` class in
`/cameras/camera0` and publishes `navigator_msgs/RoadSignsDetection` on
`/perception/stop_signs`, preserving the image header. Every new processed
frame publishes all signs or an empty array. The confidence threshold is
0.6 by default; processing is capped at 5 Hz with two CPU threads.

This replaces the old external inference-service implementation. General
road-sign classes are not currently enabled. ACC estimates sign range,
slows on approach, holds a full stop for three seconds, then suppresses
repeated stops for the same sign using travel/visibility conditions.

See [Traffic Rules](../Planning/traffic_rules.md) for model provisioning,
calibration, behavior, tests, and validation limits.

In the default CARLA control mode these camera outputs are diagnostic.
The read-only CARLA scene observer supplies authoritative sign/light facts;
set `traffic_rule_source: camera` explicitly for camera-control experiments.
