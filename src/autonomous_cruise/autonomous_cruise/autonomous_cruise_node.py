#!/usr/bin/env python3
"""
Main autonomous cruise controller node (basic version).

Author: Siddarth Nandyala
Contact: siddarth.nandyala@utdallas.edu
Organization: Nova UTD Autonomous Driving
Date: November 2025
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, CameraInfo
from navigator_msgs.msg import VehicleControl, Object3DArray, VehicleSpeed, IntersectionBehavior
from std_msgs.msg import String
from navigator_msgs.msg import RoadSignsDetection, TrafficLightDetection
from autonomous_cruise.traffic_rules import StopSignPolicy
from geometry_msgs.msg import Pose

import math
import json
import numpy as np
from typing import Optional, Tuple

from autonomous_cruise.lateral_controller import PurePursuitController
from autonomous_cruise.longitudinal_controller import AdaptiveLongitudinalController


class AutonomousCruiseController(Node):
    """
    Autonomous cruise controller node combining lateral and longitudinal control.

    This node implements a complete autonomous driving system for highway
    cruising, including:
    - Lateral control (steering) using Pure Pursuit algorithm
    - Longitudinal control (speed) using adaptive PID
    - Adaptive cruise control with safe vehicle following
    - Integration with Navigator perception and planning stack
    """

    def __init__(self):
        """Initialize the autonomous cruise controller node."""
        super().__init__('autonomous_cruise_controller')

        # Declare parameters
        self._declare_parameters()

        # Simulator facts and sensor/path timestamps share the CARLA clock,
        # including when this node is launched through launch.vehicle.py.
        if self.get_parameter('traffic_rule_source').value == 'carla':
            from rclpy.parameter import Parameter
            self.set_parameters([Parameter('use_sim_time', value=True)])

        # Load parameters
        self._load_parameters()

        # Initialize controllers
        self.lateral_controller = PurePursuitController(
            wheelbase=self.wheelbase,
            min_lookahead=self.min_lookahead,
            max_lookahead=self.max_lookahead,
            lookahead_gain=self.lookahead_gain,
            max_steer=self.max_steer
        )

        self.longitudinal_controller = AdaptiveLongitudinalController(
            target_speed=self.target_speed,
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            max_accel=self.max_accel,
            max_decel=self.max_decel,
            max_jerk=self.max_jerk,
            max_throttle=self.max_throttle,
            max_brake=self.max_brake,
            integral_limit=self.integral_limit,
            time_gap=self.time_gap,
            min_following_distance=self.min_following_distance,
            max_detection_distance=self.max_detection_distance,
            detection_angle=self.detection_angle
        )

        # State variables
        self.current_path: Optional[Path] = None
        self.path_received_time = None
        self.current_odom: Optional[Odometry] = None
        self.current_objects: Optional[Object3DArray] = None
        self.current_speed: float = 0.0
        self.intersection_action: str = 'Proceed'  # 'Wait' = stop, 'Proceed' = go
        self.traffic_light_red: bool = False
        self.camera_light_stop = False
        self.stop_sign_policy = StopSignPolicy()
        self.carla_scene = None
        self.carla_scene_received = None
        self._last_stop_actor = None
        self._recent_lidar_hazard = False
        self._was_moving_for_obstacle = False
        self._recent_lidar_y = 0.0
        self._hazard_distance = float("inf")
        self.sign_focal_length = self.get_parameter("stop_sign_focal_length_px").value
        self.lidar_obstacle_distance: float = float('inf')  # m to nearest forward obstacle
        self.obstacle_stop_latched = False
        self._stop_obstacle_y = 0.0
        self._clearance_since = None
        self._last_lidar_time = None
        self.last_control_time = self.get_clock().now()
        self.enabled = True

        # QoS profiles
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            self.path_topic,
            self.path_callback,
            qos_reliable
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_best_effort
        )

        self.objects_sub = self.create_subscription(
            Object3DArray,
            self.objects_topic,
            self.objects_callback,
            qos_reliable
        )

        self.speed_sub = self.create_subscription(
            VehicleSpeed,
            '/speed',
            self.speed_callback,
            qos_best_effort
        )

        self.intersection_sub = self.create_subscription(
            IntersectionBehavior,
            '/intersection',
            self.intersection_callback,
            qos_reliable
        )
        self.traffic_light_sub = self.create_subscription(
            String,
            "/carla/traffic_light_state",
            self.traffic_light_callback,
            qos_reliable
        )

        self.scene_sub = self.create_subscription(
            String, '/carla/scene', self.scene_callback, qos_reliable)
        self.camera_light_sub = self.create_subscription(
            TrafficLightDetection, '/traffic_lights/detections', self.camera_light_callback, qos_reliable)
        self.stop_sign_sub = self.create_subscription(
            RoadSignsDetection, '/perception/stop_signs', self.stop_sign_callback, qos_reliable)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.camera_info_callback, qos_best_effort)

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/lidar/filtered',
            self.lidar_callback,
            qos_best_effort
        )

        # Publishers
        self.control_pub = self.create_publisher(
            VehicleControl,
            self.control_topic,
            10
        )

        self.debug_pub = self.create_publisher(
            String,
            '/autonomous_cruise/debug',
            10
        )

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info('Autonomous Cruise Controller initialized')
        self.get_logger().info(f'Target speed: {self.target_speed:.2f} m/s')
        self.get_logger().info(f'Control rate: {self.control_rate} Hz')
        self.get_logger().info(f'Subscribing to path: {self.path_topic}')
        self.get_logger().info(f'Subscribing to odometry: {self.odom_topic}')
        self.get_logger().info(f'Subscribing to objects: {self.objects_topic}')
        self.get_logger().info(f'Publishing control: {self.control_topic}')

    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Topic names
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter('odom_topic', '/gnss/odometry')
        self.declare_parameter('objects_topic', '/tracked/objects3d')
        self.declare_parameter('control_topic', '/vehicle/control')

        # Control parameters
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('path_timeout', 0.5)
        self.declare_parameter('traffic_rule_source', 'carla')
        self.declare_parameter('standstill_gap_m', 1.5)
        self.declare_parameter('stop_sign_focal_length_px', 640.0)
        self.declare_parameter('stop_sign_height_m', 0.75)
        self.declare_parameter('camera_info_topic', '/carla/hero/rgb_front/camera_info')
        self.declare_parameter('enabled', True)

        # Vehicle parameters
        self.declare_parameter('wheelbase', 2.875)

        # Lateral control parameters
        self.declare_parameter('min_lookahead', 3.0)
        self.declare_parameter('max_lookahead', 15.0)
        self.declare_parameter('lookahead_gain', 0.5)
        self.declare_parameter('max_steer', 1.0)

        # Longitudinal control parameters
        self.declare_parameter('target_speed', 6.0)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.15)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('max_accel', 2.5)
        self.declare_parameter('max_decel', -3.5)
        self.declare_parameter('max_jerk', 3.0)
        self.declare_parameter('max_throttle', 0.75)
        self.declare_parameter('max_brake', 1.0)
        self.declare_parameter('integral_limit', 5.0)

        # Adaptive cruise parameters
        self.declare_parameter('time_gap', 1.5)
        self.declare_parameter('min_following_distance', 5.0)
        self.declare_parameter('max_detection_distance', 50.0)
        self.declare_parameter('detection_angle', 0.3)

    def _load_parameters(self):
        """Load parameters from ROS2 parameter server."""
        # Topic names
        self.path_topic = self.get_parameter('path_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.control_topic = self.get_parameter('control_topic').value

        # Control parameters
        self.control_rate = self.get_parameter('control_rate').value
        self.path_timeout = self.get_parameter('path_timeout').value
        self.enabled = self.get_parameter('enabled').value

        # Vehicle parameters
        self.wheelbase = self.get_parameter('wheelbase').value

        # Lateral control parameters
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value
        self.max_steer = self.get_parameter('max_steer').value

        # Longitudinal control parameters
        self.target_speed = self.get_parameter('target_speed').value
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_decel = self.get_parameter('max_decel').value
        self.max_jerk = self.get_parameter('max_jerk').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.max_brake = self.get_parameter('max_brake').value
        self.integral_limit = self.get_parameter('integral_limit').value

        # Adaptive cruise parameters
        self.time_gap = self.get_parameter('time_gap').value
        self.min_following_distance = self.get_parameter(
            'min_following_distance'
        ).value
        self.max_detection_distance = self.get_parameter(
            'max_detection_distance'
        ).value
        self.detection_angle = self.get_parameter('detection_angle').value

    def path_callback(self, msg: Path):
        """
        Callback for path messages.

        Args:
            msg: Path message from path planner.
        """
        self.path_received_time = self.get_clock().now()
        self.current_path = msg
        self.lateral_controller.set_path(msg)

        self.get_logger().debug(
            f'Received path with {len(msg.poses)} waypoints'
        )

    def odom_callback(self, msg: Odometry):
        """
        Callback for odometry messages.

        Args:
            msg: Odometry message with vehicle state.
        """
        self.current_odom = msg

    def objects_callback(self, msg: Object3DArray):
        """
        Callback for detected objects.

        Args:
            msg: Array of detected objects from perception.
        """
        self.current_objects = msg

        self.get_logger().debug(
            f'Received {len(msg.objects)} detected objects'
        )

    def speed_callback(self, msg: VehicleSpeed):
        """Callback for vehicle speed from GNSS processor."""
        self.current_speed = msg.speed

    def intersection_callback(self, msg: IntersectionBehavior):
        """Callback for intersection manager commands (Wait / Proceed)."""
        self.intersection_action = msg.action

    def traffic_light_callback(self, msg: String):
        self.traffic_light_red = msg.data.strip().lower() in ('red', 'yellow')

    def scene_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            self.carla_scene = data
            self.carla_scene_received = self.get_clock().now()
            if data.get('valid') and data.get('stop_distance') is not None:
                stop_id = data.get('stop_id')
                if self._last_stop_actor != stop_id and self.stop_sign_policy.served:
                    self.stop_sign_policy = StopSignPolicy()
                self._last_stop_actor = stop_id
                self.stop_sign_policy.observe(float(data['stop_distance']), self.get_clock().now().nanoseconds / 1e9)
        except (ValueError, TypeError):
            return

    def _fresh_detection(self, header):
        now = self.get_clock().now().nanoseconds / 1e9
        stamp = header.stamp.sec + header.stamp.nanosec / 1e9
        return -0.1 <= now - stamp <= 0.5

    def camera_info_callback(self, msg):
        if msg.k[4] > 0:
            self.sign_focal_length = float(msg.k[4])

    def camera_light_callback(self, msg):
        if self.get_parameter('traffic_rule_source').value != 'camera':
            return
        if not self._fresh_detection(msg.header):
            return
        labels = {light.label for light in msg.traffic_lights
                  if light.score >= 0.6 and light.height > 0}
        if 0 in labels or 1 in labels:
            self.camera_light_stop = True
        elif 2 in labels:
            self.camera_light_stop = False
        # An empty frame cannot release a previously observed red light.

    def stop_sign_callback(self, msg):
        if self.get_parameter('traffic_rule_source').value != 'camera':
            return
        if not self._fresh_detection(msg.header):
            return
        signs = [sign for sign in msg.road_signs
                 if sign.label.strip().lower() in ('stop', 'stop sign')
                 and sign.confidence >= 0.6 and sign.height > 0]
        if not signs:
            return
        sign = max(signs, key=lambda sign: sign.height)
        height = self.get_parameter('stop_sign_height_m').value
        distance = self.sign_focal_length * height / sign.height
        self.stop_sign_policy.observe(distance, self.get_clock().now().nanoseconds / 1e9)

    LIDAR_STOP_DISTANCE = 5.0
    LIDAR_RELEASE_DISTANCE = 8.0
    LIDAR_CLEARANCE_DURATION = 1.0
    LIDAR_TIMEOUT = 0.5

    def _lidar_stop_distance(self):
        # LiDAR measures from base_link; vehicle_gap is bumper-to-bumper.
        if self.get_parameter('traffic_rule_source').value == 'carla':
            scene = self.carla_scene
            if scene and scene.get('valid'):
                return float(scene.get('ego_front', 2.5)) + self.get_parameter('standstill_gap_m').value
        return self.LIDAR_STOP_DISTANCE

    def lidar_callback(self, msg: PointCloud2):
        """Track forward obstacles; missing returns cannot release a stopped car."""
        now = self.get_clock().now()
        if self._last_lidar_time is not None:
            age = (now - self._last_lidar_time).nanoseconds / 1e9
            if age < 0 or age > self.LIDAR_TIMEOUT:
                self._clearance_since = None
        self._last_lidar_time = now
        # Read XYZ using the advertised layout, including row padding/endianness.
        try:
            fields = {field.name: field for field in msg.fields}
            xyz = [fields[name] for name in ('x', 'y', 'z')]
            if (msg.width == 0 or msg.height == 0 or
                    any(field.datatype != 7 for field in xyz)):
                raise ValueError('No float32 XYZ points')
            dtype = np.dtype({'names': ['x', 'y', 'z'],
                              'formats': [('>' if msg.is_bigendian else '<') + 'f4'] * 3,
                              'offsets': [field.offset for field in xyz],
                              'itemsize': msg.point_step})
            cloud = np.ndarray((msg.height, msg.width), dtype=dtype,
                               buffer=bytes(msg.data),
                               strides=(msg.row_step, msg.point_step))
            xs, ys, zs = (cloud[name].ravel() for name in ('x', 'y', 'z'))
        except (KeyError, ValueError, TypeError):
            self.lidar_obstacle_distance = float('inf')
            self._clearance_since = None
            return

        pre = (np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) &
               (xs > 0.5) & (xs < 20.0) & (zs > 0.3))
        xs_f, ys_f = xs[pre], ys[pre]
        on_path = np.abs(ys_f) < 0.9
        path = self.current_path
        if not self.obstacle_stop_latched and path is not None and len(path.poses) > 1:
            wx = np.array([p.pose.position.x for p in path.poses])
            wy = np.array([p.pose.position.y for p in path.poses])
            ahead = (wx > 0.5) & (wx < 20.0)
            if ahead.any():
                dx = xs_f[:, None] - wx[ahead][None, :]
                dy = ys_f[:, None] - wy[ahead][None, :]
                on_path = (dx * dx + dy * dy).min(axis=1) < 0.6 ** 2
        elif self.obstacle_stop_latched:
            # Hold the obstacle corridor independently of replanning while stopped.
            on_path = np.abs(ys_f - self._stop_obstacle_y) < 0.9

        distance = float(xs_f[on_path].min()) if on_path.sum() >= 5 else float('inf')
        self.lidar_obstacle_distance = distance
        if distance < 12.0:
            self._was_moving_for_obstacle |= self.current_speed > 0.3
            self._hazard_distance = distance
            self._recent_lidar_hazard = True
            self._recent_lidar_y = float(np.median(ys_f[on_path & (xs_f <= distance + 0.5)]))
        if distance <= self._lidar_stop_distance():
            if not self.obstacle_stop_latched:
                self._stop_obstacle_y = float(np.median(ys_f[on_path & (xs_f <= distance + 0.5)]))
            self.obstacle_stop_latched = True
            self._clearance_since = None
        elif self.obstacle_stop_latched:
            # Require actual farther returns, rather than interpreting lost/empty
            # detections as clearance. A stale stream also breaks confirmation.
            scene_clear = True
            if self.get_parameter('traffic_rule_source').value == 'carla':
                data = self.carla_scene
                scene_clear = bool(data and data.get('valid') and
                                   (data.get('vehicle_gap') is None or data['vehicle_gap'] >= 5.0))
            confirmed_clear = (self.get_parameter('traffic_rule_source').value == 'carla' or
                               (np.isfinite(distance) and distance >= self.LIDAR_RELEASE_DISTANCE))
            if scene_clear and confirmed_clear:
                if self._clearance_since is None or now < self._clearance_since:
                    self._clearance_since = now
                if (now - self._clearance_since).nanoseconds / 1e9 >= self.LIDAR_CLEARANCE_DURATION:
                    self.obstacle_stop_latched = False
                    self._recent_lidar_hazard = False
                    self._was_moving_for_obstacle = False
                    self._clearance_since = None
            else:
                self._clearance_since = None

    def control_loop(self):
        """Main control loop executed at control_rate Hz."""
        if not self.enabled:
            self._publish_zero_control()
            return

        # Check if we have necessary data
        if self.current_odom is None:
            self.get_logger().warn(
                'No odometry received, holding brake',
                throttle_duration_sec=5.0
            )
            self._publish_zero_control()
            return

        if self.current_path is None or len(self.current_path.poses) < 2:
            self.get_logger().warn(
                'No usable path received, holding brake',
                throttle_duration_sec=5.0
            )
            self._publish_zero_control()
            return

        # A base_link path expires as the vehicle moves, even if its publisher dies.
        current_time = self.get_clock().now()
        received_age = (current_time - self.path_received_time).nanoseconds / 1e9
        stamp_age = (current_time.nanoseconds / 1e9 -
                     self.current_path.header.stamp.sec -
                     self.current_path.header.stamp.nanosec / 1e9)
        if (received_age < 0 or received_age > self.path_timeout or
                stamp_age < -self.path_timeout or stamp_age > self.path_timeout or
                self.current_path.header.frame_id != 'base_link'):
            self._publish_zero_control()
            return

        # Compute time step
        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time

        # Extract current state
        current_pose = self.current_odom.pose.pose
        current_speed = self.current_speed

        # Lateral control (steering)
        steer, cross_track_error = self.lateral_controller.compute_steering(
            current_pose,
            current_speed
        )

        # Hold the brake explicitly: PID integral/jerk state must not creep
        # forward at a zero speed target or after a detection dropout.
        if self.obstacle_stop_latched:
            self._publish_zero_control("LiDAR obstacle stop latched")
            return

        # If a detected lead disappears while we are stopping, hold instead
        # of treating the missing points as an open road.
        if self._recent_lidar_hazard and self._was_moving_for_obstacle and self.current_speed < 0.2:
            self.obstacle_stop_latched = True
            self._stop_obstacle_y = self._recent_lidar_y
            self._publish_zero_control()
            return

        SLOW_DIST = 12.0
        d = self.lidar_obstacle_distance
        if self._recent_lidar_hazard and not np.isfinite(d):
            self._hazard_distance = max(0.0, self._hazard_distance - max(0.0, current_speed) * max(0.0, dt))
            d = self._hazard_distance
        stop_distance = self._lidar_stop_distance()
        if d <= stop_distance:
            self.obstacle_stop_latched = True
            self._publish_zero_control()
            return
        if d < SLOW_DIST:
            ratio = (d - stop_distance) / (SLOW_DIST - stop_distance)
            self.longitudinal_controller.target_speed = self.target_speed * ratio
        else:
            self.longitudinal_controller.target_speed = self.target_speed

        source = self.get_parameter('traffic_rule_source').value
        if source == 'carla':
            data = self.carla_scene
            age = (current_time - self.carla_scene_received).nanoseconds / 1e9 if self.carla_scene_received is not None else float('inf')
            sim_age = current_time.nanoseconds / 1e9 - data.get('stamp', float('-inf')) if data else float('inf')
            if not data or not data.get('valid') or not 0 <= age <= 0.5 or not -0.5 <= sim_age <= 0.5:
                self._publish_zero_control("CARLA scene missing, stale, or invalid")
                return
            gap = data.get('vehicle_gap')
            if gap is not None:
                standstill_gap = self.get_parameter('standstill_gap_m').value
                if gap <= standstill_gap:
                    self._publish_zero_control(f"CARLA vehicle clearance <= {standstill_gap:.2f} m")
                    return
                limit = math.sqrt(2.0 * 1.5 * max(0.0, gap - standstill_gap))
                self.longitudinal_controller.target_speed = min(self.longitudinal_controller.target_speed, limit)
            if data.get('light_state') in ('Red', 'Yellow'):
                light_distance = data.get('light_distance')
                if light_distance is not None:
                    remaining = light_distance - data.get('ego_front', 2.5) - 2.0
                    if remaining <= 0.2:
                        self._publish_zero_control("CARLA red/yellow stop line")
                        return
                    self.longitudinal_controller.target_speed = min(
                        self.longitudinal_controller.target_speed, math.sqrt(3.0 * remaining))

        sign_limit, sign_brake = self.stop_sign_policy.limit(
            current_speed, current_time.nanoseconds / 1e9)
        if sign_brake:
            self._publish_zero_control("Stop-sign dwell")
            return
        if sign_limit is not None:
            self.longitudinal_controller.target_speed = min(
                self.longitudinal_controller.target_speed, sign_limit)

        # Longitudinal control (throttle/brake)
        throttle, brake, target_speed = (
            self.longitudinal_controller.compute_control(
                current_speed,
                self.current_objects,
                dt
            )
        )

        # Intersection / traffic light override — stop on red
        if source == 'camera' and (self.intersection_action == 'Wait' or self.traffic_light_red or self.camera_light_stop):
            throttle = 0.0
            brake = 1.0
            self.longitudinal_controller.reset()

        # Create and publish control message
        control_msg = VehicleControl()
        control_msg.header.stamp = current_time.to_msg()
        control_msg.header.frame_id = 'base_link'
        control_msg.throttle = float(throttle)
        control_msg.brake = float(brake)
        control_msg.steer = float(steer)
        control_msg.reverse = False

        self.control_pub.publish(control_msg)

        # Publish debug information
        self._publish_debug_info(
            current_speed,
            target_speed,
            steer,
            throttle,
            brake,
            cross_track_error
        )

    def _publish_zero_control(self, reason="Planning or state unavailable"):
        """Hold brake and reset speed control while planning is unavailable."""
        # Reset accumulated acceleration before resuming after invalid planning.
        self.longitudinal_controller.reset()
        self.last_control_time = self.get_clock().now()
        control_msg = VehicleControl()
        control_msg.header.stamp = self.last_control_time.to_msg()
        control_msg.header.frame_id = 'base_link'
        control_msg.throttle = 0.0
        control_msg.brake = 1.0
        control_msg.steer = 0.0
        control_msg.reverse = False

        self.control_pub.publish(control_msg)
        self.debug_pub.publish(String(data=f"BRAKE HOLD: {reason}; lidar_distance={self.lidar_obstacle_distance:.2f}"))

    def _publish_debug_info(
        self,
        current_speed: float,
        target_speed: float,
        steer: float,
        throttle: float,
        brake: float,
        cross_track_error: float
    ):
        """
        Publish debug information.

        Args:
            current_speed: Current vehicle speed (m/s).
            target_speed: Target speed (m/s).
            steer: Steering command.
            throttle: Throttle command.
            brake: Brake command.
            cross_track_error: Distance from path (m).
        """
        lead_info = ""
        if self.longitudinal_controller.lead_vehicle is not None:
            lead_info = (
                f" | Lead: {self.longitudinal_controller.lead_distance:.1f}m"
            )

        debug_msg = String()
        debug_msg.data = (
            f"Speed: {current_speed:.2f}/{target_speed:.2f} m/s | "
            f"Steer: {steer:.3f} | "
            f"Throttle: {throttle:.2f} | "
            f"Brake: {brake:.2f} | "
            f"CTE: {cross_track_error:.2f}m"
            f"{lead_info}"
        )

        self.debug_pub.publish(debug_msg)


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    node = AutonomousCruiseController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
