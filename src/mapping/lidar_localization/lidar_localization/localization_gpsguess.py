import math
import os
import queue

import numpy as np
import open3d as o3d
import rclpy
import ros2_numpy as rnp
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from lidar_localization.validation import (
    filter_valid_points,
    gnss_is_fresh,
    jump_exceeds_limits,
    transform_is_finite,
)


def matrix_from_pose(position, orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError('Invalid quaternion norm')
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = 1.0 - (yy + zz)
    transform[0, 1] = xy - wz
    transform[0, 2] = xz + wy
    transform[1, 0] = xy + wz
    transform[1, 1] = 1.0 - (xx + zz)
    transform[1, 2] = yz - wx
    transform[2, 0] = xz - wy
    transform[2, 1] = yz + wx
    transform[2, 2] = 1.0 - (xx + yy)
    transform[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float64)
    return transform


def quaternion_from_matrix(transform: np.ndarray):
    trace = transform[0, 0] + transform[1, 1] + transform[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (transform[2, 1] - transform[1, 2]) * s
        y = (transform[0, 2] - transform[2, 0]) * s
        z = (transform[1, 0] - transform[0, 1]) * s
    elif transform[0, 0] > transform[1, 1] and transform[0, 0] > transform[2, 2]:
        s = 2.0 * math.sqrt(1.0 + transform[0, 0] - transform[1, 1] - transform[2, 2])
        w = (transform[2, 1] - transform[1, 2]) / s
        x = 0.25 * s
        y = (transform[0, 1] + transform[1, 0]) / s
        z = (transform[0, 2] + transform[2, 0]) / s
    elif transform[1, 1] > transform[2, 2]:
        s = 2.0 * math.sqrt(1.0 + transform[1, 1] - transform[0, 0] - transform[2, 2])
        w = (transform[0, 2] - transform[2, 0]) / s
        x = (transform[0, 1] + transform[1, 0]) / s
        y = 0.25 * s
        z = (transform[1, 2] + transform[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + transform[2, 2] - transform[0, 0] - transform[1, 1])
        w = (transform[1, 0] - transform[0, 1]) / s
        x = (transform[0, 2] + transform[2, 0]) / s
        y = (transform[1, 2] + transform[2, 1]) / s
        z = 0.25 * s

    return x, y, z, w


class GpsGuessLocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_gpsguess')

        self.declare_parameter('map_path', '')
        self.declare_parameter('pointcloud_topic', '/lidar')
        self.declare_parameter('gnss_topic', '/gnss/odometry_raw')
        self.declare_parameter('localized_pose_topic', '/localized_pose')
        self.declare_parameter('status_topic', '/localization/status')
        self.declare_parameter('valid_topic', '/localization/is_valid')
        self.declare_parameter('global_map_topic', '/localization/global_map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'hero')
        self.declare_parameter('gnss_timeout_sec', 0.5)
        self.declare_parameter('min_points', 200)
        self.declare_parameter('voxel_size_m', 0.5)
        self.declare_parameter('map_crop_radius_m', 80.0)
        self.declare_parameter('max_pose_jump_m', 10.0)
        self.declare_parameter('max_yaw_jump_rad', 0.8)
        self.declare_parameter('icp_max_correspondence_m', 2.0)
        self.declare_parameter('icp_max_iterations', 20)
        self.declare_parameter('min_registration_fitness', 0.15)
        self.declare_parameter('processing_period_sec', 0.1)
        self.declare_parameter('map_publish_period_sec', 1.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.min_points = int(self.get_parameter('min_points').value)
        self.voxel_size = float(self.get_parameter('voxel_size_m').value)
        self.map_crop_radius_m = float(self.get_parameter('map_crop_radius_m').value)
        self.gnss_timeout_sec = float(self.get_parameter('gnss_timeout_sec').value)
        self.max_pose_jump_m = float(self.get_parameter('max_pose_jump_m').value)
        self.max_yaw_jump_rad = float(self.get_parameter('max_yaw_jump_rad').value)
        self.icp_max_correspondence_m = float(self.get_parameter('icp_max_correspondence_m').value)
        self.icp_max_iterations = int(self.get_parameter('icp_max_iterations').value)
        self.min_registration_fitness = float(self.get_parameter('min_registration_fitness').value)

        map_path = str(self.get_parameter('map_path').value)
        self._load_map(map_path)

        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter('localized_pose_topic').value, 10)
        self.status_pub = self.create_publisher(DiagnosticStatus, self.get_parameter('status_topic').value, 10)
        self.valid_pub = self.create_publisher(Bool, self.get_parameter('valid_topic').value, 10)
        self.map_pub = self.create_publisher(PointCloud2, self.get_parameter('global_map_topic').value, 1)

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('pointcloud_topic').value,
            self.cloud_cb,
            10,
        )
        self.gnss_sub = self.create_subscription(
            Odometry,
            self.get_parameter('gnss_topic').value,
            self.gnss_cb,
            10,
        )

        self.scan_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.latest_gnss_transform = None
        self.latest_gnss_time = None
        self.last_valid_transform = None

        self.processing_timer = self.create_timer(
            float(self.get_parameter('processing_period_sec').value),
            self.processing_cb,
        )
        self.map_timer = self.create_timer(
            float(self.get_parameter('map_publish_period_sec').value),
            self.publish_map,
        )

        self.publish_status(DiagnosticStatus.WARN, 'Waiting for GNSS and LiDAR inputs')

    def _load_map(self, map_path: str):
        if not map_path:
            raise RuntimeError("Parameter 'map_path' is required and cannot be empty")
        if not os.path.isfile(map_path):
            raise RuntimeError(f"Configured map_path does not exist: {map_path}")

        pcd = o3d.io.read_point_cloud(map_path)
        if pcd.is_empty():
            raise RuntimeError(f"Loaded map is empty: {map_path}")

        map_points = np.asarray(pcd.points)
        map_points, valid = filter_valid_points(map_points, self.min_points)
        if not valid:
            raise RuntimeError(f"Map has insufficient finite points for localization: {map_points.shape[0]}")

        self.map_points = map_points
        self.map_kdtree = cKDTree(self.map_points)
        map_cloud = np.zeros(
            self.map_points.shape[0],
            dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)],
        )
        map_cloud['x'] = self.map_points[:, 0].astype(np.float32)
        map_cloud['y'] = self.map_points[:, 1].astype(np.float32)
        map_cloud['z'] = self.map_points[:, 2].astype(np.float32)
        self.global_map_msg = rnp.msgify(PointCloud2, map_cloud)

    def cloud_cb(self, msg: PointCloud2):
        try:
            cloud = rnp.numpify(msg)
            points = np.column_stack((cloud['x'], cloud['y'], cloud['z']))
        except Exception as exc:
            self.publish_status(DiagnosticStatus.ERROR, f'Failed to decode LiDAR pointcloud: {exc}')
            self.valid_pub.publish(Bool(data=False))
            return

        points, valid = filter_valid_points(points, self.min_points)
        if not valid:
            self.publish_status(
                DiagnosticStatus.ERROR,
                f'Received invalid/insufficient LiDAR points ({points.shape[0]} valid points)',
            )
            self.valid_pub.publish(Bool(data=False))
            return

        if self.scan_queue.full():
            try:
                self.scan_queue.get_nowait()
            except queue.Empty:
                pass

        self.scan_queue.put_nowait(points)

    def gnss_cb(self, msg: Odometry):
        try:
            transform = matrix_from_pose(msg.pose.pose.position, msg.pose.pose.orientation)
        except ValueError as exc:
            self.publish_status(DiagnosticStatus.ERROR, f'Invalid GNSS pose quaternion: {exc}')
            self.valid_pub.publish(Bool(data=False))
            return

        if not transform_is_finite(transform):
            self.publish_status(DiagnosticStatus.ERROR, 'GNSS transform contained non-finite values')
            self.valid_pub.publish(Bool(data=False))
            return

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            self.latest_gnss_time = self.get_clock().now().nanoseconds * 1e-9
        else:
            self.latest_gnss_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.latest_gnss_transform = transform

    def _crop_map(self, center_xyz: np.ndarray) -> np.ndarray:
        indices = self.map_kdtree.query_ball_point(center_xyz, self.map_crop_radius_m)
        return self.map_points[indices] if indices else np.empty((0, 3))

    def processing_cb(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if not gnss_is_fresh(self.latest_gnss_time, now, self.gnss_timeout_sec):
            self.publish_status(DiagnosticStatus.ERROR, 'GNSS estimate unavailable or stale')
            self.valid_pub.publish(Bool(data=False))
            return

        try:
            scan = self.scan_queue.get_nowait()
        except queue.Empty:
            return

        initial_guess = np.array(self.latest_gnss_transform, copy=True)
        cropped_map = self._crop_map(initial_guess[:3, 3])

        cropped_map, map_ok = filter_valid_points(cropped_map, self.min_points)
        if not map_ok:
            self.publish_status(DiagnosticStatus.ERROR, 'Insufficient map points near GNSS guess')
            self.valid_pub.publish(Bool(data=False))
            return

        scan_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan))
        map_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped_map))

        scan_cloud = scan_cloud.voxel_down_sample(self.voxel_size)
        map_cloud = map_cloud.voxel_down_sample(self.voxel_size)

        scan_np = np.asarray(scan_cloud.points)
        map_np = np.asarray(map_cloud.points)
        if scan_np.shape[0] < self.min_points or map_np.shape[0] < self.min_points:
            self.publish_status(DiagnosticStatus.ERROR, 'Insufficient downsampled points for registration')
            self.valid_pub.publish(Bool(data=False))
            return

        result = o3d.pipelines.registration.registration_icp(
            scan_cloud,
            map_cloud,
            self.icp_max_correspondence_m,
            initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iterations),
        )

        transform = result.transformation
        if result.fitness < self.min_registration_fitness:
            self.publish_status(
                DiagnosticStatus.ERROR,
                f'Registration fitness below threshold ({result.fitness:.4f} < {self.min_registration_fitness:.4f})',
            )
            self.valid_pub.publish(Bool(data=False))
            return

        if not transform_is_finite(transform):
            self.publish_status(DiagnosticStatus.ERROR, 'Registration produced non-finite transform')
            self.valid_pub.publish(Bool(data=False))
            return

        if self.last_valid_transform is not None and jump_exceeds_limits(
            self.last_valid_transform,
            transform,
            self.max_pose_jump_m,
            self.max_yaw_jump_rad,
        ):
            self.publish_status(DiagnosticStatus.ERROR, 'Rejected localization pose jump beyond configured limits')
            self.valid_pub.publish(Bool(data=False))
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = float(transform[0, 3])
        pose.pose.position.y = float(transform[1, 3])
        pose.pose.position.z = float(transform[2, 3])
        x, y, z, w = quaternion_from_matrix(transform)
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w

        self.pose_pub.publish(pose)
        self.valid_pub.publish(Bool(data=True))
        self.last_valid_transform = transform
        self.publish_status(
            DiagnosticStatus.OK,
            'Localization update accepted',
            {
                'fitness': f'{result.fitness:.4f}',
                'rmse': f'{result.inlier_rmse:.4f}',
                'map_points': str(map_np.shape[0]),
                'scan_points': str(scan_np.shape[0]),
            },
        )

    def publish_status(self, level: int, message: str, values: dict | None = None):
        status = DiagnosticStatus()
        status.name = 'lidar_localization'
        status.level = level
        status.message = message
        for key, value in (values or {}).items():
            kv = KeyValue()
            kv.key = key
            kv.value = value
            status.values.append(kv)
        self.status_pub.publish(status)

    def publish_map(self):
        msg = self.global_map_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GpsGuessLocalizationNode()
    except Exception as exc:
        temp_logger = rclpy.logging.get_logger('localization_gpsguess')
        temp_logger.fatal(f'Localization startup failed: {exc}')
        rclpy.shutdown()
        raise

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
