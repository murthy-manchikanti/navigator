"""Exercise the real route-end callback and verify its published stop goal."""

from unittest.mock import Mock

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path

from costs.route_costmap_node import RouteCostmapNode


@pytest.mark.parametrize('points', [[], [1.0], [1.0, 1.0]])
def test_route_end_keeps_publishing_without_crashing(points):
    rclpy.init(args=[
        '--ros-args', '-p', 'global_config:=/navigator/param/global_parameters.yaml',
    ])
    node = RouteCostmapNode()
    try:
        node.clock.clock.sec = 10
        transform = TransformStamped()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.rotation.w = 1.0
        node.tf_buffer.set_transform_static(transform, 'test')
        path = Path()
        path.header.frame_id = 'map'
        for x in points:
            pose = PoseStamped()
            pose.pose.position.x = x
            path.poses.append(pose)
        node.routeCb(path)
        node.path_goal_pub = Mock()
        node.route_dist_grid_pub = Mock()
        node.goal_marker_pub = Mock()

        for stamp in (10, 11):
            node.clock.clock.sec = stamp
            node.buildRouteCostmap()

        assert node.path_goal_pub.publish.call_count == 2
        goal = node.path_goal_pub.publish.call_args.args[0]
        assert goal.pose.position.x == goal.pose.position.y == 0.0
        assert goal.header.stamp.sec == 11
        grid = node.route_dist_grid_pub.publish.call_args.args[0]
        assert len(grid.data) == grid.info.width * grid.info.height > 0
        assert grid.header.stamp.sec == 11
    finally:
        node.destroy_node()
        rclpy.shutdown()
