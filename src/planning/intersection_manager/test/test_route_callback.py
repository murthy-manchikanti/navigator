"""Regression coverage for local paths shorter than the turn lookahead."""

from types import SimpleNamespace

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from intersection_manager.intersection_manager import IntersectionManager


@pytest.mark.parametrize('length', [0, 1, 9, 10, 11, 25, 109, 110, 120])
def test_short_and_full_paths_do_not_index_past_end(length):
    state = SimpleNamespace(behavior_through_intersection='LeftTurn')
    path = Path()
    for i in range(length):
        pose = PoseStamped()
        pose.pose.position.x = float(i)
        path.poses.append(pose)

    IntersectionManager.routeCallback(state, path)

    assert state.route == path.poses
    assert state.behavior_through_intersection == (
        None if length <= 10 else 'RightTurn'
    )


@pytest.mark.parametrize('direction, expected', [(-1, 'LeftTurn'), (0, 'Straight')])
def test_existing_turn_classification_is_preserved(direction, expected):
    state = SimpleNamespace(behavior_through_intersection=None)
    path = Path()
    for i in range(120):
        pose = PoseStamped()
        pose.pose.position.x = float(i * direction)
        path.poses.append(pose)
    IntersectionManager.routeCallback(state, path)
    assert state.behavior_through_intersection == expected
