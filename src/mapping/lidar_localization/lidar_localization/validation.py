import math

import numpy as np


def filter_valid_points(points: np.ndarray, min_points: int):
    finite_mask = np.isfinite(points).all(axis=1)
    filtered = points[finite_mask]
    return filtered, filtered.shape[0] >= min_points


def gnss_is_fresh(gnss_time_sec: float | None, now_sec: float, timeout_sec: float) -> bool:
    if gnss_time_sec is None:
        return False
    if not math.isfinite(gnss_time_sec):
        return False
    return (now_sec - gnss_time_sec) <= timeout_sec


def transform_is_finite(transform: np.ndarray) -> bool:
    return transform.shape == (4, 4) and np.isfinite(transform).all()


def yaw_from_transform(transform: np.ndarray) -> float:
    return math.atan2(transform[1, 0], transform[0, 0])


def normalize_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        return wrapped + 2.0 * math.pi
    return wrapped


def jump_exceeds_limits(
    previous_transform: np.ndarray,
    next_transform: np.ndarray,
    max_translation_m: float,
    max_rotation_rad: float,
) -> bool:
    translation_delta = np.linalg.norm(next_transform[:3, 3] - previous_transform[:3, 3])
    yaw_delta = abs(normalize_angle(yaw_from_transform(next_transform) - yaw_from_transform(previous_transform)))
    return translation_delta > max_translation_m or yaw_delta > max_rotation_rad
