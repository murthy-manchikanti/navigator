import unittest

import numpy as np

from lidar_localization.validation import (
    filter_valid_points,
    gnss_is_fresh,
    jump_exceeds_limits,
    transform_is_finite,
)


class ValidationTests(unittest.TestCase):
    def test_filter_valid_points_rejects_non_finite_and_min_points(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 1.0], [2.0, 2.0, 2.0]])
        filtered, ok = filter_valid_points(points, min_points=3)
        self.assertEqual(filtered.shape[0], 2)
        self.assertFalse(ok)

    def test_gnss_freshness_requires_recent_timestamp(self):
        self.assertFalse(gnss_is_fresh(None, now_sec=10.0, timeout_sec=0.5))
        self.assertTrue(gnss_is_fresh(9.7, now_sec=10.0, timeout_sec=0.5))
        self.assertFalse(gnss_is_fresh(9.1, now_sec=10.0, timeout_sec=0.5))

    def test_transform_is_finite_checks_shape_and_values(self):
        self.assertTrue(transform_is_finite(np.eye(4)))
        bad = np.eye(4)
        bad[0, 0] = np.inf
        self.assertFalse(transform_is_finite(bad))
        self.assertFalse(transform_is_finite(np.eye(3)))

    def test_jump_limits_reject_large_translation_or_rotation(self):
        prev = np.eye(4)
        next_translation = np.eye(4)
        next_translation[0, 3] = 5.0
        self.assertTrue(
            jump_exceeds_limits(prev, next_translation, max_translation_m=3.0, max_rotation_rad=0.5)
        )

        next_rotation = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(
            jump_exceeds_limits(prev, next_rotation, max_translation_m=3.0, max_rotation_rad=0.5)
        )


if __name__ == '__main__':
    unittest.main()
