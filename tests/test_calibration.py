from __future__ import annotations

import unittest

from thirdeye.config import load_config, merge_config
from thirdeye.models import CameraName, DirectionZone
from thirdeye.spatial.calibration import orientation_checks


class OrientationCalibrationTests(unittest.TestCase):
    def test_default_rear_left_maps_to_left(self) -> None:
        checks = orientation_checks(load_config())
        rear_left = next(
            check
            for check in checks
            if check.camera == CameraName.REAR and check.camera_position == "left"
        )

        self.assertEqual(rear_left.expected_zone, DirectionZone.LEFT)

    def test_mirrored_camera_changes_calibration_output(self) -> None:
        config = merge_config(load_config(), {"cameras": {"rear": {"mirror_horizontal": True}}})
        checks = orientation_checks(config)
        rear_left = next(
            check
            for check in checks
            if check.camera == CameraName.REAR and check.camera_position == "left"
        )

        self.assertEqual(rear_left.expected_zone, DirectionZone.RIGHT)


if __name__ == "__main__":
    unittest.main()
