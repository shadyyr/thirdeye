from __future__ import annotations

import unittest

from thirdeye.models import CameraName, DirectionZone
from thirdeye.spatial.direction import DirectionMapper


class DirectionMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = DirectionMapper(
            {
                "left_boundary": 0.35,
                "right_boundary": 0.65,
                "camera_mappings": {
                    "front": {"left": "LEFT", "center": "FRONT", "right": "RIGHT"},
                    "rear": {"left": "RIGHT", "center": "REAR", "right": "LEFT"},
                },
            }
        )

    def test_front_maps_left_center_right(self) -> None:
        self.assertEqual(self.mapper.zone_for(CameraName.FRONT, 0.1), DirectionZone.LEFT)
        self.assertEqual(self.mapper.zone_for(CameraName.FRONT, 0.5), DirectionZone.FRONT)
        self.assertEqual(self.mapper.zone_for(CameraName.FRONT, 0.9), DirectionZone.RIGHT)

    def test_rear_orientation_is_configuration_driven(self) -> None:
        self.assertEqual(self.mapper.zone_for(CameraName.REAR, 0.1), DirectionZone.RIGHT)
        self.assertEqual(self.mapper.zone_for(CameraName.REAR, 0.5), DirectionZone.REAR)
        self.assertEqual(self.mapper.zone_for(CameraName.REAR, 0.9), DirectionZone.LEFT)

    def test_mirror_flips_the_camera_coordinate(self) -> None:
        self.assertEqual(
            self.mapper.zone_for(CameraName.FRONT, 0.1, mirror=True),
            DirectionZone.RIGHT,
        )


if __name__ == "__main__":
    unittest.main()
