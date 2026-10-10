from __future__ import annotations

import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import Detection
from thirdeye.perception import detection_filter
from thirdeye.perception.detection_filter import (
    DetectionFilterStats,
    box_overlap,
    family,
    filter_detections,
)


def P(conf: float, box) -> Detection:
    return Detection("person", conf, tuple(float(v) for v in box))


class DetectionFilterTests(unittest.TestCase):
    def test_family(self) -> None:
        for name in ("backpack", "handbag", "suitcase"):
            self.assertEqual(family(name), "portable")
        self.assertEqual(family("person"), "person")

    def test_box_overlap(self) -> None:
        iou, contain = box_overlap((0, 0, 10, 10), (0, 0, 10, 5))
        self.assertAlmostEqual(iou, 0.5)
        self.assertAlmostEqual(contain, 1.0)
        self.assertEqual(box_overlap((0, 0, 10, 10), (20, 20, 30, 30)), (0.0, 0.0))

    def test_nested_upper_body_box_suppressed(self) -> None:
        full = P(0.546, (1115, 151, 1266, 467))
        upper = P(0.222, (1115, 152, 1265, 368))
        self.assertEqual(filter_detections([upper, full]), [full])
        self.assertEqual(detection_filter.last_stats.suppressed_person, 1)

    def test_person_plus_bag_wide_box_suppressed(self) -> None:
        tight = P(0.599, (307, 0, 474, 570))
        wide = P(0.268, (170, 2, 475, 567))
        self.assertEqual(filter_detections([tight, wide]), [tight])

    def test_side_by_side_people_kept(self) -> None:
        a = P(0.8, (100, 100, 200, 400))
        b = P(0.7, (160, 100, 260, 400))  # IoU 0.25
        self.assertEqual(filter_detections([a, b]), [a, b])

    def test_far_small_person_inside_near_person_kept(self) -> None:
        near = P(0.85, (300, 0, 600, 700))
        far = P(0.5, (380, 100, 480, 350))  # fully contained, IoU ~0.12
        iou, contain = box_overlap(near.bbox, far.bbox)
        self.assertLess(iou, settings.DEDUP_PERSON_CONTAIN_MIN_IOU)
        self.assertEqual(contain, 1.0)
        self.assertEqual(filter_detections([near, far]), [near, far])

    def test_one_bag_with_three_labels_becomes_one_box(self) -> None:
        backpack = Detection("backpack", 0.48, (500, 400, 580, 520))
        handbag = Detection("handbag", 0.41, (503, 404, 580, 518))
        suitcase = Detection("suitcase", 0.31, (498, 398, 584, 524))
        stats = DetectionFilterStats()
        out = filter_detections([suitcase, handbag, backpack], stats)
        self.assertEqual(out, [backpack])  # survivor keeps its own label
        self.assertEqual(stats.suppressed_object, 2)
        self.assertEqual(stats.suppressed, 2)
        self.assertEqual(stats.total_in, 3)

    def test_two_separate_bags_kept(self) -> None:
        a = Detection("backpack", 0.5, (100, 400, 180, 500))
        b = Detection("handbag", 0.4, (300, 420, 360, 500))
        self.assertEqual(filter_detections([a, b]), [a, b])

    def test_bag_inside_person_box_kept(self) -> None:
        p = P(0.8, (300, 0, 480, 570))
        b = Detection("backpack", 0.4, (330, 150, 420, 300))
        self.assertEqual(filter_detections([p, b]), [p, b])
        self.assertEqual(detection_filter.last_stats.suppressed, 0)

    def test_disabled_returns_input_unchanged(self) -> None:
        dets = [P(0.546, (1115, 151, 1266, 467)), P(0.222, (1115, 152, 1265, 368))]
        with mock.patch.object(settings, "DEDUP_ENABLED", False):
            self.assertIs(filter_detections(dets), dets)

    def test_settings_read_at_call_time(self) -> None:
        a = P(0.8, (100, 100, 200, 400))
        b = P(0.7, (120, 100, 220, 400))  # IoU ~0.67
        self.assertEqual(filter_detections([a, b]), [a])
        with mock.patch.object(settings, "DEDUP_PERSON_IOU", 0.9), \
                mock.patch.object(settings, "DEDUP_PERSON_CONTAIN", 0.95):
            self.assertEqual(filter_detections([a, b]), [a, b])

    def test_empty_and_single(self) -> None:
        self.assertEqual(filter_detections([]), [])
        d = P(0.5, (0, 0, 10, 10))
        self.assertEqual(filter_detections([d]), [d])


if __name__ == "__main__":
    unittest.main()
