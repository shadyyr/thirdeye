from __future__ import annotations

import unittest

from thirdeye.models import MotionState
from thirdeye.spatial.motion import classify_area_history


class MotionClassificationTests(unittest.TestCase):
    def classify(self, areas: list[float]) -> MotionState:
        return classify_area_history(
            areas, minimum_samples=4, smoothing_window=6, area_change_threshold=0.12
        )

    def test_needs_history_before_state(self) -> None:
        self.assertEqual(self.classify([0.05, 0.06, 0.07]), MotionState.UNKNOWN)

    def test_consistent_growth_is_approaching(self) -> None:
        self.assertEqual(self.classify([0.04, 0.05, 0.07, 0.08]), MotionState.APPROACHING)

    def test_consistent_shrinkage_is_moving_away(self) -> None:
        self.assertEqual(self.classify([0.12, 0.1, 0.08, 0.07]), MotionState.MOVING_AWAY)

    def test_minor_change_is_stationary(self) -> None:
        self.assertEqual(self.classify([0.08, 0.081, 0.08, 0.079]), MotionState.STATIONARY)


if __name__ == "__main__":
    unittest.main()
