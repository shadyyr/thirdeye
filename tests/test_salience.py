from __future__ import annotations

import unittest

from tests.helpers import make_track
from thirdeye.models import DirectionZone, MotionState, ScoredEvent
from thirdeye.salience.scorer import PrioritySelector, RelevanceScorer


class SalienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = RelevanceScorer(
            {
                "class_weights": {"person": 1.0},
                "weights": {
                    "class": 0.1,
                    "approach": 0.35,
                    "closeness": 0.2,
                    "centrality": 0.15,
                    "motion": 0.1,
                    "persistence": 0.1,
                },
                "close_area_fraction": 0.16,
                "persistence_frames": 5,
            }
        )

    def test_approaching_close_person_scores_higher_than_distant_stationary_person(self) -> None:
        urgent = make_track(track_id="front-001", areas=[0.05, 0.07, 0.1, 0.16], center_x=0.5)
        urgent.motion_state = MotionState.APPROACHING
        distant = make_track(track_id="rear-001", areas=[0.01, 0.01, 0.01, 0.01], center_x=0.5)
        distant.motion_state = MotionState.STATIONARY

        self.assertGreater(
            self.scorer.score(urgent, DirectionZone.FRONT).relevance_score,
            self.scorer.score(distant, DirectionZone.REAR).relevance_score,
        )

    def test_priority_selects_best_eligible_event(self) -> None:
        lower = ScoredEvent(make_track(track_id="front-001"), DirectionZone.FRONT, 0.55, {})
        higher = ScoredEvent(make_track(track_id="rear-001"), DirectionZone.REAR, 0.75, {})
        selector = PrioritySelector(alert_threshold=0.45, hold_seconds=1.0, switch_margin=0.12)

        selected = selector.select([lower, higher], now=10.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.track_id, "rear-001")

    def test_priority_holds_current_object_without_material_improvement(self) -> None:
        current = ScoredEvent(make_track(track_id="front-001"), DirectionZone.FRONT, 0.70, {})
        nearby = ScoredEvent(make_track(track_id="rear-001"), DirectionZone.REAR, 0.76, {})
        selector = PrioritySelector(alert_threshold=0.45, hold_seconds=1.0, switch_margin=0.12)
        selector.select([current], now=10.0)

        selected = selector.select([current, nearby], now=10.2)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.track_id, "front-001")


if __name__ == "__main__":
    unittest.main()
