from __future__ import annotations

import unittest

from thirdeye.config import load_config, merge_config
from thirdeye.dashboard.state import DashboardState
from thirdeye.main import run_pipeline
from thirdeye.models import DirectionZone


class DemoScenarioTests(unittest.TestCase):
    def run_scenario(self, name: str):
        config = merge_config(load_config(), {"simulation": {"scenario": name}})
        return run_pipeline(config, max_frames=None)

    def test_approaching_right_person_triggers_right_haptics(self) -> None:
        result = self.run_scenario("front_right_approach")

        self.assertGreater(result.frames_processed, 0)
        self.assertTrue(result.alert_zones)
        self.assertEqual(set(result.alert_zones), {DirectionZone.RIGHT})

    def test_stationary_person_stays_quiet(self) -> None:
        result = self.run_scenario("front_stationary")

        self.assertEqual(result.alert_zones, ())

    def test_person_moving_away_stays_quiet(self) -> None:
        result = self.run_scenario("front_moving_away")

        self.assertEqual(result.alert_zones, ())

    def test_approaching_person_beats_stationary_competitor(self) -> None:
        result = self.run_scenario("competing_people")

        self.assertTrue(result.alert_zones)
        self.assertEqual(set(result.alert_zones), {DirectionZone.RIGHT})

    def test_disappearing_person_resolves_to_stop(self) -> None:
        result = self.run_scenario("approach_then_disappear")

        self.assertIn(DirectionZone.FRONT, result.alert_zones)
        self.assertEqual(result.decisions[-1].command_zone, DirectionZone.STOP)
        self.assertEqual(result.decisions[-1].command_intensity, 0)

    def test_rear_left_mapping_is_exercised_end_to_end(self) -> None:
        result = self.run_scenario("rear_left_approach")

        self.assertTrue(result.alert_zones)
        self.assertEqual(set(result.alert_zones), {DirectionZone.LEFT})

    def test_dashboard_receives_replay_and_haptic_state(self) -> None:
        config = merge_config(
            load_config(),
            {"simulation": {"scenario": "front_right_approach"}},
        )
        dashboard = DashboardState()

        run_pipeline(config, max_frames=None, dashboard_state=dashboard)

        snapshot = dashboard.snapshot()
        self.assertEqual(snapshot["replay"]["scenario"], "front_right_approach")
        self.assertEqual(snapshot["active_haptic"]["zone"], "RIGHT")


if __name__ == "__main__":
    unittest.main()
