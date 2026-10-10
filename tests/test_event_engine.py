from __future__ import annotations

import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import TrackedEntity
from thirdeye.events.event_engine import EventEngine
from thirdeye.spatial.zone_manager import ZoneManager
from thirdeye.state.world_state import WorldState

FRAME = (1000, 1000)
ZONES = [
    {"id": "r", "name": "Restricted Zone A", "type": "RESTRICTED",
     "points": [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]},
    {"id": "m", "name": "Monitored Area", "type": "MONITORED",
     "points": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]},
]

OVERRIDES = {
    "ZONE_CONFIRM_FRAMES": 1,
    "ZONE_ANCHOR": "center",
    "DWELL_THRESHOLD_SECONDS": 10.0,
    "DWELL_REPEAT_SECONDS": 15.0,
    "CROWD_THRESHOLD": 3,
    "CROWD_CLEAR_BELOW": 2,
    "CROWD_CONFIRM_SECONDS": 1.5,
    "OCCUPANCY_CONFIRM_SECONDS": 1.0,
}


def track(tid: int, cx: float, cls: str = "person") -> TrackedEntity:
    return TrackedEntity(tid, cls, 0.9, (cx - 40, 260, cx + 40, 340), (cx, 300.0), 0.0)


def people(n: int, x: float, start: int = 1) -> list[TrackedEntity]:
    return [track(start + i, x) for i in range(n)]


class EventEngineTestBase(unittest.TestCase):
    def setUp(self) -> None:
        patchers = [mock.patch.object(settings, k, v) for k, v in OVERRIDES.items()]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.world = WorldState()
        self.world.frame_size = FRAME
        self.zm = ZoneManager(ZONES)
        self.engine = EventEngine(self.zm)

    def step(self, tracks, now: float) -> list[dict]:
        self.world.update(tracks, now)
        transitions = self.zm.update(self.world)
        return self.engine.update(self.world, transitions, now)

    def run_for(self, tracks, start: float, end: float, dt: float = 0.1) -> list[dict]:
        events = []
        t = start
        while t <= end + 1e-9:
            events += self.step(tracks, round(t, 3))
            t += dt
        return events

    @staticmethod
    def of_type(events, event_type):
        return [e for e in events if e["event_type"] == event_type]


class EntityAndZoneEventTests(EventEngineTestBase):
    def test_entity_entered_and_exited(self) -> None:
        ev = self.step([track(3, 200), track(5, 200, "backpack")], 0.0)
        msgs = [e["message"] for e in self.of_type(ev, "ENTITY_ENTERED")]
        self.assertEqual(msgs, ["Person #3 entered view", "Backpack #5 detected"])
        self.assertEqual(self.of_type(ev, "ENTITY_ENTERED")[1]["metadata"]["class_name"], "backpack")
        self.step([track(3, 200), track(5, 200, "backpack")], 4.0)
        ev = self.step([track(5, 200, "backpack")], 4.1)
        exited = self.of_type(ev, "ENTITY_EXITED")
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0]["message"], "Person #3 left view")
        self.assertEqual(exited[0]["entity_id"], 3)
        self.assertAlmostEqual(exited[0]["metadata"]["duration_visible"], 4.0)
        # zone exit is reported too, exactly once
        self.assertEqual(len(self.of_type(ev, "ZONE_EXITED")), 1)

    def test_zone_enter_exit_severity_and_dwell_message(self) -> None:
        ev = self.step([track(3, 750), track(5, 750, "backpack"), track(6, 200)], 0.0)
        entered = {e["entity_id"]: e for e in self.of_type(ev, "ZONE_ENTERED")}
        self.assertEqual(entered[3]["message"], "Person #3 entered Restricted Zone A")
        self.assertEqual(entered[3]["severity"], "alert")
        self.assertEqual(entered[3]["zone_id"], "r")
        self.assertEqual(entered[5]["severity"], "warning")
        self.assertEqual(entered[6]["severity"], "info")
        ev = self.step([track(3, 200), track(5, 750, "backpack"), track(6, 200)], 6.2)
        exited = self.of_type(ev, "ZONE_EXITED")
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0]["message"], "Person #3 left Restricted Zone A after 6.2 sec")
        self.assertAlmostEqual(exited[0]["metadata"]["dwell_seconds"], 6.2)
        self.assertEqual(len(self.of_type(ev, "ZONE_ENTERED")), 1)  # into monitored

    def test_messages_are_ascii(self) -> None:
        events = self.run_for(people(4, 750), 0.0, 20.0, dt=0.5)
        events += self.run_for([], 20.5, 25.0, dt=0.5)
        self.assertTrue(events)
        for e in events:
            e["message"].encode("ascii")


class DwellTests(EventEngineTestBase):
    def test_dwell_fires_once_then_repeats_at_interval(self) -> None:
        events = self.run_for([track(3, 750)], 0.0, 40.0)
        dwell = self.of_type(events, "DWELLING")
        times = [e["timestamp"] for e in dwell]
        self.assertEqual(len(dwell), 3)  # 10, 25, 40
        self.assertAlmostEqual(times[0], 10.0)
        self.assertAlmostEqual(times[1], 25.0)
        self.assertAlmostEqual(times[2], 40.0)
        self.assertEqual(dwell[0]["message"], "Person #3 dwelling in Restricted Zone A for 10.0 sec")
        self.assertEqual(dwell[0]["severity"], "warning")

    def test_monitored_dwell_is_notice_and_resets_on_exit(self) -> None:
        events = self.run_for([track(3, 200)], 0.0, 12.0)
        dwell = self.of_type(events, "DWELLING")
        self.assertEqual(len(dwell), 1)
        self.assertEqual(dwell[0]["severity"], "notice")
        # leave and come back: timer restarts
        self.step([track(3, 750)], 12.1)
        events = self.run_for([track(3, 200)], 12.2, 21.0)
        self.assertEqual(self.of_type(events, "DWELLING"), [])
        events = self.run_for([track(3, 200)], 21.1, 22.5)
        self.assertEqual(len(self.of_type(events, "DWELLING")), 1)

    def test_objects_do_not_dwell(self) -> None:
        events = self.run_for([track(5, 750, "backpack")], 0.0, 20.0)
        self.assertEqual(self.of_type(events, "DWELLING"), [])


class OccupancyTests(EventEngineTestBase):
    def test_no_event_at_startup_and_debounce(self) -> None:
        events = self.run_for([], 0.0, 3.0)
        self.assertEqual(self.of_type(events, "OCCUPANCY_CHANGED"), [])
        # brief blip of 2 people shorter than confirm window -> nothing
        events = self.run_for(people(2, 200), 3.1, 3.5)
        events += self.run_for([], 3.6, 6.0)
        self.assertEqual(self.of_type(events, "OCCUPANCY_CHANGED"), [])
        self.assertEqual(self.engine.occupancy_total, 0)

    def test_increase_then_decrease(self) -> None:
        events = self.run_for(people(3, 200), 0.0, 2.0)
        occ = self.of_type(events, "OCCUPANCY_CHANGED")
        self.assertEqual([e["message"] for e in occ], ["Occupancy increased from 0 to 3"])
        self.assertAlmostEqual(occ[0]["timestamp"], 1.0)
        events = self.run_for(people(6, 200), 2.1, 4.0)
        self.assertEqual([e["message"] for e in self.of_type(events, "OCCUPANCY_CHANGED")],
                         ["Occupancy increased from 3 to 6"])
        events = self.run_for(people(2, 200), 4.1, 6.0)
        occ = self.of_type(events, "OCCUPANCY_CHANGED")
        self.assertEqual([e["message"] for e in occ], ["Occupancy decreased from 6 to 2"])
        self.assertEqual(occ[0]["metadata"]["delta"], -4)
        self.assertEqual(self.engine.occupancy_total, 2)

    def test_objects_not_counted(self) -> None:
        events = self.run_for([track(9, 200, "suitcase")], 0.0, 3.0)
        self.assertEqual(self.of_type(events, "OCCUPANCY_CHANGED"), [])


class CrowdingTests(EventEngineTestBase):
    def test_crowding_start_and_clear_with_hysteresis(self) -> None:
        events = self.run_for(people(3, 750), 0.0, 1.4)
        self.assertEqual(self.of_type(events, "CROWDING_STARTED"), [])
        events = self.run_for(people(3, 750), 1.5, 2.0)
        started = self.of_type(events, "CROWDING_STARTED")
        self.assertEqual({e["zone_id"] for e in started}, {"r", None})
        msgs = {e["zone_id"]: e["message"] for e in started}
        self.assertEqual(msgs["r"], "Crowd threshold exceeded in Restricted Zone A - occupancy 3")
        self.assertEqual(msgs[None], "Crowd threshold exceeded in the monitored view - occupancy 3")
        self.assertTrue(all(e["severity"] == "alert" for e in started))
        self.assertEqual(self.engine.crowded_zones, {"r", None})
        # drop to 2: below threshold but not below CLEAR_BELOW=2 -> still crowded
        events = self.run_for(people(2, 750), 2.1, 6.0)
        self.assertEqual(self.of_type(events, "CROWDING_CLEARED"), [])
        self.assertEqual(self.of_type(events, "CROWDING_STARTED"), [])
        # drop to 1 briefly, then back to 2 -> no clear
        events = self.run_for(people(1, 750), 6.1, 6.8)
        events += self.run_for(people(2, 750), 6.9, 7.5)
        self.assertEqual(self.of_type(events, "CROWDING_CLEARED"), [])
        # sustained 1 -> clears once per key
        events = self.run_for(people(1, 750), 7.6, 10.0)
        cleared = self.of_type(events, "CROWDING_CLEARED")
        self.assertEqual(sorted(cleared, key=lambda e: str(e["zone_id"]))[0]["severity"], "info")
        self.assertEqual({e["zone_id"] for e in cleared}, {"r", None})
        self.assertEqual(len(cleared), 2)
        self.assertEqual(self.engine.crowded_zones, set())

    def test_zone_occupancy(self) -> None:
        self.step(people(2, 750) + people(1, 200, start=10) + [track(20, 750, "backpack")], 0.0)
        self.assertEqual(self.engine.zone_occupancy(self.world), {"r": 2, "m": 1})


class AlertTests(EventEngineTestBase):
    def test_restricted_alerts(self) -> None:
        self.step([track(3, 750), track(4, 200), track(5, 750, "backpack")], 0.0)
        self.step([track(3, 750), track(4, 200), track(5, 750, "backpack")], 4.2)
        alerts = self.engine.active_alerts(self.world, 4.2)
        self.assertEqual(len(alerts), 1)
        a = alerts[0]
        self.assertEqual(a.title, "RESTRICTED ZONE ENTRY")
        self.assertEqual(a.lines, ["Person #3", "In Restricted Zone A for 4.2 sec"])
        self.assertEqual(a.since, 0.0)
        self.assertEqual(a.severity, "alert")
        self.assertEqual(a.entity_ids, [3])
        self.assertEqual(a.zone_id, "r")
        self.step([track(3, 200), track(4, 200)], 5.0)
        self.assertEqual(self.engine.active_alerts(self.world, 5.0), [])

    def test_crowding_alerts(self) -> None:
        self.run_for(people(4, 200), 0.0, 2.0)
        alerts = [a for a in self.engine.active_alerts(self.world, 2.0) if a.title == "CROWDING"]
        self.assertEqual({a.zone_id for a in alerts}, {"m", None})
        by_zone = {a.zone_id: a for a in alerts}
        self.assertEqual(by_zone["m"].lines, ["Monitored Area", "Occupancy: 4"])
        self.assertEqual(by_zone[None].lines, ["the monitored view", "Occupancy: 4"])


if __name__ == "__main__":
    unittest.main()
