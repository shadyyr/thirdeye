from __future__ import annotations

import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import (
    ALERT,
    ASSOCIATION_CANDIDATE,
    ASSOCIATION_CREATED,
    ASSOCIATION_RESTORED,
    INFO,
    NOTICE,
    OBJECT_LOST,
    OBJECT_MOVED,
    OBJECT_SEPARATING,
    UNATTENDED_OBJECT,
    WARNING,
    TrackedEntity,
)
from thirdeye.relationships.object_association import (
    ASSOCIATED,
    SEPARATING,
    UNASSOCIATED,
    UNATTENDED,
    AssociationEngine,
)
from thirdeye.state.world_state import WorldState

FPS = 20
DT = 1.0 / FPS
PERSON_HALF_W, PERSON_HALF_H = 50.0, 150.0  # 100 x 300 px person box
BAG_HALF = 30.0  # 60 x 60 px bag box


class Scene:
    """Drives a real WorldState + AssociationEngine with synthetic tracks."""

    def __init__(self) -> None:
        self.world = WorldState()
        self.engine = AssociationEngine()
        self.t = 1000.0
        self.people: dict[int, list[float]] = {}
        self.objects: dict[int, list[float]] = {}
        self.missed: dict[int, int] = {}
        self.events: list[dict] = []

    def _tracks(self) -> list[TrackedEntity]:
        tracks = []
        for tid, (cx, cy) in self.people.items():
            bbox = (cx - PERSON_HALF_W, cy - PERSON_HALF_H, cx + PERSON_HALF_W, cy + PERSON_HALF_H)
            tracks.append(TrackedEntity(tid, "person", 0.9, bbox, (cx, cy), self.t, self.missed.get(tid, 0)))
        for tid, (cx, cy) in self.objects.items():
            bbox = (cx - BAG_HALF, cy - BAG_HALF, cx + BAG_HALF, cy + BAG_HALF)
            tracks.append(TrackedEntity(tid, "backpack", 0.8, bbox, (cx, cy), self.t, self.missed.get(tid, 0)))
        return tracks

    def run(self, seconds: float, velocities: dict[int, tuple[float, float]] | None = None) -> list[dict]:
        velocities = velocities or {}
        new_events: list[dict] = []
        for _ in range(int(round(seconds * FPS))):
            self.t += DT
            for tid, (vx, vy) in velocities.items():
                pos = self.people.get(tid) or self.objects.get(tid)
                if pos is not None and self.missed.get(tid, 0) == 0:
                    pos[0] += vx * DT
                    pos[1] += vy * DT
            for tid in self.missed:
                if self.missed[tid] > 0:
                    self.missed[tid] += 1
            self.world.update(self._tracks(), self.t)
            evs = self.engine.update(self.world, self.t)
            new_events.extend(evs)
        self.events.extend(new_events)
        return new_events

    def types(self, events: list[dict] | None = None) -> list[str]:
        return [e["event_type"] for e in (self.events if events is None else events)]

    def alerts(self):
        return self.engine.active_alerts(self.world, self.t)

    # Common setup: Person #2 standing with Backpack #5 at their feet.
    def associate(self) -> None:
        self.people[2] = [300.0, 350.0]
        self.objects[5] = [380.0, 470.0]  # 30 px right of the person box
        self.run(3.0)

    def walk_away(self, seconds: float = 3.0) -> list[dict]:
        return self.run(seconds, {2: (200.0, 0.0)})


class AssociationTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests expect candidate events regardless of the demo default.
        patcher = mock.patch.object(settings, "ASSOC_EMIT_CANDIDATE_EVENTS", True, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_ascii(self, scene: Scene) -> None:
        for event in scene.events:
            event["message"].encode("ascii")
        for alert in scene.alerts():
            alert.title.encode("ascii")
            for line in alert.lines:
                line.encode("ascii")

    def test_hero_scenario_separating_then_unattended(self) -> None:
        s = Scene()
        s.associate()
        self.assertEqual(s.types(), [ASSOCIATION_CANDIDATE, ASSOCIATION_CREATED])
        self.assertEqual(s.events[0]["severity"], INFO)
        self.assertEqual(s.events[0]["message"], "Backpack #5 near Person #2")
        self.assertEqual(s.events[1]["severity"], NOTICE)
        self.assertEqual(s.events[1]["message"], "Backpack #5 associated with Person #2")
        self.assertEqual(s.engine.get(5).state, ASSOCIATED)
        self.assertEqual(s.engine.get(5).person_id, 2)
        self.assertEqual(s.alerts(), [])

        s.walk_away(3.0)  # person ends ~600 px right of the bag, then stands
        s.run(1.0)
        rel = s.engine.get(5)
        self.assertEqual(rel.state, SEPARATING)
        sep = [e for e in s.events if e["event_type"] == OBJECT_SEPARATING]
        self.assertEqual(len(sep), 1)
        self.assertEqual(sep[0]["severity"], WARNING)
        self.assertEqual(sep[0]["message"], "Backpack #5 separating from Person #2")
        self.assertAlmostEqual(sep[0]["timestamp"] - rel.separated_since, 1.0, delta=0.11)
        alerts = s.alerts()
        self.assertEqual([a.title for a in alerts], ["OBJECT SEPARATING"])
        self.assertTrue(alerts[0].lines[2].startswith("Separated: "))

        s.run(5.0)
        rel = s.engine.get(5)
        self.assertEqual(rel.state, UNATTENDED)
        un = [e for e in s.events if e["event_type"] == UNATTENDED_OBJECT]
        self.assertEqual(len(un), 1)
        self.assertEqual(un[0]["severity"], ALERT)
        self.assertEqual(un[0]["message"], "UNATTENDED OBJECT: Backpack #5, last associated with Person #2")
        self.assertEqual(un[0]["object_id"], 5)
        self.assertEqual(un[0]["metadata"]["last_associated_person_id"], 2)
        self.assertEqual(un[0]["metadata"]["object_class"], "backpack")
        self.assertAlmostEqual(un[0]["metadata"]["unattended_seconds"], 5.0, delta=0.11)
        self.assertAlmostEqual(un[0]["timestamp"] - rel.separated_since, 5.0, delta=0.11)

        alerts = s.alerts()
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, UNATTENDED_OBJECT)
        self.assertEqual(alert.title, "UNATTENDED OBJECT")
        self.assertEqual(alert.severity, ALERT)
        self.assertEqual(alert.lines[:2], ["Backpack #5", "Last associated with Person #2"])
        expected = f"Unattended: {s.t - rel.separated_since:.1f} sec"
        self.assertEqual(alert.lines[2], expected)
        self.assertEqual(alert.since, rel.separated_since)
        self.assertEqual(alert.entity_ids, [5, 2])

        # Nothing repeats while the situation persists.
        s.run(10.0)
        self.assertEqual(
            s.types(),
            [ASSOCIATION_CANDIDATE, ASSOCIATION_CREATED, OBJECT_SEPARATING, UNATTENDED_OBJECT],
        )
        self.assert_ascii(s)

    def test_person_returns_restores_association_and_clears_alert(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(3.0)
        s.run(7.0)
        self.assertEqual(s.engine.get(5).state, UNATTENDED)
        back = s.run(3.0, {2: (-200.0, 0.0)})
        self.assertEqual(s.types(back), [ASSOCIATION_RESTORED])
        self.assertEqual(back[0]["message"], "Person #2 returned to Backpack #5")
        self.assertEqual(back[0]["severity"], INFO)
        rel = s.engine.get(5)
        self.assertEqual(rel.state, ASSOCIATED)
        self.assertEqual(rel.person_id, 2)
        self.assertIsNone(rel.separated_since)
        self.assertEqual(s.alerts(), [])
        self.assertEqual(s.types(s.run(5.0)), [])

    def test_restore_while_still_separating(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(3.0)
        s.run(1.5)
        self.assertEqual(s.engine.get(5).state, SEPARATING)
        back = s.run(3.0, {2: (-200.0, 0.0)})
        self.assertEqual(s.types(back), [ASSOCIATION_RESTORED])
        self.assertNotIn(UNATTENDED_OBJECT, s.types())

    def test_carrying_bag_across_frame_stays_associated(self) -> None:
        s = Scene()
        s.associate()
        moved = s.run(4.0, {2: (150.0, 0.0), 5: (150.0, 0.0)})
        self.assertEqual(s.types(moved), [])
        self.assertEqual(s.engine.get(5).state, ASSOCIATED)
        self.assertEqual(s.alerts(), [])
        # Carried out of view: both tracks end. Info only, no alert.
        del s.people[2]
        del s.objects[5]
        gone = s.run(1.0)
        self.assertTrue(all(e["severity"] == INFO for e in gone))
        self.assertEqual(s.alerts(), [])

    def test_associated_person_leaves_world_while_moving_with_bag(self) -> None:
        s = Scene()
        s.associate()
        s.run(1.0, {2: (150.0, 0.0), 5: (150.0, 0.0)})
        del s.people[2]  # person track ends, bag still visible and moving
        evs = s.run(3.0, {5: (150.0, 0.0)})
        self.assertNotIn(OBJECT_SEPARATING, s.types(evs))
        self.assertEqual(s.alerts(), [])

    def test_passer_by_does_not_associate(self) -> None:
        s = Scene()
        s.objects[5] = [640.0, 470.0]
        s.people[3] = [200.0, 350.0]
        s.run(3.0, {3: (300.0, 0.0)})  # walks straight past the bag
        s.run(10.0, {3: (300.0, 0.0)})
        self.assertNotIn(ASSOCIATION_CREATED, s.types())
        self.assertNotIn(UNATTENDED_OBJECT, s.types())
        self.assertEqual(s.engine.get(5).state, UNASSOCIATED)
        self.assertEqual(s.alerts(), [])

    def test_unassociated_bag_never_unattended(self) -> None:
        s = Scene()
        s.objects[5] = [640.0, 470.0]
        s.people[3] = [100.0, 350.0]  # far away the whole time
        s.run(30.0)
        self.assertEqual(s.events, [])
        self.assertEqual(s.engine.get(5).state, UNASSOCIATED)
        self.assertEqual(s.alerts(), [])

    def test_associated_person_leaves_world_bag_stays(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(1.5)
        del s.people[2]
        s.run(8.0)
        types = s.types()
        self.assertEqual(types.count(OBJECT_SEPARATING), 1)
        self.assertEqual(types.count(UNATTENDED_OBJECT), 1)
        un = [e for e in s.events if e["event_type"] == UNATTENDED_OBJECT][0]
        self.assertEqual(un["message"], "UNATTENDED OBJECT: Backpack #5, last associated with Person #2")
        self.assertEqual(s.alerts()[0].title, "UNATTENDED OBJECT")

    def test_object_moved_by_different_person(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(3.0)
        s.run(7.0)
        self.assertEqual(s.engine.get(5).state, UNATTENDED)
        bag = s.objects[5]
        s.people[4] = [bag[0] - 60.0, bag[1] - 100.0]  # steps right up to the bag
        self.assertEqual(s.types(s.run(1.0)), [])  # standing near is not a move
        evs = s.run(1.5, {4: (-150.0, 0.0), 5: (-150.0, 0.0)})
        moved = [e for e in evs if e["event_type"] == OBJECT_MOVED]
        self.assertEqual(len(moved), 1)
        self.assertEqual(
            moved[0]["message"],
            "OBJECT MOVED: Backpack #5, previously associated with Person #2, currently near Person #4",
        )
        self.assertEqual(moved[0]["severity"], ALERT)
        self.assertFalse(moved[0]["metadata"]["removed_from_view"])
        self.assertEqual(moved[0]["metadata"]["last_associated_person_id"], 2)
        rel = s.engine.get(5)
        self.assertNotIn(rel.state, (SEPARATING, UNATTENDED))
        self.assertEqual(rel.last_associated_person_id, 2)

        alerts = s.alerts()
        self.assertEqual([a.title for a in alerts], ["OBJECT MOVED"])
        self.assertEqual(
            alerts[0].lines,
            ["Backpack #5", "Previously associated with Person #2", "Currently near Person #4"],
        )
        # Alert expires after OBJECT_MOVED_ALERT_SECONDS; event never repeats.
        s.run(12.0, {4: (-10.0, 0.0), 5: (-10.0, 0.0)})
        self.assertEqual(s.types().count(OBJECT_MOVED), 1)
        self.assertNotIn("OBJECT MOVED", [a.title for a in s.alerts()])
        self.assert_ascii(s)

    def test_object_disappears_near_different_person(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(3.0)
        s.run(3.0)
        self.assertEqual(s.engine.get(5).state, SEPARATING)
        bag = s.objects[5]
        s.people[4] = [bag[0] - 60.0, bag[1] - 100.0]
        s.run(0.5)
        s.missed[5] = 1  # bag picked up and hidden by the body
        s.run(1.0, {4: (-150.0, 0.0)})
        del s.objects[5]  # tracker drops the coasted bag
        evs = s.run(0.5, {4: (-150.0, 0.0)})
        moved = [e for e in evs if e["event_type"] == OBJECT_MOVED]
        self.assertEqual(len(moved), 1)
        self.assertEqual(
            moved[0]["message"], "OBJECT MOVED: Backpack #5 no longer visible, last seen near Person #4"
        )
        self.assertTrue(moved[0]["metadata"]["removed_from_view"])
        self.assertIsNone(s.engine.get(5))
        alerts = s.alerts()
        self.assertEqual([a.title for a in alerts], ["OBJECT MOVED"])
        self.assertEqual(alerts[0].lines[2], "Last seen near Person #4")
        self.assertNotIn(OBJECT_LOST, s.types())

    def test_object_disappears_with_nobody_near_is_lost_info(self) -> None:
        s = Scene()
        s.associate()
        s.walk_away(3.0)
        s.run(7.0)
        del s.objects[5]
        evs = s.run(0.5)
        self.assertEqual(s.types(evs), [OBJECT_LOST])
        self.assertEqual(evs[0]["message"], "Backpack #5 no longer visible")
        self.assertEqual(evs[0]["severity"], INFO)
        self.assertEqual(s.alerts(), [])
        # Tracker revives the same ID at the same spot: unattended state resumes silently.
        s.objects[5] = [380.0, 470.0]
        evs = s.run(1.0)
        self.assertEqual(s.types(evs), [])
        self.assertEqual(s.engine.get(5).state, UNATTENDED)
        self.assertEqual(s.alerts()[0].title, "UNATTENDED OBJECT")

    def test_own_person_picks_up_bag_no_warning(self) -> None:
        s = Scene()
        s.associate()
        s.missed[5] = 1  # picked up, hidden by the body
        s.run(2.0, {2: (200.0, 0.0)})
        del s.objects[5]
        s.run(3.0, {2: (200.0, 0.0)})
        self.assertNotIn(OBJECT_SEPARATING, s.types())
        self.assertNotIn(OBJECT_MOVED, s.types())
        self.assertEqual(s.types()[-1], OBJECT_LOST)
        self.assertEqual(s.events[-1]["message"], "Backpack #5 left view with Person #2")
        self.assertEqual(s.alerts(), [])

    def test_person_id_switch_next_to_bag_keeps_association(self) -> None:
        s = Scene()
        s.associate()
        pos = s.people.pop(2)
        s.people[7] = list(pos)  # tracker re-identifies the same person
        evs = s.run(6.0)
        self.assertEqual(s.types(evs), [])
        rel = s.engine.get(5)
        self.assertEqual(rel.state, ASSOCIATED)
        self.assertEqual(rel.person_id, 7)

    def test_second_person_nearby_does_not_steal_candidacy(self) -> None:
        s = Scene()
        s.people[2] = [300.0, 350.0]
        s.people[3] = [470.0, 350.0]  # also near the bag, but slightly farther
        s.objects[5] = [380.0, 470.0]
        s.run(3.0)
        rel = s.engine.get(5)
        self.assertEqual(rel.state, ASSOCIATED)
        self.assertEqual(rel.person_id, 2)
        self.assertEqual(s.types().count(ASSOCIATION_CREATED), 1)


if __name__ == "__main__":
    unittest.main()
