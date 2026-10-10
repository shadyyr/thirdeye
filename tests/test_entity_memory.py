from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from thirdeye.core import settings
from thirdeye.core.types import (
    ACTIVE,
    ARCHIVED,
    INFO,
    OBJECT_LEFT_WITH,
    OUT_OF_VIEW,
    WARNING,
    TrackedEntity,
    make_event,
)
from thirdeye.relationships.object_association import ASSOCIATED, AssociationEngine
from thirdeye.spatial.zone_manager import ZoneManager
from thirdeye.state.entity_memory import EntityMemory, crop_snapshot
from thirdeye.state.world_state import WorldState

FPS = 20
DT = 1.0 / FPS
ZONES = [
    {
        "id": "restricted_a",
        "name": "Restricted Zone A",
        "type": "RESTRICTED",
        "points": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        "active": True,
    },
    {
        "id": "monitored_main",
        "name": "Monitored Area",
        "type": "MONITORED",
        "points": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        "active": True,
    },
]


def person(pid: int, cx: float, t: float, missed: int = 0) -> TrackedEntity:
    return TrackedEntity(pid, "person", 0.9, (cx - 50, 250.0, cx + 50, 550.0), (cx, 400.0), t, missed)


def bag(oid: int, cx: float, t: float, missed: int = 0, cy: float = 520.0, conf: float = 0.8) -> TrackedEntity:
    return TrackedEntity(oid, "backpack", conf, (cx - 30, cy - 30, cx + 30, cy + 30), (cx, cy), t, missed)


class FakeStore:
    session_id = "fake-session"

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}
        self.calls = 0

    def upsert_entity(self, summary: dict) -> None:
        self.calls += 1
        self.rows[summary["entity_id"]] = dict(summary)


class Scene:
    """Real WorldState + ZoneManager + AssociationEngine + EntityMemory."""

    def __init__(self, snapshot_dir: str, frame: np.ndarray | None = None) -> None:
        self.world = WorldState()
        self.zones = ZoneManager(ZONES)
        self.assoc = AssociationEngine()
        self.store = FakeStore()
        self.memory = EntityMemory(store=self.store, snapshot_dir=snapshot_dir, session_id="s1")
        self.frame = frame
        self.t = 0.0
        self.memory_events: list[dict] = []
        self.extra_events: list[dict] = []

    def step(self, tracks: list[TrackedEntity]) -> list[dict]:
        self.world.update(tracks, self.t)
        self.zones.update(self.world)
        events = self.assoc.update(self.world, self.t) + self.extra_events
        self.extra_events = []
        new = self.memory.observe(self.world, self.frame, self.assoc, self.zones, events, self.t)
        self.memory_events.extend(new)
        self.t = round(self.t + DT, 6)
        return new

    def run(self, seconds: float, tracks_fn) -> None:
        end = self.t + seconds - 1e-9
        while self.t < end:
            self.step(tracks_fn(self.t))

    def left_with(self) -> list[dict]:
        return [e for e in self.memory_events if e["event_type"] == OBJECT_LEFT_WITH]


class EntityMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.snap_dir = self._tmp.name
        self._patch = mock.patch.object(settings, "SNAPSHOTS_ENABLED", False)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _associate(self, scene: Scene) -> None:
        # Person 1 stands with Backpack #2 inside the restricted zone.
        scene.run(3.5, lambda t: [person(1, 300, t), bag(2, 330, t)])
        rel = scene.assoc.get(2)
        self.assertEqual(rel.state, ASSOCIATED)
        self.assertEqual(rel.last_associated_person_id, 1)

    def test_non_associated_person_walks_out_with_bag(self) -> None:
        scene = Scene(self.snap_dir)
        self._associate(scene)
        # Person 1 walks away (stays in view); Person 8 comes to the bag.
        start = scene.t
        scene.run(2.0, lambda t: [
            person(1, 300 + (900 - 300) * min(1.0, (t - start) / 1.0), t),
            person(8, 600 - 270 * min(1.0, (t - start) / 2.0), t),
            bag(2, 330, t),
        ])
        # Person 8 carries the bag towards the right edge.
        start = scene.t
        scene.run(1.0, lambda t: [person(1, 900, t), person(8, 330 + 100 * (t - start), t),
                                  bag(2, 360 + 100 * (t - start), t)])
        last_detected = scene.t - DT
        # Both stop being detected: person dropped after 1 s, bag coasts 4 s.
        start = scene.t
        scene.run(1.0, lambda t: [person(1, 900, t), person(8, 430, t, missed=1), bag(2, 460, t, missed=1)])
        scene.step([person(1, 900, scene.t), bag(2, 460, scene.t, missed=1)])
        self.assertEqual(scene.memory.summary(8).status, ARCHIVED)
        self.assertEqual(scene.memory.summary(2).status, OUT_OF_VIEW)
        self.assertEqual(scene.memory.recent_objects(scene.t)[0].entity_id, 2)
        self.assertEqual(scene.left_with(), [])
        scene.run(3.0, lambda t: [person(1, 900, t), bag(2, 460, t, missed=1)])
        self.assertEqual(scene.left_with(), [])
        scene.run(1.0, lambda t: [person(1, 900, t)])
        scene.run(6.0, lambda t: [person(1, 900, t)])

        events = scene.left_with()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["severity"], WARNING)
        self.assertEqual(event["message"], "Backpack #2 left view with Person #8")
        self.assertEqual((event["entity_id"], event["object_id"]), (8, 2))

        summary = scene.memory.summary(2)
        self.assertEqual(summary.status, ARCHIVED)
        self.assertAlmostEqual(summary.last_seen, last_detected, places=5)
        self.assertEqual(summary.last_zone_id, "restricted_a")  # restricted preferred
        self.assertEqual(summary.last_zone_name, "Restricted Zone A")
        self.assertEqual(summary.associated_person_id, 1)
        self.assertEqual(summary.left_with_person_id, 8)
        self.assertIn(event["event_id"], summary.event_ids)
        self.assertEqual(summary.last_center, (455.0, 520.0))  # last detected box, not the coasted one

        sentence = scene.memory.where_last_seen(2)
        self.assertTrue(sentence.startswith("Backpack #2 was last seen "), sentence)
        self.assertIn("in Restricted Zone A, near Person #8; it left view with Person #8.", sentence)
        self.assertTrue(sentence.isascii())

        # Persisted: archive row updated with left_with.
        row = scene.store.rows[2]
        self.assertEqual(row["status"], ARCHIVED)
        self.assertEqual(row["left_with_person_id"], 8)
        self.assertEqual(row["session_id"], "s1")
        self.assertEqual(scene.store.rows[8]["status"], ARCHIVED)
        self.assertNotIn(1, scene.store.rows)  # still live

    def _bag_vanishes_next_to_person(self, scene: Scene) -> None:
        self._associate(scene)
        start = scene.t
        scene.run(2.0, lambda t: [
            person(1, 300 + 600 * min(1.0, (t - start) / 1.0), t),
            person(8, 600 - 270 * min(1.0, (t - start) / 2.0), t),
            bag(2, 330, t),
        ])
        # Bag track ends (e.g. picked up and hidden) while Person 8 stays.
        scene.run(1.0, lambda t: [person(1, 900, t), person(8, 330, t), bag(2, 330, t, missed=1)])
        scene.step([person(1, 900, scene.t), person(8, 330, scene.t)])
        self.assertEqual(scene.memory.summary(2).status, ARCHIVED)

    def test_person_stays_in_view_no_event(self) -> None:
        scene = Scene(self.snap_dir)
        self._bag_vanishes_next_to_person(scene)
        scene.run(settings.LEFT_WITH_WINDOW_SECONDS + 1.0, lambda t: [person(1, 900, t), person(8, 330, t)])
        # Person leaves only after the window: still no event.
        scene.run(1.0, lambda t: [person(1, 900, t)])
        self.assertEqual(scene.left_with(), [])
        self.assertIsNone(scene.memory.summary(2).left_with_person_id)

    def test_pending_resolves_when_person_leaves_within_window(self) -> None:
        scene = Scene(self.snap_dir)
        self._bag_vanishes_next_to_person(scene)
        scene.run(1.0, lambda t: [person(1, 900, t), person(8, 330, t)])
        scene.run(1.0, lambda t: [person(1, 900, t)])
        events = scene.left_with()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["message"], "Backpack #2 left view with Person #8")
        self.assertEqual(scene.store.rows[2]["left_with_person_id"], 8)

    def test_revival_cancels_pending(self) -> None:
        scene = Scene(self.snap_dir)
        self._bag_vanishes_next_to_person(scene)
        scene.run(0.5, lambda t: [person(1, 900, t), person(8, 330, t), bag(2, 330, t)])
        self.assertEqual(scene.world.archived.get(2), None)
        self.assertEqual(scene.memory.summary(2).status, ACTIVE)
        self.assertEqual(scene.store.rows[2]["status"], ACTIVE)
        scene.run(1.0, lambda t: [person(1, 900, t), bag(2, 330, t)])
        self.assertEqual(scene.left_with(), [])

    def test_associated_person_leaves_with_bag_is_recorded_without_event(self) -> None:
        # The association engine already reports this (OBJECT_LOST); memory
        # only records it, so the timeline has no duplicate / second opinion.
        scene = Scene(self.snap_dir)
        self._associate(scene)
        start = scene.t
        scene.run(1.0, lambda t: [person(1, 300 + 100 * (t - start), t), bag(2, 330 + 100 * (t - start), t)])
        scene.run(0.5, lambda t: [person(1, 400, t, missed=1), bag(2, 430, t, missed=1)])
        scene.run(1.0, lambda t: [bag(2, 430, t, missed=1)])
        scene.run(1.0, lambda t: [])
        self.assertEqual(scene.left_with(), [])
        self.assertEqual(scene.memory.summary(2).left_with_person_id, 1)
        self.assertEqual(scene.store.rows[2]["left_with_person_id"], 1)
        self.assertIn("it left view with associated Person #1.", scene.memory.where_last_seen(2))

    def test_id_handoff_is_not_a_different_person(self) -> None:
        # Rehearsal: the carrier's tracker ID switched (#1 -> #3) while picking
        # the bag up. #1's track ending must not read as someone else leaving
        # with the bag.
        scene = Scene(self.snap_dir)
        self._associate(scene)
        scene.run(0.5, lambda t: [person(3, 300, t), bag(2, 330, t)])
        rel = scene.assoc.get(2)
        self.assertEqual(rel.last_associated_person_id, 3)
        self.assertEqual(rel.handoff_ids, [1])
        scene.run(0.5, lambda t: [person(3, 300, t), bag(2, 330, t, missed=1)])
        scene.run(settings.LEFT_WITH_WINDOW_SECONDS + 1.0, lambda t: [person(3, 300, t)])
        self.assertEqual(scene.left_with(), [])
        summary = scene.memory.summary(2)
        self.assertEqual(summary.status, ARCHIVED)
        self.assertEqual(summary.associated_person_id, 3)
        self.assertIsNone(summary.previous_person_id)  # same person, not a previous one
        self.assertEqual(summary.left_with_person_id, 3)

    def test_events_for_and_where_last_seen_live(self) -> None:
        scene = Scene(self.snap_dir)
        scene.extra_events = [make_event("ZONE_ENTERED", "a", entity_id=2, zone_id="restricted_a")]
        scene.step([bag(2, 330, scene.t)])
        scene.extra_events = [make_event("OBJECT_MOVED", "b", entity_id=7, object_id=2),
                              make_event("ZONE_ENTERED", "c", entity_id=7)]
        scene.step([bag(2, 330, scene.t), person(7, 900, scene.t)])
        self.assertEqual([e["message"] for e in scene.memory.events_for(2)], ["b", "a"])
        self.assertEqual([e["message"] for e in scene.memory.events_for(2, limit=1)], ["b"])
        self.assertEqual(len(scene.memory.summary(2).event_ids), 2)
        self.assertEqual(scene.memory.where_last_seen(2), "Backpack #2 is in view.")
        self.assertIsNone(scene.memory.where_last_seen(99))
        self.assertIsNone(scene.memory.summary(99))
        self.assertEqual(scene.memory.recent_objects(scene.t), [])


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_crop_size_and_clip(self) -> None:
        frame = np.zeros((720, 1280, 3), np.uint8)
        small = crop_snapshot(frame, (100, 100, 160, 160), 200)
        self.assertEqual(small.shape[2], 3)
        self.assertLessEqual(max(small.shape[:2]), 200)
        self.assertGreater(small.shape[0], 60)  # margin added
        big = crop_snapshot(frame, (100, 100, 500, 400), 200)
        self.assertEqual(max(big.shape[:2]), 200)
        edge = crop_snapshot(frame, (-50, 650, 40, 760), 200)
        self.assertIsNotNone(edge)
        self.assertLessEqual(edge.shape[0], 720 - 650 + 20)
        self.assertIsNone(crop_snapshot(frame, (2000, 2000, 2100, 2100), 200))

    def test_single_jpeg_on_archive(self) -> None:
        frame = np.full((720, 1280, 3), 80, np.uint8)
        frame[490:550, 300:360] = (0, 0, 255)
        snap_dir = os.path.join(self._tmp.name, "snaps")
        scene = Scene(snap_dir, frame=frame)
        with mock.patch.object(settings, "SNAPSHOTS_ENABLED", True), \
                mock.patch.object(settings, "SNAPSHOT_MAX_SIDE", 50):
            scene.run(2.0, lambda t: [person(1, 900, t), bag(2, 330, t)])
            self.assertFalse(os.path.exists(snap_dir))  # nothing written per frame
            crop = scene.memory.snapshot(2)
            self.assertIsNotNone(crop)
            self.assertEqual(max(crop.shape[:2]), 50)
            self.assertIsNone(scene.memory.snapshot(1))  # never people
            scene.run(0.5, lambda t: [person(1, 900, t)])
        files = [os.path.join(r, f) for r, _, fs in os.walk(snap_dir) for f in fs]
        expected = os.path.join(snap_dir, "s1", "backpack_2.jpg")
        self.assertEqual(files, [expected])
        summary = scene.memory.summary(2)
        self.assertEqual(summary.snapshot_path, expected)
        self.assertEqual(scene.store.rows[2]["snapshot_path"], expected)
        self.assertIsNotNone(scene.memory.snapshot(2))

    def test_low_confidence_not_snapshotted(self) -> None:
        frame = np.zeros((720, 1280, 3), np.uint8)
        scene = Scene(os.path.join(self._tmp.name, "snaps"), frame=frame)
        with mock.patch.object(settings, "SNAPSHOTS_ENABLED", True):
            scene.run(1.0, lambda t: [bag(2, 330, t, conf=0.1)])
        self.assertIsNone(scene.memory.snapshot(2))


if __name__ == "__main__":
    unittest.main()
