from __future__ import annotations

import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import ACTIVE, ARCHIVED, OUT_OF_VIEW, TrackedEntity
from thirdeye.state.world_state import WorldState


def track(track_id: int, cls: str, cx: float, cy: float, t: float, missed: int = 0, half: float = 30.0):
    return TrackedEntity(
        track_id=track_id,
        class_name=cls,
        confidence=0.8,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        center=(cx, cy),
        timestamp=t,
        frames_missed=missed,
    )


class WorldStateLifecycleTests(unittest.TestCase):
    def test_active_out_of_view_archived(self) -> None:
        world = WorldState()
        world.update([track(2, "backpack", 100, 100, 0.0)], 0.0)
        self.assertEqual(world.appeared, [2])
        record = world.get(2)
        self.assertEqual(record.status, ACTIVE)

        world.update([track(2, "backpack", 100, 100, 0.1, missed=1)], 0.1)
        self.assertEqual(record.status, OUT_OF_VIEW)
        self.assertEqual(record.last_detected, 0.0)
        self.assertEqual(record.last_seen, 0.1)

        world.update([], 0.2)
        self.assertEqual([r.track_id for r in world.disappeared], [2])
        self.assertNotIn(2, world.entities)
        self.assertIs(world.archived[2], record)
        self.assertEqual(record.status, ARCHIVED)
        self.assertEqual(record.archived_at, 0.2)
        self.assertIs(world.get_any(2), record)
        self.assertIsNone(world.get(2))

        world.update([], 0.3)
        self.assertEqual(world.disappeared, [])  # only reported once

    def test_revival_restores_record(self) -> None:
        world = WorldState()
        for i in range(5):
            world.update([track(2, "backpack", 100, 100, i * 0.1)], i * 0.1)
        history_len = len(world.get(2).history)
        world.update([], 0.5)
        self.assertIn(2, world.archived)

        world.update([track(2, "backpack", 110, 100, 1.0)], 1.0)
        self.assertEqual(world.revived, [2])
        self.assertEqual(world.appeared, [])
        self.assertNotIn(2, world.archived)
        record = world.get(2)
        self.assertEqual(record.first_seen, 0.0)
        self.assertIsNone(record.archived_at)
        self.assertEqual(record.status, ACTIVE)
        self.assertEqual(len(record.history), history_len + 1)
        self.assertEqual(record.velocity, (0.0, 0.0))

        world.update([track(2, "backpack", 110, 100, 1.1)], 1.1)
        self.assertEqual(world.revived, [])

    def test_class_mismatch_is_new_entity(self) -> None:
        world = WorldState()
        world.update([track(3, "backpack", 100, 100, 0.0)], 0.0)
        world.update([], 0.1)
        world.update([track(3, "person", 100, 100, 0.2)], 0.2)
        self.assertEqual(world.appeared, [3])
        self.assertEqual(world.revived, [])
        self.assertEqual(world.get(3).first_seen, 0.2)

    def test_archive_bound_drops_oldest(self) -> None:
        world = WorldState()
        with mock.patch.object(settings, "ARCHIVE_MAX_ENTITIES", 3):
            for i in range(1, 6):
                world.update([track(i, "person", 100, 100, float(i))], float(i))
            world.update([], 10.0)
        self.assertEqual(list(world.archived), [3, 4, 5])
        self.assertEqual(len(world.disappeared), 1)

    def test_disappeared_and_appeared_same_frame(self) -> None:
        world = WorldState()
        world.update([track(1, "person", 100, 100, 0.0)], 0.0)
        world.update([track(2, "person", 300, 100, 0.1)], 0.1)
        self.assertEqual(world.appeared, [2])
        self.assertEqual([r.track_id for r in world.disappeared], [1])
        self.assertEqual(list(world.entities), [2])


if __name__ == "__main__":
    unittest.main()
