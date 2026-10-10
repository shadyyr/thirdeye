from __future__ import annotations

import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import TrackedEntity
from thirdeye.spatial.zone_manager import Zone, ZoneManager, point_in_polygon
from thirdeye.state.world_state import WorldState

FRAME = (1000, 1000)
ZONES = [
    {"id": "r", "name": "Restricted Zone A", "type": "RESTRICTED",
     "points": [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]},
    {"id": "m", "name": "Monitored Area", "type": "MONITORED",
     "points": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]},
]


def track(tid: int, cx: float, cy: float = 300.0, cls: str = "person", half: float = 50.0) -> TrackedEntity:
    return TrackedEntity(tid, cls, 0.9, (cx - half, cy - half, cx + half, cy + half), (cx, cy), 0.0)


def new_world() -> WorldState:
    w = WorldState()
    w.frame_size = FRAME
    return w


class PointInPolygonTests(unittest.TestCase):
    def test_square(self) -> None:
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon(5, 5, sq))
        self.assertFalse(point_in_polygon(15, 5, sq))
        self.assertFalse(point_in_polygon(-1, 5, sq))

    def test_concave_polygon(self) -> None:
        # L shape: notch in the top-right quadrant
        poly = [(0, 0), (0.5, 0), (0.5, 0.5), (1, 0.5), (1, 1), (0, 1)]
        zone = Zone("l", "L", "MONITORED", poly)
        self.assertTrue(zone.contains((250, 250), FRAME))
        self.assertFalse(zone.contains((750, 250), FRAME))
        self.assertTrue(zone.contains((750, 750), FRAME))

    def test_triangle_and_polygon_px(self) -> None:
        zone = Zone("t", "T", "RESTRICTED", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        self.assertTrue(zone.is_restricted)
        self.assertTrue(zone.contains((100, 100), FRAME))
        self.assertFalse(zone.contains((900, 900), FRAME))
        self.assertEqual(zone.polygon_px((200, 100)), [(0, 0), (200, 0), (0, 100)])

    def test_defaults_from_settings(self) -> None:
        zm = ZoneManager()
        self.assertEqual([z.id for z in zm.zones], [c["id"] for c in settings.ZONES])
        self.assertIsNotNone(zm.get("restricted_a"))
        self.assertIsNone(zm.get("nope"))


@mock.patch.object(settings, "ZONE_CONFIRM_FRAMES", 3)
@mock.patch.object(settings, "ZONE_ANCHOR", "center")
class ZoneDebounceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = new_world()
        self.zm = ZoneManager(ZONES)
        self.t = 0.0

    def step(self, tracks):
        self.t += 0.1
        self.world.update(tracks, self.t)
        return self.zm.update(self.world)

    def test_enter_requires_confirm_frames(self) -> None:
        self.assertEqual(self.step([track(1, 750)]), [])
        self.assertEqual(self.step([track(1, 750)]), [])
        self.assertEqual(self.step([track(1, 750)]), [("ENTER", 1, "r")])
        self.assertEqual(self.world.get(1).zones, {"r"})
        self.assertEqual(self.step([track(1, 750)]), [])

    def test_boundary_jitter_produces_no_duplicates(self) -> None:
        all_transitions = []
        for _ in range(3):
            all_transitions += self.step([track(1, 750)])
        # Jitter across the boundary, never 3 consecutive frames outside
        for x in [490, 510, 490, 490, 510, 490, 510, 510, 490, 490]:
            all_transitions += self.step([track(1, x)])
        self.assertEqual(all_transitions.count(("ENTER", 1, "r")), 1)
        self.assertNotIn(("EXIT", 1, "r"), all_transitions)
        self.assertNotIn(("ENTER", 1, "m"), all_transitions)
        # Now actually leave
        for _ in range(3):
            all_transitions += self.step([track(1, 200)])
        self.assertEqual(all_transitions.count(("EXIT", 1, "r")), 1)
        self.assertEqual(all_transitions.count(("ENTER", 1, "m")), 1)
        self.assertEqual(self.world.get(1).zones, {"m"})

    def test_exit_on_disappearance(self) -> None:
        for _ in range(3):
            self.step([track(1, 750), track(2, 750, cls="backpack")])
        self.assertEqual(self.world.get(2).zones, {"r"})
        out = self.step([track(2, 750, cls="backpack")])
        self.assertEqual(out, [("EXIT", 1, "r")])
        # Re-appearing with the same id starts fresh (no stale state)
        self.assertEqual(self.step([track(1, 750), track(2, 750, cls="backpack")]), [])

    def test_unconfirmed_disappearance_emits_nothing(self) -> None:
        self.step([track(1, 750)])
        self.assertEqual(self.step([]), [])

    def test_bottom_anchor(self) -> None:
        zones = [{"id": "floor", "name": "Floor", "type": "MONITORED",
                  "points": [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]}]
        zm = ZoneManager(zones)
        w = new_world()
        # center y=450 is above the zone, feet at y=550 are inside
        with mock.patch.object(settings, "ZONE_ANCHOR", "bottom"), \
                mock.patch.object(settings, "ZONE_CONFIRM_FRAMES", 1):
            w.update([track(1, 500, cy=450, half=100)], 1.0)
            self.assertEqual(zm.update(w), [("ENTER", 1, "floor")])
        zm2 = ZoneManager(zones)
        w2 = new_world()
        with mock.patch.object(settings, "ZONE_CONFIRM_FRAMES", 1):
            w2.update([track(1, 500, cy=450, half=100)], 1.0)
            self.assertEqual(zm2.update(w2), [])

    def test_inactive_zone_ignored(self) -> None:
        zones = [dict(ZONES[0], active=False), ZONES[1]]
        zm = ZoneManager(zones)
        w = new_world()
        for i in range(5):
            w.update([track(1, 750)], float(i))
            self.assertEqual(zm.update(w), [])
        self.assertEqual(w.get(1).zones, set())

    def test_deactivating_zone_exits_members(self) -> None:
        for _ in range(3):
            self.step([track(1, 750)])
        self.zm.get("r").active = False
        self.assertEqual(self.step([track(1, 750)]), [("EXIT", 1, "r")])
        self.assertEqual(self.step([track(1, 750)]), [])


if __name__ == "__main__":
    unittest.main()
