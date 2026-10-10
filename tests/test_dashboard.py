from __future__ import annotations

import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from thirdeye.core import settings
from thirdeye.core.types import (
    ALERT,
    ASSOCIATION_CANDIDATE,
    ENTITY_ENTERED,
    INFO,
    OCCUPANCY_CHANGED,
    UNATTENDED_OBJECT,
    WARNING,
    ZONE_ENTERED,
    ZONE_EXITED,
    Alert,
    TrackedEntity,
    make_event,
)
from thirdeye.relationships.object_association import AssociationEngine
from thirdeye.spatial.zone_manager import ZoneManager
from thirdeye.state.entity_memory import EntityMemory
from thirdeye.state.world_state import WorldState
from thirdeye.ui import dashboard as dash
from thirdeye.ui import debug_view
from thirdeye.ui import widgets as wd
from thirdeye.ui.dashboard import Dashboard, is_demo_event

DT = 0.05
ZONES = [
    {
        "id": "restricted_a",
        "name": "Restricted Zone A",
        "type": "RESTRICTED",
        "points": [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
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
SHAPE = (settings.DASHBOARD_HEIGHT, settings.DASHBOARD_WIDTH, 3)
DEBUG_ONLY = ("FPS", "conf", "miss", "IoU", "CANDIDATE")


def person(pid: int, cx: float, t: float, missed: int = 0) -> TrackedEntity:
    return TrackedEntity(pid, "person", 0.9, (cx - 50, 250.0, cx + 50, 550.0), (cx, 400.0), t, missed)


def bag(oid: int, cx: float, t: float, missed: int = 0, cy: float = 520.0) -> TrackedEntity:
    return TrackedEntity(oid, "backpack", 0.8, (cx - 30, cy - 30, cx + 30, cy + 30), (cx, cy), t, missed)


class Scene:
    """Real WorldState + ZoneManager + AssociationEngine + EntityMemory."""

    def __init__(self, snapshot_dir: str) -> None:
        self.world = WorldState()
        self.zones = ZoneManager(ZONES)
        self.assoc = AssociationEngine()
        self.memory = EntityMemory(snapshot_dir=snapshot_dir, session_id="test")
        self.frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 90, np.uint8)
        self.t = 100.0
        self.events: list[dict] = []

    def step(self, tracks: list[TrackedEntity]) -> None:
        self.world.update(tracks, self.t)
        self.zones.update(self.world)
        events = self.assoc.update(self.world, self.t)
        events += self.memory.observe(self.world, self.frame, self.assoc, self.zones, events, self.t)
        self.events.extend(events)
        self.t = round(self.t + DT, 6)

    def run(self, steps: int, tracks_fn) -> None:
        for _ in range(steps):
            self.step(tracks_fn(self.t))

    def render(self, board: Dashboard, alerts=None, stats=None):
        return board.render(
            self.frame,
            self.world,
            self.zones,
            self.assoc.relationships(),
            alerts if alerts is not None else [],
            stats if stats is not None else {},
            self.memory,
        )


def sample_alert(now: float) -> Alert:
    return Alert(
        alert_type=UNATTENDED_OBJECT,
        title="UNATTENDED OBJECT",
        lines=["Backpack #2", "Last associated with Person #1"],
        severity=ALERT,
        since=now - 6.0,
        entity_ids=[2],
    )


def sample_events(now: float) -> list[dict]:
    return [
        make_event(ENTITY_ENTERED, "Person #1 entered", entity_id=1, timestamp=now - 5, metadata={"class_name": "person"}),
        make_event(ZONE_ENTERED, "Person #1 entered Restricted Zone A", severity=WARNING, entity_id=1,
                   zone_id="restricted_a", timestamp=now - 4, metadata={"zone_type": "RESTRICTED"}),
        make_event(OCCUPANCY_CHANGED, "Occupancy 0 -> 1", timestamp=now - 3),
        make_event(ASSOCIATION_CANDIDATE, "Backpack #2 CANDIDATE near Person #1", object_id=2, timestamp=now - 2),
        make_event(UNATTENDED_OBJECT, "UNATTENDED OBJECT: Backpack #2", severity=ALERT, object_id=2, timestamp=now - 1),
    ]


STATS = {
    "fps": 24.3,
    "infer_ms": 18.2,
    "status": "LIVE",
    "filter_suppressed": 12,
    "track_merges": 3,
    "occupancy_total": 1,
    "crowded": False,
    "zone_occupancy": {"restricted_a": 1, "monitored_main": 1},
}


class DashboardTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(settings, "SNAPSHOTS_ENABLED", False)
        self._patch.start()
        self.scene = Scene(self._tmp.name)
        # Person 1 with Backpack 2 (inside the restricted zone), Backpack 3 elsewhere.
        self.scene.run(40, lambda t: [person(1, 300, t), bag(2, 330, t), bag(3, 900, t)])
        self.board = Dashboard(mode="demo")
        self.board.add_events(sample_events(self.scene.world.now))

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def record_text(self):
        """Patch widgets.text (used by dashboard, debug_view and widgets helpers)."""
        drawn: list[str] = []
        original = wd.text

        def recorder(img, s, *args, **kwargs):
            drawn.append(str(s))
            return original(img, s, *args, **kwargs)

        patcher = mock.patch.object(wd, "text", recorder)
        patcher.start()
        self.addCleanup(patcher.stop)
        return drawn


class DashboardRenderTests(DashboardTestBase):
    def test_render_demo_and_debug(self) -> None:
        now = self.scene.world.now

        for mode in ("demo", "debug"):
            self.board.mode = mode

            for alerts in ([], [sample_alert(now)]):
                canvas = self.scene.render(self.board, alerts, STATS)
                self.assertEqual(canvas.shape, SHAPE)
                self.assertEqual(canvas.dtype, np.uint8)

    def test_render_without_memory_or_frame(self) -> None:
        for mode in ("demo", "debug"):
            self.board.mode = mode
            canvas = self.board.render(None, self.scene.world, self.scene.zones, [], [], {}, None)
            self.assertEqual(canvas.shape, SHAPE)

    def test_toggle_debug_and_empty_stats(self) -> None:
        self.assertEqual(self.board.mode, "demo")
        self.assertTrue(self.board.handle_key(ord("d")))
        self.assertEqual(self.board.mode, "debug")
        self.assertEqual(self.scene.render(self.board, [], {}).shape, SHAPE)
        self.assertTrue(self.board.handle_key(ord("d")))
        self.assertEqual(self.board.mode, "demo")
        self.assertEqual(self.scene.render(self.board, [], {}).shape, SHAPE)

    def test_empty_world(self) -> None:
        world = WorldState()
        for mode in ("demo", "debug"):
            board = Dashboard(mode=mode)
            canvas = board.render(None, world, None, None, None, None, None)
            self.assertEqual(canvas.shape, SHAPE)

    def test_demo_hides_debug_text(self) -> None:
        drawn = self.record_text()
        now = self.scene.world.now
        self.scene.render(self.board, [sample_alert(now)], STATS)

        for s in drawn:
            for word in DEBUG_ONLY:
                self.assertNotIn(word, s)

        drawn.clear()
        self.board.handle_key(ord("d"))
        self.scene.render(self.board, [sample_alert(now)], STATS)
        self.assertTrue(any("FPS" in s for s in drawn))
        self.assertTrue(any("UNATTENDED_OBJECT" in s for s in drawn))  # alert + event rows
        self.assertTrue(any("ASSOCIATION_CANDIDATE" in s for s in drawn))  # unfiltered log
        self.assertTrue(any("Archived" in s for s in drawn))

    def test_debug_panel_truncates_many_tracks(self) -> None:
        tracks = [person(10 + i, 60 + i * 30, self.scene.t) for i in range(40)]
        self.scene.step(tracks)
        self.board.mode = "debug"
        drawn = self.record_text()
        canvas = self.scene.render(self.board, [], STATS)
        self.assertEqual(canvas.shape, SHAPE)
        self.assertTrue(any(s.startswith("+") and s.endswith("more") for s in drawn))

    def test_debug_panel_stays_inside_bounds(self) -> None:
        tracks = [person(10 + i, 60 + i * 30, self.scene.t) for i in range(40)]
        self.scene.step(tracks)
        canvas = np.zeros(SHAPE, np.uint8)
        events = sample_events(self.scene.world.now) * 20
        debug_view.draw_panel(canvas, self.scene.world, self.scene.zones.zones, self.scene.assoc.relationships(),
                              [sample_alert(self.scene.world.now)] * 10, STATS, self.scene.world.now)
        debug_view.draw_events(canvas, events, self.scene.world.now)
        # Nothing drawn outside the right panel and bottom panel.
        mask = np.ones(SHAPE[:2], bool)
        mask[dash.VIDEO_Y : dash.H - dash.M, dash.PANEL_X : dash.PANEL_X + dash.PANEL_W] = False
        mask[dash.BOTTOM_Y : dash.H - dash.M, dash.VIDEO_X : dash.VIDEO_X + dash.VIDEO_W] = False
        self.assertFalse(canvas[mask].any())

    def test_debug_view_tolerates_odd_inputs(self) -> None:
        canvas = np.zeros(SHAPE, np.uint8)
        debug_view.draw_panel(canvas, object(), [], None, None, None, 0.0)
        debug_view.draw_events(canvas, [], 0.0)
        debug_view.draw_events(canvas, [{"timestamp": "bad"}, "not-a-dict", {}], 0.0)
        debug_view.draw_panel(canvas, self.scene.world, [], [], [], {"zone_occupancy": None, "fps": None}, 0.0)


class DemoEventFilterTests(unittest.TestCase):
    def test_noise_hidden(self) -> None:
        for event_type in (OCCUPANCY_CHANGED, ZONE_EXITED, ASSOCIATION_CANDIDATE):
            self.assertFalse(is_demo_event(make_event(event_type, "x")))

    def test_restricted_zone_entered_shown(self) -> None:
        event = make_event(ZONE_ENTERED, "Person #1 entered", zone_id="restricted_a", metadata={"zone_type": "RESTRICTED"})
        self.assertTrue(is_demo_event(event))
        by_id = make_event(ZONE_ENTERED, "Person #1 entered", zone_id="restricted_a")
        self.assertTrue(is_demo_event(by_id, {"restricted_a"}))
        monitored = make_event(ZONE_ENTERED, "Person #1 entered", zone_id="monitored_main", metadata={"zone_type": "MONITORED"})
        self.assertFalse(is_demo_event(monitored))

    def test_entity_entered_people_only(self) -> None:
        self.assertTrue(is_demo_event(make_event(ENTITY_ENTERED, "Person #1 entered", metadata={"class_name": "person"})))
        self.assertFalse(is_demo_event(make_event(ENTITY_ENTERED, "Backpack #2 appeared", metadata={"class_name": "backpack"})))
        self.assertFalse(is_demo_event(make_event(ENTITY_ENTERED, "Backpack #2 appeared", severity=INFO)))


class RecentlySeenTests(DashboardTestBase):
    def test_disappeared_bag_selectable(self) -> None:
        # Backpack 3 leaves the frame.
        self.scene.run(3, lambda t: [person(1, 300, t), bag(2, 330, t)])
        self.assertIsNone(self.scene.world.get(3))
        self.assertIn(3, self.scene.world.archived)

        for mode in ("demo", "debug"):
            self.board.mode = mode
            self.board.selected_entity_id = None
            self.scene.render(self.board)
            self.assertIn(3, self.board._recent_ids)
            index = self.board._recent_ids.index(3)
            self.assertTrue(self.board.handle_key(ord("1") + index))
            self.assertEqual(self.board.selected_entity_id, 3)
            self.assertEqual(self.scene.render(self.board).shape, SHAPE)
            self.assertEqual(self.board.selected_entity_id, 3)
            self.assertTrue(self.board.handle_key(ord("x")))
            self.assertIsNone(self.board.selected_entity_id)
            self.assertEqual(self.scene.render(self.board).shape, SHAPE)

    def test_out_of_range_key_ignored(self) -> None:
        self.scene.render(self.board)
        self.assertFalse(self.board.handle_key(ord("9")))
        self.assertIsNone(self.board.selected_entity_id)


class RenderSpeedTests(DashboardTestBase):
    def test_render_speed(self) -> None:
        now = self.scene.world.now
        alerts = [sample_alert(now)]

        for mode in ("demo", "debug"):
            self.board.mode = mode
            self.scene.render(self.board, alerts, STATS)  # warm caches
            start = time.perf_counter()

            for _ in range(10):
                self.scene.render(self.board, alerts, STATS)

            average = (time.perf_counter() - start) / 10
            self.assertLess(average, 0.060, f"{mode} render {average * 1000:.1f} ms")


if __name__ == "__main__":
    unittest.main()
