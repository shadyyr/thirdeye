"""third_eye_app.py --incidents wiring, driven through main() with a scripted
camera and detector (no GPU, camera or real video encoder)."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import third_eye_app as app  # noqa: E402
from thirdeye.core import settings  # noqa: E402
from thirdeye.core.types import DWELLING, ZONE_ENTERED, Detection  # noqa: E402
from thirdeye.recording import IncidentRecorder  # noqa: E402
from thirdeye.recording import incident_recorder as incident_module  # noqa: E402

FPS = 10.0
W, H = 1280, 720
ENTER_FRAME = 20  # person walks into Restricted Zone A here


def recorder_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name in ("incident-writer", "incident-jpeg")]


class FakeCap:
    """A replay source: n frames at FPS, then end of file."""

    def __init__(self, frames: int) -> None:
        self.left = frames
        self.frame = np.full((H, W, 3), 90, dtype=np.uint8)

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        time.sleep(0.004)  # let the buffer's JPEG worker keep up, as with a real camera
        return True, self.frame

    def get(self, prop):
        return FPS

    def isOpened(self):
        return True

    def release(self):
        pass


class FakeDetector:
    """One person: outside the zone, then standing inside Restricted Zone A."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads_seen: set[str] = set()

    def __call__(self, model, frame):
        self.calls += 1
        self.threads_seen.update(recorder_threads())
        x = (0.30 if self.calls < ENTER_FRAME else 0.80) * W + (self.calls % 3)
        return [Detection("person", 0.9, (x - 70, 200.0, x + 70, 620.0))], 1.0


class FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.frames = 0

    def write(self, img) -> None:
        self.frames += 1

    def release(self) -> None:
        with open(self.path, "wb") as fh:
            fh.write(b"x" * max(1, self.frames))


class AppIncidentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.writers: list[FakeWriter] = []
        self._handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

    def tearDown(self) -> None:
        signal.signal(signal.SIGINT, self._handlers[0])
        signal.signal(signal.SIGTERM, self._handlers[1])
        self._tmp.cleanup()

    def fake_writer(self, path, fps, size, codecs=None, x264_bitrate=None):
        writer = FakeWriter(path)
        self.writers.append(writer)
        return writer, "fake", path

    def run_app(self, name: str, frames: int, *flags: str, detector: FakeDetector | None = None):
        """Run main() on a scripted replay; returns (exit code, db path, detector)."""
        detector = detector or FakeDetector()
        db = os.path.join(self.tmp, name, "events.sqlite3")
        with mock.patch.object(settings, "EVENT_DB_PATH", db), \
                mock.patch.object(settings, "SNAPSHOT_DIR", os.path.join(self.tmp, name, "snapshots")), \
                mock.patch.object(settings, "INCIDENT_DIR", os.path.join(self.tmp, name, "incidents"), create=True), \
                mock.patch.object(app, "open_source", lambda args: FakeCap(frames)), \
                mock.patch.object(app, "load_model", lambda: object()), \
                mock.patch.object(app, "detect", detector), \
                mock.patch.object(incident_module, "open_video_writer", self.fake_writer):
            code = app.main(["--source", "scripted.mp4", "--headless", "--quiet", *flags])
        return code, db, detector

    @staticmethod
    def events(db: str) -> list[tuple]:
        with sqlite3.connect(db) as conn:
            return conn.execute("SELECT event_type, entity_id, zone_id FROM events ORDER BY rowid").fetchall()

    @staticmethod
    def incidents(db: str) -> list[dict]:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM incidents ORDER BY rowid")]

    def test_flag_is_off_by_default(self) -> None:
        self.assertFalse(app.parse_args([]).incidents)
        self.assertTrue(app.parse_args(["--incidents"]).incidents)

    def test_disabled_creates_no_recorder_threads_or_clips(self) -> None:
        with mock.patch.object(app, "start_incident_recorder") as start:
            code, db, detector = self.run_app("off", 200)
        self.assertEqual(code, 0)
        start.assert_not_called()
        self.assertEqual(detector.threads_seen, set())
        self.assertEqual(self.incidents(db), [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "off", "incidents")))
        self.assertIn(ZONE_ENTERED, [e[0] for e in self.events(db)])  # the scenario does produce triggers

    def test_enabled_event_triggers_linked_clip(self) -> None:
        # 26 s: entry at ~2.7 s, DWELLING at ~12.7 s (merged), clip ends 12 s after it
        # (the next DWELLING repeat, ~15 s later, would open a second clip).
        code, db, detector = self.run_app("on", 260, "--incidents")
        self.assertEqual(code, 0)
        self.assertEqual(set(detector.threads_seen), {"incident-writer", "incident-jpeg"})
        rows = self.incidents(db)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["event_type"], ZONE_ENTERED)
        self.assertEqual(row["zone_id"], "restricted_a")
        self.assertLess(row["clip_start"], row["trigger_time"] - 1.0)  # pre-roll footage
        self.assertGreater(row["clip_end"], row["trigger_time"] + 10.0)  # post-roll footage
        self.assertTrue(os.path.exists(row["file_path"]))
        self.assertTrue(row["file_path"].startswith(os.path.join(self.tmp, "on", "incidents")))
        meta = json.loads(row["metadata"])
        self.assertEqual(meta["event_types"], [ZONE_ENTERED, DWELLING])
        self.assertFalse(meta["truncated"])
        # SQLite linkage: every clip event id is a stored event of the same session.
        with sqlite3.connect(db) as conn:
            stored = dict(conn.execute("SELECT event_id, session_id FROM events").fetchall())
        ids = json.loads(row["event_ids"])
        self.assertEqual(len(ids), 2)
        for event_id in ids:
            self.assertEqual(stored.get(event_id), row["session_id"])
        self.assertEqual(recorder_threads(), [])  # cleaned up on exit

    def test_events_identical_with_and_without_recorder(self) -> None:
        _, db_off, _ = self.run_app("off", 300)
        _, db_on, _ = self.run_app("on", 300, "--incidents")
        self.assertTrue(self.events(db_off))
        self.assertEqual(self.events(db_off), self.events(db_on))

    def test_shutdown_mid_clip_finalises_and_stops_threads(self) -> None:
        # Ends ~8 s after the entry trigger, before the 12 s post-roll is over.
        start = time.monotonic()
        code, db, _ = self.run_app("stop", 100, "--incidents")
        self.assertEqual(code, 0)
        self.assertLess(time.monotonic() - start, 30.0)
        rows = self.incidents(db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "complete")
        self.assertTrue(json.loads(rows[0]["metadata"])["truncated"])
        self.assertEqual(recorder_threads(), [])
        self.assertTrue(self.writers and self.writers[0].frames > 0)

    def test_recorder_errors_never_stop_the_app(self) -> None:
        _, db_off, _ = self.run_app("off", 150)
        with mock.patch.object(IncidentRecorder, "on_events", side_effect=RuntimeError("boom")), \
                mock.patch.object(IncidentRecorder, "push_frame", side_effect=RuntimeError("boom")):
            code, db_on, _ = self.run_app("broken", 150, "--incidents")
        self.assertEqual(code, 0)
        self.assertEqual(self.events(db_off), self.events(db_on))


if __name__ == "__main__":
    unittest.main()
