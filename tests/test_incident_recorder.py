from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import cv2
import numpy as np

from thirdeye.core.types import (
    DWELLING,
    ENTITY_ENTERED,
    UNATTENDED_OBJECT,
    ZONE_ENTERED,
    ZONE_EXITED,
    make_event,
)
from thirdeye.recording.incident_recorder import COMPLETE, FAILED, IncidentRecorder
from thirdeye.recording.rolling_buffer import RollingBuffer
from thirdeye.storage import event_store
from thirdeye.storage.event_store import SQLiteEventStore

W, H = 64, 48


def frame_for(index: int) -> np.ndarray:
    """Solid colour that encodes the frame index (survives JPEG)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = (index % 50) * 5
    img[:, :, 1] = (index // 50) * 10
    return img


def index_of(img: np.ndarray) -> int:
    b = float(img[:, :, 0].mean())
    g = float(img[:, :, 1].mean())
    return int(round(g / 10)) * 50 + int(round(b / 5))


class FakeStore:
    session_id = "sess"

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.calls = 0

    def upsert_incident(self, row: dict) -> None:
        self.calls += 1
        self.rows[row["incident_id"]] = row


class FakeWriter:
    def __init__(self, path: str, fail_on_write: bool = False) -> None:
        self.path = path
        self.indices: list[int] = []
        self.fail_on_write = fail_on_write

    def write(self, img) -> None:
        if self.fail_on_write:
            raise OSError("No space left on device")
        self.indices.append(index_of(img))

    def release(self) -> None:
        with open(self.path, "wb") as fh:
            fh.write(b"x" * max(1, len(self.indices)))


class FakeFactory:
    def __init__(self, fail_open: int = 0, fail_on_write: bool = False) -> None:
        self.writers: list[FakeWriter] = []
        self.fail_open = fail_open
        self.fail_on_write = fail_on_write

    def __call__(self, path, fps, size):
        if self.fail_open > 0:
            self.fail_open -= 1
            raise RuntimeError("no video writer could be opened")
        writer = FakeWriter(path, self.fail_on_write)
        self.writers.append(writer)
        return writer, "fake", path


def zone_entry(ts: float, zone_type: str | None = "RESTRICTED", zone_id: str = "restricted_a", entity: int = 1):
    return make_event(ZONE_ENTERED, f"Person #{entity} entered zone", entity_id=entity, zone_id=zone_id,
                      timestamp=ts, metadata={"class_name": "person", "zone_type": zone_type})


class RecorderCase(unittest.TestCase):
    FPS_IN = 20.0

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = FakeStore()
        self.factory = FakeFactory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make(self, **kw) -> IncidentRecorder:
        args = dict(store=self.store, out_dir=self._tmp.name, pre_roll=5.0, post_roll=10.0, max_clip=30.0,
                    fps=10.0, size=None, writer_factory=self.factory, realtime=False, min_free_mb=0,
                    restricted_zone_ids=["restricted_a"])
        args.update(kw)
        return IncidentRecorder(**args)

    def run_frames(self, rec: IncidentRecorder, seconds: float, events_at: dict | None = None,
                   start: float = 1000.0) -> None:
        """Feed 20 fps frames; frame index i has timestamp start + i/20."""
        events_at = dict(events_at or {})
        n = int(seconds * self.FPS_IN)
        for i in range(n):
            now = start + i / self.FPS_IN
            rec.push_frame(frame_for(i), now)
            rec.on_events([e(now) for e in events_at.pop(i, [])], now)

    def times(self, writer: FakeWriter, start: float = 1000.0) -> list[float]:
        return [start + i / self.FPS_IN for i in writer.indices]


class RollingBufferTests(unittest.TestCase):
    def test_time_bound(self) -> None:
        buf = RollingBuffer(seconds=2.0, fps=10.0, threaded=False)
        for i in range(100):
            buf.submit(frame_for(i), 100.0 + i * 0.1)
        self.assertLessEqual(buf.span(), 2.0 + 1e-6)
        self.assertGreaterEqual(len(buf), 20)
        self.assertLessEqual(len(buf), 21)

    def test_byte_bound(self) -> None:
        noisy = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
        one = len(RollingBuffer(threaded=False).encode(noisy))
        buf = RollingBuffer(seconds=60.0, fps=10.0, max_bytes=one * 5, threaded=False)
        for i in range(50):
            buf.submit(noisy, 100.0 + i * 0.1)
        self.assertLessEqual(buf.total_bytes, one * 5)
        self.assertLessEqual(len(buf), 5)
        self.assertGreater(buf.stats["evicted"], 0)

    def test_subsampling_to_buffer_fps(self) -> None:
        buf = RollingBuffer(seconds=60.0, fps=10.0, threaded=False)
        for i in range(90):  # 3 s at 30 fps
            buf.submit(frame_for(i), 100.0 + i / 30.0)
        self.assertIn(len(buf), (30, 31))
        ts = [t for t, _ in buf.frames_since(0)]
        gaps = np.diff(ts)
        self.assertAlmostEqual(float(np.mean(gaps)), 0.1, delta=0.01)

    def test_threaded_drops_instead_of_blocking(self) -> None:
        buf = RollingBuffer(seconds=5.0, fps=1000.0, threaded=True, queue_size=1, size=(640, 480))
        big = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(200):
            buf.submit(big, 100.0 + i * 0.01)
        buf.drain(2.0)
        buf.close()
        self.assertGreater(buf.stats["dropped"], 0)
        self.assertGreater(buf.stats["encoded"], 0)


class IncidentRecorderTests(RecorderCase):
    def test_pre_and_post_roll(self) -> None:
        rec = self.make()
        self.run_frames(rec, 30.0, {200: [zone_entry]})  # trigger at t=10 s
        rec.close()
        self.assertEqual(len(self.store.rows), 1)
        row = next(iter(self.store.rows.values()))
        self.assertEqual(row["status"], COMPLETE)
        ts = self.times(self.factory.writers[0])
        self.assertAlmostEqual(ts[0], 1005.0, delta=0.11)  # pre-roll 5 s
        self.assertAlmostEqual(ts[-1], 1020.0, delta=0.11)  # post-roll 10 s
        self.assertIn(len(ts), range(148, 152))  # 15 s at 10 fps
        self.assertEqual(row["frame_count"], len(ts))
        self.assertAlmostEqual(row["clip_end"] - row["clip_start"], 15.0, delta=0.15)
        self.assertEqual(row["entity_ids"], [1])
        self.assertEqual(row["zone_id"], "restricted_a")
        self.assertTrue(row["file_path"].endswith("_zone_entered.mp4"))
        self.assertIn(os.path.join(self._tmp.name, "sess"), row["file_path"])

    def test_merge_restricted_entry_and_dwelling(self) -> None:
        dwell = lambda t: make_event(DWELLING, "Person #1 dwelling", entity_id=1, zone_id="restricted_a",  # noqa: E731
                                     timestamp=t, metadata={"zone_type": "RESTRICTED"})
        unattended = lambda t: make_event(UNATTENDED_OBJECT, "UNATTENDED OBJECT", entity_id=1,  # noqa: E731
                                          object_id=7, timestamp=t)
        rec = self.make()
        self.run_frames(rec, 40.0, {200: [zone_entry], 300: [dwell, unattended]})  # t=10 and t=15
        rec.close()
        self.assertEqual(len(self.store.rows), 1)
        row = next(iter(self.store.rows.values()))
        self.assertEqual(len(row["event_ids"]), 3)
        self.assertEqual(row["metadata"]["event_types"], [ZONE_ENTERED, DWELLING, UNATTENDED_OBJECT])
        self.assertEqual(row["event_type"], ZONE_ENTERED)
        self.assertEqual(sorted(row["entity_ids"]), [1, 7])
        ts = self.times(self.factory.writers[0])
        self.assertAlmostEqual(ts[0], 1005.0, delta=0.11)
        self.assertAlmostEqual(ts[-1], 1025.0, delta=0.11)  # extended to 15 + 10
        self.assertEqual(rec.stats["merged"], 2)
        hist = rec.history()
        self.assertEqual(hist[0]["event_count"], 3)
        self.assertEqual(hist[0]["title"], "RESTRICTED ZONE ENTRY")

    def test_non_triggers_ignored(self) -> None:
        entered = lambda t: make_event(ENTITY_ENTERED, "Person #1 entered view", entity_id=1, timestamp=t)  # noqa: E731
        exited = lambda t: make_event(ZONE_EXITED, "left", entity_id=1, zone_id="restricted_a", timestamp=t,  # noqa: E731
                                      metadata={"zone_type": "RESTRICTED"})
        monitored = lambda t: zone_entry(t, zone_type="MONITORED", zone_id="monitored_main")  # noqa: E731
        no_type_other = lambda t: zone_entry(t, zone_type=None, zone_id="monitored_main")  # noqa: E731
        rec = self.make()
        self.run_frames(rec, 8.0, {20: [entered], 40: [exited], 60: [monitored], 80: [no_type_other]})
        rec.close()
        self.assertEqual(self.store.rows, {})
        self.assertEqual(rec.stats["triggers"], 0)
        self.assertTrue(rec.is_trigger(zone_entry(0, zone_type=None, zone_id="restricted_a")))

    def test_max_clip_cap(self) -> None:
        dwell = lambda t: make_event(DWELLING, "dwelling", entity_id=2, zone_id="restricted_a", timestamp=t)  # noqa: E731
        events = {200 + k * 60: [dwell] for k in range(10)}  # every 3 s from t=10 to t=37
        rec = self.make(max_clip=12.0)
        self.run_frames(rec, 60.0, events)
        rec.close()
        rows = sorted(self.store.rows.values(), key=lambda r: r["trigger_time"])
        self.assertGreaterEqual(len(rows), 2)
        first = self.times(self.factory.writers[0])
        self.assertAlmostEqual(first[-1] - first[0], 12.0, delta=0.15)
        for row in rows:
            self.assertLessEqual(row["clip_end"] - row["clip_start"], 12.0 + 0.11)
        self.assertGreater(len(rows[0]["event_ids"]), 1)

    def test_writer_open_failure_marks_failed_and_continues(self) -> None:
        self.factory.fail_open = 1
        rec = self.make()
        self.run_frames(rec, 60.0, {200: [zone_entry], 800: [zone_entry]})  # t=10, t=40
        rec.close()
        rows = sorted(self.store.rows.values(), key=lambda r: r["trigger_time"])
        self.assertEqual([r["status"] for r in rows], [FAILED, COMPLETE])
        self.assertIn("no video writer", rows[0]["metadata"]["error"])

    def test_write_error_marks_failed_without_raising(self) -> None:
        self.factory.fail_on_write = True
        rec = self.make()
        self.run_frames(rec, 20.0, {200: [zone_entry]})
        rec.close()
        row = next(iter(self.store.rows.values()))
        self.assertEqual(row["status"], FAILED)
        self.assertIn("No space", row["metadata"]["error"])

    def test_store_failure_does_not_raise(self) -> None:
        class BrokenStore:
            session_id = "s"

            def upsert_incident(self, row):
                raise sqlite3.OperationalError("database is locked")

        rec = self.make(store=BrokenStore())
        self.run_frames(rec, 20.0, {100: [zone_entry]})
        rec.close()
        self.assertEqual(rec.history()[0]["status"], COMPLETE)

    def test_close_during_recording(self) -> None:
        rec = self.make()
        self.run_frames(rec, 12.0, {200: [zone_entry]})  # trigger at t=10, stop at t=12
        self.assertTrue(rec.recording)
        rec.close()
        row = next(iter(self.store.rows.values()))
        self.assertEqual(row["status"], COMPLETE)
        self.assertTrue(row["metadata"]["truncated"])
        ts = self.times(self.factory.writers[0])
        self.assertAlmostEqual(ts[0], 1005.0, delta=0.11)
        self.assertAlmostEqual(ts[-1], 1011.9, delta=0.11)
        rec.push_frame(frame_for(0), 2000.0)  # after close: no-ops, no errors
        rec.on_events([zone_entry(2000.0)], 2000.0)
        rec.close()

    def test_cooldown(self) -> None:
        rec = self.make(post_roll=2.0, cooldown=5.0)
        self.run_frames(rec, 30.0, {200: [zone_entry], 260: [zone_entry], 400: [zone_entry]})
        rec.close()  # t=10 incident ends at 12; t=13 inside cooldown; t=20 records
        self.assertEqual(len(self.store.rows), 2)
        self.assertEqual(rec.stats["ignored_cooldown"], 1)

    def test_threaded_mode_records(self) -> None:
        rec = self.make(realtime=True)
        for i in range(200):
            now = 1000.0 + i / 20.0
            rec.push_frame(frame_for(i), now)
            if i % 2 == 0:
                rec.buffer.drain(0.2)
            rec.on_events([zone_entry(now)] if i == 120 else [], now)
        rec.close()
        row = next(iter(self.store.rows.values()))
        self.assertEqual(row["status"], COMPLETE)
        self.assertGreater(row["frame_count"], 40)

    def test_real_mp4_clip_readback(self) -> None:
        rec = self.make(writer_factory=None, post_roll=2.0)
        self.run_frames(rec, 10.0, {100: [zone_entry]})  # t=5: clip 0..7 s
        rec.close()
        row = next(iter(self.store.rows.values()))
        self.assertEqual(row["status"], COMPLETE, row["metadata"])
        self.assertGreater(row["bytes"], 0)
        cap = cv2.VideoCapture(row["file_path"])
        count, first = 0, None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            first = img if first is None else first
            count += 1
        cap.release()
        self.assertEqual(count, row["frame_count"])
        self.assertIn(count, range(69, 72))
        self.assertLessEqual(abs(index_of(first) - 0), 1)  # clip starts at the oldest buffered frame


class IncidentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_migration_on_old_db_and_upsert(self) -> None:
        legacy = os.path.join(self._tmp.name, "legacy.sqlite3")
        # Schema of the baseline release (events + entities, no incidents table).
        conn = sqlite3.connect(legacy)
        conn.executescript(event_store._SCHEMA.split("CREATE TABLE IF NOT EXISTS incidents")[0])
        conn.execute("INSERT INTO events (event_id, session_id, timestamp) VALUES ('evt-0', 'old', 1.0)")
        conn.commit()
        self.assertIsNone(conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'incidents'").fetchone())
        conn.close()
        store = SQLiteEventStore(legacy, session_id="s1")
        row = {"incident_id": "inc-1", "event_type": ZONE_ENTERED, "trigger_time": 1000.0, "clip_start": 995.0,
               "clip_end": 1010.0, "event_ids": ["evt-1"], "entity_ids": [1, "7", None], "zone_id": "restricted_a",
               "file_path": "data/incidents/s1/inc-1_zone_entered.mp4", "status": "recording",
               "metadata": {"codec": "x264"}}
        store.upsert_incident(row)
        store.upsert_incident({**row, "status": "complete", "event_ids": ["evt-1", "evt-2"], "frame_count": 150,
                               "bytes": 123})
        got = store.incidents()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["status"], "complete")
        self.assertEqual(got[0]["event_ids"], ["evt-1", "evt-2"])
        self.assertEqual(got[0]["entity_ids"], [1, 7])
        self.assertEqual(got[0]["metadata"], {"codec": "x264"})
        self.assertEqual(got[0]["session_id"], "s1")
        self.assertEqual(store.incident("inc-1")["frame_count"], 150)
        self.assertEqual(store.incidents(status="failed"), [])
        store.upsert_incident({"event_type": "X"})  # no id: reported, not raised
        self.assertEqual(store.error_count, 1)
        self.assertEqual(store.recent(5)[0]["event_id"], "evt-0")  # old data still readable
        store.close()
        store.upsert_incident(row)  # closed: reported, not raised
        cli = SQLiteEventStore(legacy, session_id="cli")
        self.assertEqual(cli.incidents()[0]["incident_id"], "inc-1")  # falls back to the latest session
        cli.close()


if __name__ == "__main__":
    unittest.main()
