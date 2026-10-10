from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from thirdeye.core.types import make_event
from thirdeye.storage.event_store import (
    EventSink,
    JsonlEventSink,
    MultiSink,
    SQLiteEventStore,
    main,
)


class _BrokenSink:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, event: dict) -> None:
        self.calls += 1
        raise RuntimeError("boom")

    def close(self) -> None:
        raise RuntimeError("boom on close")


class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.closed = False

    def write(self, event: dict) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


class EventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "nested", "events.sqlite3")
        self.store = SQLiteEventStore(self.db_path, session_id="test-session")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_creates_parent_dir_and_wal(self) -> None:
        self.assertTrue(os.path.exists(self.db_path))
        mode = self.store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")
        self.assertIsInstance(self.store, EventSink)

    def test_roundtrip(self) -> None:
        event = make_event(
            "ZONE_ENTERED", "Person #1 entered Restricted Zone A",
            severity="warning", entity_id=1, zone_id="restricted_a", timestamp=1000.0,
            metadata={"zone_name": "Restricted Zone A", "anchor": [10.5, 20.0]},
        )
        self.store.write(event)
        self.assertEqual(self.store.count(), 1)
        (row,) = self.store.recent()
        for key in ("event_id", "timestamp", "event_type", "entity_id", "object_id",
                    "zone_id", "severity", "message", "metadata"):
            self.assertEqual(row[key], event[key], key)
        self.assertEqual(row["session_id"], "test-session")
        self.assertTrue(row["iso_time"])

    def test_write_many_and_order(self) -> None:
        events = [make_event("DWELLING", f"e{i}", entity_id=i, timestamp=100.0 + i) for i in range(5)]
        self.store.write_many(events)
        self.assertEqual(self.store.count(), 5)
        rows = self.store.recent(limit=3)
        self.assertEqual([r["message"] for r in rows], ["e4", "e3", "e2"])

    def test_filters(self) -> None:
        self.store.write_many([
            make_event("ZONE_ENTERED", "a", timestamp=10.0),
            make_event("ZONE_EXITED", "b", timestamp=20.0),
            make_event("ZONE_ENTERED", "c", timestamp=30.0),
        ])
        self.assertEqual([r["message"] for r in self.store.recent(event_type="ZONE_ENTERED")], ["c", "a"])
        self.assertEqual([r["message"] for r in self.store.recent(since=15.0)], ["c", "b"])
        self.assertEqual([r["message"] for r in self.store.recent(event_type="ZONE_ENTERED", since=15.0)], ["c"])

    def test_metadata_json_and_empty(self) -> None:
        self.store.write(make_event("OBJECT_LOST", "x", object_id=5, metadata={"n": {"deep": [1, 2]}}))
        self.store.write(make_event("OBJECT_LOST", "y", object_id=6))
        raw = self.store._conn.execute("SELECT metadata FROM events WHERE object_id=5").fetchone()[0]
        self.assertEqual(json.loads(raw), {"n": {"deep": [1, 2]}})
        by_obj = {r["object_id"]: r for r in self.store.recent()}
        self.assertEqual(by_obj[6]["metadata"], {})

    def test_bad_metadata_does_not_raise(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        self.store.write(make_event("OBJECT_MOVED", "obj", metadata={"obj": object(), "s": {1, 2}}))
        self.store.write(make_event("OBJECT_MOVED", "circ", metadata=circular))
        self.store.write({"event_type": "WEIRD", "entity_id": "not-an-int", "timestamp": "nope"})
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(self.store.error_count, 0)

    def test_write_after_close_does_not_raise(self) -> None:
        self.store.close()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.store.write(make_event("ZONE_ENTERED", "late"))
            self.store.write(make_event("ZONE_ENTERED", "late2"))
        self.assertEqual(self.store.error_count, 2)
        self.assertEqual(err.getvalue().count("failed"), 1)  # reported once

    def test_cli_recent(self) -> None:
        self.store.write(make_event("CROWDING_STARTED", "Crowding in Monitored Area", zone_id="monitored_main"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--recent", "5", "--db", self.db_path])
        self.assertEqual(code, 0)
        self.assertIn("CROWDING_STARTED", out.getvalue())
        self.assertIn("Crowding in Monitored Area", out.getvalue())


def _summary(entity_id: int, **overrides) -> dict:
    from thirdeye.core.types import EntitySummary

    data = EntitySummary(
        entity_id=entity_id, class_name="backpack", status="ARCHIVED",
        first_seen=100.0, last_seen=118.0, last_bbox=(1.0, 2.0, 3.0, 4.0), last_center=(2.0, 3.0),
        last_zone_id="restricted_a", last_zone_name="Restricted Zone A",
        associated_person_id=1, relationship_state="ASSOCIATED", left_with_person_id=8,
        event_ids=["evt-1", "evt-2"], snapshot_path="data/snapshots/s/backpack_2.jpg",
    ).to_dict()
    data.update(overrides)
    return data


class EntityStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "events.sqlite3")
        self.store = SQLiteEventStore(self.db_path, session_id="sess-b")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_migration_on_existing_db(self) -> None:
        legacy = os.path.join(self._tmp.name, "legacy.sqlite3")
        import sqlite3
        conn = sqlite3.connect(legacy)
        conn.execute("CREATE TABLE events (event_id TEXT PRIMARY KEY, session_id TEXT, timestamp REAL, "
                     "iso_time TEXT, event_type TEXT, entity_id INTEGER, object_id INTEGER, zone_id TEXT, "
                     "severity TEXT, message TEXT, metadata TEXT)")
        conn.commit()
        conn.close()
        store = SQLiteEventStore(legacy, session_id="x")
        store.upsert_entity(_summary(2))
        self.assertEqual(store.last_seen(2)["entity_id"], 2)
        store.close()

    def test_upsert_and_query(self) -> None:
        self.store.upsert_entity(_summary(2, left_with_person_id=None))
        self.store.upsert_entity(_summary(2))  # update in place
        self.store.upsert_entity(_summary(5, class_name="suitcase", last_seen=120.0))
        self.store.upsert_entity(_summary(9, class_name="person", last_seen=90.0))
        self.store.upsert_entity(_summary(2, session_id="sess-a", last_zone_name="Old"))
        self.assertEqual(self.store.error_count, 0)

        row = self.store.last_seen(2)
        self.assertEqual(row["session_id"], "sess-b")
        self.assertEqual(row["left_with_person_id"], 8)
        self.assertEqual(row["last_bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(row["event_ids"], ["evt-1", "evt-2"])
        self.assertEqual(row["last_zone_name"], "Restricted Zone A")
        self.assertEqual(self.store.last_seen(2, session_id="sess-a")["last_zone_name"], "Old")
        self.assertIsNone(self.store.last_seen(77))

        self.assertEqual([r["entity_id"] for r in self.store.entities()], [5, 2, 9])
        self.assertEqual([r["entity_id"] for r in self.store.entities(class_name="person")], [9])
        self.assertEqual(len(self.store.entities(limit=1)), 1)

    def test_bad_summary_does_not_raise(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.store.upsert_entity({"class_name": "backpack"})
            self.store.upsert_entity({"entity_id": 3, "last_bbox": None, "first_seen": "x"})
        self.assertEqual(self.store.error_count, 1)
        self.assertEqual(self.store.last_seen(3)["last_bbox"], [])

    def test_cli_entity_and_entities(self) -> None:
        self.store.upsert_entity(_summary(2))
        self.store.write_many([
            make_event("OBJECT_LEFT_WITH", "Backpack #2 left view with Person #8",
                       entity_id=8, object_id=2, timestamp=119.0),
            make_event("ZONE_ENTERED", "Backpack #2 entered Restricted Zone A", entity_id=2, timestamp=101.0),
            make_event("ZONE_ENTERED", "Person #3 entered", entity_id=3, timestamp=102.0),
        ])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--db", self.db_path, "--entity", "2"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Backpack #2", text)
        self.assertIn("Restricted Zone A", text)
        self.assertIn("left view with: Person #8", text)
        self.assertIn("backpack_2.jpg", text)
        self.assertIn("OBJECT_LEFT_WITH", text)
        self.assertNotIn("Person #3 entered", text)
        self.assertTrue(text.isascii())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--db", self.db_path, "--entities"])
        self.assertEqual(code, 0)
        self.assertIn("Backpack #2", out.getvalue())
        self.assertIn("ARCHIVED", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--db", self.db_path, "--entity", "42"])
        self.assertEqual(code, 1)
        self.assertIn("No remembered entity #42", out.getvalue())


class SinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_jsonl_sink(self) -> None:
        path = os.path.join(self._tmp.name, "events.jsonl")
        sink = JsonlEventSink(path, session_id="s1")
        sink.write(make_event("ZONE_ENTERED", "a", metadata={"o": object()}))
        sink.write(make_event("ZONE_EXITED", "b"))
        with open(path, encoding="utf-8") as fh:  # flushed per write, readable before close
            lines = [json.loads(line) for line in fh]
        sink.close()
        self.assertEqual([l["message"] for l in lines], ["a", "b"])
        self.assertEqual(lines[0]["session_id"], "s1")

    def test_multisink_isolation(self) -> None:
        broken, good = _BrokenSink(), _ListSink()
        db = SQLiteEventStore(os.path.join(self._tmp.name, "m.sqlite3"))
        multi = MultiSink([broken, good, db])
        with contextlib.redirect_stderr(io.StringIO()) as err:
            for i in range(3):
                multi.write(make_event("ZONE_ENTERED", f"m{i}"))
            multi.write_many([make_event("ZONE_EXITED", "bulk")])
            self.assertEqual(db.count(), 4)
            multi.close()
        self.assertEqual(broken.calls, 4)
        self.assertEqual(len(good.events), 4)
        self.assertTrue(good.closed)
        self.assertEqual(err.getvalue().count("write failed"), 1)


if __name__ == "__main__":
    unittest.main()
