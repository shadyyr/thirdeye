"""Event persistence for the Third Eye event platform.

Events are the plain dicts produced by ``thirdeye.core.types.make_event``.
Storage must never crash the live demo, so every sink catches its own
exceptions, reports the first failure to stderr and carries on.

Sinks:
    SQLiteEventStore  primary store (data/events.sqlite3), queryable
    JsonlEventSink    optional append-only mirror, one JSON object per line
    MultiSink         fan-out to several sinks with failure isolation

Any other backend (e.g. Postgres / Tiger Data) only has to implement the
``EventSink`` protocol: ``write(event)`` and ``close()``.

Entity memory (table ``entities``, one row per session + entity, written by
``thirdeye.state.entity_memory.EntityMemory`` when a track ends):
    SQLiteEventStore.upsert_entity(summary_dict)
    SQLiteEventStore.last_seen(entity_id, session_id=None)
    SQLiteEventStore.entities(session_id=None, class_name=None, limit=50)

CLI:
    python -m thirdeye.storage.event_store --recent 20 [--db PATH]
        [--type EVENT_TYPE] [--since-minutes N] [--json]
    python -m thirdeye.storage.event_store --entity 2 [--session ID]
    python -m thirdeye.storage.event_store --entities [--session ID]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime
from typing import Any, Iterable, Protocol, runtime_checkable

from thirdeye.core import settings

_COLUMNS = (
    "event_id",
    "session_id",
    "timestamp",
    "iso_time",
    "event_type",
    "entity_id",
    "object_id",
    "zone_id",
    "severity",
    "message",
    "metadata",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp  REAL,
    iso_time   TEXT,
    event_type TEXT,
    entity_id  INTEGER,
    object_id  INTEGER,
    zone_id    TEXT,
    severity   TEXT,
    message    TEXT,
    metadata   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE TABLE IF NOT EXISTS entities (
    session_id           TEXT,
    entity_id            INTEGER,
    class_name           TEXT,
    status               TEXT,
    first_seen           REAL,
    last_seen            REAL,
    last_bbox            TEXT,
    last_center          TEXT,
    last_zone_id         TEXT,
    last_zone_name       TEXT,
    associated_person_id INTEGER,
    relationship_state   TEXT,
    previous_person_id   INTEGER,
    left_with_person_id  INTEGER,
    event_ids            TEXT,
    snapshot_path        TEXT,
    updated_at           REAL,
    PRIMARY KEY (session_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entities_last_seen ON entities(session_id, last_seen);
"""

_ENTITY_COLUMNS = (
    "session_id",
    "entity_id",
    "class_name",
    "status",
    "first_seen",
    "last_seen",
    "last_bbox",
    "last_center",
    "last_zone_id",
    "last_zone_name",
    "associated_person_id",
    "relationship_state",
    "previous_person_id",
    "left_with_person_id",
    "event_ids",
    "snapshot_path",
    "updated_at",
)
_ENTITY_JSON = ("last_bbox", "last_center", "event_ids")


@runtime_checkable
class EventSink(Protocol):
    """Anything that can persist events."""

    def write(self, event: dict) -> None: ...

    def close(self) -> None: ...


def default_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # e.g. circular references or non-str dict keys
        return json.dumps(str(value))


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return time.time()


def _opt_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


class _ErrorReporter:
    """Print the first failure of a sink to stderr, count the rest silently."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.error_count = 0
        self.last_error: str | None = None

    def report(self, action: str, exc: BaseException) -> None:
        self.error_count += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.error_count == 1:
            print(
                f"[thirdeye.storage] {self.name} {action} failed: {self.last_error} "
                "(further errors from this sink are suppressed)",
                file=sys.stderr,
                flush=True,
            )


class SQLiteEventStore:
    """SQLite-backed event store (WAL mode). Implements EventSink."""

    def __init__(self, path: str = settings.EVENT_DB_PATH, session_id: str | None = None) -> None:
        self.path = path
        self.session_id = session_id or default_session_id()
        self._errors = _ErrorReporter(f"SQLiteEventStore({path})")
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            path, check_same_thread=False, timeout=5.0
        )
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass  # e.g. read-only media; plain journal still works
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def error_count(self) -> int:
        return self._errors.error_count

    def _row(self, event: dict) -> tuple:
        ts = _to_float(event.get("timestamp"))
        try:
            iso_time = datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")
        except (OverflowError, OSError, ValueError):
            iso_time = None
        event_id = event.get("event_id") or f"evt-{int(ts)}-{id(event):x}"
        return (
            str(event_id),
            self.session_id,
            ts,
            iso_time,
            _opt_str(event.get("event_type")),
            _to_int(event.get("entity_id")),
            _to_int(event.get("object_id")),
            _opt_str(event.get("zone_id")),
            _opt_str(event.get("severity")),
            _opt_str(event.get("message")),
            _to_json(event.get("metadata") or {}),
        )

    _INSERT = f"INSERT OR REPLACE INTO events ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})"

    def write(self, event: dict) -> None:
        self.write_many([event])

    def write_many(self, events: Iterable[dict]) -> None:
        try:
            rows = [self._row(e) for e in events]
            if not rows:
                return
            with self._lock:
                if self._conn is None:
                    raise sqlite3.ProgrammingError("store is closed")
                with self._conn:  # one transaction
                    self._conn.executemany(self._INSERT, rows)
        except Exception as exc:
            self._errors.report("write", exc)

    def recent(
        self, limit: int = 50, event_type: str | None = None, since: float | None = None
    ) -> list[dict]:
        """Newest first, metadata decoded back to a dict."""
        clauses, params = [], []
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(float(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM events {where} ORDER BY timestamp DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(zip(_COLUMNS, row))
            try:
                item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else {}
            except (TypeError, ValueError):
                item["metadata"] = {"raw": item["metadata"]}
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Entity memory
    # ------------------------------------------------------------------
    _UPSERT_ENTITY = (
        f"INSERT OR REPLACE INTO entities ({', '.join(_ENTITY_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(_ENTITY_COLUMNS))})"
    )

    def upsert_entity(self, summary: dict) -> None:
        """Insert or replace one entity summary (EntitySummary.to_dict())."""
        try:
            entity_id = _to_int(summary.get("entity_id"))
            if entity_id is None:
                raise ValueError("entity summary without entity_id")
            row = (
                str(summary.get("session_id") or self.session_id),
                entity_id,
                _opt_str(summary.get("class_name")),
                _opt_str(summary.get("status")),
                _opt_float(summary.get("first_seen")),
                _opt_float(summary.get("last_seen")),
                _to_json(list(summary.get("last_bbox") or [])),
                _to_json(list(summary.get("last_center") or [])),
                _opt_str(summary.get("last_zone_id")),
                _opt_str(summary.get("last_zone_name")),
                _to_int(summary.get("associated_person_id")),
                _opt_str(summary.get("relationship_state")),
                _to_int(summary.get("previous_person_id")),
                _to_int(summary.get("left_with_person_id")),
                _to_json(list(summary.get("event_ids") or [])),
                _opt_str(summary.get("snapshot_path")),
                _to_float(summary.get("updated_at")),
            )
            with self._lock:
                if self._conn is None:
                    raise sqlite3.ProgrammingError("store is closed")
                with self._conn:
                    self._conn.execute(self._UPSERT_ENTITY, row)
        except Exception as exc:
            self._errors.report("upsert_entity", exc)

    def latest_session(self) -> str | None:
        """Session with the most recent entity update or event."""
        sql = (
            "SELECT session_id FROM ("
            " SELECT session_id, MAX(updated_at) AS t FROM entities GROUP BY session_id"
            " UNION ALL"
            " SELECT session_id, MAX(timestamp) AS t FROM events GROUP BY session_id"
            ") WHERE session_id IS NOT NULL ORDER BY t DESC LIMIT 1"
        )
        try:
            with self._lock:
                if self._conn is None:
                    return None
                row = self._conn.execute(sql).fetchone()
            return None if row is None else row[0]
        except Exception as exc:
            self._errors.report("latest_session", exc)
            return None

    def _default_session(self) -> str | None:
        """This store's own session if it has recorded anything, else the latest one.

        A live app asks about its own session; the CLI (session "cli")
        falls through to the most recent recorded session.
        """
        sql = (
            "SELECT 1 FROM entities WHERE session_id = ? "
            "UNION ALL SELECT 1 FROM events WHERE session_id = ? LIMIT 1"
        )
        try:
            with self._lock:
                if self._conn is None:
                    return None
                own = self._conn.execute(sql, [self.session_id, self.session_id]).fetchone()
        except Exception as exc:
            self._errors.report("session lookup", exc)
            own = None
        return self.session_id if own else self.latest_session()

    def _entity_rows(self, where: str, params: list, limit: int) -> list[dict]:
        sql = (
            f"SELECT {', '.join(_ENTITY_COLUMNS)} FROM entities WHERE {where} "
            "ORDER BY last_seen DESC, entity_id DESC LIMIT ?"
        )
        try:
            with self._lock:
                if self._conn is None:
                    return []
                rows = self._conn.execute(sql, [*params, int(limit)]).fetchall()
        except Exception as exc:
            self._errors.report("entities query", exc)
            return []
        result = []
        for row in rows:
            item = dict(zip(_ENTITY_COLUMNS, row))
            for key in _ENTITY_JSON:
                try:
                    item[key] = json.loads(item[key]) if item[key] else []
                except (TypeError, ValueError):
                    item[key] = []
            result.append(item)
        return result

    def last_seen(self, entity_id: int, session_id: str | None = None) -> dict | None:
        """Stored summary of one entity in a session (own session, else the latest)."""
        session_id = session_id or self._default_session()
        if session_id is None:
            return None
        rows = self._entity_rows("session_id = ? AND entity_id = ?", [session_id, int(entity_id)], 1)
        return rows[0] if rows else None

    def entities(
        self,
        session_id: str | None = None,
        class_name: str | None = None,
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict]:
        """Entity summaries of a session (latest by default), most recently seen first."""
        session_id = session_id or self._default_session()
        if session_id is None:
            return []
        clauses, params = ["session_id = ?"], [session_id]
        if class_name is not None:
            clauses.append("class_name = ?")
            params.append(class_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        return self._entity_rows(" AND ".join(clauses), params, limit)

    def events_for_entity(
        self, entity_id: int, session_id: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Events where entity_id or object_id matches, newest first."""
        session_id = session_id or self._default_session()
        if session_id is None:
            return []
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM events WHERE session_id = ? "
            "AND (entity_id = ? OR object_id = ?) ORDER BY timestamp DESC, rowid DESC LIMIT ?"
        )
        try:
            with self._lock:
                if self._conn is None:
                    return []
                rows = self._conn.execute(sql, [session_id, int(entity_id), int(entity_id), int(limit)]).fetchall()
        except Exception as exc:
            self._errors.report("events query", exc)
            return []
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row) -> dict:
        item = dict(zip(_COLUMNS, row))
        try:
            item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else {}
        except (TypeError, ValueError):
            item["metadata"] = {"raw": item["metadata"]}
        return item

    def count(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as exc:
                    self._errors.report("close", exc)
                self._conn = None

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JsonlEventSink:
    """Append-only JSON Lines mirror, flushed after every write."""

    def __init__(self, path: str, session_id: str | None = None) -> None:
        self.path = path
        self.session_id = session_id
        self._errors = _ErrorReporter(f"JsonlEventSink({path})")
        self._lock = threading.Lock()
        self._fh = None
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
        except Exception as exc:
            self._errors.report("open", exc)

    @property
    def error_count(self) -> int:
        return self._errors.error_count

    def write(self, event: dict) -> None:
        try:
            if self._fh is None:
                raise OSError("file is not open")
            record = dict(event)
            if self.session_id is not None:
                record.setdefault("session_id", self.session_id)
            line = _to_json(record)
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception as exc:
            self._errors.report("write", exc)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception as exc:
                    self._errors.report("close", exc)
                self._fh = None


class MultiSink:
    """Fan events out to several sinks; one failing sink never affects others."""

    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self.sinks = [s for s in sinks if s is not None]
        self._errors = {id(s): _ErrorReporter(f"MultiSink[{type(s).__name__}]") for s in self.sinks}

    def write(self, event: dict) -> None:
        for sink in self.sinks:
            try:
                sink.write(event)
            except Exception as exc:
                self._errors[id(sink)].report("write", exc)

    def write_many(self, events: Iterable[dict]) -> None:
        events = list(events)
        for sink in self.sinks:
            try:
                bulk = getattr(sink, "write_many", None)
                if callable(bulk):
                    bulk(events)
                else:
                    for event in events:
                        sink.write(event)
            except Exception as exc:
                self._errors[id(sink)].report("write", exc)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:
                self._errors[id(sink)].report("close", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_events_table(events: list[dict]) -> str:
    if not events:
        return "(no events)"
    headers = ("time", "type", "sev", "entity", "object", "zone", "message")
    rows = []
    for e in events:
        try:
            when = datetime.fromtimestamp(float(e["timestamp"])).strftime("%m-%d %H:%M:%S")
        except Exception:
            when = "?"
        rows.append(
            (
                when,
                str(e.get("event_type") or ""),
                str(e.get("severity") or ""),
                "" if e.get("entity_id") is None else str(e["entity_id"]),
                "" if e.get("object_id") is None else str(e["object_id"]),
                str(e.get("zone_id") or ""),
                str(e.get("message") or ""),
            )
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:-1]):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths[:-1]) + "  {}"
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths[:-1]), "-" * 7)]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def _fmt_time(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M:%S")
    except Exception:
        return "?"


def _label(class_name: Any, entity_id: Any) -> str:
    return f"{str(class_name or 'entity').capitalize()} #{entity_id}"


def format_entity(row: dict) -> str:
    label = _label(row.get("class_name"), row.get("entity_id"))
    lines = [f"{label}  status={row.get('status')}  session={row.get('session_id')}"]
    lines.append(f"  first seen: {_fmt_time(row.get('first_seen'))}")
    lines.append(f"  last seen:  {_fmt_time(row.get('last_seen'))}")
    zone = row.get("last_zone_name") or row.get("last_zone_id")
    lines.append(f"  last zone:  {zone or '-'}")
    center = row.get("last_center") or []
    if len(center) == 2:
        lines.append(f"  last position (px): {center[0]:.0f}, {center[1]:.0f}")
    if row.get("associated_person_id") is not None:
        lines.append(f"  associated with: Person #{row['associated_person_id']}"
                     f" ({row.get('relationship_state') or '-'})")
    if row.get("previous_person_id") is not None:
        lines.append(f"  previously associated with: Person #{row['previous_person_id']}")
    if row.get("left_with_person_id") is not None:
        lines.append(f"  left view with: Person #{row['left_with_person_id']}")
    lines.append(f"  snapshot: {row.get('snapshot_path') or '-'}")
    return "\n".join(lines)


def _cli_entity(store: "SQLiteEventStore", args: argparse.Namespace) -> int:
    session = args.session or store.latest_session()
    row = store.last_seen(args.entity, session_id=session) if session else None
    events = store.events_for_entity(args.entity, session_id=session, limit=args.recent) if session else []
    if args.json:
        print(json.dumps({"session_id": session, "entity": row, "events": events}, indent=2, default=str))
        return 0 if (row or events) else 1
    if row is None:
        print(f"No remembered entity #{args.entity} in session {session or '-'} "
              "(it may still be live, or the session did not record entities)")
    else:
        print(format_entity(row))
    print(f"\nEvents for #{args.entity} in session {session or '-'} (newest first):")
    print(format_events_table(events))
    return 0 if (row or events) else 1


def _cli_entities(store: "SQLiteEventStore", args: argparse.Namespace) -> int:
    session = args.session or store.latest_session()
    rows = store.entities(session_id=session, limit=max(args.recent, 1)) if session else []
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    print(f"Session {session or '-'}: {len(rows)} remembered entities (most recently seen first)")
    if not rows:
        return 0
    headers = ("entity", "status", "last seen", "last zone", "assoc", "left with", "snapshot")
    table = []
    for r in rows:
        table.append((
            _label(r.get("class_name"), r.get("entity_id")),
            str(r.get("status") or ""),
            _fmt_time(r.get("last_seen")),
            str(r.get("last_zone_name") or r.get("last_zone_id") or "-"),
            "-" if r.get("associated_person_id") is None else f"Person #{r['associated_person_id']}",
            "-" if r.get("left_with_person_id") is None else f"Person #{r['left_with_person_id']}",
            str(r.get("snapshot_path") or "-"),
        ))
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in table:
        print(fmt.format(*row))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m thirdeye.storage.event_store", description="Show stored Third Eye events."
    )
    parser.add_argument("--recent", type=int, default=20, metavar="N", help="number of events (default 20)")
    parser.add_argument("--db", default=settings.EVENT_DB_PATH, help=f"database path (default {settings.EVENT_DB_PATH})")
    parser.add_argument("--type", dest="event_type", default=None, help="only this event type")
    parser.add_argument("--since-minutes", type=float, default=None, help="only events from the last N minutes")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--entity", type=int, default=None, metavar="ID",
                        help="where/when entity ID (e.g. Backpack #2 -> 2) was last seen, plus its events")
    parser.add_argument("--entities", action="store_true", help="list remembered entities of the session")
    parser.add_argument("--session", default=None, help="session id (default: latest session)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"No event database at {args.db}", file=sys.stderr)
        return 1
    if args.entity is not None or args.entities:
        store = SQLiteEventStore(args.db, session_id="cli")
        try:
            if args.entity is not None:
                return _cli_entity(store, args)
            return _cli_entities(store, args)
        finally:
            store.close()
    since = time.time() - args.since_minutes * 60 if args.since_minutes is not None else None
    store = SQLiteEventStore(args.db, session_id="cli")
    try:
        events = store.recent(args.recent, event_type=args.event_type, since=since)
        total = store.count()
    finally:
        store.close()
    if args.json:
        print(json.dumps(events, indent=2, default=str))
    else:
        print(f"{args.db}: {total} events total, showing {len(events)} newest")
        print(format_events_table(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
