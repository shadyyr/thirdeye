"""Session memory of entities: where and when each one was last seen.

Answers "Where was Backpack #2 last seen?" after its live track has ended.
Call ``EntityMemory.observe`` once per frame AFTER world state, zones, events
and associations have been updated. Per frame it remembers, for every live
entity, its last box, zone and association; for portable objects also the
nearest person and a small best-quality snapshot crop. When a track ends
(``world.disappeared``) it freezes an ``EntitySummary`` (status ARCHIVED),
writes the snapshot JPEG once and persists the summary via the store.

"Left view with" inference (conservative wording, never ownership):
an object whose track ended while a person was near it shortly before its
last detection, and whose person track also ended within
LEFT_WITH_WINDOW_SECONDS, is recorded as having left view with that person
and an OBJECT_LEFT_WITH warning is emitted. If the associated person (or one
of its earlier tracker IDs, after an ID handoff) was the one near it, this is
only recorded: the association engine already reports it as OBJECT_LOST.

Person snapshots are never taken (no faces stored).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from thirdeye.core import settings
from thirdeye.core.types import (
    ACTIVE,
    ARCHIVED,
    OBJECT_LEFT_WITH,
    OUT_OF_VIEW,
    PORTABLE_OBJECT_CLASSES,
    WARNING,
    BBox,
    EntitySummary,
    entity_label,
    make_event,
)
from thirdeye.state.world_state import proximity_ratio

EVENT_IDS_PER_ENTITY = 50
SESSION_EVENT_LOG = 2000
SNAPSHOT_REFRESH_SECONDS = 0.5
SNAPSHOT_STALE_SECONDS = 3.0  # after this, a lower-confidence crop may replace the best
SNAPSHOT_MARGIN = 0.12  # fraction of the box size added on each side
SNAPSHOTS_IN_MEMORY_ARCHIVED = 32  # older archived crops are reloaded from disk
RECENT_NEAR_KEEP_SECONDS = 10.0


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _person(pid: int | None) -> str:
    return entity_label("person", pid) if pid is not None else "unknown person"


def format_age(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))} sec"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    hours = int(seconds // 3600)
    return f"{hours} h {int((seconds % 3600) // 60)} min"


@dataclass
class _Memory:
    entity_id: int
    class_name: str
    first_seen: float
    last_seen: float  # last actual detection (falls back to last tracker output)
    last_bbox: BBox
    last_center: tuple[float, float]
    status: str = ACTIVE
    archived_at: float | None = None
    last_zone_id: str | None = None
    last_zone_name: str | None = None
    in_zone_at_last_seen: bool = False
    associated_person_id: int | None = None
    relationship_state: str | None = None
    previous_person_id: int | None = None
    left_with_person_id: int | None = None
    # Earlier tracker IDs of the associated person (ID handoff): same human.
    same_person_ids: set[int] = field(default_factory=set)
    near_person_id: int | None = None  # nearest person at the last near moment
    near_time: float | None = None
    recent_near: dict[int, float] = field(default_factory=dict)  # person -> last near time
    snapshot: Any = None  # numpy BGR crop
    snapshot_conf: float = 0.0
    snapshot_time: float | None = None
    snapshot_path: str | None = None

    @property
    def is_object(self) -> bool:
        return self.class_name in PORTABLE_OBJECT_CLASSES

    @property
    def label(self) -> str:
        return entity_label(self.class_name, self.entity_id)


@dataclass
class _Pending:
    object_id: int
    archived_at: float
    last_detected: float
    associated_person_id: int | None
    candidates: list[int]  # nearest first


class EntityMemory:
    def __init__(
        self,
        store: Any = None,
        snapshot_dir: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.store = store
        self.snapshot_dir = snapshot_dir or _setting("SNAPSHOT_DIR", "data/snapshots")
        self.session_id = (
            session_id
            or getattr(store, "session_id", None)
            or datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.now: float = 0.0
        self._mem: dict[int, _Memory] = {}
        self._archived: OrderedDict[int, None] = OrderedDict()  # archive order, oldest first
        self._event_ids: dict[int, deque] = {}
        self._log: deque = deque(maxlen=SESSION_EVENT_LOG)
        self._pending: dict[int, _Pending] = {}
        self._store_errors = 0
        self._snapshot_errors = 0

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def observe(
        self,
        world,
        frame,
        associations,
        zone_manager,
        events: list[dict] | None,
        now: float,
    ) -> list[dict]:
        self.now = now
        new_events: list[dict] = []

        for track_id in getattr(world, "revived", ()) or ():
            self._on_revived(track_id, world)

        for event in events or ():
            self._record_event(event)

        people = [p for p in world.entities.values() if p.is_person]
        snapshots = bool(_setting("SNAPSHOTS_ENABLED", True)) and frame is not None
        for record in world.entities.values():
            mem = self._mem.get(record.track_id)
            if mem is None:
                mem = self._new_memory(record)
            self._update_live(mem, record, people, associations, zone_manager, now)
            if snapshots and mem.is_object and record.frames_missed == 0:
                self._maybe_snapshot(mem, record, frame, now)

        gone_people: list[_Memory] = []
        gone_objects: list[_Memory] = []
        for record in world.disappeared:
            mem = self._mem.get(record.track_id)
            if mem is None:
                mem = self._new_memory(record)
            self._archive(mem, record, now)
            (gone_objects if mem.is_object else gone_people).append(mem)

        for mem in gone_objects:
            new_events.extend(self._start_left_with(mem, now))
        for mem in gone_people:
            new_events.extend(self._resolve_left_with(mem, now))

        window = float(_setting("LEFT_WITH_WINDOW_SECONDS", 5.0))
        for object_id, pending in list(self._pending.items()):
            if now - pending.archived_at > window:
                del self._pending[object_id]

        for mem in gone_objects + gone_people:
            self._persist(mem)

        for event in new_events:
            self._record_event(event)
        return new_events

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def summary(self, entity_id: int) -> EntitySummary | None:
        mem = self._mem.get(entity_id)
        return None if mem is None else self._summary(mem)

    def recent_objects(self, now: float | None = None, limit: int | None = None) -> list[EntitySummary]:
        if limit is None:
            limit = int(_setting("RECENTLY_SEEN_LIMIT", 4))
        items = [
            m for m in self._mem.values() if m.is_object and m.status in (OUT_OF_VIEW, ARCHIVED)
        ]
        items.sort(key=lambda m: (m.last_seen, m.entity_id), reverse=True)
        return [self._summary(m) for m in items[: max(0, limit)]]

    def events_for(self, entity_id: int, limit: int = 20) -> list[dict]:
        result = []
        for event in reversed(self._log):
            if event.get("entity_id") == entity_id or event.get("object_id") == entity_id:
                result.append(event)
                if len(result) >= limit:
                    break
        return result

    def snapshot(self, entity_id: int):
        mem = self._mem.get(entity_id)
        if mem is None:
            return None
        if mem.snapshot is None and mem.snapshot_path and os.path.exists(mem.snapshot_path):
            try:
                import cv2

                return cv2.imread(mem.snapshot_path)
            except Exception:
                return None
        return mem.snapshot

    def where_last_seen(self, entity_id: int) -> str | None:
        mem = self._mem.get(entity_id)
        if mem is None:
            return None
        age = format_age(self.now - mem.last_seen)
        if mem.status == ACTIVE:
            text = f"{mem.label} is in view"
        elif mem.status == OUT_OF_VIEW:
            text = f"{mem.label} is out of view, last detected {age} ago"
        else:
            text = f"{mem.label} was last seen {age} ago"
        if mem.last_zone_name:
            if mem.in_zone_at_last_seen:
                text += f" in {mem.last_zone_name}"
            else:
                text += f" (last zone: {mem.last_zone_name})"
        near = self._near_at_last_seen(mem)
        if near is not None:
            text += f", near {_person(near)}"
        clauses = []
        if mem.left_with_person_id is not None:
            if mem.left_with_person_id == mem.associated_person_id:
                clauses.append(f"it left view with associated {_person(mem.left_with_person_id)}")
            else:
                clauses.append(f"it left view with {_person(mem.left_with_person_id)}")
        elif mem.associated_person_id is not None:
            verb = "associated with" if mem.status == ACTIVE else "last associated with"
            clauses.append(f"{verb} {_person(mem.associated_person_id)}")
        return "; ".join([text, *clauses]) + "."

    def close(self) -> None:
        # Persist entities that are still live so the session record is complete.
        for mem in self._mem.values():
            if mem.status != ARCHIVED:
                self._persist(mem)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _new_memory(self, record) -> _Memory:
        mem = _Memory(
            entity_id=record.track_id,
            class_name=record.class_name,
            first_seen=record.first_seen,
            last_seen=getattr(record, "last_detected", record.last_seen),
            last_bbox=tuple(record.bbox),
            last_center=tuple(record.center),
        )
        self._mem[record.track_id] = mem
        return mem

    def _update_live(self, mem, record, people, associations, zone_manager, now) -> None:
        mem.status = ACTIVE if record.frames_missed == 0 else OUT_OF_VIEW
        mem.archived_at = None
        mem.first_seen = record.first_seen
        if record.frames_missed == 0 or mem.last_seen < record.first_seen:
            mem.last_seen = record.last_detected
            mem.last_bbox = tuple(record.bbox)
            mem.last_center = tuple(record.center)
            zone_id = self._pick_zone(record.zones, zone_manager)
            mem.in_zone_at_last_seen = zone_id is not None
            if zone_id is not None:
                mem.last_zone_id = zone_id
                zone = zone_manager.get(zone_id) if zone_manager is not None else None
                mem.last_zone_name = getattr(zone, "name", None) or zone_id

        if not mem.is_object:
            return

        rel = associations.get(record.track_id) if associations is not None else None
        if rel is not None:
            mem.relationship_state = rel.state
            assoc = rel.last_associated_person_id
            handoff = set(getattr(rel, "handoff_ids", None) or ())
            if assoc is not None and assoc != mem.associated_person_id:
                # An ID handoff is the same person under a new tracker ID,
                # not a previous person.
                if mem.associated_person_id is not None and mem.associated_person_id not in handoff:
                    mem.previous_person_id = mem.associated_person_id
                mem.associated_person_id = assoc
            mem.same_person_ids = handoff

        # Proximity only while both are actually detected: a coasted box
        # is stale, so it is not evidence of who was near.
        if record.frames_missed == 0 and people:
            near_ratio = float(_setting("ASSOC_NEAR_RATIO", 0.5))
            best = None
            for person in people:
                if person.frames_missed:
                    continue
                ratio = proximity_ratio(record, person)
                if ratio < near_ratio:
                    mem.recent_near[person.track_id] = now
                    if best is None or ratio < best[0]:
                        best = (ratio, person.track_id)
            if best is not None:
                mem.near_person_id = best[1]
                mem.near_time = now
            if mem.recent_near:
                for pid in [p for p, t in mem.recent_near.items() if now - t > RECENT_NEAR_KEEP_SECONDS]:
                    del mem.recent_near[pid]

    @staticmethod
    def _pick_zone(zones, zone_manager) -> str | None:
        if not zones:
            return None
        ordered = sorted(zones)
        if zone_manager is not None:
            for zone_id in ordered:
                zone = zone_manager.get(zone_id)
                if zone is not None and zone.is_restricted:
                    return zone_id
        return ordered[0]

    def _maybe_snapshot(self, mem: _Memory, record, frame, now: float) -> None:
        conf = float(record.confidence)
        if conf < float(_setting("SNAPSHOT_MIN_CONFIDENCE", 0.3)):
            return
        if mem.snapshot is not None and mem.snapshot_time is not None:
            age = now - mem.snapshot_time
            if age < SNAPSHOT_REFRESH_SECONDS:
                return
            if conf < mem.snapshot_conf - 0.05 and age < SNAPSHOT_STALE_SECONDS:
                return
        crop = crop_snapshot(frame, record.bbox, int(_setting("SNAPSHOT_MAX_SIDE", 200)))
        if crop is not None:
            mem.snapshot = crop
            mem.snapshot_conf = conf
            mem.snapshot_time = now

    def _archive(self, mem: _Memory, record, now: float) -> None:
        mem.status = ARCHIVED
        mem.archived_at = now
        mem.first_seen = record.first_seen
        if record.last_detected >= mem.last_seen:
            mem.last_seen = record.last_detected
        self._archived.pop(mem.entity_id, None)
        self._archived[mem.entity_id] = None
        if mem.snapshot is not None:
            mem.snapshot_path = self._write_snapshot(mem) or mem.snapshot_path
        self._evict()

    def _evict(self) -> None:
        limit = max(0, int(_setting("ARCHIVE_MAX_ENTITIES", 1000)))
        while len(self._archived) > limit:
            old_id, _ = self._archived.popitem(last=False)
            self._mem.pop(old_id, None)
            self._event_ids.pop(old_id, None)
            self._pending.pop(old_id, None)
        # Keep only the newest archived crops in RAM; older ones live on disk.
        if len(self._archived) > SNAPSHOTS_IN_MEMORY_ARCHIVED:
            excess = len(self._archived) - SNAPSHOTS_IN_MEMORY_ARCHIVED
            for old_id in list(self._archived)[:excess]:
                mem = self._mem.get(old_id)
                if mem is not None and mem.snapshot is not None and mem.snapshot_path:
                    mem.snapshot = None

    def _write_snapshot(self, mem: _Memory) -> str | None:
        try:
            import cv2

            directory = os.path.join(self.snapshot_dir, str(self.session_id))
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{mem.class_name}_{mem.entity_id}.jpg")
            if not cv2.imwrite(path, mem.snapshot, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                raise OSError(f"cv2.imwrite failed for {path}")
            return path
        except Exception as exc:
            self._snapshot_errors += 1
            if self._snapshot_errors == 1:
                print(f"[thirdeye.memory] snapshot write failed: {exc}", flush=True)
            return None

    def _on_revived(self, track_id: int, world) -> None:
        self._pending.pop(track_id, None)
        mem = self._mem.get(track_id)
        if mem is None:
            return
        self._archived.pop(track_id, None)
        record = world.entities.get(track_id)
        mem.status = ACTIVE if record is None or record.frames_missed == 0 else OUT_OF_VIEW
        mem.archived_at = None
        mem.left_with_person_id = None
        self._persist(mem)

    def _near_at_last_seen(self, mem: _Memory) -> int | None:
        if mem.near_person_id is None or mem.near_time is None:
            return None
        if mem.last_seen - mem.near_time > float(_setting("LEFT_WITH_NEAR_SECONDS", 1.5)):
            return None
        return mem.near_person_id

    def _same_person(self, mem: _Memory, person_id: int) -> bool:
        """True for the associated person, including earlier IDs handed off to it."""
        return person_id == mem.associated_person_id or person_id in mem.same_person_ids

    def _start_left_with(self, mem: _Memory, now: float) -> list[dict]:
        near_window = float(_setting("LEFT_WITH_NEAR_SECONDS", 1.5))
        candidates = [
            (t, pid)
            for pid, t in mem.recent_near.items()
            if mem.last_seen - near_window <= t <= mem.last_seen + 1e-6
        ]
        if not candidates:
            return []
        candidates.sort(reverse=True)  # most recently near first
        # The associated person (any of its tracker IDs) carrying the object
        # out is already reported by the association engine (OBJECT_LOST):
        # just remember it, no second event.
        if any(self._same_person(mem, pid) for _, pid in candidates):
            mem.left_with_person_id = mem.associated_person_id
        others = [pid for _, pid in candidates if not self._same_person(mem, pid)]
        if not others:
            return []
        pending = _Pending(
            object_id=mem.entity_id,
            archived_at=now,
            last_detected=mem.last_seen,
            associated_person_id=mem.associated_person_id,
            candidates=others,
        )
        # A different person whose track already ended (people drop faster
        # than coasting objects) resolves immediately.
        for pid in others:
            person = self._mem.get(pid)
            if person is not None and person.status == ARCHIVED and person.archived_at is not None:
                if person.archived_at >= pending.last_detected - near_window:
                    return self._left_with(mem, pid, now)
        self._pending[mem.entity_id] = pending
        return []

    def _resolve_left_with(self, person: _Memory, now: float) -> list[dict]:
        events: list[dict] = []
        window = float(_setting("LEFT_WITH_WINDOW_SECONDS", 5.0))
        for object_id, pending in list(self._pending.items()):
            if person.entity_id not in pending.candidates:
                continue
            if now - pending.archived_at > window:
                continue
            obj = self._mem.get(object_id)
            del self._pending[object_id]
            if obj is None or obj.status != ARCHIVED:
                continue
            events.extend(self._left_with(obj, person.entity_id, now))
            self._persist(obj)
        return events

    def _left_with(self, obj: _Memory, person_id: int, now: float) -> list[dict]:
        """A person other than the associated one left view with the object."""
        obj.left_with_person_id = person_id
        metadata = {
            "object_label": obj.label,
            "person_label": _person(person_id),
            "associated_person_id": obj.associated_person_id,
            "is_associated_person": False,
            "last_seen": obj.last_seen,
            "last_zone_id": obj.last_zone_id,
            "last_zone_name": obj.last_zone_name,
            "snapshot_path": obj.snapshot_path,
        }
        return [
            make_event(
                OBJECT_LEFT_WITH,
                f"{obj.label} left view with {_person(person_id)}",
                severity=WARNING,
                entity_id=person_id,
                object_id=obj.entity_id,
                zone_id=obj.last_zone_id if obj.in_zone_at_last_seen else None,
                timestamp=now,
                metadata=metadata,
            )
        ]

    def _record_event(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        self._log.append(event)
        event_id = event.get("event_id")
        if event_id is None:
            return
        ids = set()
        for key in ("entity_id", "object_id"):
            value = event.get(key)
            if isinstance(value, int):
                ids.add(value)
        for entity_id in ids:
            bucket = self._event_ids.get(entity_id)
            if bucket is None:
                bucket = self._event_ids[entity_id] = deque(maxlen=EVENT_IDS_PER_ENTITY)
            bucket.append(event_id)

    def _summary(self, mem: _Memory) -> EntitySummary:
        return EntitySummary(
            entity_id=mem.entity_id,
            class_name=mem.class_name,
            status=mem.status,
            first_seen=mem.first_seen,
            last_seen=mem.last_seen,
            last_bbox=tuple(mem.last_bbox),
            last_center=tuple(mem.last_center),
            last_zone_id=mem.last_zone_id,
            last_zone_name=mem.last_zone_name,
            associated_person_id=mem.associated_person_id,
            relationship_state=mem.relationship_state,
            previous_person_id=mem.previous_person_id,
            left_with_person_id=mem.left_with_person_id,
            event_ids=list(self._event_ids.get(mem.entity_id, ())),
            snapshot_path=mem.snapshot_path,
        )

    def _persist(self, mem: _Memory) -> None:
        if self.store is None:
            return
        upsert = getattr(self.store, "upsert_entity", None)
        if not callable(upsert):
            return
        try:
            data = self._summary(mem).to_dict()
            data["session_id"] = self.session_id
            data["updated_at"] = time.time()
            upsert(data)
        except Exception as exc:
            self._store_errors += 1
            if self._store_errors == 1:
                print(f"[thirdeye.memory] entity persist failed: {exc}", flush=True)


def crop_snapshot(frame, bbox: BBox, max_side: int = 200):
    """Crop bbox (+ small margin) from a BGR frame, clipped, longest side <= max_side."""
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        mx = (x2 - x1) * SNAPSHOT_MARGIN
        my = (y2 - y1) * SNAPSHOT_MARGIN
        left = max(0, int(x1 - mx))
        top = max(0, int(y1 - my))
        right = min(width, int(round(x2 + mx)))
        bottom = min(height, int(round(y2 + my)))
        if right - left < 2 or bottom - top < 2:
            return None
        crop = frame[top:bottom, left:right]
        longest = max(crop.shape[0], crop.shape[1])
        if max_side > 0 and longest > max_side:
            import cv2

            scale = max_side / float(longest)
            size = (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale))))
            return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        return crop.copy()
    except Exception:
        return None
