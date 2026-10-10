"""World state: the current picture of every tracked anonymous entity.

Consumes TrackedEntity lists from the tracker each frame and keeps per-entity
history, velocity and stationary timing. Independent of UI and events.

Lifecycle of a record:
    ACTIVE       in the tracker output and detected this frame
    OUT_OF_VIEW  in the tracker output but coasting (frames_missed > 0)
    ARCHIVED     live track ended; the record moves to ``world.archived``
                 (bounded session memory). If the tracker revives the same ID
                 (spatial persistence for objects), the record is restored and
                 listed in ``world.revived`` instead of ``world.appeared``.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from thirdeye.core import settings
from thirdeye.core.types import (
    ACTIVE,
    ARCHIVED,
    OUT_OF_VIEW,
    PERSON,
    PORTABLE_OBJECT_CLASSES,
    BBox,
    TrackedEntity,
    entity_label,
)


@dataclass
class EntityRecord:
    track_id: int
    class_name: str
    bbox: BBox
    center: tuple[float, float]
    confidence: float
    first_seen: float
    last_seen: float  # last time the entity was in the tracker output
    last_detected: float  # last time it was actually detected (not coasted)
    frames_missed: int = 0
    history: deque = field(default_factory=deque)  # (t, cx, cy)
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s, smoothed
    stationary_anchor: tuple[float, float] = (0.0, 0.0)
    stationary_since: float = 0.0
    zones: set[str] = field(default_factory=set)  # maintained by ZoneManager
    archived_at: float | None = None  # set while the record is in world.archived

    @property
    def status(self) -> str:
        if self.archived_at is not None:
            return ARCHIVED
        return ACTIVE if self.frames_missed == 0 else OUT_OF_VIEW

    @property
    def label(self) -> str:
        return entity_label(self.class_name, self.track_id)

    @property
    def is_person(self) -> bool:
        return self.class_name == PERSON

    @property
    def is_object(self) -> bool:
        return self.class_name in PORTABLE_OBJECT_CLASSES

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    def age(self, now: float) -> float:
        return now - self.first_seen

    def stationary_seconds(self, now: float) -> float:
        return now - self.stationary_since

    def speed(self) -> float:
        return math.hypot(*self.velocity)


class WorldState:
    def __init__(self) -> None:
        self.entities: dict[int, EntityRecord] = {}
        self.now: float = 0.0
        self.frame_size: tuple[int, int] = (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        # IDs that appeared / disappeared in the most recent update()
        self.appeared: list[int] = []
        self.disappeared: list[EntityRecord] = []
        # IDs whose archived record was restored in the most recent update()
        self.revived: list[int] = []
        # Ended tracks kept as session memory, oldest first (bounded).
        self.archived: OrderedDict[int, EntityRecord] = OrderedDict()

    def update(self, tracks: list[TrackedEntity], now: float) -> None:
        self.now = now
        self.appeared = []
        self.disappeared = []
        self.revived = []
        seen: set[int] = set()

        for track in tracks:
            seen.add(track.track_id)
            record = self.entities.get(track.track_id)

            if record is None and self.archived:
                record = self._revive(track, now)

            if record is None:
                record = EntityRecord(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    bbox=track.bbox,
                    center=track.center,
                    confidence=track.confidence,
                    first_seen=now,
                    last_seen=now,
                    last_detected=now,
                    stationary_anchor=track.center,
                    stationary_since=now,
                )
                self.entities[track.track_id] = record
                self.appeared.append(track.track_id)
            else:
                self._update_motion(record, track, now)

            record.bbox = track.bbox
            record.center = track.center
            record.confidence = track.confidence
            record.frames_missed = track.frames_missed
            record.last_seen = now

            if track.frames_missed == 0:
                record.last_detected = now

            record.history.append((now, track.center[0], track.center[1]))

            while record.history and now - record.history[0][0] > settings.HISTORY_SECONDS:
                record.history.popleft()

        if len(seen) != len(self.entities):
            for track_id in [tid for tid in self.entities if tid not in seen]:
                record = self.entities.pop(track_id)
                self.disappeared.append(record)
                self._archive(record, now)

    def _archive(self, record: EntityRecord, now: float) -> None:
        record.archived_at = now
        self.archived.pop(record.track_id, None)
        self.archived[record.track_id] = record
        limit = max(0, int(getattr(settings, "ARCHIVE_MAX_ENTITIES", 1000)))
        while len(self.archived) > limit:
            self.archived.popitem(last=False)

    def _revive(self, track: TrackedEntity, now: float) -> EntityRecord | None:
        record = self.archived.get(track.track_id)
        if record is None or record.class_name != track.class_name:
            return None
        del self.archived[track.track_id]
        record.archived_at = None
        record.zones.clear()  # ZoneManager re-confirms membership
        # Restart motion state at the new position; keep first_seen/history.
        record.velocity = (0.0, 0.0)
        record.stationary_anchor = track.center
        record.stationary_since = now
        record.center = track.center
        record.last_seen = now
        self.entities[track.track_id] = record
        self.revived.append(track.track_id)
        return record

    def _update_motion(self, record: EntityRecord, track: TrackedEntity, now: float) -> None:
        dt = now - record.last_seen

        if dt > 0:
            vx = (track.center[0] - record.center[0]) / dt
            vy = (track.center[1] - record.center[1]) / dt
            alpha = 0.3
            record.velocity = (
                alpha * vx + (1 - alpha) * record.velocity[0],
                alpha * vy + (1 - alpha) * record.velocity[1],
            )

        radius = settings.STATIONARY_RADIUS_RATIO * max(record.diagonal, 1.0)
        ax, ay = record.stationary_anchor

        if math.hypot(track.center[0] - ax, track.center[1] - ay) > radius:
            record.stationary_anchor = track.center
            record.stationary_since = now

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, track_id: int) -> EntityRecord | None:
        return self.entities.get(track_id)

    def get_any(self, track_id: int) -> EntityRecord | None:
        """Live record, or the archived one if the track has ended."""
        return self.entities.get(track_id) or self.archived.get(track_id)

    def people(self) -> list[EntityRecord]:
        return [e for e in self.entities.values() if e.is_person]

    def objects(self) -> list[EntityRecord]:
        return [e for e in self.entities.values() if e.is_object]

    def in_zone(self, zone_id: str, people_only: bool = True) -> list[EntityRecord]:
        return [
            e
            for e in self.entities.values()
            if zone_id in e.zones and (e.is_person or not people_only)
        ]


def center_distance(a: EntityRecord, b: EntityRecord) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def point_to_bbox_distance(point: tuple[float, float], bbox: BBox) -> float:
    """Euclidean distance from a point to a box (0 if inside)."""
    x, y = point
    dx = max(bbox[0] - x, 0.0, x - bbox[2])
    dy = max(bbox[1] - y, 0.0, y - bbox[3])
    return math.hypot(dx, dy)


def proximity_ratio(obj: EntityRecord, person: EntityRecord) -> float:
    """Object-to-person distance in units of the person's box height.

    0.0 = object center inside the person's box; 1.0 = one body-height away.
    Scale-invariant, so it works near and far from the camera.
    """
    return point_to_bbox_distance(obj.center, person.bbox) / max(person.height, 1.0)
