"""Shared data contracts for the Third Eye event platform.

Every layer (tracking, world state, zones, events, relationships, storage,
dashboard) exchanges these structures. Keep them small and stable.

Coordinates are always in pixels of the original camera frame
(x right, y down). IDs are temporary anonymous tracking IDs.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PERSON = "person"
PORTABLE_OBJECT_CLASSES = ("backpack", "handbag", "suitcase")
TRACKED_CLASSES = (PERSON, *PORTABLE_OBJECT_CLASSES)

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Detection:
    """One raw detector output for a single frame."""

    class_name: str
    confidence: float
    bbox: BBox


@dataclass
class TrackedEntity:
    """Tracker output: one persistent anonymous entity for the current frame.

    frames_missed == 0 means the entity was detected in this frame; > 0 means
    the tracker is coasting it on its last known box (brief miss tolerance).
    """

    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    center: tuple[float, float]
    timestamp: float
    frames_missed: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["center"] = list(self.center)
        return data


def entity_label(class_name: str, track_id: int) -> str:
    """Human-readable anonymous label, e.g. 'Person #3', 'Backpack #5'."""
    return f"{class_name.capitalize()} #{track_id}"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

# Event types
ENTITY_ENTERED = "ENTITY_ENTERED"
ENTITY_EXITED = "ENTITY_EXITED"
ZONE_ENTERED = "ZONE_ENTERED"
ZONE_EXITED = "ZONE_EXITED"
DWELLING = "DWELLING"
OCCUPANCY_CHANGED = "OCCUPANCY_CHANGED"
CROWDING_STARTED = "CROWDING_STARTED"
CROWDING_CLEARED = "CROWDING_CLEARED"
ASSOCIATION_CANDIDATE = "ASSOCIATION_CANDIDATE"
ASSOCIATION_CREATED = "ASSOCIATION_CREATED"
ASSOCIATION_RESTORED = "ASSOCIATION_RESTORED"
OBJECT_SEPARATING = "OBJECT_SEPARATING"
UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
OBJECT_MOVED = "OBJECT_MOVED"
OBJECT_LOST = "OBJECT_LOST"
OBJECT_LEFT_WITH = "OBJECT_LEFT_WITH"
ENTITY_REAPPEARED = "ENTITY_REAPPEARED"

# Entity lifecycle
ACTIVE = "ACTIVE"  # detected in the current frame
OUT_OF_VIEW = "OUT_OF_VIEW"  # still tracked, but not detected right now (coasting)
ARCHIVED = "ARCHIVED"  # live track ended; kept as session memory

# Severities, lowest to highest
INFO = "info"
NOTICE = "notice"
WARNING = "warning"
ALERT = "alert"
SEVERITY_RANK = {INFO: 0, NOTICE: 1, WARNING: 2, ALERT: 3}

_event_counter = itertools.count(1)


def make_event(
    event_type: str,
    message: str,
    *,
    severity: str = INFO,
    entity_id: int | None = None,
    object_id: int | None = None,
    zone_id: str | None = None,
    timestamp: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an event in the one common format used everywhere.

    entity_id is the primary entity (usually a person, or the object itself
    for object-only events). object_id is the portable object involved in
    person<->object events, if any.
    """
    return {
        "event_id": f"evt-{int(time.time())}-{next(_event_counter):06d}",
        "timestamp": time.time() if timestamp is None else timestamp,
        "event_type": event_type,
        "entity_id": entity_id,
        "object_id": object_id,
        "zone_id": zone_id,
        "severity": severity,
        "message": message,
        "metadata": dict(metadata or {}),
    }


@dataclass
class Alert:
    """An ongoing condition shown on the dashboard until it clears.

    Events are instantaneous; alerts persist (e.g. 'Backpack #5 unattended
    for 8.4 sec'). Engines rebuild their alert list each frame.
    """

    alert_type: str  # usually the event type that raised it
    title: str  # e.g. "UNATTENDED OBJECT"
    lines: list[str]  # detail lines, e.g. ["Backpack #5", "Last associated with Person #2"]
    severity: str = ALERT
    since: float = 0.0
    entity_ids: list[int] = field(default_factory=list)
    zone_id: str | None = None


@dataclass
class EntitySummary:
    """Session memory of one entity (live or archived), for UI and queries.

    Wording is deliberately conservative: "associated with", never "owned by".
    """

    entity_id: int
    class_name: str
    status: str  # ACTIVE | OUT_OF_VIEW | ARCHIVED
    first_seen: float
    last_seen: float
    last_bbox: BBox
    last_center: tuple[float, float]
    last_zone_id: str | None = None
    last_zone_name: str | None = None
    associated_person_id: int | None = None  # current / last associated person
    relationship_state: str | None = None
    previous_person_id: int | None = None  # previously associated person
    left_with_person_id: int | None = None
    event_ids: list[str] = field(default_factory=list)
    snapshot_path: str | None = None

    @property
    def label(self) -> str:
        return entity_label(self.class_name, self.entity_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_bbox"] = list(self.last_bbox)
        data["last_center"] = list(self.last_center)
        return data
