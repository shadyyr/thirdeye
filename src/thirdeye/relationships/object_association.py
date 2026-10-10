"""Person <-> portable object association.

Tracks, for every visible backpack / handbag / suitcase, which person (if
any) it belongs with right now, using only anonymous track IDs and simple,
explainable geometry over time:

    UNASSOCIATED -> CANDIDATE_ASSOCIATION -> ASSOCIATED
    ASSOCIATED -> SEPARATING -> UNATTENDED      (see events/unattended_object.py)
    SEPARATING / UNATTENDED -> ASSOCIATED       (same person came back)

"Near" means proximity_ratio(object, person) < ASSOC_NEAR_RATIO and "far"
means > ASSOC_FAR_RATIO (the band between is hysteresis). All tunables are
read from thirdeye.core.settings at call time so tests can patch them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from thirdeye.core import settings
from thirdeye.core.types import (
    ASSOCIATION_CANDIDATE,
    ASSOCIATION_CREATED,
    INFO,
    NOTICE,
    Alert,
    entity_label,
    make_event,
)
from thirdeye.state.world_state import EntityRecord, WorldState, proximity_ratio

# Relationship states
UNASSOCIATED = "UNASSOCIATED"
CANDIDATE_ASSOCIATION = "CANDIDATE_ASSOCIATION"
ASSOCIATED = "ASSOCIATED"
SEPARATING = "SEPARATING"
UNATTENDED = "UNATTENDED"

# ---------------------------------------------------------------------------
# Module defaults (candidates for core/settings.py; a settings attribute with
# the same name overrides the value here).
# ---------------------------------------------------------------------------
# Emit the info-level ASSOCIATION_CANDIDATE event ("Backpack #5 near Person #2").
ASSOC_EMIT_CANDIDATE_EVENTS = True
# An object counts as stationary once it has stayed inside its stationary
# radius for at least this long.
OBJECT_STATIONARY_MIN_SECONDS = 0.5
# Tracker ID switch tolerance: if the associated person vanishes and a person
# who appeared at about that time is near the (unmoved) object within this
# many seconds, the association is silently handed to the new ID.
ASSOC_ID_HANDOFF_SECONDS = 2.5
ASSOC_ID_HANDOFF_FIRST_SEEN_SLACK = 1.0
# "Recently near" window used when an object leaves view.
OBJECT_RECENT_NEAR_SECONDS = 1.5
# Largest per-frame time step credited to cumulative near time.
ASSOC_MAX_FRAME_DT = 0.5


def setting(name: str, default):
    """Read a tunable from core.settings, falling back to a module default."""
    return getattr(settings, name, default)


@dataclass
class ObjectRelationship:
    object_id: int
    object_class: str
    state: str = UNASSOCIATED
    person_id: int | None = None  # current candidate/associated person
    last_associated_person_id: int | None = None
    state_since: float = 0.0
    separated_since: float | None = None
    # --- internal bookkeeping -------------------------------------------
    near_seconds: float = 0.0  # cumulative near time with the candidate
    last_near_time: float | None = None  # last time the candidate was near
    separation_pending_since: float | None = None
    anchor: tuple[float, float] | None = None  # object position at separation
    person_last_detected: float | None = None  # associated person's last detection
    recent_near: dict[int, float] = field(default_factory=dict)  # person -> last near time
    # Earlier track IDs of the associated person, re-bound by ID handoff
    # (the same human under a new tracker ID).
    handoff_ids: list[int] = field(default_factory=list)
    proximity: float | None = None  # current ratio to person_id (None if absent)
    last_center: tuple[float, float] = (0.0, 0.0)
    last_update: float | None = None

    @property
    def label(self) -> str:
        return entity_label(self.object_class, self.object_id)

    def set_state(self, state: str, now: float) -> None:
        self.state = state
        self.state_since = now

    def reset_candidacy(self) -> None:
        self.person_id = None
        self.near_seconds = 0.0
        self.last_near_time = None
        self.handoff_ids = []


def person_label(person_id: int | None) -> str:
    return entity_label("person", person_id) if person_id is not None else "unknown person"


class AssociationEngine:
    def __init__(self) -> None:
        # Imported here: unattended_object depends on this module's states.
        from thirdeye.events.unattended_object import UnattendedObjectMonitor

        self._relationships: dict[int, ObjectRelationship] = {}
        # Recently removed relationships that may be revived if the tracker
        # brings the same object ID back at the same place.
        self._ghosts: dict[int, tuple[ObjectRelationship, float]] = {}
        self.monitor = UnattendedObjectMonitor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, world: WorldState, now: float) -> list[dict]:
        events: list[dict] = []

        for record in world.disappeared:
            if not record.is_object:
                continue
            rel = self._relationships.pop(record.track_id, None)
            if rel is None:
                continue
            removed_events, keep = self.monitor.handle_removed(rel, record, now)
            events.extend(removed_events)
            if keep:
                self._ghosts[record.track_id] = (rel, now)

        revive_seconds = setting("TRACK_OBJECT_REVIVE_SECONDS", 15.0)
        for object_id, (_, removed_at) in list(self._ghosts.items()):
            if now - removed_at > revive_seconds:
                del self._ghosts[object_id]

        people = world.people()
        for obj in world.objects():
            rel = self._relationships.get(obj.track_id)
            if rel is None:
                rel = self._revive(obj, now) or ObjectRelationship(
                    object_id=obj.track_id, object_class=obj.class_name, state_since=now
                )
                self._relationships[obj.track_id] = rel
            events.extend(self._update_object(rel, obj, people, world, now))

        return events

    def relationships(self) -> list[ObjectRelationship]:
        return list(self._relationships.values())

    def get(self, object_id: int) -> ObjectRelationship | None:
        return self._relationships.get(object_id)

    def active_alerts(self, world: WorldState, now: float) -> list[Alert]:
        return self.monitor.alerts(self._relationships.values(), now)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _revive(self, obj: EntityRecord, now: float) -> ObjectRelationship | None:
        ghost = self._ghosts.pop(obj.track_id, None)
        if ghost is None:
            return None
        rel, _ = ghost
        max_dist = setting("TRACK_OBJECT_REVIVE_DIST_RATIO", 1.0) * max(obj.diagonal, 1.0)
        if math.dist(obj.center, rel.last_center) > max_dist:
            return None
        rel.last_update = now
        rel.separation_pending_since = None
        return rel

    def _update_object(
        self,
        rel: ObjectRelationship,
        obj: EntityRecord,
        people: list[EntityRecord],
        world: WorldState,
        now: float,
    ) -> list[dict]:
        dt = 0.0 if rel.last_update is None else min(max(now - rel.last_update, 0.0), ASSOC_MAX_FRAME_DT)
        rel.last_update = now
        rel.last_center = obj.center

        near_ratio = setting("ASSOC_NEAR_RATIO", 0.5)
        ratios = {p.track_id: proximity_ratio(obj, p) for p in people}

        for pid, ratio in ratios.items():
            if ratio < near_ratio:
                rel.recent_near[pid] = now
        keep_for = max(setting("OBJECT_RECENT_NEAR_SECONDS", OBJECT_RECENT_NEAR_SECONDS) * 4, 10.0)
        for pid, t in list(rel.recent_near.items()):
            if now - t > keep_for:
                del rel.recent_near[pid]

        near = [(ratio, pid) for pid, ratio in ratios.items() if ratio < near_ratio]
        nearest = min(near)[1] if near else None

        if rel.state in (UNASSOCIATED, CANDIDATE_ASSOCIATION):
            events = self._update_candidate(rel, obj, ratios, nearest, dt, world, now)
        elif rel.state == ASSOCIATED:
            events = self._update_associated(rel, obj, ratios, world, now)
        else:
            events = self.monitor.update_separated(rel, obj, ratios, nearest, now)

        rel.proximity = ratios.get(rel.person_id) if rel.person_id is not None else None
        return events

    def _update_candidate(
        self,
        rel: ObjectRelationship,
        obj: EntityRecord,
        ratios: dict[int, float],
        nearest: int | None,
        dt: float,
        world: WorldState,
        now: float,
    ) -> list[dict]:
        events: list[dict] = []
        near_ratio = setting("ASSOC_NEAR_RATIO", 0.5)
        candidate_near = rel.person_id is not None and ratios.get(rel.person_id, math.inf) < near_ratio

        if candidate_near:
            rel.near_seconds += dt
            rel.last_near_time = now
        elif nearest is not None:
            # A different person is now the nearest near person: restart.
            rel.person_id = nearest
            rel.near_seconds = 0.0
            rel.last_near_time = now
            if rel.state != UNASSOCIATED:
                rel.set_state(UNASSOCIATED, now)
        elif rel.person_id is not None:
            gap = now - (rel.last_near_time if rel.last_near_time is not None else now)
            if gap > setting("ASSOC_GAP_TOLERANCE_SECONDS", 1.0):
                rel.reset_candidacy()
                if rel.state != UNASSOCIATED:
                    rel.set_state(UNASSOCIATED, now)

        if rel.person_id is None:
            return events

        if rel.state == UNASSOCIATED and rel.near_seconds >= setting("ASSOC_CANDIDATE_SECONDS", 0.5):
            rel.set_state(CANDIDATE_ASSOCIATION, now)
            if setting("ASSOC_EMIT_CANDIDATE_EVENTS", ASSOC_EMIT_CANDIDATE_EVENTS):
                events.append(
                    make_event(
                        ASSOCIATION_CANDIDATE,
                        f"{rel.label} near {person_label(rel.person_id)}",
                        severity=INFO,
                        entity_id=rel.person_id,
                        object_id=rel.object_id,
                        timestamp=now,
                        metadata={"object_class": rel.object_class},
                    )
                )

        if rel.state == CANDIDATE_ASSOCIATION and rel.near_seconds >= setting("ASSOC_CONFIRM_SECONDS", 2.0):
            rel.set_state(ASSOCIATED, now)
            rel.last_associated_person_id = rel.person_id
            person = world.get(rel.person_id)
            rel.person_last_detected = person.last_detected if person is not None else now
            rel.separation_pending_since = None
            events.append(
                make_event(
                    ASSOCIATION_CREATED,
                    f"{rel.label} associated with {person_label(rel.person_id)}",
                    severity=NOTICE,
                    entity_id=rel.person_id,
                    object_id=rel.object_id,
                    timestamp=now,
                    metadata={
                        "object_class": rel.object_class,
                        "near_seconds": round(rel.near_seconds, 2),
                    },
                )
            )
        return events

    def _update_associated(
        self,
        rel: ObjectRelationship,
        obj: EntityRecord,
        ratios: dict[int, float],
        world: WorldState,
        now: float,
    ) -> list[dict]:
        person = world.get(rel.person_id) if rel.person_id is not None else None
        if person is not None and person.is_person:
            rel.person_last_detected = person.last_detected
        else:
            person = self._try_id_handoff(rel, obj, ratios, world, now)

        # While the object is not being detected (occluded, e.g. picked up),
        # hold the separation timer instead of judging a coasted box.
        if obj.frames_missed > 0:
            return []

        ratio = ratios.get(rel.person_id, math.inf) if person is not None else math.inf
        far = ratio > setting("ASSOC_FAR_RATIO", 1.0)
        stationary = obj.stationary_seconds(now) >= setting(
            "OBJECT_STATIONARY_MIN_SECONDS", OBJECT_STATIONARY_MIN_SECONDS
        )

        if not (far and stationary):
            rel.separation_pending_since = None
            return []

        if rel.separation_pending_since is None:
            rel.separation_pending_since = now
        if now - rel.separation_pending_since >= setting("SEPARATION_CONFIRM_SECONDS", 1.0):
            return self.monitor.start_separating(rel, obj, now)
        return []

    def _try_id_handoff(
        self,
        rel: ObjectRelationship,
        obj: EntityRecord,
        ratios: dict[int, float],
        world: WorldState,
        now: float,
    ) -> EntityRecord | None:
        """Re-bind to a fresh track ID of (very likely) the same person.

        Only when the associated person vanished moments ago, the object has
        not moved, and a person who appeared at about that time is near it.
        """
        last = rel.person_last_detected
        if last is None or now - last > setting("ASSOC_ID_HANDOFF_SECONDS", ASSOC_ID_HANDOFF_SECONDS):
            return None
        if obj.stationary_seconds(now) < now - last:
            return None  # object moved since the person was last seen
        slack = setting("ASSOC_ID_HANDOFF_FIRST_SEEN_SLACK", ASSOC_ID_HANDOFF_FIRST_SEEN_SLACK)
        near_ratio = setting("ASSOC_NEAR_RATIO", 0.5)
        for ratio, pid in sorted((r, p) for p, r in ratios.items() if r < near_ratio):
            candidate = world.get(pid)
            if candidate is not None and candidate.first_seen >= last - slack:
                if rel.person_id is not None and rel.person_id not in rel.handoff_ids:
                    rel.handoff_ids.append(rel.person_id)
                rel.person_id = pid
                rel.last_associated_person_id = pid
                rel.person_last_detected = candidate.last_detected
                rel.separation_pending_since = None
                return candidate
        return None
