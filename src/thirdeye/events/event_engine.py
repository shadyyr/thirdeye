"""Core temporal person events: view/zone entry and exit, dwell, occupancy
and crowding.

Consumes the WorldState and the zone transitions produced by ZoneManager
each frame and returns structured events built with make_event(). Also
exposes the ongoing dashboard alerts (restricted-zone presence, crowding).
All messages are plain ASCII so they render on the OpenCV dashboard.
"""

from __future__ import annotations

from typing import Any

from thirdeye.core import settings
from thirdeye.core.types import (
    ALERT,
    CROWDING_CLEARED,
    CROWDING_STARTED,
    DWELLING,
    ENTITY_ENTERED,
    ENTITY_EXITED,
    ENTITY_REAPPEARED,
    INFO,
    NOTICE,
    OCCUPANCY_CHANGED,
    WARNING,
    ZONE_ENTERED,
    ZONE_EXITED,
    Alert,
    entity_label,
    make_event,
)
from thirdeye.spatial.zone_manager import ENTER, EXIT, ZoneManager
from thirdeye.state.world_state import EntityRecord, WorldState

WHOLE_VIEW_NAME = "the monitored view"


class _CrowdState:
    __slots__ = ("crowded", "pending_since", "since", "peak")

    def __init__(self) -> None:
        self.crowded = False
        self.pending_since: float | None = None
        self.since = 0.0
        self.peak = 0


class EventEngine:
    def __init__(self, zone_manager: ZoneManager) -> None:
        self.zone_manager = zone_manager
        # (entity_id, zone_id) -> confirmed entry time
        self._zone_entry: dict[tuple[int, str], float] = {}
        # (entity_id, zone_id) -> time of last DWELLING event
        self._dwell_last: dict[tuple[int, str], float] = {}
        # Occupancy debounce
        self.occupancy_total: int = 0
        self._occ_candidate: int = 0
        self._occ_candidate_since: float | None = None
        # Crowding per key (zone id, or None for the whole view)
        self._crowd: dict[str | None, _CrowdState] = {}
        self.crowded_zones: set = set()

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        world: WorldState,
        transitions: list[tuple[str, int, str]],
        now: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        gone = {r.track_id: r for r in world.disappeared}

        for track_id in world.appeared:
            record = world.get(track_id)
            if record is not None:
                events.append(self._entity_entered(record, now))

        # IDs restored from session memory (e.g. a bag the tracker revived).
        for track_id in getattr(world, "revived", []):
            record = world.get(track_id)
            if record is not None:
                events.append(
                    make_event(
                        ENTITY_REAPPEARED,
                        f"{record.label} back in view",
                        severity=INFO,
                        entity_id=track_id,
                        timestamp=now,
                        metadata={"class_name": record.class_name},
                    )
                )

        for kind, entity_id, zone_id in transitions:
            record = world.get(entity_id) or gone.get(entity_id)
            class_name = record.class_name if record is not None else "entity"
            if kind == ENTER:
                event = self._zone_entered(entity_id, class_name, zone_id, now)
            elif kind == EXIT:
                event = self._zone_exited(entity_id, class_name, zone_id, now)
            else:
                event = None
            if event is not None:
                events.append(event)

        for record in world.disappeared:
            events.append(self._entity_exited(record, now))

        self._prune(world)
        events.extend(self._dwell_events(world, now))
        events.extend(self._occupancy_events(world, now))
        events.extend(self._crowding_events(world, now))
        return events

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def zone_occupancy(self, world: WorldState) -> dict[str, int]:
        return {
            zone.id: sum(1 for p in world.people() if zone.id in p.zones)
            for zone in self.zone_manager.active_zones()
        }

    def zone_entry_time(self, entity_id: int, zone_id: str) -> float | None:
        return self._zone_entry.get((entity_id, zone_id))

    def active_alerts(self, world: WorldState, now: float) -> list[Alert]:
        alerts: list[Alert] = []
        for person in sorted(world.people(), key=lambda r: r.track_id):
            for zone_id in sorted(person.zones):
                zone = self.zone_manager.get(zone_id)
                if zone is None or not zone.active or not zone.is_restricted:
                    continue
                since = self._zone_entry.get((person.track_id, zone_id), now)
                alerts.append(
                    Alert(
                        alert_type=ZONE_ENTERED,
                        title="RESTRICTED ZONE ENTRY",
                        lines=[person.label, f"In {zone.name} for {now - since:.1f} sec"],
                        severity=ALERT,
                        since=since,
                        entity_ids=[person.track_id],
                        zone_id=zone_id,
                    )
                )

        occupancy = self.zone_occupancy(world)
        keys = sorted(self.crowded_zones, key=lambda k: (k is None, k or ""))
        for key in keys:
            state = self._crowd.get(key)
            count = len(world.people()) if key is None else occupancy.get(key, 0)
            alerts.append(
                Alert(
                    alert_type=CROWDING_STARTED,
                    title="CROWDING",
                    lines=[self._zone_name(key), f"Occupancy: {count}"],
                    severity=ALERT,
                    since=state.since if state is not None else now,
                    entity_ids=[],
                    zone_id=key,
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Entity / zone events
    # ------------------------------------------------------------------
    def _entity_entered(self, record: EntityRecord, now: float) -> dict[str, Any]:
        verb = "entered view" if record.is_person else "detected"
        return make_event(
            ENTITY_ENTERED,
            f"{record.label} {verb}",
            severity=INFO,
            entity_id=record.track_id,
            timestamp=now,
            metadata={"class_name": record.class_name, "confidence": round(record.confidence, 3)},
        )

    def _entity_exited(self, record: EntityRecord, now: float) -> dict[str, Any]:
        duration = max(0.0, record.last_seen - record.first_seen)
        return make_event(
            ENTITY_EXITED,
            f"{record.label} left view",
            severity=INFO,
            entity_id=record.track_id,
            timestamp=now,
            metadata={
                "class_name": record.class_name,
                "duration_visible": round(duration, 2),
                "first_seen": record.first_seen,
                "last_seen": record.last_seen,
            },
        )

    def _zone_entered(
        self, entity_id: int, class_name: str, zone_id: str, now: float
    ) -> dict[str, Any] | None:
        zone = self.zone_manager.get(zone_id)
        key = (entity_id, zone_id)
        if key in self._zone_entry:
            return None  # already inside; never duplicate
        self._zone_entry[key] = now
        self._dwell_last.pop(key, None)
        is_person = class_name == "person"
        restricted = zone is not None and zone.is_restricted
        if restricted:
            severity = ALERT if is_person else WARNING
        else:
            severity = INFO
        return make_event(
            ZONE_ENTERED,
            f"{entity_label(class_name, entity_id)} entered {self._zone_name(zone_id)}",
            severity=severity,
            entity_id=entity_id,
            zone_id=zone_id,
            timestamp=now,
            metadata={
                "class_name": class_name,
                "zone_type": zone.type if zone is not None else None,
            },
        )

    def _zone_exited(
        self, entity_id: int, class_name: str, zone_id: str, now: float
    ) -> dict[str, Any] | None:
        key = (entity_id, zone_id)
        entered = self._zone_entry.pop(key, None)
        self._dwell_last.pop(key, None)
        if entered is None:
            return None  # never confirmed inside; nothing to close
        zone = self.zone_manager.get(zone_id)
        dwell = max(0.0, now - entered)
        return make_event(
            ZONE_EXITED,
            f"{entity_label(class_name, entity_id)} left {self._zone_name(zone_id)} "
            f"after {dwell:.1f} sec",
            severity=INFO,
            entity_id=entity_id,
            zone_id=zone_id,
            timestamp=now,
            metadata={
                "class_name": class_name,
                "zone_type": zone.type if zone is not None else None,
                "dwell_seconds": round(dwell, 2),
                "entered_at": entered,
            },
        )

    # ------------------------------------------------------------------
    # Dwell
    # ------------------------------------------------------------------
    def _dwell_events(self, world: WorldState, now: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        threshold = float(settings.DWELL_THRESHOLD_SECONDS)
        repeat = float(settings.DWELL_REPEAT_SECONDS)
        for (entity_id, zone_id), entered in sorted(self._zone_entry.items()):
            record = world.get(entity_id)
            if record is None or not record.is_person:
                continue
            zone = self.zone_manager.get(zone_id)
            if zone is None or not zone.active:
                continue
            duration = now - entered
            if duration < threshold:
                continue
            key = (entity_id, zone_id)
            last = self._dwell_last.get(key)
            if last is not None and now - last < repeat:
                continue
            self._dwell_last[key] = now
            events.append(
                make_event(
                    DWELLING,
                    f"{record.label} dwelling in {zone.name} for {duration:.1f} sec",
                    severity=WARNING if zone.is_restricted else NOTICE,
                    entity_id=entity_id,
                    zone_id=zone_id,
                    timestamp=now,
                    metadata={
                        "dwell_seconds": round(duration, 2),
                        "entered_at": entered,
                        "zone_type": zone.type,
                        "repeat": last is not None,
                    },
                )
            )
        return events

    # ------------------------------------------------------------------
    # Occupancy
    # ------------------------------------------------------------------
    def _occupancy_events(self, world: WorldState, now: float) -> list[dict[str, Any]]:
        raw = len(world.people())
        if raw != self._occ_candidate or self._occ_candidate_since is None:
            self._occ_candidate = raw
            self._occ_candidate_since = now

        if self._occ_candidate == self.occupancy_total:
            return []
        if now - self._occ_candidate_since < float(settings.OCCUPANCY_CONFIRM_SECONDS):
            return []

        old = self.occupancy_total
        new = self._occ_candidate
        self.occupancy_total = new
        direction = "increased" if new > old else "decreased"
        return [
            make_event(
                OCCUPANCY_CHANGED,
                f"Occupancy {direction} from {old} to {new}",
                severity=INFO,
                timestamp=now,
                metadata={"previous": old, "current": new, "delta": new - old},
            )
        ]

    # ------------------------------------------------------------------
    # Crowding
    # ------------------------------------------------------------------
    def _crowding_events(self, world: WorldState, now: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        threshold = int(settings.CROWD_THRESHOLD)
        clear_below = int(settings.CROWD_CLEAR_BELOW)
        confirm = float(settings.CROWD_CONFIRM_SECONDS)

        counts: dict[str | None, int] = dict(self.zone_occupancy(world))
        counts[None] = len(world.people())

        # Zones that were deactivated: drop their state silently.
        for key in [k for k in self._crowd if k not in counts]:
            del self._crowd[key]
            self.crowded_zones.discard(key)

        for key, count in counts.items():
            state = self._crowd.setdefault(key, _CrowdState())
            if state.crowded:
                state.peak = max(state.peak, count)
                condition = count < clear_below
            else:
                condition = count >= threshold

            if not condition:
                state.pending_since = None
                continue
            if state.pending_since is None:
                state.pending_since = now
            if now - state.pending_since < confirm:
                continue

            state.pending_since = None
            name = self._zone_name(key)
            if not state.crowded:
                state.crowded = True
                state.since = now
                state.peak = count
                self.crowded_zones.add(key)
                events.append(
                    make_event(
                        CROWDING_STARTED,
                        f"Crowd threshold exceeded in {name} - occupancy {count}",
                        severity=ALERT,
                        zone_id=key,
                        timestamp=now,
                        metadata={"occupancy": count, "threshold": threshold},
                    )
                )
            else:
                duration = now - state.since
                state.crowded = False
                self.crowded_zones.discard(key)
                events.append(
                    make_event(
                        CROWDING_CLEARED,
                        f"Crowding cleared in {name} - occupancy {count}",
                        severity=INFO,
                        zone_id=key,
                        timestamp=now,
                        metadata={
                            "occupancy": count,
                            "peak_occupancy": state.peak,
                            "duration_seconds": round(duration, 2),
                        },
                    )
                )
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _zone_name(self, zone_id: str | None) -> str:
        if zone_id is None:
            return WHOLE_VIEW_NAME
        zone = self.zone_manager.get(zone_id)
        return zone.name if zone is not None else str(zone_id)

    def _prune(self, world: WorldState) -> None:
        """Drop per-zone timing for entities no longer in the world."""
        for key in [k for k in self._zone_entry if k[0] not in world.entities]:
            del self._zone_entry[key]
        for key in [k for k in self._dwell_last if k[0] not in world.entities]:
            del self._dwell_last[key]
