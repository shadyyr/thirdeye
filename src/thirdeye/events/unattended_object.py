"""Unattended / moved / lost object logic and its messages and alerts.

Used by relationships.object_association.AssociationEngine once an object is
ASSOCIATED with a person:

    ASSOCIATED -> SEPARATING      associated person far (or gone) while the
                                  object stays put, for SEPARATION_CONFIRM_SECONDS
    SEPARATING -> UNATTENDED      still separated UNATTENDED_TIMEOUT_SECONDS
                                  after the separation started
    SEPARATING/UNATTENDED -> ASSOCIATED
                                  the same person comes back near it
    SEPARATING/UNATTENDED -> UNASSOCIATED (OBJECT_MOVED)
                                  the object moved away from where it was left
                                  while a different person is near it

Wording is deliberately neutral: an association is a spatial relationship
over time, not a claim about who the object belongs to.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from thirdeye.core.types import (
    ALERT,
    ASSOCIATION_RESTORED,
    INFO,
    OBJECT_LOST,
    OBJECT_MOVED,
    OBJECT_SEPARATING,
    SEVERITY_RANK,
    UNATTENDED_OBJECT,
    WARNING,
    Alert,
    make_event,
)
from thirdeye.relationships.object_association import (
    ASSOCIATED,
    OBJECT_RECENT_NEAR_SECONDS,
    SEPARATING,
    UNASSOCIATED,
    UNATTENDED,
    ObjectRelationship,
    person_label,
    setting,
)
from thirdeye.state.world_state import EntityRecord


def format_seconds(seconds: float) -> str:
    return f"{max(seconds, 0.0):.1f} sec"


@dataclass
class MovedAlert:
    """A recent OBJECT_MOVED condition kept on the dashboard for a while."""

    object_id: int
    object_label: str
    previous_person_id: int | None
    current_person_id: int | None
    since: float
    removed_from_view: bool = False

    def to_alert(self) -> Alert:
        lines = [self.object_label, f"Previously associated with {person_label(self.previous_person_id)}"]
        if self.removed_from_view:
            lines.append(f"Last seen near {person_label(self.current_person_id)}")
        else:
            lines.append(f"Currently near {person_label(self.current_person_id)}")
        ids = [self.object_id] + [p for p in (self.previous_person_id, self.current_person_id) if p is not None]
        return Alert(
            alert_type=OBJECT_MOVED,
            title="OBJECT MOVED",
            lines=lines,
            severity=ALERT,
            since=self.since,
            entity_ids=ids,
        )


class UnattendedObjectMonitor:
    def __init__(self) -> None:
        self._moved: dict[int, MovedAlert] = {}

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def start_separating(self, rel: ObjectRelationship, obj: EntityRecord, now: float) -> list[dict]:
        """ASSOCIATED -> SEPARATING (called once the separation is confirmed)."""
        started = rel.separation_pending_since if rel.separation_pending_since is not None else now
        rel.set_state(SEPARATING, now)
        rel.separated_since = started
        rel.anchor = obj.center
        rel.separation_pending_since = None
        pid = rel.last_associated_person_id
        return [
            make_event(
                OBJECT_SEPARATING,
                f"{rel.label} separating from {person_label(pid)}",
                severity=WARNING,
                entity_id=pid,
                object_id=rel.object_id,
                timestamp=now,
                metadata={
                    "last_associated_person_id": pid,
                    "object_class": rel.object_class,
                    "separated_seconds": round(now - started, 2),
                },
            )
        ]

    def update_separated(
        self,
        rel: ObjectRelationship,
        obj: EntityRecord,
        ratios: dict[int, float],
        nearest: int | None,
        now: float,
    ) -> list[dict]:
        """Per-frame update for SEPARATING / UNATTENDED objects."""
        pid = rel.last_associated_person_id
        near_ratio = setting("ASSOC_NEAR_RATIO", 0.5)
        separated_for = now - (rel.separated_since if rel.separated_since is not None else now)

        # Same person came back.
        if pid is not None and ratios.get(pid, math.inf) < near_ratio:
            rel.set_state(ASSOCIATED, now)
            rel.person_id = pid
            rel.separated_since = None
            rel.anchor = None
            rel.separation_pending_since = None
            return [
                make_event(
                    ASSOCIATION_RESTORED,
                    f"{person_label(pid)} returned to {rel.label}",
                    severity=INFO,
                    entity_id=pid,
                    object_id=rel.object_id,
                    timestamp=now,
                    metadata={
                        "object_class": rel.object_class,
                        "separated_seconds": round(separated_for, 2),
                    },
                )
            ]

        # Moved from where it was left while a different person is near it.
        if rel.anchor is not None and nearest is not None and nearest != pid:
            moved = math.dist(obj.center, rel.anchor)
            if moved > setting("OBJECT_MOVED_DIST_RATIO", 1.0) * max(obj.diagonal, 1.0):
                return self._object_moved(rel, nearest, now, separated_for, removed=False)

        # Still separated long enough -> unattended (only judged on frames
        # where the object is actually detected).
        if (
            rel.state == SEPARATING
            and obj.frames_missed == 0
            and separated_for >= setting("UNATTENDED_TIMEOUT_SECONDS", 5.0)
        ):
            rel.set_state(UNATTENDED, now)
            return [
                make_event(
                    UNATTENDED_OBJECT,
                    f"UNATTENDED OBJECT: {rel.label}, last associated with {person_label(pid)}",
                    severity=ALERT,
                    entity_id=pid,
                    object_id=rel.object_id,
                    timestamp=now,
                    metadata={
                        "last_associated_person_id": pid,
                        "unattended_seconds": round(separated_for, 2),
                        "object_class": rel.object_class,
                    },
                )
            ]
        return []

    def handle_removed(
        self, rel: ObjectRelationship, record: EntityRecord, now: float
    ) -> tuple[list[dict], bool]:
        """Object left the world. Returns (events, keep_for_revival)."""
        pid = rel.last_associated_person_id
        window_start = record.last_detected - setting(
            "OBJECT_RECENT_NEAR_SECONDS", OBJECT_RECENT_NEAR_SECONDS
        )
        recent = {p: t for p, t in rel.recent_near.items() if t >= window_start}
        assoc_was_near = pid is not None and pid in recent

        if rel.state in (SEPARATING, UNATTENDED):
            others = [(t, p) for p, t in recent.items() if p != pid]
            if others:
                other = max(others)[1]
                separated_for = now - (rel.separated_since if rel.separated_since is not None else now)
                return self._object_moved(rel, other, now, separated_for, removed=True), False
            if assoc_was_near:
                return [self._lost_event(rel, now, with_person=pid)], False
            return [self._lost_event(rel, now)], True

        if rel.state == ASSOCIATED:
            return [self._lost_event(rel, now, with_person=pid if assoc_was_near else None)], True

        return [], False

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def alerts(self, relationships: Iterable[ObjectRelationship], now: float) -> list[Alert]:
        alerts: list[Alert] = []
        for rel in relationships:
            if rel.state not in (SEPARATING, UNATTENDED):
                continue
            pid = rel.last_associated_person_id
            since = rel.separated_since if rel.separated_since is not None else rel.state_since
            ids = [rel.object_id] + ([pid] if pid is not None else [])
            if rel.state == UNATTENDED:
                alerts.append(
                    Alert(
                        alert_type=UNATTENDED_OBJECT,
                        title="UNATTENDED OBJECT",
                        lines=[
                            rel.label,
                            f"Last associated with {person_label(pid)}",
                            f"Unattended: {format_seconds(now - since)}",
                        ],
                        severity=ALERT,
                        since=since,
                        entity_ids=ids,
                    )
                )
            else:
                alerts.append(
                    Alert(
                        alert_type=OBJECT_SEPARATING,
                        title="OBJECT SEPARATING",
                        lines=[
                            rel.label,
                            f"Last associated with {person_label(pid)}",
                            f"Separated: {format_seconds(now - since)}",
                        ],
                        severity=WARNING,
                        since=since,
                        entity_ids=ids,
                    )
                )

        keep = setting("OBJECT_MOVED_ALERT_SECONDS", 10.0)
        for object_id, moved in list(self._moved.items()):
            if now - moved.since > keep:
                del self._moved[object_id]
            else:
                alerts.append(moved.to_alert())

        alerts.sort(key=lambda a: (-SEVERITY_RANK.get(a.severity, 0), a.since))
        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _object_moved(
        self,
        rel: ObjectRelationship,
        other_person: int,
        now: float,
        separated_for: float,
        *,
        removed: bool,
    ) -> list[dict]:
        pid = rel.last_associated_person_id
        if removed:
            message = f"OBJECT MOVED: {rel.label} no longer visible, last seen near {person_label(other_person)}"
        else:
            message = (
                f"OBJECT MOVED: {rel.label}, previously associated with {person_label(pid)}, "
                f"currently near {person_label(other_person)}"
            )
        self._moved[rel.object_id] = MovedAlert(
            object_id=rel.object_id,
            object_label=rel.label,
            previous_person_id=pid,
            current_person_id=other_person,
            since=now,
            removed_from_view=removed,
        )
        previous_state = rel.state
        rel.set_state(UNASSOCIATED, now)
        rel.reset_candidacy()
        rel.separated_since = None
        rel.anchor = None
        rel.separation_pending_since = None
        return [
            make_event(
                OBJECT_MOVED,
                message,
                severity=ALERT,
                entity_id=other_person,
                object_id=rel.object_id,
                timestamp=now,
                metadata={
                    "last_associated_person_id": pid,
                    "near_person_id": other_person,
                    "object_class": rel.object_class,
                    "previous_state": previous_state,
                    "separated_seconds": round(separated_for, 2),
                    "removed_from_view": removed,
                },
            )
        ]

    @staticmethod
    def _lost_event(rel: ObjectRelationship, now: float, with_person: int | None = None) -> dict:
        if with_person is not None:
            message = f"{rel.label} left view with {person_label(with_person)}"
        else:
            message = f"{rel.label} no longer visible"
        return make_event(
            OBJECT_LOST,
            message,
            severity=INFO,
            entity_id=rel.object_id,
            object_id=rel.object_id,
            timestamp=now,
            metadata={
                "last_associated_person_id": rel.last_associated_person_id,
                "object_class": rel.object_class,
                "previous_state": rel.state,
            },
        )
