"""Simple, inspectable scoring for events that deserve haptic attention."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from thirdeye.models import DirectionZone, MotionState, ScoredEvent, Track

LOGGER = logging.getLogger(__name__)


class RelevanceScorer:
    def __init__(self, config: dict) -> None:
        self.class_weights = config["class_weights"]
        self.weights = config["weights"]
        self.close_area_fraction = config["close_area_fraction"]
        self.persistence_frames = config["persistence_frames"]

    def score(self, track: Track, zone: DirectionZone) -> ScoredEvent:
        latest = track.latest_detection
        center_x = latest.normalized_center_x
        raw_components = {
            "class": self.class_weights.get(track.class_name, 0.0),
            "approach": 1.0 if track.motion_state == MotionState.APPROACHING else 0.0,
            "closeness": min(1.0, latest.normalized_area / self.close_area_fraction),
            "centrality": max(0.0, 1.0 - abs(0.5 - center_x) * 2),
            "motion": self._motion_magnitude(track),
            "persistence": min(1.0, len(track.observations) / self.persistence_frames),
        }
        components = {
            name: raw_components[name] * self.weights.get(name, 0.0) for name in raw_components
        }
        score = min(1.0, max(0.0, sum(components.values())))
        LOGGER.debug("relevance_scored track_id=%s score=%.3f", track.track_id, score)
        return ScoredEvent(
            track=track,
            zone=zone,
            relevance_score=score,
            component_scores=components,
        )

    @staticmethod
    def _motion_magnitude(track: Track) -> float:
        if len(track.observations) < 2:
            return 0.0
        previous = track.observations[-2].detection.normalized_area
        current = track.latest_detection.normalized_area
        if previous <= 0:
            return 0.0
        return min(1.0, abs(current - previous) / previous)


@dataclass
class _Selection:
    event: ScoredEvent
    selected_at: float


class PrioritySelector:
    """Select one alert-worthy event, resisting distracting rapid object switching."""

    def __init__(self, alert_threshold: float, hold_seconds: float, switch_margin: float) -> None:
        self.alert_threshold = alert_threshold
        self.hold_seconds = hold_seconds
        self.switch_margin = switch_margin
        self._selection: _Selection | None = None

    def select(self, events: list[ScoredEvent], now: float | None = None) -> ScoredEvent | None:
        now = time.monotonic() if now is None else now
        candidates = [event for event in events if event.relevance_score >= self.alert_threshold]
        candidate = max(candidates, key=lambda event: event.relevance_score, default=None)
        if candidate is None:
            self._selection = None
            return None

        active = self._active_event(events)
        if active is None:
            self._selection = _Selection(candidate, now)
            LOGGER.info(
                "active_event_selected track_id=%s zone=%s relevance=%.3f",
                candidate.track_id,
                candidate.zone.value,
                candidate.relevance_score,
            )
            return candidate
        if active.track_id == candidate.track_id:
            self._selection = _Selection(candidate, self._selection.selected_at)
            return candidate

        elapsed = now - self._selection.selected_at
        sufficiently_more_important = (
            candidate.relevance_score >= active.relevance_score + self.switch_margin
        )
        if elapsed >= self.hold_seconds or sufficiently_more_important:
            self._selection = _Selection(candidate, now)
            LOGGER.info(
                "active_event_switched track_id=%s zone=%s relevance=%.3f",
                candidate.track_id,
                candidate.zone.value,
                candidate.relevance_score,
            )
            return candidate
        return active

    def clear(self) -> None:
        self._selection = None

    def _active_event(self, events: list[ScoredEvent]) -> ScoredEvent | None:
        if self._selection is None:
            return None
        return next(
            (event for event in events if event.track_id == self._selection.event.track_id),
            None,
        )
