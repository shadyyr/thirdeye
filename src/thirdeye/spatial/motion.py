"""Smoothed bounding-box area heuristics for approachable MVP motion states."""

from __future__ import annotations

import logging
from statistics import fmean

from thirdeye.models import MotionState, Track

LOGGER = logging.getLogger(__name__)


def classify_area_history(
    areas: list[float], *, minimum_samples: int, smoothing_window: int, area_change_threshold: float
) -> MotionState:
    """Classify growth/shrinkage in normalized box area without pretending to know distance."""
    windowed = areas[-smoothing_window:]
    if len(windowed) < minimum_samples:
        return MotionState.UNKNOWN
    midpoint = len(windowed) // 2
    previous_average = fmean(windowed[:midpoint])
    current_average = fmean(windowed[midpoint:])
    if previous_average <= 0:
        return MotionState.UNKNOWN
    relative_change = (current_average - previous_average) / previous_average
    if relative_change >= area_change_threshold:
        return MotionState.APPROACHING
    if relative_change <= -area_change_threshold:
        return MotionState.MOVING_AWAY
    return MotionState.STATIONARY


class MotionAnalyzer:
    def __init__(self, config: dict) -> None:
        self.minimum_samples = config["minimum_samples"]
        self.smoothing_window = config["smoothing_window"]
        self.area_change_threshold = config["area_change_threshold"]

    def update(self, track: Track) -> MotionState:
        areas = [observation.detection.normalized_area for observation in track.observations]
        previous_state = track.motion_state
        track.motion_state = classify_area_history(
            areas,
            minimum_samples=self.minimum_samples,
            smoothing_window=self.smoothing_window,
            area_change_threshold=self.area_change_threshold,
        )
        if previous_state != track.motion_state:
            LOGGER.info(
                "motion_state_changed track_id=%s state=%s",
                track.track_id,
                track.motion_state.value,
            )
        return track.motion_state
