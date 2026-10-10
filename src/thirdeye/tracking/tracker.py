"""A conservative IoU tracker for the MVP's person-first detection path."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace

from thirdeye.models import BoundingBox, Detection, Track, TrackObservation

LOGGER = logging.getLogger(__name__)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    overlap_width = max(0.0, min(first.right, second.right) - max(first.x, second.x))
    overlap_height = max(0.0, min(first.bottom, second.bottom) - max(first.y, second.y))
    intersection = overlap_width * overlap_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


class SimpleTracker:
    """Matches same-class detections by IoU without hidden model-specific behavior."""

    def __init__(
        self,
        iou_match_threshold: float,
        max_missed_frames: int,
        history_size: int,
    ) -> None:
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        self.history_size = history_size
        self._tracks: dict[str, Track] = {}
        self._next_ids: defaultdict[str, int] = defaultdict(int)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        """Update tracks for one camera frame and return currently visible tracks."""
        unmatched_track_ids = set(self._tracks)
        visible: list[Track] = []

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            track = self._best_match(detection, unmatched_track_ids)
            if track is None:
                track = self._create_track(detection, timestamp)
            else:
                unmatched_track_ids.remove(track.track_id)
                detection = replace(detection, object_id=track.track_id)
                track.observations.append(
                    TrackObservation(detection=detection, timestamp=timestamp)
                )
                track.observations = track.observations[-self.history_size :]
                track.missed_frames = 0
            visible.append(track)

        for track_id in unmatched_track_ids:
            self._tracks[track_id].missed_frames += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }
        for track_id in unmatched_track_ids:
            if track_id not in self._tracks:
                LOGGER.info("track_lost track_id=%s", track_id)
        return visible

    def _best_match(self, detection: Detection, candidates: set[str]) -> Track | None:
        matches = [
            (
                intersection_over_union(
                    detection.bounding_box, self._tracks[track_id].latest_detection.bounding_box
                ),
                track_id,
            )
            for track_id in candidates
            if self._tracks[track_id].class_name == detection.class_name
            and self._tracks[track_id].source_camera == detection.source_camera
        ]
        if not matches:
            return None
        score, track_id = max(matches)
        return self._tracks[track_id] if score >= self.iou_match_threshold else None

    def _create_track(self, detection: Detection, timestamp: float) -> Track:
        camera_prefix = detection.source_camera.value
        self._next_ids[camera_prefix] += 1
        track_id = f"{camera_prefix}-{self._next_ids[camera_prefix]:03d}"
        tracked_detection = replace(detection, object_id=track_id)
        track = Track(
            track_id=track_id,
            source_camera=detection.source_camera,
            class_name=detection.class_name,
            observations=[TrackObservation(detection=tracked_detection, timestamp=timestamp)],
        )
        self._tracks[track_id] = track
        LOGGER.info("track_created track_id=%s class=%s", track_id, detection.class_name)
        return track
