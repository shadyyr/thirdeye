"""Persistent anonymous entity tracker (ByteTrack-inspired, kept explainable).

Per frame:
  1. Every track predicts where its box is now (damped constant velocity).
  2. Stage 1: confident detections are matched to all tracks of their
     *family*. People only match people; backpack/handbag/suitcase are one
     "portable" family (YOLO often flips the label of one bag). Each track
     keeps a confidence-weighted, slowly decaying class vote and reports the
     winning class; its public ID never changes when the label flips.
  3. Stage 2: low-confidence detections may only extend confirmed, unmatched
     tracks whose last detection was within TRACK_LOW_CONF_EXTEND_SECONDS.
     They never start a track and cannot re-acquire a long-coasting track
     (stops duplicate low-confidence boxes from keeping ghosts alive).
  4. Unmatched confident detections start tentative tracks. A tentative track
     becomes confirmed after TRACK_MIN_HITS consecutive matched frames and
     only then receives a public ID from one global counter shared by all
     classes. A tentative track that misses a frame is dropped.
  5. Confirmed tracks coast on their last box until they have been undetected
     for more than TRACK_MAX_MISSED_PERSON_SECONDS / _OBJECT_SECONDS
     (time-based, so behavior does not depend on FPS).
  6. Merge safety net: two confirmed tracks of the same family whose boxes
     overlap heavily (containment >= TRACK_MERGE_CONTAIN and IoU >=
     TRACK_MERGE_MIN_IOU) continuously for >= TRACK_MERGE_SECONDS are one
     entity. The newer (higher public ID) is merged into the older, which
     keeps its ID and takes over the most recently detected box. Merges are
     recorded in ``tracker.merges`` as (now, kept_id, merged_id).
  7. Removed portable objects are remembered for a while; a new portable
     object confirmed near the old spot reuses the old ID (any class of the
     family).

Matching uses Hungarian assignment on a cost that mixes IoU and center
distance, with a gate: a pair is allowed if IoU >= TRACK_MIN_IOU or the center
distance is within TRACK_MAX_CENTER_DIST_RATIO of the track's box diagonal.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from thirdeye.core import settings
from thirdeye.core.types import PERSON, PORTABLE_OBJECT_CLASSES, BBox, Detection, TrackedEntity

# Velocity is an exponential average of per-frame center motion.
_VELOCITY_ALPHA = 0.5
# Each missed frame shrinks the velocity so a coasting box slows to a stop.
_VELOCITY_DAMPING = 0.6
# Reported confidence = max over this many recent matched frames.
_CONFIDENCE_WINDOW = 5
# Cost for pairs rejected by the gate (never accepted after assignment).
_GATED_COST = 1e6
# Small extra cost when a detection's label differs from the track's current
# class (same family): prefers same-label pairs when two bags are close.
_CLASS_MISMATCH_COST = 0.1
# Class votes decay per matched frame (half-life ~69 matches) so a long-lived
# track can still change its mind, but one odd frame cannot flip the label.
_VOTE_DECAY = 0.99
# The reported class only switches when another label's vote beats it by this
# factor, so two near-equal labels (backpack vs handbag) do not flicker.
_CLASS_SWITCH_MARGIN = 1.3
# Diagnostics: keep only this many recent merges.
_MAX_MERGE_LOG = 200

_PORTABLE = "portable"


def _family(class_name: str) -> str:
    return _PORTABLE if class_name in PORTABLE_OBJECT_CLASSES else class_name


def _center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diagonal(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def _shift(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _overlap(a: BBox, b: BBox) -> tuple[float, float]:
    """(iou, intersection / smaller area)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0, 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def box_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two x1,y1,x2,y2 boxes."""
    return _overlap(a, b)[0]


class _Track:
    """Internal state of one track (tentative or confirmed)."""

    def __init__(self, detection: Detection, now: float) -> None:
        self.family = _family(detection.class_name)
        self.class_name = detection.class_name  # current vote winner
        self.votes: dict[str, float] = {detection.class_name: detection.confidence}
        self.bbox: BBox = tuple(detection.bbox)  # latest matched box (reported)
        self.predicted: BBox = self.bbox  # box expected in the current frame
        self.velocity = (0.0, 0.0)  # pixels per frame
        self.hits = 1  # consecutive matched frames
        self.frames_missed = 0  # frames since the last detection (0 = this frame)
        self.public_id: int | None = None  # set at confirmation
        self.last_detected = now  # time of the last matched detection
        self.confidences: deque[float] = deque([detection.confidence], maxlen=_CONFIDENCE_WINDOW)

    @property
    def confirmed(self) -> bool:
        return self.public_id is not None

    def predict(self) -> None:
        """Advance the predicted box by one frame of (damped) velocity."""
        if self.frames_missed > 0:
            vx, vy = self.velocity
            self.velocity = (vx * _VELOCITY_DAMPING, vy * _VELOCITY_DAMPING)
        self.predicted = _shift(self.predicted, *self.velocity)

    def _vote(self, class_name: str, confidence: float) -> None:
        for name in self.votes:
            self.votes[name] *= _VOTE_DECAY
        self.votes[class_name] = self.votes.get(class_name, 0.0) + confidence
        self._update_class()

    def _update_class(self) -> None:
        best = max(self.votes, key=self.votes.__getitem__)
        # Hysteresis: only switch on a clearly larger accumulated vote.
        if self.votes[best] > _CLASS_SWITCH_MARGIN * self.votes.get(self.class_name, 0.0):
            self.class_name = best

    def match(self, detection: Detection, now: float) -> None:
        old_cx, old_cy = _center(self.bbox)
        new_cx, new_cy = _center(detection.bbox)
        # Average motion over the frames since the last match.
        steps = self.frames_missed + 1
        mx, my = (new_cx - old_cx) / steps, (new_cy - old_cy) / steps
        vx, vy = self.velocity
        self.velocity = (
            _VELOCITY_ALPHA * mx + (1 - _VELOCITY_ALPHA) * vx,
            _VELOCITY_ALPHA * my + (1 - _VELOCITY_ALPHA) * vy,
        )
        self.bbox = tuple(detection.bbox)
        self.predicted = self.bbox
        self.hits += 1
        self.frames_missed = 0
        self.last_detected = now
        self.confidences.append(detection.confidence)
        self._vote(detection.class_name, detection.confidence)

    def miss(self) -> None:
        self.frames_missed += 1
        self.hits = 0

    def max_missed_seconds(self) -> float:
        if self.family == PERSON:
            return settings.TRACK_MAX_MISSED_PERSON_SECONDS
        return settings.TRACK_MAX_MISSED_OBJECT_SECONDS

    def absorb(self, other: "_Track") -> None:
        """Take over a duplicate track: newest box/velocity, merged votes."""
        if other.last_detected > self.last_detected or (
            other.last_detected == self.last_detected and other.frames_missed < self.frames_missed
        ):
            self.bbox = other.bbox
            self.predicted = other.predicted
            self.velocity = other.velocity
            self.frames_missed = other.frames_missed
            self.last_detected = other.last_detected
            self.hits = max(self.hits, other.hits)
        for name, vote in other.votes.items():
            self.votes[name] = self.votes.get(name, 0.0) + vote
        self._update_class()
        self.confidences.extend(other.confidences)


@dataclass
class _LostObject:
    """A removed confirmed portable object that may be revived."""

    family: str
    bbox: BBox
    public_id: int
    removed_at: float


class EntityTracker:
    """Assigns stable anonymous IDs to people and portable objects.

    Diagnostics: ``merges`` lists recent duplicate-track merges as
    (now, kept_id, merged_id), newest last, bounded to the last 200.
    """

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._lost_objects: list[_LostObject] = []
        self._next_id = 1
        # (older_id, newer_id) -> time the pair started overlapping.
        self._overlap_since: dict[tuple[int, int], float] = {}
        self.merges: list[tuple[float, int, int]] = []

    def reset(self) -> None:
        """Forget all tracks and memories and restart IDs at 1."""
        self._tracks.clear()
        self._lost_objects.clear()
        self._overlap_since.clear()
        self.merges.clear()
        self._next_id = 1

    def update(self, detections: list[Detection], now: float) -> list[TrackedEntity]:
        """Process all detections of one frame; return confirmed entities."""
        for track in self._tracks:
            track.predict()

        new_conf = settings.TRACK_NEW_CONFIDENCE
        high: list[Detection] = []
        low: list[Detection] = []
        for d in detections:
            (high if d.confidence >= new_conf.get(d.class_name, 1.0) else low).append(d)

        # Stage 1: confident detections vs. all tracks.
        matched, unmatched_high = self._associate(self._tracks, high)
        for track, det in matched:
            track.match(det, now)
        matched_ids = {id(t) for t, _ in matched}

        # Stage 2: weak detections may only extend confirmed, unmatched tracks
        # that were detected very recently.
        if low:
            window = settings.TRACK_LOW_CONF_EXTEND_SECONDS
            remaining = [
                t for t in self._tracks
                if id(t) not in matched_ids and t.confirmed and now - t.last_detected <= window
            ]
            matched_low, _ = self._associate(remaining, low)
            for track, det in matched_low:
                track.match(det, now)
                matched_ids.add(id(track))

        # Misses, deletions and confirmations.
        survivors: list[_Track] = []
        for track in self._tracks:
            if id(track) not in matched_ids:
                track.miss()
                if not track.confirmed:
                    continue  # tentative tracks die on their first miss
                if now - track.last_detected > track.max_missed_seconds():
                    self._forget(track, now)
                    continue
            elif not track.confirmed and track.hits >= settings.TRACK_MIN_HITS:
                self._confirm(track, now)
            survivors.append(track)

        # Unmatched confident detections start tentative tracks.
        for det in unmatched_high:
            track = _Track(det, now)
            if settings.TRACK_MIN_HITS <= 1:
                self._confirm(track, now)
            survivors.append(track)
        self._tracks = survivors

        self._merge_duplicates(now)
        self._prune_lost_objects(now)
        confirmed = sorted((t for t in self._tracks if t.confirmed), key=lambda t: t.public_id or 0)
        return [self._to_entity(t, now) for t in confirmed]

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associate(
        self, tracks: list[_Track], detections: list[Detection]
    ) -> tuple[list[tuple[_Track, Detection]], list[Detection]]:
        """Hungarian matching per family. Returns (matches, unmatched detections)."""
        matches: list[tuple[_Track, Detection]] = []
        used: set[int] = set()  # indices into detections
        if tracks and detections:
            families: dict[str, list[int]] = {}
            for i, d in enumerate(detections):
                families.setdefault(_family(d.class_name), []).append(i)
            for fam, det_idx in families.items():
                fam_tracks = [t for t in tracks if t.family == fam]
                if not fam_tracks:
                    continue
                cost = np.array(
                    [[self._cost(t, detections[i]) for i in det_idx] for t in fam_tracks]
                )
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < _GATED_COST:
                        matches.append((fam_tracks[r], detections[det_idx[c]]))
                        used.add(det_idx[c])
        # Keep input order so same-frame confirmations get IDs deterministically.
        unmatched = [d for i, d in enumerate(detections) if i not in used]
        return matches, unmatched

    @staticmethod
    def _cost(track: _Track, det: Detection) -> float:
        """0 (perfect) .. ~2.1 (worst allowed); _GATED_COST if the pair is not allowed."""
        iou = box_iou(track.predicted, det.bbox)
        tcx, tcy = _center(track.predicted)
        dcx, dcy = _center(det.bbox)
        max_dist = settings.TRACK_MAX_CENTER_DIST_RATIO * max(_diagonal(track.predicted), 1.0)
        dist_norm = math.hypot(dcx - tcx, dcy - tcy) / max_dist
        if iou < settings.TRACK_MIN_IOU and dist_norm > 1.0:
            return _GATED_COST
        cost = (1.0 - iou) + min(dist_norm, 1.0)
        if det.class_name != track.class_name:
            cost += _CLASS_MISMATCH_COST
        return cost

    # ------------------------------------------------------------------
    # Duplicate-track merge
    # ------------------------------------------------------------------

    def _merge_duplicates(self, now: float) -> None:
        """Merge confirmed same-family tracks that overlap heavily for long enough."""
        confirmed = sorted((t for t in self._tracks if t.confirmed), key=lambda t: t.public_id)
        if len(confirmed) < 2:
            self._overlap_since.clear()
            return
        contain_min = settings.TRACK_MERGE_CONTAIN
        iou_min = settings.TRACK_MERGE_MIN_IOU
        hold = settings.TRACK_MERGE_SECONDS
        recent = settings.TRACK_LOW_CONF_EXTEND_SECONDS
        overlapping: dict[tuple[int, int], float] = {}
        merged: set[int] = set()  # id() of tracks merged away this frame
        for i, older in enumerate(confirmed):
            if id(older) in merged:
                continue
            for newer in confirmed[i + 1:]:
                if id(newer) in merged or newer.family != older.family:
                    continue
                iou, contain = _overlap(older.bbox, newer.bbox)
                if contain < contain_min or iou < iou_min:
                    continue
                # People: YOLO duplicates are detected together. A coasting
                # (departed) person's stale box overlapping a newcomer is a
                # different person, so it must not hand over its ID.
                if older.family == PERSON and not (
                    now - older.last_detected <= recent and now - newer.last_detected <= recent
                ):
                    continue
                key = (older.public_id, newer.public_id)
                since = self._overlap_since.get(key, now)
                if now - since >= hold:
                    older.absorb(newer)
                    merged.add(id(newer))
                    self.merges.append((now, older.public_id, newer.public_id))
                else:
                    overlapping[key] = since
        # Pairs that stopped overlapping (or were merged) restart their clock.
        self._overlap_since = overlapping
        if merged:
            self._tracks = [t for t in self._tracks if id(t) not in merged]
            if len(self.merges) > _MAX_MERGE_LOG:
                del self.merges[:-_MAX_MERGE_LOG]

    # ------------------------------------------------------------------
    # IDs and object memory
    # ------------------------------------------------------------------

    def _confirm(self, track: _Track, now: float) -> None:
        if track.family == _PORTABLE:
            revived = self._revive_id(track, now)
            if revived is not None:
                track.public_id = revived
                return
        track.public_id = self._next_id
        self._next_id += 1

    def _forget(self, track: _Track, now: float) -> None:
        """Called when a confirmed track is removed."""
        if track.family == _PORTABLE and track.public_id is not None:
            self._lost_objects.append(_LostObject(track.family, track.bbox, track.public_id, now))

    def _prune_lost_objects(self, now: float) -> None:
        if not self._lost_objects:
            return
        horizon = settings.TRACK_OBJECT_REVIVE_SECONDS
        self._lost_objects = [m for m in self._lost_objects if now - m.removed_at <= horizon]

    def _revive_id(self, track: _Track, now: float) -> int | None:
        """Reuse the ID of the nearest recently lost object of the same family."""
        self._prune_lost_objects(now)
        cx, cy = _center(track.bbox)
        best: _LostObject | None = None
        best_dist = math.inf
        for memory in self._lost_objects:
            if memory.family != track.family:
                continue
            mx, my = _center(memory.bbox)
            dist = math.hypot(cx - mx, cy - my)
            limit = settings.TRACK_OBJECT_REVIVE_DIST_RATIO * _diagonal(memory.bbox)
            if dist <= limit and dist < best_dist:
                best, best_dist = memory, dist
        if best is None:
            return None
        self._lost_objects.remove(best)  # each memory revives at most once
        return best.public_id

    @staticmethod
    def _to_entity(track: _Track, now: float) -> TrackedEntity:
        assert track.public_id is not None
        return TrackedEntity(
            track_id=track.public_id,
            class_name=track.class_name,
            confidence=max(track.confidences),
            bbox=track.bbox,
            center=_center(track.bbox),
            timestamp=now,
            frames_missed=track.frames_missed,
        )
