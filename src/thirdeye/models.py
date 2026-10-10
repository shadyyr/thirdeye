"""Shared data contracts for the Third Eye software pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CameraName(str, Enum):
    FRONT = "front"
    REAR = "rear"


class DirectionZone(str, Enum):
    FRONT = "FRONT"
    RIGHT = "RIGHT"
    REAR = "REAR"
    LEFT = "LEFT"
    STOP = "STOP"


class MotionState(str, Enum):
    UNKNOWN = "UNKNOWN"
    STATIONARY = "STATIONARY"
    APPROACHING = "APPROACHING"
    MOVING_AWAY = "MOVING_AWAY"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space rectangle represented as top-left origin, width, and height."""

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def normalized_center(self, frame_width: int, frame_height: int) -> tuple[float, float]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        return (
            min(1.0, max(0.0, (self.x + self.width / 2) / frame_width)),
            min(1.0, max(0.0, (self.y + self.height / 2) / frame_height)),
        )


@dataclass(frozen=True)
class FramePacket:
    source_camera: CameraName
    timestamp: float
    width: int
    height: int
    payload: Any
    frame_index: int = 0


@dataclass(frozen=True)
class Detection:
    """Detector output; tracking fills in object_id and subsequent metadata."""

    class_name: str
    confidence: float
    source_camera: CameraName
    bounding_box: BoundingBox
    frame_width: int
    frame_height: int
    object_id: str | None = None
    approximate_direction: DirectionZone | None = None
    motion_state: MotionState = MotionState.UNKNOWN
    relevance_score: float | None = None

    @property
    def normalized_center_x(self) -> float:
        return self.bounding_box.normalized_center(self.frame_width, self.frame_height)[0]

    @property
    def normalized_center_y(self) -> float:
        return self.bounding_box.normalized_center(self.frame_width, self.frame_height)[1]

    @property
    def normalized_area(self) -> float:
        return min(1.0, self.bounding_box.area / (self.frame_width * self.frame_height))


@dataclass(frozen=True)
class TrackObservation:
    detection: Detection
    timestamp: float


@dataclass
class Track:
    track_id: str
    source_camera: CameraName
    class_name: str
    observations: list[TrackObservation] = field(default_factory=list)
    missed_frames: int = 0
    motion_state: MotionState = MotionState.UNKNOWN

    @property
    def latest(self) -> TrackObservation:
        if not self.observations:
            raise RuntimeError("A track must have at least one observation")
        return self.observations[-1]

    @property
    def latest_detection(self) -> Detection:
        return self.latest.detection


@dataclass(frozen=True)
class ScoredEvent:
    track: Track
    zone: DirectionZone
    relevance_score: float
    component_scores: dict[str, float]

    @property
    def track_id(self) -> str:
        return self.track.track_id


@dataclass(frozen=True)
class HapticCommand:
    zone: DirectionZone
    intensity: int

    def __post_init__(self) -> None:
        if not 0 <= self.intensity <= 100:
            raise ValueError("Haptic intensity must be between 0 and 100")
        if self.zone == DirectionZone.STOP and self.intensity != 0:
            raise ValueError("STOP commands must have zero intensity")
