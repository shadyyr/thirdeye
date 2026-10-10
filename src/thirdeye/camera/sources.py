"""Camera abstractions for mock, video-file, USB, and CSI/GStreamer sources."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thirdeye.models import CameraName, FramePacket


class CameraSource(ABC):
    """A pull-based source that produces frames tagged with their physical camera."""

    @abstractmethod
    def start(self) -> None:
        """Allocate source resources."""

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next frame, or None at end-of-stream."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""


@dataclass(frozen=True)
class MockObject:
    """A deterministic, normalized object used to exercise the full decision path."""

    object_key: str
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    area_fraction: float
    aspect_ratio: float = 0.45

    def __post_init__(self) -> None:
        if not 0 <= self.center_x <= 1 or not 0 <= self.center_y <= 1:
            raise ValueError("Mock object centers must be normalized between 0 and 1")
        if not 0 < self.area_fraction <= 1:
            raise ValueError("Mock object area_fraction must be between 0 and 1")
        if self.aspect_ratio <= 0:
            raise ValueError("Mock object aspect_ratio must be positive")


@dataclass(frozen=True)
class MockFrame:
    """Dependency-free frame payload used by the software-only demo pipeline."""

    scene: str
    index: int
    objects: tuple[MockObject, ...] = ()


class MockCameraSource(CameraSource):
    def __init__(
        self,
        source_camera: CameraName,
        *,
        width: int = 1280,
        height: int = 720,
        scene: str = "approaching_person",
        scripted_objects: Sequence[Sequence[MockObject]] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.source_camera = source_camera
        self.width = width
        self.height = height
        self.scene = scene
        self.scripted_objects = scripted_objects
        self.max_frames = max_frames
        self._frame_index = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def read(self) -> FramePacket | None:
        if not self._started:
            raise RuntimeError("Camera source must be started before reading")
        if self.max_frames is not None and self._frame_index >= self.max_frames:
            return None
        if self.scripted_objects is not None and self._frame_index >= len(self.scripted_objects):
            return None
        objects = (
            tuple(self.scripted_objects[self._frame_index])
            if self.scripted_objects is not None
            else self._default_objects()
        )
        packet = FramePacket(
            source_camera=self.source_camera,
            timestamp=time.monotonic(),
            width=self.width,
            height=self.height,
            payload=MockFrame(scene=self.scene, index=self._frame_index, objects=objects),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        self._started = False

    def _default_objects(self) -> tuple[MockObject, ...]:
        """Retain a useful direct-instantiation default outside named rehearsal scenarios."""
        if self.scene == "empty":
            return ()
        if self.source_camera == CameraName.REAR:
            return (
                MockObject(
                    object_key="rear-background",
                    class_name="person",
                    confidence=0.91,
                    center_x=0.48,
                    center_y=0.45,
                    area_fraction=0.005,
                ),
            )
        area = min(0.16, 0.015 + self._frame_index * 0.009)
        return (
            MockObject(
                object_key="front-approaching",
                class_name="person",
                confidence=0.91,
                center_x=0.68,
                center_y=0.45,
                area_fraction=area,
            ),
        )


class OpenCVCameraSource(CameraSource):
    """Optional OpenCV adapter for a camera index or a GStreamer/CSI pipeline string."""

    def __init__(
        self,
        source_camera: CameraName,
        source: int | str,
        capture_backend: int | None = None,
    ) -> None:
        self.source_camera = source_camera
        self.source = source
        self.capture_backend = capture_backend
        self._capture: Any | None = None
        self._frame_index = 0

    def start(self) -> None:
        cv2 = _import_cv2()
        if self.capture_backend is None:
            self._capture = cv2.VideoCapture(self.source)
        else:
            self._capture = cv2.VideoCapture(self.source, self.capture_backend)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            message = f"Could not open {self.source_camera.value} camera source: {self.source!r}"
            raise RuntimeError(message)

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise RuntimeError("Camera source must be started before reading")
        ok, image = self._capture.read()
        if not ok:
            return None
        height, width = image.shape[:2]
        packet = FramePacket(
            source_camera=self.source_camera,
            timestamp=time.monotonic(),
            width=width,
            height=height,
            payload=image,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoFileCameraSource(OpenCVCameraSource):
    """OpenCV-backed video replay source; useful for repeatable demos and development."""

    def __init__(self, source_camera: CameraName, path: str | Path) -> None:
        super().__init__(source_camera, str(Path(path)))


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for camera/video sources. "
            "Install it with: pip install -e '.[vision]'"
        ) from error
    return cv2
