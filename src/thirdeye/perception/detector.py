"""Detector contracts; model-specific code stays outside decision logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from thirdeye.camera.sources import MockFrame
from thirdeye.models import BoundingBox, Detection, FramePacket


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: FramePacket) -> list[Detection]:
        """Return structured detections for one camera frame."""


class MockDetector(Detector):
    """Repeatable person detections that exercise tracking, motion, and haptics."""

    def __init__(self, confidence: float = 0.91) -> None:
        self.confidence = confidence

    def detect(self, frame: FramePacket) -> list[Detection]:
        if not isinstance(frame.payload, MockFrame):
            return []
        if frame.payload.scene == "empty":
            return []

        progress = min(frame.frame_index, 20)
        if frame.source_camera.value == "front":
            width = 80 + progress * 11
            height = 170 + progress * 15
            x = frame.width * 0.68 - width / 2
        else:
            # Rear is present for dashboard visibility but deliberately lower salience.
            width, height = 45, 95
            x = frame.width * 0.48 - width / 2
        y = frame.height * 0.3
        return [
            Detection(
                class_name="person",
                confidence=self.confidence,
                source_camera=frame.source_camera,
                bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                frame_width=frame.width,
                frame_height=frame.height,
            )
        ]


class UltralyticsDetector(Detector):
    """Optional local YOLO adapter. The configured model is loaded only on demand."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        tracked_classes: set[str],
    ) -> None:
        if not model_path:
            raise ValueError("A model path must be configured for the Ultralytics detector")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Install the vision extras before using this backend."
            ) from error
        self._model: Any = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracked_classes = tracked_classes

    def detect(self, frame: FramePacket) -> list[Detection]:
        results = self._model.predict(frame.payload, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_name = str(names[int(box.cls[0])])
                confidence = float(box.conf[0])
                if class_name not in self.tracked_classes:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        source_camera=frame.source_camera,
                        bounding_box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                        frame_width=frame.width,
                        frame_height=frame.height,
                    )
                )
        return detections
