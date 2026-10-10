"""Detector contracts; model-specific code stays outside decision logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from thirdeye.camera.sources import MockFrame, MockObject
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
        return [self._to_detection(frame, mock_object) for mock_object in frame.payload.objects]

    def _to_detection(self, frame: FramePacket, mock_object: MockObject) -> Detection:
        normalized_width = (mock_object.area_fraction * mock_object.aspect_ratio) ** 0.5
        normalized_height = normalized_width / mock_object.aspect_ratio
        width = normalized_width * frame.width
        height = normalized_height * frame.height
        x = mock_object.center_x * frame.width - width / 2
        y = mock_object.center_y * frame.height - height / 2
        return Detection(
            class_name=mock_object.class_name,
            confidence=mock_object.confidence,
            source_camera=frame.source_camera,
            bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
            frame_width=frame.width,
            frame_height=frame.height,
        )


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
