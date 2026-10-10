"""Configurable mapping from a camera-local x position to belt zones."""

from __future__ import annotations

from thirdeye.models import CameraName, DirectionZone


class DirectionMapper:
    def __init__(self, config: dict) -> None:
        self.left_boundary = config["left_boundary"]
        self.right_boundary = config["right_boundary"]
        self.camera_mappings = config["camera_mappings"]

    def zone_for(
        self,
        source_camera: CameraName,
        normalized_center_x: float,
        *,
        mirror: bool = False,
    ) -> DirectionZone:
        if not 0 <= normalized_center_x <= 1:
            raise ValueError("normalized_center_x must be between 0 and 1")
        x_position = 1 - normalized_center_x if mirror else normalized_center_x
        if x_position < self.left_boundary:
            position = "left"
        elif x_position > self.right_boundary:
            position = "right"
        else:
            position = "center"
        try:
            return DirectionZone(self.camera_mappings[source_camera.value][position])
        except KeyError as error:
            message = f"No direction mapping for {source_camera.value}:{position}"
            raise ValueError(message) from error
