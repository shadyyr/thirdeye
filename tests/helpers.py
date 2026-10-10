from __future__ import annotations

from thirdeye.models import BoundingBox, CameraName, Detection, Track, TrackObservation


def make_track(
    *,
    track_id: str = "front-001",
    camera: CameraName = CameraName.FRONT,
    class_name: str = "person",
    areas: list[float] | None = None,
    center_x: float = 0.5,
) -> Track:
    areas = areas or [0.05]
    observations = []
    for index, area in enumerate(areas):
        # A 1000x1000 frame makes box area directly equal normalized area.
        side = (area * 1_000_000) ** 0.5
        box = BoundingBox(x=center_x * 1000 - side / 2, y=100, width=side, height=side)
        observations.append(
            TrackObservation(
                detection=Detection(
                    class_name=class_name,
                    confidence=0.9,
                    source_camera=camera,
                    bounding_box=box,
                    frame_width=1000,
                    frame_height=1000,
                    object_id=track_id,
                ),
                timestamp=float(index),
            )
        )
    return Track(
        track_id=track_id,
        source_camera=camera,
        class_name=class_name,
        observations=observations,
    )
