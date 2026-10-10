"""Named scenarios that make salience behavior repeatable before hardware integration."""

from __future__ import annotations

from dataclasses import dataclass

from thirdeye.camera import MockObject
from thirdeye.models import CameraName, DirectionZone


@dataclass(frozen=True)
class MockScenario:
    """A synchronized, finite set of front/rear camera frames and expected behavior."""

    name: str
    description: str
    objects_by_camera: dict[CameraName, tuple[tuple[MockObject, ...], ...]]
    expected_alert_zone: DirectionZone | None

    def frames_for(self, camera: CameraName) -> tuple[tuple[MockObject, ...], ...]:
        return self.objects_by_camera[camera]

    @property
    def frame_count(self) -> int:
        return max(len(frames) for frames in self.objects_by_camera.values())


def scenario_names() -> tuple[str, ...]:
    return tuple(_SCENARIOS)


def get_scenario(name: str) -> MockScenario:
    try:
        return _SCENARIOS[name]
    except KeyError as error:
        available = ", ".join(scenario_names())
        message = f"Unknown mock scenario {name!r}. Available scenarios: {available}"
        raise ValueError(message) from error


def _person(
    object_key: str,
    center_x: float,
    area_fraction: float,
    *,
    confidence: float = 0.91,
) -> MockObject:
    return MockObject(
        object_key=object_key,
        class_name="person",
        confidence=confidence,
        center_x=center_x,
        center_y=0.48,
        area_fraction=area_fraction,
    )


def _frames(objects: list[tuple[MockObject, ...]]) -> tuple[tuple[MockObject, ...], ...]:
    return tuple(objects)


def _single_object_frames(objects: list[MockObject]) -> list[tuple[MockObject, ...]]:
    return [(mock_object,) for mock_object in objects]


def _front_only(
    front_objects: list[tuple[MockObject, ...]],
    *,
    expected_alert_zone: DirectionZone | None,
    name: str,
    description: str,
) -> MockScenario:
    empty_rear = _frames([() for _ in front_objects])
    return MockScenario(
        name=name,
        description=description,
        objects_by_camera={CameraName.FRONT: _frames(front_objects), CameraName.REAR: empty_rear},
        expected_alert_zone=expected_alert_zone,
    )


_APPROACH_AREAS = (0.012, 0.018, 0.028, 0.045, 0.07, 0.1, 0.135, 0.16)
_RETREAT_AREAS = tuple(reversed(_APPROACH_AREAS))

_SCENARIOS: dict[str, MockScenario] = {
    "front_right_approach": _front_only(
        _single_object_frames(
            [_person("approaching-right", 0.78, area) for area in _APPROACH_AREAS]
        ),
        expected_alert_zone=DirectionZone.RIGHT,
        name="front_right_approach",
        description="A person approaches from the front-right and should produce RIGHT haptics.",
    ),
    "front_stationary": _front_only(
        _single_object_frames(
            [_person("stationary-front", 0.5, 0.035) for _ in _APPROACH_AREAS]
        ),
        expected_alert_zone=None,
        name="front_stationary",
        description="A stationary front person remains below the event threshold and stays quiet.",
    ),
    "front_moving_away": _front_only(
        _single_object_frames([_person("retreating-front", 0.5, area) for area in _RETREAT_AREAS]),
        expected_alert_zone=None,
        name="front_moving_away",
        description="A person moving away should not become an alert-worthy event.",
    ),
    "rear_left_approach": MockScenario(
        name="rear_left_approach",
        description=(
            "A person approaches at rear-camera left; "
            "calibration decides the physical belt side."
        ),
        objects_by_camera={
            CameraName.FRONT: _frames([() for _ in _APPROACH_AREAS]),
            CameraName.REAR: _single_object_frames(
                [_person("approaching-rear-left", 0.18, area) for area in _APPROACH_AREAS]
            ),
        },
        expected_alert_zone=DirectionZone.LEFT,
    ),
    "competing_people": _front_only(
        [
            (
                _person("stationary-central", 0.5, 0.12),
                _person("approaching-right", 0.78, area),
            )
            for area in _APPROACH_AREAS
        ],
        expected_alert_zone=DirectionZone.RIGHT,
        name="competing_people",
        description="An approaching person must beat a more visually prominent stationary person.",
    ),
    "approach_then_disappear": _front_only(
        [
            *[(_person("disappearing-person", 0.5, area),) for area in _APPROACH_AREAS[:5]],
            (),
            (),
            (),
        ],
        expected_alert_zone=DirectionZone.FRONT,
        name="approach_then_disappear",
        description="An alert-worthy person leaves the frame and must resolve to STOP.",
    ),
}
