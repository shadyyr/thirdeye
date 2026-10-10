"""Camera-orientation checks that can be rehearsed before the belt is connected."""

from __future__ import annotations

from dataclasses import dataclass

from thirdeye.models import CameraName, DirectionZone
from thirdeye.spatial.direction import DirectionMapper


@dataclass(frozen=True)
class OrientationCheck:
    camera: CameraName
    camera_position: str
    normalized_center_x: float
    expected_zone: DirectionZone


def orientation_checks(config: dict) -> tuple[OrientationCheck, ...]:
    """Return the physical calibration matrix implied by the current configuration."""
    mapper = DirectionMapper(config["direction"])
    positions = (("left", 0.1), ("center", 0.5), ("right", 0.9))
    checks = []
    for camera in CameraName:
        mirror = config["cameras"][camera.value]["mirror_horizontal"]
        for camera_position, x_coordinate in positions:
            checks.append(
                OrientationCheck(
                    camera=camera,
                    camera_position=camera_position,
                    normalized_center_x=x_coordinate,
                    expected_zone=mapper.zone_for(camera, x_coordinate, mirror=mirror),
                )
            )
    return tuple(checks)


def format_orientation_report(config: dict) -> str:
    """Format an operator-facing calibration checklist for terminal use."""
    lines = ["Third Eye orientation calibration", "Place a person at each camera-local position:"]
    for check in orientation_checks(config):
        lines.append(
            f"- {check.camera.value.upper():5} {check.camera_position:6} "
            f"(x={check.normalized_center_x:.1f}) -> {check.expected_zone.value} belt motor"
        )
    lines.append(
        "If a physical result differs, update camera_mappings or mirror_horizontal "
        "in config/local.json."
    )
    return "\n".join(lines)
