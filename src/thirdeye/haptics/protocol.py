"""Small line protocol shared by Jetson software and ESP32 firmware."""

from __future__ import annotations

from thirdeye.models import DirectionZone, HapticCommand


def serialize_command(command: HapticCommand) -> bytes:
    """Encode a command as ASCII: `ZONE,INTENSITY\\n`."""
    return f"{command.zone.value},{command.intensity}\n".encode("ascii")


def parse_command(raw: bytes | str) -> HapticCommand:
    """Parse one strict protocol line and reject malformed or unsafe commands."""
    text = raw.decode("ascii") if isinstance(raw, bytes) else raw
    pieces = text.strip().split(",")
    if len(pieces) != 2:
        raise ValueError("Haptic command must have exactly two comma-separated fields")
    zone_text, intensity_text = (piece.strip() for piece in pieces)
    try:
        zone = DirectionZone(zone_text.upper())
    except ValueError as error:
        raise ValueError(f"Unsupported haptic zone: {zone_text!r}") from error
    try:
        intensity = int(intensity_text)
    except ValueError as error:
        raise ValueError(f"Haptic intensity must be an integer: {intensity_text!r}") from error
    return HapticCommand(zone=zone, intensity=intensity)
