"""Safe conversion of selected events into a single haptic output."""

from __future__ import annotations

import logging

from thirdeye.haptics.transport import HapticTransport
from thirdeye.models import DirectionZone, HapticCommand, ScoredEvent

LOGGER = logging.getLogger(__name__)


class HapticController:
    def __init__(self, transport: HapticTransport, minimum_intensity: int) -> None:
        self.transport = transport
        self.minimum_intensity = minimum_intensity
        self.last_command = HapticCommand(DirectionZone.STOP, 0)

    def command_for(self, event: ScoredEvent | None) -> HapticCommand:
        if event is None:
            return HapticCommand(DirectionZone.STOP, 0)
        intensity = round(
            self.minimum_intensity + event.relevance_score * (100 - self.minimum_intensity)
        )
        intensity = min(100, max(self.minimum_intensity, intensity))
        return HapticCommand(zone=event.zone, intensity=intensity)

    def send_for(self, event: ScoredEvent | None) -> HapticCommand:
        command = self.command_for(event)
        try:
            self.transport.send(command)
        except Exception:
            LOGGER.exception("Haptic transport failed; issuing software-safe STOP state")
            self.last_command = HapticCommand(DirectionZone.STOP, 0)
            raise
        self.last_command = command
        return command

    def stop(self) -> HapticCommand:
        command = HapticCommand(DirectionZone.STOP, 0)
        try:
            self.transport.send(command)
        finally:
            self.last_command = command
        return command

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.transport.close()
