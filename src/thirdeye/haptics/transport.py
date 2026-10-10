"""Transport adapters intentionally isolated from perception and scoring logic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from thirdeye.haptics.protocol import serialize_command
from thirdeye.models import HapticCommand

LOGGER = logging.getLogger(__name__)


class HapticTransport(ABC):
    @abstractmethod
    def send(self, command: HapticCommand) -> None:
        """Transmit a validated haptic command."""

    def close(self) -> None:
        """Release resources where needed."""
        return None


class MockHapticTransport(HapticTransport):
    """Records commands and emits visible output for no-hardware development."""

    def __init__(self) -> None:
        self.commands: list[HapticCommand] = []
        self._last_logged_zone: str | None = None

    def send(self, command: HapticCommand) -> None:
        self.commands.append(command)
        if command.zone.value != self._last_logged_zone:
            LOGGER.info("[HAPTIC] zone=%s intensity=%d", command.zone.value, command.intensity)
            self._last_logged_zone = command.zone.value
        else:
            LOGGER.debug("[HAPTIC] zone=%s intensity=%d", command.zone.value, command.intensity)


class SerialHapticTransport(HapticTransport):
    """Optional USB serial path to the ESP32; no Wi-Fi dependency is required."""

    def __init__(self, port: str, baud_rate: int) -> None:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required for serial haptics. Install: pip install -e '.[serial]'"
            ) from error
        self._serial = serial.Serial(port=port, baudrate=baud_rate, timeout=1)

    def send(self, command: HapticCommand) -> None:
        self._serial.write(serialize_command(command))
        self._serial.flush()

    def close(self) -> None:
        self._serial.close()
