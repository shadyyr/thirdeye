from __future__ import annotations

import unittest

from thirdeye.haptics.controller import HapticController
from thirdeye.haptics.protocol import parse_command, serialize_command
from thirdeye.haptics.transport import MockHapticTransport
from thirdeye.models import DirectionZone, HapticCommand


class HapticProtocolTests(unittest.TestCase):
    def test_serializes_and_parses_direction_command(self) -> None:
        command = HapticCommand(DirectionZone.RIGHT, 82)
        self.assertEqual(serialize_command(command), b"RIGHT,82\n")
        self.assertEqual(parse_command("RIGHT,82\n"), command)

    def test_stop_is_explicit_and_safe(self) -> None:
        command = parse_command("STOP,0")
        self.assertEqual(command, HapticCommand(DirectionZone.STOP, 0))
        with self.assertRaises(ValueError):
            parse_command("STOP,10")

    def test_rejects_invalid_commands(self) -> None:
        with self.assertRaises(ValueError):
            parse_command("UP,50")
        with self.assertRaises(ValueError):
            parse_command("LEFT,101")

    def test_controller_fail_safe_stop_is_sent(self) -> None:
        transport = MockHapticTransport()
        controller = HapticController(transport, minimum_intensity=25)
        controller.stop()
        self.assertEqual(transport.commands[-1], HapticCommand(DirectionZone.STOP, 0))


if __name__ == "__main__":
    unittest.main()
