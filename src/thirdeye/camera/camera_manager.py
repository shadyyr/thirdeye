"""Lifecycle management for separately tagged camera streams."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from thirdeye.camera.sources import CameraSource
from thirdeye.models import CameraName, FramePacket

LOGGER = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, sources: Iterable[CameraSource]) -> None:
        self._sources = list(sources)
        if not self._sources:
            raise ValueError("CameraManager requires at least one source")

    def start(self) -> None:
        for source in self._sources:
            source.start()
            LOGGER.info("camera_connected source=%s", type(source).__name__)

    def read_all(self) -> list[FramePacket]:
        packets: list[FramePacket] = []
        for source in self._sources:
            packet = source.read()
            if packet is not None:
                packets.append(packet)
        return packets

    def close(self) -> None:
        for source in self._sources:
            source.close()
            LOGGER.info("camera_disconnected source=%s", type(source).__name__)

    def __enter__(self) -> CameraManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_camera_names(packets: Iterable[FramePacket]) -> None:
    """Reject duplicate camera frames in a collection before downstream processing."""
    names = [packet.source_camera for packet in packets]
    if len(names) != len(set(names)):
        raise ValueError("Only one frame per camera may be processed in a batch")
    if not all(isinstance(name, CameraName) for name in names):
        raise ValueError("Every frame must have a valid CameraName")
