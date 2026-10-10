"""Camera source adapters and multi-camera coordination."""

from thirdeye.camera.camera_manager import CameraManager
from thirdeye.camera.sources import (
    CameraSource,
    MockCameraSource,
    MockObject,
    OpenCVCameraSource,
    VideoFileCameraSource,
)

__all__ = [
    "CameraManager",
    "CameraSource",
    "MockCameraSource",
    "MockObject",
    "OpenCVCameraSource",
    "VideoFileCameraSource",
]
