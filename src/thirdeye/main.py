"""Runnable software-only pipeline and adapters for later Jetson integration."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from threading import Thread
from typing import Any

from thirdeye.camera import (
    CameraManager,
    MockCameraSource,
    OpenCVCameraSource,
    VideoFileCameraSource,
)
from thirdeye.config import load_config_with_override
from thirdeye.dashboard import DashboardState, create_app
from thirdeye.haptics import HapticController, MockHapticTransport, SerialHapticTransport
from thirdeye.models import CameraName, DirectionZone, Track
from thirdeye.perception import MockDetector, UltralyticsDetector
from thirdeye.salience import PrioritySelector, RelevanceScorer
from thirdeye.spatial import DirectionMapper, MotionAnalyzer
from thirdeye.tracking import SimpleTracker

LOGGER = logging.getLogger(__name__)


def build_camera_manager(config: dict[str, Any], max_frames: int | None) -> CameraManager:
    sources = []
    for camera_name in CameraName:
        camera_config = config["cameras"][camera_name.value]
        source_type = camera_config["source_type"]
        source = camera_config["source"]
        if source_type == "mock":
            sources.append(
                MockCameraSource(camera_name, scene="approaching_person", max_frames=max_frames)
            )
        elif source_type == "video":
            sources.append(VideoFileCameraSource(camera_name, source))
        elif source_type in {"opencv", "csi"}:
            device: int | str = int(source) if str(source).isdigit() else source
            sources.append(OpenCVCameraSource(camera_name, device))
        else:
            raise ValueError(f"Unsupported camera source type: {source_type!r}")
    return CameraManager(sources)


def build_detector(config: dict[str, Any]):
    detector_config = config["detector"]
    if detector_config["backend"] == "mock":
        LOGGER.info("detector_loaded backend=mock")
        return MockDetector()
    if detector_config["backend"] == "ultralytics":
        LOGGER.info("detector_loaded backend=ultralytics model=%s", detector_config["model"])
        return UltralyticsDetector(
            model_path=detector_config["model"],
            confidence_threshold=detector_config["confidence_threshold"],
            tracked_classes=set(detector_config["tracked_classes"]),
        )
    raise ValueError(f"Unsupported detector backend: {detector_config['backend']!r}")


def build_haptic_controller(config: dict[str, Any]) -> HapticController:
    haptics_config = config["haptics"]
    if haptics_config["transport"] == "mock":
        transport = MockHapticTransport()
    elif haptics_config["transport"] == "serial":
        transport = SerialHapticTransport(
            haptics_config["serial_port"],
            haptics_config["baud_rate"],
        )
    else:
        raise ValueError(f"Unsupported haptic transport: {haptics_config['transport']!r}")
    return HapticController(transport, config["salience"]["minimum_intensity"])


def run_pipeline(
    config: dict[str, Any],
    *,
    max_frames: int | None = None,
    dashboard_state: DashboardState | None = None,
) -> None:
    """Run the full camera-to-haptic path. All default adapters work without hardware."""
    camera_manager = build_camera_manager(config, max_frames)
    detector = build_detector(config)
    tracking_config = config["tracking"]
    trackers = {
        camera: SimpleTracker(
            iou_match_threshold=tracking_config["iou_match_threshold"],
            max_missed_frames=tracking_config["max_missed_frames"],
            history_size=tracking_config["history_size"],
        )
        for camera in CameraName
    }
    direction_mapper = DirectionMapper(config["direction"])
    motion_analyzer = MotionAnalyzer(config["motion"])
    scorer = RelevanceScorer(config["salience"])
    selector = PrioritySelector(
        alert_threshold=config["salience"]["alert_threshold"],
        hold_seconds=config["salience"]["selection_hold_seconds"],
        switch_margin=config["salience"]["switch_margin"],
    )
    haptics = build_haptic_controller(config)
    dashboard_state = dashboard_state or DashboardState()
    frame_times: list[float] = []

    try:
        with camera_manager:
            while True:
                loop_started = time.monotonic()
                frames = camera_manager.read_all()
                if not frames:
                    break
                events = []
                visible_tracks: list[Track] = []
                for frame in frames:
                    detections = [
                        detection
                        for detection in detector.detect(frame)
                        if detection.confidence >= config["detector"]["confidence_threshold"]
                        and detection.class_name in config["detector"]["tracked_classes"]
                    ]
                    tracks = trackers[frame.source_camera].update(detections, frame.timestamp)
                    visible_tracks.extend(tracks)
                    for track in tracks:
                        motion_analyzer.update(track)
                        zone = direction_mapper.zone_for(
                            frame.source_camera,
                            track.latest_detection.normalized_center_x,
                            mirror=config["cameras"][frame.source_camera.value]["mirror_horizontal"],
                        )
                        event = scorer.score(track, zone)
                        track.observations[-1] = replace(
                            track.observations[-1],
                            detection=replace(
                                track.latest_detection,
                                approximate_direction=zone,
                                motion_state=track.motion_state,
                                relevance_score=event.relevance_score,
                            ),
                        )
                        events.append(event)

                selected_event = selector.select(events)
                command = haptics.send_for(selected_event)
                elapsed = time.monotonic() - loop_started
                frame_times.append(elapsed)
                _update_dashboard(
                    dashboard_state,
                    frames,
                    visible_tracks,
                    selected_event,
                    command.zone,
                    command.intensity,
                    elapsed,
                    frame_times,
                )
    except KeyboardInterrupt:
        LOGGER.info("Pipeline interrupted")
    finally:
        haptics.close()


def _update_dashboard(
    state: DashboardState,
    frames: list,
    tracks: list[Track],
    selected_event: Any,
    zone: DirectionZone,
    intensity: int,
    latency_seconds: float,
    frame_times: list[float],
) -> None:
    average = sum(frame_times[-30:]) / min(len(frame_times), 30)
    state.update(
        {
            "status": "running",
            "cameras": {
                frame.source_camera.value: {
                    "frame_index": frame.frame_index,
                    "resolution": f"{frame.width}x{frame.height}",
                }
                for frame in frames
            },
            "tracked_objects": [
                {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "camera": track.source_camera.value,
                    "confidence": round(track.latest_detection.confidence, 3),
                    "motion_state": track.motion_state.value,
                    "normalized_center_x": round(track.latest_detection.normalized_center_x, 3),
                }
                for track in tracks
            ],
            "selected_event": None
            if selected_event is None
            else {
                "track_id": selected_event.track_id,
                "zone": selected_event.zone.value,
                "relevance_score": round(selected_event.relevance_score, 3),
                "components": {
                    key: round(value, 3)
                    for key, value in selected_event.component_scores.items()
                },
            },
            "active_haptic": {"zone": zone.value, "intensity": intensity},
            "metrics": {
                "fps": round(1 / average, 1) if average > 0 else 0.0,
                "latency_ms": round(latency_seconds * 1000, 1),
                "tracked_count": len(tracks),
            },
        }
    )


def start_dashboard(state: DashboardState, host: str, port: int) -> Thread:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "Dashboard needs the dashboard extras: pip install -e '.[dashboard]'"
        ) from error
    app = create_app(state)
    thread = Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port},
        daemon=True,
    )
    thread.start()
    LOGGER.info("Dashboard available at http://%s:%d", host, port)
    return thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Third Eye local pipeline")
    parser.add_argument(
        "--config",
        default="config/default.json",
        help="Base JSON configuration path",
    )
    parser.add_argument("--local-config", help="Optional local JSON override (not committed)")
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames per configured mock camera; use 0 for unbounded",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve the optional local dashboard",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_with_override(args.config, args.local_config)
    logging.basicConfig(
        level=getattr(logging, config["application"]["log_level"].upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state = DashboardState()
    if args.dashboard:
        start_dashboard(state, args.host, args.port)
    run_pipeline(
        config,
        max_frames=None if args.frames == 0 else args.frames,
        dashboard_state=state,
    )


if __name__ == "__main__":
    main()
