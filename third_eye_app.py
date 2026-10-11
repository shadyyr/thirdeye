"""Third Eye live event-intelligence app.

CAMERA -> YOLO -> TRACKING -> WORLD STATE -> ZONES -> EVENTS / RELATIONSHIPS
-> EVENT STORE -> DASHBOARD

Uses the same GStreamer camera pipeline and YOLO26n settings as the
known-good live_yolo.py (which is left untouched as a fallback).

Usage:
    python third_eye_app.py                 # live camera + dashboard window
    python third_eye_app.py --fullscreen
    python third_eye_app.py --record demo.mp4
    python third_eye_app.py --source clip.mp4
    python third_eye_app.py --headless --seconds 30
    python third_eye_app.py --incidents     # also save event-triggered incident clips
Keys: Q quit, F fullscreen, D demo/debug view, 1-4 open a recently seen
object, 0 / X close it.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.chdir(REPO_ROOT)

import cv2  # noqa: E402

from thirdeye.core import settings  # noqa: E402
from thirdeye.core.types import Detection  # noqa: E402
from thirdeye.events.event_engine import EventEngine  # noqa: E402
from thirdeye.perception.detection_filter import filter_detections  # noqa: E402
from thirdeye.relationships.object_association import AssociationEngine  # noqa: E402
from thirdeye.spatial.zone_manager import ZoneManager  # noqa: E402
from thirdeye.state.entity_memory import EntityMemory  # noqa: E402
from thirdeye.state.world_state import WorldState  # noqa: E402
from thirdeye.storage.event_store import SQLiteEventStore  # noqa: E402
from thirdeye.tracking.entity_tracker import EntityTracker  # noqa: E402
from thirdeye.ui.dashboard import Dashboard, alert_priority  # noqa: E402

WINDOW_NAME = "Third Eye"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Third Eye live event intelligence")
    parser.add_argument("--source", help="video file to replay instead of the CSI camera")
    parser.add_argument("--headless", action="store_true", help="no window (benchmark / SSH)")
    parser.add_argument("--seconds", type=float, default=0, help="stop after N seconds")
    parser.add_argument("--record", help="save the dashboard to this .mp4 file")
    parser.add_argument("--record-raw", help="save raw camera frames to this .mp4 (for replay)")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--debug", action="store_true", help="start in debug view (toggle with D)")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=0,
        help="true capture FPS of --source when it has no .times file (rehearsal.mp4: 15.2)",
    )
    parser.add_argument(
        "--zone",
        help="restricted zone rectangle as normalized x1,y1,x2,y2 (e.g. 0.6,0.1,0.95,0.9)",
    )
    parser.add_argument("--no-store", action="store_true", help="do not write events to SQLite")
    parser.add_argument("--quiet", action="store_true", help="do not print events to terminal")
    parser.add_argument(
        "--no-thread",
        action="store_true",
        help="run capture+YOLO in the main loop instead of a worker thread (fallback)",
    )
    parser.add_argument(
        "--incidents",
        action="store_true",
        help="save event-triggered incident clips (5 s before to 12 s after) under data/incidents/",
    )
    return parser.parse_args(argv)


class SafeStage:
    """Run one pipeline stage; on error, log once and keep the video alive."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failures = 0

    def run(self, fn, *args, default=None):
        try:
            return fn(*args)
        except Exception:
            self.failures += 1

            if self.failures == 1 or self.failures % 500 == 0:
                print(f"[ERROR] {self.name} failed ({self.failures}x); continuing:", file=sys.stderr)
                traceback.print_exc()

            return default


def apply_zone_override(spec: str) -> None:
    try:
        x1, y1, x2, y2 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit("[ERROR] --zone must be x1,y1,x2,y2 with values between 0 and 1")

    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise SystemExit("[ERROR] --zone must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")

    for zone in settings.ZONES:
        if zone["type"] == "RESTRICTED":
            zone["points"] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            zone["active"] = True
            break


def load_times(path: str) -> list[float] | None:
    try:
        with open(path) as handle:
            return [float(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return None


def open_source(args):
    if args.source:
        cap = cv2.VideoCapture(args.source)

        if not cap.isOpened():
            raise SystemExit(f"[ERROR] Could not open video file: {args.source}")

        return cap

    cap = cv2.VideoCapture(settings.CAMERA_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise SystemExit(
            "[ERROR] Could not open CSI camera.\n"
            "  - Is live_yolo.py or another Third Eye instance still running?\n"
            "  - Check the ribbon cable, then try: sudo systemctl restart nvargus-daemon"
        )

    return cap


def load_model():
    from ultralytics import YOLO

    if not os.path.exists(settings.MODEL_PATH):
        raise SystemExit(f"[ERROR] Model file not found: {settings.MODEL_PATH}")

    return YOLO(settings.MODEL_PATH)


def detect(model, frame) -> tuple[list[Detection], float]:
    start = time.monotonic()
    result = model.predict(
        frame,
        device=settings.YOLO_DEVICE,
        imgsz=settings.YOLO_IMGSZ,
        conf=settings.YOLO_MIN_CONFIDENCE,
        classes=settings.YOLO_CLASS_IDS,
        verbose=False,
    )[0]
    infer_ms = (time.monotonic() - start) * 1000
    detections = []

    if result.boxes is not None and len(result.boxes):
        classes = result.boxes.cls.int().tolist()
        confidences = result.boxes.conf.tolist()
        boxes = result.boxes.xyxy.tolist()

        for class_id, confidence, box in zip(classes, confidences, boxes):
            detections.append(Detection(model.names[class_id], float(confidence), tuple(box)))

    return detections, infer_ms


class FrameSource:
    """Yields (frame, detections, infer_ms, capture_time) or None at the end.

    Threaded mode runs camera capture + YOLO in a worker thread so inference
    of frame N+1 overlaps tracking/events/rendering of frame N. Only immutable
    detections cross the thread boundary; all state stays on the main thread.
    """

    def __init__(self, cap, model, is_file: bool, threaded: bool) -> None:
        self.cap = cap
        self.model = model
        self.is_file = is_file
        self.threaded = threaded
        self.error: str | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None

        if threaded:
            self._thread = threading.Thread(target=self._worker, name="capture-yolo", daemon=True)
            self._thread.start()

    def _read_and_detect(self):
        failures = 0

        while not self._stop.is_set():
            ok, frame = self.cap.read()

            if ok:
                capture_time = time.time()
                detections, infer_ms = detect(self.model, frame)
                return frame, detections, infer_ms, capture_time

            if self.is_file:
                self.error = "End of video."
                return None

            failures += 1

            if failures > 30:
                self.error = "[ERROR] Camera stopped delivering frames."
                return None

        return None

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                item = self._read_and_detect()
                self._put(item)

                if item is None:
                    return
        except Exception:
            self.error = "[ERROR] Capture/YOLO worker crashed:\n" + traceback.format_exc()
            self._put(None)

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def next(self):
        if not self.threaded:
            return self._read_and_detect()

        while True:
            try:
                return self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self._thread.is_alive():
                    return None

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            # Must finish before cap.release() so the worker never reads a released camera.
            self._thread.join(timeout=3.0)


def start_incident_recorder(store):
    """Rolling pre-event buffer + clip writer threads. Only created with --incidents."""
    from thirdeye.recording import IncidentRecorder

    recorder = IncidentRecorder(store=store, session_id=getattr(store, "session_id", None))
    print(f"Incident recording ON: clips go to {os.path.join(recorder.out_dir, recorder.session_id)}/")

    if store is None:
        print("[WARN] --no-store: incident clips are saved but not linked in SQLite", file=sys.stderr)

    return recorder


def stop_incident_recorder(recorder) -> None:
    """Finish the open clip (bounded wait) and list the clips saved this run."""
    recorder.close()
    clips = recorder.history(limit=1000)
    print(f"Incident clips this run: {len(clips)}")

    for clip in reversed(clips):
        print(f"  {clip['incident_id']} {clip['status']:<9} {clip['duration']:5.1f}s  {clip['file_path']}")


def print_event(event: dict) -> None:
    stamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
    print(f"[{stamp}] {event['severity'].upper():<7} {event['event_type']:<22} {event['message']}")


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Always run cleanup (camera release, DB close) on Ctrl+C or SIGTERM,
    # even when launched from a non-interactive shell.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    if args.zone:
        apply_zone_override(args.zone)

    cap = open_source(args)
    model = load_model()

    # Warm up CUDA so the first live frames are not seconds late.
    ok, warm_frame = cap.read()

    if ok:
        for _ in range(3):
            detect(model, warm_frame)

    tracker = EntityTracker()
    world = WorldState()
    zones = ZoneManager()
    event_engine = EventEngine(zones)
    associations = AssociationEngine()
    dashboard = Dashboard(mode="debug" if args.debug else settings.DASHBOARD_DEFAULT_MODE)
    store = None if args.no_store else SQLiteEventStore(settings.EVENT_DB_PATH)
    memory = EntityMemory(store=store, session_id=getattr(store, "session_id", None))
    incidents = start_incident_recorder(store) if args.incidents else None

    stages = {
        name: SafeStage(name)
        for name in ("filter", "tracking", "zones", "events", "relationships", "memory", "store", "dashboard")
    }

    if incidents is not None:
        stages["incidents"] = SafeStage("incidents")

    # Replay timing: exact capture times from a .times file written by
    # --record-raw, else --source-fps, else the file header.
    video_fps = 0.0
    source_times = None

    if args.source:
        source_times = load_times(args.source + ".times")
        video_fps = args.source_fps or cap.get(cv2.CAP_PROP_FPS)

    replay_start = time.time()
    recorder = None
    raw_recorder = None
    raw_times = None
    filter_suppressed = 0

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, settings.DASHBOARD_WIDTH, settings.DASHBOARD_HEIGHT)

        if args.fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Third Eye started. All perception runs locally on this Jetson.")
    print("Press Q in the window (or Ctrl+C here) to quit.")

    frame_index = 0
    fps = 0.0
    infer_ms = 0.0
    last_tick = time.monotonic()
    run_start = time.monotonic()
    totals = {"infer": 0.0, "pipeline": 0.0, "render": 0.0}

    source = FrameSource(cap, model, is_file=bool(args.source), threaded=not args.no_thread)

    try:
        while True:
            item = source.next()

            if item is None:
                if source.error:
                    print(source.error, file=sys.stderr)

                break

            frame, detections, step_ms, capture_time = item
            frame_index += 1
            tick = time.monotonic()
            dt = tick - last_tick
            last_tick = tick

            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            # Wall-clock timestamps; for replays, derive time from the frame index
            # so timers (dwell, unattended) behave like real time.
            if source_times and frame_index < len(source_times):
                now = replay_start + source_times[frame_index] - source_times[0]
            elif args.source and video_fps > 0:
                now = replay_start + frame_index / video_fps
            else:
                now = capture_time

            if raw_recorder is None and args.record_raw:
                raw_recorder = cv2.VideoWriter(
                    args.record_raw, cv2.VideoWriter_fourcc(*"mp4v"), 20, (frame.shape[1], frame.shape[0])
                )
                raw_times = open(args.record_raw + ".times", "w")

            if raw_recorder is not None:
                raw_recorder.write(frame)
                raw_times.write(f"{capture_time:.4f}\n")

            infer_ms = 0.9 * infer_ms + 0.1 * step_ms if infer_ms else step_ms
            pipeline_start = time.monotonic()

            world.frame_size = (frame.shape[1], frame.shape[0])
            raw_count = len(detections)
            detections = stages["filter"].run(filter_detections, detections, default=detections)
            filter_suppressed += raw_count - len(detections)
            tracks = stages["tracking"].run(tracker.update, detections, now, default=[])
            world.update(tracks, now)
            transitions = stages["zones"].run(zones.update, world, default=[])
            events = stages["events"].run(event_engine.update, world, transitions, now, default=[])
            events += stages["relationships"].run(associations.update, world, now, default=[])
            # Before rendering: memory may crop last-seen snapshots from the clean frame.
            events += stages["memory"].run(
                memory.observe, world, frame, associations, zones, events, now, default=[]
            )

            if events:
                dashboard.add_events(events)

                if not args.quiet:
                    for event in events:
                        print_event(event)

                if store is not None:
                    stages["store"].run(store.write_many, events)

            if incidents is not None:
                # The clean camera frame (the dashboard draws on its own canvas).
                stages["incidents"].run(incidents.push_frame, frame, now)
                stages["incidents"].run(incidents.on_events, events, now)

            alerts = stages["events"].run(event_engine.active_alerts, world, now, default=[])
            alerts += stages["relationships"].run(associations.active_alerts, world, now, default=[])
            alerts.sort(key=alert_priority)
            render_start = time.monotonic()
            totals["infer"] += step_ms
            totals["pipeline"] += (render_start - pipeline_start) * 1000

            if args.headless and not args.record:
                totals["render"] += 0.0
            else:
                stats = {
                    "fps": fps,
                    "infer_ms": infer_ms,
                    "status": "REPLAY" if args.source else "LIVE",
                    "occupancy_total": event_engine.occupancy_total,
                    "crowded": bool(event_engine.crowded_zones),
                    "zone_occupancy": stages["events"].run(event_engine.zone_occupancy, world, default={}),
                    "filter_suppressed": filter_suppressed,
                    "track_merges": len(getattr(tracker, "merges", [])),
                }
                relationships = stages["relationships"].run(associations.relationships, default=[])
                canvas = stages["dashboard"].run(
                    dashboard.render, frame, world, zones, relationships, alerts, stats, memory
                )
                totals["render"] += (time.monotonic() - render_start) * 1000

                if canvas is not None:
                    if args.record:
                        if recorder is None:
                            recorder = cv2.VideoWriter(
                                args.record,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                20,
                                (canvas.shape[1], canvas.shape[0]),
                            )

                        recorder.write(canvas)

                    if not args.headless:
                        cv2.imshow(WINDOW_NAME, canvas)

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key != 255:
                    stages["dashboard"].run(dashboard.handle_key, key)

                if key == ord("f"):
                    full = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                    mode = cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

            if args.seconds and time.monotonic() - run_start >= args.seconds:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        source.stop()
        elapsed = time.monotonic() - run_start

        if frame_index:
            print(
                f"Frames: {frame_index}  avg FPS: {frame_index / elapsed:.1f}  "
                f"YOLO: {totals['infer'] / frame_index:.1f} ms  "
                f"pipeline: {totals['pipeline'] / frame_index:.2f} ms  "
                f"render: {totals['render'] / frame_index:.1f} ms"
            )

        cap.release()

        if recorder is not None:
            recorder.release()

        if raw_recorder is not None:
            raw_recorder.release()
            raw_times.close()

        stages["memory"].run(memory.close)

        if incidents is not None:
            # Before store.close(): the clip writer records the final clip status there.
            SafeStage("incidents").run(stop_incident_recorder, incidents)

        if store is not None:
            store.close()

        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
