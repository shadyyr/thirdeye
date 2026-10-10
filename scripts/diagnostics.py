#!/usr/bin/env python3
"""Third Eye pre-demo diagnostics.

Usage:
    python scripts/diagnostics.py [--skip-camera] [--quick]

--quick        skip YOLO inference and the camera (fast; used by run_demo.sh)
--skip-camera  skip only the camera check (use while the demo is running)

Exit code 0 when every non-skipped check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import tempfile
import threading
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(REPO_ROOT)

OK, FAIL, SKIP, WARN, MISSING = "OK", "FAIL", "SKIP", "WARN", "MISSING"

CAMERA_HINT = (
    "Is another Third Eye / live_yolo.py instance running? "
    "Try: sudo systemctl restart nvargus-daemon"
)


class Reporter:
    def __init__(self) -> None:
        self.use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.failures = 0

    def _color(self, status: str) -> str:
        text = f"{status:<4}"
        if not self.use_color:
            return text
        code = {OK: "32", FAIL: "31", MISSING: "33", WARN: "33", SKIP: "90"}.get(status, "0")
        return f"\033[{code}m{text}\033[0m"

    def line(self, name: str, status: str, detail: str = "") -> None:
        if status == FAIL:
            self.failures += 1
        print(f"  {name:<13} : {self._color(status)} {detail}".rstrip(), flush=True)

    def hint(self, text: str) -> None:
        print(f"  {'':<13}   -> {text}", flush=True)


def run_check(rep: Reporter, name: str, fn) -> object:
    """Run fn() -> (status, detail[, payload]); never let an exception escape."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - diagnostics must keep going
        rep.line(name, FAIL, f"{type(exc).__name__}: {exc}")
        if os.environ.get("THIRDEYE_DIAG_DEBUG"):
            traceback.print_exc()
        return None
    status, detail = result[0], result[1]
    rep.line(name, status, detail)
    return result[2] if len(result) > 2 else None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python():
    version = ".".join(str(v) for v in sys.version_info[:3])
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    where = os.path.basename(sys.prefix) if in_venv else "system python, no venv"
    status = OK if sys.version_info >= (3, 10) else FAIL
    return status, f"{version} ({where})"


def check_opencv():
    import cv2

    return OK, cv2.__version__, cv2


def check_gstreamer(cv2):
    if cv2 is None:
        return FAIL, "OpenCV not importable"
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer:" in line:
            if "YES" in line.upper():
                return OK, "(OpenCV built with GStreamer)"
            return FAIL, "OpenCV built WITHOUT GStreamer (camera pipeline will not open)"
    return FAIL, "GStreamer not listed in cv2.getBuildInformation()"


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        return FAIL, f"torch {torch.__version__}, CUDA not available"
    return OK, f"torch {torch.__version__}", torch


def check_gpu(torch):
    if torch is None:
        return FAIL, "no CUDA"
    name = torch.cuda.get_device_name(0)
    return OK, name


def check_model_file(settings):
    path = settings.MODEL_PATH
    if not os.path.isfile(path):
        return FAIL, f"{path} not found in {REPO_ROOT}"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return OK, f"{path} ({size_mb:.1f} MB)"


def check_yolo(settings):
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(settings.MODEL_PATH)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    kwargs = dict(device=settings.YOLO_DEVICE, imgsz=settings.YOLO_IMGSZ, verbose=False)
    model.predict(frame, **kwargs)  # warm-up
    start = time.perf_counter()
    model.predict(frame, **kwargs)
    ms = (time.perf_counter() - start) * 1000.0
    return OK, f"GPU inference {ms:.0f} ms"


def check_camera(settings, timeout_s: float = 8.0, max_frames: int = 30):
    import cv2

    result: dict = {}

    def worker() -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(settings.CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                result["error"] = "pipeline did not open"
                return
            deadline = time.monotonic() + timeout_s
            frames = 0
            while frames < max_frames and time.monotonic() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames += 1
                    result["shape"] = frame.shape
            result["frames"] = frames
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_s + 4.0)
    if thread.is_alive():
        return FAIL, f"timed out after {timeout_s + 4.0:.0f} s opening camera", "hint"
    if "error" in result:
        return FAIL, result["error"], "hint"
    if not result.get("frames"):
        return FAIL, "opened but no frames received", "hint"
    h, w = result["shape"][:2]
    return OK, f"{w}x{h} frame ({result['frames']} frames read)"


def _primary_class(module, preferred: tuple[str, ...]):
    for name in preferred:
        cls = getattr(module, name, None)
        if inspect.isclass(cls):
            return cls
    own = [
        c for _, c in inspect.getmembers(module, inspect.isclass)
        if c.__module__ == module.__name__ and not c.__name__.startswith("_")
    ]
    return own[0] if own else None


def check_module(module_name: str, preferred: tuple[str, ...], smoke=None):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name) or exc.name == module_name:
            return MISSING, "(not built yet)"
        raise
    cls = _primary_class(module, preferred)
    if cls is None:
        return OK, "imported (no class to smoke-test)"
    kwargs: dict = {}
    try:
        required = [
            p.name for p in inspect.signature(cls).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
    except (TypeError, ValueError):
        required = []
    for name in required:
        factory = _ARG_FACTORIES.get(name)
        if factory is None:
            return OK, f"imported; {cls.__name__}({name}=...) needs args, smoke skipped"
        kwargs[name] = factory()
    instance = cls(**kwargs)
    detail = f"{cls.__name__}({', '.join(kwargs)}) created"
    if smoke is not None:
        detail += "; " + smoke(instance)
    return OK, detail


def _make_zone_manager():
    from thirdeye.spatial.zone_manager import ZoneManager

    return ZoneManager()


# Known constructor dependencies, resolved by parameter name.
_ARG_FACTORIES = {
    "zone_manager": _make_zone_manager,
    "zones": _make_zone_manager,
    "settings": lambda: _settings(),
}


def _tracker_smoke(tracker) -> str:
    out = tracker.update([], 0.0)
    n = len(out) if hasattr(out, "__len__") else "?"
    return f"update([], 0.0) -> {n} entities"


def check_event_store():
    from thirdeye.core.types import make_event
    from thirdeye.storage.event_store import SQLiteEventStore

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(os.path.join(tmp, "diag.sqlite3"), session_id="diag")
        try:
            store.write(make_event("ENTITY_ENTERED", "diagnostics", entity_id=1, metadata={"x": 1}))
            rows = store.recent(5)
            if store.count() != 1 or rows[0]["metadata"] != {"x": 1}:
                return FAIL, "roundtrip mismatch"
        finally:
            store.close()
    db_dir = os.path.dirname(os.path.abspath(_settings().EVENT_DB_PATH))
    parent = db_dir if os.path.isdir(db_dir) else os.path.dirname(db_dir)
    if not os.access(parent, os.W_OK):
        return FAIL, f"{db_dir} not writable"
    return OK, f"sqlite roundtrip ok; db -> {_settings().EVENT_DB_PATH}"


def _settings():
    from thirdeye.core import settings

    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Third Eye diagnostics")
    parser.add_argument("--skip-camera", action="store_true", help="do not open the camera")
    parser.add_argument("--quick", action="store_true", help="skip YOLO inference and camera")
    args = parser.parse_args(argv)

    rep = Reporter()
    print("Third Eye diagnostics", flush=True)

    run_check(rep, "Python", check_python)
    cv2 = run_check(rep, "OpenCV", check_opencv)
    run_check(rep, "GStreamer", lambda: check_gstreamer(cv2))
    torch = run_check(rep, "CUDA", check_cuda)
    run_check(rep, "GPU", lambda: check_gpu(torch))

    settings = run_check(rep, "Settings", lambda: (OK, "thirdeye.core.settings", _settings()))
    if settings is None:
        print("Cannot continue without settings.")
        return 1

    run_check(rep, "Model file", lambda: check_model_file(settings))

    if args.quick:
        rep.line("YOLO", SKIP, "(--quick)")
    else:
        run_check(rep, "YOLO", lambda: check_yolo(settings))

    if args.quick or args.skip_camera:
        rep.line("Camera", SKIP, "(--quick)" if args.quick else "(--skip-camera)")
    else:
        hint = run_check(rep, "Camera", lambda: check_camera(settings))
        if hint == "hint":
            rep.hint(CAMERA_HINT)

    run_check(rep, "Tracker", lambda: check_module(
        "thirdeye.tracking.entity_tracker", ("EntityTracker", "Tracker"), _tracker_smoke))
    run_check(rep, "Zones", lambda: check_module(
        "thirdeye.spatial.zone_manager", ("ZoneManager",)))
    run_check(rep, "Event engine", lambda: check_module(
        "thirdeye.events.event_engine", ("EventEngine",)))
    run_check(rep, "Association", lambda: check_module(
        "thirdeye.relationships.object_association",
        ("ObjectAssociationEngine", "AssociationEngine", "ObjectAssociation", "AssociationManager")))
    run_check(rep, "Event store", check_event_store)

    print(flush=True)
    if rep.failures:
        print(f"{rep.failures} check(s) FAILED", flush=True)
        return 1
    print("All checks passed", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # A camera thread stuck in native code must not keep the process alive.
    os._exit(code)
