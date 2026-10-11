"""Benchmark the incident recorder against the live camera + YOLO pipeline.

Runs the real CSI camera, the capture+YOLO worker thread from
third_eye_app.FrameSource and the full main-thread pipeline (filter, tracker,
world, zones, events, associations, entity memory, dashboard render without a
window) for N seconds per mode:

  none     no recorder
  buffer   recorder buffering only (no triggers)
  record   recorder with a forced trigger every --trigger-every seconds, so a
           clip is being written almost all the time (worst case)

Extra 'record' variants test other clip settings: --variants 960x540:x264,1280x720:mp4v

Wrap in the GPU/camera lock:
  flock <lock> .venv/bin/python scripts/bench_incidents.py --seconds 25
Falls back to rehearsal.mp4 (with real YOLO) if the camera cannot be opened.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import third_eye_app as app  # noqa: E402  (chdirs to the repo, adds src/ to sys.path)
import cv2  # noqa: E402

from thirdeye.core import settings  # noqa: E402
from thirdeye.core.types import DWELLING, make_event  # noqa: E402
from thirdeye.events.event_engine import EventEngine  # noqa: E402
from thirdeye.perception.detection_filter import filter_detections  # noqa: E402
from thirdeye.recording.incident_recorder import IncidentRecorder  # noqa: E402
from thirdeye.relationships.object_association import AssociationEngine  # noqa: E402
from thirdeye.spatial.zone_manager import ZoneManager  # noqa: E402
from thirdeye.state.entity_memory import EntityMemory  # noqa: E402
from thirdeye.state.world_state import WorldState  # noqa: E402
from thirdeye.storage.event_store import SQLiteEventStore  # noqa: E402
from thirdeye.tracking.entity_tracker import EntityTracker  # noqa: E402
from thirdeye.ui.dashboard import Dashboard, alert_priority  # noqa: E402


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def cpu_s() -> float:
    t = os.times()
    return t.user + t.system


def open_capture(use_file: bool):
    if not use_file:
        cap = cv2.VideoCapture(settings.CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, False
        cap.release()
        print("[bench] camera could not be opened; falling back to rehearsal.mp4 with real YOLO")
    cap = cv2.VideoCapture(os.path.join(REPO_ROOT, "rehearsal.mp4"))
    if not cap.isOpened():
        raise SystemExit("could not open camera or rehearsal.mp4")
    return cap, True


def run_mode(name, cap, model, is_file, seconds, recorder, trigger_every, render) -> dict:
    tracker, world, zones = EntityTracker(), WorldState(), ZoneManager()
    engine, assoc = EventEngine(zones), AssociationEngine()
    memory = EntityMemory(store=None, session_id="bench")
    dashboard = Dashboard(mode=settings.DASHBOARD_DEFAULT_MODE) if render else None
    source = app.FrameSource(cap, model, is_file=is_file, threaded=True)
    rec_ms, infer_ms, render_ms = [], [], []
    frames, next_trigger = 0, None
    rss0, cpu0, wall0 = rss_mb(), cpu_s(), time.monotonic()
    try:
        while time.monotonic() - wall0 < seconds:
            item = source.next()
            if item is None:
                print(f"[bench] {name}: source ended early ({source.error})")
                break
            frame, dets, step_ms, now = item
            frames += 1
            infer_ms.append(step_ms)
            world.frame_size = (frame.shape[1], frame.shape[0])
            dets = filter_detections(dets)
            tracks = tracker.update(dets, now)
            world.update(tracks, now)
            transitions = zones.update(world)
            events = engine.update(world, transitions, now)
            events += assoc.update(world, now)
            events += memory.observe(world, None, assoc, zones, events, now) or []
            if recorder is not None:
                if trigger_every and (next_trigger is None or now >= next_trigger):
                    next_trigger = now + trigger_every
                    events.append(make_event(DWELLING, "bench trigger", entity_id=0, zone_id="restricted_a",
                                             timestamp=now, metadata={"zone_type": "RESTRICTED"}))
                t = time.perf_counter()
                recorder.push_frame(frame, now)
                recorder.on_events(events, now)
                rec_ms.append((time.perf_counter() - t) * 1000)
            if dashboard is not None:
                t = time.perf_counter()
                alerts = engine.active_alerts(world, now) + assoc.active_alerts(world, now)
                alerts.sort(key=alert_priority)
                stats = {"fps": 0.0, "infer_ms": step_ms, "status": "LIVE", "occupancy_total": engine.occupancy_total,
                         "crowded": False, "zone_occupancy": engine.zone_occupancy(world),
                         "filter_suppressed": 0, "track_merges": 0}
                dashboard.render(frame, world, zones, assoc.relationships(), alerts, stats, memory)
                render_ms.append((time.perf_counter() - t) * 1000)
    finally:
        source.stop()
    wall = time.monotonic() - wall0
    result = {
        "mode": name,
        "frames": frames,
        "fps": frames / wall if wall else 0.0,
        "cpu_cores": (cpu_s() - cpu0) / wall if wall else 0.0,
        "yolo_ms": statistics.mean(infer_ms) if infer_ms else 0.0,
        "render_ms": statistics.mean(render_ms) if render_ms else 0.0,
        "rss_start_mb": rss0,
        "rss_end_mb": rss_mb(),
    }
    if rec_ms:
        rec_sorted = sorted(rec_ms)
        result["rec_ms_mean"] = statistics.mean(rec_ms)
        result["rec_ms_p99"] = rec_sorted[int(len(rec_sorted) * 0.99) - 1]
        result["rec_ms_max"] = rec_sorted[-1]
    if recorder is not None:
        result["buffer"] = recorder.buffer_stats()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=25.0, help="seconds per mode")
    ap.add_argument("--modes", default="none,buffer,record")
    ap.add_argument("--variants", default="", help="extra record runs, e.g. 960x540:x264,1280x720:mp4v")
    ap.add_argument("--size", default="1280x720", help="clip size for buffer/record modes")
    ap.add_argument("--codec", default="x264")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--trigger-every", type=float, default=4.0)
    ap.add_argument("--no-render", action="store_true", help="skip the dashboard render (headless app)")
    ap.add_argument("--file", action="store_true", help="use rehearsal.mp4 instead of the camera")
    ap.add_argument("--keep", action="store_true", help="keep the benchmark clips")
    args = ap.parse_args()

    cap, is_file = open_capture(args.file)
    model = app.load_model()
    ok, warm = cap.read()
    if ok:
        for _ in range(3):
            app.detect(model, warm)
    session = "bench-" + time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(REPO_ROOT, "data", "incidents")
    store = SQLiteEventStore(os.path.join(base, session, "incidents.sqlite3"), session_id=session)

    runs = [(m, args.size, args.codec) for m in args.modes.split(",") if m]
    for spec in filter(None, args.variants.split(",")):
        size, codec = spec.split(":")
        runs.append(("record", size, codec))

    print(f"source: {'rehearsal.mp4 (file)' if is_file else 'CSI camera'}, {args.seconds:.0f} s per mode, "
          f"render={'off' if args.no_render else 'on'}, quality={args.quality}")
    results = []
    for mode, size, codec in runs:
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        recorder = None
        label = mode if mode == "none" else f"{mode} {size} {codec}"
        if mode != "none":
            settings.INCIDENT_CODECS = (codec,)
            w, h = (int(v) for v in size.split("x"))
            recorder = IncidentRecorder(store=store, session_id=f"{session}", out_dir=base, size=(w, h),
                                        jpeg_quality=args.quality, min_free_mb=200)
        result = run_mode(label, cap, model, is_file, args.seconds, recorder,
                          args.trigger_every if mode == "record" else 0, not args.no_render)
        if recorder is not None:
            t = time.monotonic()
            hist = recorder.history(100)
            recorder.close()
            result["close_s"] = time.monotonic() - t
            result["recorder"] = dict(recorder.stats)
            result["clips"] = [(r["incident_id"]) for r in hist]
        results.append(result)
        print(f"  {label}: {result['frames']} frames, {result['fps']:.1f} FPS, cpu {result['cpu_cores']:.2f} cores")
    cap.release()

    rows = store.incidents(session_id=session, limit=500)
    store.close()
    print("\n=== results ===")
    print(f"{'mode':<24} {'FPS':>5} {'cores':>5} {'yolo':>5} {'render':>6} {'rec ms mean/p99/max':>20} "
          f"{'buf MB':>6} {'buf fr':>6} {'RSS d':>6} {'drops':>5}")
    for r in results:
        buf = r.get("buffer") or {}
        rec = f"{r['rec_ms_mean']:.3f}/{r['rec_ms_p99']:.2f}/{r['rec_ms_max']:.1f}" if "rec_ms_mean" in r else "-"
        drops = (buf.get("dropped", 0) + (r.get("recorder") or {}).get("dropped_frames", 0)) if buf else "-"
        print(f"{r['mode']:<24} {r['fps']:5.1f} {r['cpu_cores']:5.2f} {r['yolo_ms']:5.1f} {r['render_ms']:6.1f} "
              f"{rec:>20} {buf.get('bytes', 0) / 1e6:6.2f} {buf.get('frames', 0):6} "
              f"{r['rss_end_mb'] - r['rss_start_mb']:6.1f} {drops!s:>5}")
    print("\n=== clips (MB per 20 s = bytes / clip seconds * 20) ===")
    by_cfg: dict[str, list] = {}
    for row in rows:
        md = row["metadata"]
        length = (row["clip_end"] or 0) - (row["clip_start"] or 0)
        key = f"{md.get('codec')} {'x'.join(str(v) for v in md.get('size') or [])}"
        if row["status"] == "complete" and length > 5:
            by_cfg.setdefault(key, []).append(row["bytes"] / length * 20 / 1e6)
        print(f"  {row['incident_id']} {key:<14} {row['status']:<9} {length:5.1f}s {row['frame_count']:4} fr "
              f"{row['bytes'] / 1e6:6.2f} MB  repeated={md.get('repeated_frames')} dropped={md.get('dropped_frames')}")
    for key, values in by_cfg.items():
        print(f"  {key}: {statistics.mean(values):.2f} MB per 20 s (n={len(values)})")
    for r in results:
        if "close_s" in r:
            print(f"  close() {r['mode']}: {r['close_s']:.2f} s")
    clip_dir = os.path.join(base, session)
    if args.keep:
        print(f"clips kept in {clip_dir}")
    else:
        shutil.rmtree(clip_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
