"""Replay a recorded clip through the event pipeline and record incident clips.

Decodes the video file (no camera), uses the cached YOLO detections from
scripts/replay_eval.py (no GPU), runs filter -> tracker -> world -> zones ->
events -> associations -> entity memory exactly like third_eye_app.py, and
feeds every real frame plus that frame's events to IncidentRecorder.

Usage:
  python scripts/replay_incidents.py                      # rehearsal.mp4
  python scripts/replay_incidents.py --frames-dir DIR     # also save before/after PNGs per incident
  python scripts/replay_incidents.py --size 960x540 --quality 70 --codecs mp4v

Output: data/incidents/replay-<time>/ (clips + incidents.sqlite3), then the
incident list on stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import cv2  # noqa: E402

from replay_eval import DEFAULT_DT, DEFAULT_VIDEO, load_cache  # noqa: E402
from thirdeye.core import settings  # noqa: E402
from thirdeye.events.event_engine import EventEngine  # noqa: E402
from thirdeye.perception.detection_filter import filter_detections  # noqa: E402
from thirdeye.recording import incident_recorder  # noqa: E402
from thirdeye.recording.incident_recorder import IncidentRecorder  # noqa: E402
from thirdeye.relationships.object_association import AssociationEngine  # noqa: E402
from thirdeye.spatial.zone_manager import ZoneManager  # noqa: E402
from thirdeye.state.entity_memory import EntityMemory  # noqa: E402
from thirdeye.state.world_state import WorldState  # noqa: E402
from thirdeye.storage.event_store import SQLiteEventStore  # noqa: E402
from thirdeye.tracking.entity_tracker import EntityTracker  # noqa: E402


def parse_size(text: str | None):
    if not text or text == "native":
        return None
    w, h = text.lower().split("x")
    return int(w), int(h)


def save_before_after(row: dict, out_dir: str) -> str | None:
    """Side-by-side PNG: clip frame 3 s before and 3 s after the trigger."""
    cap = cv2.VideoCapture(row["file_path"])
    fps = float(row["metadata"].get("fps") or 10.0)
    offset = row["trigger_time"] - row["clip_start"]
    wanted = {max(0, int((offset - 3.0) * fps)): "before", int((offset + 3.0) * fps): "after"}
    tiles, index = {}, 0
    while len(tiles) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            label = f"{wanted[index]}: t={index / fps - offset:+.1f}s from trigger ({row['event_type']})"
            frame = frame.copy()
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles[wanted[index]] = frame
        index += 1
    cap.release()
    if len(tiles) < 2:
        return None
    sheet = cv2.hconcat([tiles["before"], tiles["after"]])
    sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
    path = os.path.join(out_dir, f"{row['incident_id']}_before_after.png")
    cv2.imwrite(path, sheet)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds per frame (default 112.3/1710 = 15.2 fps)")
    ap.add_argument("--size", default="native", help="clip size WxH or 'native' (default)")
    ap.add_argument("--quality", type=int, default=None, help="buffer JPEG quality")
    ap.add_argument("--codecs", default=None, help="comma list, e.g. x264,mp4v")
    ap.add_argument("--out", default=None, help="output directory (default data/incidents/replay-<time>)")
    ap.add_argument("--frames-dir", default=None, help="write before/after PNGs per incident here")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if args.codecs:
        settings.INCIDENT_CODECS = tuple(c.strip() for c in args.codecs.split(",") if c.strip())
    frames_dets = load_cache(args.video)
    session = "replay-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    base = args.out or os.path.join(REPO_ROOT, "data", "incidents")
    store = SQLiteEventStore(os.path.join(base, session, "incidents.sqlite3"), session_id=session)
    recorder = IncidentRecorder(store=store, session_id=session, out_dir=base, size=parse_size(args.size),
                                jpeg_quality=args.quality, realtime=False,
                                close_timeout=120.0)  # offline: let the writer finish its backlog

    tracker = EntityTracker()
    world = WorldState()
    zones = ZoneManager()
    engine = EventEngine(zones)
    assoc = AssociationEngine()
    memory = EntityMemory(store=None, session_id=session)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    start = time.time()
    wall = time.monotonic()
    n = len(frames_dets) if not args.max_frames else min(args.max_frames, len(frames_dets))
    all_events = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        now = start + i * args.dt
        world.frame_size = (frame.shape[1], frame.shape[0])
        dets = filter_detections(list(frames_dets[i]))
        if isinstance(dets, tuple):
            dets = dets[0]
        tracks = tracker.update(list(dets), now)
        world.update(tracks, now)
        transitions = zones.update(world)
        events = engine.update(world, transitions, now)
        events += assoc.update(world, now)
        events += memory.observe(world, None, assoc, zones, events, now) or []  # no snapshots in replay
        all_events += len(events)
        recorder.push_frame(frame, now)
        recorder.on_events(events, now)
    cap.release()
    t0 = time.monotonic()
    recorder.close()
    close_s = time.monotonic() - t0
    elapsed = time.monotonic() - wall
    rows = sorted(store.incidents(session_id=session, limit=100), key=lambda r: r["trigger_time"])
    store.close()

    print(f"{os.path.basename(args.video)}: {n} frames ({n * args.dt:.1f} s replay time) in {elapsed:.1f} s wall, "
          f"{all_events} events, {len(rows)} incidents, close() {close_s:.2f} s")
    print(f"buffer: {recorder.buffer_stats()}")
    print(f"recorder: {recorder.stats}")
    print(f"clips + DB: {os.path.join(base, session)}")
    print()
    print(f"{'trigger@':>8}  {'incident':<16} {'type':<18} {'status':<9} {'clip':>6} {'frames':>6} "
          f"{'size':>7} {'codec':<5} linked events")
    for r in rows:
        md = r["metadata"]
        rel = r["trigger_time"] - start
        length = (r["clip_end"] or 0) - (r["clip_start"] or 0)
        linked = ", ".join(md.get("event_types") or [])
        print(f"{rel:7.1f}s  {r['incident_id']:<16} {r['event_type']:<18} {r['status']:<9} {length:5.1f}s "
              f"{r['frame_count']:>6} {r['bytes'] / 1e6:6.2f}M {str(md.get('codec')):<5} "
              f"[{len(r['event_ids'])}] {linked}  entities={r['entity_ids']}")
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        for r in rows:
            if r["status"] == incident_recorder.COMPLETE:
                path = save_before_after(r, args.frames_dir)
                print(f"  {r['incident_id']}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
