"""Run the hero demo scenario through the real pipeline with synthetic detections.

No camera, no YOLO: scripted person/backpack boxes are fed into the tracker,
world state, zones, event engine, association engine and dashboard. Use it to
verify event logic end to end and to preview the dashboard.

    python scripts/simulate_scenario.py                    # print events
    python scripts/simulate_scenario.py --video sim.mp4    # also render dashboard
    python scripts/simulate_scenario.py --snapshots out/   # PNGs at key moments
"""

from __future__ import annotations

import argparse
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from thirdeye.core import settings  # noqa: E402
from thirdeye.core.types import Detection  # noqa: E402
from thirdeye.events.event_engine import EventEngine  # noqa: E402
from thirdeye.relationships.object_association import AssociationEngine  # noqa: E402
from thirdeye.spatial.zone_manager import ZoneManager  # noqa: E402
from thirdeye.state.world_state import WorldState  # noqa: E402
from thirdeye.tracking.entity_tracker import EntityTracker  # noqa: E402

FPS = 20
W, H = settings.FRAME_WIDTH, settings.FRAME_HEIGHT


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def person_box(cx, cy=400, h=480):
    w = h * 0.38
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def bag_box(cx, cy=560, s=90):
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def scenario(t: float) -> list[Detection]:
    """Scripted hero demo timeline (seconds)."""
    dets = []
    # Person A: enters left (0-3 s), stands with bag (3-7 s), walks into the
    # restricted zone leaving the bag (7-10 s), dwells there, leaves frame at 27 s.
    if t < 27:
        if t < 3:
            ax = lerp(-50, 300, t / 3)
        elif t < 7:
            ax = 300
        elif t < 10:
            ax = lerp(300, 1050, (t - 7) / 3)
        elif t < 24:
            ax = 1050
        else:
            ax = lerp(1050, 1400, (t - 24) / 3)

        if -40 < ax < W + 40:
            dets.append(Detection("person", 0.88, person_box(ax)))

    # Backpack carried with A until 7 s, then left at x=380.
    bx = ax + 80 if t < 7 else 380
    bag_visible = 1.0 < t < 30.5

    # Person B: enters right at 20 s, walks to the bag, picks it up and carries it left.
    if 20 <= t:
        if t < 25:
            px = lerp(1350, 470, (t - 20) / 5)
        elif t < 26.5:
            px = 470
        else:
            px = lerp(470, 120, (t - 26.5) / 4)

        dets.append(Detection("person", 0.85, person_box(px, h=500)))

        if t >= 26.5:
            bx = px - 90  # bag carried at B's side from where it lay

    if bag_visible and random.random() > 0.12:  # realistic detection flicker
        dets.append(Detection("backpack", random.uniform(0.35, 0.7), bag_box(bx)))

    return dets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=32)
    parser.add_argument("--video", help="render dashboard to this mp4")
    parser.add_argument("--snapshots", help="directory for dashboard PNGs at alert moments")
    args = parser.parse_args()
    random.seed(7)

    render = bool(args.video or args.snapshots)

    if render:
        import cv2
        import numpy as np

        from thirdeye.ui.dashboard import Dashboard, alert_priority

        dashboard = Dashboard()
        writer = None

        if args.snapshots:
            os.makedirs(args.snapshots, exist_ok=True)

    tracker = EntityTracker()
    world = WorldState()
    zones = ZoneManager()
    engine = EventEngine(zones)
    assoc = AssociationEngine()
    start = 1_700_000_000.0
    all_events = []
    snap_times = {5.0, 9.0, 12.0, 17.5, 26.0, 29.0}

    for i in range(int(args.seconds * FPS)):
        t = i / FPS
        now = start + t
        tracks = tracker.update(scenario(t), now)
        world.update(tracks, now)
        transitions = zones.update(world)
        events = engine.update(world, transitions, now) + assoc.update(world, now)
        all_events += events

        for event in events:
            print(f"t={t:5.2f}s  {event['severity']:<7} {event['event_type']:<22} {event['message']}")

        if render:
            frame = np.full((H, W, 3), 70, np.uint8)
            cv2.rectangle(frame, (0, 620), (W, H), (90, 90, 95), -1)
            alerts = engine.active_alerts(world, now) + assoc.active_alerts(world, now)
            alerts.sort(key=alert_priority)
            dashboard.add_events(events)
            canvas = dashboard.render(
                frame, world, zones, assoc.relationships(), alerts, {"fps": 20.0, "infer_ms": 42, "status": "SIM"}
            )

            if args.video:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (canvas.shape[1], canvas.shape[0])
                    )

                writer.write(canvas)

            if args.snapshots and round(t, 2) in snap_times:
                cv2.imwrite(os.path.join(args.snapshots, f"t{t:04.1f}.png"), canvas)

    if render and args.video and writer is not None:
        writer.release()

    types = [e["event_type"] for e in all_events]
    print("\nEvent counts:", {k: types.count(k) for k in sorted(set(types))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
