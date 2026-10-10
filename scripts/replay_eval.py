"""Offline replay evaluation of duplicate/ghost filtering on a recorded clip.

Runs YOLO once over a raw recording and caches the per-frame detections, then
replays the cached detections through tracker variants and reports how many
person/object tracks they create, how often duplicate tracks overlap, and
review contact sheets for every filter suppression / track merge that could
have hit a real, distinct person.

Variants:
  baseline     backups/entity_tracker_baseline.py, no detection filter
  filter_only  thirdeye.perception.detection_filter + baseline tracker
  new          thirdeye.perception.detection_filter + current EntityTracker

Usage:
  python scripts/replay_eval.py --cache-detections        # once, uses the GPU (~100 s)
  python scripts/replay_eval.py                           # all variants + review sheets
  python scripts/replay_eval.py --variants baseline
  python scripts/replay_eval.py --full-pipeline           # also replay events for 'new'

Outputs go to data/replay/ (detection cache, report.txt, review_*.png).
Time per frame defaults to 112.3 s / 1710 frames (the true live rate of
rehearsal.mp4, whose header claims 20 fps).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from thirdeye.core import settings  # noqa: E402
from thirdeye.core.types import PERSON, PORTABLE_OBJECT_CLASSES, Detection  # noqa: E402

DEFAULT_VIDEO = os.path.join(REPO_ROOT, "rehearsal.mp4")
OUT_DIR = os.path.join(REPO_ROOT, "data", "replay")
DEFAULT_DT = 112.3 / 1710
BASELINE_TRACKER = os.path.join(REPO_ROOT, "backups", "entity_tracker_baseline.py")
ALL_VARIANTS = ("baseline", "filter_only", "new")

DUP_CONTAIN = 0.8  # containment for a "duplicate pair" of simultaneous tracks
GHOST_SECONDS = 3.0  # confirmed tracks shorter than this are "short-lived"
AMBIGUOUS_IOU = 0.6  # person suppressions below this IoU rely on containment only
TILE_W, TILE_H = 640, 360


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Intersection / area of the smaller box."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / small if small > 0 else 0.0


def family(class_name: str) -> str:
    return "person" if class_name == PERSON else "object"


# ---------------------------------------------------------------------------
# Detection cache
# ---------------------------------------------------------------------------


def cache_path(video: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(OUT_DIR, f"{stem}_dets.json")


def cache_detections(video: str) -> str:
    import cv2
    from ultralytics import YOLO

    model_path = settings.MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(REPO_ROOT, model_path)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    frames = []
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(
            frame,
            device=settings.YOLO_DEVICE,
            imgsz=settings.YOLO_IMGSZ,
            conf=settings.YOLO_MIN_CONFIDENCE,
            classes=settings.YOLO_CLASS_IDS,
            verbose=False,
        )[0]
        dets = []
        if result.boxes is not None and len(result.boxes):
            for cid, conf, box in zip(
                result.boxes.cls.int().tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
            ):
                dets.append([model.names[cid], round(float(conf), 4), [round(v, 1) for v in box]])
        frames.append(dets)
        if len(frames) % 200 == 0:
            print(f"  {len(frames)} frames, {time.monotonic() - start:.0f} s")
    cap.release()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = cache_path(video)
    with open(path, "w") as fh:
        json.dump(
            {
                "video": os.path.basename(video),
                "model": settings.MODEL_PATH,
                "imgsz": settings.YOLO_IMGSZ,
                "conf": settings.YOLO_MIN_CONFIDENCE,
                "classes": settings.YOLO_CLASS_IDS,
                "frames": frames,
            },
            fh,
        )
    print(f"cached {len(frames)} frames -> {path} ({time.monotonic() - start:.0f} s)")
    return path


def load_cache(video: str) -> list[list[Detection]]:
    path = cache_path(video)
    if not os.path.exists(path):
        raise SystemExit(f"no detection cache at {path}; run with --cache-detections first")
    with open(path) as fh:
        data = json.load(fh)
    return [[Detection(c, float(p), tuple(b)) for c, p, b in frame] for frame in data["frames"]]


# ---------------------------------------------------------------------------
# Variant loading
# ---------------------------------------------------------------------------


def load_baseline_tracker_cls():
    # The baseline counts misses in frames; those settings were replaced by
    # *_SECONDS in the new tracker, so restore the old values for it.
    settings.TRACK_MAX_MISSED_PERSON = 20
    settings.TRACK_MAX_MISSED_OBJECT = 100
    name = "thirdeye_baseline_entity_tracker"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASELINE_TRACKER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses need the module registered
        spec.loader.exec_module(module)
    return sys.modules[name].EntityTracker


def load_filter():
    """Returns filter_detections or raises ImportError with a reason."""
    module = importlib.import_module("thirdeye.perception.detection_filter")
    fn = getattr(module, "filter_detections", None)
    if fn is None:
        raise ImportError("thirdeye.perception.detection_filter has no filter_detections()")
    return fn


def load_new_tracker_cls():
    module = importlib.import_module("thirdeye.tracking.entity_tracker")
    src = inspect.getsource(module)
    for name in ("TRACK_MAX_MISSED_PERSON", "TRACK_MAX_MISSED_OBJECT"):
        if f"settings.{name}\n" in src or f"settings.{name})" in src or f"settings.{name} " in src:
            raise ImportError(f"entity_tracker still reads frame-based settings.{name} (not updated yet)")
    return module.EntityTracker


def call_filter(fn, dets: list[Detection]) -> list[Detection]:
    out = fn(list(dets))
    if isinstance(out, tuple):  # (detections, stats)
        out = out[0]
    return list(out)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class Suppression:
    frame: int
    family: str
    suppressed: Detection
    kept: Detection | None
    iou: float
    contain: float


@dataclass
class VariantRun:
    name: str
    entities: list[list] = field(default_factory=list)  # per frame: list[TrackedEntity]
    suppressions: list[Suppression] = field(default_factory=list)
    merges: list[tuple] = field(default_factory=list)  # (frame, now, kept_id, merged_id)
    filtered: list[list[Detection]] | None = None


def diff_filter(frame: int, before: list[Detection], after: list[Detection]) -> list[Suppression]:
    """Find detections the filter dropped and the kept box that explains each."""
    remaining = list(after)
    dropped = []
    for det in before:
        match = next(
            (k for k in remaining if k is det or (k.class_name == det.class_name and tuple(k.bbox) == tuple(det.bbox))),
            None,
        )
        if match is None:
            # a filter may also relabel or rebox; treat unmatched input as dropped
            dropped.append(det)
        else:
            remaining.remove(match)
    out = []
    for det in dropped:
        fam = family(det.class_name)
        # The filter keeps boxes by descending confidence, so the box that
        # explains a suppression is a more confident survivor. Prefer one that
        # suppresses on IoU alone, else the one with the highest containment.
        partners = [k for k in after if family(k.class_name) == fam]
        stronger = [k for k in partners if k.confidence >= det.confidence] or partners
        best, best_c = None, -1.0
        if stronger:
            def rank(k):
                o, c = iou(det.bbox, k.bbox), containment(det.bbox, k.bbox)
                if fam == "person":  # mirrors the filter's person rule
                    return (o >= AMBIGUOUS_IOU, c >= getattr(settings, "DEDUP_PERSON_CONTAIN", 0.85)
                            and o >= getattr(settings, "DEDUP_PERSON_CONTAIN_MIN_IOU", 0.4), o)
                return (c, o)

            best = max(stronger, key=rank)
            best_c = containment(det.bbox, best.bbox)
        out.append(
            Suppression(
                frame,
                fam,
                det,
                best,
                iou(det.bbox, best.bbox) if best else 0.0,
                best_c if best else 0.0,
            )
        )
    return out


def run_variant(name, frames, dt, start, tracker_cls, filter_fn) -> VariantRun:
    run = VariantRun(name)
    tracker = tracker_cls()
    if filter_fn is not None:
        run.filtered = []
    seen_merges: set[tuple] = set()
    for i, dets in enumerate(frames):
        now = start + i * dt
        used = dets
        if filter_fn is not None:
            used = call_filter(filter_fn, dets)
            run.filtered.append(used)
            run.suppressions.extend(diff_filter(i, dets, used))
        run.entities.append(tracker.update(list(used), now))
        merges = getattr(tracker, "merges", None)
        if merges:
            # the tracker may trim its log, so dedupe by value, not by index
            for m in merges:
                m = tuple(m)
                if m not in seen_merges:
                    seen_merges.add(m)
                    run.merges.append((i, *m[:3]) if len(m) >= 3 else (i, now, *m))
    return run


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class TrackStats:
    track_id: int
    first: int
    last: int
    present: int = 0
    detected: int = 0
    confs: list[float] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    @property
    def fam(self) -> str:
        return family(self.classes[0])

    def flips(self) -> int:
        return sum(1 for a, b in zip(self.classes, self.classes[1:]) if a != b)


def compute_metrics(run: VariantRun, dt: float) -> dict:
    tracks: dict[int, TrackStats] = {}
    persons_per_frame = []
    pair_frames: dict[tuple, int] = defaultdict(int)
    for i, ents in enumerate(run.entities):
        persons_per_frame.append(sum(1 for e in ents if e.class_name == PERSON))
        for e in ents:
            ts = tracks.get(e.track_id)
            if ts is None:
                ts = tracks[e.track_id] = TrackStats(e.track_id, i, i)
            ts.last = i
            ts.present += 1
            ts.classes.append(e.class_name)
            if e.frames_missed == 0:
                ts.detected += 1
                ts.confs.append(e.confidence)
        for a in range(len(ents)):
            for b in range(a + 1, len(ents)):
                ea, eb = ents[a], ents[b]
                if family(ea.class_name) != family(eb.class_name):
                    continue
                if containment(ea.bbox, eb.bbox) >= DUP_CONTAIN:
                    key = (family(ea.class_name), *sorted((ea.track_id, eb.track_id)))
                    pair_frames[key] += 1
    n = max(len(run.entities), 1)
    max_p = max(persons_per_frame) if persons_per_frame else 0
    births = [(ts.first * dt, ts.track_id) for ts in tracks.values() if ts.fam == "person"]
    return {
        "tracks": tracks,
        "n_person": sum(1 for t in tracks.values() if t.fam == "person"),
        "n_object": sum(1 for t in tracks.values() if t.fam == "object"),
        "max_persons": max_p,
        "max_person_frames": [i for i, c in enumerate(persons_per_frame) if c == max_p and max_p > 0],
        "frames_at_max": sum(1 for c in persons_per_frame if c == max_p),
        "mean_persons": sum(persons_per_frame) / n,
        "persons_per_frame": persons_per_frame,
        "dup_pairs": dict(pair_frames),
        "dup_frames_total": sum(pair_frames.values()),
        "ghosts": [t for t in tracks.values() if (t.last - t.first + 1) * dt < GHOST_SECONDS],
        "births": sorted(births),
    }


def flag_far_person(s: Suppression) -> bool:
    """Suppressed box much shorter than the kept one and ending well above its
    bottom: could be a separate, farther person rather than a part-box."""
    if s.family != "person" or s.kept is None:
        return False
    sh = s.suppressed.bbox[3] - s.suppressed.bbox[1]
    kh = s.kept.bbox[3] - s.kept.bbox[1]
    if kh <= 0:
        return False
    return sh < 0.6 * kh and (s.kept.bbox[3] - s.suppressed.bbox[3]) > 0.25 * kh


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_t(frame: int, dt: float) -> str:
    return f"{frame * dt:6.1f}s"


def variant_report(run: VariantRun, m: dict, dt: float, out: list[str]) -> None:
    out.append(f"\n=== {run.name} ===")
    out.append(
        f"person tracks {m['n_person']}, object tracks {m['n_object']}, max simultaneous persons "
        f"{m['max_persons']} ({m['frames_at_max']} frames), mean persons {m['mean_persons']:.2f}"
    )
    out.append("  id  class      first    last   life   det%  medconf  flips")
    for ts in sorted(m["tracks"].values(), key=lambda t: t.track_id):
        life = (ts.last - ts.first + 1) * dt
        med = statistics.median(ts.confs) if ts.confs else 0.0
        classes = "/".join(dict.fromkeys(ts.classes))
        out.append(
            f"  {ts.track_id:>2}  {classes:<9} {fmt_t(ts.first, dt)} {fmt_t(ts.last, dt)} {life:5.1f}s "
            f"{100 * ts.detected / max(ts.present, 1):5.0f}%  {med:6.2f}  {ts.flips():>5}"
        )
    if m["dup_pairs"]:
        out.append(f"  duplicate pairs (containment >= {DUP_CONTAIN}):")
        for (fam, a, b), f in sorted(m["dup_pairs"].items(), key=lambda kv: -kv[1]):
            out.append(f"    {fam:<6} #{a} + #{b}: {f} frames ({f * dt:.1f}s)")
    else:
        out.append("  duplicate pairs: none")
    if m["ghosts"]:
        out.append(
            f"  short-lived (<{GHOST_SECONDS:.0f}s): "
            + ", ".join(f"#{t.track_id} {t.classes[0]} {(t.last - t.first + 1) * dt:.1f}s" for t in m["ghosts"])
        )
    out.append("  person births: " + ", ".join(f"#{tid}@{t:.1f}s" for t, tid in m["births"]))
    if run.filtered is not None:
        per = sum(1 for s in run.suppressions if s.family == "person")
        obj = len(run.suppressions) - per
        amb = sum(1 for s in run.suppressions if s.family == "person" and s.iou < AMBIGUOUS_IOU)
        frames_with = len({s.frame for s in run.suppressions})
        out.append(
            f"  filter suppressions: person {per} (ambiguous IoU<{AMBIGUOUS_IOU}: {amb}), object {obj}; "
            f"{frames_with} frames affected"
        )
        nokeep = [s for s in run.suppressions if s.kept is None]
        if nokeep:
            out.append(f"  WARNING: {len(nokeep)} suppressions have no kept same-family box in that frame")
    if run.name == "new":
        if run.merges:
            out.append("  tracker merges:")
            for frame, now, kept, merged in run.merges:
                out.append(f"    {fmt_t(frame, dt)} kept #{kept} absorbed #{merged}")
        else:
            out.append("  tracker merges: none (or tracker exposes no .merges)")


def table(results: dict, dt: float) -> list[str]:
    rows = [
        ("person tracks", lambda m, r: m["n_person"]),
        ("object tracks", lambda m, r: m["n_object"]),
        ("max simult. persons", lambda m, r: m["max_persons"]),
        ("mean persons", lambda m, r: f"{m['mean_persons']:.2f}"),
        ("dup pairs", lambda m, r: len(m["dup_pairs"])),
        ("dup pair-frames", lambda m, r: m["dup_frames_total"]),
        ("dup seconds", lambda m, r: f"{m['dup_frames_total'] * dt:.1f}"),
        ("tracks < 3 s", lambda m, r: len(m["ghosts"])),
        ("class flips", lambda m, r: sum(t.flips() for t in m["tracks"].values())),
        ("suppr. person/obj", lambda m, r: "-" if r.filtered is None else
            f"{sum(1 for s in r.suppressions if s.family == 'person')}/"
            f"{sum(1 for s in r.suppressions if s.family == 'object')}"),
        ("merges", lambda m, r: len(r.merges) if r.name == "new" else "-"),
    ]
    names = list(results)
    lines = ["", "metric".ljust(22) + "".join(n.rjust(14) for n in names)]
    for label, fn in rows:
        lines.append(label.ljust(22) + "".join(str(fn(results[n][1], results[n][0])).rjust(14) for n in names))
    return lines


# ---------------------------------------------------------------------------
# Review contact sheets
# ---------------------------------------------------------------------------


def read_frames(video: str, wanted: set[int]) -> dict:
    import cv2

    cap = cv2.VideoCapture(video)
    frames = {}
    i = 0
    last = max(wanted) if wanted else -1
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    return frames


def draw_tile(frame, boxes, caption):
    """boxes: list of (bbox, (b,g,r), label). Returns a TILE_W x TILE_H image."""
    import cv2

    img = frame.copy()
    for bbox, color, label in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, label, (x1 + 4, max(y1 + 28, 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
        cv2.putText(img, label, (x1 + 4, max(y1 + 28, 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    img = cv2.resize(img, (TILE_W, TILE_H))
    cv2.rectangle(img, (0, 0), (TILE_W, 26), (0, 0, 0), -1)
    cv2.putText(img, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def save_sheets(tiles, path_prefix, cols=3, rows=3) -> list[str]:
    import cv2
    import numpy as np

    paths = []
    per = cols * rows
    for s in range(0, len(tiles), per):
        chunk = tiles[s : s + per]
        r = math.ceil(len(chunk) / cols)
        c = min(cols, len(chunk))
        sheet = np.zeros((r * TILE_H, c * TILE_W, 3), dtype=np.uint8)
        for k, tile in enumerate(chunk):
            y, x = divmod(k, c)
            sheet[y * TILE_H : (y + 1) * TILE_H, x * TILE_W : (x + 1) * TILE_W] = tile
        path = f"{path_prefix}_{s // per + 1}.png" if len(tiles) > per else f"{path_prefix}.png"
        cv2.imwrite(path, sheet)
        paths.append(path)
    return paths


def spread(items, k):
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(j * step)] for j in range(k)]


def make_reviews(video: str, runs: dict, metrics: dict, dt: float, out: list[str]) -> None:
    GREEN, RED, YELLOW = (0, 220, 0), (0, 0, 255), (0, 220, 255)
    jobs = []  # (prefix, [(frame, boxes, caption)])

    supp_run = runs.get("new") or runs.get("filter_only")
    if supp_run is not None:
        amb = [s for s in supp_run.suppressions if s.family == "person" and s.iou < AMBIGUOUS_IOU]
        picks = spread(amb, 12)
        jobs.append(
            (
                "review_suppress_person",
                [
                    (
                        s.frame,
                        ([(s.kept.bbox, GREEN, f"kept {s.kept.confidence:.2f}")] if s.kept else [])
                        + [(s.suppressed.bbox, RED, f"supp {s.suppressed.confidence:.2f}")],
                        f"f{s.frame} t={s.frame * dt:.1f}s IoU {s.iou:.2f} cont {s.contain:.2f}",
                    )
                    for s in picks
                ],
            )
        )
        out.append(f"\nambiguous person suppressions (IoU<{AMBIGUOUS_IOU}): {len(amb)}, sampled {len(picks)}")
        flagged = [s for s in supp_run.suppressions if flag_far_person(s)]
        out.append(f"auto-flagged possible far-person suppressions: {len(flagged)}")
        for s in flagged[:40]:
            out.append(
                f"  f{s.frame} t={s.frame * dt:.1f}s supp {tuple(round(v) for v in s.suppressed.bbox)} "
                f"conf {s.suppressed.confidence:.2f} kept {tuple(round(v) for v in s.kept.bbox)} "
                f"IoU {s.iou:.2f} cont {s.contain:.2f}"
            )
        if flagged:
            jobs.append(
                (
                    "review_flagged",
                    [
                        (
                            s.frame,
                            [(s.kept.bbox, GREEN, "kept"), (s.suppressed.bbox, RED, "supp")],
                            f"FLAG f{s.frame} t={s.frame * dt:.1f}s IoU {s.iou:.2f} cont {s.contain:.2f}",
                        )
                        for s in spread(flagged, 9)
                    ],
                )
            )
        objs = [s for s in supp_run.suppressions if s.family == "object"]
        if objs:
            jobs.append(
                (
                    "review_suppress_object",
                    [
                        (
                            s.frame,
                            ([(s.kept.bbox, GREEN, f"{s.kept.class_name} {s.kept.confidence:.2f}")] if s.kept else [])
                            + [(s.suppressed.bbox, RED, f"{s.suppressed.class_name} {s.suppressed.confidence:.2f}")],
                            f"f{s.frame} t={s.frame * dt:.1f}s IoU {s.iou:.2f} cont {s.contain:.2f}",
                        )
                        for s in spread(objs, 9)
                    ],
                )
            )

    new = runs.get("new")
    if new is not None:
        if new.merges:
            tiles = []
            for frame, now, kept, merged in new.merges:
                kept_box = next((e.bbox for e in new.entities[frame] if e.track_id == kept), None)
                merged_box, merged_frame = None, None
                for j in range(frame, max(frame - 30, -1), -1):
                    merged_box = next((e for e in new.entities[j] if e.track_id == merged), None)
                    if merged_box is not None:
                        merged_frame = j
                        break
                boxes = []
                if kept_box:
                    boxes.append((kept_box, GREEN, f"kept #{kept}"))
                if merged_box is not None:
                    boxes.append((merged_box.bbox, RED, f"merged #{merged} {merged_box.class_name}"))
                tiles.append((frame, boxes, f"merge f{frame} t={frame * dt:.1f}s #{merged}->#{kept} (red from f{merged_frame})"))
            jobs.append(("review_merges", tiles))
        mframes = metrics["new"]["max_person_frames"]
        # one frame per contiguous episode at max count, then spread
        episodes = [f for k, f in enumerate(mframes) if k == 0 or f != mframes[k - 1] + 1]
        picks = spread(episodes, 9)
        tiles = []
        for f in picks:
            boxes = [
                (e.bbox, GREEN if e.frames_missed == 0 else YELLOW, f"{e.class_name[:6]} #{e.track_id}")
                for e in new.entities[f]
                if e.class_name == PERSON
            ]
            tiles.append((f, boxes, f"max persons={metrics['new']['max_persons']} f{f} t={f * dt:.1f}s (yellow=coasting)"))
        jobs.append(("review_max_persons", tiles))
        out.append(f"max-person episodes for 'new': {len(episodes)} (frames at max: {len(mframes)})")

    wanted = {f for _, tiles in jobs for f, _, _ in tiles}
    if not wanted:
        return
    images = read_frames(video, wanted)
    for prefix, tiles in jobs:
        rendered = [draw_tile(images[f], boxes, cap) for f, boxes, cap in tiles if f in images]
        if rendered:
            paths = save_sheets(rendered, os.path.join(OUT_DIR, prefix))
            out.append(f"wrote {', '.join(os.path.relpath(p, REPO_ROOT) for p in paths)}")


# ---------------------------------------------------------------------------
# Full pipeline replay
# ---------------------------------------------------------------------------


def full_pipeline(frames, filter_fn, tracker_cls, dt, start, out: list[str]) -> None:
    from thirdeye.events.event_engine import EventEngine
    from thirdeye.relationships.object_association import AssociationEngine
    from thirdeye.spatial.zone_manager import ZoneManager
    from thirdeye.state.world_state import WorldState

    memory = None
    try:
        from thirdeye.state import entity_memory

        memory = entity_memory.EntityMemory(store=None, session_id="replay")
    except Exception as exc:  # noqa: BLE001
        out.append(f"(EntityMemory not used: {exc.__class__.__name__}: {exc})")

    tracker = tracker_cls()
    world = WorldState()
    zones = ZoneManager()
    engine = EventEngine(zones)
    assoc = AssociationEngine()
    log = []
    for i, dets in enumerate(frames):
        now = start + i * dt
        used = call_filter(filter_fn, dets) if filter_fn else dets
        tracks = tracker.update(list(used), now)
        world.frame_size = (1280, 720)
        world.update(tracks, now)
        transitions = zones.update(world)
        events = engine.update(world, transitions, now)
        events += assoc.update(world, now)
        if memory is not None:
            try:
                # frame=None: no snapshots written during replay
                events += memory.observe(world, None, assoc, zones, events, now) or []
            except Exception as exc:  # noqa: BLE001
                out.append(f"(EntityMemory.observe failed at f{i}: {exc}); disabling")
                memory = None
        for ev in events:
            log.append((i, ev))
    out.append(f"\n=== full pipeline (new variant): {len(log)} events ===")
    counts = defaultdict(int)
    for i, ev in log:
        counts[ev["event_type"]] += 1
        out.append(f"  {i * dt:6.1f}s {ev['severity'].upper():<7} {ev['event_type']:<22} {ev['message']}")
    out.append("  counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds per frame (default 112.3/1710)")
    ap.add_argument("--cache-detections", action="store_true", help="run YOLO and (re)write the detection cache")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--no-review", action="store_true", help="skip review contact sheets")
    ap.add_argument("--full-pipeline", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.cache_detections:
        cache_detections(args.video)
    frames = load_cache(args.video)
    dt, start = args.dt, 1_000_000.0
    out = [
        f"replay_eval: {os.path.basename(args.video)}, {len(frames)} frames, dt={dt:.5f}s "
        f"({1 / dt:.1f} FPS, {len(frames) * dt:.1f}s)",
        f"raw detections: {sum(len(f) for f in frames)}",
    ]

    filter_fn = filter_err = None
    try:
        filter_fn = load_filter()
    except Exception as exc:  # noqa: BLE001
        filter_err = f"{exc.__class__.__name__}: {exc}"
    new_cls = new_err = None
    try:
        new_cls = load_new_tracker_cls()
    except Exception as exc:  # noqa: BLE001
        new_err = f"{exc.__class__.__name__}: {exc}"

    runs: dict[str, VariantRun] = {}
    metrics: dict[str, dict] = {}
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        try:
            if name == "baseline":
                runs[name] = run_variant(name, frames, dt, start, load_baseline_tracker_cls(), None)
            elif name == "filter_only":
                if filter_fn is None:
                    out.append(f"\n[skip] filter_only: detection filter unavailable ({filter_err})")
                    continue
                runs[name] = run_variant(name, frames, dt, start, load_baseline_tracker_cls(), filter_fn)
            elif name == "new":
                if filter_fn is None or new_cls is None:
                    out.append(f"\n[skip] new: filter={filter_err or 'ok'}; tracker={new_err or 'ok'}")
                    continue
                runs[name] = run_variant(name, frames, dt, start, new_cls, filter_fn)
            else:
                out.append(f"[skip] unknown variant {name}")
                continue
        except Exception:  # noqa: BLE001
            out.append(f"\n[error] variant {name} failed:\n{traceback.format_exc()}")
            continue
        metrics[name] = compute_metrics(runs[name], dt)
        variant_report(runs[name], metrics[name], dt, out)

    if runs:
        out.append("\n=== before / after ===")
        out.extend(table({n: (runs[n], metrics[n]) for n in runs}, dt))

    if not args.no_review and runs:
        try:
            make_reviews(args.video, runs, metrics, dt, out)
        except Exception:  # noqa: BLE001
            out.append(f"[error] review sheets failed:\n{traceback.format_exc()}")

    if args.full_pipeline:
        if new_cls is None or filter_fn is None:
            out.append(f"\n[skip] full pipeline: filter={filter_err or 'ok'}; tracker={new_err or 'ok'}")
        else:
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    full_pipeline(frames, filter_fn, new_cls, dt, start, out)
            except Exception:  # noqa: BLE001
                out.append(f"[error] full pipeline failed:\n{traceback.format_exc()}")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
