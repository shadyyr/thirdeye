"""Duplicate / ghost detection suppression (runs before the tracker).

YOLO26n is NMS-free and we run it with a low confidence floor, so it often
emits a second box on the same thing:
  - a nested upper-body box sharing the top edge of a full person box;
  - a wide "person + carried bag" box around a tighter person box;
  - one bag labelled backpack AND handbag AND suitcase at the same time.

filter_detections() keeps boxes greedily by descending confidence and drops a
box when an already kept box of the same *family* overlaps it:
  - person:   IoU >= DEDUP_PERSON_IOU, or
              containment >= DEDUP_PERSON_CONTAIN and IoU >= DEDUP_PERSON_CONTAIN_MIN_IOU
              (a far, much smaller person inside a near person's box has a low
              IoU and is kept);
  - portable: containment >= DEDUP_OBJECT_CONTAIN (backpack/handbag/suitcase
              are one family; the survivor keeps its own label and the tracker
              votes on the class over time).
Different families never suppress each other (a bag inside a person box is
kept). Containment = intersection / area of the smaller box.

Stats: every call stores a DetectionFilterStats in the module-level
``last_stats`` (counts for the most recent call). Pass your own
``stats=DetectionFilterStats()`` to accumulate across calls instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from thirdeye.core import settings
from thirdeye.core.types import PERSON, PORTABLE_OBJECT_CLASSES, BBox, Detection

PORTABLE_FAMILY = "portable"


def family(class_name: str) -> str:
    """'portable' for backpack/handbag/suitcase, otherwise the class itself."""
    return PORTABLE_FAMILY if class_name in PORTABLE_OBJECT_CLASSES else class_name


def box_overlap(a: BBox, b: BBox) -> tuple[float, float]:
    """Return (iou, containment) where containment = intersection / smaller area."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0, 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    iou = inter / union if union > 0 else 0.0
    contain = inter / smaller if smaller > 0 else 0.0
    return iou, contain


@dataclass
class DetectionFilterStats:
    """Counts of input / suppressed boxes (for debug display)."""

    total_in: int = 0
    suppressed_person: int = 0
    suppressed_object: int = 0

    @property
    def suppressed(self) -> int:
        return self.suppressed_person + self.suppressed_object


# Stats of the most recent filter_detections() call.
last_stats = DetectionFilterStats()


def _is_duplicate(fam: str, kept: BBox, cand: BBox) -> bool:
    iou, contain = box_overlap(kept, cand)
    if fam == PERSON:
        if iou >= settings.DEDUP_PERSON_IOU:
            return True
        return (contain >= settings.DEDUP_PERSON_CONTAIN
                and iou >= settings.DEDUP_PERSON_CONTAIN_MIN_IOU)
    if fam == PORTABLE_FAMILY:
        return contain >= settings.DEDUP_OBJECT_CONTAIN
    return False  # other classes: no suppression


def filter_detections(
    detections: list[Detection], stats: DetectionFilterStats | None = None
) -> list[Detection]:
    """Drop duplicate boxes of the same family; keep input order of survivors."""
    global last_stats
    call_stats = DetectionFilterStats(total_in=len(detections))
    last_stats = call_stats
    if stats is not None:
        stats.total_in += len(detections)
    if not settings.DEDUP_ENABLED or len(detections) < 2:
        return detections

    order = sorted(range(len(detections)), key=lambda i: -detections[i].confidence)
    kept: list[tuple[str, BBox]] = []
    keep_idx: set[int] = set()
    for i in order:
        det = detections[i]
        fam = family(det.class_name)
        if any(kf == fam and _is_duplicate(fam, kb, det.bbox) for kf, kb in kept):
            if fam == PERSON:
                call_stats.suppressed_person += 1
            else:
                call_stats.suppressed_object += 1
            continue
        kept.append((fam, det.bbox))
        keep_idx.add(i)

    if stats is not None:
        stats.suppressed_person += call_stats.suppressed_person
        stats.suppressed_object += call_stats.suppressed_object
    return [d for i, d in enumerate(detections) if i in keep_idx]
