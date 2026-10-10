"""Small OpenCV drawing helpers shared by the demo and debug dashboards.

Hershey fonts only, so every string must be ASCII. Icons are drawn as shapes.
Text measurement and truncation results are cached: getTextSize in a loop is
one of the larger per-frame costs on the Jetson CPU.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette (BGR). High contrast for a TV a few meters away.
# ---------------------------------------------------------------------------
BG = (22, 18, 16)
HEADER_BG = (14, 12, 10)
PANEL = (44, 38, 34)
PANEL_HI = (58, 51, 46)
PANEL_EDGE = (78, 70, 64)
TEXT = (245, 245, 245)
SUBTLE = (200, 198, 196)
MUTED = (150, 146, 142)
DIM = (105, 100, 96)
GREEN = (110, 210, 60)
RED = (70, 70, 240)
AMBER = (0, 175, 255)
BLUE = (255, 185, 70)
CYAN = (220, 210, 60)
PERSON_COLOR = (255, 190, 70)
OBJECT_COLOR = (0, 190, 255)
RESTRICTED_COLOR = (70, 70, 235)
MONITORED_COLOR = (200, 150, 70)
WHITE = (255, 255, 255)
INK = (18, 18, 18)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

_size_cache: dict = {}
_fit_cache: dict = {}
_CACHE_LIMIT = 4000


def text(img, s, org, scale=0.8, color=TEXT, thickness=1, font=FONT) -> None:
    cv2.putText(img, s, (int(org[0]), int(org[1])), font, scale, color, thickness, AA)


def text_size(s, scale, thickness=1, font=FONT) -> tuple[int, int]:
    key = (s, scale, thickness, font)
    size = _size_cache.get(key)

    if size is None:
        if len(_size_cache) > _CACHE_LIMIT:
            _size_cache.clear()

        size = cv2.getTextSize(s, font, scale, thickness)[0]
        _size_cache[key] = size

    return size


def text_width(s, scale, thickness=1, font=FONT) -> int:
    return text_size(s, scale, thickness, font)[0]


def fit(s: str, max_width: int, scale: float, thickness: int = 1, font=FONT) -> str:
    """Truncate with '...' so the rendered string fits in max_width pixels."""
    key = (s, max_width, scale, thickness, font)
    out = _fit_cache.get(key)

    if out is not None:
        return out

    if len(_fit_cache) > _CACHE_LIMIT:
        _fit_cache.clear()

    if text_width(s, scale, thickness, font) <= max_width:
        out = s
    else:
        lo, hi = 0, len(s)

        # Binary search the longest prefix that fits with an ellipsis.
        while lo < hi:
            mid = (lo + hi + 1) // 2

            if text_width(s[:mid].rstrip() + "...", scale, thickness, font) <= max_width:
                lo = mid
            else:
                hi = mid - 1

        out = s[:lo].rstrip() + "..." if lo else ""

    _fit_cache[key] = out
    return out


def wrap(s: str, max_width: int, scale: float, thickness: int = 1, font=FONT, max_lines: int = 2) -> list[str]:
    """Greedy word wrap; the last allowed line is truncated with '...'."""
    key = ("wrap", s, max_width, scale, thickness, font, max_lines)
    out = _fit_cache.get(key)

    if out is not None:
        return out

    words = s.split()
    lines: list[str] = []
    current = ""

    for i, word in enumerate(words):
        candidate = f"{current} {word}".strip()

        if not current or text_width(candidate, scale, thickness, font) <= max_width:
            current = candidate
            continue

        lines.append(current)
        current = word

        if len(lines) == max_lines - 1:
            current = " ".join(words[i:])
            break

    if current:
        lines.append(current)

    lines = [fit(line, max_width, scale, thickness, font) for line in lines[:max_lines]]
    _fit_cache[key] = lines
    return lines


def fill_rect(img, p1, p2, color) -> None:
    cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, -1)


def blend_rect(img, p1, p2, color, alpha: float) -> None:
    """Translucent rectangle, blended only inside its own ROI."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(w, int(p2[0])), min(h, int(p2[1]))

    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, dst=roi)


def rounded_rect(img, p1, p2, color, radius=10, thickness=-1) -> None:
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)

        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        t = thickness
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, AA)


def pill(img, s, x, y, color, scale=0.6, text_color=INK, thickness=1, pad_x=12, height=None, align="left") -> int:
    """Filled rounded label. (x, y) is the top-left (or top-right). Returns its width."""
    tw, th = text_size(s, scale, thickness)
    h = height or th + 16
    w = tw + 2 * pad_x

    if align == "right":
        x -= w

    rounded_rect(img, (x, y), (x + w, y + h), color, h // 2)
    text(img, s, (x + pad_x, y + (h + th) // 2), scale, text_color, thickness)
    return w


def label_tag(img, s, x, y, color, scale=0.7, text_color=INK, thickness=1) -> None:
    """Box label sitting on top of (x, y); kept inside the image."""
    tw, th = text_size(s, scale, thickness)
    h_img, w_img = img.shape[:2]
    w = tw + 14
    h = th + 14
    x = int(max(0, min(x, w_img - w)))
    y_top = int(y - h)

    if y_top < 0:
        y_top = int(y)  # no room above the box: put the tag just inside it

    cv2.rectangle(img, (x, y_top), (x + w, y_top + h), color, -1)
    text(img, s, (x + 7, y_top + h - 7), scale, text_color, thickness)


def warning_icon(img, center, size, color, ink=INK) -> None:
    cx, cy = int(center[0]), int(center[1])
    pts = np.array(
        [(cx, cy - size), (cx - size, cy + int(size * 0.8)), (cx + size, cy + int(size * 0.8))],
        np.int32,
    )
    cv2.fillPoly(img, [pts], color, AA)
    cv2.line(img, (cx, cy - size // 3), (cx, cy + size // 4), ink, max(2, size // 6), AA)
    cv2.circle(img, (cx, cy + int(size * 0.52)), max(1, size // 9), ink, -1, AA)


def check_icon(img, center, radius, color, ink=INK) -> None:
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), radius, color, -1, AA)
    r = radius
    pts = np.array(
        [(cx - int(r * 0.45), cy + int(r * 0.02)), (cx - int(r * 0.1), cy + int(r * 0.35)), (cx + int(r * 0.5), cy - int(r * 0.3))],
        np.int32,
    )
    cv2.polylines(img, [pts], False, ink, max(2, r // 6), AA)


def eye_icon(img, center, size, color) -> None:
    cx, cy = int(center[0]), int(center[1])
    cv2.ellipse(img, (cx, cy), (size, int(size * 0.55)), 0, 0, 360, color, 2, AA)
    cv2.circle(img, (cx, cy), int(size * 0.32), color, -1, AA)


def person_icon(img, center, size, color) -> None:
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy - int(size * 0.45)), int(size * 0.3), color, -1, AA)
    cv2.ellipse(img, (cx, cy + int(size * 0.55)), (int(size * 0.55), int(size * 0.55)), 0, 180, 360, color, -1, AA)


def bag_icon(img, center, size, color) -> None:
    cx, cy = int(center[0]), int(center[1])
    s = size
    cv2.ellipse(img, (cx, cy - int(s * 0.35)), (int(s * 0.3), int(s * 0.3)), 0, 180, 360, color, max(2, s // 8), AA)
    rounded_rect(img, (cx - int(s * 0.55), cy - int(s * 0.35)), (cx + int(s * 0.55), cy + int(s * 0.65)), color, max(2, s // 5))


def dashed_line(img, p1, p2, color, thickness=2, dash=12) -> None:
    length = math.dist(p1, p2)

    if length < 1:
        return

    steps = max(1, int(length // dash))

    for i in range(0, steps, 2):
        a = i / steps
        b = min(1.0, (i + 1) / steps)
        q1 = (int(p1[0] + (p2[0] - p1[0]) * a), int(p1[1] + (p2[1] - p1[1]) * a))
        q2 = (int(p1[0] + (p2[0] - p1[0]) * b), int(p1[1] + (p2[1] - p1[1]) * b))
        cv2.line(img, q1, q2, color, thickness, AA)


def scale_color(color, factor: float):
    return tuple(int(max(0, min(255, c * factor))) for c in color)


def mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------
def short_label(class_name: str, entity_id) -> str:
    """'Person 1', 'Backpack 2' (no '#', which reads as noise on a TV)."""
    return f"{str(class_name).capitalize()} {entity_id}"


def clean(s: str) -> str:
    """Display form of an engine message: ASCII, no '#' before IDs."""
    s = str(s).replace(" #", " ").replace("#", "")
    return s.encode("ascii", "replace").decode("ascii")


def ago(seconds: float) -> str:
    seconds = max(0.0, seconds)

    if seconds < 2:
        return "just now"

    if seconds < 60:
        return f"{int(seconds)}s ago"

    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"

    return f"{int(seconds // 3600)}h ago"


def clock(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def elapsed(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
