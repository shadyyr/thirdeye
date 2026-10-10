"""Judge-facing OpenCV dashboard for Third Eye, plus a separate debug mode.

Demo mode (default) answers, in order of visual weight:
  1. LIVE CAMERA     - boxes with stable anonymous IDs, restricted zone,
                       person<->object links, unattended object highlight
  2. ACTIVE ALERT    - one large card for the highest-priority alert
  3. CURRENT STATUS  - people / objects / occupancy, system status
  4. RECENT EVENTS   - the last few meaningful events
  5. RECENTLY SEEN   - objects out of view; press 1-4 for a detail card

Debug mode ('d') shows tracker internals, perf counters and every event.

Pure presentation: reads world/zone/relationship/event/memory state and never
mutates it. Text is ASCII only (Hershey fonts); icons are drawn as shapes.
"""

from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np

from thirdeye.core import settings
from thirdeye.core.types import (
    ACTIVE,
    ALERT,
    ARCHIVED,
    ASSOCIATION_CREATED,
    ASSOCIATION_RESTORED,
    CROWDING_STARTED,
    DWELLING,
    ENTITY_ENTERED,
    ENTITY_REAPPEARED,
    INFO,
    NOTICE,
    OBJECT_LEFT_WITH,
    OBJECT_MOVED,
    OBJECT_SEPARATING,
    OUT_OF_VIEW,
    SEVERITY_RANK,
    UNATTENDED_OBJECT,
    WARNING,
    ZONE_ENTERED,
    Alert,
)
from thirdeye.ui import widgets as wd
from thirdeye.ui.widgets import (
    AMBER,
    BG,
    BLUE,
    DIM,
    FONT_BOLD,
    GREEN,
    HEADER_BG,
    INK,
    MUTED,
    OBJECT_COLOR,
    PANEL,
    PANEL_EDGE,
    PERSON_COLOR,
    RED,
    RESTRICTED_COLOR,
    MONITORED_COLOR,
    SUBTLE,
    TEXT,
    WHITE,
)

# ---------------------------------------------------------------------------
# Layout (1920 x 1080 canvas)
# ---------------------------------------------------------------------------
W = settings.DASHBOARD_WIDTH
H = settings.DASHBOARD_HEIGHT
M = 24  # outer margin and gutter
HEADER_H = 68
VIDEO_X = M
VIDEO_Y = HEADER_H + 16
VIDEO_W = 1312
VIDEO_H = 738  # 16:9
PANEL_X = VIDEO_X + VIDEO_W + M
PANEL_W = W - PANEL_X - M
BOTTOM_Y = VIDEO_Y + VIDEO_H + 18
BOTTOM_H = H - M - BOTTOM_Y

TILE_Y = VIDEO_Y
TILE_H = 120
OCC_Y = TILE_Y + TILE_H + 16
OCC_H = 88
CARD_Y = OCC_Y + OCC_H + 16
CARD_H = VIDEO_Y + VIDEO_H - CARD_Y
CARD_BAND_H = 132

EVENT_ROWS = 4
EVENT_ROW_H = 38
SEEN_ROWS = 4
SEEN_ROW_H = 40

DETAIL_W = 1000
DETAIL_H = 640

SEVERITY_COLOR = {
    INFO: (210, 210, 210),
    NOTICE: wd.CYAN,
    WARNING: AMBER,
    ALERT: RED,
}

# Events shown in demo mode (debug mode shows everything). ENTITY_ENTERED is
# further limited to people and ZONE_ENTERED to restricted zones.
DEMO_EVENT_TYPES = {
    ENTITY_ENTERED,
    ZONE_ENTERED,
    DWELLING,
    CROWDING_STARTED,
    ASSOCIATION_CREATED,
    OBJECT_SEPARATING,
    UNATTENDED_OBJECT,
    OBJECT_MOVED,
    OBJECT_LEFT_WITH,
    ASSOCIATION_RESTORED,
    ENTITY_REAPPEARED,
}

# Short, colored tag per demo event type.
EVENT_TAGS = {
    ENTITY_ENTERED: ("ENTERED", PERSON_COLOR),
    ZONE_ENTERED: ("RESTRICTED", RED),
    DWELLING: ("DWELLING", AMBER),
    CROWDING_STARTED: ("CROWDING", RED),
    ASSOCIATION_CREATED: ("LINKED", GREEN),
    OBJECT_SEPARATING: ("SEPARATING", AMBER),
    UNATTENDED_OBJECT: ("UNATTENDED", RED),
    OBJECT_MOVED: ("MOVED", RED),
    OBJECT_LEFT_WITH: ("LEFT WITH", RED),
    ASSOCIATION_RESTORED: ("RETURNED", GREEN),
    ENTITY_REAPPEARED: ("BACK", PERSON_COLOR),
}

# Prefixes dropped from messages in demo mode (the tag already says it).
_MESSAGE_PREFIXES = ("UNATTENDED OBJECT: ", "OBJECT MOVED: ", "OBJECT LEFT WITH: ")

# Shorter phrasing for long alert lines on the card (still conservative:
# "associated with", never "owned by").
_LINE_REWRITES = (
    ("Previously associated with", "Was with"),
    ("Last associated with", "Last with"),
    ("Currently near", "Now near"),
)

# Hero alerts lead the card over ongoing zone/crowd conditions.
ALERT_TYPE_PRIORITY = {"OBJECT_MOVED": 0, "UNATTENDED_OBJECT": 1, "CROWDING_STARTED": 2}


def alert_priority(alert: Alert) -> tuple:
    """Sort key: highest severity, then hero alert types, then newest."""
    return (
        -SEVERITY_RANK.get(alert.severity, 0),
        ALERT_TYPE_PRIORITY.get(alert.alert_type, 5),
        -alert.since,
    )


def is_demo_event(event: dict, restricted_zone_ids: set | None = None) -> bool:
    """True if the event is meaningful enough for the judge-facing timeline."""
    event_type = event.get("event_type")

    if event_type not in DEMO_EVENT_TYPES:
        return False

    if event_type == ENTITY_ENTERED:
        class_name = (event.get("metadata") or {}).get("class_name")

        if class_name is not None:
            return class_name == "person"

        return str(event.get("message", "")).lower().startswith("person")

    if event_type == ZONE_ENTERED:
        zone_type = (event.get("metadata") or {}).get("zone_type")

        if zone_type is not None:
            return str(zone_type).upper() == "RESTRICTED"

        if restricted_zone_ids is not None:
            return event.get("zone_id") in restricted_zone_ids

    return True


def demo_message(event: dict) -> str:
    message = str(event.get("message", ""))

    for prefix in _MESSAGE_PREFIXES:
        if message.upper().startswith(prefix):
            message = message[len(prefix):]
            break

    message = wd.clean(message)
    return message[:1].upper() + message[1:]


def demo_line(line: str) -> str:
    line = wd.clean(line)

    for old, new in _LINE_REWRITES:
        line = line.replace(old, new)

    return line


def _rel_person(rel):
    return rel.person_id if rel.person_id is not None else rel.last_associated_person_id


class Dashboard:
    def __init__(self, mode: str = settings.DASHBOARD_DEFAULT_MODE) -> None:
        self.mode = mode if mode in ("demo", "debug") else "demo"
        self.selected_entity_id: int | None = None
        self.events: deque = deque(maxlen=200)  # newest first
        self.canvas = np.empty((H, W, 3), np.uint8)
        self._backgrounds: dict = {}
        self._recent_ids: list[int] = []  # recently-seen list from the last render
        self._snapshot_cache: dict = {}
        self.started = time.time()
        self._restricted_ids: set = set()

    # Backwards compatible alias for the v1 timeline.
    @property
    def timeline(self) -> deque:
        return self.events

    def add_events(self, events: list[dict]) -> None:
        for event in events:
            self.events.appendleft(event)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        if key is None or key < 0:
            return False

        key &= 0xFF

        if key in (ord("d"), ord("D")):
            self.mode = "debug" if self.mode == "demo" else "demo"
            return True

        if key in (ord("0"), ord("x"), ord("X")):
            self.selected_entity_id = None
            return True

        if ord("1") <= key <= ord("9"):
            index = key - ord("1")

            if index < min(len(self._recent_ids), SEEN_ROWS):
                self.selected_entity_id = self._recent_ids[index]
                return True

            return False

        return False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, frame, world, zone_manager, relationships, alerts, stats: dict, memory=None):
        stats = stats or {}
        now = world.now
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6.0)
        debug = self.mode == "debug"
        relationships = list(relationships or [])
        alerts = list(alerts or [])
        zones = list(getattr(zone_manager, "zones", []) or [])

        self._restricted_ids = {z.id for z in zones if z.is_restricted}
        recent = self._recent_objects(memory, now)
        self._recent_ids = [s.entity_id for s in recent]

        canvas = self.canvas
        np.copyto(canvas, self._background(debug, memory is not None))

        # Live camera
        video = canvas[VIDEO_Y : VIDEO_Y + VIDEO_H, VIDEO_X : VIDEO_X + VIDEO_W]

        if frame is not None and frame.size:
            fh, fw = frame.shape[:2]
            video[:] = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        else:
            fw, fh = world.frame_size

        sx, sy = VIDEO_W / max(fw, 1), VIDEO_H / max(fh, 1)
        self._draw_zones(video, world, zones, (fw, fh), sx, sy, debug)
        self._draw_relationships(video, world, relationships, sx, sy, debug)
        self._draw_entities(video, world, relationships, alerts, sx, sy, pulse, debug)
        self._draw_header(canvas, stats, debug)

        if debug:
            from thirdeye.ui import debug_view

            debug_view.draw_panel(canvas, world, zones, relationships, alerts, stats, now)
            debug_view.draw_events(canvas, list(self.events), now)
        else:
            self._draw_status(canvas, world, zones, stats)
            self._draw_alert_card(canvas, alerts, now, pulse)
            self._draw_events(canvas, now, self._restricted_ids, memory is not None)

            if memory is not None:
                self._draw_recently_seen(canvas, recent, now)

        if self.selected_entity_id is not None:
            self._draw_detail_card(canvas, memory, now)

        return canvas

    # ------------------------------------------------------------------
    # Static chrome
    # ------------------------------------------------------------------
    def _background(self, debug: bool, has_memory: bool):
        key = (debug, has_memory)
        bg = self._backgrounds.get(key)

        if bg is not None:
            return bg

        bg = np.empty((H, W, 3), np.uint8)
        bg[:] = BG
        cv2.rectangle(bg, (0, 0), (W, HEADER_H), HEADER_BG, -1)
        cv2.line(bg, (0, HEADER_H), (W, HEADER_H), (40, 36, 32), 2)
        wd.eye_icon(bg, (M + 26, HEADER_H // 2), 22, GREEN)
        wd.text(bg, "THIRD EYE", (M + 64, HEADER_H // 2 + 14), 1.15, TEXT, 2, FONT_BOLD)
        cv2.rectangle(bg, (VIDEO_X - 2, VIDEO_Y - 2), (VIDEO_X + VIDEO_W + 1, VIDEO_Y + VIDEO_H + 1), PANEL_EDGE, 2)

        if debug:
            wd.rounded_rect(bg, (PANEL_X, VIDEO_Y), (PANEL_X + PANEL_W, H - M), PANEL, 12)
            wd.rounded_rect(bg, (VIDEO_X, BOTTOM_Y), (VIDEO_X + VIDEO_W, H - M), PANEL, 12)
            wd.text(bg, "ALL EVENTS (unfiltered)", (VIDEO_X + 20, BOTTOM_Y + 30), 0.6, MUTED)
        else:
            half = (PANEL_W - 16) // 2

            for i, (name, icon) in enumerate((("PEOPLE", wd.person_icon), ("OBJECTS", wd.bag_icon))):
                x = PANEL_X + i * (half + 16)
                wd.rounded_rect(bg, (x, TILE_Y), (x + half, TILE_Y + TILE_H), PANEL, 12)
                wd.text(bg, name, (x + 22, TILE_Y + 36), 0.62, MUTED)
                icon(bg, (x + half - 46, TILE_Y + TILE_H // 2 + 6), 30, wd.scale_color(PERSON_COLOR if i == 0 else OBJECT_COLOR, 0.55))

            wd.rounded_rect(bg, (PANEL_X, OCC_Y), (PANEL_X + PANEL_W, OCC_Y + OCC_H), PANEL, 12)
            wd.text(bg, "OCCUPANCY", (PANEL_X + 22, OCC_Y + 32), 0.62, MUTED)
            wd.rounded_rect(bg, (PANEL_X, CARD_Y), (PANEL_X + PANEL_W, CARD_Y + CARD_H), PANEL, 12)

            events_w = VIDEO_W if has_memory else W - 2 * M
            wd.rounded_rect(bg, (VIDEO_X, BOTTOM_Y), (VIDEO_X + events_w, BOTTOM_Y + BOTTOM_H), PANEL, 12)
            wd.text(bg, "RECENT EVENTS", (VIDEO_X + 22, BOTTOM_Y + 34), 0.62, MUTED)

            if has_memory:
                wd.rounded_rect(bg, (PANEL_X, BOTTOM_Y), (PANEL_X + PANEL_W, BOTTOM_Y + BOTTOM_H), PANEL, 12)
                wd.text(bg, "RECENTLY SEEN", (PANEL_X + 22, BOTTOM_Y + 34), 0.62, MUTED)
                hint = "press 1-4 for details"
                wd.text(bg, hint, (PANEL_X + PANEL_W - 22 - wd.text_width(hint, 0.52), BOTTOM_Y + 33), 0.52, DIM)

        self._backgrounds[key] = bg
        return bg

    def _draw_header(self, canvas, stats, debug):
        status = str(stats.get("status", "LIVE")).upper()
        color = GREEN if status == "LIVE" else AMBER
        label = f"{status}  -  ON-DEVICE"
        x = W - M - wd.text_width(label, 0.75, 1) - 8
        cy = HEADER_H // 2
        r = 9 if status != "LIVE" else 7 + int(3 * (0.5 + 0.5 * math.sin(time.time() * 3)))
        cv2.circle(canvas, (x - 24, cy), r, color, -1, wd.AA)
        wd.text(canvas, label, (x, cy + 10), 0.75, color, 1)

        if debug:
            fps = stats.get("fps") or 0.0
            infer = stats.get("infer_ms") or 0.0
            perf = f"DEBUG   {fps:4.1f} FPS   YOLO {infer:3.0f} ms"
            wd.pill(canvas, perf, x - 70, cy - 18, AMBER, 0.62, INK, 1, height=36, align="right")

    # ------------------------------------------------------------------
    # Video overlay (drawn on the scaled video ROI of the canvas)
    # ------------------------------------------------------------------
    def _draw_zones(self, img, world, zones, frame_size, sx, sy, debug):
        occupied: dict[str, int] = {}

        for record in world.people():
            for zone_id in record.zones:
                occupied[zone_id] = occupied.get(zone_id, 0) + 1

        debug_row = 0  # stacks debug labels of zones that share a corner

        for zone in zones:
            if not zone.active:
                continue

            pts = np.array(zone.polygon_px(frame_size), np.float32)
            pts = np.round(pts * (sx, sy)).astype(np.int32)
            x1, y1 = np.maximum(pts.min(axis=0), 0)
            x2, y2 = np.minimum(pts.max(axis=0) + 1, (img.shape[1], img.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            count = occupied.get(zone.id, 0)

            if zone.is_restricted:
                roi = img[y1:y2, x1:x2]
                overlay = roi.copy()
                cv2.fillPoly(overlay, [pts - (x1, y1)], RESTRICTED_COLOR)
                alpha = 0.26 if count else 0.14
                cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
                cv2.polylines(img, [pts], True, RESTRICTED_COLOR, 5 if count else 3, wd.AA)
                name = zone.name.upper()
                wd.pill(img, wd.fit(name, int(x2 - x1) - 40, 0.62), int(x1) + 12, int(y1) + 12, RESTRICTED_COLOR, 0.62, WHITE, 1, height=34)
            else:
                cv2.polylines(img, [pts], True, MONITORED_COLOR, 2, wd.AA)
                wd.text(img, zone.name, (int(x1) + 12, int(y1) + 32), 0.62, MONITORED_COLOR)

            if debug:
                wd.text(img, f"{zone.id}: {count} people", (int(x1) + 12, int(y2) - 14 - 26 * debug_row), 0.55, WHITE)
                debug_row += 1

    def _draw_relationships(self, img, world, relationships, sx, sy, debug):
        for rel in relationships:
            obj = world.get(rel.object_id)
            person_id = _rel_person(rel)
            person = world.get(person_id) if person_id is not None else None

            if obj is None or person is None:
                continue

            p1 = (int(obj.center[0] * sx), int(obj.center[1] * sy))
            p2 = (int(person.center[0] * sx), int(person.center[1] * sy))
            state = rel.state

            if state == "ASSOCIATED":
                cv2.line(img, p1, p2, (190, 235, 190), 2, wd.AA)
                cv2.circle(img, p1, 5, (190, 235, 190), -1, wd.AA)
                cv2.circle(img, p2, 5, (190, 235, 190), -1, wd.AA)
            elif state == "SEPARATING":
                wd.dashed_line(img, p1, p2, AMBER, 2, 14)
            elif state == "UNATTENDED":
                wd.dashed_line(img, p1, p2, RED, 2, 14)
            elif debug and state == "CANDIDATE_ASSOCIATION":
                wd.dashed_line(img, p1, p2, wd.CYAN, 1, 10)

    def _draw_entities(self, img, world, relationships, alerts, sx, sy, pulse, debug):
        rel_by_object = {rel.object_id: rel for rel in relationships}
        moved_ids: set[int] = set()

        for alert in alerts:
            if alert.alert_type == OBJECT_MOVED and alert.entity_ids:
                moved_ids.add(alert.entity_ids[0])

        now = world.now

        # People first so object tags (usually lower in the frame) stay on top.
        records = sorted(world.entities.values(), key=lambda r: (not r.is_person, r.track_id))

        for record in records:
            x1, y1, x2, y2 = record.bbox
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            color = PERSON_COLOR if record.is_person else OBJECT_COLOR
            text_color = INK
            thickness = 3
            label = wd.short_label(record.class_name, record.track_id)
            rel = rel_by_object.get(record.track_id)
            highlight = False

            if record.is_person and record.zones and self._in_restricted(record):
                color, text_color = RED, WHITE
            elif rel is not None and rel.state == "UNATTENDED":
                color, text_color, highlight = RED, WHITE, True
                label += "  UNATTENDED"
            elif record.track_id in moved_ids:
                color, text_color, highlight = RED, WHITE, True
                label += "  MOVED"

            if record.frames_missed and not highlight:
                color = wd.scale_color(color, 0.6)
                thickness = 2

            if highlight:
                thickness = 3 + int(4 * pulse)
                pad = 4 + int(6 * pulse)
                glow = wd.mix(RED, (120, 120, 255), pulse)
                cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), glow, 2, wd.AA)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, wd.AA)

            if debug:
                label += f"  {record.confidence:.2f}  {record.age(now):.0f}s"

                if record.frames_missed:
                    label += f"  miss {record.frames_missed}"

                if rel is not None:
                    pid = _rel_person(rel)
                    label += f"  {rel.state[:5]}" + (f">{pid}" if pid is not None else "")

            wd.label_tag(img, label, x1 - (thickness // 2), y1 - (thickness // 2), color, 0.7, text_color, 1)

    def _in_restricted(self, record) -> bool:
        return bool(record.zones & self._restricted_ids)

    # ------------------------------------------------------------------
    # Demo panels
    # ------------------------------------------------------------------
    def _draw_status(self, canvas, world, zones, stats):
        half = (PANEL_W - 16) // 2
        people = len(world.people())
        objects = len(world.objects())

        for i, value in enumerate((people, objects)):
            x = PANEL_X + i * (half + 16)
            wd.text(canvas, str(value), (x + 22, TILE_Y + 100), 2.0, TEXT, 3, FONT_BOLD)

        crowded = bool(stats.get("crowded", False))
        status, color = ("Crowded", RED) if crowded else ("Normal", GREEN)
        cv2.circle(canvas, (PANEL_X + 32, OCC_Y + 64), 9, color, -1, wd.AA)
        wd.text(canvas, status, (PANEL_X + 52, OCC_Y + 74), 1.0, color, 2, FONT_BOLD)

        # Restricted-zone headcount on the right half of the occupancy row.
        zone = next((z for z in zones if z.active and z.is_restricted), None)

        if zone is not None:
            occupancy = stats.get("zone_occupancy") or {}
            count = occupancy.get(zone.id)

            if count is None:
                count = sum(1 for r in world.people() if zone.id in r.zones)

            x = PANEL_X + half + 16
            wd.text(canvas, wd.fit(zone.name.upper(), half - 30, 0.55), (x, OCC_Y + 32), 0.55, MUTED)
            value = f"{count} {'person' if count == 1 else 'people'}"
            wd.text(canvas, value, (x, OCC_Y + 74), 1.0, RED if count else SUBTLE, 2, FONT_BOLD)
        else:
            total = stats.get("occupancy_total", len(world.people()))
            x = PANEL_X + half + 16
            wd.text(canvas, "IN VIEW", (x, OCC_Y + 32), 0.55, MUTED)
            wd.text(canvas, f"{total} people", (x, OCC_Y + 74), 1.0, SUBTLE, 2, FONT_BOLD)

    def _draw_alert_card(self, canvas, alerts, now, pulse):
        x1, y1 = PANEL_X, CARD_Y
        x2, y2 = PANEL_X + PANEL_W, CARD_Y + CARD_H

        if not alerts:
            cx = (x1 + x2) // 2
            cy = y1 + CARD_H // 2 - 70
            wd.check_icon(canvas, (cx, cy), 52, GREEN, PANEL)
            title = "ALL CLEAR"
            wd.text(canvas, title, (cx - wd.text_width(title, 1.5, 2, FONT_BOLD) // 2, cy + 120), 1.5, GREEN, 2, FONT_BOLD)
            sub = "No active alerts"
            wd.text(canvas, sub, (cx - wd.text_width(sub, 0.85) // 2, cy + 170), 0.85, SUBTLE)
            return

        alert = alerts[0]
        color = SEVERITY_COLOR.get(alert.severity, RED)
        band_y2 = y1 + CARD_BAND_H

        # Colored header band (rounded top corners come from the panel shape).
        wd.rounded_rect(canvas, (x1, y1), (x2, band_y2), color, 12)
        cv2.rectangle(canvas, (x1, band_y2 - 14), (x2, band_y2), color, -1)
        ink = WHITE if alert.severity == ALERT else INK
        wd.warning_icon(canvas, (x1 + 50, y1 + CARD_BAND_H // 2), 30, ink, color)

        title_lines = wd.wrap(alert.title.upper(), PANEL_W - 120, 1.15, 2, FONT_BOLD, 2)
        line_h = 50
        ty = y1 + (CARD_BAND_H - line_h * len(title_lines)) // 2 + 38

        for i, line in enumerate(title_lines):
            wd.text(canvas, line, (x1 + 98, ty + i * line_h), 1.15, ink, 2, FONT_BOLD)

        # Pulsing outline for the most severe alerts.
        if alert.severity == ALERT:
            edge = wd.mix(PANEL, color, 0.35 + 0.65 * pulse)
            wd.rounded_rect(canvas, (x1, y1), (x2, y2), edge, 12, 3)

        # Detail lines
        body: list[str] = []

        for line in alert.lines[:3]:
            body += wd.wrap(demo_line(line), PANEL_W - 56, 0.92, 1, wd.FONT, 2)

        for i, line in enumerate(body[:4]):
            weight = 2 if i == 0 else 1
            wd.text(canvas, line, (x1 + 28, band_y2 + 56 + i * 46), 0.92, TEXT if i == 0 else SUBTLE, weight)

        # Elapsed timer
        wd.text(canvas, "ELAPSED", (x1 + 28, y2 - 92), 0.58, MUTED)
        wd.text(canvas, wd.elapsed(now - alert.since), (x1 + 26, y2 - 30), 1.9, TEXT, 3, FONT_BOLD)

        if len(alerts) > 1:
            more = f"+{len(alerts) - 1} more"
            wd.pill(canvas, more, x2 - 26, y2 - 76, (80, 72, 66), 0.72, TEXT, 1, height=42, align="right")

    def _draw_events(self, canvas, now, restricted_ids, has_memory):
        x = VIDEO_X
        width = VIDEO_W if has_memory else W - 2 * M
        rows = []

        for event in self.events:
            if is_demo_event(event, restricted_ids):
                rows.append(event)

                if len(rows) >= EVENT_ROWS:
                    break

        if not rows:
            wd.text(canvas, "Waiting for activity...", (x + 22, BOTTOM_Y + 84), 0.78, DIM)
            return

        tag_x = x + 190
        msg_x = x + 410

        for i, event in enumerate(rows):
            y = BOTTOM_Y + 76 + i * EVENT_ROW_H
            tag, color = EVENT_TAGS.get(event["event_type"], (event["event_type"][:10], SEVERITY_COLOR.get(event.get("severity"), TEXT)))
            stamp = wd.ago(now - float(event.get("timestamp", now)))
            wd.text(canvas, stamp, (x + 22, y), 0.7, MUTED)
            severe = SEVERITY_RANK.get(event.get("severity"), 0) >= SEVERITY_RANK[WARNING]

            if severe:
                wd.warning_icon(canvas, (tag_x - 18, y - 9), 11, color)
            else:
                cv2.circle(canvas, (tag_x - 18, y - 8), 7, color, -1, wd.AA)

            wd.text(canvas, tag, (tag_x, y), 0.72, color, 2 if severe else 1)
            message = wd.fit(demo_message(event), width - (msg_x - x) - 24, 0.78)
            wd.text(canvas, message, (msg_x, y), 0.78, TEXT if i == 0 else SUBTLE)

    def _recent_objects(self, memory, now) -> list:
        if memory is None:
            return []

        try:
            items = memory.recent_objects(now, getattr(settings, "RECENTLY_SEEN_LIMIT", SEEN_ROWS) + 4)
        except Exception:
            return []

        return [s for s in items if s.status != ACTIVE][:SEEN_ROWS]

    def _draw_recently_seen(self, canvas, recent, now):
        x = PANEL_X

        if not recent:
            wd.text(canvas, "No objects out of view", (x + 22, BOTTOM_Y + 84), 0.72, DIM)
            return

        for i, summary in enumerate(recent):
            top = BOTTOM_Y + 50 + i * SEEN_ROW_H
            base = top + 27
            selected = summary.entity_id == self.selected_entity_id
            badge = AMBER if summary.status == OUT_OF_VIEW else (120, 114, 108)

            if selected:
                wd.rounded_rect(canvas, (x + 10, top), (x + PANEL_W - 10, top + SEEN_ROW_H - 4), wd.PANEL_HI, 8)

            wd.rounded_rect(canvas, (x + 20, top + 4), (x + 50, top + 34), badge, 6)
            wd.text(canvas, str(i + 1), (x + 27, base + 1), 0.62, INK, 1)
            label = wd.short_label(summary.class_name, summary.entity_id)
            wd.text(canvas, label, (x + 64, base), 0.68, TEXT)
            when = wd.ago(now - summary.last_seen)
            when_w = wd.text_width(when, 0.6)
            wd.text(canvas, when, (x + PANEL_W - 22 - when_w, base), 0.6, MUTED)
            place = summary.last_zone_name or ("left view" if summary.status == ARCHIVED else "out of view")
            zx = x + 70 + wd.text_width(label, 0.68)
            room = x + PANEL_W - 40 - when_w - zx

            if room > 40:
                wd.text(canvas, wd.fit(place, room, 0.58), (zx + 8, base), 0.58, MUTED)

    # ------------------------------------------------------------------
    # Detail card (overlay over the video)
    # ------------------------------------------------------------------
    def _draw_detail_card(self, canvas, memory, now):
        summary = None

        if memory is not None:
            try:
                summary = memory.summary(self.selected_entity_id)
            except Exception:
                summary = None

        if summary is None:
            self.selected_entity_id = None
            return

        # Dim the video behind the card (bit shift: much cheaper than a blend).
        video = canvas[VIDEO_Y : VIDEO_Y + VIDEO_H, VIDEO_X : VIDEO_X + VIDEO_W]
        np.right_shift(video, 1, out=video)

        x1 = VIDEO_X + (VIDEO_W - DETAIL_W) // 2
        y1 = VIDEO_Y + (VIDEO_H - DETAIL_H) // 2
        x2, y2 = x1 + DETAIL_W, y1 + DETAIL_H
        wd.rounded_rect(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), PANEL_EDGE, 16)
        wd.rounded_rect(canvas, (x1, y1), (x2, y2), (36, 31, 28), 14)

        # Header
        if summary.status == ACTIVE:
            status, scolor = "IN VIEW", GREEN
        elif summary.status == OUT_OF_VIEW:
            status, scolor = "OUT OF VIEW", AMBER
        else:
            status, scolor = "LEFT VIEW", (170, 164, 158)

        is_person = summary.class_name == "person"
        icon = wd.person_icon if is_person else wd.bag_icon
        icon(canvas, (x1 + 48, y1 + 50), 26, PERSON_COLOR if is_person else OBJECT_COLOR)
        title = wd.short_label(summary.class_name, summary.entity_id)
        wd.text(canvas, title, (x1 + 92, y1 + 66), 1.35, TEXT, 2, FONT_BOLD)
        tx = x1 + 92 + wd.text_width(title, 1.35, 2, FONT_BOLD) + 24
        wd.pill(canvas, status, tx, y1 + 32, scolor, 0.7, INK, 1, height=40)
        close = "0 / X  close"
        wd.text(canvas, close, (x2 - 28 - wd.text_width(close, 0.6), y1 + 60), 0.6, MUTED)
        cv2.line(canvas, (x1 + 24, y1 + 100), (x2 - 24, y1 + 100), PANEL_EDGE, 1)

        # Snapshot
        sx1, sy1, side = x1 + 28, y1 + 124, 260
        cv2.rectangle(canvas, (sx1, sy1), (sx1 + side, sy1 + side), (24, 21, 19), -1)
        snap = self._snapshot(memory, summary)

        if snap is not None:
            sh, sw = snap.shape[:2]
            ox, oy = sx1 + (side - sw) // 2, sy1 + (side - sh) // 2
            canvas[oy : oy + sh, ox : ox + sw] = snap
        else:
            icon(canvas, (sx1 + side // 2, sy1 + side // 2 - 20), 46, DIM)
            msg = "No snapshot"
            wd.text(canvas, msg, (sx1 + (side - wd.text_width(msg, 0.6)) // 2, sy1 + side - 40), 0.6, DIM)

        cv2.rectangle(canvas, (sx1, sy1), (sx1 + side, sy1 + side), PANEL_EDGE, 1)
        wd.text(canvas, "LAST SEEN SNAPSHOT", (sx1, sy1 + side + 28), 0.5, MUTED)

        # Facts
        fx = sx1 + side + 40
        fw = x2 - 28 - fx
        facts = [
            ("LAST SEEN", f"{wd.ago(now - summary.last_seen)}  ({wd.clock(summary.last_seen)})", TEXT),
            ("LAST LOCATION", summary.last_zone_name or "Open area", TEXT),
        ]
        person = summary.associated_person_id or summary.previous_person_id

        if person is not None:
            facts.append(("ASSOCIATED WITH", wd.short_label("person", person), TEXT))

        if summary.left_with_person_id is not None:
            facts.append(("LEFT VIEW WITH", wd.short_label("person", summary.left_with_person_id), RED))
        elif summary.previous_person_id is not None and summary.previous_person_id != person:
            facts.append(("PREVIOUSLY WITH", wd.short_label("person", summary.previous_person_id), TEXT))

        for i, (name, value, color) in enumerate(facts[:4]):
            fy = sy1 + 18 + i * 68
            wd.text(canvas, name, (fx, fy), 0.55, MUTED)
            wd.text(canvas, wd.fit(value, fw, 0.9, 2 if color == RED else 1), (fx, fy + 36), 0.9, color, 2 if color == RED else 1)

        # Per-entity timeline
        ty = sy1 + side + 70
        cv2.line(canvas, (x1 + 24, ty - 6), (x2 - 24, ty - 6), PANEL_EDGE, 1)
        wd.text(canvas, "TIMELINE", (x1 + 28, ty + 26), 0.55, MUTED)

        try:
            events = memory.events_for(summary.entity_id, 4) or []
        except Exception:
            events = []

        if not events:
            wd.text(canvas, "No events recorded", (x1 + 28, ty + 64), 0.7, DIM)

        for i, event in enumerate(events[:4]):
            ey = ty + 62 + i * 36

            if ey > y2 - 14:
                break

            tag, color = EVENT_TAGS.get(event.get("event_type"), ("", SEVERITY_COLOR.get(event.get("severity"), SUBTLE)))
            wd.text(canvas, wd.clock(float(event.get("timestamp", now))), (x1 + 28, ey), 0.62, MUTED)
            cv2.circle(canvas, (x1 + 158, ey - 7), 6, color, -1, wd.AA)
            wd.text(canvas, wd.fit(demo_message(event), DETAIL_W - 210, 0.7), (x1 + 178, ey), 0.7, TEXT)

    def _snapshot(self, memory, summary):
        key = (summary.entity_id, summary.last_seen, summary.snapshot_path)
        cached = self._snapshot_cache.get(summary.entity_id)

        if cached is not None and cached[0] == key:
            return cached[1]

        try:
            img = memory.snapshot(summary.entity_id)
        except Exception:
            img = None

        if img is not None and getattr(img, "size", 0):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            h, w = img.shape[:2]
            s = 256 / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
        else:
            img = None

        self._snapshot_cache[summary.entity_id] = (key, img)
        return img
