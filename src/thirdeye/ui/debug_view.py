"""Debug-mode panels for the Third Eye dashboard ('d' toggles).

Dense developer view: perf counters, tracker table, association states,
active alerts and the unfiltered event log. Text only (plain putText), drawn
on the static panels that ``Dashboard._background(debug=True)`` provides.

Must never raise on missing or partial data: every field is read defensively.
"""

from __future__ import annotations

from thirdeye.core.types import SEVERITY_RANK, WARNING
from thirdeye.ui import dashboard as dash
from thirdeye.ui import widgets as wd
from thirdeye.ui.widgets import AMBER, DIM, GREEN, MUTED, RED, SUBTLE, TEXT

# Right panel
PAD = 20
LINE_H = 23
SCALE = 0.5
HEAD_SCALE = 0.55
SECTION_GAP = 10
TRACK_ROWS_MAX = 16
ASSOC_ROWS_MAX = 6
ALERT_ROWS_MAX = 4

# Tracks table columns (x offsets from the panel text origin)
TRACK_COLUMNS = (
    ("ID", 0),
    ("CLASS", 34),
    ("CONF", 120),
    ("MISS", 168),
    ("STATE", 210),
    ("AGE", 268),
    ("STILL", 320),
    ("ZONES", 374),
)

# Bottom event log
EVENT_TOP = 30 + 28  # first baseline below the caption (relative to BOTTOM_Y)
EVENT_ROW_H = 22
EVENT_SCALE = 0.5

_STATE_SHORT = {"ACTIVE": "ACTIVE", "OUT_OF_VIEW": "OUT", "ARCHIVED": "ARCH"}

_REL_SHORT = {"CANDIDATE_ASSOCIATION": "CANDIDATE"}

_STATE_COLOR = {
    "ASSOCIATED": GREEN,
    "CANDIDATE_ASSOCIATION": wd.CYAN,
    "SEPARATING": AMBER,
    "UNATTENDED": RED,
}

_HANDLED_STATS = {
    "fps",
    "infer_ms",
    "status",
    "filter_suppressed",
    "track_merges",
    "occupancy_total",
    "crowded",
    "zone_occupancy",
}


def _num(value, fmt: str, default: str = "-") -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return default


def _age(seconds) -> str:
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "-"

    if seconds < 100:
        return f"{seconds:.1f}s"

    if seconds < 6000:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}"

    return f"{int(seconds // 3600)}h"


def _pid(value) -> str:
    return "-" if value is None else str(value)


class _Column:
    """Cursor over the right panel; refuses to draw below its bottom."""

    def __init__(self, canvas) -> None:
        self.canvas = canvas
        self.x = dash.PANEL_X + PAD
        self.width = dash.PANEL_W - 2 * PAD
        self.y = dash.VIDEO_Y + 30
        self.bottom = dash.H - dash.M - 12

    def room(self) -> int:
        return max(0, (self.bottom - self.y) // LINE_H + 1)

    def heading(self, title: str, note: str = "") -> None:
        if self.y > self.bottom:
            return

        wd.text(self.canvas, title, (self.x, self.y), HEAD_SCALE, MUTED)

        if note:
            note = wd.fit(note, self.width - wd.text_width(title, HEAD_SCALE) - 16, SCALE)
            wd.text(self.canvas, note, (self.x + self.width - wd.text_width(note, SCALE), self.y), SCALE, DIM)

        self.y += LINE_H + 2

    def line(self, s: str, color=TEXT, x_off: int = 0) -> None:
        if self.y > self.bottom:
            return

        wd.text(self.canvas, wd.fit(s, self.width - x_off, SCALE), (self.x + x_off, self.y), SCALE, color)
        self.y += LINE_H

    def gap(self) -> None:
        self.y += SECTION_GAP

    def rows(self, lines: list, cap: int, reserve: int = 0) -> None:
        """Draw (text, color) rows, at most ``cap`` and leaving ``reserve`` lines.

        Truncated lists end with a '+N more' line.
        """
        limit = max(1, min(cap, self.room() - reserve))

        if len(lines) > limit:
            shown, hidden = lines[: limit - 1], len(lines) - (limit - 1)
        else:
            shown, hidden = lines, 0

        for s, color in shown:
            self.line(s, color, 6)

        if hidden:
            self.line(f"+{hidden} more", DIM, 6)


# ---------------------------------------------------------------------------
# Right panel
# ---------------------------------------------------------------------------
def draw_panel(canvas, world, zones, relationships, alerts, stats, now) -> None:
    stats = stats or {}
    col = _Column(canvas)
    entities = list((getattr(world, "entities", None) or {}).values())
    relationships = list(relationships or [])
    alerts = list(alerts or [])

    _draw_perf(col, stats)
    _draw_counts(col, world, entities, zones, stats)

    # Reserve space so associations / alerts / archive never get pushed out.
    later = 3 + min(len(relationships), ASSOC_ROWS_MAX) + min(len(alerts), ALERT_ROWS_MAX) + 3
    _draw_tracks(col, entities, now, later)
    _draw_associations(col, relationships, now, 3 + min(len(alerts), ALERT_ROWS_MAX) + 2)
    _draw_alerts(col, alerts, now)

    col.gap()
    archived = len(getattr(world, "archived", None) or {})
    col.line(f"Archived records: {archived}", SUBTLE)


def _draw_perf(col: _Column, stats: dict) -> None:
    status = str(stats.get("status", "LIVE")).upper()
    col.heading("PERFORMANCE", status)
    fps = _num(stats.get("fps"), ".1f")
    infer = _num(stats.get("infer_ms"), ".1f")
    col.line(f"FPS {fps}    YOLO {infer} ms")
    suppressed = stats.get("filter_suppressed", "-")
    merges = stats.get("track_merges", "-")
    col.line(f"Filter suppressed {suppressed}    Track merges {merges}", SUBTLE)

    extra = [f"{k}={v}" for k, v in stats.items() if k not in _HANDLED_STATS and isinstance(v, (int, float, str))]

    if extra:
        col.line("  ".join(extra), DIM)

    col.gap()


def _draw_counts(col: _Column, world, entities, zones, stats: dict) -> None:
    people = sum(1 for e in entities if getattr(e, "is_person", False))
    objects = sum(1 for e in entities if getattr(e, "is_object", False))
    occupancy = stats.get("occupancy_total", people)
    crowded = bool(stats.get("crowded", False))
    col.heading("COUNTS")
    col.line(
        f"People {people}   Objects {objects}   Occupancy {occupancy}   Crowded {'YES' if crowded else 'no'}",
        RED if crowded else TEXT,
    )
    zone_occ = stats.get("zone_occupancy") or {}

    if not isinstance(zone_occ, dict):
        zone_occ = {}

    if not zone_occ and zones:
        for zone in zones:
            zid = getattr(zone, "id", None)
            zone_occ[zid] = sum(1 for e in entities if getattr(e, "is_person", False) and zid in getattr(e, "zones", ()))

    if zone_occ:
        col.line("Zones  " + "   ".join(f"{k}: {v}" for k, v in zone_occ.items()), SUBTLE)

    col.gap()


def _draw_tracks(col: _Column, entities, now, reserve: int) -> None:
    col.heading("TRACKS", f"{len(entities)} live")

    if not entities:
        col.line("none", DIM, 6)
        col.gap()
        return

    # Column header
    if col.y <= col.bottom:
        for name, off in TRACK_COLUMNS:
            wd.text(col.canvas, name, (col.x + 6 + off, col.y), 0.42, DIM)

        col.y += LINE_H - 2

    records = sorted(entities, key=lambda r: getattr(r, "track_id", 0))
    limit = max(1, min(TRACK_ROWS_MAX, col.room() - reserve))
    shown = records if len(records) <= limit else records[: limit - 1]

    for record in shown:
        if col.y > col.bottom:
            break

        _track_row(col, record, now)

    hidden = len(records) - len(shown)

    if hidden:
        col.line(f"+{hidden} more", DIM, 6)

    col.gap()


def _track_row(col: _Column, record, now) -> None:
    try:
        age = record.age(now)
        still = record.stationary_seconds(now)
    except Exception:
        age = still = None

    missed = getattr(record, "frames_missed", 0) or 0
    status = str(getattr(record, "status", "") or "")
    zones = getattr(record, "zones", None) or ()
    cells = (
        str(getattr(record, "track_id", "?")),
        str(getattr(record, "class_name", "?"))[:9],
        _num(getattr(record, "confidence", None), ".2f"),
        str(missed),
        _STATE_SHORT.get(status, status[:6]),
        _age(age),
        _age(still),
        ",".join(sorted(str(z) for z in zones)) or "-",
    )
    color = TEXT if not missed else AMBER
    x0 = col.x + 6

    last = len(TRACK_COLUMNS) - 1

    for i, ((_, off), cell) in enumerate(zip(TRACK_COLUMNS, cells)):
        if i == last:  # zones: fit to the panel edge
            cell = wd.fit(cell, col.width - 6 - off, SCALE)

        wd.text(col.canvas, cell, (x0 + off, col.y), SCALE, color)

    col.y += LINE_H


def _draw_associations(col: _Column, relationships, now, reserve: int) -> None:
    col.heading("ASSOCIATIONS", f"{len(relationships)}")

    if not relationships:
        col.line("none", DIM, 6)
        col.gap()
        return

    lines = []

    for rel in sorted(relationships, key=lambda r: getattr(r, "object_id", 0) or 0):
        state = str(getattr(rel, "state", "?"))
        since = getattr(rel, "state_since", None)
        prox = getattr(rel, "proximity", None)
        s = (
            f"{getattr(rel, 'object_id', '?')} {str(getattr(rel, 'object_class', '?'))[:8]}"
            f"  {_REL_SHORT.get(state, state)}"
            f"  p={_pid(getattr(rel, 'person_id', None))}"
            f" last={_pid(getattr(rel, 'last_associated_person_id', None))}"
        )

        if prox is not None:
            s += f"  prox {_num(prox, '.2f')}"

        if since is not None:
            s += f"  {_age(now - since)}"

        lines.append((s, _STATE_COLOR.get(state, SUBTLE)))

    col.rows(lines, ASSOC_ROWS_MAX, reserve)
    col.gap()


def _draw_alerts(col: _Column, alerts, now) -> None:
    col.heading("ACTIVE ALERTS", f"{len(alerts)}")

    if not alerts:
        col.line("none", DIM, 6)
        return

    lines = []

    for alert in sorted(alerts, key=dash.alert_priority):
        severity = str(getattr(alert, "severity", "?"))
        since = getattr(alert, "since", now) or now
        ids = getattr(alert, "entity_ids", None) or []
        s = f"{getattr(alert, 'alert_type', '?')}  {severity.upper()}  {_age(now - since)}"

        if ids:
            s += "  ids " + ",".join(str(i) for i in ids)

        lines.append((s, dash.SEVERITY_COLOR.get(severity, TEXT)))

    col.rows(lines, ALERT_ROWS_MAX, 2)


# ---------------------------------------------------------------------------
# Bottom event log
# ---------------------------------------------------------------------------
def draw_events(canvas, events, now) -> None:
    x = dash.VIDEO_X + 20
    right = dash.VIDEO_X + dash.VIDEO_W - 20
    top = dash.BOTTOM_Y + EVENT_TOP
    bottom = dash.H - dash.M - 10
    rows = max(0, (bottom - top) // EVENT_ROW_H + 1)
    events = events or []

    if not events:
        wd.text(canvas, "no events yet", (x, top), EVENT_SCALE, DIM)
        return

    sev_x, type_x, msg_x = x + 84, x + 180, x + 430

    for i, event in enumerate(events[:rows]):
        if not isinstance(event, dict):
            continue

        y = top + i * EVENT_ROW_H

        try:
            dt = now - float(event.get("timestamp", now))
        except (TypeError, ValueError):
            dt = 0.0

        stamp = f"-{dt:.1f}s" if dt < 100 else f"-{dt:.0f}s"
        severity = str(event.get("severity", "") or "")
        color = dash.SEVERITY_COLOR.get(severity, TEXT)
        bold = SEVERITY_RANK.get(severity, 0) >= SEVERITY_RANK[WARNING]
        event_type = str(event.get("event_type", "?"))
        message = wd.clean(event.get("message", ""))
        wd.text(canvas, stamp, (x, y), EVENT_SCALE, MUTED)
        wd.text(canvas, severity.upper()[:7], (sev_x, y), EVENT_SCALE, color, 1)
        wd.text(canvas, wd.fit(event_type, msg_x - type_x - 10, EVENT_SCALE), (type_x, y), EVENT_SCALE, color if bold else SUBTLE)
        wd.text(canvas, wd.fit(message, right - msg_x, EVENT_SCALE), (msg_x, y), EVENT_SCALE, TEXT if bold else SUBTLE)

    hidden = len(events) - rows

    if hidden > 0:
        note = f"+{hidden} older"
        wd.text(canvas, note, (right - wd.text_width(note, 0.48), dash.BOTTOM_Y + 30), 0.48, DIM)
