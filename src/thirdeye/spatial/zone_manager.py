"""Zone definitions and debounced zone membership for tracked entities.

Zones are polygons in normalized 0..1 frame coordinates. Each update the
ZoneManager tests every entity's anchor point (center or feet) against every
active zone, debounces the result over ZONE_CONFIRM_FRAMES consecutive
updates, maintains EntityRecord.zones (the confirmed membership set) and
returns ENTER / EXIT transitions. Pure Python, no OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thirdeye.core import settings
from thirdeye.state.world_state import EntityRecord, WorldState

MONITORED = "MONITORED"
RESTRICTED = "RESTRICTED"

ENTER = "ENTER"
EXIT = "EXIT"


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (polygon given as vertex list)."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    id: str
    name: str
    type: str  # "MONITORED" | "RESTRICTED"
    points: list[tuple[float, float]]  # normalized 0..1 polygon
    active: bool = True

    @property
    def is_restricted(self) -> bool:
        return self.type.upper() == RESTRICTED

    def polygon_px(self, frame_size: tuple[int, int]) -> list[tuple[int, int]]:
        w, h = frame_size
        return [(int(round(px * w)), int(round(py * h))) for px, py in self.points]

    def contains(self, point_px: tuple[float, float], frame_size: tuple[int, int]) -> bool:
        w, h = frame_size
        if w <= 0 or h <= 0:
            return False
        # Test in normalized space to avoid rounding the polygon.
        return point_in_polygon(point_px[0] / w, point_px[1] / h, self.points)

    @classmethod
    def from_config(cls, config: dict) -> "Zone":
        points = config.get("points")
        if points is None and "rect" in config:
            # Optional rectangle form: (x1, y1, x2, y2) normalized.
            x1, y1, x2, y2 = config["rect"]
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if not points or len(points) < 3:
            raise ValueError(f"Zone {config.get('id')!r} needs at least 3 points")
        return cls(
            id=str(config["id"]),
            name=str(config.get("name", config["id"])),
            type=str(config.get("type", MONITORED)).upper(),
            points=[(float(p[0]), float(p[1])) for p in points],
            active=bool(config.get("active", True)),
        )


@dataclass
class _Membership:
    """Debounce state for one (entity, zone) pair."""

    confirmed: bool = False
    streak: int = 0  # consecutive updates where raw state != confirmed


class ZoneManager:
    def __init__(self, zone_configs: list[dict] | None = None) -> None:
        configs = settings.ZONES if zone_configs is None else zone_configs
        self.zones: list[Zone] = [Zone.from_config(c) for c in configs]
        self._state: dict[tuple[int, str], _Membership] = {}

    def get(self, zone_id: str | None) -> Zone | None:
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        return None

    def active_zones(self) -> list[Zone]:
        return [z for z in self.zones if z.active]

    def anchor_point(self, record: EntityRecord) -> tuple[float, float]:
        if str(settings.ZONE_ANCHOR).lower() == "bottom":
            return record.bottom_center
        return record.center

    def update(self, world: WorldState) -> list[tuple[str, int, str]]:
        """Advance debounce state one frame; return (kind, entity_id, zone_id)."""
        transitions: list[tuple[str, int, str]] = []
        confirm = max(1, int(settings.ZONE_CONFIRM_FRAMES))
        frame_size = world.frame_size

        # Entities that left the world: exit every confirmed zone, drop state.
        for record in world.disappeared:
            for zone_id in sorted(record.zones):
                transitions.append((EXIT, record.track_id, zone_id))
            record.zones.clear()
            self._drop_entity(record.track_id)

        for track_id, record in world.entities.items():
            anchor = self.anchor_point(record)
            for zone in self.zones:
                key = (track_id, zone.id)
                if not zone.active:
                    # Deactivated zone: release any confirmed membership.
                    state = self._state.pop(key, None)
                    if zone.id in record.zones:
                        record.zones.discard(zone.id)
                        transitions.append((EXIT, track_id, zone.id))
                    elif state is not None and state.confirmed:
                        transitions.append((EXIT, track_id, zone.id))
                    continue

                state = self._state.get(key)
                if state is None:
                    state = _Membership(confirmed=zone.id in record.zones)
                    self._state[key] = state

                raw_inside = zone.contains(anchor, frame_size)
                if raw_inside == state.confirmed:
                    state.streak = 0
                    continue

                state.streak += 1
                if state.streak >= confirm:
                    state.confirmed = raw_inside
                    state.streak = 0
                    if raw_inside:
                        record.zones.add(zone.id)
                        transitions.append((ENTER, track_id, zone.id))
                    else:
                        record.zones.discard(zone.id)
                        transitions.append((EXIT, track_id, zone.id))

        # Prune state for IDs no longer in the world (e.g. after a reset).
        live = world.entities
        for key in [k for k in self._state if k[0] not in live]:
            del self._state[key]

        return transitions

    def _drop_entity(self, track_id: int) -> None:
        for key in [k for k in self._state if k[0] == track_id]:
            del self._state[key]
