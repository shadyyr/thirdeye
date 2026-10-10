"""Thread-safe state bridge between the local pipeline and dashboard."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


class DashboardState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: dict[str, Any] = {
            "status": "waiting_for_pipeline",
            "cameras": {},
            "tracked_objects": [],
            "active_haptic": {"zone": "STOP", "intensity": 0},
            "metrics": {"fps": 0.0, "latency_ms": 0.0},
        }

    def update(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = deepcopy(snapshot)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._snapshot)
