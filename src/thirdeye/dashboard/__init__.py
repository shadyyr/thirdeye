"""Local dashboard state and optional FastAPI application."""

from thirdeye.dashboard.app import create_app
from thirdeye.dashboard.state import DashboardState

__all__ = ["DashboardState", "create_app"]
