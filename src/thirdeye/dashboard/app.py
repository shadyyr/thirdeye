"""Minimal local-only judge-facing dashboard scaffold."""

from __future__ import annotations

from thirdeye.dashboard.state import DashboardState


def create_app(state: DashboardState | None = None):
    """Build the optional FastAPI app without importing it on mock-only runs."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as error:
        raise RuntimeError(
            "FastAPI is optional. Install the dashboard extras: pip install -e '.[dashboard]'"
        ) from error

    dashboard_state = state or DashboardState()
    app = FastAPI(title="Third Eye", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/status")
    def status() -> dict:
        return dashboard_state.snapshot()

    app.state.dashboard_state = dashboard_state
    return app


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Third Eye</title>
    <style>
      body { background: #0b1020; color: #e8eeff; font: 18px system-ui; margin: 0; padding: 2rem; }
      h1 { margin-top: 0; }
      .flow { color: #82aaff; font-weight: 700; }
      .card {
        background: #151c33; border-radius: 12px; padding: 1rem; margin: 1rem 0; max-width: 850px;
      }
      pre { white-space: pre-wrap; word-break: break-word; }
    </style>
  </head>
  <body>
    <h1>Third Eye</h1>
    <p class="flow">CAMERA → DETECT → TRACK → SCORE → HAPTIC</p>
    <div class="card">
      <strong id="status">Connecting…</strong>
      <pre id="data"></pre>
    </div>
    <script>
      async function refresh() {
        try {
          const response = await fetch('/api/status');
          const data = await response.json();
          document.querySelector('#status').textContent = data.status;
          document.querySelector('#data').textContent = JSON.stringify(data, null, 2);
        } catch (error) {
          document.querySelector('#status').textContent = 'Dashboard unavailable';
        }
      }
      refresh();
      setInterval(refresh, 500);
    </script>
  </body>
</html>"""
