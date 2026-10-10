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
      .metrics {
        display: grid; gap: 0.75rem; grid-template-columns: repeat(3, minmax(120px, 1fr));
      }
      .metric-label {
        color: #aab6dc; display: block; font-size: 0.8rem; text-transform: uppercase;
      }
      .metric-value { display: block; font-size: 1.3rem; font-weight: 700; }
      pre { white-space: pre-wrap; word-break: break-word; }
    </style>
  </head>
  <body>
    <h1>Third Eye</h1>
    <p class="flow">CAMERA → DETECT → TRACK → SCORE → HAPTIC</p>
    <div class="card">
      <strong id="status">Connecting…</strong>
      <div class="metrics">
        <div>
          <span class="metric-label">Scenario</span>
          <span class="metric-value" id="scenario">—</span>
        </div>
        <div>
          <span class="metric-label">Active haptic</span>
          <span class="metric-value" id="haptic">STOP · 0</span>
        </div>
        <div>
          <span class="metric-label">Selected event</span>
          <span class="metric-value" id="event">None</span>
        </div>
      </div>
      <pre id="data"></pre>
    </div>
    <script>
      async function refresh() {
        try {
          const response = await fetch('/api/status');
          const data = await response.json();
          document.querySelector('#status').textContent = data.status;
          document.querySelector('#scenario').textContent = data.replay?.scenario ?? 'live input';
          const haptic = data.active_haptic ?? { zone: 'STOP', intensity: 0 };
          document.querySelector('#haptic').textContent = `${haptic.zone} · ${haptic.intensity}`;
          const event = data.selected_event;
          document.querySelector('#event').textContent = event
            ? `${event.zone} · ${event.relevance_score}`
            : 'None';
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
