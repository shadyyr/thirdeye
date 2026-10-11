"""Event-triggered incident clip recorder (experimental, not wired into the app).

Data flow (one incident open at a time):

    main thread                 jpeg worker                  clip writer
    push_frame(frame, now) ---> RollingBuffer (JPEG, ~10 fps,
      (time check + put_nowait)   last ~6 s, byte-capped)
                                  |  on each encoded frame
    on_events(events, now)        v
      trigger? open incident -> _on_encoded: forward frames  -> bounded queue
      + pre-roll from buffer      inside [start, end)            -> decode JPEG,
      later trigger? merge,       close when ts >= end           VideoWriter.write,
      extend end (capped)                                        store.upsert_incident

A trigger opens an incident whose clip starts PRE_ROLL seconds before it
(taken from the buffer) and ends POST_ROLL seconds after the last linked
trigger, never longer than MAX_CLIP seconds. Triggers that arrive while an
incident is open are linked to it instead of starting a new file.

Nothing here raises into the caller: failures (encoder, writer, disk, store)
mark the incident ``failed``, are logged once to stderr, and recording of
later incidents carries on. All queues are bounded; frames are dropped, never
waited for, in real-time mode.

Tunables are read as ``getattr(settings, "INCIDENT_...", default)`` so they
can move into ``core/settings.py`` later; constructor arguments override them.
"""

from __future__ import annotations

import itertools
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from thirdeye.core import settings
from thirdeye.core.types import (
    CROWDING_STARTED,
    DWELLING,
    OBJECT_LEFT_WITH,
    OBJECT_MOVED,
    UNATTENDED_OBJECT,
    ZONE_ENTERED,
)
from thirdeye.recording.rolling_buffer import RollingBuffer

RECORDING = "recording"
COMPLETE = "complete"
FAILED = "failed"

DEFAULT_TRIGGER_TYPES = (
    ZONE_ENTERED,  # only into a RESTRICTED zone
    DWELLING,
    CROWDING_STARTED,
    UNATTENDED_OBJECT,
    OBJECT_MOVED,
    OBJECT_LEFT_WITH,
)

TITLES = {
    ZONE_ENTERED: "RESTRICTED ZONE ENTRY",
    DWELLING: "DWELLING",
    CROWDING_STARTED: "CROWDING",
    UNATTENDED_OBJECT: "UNATTENDED OBJECT",
    OBJECT_MOVED: "OBJECT MOVED",
    OBJECT_LEFT_WITH: "OBJECT LEFT VIEW WITH PERSON",
}


def setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


class _OnceLog:
    """Log the first occurrence of each kind of problem, count the rest."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def __call__(self, key: str, message: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] == 1:
            print(f"[thirdeye.recording] {message} (further '{key}' messages suppressed)",
                  file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Video writers
# ---------------------------------------------------------------------------

def _gst_x264(path: str, bitrate: int) -> str:
    return (
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc speed-preset=ultrafast tune=zerolatency bitrate={int(bitrate)} threads=2 key-int-max=30 ! "
        f'h264parse ! mp4mux ! filesink location="{path}"'
    )


def open_video_writer(
    path: str, fps: float, size: tuple[int, int], codecs: Iterable[str] | None = None, x264_bitrate: int | None = None
):
    """Open the first working codec. Returns (writer, codec, path); raises if none opens.

    Codecs:
      x264  GStreamer x264enc (ultrafast, CBR kbit/s) -> H.264 .mp4 (small, browser-playable)
      mp4v  FFMPEG MPEG-4 Part 2 -> .mp4 (no GStreamer needed; ~5-7x larger)
      avc1  FFMPEG libx264 default preset -> .mp4 (H.264, but ~10x the CPU of x264)
      mjpg  FFMPEG Motion JPEG -> .avi (largest)
    """
    codecs = tuple(codecs or setting("INCIDENT_CODECS", ("x264", "mp4v")))
    bitrate = int(x264_bitrate or setting("INCIDENT_X264_BITRATE_KBPS", 2000))
    base = os.path.splitext(path)[0]
    tried = []
    for codec in codecs:
        target = base + (".avi" if codec == "mjpg" else ".mp4")
        writer = None
        try:
            if codec == "x264":
                writer = cv2.VideoWriter(_gst_x264(os.path.abspath(target), bitrate), cv2.CAP_GSTREAMER, 0,
                                         float(fps), tuple(size), True)
            elif codec in ("mp4v", "avc1", "mjpg"):
                fourcc = cv2.VideoWriter_fourcc(*("MJPG" if codec == "mjpg" else codec))
                writer = cv2.VideoWriter(target, cv2.CAP_FFMPEG, fourcc, float(fps), tuple(size))
            if writer is not None and writer.isOpened():
                return writer, codec, target
        except Exception as exc:  # noqa: BLE001
            tried.append(f"{codec}: {type(exc).__name__}: {exc}")
        else:
            tried.append(f"{codec}: not opened")
        if writer is not None:
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        if os.path.exists(target) and os.path.getsize(target) == 0:
            try:
                os.remove(target)
            except OSError:
                pass
    raise RuntimeError("no video writer could be opened (" + "; ".join(tried) + ")")


# ---------------------------------------------------------------------------
# Incident state
# ---------------------------------------------------------------------------

class Incident:
    """One clip. Main thread owns event fields; the writer owns frame fields."""

    def __init__(self, incident_id: str, session_id: str, event: dict, trigger_time: float,
                 pre_roll: float, post_roll: float, max_clip: float, file_path: str) -> None:
        self.incident_id = incident_id
        self.session_id = session_id
        self.event_type = str(event.get("event_type"))
        self.trigger_time = trigger_time
        self.nominal_start = trigger_time - pre_roll
        self.cap_time = self.nominal_start + max_clip
        self.end_time = min(trigger_time + post_roll, self.cap_time)
        self.file_path = file_path
        self.events: list[dict] = []
        self.entity_ids: list[int] = []
        self.zone_id: str | None = None
        self.status = RECORDING
        self.closing = False
        self.last_forwarded_ts = float("-inf")
        self.queued_frames = 0
        self.dropped_frames = 0
        # writer-side
        self.first_frame_ts: float | None = None
        self.last_frame_ts: float | None = None
        self.frame_count = 0
        self.source_frames = 0
        self.repeated_frames = 0
        self.bytes = 0
        self.codec: str | None = None
        self.size: tuple[int, int] | None = None
        self.error: str | None = None
        self.truncated = False
        self.link(event)

    def link(self, event: dict) -> None:
        self.events.append(event)
        for key in ("entity_id", "object_id"):
            value = event.get(key)
            try:
                value = None if value is None else int(value)
            except (TypeError, ValueError):
                value = None
            if value is not None and value not in self.entity_ids:
                self.entity_ids.append(value)
        if self.zone_id is None and event.get("zone_id") is not None:
            self.zone_id = str(event.get("zone_id"))

    def row(self, fps: float) -> dict:
        events = list(self.events)
        start = self.first_frame_ts if self.first_frame_ts is not None else self.nominal_start
        if self.last_frame_ts is not None and fps > 0:
            end = start + self.frame_count / fps
        else:
            end = self.end_time
        return {
            "incident_id": self.incident_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "trigger_time": self.trigger_time,
            "clip_start": start,
            "clip_end": end,
            "event_ids": [str(e.get("event_id")) for e in events if e.get("event_id") is not None],
            "entity_ids": list(self.entity_ids),
            "zone_id": self.zone_id,
            "file_path": self.file_path,
            "frame_count": self.frame_count,
            "bytes": self.bytes,
            "status": self.status,
            "metadata": {
                "codec": self.codec,
                "fps": fps,
                "size": list(self.size) if self.size else None,
                "source_frames": self.source_frames,
                "repeated_frames": self.repeated_frames,
                "dropped_frames": self.dropped_frames,
                "truncated": self.truncated,
                "error": self.error,
                "event_types": [e.get("event_type") for e in events],
            },
            "updated_at": time.time(),
        }

    def summary(self, fps: float) -> dict:
        """JSON-able entry for an incident history list."""
        row = self.row(fps)
        return {
            "incident_id": self.incident_id,
            "title": TITLES.get(self.event_type, self.event_type),
            "event_type": self.event_type,
            "trigger_time": self.trigger_time,
            "duration": round(max(0.0, row["clip_end"] - row["clip_start"]), 2),
            "status": self.status,
            "event_count": len(row["event_ids"]),
            "event_types": row["metadata"]["event_types"],
            "messages": [str(e.get("message") or "") for e in list(self.events)[:5]],
            "entity_ids": row["entity_ids"],
            "zone_id": self.zone_id,
            "file_path": self.file_path,
            "bytes": self.bytes,
        }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class IncidentRecorder:
    """Call ``push_frame`` and ``on_events`` once per frame, ``close`` at exit."""

    def __init__(
        self,
        store=None,
        session_id: str | None = None,
        *,
        out_dir: str | None = None,
        pre_roll: float | None = None,
        post_roll: float | None = None,
        max_clip: float | None = None,
        cooldown: float | None = None,
        fps: float | None = None,
        jpeg_quality: int | None = None,
        size: tuple[int, int] | None | str = "setting",
        max_buffer_bytes: int | None = None,
        trigger_types: Iterable[str] | None = None,
        restricted_zone_ids: Iterable[str] | None = None,
        writer_factory: Callable | None = None,
        realtime: bool = True,
        writer_queue_frames: int | None = None,
        min_free_mb: float | None = None,
        close_timeout: float | None = None,
    ) -> None:
        self.store = store
        self.session_id = str(session_id or getattr(store, "session_id", None)
                              or datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.out_dir = out_dir or setting("INCIDENT_DIR", "data/incidents")
        self.pre_roll = float(setting("INCIDENT_PRE_ROLL_SECONDS", 5.0) if pre_roll is None else pre_roll)
        # 12 s rather than 10 s: DWELLING fires DWELL_THRESHOLD_SECONDS (10 s) after a zone
        # entry, so a 10 s post-roll closes the entry clip just before the dwell event arrives.
        self.post_roll = float(setting("INCIDENT_POST_ROLL_SECONDS", 12.0) if post_roll is None else post_roll)
        self.max_clip = float(setting("INCIDENT_MAX_CLIP_SECONDS", 30.0) if max_clip is None else max_clip)
        self.cooldown = float(setting("INCIDENT_COOLDOWN_SECONDS", 0.0) if cooldown is None else cooldown)
        self.fps = float(setting("INCIDENT_BUFFER_FPS", 10.0) if fps is None else fps)
        quality = int(setting("INCIDENT_JPEG_QUALITY", 80) if jpeg_quality is None else jpeg_quality)
        if size == "setting":
            size = setting("INCIDENT_RESOLUTION", None)
        self.size = tuple(size) if size else None
        max_bytes = int(max_buffer_bytes or setting("INCIDENT_BUFFER_MAX_MB", 24) * 1024 * 1024)
        self.trigger_types = set(trigger_types or setting("INCIDENT_TRIGGER_TYPES", DEFAULT_TRIGGER_TYPES))
        if restricted_zone_ids is None:
            restricted_zone_ids = [
                z.get("id") for z in getattr(settings, "ZONES", []) if str(z.get("type", "")).upper() == "RESTRICTED"
            ]
        self.restricted_zone_ids = set(restricted_zone_ids)
        self.writer_factory = writer_factory or (lambda path, fps, size: open_video_writer(path, fps, size))
        self.realtime = realtime
        self.writer_queue_frames = int(writer_queue_frames or setting("INCIDENT_WRITER_QUEUE_FRAMES", 400))
        self.min_free_bytes = float(setting("INCIDENT_MIN_FREE_MB", 500) if min_free_mb is None else min_free_mb) * 1e6
        self.close_timeout = float(setting("INCIDENT_CLOSE_TIMEOUT_SECONDS", 4.0)
                                   if close_timeout is None else close_timeout)

        self._log = _OnceLog()
        self._lock = threading.Lock()
        self._current: Incident | None = None
        self._last_closed_end: float | None = None
        self._seen_event_ids: deque[str] = deque(maxlen=256)
        self._history: deque[Incident] = deque(maxlen=int(setting("INCIDENT_HISTORY_SIZE", 50)))
        self._counter = itertools.count(1)
        self._closed = False
        self._abandon = threading.Event()
        self.stats = {"triggers": 0, "merged": 0, "incidents": 0, "ignored_cooldown": 0,
                      "failed": 0, "dropped_frames": 0, "errors": 0}

        self._queue: queue.Queue = queue.Queue()  # bounded by the frame counter below
        self._queued_frames = 0
        self._writer_thread = threading.Thread(target=self._writer_loop, name="incident-writer", daemon=True)
        self._writer_thread.start()
        self.buffer = RollingBuffer(
            seconds=self.pre_roll + 1.0,
            fps=self.fps,
            max_bytes=max_bytes,
            jpeg_quality=quality,
            size=self.size,
            threaded=realtime,
            on_encoded=self._on_encoded,
        )

    # ------------------------------------------------------------------
    # Main-thread API (never raises)
    # ------------------------------------------------------------------
    def push_frame(self, frame, now: float) -> None:
        if self._closed:
            return
        try:
            self.buffer.submit(frame, now)
        except Exception as exc:  # noqa: BLE001
            self._fail_soft("push_frame", exc)

    def on_events(self, events: Iterable[dict], now: float) -> None:
        if self._closed:
            return
        try:
            for event in events or ():
                if isinstance(event, dict) and self.is_trigger(event):
                    self._trigger(event, now)
            self._close_if_overdue(now)
        except Exception as exc:  # noqa: BLE001
            self._fail_soft("on_events", exc)

    def is_trigger(self, event: dict) -> bool:
        event_type = event.get("event_type")
        if event_type not in self.trigger_types:
            return False
        if event_type == ZONE_ENTERED:
            zone_type = (event.get("metadata") or {}).get("zone_type")
            if zone_type is not None:
                return str(zone_type).upper() == "RESTRICTED"
            return event.get("zone_id") in self.restricted_zone_ids
        return True

    def history(self, limit: int = 10) -> list[dict]:
        """Newest first, JSON-able (for an incident history list)."""
        try:
            with self._lock:
                items = list(self._history)[-int(limit):]
            return [inc.summary(self.fps) for inc in reversed(items)]
        except Exception as exc:  # noqa: BLE001
            self._fail_soft("history", exc)
            return []

    @property
    def recording(self) -> bool:
        current = self._current
        return current is not None and not current.closing

    def buffer_stats(self) -> dict:
        return {"frames": len(self.buffer), "bytes": self.buffer.total_bytes,
                "span_s": round(self.buffer.span(), 2), **self.buffer.stats}

    def close(self) -> None:
        """Flush the open clip (truncated at the last buffered frame) and stop within close_timeout."""
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + self.close_timeout
        current = None
        try:
            self.buffer.drain(timeout=min(0.5, self.close_timeout / 4))
            self.buffer.close(timeout=0.5)
            with self._lock:
                current = self._current
                if current is not None and not current.closing:
                    current.truncated = True
                    self._close_locked(current)
            self._queue.put(None)
            self._writer_thread.join(timeout=max(0.1, deadline - time.monotonic() - 1.0))
            if self._writer_thread.is_alive():
                self._abandon.set()  # writer skips remaining frames and finalises
                self._writer_thread.join(timeout=max(0.1, deadline - time.monotonic()))
            if self._writer_thread.is_alive() and current is not None and current.status == RECORDING:
                current.status = FAILED
                current.error = "writer did not finish before shutdown"
                self._upsert(current)
                self._log("close", f"incident {current.incident_id}: clip writer did not finish in time; "
                                   "marked failed")
        except Exception as exc:  # noqa: BLE001
            self._fail_soft("close", exc)

    # ------------------------------------------------------------------
    # Incident lifecycle (lock held where noted)
    # ------------------------------------------------------------------
    def _trigger(self, event: dict, now: float) -> None:
        event_id = event.get("event_id")
        if event_id is not None:
            if event_id in self._seen_event_ids:
                return
            self._seen_event_ids.append(event_id)
        try:
            ts = float(event.get("timestamp", now))
        except (TypeError, ValueError):
            ts = now
        self.stats["triggers"] += 1
        preroll: list[tuple[float, bytes]] = []
        with self._lock:
            current = self._current
            if current is not None and not current.closing and current.status != FAILED and ts < current.end_time:
                current.link(event)
                current.end_time = min(max(current.end_time, ts + self.post_roll), current.cap_time)
                self.stats["merged"] += 1
                self._queue.put(("sync", current))
                return
            if current is not None and not current.closing:
                self._close_locked(current)
            if self.cooldown > 0 and self._last_closed_end is not None and ts - self._last_closed_end < self.cooldown:
                self.stats["ignored_cooldown"] += 1
                return
            incident = self._new_incident(event, ts)
            self._current = incident
            self._history.append(incident)
            self.stats["incidents"] += 1
            self._queue.put(("open", incident))
            preroll = self.buffer.frames_since(incident.nominal_start)
            if preroll:
                incident.last_forwarded_ts = preroll[-1][0]
            for frame_ts, data in preroll:
                self._enqueue_frame_locked(incident, frame_ts, data)

    def _new_incident(self, event: dict, ts: float) -> Incident:
        try:
            stamp = datetime.fromtimestamp(ts).strftime("%H%M%S")
        except (OverflowError, OSError, ValueError):
            stamp = "000000"
        incident_id = f"inc-{stamp}-{next(self._counter):03d}"
        kind = str(event.get("event_type") or "event").lower()
        session_dir = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.session_id)
        path = os.path.join(self.out_dir, session_dir, f"{incident_id}_{kind}.mp4")
        return Incident(incident_id, self.session_id, event, ts, self.pre_roll, self.post_roll,
                        self.max_clip, path)

    def _close_locked(self, incident: Incident) -> None:
        incident.closing = True
        if self._current is incident:
            self._current = None
        self._last_closed_end = incident.end_time
        self._queue.put(("close", incident))

    def _close_if_overdue(self, now: float) -> None:
        # Fallback when encoded frames stop arriving (e.g. encoder stalled).
        with self._lock:
            current = self._current
            if current is not None and not current.closing and now >= current.end_time + 1.0:
                self._close_locked(current)

    def _on_encoded(self, ts: float, data: bytes) -> None:
        """Called by the JPEG worker after each buffered frame."""
        with self._lock:
            current = self._current
            if current is None or current.closing or ts <= current.last_forwarded_ts:
                return
            if current.status == FAILED:
                self._close_locked(current)
                return
            if ts >= current.end_time:
                self._close_locked(current)
                return
            current.last_forwarded_ts = ts
            self._enqueue_frame_locked(current, ts, data)

    def _enqueue_frame_locked(self, incident: Incident, ts: float, data: bytes) -> None:
        if self._queued_frames >= self.writer_queue_frames:
            if self.realtime:
                incident.dropped_frames += 1
                self.stats["dropped_frames"] += 1
                return
            # Offline replay: wait for the writer instead of dropping (lock released briefly).
            self._lock.release()
            try:
                while self._queued_frames >= self.writer_queue_frames and self._writer_thread.is_alive():
                    time.sleep(0.002)
            finally:
                self._lock.acquire()
        self._queued_frames += 1
        incident.queued_frames += 1
        self._queue.put(("frame", incident, ts, data))

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------
    def _writer_loop(self) -> None:
        sinks: dict[str, dict] = {}
        while True:
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                break
            kind, incident = msg[0], msg[1]
            try:
                if kind == "frame":
                    with self._lock:
                        self._queued_frames -= 1
                    if incident.status != FAILED and not self._abandon.is_set():
                        self._write_frame(sinks, incident, msg[2], msg[3])
                elif kind == "open":
                    self._open_clip(incident)
                elif kind == "sync":
                    if incident.status == RECORDING:
                        self._upsert(incident)
                elif kind == "close":
                    self._finish_clip(sinks, incident)
            except Exception as exc:  # noqa: BLE001
                self._mark_failed(sinks, incident, exc)
        for state in list(sinks.values()):  # stop without close message
            self._finish_clip(sinks, state["incident"])

    def _open_clip(self, incident: Incident) -> None:
        try:
            directory = os.path.dirname(os.path.abspath(incident.file_path))
            os.makedirs(directory, exist_ok=True)
            free = shutil.disk_usage(directory).free
            if free < self.min_free_bytes:
                raise OSError(f"only {free / 1e6:.0f} MB free in {directory}")
        except Exception as exc:  # noqa: BLE001
            incident.status = FAILED
            incident.error = f"{type(exc).__name__}: {exc}"
            self.stats["failed"] += 1
            self._log("open", f"incident {incident.incident_id} not recorded: {incident.error}")
        self._upsert(incident)

    def _write_frame(self, sinks: dict, incident: Incident, ts: float, data: bytes) -> None:
        state = sinks.get(incident.incident_id)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("could not decode a buffered JPEG frame")
        if state is None:
            size = (image.shape[1], image.shape[0])
            writer, codec, path = self.writer_factory(incident.file_path, self.fps, size)
            incident.codec, incident.size, incident.file_path = codec, size, path
            incident.first_frame_ts = ts
            state = {"incident": incident, "writer": writer, "next_slot": 0, "last": None}
            sinks[incident.incident_id] = state
            self._upsert(incident)
        if (image.shape[1], image.shape[0]) != incident.size:
            image = cv2.resize(image, incident.size, interpolation=cv2.INTER_LINEAR)
        writer = state["writer"]
        # Keep real-time duration: repeat the previous frame over gaps (dropped frames).
        slot = int(round((ts - incident.first_frame_ts) * self.fps))
        if state["last"] is not None and slot > state["next_slot"]:
            repeats = min(slot - state["next_slot"], int(self.fps * 2))
            for _ in range(repeats):
                writer.write(state["last"])
            incident.frame_count += repeats
            incident.repeated_frames += repeats
        writer.write(image)
        state["last"] = image
        state["next_slot"] = max(state["next_slot"], slot) + 1
        incident.frame_count += 1
        incident.source_frames += 1
        incident.last_frame_ts = ts
        if incident.source_frames % max(1, int(self.fps * 2)) == 0:
            free = shutil.disk_usage(os.path.dirname(os.path.abspath(incident.file_path))).free
            if free < self.min_free_bytes / 2:
                raise OSError(f"disk space low ({free / 1e6:.0f} MB free); clip stopped")

    def _release(self, sinks: dict, incident: Incident) -> None:
        state = sinks.pop(incident.incident_id, None)
        if state is not None:
            try:
                state["writer"].release()
            except Exception as exc:  # noqa: BLE001
                if incident.status != FAILED:
                    incident.status = FAILED
                    incident.error = f"release: {type(exc).__name__}: {exc}"
                    self.stats["failed"] += 1
        try:
            incident.bytes = os.path.getsize(incident.file_path) if os.path.exists(incident.file_path) else 0
        except OSError:
            incident.bytes = 0

    def _finish_clip(self, sinks: dict, incident: Incident) -> None:
        self._release(sinks, incident)
        if incident.status == RECORDING:
            if self._abandon.is_set() and incident.queued_frames > incident.source_frames + incident.dropped_frames:
                incident.truncated = True
            if incident.frame_count > 0 and incident.bytes > 0:
                incident.status = COMPLETE
            else:
                incident.status = FAILED
                incident.error = incident.error or (
                    "no frames were buffered" if incident.frame_count == 0 else "clip file is empty"
                )
                self.stats["failed"] += 1
                self._log("empty", f"incident {incident.incident_id} has no usable clip: {incident.error}")
        self._upsert(incident)

    def _mark_failed(self, sinks: dict, incident: Incident, exc: BaseException) -> None:
        self.stats["errors"] += 1
        if incident.status != FAILED:
            incident.status = FAILED
            incident.error = f"{type(exc).__name__}: {exc}"
            self.stats["failed"] += 1
            self._log("writer", f"incident {incident.incident_id} clip failed: {incident.error}")
        try:
            self._release(sinks, incident)
        except Exception:  # noqa: BLE001
            pass
        self._upsert(incident)

    def _upsert(self, incident: Incident) -> None:
        if self.store is None:
            return
        try:
            with self._lock:
                row = incident.row(self.fps)
            self.store.upsert_incident(row)
        except Exception as exc:  # noqa: BLE001
            self._log("store", f"incident store update failed: {type(exc).__name__}: {exc}")

    def _fail_soft(self, where: str, exc: BaseException) -> None:
        self.stats["errors"] += 1
        self._log(where, f"incident recorder {where} error: {type(exc).__name__}: {exc}")
