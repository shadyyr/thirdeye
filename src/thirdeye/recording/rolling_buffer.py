"""Rolling pre-roll buffer of recent frames, stored as JPEG bytes.

The main thread only calls ``submit(frame, now)``: a timestamp check (the
buffer keeps ~INCIDENT_BUFFER_FPS frames per second out of the ~20 FPS live
stream) and a non-blocking hand-over of the frame reference to a bounded
queue. A worker thread resizes (optional) and JPEG-encodes the frame, then
appends it to a deque bounded by age (seconds) and total bytes. When the
worker is behind, new frames are dropped; the main thread never waits.

The live app never draws on the camera frame (the dashboard resizes it into
its own canvas), so no copy is taken; pass ``copy_frames=True`` otherwise.

``threaded=False`` encodes inline in ``submit`` (tests and offline replay).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from typing import Callable

import cv2


class RollingBuffer:
    def __init__(
        self,
        seconds: float = 6.0,
        fps: float = 10.0,
        max_bytes: int = 16 * 1024 * 1024,
        jpeg_quality: int = 80,
        size: tuple[int, int] | None = None,
        threaded: bool = True,
        queue_size: int = 2,
        copy_frames: bool = False,
        on_encoded: Callable[[float, bytes], None] | None = None,
    ) -> None:
        self.seconds = float(seconds)
        self.fps = float(fps)
        self.interval = 1.0 / self.fps if self.fps > 0 else 0.0
        self.max_bytes = int(max_bytes)
        self.jpeg_quality = int(jpeg_quality)
        self.size = tuple(size) if size else None  # (width, height) or None = native
        self.threaded = threaded
        self.copy_frames = copy_frames
        self.on_encoded = on_encoded
        self._frames: deque[tuple[float, bytes]] = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self._next_due: float | None = None
        self._last_ts: float | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"submitted": 0, "skipped": 0, "dropped": 0, "encoded": 0, "errors": 0, "evicted": 0}
        self._error_logged = False
        if threaded:
            self._thread = threading.Thread(target=self._worker, name="incident-jpeg", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    # Main thread
    # ------------------------------------------------------------------
    def due(self, now: float) -> bool:
        """True if a frame at ``now`` should be kept to reach the buffer FPS."""
        if self._last_ts is not None and now < self._last_ts:
            self._next_due = None  # clock went backwards (new replay); restart
        if self._next_due is None or now >= self._next_due:
            return True
        return False

    def _advance(self, now: float) -> None:
        self._last_ts = now
        if self._next_due is None or now - self._next_due > self.interval:
            self._next_due = now + self.interval  # first frame or a long gap
        else:
            self._next_due += self.interval  # keeps the average rate exact

    def submit(self, frame, now: float) -> bool:
        """Offer a frame; returns True if it was accepted for encoding. Never raises."""
        try:
            if frame is None or not self.due(now):
                self.stats["skipped"] += 1
                return False
            self._advance(now)
            item = (frame.copy() if self.copy_frames else frame, float(now))
            self.stats["submitted"] += 1
            if not self.threaded:
                self._encode_and_add(*item)
                return True
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self.stats["dropped"] += 1
                return False
        except Exception as exc:  # noqa: BLE001
            self._report(exc)
            return False

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                self._encode_and_add(*item)
            finally:
                self._queue.task_done()

    def encode(self, frame) -> bytes | None:
        if self.size and (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
        ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return data.tobytes() if ok else None

    def _encode_and_add(self, frame, ts: float) -> None:
        try:
            data = self.encode(frame)
            if data is None:
                raise RuntimeError("cv2.imencode returned no data")
            self.add(ts, data)
            self.stats["encoded"] += 1
            if self.on_encoded is not None:
                self.on_encoded(ts, data)
        except Exception as exc:  # noqa: BLE001
            self._report(exc)

    def _report(self, exc: BaseException) -> None:
        self.stats["errors"] += 1
        if not self._error_logged:
            self._error_logged = True
            print(f"[thirdeye.recording] frame buffer error: {type(exc).__name__}: {exc} "
                  "(further errors suppressed)", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Storage (thread-safe)
    # ------------------------------------------------------------------
    def add(self, ts: float, data: bytes) -> None:
        with self._lock:
            if self._frames and ts < self._frames[-1][0]:
                self._frames.clear()  # time went backwards; old frames are meaningless
                self._bytes = 0
            self._frames.append((ts, data))
            self._bytes += len(data)
            self._evict(ts)

    def _evict(self, newest: float) -> None:
        while self._frames and (
            newest - self._frames[0][0] > self.seconds or (self._bytes > self.max_bytes and len(self._frames) > 1)
        ):
            _, old = self._frames.popleft()
            self._bytes -= len(old)
            self.stats["evicted"] += 1

    def frames_since(self, since: float) -> list[tuple[float, bytes]]:
        """Buffered (timestamp, jpeg) pairs with timestamp >= since, oldest first."""
        with self._lock:
            return [item for item in self._frames if item[0] >= since]

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._frames)

    def span(self) -> float:
        with self._lock:
            return self._frames[-1][0] - self._frames[0][0] if len(self._frames) > 1 else 0.0

    def drain(self, timeout: float = 1.0) -> None:
        """Wait until queued frames are encoded (tests / shutdown)."""
        if not self.threaded:
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.002)

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=timeout)
