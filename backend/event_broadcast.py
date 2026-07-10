"""Thin helper for CLI radar modules to publish events to the FastAPI dashboard.

Used by wifi_radar.py, multi_ap_radar.py, ble_radar.py when run with the
--broadcast http://<host>:<port> flag. Fires-and-forgets POSTs in a daemon
thread queue so the radar loops never block on network IO.

Silent-fail by design: if the FastAPI server isn't running, the CLI keeps
working normally; events just don't show in the browser dashboard.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class EventBroadcaster:
    """Background thread that POSTs JSON events to /api/radar/event."""

    def __init__(self, base_url: Optional[str], queue_max: int = 256, timeout: float = 1.5):
        self.base_url = (base_url or "").rstrip("/") or None
        self.timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fail_count = 0
        self._last_warn_at = 0.0
        self._silent_after = 5  # stop printing after N consecutive failures

    def start(self) -> None:
        if self.base_url is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[broadcast] eventos -> {self.base_url}/api/radar/event")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def publish(self, **event: Any) -> None:
        """Enqueue an event for async POST. Non-blocking; drops if full."""
        if self.base_url is None:
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            pass  # drop silently to avoid backpressure on the radar loop

    def _run(self) -> None:
        url = f"{self.base_url}/api/radar/event"
        while not self._stop.is_set():
            try:
                event = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                data = json.dumps(event).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp.read()
                self._fail_count = 0
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                self._fail_count += 1
                if self._fail_count <= self._silent_after:
                    now = time.time()
                    if now - self._last_warn_at > 5.0:
                        print(f"[broadcast] no se pudo enviar evento: {exc}")
                        self._last_warn_at = now
            except Exception as exc:  # noqa: BLE001 broad on purpose - never crash the CLI
                if self._fail_count <= self._silent_after:
                    print(f"[broadcast] error inesperado: {exc}")
                self._fail_count += 1


# -- shared no-op singleton -------------------------------------------------

_GLOBAL: Optional[EventBroadcaster] = None


def init_broadcaster(base_url: Optional[str]) -> EventBroadcaster:
    """Create and start a process-wide broadcaster. Idempotent."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = EventBroadcaster(base_url)
        _GLOBAL.start()
    return _GLOBAL


def publish(**event: Any) -> None:
    """Convenience: publish an event to the global broadcaster, if initialized."""
    if _GLOBAL is not None:
        _GLOBAL.publish(**event)
