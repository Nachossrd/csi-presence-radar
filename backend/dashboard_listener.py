"""DashboardListener: background thread that subscribes to the FastAPI WebSocket
and aggregates per-anchor radar state for the camera overlay.

Used by `camera_detector.py radar --listen http://127.0.0.1:8000`. The camera
shows a small panel with the live state of every anchor (BLE, Wi-Fi APs, the
ESP32 CSI radar). When ANY anchor is in motion AND the camera sees zero
people in frame, we paint a big "ALGUIEN TRAS EL MURO" alert.

Silent-fail by design: if the dashboard isn't running, the camera keeps
working normally; the overlay just stays empty.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AnchorState:
    anchor_id: str
    state: str = "unknown"
    source: str = ""
    value: Optional[float] = None
    last_event: float = 0.0
    extra: dict = field(default_factory=dict)
    # Sustained-motion tracking: when did we first see this anchor enter motion?
    motion_started: float = 0.0
    # When the listener last counted this anchor as "confirmed motion"
    confirmed_motion: bool = False


class DashboardListener:
    """Subscribes to the FastAPI /ws WebSocket and tracks per-anchor state.

    Includes a "sustained motion" filter: an anchor must be in motion-like
    state for at least `sustained_secs` continuous seconds before
    `confirmed_motion` is True. This kills the flicker from one-off CSI
    anomalies and reduces false positives.
    """

    MOTION_STATES = {"motion", "moving", "approaching"}

    def __init__(self, base_url: str, stale_secs: float = 8.0,
                 sustained_secs: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.stale_secs = stale_secs
        self.sustained_secs = sustained_secs
        self._states: Dict[str, AnchorState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected: bool = False
        self.error: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        print(f"[listen] suscrito a {self.base_url}/ws")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_states(self) -> List[AnchorState]:
        """Return live anchor states, sorted: motion first, then by recency."""
        now = time.time()
        with self._lock:
            out = [s for s in self._states.values() if now - s.last_event < self.stale_secs]
        out.sort(key=lambda s: (
            0 if s.state in self.MOTION_STATES else 1,
            -s.last_event,
        ))
        return out

    def any_motion(self) -> bool:
        """True if any anchor has CONFIRMED (sustained) motion."""
        return any(s.confirmed_motion for s in self.get_states())

    def motion_anchors(self) -> List[AnchorState]:
        """Only anchors with sustained motion (filters out flicker)."""
        return [s for s in self.get_states() if s.confirmed_motion]

    def _run_thread(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._listen())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[listen] thread error: {self.error}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _listen(self) -> None:
        try:
            import websockets
        except ImportError:
            self.error = "websockets no instalado (pip install websockets)"
            print(f"[listen] {self.error}")
            return

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(ws_url, open_timeout=5.0,
                                              ping_interval=20, ping_timeout=20) as ws:
                    self.connected = True
                    backoff = 1.0
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        except asyncio.TimeoutError:
                            continue
                        self._on_message(raw)
            except Exception as exc:
                self.connected = False
                self.error = str(exc)
                if not self._stop.is_set():
                    await asyncio.sleep(min(backoff, 8.0))
                    backoff = min(backoff * 1.5, 8.0)

    def _on_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return
        if data.get("type") != "radar_event":
            return
        anchor_id = data.get("anchor_id")
        if not anchor_id:
            return
        # Ignore the camera's own presence_snapshot (not a wall-motion source)
        if data.get("kind") == "presence_snapshot":
            return
        now = time.time()
        with self._lock:
            st = self._states.get(anchor_id) or AnchorState(anchor_id=anchor_id)
            new_state = str(data.get("state") or "unknown")
            is_motion_now = new_state in self.MOTION_STATES
            was_motion = st.state in self.MOTION_STATES
            # Sustained-motion bookkeeping
            if is_motion_now and not was_motion:
                st.motion_started = now
                st.confirmed_motion = False
            elif is_motion_now and was_motion:
                if not st.confirmed_motion and (now - st.motion_started) >= self.sustained_secs:
                    st.confirmed_motion = True
            elif not is_motion_now:
                st.motion_started = 0.0
                st.confirmed_motion = False

            st.state = new_state
            st.source = str(data.get("source") or "")
            v = data.get("value")
            st.value = float(v) if isinstance(v, (int, float)) else st.value
            st.extra = dict(data.get("extra") or {})
            st.last_event = data.get("timestamp") or now
            self._states[anchor_id] = st
