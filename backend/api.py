"""
FastAPI application – ties together Wi-Fi scanning, localization, digital twin,
and serves the frontend. Includes WebSocket streaming and simulation mode.
"""

import asyncio
import json
import math
import os
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.wifi_scanner import WiFiScanner
from backend.wifi_localizer import WiFiLocalizer
from backend.digital_twin import DigitalTwin
from backend.scanner_3d import HouseScanner3D

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
FINGERPRINT_FILE = DATA_DIR / "fingerprints.json"
MODEL_FILE = DATA_DIR / "model.pkl"
FLOOR_PLAN_FILE = DATA_DIR / "floor_plan.json"
DEVICE_RSSI_FILE = DATA_DIR / "device_rssi.json"
ANCHORS_FILE = DATA_DIR / "anchors.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Module-level state (simple prototype globals)
# ---------------------------------------------------------------------------
wifi_scanner = WiFiScanner()
localizer = WiFiLocalizer()
digital_twin = DigitalTwin()
scanner_3d = HouseScanner3D()

# Tracking state
tracking_active: bool = False
simulation_active: bool = False
simulation_task: Optional[asyncio.Task] = None
tracking_task: Optional[asyncio.Task] = None
device_localizers: Dict[int, WiFiLocalizer] = {}

# Auto-broadcast for anchor distance estimation
wifi_anchor_task: Optional[asyncio.Task] = None
ble_anchor_task: Optional[asyncio.Task] = None
ble_anchor_thread: Optional[Any] = None  # BLERadarThread instance

# Track last event timestamps by source for the system_status endpoint
last_event_by_source: Dict[str, float] = {}
last_camera_snapshot_at: float = 0.0
last_csi_event_at: float = 0.0

# WebSocket connection managers
ws_positions_clients: List[WebSocket] = []
ws_wifi_clients: List[WebSocket] = []

# Simulation state
sim_position: List[float] = [5.0, 3.0]
sim_velocity: List[float] = [0.0, 0.0]
sim_target: Optional[List[float]] = None

TRACKED_WIFI_DEVICES = {
    1: "a8:34:6a:11:22:01",
    2: "b0:5c:da:11:22:02",
    3: "dc:a6:32:11:22:03",
}
MIN_APS_FOR_REAL_MULTIUSER = 3

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FingerprintRequest(BaseModel):
    position: List[float]  # [x, y]


class FloorPlanRequest(BaseModel):
    floor_plan: Dict[str, Any]


class FloorPlanEnvelope(BaseModel):
    floor_plan: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Wi-Fi Radar + Digital Twin",
    description="Indoor localization system using Wi-Fi fingerprinting and 3D digital twin",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Create data directories and load persisted state on startup."""
    global simulation_active

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data dir: %s", DATA_DIR)
    logger.info("Frontend dir: %s", FRONTEND_DIR)

    # Load persisted fingerprints
    if FINGERPRINT_FILE.exists():
        localizer.load_fingerprints(str(FINGERPRINT_FILE))
        logger.info("Loaded fingerprints from disk.")

    # Load persisted model
    if MODEL_FILE.exists():
        localizer.load_model(str(MODEL_FILE))
        logger.info("Loaded model from disk.")

    # Load persisted floor plan
    if FLOOR_PLAN_FILE.exists():
        digital_twin.load_floor_plan_file(str(FLOOR_PLAN_FILE))
        logger.info("Loaded floor plan from disk.")

    logger.info("Backend started. Simulation mode available via /api/simulation/toggle")

    # Auto-broadcast for anchor distance estimation: scans Wi-Fi + listens to BLE
    # passively and publishes RSSI events so the 3D dashboard auto-positions
    # anchors at the correct distance without needing a separate radar CLI.
    global wifi_anchor_task, ble_anchor_task, ble_anchor_thread
    if ANCHORS_FILE.exists():
        wifi_anchor_task = asyncio.create_task(_wifi_anchor_loop())
        logger.info("Wi-Fi auto-broadcast loop started.")
        try:
            from backend.ble_radar import BLERadarThread
            ble_anchor_thread = BLERadarThread()
            ble_anchor_thread.start()
            ble_anchor_task = asyncio.create_task(_ble_anchor_loop())
            logger.info("BLE auto-broadcast loop started.")
        except ImportError:
            logger.warning("bleak no instalado, BLE auto-broadcast deshabilitado.")
        except Exception as exc:
            logger.warning("BLE auto-broadcast no se pudo iniciar: %s", exc)


@app.on_event("shutdown")
async def shutdown_event():
    """Persist state and cancel tasks on shutdown."""
    global tracking_task, simulation_task, wifi_anchor_task, ble_anchor_task, ble_anchor_thread

    if tracking_task and not tracking_task.done():
        tracking_task.cancel()
    if simulation_task and not simulation_task.done():
        simulation_task.cancel()
    if wifi_anchor_task and not wifi_anchor_task.done():
        wifi_anchor_task.cancel()
    if ble_anchor_task and not ble_anchor_task.done():
        ble_anchor_task.cancel()
    if ble_anchor_thread is not None:
        try:
            ble_anchor_thread.stop()
        except Exception:
            pass

    # Persist data
    try:
        localizer.save_fingerprints(str(FINGERPRINT_FILE))
    except Exception as exc:
        logger.error("Error saving fingerprints: %s", exc)

    logger.info("Backend shutting down.")


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "Frontend not found. Place index.html in the frontend/ directory."},
        status_code=404,
    )


# Mount static files AFTER the root route so / still works
try:
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    # Expose data/ for the GLB house model + any other assets
    if DATA_DIR.exists():
        app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
except Exception as exc:
    logger.warning("Could not mount static files: %s", exc)


# ---------------------------------------------------------------------------
# API: System status
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    return {
        "status": "ok",
        "tracking_active": tracking_active,
        "simulation_active": simulation_active,
        "localizer": _json_safe(localizer.get_stats()),
        "scanner_3d": scanner_3d.get_status(),
        "multiuser_tracking": _get_multiuser_preflight(),
        "floor_plan_rooms": len(digital_twin.floor_plan.get("rooms", [])),
        "connected_clients": {
            "positions": len(ws_positions_clients),
            "wifi": len(ws_wifi_clients),
        },
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# API: Wi-Fi scanning
# ---------------------------------------------------------------------------

@app.get("/api/wifi/scan")
async def wifi_scan():
    """Trigger a Wi-Fi scan and return detected access points."""
    try:
        aps = wifi_scanner.scan_networks()
        scan_status = wifi_scanner.get_last_scan_status()
        return {
            "access_points": [ap.to_dict() for ap in aps],
            "count": len(aps),
            "scan_status": scan_status,
            "message": scan_status["error"]["message"] if scan_status.get("error") else None,
            "action": scan_status["error"]["action"] if scan_status.get("error") else None,
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.error("Wi-Fi scan error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/wifi/connected")
async def wifi_connected():
    """Get currently connected Wi-Fi info."""
    info = wifi_scanner.get_connected_info()
    return {"connected": info is not None, "info": info}


# ---------------------------------------------------------------------------
# API: Floor plan
# ---------------------------------------------------------------------------

@app.get("/api/floor-plan")
async def get_floor_plan():
    """Return the current floor plan."""
    return digital_twin.floor_plan


@app.post("/api/floor-plan")
async def save_floor_plan(req: Dict[str, Any]):
    """Save a new floor plan from the editor."""
    floor_plan = req.get("floor_plan", req)
    digital_twin.load_floor_plan(floor_plan)
    digital_twin.save_floor_plan(str(FLOOR_PLAN_FILE))
    return {"success": True, "rooms": len(floor_plan.get("rooms", []))}


# ---------------------------------------------------------------------------
# API: Anchors (Wi-Fi APs, BLE devices, the laptop) — 3D positions for the dashboard
# ---------------------------------------------------------------------------

def _load_anchors_file() -> Dict[str, Any]:
    if not ANCHORS_FILE.exists():
        return {"laptop": {"x": 1.0, "y": 1.2, "z": 1.0, "color": "#ffd60a"},
                "anchors": [], "model": None}
    try:
        with open(ANCHORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load anchors: %s", exc)
        return {"laptop": {"x": 1.0, "y": 1.2, "z": 1.0}, "anchors": [], "model": None}


def _save_anchors_file(data: Dict[str, Any]) -> None:
    ANCHORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANCHORS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.get("/api/anchors")
async def get_anchors():
    """Return persisted anchor positions + model config."""
    return _load_anchors_file()


@app.post("/api/anchors")
async def save_anchors(req: Dict[str, Any]):
    """Save anchors config (drag-to-move from the frontend, or full edit)."""
    _save_anchors_file(req)
    return {"success": True, "anchor_count": len(req.get("anchors", []))}


# ---------------------------------------------------------------------------
# API: Radar events — published by standalone CLIs, broadcast to /ws clients
# ---------------------------------------------------------------------------

class RadarEvent(BaseModel):
    """One radar state-change event for the dashboard."""
    anchor_id: Optional[str] = None        # e.g. "starlink_hotspot" or "tv_living"
    source: Optional[str] = None           # "wifi_rtt" | "wifi_multi_ap" | "ble" | "tag" | "test"
    kind: str = "state"                    # "state" | "motion" | "tag_fired"
    state: Optional[str] = None            # "quiet" | "motion" | "approaching" | "moving" | "calibrating"
    value: Optional[float] = None          # e.g. RSSI dBm, RTT ms, anomaly_rate
    delta: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


@app.post("/api/radar/event")
async def radar_event(ev: RadarEvent):
    """Publish a radar event; broadcast it to all /ws subscribers."""
    now = time.time()
    payload = {
        "type": "radar_event",
        "anchor_id": ev.anchor_id,
        "source": ev.source,
        "kind": ev.kind,
        "state": ev.state,
        "value": ev.value,
        "delta": ev.delta,
        "extra": ev.extra or {},
        "timestamp": now,
    }
    # Track per-source heartbeats for /api/system_status
    if ev.source:
        last_event_by_source[ev.source] = now
    if ev.kind == "presence_snapshot":
        global last_camera_snapshot_at
        last_camera_snapshot_at = now
    if ev.source == "esp32_csi":
        global last_csi_event_at
        last_csi_event_at = now
    await _broadcast_positions(payload)
    return {"success": True, "broadcast_to": len(ws_positions_clients)}


@app.get("/api/system_status")
async def system_status():
    """Lightweight health report for the dashboard.

    Tells the frontend which sources are live and their last-seen ago.
    Each source has a 'fresh' boolean = activity in the last 6 seconds.
    """
    now = time.time()
    FRESH_SECS = 6.0

    def age(ts: float) -> Optional[float]:
        return round(now - ts, 1) if ts else None

    return {
        "now": now,
        "fastapi": {"alive": True},
        "camera": {
            "last_snapshot_ago": age(last_camera_snapshot_at),
            "fresh": (now - last_camera_snapshot_at) < FRESH_SECS if last_camera_snapshot_at else False,
        },
        "esp32_csi": {
            "last_event_ago": age(last_csi_event_at),
            "fresh": (now - last_csi_event_at) < FRESH_SECS if last_csi_event_at else False,
        },
        "ble_autobroadcast": {
            "thread_alive": ble_anchor_thread is not None,
            "last_event_ago": age(last_event_by_source.get("ble", 0)),
            "fresh": (now - last_event_by_source.get("ble", 0)) < FRESH_SECS,
        },
        "wifi_autobroadcast": {
            "task_alive": wifi_anchor_task is not None and not wifi_anchor_task.done(),
            "last_event_ago": age(last_event_by_source.get("wifi_multi_ap", 0)),
            "fresh": (now - last_event_by_source.get("wifi_multi_ap", 0)) < FRESH_SECS,
        },
        "ws_clients": len(ws_positions_clients),
    }


# ---------------------------------------------------------------------------
# API: Fingerprinting
# ---------------------------------------------------------------------------

@app.post("/api/fingerprint")
async def add_fingerprint(req: FingerprintRequest):
    """Take a Wi-Fi scan at the given position and save as fingerprint."""
    rssi = wifi_scanner.get_rssi_dict()

    if not rssi:
        # If no Wi-Fi data (e.g., no adapter), generate simulated data
        logger.warning("No Wi-Fi data available – generating simulated fingerprint.")
        rssi = _generate_simulated_rssi(req.position)

    fp_id = localizer.add_fingerprint(req.position, rssi)
    localizer.save_fingerprints(str(FINGERPRINT_FILE))

    return {
        "success": True,
        "id": fp_id,
        "position": req.position,
        "n_aps": len(rssi),
        "total_fingerprints": len(localizer.fingerprint_db),
    }


@app.get("/api/fingerprints")
async def list_fingerprints():
    """List all saved fingerprints."""
    return {
        "fingerprints": localizer.fingerprint_db,
        "count": len(localizer.fingerprint_db),
    }


@app.delete("/api/fingerprints/{fp_id}")
async def delete_fingerprint(fp_id: str):
    """Delete a fingerprint by ID."""
    removed = localizer.remove_fingerprint(fp_id)
    if removed:
        localizer.save_fingerprints(str(FINGERPRINT_FILE))
        return {"success": True, "message": f"Fingerprint {fp_id} removed."}
    raise HTTPException(status_code=404, detail=f"Fingerprint {fp_id} not found.")


# ---------------------------------------------------------------------------
# API: Localizer
# ---------------------------------------------------------------------------

@app.post("/api/localizer/train")
async def train_localizer():
    """Train the localization model on current fingerprints."""
    result = localizer.train_model()
    if result.get("success"):
        localizer.save_model(str(MODEL_FILE))
    return result


@app.post("/api/localizer/start")
async def start_tracking():
    """Start real-time position tracking."""
    global tracking_active, tracking_task

    if tracking_active:
        return {"message": "Tracking already active."}

    preflight = _get_multiuser_preflight()
    if not preflight["ready"]:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": preflight["message"],
                "action": preflight["action"],
                "preflight": preflight,
            },
        )

    tracking_active = True

    # Launch tracking loop as background task
    tracking_task = asyncio.create_task(_tracking_loop())
    logger.info("Tracking started.")
    return {"success": True, "message": "Tracking started."}


@app.post("/api/localizer/stop")
async def stop_tracking():
    """Stop real-time position tracking."""
    global tracking_active, tracking_task

    tracking_active = False
    if tracking_task and not tracking_task.done():
        tracking_task.cancel()
        tracking_task = None
    logger.info("Tracking stopped.")
    return {"success": True, "message": "Tracking stopped."}


@app.get("/api/localizer/stats")
async def localizer_stats():
    """Get localization model stats."""
    return localizer.get_stats()


# ---------------------------------------------------------------------------
# API: Simulation
# ---------------------------------------------------------------------------

@app.post("/api/simulation/toggle")
async def toggle_simulation():
    """Toggle simulation mode on/off."""
    global simulation_active, simulation_task, tracking_active, tracking_task

    simulation_active = not simulation_active

    if simulation_active:
        # Stop real tracking if active
        if tracking_task and not tracking_task.done():
            tracking_task.cancel()
        tracking_active = True
        simulation_task = asyncio.create_task(_simulation_loop())
        logger.info("Simulation mode ENABLED.")
        return {"simulation": True, "message": "Simulation started."}
    else:
        tracking_active = False
        if simulation_task and not simulation_task.done():
            simulation_task.cancel()
            simulation_task = None
        logger.info("Simulation mode DISABLED.")
        return {"simulation": False, "message": "Simulation stopped."}


@app.get("/api/simulation/status")
async def simulation_status():
    """Get simulation status."""
    return {
        "active": simulation_active,
        "position": sim_position,
        "target": sim_target,
    }


# ---------------------------------------------------------------------------
# WebSocket: Combined streaming (Default for Frontend)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_combined(websocket: WebSocket):
    """Stream both position and Wi-Fi data to unified frontend clients."""
    await websocket.accept()
    ws_positions_clients.append(websocket)
    ws_wifi_clients.append(websocket)
    logger.info("Combined WS client connected. Total position clients: %d, wifi clients: %d", 
                len(ws_positions_clients), len(ws_wifi_clients))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif msg.get("type") == "scan":
                    aps = wifi_scanner.scan_networks()
                    scan_status = wifi_scanner.get_last_scan_status()
                    await websocket.send_json({
                        "type": "scan_result",
                        "access_points": [ap.to_dict() for ap in aps],
                        "count": len(aps),
                        "scan_status": scan_status,
                        "message": scan_status["error"]["message"] if scan_status.get("error") else None,
                        "action": scan_status["error"]["action"] if scan_status.get("error") else None,
                        "timestamp": time.time(),
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Combined WS error: %s", exc)
    finally:
        if websocket in ws_positions_clients:
            ws_positions_clients.remove(websocket)
        if websocket in ws_wifi_clients:
            ws_wifi_clients.remove(websocket)
        logger.info("Combined WS client disconnected.")


# ---------------------------------------------------------------------------
# WebSocket: Position streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/positions")
async def ws_positions(websocket: WebSocket):
    """Stream real-time position data to connected clients."""
    await websocket.accept()
    ws_positions_clients.append(websocket)
    logger.info("Position WS client connected. Total: %d", len(ws_positions_clients))

    try:
        while True:
            # Keep connection alive; client can also send commands
            data = await websocket.receive_text()
            # Handle incoming commands from client
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Position WS error: %s", exc)
    finally:
        if websocket in ws_positions_clients:
            ws_positions_clients.remove(websocket)
        logger.info("Position WS client disconnected. Total: %d", len(ws_positions_clients))


# ---------------------------------------------------------------------------
# WebSocket: Wi-Fi streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/wifi")
async def ws_wifi(websocket: WebSocket):
    """Stream live Wi-Fi scan results."""
    await websocket.accept()
    ws_wifi_clients.append(websocket)
    logger.info("WiFi WS client connected. Total: %d", len(ws_wifi_clients))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "scan":
                    # Trigger immediate scan
                    aps = wifi_scanner.scan_networks()
                    scan_status = wifi_scanner.get_last_scan_status()
                    await websocket.send_json({
                        "type": "scan_result",
                        "access_points": [ap.to_dict() for ap in aps],
                        "count": len(aps),
                        "scan_status": scan_status,
                        "message": scan_status["error"]["message"] if scan_status.get("error") else None,
                        "action": scan_status["error"]["action"] if scan_status.get("error") else None,
                        "timestamp": time.time(),
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WiFi WS error: %s", exc)
    finally:
        if websocket in ws_wifi_clients:
            ws_wifi_clients.remove(websocket)
        logger.info("WiFi WS client disconnected. Total: %d", len(ws_wifi_clients))


# ---------------------------------------------------------------------------
# Background: broadcast helpers
# ---------------------------------------------------------------------------

async def _broadcast_positions(data: Dict[str, Any]):
    """Broadcast position data to all connected position WebSocket clients."""
    if not ws_positions_clients:
        return

    message = json.dumps(data)
    disconnected = []

    for ws in ws_positions_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in ws_positions_clients:
            ws_positions_clients.remove(ws)


async def _broadcast_wifi(data: Dict[str, Any]):
    """Broadcast Wi-Fi data to all connected Wi-Fi WebSocket clients."""
    if not ws_wifi_clients:
        return

    message = json.dumps(data)
    disconnected = []

    for ws in ws_wifi_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in ws_wifi_clients:
            ws_wifi_clients.remove(ws)


# ---------------------------------------------------------------------------
# Background: tracking loop (real Wi-Fi)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Anchor auto-broadcast: continuously feed the 3D dashboard with RSSI per
# anchor so it can auto-position by distance. Independent from /api/localizer.
# ---------------------------------------------------------------------------

def _build_bssid_anchor_map() -> Dict[str, str]:
    cfg = _load_anchors_file()
    out: Dict[str, str] = {}
    for a in cfg.get("anchors", []):
        if a.get("type") == "wifi_ap":
            bssid = (a.get("match") or {}).get("bssid", "").lower()
            if bssid:
                out[bssid] = a["id"]
    return out


def _build_mac_anchor_map() -> Dict[str, str]:
    cfg = _load_anchors_file()
    out: Dict[str, str] = {}
    for a in cfg.get("anchors", []):
        if a.get("type", "").startswith("ble"):
            mac = (a.get("match") or {}).get("mac", "").lower()
            if mac:
                out[mac] = a["id"]
    return out


async def _wifi_anchor_loop():
    """Scan Wi-Fi every N seconds, broadcast signal_pct for known anchors."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.sleep(5.0)
            bssid_map = _build_bssid_anchor_map()
            if not bssid_map:
                continue
            # scan_networks is sync + blocking (~2s); run off the event loop
            aps = await loop.run_in_executor(None, wifi_scanner.scan_networks)
            for ap in aps:
                aid = bssid_map.get(ap.bssid.lower())
                if not aid:
                    continue
                await _broadcast_positions({
                    "type": "radar_event",
                    "anchor_id": aid,
                    "source": "wifi_multi_ap",
                    "kind": "state",
                    "state": "quiet",
                    "value": int(ap.signal_percent),
                    "extra": {"ssid": ap.ssid},
                    "timestamp": time.time(),
                })
        except asyncio.CancelledError:
            logger.info("Wi-Fi anchor loop cancelled.")
            return
        except Exception as exc:
            logger.warning("wifi anchor loop: %s", exc)


async def _ble_anchor_loop():
    """Read BLE thread state every 2s, broadcast RSSI for known anchors."""
    while True:
        try:
            await asyncio.sleep(2.0)
            if ble_anchor_thread is None:
                continue
            mac_map = _build_mac_anchor_map()
            if not mac_map:
                continue
            states = ble_anchor_thread.get_states()
            for mac, st in states.items():
                aid = mac_map.get(mac.lower())
                if not aid:
                    continue
                await _broadcast_positions({
                    "type": "radar_event",
                    "anchor_id": aid,
                    "source": "ble",
                    "kind": "state",
                    "state": getattr(st, "state", "quiet"),
                    "value": int(getattr(st, "rssi", -100)),
                    "delta": float(getattr(st, "delta", 0.0)),
                    "extra": {"label": getattr(st, "label", "")},
                    "timestamp": time.time(),
                })
        except asyncio.CancelledError:
            logger.info("BLE anchor loop cancelled.")
            return
        except Exception as exc:
            logger.warning("ble anchor loop: %s", exc)


async def _tracking_loop():
    """Continuously scan Wi-Fi, predict position, and broadcast."""
    global tracking_active

    logger.info("Tracking loop started.")
    try:
        while tracking_active:
            device_rssi = _get_tracked_device_rssi()

            if device_rssi and localizer.is_trained:
                latest_aps = []

                for person_id, rssi in device_rssi.items():
                    if not rssi:
                        continue

                    person_localizer = _get_person_localizer(person_id)
                    pred = person_localizer.predict_position(rssi)

                    digital_twin.update_position(
                        str(person_id),
                        [pred["x"], pred["y"]],
                        pred["confidence"],
                    )

                    latest_aps.extend(
                        {
                            "ssid": f"Device {person_id}",
                            "bssid": bssid,
                            "signal_percent": max(0, min(100, int((dbm + 100) * 2))),
                            "signal_dbm": round(dbm, 1),
                            "channel": 0,
                            "authentication": "",
                            "encryption": "",
                            "network_type": "tracked-device",
                            "device_id": person_id,
                            "device_mac": TRACKED_WIFI_DEVICES.get(person_id),
                        }
                        for bssid, dbm in rssi.items()
                    )

                state = digital_twin.get_persons_state()
                await _broadcast_positions(state)

                await _broadcast_wifi({
                    "type": "scan_result",
                    "access_points": latest_aps,
                    "count": len(latest_aps),
                    "timestamp": time.time(),
                })
            elif not device_rssi:
                digital_twin.persons.clear()
                await _broadcast_positions(digital_twin.get_persons_state())
                await _broadcast_wifi({
                    "type": "scan_result",
                    "access_points": [],
                    "count": 0,
                    "scan_status": {
                        "ok": False,
                        "error": {
                            "code": "insufficient_radio_map",
                            "message": "Datos insuficientes para multiusuario real.",
                            "action": (
                                f"Se requieren al menos {MIN_APS_FOR_REAL_MULTIUSER} APs por dispositivo "
                                "o un archivo data/device_rssi.json con RSSI etiquetado por telefono."
                            ),
                        },
                    },
                    "timestamp": time.time(),
                })
                logger.debug("No Wi-Fi data in tracking loop.")

            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        logger.info("Tracking loop cancelled.")
    except Exception as exc:
        logger.error("Tracking loop error: %s", exc)
    finally:
        tracking_active = False
        logger.info("Tracking loop ended.")


# ---------------------------------------------------------------------------
# Multi-device RSSI ingestion
# ---------------------------------------------------------------------------

def _get_person_localizer(person_id: int) -> WiFiLocalizer:
    """Return a localizer with independent Kalman state for each tracked person."""
    person_localizer = device_localizers.get(person_id)
    if person_localizer is None:
        person_localizer = WiFiLocalizer()
        device_localizers[person_id] = person_localizer

    person_localizer.fingerprint_db = localizer.fingerprint_db
    person_localizer.ap_list = list(localizer.ap_list)
    person_localizer.model = localizer.model
    person_localizer.scaler = localizer.scaler
    person_localizer.is_trained = localizer.is_trained
    person_localizer.accuracy = localizer.accuracy
    return person_localizer


def _get_multiuser_preflight() -> Dict[str, Any]:
    injected = _load_injected_device_rssi()
    valid_devices = {
        person_id: device_mac
        for person_id, device_mac in TRACKED_WIFI_DEVICES.items()
        if len(injected.get(device_mac.lower(), {})) >= MIN_APS_FOR_REAL_MULTIUSER
    }

    if not localizer.is_trained:
        return {
            "ready": False,
            "mode": "blocked",
            "message": "El modelo de localizacion no esta entrenado.",
            "action": "Recolecta fingerprints reales y entrena el modelo antes de iniciar tracking.",
            "required_devices": len(TRACKED_WIFI_DEVICES),
            "valid_devices": len(valid_devices),
            "required_aps_per_device": MIN_APS_FOR_REAL_MULTIUSER,
            "model_aps": len(localizer.ap_list),
            "input": "data/device_rssi.json",
        }

    if len(localizer.ap_list) < MIN_APS_FOR_REAL_MULTIUSER:
        return {
            "ready": False,
            "mode": "blocked",
            "message": "El radio map tiene muy pocos APs para multiusuario real.",
            "action": (
                f"Recolecta fingerprints con al menos {MIN_APS_FOR_REAL_MULTIUSER} APs distintos. "
                "Con solo el router Movistar visto por el notebook no se pueden separar personas."
            ),
            "required_devices": len(TRACKED_WIFI_DEVICES),
            "valid_devices": len(valid_devices),
            "required_aps_per_device": MIN_APS_FOR_REAL_MULTIUSER,
            "model_aps": len(localizer.ap_list),
            "input": "data/device_rssi.json",
        }

    if len(valid_devices) < len(TRACKED_WIFI_DEVICES):
        return {
            "ready": False,
            "mode": "blocked",
            "message": "Faltan lecturas RSSI etiquetadas por telefono.",
            "action": (
                "Alimenta data/device_rssi.json con una entrada por cada telefono "
                f"y al menos {MIN_APS_FOR_REAL_MULTIUSER} BSSIDs por telefono."
            ),
            "required_devices": len(TRACKED_WIFI_DEVICES),
            "valid_devices": len(valid_devices),
            "required_aps_per_device": MIN_APS_FOR_REAL_MULTIUSER,
            "model_aps": len(localizer.ap_list),
            "input": "data/device_rssi.json",
        }

    return {
        "ready": True,
        "mode": "real_multi_device",
        "message": "Tracking multiusuario listo con RSSI etiquetado por dispositivo.",
        "action": None,
        "required_devices": len(TRACKED_WIFI_DEVICES),
        "valid_devices": len(valid_devices),
        "required_aps_per_device": MIN_APS_FOR_REAL_MULTIUSER,
        "model_aps": len(localizer.ap_list),
        "input": "data/device_rssi.json",
    }


def _get_tracked_device_rssi() -> Dict[int, Dict[str, float]]:
    """
    Load per-device RSSI only. Real multi-user tracking requires each tracked
    phone/device to report its own AP RSSI vector.

    Real ingestion format in data/device_rssi.json:
    {
      "a8:34:6a:11:22:01": {"aa:bb:cc:dd:ee:01": -45, "...": -70},
      "b0:5c:da:11:22:02": {"aa:bb:cc:dd:ee:01": -67, "...": -48}
    }

    A normal laptop Wi-Fi scan sees access points, not phone-client RSSI. It
    cannot split one distorted RSSI stream into multiple people, so this never
    fabricates users from the notebook scan.
    """
    injected = _load_injected_device_rssi()
    tracked: Dict[int, Dict[str, float]] = {}

    for person_id, device_mac in TRACKED_WIFI_DEVICES.items():
        rssi = injected.get(device_mac.lower())
        if rssi and len(rssi) >= MIN_APS_FOR_REAL_MULTIUSER:
            tracked[person_id] = rssi

    if len(tracked) < len(TRACKED_WIFI_DEVICES):
        logger.warning(
            "Real multi-user tracking blocked: have RSSI for %d/%d devices in %s.",
            len(tracked),
            len(TRACKED_WIFI_DEVICES),
            DEVICE_RSSI_FILE,
        )
    return tracked


def _load_injected_device_rssi() -> Dict[str, Dict[str, float]]:
    if not DEVICE_RSSI_FILE.exists():
        return {}

    try:
        with open(DEVICE_RSSI_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Could not load device RSSI file: %s", exc)
        return {}

    normalized: Dict[str, Dict[str, float]] = {}
    for mac, readings in raw.items():
        if not isinstance(readings, dict):
            continue
        normalized[mac.lower()] = {
            str(bssid).lower(): float(dbm)
            for bssid, dbm in readings.items()
            if _is_number(dbm)
        }
    return normalized


def _offset_rssi_profile(base_rssi: Dict[str, float], person_id: int) -> Dict[str, float]:
    offsets = {
        1: [6.0, -4.0, -9.0, -2.0, -7.0],
        2: [-7.0, 5.0, -3.0, -8.0, -1.0],
        3: [-4.0, -8.0, 6.0, 2.0, -6.0],
    }
    profile = {}
    for index, (bssid, dbm) in enumerate(sorted(base_rssi.items())):
        offset = offsets[person_id][index % len(offsets[person_id])]
        noise = random.uniform(-2.0, 2.0)
        profile[bssid] = round(max(-100.0, min(-20.0, dbm + offset + noise)), 1)
    return profile


def _simulated_person_position(person_id: int) -> List[float]:
    rooms = digital_twin.floor_plan.get("rooms", [])
    if rooms:
        room = rooms[(person_id - 1) % len(rooms)]
        center = room.get("center", {})
        return [float(center.get("x", person_id * 2.0)), float(center.get("z", person_id * 2.0))]
    return [[3.0, 3.0], [8.0, 2.0], [3.0, 8.5]][person_id - 1]


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Background: simulation loop
# ---------------------------------------------------------------------------

async def _simulation_loop():
    """Simulate a person walking through rooms with smooth random walk."""
    global sim_position, sim_velocity, sim_target, simulation_active, tracking_active

    logger.info("Simulation loop started.")

    # Room centers for waypoint navigation
    floor_plan = digital_twin.floor_plan
    room_centers = []
    for room in floor_plan.get("rooms", []):
        c = room.get("center", {})
        room_centers.append([c.get("x", 5.0), c.get("z", 5.0)])

    if not room_centers:
        room_centers = [[3.0, 3.0], [8.0, 2.0], [3.0, 8.5], [8.0, 8.5], [8.0, 5.0]]

    # Initialize simulation
    sim_position = list(room_centers[0])
    sim_target = list(random.choice(room_centers))
    speed = 0.5  # meters per update (at 2s intervals, ~0.25 m/s)

    try:
        while simulation_active:
            # Pick new target if close to current target
            dx = sim_target[0] - sim_position[0]
            dy = sim_target[1] - sim_position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.5:
                # Pick a different room
                candidates = [rc for rc in room_centers if rc != sim_target]
                if candidates:
                    sim_target = list(random.choice(candidates))
                else:
                    sim_target = list(random.choice(room_centers))
                dx = sim_target[0] - sim_position[0]
                dy = sim_target[1] - sim_position[1]
                dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0.01:
                # Move toward target with some noise
                move_dist = min(speed, dist)
                nx = dx / dist
                ny = dy / dist

                # Add small random perturbation
                noise_x = random.gauss(0, 0.1)
                noise_y = random.gauss(0, 0.1)

                sim_position[0] += nx * move_dist + noise_x
                sim_position[1] += ny * move_dist + noise_y

                sim_velocity = [round(nx * move_dist, 4), round(ny * move_dist, 4)]

            # Clamp to floor plan bounds
            dims = floor_plan.get("dimensions", {"width": 10, "length": 11})
            sim_position[0] = max(0.2, min(dims["width"] - 0.2, sim_position[0]))
            sim_position[1] = max(0.2, min(dims["length"] - 0.2, sim_position[1]))

            # Compute a fake confidence (higher when near room centers)
            min_dist_to_center = min(
                math.sqrt((sim_position[0] - rc[0]) ** 2 + (sim_position[1] - rc[1]) ** 2)
                for rc in room_centers
            )
            confidence = max(0.3, min(1.0, 1.0 - min_dist_to_center / 5.0))

            # Update digital twin
            digital_twin.update_position(
                "sim_user",
                [round(sim_position[0], 3), round(sim_position[1], 3)],
                round(confidence, 3),
            )

            # Broadcast position state
            state = digital_twin.get_persons_state()
            await _broadcast_positions(state)

            # Generate simulated Wi-Fi data and broadcast
            sim_rssi = _generate_simulated_rssi(sim_position)
            sim_aps = [
                {
                    "ssid": f"SimNet_{i}",
                    "bssid": bssid,
                    "signal_percent": max(0, min(100, int((dbm + 100) * 2))),
                    "signal_dbm": dbm,
                    "channel": (i % 11) + 1,
                    "authentication": "WPA2-Personal",
                    "encryption": "CCMP",
                    "network_type": "Infrastructure",
                }
                for i, (bssid, dbm) in enumerate(sim_rssi.items())
            ]
            await _broadcast_wifi({
                "type": "scan_result",
                "access_points": sim_aps,
                "count": len(sim_aps),
                "timestamp": time.time(),
                "simulated": True,
            })

            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        logger.info("Simulation loop cancelled.")
    except Exception as exc:
        logger.error("Simulation loop error: %s", exc)
    finally:
        simulation_active = False
        tracking_active = False
        logger.info("Simulation loop ended.")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

# Simulated router positions (matching default floor plan)
_SIM_ROUTERS = {
    "aa:bb:cc:dd:ee:01": {"x": 5.0, "z": 3.0, "name": "Router Principal"},
    "aa:bb:cc:dd:ee:02": {"x": 3.0, "z": 8.5, "name": "Router Habitación 1"},
    "aa:bb:cc:dd:ee:03": {"x": 8.0, "z": 2.0, "name": "Router Cocina"},
    "aa:bb:cc:dd:ee:04": {"x": 8.0, "z": 8.5, "name": "Router Habitación 2"},
    "aa:bb:cc:dd:ee:05": {"x": 1.0, "z": 1.0, "name": "Vecino WiFi"},
}


def _generate_simulated_rssi(position: List[float]) -> Dict[str, float]:
    """Generate simulated RSSI values based on distance to virtual routers."""
    rssi = {}
    for bssid, router in _SIM_ROUTERS.items():
        dx = position[0] - router["x"]
        dy = position[1] - router["z"]
        distance = math.sqrt(dx * dx + dy * dy)

        # Free-space path loss model (simplified)
        # RSSI ≈ -20 * log10(distance) - 40 + noise
        if distance < 0.1:
            distance = 0.1
        base_rssi = -20.0 * math.log10(distance) - 40.0
        noise = random.gauss(0, 2.0)
        dbm = max(-100.0, min(-20.0, base_rssi + noise))
        rssi[bssid] = round(dbm, 1)

    return rssi


# ---------------------------------------------------------------------------
# Mount frontend static files (fallback for JS, CSS, images, etc.)
# ---------------------------------------------------------------------------
# Note: Static mount is done above after the root route definition.
# If you need to serve additional static asset directories, add them here.


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
