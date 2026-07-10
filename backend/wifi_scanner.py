"""
Wi-Fi Scanner module for Windows using netsh commands.
Supports English and Spanish locale output parsing.
"""

import ctypes
import re
import subprocess
import time
import logging
from ctypes import POINTER, Structure, byref, c_void_p, wintypes
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native WlanScan trigger — netsh reads a stale cache. WlanScan forces Windows
# to actually probe the air. Results take ~4s to appear in the cache.
# ---------------------------------------------------------------------------

class _GUID(Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _WLAN_INTERFACE_INFO(Structure):
    _fields_ = [
        ("InterfaceGuid", _GUID),
        ("strInterfaceDescription", wintypes.WCHAR * 256),
        ("isState", wintypes.DWORD),
    ]


class _WLAN_INTERFACE_INFO_LIST(Structure):
    _fields_ = [
        ("dwNumberOfItems", wintypes.DWORD),
        ("dwIndex", wintypes.DWORD),
        ("InterfaceInfo", _WLAN_INTERFACE_INFO * 1),
    ]


def trigger_wlan_scan(wait_seconds: float = 4.0) -> bool:
    """Force Windows to actively scan all WLAN interfaces.

    Returns True if at least one interface was asked to scan.
    Sleeps `wait_seconds` after issuing the scan so the results land in cache.
    """
    try:
        wlanapi = ctypes.WinDLL("wlanapi.dll")
    except (OSError, AttributeError):
        return False

    negotiated = wintypes.DWORD()
    handle = wintypes.HANDLE()
    if wlanapi.WlanOpenHandle(2, None, byref(negotiated), byref(handle)) != 0:
        return False

    try:
        list_ptr = c_void_p()
        if wlanapi.WlanEnumInterfaces(handle, None, byref(list_ptr)) != 0:
            return False

        header = ctypes.cast(list_ptr, POINTER(_WLAN_INTERFACE_INFO_LIST)).contents
        num = header.dwNumberOfItems
        if num == 0:
            wlanapi.WlanFreeMemory(list_ptr)
            return False

        array_addr = ctypes.addressof(header) + _WLAN_INTERFACE_INFO_LIST.InterfaceInfo.offset
        infos = ctypes.cast(array_addr, POINTER(_WLAN_INTERFACE_INFO * num)).contents

        any_scanned = False
        for info in infos:
            rc = wlanapi.WlanScan(handle, byref(info.InterfaceGuid), None, None, None)
            if rc == 0:
                any_scanned = True
            else:
                logger.debug("WlanScan returned %d for interface", rc)

        wlanapi.WlanFreeMemory(list_ptr)

        if any_scanned and wait_seconds > 0:
            time.sleep(wait_seconds)
        return any_scanned
    finally:
        wlanapi.WlanCloseHandle(handle, None)


@dataclass
class AccessPoint:
    """Represents a detected Wi-Fi access point."""
    ssid: str = ""
    bssid: str = ""
    signal_percent: int = 0
    signal_dbm: float = -100.0
    channel: int = 0
    authentication: str = ""
    encryption: str = ""
    network_type: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class WiFiScanner:
    """Scans Wi-Fi networks on Windows using netsh."""

    # English locale regex patterns
    ssid_re = re.compile(r"SSID \d+\s*:\s+(.*)")
    bssid_re = re.compile(r"BSSID \d+\s*:\s+([\dA-Fa-f:]+)")
    signal_re = re.compile(r"Signal\s*:\s+(\d+)%")
    channel_re = re.compile(r"Channel\s*:\s+(\d+)")
    auth_re = re.compile(r"Authentication\s*:\s+(.*)")
    encrypt_re = re.compile(r"Encryption\s*:\s+(.*)")
    network_type_re = re.compile(r"Network type\s*:\s+(.*)")

    # Spanish locale regex patterns
    ssid_es_re = re.compile(r"SSID \d+\s*:\s+(.*)")
    bssid_es_re = re.compile(r"BSSID \d+\s*:\s+([\dA-Fa-f:]+)")
    signal_es_re = re.compile(r"Se[ñn]al\s*:\s+(\d+)%")
    channel_es_re = re.compile(r"Canal\s*:\s+(\d+)")
    auth_es_re = re.compile(r"Autenticaci[oó]n\s*:\s+(.*)")
    encrypt_es_re = re.compile(r"Cifrado\s*:\s+(.*)")
    network_type_es_re = re.compile(r"Tipo de red\s*:\s+(.*)")

    def __init__(self):
        self.last_error: Optional[Dict[str, str]] = None
        self.last_raw_output: str = ""

    @staticmethod
    def signal_percent_to_dbm(percent: int) -> float:
        """Convert signal percentage to approximate dBm value."""
        return (percent / 2.0) - 100.0

    def _run_netsh(self, args: List[str]) -> str:
        """Run a netsh command and return stdout."""
        cmd = ["netsh"] + args
        self.last_error = None
        self.last_raw_output = ""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
            self.last_raw_output = output
            if result.returncode != 0:
                message = output or f"netsh returned exit code {result.returncode}"
                self.last_error = self._classify_netsh_error(message)
                logger.warning("netsh returned code %d: %s", result.returncode, message)
            return output
        except FileNotFoundError:
            self.last_error = {
                "code": "netsh_not_found",
                "message": "No se encontro netsh. El escaneo real solo funciona en Windows.",
                "action": "Ejecuta esta app en Windows o usa el modo simulacion.",
            }
            logger.error("netsh not found – this module requires Windows.")
            return ""
        except subprocess.TimeoutExpired:
            self.last_error = {
                "code": "timeout",
                "message": "El comando netsh tardo demasiado en responder.",
                "action": "Revisa que el adaptador Wi-Fi este activo e intenta nuevamente.",
            }
            logger.error("netsh command timed out.")
            return ""
        except Exception as exc:
            self.last_error = {
                "code": "scan_error",
                "message": f"Error ejecutando netsh: {exc}",
                "action": "Revisa el adaptador Wi-Fi y permisos de Windows.",
            }
            logger.error("Error running netsh: %s", exc)
            return ""

    def _classify_netsh_error(self, message: str) -> Dict[str, str]:
        """Convert common netsh WLAN errors into user-facing diagnostics."""
        lowered = message.lower()
        if "permiso de ubicaci" in lowered or "location permission" in lowered:
            return {
                "code": "location_permission_required",
                "message": "Windows bloqueo el escaneo Wi-Fi por permiso de ubicacion.",
                "action": "Activa Configuracion > Privacidad y seguridad > Ubicacion, y permite acceso de ubicacion para apps de escritorio.",
            }
        if "requiere elevaci" in lowered or "requires elevation" in lowered or "error 5" in lowered:
            return {
                "code": "elevation_required",
                "message": "Windows requiere ejecutar el servidor como administrador para leer la informacion WLAN.",
                "action": "Cierra el servidor y vuelve a iniciarlo desde PowerShell o Terminal como Administrador.",
            }
        if "servicio de configuraci" in lowered or "wlansvc" in lowered:
            return {
                "code": "wlan_service_unavailable",
                "message": "El servicio de Wi-Fi de Windows no esta disponible.",
                "action": "Inicia el servicio Configuracion automatica de WLAN y verifica que el Wi-Fi este encendido.",
            }
        return {
            "code": "netsh_failed",
            "message": message.splitlines()[0][:240],
            "action": "Verifica permisos de Windows, adaptador Wi-Fi y ejecuta el servidor como administrador.",
        }

    def scan_networks(self, force_refresh: bool = False) -> List[AccessPoint]:
        """
        Scan available Wi-Fi networks using `netsh wlan show networks mode=bssid`.
        If force_refresh=True, triggers a native WlanScan first (adds ~4s).
        Returns a list of AccessPoint objects.
        """
        if force_refresh:
            trigger_wlan_scan(wait_seconds=4.0)
        output = self._run_netsh(["wlan", "show", "networks", "mode=bssid"])
        primary_error = self.last_error
        primary_output = self.last_raw_output
        if not output:
            logger.info("No output from netsh scan – Wi-Fi adapter may be unavailable.")
            return []

        aps = self._parse_scan_output(output)
        
        # Fallback: if no APs found with BSSIDs, try to get the connected interface info
        if not aps:
            logger.info("No APs found in scan. Trying connected interface info as fallback.")
            conn = self.get_connected_info()
            if conn and conn.get("bssid"):
                try:
                    pct = int(conn.get("signal_percent", "0"))
                    dbm = float(conn.get("signal_dbm", "-100"))
                except ValueError:
                    pct = 0
                    dbm = -100.0
                
                ap = AccessPoint(
                    ssid=conn.get("ssid", ""),
                    bssid=conn["bssid"].lower(),
                    signal_percent=pct,
                    signal_dbm=dbm,
                    channel=int(conn.get("channel", "0") or "0"),
                    authentication=conn.get("auth", ""),
                    encryption=conn.get("cifrado", ""),
                    network_type=conn.get("radio_type", "")
                )
                aps.append(ap)
                logger.info("Added connected AP as fallback: %s (%s)", ap.ssid, ap.bssid)
            elif primary_error:
                self.last_error = primary_error
                self.last_raw_output = primary_output
                
        return aps

    def get_last_scan_status(self) -> Dict[str, object]:
        """Return diagnostic status for the last scan attempt."""
        return {
            "ok": self.last_error is None,
            "error": self.last_error,
            "raw_output": self.last_raw_output,
        }

    def _parse_scan_output(self, output: str) -> List[AccessPoint]:
        """Parse netsh scan output into AccessPoint list."""
        access_points: List[AccessPoint] = []
        lines = output.splitlines()

        current_ssid = ""
        current_network_type = ""
        current_auth = ""
        current_encrypt = ""
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check SSID (same pattern for EN/ES)
            m = self.ssid_re.match(line) or self.ssid_es_re.match(line)
            if m:
                current_ssid = m.group(1).strip()
                i += 1
                continue

            # Network type
            m = self.network_type_re.match(line) or self.network_type_es_re.match(line)
            if m:
                current_network_type = m.group(1).strip()
                i += 1
                continue

            # Authentication
            m = self.auth_re.match(line) or self.auth_es_re.match(line)
            if m:
                current_auth = m.group(1).strip()
                i += 1
                continue

            # Encryption
            m = self.encrypt_re.match(line) or self.encrypt_es_re.match(line)
            if m:
                current_encrypt = m.group(1).strip()
                i += 1
                continue

            # BSSID – start of a new AP entry
            m = self.bssid_re.match(line) or self.bssid_es_re.match(line)
            if m:
                ap = AccessPoint(
                    ssid=current_ssid,
                    bssid=m.group(1).strip().lower(),
                    authentication=current_auth,
                    encryption=current_encrypt,
                    network_type=current_network_type,
                )

                # Scan ahead for signal and channel belonging to this BSSID
                j = i + 1
                while j < len(lines) and j < i + 10:
                    sub = lines[j].strip()

                    sm = self.signal_re.match(sub) or self.signal_es_re.match(sub)
                    if sm:
                        ap.signal_percent = int(sm.group(1))
                        ap.signal_dbm = self.signal_percent_to_dbm(ap.signal_percent)

                    cm = self.channel_re.match(sub) or self.channel_es_re.match(sub)
                    if cm:
                        ap.channel = int(cm.group(1))

                    # Stop look-ahead if we hit next BSSID or SSID
                    if self.bssid_re.match(sub) or self.ssid_re.match(sub):
                        break
                    j += 1

                access_points.append(ap)
                i += 1
                continue

            i += 1

        logger.info("Scanned %d access points.", len(access_points))
        return access_points

    def get_connected_info(self) -> Optional[Dict]:
        """
        Get info about the currently connected Wi-Fi network.
        Returns dict with ssid, bssid, signal, channel, etc. or None.
        """
        output = self._run_netsh(["wlan", "show", "interfaces"])
        if not output:
            return None

        info: Dict[str, str] = {}

        # Patterns for interface info (EN + ES)
        patterns = {
            "ssid": re.compile(r"^\s*SSID\s*:\s+(.+)", re.MULTILINE),
            "bssid": re.compile(r"^\s*(?:AP\s+)?BSSID\s*:\s+([\dA-Fa-f:]+)", re.MULTILINE),
            "signal": re.compile(r"^\s*(?:Signal|Se.*al)\s*:\s+(\d+)%", re.MULTILINE),
            "channel": re.compile(r"^\s*(?:Channel|Canal)\s*:\s+(\d+)", re.MULTILINE),
            "state": re.compile(r"^\s*(?:State|Estado)\s*:\s+(.+)", re.MULTILINE),
            "auth": re.compile(r"^\s*(?:Authentication|Autenticaci.*)\s*:\s+(.+)", re.MULTILINE),
            "radio_type": re.compile(r"^\s*(?:Radio type|Tipo de radio)\s*:\s+(.+)", re.MULTILINE),
            "profile": re.compile(r"^\s*(?:Profile|Perfil)\s*:\s+(.+)", re.MULTILINE),
            "receive_rate": re.compile(r"^\s*(?:Receive rate|Velocidad de recepci.*)\s*.*:\s+(.+)", re.MULTILINE),
            "transmit_rate": re.compile(r"^\s*(?:Transmit rate|Velocidad de transmisi.*)\s*.*:\s+(.+)", re.MULTILINE),
        }

        for key, pat in patterns.items():
            m = pat.search(output)
            if m:
                info[key] = m.group(1).strip()

        if not info:
            logger.info("No connected Wi-Fi interface found.")
            return None

        # Add dBm conversion
        if "signal" in info:
            try:
                pct = int(info["signal"])
                info["signal_percent"] = str(pct)
                info["signal_dbm"] = str(self.signal_percent_to_dbm(pct))
            except ValueError:
                pass

        return info

    def get_rssi_dict(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Scan and return a dict of {bssid: signal_dbm} for fingerprinting.
        """
        aps = self.scan_networks(force_refresh=force_refresh)
        return {ap.bssid: ap.signal_dbm for ap in aps if ap.bssid}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    scanner = WiFiScanner()

    print("=== Scanning Networks ===")
    networks = scanner.scan_networks()
    for ap in networks:
        print(f"  {ap.ssid:30s}  BSSID={ap.bssid}  Signal={ap.signal_percent}% ({ap.signal_dbm:.1f} dBm)  Ch={ap.channel}")

    print("\n=== Connected Info ===")
    conn = scanner.get_connected_info()
    if conn:
        for k, v in conn.items():
            print(f"  {k}: {v}")
    else:
        print("  Not connected.")
