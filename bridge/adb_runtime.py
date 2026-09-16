"""Runtime ADB device discovery for MuMu restarts.

MuMu may keep the instance alive while changing its localhost ADB endpoint.
This module deliberately resolves the endpoint at runtime and only accepts a
configured serial as a preference, never as an unconditional requirement.
"""

from __future__ import annotations

import re
import subprocess
import time
import re


def parse_devices(output: str) -> list[str]:
    devices = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            devices.append(fields[0])
    return devices


def _is_ready(adb: str, serial: str, timeout: float = 3.0) -> bool:
    try:
        result = subprocess.run(
            [adb, "-s", serial, "get-state"], capture_output=True, text=True,
            timeout=timeout, check=False)
        return result.returncode == 0 and result.stdout.strip() == "device"
    except (OSError, subprocess.SubprocessError):
        return False


def _mumu_listener_ports() -> list[int]:
    """Discover MuMu's current host-side ADB listener after a restart."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], capture_output=True,
            text=True, timeout=2.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    ports = []
    mumu_line = False
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            mumu_line = "MuMu" in line or "mumu" in line.lower()
        if mumu_line:
            match = re.search(r":(\d+) \(LISTEN\)$", line)
            if match:
                port = int(match.group(1))
                if 1024 <= port <= 65535 and port not in ports:
                    ports.append(port)
    return ports


def discover_serial(adb: str, preferred: str = "", *, vm_index=None,
                    timeout: float = 3.0) -> str:
    """Return a currently ready serial, preferring the configured instance.

    If the preferred endpoint moved after a restart, a single ready device is
    selected automatically. With multiple ready devices, an explicit
    ``vm_index`` selects the conventional MuMu localhost endpoint only when
    it is actually present; otherwise ambiguity is reported instead of
    risking input to another instance.
    """
    preferred = str(preferred or "").strip()
    if preferred:
        # A MuMu restart commonly drops the ADB server's registration while
        # leaving the same guest port unchanged. Re-register it before
        # deciding that the configured endpoint moved.
        try:
            subprocess.run([adb, "start-server"], capture_output=True,
                           text=True, timeout=timeout, check=False)
            subprocess.run([adb, "connect", preferred], capture_output=True,
                           text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    if preferred and _is_ready(adb, preferred, timeout):
        return preferred
    # MuMu can change 266xx/163xx after a guest restart. Its own host
    # process exposes the new listener even while adb devices is empty.
    for port in _mumu_listener_ports():
        candidate = f"127.0.0.1:{port}"
        try:
            subprocess.run([adb, "connect", candidate], capture_output=True,
                           text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
    try:
        result = subprocess.run([adb, "devices", "-l"], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot enumerate ADB devices: {exc}") from exc
    devices = parse_devices(result.stdout)
    if not devices:
        raise RuntimeError("no ready ADB device; wait for MuMu Android to finish restarting")
    # Do not ever auto-select the separately configured offline engine. MuMu
    # may expose the same guest through both 5555 and a per-instance port;
    # prefer the port nearest the last configured MuMu endpoint.
    try:
        import config
        offline = str(config.SETTINGS.get("full_simulation_serial") or "").strip()
        if offline:
            devices = [serial for serial in devices if serial != offline]
    except ImportError:
        pass
    if len(devices) > 1 and preferred.startswith("127.0.0.1:"):
        try:
            old_port = int(preferred.rsplit(":", 1)[1])
            local = [serial for serial in devices if serial.startswith("127.0.0.1:")]
            ranked = sorted(local, key=lambda serial: abs(int(serial.rsplit(":", 1)[1]) - old_port))
            if ranked and abs(int(ranked[0].rsplit(":", 1)[1]) - old_port) < 4096:
                devices = [ranked[0]]
        except (TypeError, ValueError):
            pass
    if len(devices) == 1:
        return devices[0]
    if vm_index is not None:
        # MuMu's commonly used localhost range is 16384 + index*32, but do
        # not trust the formula unless the candidate is listed and healthy.
        try:
            index = int(vm_index)
        except (TypeError, ValueError):
            index = -1
        candidates = {f"127.0.0.1:{16384 + index * 32}",
                      f"127.0.0.1:{26656 + index * 32}"}
        matched = [serial for serial in devices if serial in candidates]
        if len(matched) == 1:
            return matched[0]
    raise RuntimeError("multiple ready ADB devices; set vm_index or adb_serial to select the online MuMu instance")


def resolve_and_store(adb: str, preferred: str = "", *, vm_index=None,
                      retries: int = 20) -> str:
    """Resolve a serial and update config's runtime value when available."""
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            serial = discover_serial(adb, preferred, vm_index=vm_index, timeout=1.5)
            break
        except RuntimeError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5)
    else:
        raise last_error
    try:
        import config
        config.ADB_SERIAL = serial
    except ImportError:
        pass
    return serial

