"""Runtime ADB device discovery for MuMu restarts.

MuMu may keep the instance alive while changing its localhost ADB endpoint.
This module deliberately resolves the endpoint at runtime and only accepts a
configured serial as a preference, never as an unconditional requirement.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import Iterable


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
    try:
        result = subprocess.run([adb, "devices", "-l"], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot enumerate ADB devices: {exc}") from exc
    devices = parse_devices(result.stdout)
    if not devices:
        raise RuntimeError("no ready ADB device; wait for MuMu Android to finish restarting")
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
                      retries: int = 8) -> str:
    """Resolve a serial and update config's runtime value when available."""
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            serial = discover_serial(adb, preferred, vm_index=vm_index)
            break
        except RuntimeError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(0.75, 0.15 * (attempt + 1)))
    else:
        raise last_error
    try:
        import config
        config.ADB_SERIAL = serial
    except ImportError:
        pass
    return serial

