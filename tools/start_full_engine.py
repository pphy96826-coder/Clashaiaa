"""Install the prepared offline APK and start its native engine lane."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def find_apk(downloads: Path) -> Path:
    candidates = sorted(downloads.glob("FirstLight-cr-ai-offline.apk*"),
                       key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            with zipfile.ZipFile(candidate) as archive:
                names = set(archive.namelist())
            if any(name.endswith("/libcrprobe.so") for name in names):
                return candidate
        except (OSError, zipfile.BadZipFile):
            continue
    raise FileNotFoundError("下载目录中没有找到包含 libcrprobe.so 的 FirstLight 离线 APK")


def adb_shell(adb: Path, serial: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run([str(adb), "-s", serial, "shell", *args],
                            check=False, text=True, capture_output=True)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail or f"adb shell failed: {' '.join(args)}")
    return result


def installed_app_lib_dir(adb: Path, serial: str) -> str:
    result = adb_shell(adb, serial, "pm", "path", "nullsroyale.rel.free")
    for line in result.stdout.splitlines():
        path = line.removeprefix("package:").strip()
        if path.endswith("/base.apk"):
            return path.removesuffix("/base.apk") + "/lib/arm64"
    raise RuntimeError("离线副本未安装，无法定位 libcrprobe.so")


def install_matching_probe(adb: Path, serial: str, root: Path) -> str:
    """Install the probe built for this engine protocol into the clone only.

    The downloaded APK can contain a same-named probe from a different build.
    The engine repository's prebuilt is the authoritative companion to its
    Python client, so verify and overlay it after APK installation.  This is
    intentionally scoped to the caller-supplied serial (the offline clone).
    """
    snapshot_probe = root / "build" / "libcrprobe-snapshot.so"
    if snapshot_probe.is_file():
        probe = snapshot_probe
        expected_file = None
        print("[engine] using locally rebuilt snapshot-capable probe", flush=True)
    else:
        probe = root / "prebuilt" / "libcrprobe.so"
        expected_file = root / "prebuilt" / "sha256.txt"
    if not probe.is_file():
        raise FileNotFoundError(f"匹配的 probe 不存在: {probe}")
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    expected = expected_file.read_text(encoding="utf-8").split()[0] if expected_file and expected_file.is_file() else digest
    if digest != expected:
        raise RuntimeError(f"仓库 prebuilt probe 校验失败: {digest} != {expected}")

    lib_dir = installed_app_lib_dir(adb, serial)
    remote = "/data/local/tmp/royale-engine-libcrprobe.so"
    pushed = subprocess.run([str(adb), "-s", serial, "push", str(probe), remote],
                            check=False, text=True, capture_output=True)
    if pushed.returncode:
        raise RuntimeError(pushed.stderr.strip() or "无法传输匹配的 probe")
    adb_shell(adb, serial, "am", "force-stop", "nullsroyale.rel.free")
    adb_shell(adb, serial, "cp", remote, f"{lib_dir}/libcrprobe.so")
    adb_shell(adb, serial, "chmod", "755", f"{lib_dir}/libcrprobe.so")
    installed = adb_shell(adb, serial, "sha256sum", f"{lib_dir}/libcrprobe.so").stdout.split()
    if not installed or installed[0] != digest:
        raise RuntimeError(f"离线副本 probe 校验失败: {installed[0] if installed else 'empty'} != {digest}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--port", type=int, default=26790)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--adb", required=True)
    args = parser.parse_args()
    adb = Path(args.adb)
    root = Path(args.engine_root)
    apk = find_apk(Path.home() / "Downloads")
    print(f"[engine] installing prepared APK: {apk}", flush=True)
    # A browser may preserve a duplicate-download suffix such as .apk.1.
    # ADB validates the filename suffix, so stage a temporary copy with the
    # canonical extension without renaming or changing the user's download.
    staged = Path("/tmp/royale-engine-apk-install/FirstLight-cr-ai-offline.apk")
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(apk, staged)
    installed = subprocess.run([str(adb), "-s", args.serial, "install", "-r", str(staged)],
                               check=False, text=True)
    if installed.returncode:
        raise RuntimeError("APK 安装失败")
    probe_hash = install_matching_probe(adb, args.serial, root)
    print(f"[engine] installed protocol-matched probe: {probe_hash[:12]}…", flush=True)
    env = os.environ.copy()
    env["PATH"] = str(adb.parent) + os.pathsep + env.get("PATH", "")
    bootstrap = root / "bootstrap_emulator.py"
    return subprocess.run([sys.executable, "-u", str(bootstrap), "--serial", args.serial,
                           "--port", str(args.port)], cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[engine] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
