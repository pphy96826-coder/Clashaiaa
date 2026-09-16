#!/usr/bin/env python3
"""Build a safe, source-only RoyaleHarness deployment archive.

Private runtime state, credentials, model weights, APKs, logs, and local
artifacts are deliberately excluded. The archive contains source, templates,
tests, documentation, and reproducible dependency instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXACT_EXCLUDES = {
    ".git",
    ".venv",
    "logs",
    "outputs",
    "diagnostics",
    "backups",
    "local",
    "dist",
    "wheelhouse",
    "checkpoints",
    "probe/artifacts/candidates",
}
NAME_EXCLUDES = {
    ".DS_Store",
    "settings.device2.local.json",
    "ability_calibration.local.json",
    "display_calibration.device2.json",
    "lifecycle_ui_calibration.device2.json",
}
SUFFIX_EXCLUDES = {".apk", ".apex", ".pt", ".pth", ".safetensors", ".pyc", ".so", ".dylib", ".dll", ".exe"}


def excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    if any(rel == prefix or rel.startswith(prefix + "/") for prefix in EXACT_EXCLUDES):
        return True
    if any(part in {".git", ".venv", "logs", "outputs", "diagnostics", "backups", "local", "dist", "wheelhouse", "checkpoints"} for part in parts):
        return True
    if path.name in NAME_EXCLUDES or path.suffix.lower() in SUFFIX_EXCLUDES:
        return True
    if path.name.endswith(".local.json"):
        return True
    if "/validation-" in rel or "/raw" in rel:
        return True
    return False


def collect() -> list[Path]:
    files = [p for p in ROOT.rglob("*") if p.is_file() and not excluded(p)]
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "RoyaleHarness-source.zip")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect()
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(rel, data)
            manifest.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        payload = json.dumps({"format": 1, "files": manifest}, ensure_ascii=False, indent=2).encode()
        archive.writestr("deployment/MANIFEST.sha256.json", payload)
    print(f"created {output}")
    print(f"files={len(files)} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
