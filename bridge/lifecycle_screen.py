"""Read-only menu recognition; never treats a missing battle as a lobby."""
import io
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import config


def _reader_path():
    # Use the interpreter's active virtual environment, not BASE_DIR/.venv.
    # The live runner is commonly launched with RoyaleHarness/.venv/python
    # while this repository lives elsewhere on disk.
    return Path(config.VENV_PYTHON).parent / 'lifecycle-ocr'


def ensure_reader():
    executable = _reader_path()
    if executable.is_file():
        return executable

    source = config.BASE_DIR / 'tools' / 'lifecycle_ocr.m'
    if not source.is_file():
        raise RuntimeError(f'Lifecycle screen reader source missing: {source}')

    xcrun = shutil.which('xcrun')
    if not xcrun:
        raise RuntimeError(
            'Lifecycle screen reader missing and xcrun is unavailable; '
            'install the macOS Command Line Tools'
        )

    executable.parent.mkdir(parents=True, exist_ok=True)
    temporary = executable.with_name(executable.name + '.tmp')
    try:
        result = subprocess.run(
            [
                xcrun, 'clang', '-fobjc-arc',
                '-framework', 'Foundation',
                '-framework', 'Vision',
                str(source), '-o', str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            raise RuntimeError(
                'Failed to build lifecycle screen reader'
                + (f': {detail}' if detail else '')
            )
        temporary.chmod(0o755)
        temporary.replace(executable)
    finally:
        if temporary.exists():
            temporary.unlink()
    return executable


def classify(rows):
    for row in rows:
        text = row['text'].strip().upper().rstrip('!')
        if row['confidence'] < .7 or not .3 < row['x'] < .7:
            continue
        if text == 'OK' and .85 < row['y'] < .99:
            return 'ok', (row['x'], row['y'])
        if text == 'BATTLE' and .65 < row['y'] < .9:
            return 'lobby', (row['x'], row['y'])
    return 'unknown', None


def read_screen():
    executable = ensure_reader()
    png = subprocess.run([str(config.ADB_PATH), '-s', config.ADB_SERIAL,
                          'exec-out', 'screencap', '-p'], capture_output=True,
                         timeout=5, check=True).stdout
    with Image.open(io.BytesIO(png)) as picture:
        width, height = picture.size
    result = subprocess.run([str(executable)], input=png, capture_output=True,
                            timeout=5, check=True)
    kind, point = classify(json.loads(result.stdout))
    return kind, None if point is None else (round(point[0]*width), round(point[1]*height))
