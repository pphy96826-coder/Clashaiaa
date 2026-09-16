"""Read-only menu recognition; never treats a missing battle as a lobby."""
import io
import json
import subprocess

from PIL import Image
import config


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
    executable = config.BASE_DIR / '.venv' / 'bin' / 'lifecycle-ocr'
    if not executable.is_file():
        raise RuntimeError('Lifecycle screen reader missing; build tools/lifecycle_ocr.m')
    png = subprocess.run([str(config.ADB_PATH), '-s', config.ADB_SERIAL,
                          'exec-out', 'screencap', '-p'], capture_output=True,
                         timeout=5, check=True).stdout
    with Image.open(io.BytesIO(png)) as picture:
        width, height = picture.size
    result = subprocess.run([str(executable)], input=png, capture_output=True,
                            timeout=5, check=True)
    kind, point = classify(json.loads(result.stdout))
    return kind, None if point is None else (round(point[0]*width), round(point[1]*height))
