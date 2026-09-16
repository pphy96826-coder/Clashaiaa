import json
import os
import shutil
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
# Keep macOS aliases such as /var intact.  ``resolve()`` turns them into
# /private/var, which makes externally supplied settings paths differ from
# the path the caller provided and breaks portable configuration checks.
SETTINGS_PATH = Path(os.environ.get('CR_AGENT_SETTINGS', BASE_DIR / 'settings.local.json')).absolute()
SETTINGS = json.loads(SETTINGS_PATH.read_text(encoding='utf-8-sig')) if SETTINGS_PATH.is_file() else {}
if not isinstance(SETTINGS, dict):
    raise ValueError('settings.local.json must contain an object')

def configured_path(key, default):
    value = os.path.expandvars(str(SETTINGS.get(key) or default))
    path = Path(value).expanduser()
    return (SETTINGS_PATH.parent / path).absolute() if not path.is_absolute() else path.absolute()

FIRSTLIGHT_DIR = configured_path('firstlight_dir', '../FirstLight_CR')
UPSTREAM_DIR = configured_path('upstream_dir', FIRSTLIGHT_DIR)
VENV_PYTHON = Path(sys.executable)
DEVICE = SETTINGS.get('device', 'cuda:0')
CALIBRATION_VERIFIED = SETTINGS.get('calibration_verified') is True

# Virtual Machine settings (Dedicated for online Null's Royale)
VM_INDEX = SETTINGS.get('vm_index', 0)
VM_NAME = SETTINGS.get('vm_name', '')
ADB_SERIAL = SETTINGS.get('adb_serial', '')
AUTO_DISCOVER_ADB = SETTINGS.get('auto_discover_adb', True) is True
ADB_PATH = configured_path('adb_path', shutil.which('adb') or 'adb.exe')
MUMU_MANAGER_PATH = configured_path('mumu_manager_path', 'MuMuManager.exe')
NDK_ROOT = configured_path('ndk_root', os.environ.get('ANDROID_NDK_HOME', 'android-ndk'))

# Android Game Package
PACKAGE_NAME = "nullsroyale.rel.free"
ACTIVITY_NAME = "com.supercell.clashroyale.GameApp"
PROBE_DEVICE_PORT = 26888  # Compiled into the stable probe.
PROBE_PORT = SETTINGS.get('probe_port', 26888)
if type(PROBE_PORT) is not int or not 1024 <= PROBE_PORT <= 65535:
    raise ValueError('probe_port must be an integer from 1024 to 65535')
TOUCH_AGENT_PORT = SETTINGS.get('touch_agent_port', 28123)
if type(TOUCH_AGENT_PORT) is not int or not 1024 <= TOUCH_AGENT_PORT <= 65535:
    raise ValueError('touch_agent_port must be an integer from 1024 to 65535')
TOUCH_AGENT_DEVICE_PORT = SETTINGS.get('touch_agent_device_port', 28123)
if type(TOUCH_AGENT_DEVICE_PORT) is not int or not 1024 <= TOUCH_AGENT_DEVICE_PORT <= 65535:
    raise ValueError('touch_agent_device_port must be an integer from 1024 to 65535')
LIFECYCLE_CALIBRATION_PATH = configured_path(
    'lifecycle_calibration_path', BASE_DIR / 'lifecycle_ui_calibration.json'
)

# Checkpoint settings
CHECKPOINTS_DIR = configured_path('checkpoints_dir', FIRSTLIGHT_DIR / 'checkpoints')
CHECKPOINTS = {
    "hog26": CHECKPOINTS_DIR / "2_6hog_expert" / "hog26-specialist2.pt",
    "hog26_proactive": CHECKPOINTS_DIR / "2_6hog_expert" / "hog26-specialist1.pt",
    "general": CHECKPOINTS_DIR / "General" / "checkpoint-step-00000460.pt",
    "il": CHECKPOINTS_DIR / "IL" / "checkpoint-step-00029396.pt",
    "active_il": CHECKPOINTS_DIR / "active IL" / "checkpoint-step-00000030.pt",
}
DEFAULT_CHECKPOINT = CHECKPOINTS["hog26"]
# Default to the upstream event rules; keep precise-event extensions opt-in.
DEFAULT_OBSERVATION_PROFILE = 'reference'

# Screen Layout & Geometry (1080 x 1920)
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920

# Lobby Coordinates
LOBBY_BATTLE_TAB = (540, 1850)       # Battle tab icon in bottom navigation bar
LOBBY_BATTLE_BTN = (540, 1490)       # Center "对战" Button
LOBBY_RESULT_DISMISS = (540, 1710)   # Blue "OK" button center on match result screen (Device-2)
LOBBY_RESULT_POPUP = (500, 960)      # "确定" button on trophy road / level up popup
LOBBY_RELOAD_BTN = (540, 1150)       # OK button if connection pop-up appears

# In-Battle Hand Card Slot Coordinates (Tapping to select card)
# Calibrated against 1080x1920 live battle HUD in Null's Royale
HAND_CARD_SLOTS = [
    (336, 1710),  # Slot 0 (left card)
    (540, 1710),  # Slot 1
    (744, 1710),  # Slot 2
    (946, 1710),  # Slot 3 (right card)
]

# Explicit runtime identity; the legacy probe's local_owner field is hardcoded.
LOCAL_ACCOUNT_ID = SETTINGS.get('account_id')
if LOCAL_ACCOUNT_ID is not None and (type(LOCAL_ACCOUNT_ID) is not int or LOCAL_ACCOUNT_ID <= 0):
    raise ValueError('account_id must be a positive numeric native accountId or null')
# Detect the actual tower asset. Optional explicit fallback for legacy probes.
LOCAL_TOWER_TROOP_ID = None
CALIBRATION_PATH = configured_path('calibration_path', BASE_DIR / 'calibration.json')
ABILITY_CALIBRATION_PATH = configured_path('ability_calibration_path', BASE_DIR / 'ability_calibration.local.json')
# Faster live reaction window.  Fresh-state validation still runs immediately
# before every input, so shortening the cadence does not permit stale actions.
DECISION_TICKS = 2
TICK_SECONDS = 0.05
# Compensate for observation-to-input latency in the model view only. The
# Predict at the point where the next decision is actually made: first
# compensate for the measured host/input path, then forecast 1.5 seconds so
# the policy can see beyond the native deployment animation and avoid issuing
# repeated answers to the same threat. This changes only model input timing;
# it cannot make the Android game accept a touch before the UI is ready.
# Both values affect the model view only; probe state remains authoritative.
PREDICTION_LATENCY_COMPENSATION_MS = 100
PREDICTION_HORIZON_TICKS = 30
# Use the predictive scene directly in live decisions. The console can turn
# it off immediately with `prediction shadow off` if a battle shows drift.
ENABLE_MODEL_PREDICTION_OVERLAY = True
# The downloaded native engine is a stand-alone reset/replay service.  Keep
# live use off until a verified live-snapshot import protocol exists; otherwise
# a standard-deck rollout would be mistaken for the current battle.
FULL_SIMULATION_ROOT = SETTINGS.get(
    'full_simulation_root', str(Path.home() / 'Downloads' / 'clash-royale-battle-engine-main'))
FULL_SIMULATION_PORT = SETTINGS.get('full_simulation_port', 26790)
if type(FULL_SIMULATION_PORT) is not int or not 1024 <= FULL_SIMULATION_PORT <= 65535:
    raise ValueError('full_simulation_port must be an integer from 1024 to 65535')
ENABLE_FULL_SIMULATION = SETTINGS.get('enable_full_simulation', False) is True
FULL_SIMULATION_REQUIRE_STANDARD_DECK = SETTINGS.get(
    'full_simulation_require_standard_deck', True) is True
# The live probe normally exposes the local deck but not the opponent's
# hidden deck.  In partial mode the mirror uses the local deck exactly and a
# deterministic engine placeholder for the hidden side; live opponent
# entities remain authoritative and are not replaced by that placeholder.
FULL_SIMULATION_PARTIAL_OPPONENT = SETTINGS.get(
    'full_simulation_partial_opponent', True) is True
STALE_SECONDS = 1.2
# Hand rotation is advisory telemetry, not an input gate.  Release a missed
# action quickly so a slow/missing ACK cannot stall the match for seconds;
# the touch worker remains strictly serial and never replays the write.
ACK_TIMEOUT_SECONDS = 0.35
# Keep an ambiguous touch visible to the model briefly after timeout.  The
# card remains usable and other slots are not blocked; this only prevents a
# missing ACK from making the same threat look completely untouched.
UNCERTAIN_PREDICTION_SECONDS = 1.4
# Keep the cost reserved after a short ACK timeout while the guest is still
# publishing the updated hand/elixir snapshot. This prevents stale telemetry
# from authorizing a second card that the game can no longer afford.
ELIXIR_RESERVATION_SECONDS = 1.25
ACTION_MAX_LATENESS_SECONDS = 0.8
# Minimum UI settle time between selecting a card and placing it.  Keep a
# bounded gap for the Android UI commit, but avoid adding an unnecessary
# 60ms to every action; fresh validation and serial input remain enforced.
CARD_SELECTION_GAP_SECONDS = 0.04

# Compatibility helper: grid indices are cell centers, including subcell.
# New execution code takes the whole decoded ActionV1 (and its owner).
def model_grid_to_screen(grid_x, grid_y, subcell_offset=(0.0, 0.0)):
    from bridge.coordinates import ScreenCalibration
    dx, dy = subcell_offset or (0.0, 0.0)
    return ScreenCalibration.load(CALIBRATION_PATH).project(grid_x + .5 + dx, grid_y + .5 + dy)

grid_to_screen = model_grid_to_screen

