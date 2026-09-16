"""Touch-only lifecycle controller used by continuous live runs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any, Callable

import config
if str(config.FIRSTLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(config.FIRSTLIGHT_DIR))
from native_runner.touch_executor import PersistentTouchAgentClient, TouchExecutorError


class LifecycleError(RuntimeError):
    """A menu transition was not safely acknowledged."""


class LiveLifecycle:
    """Own lobby/result taps and leave card actions to ``ActionExecutor``."""

    def __init__(self, probe, *, log: Callable[..., Any], calibration_path=None,
                 touch_host='127.0.0.1', touch_port=config.TOUCH_AGENT_PORT,
                 timeout=2.0, poll_interval=.25) -> None:
        self.probe = probe
        self.log = log
        self.poll_interval = float(poll_interval)
        self.timeout = float(timeout)
        self.touch_host = str(touch_host)
        self.touch_port = int(touch_port)
        self.touch_device_port = int(config.TOUCH_AGENT_DEVICE_PORT)
        self.post_result_delay = 0.75
        self.touch = PersistentTouchAgentClient(self.touch_host, self.touch_port, timeout=self.timeout)
        self.calibration = self._load_calibration(calibration_path or config.LIFECYCLE_CALIBRATION_PATH)

    @staticmethod
    def _load_calibration(path) -> dict[str, tuple[int, int] | None]:
        points = {
            'battle_button': tuple(config.LOBBY_BATTLE_BTN),
            'continue_button': tuple(config.LOBBY_RESULT_DISMISS),
            'post_result_button': None,
            'emote_button': (106, 1640),
        }
        candidate = Path(path)
        if not candidate.is_file():
            return points
        value = json.loads(candidate.read_text(encoding='utf-8'))
        raw = value.get('points', value.get('ui', value))
        if not isinstance(raw, dict):
            return points
        for key in points:
            point = raw.get(key)
            if isinstance(point, (list, tuple)) and len(point) == 2:
                points[key] = (int(point[0]), int(point[1]))
        slots = raw.get('emote_slots')
        if isinstance(slots, list):
            points['emote_slots'] = [tuple(map(int, p)) for p in slots
                                     if isinstance(p, (list, tuple)) and len(p) == 2]
        return points

    def send_emote(self, index: int = 0) -> None:
        """Open the in-battle emote tray and send one configured emote."""
        slots = self.calibration.get('emote_slots') or [
            (280, 1265), (475, 1265), (672, 1265), (869, 1265),
            (280, 1438), (475, 1438), (672, 1438), (869, 1438),
        ]
        if not 0 <= index < len(slots):
            raise LifecycleError(f'emote index must be 0..{len(slots) - 1}')
        self._tap(self.calibration['emote_button'], 'emote-open')
        time.sleep(.12)
        self._tap(tuple(slots[index]), f'emote-{index}')
        self.log('emote_sent', index=index, screen=list(slots[index]))

    def _recover_touch_agent(self) -> None:
        """Recreate the explicit ADB lane and resident guest touch agent."""
        serial = str(config.ADB_SERIAL or '').strip()
        adb = str(config.ADB_PATH or '').strip()
        if not serial or not adb:
            return
        forward = [adb, '-s', serial, 'forward',
                   f'tcp:{self.touch_port}', f'tcp:{self.touch_device_port}']
        subprocess.run(forward, capture_output=True, text=True, timeout=5, check=False)
        script = (
            "pid=$(pidof firstlight-touch-agent 2>/dev/null || true); "
            "if [ -n \"$pid\" ]; then kill $pid 2>/dev/null || true; fi; "
            "if [ -x /data/local/tmp/firstlight-touch-agent ]; then "
            f"nohup /data/local/tmp/firstlight-touch-agent {self.touch_device_port} "
            ">/data/local/tmp/firstlight-touch-agent.log 2>&1 & "
            "else exit 7; fi"
        )
        result = subprocess.run(
            [adb, '-s', serial, 'shell', 'sh', '-c', script],
            capture_output=True, text=True, timeout=5, check=False)
        self.log('touch_recovery', serial=serial, returncode=result.returncode,
                 stderr=(result.stderr or '').strip()[-300:])

    def _ensure_touch_ready(self) -> None:
        """Wait for a newly forwarded guest agent before sending a menu tap.

        MuMu/ADB can leave the forward listening while the guest touch agent
        is still restarting.  A single hello in that window used to abort the
        whole desktop run.  Handshake retries are safe because no tap has
        been sent yet; once a tap is sent, ``_tap`` still preserves the
        fail-closed UNKNOWN semantics and never replays it.
        """
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                self.touch.health_check()
                return
            except (TouchExecutorError, OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                self.touch.close()
                if attempt == 1:
                    try:
                        self._recover_touch_agent()
                    except (OSError, subprocess.SubprocessError) as recovery_exc:
                        self.log('touch_recovery_error', error=str(recovery_exc))
                self.touch = PersistentTouchAgentClient(
                    self.touch_host, self.touch_port, timeout=self.timeout)
                if attempt < 4:
                    time.sleep(min(.25, .05 * (attempt + 1)))
        assert last_error is not None
        raise TouchExecutorError(f'touch agent handshake unavailable after retries: {last_error}')

    def recover_runtime(self) -> dict[str, Any]:
        """Rebuild the local ADB/probe lane without entering a battle.

        This is intentionally limited to the menu/recovery path: it may
        restart the game process, but it never sends a card, tap, or match
        request.  A valid JSON response from the guest probe is the readiness
        criterion; an open TCP port alone is not sufficient.
        """
        serial = str(config.ADB_SERIAL or '').strip()
        adb = str(config.ADB_PATH or '').strip()
        if not serial or not adb:
            raise LifecycleError('ADB serial/path is not configured')
        base = [adb, '-s', serial]
        connected = subprocess.run([adb, 'connect', serial], capture_output=True,
            text=True, timeout=5, check=False)
        state = subprocess.run(base + ['get-state'], capture_output=True,
            text=True, timeout=5, check=False)
        if state.returncode != 0 or state.stdout.strip() != 'device':
            raise LifecycleError(f'Android guest is offline: {(connected.stderr or connected.stdout).strip()}')
        subprocess.run(base + ['forward', '--remove', f'tcp:{config.PROBE_PORT}'],
            capture_output=True, timeout=5, check=False)
        forward = subprocess.run(base + ['forward', f'tcp:{config.PROBE_PORT}',
            f'tcp:{config.PROBE_DEVICE_PORT}'], capture_output=True, text=True,
            timeout=5, check=False)
        if forward.returncode != 0:
            raise LifecycleError(f'probe forward failed: {(forward.stderr or forward.stdout).strip()}')
        for _ in range(4):
            result = self.probe.query()
            if isinstance(result, dict):
                self.log('runtime_recovered', probe='json-ready', restarted=False)
                return {'ok': True, 'restarted': False}
            time.sleep(.25)
        # The port may be present while the injected listener is dead.  In
        # the safe menu path, restart only the game process and rebuild the
        # forward, then wait for a real probe response.
        subprocess.run(base + ['shell', 'am', 'force-stop', config.PACKAGE_NAME],
            capture_output=True, timeout=5, check=False)
        time.sleep(.4)
        subprocess.run(base + ['shell', 'am', 'start', '-n',
            config.PACKAGE_NAME + '/' + config.ACTIVITY_NAME],
            capture_output=True, timeout=5, check=False)
        # The injected probe waits up to ~30 seconds for GameApp/libg to
        # finish initialization on a freshly restarted MuMu guest.  Keep the
        # recovery window above that bound instead of treating a slow native
        # bootstrap as a permanent failure.
        for _ in range(160):
            subprocess.run(base + ['forward', '--remove', f'tcp:{config.PROBE_PORT}'],
                capture_output=True, timeout=5, check=False)
            subprocess.run(base + ['forward', f'tcp:{config.PROBE_PORT}',
                f'tcp:{config.PROBE_DEVICE_PORT}'], capture_output=True,
                timeout=5, check=False)
            result = self.probe.query()
            if isinstance(result, dict):
                self.log('runtime_recovered', probe='json-ready', restarted=True)
                return {'ok': True, 'restarted': True}
            time.sleep(.25)
        raise LifecycleError('probe listener did not return valid JSON after game restart')

    def _tap(self, point: tuple[int, int], label: str) -> None:
        sequence = f'lifecycle-{label}-{uuid.uuid4()}'
        try:
            # This preflight may reconnect several times, but it happens
            # before the unique tap sequence is handed to the guest.
            self._ensure_touch_ready()
            receipt = self.touch.send_tap(*point, sequence_id=sequence)
        except (TouchExecutorError, OSError, RuntimeError) as exc:
            self.log('lifecycle_error', phase=label, error=str(exc))
            raise LifecycleError(f'{label} tap outcome is UNKNOWN: {exc}') from exc
        if receipt.get('type') != 'INPUT_DELIVERED':
            raise LifecycleError(f'{label} tap was not acknowledged')
        self.log('lifecycle_tap', phase=label, screen=list(point), sequence_id=sequence)

    def close(self) -> None:
        self.touch.close()

    def start_battle(self) -> None:
        # A restarted guest can leave a TCP forward open while the injected
        # listener is still dead.  Recover the control lane before RESET so a
        # startup race cannot abort the runner before the UI becomes usable.
        self.recover_runtime()
        self.probe.reset_live_context()
        self._tap(self.calibration['battle_button'], 'battle')
        self.probe.arm_live_context()
        self.log('battle_requested', mode='continuous')

    def attach_active(self) -> None:
        self.probe.attach_live_context()
        self.log('battle_attach_requested', mode='mid_battle')

    def wait_for_idle(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            data = self.probe.query()
            if isinstance(data, dict) and not data.get('in_battle'):
                return True
            time.sleep(min(self.poll_interval, max(.01, deadline - time.monotonic())))
        return False

    def return_to_lobby(self, *, timeout_seconds=20.0) -> None:
        from bridge.lifecycle_screen import read_screen
        deadline = time.monotonic() + timeout_seconds
        previous = None
        stable = 0
        taps = 0
        unknown_after_result = 0
        first_result_tap_at = None
        while time.monotonic() < deadline:
            kind, point = read_screen()
            if previous is None or kind != previous[0]:
                self.log('lifecycle_screen', screen=kind, point=point)
            key = (kind, point)
            same = (previous is not None and kind == previous[0] and point is not None
                    and previous[1] is not None
                    and max(abs(a-b) for a,b in zip(point, previous[1])) <= 12)
            stable = stable + 1 if same else 1
            previous = key
            # After the result OK and the optional reward OK have both been
            # acknowledged, one positive lobby frame is sufficient.  The
            # OCR process can briefly alternate between ``lobby`` and
            # ``unknown`` while the lobby animation is still settling;
            # requiring two consecutive frames here used to turn a successful
            # return (the Battle button was already visible) into a fatal
            # timeout for continuous mode.
            lobby_samples_required = 1 if taps >= 1 else 2
            if kind == 'lobby' and stable >= lobby_samples_required:
                self.log('lobby_ready', mode='continuous',
                         evidence='visible_battle_button', samples=stable)
                return
            if kind == 'ok' and stable >= 2:
                unknown_after_result = 0
                if taps >= 3:
                    raise LifecycleError('OK remains visible after three acknowledged taps')
                label = 'result-ok' if taps == 0 else 'post-result-ok'
                self._tap(point, label)
                if taps == 0:
                    first_result_tap_at = time.monotonic()
                taps += 1
                # A new tap requires fresh visual evidence after transition.
                previous, stable = None, 0
                time.sleep(1)
            else:
                if (kind == 'unknown' and taps == 1 and
                        self.calibration.get('post_result_button') is not None):
                    unknown_after_result += 1
                    # The reward/arena page can be visually valid while OCR
                    # misses its lower OK label.  The first result tap is
                    # already acknowledged, so use the separately calibrated
                    # second OK coordinate once after the transition window.
                    if (unknown_after_result >= 2 and first_result_tap_at is not None
                            and time.monotonic() - first_result_tap_at >= self.post_result_delay):
                        self._tap(self.calibration['post_result_button'], 'post-result-ok-fallback')
                        taps += 1
                        previous, stable = None, 0
                        unknown_after_result = 0
                        time.sleep(1)
                        continue
                time.sleep(max(.25, self.poll_interval))
        raise LifecycleError('Lobby not visually confirmed before timeout')
