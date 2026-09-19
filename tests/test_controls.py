import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from live_lifecycle import LiveLifecycle, LifecycleError
from main import parse_args
from runtime_console import AgentConsole, ConsoleCommand
from desktop_console import (
    ConsoleConfig, build_runner_command, config_from_dict, load_config,
    resolve_agent_python, save_config, validate_agent_python,
)


class ConsoleTests(unittest.TestCase):
    def test_runner_command_builds_selected_modes(self):
        command = build_runner_command(
            ConsoleConfig(
                checkpoint='general', device='cpu', continuous=True,
                max_matches='3', launch_mode='attach', log_path='logs/ui.jsonl',
            ),
            root=Path('/repo'), python_executable=Path('/repo/.venv/bin/python'),
        )
        self.assertEqual(command, [
            '/repo/.venv/bin/python', '-u', '/repo/main.py',
            '--checkpoint', 'general', '--device', 'cpu', '--console',
            '--continuous', '--max-matches', '3', '--attach-active',
            '--log', 'logs/ui.jsonl',
        ])

    def test_runner_command_rejects_max_matches_without_continuous(self):
        with self.assertRaisesRegex(ValueError, 'continuous'):
            build_runner_command(ConsoleConfig(continuous=False, max_matches='1'))

    def test_runner_command_uses_start_battle_by_default(self):
        command = build_runner_command(
            ConsoleConfig(device='cpu', continuous=False, launch_mode='start'),
            root=Path('/repo'), python_executable='/usr/bin/python3',
        )
        self.assertIn('--start-battle', command)
        self.assertNotIn('--continuous', command)

    def test_console_config_round_trip_is_json_and_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'console.json'
            expected = ConsoleConfig(
                checkpoint='general', device='mps', continuous=False,
                max_matches='', launch_mode='attach', log_path='/tmp/live.jsonl',
                agent_python='/tmp/agent-python', auto_emotes=True,
            )
            save_config(expected, path)
            raw = json.loads(path.read_text(encoding='utf-8'))
            raw['future_field'] = 'ignored'
            path.write_text(json.dumps(raw), encoding='utf-8')
            self.assertEqual(load_config(path), expected)

    def test_agent_python_prefers_saved_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'python'
            executable.write_text('#!/bin/sh\n', encoding='utf-8')
            executable.chmod(0o755)
            self.assertEqual(resolve_agent_python(str(executable), Path(directory)), str(executable))

    @patch('desktop_console.subprocess.run')
    def test_agent_python_validation_reports_import_failure(self, run):
        run.return_value = Mock(returncode=1, stderr='ModuleNotFoundError: native_runner', stdout='')
        error = validate_agent_python('/tmp/agent-python', Path('/repo'))
        self.assertIn('native_runner', error)
        self.assertIn('native_runner', run.call_args.args[0][2])

    def test_config_from_dict_falls_back_for_invalid_values(self):
        loaded = config_from_dict({'continuous': 'yes', 'launch_mode': 'invalid', 'device': 3})
        self.assertEqual(loaded.continuous, ConsoleConfig.continuous)
        self.assertEqual(loaded.launch_mode, 'start')
        self.assertEqual(loaded.device, ConsoleConfig.device)

    def test_parse_commands_and_comments(self):
        self.assertEqual(AgentConsole.parse(' model general '),
                         ConsoleCommand('model', ('general',)))
        self.assertIsNone(AgentConsole.parse('  # comment'))
        self.assertIsNone(AgentConsole.parse(''))

    def test_continuous_cli_options(self):
        args = parse_args(['--continuous', '--console', '--max-matches', '2', '--attach-active'])
        self.assertTrue(args.continuous)
        self.assertTrue(args.console)
        self.assertEqual(args.max_matches, 2)
        with self.assertRaises(SystemExit):
            parse_args(['--once', '--continuous'])


class LifecycleTests(unittest.TestCase):
    def test_unknown_tap_stops_without_retry(self):
        lifecycle = LiveLifecycle.__new__(LiveLifecycle)
        lifecycle.poll_interval = 0
        lifecycle.log = Mock()
        lifecycle._tap = Mock(side_effect=LifecycleError('UNKNOWN'))
        with patch('bridge.lifecycle_screen.read_screen', return_value=('ok', (540, 1764))), \
                patch('live_lifecycle.time.sleep'):
            with self.assertRaises(LifecycleError):
                lifecycle.return_to_lobby()
        lifecycle._tap.assert_called_once()

    def test_unknown_screen_times_out_without_tap(self):
        lifecycle = LiveLifecycle.__new__(LiveLifecycle)
        lifecycle.poll_interval = 0
        lifecycle.log = Mock()
        lifecycle._tap = Mock()
        with patch('bridge.lifecycle_screen.read_screen', return_value=('unknown', None)), \
                patch('live_lifecycle.time.monotonic', side_effect=[0, 0, 21]), \
                patch('live_lifecycle.time.sleep'):
            with self.assertRaises(LifecycleError):
                lifecycle.return_to_lobby()
        lifecycle._tap.assert_not_called()

    def test_screen_reader_rejects_active_battle_and_reward_claims(self):
        from bridge.lifecycle_screen import classify
        row = {'confidence':1, 'x':.5, 'y':.92}
        self.assertEqual(classify([{**row, 'text':'Claim'}])[0], 'unknown')
        self.assertEqual(classify([{**row, 'text':'OK', 'y':.3}])[0], 'unknown')
        self.assertEqual(classify([{**row, 'text':'OK'}])[0], 'ok')
        self.assertEqual(classify([{**row, 'text':'Battle', 'y':.77}])[0], 'lobby')

    def test_emote_sends_open_then_selected_slot(self):
        lifecycle = LiveLifecycle.__new__(LiveLifecycle)
        lifecycle.calibration = {'emote_button': (106, 1640),
                                 'emote_slots': [(100, 1265), (200, 1265)]}
        lifecycle._tap = Mock()
        lifecycle.log = Mock()
        with patch('live_lifecycle.time.sleep'):
            lifecycle.send_emote(1)
        self.assertEqual(lifecycle._tap.call_args_list, [
            call((106, 1640), 'emote-open'),
            call((200, 1265), 'emote-1'),
        ])


    def test_calibration_loads_distinct_result_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ui.json'
            path.write_text(json.dumps({'points': {
                'battle_button': [1, 2],
                'continue_button': [3, 4],
                'post_result_button': [5, 6],
                'emote_button': [7, 8],
                'emote_slots': [[9, 10], [11, 12]],
            }}), encoding='utf-8')
            points = LiveLifecycle._load_calibration(path)
        self.assertEqual(points['battle_button'], (1, 2))
        self.assertEqual(points['continue_button'], (3, 4))
        self.assertEqual(points['post_result_button'], (5, 6))
        self.assertEqual(points['emote_button'], (7, 8))
        self.assertEqual(points['emote_slots'], [(9, 10), (11, 12)])

    def test_touch_handshake_retry_replaces_stale_session(self):
        first = Mock()
        first.health_check.side_effect = TimeoutError('forward is up; guest is not ready')
        second = Mock()
        second.health_check.return_value = {'type': 'PONG'}
        lifecycle = LiveLifecycle.__new__(LiveLifecycle)
        lifecycle.touch_host = '127.0.0.1'
        lifecycle.touch_port = 28124
        lifecycle.touch_device_port = 28123
        lifecycle.timeout = 0.01
        lifecycle.touch = first
        lifecycle._recover_touch_agent = Mock()
        with patch('live_lifecycle.PersistentTouchAgentClient', return_value=second):
            with patch('live_lifecycle.time.sleep'):
                lifecycle._ensure_touch_ready()
        first.close.assert_called_once()
        second.health_check.assert_called_once_with()

    def test_configured_post_result_ok_is_sent_after_native_finalization(self):
        lifecycle = LiveLifecycle.__new__(LiveLifecycle)
        lifecycle.calibration = {
            'continue_button': (540, 1764),
            'post_result_button': (540, 1830),
        }
        lifecycle.log = Mock()
        lifecycle.poll_interval = 0
        lifecycle.post_result_delay = 0
        lifecycle._tap = Mock()
        lifecycle.wait_for_idle = Mock(return_value=True)
        with patch('bridge.lifecycle_screen.read_screen', side_effect=[
            ('unknown', None), ('ok', (540, 1764)), ('ok', (540, 1764)),
            ('unknown', None), ('ok', (540, 1830)), ('ok', (540, 1830)),
            ('lobby', (540,1490)), ('lobby', (540,1490)),
        ]), patch('live_lifecycle.time.sleep'):
            lifecycle.return_to_lobby()
        self.assertEqual(lifecycle._tap.call_args_list, [
            call((540, 1764), 'result-ok'),
            call((540, 1830), 'post-result-ok'),
        ])


if __name__ == '__main__':
    unittest.main()
