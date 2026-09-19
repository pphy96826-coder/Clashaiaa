"""Headless regression checks for Tk's process polling lifecycle."""
import queue
import unittest
from unittest.mock import Mock

from desktop_console import RoyaleHarnessConsole, OutputLine


class DesktopLifecycleTests(unittest.TestCase):
    def console(self):
        app = RoyaleHarnessConsole.__new__(RoyaleHarnessConsole)
        app.root = Mock()
        app.process = Mock()
        app.process.poll.return_value = None
        app.output_queue = queue.Queue()
        app.reader_threads = []
        app._append = Mock()
        app._stop_deadline = None
        app._close_deadline = None
        app.closing = True
        return app

    def test_closing_keeps_polling_until_child_exits(self):
        app = self.console()
        app._drain_output()
        app.root.after.assert_called_once_with(100, app._drain_output)

    def test_close_deadline_kills_unresponsive_child(self):
        app = self.console()
        app.status = Mock()
        app._close_deadline = 0
        app._drain_output()
        app.process.kill.assert_called_once()
        app.root.after.assert_called_once()

    def test_busy_output_yields_back_to_tk(self):
        app = self.console()
        for _ in range(300):
            app.output_queue.put(OutputLine('stdout', 'line'))
        app._drain_output()
        self.assertEqual(app._append.call_count, 200)
        self.assertEqual(app.output_queue.qsize(), 100)

    def test_exit_closes_pipes_and_destroys_closing_window(self):
        app = self.console()
        process = app.process
        process.returncode = 0
        app.pid = Mock()
        app.status = Mock()
        app._set_running = Mock()
        app._process_finished()
        self.assertIsNone(app.process)
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()
        app.root.destroy.assert_called_once()
