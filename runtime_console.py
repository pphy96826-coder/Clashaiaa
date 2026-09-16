"""Non-blocking operator console for the live agent."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
from typing import Callable, TextIO


@dataclass(frozen=True)
class ConsoleCommand:
    name: str
    args: tuple[str, ...] = ()


class AgentConsole:
    """Read operator commands without interrupting the telemetry loop.

    The console thread only parses input.  All state-changing work is executed
    by the live loop at a safe point, so a model switch cannot race a Touch
    submission or an outcome acknowledgement.
    """

    HELP = (
        "commands: status, pause, resume, model <hog26|general|path>, "
        "emote <0-7>, emote auto, emote off, prediction shadow <on|off>, "
        "recover, start, attach, stop, quit, help"
    )

    def __init__(self, input_stream: TextIO | None = None,
                 output: Callable[[str], None] | None = None) -> None:
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.output = output or print
        self._commands: queue.Queue[ConsoleCommand] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def parse(line: str) -> ConsoleCommand | None:
        fields = line.strip().split()
        if not fields or fields[0].startswith('#'):
            return None
        return ConsoleCommand(fields[0].lower(), tuple(fields[1:]))

    def start(self) -> None:
        if self._thread is not None:
            return
        self.output(self.HELP)
        self._thread = threading.Thread(target=self._read, name='agent-console', daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while not self._stop.is_set():
            try:
                line = self.input_stream.readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            command = self.parse(line)
            if command is not None:
                self._commands.put(command)

    def poll(self) -> list[ConsoleCommand]:
        commands: list[ConsoleCommand] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def say(self, message: str) -> None:
        self.output(f"[console] {message}")

    def close(self) -> None:
        self._stop.set()
