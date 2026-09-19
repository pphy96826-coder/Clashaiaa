"""Tkinter desktop console for starting and supervising ``main.py``.

The console is intentionally a thin process controller. It does not import
policy or execution code, so changing a field cannot mutate strategy state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "desktop_console.local.json"
CONFIG_ENV = "CR_DESKTOP_CONSOLE_CONFIG"
MAX_LOG_LINES = 5000
KEY_EVENTS = (
    "ready",
    "battle_start",
    "decision",
    "hand_ack",
    "action_suppressed",
    "action_missed",
    "stopped",
)


def default_config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))).expanduser()


@dataclass
class ConsoleConfig:
    """Persisted, non-secret console fields."""

    checkpoint: str = "hog26"
    device: str = "mps" if sys.platform == "darwin" else "cpu"
    continuous: bool = True
    max_matches: str = ""
    launch_mode: str = "start"
    log_path: str = ""
    agent_python: str = ""
    auto_emotes: bool = False


def config_from_dict(data: Any) -> ConsoleConfig:
    """Load known fields while ignoring stale or malformed extra fields."""
    if not isinstance(data, dict):
        return ConsoleConfig()
    defaults = ConsoleConfig()
    values: dict[str, Any] = {}
    for field in fields(ConsoleConfig):
        value = data.get(field.name, getattr(defaults, field.name))
        if field.name in {"continuous", "auto_emotes"}:
            if not isinstance(value, bool):
                value = getattr(defaults, field.name)
        elif not isinstance(value, str):
            value = getattr(defaults, field.name)
        values[field.name] = value
    if values["launch_mode"] not in {"start", "attach"}:
        values["launch_mode"] = defaults.launch_mode
    return ConsoleConfig(**values)


def load_config(path: Path | str | None = None) -> ConsoleConfig:
    path = Path(path or default_config_path()).expanduser()
    try:
        return config_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return ConsoleConfig()


def save_config(config: ConsoleConfig, path: Path | str | None = None) -> Path:
    """Atomically persist the small set of console preferences."""
    path = Path(path or default_config_path()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def resolve_agent_python(saved_python: str | None = None, root: Path = ROOT) -> str:
    """Select an executable that belongs to the Agent environment."""
    candidates = (
        saved_python,
        os.environ.get("CR_AGENT_PYTHON"),
        str(Path.home() / "Documents/Codex/RoyaleHarness/.venv/bin/python"),
        str(root / ".venv" / "bin" / "python"),
        str(root / ".venv" / "Scripts" / "python.exe"),
        sys.executable,
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError("没有找到可用的 Agent Python 解释器")


def validate_agent_python(agent_python: str, root: Path = ROOT) -> str | None:
    """Return a human readable import error, or None when the environment works."""
    command = (
        "import config,sys; sys.path.insert(0,str(config.FIRSTLIGHT_DIR)); "
        "import native_runner, torch"
    )
    try:
        result = subprocess.run(
            [agent_python, "-c", command], cwd=root, env=runner_environment(),
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode:
        return (result.stderr or result.stdout or "环境导入失败").strip()
    return None


def _validate_max_matches(value: str, continuous: bool) -> str:
    value = value.strip()
    if value and (not value.isdigit() or int(value) <= 0):
        raise ValueError("max-matches must be a positive integer")
    if value and not continuous:
        raise ValueError("max-matches requires continuous mode")
    return value


def build_runner_command(
    config: ConsoleConfig,
    *,
    root: Path = ROOT,
    python_executable: Path | str | None = None,
) -> list[str]:
    """Build the exact subprocess argv used by the Start button."""
    checkpoint = config.checkpoint.strip() or "hog26"
    device = config.device.strip()
    if not device:
        raise ValueError("device must not be empty")
    max_matches = _validate_max_matches(config.max_matches, config.continuous)
    if config.launch_mode not in {"start", "attach"}:
        raise ValueError("launch_mode must be start or attach")
    command = [
        str(python_executable or resolve_agent_python(config.agent_python, root)),
        "-u",
        str(root / "main.py"),
        "--checkpoint",
        checkpoint,
        "--device",
        device,
        "--console",
    ]
    if config.continuous:
        command.append("--continuous")
    if max_matches:
        command.extend(("--max-matches", max_matches))
    if config.launch_mode == "start":
        command.append("--start-battle")
    elif config.launch_mode == "attach":
        command.append("--attach-active")
    if config.log_path.strip():
        command.extend(("--log", config.log_path.strip()))
    return command


def runner_environment() -> dict[str, str]:
    environment = os.environ.copy()
    settings = environment.get("CR_AGENT_SETTINGS")
    if not settings:
        default_settings = ROOT / "settings.local.json"
        if default_settings.is_file():
            environment["CR_AGENT_SETTINGS"] = str(default_settings)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


@dataclass(frozen=True)
class OutputLine:
    stream: str
    text: str


class RoyaleHarnessConsole:
    def __init__(self, root: tk.Tk, *, config_path: Path | str | None = None) -> None:
        self.root = root
        self.root.title("RoyaleHarness 控制台")
        self.root.geometry("1060x720")
        self.root.minsize(820, 560)
        self.config_path = Path(config_path or default_config_path()).expanduser()
        saved = load_config(self.config_path)

        self.process: subprocess.Popen[str] | None = None
        self.process_kind = ""
        self.reader_threads: list[threading.Thread] = []
        self.output_queue: queue.Queue[OutputLine] = queue.Queue()
        self.closing = False
        self._stop_deadline: float | None = None
        self._close_deadline: float | None = None

        self.checkpoint = tk.StringVar(value=saved.checkpoint)
        self.model = self.checkpoint  # Compatibility with the earlier console API.
        self.device = tk.StringVar(value=saved.device)
        self.continuous = tk.BooleanVar(value=saved.continuous)
        self.max_matches = tk.StringVar(value=saved.max_matches)
        self.launch_mode = tk.StringVar(value=saved.launch_mode)
        self.log_path = tk.StringVar(value=saved.log_path)
        self.agent_python = tk.StringVar(value=saved.agent_python)
        self.auto_emotes = tk.BooleanVar(value=saved.auto_emotes)
        self.status = tk.StringVar(value="未运行")
        self.pid = tk.StringVar(value="—")
        self.last_decision = tk.StringVar(value="等待模型启动")
        self._build_ui()
        self._update_launch_controls()
        self.root.after(100, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="RoyaleHarness", font=("Arial", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, text="PID:").pack(side=tk.LEFT, padx=(24, 4))
        ttk.Label(header, textvariable=self.pid).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status, foreground="#1f6f43").pack(side=tk.RIGHT)

        controls = ttk.LabelFrame(outer, text="运行控制", padding=8)
        controls.pack(fill=tk.X, pady=(12, 8))
        self.start_button = ttk.Button(controls, text="启动 Agent", command=self.start)
        self.start_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.stop_button = ttk.Button(controls, text="安全停止", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.force_stop_button = ttk.Button(controls, text="强制停止", command=self.force_stop, state=tk.DISABLED)
        self.force_stop_button.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        self.pause_button = ttk.Button(controls, text="暂停模型", command=lambda: self.send("pause"), state=tk.DISABLED)
        self.pause_button.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        self.resume_button = ttk.Button(controls, text="恢复模型", command=lambda: self.send("resume"), state=tk.DISABLED)
        self.resume_button.grid(row=0, column=4, padx=4, pady=4, sticky="ew")
        self.status_button = ttk.Button(controls, text="刷新状态", command=lambda: self.send("status"), state=tk.DISABLED)
        self.status_button.grid(row=0, column=5, padx=4, pady=4, sticky="ew")
        self.auto_emotes_check = ttk.Checkbutton(
            controls, text="自动发表情", variable=self.auto_emotes,
            command=self.toggle_auto_emotes,
        )
        self.auto_emotes_check.grid(row=1, column=0, columnspan=2, padx=4, pady=4, sticky="w")
        for column in range(6):
            controls.columnconfigure(column, weight=1)

        settings = ttk.LabelFrame(outer, text="Agent 配置", padding=8)
        settings.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(settings, text="Checkpoint").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.checkpoint_box = ttk.Combobox(
            settings,
            textvariable=self.checkpoint,
            values=("hog26", "hog26_proactive", "general", "il", "active_il"),
            width=28,
        )
        self.checkpoint_box.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(settings, text="Device").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=4)
        self.device_box = ttk.Combobox(settings, textvariable=self.device, values=("cpu", "mps", "cuda:0"), width=14)
        self.device_box.grid(row=0, column=3, sticky="ew", padx=4, pady=4)
        self.continuous_check = ttk.Checkbutton(
            settings, text="连续对战", variable=self.continuous, command=self._update_launch_controls
        )
        self.continuous_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(settings, text="最多完成局数（空白=持续）").grid(row=1, column=2, sticky="w", padx=(16, 4), pady=4)
        self.max_matches_entry = ttk.Entry(settings, textvariable=self.max_matches, width=14)
        self.max_matches_entry.grid(row=1, column=3, sticky="ew", padx=4, pady=4)
        ttk.Label(settings, text="启动方式").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        modes = (("start", "启动新对局"), ("attach", "接管当前对局"))
        for index, (value, label) in enumerate(modes, start=1):
            ttk.Radiobutton(
                settings, text=label, variable=self.launch_mode, value=value,
                command=self._update_launch_controls,
            ).grid(row=2, column=index, sticky="w", padx=4, pady=4)
        ttk.Label(settings, text="Python 解释器").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        self.agent_python_box = ttk.Entry(settings, textvariable=self.agent_python)
        self.agent_python_box.grid(row=3, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(settings, text="选择…", command=self.choose_agent_python).grid(row=3, column=3, sticky="e", padx=4, pady=4)
        ttk.Label(settings, text="日志文件").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(settings, textvariable=self.log_path).grid(row=4, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(settings, text="选择…", command=self.choose_log_path).grid(row=4, column=3, sticky="e", padx=4, pady=4)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)

        decision_frame = ttk.LabelFrame(outer, text="实时决策", padding=8)
        decision_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(decision_frame, textvariable=self.last_decision, font=("Menlo", 11)).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(outer, text="stdout / stderr（关键事件）", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(
            log_frame, wrap=tk.NONE, state=tk.DISABLED, font=("Menlo", 10),
            background="#111827", foreground="#e5e7eb", undo=False,
        )
        self.log_text.tag_configure("stderr", foreground="#fca5a5")
        y_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        x_scroll = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

    def _update_launch_controls(self) -> None:
        if not self.continuous.get() and self.max_matches.get().strip():
            self.max_matches.set("")
        state = tk.NORMAL if self.continuous.get() and self.process is None else tk.DISABLED
        self.max_matches_entry.configure(state=state)

    def _set_running(self, running: bool) -> None:
        field_state = tk.DISABLED if running else tk.NORMAL
        for widget in (self.checkpoint_box, self.device_box, self.max_matches_entry, self.agent_python_box):
            widget.configure(state=field_state)
        self.continuous_check.configure(state=field_state)
        self.auto_emotes_check.configure(state=tk.DISABLED if self.closing else tk.NORMAL)
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        for button in (self.stop_button, self.force_stop_button, self.pause_button, self.resume_button, self.status_button):
            button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self._set_launch_radio_state(self.root, running)
        if not running:
            self._update_launch_controls()

    def _set_launch_radio_state(self, widget: tk.Misc, running: bool) -> None:
        try:
            if widget.winfo_class() == "TRadiobutton":
                widget.configure(state=tk.DISABLED if running else tk.NORMAL)
        except tk.TclError:
            return
        for child in widget.winfo_children():
            self._set_launch_radio_state(child, running)

    def _append(self, line: OutputLine | str) -> None:
        output = line if isinstance(line, OutputLine) else OutputLine("desktop", line)
        display = f"[{output.stream}] {output.text}"
        self.log_text.configure(state=tk.NORMAL)
        tag = ("stderr",) if output.stream == "stderr" else ()
        self.log_text.insert(tk.END, display + "\n", tag)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

        if "[decision]" in output.text or "[policy]" in output.text:
            self.last_decision.set(output.text)
        for event in KEY_EVENTS:
            if f"[{event}]" in output.text or f'"event": "{event}"' in output.text:
                self.status.set({
                    "ready": "已就绪",
                    "battle_start": "对局进行中",
                    "decision": "已产生决策",
                    "hand_ack": "已确认出牌",
                    "action_suppressed": "动作已抑制",
                    "action_missed": "动作未确认",
                    "stopped": "已停止",
                }[event])
                break
        if output.stream == "stderr":
            self.status.set("stderr 输出（请检查日志）")

    def _drain_output(self) -> None:
        for _ in range(200):
            try:
                output = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._append(output)
        if self.process is not None and self.process.poll() is not None:
            if not any(thread.is_alive() for thread in self.reader_threads):
                self._process_finished()
        if self.process is not None and self._stop_deadline is not None:
            if self.process.poll() is not None:
                self._stop_deadline = None
            elif time.monotonic() >= self._stop_deadline:
                self.force_stop()
        if self.closing and self.process is not None and self._close_deadline is not None:
            if time.monotonic() >= self._close_deadline:
                self.force_stop()
        if not self.closing or self.process is not None:
            self.root.after(100, self._drain_output)

    def _process_finished(self) -> None:
        process = self.process
        if process is None:
            return
        return_code = process.returncode
        while not self.output_queue.empty():
            self._append(self.output_queue.get_nowait())
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self.process = None
        self._stop_deadline = None
        self._close_deadline = None
        self.pid.set("—")
        self.status.set("已退出" if return_code == 0 else f"异常退出（{return_code}）")
        self._append(f"runner exited: {return_code}")
        self.process_kind = ""
        self._set_running(False)
        if self.closing:
            self.root.destroy()

    def _read_stream(self, stream: Iterable[str], name: str) -> None:
        for line in stream:
            self.output_queue.put(OutputLine(name, line.rstrip("\r\n")))

    def start(self) -> None:
        if self.process is not None:
            return
        config = ConsoleConfig(
            checkpoint=self.checkpoint.get(), device=self.device.get(),
            continuous=self.continuous.get(), max_matches=self.max_matches.get(),
            launch_mode=self.launch_mode.get(), log_path=self.log_path.get(),
            agent_python=self.agent_python.get(), auto_emotes=self.auto_emotes.get(),
        )
        try:
            agent_python = resolve_agent_python(config.agent_python, ROOT)
            error = validate_agent_python(agent_python, ROOT)
            if error:
                raise RuntimeError(f"Agent Python 环境不可用：\n{error}")
            config.agent_python = agent_python
            command = build_runner_command(config, root=ROOT, python_executable=agent_python)
            save_config(config, self.config_path)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("启动参数错误", str(exc))
            return
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=runner_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            self.process = None
            messagebox.showerror("启动失败", str(exc))
            return
        self.process_kind = "runner"
        self.pid.set(str(self.process.pid))
        self.status.set("启动中")
        self._set_running(True)
        self._append("command: " + " ".join(command))
        assert self.process.stdout is not None and self.process.stderr is not None
        self.reader_threads = [
            threading.Thread(target=self._read_stream, args=(self.process.stdout, "stdout"), daemon=True),
            threading.Thread(target=self._read_stream, args=(self.process.stderr, "stderr"), daemon=True),
        ]
        for thread in self.reader_threads:
            thread.start()
        if self.auto_emotes.get():
            self.root.after(500, self._apply_auto_emotes)

    def _apply_auto_emotes(self) -> None:
        if self.process is not None and self.auto_emotes.get():
            self.send("emote auto")

    def toggle_auto_emotes(self) -> None:
        if self.process is not None:
            self.send("emote auto" if self.auto_emotes.get() else "emote off")
            self.status.set("自动表情已开启" if self.auto_emotes.get() else "自动表情已关闭")

    def send(self, command: str) -> None:
        process = self.process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
            self._append(f">> {command}")
        except (BrokenPipeError, OSError) as exc:
            self._append(OutputLine("stderr", f"command failed: {exc}"))

    def stop(self) -> None:
        if self.process is None:
            return
        self.send("stop")
        self.status.set("正在安全停止")
        self._stop_deadline = time.monotonic() + 3.0

    def force_stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        self.status.set("正在强制停止")
        self._stop_deadline = None

    def choose_log_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择日志文件", initialfile=Path(self.log_path.get()).name or "battle.jsonl",
            defaultextension=".jsonl", filetypes=(("JSONL 日志", "*.jsonl"), ("所有文件", "*")),
        )
        if path:
            self.log_path.set(path)

    def choose_agent_python(self) -> None:
        path = filedialog.askopenfilename(title="选择 Agent Python 解释器")
        if path:
            self.agent_python.set(path)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            save_config(ConsoleConfig(
                checkpoint=self.checkpoint.get(), device=self.device.get(),
                continuous=self.continuous.get(), max_matches=self.max_matches.get(),
                launch_mode=self.launch_mode.get(), log_path=self.log_path.get(),
                agent_python=self.agent_python.get(), auto_emotes=self.auto_emotes.get(),
            ), self.config_path)
        except OSError:
            pass
        if self.process is None:
            self.root.destroy()
            return
        self.send("stop")
        self.status.set("关闭中：等待 Agent 停止")
        self._close_deadline = time.monotonic() + 2.0


def main() -> None:
    root = tk.Tk()
    RoyaleHarnessConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
