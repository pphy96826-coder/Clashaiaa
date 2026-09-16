"""Tk desktop console for the RoyaleHarness live runner."""

from __future__ import annotations

import os
import json
from pathlib import Path
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
SETTINGS = Path(os.environ.get("CR_AGENT_SETTINGS", str(
    ROOT / ("settings.device2.local.json" if (ROOT / "settings.device2.local.json").is_file()
            else "settings.local.json")))).expanduser().absolute()
PYTHON = ROOT / ".venv" / "bin" / "python"
# Finder may launch the universal Python.app under Rosetta.  The project
# dependencies include an arm64 PyTorch build, so pin the runner to native
# Apple-silicon execution regardless of how the desktop app was launched.
ARCH = "/usr/bin/arch"


class RoyaleHarnessConsole:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RoyaleHarness 控制台")
        self.root.geometry("980x680")
        self.root.minsize(760, 500)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.process_kind = ""
        self.closing = False

        self.model = tk.StringVar(value="hog26")
        self.max_matches = tk.StringVar(value="")
        self.status = tk.StringVar(value="未运行")
        self.last_decision = tk.StringVar(value="等待模型启动")
        self.prediction_enabled = True
        self._build_ui()
        self.root.after(100, self._drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="RoyaleHarness", font=("Arial", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status, foreground="#1f6f43").pack(side=tk.RIGHT)

        controls = ttk.LabelFrame(outer, text="运行控制", padding=8)
        controls.pack(fill=tk.X, pady=(12, 8))
        self.start_button = ttk.Button(controls, text="启动连续对战", command=lambda: self.start(False))
        self.start_button.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.attach_button = ttk.Button(controls, text="接管当前对局", command=lambda: self.start(True))
        self.attach_button.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.pause_button = ttk.Button(controls, text="暂停模型", command=lambda: self.send("pause"), state=tk.DISABLED)
        self.pause_button.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        self.resume_button = ttk.Button(controls, text="恢复模型", command=lambda: self.send("resume"), state=tk.DISABLED)
        self.resume_button.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        self.status_button = ttk.Button(controls, text="刷新状态", command=lambda: self.send("status"), state=tk.DISABLED)
        self.status_button.grid(row=0, column=4, padx=4, pady=4, sticky="ew")
        self.stop_button = ttk.Button(controls, text="安全停止", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=5, padx=4, pady=4, sticky="ew")
        self.emote_button = ttk.Button(controls, text="开启自动表情", command=lambda: self.send("emote auto"), state=tk.DISABLED)
        self.emote_button.grid(row=0, column=6, padx=4, pady=4, sticky="ew")
        self.emote_off_button = ttk.Button(controls, text="关闭表情", command=lambda: self.send("emote off"), state=tk.DISABLED)
        self.emote_off_button.grid(row=0, column=7, padx=4, pady=4, sticky="ew")
        self.recover_button = ttk.Button(controls, text="恢复运行链路", command=self.recover_runtime)
        self.recover_button.grid(row=1, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
        self.prediction_on_button = ttk.Button(controls, text="开启局面预测", command=lambda: self.set_prediction(True), state=tk.DISABLED)
        self.prediction_on_button.grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        self.prediction_off_button = ttk.Button(controls, text="关闭局面预测", command=lambda: self.set_prediction(False), state=tk.DISABLED)
        self.prediction_off_button.grid(row=1, column=3, padx=4, pady=4, sticky="ew")
        self.engine_button = ttk.Button(controls, text="启动离线引擎", command=self.start_engine)
        self.engine_button.grid(row=1, column=4, columnspan=2, padx=4, pady=4, sticky="ew")
        self.engine_status = tk.StringVar(value="离线引擎：未启动")
        ttk.Label(controls, textvariable=self.engine_status, foreground="#5b6472").grid(
            row=1, column=6, columnspan=2, padx=4, pady=4, sticky="e")
        for column in range(8):
            controls.columnconfigure(column, weight=1)

        model_frame = ttk.LabelFrame(outer, text="模型", padding=8)
        model_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(model_frame, text="当前/切换模型：").pack(side=tk.LEFT)
        self.model_box = ttk.Combobox(
            model_frame,
            textvariable=self.model,
            values=("hog26", "hog26_proactive", "general", "il", "active_il"),
            state="normal",
            width=24,
        )
        self.model_box.pack(side=tk.LEFT, padx=8)
        self.switch_button = ttk.Button(model_frame, text="切换模型", command=self.switch_model, state=tk.DISABLED)
        self.switch_button.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="最多完成局数（空白=持续）：").pack(side=tk.LEFT, padx=(24, 4))
        ttk.Entry(model_frame, textvariable=self.max_matches, width=8).pack(side=tk.LEFT)

        decision_frame = ttk.LabelFrame(outer, text="实时决策", padding=8)
        decision_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(decision_frame, textvariable=self.last_decision, font=("Menlo", 11)).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, wrap=tk.NONE, state=tk.DISABLED, font=("Menlo", 10), background="#111827", foreground="#e5e7eb")
        y_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        x_scroll = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

    def _append(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        if line.startswith("[decision]") or line.startswith("[policy]"):
            self.last_decision.set(line)
        if "[battle_start]" in line:
            self.status.set("对局进行中")
        elif "[battle_terminal]" in line:
            self.status.set("对局结束，准备下一局")
        elif "[lobby_ready]" in line:
            self.status.set("大厅")
        elif "[action_expired]" in line:
            self.status.set("动作超时，已安全停止")
        elif "[action_rejected]" in line:
            self.status.set("动作被拒绝，等待检查")
        elif "[input_fault]" in line:
            self.status.set("输入异常，已停止")
        elif "[execution_halted]" in line or "[lifecycle_error]" in line:
            self.status.set("已停止（需要检查）")
        elif "[model_switched]" in line:
            self.status.set("模型已切换")
        elif "Battle engine ready" in line:
            # bootstrap only proves that the control service/ADB lane is
            # reachable.  A battle slot is created later by the mirror when
            # it sends configure with both live decks.
            self.engine_status.set("离线引擎：服务已启动（对局开始后创建模拟槽）")
        elif "[simulation_mirror_started]" in line:
            self.engine_status.set("离线引擎：模拟槽已创建，模型使用预测")
        elif "[simulation_waiting]" in line:
            self.engine_status.set("离线引擎：正在推进模拟，等待完成")
        elif "[simulation_mirror_drift]" in line:
            self.engine_status.set("离线引擎：镜像偏移，已暂停模拟输入")

    def _drain_output(self) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._append(line)
        if self.process is not None and self.process.poll() is not None:
            self._process_finished()
        self.root.after(100, self._drain_output)

    def _process_finished(self) -> None:
        if self.process is None:
            return
        return_code = self.process.returncode
        self._append(f"[desktop] runner exited: {return_code}")
        self.process = None
        if self.process_kind == "engine":
            self.engine_status.set(
                "离线引擎：服务已启动（对局开始后创建模拟槽）" if return_code == 0
                else "离线引擎：启动失败")
        self.status.set("未运行" if return_code == 0 else "已停止（退出异常）")
        self.process_kind = ""
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        normal = tk.DISABLED if running else tk.NORMAL
        active = tk.NORMAL if running else tk.DISABLED
        self.start_button.configure(state=normal)
        self.attach_button.configure(state=normal)
        for button in (self.pause_button, self.resume_button, self.status_button, self.stop_button,
                       self.switch_button, self.emote_button, self.emote_off_button,
                       self.prediction_on_button, self.prediction_off_button):
            button.configure(state=active)
        self.recover_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.engine_button.configure(state=tk.DISABLED if running else tk.NORMAL)

    def start_engine(self) -> None:
        """Start the separate offline-engine lane through its bootstrap helper."""
        if self.process is not None:
            return
        try:
            settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
            serial = settings.get("full_simulation_serial")
            if not serial or serial == settings.get("adb_serial"):
                raise ValueError("请配置独立的 full_simulation_serial，不能在在线实例安装离线 APK")
            adb_path = Path(settings.get("adb_path", "/usr/local/bin/adb")).expanduser()
            if not adb_path.is_file():
                raise FileNotFoundError(f"找不到 ADB：{adb_path}")
            engine_root = Path(settings.get(
                "full_simulation_root", str(Path.home() / "Downloads" / "clash-royale-battle-engine-main")))
            starter = ROOT / "tools" / "start_full_engine.py"
            if not starter.is_file():
                raise FileNotFoundError(f"找不到离线引擎启动器：{starter}")
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("离线引擎启动失败", str(exc))
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = str(adb_path.parent) + os.pathsep + env.get("PATH", "")
        command = [ARCH, "-arm64", str(PYTHON), "-u", str(starter),
                   "--serial", str(serial), "--port", "26790",
                   "--engine-root", str(engine_root), "--adb", str(adb_path)]
        try:
            self.process = subprocess.Popen(
                command, cwd=engine_root, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except OSError as exc:
            messagebox.showerror("离线引擎启动失败", str(exc))
            self.process = None
            return
        self.process_kind = "engine"
        self.engine_status.set("离线引擎：正在启动")
        self.status.set("正在启动离线引擎")
        self._set_running(True)
        self._append("[desktop] 正在启动离线引擎服务（端口 26790）")
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def start(self, attach_active: bool) -> None:
        if self.process is not None:
            return
        max_matches = self.max_matches.get().strip()
        if max_matches and (not max_matches.isdigit() or int(max_matches) <= 0):
            messagebox.showerror("参数错误", "最多完成局数必须是正整数或留空。")
            return
        command = [
            ARCH,
            "-arm64",
            str(PYTHON),
            "-u",
            "main.py",
            "--checkpoint", self.model.get().strip() or "hog26",
            "--device", "mps",
            "--continuous",
            "--console",
        ]
        if attach_active:
            command.append("--attach-active")
        if max_matches:
            command.extend(["--max-matches", max_matches])
        env = os.environ.copy()
        env["CR_AGENT_SETTINGS"] = str(SETTINGS)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            self.process = None
            return
        self.process_kind = "runner"
        self.status.set("启动中")
        self._set_running(True)
        self._append("[desktop] " + " ".join(command))
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line.rstrip())

    def send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            self._append(f"[desktop] >> {command}")
        except (BrokenPipeError, OSError) as exc:
            self._append(f"[desktop] command failed: {exc}")

    def switch_model(self) -> None:
        model = self.model.get().strip()
        if not model:
            messagebox.showerror("参数错误", "请输入模型名称或权重路径。")
            return
        self.send(f"model {model}")

    def set_prediction(self, enabled: bool) -> None:
        if self.process is None:
            return
        self.send(f"prediction shadow {'on' if enabled else 'off'}")
        self.prediction_enabled = enabled
        self.status.set("局面预测已开启" if enabled else "局面预测已关闭")

    def recover_runtime(self) -> None:
        if self.process is not None:
            self.send("recover")
            return
        env = os.environ.copy()
        env["CR_AGENT_SETTINGS"] = str(SETTINGS)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                [ARCH, "-arm64", str(PYTHON), "-u", "tools/recover_runtime.py"],
                cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self.process_kind = "recover"
            self.status.set("正在恢复链路")
            self._set_running(True)
            self._append("[desktop] 正在执行运行链路恢复")
            self.reader = threading.Thread(target=self._read_output, daemon=True)
            self.reader.start()
        except OSError as exc:
            self._append(f"[desktop] 恢复启动失败: {exc}")

    def stop(self) -> None:
        if self.process is not None:
            self.send("stop")
            self.status.set("正在安全停止")

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.process is not None:
            self.send("stop")
            self.root.after(500, self._close_when_stopped)
        else:
            self.root.destroy()

    def _close_when_stopped(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.root.after(500, self._close_when_stopped)
        else:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RoyaleHarnessConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
