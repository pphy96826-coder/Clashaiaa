"""One-click, no-game-input ADB/probe recovery for the desktop console."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from bridge.probe_client import ProbeClient
from live_lifecycle import LiveLifecycle, LifecycleError


def log(event, **data):
    print(f'[{event}] {data}', flush=True)


def main():
    probe = ProbeClient(port=config.PROBE_PORT, account_id=config.LOCAL_ACCOUNT_ID)
    lifecycle = LiveLifecycle(probe, log=log, calibration_path=config.LIFECYCLE_CALIBRATION_PATH)
    try:
        result = lifecycle.recover_runtime()
        print(f'[console] 运行链路已恢复: {result}', flush=True)
        return 0
    except (LifecycleError, OSError, RuntimeError) as exc:
        print(f'[console] 运行链路恢复失败: {exc}', flush=True)
        return 1
    finally:
        lifecycle.close()


if __name__ == '__main__':
    raise SystemExit(main())
