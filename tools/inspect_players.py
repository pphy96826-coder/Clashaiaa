"""Print player identities/decks from a manual battle. Never sends touch input."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from bridge.probe_client import ProbeClient
from tools.install_probe import Adb


def main():
    adb = Adb()
    adb.call('forward', f'tcp:{config.PROBE_PORT}', f'tcp:{config.PROBE_DEVICE_PORT}')
    client = ProbeClient(port=config.PROBE_PORT)
    # This tool is intentionally used after the operator has already entered
    # a manual battle. Adopt that live controller explicitly; GET alone cannot
    # cross the probe's stale-result/previous-battle lifecycle barrier.
    client.attach_live_context()
    raw = None
    for _ in range(10):
        raw = client.query()
        if raw is not None and raw.get('in_battle'):
            break
        time.sleep(0.05)
    if raw is None:
        raise RuntimeError(client.last_error)
    if not raw.get('in_battle'):
        print('Start a manual friendly battle with different decks, then run this command again. Keep AI stopped.')
        return 1
    print(json.dumps([{'owner': p.get('owner'), 'account_id': p.get('accountId'),
                       'deck_ids': p.get('deck'), 'hand': p.get('hand')}
                      for p in raw.get('players', [])], ensure_ascii=False, indent=2))
    print('Match your visible hand/deck to one row. Save that account_id, not owner, in settings.local.json.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
