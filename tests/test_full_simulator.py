import unittest
from pathlib import Path

import json

from bridge.full_simulator import (FullBattleSimulator, FullSimulationUnavailable,
                                   LiveSnapshotEnvelope)
from bridge.live_replay_mirror import LiveReplayMirror, MirrorAction


class FullSimulatorTests(unittest.TestCase):
    def test_downloaded_engine_is_detected_without_starting_a_rollout(self):
        root = Path.home() / 'Downloads' / 'clash-royale-battle-engine-main'
        simulator = FullBattleSimulator(root=root)
        self.assertEqual(simulator.available, (root / 'engine.py').is_file())

    def test_live_prediction_refuses_unrepresentable_state(self):
        simulator = FullBattleSimulator(root='/does/not/exist')
        with self.assertRaises(FullSimulationUnavailable):
            simulator.predict_live({'tick': 1}, 20)

    def test_rebuilt_engine_supports_verified_26_action_cards(self):
        simulator = FullBattleSimulator(root='/does/not/exist')
        self.assertEqual(simulator.unsupported_action_cards(
            (26000010, 26000014, 27000000, 26000021,
             28000000, 28000011, 26000030, 26000038)),
            ())
        self.assertEqual(simulator.unsupported_action_cards((99999999,)), (99999999,))

    def test_snapshot_envelope_is_versioned_and_integrity_checked(self):
        envelope = LiveSnapshotEnvelope({'version': 'observation.v1', 'tick': 12})
        wire = json.loads(envelope.to_bytes())
        self.assertEqual(wire['schema'], 'live-snapshot.v1')
        self.assertEqual(len(wire['sha256']), 64)
        self.assertTrue(wire['payload'])

    def test_mirror_uses_one_engine_and_advances_incrementally(self):
        class Entity:
            def __init__(self, owner, card, hp=1000):
                self.owner, self.card, self.hp = owner, card, hp

        class State:
            def __init__(self, tick=90):
                self.tick, self.entities = tick, []

        class Engine:
            def __init__(self):
                self.state = State()
                self.steps = 0

            def reset(self, seed=1):
                return self.state

            def step(self, actions=(), ticks=1):
                self.steps += 1
                self.state.tick += ticks
                return type('Transition', (), {'state': self.state, 'executed': tuple(True for _ in actions)})()

            def close(self):
                pass

        engines = []
        def make_engine():
            engine = Engine()
            engines.append(engine)
            return engine

        mirror = LiveReplayMirror(make_engine, lambda o, s, x, y: (o, s, x, y))
        mirror.start()
        live = type('Live', (), {'tick': 101, 'entities': []})()
        mirror.sync(live, [MirrorAction(0, 2, 26000010, 1000, 12000, 95)])
        self.assertTrue(mirror.active)
        self.assertEqual(len(engines), 1)
        self.assertGreaterEqual(engines[0].steps, 2)

    def test_mirror_invalidates_on_own_count_drift(self):
        class State:
            tick = 90
            entities = []
        class Engine:
            def reset(self, seed=1): return State()
            def step(self, actions=(), ticks=1):
                State.tick += ticks
                return type('Transition', (), {'state': State(), 'executed': (True,)})()
            def close(self): pass
        mirror = LiveReplayMirror(Engine, lambda *args: args)
        mirror.start()
        live = type('Live', (), {'tick': 90, 'entities': [
            {'owner': 0, 'cardId': 26000010, 'hp': 1000}]})()
        self.assertIsNone(mirror.sync(live))
        self.assertFalse(mirror.active)
        self.assertIn('count drift', mirror.drift.reason)

    def test_mirror_reports_unsupported_action_without_throwing(self):
        class State:
            tick = 90
            entities = []
        class Engine:
            def reset(self, seed=1): return State()
            def snapshot(self): return {'handle': 1}
            def release_snapshot(self, handle): pass
            def supports_card(self, card_id): return False
            def close(self): pass
        mirror = LiveReplayMirror(Engine, lambda *args: args)
        mirror.start()
        live = type('Live', (), {'tick': 90, 'entities': []})()
        self.assertIsNone(mirror.sync(live, [MirrorAction(0, 0, 26000010, 1, 1, 90)]))
        self.assertIn('unsupported', mirror.drift.reason)


if __name__ == '__main__':
    unittest.main()
