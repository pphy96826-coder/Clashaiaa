import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from tools import inspect_players


class InspectPlayersTests(unittest.TestCase):
    def test_manual_battle_is_attached_before_query(self):
        events = []
        client = Mock()
        responses = [
            {'in_battle': False},
            {'in_battle': True, 'players': [
                {'owner': 0, 'accountId': 123, 'deck': [1, 2, 3], 'hand': [{'slot': 0, 'card_id': 1}]},
                {'owner': 1, 'accountId': 456, 'deck': [4, 5, 6], 'hand': [{'slot': 0, 'card_id': 4}]},
            ]},
        ]

        def attach():
            events.append('attach')
            return {'ok': True, 'attached': True}

        def query():
            events.append('query')
            return responses.pop(0)

        client.attach_live_context.side_effect = attach
        client.query.side_effect = query
        adb = Mock()

        with patch.object(inspect_players, 'Adb', return_value=adb), \
             patch.object(inspect_players, 'ProbeClient', return_value=client), \
             patch.object(inspect_players.time, 'sleep'), \
             redirect_stdout(io.StringIO()) as output:
            result = inspect_players.main()

        self.assertEqual(result, 0)
        self.assertEqual(events[:2], ['attach', 'query'])
        self.assertEqual(client.query.call_count, 2)
        adb.call.assert_called_once_with(
            'forward',
            f'tcp:{inspect_players.config.PROBE_PORT}',
            f'tcp:{inspect_players.config.PROBE_DEVICE_PORT}',
        )
        self.assertIn('"account_id": 123', output.getvalue())
        self.assertIn('"account_id": 456', output.getvalue())


if __name__ == '__main__':
    unittest.main()
