import copy
from dataclasses import replace
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import config
from bridge.probe_client import ProbeClient
from bridge.coordinates import ScreenCalibration, action_world, world_to_view
from agent.feature_adapter import FeatureAdapter, HOG_26_DECK, TelemetryError
from agent.execution import ActionExecutor
from native_runner.contracts import ActionV1, ActionKind, TargetKind
from native_runner.perspective import PerspectiveTransformV1


def opening(owner=0, cycle=True):
    deck = list(HOG_26_DECK)
    data = {'in_battle': True, 'tick': 90, 'players': [], 'entities': [], 'crowns': [0, 0]}
    for seat in (0, 1):
        player = {'owner': seat, 'accountId': 123 if seat == owner else 456, 'elixir': 10,
            'deck': deck, 'hand': [{'slot': s, 'card_id': c} for s, c in enumerate(deck[:4])]}
        if cycle:
            player['cycle'] = deck[4:]
        data['players'].append(player)
        for idx, (x, y, hp) in enumerate(((9000, 3000, 7000), (3500, 6500, 4000), (14500, 6500, 4000))):
            data['entities'].append({'id': 5000000+seat*3+idx, 'owner': seat, 'card_id': -1,
                'x': x, 'y': y if seat == 0 else 32000-y, 'hp': hp, 'max_hp': hp,
                'tower_troop_id': None if idx == 0 else 159000000})
    return data


def state(owner=0, cycle=True):
    return ProbeClient(account_id=123).parse(opening(owner, cycle))


def play(owner=0, slot=0, card=26000010, grid=(3, 10), subcell=None, delay=1):
    return ActionV1(owner=owner, kind=ActionKind.PLAY_CARD, hand_slot=slot, card_id=card,
        target_kind=TargetKind.GRID, target_grid=grid, subcell_offset=subcell,
        execute_offset_ticks=delay, next_decision_ticks=5,
        metadata={'policy_effective_cost': 1.0})


def add_enemy(s, entity_id, x, y, card_id=26000021, hp=1000):
    s.entities.append({'id': entity_id, 'owner': 1-s.local_owner, 'card_id': card_id,
        'x': x, 'y': y, 'hp': hp, 'max_hp': hp})


class CoordinateTests(unittest.TestCase):
    def test_cell_center_and_building_anchor(self):
        self.assertEqual(action_world(play(grid=(7, 9))), (7500, 9500))
        self.assertEqual(action_world(play(grid=(7, 9), subcell=(.5, .5))), (8000, 10000))

    def test_all_cells_both_seats_and_mirrors(self):
        # Decoder returns native coordinates. Apply the screen transform once.
        calibration = ScreenCalibration()
        for owner in (0, 1):
            for mirror in (False, True):
                perspective = PerspectiveTransformV1(owner, horizontal_mirror=mirror)
                for x in range(18):
                    for y in range(32):
                        model = play(owner=owner, grid=(x, y))
                        native = perspective.action_to_native(model)
                        self.assertEqual(perspective.action_to_model(native).target_grid, (x, y))
                        vx, vy = world_to_view(*action_world(native), owner)
                        self.assertEqual(vy, y+.5)
                        self.assertEqual(vx, x+.5 if mirror else 17.5-x)
                        calibration.project(vx, vy)

    def test_live_left_side_evidence_owner_zero(self):
        x, y = world_to_view(11500, 9500, 0)
        self.assertEqual((x, y), (6.5, 9.5))
        self.assertLess(ScreenCalibration().project(x, y)[0], 540)

    def test_subcell_sign_survives_seat_rotation(self):
        for owner in (0, 1):
            p = PerspectiveTransformV1(owner, horizontal_mirror=True)
            native = p.action_to_native(play(owner, grid=(5, 8), subcell=(.5, .5)))
            # Model X is mirrored but its offset is model-relative too.
            self.assertEqual(world_to_view(*action_world(native), owner), (6, 9))

    def test_resolution_scaling_and_rejection(self):
        c = ScreenCalibration()
        px, py = c.project(9, 16)
        self.assertEqual(c.project(9, 16, (540, 960)), (round(px/2), round(py/2)))
        with self.assertRaises(ValueError):
            c.project(9, 16, (1920, 1080))
        with self.assertRaises(ValueError):
            c.project(float('nan'), 16)


class ProbeTests(unittest.TestCase):
    def test_identity_wins_over_mirror_match_and_array_order(self):
        for owner in (0, 1):
            raw = opening(owner)
            raw['local_owner'] = 0  # the legacy hardcoded field is untrusted
            raw['players'].reverse()
            self.assertEqual(ProbeClient(account_id=123).parse(raw).local_owner, owner)

    def test_ambiguous_identity_is_rejected(self):
        p = ProbeClient()
        p.expected_deck = HOG_26_DECK
        with self.assertRaises(ValueError):
            p.parse(opening())

    def test_hole_and_out_of_order_slots_are_preserved(self):
        d = opening()
        d['players'][0]['hand'] = [{'slot': 3, 'card_id': 26000021}, {'slot': 1, 'card_id': 26000014}]
        self.assertEqual(ProbeClient(account_id=123).parse(d).hand_cards, [0,26000014,0,26000021])

    def test_stale_then_new_battle_requires_tick_progress(self):
        p = ProbeClient(account_id=123)
        d = opening()
        p.query = lambda: copy.deepcopy(d)
        with patch('bridge.probe_client.time.perf_counter', return_value=1):
            self.assertIsNone(p.get_live_battle_state())
            d['tick'] += 1
            self.assertIsNotNone(p.get_live_battle_state())
        with patch('bridge.probe_client.time.perf_counter', return_value=3):
            self.assertIsNone(p.get_live_battle_state())
            d['tick'] = 0
            self.assertIsNone(p.get_live_battle_state())
            d['tick'] = 1
            self.assertIsNotNone(p.get_live_battle_state())


class AdapterTests(unittest.TestCase):
    def adapter(self, owner=0, cycle=True):
        s = state(owner, cycle)
        adapter = FeatureAdapter()
        adapter.reset_match(s, 'test')
        return adapter, s

    def test_true_seat_and_native_positions_preserved(self):
        for owner in (0, 1):
            a,s = self.adapter(owner)
            b,o = a.tensorize(s)
            self.assertEqual(o.owner, owner)
            self.assertEqual(b.match_scalars[0,17].item(), float(owner==0))
            self.assertEqual(o.towers[0].position, (9000,3000))

    def test_missing_cycle_is_unknown(self):
        a,s = self.adapter(cycle=False)
        b,o = a.tensorize(s)
        self.assertIsNone(o.players[0].next_card)
        self.assertEqual(o.players[0].cycle, ())
        self.assertIsNone(a.tensorizer.tracker)

    def test_exact_cycle_and_five_minute_model_clock(self):
        a,s = self.adapter()
        _,o = a.tensorize(s)
        self.assertEqual(o.players[0].next_card, HOG_26_DECK[4])
        self.assertEqual(o.time.remaining_ms, 300000-90*50)

    def test_all_eight_base_hog_cards_reach_candidates_in_both_seats(self):
        # Ordinary Musketeer/Ice Golem also list optional hero abilities.
        # Verify the final model candidates, not just the adapter's boolean.
        for owner in (0, 1):
            for hand in (HOG_26_DECK[:4], HOG_26_DECK[4:]):
                raw = opening(owner)
                me = next(p for p in raw['players'] if p['accountId'] == 123)
                me['hand'] = [{'slot': i, 'card_id': cid} for i, cid in enumerate(hand)]
                me['cycle'] = [cid for cid in HOG_26_DECK if cid not in hand]
                s = ProbeClient(account_id=123).parse(raw)
                a = FeatureAdapter()
                a.reset_match(s, 'all-base-cards')
                b, o = a.tensorize(s)
                self.assertEqual(o.action_mask.hand_slots, (True,) * 4)
                self.assertEqual(int(b.candidates.mask.sum()), 4)
                for slot in range(4):
                    self.assertEqual(o.action_mask.reasons['slot_reasons'][str(slot)], 'playable')

    def test_stalled_hand_keeps_both_defenders_playable(self):
        raw = opening()
        hand = (28000011, 26000038, 26000014, 28000000)
        raw['players'][0]['hand'] = [{'slot': i, 'card_id': cid} for i, cid in enumerate(hand)]
        raw['players'][0]['cycle'] = [cid for cid in HOG_26_DECK if cid not in hand]
        s = ProbeClient(account_id=123).parse(raw)
        a = FeatureAdapter()
        a.reset_match(s, 'stalled-hand')
        _, o = a.tensorize(s)
        self.assertTrue(a.bundle.card_specs[26000014].ability_ids)
        self.assertTrue(a.bundle.card_specs[26000038].ability_ids)
        self.assertEqual(o.action_mask.hand_slots, (True,) * 4)
        _, o = a.tensorize(s, blocked_slots={1}, reserved_elixir=7)
        self.assertEqual(o.action_mask.reasons['slot_reasons']['1'], 'pending_or_cooldown')
        self.assertEqual(o.action_mask.reasons['slot_reasons']['2'], 'insufficient_elixir')

    def test_single_lane_threat_masks_wrong_lane_before_policy(self):
        a, s = self.adapter()
        add_enemy(s, 9001, 3500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['defensive_threat_lane'], 'left')
        defensive = o.action_mask.placement_masks['0']  # Skeletons
        self.assertTrue(any(row[x] for row in defensive['row_major'] for x in range(0, 9)))
        self.assertFalse(any(row[x] for row in defensive['row_major'] for x in range(9, 18)))

        # Hog Rider is an offensive win condition and is intentionally not
        # constrained by this conservative defensive-lane gate.
        hog = o.action_mask.placement_masks['2']
        self.assertTrue(any(row[x] for row in hog['row_major'] for x in range(9, 18)))

    def test_split_lane_threat_does_not_mask_defensive_lane(self):
        a, s = self.adapter()
        add_enemy(s, 9001, 3500, 11000, card_id=26000021)
        add_enemy(s, 9002, 14500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertIsNone(o.action_mask.reasons['defensive_threat_lane'])
        defensive = o.action_mask.placement_masks['0']
        self.assertTrue(any(row[x] for row in defensive['row_major'] for x in range(0, 9)))
        self.assertTrue(any(row[x] for row in defensive['row_major'] for x in range(9, 18)))

    def test_low_elixir_near_tower_pressure_holds_hog(self):
        a, s = self.adapter()
        s.elixir = 6.0
        add_enemy(s, 9101, 3500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertFalse(o.action_mask.hand_slots[2])  # Hog Rider
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['2'],
            'strategy_hold_attack_defense',
        )
        self.assertEqual(
            o.action_mask.reasons['attack_hold_reason'],
            'near_tower_defense',
        )
        self.assertLessEqual(
            o.action_mask.reasons['attack_hold_distance'],
            7000.0,
        )

    def test_high_elixir_can_keep_hog_available_under_pressure(self):
        a, s = self.adapter()
        s.elixir = 9.0
        add_enemy(s, 9102, 3500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertTrue(o.action_mask.hand_slots[2])
        self.assertIsNone(o.action_mask.reasons['attack_hold_reason'])

    def test_distant_enemy_does_not_hold_hog(self):
        a, s = self.adapter()
        # High elixir isolates the near-tower attack-hold rule from neutral
        # patience, which intentionally throttles four-elixir commitments.
        s.elixir = 9.0
        add_enemy(s, 9103, 3500, 16000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertTrue(o.action_mask.hand_slots[2])
        self.assertIsNone(o.action_mask.reasons['attack_hold_reason'])

    def test_surviving_support_opens_counterpush_phase_and_hog_lane(self):
        a, s = self.adapter()
        s.elixir = 5.0
        s.entities.append({
            'id': 9201,
            'owner': s.local_owner,
            'card_id': 26000014,
            'x': 14500,
            'y': 12000,
            'hp': 700,
            'max_hp': 1000,
        })
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'counterpush')
        self.assertFalse(o.action_mask.reasons['neutral_patience_active'])
        self.assertEqual(o.action_mask.reasons['counterpush_lane'], 'right')
        self.assertEqual(o.action_mask.reasons['counterpush_support_entity_id'], 9201)
        hog = o.action_mask.placement_masks['2']
        self.assertFalse(any(row[x] for row in hog['row_major'] for x in range(0, 9)))
        self.assertTrue(any(row[x] for row in hog['row_major'] for x in range(9, 18)))

    def test_near_tower_pressure_blocks_counterpush_phase(self):
        a, s = self.adapter()
        s.elixir = 9.0
        s.entities.append({
            'id': 9202,
            'owner': s.local_owner,
            'card_id': 26000014,
            'x': 14500,
            'y': 12000,
            'hp': 700,
            'max_hp': 1000,
        })
        add_enemy(s, 9203, 14500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'defend')
        self.assertIsNone(o.action_mask.reasons['counterpush_lane'])
        # At high elixir Hog is still legal; the defend phase simply prevents
        # the support-lane counterpush bias from activating prematurely.
        self.assertTrue(o.action_mask.hand_slots[2])

    def test_split_surviving_support_does_not_force_hog_lane(self):
        a, s = self.adapter()
        for entity_id, x in ((9204, 3500), (9205, 14500)):
            s.entities.append({
                'id': entity_id,
                'owner': s.local_owner,
                'card_id': 26000014,
                'x': x,
                'y': 12000,
                'hp': 700,
                'max_hp': 1000,
            })
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertIsNone(o.action_mask.reasons['counterpush_lane'])
        hog = o.action_mask.placement_masks['2']
        self.assertTrue(any(row[x] for row in hog['row_major'] for x in range(0, 9)))
        self.assertTrue(any(row[x] for row in hog['row_major'] for x in range(9, 18)))

    def test_owner_one_enemy_on_opponent_half_does_not_fake_defend(self):
        a, s = self.adapter(owner=1)
        s.elixir = 7.0
        # Owner 1 defends high Y.  An enemy still at low Y is near its own
        # baseline and must not be interpreted as pressure on us.
        add_enemy(s, 9291, 3500, 7000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertTrue(o.action_mask.reasons['neutral_patience_active'])
        self.assertIsNone(o.action_mask.reasons['defensive_threat_lane'])

    def test_owner_one_enemy_on_our_half_enters_defend(self):
        a, s = self.adapter(owner=1)
        s.elixir = 7.0
        # High Y is owner 1's side of the arena.
        add_enemy(s, 9292, 14500, 22000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'defend')
        self.assertFalse(o.action_mask.reasons['neutral_patience_active'])
        self.assertEqual(o.action_mask.reasons['defensive_threat_lane'], 'right')

    def test_neutral_patience_holds_heavy_commitments_but_keeps_wait_and_cycle(self):
        a, s = self.adapter()
        s.elixir = 7.0
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertTrue(o.action_mask.reasons['neutral_patience_active'])
        self.assertTrue(o.action_mask.kinds['wait'])
        self.assertTrue(o.action_mask.hand_slots[0])  # Skeletons stay available.
        self.assertFalse(o.action_mask.hand_slots[1])  # Musketeer held.
        self.assertFalse(o.action_mask.hand_slots[2])  # Hog held.
        self.assertTrue(o.action_mask.hand_slots[3])  # Ice Spirit stays available.
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['1'],
            'strategy_neutral_patience',
        )
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['2'],
            'strategy_neutral_patience',
        )

    def test_neutral_patience_holds_cannon_but_keeps_cheap_cycle(self):
        raw = opening()
        hand = (27000000, 26000010, 26000030, 26000038)
        raw['players'][0]['hand'] = [
            {'slot': i, 'card_id': cid} for i, cid in enumerate(hand)
        ]
        raw['players'][0]['cycle'] = [
            cid for cid in HOG_26_DECK if cid not in hand
        ]
        raw['players'][0]['elixir'] = 7.0
        s = ProbeClient(account_id=123).parse(raw)
        a = FeatureAdapter()
        a.reset_match(s, 'neutral-patience-cannon')

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertTrue(o.action_mask.reasons['neutral_patience_active'])
        self.assertFalse(o.action_mask.hand_slots[0])  # Cannon held.
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['0'],
            'strategy_neutral_patience',
        )
        self.assertTrue(o.action_mask.hand_slots[1])  # Skeletons available.
        self.assertTrue(o.action_mask.hand_slots[2])  # Ice Spirit available.
        self.assertTrue(o.action_mask.hand_slots[3])  # Ice Golem available.
        self.assertTrue(o.action_mask.kinds['wait'])

    def test_defensive_pressure_releases_cannon_from_neutral_patience(self):
        raw = opening()
        hand = (27000000, 26000010, 26000030, 26000038)
        raw['players'][0]['hand'] = [
            {'slot': i, 'card_id': cid} for i, cid in enumerate(hand)
        ]
        raw['players'][0]['cycle'] = [
            cid for cid in HOG_26_DECK if cid not in hand
        ]
        raw['players'][0]['elixir'] = 7.0
        s = ProbeClient(account_id=123).parse(raw)
        add_enemy(s, 9391, 3500, 11000, card_id=26000021)
        s.tick += 1
        a = FeatureAdapter()
        a.reset_match(s, 'defend-releases-cannon')

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'defend')
        self.assertFalse(o.action_mask.reasons['neutral_patience_active'])
        self.assertTrue(o.action_mask.hand_slots[0])
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['0'],
            'playable',
        )

    def test_recent_enemy_building_expiry_opens_short_hog_window(self):
        a, s = self.adapter()
        s.elixir = 6.0
        building_id = 9451
        s.entities.append({
            'id': building_id,
            'owner': 1 - s.local_owner,
            'card_id': 27000000,
            'x': 9000,
            'y': 22000,
            'hp': 1000,
            'max_hp': 1000,
        })
        s.tick += 1
        _, active = a.tensorize(s)
        self.assertFalse(active.action_mask.hand_slots[2])
        self.assertIsNone(active.action_mask.reasons['attack_window_reason'])

        s.entities[:] = [e for e in s.entities if e['id'] != building_id]
        s.tick += 1
        _, expired = a.tensorize(s)

        self.assertEqual(expired.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertTrue(expired.action_mask.reasons['neutral_patience_active'])
        self.assertEqual(
            expired.action_mask.reasons['attack_window_reason'],
            'recent_enemy_building_expired',
        )
        self.assertEqual(expired.action_mask.reasons['attack_window_card_id'], 27000000)
        self.assertTrue(expired.action_mask.hand_slots[2])  # Hog is released.

    def test_recent_building_attack_window_expires_back_into_patience(self):
        a, s = self.adapter()
        s.elixir = 6.0
        building_id = 9452
        s.entities.append({
            'id': building_id,
            'owner': 1 - s.local_owner,
            'card_id': 27000000,
            'x': 9000,
            'y': 22000,
            'hp': 1000,
            'max_hp': 1000,
        })
        s.tick += 1
        a.tensorize(s)
        s.entities[:] = [e for e in s.entities if e['id'] != building_id]
        s.tick += 1
        a.tensorize(s)

        s.tick += 51
        _, o = a.tensorize(s)

        self.assertIsNone(o.action_mask.reasons['attack_window_reason'])
        self.assertFalse(o.action_mask.hand_slots[2])
        self.assertEqual(
            o.action_mask.reasons['slot_reasons']['2'],
            'strategy_neutral_patience',
        )

    def test_exact_building_cycle_window_stays_open_until_four_public_plays(self):
        a, _ = self.adapter()
        a._recent_enemy_building_expiry = {
            'entity_id': 9501,
            'card_id': 27000000,
            'tick': 100,
        }
        a._opponent_last_play_count[27000000] = 7
        a._opponent_exact_play_count = 9

        window = a._exact_building_cycle_window()

        self.assertEqual(window['reason'], 'enemy_building_out_of_cycle')
        self.assertEqual(window['plays_since'], 2)
        self.assertEqual(window['plays_until_return'], 2)

        a._opponent_exact_play_count = 11
        self.assertIsNone(a._exact_building_cycle_window())

    def test_low_elixir_attack_window_requires_tight_public_bound(self):
        open_window = FeatureAdapter._low_elixir_attack_window(
            6.0, (3.25, 4.0), defensive_pressure=False)
        self.assertEqual(open_window['reason'], 'opponent_low_elixir')
        self.assertIsNone(FeatureAdapter._low_elixir_attack_window(
            6.0, (1.0, 5.0), defensive_pressure=False))
        self.assertIsNone(FeatureAdapter._low_elixir_attack_window(
            6.0, (3.25, 4.0), defensive_pressure=True))

    def test_neutral_patience_releases_near_elixir_cap(self):
        a, s = self.adapter()
        s.elixir = 8.5
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'neutral')
        self.assertFalse(o.action_mask.reasons['neutral_patience_active'])
        self.assertTrue(o.action_mask.hand_slots[1])
        self.assertTrue(o.action_mask.hand_slots[3])

    def test_defensive_pressure_immediately_releases_neutral_patience(self):
        a, s = self.adapter()
        s.elixir = 6.0
        add_enemy(s, 9301, 3500, 11000, card_id=26000021)
        s.tick += 1

        _, o = a.tensorize(s)

        self.assertEqual(o.action_mask.reasons['strategy_phase'], 'defend')
        self.assertFalse(o.action_mask.reasons['neutral_patience_active'])
        # Attack-hold may still suppress Hog, but defensive resources must be
        # restored immediately rather than waiting for 8.5 elixir.
        self.assertTrue(o.action_mask.hand_slots[1])

    def test_missing_initial_towers_do_not_open_pockets(self):
        s = state()
        s.entities.pop()
        with self.assertRaises(TelemetryError):
            FeatureAdapter().reset_match(s, 'bad')

    def test_destroyed_tower_crowns_and_pocket_side(self):
        a,s = self.adapter()
        s.entities[-1]['hp'] = 0
        s.tick += 5
        _,o = a.tensorize(s)
        self.assertEqual(o.players[0].crowns, 1)
        self.assertEqual(a.destroyed_enemy_princess_lanes, {'right'})

    def test_unknown_unit_does_not_get_parent_archetype(self):
        a,s = self.adapter()
        s.entities.append({'id': 900,'owner':1,'card_id':26000007,'x':9000,'y':17000,'hp':100,'max_hp':100})
        s.tick += 5
        _,o = a.tensorize(s)
        self.assertIsNone(o.entities[0].native_data_global_id)

    def test_slot_lock_and_resource_reservation(self):
        a,s = self.adapter()
        _,o = a.tensorize(s, blocked_slots={0}, reserved_elixir=9)
        self.assertFalse(o.action_mask.hand_slots[0])
        self.assertFalse(o.action_mask.hand_slots[1])
        self.assertEqual(o.action_mask.reasons['reserved_elixir'], 9)

    def test_invalid_deck_is_not_filled(self):
        s = state()
        s.deck_cards = s.deck_cards[:7]
        with self.assertRaises(TelemetryError):
            FeatureAdapter().reset_match(s, 'bad')

    def test_untouchable_bottom_cell_is_removed_before_inference(self):
        for owner in (0, 1):
            a,s = self.adapter(owner)
            _,o = a.tensorize(s)
            entry = o.action_mask.placement_masks['0']
            self.assertFalse(any(entry['row_major'][0 if owner == 0 else 31]))

    def test_dead_unit_without_exact_archetype_is_removed(self):
        a,s = self.adapter()
        s.entities.append({'id':900,'owner':1,'card_id':26000021,'x':3000,'y':15000,'hp':0,'max_hp':1000})
        s.tick+=5
        _,o=a.tensorize(s)
        self.assertEqual(o.entities,())

    def test_minus_one_card_id_does_not_make_a_projectile_a_tower(self):
        a,s=self.adapter()
        cat=a.bundle.entity_archetype_catalog
        gid=next(gid for gid,vid in cat._runtime_vocab_by_global_id.items()
                 if cat.metadata_for_vocab_id(vid).child_kind=='projectile')
        s.entities.append({'id':4000010,'owner':1,'card_id':-1,'x':9000,'y':16000,
                           'hp':0,'max_hp':0,'native_data_global_id':gid})
        s.tick+=5
        _,o=a.tensorize(s)
        self.assertEqual(o.entities[0].entity_kind,'projectile')
        self.assertIsNone(o.entities[0].card_id)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.actuator = Mock(calibration=ScreenCalibration())
        self.actuator.deploy_action.return_value = {'input_ms': 2, 'screen': [1,1]}
        self.executor = ActionExecutor(self.actuator, lambda event, **data: self.events.append((event,data)))

    def tearDown(self):
        self.executor.close()

    def test_wait_does_not_touch(self):
        self.executor.submit(SimpleNamespace(actions=(ActionV1(owner=0,kind=ActionKind.WAIT),)), state())
        self.assertFalse(self.executor.pending)
        self.actuator.deploy_action.assert_not_called()

    def test_action_budget_waits_for_background_ack_to_settle(self):
        s = state()
        self.executor.max_actions = 1
        self.executor.submit(SimpleNamespace(actions=(play(),)), s)
        self.assertFalse(self.executor.action_budget_exhausted)
        pending = self.executor.pending[0]
        now = time.perf_counter()
        pending.due = now - 0.01
        pending.expires = now + 1.0
        self.executor.poll(s, lambda *_: True)
        self.assertTrue(self.executor.action_budget_exhausted)
        self.assertFalse(self.executor.action_budget_settled)

        # The worker is still the active serialized touch until completion.
        self.executor.future = None
        self.executor.active = None
        if pending in self.executor.pending:
            self.executor.pending.remove(pending)
        self.executor.ack_watch.append(pending)
        self.assertFalse(self.executor.action_budget_settled)

        self.executor.ack_watch.clear()
        self.assertTrue(self.executor.action_budget_settled)

    def test_live_revalidation_rejection_does_not_consume_action_budget(self):
        s = state()
        self.executor.max_actions = 1
        self.executor.submit(SimpleNamespace(actions=(play(),)), s)
        pending = self.executor.pending[0]
        now = time.perf_counter()
        pending.due = now - 0.01
        pending.expires = now + 1.0

        self.executor.poll(s, lambda *_: False)

        self.assertEqual(self.executor.attempted_actions, 0)
        self.assertFalse(self.executor.action_budget_exhausted)
        self.assertFalse(self.executor.pending)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'action_rejected']
        self.assertIn('live_revalidation', reasons)

    def test_input_started_consumes_action_budget(self):
        s = state()
        self.executor.max_actions = 1
        self.executor.submit(SimpleNamespace(actions=(play(),)), s)
        pending = self.executor.pending[0]
        now = time.perf_counter()
        pending.due = now - 0.01
        pending.expires = now + 1.0

        self.executor.poll(s, lambda *_: True)

        self.assertEqual(self.executor.attempted_actions, 1)
        self.assertTrue(self.executor.action_budget_exhausted)
        self.assertIn('input_started', [event for event, _ in self.events])

    def test_second_slot_waits_for_first_outcome(self):
        s=state()
        self.executor.submit(SimpleNamespace(actions=(play(),play(slot=1,card=26000014,delay=4))),s)
        self.assertEqual(self.executor.blocked_slots(s),{0})
        self.assertEqual(self.executor.reserved_elixir,1)
        self.assertEqual(len(self.executor.pending),1)
        self.assertIn('in_flight_capacity', [d.get('reason') for e,d in self.events])

    def test_input_completion_moves_to_ack_watch_and_provisional_reservation(self):
        s = state()
        add_enemy(s, 9001, 3500, 11500)
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 11))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = time.perf_counter()
        pending.prior_elixir = s.elixir
        self.executor.active = pending
        future = Mock()
        future.done.return_value = True
        future.result.return_value = {'input_ms': 2, 'screen': [1, 1]}
        self.executor.future = future

        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.pending)
        self.assertEqual(self.executor.ack_watch, [pending])
        self.assertEqual(len(self.executor.threat_reservations), 1)
        self.assertEqual(self.executor.threat_reservations[0].confidence, 'provisional')
        self.assertTrue(self.executor.consume_fresh_state_required())
        events = [event for event, _ in self.events]
        self.assertIn('ack_watch_started', events)

    def test_new_source_entity_confirms_troop_without_hand_rotation(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 10))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = time.perf_counter() - 0.15
        pending.input_completed_at = time.perf_counter() - 0.10
        pending.prior_elixir = s.elixir
        pending.prior_entities = frozenset(e['id'] for e in s.entities)
        self.executor.ack_watch.append(pending)

        # Hand/elixir still look stale, but a brand-new own entity from the
        # exact source card is positive evidence that the deployment landed.
        new_id = 9009
        s.entities.append({
            'id': new_id,
            'owner': s.local_owner,
            'card_id': first.card_id,
            'x': 3500,
            'y': 10500,
            'hp': 100,
            'max_hp': 100,
        })
        s.tick += 1
        s.received_at = pending.input_completed_at + 0.01

        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.ack_watch)
        acks = [data for event, data in self.events if event == 'hand_ack']
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]['evidence'], 'new_source_entity')
        self.assertEqual(acks[0]['spawn_entity_id'], new_id)
        self.assertGreaterEqual(acks[0]['latency_ms'], 0)

    def test_new_hero_musketeer_carrier_confirms_source_card(self):
        s = state()
        first = play(slot=1, card=26000014, grid=(8, 20))
        first = replace(first, metadata={
            **first.metadata,
            'policy_effective_form_code': 2,
        })
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = time.perf_counter() - 0.15
        pending.input_completed_at = time.perf_counter() - 0.10
        pending.prior_elixir = s.elixir
        pending.prior_entities = frozenset(e['id'] for e in s.entities)
        self.executor.ack_watch.append(pending)

        new_id = 9014
        s.entities.append({
            'id': new_id,
            'owner': s.local_owner,
            'card_id': 203000014,
            'x': 8500,
            'y': 20500,
            'hp': 1000,
            'max_hp': 1000,
        })
        player = next(p for p in s.raw['players'] if p['owner'] == s.local_owner)
        player['ability_runtime'] = [{
            'known': True,
            'ability_name': 'Musketeer_hero_Ability',
            'members': [new_id],
        }]
        s.tick += 1
        s.received_at = pending.input_completed_at + 0.01

        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.ack_watch)
        acks = [data for event, data in self.events if event == 'hand_ack']
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]['evidence'], 'new_source_entity')
        self.assertEqual(acks[0]['spawn_entity_id'], new_id)

    def test_near_target_new_own_entity_confirms_when_source_card_id_differs(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(8, 20))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = time.perf_counter() - 0.15
        pending.input_completed_at = time.perf_counter() - 0.10
        pending.prior_elixir = s.elixir
        pending.prior_entities = frozenset(e['id'] for e in s.entities)
        self.executor.ack_watch.append(pending)

        new_id = 9020
        target_x, target_y = action_world(first)
        s.entities.append({
            'id': new_id,
            'owner': s.local_owner,
            'card_id': 99999999,
            'x': target_x + 700,
            'y': target_y - 400,
            'hp': 100,
            'max_hp': 100,
        })
        s.tick += 1
        s.received_at = pending.input_completed_at + 0.01

        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.ack_watch)
        acks = [data for event, data in self.events if event == 'hand_ack']
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]['evidence'], 'new_own_entity_near_target')
        self.assertEqual(acks[0]['spawn_entity_id'], new_id)

    def test_near_target_entity_fallback_disabled_with_multiple_ack_watches(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(8, 20))
        second = play(slot=1, card=s.hand_cards[1], grid=(8, 20))
        now = time.perf_counter()
        pendings = []
        for seq, action in ((1, first), (2, second)):
            pending = type('P', (), {})()
            pending.action = action
            pending.command_seq = seq
            pending.state = 'sent'
            pending.sent_at = now - 0.15
            pending.input_completed_at = now - 0.10
            pending.sent_tick = s.tick
            pending.prior_elixir = s.elixir
            pending.cost = 1.0
            pending.decision_at = now - 0.2
            pending.threat_ids = frozenset()
            pending.threat_target = None
            pending.prior_evolution_progress = None
            pending.prior_entities = frozenset(e['id'] for e in s.entities)
            pendings.append(pending)
        self.executor.ack_watch.extend(pendings)

        new_id = 9021
        target_x, target_y = action_world(first)
        s.entities.append({
            'id': new_id,
            'owner': s.local_owner,
            'card_id': 99999999,
            'x': target_x + 200,
            'y': target_y + 100,
            'hp': 100,
            'max_hp': 100,
        })
        s.tick += 1
        s.received_at = now + 0.01

        self.executor.poll(s, lambda *_: True)

        self.assertEqual(len(self.executor.ack_watch), 2)
        self.assertNotIn('hand_ack', [event for event, _ in self.events])

    def test_spell_never_uses_near_target_entity_fallback(self):
        s = state()
        # Put an actual spell in the selected native hand slot.  submit()
        # intentionally rejects card/slot mismatches before any ACK logic.
        s.hand_cards[0] = 28000000
        first = play(slot=0, card=28000000, grid=(8, 20))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = time.perf_counter() - 0.15
        pending.input_completed_at = time.perf_counter() - 0.10
        pending.prior_elixir = s.elixir
        pending.prior_entities = frozenset(e['id'] for e in s.entities)
        self.executor.ack_watch.append(pending)

        target_x, target_y = action_world(first)
        s.entities.append({
            'id': 9022,
            'owner': s.local_owner,
            'card_id': 99999999,
            'x': target_x,
            'y': target_y,
            'hp': 100,
            'max_hp': 100,
        })
        s.tick += 1
        s.received_at = pending.input_completed_at + 0.01

        self.executor.poll(s, lambda *_: True)

        self.assertEqual(self.executor.ack_watch, [pending])
        self.assertNotIn('hand_ack', [event for event, _ in self.events])

    def test_ack_watch_requires_post_input_probe_frame(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_tick = s.tick - 1
        pending.sent_at = time.perf_counter() - 0.2
        pending.input_completed_at = time.perf_counter() - 0.1
        pending.prior_elixir = s.elixir
        self.executor.ack_watch.append(pending)

        # This snapshot appears to show the card consumed, but it predates
        # input completion and therefore must not be used as ACK evidence.
        s.tick += 1
        s.hand_cards[0] = 0
        s.received_at = pending.input_completed_at - 0.01
        self.executor.poll(s, lambda *_: True)
        self.assertEqual(self.executor.ack_watch, [pending])
        self.assertNotIn('hand_ack', [event for event, _ in self.events])

        # A fresh frame captured after completion may confirm it.
        s.received_at = pending.input_completed_at + 0.01
        self.executor.poll(s, lambda *_: True)
        self.assertFalse(self.executor.ack_watch)
        acks = [data for event, data in self.events if event == 'hand_ack']
        self.assertEqual(len(acks), 1)
        self.assertGreaterEqual(acks[0]['latency_ms'], 0)
        self.assertGreaterEqual(acks[0]['input_to_ack_ms'], 0)

    def test_completed_input_ack_watch_allows_different_slot(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter()
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        self.executor.pending.remove(pending)
        self.executor.ack_watch.append(pending)
        self.executor.unconfirmed_spend.append(
            (time.perf_counter(), pending.cost, pending.command_seq))

        second = play(slot=1, card=s.hand_cards[1])
        self.executor.submit(SimpleNamespace(actions=(second,)), s)

        self.assertEqual(len(self.executor.ack_watch), 1)
        self.assertEqual(len(self.executor.pending), 1)
        self.assertEqual(self.executor.pending[0].action.hand_slot, 1)
        self.assertIn(0, self.executor.blocked_slots(s))

    def test_ack_watch_keeps_same_slot_blocked(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        pending.state = 'sent'
        pending.sent_at = time.perf_counter()
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        self.executor.ack_watch.append(pending)

        self.executor.submit(SimpleNamespace(actions=(first,)), s)

        self.assertFalse(self.executor.pending)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'action_rejected']
        self.assertIn('slot_changed_or_locked', reasons)

    def test_elixir_ack_guards_same_slot_until_authoritative_hand_rotation(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        now = time.perf_counter()
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = now - 0.10
        pending.input_completed_at = now - 0.05
        pending.prior_elixir = s.elixir
        self.executor.ack_watch.append(pending)
        self.executor.unconfirmed_spend.append(
            (now, pending.cost, pending.command_seq))

        s.tick += 1
        s.received_at = now + 0.01
        s.elixir = pending.prior_elixir - pending.cost
        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.ack_watch)
        self.assertIn(0, self.executor.blocked_slots(s))
        self.assertEqual(self.executor.reserved_elixir, 0)
        guards = self.executor.slot_consume_guards
        self.assertEqual(guards[0]['command_seq'], pending.command_seq)
        acks = [data for event, data in self.events if event == 'hand_ack']
        self.assertEqual(acks[-1]['evidence'], 'elixir_cost_drop')

        # A repeated decision from the still-stale hand must not requeue the
        # same slot/card even though the weak ACK was accepted.
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        self.assertFalse(self.executor.pending)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'action_rejected']
        self.assertIn('slot_changed_or_locked', reasons)

        # Once the native hand actually rotates, the guard disappears.
        s.hand_cards[0] = s.hand_cards[1]
        s.tick += 1
        self.assertNotIn(0, self.executor.blocked_slots(s))
        self.assertFalse(self.executor.slot_consume_guards)

    def test_spawn_ack_keeps_virtual_spend_until_hand_rotation(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 10))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending.pop(0)
        now = time.perf_counter()
        pending.state = 'sent'
        pending.sent_tick = s.tick
        pending.sent_at = now - 0.10
        pending.input_completed_at = now - 0.05
        pending.prior_elixir = s.elixir
        pending.prior_entities = frozenset(e['id'] for e in s.entities)
        self.executor.ack_watch.append(pending)
        self.executor.unconfirmed_spend.append(
            (now, pending.cost, pending.command_seq))

        target_x, target_y = action_world(first)
        s.entities.append({
            'id': 9901,
            'owner': s.local_owner,
            'card_id': first.card_id,
            'x': target_x,
            'y': target_y,
            'hp': 100,
            'max_hp': 100,
        })
        s.tick += 1
        s.received_at = now + 0.01
        self.executor.poll(s, lambda *_: True)

        self.assertFalse(self.executor.ack_watch)
        self.assertIn(0, self.executor.blocked_slots(s))
        self.assertEqual(self.executor.reserved_elixir, pending.cost)

        s.hand_cards[0] = s.hand_cards[1]
        s.tick += 1
        self.executor.blocked_slots(s)
        self.assertEqual(self.executor.reserved_elixir, 0)

    def test_prior_weak_ack_disables_new_elixir_only_attribution(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        p1 = self.executor.pending.pop(0)
        now = time.perf_counter()
        p1.state = 'sent'
        p1.sent_tick = s.tick
        p1.sent_at = now - 0.10
        p1.input_completed_at = now - 0.05
        p1.prior_elixir = s.elixir
        self.executor.ack_watch.append(p1)

        s.tick += 1
        s.received_at = now + 0.01
        s.elixir = p1.prior_elixir - p1.cost
        self.executor.poll(s, lambda *_: True)
        self.assertIn(0, self.executor.slot_consume_guards)

        second = play(slot=1, card=s.hand_cards[1])
        self.executor.submit(SimpleNamespace(actions=(second,)), s)
        p2 = self.executor.pending.pop(0)
        p2.state = 'sent'
        p2.sent_tick = s.tick
        p2.sent_at = now
        p2.input_completed_at = now + 0.01
        p2.prior_elixir = s.elixir
        self.executor.ack_watch.append(p2)

        # The aggregate resource drop is large enough for p2, but p1 still
        # lacks authoritative hand rotation, so it cannot be uniquely
        # attributed to this newer command.
        s.tick += 1
        s.received_at = now + 0.02
        s.elixir = p2.prior_elixir - p2.cost
        self.executor.poll(s, lambda *_: True)

        self.assertEqual(self.executor.ack_watch, [p2])
        acks = [data for event, data in self.events
                if event == 'hand_ack' and data['command_seq'] == p2.command_seq]
        self.assertFalse(acks)

    def test_multiple_ack_watches_do_not_use_elixir_fallback(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0])
        second = play(slot=1, card=s.hand_cards[1])
        now = time.perf_counter()
        for seq, action in ((1, first), (2, second)):
            pending = type('P', (), {})()
            pending.action = action
            pending.command_seq = seq
            pending.state = 'sent'
            pending.sent_at = now
            pending.sent_tick = s.tick
            pending.prior_elixir = 10.0
            pending.cost = 1.0
            pending.decision_at = now
            pending.threat_ids = frozenset()
            pending.threat_target = None
            pending.prior_evolution_progress = None
            self.executor.ack_watch.append(pending)

        s.tick += 1
        s.elixir = 8.0
        self.executor.poll(s, lambda *_: True)

        self.assertEqual(len(self.executor.ack_watch), 2)
        self.assertNotIn('hand_ack', [event for event, _ in self.events])

    def test_third_action_is_rejected_at_in_flight_capacity(self):
        s=state()
        self.executor.submit(SimpleNamespace(actions=(play(), play(slot=1,card=26000014))),s)
        self.executor.submit(SimpleNamespace(actions=(play(slot=2,card=26000030),)),s)
        self.assertEqual(len(self.executor.pending),1)
        self.assertIn('in_flight_capacity', [d.get('reason') for e,d in self.events])

    def test_expired_action_is_never_dispatched(self):
        s=state()
        s.received_at=time.perf_counter()-5
        self.executor.submit(SimpleNamespace(actions=(play(),)),s)
        self.executor.poll(s,lambda *_:True)
        self.assertFalse(self.executor.pending)
        self.actuator.deploy_action.assert_not_called()

    def test_validation_cannot_dispatch_after_deadline(self):
        s = state()
        s.received_at = 10.0
        clock = [10.1]
        with patch('agent.execution.time.perf_counter', side_effect=lambda: clock[0]):
            self.executor.submit(SimpleNamespace(actions=(play(),)), s)
            def slow_validate(*_):
                clock[0] = 11.0
                return True
            self.executor.poll(s, slow_validate)
        self.assertFalse(self.executor.pending)
        self.actuator.deploy_action.assert_not_called()
        self.assertIn('action_expired', [e for e, _ in self.events])

    def test_hand_ack_confirms_action_without_waiting_for_spawn(self):
        s=state()
        self.executor.submit(SimpleNamespace(actions=(play(),)),s)
        p=self.executor.pending[0]
        p.state='sent'
        p.sent_at=time.perf_counter()
        s.tick+=1
        s.hand_cards[0]=0
        self.executor.poll(s,lambda *_:True)
        self.assertEqual(self.executor.reserved_elixir,0)
        self.assertIn('hand_ack',[e for e,_ in self.events])
        self.assertEqual(self.executor.confirmed_actions, 1)

    def test_none_snapshot_never_dispatches(self):
        s=state()
        self.executor.submit(SimpleNamespace(actions=(play(),)),s)
        self.executor.poll(None,lambda *_:True)
        self.actuator.deploy_action.assert_not_called()


    def _ack_first_defender(self, s, *, enemy_id=9001, enemy_x=3500, enemy_y=11500):
        add_enemy(s, enemy_id, enemy_x, enemy_y)
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 11))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter()
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        s.tick += 1
        s.hand_cards[0] = 0
        self.executor.poll(s, lambda *_: True)

    def test_threat_reservation_suppresses_second_defender_on_same_enemy(self):
        s = state()
        self._ack_first_defender(s)
        self.assertEqual(len(self.executor.threat_reservations), 1)
        second = play(slot=1, card=s.hand_cards[1], grid=(3, 11))
        self.executor.submit(SimpleNamespace(actions=(second,)), s)
        self.assertFalse(self.executor.pending)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'action_suppressed']
        self.assertIn('threat_already_committed', reasons)
        starts = [data for event, data in self.events
                  if event == 'threat_reservation_started'
                  and data.get('confidence') == 'confirmed']
        self.assertEqual(starts[-1]['ttl_ms'], 850)

    def test_new_enemy_is_not_blocked_by_existing_threat_reservation(self):
        s = state()
        self._ack_first_defender(s)
        add_enemy(s, 9002, 14500, 11500)
        second = play(slot=1, card=s.hand_cards[1], grid=(14, 11))
        self.executor.submit(SimpleNamespace(actions=(second,)), s)
        self.assertEqual(len(self.executor.pending), 1)
        self.assertEqual(self.executor.pending[0].threat_ids, frozenset((9002,)))

    def test_confirmed_defence_uses_local_gate_not_global_settle(self):
        s = state()
        self._ack_first_defender(s)
        self.assertEqual(self.executor._post_action_settle_tick, -1)
        self.assertFalse(self.executor.decision_blocked(s))

    def test_unclassified_background_hand_ack_does_not_reintroduce_global_settle(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 10))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter()
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        s.tick += 1
        s.hand_cards[0] = 0
        self.executor.poll(s, lambda *_: True)
        self.assertEqual(self.executor._post_action_settle_tick, -1)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'post_action_settle_skipped']
        self.assertIn('background_hand_ack', reasons)

    def test_ack_timeout_with_known_threat_uses_uncertain_local_reservation(self):
        s = state()
        add_enemy(s, 9001, 3500, 11500)
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 11))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter() - config.ACK_TIMEOUT_SECONDS - 0.1
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        s.tick += 1
        self.executor.poll(s, lambda *_: True)
        self.assertFalse(self.executor.pending)
        self.assertEqual(self.executor._post_action_settle_tick, -1)
        self.assertEqual(len(self.executor.threat_reservations), 1)
        reservation = self.executor.threat_reservations[0]
        self.assertEqual(reservation.threat_ids, frozenset((9001,)))
        started = [data for event, data in self.events
                   if event == 'threat_reservation_started']
        self.assertEqual(started[-1]['confidence'], 'uncertain')
        self.assertIn('action_missed', [event for event, _ in self.events])

    def test_ack_timeout_without_known_threat_uses_uncertain_prediction_gate(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 10))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter() - config.ACK_TIMEOUT_SECONDS - 0.1
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        self.executor.predictions.append({
            'action': first,
            'command_seq': pending.command_seq,
            'expires_at': time.perf_counter() + 2.0,
        })
        s.tick += 1
        self.executor.poll(s, lambda *_: True)
        self.assertFalse(self.executor.threat_reservations)
        self.assertEqual(self.executor._post_action_settle_tick, -1)
        skipped = [data.get('reason') for event, data in self.events
                   if event == 'post_action_settle_skipped']
        self.assertIn('uncertain_prediction_ack_timeout', skipped)

    def test_ack_timeout_without_prediction_keeps_global_settle_fallback(self):
        s = state()
        first = play(slot=0, card=s.hand_cards[0], grid=(3, 10))
        self.executor.submit(SimpleNamespace(actions=(first,)), s)
        pending = self.executor.pending[0]
        pending.state = 'sent'
        pending.sent_at = time.perf_counter() - config.ACK_TIMEOUT_SECONDS - 0.1
        pending.sent_tick = s.tick
        pending.prior_elixir = s.elixir
        s.tick += 1
        with patch.object(config, 'ENABLE_MODEL_PREDICTION_OVERLAY', False):
            self.executor.poll(s, lambda *_: True)
        self.assertFalse(self.executor.threat_reservations)
        self.assertGreaterEqual(self.executor._post_action_settle_tick, s.tick)
        reasons = [data.get('reason') for event, data in self.events
                   if event == 'post_action_settle_armed']
        self.assertIn('ack_timeout', reasons)

    def test_pause_drops_plans_but_reconciles_sent_card_on_resume(self):
        s = state()
        self.executor.submit(SimpleNamespace(actions=(play(), play(slot=1, card=26000014))), s)
        self.executor.pending[0].state = 'sent'
        self.executor.pending[0].sent_at = time.perf_counter()
        self.executor.pause()
        self.assertEqual(self.executor.blocked_slots(s), {0})
        s.tick += 40
        s.hand_cards[0] = 0
        self.executor.poll(s, lambda *_: True)
        self.assertEqual(self.executor.reserved_elixir, 0)
        self.actuator.deploy_action.assert_not_called()


if __name__ == '__main__':
    unittest.main()
