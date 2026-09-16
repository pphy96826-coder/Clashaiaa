import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from agent.feature_adapter import FeatureAdapter
from native_runner.contracts import (ATTACK_STATE_FIELDS, AttackPhase, AttackStateV1,
    EntityStateV1, SemanticEvidenceLevel, SemanticProvenanceV1)


def known_attack(**kwargs):
    return AttackStateV1(
        provenance=SemanticProvenanceV1(
            {field: SemanticEvidenceLevel.NATIVE_DERIVED for field in ATTACK_STATE_FIELDS}),
        **kwargs)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FeatureAdapter.__new__(FeatureAdapter)
        self.adapter._acceleration = {7: (2.0, -1.0)}
        self.adapter.bundle = __import__('native_runner.training.v4.factory', fromlist=['production_semantic_bundle']).production_semantic_bundle()
        self.moving = EntityStateV1(
            entity_id=7, owner=0, card_id=26000010, entity_kind='character',
            position=(1000.0, 2000.0), velocity=(10.0, 20.0), age_ms=100)
        self.attack_provenance = SemanticProvenanceV1({
            **dict(self.moving.runtime_provenance.field_evidence),
            'attack_state': SemanticEvidenceLevel.NATIVE_DERIVED})
        self.building = EntityStateV1(
            entity_id=8, owner=0, card_id=27000000, entity_kind='building',
            position=(3000.0, 4000.0), velocity=(10.0, 20.0), age_ms=100)

    def test_forecast_does_not_extend_noisy_acceleration(self):
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((self.moving,), 10)[0]
        self.assertEqual(result.position, (1100.0, 2200.0))
        self.assertEqual(result.velocity, (10.0, 20.0))
        self.assertEqual(result.age_ms, 600)

    def test_buildings_are_not_moved_by_forecast(self):
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((self.building,), 10)[0]
        self.assertEqual(result.position, self.building.position)
        self.assertEqual(result.velocity, self.building.velocity)

    def test_disabled_overlay_returns_real_entities(self):
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', False):
            result = self.adapter._forecast_entities((self.moving,), 10)
        self.assertIs(result[0], self.moving)

    def test_no_pending_card_forecasts_end_to_end_plus_500ms(self):
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True), \
             patch('config.PREDICTION_LATENCY_COMPENSATION_MS', 100):
            result = self.adapter._model_entities((self.moving,), (), 10)[0]
        # The kinematic fallback caps the total extrapolation at 500ms.
        self.assertEqual(result.position, (1100.0, 2200.0))

    def test_predicted_card_stays_in_deployment_until_deploy_time(self):
        virtual = ({'command_seq': 1, 'owner': 0, 'card_id': 26000014,
                     'x': 4000, 'y': 12000, 'latency_ms': 100},)
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._model_entities((), virtual, 10)[0]
        self.assertEqual(result.position, (4000.0, 12000.0))
        self.assertEqual(result.deployment_runtime['phase'], 'deploying')
        self.assertEqual(result.deployment_runtime['remaining_wall_ms'], 500)

    def test_attack_range_stops_ranged_unit_inside_range(self):
        target = EntityStateV1(
            entity_id=99, owner=1, card_id=26000010, entity_kind='character',
            position=(6500.0, 2000.0), velocity=(0.0, 0.0), age_ms=100)
        attacker = EntityStateV1(
            entity_id=7, owner=0, card_id=26000014, entity_kind='character',
            position=(1000.0, 2000.0), velocity=(60.0, 0.0), age_ms=100,
            attack_state=known_attack(
                phase=AttackPhase.IDLE, target_entity=99, locked=False),
            runtime_provenance=SemanticProvenanceV1({
                **{field: SemanticEvidenceLevel.UNKNOWN for field in {
                    'shield_state', 'effect_states', 'projectile_state', 'visibility_state',
                    'ability_states', 'evolution_state', 'movement_runtime', 'deployment_runtime',
                    'resource_states', 'capture_runtime', 'threshold_relocation_runtime',
                    'periodic_attack_modifier'}},
                'attack_state': SemanticEvidenceLevel.NATIVE_DERIVED}))
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((attacker, target), 10)[0]
        # 5.5 tiles plus collision allowance is enough for the 6-tile
        # Musketeer range, so the model must see her holding position.
        self.assertEqual(result.position, attacker.position)
        self.assertEqual(result.velocity, (0.0, 0.0))

    def test_target_outside_range_keeps_moving(self):
        target = EntityStateV1(
            entity_id=99, owner=1, card_id=26000010, entity_kind='character',
            position=(12000.0, 2000.0), velocity=(0.0, 0.0), age_ms=100)
        attacker = EntityStateV1(
            entity_id=7, owner=0, card_id=26000014, entity_kind='character',
            position=(1000.0, 2000.0), velocity=(60.0, 0.0), age_ms=100,
            attack_state=known_attack(
                phase=AttackPhase.ACQUIRING, target_entity=99, locked=False),
            runtime_provenance=SemanticProvenanceV1({
                **{field: SemanticEvidenceLevel.UNKNOWN for field in {
                    'shield_state', 'effect_states', 'projectile_state', 'visibility_state',
                    'ability_states', 'evolution_state', 'movement_runtime', 'deployment_runtime',
                    'resource_states', 'capture_runtime', 'threshold_relocation_runtime',
                    'periodic_attack_modifier'}},
                'attack_state': SemanticEvidenceLevel.NATIVE_DERIVED}))
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((attacker, target), 10)[0]
        self.assertEqual(result.position, (1600.0, 2000.0))

    def test_crossing_tower_attack_range_stops_before_overshoot(self):
        self.adapter._towers = {'target': SimpleNamespace(
            entity_id=99, owner=1, position=(8200.0, 2000.0), hitpoints=1000)}
        attacker = replace(self.moving, card_id=26000014,
            runtime_provenance=self.attack_provenance,
            velocity=(100.0, 0.0), attack_state=known_attack(
                phase=AttackPhase.ACQUIRING, target_entity=99, locked=False))
        radius = self.adapter.bundle.card_specs[26000014].range_tiles * 1000 + 750
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((attacker,), 30)[0]
        self.assertAlmostEqual(result.position[0], 8200 - radius)
        self.assertEqual(result.velocity, (0.0, 0.0))
        self.assertEqual(attacker.position, (1000.0, 2000.0))

    def test_receding_target_does_not_cause_false_stop(self):
        target = replace(self.moving, entity_id=99, owner=1,
            position=(8200.0, 2000.0), velocity=(200.0, 0.0))
        attacker = replace(self.moving, card_id=26000014,
            runtime_provenance=self.attack_provenance,
            velocity=(100.0, 0.0), attack_state=known_attack(
                phase=AttackPhase.ACQUIRING, target_entity=99, locked=False))
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((attacker, target), 10)[0]
        self.assertEqual(result.position, (2000.0, 2000.0))
        self.assertEqual(result.velocity, (100.0, 0.0))

    def test_attack_phase_freezes_even_without_resolved_target(self):
        attacker = EntityStateV1(
            entity_id=7, owner=0, card_id=26000014, entity_kind='character',
            position=(1000.0, 2000.0), velocity=(60.0, 0.0), age_ms=100,
            attack_state=known_attack(phase=AttackPhase.WINDUP),
            runtime_provenance=SemanticProvenanceV1({
                **{field: SemanticEvidenceLevel.UNKNOWN for field in {
                    'shield_state', 'effect_states', 'projectile_state', 'visibility_state',
                    'ability_states', 'evolution_state', 'movement_runtime', 'deployment_runtime',
                    'resource_states', 'capture_runtime', 'threshold_relocation_runtime',
                    'periodic_attack_modifier'}},
                'attack_state': SemanticEvidenceLevel.NATIVE_DERIVED}))
        with patch('config.ENABLE_MODEL_PREDICTION_OVERLAY', True):
            result = self.adapter._forecast_entities((attacker,), 10)[0]
        self.assertEqual(result.position, attacker.position)
        self.assertEqual(result.velocity, (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
