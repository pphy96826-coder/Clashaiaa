"""Translate measured Null's telemetry into the original V4 contracts.

Owner identity is never renumbered. Missing telemetry remains unknown; no
invented hands, card cycles, parent groups or runtime character identities.
"""
import sys
import math
from collections import deque
from dataclasses import replace
from pathlib import Path

import config
if str(config.FIRSTLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(config.FIRSTLIGHT_DIR))

from native_runner.contracts import (ObservationV1, ObservationTier, PlayerStateV1,
    TowerStateV1, EntityStateV1, ActionMaskV1, TimeStateV1, TerminalV1, EventV1,
    TOWER_RUNTIME_SEMANTIC_FIELDS, ENTITY_RUNTIME_SEMANTIC_FIELDS, SemanticEvidenceLevel,
    SemanticProvenanceV1, AttackPhase)
from native_runner.arena import card_placement_mask, OccupiedFootprintV1, native_building_footprint
from native_runner.training.v4.factory import production_semantic_bundle, _public_tracker_v4
from native_runner.training.v4.tensorizer import UniversalObservationTensorizerV4, REL_TARGETS, REL_SOURCE_OF
from bridge.coordinates import probe_to_world, world_to_view, action_world, ScreenCalibration
from bridge.runtime_state import attack_state, attack_events, runtime_provenance, movement_state, deployment_state
from bridge.projectile_state import projectile_state, runtime_u32, live_entities
from bridge.card_state import card_selections, observed_deck_roles, supported_deck_roles
from bridge.ability_state import ability_states
from bridge.hero_execution import MUSKETEER, ABILITY, ready_abilities, covered_by_ability_hud
from bridge.evolution_state import SKELETONS, BINDINGS, CardEvolutionTracker, evolved_entity_state
from bridge.causal_state import DeploymentGroups
from bridge.effect_state import ActiveEffects
from bridge.effect_origin import EffectOrigins
from bridge.spawn_state import SpawnRelations
from bridge.impact_state import ImpactEvents
from bridge.damage_state import DamageEvents
from bridge.heal_state import HealEvents
from bridge.area_state import AreaOrigins
from bridge.projectile_origin import ProjectileOrigins
from bridge.reference_events import ReferenceEvents

HOG_26_DECK = (26000010, 26000014, 26000021, 26000030, 26000038, 27000000, 28000000, 28000011)

# Conservative impact envelopes in native world units.  They are deliberately
# wider than the visual effect so a delayed spell cannot wake an inactive king
# tower because the target moved during input latency.
SPELL_IMPACT_RADIUS = {28000000: 2500.0, 28000001: 4000.0, 28000011: 1500.0}
KING_TOWER_RADIUS = 1800.0
KING_TOWER_SAFETY_MARGIN = 1000.0


class TelemetryError(ValueError):
    pass


class FeatureAdapter:
    def __init__(self, actor_owner=0, initial_deck=None, oracle_elixir=False, own_tower=None, hero_musketeer=False,
                 skeleton_evolution=False, evolution_enabled=None, observation_profile='reference',
                 experimental_origins=False, allow_partial_tower_attach=False):
        if observation_profile not in ('reference', 'extended'):
            raise ValueError('observation_profile must be reference or extended')
        if experimental_origins and observation_profile != 'extended':
            raise ValueError('experimental_origins requires the extended observation profile')
        self.observation_profile = observation_profile
        self.experimental_origins = bool(experimental_origins)
        # Mid-battle attach may legitimately miss towers destroyed before the
        # observer connected.  This is an advisory current-state baseline; it
        # never invents IDs or turns an incomplete probe frame into a valid
        # frame.  Normal new-battle startup remains six-tower strict.
        self.allow_partial_tower_attach = bool(allow_partial_tower_attach)
        # Keep the old keyword for existing capture tools; both supported cards
        # now use the same switch. Their activation evidence stays independent.
        self.skeleton_evolution = skeleton_evolution if evolution_enabled is None else evolution_enabled
        # None is production auto mode; False remains useful for base-only
        # audits, and True is the compatible explicit hero test switch.
        self.hero_mode = hero_musketeer
        self.hero_musketeer = bool(hero_musketeer)
        self.hero_skill_ready = True  # production verifies calibration before run
        self.actor_owner = actor_owner
        self.configured_deck = tuple(initial_deck) if initial_deck else None
        self.bundle = production_semantic_bundle()
        self.tensorizer = None
        self.current_deck = ()
        self.oracle_elixir = oracle_elixir
        self.own_tower = own_tower if own_tower is not None else config.LOCAL_TOWER_TROOP_ID
        self.episode_id = ''
        self.calibration = ScreenCalibration.load(config.CALIBRATION_PATH)
        self._clear()

    def _clear(self):
        self._towers = {}
        self._tower_loadouts = {}
        self._previous_entities = {}
        self._previous_velocity = {}
        self._velocity = {}
        self._acceleration = {}
        self._first_seen = {}
        self._previous_hands = {}
        self._events = deque(maxlen=256)
        self._revealed = {0: set(), 1: set()}
        self._observed_tick = -1
        self._mask_cache = {}
        self.destroyed_enemy_princess_lanes = set()
        self.quality = {'observation_profile': self.observation_profile,
                        'experimental_origins': self.experimental_origins}
        self._reference_events = ReferenceEvents()
        self._evolutions = {cid: CardEvolutionTracker(cid) for cid in BINDINGS}
        self._deployment_groups = DeploymentGroups()
        self._spawn_relations = SpawnRelations()
        self._area_origins = AreaOrigins()
        self._projectile_origins = ProjectileOrigins()
        self._impact_events = ImpactEvents()
        self._damage_events = DamageEvents()
        self._effect_origins = EffectOrigins()
        self._effect_reader = ActiveEffects(self.bundle, origins=self._effect_origins,
            historical_sources=self.observation_profile == 'extended')
        self._heal_events = HealEvents(origins=self._effect_origins)
        self._invalid_causal_entities = set()
        self._effects_by_entity = {}

    def _effect_provenance(self, semantic, entity_id):
        if not self._effects_by_entity.get(entity_id, ((), False))[1]:
            return semantic
        return replace(semantic,
            field_evidence={**dict(semantic.field_evidence), 'effect_states': SemanticEvidenceLevel.NATIVE_DERIVED},
            source_fields={**dict(semantic.source_fields), 'effect_states': ('nulls-live.v3.entities.active_effect_runtime',)})

    def _player(self, state, owner=None):
        return next(p for p in state.raw['players'] if p['owner'] == (self.actor_owner if owner is None else owner))

    def _deck(self, state):
        deck = tuple(state.deck_cards)
        if len(deck) != 8 or len(set(deck)) != 8 or any(c not in self.bundle.card_specs for c in deck):
            raise TelemetryError('probe must supply eight unique supported deck cards')
        if self.configured_deck and set(deck) != set(self.configured_deck):
            raise TelemetryError('live deck does not match the selected specialist checkpoint')
        return deck

    def reset_match(self, state, episode_id):
        self._clear()
        self.actor_owner = state.local_owner
        self.episode_id = episode_id
        self.current_deck = self._deck(state)
        decks = tuple(tuple(self._player(state, owner).get('deck', [])) for owner in (0, 1))
        try:
            role_evidence = tuple(observed_deck_roles(self._player(state, owner)) for owner in (0, 1))
        except ValueError as exc:
            raise TelemetryError(str(exc)) from exc
        # The reference role feature is boolean. Preserve positive native
        # evidence; retain unknown-vs-false distinction separately in quality.
        validated_roles = tuple(supported_deck_roles(role_evidence[o], deck, self.bundle)
                                for o, deck in enumerate(decks))
        roles = tuple(item[0] for item in validated_roles)
        self.quality['initial_role_evidence'] = role_evidence[self.actor_owner]
        self.quality['initial_role_issues'] = validated_roles[self.actor_owner][1]
        # The native tracker requires an exact ordered cycle, not a set
        # difference. The legacy probe has no such field.
        cycle = self._player(state).get('cycle')
        complete_decks = all(len(d) == 8 and len(set(d)) == 8 and set(d) <= self.bundle.card_specs.keys() for d in decks)
        tracker = _public_tracker_v4(self.bundle, decks, roles) if cycle is not None and complete_decks else None
        self.tensorizer = UniversalObservationTensorizerV4(
            actor_owner=self.actor_owner, card_catalog=self.bundle.card_catalog,
            ability_catalog=self.bundle.ability_catalog, entity_archetype_catalog=self.bundle.entity_archetype_catalog,
            effect_catalog=self.bundle.effect_catalog, card_specs=self.bundle.card_specs,
            deck=self.current_deck, tracker=tracker, horizontal_mirror=True,
            deck_roles=roles[self.actor_owner],
            reject_unknown_public_semantics=False)
        observation = self.build_observation(state)
        initial = {owner: float(self._player(state, owner)['elixir']) for owner in (0, 1)}
        self.tensorizer.start_episode(observation, initial_elixir=initial)
        if tracker and not self.oracle_elixir and state.tick > 90:
            tracker.elixir_bounds[1 - self.actor_owner] = (0.0, 10.0)
        self.quality['public_tracker'] = tracker is not None

    def record_ability_execution(self, action, state):
        # ACK tick is the first observed confirmation, not a fabricated cast
        # start timestamp. Keep this exact spend private to the acting player.
        self._events.append(EventV1(tick=state.tick, event_type='runtime_ability_activation',
            owner=action.owner, entity_id=action.source_entity, card_id=MUSKETEER,
            data={'kind': 'activate_ability', 'ability_id': action.ability_id,
                  'private_to': action.owner, 'fair_ability_activation_exact': True,
                  'ability_execution_evidence': ['same_controller_carrier_charge_1_to_0']}))

    def observe(self, state):
        if state.local_owner != self.actor_owner or tuple(state.deck_cards) != self.current_deck:
            raise TelemetryError('owner/deck changed during the episode')
        if state.tick <= self._observed_tick:
            return
        for player in state.raw['players']:
            owner = int(player['owner'])
            hand = {int(h['slot']): int(h['card_id']) for h in player.get('hand', [])}
            previous = self._previous_hands.get(owner, {})
            # A positive card leaving its native slot is execution evidence.
            # Empty/refilling slots stay empty in the policy observation.
            for slot, old in previous.items():
                if old > 0 and old != hand.get(slot, 0) and old in self.bundle.card_specs:
                    self._revealed[owner].add(old)
                    if old == 28000006:
                        # A hand transition cannot identify Mirror's copied
                        # card/form/cost. Do not submit an invalid exact event
                        # to FirstLight's public tracker.
                        self.quality['mirror_execution_unknown'] = True
                        if self.tensorizer is not None and self.tensorizer.tracker is not None:
                            self.tensorizer.tracker.elixir_bounds[owner] = (0.0, 10.0)
                        continue
                    self._events.append(EventV1(tick=state.tick, event_type='action_executed', owner=owner,
                        card_id=old, data={'kind': 'play_card', 'hand_slot': slot,
                        'evidence': 'native_hand_transition', 'cost': self.bundle.card_specs[old].elixir_cost}))
            self._previous_hands[owner] = hand
        current = {}
        previous_velocity = self._previous_velocity
        self._velocity = {}
        self._acceleration = {}
        for ent in state.entities:
            eid = int(ent['id'])
            x, y = probe_to_world(ent['x'], ent['y'])
            current[eid] = (state.tick, x, y)
            self._first_seen.setdefault(eid, state.tick)
            prev = self._previous_entities.get(eid)
            if prev and 0 < state.tick - prev[0] <= 10:
                # BattleEnvV1._derive_events uses world units per native tick,
                # not per second. The frozen tensorizer expects that scale.
                dt = state.tick - prev[0]
                velocity = ((x - prev[1]) / dt, (y - prev[2]) / dt)
                self._velocity[eid] = velocity
                old_velocity = previous_velocity.get(eid)
                if old_velocity is not None:
                    self._acceleration[eid] = (
                        max(-120.0, min(120.0, (velocity[0] - old_velocity[0]) / dt)),
                        max(-120.0, min(120.0, (velocity[1] - old_velocity[1]) / dt)))
            # A spawned entity is not proof of a card in the opponent's deck.
            # Reference BattleEnv reveals cards from play receipts, not births.
        self._previous_entities = current
        self._previous_velocity = dict(self._velocity)
        self._observed_tick = state.tick

    def _build_towers(self, state):
        seen = set()
        for ent in state.entities:
            troop = ent.get('tower_troop_id')
            if ent.get('owner') in (0, 1) and troop in (159000000,159000001,159000002,159000004):
                owner = ent['owner']
                if troop == 159000004 or self._tower_loadouts.get(owner) != 159000004:
                    self._tower_loadouts[owner] = troop
        for ent in state.entities:
            if ent.get('card_id') != -1 or ent.get('owner') not in (0, 1):
                continue
            owner = int(ent['owner'])
            x, y = probe_to_world(ent['x'], ent['y'])
            if x == 9000 and y in (3000, 29000):
                kind = 'king'
            elif x in (3500, 14500) and y in (6500, 25500):
                kind = 'princess_left' if x == 3500 else 'princess_right'
            else:
                continue
            key = (owner, kind)
            seen.add(key)
            hp, max_hp = float(ent['hp']), float(ent['max_hp'])
            if max_hp <= 0:
                raise TelemetryError('tower max HP is unavailable')
            # Live asset identity takes precedence. A legacy capture can use
            # an explicit local override; never assume the opponent's type.
            troop_id = ent.get('tower_troop_id')
            # Preserve BattleEnvV1._tower_troop_id's training convention:
            # Chef loadout is encoded on the king; side towers use Princess ID.
            if self._tower_loadouts.get(owner) == 159000004:
                troop_id = 159000004 if kind == 'king' else 159000000
            if kind != 'king' and troop_id is None and owner == self.actor_owner:
                troop_id = self.own_tower
                self.quality['own_tower_source'] = 'explicit_override' if troop_id is not None else 'unknown'
            elif kind != 'king':
                self.quality['own_tower_source' if owner == self.actor_owner else 'enemy_tower_source'] = 'native_asset'
            if troop_id is not None and troop_id not in (159000000,159000001,159000002,159000004):
                troop_id = None
            if kind != 'king' and troop_id is None:
                raise TelemetryError(f'cannot identify tower troop for owner {owner}; native v3 probe required')
            measured_attack = attack_state(ent, state.tick, self._live_entities)
            self._towers[key] = TowerStateV1(entity_id=int(ent['id']), owner=owner, tower_kind=kind,
                position=(x, y), hitpoints=max(0, min(hp, max_hp)), max_hitpoints=max_hp,
                shield=float(ent.get('shield', 0)), attack_state=measured_attack,
                effect_states=self._effects_by_entity.get(ent['id'], ((), False))[0] if hp > 0 else (),
                visible_target=measured_attack.target_entity if measured_attack else None,
                runtime_provenance=self._effect_provenance(runtime_provenance(TOWER_RUNTIME_SEMANTIC_FIELDS, measured_attack, state.tick), ent['id']),
                tower_troop_id=troop_id, active=bool(ent.get('active', hp > 0 if kind != 'king' else
                    hp < max_hp or any(t.owner == owner and t.tower_kind != 'king' and t.hitpoints <= 0 for t in self._towers.values()))))
        # Only a previously measured tower can be declared destroyed on
        # disappearance. An incomplete opening snapshot must never open pockets.
        if seen:
            for key in self._towers.keys() - seen:
                self._towers[key] = replace(self._towers[key], hitpoints=0, active=False,
                    shield=0, attack_state=None, visible_target=None, effect_states=(),
                    runtime_provenance=runtime_provenance(TOWER_RUNTIME_SEMANTIC_FIELDS, None, state.tick))
        if len(self._towers) != 6 and not (self.allow_partial_tower_attach and self._towers):
            raise TelemetryError('need all six initial tower identities; start listening before battle')
        if len(self._towers) != 6:
            self.quality['tower_binding_mode'] = 'advisory_partial_attach'
            self.quality['tower_binding_missing_count'] = 6 - len(self._towers)
        else:
            self.quality['tower_binding_mode'] = 'full_initial'
        for (owner, kind), tower in self._towers.items():
            if owner != self.actor_owner and tower.hitpoints <= 0 and kind.startswith('princess_'):
                self.destroyed_enemy_princess_lanes.add(kind.removeprefix('princess_'))
        self.quality['tower_types'] = {f'{o}:{k}': t.tower_troop_id for (o,k),t in self._towers.items() if k != 'king'}
        self.quality['tower_loadouts'] = dict(self._tower_loadouts)
        self.quality['tower_type_unknown_count'] = sum(t.tower_troop_id is None for t in self._towers.values() if t.tower_kind != 'king')
        return tuple(self._towers.values())

    def _build_entities(self, state):
        result = []
        exact = 0
        deployment_groups, deployment_issues = self._deployment_groups.project(state.raw, state.entities, self.bundle, state.tick)
        spawn_groups, spawn_issues = self._spawn_relations.project(state.raw, state.entities, self.bundle, state.tick)
        area_origins, area_issues = {}, []
        projectile_origins, origin_issues = {}, []
        if self.experimental_origins:
            area_origins, area_issues = self._area_origins.project(state.raw, state.entities, state.tick)
            projectile_origins, origin_issues = self._projectile_origins.project(state.raw, state.entities, state.tick)
        for eid in area_origins.keys() & (spawn_groups.keys() | projectile_origins.keys()):
            area_issues.append(f'{eid}:contradictory_area_creation_kind')
            self._invalid_causal_entities.add(eid)
        for eid in deployment_groups.keys() & spawn_groups.keys():
            # Contradictory roots cannot silently prefer one producer.
            spawn_issues.append(f'{eid}:deployment_spawn_root_conflict')
            self._invalid_causal_entities.add(eid)
        for eid in self._invalid_causal_entities:
            deployment_groups.pop(eid, None)
            spawn_groups.pop(eid, None)
            area_origins.pop(eid, None)
            projectile_origins.pop(eid, None)
        projectile_issues = []
        for ent in state.entities:
            cid = int(ent.get('card_id', 0))
            if ent.get('owner') not in (0, 1):
                continue
            eid = int(ent['id'])
            if any(t.entity_id == eid for t in self._towers.values()):
                continue
            gid = runtime_u32(ent.get('native_data_global_id')) or None
            cat = self.bundle.entity_archetype_catalog
            vocab = cat.runtime_global_vocab_id(gid) if gid else 1
            # Nulls generates some per-skin/per-form runtime IDs that are not
            # members of the frozen FirstLight vocabulary. They are unknown.
            if vocab <= 1:
                gid = None
            metadata = cat.metadata_for_vocab_id(vocab) if gid else None
            kind = metadata.child_kind if metadata and metadata.child_kind_known else 'unknown'
            hp = ent.get('hp')
            max_hp = ent.get('max_hp')
            if kind == 'unknown':
                spec = self.bundle.card_specs.get(cid)
                kind = spec.kind.value if spec else 'unknown'
                if kind == 'spell':
                    kind = 'effect'
            if max_hp and max_hp > 0 and hp is not None and hp <= 0:
                continue
            if gid:
                exact += 1
            # Missing native archetype is deliberately not inferred from the
            # source card (a Witch and her Skeleton have the same source card).
            measured_attack = attack_state(ent, state.tick, self._live_entities)
            projectile, issue = projectile_state(ent, self._live_entities, state.tick, self._velocity.get(eid),
                origin=projectile_origins.get(eid), source_untrusted=(
                    self.experimental_origins and ent.get('projectile_origin') is not None
                    and eid not in projectile_origins), bundle=self.bundle)
            if issue:
                projectile_issues.append({'id': eid, 'reason': issue})
            if projectile is not None:
                kind = 'projectile'
            evolution = evolved_entity_state(ent, self.bundle, state.tick)
            movement = movement_state(ent, state.tick)
            deployment = deployment_state(ent, state.tick)
            semantic = runtime_provenance(ENTITY_RUNTIME_SEMANTIC_FIELDS, measured_attack, state.tick)
            for field, value in (('movement_runtime', movement), ('deployment_runtime', deployment)):
                if value is not None:
                    semantic = replace(semantic,
                        field_evidence={**dict(semantic.field_evidence), field: SemanticEvidenceLevel.NATIVE_DERIVED},
                        source_fields={**dict(semantic.source_fields), field: (f'nulls-live.v3.entities.{field}',)})
            if evolution is not None:
                semantic = replace(semantic,
                    field_evidence={**dict(semantic.field_evidence), 'evolution_state': SemanticEvidenceLevel.NATIVE_DERIVED},
                    source_fields={**dict(semantic.source_fields), 'evolution_state': ('nulls-live.v3.entities.native_asset',)})
            if projectile is not None:
                semantic = replace(semantic,
                    field_evidence={**dict(semantic.field_evidence), 'projectile_state': SemanticEvidenceLevel.NATIVE_DERIVED},
                    source_fields={**dict(semantic.source_fields), 'projectile_state': ('nulls-live.v3.entities.projectile_runtime',)})
            result.append(EntityStateV1(entity_id=eid, owner=int(ent['owner']),
                card_id=evolution.card_id if evolution else cid if cid > 0 else None, entity_kind=kind,
                position=probe_to_world(ent['x'], ent['y']), velocity=self._velocity.get(eid),
                age_ms=max(0, state.tick - self._first_seen.get(eid, state.tick)) * 50,
                hitpoints=float(hp) if max_hp and max_hp > 0 else None,
                max_hitpoints=float(max_hp) if max_hp and max_hp > 0 else None,
                shield=float(ent['shield']) if 'shield' in ent else None,
                attack_state=measured_attack,
                visible_target=(measured_attack.target_entity if measured_attack and measured_attack.target_entity is not None
                                else projectile.target_entity if projectile and self.observation_profile == 'extended' else None),
                source_entity=(projectile.source_entity if projectile else
                    area_origins[eid]['parent_id'] if eid in area_origins else
                    spawn_groups[eid].parent_entity_id if eid in spawn_groups else None)
                    if self.observation_profile == 'extended' else None,
                projectile_state=projectile,
                movement_runtime=movement, deployment_runtime=deployment,
                evolution_state=evolution, runtime_provenance=self._effect_provenance(semantic, eid),
                effect_states=self._effects_by_entity.get(eid, ((), False))[0],
                native_data_global_id=gid,
                causal_group=(spawn_groups.get(eid) or deployment_groups.get(eid))
                    if kind in ('character', 'building', 'troop') else None))
        self.quality.update(entity_count=len(result), exact_archetype_count=exact,
                            shield_known_count=sum(e.shield is not None for e in result),
                            attack_known_count=sum(e.attack_state is not None for e in result),
                            attack_phase_known_count=sum(e.attack_state is not None and e.attack_state.phase.value != 'unknown' for e in result),
                            projectile_runtime_count=sum(e.projectile_state is not None for e in result),
                            projectile_runtime_issues=projectile_issues,
                            attack_target_count=sum(e.attack_state is not None and e.attack_state.target_entity is not None for e in result),
                            movement_runtime_count=sum(e.movement_runtime is not None for e in result),
                            deployment_runtime_count=sum(e.deployment_runtime is not None for e in result),
                            causal_groups_available=any(e.causal_group is not None for e in result),
                            causal_group_entity_count=sum(e.causal_group is not None for e in result),
                            causal_group_issues=deployment_issues,
                            spawn_relation_count=sum(e.entity_id in spawn_groups for e in result),
                            spawn_relation_issues=spawn_issues,
                            area_origin_count=sum(e.entity_id in area_origins for e in result),
                            area_origin_issues=area_issues,
                            area_origins={str(eid): row for eid, row in area_origins.items()},
                            projectile_origin_count=sum(e.entity_id in projectile_origins for e in result),
                            projectile_origin_issues=origin_issues,
                            runtime_effects_available=any(known for _, known in self._effects_by_entity.values()))
        return tuple(result)

    def build_placement_mask(self, card_id, lanes=(), towers=(), entities=(), form_code=0, ability_hud=False):
        spec = self.bundle.card_specs[card_id]
        tower_rects = tuple(OccupiedFootprintV1(*t.position, 4 if t.tower_kind == 'king' else 3,
            4 if t.tower_kind == 'king' else 3, f'tower:{t.entity_id}') for t in towers if t.hitpoints > 0)
        building_rects = []
        for ent in entities:
            other = self.bundle.card_specs.get(ent.card_id)
            footprint = native_building_footprint(other) if other and ent.entity_kind == 'building' else None
            if footprint:
                building_rects.append(OccupiedFootprintV1(*ent.position, footprint.width_tiles,
                    footprint.height_tiles, f'building:{ent.entity_id}'))
        key = (card_id, tuple(lanes), tower_rects, tuple(building_rects), form_code, ability_hud)
        if key in self._mask_cache:
            return self._mask_cache[key]
        artifact = card_placement_mask(spec, owner=self.actor_owner, ruleset_id='0' * 64,
            # Reference placement geometry distinguishes base/evolution only;
            # hero selection remains form_code=2 in the action contract.
            form='evolution' if form_code == 1 else 'base', destroyed_enemy_princess_lanes=lanes,
            active_tower_footprints=tower_rects or None, occupied_building_footprints=tuple(building_rects))
        dx, dy = artifact.model_subcell_offset or (0.0, 0.0)
        sign = 1 if self.actor_owner == 0 else -1
        enemy_king = tuple(t for t in towers if t.owner != self.actor_owner and
                           t.tower_kind == 'king' and t.hitpoints > 0)

        def spell_safe(world_x, world_y):
            if spec.kind.value != 'spell' or not enemy_king:
                return True
            radius = SPELL_IMPACT_RADIUS.get(int(card_id), 3000.0)
            for tower in enemy_king:
                dx = world_x - tower.position[0]
                dy = world_y - tower.position[1]
                # Circular spells: reject any impact envelope intersecting the
                # inactive king footprint plus a latency margin.
                if dx * dx + dy * dy <= (radius + KING_TOWER_RADIUS + KING_TOWER_SAFETY_MARGIN) ** 2:
                    return False
                # The Log is a swept line.  When the target is aligned with
                # the king lane, keep a larger longitudinal exclusion zone.
                if int(card_id) == 28000011 and abs(dx) <= 2200.0 and abs(dy) <= 9000.0:
                    return False
            return True

        def touchable(vx, vy):
            if not self.calibration.touchable(vx, vy):
                return False
            return not ability_hud or not covered_by_ability_hud(self.calibration.project(vx, vy),
                (self.calibration.width, self.calibration.height))
        rows = tuple(tuple(allowed and spell_safe((x+.5+sign*dx)*1000,
            (y+.5+sign*dy)*1000) and touchable(*world_to_view(
            (x+.5+sign*dx)*1000, (y+.5+sign*dy)*1000, self.actor_owner))
            for x, allowed in enumerate(row)) for y, row in enumerate(artifact.rows))
        entry = {'card_id': card_id, 'shape': (32, 18), 'row_major': rows,
            'model_subcell_offset': artifact.model_subcell_offset, 'placement_rule': artifact.rule.value,
            'footprint_width_tiles': artifact.footprint_width_tiles,
            'footprint_height_tiles': artifact.footprint_height_tiles,
            'collision_radius_units': artifact.collision_radius_units,
            'accuracy': artifact.accuracy.value, 'algorithm': artifact.algorithm,
            'mask_id': artifact.mask_id, 'form': artifact.form, 'reasons': artifact.reasons,
            'visible_card_id': card_id, 'effective_card_id': card_id, 'native_effective_card_id': card_id,
            'effective_cost': float(spec.elixir_cost), 'form_code': form_code, 'native_form_code': form_code}
        if len(self._mask_cache) > 128:
            self._mask_cache.clear()
        self._mask_cache[key] = entry
        return entry

    def spell_activation_risk(self, action, towers):
        """Return whether a selected spell can affect an enemy king tower."""
        if action.target_grid is None:
            return False
        spec = self.bundle.card_specs.get(int(action.card_id))
        if spec is None or spec.kind.value != 'spell':
            return False
        x, y = action_world(action)
        for tower in towers:
            if tower.owner == self.actor_owner or tower.tower_kind != 'king' or tower.hitpoints <= 0:
                continue
            radius = SPELL_IMPACT_RADIUS.get(int(action.card_id), 3000.0)
            if (x - tower.position[0]) ** 2 + (y - tower.position[1]) ** 2 <= (radius + KING_TOWER_RADIUS + KING_TOWER_SAFETY_MARGIN) ** 2:
                return True
            if int(action.card_id) == 28000011 and abs(x - tower.position[0]) <= 2200.0 and abs(y - tower.position[1]) <= 9000.0:
                return True
        return False

    def defensive_lane_conflict(self, action, state):
        """Detect an obvious cross-lane defensive placement mistake.

        This is intentionally conservative: it only fires for core defensive
        cards when one lane has a nearby enemy and the selected lane has none.
        It is not a strategy override for attacks or split-lane situations.
        """
        defensive = {26000010, 26000014, 26000030, 26000038, 27000000}
        if int(action.card_id or 0) not in defensive or action.target_grid is None:
            return None
        target_x, _ = action_world(action)
        target_lane = 'left' if target_x < 9000.0 else 'right'
        enemy = []
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner or int(ent.get('card_id', -1)) <= 0:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            x, y = float(ent['x']), float(ent['y'])
            # Only units in the central combat/defensive half count as an
            # immediate threat; back-line units should not force a lane flip.
            if min(y, 32000.0 - y) > 14500.0:
                continue
            lane = 'left' if x < 9000.0 else 'right'
            enemy.append((lane, x, y))
        if not enemy:
            return None
        threat_lanes = {lane for lane, _, _ in enemy}
        if target_lane in threat_lanes or len(threat_lanes) != 1:
            return None
        threat_lane = next(iter(threat_lanes))
        return {'target_lane': target_lane, 'threat_lane': threat_lane,
                'enemy_count': len(enemy)}

    def lead_log_action(self, action, state, latency_ms):
        """Lead a rolling Log toward a measured moving enemy.

        The policy target is a cell center.  When the target is moving and no
        native future target was supplied by the mirror, compensate only for
        the measured input path; never guess a target for a stationary or
        ambiguous board.
        """
        if int(action.card_id or 0) != 28000011 or action.target_grid is None:
            return action, None
        target = action_world(action)
        candidates = []
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner or int(ent.get('id', 0)) <= 0:
                continue
            velocity = self._velocity.get(int(ent['id']))
            if velocity is None:
                continue
            distance = self._distance(target, (float(ent['x']), float(ent['y'])))
            speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
            if speed >= 80.0 and distance <= 4500.0:
                candidates.append((distance, velocity))
        if not candidates:
            return action, None
        _, (vx, vy) = min(candidates, key=lambda row: row[0])
        # Do not move a target by less than half a cell; that amount is within
        # normal grid/rounding error and an unconditional shift is worse than
        # using the policy's original center.
        lead_ticks = max(1, min(6, round(float(latency_ms) / 50.0)))
        if lead_ticks <= 0:
            return action, None
        x, y = target
        shifted = (max(0.0, min(17999.0, x + vx * lead_ticks)),
                   max(0.0, min(31999.0, y + vy * lead_ticks)))
        if self._distance(target, shifted) < 500.0:
            return action, None
        grid = (max(0, min(17, int(shifted[0] // 1000))),
                max(0, min(31, int(shifted[1] // 1000))))
        if grid == action.target_grid:
            return action, None
        return replace(action, target_grid=grid), {
            'lead_ticks': lead_ticks, 'from_grid': action.target_grid,
            'to_grid': grid, 'velocity': (round(vx, 2), round(vy, 2))}

    @staticmethod
    def _distance(a, b):
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    def _forecast_entities(self, entities, horizon_ticks):
        """Build a short, card-aware model view without changing live telemetry.

        This is intentionally a kinematic forecast, not a second battle engine.
        Native runtime facts are authoritative when present; CardSpec supplies
        only static traits such as range and card kind.  In particular, a unit
        is not extrapolated through deployment, an attack lock, or a target that
        is already inside its proved attack range.
        """
        if not config.ENABLE_MODEL_PREDICTION_OVERLAY:
            return tuple(entities)
        if horizon_ticks <= 0:
            return tuple(entities)
        # Kinematics cannot forecast future target switches or combat deaths.
        # Keep this fallback short even when the native rollout horizon is long.
        horizon_ticks = min(horizon_ticks, 10)
        by_id = {tower.entity_id: tower for tower in getattr(self, '_towers', {}).values()
                 if tower.hitpoints > 0}
        by_id.update({entity.entity_id: entity for entity in entities})
        movement_blocking_phases = {
            AttackPhase.WINDUP, AttackPhase.RELEASE, AttackPhase.BACKSWING,
            AttackPhase.COOLDOWN, AttackPhase.CHANNEL, AttackPhase.CHARGING,
        }
        result = []
        for entity in entities:
            # Projectiles have their own measured trajectory. Effects/spells,
            # towers and buildings do not represent walking units.
            if entity.velocity is None or entity.entity_kind in ('tower', 'building', 'effect'):
                result.append(entity)
                continue
            bundle = getattr(self, 'bundle', None)
            spec = bundle.card_specs.get(entity.card_id) if bundle is not None and entity.card_id else None
            if spec is not None and spec.kind.value in ('spell', 'building'):
                result.append(entity)
                continue

            attack = entity.attack_state
            movement = entity.movement_runtime or {}
            deployment = entity.deployment_runtime or {}
            freeze = movement.get('phase') == 'stationary'
            # Do not predict a unit walking while its native deployment timer
            # is still active.  A missing/invalid timer remains unknown and is
            # deliberately not treated as a reason to invent a pause.
            remaining = deployment.get('remaining_wall_ms', deployment.get('remaining_ms'))
            if isinstance(remaining, int) and remaining > 0:
                freeze = True
            target = None
            if attack is not None:
                freeze = freeze or attack.locked is True or attack.phase in movement_blocking_phases
                target = by_id.get(attack.target_entity) if attack.target_entity is not None else None
                # A target lock alone is not enough: melee units can acquire a
                # target before reaching it.  Freeze only when the target is
                # inside the card's static range, plus a small collision
                # allowance.  This is what prevents a Princess from being
                # forecast through her firing position while preserving Hog's
                # movement toward a building outside melee range.
                if target is not None and (target.owner == entity.owner or
                        (target.hitpoints is not None and target.hitpoints <= 0)):
                    target = None
                if target is not None and spec is not None and spec.range_tiles is not None:
                    freeze = freeze or self._distance(entity.position, target.position) <= (
                        float(spec.range_tiles) * 1000.0 + 750.0)
            if freeze:
                result.append(replace(entity, velocity=(0.0, 0.0),
                    age_ms=(entity.age_ms or 0) + horizon_ticks * 50))
                continue
            x, y = entity.position
            vx, vy = entity.velocity
            # Do not extend a noisy two-frame acceleration across the horizon.
            # Solve first entry into the target's attack-range circle using
            # relative motion. Stop there rather than overshooting the target.
            travel_ticks = horizon_ticks
            stopped = False
            if target is not None and spec is not None and spec.range_tiles is not None:
                tx, ty = target.position
                tvx, tvy = getattr(target, 'velocity', None) or (0.0, 0.0)
                target_attack = getattr(target, 'attack_state', None)
                if target_attack is not None and (target_attack.locked is True or
                        target_attack.phase in movement_blocking_phases):
                    tvx, tvy = 0.0, 0.0
                dx, dy = x - tx, y - ty
                rvx, rvy = vx - tvx, vy - tvy
                radius = float(spec.range_tiles) * 1000.0 + 750.0
                a = rvx * rvx + rvy * rvy
                b = 2.0 * (dx * rvx + dy * rvy)
                c = dx * dx + dy * dy - radius * radius
                discriminant = b * b - 4.0 * a * c
                if a > 0 and b < 0 and discriminant >= 0:
                    entry = (-b - math.sqrt(discriminant)) / (2.0 * a)
                    if 0 <= entry <= travel_ticks:
                        travel_ticks, stopped = entry, True
            result.append(replace(entity,
                position=(max(0.0, min(17999.0, x + vx * travel_ticks)),
                          max(0.0, min(31999.0, y + vy * travel_ticks))),
                velocity=(0.0, 0.0) if stopped else (vx, vy),
                age_ms=(entity.age_ms or 0) + horizon_ticks * 50))
        return tuple(result)

    def _model_entities(self, entities, virtual_entities=(), horizon_ticks=0):
        """Return the model-only predicted entity view for a real snapshot."""
        measured_latency = next((float(v['latency_ms']) for v in virtual_entities
                                 if isinstance(v.get('latency_ms'), (int, float))
                                 and v['latency_ms'] >= 0),
                                float(config.PREDICTION_LATENCY_COMPENSATION_MS))
        compensation_ticks = max(0, round(measured_latency / (config.TICK_SECONDS * 1000.0)))
        # Always forecast the complete decision horizon, even when no card is
        # pending: the model must see where the live board will be after the
        # end-to-end delay plus the next 1.5 seconds. A pending card is additionally
        # inserted at the post-delay point below and remains in deployment
        # state until its CardSpec deploy time elapses.
        forward_ticks = compensation_ticks + horizon_ticks
        forecast = self._forecast_entities(entities, forward_ticks)
        virtual = []
        for v in virtual_entities:
            card_id = int(v['card_id'])
            spec = self.bundle.card_specs.get(card_id)
            if spec is None:
                continue
            # The virtual entity enters the simulated world only after the
            # measured end-to-end input delay.  Deployment animation is a
            # card-local phase, not extra global time: adding it to
            # ``forward_ticks`` would fast-forward every other unit and could
            # make the policy issue overlapping counters.  Keep the complete
            # age for bookkeeping, while exposing the deployment clock below.
            age_ms = (float(v.get('age_ms', 0.0)) +
                      (compensation_ticks + horizon_ticks) *
                      int(config.TICK_SECONDS * 1000))
            deploy_ms = int(spec.deploy_time_ms or 0)
            deployment = None
            runtime_provenance = None
            if spec.kind.value in ('troop', 'building') and deploy_ms > 0:
                # Runtime extension fields are native millisecond counts and
                # the contract intentionally rejects fractional values. The
                # prediction clock may be fractional because it includes
                # measured host latency, so quantize only at the contract
                # boundary while keeping ``age_ms`` precise for forecasting.
                # The virtual card is scheduled to enter the game after the
                # measured end-to-end compensation.  That pre-entry interval
                # must not consume its native deployment animation; only the
                # forecast horizon advances deployment time.
                deployment_elapsed_ms = min(
                    deploy_ms,
                    horizon_ticks * int(config.TICK_SECONDS * 1000))
                remaining = max(0, int(round(deploy_ms - deployment_elapsed_ms)))
                deployment = {
                    'version': 'deployment-runtime.v1',
                    'phase': 'deploying' if remaining else 'active',
                    'remaining_native_ms': remaining,
                    'remaining_wall_ms': remaining,
                    'configured_deploy_time_ms': deploy_ms,
                    'observed_step_native_ms': None,
                    'observed_tick': int(v.get('decision_tick', 0)),
                    'complete': True,
                }
                runtime_provenance = SemanticProvenanceV1(
                    {field: (SemanticEvidenceLevel.NATIVE_DERIVED
                             if field == 'deployment_runtime' else SemanticEvidenceLevel.UNKNOWN)
                     for field in ENTITY_RUNTIME_SEMANTIC_FIELDS},
                    observed_tick=int(v.get('decision_tick', 0)),
                    notes=('model_prediction_deployment_timing',))
            virtual.append(EntityStateV1(
                entity_id=0xF0000000 + (int(v['command_seq']) & 0x0FFFFFFF),
                owner=int(v['owner']), card_id=card_id, entity_kind=spec.kind.value,
                position=probe_to_world(v['x'], v['y']), age_ms=age_ms, visible=True,
                deployment_runtime=deployment,
                runtime_provenance=runtime_provenance or
                    SemanticProvenanceV1.unknown_all(ENTITY_RUNTIME_SEMANTIC_FIELDS)))
        return forecast + tuple(virtual)

    def build_observation(self, state, blocked_slots=(), reserved_elixir=0.0,
                          blocked_abilities=(), virtual_entities=(),
                          prediction_overlay=False,
                          prediction_horizon_ticks=config.PREDICTION_HORIZON_TICKS):
        self.observe(state)
        self._live_entities = live_entities(state.entities)
        # A dying object can still occupy the native manager and be the exact
        # BUFF producer until cleanup clears its reference. Use all observed
        # identities for provenance, while only living objects receive tokens.
        observed_identities = {e['id']: e for e in state.entities}
        effect_identities = (observed_identities if self.observation_profile == 'extended'
                             else self._live_entities)
        # Diagnostics retain complete native identities in both profiles;
        # reference effect state independently uses only current live sources.
        self._effect_origins.preflight(state.raw, observed_identities, state.tick)
        self._effects_by_entity = {}
        effect_issues = []
        for ent in self._live_entities.values():
            effects, known, issues = self._effect_reader.project(ent, effect_identities, state.tick)
            self._effects_by_entity[ent['id']] = (effects, known)
            effect_issues.extend({'id': ent['id'], 'reason': issue} for issue in issues)
        self.quality.update(effect_runtime_known_count=sum(k for _, k in self._effects_by_entity.values()),
            active_effect_count=sum(len(e) for e, _ in self._effects_by_entity.values()),
            active_effect_issues=effect_issues)
        towers = self._build_towers(state)
        real_entities = self._build_entities(state)
        # This view only supplies identity/ability membership below. Position
        # prediction belongs to tensorize(), after authoritative observation
        # construction; doing it here duplicates every forecast and allocation.
        entities = real_entities + tuple(
            # The frozen contract requires non-negative entity IDs. Keep
            # predictions in a high reserved range so they cannot collide
            # with normal native runtime identities.
            EntityStateV1(entity_id=0xF0000000 + (int(v['command_seq']) & 0x0FFFFFFF), owner=int(v['owner']),
                card_id=int(v['card_id']), entity_kind=self.bundle.card_specs[int(v['card_id'])].kind.value,
                position=probe_to_world(v['x'], v['y']), age_ms=0,
                visible=True)
            for v in virtual_entities if prediction_overlay and int(v['card_id']) in self.bundle.card_specs)
        slots = {i: c for i, c in enumerate(state.hand_cards) if c > 0}
        if len(set(slots.values())) != len(slots) or any(c not in self.current_deck for c in slots.values()):
            raise TelemetryError('live hand is inconsistent with the deck')
        player = self._player(state)
        try:
            selections = card_selections(player)
        except ValueError as exc:
            raise TelemetryError(str(exc)) from exc
        self.quality['card_selection_source'] = 'native_builder' if selections is not None else 'legacy_unverified_base'
        self.quality['selected_forms'] = {str(c): r.get('active_form') for c, r in (selections or {}).items()}
        if self.hero_mode is None:
            self.hero_musketeer = bool(selections and selections.get(MUSKETEER, {}).get('active_form') == 2)
        self.quality['form_detection_mode'] = 'auto' if self.hero_mode is None else 'explicit'
        evolutions, evolution_issues = (), []
        for tracker in self._evolutions.values():
            states, issues = tracker.observe(selections, self.bundle, state.tick)
            evolutions += states
            evolution_issues.extend(issues)
        evolution_by_card = {e.card_id: e for e in evolutions}
        self.quality['evolution_runtime_issues'] = evolution_issues
        self.quality['skeleton_evolution_activation_known'] = True if self._evolutions[SKELETONS].enabled_observed else None
        self.quality['evolution_activation_known'] = {str(cid): True if t.enabled_observed else None
            for cid, t in self._evolutions.items()}
        if evolutions and self.tensorizer is not None:
            # Add newly observed role evidence without resetting recurrent/action history.
            for evolution in evolutions:
                self.tensorizer.deck_roles[evolution.card_id] = (False, True)
        cycle_raw = player.get('cycle')
        cycle = tuple(int(c) for c in cycle_raw) if cycle_raw is not None else ()
        if cycle_raw is not None and (set(cycle) & set(slots.values()) or
                len(cycle) + len(slots) != 8 or set(cycle) | set(slots.values()) != set(self.current_deck)):
            raise TelemetryError('native hand/cycle snapshot is inconsistent')
        self.quality['cycle_known'] = cycle_raw is not None
        next_card = cycle[0] if cycle else None
        masks, playable = {}, [False] * 4
        slot_reasons = {str(slot): 'empty' for slot in range(4)}
        elixir = max(0.0, state.elixir - reserved_elixir)
        lanes = tuple(sorted(self.destroyed_enemy_princess_lanes))
        ability_hud = self.hero_musketeer or any(r.get('ability_name') == ABILITY
            for r in player.get('ability_runtime', []))
        self.quality['ability_hud_excluded'] = ability_hud
        for slot, cid in slots.items():
            spec = self.bundle.card_specs[cid]
            if selections is not None:
                selected = selections[cid]
                if selected.get('active_form') is None or selected.get('selected_cost') is None:
                    slot_reasons[str(slot)] = 'native_selection_unknown'
                    continue
                supported_hero = self.hero_musketeer and cid == MUSKETEER and selected['active_form'] == 2
                supported_evolution = (self.skeleton_evolution and selected['active_form'] == 1
                    and cid in evolution_by_card and evolution_by_card[cid].ready is True)
                if selected['active_form'] != 0 and not (supported_hero or supported_evolution):
                    slot_reasons[str(slot)] = 'special_form_execution_not_ready'
                    continue
                if selected['selected_cost'] != spec.elixir_cost:
                    slot_reasons[str(slot)] = 'native_cost_mismatch'
                    continue
            # ability_ids includes OPTIONAL hero forms of ordinary cards
            # (Musketeer and Ice Golem). It does not classify their base form.
            # Keep unsupported direct champions/Mirror out of this base bridge.
            if cid == 28000006 or spec.categorical_features.get('rarity') == 'Champion':
                slot_reasons[str(slot)] = 'unsupported_base_card'
                continue
            if slot in blocked_slots:
                slot_reasons[str(slot)] = 'pending_or_cooldown'
                continue
            if elixir < spec.elixir_cost:
                slot_reasons[str(slot)] = 'insufficient_elixir'
                continue
            entry = self.build_placement_mask(cid, lanes, towers, entities,
                form_code=selections[cid]['active_form'] if selections is not None else 0,
                ability_hud=ability_hud)
            playable[slot] = any(any(row) for row in entry['row_major'])
            slot_reasons[str(slot)] = 'playable' if playable[slot] else 'no_legal_position'
            if playable[slot]:
                masks[str(slot)] = entry
        ready = ready_abilities(player, state.entities, self.bundle, elixir, blocked_abilities) if self.hero_musketeer and self.hero_skill_ready else ()
        source_ids = {a.source_entity for a in ready}
        # The validated controller binds the actual carrier, not every unit
        # sharing the source card. This also supplies the model's hero form.
        carrier_states, _, _ = ability_states(player, state.entities, self.bundle, state.tick)
        carriers = {a.source_entity for a in carrier_states if a.ability_id == 'Musketeer_hero_Ability'}
        entities = tuple(replace(e, entity_kind='hero') if e.entity_id in carriers and e.owner == self.actor_owner else e for e in entities)
        real_output_entities = tuple(replace(e, entity_kind='hero') if e.entity_id in carriers and e.owner == self.actor_owner else e
                                     for e in real_entities)
        source_ids &= {e.entity_id for e in entities}
        mask = ActionMaskV1(kinds={'wait': True, 'play_card': any(playable), 'activate_ability': bool(source_ids)},
            ability_sources=tuple(sorted(source_ids)),
            hand_slots=tuple(playable), placement_masks=masks,
            reasons={'effective_elixir': elixir, 'reserved_elixir': reserved_elixir,
                     'slot_reasons': slot_reasons})
        crowns = {}
        for owner in (0, 1):
            enemy = [t for t in towers if t.owner != owner]
            crowns[owner] = 3 if any(t.tower_kind == 'king' and t.hitpoints <= 0 for t in enemy) else sum(
                t.hitpoints <= 0 for t in enemy if t.tower_kind != 'king')
        abilities, ability_issues, player_provenance = ability_states(player, state.entities, self.bundle, state.tick)
        self.quality['ability_runtime_count'] = len(abilities)
        self.quality['ability_runtime_issues'] = ability_issues
        self.quality['hero_musketeer_execution_enabled'] = self.hero_musketeer
        self.quality['hero_skill_input_ready'] = self.hero_skill_ready
        self.quality['skeleton_evolution_execution_enabled'] = self.skeleton_evolution
        self.quality['evolution_execution_cards'] = list(BINDINGS) if self.skeleton_evolution else []
        if evolutions:
            player_provenance = replace(player_provenance,
                field_evidence={**dict(player_provenance.field_evidence), 'evolution_runtime_states': SemanticEvidenceLevel.NATIVE_DERIVED},
                source_fields={**dict(player_provenance.source_fields), 'evolution_runtime_states': ('nulls-live.v3.players.card_runtime',)})
        own = PlayerStateV1(owner=self.actor_owner, crowns=crowns[self.actor_owner],
            elixir_exact=state.elixir, elixir_visible=state.elixir, hand=tuple(slots.values()),
            cycle=cycle, next_card=next_card, deck=self.current_deck, private_state_visible=True,
            ability_runtime_states=abilities, runtime_provenance=player_provenance,
            evolution_runtime_states=evolutions,
            metadata={'hand_slot_by_card': {cid: slot for slot, cid in slots.items()},
                'hand_runtime_by_slot': {s: {'form_code': selections[cid]['active_form']}
                    if selections is not None and selections[cid].get('active_form') is not None
                    # The frozen tensorizer requires an integer, with no unknown
                    # token. Unknown selections stay masked and quality-marked.
                    else {'form_code': 0}
                    for s, cid in slots.items()}})
        enemy = PlayerStateV1(owner=1-self.actor_owner, crowns=crowns[1-self.actor_owner],
            revealed_cards=tuple(sorted(self._revealed[1-self.actor_owner])))
        elapsed = state.tick * 50
        # Training uses the full five-minute horizon, not the HUD's phase clock.
        remaining = max(0, 300000 - elapsed)
        timing = TimeStateV1(elapsed_ms=elapsed, remaining_ms=remaining,
            elixir_multiplier=1.0 if elapsed < 120000 else (2.0 if elapsed < 240000 else 3.0), tick_ms=50)
        impacts, impact_issues = self._impact_events.project(state.raw, state.entities, self.bundle, state.tick)
        self.quality['impact_event_count'] = len(impacts)
        self.quality['impact_event_issues'] = impact_issues
        attacks = attack_events(state.entities, state.tick, self.bundle)
        self.quality['attack_event_count'] = len(attacks)
        damage, damage_issues = self._damage_events.project(state.raw, state.entities, self.bundle, state.tick)
        self.quality['damage_event_count'] = len(damage)
        self.quality['damage_event_issues'] = damage_issues
        heals, heal_issues = self._heal_events.project(state.raw, state.entities, self.bundle, state.tick)
        self.quality['heal_event_count'] = len(heals)
        self.quality['heal_event_issues'] = heal_issues
        public_events = tuple(e for e in self._events if e.tick >= state.tick - 80)
        # Rich snapshot state remains available in both profiles. Exact native
        # per-hit/history events are independently recorded but only the
        # explicit extended profile injects them into the frozen event branch.
        # Event deltas must remain anchored to the real observation. Only the
        # model-facing entity positions use the short forecast.
        snapshot_events = self._reference_events.project(state.tick, real_entities, towers)
        extensions = impacts + attacks + damage + heals
        # Keep shared public transitions in extended too. Replace the damage,
        # shield and disappearance families instead of double-counting one hit
        # through both snapshot and native-exact event producers.
        replaced_types = {'damage', 'tower_damage', 'shield_damage', 'shield_break', 'death_or_despawn'}
        selected = (snapshot_events if self.observation_profile == 'reference' else
                    tuple(e for e in snapshot_events if e.event_type not in replaced_types) + extensions)
        recent = public_events + selected
        self.quality.update(reference_event_count=len(snapshot_events),
            extended_event_count=len(extensions), model_combat_event_count=len(selected),
            model_event_source='snapshot_delta' if self.observation_profile == 'reference' else 'snapshot_transitions+native_exact',
            historical_effect_sources=self.observation_profile == 'extended')
        # The returned observation is always the measured snapshot. Prediction
        # is assembled by tensorize() for the model only, so receipts, event
        # correlation and lifecycle gates never see invented positions.
        return ObservationV1(tier=ObservationTier.FAIR, owner=self.actor_owner, tick=state.tick,
            time=timing, phase='normal' if elapsed < 180000 else 'overtime', episode_id=self.episode_id,
            players=(own, enemy), towers=towers, entities=real_output_entities, events=recent, action_mask=mask,
            terminal=TerminalV1(), metadata={'telemetry_quality': dict(self.quality)})

    def tensorize(self, state, blocked_slots=(), reserved_elixir=0.0,
                 blocked_abilities=(), virtual_entities=(),
                 prediction_horizon_ticks=config.PREDICTION_HORIZON_TICKS,
                 simulation_entities=None):
        if self.tensorizer is None:
            raise RuntimeError('reset_match must be called first')
        obs = self.build_observation(state, blocked_slots, reserved_elixir,
                                     blocked_abilities, virtual_entities,
                                     True, prediction_horizon_ticks)
        if simulation_entities is not None:
            pending_entities = self._model_entities((), virtual_entities,
                                                     prediction_horizon_ticks)
            model_entities = tuple(simulation_entities) + tuple(pending_entities)
        else:
            model_entities = self._model_entities(obs.entities, virtual_entities,
                                                  prediction_horizon_ticks)
        model_obs = replace(obs, entities=model_entities)
        batch = self.tensorizer.tensorize(model_obs, validate=True)
        relations = batch.relation_edges
        kinds = relations.relation_type[0][relations.mask[0]].numpy()
        self.quality.update(relation_edge_count=len(kinds),
                            target_relation_count=int((kinds == REL_TARGETS).sum()),
                            source_relation_count=int((kinds == REL_SOURCE_OF).sum()))
        if self.oracle_elixir:
            # Optional privileged experiment, explicitly outside the training
            # FAIR elixir estimate. Names/indices come from MATCH_SCALAR_NAMES.
            enemy_elixir = float(self._player(state, 1-self.actor_owner)['elixir'])
            batch.match_scalars[0, 15] = enemy_elixir / 10.0
            batch.match_scalars[0, 16] = 0.0
        return batch, obs
