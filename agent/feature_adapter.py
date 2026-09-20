"""Translate measured Null's telemetry into the original V4 contracts.

Owner identity is never renumbered. Missing telemetry remains unknown; no
invented hands, card cycles, parent groups or runtime character identities.
"""
import sys
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import config
if str(config.FIRSTLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(config.FIRSTLIGHT_DIR))

from native_runner.contracts import (ObservationV1, ObservationTier, PlayerStateV1,
    TowerStateV1, EntityStateV1, ActionMaskV1, TimeStateV1, TerminalV1, EventV1,
    TOWER_RUNTIME_SEMANTIC_FIELDS, ENTITY_RUNTIME_SEMANTIC_FIELDS, SemanticEvidenceLevel,
    SemanticProvenanceV1, AttackPhase, ActionV1, ActionKind, TargetKind)
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
HOG_RIDER = 26000021
ICE_SPIRIT = 26000030
ICE_GOLEM = 26000038
CANNON = 27000000
FIREBALL = 28000000
THE_LOG = 28000011
MINER = 26000032
DEFENSIVE_LANE_CARDS = frozenset((26000010, 26000014, 26000030, 26000038, CANNON))
COUNTERPUSH_SUPPORT_CARDS = frozenset((26000014, 203000014, 26000038))
NEUTRAL_PATIENCE_CARDS = frozenset((HOG_RIDER, MUSKETEER, CANNON, FIREBALL))
ATTACK_HOLD_TOWER_DISTANCE = 7000.0
ATTACK_HOLD_MIN_EFFECTIVE_ELIXIR = 8.0
NEUTRAL_PATIENCE_RELEASE_ELIXIR = 8.5
RECENT_BUILDING_ATTACK_WINDOW_TICKS = 50
LOW_ELIXIR_ATTACK_MAX_UPPER = 4.0
LOW_ELIXIR_ATTACK_MAX_WIDTH = 1.0
COUNTERPUSH_MIN_PROGRESS = 9000.0
COUNTERPUSH_MAX_PROGRESS = 17000.0
INCOMING_PUSH_MIN_COST = 5.0
INCOMING_PUSH_BACKFIELD_DEPTH = 22000.0
INCOMING_PUSH_DEFENDER_RELEASE_DEPTH = 18000.0
INCOMING_PUSH_PUNISH_MIN_ELIXIR = 7.0
INCOMING_PUSH_PLAY_BIND_TICKS = 16
INCOMING_PUSH_RECENT_HEAVY_TICKS = 30
INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH = 14000.0
HEAVY_DEFEND_CANNON_RELEASE_DEPTH = 11500.0
HEAVY_DEFEND_BODY_RELEASE_DEPTH = 14500.0
HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH = 15500.0
HEAVY_DEFEND_EMERGENCY_DEPTH = 4500.0
HEAVY_DEFEND_CANNON_MIN_DEPTH = 4500.0
HEAVY_DEFEND_CANNON_MAX_DEPTH = 11500.0
HEAVY_DEFEND_MUSKETEER_MAX_DEPTH = 9500.0
HEAVY_DEFEND_ICE_GOLEM_MIN_DEPTH = 5000.0
HEAVY_DEFEND_ICE_GOLEM_MAX_DEPTH = 12500.0
HEAVY_DEFEND_CHEAP_MIN_DEPTH = 5500.0
HEAVY_DEFEND_CHEAP_MAX_DEPTH = 14000.0
HEAVY_DEFEND_FIREBALL_CLUSTER_RADIUS = 5000.0
HEAVY_DEFEND_FIREBALL_SUPPORT_MIN_COST = 3.0
HEAVY_DEFEND_FIREBALL_TARGET_RADIUS = 5500.0
INCOMING_PUSH_CORE_DEFENDERS = frozenset((MUSKETEER, CANNON))
HEAVY_DEFEND_SERIAL_CARDS = frozenset((CANNON, MUSKETEER, ICE_GOLEM))
HEAVY_DEFEND_CHEAP_CONTROL = frozenset((SKELETONS, ICE_SPIRIT))
INCOMING_PUSH_HARD_RESERVE_CARDS = frozenset((FIREBALL,))

# A visible medium/heavy troop deployed deep in the opponent backfield is a
# commitment, but not yet something that warrants mirroring with an opposite-
# lane Musketeer/Ice Golem or burning Cannon lifetime before engagement.
BACKFIELD_COMMITMENT_MIN_COST = 3.0
BACKFIELD_COMMITMENT_START_DEPTH = 22000.0
BACKFIELD_COMMITMENT_RELEASE_DEPTH = 14500.0

# Role-specific release windows.  Cannon is deliberately latest because its
# lifetime is finite; Musketeer may establish the backline earlier, while Ice
# Golem is staged close enough to become an actual body/kite.
BACKFIELD_MUSKETEER_RELEASE_DEPTH = 16000.0
BACKFIELD_ICE_GOLEM_RELEASE_DEPTH = 15000.0
BACKFIELD_CANNON_RELEASE_DEPTH = 12500.0

# Anti-overflow while defending/preparing.  At the soft threshold we may force
# a proved safe cycle action.  The hard cap permits Ice Golem as a same-lane
# body only when no cheaper safe release is available.
DEFENSE_OVERFLOW_ELIXIR = 9.5
DEFENSE_OVERFLOW_HARD_CAP_ELIXIR = 9.9
DEFENSE_OVERFLOW_SAFE_CYCLE_CARDS = frozenset((SKELETONS, ICE_SPIRIT))
LOW_VALUE_DEFENSE_MAX_COST = 1.0
LOW_VALUE_DEFENSE_HELD_CARDS = frozenset((
    MUSKETEER, CANNON, FIREBALL, ICE_GOLEM,
))
LOW_VALUE_DEFENSE_OVERFLOW_CARDS = frozenset((
    SKELETONS, ICE_SPIRIT, THE_LOG,
))

# Cannon may be placed before the tank reaches its final pull radius, provided
# the observed approach says the tank should arrive while the building still
# has useful lifetime.  Nine seconds is intentionally conservative; the
# fallback depth handles the first frame before velocity is measurable.
CANNON_PREBUILD_MAX_LEAD_TICKS = 180
CANNON_PREBUILD_URGENT_LEAD_TICKS = 90
CANNON_PREBUILD_FALLBACK_DEPTH = 18000.0

# Telemetry/replay frames can occasionally jump after a stale frame or a
# synthetic test update. Such a jump must not make Cannon look "urgently"
# due. Real troop movement is far below this ceiling in world units per native
# tick, so values above it are treated as unreliable rather than extrapolated.
CANNON_PREBUILD_MAX_REASONABLE_SPEED_PER_TICK = 500.0


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
        self._active_enemy_buildings = {}
        self._recent_enemy_building_expiry = None
        self._opponent_exact_play_count = 0
        self._opponent_last_play_count = {}
        self._last_opponent_exact_play = None
        self._recent_opponent_heavy_plays = deque(maxlen=8)
        self._incoming_push_cores = {}
        self._backfield_commitments = {}
        self._prelock_context = None

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
                changed = old > 0 and old != hand.get(slot, 0)
                if changed and owner != self.actor_owner:
                    # Every positive opponent hand transition counts as one
                    # public card play for the four-card return rule, even when
                    # the exact effect/cost is unsupported or Mirror-specific.
                    self._opponent_exact_play_count += 1
                    if old in self.bundle.card_specs and old != 28000006:
                        spec = self.bundle.card_specs[old]
                        self._opponent_last_play_count[int(old)] = self._opponent_exact_play_count
                        self._last_opponent_exact_play = {
                            'card_id': int(old),
                            'cost': float(spec.elixir_cost),
                            'kind': spec.kind.value,
                            'tick': int(state.tick),
                            'play_count': int(self._opponent_exact_play_count),
                        }
                        if (spec.kind.value == 'troop'
                                and float(spec.elixir_cost) >= INCOMING_PUSH_MIN_COST):
                            self._recent_opponent_heavy_plays.append({
                                'card_id': int(old),
                                'cost': float(spec.elixir_cost),
                                'tick': int(state.tick),
                                'play_count': int(self._opponent_exact_play_count),
                                'bound_entity_id': None,
                            })
                if changed and old in self.bundle.card_specs:
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
        # Track only public opponent building lifecycle. A building becoming
        # visible is public board state; its disappearance starts a short,
        # conservative Hog opportunity window without inferring hidden hand.
        current_enemy_buildings = {}
        for ent in state.entities:
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue
            cid = int(ent.get('card_id', -1))
            spec = self.bundle.card_specs.get(cid)
            hp = ent.get('hp')
            if (cid > 0 and spec is not None and spec.kind.value == 'building'
                    and (hp is None or float(hp) > 0)):
                current_enemy_buildings[int(ent['id'])] = cid
        expired = [
            (eid, cid) for eid, cid in self._active_enemy_buildings.items()
            if eid not in current_enemy_buildings
        ]
        for eid, cid in expired:
            # If another copy of the same building is still alive, there is no
            # clean "building is gone" attack window yet.
            if cid not in current_enemy_buildings.values():
                self._recent_enemy_building_expiry = {
                    'entity_id': eid,
                    'card_id': cid,
                    'tick': state.tick,
                }
        self._active_enemy_buildings = current_enemy_buildings

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
        miner_landing_targets = 0
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
            miner_target = None
            if cid == MINER and deployment is not None:
                # Miner is not represented as a regular projectile by this
                # client build.  During its native deployment/tunnel phase,
                # its object coordinates are the only measured landing-side
                # evidence exposed by the live probe.  Bind that measured
                # position to the nearest *opponent* tower (including king)
                # so the model receives a concrete lane relation instead of
                # having to infer left/right from an underground sprite.
                candidates = [tower for tower in self._towers.values()
                              if tower.owner != int(ent['owner']) and tower.hitpoints > 0]
                if candidates:
                    x, y = probe_to_world(ent['x'], ent['y'])
                    miner_target = min(candidates, key=lambda tower:
                        (tower.position[0] - x) ** 2 + (tower.position[1] - y) ** 2).entity_id
                    miner_landing_targets += 1
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
                # A validated projectile target is direct native runtime
                # evidence, not a guessed origin chain.  Keep it in the
                # reference profile as well: aerial delivery cards such as a
                # Goblin Barrel otherwise expose only their current position,
                # letting the policy mistake the target lane while the barrel
                # is still in flight.
                visible_target=(measured_attack.target_entity if measured_attack and measured_attack.target_entity is not None
                                else projectile.target_entity if projectile else miner_target),
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
                            miner_landing_target_count=miner_landing_targets,
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

    def _near_tower_pressure(self):
        """Return the nearest live enemy already pressuring one of our towers."""
        own_towers = [
            tower for tower in self._towers.values()
            if tower.owner == self.actor_owner and tower.hitpoints > 0
        ]
        if not own_towers:
            return None

        best = None
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue
            if int(ent.get('card_id', -1)) <= 0:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            ex, ey = probe_to_world(ent['x'], ent['y'])
            nearest = min(
                own_towers,
                key=lambda tower: (
                    (tower.position[0] - ex) ** 2 +
                    (tower.position[1] - ey) ** 2
                ),
            )
            distance = math.sqrt(
                (nearest.position[0] - ex) ** 2 +
                (nearest.position[1] - ey) ** 2
            )
            if best is None or distance < best['distance']:
                best = {
                    'distance': distance,
                    'enemy_id': int(ent['id']),
                    'tower_id': int(nearest.entity_id),
                }

        if best is None or best['distance'] > ATTACK_HOLD_TOWER_DISTANCE:
            return None
        return best


    def _prelock_princess_towers(self, state):
        """Return living own princess towers from authoritative telemetry."""
        own_y = 6500.0 if self.actor_owner == 0 else 25500.0
        rows = []

        for ent in state.entities:
            if int(ent.get('owner', -1)) != self.actor_owner:
                continue
            if int(ent.get('card_id', 0)) != -1:
                continue

            hp = ent.get('hp')
            if not isinstance(hp, (int, float)) or float(hp) <= 0:
                continue

            x, y = probe_to_world(ent.get('x', -1), ent.get('y', -1))

            if y != own_y or x not in (3500.0, 14500.0):
                continue

            rows.append({
                'tower_id': int(ent['id']),
                'kind': (
                    'princess_left'
                    if x == 3500.0
                    else 'princess_right'
                ),
                'lane': 'left' if x < 9000.0 else 'right',
                'x': float(x),
                'y': float(y),
            })

        return tuple(rows)

    @staticmethod
    def _prelock_attack_range_world(spec):
        if spec is None:
            return None

        tiles = getattr(spec, 'range_tiles', None)

        if (
            isinstance(tiles, (int, float))
            and math.isfinite(float(tiles))
            and float(tiles) >= 0.0
        ):
            return float(tiles) * 1000.0

        return None

    @staticmethod
    def _prelock_deployment_remaining_ms(entity):
        raw = entity.get('deployment_runtime')

        if not isinstance(raw, dict):
            return 0.0

        if raw.get('validated') is not True:
            return 0.0

        remaining = raw.get('remaining_ms')

        if type(remaining) is int and remaining >= 0:
            return float(remaining)

        return 0.0

    def prelock_context(
            self, state, input_latency_ms, excluded_enemy_ids=()):
        """Find an enemy whose tower-lock window is about to close.

        The detector acts before first damage. It combines public board
        geometry, CardSpec attack range, measured velocity and validated
        native target acquisition. Unknown first-frame movement uses a short
        conservative envelope only when the new unit is already near its
        firing/lock radius.
        """
        self.observe(state)

        excluded = {
            int(v) for v in excluded_enemy_ids
            if isinstance(v, int) and int(v) > 0
        }

        towers = self._prelock_princess_towers(state)

        if not towers:
            self._prelock_context = None
            return None

        live = live_entities(state.entities)

        safety_ms = (
            max(0.0, float(input_latency_ms or 0.0))
            + float(config.PRELOCK_SAFETY_MARGIN_MS)
        )

        best = None

        for ent in live.values():
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue

            entity_id = ent.get('id')
            card_id = ent.get('card_id')

            if type(entity_id) is not int or entity_id <= 0:
                continue
            if entity_id in excluded:
                continue
            if type(card_id) is not int or card_id <= 0:
                continue

            hp = ent.get('hp')
            if isinstance(hp, (int, float)) and float(hp) <= 0:
                continue

            spec = self.bundle.card_specs.get(int(card_id))

            # Both troops and offensive buildings can create an immediate
            # princess-tower lock.  Mortar/X-Bow style buildings are
            # stationary, so their urgency comes from static attack range or
            # validated native target acquisition rather than closing speed.
            # Spells/effects still never enter this detector.
            if (
                spec is not None
                and spec.kind.value not in ('troop', 'building')
            ):
                continue

            ex, ey = probe_to_world(ent['x'], ent['y'])

            measured_attack = attack_state(
                ent, state.tick, live)

            targeted_tower = None

            if (
                measured_attack is not None
                and measured_attack.target_entity is not None
            ):
                targeted_tower = next(
                    (
                        tower for tower in towers
                        if tower['tower_id']
                        == int(measured_attack.target_entity)
                    ),
                    None,
                )

            nearest = targeted_tower or min(
                towers,
                key=lambda tower:
                    (
                        (tower['x'] - ex) ** 2
                        + (tower['y'] - ey) ** 2
                    ),
            )

            dx = nearest['x'] - ex
            dy = nearest['y'] - ey
            distance = math.hypot(dx, dy)

            attack_range = self._prelock_attack_range_world(spec)

            # Same small collision allowance already used by the forecasting
            # path when deciding whether a target is inside static range.
            lock_radius = (
                float(attack_range) + 750.0
                if attack_range is not None
                else 750.0
            )

            distance_to_lock = max(
                0.0, distance - lock_radius)

            deployment_ms = (
                self._prelock_deployment_remaining_ms(ent)
            )

            velocity = self._velocity.get(int(entity_id))
            closing_speed = 0.0

            if velocity is not None and distance > 1.0:
                vx, vy = velocity
                closing_speed = max(
                    0.0,
                    (
                        float(vx) * dx
                        + float(vy) * dy
                    ) / distance,
                )

            reason = None

            if targeted_tower is not None:
                # This is already later than our preferred pre-lock path, but
                # remains an authoritative critical fallback.
                lock_eta_ms = deployment_ms
                reason = 'tower_target_acquired'

            elif (
                attack_range is not None
                and distance_to_lock <= 0.0
            ):
                # Ranged bridge units can satisfy this on their first visible
                # frame, before attack-runtime exposes a target.
                lock_eta_ms = deployment_ms
                reason = 'tower_in_attack_range'

            elif (
                closing_speed
                >= float(config.PRELOCK_MIN_CLOSING_SPEED)
            ):
                travel_ticks = (
                    distance_to_lock / closing_speed
                )

                travel_ms = (
                    travel_ticks
                    * float(config.TICK_SECONDS)
                    * 1000.0
                )

                lock_eta_ms = max(
                    deployment_ms, travel_ms)

                reason = 'predicted_lock_eta'

            else:
                first_seen = int(
                    self._first_seen.get(
                        int(entity_id), int(state.tick))
                )

                age_ticks = max(
                    0, int(state.tick) - first_seen)

                if (
                    age_ticks
                    <= int(config.PRELOCK_NEW_ENTITY_TICKS)
                    and distance_to_lock
                    <= float(
                        config.PRELOCK_FALLBACK_DISTANCE_WORLD)
                ):
                    # First-frame bridge ranged threat: no reliable measured
                    # velocity exists yet, but it is already only a few tiles
                    # from its own lock envelope.
                    lock_eta_ms = (
                        safety_ms
                        + float(config.PRELOCK_URGENT_MS)
                        - 1.0
                    )

                    reason = 'new_close_to_lock_envelope'

                else:
                    continue

            latest_safe_ms = (
                float(lock_eta_ms) - safety_ms)

            if targeted_tower is not None:
                urgency = 'critical'

            elif (
                latest_safe_ms
                <= float(config.PRELOCK_CRITICAL_MS)
            ):
                urgency = 'critical'

            elif (
                latest_safe_ms
                <= float(config.PRELOCK_URGENT_MS)
            ):
                urgency = 'prelock_urgent'

            else:
                continue

            candidate = {
                'state': urgency,
                'reason': reason,
                'enemy_id': int(entity_id),
                'enemy_card_id': int(card_id),
                'tower_id': int(nearest['tower_id']),
                'tower_kind': nearest['kind'],
                'lane': nearest['lane'],
                'enemy_x': float(ex),
                'enemy_y': float(ey),
                'distance': float(distance),
                'distance_to_lock': float(distance_to_lock),
                'attack_range': (
                    float(attack_range)
                    if attack_range is not None
                    else None
                ),
                'closing_speed_per_tick': float(closing_speed),
                'lock_eta_ms': float(lock_eta_ms),
                'latest_safe_response_ms': float(
                    latest_safe_ms),
                'tick': int(state.tick),
            }

            if (
                best is None
                or candidate['latest_safe_response_ms']
                < best['latest_safe_response_ms']
            ):
                best = candidate

        self._prelock_context = best
        return best


    def _backfield_commitment_context(self):
        """Track visible 3+ elixir troops that were genuinely deployed deep.

        Unlike incoming_push, this is not a declaration that the card is a
        heavy win-condition core.  Its job is macro patience: remember that an
        opponent committed a meaningful troop in the backfield so we do not
        immediately mirror it in the other lane or waste a finite-lifetime
        building before the troop reaches an engagement window.
        """
        live_ids = set(self._live_entities)

        self._backfield_commitments = {
            eid: row
            for eid, row in self._backfield_commitments.items()
            if eid in live_ids
        }

        for eid, ent in self._live_entities.items():
            eid = int(eid)

            if int(ent.get('owner', -1)) == self.actor_owner:
                continue

            cid = int(ent.get('card_id', -1))
            spec = self.bundle.card_specs.get(cid)

            if spec is None or spec.kind.value != 'troop':
                continue

            cost = float(spec.elixir_cost)

            if cost < BACKFIELD_COMMITMENT_MIN_COST:
                continue

            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue

            depth = self._defensive_depth(ent['y'])

            if (
                eid not in self._backfield_commitments
                and depth >= BACKFIELD_COMMITMENT_START_DEPTH
            ):
                x = float(ent['x'])

                self._backfield_commitments[eid] = {
                    'card_id': cid,
                    'cost': cost,
                    'first_seen_tick': int(
                        self._first_seen.get(
                            eid, self._observed_tick)),
                    'initial_lane': (
                        'left' if x < 9000.0 else 'right'),
                }

        active = []
        finished = []

        for eid, tracked in self._backfield_commitments.items():
            ent = self._live_entities.get(eid)

            if ent is None:
                finished.append(eid)
                continue

            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                finished.append(eid)
                continue

            x = float(ent['x'])
            depth = self._defensive_depth(ent['y'])

            # At this point ordinary live-defense logic owns the unit.
            if depth <= BACKFIELD_COMMITMENT_RELEASE_DEPTH:
                finished.append(eid)
                continue

            active.append({
                **tracked,
                'entity_id': int(eid),
                'lane': 'left' if x < 9000.0 else 'right',
                'depth': float(depth),
            })

        for eid in finished:
            self._backfield_commitments.pop(eid, None)

        if not active:
            return None

        nearest = min(
            active,
            key=lambda row: (
                row['depth'],
                -row['cost'],
                row['entity_id'],
            ),
        )

        lanes = {row['lane'] for row in active}

        return {
            'lane': (
                next(iter(lanes))
                if len(lanes) == 1
                else None
            ),
            'entity_id': nearest['entity_id'],
            'card_id': nearest['card_id'],
            'cost': nearest['cost'],
            'depth': nearest['depth'],
            'count': len(active),
            'initial_lane': nearest['initial_lane'],
        }

    def _incoming_push_context(self):
        """Track a public heavy backfield deployment before it becomes pressure.

        Prefer an entity whose public source card is directly known. Some live
        runtime carriers expose only an archetype/form ID, though, so a second
        path binds a known exact opponent hand transition for a heavy troop to
        a newly appeared enemy body in the backfield. The latter never invents
        a hidden card: card identity/cost came from the native hand transition
        and the board body is used only to locate the push.
        """
        live_ids = set(self._live_entities)
        self._incoming_push_cores = {
            eid: row for eid, row in self._incoming_push_cores.items()
            if eid in live_ids
        }

        for eid, ent in self._live_entities.items():
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue
            cid = int(ent.get('card_id', -1))
            spec = self.bundle.card_specs.get(cid)
            if spec is None or spec.kind.value != 'troop':
                continue
            cost = float(spec.elixir_cost)
            if cost < INCOMING_PUSH_MIN_COST:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            depth = self._defensive_depth(ent['y'])
            if (int(eid) not in self._incoming_push_cores
                    and depth >= INCOMING_PUSH_BACKFIELD_DEPTH):
                x = float(ent['x'])
                self._incoming_push_cores[int(eid)] = {
                    'card_id': cid,
                    'cost': cost,
                    'first_seen_tick': int(self._first_seen.get(int(eid), self._observed_tick)),
                    'initial_lane': 'left' if x < 9000.0 else 'right',
                    'evidence': 'entity_card_spec',
                }

        # Runtime entities can carry archetype/form IDs that are intentionally
        # absent from CardSpec. Recover only when an exact recent opponent
        # heavy-troop play is already proven by a native hand transition.
        cat = self.bundle.entity_archetype_catalog
        recent_heavy = [
            row for row in self._recent_opponent_heavy_plays
            if 0 <= int(self._observed_tick) - int(row['tick'])
            <= INCOMING_PUSH_RECENT_HEAVY_TICKS
        ]
        for played in recent_heavy:
            bound = played.get('bound_entity_id')
            if bound in live_ids:
                continue
            candidates = []
            for eid, ent in self._live_entities.items():
                eid = int(eid)
                if eid in self._incoming_push_cores:
                    continue
                if int(ent.get('owner', -1)) == self.actor_owner:
                    continue
                hp = ent.get('hp')
                max_hp = ent.get('max_hp')
                if hp is not None and float(hp) <= 0:
                    continue
                if max_hp is not None and float(max_hp) <= 0:
                    continue
                depth = self._defensive_depth(ent['y'])
                if depth < INCOMING_PUSH_BACKFIELD_DEPTH:
                    continue
                first_seen = int(self._first_seen.get(eid, self._observed_tick))
                delta = abs(first_seen - int(played['tick']))
                if delta > INCOMING_PUSH_PLAY_BIND_TICKS:
                    continue
                cid = int(ent.get('card_id', -1))
                spec = self.bundle.card_specs.get(cid)
                # A supported source card proves this body belongs elsewhere;
                # never hijack it for a different exact play.
                if spec is not None:
                    continue
                gid = runtime_u32(ent.get('native_data_global_id')) or None
                vocab = cat.runtime_global_vocab_id(gid) if gid else 1
                metadata = cat.metadata_for_vocab_id(vocab) if vocab > 1 else None
                runtime_kind = (
                    metadata.child_kind
                    if metadata and metadata.child_kind_known else 'unknown'
                )
                if runtime_kind not in ('unknown', 'troop'):
                    continue
                candidates.append((
                    delta,
                    -float(max_hp or hp or 0.0),
                    eid,
                    ent,
                    first_seen,
                ))
            if not candidates:
                continue
            _, _, eid, ent, first_seen = min(candidates)
            x = float(ent['x'])
            self._incoming_push_cores[eid] = {
                'card_id': int(played['card_id']),
                'cost': float(played['cost']),
                'first_seen_tick': first_seen,
                'initial_lane': 'left' if x < 9000.0 else 'right',
                'evidence': 'exact_play_plus_new_backfield_entity',
            }
            played['bound_entity_id'] = eid

        active = []
        for eid, tracked in self._incoming_push_cores.items():
            ent = self._live_entities.get(eid)
            if ent is None:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            x = float(ent['x'])
            depth = self._defensive_depth(ent['y'])
            active.append({
                **tracked,
                'entity_id': int(eid),
                'lane': 'left' if x < 9000.0 else 'right',
                'depth': float(depth),
            })

        if not active:
            return None
        lanes = {row['lane'] for row in active}
        nearest = min(active, key=lambda row: (row['depth'], -row['cost']))
        lane = next(iter(lanes)) if len(lanes) == 1 else None
        return {
            'lane': lane,
            'core_entity_id': nearest['entity_id'],
            'core_card_id': nearest['card_id'],
            'core_cost': nearest['cost'],
            'core_depth': nearest['depth'],
            'core_count': len(active),
            'evidence': nearest.get('evidence', 'entity_card_spec'),
            'reserve_core_defenders': (
                nearest['depth'] > INCOMING_PUSH_DEFENDER_RELEASE_DEPTH
            ),
        }


    def _cannon_prebuild_context(self, incoming_push):
        """Decide whether Cannon may be established before final engagement.

        A finite-lifetime building should not be dropped immediately after a
        tank is played in the far back.  It also should not wait until the tank
        is already inside pull range.

        Use measured closing velocity when available.  Before a second
        position sample exists, a conservative depth fallback opens the
        prebuild window.
        """
        if (
            incoming_push is None
            or incoming_push.get('lane') is None
        ):
            return None

        entity_id = incoming_push.get(
            'core_entity_id')

        ent = self._live_entities.get(
            entity_id)

        if ent is None:
            return None

        depth = float(
            incoming_push.get(
                'core_depth',
                self._defensive_depth(ent['y']),
            )
        )

        if depth <= HEAVY_DEFEND_CANNON_RELEASE_DEPTH:
            return {
                'allowed': True,
                'urgent': True,
                'reason': 'anchor_window',
                'lead_ticks': 0.0,
                'depth': depth,
                'lane': incoming_push['lane'],
            }

        velocity = self._velocity.get(
            int(entity_id))

        closing_per_tick = 0.0

        if velocity is not None:
            _, vy = velocity

            # Owner 0 defends low Y, so an approaching enemy has negative vy.
            # Owner 1 defends high Y, so an approaching enemy has positive vy.
            closing_per_tick = (
                max(0.0, -float(vy))
                if self.actor_owner == 0
                else max(0.0, float(vy))
            )

        velocity_outlier = (
            closing_per_tick
            > CANNON_PREBUILD_MAX_REASONABLE_SPEED_PER_TICK
        )

        if velocity_outlier:
            # Do not turn a stale-frame/telemetry jump into an emergency
            # prebuild. Fall back to the conservative depth gate below.
            closing_per_tick = 0.0

        if closing_per_tick > 1.0:
            remaining = max(
                0.0,
                depth
                - HEAVY_DEFEND_CANNON_RELEASE_DEPTH,
            )

            lead_ticks = (
                remaining / closing_per_tick
            )

            if (
                lead_ticks
                <= CANNON_PREBUILD_MAX_LEAD_TICKS
            ):
                return {
                    'allowed': True,
                    'urgent': (
                        lead_ticks
                        <= CANNON_PREBUILD_URGENT_LEAD_TICKS
                    ),
                    'reason': 'predicted_anchor_eta',
                    'lead_ticks': float(lead_ticks),
                    'depth': depth,
                    'lane': incoming_push['lane'],
                }

            return {
                'allowed': False,
                'urgent': False,
                'reason': 'tank_too_far',
                'lead_ticks': float(lead_ticks),
                'depth': depth,
                'lane': incoming_push['lane'],
            }

        allowed = (
            depth <= CANNON_PREBUILD_FALLBACK_DEPTH
        )

        return {
            'allowed': bool(allowed),
            'urgent': False,
            'reason': (
                'velocity_outlier_depth_fallback'
                if velocity_outlier and allowed
                else 'velocity_outlier_wait'
                if velocity_outlier
                else 'depth_fallback'
                if allowed
                else 'waiting_for_motion'
            ),
            'lead_ticks': None,
            'depth': depth,
            'lane': incoming_push['lane'],
        }

    def _attack_hold_context(self, effective_elixir, pressure=None):
        """Conservatively hold Hog when a live enemy is already near our tower.

        This is a macro safety gate, not an attack recommender.  It only
        suppresses the win condition when committing four elixir would leave
        too little budget to answer an immediate near-tower threat.
        """
        if float(effective_elixir) >= ATTACK_HOLD_MIN_EFFECTIVE_ELIXIR:
            return None
        pressure = pressure if pressure is not None else self._near_tower_pressure()
        if pressure is None:
            return None
        return {
            'reason': 'near_tower_defense',
            'distance': pressure['distance'],
            'enemy_id': pressure['enemy_id'],
            'tower_id': pressure['tower_id'],
            'required_effective_elixir': ATTACK_HOLD_MIN_EFFECTIVE_ELIXIR,
        }

    def _recent_building_attack_window(self, tick, defensive_pressure=False):
        """Return a short public-information Hog window after a building dies.

        This is deliberately not a hidden-hand/cycle oracle.  It only uses a
        defensive building that was visibly alive and has just disappeared.
        The window is short enough to be useful before an ordinary cycle is
        likely to return the same answer.
        """
        if defensive_pressure:
            return None
        row = self._recent_enemy_building_expiry
        if not row:
            return None
        age = int(tick) - int(row['tick'])
        if age < 0 or age > RECENT_BUILDING_ATTACK_WINDOW_TICKS:
            return None
        if int(row['card_id']) in self._active_enemy_buildings.values():
            return None
        return {
            'reason': 'recent_enemy_building_expired',
            'card_id': int(row['card_id']),
            'entity_id': int(row['entity_id']),
            'age_ticks': age,
            'remaining_ticks': RECENT_BUILDING_ATTACK_WINDOW_TICKS - age,
        }

    def _exact_building_cycle_window(self, defensive_pressure=False):
        """Return a proven building-out-of-cycle window from exact play receipts.

        Clash Royale returns a played card after four subsequent card plays.
        We only use native hand-transition receipts already accepted by the
        public tracker. If those receipts are unavailable, this function stays
        silent rather than inferring hidden hand state from spawned units.
        """
        if defensive_pressure or not self._recent_enemy_building_expiry:
            return None
        cid = int(self._recent_enemy_building_expiry['card_id'])
        last_play = self._opponent_last_play_count.get(cid)
        if last_play is None:
            return None
        plays_since = self._opponent_exact_play_count - int(last_play)
        if plays_since < 0 or plays_since >= 4:
            return None
        if cid in self._active_enemy_buildings.values():
            return None
        return {
            'reason': 'enemy_building_out_of_cycle',
            'card_id': cid,
            'plays_since': plays_since,
            'plays_until_return': 4 - plays_since,
        }

    @staticmethod
    def _low_elixir_attack_window(effective_elixir, opponent_elixir_bounds,
                                  defensive_pressure=False):
        """Use only a tight FAIR tracker bound as a low-elixir punish signal."""
        if defensive_pressure or float(effective_elixir) < 4.0:
            return None
        if opponent_elixir_bounds is None:
            return None
        low, high = map(float, opponent_elixir_bounds)
        width = max(0.0, high - low)
        if high > LOW_ELIXIR_ATTACK_MAX_UPPER or width > LOW_ELIXIR_ATTACK_MAX_WIDTH:
            return None
        return {
            'reason': 'opponent_low_elixir',
            'lower': low,
            'upper': high,
            'width': width,
        }

    @staticmethod
    def _attack_opportunity_context(effective_elixir, *, defensive_pressure=False,
                                    near_tower_pressure=None, incoming_push=None,
                                    counterpush=None, building_window=None,
                                    low_elixir_window=None):
        """Collapse offensive timing signals into one ordered public context.

        The context is advisory and deliberately conservative: defense always
        wins, counterpush support wins over generic timing windows, exact/recent
        building windows win over low-elixir pressure, and anti-overflow is the
        lowest-priority release.  This keeps the mask from applying several
        independent "go now" rules to the same frame.
        """
        if defensive_pressure or near_tower_pressure is not None:
            return None
        if (incoming_push is not None
                and float(effective_elixir) >= INCOMING_PUSH_PUNISH_MIN_ELIXIR):
            incoming_lane = incoming_push.get('lane')
            punish_lane = (
                'right' if incoming_lane == 'left' else
                'left' if incoming_lane == 'right' else
                None
            )
            return {
                'kind': 'heavy_commit',
                'reason': 'opponent_backfield_heavy_commit',
                'release_hog': True,
                'lane': punish_lane,
                'incoming_push_lane': incoming_lane,
                'card_id': incoming_push.get('core_card_id'),
                'cost': incoming_push.get('core_cost'),
            }
        if counterpush is not None:
            return {
                'kind': 'counterpush',
                'reason': 'counterpush_support',
                'release_hog': True,
                'lane': counterpush.get('lane'),
                'support_entity_id': counterpush.get('support_entity_id'),
                'support_card_id': counterpush.get('support_card_id'),
            }
        if building_window is not None:
            return {
                'kind': 'building_window',
                'reason': building_window['reason'],
                'release_hog': True,
                'lane': None,
                'card_id': building_window.get('card_id'),
                'age_ticks': building_window.get('age_ticks'),
                'plays_since': building_window.get('plays_since'),
                'plays_until_return': building_window.get('plays_until_return'),
            }
        if low_elixir_window is not None:
            return {
                'kind': 'low_elixir',
                'reason': low_elixir_window['reason'],
                'release_hog': True,
                'lane': None,
                'opponent_elixir_lower': low_elixir_window.get('lower'),
                'opponent_elixir_upper': low_elixir_window.get('upper'),
            }
        if float(effective_elixir) >= NEUTRAL_PATIENCE_RELEASE_ELIXIR:
            return {
                'kind': 'anti_overflow',
                'reason': 'near_elixir_cap',
                'release_hog': True,
                'lane': None,
            }
        return None

    def _counterpush_context(self, pressure=None):
        """Detect one unambiguous surviving support lane after defense.

        This does not force Hog.  It only gives Hog's placement mask the lane
        of a surviving Musketeer/Ice Golem that is already advancing toward
        the bridge after near-tower pressure has cleared.
        """
        if pressure is not None:
            return None
        supports = []
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) != self.actor_owner:
                continue
            card_id = int(ent.get('card_id', -1))
            if card_id not in COUNTERPUSH_SUPPORT_CARDS:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            x, y = probe_to_world(ent['x'], ent['y'])
            progress = y if self.actor_owner == 0 else 32000.0 - y
            if not (COUNTERPUSH_MIN_PROGRESS <= progress <= COUNTERPUSH_MAX_PROGRESS):
                continue
            lane = 'left' if x < 9000.0 else 'right'
            max_hp = float(ent.get('max_hp') or 0.0)
            hp_fraction = (float(hp) / max_hp) if hp is not None and max_hp > 0 else 1.0
            supports.append({
                'lane': lane,
                'entity_id': int(ent['id']),
                'card_id': card_id,
                'progress': progress,
                'hp_fraction': hp_fraction,
            })

        if not supports:
            return None
        lanes = {row['lane'] for row in supports}
        if len(lanes) != 1:
            return None
        best = max(supports, key=lambda row: (row['progress'], row['hp_fraction']))
        return {
            'lane': best['lane'],
            'support_entity_id': best['entity_id'],
            'support_card_id': best['card_id'],
            'support_progress': best['progress'],
            'support_hp_fraction': best['hp_fraction'],
        }

    def _defensive_depth(self, y):
        """Distance from our native baseline toward the enemy baseline.

        Owner 0 defends the low-Y side; owner 1 defends the high-Y side.
        Using min(y, 32000-y) here is wrong because it treats units near the
        opponent's own baseline as pressure on us.
        """
        y = float(y)
        return y if self.actor_owner == 0 else 32000.0 - y

    def _has_defensive_pressure(self):
        """Whether any live enemy is already in our central/defensive half.

        Neutral patience must never suppress a real defensive response just
        because threats are split across lanes or are not yet inside the
        stricter near-tower radius.
        """
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue
            if int(ent.get('card_id', -1)) <= 0:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            if self._defensive_depth(ent['y']) <= 14500.0:
                return True
        return False

    def _low_value_defensive_threat_context(
            self, incoming_push=None, backfield_commitment=None):
        """Return a lone one-elixir troop that should not consume core defense.

        This gate is intentionally narrow.  Any tracked heavy push, meaningful
        backfield commitment, second live enemy, unknown card identity, or
        higher-cost threat disables it and restores the normal defense policy.
        """
        if incoming_push is not None or backfield_commitment is not None:
            return None

        enemy = []
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue

            card_id = int(ent.get('card_id', -1))
            if card_id <= 0:
                continue

            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue

            enemy.append(ent)

        if len(enemy) != 1:
            return None

        ent = enemy[0]
        depth = self._defensive_depth(ent['y'])
        if depth > 14500.0:
            return None

        card_id = int(ent.get('card_id', -1))
        spec = self.bundle.card_specs.get(card_id)
        if spec is None or spec.elixir_cost is None:
            return None

        cost = float(spec.elixir_cost)
        if cost > LOW_VALUE_DEFENSE_MAX_COST:
            return None

        return {
            'entity_id': int(ent['id']),
            'card_id': card_id,
            'cost': cost,
            'depth': float(depth),
            'lane': 'left' if float(ent['x']) < 9000.0 else 'right',
        }

    def _single_defensive_threat_lane(self):
        """Return one unambiguous enemy lane inside our defensive half."""
        enemy = []
        for ent in self._live_entities.values():
            if int(ent.get('owner', -1)) == self.actor_owner or int(ent.get('card_id', -1)) <= 0:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            x, y = float(ent['x']), float(ent['y'])
            # Match the strategy phase and execution validator in actor-relative
            # coordinates: only enemies on OUR half may constrain defense.
            if self._defensive_depth(y) > 14500.0:
                continue
            lane = 'left' if x < 9000.0 else 'right'
            enemy.append((lane, x, y))
        if not enemy:
            return None
        threat_lanes = {lane for lane, _, _ in enemy}
        if len(threat_lanes) != 1:
            return None
        return {'threat_lane': next(iter(threat_lanes)),
                'enemy_count': len(enemy)}

    @staticmethod
    def _mask_to_lane(entry, lane, metadata_key):
        left = lane == 'left'
        rows = tuple(tuple(
            bool(allowed) and ((x < 9) if left else (x >= 9))
            for x, allowed in enumerate(row)
        ) for row in entry['row_major'])
        masked = dict(entry)
        masked['row_major'] = rows
        masked[metadata_key] = lane
        return masked

    @classmethod
    def _mask_to_defensive_lane(cls, entry, threat_lane):
        """Remove cells the live validator would reject for wrong-lane defense."""
        return cls._mask_to_lane(entry, threat_lane, 'defensive_threat_lane')

    def _mask_to_defensive_depth_band(self, entry, *, min_depth=0.0,
                                      max_depth=INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH,
                                      metadata_prefix='defense'):
        """Limit a placement mask to an actor-relative defensive depth band."""
        masked = dict(entry)
        subcell = masked.get('model_subcell_offset') or (0.0, 0.0)
        dy = float(subcell[1] or 0.0)
        sign = 1.0 if self.actor_owner == 0 else -1.0
        rows = []
        for y, row in enumerate(masked['row_major']):
            world_y = (float(y) + 0.5 + sign * dy) * 1000.0
            depth = self._defensive_depth(world_y)
            in_band = float(min_depth) <= depth <= float(max_depth)
            rows.append(tuple(bool(allowed) and in_band for allowed in row))
        masked['row_major'] = tuple(rows)
        masked[f'{metadata_prefix}_min_defensive_depth'] = float(min_depth)
        masked[f'{metadata_prefix}_max_defensive_depth'] = float(max_depth)
        return masked

    def _mask_to_incoming_defense(self, entry, lane):
        """Keep preparation plays on the threatened lane and our safe half."""
        masked = self._mask_to_lane(entry, lane, 'incoming_push_lane')
        masked = self._mask_to_defensive_depth_band(
            masked,
            max_depth=INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH,
            metadata_prefix='incoming_push',
        )
        # Backward-compatible diagnostic used by the existing focused tests.
        masked['incoming_push_max_defensive_depth'] = (
            INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH
        )
        return masked

    def _mask_to_heavy_defense_role(self, entry, card_id, lane):
        """Stage defenders for a tracked heavy push."""
        masked = self._mask_to_lane(entry, lane, 'heavy_defense_lane')
        cid = int(card_id)
        if cid == MUSKETEER:
            return self._mask_to_defensive_depth_band(
                masked,
                max_depth=HEAVY_DEFEND_MUSKETEER_MAX_DEPTH,
                metadata_prefix='heavy_defense_musketeer',
            )
        if cid == CANNON:
            return self._mask_to_defensive_depth_band(
                masked,
                min_depth=HEAVY_DEFEND_CANNON_MIN_DEPTH,
                max_depth=HEAVY_DEFEND_CANNON_MAX_DEPTH,
                metadata_prefix='heavy_defense_cannon',
            )
        if cid == ICE_GOLEM:
            return self._mask_to_defensive_depth_band(
                masked,
                min_depth=HEAVY_DEFEND_ICE_GOLEM_MIN_DEPTH,
                max_depth=HEAVY_DEFEND_ICE_GOLEM_MAX_DEPTH,
                metadata_prefix='heavy_defense_ice_golem',
            )
        if cid in HEAVY_DEFEND_CHEAP_CONTROL:
            return self._mask_to_defensive_depth_band(
                masked,
                min_depth=HEAVY_DEFEND_CHEAP_MIN_DEPTH,
                max_depth=HEAVY_DEFEND_CHEAP_MAX_DEPTH,
                metadata_prefix='heavy_defense_cheap',
            )
        return masked

    @staticmethod
    def _heavy_defense_priority(slots, core_depth):
        """Choose at most one key body/anchor commitment for this decision.

        Heavy defense should build a shape, not dump Cannon, Musketeer and Ice
        Golem on consecutive stale frames.  The preferred order changes with
        approach depth: set the backline first, then anchor the tank, then add
        the body.  At emergency depth all key cards are released because a
        strict sequence is less important than immediate survival.
        """
        if core_depth is None:
            return {'card_id': None, 'stage': None, 'emergency': False}
        depth = float(core_depth)
        if depth <= HEAVY_DEFEND_EMERGENCY_DEPTH:
            return {
                'card_id': None,
                'stage': 'emergency_release',
                'emergency': True,
            }
        cards = set(int(cid) for cid in slots.values())
        if depth > HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH:
            return {
                'card_id': None,
                'stage': 'wait_for_approach',
                'emergency': False,
            }
        if depth > HEAVY_DEFEND_BODY_RELEASE_DEPTH:
            order = (MUSKETEER,)
            stage = 'backline_setup'
        elif depth > HEAVY_DEFEND_CANNON_RELEASE_DEPTH:
            order = (MUSKETEER, ICE_GOLEM)
            stage = 'backline_then_body'
        else:
            order = (CANNON, MUSKETEER, ICE_GOLEM)
            stage = 'anchor_then_backline'
        return {
            'card_id': next((cid for cid in order if cid in cards), None),
            'stage': stage,
            'emergency': False,
        }

    def _heavy_defense_fireball_context(self, incoming_push):
        """Return visible support value near a tracked heavy core.

        Fireball stays reserved against a lone tank. It becomes available when
        a visible same-lane support package is close enough that a defensive
        Fireball can hit the push rather than being spent elsewhere.
        """
        if incoming_push is None:
            return None
        core_id = incoming_push.get('core_entity_id')
        lane = incoming_push.get('lane')
        core = self._live_entities.get(core_id)
        if core is None or lane is None:
            return None
        cx, cy = float(core['x']), float(core['y'])
        support = []
        support_cost = 0.0
        for eid, ent in self._live_entities.items():
            if int(eid) == int(core_id):
                continue
            if int(ent.get('owner', -1)) == self.actor_owner:
                continue
            hp = ent.get('hp')
            if hp is not None and float(hp) <= 0:
                continue
            x, y = float(ent['x']), float(ent['y'])
            ent_lane = 'left' if x < 9000.0 else 'right'
            if ent_lane != lane:
                continue
            if math.hypot(x - cx, y - cy) > HEAVY_DEFEND_FIREBALL_CLUSTER_RADIUS:
                continue
            spec = self.bundle.card_specs.get(int(ent.get('card_id', -1)))
            cost = float(spec.elixir_cost) if spec is not None else 0.0
            if cost <= 0.0:
                continue
            support.append({
                'entity_id': int(eid),
                'card_id': int(ent.get('card_id', -1)),
                'cost': cost,
                'x': x,
                'y': y,
            })
            support_cost += cost
        if support_cost < HEAVY_DEFEND_FIREBALL_SUPPORT_MIN_COST:
            return {
                'release': False,
                'support_count': len(support),
                'support_cost': support_cost,
                'core_x': cx,
                'core_y': cy,
            }
        return {
            'release': True,
            'support_count': len(support),
            'support_cost': support_cost,
            'core_x': cx,
            'core_y': cy,
        }

    @staticmethod
    def _mask_spell_near_point(entry, world_x, world_y, radius, metadata_prefix):
        masked = dict(entry)
        subcell = masked.get('model_subcell_offset') or (0.0, 0.0)
        dx, dy = float(subcell[0] or 0.0), float(subcell[1] or 0.0)
        rows = []
        for y, row in enumerate(masked['row_major']):
            out = []
            for x, allowed in enumerate(row):
                gx = (float(x) + 0.5 + dx) * 1000.0
                gy = (float(y) + 0.5 + dy) * 1000.0
                out.append(
                    bool(allowed)
                    and math.hypot(gx - float(world_x), gy - float(world_y))
                    <= float(radius)
                )
            rows.append(tuple(out))
        masked['row_major'] = tuple(rows)
        masked[f'{metadata_prefix}_target_x'] = float(world_x)
        masked[f'{metadata_prefix}_target_y'] = float(world_y)
        masked[f'{metadata_prefix}_target_radius'] = float(radius)
        return masked


    def neutral_overflow_fallback(self, state, observation):
        """Spend a safe neutral card when a near-cap model WAIT would leak elixir.

        This only runs after the policy itself chose WAIT. It deliberately
        excludes Cannon, Musketeer, and Fireball so anti-overflow cannot burn
        the defensive package just to avoid sitting at ten elixir.
        """
        mask = observation.action_mask
        reasons = mask.reasons

        if not reasons.get('neutral_overflow_active'):
            return None

        priority = {
            SKELETONS: 0,
            ICE_SPIRIT: 1,
            THE_LOG: 2,
            ICE_GOLEM: 3,
            HOG_RIDER: 4,
        }
        candidates = []

        for slot, cid in enumerate(state.hand_cards):
            cid = int(cid)

            if (
                cid not in priority
                or slot >= len(mask.hand_slots)
                or not mask.hand_slots[slot]
            ):
                continue

            entry = mask.placement_masks.get(str(slot))

            if not isinstance(entry, Mapping):
                continue

            cost = entry.get('effective_cost')

            if not isinstance(cost, (int, float)):
                continue

            candidates.append((
                priority[cid],
                slot,
                cid,
                float(cost),
                entry,
            ))

        if not candidates:
            return None

        _, slot, cid, cost, entry = min(candidates)

        enemy_princess = [
            tower
            for tower in observation.towers
            if (
                tower.owner != state.local_owner
                and tower.tower_kind.startswith('princess_')
                and tower.hitpoints > 0
            )
        ]
        weakest_tower = (
            min(
                enemy_princess,
                key=lambda tower: (
                    tower.hitpoints
                    / max(1.0, tower.max_hitpoints),
                    tower.hitpoints,
                    tower.tower_kind,
                ),
            )
            if enemy_princess
            else None
        )
        attack_x = (
            float(weakest_tower.position[0])
            if weakest_tower is not None
            else 3500.0
        )

        if cid == HOG_RIDER:
            preferred_x = attack_x
            preferred_depth = 14500.0
            mode = 'hog_pressure'
        elif cid == THE_LOG:
            preferred_x = attack_x
            preferred_depth = 15000.0
            mode = 'cheap_cycle'
        else:
            preferred_x = 9000.0
            preferred_depth = (
                5000.0
                if cid == SKELETONS
                else 6000.0
                if cid == ICE_SPIRIT
                else 7000.0
            )
            mode = 'cheap_cycle'

        subcell = (
            entry.get('model_subcell_offset')
            or (0.0, 0.0)
        )
        dx = float(subcell[0] or 0.0)
        dy = float(subcell[1] or 0.0)
        sign = 1.0 if self.actor_owner == 0 else -1.0
        legal = []

        for gy, row in enumerate(entry['row_major']):
            for gx, allowed in enumerate(row):
                if not allowed:
                    continue

                wx = (
                    gx + 0.5 + sign * dx
                ) * 1000.0
                wy = (
                    gy + 0.5 + sign * dy
                ) * 1000.0
                depth = self._defensive_depth(wy)
                score = (
                    (wx - preferred_x) ** 2
                    + 0.4
                    * (depth - preferred_depth) ** 2
                )
                legal.append((score, gx, gy))

        if not legal:
            return None

        _, gx, gy = min(legal)

        return ActionV1(
            owner=state.local_owner,
            kind=ActionKind.PLAY_CARD,
            hand_slot=slot,
            card_id=cid,
            target_kind=TargetKind.GRID,
            target_grid=(gx, gy),
            subcell_offset=(
                sign * dx,
                sign * dy,
            ),
            execute_offset_ticks=1,
            next_decision_ticks=config.DECISION_TICKS,
            metadata={
                'policy_effective_cost': cost,
                'policy_effective_form_code': int(
                    entry.get('form_code', 0) or 0),
                'neutral_overflow_fallback': True,
                'neutral_overflow_mode': mode,
            },
        )


    def defense_overflow_fallback(self, state, observation):
        """Choose one already-approved formation action after a capped WAIT."""
        mask = observation.action_mask
        reasons = mask.reasons

        if not reasons.get(
                'defense_overflow_forced'):
            return None

        safe_slots = list(
            reasons.get(
                'defense_overflow_safe_slots')
            or ()
        )

        if not safe_slots:
            return None

        mode = reasons.get(
            'defense_overflow_mode')

        priority = {}

        # Formation ordering mirrors the macro intent.
        if mode == 'backline_setup':
            priority = {
                MUSKETEER: 0,
            }

        elif mode == 'cannon_prebuild_urgent':
            priority = {
                CANNON: 0,
                ICE_SPIRIT: 1,
                SKELETONS: 2,
            }

        elif mode == 'cycle_then_prebuild':
            priority = {
                ICE_SPIRIT: 0,
                SKELETONS: 1,
                CANNON: 2,
            }

        elif mode == 'cannon_prebuild':
            priority = {
                CANNON: 0,
            }

        elif mode == 'low_value_defense_cycle':
            priority = {
                SKELETONS: 0,
                ICE_SPIRIT: 1,
                THE_LOG: 2,
            }

        else:
            priority = {
                MUSKETEER: 0,
                CANNON: 1,
                ICE_GOLEM: 2,
                ICE_SPIRIT: 3,
                SKELETONS: 4,
            }

        candidates = []

        for raw_slot in safe_slots:
            slot = int(raw_slot)

            if not (
                0 <= slot < len(state.hand_cards)
                and slot < len(mask.hand_slots)
                and mask.hand_slots[slot]
            ):
                continue

            cid = int(
                state.hand_cards[slot])

            entry = mask.placement_masks.get(
                str(slot))

            if not isinstance(entry, Mapping):
                continue

            cost = entry.get(
                'effective_cost')

            if not isinstance(cost, (int, float)):
                continue

            candidates.append((
                priority.get(cid, 99),
                slot,
                cid,
                float(cost),
                entry,
            ))

        if not candidates:
            return None

        _, slot, cid, cost, entry = min(
            candidates)

        lane = (
            reasons.get('incoming_push_lane')
            or reasons.get('backfield_commitment_lane')
            or reasons.get('defensive_threat_lane')
            or reasons.get('prelock_lane')
        )

        preferred_x = (
            3500.0
            if lane == 'left'
            else 14500.0
            if lane == 'right'
            else 9000.0
        )

        preferred_depth = (
            5500.0
            if cid == MUSKETEER
            else 8000.0
            if cid == CANNON
            else 9000.0
        )

        subcell = (
            entry.get('model_subcell_offset')
            or (0.0, 0.0)
        )

        dx = float(
            subcell[0] or 0.0)
        dy = float(
            subcell[1] or 0.0)

        sign = (
            1.0
            if self.actor_owner == 0
            else -1.0
        )

        legal = []

        for gy, row in enumerate(
                entry['row_major']):
            for gx, allowed in enumerate(row):
                if not allowed:
                    continue

                wx = (
                    gx + 0.5 + sign * dx
                ) * 1000.0

                wy = (
                    gy + 0.5 + sign * dy
                ) * 1000.0

                depth = self._defensive_depth(
                    wy)

                score = (
                    (wx - preferred_x) ** 2
                    + 0.4
                    * (depth - preferred_depth) ** 2
                )

                legal.append(
                    (score, gx, gy))

        if not legal:
            return None

        _, gx, gy = min(legal)

        return ActionV1(
            owner=state.local_owner,
            kind=ActionKind.PLAY_CARD,
            hand_slot=slot,
            card_id=cid,
            target_kind=TargetKind.GRID,
            target_grid=(gx, gy),
            subcell_offset=(
                sign * dx,
                sign * dy,
            ),
            execute_offset_ticks=1,
            next_decision_ticks=config.DECISION_TICKS,
            metadata={
                'policy_effective_cost': cost,
                'policy_effective_form_code': int(
                    entry.get('form_code', 0) or 0),
                'defense_overflow_fallback': True,
                'defense_overflow_mode': mode,
            },
        )

    def defensive_lane_conflict(self, action, state):
        """Detect an obvious cross-lane defensive placement mistake.

        This is intentionally conservative: it only fires for core defensive
        cards when one lane has a nearby enemy and the selected lane has none.
        It is not a strategy override for attacks or split-lane situations.
        """
        if int(action.card_id or 0) not in DEFENSIVE_LANE_CARDS or action.target_grid is None:
            return None
        gate = self._single_defensive_threat_lane()
        if gate is None:
            return None
        target_x, _ = action_world(action)
        target_lane = 'left' if target_x < 9000.0 else 'right'
        if target_lane == gate['threat_lane']:
            return None
        return {'target_lane': target_lane,
                'threat_lane': gate['threat_lane'],
                'enemy_count': gate['enemy_count']}

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
        defensive_lane_gate = self._single_defensive_threat_lane()
        self.quality['defensive_threat_lane'] = (
            defensive_lane_gate['threat_lane'] if defensive_lane_gate else None)
        prelock = (
            self._prelock_context
            if self._prelock_context is not None
            and int(self._prelock_context.get('tick', -1))
                == int(state.tick)
            else None
        )
        prelock_lane = (
            prelock.get('lane') if prelock else None
        )
        self.quality['prelock_state'] = (
            prelock.get('state') if prelock else None)
        self.quality['prelock_reason'] = (
            prelock.get('reason') if prelock else None)
        self.quality['prelock_enemy_id'] = (
            prelock.get('enemy_id') if prelock else None)
        self.quality['prelock_enemy_card_id'] = (
            prelock.get('enemy_card_id') if prelock else None)
        self.quality['prelock_tower_id'] = (
            prelock.get('tower_id') if prelock else None)
        self.quality['prelock_latest_safe_response_ms'] = (
            prelock.get('latest_safe_response_ms')
            if prelock else None
        )
        opponent_elixir_bounds = None
        tracker = getattr(self.tensorizer, 'tracker', None)
        if tracker is not None and hasattr(tracker, 'elixir_bounds'):
            try:
                low, high = tracker.elixir_bounds[1 - self.actor_owner]
                opponent_elixir_bounds = (float(low), float(high))
            except (IndexError, KeyError, TypeError, ValueError):
                opponent_elixir_bounds = None
        self.quality['opponent_elixir_lower'] = (
            opponent_elixir_bounds[0] if opponent_elixir_bounds else None)
        self.quality['opponent_elixir_upper'] = (
            opponent_elixir_bounds[1] if opponent_elixir_bounds else None)
        near_tower_pressure = self._near_tower_pressure()
        defensive_pressure = (
            self._has_defensive_pressure() or prelock is not None
        )
        incoming_push = self._incoming_push_context()
        backfield_commitment = self._backfield_commitment_context()
        low_value_defensive_threat = (
            self._low_value_defensive_threat_context(
                incoming_push=incoming_push,
                backfield_commitment=backfield_commitment,
            )
        )

        preparing_for_push = (
            incoming_push is not None
            and not defensive_pressure
            and near_tower_pressure is None
        )

        backfield_patience = (
            backfield_commitment is not None
            and not defensive_pressure
            and near_tower_pressure is None
        )

        backfield_depth = (
            float(backfield_commitment.get('depth'))
            if backfield_commitment is not None
            and backfield_commitment.get('depth') is not None
            else None
        )

        backfield_lane = (
            backfield_commitment.get('lane')
            if backfield_commitment is not None
            else None
        )

        attack_hold = self._attack_hold_context(
            elixir, near_tower_pressure)

        counterpush = self._counterpush_context(
            True
            if (
                defensive_pressure
                or incoming_push is not None
                or backfield_patience
            )
            else near_tower_pressure
        )

        preparation_pressure = (
            defensive_pressure
            or preparing_for_push
            or backfield_patience
        )

        exact_building_cycle_window = (
            self._exact_building_cycle_window(
                defensive_pressure=preparation_pressure)
        )

        building_attack_window = (
            exact_building_cycle_window
            or self._recent_building_attack_window(
                state.tick,
                defensive_pressure=preparation_pressure,
            )
        )

        low_elixir_attack_window = self._low_elixir_attack_window(
            elixir,
            opponent_elixir_bounds,
            defensive_pressure=preparation_pressure,
        )

        # A generic 3/4-elixir backfield troop is not enough evidence to
        # manufacture an attack window.  A proved 5+ elixir incoming push
        # keeps the existing opposite-lane Hog punish logic.
        opportunity_defensive_pressure = (
            defensive_pressure
            or (
                backfield_patience
                and incoming_push is None
            )
        )

        attack_opportunity = self._attack_opportunity_context(
            elixir,
            defensive_pressure=opportunity_defensive_pressure,
            near_tower_pressure=near_tower_pressure,
            incoming_push=incoming_push,
            counterpush=counterpush,
            building_window=building_attack_window,
            low_elixir_window=low_elixir_attack_window,
        )

        strategy_phase = (
            'defend'
            if defensive_pressure or near_tower_pressure is not None
            else 'prepare_defense'
            if incoming_push is not None or backfield_patience
            else 'counterpush'
            if (
                attack_opportunity is not None
                and attack_opportunity.get('kind') == 'counterpush'
            )
            else 'neutral'
        )

        defense_overflow_active = (
            float(elixir) >= DEFENSE_OVERFLOW_ELIXIR
            and strategy_phase in ('defend', 'prepare_defense')
        )
        neutral_overflow_active = (
            float(elixir) >= DEFENSE_OVERFLOW_ELIXIR
            and strategy_phase == 'neutral'
        )

        defense_overflow_hard_cap = (
            defense_overflow_active
            and float(elixir)
                >= DEFENSE_OVERFLOW_HARD_CAP_ELIXIR
        )
        defending_incoming_push = (
            strategy_phase == 'defend'
            and incoming_push is not None
            and incoming_push.get('lane') is not None
        )
        heavy_core_depth = (
            float(incoming_push.get('core_depth'))
            if incoming_push is not None
            and incoming_push.get('core_depth') is not None
            else None
        )
        cannon_prebuild = (
            self._cannon_prebuild_context(incoming_push)
            if preparing_for_push
            else None
        )
        cannon_prebuild_allowed = bool(
            cannon_prebuild
            and cannon_prebuild.get('allowed')
        )
        cannon_prebuild_urgent = bool(
            cannon_prebuild
            and cannon_prebuild.get('urgent')
        )
        heavy_fireball = (
            self._heavy_defense_fireball_context(incoming_push)
            if defending_incoming_push else None
        )
        heavy_priority = (
            self._heavy_defense_priority(slots, heavy_core_depth)
            if defending_incoming_push
            else {'card_id': None, 'stage': None, 'emergency': False}
        )
        heavy_priority_card = heavy_priority.get('card_id')
        neutral_patience = (
            strategy_phase == 'neutral'
            and float(elixir) < NEUTRAL_PATIENCE_RELEASE_ELIXIR
        )
        # Backward-compatible attack-window diagnostics remain limited to the
        # building/low-elixir signals; the unified context also records
        # counterpush and anti-overflow opportunities.
        attack_window = (
            building_attack_window or low_elixir_attack_window
            if attack_opportunity is not None
            and attack_opportunity.get('kind') in ('building_window', 'low_elixir')
            else None
        )
        hog_opportunity_release = bool(
            attack_opportunity is not None
            and attack_opportunity.get('release_hog') is True
        )
        hog_opportunity_lane = (
            attack_opportunity.get('lane') if attack_opportunity else None
        )
        self.quality['strategy_phase'] = strategy_phase
        self.quality['attack_hold_reason'] = (
            attack_hold['reason'] if attack_hold else None)
        self.quality['attack_hold_distance'] = (
            attack_hold['distance'] if attack_hold else None)
        self.quality['counterpush_lane'] = (
            counterpush['lane'] if counterpush else None)
        self.quality['counterpush_support_entity_id'] = (
            counterpush['support_entity_id'] if counterpush else None)
        self.quality['incoming_push_active'] = incoming_push is not None
        self.quality['incoming_push_lane'] = (
            incoming_push.get('lane') if incoming_push else None)
        self.quality['incoming_push_card_id'] = (
            incoming_push.get('core_card_id') if incoming_push else None)
        self.quality['incoming_push_cost'] = (
            incoming_push.get('core_cost') if incoming_push else None)
        self.quality['incoming_push_depth'] = (
            incoming_push.get('core_depth') if incoming_push else None)
        self.quality['incoming_push_core_count'] = (
            incoming_push.get('core_count') if incoming_push else 0)
        self.quality['incoming_push_evidence'] = (
            incoming_push.get('evidence') if incoming_push else None)
        self.quality['backfield_commitment_active'] = (
            backfield_commitment is not None)
        self.quality['backfield_commitment_lane'] = (
            backfield_commitment.get('lane')
            if backfield_commitment else None)
        self.quality['backfield_commitment_card_id'] = (
            backfield_commitment.get('card_id')
            if backfield_commitment else None)
        self.quality['backfield_commitment_cost'] = (
            backfield_commitment.get('cost')
            if backfield_commitment else None)
        self.quality['backfield_commitment_depth'] = (
            backfield_commitment.get('depth')
            if backfield_commitment else None)
        self.quality['backfield_patience_active'] = bool(
            backfield_patience)
        self.quality['defense_overflow_active'] = bool(
            defense_overflow_active)
        self.quality['neutral_overflow_active'] = bool(
            neutral_overflow_active)
        self.quality['low_value_defensive_threat_active'] = bool(
            low_value_defensive_threat)
        self.quality['low_value_defensive_threat_card_id'] = (
            low_value_defensive_threat.get('card_id')
            if low_value_defensive_threat else None)
        self.quality['low_value_defensive_threat_cost'] = (
            low_value_defensive_threat.get('cost')
            if low_value_defensive_threat else None)
        self.quality['low_value_defensive_threat_entity_id'] = (
            low_value_defensive_threat.get('entity_id')
            if low_value_defensive_threat else None)
        last_exact = self._last_opponent_exact_play
        self.quality['opponent_last_exact_play_card_id'] = (
            last_exact.get('card_id') if last_exact else None)
        self.quality['opponent_last_exact_play_cost'] = (
            last_exact.get('cost') if last_exact else None)
        self.quality['opponent_last_exact_play_kind'] = (
            last_exact.get('kind') if last_exact else None)
        self.quality['opponent_last_exact_play_age_ticks'] = (
            int(state.tick) - int(last_exact['tick']) if last_exact else None)
        recent_heavy = next((
            row for row in reversed(self._recent_opponent_heavy_plays)
            if 0 <= int(state.tick) - int(row['tick'])
            <= INCOMING_PUSH_RECENT_HEAVY_TICKS
        ), None)
        self.quality['opponent_recent_heavy_card_id'] = (
            recent_heavy.get('card_id') if recent_heavy else None)
        self.quality['opponent_recent_heavy_cost'] = (
            recent_heavy.get('cost') if recent_heavy else None)
        self.quality['opponent_recent_heavy_age_ticks'] = (
            int(state.tick) - int(recent_heavy['tick'])
            if recent_heavy else None)
        self.quality['opponent_recent_heavy_bound_entity_id'] = (
            recent_heavy.get('bound_entity_id') if recent_heavy else None)
        self.quality['incoming_push_reserve_defenders'] = bool(
            incoming_push and incoming_push.get('reserve_core_defenders'))
        self.quality['incoming_push_preparing'] = preparing_for_push
        self.quality['incoming_push_defending'] = defending_incoming_push
        self.quality['heavy_defend_cannon_release_depth'] = (
            HEAVY_DEFEND_CANNON_RELEASE_DEPTH
            if defending_incoming_push else None
        )
        self.quality['heavy_defend_cannon_held'] = bool(
            defending_incoming_push
            and heavy_core_depth is not None
            and heavy_core_depth > HEAVY_DEFEND_CANNON_RELEASE_DEPTH
        )
        self.quality['heavy_defend_musketeer_held'] = bool(
            defending_incoming_push
            and heavy_core_depth is not None
            and heavy_core_depth > HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH
        )
        self.quality['heavy_defend_priority_card_id'] = heavy_priority_card
        self.quality['heavy_defend_priority_stage'] = heavy_priority.get('stage')
        self.quality['heavy_defend_emergency_release'] = bool(
            heavy_priority.get('emergency'))
        self.quality['heavy_defend_ice_golem_held'] = bool(
            defending_incoming_push
            and heavy_core_depth is not None
            and heavy_core_depth > HEAVY_DEFEND_BODY_RELEASE_DEPTH
        )
        self.quality['heavy_defend_fireball_released'] = bool(
            heavy_fireball and heavy_fireball.get('release'))
        self.quality['heavy_defend_fireball_support_count'] = (
            heavy_fireball.get('support_count') if heavy_fireball else 0)
        self.quality['heavy_defend_fireball_support_cost'] = (
            heavy_fireball.get('support_cost') if heavy_fireball else 0.0)
        self.quality['incoming_push_hard_reserve_cards'] = (
            sorted(INCOMING_PUSH_HARD_RESERVE_CARDS)
            if preparing_for_push else []
        )
        self.quality['incoming_push_defensive_placement_max_depth'] = (
            INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH
            if preparing_for_push else None
        )
        self.quality['neutral_patience_active'] = neutral_patience
        self.quality['attack_opportunity_active'] = attack_opportunity is not None
        self.quality['attack_opportunity_kind'] = (
            attack_opportunity.get('kind') if attack_opportunity else None)
        self.quality['attack_opportunity_reason'] = (
            attack_opportunity.get('reason') if attack_opportunity else None)
        self.quality['attack_opportunity_lane'] = hog_opportunity_lane
        self.quality['attack_window_reason'] = (
            attack_window['reason'] if attack_window else None)
        self.quality['attack_window_card_id'] = (
            attack_window.get('card_id') if attack_window else None)
        self.quality['attack_window_age_ticks'] = (
            attack_window.get('age_ticks') if attack_window else None)
        self.quality['attack_window_plays_since'] = (
            attack_window.get('plays_since') if attack_window else None)
        self.quality['attack_window_plays_until_return'] = (
            attack_window.get('plays_until_return') if attack_window else None)
        self.quality['active_enemy_building_count'] = len(self._active_enemy_buildings)
        self.quality['opponent_exact_play_count'] = self._opponent_exact_play_count
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
            if prelock is not None and cid == HOG_RIDER:
                slot_reasons[str(slot)] = 'strategy_prelock_defense'
                continue
            if (
                defense_overflow_active
                and strategy_phase == 'defend'
                and cid == HOG_RIDER
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_attack_defense_overflow'
                )
                continue
            if cid == HOG_RIDER and attack_hold is not None:
                slot_reasons[str(slot)] = 'strategy_hold_attack_defense'
                continue
            if (
                low_value_defensive_threat is not None
                and cid in LOW_VALUE_DEFENSE_HELD_CARDS
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_core_defense_for_low_value_threat'
                )
                continue
            if (defending_incoming_push
                    and cid == CANNON
                    and heavy_core_depth is not None
                    and heavy_core_depth > HEAVY_DEFEND_CANNON_RELEASE_DEPTH):
                slot_reasons[str(slot)] = (
                    'strategy_hold_cannon_for_heavy_core'
                )
                continue
            if (defending_incoming_push
                    and cid == MUSKETEER
                    and heavy_core_depth is not None
                    and heavy_core_depth > HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH):
                slot_reasons[str(slot)] = (
                    'strategy_hold_musketeer_for_heavy_core'
                )
                continue
            if (defending_incoming_push
                    and cid == ICE_GOLEM
                    and heavy_core_depth is not None
                    and heavy_core_depth > HEAVY_DEFEND_BODY_RELEASE_DEPTH):
                slot_reasons[str(slot)] = (
                    'strategy_hold_ice_golem_for_heavy_core'
                )
                continue
            if (defending_incoming_push
                    and cid == FIREBALL
                    and not (heavy_fireball and heavy_fireball.get('release'))):
                slot_reasons[str(slot)] = (
                    'strategy_hold_fireball_for_heavy_support'
                )
                continue
            if (defending_incoming_push
                    and not heavy_priority.get('emergency')
                    and heavy_priority_card is not None
                    and cid in HEAVY_DEFEND_SERIAL_CARDS
                    and cid != heavy_priority_card):
                slot_reasons[str(slot)] = (
                    'strategy_wait_heavy_defense_priority'
                )
                continue
            if (preparing_for_push
                    and cid in INCOMING_PUSH_HARD_RESERVE_CARDS):
                slot_reasons[str(slot)] = (
                    'strategy_reserve_incoming_push_spell'
                )
                continue
            if (preparing_for_push
                    and cid == HOG_RIDER
                    and float(elixir) < INCOMING_PUSH_PUNISH_MIN_ELIXIR):
                slot_reasons[str(slot)] = 'strategy_hold_incoming_push'
                continue
            if (preparing_for_push
                    and incoming_push.get('reserve_core_defenders')
                    and cid in INCOMING_PUSH_CORE_DEFENDERS
                    and not (
                        (
                            cid == CANNON
                            and cannon_prebuild_allowed
                        )
                        or (
                            cid == MUSKETEER
                            and defense_overflow_active
                        )
                    )):
                slot_reasons[str(slot)] = 'strategy_reserve_incoming_push'
                continue

            # Once the old generic reserve releases, keep each defender held
            # until its own useful engagement window.  In particular Cannon
            # must not spend most of its lifetime waiting for a slow tank.
            if (
                preparing_for_push
                and cid == CANNON
                and not cannon_prebuild_allowed
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_cannon_for_incoming_push'
                )
                continue

            if (
                preparing_for_push
                and heavy_core_depth is not None
                and cid == MUSKETEER
                and heavy_core_depth
                    > HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH
                and not defense_overflow_active
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_musketeer_for_incoming_push'
                )
                continue

            if (
                preparing_for_push
                and heavy_core_depth is not None
                and cid == ICE_GOLEM
                and heavy_core_depth
                    > HEAVY_DEFEND_BODY_RELEASE_DEPTH
                and not defense_overflow_hard_cap
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_ice_golem_for_incoming_push'
                )
                continue

            # Medium backfield commitments get the same patience principle
            # without pretending they are heavy push cores.
            if (
                backfield_patience
                and incoming_push is None
                and backfield_depth is not None
                and cid == CANNON
                and backfield_depth
                    > BACKFIELD_CANNON_RELEASE_DEPTH
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_backfield_cannon'
                )
                continue

            if (
                backfield_patience
                and incoming_push is None
                and backfield_depth is not None
                and cid == MUSKETEER
                and backfield_depth
                    > BACKFIELD_MUSKETEER_RELEASE_DEPTH
                and not defense_overflow_active
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_backfield_musketeer'
                )
                continue

            if (
                backfield_patience
                and incoming_push is None
                and backfield_depth is not None
                and cid == ICE_GOLEM
                and backfield_depth
                    > BACKFIELD_ICE_GOLEM_RELEASE_DEPTH
                and not defense_overflow_hard_cap
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_backfield_ice_golem'
                )
                continue

            if (
                backfield_patience
                and incoming_push is None
                and cid == FIREBALL
            ):
                slot_reasons[str(slot)] = (
                    'strategy_hold_backfield_spell'
                )
                continue

            if neutral_patience and cid in NEUTRAL_PATIENCE_CARDS:
                if not (cid == HOG_RIDER and hog_opportunity_release):
                    slot_reasons[str(slot)] = 'strategy_neutral_patience'
                    continue
            entry = self.build_placement_mask(cid, lanes, towers, entities,
                form_code=selections[cid]['active_form'] if selections is not None else 0,
                ability_hud=ability_hud)
            if (prelock is not None
                    and prelock_lane is not None
                    and cid in DEFENSIVE_LANE_CARDS):
                entry = self._mask_to_defensive_lane(
                    entry, prelock_lane)
            elif (
                    backfield_patience
                    and incoming_push is None
                    and backfield_lane is not None
                    and cid in DEFENSIVE_LANE_CARDS):
                # Never answer a visible backfield commitment by sinking a
                # defensive body in the opposite lane.  Reuse the proved role
                # depth bands for staged same-lane preparation.
                entry = self._mask_to_heavy_defense_role(
                    entry, cid, backfield_lane)
            elif (defending_incoming_push
                    and cid in (INCOMING_PUSH_CORE_DEFENDERS | frozenset((ICE_GOLEM,)))):
                entry = self._mask_to_heavy_defense_role(
                    entry, cid, incoming_push['lane'])
            elif (defending_incoming_push
                    and cid in HEAVY_DEFEND_CHEAP_CONTROL
                    and (defensive_lane_gate is None
                         or defensive_lane_gate['threat_lane'] == incoming_push['lane'])):
                entry = self._mask_to_heavy_defense_role(
                    entry, cid, incoming_push['lane'])
            elif defensive_lane_gate is not None and cid in DEFENSIVE_LANE_CARDS:
                entry = self._mask_to_defensive_lane(
                    entry, defensive_lane_gate['threat_lane'])
            elif (
                    preparing_for_push
                    and cannon_prebuild_allowed
                    and cid == CANNON
                    and incoming_push.get('lane') is not None):
                entry = self._mask_to_heavy_defense_role(
                    entry, cid, incoming_push['lane'])
                entry['cannon_prebuild'] = True
                entry['cannon_prebuild_reason'] = (
                    cannon_prebuild.get('reason')
                    if cannon_prebuild else None)
                entry['cannon_prebuild_lead_ticks'] = (
                    cannon_prebuild.get('lead_ticks')
                    if cannon_prebuild else None)
            elif (preparing_for_push
                    and incoming_push.get('lane') is not None
                    and cid in DEFENSIVE_LANE_CARDS):
                entry = self._mask_to_incoming_defense(
                    entry, incoming_push['lane'])
            elif (incoming_push is not None
                    and incoming_push.get('lane') is not None
                    and cid in INCOMING_PUSH_CORE_DEFENDERS):
                entry = self._mask_to_lane(
                    entry, incoming_push['lane'], 'incoming_push_lane')
            if (defending_incoming_push
                    and cid == FIREBALL
                    and heavy_fireball
                    and heavy_fireball.get('release')):
                entry = self._mask_spell_near_point(
                    entry,
                    heavy_fireball['core_x'],
                    heavy_fireball['core_y'],
                    HEAVY_DEFEND_FIREBALL_TARGET_RADIUS,
                    'heavy_defense_fireball',
                )
            if cid == HOG_RIDER and hog_opportunity_lane is not None:
                entry = self._mask_to_lane(
                    entry, hog_opportunity_lane, 'attack_opportunity_lane')
                if (attack_opportunity is not None
                        and attack_opportunity.get('kind') == 'counterpush'):
                    # Preserve the older placement-mask diagnostic while the
                    # unified context becomes the single source of truth.
                    entry['counterpush_lane'] = hog_opportunity_lane
            playable[slot] = any(any(row) for row in entry['row_major'])
            slot_reasons[str(slot)] = 'playable' if playable[slot] else 'no_legal_position'
            if playable[slot]:
                masks[str(slot)] = entry

        # WAIT must stay legal in V4. These slots define the structural
        # spend used only when a near-cap defensive WAIT would waste elixir.
        wait_allowed = True
        defense_overflow_forced = False
        defense_overflow_safe_slots = []
        defense_overflow_mode = None

        if defense_overflow_active:
            if strategy_phase == 'prepare_defense':
                # Phase 1: establish the backline defender that will actually
                # belong to the coming defense.
                musketeer_slots = [
                    slot
                    for slot, cid in slots.items()
                    if playable[slot]
                    and cid == MUSKETEER
                ]

                if musketeer_slots:
                    defense_overflow_safe_slots = sorted(
                        musketeer_slots)
                    defense_overflow_mode = (
                        'backline_setup'
                    )

                else:
                    # Phase 2: after the backline card has left the hand /
                    # become blocked, use the tank's walking time to cycle.
                    cycle_slots = [
                        slot
                        for slot, cid in slots.items()
                        if playable[slot]
                        and cid
                            in DEFENSE_OVERFLOW_SAFE_CYCLE_CARDS
                    ]

                    cannon_slots = [
                        slot
                        for slot, cid in slots.items()
                        if playable[slot]
                        and cid == CANNON
                        and cannon_prebuild_allowed
                    ]

                    # If the tank is already close enough that the Cannon
                    # anchor should be established now, prefer it. Otherwise
                    # allow the cheap cycle first to work toward a second
                    # rotation while the tank continues walking.
                    if (
                        cannon_slots
                        and cannon_prebuild_urgent
                    ):
                        defense_overflow_safe_slots = sorted(
                            cannon_slots)
                        defense_overflow_mode = (
                            'cannon_prebuild_urgent'
                        )

                    elif cycle_slots:
                        defense_overflow_safe_slots = sorted(
                            cycle_slots
                            + cannon_slots
                        )
                        defense_overflow_mode = (
                            'cycle_then_prebuild'
                        )

                    elif cannon_slots:
                        defense_overflow_safe_slots = sorted(
                            cannon_slots)
                        defense_overflow_mode = (
                            'cannon_prebuild'
                        )

                    else:
                        hog_slots = sorted(
                            slot
                            for slot, cid in slots.items()
                            if playable[slot]
                            and cid == HOG_RIDER
                        )

                        if (
                            incoming_push is not None
                            and hog_opportunity_release
                            and hog_slots
                        ):
                            defense_overflow_safe_slots = hog_slots
                            defense_overflow_mode = (
                                'heavy_commit_hog_punish'
                            )

                        elif defense_overflow_hard_cap:
                            body_slots = [
                                slot
                                for slot, cid in slots.items()
                                if playable[slot]
                                and cid == ICE_GOLEM
                            ]

                            if body_slots:
                                defense_overflow_safe_slots = sorted(
                                    body_slots)
                                defense_overflow_mode = (
                                    'ice_golem_body'
                                )

                if defense_overflow_safe_slots:
                    safe = set(
                        defense_overflow_safe_slots)

                    for slot in range(4):
                        if (
                            not playable[slot]
                            or slot in safe
                        ):
                            continue

                        playable[slot] = False
                        masks.pop(str(slot), None)
                        slot_reasons[str(slot)] = (
                            'strategy_defense_overflow_formation'
                        )

                    defense_overflow_forced = True

            elif strategy_phase == 'defend':
                if low_value_defensive_threat is not None:
                    legal = [
                        slot
                        for slot in range(4)
                        if playable[slot]
                        and int(slots.get(slot, -1))
                            in LOW_VALUE_DEFENSE_OVERFLOW_CARDS
                    ]
                    defense_overflow_mode = (
                        'low_value_defense_cycle'
                    )
                else:
                    legal = [
                        slot
                        for slot in range(4)
                        if playable[slot]
                        and int(slots.get(slot, -1))
                            != HOG_RIDER
                    ]
                    defense_overflow_mode = (
                        'live_defense'
                    )

                if legal:
                    defense_overflow_safe_slots = sorted(
                        legal)
                    defense_overflow_forced = True

        self.quality['defense_overflow_forced'] = bool(
            defense_overflow_forced)
        self.quality['defense_overflow_safe_slots'] = list(
            defense_overflow_safe_slots)
        self.quality['defense_overflow_mode'] = (
            defense_overflow_mode)


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
                     'slot_reasons': slot_reasons,
                     'defensive_threat_lane': (
                         defensive_lane_gate['threat_lane']
                         if defensive_lane_gate else None),
                     'prelock_state': (
                         prelock.get('state') if prelock else None),
                     'prelock_reason': (
                         prelock.get('reason') if prelock else None),
                     'prelock_enemy_id': (
                         prelock.get('enemy_id') if prelock else None),
                     'prelock_enemy_card_id': (
                         prelock.get('enemy_card_id') if prelock else None),
                     'prelock_tower_id': (
                         prelock.get('tower_id') if prelock else None),
                     'prelock_lane': (
                         prelock.get('lane') if prelock else None),
                     'prelock_distance': (
                         prelock.get('distance') if prelock else None),
                     'prelock_distance_to_lock': (
                         prelock.get('distance_to_lock')
                         if prelock else None),
                     'prelock_attack_range': (
                         prelock.get('attack_range') if prelock else None),
                     'prelock_lock_eta_ms': (
                         prelock.get('lock_eta_ms') if prelock else None),
                     'prelock_latest_safe_response_ms': (
                         prelock.get('latest_safe_response_ms')
                         if prelock else None),
                     'attack_hold_reason': (
                         attack_hold['reason'] if attack_hold else None),
                     'attack_hold_distance': (
                         attack_hold['distance'] if attack_hold else None),
                     'attack_hold_enemy_id': (
                         attack_hold['enemy_id'] if attack_hold else None),
                     'strategy_phase': strategy_phase,
                     'backfield_commitment_active': (
                         backfield_commitment is not None),
                     'backfield_commitment_lane': (
                         backfield_commitment.get('lane')
                         if backfield_commitment else None),
                     'backfield_commitment_card_id': (
                         backfield_commitment.get('card_id')
                         if backfield_commitment else None),
                     'backfield_commitment_cost': (
                         backfield_commitment.get('cost')
                         if backfield_commitment else None),
                     'backfield_commitment_depth': (
                         backfield_commitment.get('depth')
                         if backfield_commitment else None),
                     'backfield_patience_active': bool(
                         backfield_patience),
                     'defense_overflow_active': bool(
                         defense_overflow_active),
                     'neutral_overflow_active': bool(
                         neutral_overflow_active),
                     'low_value_defensive_threat_active': bool(
                         low_value_defensive_threat),
                     'low_value_defensive_threat_card_id': (
                         low_value_defensive_threat.get('card_id')
                         if low_value_defensive_threat else None),
                     'low_value_defensive_threat_cost': (
                         low_value_defensive_threat.get('cost')
                         if low_value_defensive_threat else None),
                     'low_value_defensive_threat_entity_id': (
                         low_value_defensive_threat.get('entity_id')
                         if low_value_defensive_threat else None),
                     'defense_overflow_hard_cap': bool(
                         defense_overflow_hard_cap),
                     'defense_overflow_forced': bool(
                         defense_overflow_forced),
                     'defense_overflow_safe_slots': list(
                         defense_overflow_safe_slots),
                     'defense_overflow_mode': (
                         defense_overflow_mode),
                     'cannon_prebuild_allowed': bool(
                         cannon_prebuild_allowed),
                     'cannon_prebuild_urgent': bool(
                         cannon_prebuild_urgent),
                     'cannon_prebuild_reason': (
                         cannon_prebuild.get('reason')
                         if cannon_prebuild else None),
                     'cannon_prebuild_lead_ticks': (
                         cannon_prebuild.get('lead_ticks')
                         if cannon_prebuild else None),
                     'counterpush_lane': (
                         counterpush['lane'] if counterpush else None),
                     'counterpush_support_entity_id': (
                         counterpush['support_entity_id'] if counterpush else None),
                     'counterpush_support_card_id': (
                         counterpush['support_card_id'] if counterpush else None),
                     'incoming_push_active': incoming_push is not None,
                     'incoming_push_lane': (
                         incoming_push.get('lane') if incoming_push else None),
                     'incoming_push_card_id': (
                         incoming_push.get('core_card_id') if incoming_push else None),
                     'incoming_push_cost': (
                         incoming_push.get('core_cost') if incoming_push else None),
                     'incoming_push_depth': (
                         incoming_push.get('core_depth') if incoming_push else None),
                     'incoming_push_core_count': (
                         incoming_push.get('core_count') if incoming_push else 0),
                     'incoming_push_evidence': (
                         incoming_push.get('evidence') if incoming_push else None),
                     'opponent_last_exact_play_card_id': (
                         last_exact.get('card_id') if last_exact else None),
                     'opponent_last_exact_play_cost': (
                         last_exact.get('cost') if last_exact else None),
                     'opponent_last_exact_play_kind': (
                         last_exact.get('kind') if last_exact else None),
                     'opponent_last_exact_play_age_ticks': (
                         int(state.tick) - int(last_exact['tick'])
                         if last_exact else None),
                     'opponent_recent_heavy_card_id': (
                         recent_heavy.get('card_id') if recent_heavy else None),
                     'opponent_recent_heavy_cost': (
                         recent_heavy.get('cost') if recent_heavy else None),
                     'opponent_recent_heavy_age_ticks': (
                         int(state.tick) - int(recent_heavy['tick'])
                         if recent_heavy else None),
                     'opponent_recent_heavy_bound_entity_id': (
                         recent_heavy.get('bound_entity_id')
                         if recent_heavy else None),
                     'incoming_push_reserve_defenders': bool(
                         incoming_push and incoming_push.get('reserve_core_defenders')),
                     'incoming_push_preparing': preparing_for_push,
                     'incoming_push_defending': defending_incoming_push,
                     'heavy_defend_cannon_release_depth': (
                         HEAVY_DEFEND_CANNON_RELEASE_DEPTH
                         if defending_incoming_push else None),
                     'heavy_defend_cannon_held': bool(
                         defending_incoming_push
                         and heavy_core_depth is not None
                         and heavy_core_depth > HEAVY_DEFEND_CANNON_RELEASE_DEPTH),
                     'heavy_defend_musketeer_held': bool(
                         defending_incoming_push
                         and heavy_core_depth is not None
                         and heavy_core_depth > HEAVY_DEFEND_MUSKETEER_RELEASE_DEPTH),
                     'heavy_defend_priority_card_id': heavy_priority_card,
                     'heavy_defend_priority_stage': heavy_priority.get('stage'),
                     'heavy_defend_emergency_release': bool(
                         heavy_priority.get('emergency')),
                     'heavy_defend_ice_golem_held': bool(
                         defending_incoming_push
                         and heavy_core_depth is not None
                         and heavy_core_depth > HEAVY_DEFEND_BODY_RELEASE_DEPTH),
                     'heavy_defend_fireball_released': bool(
                         heavy_fireball and heavy_fireball.get('release')),
                     'heavy_defend_fireball_support_count': (
                         heavy_fireball.get('support_count')
                         if heavy_fireball else 0),
                     'heavy_defend_fireball_support_cost': (
                         heavy_fireball.get('support_cost')
                         if heavy_fireball else 0.0),
                     'incoming_push_hard_reserve_cards': (
                         sorted(INCOMING_PUSH_HARD_RESERVE_CARDS)
                         if preparing_for_push else []),
                     'incoming_push_defensive_placement_max_depth': (
                         INCOMING_PUSH_DEFENSIVE_PLACEMENT_MAX_DEPTH
                         if preparing_for_push else None),
                     'neutral_patience_active': neutral_patience,
                     'neutral_patience_release_elixir': NEUTRAL_PATIENCE_RELEASE_ELIXIR,
                     'attack_opportunity_active': attack_opportunity is not None,
                     'attack_opportunity_kind': (
                         attack_opportunity.get('kind') if attack_opportunity else None),
                     'attack_opportunity_reason': (
                         attack_opportunity.get('reason') if attack_opportunity else None),
                     'attack_opportunity_lane': hog_opportunity_lane,
                     'attack_opportunity_release_hog': hog_opportunity_release,
                     'attack_window_reason': (
                         attack_window['reason'] if attack_window else None),
                     'attack_window_card_id': (
                         attack_window.get('card_id') if attack_window else None),
                     'attack_window_age_ticks': (
                         attack_window.get('age_ticks') if attack_window else None),
                     'attack_window_plays_since': (
                         attack_window.get('plays_since') if attack_window else None),
                     'attack_window_plays_until_return': (
                         attack_window.get('plays_until_return') if attack_window else None),
                     'active_enemy_building_count': len(self._active_enemy_buildings),
                     'opponent_exact_play_count': self._opponent_exact_play_count,
                     'opponent_elixir_lower': (
                         opponent_elixir_bounds[0] if opponent_elixir_bounds else None),
                     'opponent_elixir_upper': (
                         opponent_elixir_bounds[1] if opponent_elixir_bounds else None)})
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
