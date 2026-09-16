"""Bounded action queue, native-slot locks, resource reservations and ACKs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import config
from bridge.coordinates import action_world
from bridge.hero_execution import ABILITY, MUSKETEER, raw_controller
from bridge.evolution_state import BINDINGS, is_evolved_entity


@dataclass
class PendingAction:
    action: object
    decision_tick: int
    due: float
    expires: float
    cost: float
    command_seq: int = 0
    state: str = 'queued'
    sent_at: float = 0.0
    prior_entities: frozenset = frozenset()
    sent_tick: int = -1
    prior_evolution_progress: int | None = None
    prior_elixir: float | None = None
    decision_at: float = 0.0
    observation_received_at: float = 0.0


class ActionExecutor:
    # Keep input strictly serial. Card selection, placement and hand rotation
    # share one UI state; overlapping actions can make taps interfere.
    MAX_IN_FLIGHT_ACTIONS = 1

    def __init__(self, actuator, log, dry_run=False, on_ability_ack=None, max_actions=None):
        self.actuator, self.log, self.dry_run = actuator, log, dry_run
        self.on_ability_ack = on_ability_ack
        if max_actions is not None and (type(max_actions) is not int or max_actions <= 0):
            raise ValueError('max_actions must be a positive integer or None')
        self.max_actions = max_actions
        self.attempted_actions = 0
        self.confirmed_actions = 0
        self.pending = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='android-input')
        self.future = None
        self.active = None
        self.cooldowns = {}
        self.unconfirmed_spend = []
        self.predictions = []
        self.fault = None
        self.spawn_watch = []
        self.ability_locks = set()
        self._next_command_seq = 1
        self._fresh_state_required = False
        self._decision_blocked_until = 0.0
        self._recent_misses = {}
        self._confirmed_for_simulation = []
        self._latency_samples_ms = []
        self.end_to_end_latency_ms = float(config.PREDICTION_LATENCY_COMPENSATION_MS)
        self._post_action_settle_tick = -1
        self._post_action_preview = None

    def _in_flight_count(self):
        return len(self.pending) + len(self.spawn_watch)

    def reset(self):
        # Do not carry queued actions, reservations or locks across matches.
        # An already submitted Android command cannot be unsent.
        self.pending.clear()
        self.spawn_watch.clear()
        self.cooldowns.clear()
        self.unconfirmed_spend.clear()
        self.predictions.clear()
        self.ability_locks.clear()
        self.fault = None
        self.attempted_actions = 0
        self.confirmed_actions = 0
        self._fresh_state_required = False
        self._decision_blocked_until = 0.0
        self._recent_misses.clear()
        self._confirmed_for_simulation.clear()
        self._post_action_settle_tick = -1
        self._post_action_preview = None

    def consume_fresh_state_required(self):
        """Return whether the caller must fetch a new probe frame first."""
        required = self._fresh_state_required
        self._fresh_state_required = False
        return required

    def consume_confirmed_for_simulation(self):
        """Return card accepts not yet visible in rich combat events."""
        actions = tuple(self._confirmed_for_simulation)
        self._confirmed_for_simulation.clear()
        return actions

    def decision_blocked(self, state=None):
        """Whether a new policy turn must wait for an authoritative frame."""
        if time.perf_counter() < self._decision_blocked_until:
            return True
        return state is not None and self._post_action_settle_tick >= 0 and state.tick < self._post_action_settle_tick

    @staticmethod
    def _candidate_key(decoded):
        action = next((item for item in decoded.actions if item.kind.value != 'wait'), None)
        if action is None:
            return None
        return (action.kind.value, int(action.card_id or 0),
                int(action.hand_slot if action.hand_slot is not None else -1),
                int(action.source_entity if action.source_entity is not None else -1),
                tuple(action.target_grid) if action.target_grid is not None else None)

    @staticmethod
    def _candidate_score(decoded):
        action = next((item for item in decoded.actions if item.kind.value != 'wait'), None)
        if action is None:
            return 0.0
        value = action.metadata.get('policy_sequence_probability', 0.0)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0

    def _arm_post_action_recheck(self, state, pending, *, reason):
        """Require a settled no-write preview before the next card input."""
        settle_tick = int(state.tick) + config.POST_ACTION_SETTLE_TICKS
        self._post_action_settle_tick = max(self._post_action_settle_tick, settle_tick)
        self._post_action_preview = None
        self.log('post_action_settle_armed', command_seq=pending.command_seq,
                 card=pending.action.card_id, reason=reason,
                 observed_tick=state.tick, settle_tick=self._post_action_settle_tick,
                 settle_ticks=config.POST_ACTION_SETTLE_TICKS)

    def post_action_recheck(self, decoded, state):
        """Return true only when a post-card candidate remains stable.

        The first settled inference is deliberately preview-only.  The next
        policy frame may submit only the same first action and only if its
        selected-sequence score has retained enough of that preview score.
        """
        if self._post_action_settle_tick < 0:
            return True
        if state.tick < self._post_action_settle_tick:
            return False
        key = self._candidate_key(decoded)
        score = self._candidate_score(decoded)
        if self._post_action_preview is None:
            self._post_action_preview = {'key': key, 'score': score, 'tick': int(state.tick)}
            self.log('post_action_preview', tick=state.tick, candidate=key,
                     sequence_probability=score, write_suppressed=True)
            # A WAIT preview has already answered the question: no second
            # response is currently warranted.  Return to normal cadence.
            if key is None:
                self._post_action_settle_tick = -1
                self._post_action_preview = None
            return False
        preview = self._post_action_preview
        self._post_action_settle_tick = -1
        self._post_action_preview = None
        minimum = preview['score'] * config.POST_ACTION_RECHECK_MIN_SCORE_RATIO
        accepted = key is not None and key == preview['key'] and score >= minimum
        self.log('post_action_recheck', tick=state.tick, preview_candidate=preview['key'],
                 candidate=key, preview_probability=preview['score'],
                 sequence_probability=score, minimum_probability=minimum,
                 accepted=accepted)
        return accepted

    @staticmethod
    def _action_key(action):
        return (int(action.card_id or 0), int(action.hand_slot if action.hand_slot is not None else -1),
                tuple(action.target_grid) if action.target_grid is not None else None)

    def pause(self):
        # Drop stale plans, retain sent actions for ACK reconciliation on resume.
        for pending in list(self.pending):
            if pending.state == 'queued':
                self.pending.remove(pending)
                self.log('action_cancelled', reason='telemetry_paused',
                         card=pending.action.card_id, decision_tick=pending.decision_tick)

    def end_battle(self):
        for pending in self.pending:
            self.log('action_cancelled' if pending.state == 'queued' else 'action_unresolved_at_terminal',
                     reason='battle_finalized', card=pending.action.card_id,
                     decision_tick=pending.decision_tick, status=pending.state)
        self.reset()

    def blocked_slots(self, state):
        blocked = {p.action.hand_slot for p in (*self.pending, *self.spawn_watch)
                   if p.action.hand_slot is not None}
        now = time.perf_counter()
        for slot, (card, until) in list(self.cooldowns.items()):
            if state.hand_cards[slot] != card or now >= until:
                del self.cooldowns[slot]
            else:
                blocked.add(slot)
        if self.fault:
            blocked.update(range(4))
        return blocked

    def blocked_abilities(self):
        return self.ability_locks | {p.action.source_entity for p in (*self.pending, *self.spawn_watch)
                                    if p.action.kind.value == 'activate_ability'}

    @property
    def reserved_elixir(self):
        return sum(p.cost for p in (*self.pending, *self.spawn_watch)) + sum(
            cost for _, cost, _ in self.unconfirmed_spend)

    def _prune_unconfirmed_spend(self, now):
        self.unconfirmed_spend[:] = [row for row in self.unconfirmed_spend
                                     if now - row[0] < config.ELIXIR_RESERVATION_SECONDS]

    def _clear_spend(self, command_seq):
        self.unconfirmed_spend[:] = [row for row in self.unconfirmed_spend
                                     if row[2] != command_seq]

    def _drop_prediction(self, command_seq):
        """Remove only the speculative unit for one unconfirmed input."""
        self.predictions[:] = [prediction for prediction in self.predictions
                               if prediction.get('command_seq') != command_seq]

    def _retain_uncertain_prediction(self, command_seq, now):
        """Keep only a short-lived visual prediction after an ACK timeout."""
        for prediction in self.predictions:
            if prediction.get('command_seq') == command_seq:
                prediction['uncertain'] = True
                prediction['expires_at'] = now + config.UNCERTAIN_PREDICTION_SECONDS
                return True
        return False

    def model_predictions(self, state):
        """Return short-lived, model-only predictions for submitted cards."""
        if not config.ENABLE_MODEL_PREDICTION_OVERLAY:
            self.predictions.clear()
            return ()
        now = time.perf_counter()
        self.predictions[:] = [p for p in self.predictions if now < p['expires_at']]
        real = tuple(state.entities) if state is not None else ()
        kept = []
        for prediction in self.predictions:
            action = prediction['action']
            # Hero abilities are button taps and have no ground target or
            # deployable entity. They must never enter the unit overlay.
            if action.kind.value != 'play_card' or action.target_grid is None:
                continue
            target = action_world(action)
            matched = any(e.get('owner') == action.owner and e.get('card_id') == action.card_id
                          and (e.get('x', target[0]) - target[0]) ** 2
                          + (e.get('y', target[1]) - target[1]) ** 2 <= 1800 ** 2
                          for e in real)
            if not matched:
                kept.append(prediction)
        self.predictions[:] = kept
        return tuple({'id': -int(p['command_seq']), 'owner': p['action'].owner,
                      'card_id': p['action'].card_id, 'x': target[0], 'y': target[1],
                      'predicted': True, 'command_seq': p['command_seq'],
                      'age_ms': max(0.0, (now - float(p.get('sent_at', now))) * 1000.0),
                      'latency_ms': self.end_to_end_latency_ms}
                     for p in kept for target in (action_world(p['action']),))

    def submit(self, decoded, state):
        if getattr(state, 'native_finalized', False) is True:
            self.log('action_rejected', reason='battle_finalized')
            return
        for action in decoded.actions:
            if action.kind.value == 'wait':
                continue
            if self.fault:
                self.log('action_rejected', reason='execution_fault', action=action.to_dict())
                continue
            if self.max_actions is not None and self.attempted_actions >= self.max_actions:
                self.log('action_rejected', reason='action_budget_reached', action=action.to_dict(),
                         max_actions=self.max_actions)
                continue
            if self._in_flight_count() >= self.MAX_IN_FLIGHT_ACTIONS:
                self.log('action_rejected', reason='in_flight_capacity', action=action.to_dict(),
                         capacity=self.MAX_IN_FLIGHT_ACTIONS)
                # One serialized input is already in flight. The remaining
                # actions belong to the same model decision and must not be
                # retried in a tight loop; the next decision will use fresh
                # hand/elixir/scene state.
                break
            skill = action.kind.value == 'activate_ability'
            if action.kind.value not in ('play_card', 'activate_ability'):
                self.log('unsupported_action', action=action.to_dict())
                continue
            slot = action.hand_slot
            if skill:
                row = raw_controller(state, action)
                if (action.owner != state.local_owner or action.ability_id != ABILITY or self.fault
                    or action.target_kind.value != 'none' or action.source_entity in self.blocked_abilities()
                    or not row or row.get('button_state') not in (2, 4) or row.get('charges') != 1
                    or row.get('cooldown_ms') != 0 or row.get('max_charges') != 1):
                    self.log('action_rejected', reason='ability_changed_or_locked', action=action.to_dict())
                    continue
                cost = 3.0  # verified Musketeer_hero_Ability contract only
            else:
                cost = float(action.metadata['policy_effective_cost'])
            if not skill and (slot is None or slot in self.blocked_slots(state) or state.hand_cards[slot] != action.card_id):
                self.log('action_rejected', reason='slot_changed_or_locked', action=action.to_dict())
                continue
            miss_until = self._recent_misses.get(self._action_key(action), 0.0)
            if not skill and time.perf_counter() < miss_until:
                self.log('action_suppressed', reason='recent_miss_same_target',
                         action=action.to_dict(), remaining_ms=(miss_until-time.perf_counter())*1000)
                continue
            if state.elixir - self.reserved_elixir < cost:
                self.log('action_rejected', reason='reserved_elixir', action=action.to_dict(),
                         elixir=state.elixir, reserved_elixir=self.reserved_elixir,
                         required_cost=cost, available_elixir=state.elixir - self.reserved_elixir)
                continue
            due = state.received_at + action.execute_offset_ticks * config.TICK_SECONDS
            pending = PendingAction(action, state.tick, due, due + config.ACTION_MAX_LATENESS_SECONDS,
                cost, command_seq=self._next_command_seq, decision_at=time.perf_counter())
            pending.observation_received_at = state.received_at
            self._next_command_seq += 1
            if self.dry_run:
                self.log('dry_run_action', action=action.to_dict(),
                    screen=self.actuator.ability_screen(action) if skill else self.actuator.calibration.action_to_screen(action),
                    world=None if skill else action_world(action))
            else:
                self.pending.append(pending)
                self.attempted_actions += 1
                self.log('action_queued', action=action.to_dict(), decision_tick=state.tick,
                         command_seq=pending.command_seq)
                # Touch selection and placement share one UI transaction.
                # Accept only the first actionable item from a decoded batch;
                # never queue a second card from the same stale observation.
                if len(decoded.actions) > 1:
                    self.log('action_rejected', reason='in_flight_capacity',
                             action=decoded.actions[1].to_dict(),
                             capacity=self.MAX_IN_FLIGHT_ACTIONS,
                             suppressed_batch=True)
                break

    def poll(self, state, validate):
        now = time.perf_counter()
        self._prune_unconfirmed_spend(now)
        if self.future is not None and self.future.done():
            completed_at = time.perf_counter()
            try:
                result = self.future.result()
                end_to_end_ms = max(0.0, (completed_at - self.active.decision_at) * 1000.0)
                self._latency_samples_ms.append(end_to_end_ms)
                # A single slow ADB/emulator frame must not move the model's
                # forecast hundreds of milliseconds into the future.  Use a
                # recent median for timing alignment; the current action's
                # measured latency is still logged independently.
                self._latency_samples_ms[:] = self._latency_samples_ms[-8:]
                ordered = sorted(self._latency_samples_ms)
                middle = len(ordered) // 2
                if len(ordered) % 2:
                    robust_latency = ordered[middle]
                else:
                    robust_latency = (ordered[middle - 1] + ordered[middle]) / 2.0
                self.end_to_end_latency_ms = max(50.0, min(350.0, robust_latency))
                self.log('input_completed', card=self.active.action.card_id,
                         ability=self.active.action.ability_id,
                         command_seq=self.active.command_seq,
                         end_to_end_latency_ms=end_to_end_ms,
                         observation_to_input_completed_ms=(
                             (completed_at-self.active.observation_received_at)*1000
                             if self.active.observation_received_at else None),
                         prediction_compensation_ms=self.end_to_end_latency_ms,
                         **result)
            except Exception as exc:
                self.fault = str(exc)
                self.pending.clear()
                self.log('input_fault', error=self.fault)
            self.future, self.active = None, None
        if state is None:
            # A transient query failure must not launch any queued touch.
            return
        if getattr(state, 'native_finalized', False) is True:
            self.end_battle()
            return
        for pending in list(self.pending):
            action = pending.action
            skill = action.kind.value == 'activate_ability'
            if pending.state == 'sent':
                if skill:
                    row = raw_controller(state, action)
                    if state.tick > pending.sent_tick and row and row.get('charges') == 0:
                        self.pending.remove(pending)
                        self._clear_spend(pending.command_seq)
                        self.log('ability_ack', ability=action.ability_id, source_entity=action.source_entity,
                                 tick=state.tick, latency_ms=(now-pending.sent_at)*1000,
                                 evidence='same_controller_carrier_charge_1_to_0')
                        self.confirmed_actions += 1
                        self._fresh_state_required = True
                        if self.on_ability_ack:
                            self.on_ability_ack(action, state)
                    elif now - pending.sent_at > config.ACK_TIMEOUT_SECONDS:
                        self.pending.remove(pending)
                        # An ambiguous ability tap must never be replayed, but
                        # it should not stop ordinary card play for the rest
                        # of the match.  Lock this carrier and let cards
                        # continue after recording UNKNOWN for the skill.
                        self.ability_locks.add(action.source_entity)
                        self.log('ability_ack_timeout', ability=action.ability_id,
                                 source_entity=action.source_entity, command_seq=pending.command_seq,
                                 outcome='unknown')
                    continue
                hand_changed = (state.tick > pending.sent_tick and
                                state.hand_cards[action.hand_slot] != action.card_id)
                # Some probe revisions publish elixir before hand rotation.
                # Treat a sufficiently large, near-immediate cost drop as a
                # fallback ACK.  Natural regeneration cannot satisfy this
                # downward threshold, and the original hand check remains
                # authoritative whenever it is available.
                elixir_changed = False
                if pending.prior_elixir is not None and state.elixir is not None:
                    elixir_changed = (state.tick > pending.sent_tick and
                        float(state.elixir) <= float(pending.prior_elixir) - pending.cost + 0.15)
                if hand_changed or elixir_changed:
                    self.pending.remove(pending)
                    self._clear_spend(pending.command_seq)
                    self.log('hand_ack', card=action.card_id, slot=action.hand_slot,
                        command_seq=pending.command_seq,
                        tick=state.tick, latency_ms=(now-pending.sent_at)*1000,
                        decision_to_ack_ms=(now-pending.decision_at)*1000,
                        input_to_ack_ms=(now-pending.sent_at)*1000,
                        outcome='accepted',
                        evidence=('hand_rotation' if hand_changed else 'elixir_cost_drop'))
                    if action.card_id in BINDINGS and action.metadata.get('policy_effective_form_code') == 1:
                        player = next(p for p in state.raw['players'] if p['owner'] == action.owner)
                        row = next((r for r in player.get('card_runtime', []) if r.get('card_id') == action.card_id), {})
                        if pending.prior_evolution_progress == BINDINGS[action.card_id].cycles and row.get('evolution_progress') == 0 and row.get('active_form') == 0:
                            self.log('evolution_ack', card=action.card_id, tick=state.tick,
                                evidence='hand_transition_and_native_cycle_2_to_0',
                                latency_ms=(now-pending.sent_at)*1000)
                    # Hand rotation is the authoritative client-side
                    # consume acknowledgement.  A troop can spawn and die
                    # between two observations, and spells have no durable
                    # spawn entity at all; neither case should halt the
                    # entire continuous runner.
                    self.confirmed_actions += 1
                    self._confirmed_for_simulation.append((action, pending.sent_tick,
                                                           pending.command_seq))
                    self._fresh_state_required = True
                    self._arm_post_action_recheck(state, pending, reason='hand_ack')
                elif now - pending.sent_at > config.ACK_TIMEOUT_SECONDS:
                    self.pending.remove(pending)
                    # A missing ACK is ambiguous: the touch may have reached
                    # the game while telemetry was late.  Keep its virtual
                    # unit briefly so the next policy frame does not spend a
                    # second card on an apparently untouched threat.  This
                    # does not disable the card or block other slots; a later
                    # authoritative entity/hand update removes it normally.
                    self._retain_uncertain_prediction(pending.command_seq, now)
                    self.cooldowns[action.hand_slot] = (action.card_id, now + 1)
                    # A completed ADB input with no observed hand transition
                    # means this attempt is ambiguous/missed, but it is not
                    # a transport failure.  Do not replay the same write and
                    # do not halt the match; briefly cool the slot and let
                    # the policy continue with fresh state.
                    self.log('action_missed', card=action.card_id, slot=action.hand_slot,
                             command_seq=pending.command_seq, outcome='unknown',
                             continue_running=True,
                             uncertain_prediction_seconds=config.UNCERTAIN_PREDICTION_SECONDS)
                    # A missed touch leaves the live hand unchanged.  Give
                    # the next probe frame time to arrive and suppress only
                    # this exact card/target briefly; other cards remain
                    # available and the card itself is never disabled.
                    # A fresh frame is required below. Do not add a global
                    # 350ms sleep after a missed ACK: it delays every other
                    # playable card even when the touch worker has finished.
                    # Slot, resource, and identical-action guards still apply.
                    self._recent_misses[self._action_key(action)] = now + 1.0
                    self._fresh_state_required = True
                    self._arm_post_action_recheck(state, pending, reason='ack_timeout')
            elif now > pending.expires:
                self.pending.remove(pending)
                self.log('action_expired', card=action.card_id, decision_tick=pending.decision_tick,
                         command_seq=pending.command_seq, outcome='skipped',
                         continue_running=True)
        for pending in list(self.spawn_watch):
            action = pending.action
            player = next(p for p in state.raw['players'] if p['owner'] == action.owner)
            hero_members = {eid for r in player.get('ability_runtime', [])
                if r.get('known') is True and r.get('ability_name') == ABILITY
                for eid in r.get('members', [])}
            evolved_play = action.card_id in BINDINGS and action.metadata.get('policy_effective_form_code') == 1
            spawned = [e for e in state.entities if e['id'] not in pending.prior_entities and
                e.get('owner') == action.owner and (is_evolved_entity(e, action.card_id) if evolved_play else (e.get('card_id') == action.card_id or
                (action.card_id == MUSKETEER and action.metadata.get('policy_effective_form_code') == 2
                 and e.get('card_id') == 203000014 and e['id'] in hero_members)))]
            if spawned:
                target = action_world(action)
                nearest = min(spawned, key=lambda e: (e['x']-target[0])**2 + (e['y']-target[1])**2)
                hero_carrier = any(r.get('known') is True and r.get('ability_name') == ABILITY
                    and r.get('members') == [nearest['id']] for r in player.get('ability_runtime', []))
                self.log('spawn_observed', card=action.card_id, target_world=target,
                    observed_world=[nearest['x'], nearest['y']], entity_id=nearest['id'],
                    command_seq=pending.command_seq,
                    tick=state.tick, input_to_observation_ms=(now-pending.sent_at)*1000,
                    observation_is_ack_gated=True,
                    selected_form=action.metadata.get('policy_effective_form_code', 0),
                    hero_carrier_confirmed=True if hero_carrier else None,
                    evolution_form_confirmed=True if evolved_play and is_evolved_entity(nearest, action.card_id) else None,
                    deployment_lineage_known=False,
                    note='first observed source-card match after hand ACK; not an exact spawn timestamp')
                self.confirmed_actions += 1
                self.spawn_watch.remove(pending)
            elif now - pending.sent_at > 3:
                self.log('spawn_unobserved', card=action.card_id, slot=action.hand_slot,
                         command_seq=pending.command_seq, outcome='unknown')
                self.spawn_watch.remove(pending)
        if self.future is not None or self.fault:
            return
        for pending in self.pending:
            if pending.state != 'queued':
                continue
            if now < pending.due:
                break
            action = pending.action
            skill = action.kind.value == 'activate_ability'
            validation_started = time.perf_counter()
            if state.local_owner != action.owner or (not skill and state.hand_cards[action.hand_slot] != action.card_id) or not validate(action, state):
                self.pending.remove(pending)
                self.log('action_rejected', reason='live_revalidation', action=action.to_dict())
                break
            now = time.perf_counter()
            if now > pending.expires:
                self.pending.remove(pending)
                self.log('action_expired', card=action.card_id, decision_tick=pending.decision_tick)
                break
            pending.state = 'sent'
            pending.sent_at = now
            pending.sent_tick = state.tick
            pending.prior_elixir = float(state.elixir)
            self.unconfirmed_spend.append((now, pending.cost, pending.command_seq))
            prediction_lifetime = (config.PREDICTION_LATENCY_COMPENSATION_MS / 1000.0
                                   + config.PREDICTION_HORIZON_TICKS * config.TICK_SECONDS + 0.4)
            self.predictions.append({'action': action, 'command_seq': pending.command_seq,
                                     'expires_at': now + prediction_lifetime})
            pending.prior_entities = frozenset(e['id'] for e in state.entities)
            if action.card_id in BINDINGS and action.metadata.get('policy_effective_form_code') == 1:
                player = next(p for p in state.raw['players'] if p['owner'] == action.owner)
                pending.prior_evolution_progress = next((r.get('evolution_progress')
                    for r in player.get('card_runtime', []) if r.get('card_id') == action.card_id), None)
            self.active = pending
            if skill:
                self.ability_locks.add(action.source_entity)  # one attempt per finite-charge carrier, even on timeout
            self.future = self.pool.submit(self.actuator.activate_ability if skill else self.actuator.deploy_action, action)
            self.log('input_started', card=action.card_id, slot=action.hand_slot,
                command_seq=pending.command_seq,
                decision_tick=pending.decision_tick, tick=state.tick,
                policy_wait_ms=action.metadata.get('policy_delay_offset_ms'),
                base_schedule_ticks=action.metadata.get('base_latency_ticks'),
                execute_offset_ticks=action.execute_offset_ticks,
                schedule_lateness_ms=(now-pending.due)*1000,
                decision_to_input_ms=(now-pending.decision_at)*1000,
                validation_ms=(now-validation_started)*1000,
                receive_to_input_submit_ms=(now-state.received_at)*1000,
                ability=action.ability_id, source_entity=action.source_entity,
                screen=self.actuator.ability_screen(action) if skill else self.actuator.calibration.action_to_screen(action))
            break

    def close(self):
        self.pending.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.actuator.close()
