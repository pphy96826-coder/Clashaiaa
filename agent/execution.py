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
    input_completed_at: float = 0.0
    threat_ids: frozenset = frozenset()
    threat_target: tuple[float, float] | None = None
    ack_timeout_seconds: float = 0.0


@dataclass(frozen=True)
class ThreatReservation:
    command_seq: int
    owner: int
    card_id: int
    threat_ids: frozenset
    target: tuple[float, float]
    expires_at: float
    confidence: str = 'confirmed'


class ActionExecutor:
    # Keep input strictly serial. Card selection, placement and hand rotation
    # share one UI state; overlapping actions can make taps interfere.
    MAX_IN_FLIGHT_ACTIONS = 1

    # Replace the global post-card stall only when a defensive play can be
    # tied to one concrete live enemy.  A short reservation suppresses another
    # card aimed at that same enemy while allowing immediate reactions to a
    # different threat.
    THREAT_RESERVATION_PROVISIONAL_SECONDS = 0.55
    THREAT_RESERVATION_CONFIRMED_SECONDS = 0.85
    THREAT_RESERVATION_UNCERTAIN_SECONDS = 0.70
    THREAT_RESERVATION_RADIUS_WORLD = 6000.0

    def __init__(self, actuator, log, dry_run=False, on_ability_ack=None, max_actions=None):
        self.actuator, self.log, self.dry_run = actuator, log, dry_run
        self.on_ability_ack = on_ability_ack
        if max_actions is not None and (type(max_actions) is not int or max_actions <= 0):
            raise ValueError('max_actions must be a positive integer or None')
        self.max_actions = max_actions
        self.attempted_actions = 0
        self.confirmed_actions = 0
        self.pending = []
        self.ack_watch = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='android-input')
        self.future = None
        self.active = None
        self.cooldowns = {}
        # A non-hand ACK (spawn/elixir) proves the input probably landed but
        # does not prove that the native hand snapshot has rotated yet. Keep
        # that exact slot/card locally blocked until the authoritative hand
        # transition arrives, without freezing other slots.
        self.slot_consume_guards = {}
        self.unconfirmed_spend = []
        self.predictions = []
        self.threat_reservations = []
        self.fault = None
        self.spawn_watch = []
        self.ability_locks = set()
        self._next_command_seq = 1
        self._fresh_state_required = False
        self._decision_blocked_until = 0.0
        self._recent_misses = {}
        self._confirmed_for_simulation = []
        self._latency_samples_ms = []
        self._ack_latency_samples_ms = []
        self.end_to_end_latency_ms = float(config.PREDICTION_LATENCY_COMPENSATION_MS)
        self._post_action_settle_tick = -1
        self._post_action_preview = None

    def _in_flight_count(self):
        return len(self.pending) + len(self.spawn_watch)

    def reset(self):
        # Do not carry queued actions, reservations or locks across matches.
        # An already submitted Android command cannot be unsent.
        self.pending.clear()
        self.ack_watch.clear()
        self.spawn_watch.clear()
        self.cooldowns.clear()
        self.slot_consume_guards.clear()
        self.unconfirmed_spend.clear()
        self.predictions.clear()
        self.threat_reservations.clear()
        self.ability_locks.clear()
        self.fault = None
        self.attempted_actions = 0
        self.confirmed_actions = 0
        self._fresh_state_required = False
        self._decision_blocked_until = 0.0
        self._recent_misses.clear()
        self._confirmed_for_simulation.clear()
        self._ack_latency_samples_ms.clear()
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

    @property
    def action_budget_exhausted(self):
        return (self.max_actions is not None
                and self.attempted_actions >= self.max_actions)

    @property
    def action_budget_settled(self):
        """Budget is spent and every submitted touch has a final ACK outcome."""
        return (self.action_budget_exhausted
                and self.future is None
                and not self.pending
                and not self.ack_watch)

    def decision_blocked(self, state=None):
        """Whether a new policy turn must wait for an authoritative frame."""
        if time.perf_counter() < self._decision_blocked_until:
            return True
        return state is not None and self._post_action_settle_tick >= 0 and state.tick < self._post_action_settle_tick

    def _card_ack_timeout_seconds(self):
        """Return a bounded ACK budget without stalling unrelated slots.

        Card ACK telemetry can lag Android input completion by close to a
        second on busy emulator frames.  Keep a one-second floor, expand it
        modestly when recent input/ACK latency is slow, and cap it below the
        resource-reservation horizon.  Only the submitted slot remains locked
        while this budget runs; other slots may continue normally.
        """
        timeout = float(config.CARD_ACK_TIMEOUT_BASE_SECONDS)
        input_based = (
            float(config.ACK_TIMEOUT_SECONDS)
            + float(config.CARD_ACK_TIMEOUT_INPUT_MULTIPLIER)
            * max(0.0, float(self.end_to_end_latency_ms)) / 1000.0
        )
        timeout = max(timeout, input_based)
        if self._ack_latency_samples_ms:
            ordered = sorted(self._ack_latency_samples_ms[-12:])
            # A small recent p80 is robust to one-off spikes while still
            # following sustained slow probe/guest hand rotation.
            index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.8) - 1))
            observed = (
                ordered[index] / 1000.0
                + float(config.CARD_ACK_TIMEOUT_MARGIN_SECONDS)
            )
            timeout = max(timeout, observed)
        return max(
            float(config.CARD_ACK_TIMEOUT_BASE_SECONDS),
            min(float(config.CARD_ACK_TIMEOUT_MAX_SECONDS), timeout),
        )

    def _record_card_ack_latency(self, latency_ms):
        if not isinstance(latency_ms, (int, float)) or not math.isfinite(latency_ms):
            return
        if latency_ms < 0:
            return
        self._ack_latency_samples_ms.append(float(latency_ms))
        self._ack_latency_samples_ms[:] = self._ack_latency_samples_ms[-12:]

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

    @staticmethod
    def _on_own_half(owner, y):
        return float(y) <= 16000.0 if owner == 0 else float(y) >= 16000.0

    @staticmethod
    def _live_entity_ids(state):
        return frozenset(
            int(entity['id']) for entity in state.entities
            if isinstance(entity.get('id'), int) and entity['id'] > 0
            and isinstance(entity.get('hp'), (int, float)) and entity['hp'] > 0
        )

    def _nearby_threat_ids(self, action, state):
        """Bind a defensive play to the nearest concrete enemy, if any.

        The first version deliberately reserves one primary threat instead of
        a whole lane/cluster.  That prevents duplicate answers to one Hog or
        Balloon without suppressing legitimate layered defence against a push.
        """
        if action.kind.value != 'play_card' or action.target_grid is None:
            return frozenset()
        target_x, target_y = action_world(action)
        if not self._on_own_half(action.owner, target_y):
            return frozenset()
        radius_sq = self.THREAT_RESERVATION_RADIUS_WORLD ** 2
        candidates = []
        for entity in state.entities:
            if entity.get('owner') == action.owner:
                continue
            entity_id = entity.get('id')
            card_id = entity.get('card_id')
            hp = entity.get('hp')
            x, y = entity.get('x'), entity.get('y')
            # Towers use non-positive card ids in this contract; projectiles
            # and effects either have no positive HP or no source card.
            if not isinstance(entity_id, int) or entity_id <= 0:
                continue
            if not isinstance(card_id, int) or card_id <= 0:
                continue
            if not isinstance(hp, (int, float)) or hp <= 0:
                continue
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            if not self._on_own_half(action.owner, y):
                continue
            distance_sq = ((float(x) - target_x) ** 2
                           + (float(y) - target_y) ** 2)
            if distance_sq <= radius_sq:
                candidates.append((distance_sq, entity_id))
        if not candidates:
            return frozenset()
        return frozenset((min(candidates)[1],))

    def _prune_threat_reservations(self, state, now=None):
        now = time.perf_counter() if now is None else now
        live_ids = self._live_entity_ids(state)
        self.threat_reservations[:] = [
            reservation for reservation in self.threat_reservations
            if now < reservation.expires_at
            and bool(reservation.threat_ids & live_ids)
        ]

    def _matching_threat_reservation(self, action, state):
        now = time.perf_counter()
        self._prune_threat_reservations(state, now)
        threat_ids = self._nearby_threat_ids(action, state)
        if not threat_ids:
            return None, threat_ids
        for reservation in reversed(self.threat_reservations):
            if reservation.owner == action.owner and threat_ids == reservation.threat_ids:
                return reservation, threat_ids
        return None, threat_ids

    def _commit_threat_reservation(self, pending, state, now, *, confidence='confirmed'):
        if not pending.threat_ids or pending.threat_target is None:
            return False
        live_ids = self._live_entity_ids(state)
        threat_ids = frozenset(pending.threat_ids & live_ids)
        if not threat_ids:
            return False
        ttl = {
            'provisional': self.THREAT_RESERVATION_PROVISIONAL_SECONDS,
            'confirmed': self.THREAT_RESERVATION_CONFIRMED_SECONDS,
            'uncertain': self.THREAT_RESERVATION_UNCERTAIN_SECONDS,
        }.get(confidence, self.THREAT_RESERVATION_CONFIRMED_SECONDS)
        reservation = ThreatReservation(
            command_seq=pending.command_seq,
            owner=int(pending.action.owner),
            card_id=int(pending.action.card_id or 0),
            threat_ids=threat_ids,
            target=pending.threat_target,
            expires_at=now + ttl,
            confidence=confidence,
        )
        self.threat_reservations[:] = [
            row for row in self.threat_reservations
            if row.command_seq != pending.command_seq
        ]
        self.threat_reservations.append(reservation)
        self.log('threat_reservation_started',
                 command_seq=pending.command_seq,
                 card=pending.action.card_id,
                 threat_ids=sorted(threat_ids),
                 target_world=list(pending.threat_target),
                 confidence=confidence,
                 ttl_ms=round(ttl * 1000))
        return True

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
        for pending in self.ack_watch:
            self.log('action_unresolved_at_terminal',
                     reason='battle_finalized', card=pending.action.card_id,
                     decision_tick=pending.decision_tick, status='ack_watch',
                     command_seq=pending.command_seq)
        self.reset()

    def _start_slot_consume_guard(self, pending, now, evidence):
        slot = pending.action.hand_slot
        if slot is None:
            return
        self.slot_consume_guards[int(slot)] = {
            'card_id': int(pending.action.card_id or 0),
            'command_seq': int(pending.command_seq),
            'evidence': str(evidence),
            'expires_at': float(now) + float(config.ELIXIR_RESERVATION_SECONDS),
        }
        self.log('slot_consume_guard_started',
                 command_seq=pending.command_seq,
                 card=pending.action.card_id,
                 slot=slot,
                 evidence=evidence,
                 ttl_ms=round(config.ELIXIR_RESERVATION_SECONDS * 1000))

    def _prune_slot_consume_guards(self, state, now=None):
        if state is None:
            return
        if now is None:
            now = time.perf_counter()
        for slot, guard in list(self.slot_consume_guards.items()):
            card_id = int(guard['card_id'])
            rotated = state.hand_cards[slot] != card_id
            expired = now >= float(guard['expires_at'])
            if not rotated and not expired:
                continue
            del self.slot_consume_guards[slot]
            self._clear_spend(int(guard['command_seq']))
            self.log('slot_consume_guard_cleared',
                     command_seq=guard['command_seq'],
                     card=card_id,
                     slot=slot,
                     evidence=guard['evidence'],
                     reason='hand_rotation' if rotated else 'timeout')

    def blocked_slots(self, state):
        blocked = {p.action.hand_slot for p in (*self.pending, *self.ack_watch, *self.spawn_watch)
                   if p.action.hand_slot is not None}
        now = time.perf_counter()
        self._prune_slot_consume_guards(state, now)
        blocked.update(self.slot_consume_guards)
        for slot, (card, until) in list(self.cooldowns.items()):
            if state.hand_cards[slot] != card or now >= until:
                del self.cooldowns[slot]
            else:
                blocked.add(slot)
        if self.fault:
            blocked.update(range(4))
        return blocked

    def blocked_abilities(self):
        return self.ability_locks | {p.action.source_entity
                                    for p in (*self.pending, *self.ack_watch, *self.spawn_watch)
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
            threat_ids = frozenset()
            threat_target = None
            if not skill:
                reservation, threat_ids = self._matching_threat_reservation(action, state)
                if reservation is not None:
                    now = time.perf_counter()
                    self.log('action_suppressed', reason='threat_already_committed',
                             action=action.to_dict(), threat_ids=sorted(threat_ids),
                             committed_by=reservation.command_seq,
                             remaining_ms=max(0.0, (reservation.expires_at-now)*1000.0))
                    continue
                if threat_ids:
                    threat_target = action_world(action)
            if state.elixir - self.reserved_elixir < cost:
                self.log('action_rejected', reason='reserved_elixir', action=action.to_dict(),
                         elixir=state.elixir, reserved_elixir=self.reserved_elixir,
                         required_cost=cost, available_elixir=state.elixir - self.reserved_elixir)
                continue
            due = state.received_at + action.execute_offset_ticks * config.TICK_SECONDS
            pending = PendingAction(action, state.tick, due, due + config.ACTION_MAX_LATENESS_SECONDS,
                cost, command_seq=self._next_command_seq, decision_at=time.perf_counter(),
                threat_ids=threat_ids, threat_target=threat_target)
            pending.observation_received_at = state.received_at
            self._next_command_seq += 1
            if self.dry_run:
                self.log('dry_run_action', action=action.to_dict(),
                    screen=self.actuator.ability_screen(action) if skill else self.actuator.calibration.action_to_screen(action),
                    world=None if skill else action_world(action))
            else:
                self.pending.append(pending)
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
                # The Android shell/touch transport can occasionally take
                # longer than the nominal ACK window.  The game cannot have
                # consumed a tap before that transport completes, so start
                # ACK timing here rather than at thread submission.  This
                # prevents a slow but still-live touch from being mislabeled
                # UNKNOWN halfway through its own input transaction.
                completed = self.active
                completed.input_completed_at = completed_at
                completed.ack_timeout_seconds = (
                    float(config.ACK_TIMEOUT_SECONDS)
                    if completed.action.kind.value == 'activate_ability'
                    else self._card_ack_timeout_seconds()
                )
                if completed in self.pending:
                    self.pending.remove(completed)
                if completed not in self.ack_watch:
                    self.ack_watch.append(completed)
                if (state is not None
                        and completed.action.kind.value == 'play_card'):
                    self._commit_threat_reservation(
                        completed, state, completed_at, confidence='provisional')
                self._fresh_state_required = True
                self.log('ack_watch_started',
                         command_seq=completed.command_seq,
                         card=completed.action.card_id,
                         slot=completed.action.hand_slot,
                         watches=len(self.ack_watch),
                         timeout_ms=round(completed.ack_timeout_seconds * 1000))
            except Exception as exc:
                self.fault = str(exc)
                self.pending.clear()
                self.log('input_fault', error=self.fault)
            self.future, self.active = None, None
            # Future completion may happen after the poll's initial timestamp.
            # Refresh monotonic time before reconciling older ACK watches.
            now = time.perf_counter()
        if state is None:
            # A transient query failure must not launch any queued touch.
            return
        if self.future is not None:
            # Never inspect hand/elixir timeout rules while the serial touch
            # transaction is still executing.  In particular, do not let the
            # the ACK budget race a slow emulator input operation.
            return
        if getattr(state, 'native_finalized', False) is True:
            self.end_battle()
            return
        self._prune_slot_consume_guards(state, now)
        for pending in list(self.pending):
            if pending.state == 'sent' and self.future is None:
                self.pending.remove(pending)
                if pending not in self.ack_watch:
                    self.ack_watch.append(pending)
        for pending in list(self.ack_watch):
            action = pending.action
            skill = action.kind.value == 'activate_ability'
            if pending.state == 'sent':
                # Never ACK a touch from the snapshot that was captured before
                # that touch completed. Wait for an authoritative post-input
                # probe frame; otherwise an older elixir/hand transition can
                # be misattributed to this action.
                input_completed_at = float(
                    getattr(pending, 'input_completed_at', 0.0) or 0.0)
                ack_started_at = input_completed_at or pending.sent_at
                if (input_completed_at > 0
                        and state.received_at <= input_completed_at):
                    continue
                if skill:
                    row = raw_controller(state, action)
                    if state.tick > pending.sent_tick and row and row.get('charges') == 0:
                        self.ack_watch.remove(pending)
                        self._clear_spend(pending.command_seq)
                        self.log('ability_ack', ability=action.ability_id, source_entity=action.source_entity,
                                 tick=state.tick, latency_ms=(now-ack_started_at)*1000,
                                 input_start_to_ack_ms=(now-pending.sent_at)*1000,
                                 evidence='same_controller_carrier_charge_1_to_0')
                        self.confirmed_actions += 1
                        self._fresh_state_required = True
                        if self.on_ability_ack:
                            self.on_ability_ack(action, state)
                    elif now - ack_started_at > config.ACK_TIMEOUT_SECONDS:
                        self.ack_watch.remove(pending)
                        # An ambiguous ability tap must never be replayed, but
                        # it should not stop ordinary card play for the rest
                        # of the match.  Lock this carrier and let cards
                        # continue after recording UNKNOWN for the skill.
                        self.ability_locks.add(action.source_entity)
                        self.log('ability_ack_timeout', ability=action.ability_id,
                                 source_entity=action.source_entity, command_seq=pending.command_seq,
                                 outcome='unknown')
                    continue
                ack_timeout_seconds = float(
                    getattr(pending, 'ack_timeout_seconds', 0.0) or 0.0)
                if ack_timeout_seconds <= 0:
                    ack_timeout_seconds = self._card_ack_timeout_seconds()
                    pending.ack_timeout_seconds = ack_timeout_seconds
                hand_changed = (state.tick > pending.sent_tick and
                                state.hand_cards[action.hand_slot] != action.card_id)

                # Troops/buildings can provide a second, strictly positive ACK
                # signal: a new own entity from the submitted source card that
                # did not exist before the input. Missing this signal never
                # proves failure, but seeing it proves the deploy reached the
                # game even when hand/elixir telemetry is late.
                card_family = int(action.card_id or 0) // 1_000_000
                spawn_ack = False
                spawn_entity_id = None
                spawn_ack_evidence = None
                if card_family in (26, 27):
                    evolved_play = (
                        action.card_id in BINDINGS
                        and action.metadata.get('policy_effective_form_code') == 1
                    )
                    hero_musketeer_play = (
                        action.card_id == MUSKETEER
                        and action.metadata.get('policy_effective_form_code') == 2
                    )
                    prior_entities = getattr(
                        pending, 'prior_entities', frozenset()) or frozenset()
                    hero_members = set()
                    if hero_musketeer_play:
                        player = next(
                            (p for p in state.raw.get('players', [])
                             if p.get('owner') == action.owner),
                            None,
                        )
                        if player:
                            hero_members = {
                                eid for row in player.get('ability_runtime', [])
                                if row.get('known') is True
                                and row.get('ability_name') == ABILITY
                                for eid in row.get('members', [])
                            }

                    new_own_entities = []
                    for entity in state.entities:
                        entity_id = entity.get('id')
                        if not isinstance(entity_id, int) or entity_id in prior_entities:
                            continue
                        if entity.get('owner') != action.owner:
                            continue
                        new_own_entities.append(entity)
                        if evolved_play:
                            matches = is_evolved_entity(entity, action.card_id)
                        elif hero_musketeer_play:
                            matches = (
                                entity.get('card_id') == 203000014
                                and entity_id in hero_members
                            )
                        else:
                            matches = entity.get('card_id') == action.card_id
                        if matches:
                            spawn_ack = True
                            spawn_entity_id = entity_id
                            spawn_ack_evidence = 'new_source_entity'
                            break

                    # Native entity card IDs can represent the spawned unit
                    # rather than the deck/source card (multi-body troops and
                    # transformed forms are common examples).  When exactly
                    # one card is awaiting ACK, a brand-new own entity that
                    # appears very close to the submitted deploy point is
                    # still strong positive evidence that this touch landed.
                    # Never use this fallback for spells or with competing ACK
                    # watches, because either case makes attribution ambiguous.
                    if (not spawn_ack and len(self.ack_watch) == 1
                            and action.target_grid is not None):
                        target_x, target_y = action_world(action)
                        nearby = []
                        for entity in new_own_entities:
                            ex, ey = entity.get('x'), entity.get('y')
                            if not isinstance(ex, (int, float)) or not isinstance(ey, (int, float)):
                                continue
                            distance2 = (float(ex) - target_x) ** 2 + (float(ey) - target_y) ** 2
                            if distance2 <= 2200.0 ** 2:
                                nearby.append((distance2, entity))
                        if nearby:
                            _, nearest = min(nearby, key=lambda row: row[0])
                            spawn_ack = True
                            spawn_entity_id = nearest.get('id')
                            spawn_ack_evidence = 'new_own_entity_near_target'
                # Some probe revisions publish elixir before hand rotation.
                # Treat a sufficiently large, near-immediate cost drop as a
                # fallback ACK.  Natural regeneration cannot satisfy this
                # downward threshold, and the original hand check remains
                # authoritative whenever it is available.
                elixir_changed = False
                # If an earlier accepted action still lacks authoritative hand
                # rotation, a new aggregate elixir drop is not uniquely
                # attributable. Keep the newer action on ACK watch until a
                # hand/spawn signal arrives instead of stealing the old drop.
                allow_elixir_fallback = (
                    len(self.ack_watch) == 1
                    and not self.slot_consume_guards
                )
                if (allow_elixir_fallback and pending.prior_elixir is not None
                        and state.elixir is not None):
                    elixir_changed = (state.tick > pending.sent_tick and
                        float(state.elixir) <= float(pending.prior_elixir) - pending.cost + 0.15)
                if hand_changed or elixir_changed or spawn_ack:
                    self.ack_watch.remove(pending)
                    ack_evidence = (
                        'hand_rotation' if hand_changed else
                        'elixir_cost_drop' if elixir_changed else
                        spawn_ack_evidence or 'new_source_entity'
                    )
                    if hand_changed:
                        self._clear_spend(pending.command_seq)
                    else:
                        # Prevent the same stale native slot/card from being
                        # selected again on the very frame that supplied only
                        # a weak positive ACK. Other slots remain eligible.
                        self._start_slot_consume_guard(
                            pending, now, ack_evidence)
                        # An observed elixir drop already accounts for the
                        # spend in the live resource value. A spawn-only ACK
                        # does not, so keep its short-lived virtual spend until
                        # hand rotation/guard expiry reconciles the snapshot.
                        if elixir_changed:
                            self._clear_spend(pending.command_seq)
                    ack_latency_ms = (now - ack_started_at) * 1000
                    self._record_card_ack_latency(ack_latency_ms)
                    self.log('hand_ack', card=action.card_id, slot=action.hand_slot,
                        command_seq=pending.command_seq,
                        tick=state.tick, latency_ms=ack_latency_ms,
                        decision_to_ack_ms=(now-pending.decision_at)*1000,
                        input_to_ack_ms=ack_latency_ms,
                        input_start_to_ack_ms=(now-pending.sent_at)*1000,
                        ack_timeout_ms=round(ack_timeout_seconds * 1000),
                        outcome='accepted',
                        evidence=ack_evidence,
                        spawn_entity_id=spawn_entity_id)
                    if action.card_id in BINDINGS and action.metadata.get('policy_effective_form_code') == 1:
                        player = next(p for p in state.raw['players'] if p['owner'] == action.owner)
                        row = next((r for r in player.get('card_runtime', []) if r.get('card_id') == action.card_id), {})
                        if pending.prior_evolution_progress == BINDINGS[action.card_id].cycles and row.get('evolution_progress') == 0 and row.get('active_form') == 0:
                            self.log('evolution_ack', card=action.card_id, tick=state.tick,
                                evidence='hand_transition_and_native_cycle_2_to_0',
                                latency_ms=(now-ack_started_at)*1000,
                                input_start_to_ack_ms=(now-pending.sent_at)*1000)
                    # Hand rotation is the authoritative client-side
                    # consume acknowledgement.  A troop can spawn and die
                    # between two observations, and spells have no durable
                    # spawn entity at all; neither case should halt the
                    # entire continuous runner.
                    self.confirmed_actions += 1
                    self._confirmed_for_simulation.append((action, pending.sent_tick,
                                                           pending.command_seq))
                    self._fresh_state_required = True
                    if self._commit_threat_reservation(
                            pending, state, now, confidence='confirmed'):
                        pass
                    else:
                        self.log('post_action_settle_skipped',
                                 command_seq=pending.command_seq,
                                 card=action.card_id,
                                 reason='background_hand_ack')
                elif now - ack_started_at > ack_timeout_seconds:
                    self.ack_watch.remove(pending)
                    # A missing ACK is ambiguous: the touch may have reached
                    # the game while telemetry was late.  Keep its virtual
                    # unit briefly so the next policy frame does not spend a
                    # second card on an apparently untouched threat.  This
                    # does not disable the card or block other slots; a later
                    # authoritative entity/hand update removes it normally.
                    prediction_retained = self._retain_uncertain_prediction(
                        pending.command_seq, now)
                    self.cooldowns[action.hand_slot] = (action.card_id, now + 1)
                    # A completed ADB input with no observed hand transition
                    # means this attempt is ambiguous/missed, but it is not
                    # a transport failure.  Do not replay the same write and
                    # do not halt the match; briefly cool the slot and let
                    # the policy continue with fresh state.
                    self.log('action_missed', card=action.card_id, slot=action.hand_slot,
                             command_seq=pending.command_seq, outcome='unknown',
                             continue_running=True,
                             ack_timeout_ms=round(ack_timeout_seconds * 1000),
                             uncertain_prediction_seconds=config.UNCERTAIN_PREDICTION_SECONDS)
                    # A missed touch leaves the live hand unchanged.  Give
                    # the next probe frame time to arrive and suppress only
                    # this exact card/target briefly; other cards remain
                    # available and the card itself is never disabled.
                    # A fresh frame is required below. Do not add a global
                    # global sleep after a missed ACK: it delays every other
                    # playable card even when the touch worker has finished.
                    # Slot, resource, and identical-action guards still apply.
                    self._recent_misses[self._action_key(action)] = now + 1.0
                    self._fresh_state_required = True
                    # On some hosts the touch is accepted before hand/elixir
                    # telemetry rotates. If this attempted play already had a
                    # concrete defensive threat assignment, keep only that
                    # threat locally reserved instead of freezing every policy
                    # decision behind the global settle/preview gate.
                    if self._commit_threat_reservation(
                            pending, state, now, confidence='uncertain'):
                        pass
                    elif config.ENABLE_MODEL_PREDICTION_OVERLAY and prediction_retained:
                        # The transport completed and the model still sees a
                        # short-lived virtual version of the submitted card.
                        # That prediction, together with reserved elixir and
                        # the exact-action miss cooldown, is enough to avoid a
                        # second stale answer without freezing unrelated play.
                        self.log('post_action_settle_skipped',
                                 command_seq=pending.command_seq,
                                 card=action.card_id,
                                 reason='uncertain_prediction_ack_timeout',
                                 protection_ms=round(
                                     config.UNCERTAIN_PREDICTION_SECONDS * 1000))
                    else:
                        self._arm_post_action_recheck(
                            state, pending, reason='ack_timeout')
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
            # Count only real Android input attempts. Queued actions rejected
            # by live revalidation or expiry never touched the device and must
            # not consume --max-actions.
            self.attempted_actions += 1
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
        self.ack_watch.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.actuator.close()
