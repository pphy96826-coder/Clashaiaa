"""One incremental offline-engine mirror for an online battle.

The mirror is deliberately conservative.  It consumes confirmed card-play
records in chronological order and compares coarse public state summaries
against live telemetry.  Any ambiguity or drift disables the mirror until the
caller starts a new battle; it never silently repairs a divergent simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bridge.full_simulator import FullSimulationUnavailable


@dataclass(frozen=True)
class MirrorAction:
    owner: int
    slot: int | None
    card_id: int
    x: int
    y: int
    tick: int


@dataclass(frozen=True)
class MirrorSummary:
    tick: int
    counts: tuple[tuple[int, int, int], ...]
    hitpoints: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class MirrorDrift:
    reason: str
    live: MirrorSummary | None
    simulated: MirrorSummary | None


class LiveReplayMirror:
    """Maintain one engine rollout alongside one live battle.

    ``engine_factory`` must return an object implementing ``reset(seed)`` and
    ``step(actions, ticks)``.  Injection makes the synchronization logic
    testable without starting a native engine.
    """

    def __init__(self, engine_factory, play_factory, *, owner=0, seed=1,
                 position_tolerance=1500.0, hp_tolerance=250.0):
        self.engine_factory = engine_factory
        self.play_factory = play_factory
        self.owner = owner
        self.seed = seed
        self.position_tolerance = float(position_tolerance)
        self.hp_tolerance = float(hp_tolerance)
        self.engine = None
        self.state = None
        self.last_tick = None
        self.next_action_index = 0
        self.actions: list[MirrorAction] = []
        self._seen_actions = set()
        self.drift: MirrorDrift | None = None
        self._ready_for_model = False
        self.partial_opponent = False
        self._last_live_state = None
        self._checkpoint_handle = None
        self._committed_tick = None
        self._speculative = False

    @property
    def active(self):
        return self.engine is not None and self.state is not None and self.drift is None

    def start(self, initial_live_tick=None, partial_opponent=False):
        self.close()
        try:
            self.engine = self.engine_factory()
            self.state = self.engine.reset(seed=self.seed)
        except Exception as exc:
            self.close()
            raise FullSimulationUnavailable(f"mirror engine reset failed: {exc}") from exc
        self.last_tick = int(self.state.tick)
        self._save_checkpoint()
        # The downloaded engine starts at its first playable boundary (90),
        # while a live observer commonly attaches at tick 0..3.  Keep the
        # mirror alive during that warm-up instead of falsely calling it a
        # tick regression.  It becomes model-eligible once live reaches the
        # same boundary.
        self._ready_for_model = (initial_live_tick is None or
                                 int(initial_live_tick) >= self.last_tick)
        self.partial_opponent = bool(partial_opponent)
        self.next_action_index = 0
        self.actions.clear()
        self._seen_actions.clear()
        self.drift = None
        return self.state

    def close(self):
        if self.engine is not None:
            close = getattr(self.engine, "close", None)
            if close is not None:
                close()
        self.engine = self.state = None
        self.last_tick = None
        self._ready_for_model = False
        self.partial_opponent = False
        self._last_live_state = None
        self._checkpoint_handle = None
        self._committed_tick = None
        self._speculative = False

    def _save_checkpoint(self):
        """Replace the native checkpoint at the latest committed live tick."""
        if self.engine is None:
            return
        # Small test doubles and legacy diagnostic engines may not implement
        # native snapshots.  Keep them usable for non-speculative mirror
        # tests; the real Engine always has this method and fails closed when
        # its installed probe lacks snapshot support.
        if not callable(getattr(self.engine, "snapshot", None)):
            self._committed_tick = int(self.state.tick)
            self._speculative = False
            return
        try:
            if self._checkpoint_handle is not None:
                self.engine.release_snapshot(self._checkpoint_handle)
            receipt = self.engine.snapshot()
            self._checkpoint_handle = int(receipt['handle'])
            self._committed_tick = int(self.state.tick)
            self._speculative = False
        except Exception as exc:
            raise FullSimulationUnavailable(f"native checkpoint failed: {exc}") from exc

    def _rollback_prediction(self):
        if not self._speculative or self._checkpoint_handle is None:
            return
        try:
            self.state = self.engine.restore_snapshot(self._checkpoint_handle)
            self.last_tick = int(self.state.tick)
            self._speculative = False
        except Exception as exc:
            self.invalidate(f"native prediction rollback failed: {exc}")
            raise FullSimulationUnavailable(str(exc)) from exc

    def forecast(self, ticks):
        """Advance a speculative branch; the next sync rolls it back first."""
        if not self.active or self._checkpoint_handle is None or int(ticks) <= 0:
            return self.state
        if self._speculative:
            self._rollback_prediction()
        try:
            self.state = self.engine.step((), ticks=int(ticks)).state
            self.last_tick = int(self.state.tick)
            self._speculative = True
            return self.state
        except Exception as exc:
            self.invalidate(f"native future forecast failed: {exc}")
            raise FullSimulationUnavailable(str(exc)) from exc

    def invalidate(self, reason, live=None):
        self.drift = MirrorDrift(reason, live, self.summary(self.state) if self.state else None)
        return self.drift

    @staticmethod
    def summary(state):
        if state is None:
            return None
        counts = {}
        hitpoints = {}
        for entity in getattr(state, "entities", ()):
            owner = int(getattr(entity, "owner", -1))
            card = int(getattr(entity, "card", 0))
            if card <= 0:
                continue
            counts[(owner, card)] = counts.get((owner, card), 0) + 1
            hp = getattr(entity, "hp", None)
            if hp is not None:
                hitpoints[(owner, card)] = hitpoints.get((owner, card), 0.0) + float(hp)
        return MirrorSummary(int(state.tick),
            tuple(sorted((o, c, n) for (o, c), n in counts.items())),
            tuple(sorted((o, c, round(h, 3)) for (o, c), h in hitpoints.items())))

    @staticmethod
    def live_summary(live_state):
        """Create a comparable summary from BattleState or raw entity mappings."""
        entities = getattr(live_state, "entities", live_state)
        tick = int(getattr(live_state, "tick", 0))
        counts = {}
        hitpoints = {}
        for entity in entities or ():
            if isinstance(entity, dict):
                owner = entity.get("owner")
                card = entity.get("card", entity.get("cardId", 0))
                hp = entity.get("hp", entity.get("hitpoints"))
            else:
                owner = getattr(entity, "owner", None)
                card = getattr(entity, "card", getattr(entity, "card_id", 0))
                hp = getattr(entity, "hp", getattr(entity, "hitpoints", None))
            if owner is None or card is None:
                continue
            key = (int(owner), int(card))
            if key[1] <= 0:
                continue
            counts[key] = counts.get(key, 0) + 1
            if hp is not None:
                hitpoints[key] = hitpoints.get(key, 0.0) + float(hp)
        return MirrorSummary(tick,
            tuple(sorted((o, c, n) for (o, c), n in counts.items())),
            tuple(sorted((o, c, round(h, 3)) for (o, c), h in hitpoints.items())))

    def _check_summary(self, live_state):
        live = self.live_summary(live_state)
        simulated = self.summary(self.state)
        if simulated is None:
            return self.invalidate("mirror has no state", live)
        if live.tick < simulated.tick:
            return self.invalidate("live tick regressed", live)
        live_counts = dict(((o, c), n) for o, c, n in live.counts)
        sim_counts = dict(((o, c), n) for o, c, n in simulated.counts)
        for key, count in live_counts.items():
            # Live visibility can omit enemy entities, so only compare keys
            # that are present in both views.  Own-card counts are public and
            # are strict enough to catch a missed/replayed deployment.
            if key[0] == self.owner and sim_counts.get(key, 0) != count:
                return self.invalidate(f"own entity count drift owner={key[0]} card={key[1]}", live)
        live_hp = dict(((o, c), hp) for o, c, hp in live.hitpoints)
        sim_hp = dict(((o, c), hp) for o, c, hp in simulated.hitpoints)
        for key, hp in live_hp.items():
            if key[0] == self.owner and key in sim_hp and abs(sim_hp[key] - hp) > self.hp_tolerance:
                return self.invalidate(f"own hitpoint drift owner={key[0]} card={key[1]}", live)
        return None

    def sync(self, live_state, actions: Iterable[MirrorAction] = ()):
        """Feed newly confirmed actions and advance to the live tick.

        Actions arriving after their execute tick are rejected as late; replay
        cannot safely insert them into an already advanced native engine.
        """
        if not self.active:
            return None
        # Any state returned by forecast() is a branch, never a new base.
        # Restore the exact committed state before incorporating newly arrived
        # real actions, so a late opponent play is inserted causally.
        self._rollback_prediction()
        live_tick = int(getattr(live_state, "tick", 0))
        self._last_live_state = live_state
        if not self._ready_for_model and live_tick < self.last_tick:
            return None
        if not self._ready_for_model:
            self._ready_for_model = True
        incoming = sorted(tuple(actions), key=lambda action: (action.tick, action.owner, action.slot))
        for action in incoming:
            key = (action.owner, action.slot, action.card_id, action.x, action.y, action.tick)
            if key in self._seen_actions:
                continue
            if action.tick < self.last_tick:
                self.invalidate("confirmed action arrived after mirror advanced", self.live_summary(live_state))
                return None
            self.actions.append(action)
            self._seen_actions.add(key)
        pending = self.actions[self.next_action_index:]
        cursor = self.last_tick
        for action in pending:
            if action.tick < cursor:
                self.invalidate("action chronology is not replayable", self.live_summary(live_state))
                return None
            if action.tick > cursor:
                self.state = self.engine.step((), ticks=action.tick - cursor).state
                cursor = action.tick
            supports_card = getattr(self.engine, 'supports_card', None)
            if callable(supports_card) and not supports_card(action.card_id):
                # The downloaded engine can decode this card in observation
                # payloads but cannot execute its native action.  Keep the
                # authoritative live path running; only speculative replay is
                # disabled, and the caller can retain its kinematic overlay.
                self.invalidate(f'engine action card unsupported: {action.card_id}',
                                self.live_summary(live_state))
                return None
            try:
                transition = self.engine.step((self._play(action),), ticks=1)
            except Exception as exc:
                # A replay response can be an engine error object rather than
                # a State payload (for example when the native slot is no
                # longer valid).  Do not let State.decode/KeyError terminate
                # the live agent; disable only the speculative mirror and
                # continue from authoritative probe telemetry.
                self.invalidate(f"offline engine rejected replay action: {exc}",
                                self.live_summary(live_state))
                return None
            self.state = transition.state
            cursor = int(self.state.tick)
            if not transition.executed or not transition.executed[0]:
                self.invalidate("offline engine rejected confirmed action", self.live_summary(live_state))
                return None
            self.next_action_index += 1
        if live_tick > cursor:
            self.state = self.engine.step((), ticks=live_tick - cursor).state
        self.last_tick = int(self.state.tick)
        self._check_summary(live_state)
        if self.drift is None:
            self._save_checkpoint()
        return self.state if self.active else None

    def _play(self, action):
        try:
            slot = action.slot
            if slot is None:
                player = self.state.players[action.owner]
                card = next((card for card in player.hand if int(card.card) == action.card_id), None)
                if card is None:
                    raise ValueError(f'card {action.card_id} is not in owner {action.owner} hand')
                slot = int(card.slot)
            return self.play_factory(action.owner, slot, action.x, action.y)
        except Exception as exc:
            self.invalidate(f"could not construct engine Play action: {exc}")
            raise FullSimulationUnavailable(str(exc)) from exc

    def model_entities(self, live_state=None):
        """Convert the engine's measured entities to model contract entities."""
        if not self.active or not self._ready_for_model or self.partial_opponent:
            return None
        from native_runner.contracts import EntityStateV1
        result = []
        for entity in getattr(self.state, 'entities', ()):
            card = int(getattr(entity, 'card', 0))
            if card <= 0:
                continue
            hp = getattr(entity, 'hp', None)
            # Dead/despawned native records may remain in the engine history
            # for a few ticks.  They are not live tactical entities and must
            # not trigger another defensive action.
            if hp is not None and float(hp) <= 0:
                continue
            max_hp = getattr(entity, 'max_hp', None)
            result.append(EntityStateV1(
                entity_id=int(getattr(entity, 'id', 0)),
                owner=int(getattr(entity, 'owner', 0)), card_id=card,
                entity_kind=str(getattr(entity, 'entity_type', 'character')),
                position=(float(getattr(entity, 'x', 0)), float(getattr(entity, 'y', 0))),
                hitpoints=None if hp is None else float(hp),
                max_hitpoints=None if max_hp is None else float(max_hp),
                visible=True))
        # Opponent intent is never known ahead of time.  Even when both deck
        # lists are visible, the native rollout can diverge as soon as the
        # opponent plays an unobserved card.  Prefer live opponent entities
        # whenever the probe has them; retain simulated opponents only when
        # telemetry truly has no opponent entities at all.
        live_state = live_state or self._last_live_state
        live_opponents = []
        for entity in getattr(live_state, 'entities', ()) if live_state is not None else ():
                if isinstance(entity, dict):
                    owner = entity.get('owner')
                    card = entity.get('card_id', entity.get('cardId', entity.get('card', 0)))
                    x, y = entity.get('x', 0), entity.get('y', 0)
                    hp = entity.get('hp', entity.get('hitpoints'))
                    max_hp = entity.get('max_hp', entity.get('maxHitpoints'))
                    entity_id = entity.get('id', entity.get('entity_id', 0))
                    kind = entity.get('entity_type', entity.get('entityKind', 'character'))
                else:
                    owner = getattr(entity, 'owner', None)
                    card = getattr(entity, 'card', getattr(entity, 'card_id', 0))
                    x, y = getattr(entity, 'x', 0), getattr(entity, 'y', 0)
                    hp = getattr(entity, 'hp', None)
                    max_hp = getattr(entity, 'max_hp', None)
                    entity_id = getattr(entity, 'id', 0)
                    kind = getattr(entity, 'entity_type', 'character')
                if owner is None or int(owner) == self.owner or int(card or 0) <= 0:
                    continue
                if hp is not None and float(hp) <= 0:
                    continue
                live_opponents.append(EntityStateV1(
                    entity_id=int(entity_id), owner=int(owner), card_id=int(card),
                    entity_kind=str(kind), position=(float(x), float(y)),
                    hitpoints=None if hp is None else float(hp),
                    max_hitpoints=None if max_hp is None else float(max_hp),
                    visible=True))
        # Full replay entities share a single forecast tick. Substituting
        # current-time opponents here makes a board that never existed.
        # Partial replays are rejected above instead of mixing time domains.
        return tuple(result)
