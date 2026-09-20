"""Real-time V4 inference and acknowledged Android card deployment."""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import config
if str(config.FIRSTLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(config.FIRSTLIGHT_DIR))
from bridge.probe_client import ProbeClient
from bridge.actuator import Actuator
from agent.feature_adapter import FeatureAdapter, HOG_26_DECK, TelemetryError
from agent.policy_engine import PolicyEngine
from agent.execution import ActionExecutor
from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK
from live_lifecycle import LiveLifecycle, LifecycleError
from runtime_console import AgentConsole
from bridge.full_simulator import (
    FullBattleSimulator,
    FullSimulationUnavailable,
    PREPARED_NATIVE_ACTION_CARDS,
)
from bridge.live_replay_mirror import LiveReplayMirror, MirrorAction
from bridge.coordinates import action_world


ACTIVE_BATTLE_REATTACH_COOLDOWN_SECONDS = 0.75


def _should_reattach_active_battle(identity, status):
    """Treat a lone probe idle state as non-terminal while a match is active.

    Native state.native_finalized is the authoritative terminal gate. A
    transient UI overlay (for example the emote tray) may briefly make the
    probe report idle even though the battle tick is still live.
    """
    return identity is not None and str(status or '').strip().lower() == 'idle'


class CustomCardDeployAgent:
    def __init__(self, checkpoint_path=None, device_str='cuda:0', *, dry_run=False,
                 account_id=config.LOCAL_ACCOUNT_ID, owner=None, sample=False,
                 oracle_elixir=False, log_path=None, own_tower=None, hero_musketeer=None,
                 observation_profile=config.DEFAULT_OBSERVATION_PROFILE, experimental_origins=False,
                 max_actions=None, defer_model_load=False, allow_partial_tower_attach=False):
        checkpoint = Path(checkpoint_path or config.DEFAULT_CHECKPOINT)
        self.checkpoint_path = checkpoint
        self.device_str = device_str
        self.sample = sample
        self.probe = ProbeClient(port=config.PROBE_PORT, account_id=account_id, owner=owner)
        specialist = 'hog' in str(checkpoint).lower()
        self.adapter = FeatureAdapter(initial_deck=HOG_26_DECK if specialist else None, oracle_elixir=oracle_elixir,
            own_tower=own_tower, hero_musketeer=hero_musketeer, evolution_enabled=hero_musketeer is not False,
            observation_profile=observation_profile, experimental_origins=experimental_origins,
            allow_partial_tower_attach=allow_partial_tower_attach)
        self.adapter.hero_skill_ready = False
        self.actuator = Actuator(guard_hero_hud=hero_musketeer)
        self.log_path = Path(log_path or config.BASE_DIR / 'logs' / (time.strftime('%Y%m%d-%H%M%S') + '.jsonl'))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.log_path.open('a', encoding='utf-8', buffering=1)
        self.executor = ActionExecutor(self.actuator, self.log, dry_run=dry_run,
            on_ability_ack=self.adapter.record_ability_execution, max_actions=max_actions)
        self.log('model_loading', checkpoint=str(checkpoint), device=device_str,
                 observation_profile=observation_profile, experimental_origins=experimental_origins)
        self.engine = None
        self.mirror = None
        self.simulator = None
        if config.ENABLE_FULL_SIMULATION:
            simulator = FullBattleSimulator(root=config.FULL_SIMULATION_ROOT,
                                            port=config.FULL_SIMULATION_PORT)
            try:
                api = simulator._api()
                self.simulator = simulator
                mirror_decks = [None, None]
                self.mirror = LiveReplayMirror(
                    lambda: simulator.create_engine(mirror_decks),
                    lambda owner, slot, x, y: api.Play(owner, slot, x, y),
                    owner=0)
                self._mirror_decks = mirror_decks
            except (OSError, ImportError, FileNotFoundError, FullSimulationUnavailable) as exc:
                self.mirror = None
                self.simulator = None
                print(f'[simulation_disabled] {exc}', flush=True)
        if not defer_model_load:
            self._load_engine()
        self.running = False
        self.dry_run = dry_run

    def _load_engine(self):
        if self.engine is None:
            self.engine = PolicyEngine(self.checkpoint_path, self.device_str, self.sample)

    def _engine(self):
        self._load_engine()
        return self.engine

    @staticmethod
    def _confirmed_live_plays(state):
        """Extract only native-confirmed card plays from rich telemetry."""
        raw = state.raw if isinstance(state.raw, dict) else {}
        rich = raw.get('rich', {}) if isinstance(raw.get('rich', {}), dict) else {}
        events = rich.get('combatEvents', {}).get('events', ())
        result = []
        for event in events if isinstance(events, (list, tuple)) else ():
            if not isinstance(event, dict) or event.get('kind') != 'card_play':
                continue
            try:
                owner = int(event.get('owner', event.get('ownerId')))
                card = int(event.get('playedCardGlobalId', event.get('cardId')))
                tick = int(event.get('tick'))
                x = int(event.get('x', event.get('worldX', event.get('positionX'))))
                y = int(event.get('y', event.get('worldY', event.get('positionY'))))
                slot = int(event.get('handIndex', event.get('slot', 0)))
            except (TypeError, ValueError):
                continue
            result.append(MirrorAction(owner, slot, card, x, y, tick))
        return result

    @staticmethod
    def resolve_checkpoint(value):
        """Resolve a console model name or an explicit checkpoint path."""
        return Path(config.CHECKPOINTS.get(value, Path(value))).expanduser().absolute()

    def _switch_model(self, checkpoint):
        checkpoint = self.resolve_checkpoint(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
        started = time.perf_counter()
        candidate = PolicyEngine(checkpoint, self.device_str, self.sample)
        self.engine = candidate
        self.checkpoint_path = checkpoint
        # Specialist checkpoints keep the existing deck contract; general
        # checkpoints accept the live deck.  The change is applied before the
        # next observation and therefore cannot silently use the old model's
        # deck assumptions.
        self.adapter.configured_deck = HOG_26_DECK if 'hog' in str(checkpoint).lower() else None
        self.log('model_switched', checkpoint=str(checkpoint), load_ms=(time.perf_counter() - started) * 1000)

    def log(self, event, **data):
        record = {'event': event, 'wall_time': time.time(),
                  'observation_profile': self.adapter.observation_profile,
                  'experimental_origins': self.adapter.experimental_origins, **data}
        self.stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        if event == 'decision':
            actions = data.get('actions') or []
            action = actions[0] if actions else {}
            if action.get('kind') == 'play_card':
                detail = (f"PLAY_CARD card={action.get('card_id')} slot={action.get('hand_slot')}"
                          f" grid={action.get('target_grid')} tick={data.get('tick')}")
            else:
                detail = f"{str(action.get('kind', 'WAIT')).upper()} tick={data.get('tick')}"
            print(f'[decision] {detail}', flush=True)
        elif event == 'policy_status':
            print(f"[policy] tick={data.get('tick')} mode={data.get('mode')}"
                  f" elixir={data.get('elixir')} hand={data.get('hand')}", flush=True)
        elif event in {'input_started', 'hand_ack', 'spawn_observed', 'action_rejected',
                       'action_expired', 'action_cancelled', 'battle_terminal', 'model_switched',
                       'battle_start', 'battle_requested', 'battle_attach_requested',
                       'lifecycle_tap', 'lobby_ready', 'execution_halted',
                       'action_missed'}:
            print(f'[{event}] {json.dumps(data, ensure_ascii=False, default=str)}', flush=True)
        elif event not in {'snapshot', 'pipeline_timing'}:
            print(f'[{event}] {json.dumps(data, ensure_ascii=False, default=str)}', flush=True)

    def _validate_action(self, action, state):
        if state.native_finalized:
            return False
        blocked_slots = set(
            self.executor.blocked_slots(state)
        )
        blocked_abilities = set(
            self.executor.blocked_abilities()
        )
        reserved_elixir = float(
            self.executor.reserved_elixir
        )

        # The action being revalidated is already present in executor.pending.
        # Do not let its own queued reservation make it fail its own live
        # legality check.  Keep every *other* unresolved action/guard intact.
        if action.kind.value == 'play_card':
            if action.hand_slot is not None:
                blocked_slots.discard(
                    int(action.hand_slot)
                )
            reserved_elixir = max(
                0.0,
                reserved_elixir
                - float(action.metadata[
                    'policy_effective_cost'
                ]),
            )
        elif action.kind.value == 'activate_ability':
            if action.source_entity is not None:
                blocked_abilities.discard(
                    int(action.source_entity)
                )
            reserved_elixir = max(
                0.0,
                reserved_elixir - 3.0,
            )

        obs = self.adapter.build_observation(
            state,
            tuple(sorted(blocked_slots)),
            reserved_elixir,
            tuple(sorted(blocked_abilities)),
        )
        self.actuator.guard_hero_hud = self.adapter.quality.get('ability_hud_excluded', False)
        if action.kind.value == 'activate_ability':
            own = next(p for p in obs.players if p.owner == state.local_owner)
            return (action.source_entity in obs.action_mask.ability_sources and
                any(a.source_entity == action.source_entity and a.ability_id == action.ability_id
                    for a in own.ability_runtime_states))
        if state.elixir < float(action.metadata['policy_effective_cost']):
            return False
        entry = obs.action_mask.placement_masks.get(str(action.hand_slot))
        if entry is None or action.target_grid is None:
            return False
        if (entry['card_id'] != action.card_id or
            entry['form_code'] != action.metadata.get('policy_effective_form_code', 0) or
            entry['effective_cost'] != action.metadata['policy_effective_cost']):
            return False
        if self.adapter.spell_activation_risk(action, obs.towers):
            self.log('action_rejected', reason='king_tower_activation_risk',
                     card=action.card_id, target_grid=action.target_grid)
            return False
        lane_conflict = self.adapter.defensive_lane_conflict(action, state)
        if lane_conflict is not None:
            self.log('action_rejected', reason='defensive_lane_conflict',
                     card=action.card_id, target_grid=action.target_grid,
                     **lane_conflict)
            return False
        x, y = action.target_grid
        return 0 <= x < 18 and 0 <= y < 32 and bool(entry['row_major'][y][x])

    def run(self, seconds=0, once=False, start_battle=False, attach_active=False, *,
            continuous=False, console=None, max_matches=None, lifecycle_calibration=None):
        if self.probe.known_account_id is None and self.probe.owner_override is None:
            raise ValueError('Set account_id in settings.local.json; see tools/inspect_players.py')
        if not self.dry_run and not config.CALIBRATION_VERIFIED:
            raise ValueError('Verify ground coordinates, then set calibration_verified=true in settings.local.json')
        self.running = True
        started = time.perf_counter()
        identity, last_seen_tick, last_decision_tick = None, -1, -1
        last_live = started
        last_message = ''
        ended = None
        suspended = False
        wait_since = None
        paused = False
        pending_model = None
        model_warmup_pending = False
        simulation_entities = None
        auto_emotes = False
        next_emote_at = 0.0
        last_prelock_fast_key = None
        last_prelock_diag_key = None
        last_prelock_conflict_key = None
        last_active_reattach = 0.0
        match_count = 0
        lifecycle = None
        if continuous:
            lifecycle = LiveLifecycle(self.probe, log=self.log,
                                      calibration_path=lifecycle_calibration)
        if console is not None:
            console.start()
        self.actuator.prepare()
        if self.adapter.hero_mode is not False:
            from bridge.hero_execution import ensure_ability_calibration
            try:
                ability_point, generated = ensure_ability_calibration(
                    config.ABILITY_CALIBRATION_PATH, self.actuator.size)
                self.adapter.hero_skill_ready = True
                self.log('ability_input_ready',
                         screen=list(ability_point),
                         calibration=str(config.ABILITY_CALIBRATION_PATH),
                         generated=generated)
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                self.log('ability_input_disabled', reason=str(exc))
        if lifecycle is not None:
            if attach_active:
                lifecycle.attach_active()
            elif start_battle:
                lifecycle.start_battle()
            else:
                # Continuous mode started without an explicit attach is a
                # lobby workflow.  Mid-battle takeover remains explicit via
                # --attach-active or the console's attach command.
                lifecycle.start_battle()
        elif start_battle:
            self.actuator.tap(*config.LOBBY_BATTLE_BTN)
            self.probe.arm_live_context()
        elif attach_active:
            # Explicit operator intent is required for a mid-battle attach;
            # a read-only launch in the lobby must remain unarmed.
            self.probe.attach_live_context()
        # An explicit mid-battle attach must win the race against model
        # initialization.  The runner is already bound to the live context
        # before the checkpoint is loaded, so short battles are not missed.
        self._load_engine()
        self.log('ready', dry_run=self.dry_run, checkpoint=str(self._engine().checkpoint_path),
                 stage=self._engine().meta.get('training_stage'), account_id=self.probe.known_account_id,
                 form_detection='auto' if self.adapter.hero_mode is None else 'explicit',
                 observation_profile=self.adapter.observation_profile,
                 experimental_origins=self.adapter.experimental_origins,
                 hero_skill_input_ready=self.adapter.hero_skill_ready,
                 size=self.actuator.size, log=str(self.log_path))
        def console_say(message):
            if console is not None:
                console.say(message)
            else:
                print(f'[console] {message}', flush=True)

        def process_console_commands(current_state):
            nonlocal paused, pending_model, model_warmup_pending, auto_emotes, next_emote_at
            if console is None:
                return
            for command in console.poll():
                name, args = command.name, command.args
                if name in {'help', '?'}:
                    console_say(AgentConsole.HELP)
                elif name == 'pause':
                    paused = True
                    self.executor.pause()
                    console_say('policy paused; observation and in-flight outcome checks continue')
                elif name == 'resume':
                    paused = False
                    console_say('policy resumed')
                elif name == 'emote':
                    if lifecycle is None:
                        console_say('emote requires continuous mode')
                        continue
                    if not args or args[0].lower() == 'auto':
                        auto_emotes = True
                        next_emote_at = time.monotonic()
                        console_say('automatic emotes enabled')
                        continue
                    if args[0].lower() == 'off':
                        auto_emotes = False
                        console_say('automatic emotes disabled')
                        continue
                    if current_state is None or not current_state.in_battle:
                        console_say('emote requires an active battle')
                        continue
                    try:
                        index = int(args[0])
                        if self.executor.pending or self.executor.future is not None:
                            console_say('emote delayed: card action is in flight')
                            continue
                        lifecycle.send_emote(index)
                    except (ValueError, LifecycleError) as exc:
                        console_say(f'emote failed: {exc}')
                elif name == 'model':
                    if len(args) != 1:
                        console_say('usage: model <hog26|hog26_proactive|general|il|active_il|checkpoint-path>')
                        continue
                    if pending_model is not None:
                        console_say(f'model switch already pending: {pending_model}')
                        continue
                    pending_model = args[0]
                    paused = True
                    console_say(f'model switch queued: {args[0]} (waits for current action outcome)')
                elif name == 'prediction':
                    if len(args) != 2 or args[0].lower() != 'shadow' or args[1].lower() not in ('on', 'off'):
                        console_say('usage: prediction shadow <on|off>')
                        continue
                    if self.executor.pending or self.executor.future is not None:
                        console_say('prediction switch delayed: card action is in flight')
                        continue
                    enabled = args[1].lower() == 'on'
                    config.ENABLE_MODEL_PREDICTION_OVERLAY = enabled
                    if not enabled:
                        self.executor.predictions.clear()
                    self.log('prediction_mode_changed', mode='shadow', enabled=enabled)
                    console_say('局面预测 shadow ' + ('已开启' if enabled else '已关闭'))
                elif name == 'status':
                    console_say(
                        f"model={self.checkpoint_path.name} paused={paused} "
                        f"prediction_shadow={config.ENABLE_MODEL_PREDICTION_OVERLAY} "
                        f"battle={'active' if current_state is not None else 'idle'} "
                        f"tick={getattr(current_state, 'tick', None)} "
                        f"confirmed={self.executor.confirmed_actions} "
                        f"pending={len(self.executor.pending)} "
                        f"ack_watch={len(self.executor.ack_watch)} matches={match_count}"
                    )
                elif name == 'recover':
                    if lifecycle is None:
                        console_say('recover requires --continuous')
                    elif current_state is not None and current_state.raw.get('in_battle'):
                        console_say('请先回到大厅再恢复运行链路；当前不会重启对局')
                    else:
                        try:
                            result = lifecycle.recover_runtime()
                            console_say('运行链路已恢复' + ('（已重启游戏进程）' if result.get('restarted') else ''))
                        except LifecycleError as exc:
                            console_say(f'运行链路恢复失败: {exc}')
                elif name == 'start':
                    if lifecycle is None:
                        console_say('start requires --continuous')
                    elif identity is not None:
                        console_say('a battle is already active')
                    else:
                        try:
                            lifecycle.start_battle()
                            console_say('new battle requested')
                        except LifecycleError as exc:
                            console_say(str(exc))
                elif name == 'attach':
                    if lifecycle is None:
                        console_say('attach requires --continuous')
                    elif identity is not None:
                        console_say('already attached to a battle')
                    else:
                        try:
                            lifecycle.attach_active()
                            console_say('mid-battle attach requested')
                        except LifecycleError as exc:
                            console_say(str(exc))
                elif name in {'stop', 'quit', 'exit'}:
                    self.running = False
                    console_say('stopping safely; no new input will be sent')
                else:
                    console_say(f'unknown command: {name}; type help')

            if pending_model is not None and not self.executor.pending and self.executor.future is None:
                requested = pending_model
                pending_model = None
                try:
                    self._switch_model(requested)
                    model_warmup_pending = identity is not None
                    paused = False
                except (OSError, ValueError, RuntimeError) as exc:
                    paused = True
                    console_say(f'model switch failed; policy remains paused: {exc}')

        try:
            while self.running and (not seconds or time.perf_counter() - started < seconds):
                process_console_commands(None)
                state = self.probe.get_live_battle_state()
                # A simulation frame is valid for exactly one live probe
                # sample. Never let a previous speculative board leak across
                # a timeout, reconnect, or lifecycle transition.
                simulation_entities = None
                now = time.perf_counter()
                process_console_commands(state)
                if state is None:
                    self.executor.poll(None, self._validate_action)

                    # A transient UI overlay can briefly make the probe report
                    # idle even though the battle itself is still active. Do
                    # not demote an episode that already has an identity; the
                    # native finalized contract below remains the only normal
                    # terminal gate. Re-attaching is read-only and lets the
                    # telemetry lane reacquire the same episode.
                    if _should_reattach_active_battle(
                            identity, self.probe.last_status):
                        if (now - last_active_reattach
                                >= ACTIVE_BATTLE_REATTACH_COOLDOWN_SECONDS):
                            try:
                                self.probe.attach_live_context()
                                self.log(
                                    'battle_idle_guard',
                                    last_tick=last_seen_tick,
                                    status=self.probe.last_status,
                                    action='reattach_live_context',
                                    telemetry_age_ms=round(
                                        max(0.0, now - last_live) * 1000.0, 1),
                                )
                            except (OSError, RuntimeError, ValueError) as exc:
                                self.log(
                                    'battle_idle_guard_error',
                                    last_tick=last_seen_tick,
                                    status=self.probe.last_status,
                                    error=str(exc),
                                )
                            last_active_reattach = now
                        message = self.probe.last_error or self.probe.last_status
                        if message != last_message:
                            self.log('waiting', status=message,
                                     guarded_active_battle=True)
                            last_message = message
                        time.sleep(.05)
                        continue

                    if identity is not None and not suspended and now-last_live > config.STALE_SECONDS:
                        self.log('telemetry_paused', last_tick=last_seen_tick, status=self.probe.last_status)
                        self.executor.pause()
                        # A socket/tick stall is not proof of a new match.
                        # Retain measured towers and recurrent history so a
                        # midgame resume does not require six surviving towers.
                        suspended = True
                    message = self.probe.last_error or self.probe.last_status
                    if message != last_message:
                        self.log('waiting', status=message)
                        last_message = message
                    time.sleep(.05)
                    continue
                last_live = now
                if once and identity is not None and (state.identity != identity or state.tick < last_seen_tick):
                    self.log('battle_replaced', last_tick=last_seen_tick, observed_tick=state.tick,
                             reason='new_episode_after_single_battle_started')
                    self.executor.end_battle()
                    break
                if suspended:
                    self.log('telemetry_resumed', tick=state.tick,
                             same_episode=state.identity == identity and state.tick >= last_seen_tick)
                    suspended = False
                if ended is not None and state.identity == ended[0] and state.tick >= ended[1]:
                    time.sleep(.05)
                    continue
                if state.native_finalized:
                    if identity is None:
                        # Launching at the previous result screen must still
                        # wait for the next live battle, including --once.
                        ended = (state.identity, state.tick)
                        self.log('waiting', status='finished_battle_waiting_for_next')
                        time.sleep(.05)
                        continue
                    winner = state.native_winner
                    self.log('battle_terminal', tick=state.tick, crowns=state.crowns, owner=state.local_owner,
                             evidence='native_world_finalized', winner=winner,
                             result='unknown' if winner is None else 'win' if winner == state.local_owner else 'loss')
                    self.executor.end_battle()
                    ended = (state.identity, state.tick)
                    identity = None
                    if once:
                        break
                    if lifecycle is not None:
                        try:
                            lifecycle.return_to_lobby(
                                terminal_confirmed=True)
                        except LifecycleError as exc:
                            self.log('lifecycle_error', phase='return-to-lobby', error=str(exc))
                            break
                        if max_matches is not None and match_count >= max_matches:
                            self.log('continuous_limit_reached', max_matches=max_matches,
                                     matches=match_count)
                            break
                        try:
                            lifecycle.start_battle(ensure_lobby=False)
                        except LifecycleError as exc:
                            self.log('lifecycle_error', phase='next-battle', error=str(exc))
                            break
                        last_seen_tick = -1
                        last_decision_tick = -1
                        last_prelock_fast_key = None
                        last_prelock_diag_key = None
                        last_prelock_conflict_key = None
                        ended = None
                        suspended = False
                        wait_since = None
                    time.sleep(.05)
                    continue
                if state.identity != identity or state.tick < last_seen_tick:
                    self.executor.reset()
                    try:
                        self.adapter.reset_match(state, f'live-{time.time_ns()}')
                    except TelemetryError as exc:
                        if str(exc) != last_message:
                            self.log('telemetry_rejected', error=str(exc))
                            last_message = str(exc)
                        time.sleep(.25)
                        continue
                    self._engine().reset()
                    simulation_entities = None
                    if getattr(self, 'mirror', None) is not None:
                        # A replay mirror must observe the opening boundary;
                        # attaching after the battle has advanced would make
                        # late card plays impossible to insert safely.
                        standard_cards = (26000000, 26000001, 26000003, 26000005,
                                          26000014, 26000018, 28000000, 28000001)
                        # The downloaded engine can only replay its prepared
                        # native card catalog.  Do this check before starting
                        # a mirror: a 2.6 deck otherwise boots successfully,
                        # then drifts on the first Log/Skeletons action and
                        # wastes one live decision on a doomed replay.
                        native_catalog_compatible = (
                            not set(state.deck_cards) - PREPARED_NATIVE_ACTION_CARDS)
                        deck_compatible = (native_catalog_compatible and
                                           (not config.FULL_SIMULATION_REQUIRE_STANDARD_DECK or
                                            tuple(sorted(state.deck_cards)) == standard_cards))
                        partial_opponent = False
                        raw_players = state.raw.get('players', ())
                        live_decks = []
                        for player in raw_players if isinstance(raw_players, (list, tuple)) else ():
                            if not isinstance(player, dict):
                                continue
                            cards = []
                            for card in player.get('deck', ()):
                                if not isinstance(card, dict):
                                    cards = []
                                    break
                                value = card.get('cardId', card.get('id'))
                                if value is None:
                                    cards = []
                                    break
                                cards.append(int(value))
                            live_decks.append(tuple(cards))
                        live_decks = tuple(live_decks)
                        if len(live_decks) == 2 and all(len(deck) == 8 for deck in live_decks):
                            self._mirror_decks[:] = live_decks
                            deck_compatible = native_catalog_compatible
                        elif (config.FULL_SIMULATION_PARTIAL_OPPONENT and
                              native_catalog_compatible and
                              len(state.deck_cards) == 8 and len(set(state.deck_cards)) == 8):
                            # The local deck is observable and safe to use;
                            # the opponent deck is hidden by the live game.
                            # Keep a deterministic placeholder only so the
                            # offline engine can boot.  The model-input merge
                            # below preserves live opponent entities.
                            decks = [tuple(standard_cards), tuple(standard_cards)]
                            decks[state.local_owner] = tuple(state.deck_cards)
                            self._mirror_decks[:] = decks
                            deck_compatible = native_catalog_compatible
                            partial_opponent = True
                        if state.tick <= 120 and deck_compatible and not partial_opponent:
                            try:
                                self.mirror.owner = state.local_owner
                                self.mirror.start(initial_live_tick=state.tick,
                                                  partial_opponent=partial_opponent)
                                self.log('simulation_mirror_started', tick=state.tick,
                                         engine_tick=self.mirror.last_tick,
                                         model_ready=bool(self.mirror._ready_for_model),
                                         owner=state.local_owner,
                                         partial_opponent=partial_opponent)
                            except FullSimulationUnavailable as exc:
                                self.mirror = None
                                self.log('simulation_mirror_disabled', reason=str(exc))
                        else:
                            live_deck_lengths = [len(deck) for deck in live_decks]
                            self.log('simulation_mirror_disabled',
                                     reason=('opponent replay unavailable; partial mirror cannot predict combat'
                                             if partial_opponent else
                                             'deck contains cards outside the downloaded engine catalog'
                                             if not deck_compatible else 'battle started before mirror boundary'),
                                     tick=state.tick, deck=state.deck_cards,
                                     configured_standard_gate=bool(config.FULL_SIMULATION_REQUIRE_STANDARD_DECK),
                                     partial_opponent_enabled=bool(config.FULL_SIMULATION_PARTIAL_OPPONENT),
                                     live_deck_lengths=live_deck_lengths,
                                     live_decks_complete=(len(live_decks) == 2 and
                                                          all(length == 8 for length in live_deck_lengths)))
                            self.mirror.close()
                    self.log('battle_start', tick=state.tick, owner=state.local_owner,
                             deck=state.deck_cards, quality=self.adapter.quality)
                    match_count += 1
                    if not self._engine().warmed_up:
                        warm_batch, _ = self.adapter.tensorize(state)
                        self._engine().warmup(warm_batch)
                        self.log('model_warmed_up', tick=state.tick)
                        # The next iteration refreshes the snapshot before
                        # selecting anything after potentially slow CUDA init.
                        identity = state.identity
                        last_seen_tick = state.tick
                        last_decision_tick = -1
                        continue
                    identity = state.identity
                    last_decision_tick = -1
                    last_prelock_fast_key = None
                    last_prelock_diag_key = None
                    last_prelock_conflict_key = None
                    ended = None
                    wait_since = None
                last_seen_tick = state.tick
                try:
                    self.adapter.observe(state)
                    # Reconcile completed input before doing any synchronous
                    # offline work, so mirror RPCs cannot delay touch ACKs.
                    self.executor.poll(state, self._validate_action)

                    # Detect a tower-lock threat before ordinary cadence gates.
                    # A threat already covered by an active reservation is
                    # excluded so an unrelated second-lane threat can still
                    # become the emergency candidate immediately.
                    raw_prelock = self.adapter.prelock_context(
                        state,
                        self.executor.end_to_end_latency_ms,
                    )

                    reserved_prelock_ids = (
                        self.executor.prelock_reserved_threat_ids(state)
                    )

                    raw_prelock_conflict = bool(
                        raw_prelock is not None
                        and int(raw_prelock['enemy_id'])
                            in reserved_prelock_ids
                    )

                    if raw_prelock_conflict:
                        conflict_key = (
                            int(raw_prelock['enemy_id']),
                            str(raw_prelock['state']),
                            str(raw_prelock['reason']),
                        )

                        if conflict_key != last_prelock_conflict_key:
                            self.log(
                                'prelock_threat',
                                tick=state.tick,
                                prelock_state=raw_prelock['state'],
                                reason=raw_prelock['reason'],
                                enemy_id=raw_prelock['enemy_id'],
                                enemy_card_id=raw_prelock[
                                    'enemy_card_id'],
                                tower_id=raw_prelock['tower_id'],
                                lane=raw_prelock['lane'],
                                distance=round(
                                    raw_prelock['distance'], 1),
                                distance_to_lock=round(
                                    raw_prelock[
                                        'distance_to_lock'], 1),
                                attack_range=(
                                    round(
                                        raw_prelock[
                                            'attack_range'], 1)
                                    if raw_prelock[
                                        'attack_range'] is not None
                                    else None
                                ),
                                closing_speed_per_tick=round(
                                    raw_prelock[
                                        'closing_speed_per_tick'], 2),
                                lock_eta_ms=round(
                                    raw_prelock['lock_eta_ms'], 1),
                                latest_safe_response_ms=round(
                                    raw_prelock[
                                        'latest_safe_response_ms'], 1),
                                reservation_conflict=True,
                                fast_path=False,
                            )

                            last_prelock_conflict_key = conflict_key

                        prelock = self.adapter.prelock_context(
                            state,
                            self.executor.end_to_end_latency_ms,
                            excluded_enemy_ids=reserved_prelock_ids,
                        )
                    else:
                        prelock = raw_prelock
                        last_prelock_conflict_key = None

                    prelock_conflict = (
                        self.executor.prelock_reservation_conflict(
                            prelock, state)
                        if prelock is not None
                        else False
                    )

                    prelock_key = (
                        (
                            int(prelock['enemy_id']),
                            str(prelock['state']),
                        )
                        if prelock is not None
                        else None
                    )

                    if prelock is None:
                        last_prelock_fast_key = None
                        last_prelock_diag_key = None

                    prelock_fast_path = bool(
                        prelock is not None
                        and not prelock_conflict
                        and prelock_key != last_prelock_fast_key
                        and not paused
                        and pending_model is None
                        and not model_warmup_pending
                        and not self.executor.pending
                        and self.executor.future is None
                    )

                    if prelock_fast_path:
                        prelock_fast_path = (
                            self.executor.interrupt_post_action_recheck(
                                prelock, state)
                        )

                    if prelock is not None:
                        diag_key = (
                            int(prelock['enemy_id']),
                            str(prelock['state']),
                            str(prelock['reason']),
                            bool(prelock_conflict),
                        )

                        if diag_key != last_prelock_diag_key:
                            self.log(
                                'prelock_threat',
                                tick=state.tick,
                                prelock_state=prelock['state'],
                                reason=prelock['reason'],
                                enemy_id=prelock['enemy_id'],
                                enemy_card_id=prelock[
                                    'enemy_card_id'],
                                tower_id=prelock['tower_id'],
                                lane=prelock['lane'],
                                distance=round(
                                    prelock['distance'], 1),
                                distance_to_lock=round(
                                    prelock['distance_to_lock'], 1),
                                attack_range=(
                                    round(prelock['attack_range'], 1)
                                    if prelock[
                                        'attack_range'] is not None
                                    else None
                                ),
                                closing_speed_per_tick=round(
                                    prelock[
                                        'closing_speed_per_tick'], 2),
                                lock_eta_ms=round(
                                    prelock['lock_eta_ms'], 1),
                                latest_safe_response_ms=round(
                                    prelock[
                                        'latest_safe_response_ms'], 1),
                                reservation_conflict=prelock_conflict,
                                fast_path=prelock_fast_path,
                            )

                            last_prelock_diag_key = diag_key

                    normal_policy_due = (
                        state.tick
                        >= last_decision_tick
                        + config.DECISION_TICKS
                    )

                    policy_due = (
                        normal_policy_due
                        or prelock_fast_path
                    )
                    self._last_simulation_forecast = None
                    mirror = getattr(self, 'mirror', None)
                    if mirror is not None and mirror.active:
                        confirmed_plays = self._confirmed_live_plays(state)
                        # Hand ACK is available before rich card_play events
                        # on some frames.  Feed that accepted action into the
                        # mirror immediately so the next policy input includes
                        # its real deployment/effect timeline; the later rich
                        # event is deduplicated by MirrorAction's key.
                        for action, sent_tick, command_seq in self.executor.consume_confirmed_for_simulation():
                            if action.target_grid is not None:
                                x, y = action_world(action)
                                # A touch-start tick is only an input timestamp;
                                # it is not proof that the game consumed the
                                # card.  Use the hand-ACK observation tick as
                                # the causal insertion point, then let the
                                # engine apply its normal deployment timeline.
                                # ``sent_tick`` is retained only as a lower
                                # bound for diagnostics/ordering.
                                # ``mirror.last_tick`` may still refer to the
                                # previous speculative forecast at this point
                                # (sync() rolls it back immediately afterward),
                                # so it must never move an ACK into the future.
                                replay_tick = max(int(sent_tick), int(state.tick))
                                confirmed_plays.append(MirrorAction(
                                    action.owner, action.hand_slot, action.card_id,
                                    int(round(x)), int(round(y)), replay_tick))
                                self.log('simulation_action_replayed',
                                         command_seq=command_seq,
                                         card=action.card_id,
                                         ack_tick=int(state.tick),
                                         sent_tick=int(sent_tick),
                                         replay_tick=replay_tick,
                                         deployment_timing='engine_native')
                        mirror.sync(state, confirmed_plays)
                        # Keep the engine ahead of the live frame on a native
                        # checkpointed branch.  The next sync restores the
                        # committed frame before applying any newly observed
                        # opponent action.
                        if (mirror.active and not paused and pending_model is None
                                and not self.executor.pending and self.executor.future is None
                                and state.tick >= FIRST_POLICY_DECISION_TICK
                                and policy_due
                                and not self.executor.decision_blocked(state)):
                            # Use the same time basis as pending virtual-card
                            # predictions: measured end-to-end input latency
                            # first, then the configured future horizon.
                            latency_ticks = max(0, round(
                                self.executor.end_to_end_latency_ms /
                                (config.TICK_SECONDS * 1000.0)))
                            forecast_ticks = latency_ticks + config.PREDICTION_HORIZON_TICKS
                            mirror.forecast(forecast_ticks)
                            self._last_simulation_forecast = {
                                'latency_ticks': latency_ticks,
                                'horizon_ticks': config.PREDICTION_HORIZON_TICKS,
                                'forecast_ticks': forecast_ticks,
                            }
                        simulation_entities = mirror.model_entities(state)
                        if mirror.drift is not None:
                            self.log('simulation_mirror_drift', reason=mirror.drift.reason,
                                     tick=state.tick)
                            simulation_entities = None
                    if (config.ENABLE_FULL_SIMULATION and mirror is not None
                            and mirror.active and simulation_entities is None
                            and mirror._ready_for_model):
                        # The mirror is advisory.  Never stall the live agent
                        # while a native forecast round-trip is pending; use
                        # the authoritative frame plus the local kinematic
                        # overlay for this decision and let the mirror retry
                        # on the next observation.
                        self.log('simulation_waiting', tick=state.tick,
                                 engine_tick=mirror.last_tick)
                    # Do not infer match termination from a destroyed
                    # princess tower (or from a transient/misbound tower
                    # record).  A single crown is still an active battle and
                    # the model must keep defending/playing.  The authoritative
                    # ``native_finalized`` branch above is the only terminal
                    # gate; it is produced by the battle result contract and
                    # is also used to dismiss the result screen safely.
                    towers = self.adapter._build_towers(state)
                    # poll() may have acknowledged or conclusively expired
                    # the previous input.  The mirror/entities computed above
                    # belong to the pre-poll frame, so never run policy on
                    # that stale simulation.  The next loop obtains a fresh
                    # probe frame and synchronizes the mirror first.
                    if self.executor.consume_fresh_state_required():
                        time.sleep(.001)
                        continue
                    if (auto_emotes and lifecycle is not None and not self.executor.pending
                            and not self.executor.ack_watch
                            and self.executor.future is None and time.monotonic() >= next_emote_at):
                        try:
                            lifecycle.send_emote(random.randrange(8))
                            # Keep emotes visibly periodic while avoiding a
                            # chat tap on every decision window.
                            next_emote_at = time.monotonic() + random.uniform(20.0, 35.0)
                        except LifecycleError as exc:
                            auto_emotes = False
                            self.log('emote_disabled', reason=str(exc))
                    if self.executor.fault:
                        # A timeout or incomplete execution outcome is not a
                        # safe reason to continue.  Stop this agent instance
                        # without issuing another touch; the unresolved
                        # command is classified as UNKNOWN in the log.
                        self.log('execution_halted', reason=self.executor.fault,
                                 tick=state.tick)
                        break
                    if self.executor.action_budget_exhausted:
                        # Do not keep running policy after the requested number
                        # of real action attempts. Let background ACK watches
                        # reconcile for at most their normal timeout, then stop
                        # with complete outcome logs.
                        if self.executor.action_budget_settled:
                            self.log('action_budget_reached',
                                     max_actions=self.executor.max_actions,
                                     attempted_actions=self.executor.attempted_actions,
                                     confirmed_actions=self.executor.confirmed_actions,
                                     tick=state.tick)
                            break
                        time.sleep(.005)
                        continue
                    if model_warmup_pending:
                        warm_batch, _ = self.adapter.tensorize(
                            state, self.executor.blocked_slots(state),
                            self.executor.reserved_elixir, self.executor.blocked_abilities(),
                            self.executor.model_predictions(state))
                        self._engine().warmup(warm_batch)
                        self._engine().reset()
                        model_warmup_pending = False
                        self.log('model_warmed_up', tick=state.tick, reason='model_switch')
                        last_decision_tick = state.tick
                        time.sleep(.01)
                        continue
                    if paused or pending_model is not None:
                        time.sleep(.01)
                        continue
                    # Touch input remains strictly serial, but ACK
                    # reconciliation is background work. Once the touch worker
                    # completes, ack_watch keeps the old slot/cost protected
                    # while policy may react with another legal slot on a
                    # fresh authoritative frame.
                    if (state.tick >= FIRST_POLICY_DECISION_TICK
                            and policy_due
                            and not self.executor.pending
                            and self.executor.future is None
                            and not self.executor.decision_blocked(state)):
                        pipeline_start = time.perf_counter()
                        batch, obs = self.adapter.tensorize(state, self.executor.blocked_slots(state), self.executor.reserved_elixir,
                            self.executor.blocked_abilities(), self.executor.model_predictions(state),
                            simulation_entities=simulation_entities)
                        decoded, inference_ms = self._engine().decide(
                            batch,
                            obs,
                            self.adapter,
                            emergency=prelock_fast_path,
                        )
                        overflow_fallback = None

                        if (
                            all(
                                action.kind.value == 'wait'
                                for action in decoded.actions
                            )
                            and obs.action_mask.reasons.get(
                                'defense_overflow_forced')
                        ):
                            overflow_fallback = (
                                self.adapter.defense_overflow_fallback(
                                    state, obs)
                            )

                        if overflow_fallback is not None:
                            decoded = type(decoded)(
                                owner=decoded.owner,
                                actions=(overflow_fallback,),
                            )

                            self.log(
                                'defense_overflow_fallback',
                                tick=state.tick,
                                elixir=state.elixir,
                                slot=overflow_fallback.hand_slot,
                                card=overflow_fallback.card_id,
                                target_grid=overflow_fallback.target_grid,
                                mode=overflow_fallback.metadata.get(
                                    'defense_overflow_mode'),
                                safe_slots=obs.action_mask.reasons.get(
                                    'defense_overflow_safe_slots'),
                                cannon_prebuild_lead_ticks=(
                                    obs.action_mask.reasons.get(
                                        'cannon_prebuild_lead_ticks')
                                ),
                            )

                        if (
                            overflow_fallback is None
                            and all(
                                action.kind.value == 'wait'
                                for action in decoded.actions
                            )
                            and obs.action_mask.reasons.get(
                                'neutral_overflow_active')
                        ):
                            neutral_overflow_fallback = (
                                self.adapter.neutral_overflow_fallback(
                                    state, obs)
                            )

                            if neutral_overflow_fallback is not None:
                                decoded = type(decoded)(
                                    owner=decoded.owner,
                                    actions=(
                                        neutral_overflow_fallback,
                                    ),
                                )
                                self.log(
                                    'neutral_overflow_fallback',
                                    tick=state.tick,
                                    elixir=state.elixir,
                                    slot=(
                                        neutral_overflow_fallback
                                        .hand_slot
                                    ),
                                    card=(
                                        neutral_overflow_fallback
                                        .card_id
                                    ),
                                    target_grid=(
                                        neutral_overflow_fallback
                                        .target_grid
                                    ),
                                    mode=(
                                        neutral_overflow_fallback
                                        .metadata.get(
                                            'neutral_overflow_mode'
                                        )
                                    ),
                                )

                        adjusted = []
                        for action in decoded.actions:
                            # A valid mirror already contains the target at
                            # the compensated future tick. Applying the
                            # kinematic Log lead on top of it would double
                            # compensate and place the Log behind the unit.
                            # The policy already saw the compensated board.
                            # Preserve its placement; never shift Log afterward.
                            corrected, detail = action, None
                            adjusted.append(corrected)
                            if detail is not None:
                                self.log('target_lead_applied', card=action.card_id,
                                         tick=state.tick, **detail)
                        if tuple(adjusted) != tuple(decoded.actions):
                            # DecodedActionSequenceV4 only carries owner and
                            # actions.  Reconstructing it with a nonexistent
                            # ``metadata`` field used to terminate the whole
                            # runner whenever target compensation changed an
                            # action (most often Log).  Preserve the contract
                            # explicitly so a correction cannot become a
                            # process-level fatal error.
                            decoded = type(decoded)(owner=decoded.owner,
                                                    actions=tuple(adjusted))
                        waiting = all(a.kind.value == 'wait' for a in decoded.actions)
                        # Keep the live log focused on actionable decisions.
                        # The full snapshot remains in memory for validation,
                        # but serializing it on every policy turn adds disk
                        # I/O and JSON work without helping execution.
                        if not waiting:
                            self.log('decision', tick=state.tick, inference_ms=inference_ms,
                                pipeline_ms=(time.perf_counter()-pipeline_start)*1000,
                                probe_query_ms=self.probe.last_query_ms,
                                frame_to_policy_start_ms=max(
                                    0.0, (pipeline_start-state.received_at)*1000),
                                tick_gap=None if last_decision_tick < 0 else state.tick-last_decision_tick,
                                prelock_fast_path=bool(prelock_fast_path),
                                emergency_policy_turn=bool(prelock_fast_path),
                                simulation_used=simulation_entities is not None,
                                simulation_entity_count=(len(simulation_entities)
                                                          if simulation_entities is not None else 0),
                                simulation_tick=(mirror.last_tick
                                                 if simulation_entities is not None and mirror is not None else None),
                                simulation_latency_ticks=(getattr(self, '_last_simulation_forecast', {})
                                                          .get('latency_ticks')
                                                          if simulation_entities is not None else None),
                                simulation_horizon_ticks=(getattr(self, '_last_simulation_forecast', {})
                                                          .get('horizon_ticks')
                                                          if simulation_entities is not None else None),
                                simulation_forecast_ticks=(getattr(self, '_last_simulation_forecast', {})
                                                           .get('forecast_ticks')
                                                           if simulation_entities is not None else None),
                                actions=[a.to_dict() for a in decoded.actions],
                                hand=state.hand_cards, elixir=state.elixir,
                                playable_slots=obs.action_mask.hand_slots,
                                mask_reasons={**dict(obs.action_mask.reasons),
                                              'slot_reasons': dict(obs.action_mask.reasons['slot_reasons'])},
                                legal_candidates=int(batch.candidates.mask.sum()))
                        wait_since = (state.tick if wait_since is None else wait_since) if waiting else None
                        last_decision_tick = state.tick
                        if prelock_fast_path:
                            last_prelock_fast_key = prelock_key
                        # After any card consume (or an ambiguous touch), use
                        # a settled frame as a no-write preview.  A follow-up
                        # card is sent only if it remains the same highest
                        # scoring action on the next fresh policy turn.
                        if self.executor.post_action_recheck(decoded, state):
                            self.executor.submit(decoded, state)
                except TelemetryError as exc:
                    self.log('telemetry_rejected', tick=state.tick, error=str(exc))
                # Refresh telemetry promptly while an already-decided action
                # approaches its due time; do not change the model's cadence.
                # Keep the control loop responsive after a decision.  The
                # probe request remains the pacing source; this fallback
                # sleep is only a yield and should not add another visible
                # scheduling slice.
                time.sleep(.001 if any(p.state == 'queued' for p in self.executor.pending) else .005)
        except Exception as exc:
            self.log('fatal_error', error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            for pending in self.executor.pending:
                self.log('action_unresolved_at_shutdown', card=pending.action.card_id,
                    slot=pending.action.hand_slot, status=pending.state, decision_tick=pending.decision_tick)
            self.executor.close()
            mirror = getattr(self, 'mirror', None)
            if mirror is not None:
                mirror.close()
            if lifecycle is not None:
                lifecycle.close()
            if console is not None:
                console.close()
            self.log('stopped', log=str(self.log_path))
            self.stream.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='hog26', help='checkpoint alias or .pt path')
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--account-id', type=int, default=config.LOCAL_ACCOUNT_ID)
    parser.add_argument('--owner', type=int, choices=(0, 1), help='explicit seat override for diagnostics')
    parser.add_argument('--own-tower', type=int, default=config.LOCAL_TOWER_TROOP_ID, help='optional local tower fallback when native identity is unavailable; default: auto')
    parser.add_argument('--dry-run', action='store_true', help='infer and log without deploying')
    forms = parser.add_mutually_exclusive_group()
    forms.add_argument('--hero-musketeer', dest='hero_musketeer', action='store_const', const=True, default=None,
        help='compatible explicit hero switch; default automatically reads live card forms')
    forms.add_argument('--base-only', dest='hero_musketeer', action='store_const', const=False,
        help='diagnostic mode: disable special-form execution')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--oracle-elixir', action='store_true', help='experimental exact opponent elixir (outside FAIR training inputs)')
    parser.add_argument('--observation-profile', choices=('reference', 'extended'),
        default=config.DEFAULT_OBSERVATION_PROFILE,
        help='reference: upstream event rules (default); extended: experimental exact combat events')
    parser.add_argument('--experimental-origins', action='store_true',
        help='enable unvalidated full effect/projectile origin chains; requires --observation-profile extended')
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--max-actions', type=int, default=None,
        help='maximum confirmed live actions for this run; stops after the last outcome is confirmed')
    parser.add_argument('--max-matches', type=int, default=None,
        help='maximum completed battles in --continuous mode; omit for continuous running')
    parser.add_argument('--once', action='store_true', help='run one battle; pause on telemetry loss and resume the same battle')
    parser.add_argument('--start-battle', action='store_true', help='tap battle once after loading; lobby must be visible')
    parser.add_argument('--attach-active', action='store_true', help='attach an already active battle without tapping the lobby')
    parser.add_argument('--continuous', action='store_true',
        help='automatically dismiss results and start the next battle')
    parser.add_argument('--console', action='store_true',
        help='enable interactive control: status/pause/resume/model/start/attach/stop')
    parser.add_argument('--lifecycle-calibration', type=Path,
        help='JSON calibration for Battle/result OK controls; defaults to settings')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args(argv)
    if args.experimental_origins and args.observation_profile != 'extended':
        parser.error('--experimental-origins requires --observation-profile extended')
    if args.max_actions is not None and args.max_actions <= 0:
        parser.error('--max-actions must be positive')
    if args.max_matches is not None and args.max_matches <= 0:
        parser.error('--max-matches must be positive')
    if args.start_battle and args.attach_active:
        parser.error('--start-battle and --attach-active are mutually exclusive')
    if args.once and args.continuous:
        parser.error('--once and --continuous are mutually exclusive')
    if args.max_matches is not None and not args.continuous:
        parser.error('--max-matches requires --continuous')
    return args


def main(argv=None):
    args = parse_args(argv)
    agent = CustomCardDeployAgent(checkpoint_path=config.CHECKPOINTS.get(args.checkpoint, Path(args.checkpoint)),
        device_str=args.device, dry_run=args.dry_run, account_id=args.account_id, owner=args.owner,
        sample=args.sample, oracle_elixir=args.oracle_elixir, log_path=args.log, own_tower=args.own_tower,
        hero_musketeer=args.hero_musketeer, observation_profile=args.observation_profile,
        experimental_origins=args.experimental_origins, max_actions=args.max_actions,
        defer_model_load=args.attach_active, allow_partial_tower_attach=args.attach_active)
    console = AgentConsole() if args.console else None
    try:
        # Preserve the historical three-argument call shape for integrations
        # that wrap/mock the runner.  The fourth argument is only needed for
        # the explicit mid-battle attach path.
        if args.attach_active or args.continuous or args.console:
            agent.run(args.seconds, args.once, args.start_battle, args.attach_active,
                      continuous=args.continuous, console=console,
                      max_matches=args.max_matches,
                      lifecycle_calibration=args.lifecycle_calibration)
        else:
            agent.run(args.seconds, args.once, args.start_battle)
    except KeyboardInterrupt:
        print('Stopped by user.')
    finally:
        if console is not None:
            console.close()


if __name__ == '__main__':
    main()
