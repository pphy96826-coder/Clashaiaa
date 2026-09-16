"""Optional adapter for the downloaded Clash Royale native battle engine.

The engine is a deterministic *stand-alone* battle, not a live-state import
API.  This module therefore refuses to pretend that a freshly reset standard
match is a prediction of an arbitrary online match.  It is usable for health
checks, speed benchmarks, and replay rollouts; a live rollout must provide an
explicit replay/state bridge before it can be used for model input.
"""

from __future__ import annotations

import importlib.util
import base64
import hashlib
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class FullSimulationUnavailable(RuntimeError):
    """The native engine exists, but cannot represent the requested live state."""


PREPARED_NATIVE_ACTION_CARDS = frozenset({
    26000000, 26000001, 26000003, 26000005,
    26000014, 26000018, 28000000, 28000001,
    26000010, 26000021, 26000030, 26000038, 27000000, 28000011,
})


@dataclass(frozen=True)
class SimulationBenchmark:
    simulated_ticks: int
    elapsed_ms: float

    @property
    def ticks_per_second(self) -> float:
        return self.simulated_ticks / max(self.elapsed_ms / 1000.0, 1e-9)

    @property
    def one_second_ms(self) -> float:
        return 20.0 * 1000.0 / max(self.ticks_per_second, 1e-9)


@dataclass(frozen=True)
class LiveSnapshotEnvelope:
    """Versioned wire payload for a future engine state-import command."""

    observation: dict
    schema: str = "live-snapshot.v1"

    def to_bytes(self) -> bytes:
        payload = {"schema": self.schema, "observation": self.observation}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        return json.dumps({"schema": self.schema, "sha256": digest,
                           "payload": base64.b64encode(body).decode("ascii")},
                          separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_observation(cls, observation):
        if not hasattr(observation, "to_dict"):
            raise TypeError("live snapshot must be an ObservationV1 contract")
        data = observation.to_dict()
        if data.get("version") != "observation.v1":
            raise ValueError("only observation.v1 snapshots can be imported")
        return cls(data)


def _load_package(root: Path):
    """Load the hyphenated downloaded package without installing it globally."""
    name = "royale_battle_engine_local"
    init = root / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(init)
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load battle engine from {root}")
    module = importlib.util.module_from_spec(spec)
    # The downloaded package uses relative imports (``.engine``, ``.batch``).
    # Register the package before executing __init__, just as importlib does
    # for a normal installed package.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class FullBattleSimulator:
    """Small, lazy wrapper around Clash-Royale-Battle-Engine.

    ``predict_live`` intentionally requires a future state-import bridge.  The
    current upstream engine only supports reset/replay, so silently using it
    against a live telemetry snapshot would create a false board prediction.
    """

    def __init__(self, root=None, host="127.0.0.1", port=26789):
        self.root = Path(root or Path.home() / "Downloads" / "clash-royale-battle-engine-main")
        self.host = host
        self.port = int(port)
        self._module = None

    @property
    def available(self):
        return (self.root / "engine.py").is_file() and (self.root / "protocol.py").is_file()

    @property
    def native_action_cards(self):
        """Action cards supported by the downloaded prepared service."""
        return PREPARED_NATIVE_ACTION_CARDS

    def unsupported_action_cards(self, deck):
        return tuple(sorted(set(map(int, deck)) - self.native_action_cards))

    def _api(self):
        if not self.available:
            raise FullSimulationUnavailable(f"battle engine not found at {self.root}")
        if self._module is None:
            self._module = _load_package(self.root)
        return self._module

    def benchmark(self, ticks=1000, seed=1):
        """Measure native rollout cost; does not touch the live game."""
        api = self._api()
        engine = api.Engine(host=self.host, port=self.port, timeout=10)
        started = time.perf_counter()
        state = engine.reset(seed=seed)
        remaining = max(1, int(ticks))
        while remaining:
            step = min(remaining, 1000)
            state = engine.step((), ticks=step).state
            remaining -= step
        elapsed = (time.perf_counter() - started) * 1000.0
        engine.close()
        return SimulationBenchmark(int(ticks), elapsed)

    def create_engine(self, decks, *, seed=1):
        """Create a resettable engine using the live battle's two decks."""
        api = self._api()
        if len(decks) != 2 or any(len(tuple(deck)) != 8 for deck in decks):
            raise FullSimulationUnavailable("both players need exactly eight cards for a mirror")
        match = json.loads((self.root / "standard_match.json").read_text(encoding="utf-8"))
        for owner, deck in enumerate(decks):
            match["battle"][f"deck{owner}"]["sp"] = [{"d": int(card)} for card in deck]

        class ConfiguredEngine(api.Engine):
            def _capture(self_inner, generation=0, cursor=0):
                # Some prepared APKs expose the full JSON ``observe`` command
                # but omit the compact ``minimal`` command used by the
                # upstream Python client.  Prefer compact capture, then use
                # the compatible full capture without pretending an error
                # mapping is a decoded State.
                compact = self_inner.request(f'minimal {generation} {cursor}')
                if isinstance(compact, dict) and compact.get('players'):
                    from royale_battle_engine_local.protocol import State
                    return State.decode(compact)
                full = self_inner.request('observe')
                if not isinstance(full, dict) or not full.get('players'):
                    detail = (compact.get('error') if isinstance(compact, dict) else 'invalid compact capture')
                    raise RuntimeError(f'engine observation unavailable: {detail}')
                from royale_battle_engine_local.protocol import State
                # Full observe uses the same player/entity/play contract; the
                # protocol decoder intentionally ignores additional telemetry
                # fields supplied by newer probes.
                return State.decode(full)

            def reset(self_inner, seed=seed):
                self_inner.state = None
                configured = json.loads(json.dumps(match))
                configured["rndSeed"] = int(seed)
                reply = self_inner.request("configure " + json.dumps(configured, separators=(",", ":")))
                if not reply.get("ok", False):
                    raise RuntimeError(reply.get("error", "custom deck configuration rejected"))
                step_reply = self_inner.request("step 90")
                if not step_reply.get("ok", False):
                    raise RuntimeError(step_reply.get("error", "initial simulation step rejected"))
                # The compact capture is generation-scoped.  Passing 0 after
                # a fresh configure makes the probe return an error mapping,
                # which the upstream State decoder then reports misleadingly
                # as KeyError('players').
                generation = int(step_reply.get("generation", reply.get("generation", 0)))
                self_inner.state = self_inner._capture(generation, 0)
                return self_inner.state

        return ConfiguredEngine(host=self.host, port=self.port, timeout=10)

    def predict_live(self, live_state, horizon_ticks, *, replay=None):
        """Predict a live state only when an explicit import/replay is supplied.

        The downloaded protocol has no state-import command.  ``replay`` is
        reserved for a future verified bridge and is deliberately rejected for
        now instead of producing a standard-deck look-alike prediction.
        """
        del live_state, horizon_ticks, replay
        raise FullSimulationUnavailable(
            "downloaded engine supports reset/replay only; live snapshot import is unavailable")

    def import_snapshot(self, observation, *, timeout=2.0):
        """Send the defined import envelope to an engine that implements it.

        The currently downloaded prebuilt service does not implement this
        command and will return an error.  Keeping the call here makes the
        missing native capability explicit and gives a compatible future
        engine a stable wire format.
        """
        envelope = LiveSnapshotEnvelope.from_observation(observation)
        command = "import-snapshot-v1 " + base64.b64encode(envelope.to_bytes()).decode("ascii") + "\n"
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as connection:
                with connection.makefile("rb") as stream:
                    connection.sendall(b"session-v1\n")
                    hello = stream.readline()
                    if not hello or not json.loads(hello).get("ok"):
                        raise FullSimulationUnavailable("engine session handshake rejected")
                    connection.sendall(command.encode("ascii"))
                    reply = json.loads(stream.readline() or b"null")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FullSimulationUnavailable(f"snapshot import transport failed: {exc}") from exc
        if not isinstance(reply, dict) or not reply.get("ok"):
            error = reply.get("error", "import-snapshot-v1 is not supported") if isinstance(reply, dict) else "invalid import response"
            raise FullSimulationUnavailable(error)
        return reply
