"""Deterministic discrete-event simulator.

Everything that could make two runs differ is either seeded, driven by the
virtual clock, or routed through `ctx.external(...)`, which records the result
on the tape so a replay can feed it back instead of calling the real thing
(an LLM call, a wall-clock read, a network fetch...).
"""
from __future__ import annotations

import copy
import heapq
import json
import random
from typing import Any, Callable

from .tape import Divergence, Tape

LINK_FAULTS = ("drop", "dup", "delay")
# Same-instant ordering, independent of push order: a clock step is visible to everything that happens at that instant.
PRIO = {"crash": 0, "restart": 1, "skew": 2, "drift": 2, "deliver": 3, "timer": 3}
DEFAULT_FAULT_KINDS = ("drop", "dup", "delay", "crash", "partition")


class Agent:
    """Base class for simulated nodes.

    `self.durable` survives crashes. Every other attribute is volatile: a restart
    builds a fresh instance and copies only `durable` across.
    """

    def __init__(self, name: str):
        self.name = name
        self.durable: dict = {}

    def on_start(self, ctx: "Ctx") -> None: ...
    def on_message(self, ctx: "Ctx", src: str, payload: Any) -> None: ...
    def on_timer(self, ctx: "Ctx", name: str) -> None: ...

    def on_restart(self, ctx: "Ctx") -> None:
        self.on_start(ctx)


class Scenario:
    """A system under test: which nodes exist, and what must always be true."""

    name = "scenario"
    duration = 1000          # virtual ms
    fault_horizon = 150      # random fault plans place faults in [0, horizon)
    max_link_n = 5           # random link faults target the first N messages on a link
    # Kinds of fault random plans may draw. "skew" is opt-in: it only means something to a scenario
    # whose agents read ctx.clock(), and leaving it out keeps every existing seed's plan unchanged.
    fault_kinds: tuple[str, ...] = DEFAULT_FAULT_KINDS
    skew_offsets: tuple[int, ...] = (-200, -80, -30, 30, 80, 200)   # wall-clock offsets (ms) a skew fault may set
    # Clock rates (parts per million, relative to true time) a "drift" fault may set. Real oscillators are
    # off by ~100 ppm; these are ~1000x larger because a 100 ms lease only notices a rate error that is
    # large compared with network jitter. Scale by (lease length x rate error) vs. jitter to compare.
    drift_ppm: tuple[int, ...] = (-200_000, -100_000, 100_000, 200_000)
    nodes: list[str] = []

    def params(self) -> dict:
        return {}

    def make_agent(self, node: str) -> Agent:
        raise NotImplementedError

    def check(self, sim: "Simulation") -> str | None:
        """Safety invariant ("never X"), evaluated after every step. Return a message to flag a violation."""
        return None

    def check_final(self, sim: "Simulation") -> str | None:
        """Liveness check ("eventually Y"), evaluated once when the run reaches its end.

        Only meaningful if the scenario guarantees faults have stopped (and been repaired)
        well before `duration`; otherwise "it never recovered" proves nothing.
        """
        return None


SCENARIOS: dict[str, type[Scenario]] = {}


def register(cls: type[Scenario]) -> type[Scenario]:
    SCENARIOS[cls.name] = cls
    return cls


def build_scenario(name: str, params: dict | None = None) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    try:
        return SCENARIOS[name](**(params or {}))
    except TypeError as e:   # e.g. a misspelled parameter name
        raise ValueError(f"bad parameters for scenario {name!r}: {e}") from None


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))


class Ctx:
    """What an agent can do. Deliberately small."""

    __slots__ = ("_sim", "node")

    def __init__(self, sim: "Simulation", node: str):
        self._sim, self.node = sim, node

    @property
    def now(self) -> int:
        return self._sim.now

    def send(self, dst: str, payload: Any) -> None:
        self._sim._send(self.node, dst, payload)

    def set_timer(self, name: str, delay: int) -> None:
        self._sim._set_timer(self.node, name, delay)

    def cancel_timer(self, name: str) -> None:
        key = (self.node, name)
        self._sim.timer_gen[key] = self._sim.timer_gen.get(key, 0) + 1

    def clock(self) -> int:
        """This node's wall clock: its monotonic clock plus whatever offset a `skew` fault has set. It can step."""
        return self._sim.clock(self.node)

    def monotonic(self) -> int:
        """A clock that never steps but can run fast or slow (`drift` fault). It does not reset on restart."""
        return self._sim.monotonic(self.node)

    def rand(self) -> float:
        return self._sim._rng(self.node).random()

    def randrange(self, *args: int) -> int:
        return self._sim._rng(self.node).randrange(*args)

    def external(self, name: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a non-deterministic call once; replays get the recorded result instead."""
        return self._sim._external(self.node, name, fn, args)


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        faults: list[dict] | tuple = (),
        *,
        externals: dict[str, list] | None = None,
        strict: bool = False,
        expected: Tape | None = None,
    ):
        self.scenario, self.seed = scenario, seed
        self.now = 0
        self.strict = strict
        meta = {
            "v": 1,
            "scenario": scenario.name,
            "params": scenario.params(),
            "seed": seed,
            "faults": [dict(f) for f in faults],
            "duration": scenario.duration,
        }
        self.tape = Tape(meta, expected=expected)
        self.externals = {k: list(v) for k, v in (externals or {}).items()}
        self.agents: dict[str, Agent] = {}
        self.up: set[str] = set()
        self.epoch: dict[str, int] = {}
        self.clock_offset: dict[str, int] = {}   # per-node wall-clock offset; survives crashes (a clock is hardware)
        # per-node oscillator: (true time of last rate change, local time then in micro-ms, rate in ppm).
        # Integer arithmetic only, so a rate change never loses a rounding remainder.
        self._osc: dict[str, tuple[int, int, int]] = {}
        self.link_count: dict[tuple[str, str], int] = {}
        self.timer_gen: dict[tuple[str, str], int] = {}
        self.violation: str | None = None
        self.done = False
        self._heap: list = []
        self._seq = 0
        self._link_faults: dict[tuple[str, str, int], list[dict]] = {}
        self._partitions: list[dict] = []
        self._net_rng = random.Random(f"{seed}:net")
        self._rngs: dict[str, random.Random] = {}
        for f in faults:
            self._install(f)
        self._boot()

    # -- setup -------------------------------------------------------------
    def _rng(self, node: str) -> random.Random:
        if node not in self._rngs:
            self._rngs[node] = random.Random(f"{self.seed}:{node}")
        return self._rngs[node]

    def _local_u(self, node: str) -> int:
        base_now, base_u, ppm = self._osc.get(node, (0, 0, 0))
        return base_u + (self.now - base_now) * (1_000_000 + ppm)

    def rate_ppm(self, node: str) -> int:
        return self._osc.get(node, (0, 0, 0))[2]

    def monotonic(self, node: str) -> int:
        """Node-local elapsed time in ms: true time scaled by the node's drift rate (exactly true time with none)."""
        return self._local_u(node) // 1_000_000

    def clock(self, node: str) -> int:
        return self.monotonic(node) + self.clock_offset.get(node, 0)

    def _push(self, t: int, kind: str, data: dict) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, PRIO[kind], self._seq, kind, data))

    def _install(self, f: dict) -> None:
        k = f["kind"]
        if k in LINK_FAULTS:
            self._link_faults.setdefault((f["src"], f["dst"], f["n"]), []).append(f)
        elif k == "partition":
            self._partitions.append(f)
        elif k == "crash":
            at = max(f["at"], self.now)
            self._push(at, "crash", {"node": f["node"]})
            if f.get("restart") is not None:
                self._push(max(f["restart"], at + 1), "restart", {"node": f["node"]})
        elif k == "skew":   # sets the node's wall-clock offset to `offset` (absolute, so the last skew wins)
            if f["node"] not in self.scenario.nodes:
                raise ValueError(f"skew fault names unknown node {f['node']!r}")
            self._push(max(f["at"], self.now), "skew", {"node": f["node"], "offset": f["offset"]})
        elif k == "drift":  # from `at` on, the node's clocks run at (1 + ppm/1e6) x true speed (absolute; last wins)
            if f["node"] not in self.scenario.nodes:
                raise ValueError(f"drift fault names unknown node {f['node']!r}")
            self._push(max(f["at"], self.now), "drift", {"node": f["node"], "ppm": f["ppm"]})
        else:
            raise ValueError(f"unknown fault kind {k!r}")

    def _boot(self) -> None:
        for node in sorted(self.scenario.nodes):
            self.agents[node] = self.scenario.make_agent(node)
            self.up.add(node)
            self.epoch[node] = 0
        for node in sorted(self.scenario.nodes):
            self._record("start", node, {})
            self.agents[node].on_start(Ctx(self, node))
        self._check()

    # -- recording / invariants -------------------------------------------
    def _record(self, kind: str, node: str, data: dict) -> dict:
        return self.tape.append(self.now, kind, node, data)

    def _check(self) -> None:
        if self.violation:
            return
        msg = self.scenario.check(self)
        if msg:
            self.violation = msg
            self._record("violation", "-", {"msg": msg})

    # -- network -----------------------------------------------------------
    def _partitioned(self, a: str, b: str) -> bool:
        return any(
            {p["a"], p["b"]} == {a, b} and p["from"] <= self.now < p["to"] for p in self._partitions
        )

    def _send(self, src: str, dst: str, payload: Any) -> None:
        payload = _jsonable(payload)
        n = self.link_count.get((src, dst), 0)
        self.link_count[(src, dst)] = n + 1
        self._record("send", src, {"dst": dst, "n": n, "p": payload})
        latency = 5 + self._net_rng.randrange(10)  # always drawn, so removing a fault never shifts later latencies
        fs = {f["kind"]: f for f in self._link_faults.get((src, dst, n), [])}
        if self._partitioned(src, dst):
            self._record("drop", src, {"dst": dst, "n": n, "why": "partition"})
            return
        if "drop" in fs:
            self._record("drop", src, {"dst": dst, "n": n, "why": "fault"})
            return
        at = self.now + latency + (fs["delay"]["by"] if "delay" in fs else 0)
        msg = {"src": src, "dst": dst, "n": n, "p": payload}
        self._push(at, "deliver", msg)
        if "dup" in fs:
            self._record("dup", src, {"dst": dst, "n": n})
            self._push(at + fs["dup"].get("by", 7), "deliver", dict(msg))

    def _set_timer(self, node: str, name: str, delay: int) -> None:
        key = (node, name)
        gen = self.timer_gen.get(key, 0) + 1
        self.timer_gen[key] = gen  # re-arming cancels the previous timer of the same name
        self._push(self.now + max(1, delay), "timer",
                   {"node": node, "name": name, "gen": gen, "epoch": self.epoch[node]})

    def _external(self, node: str, name: str, fn: Callable[..., Any], args: tuple) -> Any:
        queue = self.externals.get(name)
        if queue:
            value = queue.pop(0)
        elif self.strict:
            raise Divergence(len(self.tape), None, {"external": name},
                             f"replay needs external {name!r} but the tape has no more of them")
        else:
            value = fn(*args)
        value = _jsonable(value)
        self._record("external", node, {"name": name, "value": value})
        return value

    # -- event handlers ----------------------------------------------------
    def _on_deliver(self, m: dict) -> None:
        dst = m["dst"]
        if dst not in self.up:
            self._record("drop", m["src"], {"dst": dst, "n": m["n"], "why": "dest-down"})
            return
        self._record("deliver", dst, {"src": m["src"], "n": m["n"], "p": m["p"]})
        self.agents[dst].on_message(Ctx(self, dst), m["src"], m["p"])

    def _on_timer(self, t: dict) -> None:
        node = t["node"]
        if node not in self.up or t["epoch"] != self.epoch[node] or t["gen"] != self.timer_gen.get((node, t["name"])):
            return  # cancelled, re-armed, or owned by a previous incarnation: silent
        self._record("timer", node, {"name": t["name"]})
        self.agents[node].on_timer(Ctx(self, node), t["name"])

    def _on_skew(self, c: dict) -> None:
        self.clock_offset[c["node"]] = c["offset"]
        self._record("skew", c["node"], {"offset": c["offset"]})

    def _on_drift(self, c: dict) -> None:
        node = c["node"]
        self._osc[node] = (self.now, self._local_u(node), c["ppm"])
        self._record("drift", node, {"ppm": c["ppm"]})

    def _on_crash(self, c: dict) -> None:
        node = c["node"]
        if node in self.up:
            self.up.discard(node)
            self.epoch[node] += 1
            self._record("crash", node, {})

    def _on_restart(self, c: dict) -> None:
        node = c["node"]
        if node in self.up:
            return
        old = self.agents[node]
        fresh = self.scenario.make_agent(node)
        fresh.durable = old.durable  # only durable state survives
        self.agents[node] = fresh
        self.up.add(node)
        self._record("restart", node, {})
        fresh.on_restart(Ctx(self, node))

    # -- driving -----------------------------------------------------------
    def step(self) -> bool:
        """Process one scheduled event (plus everything it triggers synchronously)."""
        if self.done or self.violation:
            return False
        if not self._heap or self._heap[0][0] > self.scenario.duration:
            self._finish()
            return False
        t, _, _, kind, data = heapq.heappop(self._heap)
        self.now = t
        getattr(self, "_on_" + kind)(data)
        self._check()
        return True

    def _finish(self) -> None:
        """The run reached its end without a safety violation: evaluate liveness once."""
        self.done = True
        msg = self.scenario.check_final(self)
        if msg and not self.violation:
            self.now = max(self.now, self.scenario.duration)
            self.violation = msg
            self._record("violation", "-", {"msg": msg})

    def run(self, until_event: int | None = None) -> "Simulation":
        """Run to completion, or stop at the first step boundary where event `until_event` exists."""
        while self.step():
            if until_event is not None and len(self.tape) > until_event:
                break
        return self

    # -- time travel -------------------------------------------------------
    def add_faults(self, faults: list[dict]) -> None:
        """Add faults that affect only the future (so the forked tape still replays from scratch)."""
        for f in faults:
            f = dict(f)
            if f["kind"] in LINK_FAULTS:
                if f["n"] < self.link_count.get((f["src"], f["dst"]), 0):
                    raise ValueError(f"fault targets message #{f['n']} on {f['src']}->{f['dst']}, which was already sent")
            elif f["kind"] == "crash":
                f["at"] = max(f["at"], self.now + 1)
                if f.get("restart") is not None:
                    f["restart"] = max(f["restart"], f["at"] + 1)
            elif f["kind"] == "partition":
                f["from"] = max(f["from"], self.now + 1)
            elif f["kind"] in ("skew", "drift"):
                f["at"] = max(f["at"], self.now + 1)
            self._install(f)
            self.tape.meta["faults"].append(f)

    def fork(self, add_faults: list[dict] | tuple = ()) -> "Simulation":
        """An independent copy of the world at this instant, optionally with new faults."""
        expected, self.tape.expected = self.tape.expected, None
        try:
            twin = copy.deepcopy(self)
        finally:
            self.tape.expected = expected
        twin.strict = False
        twin.tape.meta = copy.deepcopy(self.tape.meta)
        twin.tape.meta["forked_at_event"] = len(self.tape) - 1
        twin.done = False
        twin.add_faults(list(add_faults))
        return twin