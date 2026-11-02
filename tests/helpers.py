"""Shared test helpers: a tiny scenario for machinery tests, builders, and fixture seeds."""
from __future__ import annotations

import random

from timemachine.sim import Agent, Ctx, Scenario, build_scenario, register

# --- fixture seeds -------------------------------------------------------------------------
# These depend on the exact behaviour of tools.random_plan. If you change plan generation they
# must be regenerated, e.g.:
#   [s for s, _, sim in search(elect("vote"), range(4000), n_faults=6) if sim.violation.startswith("two leaders")]
VOTE_SEEDS = [358, 672, 931]        # election bug=vote, 6-fault plans
BIG_PLAN_SEEDS = [203, 212, 244]    # election bug=vote, DEFAULT_FAULTS-sized plans (16)
LEASE_SEEDS = [1, 2, 4]             # lease bug="abs", 6-fault plans
DRIFT_SEEDS = [1, 6, 14]            # lease_drift bug="nomargin", 6-fault plans


# --- builders for the real scenarios -------------------------------------------------------
def elect(bug):
    return build_scenario("election", {"bug": bug})


def transfer(fixed: bool = False):
    return build_scenario("transfer", {"fixed": fixed})


def lease(bug="abs"):
    return build_scenario("lease", {"bug": bug})


def lease_drift(bug="nomargin"):
    return build_scenario("lease_drift", {"bug": bug})


# --- toy scenario ------------------------------------------------------------------------
# `a` sends three `inc` messages to `b` (at t=1, 11, 21). `b` counts them durably and keeps a
# volatile log. Invariant: count <= 3. A single duplicated message breaks it, and so does `a`
# crashing and restarting (its counter is volatile, so it re-sends everything). That makes this
# the smallest scenario that still exercises search, shrink, fork, and replay.
def _live_tag(i: int) -> int:
    """Stands in for a non-deterministic outside call."""
    return random.SystemRandom().randrange(1_000_000)


_hooks = {"tag": _live_tag}   # fixtures swap the entry to count or pin the outside call


class _Sender(Agent):
    def on_start(self, ctx: Ctx) -> None:
        self.i = 0
        ctx.set_timer("go", 1)

    def on_timer(self, ctx: Ctx, name: str) -> None:
        if name == "go" and self.i < 3:
            tag = ctx.external("tag", lambda i: _hooks["tag"](i), self.i)
            ctx.send("b", {"op": "inc", "i": self.i, "tag": tag})
            self.i += 1
            ctx.set_timer("go", 10)


class _Receiver(Agent):
    def on_start(self, ctx: Ctx) -> None:
        self.durable.setdefault("count", 0)
        self.log: list[int] = []   # volatile: wiped by a crash

    def on_message(self, ctx: Ctx, src: str, p) -> None:
        self.durable["count"] += 1
        self.log.append(p["i"])


@register
class ToyScenario(Scenario):
    name = "toy"
    nodes = ["a", "b"]
    duration = 200
    fault_horizon = 40
    max_link_n = 3

    def make_agent(self, node: str) -> Agent:
        return _Sender(node) if node == "a" else _Receiver(node)

    def check(self, sim) -> str | None:
        count = sim.agents["b"].durable.get("count", 0)
        return f"over-count: {count}" if count > 3 else None


def toy():
    return build_scenario("toy")


DUP_FIRST = {"kind": "dup", "src": "a", "dst": "b", "n": 0, "by": 7}


# --- clock toy -----------------------------------------------------------------------------
# Every node reads its clocks at t=10, 20, ... 60 and stores [true time, wall clock, monotonic] in
# durable state, so tests can see exactly what a `skew` fault did to what a node observes.
class _Reader(Agent):
    def on_start(self, ctx: Ctx) -> None:
        self.durable.setdefault("reads", [])
        ctx.set_timer("tick", 10)

    def on_timer(self, ctx: Ctx, name: str) -> None:
        self.durable["reads"].append([ctx.now, ctx.clock(), ctx.monotonic()])
        if len(self.durable["reads"]) < 6:
            ctx.set_timer("tick", 10)


@register
class ClockToyScenario(Scenario):
    name = "clocktoy"
    nodes = ["a", "b"]
    duration = 100

    def make_agent(self, node: str) -> Agent:
        return _Reader(node)


def clocktoy():
    return build_scenario("clocktoy")


def offsets(sim, node):
    """{true time: wall-clock offset} as `node` saw it."""
    return {t: c - t for t, c, _ in sim.agents[node].durable["reads"]}