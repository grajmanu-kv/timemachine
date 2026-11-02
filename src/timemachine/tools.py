"""High-level operations: run, replay, search, shrink, rewind."""
from __future__ import annotations

import math
import random
from typing import Iterator

from .sim import LINK_FAULTS, Scenario, Simulation, build_scenario
from .tape import Divergence, Tape

# Measured on the built-in scenarios (see README): bug hit rate rises steeply with plan size,
# and `shrink` recovers a 2-4 fault reproduction from big plans, so search big and diagnose small.
DEFAULT_FAULTS = 16
FAULT_WEIGHTS = {"drop": 3, "dup": 1, "delay": 2, "crash": 2, "partition": 1, "skew": 3, "drift": 3}


def run(scenario: Scenario, seed: int, faults=(), externals=None) -> Simulation:
    return Simulation(scenario, seed, faults, externals=externals).run()


def scenario_of(tape: Tape) -> Scenario:
    return build_scenario(tape.meta["scenario"], tape.meta.get("params"))


def replay(tape: Tape, scenario: Scenario | None = None) -> Simulation:
    """Re-execute a tape from scratch and require it to reproduce every event exactly.

    Recorded external inputs are fed back, so the real (non-deterministic) calls are
    never made. Raises `Divergence` at the first event that differs.
    """
    bad = tape.verify()
    if bad is not None:
        raise Divergence(bad, tape.events[bad], None, "tape is corrupt: its hash chain does not verify")
    m = tape.meta
    sim = Simulation(scenario or scenario_of(tape), m["seed"], m["faults"],
                     externals=tape.externals(), strict=True, expected=tape)
    sim.run()
    if len(sim.tape) < len(tape):
        raise Divergence(len(sim.tape), tape.events[len(sim.tape)], None, "run ended before the tape did")
    return sim


def rewind(tape: Tape, index: int, scenario: Scenario | None = None) -> Simulation:
    """The world as it was at the first step boundary where event `index` exists.

    Rewinding is deterministic re-execution of the recorded prefix, not a stored snapshot.
    Call `.fork(...)` on the result to branch into a what-if.
    """
    m = tape.meta
    sim = Simulation(scenario or scenario_of(tape), m["seed"], m["faults"],
                     externals=tape.externals(), strict=True, expected=tape)
    return sim.run(until_event=index)


def next_link_index(sim: Simulation, src: str, dst: str) -> int:
    """Index the next message on src->dst will get; handy for aiming a fault at the future."""
    return sim.link_count.get((src, dst), 0)


def random_plan(scenario: Scenario, seed: int, n_faults: int) -> list[dict]:
    r = random.Random(f"plan:{seed}")
    nodes, horizon = sorted(scenario.nodes), scenario.fault_horizon
    kinds = list(scenario.fault_kinds)
    weights = [FAULT_WEIGHTS[k] for k in kinds]
    plan: list[dict] = []
    for _ in range(n_faults):
        kind = r.choices(kinds, weights=weights)[0]
        if kind in LINK_FAULTS:
            src, dst = r.sample(nodes, 2)
            f = {"kind": kind, "src": src, "dst": dst, "n": r.randrange(scenario.max_link_n)}
            if kind == "delay":
                f["by"] = r.choice([20, 40, 80, 160])
            elif kind == "dup":
                f["by"] = r.choice([3, 15, 60])
        elif kind == "skew":
            f = {"kind": "skew", "node": r.choice(nodes), "at": r.randrange(horizon),
                 "offset": r.choice(scenario.skew_offsets)}
        elif kind == "drift":
            f = {"kind": "drift", "node": r.choice(nodes), "at": r.randrange(horizon),
                 "ppm": r.choice(scenario.drift_ppm)}
        elif kind == "crash":
            at = r.randrange(horizon)
            f = {"kind": "crash", "node": r.choice(nodes), "at": at, "restart": at + r.randrange(10, 120)}
        else:
            a, b = sorted(r.sample(nodes, 2))
            frm = r.randrange(horizon)
            f = {"kind": "partition", "a": a, "b": b, "from": frm, "to": frm + r.randrange(10, 150)}
        plan.append(f)
    return plan


def search(scenario: Scenario, seeds, n_faults: int = DEFAULT_FAULTS) -> Iterator[tuple[int, list[dict], Simulation]]:
    """Run random fault plans; yield every (seed, plan, finished sim) that violates the invariant."""
    for seed in seeds:
        plan = random_plan(scenario, seed, n_faults)
        sim = run(scenario, seed, plan)
        if sim.violation:
            yield seed, plan, sim


def shrink(scenario: Scenario, seed: int, faults: list[dict], externals=None,
           max_runs: int = 2000) -> tuple[list[dict], int]:
    """Delta-debugging: the smallest sub-plan that still violates the invariant.

    Recorded externals are fed back (when given) so the search stays deterministic.
    Returns (minimal_plan, runs_used). The result is 1-minimal: removing any single fault makes the run pass.
    """
    runs = 0

    def fails(plan: list[dict]) -> bool:
        nonlocal runs
        runs += 1
        return run(scenario, seed, plan, externals).violation is not None

    plan = list(faults)
    if not fails(plan):
        raise ValueError("this plan does not violate the invariant, nothing to shrink")
    n = 2
    while len(plan) >= 2 and runs < max_runs:
        chunk = math.ceil(len(plan) / n)
        for i in range(0, len(plan), chunk):
            cand = plan[:i] + plan[i + chunk:]
            if fails(cand):
                plan, n = cand, max(n - 1, 2)
                break
        else:
            if n >= len(plan):
                break
            n = min(len(plan), n * 2)
    changed = True
    while changed and runs < max_runs:  # final single-fault pass guarantees 1-minimality
        changed = False
        for i in range(len(plan)):
            cand = plan[:i] + plan[i + 1:]
            if fails(cand):
                plan, changed = cand, True
                break
    return plan, runs