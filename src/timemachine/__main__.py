"""CLI:  python -m timemachine <command>   (run from the directory that contains the package)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from . import tools
from .sim import build_scenario
from .tape import Divergence, Tape, first_divergence


def _show(tape: Tape, lo: int, hi: int) -> None:
    for ev in tape.events[max(lo, 0):hi]:
        print("  " + Tape.fmt(ev))


def _tail(tape: Tape, n: int = 14) -> None:
    _show(tape, len(tape) - n, len(tape))


def _scenario_params(a) -> dict:
    params: dict = {}
    if a.fixed:
        if a.scenario != "transfer":
            raise ValueError("--fixed only applies to the transfer scenario; use --param bug=null for election")
        params["fixed"] = True
    for kv in a.param:
        key, sep, raw = kv.partition("=")
        if not sep:
            raise ValueError(f"--param expects KEY=VALUE, got {kv!r}")
        try:
            params[key] = json.loads(raw)   # numbers, true/false/null; anything else stays a string
        except ValueError:
            params[key] = raw
    return params


def cmd_search(a) -> int:
    sc = build_scenario(a.scenario, _scenario_params(a))
    for seed, plan, sim in tools.search(sc, range(a.seeds), a.faults):
        print(f"violation at seed {seed} with {len(plan)} faults: {sim.violation}")
        sim.tape.save(a.out)
        print(f"tape saved: {a.out}  ({len(sim.tape)} events, head {sim.tape.head})")
        return 0
    print(f"no violation in {a.seeds} seeds x {a.faults} faults")
    return 1


def cmd_replay(a) -> int:
    tape = Tape.load(a.tape)
    try:
        sim = tools.replay(tape)
    except Divergence as d:
        print(f"REPLAY DIVERGED: {d}")
        print("  tape  :", Tape.fmt(d.expected))
        print("  actual:", Tape.fmt(d.actual) if d.actual and "i" in d.actual else d.actual)
        return 2
    print(f"replay verified: {len(sim.tape)} events reproduced exactly, head {sim.tape.head}")
    if sim.violation:
        print(f"violation reproduced: {sim.violation}")
    return 0


def cmd_shrink(a) -> int:
    tape = Tape.load(a.tape)
    sc, m = tools.scenario_of(tape), tape.meta
    plan, runs = tools.shrink(sc, m["seed"], m["faults"], tape.externals())
    print(f"shrunk {len(m['faults'])} faults -> {len(plan)} in {runs} runs:")
    for f in plan:
        print("  ", json.dumps(f, sort_keys=True))
    sim = tools.run(sc, m["seed"], plan, tape.externals())
    sim.tape.save(a.out)
    print(f"minimal tape saved: {a.out}  ({len(sim.tape)} events)  violation: {sim.violation}")
    return 0


def cmd_diff(a) -> int:
    ta, tb = Tape.load(a.a), Tape.load(a.b)
    d = first_divergence(ta, tb)
    if d is None:
        print("tapes are identical")
        return 0
    i, ea, eb = d
    print(f"first divergence at event #{i}")
    print("  A:", Tape.fmt(ea))
    print("  B:", Tape.fmt(eb))
    return 0


def cmd_show(a) -> int:
    tape = Tape.load(a.tape)
    print("meta:", json.dumps(tape.meta, sort_keys=True))
    _show(tape, a.start, a.end if a.end is not None else len(tape))
    return 0


def cmd_fork(a) -> int:
    tape = Tape.load(a.tape)
    sim = tools.rewind(tape, a.at)
    print(f"rewound to t={sim.now}ms, {len(sim.tape)} events in")
    nxt = {f"{s}->{d}": n for (s, d), n in sorted(sim.link_count.items())}
    print(f"next message index per link (faults must target these or later): {nxt}")
    twin = sim.fork([json.loads(f) for f in a.fault])
    twin.run()
    print(f"forked run finished: {len(twin.tape)} events, violation: {twin.violation}")
    twin.tape.save(a.out)
    print(f"fork tape saved: {a.out}")
    return 0


def cmd_demo(a) -> int:
    work = tempfile.mkdtemp(prefix="timemachine-")
    p = lambda name: os.path.join(work, name)
    sc = build_scenario("transfer")
    print(f"== 1. Fault-injection search (random plans of {tools.DEFAULT_FAULTS} faults) ==")
    hit = next(tools.search(sc, range(5000)), None)
    if hit is None:
        print("no violation found; try more seeds")
        return 1
    seed, plan, sim = hit
    sim.tape.save(p("bug.tape.json"))
    print(f"seed {seed}: {sim.violation}   ({len(sim.tape)} events)")
    _tail(sim.tape, 8)

    print("\n== 2. Replay: re-execute from the tape; the real pricing call is never made ==")
    rep = tools.replay(sim.tape)
    print(f"verified {len(rep.tape)} events, identical head {rep.tape.head}")

    print("\n== 3. Shrink the fault plan ==")
    minimal, runs = tools.shrink(sc, seed, plan, sim.tape.externals())
    print(f"{len(plan)} faults -> {len(minimal)} ({runs} runs):")
    for f in minimal:
        print("  ", json.dumps(f, sort_keys=True))
    small = tools.run(sc, seed, minimal, sim.tape.externals())
    small.tape.save(p("min.tape.json"))

    print("\n== 4. Divergence bisection: original failing run vs. the run with no faults ==")
    clean = tools.run(sc, seed, [], sim.tape.externals())
    i, ea, eb = first_divergence(sim.tape, clean.tape)
    print(f"first differing event: #{i}")
    print("  failing:", Tape.fmt(ea))
    print("  clean  :", Tape.fmt(eb))

    print("\n== 5. Rewind + fork: what if the first client retry had been lost? ==")
    timers = [e for e in small.tape.events if e["k"] == "timer" and e["n"] == "c"]
    if timers:
        at = timers[0]["i"] - 1
        world = tools.rewind(small.tape, at)
        nxt = tools.next_link_index(world, "c", "s")
        twin = world.fork([{"kind": "drop", "src": "c", "dst": "s", "n": nxt}])
        twin.run()
        twin.tape.save(p("fork.tape.json"))
        print(f"rewound to t={world.now}ms (event {len(world.tape)}), forked with drop(c->s #{nxt}):")
        print(f"  violation in fork: {twin.violation}")
        tools.replay(twin.tape)
        print("  forked tape also replays exactly from scratch")
    print(f"\ntapes written to {work}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="timemachine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="random fault-injection search for an invariant violation")
    s.add_argument("--scenario", default="transfer")
    s.add_argument("--seeds", type=int, default=2000)
    s.add_argument("--faults", type=int, default=tools.DEFAULT_FAULTS, help="faults per random plan (default: %(default)s)")
    s.add_argument("--fixed", action="store_true", help="transfer only: use the bug-fixed variant (should find nothing)")
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="scenario parameter, repeatable; e.g. --param bug=churn or --param bug=null")
    s.add_argument("--out", default="bug.tape.json")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("replay", help="re-execute a tape and verify every event")
    s.add_argument("tape")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("shrink", help="minimise the fault plan of a failing tape")
    s.add_argument("tape")
    s.add_argument("--out", default="min.tape.json")
    s.set_defaults(fn=cmd_shrink)

    s = sub.add_parser("diff", help="find the first event where two tapes diverge")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("show", help="print tape events")
    s.add_argument("tape")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--end", type=int)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("fork", help="rewind to an event, add faults, and run the what-if")
    s.add_argument("tape")
    s.add_argument("--at", type=int, required=True, help="event index to rewind to")
    s.add_argument("--fault", action="append", default=[], help='JSON fault, e.g. \'{"kind":"drop","src":"c","dst":"s","n":2}\'')
    s.add_argument("--out", default="fork.tape.json")
    s.set_defaults(fn=cmd_fork)

    s = sub.add_parser("demo", help="end-to-end walkthrough on the built-in scenario")
    s.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())