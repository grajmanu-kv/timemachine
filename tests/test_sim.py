"""Simulator mechanics: determinism, every fault kind, rewind and fork. Uses the toy scenario only."""
import pytest

from timemachine.sim import Simulation
from timemachine.tools import next_link_index, replay, rewind, run

from helpers import DUP_FIRST, clocktoy, offsets, toy


def events(sim, kind):
    return [e for e in sim.tape.events if e["k"] == kind]


def delivers_to_b(sim):
    return [e for e in events(sim, "deliver") if e["n"] == "b"]


# --- determinism ---------------------------------------------------------------------------
def test_same_seed_and_plan_give_the_same_tape():
    a, b = run(toy(), 7, [DUP_FIRST]), run(toy(), 7, [DUP_FIRST])
    assert a.tape.head == b.tape.head and len(a.tape) == len(b.tape)


def test_different_seeds_give_different_tapes():
    assert run(toy(), 1).tape.head != run(toy(), 2).tape.head


def test_a_fault_free_run_satisfies_the_invariant():
    sim = run(toy(), 3)
    assert sim.violation is None
    assert sim.agents["b"].durable["count"] == 3


# --- each fault kind -----------------------------------------------------------------------
def test_drop_loses_exactly_the_targeted_message():
    sim = run(toy(), 1, [{"kind": "drop", "src": "a", "dst": "b", "n": 1}])
    drops = events(sim, "drop")
    assert [(d["d"]["n"], d["d"]["why"]) for d in drops] == [(1, "fault")]
    assert sim.agents["b"].durable["count"] == 2


def test_dup_delivers_the_message_twice_and_trips_the_invariant():
    sim = run(toy(), 1, [DUP_FIRST])
    first = [e for e in delivers_to_b(sim) if e["d"]["n"] == 0]
    assert len(first) == 2 and first[1]["t"] - first[0]["t"] == 7
    assert sim.violation == "over-count: 4"


def test_delay_postpones_delivery():
    base = run(toy(), 1)
    slow = run(toy(), 1, [{"kind": "delay", "src": "a", "dst": "b", "n": 0, "by": 100}])
    t_base = next(e["t"] for e in delivers_to_b(base) if e["d"]["n"] == 0)
    t_slow = next(e["t"] for e in delivers_to_b(slow) if e["d"]["n"] == 0)
    assert t_slow == t_base + 100


def test_crash_drops_traffic_keeps_durable_state_and_wipes_volatile_state():
    sim = run(toy(), 1, [{"kind": "crash", "node": "b", "at": 16, "restart": 30}])
    delivers = delivers_to_b(sim)
    assert len(events(sim, "crash")) == 1
    assert events(sim, "restart")[0]["t"] == 30
    assert not [e for e in delivers if 16 <= e["t"] < 30]                     # nothing delivered while down
    assert any(e["d"]["why"] == "dest-down" for e in events(sim, "drop"))     # ...it was dropped instead
    assert sim.agents["b"].durable["count"] == len(delivers)                  # durable survives the crash
    after = [e for e in delivers if e["t"] >= 30]
    assert len(sim.agents["b"].log) == len(after) < sim.agents["b"].durable["count"]  # volatile log was wiped


def test_a_crash_without_restart_leaves_the_node_down():
    sim = run(toy(), 1, [{"kind": "crash", "node": "b", "at": 16}])
    assert "b" not in sim.up


@pytest.mark.parametrize("a, b", [("a", "b"), ("b", "a")])
def test_partition_drops_traffic_in_the_window_whichever_way_it_is_written(a, b):
    sim = run(toy(), 1, [{"kind": "partition", "a": a, "b": b, "from": 0, "to": 15}])
    assert [d["d"]["why"] for d in events(sim, "drop")] == ["partition", "partition"]  # sends at t=1 and t=11
    assert sim.agents["b"].durable["count"] == 1                                        # send at t=21 got through


def test_unknown_fault_kind_is_rejected():
    with pytest.raises(ValueError):
        Simulation(toy(), 1, [{"kind": "bogus"}])


# --- rewind and fork -----------------------------------------------------------------------
def after_first_send(full):
    first_send = next(e["i"] for e in full.tape.events if e["k"] == "send")
    return rewind(full.tape, first_send)


def test_rewind_lands_on_a_step_boundary_inside_the_original_run():
    full = run(toy(), 3)
    k = len(full.tape) // 2
    world = rewind(full.tape, k)
    n = len(world.tape)
    assert n > k
    assert world.tape.head == full.tape.events[n - 1]["h"]


def test_fork_without_faults_reaches_the_same_end():
    full = run(toy(), 3)
    twin = rewind(full.tape, len(full.tape) // 2).fork().run()
    assert twin.tape.head == full.tape.head


def test_a_future_fault_in_a_fork_changes_the_fork_and_nothing_else():
    full = run(toy(), 3)
    assert full.violation is None
    world = after_first_send(full)
    nxt = next_link_index(world, "a", "b")
    assert nxt == 1
    twin = world.fork([{"kind": "dup", "src": "a", "dst": "b", "n": nxt, "by": 7}]).run()
    assert twin.violation == "over-count: 4"           # the counterfactual breaks the invariant...
    assert world.violation is None and not world.done  # ...the parent world is untouched...
    assert world.run().tape.head == full.tape.head     # ...and still resumes to the original ending


def test_a_forked_tape_replays_exactly_from_scratch():
    world = after_first_send(run(toy(), 3))
    twin = world.fork([{"kind": "dup", "src": "a", "dst": "b", "n": 1, "by": 7}]).run()
    assert twin.tape.meta["forked_at_event"] == len(world.tape) - 1
    assert replay(twin.tape).tape.head == twin.tape.head


def test_a_fork_rejects_faults_aimed_at_messages_already_sent():
    world = after_first_send(run(toy(), 3))
    with pytest.raises(ValueError, match="already sent"):
        world.fork([{"kind": "drop", "src": "a", "dst": "b", "n": 0}])


def test_forked_crashes_and_partitions_are_pushed_strictly_into_the_future():
    world = after_first_send(run(toy(), 3))
    twin = world.fork([{"kind": "crash", "node": "b", "at": 0, "restart": 1},
                       {"kind": "partition", "a": "a", "b": "b", "from": 0, "to": 5}])
    crash, part = twin.tape.meta["faults"][-2:]
    assert crash["at"] > world.now and crash["restart"] > crash["at"]
    assert part["from"] > world.now
    assert replay(twin.run().tape).tape.head == twin.tape.head


# --- clock skew ----------------------------------------------------------------------------
def skew(node, at, offset):
    return {"kind": "skew", "node": node, "at": at, "offset": offset}


def test_clocks_agree_with_true_time_until_something_skews_them():
    sim = run(clocktoy(), 1)
    for node in "ab":
        assert all(t == c == m for t, c, m in sim.agents[node].durable["reads"])


def test_a_skew_steps_the_wall_clock_from_its_instant_and_never_the_monotonic_clock():
    sim = run(clocktoy(), 1, [skew("a", 20, 50)])
    assert offsets(sim, "a") == {10: 0, 20: 50, 30: 50, 40: 50, 50: 50, 60: 50}   # visible to events at the same instant
    assert all(m == t for t, _, m in sim.agents["a"].durable["reads"])
    assert set(offsets(sim, "b").values()) == {0}                                  # other nodes are unaffected
    (ev,) = events(sim, "skew")
    assert (ev["t"], ev["n"], ev["d"]) == (20, "a", {"offset": 50})


def test_a_later_skew_replaces_an_earlier_one():
    sim = run(clocktoy(), 1, [skew("a", 15, 50), skew("a", 35, -20)])
    assert offsets(sim, "a") == {10: 0, 20: 50, 30: 50, 40: -20, 50: -20, 60: -20}


def test_a_skew_survives_a_crash_and_restart():
    sim = run(clocktoy(), 1, [skew("a", 5, 50), {"kind": "crash", "node": "a", "at": 25, "restart": 33}])
    reads = offsets(sim, "a")
    assert max(reads) > 33 and set(reads.values()) == {50}   # the clock is hardware, not process state


def test_skew_of_an_unknown_node_is_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        Simulation(clocktoy(), 1, [skew("zzz", 5, 10)])


def test_a_skewed_tape_replays_exactly_and_the_skew_is_part_of_the_chain():
    plain, skewed = run(clocktoy(), 1), run(clocktoy(), 1, [skew("a", 20, 50)])
    assert plain.tape.head != skewed.tape.head
    assert replay(skewed.tape).tape.head == skewed.tape.head


def test_a_forked_skew_lands_strictly_in_the_future_and_the_fork_replays():
    full = run(clocktoy(), 1)
    world = rewind(full.tape, len(full.tape) // 2)
    twin = world.fork([skew("a", 0, 50)])
    assert twin.tape.meta["faults"][-1]["at"] > world.now
    assert replay(twin.run().tape).tape.head == twin.tape.head


# --- clock drift ---------------------------------------------------------------------------
def drift(node, at, ppm):
    return {"kind": "drift", "node": node, "at": at, "ppm": ppm}


def reads(sim, node):
    """{true time: (wall clock, monotonic)} as `node` saw it."""
    return {t: (c, m) for t, c, m in sim.agents[node].durable["reads"]}


def test_a_fast_clock_runs_ahead_and_a_slow_one_behind_from_the_instant_of_the_fault():
    fast = reads(run(clocktoy(), 1, [drift("a", 20, 100_000)]), "a")
    assert [fast[t][1] for t in (10, 20, 30, 40, 50, 60)] == [10, 20, 31, 42, 53, 64]
    slow = reads(run(clocktoy(), 1, [drift("a", 20, -200_000)]), "a")
    assert [slow[t][1] for t in (10, 20, 30, 40, 50, 60)] == [10, 20, 28, 36, 44, 52]
    assert all(c == m for c, m in slow.values())      # no skew: wall clock is the monotonic clock


def test_drift_only_affects_the_named_node():
    sim = run(clocktoy(), 1, [drift("a", 5, 300_000)])
    assert all(c == m == t for t, (c, m) in reads(sim, "b").items())


def test_rate_changes_keep_their_fractional_remainder():
    """+25% for 6 ms then -25% for 2 ms leaves 22.5 -> 24.0 local ms; rounding at each change would give 23."""
    sim = run(clocktoy(), 1, [drift("a", 15, 250_000), drift("a", 21, -250_000), drift("a", 23, 0)])
    assert [m for _, m in (reads(sim, "a")[t] for t in (10, 20, 30, 40))] == [10, 21, 31, 41]


def test_skew_and_drift_compose_and_only_the_wall_clock_is_skewed():
    sim = run(clocktoy(), 1, [drift("a", 10, 100_000), skew("a", 10, 40)])
    c, m = reads(sim, "a")[30]
    assert m == 10 + 22 and c == m + 40


def test_a_drift_survives_a_crash_and_restart():
    sim = run(clocktoy(), 1, [drift("a", 5, 100_000), {"kind": "crash", "node": "a", "at": 25, "restart": 33}])
    late = {t: m for t, (c, m) in reads(sim, "a").items() if t > 33}
    assert late and all(m == 5 + (t - 5) * 11 // 10 for t, m in late.items())


def test_drift_of_an_unknown_node_is_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        Simulation(clocktoy(), 1, [drift("zzz", 5, 1000)])


def test_a_drifted_tape_replays_and_a_forked_drift_is_clamped_into_the_future():
    full = run(clocktoy(), 1, [drift("a", 20, 100_000)])
    assert replay(full.tape).tape.head == full.tape.head
    world = rewind(full.tape, len(full.tape) // 2)
    twin = world.fork([drift("b", 0, -100_000)])
    assert twin.tape.meta["faults"][-1]["at"] > world.now
    assert replay(twin.run().tape).tape.head == twin.tape.head