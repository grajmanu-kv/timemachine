import pytest

from timemachine.tools import DEFAULT_FAULTS, random_plan, run, search, shrink

from helpers import lease, lease_drift, toy

KINDS = {"drop", "dup", "delay", "crash", "partition"}


def test_random_plans_are_deterministic_and_well_formed():
    sc = toy()
    plan = random_plan(sc, 5, 12)
    assert plan == random_plan(sc, 5, 12)
    assert plan != random_plan(sc, 6, 12)
    assert len(plan) == 12 and {f["kind"] for f in plan} <= KINDS


def test_skew_faults_are_opt_in_per_scenario():
    assert not [f for s in range(50) for f in random_plan(toy(), s, 16) if f["kind"] == "skew"]
    skews = [f for s in range(50) for f in random_plan(lease(), s, 16) if f["kind"] == "skew"]
    assert skews and {f["node"] for f in skews} == {"s", "c1", "c2"}
    assert {f["offset"] for f in skews} <= set(lease().skew_offsets)


def test_drift_faults_are_opt_in_too():
    assert not [f for s in range(50) for f in random_plan(lease(), s, 16) if f["kind"] == "drift"]
    drifts = [f for s in range(50) for f in random_plan(lease_drift(), s, 16) if f["kind"] == "drift"]
    assert drifts and {f["ppm"] for f in drifts} <= set(lease_drift().drift_ppm)


def test_the_default_plan_size_is_the_measured_one():
    assert DEFAULT_FAULTS == 16


def test_search_finds_the_over_count_bug():
    seed, plan, sim = next(search(toy(), range(200)))
    assert sim.violation.startswith("over-count")


def is_single_cause(fault):
    """The toy invariant breaks only if b hears a message twice: a duplicate on a->b, or `a` restarting
    (its volatile counter resets, so it replays its sends)."""
    return (fault["kind"] == "dup" and (fault["src"], fault["dst"]) == ("a", "b")) or \
           (fault["kind"] == "crash" and fault["node"] == "a" and fault.get("restart") is not None)


def test_shrink_reduces_a_big_plan_to_a_single_causal_fault():
    seed, plan, _ = next(search(toy(), range(200)))
    assert len(plan) == DEFAULT_FAULTS
    small, runs = shrink(toy(), seed, plan)
    assert len(small) == 1 and is_single_cause(small[0]), small
    assert run(toy(), seed, small).violation is not None
    assert run(toy(), seed, []).violation is None   # 1-minimal: the empty plan passes
    assert runs < 60                                 # cheap even from 16 faults


def test_shrink_rejects_a_plan_that_does_not_fail():
    with pytest.raises(ValueError, match="nothing to shrink"):
        shrink(toy(), 3, [])


def test_shrink_feeds_recorded_externals_instead_of_calling_out(tag_calls):
    seed, plan, sim = next(search(toy(), range(200)))
    calls_before = len(tag_calls)
    shrink(toy(), seed, plan, sim.tape.externals())
    assert len(tag_calls) == calls_before