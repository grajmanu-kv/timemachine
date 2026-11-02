"""Lease lock: a bug that exists only in the clocks. Wall-clock expiry timestamps vs. monotonic durations."""
import pytest

from timemachine.sim import build_scenario
from timemachine.tools import random_plan, replay, rewind, run, search, shrink

from helpers import LEASE_SEEDS, lease


def plan(bug, seed, n):
    return random_plan(lease(bug), seed, n)


def without_skew(faults):
    return [f for f in faults if f["kind"] != "skew"]


def test_a_fault_free_run_is_safe_and_the_lease_really_changes_hands():
    sim = run(lease("abs"), 0)
    assert sim.violation is None
    grants = [e for e in sim.tape.events if e["k"] == "deliver" and e["d"]["p"]["op"] == "grant"]
    assert {e["n"] for e in grants} == {"c1", "c2"}      # both clients got it at some point: the test has teeth


def test_the_abs_variant_is_broken_by_clock_skew_on_known_seeds():
    sc = lease("abs")
    for s in LEASE_SEEDS:
        sim = run(sc, s, plan("abs", s, 6))
        assert "both believe they hold the lease" in sim.violation, (s, sim.violation)


def test_the_counterfactual_without_skew_faults_the_same_plans_pass():
    sc = lease("abs")
    for s in LEASE_SEEDS:
        assert run(sc, s, without_skew(plan("abs", s, 6))).violation is None


def test_no_violation_without_skew_across_many_plans():
    sc = lease("abs")
    for s in range(150):
        assert run(sc, s, without_skew(plan("abs", s, 16))).violation is None, s


def test_the_monotonic_variant_survives_the_plans_that_break_the_abs_one():
    sc = lease(None)
    for s in LEASE_SEEDS:
        assert run(sc, s, plan(None, s, 6)).violation is None
    for s in range(150):
        assert run(sc, s, plan(None, s, 16)).violation is None, s


def test_shrink_isolates_the_skew_fault():
    sc, s = lease("abs"), LEASE_SEEDS[0]
    big = plan("abs", s, 16)
    sim = run(sc, s, big)
    assert sim.violation
    small, _ = shrink(sc, s, big, sim.tape.externals())
    assert 1 <= len(small) <= 2 and any(f["kind"] == "skew" for f in small)
    assert run(sc, s, small).violation
    for i in range(len(small)):                           # 1-minimal
        assert run(sc, s, small[:i] + small[i + 1:]).violation is None


def test_a_violating_run_replays_exactly():
    sc, s = lease("abs"), LEASE_SEEDS[0]
    sim = run(sc, s, plan("abs", s, 6))
    assert replay(sim.tape).tape.head == sim.tape.head


def test_search_finds_a_violation_quickly():
    seed, _, sim = next(search(lease("abs"), range(100)))
    assert seed <= 5 and sim.violation


def test_what_if_the_servers_clock_jumps_forward_while_a_client_holds_the_lease():
    """Rewind a healthy run to just after c1 is granted the lease, then fork a world where s's clock steps +200ms."""
    healthy = run(lease("abs"), 0)
    granted = next(e["i"] for e in healthy.tape.events
                   if e["k"] == "deliver" and e["n"] == "c1" and e["d"]["p"]["op"] == "grant")
    skew = {"kind": "skew", "node": "s", "at": 0, "offset": 200}
    twin = rewind(healthy.tape, granted).fork([skew]).run()
    assert "both believe" in twin.violation
    safe = run(lease(None), 0)
    granted = next(e["i"] for e in safe.tape.events
                   if e["k"] == "deliver" and e["n"] == "c1" and e["d"]["p"]["op"] == "grant")
    assert rewind(safe.tape, granted).fork([skew]).run().violation is None   # same what-if, monotonic variant: fine


def test_bug_parameter_is_validated():
    with pytest.raises(ValueError):
        build_scenario("lease", {"bug": "bogus"})


# --- lease_drift: monotonic clocks that run at different speeds -----------------------------
from helpers import DRIFT_SEEDS, lease_drift  # noqa: E402


def drift_plan(bug, seed, n):
    return random_plan(lease_drift(bug), seed, n)


def test_drift_breaks_the_monotonic_lease_that_trusts_the_full_ttl():
    sc = lease_drift("nomargin")
    for s in DRIFT_SEEDS:
        sim = run(sc, s, drift_plan("nomargin", s, 6))
        assert "both believe they hold the lease" in sim.violation and "drift ppm: {" in sim.violation, (s, sim.violation)


def test_counterfactual_without_drift_faults_the_nomargin_variant_passes():
    sc = lease_drift("nomargin")
    for s in range(150):
        faults = [f for f in drift_plan("nomargin", s, 16) if f["kind"] != "drift"]   # skew faults stay in
        assert run(sc, s, faults).violation is None, s


def test_the_margin_variant_survives_drift_skew_and_everything_else():
    sc = lease_drift(None)
    for s in DRIFT_SEEDS + list(range(150)):
        assert run(sc, s, drift_plan(None, s, 16)).violation is None, s


def test_the_margin_is_ttl_times_one_minus_e_over_one_plus_e():
    sc = lease_drift(None)
    base = run(sc, 0)
    grant = next(e for e in base.tape.events if e["k"] == "deliver" and e["n"] == "c1" and e["d"]["p"]["op"] == "grant")
    c1 = rewind(base.tape, grant["i"]).agents["c1"]
    assert c1.valid_until - c1.sent[grant["d"]["p"]["rid"]] == 100 * 800_000 // 1_200_000 == 66


def test_drift_shrinks_to_a_small_plan_that_always_contains_a_drift_fault():
    sc, s = lease_drift("nomargin"), DRIFT_SEEDS[0]
    big = drift_plan("nomargin", s, 6)
    small, _ = shrink(sc, s, big)
    assert any(f["kind"] == "drift" for f in small) and len(small) <= 4
    sim = run(sc, s, small)
    assert replay(sim.tape).tape.head == sim.tape.head
    for i in range(len(small)):
        assert run(sc, s, small[:i] + small[i + 1:]).violation is None


def test_what_if_the_clients_clock_slows_and_the_servers_speeds_up_while_the_lease_is_held():
    both = [{"kind": "drift", "node": "c1", "at": 0, "ppm": -200_000},
            {"kind": "drift", "node": "s", "at": 0, "ppm": 200_000}]
    for bug, broken in (("nomargin", True), (None, False)):
        healthy = run(lease_drift(bug), 0)
        granted = next(e["i"] for e in healthy.tape.events
                       if e["k"] == "deliver" and e["n"] == "c1" and e["d"]["p"]["op"] == "grant")
        twin = rewind(healthy.tape, granted).fork(both).run()
        assert (twin.violation is not None) is broken, (bug, twin.violation)


def test_lease_drift_validates_its_bug_parameter():
    with pytest.raises(ValueError):
        build_scenario("lease_drift", {"bug": "abs"})