"""Leader election: a safety bug (vote not persisted) and a liveness bug (timers never reset)."""
import pytest

from timemachine.sim import build_scenario
from timemachine.tools import DEFAULT_FAULTS, random_plan, replay, run, search, shrink

from helpers import BIG_PLAN_SEEDS, VOTE_SEEDS, elect


# --- safety: bug="vote" --------------------------------------------------------------------
def test_vote_bug_found_on_known_seeds():
    for s in VOTE_SEEDS:
        sim = run(elect("vote"), s, random_plan(elect("vote"), s, 6))
        assert sim.violation.startswith("two leaders"), (s, sim.violation)


def test_same_plans_pass_when_the_vote_is_persisted():
    for s in VOTE_SEEDS:
        sim = run(elect(None), s, random_plan(elect(None), s, 6))
        assert sim.violation is None, (s, sim.violation)


def test_search_rediscovers_the_first_known_seed():
    seed, _, sim = next(search(elect("vote"), range(VOTE_SEEDS[0] + 1), 6))
    assert seed == VOTE_SEEDS[0]
    assert sim.violation.startswith("two leaders")


def test_shrink_is_small_one_minimal_and_replayable():
    sc, s = elect("vote"), VOTE_SEEDS[0]
    small, _ = shrink(sc, s, random_plan(sc, s, 6))
    assert len(small) <= 3
    sim = run(sc, s, small)
    assert sim.violation.startswith("two leaders")
    assert replay(sim.tape).tape.head == sim.tape.head
    for i in range(len(small)):
        assert run(sc, s, small[:i] + small[i + 1:]).violation is None


def test_rejects_an_unknown_bug_name():
    with pytest.raises(ValueError):
        build_scenario("election", {"bug": "bogus"})


# --- liveness: bug="churn" -----------------------------------------------------------------
def test_churn_bug_fails_liveness_with_no_faults_at_all():
    sim = run(elect("churn"), 0)
    assert sim.violation.startswith("liveness"), sim.violation
    last = sim.tape.events[-1]
    assert (last["k"], last["t"]) == ("violation", 1000)   # recorded at the end of the run


def test_churn_shrinks_to_an_empty_plan():
    sc = elect("churn")
    small, _ = shrink(sc, 0, random_plan(sc, 0, 6))
    assert small == []


def test_a_liveness_failure_replays_exactly():
    sim = run(elect("churn"), 0)
    assert replay(sim.tape).tape.head == sim.tape.head


def test_correct_variant_is_clean_under_random_faults():
    sc = elect(None)
    for s in range(150):
        sim = run(sc, s, random_plan(sc, s, 6))
        assert sim.violation is None, (s, sim.violation)


def test_narrow_timeouts_livelock_the_correct_variant(monkeypatch):
    """Regression for a real finding: timeouts close to network jitter cause endless split votes."""
    from timemachine import election

    monkeypatch.setattr(election, "ELECTION_MIN", 50)
    monkeypatch.setattr(election, "ELECTION_MAX", 62)
    sc = elect(None)
    sim = run(sc, 3572, random_plan(sc, 3572, 6))
    assert sim.violation == "liveness: no leader at the end of the run"


# --- big plans -----------------------------------------------------------------------------
def test_big_plans_hit_and_shrink_to_a_handful():
    sc = elect("vote")
    for s in BIG_PLAN_SEEDS:
        plan = random_plan(sc, s, DEFAULT_FAULTS)
        assert len(plan) == DEFAULT_FAULTS
        assert run(sc, s, plan).violation.startswith("two leaders"), s
        small, _ = shrink(sc, s, plan)
        assert len(small) <= 4, (s, small)
        for i in range(len(small)):
            assert run(sc, s, small[:i] + small[i + 1:]).violation is None, (s, i)


def test_correct_variant_stays_clean_with_big_plans():
    sc = elect(None)
    for s in range(100):
        sim = run(sc, s, random_plan(sc, s, DEFAULT_FAULTS))
        assert sim.violation is None, (s, sim.violation)


@pytest.mark.slow
def test_correct_variant_is_clean_over_two_thousand_big_plans():
    sc = elect(None)
    assert next(search(sc, range(2000)), None) is None