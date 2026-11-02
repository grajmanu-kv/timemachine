"""Run from the directory containing the package:  python -m unittest timemachine.test_timemachine -v"""
import os
import tempfile
import unittest
from unittest import mock

from . import demo_agents
from .sim import build_scenario
from .tape import Divergence, Tape, first_divergence
from .tools import DEFAULT_FAULTS, next_link_index, random_plan, replay, rewind, run, search, shrink

DET = {"fixed": False}


def bug():
    return build_scenario("transfer", DET)


def deterministic_oracle(order_id):
    return 10 + order_id


class Base(unittest.TestCase):
    def setUp(self):
        # Make the "outside world" deterministic for tests that compare independent runs.
        p = mock.patch.object(demo_agents, "price_oracle", deterministic_oracle)
        p.start()
        self.addCleanup(p.stop)

    def find_failure(self, n_faults=6):
        hit = next(search(bug(), range(3000), n_faults), None)
        self.assertIsNotNone(hit, "search should find the planted bug")
        return hit


class TestDeterminism(Base):
    def test_same_seed_same_plan_same_tape(self):
        plan = random_plan(bug(), 7, 6)
        a, b = run(bug(), 7, plan), run(bug(), 7, plan)
        self.assertEqual(a.tape.head, b.tape.head)
        self.assertEqual(len(a.tape), len(b.tape))

    def test_different_seeds_differ(self):
        self.assertNotEqual(run(bug(), 1).tape.head, run(bug(), 2).tape.head)

    def test_clean_run_satisfies_invariant(self):
        self.assertIsNone(run(bug(), 3).violation)


class TestRecordReplay(Base):
    def test_replay_reproduces_failure_exactly(self):
        seed, plan, sim = self.find_failure()
        rep = replay(sim.tape)
        self.assertEqual(rep.tape.head, sim.tape.head)
        self.assertEqual(rep.violation, sim.violation)

    def test_replay_never_calls_real_external(self):
        seed, plan, sim = self.find_failure()
        with mock.patch.object(demo_agents, "price_oracle", side_effect=AssertionError("real call during replay")):
            replay(sim.tape)  # would raise if the oracle were invoked

    def test_live_nondeterministic_run_is_still_exactly_replayable(self):
        import random

        def wild(order_id):
            return 10 + random.SystemRandom().randrange(1_000_000)

        with mock.patch.object(demo_agents, "price_oracle", wild):
            sim = run(bug(), 5, random_plan(bug(), 5, 4))
        self.assertTrue(sim.tape.externals()["pricing"])  # the outside values were captured
        self.assertEqual(replay(sim.tape).tape.head, sim.tape.head)

    def test_tampered_payload_is_detected(self):
        seed, plan, sim = self.find_failure()
        tape = sim.tape
        idx = next(e["i"] for e in tape.events if e["k"] == "send")
        tape.events[idx]["d"]["p"]["amt"] = 999
        self.assertEqual(tape.verify(), idx)
        with self.assertRaises(Divergence) as cm:
            replay(tape)
        self.assertEqual(cm.exception.index, idx)

    def test_save_load_roundtrip(self):
        seed, plan, sim = self.find_failure()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            sim.tape.save(path)
            loaded = Tape.load(path)
        self.assertEqual(loaded.head, sim.tape.head)
        self.assertIsNone(loaded.verify())
        replay(loaded)


class TestDivergence(Base):
    def test_bisection_matches_linear_scan(self):
        for seed in range(25):
            a = run(bug(), seed, random_plan(bug(), seed, 5)).tape
            b = run(bug(), seed, random_plan(bug(), seed + 1000, 5)).tape
            linear = next((i for i in range(min(len(a), len(b))) if a.events[i]["h"] != b.events[i]["h"]), None)
            got = first_divergence(a, b)
            if linear is None:
                self.assertTrue(got is None or got[0] == min(len(a), len(b)))
            else:
                self.assertEqual(got[0], linear)

    def test_identical_tapes(self):
        a = run(bug(), 4).tape
        self.assertIsNone(first_divergence(a, run(bug(), 4).tape))

    def test_prefix_reports_length_of_shorter(self):
        seed, plan, sim = self.find_failure()
        full = sim.tape
        short = Tape(full.meta)
        for ev in full.events[:10]:
            short.append(ev["t"], ev["k"], ev["n"], ev["d"])
        i, ea, eb = first_divergence(short, full)
        self.assertEqual(i, 10)
        self.assertIsNone(ea)


class TestTimeTravel(Base):
    def test_rewind_matches_original_prefix(self):
        seed, plan, sim = self.find_failure()
        k = len(sim.tape) // 2
        world = rewind(sim.tape, k)
        n = len(world.tape)
        self.assertGreater(n, k)
        self.assertEqual(world.tape.head, sim.tape.events[n - 1]["h"])

    def test_fork_without_faults_reaches_same_end(self):
        seed, plan, sim = self.find_failure()
        world = rewind(sim.tape, len(sim.tape) // 2)
        twin = world.fork().run()
        self.assertEqual(twin.tape.head, sim.tape.head)

    def test_fork_is_isolated_from_parent(self):
        seed, plan, sim = self.find_failure()
        world = rewind(sim.tape, 5)
        before = (len(world.tape), world.tape.head, world.now)
        world.fork().run()
        self.assertEqual(before, (len(world.tape), world.tape.head, world.now))

    def test_forked_tape_replays_from_scratch(self):
        seed, plan, sim = self.find_failure()
        small, _ = shrink(bug(), seed, plan)
        failing = run(bug(), seed, small, sim.tape.externals())
        timers = [e["i"] for e in failing.tape.events if e["k"] == "timer" and e["n"] == "c"]
        world = rewind(failing.tape, timers[0] - 1)
        nxt = next_link_index(world, "c", "s")
        twin = world.fork([{"kind": "drop", "src": "c", "dst": "s", "n": nxt}]).run()
        self.assertEqual(replay(twin.tape).tape.head, twin.tape.head)

    def test_fork_rejects_faults_aimed_at_the_past(self):
        seed, plan, sim = self.find_failure()
        world = rewind(sim.tape, len(sim.tape) - 2)
        with self.assertRaises(ValueError):
            world.fork([{"kind": "drop", "src": "c", "dst": "s", "n": 0}])


class TestSearchAndShrink(Base):
    def test_fixed_variant_survives_the_same_search(self):
        fixed = build_scenario("transfer", {"fixed": True})
        self.assertIsNone(next(search(fixed, range(3000), 6), None))

    def test_shrink_is_small_and_one_minimal(self):
        seed, plan, sim = self.find_failure()
        small, _ = shrink(bug(), seed, plan, sim.tape.externals())
        self.assertLessEqual(len(small), len(plan))
        self.assertLessEqual(len(small), 3)
        self.assertIsNotNone(run(bug(), seed, small, sim.tape.externals()).violation)
        for i in range(len(small)):
            cand = small[:i] + small[i + 1:]
            self.assertIsNone(run(bug(), seed, cand, sim.tape.externals()).violation,
                              f"removing fault {i} should make the run pass")

    def test_shrink_rejects_passing_plan(self):
        with self.assertRaises(ValueError):
            shrink(bug(), 3, [])


def elect(bug):
    return build_scenario("election", {"bug": bug})


# Seeds where random_plan(seed, 6) trips the vote bug, found by `search` over range(4000).
VOTE_SEEDS = [358, 672, 931]


class TestElectionSafety(unittest.TestCase):
    def test_vote_bug_found_on_known_seeds(self):
        for s in VOTE_SEEDS:
            sim = run(elect("vote"), s, random_plan(elect("vote"), s, 6))
            self.assertTrue(sim.violation.startswith("two leaders"), (s, sim.violation))

    def test_same_plans_pass_when_the_vote_is_persisted(self):
        for s in VOTE_SEEDS:
            sim = run(elect(None), s, random_plan(elect(None), s, 6))
            self.assertIsNone(sim.violation, (s, sim.violation))

    def test_search_rediscovers_the_first_known_seed(self):
        seed, _, sim = next(search(elect("vote"), range(VOTE_SEEDS[0] + 1), 6))
        self.assertEqual(seed, VOTE_SEEDS[0])
        self.assertTrue(sim.violation.startswith("two leaders"))

    def test_shrink_is_small_one_minimal_and_replayable(self):
        sc, s = elect("vote"), VOTE_SEEDS[0]
        small, _ = shrink(sc, s, random_plan(sc, s, 6))
        self.assertLessEqual(len(small), 3)
        sim = run(sc, s, small)
        self.assertTrue(sim.violation.startswith("two leaders"))
        self.assertEqual(replay(sim.tape).tape.head, sim.tape.head)
        for i in range(len(small)):
            self.assertIsNone(run(sc, s, small[:i] + small[i + 1:]).violation)

    def test_rejects_unknown_bug_name(self):
        with self.assertRaises(ValueError):
            build_scenario("election", {"bug": "bogus"})


class TestLiveness(unittest.TestCase):
    def test_churn_bug_fails_liveness_with_no_faults_at_all(self):
        sim = run(elect("churn"), 0)
        self.assertTrue(sim.violation.startswith("liveness"), sim.violation)
        last = sim.tape.events[-1]
        self.assertEqual((last["k"], last["t"]), ("violation", 1000))  # recorded at the end of the run

    def test_churn_shrinks_to_an_empty_plan(self):
        sc = elect("churn")
        small, _ = shrink(sc, 0, random_plan(sc, 0, 6))
        self.assertEqual(small, [])

    def test_liveness_failure_replays_exactly(self):
        sim = run(elect("churn"), 0)
        self.assertEqual(replay(sim.tape).tape.head, sim.tape.head)

    def test_correct_variant_is_clean_under_random_faults(self):
        sc = elect(None)
        for s in range(150):
            sim = run(sc, s, random_plan(sc, s, 6))
            self.assertIsNone(sim.violation, (s, sim.violation))

    def test_narrow_timeouts_livelock_the_correct_variant(self):
        """Regression for a real finding: timeouts close to network jitter cause endless split votes."""
        from . import election

        sc = elect(None)
        with mock.patch.object(election, "ELECTION_MIN", 50), mock.patch.object(election, "ELECTION_MAX", 62):
            sim = run(sc, 3572, random_plan(sc, 3572, 6))
        self.assertEqual(sim.violation, "liveness: no leader at the end of the run")

    def test_safety_only_scenarios_are_unaffected(self):
        self.assertIsNone(run(bug(), 3).violation)  # transfer defines no check_final


class TestPlanSize(unittest.TestCase):
    """Search with big fault plans (more hits), then shrink to a few faults (easy to read)."""

    # election bug=vote with DEFAULT_FAULTS-sized plans; found by search over range(1500)
    BIG_PLAN_SEEDS = [203, 212, 244]

    def test_default_plan_size_is_the_measured_one(self):
        self.assertEqual(DEFAULT_FAULTS, 16)

    def test_big_plans_hit_and_shrink_to_a_handful(self):
        sc = elect("vote")
        for s in self.BIG_PLAN_SEEDS:
            plan = random_plan(sc, s, DEFAULT_FAULTS)
            self.assertEqual(len(plan), DEFAULT_FAULTS)
            self.assertTrue(run(sc, s, plan).violation.startswith("two leaders"), s)
            small, _ = shrink(sc, s, plan)
            self.assertLessEqual(len(small), 4, (s, small))
            for i in range(len(small)):
                self.assertIsNone(run(sc, s, small[:i] + small[i + 1:]).violation, (s, i))

    def test_correct_variants_stay_clean_with_big_plans(self):
        for sc in (elect(None), build_scenario("transfer", {"fixed": True})):
            for s in range(100):
                sim = run(sc, s, random_plan(sc, s, DEFAULT_FAULTS))
                self.assertIsNone(sim.violation, (sc.name, s, sim.violation))


if __name__ == "__main__":
    unittest.main()