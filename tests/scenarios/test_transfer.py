"""The transfer demo: a retrying client against a server with a volatile de-dup set."""
from timemachine.tools import DEFAULT_FAULTS, next_link_index, random_plan, replay, rewind, run, search, shrink

from helpers import transfer


def test_search_finds_the_double_debit(fixed_oracle):
    seed, plan, sim = next(search(transfer(), range(300)))
    assert sim.violation.startswith("double debit")


def test_the_fixed_server_survives_the_same_plans(fixed_oracle):
    sc = transfer(fixed=True)
    for seed in range(300):
        sim = run(sc, seed, random_plan(sc, seed, DEFAULT_FAULTS))
        assert sim.violation is None, (seed, sim.violation)


def test_shrinks_to_a_crash_plus_a_lost_or_delayed_message(fixed_oracle):
    seed, plan, _ = next(search(transfer(), range(300)))
    small, _ = shrink(transfer(), seed, plan)
    assert 2 <= len(small) <= 3
    assert any(f["kind"] == "crash" for f in small)   # without a crash the volatile set is never lost


def test_a_run_with_live_non_deterministic_pricing_is_exactly_replayable():
    seed, plan, sim = next(search(transfer(), range(300)))   # real, random price_oracle
    assert sim.tape.externals()["pricing"]
    assert replay(sim.tape).tape.head == sim.tape.head


def test_fork_before_a_client_retry_replays_from_scratch(fixed_oracle):
    seed, plan, _ = next(search(transfer(), range(300)))
    small, _ = shrink(transfer(), seed, plan)
    failing = run(transfer(), seed, small)
    first_retry = next(e["i"] for e in failing.tape.events if e["k"] == "timer" and e["n"] == "c")
    world = rewind(failing.tape, first_retry - 1)
    nxt = next_link_index(world, "c", "s")
    twin = world.fork([{"kind": "drop", "src": "c", "dst": "s", "n": nxt}]).run()
    assert replay(twin.tape).tape.head == twin.tape.head