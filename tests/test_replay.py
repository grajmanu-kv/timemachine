import pytest

from timemachine.tape import Divergence
from timemachine.tools import replay, run

import helpers
from helpers import DUP_FIRST, toy


def test_replay_reproduces_a_violation_exactly():
    sim = run(toy(), 1, [DUP_FIRST])
    rep = replay(sim.tape)
    assert rep.tape.head == sim.tape.head
    assert rep.violation == sim.violation == "over-count: 4"


def test_replay_never_calls_the_outside_world(tag_calls):
    sim = run(toy(), 1, [DUP_FIRST])
    assert len(tag_calls) == 3          # the live run really did call out
    replay(sim.tape)
    assert len(tag_calls) == 3          # the replay did not


def test_live_nondeterministic_runs_are_still_exactly_replayable(monkeypatch):
    monkeypatch.setitem(helpers._hooks, "tag", helpers._live_tag)
    a, b = run(toy(), 1, [DUP_FIRST]), run(toy(), 1, [DUP_FIRST])
    assert a.tape.externals() != b.tape.externals()   # the outside world really differed between runs
    assert replay(a.tape).tape.head == a.tape.head
    assert replay(b.tape).tape.head == b.tape.head


def test_a_corrupted_tape_is_rejected_before_running():
    tape = run(toy(), 1, [DUP_FIRST]).tape
    idx = next(e["i"] for e in tape.events if e["k"] == "send")
    tape.events[idx]["d"]["p"]["i"] = 99
    with pytest.raises(Divergence) as ei:
        replay(tape)
    assert ei.value.index == idx
    assert "hash chain" in ei.value.reason


def test_a_valid_tape_whose_metadata_no_longer_matches_is_caught_mid_run():
    tape = run(toy(), 1).tape
    tape.meta["faults"] = [{"kind": "drop", "src": "a", "dst": "b", "n": 0}]   # chain is intact, behaviour differs
    with pytest.raises(Divergence) as ei:
        replay(tape)
    assert ei.value.reason == "event differs from the tape"
    assert ei.value.index > 0


def test_replay_notices_when_the_run_ends_before_the_tape_does():
    tape = run(toy(), 1).tape
    tape.append(999, "timer", "a", {"name": "phantom"})   # a legitimate-looking but impossible extra event
    with pytest.raises(Divergence) as ei:
        replay(tape)
    assert "ended before the tape did" in ei.value.reason