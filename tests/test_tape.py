import pytest

from timemachine.tape import Tape, first_divergence
from timemachine.tools import random_plan, run

from helpers import DUP_FIRST, toy


def test_fresh_run_has_a_valid_chain():
    assert run(toy(), 1).tape.verify() is None


def test_tampering_with_an_event_is_detected_at_that_event():
    tape = run(toy(), 1, [DUP_FIRST]).tape
    idx = next(e["i"] for e in tape.events if e["k"] == "send")
    tape.events[idx]["d"]["p"]["i"] = 99
    assert tape.verify() == idx


def test_save_load_roundtrip(tmp_path):
    tape = run(toy(), 1, [DUP_FIRST]).tape
    path = tmp_path / "t.json"
    tape.save(str(path))
    loaded = Tape.load(str(path))
    assert loaded.head == tape.head
    assert loaded.meta == tape.meta
    assert loaded.verify() is None


def test_externals_are_grouped_by_name_in_order(tag_calls):
    tape = run(toy(), 1).tape
    assert tape.externals() == {"tag": [1000, 1001, 1002]}


def test_identical_runs_do_not_diverge():
    assert first_divergence(run(toy(), 4).tape, run(toy(), 4).tape) is None


def test_bisection_matches_a_linear_scan():
    for seed in range(25):
        a = run(toy(), seed, random_plan(toy(), seed, 4)).tape
        b = run(toy(), seed, random_plan(toy(), seed + 1000, 4)).tape
        linear = next((i for i in range(min(len(a), len(b))) if a.events[i]["h"] != b.events[i]["h"]), None)
        got = first_divergence(a, b)
        if linear is None:
            assert got is None or got[0] == min(len(a), len(b))
        else:
            assert got[0] == linear


def test_a_strict_prefix_diverges_at_the_shorter_length():
    full = run(toy(), 1, [DUP_FIRST]).tape
    short = Tape(full.meta)
    for ev in full.events[:10]:
        short.append(ev["t"], ev["k"], ev["n"], ev["d"])
    i, ea, eb = first_divergence(short, full)
    assert i == 10
    assert ea is None and eb["i"] == 10


def test_divergence_reports_the_events_on_both_sides():
    a = run(toy(), 1).tape
    b = run(toy(), 1, [{"kind": "drop", "src": "a", "dst": "b", "n": 0}]).tape
    i, ea, eb = first_divergence(a, b)
    assert ea["i"] == eb["i"] == i
    assert ea["h"] != eb["h"]
    assert (eb["k"], eb["d"].get("why")) == ("drop", "fault")