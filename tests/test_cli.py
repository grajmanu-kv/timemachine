"""The command-line interface, driven through main(argv) so exit codes and output can be asserted."""
import json

import pytest

from timemachine.__main__ import main


def run_cli(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_search_replay_shrink_diff_show_fork_pipeline(tmp_path, capsys):
    bug, mini, forked = (str(tmp_path / n) for n in ("bug.json", "min.json", "fork.json"))

    code, cap = run_cli(capsys, "search", "--scenario", "toy", "--seeds", "200", "--out", bug)
    assert code == 0 and "over-count" in cap.out

    code, cap = run_cli(capsys, "replay", bug)
    assert code == 0 and "replay verified" in cap.out and "violation reproduced" in cap.out

    code, cap = run_cli(capsys, "shrink", bug, "--out", mini)
    assert code == 0 and "faults -> 1" in cap.out

    code, cap = run_cli(capsys, "diff", bug, bug)
    assert code == 0 and "identical" in cap.out

    code, cap = run_cli(capsys, "diff", bug, mini)
    assert code == 0 and ("identical" in cap.out or "first divergence at event" in cap.out)

    code, cap = run_cli(capsys, "show", mini, "--end", "5")
    assert code == 0 and cap.out.startswith("meta:") and cap.out.count("\n  #") == 5

    fault = json.dumps({"kind": "drop", "src": "a", "dst": "b", "n": 2})
    code, cap = run_cli(capsys, "fork", mini, "--at", "3", "--fault", fault, "--out", forked)
    assert code == 0 and "next message index per link" in cap.out
    assert run_cli(capsys, "replay", forked)[0] == 0   # a forked tape is a first-class tape


def test_search_exits_1_when_nothing_is_found(tmp_path, capsys):
    code, cap = run_cli(capsys, "search", "--scenario", "election", "--param", "bug=null",
                        "--seeds", "20", "--out", str(tmp_path / "x.json"))
    assert code == 1 and "no violation" in cap.out
    assert not (tmp_path / "x.json").exists()


def test_replay_of_a_tampered_tape_exits_2(tmp_path, capsys):
    path = tmp_path / "bug.json"
    assert main(["search", "--scenario", "toy", "--seeds", "200", "--out", str(path)]) == 0
    capsys.readouterr()
    raw = json.loads(path.read_text())
    idx = next(e["i"] for e in raw["events"] if e["k"] == "send")
    raw["events"][idx]["d"]["p"]["i"] = 99
    path.write_text(json.dumps(raw))
    code, cap = run_cli(capsys, "replay", str(path))
    assert code == 2 and "REPLAY DIVERGED" in cap.out


@pytest.mark.parametrize("argv, message", [
    (["search", "--scenario", "nope"], "unknown scenario"),
    (["search", "--scenario", "election", "--param", "bug=bogus"], "bug must be"),
    (["search", "--scenario", "election", "--param", "typo=1"], "bad parameters"),
    (["search", "--scenario", "election", "--param", "novalue"], "KEY=VALUE"),
    (["search", "--scenario", "election", "--fixed"], "--fixed only applies"),
    (["replay", "does-not-exist.json"], "does-not-exist.json"),
])
def test_user_errors_exit_2_with_a_message_not_a_traceback(argv, message, capsys):
    code, cap = run_cli(capsys, *argv)
    assert code == 2
    assert message in cap.err


def test_demo_runs_end_to_end(capsys):
    code, cap = run_cli(capsys, "demo")
    assert code == 0
    for marker in ("== 1.", "== 2.", "== 3.", "== 4.", "== 5."):
        assert marker in cap.out


def test_lease_scenario_via_the_cli_shrinks_to_one_skew_fault(tmp_path, capsys):
    bug, mini = str(tmp_path / "bug.json"), str(tmp_path / "min.json")
    code, cap = run_cli(capsys, "search", "--scenario", "lease", "--seeds", "50", "--out", bug)
    assert code == 0 and "both believe they hold the lease" in cap.out
    code, cap = run_cli(capsys, "shrink", bug, "--out", mini)
    assert code == 0 and "faults -> 1" in cap.out and '"kind": "skew"' in cap.out
    code, cap = run_cli(capsys, "show", mini)
    assert code == 0 and " skew " in cap.out
    code, cap = run_cli(capsys, "search", "--scenario", "lease", "--param", "bug=null", "--seeds", "50", "--out", bug)
    assert code == 1


def test_lease_drift_via_the_cli_finds_the_bug_and_the_margin_variant_is_clean(tmp_path, capsys):
    bug = str(tmp_path / "bug.json")
    code, cap = run_cli(capsys, "search", "--scenario", "lease_drift", "--seeds", "100", "--out", bug)
    assert code == 0 and "drift ppm" in cap.out
    code, cap = run_cli(capsys, "search", "--scenario", "lease_drift", "--param", "bug=null", "--seeds", "100", "--out", bug)
    assert code == 1