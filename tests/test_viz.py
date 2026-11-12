"""The tape viewer: what gets embedded, that hostile tape content cannot break out of the page, and the CLI."""
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from timemachine import viz
from timemachine.__main__ import main
from timemachine.tape import Tape
from timemachine.tools import run

from helpers import DUP_FIRST, toy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def embedded(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="tm-data">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def tape_file(tmp_path, name="a.tape.json", faults=()):
    path = tmp_path / name
    run(toy(), 1, list(faults)).tape.save(str(path))
    return path


# --- rendering -----------------------------------------------------------------------------
def test_the_page_embeds_every_event_of_every_tape():
    a, b = run(toy(), 1).tape, run(toy(), 1, [DUP_FIRST]).tape
    data = embedded(viz.render_html([viz.tape_entry(a, "plain"), viz.tape_entry(b, "dup", note="why")]))
    assert [t["label"] for t in data["tapes"]] == ["plain", "dup"]
    assert data["tapes"][0]["events"] == a.events and data["tapes"][1]["events"] == b.events
    assert data["tapes"][1]["note"] == "why" and data["tapes"][1]["meta"]["faults"] == [DUP_FIRST]


def test_tape_content_cannot_close_the_script_element_or_inject_markup():
    tape = run(toy(), 1).tape
    tape.meta["params"] = {"x": "</script><script>alert(1)</script><!--"}
    html = viz.render_html([viz.tape_entry(tape, "</script><b>label</b>", note="<img src=x onerror=alert(1)>")])
    assert "</script><script>alert(1)" not in html and "<img src=x" not in html and "<b>label" not in html
    assert embedded(html)["tapes"][0]["meta"]["params"]["x"] == "</script><script>alert(1)</script><!--"   # and it round-trips


def test_full_pages_have_a_document_wrapper_and_fragments_do_not():
    entry = viz.tape_entry(run(toy(), 1).tape, "t")
    full, frag = viz.render_html([entry]), viz.render_html([entry], fragment=True)
    assert full.startswith("<!doctype html>") and full.rstrip().endswith("</html>")
    assert "<!doctype" not in frag.lower() and not re.search(r"<(html|head|body)[\s>]", frag)   # <header> is fine
    assert "<title>" in frag


def test_baseline_is_only_sent_when_given():
    tape = run(toy(), 1).tape
    assert "baseline" not in viz.tape_entry(tape, "t")
    assert viz.tape_entry(tape, "t", baseline=None)["baseline"] is None
    assert viz.tape_entry(tape, "t", baseline=0)["baseline"] == 0


def test_a_tampered_tape_is_flagged_not_silently_drawn():
    tape = run(toy(), 1).tape
    assert viz.tape_entry(tape, "ok")["chain_ok"] is True
    tape.events[3]["d"] = {**tape.events[3]["d"], "tampered": True}
    entry = viz.tape_entry(tape, "bad")
    assert entry["chain_ok"] is False and entry["chain_bad"] == 3


# --- the template --------------------------------------------------------------------------
def test_the_template_has_one_data_slot_and_makes_no_external_requests():
    tpl = viz.template()
    assert tpl.count(viz.PLACEHOLDER) == 1
    urls = set(re.findall(r"https?://[^\s\"')<>]+", tpl))
    assert urls <= {"http://www.w3.org/2000/svg"}, urls            # an SVG namespace is not a request
    assert "@import" not in tpl and "<link" not in tpl and " src=\"" not in tpl


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_viewer_script_parses(tmp_path):
    html = viz.render_html([viz.tape_entry(run(toy(), 1).tape, "t")])
    scripts = [s for s in re.findall(r"<script>(.*?)</script>", html, re.S)]
    assert scripts
    js = tmp_path / "viewer.js"
    js.write_text("\n".join(scripts), encoding="utf-8")
    result = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- CLI -----------------------------------------------------------------------------------
def test_view_writes_one_page_for_several_tapes(tmp_path, capsys):
    a, b = tape_file(tmp_path, "plain.tape.json"), tape_file(tmp_path, "dup.tape.json", [DUP_FIRST])
    out = tmp_path / "v.html"
    assert main(["view", str(a), str(b), "--out", str(out)]) == 0
    assert "viewer written" in capsys.readouterr().out
    data = embedded(out.read_text(encoding="utf-8"))
    assert [t["label"] for t in data["tapes"]] == ["plain", "dup"]       # ".tape.json" is dropped from the label
    assert data["tapes"][1]["file"] == "dup.tape.json"                  # so the page can print a runnable fork command


def test_view_warns_about_a_corrupt_tape_but_still_writes_the_page(tmp_path, capsys):
    path = tape_file(tmp_path)
    raw = json.loads(path.read_text())
    raw["events"][2]["d"]["tampered"] = True
    path.write_text(json.dumps(raw))
    out = tmp_path / "v.html"
    assert main(["view", str(path), "--out", str(out)]) == 0
    assert "hash chain does not verify at event #2" in capsys.readouterr().out
    assert embedded(out.read_text(encoding="utf-8"))["tapes"][0]["chain_ok"] is False


def test_view_of_a_missing_file_is_a_clean_error(tmp_path, capsys):
    assert main(["view", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err


# --- the demo page -------------------------------------------------------------------------
def test_the_demo_page_tells_the_story_the_library_actually_produces():
    spec = importlib.util.spec_from_file_location("build_viz_demo", ROOT / "scripts" / "build_viz_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tapes = embedded(mod.build(fragment=False))["tapes"]
    verdict = {t["label"]: next((e["d"]["msg"] for e in t["events"] if e["k"] == "violation"), None) for t in tapes}
    assert verdict["Healthy run"] is None
    assert "both believe they hold the lease" in verdict["Server clock +200 ms"]
    assert verdict["Same fault, correct design"] is None                 # the claim in that tab's note
    assert "drift ppm" in verdict["Clocks at different speeds"]
    assert verdict["Retry after a crash"].startswith("double debit")
    assert all(t["chain_ok"] for t in tapes)