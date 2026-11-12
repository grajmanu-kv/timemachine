"""Render tapes as one self-contained HTML page: swimlanes, a scrubber, an event log and the fault plan.

    from timemachine import viz
    html = viz.render_html([viz.tape_entry(tape, "my run")])

The page has no external requests: the tapes are embedded as JSON and everything is drawn by inline script.
With several tapes the viewer offers tabs and a "compare with" control that marks the first event whose
hash differs (the same answer `first_divergence` gives).
"""
from __future__ import annotations

import json
from importlib import resources

from .tape import Tape

PLACEHOLDER = "__TM_DATA__"
_UNSET = object()
_HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
)


def template() -> str:
    return resources.files("timemachine").joinpath("viz_template.html").read_text(encoding="utf-8")


def tape_entry(tape: Tape, label: str, *, note: str = "", baseline=_UNSET, file: str | None = None) -> dict:
    """One tape as the viewer wants it.

    `baseline` is the index (in the list given to `render_html`) of the tape to compare against; None
    means no comparison. Left out, later tapes compare against the first.
    """
    bad = tape.verify()
    entry = {
        "label": label,
        "note": note,
        "meta": tape.meta,
        "events": tape.events,
        "chain_ok": bad is None,
        "chain_bad": bad,
    }
    if file:
        entry["file"] = file
    if baseline is not _UNSET:
        entry["baseline"] = baseline
    return entry


def render_html(entries: list[dict], *, fragment: bool = False) -> str:
    """The viewer page for `entries` (see `tape_entry`).

    `fragment=True` omits the document wrapper (doctype, html and head tags), for hosts that add their own.
    """
    # "<" is escaped so that no tape content, however hostile, can close the script element.
    payload = json.dumps({"tapes": entries}, separators=(",", ":")).replace("<", "\\u003c")
    page = template()
    if page.count(PLACEHOLDER) != 1:
        raise RuntimeError("viz_template.html must contain the data placeholder exactly once")
    page = page.replace(PLACEHOLDER, payload, 1)
    return page if fragment else _HEAD + page + "\n</html>\n"