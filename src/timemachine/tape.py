"""Hash-chained event tape.

Every event stores a hash that covers the event *and* the hash of the event
before it, so "are these two runs identical up to event i?" is a single hash
comparison. That makes finding the first divergence between two runs an
O(log n) bisection instead of a linear scan.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS = "tm1"


def canon(obj: Any) -> str:
    """Canonical JSON: stable key order, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _chain(prev: str, ev: dict) -> str:
    return hashlib.sha256((prev + canon(ev)).encode()).hexdigest()[:16]


class Divergence(Exception):
    """A run stopped matching the tape it was being checked against."""

    def __init__(self, index: int, expected: dict | None, actual: dict | None, reason: str = ""):
        self.index, self.expected, self.actual, self.reason = index, expected, actual, reason
        super().__init__(f"divergence at event #{index}" + (f": {reason}" if reason else ""))


class Tape:
    def __init__(self, meta: dict, expected: "Tape | None" = None):
        self.meta = meta
        self.events: list[dict] = []
        self.expected = expected  # when set, every appended event is checked against it
        self._head = GENESIS

    # -- recording ---------------------------------------------------------
    def append(self, t: int, kind: str, node: str, data: dict) -> dict:
        ev = {"i": len(self.events), "t": t, "k": kind, "n": node, "d": data}
        h = _chain(self._head, ev)
        if self.expected is not None:
            exp = self.expected.events
            if ev["i"] >= len(exp):
                raise Divergence(ev["i"], None, {**ev, "h": h}, "run is longer than the tape")
            if exp[ev["i"]]["h"] != h:
                raise Divergence(ev["i"], exp[ev["i"]], {**ev, "h": h}, "event differs from the tape")
        ev["h"] = h
        self.events.append(ev)
        self._head = h
        return ev

    @property
    def head(self) -> str:
        return self._head

    def __len__(self) -> int:
        return len(self.events)

    # -- queries -----------------------------------------------------------
    def externals(self) -> dict[str, list]:
        """Recorded non-deterministic inputs, in order, grouped by name."""
        out: dict[str, list] = {}
        for ev in self.events:
            if ev["k"] == "external":
                out.setdefault(ev["d"]["name"], []).append(ev["d"]["value"])
        return out

    def verify(self) -> int | None:
        """Recompute the chain. Returns the first index whose hash is wrong, or None if intact."""
        prev = GENESIS
        for ev in self.events:
            body = {k: v for k, v in ev.items() if k != "h"}
            prev = _chain(prev, body)
            if prev != ev["h"]:
                return ev["i"]
        return None

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"meta": self.meta, "events": self.events}, f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Tape":
        with open(path) as f:
            raw = json.load(f)
        t = cls(raw["meta"])
        t.events = raw["events"]
        t._head = t.events[-1]["h"] if t.events else GENESIS
        return t

    # -- display -----------------------------------------------------------
    @staticmethod
    def fmt(ev: dict | None) -> str:
        if ev is None:
            return "<end of tape>"
        body = canon(ev["d"])
        if len(body) > 78:
            body = body[:75] + "..."
        return f"#{ev['i']:<4} t={ev['t']:<5} {ev['k']:<10} {ev['n']:<3} {body}"


def first_divergence(a: Tape, b: Tape) -> tuple[int, dict | None, dict | None] | None:
    """Index of the first event where two tapes differ, found by bisecting on chained hashes.

    Returns None if the tapes are identical. If one tape is a strict prefix of the
    other, the index is the length of the shorter one.
    """
    n = min(len(a), len(b))
    lo, hi = 0, n  # smallest i in [0, n] where hashes differ (n means "none within overlap")
    while lo < hi:
        mid = (lo + hi) // 2
        if a.events[mid]["h"] != b.events[mid]["h"]:
            hi = mid
        else:
            lo = mid + 1
    if lo == n and len(a) == len(b):
        return None
    ea = a.events[lo] if lo < len(a) else None
    eb = b.events[lo] if lo < len(b) else None
    return lo, ea, eb