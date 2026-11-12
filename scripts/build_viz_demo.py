"""Build the demo page for the tape viewer from live runs (nothing is hand-drawn or hand-copied).

    uv run python scripts/build_viz_demo.py                      # writes timemachine-viewer-demo.html
    uv run python scripts/build_viz_demo.py --out demo.html --fragment

Five tapes tell one story about clocks: a healthy lock service, the same run after the server's wall
clock steps, the same fault against the correct design, a rate-drift variant of the bug, and a plain
crash-and-retry bug for contrast. Every tape is produced by the library, so the page cannot drift out
of date with the code.
"""
from __future__ import annotations

import argparse

from timemachine import build_scenario, demo_agents, random_plan, rewind, run, shrink, viz


def first_grant_to_c1(tape) -> int:
    return next(e["i"] for e in tape.events
                if e["k"] == "deliver" and e["n"] == "c1" and e["d"]["p"]["op"] == "grant")


def what_if(scenario, faults):
    """Run seed 0 healthy, rewind to just after c1 receives its lease, and fork in `faults`."""
    healthy = run(scenario, 0)
    twin = rewind(healthy.tape, first_grant_to_c1(healthy.tape)).fork(faults).run()
    return healthy, twin


def build(fragment: bool) -> str:
    demo_agents.price_oracle = lambda order_id: 10 + order_id   # pin the one live call so builds are reproducible

    skew = [{"kind": "skew", "node": "s", "at": 0, "offset": 200}]
    healthy, wall = what_if(build_scenario("lease", {"bug": "abs"}), skew)
    _, monotonic = what_if(build_scenario("lease", {"bug": None}), skew)
    drift = [{"kind": "drift", "node": "c1", "at": 0, "ppm": -200_000},
             {"kind": "drift", "node": "s", "at": 0, "ppm": 200_000}]
    _, rate = what_if(build_scenario("lease_drift", {"bug": "nomargin"}), drift)

    pay = build_scenario("transfer")
    plan = random_plan(pay, 2, 16)
    found = run(pay, 2, plan)
    small, _ = shrink(pay, 2, plan, found.tape.externals())
    retry = run(pay, 2, small, found.tape.externals())

    entries = [
        viz.tape_entry(healthy.tape, "Healthy run", baseline=None, note=(
            "A lock server hands out a 100 ms lease to one of two clients at a time. c1 asks first, c2 keeps retrying, "
            "and they take turns, each renewing twice before going quiet. Nothing is injected and nothing goes wrong: "
            "at every moment at most one client believes it holds the lease.\n\n"
            "Press play, or drag along the timeline. Everything right of the teal line has not happened yet. "
            "The other tabs start from this run.")),
        viz.tape_entry(wall.tape, "Server clock +200 ms", baseline=0, note=(
            "The healthy run was rewound to the instant c1 received its lease, and one fault was added to the future: "
            "the server's wall clock steps forward 200 ms. This server stamps each grant with an absolute expiry read "
            "from its own clock, so the step makes c1's live lease look expired. c2's next retry is granted the same lock "
            "while c1, judged on c1's own clock, still holds it.\n\n"
            "The orange dashed line marks where this run first differs from the healthy one. "
            "Everything before it is event-for-event identical (the chain hashes match).")),
        viz.tape_entry(monotonic.tape, "Same fault, correct design", baseline=0, note=(
            "Same rewind, same fault, but both sides now measure durations on a clock that cannot step. "
            "The skew is recorded and changes nothing anyone can observe: the run reaches its end with no violation. "
            "The two designs behave identically until a clock lies, which is why this kind of bug survives ordinary testing.")),
        viz.tape_entry(rate.tape, "Clocks at different speeds", baseline=0, note=(
            "A monotonic clock is immune to steps, but not to a different rate. From t = 20 ms c1's clock runs 20% slow and "
            "the server's 20% fast. c1 counts out its 100 ms lease in about 125 real milliseconds; the server counts out the "
            "same lease in about 83. c2 is granted the lock while c1 still believes it has it.\n\n"
            "The correct variant shortens the client's belief by the promised rate bound, so it survives this same fault plan. "
            "The price is that a client may use only about two thirds of each lease.")),
        viz.tape_entry(retry.tape, "Retry after a crash", baseline=None, note=(
            "A different bug, for the other fault glyphs. A client pays order 1 and retries until acknowledged. "
            "The search found a 16-fault plan; shrinking reduced it to two faults. The server's first acknowledgement is lost, "
            "and the server crashes at 12 ms. Its record of what it already charged lives in volatile memory, so after the "
            "restart the client's retry is charged a second time.")),
    ]
    return viz.render_html(entries, fragment=fragment)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="timemachine-viewer-demo.html")
    ap.add_argument("--fragment", action="store_true", help="omit the doctype/html/head wrapper (for hosts that add their own)")
    a = ap.parse_args()
    html = build(a.fragment)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{a.out}: {len(html) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())