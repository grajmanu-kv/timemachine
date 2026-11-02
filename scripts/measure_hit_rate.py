"""Reproduce the README's "Search tuning" table: violating plans found per N seeds, by faults per plan.

    uv run python scripts/measure_hit_rate.py                  # N=4000, sizes 2 4 6 10 16 (about 2 minutes)
    uv run python scripts/measure_hit_rate.py --seeds 500      # quick look, noisier
    uv run python scripts/measure_hit_rate.py --controls       # also confirm the correct variants stay clean
    uv run python scripts/measure_hit_rate.py --only lease     # just the rows whose label contains "lease"

Results depend on the exact behaviour of timemachine.tools.random_plan and on the scenarios, so
re-run this after changing either and update the README table and the fixture seeds in tests/helpers.py.
"""
from __future__ import annotations

import argparse
import time

from timemachine import build_scenario, random_plan, run

CASES = {   # label -> (scenario name, params, violation prefix that counts as a hit)
    "election bug=vote": ("election", {"bug": "vote"}, "two leaders"),
    "transfer": ("transfer", {}, "double debit"),
    "lease bug=abs": ("lease", {"bug": "abs"}, "c1 and c2 both believe"),
    "lease_drift bug=nomargin": ("lease_drift", {"bug": "nomargin"}, "c1 and c2 both believe"),
}
CONTROLS = {   # label -> (scenario name, params, fault kinds removed from every plan): must produce zero violations
    "election bug=null": ("election", {"bug": None}, ()),
    "transfer fixed=True": ("transfer", {"fixed": True}, ()),
    "lease bug=null": ("lease", {"bug": None}, ()),
    "lease bug=abs, skew faults removed from the plans": ("lease", {"bug": "abs"}, ("skew",)),
    "lease_drift bug=null": ("lease_drift", {"bug": None}, ()),
    "lease_drift bug=nomargin, drift faults removed from the plans": ("lease_drift", {"bug": "nomargin"}, ("drift",)),
}


def count_hits(sc, n_seeds: int, n_faults: int, prefix: str) -> int:
    hits = 0
    for seed in range(n_seeds):
        sim = run(sc, seed, random_plan(sc, seed, n_faults))
        if sim.violation and sim.violation.startswith(prefix):
            hits += 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 10, 16])
    ap.add_argument("--only", metavar="TEXT", help="run only the cases and controls whose label contains TEXT")
    ap.add_argument("--controls", action="store_true", help="also run the correct variants at the largest size")
    a = ap.parse_args()

    start = time.time()
    print(f"violating plans found in {a.seeds} seeds, by faults per plan\n")
    print("| scenario | " + " | ".join(str(n) for n in a.sizes) + " |")
    print("|---|" + "---|" * len(a.sizes))
    for label, (name, params, prefix) in CASES.items():
        if a.only and a.only not in label:
            continue
        sc = build_scenario(name, params)
        row = [count_hits(sc, a.seeds, n, prefix) for n in a.sizes]
        print(f"| {label} | " + " | ".join(str(h) for h in row) + " |", flush=True)

    status = 0
    if a.controls:
        n = max(a.sizes)
        print(f"\ncontrols at {n} faults per plan (every count must be 0):")
        for label, (name, params, strip) in CONTROLS.items():
            if a.only and a.only not in label:
                continue
            sc = build_scenario(name, params)
            bad = sum(1 for seed in range(a.seeds)
                      if run(sc, seed, [f for f in random_plan(sc, seed, n) if f["kind"] not in strip]).violation)
            print(f"  {label}: {bad} violations in {a.seeds} seeds")
            status |= bad != 0
    print(f"\n({time.time() - start:.0f}s)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())