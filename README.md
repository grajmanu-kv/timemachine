# timemachine

Deterministic simulation for multi-agent systems: every run is exactly reproducible, every non-deterministic input is recorded, and a failing run can be replayed, minimised, diffed, rewound and forked. Standard library only. The project pins Python >= 3.14; the test suite passes on 3.13 and 3.14.

## Quick start

Layout is `src/timemachine/`; `uv sync` installs it as an editable package and registers the `timemachine` command.

```bash
uv sync
uv run timemachine demo                            # end-to-end walkthrough
uv run pytest                                      # 100 tests; add -m "not slow" to skip the long ones

uv run timemachine search --seeds 2000 --out bug.tape.json   # --faults defaults to 16
uv run timemachine replay bug.tape.json
uv run timemachine shrink bug.tape.json --out min.tape.json
uv run timemachine diff bug.tape.json min.tape.json
uv run timemachine show min.tape.json --start 10
uv run timemachine fork min.tape.json --at 8 --fault '{"kind":"drop","src":"c","dst":"s","n":2}'
uv run timemachine search --fixed                  # the bug-fixed variant: finds nothing

# second scenario: three-node leader election
uv run timemachine search --scenario election                      # safety bug: two leaders in one term
uv run timemachine search --scenario election --param bug=churn    # liveness bug: endless re-elections
uv run timemachine search --scenario election --param bug=null     # correct variant: finds nothing

# third scenario: a lease lock whose bug lives in the clocks
uv run timemachine search --scenario lease                         # wall-clock expiry timestamps + a clock-skew fault
uv run timemachine search --scenario lease --param bug=null        # monotonic durations: finds nothing

# fourth scenario: the same lease on clocks that also run at different speeds
uv run timemachine search --scenario lease_drift                   # monotonic clock, full ttl trusted: drift breaks it
uv run timemachine search --scenario lease_drift --param bug=null  # ttl shortened by the promised rate bound: finds nothing
```

## Repository layout

```
src/timemachine/   the package (tape, simulator, tools, CLI, built-in scenarios: demo_agents, election, lease)
tests/             pytest suite; machinery tests use a tiny `toy` scenario (tests/helpers.py),
                   scenario-specific tests live in tests/scenarios/
scripts/           measure_hit_rate.py reproduces the "Search tuning" table below
```

Tests are not part of the installed package. `tests/helpers.py` holds the fixture seeds, which depend on the exact behaviour of `random_plan` and must be regenerated if plan generation changes.

## Built-in scenarios

| Scenario | Seeded bug | Property that catches it |
|---|---|---|
| `transfer` | Server keeps its de-dup set in volatile memory while the ledger is durable, so a retry after a crash double-debits. | Safety: no request id appears twice in the ledger. |
| `election` `bug=vote` | `voted_for` is not persisted, so a restarted voter can vote twice in one term. | Safety: at most one leader per term. |
| `election` `bug=churn` | Followers never reset their election timer on a heartbeat. | Liveness: once faults have healed, a leader exists and no new election starts in the last 200 ms. |
| `lease` `bug=abs` | The lock server stamps each grant with an absolute expiry from its wall clock and clients trust it against theirs. Correct only while all clocks agree. | Safety: at most one client believes it holds the lease, judged on that client's own clock. Needs a `skew` fault: with the skew faults removed, the same plans pass. |
| `lease_drift` `bug=nomargin` | Durations are measured on the monotonic clock (immune to skew), but the client trusts the full ttl. A slow client clock against a fast server clock makes the client's lease outlive the server's. | Same safety property. Needs a `drift` fault. The correct variant (`bug=null`) shortens the client's belief to `ttl * (1-e)/(1+e)`, at the cost of using only 66% of every lease at e = 20%. |

## Model

| Piece | What it does |
|---|---|
| Virtual clock | Integer milliseconds. A heap orders events by `(time, kind priority, push order)`. No wall-clock reads. |
| Seeded randomness | Network latency and each node's `ctx.rand()` come from separate streams derived from `seed`. |
| Faults | Explicit, serialisable plan: `drop`, `dup`, `delay` target the *n-th message on a link*; `crash`/`restart` and `partition` target time windows. Only `durable` agent state survives a crash. `skew` sets one node's wall-clock offset from an instant onward (see Clocks). |
| Clocks | `ctx.now` is true virtual time, an oracle that never skews. `ctx.clock()` is the node's wall clock: true time plus the offset a `skew` fault last set (absolute, so the last skew wins; it survives crashes). `ctx.monotonic()` never steps. A `drift` fault changes the *rate* of a node's monotonic and wall clocks from an instant on (parts per million, absolute, last wins); local time is kept in exact integer micro-ms so rate changes never lose a remainder. Skew and drift are opt-in per scenario through `Scenario.fault_kinds` (with `skew_offsets` and `drift_ppm`), so scenarios that never read `ctx.clock()` keep their existing seeds and plans. |
| Tape | Append-only event log. Each event's hash covers the previous hash, so equality of two runs up to event *i* is one comparison. |
| `ctx.external(name, fn, *args)` | The only sanctioned door to the outside world (an LLM call, a price lookup, the clock). Live runs call `fn` and record the result; replays feed the recorded value back and never call `fn`. |

## Operations

- **replay**: re-execute from scratch and require every event hash to match; raises `Divergence` at the first mismatch. A corrupted tape is rejected before running.
- **search**: random fault plans over a range of seeds; yields every plan that breaks the scenario's invariant.
- **shrink**: delta-debugging over the fault plan; the result is 1-minimal (removing any single fault makes the run pass).
- **first_divergence**: bisects on chained hashes (O(log n) comparisons) to find where two runs part ways.
- **rewind / fork**: `rewind(tape, i)` re-executes the recorded prefix and stops at the first step boundary where event `i` exists; `.fork(add_faults)` deep-copies that world so you can run a what-if without touching the original. A forked tape replays exactly from scratch.

## Writing your own scenario

```python
from timemachine import Agent, Scenario, register

class Node(Agent):
    def on_start(self, ctx): ...
    def on_message(self, ctx, src, payload): ...   # payloads must be JSON-serialisable
    def on_timer(self, ctx, name): ...

@register
class MyScenario(Scenario):
    name = "mine"
    nodes = ["a", "b"]
    def make_agent(self, node): return Node(node)
    def check(self, sim): return "msg" if bad(sim) else None   # safety: evaluated after every step
    def check_final(self, sim): ...                            # liveness: evaluated once, at the end of the run
```

`check_final` is only meaningful if your scenario guarantees that faults stop and heal well before `duration` (the election scenario keeps a 600 ms quiet tail).

## Search tuning (measured)

Violating plans found per 4000 seeds, by number of faults per random plan (one run per cell, same seed range). Reproduce with `uv run python scripts/measure_hit_rate.py --controls` (about 3 minutes):

| faults per plan | 2 | 4 | 6 | 10 | 16 |
|---|---|---|---|---|---|
| `election` `bug=vote` | 0 | 3 | 9 | 27 | 56 |
| `transfer` | 130 | 585 | 1040 | 1495 | 1552 |
| `lease` `bug=abs` | 840 | 1402 | 1746 | 2104 | 2203 |
| `lease_drift` `bug=nomargin` | 94 | 207 | 290 | 414 | 582 |

- **Search with big plans, then shrink.** Bigger plans find the sparse election bug about 6x as often at 16 faults as at 6, and `shrink` brings failing 16-fault plans back to 2-4 faults on both scenarios, so diagnosis does not get harder. `DEFAULT_FAULTS` is 16 for that reason.
- **Correct variants stayed clean:** 0 violations in 4000 seeds at 16 faults for `election bug=null`, the fixed `transfer`, `lease bug=null` and `lease_drift bug=null` (the script's `--controls`). A fourth control removes the skew faults from the `lease bug=abs` plans: also 0 in 4000, which is the evidence that the violation comes from the clock and not from the protocol.
- **The lease bug saturates; the election bug does not.** Hits flatten at about 55% of seeds (2203 of 4000 at 16 faults) instead of climbing, because skew offsets are absolute: a second skew on the same node overwrites the first rather than adding another chance, and a skew only matters if it lands while a lease is held. The "bigger plans are better" rule above is a measurement of two scenarios, not a law.
- **Drift does not saturate, and its failures are harder to read.** `lease_drift` hits keep climbing with plan size (582 of 4000 at 16 faults) because every extra drift fault is a fresh chance and crashes or drops shift the lease schedule into the vulnerable window. In an ad hoc run over the first 1500 seeds at 16 faults, 210 plans violated; every shrunk plan contained a `drift` fault, but only 52 shrank to that single fault, 84 needed two faults, and the largest needed seven. Skew was a one-fault diagnosis; drift often needs a second fault to line up the timing.
- **The `lease_drift` rates are exaggerated about 1000-2000x** (+/-10-20% against a real oscillator's ~100 ppm) so that a 100 ms lease is sensitive to them. What matters is `ttl x rate difference` compared with network jitter, so a lease of tens of seconds with real rates sits in the same regime; the simulator does not demonstrate that your hardware is that bad.
- **Lease failures shrink to one fault.** In an ad hoc run over the first 400 seeds at 16 faults (not part of the script), 235 plans violated; 233 shrank to a single `skew` fault and 2 to two faults. The three nodes were skewed about equally often (server 108, c1 66, c2 62 of the single-fault results).
- **Swarm testing did not help here.** Giving each plan a random subset of fault kinds (Groce et al., 2012) found the election bug 12 times vs 9 for the baseline (within noise at these counts) and the transfer bug 912 vs 1040 times (worse). Both bugs need two specific fault kinds together, and subset sampling often drops one of them. It is not implemented.
- **Caveats:** two small scenarios, one run per cell, counts near 10 are noisy (roughly +/- 3), and plan sizes above 16 were not tried. Re-measure on your own scenarios before trusting these ratios.

## Limits worth knowing

- **Determinism is only as strong as your agents.** Anything non-deterministic must go through `ctx.external`. Iterating over a `set` of strings can also reorder between processes (hash randomisation); use sorted lists or set `PYTHONHASHSEED`.
- **Fork needs deep-copyable agents** (plain data, no open sockets or threads).
- **Rewind is re-execution, not a snapshot**, so its cost grows with the prefix length.
- **Shrinking removes faults but does not simplify their parameters** (e.g. it will not shorten a crash window).
- **Fault plans are keyed by per-link message index.** Removing a fault can shift which logical message a later fault hits; the shrinker tolerates this because it only asks "does it still fail?".
- **Clocks are modelled as offsets and constant-rate segments.** Temperature-dependent wander, leap-second smearing and NTP slewing are not modelled. `ctx.monotonic()` survives restarts, whereas a real `CLOCK_MONOTONIC` resets on reboot. **Timers fire on true time**, not on the node's drifted clock, so drift changes what a node measures but not when its timers fire (this affects liveness, not the lease safety property). The correct lease variants have no fencing token, so they do not cover a client that pauses (GC, VM freeze) after checking its lease.
- **Single-threaded virtual time.** This models message passing and crashes, not data races inside a node.
- **Validated on four small scenarios** (an idempotency bug, a leader-election bug pair, and two lease-and-clock bugs that share one protocol). The tests show the machinery works; they do not show it scales to large agent systems.
- **Liveness checks on randomised protocols can false-alarm.** "Eventually a leader exists" is evaluated on one finite, seeded run, so a protocol that is correct in expectation can still livelock on a given seed. The election scenario found exactly this in its own correct variant when election timeouts were close to network jitter (kept as a regression test). Treat liveness hits as "investigate", and judge a configuration by its hit *rate* over many seeds, not by a single run.
- **Fault search is sparse for subtle bugs.** Even at 16 faults per plan the election safety bug is hit on about 1.4% of seeds (56 of 4000), so budget thousands of seeds, not dozens.