"""Scenario: a lease lock service and two clients (c1, c2) that compete for it.

Safety: at most one client believes it holds the lease at any instant, where "believes" is
        judged on the client's own clock (what the client's code would actually see).

The server `s` grants the lease for LEASE_MS to whoever asks while it is free (never granted, expired,
or already held by the asker). Clients acquire, renew twice, go quiet, and later try again, so the
lease changes hands naturally during a run. Two variants, selected with the `bug` parameter:

  "abs" : the server stamps each grant with an absolute expiry read from ITS wall clock, and the
          client trusts that stamp against ITS wall clock. This is correct only while every clock
          agrees. A forward step of the server's clock expires a live lease early; a client clock
          that runs behind believes in the lease after the server has given it away.
  None  : each side measures a *duration* on a clock that cannot step (`ctx.monotonic()`). The client
          believes in the lease until (time it SENT the request) + ttl, which is never later than the
          server's expiry (time it GRANTED) + ttl, whatever the network does. Wall-clock skew is
          irrelevant, so a `skew` fault cannot break it.

The headline property is the counterfactual: with the skew faults removed from a violating plan,
the "abs" variant passes. The bug is in the clocks, not in the protocol logic.

A second scenario, `lease_drift`, adds clocks that run at different RATES (a `drift` fault) and shows
that a monotonic clock is not enough by itself:

  "nomargin" : durations on the monotonic clock, but the client trusts the full ttl. If the client's
               clock runs slow and the server's runs fast, the client's lease outlives the server's.
  None       : the client shortens its belief to ttl * (1 - e) / (1 + e), where e is the largest rate
               error the deployment promises (`max_drift_ppm`). That bounds the client's real-time
               belief below the server's real-time expiry for ANY sequence of rate changes within +/-e.
               The price is availability: the client may use only ~67% of every lease at e = 20%.

Honest limits. In the simulator `ctx.monotonic()` survives restarts; a real CLOCK_MONOTONIC resets on
reboot. Timers fire on true time, not on the node's drifted clock, so drift changes what a node
measures but not when its timers fire. A real service that cannot rule out a paused or very slow
client should also use fencing tokens, which this model does not include.
"""
from __future__ import annotations

from .sim import Agent, Ctx, Scenario, register

LEASE_MS = 100        # server-side lease length
RETRY_MS = 25         # a client that was denied asks again this often
RENEW_MS = 60         # a client holding the lease renews this often...
RENEWALS = 2          # ...this many times, then goes quiet
IDLE_MS = 200         # ...and tries again after being quiet this long
START_MS = {"c1": 1, "c2": 30}


class Server(Agent):
    def __init__(self, name: str, bug: str | None):
        super().__init__(name)
        self.bug = bug

    def on_start(self, ctx: Ctx) -> None:
        self.durable.setdefault("holder", None)
        self.durable.setdefault("exp", 0)       # in the clock domain the variant uses

    def on_message(self, ctx: Ctx, src: str, p) -> None:
        if p["op"] != "acq":
            return
        d = self.durable
        now = ctx.clock() if self.bug == "abs" else ctx.monotonic()
        if d["holder"] in (None, src) or now >= d["exp"]:
            d["holder"], d["exp"] = src, now + LEASE_MS
            ctx.send(src, {"op": "grant", "rid": p["rid"], "ttl": LEASE_MS, "exp": d["exp"]})
        else:
            ctx.send(src, {"op": "deny", "rid": p["rid"]})


class Client(Agent):
    def __init__(self, name: str, bug: str | None, max_ppm: int = 0):
        super().__init__(name)
        self.bug, self.max_ppm = bug, max_ppm
        self.valid_until: int | None = None   # volatile: a crashed client holds nothing
        self.sent: dict[int, int] = {}        # request id -> monotonic time it was sent (correct variant)
        self.holding = False
        self.renewals_left = 0

    def on_start(self, ctx: Ctx) -> None:
        self.durable.setdefault("rid", 0)     # durable, so a request id is never reused after a restart
        ctx.set_timer("try", START_MS[self.name])

    def holds(self, sim) -> bool:
        """Does this client currently believe it holds the lease, on the clock its own code would read?"""
        if self.valid_until is None:
            return False
        now = sim.clock(self.name) if self.bug == "abs" else sim.monotonic(self.name)
        return now < self.valid_until

    def _request(self, ctx: Ctx) -> None:
        self.durable["rid"] += 1
        rid = self.durable["rid"]
        if self.bug != "abs":
            self.sent[rid] = ctx.monotonic()
        ctx.send("s", {"op": "acq", "rid": rid})

    def on_timer(self, ctx: Ctx, name: str) -> None:
        if name == "try":
            self._request(ctx)
            ctx.set_timer("try", RETRY_MS)
        elif name == "renew":
            if self.renewals_left > 0:
                self.renewals_left -= 1
                self._request(ctx)
                ctx.set_timer("renew", RENEW_MS)
            else:
                self.holding = False
                ctx.set_timer("try", IDLE_MS)

    def on_message(self, ctx: Ctx, src: str, p) -> None:
        if p["op"] != "grant":
            return
        if self.bug == "abs":
            until = p["exp"]                          # the server's wall-clock timestamp, trusted as-is
        else:
            sent = self.sent.get(p["rid"])
            if sent is None:                          # a request from before our last restart
                return
            ttl = p["ttl"]
            if self.bug != "nomargin":                # distrust our own clock by the promised rate bound
                ttl = ttl * (1_000_000 - self.max_ppm) // (1_000_000 + self.max_ppm)
            until = sent + ttl
        self.valid_until = until if self.valid_until is None else max(self.valid_until, until)
        if not self.holding:
            self.holding, self.renewals_left = True, RENEWALS
            ctx.cancel_timer("try")
            ctx.set_timer("renew", RENEW_MS)


@register
class LeaseScenario(Scenario):
    name = "lease"
    nodes = ["s", "c1", "c2"]
    duration = 1200
    fault_horizon = 400
    max_link_n = 10
    fault_kinds = ("drop", "dup", "delay", "crash", "partition", "skew")
    bugs: tuple = (None, "abs")
    max_drift_ppm = 0     # the rate error clients are told to tolerate; 0 = clocks run at true speed

    def __init__(self, bug: str | None = "abs"):
        if bug not in self.bugs:
            raise ValueError(f"bug must be one of {self.bugs}, got {bug!r}")
        self.bug = bug

    def params(self) -> dict:
        return {"bug": self.bug}

    def make_agent(self, node: str) -> Agent:
        return Server(node, self.bug) if node == "s" else Client(node, self.bug, self.max_drift_ppm)

    def check(self, sim) -> str | None:
        holders = [n for n in ("c1", "c2") if n in sim.up and sim.agents[n].holds(sim)]
        if len(holders) > 1:
            skew = {n: sim.clock_offset[n] for n in self.nodes if sim.clock_offset.get(n)}
            drift = {n: sim.rate_ppm(n) for n in self.nodes if sim.rate_ppm(n)}
            return (f"{' and '.join(holders)} both believe they hold the lease "
                    f"(skew ms: {skew or 'none'}; drift ppm: {drift or 'none'})")
        return None


@register
class LeaseDriftScenario(LeaseScenario):
    """The lease service on machines whose clocks also run at different speeds (and may step)."""

    name = "lease_drift"
    fault_kinds = ("drop", "dup", "delay", "crash", "partition", "skew", "drift")
    bugs = (None, "nomargin")
    max_drift_ppm = max(abs(p) for p in Scenario.drift_ppm)

    def __init__(self, bug: str | None = "nomargin"):
        super().__init__(bug)