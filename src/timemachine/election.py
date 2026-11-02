"""Scenario: leader election among three nodes (Raft-style terms and votes, no log).

Safety  : at most one node wins any given term.
Liveness: once faults have stopped and been repaired, the cluster settles on a leader
          and stops holding elections.

Two seeded bugs, selected with the `bug` parameter:

  "vote"  : `voted_for` lives in volatile memory. A voter that crashes and restarts inside the
            same term has forgotten its vote and can grant a second one, so two candidates
            both reach a majority in one term. (Raft requires votedFor to be persisted.)
  "churn" : followers never re-arm their election timer on a heartbeat, so they keep calling
            elections under a healthy leader. Safety holds; liveness does not.
  None    : correct implementation.
"""
from __future__ import annotations

from .sim import Agent, Ctx, Scenario, register

# Election timeouts must be well above network jitter (5-14 ms here). An earlier 50-62 ms range
# produced a genuine split-vote livelock in the *correct* variant (found by this very check).
ELECTION_MIN, ELECTION_MAX = 50, 150
HEARTBEAT_MS = 20
SETTLE_MS = 200                       # no election may start in the last SETTLE_MS of the run


class Node(Agent):
    def __init__(self, name: str, peers: list[str], bug: str | None):
        super().__init__(name)
        self.peers, self.bug = peers, bug
        self.role = "follower"
        self.votes: set[str] = set()
        self._vote_mem: str | None = None   # only used by the buggy variant

    # -- persistent vs. volatile vote (the seeded bug lives here) -----------
    def _get_vote(self):
        return self._vote_mem if self.bug == "vote" else self.durable["voted_for"]

    def _set_vote(self, who):
        if self.bug == "vote":
            self._vote_mem = who
        else:
            self.durable["voted_for"] = who

    # -- lifecycle ----------------------------------------------------------
    def on_start(self, ctx: Ctx) -> None:
        d = self.durable
        d.setdefault("term", 0)
        d.setdefault("voted_for", None)
        d.setdefault("won", [])         # terms this node became leader in
        d.setdefault("elections", [])   # times this node started a campaign
        self.role, self.votes, self._vote_mem = "follower", set(), None
        self._arm(ctx)

    def _arm(self, ctx: Ctx) -> None:
        ctx.set_timer("election", ctx.randrange(ELECTION_MIN, ELECTION_MAX))

    def _majority(self) -> int:
        return (len(self.peers) + 1) // 2 + 1

    def _heartbeat(self, ctx: Ctx) -> None:
        for p in self.peers:
            ctx.send(p, {"op": "hb", "term": self.durable["term"]})
        ctx.set_timer("hb", HEARTBEAT_MS)

    # -- timers -------------------------------------------------------------
    def on_timer(self, ctx: Ctx, name: str) -> None:
        d = self.durable
        if name == "hb":
            if self.role == "leader":
                self._heartbeat(ctx)
        elif name == "election" and self.role != "leader":
            d["term"] += 1
            self._set_vote(self.name)
            self.role, self.votes = "candidate", {self.name}
            d["elections"].append(ctx.now)
            for p in self.peers:
                ctx.send(p, {"op": "rv", "term": d["term"]})
            self._arm(ctx)

    # -- messages -----------------------------------------------------------
    def on_message(self, ctx: Ctx, src: str, p) -> None:
        d = self.durable
        if p["term"] > d["term"]:
            was_leader = self.role == "leader"
            d["term"] = p["term"]
            self._set_vote(None)
            self.role, self.votes = "follower", set()
            if was_leader:
                ctx.cancel_timer("hb")
                self._arm(ctx)
        op = p["op"]
        if op == "rv":
            grant = p["term"] == d["term"] and self._get_vote() in (None, src)
            if grant:
                self._set_vote(src)
                self._arm(ctx)
            ctx.send(src, {"op": "vote", "term": d["term"], "granted": grant})
        elif op == "vote":
            if self.role == "candidate" and p["term"] == d["term"] and p["granted"]:
                self.votes.add(src)
                if len(self.votes) >= self._majority():
                    self.role = "leader"
                    d["won"].append(d["term"])
                    ctx.cancel_timer("election")
                    self._heartbeat(ctx)
        elif op == "hb" and p["term"] == d["term"]:
            self.role = "follower"
            if self.bug != "churn":
                self._arm(ctx)


@register
class ElectionScenario(Scenario):
    name = "election"
    nodes = ["n1", "n2", "n3"]
    duration = 1000
    fault_horizon = 250   # crashes/partitions start before 250ms and heal within ~150ms, leaving a long quiet tail
    max_link_n = 12

    def __init__(self, bug: str | None = "vote"):
        if bug not in (None, "vote", "churn"):
            raise ValueError(f"bug must be None, 'vote' or 'churn', got {bug!r}")
        self.bug = bug

    def params(self) -> dict:
        return {"bug": self.bug}

    def make_agent(self, node: str) -> Agent:
        return Node(node, [n for n in self.nodes if n != node], self.bug)

    def check(self, sim) -> str | None:
        winners: dict[int, list[str]] = {}
        for n in sorted(sim.agents):
            for term in sim.agents[n].durable.get("won", []):
                winners.setdefault(term, []).append(n)
        clash = {t: ns for t, ns in winners.items() if len(ns) > 1}
        return f"two leaders in one term: {clash}" if clash else None

    def check_final(self, sim) -> str | None:
        leaders = [n for n in sorted(sim.up) if sim.agents[n].role == "leader"]
        if not leaders:
            return "liveness: no leader at the end of the run"
        late = sorted((t, n) for n in sim.agents for t in sim.agents[n].durable["elections"]
                      if t > self.duration - SETTLE_MS)
        if late:
            t, n = late[0]
            return f"liveness: {n} started an election at t={t}ms although faults had ended"
        return None