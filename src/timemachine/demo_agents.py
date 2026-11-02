"""Demo scenario: a payment client retrying against a server with a subtle idempotency bug.

The client sends `pay(id, amount)` and retries if no ack arrives within 40 ms.
The server de-duplicates by request id, but keeps the set of seen ids in
*volatile* memory while the debit ledger is durable. After a crash/restart a
retried request is applied a second time: a double debit.

Reproducing it takes two things at once (an ack that is lost or delayed AND a
server crash/restart before the retry lands), which is exactly the kind of
interleaving random fault injection finds and a human rarely writes a test for.
"""
from __future__ import annotations

import random

from .sim import Agent, Ctx, Scenario, register

RETRY_MS = 40
MAX_RETRIES = 6


def price_oracle(order_id: int) -> int:
    """Stand-in for a non-deterministic outside call (an LLM, a pricing API, the wall clock)."""
    return 10 + random.SystemRandom().randrange(5)


class Client(Agent):
    orders = (1, 2, 3)

    def on_start(self, ctx: Ctx) -> None:
        self.idx, self.tries, self.amount = 0, 0, 0
        self._first_attempt(ctx)

    def _first_attempt(self, ctx: Ctx) -> None:
        oid = self.orders[self.idx]
        self.tries = 0
        # Looked up via the module global so tests can prove replay never calls the real thing.
        self.amount = ctx.external("pricing", _oracle(), oid)
        self._send(ctx)

    def _send(self, ctx: Ctx) -> None:
        ctx.send("s", {"op": "pay", "id": self.orders[self.idx], "amt": self.amount})
        ctx.set_timer("retry", RETRY_MS)

    def on_message(self, ctx: Ctx, src: str, p) -> None:
        if p.get("op") == "ack" and self.idx < len(self.orders) and p["id"] == self.orders[self.idx]:
            ctx.cancel_timer("retry")
            self.idx += 1
            if self.idx < len(self.orders):
                self._first_attempt(ctx)

    def on_timer(self, ctx: Ctx, name: str) -> None:
        if name == "retry" and self.tries < MAX_RETRIES:
            self.tries += 1
            self._send(ctx)


def _oracle():
    return price_oracle  # resolved at call time, so it can be patched


class Server(Agent):
    def __init__(self, name: str, fixed: bool = False):
        super().__init__(name)
        self.fixed = fixed

    def on_start(self, ctx: Ctx) -> None:
        self.durable.setdefault("debits", [])
        self.durable.setdefault("seen", [])
        self.seen_volatile: set[int] = set()

    def _already_applied(self, oid: int) -> bool:
        # BUG: the buggy server consults only volatile memory, which a crash wipes.
        return oid in (self.durable["seen"] if self.fixed else self.seen_volatile)

    def on_message(self, ctx: Ctx, src: str, p) -> None:
        if p["op"] == "pay":
            if not self._already_applied(p["id"]):
                self.durable["debits"].append(p["id"])
                self.durable["seen"].append(p["id"])
                self.seen_volatile.add(p["id"])
            ctx.send(src, {"op": "ack", "id": p["id"]})


@register
class TransferScenario(Scenario):
    name = "transfer"
    nodes = ["c", "s"]

    def __init__(self, fixed: bool = False):
        self.fixed = fixed

    def params(self) -> dict:
        return {"fixed": self.fixed}

    def make_agent(self, node: str) -> Agent:
        return Client(node) if node == "c" else Server(node, fixed=self.fixed)

    def check(self, sim) -> str | None:
        debits = sim.agents["s"].durable.get("debits", [])
        if len(debits) != len(set(debits)):
            return f"double debit: ledger={debits}"
        return None