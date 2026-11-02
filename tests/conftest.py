import pytest

from timemachine import demo_agents

import helpers


@pytest.fixture(autouse=True)
def tag_calls(monkeypatch):
    """Pin the toy scenario's outside call to a deterministic value and record every real call.

    Autouse so independent runs are comparable by default. Tests that want the genuinely
    non-deterministic call do `monkeypatch.setitem(helpers._hooks, "tag", helpers._live_tag)`.
    """
    calls: list[int] = []

    def fake(i: int) -> int:
        calls.append(i)
        return 1000 + i

    monkeypatch.setitem(helpers._hooks, "tag", fake)
    return calls


@pytest.fixture
def fixed_oracle(monkeypatch):
    """Make the transfer demo's pricing oracle deterministic so independent runs can be compared."""
    monkeypatch.setattr(demo_agents, "price_oracle", lambda order_id: 10 + order_id)