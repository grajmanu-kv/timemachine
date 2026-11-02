"""timemachine: deterministic simulation with record/replay, fault injection, divergence bisection, and fork/rewind."""
from . import demo_agents, election, lease  # noqa: F401  (register the built-in scenarios)
from .sim import Agent, Ctx, Scenario, Simulation, build_scenario, register
from .tape import Divergence, Tape, first_divergence
from .tools import random_plan, replay, rewind, run, search, shrink

__all__ = [
    "Agent", "Ctx", "Scenario", "Simulation", "Tape", "Divergence",
    "build_scenario", "register", "first_divergence",
    "run", "replay", "rewind", "search", "shrink", "random_plan",
]