from __future__ import annotations

from sapai.sim.models import RunState


def arena_run_value(state: RunState) -> float:
    """Return the terminal Arena reward used to train the policy value head."""

    if state.trophies >= 10:
        return 1.0
    return (state.trophies**2) / 200.0
