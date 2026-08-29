"""Shared policy target and search value contracts."""

from __future__ import annotations

from sapai.sim.models import RunState

POLICY_TARGET_SCHEMA = "sapai-policy-targets-v2"
VALUE_OBJECTIVE = "arena-completion-probability"


def arena_run_value(state: RunState) -> float:
    """Return the terminal probability target for completing an Arena run."""

    if not state.terminal:
        raise ValueError("Arena completion value is defined only for terminal states")
    return 1.0 if state.trophies >= 10 else 0.0


def normalized_trophies(state: RunState) -> float:
    """Return final trophies as a bounded auxiliary target."""

    if not state.terminal:
        raise ValueError("final trophy target is defined only for terminal states")
    return min(1.0, max(0.0, state.trophies / 10.0))
