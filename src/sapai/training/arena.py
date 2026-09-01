"""Complete Arena rollouts for policy training and visualization."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

from sapai.data.serialization import (
    action_from_dict,
    action_to_dict,
    read_jsonl,
    run_state_from_dict,
    run_state_to_dict,
    write_jsonl,
)
from sapai.rewards import (
    POLICY_TARGET_SCHEMA,
    arena_run_value,
    normalized_trophies,
)
from sapai.search.stochastic import PolicyGuidedSearch
from sapai.sim.actions import Action, ActionKind
from sapai.sim.battle import BattleResult, BattleResultKind, BattleSimulator
from sapai.sim.models import BattleOutcome, RunState
from sapai.sim.shop import ShopEnvironment
from sapai.training.population import OpponentPopulation


@dataclass(frozen=True, slots=True)
class PolicyChoice:
    action: Action
    probabilities: list[float]


class ArenaPolicy(Protocol):
    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice: ...


class HeuristicPolicy:
    """A deterministic bootstrap policy that buys useful offers and then stops."""

    PRIORITY: ClassVar[dict[ActionKind, int]] = {
        ActionKind.BUY_MERGE_PET: 5,
        ActionKind.MERGE_BOARD_PET: 4,
        ActionKind.BUY_PET: 3,
        ActionKind.BUY_FOOD: 2,
        ActionKind.ROLL: 1,
        ActionKind.END_TURN: 0,
    }

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        useful = [action for action in actions if action.kind in self.PRIORITY]
        if not useful:
            action = next(action for action in actions if action.kind is ActionKind.END_TURN)
        else:
            best_priority = max(self.PRIORITY[action.kind] for action in useful)
            best = [action for action in useful if self.PRIORITY[action.kind] == best_priority]
            action = rng.choice(best)
        probabilities = [float(candidate == action) for candidate in actions]
        return PolicyChoice(action, probabilities)


class RandomPolicy:
    """A loop-safe, action-kind-balanced exploratory policy."""

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        end = next(action for action in actions if action.kind is ActionKind.END_TURN)
        roll_available = any(action.kind is ActionKind.ROLL for action in actions)
        if state.gold < 1 or not roll_available:
            action = end
        else:
            by_kind: dict[ActionKind, list[Action]] = {}
            for candidate in actions:
                if candidate.kind is ActionKind.END_TURN:
                    continue
                by_kind.setdefault(candidate.kind, []).append(candidate)
            chosen_kind = rng.choice(list(by_kind))
            action = rng.choice(by_kind[chosen_kind])
        probabilities = [float(candidate == action) for candidate in actions]
        return PolicyChoice(action, probabilities)


class MixturePolicy:
    """Use exploration on a fixed fraction of otherwise heuristic decisions."""

    def __init__(
        self,
        primary: ArenaPolicy,
        exploratory: ArenaPolicy,
        *,
        exploration_probability: float = 0.25,
    ) -> None:
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration probability must be in [0, 1]")
        self.primary = primary
        self.exploratory = exploratory
        self.exploration_probability = exploration_probability

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        policy = (
            self.exploratory
            if rng.random() < self.exploration_probability
            else self.primary
        )
        return policy.choose(state, actions, rng)


class ModelPolicy:
    def __init__(self, evaluator, *, sample: bool = False):
        self.evaluator = evaluator
        self.sample = sample

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        probabilities, _ = self.evaluator.evaluate(state, actions)
        probabilities = list(probabilities)
        if state.gold > 0 and any(action.kind is ActionKind.ROLL for action in actions):
            for index, action in enumerate(actions):
                if action.kind is ActionKind.END_TURN:
                    probabilities[index] = 0.0
            total = sum(probabilities)
            if total <= 0:
                allowed = [
                    index
                    for index, action in enumerate(actions)
                    if action.kind is not ActionKind.END_TURN
                ]
                probabilities = [0.0] * len(actions)
                for index in allowed:
                    probabilities[index] = 1.0 / len(allowed)
            else:
                probabilities = [value / total for value in probabilities]
        if self.sample:
            action = rng.choices(actions, weights=probabilities, k=1)[0]
        else:
            action = actions[max(range(len(actions)), key=probabilities.__getitem__)]
        return PolicyChoice(action, probabilities)


class SearchPolicy:
    def __init__(self, search: PolicyGuidedSearch):
        self.search = search

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        result = self.search.search(state, seed=rng.getrandbits(63))
        counts = [result.visit_counts.get(action, 0) for action in actions]
        total = sum(counts)
        probabilities = (
            [count / total for count in counts]
            if total
            else [float(action == result.action) for action in actions]
        )
        return PolicyChoice(result.action, probabilities)


@dataclass(slots=True)
class ArenaDecision:
    state: RunState
    actions: list[Action]
    search_policy: list[float]
    next_battle_after_policy: tuple[float, float, float] = (0.0, 1.0, 0.0)
    run_value: float = 0.0
    expected_trophies: float = 0.0


@dataclass(frozen=True, slots=True)
class ShopFrame:
    label: str
    state: RunState
    action: Action | None = None


@dataclass(slots=True)
class ArenaTurn:
    turn: int
    shop_frames: list[ShopFrame] = field(default_factory=list)
    opponent_replay_id: str | None = None
    battle: BattleResult | None = None


@dataclass(frozen=True, slots=True)
class ArenaRunResult:
    final_state: RunState
    decisions: list[ArenaDecision]
    turns: list[ArenaTurn]
    battle_outcomes: tuple[BattleResultKind, ...] = ()


class ArenaRunner:
    def __init__(
        self,
        environment: ShopEnvironment,
        battle: BattleSimulator,
        population: OpponentPopulation,
        policy: ArenaPolicy,
        *,
        max_decisions_per_turn: int = 30,
        record_timeline: bool = True,
    ):
        self.environment = environment
        self.battle = battle
        self.population = population
        self.policy = policy
        self.max_decisions_per_turn = max_decisions_per_turn
        self.record_timeline = record_timeline

    def run(
        self,
        *,
        pack: str = "Turtle",
        version: str = "current",
        seed: int = 0,
    ) -> ArenaRunResult:
        rng = random.Random(seed)
        if version == "current" and len(self.population.versions) == 1:
            version = next(iter(self.population.versions))
        state = self.environment.reset(
            pack=pack,
            version=version,
            seed=rng.getrandbits(63),
        )
        decisions: list[ArenaDecision] = []
        turns: list[ArenaTurn] = []
        battle_outcomes: list[BattleResultKind] = []
        while not state.terminal:
            arena_turn = (
                ArenaTurn(
                    state.turn,
                    [ShopFrame(f"Turn {state.turn} shop", state.clone())],
                )
                if self.record_timeline
                else None
            )
            turn_decisions: list[ArenaDecision] = []
            for decision_index in range(self.max_decisions_per_turn):
                actions = self.environment.legal_actions(state)
                if not actions:
                    raise RuntimeError("non-terminal shop state has no legal actions")
                if decision_index + 1 == self.max_decisions_per_turn:
                    action = next(
                        candidate for candidate in actions if candidate.kind is ActionKind.END_TURN
                    )
                    choice = PolicyChoice(action, [float(item == action) for item in actions])
                else:
                    choice = self.policy.choose(state, actions, rng)
                decision = ArenaDecision(
                    state.clone(), list(actions), list(choice.probabilities)
                )
                decisions.append(decision)
                turn_decisions.append(decision)
                state = self.environment.step(state, choice.action, rng).state
                if arena_turn is not None:
                    arena_turn.shop_frames.append(
                        ShopFrame(
                            choice.action.kind.name.replace("_", " ").title(),
                            state.clone(),
                            choice.action,
                        )
                    )
                if state.awaiting_battle:
                    break
            if not state.awaiting_battle:
                raise RuntimeError("Arena turn failed to reach a battle")

            opponent = self.population.sample(
                pack=state.pack,
                turn=state.turn,
                version=state.version,
                rng=rng,
            )
            battle_result = self.battle.simulate(
                state.team,
                opponent.team,
                seed=rng.getrandbits(63),
                record_trace=self.record_timeline,
            )
            battle_outcomes.append(battle_result.outcome)
            if arena_turn is not None:
                arena_turn.opponent_replay_id = opponent.replay_id
                arena_turn.battle = battle_result
                turns.append(arena_turn)
            outcome, target = _outcome_target(battle_result.outcome)
            for decision in turn_decisions:
                decision.next_battle_after_policy = target
            state = self.environment.apply_outcome(state, outcome, rng).state


        run_value = arena_run_value(state)
        trophy_target = normalized_trophies(state)
        for decision in decisions:
            decision.run_value = run_value
            decision.expected_trophies = trophy_target
        return ArenaRunResult(state.clone(), decisions, turns, tuple(battle_outcomes))


def _outcome_target(
    outcome: BattleResultKind,
) -> tuple[BattleOutcome, tuple[float, float, float]]:
    if outcome is BattleResultKind.PLAYER_WIN:
        return BattleOutcome.WIN, (1.0, 0.0, 0.0)
    if outcome is BattleResultKind.OPPONENT_WIN:
        return BattleOutcome.LOSS, (0.0, 0.0, 1.0)
    return BattleOutcome.DRAW, (0.0, 1.0, 0.0)


def decision_to_dict(decision: ArenaDecision) -> dict[str, object]:
    return {
        "target_schema": POLICY_TARGET_SCHEMA,
        "state": run_state_to_dict(decision.state),
        "actions": [action_to_dict(action) for action in decision.actions],
        "search_policy": decision.search_policy,
        "next_battle_after_policy": list(decision.next_battle_after_policy),
        "run_value": decision.run_value,
        "expected_trophies": decision.expected_trophies,
    }


def decision_from_dict(value: dict[str, object]) -> ArenaDecision:
    schema = value.get("target_schema")
    if schema != POLICY_TARGET_SCHEMA:
        raise ValueError(
            f"policy target schema mismatch: expected {POLICY_TARGET_SCHEMA!r}, got {schema!r}; "
            "regenerate Arena trajectories for the completion-probability objective"
        )
    return ArenaDecision(
        state=run_state_from_dict(value["state"]),  # type: ignore[arg-type]
        actions=[action_from_dict(action) for action in value["actions"]],  # type: ignore[arg-type]
        search_policy=[float(item) for item in value["search_policy"]],  # type: ignore[union-attr]
        next_battle_after_policy=tuple(  # type: ignore[arg-type]
            float(item) for item in value["next_battle_after_policy"]
        ),
        run_value=float(value["run_value"]),
        expected_trophies=float(value["expected_trophies"]),
    )


def write_arena_decisions(path: str | Path, decisions: list[ArenaDecision]) -> int:
    return write_jsonl(path, (decision_to_dict(decision) for decision in decisions))


def read_arena_decisions(path: str | Path) -> list[ArenaDecision]:
    return [decision_from_dict(row) for row in read_jsonl(path)]


def evaluate_arena_policy(
    runner: ArenaRunner,
    *,
    episodes: int,
    pack: str,
    version: str,
    seed: int,
) -> dict[str, object]:
    """Evaluate a policy on a fixed episode seed schedule."""

    if episodes < 1:
        raise ValueError("Arena evaluation episodes must be positive")
    trophies: list[int] = []
    turns: list[int] = []
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    for episode in range(episodes):
        result = runner.run(pack=pack, version=version, seed=seed + episode)
        trophies.append(result.final_state.trophies)
        turns.append(result.final_state.turn)
        for outcome in result.battle_outcomes:
            if outcome is BattleResultKind.PLAYER_WIN:
                outcomes["win"] += 1
            elif outcome is BattleResultKind.OPPONENT_WIN:
                outcomes["loss"] += 1
            else:
                outcomes["draw"] += 1
    battles = sum(outcomes.values())
    return {
        "episodes": episodes,
        "completion_rate": sum(value >= 10 for value in trophies) / episodes,
        "mean_trophies": statistics.fmean(trophies),
        "median_trophies": statistics.median(trophies),
        "mean_turns": statistics.fmean(turns),
        "battle_rates": {
            name: count / battles if battles else 0.0 for name, count in outcomes.items()
        },
        "trophy_histogram": {
            str(value): trophies.count(value) for value in sorted(set(trophies))
        },
    }
