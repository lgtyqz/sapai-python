"""Complete Arena rollouts for policy training and visualization."""

from __future__ import annotations

import random
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
        if state.gold < 3:
            useful = [action for action in useful if action.kind is ActionKind.END_TURN]
        if not useful:
            action = next(action for action in actions if action.kind is ActionKind.END_TURN)
        else:
            best_priority = max(self.PRIORITY[action.kind] for action in useful)
            best = [action for action in useful if self.PRIORITY[action.kind] == best_priority]
            action = rng.choice(best)
        probabilities = [float(candidate == action) for candidate in actions]
        return PolicyChoice(action, probabilities)


class RandomPolicy:
    """A loop-safe exploratory policy for bootstrap data."""

    USEFUL: ClassVar[set[ActionKind]] = {
        ActionKind.BUY_PET,
        ActionKind.BUY_MERGE_PET,
        ActionKind.MERGE_BOARD_PET,
        ActionKind.BUY_FOOD,
        ActionKind.ROLL,
        ActionKind.SELL_PET,
        ActionKind.END_TURN,
    }

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        choices = [action for action in actions if action.kind in self.USEFUL]
        end = next(action for action in actions if action.kind is ActionKind.END_TURN)
        if state.gold < 1 or rng.random() < 0.15:
            action = end
        else:
            action = rng.choice(choices or [end])
        probabilities = [float(candidate == action) for candidate in actions]
        return PolicyChoice(action, probabilities)


class ModelPolicy:
    def __init__(self, evaluator, *, sample: bool = False):
        self.evaluator = evaluator
        self.sample = sample

    def choose(
        self, state: RunState, actions: list[Action], rng: random.Random
    ) -> PolicyChoice:
        probabilities, _ = self.evaluator.evaluate(state, actions)
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
    next_battle: tuple[float, float, float] = (0.0, 1.0, 0.0)
    run_value: float = 0.0
    expected_wins: float = 0.0


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


class ArenaRunner:
    def __init__(
        self,
        environment: ShopEnvironment,
        battle: BattleSimulator,
        population: OpponentPopulation,
        policy: ArenaPolicy,
        *,
        max_decisions_per_turn: int = 30,
    ):
        self.environment = environment
        self.battle = battle
        self.population = population
        self.policy = policy
        self.max_decisions_per_turn = max_decisions_per_turn

    def run(self, *, pack: str = "Turtle", seed: int = 0) -> ArenaRunResult:
        rng = random.Random(seed)
        state = self.environment.reset(pack=pack, seed=rng.getrandbits(63))
        decisions: list[ArenaDecision] = []
        turns: list[ArenaTurn] = []
        while not state.terminal:
            arena_turn = ArenaTurn(
                state.turn,
                [ShopFrame(f"Turn {state.turn} shop", state.clone())],
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
                arena_turn.shop_frames.append(
                    ShopFrame(choice.action.kind.name.replace("_", " ").title(), state.clone(), choice.action)
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
            )
            arena_turn.opponent_replay_id = opponent.replay_id
            arena_turn.battle = battle_result
            turns.append(arena_turn)
            outcome, target = _outcome_target(battle_result.outcome)
            for decision in turn_decisions:
                decision.next_battle = target
            state = self.environment.apply_outcome(state, outcome, rng).state

        run_value = 1.0 if state.trophies >= 10 else -1.0
        for decision in decisions:
            decision.run_value = run_value
            decision.expected_wins = float(state.trophies)
        return ArenaRunResult(state.clone(), decisions, turns)


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
        "state": run_state_to_dict(decision.state),
        "actions": [action_to_dict(action) for action in decision.actions],
        "search_policy": decision.search_policy,
        "next_battle": list(decision.next_battle),
        "run_value": decision.run_value,
        "expected_wins": decision.expected_wins,
    }


def decision_from_dict(value: dict[str, object]) -> ArenaDecision:
    return ArenaDecision(
        state=run_state_from_dict(value["state"]),  # type: ignore[arg-type]
        actions=[action_from_dict(action) for action in value["actions"]],  # type: ignore[arg-type]
        search_policy=[float(item) for item in value["search_policy"]],  # type: ignore[union-attr]
        next_battle=tuple(float(item) for item in value["next_battle"]),  # type: ignore[arg-type]
        run_value=float(value["run_value"]),
        expected_wins=float(value["expected_wins"]),
    )


def write_arena_decisions(path: str | Path, decisions: list[ArenaDecision]) -> int:
    return write_jsonl(path, (decision_to_dict(decision) for decision in decisions))


def read_arena_decisions(path: str | Path) -> list[ArenaDecision]:
    return [decision_from_dict(row) for row in read_jsonl(path)]
