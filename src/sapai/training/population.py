"""Empirical and synthetic Arena opponent populations."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from sapai.data.replay import BoardSnapshot, board_is_pack_compatible
from sapai.data.serialization import read_boards
from sapai.ml.encoding import encode_teams
from sapai.rewards import arena_run_value
from sapai.sim.battle import BattleResultKind, BattleSimulator
from sapai.sim.catalog import Catalog
from sapai.sim.models import BattleOutcome, RunState, Team
from sapai.sim.shop import ShopEnvironment


def load_opponent_boards(
    path: str | Path,
    catalog: Catalog,
    pack: str,
) -> list[BoardSnapshot]:
    """Load the empirical Arena pool with the same pack filter for every caller."""

    boards = [
        board for board in read_boards(path) if board_is_pack_compatible(board, catalog, pack)
    ]
    if not boards:
        raise ValueError(f"board dataset contains no compatible {pack!r} boards")
    return boards


def load_opponent_population(
    path: str | Path,
    catalog: Catalog,
    pack: str,
    version: str | None = None,
) -> OpponentPopulation:
    boards = load_opponent_boards(path, catalog, pack)
    versions = Counter(board.version for board in boards)
    selected_version = version or min(versions, key=lambda item: (-versions[item], item))
    selected = [board for board in boards if board.version == selected_version]
    if not selected:
        raise ValueError(f"board dataset contains no boards for version {selected_version!r}")
    return OpponentPopulation(selected)


@dataclass(frozen=True, slots=True)
class OpponentPopulationSplit:
    train: OpponentPopulation
    validation: OpponentPopulation
    test: OpponentPopulation
    version: str
    board_counts: dict[str, int]


def split_opponent_populations(
    boards: list[BoardSnapshot],
    *,
    version: str | None = None,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> OpponentPopulationSplit:
    """Choose one patch and split whole replay IDs before policy training."""

    if not boards:
        raise ValueError("opponent population cannot be empty")
    versions = Counter(board.version for board in boards)
    selected_version = version or min(versions, key=lambda item: (-versions[item], item))
    version_boards = [board for board in boards if board.version == selected_version]
    if not version_boards:
        raise ValueError(f"board dataset contains no boards for version {selected_version!r}")
    from sapai.data.datasets import split_boards

    split = split_boards(
        version_boards,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    values = {name: getattr(split, name) for name in ("train", "validation", "test")}
    empty = [name for name, items in values.items() if not items]
    if empty:
        raise ValueError(
            "opponent replay split is empty for "
            f"{', '.join(empty)}; add replay groups or reduce held-out fractions"
        )
    return OpponentPopulationSplit(
        train=OpponentPopulation(values["train"]),
        validation=OpponentPopulation(values["validation"]),
        test=OpponentPopulation(values["test"]),
        version=selected_version,
        board_counts={name: len(items) for name, items in values.items()},
    )


class OpponentPopulation:
    """Boards indexed by pack, turn, and patch version with safe fallbacks."""

    def __init__(self, boards: list[BoardSnapshot]):
        if not boards:
            raise ValueError("opponent population cannot be empty")
        self.boards = boards
        self.versions = frozenset(board.version for board in boards)
        self.groups: dict[tuple[str, int, str], list[BoardSnapshot]] = defaultdict(list)
        for board in boards:
            self.groups[(board.pack, board.turn, board.version)].append(board)

    def candidates(self, *, pack: str, turn: int, version: str) -> list[BoardSnapshot]:
        if version == "current" and len(self.versions) == 1:
            version = next(iter(self.versions))
        exact = self.groups.get((pack, turn, version))
        if exact:
            return exact
        same_version = [
            board
            for board in self.boards
            if board.pack == pack and board.version == version
        ]
        if not same_version:
            raise ValueError(
                f"opponent population has no {pack!r} boards for version {version!r}; "
                f"available versions: {sorted(self.versions)}"
            )
        values = same_version
        distance = min(abs(board.turn - turn) for board in values)
        return [board for board in values if abs(board.turn - turn) == distance]

    def sample(
        self, *, pack: str, turn: int, version: str, rng: random.Random
    ) -> BoardSnapshot:
        return rng.choice(self.candidates(pack=pack, turn=turn, version=version))

    def save_encoded_cache(self, path: str | Path) -> None:
        """Persist fixed opponent tensors; model-dependent embeddings stay out of the dataset."""

        try:
            import numpy as np
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("install the 'ml' extra to cache population tensors") from error
        encoded = encode_teams(
            [board.team for board in self.boards],
            turns=[board.turn for board in self.boards],
            packs=[board.pack for board in self.boards],
        )
        metadata = [
            {
                "replay_id": board.replay_id,
                "side": board.side,
                "pack": board.pack,
                "turn": board.turn,
                "version": board.version,
            }
            for board in self.boards
        ]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **encoded,
            metadata=np.asarray([json.dumps(row) for row in metadata]),
        )

    @classmethod
    def synthetic(
        cls,
        catalog: Catalog,
        *,
        pack: str = "Turtle",
        max_turn: int = 20,
        boards_per_turn: int = 16,
        seed: int = 0,
    ) -> OpponentPopulation:
        """Create a deterministic smoke/demo pool; do not use it for serious training."""

        rng = random.Random(seed)
        boards: list[BoardSnapshot] = []
        for turn in range(1, max_turn + 1):
            tier = min(6, (turn + 1) // 2)
            pool = catalog.pack_pets(pack, through_tier=tier)
            if not pool:
                continue
            for index in range(boards_per_turn):
                team_size = min(5, max(1, 2 + turn // 3))
                pets = [rng.choice(pool).create() for _ in range(team_size)]
                budget = max(0, turn - 1) * 2
                for _ in range(budget):
                    pet = rng.choice(pets)
                    pet.buff(rng.randrange(2), rng.randrange(2))
                boards.append(
                    BoardSnapshot(
                        replay_id=f"synthetic-{turn}-{index}",
                        side="opponent",
                        turn=turn,
                        pack=pack,
                        team=Team.from_pets(pets),
                        version="synthetic",
                    )
                )
        return cls(boards)


class SimulatorPopulationEvaluator:
    """Evaluate an end-turn state with exact battles against sampled opponents.

    When a continuation evaluator is supplied, each simulated outcome is
    applied to a cloned run state and the resulting next-turn state is valued
    in one batch. This keeps the battle leaf on the same final-run completion-
    probability scale as the policy/value network. Without one, non-terminal
    continuations use an uninformative prior of 0.5; terminal continuations are
    still labeled exactly as 0 or 1.
    """

    _OUTCOMES: ClassVar[dict[BattleResultKind, BattleOutcome]] = {
        BattleResultKind.PLAYER_WIN: BattleOutcome.WIN,
        BattleResultKind.DRAW: BattleOutcome.DRAW,
        BattleResultKind.OPPONENT_WIN: BattleOutcome.LOSS,
    }
    def __init__(
        self,
        environment: ShopEnvironment,
        simulator: BattleSimulator,
        population: OpponentPopulation,
        *,
        simulations: int = 8,
        continuation_evaluator: Any | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("battle evaluation simulations must be positive")
        self.environment = environment
        self.simulator = simulator
        self.population = population
        self.simulations = simulations
        self.continuation_evaluator = continuation_evaluator
        self._panels: dict[
            tuple[str, int, str], list[tuple[BoardSnapshot, int, int]]
        ] = {}
        self._values: dict[str, list[float]] = {}

    def begin_search(self, state: RunState, rng: random.Random) -> None:
        """Build one opponent/seed panel shared by all end-turn candidates."""

        self._panels = {}
        self._values = {}
        self._panel(state, rng)

    def _panel(
        self,
        state: RunState,
        rng: random.Random,
    ) -> list[tuple[BoardSnapshot, int, int]]:
        key = (state.pack, state.turn, state.version)
        existing = self._panels.get(key)
        if existing is not None:
            return existing
        candidates = self.population.candidates(
            pack=state.pack,
            turn=state.turn,
            version=state.version,
        )
        if len(candidates) >= self.simulations:
            opponents = rng.sample(candidates, self.simulations)
        else:
            opponents = list(candidates)
            rng.shuffle(opponents)
            opponents.extend(
                rng.choice(candidates) for _ in range(self.simulations - len(opponents))
            )
        panel = [
            (opponent, rng.getrandbits(63), rng.getrandbits(64))
            for opponent in opponents
        ]
        self._panels[key] = panel
        return panel

    def evaluate_battle(
        self,
        state: RunState,
        rng: random.Random,
        *,
        simulations: int | None = None,
    ) -> float:
        if not state.awaiting_battle:
            raise ValueError("battle evaluation requires an awaiting-battle state")
        requested = min(self.simulations, simulations or self.simulations)
        panel = self._panel(state, rng)
        key = state.canonical_key()
        values = self._values.setdefault(key, [])
        new_entries = panel[len(values) : requested]
        outcomes: list[BattleResultKind] = []
        for opponent, battle_seed, _ in new_entries:
            outcomes.append(
                self.simulator.simulate(
                    state.team,
                    opponent.team,
                    seed=battle_seed,
                    record_trace=False,
                ).outcome
            )
        new_values: list[float | None] = [None] * len(outcomes)
        continuation_indices: list[int] = []
        continuation_states: list[RunState] = []
        continuation_actions = []
        for index, outcome in enumerate(outcomes):
            transition_rng = random.Random(new_entries[index][2])
            next_state = self.environment.apply_outcome(
                state,
                self._OUTCOMES[outcome],
                transition_rng,
            ).state
            if next_state.terminal:
                new_values[index] = arena_run_value(next_state)
            elif self.continuation_evaluator is None:
                new_values[index] = 0.5
            else:
                continuation_indices.append(index)
                continuation_states.append(next_state)
                continuation_actions.append(self.environment.legal_actions(next_state))

        if continuation_states:
            evaluate_many = getattr(self.continuation_evaluator, "evaluate_many", None)
            if callable(evaluate_many):
                evaluated = evaluate_many(continuation_states, continuation_actions)
            else:
                evaluated = [
                    self.continuation_evaluator.evaluate(next_state, actions)
                    for next_state, actions in zip(
                        continuation_states,
                        continuation_actions,
                        strict=True,
                    )
                ]
            if len(evaluated) != len(continuation_states):
                raise ValueError("continuation evaluator returned the wrong number of values")
            for index, (_, value) in zip(continuation_indices, evaluated, strict=True):
                new_values[index] = float(value)

        if any(value is None for value in new_values):  # pragma: no cover - defensive invariant
            raise RuntimeError("battle evaluation did not value every simulated outcome")
        values.extend(float(value) for value in new_values)
        return sum(values[:requested]) / requested


class BattlePopulationEvaluator:
    """Average a battle model over empirical opponents for one Arena state."""

    def __init__(self, model, population: OpponentPopulation, *, opponents: int = 32):
        self.model = model
        self.population = population
        self.opponents = opponents

    def evaluate(self, team: Team, *, pack: str, turn: int, version: str, seed: int = 0):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("install the 'ml' extra to evaluate populations") from error
        rng = random.Random(seed)
        values = self.population.candidates(pack=pack, turn=turn, version=version)
        chosen = [rng.choice(values) for _ in range(self.opponents)]
        player = encode_teams([team] * len(chosen), turns=[turn] * len(chosen), packs=[pack] * len(chosen))
        opponent = encode_teams(
            [board.team for board in chosen],
            turns=[board.turn for board in chosen],
            packs=[board.pack for board in chosen],
        )
        inputs = {
            **{f"player_{key}": tf.convert_to_tensor(value) for key, value in player.items()},
            **{f"opponent_{key}": tf.convert_to_tensor(value) for key, value in opponent.items()},
        }
        probabilities = self.model(inputs, training=False)["probabilities"]
        return tf.reduce_mean(probabilities, axis=0).numpy()
