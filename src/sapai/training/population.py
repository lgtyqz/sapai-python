"""Empirical and synthetic Arena opponent populations."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from sapai.data.replay import BoardSnapshot
from sapai.ml.encoding import encode_teams
from sapai.sim.catalog import Catalog
from sapai.sim.models import Team


class OpponentPopulation:
    """Boards indexed by pack, turn, and patch version with safe fallbacks."""

    def __init__(self, boards: list[BoardSnapshot]):
        if not boards:
            raise ValueError("opponent population cannot be empty")
        self.boards = boards
        self.groups: dict[tuple[str, int, str], list[BoardSnapshot]] = defaultdict(list)
        for board in boards:
            self.groups[(board.pack, board.turn, board.version)].append(board)

    def candidates(self, *, pack: str, turn: int, version: str) -> list[BoardSnapshot]:
        exact = self.groups.get((pack, turn, version))
        if exact:
            return exact
        same_patch = [
            board
            for board in self.boards
            if board.pack == pack and board.turn == turn and board.version == version
        ]
        if same_patch:
            return same_patch
        same_turn = [
            board for board in self.boards if board.pack == pack and board.turn == turn
        ]
        if same_turn:
            return same_turn
        same_pack = [board for board in self.boards if board.pack == pack]
        values = same_pack or self.boards
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
