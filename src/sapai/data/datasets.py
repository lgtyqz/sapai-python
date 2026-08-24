"""Reproducible board and battle-example dataset utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sapai.data.replay import BoardSnapshot
from sapai.data.serialization import (
    read_jsonl,
    team_from_dict,
    team_to_dict,
    write_jsonl,
)
from sapai.ml.training import BattleExample, label_board_pairs
from sapai.sim.battle import BattleSimulator

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: list[BoardSnapshot]
    validation: list[BoardSnapshot]
    test: list[BoardSnapshot]


def _group_key(board: BoardSnapshot) -> str:
    if board.replay_id:
        return board.replay_id
    payload = json.dumps(team_to_dict(board.team), sort_keys=True, separators=(",", ":"))
    return f"anonymous:{board.turn}:{board.pack}:{board.version}:{payload}"


def split_boards(
    boards: Sequence[BoardSnapshot],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetSplit:
    """Split whole replay IDs deterministically to prevent board leakage."""

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")
    result: dict[str, list[BoardSnapshot]] = {name: [] for name in SPLITS}
    for board in boards:
        digest = hashlib.sha256(f"{seed}:{_group_key(board)}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < test_fraction:
            split = "test"
        elif value < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        result[split].append(board)
    return DatasetSplit(**result)


def battle_example_to_dict(example: BattleExample) -> dict[str, Any]:
    return {
        "player": team_to_dict(example.player),
        "opponent": team_to_dict(example.opponent),
        "turn": example.turn,
        "player_pack": example.player_pack,
        "opponent_pack": example.opponent_pack,
        "target": list(example.target),
    }


def battle_example_from_dict(value: dict[str, Any]) -> BattleExample:
    return BattleExample(
        player=team_from_dict(value["player"]),
        opponent=team_from_dict(value["opponent"]),
        turn=int(value["turn"]),
        player_pack=str(value["player_pack"]),
        opponent_pack=str(value["opponent_pack"]),
        target=tuple(float(item) for item in value["target"]),
    )


def write_battle_examples(path: str | Path, examples: Iterable[BattleExample]) -> int:
    return write_jsonl(path, (battle_example_to_dict(example) for example in examples))


def read_battle_examples(path: str | Path) -> list[BattleExample]:
    return [battle_example_from_dict(row) for row in read_jsonl(path)]


def build_battle_dataset(
    boards: Sequence[BoardSnapshot],
    output_dir: str | Path,
    simulator: BattleSimulator,
    *,
    examples: int,
    simulations_per_pair: int = 8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Split first, then label pairs independently within each split."""

    if examples < 1:
        raise ValueError("examples must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    board_splits = split_boards(
        boards,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    fractions = {
        "train": 1.0 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    counts: dict[str, int] = {}
    board_counts: dict[str, int] = {}
    remaining = examples
    for index, split_name in enumerate(SPLITS):
        split_boards_value = getattr(board_splits, split_name)
        board_counts[split_name] = len(split_boards_value)
        requested = (
            remaining
            if index == len(SPLITS) - 1
            else min(remaining, round(examples * fractions[split_name]))
        )
        if requested and split_boards_value:
            try:
                labeled = label_board_pairs(
                    split_boards_value,
                    simulator,
                    examples=requested,
                    simulations_per_pair=simulations_per_pair,
                    seed=seed + index,
                )
            except ValueError as error:
                raise ValueError(
                    f"{split_name} split cannot form compatible pairs; add more replay groups "
                    "or adjust split fractions"
                ) from error
        else:
            labeled = []
        counts[split_name] = write_battle_examples(output / f"{split_name}.jsonl", labeled)
        remaining -= requested

    catalog_payload = []
    if simulator.catalog is not None:
        catalog_payload = [
            {
                "id": spec.id,
                "name": spec.name,
                "tier": spec.tier,
                "attack": spec.attack,
                "health": spec.health,
                "packs": spec.packs,
                "ability_text": spec.ability_text,
            }
            for spec in sorted(simulator.catalog.pets.values(), key=lambda item: item.id)
        ]
    catalog_fingerprint = hashlib.sha256(
        json.dumps(catalog_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "format": "sapai-battle-dataset-v1",
        "seed": seed,
        "examples": counts,
        "boards": board_counts,
        "simulations_per_pair": simulations_per_pair,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "split_unit": "replay_id",
        "catalog_sha256": catalog_fingerprint,
        "rules_source": dict(simulator.rules.source),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
