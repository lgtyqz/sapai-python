"""Stable JSON representations for simulator and training records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from sapai.data.replay import BoardSnapshot
from sapai.sim.actions import Action, ActionKind
from sapai.sim.models import Food, Pet, RunState, Shop, ShopPet, Team

_METADATA_PET_KEY = "__sapai_serialized_pet__"


def _metadata_to_json(value: Any) -> Any:
    """Encode simulator metadata without discarding nested runtime pets."""

    if isinstance(value, Pet):
        return {_METADATA_PET_KEY: pet_to_dict(value)}
    if isinstance(value, Mapping):
        return {str(key): _metadata_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported simulator metadata value: {type(value).__name__}")


def _metadata_from_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {_METADATA_PET_KEY}:
            return pet_from_dict(value[_METADATA_PET_KEY])
        return {str(key): _metadata_from_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_from_json(item) for item in value]
    return value


def pet_to_dict(pet: Pet) -> dict[str, Any]:
    return {
        "id": pet.id,
        "name": pet.name,
        "tier": pet.tier,
        "attack": pet.attack,
        "health": pet.health,
        "experience": pet.experience,
        "perk": pet.perk,
        "mana": pet.mana,
        "temporary_attack": pet.temporary_attack,
        "temporary_health": pet.temporary_health,
        "triggers_consumed": pet.triggers_consumed,
        "instance_id": pet.instance_id,
        "metadata": _metadata_to_json(pet.metadata),
    }


def pet_from_dict(value: Mapping[str, Any]) -> Pet:
    pet_id = int(value["id"])
    name = str(value["name"])
    metadata = _metadata_from_json(value.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise TypeError("serialized pet metadata must be an object")
    if pet_id >= 0 and name == f"Pet #{pet_id}":
        metadata.setdefault("vanilla_fallback", "unknown_pet_id")
    perk = value.get("perk")
    if isinstance(perk, str) and perk.startswith("Perk #"):
        metadata.setdefault("perk_fallback", "unknown_perk_id")
    return Pet(
        id=pet_id,
        name=name,
        tier=int(value["tier"]),
        attack=int(value["attack"]),
        health=int(value["health"]),
        experience=int(value.get("experience", 0)),
        perk=perk,
        mana=int(value.get("mana", 0)),
        temporary_attack=int(value.get("temporary_attack", 0)),
        temporary_health=int(value.get("temporary_health", 0)),
        triggers_consumed=int(value.get("triggers_consumed", 0)),
        instance_id=int(value.get("instance_id", 0)),
        metadata=metadata,
    )


def team_to_dict(team: Team) -> list[dict[str, Any] | None]:
    return [pet_to_dict(pet) if pet is not None else None for pet in team.slots]


def team_from_dict(value: Iterable[Mapping[str, Any] | None]) -> Team:
    return Team([_team_pet_from_dict(pet) for pet in value])


def _team_pet_from_dict(value: Mapping[str, Any] | None) -> Pet | None:
    if value is None:
        return None
    pet_id = int(value.get("id", -1))
    # Historical exports used ``Pet #-1`` as an empty-slot sentinel. Runtime
    # summons also use negative IDs, but their real token names must survive a
    # battle-timeline round trip.
    if pet_id < 0 and str(value.get("name", "")) == f"Pet #{pet_id}":
        return None
    return pet_from_dict(value)


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.name,
        "source": action.source,
        "target": action.target,
        "order": list(action.order),
    }


def action_from_dict(value: Mapping[str, Any]) -> Action:
    kind = value["kind"]
    action_kind = ActionKind[kind] if isinstance(kind, str) else ActionKind(int(kind))
    return Action(
        action_kind,
        int(value.get("source", -1)),
        int(value.get("target", -1)),
        tuple(int(position) for position in value.get("order", ())),
    )


def board_to_dict(board: BoardSnapshot) -> dict[str, Any]:
    return {
        "replay_id": board.replay_id,
        "side": board.side,
        "turn": board.turn,
        "pack": board.pack,
        "team": team_to_dict(board.team),
        "toy": board.toy,
        "toy_level": board.toy_level,
        "gold_spent": board.gold_spent,
        "rolls": board.rolls,
        "summoned": board.summoned,
        "level_three_sold": board.level_three_sold,
        "transformations": board.transformations,
        "version": board.version,
    }


def board_from_dict(value: Mapping[str, Any]) -> BoardSnapshot:
    return BoardSnapshot(
        replay_id=value.get("replay_id"),
        side=str(value.get("side", "player")),
        turn=int(value["turn"]),
        pack=str(value["pack"]),
        team=team_from_dict(value["team"]),
        toy=value.get("toy"),
        toy_level=int(value.get("toy_level", 1)),
        gold_spent=int(value.get("gold_spent", 0)),
        rolls=int(value.get("rolls", 0)),
        summoned=int(value.get("summoned", 0)),
        level_three_sold=int(value.get("level_three_sold", 0)),
        transformations=int(value.get("transformations", 0)),
        version=str(value.get("version", "unknown")),
    )


def run_state_to_dict(state: RunState) -> dict[str, Any]:
    return {
        "team": team_to_dict(state.team),
        "shop": {
            "pets": [
                {
                    "pet": pet_to_dict(offer.pet),
                    "frozen": offer.frozen,
                    "reward_group": offer.reward_group,
                    "freeze_toggled": offer.freeze_toggled,
                }
                for offer in state.shop.pets
            ],
            "foods": [
                {
                    "id": food.id,
                    "name": food.name,
                    "tier": food.tier,
                    "cost": food.cost,
                    "targets_pet": food.targets_pet,
                    "frozen": food.frozen,
                    "reward_group": food.reward_group,
                    "freeze_toggled": food.freeze_toggled,
                }
                for food in state.shop.foods
            ],
        },
        "lives": state.lives,
        "turn": state.turn,
        "gold": state.gold,
        "trophies": state.trophies,
        "shop_attack": state.shop_attack,
        "shop_health": state.shop_health,
        "pack": state.pack,
        "version": state.version,
        "awaiting_battle": state.awaiting_battle,
        "next_instance_id": state.next_instance_id,
        "next_reward_group": state.next_reward_group,
        "rolls_this_turn": state.rolls_this_turn,
        "gold_spent_this_turn": state.gold_spent_this_turn,
        "metadata": _metadata_to_json(state.metadata),
    }


def run_state_from_dict(value: Mapping[str, Any]) -> RunState:
    raw_shop = value.get("shop", {})
    shop = Shop(
        pets=[
            ShopPet(
                pet_from_dict(offer["pet"]),
                frozen=bool(offer.get("frozen", False)),
                reward_group=offer.get("reward_group"),
                freeze_toggled=bool(offer.get("freeze_toggled", False)),
            )
            for offer in raw_shop.get("pets", [])
        ],
        foods=[
            Food(
                id=int(food["id"]),
                name=str(food["name"]),
                tier=int(food["tier"]),
                cost=int(food.get("cost", 3)),
                targets_pet=bool(food.get("targets_pet", True)),
                frozen=bool(food.get("frozen", False)),
                reward_group=food.get("reward_group"),
                freeze_toggled=bool(food.get("freeze_toggled", False)),
            )
            for food in raw_shop.get("foods", [])
        ],
    )
    return RunState(
        team=team_from_dict(value["team"]),
        shop=shop,
        lives=int(value.get("lives", 5)),
        turn=int(value.get("turn", 1)),
        gold=int(value.get("gold", 10)),
        trophies=int(value.get("trophies", 0)),
        shop_attack=int(value.get("shop_attack", 0)),
        shop_health=int(value.get("shop_health", 0)),
        pack=str(value.get("pack", "Turtle")),
        version=str(value.get("version", "current")),
        awaiting_battle=bool(value.get("awaiting_battle", False)),
        next_instance_id=int(value.get("next_instance_id", 1)),
        next_reward_group=int(value.get("next_reward_group", 1)),
        rolls_this_turn=int(value.get("rolls_this_turn", 0)),
        gold_spent_this_turn=int(value.get("gold_spent_this_turn", 0)),
        metadata=_state_metadata_from_json(value.get("metadata", {})),
    )


def _state_metadata_from_json(value: Any) -> dict[str, Any]:
    metadata = _metadata_from_json(value)
    if not isinstance(metadata, dict):
        raise TypeError("serialized run-state metadata must be an object")
    return metadata


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_boards(path: str | Path, boards: Iterable[BoardSnapshot]) -> int:
    return write_jsonl(path, (board_to_dict(board) for board in boards))


def read_boards(path: str | Path) -> Iterator[BoardSnapshot]:
    return (board_from_dict(row) for row in read_jsonl(path))
