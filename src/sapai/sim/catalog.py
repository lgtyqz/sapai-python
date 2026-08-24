from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sapai.sim.models import FoodSpec, PetSpec

PACK_ALIASES = {
    "Turtle": "Pack1",
    "Puppy": "Pack2",
    "Star": "Pack3",
    "Golden": "Pack4",
    "Unicorn": "Pack5",
    "Danger": "Danger",
}

UNTARGETED_FOODS = {
    "Canned Food",
    "Pizza",
    "Salad Bowl",
    "Sushi",
}


def _ability_text(raw: dict[str, Any]) -> tuple[str, ...]:
    abilities = raw.get("Abilities") or []
    values: list[str] = []
    for ability in abilities:
        if isinstance(ability, dict):
            text = ability.get("About") or ability.get("Description")
            if text:
                values.append(str(text))
        elif ability:
            values.append(str(ability))
    if not values and raw.get("Ability"):
        values.append(str(raw["Ability"]))
    return tuple(values)


@dataclass(frozen=True)
class Catalog:
    pets: dict[int, PetSpec]
    foods: dict[int, FoodSpec]
    perks: dict[int, str]

    @classmethod
    def from_json_dir(cls, directory: str | Path) -> Catalog:
        root = Path(directory)
        with (root / "pets.json").open(encoding="utf-8") as handle:
            raw_pets = json.load(handle)
        with (root / "food.json").open(encoding="utf-8") as handle:
            raw_food = json.load(handle)
        perks_path = root / "perks.json"
        raw_perks = []
        if perks_path.exists():
            with perks_path.open(encoding="utf-8") as handle:
                raw_perks = json.load(handle)

        pets: dict[int, PetSpec] = {}
        for raw in raw_pets:
            try:
                pet_id = int(raw["Id"])
                pets[pet_id] = PetSpec(
                    id=pet_id,
                    name=str(raw["Name"]),
                    tier=int(raw["Tier"]),
                    attack=int(raw.get("Attack", 0)),
                    health=int(raw.get("Health", 0)),
                    packs=tuple(str(value) for value in raw.get("Packs", [])),
                    ability_text=_ability_text(raw),
                )
            except (KeyError, TypeError, ValueError):
                continue

        foods: dict[int, FoodSpec] = {}
        for raw in raw_food:
            try:
                food_id = int(raw["Id"])
                name = str(raw["Name"])
                foods[food_id] = FoodSpec(
                    id=food_id,
                    name=name,
                    tier=int(raw["Tier"]),
                    packs=tuple(str(value) for value in raw.get("Packs", [])),
                    cost=1 if name in {"Sleeping Pill", "Pill"} else 3,
                    targets_pet=name not in UNTARGETED_FOODS,
                    ability_text=str(raw.get("Ability", "")),
                )
            except (KeyError, TypeError, ValueError):
                continue

        perks = {
            int(raw["Id"]): str(raw["Name"])
            for raw in raw_perks
            if raw.get("Id") is not None and raw.get("Name")
        }
        return cls(pets=pets, foods=foods, perks=perks)

    def pet_by_name(self, name: str) -> PetSpec:
        return next(spec for spec in self.pets.values() if spec.name == name)

    def food_by_name(self, name: str) -> FoodSpec:
        return next(spec for spec in self.foods.values() if spec.name == name)

    def pack_pets(self, pack: str, *, through_tier: int = 6) -> list[PetSpec]:
        pack_id = PACK_ALIASES.get(pack, pack)
        return [
            spec
            for spec in self.pets.values()
            if spec.tier <= through_tier and pack_id in spec.packs
        ]

    def pack_foods(self, pack: str, *, through_tier: int = 6) -> list[FoodSpec]:
        pack_id = PACK_ALIASES.get(pack, pack)
        return [
            spec
            for spec in self.foods.values()
            if spec.tier <= through_tier and pack_id in spec.packs
        ]
