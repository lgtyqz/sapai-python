from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

Expression = int | float | str | dict[str, Any]

BATTLE_EFFECTS = frozenset(
    {
        "buff",
        "copy_ability",
        "damage",
        "gain_health_from_max_friend",
        "give_perk",
        "release_swallowed",
        "remove_health",
        "remove_health_percent",
        "summon",
        "summon_random_tier",
        "swallow",
    }
)
SHOP_EFFECTS = frozenset(
    {
        "buff",
        "buff_shop_pets",
        "copy_ability",
        "damage",
        "discount_food",
        "food_multiplier",
        "gain_gold",
        "give_perk",
        "replace_food",
        "stock_food",
        "stock_scaled_food",
        "summon",
        "summon_random_tier",
    }
)
FOOD_EFFECTS = frozenset(
    {"add_experience", "buff", "buff_current_and_future_shop", "faint", "give_perk"}
)
PERK_EFFECTS = frozenset({"revive", "summon"})


def evaluate(expression: Expression, variables: Mapping[str, int | float]) -> int | float:
    """Evaluate the deliberately small arithmetic language used by rule data."""

    if isinstance(expression, (int, float)):
        return expression
    if isinstance(expression, str):
        if expression not in variables:
            raise ValueError(f"unknown rule variable: {expression}")
        return variables[expression]
    if not isinstance(expression, dict) or len(expression) != 1:
        raise ValueError(f"invalid rule expression: {expression!r}")
    operator, value = next(iter(expression.items()))
    if operator in {"add", "mul", "min", "max"}:
        values = [evaluate(item, variables) for item in value]
        if operator == "add":
            return sum(values)
        if operator == "mul":
            result: int | float = 1
            for item in values:
                result *= item
            return result
        return min(values) if operator == "min" else max(values)
    result = evaluate(value, variables)
    if operator == "ceil":
        return math.ceil(result)
    if operator == "floor":
        return math.floor(result)
    raise ValueError(f"unknown rule expression operator: {operator}")


@dataclass(frozen=True, slots=True)
class RuleBook:
    """Validated, immutable view of the JSON simulation rules."""

    data: dict[str, Any]

    @classmethod
    def turtle(cls, path: str | Path | None = None) -> RuleBook:
        source = Path(path) if path else files("sapai.sim").joinpath("rules", "turtle.json")
        with source.open(encoding="utf-8") as handle:
            data = json.load(handle)
        book = cls(data)
        book.validate()
        return book

    def validate(self) -> None:
        if self.data.get("schema_version") != 1:
            raise ValueError("unsupported simulation rule schema")
        for section in ("pets", "foods", "perks"):
            if not isinstance(self.data.get(section), dict):
                raise TypeError(f"rule section {section!r} must be an object")
        for pet, definition in self.data["pets"].items():
            rules = definition.get("rules", [])
            if not isinstance(rules, list):
                raise TypeError(f"rules for {pet!r} must be a list")
            for rule in rules:
                if rule.get("phase") not in {"battle", "shop", "both"}:
                    raise ValueError(f"invalid phase in {pet!r}: {rule!r}")
                if not rule.get("trigger") or not isinstance(rule.get("effects"), list):
                    raise ValueError(f"invalid rule for {pet!r}: {rule!r}")
                if any("op" not in effect for effect in rule["effects"]):
                    raise ValueError(f"effect without op for {pet!r}")
                allowed = (
                    BATTLE_EFFECTS
                    if rule["phase"] == "battle"
                    else SHOP_EFFECTS
                    if rule["phase"] == "shop"
                    else BATTLE_EFFECTS & SHOP_EFFECTS
                )
                unknown = {effect["op"] for effect in rule["effects"]} - allowed
                if unknown:
                    raise ValueError(f"unsupported effects for {pet!r}: {sorted(unknown)}")
        for food, definition in self.data["foods"].items():
            unknown = {effect["op"] for effect in definition.get("effects", [])} - FOOD_EFFECTS
            if unknown:
                raise ValueError(f"unsupported food effects for {food!r}: {sorted(unknown)}")
        for perk, definition in self.data["perks"].items():
            unknown = {effect["op"] for effect in definition.get("on_faint", [])} - PERK_EFFECTS
            if unknown:
                raise ValueError(f"unsupported perk effects for {perk!r}: {sorted(unknown)}")

    @property
    def source(self) -> Mapping[str, Any]:
        return self.data["source"]

    @property
    def supported_pets(self) -> frozenset[str]:
        return frozenset(self.data["pets"])

    @property
    def supported_foods(self) -> frozenset[str]:
        return frozenset(self.data["foods"])

    def pet_definition(self, name: str) -> Mapping[str, Any]:
        return self.data["pets"].get(name, {})

    def pet_rules(self, name: str, phase: str, trigger: str) -> list[Mapping[str, Any]]:
        definition = self.pet_definition(name)
        return [
            rule
            for rule in definition.get("rules", [])
            if rule["trigger"] == trigger and rule["phase"] in {phase, "both"}
        ]

    def food_effects(self, name: str) -> list[Mapping[str, Any]]:
        return list(self.data["foods"].get(name, {}).get("effects", []))

    def food_definition(self, name: str) -> Mapping[str, Any]:
        return self.data["foods"].get(name, {})

    def perk_definition(self, name: str | None) -> Mapping[str, Any]:
        if name is None:
            return {}
        return self.data["perks"].get(name, {})
