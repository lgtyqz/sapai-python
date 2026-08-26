from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sapai.sim.catalog import Catalog
from sapai.sim.models import Food, Pet, RunState
from sapai.sim.rules import RuleBook, evaluate

DEFAULT_TURTLE_RULES = RuleBook.turtle()


@dataclass(slots=True)
class ShopDispatchResult:
    value: int = 1
    summons: list[Pet] = field(default_factory=list)

    def extend(self, other: ShopDispatchResult) -> None:
        self.value = max(self.value, other.value)
        self.summons.extend(other.summons)


class ShopAbilityEngine:
    """Interprets shop, food, and shop-side combat-style rule triggers."""

    implemented_pets = frozenset(
        name
        for name, definition in DEFAULT_TURTLE_RULES.data["pets"].items()
        if any(rule["phase"] in {"shop", "both"} for rule in definition.get("rules", []))
    )
    implemented_foods = DEFAULT_TURTLE_RULES.supported_foods

    def __init__(self, catalog: Catalog, rules: RuleBook | None = None):
        self.catalog = catalog
        self.rules = rules or DEFAULT_TURTLE_RULES

    def on_sell(self, state: RunState, index: int, rng: random.Random) -> None:
        self._dispatch(state, index, "sell", rng)

    def on_buy(self, state: RunState, index: int, rng: random.Random) -> None:
        bought = state.team.slots[index]
        if bought is None:
            return
        self._dispatch(state, index, "buy", rng)
        if bought.tier == 1:
            self._dispatch_observers(
                state,
                "tier_one_friend_bought",
                rng,
                trigger_pet=bought,
                exclude=bought,
            )

    def on_summoned(self, state: RunState, index: int, rng: random.Random) -> None:
        summoned = state.team.slots[index]
        if summoned is None:
            return
        self._dispatch(state, index, "summoned", rng)
        self._dispatch_observers(
            state,
            "friend_summoned",
            rng,
            trigger_pet=summoned,
            exclude=summoned,
        )

    def on_level_up(self, state: RunState, index: int, old_level: int, rng: random.Random) -> None:
        self._dispatch(state, index, "level_up", rng, old_level=old_level)

    def on_shop_faint(self, state: RunState, index: int, rng: random.Random) -> list[Pet]:
        return self._dispatch(state, index, "faint", rng).summons

    def on_perk_faint(self, state: RunState, pet: Pet) -> list[Pet]:
        summons: list[Pet] = []
        for effect in self.rules.perk_definition(pet.perk).get("on_faint", []):
            if effect["op"] == "summon":
                for _ in range(int(effect.get("count", 1))):
                    summons.append(
                        self._token(
                            state,
                            str(effect["name"]),
                            int(effect["attack"]),
                            int(effect["health"]),
                            int(effect.get("tier", 1)),
                            1,
                        )
                    )
            elif effect["op"] == "revive":
                revived = pet.clone()
                revived.attack = int(effect["attack"])
                revived.health = int(effect["health"])
                revived.temporary_attack = 0
                revived.temporary_health = 0
                revived.perk = None
                revived.instance_id = state.allocate_instance_id()
                summons.append(revived)
        return summons

    def on_friend_fainted(
        self,
        state: RunState,
        fainted: Pet,
        position: int,
        rng: random.Random,
    ) -> list[Pet]:
        return self._dispatch_observers(
            state,
            "friend_fainted",
            rng,
            trigger_pet=fainted,
            trigger_position=position,
        ).summons

    def start_turn(self, state: RunState, rng: random.Random) -> None:
        state.metadata["turn_team_uses"] = {}
        for _, pet in self._pets(state):
            pet.metadata["turn_uses"] = {}
            pet.metadata["turn_counters"] = {}
            if self.rules.pet_definition(pet.name).get("copy_resets_at_start_turn"):
                pet.metadata.pop("copied_ability_name", None)
        self._ordered_dispatch(state, "start_turn", rng)

    def end_turn(self, state: RunState, rng: random.Random) -> None:
        team_uses: dict[str, int] = {}
        for pet in self._ordered_pets(state, rng):
            if pet not in state.team.slots:
                continue
            owner_index = state.team.slots.index(pet)
            self._dispatch(
                state,
                owner_index,
                "end_turn",
                rng,
                team_uses=team_uses,
            )
            self._dispatch_perk(
                state,
                owner_index,
                "end_turn",
                rng,
                team_uses=team_uses,
            )

    def apply_food(
        self,
        state: RunState,
        food: Food,
        target: int,
        rng: random.Random,
        *,
        faint_callback,
        level_callback,
    ) -> None:
        definition = self.rules.food_definition(food.name)
        generated = state.metadata.get("generated_food_bonuses", {})
        if not definition and food.name not in generated:
            raise NotImplementedError(f"shop food ability is not implemented: {food.name}")

        multiplier = 1
        if definition.get("stat_food") or food.name in generated:
            for owner_index, _ in self._pets(state):
                result = self._dispatch(state, owner_index, "food_multiplier", rng)
                # Cat modifiers add their extra copies instead of multiplying
                # one another or collapsing to the strongest Cat. A level-one
                # Cat contributes one extra copy, so two make Pear resolve 3x.
                multiplier += max(0, result.value - 1)

        if food.name in generated:
            attack, health = generated[food.name]
            self._buff_target(state, target, attack * multiplier, health * multiplier)
        else:
            variables = {"food_multiplier": multiplier}
            for effect in self.rules.food_effects(food.name):
                if effect["op"] == "faint":
                    faint_callback(target)
                    return
                if effect["op"] == "add_experience":
                    level_callback(target, int(effect.get("amount", 1)))
                    continue
                self._apply_effect(
                    state,
                    -1,
                    effect,
                    variables,
                    rng,
                    trigger_pet=None,
                    trigger_position=None,
                    food_target=target,
                    food_multiplier=multiplier,
                )

        if target >= 0 and state.team.slots[target] is not None:
            eater = state.team.slots[target]
            self._dispatch(state, target, "ate_food", rng)
            self._dispatch_observers(
                state,
                "friendly_ate_food",
                rng,
                trigger_pet=eater,
            )

    def _ordered_dispatch(
        self,
        state: RunState,
        trigger: str,
        rng: random.Random,
        team_uses: dict[str, int] | None = None,
    ) -> ShopDispatchResult:
        result = ShopDispatchResult()
        ordered = self._ordered_pets(state, rng)
        shared = team_uses if team_uses is not None else {}
        for pet in ordered:
            if pet not in state.team.slots:
                continue
            result.extend(
                self._dispatch(
                    state,
                    state.team.slots.index(pet),
                    trigger,
                    rng,
                    team_uses=shared,
                )
            )
        return result

    def _dispatch_perk(
        self,
        state: RunState,
        owner_index: int,
        trigger: str,
        rng: random.Random,
        *,
        team_uses: dict[str, int] | None = None,
    ) -> ShopDispatchResult:
        owner = state.team.slots[owner_index]
        result = ShopDispatchResult()
        if owner is None:
            return result
        variables = {
            "level": owner.level,
            "old_level": owner.level,
            "new_level_minus_one": max(0, owner.level - 1),
            "attack": owner.effective_attack,
            "health": owner.effective_health,
            "trigger_tier": 0,
        }
        for ordinal, rule in enumerate(self.rules.perk_rules(owner.perk, "shop", trigger)):
            key = f"perk:{owner.perk}:{trigger}:{ordinal}"
            if not self._rule_ready(
                state,
                owner_index,
                rule,
                key,
                variables,
                team_uses,
                None,
                None,
            ):
                continue
            for effect in rule["effects"]:
                result.extend(
                    self._apply_effect(
                        state,
                        owner_index,
                        effect,
                        variables,
                        rng,
                        trigger_pet=None,
                        trigger_position=None,
                    )
                )
        return result

    def _dispatch_observers(
        self,
        state: RunState,
        trigger: str,
        rng: random.Random,
        *,
        trigger_pet: Pet,
        trigger_position: int | None = None,
        exclude: Pet | None = None,
    ) -> ShopDispatchResult:
        result = ShopDispatchResult()
        for pet in self._ordered_pets(state, rng):
            if pet is exclude or pet not in state.team.slots:
                continue
            result.extend(
                self._dispatch(
                    state,
                    state.team.slots.index(pet),
                    trigger,
                    rng,
                    trigger_pet=trigger_pet,
                    trigger_position=trigger_position,
                )
            )
        return result

    def _dispatch(
        self,
        state: RunState,
        owner_index: int,
        trigger: str,
        rng: random.Random,
        *,
        trigger_pet: Pet | None = None,
        trigger_position: int | None = None,
        old_level: int | None = None,
        team_uses: dict[str, int] | None = None,
    ) -> ShopDispatchResult:
        owner = state.team.slots[owner_index]
        result = ShopDispatchResult()
        if owner is None:
            return result
        ability_name = str(owner.metadata.get("copied_ability_name", owner.name))
        variables = {
            "level": owner.level,
            "old_level": old_level if old_level is not None else owner.level,
            "new_level_minus_one": max(0, owner.level - 1),
            "attack": owner.effective_attack,
            "health": owner.effective_health,
            "trigger_tier": trigger_pet.tier if trigger_pet else 0,
        }
        for ordinal, rule in enumerate(self.rules.pet_rules(ability_name, "shop", trigger)):
            key = f"{ability_name}:{trigger}:{ordinal}"
            if not self._rule_ready(
                state,
                owner_index,
                rule,
                key,
                variables,
                team_uses,
                trigger_pet,
                trigger_position,
            ):
                continue
            for effect in rule["effects"]:
                effect_result = self._apply_effect(
                    state,
                    owner_index,
                    effect,
                    variables,
                    rng,
                    trigger_pet=trigger_pet,
                    trigger_position=trigger_position,
                )
                result.extend(effect_result)
        return result

    def _rule_ready(
        self,
        state: RunState,
        owner_index: int,
        rule: Mapping[str, Any],
        key: str,
        variables: Mapping[str, int],
        team_uses: dict[str, int] | None,
        trigger_pet: Pet | None,
        trigger_position: int | None,
    ) -> bool:
        owner = state.team.slots[owner_index]
        if owner is None:
            return False
        for condition in rule.get("conditions", []):
            kind = condition["kind"]
            if kind == "last_outcome":
                if state.metadata.get("last_outcome") != condition["value"]:
                    return False
            elif kind == "has_level_three_friend":
                if not any(pet is not owner and pet.level == 3 for _, pet in self._pets(state)):
                    return False
            elif kind == "level_below":
                if variables["level"] >= int(condition["value"]):
                    return False
            elif kind == "old_level_below":
                if variables["old_level"] >= int(condition["value"]):
                    return False
            elif kind == "trigger_ahead":
                if trigger_position is None or trigger_position >= owner_index:
                    return False
            elif kind == "trigger_name_not":
                if trigger_pet is not None and trigger_pet.name == condition["value"]:
                    return False
            elif kind == "team_has_room":
                if len(self._pets(state)) >= 5:
                    return False
            else:
                raise ValueError(f"unknown shop condition: {kind}")

        counters = owner.metadata.setdefault("turn_counters", {})
        if "counter_every" in rule:
            counters[key] = int(counters.get(key, 0)) + 1
            if counters[key] % int(rule["counter_every"]) != 0:
                return False
        uses = owner.metadata.setdefault("turn_uses", {})
        maximum = rule.get("max_uses")
        if maximum is not None:
            limit = int(evaluate(maximum, variables))
            if int(uses.get(key, 0)) >= limit:
                return False
            uses[key] = int(uses.get(key, 0)) + 1
        if rule.get("once_per_team"):
            shared = (
                team_uses
                if team_uses is not None
                else state.metadata.setdefault("turn_team_uses", {})
            )
            if shared.get(key, 0):
                return False
            shared[key] = 1
        return True

    def _apply_effect(
        self,
        state: RunState,
        owner_index: int,
        effect: Mapping[str, Any],
        variables: Mapping[str, int],
        rng: random.Random,
        *,
        trigger_pet: Pet | None,
        trigger_position: int | None,
        food_target: int | None = None,
        food_multiplier: int = 1,
    ) -> ShopDispatchResult:
        result = ShopDispatchResult()
        op = effect["op"]
        count = int(evaluate(effect.get("count", 1), variables))
        if op in {"buff", "give_perk", "damage"}:
            targets = self._targets(
                state,
                owner_index,
                str(effect["target"]),
                count,
                rng,
                trigger_pet,
                food_target,
            )
            multiplier = food_multiplier if food_target is not None or owner_index < 0 else 1
            for target_index, target in targets:
                if op == "buff":
                    target.buff(
                        int(evaluate(effect.get("attack", 0), variables)) * multiplier,
                        int(evaluate(effect.get("health", 0), variables)) * multiplier,
                        temporary=bool(effect.get("temporary")),
                    )
                elif op == "give_perk":
                    target.perk = str(effect["perk"])
                else:
                    source = state.team.slots[owner_index] if owner_index >= 0 else None
                    amount = int(evaluate(effect["amount"], variables))
                    self._damage(state, target_index, amount, rng, source=source)
            return result

        owner = state.team.slots[owner_index] if owner_index >= 0 else None
        if op == "gain_gold":
            state.gold += int(evaluate(effect["amount"], variables))
        elif op == "buff_shop_pets":
            for offer in state.shop.pets:
                offer.pet.buff(
                    int(evaluate(effect.get("attack", 0), variables)),
                    int(evaluate(effect.get("health", 0), variables)),
                )
        elif op == "stock_food":
            for _ in range(count):
                state.shop.foods.append(
                    Food(
                        -300,
                        str(effect["name"]),
                        tier=1,
                        cost=int(effect.get("cost", 3)),
                        targets_pet=bool(effect.get("targets_pet", True)),
                    )
                )
        elif op == "stock_scaled_food":
            if owner is None:
                return result
            name = effect["names"][owner.level - 1]
            state.shop.foods.append(
                Food(-200 - owner.level, name, 1, cost=int(effect["cost"]), targets_pet=True)
            )
            state.metadata.setdefault("generated_food_bonuses", {})[name] = (
                int(evaluate(effect["attack"], variables)),
                int(evaluate(effect["health"], variables)),
            )
        elif op == "replace_food":
            if owner is None:
                return result
            name = effect["names"][owner.level - 1]
            state.shop.foods = [
                Food(-100 - owner.level, name, 5, cost=int(effect["cost"]), targets_pet=True)
                for _ in range(count)
            ]
            state.metadata.setdefault("generated_food_bonuses", {})[name] = (
                int(effect["attack_by_level"][owner.level - 1]),
                int(effect["health_by_level"][owner.level - 1]),
            )
        elif op == "discount_food":
            amount = int(evaluate(effect["amount"], variables))
            for food in state.shop.foods:
                food.cost = max(0, food.cost - amount)
        elif op == "copy_ability":
            targets = self._targets(
                state, owner_index, str(effect["target"]), count, rng, trigger_pet, food_target
            )
            if owner is not None and targets:
                copied = targets[0][1]
                owner.metadata["copied_ability_name"] = copied.metadata.get(
                    "copied_ability_name", copied.name
                )
        elif op == "food_multiplier":
            result.value = int(evaluate(effect["amount"], variables))
        elif op == "increase_sell_value":
            if owner is not None:
                bonus = int(evaluate(effect["amount"], variables))
                owner.metadata["sell_value_bonus"] = (
                    int(owner.metadata.get("sell_value_bonus", 0)) + bonus
                )
        elif op == "buff_current_and_future_shop":
            attack = int(evaluate(effect.get("attack", 0), variables)) * food_multiplier
            health = int(evaluate(effect.get("health", 0), variables)) * food_multiplier
            state.shop_attack += attack
            state.shop_health += health
            for offer in state.shop.pets:
                offer.pet.buff(attack, health)
        elif op in {"summon", "summon_random_tier"}:
            for _ in range(count):
                if op == "summon_random_tier":
                    pet = self._random_tier_pet(
                        state,
                        int(effect["tier"]),
                        int(evaluate(effect["attack"], variables)),
                        int(evaluate(effect["health"], variables)),
                        int(evaluate(effect.get("level", 1), variables)),
                        rng,
                    )
                else:
                    pet = self._token(
                        state,
                        str(effect["name"]),
                        int(evaluate(effect["attack"], variables)),
                        int(evaluate(effect["health"], variables)),
                        int(effect.get("tier", 1)),
                        int(evaluate(effect.get("level", 1), variables)),
                    )
                    pet.perk = effect.get("perk")
                if pet is not None:
                    result.summons.append(pet)
        else:
            raise ValueError(f"unknown shop effect op: {op}")
        return result

    def _damage(
        self,
        state: RunState,
        target_index: int,
        amount: int,
        rng: random.Random,
        *,
        source: Pet | None,
    ) -> None:
        target = state.team.slots[target_index]
        if target is None or amount <= 0:
            return
        perk = self.rules.perk_definition(target.perk)
        reduction = int(perk.get("damage_reduction", 0))
        minimum = int(perk.get("minimum_damage", 0))
        amount = max(minimum, amount - reduction)
        if perk.get("consume_on_hurt"):
            target.perk = None
        dealt = amount
        absorbed = min(max(0, target.temporary_health), dealt)
        target.temporary_health -= absorbed
        target.health -= dealt - absorbed
        if source is not None and self.rules.perk_definition(source.perk).get("lethal_on_damage"):
            target.health = min(target.health, 0)
        self._dispatch(state, target_index, "hurt", rng, trigger_pet=source)
        self._dispatch_observers(
            state,
            "friend_hurt",
            rng,
            trigger_pet=target,
            exclude=target,
        )

    def _targets(
        self,
        state: RunState,
        owner_index: int,
        selector: str,
        count: int,
        rng: random.Random,
        trigger_pet: Pet | None,
        food_target: int | None,
    ) -> list[tuple[int, Pet]]:
        pets = self._pets(state)
        if selector == "self":
            pet = state.team.slots[owner_index]
            return [(owner_index, pet)] if pet is not None else []
        if selector == "trigger":
            if trigger_pet is None or trigger_pet not in state.team.slots:
                return []
            return [(state.team.slots.index(trigger_pet), trigger_pet)]
        if selector == "food_target":
            if food_target is None or food_target < 0:
                return []
            pet = state.team.slots[food_target]
            return [(food_target, pet)] if pet is not None else []
        if selector in {"random_friend", "random_friends"}:
            values = [(index, pet) for index, pet in pets if index != owner_index]
            return rng.sample(values, min(count, len(values)))
        if selector == "random_pets":
            return rng.sample(pets, min(count, len(pets)))
        if selector == "random_level_two_friends":
            values = [
                (index, pet) for index, pet in pets if index != owner_index and pet.level >= 2
            ]
            return rng.sample(values, min(count, len(values)))
        if selector == "all_friends":
            return [(index, pet) for index, pet in pets if index != owner_index]
        if selector == "all_pets_except_self":
            return [(index, pet) for index, pet in pets if index != owner_index]
        if selector == "adjacent":
            return [
                (index, state.team.slots[index])
                for index in (owner_index - 1, owner_index + 1)
                if 0 <= index < len(state.team.slots) and state.team.slots[index] is not None
            ]
        if selector == "nearest_ahead":
            return [
                (index, state.team.slots[index])
                for index in range(owner_index - 1, -1, -1)
                if state.team.slots[index] is not None
            ][:count]
        if selector == "nearest_behind":
            return [
                (index, state.team.slots[index])
                for index in range(owner_index + 1, len(state.team.slots))
                if state.team.slots[index] is not None
            ][:count]
        if selector == "front_friend":
            return pets[:1]
        if selector in {
            "all_enemies",
            "enemy_front",
            "highest_health_enemy",
            "last_enemy",
            "lowest_health_enemy",
            "random_enemies",
            "random_enemy",
        }:
            return []
        raise ValueError(f"unknown shop target selector: {selector}")

    def _random_tier_pet(
        self,
        state: RunState,
        tier: int,
        attack: int,
        health: int,
        level: int,
        rng: random.Random,
    ) -> Pet | None:
        values = [
            spec
            for spec in self.catalog.pack_pets(state.pack, through_tier=tier)
            if spec.tier == tier
        ]
        if not values:
            return None
        pet = rng.choice(values).create(instance_id=state.allocate_instance_id())
        pet.attack = attack
        pet.health = health
        pet.experience = self._experience_for_level(level)
        return pet

    @classmethod
    def _token(
        cls,
        state: RunState,
        name: str,
        attack: int,
        health: int,
        tier: int,
        level: int,
    ) -> Pet:
        return Pet(
            -1,
            name,
            tier,
            attack,
            health,
            experience=cls._experience_for_level(level),
            instance_id=state.allocate_instance_id(),
        )

    @staticmethod
    def _experience_for_level(level: int) -> int:
        return (0, 2, 5)[max(1, min(3, level)) - 1]

    @staticmethod
    def _ordered_pets(state: RunState, rng: random.Random) -> list[Pet]:
        return sorted(
            (pet for _, pet in ShopAbilityEngine._pets(state)),
            key=lambda pet: (pet.effective_attack, rng.random()),
            reverse=True,
        )

    @staticmethod
    def _pets(state: RunState) -> list[tuple[int, Pet]]:
        return [(index, pet) for index, pet in enumerate(state.team.slots) if pet is not None]

    @staticmethod
    def _buff_target(
        state: RunState, index: int, attack: int, health: int, *, temporary: bool = False
    ) -> None:
        pet = state.team.slots[index]
        if pet is None:
            raise ValueError("food target is empty")
        pet.buff(attack, health, temporary=temporary)
