from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sapai.sim.catalog import Catalog
from sapai.sim.models import Pet, Team
from sapai.sim.rules import RuleBook, evaluate


class BattleResultKind(str, Enum):
    PLAYER_WIN = "player_win"
    DRAW = "draw"
    OPPONENT_WIN = "opponent_win"


@dataclass(frozen=True, slots=True)
class BattleFrame:
    """An immutable visual checkpoint in a battle timeline."""

    label: str
    player: Team
    opponent: Team
    log_index: int


@dataclass(slots=True)
class BattleResult:
    outcome: BattleResultKind
    rounds: int
    player: Team
    opponent: Team
    log: list[str] = field(default_factory=list)
    frames: list[BattleFrame] = field(default_factory=list)


class BattleSimulationError(RuntimeError):
    pass


DEFAULT_TURTLE_RULES = RuleBook.turtle()


class BattleSimulator:
    """Dependency-free battle engine driven by declarative JSON rules.

    The engine owns battle lifecycle and event ordering; pet, food, and perk
    behavior lives in ``rules/turtle.json``. Team position zero is the front.
    """

    MAX_ROUNDS = 200
    MAX_EVENTS = 2_000
    SUPPORTED_TURTLE_PETS = DEFAULT_TURTLE_RULES.supported_pets

    def __init__(self, catalog: Catalog | None = None, rules: RuleBook | None = None):
        self.catalog = catalog
        self.rules = rules or DEFAULT_TURTLE_RULES
        self.rng = random.Random()
        self.teams: list[list[Pet]] = [[], []]
        self.log: list[str] = []
        self.frames: list[BattleFrame] = []
        self.events = 0
        self.team_uses: list[dict[str, int]] = [{}, {}]

    def simulate(
        self,
        player: Team,
        opponent: Team,
        *,
        seed: int | None = None,
    ) -> BattleResult:
        self.rng = random.Random(seed)
        self.teams = [
            [pet.clone() for pet in player.slots if pet is not None],
            [pet.clone() for pet in opponent.slots if pet is not None],
        ]
        self.log = []
        self.frames = []
        self.events = 0
        self.team_uses = [{}, {}]
        self._capture("Initial teams")
        for side in (0, 1):
            for pet in self.teams[side]:
                pet.metadata["battle"] = {
                    "uses": {},
                    "counters": {},
                    "faint_processed": False,
                }
                self._execute(side, pet, "summoned")

        self._before_start_battle()
        self._start_battle()
        self._resolve_deaths()
        self._capture("Start-of-battle abilities resolved")

        if len(self.teams[0]) > len(self.teams[1]):
            attacking_side = 0
        elif len(self.teams[1]) > len(self.teams[0]):
            attacking_side = 1
        else:
            attacking_side = self.rng.randrange(2)

        rounds = 0
        while self.teams[0] and self.teams[1] and rounds < self.MAX_ROUNDS:
            rounds += 1
            self._attack(attacking_side)
            self._capture(f"Round {rounds}")
            attacking_side = 1 - attacking_side

        if rounds >= self.MAX_ROUNDS and self.teams[0] and self.teams[1]:
            outcome = BattleResultKind.DRAW
        elif self.teams[0]:
            outcome = BattleResultKind.PLAYER_WIN
        elif self.teams[1]:
            outcome = BattleResultKind.OPPONENT_WIN
        else:
            outcome = BattleResultKind.DRAW
        self._capture(outcome.value.replace("_", " ").title())
        return BattleResult(
            outcome=outcome,
            rounds=rounds,
            player=Team.from_pets(self.teams[0]),
            opponent=Team.from_pets(self.teams[1]),
            log=list(self.log),
            frames=list(self.frames),
        )

    def _capture(self, label: str) -> None:
        self.frames.append(
            BattleFrame(
                label=label,
                player=Team.from_pets(pet.clone() for pet in self.teams[0]),
                opponent=Team.from_pets(pet.clone() for pet in self.teams[1]),
                log_index=len(self.log),
            )
        )

    def _tick(self) -> None:
        self.events += 1
        if self.events > self.MAX_EVENTS:
            raise BattleSimulationError("ability event limit exceeded")

    def _start_battle(self) -> None:
        events: list[tuple[int, float, int, Pet]] = []
        for side in (0, 1):
            for pet in self.teams[side]:
                events.append((pet.effective_attack, self.rng.random(), side, pet))
        events.sort(reverse=True, key=lambda event: (event[0], event[1]))
        for _, _, side, pet in events:
            if pet in self.teams[side] and pet.alive:
                self._execute(side, pet, "start_battle")
                self._resolve_deaths()

    def _before_start_battle(self) -> None:
        events = [
            (pet.effective_attack, self.rng.random(), side, pet)
            for side in (0, 1)
            for pet in self.teams[side]
        ]
        events.sort(reverse=True, key=lambda event: (event[0], event[1]))
        for _, _, side, pet in events:
            if pet in self.teams[side] and pet.alive:
                self._execute(side, pet, "before_start_battle")

    def _attack(self, side: int) -> None:
        enemy_side = 1 - side
        if not self.teams[side] or not self.teams[enemy_side]:
            return
        attacker = self.teams[side][0]
        defender = self.teams[enemy_side][0]
        self._execute(side, attacker, "before_attack")
        if not attacker.alive:
            self._resolve_deaths()
            return

        attack_damage = attacker.effective_attack + self._attack_bonus(attacker)
        defense_damage = defender.effective_attack + self._attack_bonus(defender)
        self.log.append(f"P{side + 1} {attacker.name} attacks P{enemy_side + 1} {defender.name}")
        self._damage(enemy_side, defender, attack_damage, source=attacker)
        self._damage(side, attacker, defense_damage, source=defender)
        self._splash(enemy_side, attacker)
        self._splash(side, defender)

        self._execute(side, attacker, "after_attack")
        attacker_index = self._index(side, attacker)
        if attacker_index is not None:
            for follower in list(self.teams[side][attacker_index + 1 :]):
                self._execute(side, follower, "friend_ahead_attacked", trigger_pet=attacker)

        defender_knocked_out = not defender.alive
        attacker_knocked_out = not attacker.alive
        self._resolve_deaths()
        if defender_knocked_out and attacker in self.teams[side]:
            self._execute(side, attacker, "knockout", trigger_pet=defender)
        if attacker_knocked_out and defender in self.teams[enemy_side]:
            self._execute(enemy_side, defender, "knockout", trigger_pet=attacker)
        self._resolve_deaths()

    def _attack_bonus(self, pet: Pet) -> int:
        perk = self.rules.perk_definition(pet.perk)
        bonus = int(perk.get("attack_bonus", 0))
        if perk.get("consume_on_attack"):
            pet.perk = None
        return bonus

    def _splash(self, target_side: int, attacker: Pet) -> None:
        amount = int(self.rules.perk_definition(attacker.perk).get("splash_damage", 0))
        if amount and len(self.teams[target_side]) > 1:
            self._damage(target_side, self.teams[target_side][1], amount, source=attacker)

    def _damage(self, side: int, target: Pet, amount: int, *, source: Pet | None) -> bool:
        if target not in self.teams[side] or amount <= 0:
            return False
        original = amount
        perk = self.rules.perk_definition(target.perk)
        reduction = int(perk.get("damage_reduction", 0))
        minimum = int(perk.get("minimum_damage", 0))
        amount = max(minimum, amount - reduction) if amount else 0
        if perk.get("consume_on_hurt"):
            target.perk = None

        absorbed = min(max(0, target.temporary_health), amount)
        target.temporary_health -= absorbed
        amount -= absorbed
        target.health -= amount
        hurt = amount > 0
        if (
            hurt
            and source is not None
            and self.rules.perk_definition(source.perk).get("lethal_on_damage")
        ):
            target.health = min(target.health, 0)
        if hurt:
            self.log.append(f"{target.name} takes {original} ({amount} health) damage")
            self._execute(side, target, "hurt", trigger_pet=source)
            for friend in list(self.teams[side]):
                if friend is not target:
                    self._execute(side, friend, "friend_hurt", trigger_pet=target)
        return hurt

    def _resolve_deaths(self) -> None:
        while True:
            candidates: list[tuple[int, float, int, Pet]] = []
            for side in (0, 1):
                for pet in self.teams[side]:
                    battle = pet.metadata.setdefault("battle", {})
                    if not pet.alive and not battle.get("faint_processed", False):
                        candidates.append((pet.effective_attack, self.rng.random(), side, pet))
            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
            _, _, side, pet = candidates[0]
            pet.metadata.setdefault("battle", {})["faint_processed"] = True
            position = self._index(side, pet)
            if position is None:
                continue
            summons = self._execute(side, pet, "faint")
            if pet in self.teams[side]:
                self.teams[side].remove(pet)
            summons.extend(self._perk_faint(pet))
            self._insert_summons(side, position, summons)
            for friend in list(self.teams[side]):
                self._execute(side, friend, "friend_fainted", trigger_pet=pet, position=position)
            self._remove_processed_dead()

    def _remove_processed_dead(self) -> None:
        for side in (0, 1):
            self.teams[side][:] = [
                pet
                for pet in self.teams[side]
                if pet.alive or not pet.metadata.get("battle", {}).get("faint_processed", False)
            ]

    def _perk_faint(self, pet: Pet) -> list[Pet]:
        results: list[Pet] = []
        for effect in self.rules.perk_definition(pet.perk).get("on_faint", []):
            if effect["op"] == "summon":
                for _ in range(int(effect.get("count", 1))):
                    results.append(
                        self._token(effect["name"], int(effect["attack"]), int(effect["health"]))
                    )
            elif effect["op"] == "revive":
                revived = pet.clone()
                revived.attack = int(effect["attack"])
                revived.health = int(effect["health"])
                revived.temporary_attack = revived.temporary_health = 0
                revived.perk = None
                results.append(revived)
        return results

    def _insert_summons(self, side: int, index: int, summons: list[Pet]) -> None:
        room = max(0, 5 - len(self.teams[side]))
        accepted = summons[:room]
        for pet in accepted:
            pet.metadata["battle"] = {
                "uses": {},
                "counters": {},
                "faint_processed": False,
            }
        self.teams[side][index:index] = accepted
        for pet in accepted:
            self._execute(side, pet, "summoned")
            for friend in list(self.teams[side]):
                if friend is not pet:
                    self._execute(side, friend, "friend_summoned", trigger_pet=pet)

    def _execute(
        self,
        side: int,
        pet: Pet,
        trigger: str,
        *,
        trigger_pet: Pet | None = None,
        position: int | None = None,
        level_override: int | None = None,
        allow_tiger: bool = True,
    ) -> list[Pet]:
        if pet not in self.teams[side] and trigger != "faint":
            return []
        ability_name = str(pet.metadata.get("copied_ability_name", pet.name))
        level = level_override or pet.level
        pending: list[Pet] = []
        for ordinal, rule in enumerate(self.rules.pet_rules(ability_name, "battle", trigger)):
            if level_override is not None and rule.get("ignore_repeat"):
                continue
            self._tick()
            key = f"{ability_name}:{trigger}:{ordinal}"
            if not self._rule_ready(side, pet, key, rule, trigger_pet, position, level):
                continue
            variables = self._variables(pet, level, trigger_pet)
            for effect in rule["effects"]:
                pending.extend(
                    self._apply_effect(side, pet, effect, variables, trigger_pet, position)
                )

        if allow_tiger and trigger not in {"friend_summoned", "friend_fainted"}:
            index = self._index(side, pet)
            if index is not None and index + 1 < len(self.teams[side]):
                tiger = self.teams[side][index + 1]
                if tiger.alive and self.rules.pet_definition(tiger.name).get(
                    "repeat_friend_ahead_battle"
                ):
                    pending.extend(
                        self._execute(
                            side,
                            pet,
                            trigger,
                            trigger_pet=trigger_pet,
                            position=position,
                            level_override=tiger.level,
                            allow_tiger=False,
                        )
                    )
        return pending

    def _rule_ready(
        self,
        side: int,
        pet: Pet,
        key: str,
        rule: Mapping[str, Any],
        trigger_pet: Pet | None,
        position: int | None,
        level: int,
    ) -> bool:
        for condition in rule.get("conditions", []):
            kind = condition["kind"]
            if kind == "trigger_ahead":
                owner_index = self._index(side, pet)
                if owner_index is None or position is None or position >= owner_index:
                    return False
            elif kind == "has_level_three_friend":
                if not any(friend is not pet and friend.level == 3 for friend in self.teams[side]):
                    return False
            elif kind == "level_below" and level >= int(condition["value"]):
                return False
            elif kind == "trigger_name_not":
                if trigger_pet is not None and trigger_pet.name == condition["value"]:
                    return False
            elif kind == "team_has_room" and len(self.teams[side]) >= 5:
                return False
            elif kind not in {
                "trigger_ahead",
                "has_level_three_friend",
                "level_below",
                "trigger_name_not",
                "team_has_room",
            }:
                raise BattleSimulationError(f"unknown battle condition: {kind}")

        battle = pet.metadata.setdefault("battle", {})
        counters = battle.setdefault("counters", {})
        if "counter_every" in rule:
            counters[key] = int(counters.get(key, 0)) + 1
            if counters[key] % int(rule["counter_every"]) != 0:
                return False
        uses = battle.setdefault("uses", {})
        maximum = rule.get("max_uses")
        if maximum is not None:
            limit = int(evaluate(maximum, self._variables(pet, level, trigger_pet)))
            if int(uses.get(key, 0)) >= limit:
                return False
            uses[key] = int(uses.get(key, 0)) + 1
        if rule.get("once_per_team"):
            if self.team_uses[side].get(key, 0):
                return False
            self.team_uses[side][key] = 1
        return True

    @staticmethod
    def _variables(pet: Pet, level: int, trigger_pet: Pet | None) -> dict[str, int]:
        return {
            "level": level,
            "attack": pet.effective_attack,
            "health": pet.effective_health,
            "trigger_attack": trigger_pet.effective_attack if trigger_pet else 0,
            "trigger_health": trigger_pet.effective_health if trigger_pet else 0,
            "trigger_tier": trigger_pet.tier if trigger_pet else 0,
        }

    def _apply_effect(
        self,
        side: int,
        pet: Pet,
        effect: Mapping[str, Any],
        variables: Mapping[str, int],
        trigger_pet: Pet | None,
        position: int | None,
    ) -> list[Pet]:
        op = effect["op"]
        count = int(evaluate(effect.get("count", 1), variables))
        target_ops = {"buff", "give_perk", "damage", "remove_health", "remove_health_percent"}
        if op in target_ops:
            repeat = int(evaluate(effect.get("repeat", 1), variables))
            initial_targets: list[tuple[int, Pet]] | None = None
            for _ in range(repeat):
                if initial_targets is None or effect.get("retarget"):
                    initial_targets = self._targets(
                        str(effect["target"]), side, pet, trigger_pet, count
                    )
                for target_side, target in list(initial_targets):
                    if op == "buff":
                        temporary = bool(
                            effect.get("temporary") or effect.get("temporary_in_battle")
                        )
                        target.buff(
                            int(evaluate(effect.get("attack", 0), variables)),
                            int(evaluate(effect.get("health", 0), variables)),
                            temporary=temporary,
                        )
                    elif op == "give_perk":
                        target.perk = str(effect["perk"])
                    elif op == "damage":
                        amount = int(evaluate(effect["amount"], variables))
                        if trigger_pet and trigger_pet.tier == effect.get("double_if_trigger_tier"):
                            amount *= 2
                        self._damage(target_side, target, amount, source=pet)
                    elif op == "remove_health":
                        target.health -= int(evaluate(effect["amount"], variables))
                    else:
                        percent = float(evaluate(effect["percent"], variables))
                        target.health -= max(1, math.floor(target.effective_health * percent))
            return []
        if op == "gain_health_from_max_friend":
            friends = [friend for friend in self.teams[side] if friend is not pet and friend.alive]
            if friends:
                fraction = float(evaluate(effect["fraction"], variables))
                maximum = max(friend.effective_health for friend in friends)
                pet.health += math.ceil(maximum * fraction)
            return []
        if op in {"summon", "summon_random_tier"}:
            summons: list[Pet] = []
            for _ in range(count):
                if op == "summon_random_tier":
                    token = self._random_tier_pet(
                        int(effect["tier"]),
                        int(evaluate(effect["attack"], variables)),
                        int(evaluate(effect["health"], variables)),
                        int(evaluate(effect.get("level", 1), variables)),
                    )
                else:
                    token = self._token(
                        str(effect["name"]),
                        int(evaluate(effect["attack"], variables)),
                        int(evaluate(effect["health"], variables)),
                        tier=int(effect.get("tier", 1)),
                        level=int(evaluate(effect.get("level", 1), variables)),
                    )
                    token.perk = effect.get("perk")
                if token is not None:
                    summons.append(token)
            summon_side = 1 - side if effect.get("side") == "enemy" else side
            direct = summon_side != side or effect.get("position") == "faint_position"
            if direct:
                insert_at = (
                    0
                    if effect.get("position") == "front"
                    else min(position or 0, len(self.teams[summon_side]))
                )
                self._insert_summons(summon_side, insert_at, summons)
                return []
            return summons
        if op == "swallow":
            targets = self._targets(str(effect["target"]), side, pet, trigger_pet, count)
            swallowed = pet.metadata.setdefault("swallowed", [])
            for _, target in targets:
                if target in self.teams[side]:
                    released = target.clone()
                    released.experience = self._experience_for_level(
                        int(evaluate(effect["released_level"], variables))
                    )
                    released.perk = None
                    swallowed.append(released)
                    self.teams[side].remove(target)
            return []
        if op == "release_swallowed":
            return list(pet.metadata.pop("swallowed", []))
        if op == "copy_ability":
            targets = self._targets(str(effect["target"]), side, pet, trigger_pet, count)
            if targets:
                copied = targets[0][1]
                pet.metadata["copied_ability_name"] = copied.metadata.get(
                    "copied_ability_name", copied.name
                )
            return []
        raise BattleSimulationError(f"unknown battle effect op: {op}")

    def _targets(
        self,
        selector: str,
        side: int,
        pet: Pet,
        trigger_pet: Pet | None,
        count: int,
    ) -> list[tuple[int, Pet]]:
        enemy = 1 - side
        own = [(side, value) for value in self.teams[side] if value.alive]
        enemies = [(enemy, value) for value in self.teams[enemy] if value.alive]
        index = self._index(side, pet)
        if selector == "self":
            return [(side, pet)]
        if selector == "trigger":
            return [(side, trigger_pet)] if trigger_pet is not None else []
        if selector in {"random_friend", "random_friends"}:
            values = [
                (side, value) for value in self.teams[side] if value is not pet and value.alive
            ]
            return self.rng.sample(values, min(count, len(values)))
        if selector == "all_friends":
            return [(target_side, value) for target_side, value in own if value is not pet]
        if selector == "nearest_ahead":
            values = (
                []
                if index is None
                else [
                    (side, self.teams[side][i])
                    for i in range(index - 1, -1, -1)
                    if self.teams[side][i].alive
                ]
            )
            return values[:count]
        if selector == "nearest_behind":
            values = (
                []
                if index is None
                else [
                    (side, self.teams[side][i])
                    for i in range(index + 1, len(self.teams[side]))
                    if self.teams[side][i].alive
                ]
            )
            return values[:count]
        if selector == "adjacent":
            return (
                []
                if index is None
                else [
                    (side, self.teams[side][i])
                    for i in (index - 1, index + 1)
                    if 0 <= i < len(self.teams[side]) and self.teams[side][i].alive
                ]
            )
        if selector in {"random_enemy", "random_enemies"}:
            return self.rng.sample(enemies, min(count, len(enemies)))
        if selector == "lowest_health_enemy":
            return self._extreme_targets(enemies, min)
        if selector == "highest_health_enemy":
            return self._extreme_targets(enemies, max)
        if selector in {"enemy_front", "front_enemy"}:
            return enemies[:1]
        if selector == "last_enemy":
            return enemies[-1:]
        if selector == "all_enemies":
            return enemies
        if selector == "all_pets":
            return own + enemies
        if selector == "all_pets_except_self":
            return [
                (target_side, value) for target_side, value in own + enemies if value is not pet
            ]
        raise BattleSimulationError(f"unknown battle target selector: {selector}")

    def _extreme_targets(self, values, operation):
        if not values:
            return []
        extreme = operation(value.effective_health for _, value in values)
        tied = [item for item in values if item[1].effective_health == extreme]
        return [self.rng.choice(tied)]

    def _random_tier_pet(self, tier: int, attack: int, health: int, level: int) -> Pet | None:
        if self.catalog is None:
            return self._token(f"Tier {tier} Pet", attack, health, tier=tier, level=level)
        values = [
            spec
            for spec in self.catalog.pack_pets("Turtle", through_tier=tier)
            if spec.tier == tier
        ]
        if not values:
            return None
        pet = self.rng.choice(values).create()
        pet.attack, pet.health = attack, health
        pet.experience = self._experience_for_level(level)
        return pet

    @classmethod
    def _token(cls, name: str, attack: int, health: int, *, tier: int = 1, level: int = 1) -> Pet:
        return Pet(
            -1,
            name,
            tier,
            attack,
            health,
            experience=cls._experience_for_level(level),
        )

    @staticmethod
    def _experience_for_level(level: int) -> int:
        return (0, 2, 5)[max(1, min(3, level)) - 1]

    def _index(self, side: int, pet: Pet) -> int | None:
        try:
            return self.teams[side].index(pet)
        except ValueError:
            return None
