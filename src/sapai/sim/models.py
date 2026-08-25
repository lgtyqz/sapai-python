from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

TEAM_SIZE = 5


class BattleOutcome(str, Enum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"


@dataclass(frozen=True, slots=True)
class PetSpec:
    id: int
    name: str
    tier: int
    attack: int
    health: int
    packs: tuple[str, ...] = ()
    ability_text: tuple[str, ...] = ()

    def create(self, *, instance_id: int = 0) -> Pet:
        return Pet(
            id=self.id,
            name=self.name,
            tier=self.tier,
            attack=self.attack,
            health=self.health,
            instance_id=instance_id,
        )


@dataclass(frozen=True, slots=True)
class FoodSpec:
    id: int
    name: str
    tier: int
    packs: tuple[str, ...] = ()
    cost: int = 3
    targets_pet: bool = True
    ability_text: str = ""

    def create(self) -> Food:
        return Food(
            id=self.id,
            name=self.name,
            tier=self.tier,
            cost=self.cost,
            targets_pet=self.targets_pet,
        )


@dataclass(slots=True)
class Pet:
    """A pet instance.

    Team position 0 is always the front. ``attack`` and ``health`` are the
    persistent values. Temporary values are included only while resolving a
    battle and are cleared by :meth:`clear_battle_state`.
    """

    id: int
    name: str
    tier: int
    attack: int
    health: int
    experience: int = 0
    perk: str | None = None
    mana: int = 0
    temporary_attack: int = 0
    temporary_health: int = 0
    triggers_consumed: int = 0
    instance_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> int:
        if self.experience >= 5:
            return 3
        if self.experience >= 2:
            return 2
        return 1

    @property
    def effective_attack(self) -> int:
        return max(0, self.attack + self.temporary_attack)

    @property
    def effective_health(self) -> int:
        return self.health + self.temporary_health

    @property
    def alive(self) -> bool:
        return self.effective_health > 0

    def buff(self, attack: int = 0, health: int = 0, *, temporary: bool = False) -> None:
        if temporary:
            self.temporary_attack += attack
            self.temporary_health += health
        else:
            self.attack += attack
            self.health += health

    def clear_battle_state(self) -> None:
        self.temporary_attack = 0
        self.temporary_health = 0
        self.triggers_consumed = 0
        self.metadata.pop("battle", None)

    def clone(self) -> Pet:
        return copy.deepcopy(self)


@dataclass(slots=True)
class Food:
    id: int
    name: str
    tier: int
    cost: int = 3
    targets_pet: bool = True
    frozen: bool = False
    reward_group: int | None = None
    freeze_toggled: bool = False

    def clone(self) -> Food:
        return copy.deepcopy(self)


@dataclass(slots=True)
class ShopPet:
    pet: Pet
    frozen: bool = False
    reward_group: int | None = None
    freeze_toggled: bool = False

    def clone(self) -> ShopPet:
        return copy.deepcopy(self)


@dataclass(slots=True)
class Shop:
    pets: list[ShopPet] = field(default_factory=list)
    foods: list[Food] = field(default_factory=list)

    def clone(self) -> Shop:
        return copy.deepcopy(self)


@dataclass(slots=True)
class Team:
    slots: list[Pet | None] = field(default_factory=lambda: [None] * TEAM_SIZE)

    def __post_init__(self) -> None:
        if len(self.slots) != TEAM_SIZE:
            raise ValueError(f"a team must contain exactly {TEAM_SIZE} slots")

    def living(self) -> list[Pet]:
        return [pet for pet in self.slots if pet is not None and pet.alive]

    def occupied_indices(self) -> list[int]:
        return [index for index, pet in enumerate(self.slots) if pet is not None]

    def first_empty(self) -> int | None:
        return next((index for index, pet in enumerate(self.slots) if pet is None), None)

    def compact(self) -> None:
        pets = [pet for pet in self.slots if pet is not None]
        self.slots[:] = pets + [None] * (TEAM_SIZE - len(pets))

    def clone(self) -> Team:
        return copy.deepcopy(self)

    @classmethod
    def from_pets(cls, pets: Iterable[Pet]) -> Team:
        values = list(pets)
        if len(values) > TEAM_SIZE:
            raise ValueError("a team cannot have more than five pets")
        return cls(values + [None] * (TEAM_SIZE - len(values)))


@dataclass(slots=True)
class RunState:
    team: Team = field(default_factory=Team)
    shop: Shop = field(default_factory=Shop)
    lives: int = 5
    turn: int = 1
    gold: int = 10
    trophies: int = 0
    shop_attack: int = 0
    shop_health: int = 0
    pack: str = "Turtle"
    version: str = "current"
    awaiting_battle: bool = False
    next_instance_id: int = 1
    next_reward_group: int = 1
    rolls_this_turn: int = 0
    gold_spent_this_turn: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> int:
        return min(6, (self.turn + 1) // 2)

    @property
    def terminal(self) -> bool:
        return self.lives <= 0 or self.trophies >= 10

    def clone(self) -> RunState:
        return copy.deepcopy(self)

    def allocate_instance_id(self) -> int:
        value = self.next_instance_id
        self.next_instance_id += 1
        return value

    def canonical_key(self) -> str:
        """Stable state hash input used by search transposition tables."""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
