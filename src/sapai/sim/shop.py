from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import permutations

from sapai.sim.actions import Action, ActionKind
from sapai.sim.catalog import Catalog
from sapai.sim.models import BattleOutcome, Pet, RunState, Shop, ShopPet, Team
from sapai.sim.shop_abilities import ShopAbilityEngine


class IllegalAction(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StepResult:
    state: RunState
    reward: float = 0.0
    done: bool = False


class ShopEnvironment:
    """Exact-state shop environment with injected randomness.

    Unlike the original TypeScript environment, ending a turn never queries a
    database or runs a battle implicitly. It creates an ``awaiting_battle``
    state. The caller chooses an opponent and later calls :meth:`apply_outcome`.
    This makes transitions reproducible and suitable for MCTS.
    """

    SLOTH_ROLL_DENOMINATOR = 10_000

    def __init__(self, catalog: Catalog, abilities: ShopAbilityEngine | None = None):
        self.catalog = catalog
        self.abilities = abilities or ShopAbilityEngine(catalog)

    @staticmethod
    def pet_slots(tier: int) -> int:
        return min(5, 3 + (tier - 1) // 2)

    @staticmethod
    def food_slots(tier: int) -> int:
        return 1 if tier < 3 else 2

    def reset(self, *, pack: str = "Turtle", seed: int | None = None) -> RunState:
        state = RunState(pack=pack)
        rng = random.Random(seed)
        state.shop = self.roll_shop(state, rng)
        return state

    def roll_shop(self, state: RunState, rng: random.Random) -> Shop:
        frozen_pets = [offer.clone() for offer in state.shop.pets if offer.frozen]
        frozen_foods = [food.clone() for food in state.shop.foods if food.frozen]
        # A roll is the boundary between freeze decisions. Within one shop,
        # toggling an offer twice only returns to an already-seen state.
        for offer in frozen_pets:
            offer.freeze_toggled = False
        for food in frozen_foods:
            food.freeze_toggled = False
        pet_pool = self.catalog.pack_pets(state.pack, through_tier=state.tier)
        food_pool = self.catalog.pack_foods(state.pack, through_tier=state.tier)
        if not pet_pool or not food_pool:
            raise ValueError(f"catalog contains no rollable entries for pack {state.pack!r}")

        pets = frozen_pets
        open_pet_slots = max(0, self.pet_slots(state.tier) - len(frozen_pets))
        rolled_sloth = open_pet_slots > 0 and rng.randrange(self.SLOTH_ROLL_DENOMINATOR) == 0
        for slot in range(open_pet_slots):
            spec = (
                self.catalog.pet_by_name("Sloth")
                if rolled_sloth and slot == 0
                else rng.choice(pet_pool)
            )
            pet = spec.create(instance_id=state.allocate_instance_id())
            pet.buff(state.shop_attack, state.shop_health)
            pets.append(ShopPet(pet))
        self._sort_shop_pets(pets)

        foods = frozen_foods
        for _ in range(max(0, self.food_slots(state.tier) - len(frozen_foods))):
            foods.append(rng.choice(food_pool).create())
        return Shop(pets=pets, foods=foods)

    def legal_actions(self, state: RunState) -> list[Action]:
        if state.terminal or state.awaiting_battle:
            return []
        actions: list[Action] = []
        empty_slots = [index for index, pet in enumerate(state.team.slots) if pet is None]
        occupied = state.team.occupied_indices()

        for shop_index, offer in enumerate(state.shop.pets):
            if state.gold >= 3:
                actions.extend(
                    Action(ActionKind.BUY_PET, shop_index, target) for target in empty_slots
                )
                actions.extend(
                    Action(ActionKind.BUY_MERGE_PET, shop_index, target)
                    for target in occupied
                    if state.team.slots[target].id == offer.pet.id  # type: ignore[union-attr]
                    and state.team.slots[target].level < 3  # type: ignore[union-attr]
                )
            if not offer.freeze_toggled:
                actions.append(
                    Action(
                        ActionKind.UNFREEZE_PET if offer.frozen else ActionKind.FREEZE_PET,
                        shop_index,
                    )
                )

        for food_index, food in enumerate(state.shop.foods):
            if state.gold >= food.cost:
                targets = occupied if food.targets_pet else [-1]
                actions.extend(
                    Action(ActionKind.BUY_FOOD, food_index, target) for target in targets
                )
            if not food.freeze_toggled:
                actions.append(
                    Action(
                        ActionKind.UNFREEZE_FOOD if food.frozen else ActionKind.FREEZE_FOOD,
                        food_index,
                    )
                )

        for source in occupied:
            actions.append(Action(ActionKind.SELL_PET, source))
            source_pet = state.team.slots[source]
            if source_pet.level < 3:  # type: ignore[union-attr]
                actions.extend(
                    Action(ActionKind.MERGE_BOARD_PET, source, target)
                    for target in occupied
                    if target != source
                    and state.team.slots[target].id == source_pet.id  # type: ignore[union-attr]
                    and state.team.slots[target].level < 3  # type: ignore[union-attr]
                )

        current_order = tuple(occupied)
        for order in permutations(occupied):
            if order != current_order:
                actions.append(Action(ActionKind.REORDER, order=order))

        if state.gold >= 1:
            actions.append(Action(ActionKind.ROLL))
        actions.append(Action(ActionKind.END_TURN))
        return actions

    def step(self, state: RunState, action: Action, rng: random.Random) -> StepResult:
        if action not in self.legal_actions(state):
            raise IllegalAction(f"illegal action for current state: {action}")
        new = state.clone()
        kind = action.kind

        if kind is ActionKind.BUY_PET:
            offer = new.shop.pets[action.source]
            new.team.slots[action.target] = offer.pet
            self._remove_pet_offer(new, action.source)
            self._spend(new, 3)
            self.abilities.on_summoned(new, action.target, rng)
            self.abilities.on_buy(new, action.target, rng)
        elif kind is ActionKind.BUY_MERGE_PET:
            offer = new.shop.pets[action.source]
            target = new.team.slots[action.target]
            self._merge_pet_stats(target, offer.pet)  # type: ignore[arg-type]
            self._remove_pet_offer(new, action.source)
            self._spend(new, 3)
            self._give_experience(new, action.target, 1, rng)
            self.abilities.on_buy(new, action.target, rng)
        elif kind is ActionKind.MERGE_BOARD_PET:
            source = new.team.slots[action.source]
            target = new.team.slots[action.target]
            self._merge_pet_stats(target, source)  # type: ignore[arg-type]
            amount = source.experience + 1  # type: ignore[union-attr]
            new.team.slots[action.source] = None
            self._give_experience(new, action.target, amount, rng)
        elif kind is ActionKind.BUY_FOOD:
            food = new.shop.foods.pop(action.source)
            self._spend(new, food.cost)
            self.abilities.apply_food(
                new,
                food,
                action.target,
                rng,
                faint_callback=lambda index: self._shop_faint(new, index, rng),
                level_callback=lambda index, amount: self._give_experience(new, index, amount, rng),
            )
        elif kind is ActionKind.ROLL:
            self._spend(new, 1)
            new.rolls_this_turn += 1
            new.shop = self.roll_shop(new, rng)
        elif kind in {ActionKind.FREEZE_PET, ActionKind.UNFREEZE_PET}:
            new.shop.pets[action.source].frozen = kind is ActionKind.FREEZE_PET
            new.shop.pets[action.source].freeze_toggled = True
        elif kind in {ActionKind.FREEZE_FOOD, ActionKind.UNFREEZE_FOOD}:
            new.shop.foods[action.source].frozen = kind is ActionKind.FREEZE_FOOD
            new.shop.foods[action.source].freeze_toggled = True
        elif kind is ActionKind.SELL_PET:
            pet = new.team.slots[action.source]
            # The base sale payout happens before the pet's sell ability. In
            # particular, a level-one Pig receives one base gold and then one
            # additional gold from its ability.
            new.gold += pet.level + int(  # type: ignore[union-attr]
                pet.metadata.get("sell_value_bonus", 0)  # type: ignore[union-attr]
            )
            self.abilities.on_sell(new, action.source, rng)
            new.team.slots[action.source] = None
        elif kind is ActionKind.REORDER:
            pets = [new.team.slots[index] for index in action.order]
            new.team = Team.from_pets(pet for pet in pets if pet is not None)
        elif kind is ActionKind.END_TURN:
            self.abilities.end_turn(new, rng)
            new.awaiting_battle = True
        else:  # pragma: no cover - exhaustive enum guard
            raise AssertionError(kind)
        return StepResult(new, done=new.terminal)

    def apply_outcome(
        self,
        state: RunState,
        outcome: BattleOutcome | str,
        rng: random.Random,
    ) -> StepResult:
        if not state.awaiting_battle:
            raise ValueError("state is not awaiting a battle outcome")
        new = state.clone()
        outcome = BattleOutcome(outcome)
        if outcome is BattleOutcome.WIN:
            new.trophies += 1
        elif outcome is BattleOutcome.LOSS:
            new.lives -= 1
        new.metadata["last_outcome"] = outcome.value

        for pet in new.team.living():
            pet.clear_battle_state()
        new.awaiting_battle = False
        if not new.terminal:
            new.turn += 1
            if new.turn == 3 and outcome is BattleOutcome.LOSS:
                new.lives += 1
            new.gold = 10
            new.gold_spent_this_turn = 0
            new.rolls_this_turn = 0
            new.shop = self.roll_shop(new, rng)
            self.abilities.start_turn(new, rng)
        return StepResult(new, reward=1.0 if new.trophies >= 10 else 0.0, done=new.terminal)

    def _spend(self, state: RunState, amount: int) -> None:
        state.gold -= amount
        state.gold_spent_this_turn += amount

    def _remove_pet_offer(self, state: RunState, index: int) -> None:
        group = state.shop.pets[index].reward_group
        if group is None:
            state.shop.pets.pop(index)
        else:
            state.shop.pets[:] = [offer for offer in state.shop.pets if offer.reward_group != group]

    @staticmethod
    def _merge_pet_stats(target: Pet, source: Pet) -> None:
        """Merge base stats while preserving temporary bonuses as bonuses."""

        target.attack = max(target.attack, source.attack)
        target.health = max(target.health, source.health)
        target.temporary_attack = max(target.temporary_attack, source.temporary_attack)
        target.temporary_health = max(target.temporary_health, source.temporary_health)

    @staticmethod
    def _sort_shop_pets(pets: list[ShopPet]) -> None:
        # Same-tier offers retain their roll order. Sloth is the sole tier-order
        # exception and is always displayed at the far left when it appears.
        pets.sort(
            key=lambda offer: (offer.pet.name == "Sloth", offer.pet.tier),
            reverse=True,
        )

    def _give_experience(
        self, state: RunState, index: int, amount: int, rng: random.Random
    ) -> None:
        pet = state.team.slots[index]
        if pet is None:
            raise ValueError("cannot give experience to an empty slot")
        old_level = pet.level
        old_exp = pet.experience
        pet.experience = min(5, pet.experience + amount)
        gained = pet.experience - old_exp
        pet.buff(gained, gained)
        if pet.level > old_level:
            self.abilities.on_level_up(state, index, old_level, rng)
            for _ in range(pet.level - old_level):
                self._add_tier_up_choices(state, rng)

    def _add_tier_up_choices(self, state: RunState, rng: random.Random) -> None:
        reward_tier = min(6, state.tier + 1)
        pool = [
            spec
            for spec in self.catalog.pack_pets(state.pack, through_tier=reward_tier)
            if spec.tier == reward_tier
        ]
        if not pool:
            return
        group = state.next_reward_group
        state.next_reward_group += 1
        choices = rng.sample(pool, min(2, len(pool)))
        offers = []
        for spec in choices:
            pet = spec.create(instance_id=state.allocate_instance_id())
            pet.buff(state.shop_attack, state.shop_health)
            offers.append(ShopPet(pet, reward_group=group))
        state.shop.pets.extend(offers)
        self._sort_shop_pets(state.shop.pets)

    def _shop_faint(self, state: RunState, index: int, rng: random.Random) -> None:
        self._process_shop_faint(state, index, rng)
        processed = 1
        while True:
            candidates = [
                (pet.effective_attack, rng.random(), pet)
                for pet in state.team.slots
                if pet is not None and not pet.alive
            ]
            if not candidates:
                return
            processed += 1
            if processed > 100:
                raise RuntimeError("shop faint event limit exceeded")
            _, _, pet = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
            self._process_shop_faint(state, state.team.slots.index(pet), rng)

    def _process_shop_faint(self, state: RunState, index: int, rng: random.Random) -> None:
        fainted = state.team.slots[index]
        if fainted is None:
            return
        position = sum(pet is not None for pet in state.team.slots[:index])
        summons = self.abilities.on_shop_faint(state, index, rng)
        summons.extend(self.abilities.on_perk_faint(state, fainted))
        state.team.slots[index] = None
        self._insert_shop_summons(state, position, summons, rng)
        friend_summons = self.abilities.on_friend_fainted(state, fainted, position, rng)
        self._insert_shop_summons(state, position, friend_summons, rng)

    def _insert_shop_summons(
        self,
        state: RunState,
        position: int,
        summons: list[Pet],
        rng: random.Random,
    ) -> None:
        if not summons:
            return
        pets = [pet for pet in state.team.slots if pet is not None]
        room = max(0, 5 - len(pets))
        accepted = summons[:room]
        insertion = min(position, len(pets))
        pets[insertion:insertion] = accepted
        state.team = Team.from_pets(pets)
        for pet in accepted:
            self.abilities.on_summoned(state, state.team.slots.index(pet), rng)
