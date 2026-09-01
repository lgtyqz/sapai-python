from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sapai.sim.actions import Action, ActionKind
from sapai.sim.models import RunState, Shop, Team

TEAM_SLOTS = 5
BASE_SHOP_PET_SLOTS = 5
SHOP_FOOD_SLOTS = 2
ENTITY_FEATURES = 12


@dataclass(frozen=True)
class EncodedBatch:
    entity_ids: object
    entity_types: object
    perk_ids: object
    entity_features: object
    entity_mask: object
    action_kinds: object
    action_sources: object
    action_targets: object
    action_orders: object
    action_source_entities: object
    action_target_entities: object
    action_order_entities: object
    action_mask: object

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _stable_text_id(value: str | None, buckets: int) -> int:
    if not value:
        return 0
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = (result ^ byte) * 16777619 & 0xFFFFFFFF
    return 1 + result % (buckets - 1)


def encode_states(
    states: Sequence[RunState],
    legal_actions: Sequence[Sequence[Action]],
    *,
    max_actions: int = 256,
    shop_pet_capacity: int | None = None,
    id_buckets: int = 2048,
    perk_buckets: int = 256,
) -> EncodedBatch:
    """Encode simulator objects into fixed-shape NumPy tensors.

    Pet/food IDs use separate entity-type embeddings, so an overlapping raw ID
    is harmless. Unknown perk names are deterministically hashed.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'ml' extra to encode model batches") from error

    batch = len(states)
    required_shop_pet_slots = max(
        BASE_SHOP_PET_SLOTS,
        max((len(state.shop.pets) for state in states), default=0),
    )
    if shop_pet_capacity is not None and shop_pet_capacity < required_shop_pet_slots:
        raise ValueError(
            f"shop_pet_capacity={shop_pet_capacity} cannot encode "
            f"{required_shop_pet_slots} shop pets"
        )
    shop_pet_slots = shop_pet_capacity or required_shop_pet_slots
    entity_count = TEAM_SLOTS + shop_pet_slots + SHOP_FOOD_SLOTS + 1
    global_entity = entity_count - 1
    food_entity_start = TEAM_SLOTS + shop_pet_slots
    ids = np.zeros((batch, entity_count), dtype=np.int32)
    types = np.zeros((batch, entity_count), dtype=np.int32)
    perks = np.zeros((batch, entity_count), dtype=np.int32)
    features = np.zeros((batch, entity_count, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros((batch, entity_count), dtype=bool)
    kinds = np.zeros((batch, max_actions), dtype=np.int32)
    sources = np.zeros((batch, max_actions), dtype=np.int32)
    targets = np.zeros((batch, max_actions), dtype=np.int32)
    orders = np.zeros((batch, max_actions, 5), dtype=np.int32)
    source_entities = np.full((batch, max_actions), global_entity, dtype=np.int32)
    target_entities = np.full((batch, max_actions), global_entity, dtype=np.int32)
    order_entities = np.full(
        (batch, max_actions, TEAM_SLOTS), global_entity, dtype=np.int32
    )
    action_mask = np.zeros((batch, max_actions), dtype=bool)

    for batch_index, (state, actions) in enumerate(zip(states, legal_actions, strict=True)):
        entity_index = 0
        for position, pet in enumerate(state.team.slots):
            if pet is not None:
                ids[batch_index, entity_index] = 1 + pet.id % (id_buckets - 1)
                types[batch_index, entity_index] = 1
                perks[batch_index, entity_index] = _stable_text_id(pet.perk, perk_buckets)
                features[batch_index, entity_index] = _pet_features(pet, position, on_team=True)
                mask[batch_index, entity_index] = True
            entity_index += 1

        for position in range(shop_pet_slots):
            if position < len(state.shop.pets):
                offer = state.shop.pets[position]
                pet = offer.pet
                ids[batch_index, entity_index] = 1 + pet.id % (id_buckets - 1)
                types[batch_index, entity_index] = 2
                features[batch_index, entity_index] = _pet_features(
                    pet, position, on_team=False, frozen=offer.frozen
                )
                mask[batch_index, entity_index] = True
            entity_index += 1

        for position in range(SHOP_FOOD_SLOTS):
            if position < len(state.shop.foods):
                food = state.shop.foods[position]
                ids[batch_index, entity_index] = 1 + food.id % (id_buckets - 1)
                types[batch_index, entity_index] = 3
                features[batch_index, entity_index, 4] = food.tier / 6.0
                features[batch_index, entity_index, 5] = position / 4.0
                features[batch_index, entity_index, 6] = float(food.frozen)
                features[batch_index, entity_index, 7] = food.cost / 3.0
                features[batch_index, entity_index, 8] = float(food.targets_pet)
                mask[batch_index, entity_index] = True
            entity_index += 1

        types[batch_index, entity_index] = 4
        features[batch_index, entity_index] = [
            state.turn / 20.0,
            state.gold / 20.0,
            state.lives / 5.0,
            state.trophies / 10.0,
            state.tier / 6.0,
            state.shop_attack / 50.0,
            state.shop_health / 50.0,
            state.rolls_this_turn / 20.0,
            state.gold_spent_this_turn / 30.0,
            float(state.awaiting_battle),
            _stable_text_id(state.pack, 32) / 31.0,
            _stable_text_id(state.version, 64) / 63.0,
        ]
        mask[batch_index, entity_index] = True

        if len(actions) > max_actions:
            raise ValueError(f"state has {len(actions)} legal actions; max_actions={max_actions}")
        for action_index, action in enumerate(actions):
            kinds[batch_index, action_index] = int(action.kind) + 1
            sources[batch_index, action_index] = action.source + 2
            targets[batch_index, action_index] = action.target + 2
            for order_index, position in enumerate(action.order[:5]):
                orders[batch_index, action_index, order_index] = position + 1
                order_entities[batch_index, action_index, order_index] = position
            if action.kind in {
                ActionKind.BUY_PET,
                ActionKind.BUY_MERGE_PET,
                ActionKind.FREEZE_PET,
                ActionKind.UNFREEZE_PET,
            }:
                source_entities[batch_index, action_index] = TEAM_SLOTS + action.source
            elif action.kind in {
                ActionKind.BUY_FOOD,
                ActionKind.FREEZE_FOOD,
                ActionKind.UNFREEZE_FOOD,
            }:
                source_entities[batch_index, action_index] = food_entity_start + action.source
            elif action.kind in {ActionKind.SELL_PET, ActionKind.MERGE_BOARD_PET}:
                source_entities[batch_index, action_index] = action.source
            if action.kind in {
                ActionKind.BUY_PET,
                ActionKind.BUY_MERGE_PET,
                ActionKind.BUY_FOOD,
                ActionKind.MERGE_BOARD_PET,
            } and action.target >= 0:
                target_entities[batch_index, action_index] = action.target
            action_mask[batch_index, action_index] = True

    return EncodedBatch(
        entity_ids=ids,
        entity_types=types,
        perk_ids=perks,
        entity_features=features,
        entity_mask=mask,
        action_kinds=kinds,
        action_sources=sources,
        action_targets=targets,
        action_orders=orders,
        action_source_entities=source_entities,
        action_target_entities=target_entities,
        action_order_entities=order_entities,
        action_mask=action_mask,
    )


def encode_teams(
    teams: Sequence[Team],
    *,
    turns: Sequence[int] | None = None,
    packs: Sequence[str] | None = None,
    id_buckets: int = 2048,
    perk_buckets: int = 256,
) -> dict[str, object]:
    """Encode battle teams using the same entity representation as shop states."""

    turns = turns or [1] * len(teams)
    packs = packs or ["Turtle"] * len(teams)
    if not (len(teams) == len(turns) == len(packs)):
        raise ValueError("teams, turns, and packs must have equal length")
    states = [
        RunState(team=team.clone(), shop=Shop(), turn=turn, pack=pack)
        for team, turn, pack in zip(teams, turns, packs, strict=True)
    ]
    encoded = encode_states(
        states,
        [[] for _ in states],
        max_actions=1,
        id_buckets=id_buckets,
        perk_buckets=perk_buckets,
    ).as_dict()
    return {
        key: value
        for key, value in encoded.items()
        if key in {"entity_ids", "entity_types", "perk_ids", "entity_features", "entity_mask"}
    }


def _pet_features(pet, position: int, *, on_team: bool, frozen: bool = False):
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("install the 'ml' extra") from error
    values = np.zeros(ENTITY_FEATURES, dtype=np.float32)
    values[:] = [
        pet.attack / 50.0,
        pet.health / 50.0,
        pet.temporary_attack / 50.0,
        pet.temporary_health / 50.0,
        pet.tier / 6.0,
        position / 4.0,
        float(frozen),
        pet.experience / 5.0,
        pet.level / 3.0,
        pet.mana / 10.0,
        pet.triggers_consumed / 10.0,
        float(on_team),
    ]
    return values
