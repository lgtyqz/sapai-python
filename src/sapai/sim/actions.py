from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionKind(IntEnum):
    BUY_PET = 0
    BUY_MERGE_PET = 1
    MERGE_BOARD_PET = 2
    BUY_FOOD = 3
    ROLL = 4
    FREEZE_PET = 5
    FREEZE_FOOD = 6
    SELL_PET = 7
    UNFREEZE_PET = 8
    UNFREEZE_FOOD = 9
    END_TURN = 10
    REORDER = 11


ACTION_KIND_EXPLORATION_WEIGHTS = {
    ActionKind.BUY_MERGE_PET: 6.0,
    ActionKind.MERGE_BOARD_PET: 5.0,
    ActionKind.BUY_PET: 4.0,
    ActionKind.BUY_FOOD: 3.0,
    ActionKind.ROLL: 2.0,
    ActionKind.END_TURN: 1.0,
    ActionKind.FREEZE_PET: 0.5,
    ActionKind.FREEZE_FOOD: 0.5,
    ActionKind.SELL_PET: 0.25,
    ActionKind.UNFREEZE_PET: 0.25,
    ActionKind.UNFREEZE_FOOD: 0.25,
    ActionKind.REORDER: 0.25,
}


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    source: int = -1
    target: int = -1
    order: tuple[int, ...] = ()

    @property
    def is_chance(self) -> bool:
        return self.kind is ActionKind.ROLL

    def feature_tuple(self) -> tuple[int, int, int]:
        return int(self.kind), self.source, self.target
