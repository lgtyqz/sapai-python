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
