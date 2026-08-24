from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sapai.sim.catalog import Catalog
from sapai.sim.models import Pet, Team

PACK_MAP = {
    0: "Turtle",
    1: "Puppy",
    2: "Star",
    5: "Golden",
    6: "Unicorn",
    7: "Danger",
}


@dataclass(frozen=True, slots=True)
class BoardSnapshot:
    replay_id: str | None
    side: str
    turn: int
    pack: str
    team: Team
    toy: str | None = None
    toy_level: int = 1
    gold_spent: int = 0
    rolls: int = 0
    summoned: int = 0
    level_three_sold: int = 0
    transformations: int = 0
    version: str = "unknown"


class ReplayParser:
    """Python translation of ``sap-board-query/parse-replays.js``."""

    def __init__(self, catalog: Catalog, toys: dict[int, str] | None = None):
        self.catalog = catalog
        self.toys = toys or {}

    def parse_replay(
        self, replay: dict[str, Any], *, replay_id: str | None = None
    ) -> list[BoardSnapshot]:
        result: list[BoardSnapshot] = []
        for action in replay.get("Actions", []):
            if action.get("Type") != 0 or not action.get("Battle"):
                continue
            raw = action["Battle"]
            battle = json.loads(raw) if isinstance(raw, str) else raw
            result.extend(self.parse_battle(battle, replay_id=replay_id))
        return result

    def parse_battle(
        self, battle: dict[str, Any], *, replay_id: str | None = None
    ) -> list[BoardSnapshot]:
        return [
            self._parse_board(battle.get("UserBoard", {}), replay_id=replay_id, side="player"),
            self._parse_board(
                battle.get("OpponentBoard", {}), replay_id=replay_id, side="opponent"
            ),
        ]

    def _parse_board(
        self, board: dict[str, Any], *, replay_id: str | None, side: str
    ) -> BoardSnapshot:
        pets: list[Pet | None] = [None] * 5
        values = [value for value in board.get("Mins", {}).get("Items", []) if value]
        for fallback_position, raw in enumerate(values):
            position = int(raw.get("Poi", {}).get("x", fallback_position))
            if 0 <= position < 5:
                pets[position] = self._parse_pet(raw)
        # Replay coordinates are back-to-front; the simulator uses front at 0.
        pets.reverse()
        pack_value = board.get("Pack", 0)
        pack = PACK_MAP.get(pack_value, board.get("Deck", {}).get("Title", "Custom"))
        toy = next(
            (
                value
                for value in board.get("Rel", {}).get("Items", [])
                if value and value.get("Enu")
            ),
            None,
        )
        return BoardSnapshot(
            replay_id=replay_id,
            side=side,
            turn=int(board.get("Tur", 1)),
            pack=str(pack),
            team=Team(pets),
            toy=self.toys.get(int(toy["Enu"])) if toy else None,
            toy_level=int(toy.get("Lvl", 1)) if toy else 1,
            gold_spent=int(board.get("GoSp", 0)),
            rolls=int(board.get("Rold", 0)),
            summoned=int(board.get("MiSu", 0)),
            level_three_sold=int(board.get("MSFL", 0)),
            transformations=int(board.get("TrTT", 0)),
            version=str(board.get("Ver", "unknown")),
        )

    def _parse_pet(self, raw: dict[str, Any]) -> Pet:
        pet_id = int(raw.get("Enu", -1))
        spec = self.catalog.pets.get(pet_id)
        attack = raw.get("At", {})
        health = raw.get("Hp", {})
        perk_id = raw.get("Perk")
        pet = Pet(
            id=pet_id,
            name=spec.name if spec else f"Pet #{pet_id}",
            tier=spec.tier if spec else 0,
            attack=int(attack.get("Perm", 0)) + int(attack.get("Temp", 0)),
            health=int(health.get("Perm", 0)) + int(health.get("Temp", 0)),
            experience=int(raw.get("Exp", 0)),
            perk=self.catalog.perks.get(int(perk_id)) if perk_id else None,
            mana=int(raw.get("Mana", 0)),
            triggers_consumed=self._triggers_consumed(raw),
        )
        if pet_id == 182:
            swallowed = raw.get("MiMs", {}).get("Lsts", {}).get("WhiteWhaleAbility", [])
            if swallowed:
                pet.metadata["beluga_swallowed_pet_id"] = swallowed[0].get("Enu")
        return pet

    @staticmethod
    def _triggers_consumed(raw: dict[str, Any]) -> int:
        def candidates(value: Any) -> Iterable[int]:
            if not isinstance(value, dict):
                return []
            result = []
            for key, item in value.items():
                normalized = str(key).lower()
                if not isinstance(item, (int, float)):
                    continue
                if ("trig" in normalized and "consum" in normalized) or normalized in {
                    "trgc",
                    "trgcn",
                    "trc",
                    "trcn",
                    "trco",
                }:
                    result.append(int(item))
            return result

        values = list(candidates(raw)) + list(candidates(raw.get("Pow")))
        for ability in raw.get("Abil", []):
            values.extend(candidates(ability))
        return max(values, default=0)
