"""Render battle and Arena timelines with a shared offline asset bundle."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from sapai.sim.actions import Action
from sapai.sim.battle import BattleFrame, BattleResult
from sapai.sim.models import Food, Pet, RunState, Team
from sapai.training.arena import ArenaRunResult
from sapai.visualization.assets import SpriteAtlas

STYLESHEET_NAME = "sapai.css"
SCRIPT_NAME = "sapai.js"
TEMPLATE_NAME = "timeline.html"


def _pet(pet: Pet | None, *, position: int | None = None) -> dict[str, Any] | None:
    if pet is None:
        return None
    value = {
        "name": pet.name,
        "attack": pet.effective_attack,
        "health": pet.effective_health,
        "level": pet.level,
        "experience": pet.experience,
        "perk": pet.perk,
    }
    visual_id = pet.metadata.get("battle_visual_id")
    if visual_id is not None:
        value["visualId"] = int(visual_id)
    if position is not None:
        value["position"] = position
    return value


def _team(team: Team) -> list[dict[str, Any] | None]:
    return [_pet(pet, position=position) for position, pet in enumerate(team.slots)]


def _food(food: Food) -> dict[str, Any]:
    return {"name": food.name, "cost": food.cost, "frozen": food.frozen}


def _state(state: RunState) -> dict[str, Any]:
    return {
        "turn": state.turn,
        "gold": state.gold,
        "lives": state.lives,
        "trophies": state.trophies,
        "tier": state.tier,
        "team": _team(state.team),
        "shopPets": [
            {**_pet(offer.pet), "frozen": offer.frozen}  # type: ignore[arg-type]
            for offer in state.shop.pets
        ],
        "shopFoods": [_food(food) for food in state.shop.foods],
    }


def _frame_entities(frame: BattleFrame | None) -> dict[int, tuple[str, int, Pet]]:
    if frame is None:
        return {}
    result: dict[int, tuple[str, int, Pet]] = {}
    for side, team in (("player", frame.player), ("opponent", frame.opponent)):
        for position, pet in enumerate(team.slots):
            if pet is None:
                continue
            visual_id = pet.metadata.get("battle_visual_id")
            if visual_id is not None:
                result[int(visual_id)] = (side, position, pet)
    return result


def _animated_team(
    team: Team,
    frame: BattleFrame,
    previous_entities: dict[int, tuple[str, int, Pet]],
    *,
    has_previous: bool,
) -> list[dict[str, Any] | None]:
    values: list[dict[str, Any] | None] = []
    for position, pet in enumerate(team.slots):
        value = _pet(pet, position=position)
        if pet is None or value is None:
            values.append(None)
            continue
        visual_id = value.get("visualId")
        previous = previous_entities.get(visual_id) if visual_id is not None else None
        previous_pet = previous[2] if previous else None
        value["animation"] = {
            "entered": has_previous and previous_pet is None,
            "attackDelta": (
                pet.effective_attack - previous_pet.effective_attack if previous_pet else 0
            ),
            "healthDelta": (
                pet.effective_health - previous_pet.effective_health if previous_pet else 0
            ),
            "perkChanged": previous_pet is not None and pet.perk != previous_pet.perk,
            "previousPerk": previous_pet.perk if previous_pet else None,
            "role": (
                "interactor"
                if frame.event == "clash" and visual_id in frame.participant_ids
                else None
            ),
        }
        values.append(value)
    return values


def _battle_slide(
    frame: BattleFrame,
    result: BattleResult,
    *,
    previous: BattleFrame | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    previous_entities = _frame_entities(previous)
    current_entities = _frame_entities(frame)
    departed = []
    if previous is not None:
        for visual_id, (side, position, pet) in previous_entities.items():
            if visual_id not in current_entities:
                departed.append(
                    {
                        "side": side,
                        "position": position,
                        "pet": _pet(pet, position=position),
                    }
                )
    return {
        "type": "battle",
        "event": frame.event,
        "label": f"{prefix}{frame.label}",
        "player": _animated_team(
            frame.player,
            frame,
            previous_entities,
            has_previous=previous is not None,
        ),
        "opponent": _animated_team(
            frame.opponent,
            frame,
            previous_entities,
            has_previous=previous is not None,
        ),
        "departed": departed,
        "log": result.log[: frame.log_index],
        "outcome": result.outcome.value,
    }


def _action(action: Action | None) -> str | None:
    if action is None:
        return None
    details = []
    if action.source >= 0:
        details.append(f"source {action.source + 1}")
    if action.target >= 0:
        details.append(f"target {action.target + 1}")
    if action.order:
        details.append("order " + "→".join(str(value + 1) for value in action.order))
    suffix = f" ({', '.join(details)})" if details else ""
    return action.kind.name.replace("_", " ").title() + suffix


def render_battle_html(
    result: BattleResult,
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    frames = result.frames or [
        BattleFrame("Battle result", result.player, result.opponent, len(result.log))
    ]
    slides = [
        _battle_slide(
            frame,
            result,
            previous=frames[index - 1] if index else None,
        )
        for index, frame in enumerate(frames)
    ]
    payload = {
        "title": "Super Auto Pets battle",
        "subtitle": f"{result.outcome.value.replace('_', ' ').title()} · {result.rounds} rounds",
        "slides": slides,
    }
    return _write_html(payload, output_path, assets_root)


def render_arena_html(
    run: ArenaRunResult,
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    slides: list[dict[str, Any]] = []
    for arena_turn in run.turns:
        for frame in arena_turn.shop_frames:
            slides.append(
                {
                    "type": "shop",
                    "label": frame.label,
                    "action": _action(frame.action),
                    "state": _state(frame.state),
                }
            )
        if arena_turn.battle:
            frames = arena_turn.battle.frames
            for index, frame in enumerate(frames):
                slides.append(
                    _battle_slide(
                        frame,
                        arena_turn.battle,
                        previous=frames[index - 1] if index else None,
                        prefix=f"Turn {arena_turn.turn} · ",
                    )
                )
    final = run.final_state
    payload = {
        "title": "Super Auto Pets Arena run",
        "subtitle": (
            f"{final.trophies} trophies · {final.lives} lives · "
            f"{len(run.turns)} battles"
        ),
        "slides": slides,
    }
    return _write_html(payload, output_path, assets_root)


def _collect_names(slides: list[dict[str, Any]]) -> dict[str, set[str]]:
    names = {"pet": set(), "food": set(), "toy": set()}
    for slide in slides:
        if slide["type"] == "battle":
            pets = slide["player"] + slide["opponent"]
            names["pet"].update(pet["name"] for pet in pets if pet)
            names["pet"].update(
                item["pet"]["name"] for item in slide["departed"] if item["pet"]
            )
        else:
            state = slide["state"]
            names["pet"].update(pet["name"] for pet in state["team"] if pet)
            names["pet"].update(pet["name"] for pet in state["shopPets"])
            names["food"].update(food["name"] for food in state["shopFoods"])
    return names


def _write_shared_file(destination: Path, name: str) -> Path:
    source = files("sapai.visualization").joinpath("static", name)
    output = destination.parent / name
    content = source.read_bytes()
    if not output.exists() or output.read_bytes() != content:
        output.write_bytes(content)
    return output


def _write_html(
    payload: dict[str, Any],
    output_path: str | Path,
    assets_root: str | Path,
) -> Path:
    atlas = SpriteAtlas(assets_root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["sprites"] = atlas.export_payload(
        _collect_names(payload["slides"]), destination.parent
    )
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    _write_shared_file(destination, STYLESHEET_NAME)
    _write_shared_file(destination, SCRIPT_NAME)
    template = files("sapai.visualization").joinpath("static", TEMPLATE_NAME).read_text(
        encoding="utf-8"
    )
    document = (
        template.replace("__SAPAI_STYLESHEET__", STYLESHEET_NAME)
        .replace("__SAPAI_SCRIPT__", SCRIPT_NAME)
        .replace("__SAPAI_PAYLOAD__", encoded)
    )
    destination.write_text(document, encoding="utf-8")
    return destination
