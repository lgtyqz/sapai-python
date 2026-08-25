"""Interactive human Arena UI for trusted Google Colab output frames."""

from __future__ import annotations

import json
import uuid
from importlib.resources import files
from pathlib import Path
from typing import Any

from sapai.data.serialization import run_state_from_dict, team_from_dict
from sapai.sim.battle import BattleFrame, BattleResult, BattleResultKind
from sapai.training.human import HumanArenaSession
from sapai.visualization.assets import SpriteAtlas
from sapai.visualization.html import _battle_slide, _collect_names, _state

HUMAN_TEMPLATE_NAME = "human_arena.html"
HUMAN_SCRIPT_NAME = "human_arena.js"
HUMAN_STYLESHEET_NAME = "human_arena.css"


def human_arena_payload(
    session: HumanArenaSession,
    assets_root: str | Path,
) -> dict[str, Any]:
    """Build the JSON payload consumed by the inline Colab interface."""

    snapshot = session.snapshot()
    state = run_state_from_dict(snapshot["state"])
    payload = {key: value for key, value in snapshot.items() if key not in {"state", "battle"}}
    payload["state"] = _state(state)
    slides: list[dict[str, Any]] = []
    raw_battle = snapshot.get("battle")
    if raw_battle:
        result = _battle_result_from_dict(raw_battle["result"])
        slides = [
            _battle_slide(
                frame,
                result,
                previous=result.frames[index - 1] if index else None,
                prefix=f"Turn {raw_battle['turn']} · ",
            )
            for index, frame in enumerate(result.frames)
        ]
        payload["battle"] = {
            "turn": raw_battle["turn"],
            "opponent": raw_battle["opponent"],
            "outcome": result.outcome.value,
            "rounds": result.rounds,
            "slides": slides,
        }
    else:
        payload["battle"] = None

    collection_slides = list(slides)
    if not collection_slides:
        collection_slides.append({"type": "shop", "state": payload["state"]})
    payload["sprites"] = SpriteAtlas(assets_root).payload(_collect_names(collection_slides))
    return payload


def build_human_arena_html(
    session: HumanArenaSession,
    assets_root: str | Path,
    *,
    callback_name: str,
) -> str:
    """Return a self-contained trusted-output document fragment for Colab."""

    resources = files("sapai.visualization").joinpath("static")
    template = resources.joinpath(HUMAN_TEMPLATE_NAME).read_text(encoding="utf-8")
    base_css = resources.joinpath("sapai.css").read_text(encoding="utf-8")
    human_css = resources.joinpath(HUMAN_STYLESHEET_NAME).read_text(encoding="utf-8")
    script = resources.joinpath(HUMAN_SCRIPT_NAME).read_text(encoding="utf-8")
    payload = {
        "callbackName": callback_name,
        "view": human_arena_payload(session, assets_root),
    }
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    root_id = f"sapai-human-{uuid.uuid4().hex}"
    return (
        template.replace("__SAPAI_ROOT_ID__", root_id)
        .replace("__SAPAI_STYLES__", base_css + "\n" + human_css)
        .replace("__SAPAI_PAYLOAD__", encoded)
        .replace("__SAPAI_SCRIPT__", script)
    )


def display_human_arena(
    session: HumanArenaSession,
    assets_root: str | Path,
) -> str:
    """Register a Colab callback and display the live benchmark interface."""

    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ModuleNotFoundError as error:  # pragma: no cover - Colab-only adapter
        raise RuntimeError("display_human_arena must run inside Google Colab") from error

    callback_name = f"sapai.human_arena.{uuid.uuid4().hex}"

    def callback(command: str, arguments: dict[str, Any] | None = None):
        values = arguments or {}
        try:
            if command == "action":
                session.apply_action(
                    str(values["action_id"]),
                    expected_revision=int(values["revision"]),
                    elapsed_ms=float(values["elapsed_ms"]),
                )
            elif command == "continue":
                session.continue_battle(expected_revision=int(values["revision"]))
            elif command == "new_episode":
                session.new_episode(expected_revision=int(values["revision"]))
            elif command != "refresh":
                raise ValueError(f"unknown human Arena command: {command!r}")
            response = human_arena_payload(session, assets_root)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            response = human_arena_payload(session, assets_root)
            response["error"] = str(error)
        return JSON(response)

    output.register_callback(callback_name, callback)
    display(HTML(build_human_arena_html(session, assets_root, callback_name=callback_name)))
    return callback_name


def _battle_result_from_dict(value: dict[str, Any]) -> BattleResult:
    frames = [
        BattleFrame(
            label=str(frame["label"]),
            player=team_from_dict(frame["player"]),
            opponent=team_from_dict(frame["opponent"]),
            log_index=int(frame["log_index"]),
            event=str(frame.get("event", "state")),
            actor_id=(int(frame["actor_id"]) if frame.get("actor_id") is not None else None),
            target_id=(int(frame["target_id"]) if frame.get("target_id") is not None else None),
        )
        for frame in value.get("frames", [])
    ]
    return BattleResult(
        outcome=BattleResultKind(value["outcome"]),
        rounds=int(value["rounds"]),
        player=team_from_dict(value["player"]),
        opponent=team_from_dict(value["opponent"]),
        log=[str(line) for line in value.get("log", [])],
        frames=frames,
    )
