"""Core-ipywidgets adapter for interactive human Arena sessions."""

from __future__ import annotations

import html
import json
import time
import uuid
from pathlib import Path
from typing import Any

from sapai.training.human import HumanArenaSession
from sapai.visualization.colab import human_arena_command, human_arena_payload


def _experience_label(pet: dict[str, Any]) -> str:
    experience = max(0, int(pet.get("experience", 0)))
    if experience >= 5:
        return f"Level 3 · XP {experience}/5 (max)"
    next_level = 5 if experience >= 2 else 2
    return f"Level {pet.get('level', 1)} · XP {experience}/{next_level}"


def _pet_html(
    pet: dict[str, Any] | None,
    sprite: str,
    *,
    label: str = "",
) -> str:
    if pet is None:
        return (
            '<div style="height:172px;display:grid;place-items:center;border:2px dashed #b9b7ae;'
            'border-radius:14px;color:#6a6962;background:#fffdf7">Empty slot</div>'
        )
    name = html.escape(str(pet.get("name", "Unknown")))
    image = (
        f'<img src="{html.escape(sprite, quote=True)}" alt="{name}" '
        'style="width:82px;height:82px;object-fit:contain">'
        if sprite
        else '<div style="height:82px"></div>'
    )
    perk = html.escape(str(pet.get("perk") or "No perk"))
    frozen = " · ❄ frozen" if pet.get("frozen") else ""
    return (
        '<div style="min-height:172px;padding:8px;border:2px solid #d9d5c7;border-radius:14px;'
        'text-align:center;background:#fffdf7;box-sizing:border-box">'
        f'<div style="font-size:12px;color:#68665f">{html.escape(label)}</div>{image}'
        f'<div style="font-weight:800">{name}{frozen}</div>'
        f"<div>⚔ {max(0, int(pet.get('attack', 0)))} · "
        f"♥ {max(0, int(pet.get('health', 0)))}</div>"
        f'<div title="Total experience" style="font-size:12px">{_experience_label(pet)}</div>'
        f'<div style="font-size:12px;color:#68665f">{perk}</div></div>'
    )


def _food_html(food: dict[str, Any], sprite: str) -> str:
    name = html.escape(str(food.get("name", "Unknown")))
    image = (
        f'<img src="{html.escape(sprite, quote=True)}" alt="{name}" '
        'style="width:82px;height:82px;object-fit:contain">'
        if sprite
        else '<div style="height:82px"></div>'
    )
    frozen = " · ❄ frozen" if food.get("frozen") else ""
    return (
        '<div style="min-height:142px;padding:8px;border:2px solid #d9d5c7;border-radius:14px;'
        'text-align:center;background:#fffdf7;box-sizing:border-box">'
        f'{image}<div style="font-weight:800">{name}{frozen}</div>'
        f"<div>{int(food.get('cost', 0))} gold</div></div>"
    )


def _action_name(action: dict[str, Any], state: dict[str, Any]) -> str:
    target_index = int(action.get("target", -1))
    target_pet = state["team"][target_index] if 0 <= target_index < len(state["team"]) else None
    target = (
        f"{target_pet['name']} in slot {target_index + 1}"
        if target_pet
        else f"slot {target_index + 1}"
    )
    kind = str(action["kind"])
    if kind == "BUY_PET":
        return f"Buy into {target}"
    if kind == "BUY_MERGE_PET":
        return f"Buy and merge with {target}"
    if kind == "BUY_FOOD":
        return "Buy food" if target_index < 0 else f"Feed {target}"
    if kind == "MERGE_BOARD_PET":
        return f"Merge into {target}"
    return kind.replace("_", " ").lower().capitalize()


class _HumanArenaCoreWidget:
    """Render the benchmark entirely with widget models bundled by Kaggle."""

    def __init__(
        self,
        session: HumanArenaSession,
        assets_root: str | Path,
        widgets: Any,
    ) -> None:
        self.session = session
        self.assets_root = assets_root
        self.widgets = widgets
        self.view = human_arena_payload(session, assets_root)
        self.selected: tuple[str, int] | None = None
        self.reorder_mode = False
        self.reorder_order: list[int] = []
        self.battle_index = 0
        self.decision_started = time.monotonic()
        self.token = f"sapai-human-{uuid.uuid4().hex}"
        self.root = widgets.VBox(
            layout=widgets.Layout(width="100%", border="1px solid #dedbd0", padding="12px")
        )
        self.root.add_class("sapai-human-core-widget")
        self.root.add_class(self.token)
        self.render()

    def _button(
        self,
        description: str,
        callback: Any,
        *,
        style: str = "",
        tooltip: str = "",
    ) -> Any:
        button = self.widgets.Button(
            description=description,
            button_style=style,
            tooltip=tooltip or description,
            layout=self.widgets.Layout(width="auto", min_width="112px"),
        )
        button.on_click(callback)
        return button

    def _action_button(self, action: dict[str, Any], description: str | None = None) -> Any:
        style = "danger" if action["kind"] == "SELL_PET" else "success"
        return self._button(
            description or _action_name(action, self.view["state"]),
            lambda _button, selected=action: self.invoke_action(selected),
            style=style,
        )

    def _actions(self, *kinds: str, source: int | None = None) -> list[dict[str, Any]]:
        return [
            action
            for action in self.view["actions"]
            if action["kind"] in kinds
            and (source is None or int(action.get("source", -1)) == source)
        ]

    def _target_actions(self) -> dict[int, dict[str, Any]]:
        if self.selected is None or self.reorder_mode:
            return {}
        kind, source = self.selected
        if kind == "shop_pet":
            actions = self._actions("BUY_PET", "BUY_MERGE_PET", source=source)
        elif kind == "food":
            actions = [
                action
                for action in self._actions("BUY_FOOD", source=source)
                if int(action.get("target", -1)) >= 0
            ]
        elif kind == "team":
            actions = self._actions("MERGE_BOARD_PET", source=source)
        else:
            actions = []
        return {int(action["target"]): action for action in actions}

    def _select(self, kind: str, index: int) -> None:
        if self.reorder_mode and kind == "team":
            if self.view["state"]["team"][index] and index not in self.reorder_order:
                self.reorder_order.append(index)
                self.render()
            return
        target = self._target_actions().get(index) if kind == "team" else None
        if target is not None:
            self.invoke_action(target)
            return
        value = (kind, index)
        self.selected = None if self.selected == value else value
        self.render()

    def _card_box(
        self,
        content: str,
        kind: str,
        index: int,
        *,
        target: bool = False,
        draggable: bool = False,
    ) -> Any:
        selected = self.selected == (kind, index)
        label = "Use target" if target else "Selected" if selected else "Select"
        button = self._button(
            label,
            lambda _button, card_kind=kind, card_index=index: self._select(card_kind, card_index),
            style="warning" if target else "info" if selected else "",
        )
        box = self.widgets.VBox(
            (self.widgets.HTML(value=content), button),
            layout=self.widgets.Layout(width="178px", margin="3px"),
        )
        if draggable:
            box.add_class("sapai-team-card")
            box.add_class(f"sapai-team-pos-{index}")
        return box

    def _team(self, pets: list[dict[str, Any] | None], *, interactive: bool = True) -> Any:
        sprites = self.view.get("sprites", {}).get("pet", {})
        targets = self._target_actions()
        cards = []
        # Simulator slot zero is the front. Reverse it so the front is rightmost.
        for index in reversed(range(len(pets))):
            pet = pets[index]
            content = _pet_html(
                pet,
                sprites.get(pet["name"], "") if pet else "",
                label=f"Slot {index + 1}" + (" · FRONT" if index == 0 else ""),
            )
            if interactive:
                cards.append(
                    self._card_box(
                        content,
                        "team",
                        index,
                        target=index in targets,
                        draggable=pet is not None and self.view["stage"] == "shop",
                    )
                )
            else:
                cards.append(self.widgets.HTML(value=content))
        return self.widgets.HBox(cards, layout=self.widgets.Layout(flex_flow="row wrap"))

    def _summary(self) -> Any:
        summary = self.view["summary"]
        return self.widgets.HTML(
            value=(
                "<b>Benchmark:</b> "
                f"{int(summary['games_completed'])} completed games · "
                f"{float(summary['trophies_mean']):.2f} mean trophies · "
                f"{float(summary['battle_win_rate']) * 100:.1f}% battle win rate · "
                f"{int(summary['decisions'])} decisions"
            )
        )

    def _context(self) -> Any:
        if self.reorder_mode:
            occupied = [index for index, pet in enumerate(self.view["state"]["team"]) if pet]
            names = [self.view["state"]["team"][index]["name"] for index in self.reorder_order]
            controls = [
                self._button("Reset", lambda _button: self._reset_reorder()),
                self._button("Cancel", lambda _button: self._cancel_reorder()),
            ]
            if len(self.reorder_order) == len(occupied):
                action = next(
                    (
                        candidate
                        for candidate in self._actions("REORDER")
                        if list(candidate.get("order", [])) == self.reorder_order
                    ),
                    None,
                )
                if action is not None:
                    controls.insert(0, self._action_button(action, "Apply order"))
            text = " → ".join(html.escape(str(name)) for name in names) or "Nothing selected"
            return self.widgets.VBox(
                (
                    self.widgets.HTML(
                        value=(
                            "<b>Reorder team</b><br>Click every occupied team card in "
                            "front-to-back order. Front is the rightmost card.<br>"
                            f"Current selection: {text}"
                        )
                    ),
                    self.widgets.HBox(controls, layout=self.widgets.Layout(flex_flow="row wrap")),
                )
            )

        if self.selected is None:
            return self.widgets.HTML(
                value=(
                    "<b>Choose a card.</b> Select a shop offer or team pet to expose its "
                    "legal actions. Highlighted team cards are valid targets."
                )
            )
        kind, source = self.selected
        if kind == "shop_pet":
            candidates = self._actions(
                "BUY_PET", "BUY_MERGE_PET", "FREEZE_PET", "UNFREEZE_PET", source=source
            )
        elif kind == "food":
            candidates = self._actions("BUY_FOOD", "FREEZE_FOOD", "UNFREEZE_FOOD", source=source)
        else:
            candidates = self._actions("SELL_PET", "MERGE_BOARD_PET", source=source)
        controls = [self._action_button(action) for action in candidates]
        if not controls:
            controls = [self.widgets.HTML(value="No direct action is available.")]
        return self.widgets.HBox(controls, layout=self.widgets.Layout(flex_flow="row wrap"))

    def _hidden_reorder_actions(self) -> Any:
        buttons = []
        for action in self._actions("REORDER"):
            order = "-".join(str(int(position)) for position in action.get("order", []))
            button = self._action_button(action, "reorder")
            button.add_class(f"sapai-reorder-{order}")
            buttons.append(button)
        return self.widgets.Box(buttons, layout=self.widgets.Layout(display="none"))

    def _shop(self) -> list[Any]:
        state = self.view["state"]
        pet_sprites = self.view.get("sprites", {}).get("pet", {})
        food_sprites = self.view.get("sprites", {}).get("food", {})
        shop_pets = [
            self._card_box(
                _pet_html(pet, pet_sprites.get(pet["name"], "")),
                "shop_pet",
                index,
            )
            for index, pet in enumerate(state["shopPets"])
        ]
        foods = [
            self._card_box(
                _food_html(food, food_sprites.get(food["name"], "")),
                "food",
                index,
            )
            for index, food in enumerate(state["shopFoods"])
        ]
        toolbar = []
        roll = self._actions("ROLL")
        if roll:
            toolbar.append(self._action_button(roll[0], "↻ Roll (1 gold)"))
        toolbar.append(self._button("Reorder team", lambda _button: self._start_reorder()))
        end = self._actions("END_TURN")
        if end:
            toolbar.append(self._action_button(end[0], "End turn"))
        return [
            self.widgets.HTML(value="<b>Team · front is right · drag or use Reorder team</b>"),
            self._team(state["team"]),
            self.widgets.HTML(value="<b>Shop pets</b>"),
            self.widgets.HBox(shop_pets, layout=self.widgets.Layout(flex_flow="row wrap")),
            self.widgets.HTML(value="<b>Shop food</b>"),
            self.widgets.HBox(foods, layout=self.widgets.Layout(flex_flow="row wrap")),
            self._context(),
            self.widgets.HBox(toolbar, layout=self.widgets.Layout(flex_flow="row wrap")),
            self._hidden_reorder_actions(),
            self._summary(),
        ]

    def _battle_slide_html(self, slide: dict[str, Any]) -> str:
        sprites = self.view.get("sprites", {}).get("pet", {})

        def team_cards(pets: list[dict[str, Any] | None], *, reverse: bool) -> str:
            ordered = list(reversed(pets)) if reverse else pets
            return "".join(
                _pet_html(pet, sprites.get(pet["name"], "")) for pet in ordered if pet is not None
            )

        log = "<br>".join(html.escape(str(line)) for line in slide.get("log", [])[-8:])
        return (
            f"<h3>{html.escape(str(slide['label']))}</h3>"
            '<div style="display:flex;gap:14px;align-items:center;overflow-x:auto">'
            f'<div><b>Player · front at center</b><div style="display:flex">'
            f"{team_cards(slide['player'], reverse=True)}</div></div><b>VS</b>"
            f'<div><b>Opponent · front at center</b><div style="display:flex">'
            f"{team_cards(slide['opponent'], reverse=False)}</div></div></div>"
            f'<div style="margin-top:10px;padding:9px;background:#f4f0e4">{log}</div>'
        )

    def _battle(self) -> list[Any]:
        battle = self.view["battle"]
        slides = battle["slides"]
        self.battle_index = max(0, min(len(slides) - 1, self.battle_index))
        content = self.widgets.HTML(value=self._battle_slide_html(slides[self.battle_index]))
        counter = self.widgets.HTML(value=f"{self.battle_index + 1} / {len(slides)}")
        slider = self.widgets.IntSlider(
            value=self.battle_index,
            min=0,
            max=max(0, len(slides) - 1),
            step=1,
            description="Frame",
            continuous_update=True,
            layout=self.widgets.Layout(flex="1 1 300px"),
        )

        def change_slide(change: dict[str, Any]) -> None:
            self.battle_index = int(change["new"])
            content.value = self._battle_slide_html(slides[self.battle_index])
            counter.value = f"{self.battle_index + 1} / {len(slides)}"

        slider.observe(change_slide, names="value")
        play = self.widgets.Play(
            value=self.battle_index,
            min=0,
            max=max(0, len(slides) - 1),
            step=1,
            interval=650,
            description="Play",
        )
        self.widgets.jslink((play, "value"), (slider, "value"))
        outcome = str(battle["outcome"]).replace("_", " ")
        continue_button = self._button(
            f"Continue · {outcome}",
            lambda _button: self.invoke("continue", {"revision": self.view["revision"]}),
            style="success",
        )
        return [
            content,
            self.widgets.HBox(
                (play, slider, counter, continue_button),
                layout=self.widgets.Layout(flex_flow="row wrap", align_items="center"),
            ),
            self._summary(),
        ]

    def _complete(self) -> list[Any]:
        state = self.view["state"]
        return [
            self.widgets.HTML(value="<b>Final team · front is right</b>"),
            self._team(state["team"], interactive=False),
            self._summary(),
            self._button(
                "Start next game",
                lambda _button: self.invoke("new_episode", {"revision": self.view["revision"]}),
                style="success",
            ),
        ]

    def render(self) -> None:
        state = self.view["state"]
        heading = self.widgets.HTML(
            value=(
                f"<h2 style='margin:0'>Human Arena benchmark</h2>"
                f"<div>{html.escape(str(self.view['participant_alias']))} · "
                f"{html.escape(str(self.view['pack']))} · revision {self.view['revision']}</div>"
                f"<div><b>Episode {int(self.view['episode_index']) + 1} · Turn "
                f"{state['turn']}</b> · Tier {state['tier']} · 🪙 {state['gold']} · "
                f"🏆 {state['trophies']} · ♥ {state['lives']}</div>"
            )
        )
        children: list[Any] = [heading]
        if self.view.get("error"):
            children.append(
                self.widgets.HTML(
                    value=(
                        '<div style="padding:10px;background:#ffe2de;color:#74271f">'
                        f"{html.escape(str(self.view['error']))}</div>"
                    )
                )
            )
        stage = self.view["stage"]
        children.extend(
            self._shop()
            if stage == "shop"
            else self._battle()
            if stage == "battle_review"
            else self._complete()
        )
        self.root.children = tuple(children)

    def _start_reorder(self) -> None:
        self.reorder_mode = True
        self.reorder_order = []
        self.selected = None
        self.render()

    def _reset_reorder(self) -> None:
        self.reorder_order = []
        self.render()

    def _cancel_reorder(self) -> None:
        self.reorder_mode = False
        self.reorder_order = []
        self.render()

    def invoke_action(self, action: dict[str, Any]) -> None:
        self.invoke(
            "action",
            {
                "action_id": action["id"],
                "revision": self.view["revision"],
                "elapsed_ms": max(0.0, (time.monotonic() - self.decision_started) * 1000),
            },
        )

    def invoke(self, command: str, parameters: dict[str, Any]) -> None:
        self.view = human_arena_command(
            self.session,
            self.assets_root,
            command,
            parameters,
        )
        self.selected = None
        self.reorder_mode = False
        self.reorder_order = []
        self.battle_index = 0
        self.decision_started = time.monotonic()
        self.render()


def _drag_enhancement_script(token: str) -> str:
    """Return optional drag support that operates only on core widget buttons."""

    encoded_token = json.dumps(token)
    return f"""
(() => {{
  const token = {encoded_token};
  const bind = () => {{
    const host = document.querySelector(`.${{token}}`);
    if (!host) return false;
    host.querySelectorAll(".sapai-team-card").forEach((card) => {{
      if (card.dataset.sapaiDragBound) return;
      card.dataset.sapaiDragBound = "1";
      card.draggable = true;
      card.addEventListener("dragstart", (event) => {{
        const match = [...card.classList].find((name) => name.startsWith("sapai-team-pos-"));
        if (!match) return;
        event.dataTransfer.setData("text/plain", match.slice("sapai-team-pos-".length));
        event.dataTransfer.effectAllowed = "move";
        card.style.opacity = "0.45";
      }});
      card.addEventListener("dragend", () => {{ card.style.opacity = ""; }});
      card.addEventListener("dragover", (event) => event.preventDefault());
      card.addEventListener("drop", (event) => {{
        event.preventDefault();
        const source = Number(event.dataTransfer.getData("text/plain"));
        const targetMatch = [...card.classList].find(
          (name) => name.startsWith("sapai-team-pos-"),
        );
        if (!Number.isInteger(source) || !targetMatch) return;
        const target = Number(targetMatch.slice("sapai-team-pos-".length));
        if (source === target) return;
        const visual = [...host.querySelectorAll(".sapai-team-card")].map((item) => {{
          const name = [...item.classList].find(
            (value) => value.startsWith("sapai-team-pos-"),
          );
          return Number(name.slice("sapai-team-pos-".length));
        }});
        visual.splice(visual.indexOf(source), 1);
        const targetIndex = visual.indexOf(target);
        const bounds = card.getBoundingClientRect();
        const insertAfter = event.clientX >= bounds.left + bounds.width / 2;
        visual.splice(targetIndex + (insertAfter ? 1 : 0), 0, source);
        const order = [...visual].reverse().join("-");
        const action = host.querySelector(`.sapai-reorder-${{order}} button`);
        if (action) action.click();
      }});
    }});
    return true;
  }};
  let attempts = 0;
  const timer = setInterval(() => {{
    attempts += 1;
    if (bind() || attempts >= 100) clearInterval(timer);
  }}, 100);
  const observer = new MutationObserver(bind);
  const startObserver = setInterval(() => {{
    const host = document.querySelector(`.${{token}}`);
    if (!host) return;
    clearInterval(startObserver);
    observer.observe(host, {{childList: true, subtree: true}});
    bind();
  }}, 100);
  setTimeout(() => clearInterval(startObserver), 10000);
}})();
"""


def display_human_arena_widget(
    session: HumanArenaSession,
    assets_root: str | Path,
) -> Any:
    """Display the card UI using only widget models already bundled by Kaggle."""

    try:
        import ipywidgets as widgets
        from IPython.display import Javascript, display
    except ModuleNotFoundError as error:  # pragma: no cover - optional notebook adapter
        raise RuntimeError(
            "install the 'notebook' extra to display the Jupyter human Arena widget"
        ) from error

    controller = _HumanArenaCoreWidget(session, assets_root, widgets)
    display(controller.root)
    # Dragging is progressive enhancement. Every operation remains available
    # through native widget buttons if the notebook frontend blocks JavaScript.
    display(Javascript(_drag_enhancement_script(controller.token)))
    return controller.root
