"""Jupyter-widget adapter for interactive human Arena sessions."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from sapai.training.human import HumanArenaSession
from sapai.visualization.colab import human_arena_command, human_arena_payload


def display_human_arena_widget(
    session: HumanArenaSession,
    assets_root: str | Path,
) -> Any:
    """Display the card UI over standard Jupyter widget comms, including Kaggle."""

    try:
        import anywidget
        import traitlets
        from IPython.display import display
    except ModuleNotFoundError as error:  # pragma: no cover - optional notebook adapter
        raise RuntimeError(
            "install the 'notebook' extra to display the Jupyter human Arena widget"
        ) from error

    resources = files("sapai.visualization").joinpath("static")
    esm = "\n".join(
        (
            resources.joinpath("human_arena.js").read_text(encoding="utf-8"),
            resources.joinpath("human_arena_widget.js").read_text(encoding="utf-8"),
        )
    )
    base_css = (
        resources.joinpath("sapai.css")
        .read_text(encoding="utf-8")
        .replace(":root {", ":scope {")
        .replace("body {", ":scope {")
    )
    human_css = resources.joinpath("human_arena.css").read_text(encoding="utf-8")
    # anywidget styles load globally. A CSS scope prevents generic visualizer
    # selectors such as ``.panel`` and ``.team`` from affecting Kaggle itself.
    css = f"@scope (.sapai-human-output) {{\n{base_css}\n{human_css}\n}}"

    class HumanArenaWidget(anywidget.AnyWidget):
        _esm = esm
        _css = css

        view = traitlets.Dict().tag(sync=True)
        request = traitlets.Dict().tag(sync=True)
        response = traitlets.Dict().tag(sync=True)

        @traitlets.observe("request")
        def _handle_request(self, change):
            request = change["new"]
            request_id = str(request.get("id", ""))
            if not request_id:
                return
            self.view = human_arena_command(
                session,
                assets_root,
                str(request.get("command", "")),
                request.get("parameters"),
            )
            self.response = {"id": request_id}

    widget = HumanArenaWidget(
        view=human_arena_payload(session, assets_root),
        request={},
        response={},
    )
    display(widget)
    return widget
