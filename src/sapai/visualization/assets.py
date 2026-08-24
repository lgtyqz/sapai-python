"""Resolve simulator names through SAP data NameId sprite mappings."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import ClassVar


class SpriteAtlas:
    DIRECTORIES: ClassVar[dict[str, str]] = {
        "pet": "Pets",
        "food": "Food",
        "toy": "Toys",
    }
    DATA_FILES: ClassVar[dict[str, str]] = {
        "pet": "pets.json",
        "food": "food.json",
        "toy": "toys.json",
    }

    def __init__(self, assets_root: str | Path):
        self.root = Path(assets_root)
        self._data_uris: dict[tuple[str, str], str | None] = {}
        self.mapping: dict[str, dict[str, Path]] = {}
        for kind, filename in self.DATA_FILES.items():
            path = self.root / "data" / filename
            rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            values: dict[str, Path] = {}
            for row in rows:
                name_id = row.get("NameId")
                if not name_id:
                    continue
                sprite = self.root / "Sprite" / self.DIRECTORIES[kind] / f"{name_id}.png"
                if sprite.exists():
                    values[str(row.get("Name", name_id))] = sprite
                    values[str(name_id)] = sprite
            self.mapping[kind] = values

    def path(self, kind: str, name: str | None) -> Path | None:
        if not name:
            return None
        mapped = self.mapping.get(kind, {}).get(name)
        if mapped:
            return mapped
        fallback = self.root / "Sprite" / self.DIRECTORIES[kind] / f"{name.replace(' ', '')}.png"
        return fallback if fallback.exists() else None

    def data_uri(self, kind: str, name: str) -> str | None:
        key = (kind, name)
        if key in self._data_uris:
            return self._data_uris[key]
        path = self.path(kind, name)
        if path is None:
            self._data_uris[key] = None
            return None
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        value = f"data:image/png;base64,{encoded}"
        self._data_uris[key] = value
        return value

    def payload(self, names: dict[str, set[str]]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for kind, values in names.items():
            result[kind] = {}
            for name in sorted(values):
                uri = self.data_uri(kind, name)
                if uri:
                    result[kind][name] = uri
        return result
