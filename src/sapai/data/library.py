from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sapai.data.replay import BoardSnapshot, ReplayParser


class SapLibraryClient:
    """Read-only SAP Library adapter for Neon's Postgres service.

    Neon's official Python guidance uses standard Postgres drivers; this class
    imports Psycopg lazily so simulation/model code has no database dependency.
    """

    def __init__(self, parser: ReplayParser, database_url: str | None = None):
        self.parser = parser
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")

    def sample_boards(
        self,
        *,
        pack: str | None = None,
        turn: int | None = None,
        mode: int = 0,
        limit: int = 100,
    ) -> list[BoardSnapshot]:
        if limit <= 0 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10,000")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'neon' extra to query SAP Library") from error

        query = """
            SELECT id, raw_json
            FROM replays
            WHERE mode = %s
              AND (%s IS NULL OR pack = %s OR opponent_pack = %s)
            ORDER BY RANDOM()
            LIMIT %s
        """
        boards: list[BoardSnapshot] = []
        with (
            psycopg.connect(self.database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(query, (mode, pack, pack, pack, limit))
            for row in cursor:
                raw = row["raw_json"]
                replay = json.loads(raw) if isinstance(raw, str) else raw
                boards.extend(self.parser.parse_replay(replay, replay_id=str(row["id"])))
        if turn is not None:
            boards = [board for board in boards if board.turn == turn]
        if pack is not None:
            boards = [board for board in boards if board.pack == pack]
        return boards


def read_replay_jsonl(path: str | Path, parser: ReplayParser) -> Iterator[BoardSnapshot]:
    """Offline ingestion path for reproducible Colab datasets."""

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            replay = row.get("raw_json", row)
            if isinstance(replay, str):
                replay = json.loads(replay)
            yield from parser.parse_replay(replay, replay_id=str(row.get("id", "")) or None)
