"""Resumable, audit-friendly human Arena benchmark sessions."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sapai.data.serialization import (
    action_to_dict,
    board_to_dict,
    run_state_from_dict,
    run_state_to_dict,
    team_to_dict,
)
from sapai.sim.actions import Action, ActionKind
from sapai.sim.battle import BattleResult, BattleResultKind, BattleSimulator
from sapai.sim.models import BattleOutcome, RunState
from sapai.sim.shop import ShopEnvironment
from sapai.training.population import OpponentPopulation

HUMAN_BENCHMARK_FORMAT = "sapai-human-arena-v1"
SessionStage = Literal["shop", "battle_review", "complete"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class HumanBenchmarkConfig:
    output_dir: str | Path
    participant_alias: str
    pack: str
    seed: int
    boards_sha256: str
    board_count: int
    repository_commit: str = "unknown"
    max_decisions_per_turn: int = 30

    def __post_init__(self) -> None:
        if self.pack != "Turtle":
            raise ValueError("the human benchmark currently supports only the Turtle pack")
        if not self.participant_alias.strip():
            raise ValueError("participant_alias cannot be empty")
        if len(self.participant_alias) > 80:
            raise ValueError("participant_alias cannot exceed 80 characters")
        if not self.boards_sha256:
            raise ValueError("boards_sha256 cannot be empty")
        if self.board_count < 1:
            raise ValueError("board_count must be positive")
        if self.max_decisions_per_turn < 1:
            raise ValueError("max_decisions_per_turn must be positive")

    @property
    def directory(self) -> Path:
        return Path(self.output_dir).expanduser()


@dataclass(slots=True)
class _SessionData:
    state: RunState
    stage: SessionStage
    revision: int
    episode_index: int
    decision_index: int
    decisions_this_turn: int
    episode_started_at: str
    events: list[dict[str, Any]]
    battles: list[dict[str, Any]]
    pending_battle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": HUMAN_BENCHMARK_FORMAT,
            "state": run_state_to_dict(self.state),
            "stage": self.stage,
            "revision": self.revision,
            "episode_index": self.episode_index,
            "decision_index": self.decision_index,
            "decisions_this_turn": self.decisions_this_turn,
            "episode_started_at": self.episode_started_at,
            "events": self.events,
            "battles": self.battles,
            "pending_battle": self.pending_battle,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _SessionData:
        if value.get("format") != HUMAN_BENCHMARK_FORMAT:
            raise ValueError("unsupported human benchmark checkpoint format")
        stage = str(value["stage"])
        if stage not in {"shop", "battle_review", "complete"}:
            raise ValueError(f"invalid human benchmark stage: {stage!r}")
        pending = value.get("pending_battle")
        if stage == "battle_review" and not isinstance(pending, dict):
            raise ValueError("battle-review checkpoint has no pending battle")
        if stage != "battle_review" and pending is not None:
            raise ValueError("non-battle checkpoint unexpectedly contains a pending battle")
        return cls(
            state=run_state_from_dict(value["state"]),
            stage=stage,  # type: ignore[arg-type]
            revision=int(value["revision"]),
            episode_index=int(value["episode_index"]),
            decision_index=int(value["decision_index"]),
            decisions_this_turn=int(value["decisions_this_turn"]),
            episode_started_at=str(value["episode_started_at"]),
            events=[dict(event) for event in value.get("events", [])],
            battles=[dict(battle) for battle in value.get("battles", [])],
            pending_battle=dict(pending) if isinstance(pending, dict) else None,
        )

    def clone(self) -> _SessionData:
        return self.from_dict(self.to_dict())


class HumanArenaSession:
    """One open-ended human benchmark with crash-safe per-action persistence."""

    def __init__(
        self,
        environment: ShopEnvironment,
        battle: BattleSimulator,
        population: OpponentPopulation,
        config: HumanBenchmarkConfig,
        data: _SessionData,
    ) -> None:
        self.environment = environment
        self.battle = battle
        self.population = population
        self.config = config
        self._data = data
        self._lock = threading.RLock()

    @classmethod
    def create_or_resume(
        cls,
        environment: ShopEnvironment,
        battle: BattleSimulator,
        population: OpponentPopulation,
        config: HumanBenchmarkConfig,
    ) -> HumanArenaSession:
        if len(population.boards) != config.board_count:
            raise ValueError(
                "configured board_count does not match the supplied opponent population"
            )
        config.directory.mkdir(parents=True, exist_ok=True)
        session = cls.__new__(cls)
        session.environment = environment
        session.battle = battle
        session.population = population
        session.config = config
        session._lock = threading.RLock()
        session._validate_or_create_manifest()
        current = config.directory / "current.json"
        if current.exists():
            session._data = _SessionData.from_dict(json.loads(current.read_text(encoding="utf-8")))
            if session._data.state.pack != config.pack:
                raise ValueError("checkpoint pack does not match the benchmark manifest")
        else:
            session._data = session._new_episode_data(episode_index=0, revision=0)
            session._save_current(session._data)
        if session._data.stage == "complete":
            session._finalize_episode()
        else:
            session._refresh_summary()
        return session

    @property
    def state(self) -> RunState:
        return run_state_from_dict(run_state_to_dict(self._data.state))

    @property
    def stage(self) -> SessionStage:
        return self._data.stage

    @property
    def revision(self) -> int:
        return self._data.revision

    @property
    def pending_battle(self) -> Mapping[str, Any] | None:
        if self._data.pending_battle is None:
            return None
        return json.loads(json.dumps(self._data.pending_battle))

    def snapshot(self) -> dict[str, Any]:
        actions = self._legal_actions() if self.stage == "shop" else []
        return {
            "format": HUMAN_BENCHMARK_FORMAT,
            "participant_alias": self.config.participant_alias,
            "pack": self.config.pack,
            "stage": self.stage,
            "revision": self.revision,
            "episode_index": self._data.episode_index,
            "decision_index": self._data.decision_index,
            "state": run_state_to_dict(self._data.state),
            "actions": [
                {
                    "id": self._action_id(index, action),
                    **action_to_dict(action),
                }
                for index, action in enumerate(actions)
            ],
            "battle": self._data.pending_battle,
            "summary": self._read_summary(),
        }

    def apply_action(
        self,
        action_id: str,
        *,
        expected_revision: int,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        with self._lock:
            return self._apply_action(
                action_id,
                expected_revision=expected_revision,
                elapsed_ms=elapsed_ms,
            )

    def _apply_action(
        self,
        action_id: str,
        *,
        expected_revision: int,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        self._require_revision(expected_revision)
        if self.stage != "shop":
            raise ValueError("actions can only be applied while the shop is visible")
        elapsed = float(elapsed_ms)
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 86_400_000:
            raise ValueError("elapsed_ms must be a finite value between zero and one day")
        actions = self._legal_actions()
        identified_actions = [
            (self._action_id(index, candidate), candidate)
            for index, candidate in enumerate(actions)
        ]
        action = next(
            (
                candidate
                for candidate_id, candidate in identified_actions
                if candidate_id == action_id
            ),
            None,
        )
        if action is None:
            raise ValueError("unknown or stale human benchmark action ID")

        candidate = self._data.clone()
        state_before = run_state_to_dict(candidate.state)
        transition_seed = self._seed("action", candidate.decision_index)
        candidate.state = self.environment.step(
            candidate.state,
            action,
            random.Random(transition_seed),
        ).state
        candidate.events.append(
            {
                "type": "decision",
                "decision_index": candidate.decision_index,
                "turn": candidate.state.turn,
                "revision": candidate.revision,
                "recorded_at": _now(),
                "active_decision_ms": elapsed,
                "transition_seed": transition_seed,
                "state_before": state_before,
                "legal_actions": [
                    {"id": candidate_id, **action_to_dict(item)}
                    for candidate_id, item in identified_actions
                ],
                "chosen_action": {"id": action_id, **action_to_dict(action)},
                "state_after": run_state_to_dict(candidate.state),
            }
        )
        candidate.decision_index += 1
        candidate.decisions_this_turn += 1

        if candidate.state.awaiting_battle:
            opponent_seed = self._seed("opponent", candidate.state.turn)
            opponent = self.population.sample(
                pack=candidate.state.pack,
                turn=candidate.state.turn,
                version=candidate.state.version,
                rng=random.Random(opponent_seed),
            )
            battle_seed = self._seed("battle", candidate.state.turn)
            result = self.battle.simulate(
                candidate.state.team,
                opponent.team,
                seed=battle_seed,
            )
            candidate.pending_battle = {
                "turn": candidate.state.turn,
                "opponent_seed": opponent_seed,
                "battle_seed": battle_seed,
                "outcome_seed": self._seed("outcome", candidate.state.turn),
                "opponent": board_to_dict(opponent),
                "result": _battle_result_to_dict(result, include_frames=True),
            }
            candidate.stage = "battle_review"
        candidate.revision += 1
        self._save_current(candidate)
        self._data = candidate
        return self.snapshot()

    def continue_battle(self, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            return self._continue_battle(expected_revision=expected_revision)

    def _continue_battle(self, *, expected_revision: int) -> dict[str, Any]:
        self._require_revision(expected_revision)
        if self.stage != "battle_review" or self._data.pending_battle is None:
            raise ValueError("there is no battle waiting for review")
        candidate = self._data.clone()
        pending = candidate.pending_battle
        result = pending["result"]
        outcome = _arena_outcome(BattleResultKind(result["outcome"]))
        state_before = run_state_to_dict(candidate.state)
        candidate.state = self.environment.apply_outcome(
            candidate.state,
            outcome,
            random.Random(int(pending["outcome_seed"])),
        ).state
        candidate.battles.append(
            {
                "turn": int(pending["turn"]),
                "recorded_at": _now(),
                "opponent_seed": int(pending["opponent_seed"]),
                "battle_seed": int(pending["battle_seed"]),
                "outcome_seed": int(pending["outcome_seed"]),
                "opponent": pending["opponent"],
                "result": {key: value for key, value in result.items() if key != "frames"},
                "state_before_outcome": state_before,
                "state_after_outcome": run_state_to_dict(candidate.state),
            }
        )
        candidate.pending_battle = None
        candidate.decisions_this_turn = 0
        candidate.stage = "complete" if candidate.state.terminal else "shop"
        candidate.revision += 1
        self._save_current(candidate)
        self._data = candidate
        if candidate.stage == "complete":
            self._finalize_episode()
        return self.snapshot()

    def new_episode(self, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            return self._new_episode(expected_revision=expected_revision)

    def _new_episode(self, *, expected_revision: int) -> dict[str, Any]:
        self._require_revision(expected_revision)
        if self.stage != "complete":
            raise ValueError("a new episode can only start after the current one is complete")
        self._finalize_episode()
        candidate = self._new_episode_data(
            episode_index=self._data.episode_index + 1,
            revision=self._data.revision + 1,
        )
        self._save_current(candidate)
        self._data = candidate
        self._refresh_summary()
        return self.snapshot()

    def _new_episode_data(self, *, episode_index: int, revision: int) -> _SessionData:
        reset_seed = self._seed_for_episode(episode_index, "reset", 0)
        state = self.environment.reset(pack=self.config.pack, seed=reset_seed)
        return _SessionData(
            state=state,
            stage="shop",
            revision=revision,
            episode_index=episode_index,
            decision_index=0,
            decisions_this_turn=0,
            episode_started_at=_now(),
            events=[
                {
                    "type": "episode_started",
                    "recorded_at": _now(),
                    "episode_index": episode_index,
                    "reset_seed": reset_seed,
                    "initial_state": run_state_to_dict(state),
                }
            ],
            battles=[],
        )

    def _legal_actions(self) -> list[Action]:
        actions = self.environment.legal_actions(self._data.state)
        if self._data.decisions_this_turn + 1 >= self.config.max_decisions_per_turn:
            return [action for action in actions if action.kind is ActionKind.END_TURN]
        return actions

    def _action_id(self, index: int, action: Action) -> str:
        encoded = json.dumps(
            action_to_dict(action),
            separators=(",", ":"),
            sort_keys=True,
        )
        material = f"{self.revision}:{index}:{encoded}".encode()
        return "action-" + hashlib.sha256(material).hexdigest()[:20]

    def _seed(self, purpose: str, counter: int) -> int:
        return self._seed_for_episode(self._data.episode_index, purpose, counter)

    def _seed_for_episode(self, episode: int, purpose: str, counter: int) -> int:
        material = f"{self.config.seed}:{episode}:{purpose}:{counter}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)

    def _require_revision(self, expected: int) -> None:
        if int(expected) != self.revision:
            raise ValueError(
                f"stale human benchmark revision: expected {expected}, current {self.revision}"
            )

    def _manifest_identity(self) -> dict[str, Any]:
        return {
            "format": HUMAN_BENCHMARK_FORMAT,
            "participant_alias": self.config.participant_alias,
            "pack": self.config.pack,
            "seed": self.config.seed,
            "boards_sha256": self.config.boards_sha256,
            "board_count": self.config.board_count,
            "repository_commit": self.config.repository_commit,
            "max_decisions_per_turn": self.config.max_decisions_per_turn,
            "simulator_configuration": {
                "battle_max_rounds": self.battle.MAX_ROUNDS,
                "battle_max_events": self.battle.MAX_EVENTS,
                "shop_pet_slots_by_tier": [
                    self.environment.pet_slots(tier) for tier in range(1, 7)
                ],
                "shop_food_slots_by_tier": [
                    self.environment.food_slots(tier) for tier in range(1, 7)
                ],
                "battle_rules_source": dict(self.battle.rules.source),
                "shop_rules_source": dict(self.environment.abilities.rules.source),
            },
        }

    def _validate_or_create_manifest(self) -> None:
        path = self.config.directory / "manifest.json"
        identity = self._manifest_identity()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_identity = {key: existing.get(key) for key in identity}
            if existing_identity != identity:
                raise ValueError(
                    f"human benchmark settings changed in {self.config.directory}; "
                    "use a new benchmark directory"
                )
            return
        _atomic_json(path, {**identity, "created_at": _now()})

    def _save_current(self, data: _SessionData) -> None:
        _atomic_json(self.config.directory / "current.json", data.to_dict())

    def _episode_path(self) -> Path:
        return self.config.directory / "episodes" / f"{self._data.episode_index:06d}.json"

    def _finalize_episode(self) -> None:
        if self.stage != "complete":
            return
        path = self._episode_path()
        episode = {
            "format": HUMAN_BENCHMARK_FORMAT,
            "participant_alias": self.config.participant_alias,
            "pack": self.config.pack,
            "episode_index": self._data.episode_index,
            "started_at": self._data.episode_started_at,
            "completed_at": _now(),
            "final_state": run_state_to_dict(self._data.state),
            "events": self._data.events,
            "battles": self._data.battles,
            "metrics": self._episode_metrics(),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("episode_index") != self._data.episode_index:
                raise ValueError(f"invalid completed episode file: {path}")
        else:
            _atomic_json(path, episode)
        self._refresh_summary()

    def _episode_metrics(self) -> dict[str, Any]:
        outcomes = [battle["result"]["outcome"] for battle in self._data.battles]
        decision_events = [event for event in self._data.events if event["type"] == "decision"]
        timings = [float(event["active_decision_ms"]) for event in decision_events]
        action_counts: dict[str, int] = {}
        for event in decision_events:
            kind = str(event["chosen_action"]["kind"])
            action_counts[kind] = action_counts.get(kind, 0) + 1
        return {
            "trophies": self._data.state.trophies,
            "lives": self._data.state.lives,
            "turns": len(self._data.battles),
            "decisions": len(decision_events),
            "battle_wins": outcomes.count(BattleResultKind.PLAYER_WIN.value),
            "battle_draws": outcomes.count(BattleResultKind.DRAW.value),
            "battle_losses": outcomes.count(BattleResultKind.OPPONENT_WIN.value),
            "active_decision_ms_total": sum(timings),
            "active_decision_ms_mean": sum(timings) / len(timings) if timings else 0.0,
            "active_decision_ms_median": statistics.median(timings) if timings else 0.0,
            "active_decision_ms_min": min(timings) if timings else 0.0,
            "active_decision_ms_max": max(timings) if timings else 0.0,
            "action_counts": action_counts,
        }

    def _read_summary(self) -> dict[str, Any]:
        path = self.config.directory / "summary.json"
        return (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else self._empty_summary()
        )

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "format": HUMAN_BENCHMARK_FORMAT,
            "participant_alias": self.config.participant_alias,
            "games_completed": 0,
            "episode_in_progress": self.stage != "complete",
            "trophy_histogram": {str(value): 0 for value in range(11)},
            "trophies_mean": 0.0,
            "trophies_median": 0.0,
            "turns_total": 0,
            "turns_mean": 0.0,
            "turns_median": 0.0,
            "battle_wins": 0,
            "battle_draws": 0,
            "battle_losses": 0,
            "battle_win_rate": 0.0,
            "battle_draw_rate": 0.0,
            "battle_loss_rate": 0.0,
            "decisions": 0,
            "active_decision_ms_total": 0.0,
            "active_decision_ms_mean": 0.0,
            "active_decision_ms_median": 0.0,
            "active_decision_ms_min": 0.0,
            "active_decision_ms_max": 0.0,
            "action_counts": {},
        }

    def _refresh_summary(self) -> None:
        paths = sorted((self.config.directory / "episodes").glob("*.json"))
        episodes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if not episodes:
            summary = self._empty_summary()
            summary["episode_in_progress"] = self.stage != "complete"
            _atomic_json(self.config.directory / "summary.json", summary)
            return
        metrics = [episode["metrics"] for episode in episodes]
        trophies = [int(row["trophies"]) for row in metrics]
        turns = [int(row["turns"]) for row in metrics]
        wins = sum(int(row["battle_wins"]) for row in metrics)
        draws = sum(int(row["battle_draws"]) for row in metrics)
        losses = sum(int(row["battle_losses"]) for row in metrics)
        decisions = sum(int(row["decisions"]) for row in metrics)
        timings = [
            float(event["active_decision_ms"])
            for episode in episodes
            for event in episode["events"]
            if event["type"] == "decision"
        ]
        action_counts: dict[str, int] = {}
        for row in metrics:
            for kind, count in row["action_counts"].items():
                action_counts[kind] = action_counts.get(kind, 0) + int(count)
        battles = wins + draws + losses
        summary = {
            "format": HUMAN_BENCHMARK_FORMAT,
            "participant_alias": self.config.participant_alias,
            "updated_at": _now(),
            "games_completed": len(episodes),
            "episode_in_progress": self.stage != "complete",
            "trophy_histogram": {str(value): trophies.count(value) for value in range(11)},
            "trophies_mean": sum(trophies) / len(trophies),
            "trophies_median": statistics.median(trophies),
            "turns_total": sum(turns),
            "turns_mean": sum(turns) / len(turns),
            "turns_median": statistics.median(turns),
            "battle_wins": wins,
            "battle_draws": draws,
            "battle_losses": losses,
            "battle_win_rate": wins / battles if battles else 0.0,
            "battle_draw_rate": draws / battles if battles else 0.0,
            "battle_loss_rate": losses / battles if battles else 0.0,
            "decisions": decisions,
            "active_decision_ms_total": sum(timings),
            "active_decision_ms_mean": sum(timings) / len(timings) if timings else 0.0,
            "active_decision_ms_median": statistics.median(timings) if timings else 0.0,
            "active_decision_ms_min": min(timings) if timings else 0.0,
            "active_decision_ms_max": max(timings) if timings else 0.0,
            "action_counts": action_counts,
        }
        _atomic_json(self.config.directory / "summary.json", summary)


def _arena_outcome(outcome: BattleResultKind) -> BattleOutcome:
    if outcome is BattleResultKind.PLAYER_WIN:
        return BattleOutcome.WIN
    if outcome is BattleResultKind.OPPONENT_WIN:
        return BattleOutcome.LOSS
    return BattleOutcome.DRAW


def _battle_result_to_dict(
    result: BattleResult,
    *,
    include_frames: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "outcome": result.outcome.value,
        "rounds": result.rounds,
        "player": team_to_dict(result.player),
        "opponent": team_to_dict(result.opponent),
        "log": list(result.log),
    }
    if include_frames:
        value["frames"] = [
            {
                "label": frame.label,
                "player": team_to_dict(frame.player),
                "opponent": team_to_dict(frame.opponent),
                "log_index": frame.log_index,
                "event": frame.event,
                "actor_id": frame.actor_id,
                "target_id": frame.target_id,
            }
            for frame in result.frames
        ]
    return value
