from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from sapai.data.library import SapLibraryClient, read_replay_jsonl
from sapai.data.replay import ReplayParser
from sapai.data.serialization import read_boards, write_boards
from sapai.rewards import POLICY_TARGET_SCHEMA, VALUE_OBJECTIVE
from sapai.sim.battle import BattleSimulator
from sapai.sim.catalog import PACK_ALIASES, Catalog
from sapai.sim.models import Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.arena import (
    ArenaRunner,
    HeuristicPolicy,
    MixturePolicy,
    ModelPolicy,
    RandomPolicy,
    SearchPolicy,
    evaluate_arena_policy,
    read_arena_decisions,
    write_arena_decisions,
)
from sapai.training.population import (
    OpponentPopulation,
    SimulatorPopulationEvaluator,
    load_opponent_boards,
    split_opponent_populations,
)

if TYPE_CHECKING:
    from sapai.ml.models import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SEQUENCE_FORMAT = "sapai-training-sequence-v5"
LEGACY_TRAINING_SEQUENCE_FORMATS = {"sapai-training-sequence-v4"}


def _default_path(environment: str, relative: str) -> str:
    return os.environ.get(environment, str(PROJECT_ROOT / relative))


def _catalog(path: str) -> Catalog:
    return Catalog.from_json_dir(Path(path).expanduser())


def _team(catalog: Catalog, names: str) -> Team:
    return Team.from_pets(catalog.pet_by_name(name.strip()).create() for name in names.split(","))


def _json(value: object) -> None:
    print(json.dumps(value, indent=2))


def _add_training_options(parser: argparse.ArgumentParser, *, epochs: int = 10) -> None:
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")


def _add_arena_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--boards", help="stable board JSONL; synthetic pool if omitted")
    parser.add_argument(
        "--policy",
        choices=("heuristic", "random", "model", "search"),
        default="heuristic",
    )
    parser.add_argument("--policy-weights", help="policy .weights.h5 file or model directory")
    parser.add_argument("--pack", default="Turtle", choices=sorted(PACK_ALIASES))
    parser.add_argument("--board-version", help="exact replay patch version to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions-per-turn", type=int, default=30)
    parser.add_argument("--search-simulations", type=int, default=32)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument(
        "--battle-evaluation-simulations",
        type=int,
        default=8,
        help="exact opponent battles sampled for each end-turn search leaf",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sapai")
    parser.add_argument(
        "--data",
        default=_default_path("SAP_DATA_PATH", "assets/data"),
        help="directory containing pets.json, food.json, and perks.json",
    )
    parser.add_argument(
        "--assets",
        default=_default_path("SAP_ASSETS_PATH", "assets"),
        help="sprite and data asset root",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("catalog-report", help="show pack and native ability coverage")
    report.add_argument("--pack", default="Turtle", choices=sorted(PACK_ALIASES))

    shop = commands.add_parser("shop-demo", help="print a deterministic initial shop")
    shop.add_argument("--pack", default="Turtle", choices=sorted(PACK_ALIASES))
    shop.add_argument("--seed", type=int, default=7)

    for name, help_text in (
        ("battle", "run the native battle simulator"),
        ("visualize-battle", "write a sprite-backed battle timeline"),
    ):
        battle = commands.add_parser(name, help=help_text)
        battle.add_argument("--player", required=True, help="comma-separated pets, front first")
        battle.add_argument("--opponent", required=True, help="comma-separated pets, front first")
        battle.add_argument("--seed", type=int, default=7)
        if name == "visualize-battle":
            battle.add_argument("--output", default="outputs/battle.html")

    commands.add_parser("model-smoke", help="build and execute both TensorFlow models")

    library = commands.add_parser("library-sample", help="read sample boards from Neon")
    library.add_argument("--pack", default="Turtle")
    library.add_argument("--turn", type=int)
    library.add_argument("--limit", type=int, default=10)

    export = commands.add_parser("export-boards", help="export stable board JSONL from Neon")
    export.add_argument("--output", required=True)
    export.add_argument("--pack", default="Turtle")
    export.add_argument("--turn", type=int)
    export.add_argument("--limit", type=int, default=10_000)

    parse = commands.add_parser("parse-replays", help="translate raw replay JSONL to boards")
    parse.add_argument("--input", required=True)
    parse.add_argument("--output", required=True)

    label = commands.add_parser("label-battles", help="build leakage-safe W/D/L datasets")
    label.add_argument("--boards", required=True)
    label.add_argument("--output", required=True)
    label.add_argument("--examples", type=int, default=10_000)
    label.add_argument("--simulations-per-pair", type=int, default=8)
    label.add_argument("--validation-fraction", type=float, default=0.1)
    label.add_argument("--test-fraction", type=float, default=0.1)
    label.add_argument("--seed", type=int, default=0)

    train_battle = commands.add_parser("train-battle", help="train and checkpoint BattleModel")
    train_battle.add_argument("--dataset", required=True)
    train_battle.add_argument("--output", required=True)
    _add_training_options(train_battle)

    cache = commands.add_parser("cache-population", help="cache encoded opponent tensors")
    cache.add_argument("--boards", required=True)
    cache.add_argument("--output", required=True)

    generate = commands.add_parser("generate-arena", help="generate complete Arena trajectories")
    _add_arena_options(generate)
    generate.add_argument("--episodes", type=int, default=100)
    generate.add_argument("--output", required=True)

    train_policy = commands.add_parser("train-policy", help="train and checkpoint policy/value")
    train_policy.add_argument("--dataset", required=True)
    train_policy.add_argument("--validation")
    train_policy.add_argument("--output", required=True)
    _add_training_options(train_policy)

    visualize_arena = commands.add_parser(
        "visualize-arena", help="simulate and write a shop+battle Arena timeline"
    )
    _add_arena_options(visualize_arena)
    visualize_arena.add_argument("--output", default="outputs/arena.html")

    sequence = commands.add_parser(
        "train-sequence", help="run policy bootstrap and simulator-guided search distillation"
    )
    sequence.add_argument("--boards", required=True)
    sequence.add_argument("--workdir", required=True)
    sequence.add_argument("--pack", default="Turtle", choices=("Turtle",))
    sequence.add_argument("--board-version", help="exact replay patch version to train on")
    sequence.add_argument("--validation-fraction", type=float, default=0.1)
    sequence.add_argument("--test-fraction", type=float, default=0.1)
    sequence.add_argument("--validation-episodes", type=int, default=8)
    sequence.add_argument("--test-episodes", type=int, default=16)
    sequence.add_argument("--bootstrap-episodes", type=int, default=100)
    sequence.add_argument("--bootstrap-epochs", type=int, default=10)
    sequence.add_argument("--bootstrap-exploration", type=float, default=0.10)
    sequence.add_argument("--search-episodes", type=int, default=50)
    sequence.add_argument("--search-epochs", type=int, default=5)
    sequence.add_argument(
        "--search-iterations",
        type=int,
        default=3,
        help=(
            "total search iterations desired in this workdir; increase this value to "
            "continue a completed run"
        ),
    )
    sequence.add_argument("--bootstrap-replay-fraction", type=float, default=0.35)
    sequence.add_argument("--batch-size", type=int, default=64)
    sequence.add_argument("--seed", type=int, default=0)
    sequence.add_argument("--search-simulations", type=int, default=32)
    sequence.add_argument("--search-candidates", type=int, default=8)
    sequence.add_argument("--battle-evaluation-simulations", type=int, default=8)
    sequence.add_argument(
        "--progress",
        action="store_true",
        help="print flushed stage, rollout, and epoch progress for notebook runs",
    )
    return parser


def _population(args, catalog: Catalog) -> OpponentPopulation:
    if getattr(args, "boards", None):
        boards = load_opponent_boards(args.boards, catalog, args.pack)
        versions: dict[str, int] = {}
        for board in boards:
            versions[board.version] = versions.get(board.version, 0) + 1
        version = getattr(args, "board_version", None) or min(
            versions, key=lambda item: (-versions[item], item)
        )
        selected = [board for board in boards if board.version == version]
        if not selected:
            raise ValueError(f"board dataset contains no boards for version {version!r}")
        args.resolved_board_version = version
        return OpponentPopulation(selected)
    args.resolved_board_version = "synthetic"
    return OpponentPopulation.synthetic(catalog, pack=args.pack, seed=args.seed)


def _model_config_from_weights(weights: str | Path) -> tuple[ModelConfig, Path]:
    from sapai.ml.models import MODEL_SCHEMA_VERSION, ModelConfig

    path = Path(weights)
    directory = path if path.is_dir() else path.parent
    manifest_path = directory / "run-manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"unversioned policy weights in {directory}; regenerate them with the v4 pipeline"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "policy-value"
        or manifest.get("model_schema") != MODEL_SCHEMA_VERSION
        or manifest.get("target_schema") != POLICY_TARGET_SCHEMA
    ):
        raise ValueError(
            f"incompatible policy checkpoint contract in {manifest_path}; use v4 weights"
        )
    config_path = directory / "config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = ModelConfig(**raw.get("model", {}))
    else:
        config = ModelConfig()
    return config, directory / "policy.weights.h5" if path.is_dir() else path


def _arena_policy(
    args,
    environment: ShopEnvironment,
    *,
    battle: BattleSimulator | None = None,
    population: OpponentPopulation | None = None,
):
    if args.policy == "heuristic":
        return HeuristicPolicy()
    if args.policy == "random":
        return RandomPolicy()

    from sapai.search.stochastic import PolicyGuidedSearch, SearchConfig, UniformEvaluator

    evaluator = UniformEvaluator()
    if args.policy_weights:
        import tensorflow as tf

        from sapai.ml.encoding import encode_states
        from sapai.ml.models import PolicyValueModel
        from sapai.search.tensorflow_evaluator import TensorFlowEvaluator

        config, weights = _model_config_from_weights(args.policy_weights)
        model = PolicyValueModel(config)
        state = environment.reset(pack=args.pack, seed=args.seed)
        actions = environment.legal_actions(state)
        encoded = encode_states([state], [actions])
        model(
            {key: tf.convert_to_tensor(value) for key, value in encoded.as_dict().items()},
            training=False,
        )
        model.load_weights(weights)
        evaluator = TensorFlowEvaluator(model)
    elif args.policy == "model":
        raise ValueError("--policy-weights is required for model policy")

    if args.policy == "model":
        return ModelPolicy(evaluator)
    if battle is None or population is None:
        raise ValueError("search policy requires a battle simulator and opponent population")
    battle_evaluator = SimulatorPopulationEvaluator(
        environment,
        battle,
        population,
        simulations=args.battle_evaluation_simulations,
        continuation_evaluator=evaluator,
    )
    search = PolicyGuidedSearch(
        environment,
        evaluator,
        SearchConfig(
            simulations=args.search_simulations,
            candidate_actions=args.search_candidates,
            battle_initial_simulations=min(4, args.battle_evaluation_simulations),
            battle_max_simulations=args.battle_evaluation_simulations,
        ),
        battle_evaluator=battle_evaluator,
    )
    return SearchPolicy(search)


def _runner(args, catalog: Catalog) -> ArenaRunner:
    environment = ShopEnvironment(catalog)
    battle = BattleSimulator(catalog)
    population = _population(args, catalog)
    return ArenaRunner(
        environment,
        battle,
        population,
        _arena_policy(args, environment, battle=battle, population=population),
        max_decisions_per_turn=args.max_decisions_per_turn,
    )


def _training_config(args, *, epochs: int | None = None):
    from sapai.ml.pipelines import TrainingConfig

    return TrainingConfig(
        epochs=epochs if epochs is not None else args.epochs,
        batch_size=args.batch_size,
        learning_rate=getattr(args, "learning_rate", 3e-4),
        seed=args.seed,
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "sapai").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _catalog_sha256(catalog: Catalog) -> str:
    payload = {
        "pets": [asdict(value) for _, value in sorted(catalog.pets.items())],
        "foods": [asdict(value) for _, value in sorted(catalog.foods.items())],
        "perks": sorted(catalog.perks.items()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _repository_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sequence_manifest_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable contract shared by every continuation generation."""

    settings = manifest.get("settings")
    if not isinstance(settings, Mapping):
        raise TypeError("training sequence manifest has no settings contract")
    immutable_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"search_iterations", "seed"}
    }
    return {
        "objective": manifest.get("objective"),
        "target_schema": manifest.get("target_schema"),
        "boards_sha256": manifest.get("boards_sha256"),
        "catalog_sha256": manifest.get("catalog_sha256"),
        "rules_sha256": manifest.get("rules_sha256"),
        "simulator": manifest.get("simulator"),
        "population_seed": manifest.get("population_seed", settings.get("seed")),
        "settings": immutable_settings,
    }


def _contract_differences(
    previous: object,
    current: object,
    *,
    path: str = "",
) -> list[str]:
    """Describe nested manifest differences without hiding the incompatible field."""

    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(previous) | set(current)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in previous:
                differences.append(
                    f"{child_path}: previous=<missing>, current={current[key]!r}"
                )
            elif key not in current:
                differences.append(
                    f"{child_path}: previous={previous[key]!r}, current=<missing>"
                )
            else:
                differences.extend(
                    _contract_differences(
                        previous[key],
                        current[key],
                        path=child_path,
                    )
                )
        return differences
    if previous != current:
        return [f"{path}: previous={previous!r}, current={current!r}"]
    return []


def _completed_iteration_records(root: Path) -> list[dict[str, object]]:
    """Load the longest contiguous set of atomically completed iterations."""

    by_iteration: dict[int, dict[str, object]] = {}
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary.get("iterations", []):
            if isinstance(record, dict) and isinstance(record.get("iteration"), int):
                by_iteration[int(record["iteration"])] = record

    state_dir = root / "iteration-state"
    if state_dir.exists():
        for path in sorted(state_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("iteration"), int):
                raise TypeError(f"invalid completed iteration record: {path}")
            by_iteration[int(record["iteration"])] = record

    records: list[dict[str, object]] = []
    for iteration in range(1, max(by_iteration, default=0) + 1):
        if iteration not in by_iteration:
            raise ValueError(f"training iteration history has a gap before iteration {iteration}")
        records.append(by_iteration[iteration])
    return records


def _prepare_sequence_manifest(
    path: Path,
    desired: dict[str, object],
    *,
    requested_iterations: int,
    completed_iterations: int,
) -> dict[str, object]:
    """Create, migrate, or extend a sequence manifest without weakening its contract."""

    if requested_iterations < 1:
        raise ValueError("search iterations must be positive")
    if requested_iterations < completed_iterations:
        raise ValueError(
            f"this workdir already completed {completed_iterations} search iterations; "
            f"--search-iterations cannot be reduced to {requested_iterations}"
        )

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_format = existing.get("format")
        supported = {TRAINING_SEQUENCE_FORMAT, *LEGACY_TRAINING_SEQUENCE_FORMATS}
        if existing_format not in supported:
            raise ValueError(f"unsupported training sequence format in {path}")
        existing_contract = _sequence_manifest_contract(existing)
        desired_for_contract = dict(desired)
        desired_for_contract.setdefault(
            "population_seed", existing_contract["population_seed"]
        )
        desired_contract = _sequence_manifest_contract(desired_for_contract)
        differences = _contract_differences(existing_contract, desired_contract)
        if differences:
            formatted = "\n".join(f"  - {difference}" for difference in differences)
            raise ValueError(
                f"training sequence inputs or immutable settings changed in {path.parent}; "
                f"use a new workdir or restore the original inputs/settings:\n{formatted}"
            )
        budget = existing.get("continuation")
        prior_request = int(
            budget.get("requested_search_iterations", 0)
            if isinstance(budget, Mapping)
            else existing["settings"].get("search_iterations", 0)
        )
        if requested_iterations < prior_request:
            raise ValueError(
                f"this workdir already targets {prior_request} search iterations; "
                f"--search-iterations cannot be reduced to {requested_iterations}"
            )
        manifest = dict(existing)
        manifest["format"] = TRAINING_SEQUENCE_FORMAT
        manifest.setdefault("population_seed", existing_contract["population_seed"])
        manifest.setdefault("created_with", {
            "source_sha256": existing.get("source_sha256"),
            "repository_commit": existing.get("repository_commit"),
        })
    else:
        manifest = dict(desired)
        manifest["format"] = TRAINING_SEQUENCE_FORMAT
        manifest["created_with"] = {
            "source_sha256": desired.get("source_sha256"),
            "repository_commit": desired.get("repository_commit"),
        }

    versions = manifest.setdefault("code_versions", [])
    created_version = manifest["created_with"]
    if created_version not in versions:
        versions.append(created_version)
    code_version = {
        "source_sha256": desired.get("source_sha256"),
        "repository_commit": desired.get("repository_commit"),
    }
    if code_version not in versions:
        versions.append(code_version)
    run_seeds = manifest.setdefault("run_seeds", [])
    initial_seed = manifest["settings"].get("seed")
    if initial_seed not in run_seeds:
        run_seeds.append(initial_seed)
    requested_seed = desired["settings"].get("seed")
    if requested_seed not in run_seeds:
        run_seeds.append(requested_seed)
    initial_request = int(manifest["settings"].get("search_iterations", requested_iterations))
    manifest["continuation"] = {
        "initial_search_iterations": int(
            manifest.get("continuation", {}).get(
                "initial_search_iterations", initial_request
            )
        ),
        "requested_search_iterations": requested_iterations,
        "completed_search_iterations": completed_iterations,
        "current_seed": requested_seed,
    }
    _atomic_json(path, manifest)
    return manifest


def _generate_episode_dataset(
    runner: ArenaRunner,
    output: Path,
    *,
    episode_dir: Path,
    episodes: int,
    pack: str,
    seed: int,
    version: str = "current",
    identity: dict[str, object],
    progress: Callable[[int, int, bool], None] | None = None,
) -> int:
    """Checkpoint each rollout separately, then assemble the training JSONL."""

    episode_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = episode_dir / "manifest.json"
    manifest = {
        "format": "sapai-arena-episodes-v4",
        "target_schema": POLICY_TARGET_SCHEMA,
        "pack": pack,
        "version": version,
        "seed": seed,
        **identity,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Arena episode settings changed in {episode_dir}; use a new workdir"
            )
    else:
        if any(episode_dir.glob("*.jsonl")):
            raise ValueError(f"Arena episodes have no manifest: {episode_dir}")
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    paths: list[Path] = []
    progress_interval = max(1, episodes // 20)
    for episode in range(episodes):
        episode_path = episode_dir / f"{episode:06d}.jsonl"
        paths.append(episode_path)
        reused = episode_path.exists()
        if episode_path.exists():
            if not read_arena_decisions(episode_path):
                raise ValueError(f"checkpointed Arena episode is empty: {episode_path}")
        else:
            decisions = runner.run(
                pack=pack,
                version=version,
                seed=seed + episode,
            ).decisions
            temporary = episode_path.with_suffix(".jsonl.tmp")
            write_arena_decisions(temporary, decisions)
            temporary.replace(episode_path)
        completed = episode + 1
        if progress is not None and (
            completed == 1 or completed == episodes or completed % progress_interval == 0
        ):
            progress(completed, episodes, reused)

    combined = []
    for episode_path in paths:
        combined.extend(read_arena_decisions(episode_path))
    return write_arena_decisions(output, combined)


def _write_replay_mixture(
    output: Path,
    *,
    bootstrap_path: Path,
    search_paths: list[Path],
    bootstrap_fraction: float,
    seed: int,
) -> int:
    """Mix bootstrap coverage with the latest and prior search trajectories."""

    if not 0.0 <= bootstrap_fraction < 1.0:
        raise ValueError("bootstrap replay fraction must be in [0, 1)")
    if not search_paths:
        raise ValueError("at least one search dataset is required")
    rng = random.Random(seed)
    bootstrap = read_arena_decisions(bootstrap_path)
    latest = read_arena_decisions(search_paths[-1])
    older = [
        decision
        for path in search_paths[:-1]
        for decision in read_arena_decisions(path)
    ]
    if not latest:
        raise ValueError("latest search replay dataset is empty")
    if older:
        older_count = min(len(older), len(latest))
        replay = latest + rng.sample(older, older_count)
    else:
        replay = list(latest)
    bootstrap_count = round(len(replay) * bootstrap_fraction / (1.0 - bootstrap_fraction))
    if bootstrap_count and not bootstrap:
        raise ValueError("bootstrap replay dataset is empty")
    if bootstrap_count <= len(bootstrap):
        replay.extend(rng.sample(bootstrap, bootstrap_count))
    else:
        replay.extend(rng.choice(bootstrap) for _ in range(bootstrap_count))
    rng.shuffle(replay)
    return write_arena_decisions(output, replay)


def _battle_dataset_for_sequence(
    boards,
    output: Path,
    simulator: BattleSimulator,
    *,
    examples: int,
    simulations_per_pair: int,
    seed: int,
    pack: str,
    boards_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    from sapai.data.datasets import build_battle_dataset

    identity = {
        "format": "sapai-sequence-battle-input-v1",
        "boards_sha256": boards_sha256,
        "pack": pack,
        "examples": examples,
        "simulations_per_pair": simulations_per_pair,
        "seed": seed,
    }
    identity_path = output / "sequence-input.json"
    manifest_path = output / "manifest.json"
    split_paths = [output / f"{name}.jsonl" for name in ("train", "validation", "test")]
    if manifest_path.exists() and all(path.exists() for path in split_paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("examples", {})
        legacy_matches = (
            manifest.get("format") == "sapai-battle-dataset-v1"
            and manifest.get("seed") == seed
            and manifest.get("simulations_per_pair") == simulations_per_pair
            and sum(int(counts.get(name, 0)) for name in ("train", "validation", "test"))
            == examples
        )
        if not legacy_matches:
            raise ValueError(f"battle dataset settings changed in {output}; use a new workdir")
        if identity_path.exists():
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing_identity != identity:
                raise ValueError(
                    f"battle dataset input changed in {output}; use a new workdir"
                )
        else:
            identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        return manifest

    manifest = build_battle_dataset(
        boards,
        output,
        simulator,
        examples=examples,
        simulations_per_pair=simulations_per_pair,
        seed=seed,
        progress=progress,
    )
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return manifest


def _training_progress_logger() -> Callable[[str, Mapping[str, object]], None]:
    started = time.monotonic()

    def report(event: str, payload: Mapping[str, object]) -> None:
        elapsed = max(0, round(time.monotonic() - started))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        stage = str(payload.get("stage", "sequence"))
        message = str(payload.get("message", event.replace("_", " ")))
        parts = [f"[progress {hours:02d}:{minutes:02d}:{seconds:02d}]", stage, message]
        completed = payload.get("completed")
        total = payload.get("total")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            percent = min(100.0, max(0.0, completed * 100.0 / total))
            parts.append(f"{completed}/{total} ({percent:.1f}%)")
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping):
            formatted = [
                f"{key}={float(value):.5f}"
                for key, value in sorted(metrics.items())
                if key != "epoch" and isinstance(value, (int, float))
            ]
            if formatted:
                parts.append(" ".join(formatted))
        checkpoint = payload.get("checkpoint") or payload.get("restored_checkpoint")
        if checkpoint:
            parts.append(f"checkpoint={checkpoint}")
        print(" | ".join(parts), flush=True)

    return report


def _policy_evaluation_score(evaluation: Mapping[str, object]) -> tuple[float, ...]:
    """Rank checkpoints by model play, then search play, rejecting roll collapse."""

    model = evaluation["model"]
    search = evaluation["search"]
    if not isinstance(model, Mapping) or not isinstance(search, Mapping):
        raise TypeError("invalid policy evaluation")
    behavior = model["shop_behavior"]
    if not isinstance(behavior, Mapping):
        raise TypeError("invalid model shop-behavior evaluation")
    return (
        float(model["completion_rate"]),
        float(model["mean_trophies"]),
        -float(behavior["shop_collapse_penalty"]),
        float(search["completion_rate"]),
        float(search["mean_trophies"]),
    )


def _run_training_sequence(args, catalog: Catalog) -> dict[str, object]:
    from sapai.ml.pipelines import train_policy_model

    progress = _training_progress_logger() if getattr(args, "progress", False) else None

    def emit(stage: str, message: str, **values: object) -> None:
        if progress is not None:
            progress("stage", {"stage": stage, "message": message, **values})

    def training_progress(stage: str):
        if progress is None:
            return None

        def report(event: str, payload: Mapping[str, object]) -> None:
            message = (
                "epoch checkpoint saved"
                if event == "epoch_completed"
                else "training initialized"
            )
            progress(event, {"stage": stage, "message": message, **payload})

        return report

    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    completed_iteration_records = _completed_iteration_records(root)
    completed_iterations = len(completed_iteration_records)
    emit("sequence", f"starting iterative policy improvement in {root}")
    boards = load_opponent_boards(args.boards, catalog, args.pack)
    boards_sha256 = _file_sha256(args.boards)
    simulator = BattleSimulator(catalog)
    sequence_manifest_path = root / "sequence-manifest.json"
    population_seed = args.seed
    if sequence_manifest_path.exists():
        stored_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
        stored_settings = stored_manifest.get("settings")
        if not isinstance(stored_settings, Mapping):
            raise TypeError("training sequence manifest has no settings contract")
        population_seed = int(
            stored_manifest.get("population_seed", stored_settings.get("seed", args.seed))
        )
    sequence_manifest = {
        "format": TRAINING_SEQUENCE_FORMAT,
        "objective": VALUE_OBJECTIVE,
        "target_schema": POLICY_TARGET_SCHEMA,
        "boards_sha256": boards_sha256,
        "source_sha256": _source_sha256(PROJECT_ROOT),
        "repository_commit": _repository_commit(PROJECT_ROOT),
        "catalog_sha256": _catalog_sha256(catalog),
        "rules_sha256": hashlib.sha256(
            json.dumps(
                simulator.rules.data,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "simulator": {
            "max_rounds": simulator.MAX_ROUNDS,
            "max_events": simulator.MAX_EVENTS,
            "record_training_trace": False,
        },
        "population_seed": population_seed,
        "settings": {
            key: getattr(args, key)
            for key in (
                "pack",
                "board_version",
                "validation_fraction",
                "test_fraction",
                "validation_episodes",
                "test_episodes",
                "bootstrap_episodes",
                "bootstrap_epochs",
                "bootstrap_exploration",
                "search_episodes",
                "search_epochs",
                "search_iterations",
                "bootstrap_replay_fraction",
                "batch_size",
                "seed",
                "search_simulations",
                "search_candidates",
                "battle_evaluation_simulations",
            )
        },
    }
    if not sequence_manifest_path.exists():
        existing_outputs = [
            path for path in root.iterdir() if path.name != sequence_manifest_path.name
        ]
        if existing_outputs:
            raise ValueError(
                f"legacy or unversioned training outputs found in {root}; "
                "use a new versioned workdir"
            )
    _prepare_sequence_manifest(
        sequence_manifest_path,
        sequence_manifest,
        requested_iterations=args.search_iterations,
        completed_iterations=completed_iterations,
    )
    if completed_iterations:
        emit(
            "sequence",
            "continuing completed training history",
            completed=completed_iterations,
            total=args.search_iterations,
        )
    populations = split_opponent_populations(
        boards,
        version=args.board_version,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=population_seed,
    )
    emit(
        "population",
        (
            f"prepared replay-safe populations for patch {populations.version}; "
            f"split seed={population_seed}, run seed={args.seed}"
        ),
    )

    bootstrap_path = root / "arena-bootstrap.jsonl"
    bootstrap_runner = ArenaRunner(
        ShopEnvironment(catalog),
        BattleSimulator(catalog),
        populations.train,
        MixturePolicy(
            HeuristicPolicy(),
            RandomPolicy(),
            exploration_probability=args.bootstrap_exploration,
        ),
        record_timeline=False,
    )
    emit("bootstrap-rollouts", "generating or reusing exploratory bootstrap episodes")
    bootstrap_decisions = _generate_episode_dataset(
        bootstrap_runner,
        bootstrap_path,
        episode_dir=root / "arena-bootstrap-episodes",
        episodes=args.bootstrap_episodes,
        pack=args.pack,
        version=populations.version,
        seed=population_seed,
        identity={
            "policy": "heuristic-exploration-mixture",
            "exploration_probability": args.bootstrap_exploration,
            "boards_sha256": boards_sha256,
            "board_version": populations.version,
            "board_counts": populations.board_counts,
            "episodes": args.bootstrap_episodes,
            "target_schema": POLICY_TARGET_SCHEMA,
        },
        progress=(
            lambda completed, total, reused: emit(
                "bootstrap-rollouts",
                "episode checkpoint reused" if reused else "episode checkpoint saved",
                completed=completed,
                total=total,
            )
        )
        if progress is not None
        else None,
    )
    policy_dir = root / "policy-model"
    prior_summary_path = root / "summary.json"
    prior_summary = (
        json.loads(prior_summary_path.read_text(encoding="utf-8"))
        if prior_summary_path.exists()
        else {}
    )
    if completed_iterations and isinstance(prior_summary.get("bootstrap_training"), dict):
        bootstrap_summary = prior_summary["bootstrap_training"]
        emit("bootstrap-training", "reusing the completed bootstrap checkpoint")
    else:
        bootstrap_summary = train_policy_model(
            bootstrap_path,
            policy_dir,
            training_config=_training_config(args, epochs=args.bootstrap_epochs),
            progress=training_progress("bootstrap-training"),
        )

    evaluation_dir = root / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    best_weights = policy_dir / "best-policy.weights.h5"
    best_path = evaluation_dir / "best.json"
    current_source_hash = str(sequence_manifest["source_sha256"])
    evaluation_contract_hash = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": current_source_hash,
                "population_seed": population_seed,
                "run_seed": args.seed,
                "validation_episodes": args.validation_episodes,
                "test_episodes": args.test_episodes,
                "search_simulations": args.search_simulations,
                "search_candidates": args.search_candidates,
                "battle_evaluation_simulations": args.battle_evaluation_simulations,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    def evaluate_checkpoint(
        population: OpponentPopulation,
        *,
        weights: Path,
        episodes: int,
        seed: int,
        include_heuristic: bool,
    ) -> dict[str, object]:
        evaluated: dict[str, object] = {}
        if include_heuristic:
            evaluated["heuristic"] = evaluate_arena_policy(
                ArenaRunner(
                    ShopEnvironment(catalog),
                    BattleSimulator(catalog),
                    population,
                    HeuristicPolicy(),
                    record_timeline=False,
                ),
                episodes=episodes,
                pack=args.pack,
                version=populations.version,
                seed=seed,
            )
        for policy_name in ("model", "search"):
            environment = ShopEnvironment(catalog)
            battle = BattleSimulator(catalog)
            policy_args = argparse.Namespace(
                policy=policy_name,
                policy_weights=str(weights),
                pack=args.pack,
                seed=seed,
                search_simulations=args.search_simulations,
                search_candidates=args.search_candidates,
                battle_evaluation_simulations=args.battle_evaluation_simulations,
            )
            evaluated[policy_name] = evaluate_arena_policy(
                ArenaRunner(
                    environment,
                    battle,
                    population,
                    _arena_policy(
                        policy_args,
                        environment,
                        battle=battle,
                        population=population,
                    ),
                    record_timeline=False,
                ),
                episodes=episodes,
                pack=args.pack,
                version=populations.version,
                seed=seed,
            )
        return evaluated

    bootstrap_evaluation_path = (
        evaluation_dir / f"bootstrap-{evaluation_contract_hash[:12]}.json"
    )
    historical_bootstrap_paths = list(evaluation_dir.glob("bootstrap*.json"))
    if bootstrap_evaluation_path.exists():
        bootstrap_evaluation = json.loads(
            bootstrap_evaluation_path.read_text(encoding="utf-8")
        )
    elif best_path.exists() and historical_bootstrap_paths:
        historical_bootstrap_path = min(historical_bootstrap_paths)
        bootstrap_evaluation = json.loads(
            historical_bootstrap_path.read_text(encoding="utf-8")
        )
    else:
        bootstrap_evaluation = evaluate_checkpoint(
            populations.validation,
            weights=policy_dir / "policy.weights.h5",
            episodes=args.validation_episodes,
            seed=args.seed + 500_000,
            include_heuristic=True,
        )
        _atomic_json(bootstrap_evaluation_path, bootstrap_evaluation)
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if not best_weights.exists() or _file_sha256(best_weights) != best.get(
            "weights_sha256"
        ):
            raise ValueError("best checkpoint is missing or does not match evaluations/best.json")
    else:
        _atomic_copy(policy_dir / "policy.weights.h5", best_weights)
        best = {
            "stage": "bootstrap",
            "score": list(_policy_evaluation_score(bootstrap_evaluation)),
            "weights_sha256": _file_sha256(best_weights),
            "evaluation_source_sha256": sequence_manifest["source_sha256"],
            "evaluation_contract_sha256": evaluation_contract_hash,
        }
        _atomic_json(best_path, best)

    if best.get("evaluation_contract_sha256") != evaluation_contract_hash:
        incumbent_path = evaluation_dir / (
            f"incumbent-{evaluation_contract_hash[:12]}-"
            f"{str(best['weights_sha256'])[:16]}.json"
        )
        if incumbent_path.exists():
            incumbent_evaluation = json.loads(incumbent_path.read_text(encoding="utf-8"))
        else:
            emit(
                "validation",
                "rebasing the incumbent checkpoint under the current code and seed",
            )
            incumbent_evaluation = evaluate_checkpoint(
                populations.validation,
                weights=best_weights,
                episodes=args.validation_episodes,
                seed=args.seed + 500_000,
                include_heuristic=False,
            )
            _atomic_json(incumbent_path, incumbent_evaluation)
        best = {
            **best,
            "score": list(_policy_evaluation_score(incumbent_evaluation)),
            "evaluation_source_sha256": current_source_hash,
            "evaluation_contract_sha256": evaluation_contract_hash,
        }
        _atomic_json(best_path, best)

    search_paths = [
        root / f"arena-search-{iteration:02d}.jsonl"
        for iteration in range(1, completed_iterations + 1)
    ]
    for path in search_paths:
        if not path.exists():
            raise ValueError(f"completed training history is missing {path}")
    iterations = list(completed_iteration_records)
    for iteration in range(completed_iterations + 1, args.search_iterations + 1):
        stage = f"search-{iteration}"
        search_path = root / f"arena-search-{iteration:02d}.jsonl"
        search_paths.append(search_path)
        environment = ShopEnvironment(catalog)
        battle = BattleSimulator(catalog)
        policy_args = argparse.Namespace(
            policy="search",
            policy_weights=str(best_weights),
            pack=args.pack,
            seed=args.seed,
            search_simulations=args.search_simulations,
            search_candidates=args.search_candidates,
            battle_evaluation_simulations=args.battle_evaluation_simulations,
        )
        runner = ArenaRunner(
            environment,
            battle,
            populations.train,
            _arena_policy(
                policy_args,
                environment,
                battle=battle,
                population=populations.train,
            ),
            record_timeline=False,
        )
        emit(stage, "generating or reusing search-guided episodes")
        search_decisions = _generate_episode_dataset(
            runner,
            search_path,
            episode_dir=root / f"arena-search-{iteration:02d}-episodes",
            episodes=args.search_episodes,
            pack=args.pack,
            version=populations.version,
            seed=args.seed + iteration * 100_000,
            identity={
                "policy": "search",
                "iteration": iteration,
                "teacher_weights_sha256": _file_sha256(best_weights),
                "boards_sha256": boards_sha256,
                "board_version": populations.version,
                "episodes": args.search_episodes,
                "search_simulations": args.search_simulations,
                "search_candidates": args.search_candidates,
                "battle_evaluation_simulations": args.battle_evaluation_simulations,
                "target_schema": POLICY_TARGET_SCHEMA,
            },
            progress=(
                lambda completed, total, reused, name=stage: emit(
                    name,
                    "episode checkpoint reused" if reused else "episode checkpoint saved",
                    completed=completed,
                    total=total,
                )
            )
            if progress is not None
            else None,
        )
        replay_path = root / f"policy-replay-{iteration:02d}.jsonl"
        replay_decisions = _write_replay_mixture(
            replay_path,
            bootstrap_path=bootstrap_path,
            search_paths=search_paths,
            bootstrap_fraction=args.bootstrap_replay_fraction,
            seed=args.seed + iteration,
        )
        target_epochs = args.bootstrap_epochs + iteration * args.search_epochs
        training_summary = train_policy_model(
            replay_path,
            policy_dir,
            training_config=_training_config(args, epochs=target_epochs),
            resume=True,
            progress=training_progress(f"distillation-{iteration}"),
        )
        evaluation_path = evaluation_dir / (
            f"iteration-{iteration:02d}-{evaluation_contract_hash[:12]}.json"
        )
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        else:
            evaluation = evaluate_checkpoint(
                populations.validation,
                weights=policy_dir / "policy.weights.h5",
                episodes=args.validation_episodes,
                seed=args.seed + 500_000,
                include_heuristic=False,
            )
            _atomic_json(evaluation_path, evaluation)
        candidate_score = _policy_evaluation_score(evaluation)
        best_score = tuple(float(item) for item in best["score"])
        already_promoted = best.get("stage") == f"iteration-{iteration}"
        promoted = already_promoted or candidate_score > best_score
        if promoted and not already_promoted:
            _atomic_copy(policy_dir / "policy.weights.h5", best_weights)
            best = {
                "stage": f"iteration-{iteration}",
                "score": list(candidate_score),
                "weights_sha256": _file_sha256(best_weights),
                "evaluation_source_sha256": current_source_hash,
                "evaluation_contract_sha256": evaluation_contract_hash,
            }
            _atomic_json(best_path, best)
        iteration_record = {
            "iteration": iteration,
            "search_decisions": search_decisions,
            "replay_decisions": replay_decisions,
            "training": training_summary,
            "validation": evaluation,
            "promoted": promoted,
            "best_after_iteration": dict(best),
        }
        _atomic_json(
            root / "iteration-state" / f"{iteration:06d}.json",
            iteration_record,
        )
        iterations.append(iteration_record)
        _prepare_sequence_manifest(
            sequence_manifest_path,
            sequence_manifest,
            requested_iterations=args.search_iterations,
            completed_iterations=iteration,
        )
        emit(stage, "iteration evaluated and checkpoint gate applied")

    _atomic_copy(best_weights, policy_dir / "policy.weights.h5")
    best_hash = str(best["weights_sha256"])
    final_test_path = evaluation_dir / (
        f"test-{evaluation_contract_hash[:12]}-{best_hash[:16]}.json"
    )
    if final_test_path.exists():
        final_test = json.loads(final_test_path.read_text(encoding="utf-8"))
    else:
        legacy_test_path = evaluation_dir / "test.json"
        if (
            legacy_test_path.exists()
            and prior_summary.get("best", {}).get("weights_sha256") == best_hash
            and prior_summary.get("evaluation_contract_sha256")
            == evaluation_contract_hash
        ):
            final_test = json.loads(legacy_test_path.read_text(encoding="utf-8"))
        else:
            final_test = evaluate_checkpoint(
                populations.test,
                weights=best_weights,
                episodes=args.test_episodes,
                seed=args.seed + 600_000,
                include_heuristic=True,
            )
        _atomic_json(final_test_path, final_test)
    summary = {
        "format": TRAINING_SEQUENCE_FORMAT,
        "objective": VALUE_OBJECTIVE,
        "target_schema": POLICY_TARGET_SCHEMA,
        "boards_sha256": boards_sha256,
        "source_sha256": current_source_hash,
        "evaluation_contract_sha256": evaluation_contract_hash,
        "run_seed": args.seed,
        "population_split": {
            "version": populations.version,
            "counts": populations.board_counts,
            "unit": "replay_id",
            "seed": population_seed,
        },
        "battle_evaluation": {
            "kind": "native-simulator-common-panel",
            "max_simulations_per_leaf": args.battle_evaluation_simulations,
        },
        "bootstrap_decisions": bootstrap_decisions,
        "bootstrap_training": bootstrap_summary,
        "bootstrap_validation": bootstrap_evaluation,
        "requested_search_iterations": args.search_iterations,
        "completed_search_iterations": len(iterations),
        "iterations": iterations,
        "best": best,
        "final_test": final_test,
    }
    _atomic_json(root / "summary.json", summary)
    emit("sequence", f"full training complete; summary saved to {root / 'summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = _catalog(args.data)
    if args.command == "catalog-report":
        environment = ShopEnvironment(catalog)
        pets = catalog.pack_pets(args.pack)
        battle_supported = BattleSimulator.SUPPORTED_TURTLE_PETS
        shop_supported = environment.abilities.implemented_pets
        _json(
            {
                "pack": args.pack,
                "pets": len(pets),
                "battle_supported": sum(pet.name in battle_supported for pet in pets),
                "shop_ability_supported": sum(pet.name in shop_supported for pet in pets),
                "battle_missing": [pet.name for pet in pets if pet.name not in battle_supported],
                "rules_source": BattleSimulator(catalog).rules.source,
            }
        )
    elif args.command == "shop-demo":
        state = ShopEnvironment(catalog).reset(pack=args.pack, seed=args.seed)
        _json(
            {
                "pets": [offer.pet.name for offer in state.shop.pets],
                "foods": [food.name for food in state.shop.foods],
                "legal_actions": len(ShopEnvironment(catalog).legal_actions(state)),
            }
        )
    elif args.command in {"battle", "visualize-battle"}:
        result = BattleSimulator(catalog).simulate(
            _team(catalog, args.player),
            _team(catalog, args.opponent),
            seed=args.seed,
        )
        if args.command == "battle":
            _json({"outcome": result.outcome.value, "rounds": result.rounds, "log": result.log})
        else:
            from sapai.visualization import render_battle_html

            output = render_battle_html(result, args.output, args.assets)
            _json(
                {
                    "output": str(output),
                    "stylesheet": str(output.with_name("sapai.css")),
                    "runtime": str(output.with_name("sapai.js")),
                    "assets": str(output.with_name("sapai-assets")),
                    "frames": len(result.frames),
                }
            )
    elif args.command == "model-smoke":
        from sapai.ml.smoke import run_model_smoke

        _json(run_model_smoke(catalog))
    elif args.command in {"library-sample", "export-boards"}:
        replay_parser = ReplayParser(catalog)
        boards = SapLibraryClient(replay_parser).sample_boards(
            pack=args.pack,
            turn=args.turn,
            limit=args.limit,
        )
        if args.command == "export-boards":
            if not boards:
                raise ValueError(
                    f"database sample produced no non-empty compatible {args.pack!r} boards"
                )
            _json({"output": args.output, "boards": write_boards(args.output, boards)})
        else:
            _json(
                [
                    {
                        "replay_id": board.replay_id,
                        "side": board.side,
                        "turn": board.turn,
                        "pack": board.pack,
                        "pets": [pet.name if pet else None for pet in board.team.slots],
                    }
                    for board in boards
                ]
            )
    elif args.command == "parse-replays":
        boards = read_replay_jsonl(args.input, ReplayParser(catalog))
        _json({"output": args.output, "boards": write_boards(args.output, boards)})
    elif args.command == "label-battles":
        from sapai.data.datasets import build_battle_dataset

        manifest = build_battle_dataset(
            list(read_boards(args.boards)),
            args.output,
            BattleSimulator(catalog),
            examples=args.examples,
            simulations_per_pair=args.simulations_per_pair,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        _json(manifest)
    elif args.command == "train-battle":
        from sapai.ml.pipelines import train_battle_model

        _json(
            train_battle_model(
                args.dataset,
                args.output,
                training_config=_training_config(args),
                resume=not args.no_resume,
            )
        )
    elif args.command == "cache-population":
        population = OpponentPopulation(list(read_boards(args.boards)))
        population.save_encoded_cache(args.output)
        _json({"output": args.output, "boards": len(population.boards)})
    elif args.command == "generate-arena":
        runner = _runner(args, catalog)
        decisions = []
        results = []
        for episode in range(args.episodes):
            run = runner.run(
                pack=args.pack,
                version=args.resolved_board_version,
                seed=args.seed + episode,
            )
            decisions.extend(run.decisions)
            results.append({"trophies": run.final_state.trophies, "turns": len(run.turns)})
        count = write_arena_decisions(args.output, decisions)
        _json({"output": args.output, "decisions": count, "episodes": results})
    elif args.command == "train-policy":
        from sapai.ml.pipelines import train_policy_model

        _json(
            train_policy_model(
                args.dataset,
                args.output,
                validation_path=args.validation,
                training_config=_training_config(args),
                resume=not args.no_resume,
            )
        )
    elif args.command == "visualize-arena":
        from sapai.visualization import render_arena_html

        runner = _runner(args, catalog)
        run = runner.run(
            pack=args.pack,
            version=args.resolved_board_version,
            seed=args.seed,
        )
        output = render_arena_html(run, args.output, args.assets)
        _json(
            {
                "output": str(output),
                "stylesheet": str(output.with_name("sapai.css")),
                "runtime": str(output.with_name("sapai.js")),
                "assets": str(output.with_name("sapai-assets")),
                "turns": len(run.turns),
                "trophies": run.final_state.trophies,
            }
        )
    elif args.command == "train-sequence":
        _json(_run_training_sequence(args, catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
