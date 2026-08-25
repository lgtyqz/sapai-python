from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from sapai.data.library import SapLibraryClient, read_replay_jsonl
from sapai.data.replay import ReplayParser, board_is_pack_compatible
from sapai.data.serialization import read_boards, write_boards
from sapai.sim.battle import BattleSimulator
from sapai.sim.catalog import PACK_ALIASES, Catalog
from sapai.sim.models import Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.arena import (
    ArenaRunner,
    HeuristicPolicy,
    ModelPolicy,
    RandomPolicy,
    SearchPolicy,
    read_arena_decisions,
    write_arena_decisions,
)
from sapai.training.population import OpponentPopulation

if TYPE_CHECKING:
    from sapai.ml.models import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions-per-turn", type=int, default=30)
    parser.add_argument("--search-simulations", type=int, default=32)
    parser.add_argument("--search-candidates", type=int, default=8)


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
        "train-sequence", help="run labels, battle training, bootstrap, and search distillation"
    )
    sequence.add_argument("--boards", required=True)
    sequence.add_argument("--workdir", required=True)
    sequence.add_argument("--pack", default="Turtle", choices=("Turtle",))
    sequence.add_argument("--battle-examples", type=int, default=10_000)
    sequence.add_argument("--simulations-per-pair", type=int, default=8)
    sequence.add_argument("--battle-epochs", type=int, default=10)
    sequence.add_argument("--bootstrap-episodes", type=int, default=100)
    sequence.add_argument("--bootstrap-epochs", type=int, default=10)
    sequence.add_argument("--search-episodes", type=int, default=50)
    sequence.add_argument("--search-epochs", type=int, default=5)
    sequence.add_argument("--batch-size", type=int, default=64)
    sequence.add_argument("--seed", type=int, default=0)
    sequence.add_argument("--search-simulations", type=int, default=32)
    sequence.add_argument("--search-candidates", type=int, default=8)
    return parser


def _population(args, catalog: Catalog) -> OpponentPopulation:
    if getattr(args, "boards", None):
        boards = [
            board
            for board in read_boards(args.boards)
            if board_is_pack_compatible(board, catalog, args.pack)
        ]
        if not boards:
            raise ValueError(f"board dataset contains no compatible {args.pack!r} boards")
        return OpponentPopulation(boards)
    return OpponentPopulation.synthetic(catalog, pack=args.pack, seed=args.seed)


def _model_config_from_weights(weights: str | Path) -> tuple[ModelConfig, Path]:
    from sapai.ml.models import ModelConfig

    path = Path(weights)
    directory = path if path.is_dir() else path.parent
    config_path = directory / "config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = ModelConfig(**raw.get("model", {}))
    else:
        config = ModelConfig()
    return config, directory / "policy.weights.h5" if path.is_dir() else path


def _arena_policy(args, environment: ShopEnvironment):
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
    search = PolicyGuidedSearch(
        environment,
        evaluator,
        SearchConfig(
            simulations=args.search_simulations,
            candidate_actions=args.search_candidates,
        ),
    )
    return SearchPolicy(search)


def _runner(args, catalog: Catalog) -> ArenaRunner:
    environment = ShopEnvironment(catalog)
    return ArenaRunner(
        environment,
        BattleSimulator(catalog),
        _population(args, catalog),
        _arena_policy(args, environment),
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


def _generate_episode_dataset(
    runner: ArenaRunner,
    output: Path,
    *,
    episode_dir: Path,
    episodes: int,
    pack: str,
    seed: int,
    identity: dict[str, object],
) -> int:
    """Checkpoint each rollout separately, then assemble the training JSONL."""

    episode_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = episode_dir / "manifest.json"
    manifest = {
        "format": "sapai-arena-episodes-v1",
        "pack": pack,
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
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths: list[Path] = []
    for episode in range(episodes):
        episode_path = episode_dir / f"{episode:06d}.jsonl"
        paths.append(episode_path)
        if episode_path.exists():
            if not read_arena_decisions(episode_path):
                raise ValueError(f"checkpointed Arena episode is empty: {episode_path}")
            continue
        decisions = runner.run(pack=pack, seed=seed + episode).decisions
        temporary = episode_path.with_suffix(".jsonl.tmp")
        write_arena_decisions(temporary, decisions)
        temporary.replace(episode_path)

    combined = []
    for episode_path in paths:
        combined.extend(read_arena_decisions(episode_path))
    return write_arena_decisions(output, combined)


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
    )
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return manifest


def _run_training_sequence(args, catalog: Catalog) -> dict[str, object]:
    from sapai.ml.pipelines import train_battle_model, train_policy_model

    root = Path(args.workdir)
    all_boards = list(read_boards(args.boards))
    boards = [
        board
        for board in all_boards
        if board_is_pack_compatible(board, catalog, args.pack)
    ]
    if not boards:
        raise ValueError(f"board dataset contains no {args.pack!r} boards")
    boards_sha256 = _file_sha256(args.boards)
    battle_dataset = root / "battle-dataset"
    manifest = _battle_dataset_for_sequence(
        boards,
        battle_dataset,
        BattleSimulator(catalog),
        examples=args.battle_examples,
        simulations_per_pair=args.simulations_per_pair,
        seed=args.seed,
        pack=args.pack,
        boards_sha256=boards_sha256,
    )
    battle_summary = train_battle_model(
        battle_dataset,
        root / "battle-model",
        training_config=_training_config(args, epochs=args.battle_epochs),
    )
    population = OpponentPopulation(boards)
    population.save_encoded_cache(root / "population.npz")

    environment = ShopEnvironment(catalog)
    bootstrap_runner = ArenaRunner(
        environment,
        BattleSimulator(catalog),
        population,
        HeuristicPolicy(),
    )
    bootstrap_path = root / "arena-bootstrap.jsonl"
    bootstrap_decisions = _generate_episode_dataset(
        bootstrap_runner,
        bootstrap_path,
        episode_dir=root / "arena-bootstrap-episodes",
        episodes=args.bootstrap_episodes,
        pack=args.pack,
        seed=args.seed,
        identity={
            "policy": "heuristic",
            "boards_sha256": boards_sha256,
            "episodes": args.bootstrap_episodes,
        },
    )
    policy_dir = root / "policy-model"
    bootstrap_summary = train_policy_model(
        bootstrap_path,
        policy_dir,
        training_config=_training_config(args, epochs=args.bootstrap_epochs),
    )

    policy_args = argparse.Namespace(
        policy="search",
        policy_weights=str(policy_dir),
        pack=args.pack,
        seed=args.seed,
        search_simulations=args.search_simulations,
        search_candidates=args.search_candidates,
    )
    search_runner = ArenaRunner(
        environment,
        BattleSimulator(catalog),
        population,
        _arena_policy(policy_args, environment),
    )
    search_path = root / "arena-search.jsonl"
    search_decisions = _generate_episode_dataset(
        search_runner,
        search_path,
        episode_dir=root / "arena-search-episodes",
        episodes=args.search_episodes,
        pack=args.pack,
        seed=args.seed + 100_000,
        identity={
            "policy": "search",
            "boards_sha256": boards_sha256,
            "episodes": args.search_episodes,
            "bootstrap_episodes": args.bootstrap_episodes,
            "bootstrap_epochs": args.bootstrap_epochs,
            "batch_size": args.batch_size,
            "search_simulations": args.search_simulations,
            "search_candidates": args.search_candidates,
        },
    )
    distill_summary = train_policy_model(
        search_path,
        policy_dir,
        training_config=_training_config(
            args, epochs=args.bootstrap_epochs + args.search_epochs
        ),
        resume=True,
    )
    summary = {
        "battle_dataset": manifest,
        "battle_training": battle_summary,
        "bootstrap_decisions": bootstrap_decisions,
        "bootstrap_training": bootstrap_summary,
        "search_decisions": search_decisions,
        "distillation_training": distill_summary,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
            run = runner.run(pack=args.pack, seed=args.seed + episode)
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

        run = _runner(args, catalog).run(pack=args.pack, seed=args.seed)
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
