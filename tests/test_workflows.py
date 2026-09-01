import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sapai.cli import (
    _battle_dataset_for_sequence,
    _completed_iteration_records,
    _generate_episode_dataset,
    _policy_evaluation_score,
    _prepare_sequence_manifest,
)
from sapai.data.datasets import split_boards
from sapai.data.replay import BoardSnapshot
from sapai.data.serialization import (
    pet_from_dict,
    pet_to_dict,
    read_boards,
    team_from_dict,
    team_to_dict,
    write_boards,
)
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Pet, Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.arena import (
    ArenaRunner,
    HeuristicPolicy,
    read_arena_decisions,
    write_arena_decisions,
)
from sapai.training.population import OpponentPopulation, split_opponent_populations
from sapai.visualization.assets import SpriteAtlas
from sapai.visualization.html import render_arena_html, render_battle_html
from tests.helpers import DATA_PATH, catalog

ASSETS_PATH = DATA_PATH.parent


class CliWorkflowTest(unittest.TestCase):
    def test_cli_import_does_not_initialize_tensorflow(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import sapai.cli; assert 'tensorflow' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_score_rejects_shop_collapse_when_results_are_tied(self):
        def evaluation(collapse_penalty):
            return {
                "model": {
                    "completion_rate": 0.0,
                    "mean_trophies": 0.0,
                    "shop_behavior": {
                        "shop_collapse_penalty": collapse_penalty,
                    },
                },
                "search": {"completion_rate": 0.0, "mean_trophies": 0.0},
            }

        self.assertGreater(
            _policy_evaluation_score(evaluation(0.0)),
            _policy_evaluation_score(evaluation(1.0)),
        )

    def test_sequence_manifest_allows_monotonic_continuation_and_code_updates(self):
        def manifest(source, commit, iterations):
            return {
                "format": "sapai-training-sequence-v5",
                "objective": "test",
                "target_schema": "test-v1",
                "boards_sha256": "boards",
                "source_sha256": source,
                "repository_commit": commit,
                "catalog_sha256": "catalog",
                "rules_sha256": "rules",
                "simulator": {"max_rounds": 100},
                "settings": {
                    "pack": "Turtle",
                    "seed": 7,
                    "search_iterations": iterations,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence-manifest.json"
            _prepare_sequence_manifest(
                path,
                manifest("source-a", "commit-a", 3),
                requested_iterations=3,
                completed_iterations=3,
            )
            continued = _prepare_sequence_manifest(
                path,
                manifest("source-b", "commit-b", 6),
                requested_iterations=6,
                completed_iterations=3,
            )

            self.assertEqual(
                continued["continuation"]["requested_search_iterations"], 6
            )
            self.assertEqual(len(continued["code_versions"]), 2)
            with self.assertRaisesRegex(ValueError, "cannot be reduced"):
                _prepare_sequence_manifest(
                    path,
                    manifest("source-b", "commit-b", 5),
                    requested_iterations=5,
                    completed_iterations=3,
                )
            with self.assertRaisesRegex(ValueError, "immutable settings changed"):
                changed = manifest("source-b", "commit-b", 7)
                changed["settings"]["seed"] = 8
                _prepare_sequence_manifest(
                    path,
                    changed,
                    requested_iterations=7,
                    completed_iterations=3,
                )

    def test_v4_sequence_manifest_migrates_without_discarding_prior_outputs(self):
        legacy = {
            "format": "sapai-training-sequence-v4",
            "objective": "test",
            "target_schema": "test-v1",
            "boards_sha256": "boards",
            "source_sha256": "old-source",
            "repository_commit": "old-commit",
            "catalog_sha256": "catalog",
            "rules_sha256": "rules",
            "simulator": {"max_rounds": 100},
            "settings": {
                "pack": "Turtle",
                "seed": 7,
                "search_iterations": 3,
            },
        }
        desired = {
            **legacy,
            "format": "sapai-training-sequence-v5",
            "source_sha256": "new-source",
            "repository_commit": "new-commit",
            "settings": {**legacy["settings"], "search_iterations": 6},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence-manifest.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            migrated = _prepare_sequence_manifest(
                path,
                desired,
                requested_iterations=6,
                completed_iterations=3,
            )

        self.assertEqual(migrated["format"], "sapai-training-sequence-v5")
        self.assertEqual(migrated["created_with"]["repository_commit"], "old-commit")
        self.assertEqual(migrated["continuation"]["requested_search_iterations"], 6)

    def test_iteration_state_extends_an_older_summary_after_an_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(
                json.dumps({"iterations": [{"iteration": 1, "source": "summary"}]}),
                encoding="utf-8",
            )
            state = root / "iteration-state"
            state.mkdir()
            (state / "000002.json").write_text(
                json.dumps({"iteration": 2, "source": "atomic-state"}),
                encoding="utf-8",
            )

            records = _completed_iteration_records(root)

        self.assertEqual([record["iteration"] for record in records], [1, 2])
        self.assertEqual(records[1]["source"], "atomic-state")


class DatasetWorkflowTest(unittest.TestCase):
    def test_training_sequence_reuses_completed_battle_dataset(self):
        ant = catalog().pet_by_name("Ant").create()
        boards = [
            BoardSnapshot(
                f"run-{index // 2}",
                "player" if index % 2 == 0 else "opponent",
                2,
                "Turtle",
                Team.from_pets([ant]),
                version="test",
            )
            for index in range(100)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "battle-dataset"
            first = _battle_dataset_for_sequence(
                boards,
                output,
                BattleSimulator(catalog()),
                examples=10,
                simulations_per_pair=1,
                seed=4,
                pack="Turtle",
                boards_sha256="test-input",
            )
            second = _battle_dataset_for_sequence(
                boards,
                output,
                None,
                examples=10,
                simulations_per_pair=1,
                seed=4,
                pack="Turtle",
                boards_sha256="test-input",
            )

        self.assertEqual(second, first)

    def test_board_snapshot_normalizes_negative_pet_id(self):
        ant = catalog().pet_by_name("Ant").create()
        ant.id = -1
        ant.name = "Pet #-1"
        board = BoardSnapshot("old", "player", 1, "Turtle", Team.from_pets([ant]))
        self.assertTrue(all(pet is None for pet in board.team.slots))

    def test_old_serialized_negative_pet_id_is_an_empty_slot(self):
        team = team_from_dict(
            [
                {
                    "id": -1,
                    "name": "Pet #-1",
                    "tier": 0,
                    "attack": 0,
                    "health": 0,
                },
                None,
                None,
                None,
                None,
            ]
        )
        self.assertTrue(all(pet is None for pet in team.slots))

    def test_old_serialized_unknown_pet_is_inferred_as_vanilla_fallback(self):
        team = team_from_dict(
            [
                {
                    "id": 999_999,
                    "name": "Pet #999999",
                    "tier": 0,
                    "attack": 7,
                    "health": 8,
                },
                None,
                None,
                None,
                None,
            ]
        )
        pet = team.slots[0]
        self.assertIsNotNone(pet)
        self.assertEqual(pet.metadata["vanilla_fallback"], "unknown_pet_id")
        BattleSimulator(catalog()).assert_team_supported(team)

    def test_old_serialized_unknown_perk_is_inferred_as_fallback(self):
        team = team_from_dict(
            [
                {
                    "id": 0,
                    "name": "Ant",
                    "tier": 1,
                    "attack": 2,
                    "health": 2,
                    "perk": "Perk #999999",
                },
                None,
                None,
                None,
                None,
            ]
        )
        pet = team.slots[0]
        self.assertIsNotNone(pet)
        self.assertEqual(pet.metadata["perk_fallback"], "unknown_perk_id")
        BattleSimulator(catalog()).assert_team_supported(team)

    def test_board_json_round_trip_and_replay_safe_split(self):
        ant = catalog().pet_by_name("Ant").create()
        boards = [
            BoardSnapshot(
                replay_id=f"run-{index // 2}",
                side="player" if index % 2 == 0 else "opponent",
                turn=2,
                pack="Turtle",
                team=Team.from_pets([ant]),
                version="test",
            )
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boards.jsonl"
            self.assertEqual(write_boards(path, boards), len(boards))
            loaded = list(read_boards(path))
        self.assertEqual(loaded[0].team.slots[0].name, "Ant")
        split = split_boards(loaded, seed=4)
        replay_ids = [
            {board.replay_id for board in getattr(split, name)}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(replay_ids[0] & replay_ids[1])
        self.assertFalse(replay_ids[0] & replay_ids[2])
        self.assertFalse(replay_ids[1] & replay_ids[2])

    def test_nested_pet_metadata_is_json_safe_and_round_trips(self):
        current_catalog = catalog()
        whale = current_catalog.pet_by_name("Whale").create(instance_id=1)
        swallowed = current_catalog.pet_by_name("Ant").create(instance_id=2)
        whale.metadata["swallowed"] = [swallowed]

        encoded = pet_to_dict(whale)
        json.dumps(encoded)
        restored = pet_from_dict(encoded)

        self.assertIsInstance(restored.metadata["swallowed"][0], type(swallowed))
        self.assertEqual(restored.metadata["swallowed"][0].name, "Ant")

    def test_named_runtime_token_survives_team_serialization(self):
        token = Pet(-1, "Ram", 3, 4, 4)
        restored = team_from_dict(team_to_dict(Team.from_pets([token])))
        self.assertEqual(restored.slots[0].name, "Ram")


class ArenaWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()
        self.population = OpponentPopulation.synthetic(
            self.catalog,
            max_turn=5,
            boards_per_turn=2,
            seed=7,
        )

    def test_complete_arena_round_trip_and_visualization(self):
        runner = ArenaRunner(
            ShopEnvironment(self.catalog),
            BattleSimulator(self.catalog),
            self.population,
            HeuristicPolicy(),
            max_decisions_per_turn=12,
        )
        run = runner.run(seed=3)
        self.assertTrue(run.final_state.terminal)
        self.assertEqual(len(run.turns), run.final_state.turn)
        self.assertTrue(run.decisions)
        self.assertTrue(
            all(
                decision.expected_trophies == run.final_state.trophies / 10
                for decision in run.decisions
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision_path = root / "arena.jsonl"
            html_path = root / "arena.html"
            write_arena_decisions(decision_path, run.decisions)
            loaded = read_arena_decisions(decision_path)
            render_arena_html(run, html_path, ASSETS_PATH)
            html = html_path.read_text(encoding="utf-8")
            stylesheet = (root / "sapai.css").read_text(encoding="utf-8")
            runtime = (root / "sapai.js").read_text(encoding="utf-8")
            pet_assets_created = (root / "sapai-assets" / "Pets").is_dir()
        self.assertEqual(len(loaded), len(run.decisions))
        self.assertEqual(loaded[0].actions, run.decisions[0].actions)
        self.assertEqual(loaded[0].chosen_action, run.decisions[0].chosen_action)
        self.assertNotIn("data:image/png;base64", html)
        self.assertIn("Super Auto Pets Arena run", html)
        self.assertIn('href="sapai.css"', html)
        self.assertIn('src="sapai.js"', html)
        self.assertIn(".battlefield", stylesheet)
        self.assertIn("function battle(slide)", runtime)
        self.assertIn("front is right", runtime)
        self.assertIn("function experienceLabel(pet)", runtime)
        self.assertTrue(pet_assets_created)

    def test_opponent_population_split_pins_patch_and_replay_groups(self):
        boards = []
        ant = self.catalog.pet_by_name("Ant").create()
        for index in range(60):
            version = "selected" if index < 50 else "older"
            for side in ("player", "opponent"):
                boards.append(
                    BoardSnapshot(
                        f"replay-{index}",
                        side,
                        1 + index % 3,
                        "Turtle",
                        Team.from_pets([ant.clone()]),
                        version=version,
                    )
                )

        populations = split_opponent_populations(boards, seed=7)

        self.assertEqual(populations.version, "selected")
        replay_groups = [
            {board.replay_id for board in getattr(populations, name).boards}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(replay_groups[0] & replay_groups[1])
        self.assertFalse(replay_groups[0] & replay_groups[2])
        self.assertFalse(replay_groups[1] & replay_groups[2])
        self.assertTrue(
            all(
                board.version == "selected"
                for name in ("train", "validation", "test")
                for board in getattr(populations, name).boards
            )
        )

    def test_battle_frames_and_token_sprite_mapping(self):
        cricket = self.catalog.pet_by_name("Cricket").create()
        ant = self.catalog.pet_by_name("Ant").create()
        result = BattleSimulator(self.catalog).simulate(
            Team.from_pets([cricket]),
            Team.from_pets([ant]),
            seed=2,
        )
        events = [frame.event for frame in result.frames]
        self.assertGreaterEqual(len(result.frames), result.rounds * 3 + 2)
        self.assertIn("clash", events)
        self.assertIn("impact", events)
        self.assertIn("resolve", events)
        self.assertEqual(SpriteAtlas(ASSETS_PATH).path("pet", "Zombie Cricket").name, "CricketToken.png")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "battle.html"
            render_battle_html(result, output, ASSETS_PATH)
            html = output.read_text(encoding="utf-8")
            stylesheet = (output.parent / "sapai.css").read_text(encoding="utf-8")
            runtime = (output.parent / "sapai.js").read_text(encoding="utf-8")
            cricket_sprite_created = (
                output.parent / "sapai-assets" / "Pets" / "Cricket.png"
            ).is_file()
        self.assertIn("Super Auto Pets battle", html)
        self.assertNotIn("data:image/png;base64", html)
        self.assertIn('href="sapai.css"', html)
        self.assertIn('src="sapai.js"', html)
        self.assertIn('id="play"', html)
        self.assertIn('"event":"impact"', html)
        self.assertIn('"role":"interactor"', html)
        self.assertNotIn('"role":"attacker"', html)
        self.assertIn("@keyframes clash-player", stylesheet)
        self.assertIn("@keyframes clash-opponent", stylesheet)
        self.assertIn('class="battlefield"', runtime)
        self.assertTrue(cricket_sprite_created)

    def test_episode_dataset_resumes_from_per_episode_files(self):
        runner = ArenaRunner(
            ShopEnvironment(self.catalog),
            BattleSimulator(self.catalog),
            self.population,
            HeuristicPolicy(),
            max_decisions_per_turn=12,
        )

        class CountingRunner:
            calls = 0

            def run(self, *, pack, version, seed):
                self.calls += 1
                return runner.run(pack=pack, version=version, seed=seed)

        counting = CountingRunner()
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "arena.jsonl"
            episode_dir = root / "episodes"
            first = _generate_episode_dataset(
                counting,
                output,
                episode_dir=episode_dir,
                episodes=1,
                pack="Turtle",
                seed=9,
                identity={"policy": "test"},
                progress=lambda completed, total, reused: progress.append(
                    (completed, total, reused)
                ),
            )
            output.unlink()
            second = _generate_episode_dataset(
                counting,
                output,
                episode_dir=episode_dir,
                episodes=1,
                pack="Turtle",
                seed=9,
                identity={"policy": "test"},
                progress=lambda completed, total, reused: progress.append(
                    (completed, total, reused)
                ),
            )

        self.assertGreater(first, 0)
        self.assertEqual(second, first)
        self.assertEqual(counting.calls, 1)
        self.assertEqual(progress, [(1, 1, False), (1, 1, True)])


if __name__ == "__main__":
    unittest.main()
