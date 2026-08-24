import tempfile
import unittest
from pathlib import Path

from sapai.data.datasets import split_boards
from sapai.data.replay import BoardSnapshot
from sapai.data.serialization import read_boards, write_boards
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.arena import (
    ArenaRunner,
    HeuristicPolicy,
    read_arena_decisions,
    write_arena_decisions,
)
from sapai.training.population import OpponentPopulation
from sapai.visualization.assets import SpriteAtlas
from sapai.visualization.html import render_arena_html, render_battle_html
from tests.helpers import DATA_PATH, catalog

ASSETS_PATH = DATA_PATH.parent


class DatasetWorkflowTest(unittest.TestCase):
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
        self.assertTrue(all(decision.expected_wins == run.final_state.trophies for decision in run.decisions))
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
        self.assertNotIn("data:image/png;base64", html)
        self.assertIn("Super Auto Pets Arena run", html)
        self.assertIn('href="sapai.css"', html)
        self.assertIn('src="sapai.js"', html)
        self.assertIn(".battlefield", stylesheet)
        self.assertIn("function battle(slide)", runtime)
        self.assertTrue(pet_assets_created)

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
        self.assertIn("attack", events)
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
        self.assertIn('"role":"attacker"', html)
        self.assertIn("@keyframes lunge-player", stylesheet)
        self.assertIn('class="battlefield"', runtime)
        self.assertTrue(cricket_sprite_created)


if __name__ == "__main__":
    unittest.main()
