import ast
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sapai.data.replay import BoardSnapshot
from sapai.data.serialization import team_to_dict, write_boards
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.human import HumanArenaSession, HumanBenchmarkConfig
from sapai.training.population import (
    OpponentPopulation,
    load_opponent_boards,
    load_opponent_population,
)
from sapai.visualization.colab import (
    _battle_result_from_dict,
    build_human_arena_html,
    display_human_arena,
    human_arena_payload,
)
from sapai.visualization.widget import display_human_arena_widget
from tests.helpers import DATA_PATH, catalog

ASSETS_PATH = DATA_PATH.parent


class FixedEnvironment(ShopEnvironment):
    def __init__(self, current_catalog, state):
        super().__init__(current_catalog)
        self.initial_state = state.clone()

    def reset(self, *, pack="Turtle", seed=None):
        state = self.initial_state.clone()
        state.pack = pack
        return state


class HumanArenaSessionTest(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()
        self.population = OpponentPopulation.synthetic(
            self.catalog,
            max_turn=20,
            boards_per_turn=2,
            seed=7,
        )

    def _state_with_all_action_types(self):
        state = ShopEnvironment(self.catalog).reset(seed=4)
        ant = self.catalog.pet_by_name("Ant")
        fish = self.catalog.pet_by_name("Fish")
        state.team = Team.from_pets([ant.create(instance_id=100), ant.create(instance_id=101)])
        state.shop = Shop(
            pets=[
                ShopPet(ant.create(instance_id=102)),
                ShopPet(fish.create(instance_id=103)),
            ],
            foods=[self.catalog.food_by_name("Apple").create()],
        )
        return state

    def _session(self, directory, *, state=None, alias="human", seed=11):
        state = state or self._state_with_all_action_types()
        environment = FixedEnvironment(self.catalog, state)
        config = HumanBenchmarkConfig(
            output_dir=directory,
            participant_alias=alias,
            pack="Turtle",
            seed=seed,
            boards_sha256="test-boards",
            board_count=len(self.population.boards),
            repository_commit="test-commit",
        )
        return HumanArenaSession.create_or_resume(
            environment,
            BattleSimulator(self.catalog),
            self.population,
            config,
        )

    @staticmethod
    def _action(snapshot, kind, *, source=None, target=None):
        return next(
            action
            for action in snapshot["actions"]
            if action["kind"] == kind
            and (source is None or action["source"] == source)
            and (target is None or action["target"] == target)
        )

    def test_action_ids_cover_shop_transitions_and_reject_stale_input(self):
        kinds = {
            "BUY_PET",
            "BUY_MERGE_PET",
            "MERGE_BOARD_PET",
            "BUY_FOOD",
            "ROLL",
            "FREEZE_PET",
            "FREEZE_FOOD",
            "SELL_PET",
            "END_TURN",
            "REORDER",
        }
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._session(directory).snapshot()
            self.assertTrue(kinds <= {action["kind"] for action in snapshot["actions"]})
            freeze = self._action(snapshot, "FREEZE_PET", source=0)
            session = self._session(directory)
            updated = session.apply_action(
                freeze["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=125.5,
            )
            self.assertEqual(updated["revision"], snapshot["revision"] + 1)
            with self.assertRaisesRegex(ValueError, "stale"):
                session.apply_action(
                    freeze["id"],
                    expected_revision=snapshot["revision"],
                    elapsed_ms=1,
                )
            with self.assertRaisesRegex(ValueError, "unknown or stale"):
                session.apply_action(
                    "made-up",
                    expected_revision=updated["revision"],
                    elapsed_ms=1,
                )

    def test_persistence_failure_does_not_commit_the_transition_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory)
            before = session.snapshot()
            before_key = session.state.canonical_key()
            freeze = self._action(before, "FREEZE_PET", source=0)
            with (
                patch("sapai.training.human._atomic_json", side_effect=OSError("Drive full")),
                self.assertRaisesRegex(OSError, "Drive full"),
            ):
                session.apply_action(
                    freeze["id"],
                    expected_revision=before["revision"],
                    elapsed_ms=1,
                )
            self.assertEqual(session.revision, before["revision"])
            self.assertEqual(session.state.canonical_key(), before_key)
            resumed = self._session(directory)
            self.assertEqual(resumed.revision, before["revision"])
            self.assertEqual(resumed.state.canonical_key(), before_key)

    def test_every_action_kind_can_be_applied_through_an_opaque_id(self):
        action_specs = [
            ("BUY_PET", {"source": 1, "target": 2}),
            ("BUY_MERGE_PET", {"source": 0, "target": 0}),
            ("MERGE_BOARD_PET", {"source": 0, "target": 1}),
            ("BUY_FOOD", {"source": 0, "target": 0}),
            ("ROLL", {}),
            ("FREEZE_PET", {"source": 0}),
            ("FREEZE_FOOD", {"source": 0}),
            ("SELL_PET", {"source": 0}),
            ("REORDER", {}),
            ("END_TURN", {}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (kind, filters) in enumerate(action_specs):
                session = self._session(root / str(index))
                snapshot = session.snapshot()
                action = self._action(snapshot, kind, **filters)
                updated = session.apply_action(
                    action["id"],
                    expected_revision=snapshot["revision"],
                    elapsed_ms=10,
                )
                self.assertEqual(updated["decision_index"], 1, kind)

    def test_multiple_offers_freeze_independently_and_reset_on_roll(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory)
            for kind, source in (("FREEZE_PET", 0), ("FREEZE_PET", 1), ("FREEZE_FOOD", 0)):
                snapshot = session.snapshot()
                action = self._action(snapshot, kind, source=source)
                session.apply_action(
                    action["id"],
                    expected_revision=snapshot["revision"],
                    elapsed_ms=5,
                )
            state = session.state
            self.assertTrue(all(offer.frozen for offer in state.shop.pets))
            self.assertTrue(all(food.frozen for food in state.shop.foods))
            self.assertFalse(
                any(
                    action["kind"].startswith("UNFREEZE")
                    for action in session.snapshot()["actions"]
                )
            )

            snapshot = session.snapshot()
            roll = self._action(snapshot, "ROLL")
            rolled = session.apply_action(
                roll["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=5,
            )
            self.assertEqual(
                sum(action["kind"].startswith("UNFREEZE") for action in rolled["actions"]),
                3,
            )

    def test_battle_review_is_hidden_before_end_turn_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory)
            snapshot = session.snapshot()
            self.assertIsNone(snapshot["battle"])
            end = self._action(snapshot, "END_TURN")
            battle = session.apply_action(
                end["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=50,
            )
            self.assertEqual(battle["stage"], "battle_review")
            self.assertIn("opponent", battle["battle"])
            visual = human_arena_payload(session, ASSETS_PATH)
            self.assertTrue(visual["battle"]["slides"])
            self.assertTrue(visual["sprites"]["pet"])

            resumed = self._session(directory)
            self.assertEqual(resumed.stage, "battle_review")
            next_view = resumed.continue_battle(expected_revision=resumed.revision)
            self.assertIn(next_view["stage"], {"shop", "complete"})
            self.assertIsNone(next_view["battle"])
            self.assertFalse((Path(directory) / "current.json.tmp").exists())

    def test_reported_ox_board_persists_when_an_opponent_whale_swallows_a_pet(self):
        state = ShopEnvironment(self.catalog).reset(seed=4)
        front_ox = self.catalog.pet_by_name("Ox").create(instance_id=1)
        front_ox.attack = 10
        front_ox.health = 7
        front_ox.perk = "Melon"
        squirrel = self.catalog.pet_by_name("Squirrel").create(instance_id=2)
        squirrel.attack = 7
        squirrel.health = 7
        back_ox = self.catalog.pet_by_name("Ox").create(instance_id=3)
        back_ox.attack = 1
        back_ox.health = 3
        state.team = Team.from_pets([front_ox, squirrel, back_ox])

        swallowed = self.catalog.pet_by_name("Ant").create(instance_id=4)
        whale = self.catalog.pet_by_name("Whale").create(instance_id=5)
        population = OpponentPopulation(
            [
                BoardSnapshot(
                    "whale-opponent",
                    "opponent",
                    state.turn,
                    "Turtle",
                    Team.from_pets([swallowed, whale]),
                    version=state.version,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            session = HumanArenaSession.create_or_resume(
                FixedEnvironment(self.catalog, state),
                BattleSimulator(self.catalog),
                population,
                HumanBenchmarkConfig(
                    directory,
                    "human",
                    "Turtle",
                    9,
                    "whale-board",
                    1,
                    "test-commit",
                ),
            )
            snapshot = session.snapshot()
            end = self._action(snapshot, "END_TURN")
            review = session.apply_action(
                end["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=10,
            )
            persisted = json.loads((Path(directory) / "current.json").read_text())
            visual = human_arena_payload(session, ASSETS_PATH)

        self.assertEqual(review["stage"], "battle_review")
        self.assertEqual(persisted["stage"], "battle_review")
        self.assertTrue(visual["battle"]["slides"])

    def test_whale_deer_bus_persists_in_human_battle_review_and_visualizer(self):
        state = ShopEnvironment(self.catalog).reset(seed=4)
        deer = self.catalog.pet_by_name("Deer").create(instance_id=1)
        whale = self.catalog.pet_by_name("Whale").create(instance_id=2)
        whale.health = 100
        state.team = Team.from_pets([deer, whale])
        population = OpponentPopulation(
            [
                BoardSnapshot(
                    "tank-opponent",
                    "opponent",
                    state.turn,
                    "Turtle",
                    Team.from_pets([self.catalog.pet_by_name("Sloth").create()]),
                    version=state.version,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            config = HumanBenchmarkConfig(
                directory,
                "human",
                "Turtle",
                9,
                "whale-deer-board",
                1,
                "test-commit",
            )
            environment = FixedEnvironment(self.catalog, state)
            session = HumanArenaSession.create_or_resume(
                environment,
                BattleSimulator(self.catalog),
                population,
                config,
            )
            snapshot = session.snapshot()
            end = self._action(snapshot, "END_TURN")
            session.apply_action(
                end["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=10,
            )
            resumed = HumanArenaSession.create_or_resume(
                environment,
                BattleSimulator(self.catalog),
                population,
                config,
            )
            review = resumed.snapshot()["battle"]["result"]
            visual = human_arena_payload(resumed, ASSETS_PATH)

        ability_frame = next(frame for frame in review["frames"] if frame["event"] == "ability")
        self.assertEqual(
            [pet["name"] for pet in ability_frame["player"] if pet is not None],
            ["Bus", "Whale"],
        )
        ability_slide = next(
            slide for slide in visual["battle"]["slides"] if slide["event"] == "ability"
        )
        self.assertEqual(
            [pet["name"] for pet in ability_slide["player"] if pet is not None],
            ["Bus", "Whale"],
        )

    def test_terminal_episode_writes_audit_summary_and_starts_the_next_game(self):
        state = self._state_with_all_action_types()
        state.team = Team()
        state.lives = 1
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory, state=state)
            snapshot = session.snapshot()
            end = self._action(snapshot, "END_TURN")
            session.apply_action(
                end["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=25,
            )
            completed = session.continue_battle(expected_revision=session.revision)
            self.assertEqual(completed["stage"], "complete")
            self.assertEqual(completed["summary"]["games_completed"], 1)
            battle_rates = sum(
                completed["summary"][key]
                for key in ("battle_win_rate", "battle_draw_rate", "battle_loss_rate")
            )
            self.assertAlmostEqual(battle_rates, 1.0)
            episode = json.loads((Path(directory) / "episodes" / "000000.json").read_text())
            self.assertEqual(episode["metrics"]["decisions"], 1)
            self.assertIn("battle_seed", episode["battles"][0])
            self.assertNotIn("frames", episode["battles"][0]["result"])

            next_game = session.new_episode(expected_revision=session.revision)
            self.assertEqual(next_game["stage"], "shop")
            self.assertEqual(next_game["episode_index"], 1)
            self.assertTrue(next_game["summary"]["episode_in_progress"])

    def test_completed_benchmark_directory_can_be_opened_repeatedly(self):
        state = self._state_with_all_action_types()
        state.team = Team()
        state.lives = 1
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory, state=state)
            snapshot = session.snapshot()
            end = self._action(snapshot, "END_TURN")
            session.apply_action(
                end["id"],
                expected_revision=snapshot["revision"],
                elapsed_ms=1,
            )
            session.continue_battle(expected_revision=session.revision)
            episode_path = Path(directory) / "episodes" / "000000.json"
            completed_episode = episode_path.read_bytes()

            first_resume = self._session(directory, state=state)
            second_resume = self._session(directory, state=state)

            self.assertEqual(first_resume.stage, "complete")
            self.assertEqual(second_resume.stage, "complete")
            self.assertEqual(second_resume.snapshot()["summary"]["games_completed"], 1)
            self.assertEqual(episode_path.read_bytes(), completed_episode)

    def test_resume_rejects_a_changed_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            self._session(directory)
            environment = FixedEnvironment(self.catalog, self._state_with_all_action_types())
            changed = HumanBenchmarkConfig(
                directory,
                "human",
                "Turtle",
                11,
                "different-boards",
                len(self.population.boards),
                "test-commit",
            )
            with self.assertRaisesRegex(ValueError, "settings changed"):
                HumanArenaSession.create_or_resume(
                    environment,
                    BattleSimulator(self.catalog),
                    self.population,
                    changed,
                )

    def test_notebook_mode_versions_an_incompatible_existing_directory(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "benchmark"
            original = self._session(directory)
            original_manifest = (directory / "manifest.json").read_bytes()
            original_current = (directory / "current.json").read_bytes()
            changed = HumanBenchmarkConfig(
                directory,
                "human",
                "Turtle",
                11,
                "different-boards",
                len(self.population.boards),
                "new-commit",
            )

            versioned = HumanArenaSession.create_or_resume(
                original.environment,
                BattleSimulator(self.catalog),
                self.population,
                changed,
                version_on_mismatch=True,
            )
            resumed_versioned = HumanArenaSession.create_or_resume(
                original.environment,
                BattleSimulator(self.catalog),
                self.population,
                changed,
                version_on_mismatch=True,
            )

            self.assertNotEqual(versioned.config.directory, directory)
            self.assertEqual(versioned.config.directory, resumed_versioned.config.directory)
            self.assertTrue(
                versioned.config.directory.name.startswith(directory.name + "-")
            )
            self.assertEqual(
                (directory / "manifest.json").read_bytes(), original_manifest
            )
            self.assertEqual((directory / "current.json").read_bytes(), original_current)

    def test_reload_does_not_change_seeded_episode_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / "continuous", Path(directory) / "resumed"]
            sessions = [self._session(root, seed=123) for root in roots]
            for index, session in enumerate(sessions):
                while session.stage != "complete":
                    snapshot = session.snapshot()
                    if session.stage == "shop":
                        end = self._action(snapshot, "END_TURN")
                        session.apply_action(
                            end["id"],
                            expected_revision=snapshot["revision"],
                            elapsed_ms=1,
                        )
                    else:
                        session.continue_battle(expected_revision=session.revision)
                    if index == 1 and session.stage != "complete":
                        session = self._session(roots[index], seed=123)
                sessions[index] = session
            self.assertEqual(
                sessions[0].state.canonical_key(),
                sessions[1].state.canonical_key(),
            )
            first = json.loads((roots[0] / "episodes" / "000000.json").read_text())
            second = json.loads((roots[1] / "episodes" / "000000.json").read_text())
            self.assertEqual(
                [battle["opponent"]["replay_id"] for battle in first["battles"]],
                [battle["opponent"]["replay_id"] for battle in second["battles"]],
            )

    def test_inline_payload_maps_actions_and_escapes_participant_html(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory, alias="<script>alert(1)</script>")
            payload = human_arena_payload(session, ASSETS_PATH)
            document = build_human_arena_html(
                session,
                ASSETS_PATH,
                callback_name="test.callback",
            )
            self.assertIsNone(payload["battle"])
            self.assertTrue(payload["actions"])
            self.assertTrue(payload["sprites"]["pet"])
            self.assertIn("data:image/png;base64", document)
            self.assertIn("test.callback", document)
            self.assertNotIn("<script>alert(1)</script>", document)
            self.assertIn("google.colab.kernel.invokeFunction", document)
            self.assertIn('draggable="true"', document)
            self.assertIn("front is right", document)
            self.assertIn("function experienceLabel(pet)", document)
            self.assertIn("ondragstart", document)

    def test_colab_adapter_registers_a_refreshable_callback(self):
        callbacks = {}
        displayed = []
        fake_output = types.SimpleNamespace(
            register_callback=lambda name, callback: callbacks.setdefault(name, callback)
        )
        fake_colab = types.ModuleType("google.colab")
        fake_colab.output = fake_output
        fake_ipython = types.ModuleType("IPython")
        fake_display = types.ModuleType("IPython.display")

        class FakeDisplayValue:
            def __init__(self, data):
                self.data = data

        fake_display.HTML = FakeDisplayValue
        fake_display.JSON = FakeDisplayValue
        fake_display.display = displayed.append
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory)
            with patch.dict(
                sys.modules,
                {
                    "google.colab": fake_colab,
                    "IPython": fake_ipython,
                    "IPython.display": fake_display,
                },
            ):
                callback_name = display_human_arena(session, ASSETS_PATH)
            self.assertIn(callback_name, callbacks)
            self.assertEqual(len(displayed), 1)
            refreshed = callbacks[callback_name]("refresh", {})
            self.assertEqual(refreshed.data["stage"], "shop")

    def test_old_directional_battle_frame_loads_as_an_undirected_clash(self):
        left = self.catalog.pet_by_name("Ant").create()
        right = self.catalog.pet_by_name("Fish").create()
        left.metadata["battle_visual_id"] = 11
        right.metadata["battle_visual_id"] = 22
        left_team = Team.from_pets([left])
        right_team = Team.from_pets([right])

        restored = _battle_result_from_dict(
            {
                "outcome": "draw",
                "rounds": 1,
                "player": team_to_dict(left_team),
                "opponent": team_to_dict(right_team),
                "log": [],
                "frames": [
                    {
                        "label": "legacy attack",
                        "player": team_to_dict(left_team),
                        "opponent": team_to_dict(right_team),
                        "log_index": 0,
                        "event": "attack",
                        "actor_id": 11,
                        "target_id": 22,
                    }
                ],
            }
        )

        self.assertEqual(restored.frames[0].event, "clash")
        self.assertEqual(restored.frames[0].participant_ids, (11, 22))

    def test_standard_jupyter_widget_adapter_uses_only_core_widget_models(self):
        displayed = []

        class FakeWidget:
            def __init__(self, children=(), **values):
                self.children = tuple(children)
                self.classes = []
                self._click_callbacks = []
                self._observers = []
                for key, value in values.items():
                    setattr(self, key, value)

            def add_class(self, name):
                self.classes.append(name)

            def on_click(self, callback):
                self._click_callbacks.append(callback)

            def observe(self, callback, names=None):
                self._observers.append((callback, names))

            def click(self):
                for callback in self._click_callbacks:
                    callback(self)

        class FakeLayout:
            def __init__(self, **values):
                for key, value in values.items():
                    setattr(self, key, value)

        class FakeJavascript:
            def __init__(self, data):
                self.data = data

        fake_widgets = types.ModuleType("ipywidgets")
        fake_widgets.Layout = FakeLayout
        fake_widgets.Button = FakeWidget
        fake_widgets.HTML = FakeWidget
        fake_widgets.VBox = FakeWidget
        fake_widgets.HBox = FakeWidget
        fake_widgets.Box = FakeWidget
        fake_widgets.IntSlider = FakeWidget
        fake_widgets.Play = FakeWidget
        fake_widgets.jslink = lambda *_values: object()
        fake_ipython = types.ModuleType("IPython")
        fake_display = types.ModuleType("IPython.display")
        fake_display.Javascript = FakeJavascript
        fake_display.display = displayed.append

        with tempfile.TemporaryDirectory() as directory:
            session = self._session(directory)
            with patch.dict(
                sys.modules,
                {
                    "ipywidgets": fake_widgets,
                    "IPython": fake_ipython,
                    "IPython.display": fake_display,
                },
            ):
                widget = display_human_arena_widget(session, ASSETS_PATH)

            def descendants(value):
                yield value
                for child in getattr(value, "children", ()):
                    yield from descendants(child)

            roll = next(
                item
                for item in descendants(widget)
                if getattr(item, "description", "") == "↻ Roll (1 gold)"
            )
            roll.click()

        self.assertIs(displayed[0], widget)
        self.assertIsInstance(displayed[1], FakeJavascript)
        self.assertIn("MutationObserver", displayed[1].data)
        self.assertIn("sapai-human-core-widget", widget.classes)
        self.assertEqual(session.revision, 1)

    def test_shared_population_loader_applies_the_model_pack_filter(self):
        ant = self.catalog.pet_by_name("Ant").create()
        boards = [
            BoardSnapshot("turtle", "opponent", 1, "Turtle", Team.from_pets([ant])),
            BoardSnapshot("other", "opponent", 1, "Unicorn", Team.from_pets([ant])),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boards.jsonl"
            write_boards(path, boards)
            loaded = load_opponent_boards(path, self.catalog, "Turtle")
            population = load_opponent_population(path, self.catalog, "Turtle")
        self.assertEqual([board.replay_id for board in loaded], ["turtle"])
        self.assertEqual([board.replay_id for board in population.boards], ["turtle"])

    def test_training_notebook_contains_a_guarded_parseable_launcher(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "sapai_colab_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        config = "".join(notebook["cells"][1]["source"])
        launcher = "".join(notebook["cells"][-1]["source"])
        self.assertIn("RUN_HUMAN_BENCHMARK = False", config)
        self.assertIn("HUMAN_BENCHMARK_DIR", config)
        self.assertIn("if RUN_HUMAN_BENCHMARK:", launcher)
        self.assertIn("split_opponent_populations(", launcher)
        self.assertIn("human_population = human_populations.test", launcher)
        self.assertIn("display_human_arena", launcher)
        self.assertIn("version_on_mismatch=True", launcher)
        self.assertIn("'--progress'", source)
        self.assertIn("SEARCH_ITERATIONS_PER_RUN", source)
        self.assertIn("'--additional-search-iterations'", source)
        ast.parse(launcher)

    def test_kaggle_notebook_is_native_guarded_and_parseable(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "sapai_kaggle_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RUN_HUMAN_BENCHMARK = False", source)
        self.assertIn("/kaggle/working", source)
        self.assertIn("KAGGLE_PRIOR_RUN_DIR", source)
        self.assertIn("def training_run_candidates", source)
        self.assertIn("Training run: auto-detected", source)
        self.assertIn("restored and validated", source)
        self.assertIn("UserSecretsClient().get_secret('DATABASE_URL')", source)
        self.assertIn("display_human_arena_widget", source)
        self.assertIn("version_on_mismatch=True", source)
        self.assertIn("'--progress'", source)
        self.assertIn("SEARCH_ITERATIONS_PER_RUN", source)
        self.assertIn("'--additional-search-iterations'", source)
        self.assertNotIn("anywidget", source.lower())
        self.assertNotIn("google.colab", source)
        self.assertNotIn("/content/", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell.get("source", []))
            if code.startswith("%pip "):
                code = "\n".join(code.splitlines()[1:])
            ast.parse(code)

    def test_kaggle_notebook_auto_detects_and_restores_matching_training_run(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "sapai_kaggle_training.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        setup = "".join(notebook["cells"][3]["source"])
        tree = ast.parse(setup)
        helper_names = {
            "training_run_candidates",
            "describe_training_run",
            "resolve_training_source",
            "validate_training_run",
            "restore_training_output",
        }
        helpers = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in helper_names
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(helpers)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            run = input_root / "saved-output" / "sapai-runs" / "policy-improvement-v4-001"
            (run / "policy-model" / "checkpoints").mkdir(parents=True)
            (run / "sequence-manifest.json").write_text("{}", encoding="utf-8")
            (run / "policy-model" / "run-manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            (run / "summary.json").write_text(
                json.dumps({"completed_search_iterations": 3}), encoding="utf-8"
            )
            smoke = input_root / "saved-output" / "sapai-runs" / "run-smoke-deadbeef"
            (smoke / "policy-model" / "checkpoints").mkdir(parents=True)
            (smoke / "sequence-manifest.json").write_text("{}", encoding="utf-8")
            (smoke / "policy-model" / "run-manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            destination = root / "working" / "continued-run"
            namespace = {
                "Path": Path,
                "json": json,
                "shutil": shutil,
                "input_root": input_root,
            }
            exec(compile(helpers, "kaggle-helpers", "exec"), namespace)  # noqa: S102

            restored = namespace["restore_training_output"]("", str(destination))

            self.assertEqual(restored, destination.resolve())
            self.assertTrue((destination / "sequence-manifest.json").is_file())
            self.assertTrue((destination / "policy-model" / "checkpoints").is_dir())


if __name__ == "__main__":
    unittest.main()
