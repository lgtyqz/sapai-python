import random
import unittest

from sapai.data.replay import BoardSnapshot
from sapai.search.stochastic import PolicyGuidedSearch, SearchConfig, UniformEvaluator
from sapai.sim.actions import Action, ActionKind
from sapai.sim.battle import BattleResult, BattleResultKind, BattleSimulator
from sapai.sim.models import RunState, Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
from sapai.training.arena import RandomPolicy
from sapai.training.population import OpponentPopulation, SimulatorPopulationEvaluator
from tests.helpers import catalog


class ToggleFirstEvaluator:
    def evaluate(self, state, actions):
        toggles = {
            ActionKind.FREEZE_PET,
            ActionKind.UNFREEZE_PET,
            ActionKind.FREEZE_FOOD,
            ActionKind.UNFREEZE_FOOD,
        }
        return [1.0 if action.kind in toggles else 0.0 for action in actions], 0.0


class ReorderFirstEvaluator:
    def evaluate(self, state, actions):
        return [float(action.kind is ActionKind.REORDER) for action in actions], 0.0


class SellFirstEvaluator:
    def evaluate(self, state, actions):
        return [float(action.kind is ActionKind.SELL_PET) for action in actions], 0.0


class FixedBattleEvaluator:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def evaluate_battle(self, state, rng):
        self.calls += 1
        return self.value


class AdaptiveBattleEvaluator:
    def __init__(self):
        self.requested = []

    def begin_search(self, state, rng):
        self.requested = []

    def evaluate_battle(self, state, rng, *, simulations=None):
        self.requested.append(simulations)
        return 0.5


class TrophyValueEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate_many(self, states, actions):
        self.batch_sizes.append(len(states))
        return [
            ([1.0 / len(legal)] * len(legal), state.trophies / 10.0)
            for state, legal in zip(states, actions, strict=True)
        ]


class SearchTest(unittest.TestCase):
    def test_candidate_budget_reserves_one_action_per_kind(self):
        environment = ShopEnvironment(catalog())
        state = environment.reset(seed=5)
        state.team = Team.from_pets(
            [catalog().pet_by_name("Ant").create(), catalog().pet_by_name("Fish").create()]
        )
        actions = environment.legal_actions(state)
        kinds = {action.kind for action in actions}
        search = PolicyGuidedSearch(
            environment,
            UniformEvaluator(),
            SearchConfig(
                simulations=1,
                candidate_actions=len(kinds),
                max_depth=1,
                gumbel_scale=0.0,
            ),
        )

        search.search(state, seed=3)

        root = search.transpositions[state.canonical_key()]
        self.assertEqual({action.kind for action in root.edges}, kinds)

    def test_random_bootstrap_policy_can_cover_every_legal_action_kind(self):
        actions = [Action(kind) for kind in ActionKind]
        policy = RandomPolicy()
        state = RunState(gold=10)
        selected = {
            policy.choose(state, actions, random.Random(seed)).action.kind
            for seed in range(1000)
        }
        self.assertEqual(selected, set(ActionKind))

    def test_battle_leaf_resampling_grows_only_at_visit_thresholds(self):
        environment = ShopEnvironment(catalog())
        state = RunState(team=Team(), shop=Shop(), gold=0)
        evaluator = AdaptiveBattleEvaluator()
        search = PolicyGuidedSearch(
            environment,
            UniformEvaluator(),
            SearchConfig(
                simulations=8,
                candidate_actions=1,
                max_depth=2,
                battle_initial_simulations=4,
                battle_max_simulations=16,
            ),
            battle_evaluator=evaluator,
        )

        search.search(state, seed=9)

        self.assertEqual(evaluator.requested, [4, 8, 16])

    def test_simulator_evaluator_reuses_common_opponents_and_seeds(self):
        current_catalog = catalog()
        boards = []
        for index in range(4):
            opponent = current_catalog.pet_by_name("Fish").create()
            opponent.attack = index + 1
            boards.append(
                BoardSnapshot(
                    f"opponent-{index}",
                    "opponent",
                    1,
                    "Turtle",
                    Team.from_pets([opponent]),
                    version="test",
                )
            )

        class TrackingSimulator:
            def __init__(self):
                self.calls = []

            def simulate(self, player, opponent, *, seed, record_trace):
                self.calls.append((opponent.slots[0].attack, seed, record_trace))
                return BattleResult(
                    BattleResultKind.DRAW,
                    0,
                    player.clone(),
                    opponent.clone(),
                )

        simulator = TrackingSimulator()
        evaluator = SimulatorPopulationEvaluator(
            ShopEnvironment(current_catalog),
            simulator,
            OpponentPopulation(boards),
            simulations=4,
        )
        root = RunState(turn=1, pack="Turtle", version="test")
        evaluator.begin_search(root, random.Random(12))
        first = root.clone()
        first.awaiting_battle = True
        second = first.clone()
        second.team = Team.from_pets([current_catalog.pet_by_name("Ant").create()])

        evaluator.evaluate_battle(first, random.Random(1), simulations=3)
        evaluator.evaluate_battle(second, random.Random(999), simulations=3)

        self.assertEqual(simulator.calls[:3], simulator.calls[3:])
        self.assertTrue(all(not call[2] for call in simulator.calls))

    def test_small_budget_search_returns_legal_action(self):
        environment = ShopEnvironment(catalog())
        state = environment.reset(seed=5)
        search = PolicyGuidedSearch(
            environment,
            UniformEvaluator(),
            SearchConfig(simulations=12, candidate_actions=6, max_depth=4),
        )
        result = search.search(state, seed=9)
        self.assertIn(result.action, environment.legal_actions(state))
        self.assertEqual(sum(result.visit_counts.values()), 12)

    def test_search_cannot_expand_an_inverse_freeze_cycle(self):
        environment = ShopEnvironment(catalog())
        state = environment.reset(seed=5)
        state.team = Team()
        state.shop = Shop([ShopPet(state.shop.pets[0].pet)], [])
        state.gold = 0
        search = PolicyGuidedSearch(
            environment,
            ToggleFirstEvaluator(),
            SearchConfig(simulations=8, candidate_actions=2, max_depth=6, gumbel_scale=0.0),
        )
        result = search.search(state, seed=9)

        self.assertEqual(result.action, Action(ActionKind.FREEZE_PET, 0))
        root = search.transpositions[state.canonical_key()]
        frozen = next(iter(root.edges[Action(ActionKind.FREEZE_PET, 0)].outcomes.values()))
        self.assertNotIn(Action(ActionKind.UNFREEZE_PET, 0), frozen.edges)
        self.assertIn(Action(ActionKind.END_TURN), frozen.edges)

    def test_search_skips_an_edge_that_returns_to_an_ancestor(self):
        environment = ShopEnvironment(catalog())
        state = environment.reset(seed=5)
        ant = catalog().pet_by_name("Ant")
        fish = catalog().pet_by_name("Fish")
        state.team = Team.from_pets(
            [ant.create(instance_id=100), fish.create(instance_id=101)]
        )
        state.shop = Shop()
        state.gold = 0
        reorder = Action(ActionKind.REORDER, order=(1, 0))
        search = PolicyGuidedSearch(
            environment,
            ReorderFirstEvaluator(),
            SearchConfig(simulations=8, candidate_actions=3, max_depth=6, gumbel_scale=0.0),
        )
        result = search.search(state, seed=9)

        self.assertEqual(result.action, reorder)
        root = search.transpositions[state.canonical_key()]
        reordered = next(iter(root.edges[reorder].outcomes.values()))
        inverse = reordered.edges[reorder]
        self.assertEqual(inverse.visits, 0)
        self.assertIn(state.canonical_key(), inverse.outcomes)

    def test_end_turn_leaf_uses_and_caches_battle_evaluation(self):
        environment = ShopEnvironment(catalog())
        state = RunState(team=Team(), shop=Shop(), gold=0)
        battle_evaluator = FixedBattleEvaluator(0.75)
        search = PolicyGuidedSearch(
            environment,
            UniformEvaluator(),
            SearchConfig(simulations=6, candidate_actions=1, max_depth=2),
            battle_evaluator=battle_evaluator,
        )

        result = search.search(state, seed=9)

        end_turn = Action(ActionKind.END_TURN)
        self.assertEqual(result.action, end_turn)
        self.assertAlmostEqual(result.action_values[end_turn], 0.75)
        self.assertEqual(battle_evaluator.calls, 1)

    def test_end_turn_is_retained_for_battle_scoring_when_its_prior_is_zero(self):
        current_catalog = catalog()
        environment = ShopEnvironment(current_catalog)
        state = RunState(
            team=Team.from_pets([current_catalog.pet_by_name("Fish").create()]),
            shop=Shop(),
            gold=0,
        )
        search = PolicyGuidedSearch(
            environment,
            SellFirstEvaluator(),
            SearchConfig(
                simulations=1,
                candidate_actions=1,
                max_depth=2,
                gumbel_scale=0.0,
            ),
            battle_evaluator=FixedBattleEvaluator(0.75),
        )

        search.search(state, seed=9)

        root = search.transpositions[state.canonical_key()]
        self.assertEqual(list(root.edges), [Action(ActionKind.END_TURN)])

    def test_simulator_battle_evaluator_batches_next_turn_values(self):
        current_catalog = catalog()
        environment = ShopEnvironment(current_catalog)
        player = current_catalog.pet_by_name("Fish").create()
        player.attack = 50
        player.health = 50
        opponent = current_catalog.pet_by_name("Fish").create()
        opponent.attack = 1
        opponent.health = 1
        population = OpponentPopulation(
            [
                BoardSnapshot(
                    "weak-opponent",
                    "opponent",
                    1,
                    "Turtle",
                    Team.from_pets([opponent]),
                    version="test",
                )
            ]
        )
        continuation = TrophyValueEvaluator()
        evaluator = SimulatorPopulationEvaluator(
            environment,
            BattleSimulator(current_catalog),
            population,
            simulations=4,
            continuation_evaluator=continuation,
        )
        state = RunState(
            team=Team.from_pets([player]),
            turn=1,
            pack="Turtle",
            version="test",
            awaiting_battle=True,
        )

        value = evaluator.evaluate_battle(state, random.Random(12))

        self.assertAlmostEqual(value, 0.1)
        self.assertEqual(continuation.batch_sizes, [4])

    def test_simulator_battle_evaluator_uses_neutral_nonterminal_prior(self):
        current_catalog = catalog()
        player = current_catalog.pet_by_name("Fish").create()
        player.attack = 50
        player.health = 50
        opponent = current_catalog.pet_by_name("Fish").create()
        opponent.attack = 1
        opponent.health = 1
        population = OpponentPopulation(
            [
                BoardSnapshot(
                    "weak-opponent",
                    "opponent",
                    1,
                    "Turtle",
                    Team.from_pets([opponent]),
                    version="test",
                )
            ]
        )
        evaluator = SimulatorPopulationEvaluator(
            ShopEnvironment(current_catalog),
            BattleSimulator(current_catalog),
            population,
            simulations=3,
        )
        state = RunState(
            team=Team.from_pets([player]),
            turn=1,
            pack="Turtle",
            version="test",
            awaiting_battle=True,
        )

        self.assertEqual(evaluator.evaluate_battle(state, random.Random(4)), 0.5)


if __name__ == "__main__":
    unittest.main()
