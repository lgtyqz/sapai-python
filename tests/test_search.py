import random
import unittest

from sapai.data.replay import BoardSnapshot
from sapai.search.stochastic import PolicyGuidedSearch, SearchConfig, UniformEvaluator
from sapai.sim.actions import Action, ActionKind
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import RunState, Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
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

    def test_simulator_battle_evaluator_uses_model_free_wdl_score(self):
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

        self.assertEqual(evaluator.evaluate_battle(state, random.Random(4)), 1.0)


if __name__ == "__main__":
    unittest.main()
