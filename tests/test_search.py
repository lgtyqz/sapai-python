import unittest

from sapai.search.stochastic import PolicyGuidedSearch, SearchConfig, UniformEvaluator
from sapai.sim.actions import Action, ActionKind
from sapai.sim.models import Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
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


if __name__ == "__main__":
    unittest.main()
