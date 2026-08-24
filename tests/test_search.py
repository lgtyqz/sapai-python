import unittest

from sapai.search.stochastic import PolicyGuidedSearch, SearchConfig, UniformEvaluator
from sapai.sim.shop import ShopEnvironment
from tests.helpers import catalog


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


if __name__ == "__main__":
    unittest.main()
