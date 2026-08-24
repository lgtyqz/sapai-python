import copy
import unittest

from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Pet, Team
from sapai.sim.rules import RuleBook, evaluate
from tests.helpers import catalog


class RuleBookTest(unittest.TestCase):
    def test_rulebook_tracks_audited_source_and_covers_turtle_pack(self):
        rules = RuleBook.turtle()
        turtle_names = {spec.name for spec in catalog().pack_pets("Turtle")}
        turtle_tokens = {
            "Bee",
            "Bus",
            "Chick",
            "Dirty Rat",
            "Ram",
            "Zombie Cricket",
            "Zombie Fly",
        }
        self.assertEqual(rules.source["commit"], "d165eb0a02f8aa0b54d72ed1d5490a44390d07f4")
        self.assertEqual(rules.generated_pets, turtle_tokens)
        self.assertEqual(rules.supported_pets, turtle_names | turtle_tokens)
        self.assertIn("Pizza", rules.supported_foods)

    def test_expression_language_is_small_and_deterministic(self):
        value = evaluate({"ceil": {"mul": ["attack", 0.5, "level"]}}, {"attack": 5, "level": 2})
        self.assertEqual(value, 5)

    def test_battle_behavior_can_change_in_data_only(self):
        data = copy.deepcopy(RuleBook.turtle().data)
        data["pets"]["Ant"]["rules"][0]["effects"][0]["attack"] = 7
        rules = RuleBook(data)
        rules.validate()
        simulator = BattleSimulator(catalog(), rules)
        ant = catalog().pet_by_name("Ant").create()
        ant.health = 1
        result = simulator.simulate(
            Team.from_pets([ant, Pet(100, "Friend", 1, 1, 100)]),
            Team.from_pets([Pet(101, "Enemy", 1, 1, 20)]),
            seed=3,
        )
        self.assertEqual(result.player.living()[0].effective_attack, 8)


if __name__ == "__main__":
    unittest.main()
