import unittest

from sapai.sim.battle import BattleResultKind, BattleSimulator, UnsupportedRuleError
from sapai.sim.models import Pet, Team
from tests.helpers import catalog


class BattleSimulatorTest(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()
        self.simulator = BattleSimulator(self.catalog)

    def test_base_combat(self):
        player = Team.from_pets([Pet(1, "Big", 1, 10, 10)])
        opponent = Team.from_pets([Pet(2, "Small", 1, 1, 1)])
        result = self.simulator.simulate(player, opponent, seed=1)
        self.assertEqual(result.outcome, BattleResultKind.PLAYER_WIN)
        self.assertEqual(result.rounds, 1)

    def test_ant_faint_buffs_friend(self):
        ant = self.catalog.pet_by_name("Ant").create()
        ant.health = 1
        friend = Pet(999, "Friend", 1, 1, 100)
        enemy = Pet(998, "Enemy", 1, 1, 20)
        result = self.simulator.simulate(
            Team.from_pets([ant, friend]), Team.from_pets([enemy]), seed=3
        )
        remaining = result.player.living()
        self.assertEqual(result.outcome, BattleResultKind.PLAYER_WIN)
        self.assertEqual(remaining[0].effective_attack, 2)

    def test_melon_is_consumed(self):
        tank = Pet(1, "Tank", 1, 1, 10, perk="Melon")
        attacker = Pet(2, "Attacker", 1, 25, 50)
        result = self.simulator.simulate(Team.from_pets([tank]), Team.from_pets([attacker]), seed=1)
        self.assertEqual(result.outcome, BattleResultKind.OPPONENT_WIN)

    def test_parrot_copies_nearest_pet_before_battle(self):
        mosquito = self.catalog.pet_by_name("Mosquito").create()
        parrot = self.catalog.pet_by_name("Parrot").create()
        enemies = [Pet(20, "Tank A", 1, 1, 100), Pet(21, "Tank B", 1, 1, 100)]
        result = self.simulator.simulate(
            Team.from_pets([mosquito, parrot]), Team.from_pets(enemies), seed=5
        )
        first_attack = next(index for index, line in enumerate(result.log) if " attacks " in line)
        pre_attack_damage = [line for line in result.log[:first_attack] if "takes 1" in line]
        self.assertEqual(len(pre_attack_damage), 2)

    def test_turtle_tokens_and_no_ability_catalog_pets_are_supported(self):
        token_names = (
            "Bee",
            "Bus",
            "Chick",
            "Dirty Rat",
            "Ram",
            "Zombie Cricket",
            "Zombie Fly",
        )
        team = Team.from_pets(self.catalog.pet_by_name(name).create() for name in token_names[:5])
        self.simulator.assert_team_supported(team)
        self.simulator.assert_team_supported(
            Team.from_pets(self.catalog.pet_by_name(name).create() for name in token_names[5:])
        )

    def test_known_pet_with_unimplemented_ability_still_fails_coverage(self):
        beetle = self.catalog.pet_by_name("Beetle").create()
        with self.assertRaises(UnsupportedRuleError):
            self.simulator.assert_team_supported(Team.from_pets([beetle]))


if __name__ == "__main__":
    unittest.main()
