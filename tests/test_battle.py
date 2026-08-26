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
        parrot.experience = 2
        enemies = [
            Pet(20, "Tank A", 1, 1, 100),
            Pet(21, "Tank B", 1, 1, 100),
            Pet(22, "Tank C", 1, 1, 100),
        ]
        result = self.simulator.simulate(
            Team.from_pets([mosquito, parrot]), Team.from_pets(enemies), seed=5
        )
        first_attack = next(index for index, line in enumerate(result.log) if " attacks " in line)
        pre_attack_damage = [line for line in result.log[:first_attack] if "takes 1" in line]
        self.assertEqual(len(pre_attack_damage), 3)

    def test_whale_activates_swallowed_deer_faint_ability(self):
        deer = self.catalog.pet_by_name("Deer").create()
        whale = self.catalog.pet_by_name("Whale").create()
        whale.health = 100
        enemy = Pet(20, "Tank", 1, 1, 100)

        result = self.simulator.simulate(
            Team.from_pets([deer, whale]),
            Team.from_pets([enemy]),
            seed=1,
        )

        started = next(
            frame for frame in result.frames if frame.label == "Start-of-battle abilities resolved"
        )
        names = [pet.name for pet in started.player.living()]
        self.assertEqual(names, ["Bus", "Whale"])
        bus = started.player.slots[0]
        self.assertEqual((bus.attack, bus.health, bus.perk), (5, 3, "Chili"))
        swallowed = next(frame for frame in result.frames if frame.event == "ability")
        self.assertEqual(swallowed.label, "Whale swallows Deer")
        self.assertEqual([pet.name for pet in swallowed.player.living()], ["Bus", "Whale"])

    def test_defending_boar_triggers_before_attack(self):
        boar = self.catalog.pet_by_name("Boar").create()
        attacker = Pet(20, "Attacker", 1, 1, 200)
        reserve = Pet(21, "Reserve", 1, 1, 200)

        result = self.simulator.simulate(
            Team.from_pets([boar]),
            Team.from_pets([attacker, reserve]),
            seed=1,
        )

        first_attack = next(line for line in result.log if "Attacker takes" in line)
        self.assertIn(f"Attacker takes {boar.attack + 4}", first_attack)

    def test_defending_elephant_triggers_after_attack(self):
        elephant = self.catalog.pet_by_name("Elephant").create()
        elephant.experience = 5
        elephant.health = 100
        blowfish = self.catalog.pet_by_name("Blowfish").create()
        blowfish.experience = 2
        blowfish.health = 100
        opponents = [
            Pet(20, "Tank A", 1, 1, 200),
            Pet(21, "Tank B", 1, 1, 200),
            Pet(22, "Tank C", 1, 1, 200),
        ]

        result = self.simulator.simulate(
            Team.from_pets([elephant, blowfish]),
            Team.from_pets(opponents),
            seed=1,
        )

        second_attack = next(
            index for index, line in enumerate(result.log[1:], start=1) if " attacks " in line
        )
        first_attack_log = result.log[:second_attack]
        self.assertEqual(sum("takes 6" in line for line in first_attack_log), 3)

    def test_tiger_repeats_friend_summoned_ability_at_tiger_level(self):
        cricket = self.catalog.pet_by_name("Cricket").create()
        cricket.health = 1
        turkey = self.catalog.pet_by_name("Turkey").create()
        tiger = self.catalog.pet_by_name("Tiger").create()
        tiger.experience = 2
        enemy = Pet(20, "Tank", 1, 10, 100)

        result = self.simulator.simulate(
            Team.from_pets([cricket, turkey, tiger]),
            Team.from_pets([enemy]),
            seed=1,
        )

        zombies = [
            pet
            for frame in result.frames
            for pet in frame.player.living()
            if pet.name == "Zombie Cricket"
        ]
        self.assertTrue(zombies)
        self.assertEqual(max(pet.effective_attack for pet in zombies), 10)
        self.assertEqual(max(pet.effective_health for pet in zombies), 4)

    def test_level_three_elephant_triggers_level_two_blowfish_three_times(self):
        elephant = self.catalog.pet_by_name("Elephant").create()
        elephant.experience = 5
        elephant.attack = 1
        elephant.health = 100
        blowfish = self.catalog.pet_by_name("Blowfish").create()
        blowfish.experience = 2
        blowfish.health = 100
        blowfish.temporary_health = 7
        enemy = Pet(20, "Tank", 1, 1, 200)

        result = self.simulator.simulate(
            Team.from_pets([elephant, blowfish]),
            Team.from_pets([enemy]),
            seed=1,
        )

        next_attack = next(
            index for index, line in enumerate(result.log[1:], start=1) if " attacks " in line
        )
        first_attack_log = result.log[:next_attack]
        self.assertEqual(sum("Tank takes 6" in line for line in first_attack_log), 3)

    def test_fly_does_not_trigger_when_zombie_fly_faints(self):
        zombie_fly = self.catalog.pet_by_name("Zombie Fly").create()
        zombie_fly.attack = 1
        zombie_fly.health = 1
        fly = self.catalog.pet_by_name("Fly").create()
        fly.health = 100
        enemy = Pet(20, "Tank", 1, 10, 100)

        result = self.simulator.simulate(
            Team.from_pets([zombie_fly, fly]),
            Team.from_pets([enemy]),
            seed=1,
        )

        first_round = next(frame for frame in result.frames if frame.label == "Round 1 resolved")
        self.assertEqual([pet.name for pet in first_round.player.living()], ["Fly"])
        fly_state = first_round.player.slots[0]
        self.assertEqual(fly_state.metadata["battle"]["uses"], {})

    def test_mushroom_revive_preserves_ability_trigger_usage(self):
        hippo = self.catalog.pet_by_name("Hippo").create()
        hippo.attack = 10
        hippo.health = 20
        hippo.perk = "Mushroom"
        hippo.triggers_consumed = 2
        opponents = [
            Pet(20, "Weakling", 1, 1, 1),
            Pet(21, "Tank", 1, 100, 500),
        ]

        result = self.simulator.simulate(
            Team.from_pets([hippo]),
            Team.from_pets(opponents),
            seed=1,
        )

        revived = next(
            pet
            for frame in result.frames
            for pet in frame.player.living()
            if pet.name == "Hippo" and pet.attack == 1 and pet.health == 1 and pet.perk is None
        )
        self.assertEqual(revived.triggers_consumed, 2)
        self.assertEqual(revived.metadata["battle"]["uses"]["Hippo:knockout:0"], 1)

    def test_mushroomed_fly_does_not_trigger_on_its_own_faint(self):
        fly = self.catalog.pet_by_name("Fly").create()
        fly.health = 1
        fly.perk = "Mushroom"
        enemy = Pet(20, "Tank", 1, 20, 100)

        result = self.simulator.simulate(
            Team.from_pets([fly]),
            Team.from_pets([enemy]),
            seed=1,
        )

        first_round = next(frame for frame in result.frames if frame.label == "Round 1 resolved")
        self.assertEqual([pet.name for pet in first_round.player.living()], ["Fly"])
        revived = first_round.player.slots[0]
        self.assertEqual((revived.attack, revived.health, revived.perk), (1, 1, None))
        self.assertEqual(revived.metadata["battle"]["uses"], {})

    def test_another_fly_can_trigger_on_a_mushroomed_fly_faint(self):
        mushroomed = self.catalog.pet_by_name("Fly").create()
        mushroomed.health = 1
        mushroomed.perk = "Mushroom"
        observer = self.catalog.pet_by_name("Fly").create()
        observer.health = 100
        enemy = Pet(20, "Tank", 1, 20, 100)

        result = self.simulator.simulate(
            Team.from_pets([mushroomed, observer]),
            Team.from_pets([enemy]),
            seed=1,
        )

        first_round = next(frame for frame in result.frames if frame.label == "Round 1 resolved")
        self.assertEqual(
            [pet.name for pet in first_round.player.living()],
            ["Zombie Fly", "Fly", "Fly"],
        )
        revived, other = first_round.player.slots[1:3]
        self.assertEqual(revived.metadata["battle"]["uses"], {})
        self.assertEqual(other.metadata["battle"]["uses"]["Fly:friend_fainted:0"], 1)

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
        sloth = self.catalog.pet_by_name("Sloth").create()
        self.simulator.assert_team_supported(Team.from_pets([sloth]))

    def test_known_pet_with_unimplemented_ability_still_fails_coverage(self):
        beetle = self.catalog.pet_by_name("Beetle").create()
        with self.assertRaises(UnsupportedRuleError):
            self.simulator.assert_team_supported(Team.from_pets([beetle]))
        butterfly = self.catalog.pet_by_name("Butterfly").create()
        with self.assertRaises(UnsupportedRuleError):
            self.simulator.assert_team_supported(Team.from_pets([butterfly]))

    def test_unsupported_perk_uses_tagged_no_effect_fallback(self):
        otter = self.catalog.pet_by_name("Otter").create()
        otter.perk = "Silly"
        self.simulator.assert_team_supported(Team.from_pets([otter]))
        self.assertEqual(otter.metadata["perk_fallback"], "unsupported_rulebook_perk")


if __name__ == "__main__":
    unittest.main()
