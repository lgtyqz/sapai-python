import random
import unittest

from sapai.sim.actions import Action, ActionKind
from sapai.sim.models import Food, Pet, Shop, ShopPet, Team
from sapai.sim.shop import ShopEnvironment
from tests.helpers import catalog


class ShopEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()
        self.environment = ShopEnvironment(self.catalog)

    def test_seeded_reset_is_reproducible(self):
        first = self.environment.reset(seed=42)
        second = self.environment.reset(seed=42)
        self.assertEqual(first.canonical_key(), second.canonical_key())
        self.assertEqual(len(first.shop.pets), 3)
        self.assertEqual(len(first.shop.foods), 1)

    def test_roll_preserves_frozen_offer(self):
        state = self.environment.reset(seed=4)
        name = state.shop.pets[0].pet.name
        state.shop.pets[0].frozen = True
        rolled = self.environment.step(state, Action(ActionKind.ROLL), random.Random(9)).state
        self.assertEqual(rolled.shop.pets[0].pet.name, name)
        self.assertTrue(rolled.shop.pets[0].frozen)
        self.assertEqual(rolled.gold, 9)

    def test_buy_targets_are_explicit(self):
        state = self.environment.reset(seed=4)
        actions = self.environment.legal_actions(state)
        targets = {
            action.target
            for action in actions
            if action.kind is ActionKind.BUY_PET and action.source == 0
        }
        self.assertEqual(targets, set(range(5)))

    def test_level_up_rewards_come_from_next_tier(self):
        state = self.environment.reset(seed=3)
        ant = self.catalog.pet_by_name("Ant")
        team_ant = ant.create(instance_id=100)
        team_ant.experience = 1
        state.team = Team.from_pets([team_ant])
        state.shop = Shop([ShopPet(ant.create(instance_id=101))], [])
        action = Action(ActionKind.BUY_MERGE_PET, 0, 0)
        updated = self.environment.step(state, action, random.Random(2)).state
        rewards = [offer for offer in updated.shop.pets if offer.reward_group is not None]
        self.assertEqual(updated.team.slots[0].level, 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual({offer.pet.tier for offer in rewards}, {2})

    def test_cat_multiplies_one_food_event_not_each_target(self):
        state = self.environment.reset(seed=2)
        cat = self.catalog.pet_by_name("Cat").create(instance_id=1)
        ant = self.catalog.pet_by_name("Ant").create(instance_id=2)
        fish = self.catalog.pet_by_name("Fish").create(instance_id=3)
        state.team = Team.from_pets([cat, ant, fish])
        pizza = self.catalog.food_by_name("Pizza").create()
        state.shop = Shop([], [pizza])
        state.gold = 3
        updated = self.environment.step(
            state, Action(ActionKind.BUY_FOOD, 0, -1), random.Random(1)
        ).state
        total_gain = sum(pet.attack + pet.health for pet in updated.team.living()) - sum(
            pet.attack + pet.health for pet in state.team.living()
        )
        self.assertEqual(total_gain, 16)  # two targets, +4/+4 each at level-1 Cat
        self.assertEqual(cat.metadata.get("turn_uses"), None)  # input remains immutable

    def test_turkey_permanently_buffs_a_bought_pet(self):
        state = self.environment.reset(seed=2)
        turkey = self.catalog.pet_by_name("Turkey").create(instance_id=1)
        ant = self.catalog.pet_by_name("Ant").create(instance_id=2)
        original = (ant.attack, ant.health)
        state.team = Team.from_pets([turkey])
        state.shop = Shop([ShopPet(ant)], [])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_PET, 0, 1), random.Random(1)
        ).state
        bought = updated.team.slots[1]
        self.assertEqual((bought.attack, bought.health), (original[0] + 3, original[1] + 1))
        self.assertEqual((bought.temporary_attack, bought.temporary_health), (0, 0))

    def test_horse_buff_on_bought_pet_is_temporary(self):
        state = self.environment.reset(seed=2)
        horse = self.catalog.pet_by_name("Horse").create(instance_id=1)
        ant = self.catalog.pet_by_name("Ant").create(instance_id=2)
        original_attack = ant.attack
        state.team = Team.from_pets([horse])
        state.shop = Shop([ShopPet(ant)], [])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_PET, 0, 1), random.Random(1)
        ).state
        bought = updated.team.slots[1]
        self.assertEqual(bought.attack, original_attack)
        self.assertEqual(bought.temporary_attack, 1)

    def test_shark_permanently_buffs_when_friend_faints_in_shop(self):
        state = self.environment.reset(seed=2)
        shark = self.catalog.pet_by_name("Shark").create(instance_id=2)
        original = (shark.attack, shark.health)
        state.team = Team.from_pets([Pet(999, "Target", 1, 1, 1), shark])
        state.shop = Shop([], [Food(0, "Sleeping Pill", 2, cost=1)])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_FOOD, 0, 0), random.Random(1)
        ).state
        surviving_shark = updated.team.living()[0]
        self.assertEqual(
            (surviving_shark.attack, surviving_shark.health),
            (original[0] + 2, original[1] + 2),
        )
        self.assertEqual(
            (surviving_shark.temporary_attack, surviving_shark.temporary_health),
            (0, 0),
        )

    def test_shop_faint_summon_triggers_turkey(self):
        state = self.environment.reset(seed=2)
        cricket = self.catalog.pet_by_name("Cricket").create(instance_id=1)
        turkey = self.catalog.pet_by_name("Turkey").create(instance_id=2)
        state.team = Team.from_pets([cricket, turkey])
        state.shop = Shop([], [Food(0, "Sleeping Pill", 2, cost=1)])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_FOOD, 0, 0), random.Random(1)
        ).state
        zombie = updated.team.slots[0]
        self.assertEqual(zombie.name, "Zombie Cricket")
        self.assertEqual((zombie.attack, zombie.health), (4, 2))
        self.assertEqual((zombie.temporary_attack, zombie.temporary_health), (0, 0))

    def test_shop_damage_resolves_chained_faints(self):
        state = self.environment.reset(seed=2)
        hedgehog = self.catalog.pet_by_name("Hedgehog").create(instance_id=1)
        shark = self.catalog.pet_by_name("Shark").create(instance_id=3)
        original_attack = shark.attack
        state.team = Team.from_pets([hedgehog, Pet(999, "Fragile", 1, 1, 1, instance_id=2), shark])
        state.shop = Shop([], [Food(0, "Sleeping Pill", 2, cost=1)])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_FOOD, 0, 0), random.Random(1)
        ).state
        self.assertEqual([pet.name for pet in updated.team.living()], ["Shark"])
        self.assertEqual(updated.team.living()[0].attack, original_attack + 4)

    def test_bought_scorpion_runs_summoned_ability(self):
        state = self.environment.reset(seed=2)
        scorpion = self.catalog.pet_by_name("Scorpion").create(instance_id=1)
        state.team = Team()
        state.shop = Shop([ShopPet(scorpion)], [])
        updated = self.environment.step(
            state, Action(ActionKind.BUY_PET, 0, 0), random.Random(1)
        ).state
        self.assertEqual(updated.team.slots[0].perk, "Peanut")


if __name__ == "__main__":
    unittest.main()
