import json
import unittest

from sapai.data.library import _sample_query
from sapai.data.replay import BoardSnapshot, ReplayParser, board_is_pack_compatible
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Team
from tests.helpers import catalog


class ReplayParserTest(unittest.TestCase):
    def test_library_query_omits_ambiguous_null_parameter(self):
        packed_query, packed_parameters = _sample_query(mode=0, pack="Turtle", limit=10_000)
        self.assertNotIn("IS NULL", packed_query)
        self.assertEqual(packed_parameters, (0, "Turtle", "Turtle", 10_000))

        unfiltered_query, unfiltered_parameters = _sample_query(mode=0, pack=None, limit=100)
        self.assertNotIn("pack =", unfiltered_query)
        self.assertEqual(unfiltered_parameters, (0, 100))

    def test_translates_replay_coordinates_and_stats(self):
        battle = {
            "UserBoard": {
                "Pack": 0,
                "Tur": 3,
                "Mins": {
                    "Items": [
                        {
                            "Enu": 0,
                            "Poi": {"x": 4},
                            "At": {"Perm": 2, "Temp": 1},
                            "Hp": {"Perm": 2, "Temp": 0},
                            "Exp": 0,
                        },
                        {
                            "Enu": -1,
                            "Poi": {"x": 3},
                            "At": {"Perm": 0, "Temp": 0},
                            "Hp": {"Perm": 0, "Temp": 0},
                        },
                    ]
                },
            },
            "OpponentBoard": {"Pack": 0, "Tur": 3, "Mins": {"Items": []}},
        }
        replay = {"Actions": [{"Type": 0, "Battle": json.dumps(battle)}]}
        boards = ReplayParser(catalog()).parse_replay(replay, replay_id="abc")
        self.assertEqual(len(boards), 2)
        self.assertEqual(boards[0].team.slots[0].name, "Ant")
        self.assertEqual(boards[0].team.slots[0].attack, 3)
        self.assertEqual(boards[0].pack, "Turtle")
        self.assertNotIn("Pet #-1", [pet.name for pet in boards[0].team.slots if pet])

    def test_unknown_pet_id_is_tagged_for_vanilla_fallback(self):
        battle = {
            "UserBoard": {
                "Pack": 0,
                "Tur": 2,
                "Mins": {
                    "Items": [
                        {
                            "Enu": 999_999,
                            "Poi": {"x": 4},
                            "At": {"Perm": 7, "Temp": 1},
                            "Hp": {"Perm": 8, "Temp": 2},
                        }
                    ]
                },
            },
            "OpponentBoard": {"Pack": 0, "Tur": 2, "Mins": {"Items": []}},
        }
        board = ReplayParser(catalog()).parse_battle(battle)[0]
        pet = board.team.slots[0]
        self.assertIsNotNone(pet)
        self.assertEqual((pet.name, pet.attack, pet.health), ("Pet #999999", 8, 10))
        self.assertEqual(pet.metadata["vanilla_fallback"], "unknown_pet_id")
        BattleSimulator(catalog()).assert_team_supported(board.team)

    def test_perk_zero_and_unknown_perk_ids_are_not_dropped(self):
        battle = {
            "UserBoard": {
                "Pack": 0,
                "Tur": 2,
                "Mins": {
                    "Items": [
                        {
                            "Enu": 0,
                            "Perk": 0,
                            "Poi": {"x": 4},
                            "At": {"Perm": 2},
                            "Hp": {"Perm": 2},
                        },
                        {
                            "Enu": 0,
                            "Perk": 999_999,
                            "Poi": {"x": 3},
                            "At": {"Perm": 2},
                            "Hp": {"Perm": 2},
                        },
                    ]
                },
            },
            "OpponentBoard": {"Pack": 0, "Tur": 2, "Mins": {"Items": []}},
        }
        team = ReplayParser(catalog()).parse_battle(battle)[0].team
        self.assertEqual(team.slots[0].perk, "Coconut")
        self.assertEqual(team.slots[1].perk, "Perk #999999")
        BattleSimulator(catalog()).assert_team_supported(team)
        self.assertEqual(team.slots[1].metadata["perk_fallback"], "unknown_perk_id")

    def test_missing_pack_does_not_default_to_turtle(self):
        battle = {
            "UserBoard": {"Tur": 1, "Mins": {"Items": []}},
            "OpponentBoard": {"Tur": 1, "Mins": {"Items": []}},
        }
        boards = ReplayParser(catalog()).parse_battle(battle)
        self.assertEqual([board.pack for board in boards], ["Unknown", "Unknown"])

    def test_pack_compatibility_rejects_known_cross_pack_pets(self):
        current_catalog = catalog()
        cuddle_toad = current_catalog.pet_by_name("Cuddle Toad").create()
        contaminated = BoardSnapshot(
            "cross-pack",
            "player",
            1,
            "Turtle",
            Team.from_pets([cuddle_toad]),
        )
        self.assertFalse(board_is_pack_compatible(contaminated, current_catalog, "Turtle"))

        sloth = current_catalog.pet_by_name("Sloth").create()
        neutral = BoardSnapshot("neutral", "player", 1, "Turtle", Team.from_pets([sloth]))
        self.assertTrue(board_is_pack_compatible(neutral, current_catalog, "Turtle"))


if __name__ == "__main__":
    unittest.main()
