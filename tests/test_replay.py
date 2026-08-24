import json
import unittest

from sapai.data.library import _sample_query
from sapai.data.replay import ReplayParser
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


if __name__ == "__main__":
    unittest.main()
