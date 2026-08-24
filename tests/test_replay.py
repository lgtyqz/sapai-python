import json
import unittest

from sapai.data.replay import ReplayParser
from tests.helpers import catalog


class ReplayParserTest(unittest.TestCase):
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
                        }
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


if __name__ == "__main__":
    unittest.main()
