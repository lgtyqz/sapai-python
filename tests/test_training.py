import unittest

from sapai.data.replay import BoardSnapshot
from sapai.ml.training import label_board_pairs
from sapai.sim.battle import BattleSimulator
from sapai.sim.models import Team
from tests.helpers import catalog


class TrainingDataTest(unittest.TestCase):
    def test_labels_compatible_board_pairs(self):
        current_catalog = catalog()
        ant = current_catalog.pet_by_name("Ant")
        boards = [
            BoardSnapshot(str(index), "player", 1, "Turtle", Team.from_pets([ant.create()]))
            for index in range(2)
        ]
        examples = label_board_pairs(
            boards,
            BattleSimulator(current_catalog),
            examples=1,
            simulations_per_pair=2,
            seed=1,
        )
        self.assertEqual(examples[0].target, (0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
