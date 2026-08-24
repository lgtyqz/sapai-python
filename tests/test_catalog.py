import unittest

from tests.helpers import catalog


class CatalogTest(unittest.TestCase):
    def test_loads_current_catalog_and_pack(self):
        values = catalog()
        ant = values.pet_by_name("Ant")
        self.assertEqual((ant.tier, ant.attack, ant.health), (1, 2, 2))
        self.assertGreaterEqual(len(values.pack_pets("Turtle")), 60)
        self.assertIn("Faint", ant.ability_text[0])


if __name__ == "__main__":
    unittest.main()
