import unittest

from tests.helpers import catalog


class CatalogTest(unittest.TestCase):
    def test_loads_current_catalog_and_pack(self):
        values = catalog()
        ant = values.pet_by_name("Ant")
        self.assertEqual((ant.tier, ant.attack, ant.health), (1, 2, 2))
        self.assertGreaterEqual(len(values.pack_pets("Turtle")), 60)
        self.assertIn("Faint", ant.ability_text[0])

    def test_recognizes_all_explicit_no_ability_wordings(self):
        values = catalog()
        expected = {
            "Bee",
            "Bus",
            "Chick",
            "Dirty Rat",
            "Lizard Tail",
            "Loyal Chinchilla",
            "Ram",
            "Sloth",
            "Smallest Slug",
            "Zombie Cricket",
            "Zombie Fly",
        }
        actual = {
            spec.name
            for spec in values.pets.values()
            if values.pet_has_no_ability(spec.id, spec.name)
        }
        self.assertTrue(expected <= actual)
        for name in actual:
            spec = values.pet_by_name(name)
            self.assertTrue(
                all(
                    "no ability" in text.casefold()
                    or "no special ability" in text.casefold()
                    for text in spec.ability_text
                )
            )


if __name__ == "__main__":
    unittest.main()
