import unittest

from clashvision.enums.clash_class import ClashClass


class TestClasClass(unittest.TestCase):

    def test_class_name(self):
        self.assertEqual(ClashClass.ELIXIR_STORAGE.value, 0)
        self.assertEqual(ClashClass.GOLD_STORAGE.value, 1)

    def test_from_int(self):
        self.assertEqual(ClashClass.from_int(0), ClashClass.ELIXIR_STORAGE)
        self.assertEqual(ClashClass.from_int(1), ClashClass.GOLD_STORAGE)
        with self.assertRaises(ValueError):
            ClashClass.from_int(2)

    def test_to_color(self):
        self.assertEqual(ClashClass.ELIXIR_STORAGE.to_color, "#F461FF")
        self.assertEqual(ClashClass.GOLD_STORAGE.to_color, "#FEE95F")

    def test_to_list(self):
        self.assertEqual(ClashClass.to_list(), ["ELIXIR_STORAGE", "GOLD_STORAGE"])

    def test_get_palette(self):
        palette = ClashClass.get_palette()
        self.assertEqual(palette, ["#F461FF", "#FEE95F"])
        self.assertEqual(len(palette), len(ClashClass))

    def test_str(self):
        self.assertEqual(str(ClashClass.ELIXIR_STORAGE), "Elixir Storage")
        self.assertEqual(str(ClashClass.GOLD_STORAGE), "Gold Storage")

if __name__ == "__main__":
    unittest.main()
