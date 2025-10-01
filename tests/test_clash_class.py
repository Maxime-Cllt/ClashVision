import unittest


class TestClasClass(unittest.TestCase):

    def test_class_name(self):
        from src.clashvision.enums.clash_class import ClashClass

        self.assertEqual(ClashClass.ELIXIR_STORAGE, 0)
        self.assertEqual(ClashClass.GOLD_STORAGE, 1)


if __name__ == "__main__":
    unittest.main()
