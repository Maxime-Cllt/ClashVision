import unittest
from pathlib import Path

from clashvision.core.path import get_project_root


class TestPath(unittest.TestCase):

    def test_get_project_root(self):
        # Test if the function returns a Path object
        root = get_project_root()
        self.assertIsInstance(root, Path)

        # Test if the root path exists
        self.assertTrue(root.exists())

        # Test if the root path is indeed the project root by checking for a known file
        self.assertTrue((root / "README.md").exists())

        # Test with a custom marker
        custom_marker = ".gitignore"
        custom_root = get_project_root(marker=custom_marker)
        self.assertTrue((custom_root / custom_marker).exists())

    def test_get_data_path(self):
        from clashvision.core.path import get_data_path
        data_path = get_data_path()
        self.assertIsInstance(data_path, Path)
        self.assertTrue(Path(data_path).exists())

    def test_get_images_path(self):
        from clashvision.core.path import get_images_path
        images_path = get_images_path()
        self.assertIsInstance(images_path, Path)
        self.assertTrue(Path(images_path).exists())


if __name__ == "__main__":
    unittest.main()
