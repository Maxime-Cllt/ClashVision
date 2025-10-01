from pathlib import Path


def get_project_root(marker="README.md") -> Path:
    """
    Get the root path of the project by searching for a marker file.
    :param marker:  The name of the marker file to identify the project root (default is "README.md").
    :return:  Path object pointing to the project root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'")


def get_data_path() -> Path:
    """
    Get the path to the data directory within the project.
    :return:  Path object pointing to the data directory.
    """
    import os
    return Path(os.path.join(get_project_root(), "data"))


def get_images_path() -> Path:
    """
    Get the path to the images directory within the data directory.
    :return:  Path object pointing to the images directory.
    """
    return get_data_path() / "images"


def get_labels_path() -> Path:
    """
    Get the path to the labels directory within the data directory.
    :return:  Path object pointing to the labels directory.
    """
    return get_data_path() / "labels"
