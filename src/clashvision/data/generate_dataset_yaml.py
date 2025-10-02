import os

import yaml

from clashvision.core.path import get_data_path
from clashvision.enums.clash_class import ClashClass


def generate_dataset_yaml(dataset_path: str, class_names: list[str], output_path: str = "dataset.yaml"):
    """
    Generate a YOLO dataset.yaml file.

    Args:
        dataset_path (str): Root path of dataset (contains images/ and labels/ folders).
        class_names (list): List of class names.
        output_path (str): Path to save dataset.yaml file.
    """
    yaml_dict = {
        "train": os.path.join(dataset_path, "images/train"),
        "val": os.path.join(dataset_path, "images/val"),
        "nc": len(class_names),
        "names": class_names
    }

    # Save as YAML
    with open(output_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    print(f"✅ dataset.yaml generated at: {output_path}")


# Example usage
if __name__ == "__main__":
    dataset_root: str = str(get_data_path())
    class_names: list[str] = ClashClass.to_list()

    generate_dataset_yaml(dataset_root, class_names, "dataset.yaml")
