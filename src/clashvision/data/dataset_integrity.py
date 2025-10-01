from clashvision.core.path import get_images_path, get_labels_path


def check_dataset_structure(image_dir: str, label_dir: str) -> None:
    """
    Check if each image has a corresponding label file and vice versa.
    :param image_dir:  Directory containing image files.
    :param label_dir:  Directory containing label files.
    :return: None
    """
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if not missing_labels and not missing_images:
        print("Dataset structure is valid. All images have corresponding labels and vice versa in directory:",
              image_dir)
    else:
        print("Missing labels:", missing_labels)
        print("Missing images:", missing_images)


if __name__ == '__main__':
    import os

    images_path_train = os.path.join(get_images_path(), "train")
    labels_path_train = os.path.join(get_labels_path(), "train")

    images_path_val = os.path.join(get_images_path(), "val")
    labels_path_val = os.path.join(get_labels_path(), "val")

    check_dataset_structure(images_path_train, labels_path_train)
    check_dataset_structure(images_path_val, labels_path_val)
