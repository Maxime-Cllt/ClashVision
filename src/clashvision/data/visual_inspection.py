import os
import random

import cv2

from clashvision.core.path import get_labels_path, get_images_path
from clashvision.enums.clash_class import ClashClass


def visualize_sample(image_dir: str, label_dir: str, class_names: list[str]) -> None:
    img_file = random.choice(os.listdir(image_dir))
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            cls = int(cls)
            x, y, bw, bh = x * w, y * h, bw * w, bh * h
            x1, y1, x2, y2 = int(x - bw / 2), int(y - bh / 2), int(x + bw / 2), int(y + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Sample", img)
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_path_train = os.path.join(get_images_path(), "train")
    labels_path_train = os.path.join(get_labels_path(), "train")

    visualize_sample(images_path_train, labels_path_train, ClashClass.to_list())
