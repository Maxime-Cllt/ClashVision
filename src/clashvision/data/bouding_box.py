import os

import matplotlib.pyplot as plt

from clashvision.core.path import get_labels_path

if __name__ == "__main__":

    label_dir: str = os.path.join(get_labels_path(), "train")

    img_w, img_h = 640, 640  # or actual sizes if varied
    box_widths, box_heights = [], []

    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                _, x, y, w, h = map(float, line.split())
                box_widths.append(w)
                box_heights.append(h)

    plt.scatter(box_widths, box_heights, alpha=0.3)
    plt.xlabel("Box Width (normalized)")
    plt.ylabel("Box Height (normalized)")
    plt.title("Bounding Box Size Distribution")
    plt.show()
