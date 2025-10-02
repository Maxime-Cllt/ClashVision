import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from clashvision.core.path import get_labels_path
from clashvision.enums.clash_class import ClashClass

if __name__ == "__main__":
    label_dir = os.path.join(get_labels_path(), "train")

    class_counts = Counter()
    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        palette=ClashClass.get_palette(),
    )
    plt.xlabel("Class ID")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution in Training Dataset")
    plt.tight_layout()
    plt.show()
