import os
import random

import cv2

from clashvision.core.path import get_labels_path, get_images_path
from clashvision.enums.clash_class import ClashClass


class MultiImageVisualizer:
    def __init__(self, image_dir: str, label_dir: str, num_windows: int = 10):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = ClashClass.to_list()
        self.color_map: dict = {cls.value: cls.to_bgr for cls in ClashClass}  # Added () after to_rgb        self.num_windows = num_windows
        self.num_windows = num_windows
        self.image_files = os.listdir(image_dir)
        self.windows = {}

    def load_and_process_image(self, img_file: str):
        """Load an image and draw bounding boxes with labels"""
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w, _ = img.shape

        # Draw bounding boxes if label file exists
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, bw, bh = map(float, parts[:5])
                        cls = int(cls)
                        x, y, bw, bh = x * w, y * h, bw * w, bh * h
                        x1, y1, x2, y2 = (
                            int(x - bw / 2),
                            int(y - bh / 2),
                            int(x + bw / 2),
                            int(y + bh / 2),
                        )
                        color = self.color_map[cls]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        if cls < len(self.class_names):
                            cv2.putText(
                                img,
                                self.class_names[cls],
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

        return img

    def create_window(self, window_id: int):
        """Create a new window with a random image"""
        img_file = random.choice(self.image_files)
        img = self.load_and_process_image(img_file)

        if img is not None:
            window_name = f"Sample_{window_id}"
            self.windows[window_id] = {
                'name': window_name,
                'image_file': img_file
            }

            # Position windows in a grid
            rows = int(self.num_windows ** 0.5)
            cols = (self.num_windows + rows - 1) // rows
            row = window_id // cols
            col = window_id % cols
            x_pos = col * 320
            y_pos = row * 240

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 300, 200)
            cv2.moveWindow(window_name, x_pos, y_pos)
            cv2.imshow(window_name, img)

    def replace_window(self, window_id: int):
        """Replace the content of an existing window with a new random image"""
        if window_id in self.windows:
            img_file = random.choice(self.image_files)
            img = self.load_and_process_image(img_file)

            if img is not None:
                window_name = self.windows[window_id]['name']
                self.windows[window_id]['image_file'] = img_file
                cv2.imshow(window_name, img)

    def run(self):
        """Main loop to handle the visualization"""
        # Create initial windows
        for i in range(self.num_windows):
            self.create_window(i)

        print("Controls:")
        print("- Press keys 0-9 to refresh the corresponding image")
        print("- Press 'q' to quit")
        print("- Press any other key to refresh a random image")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('9'):
                # Refresh specific window (0-9)
                window_id = key - ord('0')
                if window_id < self.num_windows:
                    self.replace_window(window_id)
                    print(f"Refreshed window {window_id}")
            else:
                # Refresh random window
                window_id = random.randint(0, self.num_windows - 1)
                self.replace_window(window_id)
                print(f"Refreshed random window {window_id}")

        cv2.destroyAllWindows()


def visualize_multiple_samples(image_dir: str, label_dir: str, num_windows: int = 10) -> None:
    """Function to maintain compatibility with original interface"""
    visualizer = MultiImageVisualizer(image_dir, label_dir, num_windows)
    visualizer.run()


if __name__ == "__main__":
    images_path_train = os.path.join(get_images_path(), "train")
    labels_path_train = os.path.join(get_labels_path(), "train")

    # Use the new multi-window visualizer
    visualize_multiple_samples(images_path_train, labels_path_train, num_windows=5)
