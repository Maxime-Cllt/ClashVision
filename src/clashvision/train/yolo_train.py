import os
import os.path

from clashvision.core.path import get_project_root

# run with uv run python src/clashvision/train/yolo_train.py


def train_yolo_model():
    from ultralytics import YOLO

    # Load a YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data=os.path.join(get_project_root(), "config", "dataset.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        optimizer="SGD",
        lr0=0.01,
        device=0,
    )

    try:
        # Save the trained model
        model.export(format="onnx")
    except Exception as e:
        print(f"Model export failed: {e}")


if __name__ == "__main__":
    train_yolo_model()
