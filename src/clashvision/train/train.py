import os
import os.path

from clashvision.core.path import get_project_root


def train_yolo_model():
    from ultralytics import YOLO

    # Load a YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Get the number of CPU cores for data loading
    cpu_count = os.cpu_count()
    workers = cpu_count if cpu_count is not None else 4
    print(
        f"Number of CPU cores: {cpu_count}, using {workers} workers for data loading."
    )

    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    print(f"Using device: {device}")
    model.to(device)  # Move model to the appropriate device

    # Train the model
    model.train(
        data=os.path.join(get_project_root(), "config", "dataset.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=workers,
        optimizer="SGD",
        lr0=0.01,
        device=device,
        project=os.path.join(get_project_root(), "runs", "train"),  # Save to runs/train
        name="yolov8n-clashvision",
    )

    export_formats: list[str] = ["onnx", "torchscript"]

    for fmt in export_formats:
        try:
            model.export(format=fmt)
            print(f"Model exported to {fmt} format successfully.")
        except Exception as e:
            print(f"Failed to export model to {fmt} format: {e}")


if __name__ == "__main__":
    # run with uv run python src/clashvision/train/train.py
    train_yolo_model()
