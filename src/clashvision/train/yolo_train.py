import os.path

import torch

from clashvision.core.path import get_project_root

if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a YOLOv8 model (pretrained weights recommended as a starting point)
    # Options: 'yolov8n.pt' (nano, fastest), 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' (largest, most accurate)
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data=os.path.join(get_project_root(), 'config', 'dataset.yaml'),
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        batch=16,  # batch size
        workers=8,  # number of dataloader workers
        optimizer="SGD",  # or "Adam", "AdamW"
        lr0=0.01,  # initial learning rate
        device=torch.cuda.is_available() and 0 or 'cpu',  # use GPU if available
    )

    try:
        # Save the trained model
        model.export(format="onnx")  # export to ONNX format
    except Exception as e:
        print(f"Model export failed: {e}")
