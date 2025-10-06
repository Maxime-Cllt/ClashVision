import os.path

from ultralytics import YOLO

from clashvision.core.path import get_models_path

path : str = os.path.join(get_models_path(), 'v1', 'best.pt')

# Load your trained YOLOv8 model
model = YOLO(path)

# Export to TorchScript format
model.export(format="torchscript")
