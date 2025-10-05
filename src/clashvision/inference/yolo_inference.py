import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

from clashvision.core.path import get_images_path, get_models_path
from clashvision.enums.clash_class import ClashClass


def load_yolo_model(model_path):
    """
    Load YOLO model properly handling the ultralytics format
    """
    try:
        # Option 1: Use ultralytics YOLO loader (recommended)
        model = YOLO(model_path)
        return model, "ultralytics"
    except Exception as e:
        print(f"Failed to load with ultralytics: {e}")

        try:
            # Option 2: Load with torch.load and weights_only=False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            return model, "pytorch"
        except Exception as e2:
            print(f"Failed to load with torch.load: {e2}")

            # Option 3: Add safe globals and try again
            try:
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                model = torch.load(model_path, map_location=device, weights_only=True)
                model.eval()
                return model, "pytorch_safe"
            except Exception as e3:
                raise RuntimeError(f"Could not load model with any method. Last error: {e3}")


def run_inference_ultralytics(model, image_path, class_names=None):
    """
    Run inference using ultralytics YOLO model
    """
    results = model(image_path)

    # Get the first result (single image)
    result = results[0]

    if len(result.boxes) <= 0:
        return {
            'predicted_class_idx': -1,
            'confidence': 0.0,
            'predicted_class_name': "No detection",
            'all_detections': 0,
            'boxes': []
        }

    # Get the detection with highest confidence
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    max_conf_idx = confidences.argmax()
    predicted_class = int(classes[max_conf_idx])
    confidence_score = float(confidences[max_conf_idx])

    return {
        'predicted_class_idx': predicted_class,
        'confidence': confidence_score,
        'predicted_class_name': class_names[predicted_class] if class_names else f"Class_{predicted_class}",
        'all_detections': len(result.boxes),
        'boxes': result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []
    }


def run_inference_pytorch(model, image_path, class_names=None):
    """
    Run inference using PyTorch model (classification style)
    """
    device = next(model.parameters()).device

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence_score = confidence.item()

    return {
        'predicted_class_idx': predicted_class,
        'confidence': confidence_score,
        'predicted_class_name': class_names[predicted_class] if class_names else f"Class_{predicted_class}"
    }


def run_inference_on_image(model_path, image_path, class_names=None):
    """
    Main function to run inference on a single image

    Args:
        model_path: Path to your trained model
        image_path: Path to the image you want to classify
        class_names: List of class names (optional)

    Returns:
        Prediction results
    """
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading model from: {model_path}")
    model, model_type = load_yolo_model(model_path)

    print(f"Running inference on: {image_path}")
    print(f"Model type: {model_type}")

    if model_type == "ultralytics":
        result = run_inference_ultralytics(model, image_path, class_names)
    else:
        result = run_inference_pytorch(model, image_path, class_names)

    return result


def run_batch_inference(model_path: str, image_directory: str, class_names: list[str] = None):
    """
    Run inference on multiple images in a directory

    Args:
        model_path: Path to your trained model
        image_directory: Directory containing images
        class_names: List of class names (optional)

    Returns:
        List of results for each image
    """
    # Load model once
    print(f"Loading model from: {model_path}")
    model, model_type = load_yolo_model(model_path)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(image_directory)
                   if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        print(f"No image files found in {image_directory}")
        return []

    results = []
    print(f"Processing {len(image_files)} images...")

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_directory, image_file)
        print(f"[{i}/{len(image_files)}] Processing: {image_file}")

        try:
            if model_type == "ultralytics":
                result = run_inference_ultralytics(model, image_path, class_names)
            else:
                result = run_inference_pytorch(model, image_path, class_names)

            result['image_file'] = image_file
            result['image_path'] = image_path
            results.append(result)

            print(f"  -> {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")

        except Exception as e:
            print(f"  -> Error processing {image_file}: {e}")
            results.append({
                'image_file': image_file,
                'image_path': image_path,
                'error': str(e)
            })

    return results


if __name__ == "__main__":

    # Get class names
    class_names = ClashClass.to_list()
    image = os.path.join(get_images_path(), 'val', 'village_1759335821.png')
    model = os.path.join(get_models_path(), 'v1', 'best.pt')

    # Single image inference
    result = run_inference_on_image(
        model_path=model,
        image_path=image,
        class_names=class_names
    )

    # Print all detections
    print(f"Predicted class: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})")
    print(f"All detections: {result['all_detections']}")
    print(f"Boxes: {result['boxes']}")

    # Draw boxes on image
    if result['all_detections'] > 0:
        from PIL import ImageDraw

        img = Image.open(image).convert('RGB')
        draw = ImageDraw.Draw(img)

        for box in result['boxes']:
            x1, y1, x2, y2 = box
            color: str = None
            class_type = ClashClass.from_int(result['predicted_class_idx'])

            # match case
            match class_type:
                case ClashClass.ELIXIR_STORAGE:
                    color = class_type.to_color
                case ClashClass.GOLD_STORAGE:
                    color = class_type.to_color
                case _:
                    color = "black"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        img.show()
