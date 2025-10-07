import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO

from clashvision.core.path import get_images_path, get_models_path, get_project_root
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            return model, "pytorch"
        except Exception as e2:
            print(f"Failed to load with torch.load: {e2}")

            # Option 3: Add safe globals and try again
            try:
                torch.serialization.add_safe_globals(
                    ["ultralytics.nn.tasks.DetectionModel"]
                )
                model = torch.load(model_path, map_location=device, weights_only=True)
                model.eval()
                return model, "pytorch_safe"
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load model with any method. Last error: {e3}"
                )


def run_inference_ultralytics(model, image_path, class_names=None):
    """
    Run inference using ultralytics YOLO model - returns ALL detections
    """
    results = model(image_path)

    # Get the first result (single image)
    result = results[0]

    if len(result.boxes) <= 0:
        return {
            "predicted_class_idx": -1,
            "confidence": 0.0,
            "predicted_class_name": "No detection",
            "all_detections": 0,
            "boxes": [],
            "detections": [],  # New: empty list for no detections
        }

    # Get ALL detections with their information
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    # Create list of all detections
    detections = []
    for i in range(len(result.boxes)):
        class_idx = int(classes[i])
        confidence = float(confidences[i])
        box = boxes[i]

        detection = {
            "class_idx": class_idx,
            "confidence": confidence,
            "class_name": (
                class_names[class_idx] if class_names else f"Class_{class_idx}"
            ),
            "box": box,
            "clash_class": (
                ClashClass.from_int(class_idx) if class_idx < len(ClashClass) else None
            ),
        }
        detections.append(detection)

    # Sort detections by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # Get the detection with highest confidence for backward compatibility
    best_detection = detections[0]

    return {
        "predicted_class_idx": best_detection["class_idx"],
        "confidence": best_detection["confidence"],
        "predicted_class_name": best_detection["class_name"],
        "all_detections": len(result.boxes),
        "boxes": boxes,
        "detections": detections,  # New: all detections with full info
    }


def run_inference_pytorch(model, image_path, class_names=None):
    """
    Run inference using PyTorch model (classification style)
    Note: This is for classification, not detection, so only one result
    """
    device = next(model.parameters()).device

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence_score = confidence.item()

    # For classification, create a single detection
    detection = {
        "class_idx": predicted_class,
        "confidence": confidence_score,
        "class_name": (
            class_names[predicted_class] if class_names else f"Class_{predicted_class}"
        ),
        "box": None,  # No bounding box for classification
        "clash_class": (
            ClashClass.from_int(predicted_class)
            if predicted_class < len(ClashClass)
            else None
        ),
    }

    return {
        "predicted_class_idx": predicted_class,
        "confidence": confidence_score,
        "predicted_class_name": (
            class_names[predicted_class] if class_names else f"Class_{predicted_class}"
        ),
        "detections": [detection],  # Single detection for classification
    }


def run_inference_on_image(model_path, image_path, class_names=None):
    """
    Main function to run inference on a single image

    Args:
        model_path: Path to your trained model
        image_path: Path to the image you want to classify
        class_names: List of class names (optional)

    Returns:
        Prediction results with ALL detections
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


def run_batch_inference(
        model_path: str, image_directory: str, class_names: list[str] = None
):
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
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in os.listdir(image_directory)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

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

            result["image_file"] = image_file
            result["image_path"] = image_path
            results.append(result)

            print(
                f"  -> {result['predicted_class_name']} (confidence: {result['confidence']:.4f})"
            )
            if "detections" in result:
                print(f"  -> Total detections: {len(result['detections'])}")
                for detection in result["detections"][:3]:  # Show first 3 detections
                    print(
                        f"     - {detection['class_name']}: {detection['confidence']:.4f}"
                    )

        except Exception as e:
            print(f"  -> Error processing {image_file}: {e}")
            results.append(
                {"image_file": image_file, "image_path": image_path, "error": str(e)}
            )

    return results


def draw_detections_on_image(image_path, detections, save_path=None, show_image=True):
    """
    Draw all detections on image with class-specific colors

    Args:
        image_path: Path to the original image
        detections: List of detection dictionaries
        save_path: Optional path to save the annotated image
        show_image: Whether to display the image
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for detection in detections:
        if detection.get("box") is None:  # Skip if no bounding box (classification)
            continue

        x1, y1, x2, y2 = detection["box"]
        clash_class = detection.get("clash_class")

        # Get color for the class
        if clash_class and hasattr(clash_class, "to_color"):
            color = clash_class.to_hex
        else:
            color = "red"  # Default color

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label with confidence
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"

        # Draw label background
        bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white")

    if save_path:
        img.save(save_path)
        print(f"Annotated image saved to: {save_path}")

    if show_image:
        img.show()

    return img


if __name__ == "__main__":

    # Get class names
    start_time = time.time()
    class_names = ClashClass.to_list()
    image = os.path.join(get_images_path(), "val", "village_1759583271.png")
    model = os.path.join(get_models_path(), "v1", "best.pt")
    img_save_path = os.path.join(
        get_project_root(), "inference_results", f"annotated_{os.path.basename(image)}"
    )

    # Single image inference
    result = run_inference_on_image(
        model_path=model, image_path=image, class_names=class_names
    )

    # Print results for ALL detections
    print(
        f"Best prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.4f})"
    )
    print(f"Total detections: {result.get('all_detections', 0)}")

    if "detections" not in result and result["detections"]:
        print("No detections found.")

    print("\nAll detections:")
    for i, detection in enumerate(result["detections"]):
        print(
            f"  {i + 1}. {detection['class_name']} - Confidence: {detection['confidence']:.4f}"
        )

    # Draw ALL detections on image with class-specific colors
    annotated_image = draw_detections_on_image(
        image_path=image,
        detections=result["detections"],
        save_path=img_save_path,
        show_image=True,
    )

    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")
