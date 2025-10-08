import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from clashvision.core.path import get_project_root, get_models_path, get_images_path
from clashvision.enums.clash_class import ClashClass


def load_model(model_path: str) -> YOLO:
    """Load the trained YOLO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return YOLO(model_path)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to BGR tuple for OpenCV"""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    # Convert RGB to BGR for OpenCV
    return (rgb[2], rgb[1], rgb[0])


def draw_custom_annotations(
    image: np.ndarray, results, conf_threshold: float = 0.5
) -> np.ndarray:
    """Draw custom annotations using ClashClass colors"""
    annotated_image = image.copy()

    for box in results[0].boxes:
        if box.conf is not None and box.conf[0] >= conf_threshold:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            try:
                # Get the ClashClass enum and its color
                clash_class = ClashClass.from_int(class_id)
                color_hex = clash_class.to_hex
                color_bgr = hex_to_bgr(color_hex)
                class_name = str(clash_class)
            except ValueError:
                # Fallback to default color if class not found
                color_bgr = (0, 0, 0)  # Black
                class_name = f"Unknown_{class_id}"

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color_bgr, 2)

            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - baseline - 10),
                (x1 + text_width, y1),
                color_bgr,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2,
            )

    return annotated_image


def run_inference_on_image(model: YOLO, image_path: str, conf_threshold: float = 0.5):
    """Run inference on a single image"""
    return model(image_path, conf=conf_threshold)


def run_inference_on_directory(
    model: YOLO, image_dir: str, output_dir: str, conf_threshold: float = 0.5
) -> list[dict]:
    """Run inference on all images in a directory"""
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            print(f"Processing: {image_path.name}")

            # Load original image
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            # Run inference
            results = model(str(image_path), conf=conf_threshold)

            # Draw custom annotations with ClashClass colors
            annotated_image = draw_custom_annotations(
                original_image, results, conf_threshold
            )

            # Save annotated image
            output_path = output_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)

            # Collect results summary
            detections = []
            for box in results[0].boxes:
                if box.conf is not None and box.conf[0] >= conf_threshold:
                    class_id = int(box.cls[0])
                    try:
                        clash_class = ClashClass.from_int(class_id)
                        class_name = str(clash_class)
                    except ValueError:
                        class_name = f"Unknown_{class_id}"

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )

            results_summary.append(
                {
                    "image": image_path.name,
                    "detections": detections,
                    "num_detections": len(detections),
                }
            )

    return results_summary


def validate_model(model, val_data_path):
    """Validate the model on validation dataset"""
    results = model.val(data=val_data_path)
    return results


if __name__ == "__main__":
    model_path = os.path.join(get_models_path(), "v1", "best.pt")

    try:
        # Load the trained model
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)

        # Run validation on your dataset
        print("\nRunning validation...")
        dataset_config = os.path.join(get_project_root(), "config", "dataset.yaml")
        val_results = validate_model(model, dataset_config)
        print(f"Validation mAP50: {val_results.box.map50}")
        print(f"Validation mAP50-95: {val_results.box.map}")

        # Test on validation images
        print("\nRunning inference on validation images...")
        val_images_dir = os.path.join(get_images_path(), "val")
        output_dir = os.path.join(get_project_root(), "inference_results")

        if os.path.exists(val_images_dir):
            results_summary = run_inference_on_directory(
                model, val_images_dir, output_dir, conf_threshold=0.5
            )

            # Print summary
            print(f"\nInference completed! Results saved to: {output_dir}")
            print(f"Processed {len(results_summary)} images")

            total_detections = sum(r["num_detections"] for r in results_summary)
            print(f"Total detections: {total_detections}")

            # Print per-image summary with class names
            for result in results_summary:
                print(f"\n{result['image']}: {result['num_detections']} detections")
                for detection in result["detections"]:
                    print(
                        f"  - {detection['class_name']}: {detection['confidence']:.2f}"
                    )
        else:
            print(f"Validation images directory not found: {val_images_dir}")
            print("Please add some images to test inference")

    except Exception as e:
        print(f"An error occurred: {e}")
