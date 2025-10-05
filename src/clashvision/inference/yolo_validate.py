import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from clashvision.core.path import get_project_root, get_models_path, get_images_path


def load_model(model_path: str) -> YOLO:
    """Load the trained YOLO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return YOLO(model_path)


def run_inference_on_image(model: YOLO, image_path: str, conf_threshold: float = 0.5):
    """Run inference on a single image"""
    return model(image_path, conf=conf_threshold)


def run_inference_on_directory(model: YOLO, image_dir: str, output_dir: str, conf_threshold: float = 0.5):
    """Run inference on all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            print(f"Processing: {image_path.name}")

            # Run inference
            results = model(str(image_path), conf=conf_threshold)

            # Save annotated image
            annotated_image = results[0].plot()
            output_path = output_dir / f"annotated_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)

            # Collect results summary
            detections = []
            for box in results[0].boxes:
                if box.conf is not None:
                    detections.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })

            results_summary.append({
                'image': image_path.name,
                'detections': detections,
                'num_detections': len(detections)
            })

    return results_summary


def validate_model(model, val_data_path):
    """Validate the model on validation dataset"""
    results = model.val(data=val_data_path)
    return results


if __name__ == '__main__':
    model_path = os.path.join(get_models_path(), 'v1', 'best.pt')

    try:
        # Load the trained model
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)

        # Run validation on your dataset
        print("\nRunning validation...")
        dataset_config = os.path.join(get_project_root(), 'config', 'dataset.yaml')
        val_results = validate_model(model, dataset_config)
        print(f"Validation mAP50: {val_results.box.map50}")
        print(f"Validation mAP50-95: {val_results.box.map}")

        # Test on validation images
        print("\nRunning inference on validation images...")
        val_images_dir = os.path.join(get_images_path(), 'val')
        output_dir = os.path.join(get_project_root(), 'inference_results')

        if os.path.exists(val_images_dir):
            results_summary = run_inference_on_directory(
                model,
                val_images_dir,
                output_dir,
                conf_threshold=0.5
            )

            # Print summary
            print(f"\nInference completed! Results saved to: {output_dir}")
            print(f"Processed {len(results_summary)} images")

            total_detections = sum(r['num_detections'] for r in results_summary)
            print(f"Total detections: {total_detections}")

            # Print per-image summary
            for result in results_summary:
                print(f"{result['image']}: {result['num_detections']} detections")
        else:
            print(f"Validation images directory not found: {val_images_dir}")
            print("Please add some images to test inference")

    except Exception as e:
        print(f"An error occurred: {e}")
