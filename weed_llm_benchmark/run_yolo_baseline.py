#!/usr/bin/env python3
"""
Run YOLO11n baseline on benchmark datasets.

Two modes:
1. Zero-shot: Pretrained YOLO11n (no weed training) — fair comparison with zero-shot LLMs
2. Fine-tuned: Trained on each dataset's train split — shows what's achievable with supervision

Outputs results in the same JSON format as the LLM pipeline for consistent evaluation.

Usage:
    # Zero-shot on weed2okok test set
    python run_yolo_baseline.py --dataset weed2okok --mode zero-shot

    # Fine-tune on CottonWeedDet12 then evaluate
    python run_yolo_baseline.py --dataset cottonweeddet12 --mode fine-tune --epochs 50

    # Evaluate existing YOLO weights
    python run_yolo_baseline.py --dataset weed2okok --weights best.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # parent of weed_llm_benchmark/
RESULT_DIR = os.path.join(BASE_DIR, "results")
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")


def get_dataset_paths(dataset_key):
    """Get standard paths for a dataset."""
    ds_dir = os.path.join(DOWNLOAD_DIR, dataset_key)
    data_yaml = os.path.join(ds_dir, "data.yaml")
    test_images = os.path.join(ds_dir, "test", "images")
    test_labels = os.path.join(ds_dir, "test", "labels")
    # Fall back to valid if test doesn't exist
    if not os.path.isdir(test_images):
        test_images = os.path.join(ds_dir, "valid", "images")
        test_labels = os.path.join(ds_dir, "valid", "labels")
    return {
        "root": ds_dir,
        "data_yaml": data_yaml,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def run_zero_shot(dataset_key, model_name="yolo11n.pt", conf_threshold=0.25):
    """Run pretrained YOLO in zero-shot mode (no fine-tuning)."""
    from ultralytics import YOLO

    paths = get_dataset_paths(dataset_key)
    test_images = paths["test_images"]

    if not os.path.isdir(test_images):
        print(f"[!] Test images not found: {test_images}")
        return None

    image_files = sorted([
        os.path.join(test_images, f) for f in os.listdir(test_images)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"[*] Zero-shot YOLO on {dataset_key}: {len(image_files)} images")

    model = YOLO(model_name)
    all_results = []

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        start = time.time()
        results = model(img_path, conf=conf_threshold, verbose=False)
        elapsed = time.time() - start

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    conf = float(boxes.conf[j].cpu())
                    cls = int(boxes.cls[j].cpu())
                    cls_name = r.names.get(cls, str(cls))

                    # Normalize to image dimensions for percentage
                    img_h, img_w = r.orig_shape
                    detections.append({
                        "label": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [
                            round(float(x1 / img_w * 100), 2),
                            round(float(y1 / img_h * 100), 2),
                            round(float(x2 / img_w * 100), 2),
                            round(float(y2 / img_h * 100), 2),
                        ],
                        "source": "yolo_zero_shot",
                    })

        all_results.append({
            "image": img_name,
            "num_detections": len(detections),
            "detections": detections,
            "time_s": round(elapsed, 3),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_files)}] processed")

    # Save results
    output_path = os.path.join(RESULT_DIR, f"yolo_zero_shot_{dataset_key}.json")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_dets = sum(r["num_detections"] for r in all_results)
    avg_time = np.mean([r["time_s"] for r in all_results])
    print(f"[+] Done: {len(image_files)} images, {total_dets} detections, "
          f"{avg_time:.3f}s avg inference")
    print(f"    Saved: {output_path}")

    return output_path


def run_fine_tune(dataset_key, epochs=50, model_name="yolo11n.pt", imgsz=640):
    """Fine-tune YOLO on dataset's training split, then evaluate on test."""
    from ultralytics import YOLO

    paths = get_dataset_paths(dataset_key)
    data_yaml = paths["data_yaml"]

    if not os.path.exists(data_yaml):
        print(f"[!] data.yaml not found: {data_yaml}")
        return None

    print(f"[*] Fine-tuning YOLO on {dataset_key}")
    print(f"    Data:   {data_yaml}")
    print(f"    Epochs: {epochs}")
    print(f"    Model:  {model_name}")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name=f"yolo_{dataset_key}",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
    )

    # Evaluate on test set
    best_weights = os.path.join(BASE_DIR, "runs", f"yolo_{dataset_key}", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"[*] Evaluating fine-tuned model: {best_weights}")
        return run_with_weights(dataset_key, best_weights)
    else:
        print(f"[!] Best weights not found at {best_weights}")
        return None


def run_with_weights(dataset_key, weights_path, conf_threshold=0.25):
    """Run YOLO with specific weights on a dataset's test split."""
    from ultralytics import YOLO

    paths = get_dataset_paths(dataset_key)
    test_images = paths["test_images"]

    if not os.path.isdir(test_images):
        print(f"[!] Test images not found: {test_images}")
        return None

    image_files = sorted([
        os.path.join(test_images, f) for f in os.listdir(test_images)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"[*] YOLO ({os.path.basename(weights_path)}) on {dataset_key}: {len(image_files)} images")

    model = YOLO(weights_path)
    all_results = []

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        start = time.time()
        results = model(img_path, conf=conf_threshold, verbose=False)
        elapsed = time.time() - start

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    conf = float(boxes.conf[j].cpu())
                    cls = int(boxes.cls[j].cpu())
                    cls_name = r.names.get(cls, str(cls))
                    img_h, img_w = r.orig_shape

                    detections.append({
                        "label": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [
                            round(float(x1 / img_w * 100), 2),
                            round(float(y1 / img_h * 100), 2),
                            round(float(x2 / img_w * 100), 2),
                            round(float(y2 / img_h * 100), 2),
                        ],
                        "source": "yolo_fine_tuned",
                    })

        all_results.append({
            "image": img_name,
            "num_detections": len(detections),
            "detections": detections,
            "time_s": round(elapsed, 3),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_files)}] processed")

    weights_name = Path(weights_path).stem
    output_path = os.path.join(RESULT_DIR, f"yolo_{weights_name}_{dataset_key}.json")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_dets = sum(r["num_detections"] for r in all_results)
    avg_time = np.mean([r["time_s"] for r in all_results])
    print(f"[+] Done: {len(image_files)} images, {total_dets} detections, "
          f"{avg_time:.3f}s avg inference")
    print(f"    Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO baseline for weed detection benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset key (e.g., weed2okok, cottonweeddet12, deepweeds)")
    parser.add_argument("--mode", type=str, default="zero-shot",
                        choices=["zero-shot", "fine-tune"],
                        help="Evaluation mode")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to YOLO weights (overrides mode)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO model variant (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for fine-tune mode")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after detection")
    args = parser.parse_args()

    if args.weights:
        output_path = run_with_weights(args.dataset, args.weights, args.conf)
    elif args.mode == "zero-shot":
        output_path = run_zero_shot(args.dataset, args.model, args.conf)
    else:
        output_path = run_fine_tune(args.dataset, args.epochs, args.model)

    # Optional: run evaluation
    if args.evaluate and output_path:
        from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels, print_evaluation
        paths = get_dataset_paths(args.dataset)
        gt = load_yolo_labels(paths["test_labels"])
        predictions = load_predictions_from_json(output_path)
        summary = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])
        print_evaluation(summary, model_name=f"YOLO ({args.mode})")


if __name__ == "__main__":
    main()
