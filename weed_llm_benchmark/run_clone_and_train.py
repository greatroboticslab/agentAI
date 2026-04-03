#!/usr/bin/env python3
"""
Clone and Train: Download external weed detection models, retrain on our data.

This implements Professor Zhang's second task:
  "Go to GitHub, find weed detection projects, clone and train."

Strategy:
  1. Download pre-trained weed detection models (HuggingFace + GitHub)
  2. Fine-tune on CottonWeedDet12 (our dataset, 12 species)
  3. Evaluate against our YOLO11n baseline
  4. Use as additional label source in the framework

Models tested:
  - DETR-ResNet50 (machinelearningzuu/detr-resnet-50_finetuned-weed-detection)
  - YOLOv8s weed (MuayThaiLegz/WeedDetection-YOLOv8s)
  - DeepWeeds ResNet50 (AlexOlsen/DeepWeeds) — classification baseline
"""

import os
import sys
import json
import time
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "clone_and_train")
MODELS_DIR = os.path.join(BASE_DIR, "models", "external")
HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

# CottonWeedDet12 paths (same as leave4out)
DATASET_DIR = os.path.join(BASE_DIR, "results", "leave4out", "dataset_8species")
FULL_DATASET = os.path.join(BASE_DIR, "datasets", "CottonWeedDet12")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================
# 1. CLONE GITHUB REPOS
# ==============================================================

GITHUB_REPOS = [
    {
        "name": "DeepWeeds",
        "url": "https://github.com/AlexOlsen/DeepWeeds.git",
        "type": "classification",
        "description": "ResNet50/InceptionV3, 17.5K images, 8 Australian weed species",
    },
]


def clone_repos():
    """Clone GitHub weed detection repositories."""
    clone_dir = os.path.join(MODELS_DIR, "github_repos")
    os.makedirs(clone_dir, exist_ok=True)

    for repo in GITHUB_REPOS:
        repo_name = repo["name"]
        repo_path = os.path.join(clone_dir, repo_name)

        if os.path.exists(repo_path):
            logger.info(f"[Clone] {repo_name} already exists, skipping")
            continue

        logger.info(f"[Clone] Cloning {repo_name} from {repo['url']}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo["url"], repo_path],
                check=True, capture_output=True, text=True, timeout=300
            )
            logger.info(f"[Clone] {repo_name} cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"[Clone] Failed to clone {repo_name}: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"[Clone] Timeout cloning {repo_name}")

    return clone_dir


# ==============================================================
# 2. DOWNLOAD HUGGINGFACE MODELS
# ==============================================================

def download_detr_weed():
    """Download DETR weed detection model from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    import torch

    model_id = "machinelearningzuu/detr-resnet-50_finetuned-weed-detection"
    logger.info(f"[Download] Loading {model_id}...")

    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=os.path.join(HF_CACHE, "hub"))
    model = AutoModelForObjectDetection.from_pretrained(model_id, cache_dir=os.path.join(HF_CACHE, "hub"))
    model = model.cuda().eval()

    logger.info("[Download] DETR weed model loaded")
    return model, processor


def download_yolov8_weed():
    """Try to download YOLOv8 weed detection model."""
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="MuayThaiLegz/WeedDetection-YOLOv8s",
            filename="best.pt",
            cache_dir=os.path.join(HF_CACHE, "hub")
        )
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"[Download] YOLOv8s weed model loaded from {model_path}")
        return model
    except Exception as e:
        logger.warning(f"[Download] YOLOv8s weed model not available: {e}")
        return None


# ==============================================================
# 3. TRAIN ON OUR DATA
# ==============================================================

def train_yolov8_on_cottonweed():
    """Train YOLOv8s from scratch on CottonWeedDet12.

    This is the "train" part of professor's "clone and train" task.
    Uses Ultralytics YOLOv8s (pre-trained on COCO) and fine-tunes on our weed data.
    """
    from ultralytics import YOLO
    import torch

    # Check if dataset exists
    data_yaml = None
    for candidate in [
        os.path.join(FULL_DATASET, "data.yaml"),
        os.path.join(DATASET_DIR, "data.yaml"),
    ]:
        if os.path.exists(candidate):
            data_yaml = candidate
            break

    if not data_yaml:
        # Create data.yaml from our dataset structure
        data_yaml = _create_data_yaml()

    if not data_yaml:
        logger.error("No dataset found for training")
        return None

    logger.info(f"[Train] Training YOLOv8s on CottonWeedDet12...")
    logger.info(f"[Train] Dataset: {data_yaml}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLOv8s pre-trained on COCO
    model = YOLO("yolov8s.pt")

    project_dir = os.path.join(RESULTS_DIR, "yolov8s_cottonweed")
    model.train(
        data=data_yaml,
        epochs=100,
        batch=-1,
        device=device,
        project=project_dir,
        name="train",
        patience=20,
        lr0=0.01,
        verbose=True,
    )

    best_pt = os.path.join(project_dir, "train", "weights", "best.pt")
    if os.path.exists(best_pt):
        logger.info(f"[Train] Training complete: {best_pt}")
        return best_pt
    else:
        logger.error("[Train] Training failed — no best.pt found")
        return None


def _create_data_yaml():
    """Create data.yaml for CottonWeedDet12 if it doesn't exist."""
    # Try to find the dataset
    for base in [FULL_DATASET, DATASET_DIR]:
        train_imgs = os.path.join(base, "train", "images")
        if os.path.isdir(train_imgs):
            classes = [
                "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
                "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
            ]
            yaml_path = os.path.join(base, "data.yaml")
            with open(yaml_path, "w") as f:
                f.write(f"train: {os.path.join(base, 'train', 'images')}\n")
                f.write(f"val: {os.path.join(base, 'valid', 'images')}\n")
                if os.path.isdir(os.path.join(base, "test", "images")):
                    f.write(f"test: {os.path.join(base, 'test', 'images')}\n")
                f.write(f"nc: {len(classes)}\n")
                f.write(f"names: {classes}\n")
            logger.info(f"[Dataset] Created {yaml_path}")
            return yaml_path
    return None


# ==============================================================
# 4. EVALUATE
# ==============================================================

def evaluate_model(model_path, test_images, test_labels, model_name="unknown"):
    """Evaluate a YOLO model and return metrics."""
    from ultralytics import YOLO
    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    # Use YOLO's built-in validation
    logger.info(f"[Eval] Evaluating {model_name}...")

    # Run prediction on test images
    tp, fp, fn = 0, 0, 0
    for img_file in sorted(os.listdir(test_images)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = Path(img_file).stem
        lbl_path = os.path.join(test_labels, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        # Ground truth
        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append([float(x) for x in parts[1:5]])

        # Predictions
        results = model.predict(os.path.join(test_images, img_file),
                                conf=0.25, device=device, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                pred_boxes.append(box.xywhn[0].tolist())

        # Match (IoU >= 0.5)
        matched = set()
        for pb in sorted(pred_boxes, key=lambda x: -x[0] if len(x) > 4 else 0):
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched:
                    continue
                # IoU
                x1_1, y1_1 = pb[0]-pb[2]/2, pb[1]-pb[3]/2
                x2_1, y2_1 = pb[0]+pb[2]/2, pb[1]+pb[3]/2
                x1_2, y1_2 = gb[0]-gb[2]/2, gb[1]-gb[3]/2
                x2_2, y2_2 = gb[0]+gb[2]/2, gb[1]+gb[3]/2
                inter = max(0, min(x2_1,x2_2)-max(x1_1,x1_2)) * max(0, min(y2_1,y2_2)-max(y1_1,y1_2))
                union = pb[2]*pb[3] + gb[2]*gb[3] - inter
                iou = inter/union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi)
                tp += 1
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    del model
    torch.cuda.empty_cache()

    result = {"model": model_name, "precision": round(prec, 4),
              "recall": round(rec, 4), "f1": round(f1, 4),
              "tp": tp, "fp": fp, "fn": fn}
    logger.info(f"[Eval] {model_name}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    return result


def evaluate_detr(model, processor, test_images, test_labels):
    """Evaluate DETR model on test set."""
    import torch
    from PIL import Image

    tp, fp, fn = 0, 0, 0
    for img_file in sorted(os.listdir(test_images)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = Path(img_file).stem
        lbl_path = os.path.join(test_labels, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append([float(x) for x in parts[1:5]])

        # DETR inference
        image = Image.open(os.path.join(test_images, img_file)).convert("RGB")
        w, h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(model.device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3)

        pred_boxes = []
        if results:
            for box in results[0]["boxes"]:
                x1, y1, x2, y2 = box.tolist()
                cx, cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
                bw, bh = (x2-x1)/w, (y2-y1)/h
                pred_boxes.append([cx, cy, bw, bh])

        # Match
        matched = set()
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched:
                    continue
                x1_1, y1_1 = pb[0]-pb[2]/2, pb[1]-pb[3]/2
                x2_1, y2_1 = pb[0]+pb[2]/2, pb[1]+pb[3]/2
                x1_2, y1_2 = gb[0]-gb[2]/2, gb[1]-gb[3]/2
                x2_2, y2_2 = gb[0]+gb[2]/2, gb[1]+gb[3]/2
                inter = max(0, min(x2_1,x2_2)-max(x1_1,x1_2)) * max(0, min(y2_1,y2_2)-max(y1_1,y1_2))
                union = pb[2]*pb[3] + gb[2]*gb[3] - inter
                iou = inter/union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi)
                tp += 1
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    result = {"model": "DETR-ResNet50-weed", "precision": round(prec, 4),
              "recall": round(rec, 4), "f1": round(f1, 4)}
    logger.info(f"[Eval] DETR: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    return result


# ==============================================================
# MAIN
# ==============================================================

def main():
    import torch
    logger.info("=" * 70)
    logger.info("CLONE AND TRAIN — External Weed Detection Models")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 70)

    all_results = []

    # Step 1: Clone repos
    logger.info("\n--- Step 1: Clone GitHub repos ---")
    clone_repos()

    # Step 2: Evaluate DETR (zero-shot on our test set)
    logger.info("\n--- Step 2: Evaluate DETR weed model (zero-shot) ---")
    test_imgs = os.path.join(DATASET_DIR, "test", "images")
    test_lbls = os.path.join(DATASET_DIR, "test", "labels")

    if os.path.isdir(test_imgs):
        try:
            detr_model, detr_proc = download_detr_weed()
            detr_result = evaluate_detr(detr_model, detr_proc, test_imgs, test_lbls)
            all_results.append(detr_result)
            del detr_model, detr_proc
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"DETR evaluation failed: {e}")
            import traceback; traceback.print_exc()
    else:
        logger.warning(f"Test images not found at {test_imgs}")

    # Step 3: Train YOLOv8s on CottonWeedDet12
    logger.info("\n--- Step 3: Train YOLOv8s on CottonWeedDet12 ---")
    yolov8_path = train_yolov8_on_cottonweed()
    if yolov8_path and os.path.isdir(test_imgs):
        yolov8_result = evaluate_model(yolov8_path, test_imgs, test_lbls, "YOLOv8s-CottonWeed")
        all_results.append(yolov8_result)

    # Step 4: Compare with our YOLO11n baseline
    logger.info("\n--- Step 4: Compare with YOLO11n baseline ---")
    yolo11n_path = os.path.join(BASE_DIR, "results", "leave4out", "yolo_8species",
                                 "train", "weights", "best.pt")
    if os.path.exists(yolo11n_path) and os.path.isdir(test_imgs):
        yolo11n_result = evaluate_model(yolo11n_path, test_imgs, test_lbls, "YOLO11n-baseline")
        all_results.append(yolo11n_result)

    # Save results
    logger.info("\n--- Results ---")
    for r in all_results:
        logger.info(f"  {r['model']:<30s} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    results_file = os.path.join(RESULTS_DIR, "clone_and_train_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "description": "Clone and train external weed detection models on CottonWeedDet12",
            "results": all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
