#!/usr/bin/env python3
"""
Ablation study experiments for the weed detection benchmark paper.

Experiments:
1. Prompt engineering: 3 prompts (detailed / grounding / simple) on Qwen-7B
2. Model size: Qwen-7B vs Qwen-3B (same arch, different size)
3. Grounding capability: Tier 1 (native bbox) vs Tier 2/3 (text-only)
4. Fusion IoU threshold: sweep 0.1 to 0.7, plot precision/recall curves

Usage:
    python run_ablations.py --experiment prompt --dataset weed2okok
    python run_ablations.py --experiment size --dataset weed2okok
    python run_ablations.py --experiment grounding --dataset weed2okok
    python run_ablations.py --experiment fusion-iou --dataset weed2okok
    python run_ablations.py --all --dataset weed2okok
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results", "ablations")


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Ablation 1: Prompt Engineering
# ============================================================
def ablation_prompt_engineering(dataset_key, model_key="qwen7b"):
    """Test 3 prompt variants on the same model and dataset."""
    from config import WEED_DETECTION_PROMPT, QWEN_GROUNDING_PROMPT, WEED_DETECTION_PROMPT_SIMPLE
    from local_datasets import get_dataset_path, is_downloaded
    from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels

    print(f"\n{'='*60}")
    print(f"ABLATION: Prompt Engineering ({model_key} on {dataset_key})")
    print(f"{'='*60}")

    if not is_downloaded(dataset_key):
        print(f"[!] Dataset {dataset_key} not downloaded. Run: python datasets.py --download {dataset_key}")
        return None

    prompts = {
        "detailed": WEED_DETECTION_PROMPT,
        "grounding": QWEN_GROUNDING_PROMPT,
        "simple": WEED_DETECTION_PROMPT_SIMPLE,
    }

    ds_path = get_dataset_path(dataset_key)
    test_dir = os.path.join(ds_path, "test", "images")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(ds_path, "valid", "images")
    gt_dir = test_dir.replace("images", "labels")

    gt = load_yolo_labels(gt_dir)
    results = {}

    for prompt_name, prompt_text in prompts.items():
        print(f"\n--- Prompt: {prompt_name} ---")

        # Run inference with this prompt
        output_dir = os.path.join(BASE_DIR, "llm_labeled", f"ablation_prompt_{prompt_name}_{dataset_key}")
        result_json = os.path.join(output_dir, "detection_results.json")

        # Use roboflow_bridge with custom prompt
        from roboflow_bridge import load_model, run_inference, extract_json, convert_bbox_to_yolo
        import cv2

        if not os.path.exists(result_json):
            os.makedirs(os.path.join(output_dir, "detected", "labels"), exist_ok=True)
            model, processor, model_type = load_model(model_key)

            image_files = sorted([
                os.path.join(test_dir, f) for f in os.listdir(test_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            all_results = []
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                start = time.time()

                # Override the module-level prompt temporarily
                import roboflow_bridge
                old_prompt = roboflow_bridge.WEED_PROMPT
                roboflow_bridge.WEED_PROMPT = prompt_text
                response = run_inference(model, processor, img_path, model_type)
                roboflow_bridge.WEED_PROMPT = old_prompt

                elapsed = time.time() - start
                parsed = extract_json(response)
                detections = parsed.get("detections", []) if parsed else []

                all_results.append({
                    "image": img_name,
                    "num_detections": len(detections),
                    "detections": detections,
                    "time_s": round(elapsed, 3),
                })

            with open(result_json, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved: {result_json}")

        # Evaluate
        predictions = load_predictions_from_json(result_json)
        eval_result = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])

        results[prompt_name] = eval_result
        print(f"  mAP@0.5: {eval_result['mAP@0.5']:.4f}, "
              f"Precision: {eval_result['precision@0.5']:.4f}, "
              f"Recall: {eval_result['recall@0.5']:.4f}")

    # Save ablation results
    ensure_result_dir()
    output_path = os.path.join(RESULT_DIR, f"ablation_prompt_{dataset_key}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Prompt ablation saved to {output_path}")

    return results


# ============================================================
# Ablation 2: Model Size
# ============================================================
def ablation_model_size(dataset_key):
    """Compare Qwen-7B vs Qwen-3B (same architecture, different size)."""
    from local_datasets import get_dataset_path, is_downloaded
    from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels

    print(f"\n{'='*60}")
    print(f"ABLATION: Model Size (Qwen 7B vs 3B on {dataset_key})")
    print(f"{'='*60}")

    models = {"qwen7b": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B"}
    ds_path = get_dataset_path(dataset_key)
    gt_dir = os.path.join(ds_path, "test", "labels")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(ds_path, "valid", "labels")

    gt = load_yolo_labels(gt_dir)
    results = {}

    for model_key, model_name in models.items():
        print(f"\n--- Model: {model_name} ({model_key}) ---")

        # Check if results already exist
        result_json = os.path.join(BASE_DIR, "llm_labeled", f"{model_key}_{dataset_key}", "detection_results.json")
        alt_json = os.path.join(RESULT_DIR, f"benchmark_{model_key}_{dataset_key}.json")

        found_json = None
        for path in [result_json, alt_json]:
            if os.path.exists(path):
                found_json = path
                break

        if not found_json:
            print(f"  [!] No results found. Run: python run_full_benchmark.py --dataset {dataset_key} --model {model_key}")
            continue

        predictions = load_predictions_from_json(found_json)
        eval_result = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])
        results[model_key] = {
            "model_name": model_name,
            "evaluation": eval_result,
        }
        print(f"  mAP@0.5: {eval_result['mAP@0.5']:.4f}")

    ensure_result_dir()
    output_path = os.path.join(RESULT_DIR, f"ablation_size_{dataset_key}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Size ablation saved to {output_path}")

    return results


# ============================================================
# Ablation 3: Grounding Capability
# ============================================================
def ablation_grounding(dataset_key):
    """Compare Tier 1 (native bbox) vs Tier 2/3 (text-only) models."""
    from local_datasets import get_dataset_path
    from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels

    print(f"\n{'='*60}")
    print(f"ABLATION: Grounding Capability on {dataset_key}")
    print(f"{'='*60}")

    tiers = {
        "tier1_native_bbox": ["qwen7b", "florence2", "moondream"],
        "tier2_text_based": ["llava:13b", "llama3.2-vision:11b"],
    }

    ds_path = get_dataset_path(dataset_key)
    gt_dir = os.path.join(ds_path, "test", "labels")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(ds_path, "valid", "labels")

    gt = load_yolo_labels(gt_dir)
    results = {}

    for tier_name, models in tiers.items():
        print(f"\n--- {tier_name} ---")
        tier_results = {}

        for model_key in models:
            model_clean = model_key.replace(":", "-").replace("/", "-")
            possible_paths = [
                os.path.join(BASE_DIR, "llm_labeled", f"{model_key}_{dataset_key}", "detection_results.json"),
                os.path.join(BASE_DIR, "results", f"ollama_{model_clean}_{dataset_key}.json"),
                os.path.join(BASE_DIR, "results", f"hf_benchmark_{model_key}_{dataset_key}.json"),
            ]

            found_json = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_json = path
                    break

            if not found_json:
                print(f"  {model_key}: no results found")
                continue

            predictions = load_predictions_from_json(found_json)
            eval_result = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])
            tier_results[model_key] = eval_result
            print(f"  {model_key}: mAP@0.5={eval_result['mAP@0.5']:.4f}, "
                  f"Recall={eval_result['recall@0.5']:.4f}")

        results[tier_name] = tier_results

    ensure_result_dir()
    output_path = os.path.join(RESULT_DIR, f"ablation_grounding_{dataset_key}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Grounding ablation saved to {output_path}")

    return results


# ============================================================
# Ablation 4: Fusion IoU Threshold
# ============================================================
def ablation_fusion_iou(dataset_key, yolo_json=None, llm_json=None):
    """Sweep IoU threshold from 0.1 to 0.7 for YOLO+LLM fusion."""
    from local_datasets import get_dataset_path
    from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels
    from yolo_llm_fusion import fuse_detections

    print(f"\n{'='*60}")
    print(f"ABLATION: Fusion IoU Threshold Sweep on {dataset_key}")
    print(f"{'='*60}")

    ds_path = get_dataset_path(dataset_key)
    gt_dir = os.path.join(ds_path, "test", "labels")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(ds_path, "valid", "labels")

    gt = load_yolo_labels(gt_dir)

    # Find YOLO and LLM result files
    if not yolo_json:
        yolo_json = os.path.join(BASE_DIR, "results", f"yolo_zero_shot_{dataset_key}.json")
    if not llm_json:
        llm_json = os.path.join(BASE_DIR, "llm_labeled", f"qwen7b_{dataset_key}", "detection_results.json")

    if not os.path.exists(yolo_json):
        print(f"[!] YOLO results not found: {yolo_json}")
        return None
    if not os.path.exists(llm_json):
        print(f"[!] LLM results not found: {llm_json}")
        return None

    # Load both result sets
    with open(yolo_json) as f:
        yolo_results = json.load(f)
    with open(llm_json) as f:
        llm_results = json.load(f)

    # Build per-image lookup
    yolo_by_image = {Path(r["image"]).stem: r.get("detections", []) for r in yolo_results}
    llm_by_image = {Path(r["image"]).stem: r.get("detections", []) for r in llm_results}

    thresholds = np.arange(0.1, 0.75, 0.05)
    results = {}

    for iou_thresh in thresholds:
        iou_thresh = round(float(iou_thresh), 2)
        print(f"\n  IoU threshold: {iou_thresh}")

        # Fuse for each image
        fused_predictions = {}
        all_images = set(yolo_by_image.keys()) | set(llm_by_image.keys())

        for img_stem in all_images:
            yolo_dets = yolo_by_image.get(img_stem, [])
            llm_dets = llm_by_image.get(img_stem, [])
            fused = fuse_detections(yolo_dets, llm_dets, iou_threshold=iou_thresh)

            # Convert fused detections to eval format
            pred_boxes = []
            for det in fused:
                bbox = det.get("bbox", [0, 0, 0, 0])
                conf = det.get("confidence", 0.5)
                if isinstance(conf, str):
                    conf = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf, 0.5)

                # Convert percentage [x1,y1,x2,y2] to normalized [cx,cy,w,h]
                x1, y1, x2, y2 = [v / 100 if v > 1 else v for v in bbox]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 0 and h > 0:
                    pred_boxes.append((0, cx, cy, w, h, float(conf)))

            fused_predictions[img_stem] = pred_boxes

        eval_result = evaluate_dataset(gt, fused_predictions, iou_thresholds=[0.25, 0.5, 0.75])
        results[str(iou_thresh)] = eval_result
        print(f"    mAP@0.5: {eval_result['mAP@0.5']:.4f}, "
              f"Prec: {eval_result['precision@0.5']:.4f}, "
              f"Rec: {eval_result['recall@0.5']:.4f}")

    ensure_result_dir()
    output_path = os.path.join(RESULT_DIR, f"ablation_fusion_iou_{dataset_key}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Fusion IoU ablation saved to {output_path}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation studies for weed detection benchmark")
    parser.add_argument("--experiment", type=str,
                        choices=["prompt", "size", "grounding", "fusion-iou"],
                        help="Which ablation to run")
    parser.add_argument("--all", action="store_true", help="Run all ablations")
    parser.add_argument("--dataset", type=str, default="weed2okok",
                        help="Dataset to use for ablations")
    parser.add_argument("--yolo-json", type=str, help="YOLO results JSON (for fusion ablation)")
    parser.add_argument("--llm-json", type=str, help="LLM results JSON (for fusion ablation)")
    args = parser.parse_args()

    if args.all:
        ablation_prompt_engineering(args.dataset)
        ablation_model_size(args.dataset)
        ablation_grounding(args.dataset)
        ablation_fusion_iou(args.dataset, args.yolo_json, args.llm_json)
    elif args.experiment == "prompt":
        ablation_prompt_engineering(args.dataset)
    elif args.experiment == "size":
        ablation_model_size(args.dataset)
    elif args.experiment == "grounding":
        ablation_grounding(args.dataset)
    elif args.experiment == "fusion-iou":
        ablation_fusion_iou(args.dataset, args.yolo_json, args.llm_json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
