#!/usr/bin/env python3
"""
Full benchmark orchestrator: run all models on all datasets.

Loops through the experiment matrix (datasets x models), handles resume,
and aggregates results into a unified comparison table.

Usage:
    # Run everything
    python run_full_benchmark.py --all

    # Run specific dataset + model
    python run_full_benchmark.py --dataset weed2okok --model qwen7b

    # Resume from checkpoint
    python run_full_benchmark.py --all --resume

    # Just aggregate existing results
    python run_full_benchmark.py --aggregate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # parent of weed_llm_benchmark/
RESULT_DIR = os.path.join(BASE_DIR, "results")
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")
CHECKPOINT_FILE = os.path.join(RESULT_DIR, "benchmark_checkpoint.json")

# ============================================================
# Experiment Matrix
# ============================================================
DATASETS = ["cottonweeddet12", "deepweeds", "weed2okok"]

# Models to run on cluster (HuggingFace)
HF_MODELS = ["qwen7b", "qwen3b", "minicpm", "internvl2", "florence2"]

# Models to run via Ollama (local or cluster)
OLLAMA_MODELS = ["moondream", "llava:13b", "llama3.2-vision:11b"]

ALL_MODELS = HF_MODELS + OLLAMA_MODELS


def get_experiment_id(dataset, model):
    """Unique ID for a dataset-model experiment."""
    model_clean = model.replace(":", "-").replace("/", "-")
    return f"{dataset}__{model_clean}"


def load_checkpoint():
    """Load progress checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "start_time": datetime.now().isoformat()}


def save_checkpoint(checkpoint):
    """Save progress checkpoint."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def run_hf_experiment(dataset_key, model_key):
    """Run a HuggingFace model on a dataset."""
    from datasets import get_dataset_path, is_downloaded, download_dataset

    # Ensure dataset is available
    if not is_downloaded(dataset_key):
        print(f"[*] Downloading {dataset_key}...")
        download_dataset(dataset_key)

    ds_path = get_dataset_path(dataset_key)
    test_dir = os.path.join(ds_path, "test", "images")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(ds_path, "valid", "images")

    if not os.path.isdir(test_dir):
        print(f"[!] No test images found for {dataset_key}")
        return None

    # Run detection using roboflow_bridge's detect_images
    from roboflow_bridge import detect_images
    output_dir = os.path.join(BASE_DIR, "llm_labeled", f"{model_key}_{dataset_key}")

    print(f"\n{'='*60}")
    print(f"  Running {model_key} on {dataset_key}")
    print(f"  Images: {test_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    start = time.time()
    result_path = detect_images(test_dir, model_key=model_key, output_dir=output_dir)
    elapsed = time.time() - start

    if result_path:
        # Run evaluation
        result_json = os.path.join(output_dir, "detection_results.json")
        gt_dir = os.path.join(ds_path, "test", "labels")
        if not os.path.isdir(gt_dir):
            gt_dir = os.path.join(ds_path, "valid", "labels")

        eval_result = None
        if os.path.isdir(gt_dir) and os.path.exists(result_json):
            from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels
            gt = load_yolo_labels(gt_dir)
            predictions = load_predictions_from_json(result_json)
            eval_result = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])

        return {
            "dataset": dataset_key,
            "model": model_key,
            "backend": "hf",
            "output_dir": output_dir,
            "result_json": result_json,
            "total_time_s": round(elapsed, 1),
            "evaluation": eval_result,
        }

    return None


def run_ollama_experiment(dataset_key, model_name):
    """Run an Ollama model on a dataset."""
    from datasets import get_dataset_path, is_downloaded, download_dataset

    if not is_downloaded(dataset_key):
        print(f"[*] Downloading {dataset_key}...")
        download_dataset(dataset_key)

    ds_path = get_dataset_path(dataset_key)
    test_dir = os.path.join(ds_path, "test", "images")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(ds_path, "valid", "images")

    if not os.path.isdir(test_dir):
        print(f"[!] No test images found for {dataset_key}")
        return None

    # Import and run ollama benchmark
    print(f"\n{'='*60}")
    print(f"  Running {model_name} (Ollama) on {dataset_key}")
    print(f"{'='*60}")

    image_files = sorted([
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    from test_ollama import query_ollama, extract_json, get_prompt_for_model
    from config import WEED_DETECTION_PROMPT

    prompt = get_prompt_for_model(model_name, use_simple=False)
    all_results = []
    start = time.time()

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        success, response, duration, eval_count = query_ollama(model_name, img_path, prompt)

        detections = []
        if success:
            parsed = extract_json(response)
            if parsed and "detections" in parsed:
                detections = parsed["detections"]

        all_results.append({
            "image": img_name,
            "num_detections": len(detections),
            "detections": detections,
            "time_s": round(duration, 3) if duration else 0,
            "success": success,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(image_files)}] processed")

    elapsed = time.time() - start

    # Save results
    model_clean = model_name.replace(":", "-").replace("/", "-")
    output_path = os.path.join(RESULT_DIR, f"ollama_{model_clean}_{dataset_key}.json")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Evaluate
    gt_dir = os.path.join(ds_path, "test", "labels")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(ds_path, "valid", "labels")

    eval_result = None
    if os.path.isdir(gt_dir):
        from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels
        gt = load_yolo_labels(gt_dir)
        predictions = load_predictions_from_json(output_path)
        eval_result = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])

    return {
        "dataset": dataset_key,
        "model": model_name,
        "backend": "ollama",
        "result_json": output_path,
        "total_time_s": round(elapsed, 1),
        "evaluation": eval_result,
    }


def run_experiment(dataset_key, model_key):
    """Run a single experiment (auto-detect backend)."""
    if model_key in HF_MODELS:
        return run_hf_experiment(dataset_key, model_key)
    else:
        return run_ollama_experiment(dataset_key, model_key)


def aggregate_results():
    """Aggregate all completed results into a comparison table."""
    checkpoint = load_checkpoint()
    completed = checkpoint.get("completed", {})

    if not completed:
        print("[!] No completed experiments found.")
        return None

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    # Build comparison table
    header = f"{'Model':<25} {'Dataset':<20} {'mAP@0.5':>8} {'mAP@0.5:95':>10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Time':>6}"
    print(header)
    print("-" * 90)

    rows = []
    for exp_id, result in sorted(completed.items()):
        ev = result.get("evaluation", {})
        if not ev:
            continue
        row = {
            "model": result["model"],
            "dataset": result["dataset"],
            "mAP@0.5": ev.get("mAP@0.5", 0),
            "mAP@0.5:0.95": ev.get("mAP@0.5:0.95", 0),
            "precision": ev.get("precision@0.5", 0),
            "recall": ev.get("recall@0.5", 0),
            "f1": ev.get("f1@0.5", 0),
            "time": result.get("total_time_s", 0),
        }
        rows.append(row)
        print(f"{row['model']:<25} {row['dataset']:<20} {row['mAP@0.5']:>8.4f} "
              f"{row['mAP@0.5:0.95']:>10.4f} {row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} {row['f1']:>6.3f} {row['time']:>5.0f}s")

    # Save aggregated results
    agg_path = os.path.join(RESULT_DIR, "benchmark_summary.json")
    with open(agg_path, "w") as f:
        json.dump({"results": rows, "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"\n[+] Summary saved to {agg_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Full weed detection benchmark orchestrator")
    parser.add_argument("--all", action="store_true", help="Run all datasets x models")
    parser.add_argument("--dataset", type=str, help="Specific dataset to run")
    parser.add_argument("--model", type=str, help="Specific model to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate existing results")
    parser.add_argument("--hf-only", action="store_true", help="Only HuggingFace models")
    parser.add_argument("--ollama-only", action="store_true", help="Only Ollama models")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
        return

    # Determine experiment list
    datasets = [args.dataset] if args.dataset else DATASETS
    if args.model:
        models = [args.model]
    elif args.hf_only:
        models = HF_MODELS
    elif args.ollama_only:
        models = OLLAMA_MODELS
    else:
        models = ALL_MODELS

    # Load checkpoint for resume
    checkpoint = load_checkpoint()

    experiments = []
    for ds in datasets:
        for model in models:
            exp_id = get_experiment_id(ds, model)
            if args.resume and exp_id in checkpoint.get("completed", {}):
                print(f"[skip] {exp_id} (already completed)")
                continue
            experiments.append((ds, model, exp_id))

    print(f"\n[*] {len(experiments)} experiments to run")
    print(f"    Datasets: {datasets}")
    print(f"    Models:   {models}")

    for i, (ds, model, exp_id) in enumerate(experiments):
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT {i+1}/{len(experiments)}: {model} x {ds}")
        print(f"{'#'*60}")

        try:
            result = run_experiment(ds, model)
            if result:
                checkpoint.setdefault("completed", {})[exp_id] = result
                save_checkpoint(checkpoint)
                print(f"[+] {exp_id} completed successfully")
            else:
                checkpoint.setdefault("failed", {})[exp_id] = {
                    "error": "No result returned",
                    "timestamp": datetime.now().isoformat(),
                }
                save_checkpoint(checkpoint)
        except Exception as e:
            print(f"[!] {exp_id} failed: {e}")
            import traceback
            traceback.print_exc()
            checkpoint.setdefault("failed", {})[exp_id] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            save_checkpoint(checkpoint)

    # Final aggregation
    print("\n" + "=" * 60)
    aggregate_results()


if __name__ == "__main__":
    main()
