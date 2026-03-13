"""
Visualize and compare weed detection results across models.
Draws bounding boxes on images and generates comparison charts.

Usage:
    python visualize_results.py --results results/hf_benchmark_*.json
    python visualize_results.py --results results/ --image images/weed1.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from config import IMAGE_DIR, RESULT_DIR


COLORS = {
    "weed": "red",
    "crop": "green",
    "grass": "yellow",
    "plant": "blue",
    "dandelion": "orange",
    "thistle": "purple",
    "clover": "pink",
}


def get_color(label):
    label_lower = label.lower()
    for key, color in COLORS.items():
        if key in label_lower:
            return color
    return "cyan"


def draw_detections(image_path, detections, model_name, output_path):
    """Draw bounding boxes on image."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    w, h = img.size
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # bbox is in percentage [x1, y1, x2, y2]
        x1 = bbox[0] / 100 * w
        y1 = bbox[1] / 100 * h
        x2 = bbox[2] / 100 * w
        y2 = bbox[3] / 100 * h

        label = det.get("label", "unknown")
        species = det.get("species", "")
        conf = det.get("confidence", "")
        color = get_color(label)

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        display_label = f"{label}"
        if species and species != label:
            display_label += f" ({species})"
        if conf:
            display_label += f" [{conf}]"

        ax.text(
            x1, y1 - 5, display_label,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )

    ax.set_title(f"Model: {model_name}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved visualization: {output_path}")


def compare_models_chart(all_results, output_path):
    """Create bar chart comparing models."""
    model_stats = {}
    for r in all_results:
        if not r.get("success"):
            continue
        model = r.get("model_key") or r.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {
                "num_detections": [],
                "num_weeds": [],
                "time_s": [],
                "json_valid": 0,
                "has_bbox": 0,
                "total": 0,
            }
        s = model_stats[model]
        s["total"] += 1
        s["num_detections"].append(r.get("num_detections", 0))
        s["num_weeds"].append(r.get("num_weeds", 0))
        s["time_s"].append(r.get("total_duration_s", 0))
        if r.get("json_valid"):
            s["json_valid"] += 1
        if r.get("has_bbox"):
            s["has_bbox"] += 1

    if not model_stats:
        print("[!] No successful results to chart.")
        return

    models = list(model_stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Avg detections
    ax = axes[0][0]
    avg_det = [sum(model_stats[m]["num_detections"]) / max(len(model_stats[m]["num_detections"]), 1) for m in models]
    ax.barh(models, avg_det, color="steelblue")
    ax.set_xlabel("Avg Detections per Image")
    ax.set_title("Detection Count")

    # 2. Avg weeds detected
    ax = axes[0][1]
    avg_weeds = [sum(model_stats[m]["num_weeds"]) / max(len(model_stats[m]["num_weeds"]), 1) for m in models]
    ax.barh(models, avg_weeds, color="firebrick")
    ax.set_xlabel("Avg Weeds Detected")
    ax.set_title("Weed Detection Count")

    # 3. JSON valid rate & bbox rate
    ax = axes[1][0]
    json_rate = [model_stats[m]["json_valid"] / max(model_stats[m]["total"], 1) * 100 for m in models]
    bbox_rate = [model_stats[m]["has_bbox"] / max(model_stats[m]["total"], 1) * 100 for m in models]
    x_pos = range(len(models))
    bar_w = 0.35
    ax.bar([p - bar_w/2 for p in x_pos], json_rate, bar_w, label="JSON Valid %", color="green")
    ax.bar([p + bar_w/2 for p in x_pos], bbox_rate, bar_w, label="Has BBox %", color="orange")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Percentage")
    ax.set_title("Output Quality")
    ax.legend()

    # 4. Inference time
    ax = axes[1][1]
    avg_time = [sum(model_stats[m]["time_s"]) / max(len(model_stats[m]["time_s"]), 1) for m in models]
    ax.barh(models, avg_time, color="mediumpurple")
    ax.set_xlabel("Avg Inference Time (s)")
    ax.set_title("Speed")

    plt.suptitle("Weed Detection LLM Benchmark Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize weed detection benchmark results")
    parser.add_argument("--results", type=str, required=True, help="Result JSON file or directory")
    parser.add_argument("--image", type=str, default=None, help="Specific image to visualize")
    args = parser.parse_args()

    # Load results
    all_results = []
    results_path = Path(args.results)
    if results_path.is_dir():
        for f in sorted(results_path.glob("*.json")):
            with open(f) as fh:
                all_results.extend(json.load(fh))
    else:
        with open(results_path) as fh:
            all_results.extend(json.load(fh))

    print(f"[*] Loaded {len(all_results)} result entries")

    vis_dir = os.path.join(RESULT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Draw detections on images
    for r in all_results:
        if not r.get("success") or not r.get("parsed_json"):
            continue

        parsed = r["parsed_json"]
        detections = parsed.get("detections", [])
        if not detections:
            continue

        img_name = r.get("image", "unknown")
        model_key = r.get("model_key") or r.get("model", "unknown")

        # Find the image file
        if args.image:
            img_path = args.image
        else:
            img_path = os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path):
                print(f"[!] Image not found: {img_path}")
                continue

        out_name = f"{Path(img_name).stem}_{model_key}.png"
        out_path = os.path.join(vis_dir, out_name)
        draw_detections(img_path, detections, model_key, out_path)

    # Generate comparison chart
    chart_path = os.path.join(vis_dir, "model_comparison.png")
    compare_models_chart(all_results, chart_path)


if __name__ == "__main__":
    main()
