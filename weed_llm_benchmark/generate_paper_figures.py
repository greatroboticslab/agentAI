#!/usr/bin/env python3
"""
Generate publication-quality figures for the weed detection benchmark paper.

Figures:
1. Main benchmark comparison (mAP bar chart across models and datasets)
2. YOLO vs LLM vs Fusion comparison
3. Ablation: prompt engineering effect
4. Ablation: model size effect
5. Ablation: fusion IoU threshold sweep (precision/recall curves)
6. Qualitative detection examples

Usage:
    python generate_paper_figures.py --results-dir results/
    python generate_paper_figures.py --figure main-benchmark
    python generate_paper_figures.py --all
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Consistent colors for models
MODEL_COLORS = {
    "YOLO11n (zero-shot)": "#2196F3",
    "YOLO11n (fine-tuned)": "#1565C0",
    "Qwen2.5-VL-7B": "#4CAF50",
    "Qwen2.5-VL-3B": "#81C784",
    "MiniCPM-V-2.6": "#FF9800",
    "InternVL2-8B": "#9C27B0",
    "Florence-2": "#F44336",
    "Moondream": "#00BCD4",
    "LLaVA-13B": "#795548",
    "Llama3.2-Vision": "#607D8B",
    "Fusion (best)": "#FFD700",
}

DATASET_LABELS = {
    "cottonweeddet12": "CottonWeedDet12",
    "deepweeds": "DeepWeeds",
    "weed2okok": "weed2okok",
}


def ensure_fig_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_summary():
    """Load benchmark summary results."""
    summary_path = os.path.join(RESULT_DIR, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        print(f"[!] No summary found at {summary_path}")
        print("    Run: python run_full_benchmark.py --aggregate")
        return None
    with open(summary_path) as f:
        return json.load(f)


def fig_main_benchmark(summary_data=None):
    """Figure 1: Main benchmark comparison — grouped bar chart of mAP@0.5 across models and datasets."""
    ensure_fig_dir()

    if summary_data is None:
        summary_data = load_summary()
    if not summary_data:
        # Use placeholder data for figure skeleton
        summary_data = {"results": []}

    results = summary_data.get("results", [])

    # Organize by dataset
    datasets = list(DATASET_LABELS.keys())
    models = list(dict.fromkeys(r["model"] for r in results))  # preserve order

    if not models:
        models = ["YOLO11n (zero-shot)", "Qwen2.5-VL-7B", "Qwen2.5-VL-3B",
                   "MiniCPM-V-2.6", "Florence-2", "Moondream"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        values = []
        for ds in datasets:
            match = [r for r in results if r["model"] == model and r["dataset"] == ds]
            values.append(match[0]["mAP@0.5"] if match else 0)

        color = MODEL_COLORS.get(model, f"C{i}")
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=model, color=color)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Weed Detection Performance: Vision LLMs vs YOLO Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    output = os.path.join(FIGURE_DIR, "fig1_main_benchmark.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def fig_yolo_vs_llm_vs_fusion(summary_data=None):
    """Figure 2: YOLO-only vs best-LLM-only vs Fusion comparison."""
    ensure_fig_dir()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    metrics = ["mAP@0.5", "precision", "recall"]
    metric_labels = ["mAP@0.5", "Precision@0.5", "Recall@0.5"]
    categories = ["YOLO11n\n(zero-shot)", "Best LLM\n(Qwen-7B)", "YOLO+LLM\nFusion"]
    colors = ["#2196F3", "#4CAF50", "#FFD700"]

    datasets = list(DATASET_LABELS.keys())

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.25

        for i, (cat, color) in enumerate(zip(categories, colors)):
            # Placeholder values — will be filled from actual results
            values = [0.0] * len(datasets)

            if summary_data:
                for j, ds in enumerate(datasets):
                    for r in summary_data.get("results", []):
                        if r["dataset"] == ds:
                            if cat.startswith("YOLO") and "yolo" in r["model"].lower() and "fusion" not in r["model"].lower():
                                values[j] = r.get(metric, 0)
                            elif cat.startswith("Best LLM") and "qwen" in r["model"].lower() and "7b" in r["model"].lower():
                                values[j] = r.get(metric, 0)
                            elif cat.startswith("YOLO+LLM") and "fusion" in r["model"].lower():
                                values[j] = r.get(metric, 0)

            ax.bar(x + (i - 1) * width, values, width * 0.9, label=cat, color=color)

        ax.set_xlabel("Dataset")
        ax.set_ylabel(label if ax_idx == 0 else "")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=8)
        ax.set_ylim(0, 1.0)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("YOLO vs LLM vs Fusion: Detection Quality Comparison", fontsize=14)
    plt.tight_layout()

    output = os.path.join(FIGURE_DIR, "fig2_yolo_vs_llm_vs_fusion.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def fig_prompt_ablation():
    """Figure 3: Prompt engineering ablation results."""
    ensure_fig_dir()

    # Try to load ablation results
    ablation_files = list(Path(RESULT_DIR).glob("ablations/ablation_prompt_*.json"))
    prompt_data = {}
    for f in ablation_files:
        with open(f) as fh:
            prompt_data.update(json.load(fh))

    prompts = list(prompt_data.keys()) if prompt_data else ["detailed", "grounding", "simple"]
    metrics = ["mAP@0.5", "precision@0.5", "recall@0.5", "f1@0.5"]
    metric_labels = ["mAP@0.5", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        values = []
        for p in prompts:
            if p in prompt_data:
                values.append(prompt_data[p].get(metric, 0))
            else:
                values.append(0)

        colors = ["#4CAF50", "#2196F3", "#FF9800"][:len(prompts)]
        ax.bar(prompts, values, color=colors)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)
        ax.set_title(label)

    plt.suptitle("Effect of Prompt Design on Detection Quality (Qwen2.5-VL-7B)", fontsize=13)
    plt.tight_layout()

    output = os.path.join(FIGURE_DIR, "fig3_prompt_ablation.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def fig_fusion_iou_sweep():
    """Figure 4: Fusion IoU threshold sweep — precision/recall curves."""
    ensure_fig_dir()

    # Try to load fusion ablation
    fusion_files = list(Path(RESULT_DIR).glob("ablations/ablation_fusion_iou_*.json"))
    fusion_data = {}
    for f in fusion_files:
        with open(f) as fh:
            fusion_data = json.load(fh)
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if fusion_data:
        thresholds = sorted(fusion_data.keys(), key=float)
        precisions = [fusion_data[t].get("precision@0.5", 0) for t in thresholds]
        recalls = [fusion_data[t].get("recall@0.5", 0) for t in thresholds]
        f1s = [fusion_data[t].get("f1@0.5", 0) for t in thresholds]
        maps = [fusion_data[t].get("mAP@0.5", 0) for t in thresholds]
        thresh_vals = [float(t) for t in thresholds]
    else:
        # Placeholder
        thresh_vals = np.arange(0.1, 0.75, 0.05)
        precisions = [0] * len(thresh_vals)
        recalls = [0] * len(thresh_vals)
        f1s = [0] * len(thresh_vals)
        maps = [0] * len(thresh_vals)

    # Left: Precision, Recall, F1 vs IoU threshold
    ax1.plot(thresh_vals, precisions, "b-o", label="Precision", markersize=4)
    ax1.plot(thresh_vals, recalls, "r-s", label="Recall", markersize=4)
    ax1.plot(thresh_vals, f1s, "g-^", label="F1", markersize=4)
    ax1.set_xlabel("Fusion IoU Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision/Recall vs Fusion IoU Threshold")
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # Right: mAP@0.5 vs IoU threshold
    ax2.plot(thresh_vals, maps, "k-D", label="mAP@0.5", markersize=5)
    ax2.set_xlabel("Fusion IoU Threshold")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_title("mAP@0.5 vs Fusion IoU Threshold")
    ax2.set_ylim(0, 1.0)
    ax2.legend()

    plt.suptitle("Effect of IoU Threshold on YOLO+LLM Fusion Quality", fontsize=13)
    plt.tight_layout()

    output = os.path.join(FIGURE_DIR, "fig4_fusion_iou_sweep.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def fig_model_comparison_radar():
    """Figure 5: Radar/spider chart comparing model capabilities."""
    ensure_fig_dir()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    categories = ["mAP@0.5", "Precision", "Recall", "Speed\n(1/time)", "JSON\nValidity", "BBox\nQuality"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Placeholder data for models (will be filled from actual results)
    models_data = {
        "Qwen2.5-VL-7B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "YOLO11n": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Florence-2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    colors = ["#4CAF50", "#2196F3", "#F44336"]
    for (model, values), color in zip(models_data.items(), colors):
        values += values[:1]  # close
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Capability Comparison", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output = os.path.join(FIGURE_DIR, "fig5_model_radar.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def fig_qualitative_examples():
    """Figure 6: Side-by-side detection examples (GT vs YOLO vs LLM vs Fusion)."""
    ensure_fig_dir()

    # This creates a template figure. Actual images need to be filled in
    # when detection results are available.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    titles = ["Ground Truth", "YOLO11n", "Qwen2.5-VL-7B", "YOLO+LLM Fusion"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11)

    for row in range(2):
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].text(0.5, 0.5, "(image)", transform=axes[row, col].transAxes,
                               ha="center", va="center", fontsize=10, color="gray")

    axes[0, 0].set_ylabel("Example 1", fontsize=11)
    axes[1, 0].set_ylabel("Example 2", fontsize=11)

    plt.suptitle("Qualitative Detection Examples", fontsize=14)
    plt.tight_layout()

    output = os.path.join(FIGURE_DIR, "fig6_qualitative_examples.pdf")
    plt.savefig(output)
    plt.savefig(output.replace(".pdf", ".png"))
    plt.close()
    print(f"[+] Saved: {output}")


def generate_all():
    """Generate all figures."""
    print("[*] Generating all paper figures...")
    summary = load_summary()

    fig_main_benchmark(summary)
    fig_yolo_vs_llm_vs_fusion(summary)
    fig_prompt_ablation()
    fig_fusion_iou_sweep()
    fig_model_comparison_radar()
    fig_qualitative_examples()

    print(f"\n[+] All figures saved to {FIGURE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--figure", type=str,
                        choices=["main-benchmark", "yolo-vs-llm", "prompt-ablation",
                                 "fusion-iou", "radar", "qualitative"],
                        help="Generate specific figure")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--results-dir", type=str, default=RESULT_DIR,
                        help="Results directory")
    args = parser.parse_args()

    if args.all:
        generate_all()
    elif args.figure:
        figure_map = {
            "main-benchmark": fig_main_benchmark,
            "yolo-vs-llm": fig_yolo_vs_llm_vs_fusion,
            "prompt-ablation": fig_prompt_ablation,
            "fusion-iou": fig_fusion_iou_sweep,
            "radar": fig_model_comparison_radar,
            "qualitative": fig_qualitative_examples,
        }
        figure_map[args.figure]()
    else:
        generate_all()


if __name__ == "__main__":
    main()
