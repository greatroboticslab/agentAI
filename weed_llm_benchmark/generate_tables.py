#!/usr/bin/env python3
"""
Generate LaTeX tables for the weed detection benchmark paper.

Tables:
1. Main benchmark results (all models x all datasets)
2. Model characteristics (size, grounding, backend)
3. Ablation: prompt engineering
4. Ablation: model size
5. YOLO vs LLM vs Fusion comparison
6. Dataset statistics

Usage:
    python generate_tables.py --all
    python generate_tables.py --table main-benchmark
"""

import argparse
import json
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
TABLE_DIR = os.path.join(BASE_DIR, "tables")


def ensure_table_dir():
    os.makedirs(TABLE_DIR, exist_ok=True)


def save_table(name, latex_str):
    """Save a LaTeX table to file."""
    ensure_table_dir()
    path = os.path.join(TABLE_DIR, f"{name}.tex")
    with open(path, "w") as f:
        f.write(latex_str)
    print(f"[+] Saved: {path}")
    return path


def table_dataset_statistics():
    """Table 1: Dataset statistics."""
    latex = r"""\begin{table}[ht]
\centering
\caption{Dataset statistics for the weed detection benchmark.}
\label{tab:datasets}
\begin{tabular}{lrrrl}
\toprule
\textbf{Dataset} & \textbf{Images} & \textbf{Classes} & \textbf{Annotations} & \textbf{Context} \\
\midrule
CottonWeedDet12 & 5,648 & 12 & BBox & Cotton fields \\
DeepWeeds & 17,509 & 9 & BBox & Rangeland \\
weed2okok & 106 & 1 & BBox & Lab data \\
\bottomrule
\end{tabular}
\end{table}"""
    return save_table("tab1_datasets", latex)


def table_model_characteristics():
    """Table 2: Model characteristics."""
    latex = r"""\begin{table}[ht]
\centering
\caption{Vision LLM models evaluated in the benchmark.}
\label{tab:models}
\begin{tabular}{llrrll}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{Size} & \textbf{Tier} & \textbf{BBox} & \textbf{Backend} \\
\midrule
YOLO11n & 2.6M & 5MB & -- & Native & Ultralytics \\
\midrule
Qwen2.5-VL-7B & 7B & 14GB & 1 & Native & HuggingFace \\
Qwen2.5-VL-3B & 3B & 6GB & 1 & Native & HuggingFace \\
MiniCPM-V-2.6 & 8B & 16GB & 1 & Text & HuggingFace \\
InternVL2-8B & 8B & 16GB & 1 & Partial & HuggingFace \\
Florence-2-large & 0.7B & 1.5GB & 1 & Native & HuggingFace \\
Moondream2 & 1.8B & 1.7GB & 1 & API & Ollama \\
LLaVA-13B & 13B & 8GB & 3 & Text & Ollama \\
Llama3.2-Vision & 11B & 7GB & 2 & Text & Ollama \\
\bottomrule
\end{tabular}
\end{table}"""
    return save_table("tab2_models", latex)


def table_main_benchmark():
    """Table 3: Main benchmark results."""
    # Load results
    summary_path = os.path.join(RESULT_DIR, "benchmark_summary.json")
    results = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
            results = data.get("results", [])

    # Build table rows
    datasets = ["cottonweeddet12", "deepweeds", "weed2okok"]
    ds_short = {"cottonweeddet12": "CWD12", "deepweeds": "DW", "weed2okok": "w2ok"}

    # Collect all unique models
    models = list(dict.fromkeys(r["model"] for r in results)) if results else [
        "YOLO11n (zero-shot)", "YOLO11n (fine-tuned)",
        "Qwen2.5-VL-7B", "Qwen2.5-VL-3B", "MiniCPM-V-2.6",
        "InternVL2-8B", "Florence-2", "Moondream",
        "LLaVA-13B", "Llama3.2-Vision",
    ]

    # Header
    header_cols = " & ".join([f"\\multicolumn{{3}}{{c}}{{{ds_short.get(d, d)}}}" for d in datasets])
    sub_header = " & ".join(["mAP & P & R"] * len(datasets))

    rows = []
    for model in models:
        row_vals = [model.replace("_", r"\_")]
        for ds in datasets:
            match = [r for r in results if r["model"] == model and r["dataset"] == ds]
            if match:
                r = match[0]
                row_vals.append(f"{r.get('mAP@0.5', 0):.3f}")
                row_vals.append(f"{r.get('precision', 0):.3f}")
                row_vals.append(f"{r.get('recall', 0):.3f}")
            else:
                row_vals.extend(["--", "--", "--"])
        rows.append(" & ".join(row_vals) + r" \\")

    rows_str = "\n".join(rows)

    latex = rf"""\begin{{table*}}[ht]
\centering
\caption{{Main benchmark results: mAP@0.5, Precision, and Recall across all datasets.
Values computed in binary (weed/not-weed) mode with IoU threshold 0.5.}}
\label{{tab:main-results}}
\begin{{tabular}}{{l|{'ccc|' * (len(datasets) - 1)}ccc}}
\toprule
\textbf{{Model}} & {header_cols} \\
 & {sub_header} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table*}}"""
    return save_table("tab3_main_benchmark", latex)


def table_fusion_comparison():
    """Table 4: YOLO vs LLM vs Fusion comparison."""
    latex = r"""\begin{table}[ht]
\centering
\caption{Detection performance comparison: YOLO-only vs best LLM vs YOLO+LLM fusion.}
\label{tab:fusion}
\begin{tabular}{llccc}
\toprule
\textbf{Dataset} & \textbf{Method} & \textbf{mAP@0.5} & \textbf{Precision} & \textbf{Recall} \\
\midrule
\multirow{3}{*}{CottonWeedDet12}
 & YOLO11n (zero-shot) & -- & -- & -- \\
 & Qwen2.5-VL-7B & -- & -- & -- \\
 & YOLO+LLM Fusion & -- & -- & -- \\
\midrule
\multirow{3}{*}{DeepWeeds}
 & YOLO11n (zero-shot) & -- & -- & -- \\
 & Qwen2.5-VL-7B & -- & -- & -- \\
 & YOLO+LLM Fusion & -- & -- & -- \\
\midrule
\multirow{3}{*}{weed2okok}
 & YOLO11n (zero-shot) & -- & -- & -- \\
 & Qwen2.5-VL-7B & -- & -- & -- \\
 & YOLO+LLM Fusion & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}"""
    return save_table("tab4_fusion", latex)


def table_prompt_ablation():
    """Table 5: Prompt engineering ablation."""
    # Try to load results
    ablation_file = os.path.join(RESULT_DIR, "ablations", "ablation_prompt_weed2okok.json")
    data = {}
    if os.path.exists(ablation_file):
        with open(ablation_file) as f:
            data = json.load(f)

    rows = []
    for prompt_name in ["detailed", "grounding", "simple"]:
        if prompt_name in data:
            d = data[prompt_name]
            rows.append(f"{prompt_name.capitalize()} & {d.get('mAP@0.5', 0):.3f} & "
                       f"{d.get('precision@0.5', 0):.3f} & {d.get('recall@0.5', 0):.3f} & "
                       f"{d.get('f1@0.5', 0):.3f} \\\\")
        else:
            rows.append(f"{prompt_name.capitalize()} & -- & -- & -- & -- \\\\")

    rows_str = "\n".join(rows)

    latex = rf"""\begin{{table}}[ht]
\centering
\caption{{Effect of prompt design on Qwen2.5-VL-7B detection quality.}}
\label{{tab:prompt-ablation}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Prompt}} & \textbf{{mAP@0.5}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}"""
    return save_table("tab5_prompt_ablation", latex)


def table_size_ablation():
    """Table 6: Model size ablation."""
    latex = r"""\begin{table}[ht]
\centering
\caption{Effect of model size: Qwen2.5-VL-7B vs 3B.}
\label{tab:size-ablation}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{mAP@0.5} & \textbf{Precision} & \textbf{Recall} & \textbf{Time (s/img)} \\
\midrule
Qwen2.5-VL-7B & 7B & -- & -- & -- & -- \\
Qwen2.5-VL-3B & 3B & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}"""
    return save_table("tab6_size_ablation", latex)


def generate_all():
    """Generate all tables."""
    print("[*] Generating all LaTeX tables...")
    table_dataset_statistics()
    table_model_characteristics()
    table_main_benchmark()
    table_fusion_comparison()
    table_prompt_ablation()
    table_size_ablation()
    print(f"\n[+] All tables saved to {TABLE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for paper")
    parser.add_argument("--table", type=str,
                        choices=["datasets", "models", "main-benchmark",
                                 "fusion", "prompt-ablation", "size-ablation"],
                        help="Generate specific table")
    parser.add_argument("--all", action="store_true", help="Generate all tables")
    args = parser.parse_args()

    if args.all:
        generate_all()
    elif args.table:
        table_map = {
            "datasets": table_dataset_statistics,
            "models": table_model_characteristics,
            "main-benchmark": table_main_benchmark,
            "fusion": table_fusion_comparison,
            "prompt-ablation": table_prompt_ablation,
            "size-ablation": table_size_ablation,
        }
        table_map[args.table]()
    else:
        generate_all()


if __name__ == "__main__":
    main()
