# Weed Detection LLM Benchmark

A benchmark framework for evaluating open-source vision LLMs on agricultural weed detection. Uses large vision models as a "second opinion" to correct and supplement YOLO11n detections.

## Overview

```
                    +-----------------+
                    | Roboflow Dataset|
                    +--------+--------+
                             |
                         Download
                             |
                    +--------v--------+
                    | Weed Images     |
                    +--------+--------+
                             |
               +-------------+-------------+
               |                           |
      +--------v--------+        +--------v--------+
      | YOLO11n         |        | Vision LLM      |
      | (fast, precise) |        | (thorough,       |
      |                 |        |  species ID)     |
      +--------+--------+        +--------+--------+
               |                           |
               +-------------+-------------+
                             |
                    +--------v--------+
                    | IoU Fusion      |
                    | - Keep all YOLO |
                    | - Add LLM-only  |
                    | - Flag disagree |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Merged Results  |
                    | + Upload back   |
                    +--------+--------+
```

**Why both?** YOLO is fast but may miss weeds or misclassify. LLMs provide species-level identification and catch what YOLO misses. The fusion pipeline merges both via IoU matching.

## Supported Models

| Key | Model | Size | Bounding Box Support |
|-----|-------|------|---------------------|
| `qwen7b` | Qwen2.5-VL-7B-Instruct | ~14GB | Native bbox output |
| `qwen3b` | Qwen2.5-VL-3B-Instruct | ~6GB | Native bbox output |
| `minicpm` | MiniCPM-V-2.6 | ~16GB | Text-based |
| `internvl2` | InternVL2-8B | ~16GB | Partial |
| `florence2` | Florence-2-large | ~1.5GB | Native grounding |

## Project Structure

```
weed_llm_benchmark/
  roboflow_bridge.py           # Main pipeline: Roboflow download -> LLM detect -> upload
  weed_lable_update_download.py # YOLO auto-labeling + Roboflow upload
  yolo_llm_fusion.py           # Merge YOLO + LLM detections via IoU
  test_hf_models.py            # Benchmark HuggingFace vision models
  test_ollama.py               # Benchmark Ollama-served models
  quick_test.py                # Quick single-image test
  visualize_results.py         # Draw bboxes on images, comparison charts
  config.py                    # Model definitions, prompts, shared settings
  run_roboflow_bridge.sh       # SLURM script: full pipeline
  run_hf_benchmark.sh          # SLURM script: HF model benchmark
  run_ollama_benchmark.sh      # SLURM script: Ollama benchmark
  requirements.txt             # Python dependencies
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/weed-llm-benchmark.git
cd weed-llm-benchmark
pip install -r requirements.txt
```

## Quick Start

### 1. Full Pipeline (download -> detect -> upload)

**Interactive mode** (on a GPU node):
```bash
python roboflow_bridge.py
```
The menu lets you: pick a Roboflow project, choose a model, run detection, and auto-upload results.

**Command line**:
```bash
python roboflow_bridge.py --all --project weed2okok --version 1 --model-key qwen7b
```

### 2. YOLO Auto-Labeling

Use a trained YOLO model to auto-label images and upload to Roboflow:
```bash
python weed_lable_update_download.py
```

### 3. Benchmark Models

Compare multiple vision LLMs on the same dataset:
```bash
# HuggingFace models
python test_hf_models.py --image images/weed1.jpg --model qwen7b
python test_hf_models.py --image-dir images/ --model all

# Ollama models
python test_ollama.py --image images/weed1.jpg --model all
```

### 4. YOLO + LLM Fusion

Merge YOLO detections with LLM detections:
```bash
python yolo_llm_fusion.py --image photo.jpg --yolo-results yolo_output.json --model qwen7b
```

Fusion strategy:
1. Keep all YOLO detections (precise bboxes from trained model)
2. LLM reviews same image independently
3. IoU matching merges overlapping detections
4. LLM-only detections flagged as "YOLO missed"
5. Disagreements flagged for human review

### 5. Visualize Results

```bash
python visualize_results.py --results results/hf_benchmark_*.json
```

## SLURM (Bridges-2 HPC)

Submit batch jobs on PSC Bridges-2:

```bash
# Full pipeline with different models
sbatch run_roboflow_bridge.sh weed2okok 1 qwen7b
sbatch run_roboflow_bridge.sh weed2okok 1 qwen3b

# Benchmark only
sbatch run_hf_benchmark.sh qwen7b
sbatch run_ollama_benchmark.sh all

# Monitor
squeue -u $USER
tail -30 slurm_roboflow_JOBID.out
```

Resource settings: `GPU-shared`, 1x V100-32GB, 5 CPUs, 40GB RAM, 4hr limit.

## Roboflow Integration

- Workspace: `mtsu-2h73y`
- API key is auto-prompted on first run and saved locally in `.roboflow_key`
- Get a key at: https://app.roboflow.com/settings/api
- Upload project is auto-named with model info (e.g., `weed2okok-qwen25-vl-7b`)

## Output Structure

```
llm_labeled/
  qwen25-vl-7b/                  # per-model output
    detected/
      images/                    # images with detections
      labels/                    # YOLO-format .txt labels
    no_detection/
      images/                    # images with no plants found
    visualized/                  # images with drawn bboxes
    detection_results.json       # full results with raw LLM responses
```
