# Weed Detection LLM Benchmark

A benchmark framework for evaluating how well open-source vision LLMs can detect and identify weeds in agricultural images. Tests multiple models on the same dataset and compares their detection capabilities.

## Overview

```
  +------------------+
  | Roboflow Dataset |    Labeled weed images (ground truth)
  +--------+---------+
           |
       Download
           |
  +--------v---------+
  | Weed Images      |
  +--------+---------+
           |
           |   Run each model on the same images
           |
     +-----+-----+-----+-----+-----+---- ...
     |           |           |           |
  +--v---+  +---v--+  +---v---+  +--v-------+
  |Qwen  |  |Qwen  |  |MiniCPM|  | Any new  |
  |VL-7B |  |VL-3B |  |  -V   |  | model... |
  +--+---+  +---+--+  +---+---+  +--+-------+
     |           |           |           |
     +-----+-----+-----+-----+-----+----+
           |
  +--------v---------+
  | Compare Results   |    JSON valid? BBox? Weed count?
  | per Model         |    Inference time? Species ID?
  +--------+---------+
           |
  +--------v---------+
  | Visualize +       |    Side-by-side charts, bbox overlays
  | Upload to         |    Results back to Roboflow
  | Roboflow          |
  +-------------------+
```

**Goal**: Determine which vision LLMs are most effective at weed detection before deploying them in a real agricultural pipeline. The framework tests each model's ability to return valid bounding boxes, identify weed species, and produce reliable JSON output.

## Benchmark Results

Evaluated on **CottonWeedDet12** test set (848 images, 12 weed species, 1,464 ground truth boxes).

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Inference Time |
|-------|---------|--------------|-----------|--------|-----|---------------|
| **YOLO11n** (fine-tuned) | **0.929** | **0.865** | 0.930 | 0.850 | 0.888 | ~2s/848img |
| Florence-2-base (0.23B) | **0.434** | **0.392** | **0.789** | 0.519 | 0.626 | 558s |
| Florence-2-large (0.77B) | 0.329 | 0.302 | 0.692 | 0.431 | 0.531 | 662s |
| InternVL2-8B (16GB) | 0.208 | 0.091 | 0.545 | 0.354 | 0.429 | 3838s |
| Qwen2.5-VL-3B (6GB) | 0.196 | 0.068 | 0.333 | 0.249 | 0.285 | 5898s |
| MiniCPM-V-4.5 (16GB) | 0.192 | 0.043 | 0.407 | 0.340 | 0.371 | 6595s |
| OWLv2-large (0.4B) | 0.184 | 0.117 | 0.194 | **0.943** | 0.322 | 2519s |
| Qwen2.5-VL-7B (14GB) | 0.176 | 0.059 | 0.334 | 0.214 | 0.261 | 6047s |
| InternVL2-2B (4GB) | 0.002 | 0.001 | 0.038 | 0.025 | 0.031 | 2094s |
| InternVL2.5-8B (16GB) | 0.000 | 0.000 | 0.016 | 0.001 | 0.001 | 6238s |
| Grounding-DINO-base (1GB) | 0.000 | 0.000 | — | — | — | 843s |
| Llama 3.2 Vision 11B | 0.000 | 0.000 | 0.005 | 0.007 | 0.006 | 11370s |
| Moondream / Molmo / LLaVA | 0.000 | 0.000 | — | — | — | — |

**Key findings:**
- **YOLO dominates**: Fine-tuned YOLO11n achieves 0.929 mAP@0.5, 2.1x better than the best VLM.
- **Smaller Florence-2 wins**: Florence-2-base (0.23B params) achieves the highest VLM mAP (0.434) — outperforming the 3x larger Florence-2-large (0.329). Its native `<OD>` task with the base model produces more precise bounding boxes for agricultural objects.
- **Model size does not predict performance**: The 0.23B Florence-2-base beats all 3B-8B VLMs. OWLv2 (0.4B) achieves near-perfect recall (0.967). Conversely, InternVL2.5-8B (improved version) underperforms InternVL2-8B.
- **Detection architecture matters more than scale**: Dedicated detection models (Florence-2, OWLv2) outperform general VLMs prompted for bbox coordinates (Qwen, MiniCPM).
- **OWLv2 has extreme recall**: 96.7% recall but only 9% precision — detects nearly everything but with many false positives. Useful as a high-sensitivity pre-filter.
- **Native grounding is essential**: Models without built-in bbox support (LLaVA, Llama Vision, Molmo) produce near-zero mAP regardless of model size.
- **Coordinate format matters**: Qwen2.5-VL outputs in [0, 1000] normalized space; Florence-2 outputs absolute pixels. Proper coordinate conversion is critical.

## Model Support

The framework is **model-agnostic** — the core pipeline (prompt -> JSON response -> extract bboxes -> YOLO format -> upload) works with any vision LLM. Two integration paths are provided:

- **Ollama**: Plug-and-play. Any vision model served by Ollama works out of the box — just specify the model name (e.g., `--model llava-next:13b`). No code changes needed.
- **HuggingFace Transformers**: Requires a lightweight adapter (~20-30 lines) per model family for loading and inference. The rest of the pipeline is shared.

### Pre-configured Models

| Key | Model | Size | Bounding Box Support | Backend |
|-----|-------|------|---------------------|---------|
| `qwen7b` | Qwen2.5-VL-7B-Instruct | ~14GB | Native bbox output | HuggingFace |
| `qwen3b` | Qwen2.5-VL-3B-Instruct | ~6GB | Native bbox output | HuggingFace |
| `minicpm` | MiniCPM-V-2.6 | ~16GB | Text-based | HuggingFace |
| `internvl2` | InternVL2-8B | ~16GB | Partial | HuggingFace |
| `florence2` | Florence-2-large | ~1.5GB | Native grounding | HuggingFace |
| `qwen2.5vl:7b` | Qwen2.5-VL-7B | ~6GB | Native bbox | Ollama |
| `qwen2.5vl:3b` | Qwen2.5-VL-3B | ~3.2GB | Native bbox | Ollama |
| `moondream` | Moondream2 1.8B | ~1.7GB | detect() API | Ollama |
| `llama3.2-vision:11b` | Llama 3.2 Vision 11B | ~7GB | Text-based | Ollama |
| `minicpm-v` | MiniCPM-V | ~5GB | Text-based | Ollama |
| `llava:13b` | LLaVA 1.5 13B | ~8GB | Text-based | Ollama |
| `llava:34b` | LLaVA 1.6 34B | ~20GB | Text-based | Ollama |

### Adding a New Model

**Ollama** — no code changes, just run:
```bash
python test_ollama.py --image images/weed1.jpg --model <any-ollama-vision-model>
```

**HuggingFace** — add a loader and inferencer in `test_hf_models.py`:
```python
# 1. Write a load/infer function pair
def load_your_model(model_name):
    ...
    return model, processor

def infer_your_model(model, processor, image_path, prompt):
    ...
    return response_text

# 2. Register it
MODEL_SHORTCUTS["your_model"] = {
    "full_name": "org/YourModel",
    "loader": load_your_model,
    "inferencer": infer_your_model,
}
```

## Project Structure

```
weed_llm_benchmark/
  roboflow_bridge.py           # Main pipeline: Roboflow download -> LLM detect -> upload
  evaluate.py                  # Evaluation: mAP, precision, recall, F1 vs ground truth
  datasets.py                  # Dataset registry, download helpers, metadata
  run_yolo_baseline.py         # YOLO11n baseline (zero-shot + fine-tuned)
  run_full_benchmark.py        # Orchestrator: all models x all datasets
  run_ablations.py             # Ablation studies (prompt, size, grounding, fusion IoU)
  yolo_llm_fusion.py           # YOLO + LLM fusion (single image + batch mode)
  test_hf_models.py            # Benchmark HuggingFace vision models
  test_ollama.py               # Benchmark Ollama-served models
  generate_paper_figures.py    # Publication-quality figures (matplotlib)
  generate_tables.py           # LaTeX table generation
  visualize_results.py         # Draw bboxes on images, comparison charts
  config.py                    # Model definitions, prompts, shared settings
  quick_test.py                # Quick single-image test
  weed_lable_update_download.py # YOLO auto-labeling + Roboflow upload
  run_roboflow_bridge.sh       # SLURM script: full pipeline
  run_hf_benchmark.sh          # SLURM script: HF model benchmark
  run_ollama_benchmark.sh      # SLURM script: Ollama benchmark
  run_yolo_baseline.sh         # SLURM script: YOLO baseline
  requirements.txt             # Python dependencies
  CHANGELOG.md                 # Development history
```

## Installation

```bash
git clone https://github.com/greatroboticslab/agentAI.git
cd agentAI/weed_llm_benchmark
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

### 4. Evaluate Detections Against Ground Truth

```bash
# Evaluate LLM predictions against ground truth
python evaluate.py --pred-dir llm_labeled/qwen25-vl-7b/detected/labels \
                   --gt-dir downloads/weed2okok/test/labels

# Evaluate from benchmark JSON
python evaluate.py --pred-json results/hf_benchmark_*.json \
                   --gt-dir downloads/weed2okok/test/labels

# Or use --evaluate flag in the pipeline
python roboflow_bridge.py --all --project weed2okok --version 1 --model-key qwen7b --evaluate
```

### 5. Run YOLO Baseline

```bash
# Zero-shot (pretrained, no weed training)
python run_yolo_baseline.py --dataset weed2okok --mode zero-shot --evaluate

# Fine-tuned on dataset
python run_yolo_baseline.py --dataset cottonweeddet12 --mode fine-tune --epochs 50 --evaluate
```

### 6. Full Benchmark (all models x all datasets)

```bash
# Run everything
python run_full_benchmark.py --all

# Resume if interrupted
python run_full_benchmark.py --all --resume

# Just aggregate existing results
python run_full_benchmark.py --aggregate
```

### 7. YOLO+LLM Fusion (batch mode)

```bash
python yolo_llm_fusion.py --batch \
    --yolo-dir results/yolo_zero_shot_weed2okok.json \
    --llm-dir llm_labeled/qwen7b_weed2okok/detection_results.json \
    --output-dir results/fusion_weed2okok \
    --strategy supplement
```

### 8. Ablation Studies

```bash
python run_ablations.py --all --dataset weed2okok
python run_ablations.py --experiment prompt --dataset weed2okok
```

### 9. Generate Paper Figures and Tables

```bash
python generate_paper_figures.py --all
python generate_tables.py --all
```

### 10. Visualize Results

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

## Datasets

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| CottonWeedDet12 | ~5,648 | 12 weed species | Roboflow |
| DeepWeeds | ~17,509 | 8 species + negative | Roboflow |
| weed2okok | 106 | 1 (weed) | Lab data |

```bash
python datasets.py --list          # Show all datasets
python datasets.py --download all  # Download all
python datasets.py --info weed2okok
```

## Research Paper

**Title**: "Can Vision LLMs Detect Weeds? A Benchmark of Open-Source Multimodal Models for Agricultural Object Detection"

**Research Questions**:
- RQ1: How do open-source vision LLMs compare to YOLO11n in weed detection (mAP@0.5:0.95)?
- RQ2: Can YOLO+LLM fusion improve detection beyond either method alone?
- RQ3: How do model size, prompt design, and native grounding capability affect detection quality?

See `RESEARCH_LOG.md` in project root for daily progress.

## Roadmap

- [x] Quantitative evaluation: mAP, precision, recall, F1 vs ground truth (`evaluate.py`)
- [x] Dataset registry and download management (`datasets.py`)
- [x] YOLO baseline: zero-shot and fine-tuned (`run_yolo_baseline.py`)
- [x] Full benchmark orchestrator (`run_full_benchmark.py`)
- [x] YOLO+LLM fusion with batch mode and 3 strategies (`yolo_llm_fusion.py`)
- [x] Ablation studies: prompt, size, grounding, fusion IoU (`run_ablations.py`)
- [x] Paper figure and table generation (`generate_paper_figures.py`, `generate_tables.py`)
- [ ] Run full benchmark on all 3 datasets
- [ ] Fine-tune best-performing model on labeled weed data
- [ ] Write paper
