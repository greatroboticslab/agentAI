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

### 4. Visualize Results

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

## Roadmap

- [ ] Add quantitative evaluation: compare LLM detections against ground truth labels (precision, recall, mAP)
- [ ] YOLO + LLM fusion pipeline: use LLM as a second opinion to supplement YOLO detections (`yolo_llm_fusion.py` — module ready, not yet integrated into main pipeline)
- [ ] Fine-tune best-performing model on labeled weed data
- [ ] Add Grounding DINO as a non-LLM baseline detector
