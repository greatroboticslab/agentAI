# Weed Detection LLM Benchmark

Use open-source vision LLMs to detect weeds in agricultural images, compare models, and complement YOLO11n.

## Project Goal

Use large vision models to **correct and supplement** YOLO11n weed detection:
- YOLO is fast but may miss weeds or misclassify
- LLMs provide a "second opinion" with species identification
- Fusion logic merges both results via IoU matching

## Location

```
/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/
```

## Supported Models

| Key | Model | Size | Grounding? | Status |
|-----|-------|------|-----------|--------|
| `qwen7b` | Qwen2.5-VL-7B-Instruct | ~14GB | Native bbox | Downloaded |
| `qwen3b` | Qwen2.5-VL-3B-Instruct | ~6GB | Native bbox | Downloaded |
| `minicpm` | MiniCPM-V-2.6 | ~16GB | No | Need download |
| `internvl2` | InternVL2-8B | ~16GB | Partial | Need download |

## Files

| File | Purpose |
|------|---------|
| `roboflow_bridge.py` | **Main tool** - Download from Roboflow, run LLM detection, upload results back |
| `run_roboflow_bridge.sh` | SLURM batch script for `roboflow_bridge.py` |
| `config.py` | Model definitions, prompts, paths |
| `test_hf_models.py` | Benchmark multiple HuggingFace models |
| `test_ollama.py` | Benchmark models via Ollama |
| `quick_test.py` | Quick single-image test with Qwen |
| `visualize_results.py` | Draw bboxes on images, comparison charts |
| `yolo_llm_fusion.py` | Merge YOLO + LLM detections via IoU |
| `weed_lable_update_download.py` | YOLO auto-labeling + Roboflow upload tool |

## Quick Start

### Interactive mode (need GPU node)
```bash
salloc --partition=GPU-shared --gpus=v100-32:1 --cpus-per-task=5 --mem=40G --time=02:00:00
conda activate qwen
cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
python roboflow_bridge.py
```
Menu lets you: pick Roboflow project, choose model, auto-upload results.

### SLURM batch mode
```bash
cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark

# Test with different models (submit multiple in parallel)
sbatch run_roboflow_bridge.sh weed2okok 1 qwen7b
sbatch run_roboflow_bridge.sh weed2okok 1 qwen3b
sbatch run_roboflow_bridge.sh weed2okok 1 minicpm

# Monitor
squeue -u byler
tail -30 slurm_roboflow_JOBID.out
```

Upload project auto-named with model: e.g. `weed2okok-qwen25-vl-7b`

### Quick single-image test
```bash
# On GPU node
conda activate qwen
python quick_test.py --image images/your_photo.jpg --model qwen7b
```

## Roboflow Setup

- Workspace: `mtsu-2h73y`
- API key stored in: `.roboflow_key` (auto-prompted on first run)
- Get new key at: https://app.roboflow.com/settings/api

## Conda Environment

Primary env: `qwen` (Python 3.10)
- transformers 5.0.0.dev0
- torch 2.5.1 + CUDA
- accelerate 1.13.0
- roboflow, opencv-python-headless

Fix broken pip: `python -m pip install --force-reinstall pip`

## SLURM Settings (Bridges-2)

```
--partition=GPU-shared
--gpus=v100-32:1
--cpus-per-task=5      # max 5 per GPU on Bridges-2
--mem=40G
--time=04:00:00
```

## Output Structure

```
llm_labeled/
  qwen25-vl-7b/           # per-model output
    detected/
      images/              # images with detections
      labels/              # YOLO format .txt labels
    no_detection/
      images/
    visualized/            # images with drawn bboxes
    detection_results.json # full results with raw LLM responses
```

## YOLO + LLM Fusion

```bash
python yolo_llm_fusion.py --image photo.jpg --yolo-results yolo_output.json --model qwen7b
```

Fusion strategy:
1. Keep all YOLO detections (precise bboxes)
2. LLM reviews same image, finds additional weeds
3. IoU matching merges overlapping detections
4. LLM-only detections flagged as "YOLO missed"
5. Disagreements flagged for review
