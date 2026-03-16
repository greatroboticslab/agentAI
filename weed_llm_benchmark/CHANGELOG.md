# Changelog

## 2026-03-10 - Initial Setup

### Created
- Full benchmark framework for testing vision LLMs on weed detection
- `roboflow_bridge.py` - main tool: download images from Roboflow -> LLM detect -> upload labeled results back
- `run_roboflow_bridge.sh` - SLURM batch script
- `config.py` - model configs, prompts, paths
- `test_hf_models.py` - HuggingFace model benchmark (Qwen, MiniCPM, Florence-2, InternVL2)
- `test_ollama.py` - Ollama model benchmark
- `quick_test.py` - quick single-image test
- `visualize_results.py` - draw bboxes, comparison charts
- `yolo_llm_fusion.py` - merge YOLO + LLM detections via IoU

### Model Research
- Evaluated 12+ open-source vision LLMs for weed detection suitability
- **Qwen2.5-VL-7B** selected as #1: only model with native bbox JSON output on ollama + HuggingFace
- Ranked models by grounding capability, JSON output quality, and availability

### First Successful Run
- Ran Qwen2.5-VL-7B on 106 weed images from `weed2okok` project
- Results uploaded to Roboflow as `weed2okok-llm-labeled`
- Pipeline: download -> detect -> upload completed in ~10 minutes

### Bug Fixes During Setup
- Fixed SLURM `cpus-per-task` from 8 to 5 (Bridges-2 max is 5/GPU)
- Fixed conda activation in SLURM: use `eval "$(conda shell.bash hook)"` instead of `source activate`
- Fixed `accelerate` version: upgraded 0.34.2 -> 1.13.0 (required by transformers 5.0.0.dev0)
- Fixed Roboflow project ID stripping workspace prefix (`mtsu-2h73y/weed2okok` -> `weed2okok`)
- Fixed `qwen` conda env broken pip shebang (was pointing to blip3o env)
- Roboflow API key: old key revoked, new key saved in `.roboflow_key`

### Multi-Model Support Added
- Added model selection menu in interactive mode (qwen7b, qwen3b, minicpm, internvl2)
- Upload project auto-named with model: e.g. `weed2okok-qwen25-vl-7b`
- SLURM script accepts model key: `sbatch run_roboflow_bridge.sh weed2okok 1 qwen3b`
- Each model's output saved in separate directory under `llm_labeled/`

## 2026-03-15 - Evaluation & Paper Infrastructure

### Created
- `evaluate.py` — compute mAP@0.5, mAP@0.5:0.95, mAP@0.25, precision, recall, F1
  - IoU-based greedy matching, class normalization, binary/multi-class modes
  - Loads YOLO format labels and benchmark JSON predictions
- `datasets.py` — dataset registry with download helpers
  - Registered: CottonWeedDet12, DeepWeeds, weed2okok, CropWeed (fallback)
  - Tracks download status, split info, class names
- `run_yolo_baseline.py` — YOLO11n baseline runner
  - Zero-shot and fine-tuned modes, same output format as LLM pipeline
- `run_yolo_baseline.sh` — SLURM script for YOLO baseline
- `run_full_benchmark.py` — orchestrator for datasets × models matrix
  - Resume support, checkpoint saving, result aggregation
- `run_ablations.py` — ablation study experiments
  - Prompt engineering (3 prompts), model size (7B vs 3B), grounding capability, fusion IoU sweep
- `generate_paper_figures.py` — publication-quality matplotlib figures (6 figure types)
- `generate_tables.py` — LaTeX table generation (6 table types)
- `RESEARCH_LOG.md` — daily research progress tracking

### Modified
- `roboflow_bridge.py` — added `--evaluate` flag to auto-run evaluation after detection
- `yolo_llm_fusion.py` — added `fuse_dataset()` batch mode with 3 fusion strategies
  - supplement (add LLM-only detections), filter (confirm YOLO with LLM), weighted (combine confidence)
- `README.md` — updated with evaluation, datasets, paper, and new file listing

## TODO
- [ ] Download CottonWeedDet12 and DeepWeeds datasets
- [ ] Run YOLO11n zero-shot baseline on all 3 datasets
- [ ] Run all LLM models on all datasets
- [ ] Test YOLO+LLM fusion strategies
- [ ] Run ablation studies
- [ ] Generate paper figures and tables
- [ ] Write paper
