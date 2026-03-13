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

## TODO
- [ ] Add quantitative evaluation against ground truth (precision, recall, mAP)
- [ ] Test qwen3b, minicpm, internvl2 on same dataset and compare
- [ ] Integrate YOLO + LLM fusion into main pipeline
- [ ] Add Grounding DINO as non-LLM baseline detector
- [ ] Fine-tune best model on labeled weed data
