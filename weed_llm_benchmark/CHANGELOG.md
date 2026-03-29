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

## 2026-03-16 - Phase 1 Complete, Phase 2 In Progress

### Phase 1 Complete
- YOLO11n fine-tuned on CottonWeedDet12 (100 epochs, V100-32GB, batch=60)
- Test results: mAP@0.5=0.929, mAP@0.5:0.95=0.865, P=0.930, R=0.850
- Model saved: `models/yolo11n_cottonweeddet12_best.pt` (5.5MB)

### Phase 2: LLM Benchmark Expanded (19 models)
- **Completed**: moondream(mAP=0.0), llava:7b/13b/bakllava (0 bounding boxes)
- **Running on cluster**: qwen7b, qwen3b, llama3.2-vision:11b, internvl2, florence2
- **7 new models added to benchmark** (coverage audit found gaps):
  - Qwen3-VL-8B — latest Qwen VL (Jan 2026), native grounding
  - Grounding DINO — #1 open-set detection model, essential baseline
  - PaliGemma2-3B — Google, native `<loc>` detection tokens
  - YOLO-World v2 — open-vocabulary YOLO, bridges YOLO and VLM
  - MiniCPM-V 4.5 — Feb 2026, replaces gated v2.6
  - Molmo-7B-D — Allen AI, precise pixel coordinate output
  - DeepSeek-VL2-Small — MoE with grounding tokens

### Bug Fixes
- Fixed `evaluate.py` mAP bug (was using Precision as mAP)
- Fixed DOWNLOAD_DIR path for cluster flat structure
- Fixed `query_ollama` return type (dict not tuple)
- Fixed Qwen OOM with pixel limits (min=256*28*28, max=1280*28*28)
- Fixed transformers 5.0 compat (3 rounds):
  - Qwen: removed `device_map`, use `.cuda()` (avoids 10hr weight materialization)
  - InternVL2: `all_tied_weights_keys = {}` (dict not set, callers use .keys())
  - Florence-2: `PretrainedConfig.forced_bos_token_id = None` (patch before config load)

### Modified
- `roboflow_bridge.py` — MODEL_REGISTRY expanded (5→12 models), 7 new inference functions
- `run_full_benchmark.py` — HF_MODELS expanded with 7 new entries
- `setup_and_train.sh` — batch=-1 (auto), workers=5

## 2026-03-17/18 - Phase 2 Complete, Phase 3 Fusion

### Phase 2 Complete (15 models evaluated)
- Florence-2-base (0.23B): mAP=0.434, best VLM — smallest model outperforms all 3-8B VLMs
- Fixed coordinate conversion: Qwen2.5-VL [0,1000] normalized → multi-scale detection
- Created `compat` env (transformers 4.46) for InternVL2/Florence-2 compatibility
- Revalidation run confirmed all 15 models with IoU spot checks

### Phase 3: YOLO+LLM Fusion (6 experiments)
- `run_phase3_fusion.py` — pairwise fusion, IoU sweep, complementarity, ensemble, bootstrap CI
- Only OWLv2 filter improves YOLO (+0.018 F1)
- LLM rescue rate <1%: LLMs cannot detect weeds that YOLO misses on known species

## 2026-03-19 - Phase 3B: Cross-Species Generalization

### Leave-4-Out Experiment
- `run_leave4out.py` — holds out 4 species, trains YOLO on 8, tests on unseen
- YOLO drops 27% on unseen species (F1: 0.830→0.606)
- Florence-2 precision exceeds YOLO on unseen (0.726 > 0.589)
- LLM pseudo-label augmentation: +0.9% new, -2.4% forgetting

## 2026-03-23/24 - Phase 3C: Anti-Forgetting Methods

### All simple methods failed
- `run_balpw.py` — background-aware label propagation: -0.022 (marginal)
- `run_antiforgetting.py` — replay 50%: -0.030 (worse); frozen: F1=0.155 (catastrophic)
- Root cause: LLM pseudo-label noise (27.4% FP), not training strategy

## 2026-03-25/28 - Phase 3D/3E: SAM + Agent Optimizer

### SAM-Enhanced Labeling (negative result)
- `run_sam_enhanced.py` — SAM segments → Florence-2 caption classification
- Result: WORST method (-6.8% old, -11% new) — SAM over-segments, caption keywords too noisy

### Autonomous Agent Optimizer — FIRST PRECISION IMPROVEMENT
- `run_agent_optimizer.py` — OPRO-inspired self-improving agent
- Tests 5 strategies automatically with multi-VLM consensus label generation
- **Best result: Florence+OWLv2 consensus** → unseen species F1: 0.606 → **0.622 (+0.016)**, forgetting only -0.020
- Key finding: 2-model consensus (high-precision + high-recall) beats 7-model voting
- Architecture: StrategyBrain → LabelGenerator → TrainManager → Evaluator → iterate

## TODO
- [ ] Refine agent optimizer (second round with parameter tuning)
- [ ] Implement R-Super soft constraint losses
- [ ] Fine-tune Florence-2 on weed domain data
- [ ] Generate paper figures and tables
- [ ] Write paper
