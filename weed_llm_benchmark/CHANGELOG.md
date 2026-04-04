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

## 2026-03-29 - Florence-2 Fine-tuning (negative) + Full mAP Evaluation

### Florence-2 Fine-tuning (negative result)
- `run_finetune_florence.py` — fine-tune Florence-2-base on 8 species, generate pseudo-labels, train YOLO
- Result: WORSE than zero-shot (-11.3% mAP on old, -7.3% mAP on new)
- Cause: autoregressive OD training format too complex for simple fine-tuning

### Full mAP@0.5:0.95 Evaluation
- First comprehensive evaluation with mAP at all IoU thresholds (0.5 to 0.95)
- Agent consensus remains best: new F1 +2.6%, old F1 -2.0%
- mAP50-95 shows tighter bbox evaluation: all methods lose more at strict IoU

## 2026-03-29/30 - HyperAgent Closed-Loop System

### HyperAgent with Qwen2.5-7B Brain
- `run_hyperagent.py` — real LLM-brained closed-loop optimization
- Qwen2.5-7B-Instruct as Brain: analyzes history → reasons in natural language → proposes JSON strategy
- GPU memory management: alternately loads Brain (14GB) and YOLO (5.5GB)
- 3 rounds executed: all Qwen-proposed strategies caused forgetting
- System architecture works (modular, swappable Brain) but Qwen-7B reasoning insufficient
- Brain proposed: freeze layers, more votes, more replay — all already proven ineffective
- Seed strategy (Florence+OWLv2 2-vote consensus, +0.016 F1) remains best

### Key finding
- HyperAgent bottleneck is Brain intelligence, not system design
- Stronger reasoning models (DeepSeek-R1, Qwen-72B) may discover novel strategies
- Architecture is future-proof: swap Brain model to benefit from LLM improvements

## 2026-03-31 - Framework Refactor: WeedOptimizer (agent-style architecture)

### CURRENT TASK — Read this section to resume work

**Goal**: Refactor single-file `weed_optimizer_framework.py` (698 lines) into a proper
multi-module framework using agent-style architecture (while loop + tool calling).

**Architecture** (Brain + Tools + Memory pattern):
```
weed_optimizer_framework/          # Python package
├── __init__.py                    # Package init + version
├── config.py                      # All paths, constants, model registry, cluster config
├── brain.py                       # SuperBrain — swappable LLM (Qwen→DeepSeek→future)
├── memory.py                      # Persistent memory (experiments, 10 hard lessons, baselines)
├── monitor.py                     # Quality monitor (forgetting, drift, per-class, mAP tracking)
├── tools/
│   ├── __init__.py                # ToolRegistry base class + dispatch
│   ├── vlm_pool.py                # VLM model loading + inference (READ-ONLY, never fine-tuned)
│   ├── yolo_trainer.py            # YOLO training with replay buffer management
│   ├── evaluator.py               # Full eval: mAP@0.5, mAP@0.5:0.95, per-class P/R/F1
│   └── label_gen.py               # Multi-VLM consensus label generation
├── orchestrator.py                # Main while loop (Brain→Tools→Evaluate→Brain)
└── run.py                         # CLI entry point with argparse
```

**Core design principle** (agent tool-calling loop):
```
while not converged:
    strategy = brain.analyze_and_propose(memory)     # Brain thinks
    if not monitor.validate(strategy): adjust()      # Safety check
    labels = tools.call("generate_labels", strategy)  # Tool execution
    model = tools.call("train_yolo", labels)          # Tool execution
    result = tools.call("evaluate", model)            # Tool execution
    memory.record(strategy, result)                   # Persist
    brain.reflect(result)                             # Brain learns
```

**Key rules**:
- ONLY YOLO gets fine-tuned. All VLMs are read-only tools.
- Old species F1 must stay ≥0.90 (forgetting threshold)
- 10 hard-coded lessons from 18 sessions prevent repeating known failures
- Brain is swappable: currently Qwen2.5-7B, future DeepSeek-R1 or Qwen-72B
- Full mAP@0.5:0.95 evaluation required (not just F1)
- Atomic file writes (.tmp → os.replace) for checkpoint safety
- GPU memory: alternate Brain (14GB) and YOLO (5.5GB), never simultaneous

**Build progress** (2,319 lines across 12 files):
- [x] `config.py` (169 lines) — paths, constants, VLM registry (7 models), Brain registry (3 models)
- [x] `memory.py` (271 lines) — persistent memory, 10 hard lessons, experiment history, Brain context generation
- [x] `monitor.py` (198 lines) — strategy validation, forgetting detection, per-class analysis, drift detection
- [x] `tools/__init__.py` (91 lines) — ToolRegistry with timing, stats, GPU-awareness
- [x] `tools/vlm_pool.py` (153 lines) — VLM metadata, label access, pair recommendation, agreement analysis
- [x] `tools/yolo_trainer.py` (195 lines) — dataset assembly, replay buffer, YOLO training, cleanup
- [x] `tools/evaluator.py` (311 lines) — FULL mAP@0.5 + mAP@0.5:0.95, per-class P/R/F1, PASCAL VOC AP
- [x] `tools/label_gen.py` (187 lines) — multi-VLM consensus with IoU clustering, box validation
- [x] `brain.py` (279 lines) — SuperBrain: analyze, propose, reflect, diagnose; chat template; JSON parsing
- [x] `orchestrator.py` (343 lines) — main while loop, 6-step round, baseline, seeding, run log
- [x] `run.py` (106 lines) — CLI with argparse, logging setup, --list-brains/--list-vlms
- [x] Syntax verify all files (12/12 passed)
- [x] Upload to cluster + verify imports (ALL_IMPORTS_OK on login node)
- [x] Submit test run on cluster (Job 38326705, Qwen2.5-7B Brain, 3 rounds)
- [x] Check job results — COMPLETED (2h44m), framework ran successfully
- [x] Record results and update docs

**10 Hard Lessons (NEVER violate)**:
1. NEVER freeze backbone (F1=0.155 catastrophic)
2. Replay >50% makes forgetting WORSE
3. SAM + caption classification too noisy (-11%)
4. Fine-tuning VLMs degrades zero-shot ability (-11.3%)
5. 2 complementary models > 7 mediocre models voting
6. Florence-2 + OWLv2 is the best VLM pair
7. Florence-2 confidence scores are NOT calibrated
8. Old species F1 must stay ≥0.90
9. Label noise (27.4% FP) is the ROOT CAUSE of failures
10. YOLO drops 27% F1 on unseen species

**Best result so far**: Florence+OWLv2 2-vote consensus → new F1: 0.606→0.622 (+2.6%), old F1: 0.917→0.897 (-2.0%)

**Cluster info**: Bridges-2 (PSC), conda env `bench` (transformers 4.57) or `compat` (4.46), V100-32GB

### v1.1 Upgrade: Agent mode + VLM live inference (2,318→2,682 lines)
- `brain.py`: Agent mode — Brain decides ONE action per step, sees result, decides next
- `tools/vlm_pool.py`: Live inference for Florence-2 + OWLv2 (load→detect→return boxes)
- `orchestrator.py`: Two modes — `agent` (Brain controls flow) and `strategy` (rigid pipeline)
- `run.py`: Added `--mode agent|strategy` flag

Agent mode actions: inspect_labels → run_vlm_inference → generate_consensus → train_yolo → evaluate → done

**Test results**:
- Job 38354715 (v1): FAILED — Qwen-7B couldn't output JSON, 30x fallback loop
- Job 38373824 (v2, simplified prompt): FAILED — Qwen-7B outputs "1" repeatedly, 20x inspect loop
- Root cause: Qwen-7B can output format but makes terrible decisions (loops on inspect, never trains)

### v1.2 Upgrade: Ollama function calling + forced progression + job chain (2,868 lines)
- `brain.py` (436 lines): Three backends — Ollama (native tool calling), HuggingFace, fallback pipeline
- Ollama function calling: model outputs structured tool calls natively, no JSON parsing needed
- Forced progression: if Brain repeats same action 2+ times, auto-advance to next step
- Job chain: SLURM script auto-submits next job if framework hasn't converged
- `run_framework_ollama.sh`: starts Ollama server, pulls model, runs framework, auto-chains
- `--backend ollama|hf|fallback|auto` flag added to CLI

**Test results** (Jobs 38381066 + 38390009, total 4h56m):
- Ollama function calling WORKS — native tool calls, Brain made real decisions
- Brain tried: different VLM combos (flo+owl, flo+large, flo+owl+intern), min_votes 2/3
- Job chain auto-submitted 2nd job, auto-stopped when no improvement
- Memory persisted across jobs: 5 experiments + 4 lessons

| Iter | Old F1 | New F1 | Forgetting? |
|------|--------|--------|-------------|
| 0 seed | 0.897 | **0.622** | No |
| 1 agent | 0.893 | 0.624 | Yes |
| 2 agent | 0.883 | 0.617 | Yes |
| 3 chain | 0.886 | 0.595 | Yes |
| 4 chain | 0.895 | 0.583 | Yes |

Architecture validated. Precision bottleneck: label noise (27.4% FP), not framework

### v1.3: Web tools — plant.id API + HuggingFace model discovery (3,522 lines)
- `tools/web_identifier.py` (222 lines): plant.id API for expert weed species identification
  - Base64 image → species name + confidence + is_weed classification
  - 40+ known weed genera for automatic weed/crop classification
  - Free tier: 10/month; paid: unlimited
- `tools/model_discovery.py` (338 lines): HuggingFace model search + download + inference
  - Pre-researched: DETR-weed, Deformable-DETR-weed, YOLOv8s-weed
  - Live HuggingFace Hub search for new models
  - Download → load → batch inference → save as YOLO labels
- Brain now has 9 actions (was 6): +identify_weed, +search_models, +run_external_model
- Orchestrator integrates all new tools into agent loop

**Professor's direction**: Agent should visit professional sites (plant.id) and discover
GitHub/HuggingFace models to autonomously expand its capabilities

**Test results** (Jobs 38403497 + 38407270, ~5h total):
- Brain **autonomously called `run_external_model(detr_weed)`** — downloaded DETR from HuggingFace!
- Brain called `run_vlm_inference(owlv2)` for live inference
- Job chain: auto-submitted 2nd job, auto-stopped when no improvement
- 5 experiments total, all caused forgetting (label noise root cause)
- Framework capabilities: FULLY VALIDATED including external model discovery

### Framework test results (Job 38326705, strategy mode)
- Framework ran 2 rounds (auto-stopped after 2 no-improve rounds)
- Round 1: new_f1=0.624 (slight improvement) but old_f1=0.893 → FORGETTING
- Round 2: new_f1=0.617, old_f1=0.883 → FORGETTING
- Seed (Phase 3E consensus) remains best: new_f1=0.622 with old_f1=0.897
- First full mAP baseline: old_mAP50=0.953, new_mAP50=0.525

### HOW TO RESUME
When user says "阅读changelog然后继续":
1. Read this CHANGELOG.md
2. Check "CURRENT TASK" section for next steps
3. Framework is complete and tested — move to TODO items below
4. Always update this file after completing each task

## 2026-04-03 - Clone-and-Train + plant.id setup

### Professor's Two Tasks status
1. **plant.id API** — code done (`tools/web_identifier.py`), needs API key from https://admin.kindwise.com/signup (free, 100 calls, no credit card)
2. **Clone + Train** — `run_clone_and_train.py` submitted (Job 38416097):
   - Git clone DeepWeeds from GitHub
   - Download DETR weed detector from HuggingFace, evaluate zero-shot
   - Train YOLOv8s on CottonWeedDet12 from COCO pretrained (100 epochs)
   - Compare against our YOLO11n baseline

### Files added
- `run_clone_and_train.py` — full pipeline: clone → download → train → evaluate → compare
- `run_clone_and_train.sh` — SLURM script for cluster

### Clone + Train results (Job 38416097, 1h53m)
- Git cloned DeepWeeds from GitHub ✅
- Downloaded DETR-ResNet50 from HuggingFace → zero-shot F1=0.000 (class mismatch)
- **Trained YOLOv8s on CottonWeedDet12 (100 epochs) → F1=0.888** ✅
- Compared with our YOLO11n baseline → F1=0.917

| Model | Source | Precision | Recall | F1 |
|-------|--------|-----------|--------|-----|
| YOLO11n (ours) | Fine-tuned | **0.881** | 0.957 | **0.917** |
| YOLOv8s (clone+train) | COCO→CottonWeed | 0.820 | **0.970** | 0.888 |
| DETR (HuggingFace) | Zero-shot | 0.000 | 0.000 | 0.000 |

**Professor's Task 2: DONE** — cloned, trained, evaluated, compared.

## 2026-04-04 - Push toward ideal autonomous system

### Three upgrades toward fully autonomous framework
1. **DeepSeek-R1:7b as Brain** — stronger reasoning model, same VRAM as Qwen-7B
   - `run_framework_ollama.sh` now accepts model as argument: `sbatch run_framework_ollama.sh deepseek-r1:7b`
   - Testing: Job 38432901
2. **Pre-cache layer** for plant.id API (cluster network workaround)
   - `precache.py`: run locally with internet → saves API results to JSON cache
   - `web_identifier.py`: checks cache first, then API, then local fallback
   - plant.id API key configured (49 credits remaining)
3. **3+ model consensus** to reduce label noise
   - `label_gen.py`: now loads external model detections (DETR, YOLOv8s) alongside VLM labels
   - Auto-discovers `ext_*` directories from previous `run_external_model` calls
   - More diverse model families → lower false positive rate (target: <20% FP vs current 27.4%)

### Files modified
- `run_framework_ollama.sh` — parameterized Brain model (`$1`, default deepseek-r1:7b)
- `tools/label_gen.py` — added `extra_label_dirs` support, auto-discover ext_ dirs
- `tools/web_identifier.py` — added cache-first lookup from `api_cache.json`
- `precache.py` — NEW: pre-cache plant.id + HuggingFace search results

## TODO
- [ ] Check Job 38432901 (DeepSeek-R1 Brain) results
- [ ] Run precache.py locally with real weed images for plant.id
- [ ] Few-Shot Grounding DINO adaptation (CVPR 2025)
- [ ] Generate paper figures and tables
- [ ] Write paper
