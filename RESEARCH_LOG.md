# Research Log

**Paper**: "Can Vision LLMs Detect Weeds? A Benchmark of Open-Source Multimodal Models for Agricultural Object Detection"

**Focus Dataset**: CottonWeedDet12 (5648 images, 12 weed species)

---

## Phase 0: Evaluation Module

### 2026-03-15 (Session 1)
**Goal**: Build evaluation code (mAP, precision, recall) and dataset management
**Done**:
- Created `evaluate.py`: mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- Created `datasets.py`: dataset registry with download helpers (4 datasets)
- Created `run_yolo_baseline.py`: YOLO baseline runner (zero-shot / fine-tuned / custom weights)
- Created `run_full_benchmark.py`: orchestrator for datasets x models matrix with resume
- Created `run_ablations.py`: ablation study experiments
- Created `generate_paper_figures.py`: publication-quality matplotlib figures
- Created `generate_tables.py`: LaTeX table generation
- Created `convert_coco_to_yolo.py`: COCO/LabelMe/VOC to YOLO format converter
- Updated `yolo_llm_fusion.py`: added batch mode `fuse_dataset()` with 3 strategies
- Updated `roboflow_bridge.py`: added `--evaluate` flag
- Updated README.md and CHANGELOG.md

### 2026-03-15 (Session 2)
**Goal**: Download CottonWeedDet12, fix evaluation bugs, set up YOLO fine-tuning
**Done**:
- Downloaded CottonWeedDet12 from Zenodo (28GB 7z) locally and on cluster
- Created train/valid/test splits (65/20/15) with seed=42 from weedImages/ + annotation_YOLO_txt/
- Split weed2okok into train(70)/valid(21)/test(15)
- **Fixed critical mAP bug**: evaluate.py was using Precision as mAP (line 367). Rewrote `evaluate_dataset()` with proper PR-curve based AP computation using `_compute_ap_at_iou()`.
- **Fixed DOWNLOAD_DIR path**: datasets.py, run_yolo_baseline.py, run_full_benchmark.py all pointed to wrong directory. Fixed to use `PROJECT_ROOT/downloads/`.
- **Discovered yolo11nweed.pt origin**: trained on `weeddataset311` (Windows), NOT on weed2okok or CottonWeedDet12. Cross-dataset transfer performance is low (mAP@0.5=0.316 on CottonWeedDet12).
- Created SLURM scripts for cluster (Bridges-2):
  - `run_benchmark_hf.sh`: HF model benchmark per dataset
  - `run_benchmark_ollama.sh`: Ollama model benchmark per dataset
  - `submit_all_jobs.sh`: master job submission script
  - `setup_and_train.sh`: all-in-one download + split + fine-tune YOLO
  - `run_finetune_cottonweed.sh`: fine-tune only
- Fixed `run_yolo_baseline.sh` cluster path (was local Mac path)
- Submitted YOLO fine-tune job on Bridges-2 V100 (Job 38007424, 100 epochs)

**Preliminary Results (weed2okok test=15 images)**:
| Method | mAP@0.5 | Precision | Recall |
|--------|---------|-----------|--------|
| YOLO11n zero-shot | 0.000 | 0.000 | 0.000 |
| yolo11nweed.pt (cross-dataset) | 0.072 | 0.172 | 0.167 |
| Qwen2.5-VL-7B (zero-shot LLM) | 0.639* | 0.639* | 0.411* |
| YOLO11n fine-tuned 20ep | 0.909* | 0.909* | 0.667* |

*These were computed before the mAP fix, actual values may differ.

**Preliminary Results (CottonWeedDet12 test=848 images, binary evaluation)**:
| Method | mAP@0.5 | mAP@0.25 | Precision | Recall |
|--------|---------|----------|-----------|--------|
| yolo11nweed.pt (cross-dataset) | 0.316 | 0.374 | 0.473 | 0.404 |
| YOLO11n fine-tuned 100ep | **0.929** | — | **0.930** | **0.850** |

**Issues**:
- Cluster curl doesn't support `--progress-bar` → switched to wget
- Cluster has no `7z` binary → used Python `py7zr` package
- rsync to cluster unreliable → used base64 encoding via ssh for file transfer
- CottonWeedDet12 structure different from expected: images in `weedImages/`, YOLO labels in `annotation_YOLO_txt/` (separate dirs, not alongside images)

**Next**:
- Wait for YOLO fine-tune to complete on cluster (target: mAP@0.5:0.95 > 0.9)
- Run LLM models on CottonWeedDet12 test set
- Small models (moondream, Qwen-3B) test locally on Mac

---

## Phase 1: YOLO Baseline (Complete)

### 2026-03-16 (Session 3)
**Goal**: Optimize YOLO fine-tuning on cluster for better GPU utilization

**Done**:
- Cancelled Job 38007424 (batch=16, only using 2.34G/32G = 7% GPU)
- Updated `setup_and_train.sh`: batch=-1 (auto), workers=5
- AutoBatch selected **batch=60** using 19G/32G (60% of V100)
- Resubmitted as Job 38007481 on V100 (v017)
- Committed and pushed root README.md update (commit 87fc225)

**Training Complete (Job 38007481, 100 epochs, 2.75 hours)**:

| Metric | Validation (best) | **Test (848 images)** |
|--------|-------------------|----------------------|
| mAP@0.5 | 0.943 | **0.929** |
| mAP@0.5:0.95 | 0.898 | **0.865** |
| Precision | 0.944 | **0.930** |
| Recall | 0.884 | **0.850** |

Per-class validation results:
| Class | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|-------------|
| Carpetweeds | 0.976 | 0.936 |
| Crabgrass | 0.979 | 0.935 |
| Eclipta | 0.920 | 0.878 |
| Goosegrass | 0.903 | 0.830 |
| Morningglory | 0.853 | 0.769 |
| Nutsedge | 0.939 | 0.878 |
| PalmerAmaranth | 0.955 | 0.917 |
| PricklySida | 0.970 | 0.947 |
| Purslane | 0.941 | 0.930 |
| Ragweed | 0.993 | 0.983 |
| Sicklepod | 0.945 | 0.858 |
| SpottedSpurge | 0.950 | 0.919 |

- Training config: batch=60 (auto), AdamW lr=0.000625, cos_lr, warmup=5ep, patience=20
- Training speed: ~1.8 min/epoch on V100-32GB
- Model: runs/detect/runs/yolo11n_cottonweeddet12/weights/best.pt (5.5MB)
- **Bug found**: setup_and_train.sh test eval used wrong path (`runs/yolo11n_cottonweeddet12/` vs actual `runs/detect/runs/yolo11n_cottonweeddet12/`). Fixed by running test eval manually.

**Next**:
- Download best.pt from cluster to local
- Run LLM models on CottonWeedDet12 test set (848 images)
- Small models (moondream, Qwen-3B) locally on Mac

---

## Phase 2: Full LLM Benchmark (In Progress)

**Decision**: Paper focuses on CottonWeedDet12 only. All models run on cluster.

### 2026-03-16 (Session 4)
**Goal**: Run 10+ vision LLMs on CottonWeedDet12 test set (848 images)

**Done**:
- Fixed `run_full_benchmark.py` bugs: query_ollama returns dict not tuple, JSON parse handles list format
- Fixed `test_ollama.py`: handle list-format JSON responses
- Ran moondream locally (848 images, 1173s = 20min)
- Submitted 5 HF model jobs to Bridges-2 — all failed due to DOWNLOAD_DIR path bug

### 2026-03-16 (Session 5)
**Goal**: Fix cluster bugs, run all 11 models on CottonWeedDet12

**Bugs Fixed**:
1. **DOWNLOAD_DIR path**: cluster has flat dir structure, `datasets.py` and `run_full_benchmark.py` pointed to wrong downloads path. Fixed with dual-check: `_dl_base if os.path.isdir(_dl_base) else _dl_project`
2. **query_ollama return type**: `run_full_benchmark.py` tried tuple unpacking but function returns dict. Fixed.
3. **JSON list format**: models return `[{...}]` instead of `{"detections": [...]}`. Fixed both files.
4. **Qwen OOM**: CottonWeedDet12 images are high-res, caused 230GB attention allocation on V100-32GB. Fixed by limiting `min_pixels=256*28*28, max_pixels=1280*28*28` in processor.
5. **InternVL2 transformers 5.0 compat**: `all_tied_weights_keys` attribute missing. Fixed with monkey-patch: `caching_allocator_warmup = lambda: None`
6. **Florence-2 transformers 5.0 compat**: `forced_bos_token_id` config error. Fixed with `revision="refs/pr/6"` + monkey-patch.
7. **MiniCPM gated repo**: `openbmb/MiniCPM-V-2_6` requires HF authentication. Skipped for now.
8. **minicpm-v Ollama**: consistently returns HTTP 500. Skipped.
9. **Added Florence-2**: new model type in `roboflow_bridge.py` MODEL_REGISTRY with `_infer_florence()` using `<OD>` task for native object detection.

**Completed Ollama Results (848 images each)**:
| Model | Size | Pred Boxes | GT Boxes | Notes |
|-------|------|-----------|----------|-------|
| moondream | 1.8B | 527 | 1464 | Has predictions but very imprecise |
| llava:7b | 7B | 0 | 1464 | Cannot produce bounding boxes |
| llava:13b | 13B | 0 | 1464 | Cannot produce bounding boxes |
| bakllava | 7B | 0 | 1464 | Cannot produce bounding boxes |

**Still Running on Cluster**:
- qwen7b (Qwen2.5-VL-7B) — resubmitted with pixel limits, Job 38016297
- qwen3b (Qwen2.5-VL-3B) — resubmitted with pixel limits, Job 38016298
- llama3.2-vision:11b — Ollama, Job 38014616 (~45min, processing ~7s/image)
- internvl2 (InternVL2-8B) — resubmitted with monkey-patch, Job 38016326
- florence2 (Florence-2-large) — resubmitted with revision fix, Job 38016327

**Cluster Environment**: transformers==5.0.0.dev0 (bleeding edge, caused many compat issues)

**Key Insight**: Ollama models without native grounding (LLaVA, BakLLaVA) produce 0 bounding boxes. Only models with native bbox output (Qwen2.5-VL, Florence-2, moondream) can generate coordinates. This is a key finding for the paper.

**Next**:
- Wait for qwen7b, qwen3b, llama3.2-vision, internvl2, florence2 to complete
- Run evaluation on all completed results
- If Qwen models succeed, we have 8-10 models for comparison
- Begin Phase 3 (YOLO+LLM Fusion) and Phase 5 (figures/tables)

### 2026-03-16 (Session 6)
**Goal**: Fix cluster bugs (round 3), expand model coverage for paper

**Bugs Fixed (Round 3)**:
1. **Qwen loading speed**: `device_map="auto"` AND `device_map={"": 0}` both trigger transformers 5.0's accelerate weight materialization (~48s/weight x 729 = 10+ hours). Fix: remove device_map entirely, use `.cuda()` instead.
2. **InternVL2 `all_tied_weights_keys`**: Previous patch used `set()` but callers do `.keys()` on it. Fix: use `{}` (dict) not `set()`.
3. **Florence-2 `forced_bos_token_id`**: Error happens INSIDE `AutoConfig.from_pretrained` during Florence2LanguageConfig construction, so patching config after load is too late. Fix: patch `PretrainedConfig.forced_bos_token_id = None` class attribute BEFORE loading.

**Git Email Fix**: Rewrote all commit history to change placeholder `your.email@university.edu` → `harry567566@gmail.com`. Force pushed to remove "claude Claude" from GitHub contributors.

**Model Coverage Expansion**: Research found 7 important missing models. Added to benchmark:

| Model | Type | Size | Why Add |
|-------|------|------|---------|
| Qwen3-VL-8B | VLM+grounding | 8B | Direct successor to Qwen2.5-VL, Jan 2026, best VLM grounding |
| Grounding DINO | Open-set detector | 172M | #1 zero-shot detection model on HuggingFace, essential baseline |
| PaliGemma2-3B | Detection VLM | 3B | Google, native `<loc>` tokens for detection, MIT license |
| YOLO-World v2 | Open-vocab YOLO | ~100M | Bridges YOLO baseline and language-driven detection |
| MiniCPM-V 4.5 | VLM+detect | 8B | Feb 2026, replaces gated v2.6, surpasses GPT-4o on benchmarks |
| Molmo-7B-D | VLM+pointing | 7B | Allen AI, precise pixel coordinates, fully open |
| DeepSeek-VL2 | MoE VLM | 4.5B active | Grounding tokens, efficient MoE architecture |

**Reference**: AgroBench (ICCV 2025) found most open-source VLMs perform near-random on weed identification → validates our paper's importance.

**Cluster Jobs Submitted**: qwen7b, qwen3b, internvl2, florence2 (resubmitted with fixes). llama3.2-vision still running.

**Full Model List (19 models total)**:
- **YOLO baseline**: YOLO11n fine-tuned (mAP@0.5=0.929)
- **HF models (12)**: qwen7b, qwen3b, qwen3_8b, internvl2, florence2, grounding_dino, paligemma2, yolo_world, minicpm_v45, molmo2, deepseek_vl2, minicpm (skipped/gated)
- **Ollama models (6)**: moondream, llava:7b, llava:13b, bakllava, llama3.2-vision:11b, minicpm-v (HTTP 500)

---

## Phase 3: YOLO+LLM Fusion (Planned)

3 strategies: supplement, filter, weighted

---

## Phase 4: Ablation Studies (Planned)

Prompt engineering, model size, grounding capability, fusion IoU sweep

---

## Phase 5: Paper Writing (Planned)

Generate figures, tables, write paper
