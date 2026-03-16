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
| YOLO11n fine-tuned (cluster, pending) | — | — | — | — |

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

## Phase 1: YOLO Baseline (In Progress)

**Status**: Fine-tuning YOLO11n on CottonWeedDet12 on Bridges-2 V100 (Job 38007424)
- 100 epochs, batch=16, imgsz=640, patience=20
- Early local results (5 epochs on MPS): mAP@0.5=0.695, trending up

---

## Phase 2: Full LLM Benchmark (Planned)

**Decision**: Paper focuses on CottonWeedDet12 only
**Models**: qwen7b, qwen3b, minicpm, internvl2, florence2 (HF) + moondream, llava:13b, llama3.2-vision:11b (Ollama)
**Small models test locally**, large models on cluster

---

## Phase 3: YOLO+LLM Fusion (Planned)

3 strategies: supplement, filter, weighted

---

## Phase 4: Ablation Studies (Planned)

Prompt engineering, model size, grounding capability, fusion IoU sweep

---

## Phase 5: Paper Writing (Planned)

Generate figures, tables, write paper
