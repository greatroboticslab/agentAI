# agentAI

A collection of AI agent projects for agricultural research at MTSU Great Robotics Lab.

## Projects

| Directory | Description |
|-----------|-------------|
| [`weed_llm_benchmark/`](weed_llm_benchmark/) | Benchmark framework for evaluating open-source vision LLMs on weed detection in agricultural images. Tests 12+ models across HuggingFace and Ollama backends, with Roboflow integration for dataset management. |

## Current Progress

### Weed LLM Benchmark — Paper: *"Can Vision LLMs Detect Weeds?"*

**Goal**: Benchmark open-source vision LLMs against YOLO for weed detection on CottonWeedDet12 (5,648 images, 12 weed species).

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0: Evaluation Module | Done | `evaluate.py` (mAP, precision, recall), `datasets.py`, format converters |
| Phase 1: YOLO Baseline | **Done** | YOLO11n fine-tuned on CottonWeedDet12 — Test: mAP@0.5=0.929, P=0.930, R=0.850 |
| Phase 2: Full LLM Benchmark | Planned | Run 8 vision LLMs on CottonWeedDet12 test set |
| Phase 3: YOLO+LLM Fusion | Planned | 3 fusion strategies: supplement, filter, weighted |
| Phase 4: Ablation Studies | Planned | Prompt engineering, model size, grounding capability |
| Phase 5: Paper Writing | Planned | Figures, tables, manuscript |

### Key Files Added (2026-03-15)

- `evaluate.py` — mAP@0.5, mAP@0.5:0.95 with proper PR-curve computation
- `datasets.py` — Dataset registry and download management
- `run_yolo_baseline.py` — YOLO baseline evaluation
- `run_full_benchmark.py` — Multi-model orchestrator with checkpoint/resume
- `run_ablations.py` — Ablation experiment runner
- `generate_paper_figures.py` / `generate_tables.py` — Publication figures and LaTeX tables
- SLURM scripts for Bridges-2 cluster (`setup_and_train.sh`, `submit_all_jobs.sh`, etc.)
- `RESEARCH_LOG.md` — Detailed daily research progress

See [`weed_llm_benchmark/README.md`](weed_llm_benchmark/README.md) for full documentation.
