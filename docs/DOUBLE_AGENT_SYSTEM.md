# The Double-Agent System — consolidated record

*Consolidated 2026-08-22 from the full git history (558 commits, 2026-03-12 → 2026-08-16),
the framework source, `CHANGELOG.md`, `RESEARCH_LOG.md`, and the on-disk result artifacts.
Every number below carries its provenance; caveats are stated inline, not hidden.*

---

## 1. What it is

Two autonomous agents run **concurrently** against one shared, locked registry, and
compound over time:

```
┌─ Job-D · COLLECTOR agent ─────────────┐   ┌─ Job-T · TRAINER agent ──────────────┐
│ Brain (LLM tool-caller) decides:      │   │ merge all registry datasets           │
│  search HF/Kaggle/GitHub/Roboflow     │   │  (dedup dHash · holdout dHash guard)  │
│  → download → OWLv2 pseudo-label      │   │ → train YOLO (mini-rounds)            │
│  → DINOv2 curation / quality filters  │   │ → eval vs sealed cwd12 holdout        │
│  → register into dataset_registry     │   │ → write metrics back                  │
└──────────────┬────────────────────────┘   └───────────────┬───────────────────────┘
               │        append-only registry (fcntl lock, atomic writes)             │
               └───────────────►  hot-reload: new data enters BETWEEN mini-rounds ◄──┘
```

- **Two SLURM jobs, truly parallel** (REQ-1): the collector harvests while the trainer
  trains; the trainer picks up newly registered datasets between mini-rounds without a
  restart (`tools/hot_reload_trainer.py`).
- **The Brain has two operating modes** (`orchestrator.py`): *strategy* (propose a full
  plan, rigid pipeline executes) and *agent* (decide one action at a time from tool
  results — inspect labels, run VLM inference, adjust thresholds, stop).
- Since mid-June the same loop's controls are **platform features**: every project has
  Collector / Filter / Labeler / Trainer / Evaluator agent cards
  (`POST /api/cluster_action/*`), and a round ledger (`/api/domain/rounds`,
  `/api/domain/round/start|step`). The piece that is *parked* is the continuous
  auto-cycling scheduler — today rounds are fired manually per project.

## 2. Component map (all verified present on `main`, 2026-08-22)

| Component | File | Lines | Role |
|---|---|---|---|
| Orchestrator | `weed_optimizer_framework/orchestrator.py` | 1,185 | round loop, 2 modes |
| Brain | `weed_optimizer_framework/brain.py` | 764 | LLM tool-caller (history: qwen2.5 → deepseek-r1 → gemma4) |
| Dataset discovery | `tools/dataset_discovery.py` | 1,533 | autonomous HF/Kaggle/GitHub search+download (no curated lists) |
| Mega trainer | `tools/mega_trainer.py` | 935 | merge + dedup + **holdout dHash guard** + train |
| Hot reload | `tools/hot_reload_trainer.py` | 250 | new datasets between mini-rounds |
| Registry lock | `tools/registry_lock.py` | — | fcntl lock + atomic writes (v3.1.0) |
| Autolabel | `tools/autolabel.py` | — | OWLv2 → YOLO pseudo-labels (conf floor 0.30) |
| Curation | `dinov2_*.py`, `label_filter.py`, `data_quality_eval.py` | — | quality gates |
| v2-era agent | `run_agent_optimizer.py`, `run_hyperagent.py` | 522/498 | StrategyBrain + OPRO + 3-VLM consensus (predecessor) |
| SBATCH entrypoints | `run_v3_0_26_jobd.sh`, `run_v3_0_30_jobd_continuous.sh`, `run_v3_0_30_dashboard_server.sh` | — | the parallel deployment |

Nothing of the double-agent core was ever deleted (only `dashboard/pages/brain_timeline.py`,
a Streamlit view, was removed with the old dashboard).

## 3. Evolution timeline (commit-verified)

| When | Milestone | Evidence |
|---|---|---|
| 2026-03-12→18 | **Phase 1 — VLM benchmark**: 15 models on CottonWeedDet12; fine-tuned YOLO11n 0.929 mAP@0.5 / 0.865 mAP50-95 vs best VLM (Florence-2) 0.434 mAP@0.5 | commits 03-17/18; RESEARCH_LOG |
| 2026-04-04→06 | **Phase 2 — agent optimizer (v2.x)**: DeepSeek-R1 Brain, 3-model consensus labels, `analyze_failure` → targeted label filtering | commits v2.0/v2.1 |
| 2026-04-27→29 | **v3.0.25**: canonical 12-class mapping, honest cwd12 val, class-balanced oversampling | commits |
| 2026-05 | **v3.0.26/30**: true parallel Job-D + hot-reload + continuous harvest; scale experiments to **244,675 images** (154,721 unique after dedup) | CHANGELOG L1736, L2270 |
| 2026-05-22 | **v3.0.38-A: cwd12 holdout mAP50-95 = 0.9033** (seed 102, best of N seeds) — *pre-leak-fix; see SCIENCE_AUDIT* | CHANGELOG L3963 |
| 2026-07-04 | **v3.1.0 audit hardening**: holdout dHash guard, registry locking, token removal, error-honesty fixes (merged to main, `bf621ce`) | CHANGELOG v3.1.0 |
| 2026-06→08 | **Phase 3 — the platform**: the pipeline generalized to multi-domain projects; robot live uplink (v3.20-3.22) becomes the second collection modality | CHANGELOG v3.2+ |

## 4. The honest scientific finding the curves actually show

The recorded numbers tell a **quality-beats-scale** story, which is the system's most
defensible result:

| Recipe | Data | cwd12 holdout mAP50-95 | Merge path | Anchor |
|---|---|---|---|---|
| Supervised YOLO11n baseline | cwd12 alone (5,648 imgs) | **0.865** (0.929 mAP@0.5) | none | v3.0.6 |
| **RF-DETR Large, cwd12-only** | cwd12 alone | **0.8974 ± 0.0040** (n=4 seeds; best 0.9033) | none — `train_rfdetr.py` stages cwd12 directly, holdout stems excluded | v3.0.31/34/38 |
| Naive scale + pseudo-labels | 244,675 harvested imgs | **0.593** ⚠ pre-guard | `mega_trainer` merge | v3.0.26-p2 |
| Cumulative scale variant | ~150–240K | 0.576 ⚠ pre-guard | `mega_trainer` merge | v3.0.32 |
| Curated clean subset (cwd12-only staging) | 3,671 | **0.896** | none (zero merge calls; reclassified v3.22.3) | v3.0.28 |
| **M1 sealed · merged raw** | 55,690 | **0.6032 ± 0.0046** (n=3) | `mega_trainer` + content-hash guard; train∩holdout = 0 | jobs 44234060, 2026-08-23 |
| **M1 sealed · merged @ DINO≥0.50** | 13,309 | **0.5894 ± 0.0025** (n=3) | same, gate skipped 36/45 slugs | jobs 44234063, 2026-08-23 |

Uncurated pseudo-labeled scale *hurt* accuracy by ~0.27 absolute; the curation stack
(DINOv2 + label filters + consensus) recovered it and beat the supervised baseline.
Supporting artifacts: `results/framework/*_results.json`, `v3_0_23_mega_iter6_train8_results.csv`,
per-species table in `RESEARCH_LOG.md` (§2026-07 entries), anti-forgetting rounds
(round-3 zero-forgetting, CHANGELOG L685-691), and the small-object failure analysis
(mAP 0.87 → 0.40 on small-box subsets, CHANGELOG L1944-1961).

**Which numbers are quotable today** (see `SCIENCE_AUDIT.md` §1): the two *single-dataset*
rows are clean — they never pass through the merge, so the C1 leak channel cannot reach
them; quote RF-DETR as **0.8974 ± 0.0040 (n=4)**, never the bare best run. The three
*merged-corpus* rows carry a pre-guard caveat until M1 re-measures them under the sealed
content-hash guard; the direction of the quality-vs-scale effect (large, ~0.3 absolute)
is far bigger than any plausible leak correction, but the exact values must be restated.

## 5. How it plugs into today's platform

- **Collection modality 1 (autonomous web harvest)**: Collector agent card → `brain_harvest`
  cluster action → registry → project datasets.
- **Collection modality 2 (robots, v3.20+)**: 241 robot / laser cart stream live sensor +
  camera data into project datasets over the public internet (`tools/robot_ingest.py`).
- **Training side**: per-project Trainer/Evaluator cards (`/api/train|eval/submit`) on
  cluster GPUs; metrics written back to the project.
- **Missing for the full closed loop**: the continuous scheduler that alternates
  harvest→train rounds unattended per project, and a compounding-metric view (round №
  vs holdout metric) on the project page. Both have their endpoints and ledger already;
  see SCIENCE_AUDIT §4 for the milestone plan.
