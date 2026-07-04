# Platform Optimization Plan — from "weed tool" to a general Physical-AI Agent Platform

Authoritative roadmap (2026-07-04). The framework is a BROAD physical-AI agent platform (robotics,
drones, sensors, vision, …), not a weed tool. This plan is grounded in a code audit, not aspiration.
Execute phases IN ORDER; each phase is small, testable (smoke + e2e stay green), honest, and deployed +
verified before the next. Do NOT regress the working dataset pipeline.

## 0. North star
A researcher in ANY physical-AI field creates a project, and a team of agents (collect / filter / label /
train / evaluate / + custom) works that field autonomously and compounds over time. "New field = fill a
config, not change code." Intelligence comes from the RIGHT model in the RIGHT place, not hardcoded rules.

## 1. Compute-placement doctrine (what runs WHERE and WHY)  ← the core question
Two tiers, chosen by latency + resource profile:

**Lab server (always-on; RTX 3060 12GB, ~15GB RAM, single-node). Runs the PRODUCT.**
- The website (FastAPI), MongoDB, the dataset image library, read-only analysis (EDA), the audit trail.
- SMALL, FAST, interactive models only: qwen2.5:3b (planning / analysis summary / fitness), Whisper-base
  (voice). Rule: a lab model call must be (a) GPU-resident only while running, unloaded when idle,
  (b) single-flight (one inference at a time behind a lock — a 12GB card can't run two), (c) time-bounded,
  and (d) **NEVER block the web event loop** (offloaded to a thread; see Phase 0). Target < ~15s p95.
- Why here: interactive UX needs low latency + no cluster queue; the 3060 is idle otherwise; keeping the
  data local = fast EDA (no Lustre).

**Bridges-2 cluster (on-demand GPU; V100/L40S/H100). Runs the HEAVY, BATCH compute.**
- Harvest (LLM brain), training, evaluation, DINOv2 curation, big-model inference (glm-4.7-flash /
  deepseek-v3), synthetic generation. All as queued sbatch jobs; the cluster is NEVER persistent.
- Why here: GPU-hours, large VRAM, long runtimes, parallelism. Governed by an SU budget (finite) → prefer
  the smallest model/hardware that does the job (glm-4.7-flash 30B for routine reasoning; 671b only for
  genuinely hard one-offs).

**Placement rules of thumb:** interactive + small → lab. Batch + heavy/parallel/large-VRAM → cluster.
Anything that would block the website or exceed ~15GB VRAM or run > ~30s → cluster (async).

## 2. Performance guardrails (Harry's explicit concern: models must not slow the site)
Evidence: `uvicorn --workers 1`; `api_agent_plan` / `api_voice_transcribe` are `async def` but call the
SYNC blocking `_llm.chat()` / `whisper.transcribe()` → they stall the single event loop → the whole site
hangs for everyone during a 10–60s inference. This is the #1 performance defect.
Guardrails:
- No blocking work on the event loop: offload every sync LLM/Whisper/subprocess call via
  `asyncio.to_thread(...)` (or make the endpoint a sync `def` so FastAPI thread-pools it). 
- A lab-local inference LOCK + queue (single-flight on the 3060) with a bounded wait + honest "busy, try
  again" instead of piling up.
- Bump uvicorn workers to ≥2 (only after handlers are non-blocking; workers don't fix a blocked loop).
- Model keep-alive tuned so idle unloads free the GPU; cap context to control latency.
- Cluster: SU-aware — log SU spend; never fire the 671b for routine work; batch/parallelize sensibly.

## 3. Phases (execute in order)

### Phase 0 — Foundations: make the site non-blocking + an async job spine  (SAFETY-CRITICAL, do first)
- Wrap all lab model calls (plan, analyze/ai, voice, models/test) in `asyncio.to_thread` + a single-flight
  lock; add a tiny "fast job" pattern for anything > a few seconds (return id, poll) so the UI never hangs.
- Make `/api/train/submit` + `/api/eval/submit` ASYNC (return a jobtag immediately; stage+sbatch in a
  background thread; poll for status) — kills the 152s button-hang.
- Verify: concurrent requests during a planning call stay responsive; smoke/e2e green.

### Phase 1 — Domain-config layer (de-weed; the "fill a config, not change code" unlock)
- One source of truth per project: a `domain config` doc (Mongo + staged to cluster) with:
  taxonomy/classes, harvest queries + accept-vocab, quality thresholds (DINO threshold, imbalance/dup
  limits, min-per-class), reference-pool policy, target Roboflow project, modality, target metric, model
  routing overrides.
- Route EVERY hardcoded weed/CWD12 constant (802 hits) through this config, with weed as just the
  default/first config. Collector/Filter/Labeler/Evaluator already thread the domain — now feed them the
  CONFIG, not env-only. New domain = new config doc.
- Verify: a non-weed project (e.g. "coral") drives harvest vocab, quality thresholds, and the roboflow
  target purely from its config; weed behaviour unchanged; tests green.

### Phase 2 — Model routing (put the big models to work; right model, right place)
- A model-router: role → (model, where). Roles: interactive-plan (lab qwen2.5:3b), analysis-summary (lab),
  harvest-brain (cluster gemma4/glm-4.7-flash), deep-review (cluster glm-4.7-flash on-demand),
  hard-reasoning (cluster deepseek-v3, rare). Configurable per domain.
- Replace the thin "3B润色" analysis + planning with genuinely useful models WHERE latency allows (fast=lab
  small; deep=cluster bigger, async). Stop leaving 671b/glm idle.
- Verify: analysis/harvest quality visibly better; latency budgets respected; degrades gracefully.

### Phase 3 — Agents fully domain- + modality-agnostic
- Generalize collector/filter/labeler/trainer/evaluator beyond vision: sensor/video/pointcloud/audio/text
  paths where feasible (trainer already general for vision; add honest paths or clear "not yet" per
  modality). Quality/label/eval driven by the domain config, not weed assumptions.

### Phase 4 — The closed, compounding loop
- One "run a round" that chains collect → filter → (human) label → train → evaluate → feed metrics back to
  bias the next collect — per domain, one click / scheduled, with provenance. This is what makes it an
  "agent that compounds", not stitched manual steps.

### Phase 5 — Structure, observability, governance
- Modularize the 12k-line monolith by domain (auth/dataset/agents/models/analysis) — carefully, tests green.
- Observability: per-domain agent-run timeline + job status + SU/cost. Data governance: dataset
  license/provenance/versioning; export. Reproducibility: config-driven, recorded runs.

## 4. Cross-cutting discipline (every phase)
UX-first + smooth + MOBILE-tested. ALL-ENGLISH product. Honest (label scaffold; failures stated). Extend
tests/smoke_test.sh + tests/e2e_dataset.sh; keep green. Deploy + verify each step. Resource-sensible (don't
burn GPU/SU/Roboflow to "prove" what a unit test can show). Small commits, CHANGELOG + README current.

## 5. Ordering rationale
Phase 0 first because a hanging site blocks everything and it's Harry's explicit concern + low-risk.
Phase 1 next because de-weeding is the single biggest lever for "general platform" and unblocks Phases 2–4.
Then intelligence (2), breadth (3), autonomy (4), structure (5).
