# Server map — orientation for `dashboard_server.py`

Read this first to be "in the zone" instead of rediscovering the codebase each time.
Scanned 2026-07-05. The dashboard is **one FastAPI app**: `weed_optimizer_framework/tools/dashboard_server.py`
— **12,953 lines · 120 routes · 221 top-level functions · 22 in-process global caches · 0 classes**.
Server-rendered HTML (f-strings) + JSON APIs. Boots without torch (see `requirements-dev.txt`).

- **Entry:** `uvicorn weed_optimizer_framework.tools.dashboard_server:app --port 8000` (lab uses 8000, workers 1 — shared in-proc state).
- **Live:** lab server, Tailscale — deploy via `reference_labserver_deploy` (rsync + `systemctl --user restart weed-dashboard`). Public: `https://lab-b660m-c.tailfa6424.ts.net`.

## Auth & request lifecycle
- Middleware `_auth_and_rate_limit` (~L290) gates every request. Order: exempt paths → `_session_user` (Google OAuth cookie, `_session_user` ~L214) → API key (`X-API-Key`) → HTTP Basic. **Students sign in with Google** (`/login`, `/auth/google/start|callback`); an **admin Basic backdoor** (`DASH_USER`/`~/.dashpass`, currently `1`/`1` on lab — SHOULD be changed) is used for scripts/testing.
- **Fail-closed** (v3.1): if no auth method is configured → 503 (not open). Rate-limit: 5 bad tries/IP → 1h lock.
- `_is_admin(actor)` ~L1893 (reads `_admin_cache` ~L1870); `_can_use_cluster` gates cluster actions.

## Routes by area (120 total)
- **Pages (HTML):** `/` (projects home → `root`), `/agent/{domain}` (project workspace → `agent_generic`; `/agent/weed` aliases it), `/console` (Mission Control), `/guide`, `/dataset/{slug}` (analysis page), `/gallery/{slug}`, `/slugs`, `/classes`+`/classes/{cls}`, `/rounds`, `/labeling`, `/roboflow`, `/manual`, `/models`, `/users`, `/control`, `/annotate`, `/audit`+`/audit/method|class`, `/synth/{kind}`, `/morning_report`, `/login`, `/logout`.
- **Projects/agents:** `POST /api/agent/create|delete|update`, `/api/agent/plan` (AI project planner), `/api/project/agents`, `/api/project/agent/add|remove`, `GET /api/domains`, `/api/agent_types`.
- **Datasets:** `POST /api/dataset/upload` (magic-byte archive/image; multipart or raw), `/api/dataset/delete`, `GET /api/dataset/uploads`, **`/api/dataset/analyze`** (EDA → `_analyze_dataset` ~L3412), **`/api/dataset/analyze/ai`** (LLM review → `_analyze_dataset_ai` ~L3812), `/api/dataset/analyze/ai/submit`.
- **Domain config / rounds:** `GET/POST /api/domain/config`, `/api/domain/rounds|activity`, `/api/domain/round/start|step`, `/api/domain/push_cap`.
- **Cluster:** `POST /api/cluster_action/{action}` (fixed argv whitelist `_CLUSTER_ACTIONS` ~L8501: brain_harvest, download_known_slugs, restart_dashboard, roboflow_sync_*, owl_*, dinov2_*, train_yolo_*, …), `/api/cluster_status`, `/api/cluster/request`, `/api/cancel_job/{jobid}`, `/api/job/status`, `/api/job_log/{jobid}`, `/api/recent_jobs`, `/api/submit/status`.
- **Train/eval:** `POST /api/train/submit` + `GET /api/train/result`; `POST /api/eval/submit` + `GET /api/eval/result`.
- **Labeling (human-in-loop):** `POST /api/labeling/push|delete|simulate`, `GET /api/labeling_status`, `/api/annotation_status`, `/api/exemplar*`, `/api/slug_verdict{,s}`, `/api/flag{,s}`.
- **Models gateway:** `GET /api/models{,/catalog}`, `POST /api/models/role|test|deploy`, `/api/llm/infer{,/result}`, `/api/voice/transcribe` (Whisper, `_WHISPER` ~L3949).
- **Users/keys:** `/api/users{,/role,/cluster_access}`, `/api/keys{,/revoke}`, `/api/me`.
- **Media serving:** `/api/img`, `/api/sample`, `/thumb*`, `/raw_reg`, `/synth/raw`, `/audit/raw`.
- **State/status:** `/healthz` (public; returns `{repo, ts}`), `/api/state`, `/api/db_status`, `/api/disk_usage`, `/api/roboflow_status`, `/api/rounds_state`, `/api/per_species_stats`, `/api/action_history`.

## Key internal helpers (grep by name — line anchors drift)
`_analyze_dataset` (EDA: modality mix, image dims, near-dup, YOLO/COCO/classification annotation detect) · `_analyze_dataset_ai` (adds LLM summary + training-readiness via `model_router`→`llm_providers`; falls back to rule-only) · `_detect_dataset_issues` (now in `dataset_quality.py`) · `_resolve_slug_dir` (slug→disk path) · `_read_manual_uploads` (manual_uploads.json) · `_cluster_action` runner (sbatch/subprocess/restart_self, argv whitelist) · `_MODALITY_EXT` ~L1796 · `_ANALYSIS_SAMPLE_CAP`.

## In-process state (LOST on restart — no persistence)
`_BG_JOBS` (async submit jobs, ~L3961) · `_admin_cache` · `_registry_parse_cache` · `_WHISPER` · `_SACCT_TERMINAL/_ACTIVE` (Slurm) · `_CLUSTER_ACTIONS` · `_MODEL_ROLES` · `_SEED_CATALOG` · plan/synonym/prompt dicts. Persisted state lives in Mongo + `results/framework/*.json` (registry, manual_uploads, flags, dataset_analysis cache, class_topic_overrides).

## Module responsibilities (`weed_optimizer_framework/`)
- **Core:** `orchestrator.py` (harvest→autolabel + train loop, 2 modes) · `brain.py` (SuperBrain LLM tool-caller) · `config.py` (paths/models) · `memory.py` · `monitor.py`.
- **Data access:** `tools/db.py` (Mongo-first / JSON-fallback; registry, domains, users, audit) · `tools/registry_lock.py` (fcntl lock + atomic writes) · `tools/storage.py` · `tools/rounds.py` (round ledger).
- **Harvest/sources:** `dataset_discovery.py` (HF/Kaggle/GitHub autonomous search+download) · `roboflow_source.py` · `extra_sources.py` · `bucketer.py` · `topic_classifier.py`/`class_topic_store.py`.
- **Curation/quality:** `dinov2_curator.py` / `dinov2_object_curator.py` / `dinov2_round_filter.py` / `dinov2_route.py` · `dino_label_verifier.py` · `label_filter.py` · `data_quality_eval.py` · `dataset_integrity_audit.py` · **`dataset_eda.py`** + **`dataset_quality.py`** (de-monolith extractions from the dashboard).
- **Labeling:** `autolabel.py` (OWLv2→YOLO) · `owl_preannotate.py`/`owl_precision.py`/`owl_upload_proposals.py` · `label_gen.py` (multi-VLM consensus) · `labeling_tracker.py` (lifecycle events) · `roboflow_sync.py`/`merge_roboflow_projects.py`.
- **Train/eval:** `mega_trainer.py` (merge + YOLO train; **holdout guards live here**) · `hot_reload_trainer.py` · `train_rfdetr.py` · `yolo_trainer.py`/`lora_yolo.py` · `train_from_roboflow.py`/`train_yolo_on_verified.py` · `evaluator.py`/`pycoco_eval.py`/`wbf_*`/`rfdetr_*_wbf`/`*_tta*` (ensembles/TTA).
- **Synth:** `synth_cutpaste.py` · `synth_diffusion.py`/`flux_lora_train.py` (FLUX).
- **Models/LLM:** `model_router.py` (role→model→where) · `llm_providers.py` (stdlib LLM abstraction) · `model_discovery.py` · `vlm_pool.py` · `web_identifier.py`.

## Core data flows
1. **Create project:** `POST /api/agent/create` → `db` writes domain (Mongo). Auto-derives harvest queries + accept-vocab from research field. Redirects to `/agent/{domain}`.
2. **Upload → analyze:** `POST /api/dataset/upload` extracts to `uploads/<slug>/{images,labels}` (v3.1: no double-nest; local_path=parent for YOLO; writes data.yaml) → registers (Mongo+JSON). `/dataset/{slug}` calls `_analyze_dataset` (EDA + annotation detect) then `_analyze_dataset_ai` (LLM summary + readiness, lab uses local `ollama:qwen2.5:3b`; rule fallback if no LLM).
3. **Harvest rounds:** Collector/Filter/Labeler/Trainer/Evaluator agents fire `POST /api/cluster_action/{action}` → sbatch on cluster. Rounds tracked (`/api/domain/round/*`); registry is the append-only source of truth (locked writes).
4. **Labeling:** push N imgs → Roboflow → human labels → export back → delete → repeat; every stage in `labeling_tracker` (Mongo+JSONL). Simulated events now flagged `meta.simulated`.
5. **Train/eval:** `/api/train|eval/submit` stage data to cluster + sbatch; metric written back to `results/framework/*_results/`.

## Known debt (see CHANGELOG v3.1.x + audit memory)
Monolith (being de-monolithed one pure unit at a time — `dataset_eda.py`, `dataset_quality.py` done) · ~47 hardcoded `/ocean/...byler` paths · lab is a non-git flat copy (deploy = rsync) · admin `1/1` creds on lab.
