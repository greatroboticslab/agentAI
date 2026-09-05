# Tiered supervision — execution specification (WP1–WP8)

**Status: SPEC v1 · 2026-09-04 · companion to `TIERED_SUPERVISION_PLAN.md` (decisions, claims,
evaluation protocol). This file is the executor's contract: file paths, interfaces, schemas,
commands, verification steps and gates. Where a fact is not yet verified it is marked `VERIFY:`
with the command that settles it. English only; repo docs are engineering records.**

---

## Part A — Executor handbook

### A.1 Topology
- **Lab server (always on)** `lab@lab-b660m-c` (Tailscale; public URL
  `https://lab-b660m-c.tailfa6424.ts.net`). Ubuntu 24.04, RTX 3060 12 GB, 15 GB RAM. Runs the
  FastAPI dashboard as `systemctl --user weed-dashboard` on :8000 with **one uvicorn worker — all
  in-process state, including the round-scheduler thread, lives in that process**; Mongo
  `mongodb://127.0.0.1:27017/agentai`; user-space ollama on 127.0.0.1:11434 (`qwen2.5:3b`,
  `qwen2.5-coder:7b`); `weed-sync.timer` pulls cluster→lab every 30 min. Repo copy
  `/home/lab/weed_llm_benchmark` is a **flat rsync copy, not a git checkout**. Playwright Firefox
  works for real-browser checks (Chrome's network is broken there).
- **Cluster (Bridges-2, on demand)** account `byler`, allocation `cis240145p` (GPU-only QoS).
  Repo `/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark` (note the extra `harry/`).
  Conda env `bench` (ultralytics 8.4.37, transformers ≥ 4.57, torch 2.5.1, huggingface_hub 1.8,
  `hf_transfer`). Two copies of `weed_optimizer_framework/` exist (git tracks the nested one; SBATCH
  imports the outer one); every job script runs `git reset --hard origin/main` and copies
  nested→outer at start ⇒ **push to `origin/main` before the first sbatch of any change; keep
  per-run configuration in untracked files on /ocean**.
- Shared assets on /ocean: ollama binary `/ocean/projects/cis240145p/byler/ollama/bin/ollama`
  (**v0.33.3 since 2026-09-04**, previous 0.20.6 kept in `bin-0.20.6-bak/`), model store
  `OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models` (442 GB: gemma4, glm-4.7-flash,
  deepseek-v3:671b, qwen3:14b, qwen2.5:7b, deepseek-r1:7b, VLMs); vLLM 0.28.0 apptainer image
  `/ocean/projects/cis240145p/byler/containers/vllm-openai.sif` (verified: torch 2.13, CUDA ok,
  registry includes `DeepseekV4ForCausalLM`, `KimiK3ForConditionalGeneration`, `Glm4MoeForCausalLM`,
  `Qwen3_5MoeForCausalLM`); HF checkpoints under `/ocean/projects/cis240145p/byler/models/hf/`.
- Hardware/queue facts (2026-09-04): H100 nodes w001–w010 = 8×H100-80 GB + ~2 TB RAM + 104 cores,
  no local disk; whole-node (`-p GPU --gres=gpu:h100-80:8`) waits ≈ a day (all nodes allocated,
  8 whole-node requests pending); **GPU-shared** (`-p GPU-shared`, `DefMemPerGPU=63000 MB`,
  1–8 GPUs on one node) flows in minutes–hours. H100 = 2 SU/GPU-h, V100 = 1. Balance 11,005 SU.
  **Storage is the lab's shared project quota**: `df -h /ocean/projects/cis240145p` shows ~1.0 TB
  headroom; exhausting it breaks every lab job. HF download from a compute node ≈ 268 MB/s.
- Network: compute nodes reach HF/Kaggle/ollama registries; GitHub is unreliable; the lab Mongo is
  unreachable from compute nodes; compute→compute TCP works; login→compute TCP is unverified.
  The plan therefore uses **file RPC on /ocean** for every model window, never network assumptions.
  Persistent daemons start in the SBATCH main body (never under `srun --overlap`).

### A.2 Reaching the cluster
From the Mac, only through the lab relay: `ssh -i ~/.ssh/id_ed25519_lab lab@lab-b660m-c
"export CPW=<cluster password from the local secret note>; python3 ~/ptyssh.py \"<one batched
command>\""`. **One session at a time, one batched command per session** (rapid SSH triggers a
login throttle; back off ≥ 30 min on `Connection closed`). File transfer = inline base64 inside
the command (no scp/SFTP), ≤ 96 KB raw per command (chunk or gzip). Verify imports on the login
node before every sbatch. Job output: `results/framework/<name>_<jobid>.out` on the cluster.

### A.3 Lab deploy iron rule
edit → `python3 -m py_compile <file>` (+ `node --check` on any extracted inline script) → commit →
`rsync` the file(s) to `lab:~/weed_llm_benchmark/…` → `systemctl --user restart weed-dashboard`
(planned restarts only; check `/api/robot/sessions` for a live robot session first) → wait ~25 s →
verify as admin with a **minted session cookie** and as a member → `tests/smoke_test.sh` green.
Cookie: on the lab, b64url(JSON `{uid, exp}`) + `.` + b64url(HMAC-SHA256 over that body with the
raw 32 bytes of `~/.dash_session_key`), sent as `agentai_session=`. **Never poll unauthenticated**
(5 failures per IP = 1 h lockout). Basic auth is disabled.

### A.3b Publishing a change to the cluster (the step that is easy to forget)

The cluster repo root holds **untracked runtime copies** of the job scripts and of the package.
`sbatch` reads the script from the repo root at submit time, and the job's own
`git reset --hard origin/main` does **not** refresh an untracked file. A pushed change therefore
does not reach a job until it is published:

```
# after `git push origin main`, in ONE batched relay command:
cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark \
  && git fetch origin && git reset --hard origin/main \
  && cp -f weed_llm_benchmark/run_*.sh . \
  && rsync -a --delete weed_llm_benchmark/weed_optimizer_framework/ weed_optimizer_framework/ \
  && git log --oneline -1 && md5sum run_m1_merged_seeds.sh weed_llm_benchmark/run_m1_merged_seeds.sh
```

The two md5 sums must match; that is the proof the outer copy is the code that was pushed.
Job scripts also rsync the package themselves at start, so the package copy is belt and braces —
the `run_*.sh` copy is the one nothing else does for you.

### A.4 Verification standard
"Launched" ≠ done. Done = the code path ran on real infrastructure, the artifact exists (`ls -la`,
`tail`), the ledger row exists (JSON), the page renders in a real browser (screenshot),
`smoke_test.sh` is green. Numbers: mean±std with n, or a Wilson interval; an effect below the
recipe's 2×seed-std is noise. Never write "fully working" while a gate line is unmet.

### A.5 Repository rules
English only in code/prompts/UI/docs; no assistant or vendor attribution anywhere in the repo,
commits carry no co-author trailer; docs record what changed / why / how verified. Every push =
code + CHANGELOG entry (v3.25.x series) + RESEARCH_LOG entry (newest-first) when a result lands;
`git rev-list --count origin/main..main` must read 0. Never commit secrets; scrub logs
(usernames, paths, tokens) before committing artifacts. Never delete data/models, never
whole-write the registry, never touch NEVER_TRAIN / TRUSTED_SLUGS / PASS_BAR / holdout guards
(R4, human only). Robot frames are private — never leave test uploads behind.

### A.6 Read before acting
`docs/TIERED_SUPERVISION_PLAN.md`, `docs/SUPERWEED_PLAN.md`, `docs/DOUBLE_AGENT_SYSTEM.md`,
`docs/SCIENCE_AUDIT.md`, `docs/SERVER_MAP.md`, `docs/BEST_MODEL_CARD.md`. Code:
`tools/round_scheduler.py` (the loop), `tools/db.py` (ledger L507-621, domain config L429-496),
`tools/model_router.py`, `tools/llm_providers.py`, `tools/dashboard_server.py` (grep
`_CLUSTER_ACTIONS`, `api_cluster_action`, `/api/models/deploy`, `_slurm`, `_log_action`,
`/api/domain/rounds`, `_inject_responsive_css`), `run_m1_merged_seeds.sh`,
`run_v3_0_43_brain_harvest_oneshot.sh`, `run_s2_dino_scores.sh`, `run_llm_infer.sh`,
`run_deploy_model.sh`, `run_mongo_node.sh` (publish pattern), `tools/sync_health.py` (alarm
pattern), `tools/registry_lock.py`, `tools/dataset_discovery.py` (`_save_registry` L272-293),
`tools/sample_audit.py`, `tools/pool_report.py`, `tools/quarantine.py`, `tools/mega_trainer.py`.

### A.7 Shared conventions for every work package
- New code lives in `weed_optimizer_framework/tools/brain/` (package with `__init__.py`); nothing
  is added to `brain.py`/`orchestrator.py` (legacy, documented as such).
- Cluster-side staged state lives under `results/framework/_brain/<domain>/` (untracked) and
  `results/framework/_endpoints/`; add both plus `results/framework/llm_infer/`,
  `results/framework/model_deploy/`, `*.out` to `.gitignore` in WP1.
- Actor strings are structured: `round-scheduler`, `tier0:<model>`, `tier1:<model>`,
  `tier2:<backend>`, `human:<email>` — set internally, never from request headers.
- Every ledger write goes through a write-ahead JSONL on the lab
  (`results/framework/_brain/<domain>/wal.jsonl`) before Mongo; a failed Mongo write raises the
  `mongo_down` signal and blocks submissions.
- Every number that reaches a doc or a slide is read from a results JSON that records n, seeds,
  corpus hash and model id.

---

## Part B — Work packages

### WP1 — Deterministic core and loop repair (lands first; nothing else starts before its gate)

**Objective.** Make the scripted loop safe to extend and fix the 2026-08-29 failure class:
training that cannot finish inside its walltime, a metric reader that can attach a foreign run,
a stop-loss that pauses silently, and a scheduler whose counters die on restart.

**Deliverables**
| path | kind | summary |
|---|---|---|
| `run_m1_merged_seeds.sh` | modified | env-driven `TRAIN_EPOCHS` (default 60), `TRAIN_TIME_H` (default 10.8 = 0.9×12), `IMGSZ` (640), `PATIENCE` (20), `ITER_NAME` (default `m1_<TIER>_s<seed>_<SLURM_JOB_ID>`); passes them into the strategy dict; writes the job-scoped artifact path into the strategy; defaults reproduce today's command byte-for-byte when no env is set |
| `weed_optimizer_framework/tools/mega_trainer.py` | modified | `train_yolo_mega` accepts `time_h` → Ultralytics `time=` (hours; overrides epochs, ends with a valid `best.pt`); `project=results/framework/mega_iter<ITER_NAME>`, `name=train`, `exist_ok=False`; registers an Ultralytics callback `on_fit_epoch_end` that appends one JSON line per epoch to the trace file (below); records lineage `{base_weights, base_weights_sha256, fresh_start}` in the strategy JSON |
| `weed_optimizer_framework/tools/brain/trace.py` | new | `append(trace_path, record: dict)` (atomic append, `sha_prev` chain), `read(trace_path)`; used by the train callback and by WP6's harvest trace |
| `weed_optimizer_framework/tools/round_scheduler.py` | modified | job-scoped `_train_metric(jobid)`; in-flight trace check for the walltime projection; persisted review state; `superseded` semantics; persisted counters; template-rendered step commands; `_log_action` on every submission; tick duration log; scheduler heartbeat |
| `weed_optimizer_framework/tools/db.py` | modified | `record_round_step` appends to `steps.<step>.attempts[]` and keeps the head entry; new optional fields `params`, `decided_by`, `review`, `su`; `DEFAULT_DOMAIN_CONFIG` gains `steps` templates and `round_params` (see WP7 for the full block) |
| `weed_optimizer_framework/tools/sync_health.py` or new `tools/brain/scheduler_health.py` | new | `GET /api/health/scheduler` (auth-exempt JSON) + banner: `crit` when any enabled domain has `paused_reason` or the scheduler heartbeat is older than 3×`TICK_S`; `warn` when a step has been `awaiting` review longer than the review timeout |
| `weed_optimizer_framework/tools/dashboard_server.py` | modified | `SAFE_PREFIXES` += `s2_dino`, `rndtrain`, `m1_merged`, `brain_`, `llm_`, `supervisor_`, `vllm_`; mount the scheduler health route + banner injection (same pattern as the sync alarm) |
| `.gitignore` | new | `results/framework/_brain/`, `results/framework/_endpoints/`, `results/framework/llm_infer/`, `results/framework/model_deploy/`, `results/framework/supervision_bench/cases/*/artifacts/raw/`, `*.out` |
| `tests/test_round_templates.py` | test | renders the weed step templates with default `round_params` and asserts byte equality with the three literal strings that were in `_WEED_STEPS` |
| `tests/test_scheduler_state.py` | test | restart drill: seed `~/.round_scheduler.json` + a ledger doc with a running step, call `_recover_inflight`, assert `fails`, `rounds_today`, `started` are preserved |

**Interfaces**
- Trace file `results/framework/_brain/<domain>/trace/<round>_<step>_<jobid>.jsonl`, one JSON per
  line: `{"ts","domain","round","step","job_id","kind":"epoch|candidate|decision|report",
  "epoch","map50_95","elapsed_s","eta_total_s","walltime_s","save_dir","sha_prev"}`.
- Job-scoped artifact `results/framework/m1_<tier>_seed<N>_<jobid>.json` (existing writer at the
  end of `run_m1_merged_seeds.sh`; now ALSO written at job start with `status:"running"` and the
  strategy, and updated by the epoch callback with `last_epoch`, `best_map50_95`, `save_dir`).
- Round doc step entry: `{"status","actor","at","job","detail","params":{…},"decided_by":
  "default|rule|advisory|tier1|human","review":{"status":"none|awaiting|applied|timed_out",
  "review_id","queued_at"},"su":float,"attempts":[{status,actor,at,job,detail}]}`.
- `~/.round_scheduler.json` domain block gains `"state":{"job","step","fails","day",
  "rounds_today","started_sacct"}` written on every transition.
- Scheduler heartbeat `results/framework/scheduler_status.json` on the lab `{"ts","tick_s",
  "domains":{…}}`, refreshed every tick.
- Domain config `steps` templates (weed seed, byte-identical to today):
  `collect: "sbatch --time={collect_time_h}:00:00 --export=ALL,BRAIN_MAX_NEW={max_new} run_v3_0_43_brain_harvest_oneshot.sh"`,
  `filter: "sbatch run_s2_dino_scores.sh"`,
  `train: "sbatch --array=1-1 --job-name=rndtrain --gres=gpu:h100-80:1 --time={train_time_h}:00:00 --export=ALL,TIER={tier},MIN_DINO_SCORE={min_dino_score},TRAIN_EPOCHS={epochs},TRAIN_TIME_H={train_time_cap_h},ITER_NAME=rnd{round}_{step}_%j run_m1_merged_seeds.sh"`
  with `round_params` defaults `{collect_time_h:10, max_new:3, tier:"curated",
  min_dino_score:0.50, epochs:60, train_time_h:12, train_time_cap_h:10.8}`.

**Tasks (order matters)**
1. `.gitignore` + trace module + tests scaffold; login-node import test.
2. `run_m1_merged_seeds.sh` / `mega_trainer.py` changes; **verify** on the cluster with a short
   smoke: `sbatch --gres=gpu:v100-32:1 --time=00:40:00 --export=ALL,TIER=curated,MIN_DINO_SCORE=0.50,TRAIN_EPOCHS=200,TRAIN_TIME_H=0.4,ITER_NAME=smoke_%j run_m1_merged_seeds.sh`
   → expected: job ends COMPLETED well before 40 min, `best.pt` exists under
   `mega_itersmoke_<jobid>/train/weights/`, the trace file has ≥ 2 epoch lines, the job-scoped
   JSON has `status:"done"`. (This is the proof that the 08-29 configuration can no longer TIMEOUT.)
3. `db.py` attempts/params/review fields (backward compatible: readers must tolerate old docs).
4. `round_scheduler.py`: (a) `_train_metric(jobid)` reads the job-scoped JSON and its
   `results.csv`; refuses if `job_id` mismatches; (b) in the in-flight branch, every tick read the
   train trace and compute `eta_total_s`; if `eta_total_s > walltime_s` and no `time=` cap is
   active, record signal `walltime_bound` (WP3 will own the detector; WP1 ships the check inline);
   (c) on a failed train, set `review.status="awaiting"` and **do not resubmit** until
   `applied|timed_out` (timeout 90 min → apply the deterministic correction: halve `epochs`, keep
   `train_time_cap_h`); (d) `superseded` outcome for supervisor-initiated `scancel` (does not
   increment `fails`); (e) persist counters; (f) render commands from templates; (g) call
   `_log_action`; (h) heartbeat + tick duration warning above 90 s.
5. Dashboard: `SAFE_PREFIXES`, health route, banner. Deploy (one planned restart).
6. Ollama upgrade already done; **re-deploy `qwen3.8:27b`** (job 45198752 submitted 2026-09-04;
   poll `/api/models/deploy/result?jobtag=d09041724137d66&model=ollama:qwen3.8:27b`).
7. Re-enable weed: `POST /api/rounds/scheduler {"domain":"weed","enabled":true}` only after
   tasks 1–5 are deployed; watch one full scripted round.

**Gate.** Smoke job proves `time=` cap; `test_round_templates` byte-equal; restart drill keeps
counters; one scripted round completes with a job-scoped metric on the ledger; the banner turns
red when a domain is paused (simulate by writing `paused_reason` in the config and restarting the
tick) and green again after re-enable; `smoke_test.sh` green; CHANGELOG v3.25.0.

### WP2 — Incident corpus and benchmark harness

**Objective.** Freeze the raw evidence of every incident before new jobs overwrite it, and build
the harness that scores every arm on the same cases.

**Deliverables**: `tools/brain/corpus.py` (export/list/verify), `tools/brain/bench.py` (run/score/
reproduce), `results/framework/supervision_bench/{cases/<case_id>/,split.json,rubric.md,
results/}`, `tests/test_bench_scoring.py`, `make_figures.py` addition (`supervision_bench.png`).

**Case format** `cases/<case_id>/bundle.json` (the WP3 bundle schema, built from archived
artifacts), `artifacts/` (raw files: `job_<id>.out` with absolute line numbers via
`awk '{printf "%6d\t%s\n", NR, $0}'`, `sacct_<id>.tsv`, `results_<id>.csv`, registry/slug-score
snapshots, harvest trace), `truth.json`:
```
{"case_id","date","incident":true|false,"class":"operational|config|code|design",
 "signals_expected":["walltime_bound",…],"load_bearing_lines":[{"artifact","line"}],
 "acceptable_corrections":[{"action","params_range","risk"}],"escalation_expected":"none|tier1|tier2|human",
 "provenance":"raw|record-only","labels":{"pre_registered":{…},"adjudicated":{…}},"notes":""}
```
**Candidate incidents** (verify artifact survival with `ls results/framework/*<jobid>*` and
`sacct -j <id>`): 2026-08-29 double TIMEOUT 44727703/44767709 (**dev case**); COMPLETED harvest
44322382 booked failed (UNKNOWN gap); round-4 collect TIMEOUT at 4 h; 10 h harvest timeout on
three zips (v3.23.2); M1 raw tier on V100 needing ~75 h (v3.22.7, mid-flight evidence in
`results.csv` epoch times); supervisory-field rollbacks (v3.22.10, v3.24.2 — registry diffs);
pool_report 15,789 vs 120,515 (v3.22.15); audit probe 0/8 (v3.22.17, `sample_audits/*.json`);
class-order permutation (v3.24.4, per-species JSON); sync dead 22 days (v3.24.3, lab heartbeat —
record-only for a cluster bundle); the −0.27 causal claim (v3.23.1) and the Mamba init confound
(v3.22.23) as design-class decision cases; earlier v3.0-era cases from CHANGELOG with job ids
(record-only unless `.out` survives). Healthy controls: completed steps of rounds #1–#3 and the
six sealed M1 runs, each re-audited with WP3 signals before being labelled clean.

**Split** (`split.json`, sha256 committed before any model run): dev = incidents dated
≤ 2026-08-25 + the 2026-08-29 case; test = later incidents, live-run incidents, plus a
leave-one-class-out pass; the dev case never appears in reported numbers.

**Arms** A0 scripted (detection defined in code: the right step recorded failed and the domain
paused within one tick), A0+ signals only, A0++ signals + alarm, L1 status+metrics only, L2 raw
excerpts with signals hidden (signal-blind per case: remove the covering signal from the bundle),
L3 signals + excerpts, L4 L3 + bounded retrieval. Models per arm from WP4.

**Metrics** (`results/<arm>_<model>.json`): recall per class **as margin over A0+**, false-alarm
rate on healthy cases at severity ≥ `warn` (pre-registered), correction correctness on the R1/R2
subset, evidence hit rate (quoted lines ∩ `load_bearing_lines`), citation validity, escalation
appropriateness, SU per review (`sacct` AllocTRES×Elapsed), `tokens_in`; case × arm × repeat
matrix (3 repeats, temperature 0.3), Wilson 95 % intervals, minimum detectable difference.
`bench.py --reproduce` re-scores committed verdict JSONs without any model call.

**Gate.** ≥ 12 raw incident cases + ≥ 10 healthy exported and hashed; `split.json` and
`rubric.md` committed with hashes; A0/A0+/A0++ scored; scrub check (grep for the cluster username, the cluster password, and token prefixes such as `hf_`/`KGAT`
over `cases/` returns nothing) before commit.

### WP3 — Deterministic signals, evidence bundle, citation validator

**Deliverables**: `tools/brain/signals.py`, `tools/brain/evidence.py`, `tools/brain/citations.py`,
`tools/brain/thresholds.json` (pre-registered), `tests/test_signals.py` with fixtures from the
2026-08-29 artifacts, `tests/test_citations.py`.

**Signals** (pure functions `f(bundle) -> list[Signal]`, `Signal = {"signal","severity":
"info|warn|crit","evidence":[{"artifact_id","line","quote"}],"value"}`):
`walltime_bound` (sacct `TIMEOUT` on train; or from the epoch trace/`results.csv` time column
after ≥ 3 rows: `eta_total_s > 0.95×walltime_s` and no `time=` cap), `pool_growth`
(iterations/epoch +20 % vs previous train), `stale_artifact` (artifact mtime < step start),
`gate_noop` (tier curated with `slug_scores.json` missing or older than the round's collect End,
kept == raw, or unscored slugs entering the merge), `source_degraded` (**new** degradation only:
a source with candidates > 0 last round and 0 now; the chronic `SKIPPING github` line is an
environment fact and is reported as `info`), `plateau` (last-3 round spread below 2× the
recipe's sealed seed std: curated 0.005, raw 0.009, cwd12-core 0.006), `job_unknown` (> 3
ticks), `budget` (round SU > `per_round_cap`, campaign > 0.8×envelope), `disk_low` (project
quota headroom < 300 GB — from `df`; filesystem free reported separately), `ownership_violation`
(mirror hash mismatch), `mongo_down`.

**Bundle** (built on the cluster by `evidence.py build --domain --round --step --jobid`, run
inside the scheduler's single batched command; output `results/framework/_brain/<domain>/
evidence/<round>_<step>_<ts>.json`):
```
{"bundle_id","sha256","domain","round","step","built_ts","sections":{
  "ledger":{…last 5 rounds, staged from the lab…},"sacct":[rows],
  "out_tail":{"artifact_id","path","sha256","lines":[[abs_no,text],…]},
  "results_csv":{"rows","best","last","epoch_time_s","mtime"},"strategy":{…job-scoped JSON…},
  "trace":[…last 50 records…],"slug_scores":{"n","unscored","p25","median","p75","mtime"},
  "registry_diff":{…registry_lock.diff since last round…},"harvest":{"per_source":{…}},
  "resources":{"df_project","quota_headroom_gb","fs_free_tb","squeue_depth"},
  "su":{"round","campaign","envelope"},"corrections":[…active…],"plan":{…active…},
  "signals":[…]},"token_estimate":int,"caps":{…per-section…}}
```
Per-section caps with a priority trimmer (sacct/strategy/signals never trimmed; `out_tail`
trimmed to WARN/ERROR/TIMEOUT lines + last 120 lines); `token_estimate` ≤ 12,000 by default and
asserted against the model's `num_ctx` before every call. Lab→cluster staging of ledger/
corrections/plan uses the existing `_domains` base64 pattern, gzip'd, ≤ 96 KB per command.

**Citation validator** `citations.resolve(bundle, quote) -> {"artifact_id","line"} | None`:
quote ≥ 20 chars, whitespace-normalised substring match over the bundle's line texts; findings
whose quotes do not resolve are recorded `rejected_unverifiable` and are never actionable.

**Gate.** Fixtures from the real 08-29 artifacts fire `walltime_bound` + `pool_growth` and
nothing else; healthy fixtures fire nothing above `info`; a bundle built on the cluster for a
live round validates against the schema and its `token_estimate` matches a tokenizer count within
15 %; validator resolves a known quote and rejects a fabricated one.

### WP4 — Supervisor call path and model serving (priority: DeepSeek-V4-Flash vs deepseek-v3:671b)

**Deliverables**: `tools/brain/supervisor.py`, `tools/brain/prompts/{supervisor,planner,
worker}.txt` (English), `run_llm_review.sh` (per-call, generalised from `run_llm_infer.sh`),
`run_vllm_window.sh` (V4-Flash window), `run_ollama_window.sh` (671b window), `run_vllm_verify.sh`,
`tools/model_router.py` (roles + `cluster_window` place + endpoint reachability),
`tools/llm_providers.py` (`num_ctx`, `format=json`, endpoint override, honest `provider_status`),
`dashboard_server.py` (`/api/models/deploy` kind=`vllm` with hardware profile; tag regex admits
`hf.co/` and local HF paths while keeping the `:cloud` rejection), `tests/test_supervisor_schema.py`.

**Verdict schema** (enforced with a JSON schema; Ollama `format=json`, vLLM `response_format`):
```
{"verdict":"ok|issue","findings":[{"signal","quote","diagnosis","severity"}],
 "corrections":[{"action","params","risk","reason","quote"}],
 "escalate":{"to":"none|tier2|human","reason"},"confidence":0..1}
```
Recorded per review: `model_requested, model_used, endpoint_used, tokens_in, tokens_out,
latency_s, su, prompt_sha256, bundle_sha256, rejected_unverifiable_count`.

**Serving shapes**
- *Fast per-call* (`run_llm_review.sh`): `#SBATCH -p GPU-shared --gres=gpu:v100-32:1 --mem=60G
  -t 00:40:00`; env `LLM_MODEL`, `LLM_JOBTAG`, `LLM_NUM_CTX` (default 16384), prompt file
  `results/framework/llm_infer/<jobtag>.prompt`; `OLLAMA_HOST=127.0.0.1:$((11434 + SLURM_JOB_ID % 1000))`
  (per-job port — shared nodes host several jobs); models `glm-4.7-flash`, `qwen3.8:27b`.
- *Deep windows* (file RPC; both interchangeable): the SBATCH main body starts the server, then
  loops: poll `results/framework/_brain/inbox/<role>/*.prompt` every 5 s → POST to the local
  OpenAI-compatible endpoint → write `outbox/<jobtag>.json` → refresh heartbeat
  `results/framework/_endpoints/<role>.json {model, host, port, job_id, started_ts,
  walltime_end_ts, heartbeat_ts, num_ctx, requests_served, idle_since_ts}`; idle-exit after 30
  min; exit 10 min before walltime; `trap` removes the heartbeat.
  - **V4-Flash** `run_vllm_window.sh`: `#SBATCH -p GPU-shared --gres=gpu:h100-80:4 --mem=240G
    --cpus-per-task=16 -t 03:00:00`; `apptainer exec --nv -B /ocean
    /ocean/projects/cis240145p/byler/containers/vllm-openai.sif vllm serve
    /ocean/projects/cis240145p/byler/models/hf/DeepSeek-V4-Flash --tensor-parallel-size 4
    --max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code --host 127.0.0.1
    --port $PORT --served-model-name deepseek-v4-flash`. `VERIFY:` FP4-expert weights load on
    Hopper in vLLM 0.28 (dequant path) — the verify job below settles it; if it fails, add
    `--quantization` per the model card's Hopper note or fall back to `--gres=gpu:h100-80:8`.
    Reasoning modes: run in non-think mode for reviews (deterministic, cheap); expose think mode as
    a per-role option. Cost 8 SU/h; 2 h window ≈ 16 SU.
  - **671b** `run_ollama_window.sh`: `#SBATCH -p GPU --gres=gpu:h100-80:8 --mem=1900G -t 02:30:00`;
    ollama v0.33.3 with `num_ctx 32768` (verify job `v671b_ctx32k` = 45198717, pending
    2026-09-04); ≈ 13 min load; 16 SU/h.
- *Verification job* `run_vllm_verify.sh` (V4-Flash): load, one 12K-token prefill timing,
  tokens/s at 1 and 4 concurrent, a JSON-mode review of the dev bundle; writes
  `results/framework/model_deploy/<jobtag>.json {ok, load_s, prefill_12k_s, tok_s, sample}` and
  registers `vllm:deepseek-v4-flash` in `model_catalog.json` with
  `hardware_profile {gres:"gpu:h100-80:4", mem:"240G", time:"03:00:00", num_ctx:65536, tp:4}`.
- *Scheduler side*: the tick's single batched command writes inbox prompts / submits per-call
  jobs and reads outbox/result files; claim-once (first responder wins; the other is cancelled and
  logged); review timeout falls back to the deterministic correction (WP1).

**Head-to-head protocol** (this is the result the user asked for first): same bundles (test split),
same prompt, same `num_ctx` (32768 for both; V4-Flash gets 65536 only in a labelled extra arm),
3 repeats, temperature 0.3; report recall-over-signals per class, false alarms, evidence hit
rate, correction correctness, latency, SU per review, and the cost–detection frontier; add
glm-4.7-flash and qwen3.8:27b rows for scale.

**Gate.** V4-Flash verify JSON `ok:true` with measured `tok_s`; 671b verify at 32K `ok:true`;
a window answers an inbox prompt end-to-end and the heartbeat is visible from the lab tick;
`run_llm_review.sh` returns a schema-valid verdict for glm and qwen3.8; `provider_status` reports
a window as configured only when its heartbeat is < 180 s old.

### WP5 — Governance: risk tiers, policy gate, approvals, correction channel, SU ledger

**Deliverables**: `tools/brain/policy.py`, `tools/brain/corrections.py`, `tools/brain/su_ledger.py`,
`tools/brain/wal.py`, `db.py` collections and ownership rule, `dashboard_server.py` (risk metadata
on `_CLUSTER_ACTIONS`, `policy.authorize` in `api_cluster_action`, routes), `round_scheduler.py`
(`policy.authorize` in `_submit`), `dataset_discovery.py` (`supervision{}` in the protected list),
`tests/test_policy.py`, `tests/test_corrections_chain.py` (static grep that no worker module opens
the mirror for writing + runtime tamper test), `tests/test_policy_adversarial.py`.

**Policy** `ACTIONS = {action_id: {template, param_bounds, risk, reversible, est_su(params),
dry_run_variant, allowed_tiers}}` covering every `_CLUSTER_ACTIONS` entry and the scheduler
steps. `authorize(actor, action, params, budget_state, resources) -> {allowed, needs_approval,
reasons}`. Tiers: R0 read-only (all); R1 reversible config in bounds (tier-0 proposes, tier-1
applies); R2 compute/corpus state (train submit, cancel/resubmit own step, quarantine with reason
— tier-1 after an artifact-cited verdict; scheduler waits on R2); R3 external side effects,
SU-heavy or design-changing (approval queue); R4 destructive/irreversible (human only). Ceilings:
tier-0 ≤ R1 propose, tier-1 ≤ R2 + propose R3, tier-2 proposes only, human all.

**Collections** `brain_corrections` (append-only, `{seq, prev_hash, hash, domain, author,
review_id, kind, target{key,old,new}, scope{from_round,until_round}, reason, quote, ts}`),
`brain_approvals` (`{domain, action, params, risk, requested_by, review_id, status, decided_by,
decided_at, reason}`), `brain_reviews`, `brain_decisions`, `su_ledger` (`{job, domain, round,
step, actor, gpu_count, gpu_type, elapsed_s, su, sacct_row, ts}`; nightly reconciliation against
`sacct -S <campaign start>`), `scheduler_state`. Single writer: only the lab scheduler thread
writes corrections; the mirror `results/framework/_brain/<domain>/corrections.json` is staged
`chmod 0444`, re-hashed every tick; divergence → `ownership_violation` + pause.
`record_round_step` ownership: an entry authored by `tier1:*` or `human:*` is never overwritten
by `round-scheduler`/`tier0:*` (it becomes a new attempt).

**Routes** `GET /api/brain/{domain}/corrections|approvals|su|decisions|reviews`,
`POST /api/brain/{domain}/approvals/{id} {"decision":"approve|deny","reason"}` (owner/admin via
`_can_manage_agent`), all audited.

**Gate.** Adversarial test: a fake tier-1 verdict proposing an R3 quarantine-delete and an R4
dataset delete → R3 lands in `brain_approvals` as pending, R4 is refused with a ledger entry,
neither appears in `sacct` or `cluster_actions.jsonl`; a job that rewrites the mirror is detected
within one tick and the domain pauses; SU ledger for the last 20 jobs matches `sacct` within 1 %.

### WP6 — Tier-0 trace/advisory, escalation, parallelism, planner, agent-proposed experiments

**Deliverables**: `tools/brain/worker.py`, `tools/brain/escalation.py`, `tools/brain/parallelism.py`,
`tools/brain/planner.py`, `tools/brain/digest.py`, `tools/brain/experiments.py`, `run_experiment_array.sh`,
`db.py` (`brain_plans`, `brain_experiments`), routes `GET/POST /api/brain/{domain}/plan|experiments`,
`tests/test_escalation.py`, `tests/test_parallelism.py`, `tests/test_planner_mock.py`.

- **Worker** (inside step jobs): per-candidate harvest trace lines from
  `dataset_discovery.harvest_new_datasets` (`{source, query, slug, verdict, reason, images,
  labels, elapsed_s}`), the per-epoch train trace (WP1), an advisory `report` record at step end
  from the in-job open model (gemma4, already loaded in the harvest job only — never load a model
  into the train/filter jobs) with a proposal for the next step's parameters inside
  `param_bounds`; `decided_by` provenance; deviations-from-default counted per round.
- **Escalation** (every trigger logged with its type): E1 tier-0→tier-1 = any signal ≥ warn, every
  step end (cheap review), every 4th step (periodic audit); E2 tier-1→tier-2 = same signal recurs
  after two applied corrections, plateau ≥ 3 rounds spanning a recipe change, correction class
  `code|design`, confidence < 0.5 on a ≥ warn finding, budget breach; E3 → human = any R3/R4,
  stop-loss, `ownership_violation`, daily citation validity < 70 % (pauses tier-1 autonomy;
  signals keep running).
- **Parallelism** `plan(domain, proposal, resources) -> {n_parallel, slots, caps, rationale}`:
  caps from SU envelope, `squeue` depth, quota headroom, and the evidence rule (seeds = 3 for any
  claim); collectors = 1 always; audit may run beside training; trainers > 1 only after WP1's
  job-scoped metric; the model chooses within caps and must cite a signal or plan step for n > 1;
  the decision is a ledger row.
- **Planner** `plan(digest) -> Plan {version, hypotheses[], ordered_experiments[{recipe, params,
  seeds, control, est_su, risk, success_criterion, stop_rule}], stop_rules}`; backends `mock`
  (rules from the lever menu; UI label "simulated"), `file` (versioned JSON under
  `results/framework/_brain/<domain>/plans/`), `byok` (any OpenAI-compatible/other provider key
  from the user's encrypted per-user store via `llm_providers.chat`), `cluster:<model>` (WP4
  windows). The digest given to a planner **under evaluation omits the sealed lever table**.
- **Agent-proposed experiments (permanent feature)**: `experiments.propose(domain)` picks from the
  domain's `lever_menu` (weed: source admission by audit precision ≥ 0.90; per-box verification
  gate (consume `verify_scores.json`); core recipe `merged_curated` vs `cwd12_core+audited`;
  `fresh_start` vs continue; COCO pretraining on/off; model family yolo11n/s/m, rfdetr, mamba-t;
  class head 12 vs 100; epochs/time cap), creates an R3 approval item with `est_su`, executes as
  `run_experiment_array.sh` (`--array=1-3`, seeds 101/102/103, sealed holdout, job-scoped
  artifacts), lands `mean±std (n=3)` on `brain_experiments` with the recipe's noise floor and a
  `verdict: better|worse|within_noise`. The first three mock proposals for weed, with controls:
  (1) `cwd12_core + audited sources` vs `merged_curated` (control = cwd12_core alone);
  (2) per-box verification gate on vs off on the same corpus; (3) `fresh_start` vs continue
  from `last_mega_weights` on the same recipe.

**Gate.** One round on weed leaves harvest + train traces with valid `sha_prev` chains; an
advisory proposal is recorded with `decided_by:"advisory"` only when inside bounds; E1/E2/E3 unit
tests; parallelism returns n = 1 with rationale for a routine round and seeds = 3 for a claim;
mock plan v1 is generated from the real digest and labelled simulated; experiment (1) is proposed,
approved via the UI, submitted as a 3-seed array and its result row shows `mean±std (n=3)`.

### WP7 — Platform surfaces and generality (no external agent session required)

**Deliverables**: project-page **Supervision card** (signals now; verdicts with quotes linking to
`/api/job_log/{jobid}` at the absolute line; corrections with actor + reason; escalation markers;
approvals queue with Approve/Deny; plan versions with the simulated label; experiment proposals
table proposed → approved → running → result; per-tier brain-select dropdowns fed by
`/api/models/catalog` filtered to deployed+verified `llm|vllm`), full page `/supervision/{domain}`,
routes `GET /api/brain/{domain}/timeline|signals|roles`, `POST /api/brain/{domain}/roles`
(owner/admin; writes `config.model_routing`), console defaults via the existing
`/api/models/role`; `DEFAULT_DOMAIN_CONFIG` additions in `db.py`:
```
"steps":{collect,filter,train templates},"round_params":{…},
"brain":{"policy":"scripted|rules|hierarchical","tiers":{"worker","fast","deep","planner"},
         "review_timeout_min":90,"periodic_audit_every":4},
"budget":{"su_envelope":1500,"daily_cap":120,"per_round_cap":60},
"lever_menu":[…],"noise_floor":{"recipe":std}
```
so a second domain (`pest_detection`, or a config-only dry run on `laser_cart`) gets the same
card with zero code; `deploy/sync_from_cluster.sh` pulls `results/framework/_brain/` traces;
`docs/SERVER_MAP.md` updated; `tests/smoke_test.sh` additions; `tests/browser_supervision.py`
(Playwright Firefox on the lab, member + admin, mobile + desktop viewports, screenshots to
`~/screens/`); dark theme and layout consistent with the existing agent page.

**Checklist "no external agent session required"**: every tier runs as a platform service
(scheduler thread + cluster jobs); every correction/approval is a web action; every result is a
page and a JSON route; the demo can be driven entirely from the browser.

**Gate.** Real-browser walkthrough screenshots (member and admin, mobile and desktop) show the
card populated from a live round; a second domain renders the card from config only; smoke green.

### WP8 — Evaluation runs, ablations, cost frontier, documentation, freeze

**Run order** (respects queue reality): single-GPU arms first (glm, qwen3.8) on the test split;
V4-Flash window (GPU-shared, 4×H100); 671b window (whole node, submit the evening before with
`--begin`); information ablation L1/L2/L3 on the same cheap model; signal-blind ablation;
tier-0 advisory on/off and retrieval on/off; cost–detection frontier (SU per review vs
recall-over-signals per arm). **Live integration** (labelled n): one scripted round + two
supervised rounds on weed with one injected walltime fault each (`TRAIN_EPOCHS=200` with
`train_time_cap_h` disabled in the fault round only), reporting time-to-detect (20 % vs 100 % of
walltime), SU discarded, the rendered-sbatch diff of the applied correction, and gate refusals
under adversarial injection. **Honest failure section**: false alarms, `rejected_unverifiable`,
disagreements with the record, deployments that failed.

**Documentation**: `docs/TIERED_SUPERVISION.md` (engineering record: what/why/how verified),
`CHANGELOG.md` v3.25.x entries, `RESEARCH_LOG.md` newest-first entry, README status line,
`make_figures.py` additions with byte-stable outputs, `results/framework/figures_data.json` keys
for the Evidence panel. **Freeze checklist**: `git rev-list --count origin/main..main` = 0; smoke
green; attribution grep (`grep -rniE "claude|anthropic|openai|gemini" --include=*.py --include=*.md
--include=*.txt .` returns only the allow-listed provider identifiers in `llm_providers.py`,
`model_router.py`, BYOK code and their docs); secrets scan; English-only scan of UI strings.
Summary-slide sentences may only carry numbers copied from results JSON with n.

**Schedule** (freeze 2026-09-17; trim order Kimi → 671b window → tier-0 advisory → live size;
never corpus/signals/channel/alarm):

| day | work |
|---|---|
| 09-05 | WP1 tasks 1–4; smoke job; WP2 export (before any new job overwrites artifacts) |
| 09-06 | WP1 5–7 (one planned restart, weed re-enabled); WP3 signals + fixtures |
| 09-07 | WP3 bundle + validator; WP2 harness + A0/A0+/A0++; WP4 V4-Flash verify job |
| 09-08 | WP4 fast per-call path + windows; cheap-model arms L1–L3 running |
| 09-09 | WP5 policy/approvals/channel/SU ledger (second planned restart); V4-Flash vs 671b runs |
| 09-10 | WP6 worker trace + escalation + parallelism; shadow mode on weed |
| 09-11 | WP6 planner + experiments loop; act mode (R1/R2); first experiment proposal |
| 09-12 | WP7 Supervision card + page + second domain; browser tests |
| 09-13 | WP8 ablations + live integration start (fault round) |
| 09-14–15 | WP8 runs finish; cost frontier; experiment (1) result lands |
| 09-16 | docs, figures, failure report, freeze checklist |
| 09-17 | freeze; buffer |

## Part C — Status (2026-09-04 evening)

| item | state |
|---|---|
| ollama on the cluster | upgraded 0.20.6 → **v0.33.3** (`bin-0.20.6-bak/`, `lib-0.20.6-bak/` kept) |
| qwen3.8:27b | first deploy failed (404 on the old binary); re-submitted via `/api/models/deploy` → job **45198752**, jobtag `d09041724137d66` |
| DeepSeek-V4-Flash weights | pull job **45198716** running → `/ocean/projects/cis240145p/byler/models/hf/DeepSeek-V4-Flash` (159.6 GB, 46 shards); check `results/framework/v4flash_pull_45198716.out` for `PULL_DONE` and `du -sh` |
| deepseek-v3:671b at num_ctx 32768 | verify job **45198717** (whole node, pending), jobtag `v671b_ctx32k` |
| vLLM apptainer image | built and verified (job 45197071): `/ocean/projects/cis240145p/byler/containers/vllm-openai.sif` |
| weed scheduler | paused by stop-loss since 2026-08-29; re-enable only after WP1 gate |
| ultralytics on the cluster | 8.4.37 — `time` present in DEFAULT_CFG_DICT and `on_fit_epoch_end` is a real callback hook, so the walltime cap and the per-epoch trace in WP1 rest on verified API |
| DeepSeek-V4-Flash checkpoint | staged, 149 GB / 46 shards; `config.json`: `DeepseekV4ForCausalLM`, block-**FP8** (e4m3, 128x128, ue8m0 scales) — native on H100, no dequant path; 43 layers, 256 routed experts / 6 active, 64 heads (divisible by TP=4), native context 65536 with YaRN x16 to 1M |
| vLLM verify job | `run_vllm_verify.sh` written and submitted as job 45225918 (GPU-shared, 4x H100, 240G, 90 min): loads, times a 12K-token prefill, measures batch-1 decode, and tests strict JSON output; writes `results/framework/model_deploy/v4flash_verify1.json` |
| WP1 | **DONE 2026-09-05** — v3.25.0 + v3.25.1 pushed (acbb8ce, a90f57c, d2a7040), deployed to the lab (`smoke_test.sh` 113 passed / 0 failed; six test files pass on the lab venv; `/api/health/scheduler` correctly reads `crit / paused — weed`), published to the cluster (md5 nested == outer, imports verified). Smoke job **45230995** pending: `TRAIN_EPOCHS=200` with `TRAIN_TIME_H=0.4` must end COMPLETED with a valid `best.pt`. The weed domain stays paused until it passes. Two deploy-time findings: the job-scoped artifact would have been counted as a second seed by `make_figures.py` (science error, fixed); `validate_step_command`'s is_file() check refused every template on the always-on server (loop-stopping, replaced by a code-held allow-list) |
| WP2–WP8 | not started |
