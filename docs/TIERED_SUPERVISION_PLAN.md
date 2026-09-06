# Tiered supervision campaign — plan

**Status: DRAFT for review · created 2026-09-04 · supersedes nothing yet (SUPERWEED_PLAN S6 stays open).**
*Companion documents: `DOUBLE_AGENT_SYSTEM.md` (what exists), `SUPERWEED_PLAN.md` (the campaign this
extends), `SCIENCE_AUDIT.md`. Every claim here is either verified from code/logs today or marked
as a hypothesis to be measured. Numbers quoted are the sealed campaign numbers (mean±std, n).*

---

## 0. The finding that motivates this plan

1. **The unattended loop stopped on 2026-08-29 17:53 and nobody was told.** Lab journal:
   `[rounds] weed: step train FAILED (TIMEOUT)` → `weed PAUSED by stop-loss`. Two consecutive
   train steps (jobs 44727703, 44767709) hit the 12 h walltime at epoch 24/60 and 16/60 because
   the merged pool grows every round (6,246 → 8,583 iterations/epoch; 60 epochs now need ~29 h).
   The scheduler *detected* the failure and paused itself, as designed. What it could not do:
   detect early (it waited for the walltime), diagnose (walltime-bound, not a training defect),
   correct (cap epochs or training time), or notify anyone. The same harvest log also carried
   `[net] WARN: SOCKS proxy … SKIPPING github` — a degraded source that nothing reads.
2. **The loop that ran unattended contains no LLM decision.** `tools/round_scheduler.py` fires
   three hardcoded sbatch strings; the collect job is deterministic keyword search plus label gates
   (its only LLM call classifies class-name topics). The LLM tool-caller (`brain.py`/`orchestrator.py`)
   is a research script frozen since 2026-04-21 and unreachable from the platform.
3. **Attribution ledger (from CHANGELOG/RESEARCH_LOG, 2026-03 → 2026-08):** every metric
   improvement (RF-DETR +0.143, COCO pretraining +0.071, leak fix, class-id fix, Mamba control) and
   every retraction (0.910 leak, the −0.27 causal claim) originated with the human supervisor;
   no LLM-proposed round (Qwen2.5-7B, DeepSeek-R1:7b, Gemma 4) ever beat the human seed.
   Agent-originated campaign decisions: **0**. Supervisor corrections in the Aug-2026 campaign:
   ~8 config-level, ~10 code-level, ~5 experimental-design-level, 1 plan-level. Most required
   reading raw evidence (job `.out` tails, `results.csv` mtimes, `sacct` vs `squeue`, registry
   diffs, montages); the dominant defect class is *silent wrongness with plausible output*.
4. **Sealed science bounds the decision space.** Harvested web volume is worth 0.00 to −0.02
   (tier ladder 0.8636/0.8599/0.8614/0.8436); scheduler rounds sit within noise (0.6019/0.5919/
   0.5951, single seed); the DINO gate is a garbage detector, not an accuracy lever (curated 0.5894
   vs raw 0.6032); levers that matter are pretraining (+0.071), architecture (+0.023), source
   admission/quarantine, epochs/walltime, when to stop, which control to run next.

## 1. Objective and the claims that will be measured

Build, on the existing platform loop, an **artifact-grounded, tiered supervision layer** —
tier-0 workers that emit structured traces, a tier-1 open-model supervisor on the cluster that
reads artifacts and writes corrections into a worker-immutable channel, a tier-2 planner
interface — and measure it against the scripted scheduler that already passed a 24 h soak.

| ID | Claim (measurable) | How |
|---|---|---|
| C1 | A replayable benchmark of real supervision incidents from a months-long autonomous pipeline on an HPC cluster can be built with raw artifacts and pre-registered truth labels, plus healthy controls. | `results/framework/supervision_bench/` (§4) |
| C2 | Given deterministic artifact invariants (signals) plus line-addressable excerpts, an open-weight supervisor detects incidents at a measured recall / false-alarm rate, and its **marginal** value over the signals alone is reported per incident class — including zero, if that is the result. | arms in §4.3 |
| C3 | Metric-only supervision (status fields + metrics, the information a worker reports about itself) is blind to the silent-wrongness class; artifact-grounded supervision is not. | information ablation, same model |
| C4 | Invariant-triggered escalation to a larger model dominates confidence-triggered escalation on the cost–detection frontier measured in service units. | cost frontier, §4.4 |
| C5 | Every action an agent takes is risk-tiered under a per-tier ceiling; destructive actions require human approval; proposals refused by the gate are counted (adversarial injection), and every decision carries actor + reason + evidence on the ledger. | governance metrics, §4.5 |

**Non-claims for this milestone:** that the supervisor raises holdout mAP round by round
(rounds are within seed noise by construction); that the tiering *mechanism* is novel (it is a
stock supervisor/worker + cascade pattern — the contribution is the benchmark, the invariants, the
measurement, and the attribution record); that on-cluster immutability is OS-enforced (all jobs run
as one account — tampering is *detected and reverted*, never prevented). The accuracy claim is
done properly after the freeze (§6, post-milestone): agent-proposed experiments run as 3-seed
arrays against the sealed table with the 0.006 (2×seed std) significance rule.

## 2. Standing constraints (inherited from SUPERWEED_PLAN §1, plus)

1. The scripted scheduler stays frozen as the **baseline** and the deterministic fallback; every
   LLM tier is measured as a delta over it and over the no-LLM signals arm.
2. Escalation authority belongs to **deterministic signals over artifacts**, never to a model's
   confidence (the record: a small model rubber-stamped 100 % of crops; a plausible narrative
   survived several human passes).
3. Every supervisor finding must **quote an artifact line verbatim**; the validator resolves the
   quote to an absolute (artifact sha256, line) address; a finding that does not resolve is recorded
   `rejected_unverifiable` and is never actionable.
4. Corrections live in a **single-writer** channel (lab-side scheduler writes, cluster gets a
   hash-chained read-only mirror, re-hashed every tick; divergence pauses the domain).
5. Pre-registration: truth labels, prompts, signal thresholds and the dev/test split are committed
   (hash printed in every result file) **before** any model run on test cases; the 2026-08-29
   bundle is the *dev* case and is excluded from reported numbers.
6. English only; every number mean±std or a Wilson interval with n; repo docs record what changed,
   why, and how it was verified.

## 3. Architecture (home = the platform loop, not `brain.py`)

- **Tier-0 worker (trace + advisory).** Each step job appends a structured JSONL trace on /ocean
  as it runs (`results/framework/_brain/<domain>/trace/<round>_<step>_<jobid>.jsonl`): per
  harvested candidate `{source, query, slug, verdict, reason, images, labels}`, per training
  epoch `{epoch, map50_95, elapsed, eta_vs_walltime}` written from an Ultralytics callback from
  epoch 1 (this also fixes the job-scoped metric and the "TIMEOUT leaves nothing" hole), and a
  `decided_by` provenance on every parameter. The in-job open model (Gemma 4, already loaded in the
  harvest job) may write an *advisory* self-report and a proposal for the next step's parameters
  within pre-registered ranges; it holds no authority and its deviations from the default are
  counted. Reviving the 19-tool SuperBrain is rejected (see §9).
- **Signals (`tools/brain/signals.py`)** — pure functions over artifacts: `walltime_bound`
  (`sacct` TIMEOUT, or projected epochs from the `results.csv` time column after ≥3 rows),
  `pool_growth` (iterations/epoch +20 % since last train → predicts the above), `stale_artifact`
  (mtime < step start), `gate_noop` (curated tier with missing/older `slug_scores.json`,
  kept == raw, unscored slugs entering the merge), `source_degraded` (**new** degradation only —
  a source that yielded last round and yields 0 now; the chronic SOCKS line is an environment fact),
  `plateau` (last-3 spread below 2× the recipe's own seed std), `job_unknown`, `budget`,
  `disk_low` (quota headroom and filesystem free tracked separately), `ownership_violation`.
- **Evidence bundle (`tools/brain/evidence.py`)** — built on the cluster in one batched command:
  ledger tail, `sacct` rows, `.out` tails with absolute line numbers, `results.csv` summary, the
  job-scoped strategy JSON, slug-score stats, registry scorecard/quarantine diff, harvest trace,
  df, SU ledger, active corrections and plan, and the signals as facts. Per-section token caps,
  sha256, `tokens_in` asserted against `num_ctx` on every call (silent truncation is exactly the
  defect class under study).
- **Tier-1 supervisor (`tools/brain/supervisor.py`, `run_llm_review.sh`)** — schema-validated JSON
  verdict `{verdict, findings[{signal, quote, diagnosis, severity}], corrections[{action, params,
  risk, reason}], escalate, confidence}`; one bounded retrieval round from a whitelist; two speeds:
  *fast* = a single-GPU model per step (glm-4.7-flash, deployed; qwen3.8:27b, deployment job
  45197085 submitted 2026-09-04), *deep* = a whole-node windowed batch job draining a file inbox
  (`_brain/inbox/*.prompt` → `outbox/*.json`, heartbeat file, idle-exit, kill switch; no network
  assumptions — login→compute TCP is unverified).
- **Correction channel** — Mongo `brain_corrections` (append-only, hash-chained, actor + reason +
  quote), mirrored read-only to /ocean; registry supervisory fields extended with a
  `supervision{}` block; `record_round_step` gains `attempts[]` and an ownership rule (a tier-1 or
  human entry is never overwritten by the scheduler or a worker); a supervisor-initiated cancel is
  recorded `superseded`, not `failed`, and does not count toward the stop-loss.
- **Risk tiers and ceilings (`tools/brain/policy.py`, one choke point called from
  `api_cluster_action` and `round_scheduler._submit`).** R0 read-only (all tiers) · R1 reversible
  config inside pre-registered ranges (tier-0 proposes, tier-1 applies) · R2 compute or corpus
  state — train submits, cancel/resubmit own step, quarantine with reason (tier-1 after an
  artifact-cited verdict; scheduler waits on R2) · R3 external side effects and SU-heavy or
  design-changing actions (approval queue) · R4 destructive/irreversible — delete data, registry
  whole-write, NEVER_TRAIN/TRUSTED_SLUGS/PASS_BAR/holdout guards, model store deletions, git,
  credentials (human only; agents may file a request). Agents never receive a shell.
  `/api/cancel_job` prefixes extended so a supervisor can actually cancel scheduler jobs.
- **Parallelism policy** — deterministic caps from SU envelope, queue depth, disk and the evidence
  rule (seeds = 3 whenever a comparison is claimed); collectors = 1 always (measured value ≈ 0);
  audits may run beside training; more than one trainer only after the job-scoped metric lands.
  The model chooses within caps and must cite a signal or plan step for n > 1.
- **Tier-2 planner interface (`tools/brain/planner.py`)** — `plan(digest) → Plan{version,
  hypotheses, ordered_experiments[{recipe, params, seeds, control, est_su, risk}], stop_rules}`;
  backends `mock` (rules from the sealed lever table, **labelled simulated** in UI and docs),
  `file` (versioned plan JSON), `byok` (a user's own commercial key through `llm_providers`).
  The digest given to any planner **omits** the sealed lever table when the planner's choice is
  being evaluated (otherwise the evaluation is an answer-key lookup).
- **Scheduler changes** — review state machine persisted on the round doc (`awaiting|applied|
  timed_out`, fallback to the deterministic correction on timeout) so a failed train is never
  resubmitted unchanged before a verdict; in-flight check of the running train's own trace for
  `walltime_bound`; `fails/rounds_today/started` persisted (started from `sacct` Start); step
  commands rendered from domain-config templates with byte-identical defaults; submissions logged
  through `_log_action`; a stop-loss raises a site-wide alarm (same positive-heartbeat semantics as
  `sync_health`) — a paused domain is never silent again.
- **Product surface** — project-page Supervision card: signals, verdicts with quotes linking to
  `/api/job_log` at the absolute line, corrections with actor + reason, escalation markers,
  approval queue, plan versions (simulated label), per-tier brain-select from the deployed catalog.

## 3.1 Why this is a platform layer and not an interactive assistant session

The mechanism — cheap workers, a stronger supervisor, a rarely-consulted planner — is commodity.
It is available today inside general-purpose coding-assistant sessions, which already run
sub-agents and already gate tools behind permissions. This section records why the workload still
needs the layer to live in the platform, so the decision is not re-litigated later. Each point is a
property the campaign demonstrably requires, not a comparison of products.

1. **Unattended continuity is the whole failure mode.** The incident this plan is built around ran
   from 2026-08-29 17:53 to 2026-09-04: two 12 h training jobs killed at the wall, the stop-loss
   correctly holding the domain, and six days in which nothing observed it. Supervision that exists
   only while a session is open cannot cover the interval in which that failure occurred. The
   scheduler thread and the cluster jobs run whether or not anyone is present.

2. **A replayable record, not a transcript.** Corrections are append-only and hash-chained, each
   carrying actor, reason and a verbatim quote; signals, the digest and the parallelism decision are
   pure functions over artifacts with no clock reads, so the same inputs yield the same output. This
   is what makes the attribution ledger and the benchmark possible at all: a decision taken in
   August can be re-scored in September against the same bundle. A session transcript is a
   narrative of what happened, not an input a later evaluation can replay.

3. **The unit of authorisation is a priced action, not a permitted tool.** Every action carries
   parameter bounds, a risk tier, an SU estimate and the list of tiers allowed to take it, checked
   at one choke point that both the web API and the scheduler pass through. Tool-level permission
   cannot express "this actor may resubmit its own training step within this budget, may file a
   request to quarantine a source, and may never delete a dataset."

4. **Supervision has to run where the data is.** The evidence bundle is assembled on the cluster in
   one batched command, and the deep model tier is a windowed batch job draining a file inbox,
   because compute nodes reach neither the lab database nor a dependable outbound network. A
   reviewer that must pull artifacts back to a workstation cannot read a running job's trace or a
   780 MB corpus, and would be measuring what it could fetch rather than what happened.

5. **The claim has to be measurable.** 162 sealed cases, a dev/test split fixed by hash, and a
   rubric committed before any model was run; the claim is a margin over a signals-only arm on that
   corpus, with a stated falsification condition. Ad-hoc supervision, however good, produces no
   number that can be compared or refuted.

6. **A second domain must cost configuration, not code.** Steps, round parameters, tier
   assignments, budget and noise floor are per-domain config, so another collection domain gets the
   same loop, the same governance and the same page with no code change.

7. **What is not claimed.** None of the above makes the tiering itself novel, and the layer is not
   claimed to raise detection accuracy round over round — the campaign's own rounds sit inside the
   sealed noise floor. The contribution under test is the artifact-grounded, governed and replayable
   instance of a known mechanism on a real months-long pipeline, plus the benchmark that makes its
   escalation rule falsifiable. An interactive session remains the right tool for writing this code
   and for one-off diagnosis; it is not the right tool for being the layer.

## 4. Evaluation protocol (pre-registered)

4.1 **Corpus.** Every incident in CHANGELOG/RESEARCH_LOG with surviving raw artifacts on /ocean
(job `.out`, `sacct`, `results.csv`, registry/slug-score snapshots, harvest traces produced
*before* the human diagnosis) becomes a bundle; incidents without raw artifacts are labelled
`record-only` and reported separately, never reconstructed from prose. Healthy controls: completed
steps of rounds #1–#3 and the six sealed M1 runs, re-audited for every signal before being called
clean. Target ≥ 30 incidents (raw + record-only) and ≥ 15 healthy; two label columns
(pre-registered, adjudicated) with a second rater on detect/no-detect and correction class.
4.2 **Split.** Dev = incidents dated ≤ 2026-08-25 plus the 2026-08-29 bundle; test = later
incidents, live-run incidents, and a leave-one-class-out pass over the dev classes; signal-blind
arm: on case *k* the signal that covers it is removed from the bundle.
4.3 **Arms.** A0 scripted scheduler (detection defined in code: right step recorded failed and
domain paused within one tick) · A0+ signals only · A0++ signals + alarm (the cheapest fair
competitor) · L1 LLM with status fields + metrics only · L2 LLM with raw excerpts, signals hidden ·
L3 LLM with signals + excerpts · L4 L3 + bounded retrieval. Models: glm-4.7-flash, qwen3.8:27b;
deepseek-v3:671b (windowed) and Kimi K2.6 (optional, §5) on L3 only.
4.4 **Metrics.** Detection recall per incident class (config/code/design) as a *margin over A0+*;
false-alarm rate on healthy cases at a pre-registered severity; correction correctness on the
R1/R2-correctable subset only; evidence hit rate (quote ∩ load-bearing lines) and citation
validity; escalation appropriateness; SU per review from `sacct`; case × arm × repeat matrix with
Wilson intervals and the minimum detectable difference stated on the slide.
4.5 **Live integration (small, labelled n).** One scripted round and two supervised rounds on the
weed domain with one injected walltime fault each; report time-to-detect (20 % vs 100 % of
walltime), SU discarded, whether the correction changed the next rendered sbatch line, and
proposals refused by the gate per tier (adversarial injection of R3/R4 proposals). Baseline row
states the baseline's own record: 24 h soak passed, then a 6-day silent pause.

## 4.9 Amendment (2026-09-06): recall is reported against two denominators

Running the deterministic checks over the frozen corpus showed that most of the
sections they read did not survive as archived artifacts. Across 162 exported cases:
`out_tail` is present in 98, `sacct` in 36, `results_csv` in 17, `strategy` in 10,
`trace` in 7, `slug_scores` in 4, and `ledger`, `harvest`, `su`, `resources`,
`corrections`, `plan`, `registry_diff` are empty in **all** of them. Of the 71 cases
carrying a `signals_expected` label, 75 expected firings are unreachable and only
`walltime_bound` fires at all.

The cause is archival, not a defect in the rules. A per-round ledger snapshot and
an SU reading live in the lab database, which compute nodes cannot reach; a `df`
reading was never captured per job; the correction channel and the plan did not
exist before this week. `signals_expected` states what a supervisor watching that
incident live should have seen, which is a claim about the campaign, and it stays
as written — it is pre-registration.

What changes is the reporting. Every recall is stated against two denominators:

* **over expected** — how much of the campaign's failure record a detector set
  covers. This is the number that says how far the deterministic arm can go.
* **over reachable** — the same measure restricted to cases whose archived bundle
  carries the sections the detector reads. This is the number that compares arms,
  because an arm cannot be credited or debited for evidence no arm was shown.

Reporting only the first understates the deterministic arm and inflates every model
arm's margin over it, which is the headline comparison; reporting only the second
hides how much of the campaign is out of reach. `bench.py reachability` computes
both and names every unreachable case, and a signal counts as reachable exactly
when its own detector returns something other than `unknown` on that bundle — the
detector's answer, not a second table of which sections each signal reads, because
a second table would drift from the first.

This also sets an honest ceiling for the live layer: six of the twelve checks have
no historical case they can run on, so their value is asserted from the design and
will be measured only on rounds recorded from now on, where the bundle is built by
`evidence.py` and carries every section.

## 5. Models and serving (cluster facts verified 2026-09-04)

- H100 nodes w001–w010: 8×80 GB HBM, ~2 TB RAM, no local disk; all allocated today, 8 full-node
  requests pending. GPU-shared V100-32 flows within minutes–hours. SU balance 11,005 (H100 = 2
  SU/GPU-h, V100 = 1). Disk: `df` on the project directory reads 5.9 of 6.9 TB used — the lab's
  shared project quota has **~1.1 TB of headroom** (the `projects` summary reports a lower usage
  figure; the live `df` number is the binding one, and exhausting it breaks every job in the
  project); the ollama store holds 442 GB (deepseek-v3:671b 376 GB, never used as a brain).
  vLLM not installed; apptainer available (probe job 45197071 pulls the image). HF download from a
  compute node measured **268 MB/s** (a 9.8 GB shard in 37 s), so a 595 GB checkpoint pulls in
  ~40 min — download is not the constraint; quota, HBM fit, queue wait and the verify job are.
- **Tier-1 fast (dependable):** glm-4.7-flash (30B-A3B, deployed) and qwen3.8:27b (18 GB Q4,
  256K ctx, Apache-2.0, Ollama library tag; deploying) on one V100 — ~0.1–0.3 SU per review.
- **Tier-1 deep (windowed):** deepseek-v3:671b (on disk; re-verify at `num_ctx` ≥ 32K; 8×H100,
  `--mem=1900G`, ~13 min load) as a 2 h file-RPC window, ~32 SU per window.
- **Kimi, honestly.** K3 (2.8T, MXFP4-only, 1,561 GB, custom licence) cannot be deployed: it
  exceeds one node's HBM by 2.4×, exceeds the free disk, and runs at ~5.8 tok/s on 32 H100s
  through a dequant path; the Ollama tag is cloud-only. K2.6 / K2.7-Code (1T, INT4 595 GB) fit one
  node only at 2–16K context, batch 1, via vLLM (not installed); the only Ollama path is the
  Unsloth GGUF (UD-Q2_K_XL 340 GB, UD-Q4_K_XL 584 GB) — Q4 would take about half of the project
  quota's remaining headroom unless deepseek-v3:671b is deleted first (an R4 decision), Q2 quality is unmeasured, the
  cluster Ollama's K2 architecture support and multi-GPU throughput are unverified, and at n ≈ 30
  the benchmark cannot separate a 1T model from a 27B one unless the gap exceeds ~5 cases.
  **Decision:** Kimi K2.6 is an *optional* benchmark arm with a go/no-go on 2026-09-11 (pull split
  from verify: download in a long V100 job, verify in a short 8×H100 job); nothing in the plan
  depends on it. Post-milestone: multi-node vLLM/SGLang or KTransformers for native INT4 at full
  context.

## 6. Phases and gates (freeze 2026-09-17; rehearsal-only afterwards)

| Phase | Dates | Deliverable | Gate |
|---|---|---|---|
| **W0 deterministic core + corpus freeze** | 09-04 → 09-06 | Export raw artifacts of every incident *before any new job overwrites them*; Ultralytics `time=` cap + env-driven epochs/walltime/ITER_NAME + per-epoch callback trace in `run_m1_merged_seeds.sh`/`mega_trainer.py`; job-scoped `_train_metric`; review state machine, `superseded` semantics, persisted counters, `attempts[]`, `.gitignore` for staged state, `SAFE_PREFIXES`, stop-loss alarm; weed domain re-enabled with the fix; qwen3.8:27b verified | replaying the 08-29 artifacts fires `walltime_bound` + `pool_growth`; a restart drill keeps the stop-loss counter; one scripted round completes with a job-scoped metric; the alarm banner appears when a domain pauses |
| **W1 signals, bundle, supervisor, benchmark** | 09-06 → 09-09 | `tools/brain/{signals,evidence,supervisor,bench}.py`, `run_llm_review.sh` (num_ctx, gres/mem/time, JSON mode, file prompt), citation validator, pre-registered corpus + split + rubric committed with hashes, A0/A0+/A0++/L1–L3 on glm-4.7-flash and qwen3.8:27b | benchmark table with Wilson intervals for ≥ 5 arms × 2 models; LLM reported as margin over A0+; imports verified on the login node before every sbatch |
| **W2 governance, channel, tiers, live shadow→act** | 09-09 → 09-13 | policy gate + risk metadata + approvals UI; single-writer correction channel + mirror hash; tier-0 trace + advisory; planner interface (mock/file/byok, simulated label); SU ledger; Supervision card; shadow mode then act mode (R1/R2) on weed; deep arm windows via file RPC; Kimi go/no-go | an injected R4 proposal is refused with a ledger entry and never reaches `sacct`; a correction visibly changes the next rendered sbatch line; mirror tampering is detected within one tick; ≥ 1 supervised round completes |
| **W3 ablations, cost frontier, failure report, docs** | 09-13 → 09-17 | signal-blind and information ablations; cost–detection frontier; honest failure section; `docs/TIERED_SUPERVISION.md` engineering record; CHANGELOG/RESEARCH_LOG; figures via `make_figures.py`; real-browser verification of every surface | every number traces to a results JSON with n and the corpus hash; `--reproduce` re-scores committed verdicts; `git rev-list origin/main..main == 0`; no attribution strings in the repo |
| **Post-milestone** | 09-21 → | agent-proposed experiments as 3-seed arrays against the sealed table (the accuracy claim done properly); real tier-2 via a user key; second domain; Kimi at full context; benchmark release; paper | first agent-proposed experiment reported mean±std against its control |

Trim order if late: Kimi arm → deep arm → tier-0 advisory LLM → live integration size. Never the
corpus, the signals, the correction channel, or the alarm.

## 7. Budget

Benchmark on single-GPU models ≈ 10 SU; deep windows ≤ 5 × 32 SU; live rounds ≈ 35 SU each
(10 h V100 collect + 12 h H100 train); optional Kimi ≈ 60 SU pull/verify + 32 SU per window.
Milestone total ≤ 700 SU of 11,005 (≈ 6 % of the balance, within the campaign's 4,000 SU envelope
of which ≈ 215 is spent). Disk: traces/bundles are KB-scale; the only large item is the optional
Kimi store.

## 8. Risks

Whole-node queue (a window may wait a day → reviews fall back to the fast tier, `model_used`
recorded) · disk on a shared filesystem (a partial pull leaves blobs; never pull from an automated
path) · answer-key circularity (time split + signal-blind arm + hashes committed first) ·
small-n (report intervals and the minimum detectable difference; no significance claims) ·
yes-to-everything supervisors (false-alarm rate on healthy cases at a fixed severity) · SSH
throttle (everything rides the 120 s tick's single batched command) · git reset at job start
(all staged state is untracked; every new script pushed before its first sbatch) · single
account (detect-and-revert, stated plainly) · Mongo outage (write-ahead JSONL; refuse to submit
when the ledger write is not acknowledged) · calendar (the trim order above).

## 9. Rejected alternatives

Reviving `brain.py`/`orchestrator.py` (frozen, 6×500-char window, reasoning discarded, five
fallback layers, 20–30× loops on record) · an always-on whole-node endpoint (384 SU/day, cluster
is not persistent) · Kimi K3 (does not fit HBM or disk) · Kimi K2.6 via vLLM before the freeze
(not installed; 2K context) · free-form tool loops for any tier holding cluster permissions ·
filesystem-permission immutability (one account) · cross-node Mongo daemon (proven once, adds a
moving part) · metrics-only supervision (blind to the dominant defect class; kept as an ablation) ·
harvest volume/queries as the supervisor's lever (measured ≈ 0) · a planner that reads the sealed
lever table while being evaluated (answer-key lookup) · live mAP A/B as the headline (within
noise; ~1 round/day) · "0 unsafe actions" as a result (it is a unit test; report refusals).

## 10. Status

| Phase | State |
|---|---|
| W0 | not started — execution contract in `TIERED_SUPERVISION_EXECUTION.md` (WP1–WP8); 2026-09-04 evening: ollama upgraded to v0.33.3, vLLM image verified, DeepSeek-V4-Flash pull (45198716), 671b 32K verify (45198717), qwen3.8:27b deploy (45198752) submitted; weed still paused by stop-loss |
| W1–W3 | not started |
