# SuperWeed campaign — master execution plan

**Status: ACTIVE · created 2026-08-22 · supersedes ad-hoc round-running.**
*The operating plan for reviving the double-agent loop at scale: autonomous web
collection → curation → multi-method training, run as a platform project, to a
reviewer-proof standard. Companion documents: `DOUBLE_AGENT_SYSTEM.md` (what exists),
`SCIENCE_AUDIT.md` (what is defensible + milestones M0–M4, which this plan operationalizes).
Live progress is tracked per supervision tick outside the repo; this file defines the
phases, gates, budgets, and protocols, and its Status lines are updated as phases close.*

---

## 0. Objective and success criteria

Build, with the platform's own Collector⇄Trainer loop, a **large, high-precision,
provenance-tracked weed detection dataset** and the **best model the data supports**,
with every claim reproducible.

| ID | Deliverable | Acceptance |
|---|---|---|
| D1 | **SuperWeed dataset**: ≥50,000 (stretch ≥150,000) unique **labelled** weed-relevant detection images (labelled is the number that trains — the merge consumes nothing else; counted from disk, not from the `local_labeled` registry field, which legacy entries never set), deduped (dHash), holdout-guarded, license-tracked, canonical-taxonomy labels | registry scorecard per source; sample-audit precision ≥90% on audited batches; 0 holdout content matches (`skipped_holdout_hash` reported) |
| D2 | **Model suite**: best honest cwd12 holdout mAP50-95 + cross-dataset generalization, over ≥3 method families (YOLO11 n/s/m; RF-DETR; Mamba-YOLO-T) ± WBF/TTA ceiling | every run: pinned config + seed; headline = mean±std over ≥3 seeds, never best-of-N |
| D3 | **Platform-native operation**: the whole campaign runs as the `weed` project — agent cards fire the jobs, rounds ledger records them, a compounding chart (round № vs holdout metric) renders on the project page | a stranger can read the project page and see what ran, when, and what it scored |
| D4 | **Reproducibility package**: `make_figures.py` regenerates every table/figure from checked-in artifacts | one command, no hand edits |

**Non-goals (this campaign):** robot-modality collection (already shipped, v3.20–3.22;
not the bottleneck), closed-model APIs in the loop, UI redesign beyond the chart card.

## 1. Standing constraints (violating any of these is a defect)

1. **Holdout sanctity** — cwd12 test/valid never trains: NEVER_TRAIN slugs + content
   dHash pre-seed (post-`bf621ce`). Every merge reports `skipped_holdout_hash`.
2. **Honesty** — a job that failed is FAILED (no `ok:true` on stubs); simulated events
   carry `meta.simulated`; outages surface as errors, not empty lists. Log-verified,
   not registry-claimed: every "success" is confirmed from the job's `.out`.
3. **Registry discipline** — locked writes only (`registry_lock`); snapshot backup
   before the campaign's first write; corrupt file = stop-and-restore, never rebuild-empty.
4. **Cluster etiquette** — imports verified on the login node before any sbatch; one
   SSH session batches many commands (throttle risk); daemons in SBATCH main body;
   nested/outer package rsync at job start; no parallel checkpoint writers.
5. **Budget** — campaign envelope **≤4,000 SU** of the remaining allocation; per-phase
   sub-envelopes below; V100 (1 SU/GPU-h) by default, H100 (2 SU/GPU-h) only when a
   run measurably needs it. SU spend is checked at every supervision tick.
6. **Data governance** — every registered dataset carries `provenance` incl. license;
   `unspecified` license ⇒ usable for training experiments, excluded from any
   redistributed/displayed figure set.
7. **Platform-first** — when a campaign action has a platform control (agent card,
   round endpoint), the campaign uses it; if the control is broken, fixing it is part
   of the phase (that is how the platform stays a working product).

## 2. Phases

### S0 — Baseline, hygiene, and the honest number (gate for everything)
*Budget: ≤400 SU · wall-clock heavy → starts first.*
- [ ] Cluster state re-audit in ONE SSH batch: registry integrity + backup, dataset
      inventory (real counts, not memory), `/ocean` usage, SU balance snapshot,
      login-node import check of Job-D/Job-T entrypoints on current `main`.
- [ ] Secrets: Kaggle token rotated (owner action) → `~/.kaggle_token` on cluster;
      HF token present; confirm no secret in tracked files.
- [ ] `requirements.lock` from the cluster env, committed.
- [ ] **M1 re-measurement submitted**: curated v3.0.38 recipe, post-guard merge,
      3 seeds × fixed hyperparameters; report mean±std + per-species +
      `skipped_holdout_hash`. *This number replaces 0.9033 everywhere.*
- **Gate:** M1 results published in `RESEARCH_LOG.md`; registry backup exists; tokens live.

### S1 — Collector revival (web harvest, supervised)
*Budget: ≤300 SU (harvest jobs are cheap; curation GPU passes moderate).*
- [ ] Fire `brain_harvest` rounds through the platform action with quality gates ON
      (topic relevance, detection-task filter, dedup, license capture).
- [ ] Per-round scorecard appended to the registry entry: images found / unique-new /
      weed-relevant % / label-source / license mix.
- [ ] **Garbage quarantine** (не delete): off-goal sources (disease-leaf classification,
      near-duplicate re-exports, watermarked stock) flagged `quarantined` with reason —
      the platform lists them greyed-out.
- [ ] Label precision sample-audit protocol: every new source ≥25 images sampled,
      OWLv2 conf histogram + DINO verifier score + montage saved to the project;
      source enters TRAIN pool only at audited precision ≥90% (else re-label at
      conf 0.30+ / consensus, or quarantine).
- [ ] Brain-quality lever (optional): route harvest planning through a stronger
      deployed cluster model if the small brain's query diversity plateaus.
- **Gate:** +10,000 unique audited-pool images with scorecards, zero registry incidents.

### S2 — Curation hardening (the "no garbage" phase)
*Budget: ≤300 SU.*
- [ ] Canonical taxonomy pass: every pooled source mapped to the canonical class set
      (cwd12 + aux slots); unmappable → aux, never silently dropped.
- [ ] Cross-source dedup sweep (dHash across the whole pool + holdout), boxes sanity
      (area/aspect outliers), small-object flag (known failure mode: tiny boxes
      degrade mAP — keep but tag).
- [ ] DINOv2 quality scoring on the full pool; per-source quality distribution into
      the scorecard; bottom tail quarantined with reason.
- [ ] Platform: dataset pages show scorecard + audit montage (existing analysis cards).
- **Gate:** pool report = size, class balance, quality histogram, license mix — regenerable by script.

### S3 — Trainer campaign (multi-method, budgeted)
*Budget: ≤2,000 SU.*
- [ ] Recipe matrix, each run pinned (config+seed) and ledgered:
      baseline (cwd12-only) → +tiers of curated pool (10K / 25K / 50K / 100K+ as S1/S2
      deliver) × {YOLO11n, YOLO11s/m} × {plain, class-balanced oversampling,
      anti-forgetting schedule}; then RF-DETR and **Mamba-YOLO-T** (env build risk:
      `selective_scan` CUDA ext — login-node compile check first); WBF/TTA ensemble
      for the ceiling number.
- [ ] The **scale-vs-quality curve** is re-plotted as tiers land (the campaign's
      signature figure; historical points from `DOUBLE_AGENT_SYSTEM.md` §4 included).
- [ ] Every run's holdout eval via the sealed protocol; failures analyzed
      (per-species drops, small-object subset) before the next recipe is chosen.
- **Gate:** ≥12 ledgered runs spanning ≥3 method families; a current best-model card
  (honest mean±std) + the updated curve.

### S4 — Continuous double-agent mode (unattended)
*Budget: ≤600 SU for the soak.*
- [ ] Scheduler: per-project unattended round cycling (harvest → curate → train →
      eval → repeat) built on the existing `/api/domain/round/*` + rounds ledger;
      SU/time guards per round; stop-loss on failing gates.
- [ ] Project page: **compounding chart card** (round № vs holdout metric + pool size),
      fed by the rounds ledger.
- [ ] **Soak gate:** left alone ≥24 h, the weed project completes ≥2 full rounds with
      real jobs, the chart advances, zero fake-success entries (log-verified).
- **Gate:** the soak passes and the chart is on the page.

### S5 — Results package
*Budget: 0 SU.*
- [ ] `make_figures.py`: benchmark table, scale-vs-quality curve, per-species table,
      anti-forgetting figure, compounding chart export — all from checked-in artifacts.
- [ ] `RESEARCH_LOG.md` headline entry: the honest numbers, with protocol references.
- **Gate:** clean checkout → one command → all assets byte-stable.

### S6 — Integration polish (ties back to M4)
- [ ] Best SuperWeed model deployed via the model gateway; detection overlay available
      on the robot live view (modality 2 rejoins here).
- [ ] Project pages of `weed`, `241_robot`, `laser_cart` read as one coherent story.

## 3. Supervision protocol (the loop that keeps 24/7 jobs honest)

The cluster agents run unattended; a **supervision tick** runs at least every few
hours (and at every conversation turn):

1. **Read state** — tracker note + `squeue/sacct` + tail of active job `.out`s +
   rounds ledger, in one batched SSH.
2. **Verify honesty** — claimed successes cross-checked against logs; any
   "done-but-empty" artifact is treated as a failure and investigated.
3. **Quality gates** — newest data batch: sample audit numbers, quarantine ratio,
   dedup/holdout counters.
4. **Budget & health** — SU spent vs phase envelope; `/ocean` headroom; queue state.
5. **Correct** — kill/requeue misbehaving jobs, adjust configs (documented in the
   tracker), quarantine bad sources; a gate violation stops the phase, not the campaign.
6. **Record** — tracker updated (state, spend, next-check time); repo docs only change
   at phase boundaries.

Interrupt authority is total: a run that violates §1 is cancelled on sight, whatever
its metric says.

## 4. Risk register

| Risk | Mitigation |
|---|---|
| SSH login throttling | serialize + batch; ≥30–60 min backoff on `Connection closed` |
| Registry corruption / lost updates | locked writes; pre-campaign backup; restore-not-rebuild |
| Garbage-data floods (seen before: disease-leaf, re-exports) | S1 gates + quarantine-with-reason; nothing silently deleted |
| Pseudo-label noise at scale (proven −0.27 effect) | audited-pool admission; tiered S3 shows the curve instead of hiding it |
| Holdout contamination | content dHash guard + `skipped_holdout_hash` in every merge log |
| SU overrun | per-phase envelopes; tick-level spend check; stop-loss in S4 scheduler |
| Cluster package drift (nested/outer copy) | rsync at job start (existing pattern) |
| `selective_scan`/Mamba build failure | login-node compile gate before any GPU submission |
| Queue contention | V100 default; H100 only with a measured reason |

## 5. Status

| Phase | State | Closed on |
|---|---|---|
| S0 | **CLOSED** — audit+backup ✅ tokens ✅ requirements.lock ✅ M1 sealed re-measurement published (raw 0.6032±0.0046, curated 0.5894±0.0025, guard verified, RESEARCH_LOG 2026-08-23) | 2026-08-23 |
| S1 | in progress — 2 harvest rounds (+8,208 labelled); gate hardened (label gate default-on, subject check, license at collection); scorecards ✅, sample-audit sweep running (job 44294276); pool = 120,515 labelled / 50 sources | — |
| S2 | **CLOSED** — `tools/pool_report.py` emits the pool report (size, class balance, DINO histogram, license mix, quarantine list) and writes per-source scorecards into the registry | 2026-08-23 |
| S3 | **CLOSED** — 20 ledgered runs across 3 method families with error bars (RF-DETR 0.8974±0.0040 · YOLO11n pretrained 0.8759±0.0030 / scratch 0.8041±0.0028 · Mamba-YOLO-T 0.8266±0.0064), tier ladder complete (0.8636→0.8599→0.8614→0.8436), levers ranked (pretraining +0.071 > architecture +0.023 > head −0.012 ≈ +40k images −0.020), and `docs/BEST_MODEL_CARD.md` published with per-species mean±std from an independent re-evaluation. Not executed: WBF/TTA ceiling (recorded as a limit) | 2026-08-25 |
| S4 | **CLOSED** — soak gate met: rounds #1 and #2 fully closed unattended (collect→filter→train→eval, real job ids, fresh-artifact metrics 0.6019 / 0.5919 on the ledger the chart reads); round #3 self-started. Zero fake-success entries | 2026-08-24 |
| S5 | **CLOSED** — two independent runs from clean copies produce byte-identical output for all five artifacts (md5 0391ee13…) | 2026-08-23 |
| S6 | **in progress** — the campaign's model is deployed on the lab server (RTX 3060, `~/models/cwd12_yolo11n_s102.pt`) and served at `/api/detect/*`: model card metadata, JPEG→JSON detections (57–148 ms), and `/api/detect/frame/{sid}.jpg` which annotates the newest live robot frame. Per-species reliability from the model card travels with every prediction. Remaining: wire the annotated frame into the live view, then measure the domain gap on real robot frames | — |
