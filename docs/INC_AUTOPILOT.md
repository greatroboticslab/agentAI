# INC autopilot: the platform runs INC campaigns and chooses next steps

Contract for `weed_optimizer_framework/tools/inc_autopilot/`. Companion to docs/INCREMENTAL_PROTOCOL.md and docs/INCREMENTAL_PROTOCOL_RUNNER.md. Every experiment of an INC campaign is built, advanced, diagnosed and followed up by the platform; the per-increment accept/reject decision stays in the pinned gate.

The design rests on a read of the dashboard, the round scheduler, the brain package and the INC package, plus direct checks: pilot_v1's `report.json`, `exp.json` and `ledger.jsonl` locally. I also checked `driver.py:223-224` (the pinned module list), `pilot.py:265-269` (an experiment is built once), `policy.py:98-103,403-409`, `report.py:56`, `run_inc_audit.sh:26-37` and `INCREMENTAL_PROTOCOL.md:190-270`. Paths are relative to `weed_llm_benchmark/weed_optimizer_framework/tools/` unless stated.

## 0. Summary and honest limits

- **What already exists.** The platform has a governed skeleton that nothing drives:
  - `policy.authorize`, `approvals`, `planner.make_experiment`, `experiments.verdict`, `citations`, `supervisor.OpenAICompatClient`, `run_llm_review.sh`;
  - the round-scheduler thread and `slurm_sh`.
- **What INC already has.** Its own per-experiment scheduler (`inc/driver.py advance`) and a structured record of each decision (`state.json`, `ledger.jsonl`, `report.json`).
- **What the autopilot adds.**
  - An INC evidence reader that never reads test values.
  - Deterministic diagnoses that map to a pre-registered lever menu. Every lever is an existing builder CLI flag.
  - A campaign ticker on the lab: advance, report, diagnose, propose, gate, execute.
  - One executor for approved items.
  - An advisory research brain that runs as a cluster sbatch job.
  - A page.
  - Nothing inside a running experiment changes. The per-increment accept/reject gate stays in the pinned driver and gate.
- **Hard truths:**
  1. The four human interventions are four data points. Rules that replay them prove only that they reproduce those cases, not that they generalise. The one real test of generalisation is **prospective**: intervention 4 (choosing the real-loop recipe) is still pending and needs pilot_v2's report. The rule's output must be frozen in git before a human decides.
  2. Two of the four human concerns have **no lever that exists today**:
     - the leaf-disease half of intervention 2 (`relevance.py:36-40,86-94` reports leaf-disease matches but never enforces them);
     - making audit results feed attribution.
     The platform can detect both and escalate them. It cannot fix them without new code and a new protocol version.
  3. The evidence for intervention 2 (`step1/*`) exists only on /ocean. The replay fixture has to be pulled once.
  4. "Knows everything" is not reachable. Section (c) says what the design does for unknown situations, and it is bounded.

---

## (a) Components and where each runs

| # | Component (new unless marked) | Runs on | Role |
|---|---|---|---|
| 1 | `inc_autopilot/evidence.py` | lab (pure Python) | Loads a snapshot: exp, state, ledger, report, build_summary, step-1 aggregates, relevance, audit. Keeps **only the `dev` exam** for decisions (§d). Gives every value a citable address: ledger line number, or JSON pointer plus value. |
| 2 | `inc_autopilot/diagnose.py` + `inc_autopilot/thresholds.json` | lab | Pure `detect(evidence, thresholds) → [Diagnosis{id, fired, severity, cites[], summary}]`. Each threshold carries its reason in the JSON, following the existing thresholds.json convention. Nothing is hardcoded as a fallback. |
| 3 | `inc_autopilot/levers.json` + `levers.py` | lab | The pre-registered INC lever menu. Maps diagnosis to lever to exact builder argv, plus control, success criterion, falsifier, literature ids and a cost formula. |
| 4 | `inc_autopilot/remote.py` | cluster **login node**, `bench` env, nested (git-tracked) copy | Fixed verbs only; each prints one marked JSON line, the same way `TRAINMETRIC` does (RS:754-806). Details below. |
| 5 | `run_inc_build.sh` | cluster, CPU sbatch (RM-shared) | Runs `inc.pilot build…` or `inc.realloop build…` (which call `driver init`) away from the login node, which kills long nohup processes (DS:13308-13309). |
| 6 | `inc_autopilot/campaign.py` | lab, inside the round-scheduler thread | A ~20-line hook in `round_scheduler._loop` (RS:1590-1599) calls `campaign.tick()` every 5th tick (600 s, matching the driver's `WATCH_INTERVAL`). It reuses `_LOCK`, the atomic `~/.round_scheduler.json` (new `campaigns` key), `_pause` and stop-loss, the heartbeat and `slurm_sh`. It sends **one batched ssh per tick**, because the Bridges-2 login throttles repeated connections. |
| 7 | `inc_autopilot/executor.py` + an `executed` record type in `brain/approvals.py` | lab | The only path that executes anything. Re-runs `policy.authorize` at execution time with real `budget_state` and `resources`. Idempotent per approval id. Used by the ticker, by approved items and by the page's manual buttons. |
| 8 | `run_inc_plan.sh` + `inc_autopilot/brain_plan.py` | cluster sbatch, GPU (cloned from `run_llm_review.sh`: ollama on a per-job port, warm-up, model from `model_router.resolve("planner")`) | The research brain (§c). It is advisory and never on the critical path. |
| 9 | `inc_autopilot/validate.py` | lab | Checks brain output: menu membership, bounds (via `policy._check_params`), citations resolved against evidence and the corpus, no test data. |
| 10 | `inc_autopilot/outcome.py` | lab | Turns a finished child experiment into `experiments.record_result` and `verdict`. The noise floor comes from the dev spread of the b0_v1 seeds. Updates the lever and brain track record. |
| 11 | `docs/literature/*.md` + `literature/index.json` | repo (read on both hosts) | A line-addressed corpus of verbatim passages. **Not found today**; it must be built. |
| 12 | `/inc` page and `/api/inc/*` routes; `/api/health/inc` | lab dashboard | See §e. |

**`remote.py` verbs** (component 4):
- `snapshot --exp X`: small files, sha256 of each, and per-source aggregates of `select_clusters.csv` and `admit_summary.json`. The CSV itself is never shipped.
- `advance --exp X`: imports the pinned `inc.driver.advance()`, which returns `{locked, passes, submitted, job_ids, done, lines}` (driver.py:1248-1273), and prints it as `INCADV {...}`.
- `report --exp X`
- `submit <builder> <bounded args>`: sbatch of `run_inc_build.sh`, `run_inc_relevance.sh` or `run_inc_audit.sh`.

**State and provenance files:**
- The campaign ledger is `results/framework/_brain/weed/inc_campaign.jsonl` on the lab. It is append-only and every entry carries `decided_by`, `trigger` (diagnosis ids with cites), `parent_exp`, `child_exp`, `approval_id` and `job_ids`.
- Snapshot history goes to `_brain/weed/inc/<exp>/snapshots/*.json`. `latest_bundle.json` is overwritten on every write (RS:296-302), so it cannot serve as history.
- Provenance on /ocean goes to `INC_DIR/_campaign/provenance/<exp>.json`, **not** inside the experiment directory. `pilot.py:265-269` refuses to build when `exp.json`, `manifests/`, `labels/` or `build_summary` already exist.

**Ticker phases** (one item in flight per campaign):

```
RUN ──(snapshot shows state.done)──> REPORT (remote report) ──> DIAGNOSE (lab, deterministic)
  ▲                                                                     │
  │                         no lever fired and goal met ──> COMPLETE <──┤
  │                                                                     ▼
EXECUTE <── GATE (authorize / approvals.propose / wait) <── VALIDATE <── PROPOSE
                                                                         (deterministic now;
                                                                          brain job async, ≤2 h,
                                                                          merged if back in time)
```

- During RUN, every tick also runs the **health diagnoses** (D5–D8, D10, D14) on the fresh snapshot. In-job advance (`INC_JOB_ADVANCE`, `run_inc_job.sh:95-110`) stays on.
- The driver's flock plus O_EXCL lease makes a double advance safe. Even so, the human's `driver watch` for that experiment must be stopped at cutover so that there is a single owner.

## (b) Deterministic diagnoses and the lever catalogue

**Rules for every diagnosis:**
- It reads only fields that already exist, reuses gate constants where possible (`GateConfig.p_recipe_flag = 0.25`, `p_accept = 0.75`; gate.py:75-77), and cites what it read.
- A diagnosis that fires with no cites is downgraded to `unknown`, the same rule as `signals.py:2076-2085`.

### Diagnoses

**Experiment-level diagnoses (the four human interventions, generalised)**

**D1 `recipe_forgets`**
- Generalises intervention 1: "the cheap recipe itself degrades the incumbent, so the gate cannot see any data effect."
- **Fires** when all three hold:
  - among gate entries, the share with `decision.p_recipe ≤ 0.25` is ≥ 0.5;
  - the share with `null_mean < inc` is ≥ 0.5;
  - at least one chain step on a clean step (`exp.steps[k].clean`) where `truth/<k>.detail.verdict == helps` is not ACCEPT.
- **pilot_v1 evidence:**
  - 20 of 21 gate entries have `p_recipe = 0.0`;
  - 21 of 21 have `null_mean < inc`;
  - the truth arm says `helps` for I2, I3 and I5, and all 9 of those chain steps are REJECT.
- **Levers:**
  - L1, if `replay_mode` is `sample` or absent;
  - if it is already `full`: X1 (off-menu), and D4 does not fire while D1 blocks it (refinement below).
- **Refinement: D1 blocks D4 only through D4's own choice (R4 review, 2026-09-27, made after pilot_v3's result; post hoc).**
  - **Why.** D1's purpose is "the recipe degrades the incumbent, so the gate cannot see the data effect". That matters for D4 only on the chain D4 would select. The pooled conditions count every chain, so a forgetting chain D4 does not pick could withhold the real loop from a chain that tracked the truth arm.
  - **Rule.** D1 still fires, with all its cites and levers, whenever its pooled conditions hold. The forgetting is real, and X1 stays a research card. D1 **blocks D4** only when either of these holds:
    - the chain D4 would select (`diagnose.rank_recipes`: argmax agreement with D4's tie-breaks) did not ACCEPT at least one clean truth-helps step. The misses are read from D1's rows (the ledger's, when the ledger is in the evidence) and from `report.json`, which D4 ranks from, so a miss only the report shows still blocks. A truth-helps step that is not clean is never one of D1's misses and never blocks, although D4's own tie-break (fewer non-ACCEPT on truth-helps steps) still counts it;
    - D4's own threshold (5/7) is not met on the pilot.
  - **D1's fire decision never depends on D4's inputs.** When D4's view cannot be read (for example a malformed `report.json` of any pilot in the evidence), D1 fires or stays silent exactly as before, keeps its levers, and blocks D4 with the reason.
  - **What D4 does.** When D1 does not block, D4 applies its pre-registered rule unchanged: ready at ≥ 5/7, recipes = the selected chain, and the replay mode and gate flips mode of the evaluated pilot. It records D1 under `detail.d1`. When D1 blocks, D4 behaves as before: silent in full replay, and `blocked_by: D1` with D1's levers in sample replay.
  - **What D1 records** in its detail:
    - `chains`: each chain's clean truth-helps steps and the ones it missed;
    - `d4_selection`: the selected chain, its rate, the threshold, whether it is ready, the decisive criterion, its misses (`misses`: D1's rows; `report_misses`: the report's clean truth-helps steps it did not ACCEPT) and D4's own tie-break count (`report_helps_not_accepted`);
    - `blocks_d4` (`true`, `false`, or `null` on an experiment D4 does not evaluate, such as a real loop or a running pilot) and `blocks_d4_why`, which names each missed step (a step only the report shows is marked `(report.json)`).
  - **Unchanged by it:**
    - D15's precedence: when D15 blocks D1, D4 still carries D15's block.
    - In sample replay, D1's own L1 still ranks ahead of D4's L2 in `levers.propose`, so the pilot is rebuilt with full replay first.
  - **pilot_v3 evidence** (full replay, gate v2 `net`, `v2_check` supported: v2 11/21, v1 on the same runs 9/21, nothing non-clean accepted):
    - D1 fires: 18/21 `P_recipe ≤ 0.25`; 19/21 `null_mean < inc`; 3/9 truth-helps chain steps not ACCEPT (freeze I3 and lora I2 on the species guard; freeze I5 on the species and flips guards).
    - Agreement with the truth arm: full 5/7, freeze 3/7, lora 3/7. D4 selects full by rate, and full ACCEPTed every truth-helps step (I2, I3, I5). Full also ACCEPTed I1 (truth hurts) and I4 (neutral), and REJECTed Bswap and Breal.
    - Final dev twelve-class mAP50-95: full 0.8006 vs T_final (the cold union of all clean data) 0.8084 ± 0.0043. For people only (no rule reads them): test 0.8475 vs 0.8472 ± 0.0014, ood22 0.7630 vs 0.7561.
    - So D1 does not block, and D4 is ready: `L2 --replay-mode full --recipes full --gate-flips-mode net`.
    - Under the rule as first written, D1 blocked D4 on pilot_v3, and the campaign's first prospective record, `prospective_d4_pilot_v3.json` (unversioned), says `ready: false`. It stays as written (below, "Prospective records are versioned").
  - **What this does and does not show.** The refinement was chosen with pilot_v3's result in view, so replay case R6 only reproduces it. Its prospective test is the real loop's truth arm: the loop's chain must keep agreeing with the truth arm (L2's falsifier: below the 5/7 the pilot showed).
  - **R1 and R5 do not change under it.** On pilot_v1 every chain missed every truth-helps step, and D4's threshold is not met (3/7). On pilot_v2, the chain D4 selects, full, missed I2 and I3, and its best rate is 4/7.

**D2 `unmeasured_source_property`**
- Generalises intervention 2: "a filter certifies property A (label species), a consumer assumes property B (in-domain), and B was never measured."
- **Fires** when any of these holds:
  - increment_pool sources where at least 95% of images are `no_evidence`, `below_gate` or `no_feature`, with 0 species crops in `retrieval.source_evidence` and about 0 `verified` boxes in `admit_summary.per_slug`, **and** no `relevance.json` matching this select build;
  - any builder refusal whose message names a missing prerequisite. For example, `realloop.py:384-403` refuses in production without relevance;
  - a `relevance.json` in the `calibration_failed` state (below), whatever the per-source evidence says.
- **Evidence:** cluster only (not found locally). The protocol names `fvossel__csgo_player_detection` (4,012 images), `kg_farukalam__tomato-leaf…` (434) and `rf_bishwarup-halder__crop-health-advisor` (15,116).
- **Levers:** L3, then L2 rebuilt with `--relevance`.
- **The states of `step1/relevance.json`** (`levers.relevance_status`, which gives the state and why):
  - `missing`: no file. D2 proposes L3.
  - `stale`: made for another select build (the tables' manifest sha256 differ from `select_summary.json`), whatever its check says.
  - `refused` (malformed): any file `relevance.load` refuses before it reaches the calibration verdict. The checks run in `relevance.load`'s order:
    - the format is not `inc.relevance/1`;
    - the file was made under another rule: its `params` block (`min_crops`, `calibration_percentile`, `tau_min`) is not `relevance.MIN_CROPS` = 20, `CAL_PERCENTILE` = 5 and `TAU_MIN` = 0.5 (mirrored in levers.json `protocol.relevance_min_crops`, `relevance_calibration_percentile` and `relevance_tau_min`, and checked against the module);
    - there is no calibrated tau in [0, 1], or no check block;
    - the check does not follow from its tau under the rule. That is, `ok` is not a boolean, the check's `tau_min` is not 0.5, or `ok` is not (tau ≥ 0.5). Both directions count: `ok: true` below 0.5 and `ok: false` at or above it.
  - `calibration_failed`: made for this select build under the protocol's rule, with a consistent check that failed (`ok: false`, tau < tau_min = 0.5). `relevance.load` refuses it for its calibration, so every `--relevance` build does too.
  - `matching`: made for this select build under the protocol's rule, with a consistent check that passed (`ok: true`, tau ≥ 0.5).
- **A stale or malformed file:** D2 escalates to a person (`OP_ESCALATE`, crit) instead of proposing L3. `relevance build` refuses to overwrite another build's file without `--force`, and `--force` is deliberately not a lever param.
- **A calibration-failed file (2026-09-27, after the cluster's relevance build failed its check: tau 0.0557 < 0.5).** The zero-shot criterion cannot build a production loop, and the protocol adopted a second, verifier-grounded criterion for it: source-level species evidence (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3; `realloop build --increment-sources evidence`).
  - **D2 always fires on such a file**, whatever the per-source domain evidence says. That covers the case where no source is flagged, and the case where `select_clusters_by_source.json` or `admit_summary.json` is not in the evidence. The file decides which criterion a real loop can use at all. D2 escalates only when the evidence criterion cannot be read or cannot hold the build (below).
  - **Capacity first.** `levers.evidence_capacity` computes the evidenced pool from Step 1's summaries the way `select.load_evidence`, `select.apply_evidence` and `realloop.evidence_capacity` do. A source is evidenced when `admit_summary.json` `per_slug[s].boxes.verified` ≥ `min_evidence` (1, levers.json `protocol.min_evidence_default`). Its increment-pool images come from `select_summary.json` `sources.increment_pool`. The pool fits when it holds (N + 1) × M images, with N = 6 and M = 10% of base B, rounded (the builders' defaults). When it does not, the loop is sized by rule (below, "Sizing").
    - The same refusals as the builder defer it: an admit summary without per_slug counts, one naming another `verified.jsonl` or `crops.csv` than the select build read, verified boxes above `retrieval.source_evidence` species crops, a counted source without a per_slug entry, or an increment-pool source without one.
    - The OtherPlant-heavy part of realloop's check (M of those images in OtherPlant-heavy near-dup groups) needs `select_clusters.csv`, which is never shipped. It is recorded as not checked here, and `realloop build` checks it before anything is written. Its refusal maps to a person (levers.json `refusals`).
  - **Sizing (R4 decision, strong-brain review, 2026-09-27).** An evidence-sourced real loop whose default size does not fit is sized by rule instead of escalated (`levers.evidence_sizing`; levers.json `protocol.min_decided_increments` and `protocol.min_increment_frac`, each with its why):
    - **N** = the smallest number of verified increments with which the loop decides at least `min_decided_increments` (6, the Steps 2-3 acceptance) increments together with UNVERIFIED and OTHER_HEAVY. realloop's sequence has N + 2 steps, so N = 4 (`levers.sized_n_verified`, checked against `realloop.sequence`).
    - **M** = min(round(0.10 × |B|), floor(evidenced pool images / (N + 1))): the protocol's size when it fits at the smaller N, else the largest size the image count holds.
    - **Floor.** If that M is below `min_increment_frac` (0.05) × |B|, the review judged the increments too small for the gate to detect an effect, and the loop is not sized: D2 escalates with the numbers. D2 also escalates when the capacity cannot be computed.
    - **Why a rule.** The acceptance needs at least 6 decided increments, and the evidence criterion's pool is what it is. The rule keeps the acceptance and gives up increment size, down to half the protocol's 10%; below that floor a person decides. Whether the gate resolves effects at the sized M is not measured here: the sized loop's own gate entries show it (D8 `gate_underpowered`).
    - **What stays realloop's.** The rule sizes by image count only. The OtherPlant-heavy part of realloop's check (M images in OtherPlant-heavy near-dup groups) needs `select_clusters.csv`, so `realloop build` checks it before anything is written. If it refuses, the build fails on the cluster and the refusal is handled as described under "Builder refusal → prerequisite" (below): DREF escalates to a person, and the campaign does not submit the identical build again.
    - **Only D4's L2 and L6 are sized.** A build that gives its own `--size` or `--n-verified` (a brain plan's, or an L5 or D1 rebuild's parent values) is never sized, but it is held to the same floor (`levers.r4_floor`): M at least `min_increment_frac` × |B| and at least `min_decided_increments` decided increments. Below the floor it is refused with the numbers (`LeverError`: a brain item is dropped as a render refusal, never filed; an L5 or D1 rebuild is listed under `refused`). At or above it, the capacity is checked at those values (`levers.evidence_capacity`, `levers.check_criterion`). A brain plan that names neither gets the sized values, the same request as the deterministic L2 (`validate.materialise` through `levers._l2_params`).
  - **When the evidenced pool fits** (at the default N and M, or sized): D2 fires `warn` with card **X9** (below) and `then: ["L2 with --increment-sources evidence"]`, or, when sized, `then: ["L2 with --increment-sources evidence --size <M> --n-verified <N>"]` (only the flags that differ from the builders' defaults). That holds even when the per-source aggregate is missing, since the capacity reads only the select and admit summaries. It never proposes L3 over the file: `levers._l3_ok` defers L3 for any proposer, deterministic or brain, because a rebuild on the same select build gives the same verdict.
    - **Deferral.** `levers.propose` reports the `then` under `deferred` only while no L2 carrying those flags side by side is proposed. When D4 is ready in the same tick, the L2 is D4's proposal, and it is not also listed as deferred (`levers._then_proposed`).
  - **The L2 itself comes from D4's decision, as always** (`levers._l2_params` through `levers.increment_criterion`): `python -m weed_optimizer_framework.tools.inc.realloop build --exp <child> --base <INC_DIR>/step1/base_B.jsonl --replay-mode <D4> --recipes <D4> --increment-sources evidence [--size <M>] [--n-verified <N>] --gate-flips-mode <pilot gate>`.
    - Flags are rendered by the L2 template, in its order, the template every realloop lever shares. `--increment-sources` takes the slot `--relevance` takes, since the two exclude each other. `--size` and `--n-verified` follow it, in that order. D4's L2 and L6 render each only when the sized value differs from the builders' default (10% of B, 6). A rebuild of an existing loop (`levers.loop_params`, for L5 and D1's L2) renders the parent's recorded `--size` and `--n-verified` whatever they are, defaults included. realloop's argparse reads the two forms the same way. `--min-evidence` is never rendered (the protocol's default), and `--gate-flips-mode` stays last. `executor.ARGV_FORMS` renders the same order.
    - **The gates accept them within bounds:** L2's own `param_bounds` (`size` 1–20,000, `n_verified` 1–12), the `inc_build_realloop` policy row (1–50,000, 1–20), `remote.py submit` (L2, L5 and L6 admit them; an `n_verified` of 13 or a `size` of 20,001 is refused) and `validate.materialise`.
    - The proposal carries D4's trigger and cites, plus the criterion's cites: the failed check and the verified boxes and pool images it rests on. It is priced at the N and M it builds (`levers.estimate_realloop`).
  - **What D2's detail records:** `detail.increment_criterion` holds which criterion the build uses and why:
    - `criterion`: `relevance`, `evidence` or null;
    - `flag` (the criterion's flag) and, on `evidence`, `flags` (the criterion's flag plus the sized `--size` and `--n-verified`);
    - `why`;
    - `relevance`: the state, why, tau and tau_min;
    - `capacity`: at the N and M the build uses (the default ones when the rule did not size): the evidenced sources with their verified boxes and pool images, `evidenced_pool_images`, `needed_images`, `n_verified`, `increment_images`, `fits`, `max_n_verified`, `base_images`, the excluded sources and images, and the cross-check;
    - `sizing`: the rule's text; `min_decided_increments`, `min_increment_frac` and `min_increment_images` (the floor in images); `default` {N, M, needed images, fits, max N, decided increments}; `sized` {N, M, needed images, decided increments, M as a share of B, fits} (null when the default fits); `applied`; `params` and `flags` (the non-default `--size` and `--n-verified`); `why`;
    - `flagged_excluded`: the sources D2 flagged that the criterion excludes. On a pool like the real one these are the non-plant sources, which hold 0 verified boxes.
    - It also cites `relevance.json` `/format`, `/calibration/tau`, `/calibration/check/ok` and `/calibration/check/tau_min`, and the evidence.
  - **When it does not fit even sized, or cannot be read:** D2 escalates (`OP_ESCALATE` and X9, crit) with the reason in `detail.needs`. "Cannot be read" means `admit_summary.json` is not in the evidence, or the summaries disagree in a way the builder refuses. "Does not fit" gives the numbers: the default N and M, and the sized M against the floor. Choosing N and M is then a review decision (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3). The L2 is deferred with the same numbers. A person may build a smaller loop by hand. A brain plan may propose its own `--size` and `--n-verified`, but `validate.materialise` holds them to the R4 floor first (`levers.r4_floor`) and then checks the capacity at the plan's N and M. A plan below the floor is dropped with the numbers. One at or above it that fits is filed as R3 for approval.
  - **Card X9, a person's to decide.** "Revisit the zero-shot relevance criterion": a new relevance protocol version must be pre-registered before it is run on this pool. It stays up next to the evidence build, because the evidence criterion excludes genuine weed datasets whose species are not cwd12 species.
  - **Not affected:** D2b does not read the statuses of a refused or calibration-failed file.
- **The live campaign (Step 1 of 2026-09-27).** The five evidenced sources hold 1,439 of 96,088 increment-pool images. The default build needs 2,751 (7 × 393); at M = 393 the image count allows at most `--n-verified 2`, four decided increments.
  - **Before the sizing rule** D2 escalated with those numbers, and the real loop waited for a person to choose N and M. The OP_ESCALATE card on the page names the escalating diagnoses and their summaries (`campaign._diagnose`), not only D13's.
  - **Under the sizing rule (R4 decision, 2026-09-27)** D2 does not escalate: N = 4 (six decided increments), M = min(393, floor(1,439 / 5)) = 287, 7.3% of B and above the floor of 196.35 images (5% of 3,927). It fires `warn` with X9, and D4's L2 on pilot_v3 is `python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl --replay-mode full --recipes full --increment-sources evidence --size 287 --n-verified 4 --gate-flips-mode net` (replay case R8).
  - The OtherPlant-heavy share of that loop (287 images in OtherPlant-heavy near-dup groups) is realloop's to check on the cluster.
  - The sized loop needs 1,435 of the 1,439 evidenced images. The capacity is necessary, not sufficient (`select.pool_capacity`: the draw keeps near-dup groups whole; across the whole Step 1 pool `select_summary.json` `near_dup` counts 23,636 images in multi-image groups, the largest of 1,840), so the draw can still fall short at build time (`select`'s "N increments of M images need …, the draw found …").
  - **Either refusal:** both are in levers.json `refusals` with no menu prerequisite ("a person") and `retry: false`, because each reads only Step 1's files and the build's flags, so the identical build refuses again. The build fails on the cluster, and `campaign._build_failed` declines that exact request (its key in the state's `declined`, ledger `not_taken`). The next DIAGNOSE puts the refusal in the context. DREF fires crit with `OP_ESCALATE`, and the ticker puts the escalation card up. D4 proposes the same L2 again, and `campaign._filter` does not take it ("declined or refused earlier in this campaign"). The campaign goes on with any other lever. With none left, it completes with a residual card that names the declined L2. One build job is spent, not two, and the stop-loss is not what stops it. A person then chooses N and M: by hand, or by approving a brain plan with its own N and M at or above the floor (a different request, so not declined).
  - Each of these changes edits `diagnose.py`, `levers.json` or `levers.py`, and all three are in the rules version. So each change is a new rules version: the ticker diagnoses pilot_v3 again and writes that version's prospective record before any L2 (D4, "Prospective records are versioned").

**D2b `relevance_blind_spot`**
- **Fires** when a source has relevance `status == pass` and `top_set_share.leaf_disease ≥ θ`.
- **Silent** on a refused (malformed) or calibration-failed file: its statuses are not trusted.
- **Evidence:** `relevance.json` (cluster).
- **Lever:** X2 (off-menu, human).

**D3 `attribution_control_failed`**
- Generalises intervention 3: "a planted positive control is misclassified by a diagnostic, so that diagnostic is untrusted; run the independent measurement the ledger already lists as not run."
- **Fires** when either holds:
  - a step with `exp.steps[k].clean == false` whose `planted` text names relabelling, but `report.chains[r].bswap.attributed_to_labels == false` (more generally, the step's `class_vs_loc` is not `labels`);
  - gate entries carry `attribution_not_run ∋ label_audit`.
- **pilot_v1 evidence:**
  - Bswap has `planted = "40% of boxes relabelled to another species"`;
  - all three chains have `attributed_to_labels = false` and `class_vs_loc = domain/localisation`;
  - `attribution_scope.not_run["4"]` is set.
- **Levers:** L4, then X3 (off-menu) if the audit separates what `class_vs_loc` did not.

**D3b `source_attribution_missing`**
- **Fires** when `chains[r].rejects_without_source_attribution` is non-empty.
- **pilot_v1 evidence:** `['s05_Breal']` in all three chains; `attribution_scope.not_run["5"]`.
- **Lever:** X4 (off-menu; leave-one-source-out has never been implemented).

**D4 `decision_slot_ready` / `no_recipe_tracks_truth`**
- Generalises intervention 4: "a downstream build has required arguments that an upstream experiment was designed to decide." The required arguments are `realloop --replay-mode --recipes` (realloop.py:572-575).
- **Rule:**
  - take the latest pilot whose steps all compared (`agreement[r].compared == len(steps)`);
  - best = argmax `agreement[r].rate`;
  - ties are broken by, in order:
    1. fewer non-ACCEPT verdicts on truth-`helps` steps;
    2. smaller `|final dev twelve(chain) − dev twelve(T_final)|`;
    3. lower `gpu_hours["chain:r"]`.
  - It fires `ready` only if best rate ≥ 5/7. Otherwise it fires `no_recipe_tracks_truth`.
- **pilot_v1 evidence:**
  - `freeze`, `full` and `lora` are all 3/7 = 0.4286;
  - no chain accepted anything, so every final is 0.7282;
  - result: `no_recipe_tracks_truth`. This matches the manual decision to build pilot_v2 rather than the real loop.
- **Levers:** L2 if ready; otherwise feed back into D1 levers.
- **Precedence of D1 (implementation, 2026-09-27 review; narrowed by the R4 review under D1).** When D1 fires on the pilot D4 picked **and blocks D4** (`blocks_d4`: D4's threshold is not met, or the chain D4 selects did not ACCEPT a clean truth-helps step), D1 wins in either replay mode:
  - in full mode D4 does not fire (the rule above);
  - in sample mode D4 still reports `decision_slot_ready` with its ranking, but with `blocked_by: D1` and D1's levers (L1 + X1). L2 is withheld, and `levers._l2_params` refuses it for any proposer. The prospective record then says `ready: false`.
  - Agreement ≥ 5/7 can coexist with a REJECTed truth-helps step on the selected chain, so without this rule one pilot would propose both "rebuild the pilot" and "build the real loop" from a chain that forgets.
  - When D1 fires but does not block (pilot_v3), D4 is ready by its own rule and records D1 under `detail.d1`.
  - D15 takes precedence over D1 in turn, but only when the flips guard explains every miss (below). When D1 is blocked by D15, D4 carries D15's block instead of D1's levers: in full mode it stays silent, in sample mode it reports `blocked_by: D15` and proposes nothing.
  - A ready pilot decided on protocol v2 whose `v2_check` is not final (provisional, or missing from its report) is not ready either: D4 reports `blocked_by: D16` with `pending` and `v2_pending`, and proposes no L2 (D16, below).
- **Prospective records are versioned (2026-09-27).** D4's decision on a pilot is frozen once per rules version, not once for all time, so a rule change after a record never rewrites it and never lets an old decision stand for new rules.
  - **Rules version:** the first 12 hex of the sha256 of `diagnose.py` + `thresholds.json` + `levers.json` + `levers.py`, concatenated in that order (`diagnose.rules_version()`). To check it by hand: `cat diagnose.py thresholds.json levers.json levers.py | shasum -a 256 | cut -c1-12`. `levers.py` is included because D2's decision and the L2 it names rest on its evidence capacity and the R4 sizing rule (`levers.evidence_sizing`, `levers.r4_floor`), not only on levers.json's constants. `diagnose.py` and `levers.py` are hashed as imported, so a deploy without a restart keeps the old version until the code that decides actually changes.
  - **One record per (pilot, rules version):** `replay/prospective_d4_<exp>__<rulesver>.json` (`diagnose.prospective_name`). Each record carries `rules_version` and each rules file's sha256 (`rules_files`).
    - An earlier record is never modified or deleted. That covers another rules version and the unversioned `prospective_d4_<exp>.json` written before versions existed. The unversioned file stays readable for history, and the page lists it with `current_rules: false` and `superseded: true`. Only the record of the current rules is compared with the first real loop built after it; a superseded row reads "superseded", not match or differs.
    - The ticker's `prospective_d4` ledger entry names every record it supersedes, with its path, sha256 and rules version. The state keeps them under `prospective.<pilot>.history`.
  - **Every L2/L6 path applies one check, `diagnose.prospective_guard`,** to a real loop built from a pilot's D4 decision (a rebuild of a real loop follows no pilot decision and is not held to it):
    - a READY record under the current rules version for the evaluated pilot (`diagnose.current_prospective`). A record of other rules never counts, whatever it says, and neither does a record that is not ready;
    - the file's sha256 is the one the ticker recorded in its state, whenever that state is at hand;
    - the build follows the record: its replay mode, recipes and gate flips mode (the build's `gate_flips_mode`, else the builders' default `negative`) are the record's (`diagnose.compare_prospective`).
  - **Where it runs:**
    - The ticker (`campaign._d4_guard`, through `_d4_unfrozen`): before it proposes an L2/L6, before it files a brain's, before it adopts an approved one, and again on its own L2/L6 right before the executor runs it. An approved item that does not follow a READY record of the current rules is refused for good, with a card. One whose decision is not frozen yet waits.
    - A rule change while the campaign's own L2/L6 waits at GATE (for a person, the envelope or the budget) drops it at once (`not_taken`, naming both rules versions). DIAGNOSE then writes the new version's record first, and the L2 is proposed again only if D4 is still ready. Unchanged, it is the same proposal with the same approval. The dropped item's approval is never executed through the ticker without these checks.
    - The INC page: the build-realloop route, and execute-approved of an `inc_build_realloop` approval. The pilot is the build's parent, else the latest finished pilot. A person may still go past the check with an acknowledgement. For a build, the acknowledgement is written into the request's reason. For execute-approved, it is written into the execution record's `note`. On a refusal the page asks for that reason.
  - **Re-diagnosis:**
    - The ticker diagnoses a finished pilot again when the rules version differs from the one its last DIAGNOSE ran under. A campaign in COMPLETE or BRAIN_WAIT with nothing in flight moves to DIAGNOSE, and the ledger gets a `rules_changed` entry.
    - A campaign with an item, a build or a job in flight reaches DIAGNOSE on its own. For example, an R2 item such as L3 or L4 ends in WAIT_JOB → DIAGNOSE. An L2/L6 waiting at GATE is dropped as soon as the rules change (above).
    - That DIAGNOSE writes the new version's record before any proposal.
    - Nothing already executed runs again: every lever passes `levers.applied` (the execution-log lineage) and the declined keys, as on any DIAGNOSE.
  - **The live campaign.** `prospective_d4_pilot_v3.json` (ready false, rules before the D1 refinement) stays as written. After the deploy, the ticker's next DIAGNOSE of pilot_v3 writes `prospective_d4_pilot_v3__<rulesver>.json`, which is ready: full replay, recipe full, gate `net`. Only then may L2 be proposed. That DIAGNOSE comes on the next tick when the campaign is COMPLETE, and after the job ends when an L3 or L4 is in flight.

**Experiment-level diagnoses of the gate protocol (docs/INCREMENTAL_PROTOCOL.md, Protocol v2)**

**D15 `guard_blocks_truth_helps`**
- Generalises the pilot_v2 finding (docs/INCREMENTAL_PROTOCOL.md, "Protocol v2 (net flips)"): "the gate saw the data effect, and the flips guard alone stopped it."
- **Reads** a finished chain experiment (`state.json` or `report.json` `done`), over every (chain, step) pair: the clean steps the truth arm says help, and among their pairs those REJECTed with `P_data ≥ p_accept` whose only failed guard is the flips guard (every other guard the decision records passed). The guard dicts come from the ledger's gate entries (`decision.guards.<g>.passed`), else from `report.json` `steps[].chains[].guards`. `p_accept` is the decision's own recorded `config.p_accept` (a gate block may change it), else thresholds.json `D15.p_accept`.
- **Fires** when at least `D15.min_pairs` (2) such pairs exist, pooled over all chains. Each pair is its own gate decision (its own incumbent, cands, nulls and flip counts), and the v2 check itself pools (chain, step) pairs the same way. One pair can be a single noisy decision; two is the smallest repetition. A step not marked clean never counts, since v2 accepting one refutes v2.
- **Cites** every counted pair: its ledger line's verdict, `P_data`, every guard's `passed`, its `config.p_accept`, and the truth verdict and clean flag of its step. It also cites the pinned flips mode and `done`. The truth-helps misses it does not count are listed in `detail.unexplained` (chain, step, line, failed guards, why).
- **Decision: pooled pairs, not a per-chain count (2026-09-27 review).** The first implementation counted pairs per chain and blocked D1 when any one firing chain was fully explained. On pilot_v2 that cited four pairs, left out lora I3 (line 23), which qualifies, and silenced D1 although freeze I5 had failed the regression guard, which is D1's own forgetting signal. Its `min_pairs` rationale also quoted pilot_v2's per-chain counts, so the design had been chosen with R5's evidence in view. The rule now follows the contract as written: pairs are pooled, every qualifying line is cited, and the precedence over D1 needs every truth-helps miss in every chain to be flips-only. R5 changed with it (below): five pairs, and D1 is not blocked. The contract's shorthand "the four flips-only rejections" was chain full's I1–I4 from the protocol's evidence table; D15 counts only truth-helps steps, and I1 (hurts) and I4 (neutral) are not among them.
- **The pinned flips mode** (`diagnose.pinned_gate`), in this order: `state.json` `gate_pin.config.flips_mode`; the ledger's `gate_pin` entry; `report.json` `gate.flips_mode`; exp.json's gate block; the config the gate decisions recorded. A v1 decision omits `flips_mode`, so pilot_v1 and pilot_v2, built before gate blocks existed, resolve to `negative`.
- **Levers:**
  - pinned to `negative` (v1), on a pilot: **L9**. The detail's `then` says L2 waits for the v2 pilot's report.
  - pinned to `negative`, on a real loop: card **X8** (R4). A person decides, because L9 rebuilds pilots and a real loop is built on v2 only after a v2 pilot's check; the card names the cheapest test (L9 on the loop's source pilot).
  - pinned to `net` (v2): no rebuild. The pre-registered alternative is what blocked the steps, so card **X6** (rethink the guard family, R4) goes up.
- **Precedence over D1.** When D15 fires, carries a lever or card (L9, X8 or X6), and every clean truth-helps pair that is not ACCEPTed, in every chain, is one of its pairs, D1 still fires with all its cites but reports `blocked_by: D15` (the pairs and D15's levers) and proposes nothing (`detail.withheld` keeps the levers it would have proposed). Then the gate saw the data effect every time and only the flips guard stopped it, so the gate protocol is tested before the recipe (L1, X1). When any miss failed another guard too, or was not decided at `P_data ≥ p_accept`, or D15 has nothing to propose (a mode that is neither v1 nor v2), D1 is not blocked: it fires with its own levers, `detail.precedence` names the misses D15 leaves, and D1 cites their guards. Both proposals then stand.
- **pilot_v2 evidence** (ledger lines):
  - five pairs: freeze I2 (15) and I3 (22), full I2 (13) and I3 (19), lora I3 (23), each REJECT at `P_data` 1.00, regression and species passed, flips failed, so D15 fires with L9;
  - three truth-helps misses failed another guard too: freeze I5 (31) on regression and species, lora I2 (18) and I5 (32) on species;
  - so D15 does not explain every miss, and D1 (20/21 P_recipe, 20/21 null < inc, 8/9 truth-helps not ACCEPT) is not blocked. In full replay its lever is card X1, and D4 stays silent (the recipe forgets). The one build proposed is L9.
- **pilot_v1:** silent. Every truth-helps REJECT there failed another guard too.
- **Ranking.** `diagnose.RULES` puts D15 right after D1, and D16 right after D4. That is the order `levers.propose` ranks proposals in, so on pilot_v2 L9 is the campaign's one item in flight, ahead of D2's L3 and D3's L4; those two come back on a later DIAGNOSE. D1's X1 there is a card, never queued.

**D16 `gate_v2_check`**
- **Reads** `report.json` `v2_check` (docs/INCREMENTAL_PROTOCOL.md, "The criterion"), on a finished experiment pinned to `net` whose check is `final`. A provisional check, a missing one (a report older than the check), an unfinished experiment and a v1 experiment are silent.
- **Outcomes** (thresholds.json `D16.severity`):
  - `refuted` → **crit**, card **X7** (revert to v1 or rethink the guard). D4 on that pilot reports `blocked_by: D16` and proposes nothing, so no L2 is built on the refuted gate; the prospective record says `ready: false`.
  - `supported` → **info**. It is recorded in D4's detail (`gate.v2_check`), and D4 may proceed.
  - `inconclusive` → **warn**. D4 proceeds only on its own ready threshold (v2 never lifts it) and records `v2_unproven`.
- **Cites** the outcome, `final`, `agree_v2`, `agree_v1_counterfactual`, `compared`, `non_clean_accepted` when set, the pinned mode and `done`.
- D4's detail always carries `gate: {flips_mode, v2_check}` of the pilot it evaluated, and `prospective_d4` records `gate_flips_mode` (when ready) and `v2_check`.
- **A check that is not final yet.** D16 is silent on a provisional or missing `v2_check`, but D4 does not treat that pilot as ready: when it would be ready, it reports `blocked_by: {id: D16, pending: "not final" | "missing"}` with `detail.v2_pending`, proposes no L2, and its `then` says L2 waits for the final check (for a missing one: rerun the report with the current `inc.report`). L9's success criterion needs `supported`, so a real loop is never built on a v2 gate before the check says so.
- `prospective_d4` writes nothing for a v2 pilot whose check is not final: it returns the record with `pending` and `written: false`, and the campaign ledger notes `prospective_pending` once. The record is written on a later DIAGNOSE, once the check is final, so a provisional report never freezes a decision.
- After a final `refuted`, no lever rebuilds on that gate: `levers.gate_params` defers for any experiment whose own final `v2_check` is refuted, so L1 on such a pilot, and L2 or L5 on such a loop, wait for card X7's decision instead of carrying the retired mode.

**Operational and health diagnoses (generalised beyond the four)**

| ID | Fires when | Evidence | Lever |
|---|---|---|---|
| D5 `chain_blocked` | `state.blocked` is non-empty | `blocked{unit:{error,cause}}` | L7 if the cause is on the transient allow-list and the unit was never auto-unblocked; otherwise pause and a human card |
| D6 `stale_advance` | `generation` unchanged for more than 2 h, with no `inc_<exp>_*` in squeue and not done | `state.json`, squeue | run `advance`; if still stale, pause |
| D7 `code_drift` | `advance` raises the drift or pin `DriverError` (driver.py:1303-1310, 339) | the `INCADV` error | pause; X5 (human) |
| D8 `gate_underpowered` | at least 50% of gate entries have P_data in (0.25, 0.75) **and** \|cand−null\| < 2·max(sd) | gate decisions | L5 (larger `--size`) |
| D9 `warmup_dominates` | `effective_warmup.incremental_effective_epochs.max / epochs > 0.15` | pilot_v1: 6.67/30 = 0.22 | informational, and supports L1 (the doc notes full replay cuts it to about 1–2 epochs) |
| D10 `budget_projection` | SU spent plus the projected cost of remaining runs exceeds the envelope | `gpu_hours`, su_ledger | pause |
| D11 `truth_cost_share` | the truth arm is more than 40% of GPU-h, and the best agreement is ≥ 6/7 over ≥ 2 pilots | pilot_v1: 8.92/17.28 = 52% | L6 (`--no-truth` in realloop), but only after agreement is established |
| D12 `chains_indistinguishable` | all chains give identical verdicts on every step | `steps[].chains` | only the cheapest recipe goes forward in L2 |
| D13 `prediction_contradicted` | a child experiment's outcome contradicts the predicted direction of the lever that launched it | `outcome.py` verdict | escalate to brain and human; the lever's track record goes down |
| D14 `test_touch` | any decision path touched test | loader audit | hard stop |

**Two generic mechanisms that cover unanticipated cases:**
- **Builder refusal → prerequisite.** Every builder already refuses bad inputs (LOCK, `select_provenance`, a missing relevance file, a stale Step 1). `levers.json` `refusals` maps refusal patterns to their prerequisite lever, and DREF proposes it. A refusal that no menu lever answers is crit, and DREF escalates it to a person with `OP_ESCALATE`, which puts the escalation card up. That covers a pattern mapped to no prerequisite ("needs" says what) and an unmapped refusal ("a person reads the refusal"). A pattern marked `retry: false` is one the identical build meets again, because it reads only Step 1's files and the build's flags: realloop's evidenced-pool capacity and select's draw shortfall. A campaign build that fails with such a refusal is declined for the campaign and never submitted a second time (`campaign._build_failed`). Any other failed build may be proposed again, and the stop-loss pauses the campaign after two consecutive failed steps.
- **Residual.** When the goal is not met, no D1–D4 rule fired, and nothing is in flight, the autopilot runs the brain and puts a human card up. It never sits idle silently.

### Lever catalogue

All in-menu levers are **existing** CLI flags. Every lever is a new experiment, because an experiment is built once. Every lever has a named control and a success criterion that `planner.make_experiment` requires (planner.py:73-113).

| Lever | Exact action | Policy id / risk | Control → success criterion | Literature |
|---|---|---|---|---|
| L1 replay full | `inc.pilot build --exp <parent>_rf --replay-mode full` via `run_inc_build.sh` | `inc_build_pilot` R3 | same definition with sample replay (the parent) → agreement ≥ 5/7 **and** at least one truth-`helps` step ACCEPTed | Ibrahim et al. 2024 TMLR; TIME 2024 (both cited at INCREMENTAL_PROTOCOL.md:206) |
| L2 real loop | `inc.realloop build --exp <n> --base <base_B> --replay-mode <D4> --recipes <D4 set> [--increment-sources evidence \| --relevance P] [--size S] [--n-verified N] [--gate-flips-mode M]` | `inc_build_realloop` R3 | the base arm (B) and truth → pre-registered in the doc | Ibrahim 2024 / LwF (doc :326) |
| L3 relevance | `sbatch run_inc_relevance.sh build` | `inc_relevance_build` R2 | calibration on `train_core` (tau) → every source gets a status | DINOv2/v3 retrieval curation (doc ~:243) |
| L4 label audit | `sbatch run_inc_audit.sh --trusted $M/P0.jsonl --audit I1=… Bswap=… Breal=… --out $INC/<exp>/audit/label_audit.json` (exactly `run_inc_audit.sh:30-36`) | `inc_label_audit` R2 | `known_truth.label_error_rate`; the planted Bswap must come out `above_baseline` | Northcutt et al. 2021, confident learning (**candidate, not in repo**) |
| L5 size | L2 with a larger `--size`; every other flag is the parent loop's own: `--base` from `exp.json base.source_manifest` (Step 1's `base_B.jsonl`, not the experiment's copy, which has no `select_summary.json` beside it), `--replay-mode`, `--recipes`, `--relevance`, `--n-verified` from `build_summary.json`, and `--no-truth` when the parent had no truth arm | inside `inc_build_realloop` R3 | same seeds → P_data leaves (0.25, 0.75) | Bouthillier et al. 2021 (doc :79) |
| L6 drop truth arm | L2 with `--no-truth` | inside R3 | allowed only after D11 | – |
| L7 transient unblock | `driver unblock --exp X --unit U --reason "auto: <cause>"` | `inc_unblock_transient` R2 | at most one per unit; the cause must be on the allow-list | – |
| L8 baseline B | `inc.pilot build-baseline --exp base_b_v1 --manifest …/base_B.jsonl` vs skip (doc :255-260) | `inc_build_baseline` R3 | B0 → B vs B0 on dev | – |
| L9 gate v2 pilot | `inc.pilot build --exp <child> --replay-mode <parent's replay_mode> --gate-flips-mode net`, from D15 on a v1 pilot (`only_after: D15`; `fixed` `gate_flips_mode: net`; `--replay-mode` always stated, the parent's) | `inc_build_pilot` R3, envelope-eligible | the parent pilot, same definition and replay mode decided by v1; the child's own report carries each step's v1 counterfactual → the pre-registered `v2_check` is `supported` **and** D4 agreement ≥ 5/7. Falsifier: `refuted` | Pioneer Agent (2604.09791, the flips guard); BCWI (2301.10546, negative flips) |
| X1 LR re-warm/re-decay | change `train.py` `PROTOCOL_*`, a **pinned** module | R4 (human card, never queued) | – | Ibrahim 2024 |
| X2 enforce leaf-disease in relevance | code change, plus a protocol-doc change | R4 | – | – |
| X3 audit feeds attribution | a new report field; `gate.py` is pinned, so this needs a new protocol version | R4 | – | Northcutt 2021 (candidate) |
| X4 leave-one-source-out | new code | R4 | – | – |
| X5 repin, outer-copy sync | `driver repin --reason`; rsync nested → outer | R4 / R3 | the sync refuses if it would change any pinned hash recorded in a running experiment's `state.code.modules` | – |
| X6 rethink the flips guard family | D15 on a `net` experiment: v2 still blocks truth-helps steps on its own; a new guard is a `gate.py` change and a new protocol version | R4 | – | Pioneer Agent |
| X7 v2 refuted | D16 `refuted`: revert the gate to v1 or pre-register a new guard; no L2 on the refuted gate, and no L1 or L5 rebuild carrying it (`gate_params` defers) | R4 | – | – |
| X8 test v2 before rebuilding a real loop | D15 on a v1 real loop: the flips guard alone blocked truth-helps steps; L9 rebuilds pilots only, so a person decides (cheapest test: L9 on the loop's source pilot). D1's levers on that loop are withheld while it stands | R4 | – | Pioneer Agent |
| X9 revisit the zero-shot relevance criterion | D2 on a calibration-failed `relevance.json`: the prompts, calibration and check are `inc/relevance.py` constants and a protocol section; prompts chosen on this pool would be fitted to it. A new relevance version is pre-registered and checked on `train_core` first, or the evidence criterion is kept as the only one | R4 | – | DINOv2/v3 retrieval curation |

- **The menu never contains** hand-written source exclusion lists (project rule), gate thresholds, or anything that overrides a per-increment verdict.
- **The gate on the menu: `--gate-flips-mode`.** It is the choice between the two pre-registered protocol versions of the flips guard (`negative` = v1, `net` = v2), not a threshold. The builder writes it into exp.json's gate block, and the driver pins it at init. `gate_flips_mode` is an enum of `driver.FLIPS_MODES` in levers.json and in the `inc_build_pilot` and `inc_build_realloop` rows of `brain/policy_actions.json` (template `[--gate-flips-mode {gate_flips_mode}]`).
  - **Forwarded from the parent's pinned config** (`levers.gate_params`, reading `diagnose.pinned_gate`) by every lever that rebuilds a pilot or a loop:
    - L1: its parent pilot's mode;
    - L2 and L6 from D4: the mode of the pilot D4 evaluated;
    - L2 from D1, and L5: the parent loop's mode (`levers.loop_params`).
  - **Rendered only when it differs from the builders' default** (levers.json `protocol.gate_flips_mode_default`, checked against `driver.DEFAULT_FLIPS_MODE`). A v1 parent's command stays the one a person ran before the flag existed; R1's pilot_v2 command is unchanged.
  - Only L9 sets the mode, to `net`.
  - The flag is always the last token, the executor's rendering order (`executor.ARGV_FORMS`).
  - `remote.py submit` validates it against `driver.FLIPS_MODES` and checks a missing flag at the builders' default. A pilot build left at `negative` is therefore never L9.
  - A parent whose own final `v2_check` is `refuted` is not forwarded: `gate_params` defers (card X7 decides between v1 and a new guard).
  - The brain's plans may not choose it (`brain_plan.AUTOPILOT_SETS`); `validate.materialise` derives it the same way.
- **The relevance criterion on the menu: `--increment-sources`** (2026-09-27; docs/INCREMENTAL_PROTOCOL.md, Steps 2-3).
  - **What it is.** The choice between the two pre-registered relevance criteria of the verified and OTHER_HEAVY increments: `relevance` (the builders' default, the relevance file) and `evidence` (source-level species evidence). It is an enum that names no source, in levers.json (L2, L5, L6; `protocol.increment_sources_modes` and `increment_sources_default`, checked against `select.SOURCE_MODES` and `SOURCES_RELEVANCE`) and in the `inc_build_realloop` row of `brain/policy_actions.json` (template `[--increment-sources {increment_sources}]`).
  - **Who sets it.** Never a proposer. L2 and L6 from D4 take `evidence` only through `levers.increment_criterion`: the Step 1 relevance file is `calibration_failed` and the evidenced pool holds (N + 1) × M images, at the default N and M or at the N and M the sizing rule gives (D2, "Sizing"), whose non-default `--size` and `--n-verified` the L2 then carries. Otherwise they take `--relevance` with a matching file, or defer. L2 from D1 and L5 keep the parent loop's own criterion (`levers.loop_params` reads exp.json `step1.increment_sources.mode`), checked again before the rebuild (`levers.check_criterion`). A parent built with a `--min-evidence` other than 1 is not rebuilt.
  - **How it is rendered.** Only when it is `evidence`, in the slot `--relevance` takes, never together with it. `executor.ARGV_FORMS` renders the same order.
  - **How it is checked.**
    - `remote.py submit` validates the enum (`INCREMENT_SOURCES`, checked against `select.SOURCE_MODES`) and checks a missing flag at the builders' default `relevance`. It does not accept `--min-evidence`.
    - `evidence` with `--relevance` is refused on the lab side too, before anything is queued or sent:
      - by `levers.check_params`, so `levers.argv` never renders the pair;
      - by `executor.resolve_params` and `executor.render`, which cover an INC-page or approval request;
      - by `validate` (a brain plan's bounds check).
      `remote.py submit` refuses it on the cluster. All of them use one message, `model.EVIDENCE_WITH_RELEVANCE`.
    - The brain's plans may not choose it (`brain_plan.AUTOPILOT_SETS`). `validate.materialise` derives it from Step 1 at the plan's own `--size` and `--n-verified`; a plan that names neither gets the sizing rule's, as the deterministic L2 does.
- **Lineage: a lever is applied to a parent once.**
  - Diagnoses run again every tick, and D4 is campaign-wide, so a diagnosis keeps firing after its lever ran.
  - Before proposing a build, `levers.applied()` looks for an earlier application:
    - L1: a full-replay pilot initialised after the parent;
    - L2 or L6 from D4: a real loop with D4's replay mode and recipes on the same Step 1 base (on either relevance criterion: the criterion is not part of the match);
    - L2 from D1: a later loop that is the parent with full replay;
    - L5: a later loop that is the parent at a larger size;
    - L8: `base_b_v1`;
    - L9: a pilot initialised after the parent and pinned to `net` (the child L9 builds); a later pilot on v1 does not count, and the child's name steps past it;
    - any lever: a `lineage` record in the ticker's context, i.e. a build in flight.
  - **The gate is part of the match.** An earlier experiment counts only when it is decided on the mode the lever forwards (`levers.gate_mode`): the parent's for L1, L2 from D1 and L5; the evaluated pilot's for L2 and L6 from D4. So a v1 loop with the same replay mode and recipes never stands for the net L2 of a v2 pilot (it is built as the next `realloop_v<n>`), and a later full-replay pilot on the other gate never stands for L1.
  - If one is found, the lever is deferred and the existing experiment is named.
  - `child_name` steps past names that other experiments hold, never past an earlier use of the same lever. Neither remote.py's job-name dedupe nor approval-id idempotency can catch a renamed duplicate.
- **Cost.** Each build row takes a bounded `est_gpu_hours` parameter using the existing formula shape `{gpu_type:"v100", gpu_count:1, hours_param:"est_gpu_hours"}` (POL:179-214). The autopilot computes it as runs × images × epochs × the seconds-per-image measured in the parent's `gpu_hours`.
  - For L1 from pilot_v1: chain runs grow from 476–548 to about 1,540–3,000 images, so the estimate is about 30–50 SU. That is an estimate; pilot_v2's real cost recalibrates it.

## (c) The cluster research brain

- **Where it runs.** `run_inc_plan.sh` is sbatch only, following the rule that brains run on cluster models.
  - It is not `run_llm_infer.sh`, which has no `num_ctx`, no JSON mode and an 8,000-character cap (DS:4596).
  - It is not the lab loopback, whose verdicts are marked non-authoritative (RS:330-354).
  - Input is staged to /ocean by `slurm_sh` (base64). Output goes to `_brain/weed/inc_plans/<campaign>/<n>.json` and is pulled back with `slurm_sh cat`.
  - Timeout is 2 h. If it fails, the deterministic proposal goes ahead alone.
- **Input digest (all dev-only):**
  - fired and unfired diagnoses with their cites;
  - short evidence excerpts: per-step chain/truth tables and per-species deltas on dev;
  - the lever menu with each lever's track record;
  - lineage, remaining budget, and the residuals no rule explains;
  - the top passages retrieved from `docs/literature` (plain BM25 over line-addressed markdown);
  - older experiments of the campaign (the last three): report summary, chains, per-step table and final dev rows, while there is room. The per-step tables were added with the trimming below; before it, an older experiment showed only its summary, chains and final dev rows. They are the first thing cut (`brain_plan.PARENT_SECTIONS`; dropping `steps` there restores the old content).
- **Fitting the digest (`brain_plan.build_digest`).**
  - **Why.** On 2026-09-27 at 22:38Z, after realloop_v1, the lab campaign logged `brain_failed`: `digest is about 46923 tokens, over num_ctx 49152 minus the reply reserve`. The deterministic path went on alone, as designed. But the digest grows with the campaign, so a fixed window lost the brain just when there was most to reason about. The digest now fits itself to its limits and refuses only when its minimum does not fit.
  - **Context size.** `num_ctx` is chosen from the digest's own size (`choose_num_ctx`):
    - 1.25 × the token estimate, because the densest prompt measured ran 1.22 × the 2.76 chars/token estimate;
    - plus 16,384 tokens for the reply, because the planner is a reasoning model and its think block shares the window;
    - rounded up to a multiple of 8,192;
    - at least 49,152 (the old fixed size, so no digest gets less room than before) and at most `MAX_NUM_CTX` = 98,304. The token budget that keeps the chosen window within the cap is 65,536.
    - The digest records the value, and `run_inc_plan.sh` asks the server for exactly that. A caller may fix `num_ctx` instead (tests, the CLI's `--num-ctx`); the digest is then trimmed to `num_ctx` − 6,000.
  - **Why 98,304, and the GPU.**
    - The planner (qwen3.8:27b) declares `context_length` 262,144 in its manifest. The job keeps GPU-shared with one H100 80 GB (`--gres=gpu:h100-80:1`). It sets `OLLAMA_NUM_PARALLEL=1`, so there is one KV cache, not one per slot. It sets `OLLAMA_FLASH_ATTENTION=1`, so llama.cpp does not also allocate an f32 KQ buffer of `num_ctx` × batch × query heads (about 13 GB at 98,304 tokens with a 512 batch and 64 heads).
    - The weights are about 19 GB. The model's attention layout is not recorded in this repo. The sizing therefore **assumes (unverified)** the layout of a dense 32B such as Qwen3-32B: an f16 KV cache of 256 KiB per token (64 layers × 8 KV heads × 128 dims). That is an assumption, not a bound. A dense 27B with 16 KV heads, or with head_dim 256, needs twice that. A hybrid layout, where most layers hold no KV cache, needs a fraction of it.
    - So the cap is the largest multiple of 8,192 that fits even at twice the assumed KV per token (`KV_SIZING_FACTOR`):
      - at 98,304 tokens: 25.8 GB of KV, 50.8 GB in all with about 6 GB of runtime, as assumed; 76.5 GB of 80 at twice the KV;
      - the next step, 106,496 tokens, would need 80.8 GB at twice the KV;
      - 131,072 tokens would need 59.4 GB as assumed but 93.7 GB at twice the KV, and the full 262,144 would need 93.7 GB even as assumed. Either would move layers to the CPU.
    - The lower cap costs the brain nearly nothing, because the ssh line below already stops a digest at about 62–69K tokens. Raise the cap only after the job's layout line has shown the real KV per token.
    - The job refuses a `num_ctx` over the cap before the server starts. Two log lines settle the assumption on the first run:
      - before warm-up, `[layout]`: the model's own layout from `/api/show` `model_info` (`brain_plan.layout_line`). It gives the KV bytes per token against the assumed ones, and what `num_ctx` needs on this GPU. It is logged, never enforced.
      - after warm-up, `[ollama] resident`: how much of the model is on the GPU (`/api/ps` `size_vram` against `size`).
  - **The ssh line.**
    - The staged digest travels gzip+base64 inside one ssh argument, capped at `executor.PLAN_MAX_STAGED_CHARS` (96 KiB). A full digest runs 1.41–1.57 such characters per estimated token, so the line stops a digest at roughly 62–69K tokens, about where the context budget does.
    - `build_digest` measures the staged form exactly (`staged_chars`, the same bytes `_plan_segment` ships). It trims until that form is within `STAGED_BUDGET_CHARS` (the cap minus 1 KiB).
    - Lifting this limit, by shipping the file rather than a command-line argument, would be a transport change; it is not made here.
    - **Expect transport cuts from the next experiment on.** The older experiments' per-step tables add about 11K tokens: realloop_v1 with its last three older experiments is 45,703 estimated tokens locally with them and 34,815 without. The lab's live digest, built without them, was 46,923 tokens. With them it would be about 58K tokens and roughly 86–90K staged characters of the 97,280 budget. A digest over the line has the oldest per-step tables cut first (`over: transport`), recorded like any other cut.
  - **Trimming rule.**
    - **Never cut (the minimal digest):**
      - the fired diagnoses with every cite;
      - the deterministic proposals;
      - the current experiment's dev tables (`CURRENT_KEPT`): report summary, chains, per-step table, final dev rows, the per-species gate rows (per-species dev deltas and the per-seed values behind P_data, which D13 and novel per-species patterns are read from), and the exp.json and state heads;
      - the lever menu with each lever's track record;
      - the budget and the lineage summary (lever, parent, child, status);
      - a one-line summary of each older experiment: exp, type, replay_mode, done, agreement and gpu_hours_total, at its report address;
      - 4 literature passages, the top-ranked ones.
    - **Cut first to last** (`TRIM_ORDER`), one cut at a time, only while the prompt is over its token budget (`over: context`) or its staged form is over the ssh line (`over: transport`):
      1. `parent_steps`: older experiments' per-step tables, oldest first.
      2. `parent_tables`: older experiments' chains and final dev rows, oldest first, down to their one-line summaries.
      3. `literature`: passages from k = 12 down to 4, lowest-ranked first.
      4. `residual_detail`, in this order:
         - the summaries of silent diagnoses (ids and names stay);
         - the Step 1 excerpts;
         - the current experiment's build excerpt, then its increment definitions;
         - lineage records, down to their summary;
         - residuals, down to 160 characters each.
    - **Recorded.**
      - The digest's `trimmed` field holds the budget, the token estimate before and after trimming, the staged size before, and each cut: its stage, what it removed, which limit it was over, and the token estimate before and after it.
      - The prompt's TRIMMED section names every cut and tells the brain that a value absent for this reason is not evidence.
      - The campaign ledger's `brain_staged` entry carries `num_ctx` and the cuts.
    - **Refused** only when the minimal digest is over a limit (the context budget, `MAX_NUM_CTX`, or the ssh line). The result is then `brain_failed`, as before.
    - **Test-blind.** The cuts read only the dev-only sections, so which cuts are made cannot depend on a non-dev value. After trimming, the fired diagnoses and the current experiment's `CURRENT_KEPT` sections are compared with the untrimmed ones, and every section is checked again by `assert_dev_only`.
  - **Schema and deployment.**
    - A digest that sizes its own `num_ctx` and may carry a TRIMMED section is `inc-plan-digest/2`. A cluster copy from before this change accepts only `/1`: it would render the prompt without TRIMMED and ask for a fixed 49,152-token window. It refuses a `/2` digest by its schema (`not an INC plan digest (schema 'inc-plan-digest/2')`), so the lab sees `brain_failed` with that reason. The job never gets a mis-rendered or truncated prompt.
    - The current copy still runs a `/1` digest staged before the change, at 49,152.
    - The job imports the git-tracked nested copy, so `brain_plan.py` and `run_inc_plan.sh` reach the cluster only by commit, push and pull. They must land together with the lab's update, before the next brain tick.
  - **Verified** (tests/test_inc_ap_brain.py, `test_num_ctx`, `test_digest_trim`, `test_run_and_merge`, `test_script`) on the whole campaign's real evidence: pilot_v3 with its 4 older experiments (sha-pinned fixtures only), and realloop_v1 with all 5.
    - The two digests are 53,873 and 52,764 estimated tokens, over the old 43,152-token budget. Each stages untrimmed at `num_ctx` 90,112.
    - At `num_ctx` 49,152, each is trimmed to fit (42,772 and 41,663 tokens), with two cuts: the per-step tables of pilot_v1 and pilot_v2.
    - Every fired diagnosis cite (200 and 123) is kept and resolves, and every value shown still resolves from its `_at`.
    - The minimal digests are 33,902 and 22,919 tokens, so the smallest `num_ctx` that holds them is 39,902 and 28,919; one less refuses. Each has cuts from every stage down to the residuals. In each:
      - the fired diagnoses equal the untrimmed ones;
      - the current experiment's summary, chains, steps, final dev rows, gate rows and exp head are unchanged;
      - the menu, the budget and the deterministic proposals are unchanged;
      - the lineage is exactly its summary projection;
      - 4 literature passages remain, the top-ranked;
      - every older experiment with a report keeps its one-line summary.
    - Squeezed to where the last older experiment is cut to its line, only `parent_steps` and `parent_tables` cuts are made, and older experiments are never removed.
    - The full, the trimmed and the minimal digest are each byte-identical, cuts included, after every non-dev value across the campaign is changed (312 and 376 values).
    - `staged_chars` equals what `executor._plan_segment` ships (84,296 and 74,492 of 98,304 characters). A digest over the ssh line is trimmed for it, and one whose minimum is over the line, over `MAX_NUM_CTX`, or over a fixed `num_ctx` is refused.
    - The cap fits 80 GB at twice the assumed KV and the next step does not. `kv_layout` reads dense, 16-KV-head, hybrid and per-layer layouts from `model_info`. The script turns flash attention on before `ollama serve` and logs the layout line before warm-up (run with `/api/show`, `/api/tags` and `nvidia-smi` stubbed).
    - A `/1` digest still runs, and a copy that accepts only `/1` refuses a `/2` digest before the model is called.
    - Mutations of a scratch copy, each killed by the suite (in-code guards removed where they would mask one): the current steps and chains cut early, the gate rows cut, the lineage cut to nothing, fired cites truncated, the literature floor set to 0 or cut from the top rank, an older experiment's line dropped, the cap back to 131,072, the schema back to `/1`, the hybrid rule ignored, flash attention or the layout line removed from the script.
- **Output schema:**
  - `ranked_menu[{lever, params, rationale, evidence_cites[{artifact, line|pointer, value}], lit_cites[{paper_id, line, quote}], predicted{metric: dev_twelve|agreement, direction, magnitude}, falsifier}]`
  - `off_menu[{hypothesis, why_menu_insufficient, required_change, cheapest_test, control, success_criterion, lit_cites}]`
  - `stop_recommendation`
- **How `validate.py` checks it:**
  - a menu lever must exist, and its params must pass `policy._check_params`;
  - every evidence cite must resolve and its quoted value must match the snapshot exactly;
  - every literature quote must be a verbatim substring at the cited line;
  - any reference to test, or to exams other than dev, is refused;
  - a failed item is dropped and recorded, but the rest of the plan survives.
- **What happens to valid output:**
  - Menu proposals are filed as `tier2:<model>` through `approvals.propose`, shown next to the deterministic proposal.
  - Off-menu proposals become **R4 human research cards**, with a pre-registration draft built by `planner.make_experiment`.
  - When a human implements an off-menu change as a new protocol version, it joins `levers.json` with its literature and a replay case.
- **How proposals are checked against reality.** Every accepted proposal's `predicted` block is scored by `outcome.py` against `experiments.verdict` once the child experiment reports.
  - The noise floor is the dev spread of the b0_v1 seeds, not the M1 recipe keys at db.py:549.
  - Per-model prediction accuracy and citation-failure rate appear on the page.
  - More autonomy for the brain has to be earned by that record and a human has to grant it (§d). It is never self-granted.
- **Literature corpus (not found; must be built).**
  - One `docs/literature/<id>.md` per paper: a bib line, 5–15 verbatim passages at fixed line numbers, and the levers each passage informs.
  - Seed papers are the ones the protocol already cites: Bouthillier 2021, Ibrahim 2024, TIME 2024, Sorscher 2022, OWL-ST, DINOv2/v3 curation, and LwF.
  - Candidate: Northcutt 2021.
  - An R0 lab action `inc_lit_fetch` can pull an arXiv abstract into a pending corpus entry. Quotes count only after the corpus file is committed.
  - **Hard part:** verbatim passages have to come from the PDFs, and model paraphrase is caught only by the substring check.
- **Unknown situations**, honestly: the brain handles cases the deterministic layer cannot see (D13 contradictions, residuals, novel per-species patterns) by proposing hypotheses. What gets *executed* is still limited to menu levers or approved R4 work. Its reach is bounded by the menu and by human conversion of off-menu ideas.

## (d) Governance

- **Actors.** The existing `_ACTOR_RE` (POL:100-103) already accepts all of these, so the v1 ceiling does not change:
  - `round-scheduler:inc-autopilot` (R0–R2 direct);
  - `tier2:<model>` (brain; propose only);
  - `human:<email>`.
- **R3 approvals.** Because `round-scheduler` has no R3 cell, R3 items are filed directly with `approvals.propose` (AP:107-133). The executor then authorises **as `human:<decided_by>`** at execution time, which makes the approval itself the authority. The executor also:
  - passes real `budget_state`: the su_ledger fold of the campaign envelope, defaulting to a 300-SU sub-envelope of `budget.su_envelope` 1500 (db.py:488), plus a daily cap;
  - passes `resources` (`mongo_ok`, cluster reachability), so `_check_budget` and `_check_resources` (POL:624-667) finally run;
  - writes `su_ledger.record` from `report.gpu_hours` when an experiment finishes (V100 at 1.0 SU per GPU-hour).
- **Risk tiers:**
  - **R0 auto:** snapshot, status, report, diagnose, lit fetch.
  - **R1 auto:** `inc_advance` (it only submits runs already paid for at init).
  - **R2 auto, within budget:** relevance build, label audit, transient unblock (at most one per unit).
  - **R3 approval:** every experiment build, scancel or abandonment of an experiment, outer-copy sync.
  - **R4, human card only** (never queued, since AP:107-113 refuses R4): repin, every code or protocol change, LOCK and never-train changes.
- **Stop-losses:** 2 consecutive failed campaign steps; any D7, D10 or D14; a blocked unit that is not transient; more than 3 submissions of the same lever per campaign. Each sets `paused_reason`, and `/api/health/inc` turns crit.
- **Test split is never read in decisions**, enforced three ways:
  1. `evidence.py` keeps an allow-list of `dev` only. It drops every exam but dev under `final[].exams`, so an exam added upstream later is dropped too. `report.py:56` puts test in every report. It also drops any score stamp or `scores/<exam>.json` path whose exam is not dev, and never opens a score file.
  2. The digest serialiser asserts that no `exams.*` key other than dev is present.
  3. A metamorphic test (§f).
  - The truth arm and gate already decide on dev (`exp.json decision_exam = "dev"`). The page may *show* the report's final table to humans; decision code never reads it.
- **Invariants that stay:**
  - the driver's gate decides every increment;
  - one definition per experiment;
  - code pinning;
  - the builders' own refusals;
  - `decide` by a human only.
- **Test invariant to update on purpose.** `tests/test_brain_api.py:387` ("nothing in the loop reads a review back") already fails today because of `supervision_health.py`. It needs a deliberate rewrite with the reason recorded, not a silent deletion.
- **Autonomy (owner decision, 2026-09-27).** The platform runs the INC campaign itself: R3 builds of menu levers L1, L2, L5, L8 and L9 run without per-item approval inside the campaign envelope, once the replay tests pass (`autonomy: envelope` in the campaign config; off by default). Without that flag, R3 builds wait for approval. It is implemented as an "approve-within-envelope" rule granted to `round-scheduler:inc-autopilot` (not a `_CEILING` change): the executor records a self-approval with the lever, its cited diagnosis and the envelope balance, and every other governance rule (budget, resources, stop-losses, R4 human-only) still applies.

## (e) Dashboard surface

- **Routes** (all lab-local reads of the ticker's cached snapshots, so no per-request ssh):
  - `GET /api/inc/campaign`, `/api/inc/lineage`, `/api/inc/{exp}/snapshot`, `/diagnoses`, `/ledger?line=`, `/api/inc/replay`, `/api/health/inc`;
  - `POST /api/inc/campaign` (admin: enable, pause, goal, envelope);
  - `POST /api/inc/action/{verb}`, which goes through the same executor and `authorize` as `human:<email>`.
  - This avoids the `_CLUSTER_ACTIONS` limits: no positional arguments, `needs_cluster` refused in lab mode, and the dead `"not a recognised action"` fallback at DS:13574.
- **Page `/inc`**, linked from `/supervision/weed` and the project page. Cards:
  1. Campaign: goal, phase, current experiment, budget burn, pause and kill.
  2. Lineage tree: each edge labelled with its trigger diagnosis and approval id.
  3. Current experiment: a step × chain grid of verdict vs truth, blocked units, and jobs with cancel.
  4. Diagnoses with cited values linking to ledger lines.
  5. Proposals, deterministic next to brain, with approve and deny on the existing `POST /api/brain/{d}/approvals/{id}`.
  6. Human research cards (R4).
  7. Track record per lever and per brain model.
  8. The last replay and prospective test results.
- **Required fixes:**
  - add `inc_` to `SAFE_PREFIXES` (DS:12535-12540);
  - add `results/framework/inc/**/logs/*_<id>*.out` to the `/api/job_log` globs (DS:13347);
  - handle array ids `<id>_<task>` in `_batch_sacct` (DS:12593-12618).
- **Implementation (inc_dashboard.py, 2026-09-27 review).**
  - The routes read the ticker's own files through `campaign.Paths`: its state (phase, current experiment, item, cards, pause), `latest_snapshot.json`, the snapshot history, `frozen.json`, its ledger copy (`snapshots/<exp>/ledger.jsonl`, which only the ticker extends, using `through_sha256` and `prefix_mismatch`), `diagnoses.json` and `campaign_status.json`. The tests drive the real `campaign.tick()`, so they cover the formats the ticker actually writes.
  - The diagnoses shown are the ticker's record. When it has none for an experiment, a preview is built from the ticker's own inputs (`campaign._Run` readers with a zero ssh budget). The page labels it and `/api/health/inc` never reads it.
  - `POST /api/inc/campaign` goes through `campaign.configure`, `pause` and `set_goal`. The config write takes the scheduler's lock, and each change lands in the ticker's state and the campaign ledger. Goals are the two forms `check_goal` reads, and a goal may name no non-dev exam.
  - `/api/health/inc`:
    - crit on a pause in the config or in the ticker's state, a campaign not ticked for 30 min, a fired crit diagnosis recorded on the current experiment, or a tree the ticker does not write;
    - warn on a card waiting on a person, a COMPLETE campaign, ticker errors or failed snapshots.
    - A person's resume answers any record from before it.
  - Card 3 deviates from the list above. The page lists jobs without a per-job cancel. Stopping an experiment's runs or build is R3 (`inc_cancel_exp` through the executor, the page's "Pause and kill"), so `/api/cancel_job` refuses `inc_<exp>_NNNN` and `inc_build_<exp>`. Other `inc_` jobs need cluster access there and are logged with the actor.
  - INC writes ignore the `X-User` header. Envelope autonomy is granted only from a person's own account.
  - A real loop built or executed from the page (build-realloop, or execute-approved of an `inc_build_realloop` approval) passes `diagnose.prospective_guard` for its parent pilot (else the latest finished pilot): a READY `replay/prospective_d4_<pilot>__<rulesver>.json` under the current rules version, the file the ticker wrote, and a build that follows it. Otherwise it is refused (409) with its reason, unless the request carries an acknowledgement, which the execution log records.
  - A manual snapshot is kept in the ticker's history format as `<stamp>.manual.json`.

## (f) Replay test: `tests/test_inc_autopilot_replay.py`

- **Fixtures** live in `tests/fixtures/inc_replay/`, with sha256 pinned in a manifest test:
  - `pilot_v1/` `{exp, report, ledger, build_summary}` (local);
  - `b0_v1/`;
  - `step1/` `{select_summary, admit_summary, per-source aggregates of select_clusters.csv, increments_summary, relevance.json}` (**must be pulled once from /ocean**; the snapshot verb can produce it).
  - `pilot_v2/` `{exp, report, ledger, build_summary}` and `base_b_v1/` `{exp, report, build_summary}`: byte copies of the local results, pinned 2026-09-27 (MANIFEST.json `added`).
  - `pilot_v3/` `{exp, report, ledger, build_summary}`: byte copies of the local results once pilot_v3 was done, pinned 2026-09-27 (MANIFEST.json `added`) for R6. A test that replays an earlier decision loads the tree as it stood then (R5: `exps` without pilot_v3), since D4 is campaign-wide and L9 on pilot_v2 is applied once pilot_v3 exists.
  - Tests that replay intervention 1 load the tree as it stood then (`exps=["pilot_v1", "b0_v1"]`). A whole-tree load now also holds pilot_v2, so L1 on pilot_v1 is applied, and D4 looks at pilot_v2.
  - `synthetic/step1_calibration_failed/` `{select_summary, admit_summary, select_clusters_by_source, relevance}` for R7, pinned 2026-09-27 under MANIFEST.json `synthetic` (sha256, bytes, and the Step 1 file each stands for; `synthetic_why`). It is synthetic, not a results copy.
    - Its `relevance.json` has the shape of the cluster's failed build and its calibration numbers (tau 0.0557 < tau_min 0.5; 34.5% of `train_core` crops below 0.5). It is made for this select build, so `relevance.load` refuses it for its calibration, not as malformed.
    - Its evidenced sources (three with the real sources' numbers, two named `synthetic__`) hold 3,213 increment-pool images, which fits the default loop.
    - It lives outside `step1/` and every experiment directory, so no loader reads it in place of the pending real Step 1. `tests/test_inc_ap_fixtures.py` checks its hashes.
  - `step1_copies/` `{select_summary, admit_summary}` for R8: byte copies of the local `results/framework/inc/step1/select_summary.json` and `admit_summary.json` (base B 3,927 images; five evidenced sources holding 1,439 of 96,088 increment-pool images), pinned 2026-09-27 under MANIFEST.json `step1_copies` (sha256, bytes, the results file each was copied from and the Step 1 file each stands for; `step1_copies_why`). Whether they equal /ocean's was not checked (no ssh). They live outside `step1/`, whose cluster pull R2 still waits for.
  - `synthetic/step1_real_calibration_failed/relevance.json` for R8, pinned under MANIFEST.json `synthetic`: R7's calibration numbers, made for the real select build (its inputs carry that build's table sha256 and the sha256 of the pinned select summary), so `relevance.load` refuses it for its calibration. Its per-source tables are empty: the cluster file's per-source scores were not pulled, and nothing reads them for a calibration-failed file.

| Case | Input | Asserted output |
|---|---|---|
| R1 | pilot_v1 | D1 fires with cites: 20/21 `p_recipe` 0.0, 21/21 `null<inc`, 9/9 non-ACCEPT on I2, I3 and I5. Lever L1; argv equals `inc.pilot build --exp pilot_v2 --replay-mode full` (the manual command). X1 appears only as an R4 card. No realloop is proposed. |
| R2 | step1 without relevance.json | D2 names `fvossel__csgo_player_detection` and `rf_bishwarup-halder__crop-health-advisor`; lever L3, then L2 with `--relevance`. With relevance.json present: D2 is silent, D2b fires on the tomato-leaf source if it passes, and X2 appears as a card. |
| R3 | pilot_v1 | D3 fires, citing `exp.steps[2].planted`, `chains.*.bswap.attributed_to_labels=false` and `attribution_not_run`. Lever L4; argv matches `run_inc_audit.sh:30-36`. D3b fires on `s05_Breal`; X4 appears as a card. |
| R4a | pilot_v1 | `no_recipe_tracks_truth` (0.4286 < 5/7). **No** L2 proposal. |
| R4b (prospective) | pilot_v2 report when it lands | D4 output is written to `prospective_4.json` and **committed before** the human's realloop decision. The test later compares it with the manually chosen `--replay-mode` and `--recipes`. This is the only case that counts as evidence of generalisation. |
| R5 | pilot_v2 (full rehearsal, gate v1) | D15 fires with cites of the five flips-only rejections, pooled over chains (freeze I2 line 15, I3 line 22; full I2 line 13, I3 line 19; lora I3 line 23; each REJECT at `P_data` 1.0, flips failed, regression and species passed, truth `helps`). Three truth-helps misses failed another guard too (freeze I5 line 31: regression and species; lora I2 line 18 and I5 line 32: species), so D1 is **not** blocked: it fires with X1, a card. The one build proposal is L9, ranked first, with argv exactly `python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v3 --replay-mode full --gate-flips-mode net`. D4 proposes no real loop. The same holds through remote.py's snapshot and under the test-blindness perturbation. Written with the evidence in view, so it is a reproduction test. `executor.REPLAY_REQUIRED` includes R5, so a replay pass (and with it envelope autonomy) needs it. |
| R6 | pilot_v3 (full rehearsal, gate v2 `net`; pinned after its result) | D16 `supported` (v2 11/21, its v1 counterfactual 9/21). D1 fires (18/21 `P_recipe ≤ 0.25`, 19/21 `null_mean < inc`, 3/9 truth-helps not ACCEPT: freeze I3 and I5, lora I2) with X1 as a card, but `blocks_d4` is false: the chain D4 selects, full (5/7, by rate), ACCEPTed I2, I3 and I5. D4 `decision_slot_ready` with recipes `['full']`, replay mode `full`, gate `net`, and the prospective record says so under the current rules version. With no relevance.json the L2 waits for L3 first. With one made for the select build in view (the pinned Step 1 files once pulled, else a synthetic `select_summary.json` and a matching `relevance.json`), the L2 argv is exactly `python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl --replay-mode full --recipes full --relevance <INC_DIR>/step1/relevance.json --gate-flips-mode net`, the `--relevance` flag where the L2 template renders it, and `realloop.py`'s own argparse reads it back. The same holds under the test-blindness perturbation. The refinement it rests on was decided after pilot_v3's result (D1), so R6 is a reproduction test. `executor.REPLAY_REQUIRED` includes R6. |
| R7 | pilot_v3 + `synthetic/step1_calibration_failed/` copied to `step1/` (a `relevance.json` made for the select build whose own calibration check failed) | `relevance_status` is `calibration_failed`. D2 fires on the unevidenced sources (csgo, tomato-leaf) but does **not** escalate: warn, card X9, no `OP_ESCALATE`, no L3. Its detail records the criterion (`evidence`), why, and the capacity (5 evidenced sources, 3,213 images ≥ 2,751 = 7 × 393), with the flagged sources among the excluded ones. D4 is R6's. The one L2 argv is exactly `python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl --replay-mode full --recipes full --increment-sources evidence --gate-flips-mode net`. Checked against the builders and the gates: realloop's argparse reads it back (evidence, no relevance file, default `--min-evidence`); `select.load_evidence` finds the same evidenced sources; the executor re-renders it inside the policy row's bounds; it follows D4's READY record (`prospective_guard`). The same holds under the test-blindness perturbation. **Unchanged cases:** with a passing `relevance.json` the L2 is R6's `--relevance` command and D2 is silent. The `then` L2 is not also listed as deferred, because D4's L2 carries it. D2 still fires `warn` with X9 when the per-source aggregate is missing: the L2 is still proposed with `--increment-sources evidence`. When `admit_summary.json` is missing, D2 escalates (crit, `OP_ESCALATE` and X9) and the L2 is deferred. **Negatives:** each of these still escalates with no L2 and no L3: a stale file, or a malformed one. Malformed covers ok false at tau ≥ tau_min, ok true at tau < tau_min, a failed check without its tau_min, `params.tau_min` 0.3 (made under another rule), and another format. R7's Step 1 carrying the real Step 1's numbers (1,439 images < 2,751) gives R8's sized L2 (it escalated before the sizing rule). Written with the criterion's design in view, so it is a reproduction test. `executor.REPLAY_REQUIRED` includes R7. |
| R8 | pilot_v3 + the real Step 1 (`step1_copies/` select and admit summaries) + `synthetic/step1_real_calibration_failed/relevance.json`, copied to `step1/` | The evidenced pool (5 sources, 1,439 images) cannot hold the default loop (N 6 × M 393 = 2,751). D2 does **not** escalate: the sizing rule (D2, "Sizing", the R4 decision of 2026-09-27) gives N = 4 (six decided increments) and M = min(393, floor(1,439 / 5)) = 287 (7.3% of B, above the floor 196.35); warn, card X9, no `OP_ESCALATE`, no L3. Its detail records the default (6 × 393, 2,751, does not fit, at most N 2) and the sized (4 × 287, 1,435, fits) N and M, the rule, the capacity at the sized N and M, and `then` `L2 with --increment-sources evidence --size 287 --n-verified 4` (not also deferred). D4 is R6's. The one L2 argv is exactly `python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl --replay-mode full --recipes full --increment-sources evidence --size 287 --n-verified 4 --gate-flips-mode net`, the flags in the L2 template's order. Checked against the builders and the gates: realloop's argparse reads every flag back (N 4, M 287, evidence, no relevance file, default `--min-evidence`); `realloop.sequence(4)` has 6 steps; `select.load_evidence` finds the same evidenced sources; the executor re-renders it inside the policy row's and L2's bounds; `validate.materialise` gives a brain L2 without `--size` and `--n-verified` the same request, checks one that names 4 × 287 at those values, defers one that names only `--size 287` (7 × 287 > 1,439), and refuses with the numbers ones below the R4 floor (4 × 196, 3 × 287, 1 × 50); a cluster refusal of the sized loop (realloop's evidenced-pool capacity, or select's "the draw found") makes DREF fire crit with `OP_ESCALATE`, marked `retry: false`; it follows D4's READY record; it is priced at 4 × 287. The same holds under the test-blindness perturbation. **Variants:** an evidenced pool of 980 images (sized M 196 < 196.35) escalates with the numbers, no L2 and no L3; 985 images (M 197) is sized to `--size 197 --n-verified 4`; 2,000 images (≥ 5 × 393) renders only `--n-verified 4`; no `admit_summary.json` escalates. The rule was decided with the real Step 1's numbers in view, so R8 is a reproduction test. `executor.REPLAY_REQUIRED` includes R8. |
| Negative controls | b0_v1; base_b_v1; pilot_v1 for D15/D16; a synthetic healthy pilot built with FakeBackend (as in `tests/test_inc_driver.py`) whose chains track truth | None of D1–D4, D15, D16 fire. |
| Test-blindness | every fixture with every test/ood/imageweeds value perturbed | Diagnoses, levers, argv and digest bytes are identical. |
| Earliest fire | ledger prefixes of pilot_v1 | Reports the first gate entry at which D1 fires (a GPU-hours-saved number). Only asserts that it fires by the end. |
| Governance | each proposal | `authorize` gives: R2 direct for `round-scheduler:inc-autopilot`; R3 goes to approvals; R4 is refused and shown as a card. The executor refuses to run a second time on the same approval id. |

- **Honest caveat:** R1, R3, R5, R6, R7 and R8 were written with the fixtures in view, so they are reproduction tests. R6 also tests a rule refinement made after its own result; the real loop's truth arm is that refinement's prospective test. R7 tests D2's reading of the failed relevance build; whether the evidence criterion draws useful increments is the real loop's to show. R8 tests the sizing rule on the numbers it was decided with; whether the gate resolves effects at M = 287 is the sized loop's to show (D8 `gate_underpowered`).
- **End to end: `tests/test_inc_ap_e2e.py`.** It drives `campaign.tick()` on a simulated clock. A fake `slurm_sh` runs the real remote.py verbs on an INC_DIR laid out from the fixtures (pilot_v1, pilot_v2, base_b_v1) and a synthetic Step 1 with no relevance.json. The campaign then builds synthetic children: pilot_v3 is pilot_v2 pinned to v2, with full-chain agreement 6/7 and `v2_check` `supported` or `refuted`.
  - **The cycle:**
    1. pilot_v2 finished → D15 (five pairs) → L9, ranked above D2's L3 and D3's L4; D1 fires unblocked, its X1 a card.
    2. Without a replay pass it waits; with one, the envelope grant (the autopilot's grant, run as the person who enabled it) executes it once.
    3. RUN pilot_v3 → REPORT (spend, outcome) → D16 info, D4 ready, and the prospective record (gate `net`, `supported`) written before any L2.
    4. D2 on Step 1 → L3 executed → WAIT_JOB.
    5. relevance.json appears → L2 with `--relevance` and `--gate-flips-mode net`, D4's replay mode and recipe → executed once → RUN realloop_v1.
  - **The same cycle in one DIAGNOSE:** with relevance.json present from the start, pilot_v3's prospective record and the L2 proposal land in the same tick, the record first. A reordering inside `_diagnose` (the record after the proposal) fails this case: the L2 is then held back by the frozen-record guard and is not proposed in that tick.
  - **The refuted branch:** D16 crit, card X7, D4 `blocked_by: D16`, the prospective record not ready. L3 still runs, no L2 is ever proposed, filed or run, and the campaign completes with a residual card.
  - **Budget exhaustion:** the envelope fits L9's estimate with 0.5 SU spare, but pilot_v3's runs cost 1.3x the measured rates, more than the estimate's `experiment_hours`; with its build job that passes the envelope. The next charge is refused on budget, the campaign pauses, and later ticks make no ssh. At the measured rates (1.0x) it does not pause.
  - Every tick of every scenario makes at most one ssh.

## (g) Ordered implementation plan

Sizes: S ≤ ½ day, M 1–2 days, L 3–5 days.

1. **S. Freeze fixtures.** Local pilot_v1 and b0_v1 now. Test: `test_inc_ap_fixtures.py` (hashes).
2. **M. `evidence.py`, test-blind.** Test: `test_inc_ap_evidence.py` (dev-only allow-list; metamorphic).
3. **M. `diagnose.py` and `thresholds.json`.** Test: `test_inc_ap_diagnose.py` (each D fires or stays silent on fixtures and synthetic data).
4. **M. `levers.json` and `levers.py`.** Test: `test_inc_ap_levers.py` (argv byte-equal to the documented human commands; bounds).
5. **S–M. Replay test** (§f; R2 marked skip until the step1 fixture is pulled).
6. **S. `policy_actions.json` rows:** `inc_snapshot`, `inc_report`, `inc_advance`, `inc_relevance_build`, `inc_label_audit`, `inc_unblock_transient`, `inc_build_pilot`, `inc_build_realloop`, `inc_build_baseline`, `inc_cancel_exp`, `inc_sync_outer`, `inc_lit_fetch`.
   - Experiment names use `str` patterns `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`; replay_mode and recipe subsets are enums; `est_gpu_hours` is a float in [0, 200].
   - Tests: extend `test_policy.py`, plus an authorize matrix.
7. **M. `remote.py` and `run_inc_build.sh`** (cluster root and lab root). Test: `test_inc_ap_remote.py` with a temporary INC_DIR and FakeBackend; marker parsing.
8. **L. `campaign.py`, the RS `_loop` hook, `executor.py`, `approvals` `executed` record, su_ledger writer, and budget/resources wired into `authorize`.** Test: `test_inc_ap_campaign.py`, a simulated clock with fake `slurm_sh` running a full cycle: pilot_v1 done, D1, proposal, approval, execution, RUN; plus stop-loss and idempotency.
9. **M. Dashboard routes, page and the three monitoring fixes.** Tests: route tests on fixtures; `test_policy_gate_paths` drift.
10. **L. `run_inc_plan.sh`, `brain_plan.py`, `validate.py`.** Tests with a mock client: a bad cite drops the item; out-of-bounds params are refused; a test leak is refused; a timeout lets the deterministic path continue.
11. **M. `docs/literature` corpus, index and validator.** Test: every seed quote resolves.
12. **M. `outcome.py` and track record.** Test: verdict from a child report; noise floor from b0_v1.
13. **Operations.** Run shadow mode on pilot_v2 (diagnose and propose only), commit R4b, then enable execution for R0–R2 and route R3 through approval.

Steps 1–6 are offline and can proceed while pilot_v2 runs. Total is about 20–25 developer-days.

**Must NOT change:**
- `inc/__init__.py`, `inc/common.py`, `inc/driver.py`, `inc/gate.py`, `inc/splits.py`, `cwd12_species.py` (`PINNED_MODULES`, driver.py:223-224);
- `inc/scorer.py`, `inc/train.py`, `inc/lora.py` (run spec and protocol-recipe hashes);
- `run_inc_job.sh` (the array script for experiments in flight);
- `LOCK.json`, split manifests, the never-train index;
- any existing `INC_DIR/<exp>/` content;
- `GateConfig` thresholds;
- `db.ROUND_STEPS` and the `validate_step_command` allow-list semantics (INC stays out of the weed round ledger);
- `_CEILING` (v1).

New code lives in `tools/inc_autopilot/`, **not** `tools/inc/`, so the pinned `inc/__init__.py` and the INC scripts' outer/nested drift checks are untouched.

**Not verified:** anything on the cluster (the pilot_v2 state, the `step1/*` values, which package copy the human currently runs `driver advance` from), and whether `approvals.propose` validates the `requested_by` format.

**Relevant paths:**
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/results/framework/inc/pilot_v1/report.json`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/results/framework/inc/pilot_v1/ledger.jsonl`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/results/framework/inc/pilot_v1/exp.json`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/weed_optimizer_framework/tools/inc/driver.py`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/weed_optimizer_framework/tools/brain/policy.py`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/weed_llm_benchmark/run_inc_audit.sh`
- `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/docs/INCREMENTAL_PROTOCOL.md`