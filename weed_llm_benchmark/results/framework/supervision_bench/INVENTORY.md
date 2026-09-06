# Supervision benchmark - incident inventory (WP2)

**Status: candidate list, nothing verified on the cluster.** Every artifact path below is a
prediction derived from the `#SBATCH --output` line of the script that produced the job, and every
`provenance` value is a judgement about what is likely to have survived on `/ocean`, not an
observation. The export tool must check survival (`ls -la`, `sacct -j`) before promoting a case to
`raw`. Machine-readable source of this table: the WP2 inventory JSON produced alongside it.

Sources read in full: `weed_llm_benchmark/CHANGELOG.md`, `RESEARCH_LOG.md`,
`docs/PLAN_COMPLIANCE_AUDIT.md`, `docs/SCIENCE_AUDIT.md`, plus `docs/TIERED_SUPERVISION_PLAN.md`
sections 0/1/4 and `docs/TIERED_SUPERVISION_EXECUTION.md`. Dates were taken from commit dates where
a version tag exists, otherwise from the entry's own heading. Job ids and per-run numbers were
cross-checked against `results/framework/figures_data.json`.

Cluster repo root for every relative path: `/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark`.

## 1. What is in here

| kind | n |
|---|---|
| incident cases, total | 127 |
| ... in benchmark scope (cluster pipeline + campaign governance) | 116 |
| ... inventoried only (platform UI, robot uplink - no `/ocean` artifact) | 11 |
| healthy controls (completed steps and sealed runs) | 31 |
| false-alarm and environment controls (correct verdict: no incident) | 4 |
| **cases in this file** | **162** |

WP2's gate asks for at least 12 raw incident cases and at least 10 healthy. The predicted raw count
in benchmark scope is **49** and the healthy-control count is **31**, so both have margin even if a
third of the survival predictions turn out wrong.

## 2. Counts by correction class and by actor

Incidents only; healthy and designed controls excluded.

| correction class | all incidents | in benchmark scope |
|---|---|---|
| config | 32 | 32 |
| code | 72 | 62 |
| design | 16 | 15 |
| plan | 7 | 7 |

| actor who corrected | all incidents | in benchmark scope |
|---|---|---|
| assistant | 108 | 102 |
| human | 16 | 11 |
| agent | 0 | 0 |
| none | 3 | 3 |

`agent` is zero. In the whole record no autonomous LLM originated a correction; the tool-caller era
produced tool choices inside a round and never a campaign parameter.

**Reading.** Code edits dominate every era. A supervisor whose action space is bounded config (the R1/R2 tiers of the governance plan) can reproduce the config and plan rows and none of the code rows, so correction correctness must be scored on the config subset only and detection recall reported per class. `actor=human` marks the cases a person caught, which are disproportionately the silent-wrongness ones (garbage on the dashboard, the holdout leak, training that had quietly become cwd12-only, an ownership hole) - exactly the class a metric-only supervisor is blind to.

Signals the cases are expected to fire, by frequency:

| signal | cases |
|---|---|
| none | 56 |
| walltime_bound | 20 |
| pool_growth | 18 |
| gate_noop | 13 |
| stale_artifact | 9 |
| plateau | 7 |
| source_degraded | 7 |
| job_unknown | 6 |
| ownership_violation | 4 |
| budget | 2 |
| mongo_down | 1 |

`none` is the largest bucket by a wide margin. Roughly half of the recorded failures are not
reachable by any deterministic artifact invariant in the WP3 list - they are library-version
mismatches, path assumptions, ownership holes and experimental-design errors. That is a finding
about the ceiling of the signals arm, and it should be stated before any arm is scored rather than
discovered afterwards.

## 3. Raw versus record-only

Rule used:

- RAW if the case's evidence is a committed results JSON, a registry/scorecard field, a run directory that a later job demonstrably re-read, or a job dated 2026-08-22 or later (campaign era, inside the retention the campaign itself relied on).
- RECORD-ONLY if the evidence is a March-July SLURM .out (five to six months old, from jobs whose directories have since been rebuilt), a lab-server journal line, transient process state, a code-review finding with no run, or the ABSENCE of an artifact.
- A case can be raw even when the defect is invisible in the artifact: the 2026-08-23 stale-metric case is two mtimes.

| | n |
|---|---|
| raw total | 59 |
| record only total | 72 |
| raw incidents | 55 |
| record only incidents | 72 |
| raw incidents in benchmark scope | 49 |
| raw incidents dated 2026 08 or later | 36 |
| record only incidents dated before 2026 08 | 59 |

**Reading.** 49 in-scope incidents are predicted raw, comfortably above WP2's gate of >= 12 raw incident cases, and the healthy-control side is not close to the limit either. The prediction is deliberately conservative for anything before 2026-08: those .out files were written by jobs whose run directories have been rebuilt many times, and several were self-chaining multi-day jobs whose logs are large. Where a pre-August case turns out to have a surviving .out it should be promoted to raw, because those are the cases that most enlarge the class coverage of the corpus (the v3.0 era is where nearly all the code-class incidents live).

Check these first, in this order - each one covers more than one case:

- results/framework/m1_merged_rndtrain_s1_44727703.out and _44767709.out - the dev case; if these are gone the whole WP2 premise weakens.
- results/framework/mega_iterm1_curated_s101/train*/ - carries the dev case's partial results.csv AND the stale-metric pair AND round 1/2/3's own metrics; one directory covers four cases.
- results/framework/sample_audits/*.json - the dual-written audit verdicts that survived a registry rollback; they are the evidence for two registry incidents and the whole S1 gate table.
- results/framework/*_40962044.out and *_40963121.out - if these survive, a two-case record gap becomes recoverable science.

## 4. Proposed splits and what they can measure

### 4.1 Date split (the plan's rule)

Rule: dev = in-scope incidents dated on or before 2026-08-25, plus the 2026-08-29 walltime case (sched-walltime-double-timeout-20260829) which is the worked example; test = every in-scope incident dated after 2026-08-25 except that one. Healthy controls and the false-alarm/environment controls follow the same date rule.

Why this date: 2026-08-25 is when the campaign's design work was written down (the retraction, the corrected ladder, the S1 verdict). Everything after it - the 22-day sync outage's real diagnosis, the class-order permutation, the robot-surface defects, the walltime pair, and the WP1-era findings - was found or fixed after the signal definitions were conceived, so a time split is the only way to separate 'the detector was written from this incident' from 'the detector generalised'.

| cell | incidents | config | code | design | plan | healthy | other controls |
|---|---|---|---|---|---|---|---|
| dev | 107 | 30 | 55 | 15 | 7 | 30 | 2 |
| test | 9 | 2 | 7 | 0 | 0 | 1 | 2 |

**The date split is badly unbalanced: the v3.0 era supplies most of the corpus and all of it is dev. The test half is small and is dominated by code-class and governance cases. Reporting a per-class recall on the test half alone will be uninformative for at least one class; the leave-one-class-out pass over the whole in-scope corpus is the more usable of the two designs, and the date split's job is to answer the hostile question about when the detectors were written, not to carry the headline number.**

Test-half case ids: `sched-silent-pause-six-days-20260829`, `v3-25-2-cpu-train-on-gpu-allocation-45250479`, `v3-25-3-probe-repointed-campaign-state-45290759`, `v3-25-3-vllm-verify-kv-dtype-misreported`, `v3-25-1-step-command-allowlist-refused-on-lab`, `wp1-job-scoped-artifact-would-deflate-a-published-std`, `wp1-array-jobid-key-would-detach-metric`, `wp1-qwen38-deploy-404-old-ollama-45197085`, `wp1-vllm-fp8-kv-cache-abort`.

### 4.2 Leave-one-class-out (over the whole in-scope corpus)

Rule: Hold out every in-scope incident of one correction class; report recall on the held-out class using thresholds and prompts frozen on the other three. Four passes.

| held-out class | held out | remaining | healthy controls available |
|---|---|---|---|
| config | 32 | 84 | 31 |
| code | 62 | 54 | 31 |
| design | 15 | 101 | 31 |
| plan | 7 | 109 | 31 |

### 4.3 Smallest difference these samples can distinguish

**The plan's date rule, applied to the record as it stands today, produces a test half of 9 in-scope incidents (6 of them raw). That is too small to carry any comparative claim. The corpus is 116 in-scope incidents and 92 percent of them predate 2026-08-25, because that is when the campaign's design work was written down. The usable design is therefore leave-one-class-out over the whole in-scope corpus for the headline numbers, with the date split reported separately and only as the answer to 'were your detectors written before or after you saw the incident they detect'. The test half will grow with WP8's live-run incidents; it should be re-counted then, and split.json should be frozen only once, before any model run, whichever way that count lands.**

- Unpaired: Two arms compared as independent proportions on the test incidents: at n=9 a 95% Wilson interval around a recall of 0.5 has a half-width of about 0.273 (interval 0.227 to 0.773), so two arms whose recalls differ by less than roughly 0.55 cannot be separated. Restricted to the raw-artifact test cases (n=6) the half-width is about 0.312.
- Paired: The arms see the same bundles, so the honest test is McNemar on discordant cases. If every disagreement favours one arm, six discordant cases give a two-sided p of about 0.031; that is the floor, and a realistic mixed pattern needs ten or more. The date-split test half holds only 9 incidents, so a paired test there can reach significance only if at least six of the 9 disagree and every one of them favours the same arm - a recall margin of about 0.67. Anything smaller must be reported as a count with a Wilson interval and the sentence that it cannot separate arms, not as a p-value.
- Whole in-scope corpus: Using all 116 in-scope incidents (leave-one-class-out rather than the date split), the paired floor of six discordant cases is a margin of about 0.05, and the unpaired Wilson half-width around 0.5 is about 0.09.

Consequence for the claims: C2 (marginal value of a model over the signals alone) is measurable only if the margin is large; a margin of 0.05-0.10 is not resolvable at this sample size and must be reported as 'not distinguishable at n=9' rather than as a null result. Per-class recall on the date-split test half is the weakest cell of all and should be reported with its interval or not at all. Three repeats at temperature 0.3 reduce within-case sampling noise and do nothing for the case-count limit - repeats must never be counted as n.

Ways to improve this without inflating n:

- Promote pre-August cases to raw wherever an .out survives; each one is a real extra case.
- Score per-signal rather than per-case where a case carries two signals (walltime_bound and pool_growth on the dev case), stating clearly that these are not independent.
- Report the false-alarm rate on the 31 healthy controls plus the 4 designed controls, where n is larger and the measurement is cheaper.
- Treat the four false-alarm/environment controls as a separate reported number, not as part of recall: they are the cases where the correct verdict is 'no incident'.

## 5. The cases

Sorted by date within each era. `class` is the correction class, `actor` is who made the correction.

### A. LLM tool-caller era (v1.x-v2.7, 2026-03-30 to 2026-04-14)

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `v2-hyperagent-qwen-worse-than-seed` | 2026-03-30 | Every Qwen2.5-7B-proposed strategy lost to the human seed | design | assistant | v1.0/HyperAgent | - | plateau | record-only |
| `v2-agent-inspect-loop-38373824` | 2026-04-02 | Agent mode v2: Brain answered '1' twenty times and never trained | code | assistant | v1.2 | 38373824 | none | record-only |
| `v2-agent-json-loop-38354715` | 2026-04-02 | Agent mode v1: Brain could not emit an action, looped 30x on the fallback | code | assistant | v1.1 | 38354715 | none | record-only |
| `v2-deepseek-no-tools-38432901` | 2026-04-04 | DeepSeek-R1:7b returned HTTP 400 'does not support tools' on every call | code | assistant | v1.2/v2.0 | 38432901 | none | record-only |
| `v2-ollama-coldstart-timeout-38477380` | 2026-04-04 | First Brain call timed out on a 5-minute model cold start | config | none | v2.0 | 38477380 | none | record-only |
| `v2-labeldir-zero-f1-38477380` | 2026-04-05 | Round 1 scored old_f1=0 / new_f1=0 from a label-directory bug | code | assistant | v2.0 | 38477380 | none | record-only |
| `v2-consensus-zero-boxes-38506488` | 2026-04-06 | Multi-model consensus produced zero boxes | code | assistant | v2.2 | 38506488 | none | record-only |
| `v2-hybrid-lora-walltime-38917938` | 2026-04-13 | Overnight hybrid-LoRA run hit its 8 h walltime mid-round | config | none | v2.6 | 38917938 | walltime_bound | record-only |

### B. v3.0 autonomous-harvest era (2026-04-16 to 2026-05-22)

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `v3-0-1-still-yolo11x-39393048` | 2026-04-16 | Second attempt still trained yolo11x on the same 5,648 images | code | human | v3.0.2 | 39393048 | pool_growth | record-only |
| `v3-0-2-datasets-shadowed-39397819` | 2026-04-16 | A local datasets.py shadowed the HuggingFace package; every download died at import | code | assistant | v3.0.3 | 39397819 | walltime_bound, pool_growth | record-only |
| `v3-0-4-rgba-jpeg-crash-39591434` | 2026-04-16 | The one candidate that passed the harvest filter crashed on save | code | assistant | v3.0.5 | 39591434 | pool_growth | record-only |
| `v3-0-keyword-table-miss-39363972` | 2026-04-16 | v3.0 shipped and no v3.0 feature fired | code | assistant | v3.0.1 | 39363972 | none | record-only |
| `v3-0-5-bestpt-path-miss-39592795` | 2026-04-17 | 6 h 21 m of successful training reported as FileNotFoundError | code | assistant | v3.0.6 | 39592795 | stale_artifact | raw |
| `v3-0-5-harvest-repeat-3x-39592795` | 2026-04-17 | Brain called an exhausted harvest three times and was force-progressed into the legacy path | code | assistant | v3.0.6 | 39592795 | pool_growth | record-only |
| `v3-0-5-yolo26x-never-loaded-39592795` | 2026-04-17 | The v3.0 model was never loaded in any v3.0 run | code | assistant | v3.0.6 | 39592795 | none | record-only |
| `v3-0-6-scale-denominator-never-audited` | 2026-04-18 | Six releases of bug fixes, none of which audited the corpus size against the stated goal | plan | human | v3.0.7 | 39682959 | pool_growth, plateau | record-only |
| `v3-0-7-kaggle-403-no-creds-39760438` | 2026-04-18 | Every Kaggle download returned 403 because the job had no credentials | config | human | v3.0.9 | 39760438 | source_degraded | record-only |
| `v3-0-7-mega-gate-blocked-39760438` | 2026-04-18 | A strict data gate blocked training three times and produced a 0.90 -> 0.51 regression | code | assistant | v3.0.8 | 39760438 | gate_noop, plateau | record-only |
| `v3-0-7-roboflow-search-401-39760438` | 2026-04-18 | Roboflow search returned 401 nineteen times and five of six project slugs 404'd | config | assistant | v3.0.8/v3.0.9 | 39760438 | source_degraded | record-only |
| `v3-0-9-curated-lists-autonomy-violation` | 2026-04-18 | Hardcoded dataset lists violated the autonomy requirement | plan | human | v3.0.9 | - | none | record-only |
| `v3-0-9-plantvillage-zero-bbox-39928698` | 2026-04-18 | Autonomous Kaggle search added 379,959 images with zero bounding boxes | code | assistant | v3.0.10 | 39928698 | pool_growth | record-only |
| `v3-0-11-brain-skipped-autolabel-39933687` | 2026-04-19 | Brain skipped the labelling step while 380K unlabelled images sat on disk | code | assistant | v3.0.12 | 39933687 | gate_noop | record-only |
| `v3-0-12-owlv2-one-image-per-second-40035529` | 2026-04-19 | Auto-labelling ran at ~1 image/second and consumed the whole walltime | code | assistant | v3.0.13 | 40035529 | walltime_bound | record-only |
| `v3-0-13-owlv2-oom-garbage-fallback-40068162` | 2026-04-20 | Every batch OOMed and the defensive except wrote whole-image boxes | code | assistant | v3.0.14 | 40068162 | gate_noop | record-only |
| `v3-0-14-autolabel-eats-walltime-40069494` | 2026-04-20 | Auto-labelling worked and still left no walltime for training | config | assistant | v3.0.15 | 40069494 | walltime_bound | record-only |
| `v3-0-15-crossdataset-duplicates-40113954` | 2026-04-20 | Four Kaggle mirrors of the same source dataset were about to be trained as distinct data | code | human | v3.0.16 | 40113954 | pool_growth | record-only |
| `v3-0-16-guardrail-roundcap-deadlock-40114079` | 2026-04-20 | Guardrail and round cap formed a loop that consumed a whole job | code | assistant | v3.0.17 | 40114079 | walltime_bound, gate_noop | record-only |
| `v3-0-17-epoch-budget-100h-40124683` | 2026-04-21 | Training started with a 100-hour epoch budget inside an 8-hour job | config | assistant | v3.0.18 | 40124683 | walltime_bound, pool_growth | raw |
| `v3-0-18-walltime-epoch2-40135781` | 2026-04-22 | First end-to-end round ended at epoch 2 of 5 with one validation pass | code | assistant | v3.0.18/v3.0.19 | 40135781 | walltime_bound | raw |
| `v3-0-19-guardrail-bypassed-cap-40144842` | 2026-04-22 | The guardrail's inline loop ignored the per-round cap and spent 6 h 12 m on one dataset | code | assistant | v3.0.20 | 40144842 | walltime_bound | record-only |
| `v3-0-20-chain-sigkill-40162939` | 2026-04-22 | The self-chain died because the walltime SIGKILL preceded the shell's submit line | code | assistant | v3.0.21 | 40162939 | walltime_bound, job_unknown | record-only |
| `v3-0-22-chain-produced-no-weights` | 2026-04-23 | Four chained rounds produced zero checkpoints and the chain believed it was progressing | config | human | v3.0.22/v3.0.23 | 40177598, 40224485, 40239932 | walltime_bound, stale_artifact | record-only |
| `v3-0-22-conda-activate-silent-40224485` | 2026-04-23 | A chain slot was consumed by a job that exited in 20 seconds with 'python: command not found' | code | assistant | v3.0.23 | 40224485 | none | record-only |
| `v3-0-23-fabricated-map-claim` | 2026-04-23 | A summary reported mAP50-95 = 0.344 that no artifact on disk supported | plan | assistant | v3.0.23 | - | stale_artifact | record-only |
| `v3-0-24-classid-zero-contamination` | 2026-04-25 | All 175,701 auto-labelled images were trained as a single class | code | assistant | v3.0.24 | 40260768, 40293571 | gate_noop | raw |
| `v3-0-24-class-passthrough-shared-path-40327811` | 2026-04-27 | Two registry slugs shared one directory and silently permuted four species | code | assistant | v3.0.25 | 40327811 | none | record-only |
| `v3-0-24-eval-label-path-bug-40325013` | 2026-04-27 | The first clean-eval job failed on a label-path assumption | code | assistant | v3.0.24 | 40325013, 40327811 | none | record-only |
| `v3-0-25-p1-aux-capacity-plateau-40329128` | 2026-04-29 | Training plateaued after 14 epochs because 77K auxiliary images ate the capacity | design | assistant | v3.0.25 | 40329128, 40357694 | plateau | raw |
| `v3-0-27-cwd12-train-blocked-by-overreading` | 2026-05-04 | An over-conservative reading of the holdout rule blocked the in-domain training split entirely | design | human | v3.0.27 | 40594919 | none | record-only |
| `v3-0-28-holdout-leak-retraction-0910` | 2026-05-05 | RETRACTION: 0.910 was memorisation — 2,313 holdout copies were inside the training set | code | human | v3.0.28 | 40612856, 40612870 | none | record-only |
| `v3-0-28-safety-walltime-40612856` | 2026-05-06 | The first clean baseline timed out at 62 of 200 epochs | config | assistant | v3.0.28/v3.0.29.1 | 40612856, 40655361 | walltime_bound | raw |
| `v3-0-29-greenpixel-threshold-40624773` | 2026-05-07 | A published curation filter kept 8.4% of the in-domain training data | config | assistant | v3.0.29.1B | 40624773 | gate_noop | record-only |
| `v3-0-29-ultralytics-augment-noop-40655183` | 2026-05-07 | A TTA flag was silently ignored and two runs reported identical numbers | design | assistant | v3.0.29 | 40655183 | gate_noop | record-only |
| `v3-0-29-wbf-multiscale-hurts-40624610` | 2026-05-07 | Multi-scale WBF scored 0.744 against 0.8953 and was nearly read as the true number | design | assistant | v3.0.29/v3.0.30.8 | 40624610 | none | record-only |
| `v3-0-29-rfdetr-missing-dep-40755677` | 2026-05-11 | RF-DETR attempt 1 died on a missing dependency | config | assistant | v3.0.29 | 40755677 | none | record-only |
| `v3-0-29-rfdetr-posembed-768-40797172` | 2026-05-12 | RF-DETR attempt 3 failed on a position-embedding size mismatch | config | assistant | v3.0.29 | 40797172, 40803244 | none | record-only |
| `v3-0-29-rfdetr-resolution-728-40770150` | 2026-05-12 | RF-DETR attempt 2 rejected a resolution that was not a multiple of 32 | config | assistant | v3.0.29 | 40770150 | none | record-only |
| `v3-0-30-3-garbage-datasets-55pct-of-corpus` | 2026-05-12 | The operator opened the dashboard and found 55% of the corpus was plant disease | config | human | v3.0.30.1/v3.0.30.3 | - | pool_growth | raw |
| `v3-0-30-5-jobd-harvest-silently-dead-40800343` | 2026-05-13 | The continuous harvester was dead for about three days and reported nothing but a warning | code | assistant | v3.0.30.5 | 40800343, 40803346 | pool_growth, source_degraded | record-only |
| `v3-0-30-6-hf-direction-kwarg-40803346` | 2026-05-13 | A library upgrade removed a keyword and all 35 keyword searches failed silently | code | assistant | v3.0.30.6 | 40803346 | source_degraded | record-only |
| `v3-0-30-6-supervision-detections-metadata-40803244` | 2026-05-13 | A finished 60-epoch run produced zero predictions because two libraries disagreed | code | assistant | v3.0.30.6 | 40803244, 40825950 | stale_artifact | raw |
| `v3-0-31-1-rfdetr-large-resolution-40832757` | 2026-05-14 | RF-DETR Large crashed at startup on the Medium model's resolution | config | assistant | v3.0.31.1 | 40832757, 40839152 | none | record-only |
| `v3-0-31-1-training-cwd12-only-for-six-days` | 2026-05-14 | The collector accumulated 1.5M images while the trainer used only 3,671 of them | plan | human | v3.0.31.1/v3.0.32 | 40839159 | pool_growth | record-only |
| `v3-0-31-selfchain-missed-at-walltime` | 2026-05-14 | Two long-running services hit 48 h and their self-chain hooks never fired | config | assistant | v3.0.31.1 | 40770062, 40803346, 40839165, 40877440 | walltime_bound, job_unknown | record-only |
| `v3-0-32-cumulative-walltime-3x-40839159` | 2026-05-15 | A cumulative run needed about 104 hours inside a 36-hour allocation | config | assistant | v3.0.32 | 40839159, 40877439 | walltime_bound, pool_growth | raw |
| `v3-0-35-gemma4-bbox-f1-0012-40884184` | 2026-05-16 | The Brain model asked for bounding boxes produced 6,284 false positives | design | assistant | v3.0.35 | 40884184 | none | record-only |
| `v3-0-35-gemma4-verifier-rubber-stamp-40884184` | 2026-05-16 | The crop verifier answered yes to every crop: 0.0% false-positive rejection | design | assistant | v3.0.35 | 40884184 | gate_noop | record-only |
| `v3-0-35-t3-no-negatives-40884184` | 2026-05-16 | A relevance test reported 94% accuracy having tested only positives | design | assistant | v3.0.35.2 | 40884184, 40891361 | none | record-only |
| `v3-0-35-uncategorized-slugs-noise` | 2026-05-16 | 35 registry slugs passed the vocabulary filter because their names said nothing | design | assistant | v3.0.35/v3.0.36 | 40891360 | pool_growth | raw |
| `v3-0-37-cumulative-clean-timeout-40896313` | 2026-05-19 | A mis-designed experiment timed out at epoch 19 with a declining metric | design | assistant | v3.0.37/v3.0.38 | 40896313 | walltime_bound, plateau | record-only |
| `v3-0-38-a-premature-ceiling-claim` | 2026-05-22 | An 'architecture ceiling' was declared and then crossed by a different seed | design | assistant | v3.0.38-A | 40912927, 40878511, 40839152 | plateau | raw |
| `v3-0-38-b-c-flux-results-never-recorded` | 2026-05-22 | Two submitted jobs have no recorded result anywhere | plan | none | v3.0.38-B/C, v3.0.39 | 40962044, 40963121 | job_unknown | raw |

### C. Platform era: audits and data governance (2026-05-28 to 2026-07-10)

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `v3-0-43-classes-prewarm-lustre-storm` | 2026-05-28 | A browse page's prewarm projected six hours of Lustre metadata calls | code | assistant | v3.0.43.22 | - | none | record-only |
| `v3-0-99-github-unreachable-from-compute` | 2026-06-08 | GitHub is unreachable from compute nodes, so one harvest source is permanently dead | config | assistant | v3.0.99 | - | source_degraded | raw |
| `v3-0-99-owl-imagecond-precision-002` | 2026-06-08 | Image-conditioned auto-labelling measured at ~0.02 precision and was gated out | design | assistant | v3.0.99 | - | gate_noop | record-only |
| `v3-0-99-roboflow-15m-file-export` | 2026-06-08 | A 10K-image export was a 1.5-million-file dump that extracted for eight hours and blocked the batch | code | assistant | v3.0.99.10-.14 | - | walltime_bound | record-only |
| `v3-0-99-tomato-disease-leaked-via-crop-detection` | 2026-06-08 | A tomato-leaf-disease set entered the pool through a 'crop detection' query | code | assistant | v3.0.99 | - | pool_growth | raw |
| `v3-1-0-a1-bg-job-marked-done-on-failure` | 2026-07-04 | A background job was marked done whenever its function returned, including when it returned failure | code | assistant | v3.1.0 | - | job_unknown | raw |
| `v3-1-0-a2-auth-fail-open` | 2026-07-04 | The dashboard authenticated everyone when its password file was missing, while exposed on a public tunnel | code | assistant | v3.1.0 | - | none | record-only |
| `v3-1-0-a3-simulated-events-unflagged` | 2026-07-04 | Simulated labelling events were written into the log the dashboard reported as real human verification | code | assistant | v3.1.0 | - | none | raw |
| `v3-1-0-a4-mongo-mirror-failure-swallowed` | 2026-07-04 | A failed database mirror was discarded, so JSON and Mongo could diverge unobserved | code | assistant | v3.1.0 | - | mongo_down | record-only |
| `v3-1-0-autolabel-conf-floor-override` | 2026-07-04 | A caller hardcoded a confidence floor that silently undid a safety change | code | assistant | v3.1.0 | - | gate_noop | record-only |
| `v3-1-0-c1-holdout-dhash-not-seeded` | 2026-07-04 | The holdout guard was filename-only, so a re-exported copy of a test image could train | code | assistant | v3.1.0 | - | none | raw |
| `v3-1-0-c2-registry-lost-update` | 2026-07-04 | The shared registry had no lock, six writers shared one temp filename, and a corrupt parse rebuilt it empty | code | assistant | v3.1.0 | - | ownership_violation | raw |
| `v3-1-0-c3-committed-kaggle-token` | 2026-07-04 | A live API token was the hardcoded default in three job scripts and is in public git history | config | human | v3.1.0 | - | none | record-only |
| `v3-0-187-small-model-number-mangling` | 2026-07-10 | The small analysis model restated computed numbers wrongly and invented issues | code | assistant | v3.0.187/v3.5.0 | - | none | raw |

### D. SuperWeed campaign (2026-08-15 to 2026-08-25)

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `v3-22-5-dino-gate-silent-noop` | 2026-08-22 | The curated tier would have trained on the raw pool and been published as curated | code | assistant | v3.22.5 | 44224997, 44225271, 44228844 | gate_noop | raw |
| `v3-22-10-offgoal-maritime-underwater-admitted` | 2026-08-23 | A sea-rescue drone dataset and an underwater fish dataset entered the weed pool | code | assistant | v3.22.10/v3.22.12 | 44236325 | pool_growth | raw |
| `v3-22-10-registry-lostupdate-quarantine-erased` | 2026-08-23 | Registry incident #1: a harvest's 29-minute-old snapshot erased a supervisory quarantine | code | assistant | v3.22.10 | 44236325 | ownership_violation | raw |
| `v3-22-12-license-resolver-case-lossy` | 2026-08-23 | Three weed datasets reported no licence because the resolver rebuilt their ids from a lowercased slug | code | assistant | v3.22.12 | - | none | raw |
| `v3-22-12-mamba-build-four-layers` | 2026-08-23 | A CUDA extension took four failed builds, each hiding the next cause | config | assistant | v3.22.12/v3.22.13 | 44275211, 44280678 | none | raw |
| `v3-22-13-exdark-gwhd-zero-label-admitted` | 2026-08-23 | Two more off-goal datasets passed the hardened subject check, and the signal that would have stopped them was already being computed and thrown away | config | assistant | v3.22.13 | 44275184 | pool_growth | raw |
| `v3-22-14-inprocess-state-orphaned-job` | 2026-08-23 | A dashboard restart orphaned a live 2 h 40 m training job and freed the scheduler to submit it again | code | assistant | v3.22.14 | 44278259 | job_unknown | record-only |
| `v3-22-14-stale-metric-train2` | 2026-08-23 | The scheduler was about to attach a day-old number to a fresh round | code | assistant | v3.22.14 | 44278259 | stale_artifact | raw |
| `v3-22-15-cluster-etiquette-skipped` | 2026-08-23 | Four jobs were submitted without the login-node verification the plan requires | plan | assistant | v3.22.15 | 44275211, 44280678, 44323304, 44325279 | budget | record-only |
| `v3-22-15-pool-report-local-labeled-15789` | 2026-08-23 | A new measurement tool reported 15,789 labelled images against a real 120,515 | code | assistant | v3.22.15 | - | none | raw |
| `v3-22-16-sync-hung-20-days-first-repair` | 2026-08-23 | The platform had been serving a 2026-08-18 registry: the cluster sync had been hung since 2026-08-03 | config | assistant | v3.22.16 | - | stale_artifact | record-only |
| `v3-22-17-audit-probe-8-of-8-fail` | 2026-08-23 | The first audit sweep failed every source, and two of the three causes were in the instrument | code | assistant | v3.22.17 | 44294276, 44300150 | gate_noop | raw |
| `v3-22-17-cotton-weed-subpixel-boxes` | 2026-08-23 | A harvested source's every sampled box was sub-pixel | config | assistant | v3.22.17 | 44294276 | none | raw |
| `v3-22-7-raw-tier-75h-on-v100` | 2026-08-23 | Walltime arithmetic caught mid-flight: the raw tier needed ~75 h in a 12 h job | config | assistant | v3.22.7 | 44224995, 44225260, 44234060, 44234063 | walltime_bound | raw |
| `v3-22-9-cluster-arm-askpass-dead` | 2026-08-23 | The platform's cluster control path was dead because an askpass file held no current password | config | assistant | v3.22.9 | - | none | record-only |
| `v3-22-20-mamba-fork-trainer-unusable` | 2026-08-24 | A third-party trainer died in 113 seconds for three independent reasons | code | assistant | v3.22.20 | 44323304, 44325279 | none | raw |
| `v3-22-21-collect-walltime-4h-timeout` | 2026-08-24 | Round 4's harvest genuinely timed out at four hours | config | assistant | v3.22.21 | - | walltime_bound, pool_growth | raw |
| `v3-22-21-job-unknown-booked-as-failed` | 2026-08-24 | A COMPLETED harvest was recorded as failed and fed the stop-loss | code | assistant | v3.22.21 | 44322382 | job_unknown | raw |
| `v3-22-21-mamba-numpy2-trapz` | 2026-08-24 | The smoke job built the model and then died in metrics on a function NumPy 2 removed | config | assistant | v3.22.21/v3.22.22 | 44325279, 44344817 | none | raw |
| `v3-22-23-mamba-init-confound` | 2026-08-24 | An architecture comparison was about to be published with the initialisation confounded | design | assistant | v3.22.23/v3.22.24 | 44351282, 44323305, 44368952 | none | raw |
| `v3-23-1-retraction-027-causal-claim` | 2026-08-25 | RETRACTION: the campaign's most-quoted causal claim was disproved by its own control | design | assistant | v3.23.1/v3.23.2 | 44383471, 44397807 | none | raw |
| `v3-23-2-harvest-10h-timeout-three-zips` | 2026-08-25 | A ten-hour harvest downloaded three datasets and timed out | config | assistant | v3.23.2 | 44385850 | walltime_bound, source_degraded | raw |
| `v3-24-2-s1-gate-not-met` | 2026-08-25 | A campaign phase was closed with its gate deliberately not met, and the shortfall is the result | plan | assistant | v3.24.2 | - | plateau, budget | raw |
| `v3-24-2-scorecard-rollback-registry-incident-2` | 2026-08-25 | Registry incident #2: the corrected audit verdicts were rolled back by a harvester's stale snapshot | code | assistant | v3.24.2 | - | ownership_violation | raw |
| `v3-24-3-sync-dead-22-days-wrong-diagnosis` | 2026-08-25 | The sync had been dead 22 days, the earlier repair fixed nothing, and the recorded diagnosis was wrong | config | assistant | v3.24.3 | - | stale_artifact | record-only |
| `v3-24-4-hflip-mirrored-coords-not-image` | 2026-08-25 | A test-time augmentation mirrored the coordinates of predictions made on an unflipped image | code | assistant | v3.24.4 | - | none | record-only |
| `v3-24-4-matcher-offset-would-have-masqueraded-as-tta` | 2026-08-25 | A metric offset of 0.024 would have been reported as a TTA effect | design | assistant | v3.24.4/v3.24.8 | 44463762, 44463922 | none | raw |
| `v3-24-4-wbf-tta-class-order-permuted` | 2026-08-25 | Every per-species AP was attributed to the wrong weed and overall mAP looked fine | code | assistant | v3.24.4 | 44463762, 44463922 | none | raw |
| `v3-24-6-fix-local-paths-two-populations` | 2026-08-25 | A progress line compared a numerator and a denominator drawn from different populations | code | assistant | v3.24.6 | - | none | record-only |
| `v3-24-7-imageweeds-license-none` | 2026-08-25 | The one audit-passing source carried no licence although its card states CC BY 4.0 | config | assistant | v3.24.7 | - | none | raw |

### E. After the split cutoff: the walltime pair and the WP1 findings (2026-08-28 onward)

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `sched-silent-pause-six-days-20260829` | 2026-08-29 | The domain paused itself and nobody was told for six days | code | assistant | v3.25.0 | - | none | record-only |
| `sched-walltime-double-timeout-20260829` | 2026-08-29 | DEV CASE: two identical 12 h training jobs died at the wall and the loop paused itself | config | assistant | v3.25.0 | 44727703, 44767709 | walltime_bound, pool_growth | raw |
| `wp1-qwen38-deploy-404-old-ollama-45197085` | 2026-09-04 | A model deploy failed with 404 because the cluster binary was two years behind the tag | config | assistant | WP1-era | 45197085, 45198752 | source_degraded | raw |
| `v3-25-1-step-command-allowlist-refused-on-lab` | 2026-09-05 | A safety check would have stopped the loop silently on the machine that runs it | code | assistant | v3.25.1 | - | gate_noop | record-only |
| `v3-25-2-cpu-train-on-gpu-allocation-45250479` | 2026-09-05 | A GPU job trained on the CPU on a driver-mismatched node and produced a real-looking metric | code | assistant | v3.25.2 | 45250479 | walltime_bound | raw |
| `wp1-array-jobid-key-would-detach-metric` | 2026-09-05 | Keying artifacts on SLURM_JOB_ID would have silently disabled both new readers for every array task after the first | code | assistant | v3.25.0 | 45227822, 45227848 | stale_artifact | raw |
| `wp1-job-scoped-artifact-would-deflate-a-published-std` | 2026-09-05 | A new artifact would have been counted as a second seed and deflated a published standard deviation | code | assistant | v3.25.0 | - | none | record-only |
| `wp1-vllm-fp8-kv-cache-abort` | 2026-09-05 | A serving job aborted every tensor-parallel worker at model build, twice, before the cause was named | config | assistant | v3.25.2 | 45225918 | none | raw |
| `v3-25-3-probe-repointed-campaign-state-45290759` | 2026-09-06 | A one-epoch probe repointed the registry's progressive-training checkpoint at itself | code | assistant | v3.25.3 | 45290759 | ownership_violation, stale_artifact | raw |
| `v3-25-3-vllm-verify-kv-dtype-misreported` | 2026-09-06 | A verification job recorded a configuration it did not run with | code | assistant | v3.25.3 | 45225918 | none | raw |

### F. Healthy controls

Completed steps of scheduler rounds 1-3, the six sealed M1 runs and the corrected S3 tier ladder are
the plan's named set; the remaining rows widen the denominator for the false-alarm rate. Each must be
re-audited with the WP3 signals before it is labelled clean - `hc-round1-train-44278259` in particular,
because its own `mega_iter` directory holds the stale artifact that the scheduler nearly published.

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `hc-s2-dino-scores-44225271` | 2026-08-22 | DINOv2 scoring pass that produced the missing gate input | - | none | v3.22.6 | 44225271 | none | - |
| `hc-m1-curated-s101-44234063` | 2026-08-23 | M1 sealed run: curated tier, seed 101 | - | none | v3.22.11 | 44234063 | none | - |
| `hc-m1-curated-s102-44234063` | 2026-08-23 | M1 sealed run: curated tier, seed 102 | - | none | v3.22.11 | 44234063_2 | none | - |
| `hc-m1-curated-s103-44234063` | 2026-08-23 | M1 sealed run: curated tier, seed 103 | - | none | v3.22.11 | 44234063_3 | none | - |
| `hc-m1-raw-s101-44234060` | 2026-08-23 | M1 sealed run: raw tier, seed 101 | - | none | v3.22.11 | 44234060 | none | - |
| `hc-m1-raw-s102-44234060` | 2026-08-23 | M1 sealed run: raw tier, seed 102 | - | none | v3.22.11 | 44234060_2 | none | - |
| `hc-m1-raw-s103-44234060` | 2026-08-23 | M1 sealed run: raw tier, seed 103 | - | none | v3.22.11 | 44234060_3 | none | - |
| `hc-round1-collect-44275184` | 2026-08-23 | Round 1 collect completed | - | none | v3.22.12 | 44275184 | none | - |
| `hc-round1-filter-44276626` | 2026-08-23 | Round 1 filter completed | - | none | v3.22.12 | 44276626 | none | - |
| `hc-round1-train-44278259` | 2026-08-23 | Round 1 train completed in 5 h 37 m at 0.6019 | - | none | v3.22.18 | 44278259 | none | - |
| `hc-round2-collect-44295046` | 2026-08-23 | Round 2 collect completed under the strict gate | - | none | v3.22.18 | 44295046 | none | - |
| `hc-round2-filter-44303585` | 2026-08-23 | Round 2 filter completed | - | none | v3.22.18 | 44303585 | none | - |
| `hc-round2-train-44304828` | 2026-08-24 | Round 2 train completed in 6 h 34 m at 0.5919 | - | none | v3.22.18 | 44304828 | none | - |
| `hc-round3-collect-44322382` | 2026-08-24 | Round 3 collect completed in 3 h 10 m | - | none | v3.22.21 | 44322382 | none | - |
| `hc-round3-train-unknown` | 2026-08-24 | Round 3 train completed at 0.5951 | - | none | v3.23.3 | - | none | - |
| `hc-s3-mamba-s101-44351282` | 2026-08-24 | S3 family run: Mamba-YOLO-T from scratch, seed 101 (0.8331) | - | none | v3.22.23 | 44351282_1 | none | - |
| `hc-s3-mamba-s102-44351282` | 2026-08-24 | S3 family run: Mamba-YOLO-T from scratch, seed 102 (0.8263) | - | none | v3.22.23 | 44351282_2 | none | - |
| `hc-s3-mamba-s103-44351282` | 2026-08-24 | S3 family run: Mamba-YOLO-T from scratch, seed 103 (0.8203) | - | none | v3.22.23 | 44351282_3 | none | - |
| `hc-s3-yolo11n-s101-44323305` | 2026-08-24 | S3 family run: YOLO11n pretrained, seed 101 (0.8739) | - | none | v3.22.21 | 44323305_1 | none | - |
| `hc-s3-yolo11n-s102-44323305` | 2026-08-24 | S3 family run: YOLO11n pretrained, seed 102 (0.8789) | - | none | v3.22.21 | 44323305_2 | none | - |
| `hc-s3-yolo11n-s103-44323305` | 2026-08-24 | S3 family run: YOLO11n pretrained, seed 103 (0.8737) | - | none | v3.22.21 | 44323305_3 | none | - |
| `hc-s3-yolo11n-scratch-44368952` | 2026-08-24 | S3 fairness control: YOLO11n from scratch, 3 seeds (0.8041 +/- 0.0028) | - | none | v3.22.24 | 44368952_1, 44368952_2, 44368952_3 | none | - |
| `hc-ladder-core0-44397807` | 2026-08-25 | Corrected tier ladder rung: core + 0 harvested (0.8636, 97 ep) | - | none | v3.23.2 | 44397807_0 | none | - |
| `hc-ladder-core15k-44397807` | 2026-08-25 | Corrected tier ladder rung: core + 15,000 (0.8614, 100 ep) | - | none | v3.23.2 | 44397807_2 | none | - |
| `hc-ladder-core40k-44397807` | 2026-08-25 | Corrected tier ladder rung: core + 40,000 (0.8436, 90 ep) | - | none | v3.23.2 | 44397807_3 | none | - |
| `hc-ladder-core5k-44397807` | 2026-08-25 | Corrected tier ladder rung: core + 5,000 (0.8599, 91 ep) | - | none | v3.23.2 | 44397807_1 | none | - |
| `hc-s3-bestmodel-reeval-44454237` | 2026-08-25 | Independent re-evaluation of the deployable checkpoint (0.8759 +/- 0.0030) | - | none | v3.23.4 | 44454237 | none | - |
| `hc-s3-tta-arms-44463762` | 2026-08-25 | TTA ceiling arms 1-5 under one matcher | - | none | v3.24.8 | 44463762 | none | - |
| `hc-s3-tta-ensemble-44463922` | 2026-08-25 | TTA ceiling arm 6 plus the Ultralytics anchor | - | none | v3.24.8 | 44463922 | none | - |
| `hc-s6-crossdataset-44465026` | 2026-08-25 | Cross-dataset transfer evaluation with its leak check | - | none | v3.24.7 | 44465026 | none | - |
| `hc-wp1-loop-restart-45326516` | 2026-09-06 | First round after the loop was re-enabled | - | none | v3.25.3 | 45326516 | none | - |

### G. False-alarm and environment controls

Cases where the correct verdict is *no incident*. These are not part of recall; they are the cheapest
honest test of whether an arm escalates on artifacts that merely look alarming.

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `ctrl-cottonweed-holdout-in-merge-accounting` | 2026-08-23 | CONTROL: a scary-looking merge line that is benign | - | none | v3.22.11 | 44234060, 44234063 | none | raw |
| `ctrl-mambayolo-declared-dependency-conflict` | 2026-08-24 | CONTROL: a declared dependency conflict that is not a real one | - | none | v3.22.22 | 44344817 | none | raw |
| `ctrl-crossdataset-explanations-killed` | 2026-08-26 | CONTROL: two plausible artifact explanations tested and rejected before a surprising number was accepted | - | none | v3.24.7/v3.24.8 | 44465026 | none | raw |
| `ctrl-socks-github-skip-chronic` | 2026-08-29 | CONTROL: the chronic degraded-source line that must stay at info | - | none | v3.0.99 onward | 44727703 | none | raw |

### H. Inventoried but out of scope

The record names these as concrete failures and they are listed for completeness, but they have no
`/ocean` artifact and no supervision-loop bundle, so they belong in neither split.

| case_id | date | title | class | actor | version | job ids | signals expected | provenance |
|---|---|---|---|---|---|---|---|---|
| `v3-8-3-expired-session-self-lockout` | 2026-07-22 | A returning browser locked its own IP out for an hour by loading the page | code | assistant | v3.8.3 | - | none | record-only |
| `v3-9-2-dataset-delete-any-owner` | 2026-07-23 | Any signed-in member could delete another member's dataset, and it was a class of defect | code | human | v3.9.2/v3.9.3 | - | none | record-only |
| `v3-16-1-share-capture-claimed-not-captured` | 2026-08-03 | A 77-character sign-in shell was displayed as a saved conversation snapshot | code | assistant | v3.16.1 | - | none | raw |
| `v3-17-1-microphone-dropped-from-recordings` | 2026-08-03 | Screen recordings silently contained no voice | code | human | v3.17.1 | - | none | record-only |
| `v3-18-0-recording-share-url-was-the-capability` | 2026-08-03 | Any signed-in member holding a recording id could watch and download it | code | assistant | v3.18.0 | - | none | raw |
| `v3-19-1-trimming-discarded-a-real-capture` | 2026-08-03 | A guard against empty pages silently became a minimum conversation length | code | assistant | v3.19.1/v3.19.2 | - | none | raw |
| `v3-20-0-csv-float-format-collapsed-timestamps` | 2026-08-15 | Epoch timestamps were written with %.6g and the shared-clock span read 0.0 seconds | code | assistant | v3.20.0 | - | none | raw |
| `v3-22-0-proxy-read-blocked-until-8kb` | 2026-08-16 | A streaming relay buffered until 8 KB and made a live camera unusable | code | assistant | v3.22.0 | - | none | record-only |
| `v3-24-10-frame-timestamp-collision-overwrite` | 2026-08-28 | Six frames were silently eaten by filename collisions | code | human | v3.24.10 | - | none | raw |
| `v3-24-11-camera-toggle-shipped-invisible` | 2026-08-28 | A shipped feature nobody could find is a feature that has not shipped | design | human | v3.24.11 | - | none | record-only |
| `v3-24-9-gallery-frames-whitelist-404` | 2026-08-28 | Thumbnails rendered and the full-size click 404'd on a file that demonstrably exists | code | human | v3.24.9 | - | none | raw |

## 6. Open items the integrator must settle on the cluster

- `VERIFY` the round-4 collect job id (the 4 h harvest TIMEOUT) and the round-3 filter/train job ids:
  `sacct -S 2026-08-23 -E 2026-08-26 -u byler --format=JobID,JobName%20,State,Elapsed,Start,End`.
- `VERIFY` survival of the dev-case artifacts before anything else:
  `ls -la results/framework/*44727703* results/framework/*44767709*` and
  `sacct -j 44727703,44767709 --format=JobID,State,Elapsed,Timelimit,AllocTRES -P`.
- `VERIFY` whether the two unread May jobs left output:
  `ls -la results/framework/*40962044* results/framework/*40963121*`. If they did, a recorded gap
  becomes both an extra case and an answer to an open scientific question.
- `VERIFY` the launcher script names behind the audit sweeps (44294276, 44300150) and the sanity
  check 40655183: `ls -la results/framework/*44294276* results/framework/*40655183*`.
- Scrub before export: the corpus must not carry the cluster username, the cluster password, or any
  `hf_`/`KGAT` token prefix. One case (`v3-1-0-c3-committed-kaggle-token`) is *about* a leaked token;
  its bundle must reference the defect without reproducing the value.
- Robot frames are private. No case here needs them, and no robot session imagery may enter a bundle.
