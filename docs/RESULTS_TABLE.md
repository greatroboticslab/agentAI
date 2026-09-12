# Results table — every number with its provenance

*Draft of 2026-09-10. Built by re-reading the artifacts, not by copying the earlier tables.
Four separate errors in the previously published tables are corrected here and listed in §D.*

**Status:** the two control arms (`45672628_[1-2]`) and the two seed repeats
(`45672672_[1-2]`) have completed; their measured values are in §5g of
[`REGRESSION_DIAGNOSIS.md`](REGRESSION_DIAGNOSIS.md) and in the readable one-page version.

**One-page web version:** `docs/results_page/index.html`, published at
<https://claude.ai/code/artifact/f13cf654-af80-4101-a615-c9eb24c5df11>. It carries the
defensible subset — the in-domain baselines, the campaign series at three readouts, the
control arms, the audit findings, the never-quote-bare list and the gaps — with the evaluator
band stated above every table. This file stays the exhaustive record; that page is what goes
in front of a reader.

---

> **Path shorthands used in every Artifact cell**
> `DIAG/` = `/private/tmp/claude-501/-Users-xiaogui-Desktop-2026spring-research-weed-llm-benchmark/1ae39708-9141-44c7-bd5c-13a7f37399e7/scratchpad/diag/`
> `REPO/` = `/Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/`
>
> **Three different evaluators are in play and they do not share a scale.** The *same* YOLO11n s102 checkpoint reads **0.8794 under Ultralytics `.val()`** and **0.8554 under the project's custom WBF/TTA matcher** — an offset of **0.0239** (`DIAG/lab/s3_tta_ceiling/ultralytics_val_s102.json` vs `plain_s102.json`). The same v3.0.28 yolo26x checkpoint was recorded as **0.896 Ultralytics / 0.7446 pyco-WBF** (`REPO/weed_llm_benchmark/CHANGELOG.md` L3112, L2538-2560; discrepancy investigated 2026-05-07, never resolved). **Never compare a number in one block against a number in another.**
> **Two different holdouts are in play**: the sealed **1,977-image / 3,257-instance** cwd12 test+valid holdout (Blocks A-F, H) and the **848-image / 1,464-instance** cwd12 test split (Block G).
> **Two different heads are in play**: `nc=12` (Blocks A, B, F, G) and `nc=100` (12 real + 88 `aux_*`, Blocks C, D, E). The head cost is **measured, not assumed: -0.012** (0.8636 nc=100 vs 0.8755 nc=12 on identical data, Block C1 vs A1).

---

### Block A — sealed cwd12 holdout · nc=12 · **Ultralytics matcher**
Holdout for every row: cwd12 test+valid, **1,977 images / 3,257 instances**, `cwd12_sealed.yaml`, NEVER_TRAIN.

| # | Recipe | Train data | Metric (mAP50-95) | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| A1 | YOLO11n, COCO-pretrained, 100 ep | cwd12 train split, 3,671 imgs | **0.8755 ± 0.0029** (per-seed 0.87392 / 0.87886 / 0.87370) | 3 seeds (101/102/103) | `DIAG/lab/s3_yolo11n_seed101.json`, `…102`, `…103`; per-epoch best re-read from `DIAG/probe4.json` → `s3_run_dirs[s3_yolo11n/s10*/results.csv]` | yes | Reported value is each run's **best epoch** (77/90/72 of 100), not last. The headline reference number of the whole project. |
| A2 | YOLO11n, **random init** (fairness control), 100 ep | cwd12 train split, 3,671 | **0.8041 ± 0.0028** (per-seed 0.80551 / 0.80089 / 0.80587) | 3 seeds | `DIAG/probe4.json` → `s3_run_dirs[s3_yolo11n/scratch_s10*/results.csv]`; `DIAG/lab/figures_data.json` → `s3_families_2026_08_24.yolo11n_scratch_control` | caveat | **The three sidecar files `DIAG/lab/s3_yolo11n_scratch_seed10*.json` are corrupted** — they record 0.8739/0.8789/0.8737, byte-identical to the *pretrained* runs A1. Use `results.csv` (probe4) / `figures_data.json` only. Job ids in the sidecars (44368976/77/52) are correct; only the metric field is wrong. |
| A3 | Mamba-YOLO-T, **random init** (fork ships no weights), 100 ep | cwd12 train split, 3,671 | **0.8266 ± 0.0064** (per-seed 0.8331 / 0.8263 / 0.8203) | 3 seeds | `DIAG/lab/s3_mamba_t_seed101.json`, `…102`, `…103` | caveat | Values exist **only** in the sidecar JSONs; `DIAG/probe4.json` lists the `s3_mamba/t_s10*` run dirs but carries no parsed `results.csv` best, so unlike A1/A2 this row has no second source. A 4th run (`s3_mamba_t_cwd12.json`) is `ok:false, "no results.csv produced"`. |
| A4 | *Derived*: value of COCO pretraining for YOLO11n | — | **+0.0714** (A1 − A2) | from 3+3 seeds | A1, A2 | yes | Difference of two means whose stds are 0.0029/0.0028 — comfortably resolved. |
| A5 | *Derived*: architecture at **equal (random) init**, Mamba-T over YOLO11n | — | **+0.0225** (A3 − A2) | from 3+3 seeds | A2, A3 | caveat | Only defensible architecture claim in the set. Params differ (6.13M vs 2.6M), so it is not a compute-matched comparison. Inherits A3's single-source weakness. |
| A6 | Best single deployable checkpoint (s102), re-validated | cwd12 train split, 3,671 | **0.87935** mAP50-95 / **0.93692** mAP50 — **single run** | single run | `DIAG/lab/s3_tta_ceiling/ultralytics_val_s102.json` | yes | Standalone `.val()` of the A1 seed-102 weights; reads +0.0005 above that run's own training-curve best (0.87886). Single checkpoint, no error bar — the error bar is A1's. |
| A7 | Same 3 checkpoints, re-evaluated in eval job 44454237 | cwd12 train split, 3,671 | **0.8759 ± 0.0030**; mAP50 per seed 0.9330 / 0.9369 / 0.9328 | 3 checkpoints (**not** 3 new runs) | `DIAG/lab/figures_data.json` → `best_model_card_2026_08_25.deployable` | caveat | Same three training runs as A1 re-scored. **Must not be presented as independent replication of A1.** |
| A8 | Per-species spread of the deployable model | cwd12 train split, 3,671 | Ragweed **0.9767** … Morningglory **0.7324**; **spread 0.2443** (class mean 0.8759) | mean over the 3 A1 checkpoints | `DIAG/lab/figures_data.json` → `best_model_card_2026_08_25.per_species_map50_95` | caveat | Matcher/aggregation for job 44454237 is not recorded in the artifact set, and these per-class values match none of the `s3_tta_ceiling/*.json` tables; the 12 values do average to 0.8759, so they are internally coherent. Spread ≈ 80× seed noise, so the weakness ranking is real. |

---

### Block B — sealed cwd12 holdout · nc=12 · **custom WBF/TTA matcher (different scale from Block A)**
Holdout: same 1,977 images / 3,257 instances (verified: `n_gt` sums to 3,257 in all six artifacts). **B0 is the baseline for this block — compare B1-B5 to B0, never to A1/A6.**

| # | Recipe | Train data | Metric (mAP50-95) | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| B0 | Plain s102 @640, custom matcher (**block baseline**) | cwd12 train split, 3,671 | **0.8554** — single run | single run (deterministic inference) | `DIAG/lab/s3_tta_ceiling/plain_s102.json` | caveat | Same weights as A6, which read 0.8794 under Ultralytics. **The 0.0239 gap is the matcher, not the model.** |
| B1 | WBF fusion only, 1 model @640 | as B0 | **0.8524** (Δ vs B0 **−0.0030**) | single run | `DIAG/lab/s3_tta_ceiling/wbf_s102_640.json` | yes | Fusion alone does nothing; Δ is inside the 0.006 seed-noise bar. |
| B2 | Multi-scale TTA (512/640/768), 1 model | as B0 | **0.8653** (Δ **+0.0099**) | single run | `DIAG/lab/s3_tta_ceiling/tta_scales.json` | yes | Inference-time only. 601 s for 1,977 images. |
| B3 | Multi-scale + hflip, 1 model | as B0 | **0.8728** (Δ **+0.0175**) | single run | `DIAG/lab/s3_tta_ceiling/tta_scales_flip.json` | yes | 2,057 s. |
| B4 | 3-seed ensemble @640 (WBF) | 3× cwd12 train split | **0.8726** (Δ **+0.0172**) | 3 checkpoints, 1 fused run | `DIAG/lab/s3_tta_ceiling/ens3_640.json` | yes | 3-seed ensembling ≈ single-model TTA. Uses the A1 checkpoints. |
| B5 | 3 models × 18 views (scales+flip, WBF) — **the ceiling arm** | 3× cwd12 train split | **0.8830** (Δ **+0.0276**) | 3 checkpoints, 1 fused run | `DIAG/lab/s3_tta_ceiling/ens3_tta.json`, `summary.json` | caveat | 5,891 s ≈ **2.44 s/image, ~660× the deployed 3.7 ms**. A ceiling probe, not a deployable configuration. |
| B6 | "Ceiling on the validator scale if the matcher offset composes" | — | **0.907** | — | `DIAG/lab/figures_data.json` → `tta_ceiling_2026_08_26.ceiling_on_validator_scale_if_offset_composes` | **no** | **Arithmetic extrapolation (0.8830 + 0.0239), never measured.** The offset was measured once, on one checkpoint, with no TTA. Do not put this number on a slide. |

---

### Block C — sealed cwd12 holdout · **nc=100 head** · YOLO11n pretrained · tier ladder, **three seeds at every rung**
Holdout: same 1,977 images / 3,257 instances. This block is the **bridge** between Block A's clean
numbers and the nc=100 pipeline numbers in Blocks D-E. **Supersedes the single-seed ladder reported
2026-08-25** — those were seed 101 of these same runs, not a different experiment.

Two arms, pre-registered before either ran. **Arm A** gives every source dataset its own class; **arm
B** collapses every harvested box into one shared `weed` class. The question arm B was built to answer
is whether the ladder's shape is a class-space artefact. It is not: arm B is worse at every rung.

| # | Recipe | Train data | Metric (mAP50-95) | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| C1 | **Arm A** · clean core + **0** harvested | real cwd12 train split, 3,671 (6,131 inst) | **0.8637 ± 0.0027** | **3 seeds** | `results/framework/s3_tier_ladder*/`; `docs/poster/poster_data.py` → `LADDER.seeds["0"]` | yes | **The measured nc=100 head cost: 0.8637 vs A1's 0.8755 = −0.012.** Quote that offset whenever an nc=100 number sits beside an nc=12 one. |
| C2 | **Arm A** · core **+5,000** | 8,671 | **0.8609 ± 0.0036** (Δ **−0.0028**) | 3 seeds | same | yes | Δ sits inside the pooled seed spread of 0.0032 — not separable. |
| C3 | **Arm A** · core **+15,000** | 18,671 | **0.8580 ± 0.0047** (Δ **−0.0057**) | 3 seeds | same | yes | Δ sits inside the pooled seed spread of 0.0039 — not separable. |
| C4 | **Arm A** · core **+40,000** | 43,671 | **0.8448 ± 0.0018** (Δ **−0.0189**) | 3 seeds | same | yes | **8.2 σ of the seed spread.** The only rung that separates, and it separates downwards. |
| C5 | **Arm B** · one shared class, +0 / +5k / +15k / +40k | as C1-C4 | **0.8636 / 0.8538 / 0.8451 / 0.8252** | single run per rung, seed 101 | `results/framework/s3_tierb_*`; jobs `45744712_[0-3]` | caveat | Single seed per rung, so each point carries the 0.004 bar. The **shape** is what this arm was for. |
| C6 | *Derived*: **arm B vs arm A at each rung** | — | **−0.0071 / −0.0142 / −0.0196** at +5k / +15k / +40k | C1-C5 | — | yes | Arm B is worse at **every** rung and the gap **widens** with volume. The pre-registered class-space hypothesis is falsified on its own condition. Surviving reading: harvested images carry **9.0 boxes/image against the core's 1.7**, so collapsing them into one class concentrates that density instead of spreading it. |
| C7 | *Derived*: honest cost of adding harvested data to a clean core | — | **−0.0189 at 12× the data, 8.2 σ** | 3 seeds/rung (arm A) | C1-C4 | yes | Supersedes and retracts the earlier "0.27 cost" claim (`DOUBLE_AGENT_SYSTEM.md` §4, retraction 2026-08-25) **and** the earlier "flat curve, 12× the data buys nothing" reading: with three seeds the top rung is not flat, it is **down**. |

**Stale doc to fix before presenting:** `REPO/docs/DOUBLE_AGENT_SYSTEM.md` §4 still prints the +40,000
rung as "0.8408 *(still training)*". The completed run's seed-101 best is 0.8436 and the three-seed
mean is 0.8448 ± 0.0018.

---


### Block D — sealed cwd12 holdout · **nc=100 head** · **yolo26x** · merged web corpora (M1 sealed, 2026-08-23)
Holdout: cwd12 test+valid, **1,977 images**; **instance count is not recorded in the M1 artifacts** (it is 3,257 in every artifact that does record it). Different backbone *and* different head from Blocks A-C.

| # | Recipe | Train data | Metric (mAP50-95) | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| D1 | Merged **raw** web corpus, yolo26x, 60 ep cap / patience 20 | 55,690 train imgs (59,134 merged, 30 datasets) | **0.6032 ± 0.0046** (per-seed 0.60553 / 0.59788 / 0.60620) | 3 seeds (101/102/103) | `DIAG/lab/m1_raw_seed101.json`, `…102`, `…103`; per-seed best from `DIAG/probe5.json` → `mega_iterm1_raw_s10*/train/results.csv` | caveat | Guard active (1,977 holdout dHashes pre-seeded; train∩holdout stems = 0). **Not a data-quality result**: the merged corpus's in-domain core is instance-starved by the pipeline's own dedup + holdout-stem filters (4,175 vs cwd12's 6,131 instances; Goosegrass 44 vs minimum 81). All runs early-stopped (ep 29/32/37), none reached the 60-ep cap. |
| D2 | Merged **DINO ≥ 0.50** curated, yolo26x | 13,309 train imgs (9 datasets) | **0.5894 ± 0.0025** (per-seed 0.58733 / 0.58875 / 0.59215) | 3 seeds | `DIAG/lab/m1_curated_seed102.json`, `…103`; **seed-101 value from `DIAG/probe5.json` → `mega_iterm1_curated_s101/train/results.csv`** | caveat | **Provenance defect: `DIAG/lab/m1_curated_seed101.json` is NOT this run** — it was overwritten by a later chained job (44304828, 48,208 merged imgs, `mega_round_count 8`). Seed 101's 13.3K value survives only in `results.csv`/`figures_data.json`. Gate skipped 36/45 slugs; curated has both higher quality **and** 4× less data — the two are confounded and volume won. |
| D3 | *Derived*: quality gate vs raw volume | — | **raw beats curated by +0.0138** | 3+3 seeds | D1, D2 | yes | The gate's demonstrated value is garbage **exclusion**, not score **lifting**. State the confound (D2) with it. |
| D4 | Tier ladder **v1** (RETRACTED), core taken from the merged corpus, +0 harvested | 2,918 imgs / 4,175 inst | **0.56005** — single run | single run, seed 101 | `DIAG/probe5.json` → `s3_tiers/run_0/results.csv`; `figures_data.json` → `tier_ladder_v1_RETRACTED_2026_08_25` | caveat | **The zero-harvest control that killed the "harvest costs 0.27" story**: 0.5601 with zero harvested images. Retracted as a *causal* result (crippled core), still valid as the control. Ladder +5k/+15k/+40k = **0.5588 / 0.54554 / 0.56117** (`run_5000/15000/40000`). Note `figures_data.json` still lists +40k as "~0.5572 (partial)" — the completed value is **0.5612**; that entry is stale. |

---

### Block E — the 15-round autonomous campaign (collect → DINO filter → train → eval)
Head **nc=100**; backbone lineage yolo26x; **seed 101 in every round → n = 1 per round, no per-round error bar**. Holdout for rounds 8-15 verified as **1,977 images / 3,257 instances** in all 8 `results.csv`/args records; rounds 1-7 have `results.csv` only (no args.yaml, no curve, no image counts).

**Series (holdout mAP50-95, one run each, each value = that run's best epoch — verified to 4 dp against the recomputed curve in 8/8 surviving rounds):**

| r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .6019 | .5919 | .5951 | .5829 | .5839 | .5738 | .5685 | .5693 | .5672 | .5711 | .5589 | .5665 | .5521 | .5594 | .5607 |

| # | Recipe / statistic | Data | Holdout | Value | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|---|
| E1 | **OLS trend, rounds 1-15** | merged pool, varies | 1,977 / 3,257 (verified r8-15 only) | **slope −0.00302 per round, se 0.00034, t = −8.84**, residual sd 0.00572 | 15 rounds × 1 run | series above; `DIAG/lab/figures_data.json` + `DIAG/probe5.json` (r1-7 = `mega_iterm1_curated_s101/train2…train8/results.csv`) | caveat | The prompt's "−0.0031, se 0.00035, t = −9.4" is slightly off; recomputed values are the ones above (t = −8.8; residual sd 0.0053 comes from dividing SSR by n instead of n−2). **The slope is dominated by rounds 1-7, which have the weakest evidence.** |
| E2 | OLS trend, **rounds 1-7** (pre-reconfiguration era) | `mega_iterm1_curated_s101/train2..8`, pool sizes **unrecorded** | **not verifiable** — no args.yaml, no val counts | **slope −0.00527/round, t = −8.11** (endpoint drop −0.0334 over 6 steps = −0.00557/round) | 7 rounds × 1 run | `DIAG/probe5.json` → `mega_iterm1_curated_s101/train2…train8/results.csv` | caveat | Carries **81% of the whole 15-round decline** (−0.0334 of −0.0412) on the half of the campaign with **no eval-set evidence at all**. These rounds *were* chained (train2 ← `mega_iterm1_raw_s103` best.pt; train3←train2; …) and ran under a different launcher (no `ITER_NAME`, no walltime cap). |
| E3 | OLS trend, **rounds 8-15** (the evidenced window) | 45,620-62,959 train imgs | 1,977 / 3,257, **verified every round** | **slope −0.00177/round, se 0.00079, t = −2.24**, residual sd 0.00512 | 8 rounds × 1 run | `DIAG/evidence_rounds.json` (all 8 rounds: `val_images` 1977, `val_instances` 3257); `DIAG/curves.txt` | yes | Residual sd 0.0051 vs a sealed per-recipe seed std of 0.0025-0.0064 → **the part of the decline standing on a provably fixed ruler is barely 2σ.** Contributes only −0.0086 (21%) of the total decline. |
| E4 | **Frozen-pool subset**: rounds 8, 10, 11, 12, 13, 14, 15 | **byte-identical merge composition**: 24 slugs, 48,752 merged / **45,620 train imgs**, 23,552 dupes skipped, 1,977 holdout stems dropped — identical in all 7 | 1,977 / 3,257 | **slope −0.00180/round, se 0.00095, t = −1.89**; spread 0.0190 (0.5521-0.5711) | 7 rounds × 1 run | `DIAG/probe3.json` (per-round `train_images` + merge log lines; per-slug line sets hash-identical across these 7 rounds); `DIAG/probe4.json` → `m1_jobscoped` (`merged_images` 48752 in all) | yes | **The strongest single row in this block: the metric keeps falling while the training set does not change at all.** Whatever drives r8→r15, it is *not* "adding more data" — nothing was added. It is also not statistically distinguishable from noise at n=7. |
| E5 | Round 9 excursion (the one round with a different pool) | **62,959** train imgs, 26 slugs — including **`fvossel__csgo_player_detection` (4,014 Counter-Strike screenshots)** and `rf_bishwarup-halder__crop-health-advisor` (15,253) | 1,977 / 3,257 | **0.5672** — within 0.004 of both neighbours (r8 0.5693, r10 0.5711) | single run | `DIAG/probe3.json` → r9 merge lines (diffed against r10) | yes | +17k images including a video-game dataset moved the holdout metric by less than the seed bar. Two facts at once: the filter admitted off-domain data, and the metric did not notice. |
| E6 | Round-over-round change within the chained segment | — | 1,977 / 3,257 | **mean −0.00122/round; 4 of 7 rounds ended ABOVE their inherited parent** (+0.00389, +0.00753, +0.00732, +0.00132) | 7 transitions | `DIAG/evidence_rounds.json` curves; `DIAG/curves.txt` | yes | Refutes "the recovery is capped" as stated. The chain is not monotonically destructive. |
| E7 | Best-epoch vs last-epoch gap | — | 1,977 / 3,257 | **mean 0.01383, range 0.00521-0.02112**, last < best in 8/8 | 8 rounds | `DIAG/curves.txt`; `DIAG/evidence_rounds.json` | caveat | **Near-vacuous as evidence**: `epochs_done == best_epoch + 20` in 8/8 with `patience=20` and no round near the 60-ep cap, so a run *can only* end on its best epoch by hitting the cap. `REPO/docs/SCIENCE_AUDIT.md` records the same property for the sealed runs that scored ~0.896. Do not present "8 of 8" as a finding. |
| E8 | Schedule defect (record-keeping) | — | — | `args.yaml` records `lr0=0.001`; the run actually used **MuSGD lr=0.01** (head group 0.03); `time: 10.0` rewrites `self.epochs` every epoch so the cosine horizon settles at **37-53, never 60**; `close_mosaic=10` fired in **1 of 8** rounds | 8 rounds | `DIAG/evidence_rounds.json` (`hyper`, `optimizer_line`, per-epoch `lr`); `DIAG/curves.txt` | caveat | Real and reportable **as a defect**, not as a cause: peak lr is constant to 0.5% across all 8 rounds while the epoch-1 dip varies 4.5× (Pearson r = 0.086), so the evidence contains **no lr contrast** and cannot identify lr as the mechanism. The last-epoch lr range is 0.00343-0.01974, not "0.016-0.020". |
| E9 | "Chained warm-start" structure | — | — | r9-r15 each init from the previous round's `weights/best.pt`; **r8 inits from `mega_iterm1_curated_s101/train8/weights/best.pt`** (the 8th link of an earlier chain), not a plain checkpoint | 8 rounds | `DIAG/evidence_rounds.json` → `init_weights` per round; `DIAG/lab/m1_curated_seed101.json` (`mega_round_count: 8`) | yes | State the lineage jump at round 8 explicitly. Rounds 1-7 were also chained (E2), so the campaign is one 15-link single-seed chain, not 15 independent trainings. |
| E10 | Ruler integrity for r8-15 | — | 1,977 / 3,257 | Holdout is **re-derived deterministically every round** from one fixed source, not reused in place | 8 rounds | `REPO/weed_llm_benchmark/weed_optimizer_framework/tools/mega_trainer.py` `_stage_cwd12_holdout` (L754-800: glob + symlink + bijective remap, no sampling/seed/filter); `DIAG/probe2.json` (`yamls`, all 8 data.yaml identical modulo round number) | yes | `val_path` is a **per-round directory** (`merged_iterrnd8…15_train/cwd12_holdout`) and the labels cache is rebuilt each round — "identical" means reproducibly re-derived, not the same directory. Equal counts alone would not have been enough; the determinism of the staging code is what carries this. |
| E11 | Class-index caution | — | — | Round space uses `CANONICAL_12_NAMES`; sealed artifacts use `CWD12_ORIGINAL_NAMES`; bijection `{0:0,1:1,2:8,3:9,4:10,5:11,6:2,7:3,8:4,9:5,10:6,11:7}` | — | `mega_trainer.py` L130-146 (`CWD12_ORIG_TO_CANON`); `DIAG/probe2.json` round `data.yaml` names | yes | Overall mAP is permutation-invariant, so the ruler is unaffected — but **any per-class comparison must be re-keyed by NAME** (index 2 = PalmerAmaranth in round space, Eclipta in sealed space). |

---

**The controlled decomposition of the decline (round 15's exact dataset, all three arms).**
Two factors, held apart. Arms A and B differ **only** in the starting weights — same data, same 30
epochs, same complete cosine, same seed 101 — so their gap is the warm-start chain and nothing else.
The third row is the recipe the campaign actually ran, whose `time=10.0` rewrites `self.epochs` every
epoch and re-plans the cosine against a clock.

| # | Arm | Start | Schedule | mAP50-95 | n | Artifact |
|---|---|---|---|---|---|---|
| E12 | **A** cold + complete | `yolo26x.pt` | 30 ep, `patience=30`, no `time=` | **0.58053** | 1 | `results/framework/ctl_chain_armA_45672628.json` |
| E13 | **B** warm + complete | round 14 `best.pt` | 30 ep, `patience=30`, no `time=` | **0.55180** | 1 | `results/framework/ctl_chain_armB_45672628.json` |
| E14 | the recipe the campaign ran | round 14 `best.pt` | 60 ep, `patience=20`, `time=10.0` | **0.5576 ± 0.0040** | **3 seeds** | round 15 + `ctl_seed_s102/s103_45672672.json` |
| E15 | *Derived*: **the start effect** | A vs B | — | **+0.0287, 5.1 σ** | — | E12, E13 |
| E16 | *Derived*: **the schedule effect** | B vs the recipe | — | **−0.0058, 1.3 σ** | — | E13, E14 |

σ throughout is **E14's own seed spread, 0.0040 over three seeds**, propagated to the difference being
quoted (√2·sd for two single runs; √(sd² + sd²/3) for a single run against a mean of three). It is the
only seed-noise measurement this project owns for this recipe — **not the borrowed 0.005**.

**E16 has the wrong sign for the obvious fix.** Completing the truncated cosine does not recover the
loss; arm B sits *below* the truncated recipe, and not separably. The warm-start chain is the factor
that carries an effect, and it is 5 σ.

**E12 and E13 are single runs.** Jobs `45824753` / `45824754` repeat both arms at seeds 102 and 103 to
put an error bar on the gap itself rather than borrowing E14's.

---


### Block F — sealed 1,977-image holdout · nc=12 · **pycocotools (COCO 101-pt) — a third evaluator**

| # | Recipe | Train data | Holdout | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|---|
| F1 | RF-DETR Large, COCO-pretrained | cwd12 train portion, 3,671 imgs (stem-filtered) | cwd12 test+valid **combined, 1,977 images** (848 test + 1,129 valid, asserted in code); **instance count not printed in any available artifact** | **0.8974 ± 0.0040**, **best run 0.9033** | **4 runs — NOT 4 seeds** | `REPO/weed_llm_benchmark/CHANGELOG.md` L3963-3993 (the 4-run table); `DIAG/lab/figures_data.json` → `rfdetr_seeds`; `REPO/weed_llm_benchmark/weed_optimizer_framework/tools/train_rfdetr.py` L167-212 (staging), L286-296 (seed), `eval_canonical` | caveat | **Three separate caveats, all mandatory.** (1) **0.9033 is the max of 4 runs and must never appear alone** — only 1 of 4 crossed 0.90. (2) **"n=4 seeds" is wrong**: `train_rfdetr.py` L286-296 states RF-DETR's `train()` accepts no seed and `--seed` is *a run label only* — the spread is GPU/cuDNN nondeterminism; two runs used the default seed, and one ran 100 epochs vs 60 for the others. Report as "4 runs, 3 configs, unseeded". (3) **pycocotools scale ≠ Ultralytics scale** — do not set 0.8974 beside A1's 0.8755. |
| F2 | *The 4 runs individually* | as F1 | as F1 | v3.0.31 60ep **0.8949** · v3.0.34-X2 100ep **0.8953** · v3.0.38 s101 60ep **0.8961** · v3.0.38 s102 60ep **0.9033** | 4 runs | `CHANGELOG.md` L3963-3975; `figures_data.json` → `rfdetr_seeds.rows` | yes | Showing all four is the honest form of F1. |
| F3 | yolo26x, "safety clean", 62/200 epochs (job timed out) | cwd12 train portion, 3,671 | cwd12 test+valid, 1,977 imgs | **0.896 — single run, Ultralytics** | single run | `CHANGELOG.md` L2440-2461 | caveat | Best at ~epoch 30 of 62 completed of 200 planned — **a truncated run**. **The same checkpoint was recorded at 0.7446 under the pyco/WBF path (`CHANGELOG.md` L3112, L2538-2560); that 0.15 discrepancy was investigated 2026-05-07 and never resolved.** This row is the reason cross-evaluator comparison is banned in this table. |

---

### Block G — **cwd12 TEST split only: 848 images / 1,464 instances** (a DIFFERENT holdout from every block above)

| # | Recipe | Train data | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| G1 | YOLO11n fine-tuned, 100 ep (2026-03 baseline) | cwd12 train split | **0.865 mAP50-95 / 0.929 mAP50 — single run** | single run | `REPO/RESEARCH_LOG.md` L2113, L2228-2232 (job 38007481); `DIAG/lab/figures_data.json` → `quality_vs_scale[7]`, `benchmark_cwd12_map50[0]` | caveat | **848-image test split — not the 1,977-image sealed holdout. Never place it in the same column as 0.8755.** No seeds. `figures_data.json` labels its training data "pre-split full set (5,648)", while RESEARCH_LOG reports distinct val (0.898) and test (0.865) numbers implying a proper split; the artifact set does not settle which. **Superseded by A1.** |
| G2 | Florence-2-base, zero-shot `<OD>` | none (zero-shot) | **0.434 mAP50 / 0.392 mAP50-95 — single pass** | single run | `RESEARCH_LOG.md` L2161, L2181; `figures_data.json` → `benchmark_cwd12_map50[1]` | yes | Best VLM. Deterministic single pass, no repeats, no prompt-sensitivity measurement. |
| G3 | Florence-2-large / InternVL2-8B / Qwen2.5-VL-3B / MiniCPM-V-4.5 / OWLv2-large / Qwen2.5-VL-7B | none | 0.329 / 0.208 / 0.196 / 0.192 / **0.184** / 0.176 mAP50 — single pass each | single runs | `figures_data.json` → `benchmark_cwd12_map50[2..7]`; `RESEARCH_LOG.md` L2109-2125 | yes | OWLv2-large is the useful one despite rank 7: **recall 0.943 / precision 0.194** — a high-recall pre-filter, which is how the pipeline actually uses it. |
| G4 | G-DINO / Molmo / Llama-Vision / Moondream / LLaVA / InternVL2-2B / InternVL2.5-8B | none | **≈ 0.000 mAP50** | single runs | `figures_data.json` → `benchmark_cwd12_map50[8..10]`; `RESEARCH_LOG.md` L2049-2051, L2138 | yes | No usable grounding (LLaVA/BakLLaVA emit 0 boxes over 848 images). Report as "no usable grounding", not as a score. |
| G5 | *Derived*: fine-tuned detector vs best VLM | — | **0.929 vs 0.434 mAP50 (2.1×)** | single runs both sides | G1, G2 | caveat | `SCIENCE_AUDIT.md` §1 row 1 calls this the strongest standalone result. Inherits G1's caveats: one run per side, 848-image split, no error bars anywhere in the benchmark. |

---

### Block H — **cross-dataset transfer: a different dataset entirely** (custom WBF matcher, class-agnostic)

| # | Recipe | Eval data | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| H1 | cwd12-trained YOLO11n (A1 checkpoints), in-domain reference, **class-agnostic** | sealed holdout, 1,977 imgs / 3,257 inst | **0.8730 ± 0.0011** | 3 seeds | `DIAG/lab/s6_crossdataset_imageweeds.json` → `summary.holdout_class_agnostic` | yes | Class-agnostic, custom matcher — **not comparable to A1's 0.8755 12-class Ultralytics number.** This is the correct in-domain anchor for H2. |
| H2 | Same models, zero-shot transfer | **ImageWeeds** (`project_agml__imageweeds_weed_detection`), **3,208 images**, CC BY 4.0 | **0.1003 ± 0.0053** | 3 seeds | same file → `summary.iw_class_agnostic`; `figures_data.json` → `crossdataset_imageweeds_2026_08_26` | yes | Leak-checked: **0 collisions at Hamming ≤ 6** vs train and vs holdout; 0 images excluded. Mechanism inspected: greenhouse/potted seedlings, median relative box area 0.036 vs cwd12's 0.105 (~3× smaller). |
| H3 | Best in-domain species, transferred | Ragweed class | **0.9604 ± 0.0018 in-domain → 0.0006 ± 0.0009 transfer** | 3 seeds | same file → `summary.holdout_ragweed`, `summary.iw_ragweed` | caveat | ImageWeeds' "ragweed" species identity (*A. artemisiifolia* vs *trifida*) is **not stated on the dataset card** — treated as same-name-class transfer. `redrootpigweed` was deliberately **not** mapped to PalmerAmaranth. Testbed is the **one** harvested source (of 6 audited) that passed the audit at precision 1.000. |

---

| H4 | **The tier ladder's own eight checkpoints, on the second exam** | ImageWeeds, 3,208 images, class-agnostic | **arm A −0.0299** and **arm B −0.0100** across +0 → +40,000 | 8 checkpoints, 1 eval each | `results/framework/s6_crossdataset_ladder.json` (job `45817696`); `docs/poster/xds_ladder.json` | yes | Same implementation, same leak check, same matcher, same conf and imgsz as H1-H3 — `crossdataset_eval` now takes its checkpoints from `XDS_WEIGHTS`, so this is not a second evaluation that can drift from the first. **0 images excluded by the leak check.** |
| H5 | *Derived*: **does the ladder fall because of the exam?** | — | cwd12 **−0.0189**, ImageWeeds **−0.0299 / −0.0100** | H4 + C1-C5 | — | yes | The objection this answers: cwd12's metric rewards training data that looks like cwd12, so a greenhouse/aerial/Latvian corpus could only ever cost. If that were the whole story, the ImageWeeds number would **rise** while cwd12 fell. **Both fall.** The ladder's shape is the data, not the exam. |



### Block I — robot frames: **not an accuracy measurement** (no ground truth)

| # | What | Data | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| I1 | cwd12-trained YOLO11n s102 over stored robot frames | **358 frames, 8 sessions**, 2 robots (lasercar, 241robot) | **0 detections at conf 0.25 / 0.40 / 0.60; false-positive frame rate 0.000; 3.7 ms/frame** | 358 frames, 1 model | `DIAG/lab/s6_domain_gap.json`; `figures_data.json` → `s6_domain_gap_2026_08_25` | caveat | **These frames contain no weeds** (indoor/bench scenes), so this measures **false-positive behaviour only — it is NOT recall and must never be shown as one.** Recall/precision on robot frames that *do* contain weeds is unmeasured; field accuracy is unknown. |

---

| I2 | **Same checkpoint, every frame the robots recorded** (supersedes I1's coverage) | **2,686 frames, 26 sessions**, 2 robots; **1,157 frames are >1/3 vegetation** by excess green | **25 frames fire at conf 0.25** (21 of them vegetated), **8 at 0.40**, **2 at 0.60**; one box per firing frame; **24 of the 25 boxes are the same class** (Purslane) | 2,686 frames, 1 model | `results/framework/s6_field_fire.json`; `weed_optimizer_framework/tools/field_fire_sweep.py` | caveat | **No ground truth exists for these frames, so this is a fire rate and NOT a recall.** The vegetation rule is what stops "there was nothing to detect" from accounting for it. I1 measured 358 indoor/bench frames; this walks the whole session tree. |
| I3 | *Why I1 undercounted* | — | the earlier sweep walked only `<slug>/frames/`, so every bulk-uploaded session was skipped — including the **1,013-frame field drive** that holds **802** of the vegetated frames and **23** of the 25 firings | — | same | yes | Recorded because it is the reason the sweep is now a committed script rather than a heredoc. |



### Block J — pre-guard historical rows: **cite as history only, never as results**

| # | Recipe | Train data | Holdout | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|---|
| J1 | Naive scale + OWLv2 pseudo-labels | 244,675 harvested imgs | **unknown/unverified** (filename-only holdout defence at the time) | 0.593 — single run | single run | `figures_data.json` → `quality_vs_scale[3]`; `DOUBLE_AGENT_SYSTEM.md` §4 | **no** | Pre-guard: no content-level holdout protection existed. Sealed replacement is D1 (0.6032 ± 0.0046), which confirmed the pre-guard value was **not** leak-inflated. Quote D1 instead. |
| J2 | Cumulative scale variant | ~150-240K | **unknown/unverified** | 0.576 — single run | single run | `figures_data.json` → `quality_vs_scale[4]` | **no** | Same as J1; x-value is approximate (`x_approx: true`). |
| J3 | "Curated clean subset" 0.896 | 3,671 (cwd12-only staging) | 1,977 imgs, Ultralytics | 0.896 — single run | single run | `figures_data.json` → `quality_vs_scale[2]`; `CHANGELOG.md` L2440-2461 | **no** | Same run as F3 — **truncated at 62/200 epochs, best at ~epoch 30, single run, and 0.7446 under the other evaluator.** It was reclassified in v3.22.3 as cwd12-only staging (zero merge calls), so it never belonged on the "curated harvest" curve at all. |

---

### Block K — corpus & governance facts a reviewer will ask for (counts, not metrics)

| # | Fact | Value | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|
| K1 | Merge funnel (M1 raw, job 44234193) | 156,521 registry labelled imgs → **59,134 unique**; 44,750 cross-dataset duplicates skipped; 1,977 holdout stems dropped; 30 datasets | `figures_data.json` → `merge_funnel_2026_08_23` | yes | The web pool is heavily redundant — the same weed photos re-exported across Roboflow projects. "More images" ≠ more information. |
| K2 | S1 harvest audit gate | 6 audited sources, 13,527 labelled imgs; **only 1 source (3,208 imgs) clears the 0.90 bar**; others 0.18-0.74; **verdict: NOT MET** | `figures_data.json` → `s1_gate_verdict_2026_08_25` | yes | The probe reads 1.000 on human-labelled cwd12, so the low scores are the data, not the instrument. |
| K3 | Registry pool | 55 datasets (51 active, 4 quarantined); 121,721 active imgs; 120,515 labelled; **105 distinct class names** | `DIAG/lab/pool_report.json` | yes | Quarantined items include maritime SAR, underwater fish, low-light street scenes, wheat-head detection — i.e. the harvester did ingest off-goal datasets. |
| K4 | License mix | 38/45 datasets carry an explicit license (**33× CC BY 4.0**); 7 unresolved/unreachable → non-redistributable | `figures_data.json` → `license_sweep_2026_08_23`; `pool_report.json` → `license_mix` | yes | `SCIENCE_AUDIT.md` §3.4: exclude `unspecified` from any redistributed figure. |
| K5 | Open security item | Kaggle API token remains in public git history; rotation **not confirmed** | `REPO/docs/SCIENCE_AUDIT.md` §3.5 | yes | The only open critical from the seven-dimension audit. Do not present the repo as clean until rotation is confirmed. |


---

### Block L — **the supervision benchmark: what catches the pipeline failing**
One frozen corpus of 162 real incidents from this project's own engineering record, scored on the
**dev split: 149 cases, 116 incidents, 33 controls**. Every number is produced by the project's own
instrument — `bench reproduce --split dev`, which makes **no model call** and re-reads the committed
verdict files, so it can be re-run at any time and cannot drift from what was measured.

**Scorer definition, because it is stricter than it looks.** A **detection** counts only when the arm
returns an `issue` verdict carrying at least one finding **at or above a severity bar**. A case that
produced no answer — model error, context overflow, undecidable export — **leaves the denominator**
instead of counting as a miss, which is why the incident denominators read 113 / 114 / 57 and not 116.
**Grounded** adds one more requirement: the finding must quote a line that resolves in the artifact.
**Citation validity** is the fraction of quoted lines that resolve, pooled over findings (not
independent within a case — its interval is optimistic and the renderer says so).

The tiers: **A0** is the scheduler's own watchdog reading status fields; **A0p** is twelve
pre-registered deterministic checks; **L2** is a model reading raw artifact excerpts; **L3** is the
same model with a retrieval round over the same artifacts.

| # | Arm | Size | Reads | Recall | Grounded | False alarms | Citation validity | SU / review | Cases |
|---|---|---|---|---|---|---|---|---|---|
| L1 | **A0** scripted watchdog | — | status fields | **—** | — | — | — | 0 | 149 |
| L2 | **A0p** deterministic signals | — | 12 checks | 0.095 | 0.095 | 0.061 | 0.846 | 0 | 149 |
| L3 | Qwen2.5-7B · **L2** | 7 B | artifacts | 0.777 | **0.223** | 0.152 | **0.304** | 0.0063 | 149 |
| L4 | Qwen2.5-7B · **L3** | 7 B | artifacts + retrieval | **0.823** | **0.274** | 0.273 | **0.195** | 0.0065 | 149 |
| L5 | Qwen3-14B · **L2** | 14 B | artifacts | 0.664 | 0.602 | **0.606** | 0.863 | 0.0163 | 149 |
| L6 | Qwen3-14B · **L3** | 14 B | artifacts + retrieval | 0.675 | 0.614 | **0.636** | 0.820 | 0.0157 | 149 |
| L7 | Qwen3.8-27B · **L2** | 27 B | artifacts | 0.553 | 0.518 | 0.212 | **0.951** | 0.0298 | 149 |
| L8 | Qwen3.8-27B · **L3** | 27 B | artifacts + retrieval | 0.702 | **0.702** | 0.242 | **0.968** | 0.0326 | **92** ⏳ |
| L9 | GLM-4.7-Flash · **L2** | 30 B | artifacts | 0.821 | 0.769 | 0.200 | 0.836 | 0.0508 | **78** ⏳ |


**L1 is not recall 0.000.** All 149 of A0's verdicts score `undecidable`: `escalate.reason: "no signal
fired"` is not a judgement about the case. 0.000 would read as an arm that looked and found nothing;
the truth is an arm that never produced a finding to score. This is the baseline the project's
dashboard ran for six months.

**Recall does not order the reviewers by size. Grounded recall does, and it does it steeply.**
The 7 B has the **highest recall on the table** — 0.823 at L3, above the 27 B's 0.702 — and **0.274**
of it is grounded. Two thirds of what it "detects" quotes a line that does not resolve in the
artifact, and its citation validity is **0.195** against the 27 B's **0.968**. Read as a single
column:

| size | recall | grounded | **unevidenced** | citation validity | SU / review |
|---|---|---|---|---|---|
| 7 B | 0.823 | 0.274 | **0.549** | 0.195 | 0.0065 |
| 14 B | 0.675 | 0.614 | 0.061 | 0.820 | 0.0157 |
| 27 B | 0.702 | 0.702 | **0.000** | 0.968 | 0.0326 |
| 30 B (L2) | 0.821 | 0.769 | 0.051 | 0.836 | 0.0508 |

**This is the measured case for the placement rule.** The reviewer this campaign ran on between
2026-09-04 and 09-11 was a 7 B-class model on the lab's 3060. On recall alone it looks like the best
arm on the page. It is the grounded column and the citation column that separate it, and neither was
being reported. Five times the compute per review (0.0065 → 0.0326 SU) buys the difference between
**55 % unevidenced detections and 0 %**.

False alarms are the other axis and they do not order by size either: the 14 B flags **21 of the 33
controls** (0.61–0.64), while the 7 B flags 0.15–0.27 and the 27 B 0.21–0.24.

**What the tier axis moves.** The retrieval round over the same artifacts is worth **+0.149** recall to
the 27 B (0.553 → 0.702) at +0.030 false alarms, **+0.012** to the 14 B, and **+0.046** to the 7 B —
but the 7 B's grounded recall moves only +0.051 while its false alarms go up +0.121, so what retrieval
buys the smallest model is mostly more firing. **Tiering pays where the model can use it and not
otherwise** — the benchmark's one genuinely new result, and the reason the arms are drawn as arrows
rather than points.

**Ceiling for any rules-only arm: 0.559** — 56 of the 127 incidents declare no deterministic signal
that could reach them, so A0p's 0.095 is 17 % of what the rules could achieve even in principle.

**Two rows are partial** (⏳). `L8` stands on 92 of 149 cases and `L9` on 78; jobs `45824752` and
`45824751` complete them with `bench run --resume`, which re-reads the committed verdicts and calls the
model only for the remainder. Their recalls may move; their denominators are stated on every row so
nothing is quoted as complete that is not.

**DeepSeek-V4-Flash is absent by the instrument's own rule**, not by hand: its verdicts predate the
2026-09-07 split re-cut, so the scorer will not mix them with the current corpus. Its earlier
0.388 came from the run retracted for context overflow (51 of 149 prompts refused).

**One superseded number, recorded so it is not re-quoted.** `docs/SUPERVISION_BENCH_LIMITS.md` §7
reported Qwen3.8-27B L2 at **0.841**; that came from a counter written for that table which scored a
case as detected whenever the arm raised anything. Under the committed scorer the same verdicts read
**0.553**. §7a of that document records the difference. **One axis, one scorer.**

Artifacts: `results/framework/supervision_rescore/results/run_reproduce-20260912T134625.json`,
committed as `docs/poster/supervision_table.json`; verdicts under
`results/framework/supervision_bench/verdicts/<arm>_<model>/`.

---


## A0. The protocol problem, and every number re-read three ways

**There is no dev set anywhere in this project.** The 1,977-image cwd12 test+valid set is
passed to Ultralytics as `val` (`cwd12_sealed.yaml`; `strategy["val_dataset_root"]` in
`run_m1_merged_seeds.sh`). It therefore drives `patience`, selects `best.pt`, and is then the
number reported. Every published figure is a **maximum over 21–100 evaluations taken on the
set it is reported on**, and that set has been the selection signal since 2026-03-15 across
S3, M1, the tier ladder, the DINO thresholds, the WBF/TTA sweep and all fifteen campaign
rounds. Calling it "sealed" is wrong and must stop.

Fixing the protocol needs a three-way split and 80–120 GPU-hours of retraining, which does
not fit before 2026-09-20. What does fit, at zero GPU cost, is **measuring the size of the
bias and reporting every row at three readouts**, straight from the `results.csv` files that
already exist.

| recipe | best epoch (**as published**) | last epoch | mean of last 5 | selection optimism |
|---|---|---|---|---|
| YOLO11n COCO-pretrained · cwd12 3,671 · 100 ep | 0.8755 ± 0.0029 | 0.8713 ± 0.0039 | 0.8708 ± 0.0045 | **+0.0047** |
| YOLO11n random init · cwd12 3,671 · 100 ep | 0.8041 ± 0.0028 | 0.8020 ± 0.0027 | 0.8018 ± 0.0020 | +0.0023 |
| Mamba-YOLO-T random init · cwd12 3,671 · 100 ep | 0.8266 ± 0.0064 | 0.8258 ± 0.0056 | 0.8253 ± 0.0057 | +0.0013 |
| M1 merged **raw** · cold · yolo26x | 0.6032 ± 0.0046 | 0.5914 ± 0.0025 | 0.5921 ± 0.0030 | **+0.0111** |
| M1 merged **curated** · cold · yolo26x | 0.5894 ± 0.0025 | 0.5725 ± 0.0038 | 0.5727 ± 0.0026 | **+0.0167** |

n = 3 seeds in every cell. Source: `results/framework/<run>/results.csv`, full mAP50-95(B)
column, 41 runs (`scratchpad/diag/probe6.json`).

**The optimism is the same size as, or larger than, the seed noise it is quoted against** —
+0.0047 against ± 0.0029 for the flagship recipe, and +0.0111 / +0.0167 against ± 0.0046 /
± 0.0025 for the merged recipes. Any effect below about 0.02 that was established by
comparing best-epoch numbers needs re-reading in this table before it is claimed.

### Which conclusions survive the re-reading

| claim | at best epoch | at mean of last 5 | verdict |
|---|---|---|---|
| COCO pretraining is worth more than architecture | +0.0714 | +0.0690 | **survives** |
| Architecture at equal random init (Mamba-T over YOLO11n) | +0.0225 | +0.0235 | **survives** |
| The DINO≥0.50 quality gate does not beat raw volume | +0.0138 | +0.0194 | **survives, and grows** |

### The fifteen-round series at three readouts

| round | epochs | best (published) | at epoch | last | mean last 5 | optimism |
|---|---|---|---|---|---|---|
| r1 | 21 | 0.60188 | **1** | 0.57746 | 0.58370 | +0.0182 |
| r2 | 24 | 0.59188 | 4 | 0.57806 | 0.57920 | +0.0127 |
| r3 | 27 | 0.59511 | 7 | 0.57007 | 0.57191 | +0.0232 |
| r4 | 28 | 0.58290 | 8 | 0.56467 | 0.56597 | +0.0169 |
| r5 | 27 | 0.58392 | 7 | 0.56415 | 0.56457 | +0.0193 |
| r6 | 27 | 0.57385 | 7 | 0.55386 | 0.55818 | +0.0157 |
| r7 | 27 | 0.56854 | 7 | 0.54677 | 0.55089 | +0.0177 |
| r8 | 25 | 0.56928 | 5 | 0.55241 | 0.55509 | +0.0142 |
| r9 | 31 | 0.56723 | 11 | 0.54768 | 0.55355 | +0.0137 |
| r10 | 28 | 0.57112 | 8 | 0.55501 | 0.55628 | +0.0148 |
| r11 | 35 | 0.55893 | 15 | 0.54593 | 0.54720 | +0.0117 |
| r12 | 32 | 0.56646 | 12 | 0.54534 | 0.54793 | +0.0185 |
| r13 | 21 | 0.55208 | **1** | 0.54687 | 0.54707 | +0.0050 |
| r14 | 32 | 0.55940 | 12 | 0.54985 | 0.54920 | +0.0102 |
| r15 | 25 | 0.56072 | 5 | 0.55146 | 0.55011 | +0.0106 |

| readout | slope / round | se | t | r1 → r15 |
|---|---|---|---|---|
| best epoch (the published series) | −0.00302 | 0.00034 | −8.86 | 0.6019 → 0.5607 |
| last epoch | −0.00213 | 0.00039 | −5.52 | 0.5775 → 0.5515 |
| mean of last 5 | −0.00237 | 0.00032 | −7.32 | 0.5837 → 0.5501 |

**The decline is not a selection artifact.** It survives all three readouts at t = −5.5 to
−8.9. What the readouts do change is the *magnitude*: the published series overstates the
level by a mean of +0.0148 per round (range +0.0050 to +0.0232).

Two rounds are worth naming: **r1's and r13's published numbers are epoch 1** — the headline
for those rounds is the inherited checkpoint after a single epoch of training, not what that
round's training produced.


---

## D. Numbers that must never be quoted bare

- 0.9033 — best-of-4 RF-DETR runs, not a seed mean. Must always read '0.8974 ± 0.0040 over 4 runs, best 0.9033'; only 1 of 4 runs crossed 0.90, and the runs were unseeded (train_rfdetr.py: --seed is a run label only) across two different epoch budgets (60 and 100).
- 0.8974 ± 0.0040 'n=4 seeds' — the phrase 'n=4 seeds' is factually wrong in both DOUBLE_AGENT_SYSTEM.md §4 and SCIENCE_AUDIT.md §1 row 4. It is 4 runs / 3 configs / no seed control; variance is GPU-cuDNN nondeterminism. Also pycocotools, not Ultralytics.
- 0.8755 ± 0.0029 vs 0.8974 ± 0.0040 — never present as a model comparison. Different evaluators (Ultralytics vs pycocotools); the one measured cross-evaluator offset on this project is 0.0239 in one direction for YOLO11n and 0.15 in the other for yolo26x.
- 0.8830 (ens3_tta), 0.8728, 0.8726, 0.8653, 0.8524 — all on the custom WBF matcher. Each must be quoted against its own baseline 0.8554, never against 0.8755 or 0.8794.
- 0.8554 (plain_s102) — same checkpoint reads 0.8794 under Ultralytics; quoting it as 'the model's score' understates it by 0.0239.
- 0.907 — never measured. It is 0.8830 + 0.0239 assuming the matcher offset composes with TTA, which was never tested. Do not show it.
- 0.896 (v3.0.28 'safety clean' / 'curated clean subset') — single run, TIMED OUT at 62 of 200 epochs, best at ~epoch 30, and the same checkpoint was recorded at 0.7446 under the other evaluator.
- 0.865 / 0.929 (2026-03 YOLO11n baseline) — measured on the 848-image / 1,464-instance TEST split, not the 1,977-image sealed holdout; single run; superseded by 0.8755 ± 0.0029 (n=3).
- 0.434 (Florence-2) and every VLM benchmark number — single deterministic pass per model on the 848-image test split; no repeats and no confidence intervals exist anywhere in that table.
- 0.593 and 0.576 — pre-guard, single runs, holdout composition unverified. Cite only as history alongside their sealed replacement 0.6032 ± 0.0046 (n=3).
- 0.5601 (tier-ladder v1 core+0) — retracted as a causal result; the core was stripped to 2,918 images / 4,175 instances by the pipeline's own filters. Valid only as the zero-harvest control that refutes the '0.27 cost' claim.
- 0.8636 (core+0, nc=100) — carries a measured −0.012 head penalty versus the nc=12 run. Always state the head when placing it beside 0.8755.
- 0.8408 (core+40,000) — STALE. DOUBLE_AGENT_SYSTEM.md §4 still marks it '(still training)'; the completed value from results.csv is 0.8436. Fix the doc.
- '~0.5572 (partial)' for tier-ladder v1 core+40,000 — stale in figures_data.json; the completed value is 0.5612.
- 0.8041 ± 0.0028 (YOLO11n from scratch) — real, but the three sidecar artifacts lab/s3_yolo11n_scratch_seed10*.json contradict it with copied pretrained values. Cite results.csv / figures_data.json, and disclose the corrupted sidecars.
- −0.0031 per round, t = −9.4 — recompute before use: the correct OLS values are slope −0.00302, se 0.00034, t = −8.84, residual sd 0.00572 (n−2). Never quote the 15-round slope without stating that rounds 1-7 carry 81% of the decline on unevidenced eval sets, and that the evidenced window r8-15 gives only −0.00177, t = −2.24.
- 'The evaluation set is identical every round' — verified for rounds 8-15 only. Rounds 1-7 have results.csv but no args.yaml, no val counts and no curves.
- 'Last epoch worse than best in 8 of 8 rounds' — an artifact of patience=20 with no round reaching the 60-epoch cap, not evidence of anything. The reported per-round metric is the best epoch, so last-epoch values never enter the regression.
- 0.1003 (ImageWeeds transfer) — class-agnostic, custom matcher, and its in-domain anchor is 0.8730 (class-agnostic), not 0.8755.
- 0 detections / 0.000 false-positive rate on 358 robot frames — NOT recall. Those frames contain no weeds; recall on weed-containing robot frames is unmeasured.
- 'The re-heat destroys the inherited model' — the inherited checkpoint was never evaluated at epoch 0 in any of the 8 rounds, so its score on this holdout is unmeasured everywhere; and 4 of 7 chained rounds finished ABOVE their parent.

---

## E. What is missing

- Path key for every provenance string: DIAG/ = /private/tmp/claude-501/-Users-xiaogui-Desktop-2026spring-research-weed-llm-benchmark/1ae39708-9141-44c7-bd5c-13a7f37399e7/scratchpad/diag/ ; REPO/ = /Users/xiaogui/Desktop/2026spring/research/weed_llm_benchmark/
- Rounds 1-7 have only results.csv best/last values (DIAG/probe5.json, mega_iterm1_curated_s101/train2..train8). Missing: args.yaml, per-epoch curves, val image/instance counts, and merged pool sizes. These 7 rounds carry 81% of the campaign decline, so the headline slope rests on the least documented half.
- No per-round error bar anywhere in the campaign: seed 101 in all 15 rounds, one run each. The nearest sealed per-recipe seed stds are 0.0046 (M1 raw), 0.0025 (M1 curated) and 0.0029 (cwd12 YOLO11n), so round-to-round differences below ~0.01 are not resolvable. No round was ever repeated.
- THE decisive missing experiment: DIAG/run_ctl_chain.sh defines arms A (cold start from yolo26x.pt, epochs=30/patience=30) and B (warm start from round 14, epochs=30/patience=30) on round 15's exact merged dataset, against round 15's own 0.5607. No result artifact for either arm exists in the evidence set. Until it runs, 'chaining causes the decline' is untested and must not be presented as a finding.
- The inherited checkpoint is never validated at epoch 0 in any of the 8 surviving rounds, so the true starting score of each round is unmeasured — every 'the warm-up destroyed X' statement compares against a number that was never taken.
- DIAG/lab/s3_yolo11n_scratch_seed101/102/103.json are corrupted: they carry the pretrained runs' metrics (0.8739/0.8789/0.8737) instead of the scratch values (0.80551/0.80089/0.80587). The sidecar JSONs should be regenerated from results.csv before anyone else reads them.
- DIAG/lab/m1_curated_seed101.json does not describe the 13,309-image curated seed-101 run; it was overwritten by a later chained job (44304828, 48,208 merged images, mega_round_count 8). That seed's 0.5873 survives only in probe5/figures_data.
- Mamba-YOLO-T (0.8266 ± 0.0064) has no second source — the values exist only in the three sidecar JSONs; probe4 lists the run dirs without parsed results.csv bests. One of the four Mamba runs failed outright ('no results.csv produced').
- Instance counts are missing for several holdouts: the M1 rows record 1,977 images but no instance count; RF-DETR's eval prints 848 + 1,129 images but no annotation total in any artifact available here; ImageWeeds records 3,208 images but no instance count.
- No common evaluator exists. Three scales are in play (Ultralytics, pycocotools, custom WBF matcher) and only one offset has ever been measured (0.0239, one checkpoint, no TTA). RF-DETR has no calibration against Ultralytics at all, and the yolo26x 0.896-vs-0.7446 discrepancy was never explained. A single-evaluator re-scoring of the headline checkpoints is the cheapest way to make the whole table comparable.
- No artifact records which class list Ultralytics averaged over in the nc=100 round runs (12 GT-bearing classes vs any aux classes that received predictions). The −0.012 head offset from C1-vs-A1 is the only bridge, and it was measured on one pair of runs.
- DOUBLE_AGENT_SYSTEM.md §4 is stale in two places: the +40,000 tier row still reads '0.8408 (still training)' when the completed value is 0.8436, and figures_data.json still lists tier-ladder v1 +40k as '~0.5572 (partial)' when the completed value is 0.5612.
- VLM benchmark: one deterministic pass per model, no repeats, no prompt-sensitivity or decoding-temperature sweep, and no error bars — 15 models compared on single runs.
- The 2026-03 baseline's training composition is unresolved: figures_data.json calls it the 'pre-split full set (5,648)' while RESEARCH_LOG reports separate val (0.898) and test (0.865) numbers. No stem list survives to prove the 848 test images were excluded from training.
- ImageWeeds transfer is class-agnostic only, and the species identity of its 'ragweed' class (A. artemisiifolia vs trifida) is not stated on the dataset card — the 0.9604 → 0.0006 collapse is same-name-class, not verified same-species.
- Robot deployment: recall and precision on robot frames that actually contain weeds are unmeasured (all 358 stored frames are weed-free indoor/bench scenes). Field accuracy is unknown, as the model card itself states.
- Rounds 10-15 added zero net new unique images to the training pool (identical 24-slug merge, 45,620 train images in all of them). No artifact explains why six consecutive collect-and-filter cycles contributed nothing — that, not the metric drift, is the loop's most reportable failure and it has no root-cause evidence.
- Kaggle API token remains in public git history with rotation unconfirmed (SCIENCE_AUDIT.md §3.5) — the only open critical from the seven-dimension audit, and a reviewer-visible one if the repo is shared.
