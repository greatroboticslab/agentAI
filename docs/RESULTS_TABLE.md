# Results table — every number with its provenance

*Draft of 2026-09-10. Built by re-reading the artifacts, not by copying the earlier tables.
Four separate errors in the previously published tables are corrected here and listed in §D.*

**Status:** the two control arms (`45672628_[1-2]`) and the two seed repeats (`45672672_[1-2]`)
are running; their rows are marked PENDING and must be filled before this is shown to anyone.

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

### Block C — sealed cwd12 holdout · **nc=100 head** · YOLO11n pretrained · tier ladder v2
Holdout: same 1,977 images / 3,257 instances. This block is the **bridge** between Block A's clean numbers and the nc=100 pipeline numbers in Blocks D-E.

| # | Recipe | Train data | Metric (mAP50-95) | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| C1 | Clean core + **0** harvested, nc=100, 100 ep cap | real cwd12 train split, 3,671 (6,131 inst) | **0.8636** (best epoch 77 of 97) — **single run** | single run, seed 101 | `DIAG/probe5.json` → `s3_tiers/v2_run_0/results.csv`; `DIAG/lab/figures_data.json` → `tier_ladder_v2_2026_08_25` | yes | **This row is the measured nc=100 head cost: 0.8636 vs A1's 0.8755 = −0.012.** Quote that offset whenever an nc=100 number is set beside an nc=12 number. |
| C2 | Core **+5,000** harvested | 8,671 | **0.85988** (Δ −0.004) — single run | single run, seed 101 | `DIAG/probe5.json` → `s3_tiers/v2_run_5000/results.csv` | yes | Δ is inside seed noise (0.003-0.006). |
| C3 | Core **+15,000** harvested | 18,671 | **0.86143** (Δ −0.002) — single run | single run, seed 101 | `DIAG/probe5.json` → `s3_tiers/v2_run_15000/results.csv` | yes | Δ inside seed noise. |
| C4 | Core **+40,000** harvested | 43,671 | **0.84362** (Δ −0.020) — single run | single run, seed 101 | `DIAG/probe5.json` → `s3_tiers/v2_run_40000/results.csv`; `figures_data.json` | caveat | **`REPO/docs/DOUBLE_AGENT_SYSTEM.md` §4 still prints "0.8408 *(still training)*" — that table is stale; the completed run's best is 0.8436.** Fix the doc before presenting. Single seed: −0.020 is ~4× the seed bar but rests on one run. |
| C5 | *Derived*: honest cost of adding harvested data to a clean core | — | **0.00 to −0.02 across +5k…+40k** | 4 single runs | C1-C4 | yes | Supersedes and retracts the earlier "0.27 cost" claim (`DOUBLE_AGENT_SYSTEM.md` §4, retraction dated 2026-08-25). The finding is a **flat curve**: 12× the data buys nothing. |

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

### Block I — robot frames: **not an accuracy measurement** (no ground truth)

| # | What | Data | Metric | n | Artifact(s) | Q | Caveat |
|---|---|---|---|---|---|---|---|
| I1 | cwd12-trained YOLO11n s102 over stored robot frames | **358 frames, 8 sessions**, 2 robots (lasercar, 241robot) | **0 detections at conf 0.25 / 0.40 / 0.60; false-positive frame rate 0.000; 3.7 ms/frame** | 358 frames, 1 model | `DIAG/lab/s6_domain_gap.json`; `figures_data.json` → `s6_domain_gap_2026_08_25` | caveat | **These frames contain no weeds** (indoor/bench scenes), so this measures **false-positive behaviour only — it is NOT recall and must never be shown as one.** Recall/precision on robot frames that *do* contain weeds is unmeasured; field accuracy is unknown. |

---

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
