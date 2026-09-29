# Incremental training protocol (INC)

The question this protocol answers: starting from a high-precision base dataset,
can a detector keep improving when small, fixed-size increments of new data are added
one at a time? Each increment is checked, and an increment that hurts is attributed
and rolled back. The protocol decides this with a measurement that can tell a real
effect from seed noise.

Nothing before this protocol tested that design (docs/CWD12_SPECIES.md §5,
docs/REGRESSION_DIAGNOSIS.md). The closest earlier runs had one or more of these
defects:
- the sealed holdout was also the validation set;
- names were mis-joined;
- the training pool was frozen;
- increments were uncontrolled;
- promotion was unconditional;
- the LoRA wiring was broken (Step 0.5).

## Fixed conventions (all INC experiments)

| Item | Value |
|---|---|
| Code | `weed_optimizer_framework/tools/inc/` |
| Results | `$REPO/results/framework/inc/` on the cluster (`REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark`) |
| Class space | nc = 13. Ids 0-11 are the cwd12 species in **cwd12 id order** (`cwd12_species.CWD12_SPECIES`: Waterhemp … CutleafGroundcherry); id 12 = `OtherPlant`. Trainer-slot ids are never used inside INC. |
| Detector | YOLO11n from `yolo11n.pt` (COCO), imgsz 640, batch 32 |
| Cold run ("union") | 100 epochs, SGD, lr0 0.01, cosine, lrf 0.01, warmup 3 epochs, `val=False` |
| Incremental run ("cheap arm") | Init from the incumbent weights; 30 epochs; SGD set explicitly (never `auto`); lr0 0.002 (LoRA: 0.01); warmup 1 epoch; cosine to lrf 0.01; `val=False` |
| Weights that are scored | The final annealed EMA weights: `last.pt`, or for LoRA `last_merged.pt` (the adapters merged into plain convs, loadable by plain `YOLO()`). `best.pt` is never used, so no split is ever used to select a checkpoint. |
| Seeds | 3 per arm (0, 1, 2), `deterministic=True` |
| Scorer | `inc/scorer.py`, the only producer of metrics (below) |

## Splits

| Split | What it is | Used for |
|---|---|---|
| `train_core` | cwd12 train minus `dev` | Training |
| `dev` | Whole capture sessions of cwd12 train (session = filename minus the frame number; 35 sessions), about 15% of images. Every species has ≥ 30 boxes. Chosen by an exact deterministic search (on the laptop copy: 8 sessions, 617 images, 16.8%, at least 31 boxes per species). | Every accept/reject decision (the search split). Never trained on. |
| `test` | cwd12 valid + test, 1,977 images (the historical sealed holdout) | Read only at anchor points: once at the end of each experiment and for base models. Never used in a decision. |
| `ood22` | 3SeasonWeedDet10 data2022 (Mississippi, 2022, 1,948 images), from the Zenodo original (record 14861516), converted from VGG JSON | Out-of-season exam |
| `ood23` | 3SeasonWeedDet10 data2023 (Michigan, 2023, 1,748 images) | Out-of-site and out-of-season exam |
| `imageweeds` | `project_agml__imageweeds_weed_detection` (3,208 images) | Class-agnostic exam, plus Ragweed |

For the exams:
- An exam image within 6 dHash bits of any cwd12 image (train, valid or test) leaves the exam and is counted, so the out-of-domain exams hold no cwd12 photograph. data2022 shares a capture session (`20220129_CanonEOS4000D_EO`) with cwd12 train.
- Boxes are normalised by the image size the trainer sees after EXIF orientation. Images whose annotation frame cannot be verified against it are dropped and counted.
- Lambsquarters (not a cwd12 species) is `OtherPlant` in the exam labels.
- The species score is reported over the 12 cwd12 ids present in the exam.
- The class-agnostic score treats every box as one class.

Every split image is added to a never-train index: its dHash, radius `near_dup.HOLDOUT_NEAR_DUP_BITS` (6). Building any training set refuses (hard error) to include an image within that radius of a dev, test or exam image. Split manifests list path, sha256 and label sha256. `LOCK.json` records the sha256 of every manifest and of `scorer.py`.

## Scorer (the only source of metrics)

`python -m weed_optimizer_framework.tools.inc.scorer --weights W --exam E --out DIR`
- **Before scoring**, it re-hashes the exam manifest and its own source file and compares them with `LOCK.json`. On any mismatch it refuses to score.
- **Metrics:**
  - Ultralytics val at imgsz 640, conf 0.001, iou 0.7, nc 13;
  - mAP50-95, mAP50 and per-class AP50-95;
  - class-agnostic mAP50-95 (the same predictions with every class collapsed; computed by the scorer, not by `single_cls` retraining);
  - a per-image correctness bit at conf 0.25, IoU 0.5: correct = every GT matched with the right class and no false positive.
- **Output:** `score.json` stamped with the scorer and manifest hashes.
- **Isolation:** training code never writes metrics. The driver reads only `score.json` files.
- **Pinned versions:** the scorer pins Ultralytics 8.4.37 (the cluster's) inside `scorer.py`, so the lock covers it: the same predictions give different AP under 8.4.22 and 8.4.37. A production score also needs CUDA, imgsz 640 and batch 32. A score taken under `INC_SCORER_TESTING=1` carries `production=false` and a `TEST-` stamp, and the gate refuses it.

## Gate (the per-increment decision) — `inc/gate.py`

For step k with candidate increment D_k, one recipe, and incumbent weights W:

| Symbol | Meaning |
|---|---|
| `inc` | W scored on dev at epoch 0 (the parent). |
| `cand[s]` | W fine-tuned on D_k plus an equal-size replay sample of the accepted pool; seeds s = 0..2. |
| `null[s]` | Identical to `cand`, but D_k is replaced by an equal-size extra replay sample. Same compute, no new data. |

These are the runs of replay mode `sample` (pilot_v1). Replay mode `full` is described under Step 4, "Pilot v2 (full rehearsal)".

Statistics on dev mAP50-95:
- `P_data = P(cand > null)` over all 3×3 seed pairs (a tie counts 0.5) (Bouthillier et al. 2021).
- `P_recipe = P(null > inc)`.

Guards:
- **Regression:** mean(cand) ≥ inc − 2·sd(null).
- **Species:** no species' mean(cand) − mean(null) < −max(0.03, 3·sd_species(null)).
- **Flips:** negative flips vs W. The mean for cand minus the mean for null must be ≤ 2·sd(null flips) + 3 images (Pioneer Agent). This is protocol v1, the default. Protocol v2 counts net flips instead (below, "Protocol v2 (net flips)"); an experiment opts in explicitly.

Decision:
- **ACCEPT:** P_data ≥ 0.75 and all guards pass.
  - D_k joins the accepted pool.
  - The new incumbent is the uniform soup of the three cand weights if the soup scores ≥ mean(cand) on dev; otherwise it is `cand[0]`.
- **HOLD:** 0.25 < P_data < 0.75 and all guards pass. The effect is not detectable. D_k goes to the neutral pool and W stays.
- **REJECT:** otherwise. W stays (rollback), D_k is quarantined, and attribution runs.

Attribution, recorded for every step and acted on at REJECT:
1. **Recipe vs data.** If P_recipe ≤ 0.25, the recipe itself degrades the model regardless of data (the 2026-09-10 warm-start finding). In that case the step is flagged `recipe`, not `data`.
2. **Classification vs localisation.** Δ12-class and Δclass-agnostic are computed for cand vs null.
   - The agnostic score holds while the 12-class score drops → `labels`.
   - Both drop → `domain/localisation`.
3. **Per-species deltas.** The species that moved.
4. **Label audit.** BioCLIP-2 (Step 1 verifier) re-reads D_k's boxes. The share of boxes it confidently contradicts estimates label noise.
   - **Tool:** `inc/audit.py`, submitted with `run_inc_audit.sh`: `python -m weed_optimizer_framework.tools.inc.audit --trusted T.jsonl --audit NAME=M.jsonl [NAME=M.jsonl ...] --out OUT.json`. It writes OUT.json, OUT.md and OUT_boxes.csv (one row per audited box).
     - `--out` must end in `.json`. No output may lie under `step1/` or be one of the audit's inputs, and an existing file is overwritten only when it is an earlier audit's output.
     - The job runs on GPU-shared and leaves its GPU idle: the allocation is a GPU allocation, and RM-shared submissions fail with "Invalid qos".
   - **Probe:** verify's probe, prototypes and joint thresholds (95% per-species recall, 5-fold CV grouped by capture session, else by source). It is fitted only on the trusted manifest's species boxes, with the labels of that manifest's own label files. No OtherPlant sample is added (it could hold the audited images), so this probe never predicts `OtherPlant`. A species the trusted boxes lack cannot be verified: audited boxes with that label are `label_unseen` and are not judged.
   - **Features:** the Step 1 crop embeddings (`step1/crops.csv` and `emb/`), refused unless `verify.check_fresh` passes. No image is opened and nothing is embedded.
   - **Box lookup:** by (image path, box index) in `crops.csv`; an image under another path is found by its sha256. A box is judged only when its crop row's geometry matches the label's within 1e-4. Any other box is reported as `no_embedding` with its reason and is never guessed: `geometry_mismatch`, `failed`, `small`, `no_crop_row`, `image_not_in_crops`, `image_changed`.
   - **Refusals:** an audited image that is also in the trusted manifest is refused, because the probe was fitted on its label. It is matched by the same path, the same sha256, or a Step 1 image of the same photograph. A cwd12 copy and its `train_core` twin (recorded in `cwd12_copies.jsonl`, within 6 dHash bits) count as the same photograph, in either direction. A label file that no longer hashes to its manifest's `label_sha256` stops the audit.
   - **Per audited manifest:**
     - verdicts from verify's verdict function;
     - the conflict rate among species-labelled boxes (the label-noise estimate), with a Wilson 95% interval;
     - the verified and unknown rates;
     - per species, per source, and the label → prediction pairs of the conflicts;
     - the rate at which OtherPlant boxes are confidently called a cwd12 species.
   - **Baseline:** held-out trusted folds, each judged by a verifier fitted (thresholds included) on the other folds.
     - With the true labels: the false-conflict rate f, overall and per species.
     - With one seeded wrong species per box: the conflict recall r, overall and per true species.
   - **Baseline for each audited set's species mix:** per-species f and r differ widely, and an increment's mix can be far from P0's. Each audited set is therefore compared with f and r weighted by its own judged species-labelled boxes per label (`baseline_for_mix`).
     - f_mix's 95% interval comes from seeded draws of each species' f from its Jeffreys posterior, so a species with few trusted boxes widens it.
     - A species without a baseline entry is listed, and its boxes are left out of the comparison.
     - `noise_estimate_corrected` = (c − f_mix)/(r_mix − f_mix), clipped to [0, 1]. It assumes in-domain labels and uniform swaps, so for harvested sources it is indicative only. The label mix stands in for the true species of the wrong labels, so for a set that is both skewed and noisy the estimate is approximate.
     - `above_baseline`: c's Wilson interval lies wholly above f_mix's interval.
     - The same two numbers against the baseline pooled over P0's mix are reported as context only.
   - **Known truth:** a box judged on a `train_core` crop carries its true label in `crops.csv`. For those boxes the true label error rate and the audit's own conflict recall and false-conflict rate are reported; for `Bswap` these are the planted swaps.
   - **Pilot, post hoc:** the driver does not run item 4 (exp.json `attribution_scope`). It is run once after pilot_v1:
     - trusted: `pilot_v1/manifests/P0.jsonl`;
     - audited: I1–I5, `Bswap` (its corrupted label copies under `pilot_v1/labels/Bswap/`) and `Breal`;
     - output: `pilot_v1/audit/label_audit.json`, with `.md` and `_boxes.csv` beside it.
     - I1–I5 and `Bswap` are `train_core` images, so each box is its own `core` crop. `Breal` is judged where verify's pool holds its images. A box that verify clipped to the frame, or whose index moved because verify dropped a degenerate box, is a `geometry_mismatch`.
   - **How it was verified:** `tests/test_inc_audit.py` builds a synthetic Step 1 world in verify's file formats, where each box's feature follows its true class.
     - A 40% swap copy made by `pilot.make_bswap` scores a conflict rate of 0.383 and a corrected estimate of 0.403. The planted share is 0.400, and its known-truth block finds exactly the planted boxes.
     - The same images with their own labels score 0.000, as does the baseline.
     - A harvested-style set with 25% wrong species scores a corrected estimate of 0.237.
     - A noisier world (CV top-1 0.845, f = 0.131, r = 0.939) exercises the f term. Two clean sets with conflict rates of 0.154 and 0.119 are estimated at 0.033 and 0.000. The swap copy (c = 0.438) is estimated at 0.381, where c/r would give 0.466.
     - A skewed world puts two confusable species at 26 of P0's 800 boxes but 219 of the 320 boxes of a clean set. Their held-out f is 0.071 and 0.167, against 0.004 pooled.
       - Against the pooled baseline, the clean set would be noisy: c = 0.072, an estimate of 0.073, and above the baseline.
       - Against its own mix (f_mix = 0.082 [0.032, 0.204], r_mix = 0.812) it is estimated at 0.000 and is not above the baseline.
       - A 40% swap copy of the skewed sessions is still above its baseline, estimated at 0.448.
     - Each overlap route is pinned on its own (path, sha256, a copy of a trusted image, the twin of a trusted copy). So are the `--out` refusals, which leave the Step 1 files untouched, determinism, a rerun over its own outputs, and a probe that does not depend on what is audited.
5. **Sources.** When D_k mixes several sources: leave-one-source-out cand runs, 3 seeds each.

### Protocol v2 (net flips)

**Evidence** (pilot_v2, full rehearsal, cluster, 2026-09-27). In the `full` chain, the clean increments I1, I2, I3 and I4 passed the regression and species guards. The flips guard alone rejected them, although P_data was 1.00 for each:

| Step | cand | null | inc | cand negative flips | null negative flips | excess | allowed (2·sd + 3) | truth arm (P) |
|---|---|---|---|---|---|---|---|---|
| I1 | 0.7316 ± 0.0038 | 0.7134 ± 0.0021 | 0.7282 | 35, 34, 31 | 27, 26, 26 | 7.00 | 4.15 | hurts (0.33) |
| I2 | 0.7535 ± 0.0057 | 0.7134 ± 0.0021 | 0.7282 | 40, 34, 35 | 27, 26, 26 | 10.00 | 4.15 | helps (1.00) |
| I3 | 0.7376 ± 0.0043 | 0.7134 ± 0.0021 | 0.7282 | 38, 34, 32 | 27, 26, 26 | 8.33 | 4.15 | helps (1.00) |
| I4 | 0.7283 ± 0.0028 | 0.7134 ± 0.0021 | 0.7282 | 35, 35, 33 | 27, 26, 26 | 8.00 | 4.15 | neutral (0.67) |

- Dev has 617 images; the incumbent gets 367 of them right.
- I2's cand mean is 0.7535 ± 0.0057, against 0.7134 for the null and 0.7282 for the incumbent. It is 0.025 above the incumbent and was rejected.
- The truth arm (cold union runs, with vs without D_k) says I2, I3 and I5 help, I4 is neutral, and I1, `Bswap` and `Breal` hurt. The `full` chain accepted only I5, whose excess was 0.33.
- Not every one of these rejections was wrong. For I1 the truth arm says hurts: P 0.33 alone would be neutral, but its species guard failed on PricklySida. v1's REJECT agreed with it. In the `full` chain, at P_data 1.00 with the regression and species guards passing, the flips guard was the only thing that stopped I1. v2 may accept such a step, and so lose a correct rejection. The pre-registered check below counts that against v2.
- Over-rejection by the flips guard was the case for I2 and I3, which the truth arm says help. For I4 (truth neutral), v1's REJECT and an ACCEPT would both disagree with the truth arm.
- The v1 guard records negative flips only. pilot_v2's positive and net counts are therefore not in its ledger, and they were not computed for this change.

**Principle:**
- The flips guard comes from Pioneer Agent. Its intent is "do not break more than you fix": an increment may not buy a mAP gain with images that stop working.
- Per-image correctness is coarse for detection. An image is correct only when every GT box is matched with the right class and there is no false positive at conf 0.25, so one box changes the bit. A model whose predictions move a lot changes many images in both directions.
- A rule that ignores the images a run fixes therefore penalises exactly the large improvements, because they change the most images.
- **The v2 rule.** Per run, net flips = negative flips − positive flips. A positive flip is an image incorrect under W and correct under the run. The threshold has the v1 form: mean(cand net) − mean(null net) ≤ 2·sd(null net) + 3 images.
  - The comparison is still against the null, so the recipe's own churn is not charged to the data.
  - Nothing else changes: P_data, the regression and species guards, their thresholds, the soup, attribution and the truth arm.
- **What v2 tests.** Per run, net flips = #(correct under W, incorrect under the run) − #(incorrect under W, correct under the run) = (W's correct images) − (the run's correct images), exactly. W's bits cancel:
  - mean(cand net) − mean(null net) = mean(null correct) − mean(cand correct), and sd(null net) = sd(null correct);
  - so v2 is a guard on the number of exactly-correct dev images, cand against null: cand may get at most 2·sd(null correct) + 3 fewer images right than the null. It no longer measures churn against W;
  - the per-run negative and positive counts the guard records are informational. They also give v1's verdict on the same runs (below).
  - Read this way, "do not break more than you fix" means that, net of what the recipe does without D_k, D_k may not lose exactly-correct images beyond the null's own noise.

**Pre-registration:**
- The rule is fixed on the principle above before pilot_v3 is built, and it is not tuned to pilot_v2. The multiplier (2 sd) and the slack (3 images) are v1's, and no pilot_v2 net count was looked at.
- It is validated prospectively on pilot_v3 and on the real loop. Both are built with `--gate-flips-mode net`, and their chain verdicts are compared with the truth arm, step by step (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts), as for v1.
- **The v1 counterfactual.** A net-mode decision records each run's negative flips. `gate.v1_counterfactual` rebuilds v1's flips guard from them, with the recorded thresholds, and combines it with the recorded P_data and regression and species guards, which do not depend on the mode. The result is exactly v1's verdict on the same incumbent, cands and nulls.
  - It is a per-step counterfactual, not a counterfactual chain. After a step where the two rules differ, the chain's incumbent is v2's, and later steps compare both rules on v2's runs.
- **The criterion**, computed by `inc.report` (report.json `v2_check`, the report's "Protocol v2 check" section) for each experiment decided on net flips. It pools every (chain, step) pair that has both a gate decision and a truth decision:
  - A2 = the pairs whose v2 verdict matches the truth arm; A1 = the pairs whose v1 counterfactual verdict matches it;
  - **refuted** if v2 ACCEPTed any step not marked clean (`Bswap`, `Breal`, the real loop's `UNVERIFIED`), whatever the truth arm says of it and whatever v1 would have done, or if A2 < A1;
  - otherwise **supported** if A2 > A1;
  - otherwise **inconclusive**: a tie, including an experiment on which the two rules never differ.
  - The outcome is provisional until the experiment is done. An experiment without a truth arm can only be refuted, by a non-clean acceptance.
- **What follows from it.** pilot_v3 is the first test. If it is refuted, v2 is retired and v1 stays the default. If it is supported, v2 is checked again on each real loop built with it, and one refuted real loop retires it. If it is inconclusive, the question stays open. v2 becomes the default only through a later, separately documented change that cites these outcomes. Nothing in the code switches the default automatically.
- v1 remains the default. `GateConfig().flips_mode` is `negative`. An exp.json without a `gate` block, and every build without the flag, is decided exactly as before.

**Where it lives:**
- `GateConfig.flips_mode`, `negative` (v1) or `net` (v2).
- exp.json `"gate": {"flips_mode": "net"}`, written by `inc.pilot build` and `inc.realloop build` with `--gate-flips-mode net`. The block may hold only `GateConfig` fields, and init validates it. `require_production` is refused, because `testing` sets it.
- The driver makes every `gate.decide`, `choose_soup` and `truth_detail` call of the experiment with that config.
- **The config is fixed at init.** For a definition with a gate block, init writes `state.json` `gate_pin`: the block and the whole resolved config, `flips_mode` spelled out, `negative` included. The first pass writes the same record to the ledger as `gate_pin/0`, after `code_pin/0`.
  - A definition without a block pins nothing and writes exactly the state and ledger it always did. Its pin is `GateConfig`'s defaults.
  - A pass whose exp.json resolves to another config refuses to run and writes nothing. That covers an edited, added or removed gate block and a changed `testing` flag. No operation changes the gate of a running experiment; a different gate means a new experiment.
  - `status` marks a chain whose config is not the defaults, e.g. `(gate flips net)` or `(gate flips_slack_images=50.0)`, and prints `GATE CONFIG CHANGED` when exp.json no longer matches the pin.
- In net mode, each decision's `config` records `flips_mode: "net"`. Its flips guard records, per run, the negative, positive and net counts, and the net statistic it tested.
- A v1 decision is byte for byte what it was before the option existed. Its `config` has no `flips_mode` key (absent = `negative`), and its flips guard has the v1 fields. For a new experiment built with a block, which is every `pilot build` and `realloop build`, `gate_pin` states the mode explicitly, so the ledger does not depend on the absence.
- The report's header shows the pinned config, never exp.json's. It checks that pin against the config every gate, soup, truth and `gate_pin` ledger entry recorded, and prints `GATE CONFIG MISMATCH` for any entry that differs. It checks it against exp.json too (`GATE CONFIG CHANGED`). In net mode the report also shows each step's v1 counterfactual verdict and the v2 check above.

**Building pilot_v3 and the real loop with v2.** Only a path that forwards `--gate-flips-mode net` builds a v2 experiment. That is the builders' CLIs, directly or through `run_inc_build.sh`, which forwards its arguments, and the autopilot (docs/INC_AUTOPILOT.md, D15, D16, L9):
- Lever L9 (from D15, which fires on pilot_v2) builds `pilot_v3` as `inc.pilot build --exp pilot_v3 --replay-mode full --gate-flips-mode net`.
- Every lever that rebuilds a pilot or a loop forwards the parent's pinned flips mode (`levers.gate_params`, `levers.loop_params`). A real loop built from a v2 pilot's D4 decision carries `--gate-flips-mode net`.
- The `inc.pilot build` and `inc.realloop build` rows of `tools/brain/policy_actions.json` take `gate_flips_mode` as an enum.
- D16 reads this experiment's `v2_check`. On `refuted`, no real loop is built on that gate.
- `inc_autopilot/thresholds.json` restates `GateConfig`'s default `p_accept`, `p_reject` and `p_recipe_flag`. A gate block can change them per experiment; D15 reads each decision's own recorded `config.p_accept`.
- Right after the build, check that exp.json holds `"gate": {"flips_mode": "net"}` and that `state.json` `gate_pin.config.flips_mode` is `net`. The cold base runs come first, so no gate decision has been taken yet. A wrong mode means a new build under a new `--exp` name, because the pin refuses any change; after that check, the pin holds.

**Existing experiments.** pilot_v1, pilot_v2, b0_v1 and base_b_v1 are finished, and they are not re-advanced or re-pinned.
- They were decided by, or pinned to, the v1 code. Their decisions stand as v1 decisions.
- They have no gate block, so their gate pin is the defaults, which is the config their decisions recorded.
- `gate.py` and `driver.py` changed, so the hashes pinned in their `state.json` no longer match the code. An advance of a done experiment returns before the code check.
- A new experiment pins the new hashes at init.

**How it was verified:**
- `tests/test_inc_gate.py`:
  - A battery of 11 v1 decisions (every verdict and guard path, with positive flips in cand and in null) hashes to the same sha256 of the full decision dicts as the gate before the option existed. This holds for `GateConfig()` and for an explicit `flips_mode="negative"`.
  - In net mode, a clearly better model with many positive flips passes, where v1 rejects it. A model that breaks more images than it fixes fails at P_data = 1.
  - The null's own positive flips count: cand net 0 against null net −4 fails, where v1 passes the same runs.
  - On every battery case, net mode leaves P_data, P_recipe, the regression and species guards and the attribution as v1 has them (blame follows the verdict). The one verdict it changes is the better model with many positive flips.
  - The recorded counts and config survive a JSON round trip. An unknown mode is refused, and `choose_soup` and the truth arm do not depend on the mode.
  - On every battery case, `v1_counterfactual` of the net decision, as an object or as its JSON, gives exactly v1's verdict and flips guard on the same runs. On a v1 decision it returns that decision's own.
  - On every battery case, each run's net count is W's correct images minus the run's, and the tested excess is mean(null correct) − mean(cand correct).
- `tests/test_inc_driver.py`:
  - The experiment has 2 chains and 3 steps, the truth arm, and planted per-image flips. Without a `gate` block, it writes no `gate_pin` to its state and ledger, and its gate decisions, truth details and soup records equal digests pinned from the driver and gate before the option existed.
  - With `"gate": {}` or `"gate": {"flips_mode": "negative"}` it writes byte for byte the same manifests, specs and runs (299 files) and makes the same submissions and decisions. Its `state.json` is the no-block state plus `gate_pin`, which spells out the v1 config with `flips_mode: "negative"`. Its ledger is the no-block ledger plus one `gate_pin/0` entry.
  - With `"gate": {"flips_mode": "net"}`, the step whose cands fix more images than they break (net −2 per run) is accepted, where v1 rejects it on flips alone. The step that breaks more than it fixes (net +5) is still rejected.
  - In net mode, every gate entry records the net config and all the counts, and the truth arm's details equal the v1 run's. `status` and the report header say which mode decided.
  - The report shows each net step's v1 counterfactual (A: v2 ACCEPT, v1 REJECT). In this synthetic world the truth arm is neutral on every step, so the v2 check is inconclusive at 0 against 0. Hand-built step rows cover the other outcomes: supported, refuted by agreement, refuted by a non-clean acceptance even when v2 agrees more and v1 would accept it too, a tie, no truth arm, and provisional before done.
  - A gate block's metric (`map50`) reaches the gate, soup and truth entries, which then carry the score files' map50 values. Every soup there scores above its cands' mean on map50 and below it on map50_95. Each is chosen as the new incumbent, so `choose_soup` itself received the block's metric.
  - Init refuses, writing nothing: an unknown key, `require_production`, a bad mode, a value of the wrong type or range (an integer too large for a float included), `p_reject` ≥ `p_accept`, `min_seeds` above the seeds, a block that is not an object, and a block on a baseline.
  - **The pin.** A chain without a block is driven to its first decision (a v1 REJECT). A gate block with net mode and slack 50 is then added to its exp.json.
    - The next advance refuses, names each changed field as [pinned, now], and writes nothing. The message is not a code-pin refusal, so the autopilot does not classify it as one.
    - `status` prints `GATE CONFIG CHANGED`. The report still shows the pinned v1 config and flags the change.
    - Removing `testing`, or changing one threshold, is refused too.
    - Restoring exp.json resumes the chain, and every decision is a v1 one at slack 3.
    - On a net experiment, removing the block, switching it to `negative` and relaxing its slack are each refused. A block that resolves to the pinned config runs.
    - The report lists every gate and `gate_pin` entry as a mismatch against a forged pin, and a soup entry decided on another metric.
  - `pilot build` writes the block into exp.json and `build_summary.json`: `negative` by default, `net` on request, through the API and the CLI. The CLI refuses the flag on `build-b0` and `build-baseline`.
- `tests/test_inc_realloop.py`: `realloop build` writes `negative` by default and `net` with `--gate-flips-mode net` (the driver then decides that chain on net flips). It refuses an unknown mode, writing nothing.

## Step 0 — measurement foundation

- **0.1 Splits:** `inc/splits.py` builds `train_core`, `dev`, `test`, `ood22`, `ood23` and `imageweeds`, plus the never-train index and `LOCK.json`.
- **0.2 Scorer:** `inc/scorer.py` and its lock check.
- **0.3 DINOv2 reference pool** (`dinov2_curator.py`):
  - The reference sample excludes holdout stems and every image within 6 bits of the holdout, dev or exams.
  - Per-slug sampling is seeded by a stable hash of the slug, not the registry index.
- **0.4 `mega_trainer.py`:**
  - The optimizer is passed explicitly, so `lr0` is honoured.
  - The merge output dir is cleared before merging.
  - An empty or incomplete holdout guard aborts the merge.
  - An image that cannot be hashed is skipped and counted, not trained unchecked.
  - Validation uses the INC dev split when present, not the sealed holdout.
  - `dataset_discovery.harvest_new_datasets` trims its final result to `max_new`.
- **0.5 LoRA:**
  - **Audit.** `lora_yolo.py` inserts `ConvLoRA` into `model.model` and then calls `model.train()`. Ultralytics' `Model.train` rebuilds the network from its yaml (`trainer.get_model(weights=self.model, cfg=self.model.yaml)`) and copies weights by key intersection. So the adapters are dropped, and the wrapped convs' pretrained weights (renamed `…conv.original_conv.weight`) are not transferred.
  - **Replacement.** `inc/lora.py` is a DetectionTrainer subclass. It:
    - injects adapters after `get_model`;
    - keeps base weights frozen through `_setup_train`, which otherwise re-enables `requires_grad`;
    - merges adapters into the conv weights on save.
  - **Tests:** only adapter and head parameters receive gradients; merged and unmerged outputs agree to within 1e-4; a merged checkpoint loads with plain `YOLO()`.
- **0.6 Noise calibration and B0:**
  - YOLO11n cold on `train_core`, 3 seeds (the union recipe).
  - Scored on dev, the exams, and test once.
  - This gives the seed noise band used everywhere and the cwd12-only baseline under the new splits.

**Acceptance for Step 0:**
- the manifests and `LOCK.json` exist and verify on the cluster;
- every new unit test passes locally and on the cluster login node;
- a 1-epoch smoke run goes train → scorer → `score.json` end to end;
- B0 has three seeds scored on all exams.

## Step 4 — pilot on known answers (runs before Steps 2–3 on real data)

- **Base P0:** about 50% of `train_core` images, by whole sessions, chosen deterministically. The largest sessions go to P0 first.
- **Clean increments I1–I5:** the remaining sessions packed into six bins of about 8% each: five clean increments plus the `Bswap` source.
  - Sessions are never split, so increments are "new field days" and never burst-neighbours of the base.
  - Bin sizes stay within ±35% of the target. The sizes are recorded.
- **Two planted bad increments**, inserted into the sequence:
  - `Bswap`: the sixth bin, held out of the clean increments. 40% of its boxes are reassigned to a random wrong species (seeded). The ground truth is that it is bad.
  - `Breal`: a real harvested source with a measured label precision below 0.90: `project_agml__weed_crop_detection` (0.737) plus `project_agml__imageweeds_aerial_weed_detection` (0.650).
    - Species join: Ragweed and Waterhemp → their cwd12 ids; every other class → `OtherPlant`.
    - Near-copies of any split image are removed.
    - Its truth is measured, not assumed.
- **Sequence:** P0 → I1 → I2 → Bswap → I3 → Breal → I4 → I5.
- **Recipes**, each with its own incumbent chain:
  - `full`: all layers;
  - `freeze`: backbone layers 0–10 frozen;
  - `lora`: r = 16 on every dense (groups = 1) 3×3 conv in the backbone and neck; Detect head trainable. On YOLO11n the only 3×3 conv left without an adapter is the grouped positional-encoding conv `model.10.m.0.attn.pe.conv`.
- **Truth arm:** for every step, cold union runs on P0 ∪ (clean increments so far) ∪ D_k vs without D_k, 3 seeds each.

Results reported:
- **Decisions:** each recipe's decision per step vs the truth arm's decision (the same gate rule applied to union runs).
- **Final quality:** the final incumbent of each recipe vs the union of all clean data, on dev, the exams, and test (read once).
- **Attribution:** whether `Bswap` is attributed to `labels`.
- **Cost:** GPU-hours per recipe.

**Acceptance:** every step decided for every recipe and the truth arm; results table and ledger written; decisions compared with the truth arm.

### Pilot v2 (full rehearsal)

**v1 finding** (pilot_v1, cluster, 2026-09-27):
- Every chain step had P_recipe = 0. The null runs fine-tuned the incumbent for 30 epochs at lr0 0.002 on R1 ∪ R2 = 2|D_k| = 476–548 images (|D_k| = 238–274), 31–36% of the 1,540-image accepted pool. They scored 0.703–0.721 on dev, against the incumbent's 0.7282.
- The regression, species and flips guards therefore rejected every increment. This happened although P_data was 1.0 for several clean increments and 0.0 for the planted bad one (`Bswap`).
- The truth arm (cold union with vs without D_k, 3 seeds) found that I2, I3 and I5 help, I4 is neutral, and I1, `Bswap` and `Breal` hurt.
- Conclusion: a fine-tune on a small replay subset forgets the rest of the pool. Ibrahim et al. 2024 (TMLR) found that replay with LR re-warming and re-decaying matches union training. TIME 2024 found plain replay to be the hard baseline to beat.

**The one change** is replay mode `full` (exp.json `"replay_mode": "full"`; `inc.pilot build --replay-mode full`):
- `cand` trains on the chain's whole accepted pool ∪ D_k.
- `null` trains on the whole accepted pool alone.
- The init (the incumbent), recipe, seeds, dev scoring, gate, soup, truth arm and final runs are unchanged. D_k is still checked disjoint from the pool.
- The default stays `sample`. A definition without the key (pilot_v1's) runs exactly as before.

**Consequences of the change** (recorded, not corrected):
- `null` now trains |D_k| fewer images per epoch than `cand`. The two arms run the same epochs, not the same iterations.
- Each run trains on the pool, about 1,540–3,000 images, instead of 476–548.
- Ultralytics' 100-iteration warmup floor now covers about 1.0–2.0 of the 30 epochs, instead of 5.6–6.7. This is recorded in exp.json `effective_warmup` and in the report's "Warmup as run".

**How it was verified:**
- `tests/test_inc_driver.py`, a full-mode pilot on FakeBackend:
  - `cand` = pool ∪ D_k and `null` = pool at every step;
  - accepted increments join the pool for later steps, and rejected or held ones do not;
  - with three chains planted to decide differently, each chain's `cand` and `null` are its own accepted pool (∪ D_k), with the expected pools computed from the planted verdicts;
  - the mode and sizes are recorded in the state and the ledger;
  - a definition without `replay_mode` writes byte for byte the same state, ledger, manifests and specs as one with `sample`;
  - in `sample` mode, the R1, R2, `cand` and `null` key lists, run ids and submissions of a fixed 2-chain, 3-step definition equal the values the driver produced before `replay_mode` existed (pinned digests).
- `tests/test_inc_integration.py`: pilot_v2's production `cand` and `null` specs pass inc/train.py's spec and protocol-recipe checks.

## Step 1 — high-precision base from harvested data

1. **Pool.** The v3.60.0 species-joined merge of the registry. It excludes `NEVER_TRAIN_SLUGS` and near-copies of the holdout, dev and exams. Boxes are read in cwd12 id space (slot → id through `cwd12_species`); all other names become `OtherPlant`.
2. **Box verifier** (`inc/verify.py`, labeler Phase B):
   - BioCLIP-2 embeddings of every box crop.
   - A linear probe trained on `train_core` crops (12 species).
   - Open-set rejection by calibrated probability and prototype-cosine thresholds, set by cross-validation on `train_core` crops at 95% per-species recall.
   - A box labelled with species k is **verified** if the probe says k confidently, and **conflict** if it confidently says j ≠ k. An `OtherPlant` box the probe confidently calls a cwd12 species is a **conflict**.
   - The verifier's precision is measured on known-truth sets:
     - the old wrong joins of cwd12 copies (true labels are known);
     - seeded label swaps on held-out `train_core` folds.
3. **Admission.** An image is admitted when every box is verified or is an `OtherPlant` box with no conflict. Conflict images go to a human verify-only queue (CSV plus crop sheet). They are never auto-trained, except in the one unverified-source test increment of Steps 2–3.
4. **Selection** (DINOv3 recipe: no single filter):
   - retrieval toward `train_core` prototypes (on-domain);
   - hierarchical k-means balanced sampling (coverage);
   - per-species caps, with pruning done per class (Sorscher et al. 2022).
   - Thresholds stay moderate (OWL-ST: 0.3 beats strict).
5. **Base B:** `train_core` plus the selected verified images. The remaining verified pool is the increment pool.
6. **Train B** cold, 3 seeds (`inc.pilot build-baseline --exp base_b_v1 --manifest INC_DIR/step1/base_B.jsonl`). Compare with B0 (Step 0.6) on dev, the exams, and test once.
   - The real loop (Steps 2–3, `inc.realloop build`) trains the same three runs as its base arm: same `base_B.jsonl` bytes, recipe, seeds and init. Its final runs read B's test too. The pinned driver cannot import base weights from another experiment, so the two experiments cannot share these runs.
   - **Either** skip `base_b_v1`: the loop's base arm and its final base row are B, and B is compared with B0 once, there. Step 1's acceptance item "B trained and compared with B0" is then met when the loop's final runs finish.
   - **Or** build `base_b_v1` when B vs B0 is needed before the loop is built (for example, to decide whether to run the loop on B). This costs three more cold 100-epoch runs, and the loop reads B's test a second time. B vs B0 is then reported from `base_b_v1`, and the loop's base row is stated as a second read of the same model (a rerun, not a new comparison). `inc.realloop build` lists every baseline experiment already built on the same bytes (`build_summary.json` `same_base_baselines`).
   - Both builders refuse, in production, a `base_B.jsonl` whose select build read another `train_core` or never-train index than `LOCK.json` and the index in place now (`pilot.select_provenance`).

**Acceptance:**
- verifier precision measured on both known-truth sets;
- base manifest written with per-source and per-species counts;
- a relevance criterion for the increments in place: a `relevance.json` for the select build that Steps 2–3 draw from that passes its calibration check (below, "Source relevance"), or the source-level species evidence criterion (Steps 2–3), adopted at the R4 review of 2026-09-27 after that check failed on the real pool;
- B trained with 3 seeds and compared with B0.

### Source relevance (the increment pool) — `inc/relevance.py`

Run after 5 and before Steps 2–3: `sbatch run_inc_relevance.sh build` (docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Source relevance").

- **Why.** The verifier certifies species labels, not relevance. An `OtherPlant` box is admitted unless the probe confidently calls it a cwd12 species, so an `OtherPlant` box on a non-plant image is admitted too.
  - The increment pool is almost all `OtherPlant`: 96,088 images, about 624k boxes, next to base B's 3,927 images (`train_core` 3,049 + 878 harvested).
  - It holds whole sources that are not weed or crop photographs, next to about 35 that are. Examples: `fvossel__csgo_player_detection` (4,012 images of a video game), `kg_farukalam__tomato-leaf-diseases-detection-computer-vision` (434), `rf_bishwarup-halder__crop-health-advisor` (15,116).
  - A hand-written exclusion list is not used (project rule: no human-curated source lists). The filter is automatic, calibrated on `train_core`, and recorded.
- **Score.** Zero-shot BioCLIP-2 with its own text tower (open_clip, the same `hf-hub:imageomics/bioclip-2` model that embedded the crops).
  - The tokenizer must be the one the model's open_clip config names: the config names no `hf_tokenizer_name`, so open_clip's `SimpleTokenizer` with the model's context length. open_clip falls back to a generic Hugging Face tokenizer when it cannot read the hub config, and for BioCLIP-2 that fallback loads and emits ids that fit the model, so the tokens alone cannot show it.
  - The config and weights files must resolve from one Hugging Face cache snapshot, through open_clip's own resolver (open_clip otherwise builds a randomly initialised model with only a logged warning). The snapshot commit and the files' sha256 are recorded. The embedding shards name the model but not its snapshot, so the two stages' weights are not compared.
  - Text and image features are L2-normalised; a logit is the model's logit scale × the cosine.
  - A crop's P(plant) is the softmax over the plant and non-plant prompts, summed over the plant prompts.
  - The prompts live in one constant, `relevance.PROMPTS`, and are recorded in the output. Plant: 'a photo of a plant', 'a photo of a weed', 'a photo of a crop plant', 'a photo of a seedling', 'Plantae'. Non-plant: 'a photo of a person', 'a screenshot of a video game', 'a photo of a vehicle', 'a photo of an animal', 'a photo of an insect', 'a photo of bare soil', 'a photo of a building', 'text'.
  - Image features are the Step 1 crop embeddings (`step1/emb`). No image is opened and nothing is embedded again.
- **Calibration.** tau = the 5th percentile of P(plant) over every `train_core` crop with a feature. These crops are all genuine weeds, and 95% of them score at or above tau. Weeds from other cameras or domains (UAV, other fields) can score lower: that is the filter's false-negative risk (see the name check below). The distribution (percentiles and a histogram) is recorded.
- **Calibration check (fails closed).** tau must be ≥ 0.5, that is, at least 95% of `train_core` crops must be judged more plant than non-plant.
  - Why: at BioCLIP-2's logit scale (about 100), P(plant) sits near 0 or 1. If more than 5% of `train_core` crops read as a non-plant prompt (for example, seedling crops as 'a photo of bare soil'), tau falls towards 0 and nearly every source passes.
  - When the check fails, the build still writes `relevance.json` and `relevance.md` (flagged) for a person to read, then exits with an error. A rerun does the same. `relevance.load` refuses the file, so no draw uses it and `realloop build` refuses.
- **Result on the real pool (cluster build, 2026-09-27): the calibration check failed.**
  - tau = 0.0557 < 0.5. 34.5% of the genuine `train_core` weed crops get P(plant) < 0.5: on seedling crops a non-plant prompt wins (most likely 'a photo of bare soil').
  - Many genuine weed sources have a median P(plant) of 0.03–0.10. That is the range of the video-game source `fvossel__csgo_player_detection` (0.049), so at this calibration the score cannot separate them.
  - `relevance.json` is refused by design, so no production real loop can be built with this criterion.
  - The prompts are not tuned after seeing this: prompts chosen on this pool would be fitted to the data they filter. A second criterion that rests on the verifier instead was adopted at the R4 review (Steps 2–3, "Source-level species evidence"). The relevance build, its check and its refusal stay as they are.
- **Pass rule.** Per source of `increment_pool.jsonl`:
  - a seeded sample of up to 300 crops with a feature (`--sample`, `--seed`; seed text `inc.relevance/<seed>/<table>/<source>`);
  - the source **passes** when the sample's median P(plant) ≥ tau, that is, when at least half of its crops score at or above the level that 95% of `train_core` crops reach;
  - a source with fewer than 20 usable crops is `insufficient` and does not pass: a source is not cleared on no evidence.
- **Reported only.** A third prompt set (leaf disease and leaf close-up) is kept out of the softmax, so it cannot move P(plant). Per source, the share of crops whose top prompt over every prompt is one of them is reported. It does not change pass/fail: a diseased leaf is a plant, and there is no calibrated rule yet for plant imagery that is off the weed-detection task.
- **Reported only: the name check.** The non-passing sources whose name holds a plant word (`relevance.PLANT_NAME_WORDS`: weed, crop, plant, seed, leaf, agri, farm, field) are listed in `relevance.md` for a person, as possible false negatives. The word list never changes a status and names no source.
- **base_selected.** Its sources are scored the same way, on its own images, for the record only. Base B is not rebuilt.
- **Outputs.** `step1/relevance.json` and `relevance.md`:
  - the prompts, the text encoder and its logit scale, and the model files (snapshot commit, config and weights sha256, tokenizer class);
  - tau, the calibration check and the `train_core` distribution;
  - per source: images in the increment pool, usable crops, the sample size, the median, the share ≥ tau, the top-prompt shares and the status;
  - the non-passing sources with their images, and the name check;
  - the input hashes: the select build's summary, `increment_pool.jsonl`, `base_selected.jsonl`, `crops.csv` and every embedding shard.
- **Use.** `select increments --relevance` and `realloop build` (Steps 2–3, `--increment-sources relevance`, the default) draw the verified and `OTHER_HEAVY` increments only from sources that pass. Before use, `relevance.load` refuses:
  - a file made for another increment pool or `base_selected`;
  - a file made under another rule (the minimum crop count, the percentile or the check's 0.5);
  - a file whose calibration check failed;
  - a file in which any status, or the check, does not follow from its own recorded numbers;
  - a malformed file.
  - These are consistency checks, not a seal: the file's sha256, recorded in every draw, identifies the exact file used.
- **How it was verified.** `tests/test_inc_relevance.py`, with fake text and image embeddings where plant crops align with the plant prompts and the text tower is injected:
  - P(plant) and tau are recomputed independently of the module; 95% of `train_core` crops pass tau, and tau ≥ 0.5;
  - a video-game source and a 40%-plant source fail; plant sources and a 60%-plant source pass; sources with 18 crops, with 15 usable of 25, or with none are `insufficient`; a source with exactly 20 passes;
  - the name check lists exactly the non-passing sources with a plant word in the name; `relevance.md` prints tau and the medians in one format that keeps values near 0 and 1 apart;
  - a soil prompt that also matches the `train_core` crops puts tau below 0.5: the build writes the flagged file, exits with an error (and again on a rerun), and `load` refuses the file;
  - the seeded sample is recomputed from the rule;
  - an encoder that moves only the leaf-disease prompts leaves every P(plant), tau and status unchanged;
  - determinism, a no-op rerun, `--force`, and refusals before anything is written;
  - `load` refuses the edits above, and a malformed file with its own error;
  - the text tower's pinning, against a stand-in open_clip module (a tiny torch model): the fallback tokenizer, a wrong context length, unresolvable weights and a config and weights from two snapshots are refused; the snapshot commit and the sha256 values are recorded.
  - `tests/test_inc_select.py` and `tests/test_inc_realloop.py` run it on select's real build (fake text towers at logit scale 100 whose calibration check holds) and check the filtered draws (Steps 2–3).

## Steps 2–3 — incremental loop on real data

- **Setup:** base B, with the recipe that tracked the truth arm best in the pilot.
- **Increments:** fixed size M = 10% of B's images (under the source-level species evidence criterion, sized by rule when its pool cannot hold that: below, "Sizing"). They come from:
  - N verified increments (default 6), drawn cluster-balanced from the verified increment pool by `inc/select.py increments`: disjoint, whole near-duplicate groups, never-train guard;
  - one increment of `OtherPlant`-heavy verified data (`select increments --other-heavy`);
  - one unverified-source increment (rule below).
- **Relevance criterion.** The verified and `OTHER_HEAVY` increments are drawn only from relevant sources. There are two criteria; a build uses one (`realloop build --increment-sources`, `select increments --sources`). An evidence build records its mode in `exp.json` and `build_summary.json` (`step1.increment_sources`). A relevance build records no such key, and a missing key means relevance: its definition is the one built before the second criterion existed, so an experiment built then can still be re-inited.
- **Relevance filter** (`--increment-sources relevance`, the default; Step 1, "Source relevance"):
  - The verified and `OTHER_HEAVY` increments are drawn only from sources that pass `step1/relevance.json`. Every increment-pool image of a failing or `insufficient` source leaves the pool before the draw.
  - A production build refuses without the file. A testing build may draw without it, and records that. Any build refuses a file whose calibration check failed.
  - Recorded: the file's sha256 in `exp.json` `step1.relevance` and in select's draw parameters; the excluded sources, their images, and the near-duplicate groups the exclusion split, in `build_summary.json` `relevance`.
  - The unverified-source increment is **not** filtered: it is the planted, realistic bad increment. Its source's relevance status is recorded (`unverified.source_relevance`).
  - On the real pool its calibration check failed (Step 1, "Source relevance", "Result"), so this criterion cannot build a production loop today.
- **Source-level species evidence** (`--increment-sources evidence`). Adopted at the R4 review of 2026-09-27, after the relevance build failed its calibration check. It is a second criterion; the first is unchanged.
  - **Rule.** A pool source is eligible for the verified and `OTHER_HEAVY` increments only if Step 1's verifier judged at least `--min-evidence` (default 1) of its cwd12-species boxes **verified** (`admit_summary.json` `per_slug[source].boxes.verified`). Every increment-pool image of any other source leaves the pool before the draw.
  - **Why this measures relevance.** Two mechanisms exclude a source, and they act on different sources:
    - **The species join.** A source whose label names join to no cwd12 species holds only `OtherPlant` boxes, so it has no box that could be verified. It is excluded before the verifier judges anything. On the Step 1 of 2026-09-27 this is 32 of the 38 increment-pool sources (`pool_summary.json` `per_slug` boxes). They include the video-game source `fvossel__csgo_player_detection` and the leaf-disease source `kg_farukalam__tomato-leaf-diseases-detection-computer-vision`. They also include weed datasets whose label names do not join, such as `project_agml__mh_weed16_weed_detection` and `francesco__grass_weeds`. The conflict boxes of such sources (1,576 in mh_weed16, 1,607 in `rf_tuf__weed-3434e`) are `OtherPlant` boxes that the probe called a cwd12 species, not species labels that the verifier rejected.
    - **The verifier.** Among sources whose labels do join to cwd12 species, the verifier is what turns "labelled as a cwd12 species" into evidence. It certifies species at 99.96–100% precision on known truth (`step1/calibration.json`: 1.0 on the cwd12 copies under the current join, 0.9996 on the old wrong joins, 0.9997 on seeded label swaps). So a source that holds verified cwd12-species boxes is a weed dataset by evidence. On the real pool 6 sources hold species-labelled boxes. The verifier verified boxes in 5 of them. It verified none of the 393 `Ragweed` boxes of `rf_weed-project-zbfhf__weed-detection-aiml-35-weed-and-crop` (63 judged unknown), so that source is excluded.
  - **It is conservative.** It excludes genuine weed datasets whose species are not cwd12 species: their boxes are all `OtherPlant`, so none can be verified. It also excludes a weed source whose cwd12 boxes were all judged unknown or conflict. It can only cost coverage; it admits no source without evidence. It says a source is a weed dataset, not that its labels are clean: per image, admission still requires every species box verified. Two of the evidenced sources on the real pool (below) are the pilot's `Breal` sources, with measured label precision 0.737 and 0.650 (Step 4); only their admitted images are drawn.
  - **Checks** (the build refuses, before anything is written):
    - `admit_summary.json` must name the `verified.jsonl` and `crops.csv` the select build read;
    - where `select_summary.json` records `retrieval.source_evidence`, the two files must agree. A verified box is a species crop with features that the verifier did not call a conflict, which is what that record counts per source. So every source needs `species_crops` ≥ its verified boxes, and every source it counts needs a `per_slug` entry;
    - every increment-pool source needs a `per_slug` entry;
    - `--min-evidence` ≥ 1, and no `--relevance` with this criterion;
    - capacity: the evidenced pool must hold (N + 1) × M images, M of them in `OtherPlant`-heavy near-duplicate groups (N verified increments plus `OTHER_HEAVY`).
  - A production build with this criterion needs no `relevance.json`.
  - **Recorded:** the mode, `--min-evidence` and the sha256 of `admit_summary.json` in select's draw parameters. In `build_summary.json` `evidence`: the evidenced sources with their verified boxes and increment-pool images, the excluded sources with their images and verified boxes, the cross-check, the near-duplicate groups the exclusion split, and the capacity.
  - The unverified-source increment is **not** filtered and is unchanged. Its source's verified boxes are recorded (`unverified.source_evidence`).
  - **On the Step 1 of 2026-09-27** (computed from its `admit_summary.json` and `select_summary.json`; the two files agree):
    - 5 of the 38 increment-pool sources hold a verified box. Verified boxes / increment-pool images: `project_agml__greenhouse_crop_weed_detection` 461 / 13, `project_agml__imageweeds_aerial_weed_detection` 253 / 113, `project_agml__weed_crop_detection` 394 / 206, `rf_karthikeya-c8pvy__weed-detection-cwp10` 647 / 755, `rf_zig-zag-lnodr__weed-detection-vanpe` 294 / 352.
    - Their increment-pool images total 1,439 of 96,088.
    - At the protocol's M = 393 (10% of base B's 3,927 images), N = 6 needs 7 × 393 = 2,751 images, so the default build refuses on capacity. At M = 393 the image count allows N ≤ 2, that is 4 decided increments, below the acceptance's 6. Six decided increments (N = 4) need M ≤ 287 (7.3% of B).
    - The `OtherPlant`-heavy share of the evidenced pool needs `select_clusters.csv` and is not in these summaries.
    - **What the evidenced pool is made of** (`select_summary.json` `sources`, `calibration.json`). It is mostly not data from new sources:
      - 4 of the 5 evidenced sources are the sources of base B's harvested part. B's 878 selected images are `project_agml__imageweeds_aerial_weed_detection` 17, `project_agml__weed_crop_detection` 49, `rf_karthikeya-c8pvy__weed-detection-cwp10` 544 and `rf_zig-zag-lnodr__weed-detection-vanpe` 268. `project_agml__greenhouse_crop_weed_detection` has none.
      - 528 of the 1,439 images (37%) are below select's retrieval gate (GATE = 0.10; `inc/select.py`: such an image "is off-domain and is not selected"). That is all of greenhouse (13), imageweeds_aerial (113) and weed_crop_detection (206), plus 143 of cwp10's 755 and 53 of vanpe's 352. imageweeds_aerial and weed_crop_detection are also the pilot's `Breal` sources.
      - cwp10 and vanpe hold 1,107 of the 1,439 images (77%). Step 1 found cwd12 copies in both (`calibration.json` `cwd12_copies.copies_per_slug`: 200 and 84 images, dropped from the pool; `current_join.per_slug`: 44 and 6 judged copy boxes).
      - So an evidence-mode loop tests adding more data from B's own sources, mostly the rest of two sources that also hold cwd12 copies and the images select's gate rejected. It does not test adding data from new sources. R4 must weigh this together with N and M. The sizing rule below does not change it: the sized loop draws from the same pool.
  - **Sizing (R4 decision, strong-brain review of 2026-09-27).** The builder only refuses and reports the numbers. When the evidenced pool cannot hold the default N and M, the autopilot sizes the loop by rule instead of escalating it (docs/INC_AUTOPILOT.md, D2, "Sizing"; `inc_autopilot/levers.py` `evidence_sizing`):
    - N = the smallest number of verified increments with which the loop decides at least 6 increments (the acceptance below) together with `UNVERIFIED` and `OTHER_HEAVY`. The sequence has N + 2 steps, so N = 4.
    - M = min(10% of B's images, rounded; floor(evidenced pool images / (N + 1))).
    - If that M is below 5% of B's images, the review judged the increments too small for the gate to detect an effect: the loop is not sized, and N and M stay a review decision. The same holds when the capacity cannot be computed.
    - The same floor holds for an evidence loop whose N and M are given instead of sized (a brain plan's, or a rebuild's): at least 6 decided increments, and M at least 5% of B's images. The autopilot does not propose or file one below it (`levers.r4_floor`); a person may still build it by hand.
    - Only the image count is sized. The `OtherPlant`-heavy part of the capacity check (M of the images in `OtherPlant`-heavy near-duplicate groups) needs `select_clusters.csv` and stays the builder's check. If it refuses, the autopilot escalates to a person and does not submit the identical build again (docs/INC_AUTOPILOT.md, "Builder refusal → prerequisite").
    - **Why.** The acceptance needs at least 6 decided increments. On this pool the default loop (6 verified increments of 393) does not fit, and at M = 393 only 4 increments could be decided. The rule keeps the 6 decided increments and gives up increment size instead, down to half the protocol's 10%.
    - **On the Step 1 of 2026-09-27:** N = 4, M = min(393, floor(1,439 / 5)) = 287 images, 7.3% of B (the floor is 196.35 images). The loop decides 6 increments of 287 images: V1, V2, UNVERIFIED, V3, OTHER_HEAVY, V4. The four verified increments and `OTHER_HEAVY` need 5 × 287 = 1,435 of the 1,439 evidenced images, so the draw takes nearly all of them. The capacity is necessary, not sufficient (`select.pool_capacity`: the draw keeps near-duplicate groups whole), so with 4 images to spare the draw can still fall short at build time. The builder then refuses, the autopilot escalates, and a person decides N and M; the same build is not submitted twice. The build is `realloop build ... --increment-sources evidence --size 287 --n-verified 4`.
    - Whether the gate resolves effects at 287 images is not known before the loop runs; its gate entries show it (P_data inside (0.25, 0.75) on most entries would say not).
- **Unverified-source increment:**
  - **Candidates.** Images that verify did not admit (image verdict conflict or unknown), from one harvested source.
  - **Checked.** Every candidate must clear the never-train guard, come from no never-train dataset, and share no key, path or image bytes with an image of B or of a verified increment. Any failure, or a missing dHash, stops the build: verify's pool holds no exact duplicates and no `train_core` copies, so a hit means the Step 1 files are stale.
  - **Eligible.** A candidate is eligible unless it lies within NEAR_DUP_BITS (3) dHash bits of an image of B or of a verified increment.
  - **Source.** The source with the most eligible images, among sources with at least M; ties go to the lower name.
  - **Draw.** M images, seeded by the experiment name, with the labels the species join gave them.
  - **Recorded:** the source and its verdict mix.
  - **Where it is trained.** It is the one place where images that were not admitted are trained on: this increment's cand runs, and the truth arm's run with it. It never enters B, T_k or a verified increment. A chain that accepts it keeps it in that chain's accepted pool, which is the failure this step tests.
- **Sequence** (N = 6): V1, V2, UNVERIFIED, V3, OTHER_HEAVY, V4, V5, V6. For other N: the first two verified increments, UNVERIFIED, the third, OTHER_HEAVY, then the rest (N = 4, the sized evidence loop: V1, V2, UNVERIFIED, V3, OTHER_HEAVY, V4).
  - Clean = every verified increment, `OTHER_HEAVY` included.
- **Per step:** the gate from above, with attribution including leave-one-source-out.
  - As built, the driver runs attribution items 1–3.
  - Item 4's evidence is Step 1 itself: every species box of a verified increment is verified, and the unverified increment's verdict mix is recorded.
  - Item 5 (leave-one-source-out runs) is not run. Every gate entry says so and lists the increment's sources.
- **Reference: the truth arm**, as in the pilot. For every step it runs cold union runs on T_{k-1} ∪ D_k and on T_{k-1}, 3 seeds each; T_k grows only by clean increments.
  - It supersedes the earlier plan of a cold union anchor on B ∪ accepted increments at every third step, because it measures every step.
  - T_final (B ∪ every verified increment) against each chain's final incumbent measures how far the cheap chain drifts from joint training (YOLO LwF, Ibrahim et al. 2024).
- **Test:** read at the end only.

**Acceptance:**
- at least 6 increments decided;
- the truth arm decided at every step;
- final comparison of each chain vs T_final vs B vs B0 on dev, the exams, and test;
- ledger and results tables written.

**Builders:**
- `inc.pilot build-baseline` builds B's own cold baseline (`base_b_v1`).
- `inc.realloop build` builds the chain experiment (docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Baseline build" and "Real-loop build").
- `tests/test_inc_realloop.py` checks both:
  - on a synthetic Step 1 world written in verify's formats, with select's real build on it;
  - sizes, disjointness, the unverified source rule and draw, near-copies of verified increments and of B kept out of the unverified increment, clean flags, the sequence and the recipes;
  - refusals set up so that only one check can catch each (stale Step 1 files, bad base or increment manifests, copies among the non-admitted images, a base selected under an earlier lock);
  - driven to done on FakeBackend, with the report's attribution of the unverified increment;
  - the relevance filter: a production build refuses without `relevance.json` and finds the one next to the base; the filtered verified and `OTHER_HEAVY` draws hold no image of a non-passing source (the unfiltered draw does) and are select's filtered draw byte for byte; the unverified increment is not filtered, and its source's relevance status is recorded, also when that source fails relevance; a file with a failed calibration check is refused;
  - source-level species evidence: a production build with `--increment-sources evidence` builds without `relevance.json`; its verified and `OTHER_HEAVY` draws hold only images of evidenced sources (the default draw holds images of the excluded source) and are select's evidence draw byte for byte; the unverified increment is the rule's draw, recomputed in the test, from a source that is not evidenced; refusals when the two files disagree, on too few evidenced images or `OtherPlant`-heavy images, and on misused flags; an explicit `--increment-sources relevance` gives the default build's manifests byte for byte, and the same increments summary, build summary and definition apart from timestamps;
  - the default build (no `--increment-sources`) is the one built before that option existed: a digest of its `exp.json`, `build_summary.json` and every manifest (paths, sha256s and timestamps left out) equals the one recorded with `realloop.py` and `select.py` as they were then, under Python 3.10 and 3.12 alike.
- `tests/test_inc_select.py` checks select's side: the default draw's digest on a hand-written base (no k-means) equals the one recorded before `--sources` existed; on a world with a non-plant source whose boxes are all `OtherPlant`, `--sources evidence` draws none of its images (the default draw does), records the evidence, applies `--min-evidence` as "at least" (a source with exactly that many verified boxes stays and is drawn from; a `--min-evidence` one higher excludes it), and refuses when `admit_summary.json` and `select_summary.json` disagree.

## Running it

- **Driver:** `python -m weed_optimizer_framework.tools.inc.driver advance --exp <name>`.
- **Idempotent** and run on the login node. Each call:
  1. reads `results/framework/inc/<exp>/state.json`;
  2. collects finished runs (`sacct` plus `score.json`);
  3. applies the gate;
  4. submits the next sbatch array (`run_inc_train.sh`, GPU-shared, 1 × V100-32).
- **Ledger:** every decision is appended to `ledger.jsonl` with its inputs. Decisions are made by `gate.py`, not by a person or an LLM.

## Protocol v3 (splits v2 and the continuous loop)

Protocol v3 is the protocol of the continuous loop (docs/CONTINUOUS_LOOP.md). It is pre-registered here before any v3 run, under the owner's decisions of 2026-09-28 (docs/CONTINUOUS_LOOP.md §2.6: L-3, L-4, L-6). Its code is the package `tools/inc2/`: `recipes.py`, `train.py`, `gate3.py`, `scorer_sidecar.py`, `baseline.py` and `pilot4.py` for this section, with `common.py`, `guard.py` and `splits.py` for splits v2. The pinned v1 modules (`inc/driver.py`, `inc/gate.py`, `inc/scorer.py` and the rest) are not edited. Everything in "Fixed conventions", "Scorer" and "Gate" above still holds unless this section replaces it.

### What changes, and what does not

| Item | Protocol v3 |
|---|---|
| Splits | Splits v2 (docs/CONTINUOUS_LOOP.md §4). The training base is `base_v2` = train_core + tsw22 + tsw23 + the kept part of base B. The evaluation splits are dev, test and imageweeds, byte copies of v1's. ood22 and ood23 are training data (tsw22, tsw23) and are never an exam: a v3 spec that lists them is refused. |
| Scorer | The v1 scorer, unchanged: the v1 LOCK's sha256, the v1 exams, imgsz 640 for every exam and every arm. |
| Never-train guard | The v2 index (dev + test + imageweeds, at 6 dHash bits) is checked against every training image and against each of its 8 flips and rotations, fail closed. `inc2.guard.GuardV2` decides; `inc2.train` also looks every variant up in the v2 index itself, so a hit refuses the run even if GuardV2 misses it. The images L-5 drops from base_v2 (`splits/v2/l5_excluded.jsonl`, its sha256 in LOCK v2) are refused by their bytes in every v2 training run, whatever key or source lists them. So are the train_core rows L-8 drops (`splits/v2/train_core_variant_drops.jsonl`, its sha256 in LOCK v2: within 6 bits of an evaluation image under a flip or rotation). |
| Gate | The pinned `inc/gate.py` decides every chain step, with protocol v2's net flips. Protocol v3 replaces one number, the species guard's tolerance (below). Every other guard, threshold and verdict rule is unchanged. |
| Recipes | The table below. Freeze and LoRA are out of stream version 1 (L-6). |
| Detector | An arm (detector and imgsz) is pinned per experiment (L-4). `n640` (YOLO11n at 640 px) is the continuity arm. |

### The recipe table

Every recipe is SGD, batch 32, momentum 0.937, weight decay 0.0005, cosine to lrf 0.01, close_mosaic 10, deterministic, trainer `full`, at the arm's imgsz. An incremental recipe's `warmup_bias_lr` equals its lr0, which is the runner's convention for incremental runs; without it the bias LR would start at Ultralytics' 0.1.

| Name | Used for | Epochs | lr0 | Warmup epochs | warmup_bias_lr |
|---|---|---|---|---|---|
| `cold` | base and union runs | 100 | 0.01 | 3 | 0.1 |
| `r0` | cand and null runs: the current full rehearsal | 30 | 0.002 | 1 | 0.002 |
| `x1a` | cand and null runs: LR re-warm | 30 | 0.005 | 3 | 0.005 |
| `x1b` | cand and null runs: LR re-warm | 50 | 0.01 | 3 | 0.01 |

- For the n640 arm, `cold` and `r0` are exactly `inc.pilot`'s cold and full recipes.
- A production run whose recipe departs from the table in any key but the seed, `cache` and `workers` is refused (`inc2.recipes.deviations`). That is stricter than v1's check, which left momentum, weight decay and close_mosaic unchecked.
- **Freeze and LoRA** are refused by the survival rule below, applied to pilot_v3's recorded chains: each agreed with the truth arm on 3 of 7 steps, against the 5 required. A later pilot that passes the rule can re-admit them, under a new stream version.

### The arms (L-4)

| Arm | Checkpoint | Training imgsz | GFLOPs of one forward pass at nc 13 |
|---|---|---|---|
| `n640` | yolo11n.pt | 640 | 6.454 |
| `s640` | yolo11s.pt | 640 | 21.574 |
| `m640` | yolo11m.pt | 640 | 68.240 |

- **Where the FLOPs come from.** `ultralytics.utils.torch_utils.get_flops` (thop) on `yolo11{n,s,m}.yaml` at 13 classes, Ultralytics 8.4.22, on the laptop.
- **How an arm is pinned.** A build records the arm in exp.json with the sha256 of its checkpoint in `$REPO`. Nothing is downloaded. The executor refuses a production cold run whose init is not that checkpoint.
- **Scoring.** Every arm trains and scores at 640 px (the locked scorer infers at 640 px), so the grid varies capacity only. A resolution arm needs a new scorer version (R4) and is not in this grid.
- **The switch rule (dev only).** Each arm's 3 cold seeds are compared on the seeds all arms share (0, 1, 2), by dev mAP50-95. An arm qualifies when mean(arm) − mean(n640) > 2 × pooled sd, with pooled sd = √((sd(arm)² + sd(n640)²) / 2). The stream takes the qualifying arm with the highest dev mean, the one with fewer FLOPs on a tie; if no arm qualifies it stays on n640.
- **The truth arm.** When one r0 step of the chosen arm with its truth arm costs more than 25 GPU-h, the truth arm runs on every ⌈cost / 25⌉-th step. The cost uses the arm's measured cold rate (its base runs' `train_seconds` per image-epoch), and the incremental rate is taken as cold × 7.4 / 6.5 (the n640 ratio, est.).
- **Test.** Test is read once per arm at R0, as a milestone read, in a report kept apart from the decision.
- **Cost before a run (est.).** The measured n640 rates (cold 6.0–7.0 ms per image-epoch, incremental 7.4, scoring 54.5 ms per exam image; docs/CONTINUOUS_LOOP.md §5.6) are bracketed for the other arms. The low end is scaled by the pixel ratio, since the data loader bounds a larger model from below. The high end is scaled by the FLOPs ratio, the contract's basis; YOLO11n is loader-bound on a V100, so this over-states a larger model's time.

### The species guard (L-3)

For each of the 12 species s: **tol_s = max(0.03, 1.96 × SE_s)**. The species guard passes s when mean(cand AP_s) − mean(null AP_s) ≥ −tol_s. That tolerance replaces the pinned max(0.03, 3 × sd_s(null)).

- **What SE_s is.** The image-bootstrap standard error of the incumbent's dev AP for s:
  - 1,000 resamples of the 617 dev images, drawn once for all species with `numpy.random.default_rng(stable_int("inc2/v3/species_se"))`;
  - an image drawn k times counts k times, with its detections and its GT boxes;
  - AP50-95 is computed with Ultralytics' `ap_per_class`, the scorer's own AP function;
  - SE_s is the sample sd (ddof 1) over the resamples that hold a GT box of s.
- **Where it comes from.** The locked scorer records per-class AP over the whole exam, but not the per-image inputs behind it. `inc2/scorer_sidecar.py` therefore calls the pinned scorer unchanged, as a library, with a validator subclass that keeps each image's metric inputs. It checks that these reproduce the score's per_class exactly, and records SE_s beside the run's dev score (`scores/dev.sidecar.json`). Every base, union, cand and soup run of a v2 experiment gets one.
- **What does not change.** P_data, P_recipe, the regression guard, the flips guard, their thresholds and the verdict rule are the pinned gate's, with the experiment's pinned config.
- **How a step is re-decided.** `inc2.gate3.decide` reads the pinned driver's ledger entry of the step. It re-derives the recorded decision from the recorded score files, each checked by sha256, and refuses a decision the pinned gate does not reproduce. It then replaces only the species tolerance. The truth arm's species guard is replaced the same way (`truth3`), with SE_s from the "without" arm's first run.
- **When it cannot apply.** A step whose incumbent has no usable sidecar, or whose recorded inputs do not verify, is recorded as unavailable. Its commit verdict never accepts: a pinned ACCEPT commits as HOLD (the v3 tolerance can be stricter than the pinned one, so an ACCEPT on the rule that was not applied is not an ACCEPT of Protocol v3), and a pinned REJECT or HOLD stands. A truth entry that cannot be re-decided keeps its pinned verdict.
- **Why.** The pinned guard scales one species' drop by the null arm's spread, and all three null runs start from the same weights. On realloop_v1 that sd was 0.0061 for PricklySida, so the threshold sat at the 0.03 floor. PricklySida has 42 dev boxes, and the guard failed on it in 5 of the 6 steps, with drops of 0.033–0.057 (`realloop_v1/ledger.jsonl`). The sampling noise of a 42-box AP is not in that rule.
- **A sensitivity reading of the recorded ledger**, with hypothetical SEs because realloop_v1 has no sidecars:
  - with SE_PricklySida = 0.025, only V4 changes, to ACCEPT: a species-only failure at P_data 1.00;
  - V1–V3 and UNVERIFIED still fail the regression guard, and OTHER_HEAVY the flips guard.
  - Stage C (docs/CONTINUOUS_LOOP.md §5.1) is the rule's first reading on measured SEs.

### Stage A: pilot_v4 (known truth)

- **Build.** `inc2.pilot4` builds pilot_v4 on pilot_v3's bins, sha-identical: P0 and [I1, I2, Bswap, I3, Breal, I4, I5], with the same seeds, cold recipe, gate block (net flips) and replay mode (full).
- **Chains.** x1a and x1b, with no truth arm. The reference is pilot_v3's recorded truth verdicts. R0 is pilot_v3's own full chain and is not re-run.
- **Final exams.** dev and imageweeds. Test is not read.
- **The survival rule.** An arm survives only if it does all of the following:
  - it ACCEPTs I2, I3 and I5;
  - it REJECTs Bswap and Breal (a HOLD is not a rejection);
  - it agrees with the truth arm on ≥ 5 of 7 steps (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts);
  - its final incumbent's dev mAP50-95 is ≥ 0.8084 − 2 × 0.0043 = 0.7998 (pilot_v3's T_final dev mean and sd; the build checks that pilot_v3's report reproduces them).
- **R0 on its record:** agreement 5/7, final dev 0.8006; it survives.
- **The record.** `stage_a.json` is READY once pilot_v4 is done and every arm has all 7 verdicts and a final dev. The best survivor has the most agreement, then the higher final dev, then fewer epochs. D30 holds the TRAIN lane until this record is READY, whether or not an arm survived.

### Stage B: segment 1

- **Chains.** Segment 1 runs two chains: r0 and the best Stage A survivor, or r0 alone if none survives.
- **The rule.** For each chain and each step, δ_min = max(0, inc − mean(null) − 2·sd(null)). The chosen chain has, in order:
  1. the smallest median δ_min;
  2. then fewer recipe flags (P_recipe ≤ 0.25);
  3. then the cheaper recipe (fewer epochs);
  4. a tie goes to r0.
- **Implementation.** `inc2.recipes.stage_b_choice` reads the pinned ledger's gate entries. Test is never an input. No arm is added after results are seen without a new pre-registration.

### Baselines

| Run | Role | Seeds | Final exams | Rule |
|---|---|---|---|---|
| B_v2 on base_v2 | milestone 0, arm n640 | 0–4 | dev, imageweeds, test | the loop's starting point |
| Capacity arms on base_v2 | s640, m640 | 0–2 | dev, imageweeds, test | the switch rule above |
| Canary: B0 seed 0 under `inc2.train` | train_core (v2: v1's minus the L-8 drops) | 0 | dev | passes when \|canary − mean(b0_v1 seeds)\| ≤ sd(b0_v1 seeds) on dev (0.8082 ± 0.0063), its scorer sidecar was written, the run and its score are production ones, and it trained b0_v1's base manifest, or that manifest minus the L-8 drops its exp.json records (accepted by b0_v1's base sha256 plus the list's sha256), with b0_v1's cold recipe |
| B0 ∪ tsw | train_core ∪ tsw22 ∪ tsw23 | 0–2 | dev, imageweeds | recommended: isolates the harvested part of base_v2 |
| Milestone | a stream pool P_s | 0–4 | dev, imageweeds, test | the only place test is read; the chain incumbent is scored on the same exams as a secondary number |

- **Test reads (P10).** Only milestone reads open test: B_v2 (milestone 0), the capacity arms at R0 (L-4) and the stream's milestones. A build of any other role that lists test is refused.
- **Roles.** A role given to the builder must match what is built: B_v2 and the capacity arms train LOCK v2's base_v2 (by sha256) on n640 and on another arm, the canary trains LOCK v2's train_core on n640. Without a role (the autopilot's argv names none) the builder infers it from the same facts; anything else is a plain baseline without test.
- **Research-only.** A baseline's `research_only` flag is false only when every row is in splits v2's provenance file, that file hashes as LOCK v2 records, and no row is research-only. Otherwise it is true.

### How it was verified (laptop, no GPU; nothing of Protocol v3 has run on the cluster)

- **`tests/test_inc2_train.py`:**
  - the v3 table has no deviation on any arm;
  - freeze and LoRA are refused;
  - ood exams are refused;
  - the v1 and v2 evaluation manifests are refused by path and by content;
  - planted exact, 1–6-bit, re-encoded, hflip and rot90 copies of dev and test images are refused, each a failed run.json at stage guard;
  - the index cross-check refuses an hflip copy that GuardV2 is made to pass;
  - an L-5 image re-listed under another key and source is refused, and an L-5 list that does not hash as LOCK v2 records stops the run; the same for an L-8 drop and the L-8 list;
  - a missing guard, a missing or mismatched LOCK v2, and (in production) a testing LOCK each stop the run;
  - real 1-epoch CPU runs (base, cand, null, final) finish; the base and cand runs carry a sidecar whose arrays reproduce the score exactly; a tampered sidecar is re-scored, not retrained;
  - production runs are refused for a tiny recipe or a foreign cold checkpoint;
  - `run_inc2_job.sh` exports INC_JOB_SCRIPT as itself and runs `inc2.train`, then `inc.driver advance`;
  - the pinned modules are unchanged against git HEAD.
- **`tests/test_inc2_gate3.py`:**
  - the bootstrap is deterministic under its seed;
  - a real CPU sidecar reproduces the pinned scorer's per_class exactly;
  - with SE_s = pinned threshold_s / 1.96, every verdict of a battery equals the pinned one;
  - tampered score files and forged decisions are refused;
  - the sidecar's SE equals the sample sd of per-resample APs recomputed independently;
  - an unavailable step with a pinned ACCEPT commits as HOLD, and one with a pinned REJECT stays REJECT;
  - on realloop_v1's recorded ledger, the reading above.
- **`tests/test_inc2_pilot4.py`:**
  - the bins are sha-identical;
  - a bin holding an hflip copy of a test image refuses the build;
  - a HOLD on a planted bin is not a rejection;
  - the rule applied to pilot_v3's recorded chains gives R0 survives (5/7, 0.8006), and freeze and LoRA fail (3/7).
- **`tests/test_inc2_baseline.py`:**
  - every build passes the pinned driver's `validate_definition` and `check_definition_data`;
  - a FakeBackend baseline runs to done through the real v2 executor, and a 1-step chain runs to done with every spec accepted by the v2 executor; only `run_inc2_job.sh` is submitted;
  - the capacity decision is identical when every test score is perturbed;
  - the autopilot's argv shapes (`--arch`/`--imgsz`, no role) build B_v2, both capacity arms and the canary with the right role, arm and exams; roles that do not match, and test outside a milestone read, are refused;
  - a test-mode canary, or one trained on another manifest than b0_v1's, does not pass; one trained on b0_v1's manifest minus the recorded L-8 drops passes, and fails when the record names another reference or another list than LOCK v2's.
- **Mutation checks.** Each of these was changed in the source, the tests were run and failed, and the source was restored: the guard's refusal, the index cross-check, the variant check, the testing-LOCK refusal, the unhashable refusal, the ood refusal, the content check of evaluation manifests, x1a's lr0, imgsz in the deviation check, the arm's checkpoint sha256, the cand sidecar, the chain's sidecar failure, the job script's INC_JOB_SCRIPT export and its drift stop, the v3 z and floor, re-derivation, the sidecar's weights and production checks, the regression guard in v3, the bootstrap seed and its ddof, the capacity threshold, the canary's sd and its production and manifest checks, the capacity decision's test blindness, the union guard, the agreement minimum, HOLD as a rejection, the final dev floor, the pilot bins' guard, the role inference and the role's manifest and arm checks, the P10 test refusal, the `--arch`/`--imgsz` mapping, the secondary's log dir, the funnel modules in the job's drift check, the research-only flag and its provenance hash check, the truth arm's sidecar binding, the unavailable-ACCEPT hold and the L-5 check (45 mutations, each run in a scratch copy of the package).
