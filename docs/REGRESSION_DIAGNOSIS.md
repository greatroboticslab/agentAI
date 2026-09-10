# Why the round campaign's holdout metric fell for fifteen rounds

*Measured 2026-09-09 from the cluster's own run artifacts. Every number below names the
file it came from. Claims that the evidence does not settle are marked **unverified** and
stay that way until a control run exists.*

---

## 1. The series

Holdout mAP50-95 by round, from the Mongo round ledger (`weed#1` … `weed#15`, field
`metrics.map50_95`, each carrying the `results.csv` path it was read from):

| r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .6019 | .5919 | .5951 | .5829 | .5839 | .5738 | .5685 | .5693 | .5672 | .5711 | .5589 | .5665 | .5521 | .5594 | .5607 |

| segment | n | OLS slope / round | se | t | mean | sd |
|---|---|---|---|---|---|---|
| r1–r15 | 15 | **−0.00302** | 0.00034 | **−8.84** | 0.5735 | 0.0141 |
| r1–r7 (pool growing) | 7 | −0.00527 | 0.00065 | −8.11 | 0.5854 | 0.0109 |
| r8–r15 (**pool frozen**) | 8 | −0.00177 | 0.00079 | −2.24 | 0.5632 | 0.0060 |

The residual sd over all 15 rounds is 0.0053, which is the sealed `merged_curated` seed
std (0.005) to within a thousandth — the noise floor is right, and the trend is not noise.

## 2. The ruler did not move

Every round 8–15 validates on **1,977 images / 3,257 instances** — identical, round to
round (`per_class.all` parsed from each train job's own log; `evidence_rounds.json`).

The sealed cwd12 holdout artifact `results/framework/s3_tta_ceiling/ens3_640.json` reports
`n_images: 1977` and its twelve `per_class[*].n_gt` sum to **3,257**. Same images, same
instances. The round campaign and the sealed S3 protocol are measured on the same holdout.

## 3. The training pool did not grow either

`train: Scanning … N images` from each train job's log:

| round | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| train images | 45,620 | 62,959 ⚠ | 45,620 | 45,620 | 45,620 | 45,620 | 45,620 | 45,620 |

Seven of the eight are byte-identical. Round 9's 62,959 is **unverified** — it is the
maximum "Scanning" figure in that job's log and no other round shows it; it is recorded
here rather than explained.

This is consistent with the collector: harvest has returned `+0` slugs for eight
consecutive rounds with the pool frozen at 65 slugs, at 2 h 11 m – 4 h 37 m of GPU-share
per round.

**So rounds 8–15 are the same data, the same holdout, and the same hyperparameters, eight
times over.** The only input that changes between them is the initial weights.

## 4. What changes between rounds: the weights

`args.yaml` `model:` field, per round:

```
r8  ← results/framework/mega_iterm1_curated_s101/train8/weights/best.pt
r9  ← results/framework/mega_iterrnd8_train/job45326516/weights/best.pt
r10 ← .../mega_iterrnd9_train/job45373309/weights/best.pt
…
r15 ← .../mega_iterrnd14_train/job45592739/weights/best.pt
```

This is `mega_trainer.py`'s v3.0.19 *progressive training*: unless
`strategy["fresh_start"]` is set, the base model is `registry["last_mega_weights"]`, the
previous round's `best.pt`. Its stated rationale is in the source — *"Data set can grow
between rounds so this is transfer-learning-continuation"*. Since round 8 the data set has
not grown.

Round 8's base was the M1 curated seed-101 checkpoint. That recipe measures
**0.5894 ± 0.0025 (n=3)** on this holdout (`figures_data.json` → `m1_sealed_2026_08_23.curated`).
Round 8's **first epoch measured 0.5354** — the warm-start cost 0.054 mAP in one epoch.
Eight rounds later the campaign sits at 0.5607, still **0.0287 below the checkpoint it
started from**, against a recipe whose seed std is 0.0025.

## 5. The learning rate never anneals

`args.yaml` declares `lr0: 0.001`. The job log declares
`optimizer: MuSGD(lr=0.01, momentum=0.9)` — `optimizer: auto` overrides `lr0` by 10×, and
nothing reports the override. The scheduled `epochs` is 60 with `cos_lr: true`; `patience`
is 20 and every round ends at 21–35 epochs, so the cosine is always truncated.

Round 15's own curve (`evidence_rounds.json`, `rounds.15.curve`):

| epoch | 1 | 2 | 3 | 4 | 5 | … | 25 |
|---|---|---|---|---|---|---|---|
| mAP50-95 | .5518 | .5326 | .5291 | .5201 | **.5607** | … | .5515 |
| lr/pg0 | .0100 | .0200 | **.0298** | .0296 | .0293 | … | .0161 |
| train/box_loss | 1.4418 | 1.4302 | 1.4782 | 1.5555 | 1.5729 | … | 1.4406 |

The run inherits a 0.5594 model, warms the learning rate to 0.0298, loses 0.032 mAP doing
it, spikes once at epoch 5 — that spike is what `best.pt` saves and what the next round
inherits — and then spends twenty patience epochs returning to where it began. After 25
epochs `train/box_loss` is 1.4406 against 1.4418 at epoch 1. The last epoch's learning rate
is 0.0161, still **above** the nominal `lr0` of 0.01 and 16× the configured 0.001.

In **8 of 8** rounds the last epoch scores below the round's best. In round 13 the best
epoch was **epoch 1**: the inherited weights beat all twenty subsequent epochs.

## 5a. Why the configured learning rate is not the learning rate

`mega_trainer.py:1059` passes `lr0=strategy.get("lr", 0.001)` correctly, and
`run_m1_merged_seeds.sh` sets `"lr": 0.001`. Neither is wrong. `optimizer` is never passed,
so Ultralytics defaults to `optimizer="auto"`, which **computes its own learning rate and
discards `lr0`** — hence `MuSGD(lr=0.01)` in the log. `args.yaml` still records the
discarded `lr0: 0.001`, so the saved configuration does not describe the run that produced
the checkpoint beside it. Passing `optimizer` explicitly is what makes `lr0` binding.

## 5b. One published artifact has been overwritten by the campaign

`run_m1_merged_seeds.sh` writes `results/framework/m1_<tier>_seed<seed>.json`, keyed only
on tier and seed. The round campaign runs the same script with `TIER=curated`, `SEED=101`,
so every round overwrites the M1 curated seed-101 artifact.

It shows: `merged_images: 48208`, `job_id: 44304828`,
`best_pt: .../mega_iterm1_curated_s101/train3/weights/best.pt` — while seeds 102 and 103
show `merged_images: 13309` and job ids in the M1 range (44234865, 44234063). Round 8's
base model is `mega_iterm1_curated_s101/**train8**/weights/best.pt`, an eighth run inside
that same iteration directory.

The published row **M1 curated 0.5894 ± 0.0025 (n=3)** therefore has one of its three seeds
backed by an artifact that no longer describes the M1 run. The mean may still be correct —
`figures_data.json` recorded it on 2026-08-23, before the overwrites — but it is no longer
reproducible from the artifacts on disk. **Audit before this row goes in front of anyone.**
The job-scoped copy `m1_<tier>_seed<seed>_<jobid>.json`, written before training, is where
the original may survive.

## 6. What this does and does not establish

**Established.**
- The evaluation set is fixed (§2), so the decline is not a moving-ruler artifact.
- The training pool is fixed across rounds 8–15 (§3), so *added data cannot be the cause of
  the decline over that segment* — there was none.
- Each round re-heats a converged model to a learning rate it never anneals from (§5), and
  the checkpoint carried forward is a single noisy epoch, not a converged model.
- The campaign is 0.0287 below the checkpoint it started from, on the same data (§4).

**Not established.**
- That chaining *causes* the −0.00177/round slope over the frozen segment. At t = −2.24
  with n = 8 that slope alone is weak evidence; the strong evidence is the level (§4), not
  the slope.
- Anything about rounds 1–7. Their run directories no longer exist, so whether they were
  chained, and on what data, is **unverified**. The steeper −0.00527/round there coincides
  with a growing pool but nothing here separates the two.
- Round 9's 62,959-image scan (§3).

**Separate, and older than this campaign.** The absolute level (~0.56) is not the same
question as the decline. On this identical holdout a plain YOLO11n trained on the
3,671-image cwd12 train split with `nc=12` measures **0.8755 ± 0.0029 (n=3)**
(`s3_yolo11n_seed10{1,2,3}.json`). The round pipeline runs `nc=100` — twelve real classes
plus 88 `aux_*` placeholders — on a merged corpus whose in-domain core is instance-starved
by the pipeline's own dedup and holdout-stem filters. `DOUBLE_AGENT_SYSTEM.md` §4 already
retracted an earlier reading of that gap as a cost of harvested data and localised it to
the merge pipeline; §3's frozen pool is further evidence that harvested data is not what
moves this number.

## 7. The one flag that tests it

`mega_trainer.train_yolo_mega` already accepts `strategy["fresh_start"]`, which skips
`registry["last_mega_weights"]` and starts from `Config.DETECTION_MODEL`. The decisive
control is round 15's **exact** merged dataset and seed, trained with `fresh_start`, with
`epochs` set to what will actually run so the cosine completes. One job, one changed flag,
one row in the results table.
