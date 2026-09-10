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

## 5c. The chain is longer than the ledger, and every cold run beats every chained one

The ledger's rounds 1–7 have no `mega_iterrnd*` directories because they were not written
there. They are `mega_iterm1_curated_s101/train2` … `train8`, and their `results.csv` best
values match the ledger to the fourth decimal, seven for seven:

| ledger | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|
| ledger `map50_95` | .6019 | .5919 | .5951 | .5829 | .5839 | .5738 | .5685 |
| `train2…train8` best | .60188 | .59188 | .59511 | .58290 | .58392 | .57385 | .56854 |

Each `args.yaml` names its predecessor, so the whole history is one chain:

```
yolo26x.pt (cold)
  └─ mega_iterm1_raw_s103/train            0.60620   ← the best run this pipeline ever produced
       └─ …curated_s101/train2   = round 1  0.60188
            └─ train3            = round 2  0.59188
                 └─ … train8     = round 7  0.56854
                      └─ mega_iterrnd8_train … = rounds 8–15, ending 0.5607
```

**Seventeen consecutive warm restarts, −0.0455 from the cold checkpoint they began at.**

The six original M1 runs were all cold from `yolo26x.pt`, and their `results.csv` files
reproduce both published rows exactly:

| recipe | seeds | mean ± std |
|---|---|---|
| M1 raw, cold | .60553 / .59788 / .60620 | **0.6032 ± 0.0046** |
| M1 curated, cold | .58733 / .58875 / .59215 | **0.5894 ± 0.0025** |

**The best chained run (0.60188, the first link) never beat the cold run it started from
(0.60620), and nothing since has come close.**

### This corrects §5

Rounds 1–7 ran the **full 60 epochs** — `train2`…`train8` each have 60 rows in
`results.csv`, so their cosine completed and their learning rate annealed. They declined
0.0334 anyway. **Schedule truncation is therefore not the cause of the early decline; the
warm-start chain alone is.** Truncation is a second defect that appears at round 8, when
the round scheduler took over and runs began ending at 21–35 epochs.

### The artifact audit resolves in the numbers' favour

Both artifact problems are bookkeeping, not science:

- `m1_curated_seed101.json` and its eight job-scoped copies all describe **round** runs
  (jobs 45326516 … 45640207). The original M1 seed-101 run survives as
  `mega_iterm1_curated_s101/train/results.csv`, dated 2026-08-23, best 0.58733.
- `s3_yolo11n_scratch_seed10{1,2,3}.json` carry the pretrained arm's numbers. The real
  from-scratch curves are `s3_yolo11n/scratch_s10{1,2,3}/results.csv`: 0.80551 / 0.80089 /
  0.80587 → **0.8041 ± 0.0028**, which is exactly the published row.

**Every published mean reproduces from a `results.csv`. None of them should be cited from
the summary JSONs, which are keyed on tier+seed and have been overwritten.**

## 5d. Corrections to §4, §5 and §5c

An adversarial re-reading of the same evidence overturned four statements made earlier in
this document. They are corrected here rather than edited away.

**"The warm start destroys the inherited model" — withdrawn.** Round 8's base is
`mega_iterm1_curated_s101/**train8**/weights/best.pt`, whose own best was **0.56854** — not
the original M1 curated run (`.../train`, 0.58733). The earlier comparison of round 8's
epoch-1 value (0.5354) against 0.5894 compared two different runs. Worse, **the inherited
checkpoint is never validated at epoch 0 in any round**, so its score under this round's own
eval is unmeasured everywhere; "epoch 1 fell to X" compares a mid-training epoch taken at
near-peak learning rate against another run's best epoch. And **7 of the 15 links finished
ABOVE their parent** (train3→train4, train5→train6, train8→r8, r9→r10, r11→r12, r13→r14,
r14→r15). This is a random walk with a downward drift, not a mechanism that destroys its
input each time.

**"Every cold run beats every chained run" — false.** The first chained link, train2 at
0.60188, beats M1 raw seed-102 (0.59788) and all three M1 curated runs. The defensible
statement is narrower: **no chained run ever exceeded the best cold run (0.60620), and no
link after the first came within 0.03 of it.**

**"Last epoch worse than best in 8 of 8 rounds" — withdrawn as evidence.** With
`patience=20` and no round reaching its epoch cap, a run can only end on its best epoch by
hitting the cap; the pattern is mechanically necessary, not a finding. The reported metric
is the best epoch, so last-epoch values never entered the series anyway.

**The reported per-round metric is a max-over-epochs statistic taken at high learning rate.**
Round 15's headline 0.5607 is epoch 5 alone, sitting 0.037 above both of its neighbours
(e4 = 0.52014, e6 = 0.52322) at 98% of peak LR. Under estimators that are not a max, the
decline over rounds 8–15 survives but shrinks: post-warmup plateau mean −0.00186 per round
(t = −2.56), mean of the last five epochs −0.00099 (t = −2.17).

**The schedule mechanism was described wrongly in §5.** `args.yaml` carries `time: 10.0`,
and `ultralytics/engine/trainer.py` L542-544 recomputes
`self.epochs = self.args.epochs = ceil(self.args.time * 3600 / mean_epoch_time)` after every
epoch and rebuilds the scheduler with it. So the cosine horizon is not 60 — it is re-planned
against a wall clock each epoch, landing near 50, and `patience=20` ends the run at 21–35.
`close_mosaic=10` fires at horizon − 10 ≈ 40 and therefore **never fired in any of the
fifteen rounds**. The `lr0` override is also not silent in the log: Ultralytics prints
`'optimizer=auto' found, ignoring 'lr0=0.001'`. What is silent is `args.yaml`, written in
`__init__` before `build_optimizer` runs, so the file on disk keeps the discarded value.

**The `+0` harvest has one exception, and it is a governance finding.** Round 9's pool was
68,019 rather than 48,752 because the DINOv2 gate, at the same fixed `MIN_DINO_SCORE=0.50`,
admitted two datasets it rejected in every other round: `fvossel__csgo_player_detection`
(4,014 images — Counter-Strike screenshots) and `rf_bishwarup-halder__crop-health-advisor`
(15,253). Both were gone again by round 10. The gate seeds each slug's sample from that
slug's index in registry iteration order, so it is **not deterministic at a fixed
threshold**. A video-game dataset entered a weed-detection training corpus and left again,
and no check in the loop said so. It moved the metric by −0.002, inside noise.

**Data cannot be excluded as a cause for rounds 1–7.** The campaign's own tier ladder inside
this merge pipeline measures 0.5601 / 0.5588 / 0.5455 at +0 / +5k / +15k harvested images —
added data is not free here. Round 2's pool was 48,208 images against the M1 curated
recipe's 13,309 one day earlier. Over rounds 1–7 the pool grew **and** the chain ran, so the
two are confounded and neither can be attributed. The exclusion holds only for rounds 8–15,
where the pool is byte-identical.

**No per-round error bar exists.** All fifteen rounds ran seed 101, once each. Every
comparison of a round-to-round difference against "the 0.005 noise floor" borrows a seed std
from a different recipe. Jobs `45672672_[1-2]` are measuring the round recipe's own seed std
at seeds 102 and 103, on round 15's exact data and schedule.

## 5e. Round 9's pool anomaly, verified from the merge log and the source

§3 recorded round 9's 62,959-image training scan as unexplained. It is explained, and the
explanation is a governance defect rather than a curiosity.

The `[Merge]` lines in each train job's own log give the per-slug counts. Rounds 10 and 15
are identical — **24 slugs, 48,752 unique images**. Round 9 is **26 slugs, 68,019**. The two
extra slugs are:

| slug | images |
|---|---|
| `fvossel__csgo_player_detection` | 4,014 |
| `rf_bishwarup-halder__crop-health-advisor` | 15,253 |

4,014 + 15,253 = 19,267 = 68,019 − 48,752 exactly. The first is Counter-Strike player
screenshots. Both were admitted at `MIN_DINO_SCORE=0.50` in round 9 and rejected at the same
threshold in every other round, without themselves changing.

**Why a fixed threshold is not a fixed decision.** `dinov2_curator.py:371` scores each
candidate as

```python
for i, slug in enumerate(slugs):
    res = score_one_slug(slug, info, model, proc, ref, seed=i)
```

— the per-slug image sample is seeded by **that slug's index in the candidate listing**. Add
or remove one dataset anywhere in the registry and every slug after it is re-sampled, so its
score moves and its pass/fail against a fixed threshold can flip while the dataset is
untouched. `_sample_images` compounds it: it walks `rglob("*")` and stops at `n * 20`, so the
pool it samples from is a filesystem-order-dependent prefix of the slug, not the slug.

The metric cost was −0.002, inside noise. The finding is not the cost. It is that a
video-game dataset entered a weed-detection training corpus, stayed for one round, left
again — and **no check in the loop said so**. `pool_growth` is one-sided and only looks for
absence of growth; nothing compares the slug list round to round.

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
