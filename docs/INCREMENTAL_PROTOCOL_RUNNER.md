# INC runner: run specs, the driver and the pilot build

Companion to docs/INCREMENTAL_PROTOCOL.md: how the protocol's runs are described,
executed and advanced on the cluster.

## Run spec

A run is one training (or soup, or scoring-only) job, described by
`INC_DIR/<exp>/runs/<run_id>/spec.json`:

```json
{
  "exp": "pilot_v1",
  "run_id": "full__s03_Bswap__cand__s1",
  "kind": "base | union | cand | null | soup | final",
  "init": "/abs/path/weights.pt  (or 'yolo11n.pt', resolved against REPO)",
  "train_manifest": "/abs/path/manifest.jsonl   (absent for soup and final)",
  "soup_of": ["/abs/.../final.pt", "..."],      (soup only)
  "recipe": {"trainer": "full | freeze | lora", "epochs": 30, "optimizer": "SGD",
             "lr0": 0.002, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
             "warmup_epochs": 1, "warmup_bias_lr": 0.002, "cos_lr": true,
             "freeze": null, "lora": null, "imgsz": 640, "batch": 32, "seed": 1,
             "cache": "ram", "workers": 5, "close_mosaic": 10, "deterministic": true},
  "exams": ["dev"],
  "out_dir": "INC_DIR/<exp>/runs/<run_id>"
}
```

What each field means:
- `run_id` is path-safe and unique within the experiment.
- `recipe.freeze` is an int n: backbone layers 0..n-1 are frozen.
- `recipe.lora` is `{"rank": 16, "alpha": 32}`.
- `recipe.trainer` is `lora` iff `recipe.lora` is set.

Kinds:

| Kind | What it is |
|---|---|
| `base`, `union` | Cold runs from `yolo11n.pt` |
| `cand`, `null` | Incremental runs from an incumbent |
| `soup` | Uniform weight average of `soup_of` (same architecture, same class space). No training; scored on dev. |
| `final` | Scoring-only for a given `init` on the listed exams. It is the only kind allowed to list `test`. |

## Executor — `python -m weed_optimizer_framework.tools.inc.train --spec SPEC`

1. Refuses a spec that lists `test` unless `kind == final`, and refuses any unknown key.
2. For training kinds:
   - reads the train manifest;
   - runs `NeverTrainGuard.load().assert_trainable(images)`, which fails closed;
   - materialises the manifest into `out_dir/data`;
   - trains with Ultralytics, always with an explicit `optimizer`, `project=out_dir`, `name='train'`, `exist_ok=True`, `val=False`, `plots=False`.
     - `full` / `freeze`: `YOLO(init).train(...)`, then copies `train/weights/last.pt` to `weights/final.pt`. The trainer's own `save_dir` is used, never a hard-coded path.
     - `lora`: `inc.lora.train_lora(...)`, then copies the merged checkpoint to `weights/final.pt`.
3. For `soup`: averages the state dicts (float tensors; non-float buffers such as `num_batches_tracked` come from the first model) and saves an Ultralytics checkpoint to `weights/final.pt`.
4. For `final`: `weights/final.pt` is a symlink to `init`.
5. For each exam, runs the scorer **as a subprocess** (`python -m …inc.scorer --weights weights/final.pt --exam E --out scores/E.json`). The executor never computes a metric.
6. Writes `run.json`, atomically:
   - `status` (done | failed), `error`, `attempt`, `seconds`;
   - `n_train_images`, `n_train_boxes`;
   - `trainable_params`, `total_params`;
   - `guard` (images checked, hits);
   - `weights_sha256`, `ultralytics_version`, `hostname`, `slurm_job_id`.
   - On failure, the traceback goes in `error`. The executor exits non-zero, but `run.json` is still written.

The job script `weed_llm_benchmark/run_inc_job.sh <list_file> [<exp>]` runs as an sbatch array:
- partition GPU-shared, `--gres=gpu:v100-32:1`, 5 CPUs, 45G memory, 3h time limit;
- conda env `bench`;
- task i runs the spec on line i+1 of `list_file`;
- `REPO` points at the cluster checkout; `PYTHONPATH` points at the outer package copy.

After its run it calls `python -m …inc.driver advance --exp <exp> --quiet` (errors ignored), so the experiment advances as soon as any run finishes.

## Driver — `python -m weed_optimizer_framework.tools.inc.driver {init,advance,status} --exp EXP`

State lives in `INC_DIR/<exp>/`:
- `exp.json` (the definition, written by `init` from a builder such as `pilot.py`);
- `state.json` (progress);
- `ledger.jsonl` (append-only decisions);
- `runs/`;
- `manifests/`.

`advance` is idempotent and takes an exclusive `fcntl` lock on `state.json.lock` (non-blocking; if the lock is held, it returns immediately). Each call:
1. Collects runs: `run.json` done + every `scores/<exam>.json` present → complete.
   - A failed run is resubmitted once (attempt 2).
   - After a second failure the owning chain is marked `blocked`, with the error.
   - A run submitted more than 8 h ago with no `run.json` and absent from `squeue` is treated as failed.
2. Advances every unit of work that is ready (below) and submits all newly created specs as one sbatch array per call. Array concurrency: `%40`.
3. Writes `state.json` atomically. Prints a one-line summary per chain unless `--quiet`.

The sbatch call is behind a small backend interface (`SlurmBackend`, `FakeBackend`) so tests can simulate runs without a cluster.

### Experiment types

**`baseline`:**
- `base` runs on one manifest, one per seed, scored on dev.
- When all are complete: `final` runs on every exam, test included.
- Used for B0 (`train_core`, `pilot build-b0`) and base B (Step 1, `pilot build-baseline`).

**`chain`** (the pilot, `pilot build`, and the real loop, `realloop build`):
- `base` runs: P0 cold, 3 seeds, scored on dev. Every chain's incumbent starts as the seed-0 base run. Its incumbent score is that run's `scores/dev.json`.
- `replay_mode` (exp.json; absent = `sample`, which is steps 1–2 below; `pilot build --replay-mode full` writes `full`): in `full` mode no R1/R2 are drawn. `cand` trains on the chain's whole accepted pool ∪ D_k and `null` on the whole accepted pool. The step state and its gate entry record `replay_mode`, `pool_images`, `d_images` and `pool_accepted`, and the report shows the mode and each step's cand/null train sizes (docs/INCREMENTAL_PROTOCOL.md, Step 4, "Pilot v2").
- `gate` (exp.json, optional; absent = `GateConfig()` exactly, which is protocol v1): the `GateConfig` every `gate.decide`, `choose_soup` and `truth_detail` call of the experiment is made with.
  - It may hold only `GateConfig` fields, and init validates their types and ranges.
  - `require_production` is refused, because `testing` sets it. So are unknown keys and a block on a `baseline` experiment, which makes no gate decision.
  - `{"flips_mode": "net"}` is protocol v2 (docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)").
  - A definition without the block, with `{}` or with `{"flips_mode": "negative"}` is decided byte for byte as before.
  - The config is fixed at init. A definition with a block pins it in `state.json` `gate_pin` (the block and the whole resolved config, `flips_mode` included) and in the ledger as `gate_pin/0`. A definition without one pins nothing and writes exactly what it always did; its pin is `GateConfig()`'s defaults.
  - A pass whose exp.json resolves to another config (an edited, added or removed block, or a changed `testing`) refuses to run and writes nothing. No command changes the gate of a running experiment; build a new one.
  - `status` marks a chain whose config is not the defaults, e.g. `(gate flips net)`, and prints `GATE CONFIG CHANGED` when exp.json no longer matches the pin.
- For every recipe r (an independent chain) and every step k of the sequence, in order:
  1. Replay samples from the chain's accepted pool (`P0 ∪ accepted increments`):
     - `R1`: |D_k| images;
     - `R2`: another |D_k| images, disjoint from R1.
     - Both are seeded by `stable_int(f"{exp}/{r}/{k}")`. If the pool is too small for disjoint samples, raise.
  2. Six runs:
     - `cand` s0..s2 on `D_k ∪ R1`;
     - `null` s0..s2 on `R1 ∪ R2`;
     - all initialised from the incumbent, with recipe r, scored on dev.
  3. When all six are complete: `gate.decide(incumbent_dev, cands, nulls)`.
     - **ACCEPT:** a `soup` run of the three cand weights, scored on dev. Then `gate.choose_soup` picks the new incumbent (soup or cand s0). D_k joins the accepted pool.
     - **HOLD / REJECT:** the incumbent is unchanged.
     - Append a ledger entry with the decision dict, the input score paths and their sha256.
- **Truth arm** (`exp.truth = true`), independent of the chains and submitted at `init`:
  - for each step k, `union` cold runs on `T_{k-1} ∪ D_k`, 3 seeds, scored on dev;
  - T_0 = P0; T_k = T_{k-1} ∪ D_k only when D_k is listed as clean in `exp.json`.
  - The "without" runs for step k are the "with" runs of the last clean step before k, or the base runs for the first step.
  - `truth_decision` is computed as soon as both sets are complete.
- **When every chain has finished every step and the truth arm is complete:** `final` runs on all exams for:
  - each chain's final incumbent;
  - the base seeds;
  - the truth runs on `T_final`.
- Then `state.done = true`.

`status` prints, per chain, the step, the pending run count, the last decision and any blocked error. For the truth arm it prints complete/total.

## Pilot build — `python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v1 [--replay-mode {sample,full}] [--gate-flips-mode {negative,net}]`

Writes the manifests and `exp.json`. `--replay-mode` (default `sample`) goes into `exp.json` and `build_summary.json`. `--gate-flips-mode` (default `negative`, protocol v1) goes into both as the gate block, `"gate": {"flips_mode": ...}`; `net` is protocol v2 (docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)"). `build-b0` and `build-baseline` make no gate decision and refuse the flag.

The flag reaches exp.json only through a path that forwards it: this CLI, directly or through `run_inc_build.sh` (which forwards its arguments), and the autopilot. The autopilot's lever L9 builds a v1 pilot's v2 rebuild with `--gate-flips-mode net`. Its other rebuilds (L1, L2, L5, L6) forward the parent's pinned mode when it is not the default (`inc_autopilot/levers.py` `gate_params`, `loop_params`). The `inc.pilot build` and `inc.realloop build` policy rows take the flag as an enum (docs/INC_AUTOPILOT.md, D15, D16, L9). Right after building a v2 experiment, check exp.json's `gate` block and `state.json` `gate_pin.config.flips_mode`. The cold base runs come first, so no gate decision has been taken yet; a wrong mode means a new build under a new `--exp` name, because the pin refuses any change.

**Base P0 and the bins:**
- Read `train_core` and group it by `session`.
- **P0:** sessions sorted by size (descending, then name). Take sessions into P0 until P0 holds ≥ 50% of `train_core` images.
- **Bins:** the remaining sessions go into 6 bins by greedy packing (largest session first into the currently smallest bin; ties go to the lowest bin index).
- Sort the bins by their lexicographically smallest session. Bin index 3 is the Bswap source; the others, in order, are I1..I5.
- Record bin sizes and flag any bin outside ±35% of the mean.

**Bswap:**
- A copy of the source bin's labels, written to `INC_DIR/<exp>/labels/Bswap/`.
- 40% of its boxes (chosen with `numpy.random.default_rng(stable_int(exp + '/Bswap'))`) get a different species id, drawn uniformly from the other 11.
- The manifest's `source` is `Bswap`. Record which boxes changed.

**Breal:**
- Images from `REPO/datasets/project_agml__weed_crop_detection` and `REPO/datasets/project_agml__imageweeds_aerial_weed_detection` (`images/`, `labels/`).
- Class order is taken from the registry (`results/framework/dataset_registry.json`, `class_names`):
  - `species_of(name)` → its cwd12 id;
  - everything else → `OtherPlant` (12).
- Images within 6 bits of any evaluation image (`NeverTrainGuard.check`) are dropped and counted.
- Then N images are sampled (seeded), with N = the median size of I1..I5.
- The class-order assumption is recorded as unverified.

**Sequence:** `[I1, I2, Bswap, I3, Breal, I4, I5]`, with clean = {I1..I5}.

**Recipes:**

| Recipe | Settings |
|---|---|
| `full` | epochs 30, SGD, lr0 0.002, warmup_bias_lr 0.002 |
| `freeze` | as `full`, plus freeze 11 |
| `lora` | lr0 0.01, warmup_bias_lr 0.01, lora {rank 16, alpha 32} |

Base and union runs use the cold recipe from the protocol.

## Baseline build — `python -m weed_optimizer_framework.tools.inc.pilot build-baseline --exp base_b_v1 --manifest PATH [--seeds 0,1,2]`

A `baseline` experiment, exactly like `build-b0` but on any training manifest. For base B: `--manifest INC_DIR/step1/base_B.jsonl --exp base_b_v1`. `--testing` / `--testing-settings` work as for the other builds; `--exp` and `--manifest` are required.

**Before anything is written**, the manifest must pass `pilot.check_training_manifest` (fail closed):
- **The executor's own manifest check** (`inc/train.py check_manifest`), the one every base run repeats:
  - every row has the manifest keys;
  - keys are unique and path-safe;
  - every label line is a box of the INC class space (ids 0–12, coordinates in [0, 1]);
  - label and image bytes hash as the manifest says;
  - it is not an evaluation split's manifest.
- **The never-train guard** over the dHashes that check computed. An image within 6 bits of a dev, test or exam image, or one that cannot be hashed, refuses the build.
- **A production build** also needs `LOCK.json`, and the never-train index must be the one `LOCK.json` recorded (`pilot.never_train_status`). A testing build only warns.
- **A select base.** When the manifest is the `base_B.jsonl` an `inc.select` build wrote (the `select_summary.json` next to it names its sha256), that build must have read the `train_core` that `LOCK.json` records and the never-train index in place now (`pilot.select_provenance`). A base selected under an earlier lock would carry the old `train_core` rows. A production build refuses it; a testing build warns.

**Then:**
- The manifest is copied into `manifests/<sanitised stem>.jsonl` (same bytes).
- One cold `base` run per seed, with the protocol's cold recipe.
- After the base runs: `final` runs on every exam, test included (the driver's `baseline` type).
- `build_summary.json` records:
  - the manifest's sha256, images, sessions and boxes per class;
  - images and boxes per source;
  - the duplicate counts and the guard record;
  - the never-train lock status;
  - `select_build`: for a select base, the `train_core` and never-train sha256 that build read, set against `LOCK.json` and the current index (null for any other manifest);
  - the effective warmup.

**`base_b_v1` and the real loop train the same base.** The real loop's base arm (below) is the same three cold runs as `base_b_v1`: same `base_B.jsonl` bytes, recipe, seeds and init. Its final runs read base B's test again. The driver cannot take base weights from another experiment, and `driver.py` is pinned, so the builder cannot remove this. Building both costs three more cold 100-epoch runs and a second read of B's test. When to build `base_b_v1` is set in docs/INCREMENTAL_PROTOCOL.md, Step 1.6.

## Source relevance — `python -m weed_optimizer_framework.tools.inc.relevance build [--sample 300] [--seed 0] [--base-dir DIR] [--out OUT.json] [--force]`

The relevance filter of Step 1 (docs/INCREMENTAL_PROTOCOL.md, Step 1, "Source relevance"): zero-shot BioCLIP-2 P(plant) per crop, tau = the 5th percentile of `train_core`'s P(plant), and a source passes when the median P(plant) of its seeded sample is ≥ tau. The calibration must hold tau ≥ 0.5 (at least 95% of `train_core` crops judged more plant than non-plant), or the file is refused.

**Run it** after `inc.select build` and before `realloop build`, as a GPU job (Slurm opens the log file before the script runs, so the log dir must exist):

```
mkdir -p $REPO/results/framework/inc/step1/logs
sbatch run_inc_relevance.sh build                  # --sample 300 --seed 0 -> step1/relevance.json, relevance.md
```

- The job script hashes every module it imports (outer copy vs nested copy) and stops on a difference, as `run_inc_verify.sh` does; `INC_RELEVANCE_ALLOW_DRIFT=1` runs the outer copies anyway. It sets `HF_HUB_OFFLINE=1`: the model comes from the Hugging Face cache that `verify embed` used.
- The text encoding and matrix products are tiny. The job loads the model and every crop embedding (`verify.load_embeddings`), and opens no image. GPU-shared, 1 × V100-32, 45G, 1 h.

**Inputs**, refused unless they are one consistent set:
- the `inc.select` build in `--base-dir` (default `INC_DIR/step1`): `select_summary.json`, whose `increment_pool.jsonl` and `base_selected.jsonl` must still hash as it recorded;
- the `crops.csv` and the embedding shards that build read: the same sha256 for `crops.csv` and every shard file, and the same shard record;
- the text tower of the model named in the shards (`hf-hub:imageomics/bioclip-2`); a text tower of another model is refused;
- that model's own files from the Hugging Face cache, resolved as open_clip resolves them: the config and the weights must come from one snapshot, and the tokenizer must be the one the config names (`SimpleTokenizer`, context 77, for BioCLIP-2; open_clip's fallback Hugging Face tokenizer is refused).

**Outputs** (`--out`, default `<base-dir>/relevance.json`, and the `.md` beside it, written atomically):
- `params`: the prompts (`relevance.PROMPTS`), sample, seed, percentile, minimum crops, the rule and its seed text;
- `text_encoder`: name, logit scale, dimension, the sha256 of the text features, and `provenance`: the hub snapshot commit, the config and weights files with their sha256, and the tokenizer (class, context length, vocabulary);
- `calibration`: tau, `check` (tau ≥ 0.5: ok or not, and the share of `train_core` crops below 0.5), the `train_core` P(plant) percentiles and histogram, and the share at or above tau. P(plant) statistics are stored unrounded;
- `increment_pool` and `base_selected`: per source, images, usable crops, sample, median and quartiles, share ≥ tau, top-prompt shares (the leaf-disease set reported only) and status; the non-passing sources with their images; `name_check` (reported only): the non-passing sources whose name holds a word of `relevance.PLANT_NAME_WORDS`, possible false negatives for a person to look at;
- `inputs`: the hashes above.
- `relevance.md` prints tau and every P(plant) value in one format that keeps values near 0 and 1 apart (for example `2.5e-07`, `1 - 3.1e-05`).

**A failed calibration check** (tau < 0.5): the build writes both files, with the `.md` flagged, then exits 2. A rerun with the same inputs exits 2 again. `relevance.load` refuses the file, so neither `select increments` nor `realloop build` can use it. Read `relevance.md` (the `train_core` top prompts show which non-plant prompt took the crops) before changing anything.

On the real pool the check failed (cluster, 2026-09-27: tau = 0.0557, 34.5% of `train_core` crops below 0.5; docs/INCREMENTAL_PROTOCOL.md, Step 1, "Source relevance", "Result"). The prompts are not retuned on this pool. Steps 2–3 use the second criterion instead, source-level species evidence (next section).

**Reruns:** the same inputs and parameters are a no-op. Other inputs or parameters over an existing file refuse unless `--force`, because a draw may have recorded its sha256.

**Use:**
- `python -m weed_optimizer_framework.tools.inc.select increments --exp E --n 6 --other-heavy --relevance INC_DIR/step1/relevance.json`
- `realloop build` (below) reads `relevance.json` next to `--base` by default.
- Both read the file through `relevance.load`, which refuses:
  - a file made for another select build's `increment_pool.jsonl` or `base_selected.jsonl`;
  - one made under another minimum crop count, percentile or check threshold;
  - one whose calibration check failed;
  - one whose statuses, check or non-passing list do not follow from its recorded numbers;
  - a malformed one.
  - These are consistency checks, not a seal; the draw records the file's sha256.
- Every increment-pool image of a non-passing source leaves the draw, for the regular and the OtherPlant-heavy increments. The file must hold every pool source, with the pool's image counts.
- The draw records the file:
  - its sha256 joins the draw's parameters (`relevance_sha256`), so a filtered draw over an unfiltered one refuses without `--force`;
  - `increments_summary.json` `relevance`: the path, sha256, tau, the excluded sources with images and status, the images before the filter, and the near-dup groups the exclusion split. A split group's members of passing sources stay drawable.
- Without `--relevance`, select's draw and its parameters are exactly as before, and the summary's `relevance` is null.

## Source-level species evidence — `python -m weed_optimizer_framework.tools.inc.select increments ... --sources evidence [--min-evidence 1] [--admit-summary PATH]`

The second relevance criterion (docs/INCREMENTAL_PROTOCOL.md, Steps 2–3, "Source-level species evidence"), adopted at the R4 review of 2026-09-27 after the relevance build failed its calibration check. No job and no model: it reads `admit_summary.json`.

- `--sources` picks the criterion: `relevance` (the default: `--relevance` if given, else no filter, byte for byte as before) or `evidence`.
- **Rule.** With `evidence`, a pool source is eligible for the regular and `OtherPlant`-heavy increments only if verify admit judged at least `--min-evidence` (default 1) of its cwd12-species boxes `verified` (`admit_summary.json` `per_slug[source].boxes.verified`; `--admit-summary`, default verify's own file). Every increment-pool image of any other source leaves the draw.
- **Refused**, before anything is written:
  - `--min-evidence` below 1; `--relevance` with `--sources evidence`; `--min-evidence` or `--admit-summary` without it;
  - an admit summary without `per_slug` box verdict counts;
  - one that does not name the `verified.jsonl` and `crops.csv` the select build read (`select_summary.json` `inputs`);
  - where `select_summary.json` records `retrieval.source_evidence`: a source with more verified boxes than species crops there, or a source counted there without a `per_slug` entry. A verified box is a species crop with features that verify did not call a conflict, which is what that record counts;
  - an increment-pool source without a `per_slug` entry;
  - a draw larger than the evidenced pool.
- **Recorded:**
  - the draw's parameters gain `sources: evidence`, `min_evidence` and `admit_summary_sha256`, so an evidence draw over a default one refuses without `--force`;
  - `increments_summary.json` gains `increment_sources: evidence` and `evidence`: the rule, the admit summary's path and sha256, the images before the filter, the evidenced sources with their verified boxes and increment-pool images, the excluded sources with their images and verified boxes, the near-dup groups the exclusion split, and the cross-check (`checked: false` with the reason when `select_summary.json` has no source-evidence record).
- In the default mode the parameters and the summary have none of these keys.
- `select.pool_capacity(rows, clusters)` gives what a filtered pool can supply: its images, and the images of its `OtherPlant`-heavy near-dup groups by the draw's own rule. `realloop build` uses it to refuse before writing.

```
python -m weed_optimizer_framework.tools.inc.select increments --exp E --n 2 --other-heavy --sources evidence
```

## Real-loop build — `python -m weed_optimizer_framework.tools.inc.realloop build --exp NAME --replay-mode {sample,full} --recipes full[,freeze,lora] [--gate-flips-mode {negative,net}] [--base PATH] [--n-verified 6] [--size M] [--no-truth] [--increment-sources {relevance,evidence}] [--relevance PATH] [--min-evidence K]`

A `chain` experiment for Steps 2–3 on real data, built the way the pilot is. It writes `manifests/`, `build_summary.json` and the definition, then calls driver init. `--replay-mode` and `--recipes` are required, because the pilot decides them; `--testing` / `--testing-settings` work as for the other builds. `--gate-flips-mode` (default `negative`) goes into `exp.json` and `build_summary.json` as the gate block, as in the pilot build.

**Inputs.** Step 1 must be one consistent set, or the build refuses:
- `--base` (default `INC_DIR/step1/base_B.jsonl`) must be the `base_B.jsonl` that `select_summary.json` next to it names, by sha256. The increment pool is the rest of the verified set only relative to that base.
- Select's pool, cluster and base outputs, and the `verified.jsonl` and `pool_meta.jsonl` it read, must hash as its summary says.
- `admit_summary.json` must name this `verified.jsonl`, `pool.jsonl` and `pool_meta.jsonl`.
- Every pool image's verdict is rebuilt from verify's files and must give `admit_summary.json`'s per-source counts:
  - admitted: in `verified.jsonl`, as the same row as in `pool.jsonl`;
  - conflict: a box in `conflicts.csv` (a `conflicts.csv` key that is not a pool image, or that was admitted, refuses the build);
  - unknown: the rest.
- The select build must have read the `train_core` that `LOCK.json` records and the never-train index in place now (`pilot.select_provenance`, as in the baseline build). A production build refuses a base selected under an earlier lock; a testing build warns and records it.

**Base:** the base manifest, copied as `manifests/base_B.jsonl` (same bytes). It goes through `check_training_manifest` and runs cold with the protocol's recipe, 3 seeds. These are the same runs as `base_b_v1`, and the final runs read B's test again (see the baseline build). The build lists every baseline experiment already built on the same bytes (`build_summary.json` `same_base_baselines`) and logs a note.

**Increments.** Each holds M images. M = `--size`; by default it is 10% of the base's images, rounded (`select.INC_FRAC`, select's own default).

| Step | What it is |
|---|---|
| `V1`..`VN` (`--n-verified`, default 6) and `OTHER_HEAVY` | `select.increments(exp, N, M, other_heavy=True, relevance=R)`, unchanged; with `--increment-sources evidence`, `select.increments(exp, N, M, other_heavy=True, sources="evidence", min_evidence=K)`. The manifests are select's `inc_01.jsonl`.. and `inc_otherplant.jsonl`, with its `increments_summary.json`, in the experiment's `manifests/`. |
| `UNVERIFIED` (`manifests/inc_unverified.jsonl`) | M images of the one harvested source that has the most eligible images verify did not admit, among sources with at least M; ties go to the lower name. The same under either criterion. |

`--increment-sources` picks the relevance criterion of `V*` and `OTHER_HEAVY`: `relevance` (the default, the filter R below) or `evidence` (source-level species evidence, below). An evidence build records it in `exp.json` and `build_summary.json` as `step1.increment_sources` (`{"mode": "evidence", "min_evidence": K, "rule": ...}`). A relevance build records no such key; a missing key means `relevance`. So a default build writes the definition it wrote before this option existed, and an experiment built then can still be re-inited under the same `--exp` (driver init compares definitions).

With `--increment-sources relevance`, the relevance filter R:
- R is `--relevance`, by default `relevance.json` next to `--base` (`INC_DIR/step1/relevance.json`) when it exists. An explicit `--relevance` that does not exist refuses.
- A production build refuses without R, before anything is written. A testing build without R warns and draws `V*` and `OTHER_HEAVY` unfiltered; its summary's `relevance` says so.
- Any build refuses, before anything is written, an R that `relevance.load` refuses: one made for another select build, one whose calibration check failed, one whose statuses do not follow from its numbers.
- `UNVERIFIED` is never filtered: it is the planted, realistic bad increment. `build_summary.json` `unverified.source_relevance` records its source's status in R (or that the source has no increment-pool image).

With `--increment-sources evidence` (`--min-evidence K`, default 1):
- No relevance file is read, and a production build needs none. An explicit `--relevance` refuses; so does `--min-evidence` without this mode, or below 1.
- Before anything is written, the build loads the evidence as select will (`select.load_evidence` on verify's `admit_summary.json`, the one the Step 1 checks above read, against `select_summary.json`) and refuses on any of select's refusals (the two files disagreeing, a pool source without a `per_slug` entry).
- **Capacity**, also before anything is written: the evidenced increment pool must hold (N + 1) × M images, and M of them in `OtherPlant`-heavy near-dup groups (`select.pool_capacity`). Otherwise the build refuses with the numbers: the images held and needed, the `OtherPlant`-heavy images held and needed, the evidenced sources with their pool images, the excluded images and sources, and the largest `--n-verified` the image count allows at this M. It is a necessary condition: select's draw keeps near-dup groups whole and can still fall short by less than a group, and then refuses itself.
- `UNVERIFIED` is unchanged and not filtered. `build_summary.json` `unverified.source_evidence` records its source's verified boxes and whether it is evidenced.
- On the Step 1 of 2026-09-27 the default N = 6 at M = 393 refuses: the 5 evidenced sources hold 1,439 increment-pool images against 2,751 needed (docs/INCREMENTAL_PROTOCOL.md, Steps 2–3). The same section says what that pool is made of: 4 of the 5 sources are the sources of B's harvested part, 37% of the images are below select's retrieval gate, and 77% come from two sources that hold cwd12 copies.
- **Rebuilding a loop from its `exp.json`** (by hand, or the autopilot's rebuild parameters) must read `step1.increment_sources`: a missing key means `relevance`, and `{"mode": "evidence", "min_evidence": K}` means `--increment-sources evidence --min-evidence K`. A rebuild that reads only `step1.relevance` turns an evidence loop into a relevance one, which in production refuses without `relevance.json`.

```
python -m weed_optimizer_framework.tools.inc.realloop build --exp real_v1 --replay-mode full --recipes full \
    --increment-sources evidence --n-verified N --size M
```

How `UNVERIFIED` is drawn:
- **Checks on every non-admitted image.** Any failure stops the build, since it means Step 1 is stale:
  - it has a dHash in `pool_meta.jsonl`, clears the never-train guard and comes from no never-train dataset (select's `guard_rows` and `check_pool_against_core`);
  - it shares no key, path or image bytes with a `train_core` image of the base (`check_pool_against_core`, which select applies to the verified images: verify strips copies of `train_core` photographs from the pool);
  - it shares no key, path or image sha256 with a harvested image of the base or of a verified increment (verify drops exact duplicates from the pool).
- **Eligible:** not within `NEAR_DUP_BITS` (3) dHash bits of an image of the base or of a verified increment (base dHashes from the base check, increment dHashes from `pool_meta.jsonl`).
- **The draw:** `numpy.random.default_rng(stable_int(exp + "/unverified"))`, over the source's eligible images sorted by key. The rows are `pool.jsonl`'s own, with the labels as the species join gave them.
- **Recorded** (`build_summary.json` `unverified`):
  - the source, and every source's candidate table (non-admitted, conflict, unknown, excluded as near-dups, eligible);
  - the first 20 near-dup exclusions, each with the base or increment image it is near and the bit distance;
  - the drawn images' verdict mix;
  - their conflict boxes (label → predicted species);
  - the source's verdict counts from `admit_summary.json`.

**Before the definition is written:**
- every increment passes `check_training_manifest`;
- every increment holds exactly M images;
- the base and the increments are pairwise disjoint (key, path, sha256).

**Sequence:** `V1, V2, UNVERIFIED, V3, OTHER_HEAVY, V4, V5, V6` for N = 6. For other N: the first two verified increments, `UNVERIFIED`, the third, `OTHER_HEAVY`, then the rest.
- Clean = every verified increment, `OTHER_HEAVY` included, so the truth arm's T grows through them.
- `UNVERIFIED` is decided but never joins T.
- T_final = base + every verified increment.

**Recipes:** `--recipes` names a subset of the pilot's table, which is used as defined, not redefined. Base and union runs use the cold recipe. The truth arm is on unless `--no-truth`.

**`exp.json`** (the definition the driver runs) records:
- the base and every step: manifest, sha256, `n_images`, clean flag and `kind` (`verified`, `otherplant_heavy`, `unverified`); `UNVERIFIED` also carries its source, its verdict mix and the rule;
- M (`increment_images`), the recipes, the replay mode, the gate block, the truth arm and its recipe;
- the Step 1 files with their sha256 (`step1`), including the `train_core` and never-train index the select build read, the relevance file R (`step1.relevance`, null in a testing build without one and with `--increment-sources evidence`), and, in an evidence build only, the criterion (`step1.increment_sources`; absent means `relevance`);
- the effective warmup (`pilot.warmup_table`);
- the attribution scope. The driver runs items 1–3; the Step 1 verdicts are item 4's evidence; item 5 is not run.

**`build_summary.json`** records the same Step 1 files, warmup and attribution scope, and in addition:
- M and the sizes;
- per-source and per-class counts of the base and of every increment;
- select's draw parameters and its per-increment table;
- the unverified section;
- `relevance`: R's path, sha256 and tau, the excluded sources with images and status, and the near-dup groups the exclusion split (with `--increment-sources evidence`: `applied: false` and the reason);
- `evidence` (only with `--increment-sources evidence`): select's evidence record (the admit summary's path and sha256, the evidenced sources with verified boxes and pool images, the excluded sources with images and verified boxes, the cross-check, the split near-dup groups) and `capacity` (images and `OtherPlant`-heavy images needed and held);
- the never-train lock status, the select provenance record and `same_base_baselines`.

**How it was verified:** `tests/test_inc_realloop.py` builds a synthetic Step 1 world in verify's own formats, runs select's real build on it, and checks both builders on it.

`build-baseline`:
- the copy is byte for byte;
- the recipe, seeds and final runs are right, and the summary's counts match;
- base B's select record matches the lock and the index; a manifest no select build names has none;
- it refuses, before anything is written: a label class outside 0..12, a row without a manifest key, duplicate keys, an image near an evaluation image, an evaluation split's manifest, bad seeds, an unlocked or mismatched never-train index in production, and in production a base_B selected under another `train_core` lock;
- the CLI.

`realloop build`:
- every increment holds M images (10% of base B by default);
- the base and the increments are pairwise disjoint, and no `UNVERIFIED` image is a near copy of a base or verified image;
- `V*` and `OTHER_HEAVY` are byte for byte select's own draw;
- `UNVERIFIED` source choice: the source with more non-admitted images but fewer eligible ones loses (its images sit 1–2 bits from `train_core` photographs);
- non-admitted images planted 1–3 dHash bits from an image of `V1`, `V4`, `OTHER_HEAVY` and base B's harvested part are excluded, each listed with the image it is near; one planted 4 bits away stays eligible;
- the drawn keys are recomputed from the rule (eligible images sorted by key, seed text `exp + "/unverified"`), and the verdict mix is recounted;
- the sequence, clean flags and the pilot's recipes; the Step 1 provenance and the same-base baselines are recorded; `exp.json` holds the definition and `build_summary.json` the counts;
- driver init on FakeBackend, with the truth arm's T never holding `UNVERIFIED`;
- driven to done with a synthetic executor: chains accept the verified increments and reject `UNVERIFIED`, the truth arm agrees, T_final is right; the report has an `UNVERIFIED` attribution line per chain and no Bswap section;
- a rebuild is deterministic;
- refusals: a base that is not select's `base_B`, `conflicts.csv` and `admit_summary.json` disagreeing, a `pool.jsonl` admit did not read, a never-train hit, no source with M eligible images, bad recipes, replay mode, N or size, a second build, and a production build without `LOCK.json`;
- refusals where Step 1 is edited so that only one check can object (the other summaries re-stamped): a `verified.jsonl` or `pool_meta.jsonl` changed after select's build; a `conflicts.csv` row for an admitted image; a verified row whose label differs from its `pool.jsonl` row; a `base_B` with a class-13 box or an image near an evaluation image; increment label bytes that differ from the manifest (a verified increment, and `UNVERIFIED`); a non-admitted image byte-identical to a `train_core` photograph or to a harvested base image; and, in production, a base selected under another `train_core` lock or never-train index (a testing build records it and goes on);
- when this was built, each of these checks was switched off in turn in a sandbox copy, and the test failed every time;
- the relevance filter, with `relevance.json` built on the world by `inc.relevance` (injected fake text tower):
  - a production build refuses without it, and finds the one next to the base;
  - `V1`..`V6` and `OTHER_HEAVY` hold no image of a non-passing source (the unfiltered `real_t` draw does) and are select's filtered draw byte for byte;
  - R's sha256 is in `exp.json` `step1.relevance`, and the exclusions are in the summary;
  - `UNVERIFIED` is not filtered: still `harvU1`, whose status in R, `insufficient`, is recorded; and still `harvU1` when a copy of R marks it `fail` (its entry re-stamped consistently, since no source of this world looks non-plant);
  - a relevance file whose calibration check failed is refused;
  - a missing explicit `--relevance` refuses;
- source-level species evidence (`--increment-sources evidence`, `--min-evidence` one above the verified boxes of `harvU1`, the world's weakest source):
  - a production build builds without `relevance.json`, and the mode is in `exp.json`'s and `build_summary.json`'s `step1` (a default build has no such key);
  - `V1`..`V6` and `OTHER_HEAVY` hold only evidenced sources' images (the default `real_t` draw holds `harvU1`'s) and are select's evidence draw byte for byte;
  - `build_summary.json` `evidence` holds the evidenced and excluded sources, the admit summary's sha256 (the one in `step1`), the cross-check and the capacity;
  - `UNVERIFIED` is the rule's draw recomputed in the test, from `harvU1` although it is not evidenced, with its verified boxes recorded;
  - an explicit `--increment-sources relevance` gives the default build's manifests byte for byte, and the same `increments_summary.json`, build summary and definition apart from timestamps;
  - the default build is the one built before `--increment-sources` existed: `real_t`'s digest (`build_digest`: `exp.json`, `build_summary.json` and every manifest, with paths, sha256s and timestamps left out) equals `DEFAULT_BUILD_DIGEST`, recorded with `realloop.py` and `select.py` as they were then, the same under Python 3.10 (numpy 2.2.6, sklearn 1.7.2) and 3.12 (numpy 2.4.4, sklearn 1.8.0);
  - refusals before anything is written: `--relevance` with the mode, `--min-evidence` without it or at 0, an unknown mode, the two files disagreeing (`admit_summary.json` or `select_summary.json` edited), too few evidenced images for 6 + 1 increments of M (the message's numbers checked), too few `OtherPlant`-heavy images (with `pool_capacity` stubbed, since this world's evidenced pool is mostly `OtherPlant`-heavy);
- the CLI.

## Report — `python -m weed_optimizer_framework.tools.inc.report --exp EXP`

Writes `INC_DIR/<exp>/report.md` and `report.json`:
- **Per step:**
  - each chain's verdict, P_data, P_recipe, guards and attribution;
  - the truth arm's decision;
  - agreement between each chain and the truth arm.
- **Attribution of the bad-label step,** per chain (verdict, class vs localisation, attributed to labels), for each such step the experiment has: the pilot's `Bswap`, and the step whose `exp.json` kind is `unverified` (the real loop's `UNVERIFIED`, with its source and verdict mix). A chain experiment without a `Bswap` step has no Bswap section.
- **Final table:**
  - dev, ood22, ood23 and imageweeds (12-class and agnostic), and test, as mean ± sd over seeds where there are seeds;
  - for each chain's final incumbent, the base, and `T_final`.
- **GPU-hours** per chain and per arm, from `run.json` seconds.
- **Header:** the replay mode, the gate config as pinned at init (every `GateConfig` value, the gate block it was pinned from or its absence, and whether the flips guard counts negative flips (v1) or net flips (v2)), and the pinned decision code.
  - The pinned gate config is checked against the config every gate, soup, truth and `gate_pin` ledger entry recorded (`GATE CONFIG MISMATCH` per entry that differs) and against exp.json (`GATE CONFIG CHANGED`). exp.json is never the source of the config shown.
- **Protocol v2 check** (net-mode experiments only): each step's v1 verdict on the same runs (`gate.v1_counterfactual`) beside the net verdict, and the pre-registered outcome (`v2_check` in report.json: supported, refuted or inconclusive; provisional until done). The rule is in docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)", Pre-registration.

## Implementation notes (as built)

**Concurrency and bookkeeping (driver):**
- `advance` holds an exclusive lease file (`advance.lease`: O_CREAT|O_EXCL, token, host, pid, 30-minute renewable expiry) as well as the `fcntl` lock, because flock is not guaranteed to be coherent across Lustre clients.
- `state.json` carries a generation counter, which is checked before every save, ledger append and sbatch.
- A submission whose sbatch outcome is uncertain is looked up by job name (squeue, then sacct) before anything is resubmitted.
- A task that died without `run.json` is re-tracked to the Slurm job that still owns its run directory (the executor's `.executor.owner` heartbeat) before it counts as a failure.

**Code pinning:**
- At `init`, `state.json` pins the sha256 of `gate.py`, `driver.py`, `common.py`, `splits.py`, `inc/__init__.py` and `cwd12_species.py`.
- Every ledger entry carries the pinned hashes.
- `advance` refuses to run on changed code until `driver repin --reason …` records the change in the ledger.
- The gate's `flips_mode` option (protocol v2) and the gate pin changed `gate.py` and `driver.py`. New experiments pin the new hashes at init.
  - pilot_v1, pilot_v2, b0_v1 and base_b_v1 are finished, and they are not re-advanced or re-pinned. Their pins name the v1 code that decided them.
  - An advance of a done experiment returns before the code check.

**Commands beyond the ones above:**
- `driver watch` advances on an interval until done.
- `driver unblock` resets a blocked unit's failed runs and records it in the ledger.
- `driver repin` accepts changed code.

**Executor:**
- The job script's in-job advance is controlled by `INC_JOB_ADVANCE`:
  - `auto` advances only where `train --flock-check` finds a flock-coherent mount;
  - `1` always advances; `0` never does.
- Each run's directory is owned through an O_EXCL owner file with a heartbeat.
- A run whose saved weights are not from the final epoch fails as `diverged`.
- A done run is skipped only while its weights, inputs and scores still hash as recorded.
- A production recipe that departs from the protocol table is refused.

**Warmup:**
- Ultralytics warms up for at least 100 iterations. For a ~470-image incremental run at batch 32 this is about 6.7 of the 30 epochs.
- With nominal batch 64, the optimizer steps every 2 iterations at batch 32.
- Both are identical for every cand and null run and are recorded in `exp.json` as `effective_warmup`.
- In replay mode `full` the sizes depend on which earlier increments a chain accepted. `effective_warmup` records cand and null at `pool_min` (P0) and `pool_max` (P0 plus every earlier increment), and the report computes each decided step's warmup from its actual sizes.

**Breal:** it also drops images within 6 dHash bits of a `train_core` image, and not only eval images.

**Pilot attribution scope:**
- Attribution items 4 (BioCLIP-2 label audit) and 5 (leave-one-source-out) are not run in the pilot.
- Every gate entry lists them as not run, with the increment's sources.
- The pilot's increments are single-source; the Step 1 verifier provides item 4 for Steps 2–3.
- Item 4 is run once after the pilot, post hoc, by `inc/audit.py` (`run_inc_audit.sh`): trusted = P0, audited = I1–I5, `Bswap` and `Breal` (docs/INCREMENTAL_PROTOCOL.md, Gate, attribution item 4). The driver, `exp.json` and the gate entries are unchanged.

**Scorer:** a model with no prediction on an exam scores 0 (and is then rejected by the gate); it does not crash the scorer.

## Protocol v3 runner (`tools/inc2/`)

How Protocol v3 (docs/INCREMENTAL_PROTOCOL.md, "Protocol v3") is run. The pinned driver (`inc/driver.py`) runs every v2 experiment unchanged; only the executor, the builders and the job script are new.

### Executor — `python -m weed_optimizer_framework.tools.inc2.train --spec SPEC`

It is a copy of `inc/train.py`, and the spec format, the run-dir lock, re-runs and run.json are the same. The differences:

- **Exams:**
  - a spec may list only dev, imageweeds and test;
  - ood22 and ood23 are refused;
  - test stays final-only.
- **Manifest:** `check_manifest` also refuses the v1 and v2 evaluation manifests, by path and by content (a manifest whose sha256 is a LOCK's dev, test, ood22, ood23 or imageweeds sha256).
- **Guard (splits v2, fail closed).** `guard_rows`:
  - loads `inc2.guard.GuardV2.load(INC_DIR/splits/v2/LOCK.json)` and passes it each image's dHash and 8 variant dHashes;
  - refuses `unhashable`, `near_eval_v2`, `near_eval_variant`, `near_eval_embed` and any reason it does not know;
  - counts `base_copy`, `exact_dup` and `near_dup_intake` without refusing (a base_v2 row is a base copy by definition);
  - looks every variant up in the v2 never-train index itself (`inc.common.NeverTrainGuard.load(path)`), after checking the index's sha256 against LOCK v2;
  - refuses an image whose sha256 is in `splits/v2/l5_excluded.jsonl` (reason `l5_excluded`: the images L-5 drops from base_v2, under any key or source), after checking that file's sha256 against LOCK v2's `l5_excluded_sha256`.
  - refuses an image whose sha256 is in `splits/v2/train_core_variant_drops.jsonl` (reason `train_core_variant_drop`: the train_core rows L-8 drops, under any key or source), after checking that file's sha256 against LOCK v2's `train_core_variant_drops_sha256`.
  - A production run refuses a LOCK v2 written by a testing build, and a LOCK v2 that records no L-5 or no L-8 list.
  - run.json `guard` records the reason counts, the cross-check hits, and the LOCK, index and guard-module sha256.
- **Recipe and arm** (stage `recipe`, before the device):
  - exp.json's `arm` record gives the imgsz;
  - `inc2.recipes.deviations` gives the departures from the v3 table;
  - a cold run's init must be the arm's checkpoint (its name, and its sha256 when the record pins one).
  - Production refuses either kind of departure; testing records them (`recipe_deviations`, `init_check`).
  - An exp.json without `arm` is the n640 arm, allowed only with `init_weights` yolo11n.pt or none.
- **Code:** `code_modules()` hashes the v1 executor modules, `tools/funnel/leak.py` with the two funnel modules it imports (`__init__.py`, `embed.py`), and every `tools/inc2/*.py` into run.json. Drift from the nested copy fails the run, as in v1.
- **Sidecar.** After the scorer, every base, union, cand and soup run scored on dev runs `python -m weed_optimizer_framework.tools.inc2.scorer_sidecar --weights weights/final.pt --exam dev --score scores/dev.json --out scores/dev.sidecar.json`, with the scorer's own test settings in test mode.
  - The sidecar must name this run's weights and this run's dev score (sha256), and its npz must hash as recorded. Otherwise the run fails at stage `sidecar`, except in a `baseline` experiment, where no gate decision reads it: there the failure is recorded (`sidecars.dev.status` "failed", a warning) and the run is done.
  - run.json `sidecars.dev` records the JSON's and the npz's path and sha256, and the SE per species.
  - A re-run whose sidecar files are missing or changed re-scores the verified weights instead of retraining.
  - Cost per run: a second validation pass on dev, plus the bootstrap. The bootstrap took 20 s on the laptop for a synthetic exam with 617 images and 108k predictions.
- **run.json** also says `protocol` "v3", `protocol_package` "inc2" and `splits_version` "v2", and records `arm`, `recipe_name` and `init_check`.

### Job script — `weed_llm_benchmark/run_inc2_job.sh <list_file> [<exp>]`

It is `run_inc_job.sh` with the v2 executor (the same sbatch header, `INC_JOB_ADVANCE` modes and exit status). In addition:
- **INC_JOB_SCRIPT.** Before anything else, it exports `INC_JOB_SCRIPT=$REPO/weed_llm_benchmark/run_inc2_job.sh`. The pinned driver's in-job advance (`driver.job_script()`) therefore keeps submitting this script, never `run_inc_job.sh`. A missing copy stops it with exit 2.
- **Drift check.** It compares every module the executor and the driver import (the funnel package's `__init__.py`, `embed.py` and `leak.py` included), and every nested `tools/inc2/*.py`, between the outer and the nested package copies. A missing outer module or a difference stops the job with exit 1; `INC_ALLOW_DRIFT=1` runs it anyway.
- **Order:**
  1. `python -m …inc2.train --spec SPEC`;
  2. `python -m …inc.driver advance --exp EXP --quiet`; in auto mode only after `inc2.train --flock-check` passes.
- **What it never does:** reset, pull or copy the checkout, or touch Roboflow.
- `INC2_JOB_REPO` and `INC2_JOB_CONDA_SH` replace the cluster paths (tests only).

The builders below set `INC_JOB_SCRIPT` to this script before `driver init` and refuse when it already names another script. Any other caller that advances a v2 experiment must export it too: the build job script and the autopilot's `advance`.

### Baselines — `python -m weed_optimizer_framework.tools.inc2.baseline …`

| Command | What it does |
|---|---|
| `build --exp E (--manifest P \| --union P1,P2,…) [--seeds …] [--arm n640\|s640\|m640 \| --arch yolo11s --imgsz 640] [--final-exams …] [--role …] [--testing \| --testing-settings JSON] [--no-init]` | A `baseline` experiment: one cold base run per seed with the arm's cold recipe, then final runs on the final exams. The arm is `--arm`, or `--arch` with `--imgsz` (the autopilot's L23B argv), which must name one table row and agree with `--arm` when both are given. Role defaults: b_v2 and milestone seeds 0–4, canary seed 0, the others 0–2; finals dev, imageweeds, test for b_v2, capacity and milestone, dev for the canary, dev and imageweeds for union and baseline. |
| `canary-verdict --exp canary_v2 [--reference b0_v1]` | `INC_DIR/<exp>/canary.json`: \|canary base__s0 dev − mean\| ≤ sd of the reference's base seeds (b0_v1 report.json's dev 12-class, else its runs' scores), the canary's sidecar written, a production run and score, and the reference's base manifest (sha256), cold recipe and seed 0 (from the reference's exp.json, which must exist). The manifest also matches as the reference's minus the L-8 drops the canary's exp.json records (`variant_drops`): by the reference's base sha256 plus the list's sha256, the list still recorded by LOCK v2, and the reference's manifest filtered by the listed image sha256s hashing to the canary's; `canary.json` `manifest_match` says how it matched. The canary is where a sidecar that does not work on the cluster's Ultralytics shows up first. |
| `capacity-verdict --n b_v2 --arms b_v2_s640,b_v2_m640 [--m M] [--out-dir D]` | The L-4 decision (dev only) → `INC_DIR/capacity/capacity_v1.json`; the report for people (dev, imageweeds and test, mean ± sd, gap to 0.90) → `capacity_v1_report.{json,md}`. |
| `secondary --exp <milestone> --weights W [--run-id secondary__incumbent] [--source TEXT]` | A final spec scoring the chain incumbent on the milestone's exams, a one-line list file, `secondary.json` and the sbatch argv of `run_inc2_job.sh`. It submits nothing; the driver does not track that run. |
| `estimate --n-images N [--seeds …] [--arm A]` | The est. cost of a baseline. |

**Roles (P10: test is read only at milestone reads).** Without `--role` the role is inferred: `--union` is union; LOCK v2's base_v2 (by sha256) is b_v2 on n640 and capacity on another arm; LOCK v2's train_core is canary; anything else is baseline. An explicit role must match: b_v2 and capacity train base_v2 (b_v2 on n640, capacity on another arm), the canary trains train_core on n640, union needs `--union`. A build that lists test for a role other than b_v2, capacity or milestone is refused.

**Build checks, before anything is written:**
- the arm resolves; its checkpoint is in `$REPO`, and a testing build may lack it;
- the role matches the manifest and the arm, and the final exams are the role's (above);
- LOCK v2 exists and its never-train index hashes as recorded; a production build refuses a testing LOCK;
- the manifest passes `inc2.train.check_manifest` and `guard_rows`;
- `--union` parts are pairwise disjoint by key, image path and sha256;
- the definition passes the pinned driver's `validate_definition` and `check_definition_data`.

**exp.json** carries `inc2.recipes.stamp(arm)`:
- `protocol` v3, `protocol_package` inc2, `splits_version` v2;
- the `arm` record with the checkpoint's sha256;
- `init_weights` = the arm's checkpoint.

It also records `role`, `base.source_manifest` and `base.source_locked` (the LOCK v2 name of the manifest, or a list for a union), `cost_estimate` (est., low and high GPU-h, and the projected longest run against the 8 h cold walltime and D26's 0.8 line), `research_only` (false only when every row is in splits v2's provenance file, that file hashes as LOCK v2 records, and none is research-only; otherwise true, with its basis), and `splits_v2` (the LOCK and index sha256). `build_summary.json` records the manifest summary, the guard record, the LOCK status and the sha256 of the builder, the executor, the guard, the driver and `inc/pilot.py`.

**capacity_v1.json (format inc2-capacity/1):**
- per arm: the seeds, dev values, mean, sd, diff_vs_n, pooled_sd, qualifies, and the score files read (path and sha256);
- `chosen_arm`, `chosen_exp` and `chosen_arm_record`;
- `step_cost` with its basis (measured or est.), `m`, and `truth_every`.
- It opens exp.json, the base runs' run.json and their `scores/dev.json`, and nothing else.

### Stage A — `python -m weed_optimizer_framework.tools.inc2.pilot4 …`

| Command | What it does |
|---|---|
| `build --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b [--testing …] [--no-init]` | The chain on pilot_v3's bins, as below. |
| `verdict --exp pilot_v4 [--no-write]` | `INC_DIR/pilot_v4/stage_a.json`, as below. |

**build:**
- **Bins.** It copies P0 and the seven bins of pilot_v3's exp.json byte for byte into `pilot_v4/manifests/`, and each copy must hash to pilot_v3's recorded `manifest_sha256`. Each step entry keeps its name, clean flag and sessions and records `source_manifest`.
- **Checks.** Every bin must pass `check_manifest` and the v2 guard. pilot_v3 must be a full-replay chain whose base recipe is the v3 cold recipe, with a truth verdict for all 7 steps.
- **Recipes.** Only x1a and x1b; r0, freeze and LoRA are refused.
- **exp.json** has `truth: false`, pilot_v3's gate block, final exams dev and imageweeds, and a `stage_a` block. That block holds the rule, the truth verdicts, R0's record judged by the same rule, the T_final cross-check, and the sha256 of pilot_v3's exp.json, ledger and report.

**stage_a.json (format inc2-stage-a/1):**
- `status` (READY | PENDING);
- per arm: its step verdicts, agreement, final dev (`runs/final__<r>__incumbent/scores/dev.json`), every check and `survives`;
- `survivors`, `best_survivor` and `segment1_recipes` (`["r0"]` plus the best survivor);
- while PENDING, what is missing;
- the sha256 of its inputs.

### Protocol v3 gate — `python -m weed_optimizer_framework.tools.inc2.gate3 {decide,show} --exp E [--no-verify] [--out P]`

`decide` writes `INC_DIR/<exp>/gate3.json` (format inc2-gate3/1). It has one record per gate entry (`steps`) and per truth entry (`truth`), each with these fields:
- `verdict` (v3), `pinned_verdict`, `changed`, `commit_verdict`, `v3_applied` and `status` (decided | unavailable); an unavailable step also has `commit_rule` and the pinned `failed_guards` and `p_data_le_p_reject`;
- `failed_guards`, `pinned_failed_guards`, `p_data_le_p_reject`, `guards` (regression and flips as recorded, species v3), `species_tolerance` and `species_se`;
- `attribution` (blame follows the v3 verdict);
- `inputs`: the ledger entry's sha256, the score files, and the sidecar's path and sha256.

The document also records the ledger's and exp.json's sha256, both configs and a summary (changed, unavailable, unavailable_held, truth_unavailable). `show` prints the same without writing. `--no-verify` decides from the recorded guards without re-deriving them from the score files; the truth arm's sidecar is still bound to the weights of its first "without" run through that run's score file (checked by sha256).

**Unavailable steps never accept.** A step whose v3 decision cannot be made commits its pinned verdict, except that a pinned ACCEPT commits as HOLD. The v3 tolerance can be stricter than the pinned one for a species whose null spread is wide, so a pinned ACCEPT is not a Protocol v3 ACCEPT; a HOLD returns the increment to the queue once. A pinned REJECT or HOLD stands.

**How a stream commit uses it** (group E, `inc2.stream commit`):
1. After segment `<sid>_sNNN` is done, run `decide` or `gate3.decide_experiment(exp)`.
2. For the chain the Stage B rule names (`inc2.recipes.stage_b_choice` on the same ledger), an increment joins P_s when its step's `commit_verdict` is ACCEPT.
3. The disposition of a REJECT (docs/CONTINUOUS_LOOP.md §3.5) reads the step's v3 `failed_guards` and `p_data_le_p_reject`: data (unless the truth record's `commit_verdict` is helps), then species, flips, and recipe (regression only).
4. A step recorded unavailable commits as its `commit_verdict` (a pinned ACCEPT as HOLD), and its disposition reads the pinned guards recorded with it.

The pinned chain inside the segment is not re-run. A step that v3 ACCEPTs and the pinned gate REJECTed joins P_s on its own step's evidence, and the next segment trains cold on P_s.

### Scorer sidecar — `python -m weed_optimizer_framework.tools.inc2.scorer_sidecar --weights W --exam dev --score S --out O`

It calls `inc.scorer.score` (pinned, unchanged) with a subclass of the scorer's validator, installed through the scorer's validator cache for its own process. It writes:
- `O` (format inc2-scorer-sidecar/1): the recorded score's path, sha256 and stamps; the sidecar score's metrics; the consistency checks (the stamps must be equal and per_class within 0.002 of the recorded score; `ap_per_class` on the captured arrays must equal its own per_class within 1e-9); the npz's path and sha256; and `species_se`, which holds the seed text, the seed, 1,000 resamples and per species `ap`, `se`, `n_valid`, `n_gt`, `mean`, `p2_5` and `p97_5`;
- `<stem>.images.npz`: `keys`, `pred_img`, `tp` (N × 10), `conf`, `pred_cls`, `target_img` and `target_cls`.

Exit codes: 0 written, 2 refused (writing nothing), 1 error. It is also the backfill tool for a run scored before the sidecar existed; that is a GPU job.
