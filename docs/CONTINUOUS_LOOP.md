# Continuous loop: the platform collects, filters, stacks and gates new data by itself

This is the contract for the continuous incremental loop on the weed domain. Five pieces exist today but have never run together: the harvest, Step 1, the increment chain, the gate and the autopilot. This contract joins them into one loop that the platform runs without a person:
1. collect targeted data;
2. filter it box by box;
3. cut it into fixed-size increments;
4. train each increment against the incumbent;
5. roll back what hurts;
6. consolidate, and repeat.

Every experiment in this contract is submitted by the platform. People do three things:
- approve R3 items outside the envelope;
- take R4 decisions;
- read the milestone reports.

**Companion documents:**
- docs/INCREMENTAL_PROTOCOL.md and docs/INCREMENTAL_PROTOCOL_RUNNER.md: splits v1, scorer, gate, recipes;
- docs/INC_AUTOPILOT.md: campaign ticker, levers L1–L14, diagnoses D1–D19, governance;
- docs/FUNNEL_AUDIT.md, docs/FUNNEL_AUDIT_RUNNER.md and docs/FUNNEL_REPRODUCE.md: the live filter audit, which shares inputs with this loop (§3.9);
- docs/SUPERWEED_PLAN.md: the previous collection campaign and why it ended.

**How this contract was checked.**
- **Sources.** It was written from the code at commit 385ca77 and from the local copies of `results/framework/inc/{splits/v1, step1, b0_v1, base_b_v1, realloop_v1, funnel}`. Line numbers refer to that commit.
- **Nothing here has run.**
- **Numbers.** A number marked *est.* is an estimate, and its basis is given. Every other number is read from those files or from RESEARCH_LOG.md (2026-09-27 and 2026-09-28 entries).
- **Path conventions:**
  - code paths are relative to `weed_llm_benchmark/weed_optimizer_framework/tools/`;
  - job scripts live in `weed_llm_benchmark/`;
  - tests live in `weed_llm_benchmark/tests/`;
  - `INC_DIR` is `$REPO/results/framework/inc` on the cluster.

---

## 1. Goal, measure, and why the loop has never run end to end

### 1.1 Goal

The professor's method, run continuously by the platform:
1. **A large, high-precision base dataset.**
2. **New data keeps arriving.** The harvest agent collects it, it is filtered, and it is stacked onto training in **fixed-size increments**. The recipes are full rehearsal, freeze and LoRA.
3. **Every increment is gated.** The gate compares the model with the incumbent. If the model gets worse, the drop is attributed (data or recipe) and the increment is rolled back. The loop then continues with the next batch.

### 1.2 The measure

- **Success measure.** The sealed cwd12 **test** mAP50-95: 1,977 images, the 12 species, scored only by the locked scorer `inc/scorer.py`. The scorer is pinned to Ultralytics 8.4.37 and hashed into `LOCK.json`.
  - Today: **B0 = 0.8541 ± 0.0074** (YOLO11n, 3,049 train_core images, 3 cold seeds).
  - Target: **≥ 0.90**, a gap of 0.046.
- **Decisions use dev only.** Dev is 617 images from 8 whole capture sessions of cwd12 train.
  - Test never enters a decision.
  - Test is scored only at milestone consolidations (§5.5) and is shown only to people.
- **Headline number.** At each milestone, the headline is the mean ± sd of test over **5 cold seeds trained on the accumulated pool**. It is directly comparable with B0's cold seeds. The warm chain incumbent's test score is reported beside it.

### 1.3 Why the loop has never run end to end

| Piece | State on 2026-09-28 | Evidence |
|---|---|---|
| Collection | **Off.** The round scheduler's `weed` domain has had `enabled: false` since 2026-09-10, and round 16's collect job (45671119) was cancelled. An automatic stop-loss had already paused it on 2026-08-29, after two 12 h training timeouts. | Rounds 8–15 each harvested +0 (`queries_tried 83, candidates_passed_filter 0`), and rounds 3–15 retrained the same 24 slugs and 48,752 images. docs/REGRESSION_DIAGNOSIS.md §5i–5k. |
| Old collect → filter → train pipeline | Trained through `mega_trainer`: the sealed holdout was the validation set; classes were `md5(slug)%88` slots (about 1/3 colliding); the class joins before v3.60 were wrong; the warm-start chain explains +0.0287 (5.0σ) of the decline from 0.6019 to 0.5607. It scored 0.58–0.60. | docs/REGRESSION_DIAGNOSIS.md, CHANGELOG v3.60. **It must never train again (§8).** |
| Harvest filters | They silently drop the target species (§7.2):<br>• a substring vocabulary rejects species-only names;<br>• "car" rejects every Carpetweed project;<br>• Kaggle, GitHub and Roboflow downloads register `class_names: []`, so every box becomes OtherPlant. | `dataset_discovery.py:1066-1088`, `roboflow_source.py:71-76`, `extra_sources.py:248,437-440`. |
| INC autopilot | Drives experiments, not collection. It goes COMPLETE when it has nothing to propose; `weed_inc_v1` is COMPLETE. | `campaign._complete_residual` (2696), `check_goal` (500). |
| Step 1 | **One-shot.** Any pool change alters the sha256 of `crops.csv`, which voids the embeddings and the verifier. A full rerun takes about 2.5 h plus the fit. | `verify.check_fresh:961-980`, `_load_fitted:1736`. |
| Step 1 admission | **Image-level.** One conflicting box drops the whole image. It lost 989 of 2,049 verified target boxes, including all 346 Palmer amaranth boxes. | `verify.image_verdict:1379`; FUNNEL_AUDIT.md:691. |
| Increment pool after base B | **206** verified target boxes against 623,891 OtherPlant boxes. That is why realloop_v1's increments were 90–99 % OtherPlant. | `step1/select_summary.json` `boxes.increment_pool`. |
| Real loop | realloop_v1 fixes its six increments when it is built. The pinned driver writes `exp.json` once, and a chain is `done` at `k >= len(steps)`. | `driver.py:1306, 2051`; `realloop.py:48`. |
| Recipe | The cheap recipe lowered the incumbent at every realloop_v1 step: P_recipe 0.0, and null 0.8164 < inc 0.8189. All 6 steps were REJECTed, and the species guard failed on PricklySida in 5 of them. No data effect could be seen. | `realloop_v1/report.json`, `ledger.jsonl`. |

**Two qualifications from the realloop_v1 ledger.** Together they explain why this contract treats the recipe as necessary to fix but not sufficient (§5.1):
- **The null was not what rejected the increments.** Setting null = inc changes none of the six verdicts:
  - s01–s04 had P_data 0.11–0.44;
  - s05 failed the flips guard (net excess 6.7 against 5.0);
  - s06 failed the species guard.
- **The incumbent `base__s0` was the best of the three base seeds:** 0.8189 against 0.8154 and 0.8055. The full-rehearsal null (0.8164) is *above* the base-seed mean (0.8133). It is below only that one seed.

---

## 2. Owner decisions (2026-09-28)

These decisions were made by the project owner on 2026-09-28. They are logged once in the stream campaign's ledger as DEC-A to DEC-D, with `decided_by: human`. They are not reopened here. Each is listed with what it costs.

### D-A. The base grows, and the 3SeasonWeedDet10 2022/2023 subsets move into training

**What the base becomes:** cwd12 train_core (3,049) + base B's 878 selected harvest images + 3SeasonWeedDet10 (Zenodo 14861516) data2022 (1,915) and data2023 (1,784). That is 7,626 images before the recorded drops of §4.2.
- ImageWeeds stays an exam.
- dev and the sealed test do not change and are never trained on.
- This needs:
  - splits **v2**, with its own LOCK;
  - a never-train index over dev + test + ImageWeeds (5,802 images, dHash radius 6);
  - new baselines.

**Costs:**
- **Out-of-season generalisation is no longer measured.** The ood22 and ood23 exams are gone. B vs B0 on ood22 (+0.025) was the only measured benefit of harvested data, and that signal is lost.
- **The class mix shifts.**
  - The base's OtherPlant share goes from 0 % of train_core boxes to 50 % (13,344 of 26,559 boxes; 12,702 of them are Lambsquarters from data2023).
  - PricklySida falls from 5.2 % to 2.0 % of target boxes, and CutleafGroundcherry from 1.0 % to 0.4 %. Neither species gains a box. These are the rare species the gate's species guard trips on.
- **The pinned v1 code refuses every 3SeasonWeedDet10 image.** Its never-train index lists all 3,699 of them. Every training path for v2 is therefore new code (§4.3).
- **ImageWeeds is same-lab for every v2 model.** Base B holds 17 images of `project_agml__imageweeds_aerial_weed_detection`, from the same NDSU project as the ImageWeeds exam. Every v2 model's ImageWeeds score is reported as "same-lab", as FUNNEL_AUDIT H6(c) already does.
- **Comparability.** B0's and B's dev, test and ImageWeeds scores stay directly comparable, because the manifests and the scorer are identical. Their ood numbers become history.

### D-B. Step 1 admits per box, with masking

An image with at least one verified target box is admitted. Its blocking boxes (conflict or unknown) are masked with a mean-colour fill, instead of dropping the whole image. Data reaches training only as gated increments, so the gate is the safety net.

**Costs:**
- **Mean-colour patches enter training.** YOLO has no ignore region.
- **Out-of-domain precision is unmeasured.** The masks rely on the verifier's precision, which is 99.96–100 % in-domain (harvested copies of cwd12) and unmeasured outside it. §10 R1 takes the first out-of-domain reading.
- **Overlaps are refused.** A masked box that overlaps a kept box (IoU ≥ 0.5) means one plant carries two contradicting labels. That image goes to the human queue (§3.3).
- **Gain:** up to +989 verified target boxes in the existing pool, 346 of them Palmer amaranth.

### D-C. Collection is re-enabled and targeted at the 12 cwd12 species

The targets are Waterhemp, MorningGlory (*Ipomoea* spp.), Purslane, SpottedSpurge, Carpetweed, Ragweed (*Ambrosia artemisiifolia*), Eclipta, PricklySida, PalmerAmaranth, Sicklepod, Goosegrass and CutleafGroundcherry. The known candidate datasets are listed in §7.3.

**Costs:**
- a new collector: the old harvest's filters cannot find these species (§7.2);
- a licence gate per image;
- a mandatory copy scan for sources from the lab that produced dev and test;
- jobs on GPU-shared, because the allocation refuses RM-shared ("Invalid qos", FUNNEL_AUDIT_RUNNER.md, note 33);
- a byte budget on /ocean.

### D-D. Fixed-size increments, the pinned gate, and rollback

- Increments have a fixed size M.
- Each increment is gated against the incumbent by the pinned gate (protocol v2, net flips), with a truth arm where affordable.
- A rejected increment is rolled back (not accepted) and recorded.
- The loop never stops by itself while collection yields data and the budget allows.

**Costs (est., V100, §5.6):**
- about 3.0 GPU-h of chain per increment, plus 4.2–4.9 GPU-h for its truth arm;
- supply, not SU, limits the rate (§10).

### 2.5 Defaults this contract sets, pending the owner (R4)

These are implementation choices the decisions above leave open. The contract runs with the default shown until the owner changes it. Each is logged the same way.

| # | Item | Default | Why |
|---|---|---|---|
| P1 | Recipe | Chosen by the pre-registered Protocol v3 rule (§5.1). The current full recipe (R0) stays eligible. | The realloop_v1 qualifications (§1.3). |
| P2 | M, K | M = ⌈0.10 × \|base v2\|⌉ = **763**, fixed for stream version 1. K ≤ 4 increments per segment. | §5.3. |
| P3 | Truth arm | On for every step. A person may switch it off for a segment only when the monthly window cannot fit it. | §5.4. |
| P4 | What an increment holds | Images with ≥ 1 kept verified target box only (§3.4). | realloop_v1: 90–99 % OtherPlant increments moved nothing, and OTHER_HEAVY hurt. |
| P5 | Budget | Campaign envelope 600 SU until 2026-12-31; monthly window 250 SU; daily cap 80 SU; an allocation reserve read from `projects`. | §6.6. Stays inside the domain's 1,500 SU (`db.py:488`), so card X14 is not needed now. |
| P6 | Licences | Unresolved → held. Non-commercial → `research_only: true` (as funnel DEC-7). A permissive copy is preferred over an NC copy. Nothing is redistributed. | §7.4. |
| P7 | "Never stops by itself" | Read as "never COMPLETEs by itself". Stop-losses still pause and raise a card. | §6.7. |
| P8 | Funnel ordering | Rows the funnel's recovery arms also use are held until funnel step F9 (§3.9). | The funnel's H10d hold-out stays clean. |
| P9 | Sources from the lab that produced dev and test (CottonWeedDet3, CottonWeedID15) | Held until the funnel's H6 copy detector (`leak_v1`) is calibrated, then scanned. | A near-copy of test would inflate the success measure. |
| P10 | Test reads | Only at milestone consolidations. | §5.5. |
| P11 | New code | Accepted once, subject to the §9 acceptance tests, as funnel DEC-5 did. The new protocol modules are pinned at acceptance. | §9. |

### 2.6 Decisions recorded before the build (2026-09-28)

These were made after the design review and before any code, under the owner's delegation for this campaign. L-8 was added on 2026-09-29, after the first real `inc2.splits build` refused (§4.2 step 5a), and L-9 the same day, after the second refused on the embedding scan (§4.2 step 5b). They are logged with `decided_by: human-delegated` when group F lands. Where one differs from §2.5, it replaces that default.

| Id | Decision | Reason |
|---|---|---|
| L-1 | P1–P4 and P6–P11 are adopted as §2.5 states them. | They are the design's defaults, and the review found no reason to change them. |
| L-2 | **Budget (replaces P5).** The campaign envelope is 1,000 SU until 2026-12-31, with a 350 SU monthly window and a 120 SU daily cap. It stays inside the allocation (10,529 GPU SU remaining on 2026-09-28) and inside the domain's 1,500 SU. [amended 2026-10-04 by the owner: the monthly window and the daily cap are removed; the envelope and its end date stay (§6.6, "Amendment (2026-10-04)").] | The capacity arms of L-4 and Stage C are added to R0. |
| L-3 | **The species guard's tolerance is per species (Protocol v3, pre-registered).** For each species s, tol_s = max(0.03, 1.96 × SE_s). SE_s is the image-bootstrap standard error of the incumbent's dev AP for s: 1,000 resamples of dev images, with the seed from `stable_int("inc2/v3/species_se")`. The overall regression guard and every other guard are unchanged. It is implemented as a new gate version in `inc2/`; `inc/gate.py` stays pinned. Stage C (§5.1) is its first reading. | The v1/v2 guard scales one species' drop by the overall null spread (sd 0.0061), with a 0.03 floor. PricklySida has 42 dev boxes, and its own sampling noise is several times larger. The guard failed on PricklySida in 5 of 6 realloop_v1 steps, drops of 0.033–0.057. That rejects data on measurement noise, not on harm. |
| L-4 | **Capacity grid at R0 (n640, s640, m640).** B_v2 is trained cold with YOLO11n at 640 px (continuity), YOLO11s at 640 px and YOLO11m at 640 px. Every arm trains and scores at 640 px, so the grid varies capacity only. Each arm gets 3 seeds, the same cold recipe, final EMA weights and the locked scorer. **Rule:** if an arm's dev mean beats YOLO11n's by more than 2 pooled sd, the stream switches to the best such arm. Otherwise it stays on YOLO11n. When the chosen arm's per-step cost with the truth arm exceeds 25 GPU-h, the truth arm runs on every ⌈cost/25⌉-th step, and that is recorded. Test is read once per arm at R0, as a milestone read, and reported with the gap to 0.90. | Class-agnostic test mAP50-95 is 0.874 for B0 and 0.873 for B. Box accuracy therefore caps the 12-class score below 0.90, whatever the species accuracy. Earlier larger models (RF-DETR-L; yolo26x at 1024) reached about 0.89–0.90 under another evaluator. That motivates the arms, but it is not evidence under the locked scorer. yolo26x at 1024 needs roughly 75× the FLOPs of YOLO11n at 640, so it is left out of a loop that trains at every step. The grid first had YOLO11s at 1024 px (s1024). The locked scorer infers at 640 px only, so an arm trained at 1024 px would be measured at 640 px and confound capacity with resolution; s640 replaces it. A resolution arm needs a new scorer version (R4) and is not in this grid. |
| L-5 | **base_v2 drops the images of `rf_karthikeya-c8pvy__weed-detection-cwp10` and `rf_zig-zag-lnodr__weed-detection-vanpe` outright.** These are 812 of base B's 878 harvested images. The other 66 go through the copy scan (§4.2). | Both sources are re-uploads from the lab that produced dev and test, and the v1 pool caught 132 and 59 near-eval copies in them. A copy that dHash misses would inflate the success measure. Their images add little beyond cwd12 itself: B against B0 moved dev by +0.005 and test by −0.004, both within noise. |
| L-6 | **Freeze and LoRA stay out of stream version 1**, by the survival rule of §5.1. A new pilot that passes the rule can re-admit them. | pilot_v3 chains: full reached 5/7 agreement and test 0.8475; LoRA 3/7 and 0.8132; freeze 3/7 and 0.7992. |
| L-7 | **The /ocean quota floor of D27 is 3 %.** The campaign PAUSEs when staging + projected bytes leave less than 3 % of the /ocean quota free, about 210 GB of the 6.84 TiB project quota. Before collection starts, free space must also cover `collect_gb_envelope`. `stream_thresholds.json` D27 `quota_free_frac` is 0.03 (it was 0.1). | Measured 2026-09-28: 454 GB free (94 % used). The whole lab shares the quota, so a 10 % floor (about 700 GB) would PAUSE the campaign on its first snapshot, before it collects anything. The stream's projected need is under 50 GB. |
| L-8 | **train_core rows within 6 bits of an evaluation image under a flip or rotation are dropped (2026-09-29).** They leave base_v2, `splits/v2/train_core.jsonl` and every v2 training manifest, through an explicit list the build writes (`splits/v2/train_core_variant_drops.jsonl`, its sha256 in LOCK v2), handled as L-5's list is: `inc2.train` refuses the listed bytes under any key. The build records them as an incident in `summary.json` (count, keys, matches) instead of refusing. More than max(1, ⌊0.5 % × train_core⌋) such rows (15 of 3,049), or a hit on the stored dHash itself, still refuses the build (R4). The real case: `train_core__20210909_NIKOND3300_YL_91`, whose transverse is 5 bits from `test__20210910_NIKOND3300_YL_160` (build job 47257533). **Note on v1 results:** B0's, B's and realloop_v1's training sets held that image (v1 compared the stored dHash only). The effect is negligible (1 of 3,049), and they are not re-run. | 1 image of 3,049: dropping it costs nothing measurable, and an evaluation image must not have a transposed near-copy in training. Nobody inspects the pair visually, because evaluation pixels are never shown to an external model. |
| L-9 | **The embedding copy threshold is recalibrated per image on hard same-domain negatives, and the 3SeasonWeedDet10 rows are judged by their capture provenance (2026-09-29).** (a) tsw22 and tsw23 come from the md5-verified Zenodo original of the lab's own dataset, and every file name carries its capture date and camera. A tsw row whose capture session (the date_camera prefix of its name) is the session of a dev or test image is dropped: same-session frames are a leak risk. Every other tsw row with a capture session is exempt from the embedding threshold and stays under the 6-bit dHash + 8-variant check, which already removed the 31 cwd12 photographs. (b) The funnel's leak_v1 list of base B copies is read only for matches against a v2 evaluation split (dev, test, imageweeds). A match against ood22 or ood23 is not a v2 leak: it is recorded as such and not applied. (c) For images without capture provenance (base B's kept part, and every future intake source), the threshold is recalibrated: each negative image is scored as the scan scores an image, by its best cosine over the v2 evaluation images outside its own capture session and date. The hard negatives are train_core images; the funnel's provenance-disjoint group and two clearly unrelated harvested sources are further tiers. The threshold is the smallest cosine with a per-image false-positive rate of at most 1 % on every tier large enough to measure it (the one-sided 97.5 % bound reported), never below the funnel's 0.8256. Recall per augmentation family is measured at it; a family below 0.95 is a known limit, not a refusal. It is `inc2/embed_calibration.py`, written to `splits/v2/embed_calibration_v2.json` (sha256 in LOCK v2), used by `inc2.splits` build and lock, bound to every held row by `collect/intake.py`, and applied by `step1_stream`'s copy scan. (d) Every refusal for a true copy stays: dHash within 6 bits under any variant of an evaluation image; an embedding hit at or above the new threshold for an image without capture provenance; L-5 and L-8. The funnel's leak_v1 and its pre-registered H6 are not changed. Each candidate's rule is recorded per row (`splits/v2/copy_rules.jsonl`). | The second real build (job 47259471) refused as an R4 incident on 7 of base B's kept images. 6 matched ood23 images through the funnel's leak_v1 list at cos 0.83–0.86 and 20–27 dHash bits; ood23 is an exam in v1 but training in v2. The 7th matched ImageWeeds at cos 0.828 and 27 bits. At the funnel's threshold 0.8256 the scan also flagged 805 of tsw22's 1,915 images (3,085 pairs, cos 0.826–0.954, 1 pair at or above 0.95), 42 of tsw23's 1,784 (85 pairs, at most 0.886) and 289 of base B's 878 (905 pairs; 11 at or above 0.95, mostly the L-5 re-uploads). tsw22 (97 sessions) and tsw23 (8) share no capture session with dev, test or train_core, and every high tsw22 pair is a 2022 capture against a 2021 capture of another session and camera. The funnel scored its negatives per pair (hard negatives: rate 0.006, bound 0.0105 over 2,000 pairs) and never on same-domain scenes. The scan flags an image on its best cosine over about 5,800 evaluation images, and the funnel quarantines a source on any single hit. The per-image and per-source false-alarm rates are therefore far above 1 %: the funnel flagged 39 of its 47 sources, among them a video-game source. At 0.8256 the detector measures scene similarity on these pairs, not copying. |

---

## 3. The data flow, component by component

```
L15 discover (lab, R0) ──► collect/…/candidates.json
        │
L16 collect (lab or GPU-shared job, R2) ──► INC_DIR/intake/staging/<source>/   content-addressed, checksums
        │
collect intake: normalise → class map (taxonomy, cards) → licence gate → inc2.guard (never-train v2,
        │        base copy, near-dup) ──► INC_DIR/intake/<batch>/ ; registry entry annotation "intake_v1"
        │
L17 admit = inc2.step1_stream (one GPU-shared job): frozen BioCLIP-2 verifier, per-box admission, mask_except
        │        ──► INC_DIR/step1_stream/batches/bNNNN/ ; step1_stream/queue/queue.jsonl (append-only)
        │
inc2.stream cut: target images, whole near-dup and capture groups, exactly M, oldest first, one source first
        │        ──► INC_DIR/stream/<sid>/increments/<inc>.jsonl
        │
L18 segment = pinned driver chain: base P_{s-1}, K steps, full rehearsal, net-flips gate, truth arm,
        │        executor inc2.train ──► INC_DIR/<sid>_sNNN/ (exp.json, ledger.jsonl, report.json)
        │
L19 commit: ACCEPT → P_s ; REJECT (data) → quarantine ; REJECT (recipe) or HOLD → returned once
        │
L20 milestone: 5 cold seeds on P_s, finals dev / imageweeds / test ──► INC_DIR/<sid>_mNNN/ ; D25 → L21 rollback
        │
reports: segment report.md, stream report, RESEARCH_LOG entry at each milestone
```

"Filtered" means three things in this contract:
- the guards (never-train v2, base copy, near-duplicate, licence);
- the verifier's per-box verdicts;
- per-box admission.

"Stacked" means full rehearsal: every candidate run trains on the incumbent's pool plus the increment.

### 3.1 Discovery and collection — `tools/collect/` (new package, group D)

- **Runs on.**
  - `plan` and every lab-only provider run on the lab.
  - `fetch` runs as a job only for the providers the network probe (§7.2) shows reachable from compute nodes, on GPU-shared with one V100, because RM-shared is refused.
  - Lab fetches reach the cluster through an rsync operation that is verified by sha256 on arrival. It works like `OP_FUNNEL_SYNC`.
- **Inputs.**
  - `collect/domains/weed.json` (format `collect-domain/1`). It holds:
    - the targets, as a pointer to `funnel/domains/weed.json` by path and sha256 (never a copy);
    - the providers and the query grammar;
    - the licence policy;
    - the byte and yield budgets;
    - the known items: D-C's list, `decided_by: owner 2026-09-28 D-C`;
    - card class tables for sources whose names need one.
  - The funnel's offline taxonomy cache (`funnel/taxonomy_cache.json`) and `funnel/names.py`.
- **Outputs.**
  - `candidates.json`: per candidate, the provider, the query that found it, the card, the licence, the annotation type, the declared classes mapped through the taxonomy, bytes and image count.
  - `INC_DIR/intake/staging/<source>/`: content-addressed raw files, the provider checksum and `fetch.json`.
- **State.** `INC_DIR/intake/sources.jsonl` (append-only). Per source it records the status (candidate, held, fetched, intaken, closed or quarantined), attempts, bytes, SU and yield.
- **Refuses when:**
  - the provider is not in the config;
  - the byte cap is exceeded;
  - credentials are missing;
  - the source is image-level only (filed R3 for a person);
  - the source is in `mega_trainer.NEVER_TRAIN_SLUGS`, quarantined, or in an evaluation lab group without the copy scan.

### 3.2 Intake: normalise, class map, licence, guard — `collect/{normalize,classmap,licence,intake}.py` (group D) and `inc2/guard.py` (group A)

- **Normalise.** COCO, VOC, VIA, WeedCOCO, the MFWD `gt.csv` and YOLO are converted to `images/` + `labels/`, with the class list in source-id order. Each directory is listed once; there is no `rglob` on Lustre.
- **Class map.** Class list → INC id, through `funnel.names` and `funnel.taxonomy.Resolver` (offline), with card `class_table` overrides.
  - `target` / `target_synonym` → the target id (0–11).
  - Any other resolved plant → 12 (OtherPlant).
  - `numeric`, `unresolvable` and `no_name` → **13 = unmapped**. This id exists only in intake labels, never in a training label. Unmapped boxes are kept, so that D-B can mask them. They become targets only through a card (FUNNEL_AUDIT H3a), and they trigger L12 or L11a as a prerequisite [review: L26, the stream's own name lever (§6.2); L11a and L12 write the funnel's directory].
  - The provenance is recorded: card sha256, taxonomy cache sha256, and the status of each name.
  - **[review] EPPO codes.** MFWD's `gt.csv` names its classes by EPPO code (POROL, …), and Weed-AI records often do the same. `funnel.names` and `funnel.taxonomy` have no EPPO resolution: a code is `unresolvable`, which maps to 13, and the source would then need an owner-approved card class table, which means a person. The collector therefore carries an offline EPPO table (code → binomial, from EPPO's published code list, versioned and pinned by sha256 in `collect/domains/weed.json`). The binomial goes through the offline resolver like any other name. PAGS8's growth-stage classes need the same treatment: a stage name that holds no taxon resolves through the card's recorded species (*A. palmeri*), and that card table must be pinned in the config before the first wave. Without one of these, the first D-C wave needs a person.
- **Licence.** Detected with:
  - `license_audit.detect_license` for Kaggle, Roboflow, GitHub and HF;
  - Mendeley `data_licence`, Zenodo `metadata.license`, and the Weed-AI and mediaTUM record licences.

  It is recorded **per image**. The P6 policy applies. There is no Roboflow auto-sync; the old job script's `AUTO_SYNC=1` would push to a public project.
- **Guard.** `inc2.guard.GuardV2` is one implementation, called here and again, fail-closed, in Step 1 (§3.3). In order, first hit wins, each counted:
  1. unhashable;
  2. `near_eval_v2`: ≤ 6 bits of dev, test or ImageWeeds;
  3. `near_eval_variant`: any of the 8 flips or rotations (`funnel.leak.dhash_variants`) ≤ 6 bits;
  4. `base_copy`: ≤ 6 bits of base v2;
  5. `exact_dup` against the Step 1 seen index;
  6. near-duplicate of earlier intake batches.
- **[review] Augmented copies need the embedding detector, not only dHash.** The eight dHash variants catch flips, rotations and re-encoding. They miss the augmentations that Roboflow and Kaggle re-uploads of cwd12 usually carry (crop 0–20 %, shear, brightness, blur, letterbox). The funnel's calibrated copy detector (`funnel/leak.py`: DINOv2 whole-image cosine ≥ the calibrated threshold, or any dHash variant ≤ 6 bits) is built for exactly these families. `GuardV2` therefore gains a seventh check, `near_eval_embed`, run in `step1_stream` (one GPU job per batch already exists) against dev + test + ImageWeeds for every row whose source is not provenance-cleared. A source is provenance-cleared only when it is a primary release from a lab outside the evaluation labs (MFWD from TUM, PAGS8 from TAMU, MH-Weed16 from its authors). Every other source, including every Roboflow Universe and Kaggle re-upload, is held with `hold_until: h6_scan` until the scan has run.
  - The leak is self-reinforcing: a copy of a dev image raises dev, so the gate prefers exactly the increments that leak. Gate acceptance is never evidence against a leak.
  - **[L-9(c)] The threshold these rows are judged by** is the v2 calibration splits v2 writes beside LOCK v2 (`embed_calibration_v2.json`: per-image false positives on hard same-domain negatives, never below the funnel's 0.8256; §4.2 step 5b). Intake binds it to every row it holds `h6_scan`, and `step1_stream`'s copy scan applies it in place of the per-pair threshold.
  - The funnel campaign runs on its own (its census passed on 2026-09-28 and its leak step, job 47245786, was submitted the same day), but its `leak_v1` calibration may arrive late. The stream therefore runs `funnel.leak.calibrate` and `detect` as a library under its own output directory (`INC_DIR/intake/leak/`, seed prefix `stream/v1/leak/`), with the funnel's pre-registered recall and false-positive gates. It never writes the funnel's directory. If the funnel's `leak_v1.json` exists and passed, the stream may reuse its threshold (recorded by sha256).
- **Capture group and lab group per source.**
  - Capture group: tray (MFWD), video (MH-Weed16), session or date. Increments draw whole groups.
  - Lab group: LuLab for CottonWeedDet3 and CottonWeedID15; NDSU; TAMU for PAGS8; TUM for MFWD.
- **Outputs: `INC_DIR/intake/<batch>/`.**
  - `sources.json`: source id, provider, ref and version, licence with evidence, lab group, `exhaustive_labels`, the class list, the class map with provenance, and fetch bytes and checksum.
  - `decisions.jsonl`: every candidate and file decision, kept or rejected, with its reason. A "+0" can never again be silent.
  - `manifest.jsonl`: rows with `key, image, sha256, label, label_sha256, source, licence, research_only, lab_group, capture_group, dhash`.
  - `guard.json`: counts per reason and the sha of each index used.
  - `summary.json`.
- **Registry.** Each source is registered through `registry_lock.update_registry` with `annotation: "intake_v1"` and `status: "intake"`. That annotation is not in `verify.VALID_ANNOTATIONS` (`verify.py:117`) nor in `mega_trainer`'s valid annotations, so both pinned v1 Step 1 and `mega_trainer` skip these sources, with no edit to pinned code.
- **Amendment 2026-10-03: continuation shards.** A capped intake batch (§7.2, "Intake cap") no longer strands the images it deferred: the next intake of the same fetch record takes them, shard by shard, until none is deferred.
  - Why: zenodo_15808623 (SIU Weed Growth Stage, one 49.7 GB zip, 230,899 boxed images, 174 class sets, one box per image, every image its own capture group) committed `i0004_zenodo_15808623` at 17:29Z with 39,958 rows of the 40,000 it took, and deferred 190,899. An intake of a committed fetch record was a no-op, and the extracted tree was deleted after the commit, so those images could never be taken. The intake took 5,151 s: about 19 images/s in the per-image loop, plus about 45 min of fixed cost (hashing the blob, extraction, normalising 230,899 labels).
  - `collect.intake` (step 1b):
    - When the committed batches of the fetch record (same fetch sha256, `shard_state`) recorded images `deferred` that none of them decided since, the intake takes the next shard instead of the no-op. Its candidates are the items no earlier batch of the fetch decided (any decision but `deferred`), drawn by `cap_items` with the same seed (the fetch sha256), at most the cap. So the shards partition the source, and a rerun takes the same shard. When nothing is deferred, the intake is the no-op it was; it also removes a tree that a run killed between the last shard's commit and the tree's removal left behind.
    - Each earlier batch's `decisions.jsonl` must hash to the sha256 its `summary.json` records, or the shard is refused (`CollectError`). A deferred image that a tree extracted afresh, from blobs hashing as recorded, does not hold is rejected `not_in_source`, so the shards always end. When a kept tree lacks a deferred image, the tree is removed, the blobs are hashed and the tree is extracted again before anything is decided (`shard.tree_lacked`); a kept tree that lost files (a partial removal, a damaged disk) never turns images into rejections.
    - At intake, the shards of one fetch are judged as one batch. The earlier shards' images seed the exact-duplicate check (`exact_dup_intake`, as within a batch) and are left out of the guard's `near_dup_intake` index, which still holds every other batch. Every other check of `GuardV2` (evaluation, base, L-5, the copy-scan binding) and D28-v2 run as for any batch. Measured read-only on the cluster on `i0004_zenodo_15808623`: 18,504 of its 39,958 rows (46 %) lie within 3 dHash bits of another row of the same batch, and only 37,943 dHashes are distinct. Had the earlier shards stayed in the index, `near_dup_intake` would have refused about half of every later shard, which one uncapped batch would have kept.
    - This holds at intake only. Step 1's `near_consumed` (3 bits from a consumed row) and the cutter's pool radius (`POOL_BITS`, 6 bits) judge each shard against what the stream has consumed by the time the shard is admitted, as for any batch: a later shard's frame near a consumed frame of an earlier shard is refused at admit or never cut, where one uncapped batch would have put it in the same draw unit as its neighbour. These rules keep increments disjoint and are not changed here. So images taken in at intake are not all trained on; D21 judges a source by its admitted target boxes, which count this.
    - The extracted tree is kept while images remain deferred and removed by the batch that leaves none. The tree's `.complete` marker now records the fetch sha256 it was extracted from; only `intake`'s `materialise` writes it, and only after `check_blobs` hashed every blob. A continuation shard over such a tree, for the same fetch record, reads no blob (the tree is reused), so its blobs are checked for presence and size only, not hashed again; `fetch.json` is still hashed. A tree whose marker names no fetch record (written before this amendment) gets the full hash, as before.
    - The images an hf_parquet reader writes (`normalize.read_hf_parquet`, under the work tree) are each written as a new file (a temporary file renamed over the path), and a file already holding the same bytes is left alone. A committed batch's images are hard links to those files, which now outlive the commit, so a later shard's read never writes into a committed image, even when a write fails part-way.
    - Every batch judges at least one boxed image before `intake_max_seconds` can defer the rest, so a shard always moves its fetch on.
    - `summary.json` gains `shard`: its number `n`, the fetch sha256, the earlier batches, the candidates, `deferred_remaining`, whether the tree was reused or extracted (and `tree_lacked`), the arrival check made, and `class_map_sha256` (`collect.intake.class_map_sha`, the source-id-to-intake-id mapping); `class_map_changed` names the two sha256s when it differs from the shard before (names or the config changed between shards), and the `intaken` event carries it too. `yield.images_deferred` equals `deferred_remaining`. `batches.jsonl` rows and the `intaken` event record `shard` and `deferred_remaining`. `guard.json` records `same_fetch_shard_images`.
    - The registry entry keeps each batch's rows (`intake_batch_rows`); a batch registered again (a run killed after its registration and before its `summary.json`, redone under the same name) replaces its count instead of adding to it.
  - `budgets.intake_max_images` is 50,000 (was 40,000). That is Step 1's admit cap for one intake batch (`inc2.step1_stream.BATCH_CAP`, 50,000 rows), so one L17 admits one shard; the smaller of the two is 50,000 either way. `intake_max_seconds` stays 6 h. The 190,899 deferred SIU images are 4 shards (50,000, 50,000, 50,000, 40,899).
  - `inc2.base3` reads only committed intake batches (`summary.json` present) and only the first shard of each fetch record; a continuation shard is recorded in summary.json's inputs as not read. Base v3 revision 2 was pre-registered on SIU's first batch (40,000 frames taken), so whenever the shards are committed (before the build, during it, or before a rebuild after a failed build) base v3 stays on that batch.
  - The stream (`inc_autopilot.stream`, `diagnose_stream`):
    - The fold records, per batch, what it left deferred and the batch of its fetch before it (`shard_log`), and the latest batch's count as `intake_deferred` (`shard.deferred_remaining`, else `yield.images_deferred`: a batch committed before shards is the first of its fetch).
    - L17 admit's completion records the batch in `admitted_batches`. A source whose intake left images deferred, or that has a committed shard not yet admitted, becomes `shard_pending`, not `admitted`, recorded as `source_status`. The L16I step's attempt count rises, so the next intake has an id of its own, and the source's failure count restarts (failures count per shard).
    - Whether a source is part-way through its shards is read from these facts (`diagnose_stream.mid_shards`: admitted batches, and images deferred or a batch not admitted), never from its status alone. A failed shard, a collector's hold and its release, the collector's ledger adopted, and a person's reopening of a source closed part-way all end in `shard_pending`; `precheck` (D20) and the L16 proposal refuse such a source whatever its status says. A collector's hold folded in the tick a shard failed is kept until the collector releases it.
    - DPIPE: an admit takes the source's oldest committed batch not yet admitted (so a shard committed by a job the stream saw fail is admitted, never skipped). Then, after every other pipeline step, a `shard_pending` source gets L16I, its next shard. It stays shard_pending until its last shard is admitted, then it is `admitted`.
    - **Order.** The DATA lane takes, first to last: DPIPE's pipeline steps (a first batch's intake and admit, every shard's admit); DR0's L17 item (eval-hits); the continuation shard (L16I on a `shard_pending` source, DPIPE's last item); D20's new sources. DPIPE holds the shard (silent, with the reason, `shard_waits`) while:
      - DR0 has a DATA item due (L17 eval-hits, Step 1's one-time jobs, the probe). DR0 proposes L23V only when it has no DATA item due, so the eval-hits it needs first runs before any shard.
      - An E1 arm that requires base v3 is not built and its build (L23V) is due (base v3 missing with R0 complete, so DR0 proposes it) or running. Every other state of base v3 (built, failed, over walltime, ended without a complete summary) lets the shards run, so a shard never waits without an end; since base v3 reads only first shards, the hold is about order, not about what base v3 reads.

      D20 waits likewise while DR0's L17 item is due, as it already waited for Step 1's one-time jobs: D20 comes before DR0 in the diagnosis order, so a fetch would otherwise take the DATA lane whenever Q is low and eval-hits, L23V and the shards behind it would wait for as long as there are candidates (live: eval-hits due from 06:56Z while D20 took the lane three times). While a shard waits, the DATA lane serves the other diagnoses (a fetch, discovery), and D29 never calls collection exhausted while a source is part-way through its shards.
    - D21 judges a source only once its intake has nothing deferred (known: the state's count, else the latest summary in the evidence; an unknown count is not judged), so its yield is the whole source's admitted boxes over its bytes; the zero-yield stop-loss records the yield once, at the same point. A source's entry in the zero-yield run leaves it as soon as the source has admitted target boxes (a failed fetch before its intake, or a source still taking shards, did not end with zero yield), and a source closed after admitting target boxes is not recorded as a zero-yield failure.
    - A failed continuation intake counts as for any job (the source's failure, the lane's step), and the source stays `shard_pending`; three failures within one shard close it. A failed shard admit counts on the lane and the step (L17's params name the batch, not the source), and the batch is admitted again under a new id.
    - The stream decides on what a snapshot shows, never on what it lacks (a snapshot whose summaries failed to read decides nothing). A continuation intake that made no new batch is decided once a snapshot holds the collector's ledger and the latest summary: when the collector's last batch left nothing deferred the source is complete (`admitted`, `shards_done`, its deferred count 0); otherwise the shard made no progress. A shard that left as many images deferred as the shard before it made no progress too. Such a source is closed with a card (`stream reopen` resumes its shards), so a shard is never proposed in a loop.
    - A source the stream admitted before this amendment (no `admitted_batches`) gets its batches as admitted at the first snapshot that holds its latest summary, and waits for its next shard when that summary shows images deferred. An admit that completes before such a snapshot leaves it admitted as before, and D21 does not judge it while its count is unknown. Live, `i0004_zenodo_15808623`'s admit was running at 18:28Z; either path makes the source `shard_pending`.
    - The limits stay as they are: `jobs_per_day` (L16: 12; L17: 6) and `in_flight` are the executor's. [amended 2026-10-04: the per-day job counts are removed; `in_flight` stays (§6.6, "Amendment (2026-10-04)").]
  - Revision before deploy (2026-10-03). A review of the first version of this amendment found, and this text now describes the fixes for: an hf_parquet shard rewriting committed images in place; `not_in_source` decided on a kept tree that lost files; a tree left after a kill between the last commit and its removal; a shard that could judge nothing and loop; the class map not recorded per shard; the registry counting a redone batch twice; base v3 able to read later shards (a failed build rerun by a person, or a build reading an uncommitted manifest); the stream deciding shards on a missing summary (an orphaned batch, or a source left admitted with images deferred); a detour of the status (a hold and release, a reopening) turning a source part-way through its shards into a candidate; failures accumulating across shards and closing a source that admitted data as a zero-yield failure; D20 taking the DATA lane while eval-hits was due; a stale zero-yield entry kept for the whole shard period; no progress check between shards; and shards waiting on a base v3 state with no end. Each has a test that fails without its fix (checked by reverting each fix alone).
  - Open items:
    - The kept tree of a source that is closed or quarantined while shards remain (up to the size of its extracted archives, about 50 GB for SIU) is not removed by the stream; a person removes `INC_DIR/intake/work/<source>/`.
    - A source whose fixed intake cost alone exceeds `intake_max_seconds` moves on by as little as one image a shard; summary.json's `intake_time` shows it.
  - Tests:
    - `tests/test_collect_intake.py`, `test_continuation_shards`: a cap of 4 over 10 boxed images in five capture groups takes 4, 4 and 2. The earlier shards keep the tree and the last removes it, and a 4th intake is the no-op. A shard killed in its loop commits nothing, and its rerun takes the same shard. Keys never collide. An exact copy across shards is `exact_dup_intake`; near copies across shards are kept. A changed blob of the same size passes over a tree that names the fetch record and refuses over one that does not.
    - `test_shard_edges`: changed decisions refuse the next shard; a kept tree that lost a deferred image's file is extracted again (blobs hashed) and the image taken; a deferred image a fresh extraction lacks ends as `not_in_source`.
    - `test_shard_files`: an hf_parquet source's shard 2 writes into no committed image's file (each keeps its inode, time and bytes), and a write failing part-way leaves no partial file; a run killed after the last shard's commit leaves the tree, which the next intake removes; a run killed between its registration and its commit counts its rows once; the class map's sha256 per shard, read from `sources.json` for a batch without a shard record, and a change flagged.
    - `test_intake_cap`: past the time budget the first image is still judged, and each continuation shard moves on.
    - `tests/test_inc2_base3.py`, `test_intake_shards`: a continuation shard and a batch without `summary.json` are recorded and not read; the first shard of each fetch is read.
    - `tests/test_stream_ap_shards.py` (stream world):
      - intake with images deferred → admit → shard_pending → L16I under a new id → admit → admitted, with no failure, no stop-loss step, no fetch, and the yield recorded once;
      - eval-hits before the shard; L23V before the shard, and D29 silent meanwhile; with Q low and an open candidate, eval-hits, then the shard, then the new source; a base v3 build ended without a complete summary does not hold the shards;
      - a failed shard; failures counted per shard; a third failure in one shard closes the source but not as a zero-yield failure; a stale zero-yield entry leaves the run at the first admitted boxes;
      - a source admitted before this amendment (D21 does not close it on its first batch);
      - a continuation that made no new batch: complete (D21 then judges it) or no progress (closed with a card); a shard that left as many deferred as the one before it (closed with a card);
      - summaries that fail to read after a continuation intake (its batch admitted once they read again); an admit completing while they fail for a source recorded before this amendment (decided once they read again, D21 silent meanwhile);
      - the collector's hold and release, and a person's reopening, end in shard_pending without a fetch; a source part-way through its shards whose status reads `candidate` is refused by D20;
      - `deferred_left` and D21 on state rows (unknown, known, shards done).

### 3.3 Incremental Step 1 with per-box admission — `inc2/step1_stream.py`, `inc2/mask.py` (new, group C)

**Why a new module.** The pinned `verify` and `select` cannot append:
- every path is a v1 constant (`common.py:34-38`, `verify.py:177-198`);
- `cmd_pool` loads the v1 never-train index, which rejects 3SeasonWeedDet10;
- any pool change voids the embeddings and the verifier;
- `Crops` requires crop ids to start at 0.

`step1_stream` uses about 45 pinned functions of `verify`, `select`, `funnel.leak`, `near_dup` and `semisup_labeler` unchanged, as a library.

**Runs on.** One GPU-shared job per batch (`run_inc2_stream.sh admit --intake <batch>`): ingest, then embed, then judge. The job:
- checks the outer against the nested module hashes of `verify.py`, `select.py`, `funnel/recover.py`, `funnel/leak.py`, `cwd12_species.py`, `near_dup.py`, `semisup_labeler.py` and `mega_trainer.py`;
- takes `step1_stream/.lock`, so there is one writer.

Batching: collect until 2,000 images or 24 h, and cap a batch at 50,000 images (about 23 min of embedding).

**Stages:**
1. **Pins.** Checked against `STREAM.json`:
   - the v2 LOCK and index sha;
   - the verifier file hashes;
   - the embedder `hf-hub:imageomics/bioclip-2+fp16` (dim 768);
   - a **canary**: 64 fixed train_core crops must re-embed at cosine ≥ 0.999 to the v1 shard rows, with identical verdicts.
2. **Inputs.**
   - Intake batches.
   - New paths in registry slugs: the per-slug listing minus the processed set. The v1 processed set = the keys of `step1/cache/dhash/<slug>.json`.
   - **[review] Registry slugs bypass the collector's gates.** A slug registered by anything else (the old harvest, a dashboard upload, a manual download) has no licence record, no lab group and no copy scan. Only slugs present in the v1 pool are read here. Every row read from them gets its licence from the funnel's card index for that slug (read only), or else `hold_until: licence`. Its lab group comes from the v1 `pool_summary.json` evidence: a slug with any `near_eval_by_split` hit or `cwd12_copies > 0` (for example cwp10 and vanpe) is lab group LuLab, and its rows are held with `hold_until: h6_scan`. A slug that is not in the v1 pool is refused here and must come through `collect intake`.
3. **Join.**
   - Intake rows use their recorded class map.
   - Registry slugs use `verify.class_join`, followed by a versioned override table (§3.3, Versioned events).
   - A slug whose join changed is refused and becomes a `rejoin` event.
4. **Guard.** `GuardV2`, plus `near_consumed` (≤ 3 bits of a base or consumed image).
   - An exact duplicate whose labels disagree with its kept twin goes to `held_join_conflict`: both images are held and a rejoin is queued. First-wins is never used.
5. **Crops, embed, judge.**
   - Local crop ids plus a recorded global offset. v1 occupies [0, 564,686).
   - Embedding uses `verify.BioclipEmbedder(amp=True)`, `_embed_images` and `_save_npz`.
   - Judging uses the **frozen** verifier copied to `step1_stream/verifier/v1/` (`meta.crops_sha256 = d8188b16…`) through `Verifier.judge_features` and `verdicts`.
6. **Admit (D-B).** Each image takes the first rule that applies:
   - **whole:** `image_verdict == admitted`, unchanged.
   - **masked:** not admitted, and ≥ 1 target box is VERIFIED.
     - Kept: VERIFIED target boxes, and OtherPlant boxes that are OTHER_OK or SMALL.
     - Masked: target boxes that are CONFLICT, UNKNOWN, FAILED or SMALL; OtherPlant boxes that are CONFLICT or FAILED; every unmapped (13) box.
   - **refused_overlap:** a masked box has IoU ≥ 0.5 with a kept box. The image goes to the human queue.
   - **not admitted:** everything else. Conflicts go to the human queue.

   `mask_except(image, mask_boxes, keep_boxes)`:
   - fills (masked rectangles rounded outward) minus (kept rectangles rounded outward) with the image's mean RGB, on the EXIF-transposed image;
   - writes a lossless PNG `<key>.<sha16>.png`;
   - records `masked_area_frac` and, for each masked box, the share of it that stays visible inside kept boxes;
   - with no overlap, is byte-identical to `recover.mask`.

   `recover.mask` itself is not used: it fills every masked rectangle whole, so it erases verified pixels where boxes overlap (`funnel/recover.py:111-144`). Verdicts come from the unmasked original. The never-train check runs on both the unmasked and the masked dHash. [review] It runs on all eight variants of both. `mega_trainer._dhash` hashes the stored pixels without EXIF transposition, while the masked PNG is written EXIF-transposed. So the masked copy of an EXIF-rotated original matches the index only through a rotation variant.
7. **Known truth.** Base copies box-matched to expert labels (train_core, tsw22, tsw23) give each batch a `verified_precision` (`verify._match_box`, `_metrics`).
8. **Place.** l1/l2 from a frozen reference pack (`reference_v1.npz`: species prototypes, the LOO percentile scale, OtherPlant prototypes and l1/l2 centres, built once with `select.species_reference` and `hierarchical_kmeans`). Near-duplicate groups use the unmasked dHash at 3 bits.
9. **Commit.** `batch.json` is written last. Then `ledger/batches.jsonl` and `queue/queue.jsonl` are appended.

**Queue row** (`queue/queue.jsonl`, append-only):
- `key, batch, source, group, capture_group, l1, l2`;
- `kind` (cwd12 | other), `score` (recorded only);
- `species_boxes[12], other_boxes`;
- `admission` (whole | masked), `n_masked`;
- `evidenced`;
- `lab_group, licence, research_only`;
- `prior` (for example `realloop_v1:V2:REJECT(recipe)`);
- `hold_until` (null | `funnel_F9` | `h6_scan` | `licence`) [review: `licence` added for registry-slug rows without a card licence];
- `verifier, stream_pins_sha`.

**Status** (`step1_stream/status.json`, read by the autopilot):
- new paths pending;
- batches by state;
- admitted images and boxes per source and per species;
- masked and refused-overlap counts;
- known-truth precision per batch;
- versions.

**Versioned events** (never per batch):
- **`rejoin --slug`.** Uses overrides from recorded resolutions only: funnel `name_status_v2.json` entries with `status_v2 = target_synonym`, and accepted maps in `class_maps.json`. MH-Weed16 ids 5 and 12 are the case in hand. No new embeddings are needed.
- **Verifier refit (verifier v2).** R4 (card X11). Proposed triggers, to be pre-registered in thresholds.json:
  - a batch's known-truth `verified_precision` is shown under 0.99 on ≥ 30 matched verified boxes: P(Binom(n, 0.01) ≥ errors) < 0.01 (`known_truth.min_precision`, `known_truth.alpha`). Amended 2026-10-03 from "a Wilson lower bound < 0.99": with no error that bound stays under 0.99 below 381 boxes, so batch b0001 of the first out-of-domain reading, 284 of 284 correct, raised X11. A batch with no error never fires; on 284 boxes 8 errors (97.2 %) fire and 7 do not; or
  - a species has ≥ 100 new target-labelled boxes with an unknown share ≥ 0.5.

  After a refit, every unconsumed queue row is re-judged from its stored embeddings.
- **Reference pack v2.** Adds the 3SeasonWeedDet10 crops: about 20k crops, about 100 s of GPU.
- **Any change to the splits version or the base** means a new stream version of `step1_stream`.

**One-time jobs:**
- **`bootstrap`:** pins, the verifier copy, the reference pack, and the seen, key and group indexes seeded from v1.
- **`knowntruth`:** see R1, §10.
- **`backfill` → batch b0000:** D-B over the existing pool.
  - Inputs: the v1 files read through the v1 constants, which is correct for v1 files.
  - Candidates: the 457 non-admitted images that hold verified target boxes.
  - Output: the v1 increment pool (96,088 rows) plus up to 457 masked rows. realloop_v1's draws are tagged `prior`, and base B's 878 are marked as base.
  - [review] The same lab-group and licence holds as for registry slugs (Inputs, above) apply to b0000. Rows of v1 slugs with evaluation near-copies (for example cwp10: 755 pool rows; vanpe: 352) carry `hold_until: h6_scan`, and rows without a card licence carry `hold_until: licence`. The supply statement of §10 counts only rows with no hold.

**Cost:**
- about 66–75 s per 1,000 new images, of which about 27 s is GPU, plus fixed job costs (est., from the recorded Step 1 timings);
- the backfill needs no embedding; masking 457 images takes about 4–15 min (est.).

### 3.4 Increment queue and cutter — `inc2/stream.py cut` (new, group E)

**Eligible rows.** A queue row is eligible when all of these hold:
- it is unconsumed;
- it is `evidenced`;
- `hold_until` is released;
- its key and dHash are not in `stream/<sid>/quarantine_dhash.json`;
- **it holds ≥ 1 kept verified target box** (P4).

OtherPlant-only images stay queued with `kind: other`. Stream version 1 never cuts them: realloop_v1's OTHER_HEAVY step was "hurts" in the truth arm, and B's ImageWeeds drop has OtherPlant-heavy data as its leading candidate cause.

**The cut:**
- **exactly M images**;
- whole near-duplicate groups and whole capture groups; a group that would overshoot is skipped for the next one;
- the oldest admission batch first;
- **one source first**: from the oldest source with ≥ M eligible images, which makes most attribution single-source; otherwise oldest-first across sources, with the mix recorded;
- the draw uses `select.cluster_lists` and `select.draw_parts` unchanged, seeded by `stable_int(sid, segment, step)`;
- if no exact fill exists, the cutter waits for more data.
- **[review] Exact M with whole capture groups can deadlock.** `draw_parts` (`select.py:1306`) passes over any unit that would overshoot and raises `SelectError` when the units cannot sum to exactly n × m. MFWD has about 211 images per tray (4,435 images in 21 trays), and an MH-Weed16 video can hold hundreds of frames. A queue made of such groups can hold ≥ M images and still have no exact fill, so "wait for more data" never ends. The draw unit is therefore the near-duplicate group (3 bits, small). Capture groups decide only the order. When the next capture group would overshoot, it is split by near-duplicate group to fill exactly M, and its remainder is cut first into the next increment. A split is recorded in `<inc>.meta.json` (`split_capture_groups`), so attribution can see it. A cutter that holds Q ≥ M and still cannot fill raises D22 with a refusal reason, never a silent wait.
- **[review] One source first makes single-species increments.** MFWD POROL yields only Purslane and PAGS8 only Palmer amaranth, so an increment cut from one of them is one species. The warm chain's species guard compares cand with null on every species. In realloop_v1 the guard failed on PricklySida in 5 of 6 steps (deltas −0.033 to −0.057 against a 0.03 floor; `realloop_v1/ledger.jsonl`), and PricklySida is in none of the planned sources. A one-species increment shifts the class prior more than a mixed one. The cutter therefore records the increment's species mix, and when two or more sources are eligible it caps any one species at 60 % of the increment's target boxes. It falls back to one source only when one source is all there is. Whether this rule is kept is decided by Stage C (§5.1).

**Output:** `stream/<sid>/increments/<inc>.jsonl` holds `C.MANIFEST_KEYS` rows. The companion `<inc>.meta.json` records:
- sources, target boxes per species, admission kinds and n_masked;
- the queue sha and the step1_stream pins sha;
- the verifier and reference versions;
- the seed.

One increment never mixes verifier versions.

**Consumption state is owned here, not by Step 1:** `stream/<sid>/consumed.jsonl` (key, increment, final disposition).

### 3.5 Segments: the streaming chain with gate and rollback — `inc2/stream.py build|commit|rollback` + pinned `inc/driver.py`, `inc/gate.py`; executor `inc2/train.py` via `run_inc2_job.sh` (groups E and B)

"Segment" is used instead of "epoch" so that it is never confused with training epochs.

**Why segments.** The pinned driver writes `exp.json` once. It cold-trains every base from `init_weights`, finishes a chain at `k >= len(steps)`, and cannot import another experiment's weights (`driver.py:1306, 1884-1896, 2051`; `realloop.py:48`). A segment is therefore an ordinary pinned `chain` experiment, `<sid>_sNNN`.

**`exp.json` written by `inc2.stream build --stream <sid> --k K`:**
- `type: chain`; `seeds: [0, 1, 2]`; `init_weights: yolo11n.pt`;
- `base` = the pool manifest P_{s-1}, content-addressed with its sha256 (P_0 = `base_v2.jsonl`);
- `steps` = K increments, each `clean: false`;
- the recipes of §5.1 (segment 1 has two);
- `replay_mode: full`; `gate: {"flips_mode": "net"}`; the truth arm on (P3);
- `final_exams: ["dev", "imageweeds"]`. They are a subset of the pinned `C.EVAL_SPLITS`, so `validate_definition` (`driver.py:472-478`) accepts them.

**Executor.**
- Every run executes `inc2.train` through `run_inc2_job.sh`, which exports `INC_JOB_SCRIPT=<itself>`, so that in-job advances (`driver.job_script()`, `driver.py:331`) keep using the v2 executor.
- The ticker's `advance` exports the same variable. An advance without it would submit the v1 executor, which fails closed on the never-train guard.

**Within a segment the pinned gate decides every step:**
- **ACCEPT:** the increment joins T_k.
- **REJECT or HOLD:** the incumbent pointer does not move. That is the rollback.

The truth arm compares P_{s-1} ∪ D_k with P_{s-1} for each step (`clean: false` steps use a fixed base).

**Commit (`inc2.stream commit --exp <sid>_sNNN`, R1).** Deterministic from the finished segment's `report.json` and ledger:
- **P_s** = P_{s-1} ∪ the ACCEPTed increments of the chosen chain. Segment 1 has two chains, and the pre-registered rule of §5.1 names the chain. P_s is written as `stream/<sid>/pool/P_<s>.<sha16>.jsonl`, checked pairwise disjoint by key, path, sha256 and 6-bit dHash.
- **REJECT with `attribution.blame == "data"`** (`gate.py:530`) → every image's dHash goes to `quarantine_dhash.json`; it is never cut again, and a re-harvest cannot bring it back.
- **REJECT with blame `recipe`, or HOLD** → the images return to the queue **once** (`returned`); a second non-accept quarantines them as `neutral`. Such a verdict says nothing about the data, and allowing one return caps the number of chances noise gets.
- The consumed state and the ledger are appended.
- **[review] The disposition reads the failed guards, not `attribution.blame`.** The gate sets `blame = "recipe"` on every REJECT whose P_recipe ≤ 0.25 (`gate._attribute`). P_recipe was 0.0 in all 6 realloop_v1 steps and in every step of pilot_v3's full chain: the full recipe's null is always below a cold incumbent. In the ledgers, **every** full-chain REJECT of realloop_v1 and pilot_v3 carries `blame: "recipe"`, including the planted Bswap (40 % of boxes relabelled, P_data 0.00). Under R0, then, the rule above would return a planted bad increment to the queue, never quarantine anything as data, and never fire D31. Only the regression guard compares against the incumbent; P_data, the species guard and the flips guard compare cand with null under the same recipe. The disposition is the first of these that applies:
  1. **data** (quarantine): P_data ≤ p_reject (0.25), unless that step's truth arm says "helps";
  2. **species** (return once, then `neutral`; counts toward D33): the species guard failed;
  3. **flips** (return once, then `neutral`): the flips guard failed;
  4. **recipe** (return once; not counted as the one return while D30 holds TRAIN): only the regression guard failed.

  `attribution.blame` stays recorded and reported beside this reading. Applied to realloop_v1: s01 and s03 data (P_data 0.11); s02, s04 and s06 species; s05 flips. Applied to pilot_v3's full chain: Bswap data, Breal species. D30 fires on neither, which agrees with §1.3 (setting null = inc changes none of the six realloop_v1 verdicts).

**Rollback across segments (`inc2.stream rollback --to P_c`, L21).**
- The current pool pointer returns to the last good milestone pool P_c.
- The increments accepted since P_c are marked `suspect`, excluded from cuts, and card X4 (leave-one-source-out bisection) is raised.
- **[review] A person-only X4 idles the data it suspends.** With supply as the binding limit, the (at most 4) suspect increments would otherwise wait for a person indefinitely. The platform therefore runs the bisection itself as a MAINT item (new lever L27, R3 envelope, ≤ 1 per rollback): one cold 3-seed arm P_c ∪ {increment i} per suspect increment, each compared with the P_c milestone seeds by `gate.truth_detail`. Cost ≤ 4 × (4.2–4.9) GPU-h. "helps" → the increment returns to the queue as a new cut. "hurts" → quarantined as data. "neutral" → quarantined as `neutral`. X4 stays raised only for what the bisection cannot separate, such as an effect that needs two increments together.

**The incumbent across segments.** Segment s+1 starts from its own cold base on P_s. The model is retrained cold on the accumulated data at every boundary, and only the data carries over. This is also the free consolidation check of §5.5.

**Stream ledger.** `stream/<sid>/ledger.jsonl` is append-only and hash-chained: every line carries the sha256 of the file before it. Its events are:
- `cut`, `build`, `commit`, `return`, `quarantine`;
- `rollback`, `milestone`, `code_change`, `withdraw`.

### 3.6 Milestone consolidation — `inc2/baseline.py` via L20 (group B builds, group E orders)

A milestone is a `baseline` experiment `<sid>_mNNN`:
- **5 cold seeds** on the current pool P_s;
- finals on dev, imageweeds and **test**;
- one extra scorer job scores the current chain incumbent on the same three exams, as a secondary number.

**Comparison with the previous milestone** (milestone 0 = the B_v2 baseline of R0), on dev only:
- `gate.truth_detail` (P over 25 seed pairs, with the species guard);
- a one-sided 5 v 5 permutation test at p ≤ 0.025, as FUNNEL_AUDIT H10a.

A verdict of "hurts" fires D25 → L21.

[review] Only the overall test rolls back. `truth_detail` calls "hurts" when the species guard alone fails. With 5 cold seeds, PricklySida (42 dev boxes) and CutleafGroundcherry (31) can move by more than the 0.03 floor through noise and through the dilution D-A already causes. A pool that raised overall dev would then be rolled back, and every increment since P_c would be suspended. So L21 fires only on the one-sided 5 v 5 permutation test (p ≤ 0.025, mean(new) < mean(old)). A species-guard-only "hurts" fires D33 and card X17, is reported in the milestone table, and does not roll back.

### 3.7 Reports — `inc2/stream_report.py` (group E)

- **Per segment:** `report.md`/`report.json` from the pinned `report.py` conventions, with exams dev and imageweeds.
- **Per stream:** `stream/<sid>/report.{md,json}`. It contains:
  - a timeline of every increment: sources, target boxes per species, gate verdict, P_data, P_recipe, blame, truth verdict, disposition;
  - pool size and dev per segment;
  - the milestone table: dev, imageweeds and test as mean ± sd over 5 seeds, the chain incumbent, and the **gap to 0.90**;
  - per-source yield (§7.4);
  - SU spent against the envelope.
- **At every milestone, a RESEARCH_LOG entry:** what changed, why, and how it was verified.

### 3.8 State layout and single writers

| Directory | Writer (one at a time) | Readers |
|---|---|---|
| `INC_DIR/splits/v2/` | `inc2.splits` (group A), once, then 0444 | everyone |
| `INC_DIR/intake/` | `collect` (group D), `intake/.lock` | step1_stream, autopilot |
| `INC_DIR/step1_stream/` | `inc2.step1_stream` (group C), `.lock` | stream cutter, autopilot |
| `INC_DIR/stream/<sid>/` | `inc2.stream` (group E), `stream.lease` | autopilot, reports |
| `INC_DIR/<sid>_sNNN/`, `<sid>_mNNN/` | pinned driver (advance lease and generation fence) | stream commit, autopilot evidence |
| `~/.round_scheduler.json` → `campaigns.<sid>` and the campaign state | autopilot (group F) | dashboard |

`evidence.RESERVED_DIRS` (`evidence.py:136`) gains `stream`, `step1_stream` and `intake`. `ALLOWED` gains `stream/*/queue_summary.json`, `stream/*/ledger.jsonl`, `step1_stream/status.json` and `intake/*/summary.json`.

### 3.9 Shared inputs with the funnel audit

The funnel audit is a live, pre-registered campaign on base B and splits v1. Its realloop_v2 (F10) uses base B, `inc.realloop` and its own arms. The two lineages train different models, and their conclusions are never pooled. Three inputs are shared:
- **The 457 vetoed images.** They are the funnel's REC-VETO arm (FUNNEL_AUDIT.md:691, 952). The stream's b0000 masked rows carry `hold_until: funnel_F9`.
  - When F9 writes `INC_DIR/step1_r1/domain_dev.jsonl`, the domain-dev keys are excluded from the stream permanently, and the rest are released.
  - The owner may release them earlier; stream models then lose H10d comparability, and only they do.
- **MH-Weed16 ids 5 and 12, and the NDSU sources.** The funnel owns their resolution: H3a, the geometry match of HF ids 0–14 against card ids 0–15, and DEC-3's *Ipomoea* spp. policy. The stream runs `rejoin` only on a resolution the funnel has recorded.
  - The MH-Weed16 R-F refetch is the funnel's `fetch.refetch`. The collector never registers that source a second time.
- **`recover.mask`.** Its overlap defect (§3.3) also affects the funnel's R-V at F9. It is reported to the funnel campaign as a dated amendment item; this contract does not edit `recover.py`.

---

## 4. Splits v2 and the new baselines

### 4.1 Design: only the training side changes

- **Byte copies of v1.** dev, test and imageweeds are copied from v1, and each must match the v1 LOCK: `a5c904cd…`, `31ba7650…` and `357936e3…`.
- **train_core is v1's minus the L-8 drops.** `splits/v2/train_core.jsonl` is v1's train_core (`242ef6b9…`) with the rows of `train_core_variant_drops.jsonl` removed. Every other line is byte-identical and in the same order, so anyone can re-derive it from the v1 file and the list. With nothing listed it is a byte copy.
  - **Why train_core.jsonl itself drops them**, instead of keeping v1's bytes for provenance beside a separate training view: L-8 removes those rows from every v2 training manifest, and train_core.jsonl is one (the canary and B0 ∪ tsw train it). A byte copy that still held a transposed near-copy of a test image would be a manifest every training run refuses (`inc2.train` checks the 8 variants). Provenance is kept by LOCK v2 instead: `derived_from.train_core` records v1's sha256, the list's sha256 and the rule, and `lock` and `verify` re-derive the file.
- **The v1 scorer, unchanged.** Every v2 model is scored by the unchanged v1 scorer (sha `18c00837…`), which reads the v1 LOCK and `exams/v1`.
- **Consequence.** The score stamps the gate compares (`gate.SHARED_KEYS`) are identical across v1 and v2, so B0, B and realloop_v1's dev, test and ImageWeeds numbers stay directly comparable. There is no new scorer, no new exam directory and no relock of v1.

### 4.2 Build steps — `inc2/splits.py build|lock|verify` (group A), output `INC_DIR/splits/v2/`

1. **Precondition:** `inc.splits.verify()` returns [], a full re-hash of v1.
2. **The three byte copies** (dev, test, imageweeds) **and v1's train_core,** each refused on any mismatch with the v1 LOCK. train_core is written minus the L-8 drops (step 5a).
3. **`tsw22.jsonl` (1,915 rows) and `tsw23.jsonl` (1,784 rows),** built from the v1 `ood22`/`ood23` rows:
   - keys `tsw22__<stem>` and `tsw23__<stem>`, so no training row carries an exam key;
   - `source = 3seasonweeddet10/data2022|data2023`;
   - `session` = the stem minus the frame number;
   - labels = the v1 converted files, byte-copied to `v2/labels/tsw2x/`, so `label_sha256` is unchanged;
   - the Zenodo record's licence field is recorded.
4. **Drops, each recorded:**
   - any row ≤ 6 bits of dev, test or imageweeds, **or of any of its 8 flips or rotations**, which v1 never checked; 0 are expected;
   - the one ood23–ood22 near-duplicate pair: the tsw22 row is kept;
   - the 31 cwd12 twins and the 2 rotated-frame images, which stay out.
   - **[review] session overlap.** cwd12 itself holds frames of the capture session `20220129_CanonEOS4000D_EO`: 26 in train, others in valid, which is part of test (v1 `summary.json`, `train_sessions` and `exam_cwd12_copies`). data2022 holds the same session, and the 31 twins are the frames that are byte-identical. Every tsw row whose session (stem minus frame number) equals a cwd12 session is counted per split in `splits/v2/summary.json`. A row whose session is a **dev** session (the 8 of v1 `summary.json` `dev.sessions`) is dropped, because dev is session-held-out and a same-session frame would leak into every gate decision.
   - **[L-9(a)] test sessions too.** A row whose session is a **test** session is dropped as well (`test_session`): same-session frames are a leak risk. The first rule kept and counted such rows, because cwd12's own split is random and train_core already shares sessions with test. Rows sharing a session with train_core are kept and counted. On the cluster, tsw22 (97 sessions) and tsw23 (8) share no session with dev, test or train_core, so this drops nothing there.
5. **Base B's 878.** The same 8-variant check runs against dev, test and imageweeds. **A hit refuses the build and is an R4 incident.** It would void B's and realloop_v1's results, which is the funnel's H6(b) stop rule.
   - **[L-9] How the scan is applied** (which rule judges which row, the threshold, the funnel's list) is step 5b.
   - **[review] The dHash check is not enough for these 878.** 812 of them come from two sources the v1 pool already caught copying the evaluation splits: `rf_karthikeya-c8pvy__weed-detection-cwp10` (544 selected; v1 dropped 200 cwd12 copies and 134 near-eval images, 120 of test and 12 of dev) and `rf_zig-zag-lnodr__weed-detection-vanpe` (268 selected; 84 cwd12 copies, 53 near-test, 6 near-dev) (`step1/pool_summary.json` `per_slug`). A re-upload that held exact copies very likely also holds augmented ones beyond 6 bits. The funnel's own H6(a) quarantines such a source as a whole once any copy is found. So before `lock`, `inc2.splits` runs the embedding copy detector of §3.2 (`near_eval_embed`) over base B's 878 and over tsw22/tsw23, against dev, test and ImageWeeds, and writes the result into LOCK v2. A hit is handled by the H6(b) rule above. Whether a hit also removes the whole source's images from base v2 (as H6(a) would) is an owner decision (R4). The funnel campaign is paused, so this scan cannot wait for its F4.
   - **[review] Licences.** Base B's 878 and the v1 increment pool were collected without a licence record (the v1 Step 1 code holds none). Their licence is read, per source, from the funnel's card index (`funnel/fetch.py` records the Roboflow, HF, Mendeley and Zenodo licence fields) and written into `base_v2.jsonl`'s provenance. Unresolved → `research_only: true` with `licence: unresolved`, recorded as an owner-accepted exemption for the base (D-A named these images), never silently.
5a. **train_core against the 8 variants (decision L-8, 2026-09-29).** v1 compared the stored dHash only, so each train_core row is checked against dev, test and imageweeds under its 8 flips and rotations.
   - **A variant hit (`near_eval_variant`) is dropped, not refused.** The row leaves base_v2 and `train_core.jsonl`. It is listed in `splits/v2/train_core_variant_drops.jsonl`: its v1 manifest row, its dHash and 8 variants, the reason, the match (evaluation split, key, bits, variant) and `decided_by: L-8`. `summary.json` `train_core_variant_drops` records it as an incident: count, cap, keys, matches, the list's sha256, and the note that B0's, B's and realloop_v1's training sets held it and are not re-run.
   - **Limits, still R4.** More drops than max(1, ⌊0.5 % × |train_core|⌋) (15 of the real 3,049) refuse the build. So does a hit on the stored dHash itself: v1 compared exactly that, so v1 and this build would disagree, and L-8 covers flips and rotations only. Nothing is written before either check passes.
   - **De-duplication keeps the dropped image.** The earlier-image index of steps 4–5 still holds every v1 train_core image, so a tsw or base B near-copy of a dropped image is dropped as `near_train_core`, never kept in its place.
   - **The real case** (build job 47257533, which refused under the rule this step replaces): `train_core__20210909_NIKOND3300_YL_91`, transverse, 5 bits from `test__20210910_NIKOND3300_YL_160`.
5b. **The embedding threshold and the copy rules (decision L-9, 2026-09-29).** The second real build (job 47259471) refused on 7 of base B's kept images, 6 of them through the funnel's list against ood23 and 1 against ImageWeeds at cos 0.828; at the funnel's threshold 0.8256 the scan flagged 805 of tsw22's rows, every one a scene of another capture session (L-9's reason, §2.6).
   - **The v2 calibration** (`inc2/embed_calibration.py`, `splits/v2/embed_calibration_v2.json`, sha256 in LOCK v2):
     - *Base.* The calibration the scan used: the funnel's passed `leak_v1.json`, else the stream's own under `splits/v2/leak/`. Its threshold is the floor. Its positives are rebuilt from the seeds and descriptor files it records and must give back its threshold, else the calibration refuses. A base that records no seeds gets positives drawn with this stream's seed prefix (`stream/v1/leak`), recorded.
     - *Negatives, per image.* Each negative image is scored as the scan scores an image: its best cosine over dev + test + imageweeds, leaving out the evaluation images of its own capture session and capture date where both are known. A negative within 6 dHash bits of an evaluation image (dHash or any of the 8 variants) is a copy, not a negative: left out and counted (the L-8 image).
     - *Tiers.* `hard`: train_core images with a capture session. `hard_session_disjoint`: the hard images whose session has no dev or test frame (reported; cwd12's split is random, so most train_core sessions also have test frames, which are left out of the image's maximum). `provenance_disjoint`: the funnel config's `leak.negative_source_pairs` groups (MH-Weed16) from Step 1's `pool.jsonl`. `easy`: `fvossel__csgo_player_detection` and `kg_farukalam__tomato-leaf-diseases-detection-computer-vision`, each named a non-plant or leaf-disease source by the funnel config's `sources.not_recoverable`.
     - *Threshold.* The smallest 6-decimal cosine at which the per-image false-positive rate is at most 1 % on every constraining tier: `hard` always (it must hold at least 100 images), each other tier when it holds at least 100. Never below the base's. Per tier: false hits, rate and the one-sided 97.5 % Clopper-Pearson bound, at the new threshold and at the base's (the base's own per-image rate is recorded, not only the new one).
     - *Strict threshold.* The smallest cosine above every negative image of every tier.
     - *Recall.* Per augmentation family (funnel.leak's eight), the base's positives at the new threshold (cos at or above it, or dHash within 6 bits under a variant) and at the base's. A family below 0.95 is recorded as a known limit, not a refusal: the dHash variants still catch flips, rotations and re-encoding.
     - *Where it runs.* After the scan: the `scan` verb and `build` (scan_mode auto) write it; `build --skip-scan` applies only a current one. It reads descriptors the base recorded (funnel `emb_dinov2_images_{reference,pos_<family>,pool}.npz`, checked by sha256 and rows digest) and the scan's `splits/v2/leak/eval_desc_v2.npz`. A file that is missing or changed is recomputed into `splits/v2/leak/v2cal_*.npz`; the funnel's directory is never written. A rerun with the same identity (base, evaluation descriptors, rows, config, parameters, code) is a no-op.
     - *Negatives file.* `embed_calibration_v2_negatives.csv`, one row per negative image with the evaluation key it is nearest to; evaluation keys only, never an evaluation image path.
   - **The copy rules** (per candidate in `splits/v2/copy_rules.jsonl`, and `copy_rule` in `base_v2_provenance.jsonl`):
     - `tsw_eval_session`: a tsw row of a dev or test capture session, dropped (L-9(a)).
     - `tsw_provenance`: every other tsw row with a capture session (8 digits, `_`, a camera token). The embedding threshold does not apply; the 6-bit dHash + 8-variant check does. What the scan says is recorded.
     - `tsw_embed_v2`: a tsw row without a capture session is judged like base B; a hit at the v2 threshold drops it (`near_eval_embed`).
     - `base_b_embed_v2`: base B's kept part. A hit at or above the v2 threshold, or within 6 dHash bits, refuses the build (H6(b), R4). A hit only at the scan's threshold is recorded, not applied.
     - `l5_excluded`: the L-5 part, recorded as before.
     - `train_core` (provenance only): v1's train_core under the 8-variant check (L-8).
     - An image the scan could not describe is dropped (tsw, also under `tsw_provenance`) or refuses (base B's kept part): fail closed.
   - **The funnel's list (L-9(b)).** An entry is applied only when its split is a v2 evaluation split and it is a copy at the v2 threshold (within 6 bits, or cos at or above it). An entry against ood22 or ood23 is recorded as `not_v2_split`, one below the v2 threshold as `below_v2_threshold`. The funnel's matches are one detector's verdict at its per-pair threshold; the v2 scan judges the same pairs at the v2 threshold, so an entry it clears is not applied either. When `leak_v1.json`'s listing may be truncated (200 pairs, or fewer images than its count), every base_B pair is read from its pairs file, so a v2 match beyond the cap is not missed.
   - **Per source** (summary.json `embed_v2.per_source`, informational): the hits at the v2 threshold against the count the hard tier's per-image bound predicts. A source is flagged only on a hit within 6 dHash bits, a hit at or above the strict threshold, or P(Binom(n, p) ≥ hits) < 0.001. Removing a whole source from base v2 stays the owner's decision (R4).
   - **Without a current v2 calibration** (`build --skip-scan`), every scan hit and every v2-split funnel entry applies at the scan's own threshold, and lock refuses.
6. **`nevertrain_dhash.json` v2:** dev 617 + test 1,977 + imageweeds 3,208 = **5,802** entries at 6 bits. It is marked complete only at lock (`splits._finalise_index` logic).
7. **`base_copies_dhash.json`:** every image of base v2 at 6 bits. Harvested copies of base photographs never return as "new" data with worse labels. Examples:
   - the AgML three_season release, where the old join turned Purslane into Palmer amaranth;
   - the cwd12 copies in cwp10 and vanpe;
   - the 1,055 v1 near-eval drops, which are copies of ood22/ood23 and so become base copies under v2.
8. **`base_v2.jsonl`** = train_core (minus the L-8 drops) ∪ tsw22 ∪ tsw23 ∪ `step1/base_selected.jsonl` (878 rows, sha `7e47d374…`). That is 7,626 before the pair drop and 7,625 expected after it, before L-5 and L-8. It is checked pairwise disjoint by key, path, sha256 and 6-bit dHash. The L-8 drops are in no v2 manifest, in `base_copies_dhash.json` or in the provenance file.
9. **`LOCK.json` v2:**
   - `splits_version: "v2"`;
   - manifests {train_core, tsw22, tsw23, base_v2, dev, test, imageweeds};
   - `nevertrain_sha256` and `base_copies_sha256`;
   - `scorer_sha256`;
   - `derived_from` {the v1 LOCK sha256, identical: [dev, test, imageweeds] (plus train_core when nothing is dropped), train_core: {v1 sha256, v2 sha256, the L-8 list and its sha256, the count, the rule}};
   - `train_core_variant_drops_sha256` and `train_core_variant_drops` {file, sha256, rows, keys, cap, decided_by};
   - the funnel H6 status of base B's part (`pending` or the `leak_v1` result);
   - plus `lock_log.jsonl`; every file is 0444.
   - **[L-8] lock re-checks the list.** It must hash as the build recorded and hold at most the cap. Each row must be a v1 train_core row with the same bytes, whose image still hashes as recorded and whose re-computed dHash and 8 variants are the recorded ones. GuardV2 must refuse each row as `near_eval_variant`. v2's train_core must be exactly v1's minus the listed rows. A changed list, or a forged one naming a clean row (even with a matching `summary.json`), refuses. `verify` re-hashes the list against LOCK v2 and re-derives train_core.
   - **[L-9] lock re-derives the copy rules.** Each tsw and base B row of base v2 is judged again by its rule through the v2 calibration; the provenance file's `copy_rule` must be the re-derived one; a kept tsw row of a dev or test session refuses. The v2 calibration must be the one the build applied and must have been made from the scan's calibration and evaluation descriptors. LOCK v2 records `embed_calibration_v2_sha256` with an `embed_calibration_v2` block (threshold, strict threshold, base, known limits), `embed_calibration_v2_negatives_sha256`, `copy_rules_sha256`, an `l9` block, and the v2 threshold in `h6` and `h6_status`. `verify` re-hashes all three files and loads the calibration against LOCK v2.
   - **[L-8] Consumers.** `inc2.train` refuses the listed image bytes under any key or source (reason `train_core_variant_drop`), after checking the list's sha256 against LOCK v2. A production LOCK without the record is refused. GuardV2 and the v2 `NeverTrainGuard` refuse the same images by their pixels (`near_eval_variant`), since that is what put them on the list.

### 4.3 The new protocol package `inc2` and what the v1 code refuses

The pinned modules (`driver, gate, splits, common, scorer, lora, train, verify, select, relevance, audit, cwd12_species`) are never edited.

**Why v2 cannot run on them:**
- `inc/common.py:34-49` fixes `SPLITS_VERSION = "v1"` and `EVAL_SPLITS = (dev, test, ood22, ood23, imageweeds)`.
- `train.guard_rows` (`train.py:675-697`) and `pilot.check_training_manifest` (`pilot.py:899`) refuse every tsw image.
- `driver.FINAL_EXAMS` (`driver.py:232`) and `report.REPORT_EXAMS` include ood22/ood23.

**`tools/inc2/`** is a new package. It is addressed by the autopilot through the campaign's `protocol_package` and pinned as a unit at acceptance (P11).

| Module | Content |
|---|---|
| `inc2/common.py` | Re-exports `inc.common`. Overrides `SPLITS_VERSION = "v2"`, the v2 paths, `EVAL_SPLITS = (dev, test, imageweeds)`, `TRAIN_SPLITS = (train_core, tsw22, tsw23)`, `FINAL_EXAMS = (dev, imageweeds, test)` and `BASE_COPIES_INDEX`. [review] Overriding constants does not rebind the functions: `inc.common.manifest_path`, `read_lock`, `verify_manifest_against_lock` and `NeverTrainGuard.load()` (no argument) read `inc.common`'s own module globals, so a re-exported call still resolves v1 paths. `inc2.common` therefore defines its own `train_manifest_path(split)` (v2 training manifests), `read_lock_v2()` and `nevertrain_v2()`. It keeps the evaluation paths explicitly v1 (`eval_manifest_path = inc.common.manifest_path`), because the v1 scorer and exams are reused. A test asserts where each function resolves, and asserts that no `inc2` module calls `NeverTrainGuard.load()` without a path. |
| `inc2/guard.py` | `GuardV2` (§3.2). `NeverTrainGuard.load(path)` (`common.py:213`) already accepts an explicit path. |
| `inc2/splits.py` | §4.2. |
| `inc2/recipes.py`, `inc2/train.py` | A copy of `inc/train.py` with these changes:<br>• `guard_rows` uses the v2 index;<br>• `check_manifest` refuses v1 and v2 evaluation manifests;<br>• exams are restricted to (dev, imageweeds, test);<br>• the recipe table becomes Protocol v3 (§5.1), replacing `train.py:174-180`;<br>• `CODE_MODULES` adds `inc2.*`;<br>• the scorer subprocess stays `tools.inc.scorer`. |
| `inc2/baseline.py` | `build` for B_v2, the canary, B0 ∪ tsw and milestones, under the v2 guard. |
| `inc2/pilot4.py` | Stage A (§5.1). |
| `inc2/step1_stream.py`, `inc2/mask.py` | §3.3. |
| `inc2/stream.py`, `inc2/stream_report.py` | §3.4–3.7. |

### 4.4 Baselines

Costs in V100 GPU-h (= SU), est. from B0 (1.77 GPU-h for 3 seeds × 3,049 × 100 epochs) and realloop_v1.

| Run | Status | GPU-h | Why |
|---|---|---|---|
| **B_v2**, 5 cold seeds on base_v2 + finals (dev, imageweeds, test) | required (milestone 0) | 6.9–7.9 | The loop's starting point and D-A's measured effect. 5 seeds make the p ≤ 0.025 test possible; at 3 v 3 the smallest p is 0.05. |
| **Canary**: B0 seed 0 under `inc2.train` | required | 0.6 | Reproduces b0_v1 `base__s0` on b0_v1's base minus the L-8 drops (LOCK v2's train_core). The manifest is accepted by b0_v1's base sha256 plus the L-8 list's sha256, and exp.json records the dropped rows. It must fall within 1 sd of B0's seeds (0.0063) on dev. |
| **B0 ∪ tsw** (6,748 images), 3 seeds | recommended (does not reopen D-A) | 3.7–4.2 | Isolates the 878 harvested images, which lowered ImageWeeds in v1 (0.061 → 0.016). |
| H100 calibration: B0 seed 0 on H100 | optional | 0.3–0.6 (= 0.6–1.2 SU) | H100 costs 2 SU/h and its speed-up is unmeasured. It uses V100 until then. |
| B0, B | **not re-run** | 0 | Identical manifests and scorer. |

---

## 5. The recipe, M, the truth arm and consolidation

### 5.1 The recipe: Protocol v3, pre-registered, dev only

It is written into docs/INCREMENTAL_PROTOCOL.md as "Protocol v3" (group B) before any run.

**Candidates:**

| Id | Recipe |
|---|---|
| R0 | The current full rehearsal: 30 epochs, lr0 0.002, warmup 1, cosine to lrf 0.01. |
| X1a | LR re-warm: 30 epochs, warmup 3, peak lr0 0.005, cosine to lrf 0.01. |
| X1b | LR re-warm: 50 epochs, peak lr0 0.01. |

- **Freeze and LoRA.** D-D names them, but they are excluded **by the same survival rule applied to pilot_v3's recorded chains**: they agreed with truth on 3 of 7 steps, against the 5 of 7 required.
- **Re-admission.** A later pilot that passes the rule re-admits them under a new stream version.

**Stage A (known truth, `pilot_v4`, about 17–18 GPU-h est.).**
- Built by `inc2.pilot4` on pilot_v3's bins, sha-identical.
- Chains X1a and X1b, with no truth arm.
- The reference is pilot_v3's recorded truth verdicts. R0 is pilot_v3's own full chain and is not re-run.

An arm survives only if it does all of the following:
- accepts I2, I3 and I5;
- rejects Bswap and Breal;
- agrees with truth on ≥ 5 of 7 steps;
- ends with final dev ≥ 0.8084 − 2 × 0.0043 = 0.7998.

R0 passes on its record (final dev 0.8006, agreement 5/7).

**Stage B (B_v2 scale, segment 1).**
- Segment 1 runs two chains: R0 and the best Stage A survivor (R0 alone if none survives).
- For each recipe, δ_min = max(0, inc − mean(null) − 2·sd(null)) is computed per step. It is the data effect a helpful increment needs just to clear the regression guard.
- Choice rule, in order:
  1. the smallest median δ_min;
  2. then fewer recipe flags (P_recipe ≤ 0.25);
  3. then the cheaper recipe;
  4. a tie goes to R0.
- P_1 follows the chosen chain.
- No arm is added after results are seen without a new pre-registration. Test is never read in this choice.

**D30 (§6.4) holds the TRAIN lane** until Stage A has a READY record. After that, D30 reads the stream's own segments. realloop_v1 serves only as its prior before segment 1 exists.

[review] Once Stage A is READY, whether or not an arm survived, D30 reads only the stream's own segments, and with no segment decided it is silent. Otherwise a Stage A with no survivor (R0 alone) would leave D30 firing on realloop_v1 forever, and TRAIN would never start. Under the guard-based reading of §3.5, D30 does not fire on realloop_v1 anyway.

**[review] Stage C: can the gate accept anything at M? (known-good increment, before any collection; est. 10.6–11.8 GPU-h with one recipe, 13.3–14.5 with two, from the §5.6 rates)** Stage A tests the recipe on pilot_v3's bins. Those bins are session slices of cwd12 train, so they contain every species, PricklySida included. The planned sources contain no PricklySida, CutleafGroundcherry or Sicklepod, and in realloop_v1 the species guard failed on PricklySida in 5 of 6 steps, with null sd 0.0061, so the guard sat at its 0.03 floor. Stage C is one pinned-driver chain experiment:
- base = base_v2 minus one increment D of exactly M = 763 tsw rows, drawn as whole sessions of data2022 target images. The 10 3SeasonWeedDet10 classes include none of the three rare species, which is what makes D a realistic stand-in for collected data;
- one step D, `clean: false`, the Stage B recipe candidates, the net gate and the truth arm on.

D is expert-labelled, so the truth arm is expected to say "helps" or "neutral". Outcomes:
- the chain ACCEPTs D → M is feasible, and the stream proceeds;
- a REJECT on the species guard alone → D33 fires prospectively, with evidence, before any SU is spent on collected data. Card X17 (a gate or dev change, which is R4) goes to the owner at R0, not after 12 rejected increments (D32).

Stage C's base is a sub-base of base_v2 and changes nothing in D-A. It runs as part of R0 (§10).

### 5.2 The incumbent

**Stream version 1 uses the pinned driver.** Its chain incumbent is `base__s0` (`driver.py:2042`). In every chain experiment so far that seed was the best of three base seeds:
- pilot_v1–v3, which share one base: 0.7282 against 0.7117 and 0.7237;
- realloop_v1: 0.8189 against 0.8154 and 0.8055.

That is two independent observations, p = 1/9 under chance. A lucky seed 0 makes null < inc more likely and lowers acceptance. It is monitored (D30 records mean(base) beside inc).

**A median-seed incumbent** needs driver v3 (`inc2/driver3.py`, subclassing `driver.Driver` by import). It is built only on the evidence trigger of R5 (§10).

### 5.3 M and K

**M = ⌈0.10 × |base_v2|⌉ = 763.** It is fixed for stream version 1.
- **Resolution.** The warm chain reached P_data = 1.0 at cand − null = +0.0035 and +0.0047 (realloop_v1 s05, s06). The expected effect of +10 % data at this scale is about 0.002–0.005 (est., learning curve: pilot union 2,777 images → dev 0.8084; B0 3,049 → 0.8082; B 3,927 → 0.8133). So 10 % sits at the edge of what the gate can see; 5 % (382) and realloop_v1's 287 fall below it.
- **Cost.** With full rehearsal a step costs about the same whatever M is, so the cost per offered image scales as 1/M.
- **Changes to M.** M changes only through L22 (a doubling when D8 shows the gate underpowered), which makes a new stream version. The owner's "fixed size M" is taken literally: M is never re-derived silently as the pool grows.

**K = min(4, ⌊Q/M⌋) ≥ 1**, where Q is the number of eligible target images.
- A segment is cut when Q ≥ 4M, or when Q ≥ M and the oldest eligible row is ≥ 7 days old (D22).
- K is capped at 4 because supply, not SU, limits the loop, and shorter segments give earlier boundary checks.

### 5.4 The truth arm

**On for every step (P3).**
- **Power.** At 3 v 3 cold seeds the minimum detectable effect is about 0.011 (FUNNEL_AUDIT §6 H10), so the truth arm reads "neutral" for most expected effects. It does resolve "hurts" and large effects: realloop_v1's truth arm found OTHER_HEAVY and V4 "hurts".
- **Role.** It is the evidence for attribution (D31) and for agreement tracking. It does not admit data: the gate decides (D-D).
- **Cost.** 4.2–4.9 GPU-h per step (est.). Because supply limits the loop, this stays inside the monthly window (§6.6).
- **Switching it off.** If the window cannot fit a segment with truth, the builder proposes the segment without truth as an R3 item for a person, as L6 does. It never drops the truth arm silently and never pauses for it.
- **Card X13** would let the truth arm decide admission. It is recorded but not adopted, because it conflicts with D-D (the pinned gate decides).

### 5.5 Consolidation

1. **Boundary check (every segment, free).** Segment s+1's 3 cold base seeds on P_s are compared with segment s's base seeds on P_{s-1} (dev).
   - If mean(new) < mean(old) − 2·sd(old), D25 fires: TRAIN holds and a milestone runs at once.
   - At 3 v 3 this is an alarm, not a claim.
2. **Milestone (L20)** is triggered by any of:
   - ≥ 4 accepted increments since the last milestone;
   - ≥ 3 segments;
   - ≥ 30 days with ≥ 1 accepted increment;
   - D25.

   It runs 5 cold seeds (§3.6). **This is the only place test is read.** The milestone's dev verdict (5 v 5) decides rollback. Its test mean ± sd is the headline, reported with the gap to 0.90.

### 5.6 Cost model (V100, 1 SU per GPU-h; est.)

**Measured rates:**
- cold runs: 6.0–7.0 ms per image-epoch (realloop_v1: 1.974 h / 3 × 3,927 × 100; B0: 7.0);
- incremental runs: 7.4 ms per image-epoch (realloop_v1: 9.05 h over 36 runs).

At N = 7,626 and M = 763:

| Item | GPU-h |
|---|---|
| Segment base, 3 cold seeds | 3.8–4.4 |
| Chain step: 3 cand on N+M and 3 null on N, 30 epochs | 3.0 per recipe |
| Truth arm step: 3 cold on N+M | 4.2–4.9 |
| Segment finals | 0.6–0.8 (realloop_v1: 0.83) |
| **Segment, K = 4, one recipe, truth on** | **33–37** |
| Segment, K = 4, truth off | 16–17 |
| Milestone, 5 cold seeds at about 9.2K images + finals | 8–9.5 |
| Step 1 stream admission | about 27 s GPU per 1,000 images, plus the job's fixed cost (≤ 0.5 per job) |
| Collection job (GPU-shared, no GPU use) | 1–4 per source |

**Scaling:**
- Each accepted increment adds about 0.28 GPU-h to every later chain step and about 0.5 to every truth step.
- **Walltime.** It binds near 41–48K images for cold runs (8 h, `driver.py:222`) and 48K for incremental runs (3 h, `run_inc_job.sh`). D26 fires at 0.8 of the limit (§6.4).
- **Memory.** Above about 20K images the RAM cache stops fitting in 0.6 × 45G, and runs fall back to reading from disk.

---

## 6. The autopilot in stream mode

### 6.1 Why today's ticker cannot run it

The ticker is built around one experiment at a time:
- **Waits for the experiment.** It diagnoses only after the current experiment is done (`campaign._phase_machine`, 1855).
- **One item.** It has a single item in flight (`_new_item`, 2604).
- **Premature COMPLETE.** It goes COMPLETE when nothing is proposable (2696).
- **Submission cap.** It pauses on the 4th submission of any lever (`executor.MAX_LEVER_SUBMISSIONS = 3`, 146).
- **Budget.** The budget is a lifetime total, and job estimates are never settled (`budget.py:131, 188`).
- **Hardcoded package.** Its levers hardcode `tools.inc` (`levers.json`).
- **Blocking lab fetch.** A lab fetch runs synchronously in the tick thread for up to an hour (`executor.funnel_fetch_hook`, 2239-2251).
- **Old training path.** One config flip would bring back `mega_trainer` training (`round_scheduler._WEED_STEPS`, 87-102).

### 6.2 Stream mode — `inc_autopilot/stream.py` (new, group F)

**Dispatch.** `campaign.tick` hands a campaign with `mode: "stream"` to the lane ticker. Campaigns with `mode: "experiment"` (`weed_inc_v1` and the funnel) behave byte for byte as before.

**Campaign config** (`~/.round_scheduler.json` → `campaigns.weed_stream_v1`):

```
{"mode": "stream", "domain": "weed", "protocol_package": "inc2",
 "stream": {"sid": "weed_stream_v1", "collect_config": "collect/domains/weed.json",
            "stream_domain": "inc_autopilot/stream_domains/weed.json", "K_max": 4},
 "goal": {"kind": "continuous"}, "autonomy": "envelope", "data_autonomy": "on",
 "envelope_su": 600, "window": "month", "window_cap_su": 250, "daily_cap_su": 80,
 "alloc_reserve_su": null, "collect_gb_envelope": 200, "collect_gb_daily": 50}
```

- `alloc_reserve_su` is set once the `projects` output is parsed (D27).
- `collect_gb_envelope` and `collect_gb_daily` are proposals, checked against /ocean headroom.

**Three lanes, one item in flight per lane:**

| Lane | Phases |
|---|---|
| **DATA** | IDLE → DISCOVER (L15, lab) → [L12 / L11a prerequisites for a new source's names] → COLLECT (L16) → WAIT_JOB → ADMIT (L17) → WAIT_JOB → IDLE. [review] L11a and L12 as they exist write into the funnel's own directory under the funnel's prereg (`levers.json`: `"out": "<LAB INC_DIR>/funnel/"`, `--prereg …/funnel/prereg_v1.json`). Run for a stream source, they would change the live audit's `taxonomy_cache.json` and cards index, which `_funnel_cards_stale` would treat as stale inputs. The stream's prerequisite is a new lab lever, L26 `collect names` (R0). It reuses `funnel.taxonomy` and `funnel.fetch` as libraries and writes to `INC_DIR/intake/names/`, layered over the funnel's cache, which it reads only. |
| **TRAIN** | IDLE → CUT + BUILD (L18) → RUN (R1 advances) → COMMIT (L19) → DIAGNOSE → IDLE, or HOLD(reason) |
| **MAINT** | IDLE → MILESTONE (L20) \| ROLLBACK (L21) \| BASELINE (L23) → RUN → COMPARE → IDLE |

**Each tick, in order:**
1. **Lab work, with no ssh:**
   - fold the last snapshot into the queue and yield state;
   - run D20–D33 on the cached evidence;
   - stage lab jobs (L15, L12, L11a) as **detached** processes with a result file, and poll them.
2. **The one ssh of the tick:**
   - if any lane has a ready item, submit all of them in one `executor.submit_many` batch, in priority order: stop/cancel, then TRAIN, then DATA, then MAINT;
   - otherwise take one `campaign-snapshot` covering every lane: live segment and milestone experiments, `step1_stream/status.json`, `stream/<sid>/queue_summary.json`, intake summaries, squeue, the `projects` balance and the /ocean quota.
3. **Health checks on every snapshot:** D5, D6, D7, D10, D14, D26 and D27.

The rotation across campaigns (`_tick_all`) stays.

### 6.3 Levers (`levers.json` menu v2, a dated pre-registered amendment)

| Lever | What | Policy action | Risk / autonomy | Command |
|---|---|---|---|---|
| L15 | discover sources | `inc_stream_discover` | R0, lab, detached | `python -m weed_optimizer_framework.tools.collect plan --config collect/domains/weed.json --classes <deficit> --out <LAB>/collect/candidates.json` |
| L16 | collect one source | `inc_stream_collect` | R2; direct only with `data_autonomy: on`, a stream replay pass and the caps of §6.6. A source failing a pre-check (licence unknown, no target class in card or taxonomy, image-level only, over the byte cap, missing credentials, same-lab without the copy scan) is filed R3 for a person. | `sbatch -p GPU-shared run_inc_collect.sh fetch --source <id> --max-bytes <B>`, or the lab hook of the same verb; then `collect intake --source <id>` |
| L17 | admit an intake batch | `inc_stream_admit` | R2, as L16 | `sbatch -p GPU-shared run_inc2_stream.sh admit --intake <batch>` |
| L18 | cut and build a segment | `inc_build_segment` | R3, envelope | `python -m weed_optimizer_framework.tools.inc2.stream build --stream <sid> --k <K>` (in `run_inc_build.sh`) [review: in the new `run_inc2_build.sh`, because `run_inc_build.sh` refuses every verb but the three v1 builders; the same applies to L20, L22, L23 and L25] |
| L19 | commit a finished segment | `inc_stream_commit` | R1 | `python -m …inc2.stream commit --exp <sid>_sNNN` |
| L20 | milestone consolidation | `inc_build_consolidation` | R3, envelope | `python -m …inc2.stream milestone --stream <sid>` (it calls `inc2.baseline build`, 5 seeds) |
| L21 | roll back to P_c | `inc_stream_rollback` | R3, envelope only when the target is the last milestone pool; any other target needs a person | `python -m …inc2.stream rollback --stream <sid> --to <P_c>` |
| L22 | double M (new stream version) | `inc_build_segment` | R3, envelope; at most 2 doublings, then card X17 | `…inc2.stream fork --stream <sid> --m <2M>` |
| L23 | splits v2 and baselines | `inc_splits_build` + L8 | R3. The splits build needs one approval by a person of the exact D-A command; a LOCK change is otherwise R4, and D-A already decided it. The baselines run in the envelope. | `python -m …inc2.splits build` / `lock`; `python -m …inc2.baseline build --exp b_v2 --manifest …/splits/v2/base_v2.jsonl --seeds 0,1,2,3,4` |
| L24 | quarantine a source | `inc_stream_quarantine` | R2, only with a firing D28 or D31 cite; reversible | `python -m …inc2.stream quarantine --source <slug> --cite <diag>` |
| L25 | Protocol v3 Stage A pilot | `inc_build_pilot` | R3, envelope after the owner accepts Protocol v3 (P1) | `python -m …inc2.pilot4 build --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b` |
| L26 [review] | resolve a new source's class names (stream-owned) | `inc_stream_names` | R0, lab, detached | `python -m weed_optimizer_framework.tools.collect names --source <id> --out <LAB INC_DIR>/intake/names/` (reuses `funnel.taxonomy` as a library and reads the funnel cache; never writes `funnel/`) |
| L27 [review] | bisect suspect increments after a rollback | `inc_build_consolidation` | R3, envelope; ≤ 1 per rollback | `python -m …inc2.stream bisect --stream <sid> --from <P_c>` (§3.5) |
| L28 [review] | Stage C gate-feasibility chain | `inc_build_segment` | R3, envelope; once per stream version | `python -m …inc2.stream feasibility --stream <sid> --holdout tsw22 --m <M>` (§5.1) |

**New R4 cards, never queued:**
- **X13:** the truth arm decides admission (not adopted, §5.4).
- **X14:** raise the domain envelope or the allocation reserve.
- **X15:** walltime limits (`driver.COLD_TIME_LIMIT`, `run_inc_job.sh --time`, both pinned).
- **X16:** provider credentials or licence policy.
- **X17:** M beyond the doubling rule, dev composition, or the species guard on a rare dev class.

**Existing cards reused:**
- X1: the recipe, through Protocol v3;
- X4: leave-one-source-out bisection after a rollback;
- X10: class policy;
- X11: the verifier refit.

### 6.4 Diagnoses (`diagnose_stream.py`; thresholds in a `stream` block of `thresholds.json`, each with its "why")

| Id | Name | Fires when (proposed value) | Proposes / operation |
|---|---|---|---|
| D20 | data_needed | Eligible target images Q < 2M (the low-water mark), or some species has fewer queued target boxes than the last 2 segments consumed; and the DATA lane is idle. It also fires on a builder refusal "holds N eligible images against M needed". | L16 on the top open candidate, ranked by expected verified target boxes per GB, with a bonus for the deficit species. With no open candidate and a last L15 more than 7 days old: L15 with `--classes <deficit>`. |
| D21 | source_low_yield | After ≥ min(2 GB, the whole source): yield < `floor_gb` target boxes per GB, or < `floor_su` per SU of collect + admit. | Close the source; update the provider and query prior. |
| D22 | cut_ready | Q ≥ 4M, or Q ≥ M with the oldest eligible row ≥ 7 days old; TRAIN idle; no hold. | L18 with K = min(4, ⌊Q/M⌋). |
| D23 | segment_finished | A segment is done and not yet committed. | L19. |
| D24 | milestone_due | ≥ 4 accepted increments since the last milestone, or ≥ 3 segments, or ≥ 30 days with ≥ 1 accepted. | L20. |
| D25 | stream_regressed | The milestone verdict is "hurts" (5 v 5), or the boundary check of §5.5 fails. | Boundary check: L20 at once, TRAIN holds. Milestone "hurts": L21 to P_c, TRAIN holds, card X4. |
| D26 | walltime_bound | The projected longest run of the next build is ≥ 0.8 × its limit (cold 8 h, incremental 3 h, build 4 h). | Hold TRAIN and MAINT builds; card X15. |
| D27 | resource_reserve | Allocation balance − committed SU of all campaigns < the reserve, or staging + projected bytes > the quota − 3 % (L-7). | OP_PAUSE. |
| D28 | source_leak | ≥ 5 % of a source's images refused by the never-train v2 guard, or ≥ 20 % base copies. [amended 2026-09-29 (with the funnel's amendment A2, docs/FUNNEL_AUDIT.md §14): a source leaks when one of its images is within 6 dHash bits of an evaluation image under any of the 8 variants, or when its embedding hits are improbable under the copy detector's per-image false-positive rate, P(Binom(images, p_false) ≥ hits) < 0.001 (`inc2.embed_calibration.source_verdict`; p_false from the batch's copy-scan record, else the LOCK's v2 calibration; with neither, any embedding hit leaks), or at ≥ 20 % base copies. The 5 % share rule is dropped: at a per-image rate p a source of n images holds about n·p chance hits.] [amended 2026-10-03 (D28-v2, amendment V3-1, pre-registered; "Amendment (2026-10-03)" at the end of this document): a dHash hit still drops its image, but it leaks its source only when its DINOv2 pair cosine with the matched evaluation image reaches the v2 calibration's cos_threshold, or when the source's confirmed hits (pair cosine ≥ 0.80; a source's intake batches judged together) are improbable, P(Binom(images, 6.29e-4) ≥ confirmed) < 0.001; a hit without a pair cosine is still a leak on its own (fail closed); batches committed before the amendment are weighed again into sidecars (L17 eval-hits).] | L24 and a card. |
| D29 | collection_exhausted | No open candidate predicted above the floor, and an L15 run in the last 14 days found nothing new. | Allows COMPLETE (§6.7). [review: under P7 it moves the DATA lane to WAIT_DATA with discovery backing off 7 → 14 → 30 days, plus a card; it never COMPLETEs.] |
| D30 | recipe_blocks_stream | On the last segment, pooled as D1 does: `p_recipe ≤ 0.25` and null < inc, with `blame == "recipe"`. Before Stage A is READY, realloop_v1 is its evidence. [review: replaced by: ≥ half of the segment's REJECTs are recipe-caused as §3.5 defines it (only the regression guard failed). `blame == "recipe"` is set on every REJECT when P_recipe = 0, which is every full-recipe step recorded so far, so it cannot separate recipe from data.] | Hold TRAIN (no L18); the recipe-blamed images return to the queue; cards X1 and X13. DATA and MAINT continue. |
| D31 | data_blamed | The recipe is healthy (P_recipe > 0.25) and a step is REJECTed with P_data ≤ 0.25, or the truth arm says "hurts". [review: the condition "recipe healthy (P_recipe > 0.25)" is dropped. P_recipe was 0.0 at every full-recipe step recorded, so as written D31 could never fire, and the planted Bswap (P_data 0.00) would not have been blamed. Fires on a §3.5 "data" disposition or on truth "hurts".] | L4 label audit on that increment first. A source blamed ≥ 2 times → L24, and D21 closes it. |
| D32 | stream_not_accepting | 0 ACCEPT in the last 12 decided increments while D30 is silent. | Hold L18 and raise a card; the DATA lane runs at half cadence. |
| D33 | rare_species_guard | The same species fails the species guard in ≥ 50 % of a segment's REJECTs (realloop_v1: PricklySida, 5 of 6). | Card X17; a D20 ranking bonus for candidates that declare that species. |

**Reused:**
- D5, D6, D7, D10 and D14: health checks.
- D8 → L22.
- D1 is read only through D30.
- D17–D19 stay with the funnel campaign.

**Recipe or data?** Read from the gate fields the pinned gate already writes (`p_recipe, p_data, guards, attribution.blame, recipe_flag, species_failed`):

| Signature | Reading | Diagnosis |
|---|---|---|
| Recipe flagged, null < inc, blame "recipe" | the recipe | D30 |
| Recipe healthy, P_data ≤ 0.25 or truth "hurts" | the data | D31 |
| P_data in (0.25, 0.75) on ≥ half of the steps | underpowered | D8 → L22 |
| A guard-only REJECT on one rare species | dev composition | D33 |

[review] The first two rows are read through §3.5's guard-based disposition: "the recipe" = only the regression guard failed; "the data" = P_data ≤ p_reject or truth "hurts", whatever P_recipe is. Also, D33 alone does not hold TRAIN, so a species guard that trips on every step would burn segments until D32 (12 decided increments, about 3–6 months at the supply-limited pace). D33 therefore also holds L18 when it has fired on 2 consecutive segments. DATA continues, and the queue keeps growing for the owner's X17 decision.

### 6.5 Governance

| Risk | Actions |
|---|---|
| R0 | L15; the L12 and L11a prerequisites; the snapshot verbs; the allocation and quota reads; the network probe |
| R1 | L19; `advance` |
| R2 | L16 and L17: downloads land in content-addressed staging, nothing trains, and data reaches a model only through gated increments (D-B). Also L24 and L4. |
| R3 | L18, L20, L21, L22, L25 (envelope-eligible); L23's splits build (one approval by a person); switching the truth arm off (a person); an L16 that fails its pre-checks (a person); cancelling a segment |
| R4 | X1, X4, X10, X11, X13–X17; any change to M outside L22, to thresholds, the verifier, a LOCK, the never-train index or the gate |

**Envelope and approvals.**
- `executor.ENVELOPE_LEVERS` (141) gains L18, L20, L21 (target P_c only), L22 and L25. [review] It also gains L27 and L28, and `brain/policy_actions.json` gains the R0 row `inc_stream_names` for L26.
- `approvals.ENVELOPE_ACTIONS` (`approvals.py:77`) gains `inc_build_segment` and `inc_build_consolidation`.
- Both are governance files, so the change invalidates the recorded replay pass until the stream replays pass. This is intended.
- **[review] One replay pass covers every campaign.** `executor.code_hash` (`executor.py:368`) hashes the whole `inc_autopilot` package plus `GOVERNANCE_FILES`. `REPLAY_REQUIRED` (`executor.py:189`) includes the funnel's cases, and adding `STREAM_REPLAY_CASES` extends it. So every deploy of group F's code voids envelope autonomy for the live funnel campaign and `weed_inc_v1` as well as for the stream, until `record-replay` passes again, now including every S-case. A deploy therefore runs `executor record-replay` in the same deploy step, at a funnel checkpoint where no funnel R3 item is waiting. A failing S-case blocks the funnel's autonomy too, so group F's acceptance requires the full replay set (inc, funnel and stream) to pass locally before any deploy.
- **[review] The collector's policy files are governance.** `collect/domains/weed.json` (licence policy, byte caps, providers, known items, card tables, EPPO table) and `inc_autopilot/stream_domains/weed.json` decide what L16 may do by itself. They are added to `GOVERNANCE_FILES`, so a change to either, by a person, a deploy or the brain, voids the replay pass. For the same reason `floor_gb` and `floor_su` (§7.4) are written by a person or a deploy, never by the running platform, even though they are derived from the measured first wave.

**R2 data levers.** A new `GATED_R2_LEVERS = (L16, L17, L24)` runs directly only with `data_autonomy: on`, a stream replay pass and the caps. That is stricter than the `_CEILING` R2 cell, because a bug in the data lane could loop downloads.

**Ledger.** DEC-A to DEC-D are written once per campaign with `decided_by: human`, as `FUNNEL_DECISIONS` is (`campaign.py:229`).

**The research brain** (cluster, advisory) gets the stream evidence in its digest and may only propose. The evidence allow-list keeps test out of the digest.

### 6.6 Budget and stop-losses

**Budget (`budget.py`, group F).**
- **Windows.** A monthly window and a daily cap on top of the lifetime envelope (P5). [amended 2026-10-04: neither has a default any more; one applies only when a campaign (or the domain's stored config) declares it. See "Amendment (2026-10-04)" at the end of this section.] [corrected 2026-10-04, E2's amendment: the domain's cap is read from `db.DEFAULT_DOMAIN_CONFIG` or a budget block the caller passes, never from the stored domain config (the ticker passes none); see "Stored domain config" in that amendment.]
- **Cross-campaign cap.** The domain's `su_envelope` (1,500, `db.py:488`) is enforced across campaigns, by summing every campaign's `inc:<campaign>` steps. It includes the funnel's own cap of 120 GPU-h (DEC-10) and what `weed_inc_v1` has already spent.
- **Settled job estimates.** Each L16, L17, L18 and L20 job's estimate is settled from sacct (`su_ledger.reconcile` / `parse_sacct`) when the job ends. Today estimates stay charged forever.
- **Allocation balance.** The Bridges-2 balance is read from the `projects` output in the snapshot and feeds D27. Whether GPU-shared hours and any RM hours draw from one balance is read from that output, not assumed.
- **A rate for every partition.** `brain/su_rates.json` gets an explicit rate for every partition the loop uses, so no job is priced "unknown".
- **Estimate.** At a supply-limited pace of one or two segments per month, spend is about 45–100 SU per month (est., §10). The monthly cap of 250 does not bind at that rate.

**Byte budget.** `collect_gb_envelope` and `collect_gb_daily`, checked against /ocean headroom (about 675 of 7,000 GB used on 2026-09-09). [amended 2026-10-04: `collect_gb_daily` and the collector's `budgets.bytes_daily` have no default; `collect_gb_envelope`, the per-source cap and D27's headroom stay.]

[review] **The headroom reading is stale and may already trip D27.** The INC loop's state record of 2026-09-26 gives /ocean as 93 % full (503 GB free). If that holds, D27's rule (staging + projected bytes > quota − 10 %, that is, less than 700 GB free) fires on the first snapshot, and the campaign PAUSEs before it collects anything. The quota is read live by the snapshot before R3. If free space is below 10 % + `collect_gb_envelope`, R3 does not start. A card names the largest directories under `INC_DIR` and `downloads/`, because deleting data is a person's action, and the 200 GB envelope proposal is resized to what fits. L-7 (§2.6) sets the floor to 3 % (about 210 GB). With 454 GB free on 2026-09-28, D27 does not fire, and free space covers 3 % plus the 200 GB envelope.

[review] **The allocation ends with the envelope.** The Bridges-2 allocation's recorded end date is 2026-12-31, the same date as the P5 envelope. After it no job can run, whatever the envelope says. The snapshot reads the allocation end date from `projects`, and the campaign raises an R4 card (renewal) 30 days before it. At the end date the campaign PAUSEs with reason `allocation_ended`, not COMPLETE, so the loop resumes on renewal without a rebuild.

[review] **The domain envelope.** `su_ledger.remaining(domain, …)` (`su_ledger.py:769`) sums every step of the domain's ledger, the pre-INC round history included. The cross-campaign cap above is defined on `inc:*` steps only. Before R0, the live domain's `budget` block (not db.py's default) and its ledger total are read. If the round history already exhausts the domain's `su_envelope` as `signals._check_budget` counts it, that must be resolved first (card X14), or the scheduler's own budget alarm will fire beside the stream.

**Limits** (`levers.json` `limits`, replacing the blanket 3-per-campaign rule for the stream levers):

| Lever | Limit |
|---|---|
| L16 / L17 | ≤ 3 attempts per source; ≤ 2 in flight; ~~≤ 6 jobs and ≤ 50 GB per day~~ (removed 2026-10-04); ≤ 50 GB per source unless a person approves |
| L18 | ≤ 1 in flight; ~~≤ 1 per day~~ (removed 2026-10-04) |
| L20 | ≤ 1 in flight |
| L21 | ≤ 1 per milestone |
| L1–L14 | keep "3 per campaign", so R1–R14 are unchanged |

**Stop-losses:**
- **Per lane:** 2 consecutive failed steps in a lane hold that lane.
- **DATA lane:** 3 consecutive sources that end **failed or with zero admitted target boxes** hold the lane and raise a card. This replaces the eight silent "+0" rounds of 2026-08/09. A single source can legitimately yield 0, so one zero-yield source is not enough.
- **The campaign pauses** when any of these happens:
  - 2 lanes are held;
  - D7, D10, D14 or D27 fires;
  - the ticker raises on 3 ticks in a row (`ERRORS_TO_PAUSE`).
- **Halt.** OP_HALT fires if a decision path touches a non-dev exam.

#### Amendment (2026-10-04, decided by the owner): no time-based throttles by default

**Why (measured).** On 2026-10-04 the cluster sat idle for about 9 h while the stream's next segment (L18) was filed "awaiting approval" for two reasons only: "estimated 116.2 SU exceeds the 86.52 SU left under today's cap of 120" and "L18 already ran 1 time(s) in the last 24 h (limit 1)". The campaign's own `daily_cap_su` was 180 (raised on 2026-10-03), but `budget.envelope` capped it at 120, the `budget.daily_cap` of `db.DEFAULT_DOMAIN_CONFIG`: a code default that nobody had set for the domain. (An interim change the same day, commit `11c01b0`, set that default to 1500, the domain envelope, so that it never bound; it kept a figure declared only because `fits(…, need_daily=True)` refused an undeclared daily cap. This amendment removes both.) The monthly window (350 SU; 176 SU used by 2026-10-04) would have been the next blocker within days. None of these caps protects anything that the fuses below do not already protect. All they did was delay healthy work.

**What changed.**
- **Daily SU cap.** There is no default anywhere:
  - `db.DEFAULT_DOMAIN_CONFIG["budget"]` no longer declares `daily_cap` (it was 120, then the interim 1500);
  - `inc2.stream.BUDGET` no longer records `daily_cap_su` (120);
  - the stream campaign's `STREAM_DEFAULTS["daily_cap_su"]` is None (it was 120).

  In `budget.envelope`, `state` and `fits`, an absent daily cap means no daily cap: no refusal and no reason. It is reported as none (`daily_cap_su` and `daily_remaining_su` are None, with the source "none (no daily cap is declared)"). `today_su` is still reported. The mechanism stays: a campaign that sets `daily_cap_su` still gets it, and a domain whose stored config declares `budget.daily_cap` still caps its campaigns. [corrected 2026-10-04, E2's amendment: "a domain whose budget block declares `budget.daily_cap`", the block being `db.DEFAULT_DOMAIN_CONFIG`'s or one the caller passes; `budget.domain_budget()` does not read the stored domain config (below, "Stored domain config").]
- **Monthly window.** It is treated the same way. `STREAM_DEFAULTS["window_cap_su"]` is None (it was 350), `inc2.stream.BUDGET` no longer records it, and an absent window is none. `fits(…, need_daily=True)` no longer refuses with "no daily cap is declared" or "no monthly window is declared". `need_daily` is still accepted but has no effect, and the executor's envelope rule no longer passes it. A window that a campaign declares still refuses past it.
- **Per-day lever counts.** These entries are removed from `limits` in `stream_levers.json`:
  - L16: `jobs_per_day` 12 and `gb_per_day` 50;
  - L17: `jobs_per_day` 6;
  - L18: `per_day` 1.

  `executor.stream_limits` still honours a per-day count if a limits entry declares one; none does now. `levers.json` has no per-day or windowed count. Experiment mode keeps its "more than 3 per campaign" stop-loss, which is a total, not a rate.
- **Per-day byte caps.**
  - The stream campaign's `collect_gb_daily` default (50 GB) is now None.
  - The collector's `budgets.bytes_daily` in `collect/domains/weed.json` (50 GB) is now null. `collect.config` accepts null or a positive number.
  - `collect.prefilter.precheck` (the `daily_bytes` hold) and the byte plan in `collect.fetch` apply the cap only when one is declared (`prefilter.daily_byte_cap`).

  Without the collector change, removing L16's `gb_per_day` would have left the same 50 GB a day in force: the collector would still clip, and then hold, every cluster fetch.
- **Configure command.** `python -m weed_optimizer_framework.tools.inc_autopilot.stream configure --name N --by human:<email> [settings]` changes settings only. Unlike `enable`, it does not enable the campaign and does not release a pause or a held lane.
  - `--daily-cap-su`, `--window-cap-su` and `--collect-gb-daily` accept `none` (`configure_stream`'s `CLEAR`), which stores None: no cap. `enable` accepts `none` too.
  - The lifetime envelope and `collect_gb_envelope` cannot be cleared: passing `none` for them is a usage error.
- **Displays and records.**
  - The /inc page shows "no daily cap" when none is declared.
  - The stream report shows "SU this month (no monthly window)". It takes the window from the current rule, not from a definition written before this amendment.
  - `stream_domains/weed.json` records decision L-2a, and L-2 (§2.6) is marked as amended.

**What stays, and why none of it delays healthy work.**
- **In-flight limits** (L16 2, L17 2, L18 1, L20 1). They bound how many jobs run at once, never when the next one may start. One segment at a time is the protocol's order, not a throttle.
- **Per-source limits.** `attempts_per_source` (3) and `gb_per_source` (50 GB, unless a person approves more). A source that keeps failing, or that grows past its size, waits for a person; a healthy source is never held by them.
- **Count totals.** The limits on L21, L22, L25, L27, L28 and LI are counts per milestone, rollback, stream version or campaign, not rates.
- **Lifetime envelopes.** The campaign's `envelope_su` (1,000 SU to 2026-12-31), the domain's `su_envelope` (1,500 SU across every campaign) and `collect_gb_envelope` (200 GB) are the fuses against a runaway bug that submits jobs in a loop.
- **Disk headroom, stop-losses and the end date.**
  - D27's disk headroom stays.
  - The stop-losses stay: 2 consecutive failed steps hold a lane; 3 zero-yield sources hold DATA; 2 held lanes, D7, D10S, D14, D27 or 3 raising ticks pause the campaign.
  - The allocation's end date stays.

**How it was verified.**
- **New tests: `tests/test_stream_ap_no_throttles.py`.**
  - The defaults: nothing above declares a time-based cap, and the fuses keep their values.
  - With no cap declared, a 116.2 SU request passes in two cases: with 120 SU already charged today (0 left under the old cap), and with the incident's 33.48 SU (86.52 left). A campaign's 180 is no longer cut to 120. With 400 SU spent this month, no window refuses.
  - A declared daily cap or window still refuses, and a domain's declared `daily_cap` still caps a campaign (180 → 100).
  - The envelopes still refuse: the campaign's 300 SU and the domain's 1,500 SU.
  - Through `executor.submit`, two L18s of 116.2 SU run within the envelope on the same UTC day. Each of these then refuses a further L18: `in_flight` 1, a 300 SU envelope, a declared 180 SU daily cap, and a declared 300 SU window. 13 L16 jobs and 65 GB in 24 h, or 7 L17 jobs, leave the next one free. The 50 GB per-source cap still refuses.
  - The `configure` command clears all three caps without enabling the campaign or lifting its pause, and records the change in the stream ledger. `enable --daily-cap-su none` works too. `--envelope-su none` is a usage error.
  - The collector neither clips nor holds a fetch when no `bytes_daily` is declared, and clips to the remainder when one is.
  - The deploy: `deploy_funnel.sh --dry-run` lists `tools/db.py` among the shipped files and this file in its pre-flight.
- **Updated tests.** They keep covering the mechanism, now with an explicit cap:
  - `test_stream_ap_cap_approved.py` declares the 120 SU cap the stream had then, and adds a case where no arm is filed when no cap is declared;
  - `test_stream_ap_units.py` covers the amended defaults, adds the measurement arms running by the envelope with none filed, and checks that `need_daily` no longer refuses;
  - `test_inc_ap_governance.py`: a campaign's cap stands without a domain cap, and is still capped by a declared one (through `executor.budget_now` and through `budget.envelope`); with neither declared there is no daily cap, so the interim 1500 (or the old 120) fails it;
  - `test_stream_ap_replay.py` S15: no per-day count by default, plus a declared six-jobs-a-day count that syncs do not use up and six fetch jobs do;
  - `test_collect_prefilter.py`, `test_collect_config.py` and `test_inc2_stream_report.py`.
- **The new checks fail on the code before this change.** With the modules reverted to the commit before the interim change (120 declared), 22 checks fail, and the configure test exits on the unknown verb. With them reverted to `main` after the interim change (1500 declared), 19 fail before the same exit; the deploy checks fail on `main`'s `deploy_funnel.sh` (no `db.py` shipped, this file not in the pre-flight), and `test_inc_ap_governance.py` fails two checks on `main`'s `db.py`.
- **Full suite.** All 146 `tests/*.py` files were run, on the change rebased onto the interim change. 145 pass. `test_brain_api.py` fails with the same failure on `main` before this change (`supervision_health.py` reads the reviewer), so the failure is unrelated.
- **Replay gate, run locally.** This is `executor.run_replay_tests`, which `record-replay` runs on the lab, run against the fixtures. It records `pass`: 56 cases pass, and R2 and R4b are the two skips the gate allows (their fixtures are not committed). Both `replay_status` and `stream_replay_status` pass.
- **No stored decision changes.** The recorded ledgers the gate replays (pilot_v1–v3, realloop_v1, the funnel's) contain no daily-cap, window or per-day refusal. The change therefore alters no decision they made; it only removes future refusals of that kind.

**Deploy.**
- **`tools/db.py`.** `deploy/deploy_funnel.sh` now ships `tools/db.py`, under `SHARED_RE`: if the file differs on the cluster, the deploy refuses unless `--allow-shared-change` is given. Without it, the lab's `db.py` may still declare `daily_cap` 120, or 1500 if the interim change (`11c01b0`) was deployed; `budget.domain_budget()` reads that file on the lab (where the ticker runs), and that cap would stay.
- **Pre-flight.** The deploy's local pre-flight set now runs `tests/test_stream_ap_no_throttles.py` as well as `test_inc_ap_governance.py`, so a tree that declares a default daily cap, window or per-day lever count again (for example a merge resolved to the interim `db.py`) is refused before anything is copied.
- **Hashes and records.** The change moves `executor.code_hash()`, so the replay must be recorded again (`record-replay`). It also moves the stream rules version (`stream_levers.json`, `diagnose_stream.py`); the ticker writes a new prospective stream record before the next L18. `inc2/stream.py` changes only `BUDGET`, which is written into a new stream's definition, so the live stream records a `code_change` event.
- **After the deploy, a person clears the live campaign's explicit caps:** `python -m weed_optimizer_framework.tools.inc_autopilot.stream configure --name weed_stream_v1 --by human:<email> --daily-cap-su none --window-cap-su none --collect-gb-daily none`.
- **Stored domain config.** Once `budget.domain_budget()` reads the stored domain config (planned), the weed domain's stored `budget` block must not declare a `daily_cap`. If it does, that cap applies to every campaign again.

**Left unchanged.** `round_scheduler`'s `max_rounds_per_day` (default 2) is a per-day count of the old round scheduler, which has been paused since 2026-08-29. It does not govern the stream campaign.

### 6.7 What keeps it running, and what stops it

- **Never idle while there is work:**
  - the DATA lane collects whenever the queue is below the low-water mark;
  - with no candidates, it re-runs discovery every 7 days (**WAIT_DATA**, an observing phase that never COMPLETEs);
  - the TRAIN lane cuts when supply allows;
  - the MAINT lane runs milestones and rollbacks.
- **COMPLETE** only when all of these hold:
  - D29 fires;
  - Q < M;
  - no lane item or job is in flight;
  - every accepted increment has been through a milestone;
  - no lever is proposable.

  [review] This conflicts with P7 ("never COMPLETEs by itself") and with the owner's goal of running continuously: D29 needs only one empty discovery within 14 days, while new public datasets appear over months, and WAIT_DATA costs no GPU. The stream campaign never COMPLETEs by itself. D29 moves the DATA lane to WAIT_DATA with discovery backing off 7 → 14 → 30 days and raises a card. The conditions above only make COMPLETE *available to a person*. S9 is changed to match.
- **[review] Holds that depend on the funnel need a deadline.** P8 (`funnel_F9`), P9 (the H6 scan) and MH-Weed16 (the funnel's H3a) all wait on the funnel campaign, which runs on its own but at its own pace (census passed 2026-09-28; leak step submitted). A hold with no deadline stalls a share of the supply for good. Each such hold carries `hold_deadline` (proposed: 21 days from the row's admission). At the deadline:
  - an `h6_scan` hold is served by the stream's own copy detector (§3.2);
  - a `funnel_F9` hold becomes an R3 item for a person (release, or keep holding);
  - the MH-Weed16 rejoin stays blocked and is listed in the stream report as supply waiting on the funnel.
- **PAUSE,** with a card, never COMPLETE:
  - the budget or the allocation is exhausted;
  - any stop-loss of §6.6 fires (P7).

### 6.8 Replay and scenario tests (`STREAM_REPLAY_CASES` in `executor.REPLAY_REQUIRED`)

| Case | Asserts |
|---|---|
| S1 | Queue below low water with candidates → exactly one L16 on the top-ranked source. The argv is byte-equal and the risk is R2. Never a NEVER_TRAIN, evaluation-lab, quarantined or image-level-only source. |
| S1b | L15 on recorded provider responses reproduces the D-C leads (CottonWeedDet3, PAGS8, MFWD, MH-Weed16, NDSU) by itself. Recall is reported as H12 is. |
| S2 | Yield below the floor after 2 GB → no further L16 for that source; a source above the floor continues. |
| S3 | Collect done → L17 for that source only, with the pinned verifier sha; refused (`retry: false`) on a verifier or LOCK mismatch. |
| S4 | Q ≥ 4M → the L18 argv is byte-equal; `inc2.stream`'s argparse reads it back; base = P_{s-1} sha; priced; passes `diagnose.prospective_guard`. |
| S5 | Commit: ACCEPT I2 and REJECT I3 (data) → P_1 = P_0 + I2, and I3 is quarantined and never cut again. A recipe-blamed REJECT and a HOLD return once, then become `neutral`. |
| S6 | **realloop_v1, pinned as a new fixture** (`tests/fixtures/inc_replay/realloop_v1/`) → D30 fires; TRAIN holds; cards X1 and X13; the DATA lane still proposes L16 when the queue is low; D33 names PricklySida. [review: under §3.5's guard-based reading, D30 does **not** fire (no REJECT is regression-only); the dispositions are s01, s03 data, s02, s04, s06 species and s05 flips; D31 fires on s01 and s03 (L4 first); D33 names PricklySida (5 of 6). The funnel's copy of the ledger (`tests/fixtures/inc_replay/funnel/realloop_v1/`) is reused by sha256 rather than duplicated.] |
| S7 | A synthetic data blame (recipe healthy, P_data 0.0 from source X) → L4, then L24 on X, then D21 closes X. |
| S8 | A milestone "hurts", or a failed boundary check → L20, then L21 to P_c; TRAIN holds. |
| S9 | No premature COMPLETE: an empty queue with a candidate above the floor and budget left → WAIT_DATA plus L16. Exhausted → COMPLETE with a residual card. Budget out → PAUSE. [review: exhausted → WAIT_DATA with backoff and a card, never COMPLETE (§6.7); allocation end date → PAUSE `allocation_ended`, and the renewal card 30 days before.] |
| S10 | D27 on the allocation fixture → pause. The daily, monthly and cross-campaign caps refuse. |
| S11 | D26 at a projected 0.8 × limit → no build; card X15. |
| S12 | Test blindness: every value of test and ImageWeeds is perturbed, and tsw rows appear only as training rows → identical diagnoses, argv and digest. |
| S13 | Domain-free: the stream code greps clean (the `test_funnel_domain_free.py` method), and a second domain (the R13 vehicles fixture) runs S1–S5 on its own config. |
| S14 | Leak: a staged source with evaluation near-duplicates → those images never reach the queue; a share ≥ 5 % → L24 and a card. |
| S15 | Limits: 10 collects on different sources do not pause; a 4th attempt on one source does. |
| S16 | Governance matrix for each new action and actor: R2 direct only with `data_autonomy` and a replay pass; R3 envelope only with the grant; R4 refused and shown as a card. |
| S17 | One ssh per tick with three lanes active (end-to-end on a simulated clock). |
| S18 | An advance of a v2 segment from the ticker exports `INC_JOB_SCRIPT=run_inc2_job.sh`; the v1 script is never submitted for a v2 experiment. |
| S19 [review] | pilot_v3 fixture: the planted Bswap (P_data 0.00, `blame: recipe` in the ledger) is disposed "data" and quarantined, and Breal is disposed "species" and returned once. The disposition never reads `attribution.blame`. |
| S20 [review] | Replay of the 2026-08/09 silent "+0" rounds: three consecutive sources ending with zero admitted target boxes → the DATA lane holds, and the card lists each source's `decisions.jsonl` reasons. Two zero-yield sources do not hold it. |
| S21 [review] | Replay of the "Invalid qos" refusal (FUNNEL_AUDIT_RUNNER note 33): an sbatch rejected on RM-shared is never retried there. The argv for every stream job names GPU-shared, and a qos rejection is recorded as a platform defect, not as a failed attempt of the source. |
| S22 [review] | Placement: a provider absent from `placement.json`, or failed there, gets the lab hook, never an sbatch. A lab fetch runs detached, and the tick returns within its budget while the fetch runs (the one-hour synchronous `funnel_fetch_hook` must not recur). |
| S23 [review] | Lab and cluster code drift (the stale `model_router.py` of 2026-09-27): the snapshot returns the cluster's sha256 of every `inc2`, `collect` and `inc_autopilot/stream*` module. A mismatch with the lab's copy refuses every stream submission and raises a card. |
| S24 [review] | Cutter deadlock: a queue of Q ≥ M images in capture groups that cannot sum to M is split by near-duplicate group to exactly M (§3.4). A queue that still cannot fill raises D22 with the reason, never a silent wait. |
| S25 [review] | Leak by augmentation: a staged source holding cropped (15 %), sheared and brightened copies of dev and test images, each more than 6 bits from its original under all 8 variants → `near_eval_embed` refuses them. Share ≥ 5 % → L24 and a card. A Roboflow re-upload declaring PricklySida is held with `h6_scan`. |
| S26 [review] | Holds: a `funnel_F9` hold past its deadline becomes an R3 item, and an `h6_scan` hold past its deadline is served by the stream's own detector; neither waits silently. |
| S27 [review] | Governance: a change to `collect/domains/weed.json` voids the replay pass; the platform never writes that file, `floor_gb`/`floor_su` or `thresholds.json`. |
| S28 [review] | Old paths closed: while a stream campaign owns the weed domain, `round_scheduler` refuses `collect` as well as `train`/`filter`; the dashboard's harvest button refuses with `AUTO_SYNC=1`; `roboflow_sync.cmd_sync_newest_slugs` skips `status: intake` (real function, not a mock). |
| — | A mutation harness over D20–D33's comparisons, in which every mutant is killed by some S-case (as `test_funnel_ap_mutations.py`). |
| — | A prospective stream record: before the first L18, the sha of (M, K, truth policy, thresholds, recipe rule, gate block) goes to the ledger (as R4b). |

---

## 7. Targeted collection

### 7.1 Target taxa and search terms

The queries are generated from the config. Their sources are:
- the funnel domain's targets (taxon, genus, `not` siblings);
- GBIF vernacular names from the taxonomy cache;
- `cwd12_species._ALIASES`.

Nothing is coded as a seed list. `dataset_discovery._cwd12_species_queries` (116) is not reused.

**Per-provider templates:**
- `"<binomial>"`, `"<common>"`, `"<common> weed"`, `"<common> detection"`, `"<genus> dataset"`;
- EPPO codes for mediaTUM and Weed-AI;
- class-name search on Roboflow Universe, the only provider that declares class lists before download.

| Class (INC id) | Binomial (DEC-3 policy) | Common names searched | EPPO | Box supply after D-A (train_core + tsw) | Priority |
|---|---|---|---|---|---|
| PricklySida (7) | *Sida spinosa* | prickly sida, prickly mallow, teaweed | SIDSP | 263 (none added) | **1** |
| CutleafGroundcherry (11) | *Physalis angulata* | cutleaf groundcherry, wild tomato | PHYAN | 50 (none added) | **1** |
| Sicklepod (9) | *Senna obtusifolia* (syn. *Cassia obtusifolia*) | sicklepod, coffeeweed | CASOB | 121 (+41 in B's harvested part) | **1** |
| Eclipta (6) | *Eclipta prostrata* (syn. *E. alba*) | eclipta, false daisy | ECLAL | 512 + 33 + ≤ 17 | 2 |
| Goosegrass (10) | *Eleusine indica* | goosegrass, Indian goosegrass | ELEIN | 107 + 373 + ≤ 17 | 2 |
| SpottedSpurge (3) | *Euphorbia maculata* | spotted spurge, prostrate spurge | EPHMA | 550 + 345 + ≤ 17 | 2 |
| Carpetweed (4) | *Mollugo verticillata* | carpetweed, green carpetweed | MOLVE | 530 + 673 + ≤ 17 | 3 |
| Ragweed (5) | *Ambrosia artemisiifolia* | common ragweed | AMBEL | 399 + 120 + 155 | 3 |
| MorningGlory (1) | *Ipomoea* spp. | morning glory, ivyleaf, pitted, tall morningglory | IPO* | 792 + 448 + ≤ 17 | 3 |
| PalmerAmaranth (8) | *Amaranthus palmeri* | Palmer amaranth, Palmer pigweed | AMAPA | 206 + 170 + 1,571 | 3 |
| Purslane (2) | *Portulaca oleracea* | common purslane, pigweed (regional) | POROL | 440 + 130 + 2,391 | 3 |
| Waterhemp (0) | *Amaranthus tuberculatus* | waterhemp, tall waterhemp | AMATU | 1,059 + 901 + ≤ 17 | 3 |

"≤ 17" is data2023's cap for its minor species. D20's first deficit list is PricklySida, CutleafGroundcherry and Sicklepod.

### 7.2 Providers, placement and the pre-download filter

- **Providers** (`collect/providers/`): `hf`, `kaggle` (bearer token), `roboflow` (Universe search and export), `mendeley_zenodo` (by importing `funnel.fetch.fetch_spec`), `weedai`, `ftp` (mediaTUM) and `github`.
  - Each gets a byte cap, a timeout scaled to size (not the flat 480 s of `extra_sources.py:399`), and checksum verification where the provider publishes one.
  - A stream that breaks off resumes rather than starting over (`collect/transport.py`). On 2026-10-02 the server closed the stream of zenodo_15808623 (49.7 GB) after 9.1 GB. The size check refused the file and deleted all 9.1 GB, and it was the second failed fetch in a row, so the stream's stop-loss held the data lane. Now the partial file and its running hashes are kept, and the rest is requested with `Range: bytes=<n>-` (FTP: `REST <n>`). Only a 206 whose Content-Range continues the file is used; with no size known, a 206 of total `*` must still reach its own last byte. A 200 to a Range request (the server ignored it), a 416, a Content-Range that does not continue the file, or a changed strong ETag starts the file over once, and a second one fails, so a server that never honours a Range moves the file about twice, not once per resume. Each file gets at most `download.resume_attempts` resumes (default 20). The deadline and the byte cap cover the whole file. Only a read or a transfer counts as a network error: a failed write to the `.tmp` (a full disk, a quota, EIO) fails the download at once with nothing left, and the `.tmp`'s size on disk must equal the bytes hashed, so a chunk that never reached the disk cannot pass under the published checksum. A refused or broken first FTP login is a ProviderError, and a failed login closes its connection. The final size and checksum checks are unchanged, and `fetch.json` records each file's `resumes`. Tested in `tests/test_collect_resume.py`.
  - A lab fetch's wall clock is sized from the bytes it may move (`inc_autopilot.stream.lab_job_timeout`): 2 h plus `max_bytes` at 1 MB/s, between 6 h and 48 h; other lab jobs keep 6 h. The flat 6 h limit would have cut the 49.7 GB fetch above near its end (about 5.8 h at the 2.4 MB/s measured), and resuming works only inside one process, so a killed fetch starts over. Tested in `tests/test_stream_ap_lab_timeout.py`.
  - A lab job killed from outside is noticed (`LabRunner.poll`): `launch` records the lab-run's pid, and a job with no result whose lab-run is gone (by pid, else by any process naming its spec) polls a failure marked `lost`, so its lane fails the step and proposes again instead of following a dead job for ever. On 2026-10-03 a deploy restart (systemd `KillMode=control-group`) killed the zenodo_15808623 fetch and left its L16L item 'running'. The dashboard unit now uses `KillMode=process` (`deploy/weed-dashboard.service`), so a restart stops uvicorn only and lab jobs survive a deploy. Tested in `tests/test_stream_ap_lab_lost.py` (real processes).
  - A lost lab job is charged to nobody: it is recorded `failed` with `lost: true`, the lane is freed and the work proposed again under a new id; it counts neither as the source's failed attempt (7.5: three close the source), nor toward the lane's stop-loss, nor as a collection attempt toward the 4th-attempt pause (S15). The same step lost `LOST_RUNS_MAX` (3) times in a row counts as a failure, so a job that is always killed still ends. A person reopens a closed source with `inc_autopilot.stream reopen --name N --source S --by human:<id> --why TEXT`: the config stamps it and the next tick makes the source a candidate with failures and attempts reset, once per stamp, recorded as `source_reopened` with who and why; a stamp older than a later closure does nothing. On 2026-10-03 zenodo_15808623 was closed by three 'failed attempts': a refusal over the collector's daily byte cap, a download the server broke off before resume existed, and the deploy restart's kill.
- **Amendment 2026-10-03: a large source reaches the queue (zenodo_15808623, checked before its fetch ended).** Its 203,567 images sit in one 49.7 GB zip, with Pascal VOC classes named `<EPPO>_week_<n>` (16 species, 11 weeks). Three defects would each have stopped it after the fetch, and all three were fixed before the fetch ended.
  - **Class names led by an EPPO code.** `collect.classmap` maps a class name whose leading token is an EPPO code before a separator by that code's binomial (basis `eppo_prefix`), but only when the pinned table holds the code. The EPPO table v2 (`collect/domains/eppo_codes_v2.json`) adds ABUTH, PANDI, SETFA, SETPU and SORHA, each checked against gd.eppo.int on 2026-10-03. v1 is kept, so earlier provenance still resolves. A rehearsal on the lab with the real caches mapped the four target species (Palmer amaranth, waterhemp, ragweed, prickly sida). The other twelve binomials were not in the names layer.
  - **The names round.** An intake refused with names_pending (a NamesPending refusal, exit 2, with a `held` event that carries the names) used to end as a charged failed step. The retry met the same refusal, and two failed steps in a row held the DATA lane. The source became `held`, and nothing proposed L26.
    - Now the snapshot's fold carries the hold's names and time (`pending_names`, `pending_ts`), and a new hold makes the source `names_pending`.
    - Its L16I job's end is recorded as `intake_names_pending`. It is not a failure and not a stop-loss step, and its id is retired.
    - DPIPE proposes L26. The L26 lab hook writes the names to the lab's `intake/work/<source>/pending_names.json`, which `collect names` reads. DPIPE then proposes `L16S --names`, after which the source is `fetched` again and L16I runs under a new id, reusing the extracted tree.
    - A source still pending after `NAMES_ROUNDS_MAX` (1) round is `held` for a person, with a card. `stream reopen` of such a source (after a person maps its names) makes it `fetched` with its rounds reset.
    - D20 never fetches a `names_pending` source.
    - Only an intake's hold (`stage: intake`) opens a round; a fetch's names hold keeps the pre-fetch path.
    - Opening a round raises the attempt count of its L26 and `L16S --names` steps, so a source whose names were resolved before its fetch does not meet an id the executor already ran.
    - An L16I whose job log carries the NamesPending refusal (`(lever L26)`) also enters the round when the collector's ledger was read before the job ended; the round waits for the fold's names.
    - An L26 that resolved only part of its names (`collect names` status `partial`) runs again, up to `NAMES_PARTIAL_MAX` (2) times, before the round counts.
    - The intake's hold now carries up to 200 names (it was 50).
    - Live, the round ran end to end on zenodo_15808623: intake → `intake_names_pending` → L26 → `L16S --names` → re-intake. The re-intake then pended on `ECHCG_week_<n>`: the binomial's answer was `unresolvable`, so the class map fell through to the raw name, which no cache holds, and the source went to a person after its round. Two changes followed. The source's card now maps all its classes. And `collect.classmap` gives a class whose code or taxon already had an answer (non-informative included) that answer, instead of waiting on the raw name.
  - **Intake cap.** One intake batch takes at most `budgets.intake_max_images` (40,000; 50,000 since the continuation-shard amendment of §3.2) images with a box.
    - The images are drawn round-robin over class sets, then capture groups, then a seeded order (`collect.intake.cap_items`; the seed is the fetch record's sha256). The rest are recorded as decision `deferred` (`over_intake_cap`), and the summary carries `intake_cap` and `yield.images_deferred`.
    - Why: measured intakes took 0.15–0.66 s per image (PAGS8: 614 images in 2 min 40 s; CottonWeedDet3: 795 in 8 min 48 s). At that rate the full source needs 8.5–37 h, against the job's 8 h. Its images are frames of 360° videos.
    - This matches the base v3.1 design (SIU capped at 40K, stratified by species × stage × capture group). The deferred images are taken by later intakes of the same fetch record, as continuation shards (§3.2, amendment 2026-10-03).
    - `budgets.intake_max_seconds` (6 h of the job's 8 h): after it, the images not yet judged are deferred (`over_intake_time`) and the batch commits, rather than timing out with nothing. The summary carries `intake_time`.
    - In `collect.classmap`, `eppo_prefix` comes after a format's hints, so a taxon the annotation carries outranks a code read off a name's first token.
  - Tests:
    - `test_collect_classmap.py`;
    - `test_collect_intake.py` (`test_intake_cap`);
    - `tests/test_stream_ap_names_round.py`, which covers the stream world: the fold, the round, a refusal line before the fold, a partial L26, held then reopened, and fresh ids after a pre-fetch L26.
- **Network probe (R0 lever, one GPU-shared job, minutes).** It tries each provider host from a compute node. The result goes to `INC_DIR/intake/placement.json`.
  - Until a provider passes, it runs on the lab (as `funnel/fetch.py` does, since it refuses inside Slurm), and the files are synced to the cluster.
  - GitHub is lab-only: the login-node SOCKS proxy is refused from compute nodes.
- **Pre-download filter (`collect/prefilter.py`).** The decision is made from declared class names: Universe `classes[]`, the HF ClassLabel or card, Kaggle metadata, or the card resolver. A candidate is kept only if:
  - (i) ≥ 1 class name has `names.py` status `target` or `target_synonym` through the offline resolver; or
  - (ii) it is on the known-items list; or
  - (iii) it has an owner-approved card class table.

  The rest of the filter:
  - The substring accept-vocabulary is dropped.
  - The reject list applies to **whole tokens of class names**. This fixes "car" → carpetweed, "bee" → sugarbeet, "drone" and "flower".
  - A probable cwd12 re-export (≥ 4 class names equal to `CWD12_LEGACY_LABELS`, the rule of `verify.old_join`) is recorded as a copy candidate and not downloaded.
  - Candidates are ranked by target classes per GB, not by image count.
  - **[review] The legacy-label rule misses the re-exports that matter most.** A re-export with corrected names (after v3.60 the right names are public), or a subset of cwd12's classes, has fewer than 4 legacy labels. PricklySida and CutleafGroundcherry have no known public source outside the lab of dev and test (§7.3), and D33's ranking bonus steers discovery toward exactly the candidates that declare them. So any Roboflow Universe, Kaggle or HF re-upload that declares PricklySida, CutleafGroundcherry, or ≥ 3 cwd12 species is presumed to be a LuLab derivative: lab group LuLab, fetched, and every row held with `hold_until: h6_scan` until the embedding copy scan of §3.2 has run. The v1 pool shows the pattern (`step1/pool_summary.json` `per_slug`). Of the 20 slugs with any evaluation near-copy or cwd12 copy, 4 kept no image at all because they are wholly cwd12 (`cottonweed_sp8`, `cottonweed_holdout`, `rf_agrobot-weed-workspace__weed-detection-sd89f`, AgML `three_season`: 1,180 to 2,846 cwd12 copies each). Two partial re-uploads, cwp10 and vanpe, kept 1,460 and 872 images after their near-exact copies were removed, and they supply 812 of base B's 878 (§4.2).

### 7.3 Known sources (D-C) and expected yield

These are recorded in `collect/domains/weed.json` as `known_items` with `decided_by: owner 2026-09-28 D-C`. Their roles:
- **Audit of the search.** They measure discovery recall (S1b), so they are not a seed list.
- **Fetchable.** Each stays fetchable by L16 even if discovery missed it. A miss is recorded as a discovery defect and raises a card.

| Source | Access and licence | Code needed | Raw target boxes | After guards and verification (est.) | Handling |
|---|---|---|---|---|---|
| **MH-Weed16 full** (Mendeley `d3n3mgjjbv` v2, CC BY 4.0) | public API | exists in the funnel: card resolver, `fetch.refetch` (R-F) | pool: id5 1,634, id12 307; R-F adds about 540 id5 and 100 id12 (proportional) | about 330 MorningGlory + 16 Sicklepod from R-F; the pool rejoin is not yet estimated (census: 1,007 of 1,634 id5 and 50 of 307 id12 confidently the target) | Owned by the funnel. Waits for H3a. Rejoin by `step1_stream`. The Kaggle BY-NC-SA copy is not used. |
| **PAGS8** (Weed-AI `5c78d067…`; the GitHub repo is code only, MIT) | public; a scripted download URL is not confirmed; possibly several parts | `weedai` provider, WeedCOCO → YOLO, card table (8 growth stages → *A. palmeri*) | 5,026 Palmer in 1,228 images (paper; which part is unconfirmed) | 1,500–2,500. Out-of-domain Palmer was verified at 46 % (346/755). Stage-1 seedlings fall under `MIN_BOX_PX`. | TAMU group. `exhaustive_labels: false`. |
| **MFWD POROL** (mediaTUM 1717366, CC BY 4.0) | public FTP (`download_by_ftp.py`) | `ftp` provider, `gt.csv` converter | about 9,400 (4,435 images at 2.12 boxes/image; about 5.4 GB) | 1,900–5,600, low diversity (21 greenhouse trays) | TUM group. Capture group = tray. Lab placement until the probe passes. |
| **CottonWeedDet3** (Kaggle `yuzhenlu/cottonweeddet3`, 5.18 GB, 848 images) | Kaggle token (rotation unconfirmed; the key is in public git history) | Kaggle size-scaled timeout, class-name read, layout normaliser | 1,532: Carpetweed 602, MorningGlory 486, Palmer 444 | 700–1,450 | **LuLab (the lab of dev and test): held until the H6 copy scan runs (P9).** |
| **NDSU greenhouse / weed_crop classification sets** | Mendeley; which records they are is not identified | none admitted | 0 (image-level) | 0 | NDSU group (same lab as ImageWeeds). Kept for recall only. An L16 pre-check files them R3 ("image-level labels only"). |
| **CottonWeedID15** (Kaggle `yuzhenlu/cottonweedid15`; AgML HF copy) | public | none admitted | 0 (5,187 images, image-level) | 0 | LuLab. A pseudo-box protocol would be a new protocol version. |

- **Total.** About 16.6k raw target boxes can be reached without a person. About **4.4k–10.4k** of them are expected to be verified (est.), against 2,049 verified from the whole earlier harvest.
  - [review] "Without a person" holds only for PAGS8 and MFWD POROL (about 14.4k raw boxes; Palmer amaranth and Purslane only), and only if their class maps resolve without a person (the EPPO table and the PAGS8 card table, §3.2) and PAGS8's scripted download works. MH-Weed16 waits on the funnel's H3a, and CottonWeedDet3 on the copy scan. The funnel campaign is paused (2026-09-28), so both need the §6.7 hold deadlines or a person.
  - [review] MFWD's 4,435 images come from 21 greenhouse trays photographed repeatedly over time, so the number of distinct plants is far below the image count. An increment of 763 MFWD images is about 3.6 trays. Its effective sample size, and its greenhouse domain (pots, top-down, controlled light), make "neutral" or "hurts" the more likely truth-arm reading. `masked_area_frac` and the capture-group count per increment are reported, so this can be checked.
- **Coverage.** The new data covers only Carpetweed, MorningGlory, Palmer amaranth, Purslane and Sicklepod.
- **Gaps.** PricklySida and CutleafGroundcherry get nothing, and Sicklepod about 100 boxes. Only targeted discovery can close them, starting with Roboflow Universe class search.

### 7.4 Yield accounting

Per source, `INC_DIR/intake/sources.jsonl` and the stream report record:
- bytes downloaded;
- images and boxes by declared class;
- boxes per INC id after the class map;
- guard drops by reason;
- the licence;
- verified target boxes per species after D-B;
- target images admitted;
- target images cut and their dispositions (accepted, rejected with blame, returned, neutral);
- the SU of collect + admit, measured.

**Yield** = verified target boxes admitted per GB and per SU. `floor_gb` and `floor_su` are set in thresholds.json before L16 becomes autonomous. They come from the measured bytes and SU of the first D-C wave, and a placeholder value is refused (§10 R3).

### 7.5 Per-source stop rules

A source is **closed** (no more shards, never retried) on any of:
- D21, the yield below the floor after ≥ min(2 GB, the whole source);
- 3 failed attempts;
- D28, a leak;
- D31 blaming it twice;
- a licence that resolves to "not research-usable".

A source is **held** (waiting, not closed) on any of:
- an unresolved licence;
- missing credentials;
- a same-lab copy scan still pending;
- the funnel ordering (P8, P9).

---

## 8. Safety invariants and the tests that prove them

| Invariant | Enforced by | Proved by |
|---|---|---|
| **Never-train v2.** No dev, test or ImageWeeds image, or a copy within 6 bits including the 8 flips and rotations, reaches any training manifest. | `inc2.guard` at intake; `step1_stream` (on both the unmasked and masked dHash); `inc2.stream` at cut and build; `inc2.train.guard_rows`, fail-closed; LOCK v2 | Planted exact, 6-bit, hflip and rot90 copies are refused at each of the four entry points (`test_inc2_guard.py`, `test_inc2_step1_stream.py`, `test_inc2_stream.py`, `test_inc2_train.py`); S14 |
| **Test blindness.** No decision reads test or ImageWeeds. | `evidence.py` allow-list; test scored only at milestones | S12 metamorphic replay; OP_HALT test |
| **Base copies never return as increments.** | `base_copies_dhash.json` in `GuardV2`; quarantine by dHash | planted base-copy test; S5 |
| **Evaluation manifests unchanged.** | byte copies checked against the v1 LOCK; the v1 scorer unchanged | `test_inc2_splits.py`; `inc.splits.verify()` still [] after the v2 build |
| **Licences.** Every admitted image has a recorded licence; unresolved → held; NC → `research_only`; nothing is redistributed. | `collect/licence.py`; queue and manifest fields | licence gate cases; a grep test that `collect/` never calls a Roboflow upload |
| **[review] Research-only data taints the model.** The end use is a deployed laser-weeding robot. A model trained on any `research_only` image is itself research-only. | Each pool P_s records its count of `research_only` rows; every segment's and milestone's `report.json` carries `research_only: true/false` for its models; the model gateway and robot uplink read that flag before deployment. | a pool with one NC row yields models flagged `research_only`; the stream report lists the flag per milestone |
| **[review] The old harvest cannot feed the pool, Roboflow or `mega_trainer` behind the stream's back.** | while a stream campaign owns the weed domain: `round_scheduler` refuses `collect` as well as `train`/`filter`; the dashboard's harvest route refuses `AUTO_SYNC=1` (`dashboard_server.py` harvest_full_round_e2e); `step1_stream` reads only v1-known slugs (§3.3); intake entries have `status: intake`, which `roboflow_sync.cmd_sync_newest_slugs` skips (it takes only `status == "downloaded"`, `roboflow_sync.py:406`) | S28 |
| **Pinned modules untouched.** | file ownership (§9); the `inc2` package | `test_pinned_unchanged.py`: sha256 of the 12 pinned files equals the values recorded in realloop_v1's `state.json.code` and the funnel's pins. [review] realloop_v1's code pin holds only the driver's 6 `PINNED_MODULES` (`driver.py:257`: `inc/__init__`, common, driver, gate, splits, `cwd12_species`). No recorded pin exists for lora, train, verify, select, relevance or audit: their outputs record no code hash, and only train, lora and scorer appear in run.json `CODE_MODULES`. The reference values for all 12 are therefore the blobs at commit 385ca77 (`git show 385ca77:<path> \| sha256sum`), written once into `tests/fixtures/pinned_modules_385ca77.json`. The 6 driver pins and the scorer's LOCK sha `18c00837…` are cross-checked against that file. |
| **No `mega_trainer` training.** | `collect/` imports neither `mega_trainer`, ultralytics, `inc.train`, `inc2.train` nor `inc.lora`; `intake_v1` keeps new sources out of `mega_trainer._merge_datasets`; `round_scheduler._advance` refuses `round_train`/`round_filter` for a domain a stream campaign owns; the weed `train`/`filter` fallback in `_WEED_STEPS` is removed; `policy_actions.round_train`/`round_filter` are `allowed_tiers: ["human"]` | `test_collect_no_train.py` (import graph); `test_round_scheduler_stream_guard.py`; a real-function test that `verify._skip_reason` and `mega_trainer`'s filter skip `intake_v1`. [review] `GuardV2` hashes through `inc.common.dhash`, which imports `mega_trainer._dhash`, and `mega_trainer` imports `dataset_discovery`, `registry_lock` and `inc.common` at module load (`mega_trainer.py:16-24`). So `collect → inc2.guard` imports `mega_trainer`, and the import-graph test as written fails by construction. The rule is instead: `collect/` and `inc2.guard` may reach `mega_trainer` only through `inc.common.dhash`. A test asserts that no other `mega_trainer` attribute is referenced, and that `ultralytics` is not in `sys.modules` after importing `collect` and running a guard check. |
| **No code drift under running jobs.** | no loop script runs `git reset` or copies the nested package to outer; job scripts compare outer against nested module hashes and refuse on a mismatch | `test_inc2_job_scripts.py` greps and hash-check tests |
| **Provenance.** | Every training row traces image → intake batch → source → fetch checksum and licence → verdicts and mask record → increment → segment verdict → pool P_s. Ledgers are append-only and hash-chained. | a round-trip test from a P_s row back to its fetch record; a broken-chain refusal test |
| **One writer per state directory.** | locks and leases (§3.8) | a racing append-vs-advance test on FakeBackend |
| **The gate decides.** | the pinned driver and gate; commit only reads the recorded verdicts | S5; the commit refuses an unfinished segment |
| **The platform reproduces the recorded manual interventions.** | the existing replays plus realloop_v1 as a fixture | the existing `test_inc_ap_*` and `test_funnel_ap_*` pass unchanged; S6 |

---

## 9. Build plan: six groups with disjoint files

Each group owns its files and tests and edits nothing another group owns. The interfaces are the CLIs and file formats of §3, fixed by this contract, so the groups build in parallel against stubs.

### Group A — the protocol package and splits v2

**Owns:**
- `inc2/__init__.py` (empty), `inc2/common.py`, `inc2/guard.py`, `inc2/splits.py`, `inc2/embed_calibration.py` (L-9);
- `run_inc2_splits.sh`;
- `tests/test_inc2_common.py`, `tests/test_inc2_guard.py`, `tests/test_inc2_splits.py`, `tests/test_inc2_embed_calibration.py` (L-9);
- `INC_DIR/splits/v2/`.

**Provides:** `inc2.common`; `GuardV2.load(lock_path)`, `.check(dhash, variants) → (reason | None, match)`; `inc2.guard.dhash_variants(path)`.

**Acceptance:**
- the three byte copies match the v1 LOCK shas, and train_core is v1's minus the L-8 list (L-8);
- 5,802 index entries;
- tsw22 has 1,915 rows and tsw23 1,784 minus the recorded drops;
- every planted copy kind is caught;
- a planted copy in B's part refuses the build;
- [L-8] a planted transverse copy of a test image in train_core is dropped, listed and refused by `inc2.train` under another key; more than the cap refuses the build;
- [review] a planted 15 % crop, shear or brightness copy of a test image in B's part, more than 6 bits away under all 8 variants, is caught by the embedding scan before lock;
- [review] a planted tsw row from a dev session is dropped, and a row sharing a session with test is counted (L-9(a): and dropped);
- [L-9] a tsw row of another capture session with high scene similarity is kept; a base B image the funnel lists against an ood split is not a v2 leak; the v2 calibration rises above planted same-scene negatives while an augmented copy of a test image is still caught; intake binds the v2 calibration;
- [review] `inc2.common` resolution test (§4.3): v2 training paths, v1 evaluation paths, and no argument-less `NeverTrainGuard.load()`;
- the files are 0444 after lock;
- v1 `splits.verify()` is still [];
- a synthetic world builds end to end with `--testing`.

### Group B — executor, recipes, baselines and Stage A

**Owns:**
- `inc2/recipes.py`, `inc2/train.py`, `inc2/baseline.py`, `inc2/pilot4.py`;
- `run_inc2_job.sh`;
- `tests/test_inc2_train.py`, `tests/test_inc2_baseline.py`, `tests/test_inc2_pilot4.py`;
- the "Protocol v3" sections of docs/INCREMENTAL_PROTOCOL.md and docs/INCREMENTAL_PROTOCOL_RUNNER.md.

**Acceptance:**
- the v3 recipe table (cold, R0, X1a, X1b) refuses deviations;
- ood exams are refused;
- a planted dev copy is refused by `guard_rows`;
- the job script exports `INC_JOB_SCRIPT` as itself and runs `inc2.train`, then `inc.driver advance`;
- pilot_v4's bins are sha-identical to pilot_v3's `exp.json` entries, except for rows whose image bytes are L-8 drops, which are removed and recorded; it has no truth arm;
- [L-8] the canary builds on v1's train_core minus the L-8 list, records the dropped rows, and its verdict accepts that manifest by b0_v1's base sha256 plus the list's sha256;
- baseline builds pass the pinned `driver.validate_definition` and `check_definition_data`;
- a FakeBackend end-to-end test (driver init + advance of a baseline and of a 1-step chain) submits only `run_inc2_job.sh`.

### Group C — incremental Step 1

**Owns:**
- `inc2/step1_stream.py`, `inc2/mask.py`;
- `run_inc2_stream.sh`;
- `tests/test_inc2_step1_stream.py`, `tests/test_inc2_mask.py`;
- `INC_DIR/step1_stream/`.

**Acceptance:**
- **Equivalence:** on the `test_inc_verify` synthetic world, with the image rule, ingesting from empty equals `verify` pool + crops + admit (keys, label sha256, box verdicts, admitted set).
- **Split invariance:** two batches give the same queue as one, apart from the ids.
- **Mask:**
  - kept pixels unchanged and masked pixels equal to the mean colour;
  - overlap refused;
  - byte-identical to `recover.mask` without overlap.
- **Guards:** each planted case is caught.
  - [review] Including an augmented (crop, shear, brightness) evaluation copy through `near_eval_embed`, and the rotation variant of a masked PNG made from an EXIF-rotated original.
  - [review] A registry slug absent from the v1 pool is refused. A v1 slug with evaluation near-copies (cwp10, vanpe) yields rows with `hold_until: h6_scan`, and a row with no card licence yields `hold_until: licence`.
- **Pins:** an altered `verifier.npz`, a different embedder, a different v2 index sha and canary drift each refuse before any write.
- **State:**
  - global crop ids are contiguous and disjoint;
  - rerunning a committed batch is a no-op;
  - a killed batch resumes.
- **b0000 reconciliation:** all 457 candidates are accounted for (whole + masked + refused_overlap + not admitted = 457), and the recovered boxes per species are ≤ the census values.
- **Status schema** test.

### Group D — targeted collector

**Owns:**
- `tools/collect/**`: `__init__`, `__main__`, `targets.py`, `prefilter.py`, `providers/*.py`, `licence.py`, `normalize.py`, `classmap.py`, `intake.py`, `probe.py`;
- `collect/domains/weed.json`;
- `run_inc_collect.sh`;
- `tests/test_collect_*.py`, including `test_collect_no_train.py` and `test_collect_domain_free.py`;
- `INC_DIR/intake/`.

**Acceptance:**
- **Prefilter:**
  - carpetweed is not rejected by "car", nor sugarbeet by "bee";
  - species-only names pass;
  - a class list with ≥ 4 legacy cwd12 labels is flagged as a copy candidate.
- **Normalisers:** fixtures for COCO, VOC, VIA, WeedCOCO, MFWD `gt.csv` and YOLO.
- **Class map:**
  - MH-Weed16 card id5 → MorningGlory (DEC-3) and id12 → Sicklepod;
  - numeric names without a card → 13;
  - provenance shas recorded.
- **Licence:** gate cases per P6.
- **Intake:**
  - a `decisions.jsonl` row for every candidate, including rejections;
  - an `intake_v1` registry entry that the real `verify._skip_reason` and `mega_trainer` filter skip.
- **Imports:** the no-train import graph holds.
- **Script:** `run_inc_collect.sh` has no `git reset`, no nested-to-outer copy and no Roboflow sync; it calls `logging.basicConfig`; it accepts only plan, fetch, intake, summary and probe.
- **Recall:** S1b recall on recorded responses.

### Group E — the stream builder, ledger and reports

**Owns:**
- `inc2/stream.py`, `inc2/stream_report.py`, and later `inc2/driver3.py` (R5 only);
- [review] `run_inc2_build.sh` (the v2 build script; see "Files the plan leaves unowned");
- `tests/test_inc2_stream.py`, `tests/test_inc2_stream_report.py`;
- `INC_DIR/stream/`.

**Acceptance:**
- **Cut:** exactly M, deterministic under `stable_int`, whole groups, target images only, holds honoured, quarantine re-entry refused by dHash. [review] Also: large capture groups that cannot sum to M are split by near-duplicate group, never waited on (S24); the one-species cap applies when two or more sources are eligible; commit dispositions follow the guard-based rule of §3.5 (S19).
- **Build:** its `exp.json` passes the pinned `validate_definition` and `check_definition_data`.
- **Commit:** the S5 semantics hold, and an unfinished segment is refused.
- **Rollback:** marks increments suspect.
- **Ledger:** the hash chain verifies and a broken chain is refused.
- **Lease:** a single-writer lease.
- **End to end:** a FakeBackend segment with 2 steps (ACCEPT, REJECT) → P_1.
- **Milestone:** the build and the 5 v 5 comparison.
- **Schema:** `queue_summary.json`.

### Group F — autopilot stream mode and the scheduler guard

**Owns:**
- new files: `inc_autopilot/stream.py`, `diagnose_stream.py`, `levers_stream.py`, `stream_domains/weed.json`;
- edits: `campaign.py` (mode dispatch, goal `continuous`, `DEFAULT_CONFIG` fields), `executor.py`, `budget.py`, `remote.py` (stream verbs, `projects`, quota, `SUBMIT_FORMS`), `evidence.py`, `model.py` (per-campaign domain), `levers.json`, `thresholds.json`, `brain/policy_actions.json`, `brain/approvals.py`, `brain/su_rates.json`, `round_scheduler.py`;
- `tests/test_stream_ap_*.py`, `tests/test_round_scheduler_stream_guard.py`, `tests/fixtures/inc_replay/realloop_v1/`.

**Two deliveries:**
- **F1, needed for R0:** `protocol_package` rendering, L23, L25 and the network probe.
- **F2:** everything else.

**Acceptance:**
- S1–S18 and the mutation harness; [review] S19–S28 as well, and the full replay set (inc, funnel and stream) passing locally before any deploy (§6.5);
- the existing autopilot and funnel suites pass unchanged;
- an experiment-mode campaign's digest and argv are byte-identical before and after;
- `round_scheduler` refuses weed train/filter while a stream campaign owns the domain.

### Dependencies and shared-file rule

| Group | Needs |
|---|---|
| A | nothing |
| B | A (`inc2.common`, `GuardV2`) |
| C | A |
| D | A (`GuardV2`) |
| E | reads C's queue format and B's `baseline` CLI |
| F | calls every CLI by argv |

- No two groups edit the same file.
- The pinned INC modules, `realloop.py`, `pilot.py`, `funnel/**` and `funnel/domains/weed.json` are read-only for all six. Editing the funnel domain file would make the running audit's inputs stale (`_funnel_cards_stale`, `campaign.py:3071`).
- **[review] Files the plan leaves unowned or shared:**
  - **`run_inc_build.sh`.** L18, L20, L23 and L25 are written "in `run_inc_build.sh`", but that script accepts only `pilot build | pilot build-baseline | realloop build` (`run_inc_build.sh:88-91`), hashes only v1 modules, and serves the live funnel's builds. No group owns it. It stays unchanged, and group E owns a new `run_inc2_build.sh` (verbs: `inc2.splits build|lock`, `inc2.baseline build`, `inc2.pilot4 build`, `inc2.stream build|milestone|fork`), which hashes the `inc2` modules as well as the v1 modules. `remote.py`'s `SUBMIT_FORMS` (group F) name it.
  - **docs/INC_AUTOPILOT.md** (the stream-mode section, levers L15–L27, D20–D33, S1–S28): group F.
  - **CHANGELOG.md, README.md, RESEARCH_LOG.md**: every push needs all three, so they are edited only by the integrator, after the six groups, from each group's handover notes. No group edits them.
  - **`executor.GOVERNANCE_FILES`, `REPLAY_REQUIRED`**: group F (§6.5).
  - **`tests/fixtures/inc_replay/`**: group F owns the stream fixtures; group E's end-to-end tests use FakeBackend worlds of their own, not the replay fixtures.
- **[review] Pinned-module check of the plan itself.** No group edits any of the 12 pinned modules. Group F edits `round_scheduler.py`, `brain/*` and `inc_autopilot/*`, none of which is pinned. The only reads of pinned code at run time are the `inc2` copies (`inc2/train.py` is a copy of `inc/train.py`, not an import that changes it) and library calls. `test_pinned_unchanged.py` (§8) runs in every group's acceptance, not only the integrator's.

---

## 10. Rollout

Each step starts only when the previous one's acceptance holds, except where noted. Every experiment is submitted by the platform.

| Step | What (by the platform) | Needs | Acceptance | GPU-h (est.) |
|---|---|---|---|---|
| **R0** splits v2, baselines, Stage A | L23: splits build and lock (one approval by a person of the D-A command); B_v2, 5 seeds; the canary; B0 ∪ tsw (recommended); L25 pilot_v4; the network probe | A, B, F1 | §4.2 checks pass; LOCK v2 written; canary within 1 sd; B_v2 finals scored; Stage A verdict recorded READY or not; `placement.json` written | 25–27 required, +3.7–4.2 recommended, +0.3–0.6 optional |
| **R0b** [review] copy scan and gate feasibility | the embedding copy scan of base B's 878 and of tsw22/tsw23 before `lock` (§4.2 step 5; one GPU-shared job); L28 Stage C after Stage A READY | A (scan), E + F1 (L28) | scan result in LOCK v2; Stage C verdict recorded. A species-guard-only REJECT of the known-good tsw increment raises X17 to the owner before R2 | about 1 (scan) + 10.6–14.5 (Stage C) |
| **R1** Step 1 on v2 | `bootstrap`; `knowntruth` over the 1,055 former near-eval images (copies of ood22/ood23, now base copies), box-matched to 3SeasonWeedDet10 expert labels; `backfill` b0000; `rejoin` MH-Weed16 only after the funnel records H3a | A, C (the jobs run through L17 once F2 lands; before that, through a one-off R3 approval of the exact argv) | Canary passes; b0000 reconciles; masked rows held per P8; **the first verifier precision outside the cwd12 capture domain** is reported with a Wilson bound (a verifier refit is proposed if it is below 0.99, §3.3); `status.json` published | < 1 |
| **R2** first segments | D22 → L18 → advances → L19. Segment 1 is Stage B (R0 + the Stage A survivor, truth on). | E, F2 (TRAIN and MAINT lanes), R0 READY, Q ≥ M | First gate decisions by the platform; recipe chosen by the §5.1 rule; P_1 committed; the boundary check computed at segment 2 | K = 1: 15–16; K = 2: 25–27 |
| **R3** collection on (may start in parallel with R2 once R1 passes) | Probe → **shadow mode for ≥ 3 days** (diagnose and propose only) → `data_autonomy: on` → first D-C wave: MH-Weed16 R-F (funnel fetch), PAGS8, MFWD POROL, then CottonWeedDet3 after the H6 scan; then L15 discovery for PricklySida, CutleafGroundcherry and Sicklepod | D, F2 (DATA lane) | Every candidate has a `decisions.jsonl` row; recall against the D-C list reported; ≥ 1 intake batch admitted; per-source yield recorded; `floor_gb`/`floor_su` set from the measured first wave; no Roboflow sync; no zero-yield outcome without its reasons | 5–20 |
| **R4** continuous | All three lanes, with envelope autonomy for L18 and L20 | R0–R3 | 14 days unattended with ≥ 1 segment decided, ≥ 1 source collected and admitted, and every pause explained by a diagnosis; ledgers verify; **the first milestone** | 45–100 per month; a milestone 8–9.5 |
| **R5** conditional, on evidence | Driver v3 (median-seed incumbent, null reuse, append without waiting for K) as stream version 2, built only if after 3 segments one of these holds: the median wait for K increments is > 7 days, the SU window binds, or D30 fires while null ≥ mean(base seeds). Verifier refit (X11). L22 resize. | E, the owner | Each is a new stream version that adopts P_n and the quarantine | not estimated |

**Supply before collection.** The eligible target images are:
- at most 206 whole-admitted pool images (206 verified target boxes);
- plus the 457 masked b0000 rows, held until F9 (P8);
- plus the MH-Weed16 rejoin, which is unknown until H3a and a re-judge.

That is **fewer than M = 763 without the MH-Weed16 rejoin or R3.** Collection is on the critical path for the first increment. The loop is designed for that: D20 fires on the first tick.

**[review] Where a person is still needed before the loop runs unattended.** Each item is one-time unless marked. Until all are done, "the platform runs it by itself" holds only from R2 on:
1. L23: approve the exact splits build and lock command (D-A).
2. Accept Protocol v3 (P1), which L25 requires.
3. Grant envelope autonomy to `weed_stream_v1` (`autonomy_granted_by: human:<email>`).
4. After ≥ 3 days of shadow mode, set `data_autonomy: on` (a governance flag the platform may not set itself).
5. Pin the EPPO table and the PAGS8 card table into `collect/domains/weed.json`, and set `floor_gb`/`floor_su` after the first wave (§6.5).
6. Rotate the Kaggle token (the key is in public git history) before any Kaggle source.
7. Recurring: R4 cards (X1, X4 residue, X10, X11, X13–X17), R3 items for sources failing a pre-check, `funnel_F9` holds past their deadline, and the allocation renewal before 2026-12-31.
8. Conditional: if /ocean has less than 3 % free (§6.6, L-7), freeing space.

**Compute to the first milestone** (est.):
- R0 27 + R1 1 + R2 27 + R3 20 + one more segment 35 + the milestone 10 ≈ **120 SU** (+4 recommended) [review: + R0b about 12–16, ≈ 132–136 SU];
- within the P5 envelope, and within the domain's 1,500 alongside the funnel's 120-GPU-h cap.

**What the owner sees first, and when:**
1. **The B_v2 baseline table:**
   - dev, ImageWeeds (same-lab) and test, as mean ± sd over 5 cold seeds;
   - next to B0 (test 0.8541 ± 0.0074) and B (test 0.8502 ± 0.0059);
   - with per-species AP, including PricklySida and CutleafGroundcherry after the dilution of D-A;
   - and the gap to 0.90.

   It comes about one day after R0 is deployed: 5 seeds of about 1.3–1.5 h each run in parallel, plus queue time. realloop_v1 spent 26.2 GPU-h in 2 h 52 min of wall clock. R0 needs only groups A, B and F1, so it does not wait for the collector or for stream mode.
2. **The Stage A recipe verdict:** the same or the next day.
3. **The first gate decision on a real increment:** about 6–10 h after the first L18. That build waits for Q ≥ M.
4. **The first milestone test number:** after 4 accepted increments, 3 segments or 30 days.

---

## 11. Open risks

1. **Supply.** Before collection there is less than one increment (§10). D-C's expected 4.4k–10.4k verified target boxes (est.) make a handful of increments. PricklySida and CutleafGroundcherry have no known source. After D-C the loop depends on discovery finding new sources, and WAIT_DATA may be its steady state.
2. **Gate power.** The expected effect of a 10 % increment is 0.002–0.005 on dev (est.), at the edge of what the chain resolves. Many HOLDs → D8 → L22 doubles M, which halves the number of increments the supply makes.
   - [review] M is fixed while the pool grows, so each accepted increment makes the next one relatively smaller. At |P| = 15K a 763-image increment is 5 %, which §5.3 already places below resolution. With at most 2 doublings (L22), the gate's power runs out once the pool is several times base_v2, and then X17 needs a person. The stream report shows M/|P_s| per segment, so this is visible before it binds.
3. **The species guard on rare dev classes.** PricklySida (42 dev boxes) failed in 5 of 6 realloop_v1 steps at the 0.03 floor. D-A dilutes it further. The gate is pinned (D-D), so a persistent D33 becomes card X17, not a code change.
   - [review] This is the most likely reason the loop accepts nothing. The null arm's per-species sd is small (0.0061 for PricklySida), because all three nulls start from the same weights, so the threshold sits at the 0.03 floor, about one box of 42. The cand arm's own spread is not in the rule. Stage C (§5.1) measures it on a known-good increment before collection, and the D33 hold (§6.4) stops segments from being spent against it. The gate's thresholds are GateConfig fields that an exp.json gate block can set without editing code (`driver.validate_gate`). Changing them is still a new protocol version and an R4 owner decision, not something the platform does.
4. **Detector capacity.** YOLO11n is a fixed convention. Dev plateaus near 0.81 (0.8082 → 0.8133 for +29 % data), and test goes from 0.854 to the 0.90 target. Data alone may not close the gap. A detector change is a new protocol version (R4). The milestone reports state the gap every time, so this is visible early.
   - [review] **The recorded numbers point to localisation, not class labels, as the larger gap.** The class-agnostic test mAP50-95 is 0.8741 ± 0.0019 for B0 and 0.8730 ± 0.0024 for B (`b0_v1/report.json`, `base_b_v1/report.json`), against 0.8541 and 0.8502 for the 12-class score. Adding 878 verified images moved the agnostic score by −0.001. The agnostic score is not a strict ceiling for the 12-class one (on ood23 the 12-class score is higher). In-domain, though, a perfect classifier on B0's boxes would land near 0.874, so reaching 0.90 needs better box quality at high IoU. Cross-source increments whose box conventions differ from cwd12's (PAGS8's growth-stage boxes, MFWD's tray crops, partially labelled images) are more likely to lower that than raise it. Waiting for milestone evidence costs months of supply. The owner may want a capacity arm (for example YOLO11s or a larger imgsz on base_v2, cold, 3 seeds, 2–4× the B_v2 cost) as a separate pre-registered protocol version from R0 on, so that data and capacity effects are measured side by side.
   - [review] Milestone reports carry the agnostic test score beside the 12-class one, so the owner can see which of the two gaps is closing.
5. **Out-of-domain verifier precision.** It is unmeasured until R1. Masks and admissions rest on it. A low reading triggers the refit (a versioned event), which also stops 3SeasonWeedDet10 copies from being independent evidence for the refit verifier.
6. **Mask artefacts.** Mean-colour patches may teach "grey rectangle = background". The gate is the safety net (D-B). `masked_area_frac` is recorded, so an effect can be looked for by admission kind.
7. **Same-lab leakage.**
   - CottonWeedDet3 is held (P9).
   - The NDSU sources and B's imageweeds_aerial images make ImageWeeds same-lab.
   - B's cwp10 and vanpe images come from cwd12 re-exports. The funnel's H6(b) is pending, and a hit voids base v2: a new base and a new stream version.
8. **The lucky incumbent seed** (§5.2) until driver v3.
9. **Provider access is unverified:**
   - compute-node reachability (the probe settles it);
   - Kaggle token rotation (the key is in public git history);
   - a scripted PAGS8 download;
   - MFWD FTP.

   RM-shared is refused, so CPU-only work costs GPU-shared hours.
10. **Budget accounting gaps.** Until group F lands:
    - the domain envelope is not summed across campaigns;
    - job estimates are never settled;
    - the RM rate is missing.

    The rollout keeps the stream inside the domain envelope to limit the exposure.
11. **Interaction with the funnel audit.** The two campaigns share inputs (§3.9) and GPU time. The funnel's `recover.mask` overlap defect needs a funnel amendment before F9. The two lineages' conclusions are reported separately and never pooled.
12. **Walltime and memory** as the pool grows: disk reading beyond about 20K images; D26 near 33–38K.
13. **The class mix of base v2.** 50 % of its boxes are OtherPlant. Its effect on the 12 species is unknown until B_v2; B0 ∪ tsw and B_v2 separate it from the 878.
14. **Label conventions.** CottonWeedDet3 and PAGS8 label only their target species, so unlabelled plants become background. `exhaustive_labels` is recorded per source, and attribution can split by it.
15. **Two package copies on the cluster** (nested and outer). Every loop job checks the module hashes, and no loop script rewrites the checkout.

---

## Appendix: design alternatives considered and the choice made

| Question | Alternatives | Chosen | Reason |
|---|---|---|---|
| Where the v2 code lives | new modules inside `inc/`, or a new package | `tools/inc2/` | The autopilot addresses it through `protocol_package`; pinned as a unit; `inc/` untouched. |
| Streaming chain | epochs appended to a live driver (driver v3), or segments on the pinned driver | segments now; driver v3 only on the R5 trigger | No new decision code before the loop has run; the segment boundary gives a free cold consolidation. |
| Truth arm per step | off (it cannot resolve the expected effect), or on | on (P3); off only by a person when the window cannot fit it | D-D says "where affordable", and supply, not SU, binds. It is evidence for attribution, not the decision. |
| When test is read | every segment, or milestones | milestones only | Fewer reads; the headline is a cold 5-seed number comparable with B0. |
| K | 6 or 4 | ≤ 4 | Supply-limited; earlier boundary checks. |
| M | proportional per segment, or fixed | 763 fixed; changed only by L22 (new stream version) | D-D's fixed size taken literally; the gate evidence stays comparable. |
| Collector → Step 1 handoff | re-run `verify pool`, or a delta module | `inc2.step1_stream`, with `intake_v1` registry entries | Re-running the pool voids the v1 verifier and base B's inputs; `intake_v1` keeps pinned v1 and `mega_trainer` out without editing them. |
| Where the guards run | collector or Step 1 | both, one implementation (`inc2.guard`) | Defence in depth at negligible cost. |
| Two proposed meanings of D20/D21 and "collect exhausted after 2 zero rounds" | — | one numbering, D20–D33; the zero-yield rule became a DATA-lane stop-loss at 3 consecutive sources | One zero-yield source is legitimate (a licence hold, a copy source). |
| D-C list | a seed list, or a recall audit | both, in config: `known_items` with the owner's decision stamp; discovery must find them; each stays fetchable after a recorded miss | It respects the owner's decision and the rule against curated seed lists in code. |
| Partition for CPU work | RM-shared, or GPU-shared | GPU-shared, one V100; lab-side fetch where possible | The allocation refuses RM-shared. |
| Step 1 job split | separate CPU and GPU jobs, or one job | one GPU-shared job per batch | Same partition either way; fewer queue waits. |
| What an increment holds | all evidenced images, or target images only | target images only (P4) | realloop_v1's OtherPlant-heavy evidence. |
| Budget | 2,400 SU (one segment per 2 days), or a supply-sized envelope | 600 SU to 2026-12-31, monthly 250, daily 80 | Supply cannot feed 2,400; it stays inside the domain's 1,500 without X14. |
| CottonWeedDet3 | fetch now, or hold | hold until the H6 scan | Same lab and seasons as dev and test. A copy would inflate the success measure. |

---

## Build notes

### Build note (group A)

**What was built.** `inc2/__init__.py` (empty), `inc2/common.py`, `inc2/guard.py`, `inc2/splits.py`, `run_inc2_splits.sh`; tests `test_inc2_common.py`, `test_inc2_guard.py`, `test_inc2_splits.py`.

**Choices where the contract is silent or ambiguous, and why.**
1. **The embedding copy scan runs inside `build`; the sequence is build → lock.** §4.2 step 5 [review] requires the scan before `lock` but names no command, and the platform's L23 submits only `inc2.splits build` and `lock` (`stream_remote.BUILD_VERBS`, `executor.py` `inc_splits_build`). So `build` runs the scan itself whenever no `embed_scan.json` covers the current candidates, and then applies it. The scan covers every candidate a build could keep besides train_core: all v1 ood22/ood23 rows under their tsw keys and all 878 base-B rows. "Covers" means the same (set, key, sha256) rows, the same v1 manifests, the same `base_selected.jsonl` and the same funnel domain config. The scan reads only v1, `base_selected.jsonl` and the funnel config, so its result does not depend on a build and never forgets a row an earlier build dropped. `build --skip-scan` applies only an existing scan, and the standalone `scan` verb stays for a person. `lock` refuses unless the scan covers every tsw and harvested base_v2 row by key and sha256 with no flag. A build therefore needs the GPU (DINOv2): `run_inc2_splits.sh` refuses `build` without one unless `--skip-scan`, and `run_inc2_build.sh` (group E) requests a V100. The embedder is loaded with `HF_HUB_OFFLINE=1` unless the caller set it, because compute nodes have no internet.
2. **An embedding hit on a tsw row drops the row** (`near_eval_embed`, or `unhashable_embed` when the image cannot be described), recorded, instead of refusing the build. Taken literally ("handled by the H6(b) rule"), a same-session frame of expert-labelled data above the calibrated threshold would stop R0 for a person; dropping keeps the copy out of training, as step 4 does for dHash hits on tsw rows. A hit, or an image that cannot be compared, in the part of base B that base_v2 keeps still refuses the build (H6(b), R4).
3. **The L-5 part is still checked.** The 812 cwp10/vanpe images get the 8-variant dHash check and the embedding scan. A copy among them is recorded as an H6(b) incident for B's and realloop_v1's results (`summary.json` `base_b.h6b`, LOCK v2 `h6.base_b.incident_in_l5_part`) without refusing the v2 build, because those images are in no v2 manifest; an image that cannot be read is recorded as unchecked, not as a copy. They are listed in `splits/v2/l5_excluded.jsonl` (sha256 in LOCK v2). They are not in `base_copies_dhash.json`, which §4.2 step 7 defines as base v2 only, so GuardV2 does not refuse them: `step1_stream`'s b0000 should treat keys in `l5_excluded.jsonl` as excluded, which is what "outright" requires.
4. **`base_copy`, the within-base de-duplication and the disjointness check use the 8 variants too.** A flipped copy of a base photograph is the same photograph.
5. **Provenance is a separate file, `base_v2_provenance.jsonl`** (per row: part, source, session, dHash, the 8 variants, licence and its basis, research_only, lab group, origin key; sha256 in LOCK v2). `common.write_manifest` keeps only `MANIFEST_KEYS`, and the pinned readers expect manifests in that form.
6. **Licences**, per source, in order: the funnel card index; the funnel config's `sources.licences`; the Zenodo record 14861516 saved at `downloads/3seasonweeddet10/zenodo_14861516.json` (`--tsw-record`); an owner table (`--licences`, each entry with its evidence). Unresolved gives `licence: unresolved`, `research_only: true` (the base exemption). No local file records the licence of cwd12 train_core or of the 3SeasonWeedDet10 record [to verify on the lab or cluster]. Without that evidence those rows are research_only, and so is every model trained on base_v2.
7. **`--testing`** lifts the real-data pins: the v1 LOCK prefixes (a5c904cd, 31ba7650, 357936e3, 242ef6b9, e5a6aff7, 8be59afb), the scorer 18c00837, base_selected 7e47d374, and the counts 617, 1,977, 3,208, 3,049, 1,915, 1,784, 878, 5,802 and 812. `run_inc2_splits.sh` refuses it. `test_inc2_splits.py` checks the pins against the local artifacts.
8. **Calibration location.** When the funnel's `leak_v1.json` is absent or failed, the stream's own calibration (seed prefix `stream/v1/leak`, the funnel prereg's H6 gates) is written under `splits/v2/leak/` (group A's directory), not `intake/leak/` (group D's). It does not depend on the evaluation splits, so `step1_stream` can reuse it by path through `guard.load_calibration`.
9. **`inc2.common`.** `read_lock` and `verify_manifest_against_lock` refuse as ambiguous (use `read_lock_v2`/`read_lock_v1` and `verify_manifest_against_lock_v2`). `manifest_path` dispatches: dev, test and imageweeds go to the v1 files, and train_core, tsw22, tsw23 and base_v2 go to v2. `NeverTrainGuard.load()` requires a path.
10. **train_core against the 8 variants: dropped under L-8, refused beyond its cap.** v1 compared the stored dHash only, so a flipped or rotated evaluation copy may sit in train_core, and `inc2.train.guard_rows` would refuse every run that holds it (the canary and B_v2 included). As first built, a hit refused the build (R4), because base v2 held train_core as a byte copy of v1. The first real build (job 47257533) refused on one image, and decision L-8 (§2.6) replaced the rule; see "Amendment: decision L-8" below.
11. **The funnel's `leak_v1.json` is applied when it is complete** (status complete, calibration passed). Its `scans.base_B` copies are applied by key or image path; when the listing is shorter than the count, the rest are read from its pairs file, checked by sha256, and a missing or changed file refuses. A listed image in the part base_v2 keeps refuses `build` and `lock` (H6(b), R4). One in the L-5 part is recorded as the incident. An absent or incomplete run is recorded as `pending`.
12. **`lock` refuses a `--testing` build** unless given `lock --testing`, which `run_inc2_splits.sh` refuses. The platform's lock takes no flags, so it can never seal a synthetic world.
13. **`lock` re-runs the never-train check** from the provenance file's recorded dHash and 8 variants of every base_v2 row. It also writes `provenance: {file, sha256, rows}` (read by `inc2.stream`) and `h6_status` (read by `stream_remote`'s lock status) beside the flat keys.
14. **`GuardV2.load` checks what it covers.** The LOCK must list exactly dev, test and imageweeds. The never-train index must hold exactly the (split, key) rows of those manifests beside the LOCK, each manifest checked by sha256. An index that silently misses a split or an image would clear copies of it.
15. **`load_calibration` does not trust `ok: true`.** It also requires:
    - a threshold that is a cosine in [-1, 1];
    - a dHash radius of 6;
    - gates at least as strict as H6's (recall 0.95 per family, false positives 0.01);
    - every `funnel.leak.FAMILIES` family with positives at its recall gate;
    - both negative sets non-empty and within the false-positive gate.
16. **`inc2.common.NeverTrainGuard.check`** (and `nevertrain_v2()`) cover the dHash and the 8 flips and rotations, which is the v2 never-train definition (§8). The inherited v1 check compared the stored dHash only.
17. **Licences.** A restriction word wins over a permissive name in the same text: non-commercial in any spelling, research-only, academic, educational, non-profit and personal use. For example, "CC BY Non Commercial" is research_only.
18. **The writer lock** is taken over when its holder's process on this host is gone, or when its Slurm job has ended (`squeue`: "Invalid job id" or a finished state). A build killed at its time limit on another node runs no release, so without this rule a person would have to remove the lock. When `squeue` cannot tell, the lock still refuses.
19. **The job script's drift check** also covers `funnel/domain.py`, `funnel/ledger.py` and `funnel/domains/weed.json`. The build and scan read the outer copy of that config for licences, lab groups, the embedder, the augmentation families and the negative groups.

**Numbers other groups depend on.** With L-5 and L-8, base_v2 is about 3,048 (3,049 minus the one L-8 drop) + 1,915 + 1,784 − 1 (the ood23–ood22 pair) + 66 ≈ 6,812 images, minus any dev-session or copy drops [to verify on cluster]. ⌈0.10 × 6,812⌉ = 682, not the 763 of P2, which was computed before L-5. The cost rows of §5.6 use N = 7,626. M is group E's; the number to read is `summary.json` `base_v2.images`.

**Not in group A's files.** L-3's per-species tolerance gate (§2.6) is to be "a new gate version in `inc2/`", but §9 assigns it to no group.

**How it was verified.** Locally, on synthetic worlds, with no network and no GPU:
- `test_inc2_common.py`: 43 checks at first build (49 with L-8), including a flipped copy that the v1 check passes and the v2 `NeverTrainGuard` refuses;
- `test_inc2_guard.py`: 68 checks, including `GuardV2.load` on a LOCK or index that leaves out an evaluation split or image, and six calibration records that say `ok: true` but fail their own gates;
- `test_inc2_splits.py`: 118 checks at first build (138 with L-8). Among them:
  - the platform's sequence: a build with no `embed_scan.json` scans by itself, then lock;
  - no second scan when one covers the candidates;
  - a planted train_core rotation of a test image refused the build (replaced by the L-8 checks below);
  - funnel `leak_v1.json` copies, listed or in its pairs file, refuse the build and lock (a missing pairs file refuses too);
  - lock refuses a `--testing` build without `--testing`, and a base v2 row left unscanned;
  - the writer lock is taken over from an ended Slurm job and refused for a running or unknown one;
  - the job script refuses `build` without a GPU unless `--skip-scan`;
  - the licence restriction words.
- **Mutation check.** Twenty single-line mutations of `guard.py`, `common.py`, `splits.py` and `run_inc2_splits.sh`, one per guard or refusal above, each make the check aimed at them fail. The files were restored and re-hashed afterwards.

The existing `test_inc_splits`, `test_funnel_leak`, `test_funnel_embed`, `test_funnel_domain_free`, `test_cwd12_species` and `test_inc_driver` pass unchanged, and so do `test_inc2_train`, `test_inc2_stream`, `test_inc2_baseline` and `test_collect_world`. Nothing has run on the real data: the real build (with its scan) and lock are the L23/R0b jobs.

**Open items for other groups.**
- `run_inc2_build.sh` (group E) runs `inc2.splits build`, which now scans. Its drift list lacks `funnel/embed.py`, `funnel/__init__.py`, `funnel/domain.py` and `funnel/domains/weed.json`. It exports no `HF_HUB_OFFLINE`; `inc2.splits` sets it by default.
- `inc2.stream._base_hashes` (group E) accepts the provenance file unchecked when the LOCK has no `provenance.sha256`. LOCK v2 now carries that key.
- `step1_stream` (group C) now validates calibrations through `guard.calibration_problems`. Its test fixture `write_calibration` still writes a bare `ok: true` record.

#### Amendment (2026-09-29): decision L-8

**What happened.** The first real `run_inc2_build.sh inc2.splits build` (cluster job 47257533) refused under item 10's first rule. One train_core image, `train_core__20210909_NIKOND3300_YL_91`, is 5 bits from `test__20210910_NIKOND3300_YL_160` under the transverse variant. Nothing was written. Decision L-8 (§2.6) drops such rows instead.

**What changed, and why.**
1. **`inc2/splits.py` build.** It computes the variant hits on train_core as before (§4.2 step 5a).
   - **Hits within the cap are dropped.** A `near_eval_variant` hit is dropped from base_v2 and from `train_core.jsonl`. It is listed in `splits/v2/train_core_variant_drops.jsonl`: the v1 row, its dHash and variants, the match and `decided_by`. It is recorded in `summary.json` `train_core_variant_drops` as an incident (count, cap, keys, matches, sha256, the v1-results note). The build goes on.
   - **The cap.** It is max(1, ⌊0.5 % × |train_core|⌋): 15 of 3,049, and 1 in a 40-image test world. A single image is always a stray; more is a pattern for a person. Past the cap the build refuses (R4).
   - **A stored-dHash hit still refuses.** v1 checked exactly that, so v1 and v2 would disagree, and L-8 covers flips and rotations only.
   - **Nothing is written before these checks pass**, as before.
2. **Where the drop is applied.** `train_core.jsonl` is v1's bytes minus the listed rows, matched by image sha256; every other line is identical and in order. `train_core.jsonl` itself drops them, rather than keeping v1's bytes beside a separate training view. L-8 removes the rows from every v2 training manifest, and a byte copy holding them would be a manifest every run refuses. `inc2.common.BYTE_COPIES` is now (dev, test, imageweeds), and `FILTERED_COPIES` is (train_core). `filter_manifest_bytes` is the one derivation rule; `read_variant_drops` is the list reader, checked against LOCK v2.
3. **LOCK v2** records `train_core_variant_drops_sha256`, a `train_core_variant_drops` block and `derived_from.train_core` (v1 sha256, v2 sha256, the list, the rule). `identical` lists train_core only when nothing is dropped.
   - `lock` re-checks the list (§4.2 step 9): the recorded sha256, the cap, v1 rows with the same bytes, re-computed hashes, GuardV2's `near_eval_variant`, and v2 = v1 minus the list. So a forged list naming a clean row is refused even when `summary.json` was edited to match.
   - `verify` re-hashes the list and re-derives train_core.
4. **Guards.** `inc2.train.guard_rows` refuses the listed bytes under any key (`train_core_variant_drop`), after checking the list against LOCK v2, as it does for L-5. A production LOCK without the record is refused. GuardV2 and the v2 `NeverTrainGuard` needed no change: they refuse the same images by their pixels, since that is what lists them.
5. **The de-duplication index keeps every v1 train_core image**, the dropped ones included. A tsw or base B near-copy of a dropped image is dropped (`near_train_core`), never kept in its place.
6. **v1 results.** B0's, B's and realloop_v1's training sets held that image. The effect is negligible (1 of 3,049), and they are not re-run (L-8); the summary and the canary's exp.json say so.

**Group B's modules, changed for L-8** (their build note has the details): the canary trains b0_v1's base minus the list and records the dropped rows (`inc2/baseline.py`); `inc2.train` refuses the listed bytes; `inc2/pilot4.py` removes listed bytes from pilot_v3's bin copies. pilot_v3's Bswap bin was built from session `20210909_NIKOND3300_YL`, the dropped image's session, so without this Stage A would refuse [to verify on cluster: whether Bswap holds that image].

**Not changed.** `collect/intake.py` (group D) refuses re-uploads of the L-5 images by its own list. It does not read the L-8 list: an intake copy of a dropped image is refused by GuardV2 when it is within 6 bits of the evaluation image under a variant, as the dropped image itself is. A near-copy of the dropped image that is more than 6 bits from the evaluation image under every variant is not refused by any list. That is the same rule as for any image, and 1 image is at stake.

**How it was verified.** Locally, on synthetic worlds:
- `test_inc2_splits.py` plants a real transverse copy of a test image in the world's train_core; v1 keeps it (more than 6 bits under its stored dHash, 0 under transverse). The build:
  - drops it, lists it with its match, records the incident, and leaves it out of `train_core.jsonl` (v1's bytes minus that line), base_v2, the base-copy index and the provenance;
  - locks, with LOCK v2 recording the list and the derivation;
  - after lock, GuardV2, the v2 `NeverTrainGuard` and `inc2.train.guard_rows` refuse the image re-listed under another key and source.
  - Two variant hits (over the cap of 1) refuse, and so does a stored-dHash hit, each with nothing written.
  - lock refuses a list changed after the build, and a forged list naming a clean row with a matching summary.
  - verify and `inc2.train` catch a list changed after lock.
- `test_inc2_common.py` covers the filter and the reader. `test_inc2_train.py`, `test_inc2_baseline.py` and `test_inc2_pilot4.py` cover group B's side.
- **Mutation check.** 18 single mutations were run in a scratch copy of the package, and each made its suite fail:
  - in `splits.py`: the cap, the stored-dHash refusal, keeping the drop in the base, and lock's list-hash, guard-reason and derivation checks; verify's list hash; the LOCK key;
  - in `train.py`: the byte refusal, the production record and the list hash;
  - in `common.py`: the filter;
  - in `baseline.py`: the canary's build derivation, its verdict's LOCK, reference and file checks, and its exp.json record;
  - in `pilot4.py`: the bin filter.


#### Amendment (2026-09-29): decision L-9

**What happened.** The second real `inc2.splits build` (cluster job 47259471, after L-8) ran the embedding scan with the funnel's passed `leak_v1.json` (DINOv2 CLS cosine ≥ 0.8256). It refused as an R4 incident on 7 of base B's kept images. 6 matched ood23 images through the funnel's list (cos 0.83–0.86, 20–27 dHash bits), and ood23 is an exam in v1 but training in v2. 1 matched ImageWeeds at cos 0.828 and 27 bits. The scan also flagged 805 of tsw22's 1,915 rows and 42 of tsw23's 1,784. Every high tsw22 pair is a 2022 capture against a 2021 capture of another session, and tsw22 and tsw23 share no capture session with dev, test or train_core. Nothing was written. Decision L-9 (§2.6) replaces the rules that judged these rows; step 5b of §4.2 states them.

**What changed.**
1. **`inc2/embed_calibration.py` (new).** The v2 calibration: per-image negatives scored as the scan scores an image, four tiers, the threshold rule, the strict threshold, recall and known limits, the source rule, the reader (`load`, `locked`, `state`) and `record_problems`. `calibrate` reads descriptors from the files its base records (sha256 and rows digest checked) and computes only what is missing, into `splits/v2/leak/v2cal_*.npz`.
2. **`inc2/splits.py`.**
   - The scan verb and `build` (scan_mode auto) write `embed_calibration_v2.json` after the scan. `build --skip-scan` applies only a current one; without it every hit applies at the scan's own threshold and lock refuses.
   - A scan whose calibration file or evaluation descriptors no longer hash as it recorded is stale, and `build` scans again.
   - The tsw and base B loops apply the copy rules and record each candidate in `copy_rules.jsonl`; the provenance file carries `copy_rule`.
   - `funnel_base_b_matches` keeps every match of an image (the old reader kept only the first) and reads the pairs file whenever the listing may be truncated. `funnel_verdict` applies L-9(b).
   - `summary.json` gains `l9` (rules, what the scan flagged at its threshold, what was applied or exempt, the funnel's verdicts) and `embed_v2` (the calibration applied, per-source accounting). `funnel_leak_v1` records its matches per evaluation split.
   - `lock` re-derives every row's rule, refuses another v2 calibration than the build's, a changed `copy_rules.jsonl`, a kept tsw row of a dev or test session and a v2 funnel copy. LOCK v2 records `embed_calibration_v2_sha256`, `embed_calibration_v2_negatives_sha256`, `copy_rules_sha256`, an `embed_calibration_v2` block, an `l9` block and the v2 threshold in `h6` and `h6_status`. `verify` re-hashes the three files and loads the calibration against LOCK v2.
3. **`inc2/guard.py`.** `calibration_problems` and `load_calibration` read a v2 record by its own rules. A per-pair record is judged as before.
4. **`collect/intake.py` (group D's file).** `load_copy_scan` binds the v2 calibration LOCK v2 records. Every row held `h6_scan` carries `copy_scan_calibration` (file, sha256, threshold), and `guard.json` and `summary.json` record it. A production LOCK without one, or a file that does not hash or load, refuses the intake (fail closed); a testing LOCK without one is recorded.
5. **`inc2/step1_stream.py` (group C's file).** `load_scanner` uses the v2 calibration when it loads and hashes as LOCK v2 records. It takes its base's role (funnel or own) and the role of any other calibration present, so P9's release rule is unchanged but no row is judged at the per-pair threshold. A v2 file that does not load is recorded as rejected and the others apply as before. This is the scan that releases or refuses intake rows, so without it the recalibration would not reach them.
6. **Job scripts.** `run_inc2_splits.sh`, `run_inc2_build.sh` and `run_inc_collect.sh` list `inc2/embed_calibration.py` in their drift checks.

**Choices where the decision is silent, and why.**
1. **Negatives per image, not per pair.** The funnel's hard negatives were pairs (rate 0.006, bound 0.0105 over 2,000 pairs). The scan flags an image on its best cosine over about 5,800 evaluation images, and the funnel quarantines a source on any hit, so a per-pair rate says little about either. The funnel flagged 39 of its 47 sources, among them a video-game source.
2. **The hard tier is every train_core image with a capture session, each scored against the evaluation images of other sessions and other dates.** L-9(c) names the nearest train_core image per dev/test image. The per-image form scores each train_core image by its best evaluation match outside its own session and date, which is the same pairing seen from the other side, measured the way the scan flags. cwd12's split is random, so few train_core sessions have no test frame. A tier of only those would be small; it is reported as `hard_session_disjoint` and constrains the threshold when it holds at least 100 images.
3. **Every tier with at least 100 images constrains the threshold.** A tier smaller than 100 cannot resolve a 1 % rate. The hard tier must reach 100, else the calibration fails. A `--testing` world may use 5, which the record says and which lock refuses outside `lock --testing`.
4. **Easy negatives are the two sources the decision names** (the video-game and tomato-leaf sources), and each must be listed as a non-plant or leaf-disease source in the funnel config's `sources.not_recoverable`. The other two sources on that list are not used. The UAV source's content is not described anywhere locally, and aerial field imagery would not be "clearly unrelated" to ImageWeeds.
5. **The base's positives must give back its threshold.** They are rebuilt from the seeds and descriptor files the base records. A changed or missing file is recomputed, and if the result does not reproduce the recorded threshold the calibration refuses. A base that records no seeds gets positives drawn with this stream's seed prefix, recorded.
6. **dHash copies are not negatives.** A negative within 6 bits of an evaluation image under a variant is a copy by the dHash rule (the L-8 image is one), so it is left out and counted.
7. **Funnel matches are re-judged at the v2 threshold (L-9(b) read with (c) and (d)).** The funnel's list is its detector's verdict at 0.8256 on the same descriptors. Read unfiltered, the incident's ImageWeeds match at 0.828 would refuse through the list even when the v2 scan clears the same pair. A v2-split match within 6 bits, or at or above the v2 threshold, still refuses.
8. **tsw rows without a capture session are judged like base B.** The exemption rests on the file names carrying date and camera; a row whose name does not has no capture provenance. On the cluster every tsw row has one.
9. **An image the scan cannot describe is dropped even under `tsw_provenance`** (fail closed), as before.
10. **The per-source accounting is informational.** A flagged source is listed with its expected false hits and p-value; removing a whole source from base v2 stays the owner's decision (R4, §4.2 step 5).
11. **`intake` binds the calibration and `step1_stream` applies it.** Intake does no embedding (the GPU job is step1_stream's), so "intake uses it" means that every held row names the calibration that will judge it, and that an intake refuses without one in production.

**Cluster files the calibration reads, and cost.**
- `INC_DIR/funnel/leak_v1.json` (the base; its `descriptor_files` and seeds), and the funnel's `emb_dinov2_images_reference.npz`, `emb_dinov2_images_pos_{flip,rot90,crop,brightness,blur,shear,letterbox640,jpeg}.npz` and `emb_dinov2_images_pool.npz` (read only, checked by sha256 and rows digest).
- `INC_DIR/splits/v2/leak/eval_desc_v2.npz` and `splits/v2/embed_scan.json` (the scan's), the v1 manifests of train_core, dev, test and imageweeds and the v1 `summary.json` (dev sessions), `INC_DIR/step1/pool.jsonl` (sources of the pool tiers), `INC_DIR/funnel/prereg_v1.json` (the H6 gates) and `funnel/domains/weed.json`.
- The existing scan stays current while its inputs are unchanged, the funnel's `leak_v1.json` and the evaluation descriptors included [to verify on cluster], so the next build does not scan again; if either changed, it scans again (a GPU job of the size of the first scan). With the funnel's files reused, the calibration is CPU work in the build job: about 3,049 × 5,802 cosines and 8-variant dHash distances for the hard tier, the same for the pool tiers (a few thousand MH-Weed16, video-game and tomato-leaf images), and reading the funnel's pool file of about 96,000 descriptors twice. That is about 1–3 minutes (est., not measured). If the funnel's files are missing or changed, DINOv2 describes about 3,049 reference images, 16,000 augmented positives and the pool tiers on the V100, roughly 10–20 minutes (est.).

**What the next real build shows.** The expected outcome is not known before the job runs. If the v2 threshold ends above 0.828, the ImageWeeds match no longer refuses, the six ood23 matches are recorded as `not_v2_split`, and every tsw row with a capture session outside dev and test is kept. If it ends at or below 0.828, that match still refuses as an R4 incident. Either way `summary.json` `embed_v2` records the threshold, the per-tier rates at 0.8256 and at the new threshold, the known limits and the per-source accounting.

**How it was verified.** Locally, on synthetic worlds, with no network and no GPU:
- `test_inc2_embed_calibration.py` (new, 51 checks), a descriptor-level world with every file in `funnel.embed`'s format. 200 train_core images include 20 frames of test scenes (cosine 0.86–0.90), 5 burst frames sharing a test image's session and date (0.97), and 1 dHash copy. The pool holds 150 MH-Weed16 and 120 non-plant images, and the base is a funnel `leak_v1.json` whose positives give 0.788. Checks:
  - the threshold rises to 0.8956, above 19 of the 20 scene frames (1 allowed at 1 % of 199), with the base's per-image rate recorded (20 false hits, 10 %);
  - the burst frames do not set it, and the dHash copy is counted out;
  - `funnel.leak.detect` at the new threshold catches a 0.975 copy of a test image and not a 0.875 scene, while at the base's both are caught;
  - crop's recall falls to 0.70 and is recorded as a known limit, and the record still loads;
  - a rerun is a no-op without the embedder;
  - a changed positives file is recomputed into the stream's directory and refuses, because it does not give back the base's threshold;
  - twelve record mutations are refused by `record_problems` and by `inc2.guard`;
  - `locked`, `state` and `step1_stream.load_scanner` bind or reject the file against LOCK v2;
  - the source rule does not flag 1 hit in 100 or 60 in 6,341 at about 1 %, and does flag 12 in 200, a dHash hit or a strict hit.
- `test_inc2_splits.py`: 164 checks (138 before). The image world adds two train_core frames of a test scene, a 2022 tsw22 capture of a test scene in its own session and a harvested scene (the test's docstring lists them). Checks:
  - copy rules per candidate and in the provenance file; the test-session tsw row is dropped;
  - the scan writes the v2 calibration, which rises above the train_core scene frames;
  - on the incident's calibration (0.80, no seeds), the sheared test image still refuses at the v2 threshold;
  - without it, the harvested scene, the flagged 2022 capture and the funnel's ood23, test and ImageWeeds matches are kept, each recorded;
  - `build --skip-scan` on a stale scan reproduces the incident except the ood23 match;
  - a scan whose calibration file changed is stale, and build scans again;
  - lock refuses another calibration, a changed `copy_rules.jsonl` and a wrong provenance `copy_rule`;
  - LOCK v2 records the three files; verify catches each changed after lock.
- `test_collect_intake.py`: 63 checks (59 before). Held rows are bound to the calibration LOCK v2 records. A file that does not hash, or a production LOCK without one, refuses the intake.
- **Mutation check.** 24 single-line mutations were run in a scratch copy of the package, and each made the suite aimed at it fail (`test_inc2_splits`, `test_inc2_embed_calibration` or `test_collect_intake`):
  - `splits.py`: the test-session drop, the provenance exemption, the tsw and base B v2 hits, the funnel's ood and threshold verdicts, the scan-hit filter, the stale-scan check, lock's rule re-derivation and calibration check, verify's re-hash;
  - `embed_calibration.py`: the floor, the session and date exclusion, the dHash exclusion, the known limits, the base's reproduction, the 1 % allowance, the known-limit check, the LOCK's sha256, the source rule, the production LOCK rule;
  - `guard.py`: the v2 dispatch;
  - `step1_stream.py`: the v2 preference;
  - `intake.py`: the row binding.

  The source tree was re-hashed afterwards and unchanged.
- Every `tests/test_inc2_*.py`, `test_collect_*.py` and `test_stream_*.py` passes, `test_stream_pipeline.py` included (its build computes the calibration through the fallback path: its funnel `leak_v1.json` records no descriptor files).

**Open items.**
- **D28 (the autopilot's source-leak diagnosis, group F).** A source with 5 % of its images refused by never-train, the embedding detector included, is quarantined. At the v2 threshold the per-image false-positive rate is at most 1 % (its bound recorded), but a small source can still reach 5 % on one chance hit. `EC.source_verdict` gives the binomial rule; D28 is outside inc2 and collect and was not changed. **Closed 2026-09-29:** D28 now applies `EC.source_verdict` (§6.4 D28; `stream_thresholds.json` D28 `source_alpha` 0.001 replaces `never_train_share`). It reads the dHash and embedding refusals of each intake summary (`guard`, else `yield.rejected`) and of `step1_stream/status.json` `per_source`, and p_false from the batch's `copy_scan` record, else from the LOCK's `embed_calibration_v2` block, which the stream snapshot's `splits/v2/lock_status.json` now carries. Checks: `test_stream_ap_units.py` (one chance hit in a 15-image source does not leak, a planted dHash copy and 5 hits in 15 do, a large source within its rate does not, no rate fails closed, the per-source Step 1 counts), S14, S25 and `test_stream_pipeline.py` unchanged.
- **Known limits.** Which families fall below 0.95 at the v2 threshold was measured by the third real build (cluster job 47260765, 2026-09-29): `crop` and `letterbox640`. That build's calibration gave the v2 threshold 0.946384 (strict 0.970972) over the funnel's 0.825636; on the 3,048 hard train_core negatives it counted 30 false hits at 0.946384 and 1,952 at 0.825636. Their augmented copies are caught only above the new threshold or by dHash. The record lists them; the stream report does not yet show them.

#### Fix (2026-09-29): base v2's near-copy drop reads both directions

**What happened.** The third real `inc2.splits build` (cluster job 47260765, after L-9) passed the scan, the v2 calibration (threshold 0.946384; 30 false hits among the 3,048 hard train_core negatives, against 1,952 at the funnel's 0.825636) and every drop, then refused at step 8: `base v2 is not pairwise disjoint across parts: train_core__20210909_NIKOND3300_YL_100 (train_core) within 6 bits of tsw23__20230720_HTRC_EDMUND_MP_709 (tsw23)`. Nothing was written. The same build logged base B's L-5 part holding 11 copies of evaluation images at the v2 threshold (an H6(b) incident recorded without refusing: those images are in no v2 manifest). The failure was the MAINT lane's second in a row, and the stop-loss held the lane.

**Why.** Step 4 dropped a row near an earlier part by one direction only: the row's dHash and 8 variants against the earlier images' stored dHash. Step 8 (`disjoint_problems`, also run by `lock`) reads every row's variants against every other part's dHash, so it also compares an earlier image's variants with the row's dHash. dHash is taken after a 9x8 resize, which is not symmetric under a rotation, so d(dHash(T(x)), dHash(y)) and d(dHash(x), dHash(T⁻¹(y))) can differ. The tsw23 row was more than 6 bits from the train_core image under each of its own variants, and one of the train_core image's variants was within 6 bits of it.

**What changed.** `inc2/splits.py`: `EarlierIndex` holds the earlier parts' dHash and their 8 variants, and `_near_earlier` reads both directions. A reverse hit is recorded with the variant `earlier:<name>` (the earlier image's variant). A row step 8 would flag is now dropped at step 4 as `near_<part>`, the rule of §4.2 step 4. The drop rule and step 8 read the same pairs; `lock` is unchanged.

**How it was verified.** `tests/test_inc2_splits.py` `test_near_both_directions`: a pair that only the reverse direction finds (none of the row's variants within 6 bits of the earlier dHash; the earlier image's rot90 3 bits from the row's dHash) is a hit with variant `earlier:rot90`; `disjoint_problems` flags that pair and not the base without the row; a forward hit and a miss behave as before. The whole splits suite passes. The real build is the platform's next L23 build.

#### Live incidents of 2026-09-29, after the lock

- **rl-b could not be priced (funnel).** `levers.estimate_funnel` indexed `su_ledger.rates()` as su_rates.json's raw tree; the first live rl-b proposal raised `KeyError: 'h100'` on every tick. It now reads `rates.<family>.su_per_gpu_hour` (commit 24d24f0; `tests/test_funnel_ap_units.py`).
- **L15 discovery wrote to the cluster's path on the lab.** The ticker's environment sets no INC_DIR, and `inc.common` defaults to /ocean. Two discoveries (targeted at CutleafGroundcherry, PricklySida and Sicklepod, the species base_v2 added nothing for) failed at `intake_lock` with PermissionError, and the stop-loss held DATA. The lab collector verbs now pass `--inc-dir` (commit 2fe0e0d; replays S1b, S22). After the fix, discovery listed 385 candidates.
- **The scorer sidecar's recompute check ran in the wrong image order.**
  - `ap_per_class` sorts with `np.argsort(-conf)`, which is not stable, so predictions with equal confidence are ranked by input order. The sidecar concatenated per-image arrays in exam key order; DetMetrics concatenates them in the validator's order. The "exact" recompute was therefore 0.0007–0.0012 off.
  - Stage A (pilot_v4) failed all three units at the sidecar stage, and the platform paused the stream (D5, not transient).
  - The canary's sidecar failed the same way, so DCAN held TRAIN, although the canary reproduced b0_v1 on dev within one sd.
  - `check_capture` recomputes in the validator's order, still to 1e-9 (commit 334f76b; `tests/test_inc2_gate3.py` with tied confidences).
  - Recovery, by a person as D5 and DCAN ask: the pilot_v4 units were unblocked (driver `unblock --all`, with the reason in its ledger; the runs re-score and nothing is retrained); the canary's run was re-run on `run_inc2_job.sh` with its own submission list (it re-scores the verified weights and writes the sidecar); then `inc2.baseline canary-verdict`.
  - Baselines b_v2, b_v2_s640 and b_v2_m640 recorded failed sidecars without failing (by design), so their dev SE is not yet available.
  - Verified on the cluster after the fix:
    - the canary's re-scored sidecar recomputes the score exactly (`full_recompute_max_abs_diff` 0.0; class-restricted 0.0031);
    - `canary-verdict` passes (dev 0.8107 against 0.8082 ± 0.0063, sidecar ok, production).
  - Its dev bootstrap SE per species runs from 0.0225 (Waterhemp) to 0.0961 (Goosegrass), with CutleafGroundcherry 0.074 and PalmerAmaranth 0.073.
  - Treated as independent across species, the 12-class dev mean has a sampling SE of about 0.016, several times the seed sd (0.003–0.006).
- **Stage A ran on the v1 executor.**
  - The pinned driver takes its job script from `$INC_JOB_SCRIPT`, with v1's `run_inc_job.sh` as the default. Only the v2 job script and the v2 builders set it.
  - So every advance from the login node submitted v1: `remote.advance`, which the stream calls each tick, `remote.unblock`, and the unblock made by hand.
  - pilot_v4's X1a/X1b candidate and null runs were refused at stage recipe. v1 admits only the protocol recipe, and Stage A exists to test these two. The base re-scores ran on v1 and wrote no sidecar.
  - `remote._job_script_for` now sets the experiment's own executor before any driver call that may submit.
- **A second tie effect.**
  - pilot_v4's x1a candidate, on the v2 executor, failed at the sidecar: its class-restricted AP differed from the full call by 0.0103.
  - The cause is the same unstable sort, this time between the global call and a single class's call.
  - The restricted check and the bootstrap now run on tie-broken arrays (`tie_break`) and agree exactly. The score check keeps the captured arrays.
- **Stream init could not start.**
  - `inc2/stream.py` took `PKG` from `__name__`, so under `python -m` (every job script) its `_inc2()` imports of recipes, gate3 and the others all failed, each reported as "not installed".
  - The tests imported the module and never ran it as `__main__`.
  - LI's first run (job 47276839) failed in one second. The platform charged the failure and filed LI's second run for a person, since LI's campaign limit is 1.
  - `PKG` now comes from `__package__`, and a test runs the module as `__main__`.

#### Fix (2026-09-30): a person's licence override, from the collect config to run.json

**What happened.** The owner accepted two harvested sources whose licence is unresolved, CottonWeedDet3 (`kg_yuzhenlu__cottonweeddet3`) and the MFWD trays (`mediatum_1717366`), for research use only. The decision could not take effect:
- `prefilter.precheck` held every unresolved licence and never read `licence_overrides`. The fetch therefore wrote no `fetch.json`, intake refused `not_fetched`, and `plan` recorded `licence_ok` null.
- Nothing appended `released` to `sources.jsonl`. The autopilot keeps a collector-held source held, so it never proposed the fetch again.
- Intake read the override, but took `research_only` from the fetched licence alone (false for an unresolved class). The registry entry recorded neither.
- `step1_stream.intake_licence_state` raised `AttributeError` on an override that is neither a text nor a record, and ignored restriction words in the override's text.
- `inc2.stream release --hold licence` lifted the hold without recording a licence or research_only, so the cut wrote `licence` null and `research_only` false (fail open).
- `run.json` carried no research-only flag, although `exp.json` records one.

**What changed.**
- `collect/config.py`: `validate` checks `licence_overrides` (an object keyed by source ids, each `{id, research_only (a boolean), decided_by human:…, decided_utc, reason}`). `research_only` false is refused unless the policy reads the `id` as permissive (`licence.canonical` then `classify`), so an unresolved licence never enters unmarked. `licence_override(cfg, source_id)` is the one reading that the pre-check, `plan` and intake share.
- `collect/prefilter.py`: `precheck` does not hold an unresolved licence that an override names. A refused licence is still rejected and closed.
- `collect/plan.py` and `collect/state.py`: under an override a candidate records `licence_ok` true, the override's id as `licence_id`, and `licence_override`; its licence record keeps its class. Before the pre-checks, `plan` calls `state.release_overridden`, which appends one `released` event (the override's `decided_by`, `decided_utc` and `reason`, with `codes: ["licence_unresolved"]`) for a source held for `licence_unresolved`. A released source is a candidate, so a second call appends nothing.
- `inc_autopilot/stream.py`: `plan` runs on the lab, and its release lands in the lab's ledger, which no snapshot folds. A `licence_unresolved` hold in the snapshot's ledger on a source that the collect config overrides is therefore read as released, once. The `source_released` event names the person's decision (`by: licence_overrides`, the override's `decided_by`, `decided_utc`, licence id and `research_only`, and the collect config's sha256), not the collector. A hold for another reason (the copy scan) stays held. `_candidates` reads the override from the collect config too, as the fold does: a candidates file written before the override (licence unresolved, a `licence_unresolved` R3 hold in its pre-check) gives the override's id, `licence_ok` true and no `licence_unresolved` in `collector_precheck`, so D20 proposes L16 without waiting for the next L15. A refused licence is never lifted.
- `collect/intake.py`: `research_only` is the licence's OR the override's. The manifest rows, `summary.json` and the registry entry's provenance record `research_only` and `licence_override`. `licence` and `licence_class` stay as fetched.
- `inc2/step1_stream.py`: an override that is neither a text nor a record, or a record without a licence text (`id`), refuses (StreamError); the row's own `unresolved` is never read as the override's text. The override is research_only unless its record says false and its text names a known licence that does not restrict use: a text that restricts use (`collect.licence.restricted`) or names no known licence (`licence_id` None) keeps research_only.
- `run_inc2_stream.sh`: `tools/collect/__init__.py` and `tools/collect/licence.py`, which `step1_stream` imports for the restriction rule, are hashed into the log and refused when an outer copy exists and differs.
- `inc2/stream.py`: `release --hold licence` takes `--licence TEXT` and `--not-research-only`, and the release event records both. `--not-research-only` is refused for a text that names no known licence or restricts use, read as `step1_stream.intake_licence_state` reads an override. The cutter applies them (`_row_licence`): unless a release records both, the rows it lifts are research_only in the rows sidecar and in the increment meta. A row the queue marks research_only stays so. Bisect arms (P_c plus one suspect increment) and Stage C (drawn from P_0) record `stream.research_only` in `exp.json` as a segment does.
- `inc2/train.py`: `run.json` copies `exp.json`'s `stream.research_only.models` or `research_only.flag` as `research_only`: true when either is true. An experiment a stream built (a `stream` block) that records neither gives "unknown" (fail closed); any other experiment that records neither leaves it out.
- `collect/domains/weed.json`: the two overrides (`id` research-only, `research_only` true, decided by the owner on 2026-09-30).

**How it was verified.**
- `test_collect_config.py`: each malformed override is refused, `research_only` false included for `unknown`, `research-only`, a non-commercial licence and an unrecognised text; `research_only` false for CC BY 4.0 is valid. The config holds the two decisions.
- `test_collect_prefilter.py`: an override lets an unresolved licence through for its source id only; a refused licence still closes.
- `test_collect_plan.py`: `plan` releases a `licence_unresolved` hold once, with the override's stamp; holds without an override, or for another reason, stay; the candidate record is as above. An override that names a candidate whose licence the policy refuses leaves `licence_ok` false, records no override and keeps the `licence_refused` close.
- `test_collect_intake.py`: an unresolved source under an override is fetched and intaken. Its rows, `summary.json` and registry provenance are research_only and carry the override, with the licence still `unresolved`. A refused licence still closes, and the MFWD trays' rows are research_only.
- `test_stream_ap_units.py`: the collector's licence hold is lifted once the config records an override, and a source without one stays held. An overridden source held for the copy scan stays held. The release event names the override's stamp and the collect config's sha256. A candidates file written before the override reads as a usable licence with no review; a source without an override is still reviewed; a refused licence stays refused.
- `test_inc2_step1_stream.py`: record, text and restriction cases; an override of another type, or a record without a licence text, refuses; an override whose text names no known licence is research-only whatever its record says. The job script logs `collect/__init__.py` and `collect/licence.py`, and an outer `collect/licence.py` that differs refuses.
- `test_inc2_stream.py`: rows lifted without the licence text are research_only in the sidecar and meta, and the segment's models are research-only. The CLI's `--licence`/`--not-research-only` are recorded and applied; `--not-research-only` is refused for an unknown or restricting text. A keys-scoped licence release applies to the row it lifts. A bisect arm whose increment holds research_only rows records its models research-only; the other arm and Stage C carry the pool's flag.
- `test_inc2_train.py`: the flag is copied for a baseline, for a segment (a failed run included), "unknown" for a stream experiment that records none, and left out otherwise.
- Each new check was run against the code with its guard removed (the keys-scoped release map, the bisect and Stage C records, the autopilot's `licence_unresolved` condition and its `_candidates` override, `plan`'s unresolved-only condition) and fails there.

**Deploy.** `collect/domains/weed.json` is one of the executor's `GOVERNANCE_FILES`, and `inc_autopilot/stream.py` is in the autopilot package, so both change `executor.code_hash()`. Until the replay suite passes again on the new code, `replay_status` and `stream_replay_status` report a pass recorded for other code, and stream envelope builds and gated R2 data levers are refused. After the lab and cluster copies are synced from one commit, run `executor.run_replay_tests` (the replay, governance, stream replay and stream mutation scripts) so that a pass is recorded for the new `code_hash`. `test_stream_ap_replay.py` and `test_stream_ap_mutations.py` pass on this tree. The lab's `sources.jsonl` records the override's `released` event on the next `collect plan` (L15). The fetch does not wait for it, because it refuses only closed or quarantined sources.

**Open.** A research_only model is not yet refused by the model router or the robot uplink.

#### Live incident (2026-10-01): an approved source review of a lab-placed source went to the cluster

**What happened.** D20 files a candidate that fails a pre-check as an R3 review for a person (L16R), and L16R always rendered the cluster's fetch (`sbatch -p GPU-shared run_inc_collect.sh fetch`). The approved review of `mediatum_1717366` (approval `ap-1790700040-42144765`) was adopted into the DATA lane and ran as job 47302914. The cluster's collector refused it: `not_placed_on_cluster: provider mediatum is not placed on compute nodes (placement.json); the lab hook fetches it`. For a provider that placement.json does not place on compute nodes (the mediaTUM FTP, github), an approved review could never succeed, and each failure counted toward the DATA lane's stop-loss. The automatic path already routed by placement (D20's L16 → L16L).

**What changed.**
- A lab form of the review, L16RL (`inc_stream_collect_review_lab`, R3). It runs L16L's command (`collect fetch --source <id> --max-bytes <B> --out <lab INC_DIR>/intake/staging/`), is priced at zero, and is an `executor.LAB_ACTIONS` action that the ticker's fetch hook runs. It is outside the envelope and is not a gated R2 action, so only a person's approval runs it. `stream_levers.json`, `policy_actions.json`, `executor.ARGV_FORMS` and `StreamRun._lab_hooks` carry it.
- `_file_reviews` picks the form from the candidate's `placement`, the field D20 routes L16 by: `lab` files L16RL, anything else L16R as before. `source_reviews` records the lever.
- An approved L16RL is adopted into the DATA lane and started by `_run_lab_items` under the approval (detached, no sbatch). It folds as an L16L run: the attempt is counted and the source is `fetching` on the lab; then `fetched` and the sync (L16S), or a failed step counted against the source.
- `_ready` never submits an approved L16R whose source's candidate is placed on the lab. It also fails closed: an approved L16R whose source has no candidate row this tick is not submitted either. That happens to a discovered source that the last plan (L15) no longer lists. Its placement cannot be read, and an L16R carries no `--candidates`, so the cluster's collector could not load its record anyway. In both cases the lane writes a `refused` ledger entry with the reason and a data card naming the source and the approval, declines the proposal, counts no attempt, failure or charge, and marks the source's review `superseded`. The source is then fetched on the lab: by D20's L16 → L16L once its pre-check passes, otherwise through a new L16RL review for a person (the record keeps the superseded approval id).
- The refused approval is closed in the approvals log (`approvals.record_executed`: `started`, then `failed` with the outcome `not_submitted` and no job). Left open, it stayed "approved, not executed": the INC page offered Run now for it, and `executor.execute_approved`, which has no placement check, would have sent it to sbatch, where the cluster's collector refuses it as it refused job 47302914. Once closed, `approvals.awaiting_execution` no longer lists it, and `execute_approved` refuses it as already executed. The execution log, the budget and the source's attempts are not touched. The `refused` entry records whether the approval was closed; if it could not be, the card says not to run it from the INC page.
- A filed L16R whose approval already ran (`done` or `failed`) is marked `superseded` when D20 lists its source again for review, if the source's candidate is now placed on the lab (`_spent_cluster_review`, ledger `review_superseded`). Its review is then filed again as L16RL, once. Before, such a review stayed `filed` for ever, so the source was never offered to a person in its lab form. The live approval has already run (its execution is recorded), so it is never run again, and its review is filed in the lab form if `mediatum_1717366` fails a pre-check again. D20 lists no source that is being fetched, so a run in progress is never superseded.

**How it was verified.**
- `tests/test_stream_ap_review_placement.py` (new, 70 checks in 9 cases, in the stream world of `test_stream_ap_world`):
  - The menu, the executor, the policy table and the lab hooks agree on L16RL.
  - A lab-placed candidate failing its pre-check files L16RL, and nothing runs before the person decides. Once approved, it is launched on the lab runner and never reaches sbatch. While it runs, the executor's campaign counts it in flight as the L16 family.
  - An ok end gives fetched, then L16S. Its completeness is read from the lab process's last line: complete, or partial with a `source_partial` entry. The L16 limits count its bytes for the source: a later 48.5 GB fetch would reach 50.5 GB. A failed end gives a failed step and a source failure.
  - A person's denial of an L16RL is recorded on `source_reviews`, and the review is not filed again.
  - A cluster-placed candidate still files L16R, run by sbatch.
  - An approved L16R of a source now placed on the lab, or of a source with no candidate row, is not submitted and counts nothing. The approval is closed as `not_submitted` with no execution-log run or charge, a data card names it, and a Run now as the person (`execute_approved` with a Context that can reach the cluster) is refused with no sbatch. The source's review is superseded and, for the lab-placed source, filed again as L16RL.
  - An L16R that already ran in its cluster form and failed for a source now placed on the lab is filed again as L16RL, once.
- Mutations in a temporary copy. The first version of the test (37 checks) failed 34 of them at HEAD, and each of its checks failed under at least one of 14 targeted mutations: the lab form never or always chosen; the `_ready` guard, the lab hook, the lab action or the argv form removed; the fold's or the fetch levers' list without L16RL; no re-filing; the row at R2; a job follow; the family's action list; a job price; the cluster argv. The current test fails in all 9 cases at HEAD. Each of 9 further mutations fails it:
  - `stream_limits`' fetch actions without L16RL;
  - the lab fold's completeness for L16L only;
  - `_person_denied` for L16R only;
  - L16RL removed from `LANE_OF`;
  - the refusal's card removed;
  - the approval not closed;
  - the approval claimed but not closed;
  - the guard failing open on a missing candidate row;
  - no superseding of a spent cluster-form review.
- `test_stream_ap_units.py` exercises L16RL in the menu checks.
- `test_stream_ap_world` now removes its temporary tree when the process exits (`STREAM_AP_KEEP=1`, or `STREAM_PIPELINE_KEEP=1` as before, keeps it). Before, every suite built on it left a `stream_ap_*` directory in the system temp dir, and 315 of them (4.5 GB) were there on 2026-09-30.

**Deploy.** `policy_actions.json`, `stream_levers.json` and the autopilot modules change `executor.code_hash()`, so re-run `executor.run_replay_tests` after the lab and the cluster are synced from one commit.

#### Live incident (2026-10-01): the fetch byte limits counted requested bytes, not fetched bytes

**What happened.** `executor.stream_limits` checks the L16 family's byte limits (`gb_per_source`; `gb_per_day` and `collect_gb_daily`; `collect_gb_envelope`). It summed the requested `max_bytes` of every past fetch attempt, whatever the outcome. On 2026-10-01 it filed the next L16L of `mediatum_1717366` for a person, with three reasons:
- "source mediatum_1717366 would reach 71.5 GB (limit 50 GB unless a person approves)";
- "today's fetches would reach 121.5 GB (limit 20 GB)";
- "the campaign's fetches would reach 121.5 GB (collect_gb_envelope 60 GB)".

What was on disk:
- 5.59 GB of CottonWeedDet3 (requested 50 GB);
- 19 MB of mediatum's lab fetch, which failed `download_failed` after `gt.csv` (requested 10.7 GB);
- nothing of mediatum's cluster review, refused `not_placed_on_cluster` before any download (requested 50 GB).

With the campaign envelope read as exceeded, every later fetch would have waited for a person.

**What changed.**
- `executor.fetched_bytes` decides what each attempt counts in the byte limits.
  - **Ended attempts.** An attempt whose end the lane saw counts what the collector's source ledger on its machine records for its source. Each `fetched` event counts at the time it was fetched. A refused attempt, or one that failed before downloading, counts nothing. One that failed after counts the files it finished: `collect.fetch` records them in a `fetched` event when it catches the failure.
  - **Attempts in flight.** An attempt not seen to end reserves its requested `max_bytes` at its request time, so concurrent fetches cannot overshoot a cap.
  - **Killed attempts.** Some attempts end with finished files in staging and no event for them. `collect.fetch` writes its closing event only after its download loop, and only for the exceptions it catches. Three cases skip it: the lab runner's timeout (`LabRunner.launch`, 6 h, SIGKILL); the cluster job's walltime (`run_inc_collect.sh`, 8 h); and an exception the collector does not catch, such as `ftplib.error_temp` from an FTP 421/426 or `http.client.IncompleteRead`, neither of which is an `OSError`. On that machine's ledger, such an attempt leaves a `fetch_started` with no closing event (`fetch_failed`, `fetched`, `held` or `closed`) before the source's next `fetch_started`. Such an open start belongs to that machine's latest attempt of the source requested before it, allowing 60 s of clock skew (`FETCH_START_SKEW_S`). An ended attempt with an open start counts its `max_bytes`. It keeps counting them after a later attempt fetches the same files again, and those files then count a second time, so the error is an over-count.
  - **Fail safe.** An ended attempt whose fetched bytes cannot be determined also counts its `max_bytes`. That covers four cases:
    - the lab ledger is missing or cannot be read whole;
    - the cluster fold carries no fetched facts, from a fold written before they were kept or from a cluster ledger the snapshot could not read whole (an unparsed line, or a read stopped at `MAX_LEDGER_ROWS`). Every ended cluster attempt then counts its `max_bytes`, including those of sources absent from the fold;
    - the cluster fold is not newer than the attempt's end. The snapshot reads the ledger before squeue, so the fold that shows a job ended may predate the job's last event;
    - a snapshot ships no fold, because the ledger is missing or unreadable. The previous fold is kept, so later ends stay unknown.
  - **Reasons.** Each reason names what it counted at `max_bytes`, for example "counted at their requested max_bytes: 10.8 GB requested by 1 fetch(es) not yet seen to end".
- The daily window counts bytes by the time they were fetched. The campaign envelope counts every fetched byte of the sources the campaign fetched, on the machines it fetched them on.
- The ticker passes these facts in its campaign (`StreamRun.camp` `fetched`):
  - `fetch_ends`: proposal id → utc, written by `_done` and `_failed` for L16, L16L, L16R and L16RL. A state written before the record existed takes the ends once from the campaign ledger's `item_done` and `failed` entries.
  - `cluster_fetched`: the cluster ledger's fetched facts (`sources`: each source's `fetched` events; `open`: its open `fetch_started` times), as of the last snapshot that folded the ledger.
  - The same facts from the lab collector's own ledger (lab `INC_DIR/intake/sources.jsonl`), read when the limits are checked.
- `stream_remote.fetch_facts` gives each source's `fetched` events (`fetched_events`: [[ts, bytes]]) and open starts (`open_fetches`: [ts]). `_fold_sources` puts them in the snapshot's `intake/sources.json` only when `_jsonl` read the whole ledger. `_jsonl` now reports a read stopped at its cap, and the snapshot's notes name an incomplete read.
- **Residual.**
  - An attempt the lane never saw end keeps its reservation. An example is a run cleared as "already ran" after a tick failed between the run and its state write.
  - The lab runner's timeout is not sized from `max_bytes`. At the lab's ~460 kB/s, the 10.8 GB L16L of mediatum needs about 6.5 h, past the 6 h timeout. A run killed that way counts its 10.8 GB from then on.

**How it was verified.**
- `tests/test_stream_ap_fetch_bytes.py` (new, 40 checks in 12 cases, in the stream world of `test_stream_ap_world`):
  - **Live.** The record as the platform holds it: three ended attempts whose ends are only in the campaign ledger, CottonWeedDet3's event in the cluster's ledger, and mediatum's 19 MB in the lab's. It counts 5.59 GB + 19 MB.
    - A 10.8 GB L16L of mediatum reaches no cap.
    - A 49.99 GB request of mediatum would reach 50.0 GB.
    - Today and the campaign read 5.6 GB.
    - The next L16L runs on the lab, with nothing filed.
  - **In flight.** A running L16L reserves its 10.8 GB: 21.4 GB today with a 5 GB request. Once it ends, its 3.0 GB count instead.
  - **Unknown bytes.** These cases each count the requested `max_bytes` and say so:
    - a torn line in the lab ledger;
    - a fold without fetched events, where both cluster attempts count (105.0 GB today), since such a fold cannot say that mediatum's review fetched nothing;
    - a cluster fetch whose end the folding snapshot showed. Its 1.5 GB count from the next snapshot on;
    - a cluster ledger that becomes unreadable before the end: no snapshot ships a fold, and the previous one is kept.
  - **Killed attempts.**
    - An L16L of mediatum killed by the lab runner's timeout, with only `fetch_started` in the lab ledger, counts its 10.8 GB: 21.4 GB today with a 5 GB request. The earlier failed attempt still counts its 19 MB. A later closed attempt (5 GB requested, 1.0 GB fetched) counts its 1.0 GB, and the killed one still counts its 10.8 GB.
    - A cluster L16 that ends TIMEOUT with only `fetch_started` counts its 2.0 GB after a fold newer than its end.
  - **Incomplete cluster ledger.** A torn line, or a read stopped at the cap (patched to 1 line), gives a fold without fetched facts: both cluster attempts count (105.0 GB), and the snapshot's notes say why. A unit check of `_fold_sources` covers the open-start rule: a start followed by the source's next start is open; a start followed by `fetch_failed` or `held` is closed; a last start with nothing after it is open.
  - **Ends recorded by `_failed`.** In a state that already records ends, an L16L that ends rc 2 with a 2.0 GB partial `fetched` event counts 2.0 GB and reserves nothing (20.1 GB today with a 12.5 GB request).
  - **Both machines.** A source with a cluster attempt (4.0 GB) listed before a lab attempt (3.0 GB) counts both: 20.5 GB today with a 13.5 GB request, and 50.5 GB for the source with a 43.5 GB request.
  - **Caps on real bytes.**
    - Per source: 53.0 GB.
    - Per day, by fetch time: 18 GB that landed 2 h ago, from a request 26 h old, give 23.0 GB.
    - Over the campaign: 63.0 GB.
    - The ticker files the next fetch for a person.
- Mutation checks, in a temporary copy:
  - Every case fails at HEAD. Two cases stop at a set-up assertion, and one raises because `_jsonl` takes no `info`.
  - Before the open-start rule, each of the first 22 checks failed under at least one of 14 targeted mutations: ended attempts never covered; in-flight attempts counted as 0; unknown attempts counted as 0; a fold as new as the end taken as fresh; ledger bytes dated at the request; the lab ledger's bytes dropped; an unreadable lab ledger read as empty; no backfill from the campaign ledger; ends not recorded by `_done` and `_failed`; a fold without fetched events; no note in the reason; the per-source sum taken over every source; a source absent from the fold read as unknown; the ticker passing no facts.
  - Each of 12 further mutations fails at least one check:
    - `_failed` not recording ends;
    - dedup by source only, not by source and machine;
    - a snapshot without the fold read as an empty fold;
    - no open-attempt rule;
    - an open start given to the latest attempt whatever its request time, or to the earliest attempt;
    - an incomplete cluster read taken as whole, or truncation ignored;
    - `held` not closing a start;
    - the cluster's open starts dropped;
    - the lab's open starts dropped;
    - a fold row without facts skipped instead of making the fold's facts unknown.
- `test_stream_ap_review_placement.py`: its camp helper now sets the StreamRun's paths, which the ticker's camp reads. Its check that the L16 limits count an L16RL's run for the source still holds. That world's lab has no collector ledger, so the run counts its 2 GB request.

**Deploy.** `executor.py`, `stream.py` and `stream_remote.py` change `executor.code_hash()` and the stream modules that S23 compares. So sync the lab and the cluster from one commit, then re-run `executor.run_replay_tests`.

Until the first snapshot after the sync, the cluster's ended fetches still count their requested `max_bytes`. A fetch filed on these limits is re-checked each tick while `data_autonomy` is on, and it runs once the limits pass.

### Build note (group C)

**What was built.** `inc2/step1_stream.py` (verbs `bootstrap`, `admit --intake <batch>` / `admit --registry [--slugs]`, `backfill`, `knowntruth`, `rejoin --slug`, `serve-holds [--hold h6_scan|licence|funnel_F9]` (also accepted as `scan-holds`, the name group F's L17 form submits), `status`, `verify`), `inc2/mask.py` (`decide`, `mask_except`), `run_inc2_stream.sh`; tests `test_inc2_step1_stream.py` and `test_inc2_mask.py`.

**Interfaces other groups rely on.**
- **Queue (group E).** `step1_stream/queue/queue.jsonl` is append-only; `queue/events.jsonl` (append-only) carries hold releases and additions, refusals (`domain_dev`, `near_eval_embed`, `superseded:rejoin_vN`), near-duplicate group merges and `rejoin_needed`. The cutter reads the queue through `inc2.step1_stream.load_queue(Layout(root=…, inc_dir=…))`: events folded, `group` = the current union-find root, `evidenced` recomputed from `index/evidence.json`, and `eligible_step1(row)` (not refused, evidenced, no hold left, at least one kept verified target box). Row fields are `QUEUE_KEYS`: the §3.3 row plus the `inc.common.MANIFEST_KEYS` the cutter writes, `holds` (every hold left, in the order `h6_scan`, `licence`, `join_conflict`, `funnel_F9`; `hold_until` is the first), `hold_deadline` per hold, `unmasked_image`/`unmasked_sha256`, `dhash` (unmasked) and `dhash_masked`, `masked_area_frac`, `admitted_utc`, `input` (the intake batch, registry slug or `v1`), `supersedes`, `scanned` and `h6_reason`.
- **Status (group F).** `step1_stream/status.json`, schema in `check_status`: `pending` (`registry_new_paths`, `registry_refused`, `rejoin_needed`, `intake_batches`), `batches` (committed, in progress, by kind), `crop_ids`, `queue` (eligible target images and boxes, admissions, boxes per species, kinds), `admission` (whole, masked, refused_overlap, not_admitted), `holds`, `holds_past_deadline` (a `funnel_F9` hold past its deadline is an R3 item for a person), `refused`, `knowntruth` (per batch, with the Wilson lower bound), `refit_triggers` (card X11), `human_queue`, `one_time` (`bootstrap`, `backfill`, `knowntruth`: when each ran, or null; a `knowntruth` run that found nothing to measure counts as run), and `per_source` (per source: `images_seen`, `near_eval_embed` (copy-scan hits at admission, on masked copies and in later `serve-holds` runs), `never_train_refused`, `target_boxes_admitted`, `decision:<reason>` and `refused:<reason>` counts, for D28 and the yield).
- **CLI and job.** `sbatch -p GPU-shared run_inc2_stream.sh <verb> …`; exit 2 on a refusal. The Slurm log goes to `results/framework/inc/logs/`, which exists before any stream job (Slurm opens the log before the script runs; `step1_stream/` does not exist before the first bootstrap). The job runs the nested copy and refuses when any of the eight library modules of §3.3 has no outer copy or a different one, or when an outer copy of an inc2 module or of a module they import (`funnel/qualify.py`, `dataset_discovery.py`, `registry_lock.py` among them) differs. A `--lock` other than the pinned splits v2 LOCK is refused. One writer at a time: a non-blocking `flock` on `.lock` plus an owner file `.lock.owner` created with `O_EXCL`, because on Lustre `flock` excludes across nodes only on a mount with the `flock` option (as `inc/train.py` and `inc/driver.py` note); a mount without flock support is tolerated with a warning. An owner file is taken over when its process on the same host has exited, when `squeue` no longer lists its job, or when it is older than 9 h (longer than the 8 h walltime).

**Choices where the contract is silent or ambiguous, and why.**
1. **Holds are a list, released by events.** The queue is append-only, so a release cannot rewrite a row. `serve-holds` releases or refuses held rows and records each run in the hash-chained `ledger/holds.jsonl`: `licence` from the funnel's cards index and domain config (as `recover` reads them; a non-commercial licence makes the row `research_only`); `funnel_F9` once F9 writes `step1_r1/domain_dev.jsonl` (the domain-dev images and their 3-bit near-duplicates are refused for good, the rest released), never by the platform past its deadline; `h6_scan` by a copy scan of the unmasked and the masked image. One new hold, `join_conflict`, marks the queued twin of an exact duplicate whose labels disagree.
2. **`h6_scan` covers every source that is not provenance-cleared** when no passed calibration was usable at admission (§3.2), not only the v1 slugs with evaluation near-copies (§3.3). A copy that dHash misses would inflate the success measure, and gate acceptance is no evidence against a leak. Until a calibration exists, no registry or b0000 row is eligible. With the funnel's passed `leak_v1.json`, or the stream's own calibration (`splits/v2/leak/leak_calibration.json`, where group A writes it; then `intake/leak/leak_v1.json`), the scan runs inside the admit job. A same-lab row (P9: a v1 slug with evaluation or train_core copies, or an intake row of the evaluation lab) is released by the funnel's calibration at once, and by the stream's own only after its 21-day deadline (§6.7). The scan is row-level: a hit refuses the row, a clean scan releases it, and an image the detector cannot describe refuses that row alone (`unhashable_embed`), never the batch. Quarantining a whole source (H6(a)) is left to D28/L24, which read the per-source counts in `status.json`. A calibration file is used only when its record shows it passed (`inc2.guard.calibration_problems`: a cosine threshold in [−1, 1], the 6-bit radius, H6's gates, every family and negative set within them) and names its embedder; one that says `ok` without showing it is not used, the refusal is logged and recorded in the scanner record, and the rows stay held.
3. **Provenance clearance** comes only from the intake: the row and its source record in `sources.json` (the collector's one-source form is read) must both say so, the collector must not have held the row for the scan itself, and the lab must not be the lab of dev and test. Registry slugs are never cleared.
3a. **Licences.** A licence text from the funnel's cards or config resolves only when it names a known licence family (Creative Commons, CC0/public domain, MIT, Apache, BSD, GPL family, ODbL, ODC-BY, CDLA, Unlicense). The provenance suffix `(fetched by L11a from <url>, …)` is cut first. "unknown", "Other (specified in description)", "Private", "All rights reserved" and unrecognised texts stay held (P6). NC and ND are research-only. For intake rows the collector's verdict decides: its `licence_class` must be permissive or research_only, or a person's override must be recorded in the source's licence record. An override that is neither a text nor a record, or a record without a licence text, refuses (StreamError). The row is research_only unless the override's record says false and its text names a known licence without a restriction word.
4. **An OtherPlant box judged `unknown`** (an out-of-fold prediction without an OtherPlant model) is masked. The contract lists it neither as kept nor as masked, and a box that is not trusted is not kept.
5. **b0000 candidates** are the non-admitted v1 images holding a verified target box. Before committing, b0000 asserts equality with `funnel/census_v1.json` (`veto.images`, `veto.lost_boxes` and the lost boxes per class = verified − admitted) and recovered ≤ lost per species; any mismatch refuses and commits nothing. The census reports 989 lost boxes in 461 images: the contract's 457 leaves out the 4 images whose only blocker is a small target box (FUNNEL_AUDIT §3.3). 457 is recorded as not asserted. Those small target boxes are masked. `--no-census` records the check as not done.
6. **Base B's 878 images are never queued.** The ones in `base_v2` are base. The L-5 ones (listed in group A's `splits/v2/l5_excluded.jsonl`, sha256-checked against LOCK v2 when it records one) are excluded outright, and a list naming a non-base key refuses.
7. **Rejoin resolutions.** `class_maps.json` holds proposals, so the accepted maps are read from recover's `step1_r1/recovery.json` (`class_maps`, `accepted: true`). Target synonyms from `name_status_v2.json` are used only with `via` scientific or override, recover's own rule. Rejoined rows are queued as `<key>__rj<version>`, and every live row of the image's chain (the original and earlier rejoins) is refused as superseded, so no image is queued twice. Changed boxes are re-judged from the stored features: out of fold for the verifier's own OtherPlant training crops. A relabelled image is a new queue row and passes what every new row passes: base B's images (base_v2 and L-5) are left out; GuardV2 runs on the unmasked image and the masked copy; near_consumed applies; an image refused for any reason but a supersession stays refused. Holds carry over (`funnel_F9` and any unknown kind; `h6_scan` and `licence` are decided anew, and `join_conflict` is what a rejoin resolves). Every row relabelled from the v1 pool holds `funnel_F9` (P8: these are the relabels the funnel's recovery arms use). A rejoin needs b0000 committed, writes its `plan.json` first, and a rejoin killed half-way is finished by its own argv. The versioned map in `index/overrides.json` then applies to later registry batches of the slug.
7a. **What blocks other verbs.** Only a planned batch (`plan.json` written, not committed) must be finished first. A directory left before planning blocks nothing, and b0000 resumes only through `backfill`, so a backfill that does not reconcile (a census mismatch, for a person) does not stop admits. An image the mask step cannot open refuses that row alone (`mask_failed`).
8. **The score of an OtherPlant-only image** is the source score select recorded (`select_summary.json`), or null. It is recorded only, so the queue does not depend on where batches split.
9. **Global crop ids are assigned at commit**, so a batch that never commits leaves no gap. v1 holds [0, n_v1), where n_v1 is read from v1 `crops.csv`; the contract's 564,686 is recorded beside it.
10. **Known truth uses the stream's own 6-bit index** (`index/base_expert.json`) over the train_core, tsw22 and tsw23 rows of LOCK v2, because GuardV2's match carries no label path.
11. **Canary verdicts are judged at float16**, the precision every shard stores.
12. **The v1 images dropped before hashing** (no label, no boxes, unmapped class, unreadable) are not in the v1 dHash cache. The first registry batch reads them once, drops them the same way, and marks them processed.

**Not done.**
- **The verifier refit (verifier v2) and re-judging the unconsumed queue.** Both are R4 (card X11). Only the proposed triggers are computed, in `status.json` `refit_triggers`, and their thresholds belong in thresholds.json (group F).
- **Reference pack v2** is a later versioned event.
- **Nothing has run on real data.** The b0000 counts and the first out-of-domain precision come from the R1 jobs [to verify on cluster], as do: v1 `crops.csv` = 564,686 crops; verifier `crops_sha256` = d8188b16…; the backfill's cost (a GuardV2 pass reads every one of about 96,500 images, in worker processes).

**How it was verified.** Locally, with no network and no GPU:
- `test_inc2_mask.py`: 27 checks.
- `test_inc2_step1_stream.py`: 167 checks. They run on test_inc_verify's synthetic world, extended with an ImageWeeds split, a tsw-like expert set, vetoed images and a split near-duplicate pair. They use group A's real GuardV2 over a synthetic LOCK v2 and cover:
  - equivalence with verify pool + crops + admit (image rule, from empty);
  - split invariance;
  - every planted guard case, including:
    - augmented evaluation copies caught through `near_eval_embed`;
    - a masked PNG of an EXIF-rotated dev copy caught through a rotation variant;
    - a masked copy refused inside the pipeline;
    - near_consumed;
  - calibration records that say `ok` without showing it: not used;
  - an image the detector cannot describe;
  - the four pin refusals, with nothing written;
  - kill and resume, including a rejoin killed half-way, a directory left before planning, and a b0000 that did not reconcile;
  - one writer across nodes: the owner file, stale takeover by pid, by `squeue` and by age, and a mount without flock;
  - b0000 reconciliation against an independently computed census;
  - rejoin: base B left out, GuardV2 on relabelled images, one live row per image across versions, `funnel_F9` carried;
  - licence texts, and the collector's licence verdict and person override;
  - holds, deadlines, `--hold`, knowntruth, intake, the status schema with `one_time` and the per-source counts, the CLI (`scan-holds`, `--lock`), and the job script's log directory, dry runs and refusals.
- **Mutation check.** 43 mutants each revert one of these corrections or break one guarded behaviour of `step1_stream.py`, `mask.py` or `run_inc2_stream.sh`. Each one fails the suite.

The existing `test_inc_verify`, `test_inc_select`, `test_funnel_recover` and `test_funnel_leak` pass unchanged. So do group A's `test_inc2_guard` and `test_inc2_common`, and group E's `test_inc2_stream`, which reads this queue through `load_queue`.

### Build note (group B)

**What was built.** `inc2/recipes.py`, `inc2/train.py` (a copy of `inc/train.py` with the §4.3 changes), `inc2/baseline.py`, `inc2/pilot4.py`, `run_inc2_job.sh`, the "Protocol v3" sections of docs/INCREMENTAL_PROTOCOL.md and docs/INCREMENTAL_PROTOCOL_RUNNER.md, and for L-3 `inc2/gate3.py` and `inc2/scorer_sidecar.py`. Tests: `test_inc2_train.py`, `test_inc2_baseline.py`, `test_inc2_pilot4.py`, `test_inc2_gate3.py`. `inc2/__init__.py` and every other file are group A's or other groups'.

**Interfaces other groups rely on.**
- **Job script.** `run_inc2_job.sh <list> [<exp>]` exports `INC_JOB_SCRIPT` as its own git-tracked path. `inc2.baseline build` and `inc2.pilot4 build` set it before `driver init` and refuse another value. `run_inc2_build.sh` (group E) and the ticker's `advance` (group F) must export it too.
- **exp.json keys the v2 executor reads:** `type`, `testing`, and `arm` (write `inc2.recipes.stamp(inc2.recipes.resolve_arm(arm, repo=REPO))` into every v2 definition, segments included). A definition without `arm` runs as n640, and only with `init_weights` yolo11n.pt.
- **Recipes.** `inc2.recipes.cold(arm)` gives base and truth runs; `incremental("r0" | "x1a" | "x1b", arm)` gives chains (name the chains `r0`, `x1a`). `stage_b_choice(ledger_entries, {"r0": "r0", "x1a": "x1a"})` gives the §5.1 Stage B chain. `step_cost`, `truth_every`, `rates`, `baseline_cost` and `measured_rates` give costs, all est. unless measured.
- **Milestones (L20).** `inc2.baseline.build(exp, manifest=P_s, role="milestone", arm=..., seeds=None, backend=..., extra={...})`, which defaults to 5 seeds and finals dev, imageweeds and test. The CLI is `build --role milestone`. `secondary(exp, weights, source=...)` writes the incumbent's spec and returns the argv; the platform submits it.
- **Baselines from the autopilot (L23B).** `build --exp E --manifest P --seeds S [--arch yolo11s --imgsz 640]` is accepted as the autopilot writes it: `--arch`/`--imgsz` name an arm of the table, and without `--role` the role is inferred (LOCK v2's base_v2 on n640 is b_v2, on another arm capacity; train_core is the canary; else baseline). Only b_v2, capacity and milestone builds read test (P10); the canary reads dev, union and baseline builds dev and imageweeds.
- **Commit (L19).** `inc2.gate3.decide_experiment(exp)` writes `INC_DIR/<exp>/gate3.json`. Read `commit_verdict`, `failed_guards` and `p_data_le_p_reject` per step, and the `truth` section for truth verdicts. The runner doc, "Protocol v3 gate", gives the exact commit reading.
- **Stage A (L25).** `inc2.pilot4 build --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b`, then `inc2.pilot4 verdict --exp pilot_v4`, which writes `INC_DIR/pilot_v4/stage_a.json`: `status` READY | PENDING and `segment1_recipes`.
- **Evidence.** `INC_DIR/capacity/capacity_v1.json` opens no test file and may join the evidence allow-list; `capacity_v1_report.{json,md}` holds test and must stay out of the digest. `canary.json`, `stage_a.json` and `gate3.json` read dev only.

**Choices where the contract is silent or ambiguous, and why.**
1. **L-3 needed a sidecar (a new module, `inc2/scorer_sidecar.py`).** The locked scorer records per-class AP over the whole exam and a correctness bit per image, not the per-image, per-class detections a bootstrap needs.
   - How it works: the sidecar calls `inc.scorer.score` unchanged, with a subclass of the scorer's own validator installed through its validator cache in the sidecar's process. It requires the captured arrays to give back the score's per_class exactly (≤ 1e-9), and the recorded score within 0.002.
   - Which runs get one: every base, union, cand and soup run scored on dev, as a second subprocess after the scorer.
   - Cost: a second dev pass plus the bootstrap (20 s on the laptop for 617 images and 108k predictions).
   - When it fails: a chain run fails at stage `sidecar`, and its retry re-scores without retraining. A baseline records the failure and goes on, since no decision there reads it, so R0's baselines cannot be stalled by it.
   - The early warning: the canary passes only if its sidecar was written. That makes the canary the first cluster reading of the hook against Ultralytics 8.4.37 (the laptop has 8.4.22) [to verify on cluster].
2. **gate3 never accepts on a rule it could not apply.** A step whose v3 decision cannot be made (no sidecar, a refused input) is recorded unavailable. Its `commit_verdict` is the pinned verdict, except that a pinned ACCEPT commits as HOLD: the v3 tolerance can be stricter than the pinned one, so a pinned ACCEPT is not a v3 ACCEPT, and a HOLD returns the increment once. The record carries the pinned failed guards for the disposition. `gate3.json` `summary.unavailable` and `summary.unavailable_held` list these steps, which a diagnosis can read.
3. **`truth3` replaces the truth arm's species tolerance too**, from the sidecar of the "without" arm's first run. The same 42-box noise drives it: realloop_v1's V4 "hurts" was species-only. Whether §3.5's "unless the truth arm says helps" reads the pinned or the v3 truth verdict is group E's call; v3 is the consistent reading.
4. **X1b's warmup is unspecified in §5.1.** It is set to 3 epochs, as in X1a ("re-warm"). Every incremental recipe has `warmup_bias_lr` = lr0, the runner's convention.
5. **The v3 recipe check compares every key but seed, cache and workers.** v1 left momentum, weight decay and close_mosaic unchecked.
6. **The capacity rule compares the seeds every arm shares (0, 1, 2).** B_v2's seeds 3 and 4 do not enter it. pooled sd = √((sd_a² + sd_n²)/2). A tie goes to the arm with fewer FLOPs.
   - `truth_every` uses the chosen arm's measured cold rate (median over its base runs), with the incremental rate at cold × 7.4/6.5 (est.).
   - M defaults to ⌈0.10 × the base's actual images⌉. With L-5 that is about 682, not 763 (group A's note).
7. **Stage A:**
   - the best survivor is chosen by agreement, then final dev, then fewer epochs;
   - the finals are dev and imageweeds (P10: no test);
   - READY means pilot_v4 is done and each arm has 7 verdicts and a final dev.
   - Every pilot_v3 bin must pass the v2 guard before the build. v1 never checked flips and rotations, so a variant copy of an evaluation image in pilot_v3's Breal would make the build refuse. That would be an incident to report, not a check to work around [to verify on cluster].
   - **[L-8, 2026-09-29]** The exception is a row whose image bytes are on splits v2's L-8 list (`train_core_variant_drops.jsonl`, read through `inc2.train.load_variant_drops`). It is removed from that bin's copy, keeping every other line byte-identical. The entry carries the new `manifest_sha256` and `n_images` beside `source_manifest_sha256`, `source_n_images` and `l8_dropped`. exp.json's `variant_drops` and the summary's `sha_identical` name the bin and the row. Why: pilot_v3's Bswap was built from session `20210909_NIKOND3300_YL`, which holds the one real L-8 image, and L-8 removes it from every v2 training manifest. One image of a 238–274-image bin is negligible next to the recorded truth verdicts, and pilot_v3 is not re-run. Any other guard refusal still refuses the build.
8. **The canary rule** is read as |canary − mean(b0_v1 seeds)| ≤ sd(b0_v1 seeds) on dev 12-class (0.8082 ± 0.0063, from `b0_v1/report.json`). It also needs a production run and score, a written sidecar, and b0_v1's base manifest (by sha256), cold recipe and seed 0, read from `b0_v1/exp.json`: a canary that trained something else, or in test mode, reproduces nothing.
   - **[L-8, 2026-09-29] b0_v1's base minus the L-8 drops.** L-8 removes from LOCK v2's train_core a row that b0_v1 trained on, so the canary trains b0_v1's base minus the listed rows.
   - **At build**, `variant_drops_record` requires v1's train_core (by the v1 LOCK's sha256, which is b0_v1's base `242ef6b9…`) filtered by the list's image sha256s (the list hashing as LOCK v2 records) to be the canary's manifest byte for byte, or the build refuses. The record goes into exp.json and `build_summary.json` as `variant_drops`: each dropped row's key, image sha256 and match, the list's sha256, the reference sha256 and the rule.
   - **At verdict**, `canary_manifest_match` accepts the manifest as identical to the reference's, or as the reference's minus the recorded drops. For the second, the canary's record must name the reference's base sha256 and the list's sha256, LOCK v2 must still record that list, the file must still hash to it, and the reference's manifest (its copy, its source or v1's train_core, whichever hashes as recorded), filtered by the listed sha256s, must hash to the canary's manifest. `canary.json` `manifest_match` says how it matched and names the dropped rows. The dev rule is unchanged: one image of 3,049 is not expected to move dev by a measurable amount.
9. **Defence in depth.** `inc2.train` refuses a GuardV2 reason it does not know, and it also looks every variant up in the v2 never-train index itself. It refuses the bytes of every L-5 image (`splits/v2/l5_excluded.jsonl`, sha256 in LOCK v2) under any key or source, and, since L-8, of every L-8 drop (`splits/v2/train_core_variant_drops.jsonl`, reason `train_core_variant_drop`). A production run refuses a LOCK v2 marked testing, or one that records no L-5 or no L-8 list.
10. **Research-only.** A baseline's exp.json records `research_only` (§8) from `splits/v2/base_v2_provenance.jsonl`, read only when it hashes as LOCK v2 records. The flag is false only when every row is known and none is research-only; a research-only row or a row of unknown licence makes it true (P6). A milestone on a pool with intake rows should pass its own record via `extra`.

**What the owner should know about L-4.**
- **Every arm trains and scores at 640 px.** The locked scorer's imgsz is 640, and the LOCK covers it, so the grid varies capacity only. A resolution arm needs a new scorer version (R4) and is not in this grid.
- **Walltime.** FLOPs-scaled from YOLO11n's rate, which is the contract's basis, one cold seed at N = 7,625 projects to at most 5.0 h for s640 and 15.8 h for m640. The pinned cold walltime is 8 h (`driver.COLD_TIME_LIMIT`). The low bracket, scaled by pixels because YOLO11n is loader-bound on a V100, projects 1.3 h for both, since every arm trains at 640 px. `cost_estimate.walltime.over_d26_line` is `[false, false]` for s640 and `[false, true]` for m640. D26 should not hold the m640 build on the high bracket alone. The first run of each arm measures the real rate [to verify on cluster]; a real overrun is card X15.
- **Batch and cache.** Every arm keeps batch 32, the same cold recipe as n640. Whether YOLO11m, the largest arm, fits a V100-32GB at batch 32 and 640 px is to verify on the cluster. The RAM cache falls back to reading from disk above 0.6 × 45G.
- **Checkpoints.** `yolo11s.pt` and `yolo11m.pt` must be placed in `$REPO` before the capacity builds. The build refuses without them and downloads nothing.
- **The truth arm.** The pinned driver's truth arm covers every step of an experiment or none. "Truth on every ⌈cost/25⌉-th step" therefore has to be realised by group E's segment builder, for example truth on in every n-th segment of K = 1; `capacity_v1.json` gives `truth_every`.

**How it was verified.** Locally, on synthetic worlds, with no network and no GPU:
- **`test_inc2_train.py` (80 checks; 87 with L-8):**
  - real 1-epoch CPU runs through the pinned scorer and the sidecar, with the real `inc2.guard`;
  - planted exact, 1–6-bit, re-encoded, hflip and rot90 copies are refused;
  - an L-5 image re-listed under another key is refused, and an L-5 list that does not hash as LOCK v2 records stops the run;
  - [L-8] the world's v1 train_core holds a transverse copy of a test image that v2's train_core and base_v2 leave out. Re-listed under another key and source, it is refused by the list and by GuardV2, and by the list alone when GuardV2 is patched to pass. A changed list stops the run, and a production LOCK with an L-5 list but no L-8 list is refused;
  - the job script's order, its `INC_JOB_SCRIPT` export and its drift stop (the funnel modules included);
  - the pinned modules are unchanged against git HEAD.
- **`test_inc2_gate3.py` (47 checks):**
  - a real CPU sidecar, and its SE against an independent recomputation;
  - the equivalence battery;
  - an unavailable step: a pinned ACCEPT commits as HOLD, a pinned REJECT stands;
  - realloop_v1's recorded ledger.
- **`test_inc2_baseline.py` (70 checks; 81 with L-8):**
  - FakeBackend baseline and 1-step chain runs to done, submitting only `run_inc2_job.sh`;
  - the autopilot's L23B argv shapes (`--arch`/`--imgsz`, no role) build B_v2, both capacity arms and the canary with the right role, arm and exams; roles that do not match, and test outside a milestone read, are refused;
  - the canary passes only as a production run on b0_v1's manifest and recipe;
  - [L-8] the canary builds on v1's train_core minus the list and records the dropped row; a manifest that is not v1's minus the list is refused. Its verdict passes on the reduced manifest, and fails with no recorded drops, with drops from another reference, with another list than LOCK v2's (even when the file matches the canary's record), or with a list changed since the build;
  - the capacity decision is test-blind.
- **`test_inc2_pilot4.py` (28 checks; 31 with L-8):** on pilot_v3's recorded chains, R0 survives with 5/7 and 0.8006, and freeze and LoRA fail with 3/7; a bin with an hflip test copy refuses the build; a HOLD on a planted bin is not a rejection; [L-8] a Bswap bin holding an L-8 drop, re-keyed and relabelled, is copied without that row and recorded, with every other bin sha-identical.
- **Mutation checks.** 45 source mutations of the group's decisions and guards (listed in docs/INCREMENTAL_PROTOCOL.md, "Protocol v3", "How it was verified") were each run against these tests in a scratch copy of the package, and every one made a test fail.
- **Existing suites, unchanged and passing:** `test_inc_train`, `test_inc_driver`, `test_inc_gate`, `test_inc_scorer`, `test_inc_realloop`, group A's `test_inc2_common` and `test_inc2_guard`, and group E's `test_inc2_stream` and `test_inc2_stream_report` (run against these modules).

Nothing has run on the cluster.

#### Amendment (2026-09-30): measurement arms m832 and s1024, beside L-4's grid

**Why.** On base_v2 (6,811 images, 3 seeds each) the capacity grid gives test 12-class mAP50-95 0.8468 (n640), 0.8653 (s640) and 0.8786 (m640). These are R0's milestone read of test (L-4). The gap to 0.90 is 0.021, and it sits in the small prostrate weeds: under m640, Carpetweed 0.736, SpottedSpurge 0.810 and Purslane 0.828. Input resolution is the next lever to measure.

**What changed.**
- `inc2/recipes.py`: two arms. `m832` is yolo11m.pt at 832 px, batch 16. `s1024` is yolo11s.pt at 1024 px, batch 32. Their `gflops` are the 640 figure × (imgsz/640)², and `gflops_at_640` stays the 640 figure, because scoring runs at 640. An arm's `batch` now replaces the table's 32 in every recipe of that arm, and no other arm sets one. `GRID_ARMS` (n640, s640, m640) and `MEASURE_ARMS` (m832, s1024) name the two sets.
- `inc2/baseline.py`: on base_v2 the two arms build as role `capacity` (3 seeds; finals dev and imageweeds). They are built after R0, at no milestone read, so they never read test (P10): a build of a measurement arm that lists test is refused, whatever its role. `capacity-verdict` refuses a measurement arm in `--arms`. `--record b_v2_m832,b_v2_s1024` lists them record only: their dev mean against n640, and their finals in `capacity_v1_report`. They never enter `qualifying` or `chosen`. An existing `capacity_v1.json` is kept byte for byte, because the stream adopted it by sha256. A recomputed decision that differs from it refuses the run.
- `inc2/stream.py`: `init` and `choose-arm` accept only a grid arm.
- Autopilot:
  - `stream_domains/weed.json` holds the baselines `cap_m832` (b_v2_m832) and `cap_s1024` (b_v2_s1024), marked `measure`, and `capacity.measure_arms`, whose cost factors are 4.0 and 3.5.
  - DR0 proposes them as L23B, within the envelope, one at a time and only while missing. It does so only once R0 is complete, meaning the stream has adopted its arm and read Stage C, and nothing else of R0 is due, in MAINT or DATA. R0 READY never waits for them. Their proposals cite only `/stage/lock` and `/stage/baselines/<id>`. The whole `/stage` changes with every segment the TRAIN lane builds, and a changed cite would leave the grant waiting for a person.
  - A failure of theirs never stops the stream. `stream._health` runs D5 and D6 on them as on every live experiment. A measurement arm's D5 loses OP_PAUSE (`_record_only`), so it neither pauses the campaign nor stands as a firing stop-loss that would refuse every envelope grant (`executor._trigger_check`); L7 still unblocks its transient units. A block that is not transient (`failed_run`, two failed attempts, for example a CUDA OOM) and a stale advance (D6) file one escalation card each, and the lanes run on. The experiment stays built, so DR0 does not propose it again; a person unblocks it or leaves it. D7 and D14 still pause the stream on any experiment.
  - They share the stream's budget. At the default caps (120 SU a UTC day, 350 SU a month) a K=4 segment on m640 with r0 prices at about 101 SU, so an arm granted earlier the same UTC day (20.7 or 18.7 SU) defers that segment's envelope grant to the next UTC day. The two arms use about 39 SU, 11 % of a month's window.
  - `stream_remote` and the `inc_build_baseline_v2` policy row admit `--arm m832 | s1024`.

**What they measure.** Production scores are the locked scorer's, and it infers at 640 px (L-4). These arms therefore measure what training at 832 or 1024 px gives under 640 px inference. They do not measure inference at the larger size: that needs a scorer version whose imgsz is the arm's (R4). **They do not change the stream's arm.** The capacity decision (m640) stays, because nothing reads them as candidates and `choose-arm` refuses them.

**Resource estimate (est.; V100-32GB, cache ram, one `run_inc2_job.sh` job with `--mem=45G`).**
- **Time.** Measured per base run (100 epochs plus the dev score): n640 about 1.2 h, s640 1.58 h and m640 2.68 h, that is 6.3, 8.3 and 14.2 ms per image-epoch. YOLO11m at 640 is GPU-bound.
  - m832: 14.2 × 1.69 gives about 4.5 h per base run, 5.0 h with 10 % for batch 16, and 14–16 GPU-h for 3 seeds with finals.
  - s1024: 8.3 × 2.56 gives at most about 4.0 h per base run, and 10–13 GPU-h.
  - Both are well under the 8 h cold walltime (`driver.COLD_TIME_LIMIT`) and D26's 6.4 h line. The platform prices the builds at 20.7 and 18.7 GPU-h, the build job included.
- **GPU memory.** The activations saved for backward were measured per image on the CPU in fp32 (Ultralytics 8.4.22, train mode with the loss): m640 833 MiB, m832 1,410 MiB, s1024 1,014 MiB.
  - At batch 32 that is 26 GiB for m640, which fits one V100-32GB under AMP as it ran; 32 GiB for s1024 (1.22 × m640, fits); and 44 GiB for m832 (1.69 × m640, about 27–30 GB under AMP, too close to 32 GB).
  - m832 therefore uses batch 16 (0.85 × m640). Ultralytics accumulates to its nominal batch of 64 and scales weight decay to match, so the optimizer step is unchanged. Only BatchNorm's batch statistics differ.
  - These are CPU extrapolations anchored on m640 having fit; m640's logged GPU peak has not been read, and neither arm has run at 832 or 1024 px on a GPU. If s1024 at batch 32 runs out of memory, its runs fail and block, which is a card (above), not a stop of the stream. Reading m640's logged `GPU_mem` first tells whether s1024 needs batch 16 like m832 (above about 24 GB for m640 it does).
- **RAM cache.** About 15.5 GB at 832 and 23.5 GB at 1024 for base_v2, under the 27 GB (0.6 × 45G) at which `inc2.train` falls back to disk.

**How it was verified.** Locally, with no GPU:
- `test_inc2_baseline.py`: the arms resolve with their checkpoint's sha256; the cost factors hold; a batch-32 m832 recipe deviates; the grid's recipes are unchanged; capacity builds by `--arm` and by `--arch/--imgsz`, with finals dev and imageweeds, and a measurement build that lists test is refused; the `--record` rules above, including a record arm that lacks one of the grid's seeds or was trained on another manifest (refused) and one with an extra seed (the decision's seeds and outcome are the grid's alone).
- `test_inc2_train.py`: the table check reads every grid arm's cold recipe at 640 px and each measurement arm's at its own size. It previously asserted 640 px for every arm.
- `test_inc2_stream.py`: `choose-arm` and `init` refuse both arms, and the stream's arm stays.
- `test_stream_ap_units.py`: prices and grammar. The arms are not proposed before R0 is complete, while Stage C runs unread, or while a DATA item of R0 (the L17 backfill) is due. Then m832 is proposed, then s1024, then nothing, with the narrow cites, with L18 still cut and with the arm, the LA count and `capacity_v1.json` unchanged. A `failed_run` block of b_v2_m832 files a card, the campaign stays enabled and s1024 is still granted within the envelope; a stale advance of it (D6) files a card too; the same block on a segment still pauses the stream.
- `test_stream_ap_replay.py` (stream_r0): R0 step by step, then both arms within the envelope.
- `test_stream_pipeline.py`: the real `inc2.baseline` builds both arms after R0 on the pinned driver, with finals dev and imageweeds, while segment 1 is cut and committed, and the stream keeps the arm its decision chose (n640 in that world).
- 11 source mutations, each run in a scratch copy, make a test fail. Nine further mutations fail a test as well: DR0 proposing an arm while a DATA item of R0 is due; `capacity-verdict` without the record arms' seed check, taking the seed intersection over the record arms, or leaving them out of the one-manifest check; a measurement arm's finals defaulting to test, or its test refusal removed; its D5 keeping OP_PAUSE; its D6 pausing; and its card filed once per summary rather than once per block.

**Deploy.** `brain/policy_actions.json`, `stream_domains/weed.json` and the autopilot modules change `executor.code_hash()`, and `diagnose_stream.py` and `levers_stream.py` change the stream rules version. After the lab and cluster copies are synced from one commit, run `executor.run_replay_tests` so that envelope builds are granted again. The next L18 writes its prospective record under the new rules version.

#### Amendment (2026-10-01): the measurement arms read at their own resolution (pre-registered)

**Why.** The first measurement arm is in: b_v2_m832's dev 12-class mAP50-95 is 0.8591 ± 0.0018 against b_v2_m640's 0.8524 ± 0.0025 (3 seeds each, the locked scorer at 640 px), and Carpetweed is down 0.014 on dev. The hypothesis behind the arms is that the small prostrate weeds (Carpetweed, SpottedSpurge, Purslane) need more pixels at **inference**. The locked scorer infers at 640 px and refuses any other size, so it cannot test that hypothesis. This amendment adds a second, separately named scorer that changes imgsz only.

**Pre-registration.** This paragraph was written before any native-resolution score existed, and it is not edited afterwards.
- **What is scored.** Each measurement arm's final runs (`final__base__s<k>` of b_v2_m832 and b_v2_s1024; their weights are the base runs' final EMA weights), on dev and on ImageWeeds, at the arm's own training imgsz (832 or 1024). b_v2_m640's final runs are scored on dev at 640 by the same code, as the reference. **Test is never scored**: the native scorer refuses the exam `test` at every size, so nothing scores test at a size other than 640. Nothing else is scored by it: an experiment whose arm is not a measurement arm, a size other than the arm's own, or a run other than a baseline's final run is refused.
- **How.** The locked scorer's own code is used as a library (`inc/scorer.py`: the LOCK checks of the exam manifest and of scorer.py, the materialised-exam check of every image and label, the model check, the temporary exam view, its validator, `conf` 0.001, `iou` 0.7, batch 32 with rect batching, fp16 on a CUDA device, Ultralytics' default `max_det` 300, the pinned Ultralytics 8.4.37, the exam manifests and their key order, `species_map50_95`). Only imgsz changes. The per-image detections are captured in the same pass with the scorer sidecar's validator (`inc2/scorer_sidecar.py`), and they must reproduce the score's per-class AP exactly. Each score is written as `scores/<exam>@<imgsz>.json` beside the run's other scores, with its per-image arrays as `scores/<exam>@<imgsz>.images.npz`, in its own format `inc2-native-score/1`. It records the locked scorer's sha256, the native scorer's sha256, imgsz and every setting. It says `production: false` and carries a scorer stamp prefixed `NATIVE<imgsz>-`, so no gate, milestone or capacity decision can take it for a protocol score. A native score is written once and never overwrites anything: a run's 640 px scores are never touched. The reference's dev score at 640 must reproduce its run's recorded protocol dev score within 0.002 (the sidecar's tolerance), or it is refused.
- **The comparison.** The statistic is dev `species_map50_95` (the 12-class mean). The arm at its native imgsz is compared with b_v2_m640 at 640 px, on the seeds both share (0, 1, 2), from the native score files. mean and sd (sample sd) are taken over those seeds, and pooled sd = √((sd_arm² + sd_m640²)/2), as in L-4. The difference is D = mean(arm) − mean(m640). Its standard error is an image bootstrap paired by exam image, the scorer sidecar's method:
  - 1,000 resamples of the dev images with replacement, drawn once as numpy.random.default_rng(stable_int("inc2/native/diff_se")) over the exam's key order, and used for every run;
  - per run and resample, each species' AP50-95 is Ultralytics' ap_per_class on that run's tie-broken per-image arrays, each image's detections and GT boxes counted as many times as the image was drawn;
  - the 12-class mean is taken over the species with a GT box in the resample; it is averaged over seeds per arm, and the arm minus m640 is D_b;
  - SE(D) is the sample sd (ddof 1) of the D_b. Each species' difference and its SE are computed the same way.
- **The decision rule.** An arm **qualifies for a stream fork proposal** only if all three hold:
  1. D > 2 × pooled sd;
  2. D > SE(D);
  3. at least one of Carpetweed, SpottedSpurge and Purslane improves: its mean dev AP50-95 over the shared seeds at the arm's imgsz is higher than m640's at 640.
  
  The verdict is recorded in `INC_DIR/capacity/native_v1.json`, dev only. A qualifying arm files one R4 card (X18) for a person, and nothing else changes: the stream's arm stays the capacity decision's, `capacity_v1.json` is not touched, nothing switches automatically and test is not read. A fork at a native resolution would also need the gate, the milestones and the stream's comparisons scored at that resolution, which is a new protocol version. An arm that does not qualify is recorded, and nothing follows from it. ImageWeeds native scores are reported for people and never enter the rule.
- **Inputs that refuse the verdict.** Any of these refuses the verdict:
  - native files of one arm that disagree on the exam manifest, key order, locked scorer, a setting other than imgsz, or Ultralytics' version;
  - a reference that disagrees with the arm on any of those;
  - fewer than 2 shared seeds;
  - a test-mode score in a production verdict.

  An arm without all its native dev scores is listed as pending.

**What was built.**
- `inc2/scorer_native.py` (new). It is the locked scorer used as a library, with `imgsz` as the only change.
  - Reused unchanged: the checks (`check_lock`, `check_exam`, `load_model`, `_check_model`), `build_view`, `_device`, `_precision_kwargs`, `CONF`, `IOU`, `BATCH` and `deviations`.
  - The validator is the sidecar's subclass of the scorer's own. The captured arrays must reproduce the pass's `per_class` exactly (`scorer_sidecar.check_capture`).
  - `score_run(exp, run, exam)` scores a done final run at the arm's own size: 832 or 1024 for a measurement arm, 640 for m640.
  - It refuses everything the pre-registration names, before anything is written. That includes m640 on any exam but dev: `score_run` refuses the reference arm's ImageWeeds, and `score` refuses any exam but dev at 640. It also refuses weights that cannot be loaded, and a size Ultralytics would round to another stride multiple.
  - A score is put in place under its own lock file, `scores/.<exam>@<imgsz>.commit`, which is created exclusively and held only for two renames. The npz is renamed into place only while no JSON is there; a leftover npz from an attempt that wrote no JSON is replaced. The JSON is then hard-linked into place only if it is absent. So a JSON always names the npz beside it. A second writer of the same score is refused and replaces neither file, and its npz is discarded. A writer that finds the lock held waits up to 60 s, then is refused and leaves the lock to its holder. The run's 640 px score is hashed before and after the pass.
- `inc2/baseline.py`: two new verbs.
  - `rescore-native --exp E [--reference b_v2_m640]` first checks that every final run of the arm and of the reference is done. It then scores each missing native file and keeps those already written. It writes `INC_DIR/<exp>/native_rescore.json`, then the verdict. The record holds the status (complete) and the dev files' names and sha256s, with no path, because the platform reads it as evidence. The returned record lists every native file, and the verdict's report lists every file it read.
  - `native-verdict [--arms b_v2_m832,b_v2_s1024]` writes `capacity/native_v1.json` (dev only) and `native_v1_report.{json,md}`. The report is for people and holds dev and ImageWeeds at the native size and at 640.
  - `--reference` and `--arms` now default per verb. canary-verdict and capacity-verdict keep their old defaults.
  - Both verbs set `YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` before Ultralytics is imported, as the locked scorer's and the native scorer's CLIs do. Importing `inc2.baseline` does not import Ultralytics.
- `run_inc2_build.sh` accepts `inc2.baseline rescore-native`. Its lock and provenance are named `native_<exp>`, so the arm's own build record is never touched, and it records status `scored`. No driver advance follows it, because it builds nothing. `scorer_native.py` joins the module drift check.
- Autopilot: lever **L23N** (`stream_levers.json`, policy action `inc_rescore_native`).
  - It is in the MAINT lane, follows its job, and is R3 within the envelope. It is priced by the new estimator `rescore`: `cost.rescore_hours_per_run` 0.25 GPU-h per final run, for the arm's 3 runs and the reference's 3, so 1.5 GPU-h. For comparison, the arm's L23B build is 20.7 GPU-h.
  - DR0 proposes it on the arms' own conditions. R0 must be complete, nothing else of R0 may be due in MAINT or DATA, the arm's experiment must be done, and `/stage/native/<id>` must be missing. It cites only `/stage/lock`, `/stage/exp_status/<exp>` and `/stage/native/<id>`.
  - `/stage/native` is `done` when the shipped `<exp>/native_rescore.json` says complete. Otherwise it is what the platform ran: `running` from submission, then `done` or `failed`. So each arm is proposed once.
  - A failure of the rescore is record only (`stream.RECORD_ONLY_LEVERS`). This covers its job ending without success and a refusal of its submission. It files one escalation card and marks the arm `failed`. It never counts as a failed step, and it neither holds the lane nor pauses the stream. It is not proposed again.
  - A submission whose outcome is unknown (the verb may have run, but no reply came back) does not pause the stream. The item is followed by its job name, `inc_build_native_<exp>`, and by its record. It stays running while a job of that name is queued. It is done once the record says complete. It fails, record only, after 3 snapshots with neither.
  - Two platform-wide rules still apply to it as to every lever. An sbatch refused on qos is a platform defect (S21): it holds the MAINT lane and files a platform card. A price the campaign envelope cannot pay pauses the stream.
  - The MAINT lane runs one item at a time, so while the L23N job is queued or running every other MAINT step waits for it, queue time included. That means a milestone compare (LC), a recommended rollback (L21, which runs while TRAIN is held), a bisect (L27), a milestone (L20) and an audit (L4). The ordering has one more effect. A compare that runs between the two arms' rescores is a login-node verb, so the lane is free as soon as it returns. DR0 can then give the lane the second arm's rescore before the compare's 'hurts' reaches the evidence, and the rollback the compare recommends waits for that job too. In total the wait is two scoring jobs (1.5 GPU-h each, est.) plus their queue time, over the stream's life.
  - DNAT reads `capacity/native_v1.json`. An arm in `qualifying` fires lever X18, so the ticker files card **X18** ("Fork the stream to a measurement arm read at its own resolution", R4) once. Nothing switches, and no lane holds.
  - The plumbing for L23N:
    - executor: `ARGV_FORMS`, `STREAM_REMOTE`, `STREAM_ENVELOPE_LEVERS`, the timeout;
    - `brain/approvals.ENVELOPE_ACTIONS`;
    - the policy row;
    - `stream_remote`: the build grammar (`--exp`, `--reference`, both required), the job name `inc_build_native_<exp>`, `native_rescore.json` among the shipped records, and `capacity/native_v1.json` in the summary;
    - `evidence.ALLOWED`: both files, never the report;
    - `stream_domains/weed.json`: `capacity.native` (the reference `b_v2_m640`, the record's path) and `cost.rescore_hours_per_run`.
- A fix in `stream._fold`. Its loop over the stream ledger reused the name `names` for a list of the experiments the ledger built, which overwrote the set of queued job names that the items below it read. So an uncertain build, which has no job id, was checked against the wrong names, and the same would have held for the record-only follow. The loop's list is now `built`.

**How it was verified.** Locally, with no GPU and no cluster:
- `tests/test_inc2_native.py` (new, 84 checks). Real Ultralytics passes on the CPU, in test mode, in test_inc2_train's synthetic world:
  - every refusal above, the CLI's `--exam test` and the reference's `--exam imageweeds` included (exit 2, nothing written);
  - putting a score in place: a leftover npz without its JSON is replaced. A second writer of the same score, staged before the first one commits, is refused and replaces neither file. While the lock is held, a writer is refused after the wait, and the lock is left to its holder;
  - both verbs run with `YOLO_OFFLINE` and `YOLO_AUTOINSTALL` set, and importing `inc2.baseline` loads no Ultralytics;
  - an m832 final run scored at 832: its file, stamps and settings, arrays that reproduce `per_class` exactly, and the 640 score unchanged;
  - the reference at 640 reproducing the pinned scorer's own score of the same weights exactly (`per_class`, `image_correct`, key order). This is the check that only imgsz differs between the two scorers;
  - a recorded score off by 0.01 refused;
  - `rescore-native`'s files and record. Every exam name, manifest request and opened path is recorded, and none touches test. `native_rescore.json` lists the dev files only, with no path, and the evidence scrub drops nothing from it. A second arm's rescore keeps the first arm's decision byte for byte. A production verdict refuses kept test-mode scores, both for a non-testing experiment and with test mode off. A second run writes nothing;
  - the rule on synthetic numbers: it qualifies with all three conditions, and fails with each condition missing alone. At D = 0.011, the 2 pooled sd and the SE conditions are each tested separately. Pooled sd is checked against √((sd_arm² + sd_m640²)/2) in every case. With unequal sds (0.006 and 0.0005), D = 0.010 qualifies, though it is below 2 × the larger sd and 2 × √(sd_arm² + sd_m640²);
  - each of the 14 stamps and settings, mismatched alone in one arm file or in the reference's files, refuses the verdict and is named in the refusal;
  - the paired bootstrap against an independent recomputation (one `ap_per_class` call per run and resample).
- `tests/test_stream_ap_units.py` (the new `t_native`, 35 checks; `t_menu` checks L23N's policy row, its bounds and the executor's rendering, as for every lever):
  - the lever against the executor and the policy table, its price, and the cluster's grammar;
  - the proposal: not for an arm still running, not before R0 is complete, not while a DATA item of R0 is due, and not once its record says complete. It is then proposed once per done arm, citing only the lock, the arm's status and its own state, and granted within the envelope;
  - the TRAIN and DATA lanes run beside it;
  - a failed job files one card, holds no lane and pauses nothing, and the arm is not proposed again;
  - a submission whose reply never came (the World's `lose_reply`) is followed by its job name. It is not lost while that job is queued, and it is done once its record is complete. Without a record it fails after 3 snapshots, record only. A qos refusal holds the lane, files a platform card and leaves the arm unmarked;
  - the MAINT wait, pinned: a milestone compare that falls due while the L23N job is queued runs only after that job ends. The second arm's rescore then takes the lane, and the 'hurts' rollback runs only after that job ends too;
  - card X18 is filed once for a qualifying verdict, with test-blindness on that path, and no card is filed when no arm qualifies.

  `t_measure` now builds the arms without finishing them, because a done arm is followed by its rescore. The test world's `ready_r0` marks the measurement arms rescored (`native_rescore.json`), so the S-cases do not change.
- `tests/test_stream_ap_replay.py` stream_r0: after the arms, both rescores run within the envelope, each once, under their own job names, in the platform's R0 sequence.
- `tests/test_stream_pipeline.py`: the platform proposes L23N once per done arm. The real `rescore-native` refuses that world's untrained stand-in weights (exit 1), so the failure path runs end to end: two cards, no pause, no held lane, not proposed again.
- `tests/test_inc2_stream.py`: `run_inc2_build.sh rescore-native` writes the `native_<exp>` provenance with status scored and no advance; a refusal is recorded as build_failed.
- **Mutation checks.** 59 source mutations of the new checks were each run in a mkdtemp copy of `weed_llm_benchmark/`, and every one makes its test fail. They cover:
  - scorer_native: 21;
  - baseline: 19, including each of the rule's three conditions, `> 0` changed to `>= 0` for "improves", and `all` changed to `any`;
  - autopilot: 17;
  - `run_inc2_build.sh`: 2.

  On the first pass 57 were killed. Two survived because a second guard still refused the call: the rescore's shared-seed check (the verdict's own check refused, but only after the record was rewritten) and the per-file key-order check (the bootstrap's refused instead). The tests now check the specific refusal, and that nothing was written. Both mutants are killed.

  After the first pass, these were not yet pinned by any test: the pooled-sd formula, four of the stamps and settings, the production gate, and the merge of the arms already recorded. The checks above were added for them and for the fixes listed under "What was built". 23 further mutations were each run the same way, and every one is killed:
  - the uncertain follow, the `names` fix, the record check and the lost count: 4;
  - the commit lock: 3;
  - the reference's dev-only rules: 2;
  - the offline environment: 1;
  - three wrong pooled-sd formulas: 3;
  - four dropped stamps and settings: 4;
  - three weakened production gates: 3;
  - the arms' merge: 1;
  - a record listing every exam, and a record carrying its path: 2.

  On that round's first pass, the four dropped stamps and settings survived, because the check read its field list from `baseline.NATIVE_STAMPS` and `NATIVE_SETTINGS`. The check now writes out the pre-registered fields itself, and all four are killed.

**Not verified here** [to verify on cluster]: a GPU pass at 832 and at 1024 in fp16 with batch 32 under Ultralytics 8.4.37. The first L23N's 640 reproduction of b_v2_m640's recorded dev scores is the first cluster reading of the native scorer.

**Deploy.** These files change `executor.code_hash()`: the autopilot modules, `stream_domains/weed.json`, `stream_levers.json`, `brain/policy_actions.json`, `brain/approvals.py` and `tests/test_stream_ap_replay.py`. `diagnose_stream.py`, `stream_levers.json` and `levers_stream.py` change the stream rules version. Sync the lab and cluster copies from one commit (outer and nested, `scorer_native.py` included), then run `executor.run_replay_tests` so that envelope grants resume. The next L18 writes its prospective record under the new rules version.

#### Amendment (2026-10-01): box-quality measurement arms l640, y26m640 and y26l640 (pre-registered)

**Why.** The 12-class test score is bounded by how well the boxes are placed, whatever the species call.
- On test (R0's milestone reads, 3 seeds each), class-agnostic mAP50-95 is 0.8751 (n640), 0.8842 (s640) and 0.8901 (m640), against 12-class 0.8468, 0.8653 and 0.8786. A perfect species call on m640's boxes would score about 0.89, under the 0.90 bar.
- Data has so far fixed the species call, not the boxes. Segment s001 (CottonWeedDet3, 755 images, 1,240 verified boxes) took dev 12-class from 0.8524 ± 0.0025 (b_v2_m640) to 0.8606 (the committed incumbent), while dev agnostic went from 0.8695 ± 0.0035 to 0.8674. The 12-class/agnostic gap on dev went from 0.017 to 0.007. A rehearsal without the new data (the s001 null arm) gave 0.8489, so the gain is the data's, not more epochs'.
- Resolution did not move the boxes either. b_v2_m832 read at 832 px gives dev 0.8566 ± 0.0033 (native verdict: D = 0.0042 < 2 pooled sd 0.0058; no target species improved), and its agnostic dev is 0.868.
- The worst species loses mostly on localisation. From the per-image arrays of b_v2_m640's three dev finals (statistics only), Carpetweed's recall at IoU 0.5 is 0.90–0.93, but of its matches at IoU 0.5 only 0.645–0.649 also match at IoU 0.9. Every other species keeps 0.72 (Ragweed) to 1.00 of its matches at 0.9. Carpetweed's dev AP50-95 stays within 0.684–0.704 under every model, data and resolution tried.

Agnostic test grew with capacity alone. So whether a larger or a newer detector places better boxes decides whether the loop can reach 0.90 on m640's data.

**Pre-registration.** This paragraph was written before any of these arms was built, and it is not edited afterwards.
- **The arms.** Each is a COCO checkpoint at 640 px with batch 32 and m640's recipes, key for key (`inc2.recipes.table(arm) == table("m640")`). The checkpoints are pinned by sha256:
  - **l640**: yolo11l.pt, `9ebd0e09…`;
  - **y26m640**: yolo26m.pt, `401cea9a…`;
  - **y26l640**: yolo26l.pt, `9fe3c544…`.

  YOLO26 is NMS-free and assigns small targets on their own. Each arm runs 3 cold seeds (0, 1, 2) on base_v2. Its finals are dev and ImageWeeds, never test (P10). Each is a measurement arm, never a candidate of the capacity decision or the stream's arm.
- **How each is read.** By `inc2.baseline rescore-native`, at its own imgsz, which is 640, on dev only: its other 640 px scores are the locked scorer's own. Each dev score must reproduce the run's recorded protocol dev score (the 640 reproduction check) or it is refused.
- **The rule.** The 2026-10-01 native rule, unchanged. An arm qualifies for a stream fork proposal only if all three hold, against b_v2_m640 at 640 on the shared seeds:
  1. D > 2 × pooled sd;
  2. D > SE(D), the paired image bootstrap with 1,000 resamples under `stable_int("inc2/native/diff_se")`;
  3. one of Carpetweed, SpottedSpurge and Purslane has a higher mean dev AP50-95.

  The verdict is recorded in `capacity/native_v1.json` beside m832's and s1024's. A qualifying arm files card X18 for a person, nothing switches automatically, and test is not read. Changing the stream's arm is a person's decision. The stream then trains on the new arm, and its next milestone (5 cold seeds on the accumulated pool) reads test.
- **Reported for people, outside the rule.** Agnostic dev mean ± sd; per-species dev; Carpetweed's share of IoU-0.5 matches that reach IoU 0.9; ImageWeeds.
- **Order and pace.** DR0 proposes y26l640, then y26m640, then l640 (the largest expected box gain first), after the two resolution arms, one at a time, under the same conditions as m832 and s1024. Each is followed by its L23N.

**Cost (est.).**
- **Pricing.** The platform prices the builds at 18.7 (y26l640), 16.7 (y26m640) and 16.7 (l640) GPU-h, the build job included. `capacity.measure_arms` cost factors are 3.5, 3.0 and 3.0, against n640's 7.0 ms per image-epoch.
- **Expected time.** From m640's measured 2.68 h per base run, scaled by GFLOPs at nc 13 (YOLO11m at 640 is GPU-bound): about 3.7–4.0, 2.9–3.2 and 3.4 h per base run. That is under D26's 6.4 h line and the pinned 8 h cold limit.
- **Rescores.** 1.5 GPU-h each.
- **Daily cap.** At 120 SU a UTC day, one or two arms run per day beside the segments. [amended 2026-10-04: there is no daily cap by default; the arms are bounded by the envelope and by MAINT's one item in flight.]
- **Left out.** yolo11x and yolo26x (195.5 and 208.7 GFLOPs) would take about 8 h per base run on one V100.

**Measured before the change** (Ultralytics 8.4.37, nc 13):
- **GFLOPs at 640:** yolo11m 68.240 (the table's figure), yolo11l 87.325, yolo26m 74.821, yolo26l 93.221.
- **Activations saved for backward** (one image, train mode with the loss, fp32, CPU): yolo11m 949 MiB, yolo11l 1,232 (1.30 ×), yolo26m 1,100 (1.16 ×), yolo26l 1,374 (1.45 ×).
- **Logged peak `GPU_mem`** (V100-32GB, seed 0): b_v2_m640 15.8 GB at batch 32, b_v2_s1024 21.7 GB at batch 32, b_v2_m832 13.7 GB at batch 16. So the largest new arm should peak at about 23–26 GB at batch 32.
- **The locked scorer on an NMS-free detector.** On the cluster, a yolo26n.yaml and a yolo11n.yaml (the control) were each trained for 40 epochs on synthetic shapes at 64 px, then scored in test mode on a synthetic exam. The locked scorer's 12-class mAP50-95 equals a plain Ultralytics val of the same exam exactly: 0.099645 for YOLO26 and 0.115328 for YOLO11.

**What changed.**
- `inc2/recipes.py`: the three rows; `MEASURE_ARMS` is now m832, s1024, l640, y26m640, y26l640.
- `inc2/baseline.py` `rescore_native`: an arm that trains at 640 is read on dev only. The native scorer reads nothing else at 640, and before this change it would have refused the arm's ImageWeeds and failed the rescore.
- `stream_domains/weed.json`: the baselines `cap_y26l640`, `cap_y26m640` and `cap_l640`, marked `measure`, and their cost factors.
- `stream_remote`'s build grammar and the `inc_build_baseline_v2` policy row admit the three arm ids. Both listed the arms by name, so the cluster would have refused the builds.

**A defect found on the way: an approval past today's cap failed instead of waiting** (`executor.execute_approved`).
- **The defect.** An item that exceeds the SU left under today's cap is filed for a person. Approved before the UTC day turned, it was refused by the policy gate, which escalated the same shortfall to an approval the item already had. The stream read that refusal as a failed step, cleared the lane, and filed the item again under a new approval id. Two in a row would hold the MAINT lane. With arms priced at 16.7–18.7 SU, this would have hit the second arm of a day.
- **The fix.** `execute_approved` now checks the campaign's caps (`budget.fits`) before the policy gate. An approval never lifts a cap, and a refusal that names today's cap or the month's window is one the stream waits on (`WAIT_REFUSALS`), with the approval still open. When the day turns, the item runs once under that approval. An item filed past the cap and left alone still runs within the envelope after the turn, as before.
- **Its test.** `tests/test_stream_ap_cap_approved.py` (new) fails 2 of its checks against the old executor and passes with the fix.

**How it was verified.** Locally, with no GPU:
- `test_inc2_baseline.py`: the three arms' models, imgsz, batch and recipes (m640's table); resolution with their checkpoint's sha256; `--arch/--imgsz`; a y26l640 build on base_v2 (role capacity, 3 seeds, dev and ImageWeeds, a definition the pinned driver accepts); a box-quality build that lists test is refused.
- `test_inc2_native.py`: a y26l640 arm is rescored at 640 on dev only, each dev score reproducing its protocol score exactly (max abs diff 0.0), and is decided against the reference beside m832.
- `test_inc2_train.py`: every measurement arm's cold recipe is at its own imgsz (832, 1024, 640, 640, 640).
- `test_stream_ap_units.py`: the five arms are proposed in the domain's order, each once its predecessor exists. The fourth, past today's cap, is filed for a person; approved, it waits with no failed step and runs under that approval when the day turns. Five rescores follow, each once.
- `test_stream_ap_replay.py`: the five arms within the envelope, a day apart.
- `test_stream_pipeline.py`: the real `inc2.baseline` builds all five on the pinned driver; their rescores refuse the stand-in weights, record only. The rollback-order check ignores record-only MAINT items (L23B and L23N after R0). The summary-refresh checks wait for the next snapshot, since a tick that submits takes none.
- Broad regression: all 102 test scripts of the inc, inc2, stream, collect, funnel, policy, approvals and round groups pass (`test_brain_api.py`, outside these groups, was not rerun).

**Deploy.** `brain/policy_actions.json`, `stream_domains/weed.json` and the autopilot modules change `executor.code_hash()`. Sync the lab and both cluster copies from one commit, then run `executor.run_replay_tests` so that envelope grants resume. The arms need their checkpoints in the cluster REPO (downloaded 2026-10-01, sha256 above).

### Build note (group E)

**What was built.** `inc2/stream.py` (verbs `init`, `choose-arm`, `cut` (dry run), `build`, `commit`, `milestone`, `compare`, `rollback`, `bisect`, `feasibility`, `fork`, `quarantine`, `unquarantine`, `release`, `withdraw`, `summary`, `verify`, `status`), `inc2/stream_report.py` (`--stream`, `--segment`), `run_inc2_build.sh`; tests `test_inc2_stream.py` and `test_inc2_stream_report.py`. `inc2/driver3.py` (R5) is not built: its trigger is evidence from three segments.

**Interfaces other groups rely on.**
- **CLI.** `inc2.stream.build_parser()` is the argparse parser. The argv forms are those of §6.3, plus flags an autopilot lever may pass (`stream_levers.json` has carried them): `build … --arch A --imgsz N` (refused unless they name the stream's own arm, which only `choose-arm` sets) and `--truth-every N` (it can only make the truth arm more frequent), and `feasibility … --recipes r0,x1a`. `feasibility --m` must equal the stream's M (another M needs a person). `quarantine` takes `--stream` only when more than one live stream exists, and cites D28 or D31 (another cite needs a person); `release --stream SID --hold funnel_F9` (no `--source`/`--keys`) releases every funnel_F9 row. `release --hold licence` takes `--licence TEXT` and `--not-research-only`, and records both; the rows it lifts are research_only in the cut's rows sidecar unless the release records both (fail closed), and `--not-research-only` is refused for a text that names no known licence or restricts use. `release` never lifts `h6_scan`: the copy scan is mandatory, so only step1_stream's serve-holds releases it. Exit 0 done, 1 refused (reason on stderr as `[inc2.stream] ERROR: …`), 3 another writer holds `stream.lease`. `--decided-by` defaults to `platform`; a `human:…` value in `INCAP_DECIDED_BY` is taken when the flag is absent.
- **Experiment names.** Every experiment the stream builds is `<sid>_[smcb]NNN`: segments `sNNN`, milestones `mNNN`, Stage C `c001`, bisect arms `bNNN` (one per suspect increment, numbered across rollbacks). Each build prints `[inc2.stream] built experiment NAME`, which `run_inc2_build.sh` reads to advance it.
- **`stream/<sid>/queue_summary.json`** (format `inc2-stream/queue-summary/1`, dev numbers only): `pool` {`current`, `images`, `sha256`, `research_only`, …}, `eligible` {`images`, `target_boxes`, `oldest_utc`, `oldest_age_days`, `by_source`, `uncuttable`}, `held` {hold: {`rows`, `past_deadline`}} (the shape DHOLD reads), `consumed_last` (target boxes per species of the last two committed segments), `species_deficit`, `cut` {`ready`, `k`, `probe`, `train_idle`, `last_refusal`}, `segments`, `in_flight`, `uncommitted_done` (D23), `milestones` {`due`, `due_reasons`, `accepted_since`, `segments_since`, `in_flight`, `last_good`, `records`}, `boundary_check` (§5.5.1), `rollback_pending` (D25 → L21), `x4`, `suspect_increments`, `dispositions`, `data_blamed_sources`, `last_segment` {`d30`, `d33`, `dispositions`, `discordant`}, `quarantined_sources`, `feasibility`, `arm`, `chosen_recipe`, `ledger` {`events`, `head_sha256`}. The `queue` block repeats the eligibility counts with every reason a row is not eligible. `eligible.images` (Q) counts only what the cutter can cut: a near-duplicate group larger than M, or a unit that mixes verifier versions, is reported under `uncuttable` and is not supply, so D20 still collects and D22 does not build against it.
- **`stream/<sid>/ledger.jsonl`.** Every line has `seq` and `prev_sha256` (the sha256 of the file's bytes before it). Events: `init`, `prospective` (the sha256 of M, K, the truth policy, the thresholds, the recipe rule and the gate block, before any build), `code_change`, `arm`, `cut`, `build`, `commit` (`exp`, `segment`, `accepted`, `recipe` = `chosen`, `dispositions`, `counted`, `d30`, `d33`, `discordant`, `gate.gate3` sha256, `gate.v3_unavailable`), `return`, `quarantine`, `unquarantine`, `withdraw`, `rollback` (`to`, `suspect`, `utc`), `withdraw` (also with `orphan_increments` when cut lines never got their build line), `milestone` (`phase` build | build_failed | compare, `verdict`, `rollback_recommended`, `to_pool`, the dev score inputs by sha256), `bisect` (`rollback_utc`, `arms`, `decisions`), `feasibility`, `fork`, `release`. No test or ImageWeeds value is ever written to the ledger or the summary.
- **`stream/<sid>/consumed.jsonl`** (`key`, `increment`, `disposition`, `seq`, `counted`): step1_stream's `near_consumed` reads the keys.
- **What the stream reads from the other groups.** From C: `inc2.step1_stream.load_queue(Layout(root, inc_dir))`, used as the queue's one reader, with the row fields of C's note. From B: `inc2.gate3.decide_experiment(exp)` (commit and Stage C); `inc2.recipes` (`table`, `resolve_arm`, `stamp`, `step_cost`, `truth_every`, `stage_b_choice`, `ARMS`); `inc2.baseline build --exp E --manifest P_s --seeds 0,1,2,3,4 --arm A --role milestone --final-exams dev,imageweeds,test [--testing]` and `inc2.baseline secondary --exp E --weights W --source S`; `INC_DIR/capacity/capacity_v1.json`. From A: `inc2.guard.GuardV2.load(LOCK v2)` and `inc2.guard.image_hashes`, and LOCK v2's base_v2 sha256 and `provenance_sha256`. Without inc2.guard, inc2.gate3 or inc2.recipes the verbs that need them refuse.
- **`run_inc2_build.sh`** accepts `inc2.splits build|lock`, `inc2.baseline build`, `inc2.pilot4 build` and `inc2.stream init|build|milestone|fork|feasibility|bisect`, in the module forms `inc2.X`, `tools.inc2.X` and `weed_optimizer_framework.tools.inc2.X`. It exports `INC_JOB_SCRIPT=$REPO/weed_llm_benchmark/run_inc2_job.sh` before the builder and checks the outer against the nested copy of 36 files: the inc2 package, the pinned v1 modules the builders import, and the copy scan's `funnel` modules (`__init__`, `embed`, `leak`, `estimate`, `domain`, `ledger`), `semisup_labeler` and `funnel/domains/weed.json`, which `inc2.splits build` reads (the set `run_inc2_splits.sh` checks). Its lock and provenance name is `--exp`, else `stream_<sid>` for a stream verb, else `splits`. It advances every experiment the builder printed as built.

**Choices where the contract is silent or ambiguous, and why.**
1. **Draw units are transitive 6-bit groups of the eligible rows,** joined with step1_stream's near-duplicate group of each row. §3.5 checks P_s pairwise disjoint at 6 bits. Two 3-bit groups within 6 bits of each other, cut into two increments that are both accepted, would make every later commit refuse. A 6-bit group larger than M (a video or a greenhouse tray chains its frames) would never fit an increment, so it is split into its 3-bit near-duplicate groups (§3.4 [review]: the draw unit is the near-duplicate group); the cut never puts two of them into different increments of one cut, and the rest waits for a later cut (the increment meta records `split_near_dup_groups`). A unit whose rows carry two verifier versions waits until a re-judge unifies them, so that no increment mixes versions.
2. **The cut order.** Sources come first: the oldest source with ≥ M eligible images leads, then the other sources oldest first. Within a source, split-remainder capture groups come first, then capture groups by their oldest batch. Within a capture group, units follow `select.balanced_order` (`select.cluster_lists` over l1 and l2), seeded by `stable_int("<sid>/<segment>/<step>")`. `select.draw_parts([order], sizes, 1, M)` then takes exactly M, passing over a unit that would overshoot, which is how a capture group is split (S24). A remainder goes first within its own source. When species-cap swaps split another source's capture group, that remainder waits its turn.
   - The 60 % species cap, when ≥ 2 sources are eligible, is met by swapping the last-picked unit of the over-cap species for later units of the same total size. When no swap is possible, the increment is cut and the achieved share is recorded (`species_cap.met: false`), because refusing would stall supply on single-species sources.
   - A draw hit by GuardV2 (the training image and, for a masked row, the original, 8 variants each), by the image sha256 check, or by a flip or rotation within 3 bits of a quarantined image or within 6 bits of the pool, an increment in flight, a suspect one or an earlier increment of the same cut (a mirrored or rotated re-upload) excludes the whole unit, and the draw is redone. GuardV2 is loaded only while LOCK v2 still hashes as `init` recorded it.
3. **`--k K` is a maximum.** The build cuts up to K increments and builds with the k ≥ 1 it could cut, recording the refusal for the rest. With none it refuses, exit 1, and writes `cut_refusal.json` for D22.
4. **The in-segment chain follows the pinned gate. Protocol v3 decides at commit.** The pinned driver cannot call another gate, so `commit` reads `inc2.gate3.decide_experiment` (L-3):
   - P_s holds the increments whose v3 `commit_verdict` is ACCEPT, including a step the pinned gate rejected (recorded as `discordant`);
   - dispositions read the v3 failed guards;
   - a step gate3 marks unavailable keeps its pinned verdict and guards, and is listed in `gate.v3_unavailable`.
   - The truth verdict in the rule's "unless the truth arm says helps" is gate3's `truth3` commit verdict: the same per-species tolerance on both arms.
   - The data carries over and each segment's base is retrained cold, so a v3 acceptance the chain did not rehearse is still consistent.
5. **The disposition rule is applied as §3.5 [review] writes it.** Two cases the rule leaves open are given names: `hold` (a HOLD, returned once) and `truth_helps` (a REJECT on P_data alone whose truth arm says helps, returned once). On realloop_v1's recorded ledger the rule gives s01 data, s02 species, s03 species, s04 species, s05 flips and s06 species. The worked example in §3.5 lists s03 as data, but s03's recorded truth verdict is "helps", which the rule exempts from data.
6. **Returns are counted per image.** A second counted non-accept makes the image `neutral`, and its dHashes join the quarantine (reason neutral). A `recipe` return is not counted when D30 fires on the same segment: at least half of its REJECTs are recipe-only (computed at commit from that segment).
7. **A segment whose base is no longer the current pool** (a rollback happened while it ran) is committed without a pool. Its images return uncounted.
8. **M is computed from the locked base at init:** ⌈0.10 × |base_v2|⌉, from the base_v2 manifest that LOCK v2 records (P2's formula). The 763 of P2 predates L-5. Group A's note gives about 682 with L-5 [to verify on cluster]. M changes only by `fork` (L22).
9. **The truth policy is per segment.** The pinned driver's truth arm covers every step of an experiment or none. Truth is therefore on for segment ordinals 0, n, 2n, …, where n = `truth_every` of the step cost: `inc2.recipes.step_cost` for the arm at the current pool size, scaled by the measured-to-estimated ratio from the capacity decision. The first segment always has it. A `--truth-every` from the autopilot is used only when it is smaller (both recorded). Switching it off needs `--decided-by human:…` (P3). One segment is in flight at a time: a second build on the same base would only return stale at commit.
10. **Milestones.**
    - A milestone is compared, on dev only, with the last good milestone on the current pool's lineage. After a "hurts", the milestone before it is the reference.
    - Milestone 0 is the chosen arm's R0 baseline (`choose-arm`, else `init --milestone0`).
    - The chain incumbent's secondary scores come from `inc2.baseline secondary`, whose sbatch argv the milestone job submits, into `<sid>_mNNN/runs/secondary__incumbent`.
    - A milestone on a pool with the same sha256 as the last milestone's is refused.
    - Only the one-sided 5 v 5 permutation test (p ≤ 0.025 with a lower mean) recommends a rollback. A species-guard-only "hurts" is recorded as `species_only`.
11. **Rollback and bisect.**
    - Rollback within the envelope means the pool the last "hurts" milestone recommended, once per milestone. Any other rollback needs `human:…`.
    - A bisect is refused before anything is built when P_c has no good milestone: its arms could never be decided.
    - Bisect arms are baselines built directly on the pinned driver: 3 seeds, the arm's cold recipe, finals dev only, so no test is read outside a milestone (P10). Each is compared with P_c's milestone seeds by `gate.truth_detail`. helps → the images return to the queue uncounted; hurts → quarantined as data; neutral → quarantined as neutral.
12. **Stage C** (`<sid>_c001`, L28). D is exactly M target images of the holdout part of P_0, taken in whole sessions in a seeded order, with the last session split and recorded. The base is P_0 minus D; the recipes are the Stage B candidates; truth is on. It is read through gate3 into `m_feasible`, `species_only_reject` and the species for a prospective D33.
13. **Fork (L22).**
    - A fork that does not double M, or a third doubling, needs a person (X17).
    - The new stream adopts the current pool's bytes, every non-eligible key state, the quarantine, the chosen recipe and a person's releases. Its default name is `<sid>_fork<M>`: `<sid>_m<M>` would read as a milestone experiment when M has three digits.
    - The old stream builds nothing more.
14. **Holds.** step1_stream's events release holds. The stream's own `release` records a person's decision for `licence` and `join_conflict` (by keys or by source) and `funnel_F9` (also stream-wide). `h6_scan` is never released by hand: the copy scan is mandatory for every source that is not provenance-cleared (D-C, P9), and a copy that dHash misses would inflate the success measure.
15. **The ledger is the only source of truth.** `state.json`, `consumed.jsonl` (append-only) and `quarantine_dhash.json` are refolded from it by every operation, so an operation killed after its ledger line is completed by the next one. An edited `consumed.jsonl` or a broken chain refuses. The event types beyond §3.5's list are named above. The next writer also repairs what a killed operation left half done: cut lines without a build line are withdrawn at once (their images released, uncounted); a built segment, a milestone or Stage C whose `state.json` never appeared is withdrawn or recorded `build_failed` once its line is older than 2 × the lease (1 h). `dhash_cache.jsonl` is a cache: a torn line is skipped, and only the lease holder appends to it. The queue is read only through step1_stream's `load_queue`; without it the stream refuses rather than fold the queue a second way.
16. **L-4's rule is inc2.baseline's `capacity-verdict`.** `choose-arm` adopts its decision by sha256 instead of implementing the rule a second time. The Stage B rule is likewise `inc2.recipes.stage_b_choice`.
17. **research_only (§8).**
    - Each pool counts its research_only rows: base_v2 rows from LOCK v2's provenance file, and increment rows from the queue.
    - A segment's, a bisect arm's and Stage C's exp.json record the flag for their models, and each run's `run.json` copies it (`inc2.train`; "unknown" for a stream experiment that records none).
    - A milestone's flag in the stream report is the pool's. `inc2.baseline` sees only the base provenance.
18. **Test blindness.** The milestone's RESEARCH_LOG entry (`milestones/mNNN/research_log_entry.md`) and the stream report carry test, for people. They must stay out of the evidence allow-list. The stream writes the entry, not RESEARCH_LOG.md, which is the integrator's. The stream report is refreshed at every commit and milestone comparison.

**Not done.**
- `inc2/driver3.py` (R5, conditional).
- The per-increment L4 label audit and D31/D33, which are group F's diagnoses; the stream records the inputs they read.
- The autopilot's `child_exp` model tracks one experiment per build (`STREAM_CHILD_RE`), while one bisect build creates one arm per suspect increment. The ledger's `bisect` line maps every arm. A bisect killed between its ledger line and the last arm's driver init leaves an arm that never runs; X4 then stays with a person.
- M is the stream's (computed at init from the locked base, about 682 [to verify on cluster]); the autopilot must take it from `queue_summary.json` `M`. A Stage C `--m` or a fork `--m` that disagrees with it is refused, never silently used.
- `queue_summary.json` is rewritten only by the stream's writing verbs. Nothing runs `inc2.stream summary` after a Step 1 admit, so between stream operations the snapshot ships the Q of the last one (a new batch shows up only at the next stream write). Running `inc2.stream summary --stream SID` in the snapshot (login node; exit 3 = busy, keep the old file) closes this; that is the autopilot's file, not this group's.
- The autopilot's snapshot regenerates `INC_DIR/<exp>/report.json` with the pinned v1 `report.py` when it is older than `state.json`, which replaces the segment report written at commit. The Protocol v3 reading stays in the stream ledger and the stream report.
- Nothing has run on the cluster or on real data. M, the step costs, the truth cadence and every yield figure are [to verify on cluster].

**How it was verified.** Locally, with no network and no GPU: FakeBackend runs a synthetic executor that writes scores and Protocol v3 sidecars; planted PNGs have controlled dHashes; group A's real GuardV2, group B's real gate3 and recipes and group C's real `load_queue` are used.
- **`test_inc2_stream.py`: 156 checks.**
  - The rule on the realloop_v1 and pilot_v3 fixtures (S19).
  - The ledger chain.
  - The cut: exact M, determinism, units, holds, the cap, split remainders, S24 and its refusal; exact, 3-bit, 4–6-bit, hflip and rot90 copies of a dev image, and a masked row with a dev original, all refused at the cut.
  - Build validation by the pinned `validate_definition` and `check_definition_data`.
  - Five segments end to end: S5; HOLD and recipe returned once, then neutral; D30's uncounted return; a species-only pinned REJECT accepted by v3.
  - Milestones helps then hurts, rollback, bisect, the boundary check, Stage C, withdraw, fork, the governance refusals.
  - The lease; the CLI including `build_parser()`; the summary schema and test blindness.
  - `run_inc2_build.sh` run end to end on stubs; its drift list covers every module the builders import, computed by importing them.
  - The review fixes: the `held` shape, h6_scan never released by hand, a 6-bit chain longer than M split by 3-bit group and an unsplittable group reported as uncuttable, mirrored and rotated re-uploads refused at the cut, one segment in flight, bisect without a milestone, Stage C at the stream's M, a fork keeping releases, the D28/D31 cite, the LOCK pin, a torn cache line, the autopilot's L18 and L28 flags, the crash repairs, and the stream report refresh. Each of the 29 mutations of the new and the most important old checks (the disposition rule, GuardV2 at the cut, the quarantine, the chain, test blindness) makes the suite fail.
- **`test_inc2_stream_report.py`: 16 checks**, every number compared with the score files.
- **Neighbouring suites passing:** `test_inc2_common`, `test_inc2_gate3`, `test_inc2_baseline`, `test_inc2_train`, `test_inc_driver`, `test_inc_gate` and `test_funnel_domain_free`. Group F's `test_stream_ap_replay` S4 reads the L18 argv back through `build_parser()`.

### Build note (group D)

**What was built.** `tools/collect/`: `__init__` (paths, formats, errors, atomic writers, hash-chained ledgers, the intake lock, the output header), `__main__` (the CLI), `config`, `targets`, `names`, `licence`, `prefilter`, `classmap`, `normalize`, `transport`, `state`, `fetch`, `intake`, `plan`, `probe` (the probe and the state summary), and `providers/` (`zenodo`, `mendeley`, `huggingface`, `kaggle`, `github`, `roboflow`, `weedai` for the Weed-AI annotation index, `mediatum` for the record server and its FTP). Also `collect/domains/weed.json`, the pinned EPPO table `collect/domains/eppo_codes_v1.json`, and `run_inc_collect.sh`. Tests: `test_collect_{world,config,prefilter,classmap,licence,normalize,providers,fetch,intake,plan,cli,job_script,no_train,domain_free}.py`. `test_collect_world.py` is the shared helper, and it also checks itself when run. The fixture is `tests/fixtures/collect/provider_recording.json`, a new directory that §9 assigns to no one.

**Interfaces other groups rely on.**
- **CLI.** `python -m weed_optimizer_framework.tools.collect`, with these verbs:
  - `plan --config C --out PATH [--classes A,B] [--providers p,q] [--resolve-names]`;
  - `fetch --source ID [--max-bytes B] [--candidates PATH] [--out <INC_DIR>/intake/staging/]`, where `--out` is the lab hook's form and names INC_DIR;
  - `intake --source ID`;
  - `names --source ID [--out DIR]` (L26, lab only);
  - `summary`;
  - `probe`.

  Every verb takes `--config`, `--inc-dir`, `--result PATH` and `--testing`. Exit 0 means done, 2 refused or held, 1 a crash. One closing line, `[collect] <verb>: <JSON>`, goes to stdout, and `--result` writes the same JSON (`collect-result/1`). A refusal carries `status` (refused, held or closed), `code`, `failures` (`[{code, detail, action, risk}]`) and `risk: "R3"` when a person is asked. `run_inc_collect.sh` accepts plan, fetch, intake, summary and probe.
- **`candidates.json` (`collect-candidates/1`).** Each candidate has `source_id`, and `id` (the same value), plus:
  - provider, ref, version, title and `found_by` (provider, query, target);
  - `known_item` (bool), `known_item_id` and `known_item_name`;
  - `classes` and `class_status` (each declared class through the resolver), `target_classes` and `card`;
  - `licence` (`{id, class, research_only, text, evidence}`) and `annotation`;
  - bytes and images, `lab_group` with `lab_group_basis` (declared or presumed_derivative), `evaluation_lab`, `provenance_cleared` and `hold_until`;
  - `decision` (`{status, reasons}`), `estimate`, `rank` and `precheck` (`{ok, risk, action, failures}`);
  - the flat fields `licence_ok` (true under a person's licence override, with the override's id as `licence_id` and the override as `licence_override`), `expected_target_boxes`, `credentials_ok`, `image_level`, `annotation_type`, `copy_scan_done` and `found_by_search`.

  The file also holds the queries run, the provider errors, `recall` (S1b) and `released` (the sources whose licence hold a person's override released). `intake/plan_latest.json` points at the newest file by path and sha256.
- **Staging.** `intake/staging/<source>/fetch.json` (`collect-fetch/1`) and `blobs/<sha256>`.
- **`intake/sources.jsonl`.** Hash-chained events: candidate, held, closed, fetch_started, fetched, fetch_failed, intaken and released. Each carries codes, bytes, seconds, `su` (seconds/3600 inside Slurm) and the Slurm job id.
- **The intake batch, `intake/i<NNNN>_<source>/`.**
  - `manifest.jsonl` holds §3.2's fields and `session`, `batch`, `hold_until`, `holds`, `provenance_cleared`, `exhaustive_labels`, `licence_class`, `licence_override`, the box counts, `width`, `height`, `rel` and `intake_utc`.
  - Labels use ids 0–11 for the targets, 12 for the reject class and 13 for unmapped. Images are hard-linked.
  - `decisions.jsonl`, `guard.json` and `sources.json` (one source: `source_id`, the `licence` record with a person's `override`, the class map and its provenance).
  - `summary.json` is written last. It holds `yield` (boxes by id, by class and by source class), `source_leak {eval_share, base_share, fires}` for D28, `zero_yield` with its reasons, and `research_only` and `licence_override`.
  - `batches.jsonl` is hash-chained.
- **`intake/state.json` (`collect-state/1`, written by `summary`).** The folded state of every source, `recent_outcomes` (yield, zero or failed, in time order, for the three-in-a-row stop-loss), bytes against the caps, the ledger chain checks, the placement and the floors.
- **The probe's output.** `intake/placement.json`, written only inside Slurm (`placement_lab.json` otherwise). Per provider it records `reachable`, `status` (pass or fail), `placement` (cluster or lab) and `lab_only`.
- **The names layer.** `intake/names/names_cache.json` (`collect-names-cache/1`) and `names_<source>.json`.
- **The registry entry.** `annotation: intake_v1`, `status: intake`, `local_path` (the batch directory), `intake_batches`, and a `provenance` with the licence, its class, `research_only` and `licence_override`.
- **What the collector calls in group A's code.** `inc2.guard.GuardV2.load(lock)`, `.check_path`, `.add_intake`, `.index_record` and `.counts`. The copy scan counts as ready when `inc2.guard.load_calibration` accepts one of `copy_scan.ready_files`: the funnel's `leak_v1.json`, `splits/v2/leak/leak_calibration.json` or `intake/leak/leak_v1.json`. The collector writes none of these.
- **Config keys the autopilot reads.** `known_items` is a list. Each item's `id` is its source id, the one identifier used in candidates, the ledger, fetch and the registry; `name` is a short handle, and `decided_by` is "owner 2026-09-28 D-C". `placement.lab_only` is `["github"]`.

**Choices where the contract is silent or ambiguous, and why.**
1. **Evaluation labs are declared in the config.** They are LuLab (by the Kaggle account `yuzhenlu/`, the CottonWeed(Det|ID)NN names, and the author "Lu, Yuzhen") and NDSU (by "North Dakota State"). A declared evaluation-lab source is not fetched until a copy-scan calibration has passed (P9, R3). A presumed derivative is fetched, and every one of its rows is held `h6_scan`, as §7.2 [review] says. Presumed derivatives come from Roboflow, Kaggle, HF and the annotation index, which also re-hosts the reference dataset.
2. **Provenance clearance comes only from a known item marked `provenance_cleared`** (PAGS8 and the MFWD trays). Rows of every other source are held `h6_scan`, and `step1_stream` releases them.
3. **Never fetched:**
   - the trainer's `NEVER_TRAIN_SLUGS`, parsed with `ast` and never imported (a file that cannot be parsed refuses every fetch);
   - the config's `never_fetch` list: Zenodo 14861516, which is 3SeasonWeedDet10. Its 2022/2023 parts are base v2, and its 2021 part holds copies of the evaluation photographs;
   - titles matching CottonWeedDet12, cwd12, ImageWeeds or WeedSense. The annotation index holds an upload of CottonWeedDet12 and seven NDSU ImageWeeds uploads (read 2026-09-28).
4. **The class map.**
   - target and target_synonym → the target's id; taxon_resolved and role → 12;
   - target_related → 12 only when it is scientific or an override at species rank or below. A target's genus, a vernacular relative or a token match → 13;
   - every other status → 13;
   - the project alias table (`cwd12_species._ALIASES`, named by module attribute in the config) is the join, which is the funnel's `target` status.

   A name the caches lack blocks intake (NamesPending, lever L26) rather than being masked as 13. A missing name has not been shown to be unresolvable.
5. **Formats.** MFWD's `gt.csv` is read by a generic box-table normaliser; its column names are the known item's `format_options`. Rows that name images outside the fetched trays are counted once. Parquet shards (HF) are read when `pyarrow` is present.
6. **Capture group.** A group column, then a regex, then a video or a capture date in the file name, and otherwise the image alone. A singleton group constrains nothing, because the cutter splits by near-duplicate group.
7. **Duplicates.** Within a batch only byte-identical images are dropped (`exact_dup_intake`); near-duplicate frames stay and are grouped by `step1_stream`. Earlier batches feed GuardV2's intake index. The Step 1 seen index (`add_seen`) is not read at intake, because `step1_stream` applies it.
8. **Licences.**
   - Canonical ids with the policy in the config; a restriction word wins, as in `inc2.splits`.
   - When the record names no licence, `license_audit.detect_license` is asked for Kaggle, Roboflow, GitHub and HF.
   - `licence_overrides` is a person's entry, by exact source id: `{id, research_only, decided_by (human:<id>), decided_utc, reason}`; `config.validate` refuses any other shape, and `research_only` false unless the policy reads the `id` as permissive. It lets an unresolved licence through the pre-check, `plan` and intake, never a refused one. The intake rows keep the fetched `licence` and `licence_class` (`unresolved`) and carry `licence_override` and `research_only` (the licence's or the override's). `step1_stream` resolves the override at admission (`intake_licence_state`), so the queue rows are not held `licence`; they are research-only unless the override says false and its text names a known licence without a restriction.
9. **`funnel.fetch.fetch_spec` is not used for downloads.** It reads whole archives into memory, and the MFWD trays are 6.8 GB. Its endpoints and `_mendeley_listing` (the 206 rule) are used as a library, and downloads stream through `transport.Net` with the same checks. A fetch takes files in order while they fit under the per-source (50 GB), daily (50 GB), envelope (200 GB) and `--max-bytes` caps, and keeps 20 GB free on its disk. What is left over is recorded as `remaining`; those are shards, and the next fetch continues them.
10. **Each machine appends to its own `sources.jsonl`.** `plan` runs on the lab and fetch and intake run where they run. `summary --extra-ledger` folds several.
11. **Pins.**
    - The EPPO table is pinned by sha256 with `verified: false`; checking it against gd.eppo.int is §10 item 5.
    - The PAGS8 card table maps its eight growth-stage names to *Amaranthus palmeri* and is pinned `pending the owner's acceptance`.
    - `floor_gb`/`floor_su` stay null, and a bare number is refused as a placeholder.

**Read live on 2026-09-28 from the build machine [to verify on the lab/cluster].**
- **PAGS8** is upload `5c78d067-8750-4803-9cbe-57df8fae55e4`: 614 images and 1,940 boxes in 8 categories, CC BY 4.0, a 1.93 GB zip. The paper's 1,228 images and 5,026 boxes are not all in this upload.
- **MFWD.** The FTP server (login `m1717366`) has 25 species directories. POROL has 21 tray archives totalling 6,781,147,584 bytes, against the contract's estimated 5.4 GB. `gt.csv` is 19.3 MB, and a `checksums.sha512` list is published.
- **Uploads on the index outside the D-C list:**
  - "Palmer amaranth - Cornell" (995 images, 1,405 boxes, CC BY);
  - "Multiple weed species detection" (905 images; Palmer amaranth and waterhemp boxes, CC BY).
- **Recall.** The recorded narrow run (three deficit classes) finds PAGS8 by search. The Mendeley and HF searches did not find MH-Weed16 or MFWD, and without credentials Kaggle and Roboflow were not searched live.

**Not done.**
- **Kaggle and Roboflow.** They are tested only on synthetic answers in their documented shapes; the Roboflow export call is `<ws>/<proj>/<version>/<format>`. The Kaggle token rotation (§10 item 6) is a person's action.
- **Group F interop.** `stream._check_recall` counts as found the rows without `known_item`. The recall to read is `candidates.json` `recall`: a known item is found when a search result matched its match rules. `stream._never_train` reads a module attribute; the collector parses the trainer's source instead.
- **Nothing has run on the lab or the cluster.**

**How it was verified.** Locally, with the socket blocked, fake networks and FTP, the GBIF recording for the taxonomy cache, and group A's real `inc2.guard` where it is used:
- 345 checks in 14 files:
  - classmap 26, cli 21, config 22, domain_free 10, fetch 23, intake 33, job_script 23;
  - licence 14, no_train 9, normalize 39, plan 26, prefilter 47, providers 45, world 7.
- The real `verify._skip_reason` and the real `mega_trainer._merge_datasets` skip an `intake_v1` source.
- These neighbouring suites pass unchanged: `test_funnel_{domain_free,fetch,taxonomy,names,job_script,weed_config,cli,domain}`, `test_inc_ap_{governance,fixtures,replay,evidence}`, `test_domain_config` and `test_species_docs`.

#### Review of group D (2026-09-28): defects found and fixed

Each item: the defect, the fix, and the check that covers it. Where an item changes a statement of the build note above, the statement here replaces it.

1. **Provenance clearance by a title match (a leak path).** A Roboflow or Kaggle re-upload whose title matched a cleared known item's match rule (for example "MFWD ..." or "Palmer Amaranth Growth Stage") took that item's `provenance_cleared`, so its rows skipped the `h6_scan` hold, and augmented evaluation copies that dHash misses could reach the queue. A stale `provenance_cleared` in a candidate record was also carried forward. **Fix:** clearance comes only from `prefilter.cleared_by_config`: the item's own record (its source id, or its provider and ref), marked cleared, outside every evaluation lab. `decide` ignores the incoming value, and intake derives it again from the config instead of trusting `fetch.json`. A record that only matches an item's rules (`known_item_role: match`) takes the item's lab only when that lab is an evaluation lab, and it is not kept as the known item. **Checks:** `test_collect_prefilter` (clearance), `test_collect_intake` (a fetch record claiming clearance).
2. **Stale candidate fields decided the fetch.** `fetch` merged the plan's candidate record into the fresh describe, so the plan's licence verdict, holds and lab group won. A licence that had changed to non-commercial since the plan would have been recorded as permissive. **Fix:** `fetch._merge` drops every derived field (`prefilter.DERIVED`) before `decide` runs again. `plan_latest.json` is checked against the sha256 it records (a changed file raises StaleInput). A pointer to a file this machine lacks is named in the `unknown_source` refusal. **Check:** `test_collect_fetch` (stale candidate, pointer).
3. **A restriction word lost to a permissive licence name.** "CC BY 4.0 (research use only)", "ODbL, research only" and "CC0 personal use only" were permissive. **Fix:** `licence.canonical` applies `inc2.splits.research_only`'s restriction rule to every family. **Check:** `test_collect_licence`, including agreement with `inc2.splits.research_only`.
4. **An unreadable registry was wiped.** `registry_lock.update_registry` reads an unparseable file as an empty registry, and intake's write would have replaced about 50 MB of sources with one entry. **Fix:** `_register` reads the registry strictly first, and refuses when it exists but does not parse, or when the read under the lock holds fewer sources. Nothing is committed in that case. **Check:** `test_collect_intake` (the registry guard, and `_register` called directly).
5. **Placed boxes of an unlisted class were dropped.** These are a COCO category id missing from the categories, a VOC object or VIA region without a name, or an id the class list lacks. Dropping them left an unlabelled plant in a kept image. **Fix:** they keep their place with `normalize.NO_CLASS`, and intake gives them the unmapped id 13, which per-box admission masks (§3.2). The rows record `unlisted_class_boxes`. **Checks:** `test_collect_normalize`, `test_collect_intake`.
6. **Job log directory (cluster).** `--output` pointed at `inc/intake/logs/`, which nothing creates before the first job. Slurm fails a job whose log directory is missing, so the first probe would have failed without a log. **Fix:** the log goes to `inc/logs/`, which `stream_remote.stream_submit` creates before every sbatch. **Check:** `test_collect_job_script`.
7. **One writer across nodes.** `intake_lock` relied on flock alone and read every flock error as "held". On a Lustre mount without cluster-wide flock, two collect jobs on different nodes could both write `intake/` (the same batch name, and an interleaved hash chain). **Fix:** flock plus an owner file created with `O_EXCL`, as in `step1_stream`'s writer lock. ENOLCK and ENOSYS are tolerated. A dead writer's owner file (its pid gone on this host, its Slurm job ended per `squeue`, or older than 9 h) is taken over. **Check:** `test_collect_fetch` (writer lock across nodes).
8. **A broken ledger was appended to.** **Fix:** `append_chained` refuses (StaleInput) to append to `sources.jsonl` or `batches.jsonl` once their hash chain is broken. **Check:** `test_collect_fetch`.
9. **Decision L-5 could be undone by a re-upload.** GuardV2 does not index base B's cwp10 and vanpe images, and `inc2.train` drops them by exact sha256 only, so a re-encoded fork would come back as new data. **Fix:** intake reads `splits/v2/l5_excluded.jsonl`, checked against LOCK v2's `l5_excluded_sha256`. An image whose bytes, or any of whose eight variants within 6 bits, copy an L-5 image is refused as `l5_copy` and counted with base copies for D28. A production LOCK without the record refuses intake. **Check:** `test_collect_intake` (real GuardV2).
10. **3SeasonWeedDet10 was blocked only by one Zenodo record id.** **Fix:** `never_fetch_title_regex` covers the name in any spelling and AgML's `three_season` copy. **Check:** `test_collect_prefilter`.
11. **Rules re-applied at intake.** Kaggle and repository archives declare no class list before download, so the legacy-label copy rule (≥ 4 legacy labels) now also runs on the class list the files declare, and a hit closes the source. The never-fetch rules also run again on `fetch.json`. An archive member that would leave the work directory closes the source. An unreadable archive holds it (R3). All three are recorded in `sources.jsonl`. **Check:** `test_collect_intake`, including a zip "../" member, an absolute tar path, and a tar symlink that is never written.
12. **P6 supersession closed unrelated sources.** Equal titles alone ("weed detection" is common on Roboflow), a title-only match of a known item, or a copy the prefilter rejected could close a fetchable source. **Fix:** copies are a known item's own record and its listed copies, or equal titles with the same image count, among kept candidates only. **Check:** `test_collect_prefilter`.
13. **Signed storage links** (Roboflow export redirects) could reach `fetch.json` with their signature. **Fix:** `transport.redact` also redacts signature and credential parameters. **Check:** `test_collect_providers`.
14. **Fields the autopilot reads.**
    - `summary.json` adds `images` (the images the guard checked), `guard` (refusals per reason) and `decisions.by_reason`, beside `source_leak`. `test_collect_intake` runs the autopilot's own D28 on a collector summary.
    - In `candidates.json`:
      - `licence_ok` is tri-state: True usable, False refused, None unresolved (a person decides; a False would drop the source without asking anyone);
      - `licence_id`, `licence_class` and `names_unresolved` are added;
      - `known_item` is true only for a known item that no search found, so a reader counting the other rows as found gets the collector's recall; `known_item_id` and `known_item_role` name the item for every row that is one, copies one or matches one.
15. **Keys** are unique across intake batches: an earlier batch's key gets the fetch sha as a suffix.

**How it was verified.** Locally, with the network blocked:
- 404 checks in 14 files, 0 failures:
  - classmap 26, cli 21, config 22, domain_free 10, fetch 33, intake 59, job_script 24;
  - licence 16, no_train 9, normalize 42, plan 29, prefilter 60, providers 46, world 7.
- One intake test runs group A's real `inc2.guard.GuardV2` over a synthetic LOCK v2.
- **Mutation check.** 48 single mutations of the guards above and of the builder's guards were applied one at a time: never-train, title rules, the copy-scan hold, placement, the licence gate, the staging sha, holds, research_only, the checksum and the byte cap, zip slip, the job script's verbs and log path. The aimed test failed for each. Each file was restored and its sha256 checked.
- `test_inc2_{common,guard,step1_stream,mask}`, the funnel suites and `test_inc_ap_*` pass.
- `test_stream_ap_units` and `test_stream_ap_replay` were failing during this review because group F's files were being edited at the same time. Their test world writes its own scripts and summaries and runs no collector code.

**Open items.**
- **Discovered sources on the cluster.** A cluster `fetch` of a discovered source (not a known item) needs the lab's `candidates.json`, and only staging is synced. The fetch refuses `unknown_source` and names the missing pointer. Group F should pass `--candidates` or sync the file.
- **Held sources are never retried.** Group F's L16 proposal skips a source whose status is `held`, so a `copy_scan_pending`, credential or licence hold is never retried after its condition clears.
- **EXIF-rotated images.** An image with EXIF rotation in a pixel-coordinate format whose annotation file gives no size is normalised against the stored size [to verify on real sources].
- **Byte caps per machine.** Each machine's `sources.jsonl` carries its own daily and envelope byte counts, so the lab and the cluster together can reach twice the cap. Group F's `gb_per_day` is the cross-machine limit.
- **Later shards re-read earlier files.** A later shard of a source reads every earlier file again; earlier images are refused as `near_dup_intake`.
- **Pending owner acceptance.** The EPPO table and the PAGS8 card table are still pending the owner's acceptance (§10 item 5).

### Build note (group F)

**What was built.** F1 and F2 in one delivery. New: `inc_autopilot/stream.py` (the lane ticker `StreamRun`, `configure_stream`, the detached lab runner, the CLI `enable | status | complete | lab-run | lab-sync`), `diagnose_stream.py`, `levers_stream.py`, `stream_remote.py`, `stream_levers.json`, `stream_thresholds.json`, `stream_domains/weed.json`. Edited: `campaign.py`, `executor.py`, `budget.py`, `remote.py`, `evidence.py`, `model.py`, `brain/policy_actions.json` (22 rows added, none changed), `brain/approvals.py`, `brain/su_rates.json`, `round_scheduler.py`. Tests: `test_stream_ap_{world,replay,units,mutations,identity}.py` and `test_round_scheduler_stream_guard.py`. Docs: the stream-mode section of docs/INC_AUTOPILOT.md.

**Interfaces other groups rely on.**
- **The commands the platform runs** (`stream_levers.json`, rendered token for token by `executor.ARGV_FORMS`). `test_stream_ap_units.py` reads each back with the owning group's own parser where one is exposed: `inc2.stream.build_parser()`, `inc2.step1_stream.build_parser()` and `collect.__main__.build_parser()`. `inc2.baseline` and `inc2.pilot4` build their parsers inside `main()`, so their argv are checked against the cluster grammar (`stream_remote.parse_submit`) instead.
  - group E, `python -m weed_optimizer_framework.tools.inc2.stream`: `build --stream SID --k K --exp SID_sNNN` (the summary's `next_segment`), `milestone --stream SID`, `fork --stream SID --m 2M`, `feasibility --stream SID --holdout tsw22 --m M` (the summary's M), `bisect --stream SID --from P_c`, `init --stream SID --stage-b R0[,X]` (Stage A's `segment1_recipes`), all as `sbatch -p GPU-shared run_inc2_build.sh inc2.stream VERB ...`; and on the login node `commit --exp`, `compare --exp SID_[mcb]NNN`, `choose-arm --stream`, `rollback --stream --to`, `quarantine --source --cite D28|D31`, `release --stream --hold funnel_F9`. The platform never passes `--arch`, `--imgsz`, `--truth-every` or `--recipes` to a build.
  - group B: `inc2.baseline build --exp E (--manifest M | --union M1,M2,M3) --seeds S --arm n640|s640|m640 --role b_v2|capacity|canary|union` (build job); `inc2.baseline canary-verdict --exp canary_v2`, `inc2.baseline capacity-verdict`, `inc2.pilot4 verdict --exp pilot_v4` (login node); `inc2.pilot4 build --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b`.
  - group C: `sbatch -p GPU-shared run_inc2_stream.sh admit --intake BATCH | bootstrap | knowntruth | backfill | scan-holds --hold h6_scan`.
  - group D: `sbatch -p GPU-shared run_inc_collect.sh fetch --source ID --max-bytes B [--candidates INC_DIR/intake/candidates_sync/ID.json] | intake --source ID | probe`; on the lab, `collect plan --config C --classes A,B --out P`, `names --source ID --out DIR`, `fetch --source ID --max-bytes B --out <lab INC_DIR>/intake/staging/`.
  - Job names: `inc_build_<child experiment>`, `inc_build_stream_init_<sid>`, `inc_stream_<kind>_<verb>_<source|batch|hold>`. The environment carries `INCAP_PARENT_EXP`, `INCAP_CHILD_EXP`, `INCAP_TRIGGER`, `INCAP_APPROVAL_ID`, `INCAP_DECIDED_BY`, `INCAP_REQUESTED_UTC`; a login-node verb approved by a person gets `INCAP_DECIDED_BY=human:...`, which inc2.stream takes for `--decided-by`.
- **What the snapshot reads** (aggregates, dev only):
  - `stream/<sid>/queue_summary.json`: `M`, `pool`, `eligible`, `held`, `consumed_last`, `queue.held_past_deadline`, `cut.probe`, `next_segment`, `uncommitted_done`, `milestones` (`records`, `in_flight`, `accepted_since`, `segments_since`, `days_since_first_accepted`, `due_reasons`, `next`), `boundary_check` (`new_mean`, `old_mean`, `old_sd`), `rollback_pending`, `stage_b`, `chosen_recipe`, `forked_to`.
  - `stream/<sid>/ledger.jsonl`, chain checked on the cluster, shipped without `prev_sha256`: `init`, `arm`, `cut` (`increment`, `sources`), `build` (`exp`), `commit` (`exp`, `chosen`, `steps[chosen][]` with `increment`, `v3` {`verdict`, `guards`, `p_data`, `species_failed`, `cand_mean`, `null_mean`, `null_sd`}, `truth`, `disposition`; `stale_base`), `milestone` (phase `build` with `exp` and `pool`; phase `compare` with `verdict`, `perm_p`, `new_mean_dev`, `old_mean_dev`, `species_failed`), `feasibility` (phase `build`; phase `read` with `result` {`m_feasible`, `species_only_reject`, `d33_prospective`, `per_recipe`}), `rollback` (`to`, `suspect`, `utc`), `bisect` (phase `build` with `rollback_utc` and `arms`; phase `decide` with `decisions`).
  - `step1_stream/status.json`: `one_time`, `versions` (`verifier`, `splits_lock_sha256`), `per_source` (`images_seen`, `near_eval_embed`, `target_boxes_admitted`), `knowntruth` (`matched_verified`, `verified_correct` per batch), `refit_triggers.species_unknown_share`, `holds_past_deadline`.
  - group D: `intake/<batch>/summary.json` (`source`, `batch`, `rows`, `yield`, `source_leak` {`eval_share`, `base_share`}, `zero_yield`, `decisions.by_reason`); `intake/sources.jsonl`, folded on the cluster by `collect.state.fold`; `intake/placement.json` (`providers.<p>.placement`, `lab_only`); on the lab, `candidates.json` (collect-candidates/1: `source_id`, `licence` {`id`, `class`}, `licence_ok`, `target_classes`, `bytes`, `expected_target_boxes`, `lab_group`, `evaluation_lab`, `copy_scan_done`, `credentials_ok`, `annotation`, `names_pending`, `decision.status`, `precheck.failures` [`code`, `action`, `risk`], `recall`).
  - group B: `capacity/capacity_v1.json` (`chosen_arm`, `chosen_exp`, `qualifying`, `truth_every`), `<exp>/canary.json` (`passed`), `pilot_v4/stage_a.json` (`status`, `segment1_recipes`, `survivors`, `best_survivor`), all found at INC_DIR's top level. `capacity_v1_report.*` is not on the allow-list.
  - The last line matching `ERROR|FATAL|refus|StreamError|CollectError` in a failed `inc_stream_*` job's log (INC_DIR/logs or INC_DIR/intake/logs). A build's log is never read.
- **Other names.** `round_scheduler.old_path_refusal(domain, step, auto_sync=False)` is the check for the dashboard's harvest route. `executor.STREAM_REPLAY_CASES` (S1–S28, `stream_prospective`, `stream_r0`) and `stream_mutations` are recorded by `executor record-replay`, and `GOVERNANCE_FILES` adds `collect/domains/weed.json` and the two stream test scripts.

**Choices where the contract is silent or ambiguous, and why.**
1. **The stream menu, thresholds and domain facts live in their own files.** `diagnose.rules_version()` hashes `thresholds.json`, `levers.json` and `levers.py` whole, and the brain's menu lists `levers.json`'s levers. Stream rows there would re-diagnose the running experiment campaigns and change their digests. `levers.json` and `thresholds.json` are therefore not edited, although §9 lists them.
2. **The stream verbs are a module of their own** (`stream_remote.py`, dispatched by `remote.py`), not rows of `SUBMIT_FORMS`. The job scripts have their own grammars and job names, and the verbs must refuse what those grammars refuse; `SUBMIT_FORMS` stays byte-identical for experiment mode.
3. **`REPLAY_REQUIRED` is not extended.** The stream cases are a list of their own. A failing stream case still fails the whole replay record, so it blocks every campaign's envelope. `test_inc_ap_governance` fixes `REPLAY_REQUIRED` for experiment mode.
4. **`round_scheduler` is guarded by ownership** (a `campaigns.<name>` block with `mode: stream` and the domain, enabled or paused), not by edits to its step tables. `_advance` refuses the domain's collect, filter and train steps, records the refusal and pauses the domain. The step tables are read by `test_policy`, `test_round_templates` and the round ledger, which pass unchanged.
5. **Levers beyond §6.3, and why.**
   - LP is the network probe.
   - L16L, L16R, L16I and L16S split L16 into its lab fetch, its person-decided fetch, its intake and its sync.
   - L23B holds L23's baselines, LV their recorded verdicts, and LH a person's funnel_F9 release.
   - LI (the stream's creation, a build job) and LA (`choose-arm`) are R2's prerequisites, which §10 gives to "the platform" without a lever.
   - LC (`compare --exp`) is how inc2.stream decides a finished milestone, Stage C chain or bisect arm.
   - LV, LA and LC are R1: deterministic readers of recorded dev scores, like L19. LI is R3 and envelope-eligible. Its inputs are pre-registered: M by P2's formula at init, and the Stage B arms from Stage A's recorded verdict.
6. **Each rule has one implementation.** The disposition of the stream's increments, the Stage B choice, the milestone test, the capacity switch, Stage A's survival and Stage C's reading are the other groups' code; the autopilot reads what they record. Where §6.4 pre-registers a value, the autopilot applies it to the recorded numbers as well:
   - D24's triggers (4 accepted, 3 segments, 30 days) are applied to the summary's counts;
   - D25 proposes L21 only when `rollback_pending` names the pool and the recorded comparison meets p ≤ 0.025 with a lower mean; a disagreement is escalated to a person;
   - the boundary check's 2 sd is applied to the summary's means and sd, not to its `fires` flag;
   - the verifier-refit bound is recomputed from step1_stream's counts.

   The guard rule of §3.5 is applied in `diagnose_stream.disposition` only to the pinned prior (realloop_v1 before Stage A is READY) and to S19's recorded pilot_v3.
7. **D30 and D33 count REJECTs only, as inc2.stream's `d30` and `d33` do.** A HOLD is not a REJECT. On realloop_v1 the result does not change: D33 names PricklySida in 5 of 6 REJECTs. §3.5's worked example lists s03 as data; the rule gives species, because s03's truth verdict is "helps" (group E's note, choice 5).
8. **D31 blames a source only for a single-source increment.** An increment drawn from several sources blames none of them alone, so two blames are needed for L24.
9. **L18 names its segment and nothing else.** `--exp` is the summary's `next_segment`, so a build that would make another experiment refuses. The arm comes from `choose-arm` (L-4), the truth cadence is inc2.stream's per-segment rule, and the Stage B arms are fixed at init. The platform computes the cadence only to price the proposal.
10. **Stale evidence after a login-node verb.** A verb's effect appears in the next snapshot. An item done since the last snapshot is not proposed again on older evidence (`done_keys`). inc2.stream's exit 3 (another writer holds `stream.lease`) is a transient, never a failed step.
11. **Pin refusals.** step1_stream's pin check exits 2 with its reason in the job log. The snapshot ships the last refusal line of a failed data job's log, and an admit refused on the verifier, the LOCK or a pin is not retried (card X11). Other failures count toward the lane's stop-loss.
12. **A discovered source fetched on the cluster.** The cluster has no copy of the lab's candidates.json (group D's open item). The lab hook first writes the source's own record to `INC_DIR/intake/candidates_sync/<source>.json` and pushes it with a sha256 check (L16S, candidate 1). The fetch then passes `--candidates` with that path. A known item of the collect config needs no record.
13. **Holds of the collector.** A source the collector's ledger holds is held here too, marked `held_by: collector`. It becomes collectable again when the collector's fold releases it (group D's open item). Holds of the platform's own are released only by a person: a D28 leak and a pin refusal.
14. **The collector's pre-check is honoured, and the platform's own is kept.** A close or refuse action refuses the source, an R3 hold files an item for a person (L16R), another hold waits. The never-train list is parsed from the trainer's source (ast), never imported, as the collector does.
15. **M and pricing.** The stream's M is the summary's. `stream_domains/weed.json` carries M = 682 and N = 6,813 (L-5, group A's note) [to verify on cluster] only to price proposals before the stream exists. The capacity arms' cost factors (1, 2, 3; est.) price segments until the capacity decision measures the rate.
16. **A failed canary holds TRAIN (DCAN)** and escalates to a person, since §10's R0 acceptance is "canary within 1 sd". B0 ∪ tsw is built after the required baselines and blocks nothing.
17. **Dev scores are no longer requested by the snapshot.** D25 reads inc2.stream's recorded comparison instead of recomputing the permutation test. `stream-snapshot --dev-scores` remains available.
18. **A STOP lane** takes L24 (quarantine) and L7 (unblock), so a busy DATA lane never delays a quarantine.
19. **The L4 label audit on a stream segment** reuses the existing v1 lever and `run_inc_audit.sh` on the segment's `exp.json`. Whether that script accepts a v2 segment's manifests is [to verify on cluster].

**Not done.**
- **Two wirings in files group F does not own.** The dashboard's harvest route does not call `old_path_refusal(..., auto_sync=True)` yet. The brain (tier2) digest covers experiment campaigns only; no brain may request a stream action, since the policy rows allow only `round-scheduler` and `human`.
- **Cluster checks.** The `projects` and `my_quotas` output layouts, the lab-to-cluster `rsync` hosts (`CLUSTER_SSH`, the data node), and the L4 audit on a v2 segment are [to verify on cluster].
- **No end-to-end run with real code on real data.** Nothing has run on the lab or the cluster, and no run joins the real `inc2.stream`, `inc2.step1_stream` and `collect` to this ticker on real data. The world fakes those verbs and writes their formats. The argv checks use their real parsers and the commit fake uses the real `inc2.stream.dispose`.
- **No `tests/fixtures/inc_replay/realloop_v1/`.** The funnel's pinned copy is reused by sha256 instead.
- **`test_brain_api`'s shadow-review check fails on `brain/supervision_health.py`,** as it did at HEAD before this build. That file is not group F's.

**How it was verified.** Locally, with no network, GPU or ssh:
- `test_stream_ap_replay.py`: 31 cases, 195 checks. S1–S28, the prospective record, and `stream_r0`, in which the platform drives R0 through R2's first segment by itself: 5 baselines, 3 verdicts, Stage A, init, choose-arm, Stage C and its reading, then L18.
- `test_stream_ap_units.py`: 268 checks, including every stream argv read back by group C's, D's and E's own parsers, and the other groups' formats as their code writes them.
- `test_stream_ap_mutations.py`: 18 of 18 mutants killed.
- `test_stream_ap_identity.py`: experiment mode byte-identical to git HEAD (16 checks).
- `test_round_scheduler_stream_guard.py`: 13 checks.
- These pass unchanged: `test_inc_ap_{brain,campaign,dashboard,diagnose,e2e,evidence,fixtures,governance,levers,remote,replay}`, `test_funnel_ap_{units,replay,mutations}`, `test_funnel_domain_free`, `test_policy`, `test_policy_adversarial`, `test_policy_gate_paths`, `test_round_templates`, `test_round_ledger`, `test_su_ledger`, `test_approvals`, `test_review_authority`, `test_scheduler_{bundle,archive,review,state}`, `test_brain_inventory_adapter`, `test_funnel_pipeline`, `test_funnel_fetch`, `test_species_robo`, and group D's `test_collect_{intake,plan,job_script}`, which read group F's code.

#### Review of group F (2026-09-28): defects found and fixed

Each item gives the defect, the fix and the check that covers it. docs/INC_AUTOPILOT.md (h), "Corrections after the adversarial review", has the full list. Every fix was proved by switching it off in place: the named check failed, then the file was restored and its sha256 checked (29 mutations, all killed).

1. **The DATA lane deadlocked after shadow mode.** A gated L16, L17 or L24 filed while `data_autonomy` was off stayed filed after a person set it on. **Fix:** such an item is re-checked against its gates every tick once `data_autonomy` is on. **Check:** S16.
2. **Items a person decides held the lane or were lost.**
   - LH (a funnel_F9 release at its deadline) held the DATA lane until a person acted.
   - An approved L16R was never run.
   - An approved item a person ran from the INC page left its lane waiting.

   **Fix:** these items are filed parked, outside their lane, and adopted into the lane once approved. The lane follows a run made elsewhere. **Checks:** S1, S26, units.
3. **A person could release a stop-loss lane hold only through a pause.** **Fix:** `stream enable` after the hold releases it. **Check:** S20.
4. **D21 judged a source's yield before its admission.** With the floors set, this closed every source at its first fetch. **Fix:** D21 judges only admitted sources with an observed yield. **Check:** S2.
5. **Sharded sources were never continued.** Nothing set `partial`. **Fix:** the snapshot carries the collector's last-fetch completeness, and a lab fetch's closing line gives the same. A source whose attempts are used up with shards left goes to a person, not to a campaign pause. **Checks:** S2, S15, units.
6. **Names were never pushed to the cluster.** A names-pending source went to a person instead of to L26. The names layer L26 writes on the lab never reached the cluster, whose collector jobs read it offline. **Fix:** the pre-check sends such a source to L26, and `L16S --names` pushes the layer; every staging push carries it. **Checks:** S22, units.
7. **D33's two-consecutive-segments hold** counted the pinned prior and segments that were not consecutive. **Fix:** the run is counted from the stream ledger's commits. **Check:** S6.
8. **D27 subtracted spent SU from a balance that already excludes it.** **Fix:** D27 subtracts committed SU only. **Check:** S10.
9. **The domain envelope was never checked against the round history (6.6 review).** **Fix:** D10S pauses with card X14 when the domain ledger, every step counted, exhausts `su_envelope`. **Check:** S10.
10. **D26 held segments on the longest recipe of the domain.** **Fix:** it reads the recipes the stream runs. The first cut also waits until the evidence carries the chosen arm. **Check:** S11.
11. **A D22 refusal held TRAIN for ever.** **Fix:** the refusal is superseded when the cutter's probe fills M again. **Check:** S24.
12. **Leak path.** Rows of a source D28 flagged stayed cuttable while its quarantine waited for a person, or waited behind an R3 build that took the tick's ssh. **Fix:** D28 holds TRAIN until the source is quarantined or a person keeps it, and STOP items go before R3 items. **Checks:** S25, units.
13. **Ordering.**
    - The next segment could be cut before the evidence showed the last commit.
    - A fork was adopted one snapshot late, and in between D8S proposed L22 again. That duplicate was then failed as a job that "ended unknown".
    - After a fork, LI was proposed for the new stream on the old stream's evidence.
    - The prospective record was not re-written for the forked stream version.

    **Fix:** all four are corrected (INC_AUTOPILOT.md (h)). **Checks:** `stream_prospective` (D8S → L22 → adoption → LA → the new stream's first L18), units.
14. **Accounting.** A stream build's own job was never settled from sacct, and a fork's estimate was never released. **Fix:** both are now settled. **Checks:** S5, units.
15. **Smaller defects.**
    - Syncs counted against L16's six jobs a day.
    - The lab sync read whole blobs into memory to hash them.
    - A person could not set a new envelope end or `collect_gb_daily` from the CLI.

    **Checks:** S15, units.

**Corrections to the build note above.**
- `brain/policy_actions.json` gains 23 rows, not 22. `inc_stream_sync` now also takes `names: 1`.
- `test_stream_ap_replay.py` now runs 31 cases and 228 checks. `test_stream_ap_units.py` runs 281 checks.

**Open items.**
- **Contract §8 "No mega_trainer training".** Two of its measures are not done, because existing suites that must pass unchanged assert the opposite: removing the weed train and filter fallback from `round_scheduler._WEED_STEPS`, and making `policy_actions.round_train`/`round_filter` `allowed_tiers: ["human"]`. `test_round_templates.py` requires the round scheduler's live train step to pass the policy gate. The ownership guard (`round_scheduler.old_path_refusal`) closes the old paths only while a stream campaign names the domain. Deciding between the contract and those suites is the integrator's call, or the owner's.
- **Unused thresholds.** `stream_thresholds.json` declares `cadence.shadow_days` (3) and `cadence.discover_min_hours`, but no code enforces them. Setting `data_autonomy` on after ≥ 3 days of shadow mode remains a person's check (§10 item 4).
- **Not F's files.** The dashboard's harvest route still does not call `old_path_refusal(..., auto_sync=True)`, and the brain digest still covers experiment campaigns only (from the build note).
- **To verify on the cluster.** The `projects` and `my_quotas` layouts. The names layer's path is `intake/names/names_cache.json` in the collect config (`taxonomy.names_layer`), and the stream pushes that path; a config change there must change `stream.NAMES_DIR` too.

---

## Verification (integration of groups A–F, 2026-09-28)

**Scope.** Everything below ran locally, on synthetic worlds, with no network, no GPU, no Slurm and no real data. Nothing has run on the lab or the cluster. No pinned INC module, `realloop.py`, `pilot.py`, `funnel/**` or `run_inc_build.sh` changed (`git status` is clean for them).

### V.1 Test suites

Each script was run as `cd weed_llm_benchmark && python3 tests/<file>`. Every one exits 0: **88 scripts, 7,680 checks, 0 failures.** Four checks skip, all of them outside the stream and all of them skipped before this integration:
- `test_inc_ap_replay`: R2 and R4b (the Step 1 fixture and the committed prospective record are not pulled);
- `test_inc_ap_brain`: the relevance fixture;
- `test_inc_ap_remote`: the step1 replay fixture.

| Group | Scripts | Checks |
|---|---|---|
| A, splits v2 | `test_inc2_common` 43, `test_inc2_guard` 68, `test_inc2_splits` 118 | 229 |
| B, executor, recipes, baselines, Stage A, gate3 | `test_inc2_train` 80, `test_inc2_baseline` 70, `test_inc2_pilot4` 28, `test_inc2_gate3` 47 | 225 |
| C, incremental Step 1 | `test_inc2_step1_stream` 167, `test_inc2_mask` 27 | 194 |
| D, collector | `test_collect_*`, 14 scripts: classmap 26, cli 21, config 22, domain_free 10, fetch 33, intake 59, job_script 24, licence 16, no_train 9, normalize 42, plan 29, prefilter 60, providers 46, world 7 | 404 |
| E, stream builder and reports | `test_inc2_stream` 156, `test_inc2_stream_report` 16 | 172 |
| F, autopilot stream mode | `test_stream_ap_replay` 228 (31 of 31 cases), `test_stream_ap_units` 281, `test_stream_ap_mutations` 74 (18 of 18 mutants killed), `test_stream_ap_identity` 16, `test_round_scheduler_stream_guard` 13, `test_stream_ap_world` (a helper, run as a smoke tick) | 612 |
| **Integration** | **`test_stream_pipeline` (new, V.2)** | **74** |
| Existing suites | `test_inc_*` 14 scripts 1,601; `test_inc_ap_*` 11 scripts 2,030; `test_funnel_*` 27 scripts 1,939; `test_domain_config` 53; `test_species_docs` 86; `test_model_router_placement` 28; `test_poster_data` 33 | 5,770 |

### V.2 The end-to-end run: `tests/test_stream_pipeline.py`

One synthetic world and one INC_DIR. The autopilot (the real `campaign.tick` → `StreamRun`, executor, policy and approvals, on `test_stream_ap_world`'s simulated cluster) proposes every lever from evidence that the real code of groups A–E wrote. Each proposal is then run by the real module it names. The run takes about 9 s.

1. **v1**, as on the cluster: the pinned `inc.splits` build and lock, then v1 Step 1 (verify pool, crops, embed, fit, admit, and select build).
2. **R0.** The autopilot files **L23** for a person twice, for `inc2.splits build` and then `lock`. Each is approved as the owner. The build runs the embedding copy scan by itself and reuses the funnel's passed `leak_v1.json`. The autopilot then runs, in order:
   - **L23B** ×5 (`inc2.baseline build`: B_v2, the canary, both capacity arms, B0 ∪ tsw), on the pinned driver with a FakeBackend;
   - **LV** ×2, **L25**, **LV**, **LI** (`inc2.stream init`), **LA** (`choose-arm`), **L28** (Stage C);
   - **LC**, which reads Stage C as M-feasible.
3. **R1.** In the DATA lane, before any discovery, the autopilot runs **L17** `bootstrap`, `knowntruth` and `backfill` (`inc2.step1_stream`). Batch b0000:
   - queues a masked veto image holding `funnel_F9`; its kept pixels are unchanged and its conflict box is filled with the mean colour;
   - refuses a mirrored test image (GuardV2, `near_eval_variant`) and a 15 % crop of a dev image (the embedding scan, `near_eval_embed`);
   - sends an overlap to the human queue and never queues base B.

   **D28** → **L24** quarantines the leaking source once the stream exists.
4. **R2.** A person resolves licences (`inc2.stream release --hold licence --keys`). `inc2.stream cut` shows one source's M images. Then:
   - **L18** builds segment 1 (K = 2), and its `exp.json` passes the pinned `validate_definition` and `check_definition_data`;
   - the snapshot's advances run it;
   - **L19** commits it through `inc2.gate3`: the planted bad increment is REJECTed as data and its dHashes are quarantined, and the good one is ACCEPTed (P_1 = P_0 + good).
5. **MAINT.**
   - **L20** builds milestone 1 (5 seeds on P_1, `inc2.baseline` role milestone, with the incumbent's secondary scores). **LC** compares it 5 v 5 with B_v2: helps.
   - Segment 2 accepts a good increment and a harmful one ("sneaky": it helps the warm chain and hurts a cold model).
   - **L20** and **LC** find that milestone 2 hurts (p 0.004). The autopilot then runs **L21** (rollback to P_1), **L27** (one bisect arm per suspect increment) and **LC**: the harmful increment is quarantined and the good one is returned to the queue.
   - `inc2.stream_report --stream` reports it.
6. **P8.** The funnel's F9 file and `serve-holds --hold funnel_F9` release the masked row's hold. The next snapshot refreshes the stream summary.
7. **DATA.** Two annotation-index sources go through the real collector: `collect.fetch` on a fake network, then `collect intake` against the real LOCK v2 and GuardV2. The autopilot adopts the clean source's intaken state and runs **L17** `admit --intake` (the real `step1_stream admit`): 5 whole-admitted rows. D28 reads the other source's `source_leak`, which holds a mirrored dev image, and L24 quarantines it before any admission.
8. **Deploy.** `deploy/deploy_funnel.sh --dry-run` ships every stream module (the S23 list), the five stream job scripts, every replay script, and every test and fixture they import.

**Invariant, checked on every spec the driver hands the executor** (173 specs, 36 training manifests):
- every training manifest passes the real `inc2.train.check_manifest` and `guard_rows`;
- an independent index finds no dev, test or ImageWeeds image in any of them, by key, sha256, or dHash within 6 bits under the 8 flips and rotations;
- the same holds for every pool P_s and increment the stream wrote.

**Stand-ins, named in the script's docstring:**
- **Models are not trained.** A synthetic executor writes scores and Protocol v3 sidecars from what each manifest holds.
- **Embedders.** The colour embedder of `test_inc_verify` stands in for BioCLIP-2, and a hue-histogram descriptor for DINOv2. The copy-scan threshold is set from the world.
- **Verdict files the world writes.** The canary is judged against b0_v1's real runs, and Stage A needs pilot_v3's real bins, so the world writes both verdict files. The builds of both still run.
- **`--testing`.** `capacity-verdict` is called with `testing_ok`, and the synthetic jobs add `--testing` where a synthetic world needs it. The platform's argv never passes it.
- **Probe, fetch and audit.** The probe's `placement.json` is the world's. The fetch runs in-process with the disk-headroom rule patched. The L4 audit is proposed, but the policy's path patterns, which pin the cluster's INC_DIR, refuse it.

### V.3 Defects the run found at the seams, and their fixes

Each check below fails when its change is switched off (V.4).

1. **D28 proposed L24 before the stream existed** (F, `diagnose_stream.d28`).
   - Defect: R1 can find a leak before R0's `init`. `inc2.stream quarantine` then exits 1, twice, which holds the STOP lane. D28's TRAIN hold then never lifts, so no segment is ever cut.
   - Fix: L24 is proposed only once the stream ledger shows `init`; the card and the TRAIN hold stay.
   - Check: segment 1 is cut; "no failed L24".
2. **D24 proposed a milestone while a rollback was pending** (F, `d24`).
   - Defect: the milestone was built after L21 on the rolled-back pool, which inc2.stream refuses: a failed MAINT step.
   - Fix: D24 is silent while `rollback_pending` names a pool.
   - Check: the MAINT order L20, LC, L21, L27, LC.
3. **D24's 30-day trigger stopped at the last stream write** (F, `d24`).
   - Fix: the days since the summary's `generated_utc` are added.
   - Check: milestone 1.
4. **Collection could start before R1** (F, `d20`).
   - Defect: D20 comes before DR0, so the one-item DATA lane took discovery (or a fetch) before Step 1's one-time jobs. §10 starts collection once R1 passes.
   - Fix: D20 is silent until bootstrap, knowntruth and backfill have run. The S6 fixture context now states that R1 ran.
   - Checks: "R1 before collection"; S6.
5. **The snapshot shipped a stale `queue_summary.json`** (E ↔ F, `stream_remote.refresh_summary`).
   - Defect: the stream rewrites its summary only in its writing verbs. A Step 1 batch that added supply was therefore invisible to D22, and the time-based counts stopped. Group E's note left this to the autopilot.
   - Fix: the snapshot runs `inc2.stream summary --stream SID` (the stream's own writer, under its lease) when step1_stream's queue or events file is newer than the summary, or the summary is older than 24 h. A busy lease keeps the old file. The call is bounded to 240 s, inside the snapshot's 600 s.
   - Check: the P8 stage.
6. **Collector states were ignored** (D ↔ F, `stream.py`).
   - Defect: a source fetched or intaken outside the campaign's own items (a person's run, a lab fetch, a restarted ticker) had no status, so DPIPE never admitted its batch.
   - Fix: its status is adopted from the collector's ledger when the ticker has none or `candidate`.
   - Check: the DATA stage.
7. **The stream report hid a rollback** (E, `inc2/stream_report.py`).
   - Defect: after milestone 2 hurt and the stream rolled back, the report led with milestone 2's test number without saying so, and listed the bisected increment as "accepted".
   - Fix (additive): the headline now carries `verdict`, `current_pool` and `rolled_back`, and the markdown says the stream rolled back. The timeline reads "accepted, then bisect_hurts", and the per-source yield adds `then_<status>`.
   - Checks: the report stage; `test_inc2_stream_report` and `test_inc2_stream` pass unchanged.
8. **A funnel deploy would have switched off every campaign's envelope** (`deploy/deploy_funnel.sh`).
   - Defect: the script ships `tools/inc_autopilot`, whose replay gate (`executor.REPLAY_SCRIPTS`) now runs the stream cases. Those cases import `tools/inc2`, `tools/collect` and the stream tests, which the script did not ship. `record-replay` would fail on the lab, and a failed record switches off envelope autonomy for every campaign.
   - Fix: the script now ships:
     - `tools/inc2` and `tools/collect` with its domain configs;
     - `round_scheduler.py` and `brain/su_rates.json`;
     - the five stream job scripts;
     - the stream, collector and inc2 tests with their fixtures;
     - this document, INCREMENTAL_PROTOCOL(_RUNNER).md and INC_AUTOPILOT.md.

     It now also:
     - ships the shared libraries the stream's job scripts hash, and refuses to change them (`--allow-shared-change`), as it refuses pinned modules;
     - runs the replay set (inc, funnel and stream, both mutation harnesses) and this pipeline locally before any copy (exit 6);
     - exits 5 when `record-replay` does not pass on the lab;
     - writes `results/framework/inc/stream_deployed.json` with the stream modules' sha256 (S23).
   - Check: the deploy stage.

**Left open, each for the owner or outside the six groups' files:**
- **§8 "No mega_trainer training".** The weed `train`/`filter` fallback in `round_scheduler._WEED_STEPS` stays, and `round_train`/`round_filter` are not person-only. `test_round_templates` requires both. The ownership guard closes these paths while a stream campaign names the domain.
- **The dashboard's harvest route** does not call `old_path_refusal(..., auto_sync=True)`, because `dashboard_server.py` belongs to no group.
- **The report's headline** is still the latest milestone's number after a rollback, which is group E's documented rule. It now states the rollback beside the number.
- **FUNNEL_REPRODUCE.md's "What it copies"** lists the funnel's paths only. The script's header and item 8 above list the stream's.
- **Group E's open item** (the snapshot regenerating a segment's `report.json` with the pinned report) did not occur in the run. The report written at commit is newer than the finished `state.json`, and segment 1's report still carries its `stream_commit`.

### V.4 Mutation checks

`tests/test_stream_pipeline.py` against 12 single-line mutations, applied one at a time. Each file was restored and its sha256 re-checked; every mutant makes the run fail.

| # | Mutation | The check that fails |
|---|---|---|
| M1 | D28 proposes L24 before the stream exists | L18 never runs (segment 1) |
| M2 | D24 ignores a pending rollback | the MAINT order after milestone 2 |
| M3 | D24 counts days only to the last stream write | milestone 1 is never built |
| M4 | the snapshot never refreshes a stale summary | the P8 refresh |
| M5 | the report hides the rollback beside its headline | the report headline |
| M6 | the report's yield drops the later bisect decision | the timeline and yield |
| M7 | GuardV2 skips the 8 flips and rotations | b0000 and the intake both keep a mirrored evaluation image |
| M8 | the disposition rule never says data | the planted bad increment is not quarantined |
| M9 | D20 collects before R1 | "R1 before collection" |
| M10 | the deploy leaves `tools/inc2` behind | the deploy coverage |
| M11 | the deploy leaves the stream tests behind | the replay-script coverage |
| M12 | the ticker ignores the collector's fetched/intaken states | L17 admit of the intake batch |

The groups' own mutation checks, recorded in their build notes, stand beside these:
- A: 20;
- B: 45;
- C: 43;
- D: 48;
- E: 29;
- F: 29 from its review, plus SM1–SM18 in `test_stream_ap_mutations`, which still kill 18 of 18 after V.3.

### V.5 What only the lab and the cluster can check

1. **Real-data counts.**
   - base_v2's size (about 6,813 with L-5) and M (about 682);
   - the tsw drops for dev sessions and near-eval copies;
   - the LOCK v2 pins: the v1 LOCK sha256s, 5,802 never-train entries, and 812 L-5 images;
   - the embedding scan's result on base B's 66 kept images and on tsw22 and tsw23.
2. **The copy detector.**
   - DINOv2 loads from the HF cache offline on a compute node.
   - The threshold comes from the funnel's `leak_v1.json` or from the stream's own calibration, which must pass H6's gates.
   - The scan's GPU time.
3. **Step 1 on BioCLIP-2.**
   - the bootstrap canary: 64 crops re-embedded at cos ≥ 0.999;
   - v1 `crops.csv` = 564,686 crops, and the verifier's `crops_sha256` d8188b16…;
   - b0000 reconciling with `census_v1.json` (461 images, or the contract's 457);
   - the first out-of-domain precision with its Wilson bound;
   - the backfill's cost (GuardV2 over about 96,500 images).
4. **`inc2.train` under Ultralytics 8.4.37** (the laptop has 8.4.22). The scorer sidecar must reproduce per-class AP (≤ 1e-9). The canary is its first reading, and the canary must fall within 1 sd of b0_v1.
5. **Capacity arms.**
   - `yolo11s.pt` and `yolo11m.pt` must be in `$REPO`;
   - YOLO11m at 640 px and batch 32 must fit a V100-32GB;
   - each arm's walltime against the 8 h cold limit (the high bracket projects 5.0 h and 15.8 h);
   - the measured step cost gives `truth_every`.
6. **Slurm.**
   - GPU-shared accepts every stream job (qos);
   - the log directories exist before the first job;
   - the outer-against-nested drift checks;
   - flock and the owner files on Lustre;
   - `INC_JOB_SCRIPT` in in-job advances.
7. **The snapshot's reads.**
   - the `projects` and `my_quotas` layouts and the allocation's end date;
   - /ocean headroom (93 % full on 2026-09-26, so D27 may pause at once);
   - the domain ledger against `su_envelope` (D10S);
   - sacct settlement;
   - the summary refresh's duration on the real queue, which must stay under 240 s.
8. **Lab and cluster.**
   - the rsync hosts, and the candidates and names syncs;
   - S23's module-hash comparison after a deploy;
   - `record-replay` passing on the lab;
   - the first run of `deploy_funnel.sh`'s pre-flight on the workstation.
9. **Providers.**
   - the network probe from a compute node;
   - the Kaggle token rotation;
   - PAGS8's scripted download (the upload holds 614 images and 1,940 boxes);
   - MFWD over FTP (6.78 GB for POROL) and the Roboflow export;
   - the owner's acceptance of the EPPO table and the PAGS8 card table;
   - `floor_gb` and `floor_su` from the first wave.
10. **The L4 label audit on a v2 segment.** `run_inc_audit.sh` must read v2 manifests, and the policy's paths must match the cluster's INC_DIR.
11. **The pinned `inc.report` on v2 experiments with real scores.** In the run it wrote every baseline's, Stage C's and every segment's report.

### V.6 The R0 command sequence the platform runs

This is the order the autopilot took in V.2. Every build job is submitted by `remote.py stream-submit` as `sbatch --parsable --job-name=<name> -p GPU-shared $REPO/weed_llm_benchmark/<script> <args>`. Every login-node verb runs through `remote.py stream-run` as `python -m weed_optimizer_framework.tools.<module> <verb> <args>`. `$INC_DIR` = `/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc`, and `SID` = `weed_stream_v1`. Before step 1, a person does §10 items 1–4. Before a gated DATA item runs, `data_autonomy` must be on; until then each such item is filed for a person, as §10 R1 allows.

The MAINT and DATA lanes run side by side, one item each. Each build is advanced by the snapshot with `INC_JOB_SCRIPT=run_inc2_job.sh`, and the next item waits until the experiment is done and its report is current.

**MAINT lane:**

| # | Lever | Where | Command |
|---|---|---|---|
| 1 | L23 (a person approves the D-A command) | job `inc_build_splits_build_inc2` | `run_inc2_build.sh inc2.splits build`, which runs the embedding copy scan itself on the GPU |
| 2 | L23 (approved) | job `inc_build_splits_lock_inc2` | `run_inc2_build.sh inc2.splits lock` |
| 3 | L23B | job `inc_build_b_v2` | `run_inc2_build.sh inc2.baseline build --exp b_v2 --manifest $INC_DIR/splits/v2/base_v2.jsonl --seeds 0,1,2,3,4 --arm n640 --role b_v2` |
| 4 | L23B | job `inc_build_canary_v2` | `run_inc2_build.sh inc2.baseline build --exp canary_v2 --manifest $INC_DIR/splits/v2/train_core.jsonl --seeds 0 --arm n640 --role canary` |
| 5 | L23B | job `inc_build_b_v2_s640` | `run_inc2_build.sh inc2.baseline build --exp b_v2_s640 --manifest $INC_DIR/splits/v2/base_v2.jsonl --seeds 0,1,2 --arm s640 --role capacity` |
| 6 | L23B | job `inc_build_b_v2_m640` | `run_inc2_build.sh inc2.baseline build --exp b_v2_m640 --manifest $INC_DIR/splits/v2/base_v2.jsonl --seeds 0,1,2 --arm m640 --role capacity` |
| 7 | L23B | job `inc_build_b0_tsw_v2` | `run_inc2_build.sh inc2.baseline build --exp b0_tsw_v2 --union $INC_DIR/splits/v2/train_core.jsonl,$INC_DIR/splits/v2/tsw22.jsonl,$INC_DIR/splits/v2/tsw23.jsonl --seeds 0,1,2 --arm n640 --role union` |
| 8 | LV | login node | `inc2.baseline canary-verdict --exp canary_v2` |
| 9 | LV | login node | `inc2.baseline capacity-verdict` |
| 10 | L25 (after a person accepts Protocol v3) | job `inc_build_pilot_v4` | `run_inc2_build.sh inc2.pilot4 build --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b` |
| 11 | LV | login node | `inc2.pilot4 verdict --exp pilot_v4` |
| 12 | LI | job `inc_build_stream_init_weed_stream_v1` | `run_inc2_build.sh inc2.stream init --stream weed_stream_v1 --stage-b r0,<Stage A's survivor>` (Stage A's `segment1_recipes`; `r0` alone when none survived) |
| 13 | LA | login node | `inc2.stream choose-arm --stream weed_stream_v1` |
| 14 | L28 | job `inc_build_weed_stream_v1_c001` | `run_inc2_build.sh inc2.stream feasibility --stream weed_stream_v1 --holdout tsw22 --m <M>` (M from `queue_summary.json`) |
| 15 | LC | login node | `inc2.stream compare --exp weed_stream_v1_c001` |

**DATA lane:**

| # | Lever | Where | Command |
|---|---|---|---|
| a | LP (only when there is no `placement.json`) | job | `run_inc_collect.sh probe` |
| b | L17 | job `inc_stream_admit_bootstrap_bootstrap` | `run_inc2_stream.sh bootstrap` |
| c | L17 | job `inc_stream_admit_knowntruth_knowntruth` | `run_inc2_stream.sh knowntruth` |
| d | L17 | job `inc_stream_admit_backfill_backfill` | `run_inc2_stream.sh backfill` |

**R2 begins** once Stage C reads M as feasible and Q ≥ M holds under D22's rule. First **L18** runs as job `inc_build_weed_stream_v1_s001`: `run_inc2_build.sh inc2.stream build --stream weed_stream_v1 --k <K> --exp weed_stream_v1_s001`. Then **L19** runs on the login node: `inc2.stream commit --exp weed_stream_v1_s001`.

Set `STREAM_PIPELINE_COMMANDS=1` to print the whole sequence the run proposed, R2 and MAINT included.

## Amendment (2026-10-03): D28-v2, cosine-confirmed source leaks (pre-registered)

Amendment V3-1. The rule below was fixed before any source was judged by it, and it is not edited afterwards. It changes D28 (§6.4) and adds the producers of the evidence D28 now reads. GuardV2 does not change.

**Why.** Under D28 as amended on 2026-09-29, one dHash hit quarantined a whole source. A hit is an image within 6 bits of one of the 5,802 never-train v2 images, under any of the 8 flips and rotations. A read-only measurement on 2026-10-01 showed that this rule misfires at scale:
- All 12 sources the stream had quarantined under D28 are dHash false positives. 49 of the 102,920 Step 1 pool images hit the never-train index, at 5–6 bits and only under flip or rotation variants.
- Each hit's DINOv2 pair cosine with the evaluation image it matched lies between −0.05 and 0.66 (median 0.106), and 0.744 at most (one intake source). True flip and rotation copies score at least 0.896 (5th percentile), and the weakest augmentation family at least 0.826.
- The per-image false-positive rate of the 6-bit, 8-variant rule is 4.76e-4 (one-sided 97.5 % upper bound 6.29e-4). A source of 200,000 images, the size of the largest collection candidate, holds about 95 chance hits, so the one-hit rule would quarantine it on arrival.

**The rule.**
1. Every dHash hit (≤ 6 bits, 8 variants) still drops its own image. GuardV2 is unchanged.
2. A hit is **confirmed** when the DINOv2 pair cosine between the hit image and the matched evaluation image is ≥ 0.80 (`stream_thresholds.json` D28 `confirm_cos`).
3. A source **leaks** when either:
   - (a) any hit has a pair cosine ≥ the copy threshold of the v2 embedding calibration (its `cos_threshold`, 0.946384 on the cluster, read through `inc2.embed_calibration`); or
   - (b) its confirmed hits are improbable by chance: P(Binom(n_images, p_c) ≥ k_confirmed) < `source_alpha` (0.001, unchanged), with p_c = `p_confirmed` = 6.29e-4. That is the bound of the unconfirmed rate, which bounds the confirmed rate from above. It holds until a calibration of the confirmed rate exists.

   Embedding hits keep their own binomial rule (`inc2.embed_calibration.source_verdict`), and the base-copy share keeps its 20 % rule.
4. **Fail closed.** An intake summary or a Step 1 per-source row whose dHash hits are not all weighed falls back to the old rule: one hit is a leak. That covers a row with no record, and a row with fewer pair cosines than hits.
5. For every source with dHash hits, the diagnosis states its hits, confirmed hits, maximum pair cosine, P and verdict.

What follows from the numbers: with p_c = 6.29e-4, one confirmed hit gives P ≥ 0.001 in any source of two or more images. So a single hit below the copy threshold never quarantines a source on its own, though its image is still dropped. A hit at the copy threshold does quarantine it, and so do, for example, 8 confirmed hits in 2,000 images (1.26 expected, P < 0.001).

**What changed.**
1. **`inc2/eval_hits.py` (new).**
   - `pair_cosines`: the pair cosine of each hit. Both images are described as the copy scan describes them: `funnel.leak.prepare_hashed` through `funnel.embed.embed_images`, with the calibration's embedder. A hit is weighed against the evaluation image GuardV2 matched and every other evaluation image within the never-train radius of it under any of the 8 variants (`eval_matches`), and its pair cosine is the highest of them. An evaluation descriptor is the copy scanner's EvalIndex row when one is held; otherwise the evaluation image is described from its path. A hit that cannot be scored, or one of whose evaluation images cannot be described, gets None and the reason, and the function never raises.
   - `record`: the per-batch record (`inc2-eval-hit-cosines/1`). It holds the embedder, the copy threshold and calibration the producer read, the counts of hits and of weighed hits, why any hit was not weighed, and per source the descending pair cosines. It holds no evaluation keys.
   - `verdict` applies the rule above, and `describe` states it in one line. On a fail-closed row both also give the confirmed and copy-threshold hits among the weighed ones; they decide nothing there.
   - `sidecar`, `sidecar_pairs` and `usable_sidecar`: the sidecar format (`inc2-eval-hit-sidecar/1`) of a batch committed before this amendment, and the check every reader applies to one. A sidecar is used only when it names the batch and weighed exactly the dHash hits the batch counted, source by source.
   - The module is not part of the v2 calibration's identity, which hashes `embed_calibration.py` and `guard.py` (`embed_calibration.identity_of`). Editing it never makes the locked calibration stale. `guard.py` is read, never edited: `eval_matches` reads GuardV2's never-train index (`_eval`, a `near_dup.NearHashIndex`, whose `matches` lists every hash within range).
2. **`collect/intake.py`.**
   - An image the guard refuses as `near_eval_v2` or `near_eval_variant` is counted and recorded as before. Its copy is now kept until every image of the batch has been judged, with the list of every evaluation image within the radius of it. It is then weighed (`score_eval_hits`) and removed. The removal runs in a `finally` around the whole per-image loop and the weighing, so a failure anywhere in them (an unreadable image, a full disk, the embedder) never leaves a copy of an evaluation image in the uncommitted batch directory.
   - The descriptor is the embedder the bound v2 calibration names (`load_copy_scan`, from LOCK v2). The evaluation images come from the evaluation manifests LOCK v2 records.
   - The hit's decision in `decisions.jsonl` gets `pair_cos` (and `pair_cos_best` when an evaluation image other than the guard's match gave it), or `pair_cos_why` when the hit was not weighed. `summary.json` gets `eval_hits`.
   - A batch without a hit loads no model. With no bound calibration (a testing LOCK, a stand-in guard), with another embedder, or on any failure, the hits are recorded unweighed with the reason. The intake never fails because of this.
   - `rescore_eval_hits` (with `eval_hits_needed`) writes the sidecar `intake/<batch>/eval_hits.json` of a committed batch whose summary does not weigh its hits; see choice 3. The batch's own files are never rewritten.
3. **`inc2/step1_stream.py`.**
   - `score_eval_hits` runs in `ingest` (registry, intake and known-truth batches) and in `backfill` (b0000), after the copy scan, with the scanner's index and embedder, and with the guard for the list of every evaluation image within the radius.
   - Each refused row keeps `pair_cos` in `ingest.jsonl`. A future b0000 writes no `ingest.jsonl`, so its `batch.json` keeps each hit's match and pair cosine (`eval_hit_pairs`). Every `batch.json` gets `eval_hits`.
   - New verb `eval-hits` (`eval_hits`, `step1_eval_hits`): the sidecars of batches committed before this amendment, `step1_stream/eval_hits/<batch>.json` for a Step 1 batch and, through `collect.intake.rescore_eval_hits`, `intake/<batch>/eval_hits.json` for an intake batch; one attempt per batch (`--force` writes one again); a batch it cannot give a sidecar makes the job exit 2. It records when it ran in `one_time.eval_hits`.
   - A Step 1 sidecar records the sha256 of the `batch.json` bytes the job read and parsed (`batch_json_sha256`), never the ledger's. A `batch.json` that no longer hashes to the sha256 the ledger commits gets no sidecar (it would never be folded): the batch is failed.
   - `--batch-id B` (repeatable) weighs only those Step 1 batches and no intake batch; `--intake X` only that intake batch and no Step 1 batch; both, both; neither, every batch that needs a sidecar (`eval_hits_scope`).
   - `write_status` folds per source into `eval_hits_scored`, `eval_hit_pair_cos` (descending) and `eval_hit_copy_threshold` each batch's own record, or, where that does not weigh the batch's hits, its sidecar, when the sidecar was made from the `batch.json` the ledger commits and weighed exactly the hits the batch counted. status.json's new section `eval_hits` lists the batches still due for a sidecar (`due`) and how each sidecar was read (`sidecars`). Known-truth batches are left out, as they are for the decision counts.
   - `copy_threshold(scanner)` gives the v2 calibration's `cos_threshold` when the scanner holds it, else the lowest threshold it holds.
   - `inc2/eval_hits.py` is added to the module hashes a commit records.
4. **`inc_autopilot/diagnose_stream.py` (D28, DR0) and `stream_thresholds.json`.**
   - `stream_thresholds.json` gains `confirm_cos` (0.80) and `p_confirmed` (6.29e-4), each with its why, and `source_alpha`'s why is updated.
   - D28 judges a source over all its intake batches together (choice 4), from each summary's `eval_hits` or, where that does not weigh the batch's hits, the sidecar `intake/<batch>/eval_hits.json` the snapshot ships; and Step 1 sources from status.json's per-source fold. It applies `eval_hits.verdict` beside the unchanged embedding and base-copy rules.
   - The copy threshold is the lowest of the producers' records and LOCK v2's `embed_calibration_v2.cos_threshold`, which the snapshot's `splits/v2/lock_status.json` carries. With none, `confirm_cos` stands in, which is never less strict.
   - The summary states every source with dHash hits, each with its hits, confirmed hits (on a fail-closed row, among the weighed ones), maximum pair cosine, P and verdict: the leaks, then every source judged chance (also listed in `detail.cleared`). Quarantined sources it now judges chance are named for a person (`detail.cleared_quarantined`).
   - A source has one verdict across its intake batches and Step 1. One that leaks in either path is never in `detail.cleared` or `detail.cleared_quarantined`; its dHash numbers from the other path are stated with its leak (`dhash_elsewhere`). One judged chance in both paths is listed once, with each path's numbers (`dhash_by_path`).
   - DR0 proposes L17 `eval-hits` in the DATA lane once Step 1's one-time jobs have run, while a committed batch counts dHash hits that neither its record nor a sidecar weighs: status.json's `eval_hits.due` (or, in a status.json written before the sidecars existed, a source row with more dHash hits than pair cosines), or an intake summary without its sidecar.
   - The SM10 mutation line keeps its form.
   - §6.4's D28 row points here.
5. **The platform's plumbing for `eval-hits`.** L17 (`inc_stream_admit`, R2, gated as every L17 job) takes the verb: `policy_actions.json` (its verb enum and description), `stream_remote.py` (`stream-submit admit eval-hits`, no flags; the snapshot ships every `intake/<batch>/eval_hits.json`), `evidence.py` (`intake/<batch>/eval_hits.json` on the allow-list; a Step 1 sidecar is not, because status.json carries its fold and the sidecar names each hit's evaluation key) and `stream_levers.json` (L17's note). It is priced as an admit of unknown size (`admit_fixed_hours` + 1 h, an upper bound). `collect/__init__.py` names the sidecar's format.
6. **Job scripts.**
   - `run_inc_collect.sh` now also logs `inc2/eval_hits.py` (an intake without it is refused), `funnel/embed.py` and `semisup_labeler.py`. torch and transformers are not a precondition: only a batch with a hit loads the model, and without one that hit is recorded unweighed (fail closed). intake, and only intake, runs with `HF_HUB_OFFLINE=1` (compute nodes have no internet; the model comes from the Hugging Face cache, as in `run_inc2_stream.sh`), while fetch keeps the network. The script no longer says the collector never uses the GPU: intake's DINOv2 runs on it.
   - `run_inc2_stream.sh` accepts `eval-hits`, checks torch and transformers for it, and, for it only, hashes every file of the collector package into the log and refuses an outer copy that differs.
   - `run_inc2_build.sh` lists `inc2/eval_hits.py` in its drift check, because `inc2.stream` imports `inc2.step1_stream`. `run_inc2_stream.sh` and `run_inc2_job.sh` already check every `inc2` module.

**Choices where the rule is silent, and why.**
1. **Two producers.** Intake drops a hit's image, and D28 reads the intake summary before the batch is admitted. So the pair cosine of an intake hit can only be taken in the intake job, which already runs on a GPU-shared node. The Step 1 jobs (`admit`, `backfill`) already hold the scanner's evaluation descriptors and embedder, and weigh their own hits there.
2. **The pair.** GuardV2 returns the first match it finds (the image's own dHash first, then the variants), and an unrelated evaluation image there must not hide a copy that lies within the radius as well. So a hit is weighed against every evaluation image within 6 bits of it under any of the 8 variants, the guard's match among them, and keeps the highest pair cosine. A hit any of whose evaluation images cannot be described is left unweighed, never guessed.
3. **Batches committed before this amendment.** Their own records hold no pair cosines, and their hits are weighed again, deterministically, from the files they were judged on, into sidecars; a batch file is never rewritten.
   - A Step 1 batch: b0000 has no `ingest.jsonl`, so its hits are the rows `admission.jsonl` records refused `near_eval_v2` or `near_eval_variant`, each image and dHash from the v1 pool manifest and dHash cache (the image checked against the manifest's sha256), the match decided again by GuardV2 under the pinned LOCK. A later batch: its `ingest.jsonl` rows (the file checked against the sha256 `batch.json` records), each image checked against the row's sha256 and decided again. Both are weighed with the copy scanner's EvalIndex and DINOv2, as at admission. `batch.json` stays as committed (the ledger hash-locks it); status.json folds the sidecar in its place.
   - An intake batch: its `decisions.jsonl` (checked against the sha256 its summary records) names each hit's rel, key, sha256 and match; the fetch it was intaken from (its summary's fetch sha256; the staging record and blobs checked) is materialised and normalised again in a work directory of its own, removed at the end; the file at each rel must hash to the decision's sha256 and be refused again by GuardV2 with the same reason and match; it is then weighed as intake weighs a hit and removed. A staging that now holds another fetch, or anything that cannot be re-derived, leaves the hit unweighed with the reason (fail closed). A reason states the guard's reasons only, never an evaluation split or key, because the sidecar ships.
   - **How the 12 quarantined sources are re-judged after deploy.** (1) Once Step 1's one-time jobs have run, DR0 sees the intake batch without its sidecar and Step 1's per-source rows with dHash hits and no pair cosines, and proposes L17 `eval-hits` (DATA lane, R2, gated and capped as every L17 job). (2) The job `run_inc2_stream.sh eval-hits` writes `step1_stream/eval_hits/<batch>.json` for every Step 1 batch holding the Step 1 sources' hits (b0000 first) and `intake/<batch>/eval_hits.json` (the intake source's hit), and rewrites status.json. (3) The next snapshot carries status.json's fold and the intake sidecar, and D28 judges the 12 sources by their pair cosines. The 2026-10-01 measurement (every hit below 0.80, at most 0.744) predicts all 12 chance: they then appear in D28's summary with their numbers and under `detail.cleared_quarantined`. (4) A quarantine L24 has applied stays until a person lifts it: `python -m weed_optimizer_framework.tools.inc2.stream unquarantine --source <S> --stream weed_stream_v1 --decided-by human:<who>` for each source D28 names there. A source whose L24 is still pending is kept by denying its L24 item, as before (S25).
4. **A source is one source across its intake batches.** n is the number of images the guard checked (`images` in an intake summary, `images_seen` in Step 1's status), summed over every intake batch of the source, as are its dHash and embedding hits; its pair cosines are those of every batch (each batch's at most as many as its hits, so a batch that weighs fewer than its hits leaves the whole source short of pair cosines, and it fails closed); its copy threshold is the lowest any batch recorded; the embedding rule's per-image rate is the lowest of the batches with embedding hits (none when one of them has none: fail closed); the base-copy share is the highest of any batch (a share is a batch's, and the highest is never less strict than their mean). A source split into shards is therefore judged as one: 5 batches of 1,000 images with 3, 3, 2, 2, 2 confirmed hits are 12 in 5,000 (P ≈ 1.1e-4), a leak, though each batch alone is chance. Intake and Step 1 are not added together: the intake guard already dropped what Step 1 would count again.
5. **Evaluation keys stay out of what the autopilot reads**: the summaries, status.json and the intake sidecars the snapshot ships hold cosines and hit keys only. Each hit's match is in the batch's own files (`decisions.jsonl`, `ingest.jsonl`) or, for b0000, in its Step 1 sidecar, which stays on the cluster and is not on the evidence allow-list. A re-derived intake hit that GuardV2 now judges otherwise is recorded by its reasons only.
6. **One attempt per batch.** A batch with a sidecar is not weighed again unless a person asks (`--force`). A sidecar records what the batch's own files decide, including a hit they no longer let be re-derived. When the environment fails instead (no GuardV2, no v2 calibration, no copy scanner index, an embedder that weighs none of the re-derived hits), no sidecar is written, so the attempt is not used up: the batch fails the job (exit 2), an L17 failure the lane's stop-loss counts, never a silent re-proposal.
7. **The pipeline test's stand-in descriptor** is now summed over the 8 flips and rotations (test code only). A mirrored copy then scores 1.0 against its original, as DINOv2's nearly does. The earlier spatial-grid descriptor gave the planted mirrored copy 0.53, which no real copy scores.
8. **Unchanged:** splits' per-source accounting (`summary.json` `embed_v2`, informational), the funnel, `inc2/embed_calibration.py`, `inc2/guard.py` and every pinned module.

**How it was verified.** Locally, with no GPU and no network.
- `test_stream_ap_units.py`, section `t_d28_v2` (19 checks; the existing `t_d28` checks are unchanged):
  - The 12 live-like sources (49 hits, 1–23 per source, pair cosines from −0.05 to 0.66 with a median near 0.10, and the intake source at 0.744) do not quarantine. Each is judged chance with 0 confirmed hits and P = 1, and the summary states each one's numbers. Without the records, the same 12 quarantine (fail closed).
  - A planted hit at 0.95 quarantines, in an intake summary and in Step 1's counts.
  - One confirmed hit at 0.94 in 1,000 images is chance (P = 0.47). 8 confirmed hits in 2,000 images are a leak (P < 0.001); 3 in 2,000 are chance.
  - Fail closed: no record, 1 of 2 hits weighed, and a Step 1 source with 3 hits and 2 cosines each quarantine.
  - An SIU-sized source (200,000 images, 95 hits all below 0.80, 125.8 confirmed hits expected by chance) does not quarantine, nor with 3 of its hits confirmed. Without the records it does.
  - With no copy threshold known, 0.80 stands in. Embedding hits keep their rule.
- `test_stream_ap_units.py`, section `t_d28_v2_sources` (20 checks):
  - The 5-shard source above is a leak (12 confirmed in 5,000, P ≈ 1.1e-4, every batch cited), while each shard alone is chance (P 0.026 or 0.13). One shard without a record fails the whole source closed, and the row states "only 10 with a pair cosine, 10 confirmed among them". A source's copy threshold is the lowest any of its batches recorded.
  - The boundaries: 8 hits exactly at 0.80 in 2,000 images are a leak; one hit exactly at 0.946384 is a leak; a producer threshold of 0.95 with the LOCK's 0.946384 makes a hit at 0.948 a leak, and so does the reverse.
  - A legacy intake batch fails closed alone; its sidecar at 0.744 makes it chance (stated in the summary); a sidecar at 0.97 makes it a leak and is cited; a sidecar with another number of hits, or of another batch, is not used.
  - 25 sources judged chance are all stated, with no "and N more". A quarantined source D28 now clears is named, with the unquarantine command.
  - DR0 proposes L17 `eval-hits` for a due Step 1 batch and an intake batch without its sidecar, and from a status.json written before the sidecars; not once every batch has its sidecar; not before the one-time jobs. L17 `eval-hits` renders inside the policy bounds, the executor reads it back, and `stream-submit admit eval-hits` parses. The allow-list admits the intake sidecar path and nothing beside it; a Step 1 sidecar path is refused (`t_evidence`).
  - A mutation run on a copy: `>=` → `>` on the confirmed or the copy-threshold count, `min` → `max` of the copy threshold (in the verdict or in a source's batches), and judging intake batches one by one again are each killed by this file.
- `test_stream_ap_units.py`, section `t_d28_v2_round3` (13 checks):
  - A quarantined source whose intake dHash hit is chance (0.30) while Step 1's embedding hits leak (20 in 100) is a leak only: not cleared, not named for unquarantine, its intake numbers stated with the leak. The reverse (an intake copy at 0.97, a chance Step 1 hit) likewise, with its Step 1 numbers stated. A source judged chance in both paths is listed once, with each path's numbers.
  - A batch whose record holds more pair cosines (2) than its guard counted hits (1) lends none to another batch of the source: 1 of 2 weighed, fail closed.
  - The embedding rate is the lowest of the batches with embedding hits (3 in 200 at 1e-4: a leak; at 0.0105 chance), and a batch with embedding hits and no rate fails the source's embedding rule closed. The base-copy share is the highest of any batch (25 % then 0 %: a leak).
  - DR0 does not propose `eval-hits` again for an intake batch whose sidecar exists but weighed none of its hit, another number of hits, or another batch; D28 fails it closed; without a sidecar it is proposed.
- `test_stream_ap_replay.py` S14 adds a 3,000-image source whose 2 hits were weighed at 0.12 and 0.31. It is judged chance and never quarantined, while the unweighed source beside it is. `test_stream_ap_mutations.py` kills every mutant, SM10 by S14.
- `test_collect_intake.py`, with the real GuardV2 and a bound v2 calibration:
  - A flipped dev copy is weighed at 1.0 before its copy is removed, and the autopilot's D28 reads it as a leak. Another picture 2 dHash bits from the dev image is weighed below 0.80, and D28 judges it chance. A failing embedder, or another embedder, leaves the hit unweighed, and D28 fails closed. A batch without a hit never calls the embedder. A stand-in guard, or a testing LOCK without a calibration, records its hits unweighed.
  - A mirrored copy whose guard match is an unrelated neighbour (the image's own dHash 1–2 bits from a test image, pair cosine below 0.80 alone) keeps 1.0 from the dev original within the radius, and D28 quarantines it.
  - An injected failure while writing a later image's label leaves no copy of the evaluation image in the uncommitted batch, and the next intake redoes the batch.
  - `rescore_eval_hits` on legacy batches: with a failing embedder it refuses and writes no sidecar (the batch stays due); otherwise the hit is re-derived from the staging blob and weighed (below 0.80 for the near picture: D28 chance; 1.0 for the flipped copy: D28 a leak by the copy threshold); the batch's own files unchanged and the re-derived images gone; no evaluation path or key in the sidecar; "exists" on a second run, rewritten with `force`; a re-fetched staging leaves the hit unweighed (D28 fails closed); a batch whose record weighs its hits needs none; decisions that no longer hash refuse.
  - A mutation run on a copy: removing the copies only after a successful loop, or weighing against the guard's match alone, are each killed.
  - `rescore_eval_hits` leaves a hit unweighed, with the reason, when the decision's sha256 is not the re-derived file's, or when GuardV2 refuses it for another reason or matches it to another evaluation image than the decision records. Neither the sidecar nor the summary then names an evaluation split or key.
- `test_inc2_step1_stream.py`:
  - The registry batch's three dHash copies carry pair cosines (the byte copy of dev: 1.0); `batch.json` and the status fold record them; D28 reads the fold. Without a scanner index, or for a match without an evaluation key, a hit is left unweighed with the reason.
  - The same batch admitted without a copy scanner (a legacy record): status.json lists it as due and D28 fails closed; `eval-hits` without a scanner index writes nothing and reports the batch failed (it stays due); with the scanner it writes its sidecar (3 of 3 weighed, the byte copy 1.0), `batch.json` and the state unchanged; status.json folds it (nothing due) and D28 judges a_species by the byte copy; a second run leaves it; a sidecar from another `batch.json`, or with another number of hits, is not used. b0000's path: a v1 pool image copying a dev image is re-derived from `admission.jsonl`, the v1 pool and dHash cache, decided again by GuardV2 and weighed at 1.0; a row GuardV2 no longer refuses as recorded is left unweighed with the reason.
  - The CLI `eval-hits` exits 2 naming an intake batch it cannot give a sidecar, and 0 with nothing left, stamping `one_time.eval_hits`. Through `build_parser`, `--intake X` alone weighs no Step 1 batch.
  - A hit whose guard match is planted as the evaluation image least like it (pair cosine 0.18 alone) keeps 1.0 from the dev original within the radius.
  - `eval-hits` gives no sidecar to a batch whose `batch.json` changed since commit (failed), and a sidecar made from a changed `batch.json` records that file's sha256 and is not folded. `run_inc2_stream.sh eval-hits` runs, hashes the collector package (only for that verb) and refuses an outer `collect/intake.py` that differs.
- `test_collect_job_script.py`: intake runs with `HF_HUB_OFFLINE=1`, fetch and probe without; the GPU claim is gone.
- `test_stream_pipeline.py`:
  - b0000 weighs the mirrored test image at 1.0, at or above this world's copy threshold of 0.9465. The intake weighs the mirrored dev image before removing it. D28 quarantines both sources by the pair cosine, not by the bare hit, and its summary states the numbers.
  - With the leaking intake batch's record removed (a batch committed before the amendment), DR0 proposes L17 `eval-hits` once, the platform runs the real `step1_stream eval-hits` (exit 0), which writes `intake/<batch>/eval_hits.json` from the staging blob (1.0); the next snapshot ships it and D28 judges the source by it, citing it.
- A mutation run on scratch copies (28 mutants, each a one-line change; the null copy fails nothing beyond the pipeline test's 4 deploy checks that need `.git`): every mutant is killed by `test_stream_ap_units`, `test_collect_intake` or `test_inc2_step1_stream`. They are: `>`/`<` and `min`/`max` swaps in the rule; judging intake batches one by one; dropping the per-batch cap on pair cosines; the highest instead of the lowest embedding rate; the last instead of the highest base-copy share; no fail-closed on a batch without a rate; DR0 proposing a batch that has a sidecar; `rescore_eval_hits` skipping the sha256 or GuardV2 check; Step 1 weighing the guard's match alone; a leak also listed as cleared, or a source listed twice; an evaluation key in a rescore reason; the Step 1 sidecar on the allow-list; `--intake` alone weighing every Step 1 batch; and the sidecar's sha256 copied from the ledger, or written without the check.
- Every `test_inc2_*`, `test_collect_*` and `test_stream_*` script passes, `test_round_scheduler_stream_guard.py` and `test_inc_ap_evidence.py` included.

**Deploy.**
- `diagnose_stream.py`, `stream_thresholds.json`, `stream_levers.json`, `stream_remote.py`, `evidence.py` and `policy_actions.json` change the stream rules version and `executor.code_hash()`. Sync the lab and both cluster copies (nested and outer, the collector package included) from one commit; `run_inc2_build.sh` refuses a build whose outer copy lacks `inc2/eval_hits.py`, and `run_inc2_stream.sh eval-hits` refuses an outer collector package that differs. Sync `$REPO/run_inc_collect.sh` and `$REPO/run_inc2_stream.sh` from the nested copies, since the jobs refuse a copy that differs. Then run `executor.run_replay_tests`.
- [to verify on cluster] The intake job must be able to load the calibration's DINOv2 model from the Hugging Face cache, as the Step 1 jobs do. If it cannot, its hits are recorded unweighed and D28 fails closed.
- [to verify on cluster] The first `eval-hits` job: b0000's sidecar and the intake sidecar weigh every one of their dHash hits (none left unweighed with a reason), and status.json shows `eval_hits.sidecars.b0000.used` true.

**Open items.**
- A calibration of the confirmed-hit rate, to replace `p_confirmed`'s bound. It is a person's decision, recorded in `stream_thresholds.json`.
- The 12 sources quarantined before this amendment: re-judged by the sidecars once `eval-hits` has run (choice 3); a person lifts the quarantine of each source D28 then names as cleared.

## Amendment (2026-10-03): E1, weed-box base v3 (pre-registered)

Written before base v3 was built and before any E1 run; not edited afterwards. Decided by the owner's delegate under the 2026-09-30 grant (`human:harry567566@gmail.com`).

**Why.** The best sealed cwd12 test score is 0.8786 ± 0.0018 (YOLO11m at 640 on base_v2's 6,811 images); the gap to 0.90 is 0.021. Class-agnostic the same models score 0.8901 on test, so the boxes cap the score: a perfect species call on them would still miss 0.90. Five larger or newer detectors (m832, s1024, y26l640, y26m640, l640) gave no agnostic gain. Data has only ever been added in hundreds of images (s001: 682). E1 asks one question: does a much larger, cleaned, leak-checked weed-box training set place better boxes?

**Design (lean: no custom loss).** Every weed box of every admitted source is labelled class 12 (OtherPlant) in the existing 13-class INC space. The models are one-class weed detectors that every inc2 check, the pinned driver and the locked scorer accept, and their class-agnostic dev mAP50-95 compares directly with b_v2_m640's (agnostic dev 0.8695, test 0.8901). Crops are background: crop and non-plant boxes are dropped, and an image without a weed box is not admitted. Species come later.

### Pre-registration

**Arms.** YOLO11m at 640 (m640's COCO checkpoint), 3 seeds each, equal compute:
- **E1-A** (`e1_a_m640`): base_v2's 6,811 images, every box → 12 (`splits/v3/base_v2_weed.jsonl`).
- **E1-B** (`e1_b_m640`): E1-A plus every admitted external weed source (`splits/v3/base_v3_weed.jsonl`). E1-A's rows are byte-identical inside E1-B.

**Recipe `cold_budget`** (`inc2.recipes.cold_budget`): m640's cold recipe key for key except
- epochs = round_half_up(1.2e6 / N) (E1-A 176; E1-B about 40 at 30,000);
- warmup_epochs = round(640 / ceil(N / 32), 6), so Ultralytics warms up for exactly 640 iterations (base_v2's 3 epochs x 213 batches) at any N;
- close_mosaic = max(1, round_half_up(0.1 x epochs)).

round_half_up(x) = floor(x + 0.5). At m640's measured 14.2 ms per image-epoch a base run trains for about 4.7 GPU-h, under the pinned 8 h cold limit and D26's 6.4 h line.

**Decision (dev only, record only; never the stream's arm).** D = mean agnostic dev mAP50-95 (E1-B) − mean (E1-A) over the seeds both share. E1-B qualifies when D > 2 × pooled sd (√((sd_B² + sd_A²)/2)) **and** D > SE(D). SE(D) is a paired image bootstrap: 1,000 resamples of the dev images under `stable_int("inc2/e1/agnostic_se")`, one draw for every run; per run and resample the locked scorer's class-collapsed AP50-95 (`inc.scorer.collapsed_ap`) on the run's tie-broken per-image arrays; the mean over seeds per arm; B minus A. Qualifying switches nothing: the stream's arm stays the capacity decision's.

**Finals and the test read (an exception to P10, declared here).** Each arm's finals are dev and ImageWeeds (agnostic), never test. The sealed test is read once per arm, only after `capacity/e1_v1.json` holds a decided verdict. That read is a milestone read for P10: `inc2.baseline e1-test-read` writes one kind-final spec per seed on exam test and the `run_inc2_job.sh` argv, a person submits it, and the scores stay off the platform's evidence. It refuses when the verdict is missing, pending or changed, or a test score of the arm exists.

**Base v3 (`inc2/base3.py`, config `inc2/base3_v1.json`, both pinned by sha256 in summary.json).** (Since b8c8aef the builder loads `inc2/base3_v2.json`, revisions 2 and 3 below; `base3_v1.json` stays as revision 1's record.)
- *Sources.* An allow-list; any registry slug not listed is not read.
  - Included: the 16 dock slugs ("0 ridderzuring" / "grass weeds", *Rumex obtusifolius*; one family, its Roboflow export stems shared across the family); MH-Weed16 (all 15 ids weeds); gh_tehreemnoor; rf_school '0'; rf_zbm50; rf_tuf (empty label files are not admitted); rf_kinjj (polygons → boxes); rf_itmo (lawn forbs; *Poa*, litter, bare patch and dry grass dropped); the weed boxes of rf_cotton-weed-detect (a dock slug), rf_srec crop-and-weed fqrtg and rf_weed-tnf9e; the intake sources CottonWeedDet3 and MFWD (POROL and Weed), read only through their intake manifests (INC ids 0–13 all → 12), never through their registry paths.
  - Excluded (named in the config with the reason): the test groups (NDSU: ImageWeeds, weed_crop_detection, imageweeds_aerial, greenhouse; Latvia + francesco_weed_crop_aerial + peradeniya + vitif246x + test-8qezo + project-5nvic + kg_vinayakshanawad; sesame: ravirajsinh45, leopard, poxtn, zbfhf; maize; PAGS8), the OOD-dev groups (paddy rf_main-otq0a, chilli rf_1111-gzfxi), cwd12 and 3SeasonWeedDet10 copies (cottonweed_sp8, cottonweed_holdout, agrobot sd89f, three_season, cwp10, vanpe), crop-only (rf_robtica), synthetic or broken (cowpea, cotton_weed_detection), unresolved class names (bishwarup, test-gzc3r, gh_07931350), non-plant and disease sets.
  - The stream's source quarantines are an input (`--stream SID`: `inc2.stream`'s ledger fold, its head recorded); a quarantined source is not admitted until a person lifts it.
  - An intake row with holds is admitted only when Step 1's queue (`inc2.stream.QueueView`) holds it with no hold left.
- *Box side* (one definition for every rule): √(w × h) in pixels after the image's long side is scaled to 640 (COCO's area convention). Chosen over the shorter side because the thresholds are object sizes and a thin, long plant part is not a small object. This choice decides MFWD's admission (21.8 % of its boxes under 16 px by √(w·h), 27.0 % by the shorter side); the spec check measured both before this text was written.
- *Source rule*, over the source's weed boxes as delivered: median side ≥ 32 px; ≤ 25 % of boxes under 16 px; ≤ 10 % of boxes covering > 80 % of their image; median weed boxes per image ≤ 25. A failing source is excluded whole.
- *Box and image rules*: a weed box under 8 px is removed and its rectangle filled with the image's mean colour (`inc2.mask.masked_array`; kept boxes' pixels are never filled); a polygon becomes its bounding box; an image is dropped if a weed box covers > 90 % of it, its shorter side is < 320 px, the masked area is > 50 %, or no weed box remains. base_v2's rows are exempt from every rule (arm A is base_v2 whole; its 66 NDSU rows stay in both arms, so ImageWeeds is same-lab for both).
- *Dedupe* (never at 4–6 bits): the same original bytes (sha256); a dHash within 3 bits under one of the 8 flips and rotations **and** ≥ 80 % of the larger box set matched one-to-one at IoU ≥ 0.8 after that transform; or an identical label layout (rounded to 0.01, ≥ 3 boxes). One row is kept per copy group: base_v2 rows always (an external copy of one is dropped), else the lowest provider tier (intake and MH-Weed16 1; rf_unitec, the dock release with the original names, and gh_tehreemnoor 2; others 3), then the most pixels, then the slug.
- *Leak guard* (on the exact files written): GuardV2 and the index cross-check (`inc2.train.guard_verdicts`), the L-5 and L-8 lists on the original bytes, the calibrated embedding copy detector (`GuardV2.check_embed`, LOCK v2's calibration) on every external row, and D28-v2 per source (each dHash hit's pair cosine through `inc2.eval_hits`; the embedding binomial rule; the 20 % base-copy share; thresholds read from `stream_thresholds.json`). A refused row is dropped; a new row that copies a base_v2 image (`base_copy`) is dropped; a source that leaks is excluded whole. A base_v2 row the guard refuses for a never-train reason leaves both arms (none is expected; base_v2 rows are never dropped as base copies). base_v2 is not re-scanned by the embedding detector: LOCK v2 scanned it.
- *Main-test holdout v1.* Capture groups join images within 6 dHash bits under the 8 variants (both directions), their copy edges, the same Roboflow export stem (across the dock family), the intake capture group and DeepWeeds' capture time in gh_tehreemnoor's file names. From each included external source, whole groups are held out in an order seeded by `stable_int("inc2/base3/test_v1/<source>")` until ~15 % of its kept rows (min 30, max 400 images); a group touching base_v2, or any row of the stream's pools (P_1 holds 682 CottonWeedDet3 images of inc0001), or larger than 400 images, is never held out: test v1 must not hold an image a current model trained on. Held-out rows never train and are listed in `splits/v3/test_v1/<source>.jsonl` (original and written file, dHash and variants) for a later never-train index v3.
  - *Revision 2 (2026-10-03, before any build): SIU joins, capped as a family.* The verification's rehearsal put arm B at 23,015 images as is and 28,948 with rf_tuf and rf_zbm50 lifted. Both are under 30,000, because the dock family is mostly re-exports and the convention rule removes 16,552.
    - `inc2/base3_v2.json` (now `base3.CONFIG`) is v1 plus `zenodo_15808623` as an intake source of family `siu`. That source is SIU Weed Growth Stage: one greenhouse, potted plants, 16 species over 11 weeks, frames of 360° videos, of which the intake batch takes 40,000.
    - The `siu` family is capped at 35 % of base v3, as the dock family is: S ≤ floor(0.35 / 0.65 × non-SIU images), applied after the holdout by seeded whole capture groups. Uncapped, its 40,000 frames would be about 63 % of arm B, so arm B would mostly measure one greenhouse. Capped, arm B is about 35K images (23K plus about 12K SIU, estimated from counts only).
    - Intake rows now carry their source's family (`intake_rows`), and an intake family without a rule refuses the config.
    - Like every intake row, an SIU row enters base v3 only once Step 1 has released its intake holds. The platform's order already ensures this: DPIPE's admit of the source comes before DR0's eval-hits, and L23V is proposed only when the DATA lane has nothing due.
    - The source rules still apply to SIU as delivered: if more than 10 % of its boxes cover over 80 % of their image, or more than 25 % fall under 16 px, it is excluded whole, and the build reports it.
    - Tests: `test_inc2_base3.py` (`test_config` v2, `test_intake_family`).
  - *The stream never cuts a test v1 row* (2026-10-03, after the verification found 1,969 of the 2,636 test v1 rows in the v1 Step 1 pool with nothing to stop a segment from taking them). `inc2.step1_stream.load_queue`, the queue's one reader, marks every unrefused row that a test list under any `splits/*/test_v1/` holds: the same original or written bytes, the same key, or a dHash within 6 bits under the list's 8 variants (`inc2.base3.mark_prior`, the rule the builder applies to earlier lists). Such a row gets `test_v1` and is not eligible, and the cutter (`inc2.stream` eligibility) skips it with reason `test_v1`. An unreadable list refuses the read, so the cutter stops rather than guess. Tests: `test_inc2_stream.py` (`test_test_v1_never_cut`).
- *Dock cap*: after the holdout, D ≤ floor(0.35 / 0.65 × non-dock images), enforced by dropping whole capture groups in an order seeded by `stable_int("inc2/base3/test_v1/cap/dock")`.
- *Materialisation.* An image whose long side exceeds 640, or that has a masked box, is written as a lossless PNG of exactly the pixels Ultralytics trains on: its own `BaseDataset.load_image` (imread, long side → 640 with INTER_LINEAR) on the original, then the mask; every PNG is read back through `load_image` and must be equal. Ultralytics' RAM cache holds those pixels, so a cached run of the original and a run of the PNG train on the same arrays, and a 30K-image base needs no 12 MP decode per sample. The original's path and sha256 are kept in `provenance_v1.jsonl`.
- *Walltime guard.* The build measures arm B's loader through Ultralytics' own training dataset and DataLoader (default augmentation, no cache, 5 workers, a seeded 3,200-row sample). A base run is projected at max(loader rate, m640's measured 12.65 ms) × 1.2e6 + 0.27 h of fixed overhead; over 0.8 × 8 h the build writes `summary.json` with status `over_walltime`, which no E1 arm builds on, and ends with a refusal. This replaces a manual 1-epoch smoke: the GPU rate is measured, the loader was the open risk.

**Pricing.** An E1 arm is priced from its budget, whatever N: seeds × 1.2e6 × 14.2 ms / 3.6e6 + the finals + the build job = 19.0 GPU-h each (the 100-epoch basis would price E1-A at 16.7, too low, and E1-B at 57.3). L23V is one build job (4.0 GPU-h, its walltime); L23E six scoring passes (1.5 GPU-h).

### What changed

- `inc2/recipes.py`: `cold_budget`, `budget_record`, `check_budget_record`, `budget_cost`; `deviations` and `match` accept `cold_budget` for kind base only, when the caller names it with the base's N.
- `inc2/train.py`: `experiment_budget` reads exp.json's `recipe_name`, `base.n_images` and `budget` (the pre-registered constants, checked); a production base run of a baseline that names `cold_budget` is compared with `cold_budget(arm, n_images)`; a union, an incremental run or a chain naming it is refused. `guard_verdicts` (per-row verdicts) is factored out of `guard_rows`, which refuses exactly as before.
- `inc2/base3.py` (new) and `inc2/base3_v1.json` (new): the builder (`build`, a GPU job) and its read-only rehearsal (`count`, the login node, stored hashes, the guard on the originals' hashes, the embedding check and pair cosines pending).
- `inc2/scorer_agnostic.py` (new): the locked scorer as a library with a validator subclass that keeps each image's class-collapsed matches; `scores/dev.agnostic.{json,npz}` per final run, written once; the pass must match the recorded protocol score's stamps and its agnostic AP within 0.002, the captured arrays (capture order) must reproduce the pass's agnostic AP exactly (1e-9), and the tie-broken arrays in exam key order, the bootstrap's input, must lie within 0.02 of the recorded value (they differ only in how equal confidences rank: up to 0.0103 for a species' AP on a real dev exam, 2026-09-29), the shift recorded.
- `inc2/baseline.py`: `e1_record` (a baseline build on one of the two manifests a complete splits v3 summary records, by sha256, trains `cold_budget`, records `recipe_name`, `budget` and `e1` in exp.json and is priced by `budget_cost`; another arm than m640 refuses); verbs `rescore-agnostic` (L23E: every final run of both arms, then the verdict; `<exp>/agnostic_rescore.json`), `agnostic-verdict` and `e1-test-read`.
- `run_inc2_build.sh`: `inc2.base3 build` (name `base3_v3`, `HF_HUB_OFFLINE=1`, `YOLO_OFFLINE=true`, torch, transformers, Ultralytics, scipy and cv2 checked, no advance) and `inc2.baseline rescore-agnostic` (name `agnostic_<exp>`, no advance); the drift check covers `base3.py`, `base3_v1.json`, `scorer_agnostic.py` and `stream_thresholds.json`.
- Autopilot:
  - levers **L23V** (`inc_build_base3`) and **L23E** (`inc_rescore_agnostic`), MAINT, R3 in the envelope, record only (a failure is a card and is not proposed again; an uncertain submission is followed by its job name, `inc_build_base3_v3` and `inc_build_agnostic_<exp>`, and its record): `stream_levers.json`, `brain/policy_actions.json`, `brain/approvals.ENVELOPE_ACTIONS`, `executor` (ARGV_FORMS, STREAM_REMOTE, STREAM_ENVELOPE_LEVERS, timeouts), `stream_remote` (build grammar, job names, `agnostic_rescore.json` among the records, `splits/v3/summary.json` and `capacity/e1_v1.json` in the summary, `tools/inc2/*.json` in the S23 module hashes), `evidence.ALLOWED` (those three files, never the report, a manifest or a holdout list);
  - the role enum of `inc_build_baseline_v2` and of the build grammar admits `baseline`;
  - `stream_domains/weed.json`: baselines `e1_a` and `e1_b` (measure, `requires: base3`, `native: false`, their budget) and the block `e1`;
  - `stream.py`: `/stage/base3` (done once the shipped summary says complete) and `/stage/agnostic`; the L23V and L23E items; `RECORD_ONLY_LEVERS`;
  - `diagnose_stream.DR0`: a measure item that requires base3 waits for it; when it is next and splits v3 is missing, L23V is proposed once; an item marked `native: false` gets no L23N; once both E1 arms are done, L23E once;
  - `levers_stream.price`: an item with a budget is priced from it.

### Choices where the plan was silent or the spec check asked, and why

1. *The "any box > 90 %" rule* reads weed boxes: a crop box that fills a close-up of a crop says nothing about the weed boxes beside it.
2. *The source rule* is computed on the source as delivered (after the class mapping, before the image rules), so a source cannot pass by losing its worst images first.
3. *min 30* holds for every included source, so a tiny source (rf_kinjj) contributes mostly to test v1. That is the rule as written.
4. *base_v2 is exempt from every rule*, not only the mask and the 320 px rule, so that arm A is base_v2 whole.
5. *The smoke run* became the build's loader measurement and walltime guard (above).
6. *The embedding check* runs on external rows only; base_v2 was scanned at LOCK v2.
7. *The stream's pools* (read through the same ledger fold) extend the critique's "a group touching base_v2 is never held out": an image any current model trained on (inc0001's CottonWeedDet3 rows) never enters test v1. The first rehearsal held out 135 CottonWeedDet3 images before this rule was added.

### How it was verified

Locally, no GPU:
- `tests/test_inc2_base3.py` (new): the shipped config (the 16 dock slugs, MH-Weed16's 15 ids, the exclusions, the pinned numbers, malformed configs refused); geometry (polygons, clipping, √(w·h), each of the 8 box transforms against `funnel.leak.dhash_variants`' image transforms, one-to-one layout matching); the pair search against brute force at 3 and 6 bits; a full build in a synthetic world (arm A whole and byte-identical inside B; both manifests pass `inc2.train.check_manifest` and `guard_rows`; crop boxes dropped, a sub-8 px box masked with the PNG equal to `load_image` + `inc2.mask`, an 800 × 600 image trained as a 640 × 480 PNG that reads back equal, the 90 % / 320 px / no-weed / no-label drops; the convention rule, the quarantine, unknown class names and a fail-closed dHash leak each excluding a source; a base copy dropped; an unreleased intake hold kept out; exact, flip and identical-layout copies deduped toward the lower tier while a flip copy whose boxes do not follow the flip is kept; whole capture groups held out, never one touching base_v2, a Roboflow stem joining one group across the family; the dock cap; no non-dev key in summary.json; a second build refused); D28-v2 with pair cosines (0.30: only the hit image leaves; 0.97: the source leaks); `count` read-only (nothing under INC_DIR), the as-is and lifted scenarios, an unquarantine event admitting the source, a deterministic holdout, fail-closed without a stream ledger or Step 1's queue; the walltime guard (`over_walltime`, no E1 arm); `inc2.baseline` building both arms with `cold_budget` (the pinned driver accepts the definition) and the cold table for any other manifest.
- `tests/test_inc2_e1.py` (new): the recipe formula at every N from 1,000 to 200,000 (exactly 640 warmup iterations), round-half-up, only m640, base-only acceptance; `inc2.train` in production (cold_budget passes stage recipe and stops at device; one epoch off, a changed budget record, a chain or a union refused); `guard_verdicts`; the agnostic scorer with real CPU passes (refusals, written once, arrays reproducing the recorded agnostic AP, nothing asked of test); the bootstrap (deterministic, SE 0 for identical runs, equal to an independent recomputation); the verdict rule (each condition alone, pooled sd, pending, mismatched summaries, swapped arms, test-mode files refused; the decision file dev only); `rescore-agnostic` end to end; `e1-test-read` (refused before the verdict, specs the v2 executor accepts, refused once a test score exists or the verdict changed).
- `tests/test_stream_ap_units.py` (new `t_e1`): prices, the build grammar (including `--role baseline`), the allow-list, and the platform's sequence: L23V once with narrow cites, E1-A then E1-B priced from the budget, no L23N for them, L23E once both are done, DR0 silent after; a failed L23V is one card, no pause, no held lane, no E1 arm; an uncertain L23V is followed by its job name and summary.json.
- `tests/test_stream_ap_replay.py` (stream_r0): R0 now ends with L23V, E1-A, E1-B (role baseline), the five native rescores, then L23E, each once, within the envelope.
- `tests/test_stream_pipeline.py`: the platform proposes L23V once; the real `inc2.base3 build` refuses in that world (no dataset registry); one card, the stream runs on, no E1 arm.

### Rehearsal on the cluster (read-only, `count`)

`python -m weed_optimizer_framework.tools.inc2.base3 count --stream weed_stream_v1 --out /jet/home/byler/e1_count2` on the Bridges-2 login node, 2026-10-03 09:23 UTC, 1,121 s, from this commit's code (config sha256 `a077e116…`; registry `eb66759c…`, base_v2 `6f54fa14…`, LOCK v2 `1a34ba87…`, the stream ledger head `acba1337…` with 12 quarantined sources and the pools P_0 + P_1, 7,493 rows; Step 1's queue read, 96,508 rows). It reads stored hashes of the originals (the v1 pool's 8-variant dHashes; 30,635 small files hashed on the spot), runs the guard on them, and writes nothing under INC_DIR. The build adds the embedding detector and D28-v2's pair cosines on the files it writes; both can only drop more rows. 1,566 rows of arm B had no stored variants (large files) and are judged there.

"As is" applies the stream's quarantines; "lifted" assumes a person lifted rf_tuf, rf_zbm50, rf_srec fqrtg and rf_weed-tnf9e (the four quarantined sources that the config includes).

| source | files | in B, as is | test v1, as is | in B, lifted | test v1, lifted | what removed the rest (as is) |
|---|---:|---:|---:|---:|---:|---|
| base_v2 (arm A, exempt) | 6,811 | 6,811 | 0 | 6,811 | 0 | nothing |
| MH-Weed16 | 5,000 | 4,590 | 400 | 4,590 | 400 | 7 without a weed box, 2 base copies, 1 duplicate |
| rf_tuf | 8,700 | 0 | 0 | 4,784 | 400 | D28 quarantine (6,768); 1,843 without a weed box; 88 with a box > 90 % |
| rf_zbm50 | 1,977 | 0 | 0 | 1,666 | 296 | D28 quarantine (1,970) |
| dock family, 16 slugs (*Rumex obtusifolius*) | 39,903 | 5,496 | 1,471 | 5,466 | 1,501 | 32,290 duplicates (9 slugs are whole re-exports of one release: 17,507); 38 base copies; 608 without a weed box |
| MFWD (intake, POROL + Weed) | 4,079 | 1,883 | 0 | 1,883 | 0 | 2,021 duplicates (tray time series); 175 with no box left after masking; no capture group eligible for test v1 |
| rf_school | 2,071 | 1,748 | 308 | 1,746 | 310 | 13 duplicates, 2 base copies |
| gh_tehreemnoor (DeepWeeds boxes) | 2,006 | 1,354 | 239 | 1,354 | 239 | 245 with a box > 90 %; 167 without a weed box |
| CottonWeedDet3 (intake) | 795 | 727 | 12 | 727 | 12 | 40 intake holds; 16 with a box > 90 %; its 682 P_1 rows are never held out |
| rf_itmo | 787 | 565 | 100 | 564 | 101 | 122 without a forb box |
| rf_kinjj | 87 | 56 | 30 | 56 | 30 | 1 without a weed box; min 30 to test v1 |
| rf_srec fqrtg | 6,552 | 0 | 0 | 0 | 0 | convention rule: median side 28.4 px < 32 (also D28-quarantined) |
| rf_weed-tnf9e | 10,000 | 0 | 0 | 0 | 0 | convention rule: median side 16.4 px, 47.5 % of boxes < 16 px (also D28-quarantined) |
| **total** | **88,768** | **23,230** | **2,560** | **29,647** | **3,289** | |

- **Arm A**: 6,811 images, 25,703 boxes.
- **Arm B as is: 23,230 images** (107,534 boxes; 17,820 distinct photos after joining 6-bit near copies), test v1 2,560 images in 1,619 groups, 34,326 copies removed, dock family 5,496 (its cap, 9,549, does not bind). Recipe: 52 epochs, warmup 0.881543 epochs, close_mosaic 5; 1.21M image-epochs, about 4.8 GPU-h a seed.
- **Arm B lifted: 29,647 images** (114,842 boxes; 22,384 distinct photos), test v1 3,289. Recipe: 40 epochs, close_mosaic 4.
- Guard on the originals' hashes: 65,712 rows checked; 6 refused `near_eval_variant` (3 in rf_tuf, 3 in rf_zbm50: the two sources' D28 dHash hits) and 6 index cross-check hits; 6,894 `base_copy` verdicts = base_v2's own 6,811 rows + 83 external copies of them (42 dropped as is, 83 lifted).
- CottonWeedDet3: 682 of its rows are inc0001's (P_1), so they stay in arm B and never enter test v1 (choice 7); the first rehearsal, before that rule, held out 135 of its images, this one 12.
- MFWD: its 21 intake capture groups (at most 298 images each) join through 6-bit near copies into groups none of which is eligible (over 400 images or touching base_v2; the report does not separate the two), so MFWD gives nothing to test v1.

**Why arm B is under 30,000, and what would raise it without changing a rule.**
- The D28 quarantines: rf_tuf and rf_zbm50 hold 6,450 arm-B images and 696 test-v1 images. Lifting them is a person's call after D28-v2's sidecars weigh their 3 + 3 dHash hits (the D28-v2 amendment found every quarantined source's hits to be dHash false positives, pair cosine at most 0.744): **29,647**, 353 short of 30,000.
- The convention rule excludes rf_srec fqrtg (6,552 files, median box side 28.4 px at 640) and rf_weed-tnf9e (10,000 files, median 16.4 px, 47.5 % of boxes under 16 px); lifting their quarantines changes nothing.
- Copies: the 16 dock-family slugs (39,903 files) are Roboflow exports that largely share one *Rumex* release; 32,290 files are copies (nine slugs add no image at all), leaving 6,967 (5,496 train, 1,471 test v1). MFWD loses 2,021 of 4,079 to near copies.
- Test v1 takes 2,560 images from the included sources (the 15 % holdout, min 30, max 400 per source).
- What raises it: new admitted supply through intake and Step 1, added to the config's allow-list as `base3_v2.json` (a source list, not a rule): the SIU set `zenodo_15808623` (about 203K images; reopened, and commit ce98b15 on main, after this branch's base, lets it reach Step 1's queue) and the full MH-Weed16 release (the registry holds a 5,000-image subset). How many of their images pass the rules is not measured yet.

### Deploy

These files change `executor.code_hash()` and the stream rules version: the autopilot modules, `stream_domains/weed.json`, `stream_levers.json`, `brain/policy_actions.json`, `brain/approvals.py`, `diagnose_stream.py`, `levers_stream.py` and the replay test. `inc2.train` hashes every `tools/inc2/*.py` into each run's drift check, so sync the lab and both cluster copies (nested and outer) from one commit, then run `executor.run_replay_tests` so envelope grants resume. The build needs the calibration's DINOv2 in the Hugging Face cache (as the Step 1 jobs do).

### Open items

- The 12 D28 quarantines stand until a person lifts them after D28-v2's sidecars clear them; rf_tuf and rf_zbm50 are included sources only once lifted; rf_srec fqrtg and rf_weed-tnf9e fail the convention rule whether lifted or not.
- A later never-train index v3 should cover `splits/v3/test_v1/*.jsonl` (and the test groups) before test v1 is frozen.

### Revision 1 (2026-10-03, before any build or run)

Decided under the same grant, before L23V was deployed or base v3 built. `base3_v1.json` is revised in place (summary.json pins the revised file's sha256, `209d902a…`); nothing was built from the first version. Where this revision and the text above differ, this revision holds.

**Why.** A read-only adversarial rehearsal of the first version (b3c804a) on the cluster and a mutation test found:
1. *Test v1 depended on the quarantine and on row positions.* Groups were named after row indices and drawn by a seeded shuffle of the sorted candidates, and quarantined rows were outside the draw. Lifting rf_tuf and rf_zbm50 reshuffled it: 1,312 of the 2,560 as-is test images were in the lifted arm B, so "build, then lift and rebuild" would have trained on half of test v1. No build read an earlier build's test lists.
2. *The new evaluation groups were excluded by name only.* The guard checks LOCK v2's index. In the lifted arm B, rows within 6 dHash bits of: sesame 279 (276 of them rf_tuf; 26 within 3 bits), Latvia 253 (rf_tuf 154), NDSU 162 (137), paddy 122 (115; 13 within 3 bits), chilli 29 (26); as is: Latvia 77, NDSU 13, paddy 5, chilli 4, sesame 1.
3. *A splits v3 manifest fell back to the 100-epoch cold table without a word* when summary.json was not complete (over_walltime, moved, edited): about 23K × 100 × 12.65 ms ≈ 8.2 h, past the 8 h limit, and not an E1 arm. Only DR0's state check stood in the way, and a finished L23V job counted as done whatever its summary said.
4. *The 400 cap held per owning source, but every row of a held group was held*: rf_unitec 555 test rows, rf_vizerion 464; the dock family lost 1,471 of 6,967 rows (21 %) to test v1.
5. *inc2.train accepted `cold_budget` for any baseline* whose exp.json named it with the pre-registered constants.
6. *An identical layout merged unrelated photos*: 7 pairs of base_v2 photos (tsw23) more than 10 dHash bits apart; harmless for base rows, which are all kept, but the rule could drop unrelated external rows.
7. 22 of 35 mutants of the builder, the verdict and the platform survived the tests (among them: test-v1 groups without the 6-bit components or the variants, the intake capture group or the file-name session ignored, dedupe at 6 bits, the lower resolution kept, each source rule off, the masked-area rule off, an unknown class id read as a weed, the cross-check and embedding refusals ignored, pool rows matched by key only, the 400 cap overshot, the 15 % doubled, the loader-bound walltime, the 2 pooled sd condition, L23E on one arm, an over_walltime summary read as done).

**What changed.**
- *Test v1 is computed before the quarantine.* Every step up to and including the holdout (dedupe, capture groups, the draw) ignores the quarantine. A quarantined source's held rows stay in test v1 (its list rows say `quarantined_source`); its other rows stay out of arm B; a copy group whose kept row is quarantined keeps its best copy from a source that is not. Lifting a quarantine changes arm B only.
- *Groups are named and ordered by content.* A group's digest is the sha256 of its members' sorted original sha256s (count now reads every original's sha256); per source, groups are drawn in the order `stable_int("inc2/base3/test_v1/<source>/<digest>")`, and the dock cap drops groups in the order `stable_int("inc2/base3/test_v1/cap/dock/<digest>")`. A new source whose photos join no existing group leaves every other source's test rows unchanged; so does reading the rows in another order.
- *Caps per source.* A source's held count, and the count its target is met by, include its rows held in other sources' groups; a group is taken only while every source with rows in it stays at or under 400. A group over 400 images, or touching base_v2 or a pool row, is never held; summary.json counts each source's rows in such groups by reason (`over_max_group`, `base_or_pool`).
- *Earlier test lists are never-train.* Every `splits/*/test_v1/*.jsonl`, whatever its directory is now called, is read: a candidate with the same original or written bytes, the same key, a shared capture relation (each list row now stores its `capture_keys`), or a dHash within 6 bits under the 8 variants is `prior_test`. Its group is drawn first, and held whatever the target (subject to the caps and eligibility); otherwise the row is dropped (`prior_test_v1`). A build refuses while `splits/v3/test_v1` exists without a summary.json (a build that did not finish): it is moved aside inside `splits/` first, and its lists are then read as never-train.
- *Evaluation-group guard* (config `evaluation_groups`, 6 bits): NDSU (ImageWeeds, weed_crop_detection, imageweeds_aerial, greenhouse), Latvia (Latvia, francesco_weed_crop_aerial, peradeniya, vitif246x, test-8qezo, project-5nvic, kg_vinayakshanawad), sesame (ravirajsinh45, leopard, poxtn, zbfhf), maize, PAGS8 (weedai 5c78d067), and the OOD-dev paddy and chilli; each slug must also be excluded. Their 8-variant hashes come from the v1 Step 1 pool, else are hashed from the registry's `local_path` (build and count alike). A candidate within 6 bits in either direction (its variants against the image's dHash, the image's variants against its dHash) is dropped (`near_eval_group`): never in arm B, never in test v1. summary.json counts the rows within 3 and within 6 bits per group and source (base_v2's counted, never dropped) and names the slugs it could not read.
- *Identical layout*: a copy only when the two dHashes also lie within 10 bits under one of the 8 variants (`rules.dedupe.layout_max_bits`).
- *E1 manifests train only as E1 arms.* `inc2.baseline` refuses any build on a manifest under splits/v3, or whose sha256 any base v3 summary records (any status), unless it is a complete summary's E1 arm built as role baseline (`base3.v3_claim`); a union containing one is refused. `inc2.train` runs `cold_budget` only when exp.json's `e1` record names arm A or B, its base manifest is that arm of the complete summary (`base3.e1_arm_of`), and the summary still hashes as the record says (`e1_problems`). The stream's `/stage/base3` is done only when the shipped summary says complete: an over_walltime summary reads as `over_walltime`, a finished job without a complete summary as `unconfirmed`; neither builds an E1 arm or proposes L23V again.
- summary.json also records the quarantine, the sources not read and why (unknown class names), the evaluation-group record, the earlier test lists read, the five largest capture groups; `dropped_v1.jsonl` carries each row's group, evaluation-group distances and prior-test match.
- Unchanged, as disclosed above: the 0.02 tie tolerance of the agnostic arrays (about the size of the expected effect; each file records its shift) and the √(w·h) box side that admits MFWD.

**MFWD and test v1.** MFWD's 3,848 candidate photos form one capture group with 21 other rows chained in at 6 bits (17 from the 16 dock slugs, 3 from rf_zbm50, 1 from rf_school; 3,869 rows): every photo shows the same greenhouse set-up. On the first rehearsal's rows each MFWD photo had a median of 128 others within 6 bits, and at 3 bits one component still held 3,421 of them. The component is over 400 images (it touches no base_v2 or pool row), so under the 6-bit rule no MFWD row can enter test v1 without dropping nearly all of MFWD from arm B. A pot-level test split for MFWD would be a rule of its own (open item).

**How verified.**
- `tests/test_inc2_base3.py`: each rule alone on synthetic rows (the source rule's big-box share, boxes per image and small-box share; the masked area; an unmapped class id; a cross-check hit; an embedding refusal; the loader-bound walltime); the selection on synthetic rows (4–6 bit pairs kept but grouped; a pair near only under a variant grouped; the higher resolution kept; base_v2 copies kept; capture group, session and family stem each join a group; identical layout within and beyond 10 bits; content names; exactly 15 % of singletons; no group past any source's max, rows in other sources' groups counted against a source's max and target; a group over max_group never held; the quarantine, a new disjoint source and the row order leave test v1 unchanged; promotion of a clean copy; earlier test rows held first or dropped); the evaluation-group guard (both directions, 6 vs 7 bits, base_v2 counted); the pair searches against brute force; end to end: an evaluation-group copy dropped and counted, the quarantined source's held rows in test v1, the session, capture-group and variant joins, count's test v1 equal to the build's key by key (except src_leak, whose dHash hit count cannot weigh), one test v1 in both scenarios, a rebuild after lifting the quarantine holding the same test v1 and training none of it, a rebuild with a new row holding every earlier test row again or never training it, a partial build blocking a rebuild, inc2.baseline refusing an over_walltime summary's arm, a copy of it, an unrecorded file under splits/v3, a union, another arm and an edited summary.
- `tests/test_inc2_e1.py`: `cold_budget` refused at stage recipe without an e1 record, for a base manifest the summary does not record, for the other arm, under an over_walltime summary and after the summary changed; the 2 pooled sd condition alone (D 0.015 over SE 0.001 and 1 pooled sd, under 2).
- `tests/test_stream_ap_units.py`: no L23E while one E1 arm is done; a finished L23V with an over_walltime summary or none builds no E1 arm and is not proposed again.
- Mutation test: the 35 mutants, adapted to the revised code, and 30 more on it (the quarantine before the holdout, row-index group digests, no prior-first order, prior groups bound by the target, each prior match, the evaluation-group drop, one search direction, the max check on the owner only, a target counting own groups only, no promotion, any-distance layouts, each `v3_claim` branch, the baseline and union refusals, each `e1_problems` check, `unconfirmed`, the partial-build refusal, the earlier lists not read, count or build ignoring the quarantine, count without sha256s, max_group ignored, evaluation hashes not computed, base_v2 rows dropped by the evaluation guard): all 65 killed.
- The full test suite: every script passes except `test_brain_api.py`, which fails the same way on main.

**Rehearsal of revision 1 (read-only, the cluster).** The revised selection on the cluster's data, 2026-10-03 11:34 UTC, config sha256 `209d902a…`, the stream ledger head `94b381e1…` (12 quarantined sources; pools P_0 + P_1, 7,493 rows): the same steps as `count` (originals' stored hashes, 30,635 small files hashed on the spot, every original's sha256 read), then independent checks. `inc2.base3 count --stream weed_stream_v1` from the same code (`/jet/home/byler/e1_fix_out/count/e1_base3_count.json`, 12:03 UTC, 1,709 s) gives the same arms and the same test v1 keys (`holdout_same_in_every_scenario: true`). Guard: 65,712 rows checked, 6 refused `near_eval_variant`, 6 cross-check hits, 6,894 `base_copy` (as before). Evaluation groups: 39,898 images hashed (kg_vinayakshanawad is not in the registry); 1,611 candidate rows dropped within 6 bits.

| source | in B, as is | test v1, as is | in B, lifted | test v1, lifted |
|---|---:|---:|---:|---:|
| base_v2 (arm A) | 6,811 | 0 | 6,811 | 0 |
| MH-Weed16 | 4,576 | 400 | 4,576 | 400 |
| rf_tuf | 0 (quarantined) | 400 | 4,315 | 400 |
| rf_zbm50 | 0 (quarantined) | 288 | 1,618 | 288 |
| dock family (16 slugs) | 5,347 | 864 | 5,347 | 864 |
| MFWD | 1,866 | 0 | 1,866 | 0 |
| rf_school | 1,722 | 304 | 1,722 | 304 |
| gh_tehreemnoor | 1,352 | 239 | 1,352 | 239 |
| CottonWeedDet3 | 727 | 12 | 727 | 12 |
| rf_itmo | 558 | 99 | 558 | 99 |
| rf_kinjj | 56 | 30 | 56 | 30 |
| **total** | **23,015** | **2,636** | **28,948** | **2,636** |

- **Arm A** 6,811 images, 25,703 boxes. **Arm B as is 23,015** images (105,790 boxes; 17,886 distinct photos): 52 epochs, warmup 0.888889, close_mosaic 5. **Lifted 28,948** (112,760 boxes; 22,276 distinct photos): 41 epochs, warmup 0.707182, close_mosaic 4. Test v1: 2,636 images in 1,889 groups, the same keys in both scenarios.
- Checks, in both scenarios: no arm-B row within 6 bits (8 variants, both directions) of a test v1 row; no base_v2 row within 6 bits of a test v1 row; no arm-B or test v1 row within 6 bits of any evaluation-group image; no source over 400 test rows (rf_unitec 400, rf_vizerion 222); the dock family holds 864 of its 6,211 kept rows in test v1 (13.9 %, was 21 %). 1,568 arm-B rows (CottonWeedDet3 725, MFWD 843) have no stored variants here: GuardV2 does not judge them in a rehearsal and the evaluation-group guard sees their dHash only; the build hashes every file it writes, so it can only shrink arm B.
- Rows within 6 bits (3 bits) of each evaluation group, before the drop: sesame 412 (80), 406 (80) of them rf_tuf; Latvia 924 (53), rf_tuf 245, base_v2 138, the dock slugs 22–53 each (the same photos re-exported); PAGS8 390 (30), spread over the dock slugs, rf_zbm50 33 and MFWD 35; NDSU 364 (67), rf_tuf 200, base_v2 68 (66 within 3 bits: its NDSU rows); paddy 171 (22), rf_tuf 141 (22); chilli 34, rf_tuf 28; maize none. rf_tuf is near the evaluation groups throughout: lifting it adds 4,315 rows that pass this guard, and a person decides on that with these counts in hand.
- Against the first rehearsal (as is 23,230, lifted 29,647, test v1 2,560 / 3,289): the evaluation-group guard; the caps (rf_unitec 555 → 400 and rf_vizerion 464 → 222 test rows, both gaining arm-B rows); and the originals' sha256s: rf_zijian-peng's 2,311 candidate rows are byte copies of rf_unitec (2,153) and rf_cotton-weed-detect (156) files, which the first rehearsal, holding no sha256 for most registry rows, did not join through its stand-in key (it kept 533 of them in arm B). The build always reads the sha256s, so this corrects the count, not a rule.

**Open items (added).**
- A pot-level test split for MFWD (its photos are one 6-bit component).
- A card when an L23V job finishes without a complete summary (`unconfirmed`): today the stream waits silently.
- The never-train index v3 should cover every `splits/*/test_v1/*.jsonl` and the evaluation groups' images.

### Revision 3 and the build's pre-flight fixes (2026-10-03, before any build)

Decided under the same grant, before L23V was proposed or base v3 built. `base3_v2.json` keeps its name and is revised in place (`version` "v2 revision 3"; summary.json pins its sha256). Where this section and the text above differ, this section holds. A pre-flight audit of the build path, run while SIU's intake batch `i0004_zenodo_15808623` waited for Step 1's admit, found four defects.

**1. The build's time limit.** `run_inc2_build.sh` asked 4 h for every verb. The base v3 build has never run at its size: about 107K candidate rows, SIU's 40,000 frames (720 × 960) written as 640 px PNGs, the DINOv2 copy check on every external row, 39,898 evaluation-group images hashed; the read-only `count` alone took 28–61 min on 65,712 rows.
- The limit is now 12 h for every verb. sbatch reads `#SBATCH` lines at submission and a running job can only lower its own limit, so the script cannot give one verb a longer limit than another.
- Effect on the other verbs (stream build, milestone, fork, feasibility, bisect, splits, baseline, pilot4): Bridges-2's Slurm (priority/multifactor; age 10,000, fair share 1,000,000, QOS 5,000,000, job size 0, partition 0; read from `scontrol show config`) does not weigh the time limit, so priority is unchanged. The backfill scheduler (bf_window 7,200 min, bf_resolution 3,600 s) fits a 12 h request into fewer gaps than a 4 h one, so these jobs may start later than before. A build that hangs holds its V100 for up to 12 h instead of 4 h.
- Prices are unchanged: a build job is still priced at the stream domain's `build_job_hours` (4 GPU-h) and settled from sacct. `walltime.build_h` (D26) and the policy rows' text say 12 h. D26 reads a build's limit only with a recorded build duration, and the platform records none.
- The drift list hashes `tools/inc2/base3_v2.json`, the config the builder loads, in place of `base3_v1.json`.

**2. Increments in flight are pool rows for the holdout.**
- *The defect.* `inc2.base3.quarantined_sources` returned the rows of every accepted pool, and `select` never holds such a row out to test v1. A segment cut before the build (s002 is expected right after SIU's admit) trains on rows that no accepted pool holds yet. If base3 held some of them out and the segment were then accepted, the stream's next pool would hold test v1 images.
- *The fix.* Every row of an increment whose status has not released its rows now counts as a pool row, by key or by the sha256 of the image it trains on. That covers `in_segment` (cut, not committed), `suspect` (accepted, then rolled back) and any status the builder does not know (fail closed).
- *What is excluded.* `inc2.base3.released_status` names the statuses that are not counted: `data`, `stale`, `withdrawn`, every return disposition and a bisect's verdicts. A returned or released row reaches a pool again only through a new cut, and the cutter never cuts a test v1 row.
- *The record.* summary.json's `inputs.stream` records `in_flight` (each increment's status, segment and rows; the rows in all) next to `pool_rows`, plus `pool_marked_rows`, the candidate rows the build marked.
- *Failure.* An increment's rows sidecar that does not hash as its cut line records refuses the build.

**3. SIU's frames are grouped by video (revision 3).** The intake's capture group for `zenodo_15808623` is the single image. Without a video relation, frames of one video could be split between test v1 and arm B. Neighbouring frames are near copies that 6 dHash bits do not always join, so test v1 would depend on arm B.
- *The fix.* `intake.zenodo_15808623.group_regex` is `(?P<group>[A-Z]{5}_week_\d+_IMG_\d+)_frame_`. `intake_rows` applies an intake source's `group_regex` the way `registry_rows` applies a registry source's, to the row's original file name (`rel`, else the image's name). It sets `group_key` = `<source>|<video>`, so the capture groups, the holdout, the earlier-test-list match (each test list row's `capture_keys`) and the siu family cap all take whole videos.
- *Config check.* A `group_regex` that does not compile or captures no group refuses the config, for registry and intake sources alike.
- *Read on the cluster (read-only, 2026-10-03).* All 39,958 rows of `i0004_zenodo_15808623` match. There are 331 videos of 14–230 frames each (median 116; 264 with over 100 frames). The batch's `intake_cap` record says 230,899 eligible frames, 40,000 taken and 190,899 deferred.
- *Group sizes.* On the manifest's stored dHashes (the identity variant only), the 6-bit components alone number 19,939 (largest 518 frames; 85 already span 2–18 videos). With the video relation there are 218 groups: the largest hold 4,437 (35 videos), 2,393, 2,092, 1,579 and 674 frames, and 210 groups of at most 400 frames hold 27,169 frames.
- *Consequences.* SIU's part of test v1 is at most 400 frames, a few whole videos. The siu cap drops whole groups in its seeded order, so SIU in arm B can end below its cap by up to one group (4,437 frames on these numbers; the 8 variants can only join more).

**4. L23V waits, bounded, for a person to lift cleared quarantines.**
- *The defect.* After L17 eval-hits writes the D28-v2 sidecars, D28 may judge a quarantined source's hits chance: rf_tuf and rf_zbm50 hold about 5,900 arm-B images. Lifting a quarantine is a person's decision (`inc2.stream unquarantine`). DR0 would have proposed L23V on the next tick, and the build reads the quarantine when its job starts, so those sources would have been built out of base v3 for good.
- *What D28 reports.* D28's detail adds `lift_pending`: the sources judged chance that the stream itself still quarantines (its queue summary's `quarantined_sources`, the fold the build reads). The platform's own source record can still say quarantined after a person's unquarantine on the cluster, which is why the list is not D28's `cleared_quarantined`.
- *How DR0 waits.* While the list is non-empty, DR0 does not propose L23V. Its detail carries `lift_wait`: the sources, the first-seen time, the deadline and one exact command per source, `python -m weed_optimizer_framework.tools.inc2.stream unquarantine --source S --stream SID --decided-by human:<id>`. The ticker raises one card (kind `quarantine_lift`) naming them, deduplicated by its text.
- *How the stream remembers.* Diagnoses are recomputed every tick, so the stream state keeps the first-seen time (`stage.r0.lift_wait`, context `/lift_wait`, outside `/stage`, which some R0 items cite whole), with ledger lines `lift_wait` and `lift_wait_ended`. The first tick that sees a non-empty list starts the wait, a change in its membership does not restart it, an empty list ends it, and a D28 that could not be judged changes nothing.
- *The bound.* After `D28.lift_wait_hours` (12, stream_thresholds.json, with its why) from the first sight, L23V is proposed with the quarantine as it stands, citing only the lock and `/stage/base3` as before. Other R0 items (L23N, L23E) are not held by the wait.

**How verified.**
- `tests/test_inc2_base3.py`:
  - *the config*: the shipped revision's version and decided_by; SIU's regex on real frame names; a regex without a group or that does not compile refused, for intake and registry sources;
  - *`released_status`*: every status;
  - *`quarantined_sources` on a ledger with five increments*: in_segment and suspect counted, withdrawn and data not, an unknown status counted, a sidecar that no longer hashes refused;
  - *a full build with an increment in flight*: the rule would hold every group, yet none of its 4 rows is held out; summary.json records 4 in-flight rows;
  - *intake rows named like the real batch*: every frame of a video, including one under the source's `train/` folder, gets `group_key` `<source>|<video>`, and an unmatched name gets none. Each video is one capture group, the holdout takes whole videos, and without the regex 3 of 4 videos are split. The siu cap drops whole videos, never part of one.
- `tests/test_inc2_stream.py`: the drift list hashes `base3_v2.json`, not `base3_v1.json`; the time limit is 12 h.
- `tests/test_stream_ap_units.py` (`t_e1_lift_wait`), in the platform's world:
  - *D28 judges a quarantined source's hit chance (pair cos 0.31)*: no L23V; the first-seen time kept; one card with the exact command; an hour later still one card and the same first-seen time.
  - *The stream lifts it*: L23V is proposed with its usual two cites and the wait ends, even with the platform's own source record still saying quarantined.
  - *Not lifted*: no L23V before 12 h from the first sight, L23V at 12 h, still one card.
  - *Neither wait applies*: for a source judged chance that the stream does not quarantine, or one whose hit is a copy (pair cos 0.97, a leak).
- Mutation check: 9 mutants of the new logic, each killed by these tests. They covered:
  - the in-flight keys dropped;
  - the intake `group_key` left unset;
  - every status released;
  - the group-count check off;
  - the 12 h bound ignored;
  - `lift_pending` read from `cleared_quarantined`;
  - the wait off;
  - the first-seen time reset every tick;
  - no card.
- The affected suites pass with 0 failures: `test_inc2_base3.py`, `test_inc2_e1.py`, `test_inc2_stream.py`, `test_inc2_step1_stream.py`, every `test_stream_ap_*.py`, `test_stream_pipeline.py`, plus the policy and governance tests the policy texts touch.

**Deploy.** The changed files change `executor.code_hash()` and the stream rules version: `diagnose_stream.py`, `stream.py`, `stream_thresholds.json`, `stream_domains/weed.json` and `brain/policy_actions.json`. Sync the lab and both cluster copies from one commit, then run `executor.run_replay_tests` so envelope grants resume. `inc2/base3.py` and `base3_v2.json` are read only by the build job. Nothing here changes `inc2/step1_stream.py` or what `collect/` imports.

**Open items (added).**
- The stream's cutter matches test v1 by bytes, key and dHash only (`step1_stream.test_v1_rows` → `base3.mark_prior`), not by capture relation. A later SIU batch (190,899 frames are deferred) could hold other frames of a held-out video that lie more than 6 bits from every held frame, and the cutter would not stop them. `test_v1_rows` should pass each queue row's capture keys (the same `group_regex`) once `step1_stream.py` may change.
- After a person's unquarantine, the platform's own source record stays `quarantined`, and D28 keeps naming the source in `cleared_quarantined` (L23V is not held by it, since `lift_pending` reads the stream's quarantine).
- A read-only `count` after SIU's admit shows SIU's groups, its test v1 share and how far the whole-group cap falls below 35 %.

### Review fixes to revision 3 (2026-10-03, before any build)

Decided under the same grant, before L23V was proposed or base v3 built. Two reviews of revision 3 (correctness; liveness and side effects) found the defects below; each was reproduced in the test worlds before it was fixed. Where this subsection and revision 3's text above differ, this subsection holds. `base3_v2.json` is unchanged.

**1. A segment cut while the build runs.** `gather` read the stream's fold once, when the job started, and nothing stopped a cut during the build: base3 takes no stream lease, the cutter refuses only rows of test lists that already exist, and the ticker did not order L18 against L23V. Reproduced: a cut right after `gather` of 4 intake rows, all 4 then held out to test v1 by a build that finished `complete`. The 12 h limit widened that window.
- *The build reads the fold twice more.* Once the selection is made, before anything is written (`recheck_pool`): if the pools or in-flight rows changed, the rows are marked again and selected again from the state before the first selection. And once the test lists and their companions (item 2) are written, while the job holds the stream's lease (`final_pool_check`, `inc.driver.Lease` on `stream/<sid>/stream.lease`, the lease `inc2.stream`'s writing verbs hold to cut and build): a cut that began before the lists existed has written its ledger lines by then, and a later cut reads the lists. If a listed row, or a row sharing a held-out row's capture relation, is then in a pool or in flight, the build refuses and summary.json is never written (the lists stay as never-train). A lease held past 30 min refuses too (fail closed). summary.json records `inputs.stream.recheck` and `inputs.stream.final_check`.
- *The ticker orders the two jobs* (`StreamRun._cut_order`, at taking and at submission): L18 is not submitted while base v3 is submitted and not finished (`stage.r0.base3` running), and L23V is not submitted while a submitted segment's cut is not yet in the evidence (the stream ledger's `build` line of its experiment). Proposed or filed, neither holds the other; the lanes' submission order (TRAIN before MAINT) decides.

**2. Frames of a held-out video that the build drops before the grouping.** The video groups covered only the rows still standing at `select`. Rows dropped earlier (an intake hold not yet released, a box or image rule, an evaluation group, a duplicate in a held group) are in no group, and Step 1's queue may still offer them to the cutter, which matches a test list by bytes, key and dHash within 6 bits, never by capture relation.
- *Companions.* Every row that is not a test row but shares a capture relation with a held-out row (export stem, intake capture group, file-name session: SIU's video), or lies in a held-out group, is listed in `splits/v3/test_v1_companions/<source>.jsonl` with its key, hashes and capture keys. `prior_test_lists` reads them with the test lists, so they never train in a later build, and `step1_stream.test_v1_rows` (unchanged) never offers them to the cutter: SIU's Step 1 keys equal its intake keys, so the key match holds. summary.json counts them (`holdout_v1.companions`). A build that did not finish and left companions blocks a rebuild as its test lists do.
- *A pool frame dropped before the grouping.* A group is now a pool's group, never held out, also when it shares a capture relation with a pool or in-flight row that a rule dropped before the grouping (a video with one frame in a pool is not held out).

**3. Masked rows in a pool or in flight.** A masked row trains on a masked copy, and its sidecar's `sha256` is that copy's; Step 1's registry keys (`slug__split__stem`) are not this builder's (`slug__rel`). `quarantined_sources` now also takes, for every masked row of an accepted or in-flight increment, its original's path (`unmasked_image`) and that file's sha256; `mark_pool` matches the candidate row by key, original sha256 or original path. summary.json counts them (`inputs.stream.masked_originals`: rows, hashed, unreadable). This also closes the same gap for accepted pools, which predates revision 3.

**4. A frame name the video pattern does not match.** It was left ungrouped silently. Such an intake row is now dropped (`group_unmatched`: its video is unknown, so neither test v1 nor arm B can take it), and each intake batch's record counts the matched and unmatched rows per source (`inputs.intake.<batch>.group_regex`); a registry source's counts are in its `per_source.<slug>.registry.group_regex`. All 39,958 rows of `i0004_zenodo_15808623` match, so nothing changes for it.

**5. The lift wait.** Revision 3's wait had five faults:
- a tick on which D28 raised proposed L23V (in the platform's world it was granted and submitted the same tick), and a queue summary without `quarantined_sources` read as an empty list and ended the wait;
- a tick whose snapshot could not be read ended the wait, and the next tick started a new 12 h with a second card, so the bound did not hold;
- the clock started when D28 first listed sources, whatever DR0 was doing, so it could run out while DATA work was due and no card had been raised;
- the card replaced an escalation raised the same tick as the current card, and its kind was not in the alarm's list, so the page's alarm fell from warn to ok;
- the card offered "or decides to keep it", which nothing records.

Now:
- D28's `lift_pending` is None (unknown, never an empty list) when the queue summary cannot be read or does not state `quarantined_sources`, and it lists only quarantines cited `D28` (or with no cite recorded): a quarantine D31 or a person cited is not D28's to reconsider.
- `diagnose_stream.lift_wait` returns a state. *waiting*: L23V waits, on D28's list, on the recorded wait's sources while D28 cannot judge, or, with no wait recorded and D28 unknown, on the unknown (bounded the same way, with a card saying D28 cannot judge). *expired*: the bound passed, and L23V is proposed as it stands. *lifted*: D28, judging a readable queue summary, lists nothing, and L23V is proposed.
- The bound runs from the first tick DR0 deferred L23V; the ticker records that tick (`stage.r0.lift_wait`) and raises the card from it. The record ends only on *lifted* or once base v3 is no longer missing (running, done, failed); past the bound it is kept, marked expired, so it never starts again. A tick on which D28 cannot judge, or whose evidence cannot be read, changes nothing. Ledger lines: `lift_wait`, `lift_wait_expired`, `lift_wait_ended`.
- `_lift_wait` runs before the fired diagnoses' cards, so an escalation raised the same tick stays the current card, and `quarantine_lift` is among the kinds that make the alarm warn.
- The card says: to keep a quarantine, do nothing; L23V is proposed at the deadline with the quarantine as it stands. If `inc2.stream` answers that the stream is being written (its lease), run the command again a little later.

**6. L23V's price.** `zero_job` priced the build at `build_job_hours` (4 GPU-h) while it may now hold its V100 for 12 h, against the domain's rule that a price is never low. L23V has its own estimator, `base3_job`, at `cost.base3_job_hours` = 12.0 (the job's walltime), and its policy row's `est_gpu_hours` bound is 12.0 (it was 8.0, which would have refused the new price). The other build jobs ran within 4 h and keep `build_job_hours`; every build is still settled from sacct.

**How verified.**
- `tests/test_inc2_base3.py`:
  - `test_build_race`: a cut after `gather`: the build selects again, holds none of the 4 cut rows out (the unfixed code held all 4) and records both reads; with the fold unchanged nothing is selected again; a cut after the re-check (5 held rows) makes the last check refuse, with no summary.json and the lists left on disk; the last check takes and releases a free lease, and refuses when another writer holds it.
  - `test_companions_and_masked`: int_src__01 (tray0, its hold not released) is a companion of the held int_src__00 and a held group's duplicates are companions; `step1_stream.test_v1_rows`, unchanged, marks int_src__01 never-cut by key and leaves a row of no held group cuttable; a pool frame dropped before the grouping (box_over_90) keeps its whole video out of test v1 (the unfixed code held the other 3 frames); two masked in-flight rows, one matched by its original's path and one by a byte copy's sha256, keep their candidate rows and capture groups out of test v1.
  - `test_build`: a registry source's group_regex counts in summary.json. `test_intake_video_groups`: an unmatched intake name is dropped `group_unmatched` and counted (17 matched, 1 not). `test_prior_lists`: the companions are read with the test lists.
- `tests/test_stream_ap_units.py`:
  - `t_e1_lift_faults`: D28 raising while L23V waits, a queue summary without `quarantined_sources`, and an evidence-error tick each keep the wait with its first deferral and its one card; L23V comes 12 h after the first deferral, once, and the wait ends once base v3 runs; D28 raising from the first due tick starts a wait on the unknown with its own card, ended (lifted) when D28 judges again; a list present for 13 h while DATA work is due starts no wait until DR0 defers L23V; with a D28 leak the escalation stays the current card and the alarm warns, and the lift card alone warns; the card says to keep a quarantine, do nothing; a D31-cited quarantine starts no wait.
  - `t_e1_cut_order`: with both due, the segment is submitted and L23V waits until the segment's build line is in the evidence, then runs; with base v3 running, a due segment is not taken until base v3 is done.
  - `t_e1`: L23V is priced at 12 GPU-h and its policy row admits it.
- Every new check of `t_e1_lift_faults` and `t_e1_cut_order` that names a fault fails on the unfixed code (15 failures), and the race, the pool-frame rule and the companions fail there too.
- The affected suites pass with 0 failures: `test_inc2_base3.py`, `test_inc2_e1.py`, `test_inc2_stream.py`, `test_inc2_step1_stream.py`, `test_inc2_splits.py`, `test_inc2_baseline.py`, `test_inc2_train.py`, every `test_stream_ap_*.py` (the mutation suite included), `test_stream_pipeline.py`, and the policy, governance, levers and funnel suites `executor.REPLAY_SCRIPTS` runs.

**Deploy.** Revision 3's note that `inc2/base3.py` is read only by the build job was wrong: `step1_stream.load_queue` → `test_v1_rows` imports it at run time (`prior_test_lists`, `mark_prior`). This subsection changes what `prior_test_lists` reads (the companions, which exist only once base v3 is built), so behaviour is the same until then. Sync `base3.py` once the Step 1 admit job running now (47384911) has ended, so that job does not load a module that differs from its recorded hashes; as before, sync the lab and both cluster copies from one commit, then run `executor.run_replay_tests` (the stream rules version and `executor.code_hash()` move: `diagnose_stream.py`, `stream.py`, `levers_stream.py`, `stream_levers.json`, `stream_domains/weed.json`, `brain/policy_actions.json`).

**Open items (added).**
- *Order of s002 and base v3.* The stream's TRAIN lane had s002 (L18) filed when this was written. If s002 is submitted first, base v3 waits for its cut and then keeps every SIU video s002 touches (and every group joined to one) out of test v1; s002 can take up to 4 × 682 rows, mostly SIU frames once SIU is admitted, so SIU's share of test v1 may fall to about none. Building base v3 before s002 is cut avoids that, since the cutter then refuses test v1 rows and their companions. Which goes first is a scheduling decision (a person can hold s002's approval until splits v3 exists); the code only guarantees that neither job runs while the other's outcome is unknown to it.
- Frames of a held-out video that are not yet admitted (190,899 SIU frames are deferred) are in no companion list; the cutter still needs each queue row's capture keys (`step1_stream.test_v1_rows`) once `step1_stream.py` may change.
- Registry companions are matched by the cutter by bytes or dHash only (their Step 1 keys differ from this builder's), and a registry row dropped before materialisation may carry no original sha256.
- The SIU plants: each species has 22 videos (11 weeks × 2), and no IMG id appears in two weeks, which fits two pots per species filmed weekly (an inference). If so, a test v1 SIU video shares its plants with arm B videos of other weeks; grouping by video does not remove that.

## Amendment (2026-10-04): E2, the 12-class detector on E1-B's backbone (pre-registered)

Written before any E2 experiment was built or run, and before any E2 number existed, and revised before deploy after a review (the test read per qualifying arm, the verdict's parameters gating it, the report tied to the read, a failed build told from a failed submission, the build-ended budget release). Not edited afterwards, except for the last line of "How it is verified". Decided by the owner's delegate under the 2026-09-30 grant (`human:harry567566@gmail.com`).

E1's sealed test read (under Why) is part of E2's motivation. No E2 parameter was chosen from a test number: the arms, the recipe, the seeds, the reference and the rule are the ones fixed before E1's test was read, or the table's.

**Why.**
- The best sealed cwd12 test score is 0.8786 ± 0.0018. That is the 12-class mAP50-95 of b_v2_m640: YOLO11m at 640 on base_v2's 6,811 images. The gap to 0.90 is 0.021.
- Its weakest species on test are Carpetweed 0.736, SpottedSpurge 0.810 and Purslane 0.828.
- Five larger or newer detectors and larger inputs (l640, y26m640, y26l640, m832, s1024) gave at most +0.005 on dev. None qualified. On 6,811 images the detector is not the limit; the data is.
- E1 (amendment 2026-10-03) answered the box half of the gap. One-class weed detectors trained on base v3's 44,485 images (36,531 distinct photos) scored:
  - agnostic dev 0.8787 ± 0.0033, against 0.8570 ± 0.0014 on base_v2's 6,811. D = +0.0217, above 2 pooled sd (0.0050) and above the bootstrap SE (0.0068), so E1-B qualifies (`capacity/e1_v1.json`);
  - sealed test 0.8996 ± 0.0015, against 0.8838 ± 0.0018. The best 12-class model's boxes score 0.8901 there.
- No 12-class model uses those boxes yet.

E2 asks one question: does starting from E1-B's weights raise the 12-class score when the detector then learns the species on base_v2?

**Design (one variable: the init).**
- The 12-class labels stay base_v2's. Base v3's external boxes bring no species label into E2.
- E1-B trained every weed box as class 12 in the same 13-class INC space. Its whole network, head included, therefore loads into a 13-class model.
- Its classes 0–11 were trained only as negatives. E2 learns them on base_v2.

### Pre-registration

**Reference.** b_v2_m640 is not retrained; its stored scores are read. It is:
- YOLO11m at 640 on base_v2 (by LOCK v2's sha256), seeds 0, 1, 2;
- the cold recipe: 100 epochs, SGD, lr0 0.01, warmup 3 epochs, warmup_bias_lr 0.1, cosine to lrf 0.01.

**Arms.** Both arms are YOLO11m at 640 on base_v2 (by LOCK v2's sha256), seeds 0, 1, 2.
- *The init.* Arm-seed s starts from E1-B's final EMA weights of seed s: `e1_b_m640/runs/base__s<s>/weights/final.pt`. This is the base run's own file; `final__base__s<s>` links to it. It is recorded by absolute path and by sha256, which must equal that run.json's `weights_sha256`.
- **E2-W** (`e2_w_m640_seed0`, `_seed1`, `_seed2`): b_v2_m640's definition except the init. That means the same manifest, arm record, cold recipe key for key, LOCK v2 and never-train index.
- **E2-S** (`e2_s_m640_seed0`, `_seed1`, `_seed2`): the same data and init, with recipe x1b of the Protocol v3 table. x1b is 50 epochs, warmup 3, peak lr0 0.01, cosine to lrf 0.01, and warmup_bias_lr 0.01 (the table's convention for re-warm recipes). Question: does a shorter schedule keep more of the pre-training? x1b differs from cold in epochs (50 vs 100) and warmup_bias_lr (0.01 vs 0.1), and in nothing else.

**Six experiments, one per arm and seed.**
- Why: the pinned driver gives every base run of an experiment one init (exp.json `init_weights`, `inc/driver.py` `cold_init`), but seed s needs E1-B's seed s.
- So each E2 experiment trains one seed, and its `init_weights` is the absolute path of that seed's E1-B file. The driver passes the path into the base run's spec unchanged; the driver itself is not changed.
- exp.json carries an `e2` record:
  - the arm and the seed;
  - the init: E1-B's experiment, run id, path and sha256, the sha256 of its run.json and exp.json, its splits v3 summary sha256, its run's never-train guard record (LOCK v2 and index sha256, rows checked, refused, cross-check hits) and its research-only flag;
  - E1's verdict: its path, its sha256 and its decision keys;
  - the reference: the sha256 of its exp.json, its manifest sha256, recipe, role, seeds, finals and the training environment its base runs recorded (Ultralytics and torch versions);
  - what differs from the reference: `init_weights` (E2-W), or `init_weights` and `base.recipe` (E2-S).
- The build refuses any other difference in the keys that define training: the manifest and its image count, the arm record, the protocol stamp, LOCK v2 and its never-train index, the decision exam and, for E2-W, the recipe. It also refuses an E1-B built from another splits v3 summary than the one E1's verdict was decided on, and (production) a reference whose base runs do not share one recorded training environment, and an E1-B run whose guard record is not the current LOCK v2's with nothing refused: E2's dev verdict and its test read inherit that guard's judgement, including the sources a person un-quarantined on 2026-10-03.

**Declared differences from b_v2_m640's definition, besides the init and E2-S's recipe.**
- *Role `baseline`.* Finals are dev and ImageWeeds, never test. b_v2_m640 read test at R0 (role capacity). E2's test is read once per qualifying arm, after the verdict, by a person (below). This is an exception to P10, declared here as E1's was.
- *One seed per experiment* (above).
- *research_only.* An E2 model inherits E1-B's flag, since a model trained from E1-B's weights carries E1-B's rows (§8, P6). The flag is base_v2's flag OR E1-B's flag; an E1-B flag that is missing or unknown counts as true.

**What inc2.train checks on every E2 base run.** These come on top of every existing check (the manifest checks, the v2 never-train guard, materialisation, the final-epoch check, the locked scorer, the sidecar).
- The run's init is the e2 record's path and hashes to its sha256.
- E1-B's run.json still records that sha256.
- E1's verdict still holds the recorded decision (its decision keys; the file's sha256 is not compared, because `e1_verdict` rewrites `generated_utc` whenever it runs).
- The run's seed is the e2 record's seed.
- E2-W trains the cold recipe and E2-S trains x1b, key for key. x1b trains a base run only in an E2-S experiment.
- *The environment.* The run's Ultralytics and torch versions are the ones b_v2_m640's base runs trained with (`training_env_check`).
- *Whole load.* At Ultralytics' setup (`on_pretrain_routine_end`), before the first step, every tensor of the state_dict of the model it trains (parameters and buffers such as BatchNorm's running statistics; `num_batches_tracked` skipped; a DDP or `torch.compile` wrapper removed) must equal the init's by name, shape and value, and no tensor of the init may be left out (`init_transfer`). Ultralytics loads an init by `intersect_dicts`, which skips a tensor whose name or shape differs, so this proves the init loaded whole.

A production run refuses every departure; a testing run records it.

**Statistic.**
- *The number.* Each run's dev species_map50_95 (the 12-class mean AP50-95), read from its final run's dev score at 640 by `inc2.scorer_native`: `scores/dev@640.json` plus its per-image arrays.
- *How it is scored.* scorer_native runs the locked scorer's own code and settings as a library. It must reproduce the run's recorded protocol dev score within 0.002 for the mAP and for every class (`vs_protocol_score`), or it refuses. So D and its SE come from one pass.
- *The reference's files.* b_v2_m640's three files exist: L23N wrote them as the native verdict's reference, and `capacity/native_v1.json` records their sha256s. They are read, never rewritten.
- *Reported beside, not deciding.* The base runs' protocol dev scores (b_v2_m640: 0.8524 ± 0.0025).

**Rule (dev only, record only).**
- *D.* For each arm, over the seeds it shares with b_v2_m640 (0, 1, 2): D = mean(arm) − mean(b_v2_m640).
- *Qualifies when* D > 2 × pooled sd **and** D > SE(D). Pooled sd = √((sd_arm² + sd_ref²)/2), with sample sd.
- *SE(D)* is the native rule's paired image bootstrap (`inc2.baseline.native_bootstrap`, unchanged):
  - 1,000 resamples of the dev images under `stable_int("inc2/e2/species_se")`, one draw for every run;
  - per run and resample, each species' AP50-95 on the run's tie-broken per-image arrays, then the 12-class mean over the species with a GT box in the resample;
  - the mean over seeds per arm, then the arm minus the reference;
  - SE is the sample sd over the resamples.
- *Choice.* When both arms qualify, the larger D is E2's choice. A tie goes to E2-S, the shorter schedule.
- *Two comparisons.* Both arms are compared with one shared reference. With 3 seeds per side, the 2 pooled sd condition alone passes a null arm about 3.5 % of the time (t ≈ 2.45 on 4 df) and either of two null arms about 6.4 % (simulated, sharing the reference), before the SE condition. The rule is kept as written; the reader weighs a single qualifying arm with that in mind.
- *Refusals.* The verdict refuses when native files of an arm or of the reference disagree on the exam manifest, key order, locked scorer, a setting or Ultralytics' version. It also refuses (production) a test-mode file, a file not checked against its production protocol score, a native file whose weights are not its base run's, a run whose base run.json does not show the recorded init, the arm's recipe, the passed init and environment checks and the whole load, two experiments of one arm with one seed, experiments that start from different E1-B records, and an experiment built against another reference manifest.
- *Output.* The verdict goes to `capacity/e2_v1.json`, beside `e1_v1.json`. A decided verdict is never rewritten: a recomputation whose decision agrees keeps the file byte for byte, and one whose decision differs is refused. The decision compared is the status, the qualifying arms, the choice, `testing_allowed`, the bootstrap's seed text and resamples and, per arm, the seeds, the values, D, pooled sd, SE, the conditions, `qualifies` and the inputs' sha256s; so a file decided under other parameters (fewer resamples, test-mode files admitted) is refused by L23C's production recomputation rather than kept. The fields reported beside it (the protocol dev means, which a re-score attempt of a base run rewrites, and the cross-check against `native_v1.json`) never block a recomputation.
- *Effect.* Qualifying switches nothing. The stream's arm, pool and incumbent stay as they are.

**Reported beside the verdict, not deciding.**
- In `e2_v1.json` (dev only), for each arm and the reference:
  - dev agnostic mAP50-95 from the same files;
  - the dev AP50-95 of Carpetweed, SpottedSpurge and Purslane, with their bootstrap SE;
  - the protocol dev means;
  - the training environment of each run and whether it is one;
  - whether the reference's files are the ones `native_v1.json` records.
- In `e2_v1_report.{json,md}` (for people, not evidence): ImageWeeds 12-class and agnostic, from the finals.

**The test read (a person's step).**
- *When and which.* Once per qualifying arm, and only after `e2_v1.json` holds a decided verdict. E2's headline test number is the chosen arm's (the larger dev D, fixed before any test is read), so reading the other qualifying arm too cannot move the headline after seeing test; its number is reported beside, marked as not the headline.
- *Which verdict.* Only one decided under the pre-registered parameters opens a read or a report: `rule` is E2's rule, the reference is b_v2_m640, the bootstrap is `inc2/e2/species_se` with 1,000 resamples, and `testing_allowed` is false (production). A verdict a person computed by hand under other parameters (`e2_verdict(resamples=200)` to look early, or `testing_ok=True` past a production refusal) opens no sealed test.
- *Preparing it.* `inc2.baseline e2-test-read --e2 W|S` writes, before submitting anything:
  - one kind-final spec on exam test per seed, from the arm's base weights (the weights the verdict read, by sha256);
  - one submission list per experiment;
  - the three `run_inc2_job.sh` argvs.

  A person submits them.
- *Reporting it.* `inc2.baseline e2-test-report --e2 W|S` reports 12-class and agnostic test, mean ± sd over the 3 seeds, against b_v2_m640's final test files (12-class 0.8786) with the gap to 0.90, and says whether the arm is the headline. It writes `capacity/e2_test_<W|S>.{json,md}`, which the platform's evidence does not admit, with the sha256s it read. It reads only the read that was prepared: every input has its `e2_test_read.json`, made on the verdict input's weights; each test score names those weights; each reference score names its final run's weights; and every file, the arm's and the reference's, shares one `scorer_sha256`, `manifest_sha256` and `key_order_sha256`. A score from other weights (a spec edited and resubmitted after a failed job) is refused, not averaged.
- *Refusals.* The read refuses before the verdict or on one decided under other parameters, for an arm that did not qualify, and a second time: once prepared, or once a run, attempt or test score of the arm exists. It refuses when the weights no longer hash as the verdict read them, and a repeated call says so when the verdict changed since the read was prepared. It checks everything before writing anything.

**Platform flow.** The stream proposes everything up to the verdict.
1. *The six builds.* Once R0 is complete, the MAINT lane is free and DATA has nothing due, DR0 proposes the six builds one at a time, in the order W0, S0, W1, S1, W2, S2. Each is lever L23B: `inc2.baseline build --exp e2_<w|s>_m640_seed<k> --manifest INC_DIR/splits/v2/base_v2.jsonl --seeds <k> --arm m640 --role baseline --e2 W|S`. The next build is proposed once the previous one's experiment exists, so the builds take about six queue-plus-build cycles of one GPU-shared build job each; the experiments' runs overlap.
2. *Their gate.* A build is proposed only while `capacity/e1_v1.json` is decided and qualifies E1-B, and E1-B's experiment is done. Each item cites the verdict's `/qualifies` and `/exp`, E1-B's status, the lock and the item's own state. Whether E1-B's three base weights still exist and hash as recorded is the build's own check: a missing or changed file fails that one build (a card), and the other E2 builds wait behind it (item 4).
3. *Their runs* are measurement arms. They get no L23N. A blocked unit or a stale advance is a person's card, never a pause.
4. *A failed build.* Three cases, told apart by where the build ended:
   - *A submission that failed before anything was queued* (an sbatch socket timeout, squeue unavailable, a duplicate job name): the lane's ordinary failure, as on every other lever. The item is proposed again under a new id; no card.
   - *A build whose job ran and ended without its experiment because the build refused* (inc2.baseline's ERROR line in its provenance record), *or whose job was cancelled*: record only. One card with the refusal, `/stage/baselines/<id>` says failed, DR0 does not propose it again, and the lane's failure count is not touched, so a refusal that every E2 build would meet cannot hold MAINT.
   - *A build whose job was killed from outside* (NODE_FAIL, PREEMPTED, BOOT_FAIL, TIMEOUT, or no sacct state and no refusal line): built again under a new id, with `/stage/baselines/<id>` back to missing, up to `LOST_RUNS_MAX` (3) times in a row; after that, record only as above, the card saying so.

   E2's six builds are one group: while one of them is failed, DR0 proposes none of the others (one card, not six). The card says that L23C waits for all six experiments and gives the exact build command for a person; the wait ends once the failed item's experiment exists.
5. *The verdict job.* Once all six experiments and b_v2_m640 are done and the verdict's record is missing, DR0 proposes **L23C** once. It runs `inc2.baseline rescore-e2` as one GPU job of `run_inc2_build.sh`, job name `inc_build_e2_v1`. The job:
   - scores the six final runs on dev at 640, and any reference file that is missing;
   - writes `capacity/e2_v1.json`;
   - then writes `capacity/e2_rescore.json`: complete, the files' names and sha256s and the verdict's sha256, no path.
6. *L23C is record only.* A failure is a card, and L23C is not proposed again. A submission whose outcome is unknown is followed by its job name and `capacity/e2_rescore.json`.
7. *The card.* When the verdict is decided, the ticker raises one card naming the qualifying arms, each with its D, 2 × pooled sd and SE, the choice (E2's headline), and the exact `e2-test-read` and `e2-test-report` commands for each qualifying arm.
8. *A restart during L23C's submission.* When the lab stops between L23C's submission and the state write, the restart meets "already executed". L23C (like every record-only lever) is then followed by its job name and its record, as an unknown outcome is, so a failure of its job is still a card and `/stage/e2` says failed; before, the item was declined and a failed job left no card.

**Pricing.**
- An E2 build item is priced from its image-epochs at m640's measured 14.2 ms per image-epoch, as E1's arms were:
  - E2-W: 6,811 × 100 = 681,100 image-epochs = 2.69 GPU-h;
  - E2-S: 340,550 image-epochs = 1.34 GPU-h;
  - each item adds a seed's share of the finals (0.8 GPU-h prices an experiment of 3 seeds; one seed pays 0.27) and the build job (4.0): 6.95 GPU-h for an E2-W item and 5.61 for an E2-S item.
- The six items total 37.7 GPU-h, 24 of it the build jobs' price, which is settled from sacct.
- A build whose job ran and ended without its experiment releases its estimate once sacct settled the job: the stream writes a 0 SU marker (`inc:job<ID>:build_ended`, `budget.record_build_ended`), and `budget.committed` releases a build whose every job is both settled and marked. The job's real SU stays in spent (its sacct entry). Before, such an estimate (6.95 SU for an E2-W item) stayed committed for good while the job's sacct SU was spent as well.
- L23C is nine scoring passes at 0.25 GPU-h: 2.25. E2 commits 39.9 SU in all.
- exp.json's cost_estimate uses the same rate (`inc2.recipes.e2_cost`).
- A base run takes about 2.7 h for E2-W (b_v2_m640 measured 2.68 h) and 1.35 h for E2-S, under the pinned 8 h cold limit.

### What changed

- `inc2/recipes.py`:
  - E2's pre-registered constants (`E2_*`);
  - `e2_exp`, `e2_recipe` and `e2_cost`;
  - `deviations` and `match` accept x1b for kind base only, when the caller names it (`_named_wanted`, which replaces `_budget_wanted`; cold_budget unchanged).
- `inc2/train.py`:
  - `experiment_e2`, `e2_problems`, `e2_env_check` and `e1_verdict_path`;
  - `experiment_arm` accepts an init other than the arm's checkpoint only when the e2 record names it;
  - `init_check` aims at the recorded init (path, sha256, seed);
  - `experiment_budget` admits x1b for E2-S only and returns an E2 experiment's problems;
  - the environment check and the `init_transfer` check at setup (`init_transfer`, `require_whole_load`).
- `inc2/baseline.py`:
  - `build --e2 W|S` (`e2_record`, `e2_research_only`; inc2.train's `e2_problems` on the definition before the manifest is copied);
  - verbs `rescore-e2` (L23C), `e2-verdict`, `e2-test-read` and `e2-test-report`;
  - `rescore_native`'s per-file step factored out unchanged (`_native_one`).
- `run_inc2_build.sh`: `inc2.baseline rescore-e2` (name `e2_v1`, no advance).
- Autopilot:
  - `stream_domains/weed.json`: six E2 baselines (`requires: e1`, `native: false`, `e2`, `images`, `budget`) and the block `e2`;
  - `diagnose_stream.DR0`: the `e1` gate (`_e1_qualified`), E2's builds waiting as a group behind a failed one, and L23C;
  - `stream.py`: `/stage/e2`, L23C as record only (followed after a restart's "already executed" too), a measurement arm's build that ran and ended without its experiment as record only (a refusal or a cancellation) or built again (killed from outside, up to `LOST_RUNS_MAX`), the build-ended budget release, the E2 verdict card, and L23B's `--e2`;
  - `budget.py`: `record_build_ended` and its release in `committed` (`spent` returns `ended_builds`);
  - `stream_levers.json`: L23B's optional `--e2`, the L23C row, the envelope;
  - `levers_stream.py`: a budget-priced item pays its seeds' share of the finals; an argv placeholder may hold digits (`{e2}`);
  - `executor`, `stream_remote`, `evidence.ALLOWED`;
  - `brain/policy_actions.json`: the `inc_rescore_e2` row and `inc_build_baseline_v2`'s `e2` bound;
  - `brain/approvals.ENVELOPE_ACTIONS`.

### Choices where the plan was silent, and why

1. *Six single-seed experiments* rather than an init template in `init_weights`. A template would put a non-file in every spec's `init` and need expanding wherever an init is read. Six experiments leave the pinned driver's contract ("a weights path or name") whole.
2. *One flag, `--e2 W|S`*, rather than free `--init-from` and `--recipe` flags. Neither the init nor the recipe is a parameter an item may choose: the build derives both from E1's verdict and the arm.
3. *The statistic comes from the native dev files at 640.* The base runs' sidecars of b_v2_m640 failed (2026-09-29), so its only per-image arrays are the native reference files, and D and SE must come from the same pass.
4. *A tie in D goes to E2-S.*
5. *The whole-load check* compares the state_dict, not only the parameters: BatchNorm's running statistics are part of the init.
6. *The training environment* is checked at train time (a run refuses before it trains), not only read by the verdict: a confounded run would cost its GPU-hours.
7. *The finals exclude test*, as E1's did; each qualifying arm's test is read once, and the chosen arm's is the headline.
8. *A failed build of any measurement arm is record only* when its job ran and the build refused (or the job was cancelled), not only E2's: no lane waits for a measurement arm, and the next one would meet the same refusal. A submission that failed before anything was queued is retried as on every lever, and a job killed from outside is built again a bounded number of times.
9. *The stored domain budget is not read.* `budget.domain_budget()` still reads `db.DEFAULT_DOMAIN_CONFIG` (or a block its caller passes), never the domain's stored config. The symptom that motivated reading it (weed_stream_v1's 180 SU daily cap cut to the code default's 120, L18 filed for a person) was removed on main by 11c01b0 and the 2026-10-04 amendment of §6.6 (no default daily cap). Reading the stored config now would re-apply whatever `budget.daily_cap` the stored weed config holds; it may still hold the old 120 default, which cannot be checked without the lab's database, and that would bring back the throttle the owner removed. The read stays planned (§6.6, "Stored domain config"), with that check made first.

### How it is verified

- `tests/test_inc2_e2.py` (new): the constants, recipes, the build and its refusals (each leaving no directory; every key of the reference compared, the reference's seeds included), inc2.train in production (stops at device; every refusal at stage recipe), the whole load (a real 1-epoch CPU run; a 12-class head; a BatchNorm buffer; a DDP-style wrapper), the bootstrap, the rule on synthetic native files (each condition alone, pooled sd, the choice, a tie, pending, every refusal, dev only by `brain_plan.dev_leaks`, the evidence scrub and an exact-value check), the kept verdict (refused under other resamples or with test-mode files admitted), `rescore-e2` end to end with real CPU passes, the test read (each qualifying arm once; a verdict under other parameters refused) and report (the headline; every score tied to the prepared read), the CLI.
- `tests/test_stream_ap_units.py` (`t_e2`): the domain items, prices, argv and proposal ids, the grammar and the evidence list, the gate in five states, the six builds in order, L23C at six and not at five, its failure and its unknown outcome (a job queued past `BUILD_LOST_SNAPSHOTS` stays followed by its name), a restart after L23C's submission (its job failing, then completing), a build the build refused (one card, the others waiting, the wait ending once a person builds it, the estimate released), a cancelled build job, a job killed from outside (built again, then failed after three in a row; a retry that builds), a submission refused by an sbatch socket timeout and one meeting an unavailable squeue (the lane's ordinary failure, proposed again), an E2 D5, the verdict card with one and with two qualifying arms; the budget's build-ended release (settled and marked, never either alone).
- `tests/test_stream_ap_replay.py`: stream_r0 now ends with L23E, E2's six builds, then L23C, each once, within the envelope.
- `tests/test_stream_pipeline.py`: E2 is never built in a world that never reaches E1's verdict. `tests/test_inc2_stream.py`: `run_inc2_build.sh inc2.baseline rescore-e2`.
- An ad hoc mutation run on scratch copies of the package and tests: 36 mutants of the new logic, each killed by a failing check of the tests above. They covered:
  - inc2.train: any init accepted by `experiment_arm`; `init_check` without the sha256 or the seed; `e2_problems` without E1's verdict, E1-B's run or the seed and name checks; the verdict compared by its sha256; x1b without arm S; the whole-load refusal off; the comparison on the parameters only; the environment check off;
  - the build: no reference compare, no recipe compare for W, no splits v3 summary check, research-only not inherited, x1b for any kind;
  - the verdict: either condition off, the smaller D chosen, a tie to W, the kept file rewritten, the whole document compared;
  - the test read: no qualifying check, no choice check, attempt.json not checked, a spec written before every check;
  - the platform: the e1 gate ignoring `qualifies` or E1-B's status, L23C at five of six, a failed L23C marked agnostic, L23C followed by another job name, a failed measure build counted as a lane failure, a failed build read as missing, the finals not scaled, the card without the command, L23B without `--e2`.
- A second mutation run, after the review that revised the amendment: 29 mutants, each killed by a failing check (or, for one, an uncaught refusal) of `test_stream_ap_units.py` or `test_inc2_e2.py`:
  - the platform: record only for any failed measure build (a submission failure included); no retry of a killed job; a cancelled job retried; a retry leaving `/stage/baselines` building; a build refusal retried; no group wait for E2; "already executed" declined for a record-only lever; L23C followed by another job name; no build-ended marker written; `committed` not releasing a marked build, or releasing one sacct has not settled; the card naming the chosen arm's read only;
  - the test read: the chosen arm only; no check of `testing_allowed`, the bootstrap, the rule or the reference; the kept verdict compared without `testing_allowed` or the bootstrap; the report without its read record, the scores' weights, the shared stamps, the reference's weights, or the read record's weights against the verdict's;
  - the build: no comparison of LOCK v2, the never-train index, the arm record or the decision exam; no check that the reference has the seed.
- The affected suites pass with 0 failures: `test_inc2_e2.py`, `test_inc2_e1.py`, `test_inc2_native.py`, `test_inc2_baseline.py`, `test_inc2_train.py`, `test_inc2_base3.py`, `test_inc2_stream.py`, every `test_stream_ap_*.py` (the mutation suite included), `test_stream_pipeline.py`, `test_inc_ap_replay.py`, `test_inc_ap_governance.py`, `test_funnel_ap_replay.py`, `test_funnel_ap_mutations.py`, `test_funnel_domain_free.py`. After the review's fixes, on main's 2026-10-04 commits (54de0cb), the full suite (142 scripts) passes except `test_brain_api.py`, which fails the same way on main (a decision path reads the reviewer: `supervision_health.py`).

### Deploy

These files change `executor.code_hash()` and the stream rules version:
- `inc_autopilot/{executor.py, budget.py, stream.py, stream_remote.py, diagnose_stream.py, evidence.py, levers_stream.py, stream_levers.json, stream_domains/weed.json}`;
- `brain/{policy_actions.json, approvals.py}`;
- `tests/test_stream_ap_replay.py`.

`inc2/{recipes.py, train.py, baseline.py}` are hashed into every run's drift check and S23.

Sync the lab and both cluster copies from one commit that contains main's 2026-10-04 commits (no default time-based caps, `db.py` to the lab only: its pre-flight then runs `test_stream_ap_no_throttles.py`), restart the dashboard (`inc_dashboard.py`), then run `executor.run_replay_tests` so envelope grants resume.

Before autonomy is turned on, these read-only checks are made. Each one that fails would refuse every E2 build or run, or hold the item silently:
1. *Budget.* weed_stream_v1's campaign `remaining_su` and `domain_remaining_su` are each at least E2's 39.9 SU plus the next L18 (about 116 SU): "of the domain's" is a wait refusal that holds MAINT silently, and "left in the campaign envelope" pauses the stream. Since the §6.6 amendment of 2026-10-04 no daily cap or monthly window has a default; the campaign's `daily_cap_su` and `window_cap_su` read none once a person ran `stream configure ... --daily-cap-su none --window-cap-su none`. If either is still declared, `daily_remaining_su` and `window_remaining_su` must cover the same 39.9 + 116 SU, or a person clears or raises it. If an envelope is short, a person raises it.
2. *The reference's definition.* `b_v2_m640/exp.json`: `splits_v2.lock_sha256` and `nevertrain_sha256` equal the current LOCK v2's; `base.manifest_sha256` is base_v2's; `base.recipe` equals `inc2.recipes.cold('m640')`; `arm` equals `inc2.recipes.resolve_arm('m640', REPO)`, yolo11m.pt's sha256 included; the protocol stamps are v3, inc2, v2.
3. *The reference's files.* `b_v2_m640/runs/final__base__s{0,1,2}/scores/dev@640.json` and `.images.npz` exist and hash as `capacity/native_v1.json`'s `reference_inputs` record.
4. *E1-B.* `capacity/e1_v1.json` is decided, qualifies `e1_b_m640` and has `testing_allowed` false; `e1_b_m640/runs/base__s{0,1,2}/run.json` is done, `testing` false, its guard record names the current LOCK v2 with nothing refused, and `weights/final.pt` is a regular file that hashes to its `weights_sha256`.
5. *The environment.* `b_v2_m640/runs/base__s{0,1,2}/run.json` record one `ultralytics_version` and `torch_version`, and they are the cluster environment's now (E1-B's run.json show whether it changed since); otherwise every E2 run refuses at stage recipe. The rescore job's Ultralytics is the version stamped in the reference files (8.4.37); otherwise the verdict refuses on stamps.

### Open items

- E2 measures "E1-B as the init", not "data at scale" alone. E1-B's base holds base_v2's 6,811 images byte-identical, trained about 27 epochs as class 12, so E2 differs from b_v2_m640 in the init and in about 1.2M extra image-epochs, its own images included. E1-A (one class, 6,811 images) scored agnostic dev 0.8570, below b_v2_m640's own 0.8695, so against the 12-class model's boxes E1-B gains about +0.009 on dev and +0.0095 on test. An E2-A control (init from E1-A) would separate the two; it is not run.
- E1-B's test agnostic is 0.8996, under 0.90: boxes inherited from E1-B alone cannot carry E2 to 0.90 on test.
- E2 learns species from base_v2's 6,811 labels only. Species labels at scale (base v3's verified target boxes) are a separate experiment.
- x1b's warmup_bias_lr (0.01) differs from cold's (0.1), and close_mosaic 10 is 20 % of x1b's 50 epochs against 10 % of cold's 100. E2-S against E2-W therefore confounds the schedule's length with the bias warm-up and the share of epochs without mosaic: read E2-S as "the table's x1b", not "cold, shorter".
- Every E2 model is research-only if E1-B is (external rows of unknown licence): a qualifying E2 cannot become a deployable or incumbent model, the robot's included, without a separate licence step.
- E2 does not read test v1.

## Amendment (2026-10-04, later): E2-C, the attribution control (pre-registered)

Written after E2's amendment and before any E2 build, run or number existed. Decided by the owner's delegate under the 2026-09-30 grant (`human:harry567566@gmail.com`). It supersedes the first Open item above ("it is not run").

**Why.** A qualifying E2 arm differs from b_v2_m640 in two things at once: the pre-training data (base v3's 44,485 images, base_v2's 6,811 among them) and the one-class pre-training stage itself (about 1.2M extra image-epochs). Without a control, a gain cannot be credited to data scale.

**Arm.** E2-C (`e2_c_m640_seed0`, `_seed1`, `_seed2`) is E2-W's definition exactly (base_v2 by LOCK v2's sha256, the cold recipe key for key, YOLO11m at 640, seeds 0, 1, 2, role baseline, finals dev and ImageWeeds, every check of E2's build and of inc2.train), except the init: arm-seed s starts from **E1-A's** final EMA weights of seed s, `e1_a_m640/runs/base__s<s>/weights/final.pt`, recorded by path and sha256, which must equal that run.json's `weights_sha256`. E1-A trained base_v2's 6,811 images as class 12 under the same cold_budget (1.2M image-epochs) as E1-B. E2-W against E2-C therefore differs in one variable: the pre-training data (44,485 against 6,811 images at equal pre-training compute).

**Attribution rule (dev, record only).**
- D_data = mean(E2-W) − mean(E2-C) over seeds 0, 1, 2, on dev species_map50_95 from `inc2.scorer_native` at 640, as E2's statistic.
- SE(D_data): E2's paired image bootstrap, 1,000 resamples under `stable_int("inc2/e2/attribution_se")`.
- E2's gain is credited to base v3's data when D_data > 2 pooled sd **and** D_data > SE(D_data). Otherwise E2's result is reported as a property of one-class pre-training, not of data scale.
- When E2-S is E2's choice, E2-S − E2-C is computed the same way and reported as confounded by the recipe (E2-C trains the cold recipe).
- Reported beside, not deciding: E2-C − b_v2_m640 (two-stage pre-training without extra data), and dev agnostic and Carpetweed, SpottedSpurge and Purslane for E2-C.
- The record is `capacity/e2_attr_v1.json`. It never rewrites `capacity/e2_v1.json`: E2's qualification and choice stay exactly as pre-registered above, and E2-C can neither qualify nor be chosen.

**Test.** When E2's chosen arm's test is read, E2-C's test is read once as well, by the same person's step, and reported beside the headline (12-class and agnostic, mean ± sd over 3 seeds). No decision rests on it. If no E2 arm qualifies, E2-C's test is not read.

**Platform.** The stream proposes E2-C's three builds after E2's six (C0, C1, C2), gated on E1's verdict being decided and E1-A's experiment being done, and proposes the attribution record once E2-W's and E2-C's runs are scored. The implementation follows in its own change, deployed before any E2 dev score exists; E2's six builds and L23C proceed as written above meanwhile. Price: as an E2-W item, 6.95 GPU-h per build item, 20.9 for the three.

### What changed

- `inc2/recipes.py`:
  - `E2C_ARM`, `E2C_EXP`, `E2C_INIT_EXP` (`e1_a_m640`), `E2C_DECIDED_BY`;
  - `E2_BUILD_ARMS` (W, S and C) and `E2_BUILD_EXPS`, the letters a build takes. `E2_ARMS` stays E2's two arms, the only ones E2's verdict reads;
  - `E2_INIT_ARM` (W and S start from E1-B, C from E1-A);
  - `e2_exp`, `e2_recipe` and `e2_cost` take C: E2-W's cold recipe and E2-W's price.
- `inc2/train.py`: `e2_problems` takes arm C. E1's verdict must be decided, still hold the decision the record holds, and name the init's experiment as its arm A (its `reference`, the pre-registered `e1_a_m640`). E1-A's run must still record the weights. The init check, the environment check and the whole load read the e2 record, so they apply to E2-C unchanged.
- `inc2/baseline.py`:
  - `build --e2 C` (`e2_record`): E2-W's checks with E1-A in place of E1-B. E1's verdict is decided and names `e1_a_m640` as its arm A. E1-A is E1's arm A on m640 with cold_budget, built from the splits v3 summary the verdict was decided on. Its run is done and production, under the current LOCK v2 with a clean guard, and its own final.pt hashes as its run.json records. The model's research-only flag is base_v2's OR E1-A's;
  - `rescore-e2-attr` (L23D) and `e2-attr-verdict` write `capacity/e2_attr_v1.json` and `e2_attr_v1_report.md`. `rescore-e2-attr` then writes `capacity/e2_attr_rescore.json` (`e2_attr_decision`, `_e2_attr_side`, `_e2_attr_pair`, `_e2_attr_canonical`);
  - `e2-test-read --e2 C` and `e2-test-report --e2 C` (`_e2c_gate`, `_e2_attr_doc`).
- `run_inc2_build.sh`: `inc2.baseline rescore-e2-attr` (provenance and lock `e2_attr_v1`, no advance).
- Autopilot:
  - `stream_domains/weed.json`: three baselines `e2_c0`, `e2_c1`, `e2_c2` after E2's six (`requires: e1`, `e2: C`, priced as an E2-W item) and the block `e2_attr`;
  - `diagnose_stream.DR0`: `_e1a_done` on top of E2's gate for E2-C's builds, and L23D;
  - `stream.py`: `/stage/e2_attr`, L23D as record only (phase E2_ATTR, followed by its job name and record after an unknown outcome or a restart), the card of a failed E2-C build, and the attribution card;
  - `stream_levers.json`: the L23D row and the envelope;
  - `executor` and `stream_remote`: the grammar, the job name `inc_build_e2_attr_v1`, `--e2 C`, and the stream summary's two new files;
  - `evidence.ALLOWED`: `capacity/e2_attr_v1.json` and `capacity/e2_attr_rescore.json`;
  - `brain/policy_actions.json`: the `inc_rescore_e2_attr` row and `--e2 C`;
  - `brain/approvals.ENVELOPE_ACTIONS`.

### Choices where the amendment was silent, and why

1. *Two letter sets.* `E2_ARMS` stays `{W, S}`. E2's verdict, L23C, the stream's `e2` block and E2's card read only those, so E2-C cannot qualify, be chosen or hold L23C. A build takes `E2_BUILD_ARMS`.
2. *What E2-C's build asks of E1's verdict.* It must be decided and name `e1_a_m640` as its arm A. It need not qualify E1-B: the amendment gates E2-C on "E1's verdict being decided and E1-A's experiment being done". The platform proposes E2-C's builds only under E2's own gate as well (E1-B qualified and done), so E2-C is built only as the control of an E2-W that exists.
3. *L23D waits for E2's verdict record* (`/stage/e2` done). Whether E2-S − E2-C is computed rests on E2's choice, and a decided attribution is never rewritten, so it cannot be decided before the choice exists. L23C also writes the E2-W files the attribution reads. After a failed L23C, L23D waits for a person's rerun of `rescore-e2`.
4. *E2's files as E2's verdict read them.* The attribution reads E2-W's (and, when chosen, E2-S's) and the reference's native files only when their names and sha256s are the ones `e2_v1.json` records. It reads `e2_v1.json` only when it was decided under E2's pre-registered parameters. It records E2's verdict by name and sha256 and never writes it.
5. *One seed text for the record.* W − C, S − C and C − b_v2_m640 all draw under `inc2/e2/attribution_se`, one draw for every run, as E2's two arms share `inc2/e2/species_se`. C − b_v2_m640 is reported with its D, pooled sd and SE; it decides nothing.
6. *S − C's conditions are computed and stored*, the same way as D_data, labelled confounded by the recipe. They credit nothing: `credited_to_data` is W − C's alone.
7. *What a recomputation must reproduce* to keep the record byte for byte: the status, `credited_to_data`, `testing_allowed`, the bootstrap's seed text and resamples, E2's qualifying arms and choice, W − C's and S − C's rule fields, and every arm's inputs' sha256s. C − b_v2_m640 and the reported species are not part of it, as E2's reported fields are not part of E2's decision.
8. *E2-C's test read needs the attribution decided* under its pre-registered parameters (the rule, the reference, `inc2/e2/attribution_se` with 1,000 resamples, no test-mode file) on the same E2 verdict (by sha256). That record pins E2-C's weights by sha256 before any test is read, as E2's verdict pins its arms'. The read is prepared only after the chosen arm's read was prepared, once. The report says it sits beside the chosen arm's headline.
9. *A separate rescore record.* `capacity/e2_attr_rescore.json`, written after the attribution, marks L23D done, as `e2_rescore.json` marks L23C done. The card reads `e2_attr_v1.json`.
10. *E2-C's builds join E2's group.* While any of the nine builds is failed, DR0 proposes none of the others. A failed E2-C build's card says that L23D needs all three and that L23C does not wait.
11. *L23D's price*: nine scoring passes at 0.25 GPU-h (2.25 GPU-h), as L23C's: E2-C's three finals, and up to three E2-W and three reference files that may be missing.

### How it is verified

- `tests/test_inc2_e2.py`, new cases (E1-A is a fixture as E1-B is):
  - the constants; `e2_cost` for C equal to W's;
  - `build --e2 C`: E2-W's definition but the init (E1-A's base__s0 by absolute path and run.json sha256); research-only from E1-A; inc2.train's `e2_problems` accepting it and refusing an E2-C record whose init is E1-B's;
  - its refusals, each leaving no directory: the name, the seeds, the role, the arm, the finals; the reference's recipe, manifest and LOCK v2; E1's verdict missing, pending, naming another arm A or built from another summary; E1-A not arm A; its run not done; its final.pt missing, a symlink or modified. A decided E1 verdict that does not qualify E1-B still builds it;
  - inc2.train in production for E2-C (stops at device). Refused at stage recipe: x1b, E1-B's or the arm's weights as init, another seed, E1-A's run.json changed, E1's verdict naming another arm A, another environment. A real 1-epoch CPU run from E1-A's weights loads whole;
  - the attribution on synthetic native files: D_data and pooled sd; the SE equal to an independent recomputation under `inc2/e2/attribution_se` and unlike E2's draw; each condition alone not crediting; E2-S − E2-C only when E2 chose S, labelled confounded; what is reported beside; three pending cases;
  - every refusal of the attribution: E2's verdict under other parameters, an E2-W file changed after E2's verdict read it, an E2-C stamp, test-mode or unchecked file, init or weights, E2-C from another E1 verdict or record or from E1-B, another reference manifest, two experiments of one seed;
  - the record: dev only (`brain_plan.dev_leaks`, the evidence scrub), never rewriting `e2_v1.json`, kept byte for byte, refused under other resamples, with test-mode files admitted or when its decision changed; a pending file overwritten;
  - `rescore-e2-attr` end to end with real CPU passes: refused before anything is scored without E2's decided verdict or with an undone E2-C final run; then E2-C's three written, E2-W's and the reference's kept, a complete rescore record; a second run keeps everything;
  - `e2-test-read --e2 C`: refused before the chosen arm's read is prepared, when no arm qualifies, on E2's verdict under other parameters, without the attribution, on one decided under other parameters or on another E2 verdict, and when E2-C's weights changed. Then three specs, once, each record naming the arm it sits beside. `e2-test-report --e2 C`: pending, then complete and never the headline, refusing a score from other weights and an attribution that admitted test-mode files;
  - the CLI.
- `tests/test_stream_ap_units.py` (`t_e2`, its E2-C part):
  - the domain items and block, prices, argv, the grammar and the evidence list;
  - the gate in five states; C0, C1, C2 in order with the full cites;
  - L23D at three of three and not at two, with its cites; its record complete; one attribution card in three variants (credited with the read commands, E2 chose S, no arm qualifying);
  - L23C before L23D when both are due; no L23D after a failed L23C;
  - a failed L23D (one card, not proposed again, no longer due in DR0); an unknown outcome followed by its job name, also while that job stays queued past `BUILD_LOST_SNAPSHOTS`;
  - an E2-C build the build refused (one card, the next E2-C build waiting until a person builds it);
  - on a forked stream version (`_t_e2c_fork`: the stream's ledger re-chained without Stage C's feasibility build and read after the campaign recorded Stage C, as L22's fork leaves it): C0, C1, C2 in order once E1's verdict qualifies, no L28, then L23D once. With `_stream_state` as it was before 16c56e2 (Stage C read from the version's own ledger only), these checks fail: nothing of E2-C is proposed and L28 is.
- `tests/test_stream_ap_replay.py`: stream_r0 now runs E2's six builds, E2-C's three, L23C, then L23D, each once, within the envelope.
- `tests/test_stream_pipeline.py`: E2 and E2-C are never built in a world that never reaches E1's verdict. `tests/test_inc2_stream.py`: `run_inc2_build.sh inc2.baseline rescore-e2-attr` under `e2_attr_v1`.
- An ad hoc mutation run on scratch copies of the package: 28 mutants of the new logic, each killed by a failing check of `test_inc2_e2.py` (the test functions each targets) or `test_stream_ap_units.py`. Two platform mutants survived the first pass; the two cases above added for them (a job queued past `BUILD_LOST_SNAPSHOTS`, DR0 after a failed L23D) kill them. They covered:
  - inc2.train: E2-C accepted under any arm A; E2-C requiring E1-B to qualify;
  - the build: E2-C from E1-B's weights; no check that E1's verdict names `e1_a_m640`, that E1-A is arm A, or of the reference's recipe;
  - the attribution: either condition alone crediting; E2's seed text; E2's files not checked against E2's verdict; E2-S − E2-C always or never computed; the E1 decisions not compared; the decision compared without the bootstrap; a differing decided record overwritten; `rescore-e2-attr` scoring before it checks E2's verdict's parameters;
  - the test read: without the chosen arm's prepared read, the verdict's sha256, a qualifying choice or the attribution's parameters;
  - the platform: E2-C built without E1-A done; L23D before E2's verdict or at two of three; L23D not record only; L23D followed by L23C's job name; `/stage/e2_attr` ignoring what the platform ran; the card without the read commands; a failed E2-C build's card without its own text.
- The full suite (142 scripts) on the lab passes but for 24 scripts, which fail the same way, check for check, on main's tree (16c56e2) in the same layout. Their causes lie in that environment, not in this change: the lab venv's Ultralytics is 8.4.129, whose args no longer carry the `half` the locked scorer reads; pytest is absent; the CI copy is not a git checkout (the deploy dry-run); there is no bench env with numpy for the job scripts and no Ollama `/api/show`; `test_brain_api.py` fails as on main. `test_inc2_e2.py` has one failing check more there than on main: E2-C's real 1-epoch run meets the same scorer error as E2-W's. On macOS with Ultralytics 8.4.22, the 13 inc2 and inc scripts among them, `test_stream_pipeline.py` and `test_stream_ap_no_throttles.py` pass with 0 failures (`test_inc2_e2.py`: 112 checks).

### Deploy

These files change `executor.code_hash()` and the stream rules version:
- `inc_autopilot/{executor.py, stream.py, stream_remote.py, diagnose_stream.py, evidence.py, stream_levers.json, stream_domains/weed.json}`;
- `brain/{policy_actions.json, approvals.py}`;
- `tests/test_stream_ap_replay.py`.

`inc2/{recipes.py, train.py, baseline.py}` are hashed into every run's drift check and S23. Sync the lab and both cluster copies from one commit that contains E2's commits and 16c56e2 (a forked stream version inherits the campaign's Stage C decision), restart the dashboard (`inc_dashboard.py`), then run `executor.run_replay_tests` so envelope grants resume. A run of E2's six that starts after the sync meets the same E2 checks as before (W and S are unchanged).

A commit without 16c56e2 puts `diagnose_stream._stream_state` back to reading Stage C from the stream version's own ledger only. The live stream is the fork `weed_stream_v1_fork1364` (05:57Z on 2026-10-04), whose ledger holds no feasibility event, so R0 would read as incomplete again and DR0 would propose none of E2's builds, E2-C's builds, L23C or L23D. Before syncing, `git merge-base --is-ancestor 16c56e2 <commit>` must succeed.

Before autonomy is turned on again, these read-only checks are made:
1. *No E2 dev score exists yet* (the amendment's condition): `capacity/e2_v1.json` and `capacity/e2_rescore.json` are absent, and no `e2_*_m640_seed*/runs/final__base__s*/scores/dev@640.json` exists. No `e2_c_m640_seed*` experiment exists.
2. *E1-A.* `capacity/e1_v1.json` names `e1_a_m640` as its `reference`. `e1_a_m640/runs/base__s{0,1,2}/run.json` are done and `testing` false; each guard record names the current LOCK v2 with nothing refused; each `weights/final.pt` is a regular file that hashes to its `weights_sha256`. `e1_a_m640/exp.json` is E1's arm A on m640 with cold_budget, built from the summary the verdict records.
3. *Budget.* The campaign's and the domain's remaining SU cover E2-C's 20.9 GPU-h of builds and L23D's 2.25, on top of E2's 39.9 and the next L18 (about 116).

## Amendment (2026-10-04): Z1, the model-zoo audit (pre-registered usage)

Written before any zoo score, conversion or contamination count existed, and revised before deploy after a review (the pilot's deadline from its own start and no assumed rate, dev_gated for any row initialised from an INC row, cwd12 membership by the bytes and legacy_join's holdout condition, code version and recipe per row in the report, the script's own path for a relative submit, tests that hold the ranking and the shortlist to dev and INC rows to dev alone). Revised a second time before deploy (2026-10-05), on top of E2-C (the amendment of 2026-10-04, later): L23Z waits for E2-C's attribution as well as E2's verdict; a training list with entries the zoo could not read is never clean; an own val set that could not be read leaves best.pt's selection unknown; a scoring task refuses an INC row off dev; tests that perturb every non-dev number. Revised a third time before deploy (2026-10-05): a checkpoint chosen on a val set the zoo cannot place by a marker or the dataset's root (selected_on unknown) is judged by its val list as one on its own split is, and so is the last.pt of an early stop; dev images in the val list of any chosen checkpoint make it dev_selected; a failed conversion check is counted as unscorable_fidelity in the report's counts and the platform record. Decided by the owner's delegate under the 2026-09-30 grant (`human:harry567566@gmail.com`). Not edited afterwards, except for the last line of "How it is verified".

**Why.**
- From 2026-03 to 2026-10 the project trained about 1,000 detectors. The cluster holds 4,785 .pt files: 805 GB in 1,704 run directories (`~/zoo/allpt_stat.txt`, 2026-10-04, sha256 bf98c064...ec67, one `size mtime ./path` line per file).
- Their published numbers cannot be compared:
  - at least four evaluators were used (pycocotools, Ultralytics val, a custom WBF matcher, and since 2026-09-27 the locked scorer);
  - the cwd12 test was the validation and checkpoint-selection set from 03-15 to 09-26;
  - some training sets held test copies (2,313 exact, 364 near) and ImageWeeds (old merged pools);
  - species names were wrong project-wide before 09-21;
  - about 15 claims were retracted.
- Z1 builds one table: every detector, scored under one protocol, with its date, method, data, recipe, code version and contamination flags.

**Scope.**
- The input is the pinned 2026-10-04 list (`zoo_v1.json` records its path and sha256; the audit copies it to `_zoo/v1/inputs/`), plus every INC run done when the audit runs (a fixed-depth glob of `INC_DIR/*/runs/*/weights/final.pt`; 824 such files on 2026-10-04 against 710 in the list).
- Composition of the list (recounted from the pinned file itself):
  - 2,078 epochN.pt;
  - 710 INC final.pt;
  - 30 INC train/weights best/last (runs in flight);
  - 1,556 MLflow copies (645 best, 778 last, 133 last_merged);
  - 378 best/last of 189 other run directories;
  - 14 classifier files;
  - 12 stock weights;
  - 6 third-party files;
  - 1 models/ copy.
- Detection models only. Skipped: classifiers, segmentation, pose/OBB, stock releases (by their 80 COCO names, never by file name alone), third-party files, unit-test fixtures, and LoRA adapters that have a merged sibling.
- Out of scope, counted: checkpoints that are not Ultralytics `.pt` files and so are not in the list. A read-only search (depth 5 under `results/`) found 13 RF-DETR run directories (`.pth`) and 1 Florence-2 fine-tune (`save_pretrained`); the locked scorer validates Ultralytics detectors only. This is a lower bound.
- Every file of the list gets exactly one decision: keep, skip (with a reason) or unscorable (with a reason). The counts must reconcile with the list or the audit refuses. Nothing is capped silently.

**Checkpoint selection (one rule, applied in this order; the first reason that applies is recorded).**
1. Skipped at listing, each counted by reason:
   - a line that is not a .pt (not_checkpoint); a listed file now missing (gone); a file whose size or mtime differs from the list (changed_since_list);
   - epoch<N>.pt (epoch_snapshot; never opened or hashed);
   - third-party `datasets/gh_*` (third_party; never opened);
   - an MLflow copy whose params/project lies under /tmp or /var (test_fixture);
   - INC runs: train/weights/{best,last}.pt left in a run directory (inc_train_weights), no done run.json (inc_not_done), a testing run (inc_testing), a kind-final symlink (inc_final_link);
   - INC MLflow copies: best.pt, chosen on a 4-image subset of training (mlflow_inc_best); a last/last_merged whose sha256 is not the run's weights_sha256 (inc_superseded_attempt);
   - last.pt beside a last_merged.pt (lora_unmerged);
   - an exact sha256 duplicate of a kept file (sha256_duplicate). The kept copy: INC final.pt, then a run directory's file, then an MLflow copy; within a class, the shortest path, then lexicographic order.
2. Skipped when the checkpoint is read: classifier, segmentation, other_task (pose, OBB), stock_weights (a release file name with the 80 COCO names), stock_coco (the 80 COCO names under another name).
3. Kept rows:
   - INC: weights/final.pt of every done, trained run, and the soups;
   - Ultralytics run directories: best.pt and last.pt, both kept and labelled (best=last when byte-identical: one row, `also_role`);
   - an MLflow copy whose source run file is gone, with its project's family.
   - Rows holding the same weights (sha256 over the class names and every state_dict tensor) are scored once; the others carry `same_weights_as`.
4. Kept but unscorable, shown as rows with their reason:
   - cannot be loaded under Ultralytics 8.4.37 (unscorable_load; s3_mamba), or an INC final.pt that no longer hashes as its run.json records;
   - modules that are neither torch nor Ultralytics and no merged sibling (unscorable_foreign);
   - a head that is not Ultralytics' Detect or v10Detect (unscorable_head);
   - a failed conversion check (unscorable_fidelity). The convert step decides it after meta's reconciliation, so report.json, report.md and the platform record take each such row out of keep and reconcile the counts again; a row holding the same weights as one that failed (scored only through it) is unscorable_fidelity as well.

**Exams (the zoo root `INC_DIR/_zoo/v1/root`, its own LOCK recording the locked scorer's sha256 18c00837...).**
- *dev* (617 images, 8 cwd12 sessions): the decision exam. A byte copy of LOCK v1's manifest; same sha256, key order and exam directory.
- *ImageWeeds* (3,208): a byte copy of v1. 12-class means Ragweed only (2,010 boxes); OtherPlant 4,922. External.
- *cwd12 test* (1,977): a byte copy of v1. **Descriptive historical read only.**
- *test v1* (3,030 images, 16 sources): built from `splits/v3/test_v1/*.jsonl` (summary.json complete, sha256 recorded), each row's written image and class-12 label, bytes checked; agnostic only; key `tv1__` + the sanitised base3 key. **Sealed** (below).
- *evalgroups_v1*: one combined sample of the five test groups (NDSU, Latvia, sesame, maize, PAGS8; base3_v2.json), agnostic only. **Sealed.**
- *ooddev_v1*: the same for the two OOD-dev groups (paddy, chilli). External.
- Evaluation-group labels:
  - each box's class name has a role in the committed table (`zoo_v1.json` exams.eval.roles, generated by the rule below from each slug's data.yaml, else its registry class names, as read on 2026-10-04); a weed box becomes class 12, a crop box is removed;
  - the rule fails closed: weed only through `species_of`, the weed keys or suffixes, the `weed:` prefix or the listed weed names; drop through the crop keys; any other name, an ambiguous one (others, plant, object, ...), a number or an empty name is unresolved;
  - an image holding a box of an unresolved or unnamed class is not read; images without a weed box are dropped, as in base3; a slug whose names are not all in the table, or with no weed class, is not read;
  - ImageWeeds is left out of NDSU (it is its own exam).
- The role table (slug: name -> role):
  - NDSU weed_crop_detection and greenhouse_crop_weed_detection: Horseweed, Kochia, Palmer Amaranth (greenhouse only), Ragweed, Redroot Pigweed, Waterhemp -> weed; Blackbean, Canola, Corn, Field Pea, Flax, Lentil, Soybean, Sugar beet -> drop. imageweeds_aerial: all five (ragweed, waterhemp, horseweed, redrootpigweed, kochia) -> weed.
  - Latvia crop_weed_detection_latvia, project-5nvic, university-of-peradeniya, vitif246x: weed -> weed, crop -> drop. francesco__weed_crop_aerial: weed -> weed, crop -> drop, 'weed-crop-aerial' unresolved (its boxes, if any, refuse their images). rf_test-8qezo: '0', '1' unresolved (not read). kg_vinayakshanawad: not in the registry (not read).
  - sesame ravirajsinh45 and leopard-ai: weed -> weed, crop -> drop. srec-dthh0: '0'..'3' unresolved (not read). zbfhf: Alternanthera pungens, Chenopodium, Crab Grass, Euphorbia Prostata, Grass, Leptadenia reticulata, Spermacoce hispida, broadleaf plantain, parthenium hysterophorous, phhyllanthus, ragweed -> weed; Arachius, Crop, Dry leaf -> drop.
  - maize: weed -> weed, maize -> drop. PAGS8: the eight 'weed: amaranthus palmeri (BBCH ...)' names -> weed.
  - paddy: weed -> weed. chilli: chilli -> drop, 'others' unresolved: chilli yields no image, so ooddev_v1 is paddy alone.
- Dedupe: images within 6 dHash bits under the 8 flips and rotations (both directions) of any cwd12 dev or test image are dropped; within and across groups, one image per 6-bit component, the first group by name (case-insensitive) keeps.
- Sampling: per group, whole capture groups (base3.capture_groups: 6-bit components, the Roboflow export stem across the group) in the order stable_int("inc2/zoo/eval_v1/<group>/<digest>"), until 500 images; a capture group larger than 500 is skipped; at most 2,500 images for evalgroups_v1 and 500 for ooddev_v1.
- ood22/ood23 are not zoo exams: they are tsw22/tsw23, trained by every v2+ model.
- Every label convention, whether an exam is agnostic only, its role and its unread slugs are printed in the report.

**What test v1 and the test groups are after Z1.** They stay the frozen multi-source test of every model trained after Z1, and the 0.90 claim is still stated on them. The zoo reads them only for historical non-INC checkpoints (stage B and the shortlist), whose training ended before test v1 existed (2026-10-03), so no choice that produced those checkpoints could use them. Those values are sealed: written only to `report_external.{json,csv}`, never to report.md, report.json's rows or the platform's record, and no recipe, data or model choice may cite them (usage rule 4). An amendment that wants to read them must say so.

**Scoring.**
- *The scorer.* The locked scorer at its own settings: 640 px, batch 32 rect, conf 0.001, iou 0.7, fp16 on a V100, Ultralytics 8.4.37. inc/scorer.py `score()` is called unchanged, with every check on every call (LOCK, every exam image and label hashed, the INC class space, no foreign module). Exam images may be read from a node-local copy of the zoo root, whose bytes the scorer verifies.
- *INC rows are read on dev only.* The zoo scores an INC row on dev only, and only when no reusable record exists. Its ImageWeeds and cwd12 test columns are its own runs' records (reused), and it is never read on test v1 or an evaluation group. So P10 ("test reads only at milestone consolidations"), E1's and E2's test-read rules ("read once, after the verdict, by a person"; "E2 does not read test v1") and the capacity and measurement arms' "never read test" are untouched: Z1 reads no test of an INC model, before or after E2's person-step read.
- *Agnostic always.* agnostic_map50_95 for every scored row and exam. For test v1 and the evaluation groups, the per-source and per-group agnostic AP comes from the same pass (scorer_agnostic.capture) and must reproduce the pass's agnostic AP within 1e-9. A 12-class value or image_correct is never kept for an agnostic-only exam.
- *12-class only where the classes map to cwd12 species.* The class map is read from the whole model.names list (cwd12_species, read-only):
  - R0: INC's 13 names; nothing converted.
  - R1: CWD12_LEGACY_LABELS or CWD12_SPECIES (cwd12 id order); id i -> INC i.
  - R2: TRAINER_SLOT_LEGACY or TRAINER_SLOT_SPECIES, optionally followed by aux_<k>. Slot j -> its species' INC id; every aux slot -> 12.
  - R3: the first 8 trainer slots plus novel names. 8 species; the novel channels -> 12.
  - R4: every name resolves (species_of -> its INC id; a weed name of the table -> 12; a crop or non-plant name -> dropped). A list holding a legacy-only label (Carpetweeds, Crabgrass, Morningglory, Nutsedge) never resolves. A checkpoint dated before 2026-09-21 (v3.60.0), or undated, resolves no name of CWD12_LEGACY_LABELS one by one: in the legacy vocabulary 'Ragweed' is Sicklepod and 'Purslane' is Palmer amaranth; such a list is R5.
  - R5: anything else that is not COCO: every channel -> 12, agnostic columns only.
  - *legacy_join.* An R1, R2 or R3 head dated before 2026-09-21 whose training list is unknown, holds entries of the cottonweed_holdout slug (listed under the merge's name for them, `cottonweed_holdout_<stem>`), or holds any image that is not a cwd12 image (by its bytes, below: a copy of a cwd12 train, valid or test image anywhere, the leave4out datasets included, is a cwd12 image): its slots mixed species (external ragweed in the Sicklepod slot, CottonWeedID15's crabgrass and nutsedge in MorningGlory's and Ragweed's, cottonweed_holdout's ids 4-11 deleted; CHANGELOG 10243-10262). Its species columns and ImageWeeds Ragweed are shown as not interpretable.
- *Head conversion* (R1-R5). The classification head is rewritten into 13 channels:
  - the last Conv2d of every cv3[i], and of one2one_cv3[i] on end2end heads, becomes Conv2d(c3, 13*K) -> Unflatten(1, (13, K)) -> MaxPool3d((K,1,1)) -> Flatten(2,3); a group of K source channels gives the maximum of their logits (the maximum of their sigmoids); empty slots weight 0 and bias -1e4; box branches untouched; nc = 13, no = 13 + 4*reg_max, yaml nc 13, names = INC's; single_cls set false (Ultralytics keeps it from a checkpoint), the original value recorded.
  - The converted file is content-addressed by the source sha256, saved in the source's dtype and written once. It is scored only when both files, loaded as the scorer loads them (`S.load_model`, fp32, fused as the validator's AutoBackend fuses them), give on 4 dev images head outputs with equal boxes, every INC channel's score the maximum of its source group's within 1e-4 and every empty channel's at most 1e-4, and when `S._check_model` accepts the converted file.
  - Before any conversion the inventory converts three synthetic heads (yolo11n R1, yolo26n end2end R2 of 100 classes, yolo11n R3) under the job's Ultralytics and refuses (exit 2) unless each passes. A converter exception with one normalised message on three models refuses as well (systemic); a tolerance miss is that row's unscorable_fidelity.
  - Converted rows are labelled with their rule. Max-merging changes per-class NMS relative to the native head; that is part of "the model read in INC space".
- *Columns (report.md).* dev: species_map50_95 (12-class, over the GT species), agnostic, and the mean of per_class over the species channels the model has; ImageWeeds: Ragweed AP and agnostic; ooddev_v1 agnostic; cwd12 test: species and agnostic, marked descriptive; trained imgsz (every read is at 640); code (exact or approx, with the commit `zoo codever` found). Table F gives every row's method, recipe (model, epochs, imgsz, batch, lr0, optimizer; INC rows also the recipe name and arm), init, data and code; report.csv carries the same (method, recipe, init, code, code_commit) and report.json each row's whole recipe and code. Species columns are empty for R5, for 1-class R4 heads, for INC runs whose training had no species box (E1), and for legacy_join rows. Test v1 and evalgroups_v1 (with per source and per group) only in report_external.
- *Reuse.* An INC run's recorded score (`runs/*/scores/<exam>.json` of any run whose weights are the row's: its own and its kind-final links) is copied into the zoo as "reused", never re-scored, when all of these hold: production true; scorer_sha256 equals the zoo LOCK's; manifest_sha256 equals the zoo LOCK's entry for the exam; weights_sha256 equals the file's sha256; settings are the protocol's; Ultralytics 8.4.37.
- Every zoo score record carries production false and the stamp `ZOO-<locked sha>`. Records live only under `INC_DIR/_zoo/v1/scores/`, never in a run directory, so no gate, milestone or evidence reader can take one for a protocol score.

**Contamination flags (per row and exam: dev, cwd12 test, ImageWeeds, test v1, evalgroups_v1, ooddev_v1, plus each evaluation group as a whole).**
- *Training list per row, with its rating:*
  - *exact*: INC run.json train_manifest, sha256 checked;
  - *listed_exact*: the data.yaml train entries of an Ultralytics run, listed as Ultralytics lists them (`path:` as check_det_dataset resolves it; a directory by a recursive walk that follows symlinked directories, skips hidden names and ends symlink loops; a .txt list by its lines), with symlinks resolved, when neither the yaml nor any listed entry (its own lstat mtime) is newer than the run's start (args.yaml mtime) and args.yaml is not newer than the checkpoint;
  - *listed_after_rebuild*: the same, failing one of those dates, or without a start date;
  - *derived_superset*: no yaml; the pinned code-derived superset (yolo_iter: leave4out dataset_8species/train ∪ dataset_holdout/train, tools/yolo_trainer.py);
  - *inherits*: a soup, or an INC run without a training manifest of its own: its parents' states only;
  - *none*: nothing recoverable (a gone yaml or list, a relative `path:`, a changed manifest).
- *Checks per training image* (each distinct image hashed once, cached; stored hashes first: the Step 1 pool, splits v3's and v2's provenance, INC manifests' sha256, the registry's dHash cache):
  - the same real path, or the same sha256 (exact; test v1 also by its originals' sha256): stored, or read for every training image whose size is an exam image's or a cwd12 image's (a byte copy has its original's size);
  - cwd12 membership, by the bytes: an image whose sha256 is one of cwd12's own images (LOCK v1's dev, train_core and test manifests, and every image under the cwd12 train, valid and test roots, hashed) is a cwd12 train or valid/test image wherever its copy lies (run_leave4out.py copied cwd12 into `results/leave4out/dataset_*` under the original names, and merges since v3.0.22 symlink to those copies, so neither the real path nor the merge prefix shows it). An image whose sha256 was not read counts by its real path (the cwd12 roots, the leave4out split folders) or its merge prefix;
  - a dHash within 6 bits under the 8 flips and rotations, in both directions (near; base3.cross_nearest); an image without a hash is never compared as dHash 0: it is counted unhashed;
  - dev only: a cwd12 train image of one of dev's 8 sessions (session);
  - test v1 only: a hit on a test_v1_companions row (companion).
  - Hashing has a budget (4 h of the inventory job, timed on a sample first); images not reached stay unhashed.
- *Entries the list could not read.* A relative line of a .txt list (Ultralytics resolves it against the training process's working directory, which is unknown) is counted unresolved; a .txt list, or a directory of the walk, that cannot be read is counted unreadable. Its images are unknown, never absent: each row records both counts (`n_unresolved`, `n_unreadable` in provenance, contamination and report.json; `+N unread` beside the image count in report.md).
- *State per exam:*
  - **Y**: one or more exact, near or companion hits, on an exact or listed_exact list (with or without unread entries: a copy the list shows is a copy);
  - **P** (possible): any listed_after_rebuild or derived_superset list, or no hit while more than 1 % of the list could not be hashed;
  - **N**: no hit on a complete, non-empty list: every entry read;
  - **U**: no list, an empty one, or no hit on a list with one or more unread entries (never N).
  - Session-only dev hits are shown beside the state, not in it.
- *Inherited.* A row inherits its init's states (train_args.model, or a `pretrained` path, or run.json init resolved to another row; soups: every member). An init that cannot be resolved to a row is U. Its state is the worse of its own and the inherited one (Y > P > U > N); stock and yaml inits inherit nothing.
- *Selection.*
  - selected_on, from the yaml's val (a `validation` key is read as val, as check_det_dataset renames it): cwd12_test (the sealed 1,977 or a cwd12_holdout copy), cwd12_test_part (leave4out valid), dev (INC dev staged as val), own_split (a val under the dataset's own root), train_subset (INC), unknown (a val outside the root that no marker names, no val, or no yaml);
  - chosen on its val set: best.pt; the last.pt of a run that stopped early (fewer results.csv rows than its epochs, patience below the epochs, at least `patience` epochs after its best fitness, no time limit); a last.pt without results.csv is unknown;
  - the val list is checked like a training list. Dev images in it make a chosen checkpoint dev_selected, whatever its selected_on (a val named as a cwd12 test copy included). For selected_on own_split or unknown: cwd12 test images make a chosen best.pt `best_partial` and a chosen last.pt `early_stop_partial`; a val that could not be listed (none, or a folder that is gone), or with unread entries and no such hit, leaves its test selection unknown and its dev selection unknown (`+sel?`: not dev-clean, and inherited as dev_gated). For a last.pt without results.csv a hit leaves the selection unknown, never none; a val list read whole with no hit selects nothing. The dataset's root is compared as written and as its real path (the val entries are real paths);
  - test_selected: best (best.pt chosen on cwd12 test), best_partial, early_stop (the last.pt of an early stop on cwd12 test), early_stop_partial, none, unknown; inherited through the init chain (shown as s);
  - dev_selected: best.pt, or the last.pt of an early stop, chosen on dev (a val staged from dev, or a val whose list holds dev images); a last.pt without results.csv on such a val is `+sel?`;
  - dev_gated: a row whose data or init a dev gate chose (a stream pool after an accepted increment; a pilot or real-loop candidate or soup; any row of any family, an INC baseline such as E2's runs included, initialised from an INC row or a soup of one); inherited through the init chain (a row initialised from a dev-gated or dev_selected row).
- Every row from before INC (09-27) whose list contains cwd12 train images carries the note "trained on cwd12 train; dev is 8 sessions of it". Pilot rows trained on the Bswap increment (40 % of its boxes relabelled on purpose, inc/pilot.py) are flagged planted_noise.

**Usage rule (binding).**
1. The zoo table is descriptive. No checkpoint is adopted, called "best", ranked as a result, or quoted as a model's accuracy from its cwd12 test, test v1 or evaluation-group number. RESEARCH_LOG, CHANGELOG, README and posters may cite the zoo only as "the historical record under one protocol", with its flags.
2. The only ranking is by dev, among dev-clean rows (dev state N, not selected on dev, not dev-gated), within 12-species, partial-species and agnostic-only rows separately. External exams are shown beside it and never re-order it. Rows that are not dev-clean are listed by family and date, unranked.
3. A checkpoint the project wants to use (an init, an incumbent, a deployable or robot model) re-qualifies under a dev rule pre-registered in its own amendment, as E1 and E2 did. A row that is not dev-clean cannot re-qualify on dev.
4. The zoo's test v1 and evaluation-group reads are sealed descriptive reads of historical checkpoints (report_external.*): no recipe, data or model choice may cite them, they are no model's pre-registered test read, and no zoo checkpoint becomes a candidate through them; test v1 and the five test groups stay the frozen test of every model trained after Z1.
5. A retracted claim is not reinstated by its zoo row. The row shows the claim's citation.

**Shortlist (stage C; computed from dev and provenance only, never from an external exam; non-INC rows, at most one per run directory, one per weights digest).** Rules, in order:
1. every checkpoint matched by zoo_v1.json "claims" (18 entries: run paths quoted in RESEARCH_LOG.md, CHANGELOG.md, README.md or docs/, resolved by `zoo claims` against the pinned list);
2. per family, the 3 highest by dev among dev-clean rows (12-species rows by dev species, the others by dev agnostic); a family with fewer than 3 clean rows is filled selection-free (rows evenly spaced by date), never by a contaminated dev;
3. per family, the latest by date.
At most 120 checkpoints: rule-3 entries are dropped first, then rule-2 fills, from the family with the most entries (ties by family name). Exams: test v1 when stage B did not score it, then evalgroups_v1, ooddev_v1, ImageWeeds, cwd12 test. The retracted claims (`zoo_v1.json` "retracted": v3.0.24-v3.0.27's cwd12 holdout numbers, the tier ladder v1's 0.27 cost) are flagged on their rows.

**Outputs.**
- People: `INC_DIR/_zoo/v1/report.{json,csv,md}`, `report_external.{json,csv}` (the sealed reads), plus every per-file record.
- The platform: `INC_DIR/capacity/zoo_v1.json`. Status, job ids, counts and sha256s only: no exam names as keys, no score paths, no metric.

**Cost (inference only; cap 40 GPU-h, all GPU-shared V100).**
- *I, inventory:* list, load, provenance, conversion, exams, contamination hashing, pilot, plan. One job, about 3-6 h (hashing at most 4 h of it).
- *A:* dev for every scorable non-INC row (about 372) and any INC row without a reusable dev record. About 0.23M image passes, about 4 GPU-h.
- *B:* test v1 for every scorable non-INC row (about 372). About 1.1M passes, at most 20 GPU-h. What does not fit is "not_scored_budget", counted.
- *select:* at most 0.5. *C:* the shortlist on what is left of the cap after I, A, B and a 1 GPU-h report reserve (about 8). *report:* at most 1.
- *Planning:* each item is priced from a pilot (one non-INC checkpoint per size class, S ≤ 10M, M ≤ 40M, L > 40M parameters, on every exam with images, kept as real records) × 1.2. The pilot runs inside the inventory job, after hours of hashing: its deadline runs from its own start, not the job's. An exam for which the pilot measured no rate stops the inventory (exit 2, record refused; the pilot is not marked done, so a resubmission runs it again): no item is priced from an assumed rate. The inventory refuses (exit 2, record refused) before any scoring task when stage A alone does not fit.
- *Enforcement.* A ledger under `_zoo/v1/ledger/` holds every attempt of every job (started at the job's own start, SLURM_JOB_START_TIME, so the exam copy and start-up count). Before each item a task stops when the chain's total plus the item's price would pass the cap (less the report reserve; stage A also keeps the select reserve), when its own time passes twice its shard's price (at least 30 min), or at its deadline (4 h less 15 min after the job's start; the pilot's after its own); the rest are not_scored_budget or not_scored_time. A resubmission adds to the same ledger, so the cap holds across attempts. Real spend is settled from sacct.

**Platform flow.**
- *Lever L23Z* (zoo_audit, MAINT lane, phase ZOO, R3 within the envelope, record only, policy action inc_audit_zoo, price: the cap, 40).
- *When DR0 proposes it:* once, last in DR0, when MAINT and DATA have nothing due, the lock exists, the stream's arm is adopted, Stage C was read, every stage the stream domain's `zoo.requires` names is done (/stage/e2: E2's rescore and verdict; /stage/e2_attr: E2-C's attribution, which follows E2-C's three builds) and /stage/zoo is missing. The chain holds MAINT for 8-12 h, so it never delays E2's verdict, E2-C's builds or E2-C's attribution; a failed L23C or L23D holds it until a person's rerun completes that record. Since an INC row is never read on a test by the zoo, it does not wait for E2's or E2-C's person-step test reads.
- *What it submits:* `bash run_inc2_zoo.sh submit` on the login node writes `capacity/zoo_v1.json` (submitting, with each id as it is obtained, then submitted) and submits five jobs, none held: inventory; score array A, afterok:inventory; select, afterany:A; score array C, afterok:select; report, afterany:C (`--kill-on-invalid-dep=yes` on the four dependents). An sbatch failure cancels everything submitted and restores the earlier record. A submission that times out part-way is an unknown outcome: what it queued runs, and the jobs write the record.
- *Following it:* by the five job ids, array tasks folded into one sacct entry per array (RUNNING while a task is live, else COMPLETED when every task completed, else the first other state; elapsed summed), and after an unknown outcome by the five job names and the record.
- *Done:* when all five completed; /stage/zoo is done once the record says complete, partial when it says partial (an escalation card naming the failed shards and the rerun commands).
- *Failure:* a person's card with the rerun commands. L23Z is never proposed again. A failed submission before anything stayed queued is the lane's ordinary failure and is proposed again.
- *Stale:* a live record with no zoo job queued and no platform item following it (a person's submission or a killed submission that stopped part-way) is /stage/zoo stale: one card, never a second proposal.
- *A person's run:* the same submit command may be run once by a person under the grant before the lever is deployed. Its record keeps the platform from proposing a second run. A person may resubmit after a stale, partial or failed record (written records are kept: resumable); a complete record refuses, and so does another --shards-a or --shards-c than the plan's.

### Implementation

- `inc2/zoo.py` (new): the verbs (preflight, submit, status, eval-names on the login node; inventory, exam-root, score, select, report as jobs; claims, codever anywhere; the inventory's steps one at a time). Imports at load: the standard library, `inc.common` and `cwd12_species`; numpy, torch, ultralytics, PIL, yaml, `inc.scorer`, `inc.splits`, `inc2.base3`, `inc2.guard` and `inc2.scorer_agnostic` are imported where used (submit never imports torch). Everything it writes lies under `INC_DIR/_zoo/v1/` and `capacity/zoo_v1.json`; a step whose marker says done under the same config sha256 is kept; per-item records are written once (tmp + `os.link`). `score` refuses (exit 2) a pilot, shard or `--item` that holds an INC row on any exam but dev (`inc_off_dev`), before it scores anything.
- `inc2/zoo_v1.json` (new): the config, recorded by sha256 in every record: the pinned list, the families, the class-map tables, the conversion tolerances, the exams (the evaluation role table generated by the rule above from `zoo eval-names`-equivalent reads of 2026-10-04), the contamination tables, the stages' budget, the shortlist's 18 claims and 5 retracted-claim patterns, the out-of-scope count. Any change is a v2.
- `run_inc2_zoo.sh` (new): GPU-shared, one v100-32, 5 CPUs, 45 GB, 12 h (the score arrays and select/report pass shorter `--time`). submit and preflight refuse inside a job, the job modes outside one. The outer copy runs; a module that differs from the nested copy stops the job (`INC_BUILD_ALLOW_DRIFT=1` runs it, recorded). Its module list is the import and config closure of a full test-world run (every inventory step and a score), the package `__init__`'s agent-framework imports included. Provenance and locks under `_zoo/v1/`, never `_campaign/`. The script takes its own path before it changes directory, so `bash run_inc2_zoo.sh submit` by a relative path from `$REPO/weed_llm_benchmark` submits the nested copy.
- The platform: `inc_autopilot/{stream.py, diagnose_stream.py, executor.py, stream_remote.py, levers_stream.py, evidence.py, stream_levers.json, stream_domains/weed.json}`, `brain/{policy_actions.json, approvals.py}`; `deploy/deploy_funnel.sh` ships `run_inc2_zoo.sh`. `stream_remote.sacct` now asks for `JobID` first and folds an array's tasks into its id (a plain job is unchanged).

### How it is verified

- `tests/test_inc2_zoo.py` (new; CPU, no scoring pass): the list's decisions per fixture and their reconciliation (a list of another sha256 and a dropped decision refuse); the class maps R0-R5, the legacy-vocabulary rule and legacy_join; meta's skips and unscorables; the converter on R1, R2 (yolo26n end2end, one2one_cv3 included), R3, R4 and R5 heads, its self-test, written once, a planted wrong weight row failing the fidelity check; provenance (exact, listed_exact with oversampled symlinks counted once, listed_after_rebuild, none, derived_superset, init resolution, a changed manifest, a symlinked subdirectory, a loop, a relative `path:`), codever; the exams (LOCK, byte copies, test v1 keys and labels, a changed label refusing, the maize slug read with its crop box removed and its crop-only and dev-copy images dropped, unread slugs with their reasons, a deterministic sample, `exam_problems` empty, `splits/` untouched, a never-train mismatch refusing); contamination (exact, near only through the 8 variants, session, companion, P, U, inheritance, an unknown init U, the unhashed tolerance, an empty list U, the weights digest; cwd12 membership by the bytes: a leave4out R3 head on dataset_8species copies without legacy_join, with the cwd12-train note and a session hit; an R1 head on cwd12 train and valid without legacy_join; a merge entry of the cottonweed_holdout slug with legacy_join); dev_gated (an INC baseline and a stream run on P_0 initialised from an INC row, any family's row initialised from one, a soup of one, inheritance from a dev_selected init, an init through a kind-final link); the plan (reuse, stage A whole with the INC rows' dev, B cut, no INC row in stage B whether taken or cut, A over the cap refusing with no shard, a pilot without a measured rate refusing with no shard, a pilot picking no INC row when an INC row has the smallest model_id of its size class, LPT, written once); select (claims, top-k among dev-clean rows by dev and not by a planted cwd12 test score in the reverse order, the selection-free fill, no INC row in any rule or in stage C under a cap that cuts nothing, the cap, exam order, budget_c, no plan refusing); the report (partial naming the shard, complete, the usage rule verbatim, every reason's count, sections and ranking (Table A's agnostic and species12 dev-clean rows by dev, the reverse of their planted cwd12 test order; three rows of one family per dev-clean section ordered by dev alone, not by dev agnostic alone, the cwd12 test or model_id; dev-gated and dev_selected rows flagged and unranked), the CSV columns, a planted codever commit and the recipe in report.json, report.csv and report.md, the sealed values only in report_external, the platform record passing the dev-only scrub); submit (five argvs without a hold, the chain's dependencies, the record, the INCZOO line, a third sbatch failing cancelling two and keeping the earlier record, a queued name, a complete record, a Slurm job, another shard count, no torch import); the job script (bash -n, partition and GPU, no package copy, modes refusing in the wrong place, exam-root then score with the right INC_DIR, submit by a relative path from the nested directory passing the nested script, the MODULES list against the computed closure); the pinned modules unchanged against git HEAD.
- `tests/test_inc2_zoo_score.py` (new; real Ultralytics passes on the CPU in test mode): records stamped TEST-ZOO-<scorer>, production false, equal to a direct `S.score` under the zoo root within 1e-9; test v1's per-source APs recomputing the pass; a second run scoring nothing; a refusal final, an error retried once then final, one message on three models stopping the task; the deadline; the chain's GPU-hour cap with an earlier attempt's spend, and a task's allowance; the root checks; a node-local exam root equal to the Lustre root; `score --item`; the pilot run as the inventory runs it, inside a job that started 4.2 h earlier, scoring every item, and a pilot that measures nothing refusing without its done marker; nothing written under any run directory.
- `tests/test_stream_ap_units.py` (`t_zoo`), `test_stream_ap_replay.py` (stream_r0 ends with L23Z, once, within the envelope), `test_stream_ap_world.py` (the zoo's submit, record and end).
- The second revision's tests. `test_inc2_zoo.py`: `test_unread` (the Lister's unresolved lines, an unreadable .txt list and an unreadable directory; `own_state`; provenance and contamination end to end on five runs outside the list: a complete list N on every exam, two relative lines or an unreadable .txt among its entries U on every exam, a list showing every dev image beside relative lines dev Y and U elsewhere, the counts in contamination and report.json, dev-clean only on the complete list, and best.pt on an own val set that cannot be read with test_selected unknown, `+sel?` and not dev-clean); `test_test_blind` (every non-dev number of every score record, the cwd12 test, ImageWeeds, OOD dev, test v1 and the evaluation groups, set reversed, random and constant in turn: the report's sections and their order, report.csv's ranks, and the shortlist's rules, ids and stage C items, under the default cap and a cap of 5, stay identical); `test_select` (no shard holds an INC row off dev; `inc_off_dev`). `test_inc2_zoo_score.py`: `score --item` of an INC row on test or test v1, and a shard holding one of its items off dev, refused with nothing scored and no task record; its dev item scored. `test_stream_ap_units.py` (`t_zoo`, `_t_zoo_order`): with E2's six and E2-C's three done and neither record, L23C, then L23D, then L23Z, each once and L23Z last in MAINT; E2-C's last build still to come is built first, then L23D, then L23Z; a failed L23D holds L23Z until a rerun writes the attribution's record; a failed L23C holds both; L23Z cites /stage/e2 and /stage/e2_attr. `test_stream_ap_replay.py`: stream_r0 ends L23C, L23D, L23Z.
- The second revision, on the tree rebased on main (2fb98d3: E2, E2-C with L23D, 16c56e2): `test_inc2_zoo.py` (129 checks), `test_inc2_zoo_score.py` (22) and `test_inc2_e2.py` pass under Ultralytics 8.4.22 (local); `test_stream_ap_no_throttles.py`, `test_stream_pipeline.py` and `test_inc2_stream.py` pass in the git worktree (local). On the lab (copied tree, no git), `test_stream_ap_units.py` (594 checks; 562 on main), `test_stream_ap_replay.py` (258; 256 on main), `test_stream_ap_mutations.py`, the world-based suites (cap_approved, lab_filed, lab_lost, lab_timeout, review_placement, names_round, fetch_bytes, shards) and the rest of the deploy pre-flight (`test_inc_ap_replay.py`, `test_inc_ap_governance.py`, `test_funnel_ap_replay.py`, `test_funnel_ap_mutations.py`, `test_funnel_domain_free.py`) pass; there `test_stream_ap_no_throttles.py`, `test_stream_pipeline.py` and `test_inc2_stream.py` fail the same checks as on main's tree (the deploy dry-run needs git; the base3 job-script case).
- 27 mutants of the second revision, each applied alone to a copy of the final tree, fail a check of a zoo or stream test file: Table A ranked by the cwd12 test (agnostic; species) or by ImageWeeds; the shortlist keyed by the cwd12 test, or by OOD dev and ImageWeeds; INC rows put on test v1 in stage B, shortlisted, or picked by the pilot; the scoring guard returning nothing or not called; `own_state` ignoring unread entries; an unreadable .txt list or directory not counted; provenance not summing unreadable entries, or recording no unresolved ones; contamination ignoring unread entries (all; the unreadable ones); the per-row counts left out of contamination or of the report row; an own val set that cannot be read leaving test_selected none, or leaving the row dev-clean; the own-split root compared as written only; L23Z requiring E2's verdict alone (in DR0; in the stream domain), its requirements ignored, proposed while another MAINT item is due, or proposed again after it ran.
- The third revision's tests. `test_inc2_zoo.py`: `test_val_selection` (thirteen runs outside the list on one clean training list: an own clean val and a clean val outside the root select nothing and stay dev-clean; best.pt on a val outside its root holding every dev image dev_selected (`+sel`), not dev-clean, though its list is N; on one holding three cwd12 test images `best_partial`; on a val named as a cwd12 test copy holding dev images `best` and dev_selected; a val outside the root that is gone, and a yaml without a val, test selection unknown and `+sel?`; the last.pt of an early stop (6 of 10 epochs, patience 3) on a val staged from dev dev_selected, the same after all ten epochs neither, without results.csv `+sel?`; the last.pt of an early stop on vals outside its root dev_selected by their dev images and `early_stop_partial` by their test images, neither after all ten epochs; a `validation` key read as the val); `test_report` (d's conversion record set failed: unscorable_fidelity 1 and keep one fewer in report.json with the counts reconciled, in report.md (the reason's row and the decisions) and in the platform record (kept, unscorable, `unscorable_by_reason`, converted one fewer), d in Table E with that reason; e's as well: e and w, which holds e's weights, 3 in all; the world's counts back once the records are restored). The world's clean list and x's merge carry a val under their own root and y's yaml cwd12's valid (every Ultralytics data yaml has a val); `test_unread`'s runs carry a clean val of their own.
- The third revision: `test_inc2_zoo.py` (143 checks) and `test_inc2_zoo_score.py` (22) pass under Ultralytics 8.4.22 and 8.4.37 (local, torch 2.10); no platform file changes. Each of 13 mutants, applied alone to a copy of the final tree, fails a check of `test_inc2_zoo.py`: selected_on unknown left out of the val-list rule; dev images in a val list ignored, or read only for own_split and unknown vals; the last.pt of an early stop not chosen on dev; a last.pt without results.csv on dev taken as not chosen; a val that cannot be listed leaving the dev selection known; contamination dropping provenance's unknown dev selection; `early_stop_partial` not shown as S; the report's counts read from meta's summary; a weights duplicate of a failed conversion kept; the platform record without the unscorable reasons; `converted` counting failed conversions; a `validation` key not read.
- Both zoo test files pass under Ultralytics 8.4.22 (local, torch 2.10) and 8.4.37 (the cluster's release; on the lab, torch 2.13). Under 8.4.129 they stop at the pinned scorer (below).
- Each fix of the revision before deploy, reverted alone on the final tree, fails a zoo test file (ten mutants: the pilot's deadline from the job's start, a plan priced from an assumed rate, dev_gated without the INC-init and inherited cases, dev-clean ignoring dev_selected, dev-clean ignoring dev_gated, cwd12 membership by path only, legacy_join's old holdout condition, codever not joined, the script's path taken after `cd`, a kind-final link overwriting its target's entry). So does each of seven mutants of the usage rules: Table A ranked by the cwd12 test (agnostic; species), the shortlist's top_dev_clean keyed by the cwd12 test (agnostic; species), INC rows put on test v1 in stage B, INC rows shortlisted, the pilot picking an INC row. Run on the lab under 8.4.37.
- Full suite on the lab (Ultralytics 8.4.129, the repository's files without git, docs files over 2 MB left out): 144 files, 117 pass. Each of the 27 that fail also fails on the merge base's tree there (598b865: 142 files, 117 pass), except `test_inc2_zoo.py` and `test_inc2_zoo_score.py`, which build their world with the pinned scorer: it reads `args.half`, which 8.4.129 no longer has, as every scorer-based test does; both pass on the lab under 8.4.37. The shared causes: the pinned scorer under 8.4.129, no pytest, no git, the job-script tests' system python without numpy, no python-pptx and the poster's large figures left out, the lab's GPU and services, base3's loader projection on a loaded machine, and `test_brain_api.py`, which fails on main.

### Deploy

- These files change `executor.code_hash()` and the stream rules version: record-replay must pass before envelope grants resume.
- `run_inc2_job.sh` globs the nested `inc2/*.py` and exits when an outer twin is missing, so `inc2/zoo.py` (and `zoo_v1.json`) reach the outer copy before, or together with, the nested one. Adding `zoo.py` changes the module list every later `inc2.train` run records; this is harmless.
- `remote.sync_outer` waits while zoo jobs are queued.
- Before autonomy: the campaign's `remaining_su` and `domain_remaining_su` must each be at least the zoo's 40 GPU-h estimate plus the next L18; otherwise "left in the campaign envelope" pauses the stream.
- A person's run before the lever is deployed: `bash run_inc2_zoo.sh preflight --version v1`, then `INCAP_DECIDED_BY=human:harry567566@gmail.com INCAP_TRIGGER=Z1 bash run_inc2_zoo.sh submit --version v1 --shards-a 32 --shards-c 16 --concurrency 4 --max-gpu-hours 40` from `$REPO/weed_llm_benchmark` on the login node.

### Open items

- Converted rows read a 100-class or 9-class detector through a max-merged head. Aux slots of mega, m1, rnd, ctl and s3 held crops and pests as well as weeds, so their agnostic reads count crop detections against weed-only exams. Agnostic AP also penalises a multi-class head against a one-class head (cross-class duplicates at different anchors survive per-class NMS); the legend says so.
- Training lists of 78 yolo_iter runs, 3 hyperagent, 4 agent_optimizer and yolo_lora are gone. Their flags are P or U, never N.
- Merged directories were rebuilt per project. Runs before a project's last merge are listed_after_rebuild (P).
- Code versions outside INC are approximate (main@<last commit before the run started>). INC rows map their per-module sha256 to commits with `zoo codever`, run where git history exists; `zoo codever --git <repo> --report` writes the report again with each row's commit.
- The audit occupies MAINT for the whole chain (about 8-12 h plus queue). L20, LC and L21 wait behind it; L21 is not exempted.
- L23Z waits for E2-C's attribution record. If a person leaves an E2-C build failed (E2-C incomplete), L23D never runs and the platform never proposes the zoo; a person then submits it (the person's run above).
- A person's chain that ends outside the platform is not charged to the campaign's SU ledger; the platform charges only a chain it submitted (sacct, per job, array tasks summed).
- The evaluation-group samples are not filtered against base_v2, arm B or the stream's pools: no INC row is read on them, and the non-INC rows predate those pools. A later read of an INC lineage on them would need that filter.
- The 8-variant hashes decode each image twice (`inc2.guard.image_hashes`, shared); the hashing budget bounds the time, and images it does not reach stay unhashed (P when over 1 %).
- L23N's native-resolution reads of the measurement arms are pointed to from the legend (`capacity/native_v1.json`), not joined to their rows.
- Only a reread through `zoo eval-names` shows a role table change since 2026-10-04: a slug whose names no longer match the committed table is not read, with its reason.

## Fix (2026-10-05): lost runs, declined proposals that stall a lane, zero-estimate candidates, and cuts refused short

Group F's ticker (`inc_autopilot/stream.py`, `diagnose_stream.py`). §6.6's stop-losses and §6.7's "never idle while there is work" hold as written; these defects broke them on the live campaign `weed_stream_v1`.

### What happened

1. **A run lost with the tick's state, then declined.** On 2026-10-03 at 07:44:44Z (tick 607) the DATA lane executed the L16S candidate sync of `rf_a-programlama__ag-programlama` (proposal `22b59a187f1c00415b9c2d70a0858c3f`, lab job `sync_rf_a_programlama__ag_programlama_c2ad2f813a`, which succeeded at 07:45:21Z). The dashboard restarted before the tick wrote `state.json`; the next tick (07:56:47Z) ran again as tick 607 from the state before. The execution log and the campaign ledger kept the run; the state did not. At 15:36:46Z (tick 651) D20 proposed the same sync (a deterministic id), the executor refused it as already run, and `_on_result`'s "recovered" branch declined it, on the belief that a run's effect shows in the snapshots. An L16S's effect, `candidate_synced`, lives only in the ticker's state.
2. **A declined id skipped in silence.** From then on `_materialise` skipped the declined id on every tick with no event and no card. Once the zenodo_15808623 shards were done (last DATA item 2026-10-04 09:27:20Z), the DATA lane stood idle while D20 called for L16 on that source every tick (about 24 h by 2026-10-05 09:05Z). D31's L4 audit (`befef2c88ba29b1313e5cfcd2cc5178e`, declined the same way at 05:55:05Z on 2026-10-05) is skipped likewise on MAINT.
3. **A candidate predicted to give no target box.** The source D20 chose ranked first with a score of 0: its collector estimate is 0.0 target boxes (one class, `true`, of a dataset titled "Trypophobia"). Every one of the 123 candidates open on 2026-10-05 has a zero estimate; for all of them the zero comes from class names still pending (the collector counts a pending name as no target class).
4. **A cut refused for want of data counted as TRAIN's failure.** At 04:53:46Z on 2026-10-05 (tick 870) D31 called for L24 on zenodo_15808623 and D22 for L18 (segment `weed_stream_v1_fork1364_s002`) in the same tick. The quarantine ran at once; L18 was filed (a replay pass for other code) and submitted at 05:15:05Z on evidence taken after the quarantine, which showed Q 319 against M 1,364. The cutter refused (`[inc2.stream] ERROR: nothing cut for weed_stream_v1_fork1364_s002: short`), and at 05:35:05Z the ticker counted TRAIN's failed step (lane fails 1, a step failure; a second would have held TRAIN). The refusal line was not read either: `run_inc2_build.sh` writes every inc2.stream build's provenance as `provenance/stream_<sid>.json`, and the ticker looked it up under the segment's name. The S24 replay fixture wrote it under the segment's name, so the tests passed.

### What changed

- **Lost runs are taken back** (`StreamRun._adopt_lost_runs`, at the start of every tick, after the lab poll). The executor writes a run's record before the ticker writes its state. An executed record of a lane lever is adopted as its lane's running item when its proposal id is in no lane (or in its lane still `proposed`, which takes it in place), no item ended with it (`ended_ids`, a new list `_clear` keeps for every item leaving a lane; `failed_ids`; `declined`), no person's parked item holds it, and its run was not adopted before (`adopted_runs`). Only records at or after the state's last successful write (`updated_utc`) are read: an earlier run was seen by a tick whose state was written. A lost run whose lane is busy waits (`adopt_from_utc` keeps the horizon). A state written before `ended_ids` existed (the first tick after this deploy) takes them once from the campaign ledger (`_seed_ended`): the items that ended at or after the horizon (`item_done`, `failed`, `withdrawn`, `intake_names_pending` by proposal id; `lab_job_finished` by the lab job its run recorded). Without it, every run of the last tick before the deploy that no lane holds (an L24, L19, L21, LV, LA or LC done when it ran) would read as lost and its effect would be applied a second time (an L24 moved `quarantined_utc`, an L21 added a second rollback). A ledger that cannot be read moves the horizon to the tick instead. (`done_keys` cannot tell: `_done` keys on the proposal's own params, the execution record holds the executor's resolved ones and `meta_params`, and it stamps a tick's start, not an execution time.) The proposal is rebuilt from the record (`_proposal_of`: id, lever, action, params with the price, trigger, experiments). The adopted item is followed like any item (`_adopt_run`): a lab job by its result, a cluster job by squeue and sacct, an experiment by name; a login verb is done at once. `_on_started`'s bookkeeping, lost with the state, is done again. Ledger event: `adopted_run`.
- **"Already ran" for a lab item** (`_recover_lab`): the lane takes back the earlier run by the lab job its execution record holds, so its result is read and its effect recorded, or it fails and is proposed again under a new id. Only a run no lane followed is taken back (`_run_followed`: its proposal id is not in `ended_ids` and the campaign ledger records no end of its item and no `lab_job_finished` of its lab job). A run a lane already followed to its end is the same work asked for again under a deterministic id: the step's attempt count rises past the proposal's (`_renew`, a `not_taken`), so the next tick proposes it under a new id; it is neither taken back (its old result would be read as a new one) nor declined. An L15 taken out of D29's wait goes back to that wait, whose date has passed, so the renewed discovery runs on the next tick and no new wait starts. It is declined only when no lab job was recorded or the run was taken back once already, and then with a `platform` card. Job, build and record-only items keep their paths.
- **A declined lab proposal is not skipped when its run can be accounted for** (`_adopt_declined`): a run no lane followed is taken back, and a run a lane followed is renewed as above. This recovers an item declined before this fix without editing the state.
- **A discovery (L15) takes a new attempt when it finishes**, as L26, L16S and L16I already did: the next L15 with the same classes runs under a new id. Before, it met "already ran" and was declined in silence; taking back the earlier run instead would read its old result as a new empty discovery (`discover.runs` and `empty_runs` up one, no discovery launched) and double D29's wait. Live, the last L15 (`3636aad0168c8e588ca3f1eebdb3fca3`, attempt 2, classes CutleafGroundcherry, PricklySida, Sicklepod; lab job `discover__a943411f0d`, 2026-09-29 17:30:31Z to 17:39:52Z, `item_done`) would have met it the next time D20 asked for those three classes.
- **No silent skips.** A declined proposal writes one `not_taken` (with its id). A watchdog (`_watchdog`) counts the ticks on which a fired diagnosis calls for a lever on an idle, unheld lane and nothing takes it for a reason that can last (declined; cannot be rendered; R0 not READY; the cut order; done after the last snapshot). After `watchdog.stall_ticks` such ticks in a row (12, 2 h at the 10-minute tick, `stream_thresholds.json` with its reason) it raises one `stall` card naming the lane, the diagnoses, the levers and the last reason, and writes `lane_stalled`; a tick on which the lane holds an item, is held, or nothing calls for it ends the stall (`lane_stall_ended`). Waits by design do not count and do not reset the count: DATA's half cadence, WAIT_DATA's discovery date, and a proposal with nothing to act on (`_proposal` returns None: an L24 on a source already quarantined).
- **D20 never fetches a zero-estimate candidate** (`diagnose_stream.zero_estimate`, `actionable`). A candidate whose first stated estimate (expected_target_boxes, else target_boxes, else images: `rank`'s order) is ≤ 0 is not ranked; one stating none of them is unknown, not zero, and stays fetchable (a known item the collect config lists without counts, S1b). With only such candidates D20 takes the path it takes with none: L15 when the last discovery is older than `discover_after_days`, else D29 decides; D29 counts none of them open. The detail lists them (`zero_estimate`). One exception: a candidate whose class names are pending still gets L26 (lab only, no SU), never a fetch while its estimate stays 0; the next L15 estimates it again from the names layer L26 extended. Positive candidates rank as before.
- **A cut refused short while Q < M is not a failure** (`_cut_short`). The build-ended follow reads the refusal from the experiment's own provenance record, else from `stream_<sid>` when its last attempt is the item's job (`_build_prov`). A refusal `nothing cut for <exp>: short` or `short_after_guard`, while the queue summary that snapshot shipped holds Q < M, is recorded `not_taken` with Q and M: the id goes to `failed_ids` (the next cut runs under a new one), the build's estimate is released, and neither the lane's failure count nor the step's is touched. While Q ≥ M (or Q unknown), the cutter and D22 disagree: a failure as before, and an escalation card.
- **A cut waits for a quarantine** (`_quarantine_unseen`). L18 is not proposed while an L24 is in the STOP lane or a quarantine L24 ran after the last snapshot (its source's rows leave Q). A proposed or filed L18 waits for the same before submission, and is withdrawn when D22, read again on the tick's evidence with the TRAIN lane idle (the item holds the lane, so the tick's own D22 is silent), no longer calls for L18 (`_cut_recheck`, `_cut_still_called`, `_withdraw_cut`; ledger `withdrawn`; an approved approval is closed as `not_submitted`). A filed item whose approval a person denied or already ran is left to its approval (`_ready` follows it). Its id goes to `failed_ids`, not to `declined` as `_adopt_fork`'s withdrawals do: D22 calls for the same cut (k, exp) again once the queue refills, and a declined id would never be proposed again.
- **A filed cut is withdrawn only once its approval is closed** (`_hold_cut`). The ticker cannot close a pending approval (only a person decides one), and a person can still approve it and run it from the INC page. Withdrawn, its id in `failed_ids`, that run (a segment cut, R3, about 116 GPU-h) would be followed by no lane, since `_adopt_lost_runs` reads the id as ended, and D22 could propose another cut beside it. Live, every L18 filed after a deploy waits pending until the replay gate passes (05:03Z on 2026-10-05), so this path is reachable. While its approval is pending, or when closing it fails (a person started it between the tick's read of the approvals and the close), the item is kept filed and not submitted: one `waiting` event and one escalation card name the approval. A person's run of it is then followed (`_executed_elsewhere`); approved and not yet run, it is closed and withdrawn on the next tick that D22 is still silent; if D22 calls for the cut again, it goes on as any filed cut (under the envelope the same approval is granted and run, with no second approval).

### How it was verified

- `tests/test_stream_ap_lost_runs.py` (new, in the stream world): the 10-03 sequence replayed (the executed record and the ok lab result kept, the state rolled back): the next tick adopts the run before anything is proposed, `candidate_synced` is set, the cluster fetch (L16) follows under a new id, and the sync is launched once. The same run older than the last state write: the re-proposal refused as already run follows the recorded lab job; with no lab job recorded it is declined with a card. The live condition (the id already declined): adopted on the next tick. A declined proposal with no run: exactly one `not_taken` over 11 ticks, a `stall` card at the 12th, once, ended when the lane takes an item. D20 with only zero-estimate candidates: no L16 or L16S on them, L15 instead; D29 counts none open (WAIT_DATA after an empty recent discovery); a names-pending zero gets L26 and no fetch, before and after its names resolve; beside a positive candidate, L16 on the positive one. A second L15 with the same classes after D29's wait: proposed under a new id and launched, no run taken back, `discover.runs` 1 then 2 by its own result. On a state as main leaves it after an L15 (no new attempt for the step), the repeat refused as already run, and the repeat already declined, are not taken back (no `adopted_run`, `runs` stays 1, no card, not declined anew): one `not_taken`, and on the next tick L15 under a new id is launched. A run taken back once and not followed since is declined with a card, not taken back again. The first tick on a state stripped of `ended_ids`, with the executor's clock 0 s and 20 s ahead of the ticker's: the L24 done in the tick before is not taken back (one `item_done`, `quarantined_utc` unchanged, its id seeded into `ended_ids`); a run lost on such a state is still taken back.
- `tests/test_stream_ap_cut_short.py` (new): a `short` refusal under `stream_<sid>` with Q < M: `not_taken`, no lane or step failure, the id retired, the estimate released; again (`short_after_guard`), no hold after two. With Q ≥ M: a failure and a card. D31's L24 and D22's L18 due in one tick: no L18 that tick, a wait until the snapshot shows the quarantine, and with the queue below M no second cut ever submitted. A filed L18 whose D22 stopped (the queue below M), its approval pending: kept filed, not withdrawn and not submitted, one `waiting` and one card over three ticks; a person then approves and runs it from the INC page, and TRAIN follows that run (`executed_elsewhere`, running, the same id, no second cut) to its `short` refusal (`not_taken`, no lane failure). Approved and not run: its approval closed (`not_submitted`), the item withdrawn, its id retired, a person's Run refused afterwards, and the cut re-proposed under a new id once the queue refills. One a person approved and ran from the INC page is followed, never withdrawn; one a person starts between the tick's read of the approvals and the close is kept filed and followed on the next tick. Under the envelope with the replay gate stale: filed pending, kept while the queue is below M, and once the queue refills and the replay passes the same approval is granted and runs (one build, one L18 approval).
- Each new test case fails with its fix disabled, one at a time in the tree (19 variants: the lost-run scan; `_recover_lab`; `_adopt_declined`; the `not_taken`; the watchdog; `actionable`; the provenance fallback; `_cut_short`; the quarantine wait in `_train_ready`; `_cut_recheck`; its guard for an approval a person ran; L15's new attempt when it finishes; `_run_followed` in `_recover_lab`; `_run_followed` in `_adopt_declined`; `_seed_ended`; the `adopted_runs` guard in `_recover_lab`; the L15 return to D29's wait; the hold of a filed cut whose approval is pending; the hold when its approval cannot be closed). Without the pending hold, the pending cut is withdrawn (`close_error` 'item is pending, not approved'), the person's run of it is followed by no lane, and under the envelope a second L18 is proposed and run beside the still-pending approval.
- `tests/test_stream_ap_replay.py` S24 now writes the refusal where production does (`provenance/stream_<sid>.json`, its job id). `tests/test_stream_ap_units.py`: the L23C restart case now expects the run taken back from the execution log and followed by its job id (it met "already executed" and was followed by its job name before); its outcomes are unchanged.
- The platform suites and the deploy pre-flight set pass on the lab (`test_stream_ap_*`, `test_stream_pipeline`, `test_stream_ap_mutations` with SM1–SM18 killed, `test_inc_ap_governance`, `test_inc_ap_replay`, `test_funnel_ap_replay`, `test_funnel_ap_mutations`, `test_funnel_domain_free`, `test_collect_intake`, `test_collect_plan`, `test_round_scheduler_stream_guard`). There, four scripts fail exactly as on main's tree (2fb98d3) in the same layout, for the environment (not a git checkout for the deploy dry-run, no numpy or transformers for the job scripts, the dashboard's `/inc` route): `test_stream_ap_no_throttles` (t_deploy), `test_inc2_stream`, `test_inc2_step1_stream`, `test_inc_ap_dashboard`; `test_stream_pipeline`'s deploy checks fail there for the same reason. In the git worktree on macOS `test_stream_pipeline.py` and `test_stream_ap_no_throttles.py` pass.
- With the hold of a filed cut (`_hold_cut`) added, these were run again in the git worktree on macOS: `test_stream_ap_cut_short`, `test_stream_ap_units`, `test_stream_ap_replay`, `test_stream_ap_lab_lost`, `test_stream_ap_lost_runs`, `test_stream_ap_lab_filed`, `test_stream_ap_review_placement`, `test_stream_ap_cap_approved`, `test_stream_ap_identity`, `test_stream_pipeline`, `test_stream_ap_mutations` (SM1–SM18 killed), `test_stream_ap_no_throttles`, `test_inc_ap_governance`, `test_inc_ap_replay`, `test_funnel_ap_replay`, `test_funnel_ap_mutations`, `test_funnel_domain_free`: all pass.
- A simulation of the next ticks on copies of the live files (state, snapshot, execution log, ledger, approvals, candidates; no ssh, a fake lab runner) with this code: no run is adopted on the live state; D20 proposes L26 on the first names-pending candidate (`rf_agaymanto__cosecha-de-aguaymanto`), then the next, one per DATA item; with MAINT idle, D31's L4 writes one `not_taken` and the `Lane MAINT stalled` card follows 12 ticks later; D31's L24 on the quarantined zenodo_15808623 counts no stall. With the running L23D removed from the state copy, the next tick takes it back from the execution log (`adopted_run`). On copies of the live state (written 2026-10-05 11:35:05Z, no `ended_ids`), the execution log and the ledger: the first tick seeds `ended_ids` from the ledger (none ended since that write) and takes back no run; the last L15's run (`3636aad0168c8e588ca3f1eebdb3fca3`) reads as followed, so a repeat of it is renewed, and the declined L16S's (`22b59a187f1c00415b9c2d70a0858c3f`) as not followed, so it is taken back if D20 proposes it again.

### Deploy

`stream.py`, `diagnose_stream.py` and `stream_thresholds.json` change `executor.code_hash()` and the stream rules version (a new prospective stream record before the next L18). Sync the lab and both cluster copies from one commit, restart the dashboard, then run `executor.run_replay_tests` so envelope grants resume. No state edit is needed.

### Open items

- D31 keeps calling for an L4 audit that already ran (`_audit_proposal`'s id has no attempt): the declined id is now visible (one `not_taken`, then a `stall` card), not fixed.
- TRAIN's failure count is still 1 from the 05:35:05Z short refusal; this fix does not rewrite past records. The next ordinary failure would hold TRAIN; a person's `stream release` (or `enable`) clears it, as does any TRAIN item that ends done.
- The declined L16S `22b59a187f1c00415b9c2d70a0858c3f` is not proposed again (its source's estimate is 0), so its run stays unread; its source is never fetched while the estimate stays 0.

## Amendment (2026-10-05): E3, two-stage species detection (pre-registered)

Written after E2's dev verdict was recorded (`capacity/e2_v1.json`, decided 2026-10-05: no arm qualifies) and before any E3 classifier, score or number existed, revised before any build after a review (E3-M's boxes, the emitted rows' limits, the classifier's pin, what the spread measures, the geometry check), and revised again before deploy and before any E3 number (Revision 1, below: E3-M's NMS back to the design's class-agnostic NMS, base_v2's EXIF-tagged training images read in their labels' frame, E3's order with the model-zoo audit). Decided by the owner's delegate under the 2026-09-30 grant (`human:harry567566@gmail.com`).

No E3 parameter was chosen from an E3 number or from any test number. E2 has no test read (no arm qualified). E1's sealed test read is part of the motivation, as it was E2's. The embedder was chosen on the semi-supervised Phase A figure (below), which used cwd12 train's 3,669 images: dev's 617 are among them, since dev is whole capture sessions of cwd12 train (`inc/splits.py`). No E3 parameter was tuned on that figure.

**Why.**
- The best sealed cwd12 test score is 0.8786 ± 0.0018 (b_v2_m640: YOLO11m at 640 on base_v2's 6,811 images; dev 12-class 0.8524 ± 0.0025). The gap to 0.90 is 0.021.
- On its own boxes b_v2_m640 loses 0.017 on dev (agnostic 0.8695 against 12-class 0.8524) and 0.0115 on test (0.8901 against 0.8786) to naming.
- E1: one-class boxes from base v3's 44,485 images score agnostic dev 0.8787 ± 0.0033 and sealed test 0.8996 ± 0.0015 (E1-A, base_v2 only: 0.8570 ± 0.0014 and 0.8838 ± 0.0018).
- E2: a 12-class detector warm-started from E1-B and trained on base_v2's species did not transfer: E2-W 0.8401 ± 0.0034 (D −0.0123), E2-S 0.8479 ± 0.0067 (D −0.0045); E2-W's dev agnostic fell to 0.8589. Learning the species in the same network appears to cost the boxes what E1-B gained.
- BioCLIP-2 (`hf-hub:imageomics/bioclip-2`, the embedder of inc.audit and the semi-supervised Phase A) names cwd12 ground-truth crops at 0.9858 ± 0.0007 from 61 exemplars (`results/framework/semisup/phaseA/results.json`; the figure includes dev's crops, above).

E3 asks: does a two-stage detector (a one-class box detector, then a crop species classifier) beat the best 12-class detector on the 12-class score, and does E1-B's larger box base help?

**Design.** Nothing is trained but one classifier. Three stage-1 detectors that exist are read; one stage-2 classifier serves all of them.

### Pre-registration

**Arms (stage 1: boxes, no retraining).** Seed s of an arm uses the final EMA weights of an existing base run, `<exp>/runs/base__s<s>/weights/final.pt`, recorded by path and sha256, which must equal that run.json's `weights_sha256` and its final run's (`final__base__s<s>`), seeds 0, 1, 2:
- **E3-B**: e1_b_m640 (E1-B, one class, base v3);
- **E3-A**: e1_a_m640 (E1-A, one class, base_v2: the control for data);
- **E3-M**: b_v2_m640 with its classes collapsed to one (the control: the same classifier on the best 12-class detector's own boxes).

Boxes come from the locked scorer's own inference settings, unchanged for every arm: 640 px, batch 32 (rect), fp16 on a CUDA device, conf 0.001, IoU 0.7, Ultralytics' default max_det (300), Ultralytics 8.4.37. E3-A and E3-B (one class) take Ultralytics' own NMS as the locked scorer runs it (multi-label, class-aware); E3-M takes class-agnostic NMS (`agnostic_nms`), which is how its twelve classes are collapsed to one box set. Rows at identical coordinates are then reduced to their most confident one (inc/scorer.py `one_per_box`, the reduction the locked agnostic score uses). A stage-1 box's confidence q is its row's conf.
- E3-A's and E3-B's stage-1 box sets are therefore exactly the ones their runs' recorded protocol agnostic dev scores were computed on (`final__base__s<s>/scores/dev.json`). Their agnostic AP must reproduce that score within 0.002, or the pass refuses.
- E3-M's box set under class-agnostic NMS was never scored: its stage-1 agnostic AP is recorded beside the run's recorded one, not compared with it. Class-agnostic NMS suppresses cross-class near-duplicates of one plant (two anchors whose top classes differ), which the locked NMS keeps; so E3-M minus the reference is the species stage together with this NMS change.
- *Reported beside, never deciding:* E3-M's boxes under the locked scorer's own NMS (the box set of the recorded agnostic dev score), on dev, with the same classifier (three more dev passes): this reading is the species stage alone.

**Geometry.** A box is mapped back to the original image by the inverse of the letterbox the validator applied: per axis, the pad and the resize gain of that axis (Ultralytics' `ratio_pad`), clipped to the image, then normalised by its size. In every two-stage pass the ground-truth boxes of every image go through the same code and must equal the exam label's normalised boxes within 1e-3 (each coordinate, the label clipped to the image), or the pass refuses. A pass reads only images whose EXIF orientation tag is absent, 1 or 0 (0 is invalid; PIL's `exif_transpose` and OpenCV, Ultralytics' reader, both leave it unrotated); any other tag refuses: the crop and Ultralytics' frame must agree, and the size check below cannot see a tag 3. Every image of dev (617), test (1,977) and ImageWeeds (3,208) has no tag (read on the cluster, 2026-10-05). The image's size must equal Ultralytics' `ori_shape`, or the pass refuses.

**Stage 2 (species).**
- *Crops*: inc.audit's protocol, through the same code: the image opened with PIL, EXIF-transposed, RGB (inc/verify.py `_cut_task`); the square crop of the box with a 10 % margin of its long side, padded with grey (124, 124, 124) off-frame, resized to 224 px bicubic (semisup_labeler `_cut`); BioCLIP-2 image features, fp16 autocast on CUDA (inc/verify.py `BioclipEmbedder`), L2-normalised (`verify._norm`).
- *Training rows*: the ground-truth boxes of base_v2's rows only.
  - The manifest is LOCK v2's base_v2 by sha256, and the manifest b_v2_m640 trained on: its exp.json and every base run record that sha256.
  - Every row passes inc2.train's `check_manifest` and the v2 never-train guard (`guard_rows`, fail closed). No row may be in a test v1 list or its companions (inc2.base3 `prior_test_lists`, by image sha256 and path).
  - No dev, test, ImageWeeds or test v1 row is read by the fit.
  - Labels are parsed once, from the bytes `check_manifest` verified (inc2.train `read_label_strict`).
  - Boxes under 16 px on a side (semisup_labeler `MIN_BOX_PX`) are left out and counted, as `verify crops` leaves them out.
  - EXIF orientation: 660 of base_v2's 6,811 images carry a tag other than none or 1 (6,109 none, 42 tag 1, 54 tag 0, 156 tag 3, 401 tag 6, 49 tag 8; all from 3seasonweeddet10/data2023, sessions 20230614, 20230616 and 20230804_HTRC_*; 5,521 boxes: 4,750 OtherPlant, 533 Purslane, 136 Ragweed, 102 PalmerAmaranth, that is 36 % of base_v2's OtherPlant boxes and 18 % of its Purslane). Their label boxes lie on the plants in the EXIF-transposed frame, which is the frame Ultralytics trained b_v2_m640 in and the frame the crops are cut in (`verify._cut_task`). The fit reads tags 3, 6 and 8 through `exif_transpose` and 0 as no rotation; any other tag refuses. classifier.json records the images and training boxes per tag.
- *Classifier*: multinomial logistic regression (scikit-learn, lbfgs, max_iter 3000, no class weights) over 13 classes, the 12 cwd12 species and OtherPlant, on the L2-normalised fp32 features (fitted in float64).
  - C is fixed by 5-fold cross-validation on the training boxes only: folds grouped by the manifest's capture session (GroupKFold, `verify._assign_folds`), grid {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}, criterion the pooled held-out multinomial log-loss. A fold whose training part lacks a class gives that class probability 0 for its held-out boxes.
  - A C value with any fold fit that does not converge is left out of the choice and recorded. A tie goes to the smaller C.
  - Per-fold class counts are recorded. Two facts known before the fit: Sicklepod's 121 boxes lie in 4 sessions (83 in `20210903_iPhoneSE_YL`), and base_v2's 66 NDSU rows have no session and form one group.
  - The final fit uses all training boxes and is deterministic (random_state `stable_int("inc2/e3/classifier")`; lbfgs is convex on fixed features). A final fit that does not converge refuses.
  - Its weights (coefficients, intercepts, classes, the training class prior) are stored as an npz and recorded with their sha256. Probabilities are softmax(X W' + b), checked at fit time against scikit-learn's `predict_proba` within 1e-6. The training features are stored in fp32, so the stored file reproduces the fit.
- *The pin*: the first fit writes `capacity/e3_classifier_pin.json` once, outside E3's directory: the npz's and the record's sha256, the training boxes' sha256 and the fit's time. A fit is refused while the pin exists, and while any E3 score, verdict or dev ground-truth file exists. Recovery (E3's directory moved aside) copies the pinned files back and checks their sha256s; it never refits. Every scoring job, the verdict and the test read check the classifier against the pin. The classifier is fitted once, before any E3 dev score exists, and the same classifier serves every arm and seed.
- *Crop-protocol check* (recorded; production refuses otherwise): the fit's fresh features of base_v2's cwd12 train boxes against Step 1's stored features of the same boxes (`step1/crops.csv` core rows, matched by key with the same image sha256, box index, and geometry within 1e-4; `verify.check_fresh`, `load_embeddings`). In production at least 100 boxes must match and the median cosine must be at least 0.99.

**Prediction.** For each stage-1 box with confidence q, the top 3 classes c by p(c | crop) (equal p: the lower class id first) are emitted as three predictions of that box with score q × p(c | crop).
- The emitted rows keep the locked scorer's own limits: a row whose score is not above 0.001 (its conf) is dropped, and at most max_det (300) rows per image are kept, by score (ties: the lower box index, then the lower class id). The identity path (below) goes through the same filter, where it changes nothing. The rows each rule drops and the images capped are recorded per pass.
- OtherPlant predictions are kept; the 12-class mean leaves that class out, as the locked scorer does.
- A box whose square side rounds below 1 px takes the classifier's training class prior as p and is counted. A box under 16 px is cut and classified as any other and counted (the classifier saw none in training).

**Scoring and the equivalence check.**
- The predictions enter the locked scorer's own matching and AP code. The locked scorer (`inc.scorer.score`, every LOCK, exam, model and settings check unchanged) runs with a subclass of its own validator (through the scorer sidecar's capturing subclass) that replaces each image's predictions after NMS and before Ultralytics' metric update. Ultralytics' matching, `ap_per_class` and the scorer's per-class, species and agnostic definitions are then computed on them, unchanged.
- *Equivalence*: the same path, fed each b_v2_m640 run's own predictions unchanged (identity mode: every row emitted as itself, through the same emission, filter and conversion code), must reproduce that run's recorded protocol dev score (`final__base__s<s>/scores/dev.json`) within 0.002 for the mAP, the species mean and every class, with the same exam, manifest, key order, weights, GT counts, locked-scorer hash and settings (inc2.scorer_native `compare_with_protocol`), for each of the three seeds.
- The check is recorded (`twostage/e3_v1/equivalence.json`, with the sha256 of the code it ran). No E3 score is taken and no verdict is decided without it, or on code other than the one it ran.

**Statistic.** Each arm-seed's dev species_map50_95 (the 12-class mean AP50-95) from its E3 dev file. The reference is b_v2_m640's three native dev files at 640 (the files `capacity/e2_v1.json` read, by name and sha256), not rescored.

**Rule (dev only, record only).**
- D = mean(arm) − mean(b_v2_m640) over seeds 0, 1, 2. An arm qualifies when D > 2 × pooled sd (√((sd_arm² + sd_ref²)/2), sample sd) **and** D > SE(D).
- SE(D): inc2.baseline `native_bootstrap` unchanged: 1,000 resamples of the dev images under `stable_int("inc2/e3/species_se")`, one draw for every run; per run and resample each species' AP50-95 on the run's tie-broken per-image arrays, the 12-class mean over the species with a GT box in the resample, the mean over seeds per arm, the arm minus the reference; SE is the sample sd over resamples.
- Choice: the largest qualifying D. A tie goes to E3-M (fewest new parts), then E3-A, then E3-B.
- Three comparisons share one reference: with 3 seeds per side the 2 pooled sd condition alone passes a null arm about 3.5 % of the time and any of three null arms about 8.6 % (400,000 simulated draws), before the SE condition. The rule is kept as written.
- *What the spread measures.* The fit is convex and deterministic, so a refit "at another seed" is the same classifier: there is no classifier seed component for the seed sd to miss. Neither side's sd measures sensitivity to the training sample; both are conditional on base_v2. The image bootstrap is the dev-sampling SE of D, given the trained systems.
- E3-M seed s uses the reference's seed-s detector, so the unpaired pooled-sd condition is conservative for E3-M; its paired per-seed differences are reported beside, not deciding.
- The verdict is `capacity/e3_v1.json`. A decided verdict is never rewritten: a recomputation whose decision agrees keeps the file byte for byte; one that differs is refused. Qualifying switches nothing.

**Attribution (record only).** D_data = mean(E3-B) − mean(E3-A), its SE by the same construction under `stable_int("inc2/e3/attribution_se")`. E3's box gain is credited to base v3's data when D_data > 2 pooled sd and > SE. It is recorded in `e3_v1.json` and changes neither qualification nor the choice.

**Reported beside, not deciding.** Each stage-1 detector's dev agnostic AP on its own stage-1 box set; E3-M's locked-NMS reading; the classifier's top-1 (and top-3) accuracy on dev ground-truth crops (cut through the same geometry code as the detected boxes), overall and per species; per-species dev AP of Carpetweed, SpottedSpurge and Purslane with their bootstrap SE; E3-M's paired per-seed differences; ImageWeeds 12-class and agnostic for every arm and the reference (in the report, for people). After the verdict, optionally, a person may run the sensitivity of the choice to the training sample: the five cross-validation fold classifiers at the chosen C applied to seed 0 of each arm on dev, their spread reported (`inc2.twostage sensitivity`), never deciding.

**Order inside a scoring job.** Every dev pass, and the dev part of the arm's score record, come first. A failure of a reported pass afterwards (ImageWeeds, E3-M's locked-NMS reading) is recorded and does not fail the job, so a reported exam cannot hold the deciding verdict.

**The test read (a person's step).** Once per qualifying arm, only after `e3_v1.json` holds a verdict decided under these parameters (the rule, the reference, `inc2/e3/species_se` with 1,000 resamples, `testing_allowed` false). The headline is the chosen arm's.
- `inc2.twostage test-read --arm X` pins the verdict, the classifier and the three stage-1 weights by sha256 and writes one job argv (`run_inc2_build.sh`).
- The job (`score-test`) first reproduces b_v2_m640's three recorded test scores through the identity path within 0.002 (or refuses before any E3 test score), then scores the arm's three seeds on test once.
- `test-report --arm X` reports 12-class and agnostic test, mean ± sd, against b_v2_m640's final test files (12-class 0.8786) with the gap to 0.90, and says whether the arm is the headline.
- If no arm qualifies, no test is read.

**Platform.** DR0 proposes E3's scoring jobs (L23F, one GPU job per arm, MAINT lane, record only, in the order M, A, B; the first also fits the classifier, takes the dev ground-truth accuracy and runs the equivalence check), once E2's verdict is recorded and decided and b_v2_m640, e1_a_m640 and e1_b_m640 are done; then E3's verdict (L23G, one job, record only); then one card with the result and the exact test-read commands. E2-C's attribution (L23D) goes first: no E3 job is proposed until L23D's record is complete or L23D failed, or one of E2-C's builds failed (L23D can then not be due without a person). The MAINT lane holds one item at a time, so this keeps an E3 job from holding L23D back. The model-zoo audit (L23Z, Amendment 2026-10-04 Z1) becomes due at the same point (after L23D's record); DR0 runs E3's block before the zoo's, so E3's jobs come first and L23Z stays once and last (L23D, L23F × 3, L23G, L23Z).

**Pricing.** L23F: 0.25 GPU-h per arm-seed (dev, the reported passes and ImageWeeds), plus 0.75 GPU-h for the classifier fit, the dev ground-truth crops and the equivalence check in the first job: E3-M 1.5, E3-A 0.75, E3-B 0.75. L23G: 0.5 GPU-h. E3 commits 3.5 SU, settled from sacct. A test read is about 0.5 GPU-h per qualifying arm.

**Caveats.**
- BioCLIP-2's pre-training data (TreeOfLife) may include web images of these species; as far as is recorded it does not include CottonWeedDet12.
- The classifier is trained on ground-truth crops and applied to detected crops, whose boxes are looser and include false positives.
- A two-stage system is two models to deploy on the robot.
- Every arm and the classifier are research-only: b_v2_m640's exp.json lists 6,762 of base_v2's 6,811 rows as research-only, and E1-A and E1-B carry rows of unknown licence. A qualifying E3 arm cannot become a deployed model without a separate licence step.
- E3-M's deciding boxes depart from the locked scorer in one setting (class-agnostic NMS); their stage-1 agnostic AP is reported, not compared with a recorded score. The locked-NMS reading, reported beside, is E3-M's species stage alone.

**State when this was written.** All six E2-C runs are done and `capacity/e2_attr_rescore.json` does not exist: L23D is due, and E3 waits behind it.

### What changed

- `inc2/twostage.py` (new):
  - E3's pre-registered constants (`ARMS`, `ARM_ORDER`, `NMS`, `REPORTED_NMS`, the seed texts, `C_GRID`, the tolerances, `RULE`, `ATTR_RULE`);
  - the geometry (`to_original`, `normalise`, `clip_label`), the emission (`emit`, `onehot`, `_rows_pred`), the classifier's probabilities (`proba`);
  - `e3_validator_class`: a subclass of the scorer sidecar's capturing validator (itself the locked scorer's own) that replaces each image's predictions in `update_metrics`, before Ultralytics' metric update, in mode identity, two_stage or gt; the stage-1 box set and its class-collapsed matches by the scorer's own per-image hook's lines; the ground-truth round trip; the EXIF and size checks; crops cut by `verify._cut_task` on a worker pool and embedded one embedder batch at a time;
  - `_score_pass`: one `inc.scorer.score` call with that class installed (restored afterwards whatever happens), the captured arrays checked against the score (`scorer_sidecar.check_capture`), test read only with score-test's own token;
  - `fit_classifier` (the manifest checks, the guard, the test v1 lists, every image's EXIF header, `verify._embed_images`, `cross_validate`, the final fit, the crop-protocol check `protocol_check`, the pin), `load_classifier`, `restore_classifier`;
  - `dev_gt_accuracy`, `equivalence`, `check_equivalence`, `score_arm` (L23F) with `_reported_passes`, `e3_decision`, `canonical`, `verdict`, `rescore_verdict` (L23G), `test_read`, `score_test`, `test_report`, `sensitivity`, and the CLI.
- `run_inc2_build.sh`: `inc2.twostage score-arm --arm M|A|B` (provenance and lock `e3_score_<arm>`), `verdict` (`e3_v1`) and `score-test --arm X` (`e3test_<arm>`), never advanced; for these, `HF_HUB_OFFLINE=1`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false` and an import check of torch, Ultralytics, scikit-learn and open_clip; `tools/inc2/twostage.py` and `tools/inc/audit.py` in the drift check.
- Autopilot:
  - `stream_domains/weed.json`: the block `e3` (arms, order, seeds, records) and the costs `e3_hours_per_run`, `e3_fit_hours`, `e3_verdict_hours`;
  - `diagnose_stream.DR0`: `_e3_gate` and the proposals of L23F (one arm at a time, in order; an arm whose job failed holds the others) and L23G;
  - `stream.py`: `/stage/e3_score` and `/stage/e3`, L23F and L23G as record only (phases E3_SCORE and E3_VERDICT; followed by their job names and records after an unknown outcome or a restart), their failure cards, and the E3 verdict card (`_e3_card`);
  - `stream_levers.json`: the L23F and L23G rows and the envelope; `levers_stream.price`: the estimators `e3_score` and `e3_verdict`;
  - `executor` (envelope levers, timeouts, the argv grammar, `STREAM_REMOTE`), `stream_remote` (the build verbs `twostage score-arm|verdict`, `--arm M|A|B`, the job names `inc_build_e3_score_<arm>` and `inc_build_e3_v1`, the stream summary's five files), `evidence` (`ALLOWED` and `twostage` among `RESERVED_DIRS`);
  - `brain/policy_actions.json`: the rows `inc_score_e3` and `inc_verdict_e3`; `brain/approvals.ENVELOPE_ACTIONS`.

### Choices where the amendment was silent, and why

1. *The map back to the original image* inverts the letterbox per axis (`ratio_pad`'s height and width gains and its pads), not through Ultralytics' `scale_preds`, which divides both axes by the height gain. Ultralytics resizes the long side to 640 and rounds the short side up, so the two gains differ by up to one resized pixel: on a 3456 × 4032 dev image `scale_preds` would place a box at the far edge 7.8e-4 of the width off, and on ImageWeeds' small images more. The per-axis map is the exact inverse of how the labels were transformed, so the ground-truth round trip holds to float precision (about 1e-7 on the test world) and 1e-3 is a real check.
2. *The ground-truth round trip* reads the exam label the scorer verified (its view's copy) and matches each ground-truth box one to one to a label box of its class: Ultralytics drops duplicate label rows, so the counts are compared on distinct rows.
3. *The dev ground-truth accuracy* is its own locked-scorer pass of E3-M seed 0's detector on dev in mode gt: its predictions go through unchanged (the pass's species score is recorded beside the run's recorded one), and the ground-truth boxes of at least 16 px are cut through the same geometry, header and size checks as detected boxes. It is taken in the first scoring job, after the fit and before the equivalence check.
4. *The pin* is `capacity/e3_classifier_pin.json`. Besides the pin, a fit is refused while E3's directory holds arm files, `dev_gt.json`, `equivalence.json` or test reads, and while any `capacity/e3_*` file exists.
5. *A class absent from a fit's training boxes* gets probability 0: the npz stores the classes the fit saw. In production all 13 are present; in a cross-validation fold a class may be missing, which the log-loss then charges.
6. *What the platform's records hold.* `capacity/e3_score_<arm>.json` holds the arm's dev files (names relative to `twostage/e3_v1`, sha256s, values) and only the status and counts of its reported passes; ImageWeeds numbers stay in `arms/<arm>/reported.json`, the arm files and `e3_v1_report.*`, which the platform never reads. The record is written complete after the dev passes and written again after the reported ones.
7. *A failed L23F holds the other arms* (one card, as E2's builds wait behind a failed one); the wait ends once a person's rerun completes the arm's record. L23G needs all three records.
8. *L23F's price.* The platform cannot see the classifier (E3's directory is not on the evidence), so the fit's 0.75 GPU-h is added to the first arm in the domain's order.
9. *The test read's job* is `run_inc2_build.sh inc2.twostage score-test --arm X` (job `inc_build_e3test_<arm>`); `score-test` is not a verb the platform's grammar admits. The reference's identity readings on test are written once and reused by a second qualifying arm's job.
10. *Crops and memory.* The cut pool (`verify._Workers`, five processes) is created before BioCLIP-2 is loaded and reused by every pass of a job; crops are embedded one embedder batch (128) at a time.
11. *Every base_v2 image's EXIF header* is read before the fit; a tag other than none, 0, 1, 3, 6 or 8 refuses (Revision 1); a pass reads none, 0 and 1 only.
12. *Timestamps* in E3's files carry microseconds, so "the classifier fitted before every dev file and the dev ground truth" is a strict comparison.
13. *The stage-1 record* (`<exam>.stage1.npz`) holds each stage-1 box in original pixels, its q, its top-3 classes and probabilities, its small and degenerate flags and its fp16 feature.
14. *The sensitivity* (`inc2.twostage sensitivity`) needs E3's decided verdict and writes `twostage/e3_v1/sensitivity.json` once; nothing reads it.
15. *The CLI* refuses `--batch`, `--device` and `--no-lock-check-for-tests` outside `INC_SCORER_TESTING=1` itself, before the scorer would.
16. *Labels at the image's edge.* Dev, test and ImageWeeds hold 3, 10 and 1 label boxes that reach past the image (read on the cluster, 2026-10-05). The geometry clips a mapped box to the image, and the round trip compares with the label clipped the same way, so such a box neither refuses a pass nor is cut off-frame; every exam label file holds only distinct rows (1,094, 3,257 and 6,932, the scorer's GT counts).
17. *ImageWeeds' crop count was not measured.* A pass emits at most 300 rows per image, so an arm whose detector puts many low-confidence boxes on ImageWeeds' 3,208 images could cut far more crops than dev's 5.8K; the reported passes then lengthen the job (within its 12 h limit), never the dev verdict, and sacct settles what it costs.

### How it is verified

- `tests/test_inc2_e3.py` (new; 97 checks). The synthetic world of `tests/test_inc2_train.py`, with the three sources as fixtures (real 13-class checkpoints, final runs scored by the locked scorer on dev, ImageWeeds and, for b_v2_m640, test; b_v2_m640's native dev files listed by a decided `e2_v1.json`). Every pass is a real Ultralytics CPU pass in test mode. A test device replaces each pass's NMS output with deterministic predictions derived from the ground truth and seeded by the weights' sha256 (the real NMS still runs first), so the scores are not trivial (dev agnostic about 0.72, species about 0.53); a deterministic 16-dimensional stub replaces BioCLIP-2. It pins:
  - the constants; `emit` (top 3, ties to the lower id, float32(q × p), the conf and max_det limits with their ties, OtherPlant kept, identity unchanged), `_rows_pred` (conf float32, never fp16), the per-axis geometry, `proba`;
  - the cross-validation rule (the argmin, equal losses to the smaller C, a non-converged C left out, sessions never split, per-fold class counts) and the crop-protocol check (pass; refused below 100 matches, on other features, on a geometry 0.01 off, on another image sha256; "not compared" in test mode);
  - `fit-classifier`: its refusals, each writing nothing (base_v2 not as LOCK v2 records it, also when b_v2_m640's records name the new bytes, b_v2_m640 trained on another manifest, a dev image planted in base_v2, a row in a test v1 list or a companion list, an evaluation manifest, a fit after any dev read or with the pin, a final fit that does not converge); the calls to `check_manifest`, `guard_rows`, LOCK v2's check and the test v1 lists; no exam file opened and no evaluation manifest read; the record and the pin; the stored fp32 features reproducing the fit; a second fit in a fresh directory giving the same npz; the pin refusing a refit after E3's directory is moved aside, and `restore-classifier` copying only files that hash as pinned;
  - the crops: `verify._cut_task`'s arrays bit for bit, an EXIF orientation 6 header; the dev ground-truth crops equal to `_cut_task`'s of the exam labels' boxes, through the detected boxes' geometry;
  - the validator: two passes in one process and a pass whose embedder fails, the scorer's validator restored after each, exactly one E3 layer; identity going through `emit` and `_rows_pred` once per image and reproducing the locked scorer's score exactly, on the test device's predictions and on an untrained model's real NMS output, with the stage-1 helper equal to its agnostic AP;
  - the locked NMS against agnostic NMS (another box set) and the max_det cap on an untrained model's 300 rows per image; a sub-pixel box taking the prior; the refusals of an EXIF orientation, a size other than `ori_shape`, a geometry 2 px off, and test outside score-test;
  - `equivalence` (refused on a recorded mAP 0.01 off and on a species mean 0.003 off; written once; code changed since, or a record that did not pass, refusing `score-arm` before any pass);
  - `score-arm` (refused before any pass: a source not done, weights that do not hash, final weights that differ, a stage-1 agnostic AP 0.003 off; then M, A and B end to end; files written once; the record dev only; ImageWeeds failing on every seed recorded without failing the job, then written by a rerun);
  - the verdict on the real files (decided, inputs paired by seed, the attribution under its own seed text, a planted `arms/*/s*/test.json` never opened, dev only, kept byte for byte, refused under other resamples, test-mode files refused in production), its refusals (24 cases, each on one changed record), pending; the rule on synthetic files (each condition alone, pooled sd, the largest D, ties M then A then B, the attribution never qualifying an arm, its SE equal to an independent recomputation and unlike the species draw; on the real files E3-M's SE equal to an independent recomputation); the files (pending overwritten, decided kept, a differing decision refused, the same decision under other bootstrap parameters refused); the L23G record;
  - the test read (refused before the verdict, on a pending verdict or another format, under other resamples, rule or test-mode files, for a non-qualifying arm, on another classifier or changed weights, a second time, and again once the arm's test scores exist even with its read record gone; score-test without a read, and before any E3 test score when the reference's recorded test score is not reproduced), score-test and test-report (pending, complete with the gap to 0.90, the headline and a qualifying arm that is not, refusing test-mode scores in production, other weights, another classifier, another test manifest), the sensitivity, the CLI.

  It passes with Ultralytics 8.4.22 and with the pinned 8.4.37 (its wheel unpacked locally).
- `tests/test_stream_ap_units.py` (`t_e3`, 30 checks): the domain block, the prices (1.5, 0.75, 0.75, 0.5), the argv and the executor's reading of it, the cluster's grammar (score-test and test-read refused), the evidence list; the gate (E2's verdict pending; a source not done; L23D due and then running; L23D failed; an E2-C build failed; an E2-C experiment still running); M, A, B in order with their cites, L23G at three of three and not at two; the card with two, one and no qualifying arms; a failed L23F (one card, the other arms waiting until a person's rerun completes the record); an unknown outcome followed by its job name past `BUILD_LOST_SNAPSHOTS`; L23G after a restart following its submission.
- `tests/test_stream_ap_replay.py`: stream_r0 now ends with L23F M, A, B and L23G, each once, within the envelope. `tests/test_stream_pipeline.py`: E3 is never proposed in a world that never reaches E2's verdict. `tests/test_inc2_stream.py`: `run_inc2_build.sh inc2.twostage score-arm|verdict|score-test` under `e3_score_<arm>`, `e3_v1`, `e3test_<arm>`, the import check, a refusal, the usage errors.
- An ad hoc mutation run on scratch copies of the package and tests: 61 mutants of the new logic.
  - In `inc2.twostage`, 47, each killed by a failing check of `test_inc2_e3.py`: 42 in the first pass. Five survived it, and a case was added for each, which now kills it: the LOCK v2 check (masked by the reference's manifest check), the test read's verdict format and status, identity output bypassing `emit` and `_rows_pred`, an equivalence record that did not pass (masked in the verdict by its sha256), a new read once the read record is gone (masked by the write-once).
  - On the platform, 14, run against `t_e3`: 13 killed. The 14th, removing the explicit hold on a failed arm, is equivalent: DR0 proposes the arms in order and stops at the first that is neither done nor missing, so a failed arm holds the later ones without it.
  - Not mutated, because it is implied: E3-M's stage-1 weights equal to the reference's seed-s weights (E3-M's source is the reference's own base run, whose final run must carry the same weights).
  - They covered the classifier touching dev or test rows (dev rows trained, the guard, the LOCK v2 check, test v1, a refit after a dev read or with the pin, C not the CV argmin, the pin unchecked), ranking by test (a planted `test.json` read, a read before the verdict), seed pairing (the seed field, the reference of another seed), the equivalence check (score-arm without it, its tolerance, the species check, a species-only comparison, identity bypassing `emit`, conf in fp16, one row dropped, `all_passed` and the code unchecked), the pipeline (agnostic NMS for any arm, `one_per_box` skipped, the EXIF and size checks, top-1 only, no q factor, OtherPlant dropped, the stage-1 comparison and the ground-truth round trip off, crops in the letterbox frame, one gain for both axes), the rule (either condition alone, the smaller D, ties to B, the attribution's seed text, the attribution qualifying B, a decided file rewritten, either bootstrap left out of the comparison), the test read (a non-qualifying arm, a second read, score-test without a read, other weights reported, the reference identity off) and the platform (E3 before E2's verdict is decided, while L23D is due, a failed L23D holding E3, the sources unchecked, the order, L23G at two of three, L23F not record only, the card without commands, a follow by another job name, `/stage` ignoring the platform's state, the fit not priced, an open `--arm` grammar, the job name's case).
- The suites, run on this change on 2026-10-05, each with 0 failures:
  - on the Mac (scorer-based; Ultralytics 8.4.22): `test_inc2_e3.py` (97 checks; and with the pinned 8.4.37), `test_inc2_e2.py` (112), `test_stream_pipeline.py` (90), `test_inc2_stream.py` (195), `test_stream_ap_no_throttles.py` (46);
  - on the lab (non-scorer, its venv): `test_stream_ap_units.py` (600), `test_stream_ap_replay.py` (258), `test_stream_ap_mutations.py` (74), `test_inc_ap_governance.py` (305), `test_inc_ap_replay.py`, `test_funnel_ap_replay.py`, `test_funnel_ap_mutations.py`, `test_funnel_domain_free.py` and every other `test_stream_ap_*.py`. There `test_inc2_stream.py` fails one check, inc2.base3's import check (the lab venv has no transformers); it passes on the Mac.

### Deploy

These files change `executor.code_hash()` and the stream rules version:
- `inc_autopilot/{executor.py, stream.py, stream_remote.py, diagnose_stream.py, evidence.py, levers_stream.py, stream_levers.json, stream_domains/weed.json}`;
- `brain/{policy_actions.json, approvals.py}`;
- `tests/test_stream_ap_replay.py`.

`inc2/twostage.py` and `run_inc2_build.sh` are new on the cluster. Sync the lab and both cluster copies from one commit that contains main's 8769024 (Z1 with L23Z and the stream ticker fixes of 2026-10-05, deployed and running) and E3's Revision 1 (2e36eee): `git merge-base --is-ancestor 8769024 <commit>` and `git merge-base --is-ancestor 2e36eee <commit>` must both succeed (Revision 1, item 3, says what either would remove). Restart the dashboard (`inc_dashboard.py`), then run `executor.run_replay_tests` so envelope grants resume.

Before autonomy, these read-only checks are made (each was true on 2026-10-05 after this change was built, read from the cluster):
1. `twostage/` and every `capacity/e3_*` file are absent.
2. `capacity/e2_v1.json` is decided under E2's parameters (`inc2/e2/species_se`, 1,000 resamples, `testing_allowed` false), and its reference inputs are b_v2_m640's three `dev@640.json` by sha256.
3. The three sources' `base__s0-2` and `final__base__s0-2` are done and production with matching weights sha256s, the base weights are regular files, and the final runs' `dev.json` are production (Ultralytics 8.4.37).
4. b_v2_m640's exp.json and its three base runs record base_v2's LOCK v2 sha256 as their training manifest.
5. The `bench` env (scikit-learn 1.7.2, open_clip 3.3.0, Ultralytics 8.4.37, torch 2.5.1) and the Hugging Face cache entry for BioCLIP-2 are present.
6. `step1/crops.csv` and its four embedding shards are current: its sha256, `crops_skipped.csv` and every input crops_info.json records hash as recorded.
7. The campaign's and the domain's remaining SU cover E3's 3.5 SU on top of the next L18 (about 116).
8. Every base_v2 image's EXIF orientation tag is none, 0, 1, 3, 6 or 8 (2026-10-05: 6,109 none, 42 tag 1, 54 tag 0, 156 tag 3, 401 tag 6, 49 tag 8), and every dev, test and ImageWeeds image has none, 0 or 1 (2026-10-05: none, all 5,802).
9. The zoo's chain: while L23Z holds MAINT (`/stage/zoo` running; live on 2026-10-05, jobs 47437012-47437018), E3's jobs wait behind it; nothing is to be done.

State on 2026-10-05, after the amendment was written: `capacity/e2_attr_rescore.json` is complete (E2-C's attribution decided, not credited to data), so L23D is settled and E3 is not held behind it.

### Revision 1 (2026-10-05, before deploy and before any E3 number)

E3 has not been deployed, so no E3 job has run and no E3 classifier, score or file exists. Three defects were found before deploy; the pre-registration above now reads as revised here.

1. **E3-M's NMS.** The design this amendment pre-registers gives E3-M class-agnostic NMS. The first text gave E3-M the locked scorer's own NMS (multi-label, class-aware, then `one_per_box`) and kept class-agnostic NMS as a reported reading, a change the design's owner had not accepted. Restored: E3-M decides on class-agnostic NMS (`twostage.NMS["M"]`); its locked-NMS boxes are the reported reading (`REPORTED_NMS`, `arms/M/s<k>/dev.locked_nms.*`, `locked_nms_reading` in the verdict); the stage-1 comparison with the recorded agnostic dev score applies to E3-A and E3-B (`STAGE1_COMPARED`), whose box sets are their recorded scores' own. What it changes for reading E3-M: class-agnostic NMS suppresses cross-class near-duplicates of a plant, which the locked NMS keeps and the classifier would name alike (same-class false positives), so E3-M minus the reference is the species stage together with that NMS change; the locked-NMS reading, reported beside, is the species stage alone, and its stage-1 agnostic AP is recorded beside the run's recorded one.
2. **EXIF orientation of base_v2's training images.** The first text refused any image whose tag is not absent or 1 and said none was found in a 2,500-row sample of base_v2. That sample held only untagged images. Every base_v2 image's tag, read on the cluster (2026-10-05, PIL `getexif()[0x0112]`, the call `twostage.header` makes): 6,109 none, 42 tag 1, 54 tag 0, 156 tag 3, 401 tag 6, 49 tag 8. The 660 with another tag all come from 3seasonweeddet10/data2023 and hold 5,521 boxes (4,750 OtherPlant, 533 Purslane, 136 Ragweed, 102 PalmerAmaranth). The first L23F job would have refused after its manifest checks (`fit_classifier`), and leaving the images out would drop 36 % of base_v2's OtherPlant boxes and 18 % of its Purslane. Drawn on one image of each of tags 3, 6 and 8, the label boxes lie on the plants only in the EXIF-transposed frame; drawn on a portrait and a landscape tag-0 image (session 20230616_HTRC_iPhone13_BD), they lie on the plants in the stored frame. These are the frames PIL's `exif_transpose` gives (it leaves the invalid 0 unrotated, as OpenCV does), the frames Ultralytics trained b_v2_m640 in and the frames `verify._cut_task` cuts in. So the fit reads tags 3, 6 and 8 through `exif_transpose` and 0 as no rotation (`FIT_ORIENTATIONS`), refuses any other tag (2, 4, 5 and 7, none in base_v2), and records images and training boxes per tag (`orientations`, `orientation_train_boxes` in classifier.json). A pass reads none, 0 and 1 (`OK_ORIENTATIONS`): its crop and Ultralytics' frame must agree, and its size check cannot see a tag 3. Dev, test and ImageWeeds carry no tag. A read-only pre-flight check (Deploy, item 8) reads every tag again.
3. **E3 and the model-zoo audit.** E3 was built on 2fb98d3. Main has since gained Z1 (L23Z, Amendment 2026-10-04) and the stream ticker fixes of 2026-10-05, both deployed; the live campaign's MAINT lane held L23Z (running, jobs 47437012-47437018) when this was revised. L23Z is due once E2's verdict and E2-C's attribution are recorded, and E3 once L23D is settled: both become due when L23D's record completes. This branch now merges main (8769024). DR0 runs E3's block before the zoo's, so on a tick where both are due E3's job is proposed and L23Z waits; in a fresh campaign the order is L23D, L23F × 3, L23G, L23Z, and L23Z stays once and last. In the live campaign L23Z already holds MAINT, and E3's jobs follow it. While an E3 job runs, DR0 may list L23Z as due, but the MAINT lane holds one item at a time; when the job ends with its record complete, the next E3 job is due on the same tick. A failed E3 arm holds E3 (one card), not the zoo. Both levers are kept wherever levers are listed: the lanes, the record-only levers and their titles, the phases, `/stage`, the follow table, the envelope and its rationale, the executor's tables, timeouts and argv grammar, `STREAM_REMOTE`, `stream_remote`'s build grammar and stream summary, the evidence allow-list, the policy rows and `approvals.ENVELOPE_ACTIONS`. A deploy of a commit without main's 8769024 would remove L23Z, its policy row, its evidence entry and `run_inc2_zoo.sh` from the live platform while the zoo's chain runs; a commit without this revision would refuse the classifier's fit and decide E3-M on the locked NMS. The deploy above requires both.

**What changed.**
- `inc2/twostage.py`: `NMS` (`M` agnostic), `REPORTED_NMS` (`M` locked), `STAGE1_COMPARED` (A, B), `reported_variant`; `score_arm` compares the stage-1 AP for A and B only and records E3-M's beside; `_reported_passes` writes the locked-NMS reading and records each reported pass's stage-1 AP beside the run's recorded one; the verdict requires the comparison for A and B and reads `locked_nms_reading`; `OK_ORIENTATIONS` (none, 0, 1) and `FIT_ORIENTATIONS` (none, 0, 1, 3, 6, 8); `fit_classifier` refuses tags outside `FIT_ORIENTATIONS` and records `orientation_train_boxes` and `orientation_rule`; `RULE` names E3-M's NMS.
- The merge of main (8769024): every conflict kept both sides (`diagnose_stream.DR0`'s E3 block before the zoo's; the lists, tables, regex and rows of both levers; the docs, CHANGELOG, README and RESEARCH_LOG entries of both); the E3 kill case of `t_e3` follows main's restart semantics (the run taken back from the execution log, `adopted_run`).
- The texts that describe E3-M's boxes (`stream_domains/weed.json` e3.why, L23F's control and success in `stream_levers.json`, `inc_score_e3`'s cost rationale in `brain/policy_actions.json`).
- Tests: `test_inc2_e3.py` (`test_fit_exif`; the constants, E3-M's and E3-A's dev files, the locked-NMS reading, a verdict refusal of an E3-M file under the locked NMS, tag 0 read in a pass); `test_stream_ap_units.py` (`t_e3_zoo`; t_e3's kill case); `test_stream_ap_replay.py` (`s_r0` ends L23C, L23D, L23F × 3, L23G, L23Z); `test_stream_ap_world.py` (`ready_r0` writes the zoo's record and E3's).

**How it is verified.**
- Each fix has a test that fails without it, run on scratch copies of the final tree: with the zoo's block before E3's in DR0, both checks of `t_e3_zoo` fail (L23Z is proposed first and takes MAINT); under the first text's orientation rule (none or 1), four checks of `test_fit_exif` fail (the fit refuses the tagged images); with E3-M back on the locked NMS and its agnostic boxes reported, three checks fail (the constants, E3-M's record and its dev file) and the run stops at the missing `dev.locked_nms.json`.
- `test_fit_exif` builds training images whose stored frame is the EXIF-transposed frame turned back (tags 6, 3, 0) with a coloured patch under the label, fits with them, and checks that each crop the fit embeds is `verify._cut_task`'s array bit for bit, that its centre is the patch, and, for tags 6 and 3, that the stored frame cut at the same box is not; a tag-2 image refuses the fit and nothing is written.
- On the final tree (macOS, `taskpolicy -b`): `test_inc2_e3.py` (105 checks) under Ultralytics 8.4.22 and the pinned 8.4.37 (its wheel unpacked locally); `test_stream_ap_units.py` (634), `test_stream_ap_replay.py` (260), `test_stream_pipeline.py` (91), `test_inc2_stream.py` (195), `test_stream_ap_no_throttles.py` (46), `test_inc_ap_governance.py`, `test_approvals.py`: 0 failures.
- On the merged platform code before the E3-M description strings changed (the merge commit 87b3512): `test_stream_ap_mutations.py` (SM1–SM18 killed), `test_inc_ap_replay.py` (R2 and R4b skipped, as allowed), `test_funnel_ap_replay.py`, `test_funnel_ap_mutations.py`, `test_funnel_domain_free.py`, `test_inc2_zoo.py`, `test_inc2_zoo_score.py` and the world-based suites (lost_runs, cut_short, identity, cap_approved, lab_filed, lab_lost, lab_timeout, review_placement, names_round, fetch_bytes, shards): 0 failures.
- Read-only on 2026-10-05: on the cluster, the EXIF tag of every base_v2, dev, test and ImageWeeds image, and label boxes drawn on tagged base_v2 images (tags 3, 6, 8, and a portrait and a landscape tag-0 image); on the lab, the live campaign's MAINT lane (L23Z running, `/stage/r0` e2 and e2_attr done).

#### Note (2026-10-05, 18:21Z): E3's jobs submitted by a person

When E3 was deployed (fb99375), the model-zoo audit (L23Z) held the MAINT lane with a chain expected to run 8-12 h, and the lane holds one item at a time, so DR0 would have proposed E3's L23F jobs only after it. The owner's delegate submitted E3's four jobs by hand under the 2026-09-30 grant (`INCAP_DECIDED_BY=human:harry567566@gmail.com`, `INCAP_TRIGGER=person:E3-MAINT-held-by-L23Z`), with the platform's own commands and job names: `run_inc2_build.sh inc2.twostage score-arm --arm M` (job 47444540, `inc_build_e3_score_m`), then `--arm A` (47444541) and `--arm B` (47444543) after M, then `inc2.twostage verdict` (47444545, `inc_build_e3_v1`) after A and B. Nothing in the pre-registered design changes: the arms, the classifier fit, the scorer path, the rule and the test read are as written above. The platform reads the records these jobs write (`capacity/e3_score_<arm>.json`, `capacity/e3_v1.json`) and does not propose L23F or L23G again for a complete record.
