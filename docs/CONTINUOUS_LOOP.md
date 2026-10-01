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
| L-2 | **Budget (replaces P5).** The campaign envelope is 1,000 SU until 2026-12-31, with a 350 SU monthly window and a 120 SU daily cap. It stays inside the allocation (10,529 GPU SU remaining on 2026-09-28) and inside the domain's 1,500 SU. | The capacity arms of L-4 and Stage C are added to R0. |
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
  - a batch's known-truth `verified_precision` has a Wilson lower bound < 0.99 on ≥ 30 matched verified boxes; or
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
| D28 | source_leak | ≥ 5 % of a source's images refused by the never-train v2 guard, or ≥ 20 % base copies. [amended 2026-09-29 (with the funnel's amendment A2, docs/FUNNEL_AUDIT.md §14): a source leaks when one of its images is within 6 dHash bits of an evaluation image under any of the 8 variants, or when its embedding hits are improbable under the copy detector's per-image false-positive rate, P(Binom(images, p_false) ≥ hits) < 0.001 (`inc2.embed_calibration.source_verdict`; p_false from the batch's copy-scan record, else the LOCK's v2 calibration; with neither, any embedding hit leaks), or at ≥ 20 % base copies. The 5 % share rule is dropped: at a per-image rate p a source of n images holds about n·p chance hits.] | L24 and a card. |
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
- **Windows.** A monthly window and a daily cap on top of the lifetime envelope (P5).
- **Cross-campaign cap.** The domain's `su_envelope` (1,500, `db.py:488`) is enforced across campaigns, by summing every campaign's `inc:<campaign>` steps. It includes the funnel's own cap of 120 GPU-h (DEC-10) and what `weed_inc_v1` has already spent.
- **Settled job estimates.** Each L16, L17, L18 and L20 job's estimate is settled from sacct (`su_ledger.reconcile` / `parse_sacct`) when the job ends. Today estimates stay charged forever.
- **Allocation balance.** The Bridges-2 balance is read from the `projects` output in the snapshot and feeds D27. Whether GPU-shared hours and any RM hours draw from one balance is read from that output, not assumed.
- **A rate for every partition.** `brain/su_rates.json` gets an explicit rate for every partition the loop uses, so no job is priced "unknown".
- **Estimate.** At a supply-limited pace of one or two segments per month, spend is about 45–100 SU per month (est., §10). The monthly cap of 250 does not bind at that rate.

**Byte budget.** `collect_gb_envelope` and `collect_gb_daily`, checked against /ocean headroom (about 675 of 7,000 GB used on 2026-09-09).

[review] **The headroom reading is stale and may already trip D27.** The INC loop's state record of 2026-09-26 gives /ocean as 93 % full (503 GB free). If that holds, D27's rule (staging + projected bytes > quota − 10 %, that is, less than 700 GB free) fires on the first snapshot, and the campaign PAUSEs before it collects anything. The quota is read live by the snapshot before R3. If free space is below 10 % + `collect_gb_envelope`, R3 does not start. A card names the largest directories under `INC_DIR` and `downloads/`, because deleting data is a person's action, and the 200 GB envelope proposal is resized to what fits. L-7 (§2.6) sets the floor to 3 % (about 210 GB). With 454 GB free on 2026-09-28, D27 does not fire, and free space covers 3 % plus the 200 GB envelope.

[review] **The allocation ends with the envelope.** The Bridges-2 allocation's recorded end date is 2026-12-31, the same date as the P5 envelope. After it no job can run, whatever the envelope says. The snapshot reads the allocation end date from `projects`, and the campaign raises an R4 card (renewal) 30 days before it. At the end date the campaign PAUSEs with reason `allocation_ended`, not COMPLETE, so the loop resumes on renewal without a rebuild.

[review] **The domain envelope.** `su_ledger.remaining(domain, …)` (`su_ledger.py:769`) sums every step of the domain's ledger, the pre-INC round history included. The cross-campaign cap above is defined on `inc:*` steps only. Before R0, the live domain's `budget` block (not db.py's default) and its ledger total are read. If the round history already exhausts the domain's `su_envelope` as `signals._check_budget` counts it, that must be resolved first (card X14), or the scheduler's own budget alarm will fire beside the stream.

**Limits** (`levers.json` `limits`, replacing the blanket 3-per-campaign rule for the stream levers):

| Lever | Limit |
|---|---|
| L16 / L17 | ≤ 3 attempts per source; ≤ 2 in flight; ≤ 6 jobs and ≤ 50 GB per day; ≤ 50 GB per source unless a person approves |
| L18 | ≤ 1 in flight; ≤ 1 per day |
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
