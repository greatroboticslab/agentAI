# Which weed each cwd12 class id is

*Written 2026-09-21 (v3.60.0). Every claim below names the file and line it was checked
against. The code that carries this table is
`weed_llm_benchmark/weed_optimizer_framework/tools/cwd12_species.py`.*

From 2026-03-15 until v3.60.0 the twelve CottonWeedDet12 (cwd12) class ids were named
with an alphabetical list that is not the dataset's class list. Only one of the twelve
names (PricklySida) was right. cwd12 has **no Crabgrass and no Nutsedge**. It has
**Waterhemp and Cutleaf groundcherry**, which the list left out. The ids and the labels
on disk were never wrong. Only the names attached to the ids were wrong. So a number
computed over ids (overall mAP, a per-id AP, a class-agnostic score) keeps its value,
but any sentence that names a species from those labels names the wrong plant. Any
join that matched another dataset's class name against those labels put boxes into the
wrong slot.

---

## 1. The table

| cwd12 id | legacy label (old `data.yaml`) | species | trainer slot | binomial |
|---|---|---|---|---|
| 0 | Carpetweeds | **Waterhemp** | 0 | *Amaranthus tuberculatus* |
| 1 | Crabgrass | **Morning glory** | 1 | *Ipomoea* spp. |
| 2 | Eclipta | **Purslane** | 8 | *Portulaca oleracea* |
| 3 | Goosegrass | **Spotted spurge** | 9 | *Euphorbia maculata* |
| 4 | Morningglory | **Carpetweed** | 10 | *Mollugo verticillata* |
| 5 | Nutsedge | **Ragweed** | 11 | *Ambrosia artemisiifolia* |
| 6 | PalmerAmaranth | **Eclipta** | 2 | *Eclipta prostrata* |
| 7 | PricklySida | **Prickly sida** | 3 | *Sida spinosa* |
| 8 | Purslane | **Palmer amaranth** | 4 | *Amaranthus palmeri* |
| 9 | Ragweed | **Sicklepod** | 5 | *Senna obtusifolia* |
| 10 | Sicklepod | **Goosegrass** | 6 | *Eleusine indica* |
| 11 | SpottedSpurge | **Cutleaf groundcherry** | 7 | *Physalis angulata* |

- *trainer slot* is the merged class space of `mega_trainer` (slots 0-11 of the nc=100
  head). It is a fixed permutation of the cwd12 ids
  (`CWD12_ORIG_TO_CANON`, `mega_trainer.py:146`, asserted equal to
  `cwd12_species.CWD12_ID_TO_SLOT` at `:147`). The slot order has not changed, so every
  existing checkpoint and every hard-coded id table in the campaign scripts (for example
  `run_s3_tier_ladder.sh:52-56`) is still correct.
- **One legacy label maps to one species in both id spaces.** The trainer permutation was
  built by matching labels (`{i: CANONICAL_12_NAMES.index(n) for i, n in
  enumerate(CWD12_ORIGINAL_NAMES)}`), so "Ragweed" is Sicklepod whether it appears as
  cwd12 id 9 or as trainer slot 5. A per-species result whose names list is known to be
  the legacy list, in cwd12 id order or in slot order, translates by the label alone:
  `legacy_to_species(label)`. Results named from any other list do not (§5 row 20).
- In code the species keys are `CWD12_SPECIES` (`Waterhemp`, `MorningGlory`, …,
  `CutleafGroundcherry`). The display names above are `CWD12_COMMON` and the binomials
  are `CWD12_BINOMIAL`.

## 2. The evidence

Established two independent ways. Both match cwd12's YOLO boxes to another source's named
boxes (`cwd12_species.py:1-23`, CHANGELOG v3.60.0):

1. **The dataset's own VGG annotations**
   (`CottonWeedDet12/annotation_VGG_json`, `region_attributes {"CottonWeed":
   {"<species>": true}}`). Every one of the 3,671 train images has a VGG file of the same
   stem. For the 3,661 whose VGG region count equals the YOLO line count, boxes were paired
   in file order: **6,090 train boxes**, and every id is 100 % one species. The other 10
   images differ in count. (Re-derived read-only on the lab copy on 2026-09-21; a VGG file
   with one region stores `regions` as a single dict, which must be read as one region.)
2. **AgML 3SeasonWeedDet10.** It shares **2,904 photographs** with the cwd12 train split
   and names its own boxes: **4,849 boxes**, 100 % agreement, none unmatched.

`semisup_labeler._verify_names` repeats the same check on every Phase A run (452 boxes
in the first run) and stops the run if any box disagrees.

## 3. Where the invented list came from

Commit **`a5d67c8`** (2026-03-15 23:39:54 −0500, "Add evaluation framework, YOLO
baseline, SLURM scripts, and paper infrastructure") added `setup_and_train.sh`. The
script copies the dataset's own `annotation_YOLO_txt` ids unchanged into a seed-42
65/20/15 split (3,671 / 1,129 / 848) and then writes `downloads/cottonweeddet12/data.yaml`
from a heredoc (`setup_and_train.sh:110-128`) that holds the alphabetical list. The same
commit put the same list into `convert_coco_to_yolo.CLASS_NAMES` and the `datasets.py`
registry. The only later edit to `setup_and_train.sh` (`7502f6c`) changed `batch` and
`workers`. The dataset's own `README.txt` (10 lines) names no class, so the ids were named by
hand.

The twelve names match, in order, the first twelve of CottonWeedID15's fifteen class names
sorted alphabetically (the other three are SpurredAnoda, Swinecress and Waterhemp).
Every later copy of the list comes from that `data.yaml`: `eval_v3_0_23.V3_NAMES`,
`mega_trainer.CWD12_ORIGINAL_NAMES` / `CANONICAL_12_NAMES`, `config.ALL_CLASSES`
(`LEGACY_ALL_CLASSES` since v3.60.0), `train_from_roboflow.V3_NAMES`, the hand-made
`cwd12_sealed.yaml` on the cluster (alphabetical, CHANGELOG v3.24.4) and the `run_v3_0_2x`
scripts.

A second defect sat next to the first. The registry entry for `cottonweed_holdout`
(`results/leave4out/dataset_holdout`) lists four names
(`Eclipta, Goosegrass, Morningglory, Nutsedge`: the legacy labels of
`config.HOLDOUT_SPECIES_IDS = {2, 3, 4, 5}`), but its
label files use all twelve original ids. Any reader that indexes `class_names[cid]` for
that slug is wrong even in legacy terms. The v3.60.0 merge comparison on the live lab
registry found **0 of 3,113** of its boxes in the right slot: ids 0-3 went to the wrong
species and ids 4-11 were deleted.

## 4. How the code reads each vocabulary now

A class-name string does not say which vocabulary it comes from. "Ragweed" is Sicklepod
as a legacy label and ragweed as a real name, and the CottonWeedID15 copy
`rf_zig-zag-lnodr__weed-detection-vanpe` uses "Crabgrass" and "Nutsedge" for real
crabgrass and nutsedge. So a legacy label is read only where the source is known to be
legacy (`cwd12_species.py:149-243`):

| source | how it is read |
|---|---|
| a trained checkpoint's `model.names`, an old `data.yaml`, an old per-class JSON | `species_names_for(whole list)`. It translates only a list that is one of the project's legacy lists, and it returns a species list unchanged, so a checkpoint trained after v3.60.0 is not translated twice |
| a cwd12 copy (`cottonweeddet12`, `cottonweed_holdout`, `cottonweed_sp8`) | `class_species(slug, cid)` by id through `CWD12_ID_SPACE`. The stored `class_names` of these slugs are never read |
| boxes our uploader wrote into the three legacy Roboflow projects (`cwd12-multiclass-v1`, `weed-crop-agent-dataset`, `cwd12-weeds`) | `uploaded_label_species`, for display and counts only. They are never read back for training |
| boxes a person drew on any other photograph, and every other dataset's class names | `species_of(real_name)`: a normalised exact alias match. "Giant ragweed" does not join Ragweed. A real name that is no cwd12 species (Crabgrass, Nutsedge, Lambsquarters, …) is not cwd12 |

Writing follows the same split. Anything new is written in species: `data.yaml` names,
stats keys, reports, API responses, and labelmaps for projects that use real names. The
exception is an upload into one of the three legacy Roboflow projects, which keeps that
project's legacy vocabulary (`species_to_legacy`) so that no project holds two names for
one plant. Renaming the classes inside Roboflow is left to a person. Where a legacy
label still has to be shown, the species comes first, for example
"Sicklepod (Roboflow label: Ragweed)".

What changed in the trainer (`mega_trainer.py`) in v3.60.0:
- External class names join slots through the species.
- The two cwd12 copies are mapped by id space and re-checked against cwd12's own labels
  on every merge.
- New checkpoints carry `CANONICAL_12_SPECIES` in `data.yaml` names. The slot ids are
  unchanged.
- Merge stats are keyed by species.
- `project_agml__imageweeds_weed_detection` is in `NEVER_TRAIN_SLUGS` (`:185`); it had
  been harvested and used for training (lab registry: `used_for_training`, harvest
  round 4).

`config.Config.ALL_CLASSES` holds the species by cwd12 id. `dataset_discovery` registers
the two leave-4-out copies with the species of their label-file ids and an `id_space`
field.

Where a directory or file name is a class name, species-era output never shares a path
with legacy output, because eight names ("Ragweed", "Goosegrass", …) are both a legacy
label and a species:

| artifact | legacy (kept, read as legacy) | species era |
|---|---|---|
| OWL proposals | `owl_red_proposals/<label>/` | `owl_red_proposals_species/<species>/`, stamped `_owl_proposals.json` with its `target_dir` |
| object bank | `object_bank/<label>/`, exemplar keys `bank/…` | `object_bank_species/<species>/` (marked `.vocabulary`), exemplar keys `banksp/…` |
| FLUX images | `synth_diffusion/fluxsynth_*` | `synth_diffusion_species/fluxsp_*` |
| cut-paste scenes | `synth_cutpaste/images/` | `synth_cutpaste/composed_species/` |
| backgrounds | `synth_cutpaste/backgrounds/` (no holdout filter) | `synth_cutpaste/backgrounds_guarded/` (marked `.holdout_guarded`) |

A command-line class argument goes through `cli_species`, which refuses the four
legacy-only labels (Carpetweeds, Crabgrass, Morningglory, Nutsedge) instead of folding
"Carpetweeds" onto Carpetweed. The holdout near-duplicate test uses
`near_dup.HOLDOUT_NEAR_DUP_BITS` (6 bits; 3 for every other near test): two Roboflow
re-exports of holdout photographs sit 4 and 5 bits from their originals.

## 5. Past findings and their status

Status values:
- **still valid**: names play no part.
- **valid after translating names**: the value belongs to the id. Only the species name
  in the text changes.
- **artifact of the name bug**: the number compares two different plants.
- **unknown**: the name bug is a plausible confound, and no measurement separates it
  from the stated cause.

Nothing listed here has been rewritten at its source. The affected sections carry a
dated note that points here. Line numbers in the documents cited below (`RESULTS_TABLE.md`,
`BEST_MODEL_CARD.md`, `SCIENCE_AUDIT.md`, `SUPERWEED_PLAN.md`, `DOUBLE_AGENT_SYSTEM.md`,
README, CHANGELOG, RESEARCH_LOG) and in the code that produced a past result
(`crossdataset_eval.py`, `synth_cutpaste.py`, `run_s3_bestmodel_eval.sh`) refer to commit
`420a449`, the last commit before v3.60.0: the dated notes added in v3.60.0 shift the
document lines after them. `mega_trainer.py` and `cwd12_species.py` line numbers refer to
the v3.60.0 tree.

| # | finding (where it is stated) | status | mechanism / translation |
|---|---|---|---|
| 1 | Overall holdout mAP of every recipe: 0.8755 ± 0.0029, 0.8759 ± 0.0030, RF-DETR 0.8974 ± 0.0040, M1, tier ladder, campaign rounds (`RESULTS_TABLE.md` Blocks A-F) | **still valid** | mAP averages over ids. The E11 row (`RESULTS_TABLE.md:119`) already records that the permutation leaves it unchanged |
| 2 | Model card per-species table: best Ragweed 0.9767, weak three Morningglory / Goosegrass / Eclipta (`BEST_MODEL_CARD.md:41-61`), and A8 (`RESULTS_TABLE.md:41`) | **valid after translating names** | `run_s3_bestmodel_eval.sh:24` keys ids by the legacy list in cwd12 id order, and the checkpoints use cwd12 ids (`cwd12_sealed.yaml`). The best class is **Sicklepod** 0.9767. The three weakest are **Carpetweed 0.7324, Spotted spurge 0.7973 and Purslane 0.8219**. The spread of 0.2443 stands. The full translation is in the card's §3 note |
| 3 | **ImageWeeds same-name ragweed collapse, 0.9604 ± 0.0018 in-domain → 0.0006 ± 0.0009 transfer** (`RESULTS_TABLE.md:179` and §E; `BEST_MODEL_CARD.md:85-86`; `RESEARCH_LOG.md` 2026-08-26 "the domain wall"; CHANGELOG v3.24.x at `:7793` and `:7848`) | **artifact of the name bug** | `crossdataset_eval.py:177` picks the prediction id whose `model.names` value is "Ragweed". On the sealed `s3_yolo11n` checkpoints that is id 9, which is **Sicklepod**. ImageWeeds GT is its id 3 `ragweed` (`:83`, `:160`). The in-domain 0.9604 scores id-9 predictions against id-9 holdout GT (`:178`), which is Sicklepod's AP. The transfer number scores a Sicklepod class against real ragweed, so even a perfect model would score near 0. The class that holds ragweed is **cwd12 id 5 (legacy "Nutsedge", trainer slot 11)**. Its in-domain AP is 0.8585 ± 0.0071 on the card. **Its transfer to ImageWeeds was never measured** |
| 4 | "Where the model does localise, it assigns the wrong species" (`BEST_MODEL_CARD.md:89-90`; `RESEARCH_LOG.md` 2026-08-26) | **unknown** | The species was judged against the Sicklepod id. The montage cited for it (`results/framework/xds_verify_montage.jpg`) draws GT and prediction boxes without class labels, so it cannot show a species assignment |
| 5 | **ImageWeeds class-agnostic transfer 0.1003 ± 0.0053 vs 0.8730 ± 0.0011 in-domain** (`RESULTS_TABLE.md` H1-H2; `BEST_MODEL_CARD.md:84`) | **still valid** | These arms ignore class (`crossdataset_eval.py:159-161`). The checkpoints scored are the three cwd12-only `s3_yolo11n` seeds (`:77`), so ImageWeeds having been in the merged training pool does not reach them |
| 6 | Ladder second exam, per-species fields `holdout_ragweed` = 0.0 on all eight checkpoints (`docs/poster/xds_ladder.json`) | **artifact of the name bug** | On a slot-space head, "Ragweed" is slot 5 (Sicklepod). It was compared with original-id-5 holdout GT (true ragweed) and with ImageWeeds ragweed. The result tables never cite these fields; H4 uses the class-agnostic arm |
| 7 | Ladder on ImageWeeds, class-agnostic: arm A −0.0299, arm B −0.0100, and the H5 reading "both exams fall" (`RESULTS_TABLE.md` H4-H5) | **unknown** | Class names play no part, but `project_agml__imageweeds_weed_detection` was `used_for_training` (harvest round 4). The leak check compares ImageWeeds only with cwd12 train and holdout (`crossdataset_eval.py:137-138`). The ladder's add-ons come from `merged_iterm1_raw_s101`, whose slug list is recorded only on the cluster. If ImageWeeds images were among the add-ons, the "second exam" was partly trained on |
| 8 | Tier ladder: +40,000 harvested images cost −0.0189, "more harvested data is not a lever" (`RESULTS_TABLE.md` C4/C7; `BEST_MODEL_CARD.md:72-76`; `SUPERWEED_PLAN.md` S1/S3) | **unknown** | The core is remapped by id (`run_s3_tier_ladder.sh:52-56`) and is correct. The add-ons carry labels from the pre-v3.60.0 merge, which joined by legacy label: 3SeasonWeedDet10 Purslane into the Palmer amaranth slot, AgML Ragweed into Sicklepod, zig-zag Crabgrass/Nutsedge into Morning glory/Ragweed, and Waterhemp, Carpetweed and Morning glory boxes deleted. How much of the cost is label noise from the join is not measured |
| 9 | Retraction mechanism: "the merge pipeline's dedup and holdout-stem filters starve the in-domain core (4,175 instances; Goosegrass down to 44)" (`RESEARCH_LOG.md` 2026-08-25; `RESULTS_TABLE.md` rows D1 and D4 and §D; `SCIENCE_AUDIT.md` rows 3 and 4b; `DOUBLE_AGENT_SYSTEM.md:121`) | **unknown** | The 0.5601 zero-harvest control and the retraction of the "0.27 cost" claim stand. The stated mechanism has a competing explanation that no control rules out: the same merge read `cottonweed_holdout` through its four-name list (0 of 3,113 boxes in the right slot on the live registry, §3). "Goosegrass 44" is a slot label, meaning **Spotted spurge**. "The four" the holdout filter landed on are that list's four labels, meaning Purslane, Spotted spurge, Carpetweed and Ragweed |
| 10 | Leave-4-out: "held out Morningglory, Goosegrass, Eclipta, Nutsedge" (`RESEARCH_LOG.md` 2026-03-19; `run_leave4out.py:45-46`; README "Cross-Species Generalization"; `SCIENCE_AUDIT.md` row 5 anti-forgetting via CHANGELOG `:685-731`) | **valid after translating names** | Held out by id, `HOLDOUT_IDS = {2, 3, 4, 5}`. The split, remap and scoring are id-based. What was held out is **Purslane, Spotted spurge, Carpetweed and Ragweed** |
| 11 | v3.0.23: four species near zero, "Eclipta, Goosegrass, Morningglory, Nutsedge" (`RESEARCH_LOG.md` Session 36, 2026-04-24; README `:199-204`; CHANGELOG `:1948-1951`) | **valid after translating names** | Those are trainer slots 8-11, meaning **Purslane, Spotted spurge, Carpetweed and Ragweed**. The first hypothesis, that the corpus "shares no classes with these 4", was a name argument. The recorded root causes are id bugs (autolabel `class_id=0`, then the sp8/holdout shared-path passthrough, CHANGELOG `:2120-2134`) |
| 12 | Session-3 YOLO11n per-class validation table (`RESEARCH_LOG.md` Phase 1; `results/figures/per_species.md`) | **valid after translating names** | Trained on the legacy `data.yaml`, ids unchanged. `make_figures.py` now names the rows by species |
| 13 | Robot field fire sweep: "24 of the 25 boxes are the same class (Purslane)" (`RESULTS_TABLE.md` I2) | **valid after translating names** | Named from the served checkpoint's `model.names`: id 8 "Purslane" is **Palmer amaranth**. The fire rate is unaffected |
| 14 | S6 status: "a Morningglory detection (0.7324) does not read as a Ragweed one (0.9767)" (`SUPERWEED_PLAN.md` §5) | **valid after translating names** | A **Carpetweed** detection (0.7324) against a **Sicklepod** one (0.9767). The reliability mechanism is unaffected |
| 15 | v3.24.0: a field photograph returns "Goosegrass conf=0.44", one of the card's weak three (CHANGELOG `:7565`) | **valid after translating names** | Model id 3 is **Spotted spurge** (0.7973 on the card) |
| 16 | "Missing CWD12 species (Eclipta/Goosegrass/Morningglory/Nutsedge) filled (zig-zag set)" (CHANGELOG `:4481`) | **artifact of the name bug** | Those slots hold Purslane, Spotted spurge, Carpetweed and Ragweed. The zig-zag set is a CottonWeedID15 copy with real names, and its filename prefixes agree with its class ids. The fill put real eclipta, goosegrass, morning glory and nutsedge boxes into slots that hold other plants |
| 17 | Synthetic-bank class counts "Goosegrass 75 … target list for the FLUX weak-class augmentation" (CHANGELOG `:4105-4117`) | **artifact of the name bug** (for any per-species reading) | The pre-v3.60.0 bank name-matched every trusted slug into the legacy label directories (`synth_cutpaste.py` at HEAD, `_canon_name_for_label`). As a result a directory mixed a cwd12 slot's plant with real-name crops of a different plant. "Goosegrass" there is Spotted spurge plus real goosegrass |
| 18 | v3.40.0 binomial map: `Goosegrass → Eleusine indica`, `PalmerAmaranth → Amaranthus palmeri`, … (CHANGELOG `:9186`) | **artifact of the name bug** (display) | Keyed by legacy label, so every binomial except *Sida spinosa* was shown beside the wrong plant. The correct binomials are in §1 |
| 19 | v3.0.25 "per-class instance counts confirm class fix: Eclipta 485, Goosegrass 75, …" (CHANGELOG `:2215`) | **valid after translating names** (counts); **unknown** (the "confirm") | Slot counts, meaning Purslane, Spotted spurge, Carpetweed and Ragweed. Whether that merge also read `cottonweed_holdout` through its four-name list is not recorded |
| 20 | v3.0.28 per-class pycocotools: "Strong: Purslane, Ragweed, Crabgrass, SpottedSpurge / Weak: Morningglory, Goosegrass, Eclipta, Nutsedge" (CHANGELOG `:2992-2995`) | **unknown** | If the names follow the legacy list, strong = Palmer amaranth, Sicklepod, Morning glory, Cutleaf groundcherry and weak = Carpetweed, Spotted spurge, Purslane, Ragweed. Before v3.24.4, however, `wbf_tta_eval` named ids from a list in a different order (CHANGELOG `:7719-7723`), and the job's names list is not in the record |

## 6. What would close the open rows

- **True-ragweed transfer (row 3).** Re-run the species arm of `crossdataset_eval`, with
  the prediction id and the holdout GT id chosen through the species (cwd12 id 5 on the
  sealed checkpoints), and report it as "Ragweed, cwd12 id 5". Since v3.60.0 the tool
  chooses the class by species and writes `iw_ragweed_species` /
  `holdout_ragweed_species` to a new artifact. It has not been run yet. It is still a
  same-name comparison until someone confirms that ImageWeeds' `ragweed` is
  *A. artemisiifolia*. The dataset card does not say.
- **Rows 7-8.** Rebuild the ladder add-ons from a merge made with the species join, the
  near-duplicate holdout guard (6 bits) and ImageWeeds in `NEVER_TRAIN_SLUGS`, then re-run both
  exams.
- **Row 9.** Run the pre- and post-v3.60.0 `_merge_datasets` on the M1 registry snapshot
  and count boxes per slot, which separates the mis-join from the filters.

## 7. Stored records that still carry legacy labels

These are records of what was run. They are read through the rules of §4 and are not
rewritten: the registry JSON and Mongo `slugs.class_names` for the two cwd12 copies;
`class_exemplars/*.jsonl`; `results/**` (including `figures_data.json`,
`s3_best_model_eval.json`, `s3_tta_ceiling/*.json`, `s6_crossdataset_imageweeds.json`,
`s6_field_fire.json`, `v3_0_23_eval.json`, `leave4out/`); the supervision-benchmark cases;
the CHANGELOG and RESEARCH_LOG; the final poster (`docs/poster/**`); and the served
checkpoint `~/models/cwd12_yolo11n_s102.pt`, whose `model.names` are the legacy labels in
cwd12 id order.
