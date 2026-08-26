# Best model card — cwd12 weed detection

*S3's closing artifact. Everything here is measured on the sealed protocol; nothing is
carried over from training-time logs. Regenerate the numbers with
`run_s3_bestmodel_eval.sh` (job 44454237) — they land in
`results/framework/s3_best_model_eval.json`.*

---

## 1. Two answers, because there are two questions

| question | model | cwd12 holdout mAP50-95 |
|---|---|---|
| **Best model we can train** | RF-DETR Large, COCO-pretrained | **0.8974 ± 0.0040** (n=4 seeds) |
| **Best model we can deploy today** | **YOLO11n, COCO-pretrained** | **0.8759 ± 0.0030** (n=3 seeds) |

RF-DETR scores higher and is the honest headline for accuracy. YOLO11n is the
**recommended deployment**: 2.6 M parameters and a 5.2 MB checkpoint against RF-DETR
Large's ~128 M, which matters because the destination is a Jetson Nano on a robot
(`reference_robot_uplink`), and the gap is 0.021 — smaller than the pretraining effect
and only about the size of the architecture effect measured in §4.

**Deployed artifact:** `results/framework/s3_yolo11n/s102/weights/best.pt`
(seed 102, the median-to-best of the three; all three are within 0.006 of each other).

## 2. Protocol — what the number means

- **Train:** cwd12 train split, 3,671 images / 6,131 annotated instances, 12 classes.
- **Validate:** the sealed holdout — cwd12 test + valid, **1,977 images, never trained
  on**, protected by a NEVER_TRAIN slug list *and* content-level dHash pre-seeding; the
  train∩holdout stem intersection was verified to be **0** by direct file comparison.
- **Runs:** 100 epochs, imgsz 640, batch auto, `deterministic=True`, patience 30,
  seeds 101 / 102 / 103. Reported as mean ± std over seeds — never a best-of-N.
- **Evaluation:** re-run from the checkpoints after training (job 44454237), so the card
  does not inherit the training loop's own bookkeeping. Per-seed mAP50-95:
  0.8746 / 0.8793 / 0.8737. mAP@0.5: 0.9330 / 0.9369 / 0.9328.
- **Environment:** `requirements.lock` (torch 2.5.1+cu121, ultralytics 8.4.37), V100-32.

## 3. Where it is strong, and where it is not

| species | mAP50-95 (mean ± std, n=3) |
|---|---|
| Ragweed | 0.9767 ± 0.0021 |
| Purslane | 0.9276 ± 0.0093 |
| PalmerAmaranth | 0.9163 ± 0.0088 |
| Crabgrass | 0.9157 ± 0.0067 |
| Carpetweeds | 0.9151 ± 0.0053 |
| PricklySida | 0.9118 ± 0.0016 |
| SpottedSpurge | 0.8818 ± 0.0130 |
| Nutsedge | 0.8585 ± 0.0071 |
| Sicklepod | 0.8555 ± 0.0098 |
| Eclipta | 0.8219 ± 0.0085 |
| Goosegrass | 0.7973 ± 0.0023 |
| **Morningglory** | **0.7324 ± 0.0052** |

The spread is **0.244 between best and worst species** — an order of magnitude larger
than the seed noise (±0.003), so per-species weakness is a real property of the model
and not run-to-run variation. Morningglory, Goosegrass and Eclipta are the three to
watch; a laser-weeding system acting on this model should expect its highest miss rate
there. The project's earlier small-object analysis is the likely mechanism and remains
the open follow-up (`CHANGELOG` L1944-1961: mAP 0.87 → 0.40 on small-box subsets).

## 4. What actually produced this number

Measured on this task, ranked (see `RESEARCH_LOG.md` 2026-08-25):

| lever | effect on mAP50-95 |
|---|---|
| COCO pretraining (YOLO11n) | **+0.0714** |
| architecture at equal initialisation (Mamba-YOLO-T over YOLO11n) | **+0.0225** |
| 100-class head vs 12-class head | −0.012 |
| adding 40,000 web-harvested images to a clean core | −0.020 |

Two consequences for anyone extending this: initialisation is the lever worth pulling
first, and **more harvested data is not a lever at all** on this task — the tier ladder
is flat from 3,671 to 43,671 training images.

## 5. Honest limits

1. **Cross-dataset generalisation: measured 2026-08-26, and it collapses.** Zero-shot on
   `project_agml/imageweeds_weed_detection` — the one harvested source that passed the
   calibrated audit (precision 1.0000), leak-checked at 0 content-hash collisions
   (Hamming ≤ 6 vs train *and* holdout, EXIF all clean) — the same checkpoints score
   **0.1003 ± 0.0053 class-agnostic** against **0.8730 ± 0.0011** on the cwd12 holdout
   under the identical matcher, and **0.0006 ± 0.0009** on ragweed against **0.9604**
   in-domain for the same-name class (species-level identity of their `ragweed` is not
   stated on the card). Inspection (`xds_verify_montage.jpg`) shows why: ImageWeeds is
   greenhouse/potted-seedling imagery with ~3× smaller relative boxes; tiny seedlings go
   undetected (the known small-object weakness) and, where the model does localise —
   often accurately — it assigns the wrong species. **This model does not travel; treat
   any new deployment domain as unmeasured until evaluated in it.** Artifact:
   `s6_crossdataset_imageweeds.json` (job 44465026).
2. **The deployment gap is half-measured (2026-08-25).** Run over **358 real robot
   frames** from eight stored sessions of both robots (indoor/bench scenes, no weeds
   present), the model produced **zero detections at the deployment threshold** —
   3.7 ms per frame. That is the property a laser-weeding system depends on when it
   drives past non-target scenery: it does not fire at nothing. What remains unmeasured
   is **recall and precision on robot frames that do contain weeds**, which needs an
   outdoor run; field accuracy stays unknown until then. Artifact:
   `results/framework/s6_domain_gap.json`.
3. **TTA/WBF ceiling: measured 2026-08-26** (jobs 44463762 + 44463922). Six arms under
   one shared matcher (its plain baseline reads 0.8554 for the checkpoint the validator
   scores 0.8793 — a −0.024 metric offset that would otherwise masquerade as a TTA
   effect): WBF fusion alone **−0.003** (nothing); multi-scale **+0.010**; scale+flip
   **+0.017**; 3-seed ensemble **+0.017**; all stacked (18 views) **+0.028**, ≈ 0.907 on
   the validator scale if the offset composes. At 2.44 s/image on a V100 — **~660× the
   deployed 3.7 ms** — this is a ceiling, not a deployment option; the §1 recommendation
   is unchanged. Artifacts: `s3_tta_ceiling/`.
4. **RF-DETR's four seeds** come from May 2026 runs under the same cwd12-only staging
   (verified leak-free in v3.22.3) but not re-run in this campaign.
5. **Licensing.** cwd12 is the training source for this model; the harvested pool — which
   this model does *not* use — carries mixed licenses (41/47 explicit, mostly CC BY 4.0).
