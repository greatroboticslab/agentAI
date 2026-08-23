# Scientific audit — what is defensible, what is not, and the hardening plan

*A reviewer-grade pass over every headline claim, 2026-08-22. Written to the standard of
a skeptical external academic reviewer: each claim is judged on evidence, threats to
validity, and reproducibility. Sources: git history, `CHANGELOG.md` (incl. the v3.1.0
seven-dimension audit), `RESEARCH_LOG.md`, on-disk artifacts.*

---

## 1. Claim-by-claim verdicts

| # | Claim | Evidence | Verdict | Required action |
|---|---|---|---|---|
| 1 | VLM benchmark: 15 models on CottonWeedDet12; best VLM (Florence-2) 0.434 mAP@0.5 vs fine-tuned YOLO11n 0.929 | verified runs, per-model table, fixed eval protocol (2026-03) | **DEFENSIBLE** | none — strongest standalone result |
| 2 | Supervised baseline cwd12: 0.865 mAP50-95 / 0.929 mAP@0.5 | standard train/val/test split, single dataset, no external data → leak channel not applicable | **DEFENSIBLE** | re-run once with pinned seeds for error bars (cheap) |
| 3 | Quality-beats-scale: 244K noisy pseudo-labeled imgs → 0.593; curated subset → 0.896 | multiple recorded runs across v3.0.26–.32; direction is large (−0.27) and consistent | **DEFENSIBLE as a finding** (honest negative + recovery) | present as the system's core lesson; state pseudo-label noise as the mechanism |
| 4 | **cwd12 holdout mAP50-95 = 0.9033** (v3.0.38-A) | measured **before** the v3.1.0 holdout content-hash guard; filename-only guard was bypassable by re-exported copies (audit C1); also best-of-N-seeds selection | **NOT DEFENSIBLE as stated** | re-train with the sealed guard (post-`bf621ce` code), fixed protocol, report mean±std over ≥3 seeds; quote whatever comes out |
| 5 | Anti-forgetting: round-3 zero forgetting on old species while +9.7% mAP@0.5 on new | recorded rounds (CHANGELOG L685-731) | **SUGGESTIVE** | small n, single config — present as observation, not law |
| 6 | Autonomous discovery works (no curated lists) | `dataset_discovery.py` + registry provenance; harvests recorded across rounds | **DEFENSIBLE mechanically** | licensing review per source before showing any harvested image (see §3.4) |
| 7 | Robot→platform live pipeline (v3.20-3.22) | contract docs + joint verifications recorded (74-batch stream, 0 loss; soak 14 min, 0 loss; both counters reconciled) | **DEFENSIBLE** (engineering claim) | none |
| 8 | Dashboard headline counts (labeled/verified totals) | v3.1.0 fixed the `simulate_cycle` unflagged-fake-events and failed-jobs-marked-done classes; historical totals predating the fix may still include simulated events | **QUOTE WITH CARE** | regenerate any presented count from `meta.simulated`-aware queries only |

## 2. The one measurement that must be redone

**Protocol for an honest headline number** (runs on the cluster, days not weeks):

1. Code: current `main` (post-`bf621ce`): holdout dHashes pre-seeded (`__HOLDOUT__`
   sentinel), `skipped_holdout_hash` stat must be reported alongside the metric.
2. Data: the v3.0.38 curated recipe, re-merged from the registry with the guard active.
3. Training: 3 seeds minimum, identical hyperparameters; report mean ± std, not max.
4. Eval: pycocotools mAP50-95 on the sealed cwd12 test split; per-species table.
5. Publish the run artifacts (config, seed list, `skipped_holdout_hash`, curves) under
   `results/framework/` and reference them from `RESEARCH_LOG.md`.

Whatever the number is — 0.86 or 0.91 — it becomes the citable one. The current 0.9033
must not be shown without the "pre-guard, best-of-N" caveat attached.

## 3. Threats to validity a strong reviewer will raise (and our answers)

1. **Holdout integrity** — raised and fixed (content-level dHash guard, v3.1.0);
   the re-measurement in §2 is the outstanding proof. Until then: leakage risk applies
   to every pre-July mega-merge number.
2. **Pseudo-label circularity** — OWLv2 labels train YOLO; both are detectors. Answer:
   the sealed holdout is human-labeled (CottonWeedDet12), so the *evaluation* is
   independent even when training labels are synthetic; the quality-beats-scale result
   is itself evidence the pipeline does not blindly self-confirm.
3. **Selection effects** — best-of-N seed reporting (v3.0.38-A) and recipe-shopping
   across rounds. Answer: §2 protocol (mean±std, pre-registered recipe).
4. **Data governance** — harvested datasets carry heterogeneous licenses; registry has
   provenance fields (v3.0.191) but no license audit has been run. Action: license
   sweep over registry `provenance.license` before showing harvested imagery; exclude
   `unspecified` from any redistributed figure.
5. **Secret hygiene** — a Kaggle API token was committed historically; HEAD is clean
   (v3.1.0) but **the token remains in public git history and rotation is not
   confirmed**. Action (owner): rotate the token, then history purge or accept the
   dead-token state after rotation. This is the only open security item from the
   seven-dimension audit's criticals.
6. **Reproducibility** — CI exists (compile + unit tests, green); deps partially
   pinned (`requirements-dev.txt` policy header); exact ML environment pin
   (`requirements.lock` from the cluster env) still missing. Cheap to close.
7. **Single-benchmark scope** — most detection numbers are cwd12 (12 cotton-field
   weeds). Cross-dataset transfer runs exist (`run_cross_dataset.py`, leave-4-out) but
   are not consolidated; present scope honestly as single-domain + platform generality
   shown by the multi-domain projects (sensor/image/robot).

## 4. Hardening milestones (each independently verifiable)

| M | Deliverable | Gate |
|---|---|---|
| **M1 — honest number** | §2 protocol run on the cluster (3 seeds) + per-species table regenerated | `skipped_holdout_hash` reported; mean±std published in RESEARCH_LOG |
| **M2 — closed loop visible** | continuous round scheduler per project (auto: harvest → curate → train → eval → repeat) + a compounding chart (round № vs holdout metric) on the project page | a project left alone for N hours shows ≥2 completed rounds and a live curve, from real jobs |
| **M3 — results package** | one script regenerates every presented figure/table from checked-in artifacts (benchmark table, quality-vs-scale curve, per-species, anti-forgetting) | `python make_figures.py` → all assets, no hand edits |
| **M4 — end-to-end story** | robot live data → project dataset → cluster training (incl. the Mamba-YOLO deployment task) → model deployed on the lab server → detections rendered on the live robot stream | one recorded pass of the full chain on real hardware |
| **M0 — hygiene (parallel)** | Kaggle token rotation + license sweep + `requirements.lock` | each is a one-line verifiable state |

Ordering: M1 and M0 start immediately (M1 is wall-clock bound on cluster queue+training);
M2 builds on existing endpoints (`/api/domain/round/*`, rounds ledger); M3 after M1
lands; M4 reuses the shipped robot uplink (v3.20-3.22) and the model gateway.
