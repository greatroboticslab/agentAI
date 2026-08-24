# Plan-compliance audit — is execution actually following SUPERWEED_PLAN?

*2026-08-23, ~24 h into the campaign. Checks the running work against
`SUPERWEED_PLAN.md`'s standing constraints and phase gates. Written to be useful
rather than flattering: a gate is met or it is not, and a deviation is named even
when the outcome was good.*

---

## 1. Standing constraints (§1) — violating any is a defect

| # | Constraint | State | Evidence |
|---|---|---|---|
| 1 | **Holdout sanctity** | ✅ held | guard logged 1,977 dHashes pre-seeded in every merge; file-level check found 0 holdout stems in the 55,690-image train set |
| 2 | **Honesty, log-verified** | ✅ held, and it did work | three "looks-fine" defects caught by this rule: the silently-disabled DINO gate, the erased quarantine, the stale metric |
| 3 | **Registry discipline** | ⚠️ **one real incident** | pre-campaign backup exists and locked writes were in force, but the documented last-writer-wins window erased a supervisory quarantine on 2026-08-23. Fixed (merge-write, `f89d54c`) and re-verified. The S1 gate says "zero registry incidents" — **this counts as one** |
| 4 | **Cluster etiquette** | ⚠️ **deviation** | M1 followed the rule (login-node import check + `sbatch --test-only` before submitting). The Mamba builds did not: four jobs were submitted with the toolchain unverified. They were cheap (2–5 min each, ~1 SU total) but the rule exists precisely to avoid that sequence |
| 5 | **Budget ≤4,000 SU** | ✅ held | 84 SU consumed of the envelope (2 %); balance 11,392 → 11,308 |
| 6 | **Data governance** | ✅ held | licenses captured at collection + backfilled; 41 of 47 pre-round datasets explicit, 0 unreachable; the newly harvested sets resolve on the next backfill |
| 7 | **Platform-first** | ✅ held | harvest fired through `POST /api/cluster_action/brain_harvest`; the scheduler drives the platform's own rounds ledger, which the project page reads |

## 2. Phase gates — met, or not

| Phase | Gate as written | Verdict |
|---|---|---|
| **S0** | M1 published, registry backup, tokens live | ✅ **MET** — sealed results in RESEARCH_LOG (raw 0.6032 ± 0.0046, curated 0.5894 ± 0.0025), backup `.bak-20260822-222730`, Kaggle rotated + HF live |
| **S1** | +10,000 audited-pool images **with scorecards**, **zero registry incidents** | ❌ **NOT MET** — three deficits: (a) +8,208 labelled images, short of 10,000; (b) **the per-round scorecard was never built** — no `images found / unique-new / weed-relevant % / label-source / license mix` record exists per round; (c) one registry incident occurred (§1.3) |
| **S2** | pool report regenerable by script | ✅ **MET (same day)** — `tools/pool_report.py report --json` emits size, class balance, DINO quality histogram, license mix, quarantine list and the audited-pool criteria; `scorecards` writes a per-source card into each registry entry. First run exposed and fixed a label-counting error in the tool itself (see §3.3) |
| **S3** | ≥12 ledgered runs across ≥3 method families | ❌ **NOT STARTED** — 7 runs so far, all one family (yolo26x). Mamba-YOLO's extension only became buildable today; RF-DETR not re-run |
| **S4** | ≥24 h unattended, ≥2 complete rounds, zero fake-success | 🔄 **IN PROGRESS** — round 1 is at collect ✅ → filter ✅ → label skipped → train running (5 h). Zero fake-success **so far, but only because the stale-metric bug was caught before publishing**, not because it never happened |
| **S5** | clean checkout → one command → byte-stable assets | ⚠️ **PARTIAL** — `make_figures.py` regenerates everything and auto-fills M1, but it has only ever run on the lab server against a hand-copied data file; byte-stability was never tested |
| **S6** | deployment integration | ⏸ not started (correctly — it follows S3) |

## 3. Deviations worth naming

1. **Two S1 mechanisms were specified and never built** — the per-source **sample-audit
   protocol** (≥25 images sampled, OWLv2 confidence histogram, DINO verifier score,
   montage saved to the project, ≥90 % precision required before a source joins the
   train pool) and the **per-round scorecard**. Garbage has instead been caught by
   supervisor judgement plus, since v3.22.13, the label gate. That works, but it is not
   the plan, it does not scale past a human in the loop, and the S1 gate cannot be
   honestly declared met without it.
2. **Quarantine is enforced but invisible** — the plan says quarantined sources are
   "listed greyed-out" on the platform. The merge skips them with a counted stat and
   the CLI lists them, but no UI surfaces them, so an operator sees no trace.
3. **D1's target is already met, and the first measurement of it was wrong.** The pool
   report initially read **15,789 labelled** because `local_labeled` is a field only
   later harvests populate — legacy entries report 0 even for `cottonweed_sp8`, the
   in-domain core the merge trains on happily. Counting label files on disk (bounded to
   `labels*` directories, never descending into image trees) gives the real figure:
   **120,515 labelled images across 50 audited-pool sources, 121,721 active**. D1's
   "≥50,000 unique weed-relevant detection images" is therefore **met on volume**; what
   remains open is the *audited* half of that requirement (§3.1). The plan should still
   say **labelled** explicitly, since that is the number that trains.
4. **S3's tier ladder is reachable, but the merge does not currently reach it** — the
   labelled pool is 120,515 while the largest merge produced 55,690. The gap is the
   merge's own filters (annotation kind, class mapping, dedup, holdout). Worth
   measuring before assuming more harvest is the answer.

## 4. What is genuinely ahead of plan

- The **compounding effect is already visible**: round 1's training, on a pool that
  grew by the harvest's +8,208 labelled images, reads **0.6019 at epoch 17** versus the
  sealed M1 curated **0.5894 ± 0.0025** at the same recipe. Suggestive, not yet a
  result — it is one seed and an unfinished run.
- **Mamba-YOLO builds** (S3's hardest blocker) a full phase before S3 starts.
- The supervision loop caught three publish-grade defects in 24 h, which is the
  campaign's real thesis working.

## 5. Corrective actions (ordered)

1. Build the **image-level sample-audit tool** (≥25 images per source, OWLv2
   confidence histogram, DINO verifier score, montage) — the last missing S1
   mechanism. The per-source **scorecard** now exists (`pool_report.py scorecards`).
2. ~~Write `pool_report.py`~~ — **done**, S2 gate met.
3. Surface **quarantined datasets in the UI**, greyed-out with their reason.
4. Amend the plan: state **labelled** images in D1/S1/S3 targets, and schedule the
   autolabel decision that the tier ladder depends on.
5. Run `make_figures.py` from a **clean checkout** twice and diff, to earn S5's gate.
6. Restore cluster etiquette on every future build/submit — no exceptions for "cheap"
   jobs.
