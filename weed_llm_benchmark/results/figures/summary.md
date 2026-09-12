# Headline numbers with their evidentiary state

- **YOLO11n · cwd12-only (sealed, n=3)** — 0.8755 ± 0.0029 (n=3) · guard: not-applicable (single dataset; train split only, holdout=test+valid as val) · jobs 44323305_[1-3] 2026-08-24, cwd12_sealed.yaml, 100 ep, seeds 101/102/103, per-seed [0.8739, 0.8789, 0.8737]; supersedes the single pre-split run 0.865 (RESEARCH_LOG Phase 1) which is kept below for history
- **RF-DETR L · cwd12-only** — 0.8974 ± 0.0040 (n=4) · guard: not-applicable (stages cwd12 directly, holdout stems excluded) · CHANGELOG 2026-05-22 (v3.0.31/34/38: 0.8949/0.8953/0.8961/0.9033)
- **yolo26x · safety-clean cwd12-only** — 0.8960 (n=1) · guard: not-applicable (cwd12-only staging, zero merge calls) · CHANGELOG v3.0.28
- **merged raw · 244K pseudo-labeled** — 0.5930 (n=1) · guard: PRE-GUARD (filename-only holdout defence at the time) · CHANGELOG v3.0.26 phase 2
- **merged cumulative · ~200K** — 0.5760 (n=1) · guard: PRE-GUARD · CHANGELOG v3.0.32
- **M1 raw · 55.7K merged (sealed)** — 0.6032 ± 0.0046 (n=3) · guard: SEALED (1,977 holdout dHashes pre-seeded) · jobs 44234060_[1-3] (h100, yolo26x@640, patience-20 early stop at ep 29/32/37); per-seed 0.6055/0.5979/0.6062 from results.csv; guard active (1,977 dHashes pre-seeded), train∩holdout stems = 0
- **M1 curated · 13.3K @ DINO≥0.50 (sealed)** — 0.5894 ± 0.0025 (n=3) · guard: SEALED · jobs 44234063_[1-3] (h100, same recipe; DINO gate skipped 36/45 slugs; early stop ep 37/44/46); per-seed 0.5873/0.5887/0.5921; guard active, train∩holdout stems = 0
- **YOLO11n · cwd12-only (2026-03, single run)** — 0.8650 (n=1) · guard: not-applicable · RESEARCH_LOG Phase 1 (job 38007481, 100 ep) — pre-split full set
- **Mamba-YOLO-T · cwd12-only (from scratch)** — 0.8266 ± 0.0064 (n=3) · guard: not-applicable (single dataset; sealed 1,977 holdout as val) · jobs 44351282_[1-3] 2026-08-24; random init — the fork releases no weights
- **YOLO11n · cwd12-only (from scratch, fairness control)** — 0.8041 ± 0.0028 (n=3) · guard: not-applicable (single dataset; sealed holdout as val) · jobs 44368952_[1-3] 2026-08-24; the like-for-like opponent for Mamba-YOLO-T

## Data governance (license audit + backfill, 2026-08-23)
- 45 registry datasets, **38 with an explicit license after backfill**; mix: {'CC BY 4.0': 33, 'CC0/Public Domain': 3, 'MIT': 1, 'AGPL-3.0': 1, 'unresolved': 5, 'unreachable': 2}
- Consequence: 38/45 datasets now carry an explicit license (33x CC BY 4.0 - reusable with attribution); the 7 unresolved/unreachable stay non-redistributable; the S1 harvest gate captures license at collection time
