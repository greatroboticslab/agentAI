# Label audit (INC attribution item 4)

Trusted: `/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/pilot_v1/manifests/P0.jsonl` (1540 images, 2604 species boxes used from 7 groups: 2604 by session, 0 by source). Probe: verify's 12-species logistic regression on BioCLIP-2 crop features, thresholds at 0.95 per-species recall (5-fold grouped CV); CV top-1 on species 0.987.

The noise estimate and 'above baseline' use the baseline for each set's own species mix: the held-out false-conflict rate f (95 % CI) and the conflict recall r on a wrong species of each species, weighted by the set's judged species-labelled boxes per label. The pooled columns use the baseline over the trusted set's own mix, for context.

| Set | Images | Species boxes judged | Conflict rate [95 % CI] | Verified | Unknown | Baseline f for its mix [95 % CI] | r for its mix | Noise estimate (corrected) | Above baseline | Pooled: estimate, above | OtherPlant conflict (n) | Not judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trusted, held-out CV (baseline) | 1540 | 2604 | 0.006 [0.004, 0.010] | 0.793 | 0.201 | - | 0.799 | - | - | - | - | 0 |
| I1 | 238 | 406 | 0.007 [0.003, 0.021] | 0.931 | 0.062 | 0.010 [0.006, 0.024] | 0.797 | 0.000 | no | 0.002, no | - (0) | 0 |
| I2 | 238 | 312 | 0.003 [0.001, 0.018] | 0.971 | 0.026 | 0.001 [0.001, 0.020] | 0.682 | 0.003 | no | 0.000, no | - (0) | 0 |
| I3 | 238 | 431 | 0.007 [0.002, 0.020] | 0.968 | 0.025 | 0.007 [0.005, 0.014] | 0.744 | 0.000 | no | 0.001, no | - (0) | 0 |
| I4 | 249 | 380 | 0.026 [0.014, 0.048] | 0.950 | 0.024 | 0.007 [0.005, 0.014] | 0.912 | 0.022 | yes | 0.025, yes | - (0) | 0 |
| I5 | 274 | 404 | 0.010 [0.004, 0.025] | 0.951 | 0.040 | 0.006 [0.005, 0.013] | 0.797 | 0.005 | no | 0.005, no | - (0) | 0 |
| Bswap | 272 | 492 | 0.380 [0.338, 0.424] | 0.555 | 0.065 | 0.008 [0.006, 0.021] | 0.706 | 0.533 | yes | 0.472, yes | - (0) | 0 |
| Breal | 238 | 360 | 0.078 [0.054, 0.110] | 0.228 | 0.694 | 0.007 [0.003, 0.020] | 0.942 | 0.075 | yes | 0.090, yes | 0.173 (2323) | 57 |

Baseline conflict recall on a wrong species (held-out trusted boxes, one uniform wrong species each): 0.799 over 2604 boxes.

## Known truth (boxes judged on a train_core crop)

| Set | Boxes | True label error rate | Conflict recall on wrong labels | False conflict on right labels | Verified precision |
|---|---|---|---|---|---|
| I1 | 406 | 0.000 | - | 0.007 | 1.000 |
| I2 | 312 | 0.000 | - | 0.003 | 1.000 |
| I3 | 431 | 0.000 | - | 0.007 | 1.000 |
| I4 | 380 | 0.000 | - | 0.026 | 1.000 |
| I5 | 404 | 0.000 | - | 0.010 | 1.000 |
| Bswap | 492 | 0.400 | 0.924 | 0.017 | 1.000 |

## Conflict rate per species (of the label given; boxes judged)

| Species | baseline f (n) | baseline r | I1 | I2 | I3 | I4 | I5 | Bswap | Breal |
|---|---|---|---|---|---|---|---|---|---|
| Waterhemp | 0.011 (470) | 0.949 | 0.000 (124) | 0.000 (2) | 0.014 (144) | 0.000 (134) | 0.000 (81) | 0.175 (80) | 0.199 (136) |
| MorningGlory | 0.000 (385) | 0.940 | 0.000 (45) | 0.006 (166) | 0.000 (68) | 0.021 (48) | 0.000 (67) | 0.739 (23) | - |
| Purslane | 0.009 (353) | 0.926 | 0.000 (18) | 0.000 (11) | 0.000 (5) | 0.000 (2) | 0.000 (26) | 0.433 (30) | - |
| SpottedSpurge | 0.006 (352) | 0.869 | 0.038 (26) | 0.000 (28) | 0.000 (14) | 0.013 (77) | 0.000 (24) | 0.548 (31) | - |
| Carpetweed | 0.011 (185) | 0.957 | 0.020 (100) | 0.000 (1) | 0.015 (66) | 0.000 (14) | 0.046 (43) | 0.158 (95) | - |
| Ragweed | 0.005 (194) | 0.938 | 0.000 (9) | 0.000 (4) | 0.000 (27) | 0.000 (62) | 0.000 (73) | 0.191 (21) | 0.004 (224) |
| Eclipta | 0.004 (280) | 0.004 | 0.000 (21) | - | 0.000 (85) | 0.000 (4) | 0.038 (52) | 0.304 (56) | - |
| PricklySida | 0.005 (188) | 0.957 | - | 0.000 (12) | 0.000 (7) | 0.037 (27) | 0.000 (14) | 0.630 (27) | - |
| PalmerAmaranth | 0.021 (48) | 0.312 | 0.000 (62) | 0.000 (1) | 0.000 (9) | 0.000 (1) | 0.000 (9) | 0.164 (55) | - |
| Sicklepod | 0.000 (36) | 0.000 | - | 0.000 (83) | - | 1.000 (2) | - | 0.905 (21) | - |
| Goosegrass | 0.000 (81) | 0.667 | 0.000 (1) | - | 0.000 (3) | 0.000 (2) | 0.000 (11) | 0.844 (32) | - |
| CutleafGroundcherry | 0.000 (32) | 0.938 | - | 0.000 (4) | 0.000 (3) | 0.714 (7) | 0.000 (4) | 0.857 (21) | - |
| OtherPlant | - | - | - | - | - | - | - | - | 0.173 (2323) |

## Boxes not judged

| Set | in_trusted | label_unseen | no_embedding | reasons |
|---|---|---|---|---|
| I1 | 0 | 0 | 0 | - |
| I2 | 0 | 0 | 0 | - |
| I3 | 0 | 0 | 0 | - |
| I4 | 0 | 0 | 0 | - |
| I5 | 0 | 0 | 0 | - |
| Bswap | 0 | 0 | 0 | - |
| Breal | 0 | 0 | 57 | geometry_mismatch 57 |

## Notes

- conflict_rate: conflicts / species-labelled boxes with a verdict (verified, conflict, unknown); the label-noise estimate. Boxes not judged (no_embedding, in_trusted, label_unseen) are counted, never guessed.
- baseline_cv: held-out trusted boxes judged by a verifier fitted, thresholds included, on the other folds (grouped by capture session, else source): the false-conflict rate f on true labels, and the conflict recall r under one uniform wrong species per box.
- baseline_for_mix: f and r of each species (baseline_cv.per_species) weighted by the audited set's judged species-labelled boxes per label; f's 95 % interval from seeded draws of each species' f from its Jeffreys posterior. Species without a baseline entry are listed and left out, with their boxes.
- noise_estimate_corrected = (c - f) / (r - f) with f and r of baseline_for_mix, clipped to [0, 1]; it assumes in-domain labels and uniform swaps, so it is indicative only for another domain.
- above_baseline: the conflict rate's Wilson 95 % interval lies wholly above baseline_for_mix's interval of f.
- noise_estimate_pooled_baseline, above_pooled_baseline: the same against the baseline pooled over the trusted set's own species mix; context only, since per-species f and r differ and an audited set's mix need not be the trusted set's.
- known_truth: boxes judged on a train_core crop, whose true label is the crop's train_core label.

Per-box verdicts: `pilot_v1_audit_boxes.csv` (beside this file).
