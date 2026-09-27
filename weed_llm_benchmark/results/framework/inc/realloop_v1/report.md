# INC report: realloop_v1

- Type: chain; seeds [0, 1, 2]; generated 2026-09-27T22:38:43Z
- Done: yes (2026-09-27T22:37:58Z)
- Replay mode: full rehearsal (cand on the chain's whole accepted pool + D_k, null on the whole accepted pool)
- Gate: the flips guard counts net flips (protocol v2: negative - positive flips; a positive flip is an image incorrect under the incumbent and correct under the run); pinned at init (state.json gate_pin) from exp.json gate block {"flips_mode": "net"}; GateConfig metric=map50_95, p_accept=0.75, p_reject=0.25, p_recipe_flag=0.25, regression_sd_mult=2.0, species_min_drop=0.03, species_sd_mult=3.0, min_species_gt=30, flips_sd_mult=2.0, flips_slack_images=3.0, attr_drop_sd_mult=2.0, min_seeds=3, require_production=True, flips_mode=net
- Gate config check: the ledger's 6 gate, 0 soup and 6 truth entries and its gate_pin entry record the pinned config
- Decision code pinned 2026-09-27T19:46:23Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py 6e9aad4be85e, gate.py 48feed381afa, splits.py 79bd12f270a6

## Decisions per step

Chain verdicts against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts).

| Step | Clean | Truth (P) | full (P_data) | agrees: full | v1 same runs: full |
|---|---|---|---|---|---|
| s01_V1 | yes | neutral (0.67) | REJECT (0.11) | no | REJECT (agrees: no) |
| s02_V2 | yes | neutral (0.44) | REJECT (0.44) | no | REJECT (agrees: no) |
| s03_UNVERIFIED | no | helps (0.89) | REJECT (0.11) | no | REJECT (agrees: no) |
| s04_V3 | yes | neutral (0.33) | REJECT (0.44) | no | REJECT (agrees: no) |
| s05_OTHER_HEAVY | yes | hurts (0.67) | REJECT (1.00) | yes | REJECT (agrees: yes) |
| s06_V4 | yes | hurts (0.33) | REJECT (1.00) | yes | REJECT (agrees: yes) |

- full agrees with the truth arm on 2 of 6 compared steps; v1 on the same runs would on 2 of 6

## Protocol v2 check (pre-registered)

docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)", Pre-registration. The v1 verdict of each step is gate.v1_counterfactual: the same runs, P_data and regression and species guards, with v1's flips guard on the negative counts the net guard recorded.

- Outcome: inconclusive: v2 and v1 on the same runs agree with the truth arm on as many steps, and v2 accepted no step not marked clean
- Agreement with the truth arm over 6 (chain, step) pairs: v2 2, v1 on the same runs 2
- Steps where the two rules differ: none
- Steps not marked clean that v2 accepted: none

## Gate details

### full (done)

Accepted none; neutral none; quarantined ['V1', 'V2', 'UNVERIFIED', 'V3', 'OTHER_HEAVY', 'V4']; final incumbent base__s0.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_V1 | REJECT | 0.111 | 0.000 | 0.8189 | 0.8135 +- 0.0021 | 0.8164 +- 0.0008 | 4214/3927 | FAIL | FAIL | pass | recipe | labels | PricklySida | - |
| s02_V2 | REJECT | 0.444 | 0.000 | 0.8189 | 0.8164 +- 0.0040 | 0.8164 +- 0.0008 | 4214/3927 | FAIL | FAIL | pass | recipe | none | PricklySida | - |
| s03_UNVERIFIED | REJECT | 0.111 | 0.000 | 0.8189 | 0.8135 +- 0.0033 | 0.8164 +- 0.0008 | 4214/3927 | FAIL | FAIL | pass | recipe | labels | PricklySida | - |
| s04_V3 | REJECT | 0.444 | 0.000 | 0.8189 | 0.8150 +- 0.0032 | 0.8164 +- 0.0008 | 4214/3927 | FAIL | FAIL | pass | recipe | none | PricklySida | - |
| s05_OTHER_HEAVY | REJECT | 1.000 | 0.000 | 0.8189 | 0.8199 +- 0.0018 | 0.8164 +- 0.0008 | 4214/3927 | pass | pass | FAIL | recipe | none | - | - |
| s06_V4 | REJECT | 1.000 | 0.000 | 0.8189 | 0.8211 +- 0.0014 | 0.8164 +- 0.0008 | 4214/3927 | pass | FAIL | pass | recipe | none | PricklySida | - |

## Attribution of UNVERIFIED

UNVERIFIED: 287 images of rf_tuf__weed-3434e that verify did not admit (image verdicts conflict 287); its labels are as the species join gave them.

- full: REJECT, class_vs_loc=labels, attributed to labels: yes

## Attribution not run

- Step 4: label audit: not re-run per step; the Step 1 verifier judged every box already (every species box of a verified increment is verified; the unverified increment's verdicts are in build_summary.json 'unverified')
- Step 5: leave-one-source-out cand runs for a multi-source increment: not run by the driver; each increment's per-source counts are in build_summary.json
- full: REJECT of a multi-source increment without leave-one-source-out attribution: s01_V1, s02_V2, s04_V3, s05_OTHER_HEAVY, s06_V4

## Warmup as run

Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), 100) iterations; replay mode full: cand holds the accepted pool + D_k and null the pool, recorded at pool_min (P0) and pool_max (P0 + every earlier increment); the driver records each step's actual sizes. Incremental runs: warmup lasts 1.00-1.00 of their 30 epochs (the recipe asks for 1); the cold base run's lasts 3.00 of 100.

From the decided steps' train sizes: cand warmup 1.00-1.00, null warmup 1.00-1.00 epochs.

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| chain full: final incumbent | 0.8189 | 0.8524 | 0.7718 | 0.8454 | 0.2822 | 0.1295 | 0.0174 | 0.0411 | 0.8552 | 0.8730 |
| base base_B | 0.8133 +- 0.0070 | 0.8532 +- 0.0010 | 0.7752 +- 0.0093 | 0.8458 +- 0.0030 | 0.1901 +- 0.0801 | 0.1355 +- 0.0062 | 0.0160 +- 0.0026 | 0.0425 +- 0.0036 | 0.8502 +- 0.0059 | 0.8730 +- 0.0024 |
| T_final (union of clean data) | 0.8093 +- 0.0027 | 0.8523 +- 0.0011 | 0.7734 +- 0.0053 | 0.8440 +- 0.0013 | 0.2030 +- 0.0250 | 0.1340 +- 0.0091 | 0.0223 +- 0.0108 | 0.0684 +- 0.0012 | 0.8410 +- 0.0041 | 0.8695 +- 0.0025 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 1.97 |
| chain full | 36 | 9.05 |
| final | 7 | 0.83 |
| truth | 18 | 14.34 |
| total | 64 | 26.20 |
