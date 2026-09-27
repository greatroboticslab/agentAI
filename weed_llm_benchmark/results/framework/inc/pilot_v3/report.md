# INC report: pilot_v3

- Type: chain; seeds [0, 1, 2]; generated 2026-09-27T15:01:23Z
- Done: yes (2026-09-27T14:51:38Z)
- Replay mode: full rehearsal (cand on the chain's whole accepted pool + D_k, null on the whole accepted pool)
- Gate: the flips guard counts net flips (protocol v2: negative - positive flips; a positive flip is an image incorrect under the incumbent and correct under the run); pinned at init (state.json gate_pin) from exp.json gate block {"flips_mode": "net"}; GateConfig metric=map50_95, p_accept=0.75, p_reject=0.25, p_recipe_flag=0.25, regression_sd_mult=2.0, species_min_drop=0.03, species_sd_mult=3.0, min_species_gt=30, flips_sd_mult=2.0, flips_slack_images=3.0, attr_drop_sd_mult=2.0, min_seeds=3, require_production=True, flips_mode=net
- Gate config check: the ledger's 21 gate, 12 soup and 7 truth entries and its gate_pin entry record the pinned config
- Decision code pinned 2026-09-27T12:24:45Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py 6e9aad4be85e, gate.py 48feed381afa, splits.py 79bd12f270a6

## Decisions per step

Chain verdicts against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts).

| Step | Clean | Truth (P) | freeze (P_data) | full (P_data) | lora (P_data) | agrees: freeze | agrees: full | agrees: lora | v1 same runs: freeze | v1 same runs: full | v1 same runs: lora |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | yes | hurts (0.33) | ACCEPT (0.89) | ACCEPT (1.00) | ACCEPT (0.78) | no | no | no | ACCEPT (agrees: no) | REJECT (agrees: yes) | ACCEPT (agrees: no) |
| s02_I2 | yes | helps (1.00) | ACCEPT (1.00) | ACCEPT (1.00) | REJECT (1.00) | yes | yes | no | REJECT (agrees: no) | ACCEPT (agrees: yes) | REJECT (agrees: no) |
| s03_Bswap | no | hurts (0.22) | REJECT (0.00) | REJECT (0.00) | REJECT (0.00) | yes | yes | yes | REJECT (agrees: yes) | REJECT (agrees: yes) | REJECT (agrees: yes) |
| s04_I3 | yes | helps (1.00) | REJECT (1.00) | ACCEPT (1.00) | ACCEPT (1.00) | no | yes | yes | REJECT (agrees: no) | REJECT (agrees: no) | REJECT (agrees: no) |
| s05_Breal | no | hurts (0.22) | REJECT (0.89) | REJECT (0.67) | HOLD (0.67) | yes | yes | no | REJECT (agrees: yes) | REJECT (agrees: yes) | REJECT (agrees: yes) |
| s06_I4 | yes | neutral (0.67) | ACCEPT (0.89) | ACCEPT (1.00) | ACCEPT (0.89) | no | no | no | REJECT (agrees: no) | ACCEPT (agrees: no) | REJECT (agrees: no) |
| s07_I5 | yes | helps (1.00) | REJECT (1.00) | ACCEPT (0.89) | ACCEPT (0.89) | no | yes | yes | REJECT (agrees: no) | ACCEPT (agrees: yes) | REJECT (agrees: no) |

- freeze agrees with the truth arm on 3 of 7 compared steps; v1 on the same runs would on 2 of 7
- full agrees with the truth arm on 5 of 7 compared steps; v1 on the same runs would on 5 of 7
- lora agrees with the truth arm on 3 of 7 compared steps; v1 on the same runs would on 2 of 7

## Protocol v2 check (pre-registered)

docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)", Pre-registration. The v1 verdict of each step is gate.v1_counterfactual: the same runs, P_data and regression and species guards, with v1's flips guard on the negative counts the net guard recorded.

- Outcome: supported: v2 agrees with the truth arm on more steps than v1 on the same runs, and accepted no step not marked clean
- Agreement with the truth arm over 21 (chain, step) pairs: v2 11, v1 on the same runs 9
- Steps where the two rules differ: full I1: v2 ACCEPT, v1 REJECT, truth hurts; freeze I2: v2 ACCEPT, v1 REJECT, truth helps; full I3: v2 ACCEPT, v1 REJECT, truth helps; lora I3: v2 ACCEPT, v1 REJECT, truth helps; lora Breal: v2 HOLD, v1 REJECT, truth hurts; freeze I4: v2 ACCEPT, v1 REJECT, truth neutral; lora I4: v2 ACCEPT, v1 REJECT, truth neutral; lora I5: v2 ACCEPT, v1 REJECT, truth helps
- Steps not marked clean that v2 accepted: none

## Gate details

### freeze (done)

Accepted ['I1', 'I2', 'I4']; neutral none; quarantined ['Bswap', 'I3', 'Breal', 'I5']; final incumbent freeze__s06_I4__cand__s0.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | ACCEPT | 0.889 | 0.000 | 0.7282 | 0.7300 +- 0.0044 | 0.7237 +- 0.0023 | 1778/1540 | pass | pass | pass | - | none | - | soup |
| s02_I2 | ACCEPT | 1.000 | 0.000 | 0.7305 | 0.7429 +- 0.0026 | 0.7178 +- 0.0021 | 2016/1778 | pass | pass | pass | - | none | - | soup |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7439 | 0.7255 +- 0.0047 | 0.7389 +- 0.0033 | 2288/2016 | FAIL | FAIL | pass | recipe | domain/localisation | PalmerAmaranth, Goosegrass, CutleafGroundcherry | - |
| s04_I3 | REJECT | 1.000 | 0.000 | 0.7439 | 0.7452 +- 0.0014 | 0.7389 +- 0.0033 | 2254/2016 | pass | FAIL | pass | recipe | none | Goosegrass | - |
| s05_Breal | REJECT | 0.889 | 0.000 | 0.7439 | 0.7428 +- 0.0015 | 0.7389 +- 0.0033 | 2254/2016 | pass | pass | FAIL | recipe | none | - | - |
| s06_I4 | ACCEPT | 0.889 | 0.000 | 0.7439 | 0.7454 +- 0.0047 | 0.7389 +- 0.0033 | 2265/2016 | pass | pass | pass | - | none | - | cand0 |
| s07_I5 | REJECT | 1.000 | 1.000 | 0.7406 | 0.7475 +- 0.0027 | 0.7428 +- 0.0001 | 2539/2265 | pass | FAIL | FAIL | data | none | PalmerAmaranth | - |

### full (done)

Accepted ['I1', 'I2', 'I3', 'I4', 'I5']; neutral none; quarantined ['Bswap', 'Breal']; final incumbent full__s07_I5__soup.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | ACCEPT | 1.000 | 0.000 | 0.7282 | 0.7316 +- 0.0038 | 0.7134 +- 0.0021 | 1778/1540 | pass | pass | pass | - | none | - | soup |
| s02_I2 | ACCEPT | 1.000 | 0.000 | 0.7328 | 0.7571 +- 0.0025 | 0.7204 +- 0.0084 | 2016/1778 | pass | pass | pass | - | none | - | soup |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7597 | 0.7336 +- 0.0036 | 0.7539 +- 0.0033 | 2288/2016 | FAIL | FAIL | FAIL | recipe | domain/localisation | Eclipta, PalmerAmaranth, Goosegrass | - |
| s04_I3 | ACCEPT | 1.000 | 0.000 | 0.7597 | 0.7732 +- 0.0039 | 0.7539 +- 0.0033 | 2254/2016 | pass | pass | pass | - | none | - | soup |
| s05_Breal | REJECT | 0.667 | 0.000 | 0.7749 | 0.7732 +- 0.0057 | 0.7708 +- 0.0016 | 2492/2254 | pass | FAIL | pass | recipe | none | PalmerAmaranth | - |
| s06_I4 | ACCEPT | 1.000 | 0.000 | 0.7749 | 0.7874 +- 0.0032 | 0.7708 +- 0.0016 | 2503/2254 | pass | pass | pass | - | none | - | soup |
| s07_I5 | ACCEPT | 0.889 | 0.000 | 0.7942 | 0.7989 +- 0.0068 | 0.7846 +- 0.0064 | 2777/2503 | pass | pass | pass | - | none | - | soup |

### lora (done)

Accepted ['I1', 'I3', 'I4', 'I5']; neutral ['Breal']; quarantined ['I2', 'Bswap']; final incumbent lora__s07_I5__soup.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | ACCEPT | 0.778 | 0.667 | 0.7282 | 0.7301 +- 0.0036 | 0.7285 +- 0.0025 | 1778/1540 | pass | pass | pass | - | none | - | soup |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7341 | 0.7489 +- 0.0035 | 0.7290 +- 0.0023 | 2016/1778 | pass | FAIL | pass | recipe | none | PalmerAmaranth | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7341 | 0.7181 +- 0.0057 | 0.7290 +- 0.0023 | 2050/1778 | FAIL | FAIL | pass | recipe | domain/localisation | PalmerAmaranth, Sicklepod | - |
| s04_I3 | ACCEPT | 1.000 | 0.000 | 0.7341 | 0.7477 +- 0.0011 | 0.7290 +- 0.0023 | 2016/1778 | pass | pass | pass | - | none | - | soup |
| s05_Breal | HOLD | 0.667 | 0.000 | 0.7509 | 0.7435 +- 0.0017 | 0.7419 +- 0.0045 | 2254/2016 | pass | pass | pass | - | none | - | - |
| s06_I4 | ACCEPT | 0.889 | 0.000 | 0.7509 | 0.7475 +- 0.0059 | 0.7419 +- 0.0045 | 2265/2016 | pass | pass | pass | - | none | - | soup |
| s07_I5 | ACCEPT | 0.889 | 0.333 | 0.7489 | 0.7520 +- 0.0044 | 0.7451 +- 0.0045 | 2539/2265 | pass | pass | pass | - | none | - | soup |

## Attribution of Bswap

- freeze: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- full: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- lora: REJECT, class_vs_loc=domain/localisation, attributed to labels: no

## Attribution not run

- Step 4: label audit (BioCLIP-2 re-reading D_k's boxes): out of scope for the pilot
- Step 5: leave-one-source-out cand runs for a multi-source increment (Breal mixes two sources): out of scope for the pilot
- freeze: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal
- full: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal

## Warmup as run

Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), 100) iterations; replay mode full: cand holds the accepted pool + D_k and null the pool, recorded at pool_min (P0) and pool_max (P0 + every earlier increment); the driver records each step's actual sizes. Incremental runs: warmup lasts 1.00-2.04 of their 30 epochs (the recipe asks for 1); the cold base run's lasts 3.00 of 100.

From the decided steps' train sizes: cand warmup 1.15-1.79, null warmup 1.27-2.04 epochs.

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| chain freeze: final incumbent | 0.7406 | 0.8184 | 0.7133 | 0.7956 | 0.0994 | 0.1196 | 0.0266 | 0.1079 | 0.7992 | 0.8483 |
| chain full: final incumbent | 0.8006 | 0.8454 | 0.7630 | 0.8265 | 0.1833 | 0.1174 | 0.0489 | 0.1064 | 0.8475 | 0.8667 |
| chain lora: final incumbent | 0.7558 | 0.8349 | 0.7465 | 0.8209 | 0.1161 | 0.1195 | 0.0361 | 0.1004 | 0.8132 | 0.8639 |
| base P0 | 0.7212 +- 0.0085 | 0.8195 +- 0.0028 | 0.7041 +- 0.0118 | 0.8051 +- 0.0032 | 0.0870 +- 0.0078 | 0.1269 +- 0.0098 | 0.0423 +- 0.0132 | 0.1101 +- 0.0064 | 0.7761 +- 0.0063 | 0.8428 +- 0.0022 |
| T_final (union of clean data) | 0.8084 +- 0.0043 | 0.8548 +- 0.0037 | 0.7561 +- 0.0047 | 0.8372 +- 0.0031 | 0.1637 +- 0.0459 | 0.1252 +- 0.0037 | 0.0550 +- 0.0019 | 0.1223 +- 0.0090 | 0.8472 +- 0.0014 | 0.8717 +- 0.0009 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 1.00 |
| chain freeze | 45 | 4.72 |
| chain full | 47 | 6.04 |
| chain lora | 46 | 6.06 |
| final | 9 | 1.07 |
| truth | 21 | 8.90 |
| total | 171 | 27.78 |
