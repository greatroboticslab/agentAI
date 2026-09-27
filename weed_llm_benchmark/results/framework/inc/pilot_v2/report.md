# INC report: pilot_v2

- Type: chain; seeds [0, 1, 2]; generated 2026-09-27T07:51:23Z
- Done: yes (2026-09-27T07:49:08Z)
- Replay mode: full rehearsal (cand on the chain's whole accepted pool + D_k, null on the whole accepted pool)
- Decision code pinned 2026-09-27T05:55:39Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py ae9ffa8daedd, gate.py e6eadeb76d0d, splits.py 79bd12f270a6

## Decisions per step

Chain verdicts against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts).

| Step | Clean | Truth (P) | freeze (P_data) | full (P_data) | lora (P_data) | agrees: freeze | agrees: full | agrees: lora |
|---|---|---|---|---|---|---|---|---|
| s01_I1 | yes | hurts (0.33) | ACCEPT (0.89) | REJECT (1.00) | ACCEPT (0.78) | no | yes | no |
| s02_I2 | yes | helps (1.00) | REJECT (1.00) | REJECT (1.00) | REJECT (1.00) | no | no | no |
| s03_Bswap | no | hurts (0.00) | REJECT (0.00) | REJECT (0.00) | REJECT (0.33) | yes | yes | yes |
| s04_I3 | yes | helps (1.00) | REJECT (1.00) | REJECT (1.00) | REJECT (1.00) | no | no | no |
| s05_Breal | no | hurts (0.00) | REJECT (0.67) | REJECT (0.00) | REJECT (0.22) | yes | yes | yes |
| s06_I4 | yes | neutral (0.67) | REJECT (1.00) | REJECT (1.00) | REJECT (0.78) | no | no | no |
| s07_I5 | yes | helps (1.00) | REJECT (1.00) | ACCEPT (1.00) | REJECT (1.00) | no | yes | no |

- freeze agrees with the truth arm on 2 of 7 compared steps
- full agrees with the truth arm on 4 of 7 compared steps
- lora agrees with the truth arm on 2 of 7 compared steps

## Gate details

### freeze (done)

Accepted ['I1']; neutral none; quarantined ['I2', 'Bswap', 'I3', 'Breal', 'I4', 'I5']; final incumbent freeze__s01_I1__soup.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | ACCEPT | 0.889 | 0.000 | 0.7282 | 0.7300 +- 0.0044 | 0.7237 +- 0.0023 | 1778/1540 | pass | pass | pass | - | none | - | soup |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7305 | 0.7429 +- 0.0026 | 0.7178 +- 0.0021 | 2016/1778 | pass | pass | FAIL | recipe | none | - | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7305 | 0.7025 +- 0.0026 | 0.7178 +- 0.0021 | 2050/1778 | FAIL | FAIL | FAIL | recipe | domain/localisation | PricklySida, Sicklepod, CutleafGroundcherry | - |
| s04_I3 | REJECT | 1.000 | 0.000 | 0.7305 | 0.7296 +- 0.0016 | 0.7178 +- 0.0021 | 2016/1778 | pass | pass | FAIL | recipe | none | - | - |
| s05_Breal | REJECT | 0.667 | 0.000 | 0.7305 | 0.7200 +- 0.0027 | 0.7178 +- 0.0021 | 2016/1778 | FAIL | FAIL | FAIL | recipe | none | Sicklepod | - |
| s06_I4 | REJECT | 1.000 | 0.000 | 0.7305 | 0.7253 +- 0.0012 | 0.7178 +- 0.0021 | 2027/1778 | FAIL | pass | FAIL | recipe | none | - | - |
| s07_I5 | REJECT | 1.000 | 0.000 | 0.7305 | 0.7264 +- 0.0004 | 0.7178 +- 0.0021 | 2052/1778 | FAIL | FAIL | FAIL | recipe | none | Goosegrass | - |

### full (done)

Accepted ['I5']; neutral none; quarantined ['I1', 'I2', 'Bswap', 'I3', 'Breal', 'I4']; final incumbent full__s07_I5__soup.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7316 +- 0.0038 | 0.7134 +- 0.0021 | 1778/1540 | pass | pass | FAIL | recipe | none | - | - |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7535 +- 0.0057 | 0.7134 +- 0.0021 | 1778/1540 | pass | pass | FAIL | recipe | none | - | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7282 | 0.6989 +- 0.0026 | 0.7134 +- 0.0021 | 1812/1540 | FAIL | FAIL | FAIL | recipe | domain/localisation | Ragweed, Eclipta, PalmerAmaranth, CutleafGroundcherry | - |
| s04_I3 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7376 +- 0.0043 | 0.7134 +- 0.0021 | 1778/1540 | pass | pass | FAIL | recipe | none | - | - |
| s05_Breal | REJECT | 0.000 | 0.000 | 0.7282 | 0.7082 +- 0.0022 | 0.7134 +- 0.0021 | 1778/1540 | FAIL | FAIL | FAIL | recipe | labels | CutleafGroundcherry | - |
| s06_I4 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7283 +- 0.0028 | 0.7134 +- 0.0021 | 1789/1540 | pass | pass | FAIL | recipe | none | - | - |
| s07_I5 | ACCEPT | 1.000 | 0.000 | 0.7282 | 0.7338 +- 0.0060 | 0.7134 +- 0.0021 | 1814/1540 | pass | pass | pass | - | none | - | soup |

### lora (done)

Accepted ['I1']; neutral none; quarantined ['I2', 'Bswap', 'I3', 'Breal', 'I4', 'I5']; final incumbent lora__s01_I1__soup.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | ACCEPT | 0.778 | 0.667 | 0.7282 | 0.7301 +- 0.0036 | 0.7285 +- 0.0025 | 1778/1540 | pass | pass | pass | - | none | - | soup |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7341 | 0.7489 +- 0.0035 | 0.7290 +- 0.0023 | 2016/1778 | pass | FAIL | FAIL | recipe | none | PalmerAmaranth | - |
| s03_Bswap | REJECT | 0.333 | 0.000 | 0.7341 | 0.7256 +- 0.0046 | 0.7290 +- 0.0023 | 2050/1778 | FAIL | FAIL | FAIL | recipe | none | PalmerAmaranth, Sicklepod | - |
| s04_I3 | REJECT | 1.000 | 0.000 | 0.7341 | 0.7477 +- 0.0011 | 0.7290 +- 0.0023 | 2016/1778 | pass | pass | FAIL | recipe | none | - | - |
| s05_Breal | REJECT | 0.222 | 0.000 | 0.7341 | 0.7248 +- 0.0041 | 0.7290 +- 0.0023 | 2016/1778 | FAIL | FAIL | FAIL | recipe | none | PalmerAmaranth, Sicklepod | - |
| s06_I4 | REJECT | 0.778 | 0.000 | 0.7341 | 0.7327 +- 0.0042 | 0.7290 +- 0.0023 | 2027/1778 | pass | pass | FAIL | recipe | none | - | - |
| s07_I5 | REJECT | 1.000 | 0.000 | 0.7341 | 0.7490 +- 0.0052 | 0.7290 +- 0.0023 | 2052/1778 | pass | FAIL | FAIL | recipe | none | PalmerAmaranth | - |

## Attribution of Bswap

- freeze: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- full: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- lora: REJECT, class_vs_loc=none, attributed to labels: no

## Attribution not run

- Step 4: label audit (BioCLIP-2 re-reading D_k's boxes): out of scope for the pilot
- Step 5: leave-one-source-out cand runs for a multi-source increment (Breal mixes two sources): out of scope for the pilot
- freeze: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal
- full: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal
- lora: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal

## Warmup as run

Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), 100) iterations; replay mode full: cand holds the accepted pool + D_k and null the pool, recorded at pool_min (P0) and pool_max (P0 + every earlier increment); the driver records each step's actual sizes. Incremental runs: warmup lasts 1.00-2.04 of their 30 epochs (the recipe asks for 1); the cold base run's lasts 3.00 of 100.

From the decided steps' train sizes: cand warmup 1.54-1.79, null warmup 1.79-2.04 epochs.

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| chain freeze: final incumbent | 0.7305 | 0.8173 | 0.7125 | 0.7993 | 0.0903 | 0.1241 | 0.0303 | 0.1181 | 0.7861 | 0.8461 |
| chain full: final incumbent | 0.7341 | 0.8247 | 0.7357 | 0.8123 | 0.1072 | 0.1051 | 0.0345 | 0.0917 | 0.7923 | 0.8456 |
| chain lora: final incumbent | 0.7341 | 0.8236 | 0.7089 | 0.8036 | 0.0809 | 0.1257 | 0.0279 | 0.1062 | 0.7856 | 0.8515 |
| base P0 | 0.7212 +- 0.0085 | 0.8195 +- 0.0028 | 0.7041 +- 0.0118 | 0.8051 +- 0.0032 | 0.0870 +- 0.0078 | 0.1269 +- 0.0098 | 0.0423 +- 0.0132 | 0.1101 +- 0.0064 | 0.7761 +- 0.0063 | 0.8428 +- 0.0022 |
| T_final (union of clean data) | 0.8084 +- 0.0043 | 0.8548 +- 0.0037 | 0.7561 +- 0.0047 | 0.8372 +- 0.0031 | 0.1637 +- 0.0459 | 0.1252 +- 0.0037 | 0.0550 +- 0.0019 | 0.1223 +- 0.0090 | 0.8472 +- 0.0014 | 0.8717 +- 0.0009 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 0.77 |
| chain freeze | 43 | 4.42 |
| chain full | 43 | 4.73 |
| chain lora | 43 | 5.02 |
| final | 9 | 1.08 |
| truth | 21 | 8.17 |
| total | 162 | 24.20 |
