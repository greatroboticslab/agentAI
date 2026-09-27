# INC report: pilot_v1

- Type: chain; seeds [0, 1, 2]; generated 2026-09-27T05:32:11Z
- Done: yes (2026-09-27T05:26:15Z)
- Replay mode: sample (cand on D_k + R1, null on R1 + R2; R1, R2 disjoint samples of |D_k| images from the chain's accepted pool)
- Decision code pinned 2026-09-27T04:05:20Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py 1d65393e0b29, gate.py e6eadeb76d0d, splits.py 79bd12f270a6

## Decisions per step

Chain verdicts against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts).

| Step | Clean | Truth (P) | freeze (P_data) | full (P_data) | lora (P_data) | agrees: freeze | agrees: full | agrees: lora |
|---|---|---|---|---|---|---|---|---|
| s01_I1 | yes | hurts (0.33) | REJECT (1.00) | REJECT (1.00) | REJECT (0.33) | yes | yes | yes |
| s02_I2 | yes | helps (1.00) | REJECT (0.44) | REJECT (1.00) | REJECT (1.00) | no | no | no |
| s03_Bswap | no | hurts (0.11) | REJECT (0.00) | REJECT (0.00) | REJECT (0.00) | yes | yes | yes |
| s04_I3 | yes | helps (1.00) | REJECT (0.22) | REJECT (1.00) | REJECT (0.11) | no | no | no |
| s05_Breal | no | hurts (0.00) | REJECT (0.00) | REJECT (0.00) | REJECT (0.00) | yes | yes | yes |
| s06_I4 | yes | neutral (0.67) | REJECT (0.00) | REJECT (1.00) | REJECT (0.00) | no | no | no |
| s07_I5 | yes | helps (1.00) | REJECT (0.00) | REJECT (1.00) | REJECT (1.00) | no | no | no |

- freeze agrees with the truth arm on 3 of 7 compared steps
- full agrees with the truth arm on 3 of 7 compared steps
- lora agrees with the truth arm on 3 of 7 compared steps

## Gate details

### freeze (done)

Accepted none; neutral none; quarantined ['I1', 'I2', 'Bswap', 'I3', 'Breal', 'I4', 'I5']; final incumbent base__s0.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7193 +- 0.0019 | 0.7134 +- 0.0007 | 476/476 | FAIL | pass | FAIL | recipe | none | - | - |
| s02_I2 | REJECT | 0.444 | 0.000 | 0.7282 | 0.7174 +- 0.0025 | 0.7183 +- 0.0020 | 476/476 | FAIL | FAIL | FAIL | recipe | none | Carpetweed, PalmerAmaranth, CutleafGroundcherry | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7282 | 0.6914 +- 0.0007 | 0.7169 +- 0.0023 | 544/544 | FAIL | FAIL | FAIL | recipe | domain/localisation | Ragweed, PalmerAmaranth, Goosegrass, CutleafGroundcherry | - |
| s04_I3 | REJECT | 0.222 | 0.000 | 0.7282 | 0.7081 +- 0.0009 | 0.7151 +- 0.0062 | 476/476 | FAIL | FAIL | FAIL | recipe | none | CutleafGroundcherry | - |
| s05_Breal | REJECT | 0.000 | 0.000 | 0.7282 | 0.6893 +- 0.0012 | 0.7103 +- 0.0015 | 476/476 | FAIL | FAIL | FAIL | recipe | domain/localisation | SpottedSpurge, Ragweed, PalmerAmaranth, Sicklepod | - |
| s06_I4 | REJECT | 0.000 | 0.000 | 0.7282 | 0.7005 +- 0.0018 | 0.7137 +- 0.0041 | 498/498 | FAIL | FAIL | FAIL | recipe | labels | PricklySida, Sicklepod, CutleafGroundcherry | - |
| s07_I5 | REJECT | 0.000 | 0.000 | 0.7282 | 0.7131 +- 0.0023 | 0.7228 +- 0.0033 | 548/548 | FAIL | FAIL | FAIL | recipe | labels | Purslane, Eclipta, Goosegrass | - |

### full (done)

Accepted none; neutral none; quarantined ['I1', 'I2', 'Bswap', 'I3', 'Breal', 'I4', 'I5']; final incumbent base__s0.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7159 +- 0.0022 | 0.7028 +- 0.0009 | 476/476 | FAIL | FAIL | FAIL | recipe | none | Purslane, Ragweed | - |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7278 +- 0.0074 | 0.7033 +- 0.0026 | 476/476 | pass | FAIL | FAIL | recipe | none | PalmerAmaranth | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7282 | 0.6729 +- 0.0049 | 0.7062 +- 0.0053 | 544/544 | FAIL | FAIL | FAIL | recipe | domain/localisation | MorningGlory, Purslane, SpottedSpurge, Ragweed, Eclipta, PalmerAmaranth, CutleafGroundcherry | - |
| s04_I3 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7257 +- 0.0041 | 0.7014 +- 0.0022 | 476/476 | pass | FAIL | FAIL | recipe | none | PalmerAmaranth | - |
| s05_Breal | REJECT | 0.000 | 0.000 | 0.7282 | 0.6662 +- 0.0017 | 0.7071 +- 0.0032 | 476/476 | FAIL | FAIL | FAIL | recipe | domain/localisation | Purslane, SpottedSpurge, Ragweed, Sicklepod, CutleafGroundcherry | - |
| s06_I4 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7058 +- 0.0016 | 0.6962 +- 0.0029 | 498/498 | FAIL | pass | FAIL | recipe | none | - | - |
| s07_I5 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7176 +- 0.0057 | 0.7083 +- 0.0020 | 548/548 | FAIL | FAIL | pass | recipe | none | Purslane, Sicklepod | - |

### lora (done)

Accepted none; neutral none; quarantined ['I1', 'I2', 'Bswap', 'I3', 'Breal', 'I4', 'I5']; final incumbent base__s0.

| Step | Verdict | P_data | P_recipe | inc | cand | null | images cand/null | regression | species | flips | blame | class vs loc | species failed | soup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01_I1 | REJECT | 0.333 | 0.000 | 0.7282 | 0.7125 +- 0.0051 | 0.7157 +- 0.0015 | 476/476 | FAIL | FAIL | FAIL | recipe | labels | Eclipta | - |
| s02_I2 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7255 +- 0.0034 | 0.7099 +- 0.0042 | 476/476 | pass | FAIL | FAIL | recipe | none | PalmerAmaranth, CutleafGroundcherry | - |
| s03_Bswap | REJECT | 0.000 | 0.000 | 0.7282 | 0.6898 +- 0.0027 | 0.7209 +- 0.0006 | 544/544 | FAIL | FAIL | FAIL | recipe | domain/localisation | MorningGlory, Ragweed, PalmerAmaranth, Goosegrass, CutleafGroundcherry | - |
| s04_I3 | REJECT | 0.111 | 0.000 | 0.7282 | 0.7228 +- 0.0036 | 0.7257 +- 0.0031 | 476/476 | pass | FAIL | FAIL | recipe | none | SpottedSpurge, Sicklepod | - |
| s05_Breal | REJECT | 0.000 | 0.000 | 0.7282 | 0.6546 +- 0.0047 | 0.7133 +- 0.0005 | 476/476 | FAIL | FAIL | FAIL | recipe | domain/localisation | Purslane, SpottedSpurge, Ragweed, PricklySida, PalmerAmaranth, Sicklepod, CutleafGroundcherry | - |
| s06_I4 | REJECT | 0.000 | 0.333 | 0.7282 | 0.7134 +- 0.0027 | 0.7277 +- 0.0015 | 498/498 | FAIL | FAIL | FAIL | data | labels | PricklySida, CutleafGroundcherry | - |
| s07_I5 | REJECT | 1.000 | 0.000 | 0.7282 | 0.7231 +- 0.0018 | 0.7152 +- 0.0031 | 548/548 | pass | FAIL | FAIL | recipe | none | Purslane, SpottedSpurge, PalmerAmaranth | - |

## Attribution of Bswap

- freeze: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- full: REJECT, class_vs_loc=domain/localisation, attributed to labels: no
- lora: REJECT, class_vs_loc=domain/localisation, attributed to labels: no

## Attribution not run

- Step 4: label audit (BioCLIP-2 re-reading D_k's boxes): out of scope for the pilot
- Step 5: leave-one-source-out cand runs for a multi-source increment (Breal mixes two sources): out of scope for the pilot
- freeze: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal
- full: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal
- lora: REJECT of a multi-source increment without leave-one-source-out attribution: s05_Breal

## Warmup as run

Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), 100) iterations; cand and null sets hold 2|D_k| images. Incremental runs: warmup lasts 5.56-6.67 of their 30 epochs (the recipe asks for 1); the cold base run's lasts 3.00 of 100.

From the decided steps' train sizes: cand warmup 5.56-6.67, null warmup 5.56-6.67 epochs.

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| chain freeze: final incumbent | 0.7282 | 0.8221 | 0.7041 | 0.8031 | 0.0839 | 0.1181 | 0.0370 | 0.1037 | 0.7735 | 0.8403 |
| chain full: final incumbent | 0.7282 | 0.8221 | 0.7041 | 0.8031 | 0.0839 | 0.1181 | 0.0370 | 0.1037 | 0.7735 | 0.8403 |
| chain lora: final incumbent | 0.7282 | 0.8221 | 0.7041 | 0.8031 | 0.0839 | 0.1181 | 0.0370 | 0.1037 | 0.7735 | 0.8403 |
| base P0 | 0.7212 +- 0.0085 | 0.8195 +- 0.0028 | 0.7041 +- 0.0118 | 0.8051 +- 0.0032 | 0.0870 +- 0.0078 | 0.1269 +- 0.0098 | 0.0423 +- 0.0132 | 0.1101 +- 0.0064 | 0.7761 +- 0.0063 | 0.8428 +- 0.0022 |
| T_final (union of clean data) | 0.8084 +- 0.0043 | 0.8548 +- 0.0037 | 0.7561 +- 0.0047 | 0.8372 +- 0.0031 | 0.1637 +- 0.0459 | 0.1252 +- 0.0037 | 0.0550 +- 0.0019 | 0.1223 +- 0.0090 | 0.8472 +- 0.0014 | 0.8717 +- 0.0009 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 1.08 |
| chain freeze | 42 | 1.90 |
| chain full | 42 | 2.16 |
| chain lora | 42 | 2.26 |
| final | 9 | 0.95 |
| truth | 21 | 8.92 |
| total | 159 | 17.28 |
