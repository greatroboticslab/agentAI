# V3F: the final decision of the follow-up experiments (26 October 2026) and its two sealed-test reads

**Source.** The decision is `results/framework/inc/capacity/v3fam_v1_cont2_report.md` (with `v3fam_v1_cont2.json`,
`v3fam_v1_cont2_external.json`, `v3fam_ext_v1_cont2.json` and `v3fam_v1_cont2_report.json`), written on Bridges-2 by
job 49852022 at 2026-10-26T13:21:35Z under deviation 5 (`docs/prereg/V3F_dev5_2026-10-24.md`). The reads are
`capacity/v3fam_test_d6.md` and `capacity/v3fam_test_d8m.md` (jobs 49881549 and 49885336), taken in the registered
order after the decision. Bridges-2 has been offline since 2026-11-09 for PSC's data-centre move (expected back about
2026-12-14), so this file holds the text of those files as it was read on the cluster on 2026-10-26 (13:25 and
19:03 UTC). The JSON records and the score files are to be copied here once the cluster is back.

## The nine registered primary comparisons

| | Set | Comparison | D | 90 % interval | z | Rule | Confirmed |
|---|---|---|---|---|---|---|---|
| P1 | development | BioCLIP 2 vs its LAION-2B CLIP ViT-L/14 starting model, naming two-stage B's boxes | +0.0757 | [+0.0597, +0.0918] | 7.77 | qualifies | yes |
| P2 | development | two-stage B vs box detector on base B's 15,432 field images | +0.0102 | [-0.0002, +0.0218] | 1.62 | qualifies | no |
| P3 | development | two-stage B vs box detector on base B without the greenhouse frames | -0.0011 | [-0.0083, +0.0061] | -0.25 | inconclusive | no |
| P4 | development | weighted box fusion of two-stage B's three box-detector seeds vs two-stage B | -0.0042 | [-0.0087, +0.0004] | -1.51 | inconclusive | no |
| P5 | development | two-stage B vs RF-DETR-L (12 species, base A) | +0.0365 | [+0.0180, +0.0564] | 3.19 | qualifies | yes |
| P6 | 2022 season | two-stage B′ vs 12-species YOLO11m on base A′ | +0.0345 | [+0.0087, +0.0602] | 2.20 | qualifies | yes |
| P7 | 2023 season | one-class detector on base B′ vs on base A′ (target-species class-agnostic AP) | +0.0423 | [+0.0157, +0.0690] | 2.61 | qualifies | yes |
| P8 | copy test | near copies of all 476 copy-test images vs none | +0.1381 | [+0.0695, +0.2067] | 3.31 | qualifies | yes |
| P9 | copy test | near copies of 119 of them vs none | +0.0530 | [-0.0013, +0.1073] | 1.60 | qualifies | no |

P7's statistic per seed (not printed in the report): base B′ 0.149603, 0.159426, 0.163401; base A′ 0.109522,
0.120397, 0.115479 (`docs/prereg/DS1_dev1_2026-10-30.md`). P1 and P4 reproduce the first decision step of
2026-10-13 exactly (D 0.0757, z 7.7723; D -0.0042, z -1.5056).

**Sealed-test reads (CottonWeedDet12, 1,977 images, 12-species mAP50-95, three seeds):** RF-DETR-L on base A
0.8855 ± 0.0056; YOLO11m on the CottonWeedDet12 training part alone 0.8850 ± 0.0022; reference YOLO11m on base A
0.8786; two-stage B stays best at 0.8950 ± 0.0029. Two of the five reads the registration allows have been used.

## The report as read (verbatim)

```
END:  COMPLETED   02:01:27 
-rw-r--r--  1 byler cis240145p  13385 Oct 26 09:21 v3fam_ext_v1_cont2.json
-rw-r--r--  1 byler cis240145p  33643 Oct 26 09:21 v3fam_v1_cont2_external.json
-rw-r--r--  1 byler cis240145p  80270 Oct 26 09:21 v3fam_v1_cont2.json
-rw-r--r--  1 byler cis240145p 143271 Oct 26 09:21 v3fam_v1_cont2_report.json
-rw-r--r--  1 byler cis240145p  14896 Oct 26 09:21 v3fam_v1_cont2_report.md
# V3F family report (2026-10-26T13:21:35.384816Z)

Decided on: dev. Dev passes of the family: 263 (builds_protocol 66, d2_gt 11, d9a 15, family_scores 166, reproductions 5).

| id | exam | X | Y | D | D_sel | 2 sd | SE image | SE session | SE_sel | z | 90 % interval | outcome | confirmed / opens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | dev | E3B | D2LAION | 0.0757 | - | 0.0061 | 0.0094 | 0.0212 | - | 7.7723 | [0.0597, 0.0918] | qualifies | family-confirmed |
| P2 | dev | E3B | D4F | 0.0102 | 0.0108 | 0.0056 | 0.0063 | 0.0082 | 0.0063 | 1.6160 | [-0.0002, 0.0218] | qualifies | not confirmed |
| P3 | dev | E3B | D4N | -0.0011 | -0.0011 | 0.0039 | 0.0041 | 0.0047 | 0.0041 | -0.2455 | [-0.0083, 0.0061] | inconclusive | not confirmed |
| P4 | dev | D3 | E3B | -0.0042 | - | 0.0026 | 0.0025 | 0.0020 | - | -1.5056 | [-0.0087, 0.0004] | inconclusive | not confirmed |
| P5 | dev | E3B | D6 | 0.0365 | 0.0372 | 0.0082 | 0.0112 | 0.0200 | 0.0112 | 3.1892 | [0.0180, 0.0564] | qualifies | family-confirmed |
| P6 | ood22 | TSB | D1A12 | 0.0345 | - | 0.0072 | 0.0096 | 0.0154 | - | 2.2012 | [0.0087, 0.0602] | qualifies | family-confirmed |
| P7 | ood23 | D1B1 | D1A1 | 0.0423 | - | 0.0127 | 0.0032 | 0.0154 | - | 2.6111 | [0.0157, 0.0690] | qualifies | family-confirmed |
| P8 | probe | D5B100 | D5B0 | 0.1381 | - | 0.0085 | 0.0133 | 0.0416 | - | 3.3114 | [0.0695, 0.2067] | qualifies | family-confirmed |
| P9 | probe | D5B25 | D5B0 | 0.0530 | - | 0.0084 | 0.0089 | 0.0328 | - | 1.6042 | [-0.0013, 0.1073] | qualifies | not confirmed |
| C_D2LAION | dev | D2LAION | E3B | -0.0757 | - | 0.0061 | 0.0096 | 0.0214 | - | -7.6087 | [-0.0921, -0.0594] | inconclusive (reversed) | no read |
| C_D2CLIP | dev | D2CLIP | E3B | -0.0800 | - | 0.0069 | 0.0099 | 0.0214 | - | -7.8083 | [-0.0968, -0.0631] | inconclusive (reversed) | no read |
| C_D2DINOL | dev | D2DINOL | E3B | -0.0147 | - | 0.0024 | 0.0055 | 0.0079 | - | -2.6260 | [-0.0239, -0.0055] | inconclusive (reversed) | no read |
| C_D2DINOB | dev | D2DINOB | E3B | -0.0215 | - | 0.0031 | 0.0051 | 0.0071 | - | -4.0556 | [-0.0302, -0.0128] | inconclusive (reversed) | no read |
| C_D2SIGLIP | dev | D2SIGLIP | E3B | -0.0312 | - | 0.0031 | 0.0056 | 0.0085 | - | -5.3928 | [-0.0407, -0.0217] | inconclusive (reversed) | no read |
| C_D2RN50 | dev | D2RN50 | E3B | -0.2213 | - | 0.0027 | 0.0165 | 0.0241 | - | -13.3909 | [-0.2485, -0.1941] | inconclusive (reversed) | no read |
| C_D2BC1 | dev | D2BC1 | E3B | -0.0238 | - | 0.0023 | 0.0057 | 0.0095 | - | -4.1489 | [-0.0332, -0.0143] | inconclusive (reversed) | no read |
| C_D2BC2640 | dev | D2BC2640 | E3B | -0.0028 | - | 0.0020 | 0.0020 | 0.0029 | - | -1.2973 | [-0.0062, 0.0007] | inconclusive | no read |
| C_D2FT | dev | D2FT | E3B | -0.0372 | - | 0.0077 | 0.0069 | 0.0118 | - | -4.9165 | [-0.0496, -0.0247] | inconclusive (reversed) | no read |
| C_D4F | dev | D4F | E3B | -0.0102 | - | 0.0056 | 0.0067 | 0.0083 | - | -1.4540 | [-0.0218, 0.0013] | inconclusive | no read |
| C_D4N | dev | D4N | E3B | 0.0011 | - | 0.0039 | 0.0043 | 0.0048 | - | 0.2368 | [-0.0064, 0.0086] | inconclusive | no read |
| C_D4R15 | dev | D4R15 | E3B | -0.0352 | - | 0.0047 | 0.0068 | 0.0127 | - | -4.9598 | [-0.0469, -0.0235] | inconclusive (reversed) | no read |
| C_D6 | dev | D6 | E3B | -0.0365 | - | 0.0082 | 0.0102 | 0.0201 | - | -3.3930 | [-0.0542, -0.0188] | inconclusive (reversed) | no read |
| S_bc2_over_bc2640 | dev | E3B | D2BC2640 | 0.0028 | - | 0.0020 | 0.0020 | 0.0028 | - | 1.2703 | [-0.0008, 0.0063] | qualifies |  |
| S_ft_over_rn50 | dev | D2FT | D2RN50 | 0.1841 | - | 0.0077 | 0.0145 | 0.0221 | - | 12.3698 | [0.1596, 0.2086] | qualifies |  |
| S_bc2_over_clip | dev | E3B | D2CLIP | 0.0800 | - | 0.0069 | 0.0098 | 0.0214 | - | 7.8132 | [0.0631, 0.0968] | qualifies |  |
| S_laion_over_clip | dev | D2LAION | D2CLIP | 0.0043 | - | 0.0088 | 0.0074 | 0.0057 | - | 0.5166 | [-0.0093, 0.0178] | inconclusive |  |
| S_dinol_over_clip | dev | D2DINOL | D2CLIP | 0.0653 | - | 0.0068 | 0.0087 | 0.0155 | - | 7.1519 | [0.0503, 0.0803] | qualifies |  |
| S_bc1_over_clip | dev | D2BC1 | D2CLIP | 0.0562 | - | 0.0067 | 0.0079 | 0.0150 | - | 6.7059 | [0.0424, 0.0700] | qualifies |  |
| S_tsf_over_tsr15 | dev | D4F | D4R15 | 0.0250 | - | 0.0068 | 0.0066 | 0.0093 | - | 3.4792 | [0.0132, 0.0368] | qualifies |  |
| S_e3b_over_tsr15 | dev | E3B | D4R15 | 0.0352 | - | 0.0047 | 0.0069 | 0.0124 | - | 4.9129 | [0.0234, 0.0470] | qualifies |  |
| S_tsf_over_e3a | dev | D4F | E3A | 0.0117 | - | 0.0059 | 0.0056 | 0.0036 | - | 1.9353 | [0.0018, 0.0217] | qualifies |  |
| S_e3a_over_d6 | dev | E3A | D6 | 0.0145 | - | 0.0084 | 0.0105 | 0.0160 | - | 1.3121 | [-0.0037, 0.0328] | qualifies |  |
| S_d6_over_ref | dev | D6 | REF | -0.0030 | - | 0.0087 | 0.0095 | 0.0172 | - | -0.2963 | [-0.0196, 0.0136] | inconclusive |  |
| S_ref_over_d8m | dev | REF | D8M | 0.0040 | - | 0.0051 | 0.0062 | 0.0113 | - | 0.6143 | [-0.0068, 0.0148] | inconclusive |  |
| S_d8m_over_d8s | dev | D8M | D8S | 0.0182 | - | 0.0068 | 0.0060 | 0.0117 | - | 2.7451 | [0.0073, 0.0290] | qualifies |  |
| S_bclean_over_ref | dev | BCLEAN | REF | not run: BCLEAN: not needed: no family score at /ocean/projects/cis240145p/byler/harry/we |  |  |  |  |  |  |  |  |  |
| S_bclean_over_e3b | dev | BCLEAN | E3B | not run: BCLEAN: not needed: no family score at /ocean/projects/cis240145p/byler/harry/we |  |  |  |  |  |  |  |  |  |
| S_tsa_over_a12_ood22 | ood22 | TSA | D1A12 | 0.0462 | - | 0.0063 | 0.0091 | 0.0147 | - | 3.0862 | [0.0216, 0.0708] | qualifies |  |
| S_tsb_over_tsa_ood22 | ood22 | TSB | TSA | -0.0117 | - | 0.0048 | 0.0052 | 0.0073 | - | -1.5485 | [-0.0242, 0.0007] | inconclusive |  |
| S_d3_stage1_over_e3b | dev | D3 | E3B | -0.0043 | - | 0.0058 | 0.0015 | 0.0015 | - | -1.5339 | [-0.0090, 0.0003] | inconclusive |  |
| S_s1_tsf_over_e1a | dev | D4F | E3A | 0.0113 | - | 0.0050 | 0.0053 | 0.0069 | - | 1.9902 | [0.0020, 0.0207] | qualifies |  |
| S_s1_tsf_over_e1b | dev | D4F | E3B | -0.0108 | - | 0.0062 | 0.0049 | 0.0045 | - | -1.9439 | [-0.0199, -0.0017] | inconclusive (reversed) |  |
| S_s1_tsn_over_e1a | dev | D4N | E3A | 0.0246 | - | 0.0024 | 0.0063 | 0.0052 | - | 3.8306 | [0.0140, 0.0351] | qualifies |  |
| S_s1_tsn_over_e1b | dev | D4N | E3B | 0.0025 | - | 0.0043 | 0.0029 | 0.0017 | - | 0.7234 | [-0.0032, 0.0081] | inconclusive |  |
| S_s1_tsr15_over_e1a | dev | D4R15 | E3A | -0.0178 | - | 0.0053 | 0.0078 | 0.0106 | - | -2.1991 | [-0.0311, -0.0045] | inconclusive (reversed) |  |
| S_s1_tsr15_over_e1b | dev | D4R15 | E3B | -0.0399 | - | 0.0064 | 0.0059 | 0.0102 | - | -6.1774 | [-0.0505, -0.0293] | inconclusive (reversed) |  |
| S_tsb_over_a12_ood22_8sp | ood22 | TSB | D1A12 | 0.0100 | - | 0.0082 | 0.0050 | 0.0086 | - | 1.0820 | [-0.0052, 0.0253] | qualifies |  |
| S_tsa_over_a12_ood22_8sp | ood22 | TSA | D1A12 | 0.0160 | - | 0.0083 | 0.0045 | 0.0078 | - | 1.8773 | [0.0020, 0.0299] | qualifies |  |
| S_tsb_over_tsa_ood22_8sp | ood22 | TSB | TSA | -0.0059 | - | 0.0064 | 0.0037 | 0.0045 | - | -1.1407 | [-0.0144, 0.0026] | inconclusive |  |
| S_tsb_over_a12_ood23_3sp | ood23 | TSB | D1A12 | 0.1235 | - | 0.0151 | 0.0092 | 0.0233 | - | 5.1335 | [0.0839, 0.1630] | qualifies |  |
| S_tsa_over_a12_ood23_3sp | ood23 | TSA | D1A12 | 0.0726 | - | 0.0114 | 0.0079 | 0.0260 | - | 2.7513 | [0.0292, 0.1160] | qualifies |  |
| S_tsb_over_tsa_ood23_3sp | ood23 | TSB | TSA | 0.0508 | - | 0.0164 | 0.0058 | 0.0119 | - | 3.7226 | [0.0284, 0.0733] | qualifies |  |
| S_p8_12class | probe | D5B100 | D5B0 | 0.1078 | - | 0.0132 | 0.0107 | 0.0341 | - | 3.1254 | [0.0511, 0.1645] | qualifies |  |
| S_p9_12class | probe | D5B25 | D5B0 | 0.0390 | - | 0.0140 | 0.0075 | 0.0223 | - | 1.6908 | [0.0011, 0.0769] | qualifies |  |
| S_d5b_copy_jpeg75 | probe | D5B100 | D5B0 | 0.1268 | - | 0.0113 | 0.0216 | 0.0567 | - | 2.2287 | [0.0332, 0.2204] | qualifies |  |
| S_d5b_copy_crop90 | probe | D5B100 | D5B0 | 0.1407 | - | 0.0165 | 0.0239 | 0.0404 | - | 3.4315 | [0.0732, 0.2081] | qualifies |  |
| S_d5b_copy_letterbox640 | probe | D5B100 | D5B0 | 0.1188 | - | 0.0088 | 0.0220 | 0.0364 | - | 3.2507 | [0.0587, 0.1790] | qualifies |  |
| S_d5b_copy_hflip | probe | D5B100 | D5B0 | 0.1406 | - | 0.0262 | 0.0272 | 0.0501 | - | 2.7477 | [0.0565, 0.2248] | qualifies |  |
| C_D3 | dev | D3 | E3B | -0.0042 | - | 0.0026 | 0.0025 | 0.0020 | - | -1.5056 | [-0.0087, 0.0004] | inconclusive | no read (= P4) |

Selection-corrected (V3F.3.3):
- P2: E3-B chosen in 99.5 % of the image resamples; plain D 0.0102, plain image SE 0.0063; D_sel 0.0108, SE_sel 0.0063
- P3: E3-B chosen in 99.5 % of the image resamples; plain D -0.0011, plain image SE 0.0041; D_sel -0.0011, SE_sel 0.0041
- P5: E3-B chosen in 98.8 % of the image resamples; plain D 0.0365, plain image SE 0.0112; D_sel 0.0372, SE_sel 0.0112

Reads in order: D6 (reference), D8M (reference).

| side | exam | species (seeds) | mean | agnostic | stage-1 agnostic |
|---|---|---|---|---|---|
| A'1 | dev | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.8479, 0.8449, 0.8422 | 0.8479, 0.8449, 0.8422 |
| A'1 | ood22 | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.8258, 0.8303, 0.8234 | 0.8258, 0.8303, 0.8234 |
| A'1 | ood23 | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.1461, 0.1509, 0.1436 | 0.1461, 0.1509, 0.1436 |
| A'1 | ooddev_v1 | -, -, - | - | 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0000 |
| A'12 | dev | 0.8500, 0.8415, 0.8520 | 0.8478 | 0.8686, 0.8611, 0.8685 | 0.8686, 0.8611, 0.8685 |
| A'12 | ood22 | 0.8112, 0.8046, 0.8122 | 0.8094 | 0.8647, 0.8581, 0.8612 | 0.8647, 0.8581, 0.8612 |
| A'12 | ood23 | 0.1877, 0.2175, 0.3374 | 0.2475 | 0.1495, 0.1525, 0.1597 | 0.1495, 0.1525, 0.1597 |
| A'12 | ooddev_v1 | -, -, - | - | 0.0001, 0.0000, 0.0003 | 0.0001, 0.0000, 0.0003 |
| B'1 | dev | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.8720, 0.8785, 0.8760 | 0.8720, 0.8785, 0.8760 |
| B'1 | ood22 | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.8226, 0.8213, 0.8259 | 0.8226, 0.8213, 0.8259 |
| B'1 | ood23 | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.2562, 0.2517, 0.2598 | 0.2562, 0.2517, 0.2598 |
| B'1 | ooddev_v1 | -, -, - | - | 0.0033, 0.0063, 0.0050 | 0.0033, 0.0063, 0.0050 |
| BioCLIP-1 | dev | 0.8618, 0.8614, 0.8632 | 0.8622 | 0.8733, 0.8789, 0.8782 | 0.8749, 0.8810, 0.8802 |
| BioCLIP-2@640 | dev | 0.8826, 0.8836, 0.8834 | 0.8832 | 0.8748, 0.8798, 0.8795 | 0.8749, 0.8810, 0.8802 |
| OpenAI CLIP ViT-L/14 | dev | 0.8062, 0.8011, 0.8105 | 0.8059 | 0.8664, 0.8707, 0.8696 | 0.8749, 0.8810, 0.8802 |
| DINOv2-B | dev | 0.8626, 0.8646, 0.8661 | 0.8644 | 0.8723, 0.8781, 0.8773 | 0.8749, 0.8810, 0.8802 |
| DINOv2-L | dev | 0.8703, 0.8709, 0.8725 | 0.8712 | 0.8732, 0.8787, 0.8777 | 0.8749, 0.8810, 0.8802 |
| FT-ResNet-50 | dev | 0.8525, 0.8427, 0.8511 | 0.8487 | 0.8703, 0.8783, 0.8776 | 0.8749, 0.8810, 0.8802 |
| LAION-2B CLIP ViT-L/14 | dev | 0.8108, 0.8058, 0.8140 | 0.8102 | 0.8650, 0.8697, 0.8699 | 0.8749, 0.8810, 0.8802 |
| ResNet-50 | dev | 0.6631, 0.6655, 0.6653 | 0.6646 | 0.8577, 0.8625, 0.8603 | 0.8749, 0.8810, 0.8802 |
| SigLIP2 | dev | 0.8554, 0.8527, 0.8561 | 0.8547 | 0.8694, 0.8748, 0.8743 | 0.8749, 0.8810, 0.8802 |
| WBF of E3-B's three stage-1 seeds | dev | 0.8818 | 0.8818 | 0.8750 | 0.8748 |
| WBF of E3-B's three stage-1 seeds | imageweeds | 0.2461 | 0.2461 | 0.2036 | 0.1972 |
| WBF of E3-B's three stage-1 seeds | ooddev_v1 | - | - | 0.0035 | 0.0036 |
| TS-F | dev | 0.8721, 0.8755, 0.8795 | 0.8757 | 0.8619, 0.8665, 0.8702 | 0.8644, 0.8683, 0.8715 |
| TS-F | imageweeds | 0.2374, 0.2307, 0.2378 | 0.2353 | 0.1667, 0.1813, 0.1782 | 0.1625, 0.1751, 0.1749 |
| TS-F | ooddev_v1 | -, -, - | - | 0.0017, 0.0028, 0.0030 | 0.0017, 0.0028, 0.0030 |
| TS-N | dev | 0.8895, 0.8868, 0.8847 | 0.8870 | 0.8812, 0.8813, 0.8797 | 0.8814, 0.8819, 0.8804 |
| TS-N | imageweeds | 0.2387, 0.2399, 0.2330 | 0.2372 | 0.1745, 0.1882, 0.1712 | 0.1618, 0.1783, 0.1616 |
| TS-N | ooddev_v1 | -, -, - | - | 0.0022, 0.0022, 0.0033 | 0.0022, 0.0022, 0.0033 |
| TS-R15 | dev | 0.8505, 0.8538, 0.8477 | 0.8507 | 0.8414, 0.8419, 0.8367 | 0.8414, 0.8403, 0.8351 |
| TS-R15 | imageweeds | 0.2089, 0.2234, 0.1810 | 0.2044 | 0.1677, 0.1742, 0.1515 | 0.1629, 0.1692, 0.1490 |
| TS-R15 | ooddev_v1 | -, -, - | - | 0.0035, 0.0018, 0.0030 | 0.0035, 0.0018, 0.0030 |
| dose 0 | probe | 0.8485, 0.8581, 0.8670 | 0.8579 | 0.8352, 0.8411, 0.8521 | 0.8352, 0.8411, 0.8521 |
| dose 1.0 | probe | 0.9638, 0.9666, 0.9665 | 0.9656 | 0.9616, 0.9641, 0.9642 | 0.9616, 0.9641, 0.9642 |
| dose 0.25 | probe | 0.8998, 0.8929, 0.8978 | 0.8968 | 0.8892, 0.8824, 0.8858 | 0.8892, 0.8824, 0.8858 |
| RF-DETR-L 12-class | dev | 0.8546, 0.8434, 0.8503 | 0.8494 | 0.8708, 0.8682, 0.8757 | 0.8708, 0.8682, 0.8757 |
| RF-DETR-L 12-class | imageweeds | 0.0917, 0.0932, 0.0981 | 0.0943 | 0.1596, 0.1569, 0.1424 | 0.1596, 0.1569, 0.1424 |
| RF-DETR-L 12-class | ooddev_v1 | -, -, - | - | 0.0077, 0.0034, 0.0130 | 0.0077, 0.0034, 0.0130 |
| D8-m (YOLO11m train_core) | dev | 0.8468, 0.8514, 0.8470 | 0.8484 | 0.8565, 0.8657, 0.8628 | 0.8565, 0.8657, 0.8628 |
| D8-m (YOLO11m train_core) | imageweeds | 0.0857, 0.0654, 0.0855 | 0.0789 | 0.1359, 0.1255, 0.1419 | 0.1359, 0.1255, 0.1419 |
| D8-m (YOLO11m train_core) | ooddev_v1 | -, -, - | - | 0.0001, 0.0003, 0.0002 | 0.0001, 0.0003, 0.0002 |
| D8-s (YOLO11s train_core) | dev | 0.8271, 0.8288, 0.8348 | 0.8302 | 0.8539, 0.8517, 0.8582 | 0.8539, 0.8517, 0.8582 |
| D8-s (YOLO11s train_core) | imageweeds | 0.0818, 0.0889, 0.0811 | 0.0839 | 0.1486, 0.1427, 0.1376 | 0.1486, 0.1427, 0.1376 |
| D8-s (YOLO11s train_core) | ooddev_v1 | -, -, - | - | 0.0003, 0.0005, 0.0002 | 0.0003, 0.0005, 0.0002 |
| E3-A | dev | 0.8633, 0.8626, 0.8660 | 0.8640 | 0.8551, 0.8527, 0.8573 | 0.8569, 0.8556, 0.8584 |
| E3-B | dev | 0.8850, 0.8853, 0.8874 | 0.8859 | 0.8747, 0.8798, 0.8797 | 0.8749, 0.8810, 0.8802 |
| E3-B | imageweeds | 0.2515, 0.2419, 0.2541 | 0.2492 | 0.2069, 0.2002, 0.1999 | 0.1997, 0.1920, 0.1920 |
| E3-M | dev | 0.8714, 0.8730, 0.8659 | 0.8701 | 0.8700, 0.8722, 0.8646 | 0.8714, 0.8746, 0.8676 |
| E3-B's boxes, perfect naming (O5) | dev | 0.8948, 0.8946, 0.8962 | 0.8952 | 0.8773, 0.8836, 0.8828 | 0.8749, 0.8810, 0.8802 |
| b_v2_m640 (YOLO11m reference) | dev | 0.8547, 0.8498, 0.8528 | 0.8524 | 0.8708, 0.8722, 0.8655 | 0.8708, 0.8722, 0.8655 |
| b_v2_m640 (YOLO11m reference) | ooddev_v1 | -, -, - | - | 0.0000, 0.0004, 0.0002 | 0.0000, 0.0004, 0.0002 |
| TS-A' | dev | 0.8586, 0.8550, 0.8538 | 0.8558 | 0.8501, 0.8451, 0.8422 | 0.8479, 0.8449, 0.8422 |
| TS-A' | ood22 | 0.8558, 0.8570, 0.8538 | 0.8555 | 0.8487, 0.8518, 0.8429 | 0.8258, 0.8303, 0.8234 |
| TS-A' | ood23 | 0.2577, 0.2919, 0.2778 | 0.2758 | 0.1223, 0.1299, 0.1244 | 0.1461, 0.1509, 0.1436 |
| TS-B' | dev | 0.8834, 0.8903, 0.8813 | 0.8850 | 0.8757, 0.8803, 0.8788 | 0.8720, 0.8785, 0.8760 |
| TS-B' | ood22 | 0.8407, 0.8441, 0.8466 | 0.8438 | 0.8386, 0.8401, 0.8457 | 0.8226, 0.8213, 0.8259 |
| TS-B' | ood23 | 0.2561, 0.2584, 0.2997 | 0.2714 | 0.2289, 0.2210, 0.2281 | 0.2562, 0.2517, 0.2598 |

==== LOG
module tools/inc2/twostage.py: 8fe2edc0e0d2a00d  = nested
module tools/inc/audit.py: 83870aa9897c5caa  = nested
module tools/inc2/v3fam.py: 3d776daed91d487d  = nested
module tools/inc2/v3fam_data.py: d7e62d895e4c3642  = nested
module tools/inc2/v3fam_sweep.py: 78d9ba50a40209ce  = nested
module tools/inc2/v3fam_embed.py: 6caf3e87d4d2bca7  = nested
module tools/inc2/v3fam_wbf.py: 099ae63f63a1b815  = nested
module tools/inc2/v3fam_detr.py: 2a6e40eacb012c04  = nested
module tools/inc2/v3fam_systems.py: 4d3842a55e5ef15c  = nested
module tools/inc2/v3fam_trained.py: c1d6dde44dc551d5  = nested
module tools/inc2/v3fam_verdict.py: c637ef1863492341  = nested
module tools/inc2/v3fam_supersede.py: 17308e6bb1b4386a  = nested
module tools/inc2/v3fam_loader.py: eead7415c190073e  = nested
module tools/inc2/v3fam_detr_env.txt: d05b539cbc7989de  = nested
module tools/inc_autopilot/stream_thresholds.json: 66bd3096b8f5b2df  = nested
numpy 2.2.6
torch 2.5.1+cu121 ultralytics 8.4.37 sklearn 1.7.2 open_clip 3.3.0 transformers 5.8.0 timm 1.0.25 cv2 4.10.0
[inc2.v3fam] second continuation verdict (deviation 5): P1 qualifies, P2 qualifies, P3 inconclusive, P4 inconclusive, P5 qualifies; reads ['D6', 'D8M'] (6872s)
=== build exit 0 after 6907s  Mon Oct 26 09:22:23 EDT 2026 ===
=== done Mon Oct 26 09:22:25 EDT 2026 ===

[exited with code 0]
```

## The two reads as read (verbatim)

```
capacity/v3fam_test_d6.json
capacity/v3fam_test_d8m.md
capacity/v3fam_test_d6.md
capacity/v3fam_test_d8m.json
== capacity/v3fam_test_d8m.md
# D8-m (YOLO11m train_core)'s sealed test (reference read, read by Claude (delegate under the 2026-09-30 owner grant))

12-class mAP50-95 0.8850 +- 0.0022 (gap to 0.90: 0.0150); the 329 test images from dev's sessions: 0.8837. Reference b_v2_m640: 0.8786.
== capacity/v3fam_test_d6.md
# RF-DETR-L 12-class's sealed test (reference read, read by Claude (delegate under the 2026-09-30 owner grant))

12-class mAP50-95 0.8855 +- 0.0056 (gap to 0.90: 0.0145); the 329 test images from dev's sessions: 0.8777. Reference b_v2_m640: 0.8786.
```
