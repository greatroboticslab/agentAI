# INC report: base_b_v1

- Type: baseline; seeds [0, 1, 2]; generated 2026-09-27T08:22:09Z
- Done: yes (2026-09-27T08:21:13Z)
- Decision code pinned 2026-09-27T07:29:36Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py ae9ffa8daedd, gate.py e6eadeb76d0d, splits.py 79bd12f270a6

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| base base_B | 0.8133 +- 0.0070 | 0.8532 +- 0.0010 | 0.7752 +- 0.0093 | 0.8458 +- 0.0030 | 0.1901 +- 0.0801 | 0.1355 +- 0.0062 | 0.0160 +- 0.0026 | 0.0425 +- 0.0036 | 0.8502 +- 0.0059 | 0.8730 +- 0.0024 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 1.75 |
| final | 3 | 0.38 |
| total | 6 | 2.12 |
