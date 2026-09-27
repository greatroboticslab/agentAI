# INC report: b0_v1

- Type: baseline; seeds [0, 1, 2]; generated 2026-09-27T08:21:34Z
- Done: yes (2026-09-27T04:20:33Z)
- Decision code pinned 2026-09-27T03:28:31Z (init): cwd12_species.py a7b7efbb1ce8, __init__.py cd6ec55a96ee, common.py 6ba51dbea379, driver.py 1d65393e0b29, gate.py e6eadeb76d0d, splits.py 79bd12f270a6

## Final quality

From the final runs only (the only runs that read test). 12-class = the scorer's species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic mAP50-95; mean +- sd over seeds where there are seeds.

| Model | dev 12-class | dev agn | ood22 12-class | ood22 agn | ood23 12-class | ood23 agn | imageweeds 12-class | imageweeds agn | test 12-class | test agn |
|---|---|---|---|---|---|---|---|---|---|---|
| base train_core | 0.8082 +- 0.0063 | 0.8510 +- 0.0007 | 0.7505 +- 0.0131 | 0.8388 +- 0.0052 | 0.1861 +- 0.0138 | 0.1236 +- 0.0113 | 0.0610 +- 0.0159 | 0.1149 +- 0.0176 | 0.8541 +- 0.0074 | 0.8741 +- 0.0019 |

## GPU-hours

| Unit | Runs | GPU-hours |
|---|---|---|
| base | 3 | 1.77 |
| final | 3 | 0.43 |
| total | 6 | 2.20 |
