# Per-species validation — YOLO11n baseline

*RESEARCH_LOG Phase 1 per-class validation (YOLO11n baseline)*

| Species | Label in source | mAP@0.5 | mAP50-95 |
|---|---|---|---|
| Waterhemp | Carpetweeds | 0.976 | 0.936 |
| Morning glory | Crabgrass | 0.979 | 0.935 |
| Purslane | Eclipta | 0.92 | 0.878 |
| Spotted spurge | Goosegrass | 0.903 | 0.83 |
| Carpetweed | Morningglory | 0.853 | 0.769 |
| Ragweed | Nutsedge | 0.939 | 0.878 |
| Eclipta | PalmerAmaranth | 0.955 | 0.917 |
| Prickly sida | PricklySida | 0.97 | 0.947 |
| Palmer amaranth | Purslane | 0.941 | 0.93 |
| Sicklepod | Ragweed | 0.993 | 0.983 |
| Goosegrass | Sicklepod | 0.945 | 0.858 |
| Cutleaf groundcherry | SpottedSpurge | 0.95 | 0.919 |

The source recorded the old cwd12 labels; each is translated to the species of its class id (docs/CWD12_SPECIES.md).
