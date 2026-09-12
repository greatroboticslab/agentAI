# CottonWeedDet12 benchmark (mAP@0.5)

*RESEARCH_LOG Phase 2 table (15 models, CottonWeedDet12)*

| # | Model | Params | mAP@0.5 | mAP50-95 | Note |
|---|---|---|---|---|---|
| 1 | YOLO11n (fine-tuned) | 2.6M | 0.929 | 0.865 |  |
| 2 | Florence-2-base | 0.23B | 0.434 | 0.392 |  |
| 3 | Florence-2-large | 0.77B | 0.329 | 0.302 |  |
| 4 | InternVL2-8B | 8B | 0.208 | 0.091 |  |
| 5 | Qwen2.5-VL-3B | 3B | 0.196 | 0.068 |  |
| 6 | MiniCPM-V-4.5 | 8B | 0.192 | 0.043 |  |
| 7 | OWLv2-large | 0.4B | 0.184 | 0.117 | recall 0.943 / precision 0.194 — the high-recall pre-filter finding |
| 8 | Qwen2.5-VL-7B | 7B | 0.176 | 0.059 |  |
| 9 | InternVL2-2B | 2B | 0.002 | 0.001 |  |
| 10 | InternVL2.5-8B | 8B | 0.0 | 0.0 |  |
| 11-15 | G-DINO / Molmo / Llama-Vision / Moondream / LLaVA | - | 0.0 | None | no usable grounding — mAP ≈ 0 |
