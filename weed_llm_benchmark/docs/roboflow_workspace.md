# Roboflow workspace — `research-lhi4x`

Active labeling target for the agentAI weed-detection pipeline.
See [`weed_optimizer_framework/tools/roboflow_sync.py`](../weed_optimizer_framework/tools/roboflow_sync.py)
for the upload/sync code and `memory/project_roboflow_pipeline_plan.md`
for the master plan.

## Project layout

13 object-detection projects in workspace `research-lhi4x`:

| Project | Purpose | Classes | Images (gold seed) |
|---|---|---|---|
| **cwd12-weeds** | Combined multi-class view; the "what we train" target. | 12 (Carpetweeds, Crabgrass, Eclipta, Goosegrass, Morningglory, Nutsedge, PalmerAmaranth, PricklySida, Purslane, Ragweed, Sicklepod, SpottedSpurge) | 598 |
| **cwd12-carpetweeds** | Per-species labeling slot for Carpetweeds. | 1 (Carpetweeds, cid=0) | 50 |
| **cwd12-crabgrass** | per-species Crabgrass | 1 | 50 |
| **cwd12-eclipta** | per-species Eclipta | 1 | 50 |
| **cwd12-goosegrass** | per-species Goosegrass | 1 | 50 |
| **cwd12-morningglory** | per-species Morningglory | 1 | 50 |
| **cwd12-nutsedge** | per-species Nutsedge | 1 | 50 |
| **cwd12-palmeramaranth** | per-species PalmerAmaranth | 1 | 50 |
| **cwd12-pricklysida** | per-species PricklySida | 1 | 50 |
| **cwd12-purslane** | per-species Purslane | 1 | 50 |
| **cwd12-ragweed** | per-species Ragweed | 1 | 50 |
| **cwd12-sicklepod** | per-species Sicklepod | 1 | 50 |
| **cwd12-spottedspurge** | per-species SpottedSpurge (only 48 available in gold) | 1 | 48 |

## Gold seed

All 598 + (12 × ~50) images come from `downloads/cottonweeddet12/train` —
the canonical CottonWeedDet12 training split. **`valid/` and `test/`
splits are the evaluation holdout and MUST NEVER be uploaded** (eval
contamination — see `feedback_research_goal_locked` in memory).

In the per-species projects, multi-species mixed images are filtered:
only that species' bboxes are kept and class_id is remapped to 0.

## Provenance tags

Every uploaded image carries two tags:

- `green` — provenance: human-verified or gold ground truth (trusted)
- `<species>` — e.g. `Carpetweeds`

Plus `batch_name = green-<species>` for upload-batch grouping in the
Roboflow Annotate UI.

Future uploads from OWL pre-annotation will use `red` tag (model
proposals, awaiting human approval). When approved → batch_name flips to
`green-<species>` so only verified annotations enter training pools.

## Workflow

```
cluster harvest  →  /classes selection  →  Roboflow per-species project
                                            ├── human labels new images (green)
                                            ├── OWL pre-annotates (red)
                                            └── human approves red → green
                                                       ↓
                                            cwd12-weeds (combined view)
                                                       ↓
                                            merge_back script → YOLO multi-class
                                                       ↓
                                            cluster training (cwd12 ≥0.90)
```

## API key

Key lives in env `ROBOFLOW_API_KEY` only. Never hardcoded, never logged.
See `memory/reference_roboflow_key.md`.

## Re-uploading is safe

Roboflow dedups identical images. Re-running
`roboflow_sync.py species-upload` is idempotent — already-uploaded images
are no-ops.

## Known platform quirks

- **Stats API lag ~minutes** on newly created projects. Counts may show 0
  for 8+ minutes after a successful upload. Do not interpret "0 images"
  on a fresh project as failure within the first ~10 min.
- **Free-tier rate limits** apply. Concurrent uploads from multiple
  processes (e.g. duplicate runs from different login nodes) can throttle
  or drop silently.
- **Roboflow doesn't have folders within a project** — the per-class
  filter in the Dataset view is the closest, hence we use separate
  projects per species.

## Merge-back to multi-class

See `tools/merge_roboflow_projects.py` (Phase C3, planned). It will pull
verified annotations from each `cwd12-<species>` project, remap each
project's cid=0 back to its CWD12 index, and write a combined YOLO
dataset for training.
