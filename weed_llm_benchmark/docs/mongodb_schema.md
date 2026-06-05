# MongoDB schema design (Phase E1)

Status: **design draft**, no migration yet. Per Professor Zhang's directive
(2026-05-28): "Please use mongodb as the database for the labeler". This
document is the schema we'll migrate the current 52MB JSON registry
toward, in an incremental, reversible way.

## Why MongoDB (not just "because the professor said so")

The 52MB `dataset_registry.json` is the cause of two real problems we
already paid for in this session:

1. **Perf**: `/classes` reparsed the 52MB JSON 355× per render before
   v3.0.43.17 cached it in-memory. With many concurrent renders, this
   thrashes. A proper indexed DB makes per-class / per-slug queries cheap.
2. **Migration risk**: hardcoded Lustre paths inside the JSON couple us
   to Bridges-2. When the cluster goes away (a "few months", per the
   professor), the labeler app must move. A DB with a storage abstraction
   (see [storage.py](#storage-abstraction-phase-e2)) decouples app from
   filesystem.

MongoDB is a pragmatic pick: document model fits our current JSON shapes
with minimal rework; aggregation pipeline replaces our ad-hoc Python
scans; cheap to host on the university server alongside the FastAPI app.

## Three migration scenarios this schema must serve

The schema must support all three, so we don't have to re-design when the
deployment story shifts:

(a) **Labeler stays on cluster, uses MongoDB locally** — fixes the JSON
    re-parse perf problem without changing where the app runs. Lowest
    risk first step.

(b) **Labeler moves to Uni server with Mongo** — long-term direction per
    professor. Cluster keeps the heavy compute jobs (harvest, train);
    Uni server hosts the labeler+Mongo + serves the agent-trigger UI.
    Images stay on cluster (or move to S3/NAS as needed) accessed via
    URLs the storage abstraction resolves.

(c) **Mongo replicates state to multiple hosts for redundancy** — paper
    insurance against the cluster going dark unexpectedly. Mongo's
    replica sets handle this cleanly.

## Collections

### `slugs` — one document per harvested dataset slug

Replaces `dataset_registry.json["datasets"]`.

```js
{
  _id: "kg_arjuntejaswi__plant-village",          // slug as primary key
  source: "kaggle",                                // kaggle | huggingface | github | canonical
  status: "downloaded",                            // discovered | known | downloaded | failed | excluded
  topic: "disease",                                // cwd12 | weed | crop | disease | pest | other
  topic_source: "llm",                             // override | llm | keyword | fallback
  topic_confidence: 0.9,
  local_path: "/ocean/.../datasets/kg_arjuntejaswi__plant-village",
  storage_backend: "lustre",                       // lustre | s3 | uniserv_nas (used by storage.py)
  storage_key: "datasets/kg_arjuntejaswi__plant-village",  // backend-relative
  local_images: 20638,
  class_names: ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", ...],
  class_names_source: "yaml:data.yaml",            // yaml/file/subdirs/...
  class_names_backfilled_at: ISODate,
  bucket: "B",                                     // A | B | C (audit-derived; see bucketer)
  verdict: "ok",                                   // ok | junk | unsure (from /slugs UI)
  verdict_at: ISODate,
  verdict_by: "user",
  notes: "...",
  downloaded_at: ISODate,
  updated_at: ISODate,
}
```

Indexes:
- `topic` — every /classes group query.
- `bucket` — bucket-audit aggregations.
- `status, topic` compound — most-used filter combo.
- `local_path` unique — sanity.

### `images` — one document per image (when we need per-image metadata)

Optional / on-demand. Today we don't index images individually; just walk
disk. With Mongo, per-image metadata becomes cheap, enabling:

- Per-image provenance (green/red/yellow tags, who labeled).
- Near-dup dedup (DINOv2 embedding hash → fast lookup).
- Train/eval split discipline (mark which images are holdout).

```js
{
  _id: ObjectId(),
  slug: "cottonweed_sp8",
  image_key: "train/images/0001.jpg",              // storage-backend-relative
  bbox_count: 3,
  classes_present: ["Carpetweeds", "Crabgrass"],
  provenance: "gold",                              // gold | green | red | yellow
  labeled_by: "cottonweeddet12-gold",              // gold | human:<userid> | model:owlv2 | model:yolo-v3.0.41
  labeled_at: ISODate,
  dinov2_embedding_hash: "ab12...",                // for near-dup lookup
  cwd12_eval_holdout: false,                       // CRITICAL — never train on these
  roboflow_uploaded_to: ["cwd12-weeds", "cwd12-carpetweeds"],
  created_at: ISODate,
}
```

Indexes:
- `slug` — slug-scope queries (sample for thumbnail, etc).
- `classes_present` — per-class browsing.
- `provenance` — gold-only training set assembly.
- `cwd12_eval_holdout` — enforce eval discipline at query time.
- `dinov2_embedding_hash` — dedup.

### `classes` — canonical class registry (mostly static)

```js
{
  _id: "Goosegrass",                               // canonical PascalCase
  topic: "cwd12",                                  // cwd12 | weed | crop | disease | pest | other
  is_cwd12: true,
  cwd12_index: 3,                                  // for YOLO cid mapping
  cn_zh: "蟋蟀草",                                  // optional translation
  synonyms: ["eleusine indica", "goose grass", ...],
  notes: "...",
}
```

Indexes:
- `topic` — sidebar filter on /classes.
- `cwd12_index` — fast cid lookups.
- `synonyms` (multikey) — alias resolution at harvest time.

### `exemplars` — per-class verified visual exemplars

Powers OWL pre-annotation (image queries) and DINOv2 routing (nearest
neighbor).

```js
{
  _id: ObjectId(),
  species: "Goosegrass",
  storage_backend: "lustre",
  storage_key: "results/framework/synth_cutpaste/object_bank/Goosegrass/g0042.jpg",
  bbox_yolo: [0.5, 0.5, 0.3, 0.4],                // [cx, cy, w, h]
  source: "human" | "cutpaste" | "approved_owl",
  approved_at: ISODate,
  approved_by: "user",
  dinov2_embedding: BinData,                       // 768-dim float16; ~1.5KB
}
```

Indexes:
- `species` — per-species exemplar pulls (OWL/DINOv2 use these).
- `source` — separate human-drawn from cut-paste synth.

### `agent_tasks` — runs of cluster_action subprocesses (replaces /tmp logs)

```js
{
  _id: ObjectId(),
  action: "roboflow_sync_species",                 // matches cluster_actions key
  type: "subprocess",                              // subprocess | sbatch | refresh | restart_self
  triggered_at: ISODate,
  triggered_by: "user",                            // user (via /control click) | scheduler | tick
  pid: 1234567,                                    // subprocess pid (if subprocess)
  slurm_job_id: "41080494",                        // (if sbatch)
  log_storage_backend: "lustre",
  log_storage_key: "logs/agent_tasks/roboflow_sync_species_20260530_024500.log",
  status: "completed",                             // running | completed | failed | killed
  exit_code: 0,
  started_at: ISODate,
  ended_at: ISODate,
  result_msg: "...",
}
```

Indexes:
- `action, triggered_at` — /control history panel.
- `status` — show running tasks.

### `audit_trail` — append-only event log

Catches everything: harvest events, topic reclassifications, manual
verdicts, holdout decisions, deletions. Useful for paper methodology
("we cleaned up the dataset on date X by removing Y because Z") and for
rollback when something is wrong.

```js
{
  _id: ObjectId(),
  ts: ISODate,
  actor: "user" | "agent:<name>" | "system",
  event: "slug.verdict_set" | "slug.deleted" | "class.topic_overridden" | ...,
  target: { kind: "slug" | "class" | "image", id: "..." },
  before: { ... },
  after: { ... },
  reason: "user_audit_2026-05-27_garbage_classes_removal",
}
```

## Migration plan (incremental, reversible)

1. **Iter+1**: stand up local Mongo on the dashboard SLURM node (single
   process, no replica). Persist data file on /ocean so it survives node
   restarts. Test connection.
2. **Read path first**: introduce a `tools/db.py` wrapper that reads from
   Mongo *first* and falls back to the JSON registry. New endpoints use
   Mongo only.
3. **Write path**: when a slug is added/updated, write to BOTH Mongo and
   JSON (dual-write). Both authoritative.
4. **Backfill**: one-shot script reads current JSON → Mongo, fills all
   collections. Idempotent (key by `_id`).
5. **Switch authority**: flip `tools/db.py` to Mongo-authoritative. JSON
   becomes a snapshot export.
6. **Move to Uni server**: when ready, dump Mongo + storage manifests,
   reload on Uni server. Storage abstraction handles the path swap.

Reversible: at any point, JSON snapshot allows rollback to file-based
state.

## Storage abstraction (Phase E2)

Sibling concern: paths to actual image files must not be hardcoded to
Lustre. `tools/storage.py` defines:

```python
class StorageBackend(Protocol):
    name: str
    def get_image_path(self, slug, image_key) -> str: ...
    def get_labels_dir(self, slug) -> str: ...
    def open_image(self, slug, image_key) -> IO[bytes]: ...
    def list_images(self, slug, max_n=None) -> Iterator[str]: ...

class LustreBackend(StorageBackend):
    name = "lustre"
    root = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/datasets"
    # implements all methods by file ops

class S3Backend(StorageBackend):
    name = "s3"
    bucket = "agentai-weeds"
    # implements all methods by boto3

class UniServerNASBackend(StorageBackend):
    name = "uniserv_nas"
    root = "/srv/agentai/datasets"
    # same shape as Lustre, just different root
```

Each slug document in Mongo carries `storage_backend` + `storage_key` so
the right backend resolves the path at query time. Adding S3 later is a
config change, not a rewrite.

## What's NOT in this schema

- No raw image bytes in Mongo. Mongo holds metadata only; images stay on
  the storage backend (Lustre / S3 / NAS). This is the standard pattern.
- No model weights or training data. Those live on the storage backend
  too, referenced by `storage_key` in agent_tasks results.
- No secrets. Those stay in environment / per-key files (see
  `reference_roboflow_key`, the GH PAT).

## Open questions for the user / professor

1. Run Mongo on cluster node (transient) or on Uni server (persistent)?
   Suggest cluster-node first as a "dev DB" → move to Uni-server-hosted
   when stable.
2. Single replica or replica set? Single is simpler; replica set adds
   redundancy. Single fine for development.
3. Authentication: simple admin password env var for now; full ACL once
   the team grows. Document the connection string lives in
   `/jet/home/byler/.mongo_url`, same secret-file pattern as the
   Roboflow key and GH PAT.

— autonomous-loop iter 12, 2026-05-30

---

## v3.0.82 — Multi-domain extensibility (Prof directive 2026-06-05)

"Consider future flexibility when we collect different datasets." The weed
agent is ONE of many future dataset-collection agents (pest, crop-disease, …).
The schema must let a new agent be additive config, not a migration.

### `domains` collection — one doc per dataset-collection agent

```js
{
  _id: "weed",                                  // domain key
  display_name: "Weed detection",
  taxonomy: "cwd12",                            // canonical class taxonomy
  target_metric: { dataset: "cwd12_holdout", metric: "mAP50-95", goal: 0.90 },
  harvest_queries: ["weed detection", "cotton weed", ...],
  status: "active",                             // active | planned | paused
}
// future: { _id:"pest", taxonomy:"ip102", ... } — inserted, no schema change
```

### Changes to existing collections

- `slugs` gain **`domain`** (e.g. "weed"). Backfilled/legacy data defaults to
  "weed". All queries/UI can filter by domain → one Mongo holds every agent's
  data side by side. `db.get_registry(domain=…)`, `db.list_slugs(domain=…)`.
- `classes` gain **`domain`** + **`taxonomies: [{taxonomy, index}]`** (a class
  may live in several taxonomies). `cwd12_index`/`is_cwd12` kept for back-compat
  until readers migrate. `db.list_classes(domain=…)`.

### API

- `db.get_domains()` / `db.get_domain(id)`; `GET /api/domains` (per-domain slug
  counts + target metric).

Adding a new collection agent = insert a `domains` doc + its taxonomy + tag its
harvested slugs with `domain`. No migration of existing data.
