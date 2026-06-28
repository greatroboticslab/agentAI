# AgentAI Platform — Feature-Completeness Review & Roadmap

_Professional gap analysis of the foundational features, 2026-06-27._
_Goal (Prof Zhang): a general platform where many students/agents collect data of any kind and
train any model — not a weed/YOLO-only tool. "Physical AI" scope: images, video, and robot sensor
streams (GPS/IMU/LiDAR), with detection, classification, segmentation, and RL/policy learning._

Legend: ✅ done · 🟡 partial · ❌ missing

---

## 1. Agent model & creation
Today: an "agent" = a `domain` doc {display_name, taxonomy, target_metric, harvest_queries,
n_subagents}. The New-Agent UI only captures name + queries. Every agent is implicitly an
image-harvesting, YOLO-detection collector.

- ✅ Create domain, list on launcher, per-domain workspace, upload panel.
- ❌ **Agent TYPE** (Collector / Trainer / Labeler / Evaluator / Robot-client).
- ❌ **Modality** (image / video / multimodal-sensor / point-cloud / text / audio).
- ❌ **Task** (detection / classification / segmentation / pose / tracking / RL-policy / SSL-pretrain).
- ❌ **Model choice** + default hyperparameters per agent.
- ❌ Agent lifecycle/state machine beyond "created".

**Recommendation:** extend the agent schema + a New-Agent **wizard** (type → modality → task →
model → taxonomy/eval). This is the keystone — almost everything below depends on it.

## 2. Dataset formats & ingestion
Today: upload accepts a **.zip of images only** (.jpg/.jpeg/.png/.bmp/.webp) + YOLO `labels/*.txt`
+ `data.yaml`. Raw-body in memory, 2 GB cap.

- ✅ Images (labeled via YOLO txt, or unlabeled).
- 🟡 Annotation formats: YOLO only. ❌ COCO JSON, ❌ Pascal VOC XML, ❌ segmentation masks,
  ❌ keypoints/pose, ❌ CreateML/CVAT.
- ❌ **Video** (.mp4/.mov) — no ingest, no frame extraction.
- ❌ **Robot sensor data** (the "physical AI" core): GPS, IMU, LiDAR/point-cloud (.pcd/.bin),
  depth, multi-camera, CSV/parquet time-series, **rosbag/.mcap**.
- ❌ Audio, text.
- ❌ Generic "declare-a-schema + arbitrary files" fallback so nothing is rejected.
- ❌ Streaming / chunked / resumable upload (robotics datasets are large), checksum, dedup.

**Recommendation:** add a **modality+format field** at upload; format-specific validators &
previewers; at minimum **accept + catalog any archive with a declared type** even if we can't
preview it yet; add video→frame extraction and COCO/VOC→normalize importers; stream large uploads
to disk instead of memory.

## 3. Labeling / Roboflow
Today: push N sampled images (with YOLO annotations) to Roboflow for human labeling; user-set cap;
push-back/merge; verdicts in dashboard.

- ✅ Image bounding-box labeling via Roboflow, cap, push-back, attribution.
- ❌ Segmentation / keypoint / polygon labeling.
- ❌ Video & sensor labeling (Roboflow can't; need CVAT / Label Studio).
- ❌ Label-store abstraction (single hard dependency on Roboflow + its free-tier cap).
- 🟡 In-dashboard review exists (verdicts/exemplars) but isn't a full labeling tool.

**Recommendation:** abstract a "label backend" interface; add CVAT/Label-Studio path for
non-box/non-image; keep Roboflow as the image-detection default.

## 4. Training & research methods
Today: one path — supervised **YOLO object detection** (clean_train_d: DINO-gated subset →
train → cwd12 eval → mAP write-back). RF-DETR & LoRA exist in code, classification/segmentation
referenced but not wired to the UI.

- ✅ Supervised detection (YOLO), cwd12 eval, round write-back, user-set epochs.
- 🟡 RF-DETR (code present, not a one-click UI mode).
- ❌ **Classification** training UI, ❌ **Segmentation**, ❌ pose/tracking.
- ❌ **Self-supervised / pretraining** (DINOv2 is used to FILTER, not to train a model).
- ❌ **Semi-/weakly-supervised**, ❌ **active learning** loop (only a sampling seed), ❌ **continual/
  online learning** (rounds are a seed but no incremental fine-tune-from-last).
- ❌ **Reinforcement / imitation learning** (needed for the laser-weeding & humanoid policies — the
  actual "physical AI" endgame).
- ❌ Model zoo / picker, hyperparameter UI, multi-dataset/round selection, transfer-learning &
  **generalization/cross-domain** eval (train on A, test on B).

**Recommendation:** a **"Training job" abstraction** = pick task → model → dataset(s)/round →
hyperparams → submit (whitelisted, safe templates per task). Ship classification + segmentation
templates next; design an **RL/policy agent type** for robot control as a first-class future track.

## 5. Generalization / multi-domain
Today: domain docs exist, but downstream is weed-specific (CWD12 classes, weed vocab, YOLO,
cottonweed eval). A new domain can upload but has no harvest vocab / taxonomy / training / eval.

- 🟡 Domain scoping in views; ❌ per-domain harvest vocab, ❌ per-domain taxonomy/eval set,
  ❌ per-domain training template. **Recommendation:** make taxonomy + eval + training-template
  per-domain config; decouple the weed specifics into the "weed" domain's config.

## 6. Evaluation & metrics
- ✅ cwd12 mAP50-95 (weed). ❌ per-domain eval sets, ❌ per-task metrics (acc/F1, IoU/Dice,
  RL success-rate), ❌ leaderboards / run comparison / training-curve visualization.

## 7. Data governance (critical once public)
- ✅ Attribution, delete-own, audit_trail. ❌ **license/consent metadata** (community uploads!),
  ❌ dataset **cards** (provenance, schema, splits, stats), ❌ versioning/lineage, ❌ PII/safety
  review queue for public data, ❌ curated-dataset **export/download**, ❌ cross-user dedup.

## 8. Programmatic access & ops
- ✅ Upload API + laptop client; RBAC; admin notifications; smoke tests.
- ❌ **Per-user API tokens** (the laptop/robot clients currently reuse Basic 1/1), ❌ documented
  public API/SDK, ❌ per-token rate limits, ❌ per-user storage/GPU quota dashboards.

---

## Recommended roadmap (highest-leverage first)
1. **Generalize the Agent + Dataset schema** (type · modality · task · model) + New-Agent wizard.
   Keystone — unlocks everything else. (Backwards-compatible: weed becomes one configured agent.)
2. **Accept-any-data ingestion**: modality/format field, store+catalog unknown types, video frame
   extraction, COCO/VOC importers, stream large files to disk. So "nothing gets rejected".
3. **Training-job abstraction** + classification & segmentation templates (beyond YOLO detection).
4. **Per-domain eval + metric registry** (so non-weed agents have a real target & leaderboard).
5. **Governance for public data**: dataset cards + license field + export/download.
6. **Per-user API tokens** for robot/laptop clients (retire shared Basic for automation).
7. **RL/policy agent track** for the laser-weeding & humanoid control endgame.

Each item ships behind the existing safety model (RBAC, whitelisted cluster templates, smoke tests).
