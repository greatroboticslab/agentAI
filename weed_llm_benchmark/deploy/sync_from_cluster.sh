#!/usr/bin/env bash
# sync_from_cluster.sh — run ON the lab server (via cron/systemd timer) to PULL
# newly harvested data + records from the cluster. Direction = lab→cluster ssh
# (outbound from lab is reliable; cluster→lab is usually firewalled).
#
# Needs passwordless ssh from lab → cluster (ssh-copy-id once, or expect wrapper).
# CONFIRM/EDIT:
CLUSTER="${CLUSTER:-byler@bridges2.psc.edu}"
CLUSTER_REPO="${CLUSTER_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
LAB_REPO="${LAB_REPO:-$HOME/weed_llm_benchmark}"
set -euo pipefail

echo "[sync $(date -u +%FT%TZ)] pulling images + records from cluster ..."

# 1) dataset images (downloads/) + key framework artifacts (registry, dinov2 scores,
#    labeling events, slug verdicts). -a archive, --partial resume, exclude huge tmp.
rsync -az --partial --info=progress2 \
  --include='downloads/***' \
  --include='results/framework/dataset_registry.json' \
  --include='results/framework/labeling_events.jsonl' \
  --include='results/framework/dinov2_curator/***' \
  --include='results/framework/slug_verdicts.jsonl' \
  --include='results/' --include='results/framework/' \
  --exclude='*' \
  "$CLUSTER:$CLUSTER_REPO/" "$LAB_REPO/" || echo "[sync] rsync had issues (continuing)"

# 2) import harvest/labeling records into lab MongoDB.
#    labeling_tracker already dual-writes Mongo + JSONL fallback, so replaying the
#    pulled labeling_events.jsonl into the lab Mongo keeps the source-of-truth in sync.
cd "$LAB_REPO"
[ -d .venv ] && source .venv/bin/activate || true
REPO_ROOT="$LAB_REPO" MONGO_URL="${MONGO_URL:-mongodb://127.0.0.1:27017}" \
  python3 -m weed_optimizer_framework.tools.mongo_import_events \
    --events "$LAB_REPO/results/framework/labeling_events.jsonl" 2>/dev/null \
  || echo "[sync] NOTE: mongo_import_events not present yet — to be added during migration"

echo "[sync] done."
# To push human-verified ground truth BACK to the cluster for training, use the
# reverse rsync (lab→cluster) in a separate step; training reads it on the cluster.
