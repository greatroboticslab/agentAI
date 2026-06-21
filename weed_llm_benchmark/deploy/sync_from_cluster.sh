#!/usr/bin/env bash
# Runs ON lab (via weed-sync.timer). Pulls NEW harvested data + records from the
# Bridges-2 data node (incremental rsync) and refreshes the lab MongoDB.
# Auth: password via ~/.cluster_askpass.sh + ControlMaster reuse. lab->cluster only.
set -uo pipefail
CL=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
LAB="$HOME/weed_llm_benchmark"
HOST=byler@data.bridges2.psc.edu
LOG="$HOME/sync_from_cluster.log"
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=3m"
exec >>"$LOG" 2>&1
echo "[sync $(date -u +%FT%TZ)] START"

# 1) small metadata (registry / labeling events / dinov2 scores / verdicts)
mkdir -p "$LAB/results/framework/dinov2_curator"
for f in results/framework/dataset_registry.json results/framework/labeling_events.jsonl results/framework/slug_verdicts.jsonl results/framework/dinov2_curator/slug_scores.json; do
  setsid -w rsync -az -e "$RSH" "$HOST:$CL/$f" "$LAB/$f" && echo "  pulled $f" || echo "  skip $f"
done

# 2) ALL dataset dirs the registry references (incremental: only new/changed).
#    Registry slugs live in datasets/ + results/leave4out/ + downloads/ — sync all 3.
for d in downloads datasets results/leave4out; do
  mkdir -p "$LAB/$d"
  setsid -w rsync -az --partial -e "$RSH" "$HOST:$CL/$d/" "$LAB/$d/" && echo "  synced $d ($(du -sh $LAB/$d 2>/dev/null|cut -f1))"
done

# 3) fix local_path (cluster prefix -> lab REPO_ROOT) THEN re-mirror Mongo + import events.
#    fix_local_paths.py rewrites the just-pulled registry so the dashboard finds images
#    at lab paths, then mirrors to Mongo (db.get_registry reads Mongo).
cd "$LAB"; source .venv/bin/activate
REPO_ROOT="$LAB" CLUSTER_REPO="$CL" PYTHONPATH="$LAB" python deploy/fix_local_paths.py | tail -2
REPO_ROOT="$LAB" PYTHONPATH="$LAB" python -m weed_optimizer_framework.tools.mongo_import_events | tail -1
echo "[sync $(date -u +%FT%TZ)] DONE"
