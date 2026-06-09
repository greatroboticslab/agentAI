#!/bin/bash
#SBATCH --job-name=rf_pull
# v3.0.99.11: byler's allocation cis240145p is a GPU allocation (qos=gpu); RM-shared
# returned "Invalid qos". This pull is CPU work (download+extract) but GPU-shared is
# the only partition the account can use, so run there (1 GPU idle is acceptable for a
# one-off bulk grow). mem 16G is fine on GPU-shared (per-core limit is higher there).
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --output=results/framework/rf_pull_%j.out

# v3.0.99.9 (2026-06-08): reliable batch puller for Roboflow Universe datasets.
# A login-node nohup loop got reaped mid-batch, so manual Universe pulls now run as
# a proper RM-shared CPU job (survives, dedicated resources). Pass datasets via env:
#   RF_PULLS="workspace1:project1 workspace2:project2 ..."   (space-separated)
# Each is downloaded + registered + dual-written to Mongo by roboflow_source.pull.
# CPU-only (no GPU); the Roboflow SDK download works from compute via Kaggle/HF-style
# HTTPS (Roboflow CDN is reachable from compute, unlike github).

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

# self-sync git + nested→outer (drift guard, same as the other sbatch scripts)
git fetch origin main >/dev/null 2>&1 && git reset --hard origin/main >/dev/null 2>&1 \
    && echo "[sync] git reset to $(git rev-parse --short HEAD)"
if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rm -rf "$REPO/weed_optimizer_framework" 2>/dev/null
    cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/weed_optimizer_framework" \
        && echo "[sync] nested → outer ok"
fi

export ROBOFLOW_KEY=$(cat /jet/home/byler/.roboflow_key 2>/dev/null)
export ROBOFLOW_API_KEY="$ROBOFLOW_KEY"

# Two modes:
#   RF_BULK=1 [RF_TARGET=35000] [RF_MAXPULLS=30] → auto-discover + pull toward target
#   RF_PULLS='ws1:proj1 ws2:proj2 …'             → explicit list
echo "=== v3.0.99.15 Roboflow Universe batch pull ($(date)) ==="
if [ "${RF_SYNC_ONLY:-0}" = "1" ]; then
    echo "[sync-only] skipping pulls — uploading existing registry backlog to Roboflow"
elif [ "${RF_BULK:-0}" = "1" ]; then
    echo "[bulk] target=${RF_TARGET:-35000} max_pulls=${RF_MAXPULLS:-30}"
    python -u -m weed_optimizer_framework.tools.roboflow_source bulk \
        "${RF_TARGET:-35000}" "${RF_MAXPULLS:-30}"
elif [ -n "${RF_PULLS:-}" ]; then
    echo "RF_PULLS=$RF_PULLS"
    for pair in $RF_PULLS; do
        ws="${pair%%:*}"; proj="${pair#*:}"
        echo "==PULL $ws/$proj=="
        python -u -m weed_optimizer_framework.tools.roboflow_source pull "$ws" "$proj" latest
    done
else
    echo "FATAL: set RF_BULK=1 or RF_PULLS='ws1:proj1 …'"; exit 2
fi
echo "ALLPULLS_DONE $(date)"

# v3.0.99.15: prof directive — ALL collected data must land in Roboflow so humans
# review there. Auto-sync every newly-pulled slug to the Roboflow folder right after
# pulling (resumable: sync-newest-slugs skips already-synced). Disable with RF_AUTOSYNC=0.
if [ "${RF_AUTOSYNC:-1}" = "1" ]; then
    echo "=== [auto-sync] uploading new slugs to Roboflow folder weed_crop_agent_dataset ($(date)) ==="
    ROBOFLOW_FOLDER=weed_crop_agent_dataset \
    python -u -m weed_optimizer_framework.tools.roboflow_sync sync-newest-slugs \
        --folder weed_crop_agent_dataset --cap-per-slug "${RF_SYNC_CAP:-0}" 2>&1 | tail -60
    echo "=== [auto-sync] done $(date) ==="
fi
