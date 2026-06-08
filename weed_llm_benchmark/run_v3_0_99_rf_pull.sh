#!/bin/bash
#SBATCH --job-name=rf_pull
#SBATCH --partition=RM-shared
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

RF_PULLS="${RF_PULLS:-}"
if [ -z "$RF_PULLS" ]; then
    echo "FATAL: set RF_PULLS='ws1:proj1 ws2:proj2 ...'"; exit 2
fi

echo "=== v3.0.99.9 Roboflow Universe batch pull ==="
echo "RF_PULLS=$RF_PULLS  Date: $(date)"
for pair in $RF_PULLS; do
    ws="${pair%%:*}"; proj="${pair#*:}"
    echo "==PULL $ws/$proj=="
    python -u -m weed_optimizer_framework.tools.roboflow_source pull "$ws" "$proj" latest
done
echo "ALLPULLS_DONE $(date)"
