#!/bin/bash
#SBATCH --job-name=rf_loop
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_89_roboflow_loop_%j.out

# v3.0.89 — CLOSE THE ROBOFLOW LOOP (user top priority, professor pushing).
# Pull labeled Roboflow ground truth → train a REAL YOLO → eval cwd12 holdout.
#
# Env:
#   TRAIN_DATA   data.yaml of a labeled YOLO dataset (Roboflow export /
#                download-merge output). REQUIRED unless RF_DOWNLOAD=1.
#   RF_DOWNLOAD  1 → run download-merge first to pull the labeled version,
#                then train on its merged data.yaml.
#   EPOCHS       default 80
#   BASE_MODEL   default yolo11s.pt

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
# keep nested→outer package in sync so we run the latest code
[ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ] && \
    rm -rf "$REPO/weed_optimizer_framework" && \
    cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/weed_optimizer_framework"

EPOCHS="${EPOCHS:-80}"
BASE_MODEL="${BASE_MODEL:-yolo11s.pt}"
MERGE_OUT="$REPO/results/framework/rf_loop_merged"

echo "=== v3.0.89 Roboflow loop  SLURM_JOB_ID=$SLURM_JOB_ID  $(date) ==="

if [ "${RF_DOWNLOAD:-0}" = "1" ]; then
    [ -f /jet/home/byler/.roboflow_key ] && export ROBOFLOW_API_KEY=$(cat /jet/home/byler/.roboflow_key)
    echo "[rf] download-merge labeled versions → $MERGE_OUT"
    python -u -m weed_optimizer_framework.tools.merge_roboflow_projects \
        download-merge --out-dir "$MERGE_OUT" 2>&1 | tail -30
    TRAIN_DATA="${TRAIN_DATA:-$MERGE_OUT/data.yaml}"
fi

if [ -z "$TRAIN_DATA" ] || [ ! -f "$TRAIN_DATA" ]; then
    echo "FATAL: TRAIN_DATA not set or not found ($TRAIN_DATA)."
    echo "       Set TRAIN_DATA=/path/to/roboflow_export/data.yaml, or RF_DOWNLOAD=1."
    exit 2
fi

echo "[train] TRAIN_DATA=$TRAIN_DATA  EPOCHS=$EPOCHS  BASE=$BASE_MODEL"
python -u -m weed_optimizer_framework.tools.train_from_roboflow \
    --train-data "$TRAIN_DATA" --epochs "$EPOCHS" --model "$BASE_MODEL" 2>&1

echo "=== done $(date) ==="
