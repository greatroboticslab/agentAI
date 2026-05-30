#!/bin/bash
#SBATCH --job-name=dinov2_rt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/framework/v3_0_52_dinov2_route_%j.out

# v3.0.52 — DINOv2 embedding + nearest-neighbor routing (weed-vs-not-weed
# quality gate + species router for bucket C unknowns + near-dup dedup).
# See weed_optimizer_framework/tools/dinov2_route.py.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

# Defaults — override via env
DINOV2_TARGET_DIR="${DINOV2_TARGET_DIR:-$REPO/downloads/cottonweeddet12/valid/images}"
DINOV2_EXEMPLAR_ROOT="${DINOV2_EXEMPLAR_ROOT:-$REPO/results/framework/synth_cutpaste/object_bank}"
DINOV2_OUT="${DINOV2_OUT:-$REPO/results/framework/dinov2_routing.json}"
DINOV2_MODEL="${DINOV2_MODEL:-facebook/dinov2-base}"
DINOV2_TOP_K="${DINOV2_TOP_K:-5}"
DINOV2_REJECT_BELOW="${DINOV2_REJECT_BELOW:-0.5}"
DINOV2_MAX_IMAGES="${DINOV2_MAX_IMAGES:-0}"

echo "=== v3.0.52 DINOv2 routing ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "  target:        $DINOV2_TARGET_DIR"
echo "  exemplar-root: $DINOV2_EXEMPLAR_ROOT"
echo "  model:         $DINOV2_MODEL"
echo "  out:           $DINOV2_OUT"
echo "  top-K:         $DINOV2_TOP_K"
echo "  reject-below:  $DINOV2_REJECT_BELOW"

if [ ! -d "$DINOV2_TARGET_DIR" ]; then
    echo "FATAL: target-dir not found: $DINOV2_TARGET_DIR"
    exit 2
fi
if [ ! -d "$DINOV2_EXEMPLAR_ROOT" ]; then
    echo "FATAL: exemplar-root not found: $DINOV2_EXEMPLAR_ROOT"
    exit 2
fi

python -u -m weed_optimizer_framework.tools.dinov2_route \
    --target-dir "$DINOV2_TARGET_DIR" \
    --exemplar-root "$DINOV2_EXEMPLAR_ROOT" \
    --out "$DINOV2_OUT" \
    --model-id "$DINOV2_MODEL" \
    --top-k "$DINOV2_TOP_K" \
    --reject-below "$DINOV2_REJECT_BELOW" \
    --max-images "$DINOV2_MAX_IMAGES" 2>&1 \
  | tee -a $REPO/results/framework/v3_0_52_dinov2_route_${SLURM_JOB_ID}.log

echo "=== exit: $? ==="
