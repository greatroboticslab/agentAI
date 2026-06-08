#!/bin/bash
#SBATCH --job-name=v3036_dino
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=04:00:00
# v3.0.99.3: 24G→60G — job was oom-killed at exit (ExitCode 0:125) on the 2026-06-07
# run; products written but sacct=OUT_OF_MEMORY mislabeled it failed. (GPU-shared caps
# at 63000M/gpu, so 60G is the safe ceiling — 64G is rejected.)
#SBATCH --output=results/framework/v3_0_36_dinov2_curator_%j.out

# v3.0.36 — DINOv2 similarity curator: implements Hongbo's collection-phase
# comparison idea. Build reference from trusted real-bbox slugs, score
# every slug in registry, report ranked scores for threshold calibration.
#
# Stage 1: build reference pool from 8 trusted slugs (cwd12 family +
#          weedsense + crop_weed_research + grass_weeds + weed_crop_aerial
#          + francesco). ~500 imgs each = ~4000 reference embeddings.
# Stage 2: score every of 80 registry slugs (50 imgs each = 4000 imgs).
# Stage 3: report sorted; user calibrates threshold; user runs flag command
#          separately (not in this job — gives time to review).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

# v3.0.99.5: self-sync git + nested→outer (same drift guard as run_v3_0_43/50) so
# `python -m weed_optimizer_framework…` always resolves the latest code regardless
# of when the dashboard last synced. See feedback_nested_outer_package_drift.
git fetch origin main >/dev/null 2>&1 && git reset --hard origin/main >/dev/null 2>&1 \
    && echo "[sync] git reset to $(git rev-parse --short HEAD)"
if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rm -rf "$REPO/weed_optimizer_framework" 2>/dev/null
    cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/weed_optimizer_framework" \
        && echo "[sync] nested → outer weed_optimizer_framework ok"
fi

echo "=== v3.0.36 DINOv2 curator (stage 1+2+3) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "=== Stage 1: build reference pool ==="
python -m weed_optimizer_framework.tools.dinov2_curator build-reference

echo ""
echo "=== Stage 2: score all slugs ==="
python -m weed_optimizer_framework.tools.dinov2_curator score-all

echo ""
echo "=== Stage 3: ranked report (for threshold calibration) ==="
python -m weed_optimizer_framework.tools.dinov2_curator report

echo ""
echo "=== Done (exit=$?) ==="
echo "Date: $(date)"
echo ""
echo "Next step (DO NOT auto-run): user reviews scores in"
echo "  $REPO/results/framework/dinov2_curator/slug_scores.json"
echo "Then runs (still dry-run):"
echo "  python -m weed_optimizer_framework.tools.dinov2_curator flag --threshold 0.45"
echo "And to apply:"
echo "  python -m weed_optimizer_framework.tools.dinov2_curator flag --threshold 0.45 --no-dry-run"
