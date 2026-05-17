#!/bin/bash
#SBATCH --job-name=v3036_dino
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=04:00:00
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
