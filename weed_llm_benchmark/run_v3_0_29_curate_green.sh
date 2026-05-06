#!/bin/bash
#SBATCH --job-name=v3029_cur
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_29_curate_%j.out

# v3.0.29 Phase 1B — green-pixel curation (REQ-3 quality)
#
# Replicates the curation step from "DINOv3 Meets YOLO26 for Weed Detection
# in Vegetable Crops" (arXiv 2603.00160, 2026): drop any image whose green
# pixel coverage < 20% (HSV: H ∈ [35,85], S > 50, V > 30). This automatically
# filters out non-plant noise like kg_parohod__warp-waste-recycling, indoor
# pest macro shots, etc. — the lingering REQ-3 violations from previous
# autonomous harvests.
#
# Output: results/framework/v3_0_29_curated_imgs.json containing
#   {slug -> {kept_count, dropped_count, kept_paths: [...]}}
# v3.0.29 pretrain reads this file and only includes kept_paths.
#
# This is a PURE CPU job — no GPU contention with v3.0.28 SAFETY/PRETRAIN
# already running. Walk the registry, hash-cache results so repeat runs are fast.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.29 Phase 1B: GREEN-PIXEL CURATION ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -m weed_optimizer_framework.tools.curate_green_pixel \
    --registry results/framework/dataset_registry.json \
    --out results/framework/v3_0_29_curated_imgs.json \
    --threshold 0.20 \
    --workers 5

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
