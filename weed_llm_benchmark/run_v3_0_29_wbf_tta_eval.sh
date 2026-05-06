#!/bin/bash
#SBATCH --job-name=v3029_wbf
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_29_wbf_eval_%j.out

# v3.0.29 Phase 1A — WBF + multi-scale TTA evaluation on v3.0.28 SAFETY best.pt
#
# Per arXiv 2603.00160 + 2026 detection-competition meta:
#   inference at imgsz = [768, 1024, 1280, 1536]
#   each with horizontal flip variant (8 scale-flip combos total)
#   merge per-image detections via Weighted Boxes Fusion (WBF)
#
# Expected gain: +0.02 to +0.05 mAP50-95 absolute over single-scale eval.
# v3.0.28 SAFETY single-scale: 0.896 → expected WBF/TTA: 0.92-0.95.
#
# This script DOES NOT retrain anything. It only changes inference.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.29 WBF+TTA eval ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

pip install --quiet ensemble-boxes 2>&1 | tail -3

# Default: evaluate v3.0.28 SAFETY best.pt. Override via WEIGHTS env var.
WEIGHTS=${WEIGHTS:-"results/framework/mega_iterv3_0_28_safety/runs/safety/weights/best.pt"}
echo "Weights: $WEIGHTS"
ls -la "$WEIGHTS" 2>&1

python -m weed_optimizer_framework.tools.wbf_tta_eval \
    --weights "$WEIGHTS" \
    --val-imgs results/framework/mega_iterv3_0_28_safety/valid/images \
    --val-lbls results/framework/mega_iterv3_0_28_safety/valid/labels \
    --imgszs 768 1024 1280 1536 \
    --hflip \
    --conf 0.001 --iou 0.6 \
    --wbf-iou 0.55 --wbf-skip 0.001 \
    --out results/framework/v3_0_29_wbf_eval_summary.json

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
