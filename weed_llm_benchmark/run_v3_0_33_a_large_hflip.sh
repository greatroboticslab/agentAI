#!/bin/bash
#SBATCH --job-name=v3033a_LhT
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=results/framework/v3_0_33_a_large_hflip_%j.out

# v3.0.33 Path A — Hflip TTA on RFDETRLarge with WBF iou=0.85
# (the v3.0.30.9 hflip used iou=0.55 → -0.060 from box averaging; fixed here).
# Goal: 0.8949 → ≥ 0.90 via second-view recall + position-preserving WBF.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.33 Path A: RFDETRLarge + hflip TTA (iou=0.85) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"; echo "Date: $(date)"

OUT=$REPO/results/framework/v3_0_33_a_large_hflip
W=$REPO/results/framework/mega_iterv3_0_31_rfdetr_large/run/checkpoint_best_total.pth
ls -la "$W" || { echo "Large weights missing"; exit 1; }

mkdir -p "$OUT"
if [ ! -d "$OUT/dataset" ] && [ -d "$REPO/results/framework/mega_iterv3_0_31_rfdetr_large/dataset/valid" ]; then
    ln -sfn $REPO/results/framework/mega_iterv3_0_31_rfdetr_large/dataset $OUT/dataset
fi

python -m weed_optimizer_framework.tools.rfdetr_hflip_tta \
    --rfdetr-weights "$W" \
    --model large \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --threshold 0.001 \
    --wbf-iou 0.85 \
    --wbf-skip 0.001

echo "=== Done (exit=$?) ==="
SUMMARY=$OUT/tta_pyco_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
