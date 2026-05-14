#!/bin/bash
#SBATCH --job-name=v3_0_30_9_tta
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_30_9_hflip_tta_%j.out

# v3.0.30.9 — RF-DETR Medium @576 + horizontal-flip TTA.
# Predicts each cwd12 holdout image both original and hflipped, mirrors
# flipped boxes back, WBF-fuses, then pycocotools.
#
# Goal: push pyco mAP50-95 from 0.8877 (no TTA) to >= 0.90.
# Expected gain: +0.005 to +0.015. Cheap test before pivoting to
# RFDETRLarge (24h retrain).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.30.9 RF-DETR hflip TTA ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/v3_0_30_9_hflip_tta
RFDETR_W=$REPO/results/framework/mega_iterv3_0_29_rfdetr/run/checkpoint_best_total.pth

ls -la "$RFDETR_W" || { echo "rfdetr weights missing"; exit 1; }

mkdir -p "$OUT"
if [ ! -d "$OUT/dataset" ] && [ -d "$REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset/valid" ]; then
    ln -sfn $REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset $OUT/dataset
fi

python -m weed_optimizer_framework.tools.rfdetr_hflip_tta \
    --rfdetr-weights "$RFDETR_W" \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --threshold 0.001 \
    --wbf-iou 0.55 \
    --wbf-skip 0.001

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/tta_pyco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== TTA PYCO RESULT ==="
    cat "$SUMMARY"
fi
