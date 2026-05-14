#!/bin/bash
#SBATCH --job-name=v3_0_30_7_ens
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_30_7_ensemble_%j.out

# v3.0.30.7 — WBF ensemble of RF-DETR (Medium @576, pyco 0.8877) and
# yolo26x SAFETY (cwd12-train-only @1024, pyco 0.7446). Different
# inductive biases (DETR transformer vs CNN anchor-free) → independent
# failure modes → ensemble should beat either alone.
#
# Goal: pyco mAP50-95 ≥ 0.90. Current best (RF-DETR alone) = 0.8877.
# Gap to close: 0.012. WBF typically adds +0.01-0.04 between different
# arches.

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

echo "=== v3.0.30.7 RF-DETR + yolo26x WBF ensemble eval ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/v3_0_30_7_ensemble
RFDETR_W=$REPO/results/framework/mega_iterv3_0_29_rfdetr/run/checkpoint_best_total.pth
YOLO_W=$REPO/results/framework/mega_iterv3_0_29_safety_long/runs/safety_long/weights/best.pt

ls -la "$RFDETR_W" || { echo "rfdetr weights missing"; exit 1; }
ls -la "$YOLO_W"   || { echo "yolo weights missing"; exit 1; }

# Reuse cwd12 staging from RF-DETR run (saves ~30s)
mkdir -p "$OUT"
if [ ! -d "$OUT/dataset/valid" ] && [ -d "$REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset/valid" ]; then
    echo "Reusing cwd12 staging from rfdetr run..."
    ln -sfn $REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset $OUT/dataset
fi

python -m weed_optimizer_framework.tools.rfdetr_yolo_ensemble_eval \
    --rfdetr-weights "$RFDETR_W" \
    --yolo-weights "$YOLO_W" \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --rfdetr-resolution 576 \
    --yolo-imgsz 1024 \
    --rfdetr-weight 2.0 \
    --yolo-weight 1.0 \
    --wbf-iou 0.55 \
    --wbf-skip 0.001 \
    --threshold 0.001

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/ensemble_pyco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== ENSEMBLE PYCO RESULT ==="
    cat "$SUMMARY"
fi
