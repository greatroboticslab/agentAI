#!/bin/bash
#SBATCH --job-name=v3_0_30_8_wbf
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/v3_0_30_8_wbf_sweep_%j.out

# v3.0.30.8 — cache per-model preds + sweep WBF params over 9 combinations.
# Goal: confirm or refute the ensemble path. If best combo ≥ 0.89, keep
# ensemble alive; if all < 0.89, ensemble is dead, pivot to TTA / Large.

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

echo "=== v3.0.30.8 WBF param sweep ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/v3_0_30_8_wbf_sweep
RFDETR_W=$REPO/results/framework/mega_iterv3_0_29_rfdetr/run/checkpoint_best_total.pth
YOLO_W=$REPO/results/framework/mega_iterv3_0_29_safety_long/runs/safety_long/weights/best.pt

ls -la "$RFDETR_W" || { echo "rfdetr weights missing"; exit 1; }
ls -la "$YOLO_W"   || { echo "yolo weights missing"; exit 1; }

mkdir -p "$OUT"
# Reuse staging from prior ensemble run if present
if [ ! -d "$OUT/dataset" ] && [ -d "$REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset/valid" ]; then
    ln -sfn $REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset $OUT/dataset
fi

python -m weed_optimizer_framework.tools.wbf_sweep \
    --rfdetr-weights "$RFDETR_W" \
    --yolo-weights "$YOLO_W" \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --rfdetr-threshold 0.001 \
    --yolo-threshold 0.001

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/wbf_sweep_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== WBF SWEEP BEST ==="
    python -c "import json; d=json.load(open('$SUMMARY')); b=d['best_combo']; print(f\"BEST: {b['name']} → mAP50-95={b['mAP50_95']:.4f}\")"
fi
