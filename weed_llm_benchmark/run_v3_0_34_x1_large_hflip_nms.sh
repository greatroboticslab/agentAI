#!/bin/bash
#SBATCH --job-name=v3034x1_LhN
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=results/framework/v3_0_34_x1_large_hflip_nms_%j.out

# v3.0.34 X1 — Large + hflip TTA with greedy NMS fusion (NOT WBF).
# 5 prior WBF experiments all dropped 0.018-0.060 below the best single
# model. The box-position averaging is structurally bad for mAP50-95 at
# high IoU thresholds. Greedy NMS preserves the strong model's positions
# (no averaging) while still capturing the second-view recall benefit.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.34 X1: Large + hflip + greedy NMS (no WBF) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"; echo "Date: $(date)"

OUT=$REPO/results/framework/v3_0_34_x1_large_hflip_nms
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
    --fusion nms \
    --nms-iou 0.5

echo "=== Done (exit=$?) ==="
SUMMARY=$OUT/tta_pyco_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
