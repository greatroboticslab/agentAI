#!/bin/bash
#SBATCH --job-name=v3033d_MLW
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=results/framework/v3_0_33_d_med_large_wbf_%j.out

# v3.0.33 Path D — Medium + Large peer-strength WBF ensemble.
# Medium 0.8877 + Large 0.8949 (delta only 0.007 — peer, unlike the failed
# yolo+RFDETR with delta 0.143). WBF iou=0.85 + weights=[1, 1.5].
# Goal: push above 0.90.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.33 Path D: Medium + Large WBF ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"; echo "Date: $(date)"

OUT=$REPO/results/framework/v3_0_33_d_med_large_wbf
M=$REPO/results/framework/mega_iterv3_0_29_rfdetr/run/checkpoint_best_total.pth
L=$REPO/results/framework/mega_iterv3_0_31_rfdetr_large/run/checkpoint_best_total.pth
ls -la "$M" "$L" || { echo "weights missing"; exit 1; }

mkdir -p "$OUT"
if [ ! -d "$OUT/dataset" ] && [ -d "$REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset/valid" ]; then
    ln -sfn $REPO/results/framework/mega_iterv3_0_29_rfdetr/dataset $OUT/dataset
fi

python -m weed_optimizer_framework.tools.rfdetr_medium_large_wbf \
    --medium-weights "$M" \
    --large-weights "$L" \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --threshold 0.001 \
    --wbf-iou 0.85 \
    --wbf-skip 0.001 \
    --medium-weight 1.0 \
    --large-weight 1.5

echo "=== Done (exit=$?) ==="
SUMMARY=$OUT/ml_wbf_pyco_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
