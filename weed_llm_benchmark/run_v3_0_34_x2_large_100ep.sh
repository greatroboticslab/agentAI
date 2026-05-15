#!/bin/bash
#SBATCH --job-name=v3034x2_L100
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=results/framework/v3_0_34_x2_large_100ep_%j.out

# v3.0.34 X2 — RFDETRLarge @704 trained for 100 epochs (vs 60 in v3.0.31).
# X1 NMS-fusion = 0.8911, still < Large alone 0.8949. TTA/ensemble paths
# all fail (WBF=5x negative, NMS=-0.004). Only remaining detection-side
# lever: train longer. v3.0.31 ran 60 epochs in 14.5h; 100 epochs fits 36h
# walltime comfortably.
#
# Expected: 0.8949 + 0.005-0.015 = 0.90-0.91 (if model hasn't plateaued).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.34 X2: RFDETRLarge @704 100 epochs ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/mega_iterv3_0_34_x2_large_100ep

python -m weed_optimizer_framework.tools.train_rfdetr \
    --model large \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --epochs 100 \
    --batch 2 \
    --grad-accum 8 \
    --resolution 704 \
    --lr 1e-4 \
    --weight-decay 1e-4

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/v3_0_29_rfdetr_pycoco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== RFDETRLarge 100-ep PYCO RESULT ==="
    cat "$SUMMARY"
fi
