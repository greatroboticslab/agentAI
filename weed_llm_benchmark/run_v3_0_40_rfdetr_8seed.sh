#!/bin/bash
#SBATCH --job-name=v3_0_40_8s
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/framework/v3_0_40_rfdetr_seed%a_%j.out

# v3.0.40 — decisive ROBUST run: 8 seeds × RF-DETR Large cwd12-only.
#
# Why: v3.0.38-A 4-seed run gave best-of-N 0.9033, mean 0.8974 ± 0.0040.
# Goal is cwd12 mAP50-95 >= 0.90 ROBUST, not just lucky best-of-N. 4 more
# seeds (-> 8 total when combined with v3.0.31, X2, 38-A) gives us enough
# power to claim median >= 0.90 (or honestly report we cannot).
#
# This run does NOT yet include FLUX augmentation or the cleaned cumulative
# corpus — keeping the comparison apples-to-apples with v3.0.38-A so any
# delta we eventually see from v3.0.40 vs v3.0.40+augmentation is purely
# attributable to the augmentation, not seed luck.
#
# Submit:  sbatch --array=1-8 run_v3_0_40_rfdetr_8seed.sh
# (8 separate jobs, run in parallel as the GPU partition allows.)

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

# Seed = 200 + array index → seeds 201..208 (disjoint from 101/102 used
# in v3.0.38-A and the RF-DETR library default used in v3.0.31/X2).
SEED=$((200 + ${SLURM_ARRAY_TASK_ID:-1}))

echo "=== v3.0.40 RFDETRLarge cwd12-only 8-seed array ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-1}  SEED=$SEED"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/mega_iterv3_0_40_rfdetr_large_seed${SEED}

# Same config as v3.0.31 / v3.0.38: Large @704, 60ep, batch 2, ga 8.
python -m weed_optimizer_framework.tools.train_rfdetr \
    --model large \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --epochs 60 \
    --batch 2 \
    --grad-accum 8 \
    --resolution 704 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED"

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/v3_0_29_rfdetr_pycoco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== v3.0.40 seed=$SEED PYCO RESULT ==="
    cat "$SUMMARY"
fi
