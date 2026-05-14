#!/bin/bash
#SBATCH --job-name=v3_0_31_rfdL
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=results/framework/v3_0_31_rfdetr_large_%j.out

# v3.0.31 — RFDETRLarge (87M params vs Medium 33M) finetune on cwd12.
# Backup plan if WBF sweep + hflip TTA can't close the 0.012 gap to 0.90.
#
# Key risks:
# - VRAM: 87M model on V100-32GB at batch=4 res=576 may push 25-28GB
#   (Medium used ~6GB). If OOM, halve batch to 2 with grad_accum=8.
# - Resolution: Large checkpoint may have different patch size/embed dim;
#   verify via attempt #1 before assuming res=576 works.

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

echo "=== v3.0.31 RFDETRLarge finetune ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/mega_iterv3_0_31_rfdetr_large

python -m weed_optimizer_framework.tools.train_rfdetr \
    --model large \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --epochs 60 \
    --batch 2 \
    --grad-accum 8 \
    --resolution 576 \
    --lr 1e-4 \
    --weight-decay 1e-4

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/v3_0_29_rfdetr_pycoco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== RFDETRLarge PYCO RESULT ==="
    cat "$SUMMARY"
fi
