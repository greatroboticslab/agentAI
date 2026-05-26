#!/bin/bash
#SBATCH --job-name=lora_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/v3_0_41_lora_smoke_%j.out

# v3.0.41 Phase 1 — LoRA SMOKE TEST.
# 1 epoch of FLUX LoRA fine-tuning on the highest-confidence class
# (PalmerAmaranth: head acc 0.985, 400 clean bank crops, certainly correct).
# Purpose: catch any code-level bugs in flux_lora_train.py BEFORE we launch
# 4 parallel multi-hour overnight runs.
#
# Pass criterion: produces results/framework/flux_lora/PalmerAmaranth/
# pytorch_lora_weights.safetensors + train_meta.json without crashing.
# The 4-class array job is SLURM-dependency-chained to afterok on this.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

echo "=== v3.0.41 LoRA SMOKE: PalmerAmaranth 1-epoch ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m weed_optimizer_framework.tools.flux_lora_train \
    --class-name PalmerAmaranth \
    --epochs 1 \
    --rank 32 \
    --alpha 16

EXIT=$?
echo "=== Smoke exit=$EXIT $(date) ==="
ls -la $REPO/results/framework/flux_lora/PalmerAmaranth/ 2>/dev/null
