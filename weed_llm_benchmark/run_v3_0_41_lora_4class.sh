#!/bin/bash
#SBATCH --job-name=lora_4cls
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_41_lora_4cls_%a_%j.out

# v3.0.41 Phase 1 — LoRA fine-tune 4 highest-confidence cwd12 species.
# Submit as: sbatch --array=1-4 --dependency=afterok:<smoke_jobid> ...
#
# 4 classes chosen because head per-class accuracy on the cleaned bank is
# already >= 0.91 — bank correctness is near-certain. Reserves disputed
# classes (Goosegrass etc.) for after user visual audit confirms bank.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

CLASSES=(Carpetweeds Crabgrass PalmerAmaranth PricklySida)
IDX=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
CLS=${CLASSES[$IDX]}
SEED=$((300 + IDX))

echo "=== v3.0.41 LoRA 4-class array — task $IDX = $CLS ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m weed_optimizer_framework.tools.flux_lora_train \
    --class-name "$CLS" \
    --epochs 5 \
    --rank 32 \
    --alpha 16 \
    --seed "$SEED"

EXIT=$?
echo "=== $CLS exit=$EXIT $(date) ==="
ls -la $REPO/results/framework/flux_lora/$CLS/ 2>/dev/null
