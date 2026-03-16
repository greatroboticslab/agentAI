#!/bin/bash
#SBATCH --job-name=yolo_baseline
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_yolo_%j.out
#SBATCH --error=slurm_yolo_%j.err

# Usage:
#   sbatch run_yolo_baseline.sh weed2okok zero-shot
#   sbatch run_yolo_baseline.sh cottonweeddet12 fine-tune 50
#   sbatch run_yolo_baseline.sh deepweeds zero-shot

DATASET=${1:-weed2okok}
MODE=${2:-zero-shot}
EPOCHS=${3:-50}

echo "============================================"
echo "YOLO Baseline: ${DATASET} (${MODE})"
echo "============================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Setup environment
export HF_HOME="/ocean/projects/cis240145p/byler/hf_cache"
eval "$(conda shell.bash hook)"
conda activate qwen

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark

python run_yolo_baseline.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --epochs "${EPOCHS}" \
    --evaluate

echo ""
echo "Done: $(date)"
