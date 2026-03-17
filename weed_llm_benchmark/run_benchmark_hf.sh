#!/bin/bash
#SBATCH --job-name=weed_bench
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_bench_%j.out
#SBATCH --error=slurm_bench_%j.err

# ==============================================================
# Weed Detection Benchmark - HuggingFace Model on Single Dataset
# ==============================================================
# Usage:
#   sbatch run_benchmark_hf.sh MODEL_KEY DATASET_KEY
#   sbatch run_benchmark_hf.sh qwen7b cottonweeddet12
#   sbatch run_benchmark_hf.sh qwen7b weed2okok
#   sbatch run_benchmark_hf.sh qwen3b cottonweeddet12
#   sbatch run_benchmark_hf.sh minicpm weed2okok
#   sbatch run_benchmark_hf.sh internvl2 deepweeds
#   sbatch run_benchmark_hf.sh florence2 cottonweeddet12
#
# Models:  qwen7b, qwen3b, minicpm, internvl2, florence2
# Datasets: cottonweeddet12, deepweeds, weed2okok
# ==============================================================

set -e

MODEL_KEY="${1:?Usage: sbatch run_benchmark_hf.sh MODEL_KEY DATASET_KEY}"
DATASET_KEY="${2:?Usage: sbatch run_benchmark_hf.sh MODEL_KEY DATASET_KEY}"

WORK_DIR="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
cd "$WORK_DIR"

echo "============================================"
echo "Weed Detection Benchmark (HuggingFace)"
echo "============================================"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "Model:   $MODEL_KEY"
echo "Dataset: $DATASET_KEY"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Initialize conda
eval "$(conda shell.bash hook)"

# Select conda env based on model
# - bench:  transformers 4.57 (Qwen2.5-VL, MiniCPM, etc.)
# - compat: transformers 4.46 (InternVL2, Florence-2 need older API)
case "$MODEL_KEY" in
    internvl2|florence2)
        ENV_NAME="compat"
        ;;
    *)
        ENV_NAME="bench"
        ;;
esac
echo "[*] Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"

export HF_HOME="/ocean/projects/cis240145p/byler/hf_cache"
export TRANSFORMERS_CACHE="/ocean/projects/cis240145p/byler/hf_cache/hub"

# Verify environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import accelerate; print(f'accelerate {accelerate.__version__}')"

# Run benchmark with evaluation
echo ""
echo "[*] Running: python run_full_benchmark.py --dataset $DATASET_KEY --model $MODEL_KEY --resume"
python run_full_benchmark.py \
    --dataset "$DATASET_KEY" \
    --model "$MODEL_KEY" \
    --resume

echo ""
echo "[+] Benchmark complete at $(date)"
