#!/bin/bash
#SBATCH --job-name=weed_llm_bench
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_hf_%j.out
#SBATCH --error=slurm_hf_%j.err

# ==============================================================
# Weed Detection LLM Benchmark - HuggingFace Models
# ==============================================================
# Usage:
#   sbatch run_hf_benchmark.sh                    # test all models
#   sbatch run_hf_benchmark.sh qwen7b             # test specific model
#   sbatch run_hf_benchmark.sh qwen7b myimage.jpg # test specific model + image
# ==============================================================

set -e

WORK_DIR="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
cd "$WORK_DIR"

# Model to test (default: qwen7b, pass 'all' for all models)
MODEL="${1:-qwen7b}"
IMAGE_ARG="${2:-}"

echo "============================================"
echo "Weed Detection LLM Benchmark (HuggingFace)"
echo "============================================"
echo "Date:  $(date)"
echo "Node:  $(hostname)"
echo "Model: $MODEL"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Activate conda env based on model
case "$MODEL" in
    qwen7b|qwen3b)
        echo "[*] Activating conda env: qwen"
        source activate qwen 2>/dev/null || conda activate qwen
        ;;
    minicpm)
        echo "[*] Activating conda env: minicpm"
        source activate minicpm 2>/dev/null || conda activate minicpm
        ;;
    all)
        # For 'all', we use a general env; individual models may fail if deps missing
        echo "[*] Activating conda env: qwen (default for 'all')"
        source activate qwen 2>/dev/null || conda activate qwen
        ;;
    *)
        echo "[*] Activating conda env: qwen (default)"
        source activate qwen 2>/dev/null || conda activate qwen
        ;;
esac

export HF_HOME="/ocean/projects/cis240145p/byler/hf_cache"
export TRANSFORMERS_CACHE="/ocean/projects/cis240145p/byler/hf_cache/hub"

# Build command
CMD="python test_hf_models.py --model $MODEL"
if [ -n "$IMAGE_ARG" ]; then
    CMD="$CMD --image $IMAGE_ARG"
fi

echo "[*] Running: $CMD"
echo ""
$CMD

echo ""
echo "[+] Benchmark complete at $(date)"
