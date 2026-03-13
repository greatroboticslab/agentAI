#!/bin/bash
#SBATCH --job-name=weed_ollama
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_ollama_%j.out
#SBATCH --error=slurm_ollama_%j.err

# ==============================================================
# Weed Detection LLM Benchmark - Ollama Models
# ==============================================================
# Usage:
#   sbatch run_ollama_benchmark.sh                          # test all models
#   sbatch run_ollama_benchmark.sh llava:13b                # specific model
#   sbatch run_ollama_benchmark.sh llava:13b myimage.jpg    # specific model + image
# ==============================================================

set -e

WORK_DIR="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
OLLAMA_BIN="/ocean/projects/cis240145p/byler/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/ocean/projects/cis240145p/byler/ollama/models"

cd "$WORK_DIR"

MODEL="${1:-all}"
IMAGE_ARG="${2:-}"

echo "============================================"
echo "Weed Detection LLM Benchmark (Ollama)"
echo "============================================"
echo "Date:  $(date)"
echo "Node:  $(hostname)"
echo "Model: $MODEL"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Set ollama environment
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="0.0.0.0:11434"

# Start ollama server
echo "[*] Starting ollama server..."
$OLLAMA_BIN serve &
OLLAMA_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[+] Ollama server ready."
        break
    fi
    sleep 1
done

# Activate Python env
source activate qwen 2>/dev/null || conda activate qwen

# Build command
CMD="python test_ollama.py --model $MODEL"
if [ -n "$IMAGE_ARG" ]; then
    CMD="$CMD --image $IMAGE_ARG"
fi

echo "[*] Running: $CMD"
echo ""
$CMD

# Cleanup
echo "[*] Stopping ollama server..."
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null

echo ""
echo "[+] Benchmark complete at $(date)"
