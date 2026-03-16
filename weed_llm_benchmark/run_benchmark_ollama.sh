#!/bin/bash
#SBATCH --job-name=weed_ollama_bench
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_ollama_bench_%j.out
#SBATCH --error=slurm_ollama_bench_%j.err

# ==============================================================
# Weed Detection Benchmark - Ollama Model on Single Dataset
# ==============================================================
# Usage:
#   sbatch run_benchmark_ollama.sh MODEL_NAME DATASET_KEY
#   sbatch run_benchmark_ollama.sh moondream weed2okok
#   sbatch run_benchmark_ollama.sh llava:13b cottonweeddet12
#   sbatch run_benchmark_ollama.sh llama3.2-vision:11b deepweeds
#
# Models:  moondream, llava:13b, llama3.2-vision:11b
# Datasets: cottonweeddet12, deepweeds, weed2okok
# ==============================================================

set -e

MODEL_NAME="${1:?Usage: sbatch run_benchmark_ollama.sh MODEL_NAME DATASET_KEY}"
DATASET_KEY="${2:?Usage: sbatch run_benchmark_ollama.sh MODEL_NAME DATASET_KEY}"

WORK_DIR="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
OLLAMA_BIN="/ocean/projects/cis240145p/byler/ollama/bin/ollama"
OLLAMA_MODELS_DIR="/ocean/projects/cis240145p/byler/ollama/models"

cd "$WORK_DIR"

echo "============================================"
echo "Weed Detection Benchmark (Ollama)"
echo "============================================"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "Model:   $MODEL_NAME"
echo "Dataset: $DATASET_KEY"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Set ollama environment
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="0.0.0.0:11434"

# Start ollama server in background
echo "[*] Starting ollama server..."
$OLLAMA_BIN serve &
OLLAMA_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[+] Ollama server ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[!] Ollama server failed to start"
        exit 1
    fi
    sleep 1
done

# Pull model if needed
echo "[*] Ensuring model is available: $MODEL_NAME"
$OLLAMA_BIN pull "$MODEL_NAME" 2>/dev/null || true

# Initialize conda and activate
eval "$(conda shell.bash hook)"
conda activate qwen

export HF_HOME="/ocean/projects/cis240145p/byler/hf_cache"

# Run benchmark
echo ""
echo "[*] Running: python run_full_benchmark.py --dataset $DATASET_KEY --model $MODEL_NAME --resume"
python run_full_benchmark.py \
    --dataset "$DATASET_KEY" \
    --model "$MODEL_NAME" \
    --resume

# Cleanup ollama
echo "[*] Stopping ollama server..."
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null

echo ""
echo "[+] Benchmark complete at $(date)"
