#!/bin/bash
#SBATCH --job-name=weed_agent_ollama
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/framework/slurm_ollama_%j.out

eval "$(conda shell.bash hook)"
conda activate bench

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models

echo "=== Weed Optimizer Framework — OLLAMA AGENT ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Start Ollama server in background
echo "Starting Ollama server..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve &
OLLAMA_PID=$!
sleep 5

# Pull model if not cached
echo "Loading model..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama pull qwen2.5:7b 2>&1 | tail -3

# Run framework with Ollama backend
python -m weed_optimizer_framework.run \
    --mode agent \
    --backend ollama \
    --brain qwen2.5:7b \
    --rounds 3

EXIT_CODE=$?

# Stop Ollama
kill $OLLAMA_PID 2>/dev/null

echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# ============================================
# JOB CHAIN: auto-submit next round if needed
# ============================================
if [ -f results/framework/should_continue.txt ]; then
    rm results/framework/should_continue.txt
    echo "Auto-submitting next optimization round..."
    sbatch run_framework_ollama.sh
fi
