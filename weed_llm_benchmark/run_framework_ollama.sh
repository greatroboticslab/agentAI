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

# Wait for Ollama server to be ready (up to 60s)
echo "Waiting for Ollama server..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama server ready (${i}s)"
        break
    fi
    sleep 1
done

# Pull model — retry up to 3 times
BRAIN_MODEL="${1:-qwen3:14b}"
echo "Pulling Brain model: $BRAIN_MODEL"
for attempt in 1 2 3; do
    /ocean/projects/cis240145p/byler/ollama/bin/ollama pull $BRAIN_MODEL 2>&1
    # Verify model was downloaded
    if /ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1 | grep -q "$BRAIN_MODEL"; then
        echo "Model $BRAIN_MODEL ready (attempt $attempt)"
        break
    fi
    echo "Pull attempt $attempt failed, retrying in 10s..."
    sleep 10
done

# Final check
if ! /ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1 | grep -q "$BRAIN_MODEL"; then
    echo "ERROR: Failed to pull $BRAIN_MODEL after 3 attempts"
    echo "Available models:"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1
    echo "Falling back to any available model..."
    BRAIN_MODEL=$(/ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1 | tail -1 | awk '{print $1}')
    echo "Using fallback: $BRAIN_MODEL"
fi
echo "Brain model: $BRAIN_MODEL"

# Run mode: "quick" (1h test, 3 rounds) or "long" (8h, 12 rounds)
RUN_MODE="${2:-quick}"
if [ "$RUN_MODE" = "long" ]; then
    ROUNDS=12
    NO_IMPROVE=10
    echo "Mode: LONG (12 rounds, ~8h)"
else
    ROUNDS=3
    NO_IMPROVE=2
    echo "Mode: QUICK (3 rounds, ~1h)"
fi

# Brain has 13 tools: freeze_train, lora_train, distill_train, etc.
python -m weed_optimizer_framework.run \
    --mode agent \
    --backend ollama \
    --brain $BRAIN_MODEL \
    --rounds $ROUNDS \
    --no-improve-limit $NO_IMPROVE

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
