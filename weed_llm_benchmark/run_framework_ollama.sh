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

# v3.0.9: Kaggle v2 API token for autonomous dataset search.
# Brain uses `_kaggle_http_search()` with bearer auth — no ~/.kaggle/kaggle.json needed.
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-KGAT_67eb9458d9e565587c47c967c5249584}
# Redirect kagglehub cache off the tiny HOME quota onto /ocean (7TB budget).
export KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-/ocean/projects/cis240145p/byler/kagglehub_cache}
mkdir -p "$KAGGLEHUB_CACHE" 2>/dev/null

# v3.0.7: mega gate — target is 50K real bbox images cumulative.
# With Kaggle autonomous search active (v3.0.9), 50K is reachable per round.
export WEED_MEGA_MIN_IMAGES=${WEED_MEGA_MIN_IMAGES:-50000}

echo "=== Weed Optimizer Framework — OLLAMA AGENT ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# v3.0.21: chain depth hard cap + stop_flag pre-check. Prevents accidental
# infinite chain if mega never runs and orchestrator never writes stop_flag.
CHAIN_DEPTH_FILE=results/framework/chain_depth.txt
CHAIN_DEPTH=$(cat "$CHAIN_DEPTH_FILE" 2>/dev/null || echo 0)
CHAIN_DEPTH=$((CHAIN_DEPTH + 1))
mkdir -p results/framework
echo "$CHAIN_DEPTH" > "$CHAIN_DEPTH_FILE"
echo "Chain depth: $CHAIN_DEPTH"
if [ "$CHAIN_DEPTH" -gt 40 ]; then
    echo "ABORT: chain depth $CHAIN_DEPTH > 40 safety cap"
    exit 0
fi
if [ -f results/framework/stop_chain.txt ]; then
    echo "ABORT: stop_chain.txt present at start → not running this job"
    cat results/framework/stop_chain.txt
    exit 0
fi

# v3.0.21: BULLETPROOF auto-chain. Pre-queue a follow-up job with afterany
# dependency so it runs no matter how this one ends (success, walltime, crash).
# Orchestrator writes results/framework/stop_chain.txt when plateau detected,
# and this script scancels the queued follow-up at its own end if that flag
# exists. Previously we relied on post-python `if [-f should_continue]; sbatch`
# which SLURM's walltime SIGKILL could skip entirely — killing the chain.
PRE_ARGS_1="${1:-gemma4}"
PRE_ARGS_2="${2:-quick}"
NEXT_JOB_ID=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID:-0} \
    run_framework_ollama.sh "$PRE_ARGS_1" "$PRE_ARGS_2" 2>/dev/null | tail -1)
if [ -n "$NEXT_JOB_ID" ] && [ "$NEXT_JOB_ID" -eq "$NEXT_JOB_ID" ] 2>/dev/null; then
    echo "$NEXT_JOB_ID" > results/framework/next_job_id.txt
    echo "Pre-queued next-in-chain job: $NEXT_JOB_ID (afterany dependency)"
else
    echo "Pre-queue failed or not applicable (likely first-run probe)"
    rm -f results/framework/next_job_id.txt
fi

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
BRAIN_MODEL="${1:-gemma4}"
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
# v3.0.21: CHAIN TEARDOWN — cancel the pre-queued follow-up ONLY if
# orchestrator wrote stop_chain.txt (plateau or safety cap reached).
# If walltime killed us before this point, follow-up still runs from
# its afterany dependency — exactly what we want for bulletproofness.
# ============================================
if [ -f results/framework/stop_chain.txt ] && [ -f results/framework/next_job_id.txt ]; then
    NEXT_ID=$(cat results/framework/next_job_id.txt | tr -d '\n')
    echo "Orchestrator requested chain stop — cancelling follow-up job $NEXT_ID"
    scancel "$NEXT_ID" 2>&1 || true
    rm -f results/framework/stop_chain.txt results/framework/next_job_id.txt
else
    echo "Chain continues — follow-up job already queued (afterany)."
fi
