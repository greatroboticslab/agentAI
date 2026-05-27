#!/bin/bash
#SBATCH --job-name=topic_bf
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=results/framework/v3_0_43_topic_backfill_%j.out

# v3.0.43.4 — one-shot LLM-based topic backfill for all 416 known classes.
# Triggerable from /control UI.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
export CLASS_TOPIC_OVERRIDES_FILE="$REPO/results/framework/class_topic_overrides.json"

echo "=== v3.0.43.4 topic backfill (triggered from /control) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Start ollama
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 ; then
    echo "[ollama] starting…"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama serve \
        >/tmp/ollama_$SLURM_JOB_ID.log 2>&1 &
    OLLAMA_PID=$!
    for i in $(seq 1 60); do
        curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

# Pre-pull the model we use (gemma4:latest)
MODEL=gemma4:latest
echo "[ollama] ensuring $MODEL is pulled…"
/ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1 | head -20
if ! /ocean/projects/cis240145p/byler/ollama/bin/ollama list 2>&1 | grep -q "$MODEL" ; then
    echo "[ollama] pulling $MODEL (this can take a few min for first run)…"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama pull $MODEL 2>&1 | tail -5
fi

# Run the backfill
echo "[backfill] starting…"
python -m weed_optimizer_framework.tools.topic_backfill_all --model $MODEL

# Stop ollama to free GPU
if [ -n "$OLLAMA_PID" ]; then
    echo "[ollama] stopping pid=$OLLAMA_PID"
    kill $OLLAMA_PID 2>/dev/null
fi

echo "=== topic backfill done $(date) ==="
