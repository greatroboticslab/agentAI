#!/bin/bash
#SBATCH --job-name=v3035_qual
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/v3_0_35_quality_eval_%j.out

# v3.0.35 — Empirical benchmark: OWLv2 vs Gemma 4 vision for weed detection /
# data quality filtering. Decides v3.0.36 architecture (Gemma 4's role).
#
# T1 OWLv2 baseline (full 1977 cwd12 holdout) — fast
# T2 Gemma 4 direct bbox (100 sample)            — Gemma is slow
# T3 Gemma 4 image relevance (100 pos + 50 neg)
# T4 Gemma 4 bbox verification (100 sample)

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export OLLAMA_HOST=127.0.0.1:11434

echo "=== v3.0.35 data quality eval ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Start Ollama for Gemma 4 vision (T2/T3/T4)
echo "Starting Ollama..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve &
OLLAMA_PID=$!
sleep 5
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready (${i}s)"
        break
    fi
    sleep 1
done

echo "Pulling gemma4..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama pull gemma4 2>&1 | tail -3

OUT=$REPO/results/framework/v3_0_35_quality_eval

python -m weed_optimizer_framework.tools.data_quality_eval \
    --cwd12 downloads/cottonweeddet12 \
    --out "$OUT" \
    --sample-n 1977 \
    --owlv2-conf 0.30 \
    --owlv2-prompt "weed" \
    --tests "T1,T2,T3,T4"

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

# Cleanup Ollama
kill $OLLAMA_PID 2>/dev/null

SUMMARY=$OUT/v3_0_35_quality_eval_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
