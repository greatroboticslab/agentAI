#!/bin/bash
#SBATCH --job-name=v3035_t3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_35_2_t3_only_%j.out

# v3.0.35.2 — T3 only with PROPER negatives.
# Original T3 (job 40884184) found 0 negatives — flagged-garbage slug
# downloads/ paths didn't exist. Now reaches into UNCATEGORIZED slugs
# (commonforms, mytwu, yonder, colo, warp, beehive, plantifydr, plantdoc)
# which ARE downloaded but are NOT weed/crop.
#
# Decision: if Gemma 4 image-relevance accuracy on 100 cwd12 + 50 uncat
# is >= 85% → use it as image-level filter; else drop the Gemma path.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export OLLAMA_HOST=127.0.0.1:11434

echo "=== v3.0.35.2 T3 only ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"; echo "Date: $(date)"

echo "Starting Ollama..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve &
OLLAMA_PID=$!
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready (${i}s)"; break
    fi
    sleep 1
done
echo "Pulling gemma4..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama pull gemma4 2>&1 | tail -3

OUT=$REPO/results/framework/v3_0_35_2_t3_only
python -m weed_optimizer_framework.tools.data_quality_eval \
    --cwd12 downloads/cottonweeddet12 \
    --out "$OUT" \
    --sample-n 100 \
    --tests "T3"

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
kill $OLLAMA_PID 2>/dev/null
SUMMARY=$OUT/v3_0_35_quality_eval_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
