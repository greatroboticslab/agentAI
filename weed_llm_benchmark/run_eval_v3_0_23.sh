#!/bin/bash
#SBATCH --job-name=v3_0_23_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/framework/v3_0_23_eval_%j.out

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.23 clean eval ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python eval_v3_0_23.py

echo "=== Done (exit=$?) ==="
echo "Date: $(date)"
