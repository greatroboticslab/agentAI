#!/bin/bash
#SBATCH --job-name=v3027_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_27_eval_%j.out

# v3.0.27 — Ensemble + multi-scale TTA evaluation, target ≥ 0.90 mAP50-95.
# Inference-only: pulls 2-3 best.pt from prior v3.0.* phases, evaluates each
# with and without TTA, eventually combines via WBF.
#
# Walltime 2h: each model eval ~10min, 3 models × 2 (plain+TTA) = 1h, plus
# WBF post-processing.

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

# Try to install ensemble_boxes if missing (for WBF). Best-effort, eval still
# runs without it via per-model fallback.
pip install ensemble_boxes 2>&1 | tail -1

echo "=== v3.0.27 ensemble + TTA eval ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python eval_v3_0_27_ensemble.py

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
