#!/bin/bash
#SBATCH --job-name=semisupA
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/semisup_phaseA_%j.out
#
# Semi-supervised labeler, Phase A: does crop -> frozen embedding -> clustering ->
# small labelled budget -> propagation recover species on data whose labels we hold?
#
# Runs on the cottonweeddet12 TRAIN split only. The sealed test+valid holdout is
# dHash-guarded inside the module and the run fails if a holdout stem appears.
# Output: results/framework/semisup/phaseA/{manifest.csv, emb_*.npz, results.json}
#
# Submit:  sbatch run_semisup_phaseA.sh
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1
export REPO

if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rsync -a --delete \
        "$REPO/weed_llm_benchmark/weed_optimizer_framework/" \
        "$REPO/weed_optimizer_framework/" \
        && echo "[sync] outer package refreshed from the tracked nested copy"
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u -c "import sklearn, torch, transformers, open_clip; print('sklearn', sklearn.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" || exit 1

echo "=== 1/3 crops ==="
python -u -m weed_optimizer_framework.tools.semisup_labeler crops || exit 1
echo "=== 2/3 embed ==="
python -u -m weed_optimizer_framework.tools.semisup_labeler embed || exit 1
echo "=== 3/3 evaluate ==="
python -u -m weed_optimizer_framework.tools.semisup_labeler evaluate || exit 1
echo "=== done ==="
ls -la results/framework/semisup/phaseA/
