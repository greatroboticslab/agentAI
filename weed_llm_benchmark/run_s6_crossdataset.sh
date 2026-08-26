#!/bin/bash
#SBATCH --job-name=s6_xds
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=02:30:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s6_xds_%j.out
#
# Cross-dataset transfer of the deployed cwd12 model onto the one harvested
# source that cleared the calibrated sample audit (ImageWeeds, audited precision
# 1.0000). Converts BEST_MODEL_CARD limit #1 ("cross-dataset generalisation is
# not established") into a measurement. Leak check (content dHash vs cwd12
# train AND holdout, Hamming <= 6) runs before any evaluation; the in-domain
# reference is computed with the same checkpoints, matcher, conf and imgsz so
# the transfer gap can only come from the data. See crossdataset_eval.py.
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$REPO
nvidia-smi --query-gpu=name --format=csv,noheader
python -u -m weed_optimizer_framework.tools.crossdataset_eval
echo "[xds] DONE"
