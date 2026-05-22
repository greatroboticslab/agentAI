#!/bin/bash
#SBATCH --job-name=v3_0_39_flux
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/v3_0_39_synth_diffusion_%j.out

# v3.0.39 — FLUX.1-Fill layout-conditioned synthetic weed generation.
#
# Per Prof. Zhang: synthetic data must also be used FOR TRAINING. This is
# the strong form — FLUX inpainting on real backgrounds, bbox layout we
# control -> realistic images + pixel-exact GT. Output goes to
# results/framework/synth_diffusion/ as a training-augmentation pool.
#
# PREREQUISITE — FLUX.1-Fill-dev is GATED on HuggingFace. Before the first
# run, on a login node:
#   conda activate bench
#   pip install -U diffusers accelerate            # if not already present
#   huggingface-cli login                          # paste an HF token that
#                                                  # has accepted the
#                                                  # black-forest-labs/FLUX.1-Fill-dev license
# Without this the job exits early with a clear message (no GPU wasted).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

echo "=== v3.0.39 FLUX synthetic generation ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 600 images, 28 steps — modest first batch to review visual quality
# before scaling. ~5-15s per FLUX-Fill call; budget covers it in 12h.
python -m weed_optimizer_framework.tools.synth_diffusion generate \
    --n 600 --steps 28 --guidance 30.0
EXIT=$?
echo "generate exit=$EXIT"

echo ""; echo "=== v3.0.39 DONE ($(date)) ==="
echo "Review the visual sample sheet:"
echo "  results/framework/synth_diffusion/sample_montage.jpg"
echo "  results/framework/synth_diffusion/images/"
