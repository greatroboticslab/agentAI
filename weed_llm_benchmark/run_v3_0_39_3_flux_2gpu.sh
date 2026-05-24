#!/bin/bash
#SBATCH --job-name=v3_0_39_3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=results/framework/v3_0_39_3_flux_2gpu_%j.out

# v3.0.39.3 — FLUX.1-Fill on 2x V100-32GB, no CPU offload.
#
# Why: v3.0.39 (1 GPU + CPU offload) ran at 5 min/image — 600 images
# would take >50h, infeasible. Single-image quality was confirmed good
# (see /Users/.../Desktop/fluxsynth_000005.jpg). Speed problem only.
#
# 2 V100-32GB gives 64GB combined VRAM. FLUX is ~24GB in bf16; with the
# default diffusers device_map=balanced it splits transformer/T5 across
# the two cards, no offload needed, ~10s per image expected.
# 600 images -> ~100 min generation + 15 min model load = ~2h total.
#
# Set environment hint for diffusers to use both GPUs cleanly.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
export FORCE_FLUX_NO_OFFLOAD=1   # synth_diffusion._load_flux respects this

echo "=== v3.0.39.3 FLUX.1-Fill on 2x V100 ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 600 images, 28 steps, weak-class biased
python -m weed_optimizer_framework.tools.synth_diffusion generate \
    --n 600 --steps 28 --guidance 30.0
EXIT=$?
echo "generate exit=$EXIT"

echo ""; echo "=== v3.0.39.3 DONE ($(date)) ==="
echo "Review via dashboard:"
echo "  https://<dashboard-tunnel>/synth/flux"
