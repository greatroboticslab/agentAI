#!/bin/bash
#SBATCH --job-name=flux_12cls
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_41_flux_12class_%a_%j.out

# v3.0.41 — balanced 12-class vanilla FLUX baseline, one species per array task.
#
# Why: per-class /audit comparison currently shows real bank crops for all 12
# classes but FLUX outputs only for ~4 classes (the others got 0 due to the
# weak-class-bias config not kicking in, plus job 40970083 being cancelled
# after 42 images). User wants a fair per-class side-by-side, so we need at
# least N FLUX outputs PER class.
#
# Submit:  sbatch --array=1-12 run_v3_0_41_flux_12class_baseline.sh
# Each array task generates N=5 images of one species (CPU-offload path,
# ~3-5 min per 768² image → ~15-25 min per task). Total wall clock if all
# parallel: ~25 min. If serialised on one GPU: ~5 h.
#
# Output filenames are prefixed with species (fluxsynth_Goosegrass_000000.jpg
# etc.) so array tasks don't collide.

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

# Map array index to species
CANONICAL_12=(Carpetweeds Crabgrass Eclipta Goosegrass Morningglory Nutsedge \
              PalmerAmaranth PricklySida Purslane Ragweed Sicklepod SpottedSpurge)
IDX=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
if [ "$IDX" -lt 0 ] || [ "$IDX" -ge 12 ]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range 1..12"
    exit 2
fi
CLS=${CANONICAL_12[$IDX]}

echo "=== v3.0.41 FLUX 12-class baseline ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  ARRAY_IDX=${SLURM_ARRAY_TASK_ID}  CLASS=$CLS"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Each task generates 5 images of its species. Default CPU-offload path.
python -m weed_optimizer_framework.tools.synth_diffusion generate \
    --n 5 \
    --class-name "$CLS" \
    --steps 28 \
    --guidance 30.0 \
    --seed $((100 + IDX))

EXIT=$?
echo "=== Done $CLS (exit=$EXIT) ==="
echo "Date: $(date)"
ls -la $REPO/results/framework/synth_diffusion/images/fluxsynth_${CLS}_*.jpg 2>/dev/null | head -10
