#!/bin/bash
#SBATCH --job-name=v3_0_38_1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=results/framework/v3_0_38_1_rerun_%j.out

# v3.0.38.1 — rerun the curator + verifier stages AFTER the registry-iteration
# bug fix. Steps 1-3 (cut-paste bank/backgrounds/compose -> 2000 imgs) already
# completed under job 40962044 and are unaffected; we skip them.
#
# Fixed: dinov2_object_curator and dino_label_verifier now iterate
# registry["datasets"] instead of the top-level registry dict (which contained
# bookkeeping keys like "discovered", "total_downloaded"). The previous run
# scored 0 real slugs because of this bug.
#
# Steps 4-9 are non-destructive — overwrite scores + report. Head retrains
# from the existing object bank (no GPU cost beyond ~15 min).

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

echo "=== v3.0.38.1 rerun (curator + verifier, fixed) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""; echo "### STEP 4: build object reference pool ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator build-object-reference
echo "step4 exit=$?"

echo ""; echo "### STEP 5: score every slug at object level ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator score-objects
echo "step5 exit=$?"

echo ""; echo "### STEP 6: object-curator report ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator report

echo ""; echo "### STEP 7: train DINO label-verifier head ###"
python -m weed_optimizer_framework.tools.dino_label_verifier train
echo "step7 exit=$?"

echo ""; echo "### STEP 8: verify every slug's labels ###"
python -m weed_optimizer_framework.tools.dino_label_verifier verify
echo "step8 exit=$?"

echo ""; echo "### STEP 9: label-verifier report ###"
python -m weed_optimizer_framework.tools.dino_label_verifier report

echo ""; echo "=== v3.0.38.1 DONE ($(date)) ==="
