#!/bin/bash
#SBATCH --job-name=v3_0_38_cur
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/framework/v3_0_38_curator_%j.out

# v3.0.38-B/C pipeline — object-level data curation, end to end.
#
#   1. synth_cutpaste bank        — crop trusted GT objects -> object_bank/
#   2. synth_cutpaste backgrounds — collect field backgrounds
#   3. synth_cutpaste compose     — paste -> synthetic images + exact GT
#   4. dinov2_object_curator build-object-reference
#                                 — DINOv2-embed trusted + synthetic crops
#   5. dinov2_object_curator score-objects
#                                 — per-bbox similarity score for every slug
#   6. dinov2_object_curator report
#                                 — print calibration; NO auto-flag (review first)
#
# Non-destructive: produces scores for the user to review. Nothing is
# flagged or dropped automatically.

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

echo "=== v3.0.38-B/C object-level curation pipeline ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""; echo "### STEP 1: synth_cutpaste bank ###"
python -m weed_optimizer_framework.tools.synth_cutpaste bank --max-per-class 400
echo "step1 exit=$?"

echo ""; echo "### STEP 2: synth_cutpaste backgrounds ###"
python -m weed_optimizer_framework.tools.synth_cutpaste backgrounds --n 300
echo "step2 exit=$?"

echo ""; echo "### STEP 3: synth_cutpaste compose ###"
python -m weed_optimizer_framework.tools.synth_cutpaste compose --n 2000
echo "step3 exit=$?"

echo ""; echo "### STEP 4: build object reference pool ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator build-object-reference
echo "step4 exit=$?"

echo ""; echo "### STEP 5: score every slug at object level ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator score-objects
echo "step5 exit=$?"

echo ""; echo "### STEP 6: report ###"
python -m weed_optimizer_framework.tools.dinov2_object_curator report

echo ""; echo "=== v3.0.38-B/C DONE ($(date)) ==="
