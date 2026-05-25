#!/bin/bash
#SBATCH --job-name=v3_41_p0
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/framework/v3_0_41_phase0_bankfix_%j.out

# v3.0.41 Phase 0 — rebuild the cleaned object bank, retrain the linear
# classification head on it, rebuild visual comparison montages.
#
# Diagnosis (2026-05-24): cottonweed_holdout registry IS correctly
# registered with class_names ['Eclipta','Goosegrass','Morningglory',
# 'Nutsedge']. My v3.0.39.4 canonical-mapping fix was ALREADY correct;
# the prior bank rebuild was just interrupted before reaching the
# holdout slug (login-node nohup process killed). Running this as a
# SLURM batch guarantees the build runs to completion.
#
# Steps:
#   1. wipe bank dir + rebuild via synth_cutpaste (canonical mapping)
#   2. retrain linear head on cleaned bank
#   3. rebuild Goosegrass-comparison, per-class-comparison, 6x6 montage
#      for user visual review

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

echo "=== v3.0.41 Phase 0 — bank rebuild + head retrain + compare ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""; echo "### STEP 1: wipe + rebuild bank (canonical-mapped) ###"
rm -rf $REPO/results/framework/synth_cutpaste/object_bank
python -m weed_optimizer_framework.tools.synth_cutpaste bank --max-per-class 400
echo "step1 exit=$?"

echo ""; echo "### bank inventory ###"
for d in $REPO/results/framework/synth_cutpaste/object_bank/*/ ; do
  c=$(ls "$d" 2>/dev/null | wc -l)
  echo "  $(basename $d): $c"
done

echo ""; echo "### STEP 2: retrain linear head on cleaned bank ###"
python -m weed_optimizer_framework.tools.dino_label_verifier train
echo "step2 exit=$?"

echo ""; echo "### head per-class accuracy ###"
python3 -c "
import json
d = json.load(open('$REPO/results/framework/dino_label_verifier/train_meta.json'))
print(f'overall: {d[\"heldout_accuracy\"]:.4f}')
for c, a in d['per_class_heldout_accuracy'].items():
    s = f'{a:.3f}' if a is not None else 'n/a'
    print(f'  {c:18s} {s}')
"

echo ""; echo "### STEP 3: rebuild comparison montages ###"
python3 /tmp/build_compare.py 2>&1 || \
  python3 weed_optimizer_framework/tools/build_compare.py 2>&1 || \
  echo "build_compare.py not found — will pull bank crops directly later"

ls -la $REPO/results/framework/goosegrass_compare_2x6.jpg \
       $REPO/results/framework/perclass_compare_2x12.jpg 2>/dev/null

echo ""; echo "=== Phase 0 DONE ($(date)) ==="
