#!/bin/bash
#SBATCH --job-name=v3_0_38_seed
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/framework/v3_0_38_rfdetr_seed%a_%j.out

# v3.0.38 — RF-DETR Large cwd12-only seed-variance re-test.
#
# WHY: architecture lever is exhausted at pyco mAP50-95 ~0.895:
#   v3.0.31 RFDETRLarge 60ep = 0.8949
#   v3.0.34 X2 RFDETRLarge 100ep = 0.8953
# Gap to the 0.90 goal is only -0.0047. Both prior runs used the RF-DETR
# default seed, so we have N=1 effective sample. This run adds 2 more
# seeds → 4 total data points → honest mean ± std of the ceiling.
# If a seed legitimately crosses 0.90, that is a real (if lucky) result;
# if all cluster around 0.895, it confirms the architecture ceiling and
# the 0.90 path must come from data quality (v3.0.38 object-level curator).
#
# Cheap + no new merge code: identical to v3.0.31 except --seed.
# Submit as an array: sbatch --array=1-2 run_v3_0_38_rfdetr_seed.sh
# (each array task picks SEED from the value below).

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

# Seed = 100 + array index → seeds 101, 102 (distinct from library default)
SEED=$((100 + ${SLURM_ARRAY_TASK_ID:-1}))

echo "=== v3.0.38 RFDETRLarge cwd12-only seed re-test ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-1}  SEED=$SEED"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/mega_iterv3_0_38_rfdetr_large_seed${SEED}

# Same config as v3.0.31 (the 0.8949 run): Large @704, 60ep, batch 2,
# grad_accum 8. resolution 704 — Large checkpoint's pretrained pos-embed size.
python -m weed_optimizer_framework.tools.train_rfdetr \
    --model large \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --epochs 60 \
    --batch 2 \
    --grad-accum 8 \
    --resolution 704 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED"

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

SUMMARY=$OUT/v3_0_29_rfdetr_pycoco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== v3.0.38 seed=$SEED PYCO RESULT ==="
    cat "$SUMMARY"
fi
