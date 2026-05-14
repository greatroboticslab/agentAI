#!/bin/bash
#SBATCH --job-name=v3029_rfd_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/v3_0_29_rfdetr_eval_%j.out

# v3.0.29.5 — pyco eval ONLY on the saved RF-DETR best checkpoint.
# No retraining. Just inference on cwd12 holdout (1977 imgs) +
# pycocotools mAP50-95 — the canonical research metric.
#
# Why a separate job: training run 40803244 finished successfully (saving
# checkpoint_best_total.pth, internal EMA val=0.8994) but final pyco eval
# crashed with "'Detections' object has no attribute 'metadata'" because
# supervision 0.6.0 (pinned by groundingdino-py) lacks .metadata that
# rfdetr 1.6+ writes. Fix in v3.0.30.6: train_rfdetr.py monkey-patches
# sv.Detections at import time. This job re-runs only the eval.

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

echo "=== v3.0.29.5 RF-DETR pyco-eval ONLY ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OUT=$REPO/results/framework/mega_iterv3_0_29_rfdetr
WEIGHTS=$OUT/run/checkpoint_best_total.pth
ls -la "$WEIGHTS" || { echo "weights missing"; exit 1; }

python -m weed_optimizer_framework.tools.train_rfdetr \
    --eval-only \
    --out "$OUT" \
    --cwd12 downloads/cottonweeddet12 \
    --resolution 576 \
    --weights "$WEIGHTS"

EXIT=$?
echo "=== Done (exit=$EXIT) ==="
echo "Date: $(date)"

# Surface result file path for dashboard pickup
SUMMARY=$OUT/v3_0_29_rfdetr_pycoco_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== PYCOCO RESULT (canonical) ==="
    cat "$SUMMARY"
fi
