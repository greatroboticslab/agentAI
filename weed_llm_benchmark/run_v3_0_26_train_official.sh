#!/bin/bash
#SBATCH --job-name=v3026_T_official
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_26_train_%j.out

# v3.0.26 OFFICIAL PARALLEL TEST — Job-T (training).
# Concurrent with run_v3_0_26_jobd_official.sh (Job-D). Both 48h walltime.
#
# Inherits:
#   - P2's best.pt as base_model (progressive transfer-learning continuation)
#   - All datasets in registry at start time (incl. anything Job-D added
#     during the prior 24h pre-test run)
#
# What's new vs P2:
#   - 48h dedicated training (v3.0.25 P1 had 48h but P2 was capped to 24h)
#   - Job-D writing new data to registry while we train (REQ-1 parallel)
#   - Class balance + autolabel + canonical class fix all in place from v3.0.25

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

echo "=== v3.0.26 OFFICIAL Job-T (parallel test, 48h) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import logging, os, glob
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

# Find latest best.pt across all v3.0.25 phases
candidates = sorted(
    glob.glob(f"{REPO}/results/framework/mega_iterv3_0_25_p*/train*/weights/best.pt"),
    key=os.path.getmtime, reverse=True
)
PRIOR_BEST = candidates[0] if candidates else None
print(f"Found {len(candidates)} prior best.pt candidates")
if PRIOR_BEST:
    print(f"Using as base: {PRIOR_BEST}")

strategy = {
    "include_autolabel": True,
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",
    "fresh_start": PRIOR_BEST is None,
    "base_model": PRIOR_BEST,

    "epochs": 200,
    "imgsz": 1024,
    "batch_size": 5,
    "lr": 0.0005,
    "patience": 25,
    "workers": 4,
}

best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_26")
print("\n=== TRAIN COMPLETE ===")
print("best_pt:", best_pt)
import json
print(json.dumps(summary, indent=2, default=str))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Auto-eval on cwd12 holdout
if [ $EXIT_CODE -eq 0 ]; then
    BEST=$(ls -t results/framework/mega_iterv3_0_26/train*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$BEST" ] && [ -f "$BEST" ]; then
        echo "Running clean cwd12 eval on $BEST"
        sed "s|mega_iter6/train8/weights/best.pt|${BEST#$PWD/}|; s|results/v3_0_23_eval|results/v3_0_26_eval|; s|v3_0_23_eval.json|v3_0_26_eval.json|" eval_v3_0_23.py > eval_v3_0_26.py
        python eval_v3_0_26.py
    fi
fi
echo "=== ALL DONE ==="
echo "Date: $(date)"
