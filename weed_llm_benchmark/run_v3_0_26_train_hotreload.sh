#!/bin/bash
#SBATCH --job-name=v3026_T_hot
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_26_train_hot_%j.out

# v3.0.26 OFFICIAL — Job-T2 with TRUE hot-reload.
#
# K=5 epoch mini-phases. Between phases:
#   - Re-snapshot dataset_registry.json (atomic read)
#   - Re-merge data (incl. anything Job-D2 added in the prior 5-epoch window)
#   - Continue training from previous phase's best.pt
#
# Concurrent with Job-D2 (separate SLURM job, shares atomic registry).
# This is the v3.0.26 architecture professor described:
#   "agent finds new datasets → labeled → join the currently-training model"
#
# Hot-reload trigger: every K=5 epochs.
# Plateau exit: 3 consecutive phases with mAP50-95 spread < 0.005 AND no new
#   datasets added since plateau start (i.e., we've fully exploited current data).

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

echo "=== v3.0.26 OFFICIAL Job-T2 (hot-reload, 48h) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import logging, os, glob
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.hot_reload_trainer import hot_reload_train

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

# Find latest best.pt across all v3.0.25 + v3.0.26 phases (hot-reload chains)
candidates = sorted(
    glob.glob(f"{REPO}/results/framework/mega_iterv3_0_*/train*/weights/best.pt"),
    key=os.path.getmtime, reverse=True
)
PRIOR_BEST = candidates[0] if candidates else None
print(f"Found {len(candidates)} prior best.pt candidates")
if PRIOR_BEST:
    print(f"Using as initial base_model: {PRIOR_BEST}")

strategy = {
    # Hot-reload schedule
    "epochs_per_phase": 5,         # K = 5 epochs per phase, then re-merge
    "max_phases": 50,              # safety cap
    "walltime_soft_sec": 47 * 3600,  # exit clean 1h before SLURM cap

    # Data + val
    "include_autolabel": True,
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",
    "base_model": PRIOR_BEST,

    # Training hparams
    "imgsz": 1024,
    "batch_size": 5,
    "lr": 0.0005,
    "workers": 4,
}

best_pt, summary = hot_reload_train(strategy)
print("\n=== HOT-RELOAD COMPLETE ===")
print("final best_pt:", best_pt)
import json
print(json.dumps(summary, indent=2, default=str))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Auto-eval on cwd12 holdout
if [ $EXIT_CODE -eq 0 ]; then
    BEST=$(ls -t results/framework/mega_iterv3_0_26_phase_*/train*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$BEST" ] && [ -f "$BEST" ]; then
        echo "Running clean cwd12 eval on $BEST"
        sed "s|mega_iter6/train8/weights/best.pt|${BEST#$PWD/}|; s|results/v3_0_23_eval|results/v3_0_26_eval|; s|v3_0_23_eval.json|v3_0_26_eval.json|" eval_v3_0_23.py > eval_v3_0_26.py
        python eval_v3_0_26.py
    fi
fi
echo "=== ALL DONE ==="
echo "Date: $(date)"
