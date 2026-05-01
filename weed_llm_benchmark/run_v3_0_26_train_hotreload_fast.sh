#!/bin/bash
#SBATCH --job-name=v3026_T_fast
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_26_train_fast_%j.out

# v3.0.26 Job-T2 FAST — imgsz=640, K=5 phases, 47h walltime.
#
# Why faster: with 244K imgs at imgsz=1024 batch=5, each epoch took 6.5h
# → 48h walltime only fits ~7 epochs total (1-2 hot-reload phases).
# At imgsz=640 batch=5, each epoch ~2.5h → ~19 epochs total → 4 hot-reload
# phases. Multi-phase demonstration is the v3.0.26 research contribution.
#
# Trade-off: imgsz=640 caps single-model mAP slightly (~0.02-0.05) vs 1024.
# v3.0.27 will recover via WBF ensemble + multi-scale TTA at inference
# (TTA can use multiple resolutions including 1024 even if trained at 640).
#
# Goal alignment: cwd12 mAP50-95 ≥ 0.90 needs multi-round trajectory.
# 1 phase of training at higher resolution is worth less than 4 phases
# demonstrating hot-reload + class balance + autolabel + parallel data.

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

echo "=== v3.0.26 Job-T2 FAST (imgsz=640, K=5 phase hot-reload, 48h) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import logging, os, glob
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.hot_reload_trainer import hot_reload_train

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

# Find latest best.pt across all v3.0.* phases (incl. the prior aborted P2 + Phase 1)
candidates = sorted(
    glob.glob(f"{REPO}/results/framework/mega_iterv3_0_*/train*/weights/best.pt"),
    key=os.path.getmtime, reverse=True
)
PRIOR_BEST = candidates[0] if candidates else None
print(f"Found {len(candidates)} prior best.pt candidates")
if PRIOR_BEST:
    print(f"Using as base_model: {PRIOR_BEST}")

strategy = {
    "epochs_per_phase": 5,
    "max_phases": 50,
    "walltime_soft_sec": 47 * 3600,

    "include_autolabel": True,
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",
    "base_model": PRIOR_BEST,

    "imgsz": 640,                  # FAST: was 1024, ~2.5h/epoch instead of ~6.5h
    "batch_size": 16,              # bigger batch fits at 640
    "lr": 0.0005,
    "workers": 4,
}

best_pt, summary = hot_reload_train(strategy)
print("\n=== HOT-RELOAD FAST COMPLETE ===")
print("final best_pt:", best_pt)
import json
print(json.dumps(summary, indent=2, default=str))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

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
