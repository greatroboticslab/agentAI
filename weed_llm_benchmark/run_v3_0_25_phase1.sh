#!/bin/bash
#SBATCH --job-name=v3_0_25_p1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_25_p1_%j.out

# v3.0.25 Phase 1: validate the canonical-12-class fix end-to-end.
#
# What's different from v3.0.24:
#   - Canonical 12-class enforcement (CWD12_ORIG_TO_CANON for cottonweed_*).
#     Fixes the v3.0.24 "Eclipta/Goosegrass/Morningglory/Nutsedge stay at 0
#     mAP" bug, where cottonweed_sp8 and cottonweed_holdout share a physical
#     path and sp8's class_map was applied to all 4 holdout species' images.
#   - Strict class remap: drop bbox if src_cls is unrecognized (no silent
#     passthrough).
#   - NEVER_TRAIN protection: cottonweeddet12, weedsense, francesco never
#     reach training. They are the immutable evaluation gold standard.
#   - val OVERRIDE = cottonweeddet12 holdout (test+valid combined, 1977 imgs
#     hand-labeled) → ultralytics' early-stop patience operates on the
#     honest paper-grade signal, not on a 10% slice of the (possibly noisy)
#     training corpus.
#   - nc fixed at 100 (12 weed + 88 aux slots) so future autolabel data can
#     be added in v3.0.25 Phase 2 without head expansion.
#
# Phase 1 still skips yolo_autolabel data (include_autolabel=False) to
# isolate the class-mapping fix. Phase 2 re-introduces 175K autolabel data
# with per-dataset aux class slot via slug hash, and adds the parallel
# Job-D / hot-reload architecture.

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

echo "=== v3.0.25 Phase 1 — canonical-12-class fix + cwd12 holdout val ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

strategy = {
    # Architecture
    "include_autolabel": False,        # Phase 1: real bbox only (~10-15K)
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",
    "fresh_start": True,               # don't load v3.0.24 best.pt (different class layout: 12 vs 100)

    # Training schedule (longer than v3.0.24 since data is smaller and class
    # layout changed, model needs to relearn from scratch)
    "epochs": 200,                     # patience handles early stop on cwd12 mAP plateau
    "imgsz": 1024,
    "batch_size": 5,
    "lr": 0.001,
    "patience": 30,                    # 30 epochs no improvement on cwd12 holdout → stop
    "workers": 4,
}

best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_25_p1")
print("\n=== TRAIN COMPLETE ===")
print("best_pt:", best_pt)
import json
print(json.dumps(summary, indent=2, default=str))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Auto-eval against cwd12 holdout (separate JSON, in case the during-training
# val numbers differ from a final clean inference pass).
if [ $EXIT_CODE -eq 0 ]; then
    BEST=$(ls -t results/framework/mega_iterv3_0_25_p1/train*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$BEST" ] && [ -f "$BEST" ]; then
        echo "Running clean cwd12 eval on $BEST"
        sed "s|mega_iter6/train8/weights/best.pt|${BEST#$PWD/}|; s|results/v3_0_23_eval|results/v3_0_25_p1_eval|; s|v3_0_23_eval.json|v3_0_25_p1_eval.json|" eval_v3_0_23.py > eval_v3_0_25_p1.py
        python eval_v3_0_25_p1.py
    fi
fi

echo "=== ALL DONE ==="
echo "Date: $(date)"
