#!/bin/bash
#SBATCH --job-name=v3_0_25_p2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_25_p2_%j.out

# v3.0.25 Phase 2 — re-introduce 175K autolabel data + class balance.
#
# Diff from Phase 1:
#   - include_autolabel=True (yolo_autolabel datasets enter the merge,
#     routed to deterministic aux class slots [12, 100) via slug hash;
#     they cannot pollute the 12 weed slots).
#   - class-balanced oversampling for weed classes (target_min=500,
#     symlink duplication of images with weak weed classes).
#   - Progressive: starts from Phase 1 best.pt (mega_iterv3_0_25_p1)
#     so Phase 2 doesn't re-learn the canonical 12 from scratch.
#
# Expected: cwd12 mAP50-95 from Phase 1's ~0.55 plateau → 0.65-0.75 by
# either (a) more weed instances via class balancing, (b) better
# representation learning from the 100K+ aux plant data, or both.

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

echo "=== v3.0.25 Phase 2 — autolabel re-enable + class balance ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import logging, os
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
PHASE1_BEST = f"{REPO}/results/framework/mega_iterv3_0_25_p1/train/weights/best.pt"

# Progressive: start from Phase 1 best.pt if it exists; otherwise fresh.
fresh_start = not os.path.exists(PHASE1_BEST)
print(f"Phase1 best.pt exists: {os.path.exists(PHASE1_BEST)}")
print(f"fresh_start: {fresh_start}")

strategy = {
    "include_autolabel": True,         # P2: enable 100K+ aux data
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",
    "fresh_start": fresh_start,
    "base_model": PHASE1_BEST if not fresh_start else None,

    "epochs": 200,                     # patience handles early stop
    "imgsz": 1024,
    "batch_size": 5,
    "lr": 0.0005,                      # lower than P1 (0.001) since starting from trained weights
    "patience": 25,
    "workers": 4,
}

best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_25_p2")
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
    BEST=$(ls -t results/framework/mega_iterv3_0_25_p2/train*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$BEST" ] && [ -f "$BEST" ]; then
        echo "Running clean cwd12 eval on $BEST"
        sed "s|mega_iter6/train8/weights/best.pt|${BEST#$PWD/}|; s|results/v3_0_23_eval|results/v3_0_25_p2_eval|; s|v3_0_23_eval.json|v3_0_25_p2_eval.json|" eval_v3_0_23.py > eval_v3_0_25_p2.py
        python eval_v3_0_25_p2.py
    fi
fi

echo "=== ALL DONE ==="
echo "Date: $(date)"
