#!/bin/bash
#SBATCH --job-name=v3028_pre
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_28_pretrain_%j.out

# v3.0.28 CLEAN PRETRAIN — full pipeline with the v3.0.27 leak fixed.
#
# Re-merges via the patched mega_trainer.py (stem-level holdout filter active)
# so the 2,313 cwd12 holdout copies that were in cottonweed_sp8 / cottonweed_holdout
# slugs no longer enter training. Then trains yolo26x from COCO weights on the
# cleaned ~139K corpus. The resulting best.pt is the legitimate pretrained model
# for the v3.0.28 finetune.
#
# fresh_start=True is critical — we cannot continue from any v3.0.2[4-7] checkpoint
# because all of those saw the holdout during their own pretrain.
#
# include_autolabel=True keeps the full ~244K registry corpus available; after
# stem filter + dHash dedup we end up around 139K unique training images.

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

echo "=== v3.0.28 CLEAN PRETRAIN — yolo26x on de-leaked merged corpus ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import os, sys, json, glob
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega, _load_holdout_stems

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"

# Sanity check the patch is active
hs = _load_holdout_stems()
print(f"v3.0.28 patch sanity: holdout stems loaded = {len(hs)}")
assert len(hs) >= 1900, f"holdout stem filter looks broken (got {len(hs)})"

strategy = {
    "base_model": "yolo26x.pt",   # COCO weights, never seen cwd12 holdout
    "fresh_start": True,           # MANDATORY: do NOT continue from contaminated checkpoints
    "epochs": 100,
    "batch_size": 5,               # imgsz=1024 + yolo26x → 5 fits 32GB V100
    "imgsz": 1024,
    "lr": 0.001,
    "patience": 30,
    "workers": 4,
    "include_autolabel": True,
    "val_dataset_root": CWD12_ROOT,  # honest val = cwd12 test+valid
}

print("Strategy:", json.dumps(strategy, indent=2))
best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_28_pretrain")
print("\n=== v3.0.28 CLEAN PRETRAIN COMPLETE ===")
print(json.dumps(summary, indent=2, default=str))

# Stash the best.pt path for the FT job to find
with open(f"{REPO}/results/framework/v3_0_28_pretrain_best.txt", "w") as f:
    f.write(best_pt + "\n")
print(f"best.pt path written to: results/framework/v3_0_28_pretrain_best.txt")
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Auto-chain: if pretrain succeeded, queue the FT job afterok
if [ $EXIT_CODE -eq 0 ]; then
    echo "[chain] Submitting FT job afterok:$SLURM_JOB_ID"
    sbatch --dependency=afterok:$SLURM_JOB_ID run_v3_0_28_clean_ft.sh || \
        echo "[chain] FT submit failed — submit manually after this completes"
fi
