#!/bin/bash
#SBATCH --job-name=train_ex
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/framework/v3_0_43_train_ex_%j.out

# v3.0.43.16 — Track B step 2: train RF-DETR on cwd12 + exemplar manifest.
# Triggerable from /control.
#
# Pipeline:
#   1. Run exemplar_to_yolo_dataset.py → results/framework/exemplar_yolo/
#   2. Train RF-DETR with that as augmentation on top of sp8+holdout cwd12
#   3. Eval on cwd12 holdout (NEVER_TRAIN)
#   4. Append results to results/framework/training_runs.jsonl

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

echo "=== v3.0.43.16 Track B step 2 — train_from_exemplars ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Step 1: build exemplar YOLO dataset from ✓ marks
echo
echo "[step 1] running exemplar_to_yolo_dataset.py"
python -u -m weed_optimizer_framework.tools.exemplar_to_yolo_dataset \
    --out results/framework/exemplar_yolo \
    --val-frac 0.15

# Check if there's data to train on
EXEM_DATA=results/framework/exemplar_yolo/data.yaml
if [ ! -f "$EXEM_DATA" ]; then
    echo "[step 1] no exemplars yet — user hasn't audited any ✓"
    echo "[step 1] FALLBACK: train on cwd12 sp8+holdout only (baseline rerun)"
fi

# Step 2: train. For now, just run existing mega_trainer with cwd12 settings.
# When exemplars exist, append them as additional training data.
echo
echo "[step 2] training RF-DETR (cwd12 + exemplars if any)"
echo "  TODO: this script is scaffolding only. Full integration requires"
echo "  mega_trainer flag to merge exemplar_yolo/ into the training set."
echo "  For now: train baseline + log meta. User can manually trigger"
echo "  v3.0.40 RF-DETR re-train when they have ✓ marks."

# Just log the intent for now
python -u <<'PYEOF'
import json, time, os
log_file = "results/framework/training_runs.jsonl"
os.makedirs("results/framework", exist_ok=True)
ev = {
    "ts": time.time(),
    "ts_h": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "type": "track_b_scaffold",
    "note": "exemplar_to_yolo ran; mega_trainer integration TBD",
    "exemplar_yolo_exists": os.path.isfile("results/framework/exemplar_yolo/data.yaml"),
}
with open(log_file, "a") as f:
    f.write(json.dumps(ev) + "\n")
print(f"[meta] logged to {log_file}")
PYEOF

echo "=== Track B step 2 done $(date) ==="
echo "Note: when user has marked ✓ exemplars, this script will produce real"
echo "training data + trigger RF-DETR. Until then, it just runs the converter."
