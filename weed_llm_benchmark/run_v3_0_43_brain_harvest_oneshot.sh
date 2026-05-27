#!/bin/bash
#SBATCH --job-name=brain_hrv_1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/v3_0_43_brain_harvest_%j.out

# v3.0.43.4 — single-round Brain harvest, triggerable from /control UI.
# Uses harvest_new_datasets() (real method, not the broken .search() path).
# Auto-classifies new class_names via topic_classifier as part of harvest.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
export CLASS_TOPIC_OVERRIDES_FILE="$REPO/results/framework/class_topic_overrides.json"

echo "=== v3.0.43.4 Brain harvest (1 round, triggered from /control) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"

# Start ollama if not already up (needed for topic classify on new classes)
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 ; then
    echo "[ollama] starting…"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama serve \
        >/tmp/ollama_$SLURM_JOB_ID.log 2>&1 &
    OLLAMA_PID=$!
    for i in $(seq 1 60); do
        curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

# Run a single harvest_new_datasets round
python -u - <<'PYEOF' 2>&1 | tee -a $REPO/results/framework/v3_0_43_brain_harvest_oneshot.log
import os, sys, time, json
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery

print(f"[harvest] start {time.strftime('%H:%M:%S')}")
d = DatasetDiscovery()
before = len(d.registry["datasets"])
print(f"[harvest] before: {before} slugs in registry")

result = d.harvest_new_datasets(
    max_new=5,
    confirm_schema=True,
    max_images_per_ds=5000,
)

d._load_registry()
after = len(d.registry["datasets"])
print(f"[harvest] after:  {after} slugs (+{after - before})")
print(f"[harvest] return: {json.dumps(result, default=str)[:600]}")
print(f"[harvest] done {time.strftime('%H:%M:%S')}")
PYEOF

echo "=== Brain harvest oneshot done $(date) ==="
