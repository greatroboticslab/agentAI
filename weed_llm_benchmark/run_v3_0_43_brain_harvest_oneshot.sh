#!/bin/bash
#SBATCH --job-name=brain_hrv_1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
echo "[config] ROUND_BUMP=${ROUND_BUMP:-0}  AUTO_SYNC=${AUTO_SYNC:-0}  BRAIN_STRICT_MIN_LABELS=${BRAIN_STRICT_MIN_LABELS:-50}"

# v3.0.77: optional pre-step — bump round + create RF project. Used by
# the harvest_full_round_e2e button so user gets one-click new round.
if [ "${ROUND_BUMP:-0}" = "1" ]; then
    echo "[round-bump] starting new harvest round + creating new RF project..."
    if [ -f /jet/home/byler/.roboflow_key ]; then
        export ROBOFLOW_API_KEY=$(cat /jet/home/byler/.roboflow_key)
    fi
    python -m weed_optimizer_framework.tools.rounds start-new 2>&1 | tail -10
    echo "[round-bump] done. brain_harvest will tag new slugs with the bumped round."
fi

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

# v3.0.68: read configurable knobs from env (set by dashboard POST body).
# BRAIN_STRICT=1     → reject downloads with <100 labels or 0 classes
# BRAIN_MAX_NEW=N    → cap new datasets per round (default 5)
# BRAIN_MAX_IMGS=N   → cap images per dataset (default 5000)
echo "[config] BRAIN_STRICT=${BRAIN_STRICT:-0}"
echo "[config] BRAIN_MAX_NEW=${BRAIN_MAX_NEW:-5}"
echo "[config] BRAIN_MAX_IMGS=${BRAIN_MAX_IMGS:-5000}"

# Run a single harvest_new_datasets round
python -u - <<'PYEOF' 2>&1 | tee -a $REPO/results/framework/v3_0_43_brain_harvest_oneshot.log
import os, sys, time, json
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery

print(f"[harvest] start {time.strftime('%H:%M:%S')}")
d = DatasetDiscovery()
before = len(d.registry["datasets"])
print(f"[harvest] before: {before} slugs in registry")

# v3.0.68: env-driven knobs (env var > kwarg default)
max_new = int(os.environ.get("BRAIN_MAX_NEW", "5"))
max_imgs = int(os.environ.get("BRAIN_MAX_IMGS", "5000"))
strict = bool(int(os.environ.get("BRAIN_STRICT", "0")))
print(f"[harvest] config max_new={max_new} max_imgs={max_imgs} strict={strict}")

result = d.harvest_new_datasets(
    max_new=max_new,
    confirm_schema=True,
    max_images_per_ds=max_imgs,
    strict_topic=strict,
)

d._load_registry()
after = len(d.registry["datasets"])
print(f"[harvest] after:  {after} slugs (+{after - before})")
print(f"[harvest] return: {json.dumps(result, default=str)[:600]}")
print(f"[harvest] done {time.strftime('%H:%M:%S')}")
PYEOF

echo "=== Brain harvest oneshot done $(date) ==="

# v3.0.77: optional post-step — auto-sync newly downloaded slugs to RF.
# Used by harvest_full_round_e2e button so user gets data into v{N}
# project without another click.
if [ "${AUTO_SYNC:-0}" = "1" ]; then
    echo
    echo "=== [auto-sync] uploading new slugs to per-round Roboflow project ==="
    if [ -f /jet/home/byler/.roboflow_key ]; then
        export ROBOFLOW_API_KEY=$(cat /jet/home/byler/.roboflow_key)
    fi
    python -m weed_optimizer_framework.tools.roboflow_sync sync-newest-slugs \
        --project weed-crop-agent-dataset \
        --folder weed_crop_agent_dataset \
        --cap-per-slug 100 2>&1 | tail -40
    echo "=== [auto-sync] done $(date) ==="
fi
