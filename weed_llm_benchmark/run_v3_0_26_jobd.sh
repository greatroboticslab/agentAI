#!/bin/bash
#SBATCH --job-name=v3026_jobD
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_26_jobD_%j.out

# v3.0.26 Job-D — TRUE PARALLEL ARCHITECTURE.
#
# Runs CONCURRENTLY with Job-T (currently P2 = 40357694, future = v3.0.26
# training). This Job-D ONLY does Brain-driven harvest + OWLv2 autolabel.
# No training. The new datasets it discovers and labels go into the
# shared dataset_registry.json via atomic-rename writes (registry_lock.py),
# so the next Job-T mini-round automatically picks them up.
#
# Why parallel matters (per professor's directive + REQ-1):
#   - Job-T can spend all 48h training instead of waiting on harvest
#   - Job-D can spend all 48h discovering instead of being blocked by training
#   - Combined: data scale + training time both maximized within the
#     same wall-clock window.

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
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-KGAT_67eb9458d9e565587c47c967c5249584}
export KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-/ocean/projects/cis240145p/byler/kagglehub_cache}
mkdir -p "$KAGGLEHUB_CACHE" 2>/dev/null

echo "=== v3.0.26 Job-D — Brain harvest + OWLv2 autolabel (parallel) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Start Ollama for Brain (Gemma 4)
echo "Starting Ollama..."
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve &
OLLAMA_PID=$!
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready (${i}s)"
        break
    fi
    sleep 1
done
/ocean/projects/cis240145p/byler/ollama/bin/ollama pull gemma4 2>&1 | tail -3

# Loop: harvest → autolabel → sleep → repeat. Atomic registry writes mean
# Job-T can read the registry safely at any time.
python - <<'PYEOF'
import logging, time, json, os
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
log = logging.getLogger("jobD")

from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery
from weed_optimizer_framework.tools.autolabel import autolabel_dataset
from weed_optimizer_framework.tools import registry_lock

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
REGISTRY_PATH = f"{REPO}/results/framework/dataset_registry.json"

# Try harvest_new_datasets if present
try:
    from weed_optimizer_framework.tools.dataset_discovery import harvest_new_datasets
    HARVEST_FN = harvest_new_datasets
except Exception:
    HARVEST_FN = None
    log.warning("harvest_new_datasets not importable; Job-D will only autolabel pending")

ITERATIONS = 0
START = time.time()
WALLTIME_SOFT_LIMIT = 47 * 3600  # exit cleanly 1h before walltime cap

while True:
    elapsed = time.time() - START
    if elapsed > WALLTIME_SOFT_LIMIT:
        log.info(f"[Job-D] approaching walltime ({elapsed/3600:.1f}h), exiting cleanly.")
        break

    ITERATIONS += 1
    log.info(f"[Job-D] iteration {ITERATIONS}, elapsed {elapsed/3600:.1f}h")

    # Phase 1: harvest new datasets (if function available)
    if HARVEST_FN is not None:
        try:
            log.info(f"[Job-D] harvesting new datasets...")
            result = HARVEST_FN(max_new=5, max_images_per_ds=20000)
            log.info(f"[Job-D] harvest result: {json.dumps(result, default=str)[:500]}")
        except Exception as e:
            log.exception(f"[Job-D] harvest failed: {e}")

    # Phase 2: autolabel any pending datasets in registry
    disc = DatasetDiscovery()
    pending = [
        slug for slug, info in disc.registry["datasets"].items()
        if info.get("annotation") == "needs_autolabel"
        and info.get("local_path")
        and os.path.isdir(info.get("local_path", ""))
    ]
    log.info(f"[Job-D] {len(pending)} datasets need autolabel")

    for slug in pending[:5]:  # cap per iteration to leave time for Job-T-friendly merging
        try:
            log.info(f"[Job-D] autolabel {slug}")
            stats = autolabel_dataset(
                slug,
                registry_cb={"get": lambda s: disc.registry["datasets"].get(s),
                              "update": lambda s, u: (disc.registry["datasets"].setdefault(s, {}).update(u),
                                                       registry_lock.atomic_write_json(REGISTRY_PATH, disc.registry))},
                conf_threshold=0.25,
                max_images=10000,
                fallback_whole_image=False,  # v3.0.26: drop fallback noise
                batch_size=4,
            )
            log.info(f"[Job-D] {slug} stats: {stats}")
        except Exception as e:
            log.exception(f"[Job-D] {slug} autolabel failed: {e}")

    # Atomic save full registry after each iteration
    try:
        registry_lock.atomic_write_json(REGISTRY_PATH, disc.registry)
    except Exception as e:
        log.exception(f"[Job-D] atomic_write failed: {e}")

    # Sleep between iterations to avoid hammering external APIs.
    log.info(f"[Job-D] sleep 5min before next iteration...")
    time.sleep(300)

log.info(f"[Job-D] done. Total iterations: {ITERATIONS}")
PYEOF

EXIT_CODE=$?
kill $OLLAMA_PID 2>/dev/null
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
