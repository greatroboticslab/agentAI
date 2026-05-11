#!/bin/bash
#SBATCH --job-name=v3030_jD
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_30_jobD_%j.out

# v3.0.30 Job-D CONTINUOUS — self-chaining autonomous harvest.
#
# Why this matters (user 2026-05-11):
#   "我们的目的就是 一个可以自动收集数据集的程序 不断的跑 可能跑几天效果不大
#    但是 几个月 一年呢？" — vision is months-to-years of continuous discovery.
#
# Differences vs v3.0.26 jobD:
#   1. AT END: sbatch --dependency=afterany:$SLURM_JOB_ID resubmits itself,
#      creating an infinite chain (limited only by SU budget / kill switch).
#   2. KILL SWITCH: if `.stop_jobd` file exists in repo root, do NOT chain.
#   3. IDLE GUARD: if the last 20 harvest iterations found ZERO new slugs,
#      do NOT chain (Brain exhausted; need user intervention).
#   4. DASHBOARD HOOK: at end of each run, regenerate the public dashboard
#      and git push it (see tools/dashboard_generator.py).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-KGAT_67eb9458d9e565587c47c967c5249584}
export KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-/ocean/projects/cis240145p/byler/kagglehub_cache}
mkdir -p "$KAGGLEHUB_CACHE" 2>/dev/null

echo "=== v3.0.30 Job-D CONTINUOUS — Brain harvest + OWLv2 autolabel ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Check kill switch BEFORE doing any work
if [ -f "$REPO/.stop_jobd" ]; then
    echo "[chain] .stop_jobd present — exit, do NOT chain"
    exit 0
fi

# Snapshot registry size for idle-guard
PRE_REG_SIZE=$(python -c "import json; r=json.load(open('results/framework/dataset_registry.json')); print(len(r.get('datasets',{})))" 2>/dev/null || echo "0")
echo "[idle-guard] registry size at start: $PRE_REG_SIZE slugs"

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

# Harvest + autolabel loop
python - <<'PYEOF'
import logging, time, json, os, sys
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
log = logging.getLogger("jobD")

from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery
from weed_optimizer_framework.tools.autolabel import autolabel_dataset
from weed_optimizer_framework.tools import registry_lock

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
REGISTRY_PATH = f"{REPO}/results/framework/dataset_registry.json"

try:
    from weed_optimizer_framework.tools.dataset_discovery import harvest_new_datasets
    HARVEST_FN = harvest_new_datasets
except Exception:
    HARVEST_FN = None

# Track per-iteration findings for idle-guard
new_slug_history = []  # list[int] — new slugs found per iteration
WALL_BUDGET_SEC = 46 * 3600  # leave 2h for dashboard + chain
start = time.time()

ITER = 0
while time.time() - start < WALL_BUDGET_SEC:
    ITER += 1
    elapsed_h = (time.time() - start) / 3600.0
    log.info(f"[Job-D] iteration {ITER}, elapsed {elapsed_h:.1f}h")

    # Snapshot registry size pre-iteration
    disc = DatasetDiscovery()
    pre_size = len(disc.registry.get("datasets", {}))

    # Brain harvest
    if HARVEST_FN is not None:
        try:
            HARVEST_FN()
        except Exception as e:
            log.warning(f"[Job-D] harvest failed: {e}")
    else:
        log.warning("[Job-D] harvest_new_datasets not available")

    # Recount post-harvest
    disc2 = DatasetDiscovery()
    post_size = len(disc2.registry.get("datasets", {}))
    new_this_iter = max(0, post_size - pre_size)
    new_slug_history.append(new_this_iter)
    log.info(f"[Job-D] this iter found {new_this_iter} new slugs (registry now {post_size})")

    # Autolabel any "needs_autolabel" entries
    needs = [s for s, info in disc2.registry.get("datasets", {}).items()
             if isinstance(info, dict) and info.get("status") == "needs_autolabel"]
    log.info(f"[Job-D] {len(needs)} datasets need autolabel")
    for slug in needs[:5]:  # per-iteration cap: 5 datasets
        try:
            autolabel_dataset(slug, max_imgs=20000)
        except Exception as e:
            log.warning(f"[Job-D] autolabel {slug} failed: {e}")
        if time.time() - start > WALL_BUDGET_SEC:
            break

    log.info(f"[Job-D] sleep 5min before next iteration...")
    time.sleep(300)

# Idle guard: if last 20 iters all had 0 new slugs, signal exhaustion
recent = new_slug_history[-20:] if len(new_slug_history) >= 20 else new_slug_history
exhausted = len(recent) >= 20 and sum(recent) == 0
exhausted_path = f"{REPO}/.jobd_exhausted"
if exhausted:
    log.warning(f"[idle-guard] last 20 iters all 0 new slugs — Brain exhausted")
    open(exhausted_path, "w").write("Brain exhausted at " + time.strftime("%Y-%m-%d %H:%M:%S"))
else:
    # clear stale exhausted flag
    if os.path.exists(exhausted_path):
        os.remove(exhausted_path)

# Write per-run summary for dashboard
summary = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "?"),
    "iters": ITER,
    "new_slugs_per_iter": new_slug_history,
    "total_new_slugs": sum(new_slug_history),
    "exhausted": exhausted,
    "elapsed_h": (time.time() - start) / 3600.0,
}
os.makedirs(f"{REPO}/results/framework/jobd_runs", exist_ok=True)
sum_path = f"{REPO}/results/framework/jobd_runs/run_{os.environ.get('SLURM_JOB_ID','?')}.json"
with open(sum_path, "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"[Job-D] wrote summary {sum_path}")
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Regenerate dashboard + push to GitHub Pages
echo "[dashboard] regenerating..."
python -m weed_optimizer_framework.tools.dashboard_generator \
    --registry results/framework/dataset_registry.json \
    --out-dir docs/dashboard \
    || echo "[dashboard] regen failed (non-fatal)"

# Self-chain if not killed and not exhausted
if [ -f "$REPO/.stop_jobd" ]; then
    echo "[chain] .stop_jobd present — exit, do NOT chain"
elif [ -f "$REPO/.jobd_exhausted" ]; then
    echo "[chain] Brain exhausted (last 20 iters 0 new) — exit, do NOT chain. "
    echo "[chain] To resume: touch $REPO/.jobd_force_resume && rm $REPO/.jobd_exhausted"
elif [ -f "$REPO/.jobd_force_resume" ]; then
    rm "$REPO/.jobd_force_resume"
    echo "[chain] force_resume found — chaining anyway"
    sbatch --dependency=afterany:$SLURM_JOB_ID "$REPO/run_v3_0_30_jobd_continuous.sh" || \
        echo "[chain] sbatch failed"
else
    echo "[chain] scheduling next Job-D via afterany:$SLURM_JOB_ID"
    sbatch --dependency=afterany:$SLURM_JOB_ID "$REPO/run_v3_0_30_jobd_continuous.sh" || \
        echo "[chain] sbatch failed (this run is the last)"
fi
