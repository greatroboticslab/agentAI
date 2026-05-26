#!/bin/bash
#SBATCH --job-name=brain_hrv
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_41_brain_harvest_%j.out

# v3.0.41 Track B preview — Brain-driven autonomous dataset discovery.
# Uses the existing framework's Brain (Gemma 4 via Ollama) + harvester
# tools to propose new weed/crop dataset queries based on current class
# gaps, then searches Kaggle / HuggingFace and registers candidates.
#
# Read-mostly: doesn't auto-train, doesn't auto-merge. Adds entries to
# dataset_registry.json (append-only). Morning review surfaces what
# Brain found.

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

echo "=== v3.0.41 Brain harvest (overnight, autonomous) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"

# Start ollama if not already running
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 ; then
    echo "[ollama] starting…"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama serve >/tmp/ollama_$SLURM_JOB_ID.log 2>&1 &
    for i in $(seq 1 60); do
        curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

# Run the existing framework's "harvest-only" entry. The framework has an
# orchestrator that can be configured for harvest-without-train.
# If a dedicated harvest-only entry doesn't exist, fall back to the round
# orchestrator with --no-train.
python - <<'PYEOF' 2>&1 | tee -a $REPO/results/framework/v3_0_41_brain_harvest.log
import json, os, sys, time
sys.path.insert(0, ".")
os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
os.environ.setdefault("BRAIN_MODEL", "gemma4:26b")

print(f"[brain_harvest] start {time.strftime('%Y-%m-%d %H:%M:%S')}")
try:
    from weed_optimizer_framework.tools.harvester import (
        DEFAULT_HARVEST_QUERIES, harvest_round,
    )
except Exception as e:
    print(f"[brain_harvest] cannot import harvester: {e}")
    print("[brain_harvest] falling back to direct DatasetDiscovery search")
    from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery
    d = DatasetDiscovery()
    print(f"[brain_harvest] registry currently has {len(d.registry['datasets'])} slugs")
    # Run a few Brain-proposed queries
    queries = [
        "goosegrass eleusine indica field detection bbox",
        "sicklepod cassia obtusifolia annotated dataset",
        "spotted spurge euphorbia maculata detection",
        "eclipta prostrata weed image dataset",
        "broadleaf weeds cotton field bbox yolo",
        "agricultural weed seedling identification dataset",
        "morningglory ipomoea weed annotated images",
        "ragweed ambrosia detection bbox",
    ]
    for q in queries:
        print(f"[brain_harvest] search: {q!r}")
        try:
            for src in ("kaggle", "hf", "github"):
                try:
                    results = d.search(q, source=src, max_results=5)
                    for r in results:
                        print(f"  [{src}] {r}")
                except Exception as ex:
                    print(f"  [{src}] error: {ex}")
            time.sleep(2)  # be polite
        except Exception as ex:
            print(f"  query error: {ex}")
    print(f"[brain_harvest] done {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0)

# If harvester import worked, do 4 rounds (autonomous Brain-driven queries)
for round_i in range(1, 5):
    print(f"\n[brain_harvest] === Round {round_i}/4 ===")
    try:
        harvest_round(max_new=10, round_idx=round_i)
    except Exception as ex:
        print(f"  round error: {ex}")
        time.sleep(5)
        continue
    time.sleep(10)
print(f"[brain_harvest] all done {time.strftime('%Y-%m-%d %H:%M:%S')}")
PYEOF

echo "=== Brain harvest done $(date) ==="
