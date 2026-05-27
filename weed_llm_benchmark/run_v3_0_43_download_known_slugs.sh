#!/bin/bash
#SBATCH --job-name=dl_known
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/framework/v3_0_43_dl_known_%j.out

# v3.0.43.9 — Fetch every slug currently status=known (discovered but not on
# disk) from HuggingFace to /ocean/datasets/<slug>/. Triggerable from /control.
#
# Cluster-as-server philosophy:
#   - All data lives on /ocean. User's browser never sees the bytes; just
#     reads dashboard URLs which serve from /ocean.
#   - Brain discovers via HF API → marks status=known. This script does
#     the actual fetch → marks status=downloaded.
#   - After this run, the dashboard /classes will auto-show the new
#     slugs' images (thumbnail cache prewarms on next dashboard restart).

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
export CLASS_TOPIC_OVERRIDES_FILE="$REPO/results/framework/class_topic_overrides.json"

echo "=== v3.0.43.9 download known slugs (triggered from /control) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "Disk free /ocean: $(df -h /ocean | tail -1)"

# Start ollama so harvest's auto-topic-classify can use LLM for new species.
# Graceful degrade to keyword-only if ollama isn't reachable in 30s.
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 ; then
    echo "[ollama] starting…"
    /ocean/projects/cis240145p/byler/ollama/bin/ollama serve \
        >/tmp/ollama_$SLURM_JOB_ID.log 2>&1 &
    OLLAMA_PID=$!
    for i in $(seq 1 30); do
        if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 ; then
            echo "[ollama] ready (${i}s)"
            break
        fi
        sleep 1
    done
fi

python -u - <<'PYEOF' 2>&1
import os, sys, time
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery

print(f"[dl_known] start {time.strftime('%H:%M:%S')}", flush=True)
d = DatasetDiscovery()

# Pick HF slugs marked status=known and not actually on disk
candidates = []
for slug, info in d.registry["datasets"].items():
    if info.get("status") != "known":
        continue
    lp = info.get("local_path") or ""
    if lp and os.path.isdir(lp):
        # already there — skip; status will get fixed on next save
        continue
    if info.get("source") != "huggingface":
        # Kaggle / GitHub / Roboflow need their own harvest paths
        continue
    if not info.get("hf_id"):
        continue
    candidates.append((slug, info))

print(f"[dl_known] {len(candidates)} HF slugs to download:", flush=True)
for slug, info in candidates:
    print(f"  - {slug}  hf_id={info.get('hf_id')}  expected_imgs={info.get('images')}",
          flush=True)

if not candidates:
    print("[dl_known] nothing to do.", flush=True)
    sys.exit(0)

# Per-slug cap. Mass weedsense-class slugs (120K imgs) are heavy on disk +
# HF API. 20K is enough to populate the dashboard preview + train.
PER_SLUG_CAP = 20000

print(f"\n[dl_known] downloading (cap {PER_SLUG_CAP} imgs/slug)…", flush=True)
n_ok = 0; n_fail = 0; n_skipped = 0
for i, (slug, info) in enumerate(candidates, 1):
    t0 = time.time()
    print(f"\n[dl_known] [{i}/{len(candidates)}] === {slug} === "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    try:
        local_path, result = d.download_dataset(slug, max_images=PER_SLUG_CAP)
        elapsed = time.time() - t0
        status = result.get("status", "?")
        n_imgs = result.get("images", 0)
        n_labels = result.get("labeled", 0)
        n_classes = result.get("classes", 0)
        print(f"  → {status}: {n_imgs} imgs, {n_labels} labeled, "
              f"{n_classes} classes ({elapsed:.1f}s)", flush=True)
        if status == "downloaded":
            n_ok += 1
        elif status == "already_downloaded":
            n_skipped += 1
        else:
            n_fail += 1
            print(f"  detail: {result}", flush=True)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
        n_fail += 1
    # Sleep briefly between to be nice to HF API
    time.sleep(2)

print(f"\n[dl_known] DONE", flush=True)
print(f"  ok:      {n_ok}", flush=True)
print(f"  skipped: {n_skipped}", flush=True)
print(f"  failed:  {n_fail}", flush=True)
print(f"  finished {time.strftime('%H:%M:%S')}", flush=True)
PYEOF

echo "=== dl_known done $(date) ==="
echo "Now: refresh /control or POST /api/refresh_registry, then /classes will"
echo "auto-show the new slugs' images on next page load."
