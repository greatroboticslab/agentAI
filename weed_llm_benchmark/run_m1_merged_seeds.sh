#!/bin/bash
#SBATCH --job-name=m1_merged
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/m1_merged_%x_s%a_%j.out
#
# M1 — honest re-measurement of the MERGED-CORPUS points under the sealed
# holdout guard (SCIENCE_AUDIT.md §2, SUPERWEED_PLAN.md S0).
#
# Why this exists: the cumulative/scale numbers (0.593 @244K, 0.576) were
# measured before the v3.1.0 content-level holdout guard, and each was a single
# run. They are the two ends of the quality-vs-scale curve — the campaign's
# signature claim — so they must be re-established with (a) the guard active and
# its `skipped_holdout_hash` counter reported, and (b) multiple seeds so the
# result is a mean +/- std rather than one unreproducible number.
#
# Submit (one array per tier):
#   sbatch --array=1-3 --job-name=m1raw  --export=ALL,TIER=raw      run_m1_merged_seeds.sh
#   sbatch --array=1-3 --job-name=m1cur  --export=ALL,TIER=curated  run_m1_merged_seeds.sh
#
# TIER=raw      -> include_autolabel=True, no DINO gate   (the 244K noisy point)
# TIER=curated  -> include_autolabel=True + DINO quality gate (the clean point)
#
# Each task writes results/framework/m1_<tier>_seed<N>.json with the merge stats
# (incl. holdout blocks) and the holdout mAP — the artifacts make_figures.py reads.
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

TIER="${TIER:-curated}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
SEEDS=(0 101 102 103)                 # index 1..3 -> 101,102,103
SEED="${SEEDS[$SEED_INDEX]}"
ITER="m1_${TIER}_s${SEED}"

echo "=== M1 $TIER seed=$SEED job=${SLURM_JOB_ID:-none} $(date) ==="

# The cluster keeps two copies of the package (repo-nested + outer import path);
# refresh the outer one from the tracked nested copy so the job runs the code we
# committed, not a stale hand-synced copy.
if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rsync -a --delete \
        "$REPO/weed_llm_benchmark/weed_optimizer_framework/" \
        "$REPO/weed_optimizer_framework/" \
        && echo "[sync] outer package refreshed from the tracked nested copy"
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<PYEOF
import json, os, time, traceback
from pathlib import Path

REPO = "$REPO"
TIER = "$TIER"
SEED = int("$SEED")
ITER = "$ITER"
os.chdir(REPO)

from weed_optimizer_framework.config import Config
from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

CWD12 = os.path.join(REPO, "downloads", "cottonweeddet12")
assert Path(CWD12).is_dir(), "cwd12 holdout root missing: " + CWD12

strategy = {
    "epochs": 60,
    "imgsz": 640,             # matches the historical scale runs being re-measured
    "batch_size": -1,
    "lr": 0.001,
    "patience": 20,
    "workers": 4,
    "seed": SEED,             # v3.22.3 passthrough -> ultralytics
    "deterministic": True,
    "include_autolabel": True,
    # honest holdout: val is the hand-labelled cwd12 test+valid, never trained on
    "val_dataset_root": CWD12,
}
if TIER == "curated":
    # quality-core gate: drop slugs whose DINOv2 trusted-pool score is too low
    strategy["min_dino_score"] = float(os.environ.get("MIN_DINO_SCORE", "0.5"))

out = {"tier": TIER, "seed": SEED, "iteration": ITER,
       "job_id": os.environ.get("SLURM_JOB_ID"), "strategy": strategy,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
print("[m1] strategy:", json.dumps(strategy))

try:
    best_pt, summary = train_yolo_mega(strategy, iteration=ITER)
    out["best_pt"] = str(best_pt)
    out["summary"] = summary if isinstance(summary, dict) else str(summary)
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(e).__name__, e)
    out["traceback"] = traceback.format_exc()[-2000:]
    print("[m1] FAILED:", out["error"])

out["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")

# Surface the holdout-guard counters explicitly: a re-measurement is only
# meaningful if we can show how many holdout images the guard blocked.
s = out.get("summary")
if isinstance(s, dict):
    for k in ("skipped_holdout_hash", "skipped_holdout_stem", "images",
              "datasets", "classes"):
        if k in s:
            print("[m1] %-22s %s" % (k, s[k]))

dest = Path(REPO) / "results" / "framework" / ("m1_%s_seed%d.json" % (TIER, SEED))
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, default=str))
print("[m1] wrote", dest)
PYEOF

echo "=== M1 $TIER seed=$SEED finished rc=$? $(date) ==="
