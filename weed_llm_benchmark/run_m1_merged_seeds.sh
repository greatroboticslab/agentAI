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
#
# v3.25.0: every recipe knob below is read from the environment so a scheduler
# can render a capped train command without editing this file, and the run
# writes a second, job-scoped artifact. Defaults reproduce the historical run.
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

TIER="${TIER:-curated}"
SEED_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
SEEDS=(0 101 102 103)                 # index 1..3 -> 101,102,103
SEED="${SEEDS[$SEED_INDEX]}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-60}"
TRAIN_TIME_H="${TRAIN_TIME_H:-}"      # hours for ultralytics time=; empty = uncapped
IMGSZ="${IMGSZ:-640}"
PATIENCE="${PATIENCE:-20}"
ITER="${ITER_NAME:-m1_${TIER}_s${SEED}}"
BRAIN_DOMAIN="${BRAIN_DOMAIN:-weed}"
BRAIN_ROUND="${BRAIN_ROUND:-}"        # set by the round scheduler; blank otherwise
BRAIN_TRACE="${BRAIN_TRACE:-}"

# The scheduler finds this run's trace and its job-scoped artifact by the id
# sbatch printed, which for a Slurm array is SLURM_ARRAY_JOB_ID: every task after
# the first gets a fresh SLURM_JOB_ID, so naming those files after SLURM_JOB_ID
# would silently detach their metric and their walltime projection — the same
# class of silence that let two 12 h trains die at their walltime on 2026-08-29
# and go unnoticed for six days. SLURM_ARRAY_JOB_ID is unset outside an array and
# equals SLURM_JOB_ID for --array=1-1, so today's submissions are unchanged.
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}"

# Ultralytics auto-increments train/ -> train2/ when the run directory already
# exists, so a rerun of the same ITER can leave a metric reader attached to a
# foreign run. A job-scoped run name makes the save_dir deterministic; off SLURM
# there is no job id and the historical "train" name is kept.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    RUN_TAG="job${SLURM_JOB_ID}"
else
    RUN_TAG=""
fi

if [ -z "$BRAIN_TRACE" ]; then
    BRAIN_TRACE="$REPO/results/framework/_brain/$BRAIN_DOMAIN/trace/${ITER}_${JOB_KEY}.jsonl"
fi
mkdir -p "$(dirname "$BRAIN_TRACE")" 2>/dev/null || true

# The per-epoch trace projects the finish time against this job's own walltime —
# the check that was missing when two 12 h trains were killed at epoch 24/60 and
# 16/60 on 2026-08-29. squeue is the only source of the limit; when it is absent
# or prints something unparseable the projection is simply omitted, never fatal.
SLURM_WALLTIME_S=""
if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
    _TIMELIMIT="$(squeue -h -j "$SLURM_JOB_ID" -o "%l" 2>/dev/null | tr -d '[:space:]')"
    SLURM_WALLTIME_S="$(printf '%s' "$_TIMELIMIT" | awk -F'[-:]' '
        /^[0-9]+(-[0-9]+)?(:[0-9]+)*$/ {
            if (NF == 4)      { print $1*86400 + $2*3600 + $3*60 + $4 }
            else if (NF == 3) { print $1*3600 + $2*60 + $3 }
            else if (NF == 2) { print $1*60 + $2 }
        }')"
fi
export SLURM_WALLTIME_S

echo "=== M1 $TIER seed=$SEED job=${SLURM_JOB_ID:-none} $(date) ==="
echo "[cfg] epochs=$TRAIN_EPOCHS time_h=${TRAIN_TIME_H:-none} imgsz=$IMGSZ patience=$PATIENCE"
echo "[cfg] iter=$ITER run_tag=${RUN_TAG:-train} job_key=$JOB_KEY walltime_s=${SLURM_WALLTIME_S:-unknown}"
echo "[cfg] trace=$BRAIN_TRACE"

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

python -u - <<PYEOF
import json, logging, os, sys, time, traceback
from pathlib import Path

# The merge scans ~100k images before it prints anything of its own; stream the
# framework's logger to stdout so a multi-hour job is observable while it runs
# instead of only at exit.
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(name)s %(message)s")

REPO = "$REPO"
TIER = "$TIER"
SEED = int("$SEED")
ITER = "$ITER"
RUN_TAG = "$RUN_TAG" or None
DOMAIN = "$BRAIN_DOMAIN"
ROUND = "$BRAIN_ROUND"
TRACE_PATH = "$BRAIN_TRACE"
JOB_ID = os.environ.get("SLURM_JOB_ID")          # this task's own id
JOB_KEY = "$JOB_KEY"                            # the id sbatch printed
ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID")
os.chdir(REPO)

from weed_optimizer_framework.config import Config
from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

CWD12 = os.path.join(REPO, "downloads", "cottonweeddet12")
assert Path(CWD12).is_dir(), "cwd12 holdout root missing: " + CWD12

strategy = {
    "epochs": int(float("$TRAIN_EPOCHS")),
    "imgsz": int(float("$IMGSZ")),  # matches the historical scale runs being re-measured
    "batch_size": -1,
    "lr": 0.001,
    "patience": int(float("$PATIENCE")),
    "workers": 4,
    "seed": SEED,             # v3.22.3 passthrough -> ultralytics
    "deterministic": True,
    "include_autolabel": True,
    # honest holdout: val is the hand-labelled cwd12 test+valid, never trained on
    "val_dataset_root": CWD12,
    # v3.25.0: one JSONL record per validated epoch. The 2026-08-29 TIMEOUTs left
    # no per-epoch evidence, so the overrun was invisible until the kill.
    "trace_path": TRACE_PATH,
    "trace_meta": {"domain": DOMAIN, "round": ROUND, "step": "train",
                   "job_id": JOB_ID},
}

# v3.25.0: ultralytics time= is a wall-clock cap in hours. It overrides epochs
# and still writes a valid best.pt, so a pool that outgrew its walltime ends in
# a usable checkpoint instead of a TIMEOUT. Empty keeps the uncapped behaviour.
TRAIN_TIME_H = "$TRAIN_TIME_H".strip()
if TRAIN_TIME_H:
    strategy["time_h"] = float(TRAIN_TIME_H)

if TIER == "curated":
    # Quality-core gate: drop slugs whose DINOv2 trusted-pool score is too low.
    # The gate reads results/framework/dinov2_curator/slug_scores.json and, if that
    # file is missing, mega_trainer logs a warning and silently DISABLES itself —
    # which would produce a run labelled "curated" that is byte-identical to "raw".
    # Fail loudly instead: a mislabelled data point is worse than no data point.
    scores = Path(REPO) / "results" / "framework" / "dinov2_curator" / "slug_scores.json"
    if not scores.is_file():
        raise SystemExit(
            "REFUSING TO RUN: TIER=curated needs %s, which does not exist. "
            "Run the DINOv2 curator scoring pass first (run_s2_dino_scores.sh), "
            "then choose MIN_DINO_SCORE from the real score distribution." % scores)
    strategy["min_dino_score"] = float(os.environ["MIN_DINO_SCORE"])

out = {"tier": TIER, "seed": SEED, "iteration": ITER,
       "job_id": JOB_KEY, "task_job_id": JOB_ID, "strategy": strategy,
       "run_tag": RUN_TAG, "trace_path": TRACE_PATH,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
if ARRAY_TASK_ID:
    out["array_task_id"] = ARRAY_TASK_ID
print("[m1] strategy:", json.dumps(strategy))

FW = Path(REPO) / "results" / "framework"
FW.mkdir(parents=True, exist_ok=True)
dest = FW / ("m1_%s_seed%d.json" % (TIER, SEED))
dest_job = FW / ("m1_%s_seed%d_%s.json" % (TIER, SEED, JOB_KEY))

# The job-scoped copy is written BEFORE training. A walltime kill used to leave
# no artifact at all, so nothing could say which recipe had run or where its
# trace was. The legacy path is still written only at the end, unchanged, so
# every existing reader sees exactly the file it saw before.
started_record = dict(out)
started_record["status"] = "running"
dest_job.write_text(json.dumps(started_record, indent=1, default=str))
print("[m1] wrote", dest_job, "(status=running)")

try:
    best_pt, summary = train_yolo_mega(strategy, iteration=ITER, run_tag=RUN_TAG)
    out["best_pt"] = str(best_pt)
    out["summary"] = summary if isinstance(summary, dict) else str(summary)
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(e).__name__, e)
    out["traceback"] = traceback.format_exc()[-2000:]
    print("[m1] FAILED:", out["error"])

out["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
out["status"] = "done" if out.get("ok") else "failed"

# Surface the holdout-guard counters explicitly: a re-measurement is only
# meaningful if we can show how many holdout images the guard blocked.
s = out.get("summary")
if isinstance(s, dict):
    for k in ("skipped_holdout_hash", "skipped_holdout_stem", "images",
              "datasets", "classes", "run_tag", "save_dir",
              "base_weights_sha256", "fresh_start",
              # requested vs completed: a time= cap ends the run early and still
              # reports COMPLETED, so the shortfall has to be in the log too.
              "epochs_requested", "epochs_completed", "time_h"):
        if k in s:
            print("[m1] %-22s %s" % (k, s[k]))

dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, default=str))
print("[m1] wrote", dest)
dest_job.write_text(json.dumps(out, indent=1, default=str))
print("[m1] wrote", dest_job, "(status=%s)" % out["status"])
PYEOF

echo "=== M1 $TIER seed=$SEED finished rc=$? $(date) ==="
