#!/bin/bash
#SBATCH --job-name=inc2_train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%A_%a.out
#
# One Protocol v3 / splits v2 INC run per array task (docs/CONTINUOUS_LOOP.md
# §3.5 "Executor"; docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Protocol v3").
#
#   sbatch --array=0-N%40 run_inc2_job.sh <list_file> [<exp>]
#
# The same job as run_inc_job.sh, with the v2 executor: array task i runs line
# i+1 of <list_file> with
#   python -m weed_optimizer_framework.tools.inc2.train --spec <that path>
# (the v2 never-train guard, the Protocol v3 recipe table, the arm, the scorer
# sidecar; the scorer itself is still the locked inc.scorer), then advances the
# experiment with the pinned driver:
#   python -m weed_optimizer_framework.tools.inc.driver advance --exp <exp> --quiet
# The job exits with the executor's status, never the driver's.
#
# INC_JOB_SCRIPT. The pinned driver submits whatever $INC_JOB_SCRIPT names
# (driver.job_script(); default run_inc_job.sh, the v1 executor, which refuses
# every tsw image). This script exports INC_JOB_SCRIPT as itself, the
# git-tracked copy $REPO/weed_llm_benchmark/run_inc2_job.sh (Slurm runs a
# spooled copy, so $0 is not that path), before anything else runs, so the
# in-job advance of a v2 experiment submits this script again and never the
# v1 one. A missing copy stops the job (exit 2).
#
# Code drift. The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH. Before the executor starts, every module the executor
# and the driver import (MODULES below, plus every tools/inc2/*.py of the
# nested copy) is compared with its git-tracked nested twin
# ($REPO/weed_llm_benchmark/weed_optimizer_framework); a missing outer module or
# a difference stops the job (exit 1), unless INC_ALLOW_DRIFT=1 (the same
# switch the executor and the driver honour; the log names every difference).
# Nothing here syncs, copies, resets or pulls the checkout.
#
# The in-job advance follows INC_JOB_ADVANCE as run_inc_job.sh does: auto (the
# default) advances only when inc2.train --flock-check finds flock on
# INC_DIR/<exp> coherent across nodes; 1 always; 0 never.
#
# Slurm opens the log file before this script runs, so results/framework/inc/logs
# must exist. --time is the default for a 30-epoch run; the driver passes
# --time 08:00:00 for an array holding a cold run.
#
# INC2_JOB_REPO and INC2_JOB_CONDA_SH replace the cluster paths below (tests only).
set -uo pipefail

REPO="${INC2_JOB_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC2_JOB_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
INC_ROOT="${INC_DIR:-$REPO/results/framework/inc}"

SELF="$REPO/weed_llm_benchmark/run_inc2_job.sh"
if [ ! -f "$SELF" ]; then
    echo "FATAL: $SELF is missing: the v2 job script must be the git-tracked copy (INC_JOB_SCRIPT)" >&2
    exit 2
fi
export INC_JOB_SCRIPT="$SELF"

LIST="${1:-}"
EXP="${2:-}"
if [ -z "$LIST" ] || [ ! -f "$LIST" ]; then
    echo "usage: sbatch --array=0-N run_inc2_job.sh <list_file> [<exp>]   (list file '$LIST' not found)" >&2
    exit 2
fi
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$((TASK + 1))
SPEC="$(sed -n "${LINE}p" "$LIST" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [ -z "$SPEC" ]; then
    echo "FATAL: $LIST has no spec on line $LINE (array task $TASK)" >&2
    exit 2
fi

source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
# >>> mlflow-off (the same block in run_inc_job.sh and run_inc2_job.sh; tests/test_job_mlflow_off.py)
# Ultralytics' MLflow callback registers itself whenever its settings say "mlflow": true (the shared
# ~/.config/Ultralytics/settings.json does), and at the end of every training run it copies the weights directory
# into runs/mlflow: about 0.08 GB per run, 337 GB by 2026-10-26, byte-identical to the run folders' epoch files. No
# MLFLOW_* variable turns it off, and pointing YOLO_CONFIG_DIR at a bare directory makes Ultralytics write fresh
# default settings there (datasets_dir becomes ~/datasets). So the job reads its own copy of the shared settings with
# only "mlflow" set false: YOLO_CONFIG_DIR is exported only after Ultralytics, imported under it, reads every other
# key equal to the shared file's, rewrites nothing and registers no MLflow hook. Otherwise the job runs as before,
# MLflow on, with a WARNING. The shared settings.json, datasets_dir and runs/mlflow are never touched.
# A chained array inherits this job's environment (the in-job advance submits it from inside the job, and
# inc.driver.submission_env keeps every variable but SLURM_*/SBATCH_*). A YOLO_CONFIG_DIR that is a job's own
# copy (ultralytics_cfg_*, on node-local disk that ends with that job) is never read as the shared settings:
# Ultralytics would fall back to a fresh default config (another datasets_dir, MLflow on). The caller's own
# value is kept (UL_CALLER) and the advance below hands it, never the job's copy, to the next array.
case "${YOLO_CONFIG_DIR:-}" in */ultralytics_cfg_*) unset YOLO_CONFIG_DIR ;; esac
UL_CALLER_SET="${YOLO_CONFIG_DIR+1}"
UL_CALLER="${YOLO_CONFIG_DIR:-}"
UL_SHARED="${YOLO_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}}/Ultralytics"
UL_JOB="${LOCAL:-${TMPDIR:-/tmp}}/ultralytics_cfg_${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
if mkdir -p "$UL_JOB/Ultralytics" 2>/dev/null && python - "$UL_SHARED" "$UL_JOB/Ultralytics" <<'PYEOF'
import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
with open(os.path.join(src, "settings.json")) as fh:
    s = json.load(fh)
if not isinstance(s, dict) or not isinstance(s.get("mlflow"), bool):
    sys.exit("the shared settings carry no boolean 'mlflow'")
s["mlflow"] = False
with open(os.path.join(dst, "settings.json"), "w") as fh:
    json.dump(s, fh, indent=2)
for f in sorted(os.listdir(src)):
    p = os.path.join(src, f)
    if f.endswith(".ttf") and os.path.isfile(p) and not os.path.lexists(os.path.join(dst, f)):
        os.symlink(p, os.path.join(dst, f))
if os.path.isfile(os.path.join(src, "persistent_cache.json")):
    shutil.copyfile(os.path.join(src, "persistent_cache.json"), os.path.join(dst, "persistent_cache.json"))
PYEOF
then
    UL_CHECK="$(YOLO_CONFIG_DIR="$UL_JOB" python - "$UL_SHARED" "$UL_JOB/Ultralytics" 2>&1 <<'PYEOF'
import hashlib, json, os, sys
src, dst = sys.argv[1], sys.argv[2]
f = os.path.join(dst, "settings.json")
before = hashlib.sha256(open(f, "rb").read()).hexdigest()
from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
from ultralytics.utils.callbacks import mlflow as ul_mlflow
shared = json.load(open(os.path.join(src, "settings.json")))
mine = dict(SETTINGS)
off = sorted(k for k in set(shared) | set(mine) if k != "mlflow" and shared.get(k) != mine.get(k))
why = []
if os.path.realpath(str(USER_CONFIG_DIR)) != os.path.realpath(dst):
    why.append("the user config dir is %s" % USER_CONFIG_DIR)
if mine.get("mlflow") is not False:
    why.append("mlflow is %r" % mine.get("mlflow"))
if off:
    why.append("keys differ from the shared settings: %s" % ", ".join(off))
if hashlib.sha256(open(f, "rb").read()).hexdigest() != before:
    why.append("Ultralytics rewrote the job's settings")
if ul_mlflow.callbacks:
    why.append("MLflow hooks registered: %s" % ", ".join(sorted(ul_mlflow.callbacks)))
if why:
    sys.exit("; ".join(why))
print("datasets_dir %s, runs_dir %s, weights_dir %s" % (mine.get("datasets_dir"), mine.get("runs_dir"),
                                                       mine.get("weights_dir")))
PYEOF
)"
    if [ $? -eq 0 ]; then
        export YOLO_CONFIG_DIR="$UL_JOB"
        echo "MLflow logging off: YOLO_CONFIG_DIR=$UL_JOB (a copy of $UL_SHARED/settings.json, mlflow false; $UL_CHECK)"
    else
        echo "WARNING: MLflow logging stays on for this job: the job's settings did not check ($(echo "$UL_CHECK" | tail -1))" >&2
    fi
else
    echo "WARNING: MLflow logging stays on for this job: no copy of $UL_SHARED/settings.json in $UL_JOB" >&2
fi
# <<< mlflow-off
echo "=== inc2.train $SPEC  $(date)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "INC_JOB_SCRIPT=$INC_JOB_SCRIPT"

sha16() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-16; else shasum -a 256 "$1" | cut -c1-16; fi
}
OUTER_PKG="$REPO/weed_optimizer_framework"
NESTED_PKG="$REPO/weed_llm_benchmark/weed_optimizer_framework"
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/driver.py tools/inc/gate.py tools/inc/splits.py
         tools/inc/scorer.py tools/inc/lora.py tools/cwd12_species.py tools/near_dup.py tools/mega_trainer.py
         tools/funnel/__init__.py tools/funnel/embed.py tools/funnel/leak.py)
if [ -d "$NESTED_PKG/tools/inc2" ]; then
    for f in "$NESTED_PKG"/tools/inc2/*.py; do
        [ -f "$f" ] && MODULES+=("tools/inc2/$(basename "$f")")
    done
fi
if [ -d "$NESTED_PKG" ] && [ "$(cd "$NESTED_PKG" && pwd -P)" != "$(cd "$OUTER_PKG" 2>/dev/null && pwd -P)" ]; then
    drift=0
    for m in "${MODULES[@]}"; do
        OUTER="$OUTER_PKG/$m"
        NESTED="$NESTED_PKG/$m"
        if [ ! -f "$OUTER" ]; then
            echo "FATAL: $OUTER missing: deploy the outer package copy first" >&2
            exit 1
        fi
        if [ ! -f "$NESTED" ]; then
            state="no nested copy"
        elif cmp -s "$OUTER" "$NESTED"; then
            state="= nested"
        else
            state="DIFFERS from nested $(sha16 "$NESTED")"
            drift=1
        fi
        echo "module $m: $(sha16 "$OUTER")  $state"
    done
    if [ "$drift" = 1 ]; then
        if [ "${INC_ALLOW_DRIFT:-0}" = 1 ]; then
            echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC_ALLOW_DRIFT=1)"
        else
            echo "FATAL: outer modules differ from the nested copy (above): deploy them, or set INC_ALLOW_DRIFT=1" >&2
            exit 1
        fi
    fi
else
    echo "no separate nested copy at $NESTED_PKG: the running package is the git-tracked one"
fi

t0=$(date +%s)
python -u -m weed_optimizer_framework.tools.inc2.train --spec "$SPEC"
rc=$?
echo "=== executor exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="

advance() {
    # the next array takes the caller's YOLO_CONFIG_DIR (or none), never this job's settings copy (mlflow-off)
    (if [ -n "${UL_CALLER_SET:-}" ]; then export YOLO_CONFIG_DIR="$UL_CALLER"; else unset YOLO_CONFIG_DIR; fi
     python -u -m weed_optimizer_framework.tools.inc.driver advance --exp "$EXP" --quiet) \
        || echo "WARNING: driver advance --exp $EXP failed (ignored; the next advance collects this run)" >&2
}

if [ -z "$EXP" ]; then
    EXP="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["exp"])' "$SPEC" 2>/dev/null)"
fi
if [ -z "$EXP" ]; then
    echo "WARNING: no exp given and none readable from $SPEC; the experiment was not advanced" >&2
else
    case "${INC_JOB_ADVANCE:-auto}" in
        1)
            advance ;;
        0)
            echo "in-job advance off (INC_JOB_ADVANCE=0): advance $EXP from the login node" ;;
        auto)
            if python -u -m weed_optimizer_framework.tools.inc2.train --flock-check "$INC_ROOT/$EXP"; then
                advance
            else
                echo "WARNING: flock on $INC_ROOT/$EXP is not known to be coherent across nodes here, so the" \
                     "driver was not advanced from this node; advance $EXP from the login node" \
                     "(with INC_JOB_SCRIPT=$INC_JOB_SCRIPT), or set INC_JOB_ADVANCE=1 once flock is verified" >&2
            fi ;;
        *)
            echo "WARNING: INC_JOB_ADVANCE=${INC_JOB_ADVANCE} is not auto, 1 or 0; the experiment was not advanced" >&2 ;;
    esac
fi
exit $rc
