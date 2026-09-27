#!/bin/bash
#SBATCH --job-name=inc_train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%A_%a.out
#
# One INC run per array task: docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Executor".
#
#   sbatch --array=0-N%40 run_inc_job.sh <list_file> [<exp>]
#
# <list_file> holds one spec.json path per line; array task i runs line i+1
# with python -m weed_optimizer_framework.tools.inc.train --spec <that path>,
# which trains (or soups, or only scores), scores every exam through the locked
# scorer and writes run.json, done or failed. Then the experiment is advanced
# at once (inc.driver advance --exp <exp> --quiet, errors ignored), so the next
# array goes in as soon as any run finishes. <exp> defaults to the spec's own
# "exp". The job exits with the executor's status, never the driver's.
#
# The in-job advance runs on a compute node, alongside other tasks' advances
# and the login node's: the driver's exclusion is fcntl.flock on
# INC_DIR/<exp>/state.json.lock, which holds across nodes only on a filesystem
# mounted for it (Lustre 'flock', not 'localflock'). INC_JOB_ADVANCE decides:
#   auto (default)  advance only when inc.train --flock-check finds flock on
#                   INC_DIR/<exp> coherent across nodes on this node's mount
#                   (/proc/self/mountinfo); otherwise skip it, with a warning,
#                   and the login node's advance collects the run;
#   1               always advance here (flock checked by hand across two nodes);
#   0               never; advance from the login node only.
# The login node's own mount is not checked here: run the same --flock-check
# there once before an experiment. The executor does not rely on flock alone:
# its run-dir lock adds an O_EXCL owner file (inc/train.py).
#
# The driver submits this; by hand, only for a single spec. Slurm opens the log
# file before this script runs, so results/framework/inc/logs must exist first
# (mkdir -p it). --time is the default for a cheap (30-epoch) run; pass a longer
# one for cold 100-epoch runs.
#
# On a GPU, Ultralytics' AMP check loads yolo26n.pt (8.4.x) from this working
# directory ($REPO); the executor refuses a CUDA training run unless a complete
# copy is already there, rather than let parallel tasks download it in place.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH; the executor hashes the modules it runs into run.json and
# fails the run when one differs from the git-tracked nested copy
# ($REPO/weed_llm_benchmark/weed_optimizer_framework). Export INC_ALLOW_DRIFT=1
# before sbatch to run the outer copy anyway (recorded as a warning).
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
CONDA_SH=/jet/home/byler/miniconda3/etc/profile.d/conda.sh
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
INC_ROOT="${INC_DIR:-$REPO/results/framework/inc}"

LIST="${1:-}"
EXP="${2:-}"
if [ -z "$LIST" ] || [ ! -f "$LIST" ]; then
    echo "usage: sbatch --array=0-N run_inc_job.sh <list_file> [<exp>]   (list file '$LIST' not found)" >&2
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
echo "=== inc.train $SPEC  $(date)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

t0=$(date +%s)
python -u -m weed_optimizer_framework.tools.inc.train --spec "$SPEC"
rc=$?
echo "=== executor exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="

advance() {
    python -u -m weed_optimizer_framework.tools.inc.driver advance --exp "$EXP" --quiet \
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
            if python -u -m weed_optimizer_framework.tools.inc.train --flock-check "$INC_ROOT/$EXP"; then
                advance
            else
                echo "WARNING: flock on $INC_ROOT/$EXP is not known to be coherent across nodes here, so the" \
                     "driver was not advanced from this node; advance $EXP from the login node" \
                     "(or set INC_JOB_ADVANCE=1 once flock is verified across two nodes)" >&2
            fi ;;
        *)
            echo "WARNING: INC_JOB_ADVANCE=${INC_JOB_ADVANCE} is not auto, 1 or 0; the experiment was not advanced" >&2 ;;
    esac
fi
exit $rc
