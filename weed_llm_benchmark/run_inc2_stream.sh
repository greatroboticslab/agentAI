#!/bin/bash
#SBATCH --job-name=inc2_stream
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# The incremental Step 1 of the continuous loop as batch jobs
# (docs/CONTINUOUS_LOOP.md §3.3; lever L17 "admit an intake batch").
# One job runs one verb of the CLI:
#
#   python -u -m weed_optimizer_framework.tools.inc2.step1_stream VERB [options]
#
# VERB is one of bootstrap, admit, backfill, knowntruth, rejoin, serve-holds
# (also accepted as scan-holds, the name the autopilot submits), status,
# verify. The platform submits, for example:
#
#   sbatch -p GPU-shared run_inc2_stream.sh admit --intake <batch>
#   sbatch -p GPU-shared run_inc2_stream.sh admit --registry
#   sbatch -p GPU-shared run_inc2_stream.sh backfill           # batch b0000, once
#   sbatch -p GPU-shared run_inc2_stream.sh knowntruth --sets tsw22,tsw23
#   sbatch -p GPU-shared run_inc2_stream.sh scan-holds --hold h6_scan
#
# Log. Slurm opens the --output file before this script runs, and a job whose
# log directory does not exist fails without a log. The log goes to
# results/framework/inc/logs (the directory run_inc_job.sh, run_inc2_job.sh
# and the autopilot's stream-submit create), never to step1_stream/, which
# does not exist before the first bootstrap.
#
# Partition. Every verb runs on GPU-shared with one V100: the allocation
# refuses RM-shared ("Invalid qos", FUNNEL_AUDIT_RUNNER.md note 33). admit and
# knowntruth embed crops with BioCLIP-2; backfill, rejoin and serve-holds run
# the DINOv2 copy scan when a passed calibration exists; the others leave the
# GPU idle.
#
# What the job checks before the verb runs (a refusal exits 2, an
# environment failure exits 1, the verb's own exit code is the job's otherwise):
#   * the verb is one of the list; no array job (step1_stream has one writer,
#     and the module takes step1_stream/.lock itself);
#   * admit and knowntruth see a GPU;
#   * the script that runs, and $REPO/run_inc2_stream.sh when it exists, equal
#     the git-tracked nested copy;
#   * the code: the job imports the NESTED copy ($REPO/weed_llm_benchmark).
#     The modules it calls as a library (inc/verify.py, inc/select.py,
#     funnel/recover.py, funnel/leak.py, cwd12_species.py, near_dup.py,
#     semisup_labeler.py, mega_trainer.py) must have an OUTER copy
#     ($REPO/weed_optimizer_framework) byte-equal to the nested one: other jobs
#     import the outer copy, and a drift between the two (the stale
#     model_router.py of 2026-09-27) would let two jobs judge with different
#     code. The inc2 modules and the other modules it imports are hashed into
#     the log, and refused when an outer copy exists and differs. This job
#     never rewrites either copy: it only compares them;
#   * the imports the verb needs.
#
# INC2_STREAM_REPO and INC2_STREAM_CONDA_SH replace the cluster paths below,
# and INC2_STREAM_DRY_RUN=1 stops after every check with the command that
# would run (tests only).
set -uo pipefail

REPO="${INC2_STREAM_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC2_STREAM_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
DRY_RUN="${INC2_STREAM_DRY_RUN:-0}"
export REPO
INC_DIR="$REPO/results/framework/inc"
export INC_DIR
CODE="$REPO/weed_llm_benchmark"
OUTER="$REPO/weed_optimizer_framework"
NESTED="$CODE/weed_optimizer_framework"
LOGS="$INC_DIR/logs"
MOD=weed_optimizer_framework.tools.inc2.step1_stream
cd "$CODE" || exit 1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
# Compute nodes have no internet: BioCLIP-2 and DINOv2 come from the Hugging Face cache.
export HF_HUB_OFFLINE=1

usage() {
    echo "usage: sbatch run_inc2_stream.sh VERB [options]" >&2
    echo "  VERB: bootstrap admit backfill knowntruth rejoin serve-holds scan-holds status verify" >&2
    exit 2
}

VERB="${1:-}"
case "$VERB" in
    bootstrap|admit|backfill|knowntruth|rejoin|serve-holds|scan-holds|status|verify) shift ;;
    *) usage ;;
esac
ARGS=("$@")

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "FATAL: step1_stream has one writer; it never runs as an array job" >&2
    exit 2
fi

mkdir -p "$LOGS" || exit 1
source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc2_stream $VERB ${ARGS[*]-}  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(command -v python)  $(python -V 2>&1)"

GPUS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)"
echo "GPU: ${GPUS:-none}"
case "$VERB" in
    admit|knowntruth)
        [ -n "$GPUS" ] || { echo "FATAL: $VERB embeds crops and nvidia-smi lists no GPU; submit it with -p GPU-shared --gres=gpu:v100-32:1" >&2; exit 2; } ;;
esac

sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi
}

# This script: the one that runs must be the nested (git-tracked) copy.
SELF="$CODE/run_inc2_stream.sh"
[ -f "$SELF" ] || { echo "FATAL: $SELF missing: the nested copy is what this job runs" >&2; exit 1; }
echo "script $SELF: $(sha "$SELF")"
if [ -r "$0" ] && ! cmp -s "$0" "$SELF"; then
    echo "FATAL: the script that runs ($0) differs from $SELF: submit the git-tracked copy" >&2
    exit 2
fi
if [ -f "$REPO/run_inc2_stream.sh" ] && ! cmp -s "$REPO/run_inc2_stream.sh" "$SELF"; then
    echo "FATAL: $REPO/run_inc2_stream.sh differs from $SELF: sync it from the nested copy" >&2
    exit 2
fi

# The library modules: an outer copy must exist and equal the nested one.
PINNED=(tools/inc/verify.py tools/inc/select.py tools/funnel/recover.py tools/funnel/leak.py
        tools/cwd12_species.py tools/near_dup.py tools/semisup_labeler.py tools/mega_trainer.py)
drift=0
for m in "${PINNED[@]}"; do
    n="$NESTED/$m"
    o="$OUTER/$m"
    [ -f "$n" ] || { echo "FATAL: module missing in the nested copy: $m" >&2; exit 1; }
    if [ ! -f "$o" ]; then
        echo "module $m: $(sha "$n")  NO OUTER COPY"
        drift=1
    elif cmp -s "$o" "$n"; then
        echo "module $m: $(sha "$n")  = outer"
    else
        echo "module $m: nested $(sha "$n")  DIFFERS from outer $(sha "$o")"
        drift=1
    fi
done
# The stream's own modules and the rest it imports: logged; an outer copy that exists must agree.
OWN="$(cd "$NESTED" && find tools/inc2 -maxdepth 1 -type f -name '*.py' | LC_ALL=C sort)"
for m in $OWN tools/inc/__init__.py tools/inc/common.py tools/funnel/__init__.py tools/funnel/embed.py \
         tools/funnel/domain.py tools/funnel/qualify.py tools/dataset_discovery.py tools/registry_lock.py; do
    n="$NESTED/$m"
    o="$OUTER/$m"
    [ -f "$n" ] || { echo "FATAL: module missing in the nested copy: $m" >&2; exit 1; }
    if [ -f "$o" ] && ! cmp -s "$o" "$n"; then
        echo "module $m: nested $(sha "$n")  DIFFERS from outer $(sha "$o")"
        drift=1
    else
        echo "module $m: $(sha "$n")"
    fi
done
if [ "$drift" = 1 ]; then
    echo "FATAL: the outer package copy differs from the nested one (above): deploy both copies from one commit; this job never syncs them" >&2
    exit 2
fi

IMPORTS="numpy, sklearn, joblib, PIL"
case "$VERB" in
    admit|knowntruth) IMPORTS="$IMPORTS, torch, open_clip" ;;
    backfill|rejoin|serve-holds|scan-holds) IMPORTS="$IMPORTS, torch, transformers" ;;
esac
python -u -c "import $IMPORTS; print('imports: $IMPORTS')" || { echo "FATAL: the bench env lacks one of: $IMPORTS" >&2; exit 1; }

t0=$(date +%s)
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN: python -u -m $MOD $VERB ${ARGS[*]-}"
    rc=0
else
    python -u -m "$MOD" "$VERB" ${ARGS[@]+"${ARGS[@]}"}
    rc=$?
fi
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
