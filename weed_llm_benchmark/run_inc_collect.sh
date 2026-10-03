#!/bin/bash
#SBATCH --job-name=inc_collect
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# The targeted collector's cluster verbs as batch jobs (docs/CONTINUOUS_LOOP.md
# section 3.1, section 6.3 levers L15, L16, section 7.2). One job runs one verb:
#
#   python -m weed_optimizer_framework.tools.collect VERB [options]
#
# VERB is one of plan, fetch, intake, summary, probe. `names` (lever L26) needs
# the taxonomy authority's network and runs on the lab only, never here.
#
# Partition. GPU-shared with one V100: the allocation is a GPU allocation and
# RM-shared submissions fail with "Invalid qos" (FUNNEL_AUDIT_RUNNER.md, note
# 33). The collector never trains. Only intake may use the GPU: a batch with a
# dHash copy of an evaluation image describes it with the v2 calibration's
# DINOv2 (D28-v2), which runs on the GPU when CUDA is there.
#
# Network. Compute nodes have no internet, so intake runs with
# HF_HUB_OFFLINE=1 and its DINOv2 comes from the Hugging Face cache (as in
# run_inc2_stream.sh). Only intake: fetch may need the network (a provider
# served by Hugging Face), and the other verbs load no model.
#
# Placement. fetch runs here only for a provider the network probe placed on
# the cluster (INC_DIR/intake/placement.json); the collector itself refuses any
# other provider inside Slurm, so its lab hook fetches it instead.
#
# Log. Slurm opens the --output file before this script runs, and a job whose
# log directory does not exist fails without a log. The log goes to
# results/framework/inc/logs, the directory the autopilot's stream-submit
# creates before every sbatch (as run_inc2_stream.sh's does), never to
# intake/logs, which nothing creates before the first job (the R0 probe).
#
# Submit from $REPO:
#   mkdir -p $REPO/results/framework/inc/logs
#   sbatch -p GPU-shared run_inc_collect.sh probe
#   sbatch -p GPU-shared run_inc_collect.sh fetch --source ID --max-bytes B
#   sbatch -p GPU-shared run_inc_collect.sh intake --source ID
#
# What the job checks before the verb runs (a refusal exits 2, an environment
# failure exits 1, and the verb's own exit code is the job's otherwise):
#   * the verb is one of the five above;
#   * the code: this job imports the git-tracked NESTED copy
#     ($REPO/weed_llm_benchmark) and logs the sha256 of every file of the
#     collector package and of the modules it calls (the funnel's package
#     init, domain, names, taxonomy and fetch; inc/__init__, inc/common and
#     inc/verify; inc2/__init__, inc2/common, inc2/guard,
#     inc2/embed_calibration and inc2/eval_hits, which intake needs; near_dup,
#     cwd12_species, registry_lock, license_audit, and mega_trainer.py, whose
#     dHash every guard uses and whose never-train slugs the collector parses;
#     funnel/leak, funnel/embed and semisup_labeler, through which intake
#     describes the images it refuses as dHash copies of evaluation images,
#     D28-v2). It refuses when the script that runs, or
#     $REPO/run_inc_collect.sh, differs from the nested copy of this script,
#     and when a module intake needs is missing. It never rewrites the
#     checkout: no reset, no copying of the nested package over the outer one;
#   * the imports each verb needs (PIL and numpy for intake). torch and
#     transformers are not required: only a batch with a dHash copy of an
#     evaluation image loads the descriptor model, and without it that hit is
#     recorded unweighed, which D28 reads as a leak (fail closed);
#   * Python logging is configured (logging.basicConfig) before the verb runs,
#     so the collector's log lines reach the job log.
# Nothing here uploads anywhere or syncs any labelling service.
#
# INC_COLLECT_REPO and INC_COLLECT_CONDA_SH replace the cluster paths below,
# and INC_COLLECT_DRY_RUN=1 stops after every check with the command that
# would run (tests only).
set -uo pipefail

REPO="${INC_COLLECT_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC_COLLECT_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
DRY_RUN="${INC_COLLECT_DRY_RUN:-0}"
export REPO
INC_DIR="$REPO/results/framework/inc"
export INC_DIR
CODE="$REPO/weed_llm_benchmark"
COL="$CODE/weed_optimizer_framework/tools/collect"
LOGS="$INC_DIR/logs"
MOD=weed_optimizer_framework.tools.collect
cd "$CODE" || exit 1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    echo "usage: sbatch -p GPU-shared run_inc_collect.sh VERB [options]" >&2
    echo "  VERB: plan fetch intake summary probe (names runs on the lab)" >&2
    exit 2
}

VERB="${1:-}"
case "$VERB" in
    plan|fetch|intake|summary|probe) shift ;;
    *) usage ;;
esac
ARGS=("$@")
if [ "$VERB" = intake ]; then
    export HF_HUB_OFFLINE=1
fi

mkdir -p "$LOGS" || exit 1
source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc_collect $VERB ${ARGS[*]-}  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(command -v python)  $(python -V 2>&1)"
echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-unset}"

sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi
}
# This script: the one that runs must be the nested (git-tracked) copy.
SELF="$CODE/run_inc_collect.sh"
[ -f "$SELF" ] || { echo "FATAL: $SELF missing: the nested copy is what this job runs" >&2; exit 1; }
echo "script $SELF: $(sha "$SELF")"
if [ -r "$0" ] && ! cmp -s "$0" "$SELF"; then
    echo "FATAL: the script that runs ($0) differs from $SELF: submit the git-tracked copy" >&2
    exit 2
fi
if [ -f "$REPO/run_inc_collect.sh" ] && ! cmp -s "$REPO/run_inc_collect.sh" "$SELF"; then
    echo "FATAL: $REPO/run_inc_collect.sh differs from $SELF: sync it from the nested copy" >&2
    exit 2
fi
# The code: every file of the collector package, and the modules it calls, from the nested copy.
[ -d "$COL" ] || { echo "FATAL: $COL missing in the nested copy" >&2; exit 1; }
COL_FILES="$(cd "$CODE/weed_optimizer_framework" && find tools/collect -type f \( -name '*.py' -o -name '*.json' \) \
             ! -path '*/__pycache__/*' | LC_ALL=C sort)"
for m in $COL_FILES tools/funnel/__init__.py tools/funnel/domain.py tools/funnel/names.py tools/funnel/taxonomy.py \
         tools/funnel/fetch.py tools/funnel/leak.py tools/funnel/embed.py tools/funnel/domains/weed.json \
         tools/inc/__init__.py tools/inc/common.py tools/inc/verify.py tools/near_dup.py tools/cwd12_species.py \
         tools/registry_lock.py tools/license_audit.py tools/mega_trainer.py tools/semisup_labeler.py; do
    f="$CODE/weed_optimizer_framework/$m"
    [ -f "$f" ] || { echo "FATAL: module missing in the nested copy: $m" >&2; exit 1; }
    echo "module $m: $(sha "$f")"
done
for m in tools/inc2/__init__.py tools/inc2/common.py tools/inc2/guard.py tools/inc2/embed_calibration.py \
         tools/inc2/eval_hits.py; do
    f="$CODE/weed_optimizer_framework/$m"
    if [ -f "$f" ]; then
        echo "module $m: $(sha "$f")"
    elif [ "$VERB" = intake ]; then
        echo "FATAL: intake needs $m (the copy guard, fail closed) and the nested copy lacks it" >&2
        exit 2
    else
        echo "module $m: absent"
    fi
done

IMPORTS="json"
[ "$VERB" = intake ] && IMPORTS="$IMPORTS, PIL, numpy"
python -u -c "import $IMPORTS; print('imports: $IMPORTS')" || { echo "FATAL: the bench env lacks one of: $IMPORTS" >&2; exit 1; }

RUN='import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from weed_optimizer_framework.tools.collect.__main__ import main
sys.exit(main(sys.argv[1:]))'

t0=$(date +%s)
if [ "$DRY_RUN" = 1 ]; then
    ARGSTR="${ARGS[*]-}"
    echo "DRY RUN: python -u -c <logging.basicConfig; $MOD main> $VERB${ARGSTR:+ $ARGSTR}"
    rc=0
else
    python -u -c "$RUN" "$VERB" ${ARGS[@]+"${ARGS[@]}"}
    rc=$?
fi
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
