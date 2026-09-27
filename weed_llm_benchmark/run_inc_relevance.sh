#!/bin/bash
#SBATCH --job-name=inc_relevance
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/step1/logs/relevance_%x_%j.out
#
# INC Step 1 source relevance: docs/INCREMENTAL_PROTOCOL.md, Step 1 ("Source
# relevance"). The Step 1 verifier certifies species labels, not relevance, so
# the increment pool holds whole sources that are not plant photographs. This
# job scores every source zero-shot with BioCLIP-2's own text tower: P(plant)
# per crop from the Step 1 crop embeddings (no image is opened, nothing is
# embedded again), tau = the 5th percentile of train_core's P(plant), and a
# source passes when the median P(plant) of its sampled crops is >= tau.
# Outputs: $REPO/results/framework/inc/step1/relevance.json and relevance.md.
# The text encoding and the matrix products are tiny; the job loads the model
# (from the Hugging Face cache, as verify embed did), hashes its weights file
# (recorded with the snapshot commit) and reads every crop embedding.
# Exit 2 with both files written means the calibration check failed (tau <
# 0.5: the prompts do not separate train_core's weeds from the non-plant
# set); relevance.load refuses that file. Read relevance.md.
#
# Submit (Slurm opens the log file before this script runs, so the log dir must exist):
#   mkdir -p /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/step1/logs
#   sbatch run_inc_relevance.sh build                    # --sample 300 --seed 0
#   sbatch run_inc_relevance.sh build --sample 500 --seed 1 --out <other.json>
# A rerun with the same inputs and parameters is a no-op; one over a file made
# from other inputs or parameters refuses unless --force.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH; deploy there first. Nothing is synced here. Every module
# the job imports is hashed into the log, and the job stops when an outer
# module differs from the git-tracked nested copy. Set
# INC_RELEVANCE_ALLOW_DRIFT=1 to run the outer copies anyway (the log still
# names every difference).
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Compute nodes may have no internet; BioCLIP-2 is in the Hugging Face cache.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CMD="${1:-}"
case "$CMD" in
    build) shift ;;
    *) echo "usage: sbatch run_inc_relevance.sh build [--sample 300] [--seed 0] [--base-dir DIR] [--out OUT.json] [--force]" >&2
       exit 2 ;;
esac

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc.relevance $CMD $*  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Every module this job imports, outer copy (what runs) vs nested (git-tracked).
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/verify.py tools/inc/select.py
         tools/inc/relevance.py tools/cwd12_species.py tools/near_dup.py tools/semisup_labeler.py)
drift=0
for m in "${MODULES[@]}"; do
    OUTER=$REPO/weed_optimizer_framework/$m
    NESTED=$REPO/weed_llm_benchmark/weed_optimizer_framework/$m
    [ -f "$OUTER" ] || { echo "FATAL: $OUTER missing: deploy the outer package copy first" >&2; exit 1; }
    if [ ! -f "$NESTED" ]; then
        state="no nested copy"
    elif cmp -s "$OUTER" "$NESTED"; then
        state="= nested"
    else
        state="DIFFERS from nested $(sha256sum "$NESTED" | cut -c1-16)"
        drift=1
    fi
    echo "module $m: $(sha256sum "$OUTER" | cut -c1-16)  $state"
done
if [ "$drift" = 1 ]; then
    if [ "${INC_RELEVANCE_ALLOW_DRIFT:-0}" = 1 ]; then
        echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC_RELEVANCE_ALLOW_DRIFT=1)"
    else
        echo "FATAL: outer modules differ from the nested copy (above): sync them, or set INC_RELEVANCE_ALLOW_DRIFT=1" >&2
        exit 1
    fi
fi

python -u -c "import numpy, torch, open_clip; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'open_clip', open_clip.__version__)" || exit 1

t0=$(date +%s)
python -u -m weed_optimizer_framework.tools.inc.relevance "$CMD" "$@"
rc=$?
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
