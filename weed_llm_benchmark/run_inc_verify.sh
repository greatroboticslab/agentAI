#!/bin/bash
#SBATCH --job-name=inc_verify
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/step1/logs/verify_%x_%j.out
#
# INC Step 1 box verifier (labeler Phase B): docs/INCREMENTAL_PROTOCOL.md, Step 1.
# Reads every harvested box back from its pixels (BioCLIP-2 crop features, a
# linear probe on train_core, per-class thresholds at 95 % recall) before any of
# it may enter the high-precision base. Outputs: $REPO/results/framework/inc/step1/
#   pool.jsonl, pool_meta.jsonl, pool_summary.json, cwd12_copies.jsonl,
#   labels/<slug>/<key>.<sha256[:16]>.txt (never rewritten)   (pool)
#   crops.csv, crops_skipped.csv, crops_info.json        (crops)
#   emb/emb_sXXX_of_NNN.npz                              (embed, one file per shard)
#   verifier/{probe.joblib,verifier.npz,thresholds.json,fit_info.json}  (fit)
#   calibration.json                                     (calibrate)
#   verified.jsonl, conflicts.csv, conflicts_sheet.png, admit_summary.json  (admit)
#
# Every stage resumes: hashes are cached per slug, embed keeps finished chunks
# and skips finished shards, so a job cut by the 8 h limit is simply resubmitted.
# After a `pool` re-run, every later stage must re-run too: embed, fit, calibrate
# and admit refuse a crops.csv made from another pool, and calibrate and admit
# refuse another shard set than the one the verifier was fitted on.
#
# Submit (Slurm opens the log file before this script runs, so the log dir must exist):
#   mkdir -p /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/step1/logs
#   sbatch run_inc_verify.sh pool
#   sbatch run_inc_verify.sh crops
#   sbatch --array=0-3 run_inc_verify.sh embed --nshards 4    # array task i -> --shard i
#   sbatch run_inc_verify.sh fit
#   sbatch run_inc_verify.sh calibrate
#   sbatch run_inc_verify.sh admit
#   sbatch run_inc_verify.sh all                               # every stage in one job
# Chain with --dependency=afterok:<jobid> as usual.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework) through
# PYTHONPATH; deploy there first. Nothing is synced here. Every module the job
# imports (inc/__init__.py, inc/common.py, inc/verify.py, cwd12_species.py,
# near_dup.py, mega_trainer.py, semisup_labeler.py) is hashed into the log, and
# the job stops when an outer module differs from the git-tracked nested copy:
# a stale outer mega_trainer would silently build the pool through an old class
# join. Set INC_VERIFY_ALLOW_DRIFT=1 to run the outer copies anyway (the log
# still names every difference). verify.py itself also refuses a mega_trainer
# whose join is not the v3.60.0 species join.
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Compute nodes may have no internet; BioCLIP-2 is in the Hugging Face cache.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CMD="${1:-}"
case "$CMD" in
    pool|crops|embed|fit|calibrate|admit|all) shift ;;
    *) echo "usage: sbatch [--array=0-N] run_inc_verify.sh {pool|crops|embed|fit|calibrate|admit|all} [--nshards n] [--shard i] [...]" >&2
       exit 2 ;;
esac
ARGS=("$@")
# In an array job, embed task i does shard i unless --shard was given.
if [ "$CMD" = embed ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    has_shard=0
    for a in ${ARGS[@]+"${ARGS[@]}"}; do [ "$a" = "--shard" ] && has_shard=1; done
    [ "$has_shard" = 1 ] || ARGS+=(--shard "$SLURM_ARRAY_TASK_ID")
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc.verify $CMD ${ARGS[*]-}  $(date)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Every module this job imports, outer copy (what runs) vs nested (git-tracked).
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/verify.py tools/cwd12_species.py
         tools/near_dup.py tools/mega_trainer.py tools/semisup_labeler.py)
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
    if [ "${INC_VERIFY_ALLOW_DRIFT:-0}" = 1 ]; then
        echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC_VERIFY_ALLOW_DRIFT=1)"
    else
        echo "FATAL: outer modules differ from the nested copy (above): sync them, or set INC_VERIFY_ALLOW_DRIFT=1" >&2
        exit 1
    fi
fi

python -u -c "import numpy, sklearn, torch, PIL, joblib; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
if [ "$CMD" = embed ] || [ "$CMD" = all ]; then
    python -u -c "import open_clip; print('open_clip', open_clip.__version__)" || exit 1
fi

t0=$(date +%s)
python -u -m weed_optimizer_framework.tools.inc.verify "$CMD" ${ARGS[@]+"${ARGS[@]}"}
rc=$?
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
