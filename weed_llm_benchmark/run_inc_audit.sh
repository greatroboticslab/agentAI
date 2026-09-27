#!/bin/bash
#SBATCH --job-name=inc_audit
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# INC attribution item 4, the BioCLIP-2 label audit: docs/INCREMENTAL_PROTOCOL.md,
# "Gate", attribution. A 12-species probe is fitted on the trusted manifest's
# boxes only (verify's probe, CV and 95 %-recall thresholds) and judges every
# box of each audited manifest against that manifest's own labels. It reads
# the Step 1 crops and BioCLIP-2 embeddings (results/framework/inc/step1/:
# crops.csv, crops_skipped.csv, crops_info.json, emb/); it opens no image and
# embeds nothing, so its GPU stays idle. It still runs on GPU-shared because
# the allocation cis240145p is a GPU allocation: RM-shared submissions fail
# with "Invalid qos" (CHANGELOG v3.0.99.19, run_v3_0_99_rf_pull.sh). Memory is
# sized for loading every crop's embedding, as `verify fit` does.
# Outputs: OUT.json, OUT.md and OUT_boxes.csv (one row per audited box);
# --out must end in .json and may only overwrite an earlier audit's outputs.
#
# Every argument goes to the module:
#   python -m weed_optimizer_framework.tools.inc.audit --trusted T.jsonl \
#       --audit NAME=M.jsonl [NAME=M.jsonl ...] --out OUT.json [--nshards N]
#
# The pilot's post-hoc item 4 (Slurm opens the log file before this script
# runs, so the log dir must exist):
#   REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
#   INC=$REPO/results/framework/inc; M=$INC/pilot_v1/manifests
#   mkdir -p $INC/logs
#   sbatch run_inc_audit.sh --trusted $M/P0.jsonl \
#       --audit I1=$M/I1.jsonl I2=$M/I2.jsonl I3=$M/I3.jsonl I4=$M/I4.jsonl I5=$M/I5.jsonl \
#               Bswap=$M/Bswap.jsonl Breal=$M/Breal.jsonl \
#       --out $INC/pilot_v1/audit/label_audit.json
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH; deploy there first. Nothing is synced here. Every module
# the job imports is hashed into the log, and the job stops when an outer
# module differs from the git-tracked nested copy. Set INC_AUDIT_ALLOW_DRIFT=1
# to run the outer copies anyway (the log still names every difference).
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch run_inc_audit.sh --trusted T.jsonl --audit NAME=M.jsonl [NAME=M.jsonl ...] --out OUT.json [--nshards N]" >&2
    exit 2
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc.audit $*  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"

# Every module this job imports, outer copy (what runs) vs nested (git-tracked).
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/verify.py tools/inc/audit.py
         tools/cwd12_species.py tools/near_dup.py)
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
    if [ "${INC_AUDIT_ALLOW_DRIFT:-0}" = 1 ]; then
        echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC_AUDIT_ALLOW_DRIFT=1)"
    else
        echo "FATAL: outer modules differ from the nested copy (above): sync them, or set INC_AUDIT_ALLOW_DRIFT=1" >&2
        exit 1
    fi
fi

python -u -c "import numpy, sklearn, joblib; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)" || exit 1

t0=$(date +%s)
python -u -m weed_optimizer_framework.tools.inc.audit "$@"
rc=$?
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
