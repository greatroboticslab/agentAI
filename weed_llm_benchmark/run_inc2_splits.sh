#!/bin/bash
#SBATCH --job-name=inc2_splits
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# Splits v2 as batch jobs: docs/CONTINUOUS_LOOP.md 4.2 (lever L23, rollout R0 and
# R0b). One job runs one verb of
#
#   python -m weed_optimizer_framework.tools.inc2.splits VERB [flags]
#
# VERB is build, scan, lock, verify or summary. The sequence is
#   build -> lock
# (L23's two verbs, as the platform submits them through run_inc2_build.sh):
# build runs the embedding copy scan itself when no embed_scan.json covers
# the current candidates, then applies it; scan runs the scan alone. build,
# lock and verify re-hash every image of splits v1 and v2, which outlives
# what the login node lets a process run; the scan describes about 30k images
# with DINOv2 and needs the GPU. lock, verify and summary leave the V100 idle:
# the allocation cis240145p is a GPU allocation and RM-shared submissions
# fail with "Invalid qos" (run_inc_build.sh, FUNNEL_AUDIT_RUNNER note 33).
#
# Refused here (exit 2), before anything runs:
#   * a verb outside the list above;
#   * --testing (a synthetic world's flag: it lifts the real-data pins, and
#     lock --testing would seal such a build);
#   * scan, or build without --skip-scan, without a GPU listed by nvidia-smi.
#
# Code: the job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH, as the other INC jobs do. Every module the verbs import
# is hashed into the log, and the job stops (exit 1) when an outer module
# differs from the git-tracked nested copy ($REPO/weed_llm_benchmark/...).
# INC2_SPLITS_ALLOW_DRIFT=1 runs the outer copies anyway (the log names every
# difference). Nothing is synced, copied or reset here.
#
# One writer: inc2.splits itself holds INC_DIR/splits/.v2.writer.lock.
# Compute nodes have no internet: DINOv2 comes from the Hugging Face cache.
#
# Submit from $REPO (Slurm opens the log file first, so the log dir must exist):
#   mkdir -p $REPO/results/framework/inc/logs
#   sbatch run_inc2_splits.sh build
#   sbatch run_inc2_splits.sh lock
#
# INC2_SPLITS_REPO and INC2_SPLITS_CONDA_SH replace the cluster paths below and
# INC2_SPLITS_DRY_RUN=1 stops after every check, printing the command that
# would run (tests only).
set -uo pipefail

REPO="${INC2_SPLITS_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC2_SPLITS_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
DRY_RUN="${INC2_SPLITS_DRY_RUN:-0}"
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1

usage() {
    echo "usage: sbatch run_inc2_splits.sh {build | scan | lock | verify | summary} [flags ...]" >&2
    exit 2
}

VERB="${1:-}"
case "$VERB" in
    build|scan|lock|verify|summary) shift ;;
    *) usage ;;
esac
for a in "$@"; do
    case "$a" in
        --testing|--testing=*)
            echo "FATAL: --testing is a synthetic world's flag; the job never lifts the real-data pins" >&2
            exit 2 ;;
    esac
done

source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc2.splits $VERB $*  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(command -v python)  $(python -V 2>&1)"
GPUS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "GPU: ${GPUS:-none}"
EMBEDS=0
if [ "$VERB" = scan ]; then
    EMBEDS=1
elif [ "$VERB" = build ]; then
    EMBEDS=1
    for a in "$@"; do [ "$a" = "--skip-scan" ] && EMBEDS=0; done
fi
if [ "$EMBEDS" = 1 ] && [ -z "$GPUS" ]; then
    echo "FATAL: $VERB describes images with DINOv2 (the embedding copy scan) and nvidia-smi lists no GPU" >&2
    echo "submit it as:  sbatch run_inc2_splits.sh $VERB $*" >&2
    exit 2
fi

# Every module the verbs import (the scan's included), outer copy (what runs)
# against the nested one (git-tracked).
# funnel/domains/weed.json is read by build and scan (licences, lab groups, the
# embedder, the augmentation families, the negative groups and, for the v2
# calibration of decision L-9, the non-plant sources).
MODULES=(tools/inc2/__init__.py tools/inc2/common.py tools/inc2/guard.py tools/inc2/splits.py
         tools/inc2/embed_calibration.py
         tools/inc/__init__.py tools/inc/common.py tools/inc/splits.py tools/inc/scorer.py tools/inc/verify.py
         tools/funnel/__init__.py tools/funnel/leak.py tools/funnel/embed.py tools/funnel/estimate.py
         tools/funnel/domain.py tools/funnel/ledger.py tools/funnel/domains/weed.json
         tools/near_dup.py tools/cwd12_species.py tools/mega_trainer.py tools/semisup_labeler.py)
drift=0
for m in "${MODULES[@]}"; do
    OUTER=$REPO/weed_optimizer_framework/$m
    NESTED=$REPO/weed_llm_benchmark/weed_optimizer_framework/$m
    if [ ! -f "$OUTER" ]; then
        echo "FATAL: $OUTER missing: deploy the outer package copy first" >&2
        exit 1
    fi
    if [ ! -f "$NESTED" ]; then
        state="no nested copy"
        drift=1
    elif cmp -s "$OUTER" "$NESTED"; then
        state="= nested"
    else
        state="DIFFERS from nested $(sha256sum "$NESTED" | cut -c1-16)"
        drift=1
    fi
    echo "module $m: $(sha256sum "$OUTER" | cut -c1-16)  $state"
done
if [ "$drift" = 1 ]; then
    if [ "${INC2_SPLITS_ALLOW_DRIFT:-0}" = 1 ]; then
        echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC2_SPLITS_ALLOW_DRIFT=1)"
    else
        echo "FATAL: outer modules differ from the nested copy (above): deploy them, or set INC2_SPLITS_ALLOW_DRIFT=1" >&2
        exit 1
    fi
fi

python -u -c "import numpy, PIL; print('numpy', numpy.__version__, 'PIL', PIL.__version__)" || exit 1
if [ "$EMBEDS" = 1 ]; then
    python -u -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)" || exit 1
fi

CMD=(python -u -m weed_optimizer_framework.tools.inc2.splits "$VERB" "$@")
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN: ${CMD[*]}"
    exit 0
fi
t0=$(date +%s)
"${CMD[@]}"
rc=$?
echo "=== inc2.splits $VERB exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit "$rc"
