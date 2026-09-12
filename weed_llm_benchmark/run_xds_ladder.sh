#!/bin/bash
#SBATCH --job-name=xds_ladder
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=06:00:00
#SBATCH --output=results/framework/xds_ladder_%j.out
#
# Change the exam. Every number on the tier ladder is measured on CottonWeedDet12's
# own holdout, and that metric rewards training data that looks like CottonWeedDet12
# -- so a corpus of greenhouse, aerial and three-season images can only ever cost.
# This scores the SAME ladder checkpoints on ImageWeeds instead, class-agnostic,
# with the same leak check, matcher, conf and imgsz on both sides.
#
# If adding harvested data raises the ImageWeeds number while lowering the
# CottonWeedDet12 number, "more data does not help" is a statement about the exam
# rather than about the data, and that is the finding.
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$REPO REPO_ROOT=$REPO
nvidia-smi --query-gpu=name --format=csv,noheader

T=$REPO/results/framework/s3_tiers
W=""
for r in 0 5000 15000 40000; do
  [ -f "$T/v2_run_$r/weights/best.pt" ] && W="$W,A+$r=$T/v2_run_$r/weights/best.pt"
done
for r in 0 5000 15000 40000; do
  [ -f "$T/v2b_run_$r/weights/best.pt" ] && W="$W,B+$r=$T/v2b_run_$r/weights/best.pt"
done
W="${W#,}"
echo "[xds] checkpoints: $W"
[ -z "$W" ] && { echo "FATAL: no ladder checkpoints found under $T" >&2; exit 1; }

export XDS_WEIGHTS="$W"
export XDS_OUT=$REPO/results/framework/s6_crossdataset_ladder.json
python -u -m weed_optimizer_framework.tools.crossdataset_eval
echo "[xds] DONE rc=$?"
