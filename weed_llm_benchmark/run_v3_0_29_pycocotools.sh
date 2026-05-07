#!/bin/bash
#SBATCH --job-name=v3029_coc
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_29_pycoco_%j.out

# v3.0.29 — pycocotools cross-check on safety best.pt
# Third-party adjudicator: settles whether 0.8953 (ultralytics) or 0.744
# (our custom WBF/TTA) is the canonical mAP50-95 on cwd12 holdout.
#
# Plan:
#  1. Convert cwd12 holdout YOLO labels → COCO ground-truth JSON
#  2. Run safety best.pt on cwd12 holdout, write predictions as COCO JSON
#  3. Use pycocotools.cocoeval.COCOeval to compute the CANONICAL mAP50, mAP50-95

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

pip install --quiet pycocotools 2>&1 | tail -3

python -m weed_optimizer_framework.tools.pycoco_eval \
    --weights results/framework/mega_iterv3_0_28_safety/runs/safety/weights/best.pt \
    --val-imgs results/framework/mega_iterv3_0_28_safety/valid/images \
    --val-lbls results/framework/mega_iterv3_0_28_safety/valid/labels \
    --imgsz 1024 \
    --conf 0.001 --iou 0.6 \
    --out results/framework/v3_0_29_pycoco_summary.json
