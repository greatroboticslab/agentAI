#!/bin/bash
#SBATCH --job-name=v3029_rfd
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/framework/v3_0_29_rfdetr_%j.out

# v3.0.29 Phase 2B — RF-DETR finetune on cwd12 train + cwd12 holdout val
#
# RF-DETR (Roboflow, ICLR 2026, arXiv 2511.09554):
#   - DINOv2 ViT backbone (frozen by default during finetune)
#   - Detection transformer head (DETR-style, NAS-discovered architecture)
#   - SOTA real-time mAP on COCO + RF100-VL cross-domain benchmark
#   - pip install rfdetr — much simpler than custom DINOv3+YOLO26 dual-branch
#
# Our setup:
#   - Base: RFDETRMedium (good balance for V100-32GB)
#   - Train: cwd12 train (3,671) — stem-filtered from weedImages
#   - Val:   cwd12 test+valid (1,977 NEVER_TRAIN holdout)
#   - Resolution: 728 (divisible by 56 as RF-DETR requires)
#   - Eval: pycocotools (industry standard, NOT ultralytics)
#
# Expected: cwd12 mAP50-95 pycocotools >= 0.78 (vs safety yolo26x = 0.7446).
# If we hit 0.78+, this is a +3.5% absolute improvement and the cleanest
# arch upgrade we've tried.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.29 Phase 2B RF-DETR ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

pip install --quiet rfdetr pycocotools 2>&1 | tail -3

python -m weed_optimizer_framework.tools.train_rfdetr \
    --out results/framework/mega_iterv3_0_29_rfdetr \
    --epochs 60 \
    --batch 4 \
    --grad-accum 4 \
    --resolution 728 \
    --lr 1e-4

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
