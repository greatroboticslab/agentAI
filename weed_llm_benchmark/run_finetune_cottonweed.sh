#!/bin/bash
#SBATCH --job-name=yolo_finetune
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_finetune_%j.out
#SBATCH --error=slurm_finetune_%j.err

# ==============================================================
# Fine-tune YOLO11n on CottonWeedDet12
# Target: mAP@0.5:0.95 > 0.9, Precision > 0.9, Recall > 0.9
# ==============================================================

set -e
cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark

echo "============================================"
echo "YOLO11n Fine-tune on CottonWeedDet12"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate qwen

python -c "
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/downloads/cottonweeddet12/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    project='runs',
    name='yolo11n_cottonweeddet12',
    exist_ok=True,
    device=0,
    cos_lr=True,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    augment=True,
    mixup=0.3,
    copy_paste=0.2,
    workers=4,
)

print()
print('=== Training Complete ===')
print(f'Best mAP@0.5:      {results.box.map50:.4f}')
print(f'Best mAP@0.5:0.95: {results.box.map:.4f}')
print(f'Precision:          {results.box.mp:.4f}')
print(f'Recall:             {results.box.mr:.4f}')

# Run test set evaluation
best_pt = 'runs/yolo11n_cottonweeddet12/weights/best.pt'
print(f'\n=== Evaluating best.pt on test set ===')
best_model = YOLO(best_pt)
test_results = best_model.val(
    data='/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/downloads/cottonweeddet12/data.yaml',
    split='test',
)
print(f'Test mAP@0.5:      {test_results.box.map50:.4f}')
print(f'Test mAP@0.5:0.95: {test_results.box.map:.4f}')
print(f'Test Precision:    {test_results.box.mp:.4f}')
print(f'Test Recall:       {test_results.box.mr:.4f}')
"

echo ""
echo "Done: $(date)"
