#!/bin/bash
#SBATCH --job-name=yolo_setup_train
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_finetune_%j.out
#SBATCH --error=slurm_finetune_%j.err

# ==============================================================
# Download CottonWeedDet12 + Fine-tune YOLO11n (all-in-one)
# ==============================================================

set -e

WORK_DIR="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
DATA_DIR="$WORK_DIR/downloads/cottonweeddet12"

cd "$WORK_DIR"

echo "============================================"
echo "Setup + Fine-tune YOLO on CottonWeedDet12"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate qwen
pip install ultralytics py7zr -q

# ---- Step 1: Download dataset if needed ----
if [ ! -f "$DATA_DIR/data.yaml" ]; then
    echo "[*] Setting up CottonWeedDet12..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    # Download 7z archive
    if [ ! -f "CottonWeedDet12.7z" ] && [ ! -d "CottonWeedDet12" ]; then
        echo "[*] Downloading from Zenodo..."
        wget -O CottonWeedDet12.7z \
            "https://zenodo.org/api/records/7535814/files/CottonWeedDet12.7z/content"
    fi

    # Extract
    if [ ! -d "CottonWeedDet12" ]; then
        echo "[*] Extracting with py7zr..."
        python -c "
import py7zr
print('Extracting CottonWeedDet12.7z...')
with py7zr.SevenZipFile('CottonWeedDet12.7z', mode='r') as z:
    z.extractall(path='.')
print('Done extracting.')
"
    fi

    # Split into train/valid/test using existing YOLO annotations
    echo "[*] Splitting into train/valid/test (65/20/15)..."
    cd "$WORK_DIR"
    python -c "
import os, random, shutil
from pathlib import Path

SRC_IMG = '$DATA_DIR/CottonWeedDet12/weedImages'
SRC_LBL = '$DATA_DIR/CottonWeedDet12/annotation_YOLO_txt'
OUT = '$DATA_DIR'

# Get all image stems that have both image and label
imgs = sorted([f for f in os.listdir(SRC_IMG) if f.endswith(('.jpg','.jpeg','.png'))])
print(f'Found {len(imgs)} images')

# Shuffle with fixed seed
random.seed(42)
random.shuffle(imgs)

n = len(imgs)
n_train = int(n * 0.65)
n_valid = int(n * 0.20)

splits = {
    'train': imgs[:n_train],
    'valid': imgs[n_train:n_train+n_valid],
    'test': imgs[n_train+n_valid:],
}

for split_name, split_imgs in splits.items():
    img_dir = os.path.join(OUT, split_name, 'images')
    lbl_dir = os.path.join(OUT, split_name, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_file in split_imgs:
        stem = Path(img_file).stem
        src_img = os.path.join(SRC_IMG, img_file)
        src_lbl = os.path.join(SRC_LBL, stem + '.txt')

        shutil.copy2(src_img, os.path.join(img_dir, img_file))
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(lbl_dir, stem + '.txt'))

    print(f'  {split_name}: {len(split_imgs)} images')

print('Split complete.')
"

    # Write data.yaml
    cat > "$DATA_DIR/data.yaml" << 'YAML'
names:
- 'Carpetweeds'
- 'Crabgrass'
- 'Eclipta'
- 'Goosegrass'
- 'Morningglory'
- 'Nutsedge'
- 'PalmerAmaranth'
- 'PricklySida'
- 'Purslane'
- 'Ragweed'
- 'Sicklepod'
- 'SpottedSpurge'
nc: 12
train: ../train/images
val: ../valid/images
test: ../test/images
YAML

    # Clean up archive to save space
    rm -f "$DATA_DIR/CottonWeedDet12.7z"
    echo "[+] Dataset ready"
else
    echo "[*] CottonWeedDet12 already set up"
fi

cd "$WORK_DIR"

# ---- Step 2: Fine-tune YOLO11n ----
echo ""
echo "[*] Starting YOLO11n fine-tune..."
python -c "
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='$DATA_DIR/data.yaml',
    epochs=100,
    imgsz=640,
    batch=-1,
    patience=20,
    project='runs',
    name='yolo11n_cottonweeddet12',
    exist_ok=True,
    device=0,
    cos_lr=True,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    workers=5,
)

print()
print('=== Training Complete ===')
print(f'Best mAP@0.5:      {results.box.map50:.4f}')
print(f'Best mAP@0.5:0.95: {results.box.map:.4f}')
print(f'Precision:          {results.box.mp:.4f}')
print(f'Recall:             {results.box.mr:.4f}')

# Evaluate on test set
best_pt = 'runs/yolo11n_cottonweeddet12/weights/best.pt'
print(f'\n=== Test Set Evaluation ===')
best_model = YOLO(best_pt)
test_results = best_model.val(data='$DATA_DIR/data.yaml', split='test')
print(f'Test mAP@0.5:      {test_results.box.map50:.4f}')
print(f'Test mAP@0.5:0.95: {test_results.box.map:.4f}')
print(f'Test Precision:    {test_results.box.mp:.4f}')
print(f'Test Recall:       {test_results.box.mr:.4f}')
"

echo ""
echo "[+] All done: $(date)"
