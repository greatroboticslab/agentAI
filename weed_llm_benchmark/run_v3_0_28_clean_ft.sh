#!/bin/bash
#SBATCH --job-name=v3028_ft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/v3_0_28_ft_%j.out

# v3.0.28 CLEAN FT — finetune the v3.0.28 pretrained model on cwd12 train.
#
# Loads the best.pt path written by run_v3_0_28_clean_pretrain.sh, then runs
# the same finetune flow as v3.0.27 (which we trust — only the BASE was
# contaminated, not the FT staging logic). Stem-level exclusion of test+valid
# from train is double-checked here as well.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.28 CLEAN FT on cwd12 train ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

PRETRAIN_PTR=results/framework/v3_0_28_pretrain_best.txt
if [ ! -f "$PRETRAIN_PTR" ]; then
    echo "FATAL: $PRETRAIN_PTR not found — pretrain must run first" >&2
    exit 3
fi
PRETRAIN_BEST=$(cat "$PRETRAIN_PTR" | head -1)
echo "Pretrained base.pt: $PRETRAIN_BEST"
if [ ! -f "$PRETRAIN_BEST" ]; then
    echo "FATAL: pretrained best.pt does not exist on disk" >&2
    exit 4
fi

python - <<PYEOF
import os, glob, shutil, sys, json
from pathlib import Path
import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"
OUT = f"{REPO}/results/framework/mega_iterv3_0_28_FT"
PRETRAINED = "$PRETRAIN_BEST"

V3_NAMES = ["Carpetweeds","Crabgrass","PalmerAmaranth","PricklySida",
            "Purslane","Ragweed","Sicklepod","SpottedSpurge",
            "Eclipta","Goosegrass","Morningglory","Nutsedge"]
CWD12_ORIG = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass",
              "Morningglory","Nutsedge","PalmerAmaranth","PricklySida",
              "Purslane","Ragweed","Sicklepod","SpottedSpurge"]
ORIG_TO_CANON = {i: V3_NAMES.index(n) for i,n in enumerate(CWD12_ORIG)}

out_train_img = Path(OUT) / "train" / "images"
out_train_lbl = Path(OUT) / "train" / "labels"
out_val_img   = Path(OUT) / "valid" / "images"
out_val_lbl   = Path(OUT) / "valid" / "labels"
for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

holdout_stems = set()
for split in ("test", "valid"):
    sd = Path(CWD12_ROOT) / split / "images"
    if sd.is_dir():
        for img in sd.glob("*.jpg"):
            holdout_stems.add(img.stem)
print(f"Holdout stems: {len(holdout_stems)}")

weedimg_dir = Path(CWD12_ROOT) / "CottonWeedDet12" / "weedImages"
yolotxt_dir = Path(CWD12_ROOT) / "CottonWeedDet12" / "annotation_YOLO_txt"
n_train = 0
for img in weedimg_dir.glob("*.jpg"):
    if img.stem in holdout_stems:
        continue
    lbl_src = yolotxt_dir / (img.stem + ".txt")
    if not lbl_src.exists():
        continue
    os.symlink(img.resolve(), out_train_img / img.name)
    out_lines = []
    for line in lbl_src.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            orig = int(parts[0])
        except ValueError:
            continue
        if orig in ORIG_TO_CANON:
            out_lines.append(f"{ORIG_TO_CANON[orig]} {' '.join(parts[1:])}")
    (out_train_lbl / (img.stem + ".txt")).write_text("\n".join(out_lines) + "\n")
    n_train += 1

n_val = 0
for split in ("test", "valid"):
    sdi = Path(CWD12_ROOT) / split / "images"
    sdl = Path(CWD12_ROOT) / split / "labels"
    if not sdi.is_dir():
        continue
    for img in sdi.glob("*.jpg"):
        stem = f"{split}__{img.stem}"
        os.symlink(img.resolve(), out_val_img / (stem + ".jpg"))
        lbl_src = sdl / (img.stem + ".txt")
        if lbl_src.exists():
            out_lines = []
            for line in lbl_src.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    orig = int(parts[0])
                except ValueError:
                    continue
                if orig in ORIG_TO_CANON:
                    out_lines.append(f"{ORIG_TO_CANON[orig]} {' '.join(parts[1:])}")
            (out_val_lbl / (stem + ".txt")).write_text("\n".join(out_lines) + "\n")
        n_val += 1

print(f"Staged: train={n_train} (cwd12 train portion), val={n_val} (cwd12 test+valid holdout)")
assert n_train == 3671, f"expected 3671 train, got {n_train}"
assert n_val == 1977,   f"expected 1977 val, got {n_val}"

# data.yaml — nc=100 to match v3.0.28 pretrain head
yaml_path = Path(OUT) / "data.yaml"
all_names = V3_NAMES + [f"aux_{i}" for i in range(12, 100)]
yaml.safe_dump({
    "train": str(out_train_img),
    "val":   str(out_val_img),
    "nc":    100,
    "names": all_names,
}, open(yaml_path, "w"))

print(f"Pretrained base.pt: {PRETRAINED}")
model = YOLO(PRETRAINED)
model.train(
    data=str(yaml_path),
    epochs=100,
    imgsz=1024,
    batch=8,
    lr0=0.0003,
    lrf=0.01,
    patience=30,
    device=0,
    project=f"{OUT}/runs",
    name="finetune",
    verbose=False,
    save_period=1,
    cos_lr=True,
    mosaic=0.5,
    mixup=0.0,
    close_mosaic=10,
)

print("\n=== v3.0.28 CLEAN FT COMPLETE ===")
weights = sorted(glob.glob(f"{OUT}/runs/finetune*/weights/best.pt"),
                 key=os.path.getmtime, reverse=True)
print(f"best.pt: {weights[0] if weights else 'NONE'}")

import csv
res_path = sorted(glob.glob(f"{OUT}/runs/finetune*/results.csv"),
                  key=os.path.getmtime, reverse=True)
if res_path:
    rows = list(csv.reader(open(res_path[0])))
    header = rows[0]
    last = rows[-1]
    summary = dict(zip(header, last))
    print(f"Final epoch: {summary.get('epoch','?')}")
    print(f"Final mAP50:    {summary.get('metrics/mAP50(B)','?')}")
    print(f"Final mAP50-95: {summary.get('metrics/mAP50-95(B)','?')}")
    json.dump(summary, open(f"{OUT}/v3_0_28_FT_summary.json","w"),
              indent=2, default=str)
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
