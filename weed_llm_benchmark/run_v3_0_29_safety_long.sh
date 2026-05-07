#!/bin/bash
#SBATCH --job-name=v3029_sl
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/framework/v3_0_29_safety_long_%j.out

# v3.0.29.1 — extended safety run (200 epochs, was 62 timeout)
#
# v3.0.28 SAFETY peaked at ~0.8953 around epoch 30, then plateau-oscillated.
# Resubmit with 24h walltime to complete all 200 planned epochs and confirm
# whether the plateau is real or just a transient. patience=50 (was 40) so
# we don't early-stop prematurely.
#
# Same setup: yolo26x + COCO weights, cwd12 train alone, val on cwd12 holdout.
# Stem-level filter active (0 leak verified).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.29.1 SAFETY LONG — 200 epochs, 24h walltime ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import os, glob, shutil, sys, json
from pathlib import Path
import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"
OUT = f"{REPO}/results/framework/mega_iterv3_0_29_safety_long"

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
print(f"Holdout stems (excluded from train): {len(holdout_stems)}")

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

yaml_path = Path(OUT) / "data.yaml"
yaml.safe_dump({
    "train": str(out_train_img),
    "val":   str(out_val_img),
    "nc":    12,
    "names": V3_NAMES,
}, open(yaml_path, "w"))

# Resume from v3.0.28 SAFETY best.pt to avoid wasting the first 30 epochs
PRETRAIN_BEST = (
    f"{REPO}/results/framework/mega_iterv3_0_28_safety/runs/safety/weights/best.pt"
)
if not os.path.exists(PRETRAIN_BEST):
    PRETRAIN_BEST = "yolo26x.pt"
print(f"Base: {PRETRAIN_BEST}")

model = YOLO(PRETRAIN_BEST)
model.train(
    data=str(yaml_path),
    epochs=200,
    imgsz=1024,
    batch=8,
    lr0=0.0003,           # lower LR since we're continuing from a good point
    lrf=0.01,
    patience=50,          # raise from 40 to give room
    device=0,
    project=f"{OUT}/runs",
    name="safety_long",
    verbose=False,
    save_period=5,
    cos_lr=True,
    mosaic=0.5,           # reduce aug intensity vs first run (was 1.0)
    mixup=0.05,           # very light
    close_mosaic=20,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5,
    fliplr=0.5,
)

print("\n=== v3.0.29.1 SAFETY LONG DONE ===")
weights = sorted(glob.glob(f"{OUT}/runs/safety_long*/weights/best.pt"),
                 key=os.path.getmtime, reverse=True)
print(f"best.pt: {weights[0] if weights else 'NONE'}")

import csv
res_path = sorted(glob.glob(f"{OUT}/runs/safety_long*/results.csv"),
                  key=os.path.getmtime, reverse=True)
if res_path:
    rows = list(csv.reader(open(res_path[0])))
    header = rows[0]
    last = rows[-1]
    summary = dict(zip(header, last))
    print(f"Final mAP50-95: {summary.get('metrics/mAP50-95(B)','?')}")
    json.dump(summary, open(f"{OUT}/v3_0_29_1_safety_long_summary.json","w"),
              indent=2, default=str)
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
