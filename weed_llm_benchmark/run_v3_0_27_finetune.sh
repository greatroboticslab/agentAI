#!/bin/bash
#SBATCH --job-name=v3027_FT
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/framework/v3_0_27_FT_%j.out

# v3.0.27 — KEY MISSING STEP: finetune the v3.0.26 pretrained model on
# cottonweeddet12 train (3,671 hand-labeled in-distribution images).
#
# Why this matters (audit revealed this):
#   v3.0.6 baseline: YOLO11n FT'd on cwd12 train (3,671) → mAP50-95 = 0.865
#   v3.0.26 ours:    yolo26x trained on 244K diverse data → mAP50-95 = 0.59
#   Gap = -0.27 because we trained on OOD data and tested on the
#   in-distribution cwd12 holdout.
#
# Fix: pretrain → finetune sequence (standard practice, arXiv:2505.01016
# shows +10% absolute mAP from deeper finetuning of pretrained backbone).
# Our v3.0.26 is RICHER pretraining than COCO-only (244K vs 118K, all
# weed/plant domain), so finetuning on cwd12 train should match or beat
# the v3.0.6 baseline of 0.865.
#
# NEVER_TRAIN discipline preserved:
#   - cwd12 train (3,671) → ALLOWED for training (these were always
#     fair game; my earlier blanket ban was over-conservative)
#   - cwd12 test (848) + valid (1,129) = 1,977 holdout → NEVER trained
#   - These are isolated by file stem at staging time
#
# Concurrent with Job-D3 (separate SLURM, harvesting more data for v3.0.28).

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

echo "=== v3.0.27 FINETUNE on cwd12 train (3,671 imgs) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import os, glob, shutil, sys, json
from pathlib import Path
import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"
OUT = f"{REPO}/results/framework/mega_iterv3_0_27_FT"

V3_NAMES = ["Carpetweeds","Crabgrass","PalmerAmaranth","PricklySida",
            "Purslane","Ragweed","Sicklepod","SpottedSpurge",
            "Eclipta","Goosegrass","Morningglory","Nutsedge"]
CWD12_ORIG = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass",
              "Morningglory","Nutsedge","PalmerAmaranth","PricklySida",
              "Purslane","Ragweed","Sicklepod","SpottedSpurge"]
ORIG_TO_CANON = {i: V3_NAMES.index(n) for i,n in enumerate(CWD12_ORIG)}

# Build train split: weedImages MINUS test+valid stems
out_train_img = Path(OUT) / "train" / "images"
out_train_lbl = Path(OUT) / "train" / "labels"
out_val_img   = Path(OUT) / "valid" / "images"
out_val_lbl   = Path(OUT) / "valid" / "labels"
for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

# Identify holdout stems (NEVER train on these)
holdout_stems = set()
for split in ("test", "valid"):
    sd = Path(CWD12_ROOT) / split / "images"
    if sd.is_dir():
        for img in sd.glob("*.jpg"):
            holdout_stems.add(img.stem)
print(f"Holdout stems (excluded from train): {len(holdout_stems)}")

# Train = weedImages where stem NOT in holdout
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

# Val = test + valid (1,977 hand-labeled holdout)
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

# data.yaml — nc=100 to match v3.0.26 model head
yaml_path = Path(OUT) / "data.yaml"
all_names = V3_NAMES + [f"aux_{i}" for i in range(12, 100)]
yaml.safe_dump({
    "train": str(out_train_img),
    "val":   str(out_val_img),
    "nc":    100,
    "names": all_names,
}, open(yaml_path, "w"))

# Find latest pretrained checkpoint (v3.0.26 phase 2 best is the highest mAP)
candidates = sorted(
    glob.glob(f"{REPO}/results/framework/mega_iterv3_0_26_phase_*/train*/weights/best.pt"),
    key=os.path.getmtime, reverse=True,
)
if not candidates:
    candidates = sorted(
        glob.glob(f"{REPO}/results/framework/mega_iterv3_0_2*/train*/weights/best.pt"),
        key=os.path.getmtime, reverse=True,
    )
PRETRAINED = candidates[0] if candidates else None
print(f"Pretrained base.pt: {PRETRAINED}")
if not PRETRAINED:
    sys.exit("FATAL: no pretrained best.pt found")

# Finetune
model = YOLO(PRETRAINED)
model.train(
    data=str(yaml_path),
    epochs=100,
    imgsz=1024,
    batch=8,
    lr0=0.0003,         # low — we're polishing, not relearning
    lrf=0.01,           # final LR is 1% of initial → cosine decay smooth
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

# Eval on holdout (val set IS the holdout — ultralytics auto-evals during)
print("\n=== FINETUNE COMPLETE ===")
weights = sorted(
    glob.glob(f"{OUT}/runs/finetune*/weights/best.pt"),
    key=os.path.getmtime, reverse=True,
)
print(f"best.pt: {weights[0] if weights else 'NONE'}")

# Save summary
import csv
res_path = sorted(glob.glob(f"{OUT}/runs/finetune*/results.csv"), key=os.path.getmtime, reverse=True)
if res_path:
    rows = list(csv.reader(open(res_path[0])))
    header = rows[0]
    last = rows[-1]
    summary = dict(zip(header, last))
    print(f"Final epoch: {summary.get('epoch','?')}")
    print(f"Final mAP50:    {summary.get('metrics/mAP50(B)','?')}")
    print(f"Final mAP50-95: {summary.get('metrics/mAP50-95(B)','?')}")
    json.dump(summary, open(f"{OUT}/v3_0_27_FT_summary.json","w"), indent=2, default=str)
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"
