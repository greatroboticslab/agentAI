#!/bin/bash
#SBATCH --job-name=v3_0_37_clc
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=results/framework/v3_0_37_yolo26x_cumulative_clean_%j.out

# v3.0.37 — yolo26x trained on cumulative cwd12 + Brain-collected data,
# AFTER DINOv2-based cleanup (51 garbage slugs auto-flagged at threshold=0.4).
#
# This is the v3.0.32 retry. v3.0.32 trained on 202K imgs raw -> pyco 0.5760
# (much worse than cwd12-only 0.7446). Hypothesis: autolabel noise + 35
# UNCATEGORIZED garbage slugs dominated the signal. With DINOv2 curator
# applied:
#   - Commonforms, mytwu, yonder, uvh_coco -> flagged garbage
#   - Plant disease + pest -> flagged garbage (off-domain for cwd12 task)
#   - Trusted + legit weed/crop datasets -> retained
#
# Expected merge corpus: 8 trusted + 3-5 legit + clean autolabel = 20-30K
# images (down from 202K, much cleaner).
#
# Goal: pyco mAP50-95 on cwd12 holdout > 0.8953 (current best, cwd12-only).
# Validates v3.0 thesis: autonomous discovery + cumulative training works
# WHEN data quality is properly filtered.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.37 yolo26x CUMULATIVE clean (DINOv2-filtered) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import os, sys, json
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.mega_trainer import (
    train_yolo_mega, _load_holdout_stems, NEVER_TRAIN_SLUGS
)

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"

# Sanity: stem-filter active
hs = _load_holdout_stems()
print(f"[v3.0.37] holdout stem filter: {len(hs)} stems loaded")
assert len(hs) >= 1900

# Sanity: DINOv2 flags applied
flags_path = f"{REPO}/results/framework/dataset_flags.json"
flags = json.load(open(flags_path))
auto = [s for s, v in flags.items()
        if isinstance(v, dict) and v.get("auto_flagged_by") == "dinov2_curator"]
manual = [s for s, v in flags.items()
          if isinstance(v, dict) and v.get("auto_flagged_by") != "dinov2_curator"]
print(f"[v3.0.37] garbage flags: {len(flags)} total ({len(auto)} DINOv2 auto + "
      f"{len(manual)} manual) — mega_trainer will skip all")

strategy = {
    "base_model": "yolo26x.pt",
    "fresh_start": True,
    "epochs": 30,
    "batch_size": 8,
    "imgsz": 896,
    "lr": 0.001,
    "patience": 10,
    "workers": 4,
    "include_autolabel": True,  # let CLEAN autolabel slugs in (rest already flagged)
    "val_dataset_root": CWD12_ROOT,
}
print(f"[v3.0.37] strategy: {json.dumps(strategy, indent=2)}")

best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_37_cumulative_clean")
print(f"\n=== v3.0.37 TRAIN COMPLETE ===")
print(json.dumps(summary, indent=2, default=str))

with open(f"{REPO}/results/framework/v3_0_37_best.txt", "w") as f:
    f.write(best_pt + "\n")
print(f"[v3.0.37] best.pt = {best_pt}")
PYEOF

TRAIN_EXIT=$?
echo "=== Train done (exit=$TRAIN_EXIT) ==="
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

# Pyco eval on cwd12 holdout
BEST_PT=$(cat $REPO/results/framework/v3_0_37_best.txt)
ls -la "$BEST_PT" || { echo "best.pt missing"; exit 1; }

OUT=$REPO/results/framework/v3_0_37_pyco
mkdir -p "$OUT"

python - <<PYEOF
import os, json, sys
sys.path.insert(0, ".")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pathlib import Path

REPO = "$REPO"
BEST_PT = "$BEST_PT"
OUT = f"{REPO}/results/framework/v3_0_37_pyco"

CANON = ["Carpetweeds","Crabgrass","PalmerAmaranth","PricklySida",
         "Purslane","Ragweed","Sicklepod","SpottedSpurge",
         "Eclipta","Goosegrass","Morningglory","Nutsedge"]
CWD12_ORIG = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass",
              "Morningglory","Nutsedge","PalmerAmaranth","PricklySida",
              "Purslane","Ragweed","Sicklepod","SpottedSpurge"]
ORIG_TO_CANON = {i: CANON.index(n) for i,n in enumerate(CWD12_ORIG)}

cwd12 = f"{REPO}/downloads/cottonweeddet12"
img_id = 1; ann_id = 1
images = []; anns = []; img_paths = {}
for split in ("test", "valid"):
    img_dir = Path(cwd12) / split / "images"
    lbl_dir = Path(cwd12) / split / "labels"
    if not img_dir.is_dir(): continue
    for img in sorted(img_dir.glob("*.jpg")):
        with Image.open(img) as im: w, h = im.size
        images.append({"id": img_id, "file_name": img.name, "width": w, "height": h})
        img_paths[img_id] = str(img)
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5: continue
                try:
                    orig = int(p[0]); cx,cy,bw,bh = map(float, p[1:5])
                except ValueError: continue
                if orig not in ORIG_TO_CANON: continue
                cid = ORIG_TO_CANON[orig]
                x = (cx - bw/2)*w; y = (cy - bh/2)*h
                anns.append({"id": ann_id, "image_id": img_id, "category_id": cid,
                             "bbox": [x, y, bw*w, bh*h], "area": bw*w*bh*h, "iscrowd": 0})
                ann_id += 1
        img_id += 1

categories = [{"id": i, "name": n, "supercategory": "weed"} for i,n in enumerate(CANON)]
gt = {"images": images, "annotations": anns, "categories": categories}
gt_path = f"{OUT}/cwd12_gt.json"
with open(gt_path, "w") as f: json.dump(gt, f)
print(f"[pyco] cwd12 holdout: {len(images)} imgs, {len(anns)} anns")

model = YOLO(BEST_PT)
preds = []
for iid, p in img_paths.items():
    res = model.predict(source=p, imgsz=896, conf=0.001, iou=0.6,
                        augment=False, verbose=False, device=0)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0: continue
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().tolist()
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    for j in range(len(xyxy)):
        if cls[j] >= 12: continue
        x1,y1,x2,y2 = xyxy[j]
        preds.append({"image_id": int(iid), "category_id": int(cls[j]),
                      "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                      "score": float(conf[j])})
print(f"[pyco] total preds: {len(preds)}")
pred_path = f"{OUT}/cwd12_pred.json"
with open(pred_path, "w") as f: json.dump(preds, f)

coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(pred_path)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
s = coco_eval.stats.tolist()
summary = {
    "weights": BEST_PT, "n_images": len(images),
    "n_annotations": len(anns), "n_predictions": len(preds),
    "mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
    "note": "v3.0.37 yolo26x cumulative AFTER DINOv2 cleanup (threshold=0.4)",
}
print()
print("=== v3.0.37 yolo26x CUMULATIVE CLEAN pyco ===")
print(f"  mAP50-95: {s[0]:.4f}")
print(f"  mAP50:    {s[1]:.4f}")
print(f"  mAP75:    {s[2]:.4f}")
out_path = f"{OUT}/v3_0_37_pyco_summary.json"
with open(out_path, "w") as f: json.dump(summary, f, indent=2)
print(f"[pyco] wrote {out_path}")
PYEOF

EVAL_EXIT=$?
echo "=== Done (eval_exit=$EVAL_EXIT) ==="
echo "Date: $(date)"
