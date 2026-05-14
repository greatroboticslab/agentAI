#!/bin/bash
#SBATCH --job-name=v3_0_32_cum
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=results/framework/v3_0_32_yolo26x_cumulative_%j.out

# v3.0.32 — yolo26x trained on cwd12 + Brain-collected cumulative data.
#
# WHY THIS EXISTS:
# Per user audit 2026-05-14: training has been using cwd12-train-only
# (3,671 imgs) since v3.0.28 SAFETY (post v3.0.27 leak retraction). The
# Brain has accumulated ~1.5M images / 73 slugs / 29 real-bbox slugs but
# NONE of it has entered training in 6 days. This violates REQ-2 (massive
# autonomous data) + REQ-4 (cumulative training) — the v3.0 north star.
#
# This run actually USES the cumulative corpus:
#   - mega_trainer._merge_datasets gathers all real-bbox + autolabel slugs
#   - Triple defense ensures no leak:
#       1. NEVER_TRAIN_SLUGS skips cottonweeddet12 / weedsense / francesco
#       2. v3.0.28 stem-level filter blocks cwd12 holdout stems even from
#          legitimate cottonweed_sp8 / cottonweed_holdout slugs
#       3. user dataset_flags.json garbage skip
#   - val_dataset_root=cwd12 → ultralytics val on cwd12 holdout (honest)
#   - After training, pycoco_eval re-validates the best.pt independently
#     (because v3.0.27 0.910 was retracted when ult-val ≠ pyco eval)
#
# Compare to v3.0.31 RFDETRLarge (40839152) running in parallel:
#   - v3.0.31: tests architecture scaling (87M Large vs 33M Medium, cwd12 only)
#   - v3.0.32: tests data scaling (cumulative ~100K-1.5M, yolo26x base)
# Together: 2 orthogonal levers measured cleanly.

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

echo "=== v3.0.32 yolo26x CUMULATIVE train ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python - <<'PYEOF'
import os, sys, json, glob
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.mega_trainer import (
    train_yolo_mega, _load_holdout_stems, NEVER_TRAIN_SLUGS
)

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
CWD12_ROOT = f"{REPO}/downloads/cottonweeddet12"

# Sanity: stem-filter active
hs = _load_holdout_stems()
print(f"[v3.0.32] holdout stem filter: {len(hs)} stems loaded")
assert len(hs) >= 1900, f"holdout stem filter broken (got {len(hs)})"
print(f"[v3.0.32] NEVER_TRAIN_SLUGS = {sorted(NEVER_TRAIN_SLUGS)}")

# Strategy: yolo26x on cumulative real_bbox + autolabel, eval on cwd12 holdout.
# Differences from v3.0.28 PRETRAIN:
#   - epochs=30 with patience=10 (fits 36h walltime safely)
#   - batch=6 imgsz=896 (smaller imgsz than 1024 for throughput; 896 still
#     captures detail vs 640)
#   - fresh_start=True (cannot trust v3.0.27 contaminated checkpoints)
strategy = {
    "base_model": "yolo26x.pt",
    "fresh_start": True,
    "epochs": 30,
    "batch_size": 6,
    "imgsz": 896,
    "lr": 0.001,
    "patience": 10,
    "workers": 4,
    "include_autolabel": True,
    "val_dataset_root": CWD12_ROOT,
}
print("[v3.0.32] strategy:", json.dumps(strategy, indent=2))

best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_32_cumulative")
print("\n=== v3.0.32 CUMULATIVE TRAIN COMPLETE ===")
print(json.dumps(summary, indent=2, default=str))

# Persist best.pt path so the eval step can find it
with open(f"{REPO}/results/framework/v3_0_32_best.txt", "w") as f:
    f.write(best_pt + "\n")
print(f"[v3.0.32] best.pt = {best_pt}")
PYEOF

TRAIN_EXIT=$?
echo "=== Train done (exit=$TRAIN_EXIT) ==="
echo "Date: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Train failed; skipping eval"
    exit $TRAIN_EXIT
fi

# Pycocotools eval (canonical metric, NOT ultralytics generous val)
echo ""
echo "=== v3.0.32 PYCOCOTOOLS EVAL (canonical) ==="
BEST_PT=$(cat $REPO/results/framework/v3_0_32_best.txt)
ls -la "$BEST_PT" || { echo "best.pt missing"; exit 1; }

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
OUT = f"{REPO}/results/framework/v3_0_32_pyco"
os.makedirs(OUT, exist_ok=True)

# Build cwd12 holdout COCO GT (1977 imgs, valid+test stem-filtered)
CANON = ["Carpetweeds","Crabgrass","PalmerAmaranth","PricklySida",
         "Purslane","Ragweed","Sicklepod","SpottedSpurge",
         "Eclipta","Goosegrass","Morningglory","Nutsedge"]
CWD12_ORIG = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass",
              "Morningglory","Nutsedge","PalmerAmaranth","PricklySida",
              "Purslane","Ragweed","Sicklepod","SpottedSpurge"]
ORIG_TO_CANON = {i: CANON.index(n) for i,n in enumerate(CWD12_ORIG)}

cwd12 = f"{REPO}/downloads/cottonweeddet12"
img_id = 1
ann_id = 1
images = []
anns = []
img_paths = {}
for split in ("test", "valid"):
    img_dir = Path(cwd12) / split / "images"
    lbl_dir = Path(cwd12) / split / "labels"
    if not img_dir.is_dir(): continue
    for img in sorted(img_dir.glob("*.jpg")):
        with Image.open(img) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img.name, "width": w, "height": h})
        img_paths[img_id] = str(img)
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5: continue
                try:
                    orig = int(p[0])
                    cx,cy,bw,bh = map(float, p[1:5])
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

# Run yolo predict
model = YOLO(BEST_PT)
preds = []
for img_id, p in img_paths.items():
    res = model.predict(source=p, imgsz=896, conf=0.001, iou=0.6,
                        augment=False, verbose=False, device=0)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0: continue
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().tolist()
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    for j in range(len(xyxy)):
        x1,y1,x2,y2 = xyxy[j]
        # Only keep canonical 12 weed classes (drop aux 12-99 if any)
        if cls[j] >= 12: continue
        preds.append({"image_id": int(img_id), "category_id": int(cls[j]),
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
    "weights": BEST_PT,
    "n_images": len(images),
    "n_annotations": len(anns),
    "n_predictions": len(preds),
    "mAP50_95": float(s[0]),
    "mAP50": float(s[1]),
    "mAP75": float(s[2]),
}
print()
print("=== v3.0.32 yolo26x CUMULATIVE pyco ===")
print(f"  mAP50-95: {s[0]:.4f}")
print(f"  mAP50:    {s[1]:.4f}")
print(f"  mAP75:    {s[2]:.4f}")
out_path = f"{OUT}/v3_0_32_pyco_summary.json"
with open(out_path, "w") as f: json.dump(summary, f, indent=2)
print(f"[pyco] wrote {out_path}")
PYEOF

EVAL_EXIT=$?
echo "=== Done (eval_exit=$EVAL_EXIT) ==="
echo "Date: $(date)"
