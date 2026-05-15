#!/bin/bash
#SBATCH --job-name=v3_0_32_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/v3_0_32_pyco_eval_%j.out

# v3.0.32 PYCO EVAL ONLY — train (job 40839159) timed out at epoch 7/30
# but produced best.pt. Evaluate it now to get the data-scaling datapoint
# (yolo26x trained on 202K cumulative imgs).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.32 PYCO eval ONLY ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"

BEST_PT=$REPO/results/framework/mega_iterv3_0_32_cumulative/train/weights/best.pt
ls -la "$BEST_PT" || { echo "best.pt missing"; exit 1; }

OUT=$REPO/results/framework/v3_0_32_pyco
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
OUT = f"{REPO}/results/framework/v3_0_32_pyco"

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
        if cls[j] >= 12: continue  # drop aux 12-99
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
    "note": "v3.0.32 yolo26x trained 7/30 epochs on 202K cumulative imgs (TIMEOUT)",
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

echo "=== Done (exit=$?) ==="
echo "Date: $(date)"
SUMMARY=$REPO/results/framework/v3_0_32_pyco/v3_0_32_pyco_summary.json
[ -f "$SUMMARY" ] && cat "$SUMMARY"
