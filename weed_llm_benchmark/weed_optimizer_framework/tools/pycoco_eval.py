"""
v3.0.29 — Independent pycocotools-based mAP cross-check.

pycocotools is the gold standard for COCO-style mAP. Use it as third-party
adjudicator on the cwd12 holdout to settle which number is canonical:
  - 0.8953 (ultralytics .val() on safety best.pt)
  - 0.744  (our custom WBF/TTA pipeline)

Procedure:
  1. Convert cwd12 holdout YOLO labels → COCO ground-truth JSON.
  2. Run safety best.pt at imgsz=1024 single-scale (no TTA) — match the
     ultralytics single-scale baseline.
  3. Write predictions in COCO results JSON format.
  4. COCOeval(annType='bbox') → official mAP50, mAP50-95.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]


def yolo_norm_to_coco_xywh(cx, cy, bw, bh, w, h):
    """Convert normalized YOLO (cx, cy, bw, bh) to COCO pixel xywh."""
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return [float(x), float(y), float(bw * w), float(bh * h)]


def build_gt_coco(val_imgs_dir: Path, val_lbls_dir: Path) -> dict:
    """Build COCO ground-truth dict from YOLO-format val split."""
    images = []
    annotations = []
    categories = [{"id": i, "name": n, "supercategory": "weed"}
                  for i, n in enumerate(CANONICAL_12)]
    img_id_map = {}
    ann_id = 1
    for i, img_path in enumerate(sorted(val_imgs_dir.glob("*.jpg")), start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        img_id_map[img_path.stem] = i
        images.append({
            "id": i,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })
        lbl = val_lbls_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            if cid >= len(CANONICAL_12):
                continue  # only score the 12 weed classes
            xywh = yolo_norm_to_coco_xywh(cx, cy, bw, bh, w, h)
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": cid,
                "bbox": xywh,
                "area": float(xywh[2] * xywh[3]),
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }, img_id_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-imgs", required=True)
    ap.add_argument("--val-lbls", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val_imgs_dir = Path(args.val_imgs)
    val_lbls_dir = Path(args.val_lbls)

    print(f"[pycoco] building GT JSON from {val_imgs_dir}")
    gt_dict, img_id_map = build_gt_coco(val_imgs_dir, val_lbls_dir)
    n_imgs = len(gt_dict["images"])
    n_anns = len(gt_dict["annotations"])
    print(f"[pycoco] images={n_imgs}  annotations={n_anns}  cats={len(gt_dict['categories'])}")

    gt_path = Path(args.out).parent / "v3_0_29_pycoco_gt.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(gt_dict, f)
    print(f"[pycoco] wrote GT → {gt_path}")

    print(f"[pycoco] loading model {args.weights}")
    model = YOLO(args.weights)

    # Predict & accumulate
    preds_coco = []
    img_files = sorted(val_imgs_dir.glob("*.jpg"))
    for k, img_path in enumerate(img_files):
        res = model.predict(source=str(img_path), imgsz=args.imgsz,
                            conf=args.conf, iou=args.iou,
                            augment=False, verbose=False, device=0)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        h, w = res.orig_shape
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int32)
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            preds_coco.append({
                "image_id": img_id_map[img_path.stem],
                "category_id": int(cls[j]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[j]),
            })
        if (k + 1) % 200 == 0:
            print(f"[pycoco] predicted {k+1}/{len(img_files)}")
    print(f"[pycoco] total predictions: {len(preds_coco)}")

    pred_path = Path(args.out).parent / "v3_0_29_pycoco_pred.json"
    with open(pred_path, "w") as f:
        json.dump(preds_coco, f)

    # COCOeval
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Stats: [AP @ [.5:.95], AP@.5, AP@.75, AP-small, AP-medium, AP-large,
    #        AR@1, AR@10, AR@100, AR-small, AR-medium, AR-large]
    s = coco_eval.stats.tolist()
    out = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "n_images": n_imgs,
        "n_annotations": n_anns,
        "n_predictions": len(preds_coco),
        "mAP50_95": float(s[0]),
        "mAP50":    float(s[1]),
        "mAP75":    float(s[2]),
        "mAR_max1":  float(s[6]),
        "mAR_max10": float(s[7]),
        "mAR_max100": float(s[8]),
        "all_stats": s,
    }
    print(f"\n=== pycocotools (canonical) ===")
    print(f"  mAP50-95: {s[0]:.4f}")
    print(f"  mAP50:    {s[1]:.4f}")
    print(f"  mAP75:    {s[2]:.4f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[pycoco] wrote {args.out}")


if __name__ == "__main__":
    main()
