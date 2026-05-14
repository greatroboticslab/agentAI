"""
v3.0.30.7 — RF-DETR + yolo26x WBF ensemble evaluation on cwd12 holdout.

RF-DETR alone (v3.0.29) = 0.8877 pyco mAP50-95.
yolo26x SAFETY (v3.0.28) = 0.7446 pyco mAP50-95.
Goal: ensemble pushes ≥ 0.90 (gap to close = -0.0123).

Per WBF (Solovyev 2021), confidence-weighted box fusion across DIFFERENT
architectures consistently beats either alone (+0.01-0.04 typical) when
their failure modes are independent. RF-DETR (DINOv2 + transformer decoder)
and yolo26x (CNN + anchor-free) have very different inductive biases, so
the ensemble assumption holds.

Pipeline per image:
  1. RF-DETR predict (threshold=0.001) → xyxy, conf, cls
  2. YOLO predict (conf=0.001) → xyxy, conf, cls
  3. Normalize both to [0,1] xyxy
  4. WBF fuse (iou_thr=0.55, skip_box_thr=0.001), weights = [2, 1]
     (RF-DETR weighted higher because its standalone pyco is +0.143 better)
  5. Denormalize → COCO format → pycocotools eval on cwd12 valid+test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _patch_supervision_for_rfdetr():
    """Same monkey-patch as train_rfdetr.py — supervision 0.6 needs metadata/data."""
    import supervision as sv
    if getattr(sv.Detections, "_rfdetr_patched", False):
        return
    _orig_init = sv.Detections.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, "metadata") or self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if not hasattr(self, "data") or self.data is None:
            object.__setattr__(self, "data", {})

    sv.Detections.__init__ = _patched_init
    sv.Detections._rfdetr_patched = True


_patch_supervision_for_rfdetr()


def predict_rfdetr(model, img_path, w, h, threshold=0.001):
    """Run RF-DETR on one image. Returns (xyxy_norm, conf, cls) or empty lists."""
    try:
        det = model.predict(str(img_path), threshold=threshold)
    except Exception as e:
        print(f"  [rfdetr] pred fail on {img_path.name}: {e}")
        return [], [], []
    xyxy = det.xyxy if hasattr(det, "xyxy") else None
    conf = det.confidence if hasattr(det, "confidence") else None
    cls = det.class_id if hasattr(det, "class_id") else None
    if xyxy is None or len(xyxy) == 0:
        return [], [], []
    boxes = []
    for j in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[j]
        boxes.append([
            float(x1) / w, float(y1) / h,
            float(x2) / w, float(y2) / h,
        ])
    return boxes, conf.tolist(), cls.tolist()


def predict_yolo(model, img_path, threshold=0.001, imgsz=1024):
    """Run YOLO on one image. Returns (xyxy_norm, conf, cls) or empty lists."""
    res = model.predict(source=str(img_path), imgsz=imgsz, conf=threshold,
                        iou=0.6, augment=False, verbose=False, device=0)[0]
    boxes_obj = res.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return [], [], []
    h, w = res.orig_shape
    xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
    xyxy[:, [0, 2]] /= w
    xyxy[:, [1, 3]] /= h
    conf = boxes_obj.conf.cpu().numpy().astype(np.float32).tolist()
    cls = boxes_obj.cls.cpu().numpy().astype(np.int32).tolist()
    # Clamp to [0,1]
    xyxy = np.clip(xyxy, 0.0, 1.0)
    return xyxy.tolist(), conf, cls


def stage_dataset_if_needed(dataset_dir, cwd12_root):
    """Reuse the same staging from train_rfdetr.py if combined dir doesn't exist."""
    valid_ann = dataset_dir / "valid" / "_annotations.coco.json"
    test_ann = dataset_dir / "test" / "_annotations.coco.json"
    if valid_ann.exists() and test_ann.exists():
        return
    print("[stage] combined dataset not found — staging via train_rfdetr.stage_dataset")
    from weed_optimizer_framework.tools.train_rfdetr import stage_dataset
    stage_dataset(dataset_dir, cwd12_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfdetr-weights", required=True)
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--out", required=True, help="output dir for eval artifacts")
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--rfdetr-resolution", type=int, default=576,
                    help="RF-DETR was trained at 576")
    ap.add_argument("--yolo-imgsz", type=int, default=1024,
                    help="yolo26x SAFETY was trained at 1024")
    ap.add_argument("--rfdetr-weight", type=float, default=2.0,
                    help="WBF weight for RF-DETR boxes (stronger model gets more)")
    ap.add_argument("--yolo-weight", type=float, default=1.0)
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument("--wbf-skip", type=float, default=0.001)
    ap.add_argument("--threshold", type=float, default=0.001,
                    help="per-model conf threshold before WBF")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cwd12_root = Path(args.cwd12).resolve()

    # 1. Stage / re-use the combined cwd12 holdout dataset (1977 imgs)
    dataset_dir = out_root / "dataset"
    stage_dataset_if_needed(dataset_dir, cwd12_root)

    # 2. Build a combined GT JSON (val+test merged, same logic as eval_canonical)
    val_coco = json.load(open(dataset_dir / "valid" / "_annotations.coco.json"))
    test_coco = json.load(open(dataset_dir / "test" / "_annotations.coco.json"))
    combined = {
        "images": list(val_coco["images"]),
        "annotations": list(val_coco["annotations"]),
        "categories": val_coco["categories"],
    }
    img_id_offset = max((im["id"] for im in val_coco["images"]), default=0)
    ann_id_offset = max((a["id"] for a in val_coco["annotations"]), default=0)
    img_id_remap = {}
    for im in test_coco["images"]:
        new_id = im["id"] + img_id_offset
        img_id_remap[im["id"]] = new_id
        nim = dict(im); nim["id"] = new_id
        combined["images"].append(nim)
    for a in test_coco["annotations"]:
        new_id = a["id"] + ann_id_offset
        na = dict(a); na["id"] = new_id
        na["image_id"] = img_id_remap[a["image_id"]]
        combined["annotations"].append(na)

    combined_dir = out_root / "eval_combined"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True)
    file_id_map = {}
    for split in ("valid", "test"):
        for f in (dataset_dir / split).glob("*.jpg"):
            link = combined_dir / f.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(f.resolve(), link)
    for im in combined["images"]:
        file_id_map[im["file_name"]] = im["id"]

    gt_path = out_root / "ensemble_gt.json"
    with open(gt_path, "w") as f:
        json.dump(combined, f)
    print(f"[ensemble] combined: {len(combined['images'])} imgs, "
          f"{len(combined['annotations'])} anns")

    # 3. Load both models
    print(f"[ensemble] loading RF-DETR from {args.rfdetr_weights}")
    from rfdetr import RFDETRMedium
    rfdetr_model = RFDETRMedium(pretrain_weights=str(args.rfdetr_weights))

    print(f"[ensemble] loading yolo26x from {args.yolo_weights}")
    from ultralytics import YOLO
    yolo_model = YOLO(str(args.yolo_weights))

    # 4. WBF lib
    from ensemble_boxes import weighted_boxes_fusion

    img_files = sorted(combined_dir.glob("*.jpg"))
    print(f"[ensemble] {len(img_files)} images to process")

    preds = []
    n_rfdetr_only = 0
    n_yolo_only = 0
    n_both = 0
    n_neither = 0
    for i, img_path in enumerate(img_files):
        # Image dims
        with Image.open(img_path) as im:
            w, h = im.size

        # RF-DETR
        rd_box, rd_conf, rd_cls = predict_rfdetr(
            rfdetr_model, img_path, w, h, threshold=args.threshold
        )
        # YOLO
        yo_box, yo_conf, yo_cls = predict_yolo(
            yolo_model, img_path, threshold=args.threshold,
            imgsz=args.yolo_imgsz
        )

        if rd_box and yo_box:
            n_both += 1
        elif rd_box:
            n_rfdetr_only += 1
        elif yo_box:
            n_yolo_only += 1
        else:
            n_neither += 1
            continue

        # Build WBF inputs (lists of lists per model)
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        if rd_box:
            boxes_list.append(rd_box)
            scores_list.append(rd_conf)
            labels_list.append(rd_cls)
            weights.append(args.rfdetr_weight)
        if yo_box:
            boxes_list.append(yo_box)
            scores_list.append(yo_conf)
            labels_list.append(yo_cls)
            weights.append(args.yolo_weight)

        # WBF
        try:
            f_box, f_score, f_label = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=args.wbf_iou,
                skip_box_thr=args.wbf_skip,
            )
        except Exception as e:
            print(f"  [wbf] fail on {img_path.name}: {e}")
            continue

        img_id = file_id_map[img_path.name]
        for j in range(len(f_box)):
            x1n, y1n, x2n, y2n = f_box[j]
            x1, y1, x2, y2 = x1n * w, y1n * h, x2n * w, y2n * h
            preds.append({
                "image_id": int(img_id),
                "category_id": int(f_label[j]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(f_score[j]),
            })

        if (i + 1) % 200 == 0:
            print(f"[ensemble] {i+1}/{len(img_files)}  preds={len(preds)}  "
                  f"both={n_both} rd_only={n_rfdetr_only} "
                  f"yo_only={n_yolo_only} neither={n_neither}")

    print(f"[ensemble] DONE: total preds={len(preds)}")
    print(f"  both={n_both} rd_only={n_rfdetr_only} "
          f"yo_only={n_yolo_only} neither={n_neither}")

    pred_path = out_root / "ensemble_pred.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    # 5. pycocotools eval
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    s = coco_eval.stats.tolist()

    summary = {
        "rfdetr_weights": str(args.rfdetr_weights),
        "yolo_weights": str(args.yolo_weights),
        "rfdetr_resolution": args.rfdetr_resolution,
        "yolo_imgsz": args.yolo_imgsz,
        "wbf_weights": [args.rfdetr_weight, args.yolo_weight],
        "wbf_iou": args.wbf_iou,
        "wbf_skip": args.wbf_skip,
        "n_images": len(combined["images"]),
        "n_annotations": len(combined["annotations"]),
        "n_predictions": len(preds),
        "img_overlap": {"both": n_both, "rd_only": n_rfdetr_only,
                        "yo_only": n_yolo_only, "neither": n_neither},
        "mAP50_95": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
    }
    print(f"\n=== RF-DETR + yolo26x WBF ENSEMBLE pyco ===")
    print(f"  mAP50-95: {s[0]:.4f}")
    print(f"  mAP50:    {s[1]:.4f}")
    print(f"  mAP75:    {s[2]:.4f}")
    out_path = out_root / "ensemble_pyco_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ensemble] wrote {out_path}")


if __name__ == "__main__":
    main()
