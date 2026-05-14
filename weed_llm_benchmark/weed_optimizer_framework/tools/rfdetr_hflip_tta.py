"""
v3.0.30.9 — RF-DETR + horizontal-flip TTA, single-model.

RF-DETR Medium @576 alone gives pyco mAP50-95 = 0.8877. Goal 0.90, gap
-0.0123. Hflip TTA is the cheapest test-time augmentation: predict on the
original AND on the horizontally-flipped image, mirror the flipped boxes
back to original space, then WBF the two views' predictions.

Hflip is valid for cwd12 because top-down field photos of weeds have no
inherent left-right asymmetry. Expected gain: +0.005-0.015 mAP50-95.

If hflip TTA alone hits 0.90 → done.
If 0.89-0.90 → add multi-scale or pivot to RFDETRLarge.
If still < 0.89 → hflip didn't help; pivot to RFDETRLarge.

Note: rfdetr's predict() doesn't take a resolution arg, so we can't
multi-scale via the public API without internal hacks. Hflip is the only
TTA that works through the public predict() interface.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _patch_supervision_for_rfdetr():
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


def predict_rfdetr_norm(model, img_path_or_pil, w, h, threshold):
    """Run rfdetr predict, return normalized xyxy + conf + cls lists."""
    try:
        if isinstance(img_path_or_pil, (str, Path)):
            det = model.predict(str(img_path_or_pil), threshold=threshold)
        else:
            det = model.predict(img_path_or_pil, threshold=threshold)
    except Exception as e:
        return [], [], []
    xyxy = det.xyxy if hasattr(det, "xyxy") else None
    conf = det.confidence if hasattr(det, "confidence") else None
    cls = det.class_id if hasattr(det, "class_id") else None
    if xyxy is None or len(xyxy) == 0:
        return [], [], []
    boxes = []
    confs = []
    classes = []
    for j in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[j]
        boxes.append([float(x1) / w, float(y1) / h,
                      float(x2) / w, float(y2) / h])
        confs.append(float(conf[j]))
        classes.append(int(cls[j]))
    return boxes, confs, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfdetr-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument("--wbf-skip", type=float, default=0.001)
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cwd12_root = Path(args.cwd12).resolve()
    dataset_dir = out_root / "dataset"

    # Reuse staging
    valid_ann = dataset_dir / "valid" / "_annotations.coco.json"
    test_ann = dataset_dir / "test" / "_annotations.coco.json"
    if not (valid_ann.exists() and test_ann.exists()):
        from weed_optimizer_framework.tools.train_rfdetr import stage_dataset
        stage_dataset(dataset_dir, cwd12_root)

    # Combined GT
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
    if not combined_dir.exists():
        combined_dir.mkdir(parents=True)
        for split in ("valid", "test"):
            for f in (dataset_dir / split).glob("*.jpg"):
                link = combined_dir / f.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                os.symlink(f.resolve(), link)

    file_id_map = {im["file_name"]: im["id"] for im in combined["images"]}
    gt_path = out_root / "tta_gt.json"
    with open(gt_path, "w") as f:
        json.dump(combined, f)

    # Load model
    print(f"[tta] loading RF-DETR from {args.rfdetr_weights}")
    from rfdetr import RFDETRMedium
    model = RFDETRMedium(pretrain_weights=str(args.rfdetr_weights))

    from ensemble_boxes import weighted_boxes_fusion

    img_files = sorted(combined_dir.glob("*.jpg"))
    print(f"[tta] {len(img_files)} images")

    preds = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="rfdetr_hflip_"))

    for i, img_path in enumerate(img_files):
        with Image.open(img_path) as im:
            w, h = im.size

        # Original prediction
        orig_boxes, orig_conf, orig_cls = predict_rfdetr_norm(
            model, img_path, w, h, args.threshold
        )

        # Hflip prediction
        with Image.open(img_path) as im:
            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
        flip_path = tmp_dir / img_path.name
        im_flip.save(flip_path, quality=95)
        flip_boxes, flip_conf, flip_cls = predict_rfdetr_norm(
            model, flip_path, w, h, args.threshold
        )
        # Mirror back: x' = 1 - x  (in normalized space)
        flip_boxes_mirrored = [
            [1.0 - x2, y1, 1.0 - x1, y2] for (x1, y1, x2, y2) in flip_boxes
        ]
        # Cleanup tmp file
        try:
            os.unlink(flip_path)
        except Exception:
            pass

        # WBF the two views
        if not orig_boxes and not flip_boxes_mirrored:
            continue
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        if orig_boxes:
            boxes_list.append(orig_boxes)
            scores_list.append(orig_conf)
            labels_list.append(orig_cls)
            weights.append(1.0)
        if flip_boxes_mirrored:
            boxes_list.append(flip_boxes_mirrored)
            scores_list.append(flip_conf)
            labels_list.append(flip_cls)
            weights.append(1.0)

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
            print(f"[tta] {i+1}/{len(img_files)}  preds={len(preds)}")

    print(f"[tta] DONE: total preds={len(preds)}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    pred_path = out_root / "tta_pred.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    # pycocotools eval
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
        "weights": str(args.rfdetr_weights),
        "tta_strategy": "hflip",
        "wbf_iou": args.wbf_iou,
        "wbf_skip": args.wbf_skip,
        "n_images": len(combined["images"]),
        "n_annotations": len(combined["annotations"]),
        "n_predictions": len(preds),
        "mAP50_95": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
    }
    print(f"\n=== RF-DETR HFLIP TTA pyco ===")
    print(f"  mAP50-95: {s[0]:.4f}")
    print(f"  mAP50:    {s[1]:.4f}")
    print(f"  mAP75:    {s[2]:.4f}")
    out_path = out_root / "tta_pyco_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[tta] wrote {out_path}")


if __name__ == "__main__":
    main()
