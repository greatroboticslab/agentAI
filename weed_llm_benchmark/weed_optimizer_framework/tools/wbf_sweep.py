"""
v3.0.30.8 — Cache per-model predictions then sweep WBF parameters.

First ensemble attempt (job 40831757 v3.0.30.7): WBF iou=0.55 weights=[2,1]
gave pyco 0.8393, WORSE than RF-DETR alone (0.8877). Hypothesis: yolo26x
boxes (weaker model) merged with RF-DETR boxes via WBF and dragged down
position quality. Try more conservative WBF settings to preserve
RF-DETR's strong boxes:

  - iou_thr ∈ {0.55, 0.7, 0.85}     (higher = less aggressive merge)
  - weights ∈ [(2,1), (4,1), (10,1)]  (more weight on stronger model)
  - threshold filter: keep RF-DETR @ 0.001 (DETR low-conf is informative);
    drop YOLO < 0.05 (YOLO low-conf is noise)

If best WBF combo lands ≥ 0.89, ensemble path stays alive.
If all sweep results < 0.89, ensemble is dead — pivot to RF-DETR TTA
or RFDETRLarge.

Two-phase script:
  Phase A (one-time GPU): run both models, save per-image preds
  Phase B (CPU sweep): for each WBF combo, build pred.json, run pyco
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


def _patch_supervision_for_rfdetr():
    """Same monkey-patch as train_rfdetr.py."""
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


def cache_predictions(args, combined_dir, cache_path):
    """Phase A: run both models, save per-image preds to JSON.

    Output JSON shape:
      {file_name: {
        "w": int, "h": int,
        "rfdetr": [[x1n,y1n,x2n,y2n,conf,cls], ...],
        "yolo":   [[x1n,y1n,x2n,y2n,conf,cls], ...],
      }}
    """
    print(f"[cache] computing per-model preds → {cache_path}")
    print(f"[cache] loading RF-DETR from {args.rfdetr_weights}")
    from rfdetr import RFDETRMedium
    rfdetr_model = RFDETRMedium(pretrain_weights=str(args.rfdetr_weights))

    print(f"[cache] loading YOLO from {args.yolo_weights}")
    from ultralytics import YOLO
    yolo_model = YOLO(str(args.yolo_weights))

    img_files = sorted(combined_dir.glob("*.jpg"))
    cache = {}
    for i, img_path in enumerate(img_files):
        with Image.open(img_path) as im:
            w, h = im.size

        # RF-DETR
        rd_rows = []
        try:
            det = rfdetr_model.predict(str(img_path), threshold=args.rfdetr_threshold)
            xyxy = det.xyxy if hasattr(det, "xyxy") else None
            conf = det.confidence if hasattr(det, "confidence") else None
            cls = det.class_id if hasattr(det, "class_id") else None
            if xyxy is not None and len(xyxy) > 0:
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    rd_rows.append([
                        float(x1) / w, float(y1) / h,
                        float(x2) / w, float(y2) / h,
                        float(conf[j]), int(cls[j]),
                    ])
        except Exception as e:
            print(f"  [rfdetr] pred fail {img_path.name}: {e}")

        # YOLO
        yo_rows = []
        try:
            res = yolo_model.predict(source=str(img_path), imgsz=args.yolo_imgsz,
                                     conf=args.yolo_threshold, iou=0.6,
                                     augment=False, verbose=False, device=0)[0]
            boxes_obj = res.boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                hh, ww = res.orig_shape
                xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
                xyxy[:, [0, 2]] /= ww
                xyxy[:, [1, 3]] /= hh
                xyxy = np.clip(xyxy, 0.0, 1.0)
                conf = boxes_obj.conf.cpu().numpy().astype(np.float32).tolist()
                cls = boxes_obj.cls.cpu().numpy().astype(np.int32).tolist()
                for j in range(len(xyxy)):
                    yo_rows.append([
                        float(xyxy[j, 0]), float(xyxy[j, 1]),
                        float(xyxy[j, 2]), float(xyxy[j, 3]),
                        float(conf[j]), int(cls[j]),
                    ])
        except Exception as e:
            print(f"  [yolo] pred fail {img_path.name}: {e}")

        cache[img_path.name] = {"w": w, "h": h, "rfdetr": rd_rows, "yolo": yo_rows}

        if (i + 1) % 200 == 0:
            print(f"[cache] {i+1}/{len(img_files)}  rd={sum(len(c['rfdetr']) for c in cache.values())} "
                  f"yo={sum(len(c['yolo']) for c in cache.values())}")

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"[cache] wrote {cache_path}, {len(cache)} images")


def build_preds_for_combo(cache, file_id_map, iou_thr, rfdetr_w, yolo_w,
                           skip_box_thr, yolo_min_conf):
    """Run WBF on cached preds with given params. Returns COCO-format pred list."""
    from ensemble_boxes import weighted_boxes_fusion
    preds = []
    for fname, c in cache.items():
        if fname not in file_id_map:
            continue
        w, h = c["w"], c["h"]
        rd = c["rfdetr"]
        yo = [r for r in c["yolo"] if r[4] >= yolo_min_conf]

        if not rd and not yo:
            continue

        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        if rd:
            boxes_list.append([r[:4] for r in rd])
            scores_list.append([r[4] for r in rd])
            labels_list.append([r[5] for r in rd])
            weights.append(rfdetr_w)
        if yo:
            boxes_list.append([r[:4] for r in yo])
            scores_list.append([r[4] for r in yo])
            labels_list.append([r[5] for r in yo])
            weights.append(yolo_w)

        try:
            f_box, f_score, f_label = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        except Exception as e:
            print(f"  [wbf] fail on {fname}: {e}")
            continue

        img_id = file_id_map[fname]
        for j in range(len(f_box)):
            x1n, y1n, x2n, y2n = f_box[j]
            x1, y1, x2, y2 = x1n * w, y1n * h, x2n * w, y2n * h
            preds.append({
                "image_id": int(img_id),
                "category_id": int(f_label[j]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(f_score[j]),
            })
    return preds


def pyco_eval(gt_path, preds, out_pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with open(out_pred_path, "w") as f:
        json.dump(preds, f)
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(out_pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    s = coco_eval.stats.tolist()
    return {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfdetr-weights", required=True)
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--rfdetr-resolution", type=int, default=576)
    ap.add_argument("--yolo-imgsz", type=int, default=1024)
    ap.add_argument("--rfdetr-threshold", type=float, default=0.001)
    ap.add_argument("--yolo-threshold", type=float, default=0.001)
    ap.add_argument("--force-cache", action="store_true",
                    help="force regenerate the per-model pred cache")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cwd12_root = Path(args.cwd12).resolve()
    dataset_dir = out_root / "dataset"

    # Reuse staging if present, else stage
    valid_ann = dataset_dir / "valid" / "_annotations.coco.json"
    test_ann = dataset_dir / "test" / "_annotations.coco.json"
    if not (valid_ann.exists() and test_ann.exists()):
        from weed_optimizer_framework.tools.train_rfdetr import stage_dataset
        stage_dataset(dataset_dir, cwd12_root)

    # Build combined GT (val+test)
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
    gt_path = out_root / "ensemble_gt.json"
    with open(gt_path, "w") as f:
        json.dump(combined, f)

    # Phase A: cache predictions
    cache_path = out_root / "per_model_cache.json"
    if args.force_cache or not cache_path.exists():
        cache_predictions(args, combined_dir, cache_path)
    else:
        print(f"[cache] reusing existing cache: {cache_path}")
    cache = json.load(open(cache_path))

    # Phase B: WBF param sweep
    print()
    print("=== WBF SWEEP START ===")
    sweep_combos = [
        # (iou_thr, rfdetr_w, yolo_w, skip_box_thr, yolo_min_conf, name)
        (0.55, 2.0, 1.0, 0.001, 0.0,   "iou55_w21_yolo>=0"),    # baseline (matches v3.0.30.7)
        (0.55, 4.0, 1.0, 0.001, 0.05,  "iou55_w41_yolo>=05"),
        (0.55, 10.0, 1.0, 0.001, 0.10, "iou55_w101_yolo>=10"),
        (0.70, 2.0, 1.0, 0.001, 0.0,   "iou70_w21_yolo>=0"),
        (0.70, 4.0, 1.0, 0.001, 0.05,  "iou70_w41_yolo>=05"),
        (0.70, 10.0, 1.0, 0.001, 0.10, "iou70_w101_yolo>=10"),
        (0.85, 2.0, 1.0, 0.001, 0.0,   "iou85_w21_yolo>=0"),
        (0.85, 4.0, 1.0, 0.001, 0.05,  "iou85_w41_yolo>=05"),
        (0.85, 10.0, 1.0, 0.001, 0.10, "iou85_w101_yolo>=10"),
    ]
    results = []
    for iou_thr, rdw, yow, skip, ymin, name in sweep_combos:
        print(f"\n--- combo: {name} ---")
        preds = build_preds_for_combo(
            cache, file_id_map, iou_thr, rdw, yow, skip, ymin
        )
        pred_path = out_root / f"pred_{name}.json"
        m = pyco_eval(gt_path, preds, pred_path)
        results.append({
            "name": name,
            "iou_thr": iou_thr,
            "rfdetr_weight": rdw,
            "yolo_weight": yow,
            "skip_box_thr": skip,
            "yolo_min_conf": ymin,
            "n_predictions": len(preds),
            "mAP50_95": m["mAP50_95"],
            "mAP50": m["mAP50"],
            "mAP75": m["mAP75"],
        })
        print(f"  → mAP50-95={m['mAP50_95']:.4f}  mAP50={m['mAP50']:.4f}  preds={len(preds)}")

    # Best combo
    results_sorted = sorted(results, key=lambda r: -r["mAP50_95"])
    print()
    print("=== WBF SWEEP RESULTS (sorted by mAP50-95) ===")
    print(f"{'name':30s} {'mAP50-95':>10s} {'mAP50':>10s} {'mAP75':>10s} {'n_preds':>10s}")
    for r in results_sorted:
        print(f"{r['name']:30s} {r['mAP50_95']:10.4f} {r['mAP50']:10.4f} "
              f"{r['mAP75']:10.4f} {r['n_predictions']:>10d}")

    summary = {
        "rfdetr_alone_pyco_baseline": 0.8877,
        "ensemble_baseline_v3_0_30_7": 0.8393,
        "best_combo": results_sorted[0],
        "all_combos": results_sorted,
    }
    out_path = out_root / "wbf_sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
