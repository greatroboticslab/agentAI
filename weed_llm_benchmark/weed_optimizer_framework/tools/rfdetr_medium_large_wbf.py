"""
v3.0.33 Path D — Medium + Large peer-strength WBF ensemble.

Medium pyco mAP50-95 = 0.8877 (cwd12 only, res 576)
Large  pyco mAP50-95 = 0.8949 (cwd12 only, res 704)
Strength delta = 0.007 (peer; vs the failed yolo+RFDETR which was 0.143).

WBF works when models are roughly comparable. Same architecture (RF-DETR)
but different scale / resolution → moderately independent failure modes.
Use WBF iou=0.85 (preserve boxes) + weights=[1, 1.5] (Large slightly heavier).

Goal: push above 0.90.
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
from PIL import Image


def _patch_supervision_for_rfdetr():
    import supervision as sv
    if getattr(sv.Detections, "_rfdetr_patched", False): return
    _o = sv.Detections.__init__
    def _new(self, *a, **k):
        _o(self, *a, **k)
        if not hasattr(self, "metadata") or self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if not hasattr(self, "data") or self.data is None:
            object.__setattr__(self, "data", {})
    sv.Detections.__init__ = _new
    sv.Detections._rfdetr_patched = True


_patch_supervision_for_rfdetr()


def predict_norm(model, img_path, w, h, threshold):
    try:
        det = model.predict(str(img_path), threshold=threshold)
    except Exception as e:
        print(f"  pred fail {img_path.name}: {e}")
        return [], [], []
    xyxy = det.xyxy if hasattr(det, "xyxy") else None
    conf = det.confidence if hasattr(det, "confidence") else None
    cls = det.class_id if hasattr(det, "class_id") else None
    if xyxy is None or len(xyxy) == 0: return [], [], []
    boxes = []
    for j in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[j]
        boxes.append([float(x1)/w, float(y1)/h, float(x2)/w, float(y2)/h])
    return boxes, conf.tolist(), cls.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium-weights", required=True)
    ap.add_argument("--large-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--wbf-iou", type=float, default=0.85)
    ap.add_argument("--wbf-skip", type=float, default=0.001)
    ap.add_argument("--medium-weight", type=float, default=1.0)
    ap.add_argument("--large-weight", type=float, default=1.5)
    args = ap.parse_args()

    out_root = Path(args.out).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    cwd12_root = Path(args.cwd12).resolve()
    dataset_dir = out_root / "dataset"

    valid_ann = dataset_dir / "valid" / "_annotations.coco.json"
    test_ann = dataset_dir / "test" / "_annotations.coco.json"
    if not (valid_ann.exists() and test_ann.exists()):
        from weed_optimizer_framework.tools.train_rfdetr import stage_dataset
        stage_dataset(dataset_dir, cwd12_root)

    val_coco = json.load(open(dataset_dir / "valid" / "_annotations.coco.json"))
    test_coco = json.load(open(dataset_dir / "test" / "_annotations.coco.json"))
    combined = {"images": list(val_coco["images"]),
                "annotations": list(val_coco["annotations"]),
                "categories": val_coco["categories"]}
    iio = max((im["id"] for im in val_coco["images"]), default=0)
    aio = max((a["id"] for a in val_coco["annotations"]), default=0)
    irmap = {}
    for im in test_coco["images"]:
        nid = im["id"] + iio; irmap[im["id"]] = nid
        nim = dict(im); nim["id"] = nid
        combined["images"].append(nim)
    for a in test_coco["annotations"]:
        na = dict(a); na["id"] = a["id"] + aio
        na["image_id"] = irmap[a["image_id"]]
        combined["annotations"].append(na)

    combined_dir = out_root / "eval_combined"
    if not combined_dir.exists():
        combined_dir.mkdir(parents=True)
        for split in ("valid", "test"):
            for f in (dataset_dir / split).glob("*.jpg"):
                link = combined_dir / f.name
                if link.exists() or link.is_symlink(): link.unlink()
                os.symlink(f.resolve(), link)

    file_id_map = {im["file_name"]: im["id"] for im in combined["images"]}
    gt_path = out_root / "ml_wbf_gt.json"
    with open(gt_path, "w") as f: json.dump(combined, f)

    print(f"[ml-wbf] loading Medium from {args.medium_weights}")
    from rfdetr import RFDETRMedium, RFDETRLarge
    m_model = RFDETRMedium(pretrain_weights=str(args.medium_weights))
    print(f"[ml-wbf] loading Large from {args.large_weights}")
    l_model = RFDETRLarge(pretrain_weights=str(args.large_weights))

    from ensemble_boxes import weighted_boxes_fusion
    img_files = sorted(combined_dir.glob("*.jpg"))
    print(f"[ml-wbf] {len(img_files)} images, WBF iou={args.wbf_iou} weights=[M={args.medium_weight},L={args.large_weight}]")

    preds = []
    for i, img_path in enumerate(img_files):
        with Image.open(img_path) as im: w, h = im.size
        m_box, m_conf, m_cls = predict_norm(m_model, img_path, w, h, args.threshold)
        l_box, l_conf, l_cls = predict_norm(l_model, img_path, w, h, args.threshold)
        if not m_box and not l_box: continue
        bl, sl, ll, ws = [], [], [], []
        if m_box:
            bl.append(m_box); sl.append(m_conf); ll.append(m_cls); ws.append(args.medium_weight)
        if l_box:
            bl.append(l_box); sl.append(l_conf); ll.append(l_cls); ws.append(args.large_weight)
        try:
            fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, weights=ws,
                                                iou_thr=args.wbf_iou,
                                                skip_box_thr=args.wbf_skip)
        except Exception as e:
            print(f"  wbf fail {img_path.name}: {e}"); continue
        iid = file_id_map[img_path.name]
        for j in range(len(fb)):
            x1n, y1n, x2n, y2n = fb[j]
            x1, y1, x2, y2 = x1n*w, y1n*h, x2n*w, y2n*h
            preds.append({"image_id": int(iid), "category_id": int(fl[j]),
                          "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                          "score": float(fs[j])})
        if (i+1) % 200 == 0:
            print(f"[ml-wbf] {i+1}/{len(img_files)}  preds={len(preds)}")

    print(f"[ml-wbf] DONE: total preds={len(preds)}")
    pred_path = out_root / "ml_wbf_pred.json"
    with open(pred_path, "w") as f: json.dump(preds, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    s = coco_eval.stats.tolist()
    summary = {
        "medium_weights": str(args.medium_weights),
        "large_weights": str(args.large_weights),
        "wbf_iou": args.wbf_iou, "wbf_skip": args.wbf_skip,
        "weights": [args.medium_weight, args.large_weight],
        "n_images": len(combined["images"]),
        "n_annotations": len(combined["annotations"]),
        "n_predictions": len(preds),
        "mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
    }
    print()
    print("=== Medium + Large WBF ENSEMBLE pyco ===")
    print(f"  mAP50-95: {s[0]:.4f}")
    print(f"  mAP50:    {s[1]:.4f}")
    print(f"  mAP75:    {s[2]:.4f}")
    out_path = out_root / "ml_wbf_pyco_summary.json"
    with open(out_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"[ml-wbf] wrote {out_path}")


if __name__ == "__main__":
    main()
