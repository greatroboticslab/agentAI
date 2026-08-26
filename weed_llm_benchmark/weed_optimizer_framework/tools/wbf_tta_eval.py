"""
v3.0.29 — Weighted Boxes Fusion + multi-scale TTA evaluation on cwd12 holdout.

Per arXiv 2603.00160 (DINOv3+YOLO26, 2026) + the 2026 detection-competition
meta: run inference at multiple input resolutions, optionally with hflip,
then fuse per-image detections via WBF (instead of NMS) using each
detection's confidence as the fusion weight. WBF preserves more useful
overlapping boxes than NMS and consistently delivers +0.02-0.05 mAP50-95
on standard benchmarks.

We then compute mAP50, mAP50-95, P, R against the cwd12 test+valid holdout
labels using a deterministic IoU-based matcher (no leak path: val staging
mirrors what the YOLO trainer used during training).

Output: a JSON summary plus per-class AP for the 12 weed species.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from ultralytics import YOLO

# v3.24.4: this list is NOT the sealed protocol's class order. cwd12_sealed.yaml
# orders the 12 species alphabetically; this one does not, so using it to name
# per-class APs silently attributes every species' score to a different weed
# (id 2 is Eclipta in the data, PalmerAmaranth here). Overall mAP is unaffected —
# it averages over ids — which is exactly why the error is easy to miss. Class
# names now come from the dataset yaml via --data-yaml; this remains only so old
# invocations keep working, and it is never the default when a yaml is given.
_LEGACY_CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                        "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                        "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]


def yolo_to_xyxy_norm(box, w, h):
    """YOLO format (cx, cy, bw, bh) normalized → (x1, y1, x2, y2) normalized."""
    cx, cy, bw, bh = box
    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0, cx + bw / 2)
    y2 = min(1.0, cy + bh / 2)
    return [x1, y1, x2, y2]


def read_yolo_labels(lbl_path: Path):
    """Return list of (class_id, [x1,y1,x2,y2] normalized)."""
    if not lbl_path.exists():
        return []
    out = []
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        out.append((cid, yolo_to_xyxy_norm((cx, cy, bw, bh), 1, 1)))
    return out


def predict_one_scale(model, img_path, imgsz, conf, iou_nms, hflip=False):
    """Run YOLO inference at one scale. Returns (xyxy_norm, scores, classes).

    hflip must stay False here: this function would mirror the *coordinates* of
    predictions made on an UNflipped image, which is not test-time augmentation,
    it is a wrong answer. Use predict_one_scale_hflip, which flips the image.
    """
    if hflip:
        raise ValueError("predict_one_scale(hflip=True) is not TTA — "
                         "use predict_one_scale_hflip()")
    res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf,
                        iou=iou_nms, augment=False, verbose=False, device=0)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], []
    # ultralytics returns xyxy in pixel space; we normalize by image size
    h, w = res.orig_shape
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    xyxy[:, [0, 2]] /= w
    xyxy[:, [1, 3]] /= h
    xyxy = np.clip(xyxy, 0.0, 1.0)
    scores = boxes.conf.cpu().numpy().astype(np.float32)
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    return xyxy.tolist(), scores.tolist(), cls.tolist()


def predict_one_scale_hflip(model, img_path, imgsz, conf, iou_nms):
    """Run YOLO with horizontal flip — flip image first, predict, then flip
    box coords back. ultralytics doesn't expose hflip toggle via predict() so
    we route through PIL."""
    pil = Image.open(img_path).convert("RGB")
    flipped = pil.transpose(Image.FLIP_LEFT_RIGHT)
    res = model.predict(source=flipped, imgsz=imgsz, conf=conf,
                        iou=iou_nms, augment=False, verbose=False, device=0)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], []
    h, w = res.orig_shape
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    xyxy[:, [0, 2]] /= w
    xyxy[:, [1, 3]] /= h
    # un-flip x coordinates: x' = 1 - x
    new = xyxy.copy()
    new[:, 0] = 1.0 - xyxy[:, 2]
    new[:, 2] = 1.0 - xyxy[:, 0]
    new = np.clip(new, 0.0, 1.0)
    return new.tolist(), boxes.conf.cpu().numpy().astype(np.float32).tolist(), \
           boxes.cls.cpu().numpy().astype(np.int32).tolist()


def iou_xyxy(a, b):
    """IoU between two arrays of [x1,y1,x2,y2]. a=(N,4) b=(M,4) → (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def compute_map(per_image_preds, per_image_gts, n_classes, names=None):
    """COCO-style mAP at IoU thresholds 0.5 and 0.5:0.95.

    per_image_preds[i] = (xyxy, scores, classes) lists for image i.
    per_image_gts[i]   = list of (class, [xyxy]) for image i.
    names              = class-id -> species name, from the dataset yaml.

    Returns: dict with mAP50, mAP50-95, per-class AP for each threshold.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 10 thresholds
    # Per-class accumulators: for each class, gather per-detection (score, tp_at_each_iou)
    cls_records = {c: {"tps": [], "scores": [], "n_gt": 0} for c in range(n_classes)}
    for c in range(n_classes):
        # Total GT count for this class across all images
        for gts in per_image_gts:
            cls_records[c]["n_gt"] += sum(1 for cid, _ in gts if cid == c)

    for img_idx, ((p_xyxy, p_scores, p_cls), gts) in enumerate(
            zip(per_image_preds, per_image_gts)):
        # Per-class processing
        for c in range(n_classes):
            # Gather class-c predictions
            mask_c = [i for i, cc in enumerate(p_cls) if cc == c]
            if not mask_c:
                continue
            preds_c_xyxy = np.asarray([p_xyxy[i] for i in mask_c], dtype=np.float32)
            preds_c_scores = np.asarray([p_scores[i] for i in mask_c], dtype=np.float32)
            # Gather class-c GTs
            gt_c = np.asarray([b for cid, b in gts if cid == c], dtype=np.float32) \
                   .reshape(-1, 4)
            # Sort preds by descending score
            order = np.argsort(-preds_c_scores)
            preds_c_xyxy = preds_c_xyxy[order]
            preds_c_scores = preds_c_scores[order]
            tp_at_iou = np.zeros((len(preds_c_xyxy), len(iou_thresholds)),
                                 dtype=bool)
            if len(gt_c) == 0:
                # All preds are FP at every IoU
                pass
            else:
                ious = iou_xyxy(preds_c_xyxy, gt_c)  # (P, G)
                # Greedy match per IoU threshold (each GT can be matched once)
                for t_idx, t in enumerate(iou_thresholds):
                    matched = np.zeros(len(gt_c), dtype=bool)
                    for p_idx in range(len(preds_c_xyxy)):
                        # best unmatched GT above threshold
                        best_g = -1; best_iou = t
                        for g_idx in range(len(gt_c)):
                            if matched[g_idx]:
                                continue
                            if ious[p_idx, g_idx] >= best_iou:
                                best_iou = ious[p_idx, g_idx]
                                best_g = g_idx
                        if best_g >= 0:
                            matched[best_g] = True
                            tp_at_iou[p_idx, t_idx] = True
            cls_records[c]["tps"].append(tp_at_iou)
            cls_records[c]["scores"].append(preds_c_scores)

    # Compute AP per class per IoU
    per_class_ap = {c: np.zeros(len(iou_thresholds), dtype=np.float32)
                    for c in range(n_classes)}
    for c, rec in cls_records.items():
        if rec["n_gt"] == 0 or not rec["tps"]:
            continue
        all_tp = np.concatenate(rec["tps"], axis=0)         # (N, T)
        all_scores = np.concatenate(rec["scores"], axis=0)  # (N,)
        order = np.argsort(-all_scores)
        all_tp = all_tp[order]
        n_gt = rec["n_gt"]
        for t_idx in range(len(iou_thresholds)):
            tp_cum = np.cumsum(all_tp[:, t_idx].astype(np.float32))
            fp_cum = np.cumsum((~all_tp[:, t_idx]).astype(np.float32))
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            # 101-pt VOC2012 / COCO-style
            ap = 0.0
            for r in np.linspace(0, 1, 101):
                p_at_r = precision[recall >= r].max() if (recall >= r).any() else 0.0
                ap += p_at_r
            ap /= 101
            per_class_ap[c][t_idx] = ap

    # Aggregate
    ap_at_50 = np.array([per_class_ap[c][0] for c in range(n_classes)
                         if cls_records[c]["n_gt"] > 0])
    ap_at_5095 = np.array([per_class_ap[c].mean() for c in range(n_classes)
                           if cls_records[c]["n_gt"] > 0])
    return {
        "mAP50": float(ap_at_50.mean()) if len(ap_at_50) else 0.0,
        "mAP50_95": float(ap_at_5095.mean()) if len(ap_at_5095) else 0.0,
        "per_class": {
            (names[c] if names and c < len(names) else f"class_{c}"): {
                "AP50": float(per_class_ap[c][0]),
                "AP50_95": float(per_class_ap[c].mean()),
                "n_gt": cls_records[c]["n_gt"],
            }
            for c in range(n_classes) if cls_records[c]["n_gt"] > 0
        },
    }


def _resolve_holdout(data_yaml):
    """Val image dirs + class names, straight from the dataset yaml.

    The sealed protocol's `val` is a LIST (test/images + valid/images = the 1,977
    holdout), and its class order is alphabetical. Reading both from the yaml the
    training run actually used is the only way this evaluation and that run agree
    on what class 2 means.
    """
    import yaml
    d = yaml.safe_load(open(data_yaml))
    root = Path(d.get("path", "")).expanduser()
    val = d["val"]
    if isinstance(val, str):
        val = [val]
    dirs = [Path(v) if Path(v).is_absolute() else root / v for v in val]
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return dirs, list(names)


def _collect(dirs):
    """(image, label) pairs. Labels mirror images: .../images/x.jpg -> .../labels/x.txt"""
    pairs = []
    for d in dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for img in sorted(d.glob(ext)):
                lbl = Path(str(img.parent).replace("/images", "/labels")) / (img.stem + ".txt")
                pairs.append((img, lbl))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, nargs="+",
                    help="one checkpoint, or several to ensemble via WBF")
    ap.add_argument("--data-yaml", help="dataset yaml (preferred: gives val dirs AND "
                                        "the class order the model was trained with)")
    ap.add_argument("--val-imgs", help="legacy: single image dir")
    ap.add_argument("--val-lbls", help="legacy: single label dir")
    ap.add_argument("--imgszs", type=int, nargs="+", default=[640])
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--no-wbf", action="store_true",
                    help="skip fusion entirely and report the raw NMS output. This is "
                         "the honest baseline: it shares this script's matcher with "
                         "every TTA arm, so their differences are the augmentation "
                         "and not the metric implementation.")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="per-scale NMS IoU")
    ap.add_argument("--wbf-iou", type=float, default=0.55, help="WBF merge IoU")
    ap.add_argument("--wbf-skip", type=float, default=0.001,
                    help="drop boxes below this score before WBF")
    ap.add_argument("--n-classes", type=int, default=None)
    ap.add_argument("--label", default="", help="arm name, recorded in the output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.no_wbf and (len(args.weights) > 1 or len(args.imgszs) > 1 or args.hflip):
        raise SystemExit("--no-wbf is a single-model single-scale no-flip baseline; "
                         "with more views there is nothing to combine them but fusion.")

    if args.data_yaml:
        val_dirs, names = _resolve_holdout(args.data_yaml)
    elif args.val_imgs:
        val_dirs, names = [Path(args.val_imgs)], _LEGACY_CANONICAL_12
        print("[wbf-tta] WARNING: no --data-yaml; per-class names fall back to the "
              "legacy order, which does NOT match cwd12_sealed.yaml")
    else:
        raise SystemExit("need --data-yaml (preferred) or --val-imgs")
    n_classes = args.n_classes or len(names)

    print(f"[wbf-tta] arm={args.label or '(unnamed)'}")
    print(f"[wbf-tta] models={len(args.weights)}: {[Path(w).parent.parent.name for w in args.weights]}")
    print(f"[wbf-tta] imgszs={args.imgszs}  hflip={args.hflip}  wbf={'off' if args.no_wbf else 'on'}")
    print(f"[wbf-tta] classes={n_classes} {names}")

    models = [YOLO(w) for w in args.weights]
    pairs = _collect(val_dirs)
    if args.val_lbls:      # legacy override
        pairs = [(i, Path(args.val_lbls) / (i.stem + ".txt")) for i, _ in pairs]
    n_lbl = sum(1 for _, l in pairs if l.exists())
    print(f"[wbf-tta] holdout: {len(pairs)} images from {[str(d) for d in val_dirs]}"
          f"  ({n_lbl} with labels)")
    if not pairs:
        raise SystemExit("no images found — check the yaml's path/val entries")

    per_img_preds, per_img_gts = [], []
    t0 = time.time()
    for i, (img_path, lbl) in enumerate(pairs):
        per_img_gts.append(read_yolo_labels(lbl))
        boxes_lists, scores_lists, labels_lists, wts = [], [], [], []
        for model in models:
            for sz in args.imgszs:
                xyxy, scs, cls = predict_one_scale(model, img_path, sz,
                                                   args.conf, args.iou)
                if xyxy:
                    boxes_lists.append(xyxy); scores_lists.append(scs)
                    labels_lists.append(cls); wts.append(1.0)
                if args.hflip:
                    fx, fs, fc = predict_one_scale_hflip(model, img_path, sz,
                                                         args.conf, args.iou)
                    if fx:
                        boxes_lists.append(fx); scores_lists.append(fs)
                        labels_lists.append(fc); wts.append(1.0)

        if not boxes_lists:
            per_img_preds.append(([], [], []))
        elif args.no_wbf:
            per_img_preds.append((boxes_lists[0], scores_lists[0], labels_lists[0]))
        else:
            fb, fs, fl = weighted_boxes_fusion(
                boxes_lists, scores_lists, labels_lists, weights=wts,
                iou_thr=args.wbf_iou, skip_box_thr=args.wbf_skip)
            per_img_preds.append((fb.tolist(), fs.tolist(), [int(c) for c in fl]))

        if (i + 1) % 250 == 0:
            print(f"[wbf-tta] {i+1}/{len(pairs)}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[wbf-tta] computing mAP...")
    res = compute_map(per_img_preds, per_img_gts, n_classes=n_classes, names=names)
    print(f"\n=== {args.label or 'RESULT'} ===")
    print(f"mAP50:    {res['mAP50']:.4f}")
    print(f"mAP50-95: {res['mAP50_95']:.4f}")
    for cname, st in sorted(res["per_class"].items(),
                            key=lambda kv: -kv[1]["AP50_95"]):
        print(f"  {cname:18s} AP50={st['AP50']:.3f}  AP50-95={st['AP50_95']:.3f}  "
              f"n_gt={st['n_gt']}")

    out = {"label": args.label, "weights": args.weights, "n_models": len(args.weights),
           "imgszs": args.imgszs, "hflip": args.hflip, "wbf": not args.no_wbf,
           "wbf_iou": args.wbf_iou, "data_yaml": args.data_yaml,
           "n_images": len(pairs), "names": names,
           "seconds": round(time.time() - t0, 1), **res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[wbf-tta] wrote {args.out}")


if __name__ == "__main__":
    main()
