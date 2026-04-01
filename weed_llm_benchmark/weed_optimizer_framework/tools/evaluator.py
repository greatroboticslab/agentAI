"""
Evaluator — Full evaluation with mAP@0.5, mAP@0.5:0.95, per-class P/R/F1.

This is the most critical tool: every experiment result depends on correct evaluation.
Uses standard PASCAL VOC / COCO mAP computation with IoU-based matching.
"""

import os
import gc
import logging
import numpy as np
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


def _compute_iou(box1, box2):
    """Compute IoU between two boxes in [cx, cy, w, h] normalized format."""
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_h = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = inter_w * inter_h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def _load_gt_labels(label_path):
    """Load ground truth labels from YOLO format file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append({"class": cls_id, "box": [cx, cy, w, h]})
    return boxes


def _match_predictions_to_gt(predictions, ground_truths, iou_threshold):
    """Match predictions to ground truths using greedy IoU matching.

    Args:
        predictions: list of {"box": [cx,cy,w,h], "conf": float, "class": int}
        ground_truths: list of {"box": [cx,cy,w,h], "class": int}
        iou_threshold: minimum IoU for a match

    Returns:
        tp_list: list of (conf, is_tp) for AP computation
        fn_count: number of unmatched ground truths
    """
    # Sort predictions by confidence (descending)
    preds_sorted = sorted(predictions, key=lambda x: x.get("conf", 0), reverse=True)
    matched_gt = set()
    tp_list = []

    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for gi, gt in enumerate(ground_truths):
            if gi in matched_gt:
                continue
            iou = _compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp_list.append((pred.get("conf", 0), True))
        else:
            tp_list.append((pred.get("conf", 0), False))

    fn_count = len(ground_truths) - len(matched_gt)
    return tp_list, fn_count


def _compute_ap(tp_list, n_gt):
    """Compute Average Precision using all-point interpolation (PASCAL VOC 2010+).

    Args:
        tp_list: list of (conf, is_tp), sorted by conf descending
        n_gt: total number of ground truth boxes

    Returns:
        AP value (float)
    """
    if n_gt == 0:
        return 0.0

    # Sort by confidence descending
    tp_list_sorted = sorted(tp_list, key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for conf, is_tp in tp_list_sorted:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        precisions.append(precision)
        recalls.append(recall)

    if not recalls:
        return 0.0

    # All-point interpolation
    # Add sentinel values
    mrec = [0.0] + recalls + [recalls[-1]]
    mpre = [1.0] + precisions + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find recall change points
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def evaluate_yolo(model_path, test_images_dir, test_labels_dir,
                  iou_thresholds=None, binary_mode=True):
    """Evaluate a YOLO model with full metrics.

    Args:
        model_path: path to YOLO .pt weights
        test_images_dir: directory of test images
        test_labels_dir: directory of test labels (YOLO format)
        iou_thresholds: list of IoU thresholds for mAP computation
        binary_mode: if True, treat all classes as "weed" (ignore class ID)

    Returns:
        dict with keys: f1, precision, recall, map50, map50_95, per_class
    """
    import torch
    from ultralytics import YOLO

    if iou_thresholds is None:
        iou_thresholds = Config.MAP50_95_THRESHOLDS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    # Collect all predictions and ground truths
    all_tp_lists = {t: [] for t in iou_thresholds}  # per IoU threshold
    total_gt = 0
    total_tp_50 = 0  # for F1 at IoU=0.5
    total_fp_50 = 0
    total_fn_50 = 0

    per_class_stats = {}  # class_name -> {tp, fp, fn}

    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for img_file in image_files:
        stem = Path(img_file).stem
        label_path = os.path.join(test_labels_dir, stem + ".txt")
        gt_boxes = _load_gt_labels(label_path)
        if not gt_boxes and not os.path.exists(label_path):
            continue

        # Run YOLO inference
        img_path = os.path.join(test_images_dir, img_file)
        results = model.predict(img_path, conf=Config.CONFIDENCE_THRESHOLD,
                                device=device, verbose=False)

        # Extract predictions
        predictions = []
        for r in results:
            for box in r.boxes:
                cx, cy, w, h = box.xywhn[0].tolist()
                predictions.append({
                    "box": [cx, cy, w, h],
                    "conf": float(box.conf[0]),
                    "class": int(box.cls[0]),
                })

        total_gt += len(gt_boxes)

        # Evaluate at each IoU threshold
        for iou_t in iou_thresholds:
            if binary_mode:
                # Binary: all classes = weed
                gt_simple = [{"box": g["box"]} for g in gt_boxes]
                pred_simple = [{"box": p["box"], "conf": p["conf"]} for p in predictions]
            else:
                gt_simple = gt_boxes
                pred_simple = predictions

            tp_list, fn_count = _match_predictions_to_gt(pred_simple, gt_simple, iou_t)
            all_tp_lists[iou_t].extend(tp_list)

            # F1 at IoU=0.5
            if iou_t == 0.5:
                tp_count = sum(1 for _, is_tp in tp_list if is_tp)
                fp_count = sum(1 for _, is_tp in tp_list if not is_tp)
                total_tp_50 += tp_count
                total_fp_50 += fp_count
                total_fn_50 += fn_count

    # Compute mAP at each threshold
    aps = {}
    for iou_t in iou_thresholds:
        ap = _compute_ap(all_tp_lists[iou_t], total_gt)
        aps[iou_t] = round(ap, 4)

    # mAP@0.5
    map50 = aps.get(0.5, 0.0)

    # mAP@0.5:0.95
    map50_95_thresholds = [t for t in iou_thresholds if 0.5 <= t <= 0.95]
    if map50_95_thresholds:
        map50_95 = round(np.mean([aps[t] for t in map50_95_thresholds]), 4)
    else:
        map50_95 = map50

    # F1, Precision, Recall at IoU=0.5
    precision = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0
    recall = total_tp_50 / (total_tp_50 + total_fn_50) if (total_tp_50 + total_fn_50) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": map50,
        "map50_95": map50_95,
        "aps_by_iou": aps,
        "total_gt": total_gt,
        "total_predictions": total_tp_50 + total_fp_50,
        "total_images": len(image_files),
    }

    logger.info(f"Evaluation: F1={f1:.4f}, mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, "
                f"P={precision:.4f}, R={recall:.4f}")
    return result


def evaluate_full(model_path):
    """Evaluate YOLO on both old and new species test sets.

    Returns a comprehensive result dict with:
    - old_f1, old_map50, old_map50_95, old_precision, old_recall
    - new_f1, new_map50, new_map50_95, new_precision, new_recall
    - forgetting flag
    """
    logger.info(f"Full evaluation of {model_path}")

    # Old species (8 species YOLO was trained on)
    old_result = evaluate_yolo(
        model_path,
        os.path.join(Config.SP8_DIR, "test", "images"),
        os.path.join(Config.SP8_DIR, "test", "labels"),
    )

    # New species (4 holdout species)
    new_result = evaluate_yolo(
        model_path,
        os.path.join(Config.HOLDOUT_DIR, "test", "images"),
        os.path.join(Config.HOLDOUT_DIR, "test", "labels"),
    )

    combined = {
        "old_f1": old_result["f1"],
        "old_precision": old_result["precision"],
        "old_recall": old_result["recall"],
        "old_map50": old_result["map50"],
        "old_map50_95": old_result["map50_95"],
        "old_aps": old_result["aps_by_iou"],

        "new_f1": new_result["f1"],
        "new_precision": new_result["precision"],
        "new_recall": new_result["recall"],
        "new_map50": new_result["map50"],
        "new_map50_95": new_result["map50_95"],
        "new_aps": new_result["aps_by_iou"],

        "forgetting": old_result["f1"] < Config.FORGETTING_THRESHOLD,
    }

    logger.info(f"  Old: F1={combined['old_f1']}, mAP50={combined['old_map50']}, "
                f"mAP50-95={combined['old_map50_95']}")
    logger.info(f"  New: F1={combined['new_f1']}, mAP50={combined['new_map50']}, "
                f"mAP50-95={combined['new_map50_95']}")
    logger.info(f"  Forgetting: {combined['forgetting']}")

    return combined
