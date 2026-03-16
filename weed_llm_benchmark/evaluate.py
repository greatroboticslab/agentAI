#!/usr/bin/env python3
"""
Evaluation module for weed detection benchmark.

Computes mAP@0.5, mAP@0.5:0.95, precision, recall, F1 by comparing
LLM/YOLO predictions against ground truth annotations in YOLO format.

Usage:
    python evaluate.py --pred-dir llm_labeled/qwen25-vl-7b/detected/labels \
                       --gt-dir downloads/weed2okok/test/labels \
                       --img-dir downloads/weed2okok/test/images

    python evaluate.py --pred-json results/hf_benchmark_*.json \
                       --gt-dir downloads/weed2okok/test/labels \
                       --img-dir downloads/weed2okok/test/images
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Class Name Normalization
# ============================================================
# Maps species names from LLM outputs to canonical class IDs.
# Binary mode: everything maps to 0 (weed) or 1 (crop).

WEED_SYNONYMS = {
    "weed", "dandelion", "thistle", "clover", "crabgrass", "pigweed",
    "lambsquarters", "lamb's quarters", "ragweed", "chickweed",
    "purslane", "bindweed", "nutsedge", "foxtail", "barnyard grass",
    "goosegrass", "spurge", "henbit", "nightshade", "horseweed",
    "waterhemp", "palmer amaranth", "morningglory", "morning glory",
    "velvetleaf", "cocklebur", "johnsongrass", "johnson grass",
    "quackgrass", "wild oat", "cheat grass", "downy brome",
    "kochia", "marestail", "common ragweed", "giant ragweed",
    "grass", "broadleaf weed", "grassy weed", "unknown weed",
}

CROP_SYNONYMS = {
    "crop", "corn", "soybean", "wheat", "cotton", "rice",
    "sunflower", "canola", "barley", "sorghum", "oat",
    "sugarbeet", "sugar beet", "lettuce", "tomato",
}


def normalize_class(label, binary=True):
    """Normalize a class label to a canonical form.

    Args:
        label: Raw label string from LLM or annotation.
        binary: If True, returns 0 (weed) or 1 (crop). If False, returns
                the cleaned lowercase label string.
    Returns:
        int (binary mode) or str (multi-class mode).
    """
    if label is None:
        return 0 if binary else "unknown"
    cleaned = str(label).strip().lower()
    # Remove common prefixes
    cleaned = re.sub(r"^(class[_\s]*\d+[:\s]*)", "", cleaned)

    if binary:
        if cleaned in CROP_SYNONYMS or "crop" in cleaned:
            return 1
        return 0  # Default: weed
    return cleaned


# ============================================================
# IoU Computation
# ============================================================
def compute_iou(box1, box2):
    """Compute IoU between two boxes in [cx, cy, w, h] normalized format."""
    # Convert from center format to corner format
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # Intersection
    ix1 = max(b1_x1, b2_x1)
    iy1 = max(b1_y1, b2_y1)
    ix2 = min(b1_x2, b2_x2)
    iy2 = min(b1_y2, b2_y2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ============================================================
# Load Annotations
# ============================================================
def load_yolo_labels(label_dir, class_names=None):
    """Load YOLO-format labels from a directory.

    Each .txt file has lines: class_id cx cy w h (all normalized 0-1).

    Returns:
        dict: {image_stem: [(class_id, cx, cy, w, h), ...]}
    """
    annotations = {}
    if not os.path.isdir(label_dir):
        print(f"[!] Label directory not found: {label_dir}")
        return annotations

    for label_file in sorted(Path(label_dir).glob("*.txt")):
        stem = label_file.stem
        boxes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((class_id, cx, cy, w, h))
        annotations[stem] = boxes

    return annotations


def load_predictions_from_json(json_path, binary=True):
    """Load predictions from benchmark JSON output.

    Handles format from test_hf_models.py and roboflow_bridge.py:
    detection_results.json or hf_benchmark_*.json

    Returns:
        dict: {image_stem: [(class_id, cx, cy, w, h, confidence), ...]}
    """
    with open(json_path) as f:
        data = json.load(f)

    predictions = {}
    results_list = data if isinstance(data, list) else data.get("results", [data])

    for entry in results_list:
        img_name = entry.get("image", "")
        stem = Path(img_name).stem

        boxes = []

        # Prefer yolo_labels (already converted to normalized YOLO format by roboflow_bridge)
        yolo_labels = entry.get("yolo_labels", [])
        if yolo_labels:
            for line in yolo_labels:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cid, cx, cy, w, h, 0.5))
        else:
            # Fall back to raw detections
            detections = entry.get("detections", [])
            if isinstance(detections, str):
                continue

            for det in detections:
                label = det.get("label", "weed")
                class_id = normalize_class(label, binary=binary)
                confidence = _parse_confidence(det.get("confidence", "medium"))

                bbox = det.get("bbox") or det.get("bbox_2d")
                if not bbox or len(bbox) != 4:
                    continue

                # Convert [x1, y1, x2, y2] to YOLO [cx, cy, w, h] normalized
                x1, y1, x2, y2 = [float(v) for v in bbox]

                # Detect coordinate system:
                # - values in 0-1: already normalized
                # - values in 0-100: percentage
                # - values > 100: pixel coordinates (need image dims, skip)
                if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                    pass  # Already normalized
                elif all(0 <= v <= 100 for v in [x1, y1, x2, y2]) and any(v > 1 for v in [x1, y1, x2, y2]):
                    x1, y1, x2, y2 = x1 / 100, y1 / 100, x2 / 100, y2 / 100
                elif any(v > 100 for v in [x1, y1, x2, y2]):
                    # Pixel coordinates — skip since we don't have image dimensions
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = min(w, 1)
                h = min(h, 1)

                if w > 0 and h > 0:
                    boxes.append((class_id, cx, cy, w, h, confidence))

        predictions[stem] = boxes

    return predictions


def _parse_confidence(conf):
    """Convert confidence string/float to a numeric score."""
    if isinstance(conf, (int, float)):
        return float(conf)
    conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
    return conf_map.get(str(conf).lower(), 0.5)


# ============================================================
# Matching & Metrics
# ============================================================
def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Match predictions to ground truth using greedy IoU matching.

    Args:
        gt_boxes: [(class_id, cx, cy, w, h), ...]
        pred_boxes: [(class_id, cx, cy, w, h, confidence), ...]
        iou_threshold: Minimum IoU for a match.

    Returns:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives (unmatched ground truths)
        matches: List of (gt_idx, pred_idx, iou) tuples
    """
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []

    if not gt_boxes:
        return 0, len(pred_boxes), 0, []

    # Sort predictions by confidence (descending)
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1][-1], reverse=True)

    matched_gt = set()
    matches = []
    tp = 0
    fp = 0

    for pred_idx, pred in sorted_preds:
        pred_box = pred[1:5]  # cx, cy, w, h
        pred_class = pred[0]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_box = gt[1:5]  # cx, cy, w, h
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matches.append((best_gt_idx, pred_idx, best_iou))
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matches


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation (PASCAL VOC style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above = [p for r, p in zip(recalls, precisions) if r >= t]
        if precisions_above:
            ap += max(precisions_above) / 11
    return ap


def compute_ap_all_points(recalls, precisions):
    """Compute AP using all-point interpolation (COCO style)."""
    # Prepend sentinel values
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


# ============================================================
# Main Evaluation
# ============================================================
def _compute_ap_at_iou(gt_annotations, predictions, iou_threshold, binary=True):
    """Compute AP at a single IoU threshold using PR-curve.

    Collects all predictions across all images, sorts by confidence,
    computes cumulative precision/recall, and returns AP via all-point
    interpolation.
    """
    # Collect all predictions with image info
    all_preds = []
    for img_stem, pred_boxes in predictions.items():
        gt_boxes = gt_annotations.get(img_stem, [])
        if binary:
            gt_boxes = [(0, *b[1:]) for b in gt_boxes]
        for pred in pred_boxes:
            if binary:
                pred = (0, *pred[1:])
            all_preds.append((img_stem, pred))

    # Sort by confidence descending
    all_preds.sort(key=lambda x: x[1][-1], reverse=True)

    # Count total GT boxes
    total_gt = 0
    for boxes in gt_annotations.values():
        total_gt += len(boxes)

    if total_gt == 0 or len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0, 0, total_gt

    # Track which GT boxes have been matched per image
    matched_gt = {}

    tp_list = []

    for img_stem, pred in all_preds:
        gt_boxes = gt_annotations.get(img_stem, [])
        if binary:
            gt_boxes = [(0, *b[1:]) for b in gt_boxes]

        pred_box = pred[1:5]  # cx, cy, w, h

        best_iou = 0
        best_gt_idx = -1

        if img_stem not in matched_gt:
            matched_gt[img_stem] = set()

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt[img_stem]:
                continue
            gt_box = gt[1:5]
            iou_val = compute_iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt[img_stem].add(best_gt_idx)
            tp_list.append(1)
        else:
            tp_list.append(0)

    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - t for t in tp_list])

    # Precision and recall at each point
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt

    # AP via all-point interpolation
    ap = compute_ap_all_points(recalls, precisions)

    # Final totals
    total_tp = int(tp_cumsum[-1])
    total_fp = int(fp_cumsum[-1])
    total_fn = total_gt - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    return ap, precision, recall, total_tp, total_fp, total_fn


def evaluate_dataset(gt_annotations, predictions, iou_thresholds=None, binary=True):
    """Evaluate predictions against ground truth across a dataset.

    Args:
        gt_annotations: {image_stem: [(class_id, cx, cy, w, h), ...]}
        predictions: {image_stem: [(class_id, cx, cy, w, h, conf), ...]}
        iou_thresholds: List of IoU thresholds. Default: [0.5] for mAP@0.5.
        binary: If True, ignore class mismatch (all detections = weed).

    Returns:
        dict with mAP, precision, recall, F1, per-threshold results.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    results_per_threshold = {}
    for iou_thresh in iou_thresholds:
        ap, precision, recall, tp, fp, fn = _compute_ap_at_iou(
            gt_annotations, predictions, iou_thresh, binary=binary
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results_per_threshold[iou_thresh] = {
            "ap": round(ap, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # mAP@0.5
    map_50 = results_per_threshold.get(0.5, {}).get("ap", 0.0)

    # mAP@0.5:0.95 (COCO-style: average AP over 10 IoU thresholds)
    coco_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]
    coco_aps = {}
    for iou_thresh in coco_thresholds:
        if iou_thresh in results_per_threshold:
            coco_aps[iou_thresh] = results_per_threshold[iou_thresh]
        else:
            ap, precision, recall, tp, fp, fn = _compute_ap_at_iou(
                gt_annotations, predictions, iou_thresh, binary=binary
            )
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            coco_aps[iou_thresh] = {
                "ap": round(ap, 4),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    map_50_95 = np.mean([r["ap"] for r in coco_aps.values()])

    # mAP@0.25 (relaxed, for imprecise LLM bboxes)
    if 0.25 not in results_per_threshold:
        ap, precision, recall, tp, fp, fn = _compute_ap_at_iou(
            gt_annotations, predictions, 0.25, binary=binary
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results_per_threshold[0.25] = {
            "ap": round(ap, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    summary = {
        "num_images_gt": len(gt_annotations),
        "num_images_pred": len(predictions),
        "num_images_overlap": len(set(gt_annotations.keys()) & set(predictions.keys())),
        "total_gt_boxes": sum(len(v) for v in gt_annotations.values()),
        "total_pred_boxes": sum(len(v) for v in predictions.values()),
        "mAP@0.25": results_per_threshold[0.25]["ap"],
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": round(float(map_50_95), 4),
        "precision@0.5": results_per_threshold.get(0.5, {}).get("precision", 0.0),
        "recall@0.5": results_per_threshold.get(0.5, {}).get("recall", 0.0),
        "f1@0.5": results_per_threshold.get(0.5, {}).get("f1", 0.0),
        "per_threshold": {str(k): v for k, v in sorted(results_per_threshold.items())},
        "coco_thresholds": {str(round(k, 2)): v for k, v in sorted(coco_aps.items())},
    }

    return summary


def print_evaluation(summary, model_name=""):
    """Pretty-print evaluation results."""
    header = f"EVALUATION RESULTS" + (f" — {model_name}" if model_name else "")
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Images (GT / Pred / Overlap): {summary['num_images_gt']} / "
          f"{summary['num_images_pred']} / {summary['num_images_overlap']}")
    print(f"  Total boxes (GT / Pred):      {summary['total_gt_boxes']} / "
          f"{summary['total_pred_boxes']}")
    print()
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    print(f"  {'mAP@0.25':<20} {summary['mAP@0.25']:>10.4f}")
    print(f"  {'mAP@0.5':<20} {summary['mAP@0.5']:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {summary['mAP@0.5:0.95']:>10.4f}")
    print(f"  {'Precision@0.5':<20} {summary['precision@0.5']:>10.4f}")
    print(f"  {'Recall@0.5':<20} {summary['recall@0.5']:>10.4f}")
    print(f"  {'F1@0.5':<20} {summary['f1@0.5']:>10.4f}")
    print()

    # Per-threshold breakdown
    print(f"  IoU Threshold Breakdown:")
    print(f"  {'IoU':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*50}")
    for thresh in sorted(summary.get("per_threshold", {}).keys(), key=float):
        r = summary["per_threshold"][thresh]
        print(f"  {thresh:>6} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate weed detection predictions against ground truth")
    parser.add_argument("--pred-dir", type=str, help="Directory with predicted YOLO labels (.txt)")
    parser.add_argument("--pred-json", type=str, help="Prediction JSON file (from benchmark)")
    parser.add_argument("--gt-dir", type=str, required=True, help="Ground truth YOLO labels directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory (for metadata)")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="IoU thresholds to evaluate")
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Binary evaluation (weed vs not-weed)")
    parser.add_argument("--model-name", type=str, default="", help="Model name for display")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load ground truth
    print(f"[*] Loading ground truth from: {args.gt_dir}")
    gt = load_yolo_labels(args.gt_dir)
    print(f"    Found {len(gt)} images, {sum(len(v) for v in gt.values())} boxes")

    # Load predictions
    if args.pred_dir:
        print(f"[*] Loading predictions from labels: {args.pred_dir}")
        pred_raw = load_yolo_labels(args.pred_dir)
        # Add dummy confidence
        predictions = {}
        for stem, boxes in pred_raw.items():
            predictions[stem] = [(b[0], b[1], b[2], b[3], b[4], 0.5) for b in boxes]
    elif args.pred_json:
        print(f"[*] Loading predictions from JSON: {args.pred_json}")
        predictions = load_predictions_from_json(args.pred_json, binary=args.binary)
    else:
        parser.error("Provide either --pred-dir or --pred-json")
        return

    print(f"    Found {len(predictions)} images, {sum(len(v) for v in predictions.values())} boxes")

    # Evaluate
    summary = evaluate_dataset(gt, predictions, iou_thresholds=args.iou_thresholds, binary=args.binary)
    print_evaluation(summary, model_name=args.model_name)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[+] Results saved to {args.output}")


if __name__ == "__main__":
    main()
