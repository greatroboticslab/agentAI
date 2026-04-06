"""
Label Filter — Use YOLO's own predictions to filter noisy pseudo-labels.

This is the key tool for attacking the 27% false positive rate problem.

Algorithm (self-training filter):
1. Take consensus pseudo-labels (noisy, ~27% FP)
2. Run YOLO (trained or baseline) on the same images at HIGH confidence (>0.7)
3. For each pseudo-label box, check if YOLO also detected something nearby
4. Keep only boxes that YOLO confirms → removes most false positives
5. Result: cleaner labels for retraining

Why this works: YOLO's confidence IS calibrated (unlike Florence-2).
A YOLO prediction at conf=0.8 is genuinely 80% likely to be correct.
So filtering pseudo-labels against high-conf YOLO predictions removes
boxes that even YOLO doesn't think are real.
"""

import os
import gc
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


def _compute_iou(box1, box2):
    """IoU between two [cx, cy, w, h] normalized boxes."""
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_h = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = inter_w * inter_h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def filter_labels_with_yolo(model_path, label_dir, image_dir,
                             conf_threshold=0.7, iou_match=0.3, iteration=0):
    """Filter pseudo-labels using YOLO's high-confidence predictions.

    For each image:
    1. Read pseudo-labels from label_dir
    2. Run YOLO at high confidence (conf_threshold)
    3. Keep pseudo-label boxes that overlap with a YOLO prediction (IoU > iou_match)
    4. Discard pseudo-label boxes that YOLO doesn't confirm

    Args:
        model_path: YOLO .pt weights (baseline or first-round trained)
        label_dir: directory with noisy pseudo-labels
        image_dir: directory with corresponding images
        conf_threshold: YOLO confidence threshold for filtering (higher = stricter)
        iou_match: IoU threshold for matching pseudo-label to YOLO prediction
        iteration: for output dir naming

    Returns:
        (filtered_dir, stats)
    """
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    filtered_dir = os.path.join(Config.FRAMEWORK_DIR, f"filtered_labels_iter{iteration}")
    os.makedirs(filtered_dir, exist_ok=True)

    stats = {"original": 0, "kept": 0, "removed": 0, "images": 0, "empty_after_filter": 0}

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for lbl_file in sorted(label_files):
        stem = lbl_file.replace(".txt", "")
        lbl_path = os.path.join(label_dir, lbl_file)
        stats["images"] += 1

        # Read pseudo-labels
        pseudo_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    pseudo_boxes.append((cls_id, box, line.strip()))
                    stats["original"] += 1

        if not pseudo_boxes:
            # Copy empty files as-is
            with open(os.path.join(filtered_dir, lbl_file), "w") as f:
                f.write("")
            continue

        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            # No image found, keep all labels (can't filter)
            with open(os.path.join(filtered_dir, lbl_file), "w") as f:
                f.write("\n".join(line for _, _, line in pseudo_boxes) + "\n")
            stats["kept"] += len(pseudo_boxes)
            continue

        # Run YOLO at high confidence
        results = model.predict(img_path, conf=conf_threshold, device=device, verbose=False)
        yolo_boxes = []
        for r in results:
            for box in r.boxes:
                yolo_boxes.append(box.xywhn[0].tolist())

        # Filter: keep pseudo-labels that match a YOLO prediction
        kept_lines = []
        for cls_id, pbox, line in pseudo_boxes:
            # Check if any YOLO box overlaps with this pseudo-label
            matched = False
            for ybox in yolo_boxes:
                if _compute_iou(pbox, ybox) >= iou_match:
                    matched = True
                    break

            # For OLD species (cls_id in train set), always keep
            if cls_id in Config.TRAIN_SPECIES_IDS:
                kept_lines.append(line)
                stats["kept"] += 1
            elif matched:
                # NEW species: only keep if YOLO confirms
                kept_lines.append(line)
                stats["kept"] += 1
            else:
                stats["removed"] += 1

        # Write filtered labels
        with open(os.path.join(filtered_dir, lbl_file), "w") as f:
            f.write("\n".join(kept_lines) + "\n" if kept_lines else "")

        if not kept_lines:
            stats["empty_after_filter"] += 1

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    total = stats["original"]
    stats["removal_rate"] = stats["removed"] / total if total > 0 else 0

    logger.info(f"[Filter] {stats['original']} boxes → {stats['kept']} kept, "
                f"{stats['removed']} removed ({stats['removal_rate']:.1%})")
    return filtered_dir, stats
