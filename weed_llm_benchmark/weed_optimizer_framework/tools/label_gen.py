"""
Label Generator — Multi-VLM consensus pseudo-label generation.

Core algorithm:
1. For each holdout image, collect bboxes from multiple VLMs
2. Cluster overlapping boxes by IoU
3. Keep only clusters with min_votes from different VLMs (consensus)
4. Optionally add YOLO old-species detections for replay
"""

import os
import gc
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


def _compute_iou(box1, box2):
    """Compute IoU between two [cx, cy, w, h] boxes."""
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_h = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = inter_w * inter_h
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def _load_vlm_boxes(vlm_key, stem):
    """Load bounding boxes from a VLM's pre-generated label file."""
    label_dir = Config.get_vlm_label_dir(vlm_key)
    if not os.path.isdir(label_dir):
        return []

    label_file = os.path.join(label_dir, f"{stem}.txt")
    if not os.path.exists(label_file):
        return []

    boxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Sanity check: box dimensions should be reasonable
                if 0 < w < 1 and 0 < h < 1 and 0 < cx < 1 and 0 < cy < 1:
                    boxes.append((cx, cy, w, h, vlm_key))
    return boxes


def _cluster_boxes(all_boxes, iou_threshold):
    """Cluster overlapping boxes by IoU. Returns list of clusters."""
    used = set()
    clusters = []

    for i in range(len(all_boxes)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(all_boxes)):
            if j in used:
                continue
            if _compute_iou(all_boxes[i][:4], all_boxes[j][:4]) >= iou_threshold:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)

    return clusters


def _load_external_boxes(ext_dir, stem):
    """Load boxes from an external model's label directory."""
    boxes = []
    label_file = os.path.join(ext_dir, f"{stem}.txt")
    if not os.path.exists(label_file):
        return boxes
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if 0 < w < 1 and 0 < h < 1 and 0 < cx < 1 and 0 < cy < 1:
                    boxes.append((cx, cy, w, h, os.path.basename(ext_dir)))
    return boxes


def generate_consensus_labels(strategy, iteration):
    """Generate consensus pseudo-labels from multiple VLMs + external models.

    Now supports 3+ model consensus by also loading detections from external
    model directories (DETR, YOLOv8s, etc.) alongside VLM pre-generated labels.
    More diverse models = lower false positive rate.

    Args:
        strategy: dict with vlm_models, min_votes, consensus_iou, use_yolo_old,
                  extra_label_dirs (optional: list of paths to external model labels)
        iteration: current iteration number (for output directory naming)

    Returns:
        (label_dir, stats) where label_dir is the path to generated labels
    """
    vlm_keys = strategy.get("vlm_models", ["florence2_base", "owlv2"])
    min_votes = strategy.get("min_votes", 2)
    iou_threshold = strategy.get("consensus_iou", 0.3)
    use_yolo_old = strategy.get("use_yolo_old", True)
    extra_label_dirs = strategy.get("extra_label_dirs", [])

    # Also auto-discover external model label dirs from previous runs
    for f in os.listdir(Config.FRAMEWORK_DIR) if os.path.isdir(Config.FRAMEWORK_DIR) else []:
        ext_path = os.path.join(Config.FRAMEWORK_DIR, f)
        if f.startswith("ext_") and os.path.isdir(ext_path) and ext_path not in extra_label_dirs:
            extra_label_dirs.append(ext_path)
    if extra_label_dirs:
        logger.info(f"External model dirs for consensus: {[os.path.basename(d) for d in extra_label_dirs]}")

    # Validate VLM keys
    valid_vlms = [v for v in vlm_keys if v in Config.VLM_REGISTRY]
    if len(valid_vlms) < len(vlm_keys):
        invalid = set(vlm_keys) - set(valid_vlms)
        logger.warning(f"Invalid VLM keys ignored: {invalid}")
    if not valid_vlms and not extra_label_dirs:
        raise ValueError(f"No valid VLM keys in {vlm_keys} and no external label dirs")

    # Ensure min_votes doesn't exceed total number of sources
    total_sources = len(valid_vlms) + len(extra_label_dirs)
    min_votes = min(min_votes, max(total_sources, 1))

    # Output directory
    label_dir = os.path.join(Config.FRAMEWORK_DIR, f"labels_iter{iteration}")
    os.makedirs(label_dir, exist_ok=True)

    # Optionally load YOLO for old species detection
    yolo_model = None
    if use_yolo_old and os.path.exists(Config.YOLO_8SP_WEIGHTS):
        import torch
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo_model = YOLO(Config.YOLO_8SP_WEIGHTS)

    holdout_imgs_dir = os.path.join(Config.HOLDOUT_DIR, "train", "images")
    if not os.path.isdir(holdout_imgs_dir):
        raise FileNotFoundError(f"Holdout images not found: {holdout_imgs_dir}")

    stats = {"images": 0, "consensus_boxes": 0, "yolo_old_boxes": 0,
             "total_vlm_boxes": 0, "rejected_boxes": 0}

    for img_file in sorted(os.listdir(holdout_imgs_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = Path(img_file).stem
        stats["images"] += 1

        # --- YOLO old species detections ---
        old_lines = []
        if yolo_model is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = yolo_model.predict(
                os.path.join(holdout_imgs_dir, img_file),
                conf=Config.CONFIDENCE_THRESHOLD, device=device, verbose=False
            )
            for r in results:
                for box in r.boxes:
                    cx, cy, w, h = box.xywhn[0].tolist()
                    cls_id = int(box.cls[0])
                    old_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats["yolo_old_boxes"] += 1

        # --- VLM + external model consensus for new species ---
        all_boxes = []
        for vlm_key in valid_vlms:
            boxes = _load_vlm_boxes(vlm_key, stem)
            all_boxes.extend(boxes)
            stats["total_vlm_boxes"] += len(boxes)

        # Load external model detections (DETR, YOLOv8s, etc.)
        for ext_dir in extra_label_dirs:
            ext_boxes = _load_external_boxes(ext_dir, stem)
            all_boxes.extend(ext_boxes)
            stats["total_vlm_boxes"] += len(ext_boxes)

        new_lines = []
        if all_boxes:
            clusters = _cluster_boxes(all_boxes, iou_threshold)
            for cluster in clusters:
                # Count unique VLMs in this cluster
                vlm_sources = set(all_boxes[k][4] for k in cluster)
                if len(vlm_sources) >= min_votes:
                    # Pick the box from the highest-precision VLM
                    best_idx = max(cluster,
                                   key=lambda k: Config.get_vlm_precision(all_boxes[k][4]))
                    b = all_boxes[best_idx]
                    new_lines.append(
                        f"{Config.NOVEL_CLASS_ID} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                    )
                    stats["consensus_boxes"] += 1
                else:
                    stats["rejected_boxes"] += len(cluster)

        # Write label file
        all_lines = old_lines + new_lines
        label_path = os.path.join(label_dir, f"{stem}.txt")
        with open(label_path, "w") as f:
            if all_lines:
                f.write("\n".join(all_lines) + "\n")

    # Cleanup YOLO
    if yolo_model is not None:
        del yolo_model
        import torch
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Labels generated: {stats['images']} images, "
                f"{stats['consensus_boxes']} consensus boxes, "
                f"{stats['yolo_old_boxes']} YOLO old boxes, "
                f"{stats['rejected_boxes']} rejected")
    return label_dir, stats
