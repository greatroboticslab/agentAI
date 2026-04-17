"""
Mega Trainer — v3.0 direction: train largest YOLO on merged real-labeled datasets.

Key differences from yolo_trainer.py (v2.x):
- Uses Config.DETECTION_MODEL (yolo11x/yolo26x etc.) not Config.YOLO_8SP_WEIGHTS
- Merges ALL downloaded datasets (cumulative, not pseudo-labels)
- No replay buffer needed — real labels already present
- Target: theoretical accuracy limit via massive data + large model
"""

import os
import gc
import shutil
import logging
from pathlib import Path
from ..config import Config
from .dataset_discovery import DatasetDiscovery

logger = logging.getLogger(__name__)


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp')


def _resolve_best_pt(model, project_dir):
    """Resolve actual best.pt after ultralytics training (handles train, train2, ... dirs)."""
    try:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            cand = Path(save_dir) / "weights" / "best.pt"
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    try:
        train_dirs = sorted(
            (p for p in Path(project_dir).glob("train*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in train_dirs:
            cand = d / "weights" / "best.pt"
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    return None


def _find_images(root):
    """Recursively find image files under root."""
    return [p for p in Path(root).rglob("*") if p.suffix in IMG_EXTS]


def _find_label_for_image(img_path, dataset_root):
    """Find YOLO label (.txt) corresponding to img_path. Supports:
    - Sibling labels dir: images/x.jpg -> labels/x.txt
    - Same dir: x.jpg -> x.txt
    """
    stem = img_path.stem
    # Try images/labels sibling pattern
    if "images" in img_path.parts:
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if label_path.exists():
            return label_path
    # Try same directory
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same
    # Try walking up to find a labels/ dir
    for parent in img_path.parents:
        cand = parent / "labels" / (stem + ".txt")
        if cand.exists():
            return cand
        if parent == Path(dataset_root):
            break
    return None


def _merge_datasets(out_dir, val_fraction=0.1):
    """Merge all downloaded datasets (with labels) into one YOLO-format dataset.

    Returns: (merged_dir, data_yaml, stats, merged_names_list)
    """
    disc = DatasetDiscovery()
    registry = disc.registry["datasets"]

    merged = {"train/images": [], "train/labels": [], "valid/images": [], "valid/labels": []}
    for sub in merged:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Collect all unique class names across datasets (order = merged class IDs)
    class_name_to_id = {}
    used_datasets = []

    stats = {"datasets": 0, "images": 0, "labels": 0, "skipped_no_label": 0}

    for ds_name, info in registry.items():
        local_path = info.get("local_path")
        if not local_path or not os.path.isdir(local_path):
            continue
        if info.get("annotation") not in ("bbox", "bbox+segmentation", "yolo"):
            # Skip classification-only datasets (no bboxes)
            continue

        imgs = _find_images(local_path)
        if not imgs:
            continue

        ds_class_map = {}  # per-dataset source_id -> merged_id
        # Try to read the dataset's own class names from metadata if available
        src_names = info.get("class_names") or []
        for i, name in enumerate(src_names):
            if name not in class_name_to_id:
                class_name_to_id[name] = len(class_name_to_id)
            ds_class_map[i] = class_name_to_id[name]

        ds_img_count = 0
        for img in imgs:
            lbl = _find_label_for_image(img, local_path)
            if lbl is None:
                stats["skipped_no_label"] += 1
                continue

            # Optionally remap class IDs using ds_class_map
            with open(lbl) as f:
                label_lines = f.read().strip().splitlines()
            if ds_class_map:
                remapped = []
                for ln in label_lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    src_cls = int(parts[0])
                    new_cls = ds_class_map.get(src_cls, src_cls)
                    remapped.append(" ".join([str(new_cls)] + parts[1:]))
                new_label_text = "\n".join(remapped)
            else:
                new_label_text = "\n".join(label_lines)

            # Split: 1/10 to val
            bucket = "valid" if (ds_img_count % int(1 / val_fraction)) == 0 else "train"
            dst_stem = f"{ds_name}_{img.stem}"
            shutil.copy2(img, os.path.join(out_dir, bucket, "images", dst_stem + img.suffix))
            with open(os.path.join(out_dir, bucket, "labels", dst_stem + ".txt"), "w") as f:
                f.write(new_label_text)
            ds_img_count += 1
            stats["images"] += 1
            stats["labels"] += 1

        if ds_img_count > 0:
            used_datasets.append(ds_name)
            stats["datasets"] += 1
            logger.info(f"[Merge] {ds_name}: {ds_img_count} images")

    # Build names list sorted by assigned id
    names_list = sorted(class_name_to_id.keys(), key=lambda n: class_name_to_id[n])
    if not names_list:
        names_list = ["weed"]  # single-class fallback

    # Write data.yaml
    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: {os.path.join(out_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(out_dir, 'valid', 'images')}\n")
        f.write(f"nc: {len(names_list)}\n")
        f.write(f"names: {names_list}\n")

    logger.info(f"[Merge] Total: {stats['images']} images from {stats['datasets']} datasets; {len(names_list)} classes")
    return out_dir, data_yaml, stats, used_datasets, names_list


def train_yolo_mega(strategy, iteration):
    """Train the largest YOLO on merged real-labeled datasets (v3.0 approach).

    Strategy keys:
      base_model: override, defaults to Config.DETECTION_MODEL
      epochs, batch_size, lr, patience, workers, imgsz

    Returns: (best_pt_path, result_summary)
    """
    import torch
    from ultralytics import YOLO

    merged_dir = os.path.join(Config.FRAMEWORK_DIR, f"merged_iter{iteration}")
    _, data_yaml, stats, used_datasets, names_list = _merge_datasets(merged_dir)

    if stats["images"] < 100:
        raise ValueError(
            f"Not enough labeled images in merged dataset ({stats['images']}). "
            f"Download more datasets first (search_datasets/download_dataset)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try requested base model, then fallbacks
    candidates = []
    if strategy.get("base_model"):
        candidates.append(strategy["base_model"])
    candidates.append(Config.DETECTION_MODEL)
    for fb in getattr(Config, "DETECTION_MODEL_FALLBACKS", []):
        if fb not in candidates:
            candidates.append(fb)

    model = None
    base_weights = None
    last_err = None
    for cand in candidates:
        try:
            logger.info(f"[Mega] Trying base model: {cand}")
            model = YOLO(cand)
            base_weights = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(f"[Mega] {cand} unavailable: {e}")
    if model is None:
        raise RuntimeError(f"No base model loaded. Tried: {candidates}. Last error: {last_err}")

    logger.info(f"[Mega] base={base_weights}, imgs={stats['images']}, "
                f"classes={len(names_list)}, datasets={used_datasets}")
    project_dir = os.path.join(Config.FRAMEWORK_DIR, f"mega_iter{iteration}")

    model.train(
        data=data_yaml,
        epochs=strategy.get("epochs", 100),
        batch=strategy.get("batch_size", -1),
        imgsz=strategy.get("imgsz", 640),
        device=device,
        project=project_dir,
        name="train",
        patience=strategy.get("patience", 30),
        lr0=strategy.get("lr", 0.001),
        workers=strategy.get("workers", 4),
        verbose=False,
    )

    # Resolve actual save_dir (ultralytics increments train/train2/... if dir exists)
    best_pt = _resolve_best_pt(model, project_dir)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if not best_pt or not os.path.exists(best_pt):
        raise FileNotFoundError(
            f"Mega training finished but best.pt not found under {project_dir}. "
            f"Existing subdirs: {[p.name for p in Path(project_dir).glob('*') if p.is_dir()]}"
        )

    # Mark all used datasets as trained
    disc = DatasetDiscovery()
    for ds_name in used_datasets:
        disc.mark_as_used(
            ds_name,
            model_name=os.path.basename(base_weights),
            epochs=strategy.get("epochs", 100),
            result_summary={"iteration": iteration, "merged_images": stats["images"]},
        )

    summary = {
        "best_pt": best_pt,
        "merged_images": stats["images"],
        "datasets_used": used_datasets,
        "num_classes": len(names_list),
        "base_model": base_weights,
    }
    logger.info(f"[Mega] Complete: {summary}")
    return best_pt, summary
