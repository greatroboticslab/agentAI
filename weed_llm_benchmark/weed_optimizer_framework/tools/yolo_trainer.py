"""
YOLO Trainer — Train YOLO with pseudo-labels and replay buffer.

This is the ONLY model that gets fine-tuned. All VLMs are read-only.
Handles:
- Dataset assembly (new pseudo-labels + replay buffer from old species)
- YOLO training with configurable hyperparameters
- Cleanup after training to free disk space
"""

import os
import gc
import random
import shutil
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


def _count_nonempty_labels(label_dir):
    """Count label files that have actual content."""
    count = 0
    for f in os.listdir(label_dir):
        if not f.endswith('.txt'):
            continue
        path = os.path.join(label_dir, f)
        with open(path) as fh:
            if fh.read().strip():
                count += 1
    return count


def _assemble_dataset(strategy, label_dir, iteration):
    """Assemble training dataset: pseudo-labels + replay buffer + validation set.

    Returns: (dataset_dir, data_yaml_path, stats)
    """
    ds_dir = os.path.join(Config.FRAMEWORK_DIR, f"dataset_iter{iteration}")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    stats = {"new_images": 0, "replay_images": 0, "valid_images": 0}

    # --- 1. New pseudo-labeled images ---
    holdout_imgs_dir = os.path.join(Config.HOLDOUT_DIR, "train", "images")
    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"):
            continue
        stem = lbl_file.replace(".txt", "")

        # Check label has content
        lbl_path = os.path.join(label_dir, lbl_file)
        with open(lbl_path) as f:
            if not f.read().strip():
                continue

        # Find corresponding image
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img_src = os.path.join(holdout_imgs_dir, stem + ext)
            if os.path.exists(img_src):
                dst_name = f"new_{stem}"
                shutil.copy2(img_src, os.path.join(ds_dir, f"train/images/{dst_name}{ext}"))
                shutil.copy2(lbl_path, os.path.join(ds_dir, f"train/labels/{dst_name}.txt"))
                stats["new_images"] += 1
                break

    # --- 2. Replay buffer (old species images) ---
    replay_ratio = strategy.get("replay_ratio", 0.3)
    old_imgs_dir = os.path.join(Config.SP8_DIR, "train", "images")
    old_lbls_dir = os.path.join(Config.SP8_DIR, "train", "labels")

    if os.path.isdir(old_imgs_dir):
        all_old = [f for f in os.listdir(old_imgs_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Calculate replay count
        n_new = stats["new_images"]
        if n_new > 0 and replay_ratio > 0:
            n_replay = min(
                int(n_new * replay_ratio / max(1 - replay_ratio, 0.01)),
                len(all_old)
            )
        else:
            n_replay = min(int(len(all_old) * 0.1), 100)  # minimal replay

        random.seed(42 + iteration)
        replay_files = random.sample(all_old, min(n_replay, len(all_old)))

        for img_file in replay_files:
            stem = Path(img_file).stem
            lbl_src = os.path.join(old_lbls_dir, stem + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy2(os.path.join(old_imgs_dir, img_file),
                             os.path.join(ds_dir, "train/images", img_file))
                shutil.copy2(lbl_src,
                             os.path.join(ds_dir, "train/labels", stem + ".txt"))
                stats["replay_images"] += 1

    # --- 3. Validation set (from old species) ---
    valid_imgs_dir = os.path.join(Config.SP8_DIR, "valid", "images")
    valid_lbls_dir = os.path.join(Config.SP8_DIR, "valid", "labels")

    if os.path.isdir(valid_imgs_dir):
        for img_file in os.listdir(valid_imgs_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem = Path(img_file).stem
            lbl_src = os.path.join(valid_lbls_dir, stem + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy2(os.path.join(valid_imgs_dir, img_file),
                             os.path.join(ds_dir, "valid/images", img_file))
                shutil.copy2(lbl_src,
                             os.path.join(ds_dir, "valid/labels", stem + ".txt"))
                stats["valid_images"] += 1

    # --- 4. Create data.yaml ---
    names = Config.get_species_names()
    nc = len(names)
    data_yaml = os.path.join(ds_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(ds_dir, 'valid', 'images')}\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

    logger.info(f"Dataset assembled: {stats['new_images']} new, "
                f"{stats['replay_images']} replay, {stats['valid_images']} valid")
    return ds_dir, data_yaml, stats


def train_yolo(strategy, label_dir, iteration):
    """Train YOLO with the given labels and strategy config.

    Args:
        strategy: dict with lr, epochs, freeze_layers, replay_ratio, etc.
        label_dir: path to directory with pseudo-labels
        iteration: current iteration number

    Returns:
        path to best.pt weights file
    """
    import torch
    from ultralytics import YOLO

    # Check we have enough labels
    n_labels = _count_nonempty_labels(label_dir)
    if n_labels < Config.MIN_CONSENSUS_BOXES:
        raise ValueError(f"Too few labels ({n_labels} < {Config.MIN_CONSENSUS_BOXES})")

    # Assemble dataset
    ds_dir, data_yaml, ds_stats = _assemble_dataset(strategy, label_dir, iteration)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Base weights selection:
    #   1. strategy["base_model"] — explicit override (highest priority)
    #   2. strategy["use_legacy_baseline"]=True — use YOLO_8SP_WEIGHTS (leave4out / forgetting studies)
    #   3. Config.DETECTION_MODEL — v3.0 default (yolo26x with fallbacks)
    candidates = []
    if strategy.get("base_model"):
        candidates.append(strategy["base_model"])
    elif strategy.get("use_legacy_baseline") and os.path.exists(Config.YOLO_8SP_WEIGHTS):
        candidates.append(Config.YOLO_8SP_WEIGHTS)
    else:
        candidates.append(Config.DETECTION_MODEL)
        for fb in getattr(Config, "DETECTION_MODEL_FALLBACKS", []):
            if fb not in candidates:
                candidates.append(fb)
        # Legacy fallback as last resort
        if os.path.exists(Config.YOLO_8SP_WEIGHTS):
            candidates.append(Config.YOLO_8SP_WEIGHTS)

    model = None
    base_weights = None
    last_err = None
    for cand in candidates:
        try:
            logger.info(f"Trying base model: {cand}")
            model = YOLO(cand)
            base_weights = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(f"{cand} unavailable: {e}")
    if model is None:
        raise RuntimeError(f"No base model loaded. Tried: {candidates}. Last error: {last_err}")

    logger.info(f"Base weights chosen: {base_weights}")
    project_dir = os.path.join(Config.FRAMEWORK_DIR, f"yolo_iter{iteration}")

    logger.info(f"Training YOLO: lr={strategy.get('lr', 0.001)}, "
                f"epochs={strategy.get('epochs', 50)}, "
                f"freeze={strategy.get('freeze_layers', 0)}, "
                f"batch={strategy.get('batch_size', -1)}")

    model.train(
        data=data_yaml,
        epochs=strategy.get("epochs", 50),
        batch=strategy.get("batch_size", -1),
        device=device,
        project=project_dir,
        name="train",
        patience=strategy.get("patience", 15),
        lr0=strategy.get("lr", 0.001),
        freeze=strategy.get("freeze_layers", 0),
        workers=4,  # limit to prevent OOM (cluster has 5 CPUs/task)
        verbose=False,
    )

    # Resolve actual save_dir (ultralytics auto-increments: train, train2, train3, ...)
    best_pt = _resolve_best_pt(model, project_dir)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Remove assembled dataset to save disk
    shutil.rmtree(ds_dir, ignore_errors=True)

    if not best_pt or not os.path.exists(best_pt):
        raise FileNotFoundError(f"Training failed: best.pt not found under {project_dir}")

    logger.info(f"Training complete: {best_pt}")
    return best_pt


def _resolve_best_pt(model, project_dir):
    """Resolve the actual best.pt path after ultralytics training.

    Ultralytics auto-increments save_dir (train, train2, ...) when the dir already exists.
    Prefer model.trainer.save_dir; fall back to the newest train*/ subdir under project_dir.
    """
    # Preferred: ask the trainer directly
    try:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            cand = Path(save_dir) / "weights" / "best.pt"
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    # Fallback: newest train*/ dir by mtime
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
