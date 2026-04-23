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
    """Resolve actual best.pt after ultralytics training.

    v3.0.22: fallback to last.pt if best.pt doesn't exist (walltime cut
    training before first val epoch → no best.pt but last.pt saved every
    epoch). Prefer best.pt > newer last.pt. Without this fallback, the
    progressive training chain stalls: if mega gets cut mid-epoch-1, no
    best.pt → no last_mega_weights in registry → next round starts from
    yolo26x again → infinite no-progress loop.
    """
    candidates = []
    try:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            candidates.append(Path(save_dir) / "weights")
    except Exception:
        pass
    try:
        train_dirs = sorted(
            (p for p in Path(project_dir).glob("train*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in train_dirs:
            candidates.append(d / "weights")
    except Exception:
        pass

    # Preference 1: best.pt from the most recently-modified train dir
    for wdir in candidates:
        cand = wdir / "best.pt"
        if cand.exists():
            return str(cand)
    # Preference 2: last.pt fallback (walltime cut before first val)
    for wdir in candidates:
        cand = wdir / "last.pt"
        if cand.exists():
            logger.warning(f"[Mega] best.pt not found, using last.pt fallback: {cand}")
            return str(cand)
    return None


def _find_images(root):
    """Recursively find image files under root."""
    return [p for p in Path(root).rglob("*") if p.suffix in IMG_EXTS]


def _dhash(img_path, hash_size=8):
    """64-bit dHash for image duplicate detection. Pure PIL + numpy, no new deps.

    Standard dHash: resize to (hash_size+1, hash_size) grayscale, take horizontal
    pixel differences, pack as int. Two images with identical dHash are visually
    identical or near-identical (JPEG re-encoding, slight resize).
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path).convert("L").resize(
            (hash_size + 1, hash_size), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.int16)
        diff = arr[:, 1:] > arr[:, :-1]
        # Pack 64 bits into one Python int
        out = 0
        for bit in diff.flatten():
            out = (out << 1) | int(bit)
        return out
    except Exception:
        return None


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

    stats = {"datasets": 0, "images": 0, "labels": 0, "skipped_no_label": 0,
             "skipped_duplicates": 0, "unique_hashes": 0}
    # v3.0.16: cross-dataset image dedup via dHash. PlantVillage has 4+ Kaggle
    # mirrors in our registry (abdallahalidev, mohitsingh1804, arjuntejaswi,
    # vipoooool-augmented) — training on duplicates inflates apparent scale
    # and biases model toward over-represented sources. dHash exact match
    # catches identical+near-identical uploads; augmentations of the same
    # base image also tend to hash the same at 8x8 resolution.
    seen_hashes = {}

    for ds_name, info in registry.items():
        local_path = info.get("local_path")
        if not local_path or not os.path.isdir(local_path):
            continue
        if info.get("annotation") not in ("bbox", "bbox+segmentation", "yolo", "yolo_autolabel"):
            # v3.0.11: yolo_autolabel = OWLv2-generated pseudo-bboxes on
            # classification datasets. Quality is lower than real bbox but
            # these datasets (plant-village etc) are orders of magnitude
            # larger than the hand-labeled pool, and class-known OWLv2 is
            # much cleaner than blind VLM consensus (27% FP → target <10% FP).
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

        # v3.0.19: load dHash cache for this dataset if present. First run writes
        # the cache; subsequent rounds skip the 2-3ms-per-image hash compute
        # (185K images × 2ms = 6min saved per round; with auto-chain this
        # compounds over many rounds).
        cache = info.get("dhash_cache") or {}
        cache_updated = False
        ds_img_count = 0
        ds_dup_count = 0
        for img in imgs:
            lbl = _find_label_for_image(img, local_path)
            if lbl is None:
                stats["skipped_no_label"] += 1
                continue

            # v3.0.16: image-hash dedup across ALL datasets
            # v3.0.19: read from per-dataset cache first
            rel_key = str(img.relative_to(local_path))
            if rel_key in cache:
                h = cache[rel_key]
            else:
                h = _dhash(img)
                if h is not None:
                    cache[rel_key] = h
                    cache_updated = True
            if h is not None:
                if h in seen_hashes:
                    # Already saw an identical image from another dataset
                    stats["skipped_duplicates"] += 1
                    ds_dup_count += 1
                    continue
                seen_hashes[h] = ds_name

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
            dst_img = os.path.join(out_dir, bucket, "images", dst_stem + img.suffix)
            # v3.0.22: SYMLINK instead of copy. 244K file copies on /ocean took
            # 3h in v3.0.20 merge. Symlinks are nearly instant and ultralytics
            # follows them transparently. Labels still get written fresh (class
            # remapping differs per merge).
            try:
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(img), dst_img)
            except OSError:
                # Fallback to copy if symlink fails (rare on /ocean)
                shutil.copy2(img, dst_img)
            with open(os.path.join(out_dir, bucket, "labels", dst_stem + ".txt"), "w") as f:
                f.write(new_label_text)
            ds_img_count += 1
            stats["images"] += 1
            stats["labels"] += 1

        # v3.0.19: persist newly-computed hashes so next round skips recompute
        if cache_updated:
            registry[ds_name]["dhash_cache"] = cache

        if ds_img_count > 0:
            used_datasets.append(ds_name)
            stats["datasets"] += 1
            logger.info(f"[Merge] {ds_name}: {ds_img_count} unique images "
                        f"(+{ds_dup_count} deduped vs prior datasets; "
                        f"hash cache {'updated' if cache_updated else 'hit'})")

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

    stats["unique_hashes"] = len(seen_hashes)
    logger.info(f"[Merge] Total: {stats['images']} unique images from "
                f"{stats['datasets']} datasets; {len(names_list)} classes. "
                f"Cross-dataset duplicates skipped: {stats['skipped_duplicates']}. "
                f"(dedup via 8x8 dHash exact-match)")
    # v3.0.19: persist dHash caches written back to registry entries
    disc._save_registry()
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

    # v3.0.19: progressive training — if a prior round saved best.pt, use it as
    # base so each job picks up where the last left off. Data set can grow between
    # rounds so this is transfer-learning-continuation, not ultralytics `resume=True`.
    # registry["last_mega_weights"] is the checkpoint written by prior mega run.
    candidates = []
    if strategy.get("base_model"):
        candidates.append(strategy["base_model"])
    disc = DatasetDiscovery()
    last_ckpt = disc.registry.get("last_mega_weights")
    if last_ckpt and os.path.exists(last_ckpt) and not strategy.get("fresh_start"):
        logger.info(f"[Mega] Progressive: continuing from prior best.pt = {last_ckpt}")
        candidates.append(last_ckpt)
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
        save_period=1,  # v3.0.22: save last.pt every epoch so walltime-cut
                         # mid-training still leaves a usable checkpoint
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

    # v3.0.19: persist best.pt path for next round's progressive training
    disc.registry["last_mega_weights"] = best_pt
    disc.registry["mega_round_count"] = disc.registry.get("mega_round_count", 0) + 1
    disc._save_registry()
    logger.info(f"[Mega] Saved last_mega_weights={best_pt} "
                f"(mega_round_count={disc.registry['mega_round_count']})")

    summary = {
        "best_pt": best_pt,
        "merged_images": stats["images"],
        "datasets_used": used_datasets,
        "num_classes": len(names_list),
        "base_model": base_weights,
        "mega_round_count": disc.registry["mega_round_count"],
    }
    logger.info(f"[Mega] Complete: {summary}")
    return best_pt, summary
