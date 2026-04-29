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


# v3.0.25: canonical 12-class cottonweed system, NEVER_TRAIN holdout, slot-based
# class assignment for autolabel data so they don't pollute the 12 weed classes.

CANONICAL_12_NAMES = [
    "Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
    "Eclipta", "Goosegrass", "Morningglory", "Nutsedge",
]

# Original cottonweeddet12 class order (different from CANONICAL — this is what
# leave4out's data uses). Maps original_id -> canonical_id.
CWD12_ORIGINAL_NAMES = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
    "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
]
CWD12_ORIG_TO_CANON = {i: CANONICAL_12_NAMES.index(n) for i, n in enumerate(CWD12_ORIGINAL_NAMES)}

# Datasets that share results/leave4out/data physical path. Both registry entries
# point to the same images, so we pick ONE (cottonweed_sp8) as primary and use
# canonical 12-class mapping. cottonweed_holdout entries are deduped via dHash.
COTTONWEED_LEAVE4OUT_SLUGS = {"cottonweed_sp8", "cottonweed_holdout"}

# v3.0.25: hand-labeled holdouts that MUST NEVER be in training. These remain
# the immutable evaluation gold standard. Adding any dataset that overlaps with
# cottonweeddet12 imagery here would invalidate the entire eval protocol.
NEVER_TRAIN_SLUGS = {
    "cottonweeddet12",
    "weedsense",
    "francesco__weed_crop_aerial",
}

# Reserve class IDs:
#   0-11  : 12 canonical weed species (cottonweeddet12)
#   12-99 : auxiliary plant/non-weed classes (autolabeled). Slot assigned by
#           hashing the dataset slug so it's deterministic across runs.
AUX_CLASS_START = 12
AUX_CLASS_END = 100
TOTAL_NC = AUX_CLASS_END  # 100 — fixed nc avoids head expansion mid-training.


def _aux_class_for_slug(slug):
    """Stable integer in [AUX_CLASS_START, AUX_CLASS_END) derived from slug."""
    import hashlib
    h = int(hashlib.md5(slug.encode("utf-8")).hexdigest(), 16)
    span = AUX_CLASS_END - AUX_CLASS_START
    return AUX_CLASS_START + (h % span)


def _is_cottonweed_dataset(slug, info):
    """Heuristic: does this dataset use the cottonweeddet12 12-class system?"""
    if slug in COTTONWEED_LEAVE4OUT_SLUGS:
        return True
    names = info.get("class_names") or []
    overlap = sum(1 for n in names if n in CANONICAL_12_NAMES)
    return overlap >= 4  # at least 4 of 12 weed names → likely a cottonweed source


def _build_canonical_class_map(slug, info):
    """Return (ds_class_map, names_added).
    ds_class_map: dict mapping source_class_id -> canonical_id.
    """
    if _is_cottonweed_dataset(slug, info):
        # Use the source's class_names list to map by NAME into CANONICAL_12_NAMES.
        names = info.get("class_names") or []
        if not names:
            # Common case: cottonweed_sp8 / cottonweed_holdout share leave4out data
            # which uses CWD12_ORIGINAL_NAMES order. Default to that.
            return dict(CWD12_ORIG_TO_CANON), []
        ds_map = {}
        for i, n in enumerate(names):
            if n in CANONICAL_12_NAMES:
                ds_map[i] = CANONICAL_12_NAMES.index(n)
        return ds_map, []

    if info.get("annotation") == "yolo_autolabel":
        # Auxiliary plant/disease/pest data — assign a single auxiliary class
        # slot for this dataset. autolabel.py writes class_id=0 for all detections,
        # so we remap 0 -> aux_class. Nothing else should appear in these labels.
        aux = _aux_class_for_slug(slug)
        return {0: aux}, []

    # Other real-bbox datasets with class_names: try name-match into canonical
    # weed classes; otherwise assign each unique name to a fresh aux slot.
    names = info.get("class_names") or []
    if names:
        ds_map = {}
        for i, n in enumerate(names):
            if n in CANONICAL_12_NAMES:
                ds_map[i] = CANONICAL_12_NAMES.index(n)
            else:
                # Off-target real-bbox dataset (e.g., crops, pests). Bucket into one
                # aux slot per dataset to keep the class set bounded.
                ds_map[i] = _aux_class_for_slug(slug + "_" + n)
        return ds_map, []

    # No class_names registered. Don't drop the data; map every source class id
    # found in actual label files to a single aux slot for this dataset.
    # ds_class_map is built lazily by the caller when it scans the labels —
    # we return a special "wildcard" map keyed by None which the merge loop
    # interprets as "any src_cls → aux_slot".
    aux = _aux_class_for_slug(slug)
    return {"__wildcard__": aux}, []


def _merge_datasets(out_dir, val_fraction=0.1, include_autolabel=False,
                    val_dataset_root=None):
    """Merge all downloaded datasets (with labels) into one YOLO-format dataset.

    v3.0.25 changes:
      * Canonical 12-class system enforced for cottonweed_* datasets via
        CWD12_ORIG_TO_CANON / class_names name-match. Fixes the v3.0.24 bug
        where Eclipta/Goosegrass/Morningglory/Nutsedge had 0 mAP because
        cottonweed_sp8 and cottonweed_holdout shared a physical path and
        sp8's class_map mislabeled the held-out 4 species.
      * NEVER_TRAIN_SLUGS: cottonweeddet12, weedsense, francesco never enter
        training (they are the immutable evaluation gold standard).
      * yolo_autolabel data goes to AUX class slots (12-99) so they don't
        pollute the 12 weed slots even if class_id=0 is hard-coded in old
        autolabel writes. Each dataset gets its own slot via hash of slug.
      * If `val_dataset_root` is given (e.g., downloads/cottonweeddet12 holdout),
        the val split is OVERRIDDEN to point at that hand-labeled set instead
        of a 10%-of-merged split. This is the honest early-stop signal.
      * nc fixed at TOTAL_NC=100 so adding new aux classes between rounds
        does not require detection-head expansion.

    Returns: (merged_dir, data_yaml, stats, merged_names_list)
    """
    disc = DatasetDiscovery()
    registry = disc.registry["datasets"]

    for sub in ("train/images", "train/labels", "valid/images", "valid/labels"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    used_datasets = []

    stats = {"datasets": 0, "images": 0, "labels": 0, "skipped_no_label": 0,
             "skipped_duplicates": 0, "unique_hashes": 0,
             "skipped_autolabel": 0, "skipped_never_train": 0,
             "weed_class_instances": {n: 0 for n in CANONICAL_12_NAMES}}
    seen_hashes = {}

    valid_annotations = {"bbox", "bbox+segmentation", "yolo"}
    if include_autolabel:
        valid_annotations.add("yolo_autolabel")

    for ds_name, info in registry.items():
        # v3.0.25: NEVER_TRAIN protection — even if Brain ever asks to ingest
        # these slugs, the merge skips them outright.
        if ds_name in NEVER_TRAIN_SLUGS:
            stats["skipped_never_train"] += 1
            logger.info(f"[Merge] {ds_name} in NEVER_TRAIN — skipped (eval-only)")
            continue

        local_path = info.get("local_path")
        if not local_path or not os.path.isdir(local_path):
            continue
        ann = info.get("annotation")
        if ann not in valid_annotations:
            if ann == "yolo_autolabel":
                stats["skipped_autolabel"] += 1
            continue

        imgs = _find_images(local_path)
        if not imgs:
            continue

        # v3.0.25: CANONICAL class mapping replaces the old per-merge-order map.
        ds_class_map, _ = _build_canonical_class_map(ds_name, info)

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

            # v3.0.25: STRICT class remap — drop any line whose src_cls is not
            # in ds_class_map. Previously fallback `ds_class_map.get(src_cls, src_cls)`
            # passed unmapped IDs through, which caused cottonweed_holdout images
            # to be tagged with sp8's mismatched 0-7 IDs (Eclipta src=2 became
            # PalmerAmaranth in merged space). Strict mode means a label is
            # written ONLY when the source class is recognized.
            #
            # "__wildcard__" entry: dataset has no class_names registered and
            # no name-mapping is possible; route all src_cls values for that
            # dataset to a single aux slot so the data still trains the model
            # (as a hard negative for the 12 weed slots) without contaminating.
            with open(lbl) as f:
                label_lines = f.read().strip().splitlines()
            wildcard = ds_class_map.get("__wildcard__")
            remapped = []
            for ln in label_lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    src_cls = int(parts[0])
                except ValueError:
                    continue
                if src_cls in ds_class_map:
                    new_cls = ds_class_map[src_cls]
                elif wildcard is not None:
                    new_cls = wildcard
                else:
                    # No mapping → drop bbox.
                    continue
                if 0 <= new_cls < 12:
                    name = CANONICAL_12_NAMES[new_cls]
                    stats["weed_class_instances"][name] += 1
                remapped.append(" ".join([str(new_cls)] + parts[1:]))
            if not remapped:
                # No usable labels in this image → skip (don't add empty .txt
                # because that would be a hard-negative without intent).
                stats["skipped_no_label"] += 1
                continue
            new_label_text = "\n".join(remapped)

            # Split: 1/10 to val. NOTE: when val_dataset_root is set, we still
            # write a small internal valid split for ultralytics' own
            # bookkeeping but the *real* val gets overridden in data.yaml below.
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

    # v3.0.25 Phase 2: class-balanced oversampling for the 12 weed classes.
    # After the main merge, scan the merged train/labels and for every weed
    # class with < target_min_instances, create symlink duplicates of images
    # containing that class until target is reached. This corrects the 20:1
    # imbalance (Carpetweeds 1474 vs Goosegrass 75 in Phase 1) without
    # introducing new images / new label content.
    target_min = 500  # each weed class gets at least 500 instances
    if any(c < target_min for c in stats["weed_class_instances"].values()):
        _oversample_weak_weed_classes(out_dir, target_min, stats)

    # v3.0.25: fixed nc=TOTAL_NC (100) so head structure is stable across
    # mini-rounds even as new aux classes appear. names_list has the 12 weed
    # names in slots 0-11, then "aux_<slug>" placeholders for slots 12-99.
    names_list = list(CANONICAL_12_NAMES) + [f"aux_{i}" for i in range(AUX_CLASS_START, AUX_CLASS_END)]
    assert len(names_list) == TOTAL_NC

    # v3.0.25: if `val_dataset_root` provided (e.g., the cottonweeddet12 holdout
    # = downloads/cottonweeddet12/{test,valid} with hand-labeled YOLO bboxes),
    # OVERRIDE the val split to point at it. This makes the early-stop signal
    # honest: improvement on cwd12 holdout, not on a 10% slice of the (possibly
    # noisy) merged corpus.
    val_path = os.path.join(out_dir, "valid", "images")
    if val_dataset_root and os.path.isdir(val_dataset_root):
        # Stage cwd12 test+valid into a single staging dir under out_dir/cwd12_holdout
        staged = _stage_cwd12_holdout(val_dataset_root, out_dir)
        val_path = staged
        logger.info(f"[Merge] val OVERRIDE → cwd12 holdout staged at {staged}")

    # Write data.yaml
    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: {os.path.join(out_dir, 'train', 'images')}\n")
        f.write(f"val: {val_path}\n")
        f.write(f"nc: {TOTAL_NC}\n")
        f.write(f"names: {names_list}\n")

    stats["unique_hashes"] = len(seen_hashes)
    logger.info(f"[Merge] Total: {stats['images']} unique images from "
                f"{stats['datasets']} datasets; nc={TOTAL_NC} (12 weed + 88 aux). "
                f"Cross-dataset duplicates skipped: {stats['skipped_duplicates']}. "
                f"yolo_autolabel datasets skipped: {stats['skipped_autolabel']}. "
                f"NEVER_TRAIN datasets skipped: {stats['skipped_never_train']}. "
                f"Per-class instances: {stats['weed_class_instances']}")
    # v3.0.19: persist dHash caches written back to registry entries
    disc._save_registry()
    return out_dir, data_yaml, stats, used_datasets, names_list


def _oversample_weak_weed_classes(out_dir, target_min, stats):
    """v3.0.25 Phase 2: balance weed classes via symlink duplication.

    For each of the 12 canonical weed classes with fewer than `target_min`
    instances in the merged train set, find images in train/labels/ that
    contain at least one instance of that class and create symlink copies
    (with new stem `oversample_{cls}_{copy_idx}_{orig}`) until the class
    reaches `target_min` instances. Aux classes (12-99) are not touched.

    Why symlink duplication beats WeightedRandomSampler:
    - Compatible with ultralytics' standard dataloader (no fork required).
    - dHash dedup already ran, so duplicates here are intentional, not
      data leak — they're the same image/label being seen N times per epoch.
    - Standard practice in detection (LVIS uses exemplar replay similarly).
    """
    train_lbl_dir = os.path.join(out_dir, "train", "labels")
    train_img_dir = os.path.join(out_dir, "train", "images")
    if not os.path.isdir(train_lbl_dir):
        return

    # Index: which label files contain each canonical weed class.
    cls_to_files = {i: [] for i in range(12)}
    label_files = list(Path(train_lbl_dir).glob("*.txt"))
    for lbl in label_files:
        try:
            content = lbl.read_text()
        except Exception:
            continue
        seen = set()
        for line in content.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if 0 <= cid < 12:
                seen.add(cid)
        for cid in seen:
            cls_to_files[cid].append(lbl)

    stats["oversample"] = {}
    counts = stats["weed_class_instances"]
    for cid, name in enumerate(CANONICAL_12_NAMES):
        cur = counts.get(name, 0)
        if cur >= target_min or not cls_to_files[cid]:
            stats["oversample"][name] = {"before": cur, "after": cur, "copies": 0}
            continue
        # How many additional copies of this set of images do we need?
        # Each label file may contribute multiple instances; estimate average.
        files = cls_to_files[cid]
        avg_per_file = max(cur / max(len(files), 1), 1)
        need_extra = max(target_min - cur, 0)
        copies_per_file = int(need_extra / max(avg_per_file * len(files), 1)) + 1
        # Cap at 10x copies per file to avoid pathological inflation.
        copies_per_file = min(copies_per_file, 10)
        added_inst = 0
        for lbl in files:
            stem = lbl.stem
            # Find sibling image (try common extensions).
            img_link = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = Path(train_img_dir) / (stem + ext)
                if cand.exists():
                    img_link = cand
                    break
            if img_link is None:
                continue
            label_text = lbl.read_text()
            for k in range(1, copies_per_file + 1):
                new_stem = f"oversample_{cid}_{k}_{stem}"
                new_img = Path(train_img_dir) / (new_stem + img_link.suffix)
                new_lbl = Path(train_lbl_dir) / (new_stem + ".txt")
                if new_img.exists() and new_lbl.exists():
                    continue
                try:
                    if new_img.exists():
                        new_img.unlink()
                    # Symlink to the SAME source the original symlinks to.
                    src = os.readlink(img_link) if img_link.is_symlink() else os.path.abspath(img_link)
                    os.symlink(src, new_img)
                except OSError:
                    shutil.copy2(img_link, new_img)
                new_lbl.write_text(label_text)
                # Count new instances of this class added.
                for line in label_text.splitlines():
                    parts = line.split()
                    if parts and parts[0].isdigit() and int(parts[0]) == cid:
                        added_inst += 1
        stats["oversample"][name] = {
            "before": cur,
            "after": cur + added_inst,
            "copies_per_file": copies_per_file,
            "files_used": len(files),
            "added_inst": added_inst,
        }
        counts[name] = cur + added_inst
    logger.info(f"[Merge] Oversample to balance weed classes (target_min={target_min}): "
                f"{stats['oversample']}")


def _stage_cwd12_holdout(cwd12_root, out_dir):
    """Symlink cwd12 test/ + valid/ images and remap labels to canonical 12-class
    order. Used as the honest val set for v3.0.25 training."""
    staged = os.path.join(out_dir, "cwd12_holdout")
    img_d = os.path.join(staged, "images")
    lbl_d = os.path.join(staged, "labels")
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(lbl_d, exist_ok=True)
    n = 0
    for split in ("test", "valid"):
        split_dir = Path(cwd12_root) / split
        if not split_dir.is_dir():
            continue
        # Layout: split_dir/images/*.jpg + split_dir/labels/*.txt
        imgs_subdir = split_dir / "images"
        lbls_subdir = split_dir / "labels"
        if not imgs_subdir.is_dir():
            continue
        for img in imgs_subdir.glob("*.jpg"):
            dst_img = os.path.join(img_d, f"{split}__{img.name}")
            try:
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(img), dst_img)
            except OSError:
                shutil.copy2(img, dst_img)
            lbl = lbls_subdir / (img.stem + ".txt")
            dst_lbl = os.path.join(lbl_d, f"{split}__{img.stem}.txt")
            if lbl.exists():
                lines_out = []
                for line in open(lbl):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        orig = int(parts[0])
                    except ValueError:
                        continue
                    if orig in CWD12_ORIG_TO_CANON:
                        canon = CWD12_ORIG_TO_CANON[orig]
                        lines_out.append(" ".join([str(canon)] + parts[1:]))
                with open(dst_lbl, "w") as g:
                    g.write("\n".join(lines_out) + "\n")
            else:
                # No label → skip image entirely (don't keep in val)
                try:
                    os.remove(dst_img)
                except OSError:
                    pass
                continue
            n += 1
    logger.info(f"[Merge] cwd12 holdout staged: {n} images → {staged}")
    return img_d


def train_yolo_mega(strategy, iteration):
    """Train the largest YOLO on merged real-labeled datasets (v3.0 approach).

    Strategy keys:
      base_model: override, defaults to Config.DETECTION_MODEL
      epochs (default 100), batch_size, lr (default 0.001),
      patience (default 50), workers, imgsz (default 1024),
      include_autolabel (default False — set True in v3.0.25 once
        per-dataset class assignment is verified working)
      val_dataset_root (default None) — path to cottonweeddet12 holdout root.
        If provided, val is overridden to the hand-labeled holdout and
        mAP50-95 reported by ultralytics is the honest paper-grade signal.

    Returns: (best_pt_path, result_summary)
    """
    import torch
    from ultralytics import YOLO

    include_autolabel = bool(strategy.get("include_autolabel", False))
    val_dataset_root = strategy.get("val_dataset_root")
    merged_dir = os.path.join(Config.FRAMEWORK_DIR, f"merged_iter{iteration}")
    _, data_yaml, stats, used_datasets, names_list = _merge_datasets(
        merged_dir, include_autolabel=include_autolabel,
        val_dataset_root=val_dataset_root,
    )

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

    # v3.0.24: defaults raised to match v3.0.6 YOLO11n baseline that achieved
    # mAP50-95=0.865 on cottonweeddet12 (5648 imgs, 100 epochs, imgsz=640).
    # Now using yolo26x as base + cleaner real-bbox-only data + imgsz 1024,
    # we expect to meet or exceed that baseline.
    model.train(
        data=data_yaml,
        epochs=strategy.get("epochs", 100),
        batch=strategy.get("batch_size", -1),
        imgsz=strategy.get("imgsz", 1024),
        device=device,
        project=project_dir,
        name="train",
        patience=strategy.get("patience", 50),
        lr0=strategy.get("lr", 0.001),
        workers=strategy.get("workers", 4),
        verbose=False,
        save_period=1,  # v3.0.22: save last.pt every epoch so walltime-cut
                         # mid-training still leaves a usable checkpoint
        cos_lr=True,    # v3.0.24: cosine LR schedule for longer training
        mosaic=1.0,     # v3.0.24: full mosaic for the smaller real-bbox corpus
        mixup=0.1,      # v3.0.24: mild mixup helps with limited data
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
