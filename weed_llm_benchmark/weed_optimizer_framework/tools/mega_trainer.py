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
from .dataset_discovery import DatasetDiscovery, REGISTRY_PATH
from .registry_lock import update_registry

logger = logging.getLogger(__name__)


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp')


def _resolve_best_pt(model, project_dir, run_tag=None):
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
        # v3.25.0: a job-scoped run name (run_tag) does not start with "train",
        # so the directory scan below would miss it and the last.pt fallback
        # would go blind exactly when the trainer object is unavailable.
        patterns = ["train*"] + ([str(run_tag)] if run_tag else [])
        seen = set()
        train_dirs = []
        for pat in patterns:
            train_dirs += [p for p in Path(project_dir).glob(pat) if p.is_dir()]
        train_dirs = sorted(
            (d for d in train_dirs if not (str(d) in seen or seen.add(str(d)))),
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

# v3.0.28: stem-level defence. Even for SLUGS that are legitimately merge-eligible
# (e.g., cottonweed_sp8 / cottonweed_holdout, which contain the cwd12 train split
# AND a copy of the test+valid images mixed together), drop any image whose
# filename stem matches a cwd12 test/valid stem. This is the only correct fix for
# the v3.0.27 leak where 2,313 holdout copies entered training under cottonweed_*
# prefixes and bypassed the slug-level NEVER_TRAIN check.
# Sentinel source marker for pre-seeded holdout dHashes in `seen_hashes`, so a
# harvested image that collides with a holdout image is reported as a leak
# (skipped_holdout_hash) rather than a benign cross-dataset duplicate.
HOLDOUT_HASH_SENTINEL = "__HOLDOUT__"


def _holdout_image_dirs():
    """The cwd12 test+valid image dirs (the immutable eval set). Resolved against
    both a cwd-relative path and the absolute project path so this works
    regardless of caller cwd. Returns only dirs that actually exist."""
    candidates = [
        Path("downloads/cottonweeddet12") / "test" / "images",
        Path("downloads/cottonweeddet12") / "valid" / "images",
        Path("/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark") / "downloads/cottonweeddet12" / "test" / "images",
        Path("/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark") / "downloads/cottonweeddet12" / "valid" / "images",
    ]
    return [c for c in candidates if c.is_dir()]


def _iter_holdout_images():
    """Yield every cwd12 test+valid image path (dedup by resolved absolute path so
    the two candidate roots pointing at the same files aren't double-counted)."""
    seen_paths = set()
    for c in _holdout_image_dirs():
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
            for p in c.glob(ext):
                rp = str(p.resolve())
                if rp not in seen_paths:
                    seen_paths.add(rp)
                    yield p


def _load_holdout_stems():
    """Stems of cwd12 test+valid — the cheap first-pass filename filter."""
    return {p.stem for p in _iter_holdout_images()}


def _load_holdout_dhashes():
    """v3.1: content-level holdout guard. dHash of every cwd12 test+valid image,
    so a re-exported / renamed copy of a holdout image (e.g. Roboflow's
    `orig_jpg.rf.<hex>.jpg`, whose stem no longer matches) is still caught and
    dropped from training. The stem filter alone only blocks copies that KEPT
    their original filename; this closes the rename-bypass leak that inflates the
    'never-train' holdout mAP. Returns {dhash_int: '__HOLDOUT__'}."""
    hashes = {}
    for p in _iter_holdout_images():
        h = _dhash(p)
        if h is not None:
            hashes[h] = HOLDOUT_HASH_SENTINEL
    return hashes

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
                    val_dataset_root=None, min_dino_score=None):
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
             "skipped_holdout_stem": 0, "skipped_holdout_hash": 0,
             "skipped_user_flag": 0,
             "skipped_low_dino": 0,
             "weed_class_instances": {n: 0 for n in CANONICAL_12_NAMES}}
    seen_hashes = {}

    # v3.0.28: load cwd12 holdout stems to block alias contamination at the
    # per-image level. See NEVER_TRAIN_SLUGS comment above for rationale.
    holdout_stems = _load_holdout_stems()
    logger.info(f"[Merge] holdout stem filter active: {len(holdout_stems)} stems "
                f"blocked from training regardless of source slug")

    # v3.1: content-level holdout guard. Pre-seed the dedup set with the dHash of
    # every cwd12 test+valid image so a renamed / re-exported holdout copy (whose
    # stem no longer matches, so the stem filter above misses it) is caught by the
    # existing cross-dataset dedup and dropped. Without this, the stem filter and
    # NEVER_TRAIN slug list are the ONLY guards, and both are filename-based — a
    # Roboflow/Kaggle re-upload of cwd12 test images leaks straight into training
    # and silently inflates the 'never-train' holdout mAP.
    seen_hashes.update(_load_holdout_dhashes())
    logger.info(f"[Merge] holdout dHash guard active: {len(seen_hashes)} holdout "
                f"image hashes pre-seeded (renamed holdout copies now blocked)")

    # v3.0.30.1: user-driven REQ-3 quality feedback — slugs the user marked
    # as garbage via the dashboard get skipped here. The flags file is
    # written by dashboard_server.py (POST /api/flag/{slug}).
    flags_path = os.path.join(os.path.dirname(__file__), "..", "..",
                              "results", "framework", "dataset_flags.json")
    flags_path = os.path.abspath(flags_path)
    user_flags = {}
    if os.path.isfile(flags_path):
        try:
            import json as _json
            with open(flags_path) as f:
                user_flags = _json.load(f)
            garbage = [s for s, fd in user_flags.items()
                       if isinstance(fd, dict) and fd.get("flag") == "garbage"]
            logger.info(f"[Merge] user-flag filter active: {len(garbage)} slugs "
                        f"marked garbage will be skipped: {garbage[:5]}"
                        f"{'…' if len(garbage) > 5 else ''}")
        except Exception as e:
            logger.warning(f"[Merge] could not load user flags from {flags_path}: {e}")

    # v3.0.99.28 (D): DINOv2 quality gate for the "clean subset" experiment.
    # When strategy sets min_dino_score, slugs whose trusted-pool similarity score
    # (results/framework/dinov2_curator/slug_scores.json, written by dinov2_curator)
    # is below the threshold are dropped from training. This lets us build a
    # quality-core training set (e.g. min_dino_score=0.45 keeps only cotton-like
    # high-similarity weed data) to test the "quality > quantity" hypothesis
    # against the 175K-noisy baseline. Off-topic garbage (coconut/beehive ~0.12)
    # is excluded automatically. None = no gate (default, back-compat).
    dino_scores = {}
    if min_dino_score is not None:
        dino_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                 "results", "framework", "dinov2_curator",
                                 "slug_scores.json")
        dino_path = os.path.abspath(dino_path)
        try:
            import json as _json2
            with open(dino_path) as f:
                raw = _json2.load(f) or {}
            for s, rec in raw.items():
                if isinstance(rec, dict) and rec.get("score") is not None:
                    dino_scores[s] = float(rec["score"])
            logger.info(f"[Merge] DINO gate active: min_dino_score={min_dino_score}; "
                        f"{len(dino_scores)} slugs scored; "
                        f"{sum(1 for v in dino_scores.values() if v < min_dino_score)} "
                        f"below threshold will be skipped")
        except Exception as e:
            logger.warning(f"[Merge] min_dino_score set but could not load "
                           f"{dino_path}: {e} — DINO gate DISABLED this run")

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

        # v3.0.30.1: user flagged this slug as garbage from the dashboard.
        slug_flag = (user_flags.get(ds_name) or {}).get("flag")
        if slug_flag == "garbage":
            stats["skipped_user_flag"] += 1
            logger.info(f"[Merge] {ds_name} user-flagged GARBAGE — skipped. "
                        f"reason: {(user_flags[ds_name] or {}).get('reason','')[:80]}")
            continue

        # v3.22.9: registry-level quarantine (SUPERWEED_PLAN S1) — sources parked
        # with a reason (off-goal, failed sample audit, license problem) never
        # train, but are kept on disk and listed grey in the UI, not deleted.
        if str(info.get("status")) == "quarantined":
            stats["skipped_quarantined"] = stats.get("skipped_quarantined", 0) + 1
            logger.info(f"[Merge] {ds_name} QUARANTINED — skipped. reason: "
                        f"{str(info.get('quarantine_reason'))[:80]}")
            continue

        # v3.0.99.28 (D): DINOv2 quality gate — drop low-similarity slugs when the
        # clean-subset experiment sets min_dino_score. A slug with NO score is kept
        # (don't penalize unscored data); only an explicit below-threshold score skips.
        if min_dino_score is not None and ds_name in dino_scores \
                and dino_scores[ds_name] < min_dino_score:
            stats["skipped_low_dino"] += 1
            logger.info(f"[Merge] {ds_name} DINO score {dino_scores[ds_name]:.3f} "
                        f"< {min_dino_score} — skipped (clean-subset gate)")
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

            # v3.0.28: stem-level holdout filter. cottonweed_sp8 and
            # cottonweed_holdout legitimately provide cwd12 train images, but
            # their physical directories ALSO contain copies of test+valid
            # imagery (the v3.0.27 leak). Drop here regardless of slug.
            if img.stem in holdout_stems:
                stats["skipped_holdout_stem"] += 1
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
                    if seen_hashes[h] == HOLDOUT_HASH_SENTINEL:
                        # v3.1: this harvested image is a content-match for a cwd12
                        # holdout image (renamed copy that slipped past the stem
                        # filter). Dropping it is what keeps the eval set out of
                        # training. Tracked separately so leakage is observable.
                        stats["skipped_holdout_hash"] += 1
                    else:
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
            # v3.0.99.31: cap the per-file name to stay under NAME_MAX (255 bytes).
            # Roboflow slugs + augmented filenames (e.g. *_jpg.rf.<long-hash>) can
            # concatenate past 255 → OSError(36) ENAMETOOLONG crashed the whole
            # v3.0.99 clean-train merge. Truncate + append a stable hash so names
            # stay unique. Keep the matching label name in sync via dst_stem.
            if len(dst_stem.encode("utf-8")) > 180:
                import hashlib as _hl
                _h = _hl.md5(dst_stem.encode("utf-8")).hexdigest()[:12]
                dst_stem = dst_stem[:160] + "_" + _h
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
                f"v3.0.28 holdout-stem filter dropped: {stats['skipped_holdout_stem']}. "
                f"v3.0.30 user-flag GARBAGE skipped: {stats['skipped_user_flag']}. "
                f"Per-class instances: {stats['weed_class_instances']}")
    # v3.0.19 / v3.1: persist ONLY the dHash caches we computed, via a locked
    # re-read-modify-write. The old `disc._save_registry()` here flushed the
    # WHOLE registry snapshot that DatasetDiscovery loaded at merge start — so a
    # multi-hour merge would erase every slug a concurrent harvest (Job-D) wrote
    # during that window (the framework's worst lost-update). Now we graft only
    # dhash_cache onto the CURRENT on-disk registry and leave the rest untouched.
    dhash_updates = {
        ds: entry["dhash_cache"]
        for ds, entry in disc.registry["datasets"].items()
        if entry.get("dhash_cache")
    }

    def _graft_dhash(reg):
        dsets = reg.setdefault("datasets", {})
        for ds, cache in dhash_updates.items():
            if ds in dsets:  # don't resurrect a slug Job-D deleted mid-merge
                dsets[ds]["dhash_cache"] = cache

    if dhash_updates:
        update_registry(REGISTRY_PATH, _graft_dhash)
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


def train_yolo_mega(strategy, iteration, run_tag=None):
    """Train the largest YOLO on merged real-labeled datasets (v3.0 approach).

    Strategy keys:
      base_model: override, defaults to Config.DETECTION_MODEL
      epochs (default 100), batch_size, lr (default 0.001),
      patience (default 50), workers, imgsz (default 1024),
      seed (default 0) — RNG seed passed to ultralytics; vary it to obtain a
        mean +/- std over repeats instead of a single run (v3.22.3)
      deterministic (default True)
      include_autolabel (default False — set True in v3.0.25 once
        per-dataset class assignment is verified working)
      val_dataset_root (default None) — path to cottonweeddet12 holdout root.
        If provided, val is overridden to the hand-labeled holdout and
        mAP50-95 reported by ultralytics is the honest paper-grade signal.
      min_dino_score (default None) — v3.0.99.28 (D) clean-subset gate. When set,
        slugs with DINOv2 trusted-pool score below it are dropped from training
        (quality-core experiment). Unscored slugs are kept.
      time_h (default None) — v3.25.0 wall-clock cap in hours, passed to
        ultralytics `time=`. It overrides `epochs` and still ends with a valid
        best.pt. None (the default) keeps the uncapped epoch budget.
      trace_path (default None) — JSONL file that receives one record per
        validated epoch plus a start/end pair. None writes no trace.
      trace_meta (default None) — dict merged into every trace record
        (domain, round, step, job_id).

    run_tag (default None) — ultralytics run name. None keeps the historical
      "train", which ultralytics auto-increments to train2/train3 when the
      directory already exists; that ambiguity is how a metric reader can
      attach results.csv from a foreign run. A job-scoped tag makes the
      save_dir deterministic: <FRAMEWORK_DIR>/mega_iter<iteration>/<run_tag>/.

    Returns: (best_pt_path, result_summary)
    """
    import hashlib
    import time as _time

    import torch
    from ultralytics import YOLO

    # v3.25.0 tracing. The 2026-08-29 double TIMEOUT (12 h walltime reached at
    # epoch 24/60 and 16/60) left no per-epoch artifact at all, so nothing could
    # project the finish time while the job was still running. These records are
    # that evidence, and being evidence they must never abort a training run.
    _trace_path = strategy.get("trace_path")
    _trace_meta = strategy.get("trace_meta") or {}

    def _trace(record):
        if not _trace_path:
            return
        try:
            from .brain import trace as _trace_mod
            rec = dict(_trace_meta)
            rec.update(record)
            _trace_mod.append(_trace_path, rec)
        except Exception:
            pass

    def _walltime_s():
        # Exported by the job script from squeue; absent outside SLURM.
        try:
            return int(float(os.environ.get("SLURM_WALLTIME_S") or ""))
        except (TypeError, ValueError):
            return None

    def _file_sha256(path):
        # Empty string when the base is a bare model name that gets downloaded.
        try:
            p = Path(str(path))
            if not p.is_file():
                return ""
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    include_autolabel = bool(strategy.get("include_autolabel", False))
    val_dataset_root = strategy.get("val_dataset_root")
    min_dino_score = strategy.get("min_dino_score")  # v3.0.99.28 (D) clean-subset gate
    merged_dir = os.path.join(Config.FRAMEWORK_DIR, f"merged_iter{iteration}")
    _, data_yaml, stats, used_datasets, names_list = _merge_datasets(
        merged_dir, include_autolabel=include_autolabel,
        val_dataset_root=val_dataset_root, min_dino_score=min_dino_score,
    )

    if stats["images"] < 100:
        raise ValueError(
            f"Not enough labeled images in merged dataset ({stats['images']}). "
            f"Download more datasets first (search_datasets/download_dataset)."
        )

    # v3.25.2: a GPU job that cannot see a GPU must fail here, not train on the
    # CPU. Job 45250479 was allocated a GPU on a node whose driver was mismatched
    # ("Failed to initialize NVML: Driver/library version mismatch"), torch fell
    # back silently, one epoch took 48 minutes instead of ~2, and the run still
    # produced a real-looking mAP. A slow correct-looking number burned against an
    # unusable allocation is worse than a refusal, and Slurm keeps scheduling onto
    # such a node because it is not drained. WEED_ALLOW_CPU_TRAIN=1 re-enables the
    # fallback for a deliberate CPU run.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not os.environ.get("WEED_ALLOW_CPU_TRAIN"):
        gpus = (os.environ.get("SLURM_JOB_GPUS")
                or os.environ.get("SLURM_GPUS_ON_NODE")
                or os.environ.get("CUDA_VISIBLE_DEVICES") or "")
        if os.environ.get("SLURM_JOB_ID") or gpus:
            raise RuntimeError(
                "CUDA is not available in a job that was allocated GPUs "
                "(SLURM_JOB_ID=%s, gpus=%r). Training on the CPU here is ~25x "
                "slower and would report a metric against an allocation it never "
                "used. Check the node's driver (nvidia-smi) and resubmit; set "
                "WEED_ALLOW_CPU_TRAIN=1 only for a deliberate CPU run."
                % (os.environ.get("SLURM_JOB_ID"), gpus))

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

    t0 = _time.time()
    base_weights_sha256 = _file_sha256(base_weights)
    fresh_start = bool(strategy.get("fresh_start", False))

    # v3.25.0: `time=` is added only when a cap was asked for, so an uncapped
    # call passes exactly the arguments it passed before this change.
    train_kwargs = {}
    time_h = strategy.get("time_h")
    try:
        if time_h is not None and float(time_h) > 0:
            train_kwargs["time"] = float(time_h)
    except (TypeError, ValueError):
        logger.warning(f"[Mega] ignoring unparseable time_h={time_h!r}")
    if "time" in train_kwargs:
        logger.info(f"[Mega] Wall-clock cap: time={train_kwargs['time']}h "
                    f"(overrides epochs={strategy.get('epochs', 100)}; "
                    f"training still ends with a valid best.pt)")

    effective = {
        "epochs": strategy.get("epochs", 100),
        "imgsz": strategy.get("imgsz", 1024),
        "batch": strategy.get("batch_size", -1),
        "patience": strategy.get("patience", 50),
        "lr0": strategy.get("lr", 0.001),
        "workers": strategy.get("workers", 4),
        "seed": int(strategy.get("seed", 0)),
        "deterministic": bool(strategy.get("deterministic", True)),
        "name": run_tag or "train",
        "time_h": train_kwargs.get("time"),
    }

    def _on_fit_epoch_end(trainer):
        # One record per validated epoch. Every trainer attribute is read
        # defensively: an ultralytics API change must degrade the trace, not
        # kill the job.
        try:
            metrics = getattr(trainer, "metrics", None) or {}
            map5095 = None
            for k in metrics:
                if "mAP50-95" in str(k):
                    try:
                        map5095 = float(metrics[k])
                    except (TypeError, ValueError):
                        map5095 = None
                    break
            done = int(getattr(trainer, "epoch", 0) or 0) + 1
            # Under a `time=` cap ultralytics rewrites trainer.epochs to the
            # count it now expects to finish, which is the honest ETA base.
            total = int(getattr(trainer, "epochs", 0) or effective["epochs"] or 0)
            started = getattr(trainer, "train_time_start", None) or t0
            elapsed = max(0.0, _time.time() - float(started))
            eta = (elapsed / done) * total if done > 0 and total > 0 else None
            _trace({
                "kind": "epoch",
                "epoch": done,
                "map50_95": map5095,
                "elapsed_s": round(elapsed, 3),
                "eta_total_s": round(eta, 3) if eta is not None else None,
                "walltime_s": _walltime_s(),
                "save_dir": str(getattr(trainer, "save_dir", "") or ""),
            })
        except Exception:
            pass

    if _trace_path:
        try:
            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        except Exception as e:
            logger.warning(f"[Mega] epoch trace callback not registered: {e}")

    _trace({
        "kind": "start",
        "iteration": str(iteration),
        "run_tag": run_tag or "train",
        "project_dir": project_dir,
        "base_weights": base_weights,
        "base_weights_sha256": base_weights_sha256,
        "fresh_start": fresh_start,
        "merged_images": stats["images"],
        "datasets_used": len(used_datasets),
        "num_classes": len(names_list),
        "walltime_s": _walltime_s(),
        "strategy": effective,
    })

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
        name=run_tag or "train",
        patience=strategy.get("patience", 50),
        lr0=strategy.get("lr", 0.001),
        workers=strategy.get("workers", 4),
        verbose=False,
        save_period=1,  # v3.0.22: save last.pt every epoch so walltime-cut
                         # mid-training still leaves a usable checkpoint
        cos_lr=True,    # v3.0.24: cosine LR schedule for longer training
        mosaic=1.0,     # v3.0.24: full mosaic for the smaller real-bbox corpus
        mixup=0.1,      # v3.0.24: mild mixup helps with limited data
        # v3.22.3: explicit seed so a recipe can be repeated across seeds and
        # reported as mean +/- std instead of a single unreproducible run.
        # Ultralytics defaults to seed=0; passing it through makes the default
        # explicit and the sweep intentional.
        seed=int(strategy.get("seed", 0)),
        deterministic=bool(strategy.get("deterministic", True)),
        **train_kwargs,
    )

    # Resolve actual save_dir (ultralytics increments train/train2/... if dir exists)
    best_pt = _resolve_best_pt(model, project_dir, run_tag=run_tag)

    # Read save_dir off the trainer while the model still exists: it is the only
    # unambiguous pointer to the results.csv that belongs to THIS run.
    save_dir = ""
    try:
        save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", "") or "")
    except Exception:
        save_dir = ""
    if not save_dir and best_pt:
        save_dir = str(Path(best_pt).parent.parent)

    # A `time=` cap ends the run early and still reports COMPLETED, so a recipe
    # that asked for 60 epochs and got 24 looks identical to one that ran in
    # full unless the two counts are recorded side by side. Silence here would
    # replace the loud 2026-08-29 TIMEOUT with a quiet short run.
    epochs_completed = None
    try:
        _ep = getattr(getattr(model, "trainer", None), "epoch", None)
        if _ep is not None:
            epochs_completed = int(_ep) + 1
    except Exception:
        epochs_completed = None

    _trace({
        "kind": "end",
        "iteration": str(iteration),
        "run_tag": run_tag or "train",
        "save_dir": save_dir,
        "best_pt": best_pt,
        "ok": bool(best_pt and os.path.exists(best_pt)),
        "elapsed_s": round(_time.time() - t0, 3),
        "walltime_s": _walltime_s(),
        "epochs_requested": effective["epochs"],
        "epochs_completed": epochs_completed,
        "time_h": effective["time_h"],
    })

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

    # v3.0.19 / v3.1: persist best.pt pointer via locked re-read-modify-write so
    # this doesn't clobber concurrent Job-D writes (was disc._save_registry(),
    # which rewrote the whole loaded snapshot). Bump the round count against the
    # LATEST on-disk value, not this process's stale copy.
    def _graft_weights(reg):
        reg["last_mega_weights"] = best_pt
        reg["mega_round_count"] = int(reg.get("mega_round_count", 0)) + 1

    # v3.25.3: a probe run must not move campaign state. The WP1 walltime-cap
    # smoke completed one deliberate epoch and repointed last_mega_weights at its
    # own checkpoint, so the next real round would have continued from a probe and
    # deleting the probe directory would have broken the progressive chain
    # entirely. WEED_SMOKE=1 keeps the training and the artifacts and skips only
    # the registry mutation.
    if os.environ.get("WEED_SMOKE"):
        logger.warning("[Mega] WEED_SMOKE=1 — leaving last_mega_weights and "
                       "mega_round_count untouched (best_pt=%s)" % best_pt)
        written = {}
    else:
        written = update_registry(REGISTRY_PATH, _graft_weights)
        disc.registry["last_mega_weights"] = best_pt
        disc.registry["mega_round_count"] = written.get("mega_round_count", 1)
    if written:
        logger.info(f"[Mega] Saved last_mega_weights={best_pt} "
                    f"(mega_round_count={disc.registry['mega_round_count']})")

    summary = {
        "best_pt": best_pt,
        "merged_images": stats["images"],
        "datasets_used": used_datasets,
        "num_classes": len(names_list),
        "base_model": base_weights,
        "mega_round_count": disc.registry["mega_round_count"],
        # v3.25.0 run identity + lineage: which directory holds this run's
        # results.csv, and which checkpoint it continued from.
        "save_dir": save_dir,
        "run_tag": run_tag or "train",
        "base_weights": base_weights,
        "base_weights_sha256": base_weights_sha256,
        "fresh_start": fresh_start,
        "epochs_requested": effective["epochs"],
        "epochs_completed": epochs_completed,
        "time_h": effective["time_h"],
    }
    logger.info(f"[Mega] Complete: {summary}")
    return best_pt, summary
