"""
Dataset Discovery — Brain autonomously searches, downloads, and manages weed datasets.

Three key mechanisms:
1. Used tracking — datasets used for training are marked, not re-trained on blindly
2. Autonomous search — Brain finds NEW datasets beyond pre-researched list
3. Deduplication — checks before downloading, avoids redundant downloads

State persistence: dataset_registry.json tracks all known, downloaded, and used datasets.
"""

import os
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from ..config import Config

logger = logging.getLogger(__name__)

# Pre-researched datasets
KNOWN_DATASETS = {
    "weedsense": {
        "source": "huggingface", "hf_id": "baselab/weedsense",
        "images": 120341, "classes": 16,
        "annotation": "bbox+segmentation", "format": "voc_xml",
        "description": "Largest weed dataset. 16 species, VOC XML bboxes + segmentation.",
    },
    "deepweeds": {
        "source": "huggingface", "hf_id": "imsparsh/deepweeds",
        "images": 17509, "classes": 9,
        "annotation": "classification", "format": "csv",
        "description": "8 Australian weed species + negative. Classification only.",
    },
    "cottonweeddet12": {
        "source": "local", "hf_id": None,
        "images": 5648, "classes": 12,
        "annotation": "bbox", "format": "yolo",
        "description": "Current primary dataset. 12 cotton weed species.",
    },
    "crop_weed_research": {
        "source": "huggingface", "hf_id": "ivliev123/crop_weed_research_data",
        "images": 4307, "classes": "multi",
        "annotation": "bbox", "format": "voc_xml",
        "description": "Crop and weed bounding boxes in Pascal VOC.",
    },
    "grass_weeds": {
        "source": "huggingface", "hf_id": "Francesco/grass-weeds",
        "images": 2490, "classes": 2,
        "annotation": "bbox", "format": "coco",
        "description": "Grass vs weeds, COCO format.",
    },
    "weed_crop_aerial": {
        "source": "huggingface", "hf_id": "LibreYOLO/weed-crop-aerial",
        "images": 1176, "classes": 2,
        "annotation": "bbox", "format": "yolo",
        "description": "Aerial weed-crop, YOLO format ready.",
    },
    "rice_weeds_ph": {
        "source": "huggingface", "hf_id": "muromaine/Major_Rice_Weeds_Common_in_the_Philippines",
        "images": 4620, "classes": 10,
        "annotation": "classification", "format": "imagefolder",
        "description": "10 rice weed species from Philippines.",
    },
    "weeds7kpd": {
        "source": "huggingface", "hf_id": "LIU1248/Weeds7KPD",
        "images": 9330, "classes": 3,
        "annotation": "classification", "format": "imagefolder",
        "description": "7K+ weed images, 3 classes.",
    },
}

REGISTRY_PATH = os.path.join(Config.FRAMEWORK_DIR, "dataset_registry.json")

# v3.0.43.18: HF schema blacklist — record hf_ids whose schema we can't read
# so we don't waste time re-probing them on every harvest round.
HF_SCHEMA_BLACKLIST_PATH = os.path.join(
    Config.FRAMEWORK_DIR, "hf_schema_blacklist.json"
)


def _load_hf_blacklist() -> dict:
    """Return {hf_id: {reason, ts, n_failed_probes}}."""
    if not os.path.isfile(HF_SCHEMA_BLACKLIST_PATH):
        return {}
    try:
        with open(HF_SCHEMA_BLACKLIST_PATH) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _blacklist_hf_id(hf_id: str, reason: str):
    """Mark an HF id as 'don't waste time on this — schema doesn't fit'."""
    bl = _load_hf_blacklist()
    existing = bl.get(hf_id, {})
    bl[hf_id] = {
        "reason": reason[:200],
        "ts": datetime.now().isoformat(),
        "n_failed_probes": existing.get("n_failed_probes", 0) + 1,
    }
    try:
        os.makedirs(os.path.dirname(HF_SCHEMA_BLACKLIST_PATH), exist_ok=True)
        tmp = HF_SCHEMA_BLACKLIST_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(bl, f, indent=2)
        os.replace(tmp, HF_SCHEMA_BLACKLIST_PATH)
    except Exception as e:
        logger.warning(f"[HF blacklist] write fail: {e}")


class DatasetDiscovery:
    """Search, download, track, and deduplicate weed detection datasets."""

    def __init__(self):
        self.data_dir = os.path.join(Config.BASE_DIR, "datasets")
        os.makedirs(self.data_dir, exist_ok=True)
        self.registry = self._load_registry()
        # v3.0.108 (domain-aware harvest): per-harvest scope. Defaults to the
        # weed domain so all existing behavior is byte-identical. harvest_new_
        # datasets(domain=...) overrides these for a different collection agent.
        self._harvest_domain = "weed"
        self._accept_vocab = self.AG_VOCAB_ACCEPT
        self._reject_vocab = self.AG_VOCAB_REJECT

    # =========================================================
    # REGISTRY — persistent tracking of all datasets
    # =========================================================

    def _load_registry(self):
        """Load dataset registry (tracks downloaded, used, discovered).

        v3.0.42.5: don't blindly re-save on every __init__ — registry grew to
        50+MB after class_names backfill, and atomic os.replace is flaky on
        Lustre for large files. Only save if _discover_preexisting actually
        added new entries."""
        if os.path.exists(REGISTRY_PATH):
            # safe_read_json retries on JSONDecodeError, so a read that races a
            # concurrent mid-rename write recovers instead of being seen as corrupt.
            from .registry_lock import safe_read_json
            registry = safe_read_json(REGISTRY_PATH)
            if registry is None or "datasets" not in registry:
                # The file exists but is unparseable/malformed even after retries.
                # CRITICAL: do NOT fall through to rebuild-from-KNOWN_DATASETS and
                # save — that overwrites a recoverable (~50MB) registry with a
                # near-empty one and permanently loses every harvested slug. Back
                # it up and fail loud so a human restores a good copy.
                backup = f"{REGISTRY_PATH}.corrupt-{int(time.time())}"
                try:
                    shutil.copy2(REGISTRY_PATH, backup)
                except Exception:
                    backup = "(backup copy failed)"
                raise RuntimeError(
                    f"dataset_registry.json is corrupt/unparseable at {REGISTRY_PATH}. "
                    f"Refusing to silently rebuild an empty registry (that would erase "
                    f"all harvested data). Corrupt file preserved at {backup}. Restore "
                    f"from a good copy / snapshot before continuing.")
            before_keys = set(registry.get("datasets", {}).keys())
            self._discover_preexisting(registry)
            after_keys = set(registry.get("datasets", {}).keys())
            if after_keys != before_keys:
                # New local datasets discovered — worth saving.
                try:
                    self._save_registry(registry)
                except Exception as e:
                    logger.warning(f"[Dataset] registry resave failed (continuing): {e}")
            return registry

        # File genuinely absent (first run only) — build the default registry.
        registry = {"datasets": {}, "discovered": [], "total_downloaded": 0,
                    # v3.0.74: round-tracking. Bumped by start_new_round
                    # cluster_action. Every new slug downloaded gets
                    # tagged with this value.
                    "current_round": 1,
                    "rounds": {}}  # {round_n: {started_at, ended_at, dinov2_subversions: [...]}}
        for name, info in KNOWN_DATASETS.items():
            registry["datasets"][name] = {
                **info,
                "status": "known",
                "local_path": None,
                "local_images": 0,
                "class_names": info.get("class_names", []),
                "downloaded_at": None,
                "used_for_training": False,
                "training_runs": [],
            }
        self._discover_preexisting(registry)
        self._save_registry(registry)
        return registry

    def _discover_preexisting(self, registry):
        """Detect datasets that already exist on disk (cluster mount + local downloads)."""
        # Scan the downloads dir for any known dataset
        for name in KNOWN_DATASETS:
            path = os.path.join(self.data_dir, name)
            if os.path.isdir(path):
                n = sum(1 for f in Path(path).rglob("*") if f.suffix.lower() in
                        ('.jpg', '.jpeg', '.png', '.bmp'))
                if n > 0 and name in registry["datasets"]:
                    registry["datasets"][name]["status"] = "downloaded"
                    registry["datasets"][name]["local_path"] = path
                    registry["datasets"][name]["local_images"] = n

        # Auto-register existing leave4out splits so mega_trainer has something to train on
        # even if HF downloads haven't happened yet
        def _register_local(key, root_dir, class_names, desc):
            if not os.path.isdir(root_dir):
                return
            n = sum(1 for f in Path(root_dir).rglob("*") if f.suffix.lower() in
                    ('.jpg', '.jpeg', '.png', '.bmp'))
            if n < 10:
                return
            entry = registry["datasets"].get(key, {})
            entry.update({
                "source": "local", "hf_id": None,
                "images": n, "classes": len(class_names),
                "annotation": "bbox", "format": "yolo",
                "description": desc,
                "status": "downloaded", "local_path": root_dir, "local_images": n,
                "class_names": class_names,
                "downloaded_at": entry.get("downloaded_at"),
                "used_for_training": entry.get("used_for_training", False),
                "training_runs": entry.get("training_runs", []),
            })
            registry["datasets"][key] = entry

        _register_local(
            "cottonweed_sp8", Config.SP8_DIR,
            [Config.ALL_CLASSES[i] for i in sorted(Config.TRAIN_SPECIES_IDS)],
            "CottonWeedDet12 8-species train split (pre-existing, YOLO format)"
        )
        _register_local(
            "cottonweed_holdout", Config.HOLDOUT_DIR,
            [Config.ALL_CLASSES[i] for i in sorted(Config.HOLDOUT_SPECIES_IDS)],
            "CottonWeedDet12 4-species holdout split (pre-existing, YOLO format)"
        )

        registry["total_downloaded"] = sum(
            d.get("local_images", 0) for d in registry["datasets"].values()
        )

    def _save_registry(self, registry=None):
        """Save registry atomically under an advisory lock.

        Uses registry_lock.atomic_write_json (unique mkstemp temp), so concurrent
        writers can never interleave into a shared fixed `.tmp` and install a
        syntactically-broken file — which the corrupt-guard in _load_registry
        would then have to refuse. The lock serializes writers on the node."""
        if registry is None:
            registry = self.registry
        from .registry_lock import atomic_write_json, registry_lock, safe_read_json
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with registry_lock(REGISTRY_PATH):
            # v3.22.10 — MERGE-write instead of whole-write. The whole-write of a
            # snapshot loaded at job start was the audit's documented last-writer-
            # wins window, and it bit for real on 2026-08-23: a supervisor
            # quarantine (+ license backfill) written MID-harvest was erased by the
            # harvest's end-of-job save of its 28-minute-old dict. Under the lock we
            # re-read the latest on-disk registry and graft only OUR slugs onto it;
            # supervisory fields (quarantine block, provenance.license) on disk are
            # never resurrected-over by our stale copy.
            disk = safe_read_json(REGISTRY_PATH)
            if isinstance(disk, dict) and isinstance(disk.get("datasets"), dict):
                merged = disk
                for slug, info in (registry.get("datasets") or {}).items():
                    cur = merged["datasets"].get(slug)
                    if isinstance(cur, dict) and isinstance(info, dict):
                        nu = dict(info)
                        # v3.24.2 — SUPERVISORY FIELDS ARE NEVER WRITTEN BY A COLLECTOR.
                        # v3.22.10 protected the quarantine block and the license after a
                        # harvest erased a quarantine; on 2026-08-25 the same mechanism
                        # rolled a source's `scorecard` back from its corrected 23:04
                        # audit to the buggy-probe 20:48 one, because every other key
                        # still came from the harvester's stale snapshot. Enumerating the
                        # fields a harvest may not own fixes the class, not the instance:
                        # a data collector reports what it downloaded, never what a
                        # supervisor concluded about it.
                        for k in ("scorecard", "status", "quarantine_reason",
                                  "quarantined_at", "status_before_quarantine"):
                            if k in cur and k != "status":
                                nu[k] = cur[k]
                        if (cur.get("status") == "quarantined"
                                and info.get("status") != "quarantined"):
                            nu["status"] = cur["status"]
                        prov_d = cur.get("provenance") or {}
                        prov_o = nu.get("provenance") or {}
                        if prov_d.get("license") and not prov_o.get("license"):
                            keep = dict(prov_d)
                            keep.update({k: v for k, v in prov_o.items() if v})
                            nu["provenance"] = keep
                        merged["datasets"][slug] = nu
                    else:
                        merged["datasets"][slug] = info
                for k, v in registry.items():
                    if k != "datasets":
                        merged[k] = v
                merged["total_downloaded"] = sum(
                    (d.get("local_images", 0) or 0)
                    for d in merged["datasets"].values() if isinstance(d, dict))
                registry = merged
                self.registry = merged
            atomic_write_json(REGISTRY_PATH, registry)
        # Phase 3 dual-write: mirror to Mongo so new harvest lands there too.
        # Best-effort — never blocks/raises if Mongo is down (db handles it) — but
        # DO NOT silently discard the result: a swallowed failure is exactly how
        # the JSON and Mongo copies drift apart with no signal. Log it so the
        # divergence is observable.
        try:
            from . import db as _db
            res = _db.mirror_registry_to_mongo(registry)
            if isinstance(res, dict) and not res.get("ok", True):
                logger.warning(f"[Dataset] Mongo mirror reported failure "
                               f"(JSON saved, Mongo NOT updated — stores may drift): "
                               f"{res.get('error') or res}")
        except Exception as e:
            logger.warning(f"[Dataset] Mongo mirror raised (JSON saved, Mongo NOT "
                           f"updated — stores may drift): {type(e).__name__}: {e}")

    # =========================================================
    # STATUS TRACKING — mark datasets as used
    # =========================================================

    def mark_as_used(self, name, model_name, epochs, result_summary):
        """Mark a dataset as used for training."""
        if name in self.registry["datasets"]:
            ds = self.registry["datasets"][name]
            ds["status"] = "used_for_training"
            ds["used_for_training"] = True
            ds["training_runs"].append({
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "epochs": epochs,
                "result": result_summary,
            })
            self._save_registry()
            logger.info(f"[Dataset] Marked '{name}' as used for training")

    def get_unused_datasets(self):
        """Get datasets that have been downloaded but not yet used for training."""
        unused = []
        for name, info in self.registry["datasets"].items():
            if info["status"] == "downloaded" and not info["used_for_training"]:
                unused.append({"name": name, **info})
        return unused

    def get_used_datasets(self):
        """Get datasets that have been used for training."""
        used = []
        for name, info in self.registry["datasets"].items():
            if info["used_for_training"]:
                used.append({"name": name, **info})
        return used

    # =========================================================
    # DEDUPLICATION — prevent downloading same data twice
    # =========================================================

    def is_downloaded(self, name):
        """Check if a dataset is already downloaded."""
        ds = self.registry["datasets"].get(name, {})
        return ds.get("status") in ("downloaded", "used_for_training")

    def is_duplicate(self, hf_id):
        """Check if a HuggingFace dataset ID is already in registry (any name)."""
        for info in self.registry["datasets"].values():
            if info.get("hf_id") == hf_id:
                return True
        # Also check discovered list
        for d in self.registry.get("discovered", []):
            if d.get("hf_id") == hf_id:
                return True
        return False

    # =========================================================
    # SEARCH — Brain finds new datasets autonomously
    # =========================================================

    def search_huggingface(self, query="weed detection", max_results=20):
        """Search HuggingFace for NEW datasets not already in registry."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # huggingface_hub removed `direction` kwarg (v3.0.30.6 fix);
            # sort="downloads" already returns highest-downloads-first.
            datasets = api.list_datasets(search=query, sort="downloads",
                                          limit=max_results)
            results = []
            new_found = 0
            for d in datasets:
                is_dup = self.is_duplicate(d.id)
                entry = {
                    "hf_id": d.id,
                    "downloads": getattr(d, "downloads", 0),
                    "likes": getattr(d, "likes", 0),
                    "already_known": is_dup,
                }
                results.append(entry)

                # Auto-register newly discovered datasets
                if not is_dup:
                    self.registry["discovered"].append({
                        "hf_id": d.id,
                        "found_at": datetime.now().isoformat(),
                        "query": query,
                        "downloads": getattr(d, "downloads", 0),
                    })
                    new_found += 1

            if new_found > 0:
                self._save_registry()
                logger.info(f"[Dataset] Discovered {new_found} new datasets on HuggingFace")

            return results
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {e}")
            return [{"hf_id": v["hf_id"], "images": v["images"], "already_known": True}
                    for v in KNOWN_DATASETS.values() if v.get("hf_id")]

    # =========================================================
    # DOWNLOAD — with dedup check
    # =========================================================

    def download_dataset(self, name, max_images=None, force=False):
        """Download a dataset. Checks for duplicates first.

        v3.0.8: `force=True` bypasses the dedup check so v3.0.7's all-configs
        improvement can re-download weedsense (stuck at 1131 because harvest
        skips already-registered datasets).
        """
        if self.is_downloaded(name) and not force:
            info = self.registry["datasets"][name]
            logger.info(f"[Dataset] '{name}' already downloaded ({info['local_images']} images). Skipping.")
            return info.get("local_path", ""), {"status": "already_downloaded",
                                                  "images": info["local_images"]}

        if name not in self.registry["datasets"] and name not in KNOWN_DATASETS:
            return "", {"status": "unknown_dataset", "error": f"'{name}' not in registry"}

        info = self.registry["datasets"].get(name, KNOWN_DATASETS.get(name, {}))
        local_path = os.path.join(self.data_dir, name)
        os.makedirs(local_path, exist_ok=True)

        logger.info(f"[Dataset] Downloading '{name}' ({info.get('images', '?')} images)...")

        if info.get("source") == "local":
            return local_path, {"status": "local_dataset", "note": "Already on disk"}

        if info.get("source") == "huggingface" and info.get("hf_id"):
            return self._download_hf(name, info["hf_id"], local_path, max_images)

        return local_path, {"status": "unsupported_source"}

    @staticmethod
    def _extract_class_names_from_hf_features(features) -> list:
        """Walk HF Features tree looking for a ClassLabel (.names list).
        Most HF object-detection datasets nest like:
          {'image': Image, 'objects': Sequence({'category': ClassLabel(names=[...])})}
        We return the first non-empty .names found, or [].

        This closes the long-standing gap where _download_hf collected
        integer class_ids_seen but never recorded the actual class names —
        the root cause of the 78-empty-class_names registry state."""
        try:
            from datasets import ClassLabel, Sequence
        except Exception:
            return []

        def walk(f):
            if isinstance(f, ClassLabel) and f.names:
                return list(f.names)
            if isinstance(f, Sequence):
                return walk(f.feature)
            if isinstance(f, dict):
                for v in f.values():
                    r = walk(v)
                    if r:
                        return r
            return None

        try:
            if hasattr(features, "values"):
                for v in features.values():
                    r = walk(v)
                    if r:
                        return r
        except Exception:
            pass
        return []

    @staticmethod
    def _detect_class_names_filesystem(slug: str, local_path: str) -> tuple:
        """Fallback: reuse class_metadata_backfiller's detector on the
        downloaded directory (yaml/classes.txt/subdirs). Returns (names, source)."""
        try:
            from .class_metadata_backfiller import detect_class_names as _det
            from pathlib import Path as _P
            return _det(slug, _P(local_path))
        except Exception as e:
            logger.debug(f"[Dataset] backfiller fallback failed for {slug}: {e}")
            return [], ""

    def _extract_yolo_labels(self, item, width, height):
        """Convert various HF dataset annotation schemas to YOLO-format lines.

        Handles:
          - HuggingFace "detection" schema: item["objects"]["bbox"] + ["category"]
          - Flat: item["bbox"]/["boxes"] + item["labels"]/["category"]/["class"]
          - COCO bbox = [x, y, w, h] (absolute pixels)
          - VOC/xyxy bbox = [x1, y1, x2, y2]
        Returns (yolo_lines, class_names_seen).
        """
        def _to_yolo(box, cls):
            if len(box) != 4:
                return None
            # Heuristic: if the 3rd value > the 1st AND 4th > 2nd, treat as xyxy
            # Otherwise assume xywh (COCO)
            if box[2] > box[0] and box[3] > box[1] and box[2] > 1.0 and box[3] > 1.0:
                # Could be either; check if width/height exceed image dims
                if box[2] > width or box[3] > height:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w/2, y1 + h/2
                else:
                    # Assume xywh
                    x, y, w, h = box
                    cx, cy = x + w/2, y + h/2
            else:
                x, y, w, h = box
                cx, cy = x + w/2, y + h/2
            if width <= 0 or height <= 0:
                return None
            return f"{int(cls)} {cx/width:.6f} {cy/height:.6f} {w/width:.6f} {h/height:.6f}"

        lines = []
        classes_seen = set()

        # Pattern 1: HF object detection "objects" dict
        if "objects" in item and isinstance(item["objects"], dict):
            objs = item["objects"]
            bboxes = objs.get("bbox") or objs.get("boxes") or []
            cats = (objs.get("category") or objs.get("categories")
                    or objs.get("label") or objs.get("labels") or objs.get("class_id") or [])
            for b, c in zip(bboxes, cats):
                line = _to_yolo(list(b), c)
                if line:
                    lines.append(line)
                    classes_seen.add(int(c))
            return lines, classes_seen

        # Pattern 2: flat keys
        bboxes = item.get("bbox") or item.get("boxes") or item.get("bboxes")
        cats = item.get("labels") or item.get("category") or item.get("categories") or item.get("class")
        if bboxes is not None and cats is not None:
            # Some datasets have per-image single box with scalar label
            if not isinstance(bboxes[0], (list, tuple)):
                bboxes = [bboxes]
                cats = [cats]
            for b, c in zip(bboxes, cats):
                line = _to_yolo(list(b), c)
                if line:
                    lines.append(line)
                    classes_seen.add(int(c))
            return lines, classes_seen

        # Pattern 3: annotations list
        if "annotations" in item and isinstance(item["annotations"], list):
            for a in item["annotations"]:
                b = a.get("bbox") or a.get("box")
                c = a.get("category_id") or a.get("label") or a.get("class")
                if b and c is not None:
                    line = _to_yolo(list(b), c)
                    if line:
                        lines.append(line)
                        classes_seen.add(int(c))
            return lines, classes_seen

        return [], set()

    def _download_hf(self, name, hf_id, local_path, max_images):
        """Download from HuggingFace with schema-aware YOLO label extraction.

        v3.0.7: iterate ALL dataset configs.
        v3.0.8: also iterate ALL splits (train/validation/test/...). Some HF
        datasets (e.g. baselab/weedsense) only have 1 config but split their
        120K images across multiple splits — loading only 'train' gave 1131.
        """
        try:
            from datasets import load_dataset
            from datasets import get_dataset_config_names
            try:
                from datasets import get_dataset_split_names
            except Exception:
                get_dataset_split_names = None
        except Exception as e:
            return local_path, {"status": "error", "error": f"datasets import: {e}"}

        expected = KNOWN_DATASETS.get(name, {}).get("images", 0)
        limit = max_images or expected or 999999

        try:
            configs = get_dataset_config_names(hf_id) or [None]
        except Exception:
            configs = [None]
        if not configs:
            configs = [None]
        logger.info(f"[Dataset] {hf_id}: {len(configs)} config(s): "
                    f"{configs[:5]}{'...' if len(configs) > 5 else ''}")

        img_dir = os.path.join(local_path, "images")
        lbl_dir = os.path.join(local_path, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        count = 0
        label_count = 0
        all_classes = set()
        save_errors = 0
        # v3.0.42.5: actual class-name strings from HF features (not just IDs).
        # Populated from ClassLabel.names of the first probed (cfg, split).
        extracted_class_names: list = []
        class_names_source: str = ""
        # v3.0.11: default to needs_autolabel for image-only datasets (was
        # "classification"). Brain's autolabel_dataset tool will upgrade them
        # to yolo_autolabel after running OWLv2.
        annotation_kind = "needs_autolabel"

        try:
            for cfg in configs:
                if count >= limit:
                    break
                cfg_tag = (cfg or "default").replace("/", "_")[:40]

                # Enumerate splits for this config
                splits = ["train"]
                if get_dataset_split_names:
                    try:
                        sp = (get_dataset_split_names(hf_id, cfg)
                              if cfg else get_dataset_split_names(hf_id))
                        if sp:
                            splits = list(sp)
                    except Exception:
                        pass
                logger.info(f"[Dataset] {name}/{cfg_tag}: splits={splits}")

                cfg_start = count
                for split in splits:
                    if count >= limit:
                        break
                    split_tag = split.replace("/", "_")[:20]
                    # Probe this (cfg, split)
                    try:
                        probe_ds = (load_dataset(hf_id, cfg, split=split, streaming=True)
                                    if cfg else load_dataset(hf_id, split=split, streaming=True))
                        probe_item = next(iter(probe_ds))
                    except Exception as e:
                        logger.warning(f"[Dataset] {hf_id}/{cfg_tag}/{split_tag}: probe failed "
                                       f"({str(e)[:100]}) — skip")
                        continue

                    # v3.0.42.5: extract class names from HF schema once.
                    # Streaming datasets expose .features; ClassLabel.names is
                    # the real source of class strings. First successful extract wins.
                    if not extracted_class_names:
                        try:
                            feats = getattr(probe_ds, "features", None)
                            names = self._extract_class_names_from_hf_features(feats)
                            if names:
                                extracted_class_names = names
                                class_names_source = f"hf_features:{cfg_tag}/{split_tag}"
                                logger.info(f"[Dataset] {name}: class_names from HF features "
                                            f"({len(names)}): {names[:8]}{'...' if len(names) > 8 else ''}")
                        except Exception as e:
                            logger.debug(f"[Dataset] features extract fail {name}: {e}")

                    has_bbox = any(k in probe_item for k in
                                   ("objects", "bbox", "boxes", "bboxes", "annotations"))
                    if has_bbox:
                        annotation_kind = "bbox"
                    logger.info(f"[Dataset] {name}/{cfg_tag}/{split_tag}: "
                                f"keys={list(probe_item.keys())[:6]} bbox={has_bbox}")

                    use_streaming = expected > 10000 or (limit - count) > 10000
                    try:
                        ds = (load_dataset(hf_id, cfg, split=split, streaming=use_streaming)
                              if cfg else load_dataset(hf_id, split=split, streaming=use_streaming))
                    except Exception as e:
                        logger.warning(f"[Dataset] {hf_id}/{cfg_tag}/{split_tag}: load failed "
                                       f"({str(e)[:100]}) — skip")
                        continue

                    iterator = ds if use_streaming else iter(ds)
                    split_start = count

                    for item in iterator:
                        if count >= limit:
                            break
                        if "image" not in item:
                            count += 1
                            continue
                        img = item["image"]
                        w = getattr(img, "width", item.get("width", 0))
                        h = getattr(img, "height", item.get("height", 0))
                        stem = f"{cfg_tag}_{split_tag}_{count:06d}"

                        try:
                            if img.mode not in ("RGB", "L"):
                                if img.mode in ("RGBA", "LA"):
                                    bg = img.__class__.new("RGB", img.size, (255, 255, 255))
                                    try:
                                        bg.paste(img.convert("RGBA"),
                                                 mask=img.convert("RGBA").split()[-1])
                                    except Exception:
                                        bg = img.convert("RGB")
                                    img = bg
                                else:
                                    img = img.convert("RGB")
                            img.save(os.path.join(img_dir, f"{stem}.jpg"))
                        except Exception as e:
                            save_errors += 1
                            if save_errors <= 3:
                                logger.warning(f"[Dataset] {name} save fail on #{count} "
                                               f"mode={getattr(img,'mode','?')}: {str(e)[:100]}")
                            count += 1
                            continue

                        if has_bbox and w and h:
                            lines, classes = self._extract_yolo_labels(item, w, h)
                            if lines:
                                with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
                                    f.write("\n".join(lines))
                                label_count += 1
                                all_classes.update(classes)
                            else:
                                open(os.path.join(lbl_dir, f"{stem}.txt"), "w").close()

                        count += 1
                        if count % 2000 == 0:
                            logger.info(f"[Dataset] {name}/{cfg_tag}/{split_tag}: {count}/{limit} imgs, "
                                        f"{label_count} labeled, classes={len(all_classes)}")
                    # end for-item
                    logger.info(f"[Dataset] {name}/{cfg_tag}/{split_tag}: "
                                f"+{count - split_start} imgs this split")
                # end for-split
                cfg_added = count - cfg_start
                logger.info(f"[Dataset] {name}/{cfg_tag}: +{cfg_added} imgs total "
                            f"({label_count} labeled)")

            # v3.0.42.5: if HF features didn't yield names, fall back to the
            # filesystem-based detector (yaml/classes.txt/subdirs). This closes
            # Q1+Q2 (auto-display of new harvests by class) at harvest time
            # instead of needing a separate backfiller pass.
            if not extracted_class_names:
                try:
                    fs_names, fs_src = self._detect_class_names_filesystem(name, local_path)
                    if fs_names:
                        extracted_class_names = fs_names
                        class_names_source = f"fs:{fs_src}"
                        logger.info(f"[Dataset] {name}: class_names from filesystem "
                                    f"({fs_src}, {len(fs_names)}): "
                                    f"{fs_names[:8]}{'...' if len(fs_names) > 8 else ''}")
                except Exception as e:
                    logger.debug(f"[Dataset] fs fallback fail {name}: {e}")

            # All configs processed — register result
            self.registry["datasets"].setdefault(name, {**KNOWN_DATASETS.get(name, {})})
            update_dict = {
                "status": "downloaded",
                "local_path": local_path,
                "local_images": count,
                "local_labeled": label_count,
                "class_ids_seen": sorted(all_classes),
                "annotation": annotation_kind,
                "downloaded_at": datetime.now().isoformat(),
                "configs_iterated": len(configs),
                # v3.0.108: tag with the harvest's domain (defaults to "weed")
                "domain": getattr(self, "_harvest_domain", "weed"),
            }
            # v3.22.9 (S1 gate): capture the source license AT collection time —
            # the 2026-08-23 audit found 0/45 datasets with any license record.
            try:
                from .license_audit import detect_license
                prov = self.registry["datasets"][name].get("provenance") or {}
                if not prov.get("license") or prov.get("license") in ("unresolved", "unreachable"):
                    prov.update(detect_license(name, self.registry["datasets"].get(name)))
                    prov["license_checked_at"] = datetime.now().isoformat()
                    update_dict["provenance"] = prov
            except Exception as e:
                logger.warning(f"[Dataset] {name}: license capture failed: {e}")
            # Only WRITE class_names if we extracted any — never clobber a
            # previously curated set with [].
            if extracted_class_names:
                update_dict["class_names"] = extracted_class_names
                update_dict["class_names_source"] = class_names_source
                update_dict["class_names_backfilled_at"] = int(__import__("time").time())
            self.registry["datasets"][name].update(update_dict)

            # v3.0.43.3: auto-classify each new class_name into a topic
            # (weed/disease/pest/crop/other) via Brain LLM + keyword hybrid.
            # Persists to class_topic_overrides.json so /classes UI groups
            # new species correctly the moment harvest finishes.
            if extracted_class_names:
                try:
                    from .topic_classifier import classify_batch
                    from .class_topic_store import load_overrides
                    existing = load_overrides()
                    new_classes = [c for c in extracted_class_names if c not in existing]
                    if new_classes:
                        results = classify_batch(
                            new_classes, use_llm=True, persist=True,
                        )
                        # Summary log for audit
                        by_source = {}
                        by_topic = {}
                        for r in results:
                            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
                            t = r.get("topic", "?")
                            by_topic[t] = by_topic.get(t, 0) + 1
                        logger.info(
                            f"[Dataset] {name}: auto-classified {len(results)} new "
                            f"class topics — by_source={by_source} by_topic={by_topic}"
                        )
                    else:
                        logger.info(
                            f"[Dataset] {name}: all {len(extracted_class_names)} "
                            f"classes already have topic overrides — skip LLM"
                        )
                except Exception as e:
                    logger.warning(
                        f"[Dataset] {name}: topic auto-classify failed (continuing): {e}"
                    )
            self.registry["total_downloaded"] = sum(
                d.get("local_images", 0) for d in self.registry["datasets"].values()
            )
            self._save_registry()

            logger.info(f"[Dataset] '{name}': {count} imgs across {len(configs)} cfg(s), "
                        f"{label_count} labeled, {len(all_classes)} classes, "
                        f"kind={annotation_kind}")
            return local_path, {
                "status": "downloaded",
                "images": count,
                "labeled": label_count,
                "classes": len(all_classes),
                "annotation_kind": annotation_kind,
                "configs": len(configs),
            }
        except Exception as e:
            logger.error(f"[Dataset] Download failed: {e}")
            return local_path, {"status": "error", "error": str(e)}

    # =========================================================
    # HARVEST — each run discovers N new datasets, accumulates forever
    # =========================================================

    # v3.0.30.3 — user-tuned 2026-05-12: ONLY weed/crop/field plants.
    # Reject pest/insect/disease/tree/indoor per direct user spec:
    #   "我们只要plant. 但是这个plant主要是weed和作物 以及少部分的什么花朵啊
    #    或者什么路边的小草啊什么乱七八糟的在草地上或者农田上经常出现的植物.
    #    而不是大树啊虫害啊什么的乱七八糟的的"
    # → Accept: weed, crop, field plant, common roadside grass, agricultural flower
    # → Reject: pest, insect, disease, tree, decorative/indoor plant
    DEFAULT_HARVEST_QUERIES = [
        # Direct weed targets
        "weed", "weed detection", "weed yolo", "weed bounding box",
        "weed segmentation", "weed dataset",
        # Crop targets (the things weeds compete with)
        "crop detection", "crop dataset", "crop yolo",
        "cotton field", "rice field", "wheat field", "corn field", "maize field",
        "soybean field", "sugar beet field", "lettuce field",
        # Field / agricultural scenes (likely to contain weeds among crops)
        "agricultural field", "farmland", "field crops",
        "uav crop", "uav weed", "drone agriculture",
        # Common roadside/field grass + small wild plants
        "grass dataset", "wild plants field", "plant seedling",
        # Specific weed-related projects
        "cottonweed", "deepweeds", "weedsense", "broadleaf weed",
        "grass weed", "weed species",
        # Real-bbox-hinting keywords
        "yolo agriculture", "coco agriculture", "bounding box field",
        # v3.0.77 (2026-06-01): expanded to find more datasets after
        # earlier rounds exhausted basic terms. Round 2/3 needs fresh
        # candidates. These are species-specific + technique-specific.
        "amaranth", "ragweed", "morningglory dataset", "carpetweed",
        "crabgrass", "purslane plant", "sicklepod", "goosegrass",
        "palmer amaranth", "spurge weed", "nutsedge plant",
        # v3.0.99.6 (2026-06-08): bias to the 4 CWD12 species still missing data
        # (Eclipta/Goosegrass/Morningglory/Nutsedge). Eclipta had NO query before.
        "eclipta", "eclipta prostrata", "eclipta weed", "false daisy weed",
        "goosegrass detection", "eleusine indica", "nutsedge detection",
        "cyperus weed", "morning glory weed detection", "ipomoea weed",
        # Generic detection-relevant for plants
        "plant detection yolo", "leaf detection bbox", "field scene yolo",
        "agriculture object detection", "crop health detection",
        "early stage plant", "vegetation segmentation", "leaf segmentation",
        # New: bigger-scale general
        "yolov8 weed", "yolov5 crop", "weed control vision",
        "smart farming dataset", "precision agriculture dataset",
        "robot crop weeding", "crop row dataset",
        # Common synonyms / species + variants
        "horseweed", "pigweed", "lambsquarters", "thistle plant",
        "dandelion detection", "clover weed", "chickweed",
        "annual weed", "perennial weed", "broad leaf weeds",
        "grassy weed", "sedge weed",
    ]

    # Positive matches: a slug name / description containing any of these is
    # a likely WEED/CROP/FIELD-PLANT detection candidate.
    AG_VOCAB_ACCEPT = (
        "weed", "crop", "field", "farm", "agri", "agriculture",
        "cotton", "rice", "wheat", "corn", "maize", "soybean", "soy",
        "sugar beet", "sugarbeet", "lettuce", "tomato", "potato",
        "seedling", "sprout", "grass", "broadleaf",
        "uav", "drone", "aerial",
        "cottonweed", "deepweeds", "weedsense",
    )

    # Reject overrides: even if a slug matches positive vocab, drop it if it
    # also matches reject vocab. Per 2026-05-12 user spec.
    # v3.0.43.20 (2026-05-27): user audited /classes, found 14+ garbage slugs
    # that bypassed earlier filter. Reject now includes their exact substrings
    # so Brain CAN'T re-harvest these on next round.
    AG_VOCAB_REJECT = (
        # pests + insects (we keep disease but NOT bug)
        "pest", "insect", "bug-detect", "fly", "mosquito", "beetle",
        "weevil", "caterpillar", "earthworm", "grasshopper",
        "bee", "beehive", "honeybee", "spider", "ant-detect",
        "apiary", "honey", "pollinator",
        # v3.0.92: animals + specific off-target crops/categories that leaked in
        # (gh_minhvuongvu beehive, gh_vuyyurusairamreddy coconut-disease,
        #  project_agml cowpea). Block at source so they can't be re-harvested.
        "coconut", "cowpea", "poultry", "livestock", "cattle", "fish-detect",
        "fruit-detect", "flower-detect", "face-detect", "vehicle-detect",
        # v3.22.12: the maritime/aquatic family that got through on 2026-08-23
        # (dronefreak/SeaDronesSee = sea rescue, dronefreak/Brackish = underwater
        # fish). They were accepted on task_categories alone — see the bbox-hint
        # fix below, which no longer lets a detection tag bypass the vocabulary.
        "seadrones", "sea-drones", "brackish", "underwater", "marine",
        "maritime", "boat", "vessel", "swimmer", "sonar", "aquatic",
        "coral", "fishery", "shark", "turtle-detect",
        # ** v3.0.43.20 NEW: non-plant categories that slipped through **
        "price_tag", "price-tag",        # d_shatnev__price_tag_detection
        "commonform",                    # jbarrow/kurianmelvin/wewocram commonforms (forms!)
        "yonder",                        # astralhf__yonder
        "d-kap", "dkap",                 # kg_hsmcaju__d-kap
        "uvh_coco",                      # surenreddy__uvh_coco (generic COCO)
        "cheque", "bofa",                # ashishkamra79__bofa_cheques
        "tomatoweight",                  # gh_lunaimer (QR code, not weed)
        "qr-code", "qr_code", "qr-detect",
        "kapr", "kapp",                  # generic noise
        "agropestic",                    # pesticide dataset (off domain)
        "dangerous-insect", "scenes",    # wider non-plant scenes
        "warehouse", "robotics-arm",
        # NEW: trees specifically — coconut DISEASE is plant disease so OK, but
        # tree-* slugs are typically inventory not weed
        # plantdoc kept in reject (plant doc is generic doc, NOT weed-focused)
        # weeds vs leafdoc decision: still reject classification-only sets unless
        # they're clearly weed-labeled
        # v3.0.99.6 (2026-06-08): TIGHTEN against plant-DISEASE. Earlier we kept
        # disease sets as a "leaf visual prior for OWLv2 autolabel", but OWLv2
        # image-conditioned autolabel proved too imprecise (2026-06-07 finding),
        # so that rationale is gone. A tomato-leaf-DISEASE set leaked through as
        # "crop detection" (tomato ∈ ACCEPT) — reject disease vocab now (reject is
        # checked fail-fast BEFORE accept, so it overrides the 'tomato' match).
        "disease", "diseased", "leaf-disease", "leaf_disease", "leaf disease",
        "plant-disease", "plant disease", "plantdisease", "plantvillage",
        "leaf-spot", "leaf_spot", "leafspot", "leaf-blight", "anthracnose",
        "scab", "powdery", "downy", "necrosis", "necrotic", "pathogen",
        "blight", "rust-detect", "rot-detect", "infection",
        "virus", "fungus", "fungal", "mildew", "lesion", "bacterial-spot",
        # tree/non-plant retained
        "warp", "recycling", "waste",
        "tree-species", "tree-classification",
        "houseplant", "indoor plant", "decorative",
        "bonsai", "succulent",
        # generic classification (not detection)
        "image-classification",
    )

    # v3.0.108: domain-AGNOSTIC junk — rejected for ANY collection agent (forms,
    # codes, vehicles, faces, warehouse, generic COCO). The weed agent rejects
    # MORE (disease/pest/bee via AG_VOCAB_REJECT); a NEW domain uses only this so
    # its own topic (e.g. disease) is never blocked.
    GENERIC_REJECT = (
        "price_tag", "price-tag", "commonform", "yonder", "d-kap", "dkap",
        "uvh_coco", "cheque", "bofa", "qr-code", "qr_code", "qr-detect",
        "kapr", "kapp", "warehouse", "robotics-arm", "recycling", "waste",
        "face-detect", "vehicle-detect", "scenes", "warp",
    )

    def _card_suggests_bbox(self, ds_info):
        """Fast heuristic from HF dataset_info (no actual data load):
        - task_categories contains 'object-detection'
        - tags include detection/yolo/bbox
        - siblings list has .xml, annotations.json, labels.txt patterns
        Returns (has_bbox_hint, reason).
        """
        try:
            card = getattr(ds_info, "card_data", None) or {}
            tags = getattr(ds_info, "tags", []) or []
            # task_categories check
            tasks = []
            if isinstance(card, dict):
                tasks = card.get("task_categories") or []
            else:
                tasks = getattr(card, "task_categories", []) or []
            if any("detection" in str(t).lower() for t in tasks):
                # v3.22.12: a detection tag proves the ANNOTATION TYPE, never the
                # SUBJECT. Returning True here bypassed the topic vocabulary and
                # is exactly how maritime rescue and underwater fish datasets
                # entered the pool on 2026-08-23. The hint now has to survive a
                # subject check against the reject vocabulary.
                blob = " ".join([str(getattr(ds_info, "id", "")),
                                 " ".join(str(t) for t in tags)]).lower()
                bad = [w for w in getattr(self, "_reject_vocab", self.AG_VOCAB_REJECT)
                       if w in blob]
                if bad:
                    return False, f"reject-vocab {bad[:3]} (detection tag is not a subject)"
                return True, f"task_categories={tasks}"
            # tags check
            for t in tags:
                tl = str(t).lower()
                if any(k in tl for k in ["object-detection", "yolo", "bbox", "bounding-box"]):
                    return True, f"tag={t}"
            # sibling file patterns
            siblings = [getattr(s, "rfilename", "") for s in getattr(ds_info, "siblings", [])]
            patterns = [".xml", "annotations.json", "annotation", "labels.txt", "/labels/"]
            for p in patterns:
                if any(p in s.lower() for s in siblings):
                    return True, f"sibling pattern {p}"
            return False, "no bbox hints in card/tags/siblings"
        except Exception as e:
            return False, f"card probe err: {e}"

    def _slugify(self, hf_id):
        return hf_id.replace("/", "__").replace("-", "_").lower()[:60]

    def _is_relevant_dataset(self, slug_or_id: str, description: str = "") -> bool:
        """v3.0.30.3 relevance filter (user spec 2026-05-12):
        ACCEPT: weed / crop / field-plant / agricultural-flower / roadside-grass
        REJECT: pest / insect / disease / tree / indoor / decorative / classification-only

        Returns True if candidate likely contains weed-detection-relevant content.

        Both vocab lists are class attributes (AG_VOCAB_ACCEPT, AG_VOCAB_REJECT).
        """
        blob = (slug_or_id + " " + (description or "")).lower()
        # v3.0.108: use per-harvest vocab (defaults to the weed class attrs, so
        # weed behavior is unchanged; a non-weed domain swaps these in).
        reject_vocab = getattr(self, "_reject_vocab", self.AG_VOCAB_REJECT)
        accept_vocab = getattr(self, "_accept_vocab", self.AG_VOCAB_ACCEPT)
        # First check reject — fail-fast on obvious off-domain
        for r in reject_vocab:
            if r in blob:
                return False
        # Then require at least one positive match
        for a in accept_vocab:
            if a in blob:
                return True
        return False

    def _resolve_domain_config(self, domain: str) -> dict:
        """v3.0.108: build a harvest config for a NON-weed domain from its
        domain doc (db COLL_DOMAINS): accept-vocab is derived from the domain's
        taxonomy + harvest_query words; reject-vocab is GENERIC_REJECT only (so
        the domain's own topic is never blocked); queries are its harvest_queries.
        Returns {accept, reject, queries}."""
        import re as _re
        _STOP = {"detection", "dataset", "datasets", "image", "images", "the",
                 "and", "for", "with", "from", "object", "objects", "data",
                 "yolo", "bbox", "annotated", "annotation", "set", "sets"}
        try:
            from . import db as _db
            d = _db.get_domain(domain) or {}
        except Exception:
            d = {}
        # v3.0.110: the cluster compute node can't reach the lab Mongo, so the
        # dashboard stages the domain config to the shared FS before sbatch.
        # Read it when Mongo gave us no queries.
        if not d.get("harvest_queries"):
            try:
                import json as _json
                _p = os.path.join(os.path.dirname(__file__), "..", "..",
                                  "results", "framework", "_domains",
                                  f"{domain}.json")
                if os.path.isfile(_p):
                    with open(_p) as _f:
                        _fd = _json.load(_f)
                    for _k, _v in (_fd or {}).items():
                        if not d.get(_k):
                            d[_k] = _v
            except Exception:
                pass
        queries = list(d.get("harvest_queries") or [])
        # v3.0.180 (P1): an EXPLICIT accept_vocab in the domain config wins over
        # the derived one — lets a project curate exactly which topic words pass,
        # instead of only inferring from taxonomy/queries.
        explicit_accept = [str(w).lower().strip() for w in (d.get("accept_vocab") or []) if str(w).strip()]
        if explicit_accept:
            accept = tuple(sorted(set(explicit_accept)))
        else:
            words = set()
            for s in list(d.get("taxonomy") or []) + queries + [
                    str(d.get("display_name") or ""), domain]:
                for w in _re.findall(r"[a-z0-9]{3,}", str(s).lower()):
                    if w not in _STOP:
                        words.add(w)
            accept = tuple(sorted(words)) or (domain,)
        return {"accept": accept, "reject": self.GENERIC_REJECT,
                "queries": queries}

    def _is_already_flagged_garbage(self, candidate_slug: str) -> bool:
        """Check if user (or prior Brain run) has marked this slug as garbage.
        Stops Brain from re-harvesting the same junk.

        v3.0.92: checks BOTH sources (they were divergent — a real leak):
          1. results/framework/dataset_flags.json  (flag == 'garbage')
          2. results/framework/slug_verdicts.jsonl  (latest verdict == 'junk')
             ← this is where the dashboard /slugs ✗ button writes. Previously the
             harvest guard ignored it, so user-junked off-topic slugs (bee,
             coconut-disease) could be re-harvested. Now they stick."""
        import os, json
        base = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "framework"))
        # 1) dataset_flags.json
        flags_path = os.path.join(base, "dataset_flags.json")
        if os.path.isfile(flags_path):
            try:
                with open(flags_path) as f:
                    flags = json.load(f)
                entry = flags.get(candidate_slug)
                if bool(entry) and entry.get("flag") == "garbage":
                    return True
            except Exception:
                pass
        # 2) slug_verdicts.jsonl — replay to latest verdict per slug
        vpath = os.path.join(base, "slug_verdicts.jsonl")
        if os.path.isfile(vpath):
            try:
                latest = None
                with open(vpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ev = json.loads(line)
                        if ev.get("slug") == candidate_slug:
                            latest = ev.get("verdict")
                if latest == "junk":
                    return True
            except Exception:
                pass
        return False

    def harvest_new_datasets(self, max_new=5, queries=None, confirm_schema=True,
                              max_images_per_ds=30000, strict_topic=None,
                              domain=None):
        """Search HF for NEW datasets, fast-filter by card metadata, download up to max_new.

        Strategy:
        - Iterate queries (weed + crop + plant + agriculture)
        - For each search result: skip if already in registry; skip if card says no bbox hints
        - For passing candidates, optionally confirm schema by loading first item
        - Download up to max_new — each one registered permanently

        v3.0.68 (2026-05-31): user feedback "ibm_research__cif_dataset got 5000
        imgs with 0 labels — useless garbage". Adds strict_topic mode that:
          - Post-download: reject if stats['labeled'] < 100 OR stats['classes']
            in (0, '?') — deletes from registry + removes downloaded files
          - Same for GitHub + Kaggle phases
        Defaults: env BRAIN_STRICT=1 enables it; max_new from BRAIN_MAX_NEW.

        Returns: {attempted: n, downloaded: n, rejected_strict: n,
                  results: [{hf_id, local_images, labeled, kind}]}
        """
        # v3.0.68: read overrides from env so the SBATCH/dashboard can configure
        # without a code change.
        if strict_topic is None:
            strict_topic = bool(int(os.environ.get("BRAIN_STRICT", "0")))
        env_max_new = os.environ.get("BRAIN_MAX_NEW")
        if env_max_new:
            try:
                max_new = int(env_max_new)
            except ValueError:
                pass
        logger.info(
            f"[Harvest] config: max_new={max_new} strict_topic={strict_topic} "
            f"max_images_per_ds={max_images_per_ds}"
        )
        n_rejected_strict = 0

        def _strict_reject_if_garbage(slug, stats, src_tag):
            """Strict-mode post-download check: drop low-label/no-class downloads.
            Returns True if rejected (caller should NOT register/append)."""
            if not strict_topic:
                return False
            labeled = stats.get("labeled", 0) or 0
            classes = stats.get("classes", 0)
            cls_n = 0 if classes in (0, "?", None, "0") else (
                int(classes) if str(classes).isdigit() else 1)
            # v3.0.77 (2026-06-01): loosened — brain was rejecting fine
            # datasets just because their card metadata says labeled=0
            # but disk actually has labels. Now: only reject if NEITHER
            # labeled>=50 NOR class info is present.
            # Honor env BRAIN_STRICT_MIN_LABELS override (default 50).
            strict_min = int(os.environ.get("BRAIN_STRICT_MIN_LABELS", "50") or 50)
            if labeled < strict_min and cls_n == 0:
                logger.info(
                    f"[Harvest][strict] REJECT {src_tag} {slug}: "
                    f"labeled={labeled} classes={classes} "
                    f"(< {strict_min} labels AND 0 classes)"
                )
                # Remove from registry + delete the downloaded files
                self.registry["datasets"].pop(slug, None)
                local_path = os.path.join(self.data_dir, slug)
                if os.path.isdir(local_path):
                    import shutil
                    try:
                        shutil.rmtree(local_path)
                        logger.info(
                            f"[Harvest][strict] cleaned disk: {local_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Harvest][strict] could not rmtree {local_path}: {e}"
                        )
                return True
            return False

        try:
            from huggingface_hub import HfApi
        except ImportError:
            return {"status": "error", "error": "huggingface_hub not installed"}

        api = HfApi()
        # v3.0.108: domain-aware harvest. Default = "weed" → byte-identical to
        # prior behavior (weed queries + weed accept/reject vocab). A non-weed
        # domain pulls queries + accept-vocab from its domain doc and uses only
        # GENERIC_REJECT, then every harvested slug is tagged with that domain.
        domain = domain or os.environ.get("BRAIN_DOMAIN") or "weed"
        if domain == "weed":
            self._harvest_domain = "weed"
            self._accept_vocab = self.AG_VOCAB_ACCEPT
            self._reject_vocab = self.AG_VOCAB_REJECT
        else:
            cfg = self._resolve_domain_config(domain)
            self._harvest_domain = domain
            self._accept_vocab = cfg["accept"]
            self._reject_vocab = cfg["reject"]
            if not queries:
                queries = cfg["queries"]
            logger.info(f"[Harvest] domain={domain} "
                        f"queries={len(queries or [])} "
                        f"accept_kw={len(cfg['accept'])}")
        queries = queries or self.DEFAULT_HARVEST_QUERIES
        results = []
        seen_ids = set()

        # ------------ Phase 1: task-filtered bulk discovery (high precision) ------------
        logger.info("[Harvest] Phase 1: task=object-detection bulk list")
        try:
            det = list(api.list_datasets(
                filter="task_categories:object-detection",
                sort="downloads", limit=200,
            ))
        except Exception as e:
            logger.warning(f"[Harvest] task-filter list failed: {e}")
            det = []

        # v3.0.30.3: tightened relevance — only weed/crop/field-plants, no
        # pest/insect/disease/tree (user spec 2026-05-12).
        prioritized = []
        for d in det:
            name_l = d.id.lower()
            if self._is_relevant_dataset(name_l,
                getattr(d, "description", "") or ""):
                prioritized.append(d)
        logger.info(f"[Harvest] Phase 1: {len(prioritized)}/{len(det)} "
                    f"bulk results passed weed/crop/field-plant filter")

        # ------------ Phase 2: keyword search fallback ------------
        keyword_results = []
        for q in queries:
            try:
                found = list(api.list_datasets(search=q, sort="downloads",
                                                limit=10))
                keyword_results.extend(found)
            except Exception as e:
                logger.warning(f"[Harvest] search failed '{q}': {e}")
                continue

        # v3.0.30.3: same relevance filter applied to keyword hits.
        keyword_filtered = []
        for d in keyword_results:
            if self._is_relevant_dataset(d.id.lower(),
                                          getattr(d, "description", "") or ""):
                keyword_filtered.append(d)
        logger.info(f"[Harvest] Phase 2: {len(keyword_filtered)}/{len(keyword_results)} "
                    f"keyword hits passed weed/crop/field-plant filter")

        # Combine: task-filtered priorities first, then keyword results
        combined = prioritized + keyword_filtered
        logger.info(f"[Harvest] processing {len(combined)} total candidates")

        n_rejected_garbage = 0
        n_rejected_irrelevant = 0
        n_rejected_blacklisted = 0
        # v3.0.43.18: load HF schema blacklist once per harvest
        hf_blacklist = _load_hf_blacklist()
        for d in combined:
            if len(results) >= max_new:
                break
            if d.id in seen_ids:
                continue
            seen_ids.add(d.id)
            if self.is_duplicate(d.id):
                continue
            # v3.0.43.18: skip HF ids we previously failed to schema-probe
            if d.id in hf_blacklist:
                n_rejected_blacklisted += 1
                logger.debug(
                    f"[Harvest] skip {d.id} — in HF schema blacklist "
                    f"(reason: {hf_blacklist[d.id].get('reason', '?')[:60]})"
                )
                continue
            # v3.0.30.3: enforce relevance + flag filter at every entry point.
            slug_guess = self._slugify(d.id)
            if self._is_already_flagged_garbage(slug_guess):
                n_rejected_garbage += 1
                logger.info(f"[Harvest] skip {d.id} — previously flagged garbage")
                continue
            if not self._is_relevant_dataset(d.id.lower(),
                                              getattr(d, "description", "") or ""):
                n_rejected_irrelevant += 1
                logger.info(f"[Harvest] skip {d.id} — failed weed/crop/field filter")
                continue

            # Fast metadata check
            try:
                info = api.dataset_info(d.id)
            except Exception as e:
                logger.debug(f"[Harvest] info fail {d.id}: {e}")
                continue
            has_bbox_hint, reason = self._card_suggests_bbox(info)
            if not has_bbox_hint:
                logger.debug(f"[Harvest] skip {d.id}: {reason}")
                continue

            # Optional full schema confirmation (slower; one streaming iter).
            # Try default config + alternative configs if default has no bbox.
            probe_ok = not confirm_schema
            if confirm_schema:
                from datasets import load_dataset
                configs_to_try = [None]
                try:
                    from datasets import get_dataset_config_names
                    extra = get_dataset_config_names(d.id)
                    # Try detection-looking configs first
                    detection_first = sorted(extra,
                        key=lambda c: 0 if any(k in c.lower()
                                                for k in ("detect", "bbox", "yolo", "coco"))
                                      else 1)
                    configs_to_try = [None] + detection_first[:3]
                except Exception:
                    pass
                last_err = ""
                for cfg in configs_to_try:
                    try:
                        probe = load_dataset(d.id, cfg, split="train", streaming=True) \
                                if cfg else load_dataset(d.id, split="train", streaming=True)
                        item = next(iter(probe))
                        if any(k in item for k in ("objects", "bbox", "boxes",
                                                     "bboxes", "annotations")):
                            probe_ok = True
                            if cfg:
                                reason = f"{reason}; config={cfg}"
                            break
                    except Exception as e:
                        last_err = str(e)[:150]
                        logger.debug(f"[Harvest] probe {d.id} cfg={cfg} err: {last_err}")
                        continue
                if not probe_ok:
                    logger.info(f"[Harvest] skip {d.id}: no bbox in any tested config")
                    # v3.0.43.18: blacklist so we don't re-probe this id
                    # on every harvest round
                    _blacklist_hf_id(
                        d.id,
                        reason=f"probe_no_bbox_or_cast_error: {last_err}" if last_err
                                else "no_bbox_in_probe",
                    )
                    continue

            # Good candidate — download
            slug = self._slugify(d.id)
            logger.info(f"[Harvest] Downloading {d.id} (slug={slug}) — {reason}")
            local_path = os.path.join(self.data_dir, slug)
            os.makedirs(local_path, exist_ok=True)

            if slug not in self.registry["datasets"]:
                self.registry["datasets"][slug] = {
                    "source": "huggingface", "hf_id": d.id,
                    "images": 0, "classes": "?",
                    "annotation": "bbox_suspected", "format": "hf",
                    "description": f"Auto-harvested: {reason}",
                    "status": "known", "local_path": None, "local_images": 0,
                    "class_names": [], "downloaded_at": None,
                    "used_for_training": False, "training_runs": [],
                    "harvest_reason": reason,
                    # v3.0.74 (2026-06-01): tag the slug with the current
                    # round so per-round UI + Roboflow batch naming can
                    # group correctly. current_round defaults to 1 if the
                    # registry was created before the round-tracking
                    # feature shipped.
                    "harvest_round": int(
                        self.registry.get("current_round", 1)),
                    "harvest_round_ts": int(time.time()),
                }

            _, stats = self._download_hf(slug, d.id, local_path, max_images_per_ds)
            # v3.0.68: strict-mode post-download quality gate
            if _strict_reject_if_garbage(slug, stats, "HF"):
                n_rejected_strict += 1
                continue
            results.append({
                "hf_id": d.id, "slug": slug,
                "stats": stats, "reason": reason,
            })

        # ------------ Phase 3: GitHub (if HF quota not hit) ------------
        if len(results) < max_new:
            try:
                from .extra_sources import harvest_github_datasets
                gh_quota = max_new - len(results)
                gh = harvest_github_datasets(
                    data_dir=self.data_dir,
                    queries=[q for q in queries if any(k in q for k in ("weed", "crop", "plant", "agri"))] or queries[:3],
                    already_known_cb=lambda s: s in self.registry["datasets"] or self.is_duplicate(s),
                    max_new=min(gh_quota, 3),
                )
                for entry in gh:
                    slug = entry["slug"]
                    self.registry["datasets"][slug] = entry["info"]
                    # v3.0.68: same strict gate for GitHub-sourced slugs
                    if _strict_reject_if_garbage(slug, entry["stats"], "GitHub"):
                        n_rejected_strict += 1
                        continue
                    results.append({
                        "hf_id": entry["hf_id"], "slug": slug,
                        "stats": entry["stats"], "reason": entry["reason"],
                    })
            except Exception as e:
                logger.warning(f"[Harvest] GitHub phase failed: {e}")

        # ------------ Phase 4: Kaggle (if kagglehub + creds present) ------------
        if len(results) < max_new:
            try:
                from .extra_sources import harvest_kaggle_datasets
                kg_quota = max_new - len(results)
                kg = harvest_kaggle_datasets(
                    data_dir=self.data_dir,
                    queries=[q for q in queries if any(k in q for k in ("weed", "crop", "field", "agri"))] or queries[:3],
                    already_known_cb=lambda s: (
                        s in self.registry["datasets"]
                        or self.is_duplicate(s)
                        or self._is_already_flagged_garbage(s)
                    ),
                    max_new=min(kg_quota, 3),
                )
                # v3.0.30.3: post-filter Kaggle results too (extra_sources doesn't
                # know our reject vocab yet).
                for entry in kg:
                    slug = entry["slug"]
                    desc = entry.get("info", {}).get("description", "") or ""
                    if not self._is_relevant_dataset(slug, desc):
                        logger.info(f"[Harvest] kaggle skip {slug} — failed weed/crop filter")
                        continue
                    self.registry["datasets"][slug] = entry["info"]
                    # v3.0.68: strict gate on Kaggle too
                    if _strict_reject_if_garbage(slug, entry["stats"], "Kaggle"):
                        n_rejected_strict += 1
                        continue
                    results.append({
                        "hf_id": entry["hf_id"], "slug": slug,
                        "stats": entry["stats"], "reason": entry["reason"],
                    })
            except Exception as e:
                logger.warning(f"[Harvest] Kaggle phase failed: {e}")

        # ------------ Phase 5: Roboflow Universe (bulk driver) ------------
        # Roboflow is the north-star source — one project typically gives 1K-10K
        # bbox images, so a single harvest round can easily add 20K+ images.
        if len(results) < max_new:
            try:
                from .roboflow_source import harvest_roboflow_datasets
                rf_quota = max_new - len(results)
                rf_queries = [q for q in queries if any(k in q for k in ("weed", "crop", "plant", "agri"))] or queries[:3]
                rf = harvest_roboflow_datasets(
                    data_dir=self.data_dir,
                    queries=rf_queries,
                    already_known_cb=lambda s: (
                        s in self.registry["datasets"]
                        or self.is_duplicate(s)
                        or self._is_already_flagged_garbage(s)
                    ),
                    # Allow Roboflow to dominate the quota since it's the big source
                    max_new=max(rf_quota, 8),
                )
                # v3.0.30.3: post-filter Roboflow results too
                for entry in rf:
                    slug = entry["slug"]
                    desc = entry.get("info", {}).get("description", "") or ""
                    if not self._is_relevant_dataset(slug, desc):
                        logger.info(f"[Harvest] roboflow skip {slug} — failed weed/crop filter")
                        continue
                    self.registry["datasets"][slug] = entry["info"]
                    results.append({
                        "hf_id": entry["hf_id"], "slug": slug,
                        "stats": entry["stats"], "reason": entry["reason"],
                    })
            except Exception as e:
                logger.warning(f"[Harvest] Roboflow phase failed: {e}")

        # v3.0.68: also strict-gate any Roboflow phase entries that may have
        # accumulated in registry. (The Roboflow loop above appends directly
        # to results; we honor strict on those too.)
        if strict_topic and results:
            kept = []
            for r in results:
                slug = r.get("slug")
                if slug and slug in self.registry["datasets"]:
                    if _strict_reject_if_garbage(slug, r.get("stats", {}), "RF"):
                        n_rejected_strict += 1
                        continue
                kept.append(r)
            results = kept

        # v3.0.77.1 (2026-06-02): stamp harvest_round on EVERY slug
        # downloaded in this run. Earlier code only set it in the HF
        # phase; GitHub/Kaggle/RoboflowUniverse paths registered slugs
        # via entry["info"] which didn't include harvest_round. As a
        # result the 2 new slugs (gh_minhvuongvu + kg_ravirajsinh45)
        # from run 41106922 came back tagged round=0 instead of
        # current_round=2. Fix: post-hoc stamp every slug in `results`.
        cur_round = int(self.registry.get("current_round", 1))
        for r in results:
            slug = r.get("slug")
            if not slug or slug not in self.registry["datasets"]:
                continue
            info = self.registry["datasets"][slug]
            if not info.get("harvest_round"):
                info["harvest_round"] = cur_round
                info["harvest_round_ts"] = int(time.time())

        self._save_registry()
        return {
            "status": "ok",
            "strict_topic": strict_topic,
            "queries_tried": len([q for q in queries]),
            "candidates_passed_filter": len(results),
            "downloaded": len([r for r in results
                               if r["stats"].get("status") == "downloaded"]),
            "rejected_strict_garbage": n_rejected_strict,
            "results": results,
        }

    # =========================================================
    # LIST & SUMMARY
    # =========================================================

    def list_all(self):
        """List all datasets with full status."""
        result = []
        for name, info in self.registry["datasets"].items():
            result.append({
                "name": name,
                "images": info.get("images", info.get("local_images", 0)),
                "classes": info.get("classes", "?"),
                "status": info.get("status", "unknown"),
                "used": info.get("used_for_training", False),
                "training_runs": len(info.get("training_runs", [])),
                "annotation": info.get("annotation", "?"),
            })
        return result

    def get_total_images(self):
        """Total downloaded images."""
        return sum(d.get("local_images", 0) for d in self.registry["datasets"].values())

    def get_summary_for_brain(self):
        """Summary for Brain context."""
        lines = ["Dataset Registry:"]
        total_dl = 0
        total_used = 0
        for name, info in self.registry["datasets"].items():
            status = info.get("status", "?")
            used = "TRAINED" if info.get("used_for_training") else ""
            imgs = info.get("local_images", 0) or info.get("images", 0)
            total_dl += info.get("local_images", 0)
            if info.get("used_for_training"):
                total_used += info.get("local_images", 0)
            lines.append(f"  {name}: {imgs} imgs [{status}] {used}")

        n_discovered = len(self.registry.get("discovered", []))
        lines.append(f"\nTotal downloaded: {total_dl} images")
        lines.append(f"Total used for training: {total_used} images")
        lines.append(f"Discovered (not yet added): {n_discovered}")
        lines.append("Tools: search_datasets (find new), download_dataset (get data)")
        return "\n".join(lines)
