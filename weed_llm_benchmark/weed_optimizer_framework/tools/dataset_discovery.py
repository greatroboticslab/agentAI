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


class DatasetDiscovery:
    """Search, download, track, and deduplicate weed detection datasets."""

    def __init__(self):
        self.data_dir = os.path.join(Config.BASE_DIR, "datasets")
        os.makedirs(self.data_dir, exist_ok=True)
        self.registry = self._load_registry()

    # =========================================================
    # REGISTRY — persistent tracking of all datasets
    # =========================================================

    def _load_registry(self):
        """Load dataset registry (tracks downloaded, used, discovered)."""
        if os.path.exists(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH) as f:
                    registry = json.load(f)
                self._discover_preexisting(registry)
                self._save_registry(registry)
                return registry
            except (json.JSONDecodeError, KeyError):
                pass

        registry = {"datasets": {}, "discovered": [], "total_downloaded": 0}
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
        """Save registry with atomic write."""
        if registry is None:
            registry = self.registry
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        tmp = REGISTRY_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2, default=str)
        os.replace(tmp, REGISTRY_PATH)

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
            datasets = api.list_datasets(search=query, sort="downloads",
                                          direction=-1, limit=max_results)
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

    def download_dataset(self, name, max_images=None):
        """Download a dataset. Checks for duplicates first."""
        # Dedup check
        if self.is_downloaded(name):
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
        """Download from HuggingFace with schema-aware YOLO label extraction."""
        try:
            from datasets import load_dataset
            from datasets import get_dataset_config_names

            expected = KNOWN_DATASETS.get(name, {}).get("images", 0)
            limit = max_images or expected or 999999

            # Probe schema (small sample, non-streaming)
            logger.info(f"[Dataset] Probing {hf_id} schema...")
            try:
                probe_ds = load_dataset(hf_id, split="train", streaming=True)
                probe_item = next(iter(probe_ds))
                schema_keys = list(probe_item.keys())
                logger.info(f"[Dataset] {name} schema keys: {schema_keys}")
            except Exception as e:
                logger.warning(f"[Dataset] Probe failed ({e}); proceeding anyway")
                probe_item = None

            # Decide annotation type: bbox-capable or classification-only
            has_bbox = probe_item is not None and (
                "objects" in probe_item or "bbox" in probe_item
                or "boxes" in probe_item or "bboxes" in probe_item
                or "annotations" in probe_item
            )
            annotation_kind = "bbox" if has_bbox else "classification"

            # Use streaming for large datasets to avoid RAM blowup
            use_streaming = expected > 10000 or limit > 10000
            ds = load_dataset(hf_id, split="train", streaming=use_streaming)

            img_dir = os.path.join(local_path, "images")
            lbl_dir = os.path.join(local_path, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            count = 0
            label_count = 0
            all_classes = set()

            iterator = ds if use_streaming else iter(ds)
            for item in iterator:
                if count >= limit:
                    break
                if "image" not in item:
                    count += 1
                    continue
                img = item["image"]
                w = getattr(img, "width", item.get("width", 0))
                h = getattr(img, "height", item.get("height", 0))
                stem = f"{count:06d}"
                img.save(os.path.join(img_dir, f"{stem}.jpg"))

                if has_bbox and w and h:
                    lines, classes = self._extract_yolo_labels(item, w, h)
                    if lines:
                        with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
                            f.write("\n".join(lines))
                        label_count += 1
                        all_classes.update(classes)
                    else:
                        # Empty label file still counts as "image with no objects"
                        open(os.path.join(lbl_dir, f"{stem}.txt"), "w").close()

                count += 1
                if count % 2000 == 0:
                    logger.info(f"[Dataset] {name}: {count}/{limit} imgs, "
                                f"{label_count} labeled, classes={len(all_classes)}")

            # Register result
            self.registry["datasets"].setdefault(name, {**KNOWN_DATASETS.get(name, {})})
            self.registry["datasets"][name].update({
                "status": "downloaded",
                "local_path": local_path,
                "local_images": count,
                "local_labeled": label_count,
                "class_ids_seen": sorted(all_classes),
                "annotation": annotation_kind,
                "downloaded_at": datetime.now().isoformat(),
            })
            self.registry["total_downloaded"] = sum(
                d.get("local_images", 0) for d in self.registry["datasets"].values()
            )
            self._save_registry()

            logger.info(f"[Dataset] '{name}': {count} imgs, {label_count} with bbox labels, "
                        f"{len(all_classes)} classes, kind={annotation_kind}")
            return local_path, {
                "status": "downloaded",
                "images": count,
                "labeled": label_count,
                "classes": len(all_classes),
                "annotation_kind": annotation_kind,
            }

        except Exception as e:
            logger.error(f"[Dataset] Download failed: {e}")
            return local_path, {"status": "error", "error": str(e)}

    # =========================================================
    # HARVEST — each run discovers N new datasets, accumulates forever
    # =========================================================

    DEFAULT_HARVEST_QUERIES = [
        "weed detection", "weed bounding box", "weed yolo",
        "crop detection", "crop disease detection", "plant detection",
        "agriculture object detection", "agricultural bbox",
        "pest detection", "plant disease bbox",
    ]

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

    def harvest_new_datasets(self, max_new=5, queries=None, confirm_schema=True,
                              max_images_per_ds=30000):
        """Search HF for NEW datasets, fast-filter by card metadata, download up to max_new.

        Strategy:
        - Iterate queries (weed + crop + plant + agriculture)
        - For each search result: skip if already in registry; skip if card says no bbox hints
        - For passing candidates, optionally confirm schema by loading first item
        - Download up to max_new — each one registered permanently

        Returns: {attempted: n, downloaded: n, results: [{hf_id, local_images, labeled, kind}]}
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return {"status": "error", "error": "huggingface_hub not installed"}

        api = HfApi()
        queries = queries or self.DEFAULT_HARVEST_QUERIES
        results = []
        seen_ids = set()

        for q in queries:
            if len(results) >= max_new:
                break
            logger.info(f"[Harvest] Query: '{q}'")
            try:
                found = list(api.list_datasets(search=q, sort="downloads",
                                                direction=-1, limit=15))
            except Exception as e:
                logger.warning(f"[Harvest] search failed '{q}': {e}")
                continue

            for d in found:
                if len(results) >= max_new:
                    break
                if d.id in seen_ids:
                    continue
                seen_ids.add(d.id)
                if self.is_duplicate(d.id):
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

                # Optional full schema confirmation (slower; one streaming iter)
                if confirm_schema:
                    try:
                        from datasets import load_dataset
                        probe = load_dataset(d.id, split="train", streaming=True)
                        item = next(iter(probe))
                        has_bbox_real = any(k in item for k in
                                             ("objects", "bbox", "boxes", "bboxes",
                                              "annotations"))
                        if not has_bbox_real:
                            logger.info(f"[Harvest] skip {d.id}: schema keys "
                                        f"{list(item.keys())} no bbox")
                            continue
                    except Exception as e:
                        logger.info(f"[Harvest] probe fail {d.id}: {str(e)[:150]}")
                        continue

                # Good candidate — download
                slug = self._slugify(d.id)
                logger.info(f"[Harvest] Downloading {d.id} (slug={slug}) — "
                            f"hint reason: {reason}")
                local_path = os.path.join(self.data_dir, slug)
                os.makedirs(local_path, exist_ok=True)

                # Register as known first, then download
                if slug not in self.registry["datasets"]:
                    self.registry["datasets"][slug] = {
                        "source": "huggingface", "hf_id": d.id,
                        "images": 0, "classes": "?",
                        "annotation": "bbox_suspected", "format": "hf",
                        "description": f"Auto-harvested via query: {q}",
                        "status": "known", "local_path": None, "local_images": 0,
                        "class_names": [], "downloaded_at": None,
                        "used_for_training": False, "training_runs": [],
                        "harvest_query": q, "harvest_reason": reason,
                    }

                _, stats = self._download_hf(slug, d.id, local_path, max_images_per_ds)
                results.append({
                    "hf_id": d.id, "slug": slug, "query": q,
                    "stats": stats, "reason": reason,
                })

        self._save_registry()
        return {
            "status": "ok",
            "queries_tried": len([q for q in queries]),
            "candidates_passed_filter": len(results),
            "downloaded": len([r for r in results
                               if r["stats"].get("status") == "downloaded"]),
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
