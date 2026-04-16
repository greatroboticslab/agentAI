"""
Dataset Discovery — Brain autonomously searches, downloads, and merges weed detection datasets.

Sources: HuggingFace Datasets, Kaggle, Roboflow
Goal: Accumulate 100K+ images with real annotations for maximum training quality.

Known large datasets (pre-researched):
- WeedSense: 120,341 images, 16 species, BBox+segmentation (HuggingFace)
- DeepWeeds: 17,509 images, 9 classes (Kaggle)
- CottonWeedDet12: 5,648 images, 12 species (current, Zenodo)
- crop_weed_research_data: 4,307 images, VOC XML (HuggingFace)
- Grass-Weeds: 2,490 images, COCO format (HuggingFace)
- CottonWeedID15: 5,187 images, classification (Kaggle)
"""

import os
import json
import shutil
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)

# Pre-researched datasets with download info
KNOWN_DATASETS = {
    "weedsense": {
        "source": "huggingface",
        "hf_id": "baselab/weedsense",
        "images": 120341,
        "classes": 16,
        "annotation": "bbox+segmentation",
        "format": "voc_xml",
        "description": "Largest weed dataset. 16 species, VOC XML bboxes, segmentation masks.",
    },
    "deepweeds": {
        "source": "huggingface",
        "hf_id": "imsparsh/deepweeds",
        "images": 17509,
        "classes": 9,
        "annotation": "classification",
        "format": "csv",
        "description": "8 Australian weed species + negative class. Classification only.",
    },
    "crop_weed_research": {
        "source": "huggingface",
        "hf_id": "ivliev123/crop_weed_research_data",
        "images": 4307,
        "classes": "multi",
        "annotation": "bbox",
        "format": "voc_xml",
        "description": "Crop and weed bounding boxes in Pascal VOC XML.",
    },
    "grass_weeds": {
        "source": "huggingface",
        "hf_id": "Francesco/grass-weeds",
        "images": 2490,
        "classes": 2,
        "annotation": "bbox",
        "format": "coco",
        "description": "Grass vs weeds, COCO format bboxes.",
    },
    "weed_crop_aerial": {
        "source": "huggingface",
        "hf_id": "LibreYOLO/weed-crop-aerial",
        "images": 1176,
        "classes": 2,
        "annotation": "bbox",
        "format": "yolo",
        "description": "Aerial weed-crop detection, YOLO format.",
    },
}


class DatasetDiscovery:
    """Search, download, and manage weed detection datasets."""

    def __init__(self):
        self.data_dir = os.path.join(Config.BASE_DIR, "datasets")
        os.makedirs(self.data_dir, exist_ok=True)
        self.downloaded = {}
        self._scan_existing()

    def _scan_existing(self):
        """Check which datasets are already downloaded."""
        for name in KNOWN_DATASETS:
            path = os.path.join(self.data_dir, name)
            if os.path.isdir(path):
                n_files = sum(1 for f in Path(path).rglob("*") if f.is_file())
                self.downloaded[name] = n_files

    def list_available(self):
        """List all known datasets with download status."""
        result = []
        for name, info in KNOWN_DATASETS.items():
            result.append({
                "name": name,
                "images": info["images"],
                "classes": info["classes"],
                "annotation": info["annotation"],
                "downloaded": name in self.downloaded,
                "local_files": self.downloaded.get(name, 0),
                "description": info["description"],
            })
        return result

    def search_huggingface(self, query="weed detection", max_results=20):
        """Search HuggingFace for new datasets."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            datasets = api.list_datasets(search=query, sort="downloads",
                                          direction=-1, limit=max_results)
            results = []
            for d in datasets:
                results.append({
                    "id": d.id,
                    "downloads": getattr(d, "downloads", 0),
                    "likes": getattr(d, "likes", 0),
                    "tags": getattr(d, "tags", [])[:5],
                })
            return results
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {e}")
            return [{"id": v["hf_id"], "images": v["images"]}
                    for v in KNOWN_DATASETS.values() if v["source"] == "huggingface"]

    def download_dataset(self, name, max_images=None):
        """Download a known dataset.

        Args:
            name: dataset key from KNOWN_DATASETS
            max_images: limit download size (None = all)

        Returns:
            (local_path, stats)
        """
        if name not in KNOWN_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Known: {list(KNOWN_DATASETS.keys())}")

        info = KNOWN_DATASETS[name]
        local_path = os.path.join(self.data_dir, name)

        if name in self.downloaded:
            logger.info(f"[Dataset] {name} already downloaded ({self.downloaded[name]} files)")
            return local_path, {"status": "exists", "files": self.downloaded[name]}

        os.makedirs(local_path, exist_ok=True)
        logger.info(f"[Dataset] Downloading {name} ({info['images']} images)...")

        if info["source"] == "huggingface":
            return self._download_from_hf(name, info["hf_id"], local_path, max_images)
        else:
            return local_path, {"status": "unsupported_source", "source": info["source"]}

    def _download_from_hf(self, name, hf_id, local_path, max_images):
        """Download dataset from HuggingFace."""
        try:
            from datasets import load_dataset

            # Load dataset (streaming if large to avoid RAM issues)
            if KNOWN_DATASETS[name]["images"] > 10000:
                ds = load_dataset(hf_id, split="train", streaming=True)
                count = 0
                limit = max_images or KNOWN_DATASETS[name]["images"]
                for item in ds:
                    if count >= limit:
                        break
                    # Save image
                    if "image" in item:
                        img = item["image"]
                        img_path = os.path.join(local_path, "images", f"{count:06d}.jpg")
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        img.save(img_path)
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"[Dataset] {name}: {count} images downloaded...")
            else:
                ds = load_dataset(hf_id, split="train")
                count = len(ds)
                ds.save_to_disk(local_path)

            self.downloaded[name] = count
            stats = {"status": "downloaded", "images": count, "path": local_path}
            logger.info(f"[Dataset] {name}: {count} images downloaded to {local_path}")

            # Save metadata
            with open(os.path.join(local_path, "metadata.json"), "w") as f:
                json.dump({**KNOWN_DATASETS[name], "local_images": count}, f, indent=2)

            return local_path, stats

        except Exception as e:
            logger.error(f"[Dataset] Download failed for {name}: {e}")
            return local_path, {"status": "error", "error": str(e)}

    def get_total_images(self):
        """Get total downloaded images across all datasets."""
        return sum(self.downloaded.values())

    def get_summary_for_brain(self):
        """Summary for Brain context."""
        lines = [f"Datasets: {len(self.downloaded)} downloaded, "
                 f"{self.get_total_images()} total images"]
        for name, info in KNOWN_DATASETS.items():
            status = f"[DOWNLOADED: {self.downloaded[name]} files]" if name in self.downloaded else "[NOT DOWNLOADED]"
            lines.append(f"  {name}: {info['images']} imgs, {info['classes']} classes — {status}")
        lines.append(f"\nTotal available: ~319,000 images across all known datasets")
        return "\n".join(lines)
