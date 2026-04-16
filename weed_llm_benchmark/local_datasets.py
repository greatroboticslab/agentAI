#!/usr/bin/env python3
"""
Dataset registry and management for weed detection benchmark.

Handles download, directory layout, and metadata for all benchmark datasets.

Supported datasets:
    - CottonWeedDet12: 12-class weed detection (~5,648 images)
    - DeepWeeds: 8 weed species + background (~17,509 images)
    - weed2okok: Lab's own 1-class weed data (106 images)

Usage:
    python datasets.py --list
    python datasets.py --download cottonweeddet12
    python datasets.py --download all
    python datasets.py --info weed2okok
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # parent of weed_llm_benchmark/
# Check both possible download locations (local vs cluster layout)
_dl_project = os.path.join(PROJECT_ROOT, "downloads")
_dl_base = os.path.join(BASE_DIR, "downloads")
DOWNLOAD_DIR = _dl_base if os.path.isdir(_dl_base) else _dl_project

# ============================================================
# Dataset Registry
# ============================================================
DATASET_REGISTRY = {
    "cottonweeddet12": {
        "name": "CottonWeedDet12",
        "description": "12 cotton weed species with bbox annotations",
        "num_images": 5648,
        "num_classes": 12,
        "classes": [
            "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
            "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
            "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
        ],
        "source": "zenodo",
        "zenodo_url": "https://zenodo.org/api/records/7535814/files/CottonWeedDet12.7z/content",
        "zenodo_filename": "CottonWeedDet12.7z",
        "zenodo_md5": "f02e4c32d27b042a4f599a3305c99414",
        "original_format": "labelme_json+voc_xml",
        "format": "yolov8",
        "paper": "Dang et al., Computers and Electronics in Agriculture, 2023",
        "doi": "10.5281/zenodo.7535814",
        "split": {"train": 0.65, "valid": 0.20, "test": 0.15},
        "why": "Multi-class, published, bbox annotations, tests species-level recognition",
    },
    "deepweeds": {
        "name": "DeepWeeds",
        "description": "8 weed species in rangeland environments",
        "num_images": 17509,
        "num_classes": 9,  # 8 species + negative
        "classes": [
            "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
            "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed",
            "Negative",
        ],
        "source": "roboflow",
        "roboflow_project": "deepweeds-yjbbr",
        "roboflow_version": 1,
        "format": "yolov8",
        "paper": "Olsen et al., Scientific Reports, 2019",
        "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
        "why": "Large sample, different context (rangeland), peer-reviewed",
    },
    "weed2okok": {
        "name": "weed2okok",
        "description": "Lab's own 1-class weed detection data",
        "num_images": 106,
        "num_classes": 1,
        "classes": ["weed"],
        "source": "roboflow",
        "roboflow_project": "weed2okok",
        "roboflow_version": 1,
        "format": "yolov8",
        "paper": None,
        "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
        "why": "Lab's own data, shows practical deployment",
    },
    "cropweed": {
        "name": "Crop and Weed Detection",
        "description": "Fallback 2-class crop/weed detection dataset",
        "num_images": 1300,
        "num_classes": 2,
        "classes": ["crop", "weed"],
        "source": "roboflow",
        "roboflow_project": "crop-and-weed-detection-ksnrg",
        "roboflow_version": 1,
        "format": "yolov8",
        "paper": None,
        "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
        "why": "Fallback dataset, 2-class, ~1,300 images from Roboflow",
    },
}


def get_dataset_path(dataset_key):
    """Return the local path for a dataset."""
    return os.path.join(DOWNLOAD_DIR, dataset_key)


def is_downloaded(dataset_key):
    """Check if a dataset is already downloaded."""
    path = get_dataset_path(dataset_key)
    if not os.path.isdir(path):
        return False
    # Check for at least one split with images
    for split in ["train", "valid", "test", ""]:
        img_dir = os.path.join(path, split, "images") if split else os.path.join(path, "images")
        if os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0:
            return True
    return False


def get_split_info(dataset_key):
    """Get image counts per split for a downloaded dataset."""
    path = get_dataset_path(dataset_key)
    info = {}
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(path, split, "images")
        label_dir = os.path.join(path, split, "labels")
        if os.path.isdir(img_dir):
            imgs = [f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            labels = []
            if os.path.isdir(label_dir):
                labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            info[split] = {"images": len(imgs), "labels": len(labels)}
    return info


def get_class_names(dataset_key):
    """Get class names from dataset registry or data.yaml."""
    # Try data.yaml first
    data_yaml_path = os.path.join(get_dataset_path(dataset_key), "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            import yaml
            with open(data_yaml_path) as f:
                data = yaml.safe_load(f)
            if "names" in data:
                return data["names"]
        except ImportError:
            pass

    # Fall back to registry
    if dataset_key in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_key]["classes"]
    return ["weed"]


def download_dataset(dataset_key, force=False):
    """Download a dataset from Roboflow."""
    if dataset_key not in DATASET_REGISTRY:
        print(f"[!] Unknown dataset: {dataset_key}")
        print(f"    Available: {list(DATASET_REGISTRY.keys())}")
        return None

    ds = DATASET_REGISTRY[dataset_key]
    path = get_dataset_path(dataset_key)

    if is_downloaded(dataset_key) and not force:
        print(f"[*] {ds['name']} already downloaded at {path}")
        return path

    if ds["source"] == "roboflow":
        try:
            from roboflow_bridge import download_from_roboflow
            return download_from_roboflow(
                ds["roboflow_project"],
                str(ds["roboflow_version"]),
                ds["format"],
                download_path=path,
            )
        except ImportError:
            print("[!] roboflow_bridge not available. Install roboflow package.")
            return None
    elif ds["source"] == "zenodo":
        return download_from_zenodo(dataset_key, ds, path)
    else:
        print(f"[!] Unsupported source: {ds['source']}")
        return None


def download_from_zenodo(dataset_key, ds_info, output_path):
    """Download and extract a dataset from Zenodo, then convert to YOLO format."""
    import subprocess

    url = ds_info["zenodo_url"]
    filename = ds_info["zenodo_filename"]
    archive_path = os.path.join(output_path, filename)

    os.makedirs(output_path, exist_ok=True)

    # Step 1: Download if not already present
    if not os.path.exists(archive_path):
        print(f"[*] Downloading {filename} from Zenodo...")
        print(f"    URL: {url}")
        print(f"    This may take a while for large datasets.")
        result = subprocess.run(
            ["curl", "-L", "-o", archive_path, "--progress-bar", url],
            cwd=output_path,
        )
        if result.returncode != 0:
            print(f"[!] Download failed with return code {result.returncode}")
            return None
    else:
        print(f"[*] Archive already exists: {archive_path}")

    # Step 2: Extract
    extracted_dir = os.path.join(output_path, Path(filename).stem)
    if not os.path.isdir(extracted_dir):
        print(f"[*] Extracting {filename}...")
        if filename.endswith(".7z"):
            result = subprocess.run(["7z", "x", archive_path, f"-o{output_path}", "-y"],
                                   capture_output=True, text=True)
        elif filename.endswith(".zip"):
            result = subprocess.run(["unzip", "-o", archive_path, "-d", output_path],
                                   capture_output=True, text=True)
        else:
            print(f"[!] Unknown archive format: {filename}")
            return None

        if result.returncode != 0:
            print(f"[!] Extraction failed: {result.stderr[:500]}")
            return None
        print(f"[+] Extracted to {extracted_dir}")
    else:
        print(f"[*] Already extracted: {extracted_dir}")

    # Step 3: Convert to YOLO format if needed
    data_yaml = os.path.join(output_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"[*] Converting to YOLO format...")
        try:
            from convert_coco_to_yolo import convert_dataset
            convert_dataset(extracted_dir, output_path,
                           split_ratios=tuple(ds_info.get("split", {}).values()) or (0.65, 0.20, 0.15))
        except ImportError:
            print("[!] convert_coco_to_yolo module not found.")
            return None
    else:
        print(f"[*] data.yaml exists, skipping conversion")

    return output_path


def list_datasets():
    """Print all registered datasets."""
    print(f"\n{'='*70}")
    print("REGISTERED DATASETS")
    print(f"{'='*70}")
    print(f"  {'Key':<20} {'Name':<25} {'Images':>8} {'Classes':>8} {'Downloaded':>12}")
    print(f"  {'-'*75}")
    for key, ds in DATASET_REGISTRY.items():
        downloaded = "Yes" if is_downloaded(key) else "No"
        print(f"  {key:<20} {ds['name']:<25} {ds['num_images']:>8} {ds['num_classes']:>8} {downloaded:>12}")
    print()


def dataset_info(dataset_key):
    """Print detailed info about a dataset."""
    if dataset_key not in DATASET_REGISTRY:
        print(f"[!] Unknown dataset: {dataset_key}")
        return

    ds = DATASET_REGISTRY[dataset_key]
    print(f"\n{'='*50}")
    print(f"DATASET: {ds['name']}")
    print(f"{'='*50}")
    print(f"  Description: {ds['description']}")
    print(f"  Images:      {ds['num_images']}")
    print(f"  Classes:     {ds['num_classes']}")
    print(f"  Class names: {', '.join(ds['classes'])}")
    print(f"  Source:      {ds['source']}")
    if ds.get("paper"):
        print(f"  Paper:       {ds['paper']}")
    print(f"  Why:         {ds['why']}")

    if is_downloaded(dataset_key):
        print(f"\n  Local path:  {get_dataset_path(dataset_key)}")
        split_info = get_split_info(dataset_key)
        for split, counts in split_info.items():
            print(f"    {split}: {counts['images']} images, {counts['labels']} labels")
    else:
        print(f"\n  Not downloaded. Run: python datasets.py --download {dataset_key}")


def main():
    parser = argparse.ArgumentParser(description="Dataset management for weed detection benchmark")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--download", type=str, help="Download a dataset (key or 'all')")
    parser.add_argument("--info", type=str, help="Show dataset details")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.download:
        if args.download == "all":
            for key in DATASET_REGISTRY:
                download_dataset(key, force=args.force)
        else:
            download_dataset(args.download, force=args.force)
    elif args.info:
        dataset_info(args.info)
    else:
        list_datasets()


if __name__ == "__main__":
    main()
