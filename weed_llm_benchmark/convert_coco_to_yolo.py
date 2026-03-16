#!/usr/bin/env python3
"""
Convert CottonWeedDet12 dataset from COCO JSON format to YOLO format.

CottonWeedDet12 structure (from Zenodo):
    CottonWeedDet12/
        image1.jpg
        image1.json   (COCO per-image annotation)
        image1.xml    (VOC annotation)
        ...

Each .json annotation file contains:
{
    "shapes": [
        {"label": "Waterhemp", "points": [[x1,y1],[x2,y2]], ...},
        ...
    ],
    "imageWidth": ...,
    "imageHeight": ...
}

Or standard COCO format with "annotations" list.

Output YOLO format:
    cottonweeddet12/
        data.yaml
        train/images/*.jpg
        train/labels/*.txt
        valid/images/*.jpg
        valid/labels/*.txt
        test/images/*.jpg
        test/labels/*.txt

Usage:
    python convert_coco_to_yolo.py --source downloads/cottonweeddet12/CottonWeedDet12 \
                                   --output downloads/cottonweeddet12 \
                                   --split 0.65 0.20 0.15
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 12 weed classes in CottonWeedDet12
CLASS_NAMES = [
    "Carpetweeds",
    "Crabgrass",
    "Eclipta",
    "Goosegrass",
    "Morningglory",
    "Nutsedge",
    "PalmerAmaranth",
    "PricklySida",
    "Purslane",
    "Ragweed",
    "Sicklepod",
    "SpottedSpurge",
]

# Aliases for class name normalization
CLASS_ALIASES = {
    "carpetweeds": "Carpetweeds",
    "carpetweed": "Carpetweeds",
    "crabgrass": "Crabgrass",
    "eclipta": "Eclipta",
    "goosegrass": "Goosegrass",
    "morningglory": "Morningglory",
    "morning glory": "Morningglory",
    "nutsedge": "Nutsedge",
    "palmeramaranth": "PalmerAmaranth",
    "palmer amaranth": "PalmerAmaranth",
    "palmer_amaranth": "PalmerAmaranth",
    "pricklysida": "PricklySida",
    "prickly sida": "PricklySida",
    "prickly_sida": "PricklySida",
    "purslane": "Purslane",
    "ragweed": "Ragweed",
    "sicklepod": "Sicklepod",
    "spottedspurge": "SpottedSpurge",
    "spotted spurge": "SpottedSpurge",
    "spotted_spurge": "SpottedSpurge",
    "waterhemp": "PalmerAmaranth",  # Often grouped with Palmer Amaranth
    "cutleaf groundcherry": "SpottedSpurge",  # Map to nearest if needed
}


def get_class_id(label):
    """Map a label string to class ID (0-indexed)."""
    normalized = label.strip()
    # Direct match
    if normalized in CLASS_NAMES:
        return CLASS_NAMES.index(normalized)
    # Alias match
    lower = normalized.lower().replace("_", " ")
    if lower in CLASS_ALIASES:
        return CLASS_NAMES.index(CLASS_ALIASES[lower])
    # Fuzzy match
    for i, name in enumerate(CLASS_NAMES):
        if name.lower() in lower or lower in name.lower():
            return i
    print(f"  [!] Unknown class: '{label}' — defaulting to 0")
    return 0


def parse_labelme_json(json_path):
    """Parse LabelMe-style per-image JSON annotation."""
    with open(json_path) as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    boxes = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "weed")
        points = shape.get("points", [])
        if len(points) >= 2:
            # Points are [[x1,y1], [x2,y2]] for rectangle
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            class_id = get_class_id(label)

            # Convert to YOLO format: cx cy w h (normalized)
            if img_w > 0 and img_h > 0:
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                if w > 0 and h > 0:
                    boxes.append((class_id, cx, cy, w, h))

    return boxes, img_w, img_h


def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text) if size is not None and size.find("width") is not None else 0
    img_h = int(size.find("height").text) if size is not None and size.find("height") is not None else 0

    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text if obj.find("name") is not None else "weed"
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)

            class_id = get_class_id(label)

            if img_w > 0 and img_h > 0:
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                if w > 0 and h > 0:
                    boxes.append((class_id, cx, cy, w, h))

    return boxes, img_w, img_h


def find_images_and_annotations(source_dir):
    """Find all image files and their corresponding annotations."""
    source = Path(source_dir)
    entries = []

    # Find all images
    image_files = sorted(
        list(source.rglob("*.jpg")) +
        list(source.rglob("*.jpeg")) +
        list(source.rglob("*.png")) +
        list(source.rglob("*.JPG"))
    )

    for img_path in image_files:
        stem = img_path.stem
        parent = img_path.parent

        # Look for annotation files
        json_path = parent / f"{stem}.json"
        xml_path = parent / f"{stem}.xml"

        ann_path = None
        ann_type = None
        if json_path.exists():
            ann_path = json_path
            ann_type = "json"
        elif xml_path.exists():
            ann_path = xml_path
            ann_type = "xml"

        entries.append({
            "image": img_path,
            "annotation": ann_path,
            "ann_type": ann_type,
            "stem": stem,
        })

    return entries


def convert_dataset(source_dir, output_dir, split_ratios=(0.65, 0.20, 0.15), seed=42):
    """Convert full dataset to YOLO format with train/valid/test splits."""
    print(f"[*] Scanning source: {source_dir}")
    entries = find_images_and_annotations(source_dir)
    print(f"    Found {len(entries)} images")

    annotated = [e for e in entries if e["annotation"] is not None]
    unannotated = [e for e in entries if e["annotation"] is None]
    print(f"    With annotations: {len(annotated)}")
    print(f"    Without annotations: {len(unannotated)} (will be skipped)")

    if not annotated:
        print("[!] No annotated images found!")
        return

    # Shuffle and split
    random.seed(seed)
    random.shuffle(annotated)

    n = len(annotated)
    n_train = int(n * split_ratios[0])
    n_valid = int(n * split_ratios[1])

    splits = {
        "train": annotated[:n_train],
        "valid": annotated[n_train:n_train + n_valid],
        "test": annotated[n_train + n_valid:],
    }

    print(f"    Split: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

    # Create output directories
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Process each split
    total_boxes = 0
    class_counts = [0] * len(CLASS_NAMES)

    for split_name, split_entries in splits.items():
        print(f"\n  Processing {split_name} ({len(split_entries)} images)...")
        img_dir = os.path.join(output_dir, split_name, "images")
        label_dir = os.path.join(output_dir, split_name, "labels")

        for i, entry in enumerate(split_entries):
            # Parse annotation
            boxes = []
            if entry["ann_type"] == "json":
                boxes, _, _ = parse_labelme_json(str(entry["annotation"]))
            elif entry["ann_type"] == "xml":
                boxes, _, _ = parse_voc_xml(str(entry["annotation"]))

            # Copy image
            dst_img = os.path.join(img_dir, entry["image"].name)
            if not os.path.exists(dst_img):
                shutil.copy2(str(entry["image"]), dst_img)

            # Write YOLO label
            label_file = os.path.join(label_dir, f"{entry['stem']}.txt")
            with open(label_file, "w") as f:
                for box in boxes:
                    class_id, cx, cy, w, h = box
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_counts[class_id] += 1
                    total_boxes += 1

            if (i + 1) % 500 == 0:
                print(f"    [{i+1}/{len(split_entries)}] processed")

    # Write data.yaml
    yaml_content = f"""names:
{chr(10).join(f"- '{name}'" for name in CLASS_NAMES)}
nc: {len(CLASS_NAMES)}
train: ../train/images
val: ../valid/images
test: ../test/images
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Summary
    print(f"\n{'='*50}")
    print("CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"  Output: {output_dir}")
    print(f"  Total images: {len(annotated)}")
    print(f"  Total boxes:  {total_boxes}")
    print(f"  data.yaml:    {yaml_path}")
    print(f"\n  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        if class_counts[i] > 0:
            print(f"    {i}: {name:<20} {class_counts[i]:>5} boxes")


def main():
    parser = argparse.ArgumentParser(description="Convert CottonWeedDet12 to YOLO format")
    parser.add_argument("--source", type=str, required=True,
                        help="Source directory with images and annotations")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for YOLO format dataset")
    parser.add_argument("--split", type=float, nargs=3, default=[0.65, 0.20, 0.15],
                        help="Train/valid/test split ratios (default: 0.65 0.20 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    convert_dataset(args.source, args.output, tuple(args.split), args.seed)


if __name__ == "__main__":
    main()
