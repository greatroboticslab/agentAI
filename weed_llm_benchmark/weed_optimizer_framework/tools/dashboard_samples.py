"""
v3.0.30.1 — Dashboard image samples generator.

For each slug, pick a handful of images, draw their bboxes (from YOLO/COCO
labels), save 400px-wide thumbnails to docs/dashboard/samples/{slug}/.
The slug page renders a 2x3 image grid so a human can SEE label quality
at a glance.

Color-codes:
  green box = real human-labeled bbox (annotation = bbox / yolo / bbox+seg)
  orange box = AI-labeled (OWLv2 / yolo_autolabel)
  red border on tile = NEVER_TRAIN slug (must never leak into training)

Cap: 6 images per slug × 71 slugs = ~430 thumbnails. At ~50KB each = ~21MB.
Acceptable for git LFS-free GitHub repo.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

NEVER_TRAIN = {"cottonweeddet12", "weedsense", "francesco__weed_crop_aerial"}


def find_label_for_image(img_path: Path, ds_root: Path):
    """Try the standard YOLO conventions to find an image's label."""
    stem = img_path.stem
    # sibling labels/ swap
    if "images" in img_path.parts:
        cand = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if cand.exists():
            return cand
    # same dir
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same
    # walk up to ds_root looking for labels/
    for parent in img_path.parents:
        cand = parent / "labels" / (stem + ".txt")
        if cand.exists():
            return cand
        if parent == ds_root:
            break
    return None


def parse_yolo_labels(label_path: Path):
    """Return list of (class_id, [cx, cy, bw, bh] normalized)."""
    out = []
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            out.append((cid, [cx, cy, bw, bh]))
    except Exception:
        pass
    return out


def render_sample(img_path: Path, label_path: Path | None,
                  output_path: Path, max_width: int,
                  box_color: tuple, class_names: list[str]):
    """Read image, draw bboxes if present, write thumbnail."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if label_path is not None:
        for cid, (cx, cy, bw, bh) in parse_yolo_labels(label_path):
            x1 = int(max(0, (cx - bw / 2) * w))
            y1 = int(max(0, (cy - bh / 2) * h))
            x2 = int(min(w, (cx + bw / 2) * w))
            y2 = int(min(h, (cy + bh / 2) * h))
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, max(2, w // 250))
            label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = max(0.4, w / 1200)
            cv2.putText(img, label, (x1, max(15, y1 - 6)), font, fontscale,
                        box_color, max(1, w // 600), cv2.LINE_AA)
    # downscale
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def collect_images(local_path: Path, max_imgs: int = 6) -> list[Path]:
    """Find up to max_imgs random image paths under local_path."""
    if not local_path.is_dir():
        return []
    candidates = []
    for ext in IMG_EXTS:
        for p in local_path.rglob(f"*{ext}"):
            candidates.append(p)
            if len(candidates) > 1000:  # cap scan for speed
                break
        if len(candidates) > 1000:
            break
    if not candidates:
        return []
    random.shuffle(candidates)
    return candidates[:max_imgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="docs/dashboard/samples")
    ap.add_argument("--max-imgs", type=int, default=6)
    ap.add_argument("--max-width", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with open(args.registry) as f:
        registry = json.load(f)
    datasets = registry.get("datasets", {})

    per_slug_samples = {}
    for slug, info in datasets.items():
        if not isinstance(info, dict):
            continue
        local = info.get("local_path")
        if not local:
            continue
        local = Path(local)
        if not local.is_dir():
            continue
        ann = info.get("annotation") or ""
        is_ai = (ann == "yolo_autolabel")
        is_real = ann in ("bbox", "yolo", "bbox+segmentation")
        box_color = (0, 200, 0) if is_real else (0, 140, 240) if is_ai else (160, 160, 160)
        # class names: cottonweed slugs use canonical 12; autolabel uses single "weed"
        if slug in ("cottonweed_sp8", "cottonweed_holdout", "cottonweeddet12"):
            class_names = ["Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
                           "Morningglory", "Nutsedge", "PalmerAmaranth",
                           "PricklySida", "Purslane", "Ragweed", "Sicklepod",
                           "SpottedSpurge"]  # cwd12 original order
        else:
            class_names = info.get("class_names") or ["target"]

        imgs = collect_images(local, args.max_imgs)
        if not imgs:
            continue

        slug_dir = out_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        rendered = []
        for img_path in imgs:
            lbl = find_label_for_image(img_path, local)
            out_path = slug_dir / (img_path.stem + ".jpg")
            ok = render_sample(img_path, lbl, out_path, args.max_width,
                               box_color, class_names)
            if ok:
                rendered.append({
                    "img": f"samples/{slug}/{img_path.stem}.jpg",
                    "label_present": lbl is not None,
                    "n_boxes": len(parse_yolo_labels(lbl)) if lbl else 0,
                })
        per_slug_samples[slug] = {
            "annotation_h": _ann_human(ann),
            "is_real": is_real,
            "is_ai": is_ai,
            "is_never_train": slug in NEVER_TRAIN,
            "samples": rendered,
        }
        print(f"[samples] {slug}: rendered {len(rendered)} imgs")

    summary_path = out_dir / "_samples_summary.json"
    with open(summary_path, "w") as f:
        json.dump(per_slug_samples, f, indent=1)
    print(f"[samples] wrote {summary_path} — covered {len(per_slug_samples)} slugs")


def _ann_human(ann: str) -> str:
    return {
        "bbox": "Real bbox (human)",
        "bbox+segmentation": "Real bbox+seg (human)",
        "yolo": "Real YOLO (human)",
        "yolo_autolabel": "AI-labeled (OWLv2)",
        "classification": "Classification only (no bbox)",
    }.get(ann, ann or "unknown")


if __name__ == "__main__":
    main()
