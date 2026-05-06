"""
v3.0.29 — Green-pixel curation per arXiv 2603.00160 (DINOv3+YOLO26, 2026).

Drop images whose HSV green-pixel coverage < threshold (default 20%). This
filters out non-plant noise that Brain harvested but shouldn't be in a
weed/crop detection corpus (e.g., kg_parohod__warp-waste-recycling, indoor
pest macros, raw classification disease images with no plant context).

The filter is conservative: it only drops images. cwd12 holdout images
already have stem-level NEVER_TRAIN protection upstream in mega_trainer.py.

Output: a JSON file mapping slug -> {kept, dropped, kept_paths} that the
v3.0.29 pretrain merge step reads to subset its inputs.

Pure CPU. Multi-process via concurrent.futures. ~0.5ms per image at
518×518. 244K imgs ≈ 2 minutes on 16 workers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp")


def green_pixel_fraction(img_path: str) -> Optional[float]:
    """Fraction of pixels classified as 'green plant material'.
    HSV thresholds match arXiv 2603.00160's curation pipeline:
      H ∈ [35, 85], S > 50, V > 30 (out of 255).
    Returns None on read failure.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Resize for speed — downsampling doesn't bias the fraction estimate.
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = (H >= 35) & (H <= 85) & (S > 50) & (V > 30)
        return float(mask.mean())
    except Exception:
        return None


def _worker(args):
    img_path, threshold = args
    frac = green_pixel_fraction(img_path)
    if frac is None:
        return img_path, None, False  # unreadable → drop
    return img_path, frac, frac >= threshold


def collect_imgs_under(local_path: str) -> list[str]:
    out = []
    p = Path(local_path)
    if not p.is_dir():
        return out
    for ext in IMG_EXTS:
        for f in p.rglob(f"*{ext}"):
            out.append(str(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-per-slug", type=int, default=0,
                    help="0 = no cap; for quick smoke test")
    args = ap.parse_args()

    print(f"[curate] threshold={args.threshold}  workers={args.workers}")
    with open(args.registry) as f:
        registry = json.load(f)

    datasets = registry.get("datasets", registry)
    work: list[tuple[str, str]] = []   # (slug, img_path)
    for slug, info in datasets.items():
        if not isinstance(info, dict):
            continue
        local = info.get("local_path")
        if not local or not os.path.isdir(local):
            continue
        imgs = collect_imgs_under(local)
        if args.limit_per_slug:
            imgs = imgs[: args.limit_per_slug]
        for ip in imgs:
            work.append((slug, ip))

    print(f"[curate] total imgs to score: {len(work)}")

    results: dict[str, dict] = {}
    for slug, _ in work:
        results.setdefault(slug, {"kept": 0, "dropped": 0,
                                  "kept_paths": [], "dropped_paths": []})

    payload = [(ip, args.threshold) for _, ip in work]
    slug_lookup = {ip: slug for slug, ip in work}
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for img_path, frac, keep in ex.map(_worker, payload, chunksize=32):
            slug = slug_lookup[img_path]
            entry = results[slug]
            if keep:
                entry["kept"] += 1
                entry["kept_paths"].append(img_path)
            else:
                entry["dropped"] += 1
                # only sample dropped paths for inspection
                if len(entry["dropped_paths"]) < 50:
                    entry["dropped_paths"].append(
                        {"path": img_path, "green_frac": frac}
                    )
            n += 1
            if n % 5000 == 0:
                kept_total = sum(r["kept"] for r in results.values())
                drop_total = sum(r["dropped"] for r in results.values())
                print(f"[curate] processed {n}/{len(work)} "
                      f"kept={kept_total} dropped={drop_total}")

    # Per-slug summary
    print("\n=== PER-SLUG CURATION SUMMARY ===")
    for slug, r in sorted(results.items(),
                          key=lambda kv: -(kv[1]["kept"] + kv[1]["dropped"])):
        total = r["kept"] + r["dropped"]
        if total == 0:
            continue
        ratio = r["kept"] / total
        flag = "  ← MOSTLY DROPPED" if ratio < 0.20 else ""
        print(f"  {slug:60s} kept={r['kept']:6d}/{total:6d} ({ratio:5.1%}){flag}")

    total_kept = sum(r["kept"] for r in results.values())
    total_dropped = sum(r["dropped"] for r in results.values())
    print(f"\n=== TOTAL kept={total_kept}  dropped={total_dropped}  "
          f"keep_rate={total_kept/(total_kept+total_dropped):.1%} ===")

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "total_kept": total_kept,
            "total_dropped": total_dropped,
            "per_slug": results,
        }, f, indent=1)
    print(f"\n[curate] wrote {args.out}")


if __name__ == "__main__":
    main()
