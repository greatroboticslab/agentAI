"""
bucketer.py — classify each downloaded slug into A/B/C buckets for the
Roboflow active-learning pipeline.

Buckets (see memory/project_roboflow_pipeline_plan.md):
  A — detection-ready: has YOLO labels/ dirs with .txt files. Boxes exist.
      → directly trainable (after sanity checks).
  B — classification-only: no labels/, but class_names recorded.
      → species KNOWN, location UNKNOWN. Needs bbox labeling (Roboflow).
  C — unknown: no labels, no class_names.
      → species + location both unknown. Needs DINOv2 routing + bbox.

Output: JSON to --out path (default results/framework/buckets.json) with
  - per-slug bucket assignment + image count + class_names
  - per-bucket totals
  - per-species coverage in bucket A (which species have how many boxed imgs)

Probes use bounded iterdir (no rglob) — same Lustre-safety rule as the
dashboard's _find_label_dirs (see [[project_classes_thumb_perf]]). A
classification slug yields [] instantly.

Usage:
  python -m weed_optimizer_framework.tools.bucketer
  python -m weed_optimizer_framework.tools.bucketer --out /path/to/buckets.json
  python -m weed_optimizer_framework.tools.bucketer --verbose

Wired as a /control action of type=subprocess (action name: build_buckets).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
DEFAULT_OUT = REPO / "results" / "framework" / "buckets.json"


CWD12 = (
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
)


def _find_label_dirs(local_p: Path, max_dirs: int = 64) -> list:
    """Bounded probe for YOLO `labels/` dirs — never rglob. Mirrors the
    dashboard's `_find_label_dirs` so behavior matches."""
    SPLIT_HINTS = {
        "train", "training", "valid", "validation", "val", "test", "testing",
        "images", "data", "dataset", "yolo", "splits", "obj_train_data",
        "obj_val_data", "obj_test_data",
    }
    found: list = []
    d0 = local_p / "labels"
    if d0.is_dir():
        found.append(d0)
    try:
        for child in local_p.iterdir():
            if not child.is_dir() or child.name == "labels":
                continue
            d1 = child / "labels"
            if d1.is_dir():
                found.append(d1)
                if len(found) >= max_dirs:
                    return found
            elif child.name.lower() in SPLIT_HINTS:
                try:
                    for gchild in child.iterdir():
                        if gchild.is_dir():
                            d2 = gchild / "labels"
                            if d2.is_dir():
                                found.append(d2)
                                if len(found) >= max_dirs:
                                    return found
                except Exception:
                    pass
    except Exception:
        pass
    return found


def _has_yolo_txt(label_dirs: list, max_scan: int = 50) -> bool:
    """True if any of the label dirs holds >=1 .txt file (bounded scan)."""
    n = 0
    for ld in label_dirs:
        try:
            for f in ld.iterdir():
                n += 1
                if f.suffix.lower() == ".txt":
                    return True
                if n >= max_scan:
                    return False
        except Exception:
            pass
    return False


def _scan_bucket_a_species(local_p: Path, label_dirs: list,
                            class_names: list,
                            per_slug_max_files: int = 4000) -> Counter:
    """For an A-bucket slug, count images per cwd12 species (if its
    class_names overlap CWD12). Bounded label scan."""
    counts: Counter = Counter()
    n_scanned = 0
    cwd12_index_by_canon = {c.lower(): i for i, c in enumerate(CWD12)}

    # build a mapping from this slug's cid → CWD12 canonical (if match)
    cid_to_cwd12: dict = {}
    for cid, raw in enumerate(class_names):
        key = "".join(ch for ch in str(raw).lower() if ch.isalnum())
        # try direct CWD12 match
        for sp_canon, sp_idx in cwd12_index_by_canon.items():
            if "".join(ch for ch in sp_canon if ch.isalnum()) == key:
                cid_to_cwd12[cid] = CWD12[sp_idx]
                break

    if not cid_to_cwd12:
        return counts

    for ldir in label_dirs:
        if n_scanned >= per_slug_max_files:
            break
        try:
            txts = sorted(ldir.glob("*.txt"))
        except Exception:
            continue
        for lbl in txts:
            n_scanned += 1
            if n_scanned >= per_slug_max_files:
                break
            try:
                text = lbl.read_text(errors="ignore")
            except Exception:
                continue
            cids_in_img: set = set()
            for line in text.splitlines():
                p = line.split()
                if p and p[0].lstrip("-").isdigit():
                    cid = int(p[0])
                    if cid in cid_to_cwd12:
                        cids_in_img.add(cid_to_cwd12[cid])
            for sp in cids_in_img:
                counts[sp] += 1
    return counts


def classify_one(slug: str, info: dict) -> dict:
    out = {"slug": slug, "bucket": None, "n_images": info.get("local_images", 0),
           "class_names": info.get("class_names") or [],
           "local_path": info.get("local_path") or "",
           "n_label_dirs": 0, "label_dirs_sample": []}
    lp = info.get("local_path") or ""
    if not lp or not os.path.isdir(lp):
        out["bucket"] = "absent"
        return out
    label_dirs = _find_label_dirs(Path(lp))
    if label_dirs and _has_yolo_txt(label_dirs):
        out["bucket"] = "A"
        out["n_label_dirs"] = len(label_dirs)
        out["label_dirs_sample"] = [str(d.relative_to(Path(lp))) for d in label_dirs[:3]]
    elif info.get("class_names"):
        out["bucket"] = "B"
    else:
        out["bucket"] = "C"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cwd12-coverage", action="store_true", default=True,
                    help="Also compute per-cwd12-species image counts for A-bucket slugs")
    args = ap.parse_args()

    with open(args.registry) as f:
        reg = json.load(f)

    t0 = time.time()
    rows: list = []
    bucket_counts: Counter = Counter()
    bucket_imgs: Counter = Counter()
    cwd12_species_imgs: Counter = Counter()

    for slug, info in (reg.get("datasets") or {}).items():
        if info.get("status") != "downloaded":
            continue
        row = classify_one(slug, info)
        rows.append(row)
        bucket_counts[row["bucket"]] += 1
        bucket_imgs[row["bucket"]] += row["n_images"]
        if args.verbose:
            print(f"  [{row['bucket']}] {slug:55s} imgs={row['n_images']:>7,} cn={len(row['class_names'])}")

    # Per-CWD12-species coverage from A-bucket slugs (bounded label scan)
    species_per_slug: dict = {}
    if args.cwd12_coverage:
        for row in rows:
            if row["bucket"] != "A":
                continue
            slug = row["slug"]
            lp = row["local_path"]
            label_dirs = _find_label_dirs(Path(lp))
            counts = _scan_bucket_a_species(Path(lp), label_dirs, row["class_names"])
            species_per_slug[slug] = dict(counts)
            for sp, c in counts.items():
                cwd12_species_imgs[sp] += c

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "registry_path": args.registry,
        "elapsed_sec": round(time.time() - t0, 1),
        "totals_by_bucket": dict(bucket_counts),
        "images_by_bucket": dict(bucket_imgs),
        "cwd12_species_images_in_A": dict(cwd12_species_imgs),
        "cwd12_species_missing_from_A":
            [sp for sp in CWD12 if cwd12_species_imgs.get(sp, 0) == 0],
        "n_slugs_audited": len(rows),
        "per_slug": rows,
        "cwd12_species_per_slug_in_A": species_per_slug,
    }

    # Print compact summary always
    print(f"\n=== bucket audit ({summary['elapsed_sec']}s) ===")
    print(f"  A detection-ready: {bucket_counts.get('A', 0)} slugs, {bucket_imgs.get('A', 0):>9,} imgs")
    print(f"  B class-only     : {bucket_counts.get('B', 0)} slugs, {bucket_imgs.get('B', 0):>9,} imgs")
    print(f"  C unknown        : {bucket_counts.get('C', 0)} slugs, {bucket_imgs.get('C', 0):>9,} imgs")
    print(f"\n=== cwd12 coverage in A (excl. holdout if separately tagged) ===")
    for sp in CWD12:
        n = cwd12_species_imgs.get(sp, 0)
        flag = "OK" if n >= 50 else ("THIN" if n > 0 else "MISSING")
        print(f"  {sp:18s} {n:>5}  [{flag}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"\nWROTE: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
