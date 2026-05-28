"""
class_metadata_backfiller.py — fill in missing class_names for registered slugs.

For each slug in dataset_registry.json with class_names=[] and an existing
local_path, infer class names from:
  1. data.yaml / data.yml (YOLO standard, key 'names:')
  2. classes.txt / obj.names / _classes.csv (Darknet / Roboflow)
  3. Subdirectory structure (classification-style: train/<ClassA>/*.jpg)

Dry-run by default. Pass --commit to actually write back to registry.
A timestamped .bak is created before write.

Usage (run on cluster where local_paths exist):
  python -m weed_optimizer_framework.tools.class_metadata_backfiller --dry-run
  python -m weed_optimizer_framework.tools.class_metadata_backfiller --commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"

# Subdirs that are clearly structural, not class folders
_STRUCT_DIRS = {
    "images", "labels", "annotations", "masks", "img", "gt",
    "val", "test", "train", "valid", "validation", "training", "testing",
    "data", "raw", "processed", "yolo", ".git", "__pycache__",
    "splits", "datasets", "sample", "samples", "examples", "metadata",
    "obj_train_data", "obj_test_data", "obj_val_data",
    # Repo / code dirs (false-positive sources observed in practice)
    "docs", "docker", "tests", "ultralytics", "src", "code", "scripts",
    "predictions", "runs", "weights", "checkpoints", "models", "venv",
    "node_modules", "build", "dist", ".github", ".vscode", ".idea",
    "notebooks", "logs", "log", "outputs", "output", "results",
    "yolov5", "yolov7", "yolov8", "yolov9",
}

# Substrings indicating a dir name is a DATASET VARIANT, not a class name.
_GARBAGE_NAME_KW = (
    "dataset", "augmented", "_aug", "all images", "merged",
    "original", "version", "kag2", "kag3", "kag4",
    "invalid", "unsorted", "todo", "unused", "leaves",
    "hog", "black and white", "old", "new ",
    "plantvillage", "plant-village",
)


def _looks_garbage(name: str) -> bool:
    """Does this directory name look like a NON-class (dataset variant)?"""
    nl = name.strip().lower()
    if any(kw in nl for kw in _GARBAGE_NAME_KW):
        return True
    # Parenthesized version notation: "Foo (1)", "Bar(Augmented)"
    if "(" in name or ")" in name:
        return True
    return False


def _looks_repo(p: Path) -> bool:
    """Is this directory a code repository (not a dataset)?"""
    if not p.is_dir():
        return False
    indicators = (
        ".git", ".github", "setup.py", "pyproject.toml",
        "package.json", "Cargo.toml", "requirements.txt",
        "Dockerfile", ".gitignore",
    )
    return any((p / x).exists() for x in indicators)


def _read_yaml_names(yp: Path) -> Optional[list[str]]:
    """Parse YOLO data.yaml-style 'names:' field. Returns list of class names
    in cid order, or None on failure."""
    try:
        import yaml  # type: ignore
        with open(yp) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    names = data.get("names")
    if isinstance(names, dict):
        try:
            ordered = sorted(((int(k), str(v)) for k, v in names.items()),
                             key=lambda kv: kv[0])
            return [v for _, v in ordered]
        except Exception:
            return None
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def _read_txt_lines(fp: Path) -> Optional[list[str]]:
    """Parse a simple one-name-per-line file. Strips blanks/comments."""
    try:
        raw = fp.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    names: list[str] = []
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Handle CSV-like "0,catname" by dropping leading int+sep
        if "," in s:
            parts = s.split(",", 1)
            if parts[0].strip().isdigit():
                names.append(parts[1].strip())
                continue
        names.append(s)
    if 1 <= len(names) <= 500:
        return names
    return None


def _detect_subdirs(base: Path, depth: int = 0,
                    max_depth: int = 3) -> Optional[list[str]]:
    """Look for one subdir per class. Smart:
      - Skips structural dirs (images/labels/train/etc.)
      - Skips code repos (.git/setup.py/etc.) at depth > 0
      - Recurses into 'wrapper' dirs that contain a single PlantVillage-style
        sub-layer
      - Filters out 'dataset variant' names (Augmented/PlantVillage/Invalid).
    """
    if depth > max_depth or not base.is_dir():
        return None
    # At deeper levels, reject if it looks like a code repo not data
    if depth > 0 and _looks_repo(base):
        return None
    try:
        subdirs = [d for d in base.iterdir() if d.is_dir()]
    except Exception:
        return None

    # Strip structural & garbage names
    non_struct = [d for d in subdirs if d.name.lower() not in _STRUCT_DIRS]
    clean = [d for d in non_struct if not _looks_garbage(d.name)]

    # If few clean candidates but more total, treat the dir as a wrapper:
    # recurse into the most-promising candidate.
    if len(clean) <= 2 and len(non_struct) <= 4 and depth < max_depth:
        # Prefer recursing into a non-garbage candidate first, else fall back
        recurse_order = clean + [d for d in non_struct if d not in clean]
        for cand in recurse_order:
            nested = _detect_subdirs(cand, depth + 1, max_depth)
            if nested and len(nested) >= 3:
                return nested

    if 2 <= len(clean) <= 200:
        return sorted(d.name for d in clean)
    return None


def detect_class_names(slug: str, lp: Path) -> tuple[list[str], str]:
    """Return (class_names, source_marker) for slug at local_path lp.
    source_marker examples:
      'yaml:data.yaml'              — YOLO standard
      'file:classes.txt'            — flat list file
      'subdirs:train'               — classification dirs
      ''                            — could not detect
    """
    # 1) data.yaml in root, then any subdir up to 1 level deep
    yaml_candidates: list[Path] = []
    for fn in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = lp / fn
        if p.is_file():
            yaml_candidates.append(p)
    if lp.is_dir():
        try:
            for child in list(lp.iterdir())[:80]:
                if child.is_dir():
                    for fn in ("data.yaml", "data.yml"):
                        p = child / fn
                        if p.is_file():
                            yaml_candidates.append(p)
        except Exception:
            pass
    for yp in yaml_candidates:
        names = _read_yaml_names(yp)
        if names:
            return names, f"yaml:{yp.relative_to(lp)}"

    # 2) flat name files
    for fn in ("classes.txt", "class_names.txt", "labels.txt", "obj.names",
               "_classes.csv", "classes.csv", "names.txt"):
        try:
            for p in sorted(lp.rglob(fn))[:5]:
                names = _read_txt_lines(p)
                if names:
                    return names, f"file:{p.relative_to(lp)}"
        except Exception:
            pass

    # 3) classification-style: try common bases.
    # v3.0.43.22: added bare wrapper dirs 'dataset','data','Data','train_images'
    # — kg_saroz014__plant-disease stores classes under {lp}/dataset/<class>,
    # kg_nirmalsankalana__plant-diseases-training-dataset under {lp}/data/<class>.
    # These were invisible because only '<wrapper>/train' was probed before.
    for base_rel in ("train", "training", "images/train", "train/images",
                     "data/train", "dataset", "data", "Data",
                     "dataset/train", "train_images", ""):
        base = lp / base_rel if base_rel else lp
        names = _detect_subdirs(base)
        if names:
            return names, f"subdirs:{base_rel or '.'}"

    return [], ""


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="write changes back to registry (default dry-run)")
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--max-print-names", type=int, default=8,
                    help="how many detected names to print per slug")
    ap.add_argument("--only", default=None,
                    help="comma-list of slugs to operate on (default all empty)")
    args = ap.parse_args()

    reg_path = Path(args.registry)
    if not reg_path.is_file():
        print(f"FATAL: registry not found at {reg_path}", file=sys.stderr)
        sys.exit(2)
    with open(reg_path) as f:
        reg = json.load(f)

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None

    detected_rows: list[dict] = []
    skip_no_local: list[str] = []
    skip_already: list[str] = []
    manual_needed: list[str] = []

    datasets = reg.get("datasets") or {}
    for slug, info in datasets.items():
        if only_set and slug not in only_set:
            continue
        cn = info.get("class_names") or []
        lp = info.get("local_path") or ""
        if cn:
            skip_already.append(slug)
            continue
        if not lp or not os.path.isdir(lp):
            skip_no_local.append(slug)
            continue
        names, src = detect_class_names(slug, Path(lp))
        if names:
            detected_rows.append({
                "slug": slug, "n": len(names), "src": src,
                "names": names,
                "classes_field": info.get("classes"),
            })
        else:
            manual_needed.append(slug)

    print(f"\n==== class_names backfill — registry={reg_path} ====")
    print(f"  total slugs:       {len(datasets)}")
    print(f"  already filled:    {len(skip_already)}  (skip)")
    print(f"  no local path:     {len(skip_no_local)}  (skip)")
    print(f"  ✓ AUTO-DETECTED:   {len(detected_rows)}")
    print(f"  ✗ NEEDS MANUAL:    {len(manual_needed)}")
    print(f"  mode:              {'COMMIT' if args.commit else 'DRY-RUN'}")
    print()

    if detected_rows:
        print("---- DETECTED (will be written) ----")
        # widest slug name for alignment
        wid = max(len(r["slug"]) for r in detected_rows)
        for r in sorted(detected_rows, key=lambda x: x["src"]):
            slug = r["slug"]; n = r["n"]; src = r["src"]
            sample = ", ".join(r["names"][:args.max_print_names])
            more = f" (+{n-args.max_print_names})" if n > args.max_print_names else ""
            cf = r.get("classes_field")
            cf_mark = ""
            if isinstance(cf, int) and cf != n:
                cf_mark = f"  ⚠ registry classes={cf} != detected n={n}"
            print(f"  {slug:<{wid}}  n={n:<3} src={src:<32} [{sample}{more}]{cf_mark}")
        print()

    if manual_needed:
        print("---- ✗ NEEDS MANUAL (no class metadata source found) ----")
        for s in sorted(manual_needed):
            print(f"  {s}")
        print()

    if not args.commit:
        print("DRY-RUN — nothing written. Re-run with --commit to apply.")
        return

    # Backup + write
    bak = reg_path.with_suffix(f".bak.{int(time.time())}.json")
    with open(bak, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"BACKUP: {bak}")

    for r in detected_rows:
        slug = r["slug"]
        datasets[slug]["class_names"] = r["names"]
        datasets[slug]["class_names_source"] = r["src"]
        datasets[slug]["class_names_backfilled_at"] = int(time.time())

    tmp = reg_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, reg_path)
    print(f"WROTE: {reg_path}  ({len(detected_rows)} slugs updated)")


if __name__ == "__main__":
    main()
