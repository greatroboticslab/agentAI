"""v3.0.43 — Track B step 1: exemplar manifest → YOLO training dataset.

Closes the human-in-loop circle:
   human ✓ in /classes/{cls}
     → results/framework/class_exemplars/{cls}.jsonl
     → THIS SCRIPT
     → results/framework/exemplar_yolo/{train,val}/{images,labels}/
     → YOLO data.yaml (12 cwd12 species, canonical class_ids)
     → consumed by mega_trainer / RF-DETR training jobs

For each ✓ entry, we emit one (image, label) pair:
  - kind='bank' (synth_cutpaste crop on transparent bg)
        → label = single bbox covering 80% of center (it's a tight crop)
  - kind='flux' (full FLUX synthetic scene with bbox label)
        → label = preserved from synth_diffusion/labels/
  - kind='reg' (real harvested image, multi-class label)
        → label = ONLY the bboxes whose class_id maps to canonical cwd12.
          Other-class bboxes dropped because the verifier said THIS species
          is correct; non-target boxes weren't verified.

Output filenames are content-hashed to avoid collisions across sources.

Run on cluster (where exemplar JSONLs and bank/flux/reg images live):
  python -m weed_optimizer_framework.tools.exemplar_to_yolo_dataset \\
      --out results/framework/exemplar_yolo \\
      --val-frac 0.15
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Cluster-default repo root (overridable via REPO_ROOT env var)
REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
sys.path.insert(0, str(REPO))

# Canonical 12-class encoding (matches dashboard, mega_trainer, etc.)
CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
CWD12_TO_ID = {n: i for i, n in enumerate(CWD12)}

EXEMPLAR_DIR = REPO / "results" / "framework" / "class_exemplars"
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"


def _content_hash(p: Path) -> str:
    h = hashlib.sha1(); h.update(str(p).encode())
    try:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk: break
                h.update(chunk)
    except Exception: pass
    return h.hexdigest()[:16]


def _read_exemplars(cls: str) -> list:
    """Replay the exemplar log → list of dicts with verdict='exemplar'."""
    fp = EXEMPLAR_DIR / f"{cls}.jsonl"
    if not fp.is_file(): return []
    state: dict = {}
    try:
        for line in fp.read_text().splitlines():
            if not line.strip(): continue
            ev = json.loads(line)
            img_key = ev.get("img", ""); v = ev.get("verdict", "")
            if not img_key or not v: continue
            if v == "clear":
                state.pop(img_key, None)
            else:
                state[img_key] = ev
    except Exception as e:
        print(f"WARN: read {fp}: {e}", file=sys.stderr)
    return [v for v in state.values() if v.get("verdict") == "exemplar"]


def _resolve_source(cls: str, img_key: str, registry: dict):
    """Resolve an exemplar img_key to (kind, abs_img_path, label_lines).
    Returns None if the source can't be located on disk."""
    parts = img_key.split("/", 2)
    if not parts:
        return None
    kind = parts[0]
    cid_canonical = CWD12_TO_ID.get(cls)

    if kind == "bank" and len(parts) == 3:
        fn = parts[2]
        src = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank" / cls / fn
        if not src.is_file(): return None
        # Bank crops are tight; emit a single near-full-image bbox.
        # Since canonical class id for cls is required, skip non-cwd12 classes.
        if cid_canonical is None:
            return None
        lines = [f"{cid_canonical} 0.5 0.5 0.95 0.95"]
        return ("bank", src, lines)

    if kind == "flux" and len(parts) >= 2:
        fn = parts[1] if len(parts) == 2 else parts[2]
        src = REPO / "results" / "framework" / "synth_diffusion" / "images" / fn
        lbl_p = REPO / "results" / "framework" / "synth_diffusion" / "labels" / (
            Path(fn).stem + ".txt")
        if not src.is_file(): return None
        # FLUX labels already use canonical cwd12 ids — use as-is.
        lines = []
        if lbl_p.is_file():
            try: lines = [l for l in lbl_p.read_text().splitlines() if l.strip()]
            except Exception: pass
        return ("flux", src, lines)

    if kind == "reg" and len(parts) == 3:
        slug = parts[1]; fn = parts[2]
        if cid_canonical is None:
            return None
        # Find the image inside slug's local_path
        info = (registry.get("datasets") or {}).get(slug)
        if not info or not info.get("local_path"):
            return None
        lp = Path(info["local_path"])
        if not lp.is_dir(): return None
        # Resolve filename (may live deeper)
        matches = list(lp.rglob(fn))
        if not matches: return None
        src = matches[0]
        # Find label file via standard images→labels swap
        lbl_p = None
        try_paths = [
            Path(str(src).replace("/images/", "/labels/")).with_suffix(".txt"),
            src.with_suffix(".txt"),
        ]
        for cand in try_paths:
            if cand.is_file():
                lbl_p = cand; break
        # Map this slug's class_id_in_slug for `cls` → canonical cwd12 id
        cn_in_slug = info.get("class_names") or []
        slug_cid = None
        for i, n in enumerate(cn_in_slug):
            if n == cls or n.lower() == cls.lower():
                slug_cid = i; break
        if slug_cid is None: return None
        # Keep only bboxes whose slug-internal cid == slug_cid; remap to canonical
        lines: list = []
        if lbl_p:
            try:
                for line in lbl_p.read_text().splitlines():
                    parts2 = line.split()
                    if len(parts2) >= 5 and parts2[0].isdigit():
                        if int(parts2[0]) == slug_cid:
                            new_line = " ".join([str(cid_canonical)] + parts2[1:])
                            lines.append(new_line)
            except Exception: pass
        if not lines:
            return None  # nothing to put in label — skip
        return ("reg", src, lines)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "framework" / "exemplar_yolo"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts but write nothing")
    args = ap.parse_args()

    if not REGISTRY_PATH.exists():
        print(f"FATAL: registry not found at {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(2)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # Resolve all ✓ entries across CWD12 classes
    resolved: list = []
    skipped: dict = {"bank_missing": 0, "flux_missing": 0,
                     "reg_no_local": 0, "reg_no_label_match": 0,
                     "non_cwd12": 0, "unknown_kind": 0}
    by_class: dict = {}
    by_kind: dict = {"bank": 0, "flux": 0, "reg": 0}

    if not EXEMPLAR_DIR.is_dir():
        print(f"WARN: no exemplar dir {EXEMPLAR_DIR} — no ✓ marks yet")

    for cls in CWD12:
        for ev in _read_exemplars(cls):
            img_key = ev.get("img", "")
            res = _resolve_source(cls, img_key, registry)
            if res is None:
                if img_key.startswith("bank/"): skipped["bank_missing"] += 1
                elif img_key.startswith("flux/"): skipped["flux_missing"] += 1
                elif img_key.startswith("reg/"): skipped["reg_no_local"] += 1
                else: skipped["unknown_kind"] += 1
                continue
            kind, src, lines = res
            resolved.append({
                "cls": cls, "kind": kind, "src": src, "lines": lines,
                "img_key": img_key, "ts": ev.get("ts", 0),
            })
            by_class[cls] = by_class.get(cls, 0) + 1
            by_kind[kind] += 1

    # Train/val split (deterministic by content hash)
    import random as _r
    _r.seed(args.seed)
    _r.shuffle(resolved)
    n_val = int(len(resolved) * args.val_frac)
    val_set = set(id(x) for x in resolved[:n_val])

    print(f"\n==== exemplar → YOLO dataset ====")
    print(f"  total exemplars resolved: {len(resolved)}")
    print(f"  per-class:")
    for cls in CWD12:
        print(f"    {cls:18s} {by_class.get(cls, 0)}")
    print(f"  per-kind: {by_kind}")
    print(f"  skipped: {skipped}")
    print(f"  train/val: {len(resolved)-n_val} / {n_val}")
    print(f"  out: {args.out}")
    print(f"  mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")

    if args.dry_run or len(resolved) == 0:
        if len(resolved) == 0:
            print("\n  no exemplars to write — has any ✓ been recorded yet?")
        sys.exit(0)

    out_root = Path(args.out)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for entry in resolved:
        split = "val" if id(entry) in val_set else "train"
        src = entry["src"]
        h = _content_hash(src)
        stem = f"{entry['cls']}_{entry['kind']}_{h}"
        ext = src.suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        img_out = out_root / split / "images" / (stem + ext)
        lbl_out = out_root / split / "labels" / (stem + ".txt")
        try:
            shutil.copy2(src, img_out)
            lbl_out.write_text("\n".join(entry["lines"]) + "\n")
            n_written += 1
        except Exception as e:
            print(f"WARN: copy {src} → {img_out}: {e}", file=sys.stderr)

    # data.yaml — canonical CWD12
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"# Auto-generated by exemplar_to_yolo_dataset v3.0.43\n"
        f"# Source: human ✓ marks from /classes UI exemplar JSONLs.\n"
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"path: {out_root}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(CWD12)}\n"
        f"names: {json.dumps(CWD12)}\n"
    )

    # Manifest for audit
    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total": n_written,
        "by_class": by_class,
        "by_kind": by_kind,
        "skipped": skipped,
        "val_frac": args.val_frac,
        "seed": args.seed,
    }, indent=2))

    print(f"\n  WROTE {n_written} image+label pairs to {out_root}")
    print(f"  data.yaml: {data_yaml}")
    print(f"  manifest:  {manifest}")
    print(f"  → train via: yolo train data={data_yaml} ...")
    print(f"     or RF-DETR — see mega_trainer for canonical CLI")


if __name__ == "__main__":
    main()
