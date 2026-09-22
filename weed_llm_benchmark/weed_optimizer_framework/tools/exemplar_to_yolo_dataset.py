"""v3.0.43 — Track B step 1: exemplar manifest → YOLO training dataset.

Closes the human-in-loop circle:
   human ✓ in /classes/{cls}
     → results/framework/class_exemplars/{cls}.jsonl
     → THIS SCRIPT
     → results/framework/exemplar_yolo_species/{train,val}/{images,labels}/
     → YOLO data.yaml (12 cwd12 species, label id = cwd12 id)
     → consumed by mega_trainer / RF-DETR training jobs

v3.60.0: every log is read and each event is keyed by the species it records
("class"; older events are re-keyed by their source, as the dashboard does).
Nothing from a NEVER_TRAIN slug or a cwd12 test/valid photograph is written.

For each ✓ entry, we emit one (image, label) pair:
  - kind='bank' (synth_cutpaste crop on transparent bg)
        → label = single bbox covering 80% of center (it's a tight crop)
  - kind='flux' (full FLUX synthetic scene with bbox label)
        → label = preserved from synth_diffusion_species/labels/ (the older
          synth_diffusion/ output is skipped: its ids do not name the plant)
  - kind='reg' (real harvested image, multi-class label)
        → label = ONLY the bboxes whose class is this species.
          Other-class bboxes dropped because the verifier said THIS species
          is correct; non-target boxes weren't verified.

Output filenames are content-hashed to avoid collisions across sources.

Run on cluster (where exemplar JSONLs and bank/flux/reg images live):
  python -m weed_optimizer_framework.tools.exemplar_to_yolo_dataset \\
      --out results/framework/exemplar_yolo_species \\
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

from weed_optimizer_framework.tools.cwd12_species import (  # noqa: E402
    CWD12_ID_SPACE, CWD12_LEGACY_LABELS, CWD12_SPECIES, class_species,
    is_legacy_label_list, legacy_to_species, species_of,
)

# v3.60.0: classes are the 12 cwd12 species and label ids are cwd12 ids
# (CWD12_SPECIES order), not positions in the legacy label list, which put
# e.g. SpottedSpurge at id 11 (CutleafGroundcherry).
CWD12 = list(CWD12_SPECIES)
CWD12_TO_ID = {n: i for i, n in enumerate(CWD12)}

EXEMPLAR_DIR = REPO / "results" / "framework" / "class_exemplars"
REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
# v3.60.0: crops of the species bank are keyed 'banksp/<species>/<fn>' (the
# dashboard marks whichever bank synth_cutpaste.default_bank_dir() selects);
# 'bank/<folder>/<fn>' always names the legacy object_bank/.
SPECIES_BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank_species"
FLUX_SPECIES_PREFIX = "fluxsp_"   # synth_diffusion.FLUX_SPECIES_PREFIX
FLUX_LEGACY_DIR = REPO / "results" / "framework" / "synth_diffusion"
FLUX_SPECIES_DIR = REPO / "results" / "framework" / "synth_diffusion_species"
# the four names cottonweed_holdout was registered with before v3.60.0
_HOLDOUT_LEGACY_NAMES = list(CWD12_LEGACY_LABELS[2:6])


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


def _legacy_entry_class(logged_as: str, img_key: str, registry: dict) -> str:
    """Class of an exemplar event that records no "class" (written before
    v3.60.0, when a log was named by its /classes page, a legacy label for the
    cwd12 group): the species of its source when that is a cwd12 class, else
    the name it was logged under. Same rule as dashboard_server's reader."""
    parts = str(img_key).split("/")
    kind = parts[0]
    legacy = logged_as in CWD12_LEGACY_LABELS
    if kind == "banksp" and len(parts) >= 3 and parts[1] in CWD12_SPECIES:
        return parts[1]
    if kind in ("bank", "flux"):
        if kind == "bank" and len(parts) >= 3 and parts[1] in CWD12_LEGACY_LABELS:
            return legacy_to_species(parts[1])
        return legacy_to_species(logged_as) if legacy else logged_as
    slug = parts[1] if kind == "reg" and len(parts) >= 3 else kind
    if slug == "cottonweed_holdout" and logged_as in _HOLDOUT_LEGACY_NAMES:
        return CWD12_ID_SPACE[slug][_HOLDOUT_LEGACY_NAMES.index(logged_as)]
    if legacy and slug in CWD12_ID_SPACE:
        return legacy_to_species(logged_as)
    if legacy:
        info = (registry.get("datasets") or {}).get(slug) or {}
        if is_legacy_label_list(info.get("class_names") or []):
            return legacy_to_species(logged_as)
    return species_of(logged_as) or logged_as


def _read_all_exemplars(registry: dict) -> dict:
    """Replay every exemplar log in time order -> {species: [event]} with the
    latest verdict 'exemplar'. v3.60.0: an event is keyed by its "class" field
    (or, for an older event, _legacy_entry_class), never by the log file name
    alone, since a legacy-named log mixed species."""
    events = []
    if not EXEMPLAR_DIR.is_dir():
        return {}
    for fi, fp in enumerate(sorted(EXEMPLAR_DIR.glob("*.jsonl"))):
        try:
            lines = fp.read_text().splitlines()
        except Exception as e:
            print(f"WARN: read {fp}: {e}", file=sys.stderr)
            continue
        for n, line in enumerate(lines):
            if not line.strip(): continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            img_key = ev.get("img", ""); v = ev.get("verdict", "")
            if not img_key or not v: continue
            cls = ev.get("class") or _legacy_entry_class(fp.stem, img_key, registry)
            try:
                ts = float(ev.get("ts") or 0)
            except Exception:
                ts = 0.0
            events.append((ts, fi, n, cls, img_key, ev))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    state: dict = {}
    for _ts, _fi, _n, cls, img_key, ev in events:
        if ev.get("verdict") == "clear":
            state.get(cls, {}).pop(img_key, None)
        else:
            state.setdefault(cls, {})[img_key] = ev
    return {cls: [e for e in d.values() if e.get("verdict") == "exemplar"]
            for cls, d in state.items()}


def _holdout_guard():
    """(synth_cutpaste module, guard): NEVER_TRAIN slugs + cwd12 test/valid
    stems and dHashes, the filter mega_trainer applies before training."""
    from weed_optimizer_framework.tools import synth_cutpaste as _sc
    return _sc, _sc._holdout_guard(with_hashes=True)


def _resolve_source(cls: str, img_key: str, registry: dict, guard=None):
    """Resolve an exemplar img_key to (kind, abs_img_path, label_lines), or
    (None, reason) when it can't be used."""
    parts = img_key.split("/", 2)
    if not parts:
        return None, "unknown_kind"
    kind = parts[0]
    cid_canonical = CWD12_TO_ID.get(cls)
    if cid_canonical is None:
        return None, "non_cwd12"
    sc, g = guard if guard is not None else _holdout_guard()

    if kind in ("bank", "banksp") and len(parts) == 3:
        folder, fn = parts[1], parts[2]
        # 'bank' = the legacy object_bank (folders named by legacy label),
        # 'banksp' = object_bank_species (bank_folder_species reads both)
        root = SPECIES_BANK_DIR if kind == "banksp" else BANK_DIR
        src = root / folder / fn
        if not src.is_file(): return None, "bank_missing"
        if sc.bank_folder_species(root, folder) != cls:
            return None, "bank_other_species"
        if not sc.bank_crop_usable(root, src):
            return None, "holdout"
        # Bank crops are tight; emit a single near-full-image bbox.
        return ("bank", src, [f"{cid_canonical} 0.5 0.5 0.95 0.95"]), None

    if kind == "flux" and len(parts) >= 2:
        fn = parts[1] if len(parts) == 2 else parts[2]
        # v3.60.0: images in synth_diffusion/ were prompted with a legacy label
        # as if it were the species and labelled with that label's id, so the
        # id does not name the plant drawn; only species-era output is used.
        # Species-era files carry FLUX_SPECIES_PREFIX, so a legacy file of
        # the same class name never shadows one.
        if not fn.startswith(FLUX_SPECIES_PREFIX):
            if (FLUX_LEGACY_DIR / "images" / fn).is_file():
                return None, "flux_legacy_ids"
            return None, "flux_missing"
        src = FLUX_SPECIES_DIR / "images" / fn
        lbl_p = FLUX_SPECIES_DIR / "labels" / (Path(fn).stem + ".txt")
        if not src.is_file(): return None, "flux_missing"
        lines = []
        if lbl_p.is_file():
            try: lines = [l for l in lbl_p.read_text().splitlines() if l.strip()]
            except Exception: pass
        return ("flux", src, lines), None

    if kind == "reg" and len(parts) == 3:
        slug = parts[1]; fn = parts[2]
        if slug in g["never"]:
            return None, "holdout"
        # Find the image inside slug's local_path
        info = (registry.get("datasets") or {}).get(slug)
        if not info or not info.get("local_path"):
            return None, "reg_no_local"
        lp = Path(info["local_path"])
        if not lp.is_dir(): return None, "reg_no_local"
        # Resolve filename (may live deeper)
        matches = list(lp.rglob(fn))
        if not matches: return None, "reg_no_local"
        src = matches[0]
        if sc._is_holdout_photo(src, g):
            return None, "holdout"
        # Find label file via standard images→labels swap
        lbl_p = None
        try_paths = [
            Path(str(src).replace("/images/", "/labels/")).with_suffix(".txt"),
            src.with_suffix(".txt"),
        ]
        for cand in try_paths:
            if cand.is_file():
                lbl_p = cand; break
        # Keep only bboxes whose slug class is species `cls` (cwd12 copies by
        # label-file id, other slugs by the real name); remap to its cwd12 id
        cn_in_slug = info.get("class_names") or []
        lines: list = []
        if lbl_p:
            try:
                for line in lbl_p.read_text().splitlines():
                    parts2 = line.split()
                    if len(parts2) >= 5 and parts2[0].isdigit():
                        if class_species(slug, int(parts2[0]), cn_in_slug) == cls:
                            new_line = " ".join([str(cid_canonical)] + parts2[1:])
                            lines.append(new_line)
            except Exception: pass
        if not lines:
            return None, "reg_no_label_match"
        return ("reg", src, lines), None

    return None, "unknown_kind"


def _out_dir_conflict(out_root: Path):
    """Why `out_root` must not be written, or None. Label files are added to
    whatever is there, so a directory holding output in another class space
    (a data.yaml whose names are not the species) would end up mixing ids."""
    dy = out_root / "data.yaml"
    if not dy.is_file():
        return None
    try:
        for line in dy.read_text().splitlines():
            if line.startswith("names:"):
                if json.loads(line.split(":", 1)[1].strip()) == CWD12:
                    return None
                break
    except Exception:
        pass
    return (f"{dy} is not in the species class space (written before v3.60.0?); "
            f"pass another --out")


def main():
    ap = argparse.ArgumentParser()
    # v3.60.0: exemplar_yolo/ holds output in the legacy label ids; species
    # output goes to its own directory (see _out_dir_conflict)
    ap.add_argument("--out", default=str(REPO / "results" / "framework" / "exemplar_yolo_species"))
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
    skipped: dict = {"bank_missing": 0, "bank_other_species": 0,
                     "flux_missing": 0, "flux_legacy_ids": 0,
                     "reg_no_local": 0, "reg_no_label_match": 0,
                     "holdout": 0, "non_cwd12": 0, "unknown_kind": 0}
    by_class: dict = {}
    by_kind: dict = {"bank": 0, "flux": 0, "reg": 0}   # banksp counts as bank

    if not EXEMPLAR_DIR.is_dir():
        print(f"WARN: no exemplar dir {EXEMPLAR_DIR} — no ✓ marks yet")

    guard = _holdout_guard()
    exemplars = _read_all_exemplars(registry)
    for cls, evs in sorted(exemplars.items()):
        if cls not in CWD12_TO_ID:
            skipped["non_cwd12"] += len(evs)
            continue
        for ev in evs:
            img_key = ev.get("img", "")
            res, why = _resolve_source(cls, img_key, registry, guard)
            if res is None:
                skipped[why] = skipped.get(why, 0) + 1
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
    conflict = _out_dir_conflict(out_root)
    if conflict:
        print(f"FATAL: {conflict}", file=sys.stderr)
        sys.exit(2)
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

    # data.yaml — the cwd12 species by cwd12 id
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"# Auto-generated by exemplar_to_yolo_dataset v3.60.0\n"
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
