"""
export_owl_exemplars.py — generate per-species exemplar JSON configs that
owl_preannotate.py consumes.

Without this, `owl_preannotate_one` fails with
`FATAL: exemplar config not found` because there's no
`results/framework/owl_exemplars/<species>.json` on disk.

Sources (in priority order — most-trusted first):
  1. /classes ✓ exemplar set (user-confirmed in the human-in-loop UI).
     Currently 0 marked (per /control "0 ✓ exemplars all classes").
  2. object_bank/<legacy label>/*.jpg — 600 CWD12 cut-paste crops (already
     vetted as tight-crop = single object). bbox_yolo = [0.5, 0.5, 1.0, 1.0].
  3. (future) Roboflow-approved green annotations pulled back via
     merge_roboflow_projects download-merge.

Output format (one JSON per species, owl_preannotate.py-compatible):
  {
    "species": "SpottedSpurge",
    "vocabulary": "species",
    "exemplars": [
      {"image": "/abs/path.jpg", "bbox_yolo": [0.5, 0.5, 1.0, 1.0]},
      ...
    ]
  }

CLI:
  python -m weed_optimizer_framework.tools.export_owl_exemplars
    [--species SpottedSpurge | --all-species]
    [--per-species 5]
    [--source bank | classes | both]
    [--out-dir results/framework/owl_exemplars]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))

from .cwd12_species import CWD12_SPECIES, cli_species, species_to_legacy

# v3.60.0: exemplar configs are keyed by species (<species>.json, with
# "vocabulary": "species" so owl_preannotate can tell them from the old
# legacy-keyed files). The bank folder of a species depends on the bank's
# vocabulary (synth_cutpaste): object_bank/ was built under the legacy labels,
# so there a species' crops are in object_bank/<species_to_legacy(species)>;
# a bank marked "species" is keyed by the species itself.

LEGACY_BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
BANK_DIR = None      # None: synth_cutpaste.default_bank_dir(); set to override
DEFAULT_OUT_DIR = REPO / "results" / "framework" / "owl_exemplars"


def _bank_dir_for(species: str):
    """(folder holding `species` crops, usable(crop_path)) for the bank in use.
    Without synth_cutpaste's vocabulary helpers the bank is the legacy one."""
    try:
        from . import synth_cutpaste as _sc
        root = BANK_DIR if BANK_DIR is not None else _sc.default_bank_dir()
        vocab = _sc.bank_vocabulary(root)

        def usable(p):
            return _sc.bank_crop_usable(root, p)
    except (ImportError, AttributeError):
        root = BANK_DIR if BANK_DIR is not None else LEGACY_BANK_DIR
        vocab = "legacy"

        def usable(p):
            return True
    folder = species if vocab == "species" else species_to_legacy(species)
    return Path(root) / folder, usable


def _bank_exemplars(species: str, max_n: int) -> list:
    """Read object_bank/<legacy label of species>/*.jpg as exemplars. The crops
    are already tight-cropped so the whole image IS the object;
    bbox_yolo = [0.5, 0.5, 1.0, 1.0] (centered, full extent)."""
    sp_dir, usable = _bank_dir_for(species)
    if not sp_dir.is_dir():
        return []
    out = []
    for p in sorted(sp_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and usable(p):
            out.append({"image": str(p),
                        "bbox_yolo": [0.5, 0.5, 1.0, 1.0]})
        if len(out) >= max_n:
            break
    return out


def _classes_exemplars(species: str, max_n: int) -> list:
    """Read /classes ✓ marks from `results/framework/class_exemplars.json`
    (or wherever the dashboard persists them). Returns same shape.
    Currently 0 marked — placeholder for when human starts approving.
    v3.60.0: only a store that declares "vocabulary": "species" is read. The
    /classes pages grouped boxes by a class-name string that meant different
    plants in different datasets, so a store keyed by those names holds
    mixed species under one key.
    v3.60.0: the dashboard persists verdicts in class_exemplars/<cls>.jsonl
    (each event names its species in "class"); those logs are read first,
    the same way exemplar_to_yolo_dataset replays them, including its holdout
    and NEVER_TRAIN guard. The two single-file stores below are older
    candidates that the dashboard never wrote."""
    out = _logged_exemplars(species, max_n)
    if out:
        return out
    # The dashboard's exemplar state file (per /control 'exemplar set' UI)
    candidates = [
        REPO / "results" / "framework" / "class_exemplars.json",
        REPO / "results" / "framework" / "exemplar_set.json",
    ]
    for cf in candidates:
        if cf.is_file():
            try:
                data = json.load(open(cf))
            except Exception:
                continue
            if not isinstance(data, dict) or data.get("vocabulary") != "species":
                print(f"  [classes] skip {cf.name}: not keyed by species",
                      file=sys.stderr)
                continue
            # Format unknown — try a few shapes
            if species in data:
                items = data[species] or []
            elif "by_species" in data:
                items = (data["by_species"] or {}).get(species, [])
            else:
                continue
            out = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                img = it.get("image") or it.get("path") or it.get("abs_path")
                bbox = it.get("bbox_yolo") or it.get("bbox") or [0.5, 0.5, 1.0, 1.0]
                if img and Path(img).is_file():
                    out.append({"image": str(img), "bbox_yolo": list(bbox)})
                if len(out) >= max_n:
                    break
            return out
    return []


def _logged_exemplars(species: str, max_n: int) -> list:
    """✓ marks for `species` from the dashboard's class_exemplars/*.jsonl, as
    {image, bbox_yolo} (one per box of this species; a bank crop is the whole
    image). Sources that exemplar_to_yolo_dataset refuses are skipped."""
    try:
        from . import exemplar_to_yolo_dataset as _ex
    except Exception as e:
        print(f"  [classes] exemplar logs unreadable: {e!r}", file=sys.stderr)
        return []
    try:
        registry = json.load(open(_ex.REGISTRY_PATH))
    except Exception:
        registry = {}
    events = _ex._read_all_exemplars(registry).get(species, [])
    if not events:
        return []
    guard = _ex._holdout_guard()
    cid = _ex.CWD12_TO_ID[species]
    out: list = []
    for ev in events:
        res, _why = _ex._resolve_source(species, ev.get("img", ""), registry, guard)
        if res is None:
            continue
        kind, src, lines = res
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or parts[0] != str(cid):
                continue
            try:
                bbox = [float(x) for x in parts[1:5]]
            except ValueError:
                continue
            if kind == "bank":
                bbox = [0.5, 0.5, 1.0, 1.0]
            out.append({"image": str(src), "bbox_yolo": bbox})
            if len(out) >= max_n:
                return out
    return out


def export_one(species: str, source: str, per_species: int,
               out_dir: Path) -> dict:
    """Export exemplars for one species. Returns a small status dict."""
    chosen: list = []
    sources_used: list = []
    if source in ("classes", "both"):
        c = _classes_exemplars(species, per_species)
        if c:
            chosen.extend(c)
            sources_used.append(f"classes({len(c)})")
    if (source in ("bank", "both")) and len(chosen) < per_species:
        need = per_species - len(chosen)
        b = _bank_exemplars(species, need)
        if b:
            chosen.extend(b)
            sources_used.append(f"bank({len(b)})")
    cfg = {
        "species": species,
        "vocabulary": "species",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "+".join(sources_used) or "none",
        "exemplars": chosen,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{species}.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, out_path)
    return {"species": species, "n": len(chosen),
            "source": cfg["source"], "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--species", default="", help="single cwd12 species, e.g. SpottedSpurge")
    g.add_argument("--all-species", action="store_true",
                   help="generate for all 12 CWD12 species (default)")
    ap.add_argument("--per-species", type=int, default=5,
                    help="how many exemplars per species (default 5)")
    ap.add_argument("--source", default="bank",
                    choices=["bank", "classes", "both"],
                    help="exemplar source. Default 'bank' (object_bank/).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.species and cli_species(args.species) is None:   # v3.60.0
        print(f"FATAL: {args.species!r} is not a cwd12 species "
              f"({', '.join(CWD12_SPECIES)})", file=sys.stderr)
        sys.exit(2)
    targets = ((cli_species(args.species),) if args.species
               else tuple(CWD12_SPECIES))

    print(f"=== export_owl_exemplars ===")
    print(f"  source: {args.source}")
    print(f"  per-species cap: {args.per_species}")
    print(f"  out-dir: {out_dir}")
    print(f"  species: {len(targets)} ({'all CWD12' if not args.species else args.species})")
    print()
    rows = []
    for sp in targets:
        r = export_one(sp, args.source, args.per_species, out_dir)
        rows.append(r)
        mark = "✓" if r["n"] >= 3 else ("·" if r["n"] > 0 else "✗")
        print(f"  {mark} {sp:20s} n={r['n']:>2} ({r['source']}; "
              f"bank dir {_bank_dir_for(sp)[0].name})")
    n_ok = sum(1 for r in rows if r["n"] >= 3)
    n_thin = sum(1 for r in rows if 0 < r["n"] < 3)
    n_empty = sum(1 for r in rows if r["n"] == 0)
    print()
    print(f"DONE: {n_ok} species ready (≥3 exemplars), "
          f"{n_thin} thin (1-2), {n_empty} empty")
    print(f"      → ready to run owl_preannotate_one for species marked ✓")


if __name__ == "__main__":
    main()
