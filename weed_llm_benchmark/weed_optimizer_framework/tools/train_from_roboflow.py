"""
train_from_roboflow.py — Master-plan TOP PRIORITY: close the Roboflow loop.

Roboflow loop = curated data → Roboflow → human bbox label → pull labeled
ground truth (a Roboflow YOLO export) → **train a REAL YOLO** → eval on the
cwd12 holdout → report mAP. This is the missing closure link (train_yolo was a
stub). It makes Agent-2 real (P4) in service of the Roboflow loop.

Pipeline:
  1. Read a Roboflow YOLO export data.yaml (train+valid).
  2. Remap its labels to the CANONICAL_12 class order (V3_NAMES) by NAME, so the
     trained model's class order matches the holdout eval harness.
  3. Train ultralytics YOLO on the remapped data.
  4. Eval best.pt on cwd12 holdout (test 848 + valid 1129 = 1977, hand-labeled),
     remapping holdout GT to the same order. Report mAP50 / mAP50-95 / per-class.
  5. Write a JSON result.

NEVER_TRAIN safety: the holdout (cottonweeddet12 test/valid) is the EVAL set and
is never in the Roboflow training export. Stem-level guard verifies no holdout
image stem leaked into the training images (aborts if it finds one).

    python -m weed_optimizer_framework.tools.train_from_roboflow \\
        --train-data <roboflow_export>/data.yaml --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()
if not REPO.exists():
    REPO = Path(__file__).resolve().parents[2]

# CANONICAL_12 order (mirrors mega_trainer.CANONICAL_12_NAMES / eval_v3_0_23.V3_NAMES).
# The trained model uses THIS order so the holdout eval aligns.
V3_NAMES = [
    "Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
    "Eclipta", "Goosegrass", "Morningglory", "Nutsedge",
]
CWD12_YAML = REPO / "downloads" / "cottonweeddet12" / "data.yaml"


def _names_list(raw):
    if isinstance(raw, dict):
        return [raw[i] for i in sorted(raw.keys())]
    return list(raw)


def _holdout_stems() -> set:
    stems = set()
    for sub in ("test", "valid"):
        d = REPO / "downloads" / "cottonweeddet12" / sub / "images"
        if d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                for p in d.glob(ext):
                    stems.add(p.stem)
    return stems


def _remap_split(src_img_dir: Path, src_lbl_dir: Path, src_names: list,
                 out_dir: Path, holdout_stems: set) -> tuple:
    """Symlink images + write class-remapped labels (src order → V3_NAMES) into
    out_dir/{images,labels}. Returns (n_imgs, n_lbl, n_leak)."""
    mapping = {i: V3_NAMES.index(nm) for i, nm in enumerate(src_names) if nm in V3_NAMES}
    img_o = out_dir / "images"; lbl_o = out_dir / "labels"
    img_o.mkdir(parents=True, exist_ok=True); lbl_o.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = n_leak = 0
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    if not src_img_dir.is_dir():
        return 0, 0, 0
    for img in sorted(src_img_dir.iterdir()):
        if img.suffix not in exts:
            continue
        if img.stem in holdout_stems:          # NEVER_TRAIN stem guard
            n_leak += 1
            continue
        try:
            os.symlink(img.resolve(), img_o / img.name)
        except FileExistsError:
            pass
        n_img += 1
        lbl = src_lbl_dir / (img.stem + ".txt")
        outl = lbl_o / (img.stem + ".txt")
        if not lbl.is_file():
            outl.touch(); continue
        kept = []
        for line in lbl.read_text().splitlines():
            t = line.split()
            if not t:
                continue
            try:
                c = int(t[0])
            except ValueError:
                continue
            if c in mapping:
                kept.append(f"{mapping[c]} {' '.join(t[1:])}")
        outl.write_text("\n".join(kept))
        if kept:
            n_lbl += 1
    return n_img, n_lbl, n_leak


def build_train_dataset(train_yaml: Path, stage: Path) -> tuple:
    import yaml
    cfg = yaml.safe_load(open(train_yaml))
    src_names = _names_list(cfg["names"])
    base = train_yaml.parent
    holdout = _holdout_stems()

    def resolve(split):
        v = cfg.get(split)
        if not v:
            return None
        p = Path(v) if os.path.isabs(v) else (base / v)
        # roboflow exports: train: ../train/images  OR  train/images
        return p if p.is_dir() else (base / split / "images")

    total_leak = 0
    for split, out in (("train", stage / "train"), ("val", stage / "val"),
                       ("valid", stage / "val")):
        idir = resolve(split)
        if idir is None or not idir.is_dir():
            continue
        ldir = Path(str(idir).replace("/images", "/labels"))
        ni, nl, leak = _remap_split(idir, ldir, src_names, out, holdout)
        total_leak += leak
        print(f"  [{split}] {ni} imgs, {nl} labeled, {leak} holdout-leak-skipped → {out}")
    data_yaml = stage / "data.yaml"
    yaml.safe_dump({"train": str(stage / "train" / "images"),
                    "val": str(stage / "val" / "images"),
                    "nc": len(V3_NAMES), "names": V3_NAMES}, open(data_yaml, "w"))
    return str(data_yaml), total_leak


def build_holdout_yaml(split: str, stage: Path) -> str:
    """Stage cwd12 test/valid with labels remapped to V3_NAMES (reuses the proven
    eval_v3_0_23 approach)."""
    import yaml
    cfg = yaml.safe_load(open(CWD12_YAML))
    src_names = _names_list(cfg["names"])
    mapping = {i: V3_NAMES.index(nm) for i, nm in enumerate(src_names) if nm in V3_NAMES}
    base = CWD12_YAML.parent
    sdir = base / split / "images"
    ldir = base / split / "labels"
    out = stage / ("holdout_" + split)
    img_o = out / "images"; lbl_o = out / "labels"
    img_o.mkdir(parents=True, exist_ok=True); lbl_o.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(sdir.glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            os.symlink(img.resolve(), img_o / img.name)
        except FileExistsError:
            pass
        lbl = ldir / (img.stem + ".txt")
        o = lbl_o / (img.stem + ".txt")
        if not lbl.is_file():
            o.touch(); continue
        kept = []
        for line in lbl.read_text().splitlines():
            t = line.split()
            if t and t[0].isdigit() and int(t[0]) in mapping:
                kept.append(f"{mapping[int(t[0])]} {' '.join(t[1:])}")
        o.write_text("\n".join(kept)); n += 1
    y = out / "data.yaml"
    yaml.safe_dump({"train": str(img_o), "val": str(img_o),
                    "nc": len(V3_NAMES), "names": V3_NAMES}, open(y, "w"))
    print(f"  [holdout_{split}] {n} labeled → {y}")
    return str(y)


def evaluate(model, yaml_path, name, out_dir):
    res = model.val(data=yaml_path, split="val", device=0, save=False,
                    save_json=False, plots=False, verbose=False,
                    project=str(out_dir), name=name + "_run", exist_ok=True)
    b = res.box
    return {"mAP50": float(b.map50), "mAP50_95": float(b.map),
            "precision": float(b.mp), "recall": float(b.mr),
            "per_class_mAP50_95": {V3_NAMES[i]: float(b.maps[i])
                                   for i in range(len(V3_NAMES)) if i < len(b.maps)}}


def main():
    ap = argparse.ArgumentParser(description="Close the Roboflow loop: train + eval")
    ap.add_argument("--train-data", required=True, help="Roboflow export data.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--model", default="yolo11s.pt", help="base weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--out", default=str(REPO / "results" / "framework" / "roboflow_loop"))
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    print("=== CLOSE ROBOFLOW LOOP: train_from_roboflow ===")
    print(f"  train-data: {args.train_data}")
    data_yaml, leak = build_train_dataset(Path(args.train_data), stage)
    if leak:
        print(f"  ⚠️ {leak} holdout-stem images were SKIPPED from training (leak guard)")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  base model: {args.model}  device: {device}  epochs: {args.epochs}")
    model = YOLO(args.model)
    t0 = time.time()
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, project=str(out_dir),
                name="train", patience=20, workers=4, verbose=False, exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"  trained in {time.time()-t0:.0f}s → {best}")

    # Eval on cwd12 holdout (the locked baseline metric)
    ev = YOLO(str(best))
    results = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "train_data": args.train_data, "epochs": args.epochs,
               "base_model": args.model, "best_pt": str(best),
               "holdout_leak_skipped": leak}
    for split in ("test", "valid"):
        hy = build_holdout_yaml(split, stage)
        results[f"cwd12_{split}"] = evaluate(ev, hy, f"cwd12_{split}", out_dir)
        m = results[f"cwd12_{split}"]
        print(f"  >>> cwd12 {split}: mAP50-95={m['mAP50_95']:.4f} mAP50={m['mAP50']:.4f}")

    rj = out_dir / "roboflow_loop_result.json"
    json.dump(results, open(rj, "w"), indent=2, default=str)
    m95 = results.get("cwd12_test", {}).get("mAP50_95", 0)
    gap = round(0.90 - m95, 4)
    print(f"\n=== ROBOFLOW LOOP CLOSED ===")
    print(f"  cwd12 test mAP50-95 = {m95:.4f}  (interim baseline; gap to 0.90 = {gap})")
    print(f"  result → {rj}")


if __name__ == "__main__":
    main()
