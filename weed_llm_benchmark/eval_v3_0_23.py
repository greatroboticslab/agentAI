"""
v3.0.23 evaluation on clean human-labeled holdouts.

Loads `mega_iter6/train8/weights/best.pt` and runs `model.val()` against:
  A. cottonweeddet12 test split (848 imgs, in-distribution)
  B. cottonweeddet12 valid split (1129 imgs, also in-distribution but disjoint)

Class IDs differ between original cottonweeddet12 and our v3.0.23 model
(both have 12 classes, but ordering differs). Script remaps the test
labels in-place to v3.0.23 order before running val. Outputs JSON with
mAP50, mAP50-95, P, R, per-class mAP50-95.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
BEST_PT = f"{REPO}/results/framework/mega_iter6/train8/weights/best.pt"
OUT = f"{REPO}/results/v3_0_23_eval"
os.makedirs(OUT, exist_ok=True)

# v3.0.23 class order from merged_iter6/data.yaml
V3_NAMES = [
    "Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
    "Eclipta", "Goosegrass", "Morningglory", "Nutsedge",
]


def build_eval_dataset(name, src_yaml_path, split_field):
    """Stage a copy of `split_field` images + remapped labels under OUT/{name}/."""
    src_yaml = yaml.safe_load(open(src_yaml_path))
    raw_names = src_yaml["names"]
    if isinstance(raw_names, dict):
        src_names = [raw_names[i] for i in sorted(raw_names.keys())]
    else:
        src_names = list(raw_names)

    mapping = {}
    dropped = []
    for i, nm in enumerate(src_names):
        if nm in V3_NAMES:
            mapping[i] = V3_NAMES.index(nm)
        else:
            dropped.append(nm)
    if dropped:
        print(f"  [{name}] WARN: classes dropped (not in v3.0.23): {dropped}")

    src_dir = Path(src_yaml_path).parent
    split_val = src_yaml.get(split_field, split_field)
    src_split_dir = Path(split_val) if os.path.isabs(split_val) else src_dir / split_val
    if not src_split_dir.exists():
        # cottonweeddet12 layout: split is a top-level dir (test/ or valid/) in src_dir
        cand = src_dir / split_field
        if cand.exists():
            src_split_dir = cand
        else:
            print(f"  [{name}] FATAL: split dir not found: {src_split_dir} or {cand}")
            sys.exit(2)

    out_dir = Path(OUT) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    images = sorted(
        list(src_split_dir.rglob("*.jpg"))
        + list(src_split_dir.rglob("*.png"))
        + list(src_split_dir.rglob("*.jpeg"))
    )

    n_lbl = 0
    for img in images:
        lbl_candidates = [img.with_suffix(".txt")]
        if not lbl_candidates[0].exists():
            same_stem = list(src_split_dir.rglob(img.stem + ".txt"))
            lbl_candidates = same_stem
        lbl = lbl_candidates[0] if lbl_candidates and lbl_candidates[0].exists() else None

        dst_img = img_dir / img.name
        try:
            os.symlink(img.resolve(), dst_img)
        except FileExistsError:
            pass

        if lbl is None:
            (lbl_dir / (img.stem + ".txt")).touch()
            continue

        with open(lbl) as f:
            lines = f.readlines()
        with open(lbl_dir / (img.stem + ".txt"), "w") as g:
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    orig = int(parts[0])
                except ValueError:
                    continue
                if orig in mapping:
                    g.write(f"{mapping[orig]} {' '.join(parts[1:])}\n")
        n_lbl += 1

    yaml_out = out_dir / "data.yaml"
    yaml.safe_dump(
        {
            "train": str(img_dir),
            "val": str(img_dir),
            "nc": len(V3_NAMES),
            "names": V3_NAMES,
        },
        open(yaml_out, "w"),
    )
    print(f"  [{name}] {len(images)} imgs, {n_lbl} non-empty label files → {yaml_out}")
    return str(yaml_out)


def run_val(yaml_path, name, model):
    print(f"\n=== EVAL: {name} ===")
    res = model.val(
        data=yaml_path,
        split="val",
        device=0,
        save=False,
        save_json=False,
        plots=False,
        verbose=True,
        project=OUT,
        name=name + "_run",
        exist_ok=True,
    )
    box = res.box
    metrics = {
        "n_classes_with_data": int(box.nc),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_mAP50_95": {
            V3_NAMES[i]: float(box.maps[i]) for i in range(len(V3_NAMES)) if i < len(box.maps)
        },
    }
    print(f"  mAP50 = {metrics['mAP50']:.4f}, mAP50-95 = {metrics['mAP50_95']:.4f}")
    print(f"  P = {metrics['precision']:.4f}, R = {metrics['recall']:.4f}")
    return metrics


def main():
    if not os.path.exists(BEST_PT):
        print(f"FATAL: best.pt not found at {BEST_PT}")
        sys.exit(2)

    model = YOLO(BEST_PT)
    print(f"Loaded best.pt: {BEST_PT}")

    cwd12_yaml = f"{REPO}/downloads/cottonweeddet12/data.yaml"
    if not os.path.exists(cwd12_yaml):
        print(f"FATAL: {cwd12_yaml} not found")
        sys.exit(2)

    out = {}

    yaml_test = build_eval_dataset("cwd12_test", cwd12_yaml, "test")
    out["cwd12_test"] = run_val(yaml_test, "cwd12_test", model)

    yaml_valid = build_eval_dataset("cwd12_valid", cwd12_yaml, "valid")
    out["cwd12_valid"] = run_val(yaml_valid, "cwd12_valid", model)

    json_path = f"{OUT}/v3_0_23_eval.json"
    json.dump(out, open(json_path, "w"), indent=2)
    print(f"\n=== ALL DONE ===\nMetrics → {json_path}")


if __name__ == "__main__":
    main()
