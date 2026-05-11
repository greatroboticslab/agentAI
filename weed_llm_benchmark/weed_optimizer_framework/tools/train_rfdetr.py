"""
v3.0.29 Phase 2B — RF-DETR (ICLR 2026, arXiv 2511.09554) on cwd12.

Stages cwd12 train (3,671 imgs, stem-filtered) + cwd12 holdout (1,977 imgs,
NEVER_TRAIN) into RF-DETR's expected COCO JSON layout, then finetunes
RFDETRMedium. Final canonical eval via pycocotools (not ultralytics).

Stem-level holdout filter — identical to v3.0.28 SAFETY staging:
  weedImages contains all 5,648 cwd12 imgs (train + test + valid mixed)
  test/images + valid/images stems = 1,977 holdout (NEVER_TRAIN)
  weedImages minus holdout stems = 3,671 train portion
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]
CWD12_ORIG = ["Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
              "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
              "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge"]
ORIG_TO_CANON = {i: CANONICAL_12.index(n) for i, n in enumerate(CWD12_ORIG)}


def yolo_to_coco_xywh(cx, cy, bw, bh, w, h):
    """Normalized YOLO (cx, cy, bw, bh) → COCO pixel xywh."""
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return [float(x), float(y), float(bw * w), float(bh * h)]


def stage_cwd12_split(
    cwd12_root: Path,
    out_split_dir: Path,
    img_stems_keep: set | None = None,
    img_stems_exclude: set | None = None,
    name_prefix: str = "",
):
    """Stage a cwd12 subset (train OR val) into COCO-formatted directory.

    img_stems_keep: only include images whose stem is in this set (None = all)
    img_stems_exclude: exclude images whose stem is in this set (None = none)
    """
    out_split_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": n, "supercategory": "weed"}
            for i, n in enumerate(CANONICAL_12)
        ],
    }

    weedimg_dir = cwd12_root / "CottonWeedDet12" / "weedImages"
    yolotxt_dir = cwd12_root / "CottonWeedDet12" / "annotation_YOLO_txt"
    img_id = 1
    ann_id = 1
    n_imgs = 0
    n_anns = 0

    src_imgs: list[Path]
    if img_stems_keep is not None:
        # val split — explicit list of stems to include (from test+valid subdirs)
        # but read images from weedImages (which has the full data)
        src_imgs = sorted(
            p for p in weedimg_dir.glob("*.jpg") if p.stem in img_stems_keep
        )
    else:
        src_imgs = sorted(weedimg_dir.glob("*.jpg"))

    for img_path in src_imgs:
        stem = img_path.stem
        if img_stems_exclude is not None and stem in img_stems_exclude:
            continue
        lbl_src = yolotxt_dir / (stem + ".txt")
        if not lbl_src.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        new_name = f"{name_prefix}{stem}.jpg"
        dst_img = out_split_dir / new_name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.symlink(img_path.resolve(), dst_img)

        coco["images"].append({
            "id": img_id,
            "file_name": new_name,
            "width": w,
            "height": h,
        })

        for line in lbl_src.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                orig = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            if orig not in ORIG_TO_CANON:
                continue
            cid = ORIG_TO_CANON[orig]
            xywh = yolo_to_coco_xywh(cx, cy, bw, bh, w, h)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": xywh,
                "area": float(xywh[2] * xywh[3]),
                "iscrowd": 0,
            })
            ann_id += 1
            n_anns += 1
        img_id += 1
        n_imgs += 1

    with open(out_split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    return n_imgs, n_anns


def stage_dataset(out_root: Path, cwd12_root: Path):
    """Build RF-DETR's expected layout: out_root/{train,valid,test}/[_annotations.coco.json + images]"""
    # Build holdout stem set (test + valid)
    holdout_stems: set[str] = set()
    test_stems: set[str] = set()
    valid_stems: set[str] = set()
    for split, target in (("test", test_stems), ("valid", valid_stems)):
        sd = cwd12_root / split / "images"
        if sd.is_dir():
            for img in sd.glob("*.jpg"):
                target.add(img.stem)
                holdout_stems.add(img.stem)
    print(f"[stage] holdout stems: test={len(test_stems)} valid={len(valid_stems)} total={len(holdout_stems)}")

    # Wipe and stage 3 splits
    for sub in ("train", "valid", "test"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # train = weedImages minus holdout stems
    print("[stage] staging TRAIN split (cwd12 train portion)...")
    n_imgs_tr, n_anns_tr = stage_cwd12_split(
        cwd12_root, out_root / "train",
        img_stems_exclude=holdout_stems,
    )
    print(f"[stage]   train: {n_imgs_tr} images, {n_anns_tr} annotations")
    assert n_imgs_tr == 3671, f"expected 3671 cwd12 train imgs, got {n_imgs_tr}"

    # valid = cwd12 valid split
    print("[stage] staging VALID split (cwd12 valid)...")
    n_imgs_va, n_anns_va = stage_cwd12_split(
        cwd12_root, out_root / "valid",
        img_stems_keep=valid_stems, name_prefix="valid__",
    )
    print(f"[stage]   valid: {n_imgs_va} images, {n_anns_va} annotations")
    assert n_imgs_va == 1129, f"expected 1129 cwd12 valid imgs, got {n_imgs_va}"

    # test = cwd12 test split (RF-DETR auto-evals; we also want this for final)
    print("[stage] staging TEST split (cwd12 test)...")
    n_imgs_te, n_anns_te = stage_cwd12_split(
        cwd12_root, out_root / "test",
        img_stems_keep=test_stems, name_prefix="test__",
    )
    print(f"[stage]   test: {n_imgs_te} images, {n_anns_te} annotations")
    assert n_imgs_te == 848, f"expected 848 cwd12 test imgs, got {n_imgs_te}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset+model dir")
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=728,
                    help="must be divisible by 56")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    assert args.resolution % 56 == 0, "RF-DETR resolution must be divisible by 56"

    out_root = Path(args.out).resolve()
    cwd12_root = Path(args.cwd12).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Stage data
    dataset_dir = out_root / "dataset"
    stage_dataset(dataset_dir, cwd12_root)

    # 2. Train
    print("\n[train] importing rfdetr...")
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()
    output_dir = out_root / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] starting RF-DETR Medium finetune")
    print(f"[train]   epochs={args.epochs} batch={args.batch} "
          f"grad_accum={args.grad_accum} (effective={args.batch*args.grad_accum})")
    print(f"[train]   resolution={args.resolution} lr={args.lr}")
    print(f"[train]   dataset_dir={dataset_dir}")
    print(f"[train]   output_dir={output_dir}")

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(output_dir),
        resolution=args.resolution,
        weight_decay=args.weight_decay,
        checkpoint_interval=5,
        tensorboard=True,
    )

    print("\n=== v3.0.29 RF-DETR TRAINING DONE ===")
    # Locate best checkpoint
    best_pt = output_dir / "checkpoint_best_total.pth"
    if not best_pt.exists():
        best_pt = output_dir / "checkpoint_best_ema.pth"
    if not best_pt.exists():
        best_pt = output_dir / "checkpoint_best_regular.pth"
    print(f"best checkpoint: {best_pt}")

    if not best_pt.exists():
        print("!! no best.pth found — training may have failed")
        sys.exit(1)

    # 3. Canonical pycocotools eval on cwd12 test+valid combined
    print("\n[eval] pycocotools canonical eval...")
    eval_canonical(out_root, best_pt, args.resolution)


def eval_canonical(out_root: Path, weights: Path, resolution: int):
    """Run RF-DETR predictions on combined cwd12 test+valid, eval via pycocotools."""
    from rfdetr import RFDETRMedium
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset_dir = out_root / "dataset"
    combined_dir = out_root / "eval_combined"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True)

    # Merge valid + test annotations into one COCO file
    val_coco = json.load(open(dataset_dir / "valid" / "_annotations.coco.json"))
    test_coco = json.load(open(dataset_dir / "test" / "_annotations.coco.json"))
    combined = {
        "images": list(val_coco["images"]),
        "annotations": list(val_coco["annotations"]),
        "categories": val_coco["categories"],
    }
    img_id_offset = max((im["id"] for im in val_coco["images"]), default=0)
    ann_id_offset = max((a["id"] for a in val_coco["annotations"]), default=0)
    img_id_remap = {}
    for im in test_coco["images"]:
        new_id = im["id"] + img_id_offset
        img_id_remap[im["id"]] = new_id
        nim = dict(im); nim["id"] = new_id
        combined["images"].append(nim)
    for a in test_coco["annotations"]:
        new_id = a["id"] + ann_id_offset
        na = dict(a)
        na["id"] = new_id
        na["image_id"] = img_id_remap[a["image_id"]]
        combined["annotations"].append(na)

    # Symlink all images into combined_dir
    file_id_map = {}
    for split in ("valid", "test"):
        for f in (dataset_dir / split).glob("*.jpg"):
            link = combined_dir / f.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(f.resolve(), link)
    for im in combined["images"]:
        file_id_map[im["file_name"]] = im["id"]

    gt_path = out_root / "v3_0_29_rfdetr_combined_gt.json"
    with open(gt_path, "w") as f:
        json.dump(combined, f)

    print(f"[eval] combined: {len(combined['images'])} images, {len(combined['annotations'])} anns")
    print(f"[eval] loading model {weights}")
    model = RFDETRMedium(pretrain_weights=str(weights))

    preds = []
    img_files = sorted(combined_dir.glob("*.jpg"))
    for i, img_path in enumerate(img_files):
        # rfdetr predict returns a Detections-like object with xyxy, confidence, class_id
        try:
            det = model.predict(str(img_path), threshold=0.001)
        except Exception as e:
            print(f"[eval]   pred fail on {img_path.name}: {e}")
            continue
        img_id = file_id_map[img_path.name]
        xyxy = det.xyxy if hasattr(det, "xyxy") else det.bboxes
        conf = det.confidence if hasattr(det, "confidence") else det.scores
        cls = det.class_id if hasattr(det, "class_id") else det.labels
        if xyxy is None or len(xyxy) == 0:
            continue
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            preds.append({
                "image_id": int(img_id),
                "category_id": int(cls[j]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf[j]),
            })
        if (i + 1) % 200 == 0:
            print(f"[eval] predicted {i+1}/{len(img_files)}  preds_total={len(preds)}")
    print(f"[eval] total preds: {len(preds)}")

    pred_path = out_root / "v3_0_29_rfdetr_combined_pred.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    s = coco_eval.stats.tolist()
    summary = {
        "weights": str(weights),
        "resolution": resolution,
        "n_images": len(combined["images"]),
        "n_annotations": len(combined["annotations"]),
        "n_predictions": len(preds),
        "mAP50_95": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
    }
    print(f"\n=== RF-DETR pycocotools canonical ===")
    print(f"  mAP50-95: {s[0]:.4f}")
    print(f"  mAP50:    {s[1]:.4f}")
    print(f"  mAP75:    {s[2]:.4f}")
    out_path = out_root / "v3_0_29_rfdetr_pycoco_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
