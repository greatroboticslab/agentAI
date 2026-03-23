#!/usr/bin/env python3
"""
Background-Aware Label Propagation for Weed Detection (BA-LPW)

Solves catastrophic forgetting by creating COMPLETE annotations for new-species images:
  - Old species: labeled by the existing YOLO model
  - New species: labeled by Florence-2 VLM
  - Merge → every object in every image is properly labeled
  - YOLO never learns "old species = background"

Also tests replay buffer (mixing old training data) and lower learning rate
as additional anti-forgetting measures.

Compares:
  1. YOLO 8sp only (baseline)
  2. YOLO 8sp + naive LLM aug (previous result, -2.4% forgetting)
  3. YOLO 8sp + BA-LPW (complete labels, should fix forgetting)
  4. YOLO 8sp + BA-LPW + replay (complete labels + old data mixed in)
"""
import json, os, shutil, time
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
RESULT_DIR = os.path.join(BASE_DIR, "results", "balpw")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")

os.makedirs(RESULT_DIR, exist_ok=True)

# Class mapping
ALL_CLASSES = {
    0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
    4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
    8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
}
HOLDOUT_IDS = {2, 3, 4, 5}
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}
REMAP_TRAIN = {old: new for new, old in enumerate(sorted(TRAIN_IDS))}  # old→0-7


def compute_iou(box1, box2):
    x1_1, y1_1 = box1[0]-box1[2]/2, box1[1]-box1[3]/2
    x2_1, y2_1 = box1[0]+box1[2]/2, box1[1]+box1[3]/2
    x1_2, y1_2 = box2[0]-box2[2]/2, box2[1]-box2[3]/2
    x2_2, y2_2 = box2[0]+box2[2]/2, box2[1]+box2[3]/2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return inter/union if union > 0 else 0


def evaluate_binary(model_path, test_imgs, test_lbls, label):
    """Evaluate YOLO on test images with binary weed detection."""
    from ultralytics import YOLO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    total_tp, total_fp, total_fn = 0, 0, 0
    imgs = sorted([f for f in os.listdir(test_imgs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in imgs:
        stem = Path(img_file).stem
        lbl_path = os.path.join(test_lbls, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    gt_boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        results = model.predict(os.path.join(test_imgs, img_file), conf=0.25, device=device, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                pred_boxes.append((x, y, w, h, conf))

        pred_boxes.sort(key=lambda x: x[4], reverse=True)
        matched = set()
        tp = 0
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched:
                    continue
                iou = compute_iou(pb[:4], gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi)
                tp += 1

        total_tp += tp
        total_fp += len(pred_boxes) - tp
        total_fn += len(gt_boxes) - len(matched)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  {label}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={total_tp} FP={total_fp} FN={total_fn})")
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn}


def step1_create_balpw_labels():
    """Create complete labels for holdout images using YOLO + Florence-2."""
    print("\n" + "=" * 70)
    print("STEP 1: Background-Aware Label Propagation")
    print("  Use YOLO to label OLD species + Florence-2 to label NEW species")
    print("=" * 70)

    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_8sp = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")
    model = YOLO(yolo_8sp)

    holdout_train_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    florence_labels = os.path.join(LABELED_DIR, "florence2_base_cottonweeddet12", "detected", "labels")

    balpw_dir = os.path.join(RESULT_DIR, "balpw_labels")
    os.makedirs(balpw_dir, exist_ok=True)

    stats = {"total_images": 0, "yolo_old_detections": 0, "florence_new_detections": 0}

    imgs = sorted([f for f in os.listdir(holdout_train_imgs)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in imgs:
        stem = Path(img_file).stem
        img_path = os.path.join(holdout_train_imgs, img_file)

        # 1. YOLO detects OLD species (classes 0-7 in 8sp model)
        results = model.predict(img_path, conf=0.25, device=device, verbose=False)
        yolo_lines = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # 0-7 in 8sp model
                x, y, w, h = box.xywhn[0].tolist()
                yolo_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                stats["yolo_old_detections"] += 1

        # 2. Florence-2 detects NEW species (stored as class 0 in florence labels)
        florence_lines = []
        florence_lbl = os.path.join(florence_labels, f"{stem}.txt")
        if os.path.exists(florence_lbl):
            with open(florence_lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Remap to class 8 = "new_species"
                        florence_lines.append(f"8 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                        stats["florence_new_detections"] += 1

        # 3. Merge: remove Florence detections that overlap with YOLO (avoid duplicates)
        merged = list(yolo_lines)
        for fl in florence_lines:
            fp = fl.split()
            f_box = (float(fp[1]), float(fp[2]), float(fp[3]), float(fp[4]))
            is_duplicate = False
            for yl in yolo_lines:
                yp = yl.split()
                y_box = (float(yp[1]), float(yp[2]), float(yp[3]), float(yp[4]))
                if compute_iou(f_box, y_box) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(fl)

        # Save merged labels
        with open(os.path.join(balpw_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(merged) + "\n" if merged else "")

        stats["total_images"] += 1

    print(f"  Images processed: {stats['total_images']}")
    print(f"  YOLO old-species detections: {stats['yolo_old_detections']}")
    print(f"  Florence new-species detections: {stats['florence_new_detections']}")
    return balpw_dir, stats


def step2_train_balpw():
    """Train YOLO with BA-LPW complete labels."""
    print("\n" + "=" * 70)
    print("STEP 2: Train YOLO with BA-LPW labels")
    print("=" * 70)

    from ultralytics import YOLO
    import torch

    # Create training dataset
    aug_dir = os.path.join(RESULT_DIR, "dataset_balpw")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(aug_dir, sub), exist_ok=True)

    # Copy 8-species training data
    sp8_dir = os.path.join(L4O_DIR, "dataset_8species")
    count_orig = 0
    for split in ["train", "valid"]:
        src_imgs = os.path.join(sp8_dir, split, "images")
        src_lbls = os.path.join(sp8_dir, split, "labels")
        if not os.path.isdir(src_imgs):
            continue
        for f in os.listdir(src_imgs):
            shutil.copy2(os.path.join(src_imgs, f), os.path.join(aug_dir, split, "images", f))
            stem = Path(f).stem
            lbl = os.path.join(src_lbls, f"{stem}.txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(aug_dir, split, "labels", f"{stem}.txt"))
                count_orig += 1

    # Add holdout images with BA-LPW complete labels
    holdout_train_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    balpw_labels = os.path.join(RESULT_DIR, "balpw_labels")

    count_balpw = 0
    for lbl_file in os.listdir(balpw_labels):
        if not lbl_file.endswith(".txt"):
            continue
        stem = lbl_file.replace(".txt", "")
        # Check if label has content
        with open(os.path.join(balpw_labels, lbl_file)) as f:
            if not f.read().strip():
                continue
        # Find image
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            img_path = os.path.join(holdout_train_imgs, stem + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(aug_dir, "train", "images", f"balpw_{stem}{ext}"))
                shutil.copy2(os.path.join(balpw_labels, lbl_file),
                             os.path.join(aug_dir, "train", "labels", f"balpw_{stem}.txt"))
                count_balpw += 1
                break

    print(f"  Original 8-species images: {count_orig}")
    print(f"  BA-LPW labeled images: {count_balpw}")

    # Create data.yaml (9 classes: 8 original + 1 new)
    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed"]
    yaml_path = os.path.join(aug_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(aug_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(aug_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\n")
        f.write(f"names: {names}\n")

    # Train — fine-tune from 8sp weights (not from scratch) with lower lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_8sp = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")
    model = YOLO(yolo_8sp)

    print(f"[*] Training on {device} from 8sp weights (fine-tune, lower lr)...")
    model.train(
        data=yaml_path,
        epochs=50,
        batch=-1,
        device=device,
        project=os.path.join(RESULT_DIR, "yolo_balpw"),
        name="train",
        patience=15,
        lr0=0.001,  # lower lr to preserve old knowledge
        verbose=True,
    )

    best_path = os.path.join(RESULT_DIR, "yolo_balpw", "train", "weights", "best.pt")
    print(f"[+] Model saved to {best_path}")
    return best_path


def step3_evaluate():
    """Compare all approaches on both old and new species."""
    print("\n" + "=" * 70)
    print("STEP 3: Comprehensive Evaluation")
    print("=" * 70)

    holdout_test_imgs = os.path.join(L4O_DIR, "dataset_holdout", "test", "images")
    holdout_test_lbls = os.path.join(L4O_DIR, "dataset_holdout", "test", "labels")
    sp8_test_imgs = os.path.join(L4O_DIR, "dataset_8species", "test", "images")
    sp8_test_lbls = os.path.join(L4O_DIR, "dataset_8species", "test", "labels")

    results = {}

    models = {
        "yolo_8sp": os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt"),
        "yolo_naive_aug": os.path.join(L4O_DIR, "yolo_augmented", "train", "weights", "best.pt"),
        "yolo_balpw": os.path.join(RESULT_DIR, "yolo_balpw", "train", "weights", "best.pt"),
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"  [!] {name}: model not found")
            continue

        print(f"\n--- {name} ---")
        results[name] = {}

        # Test on OLD species (8 training species)
        print(f"  On ORIGINAL 8 species:")
        results[name]["old_species"] = evaluate_binary(path, sp8_test_imgs, sp8_test_lbls, f"  {name} (old)")

        # Test on NEW species (4 holdout species)
        print(f"  On UNSEEN 4 species:")
        results[name]["new_species"] = evaluate_binary(path, holdout_test_imgs, holdout_test_lbls, f"  {name} (new)")

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25s} {'Old 8sp F1':>12s} {'New 4sp F1':>12s} {'Old Δ':>8s} {'New Δ':>8s}")
    print("-" * 65)

    baseline_old = results.get("yolo_8sp", {}).get("old_species", {}).get("f1", 0)
    baseline_new = results.get("yolo_8sp", {}).get("new_species", {}).get("f1", 0)

    for name in ["yolo_8sp", "yolo_naive_aug", "yolo_balpw"]:
        if name not in results:
            continue
        old_f1 = results[name]["old_species"]["f1"]
        new_f1 = results[name]["new_species"]["f1"]
        d_old = old_f1 - baseline_old
        d_new = new_f1 - baseline_new
        print(f"{name:<25s} {old_f1:12.3f} {new_f1:12.3f} {d_old:+8.3f} {d_new:+8.3f}")

    # Save
    out_path = os.path.join(RESULT_DIR, "balpw_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")
    return results


def main():
    print("=" * 70)
    print("BA-LPW: Background-Aware Label Propagation for Weed Detection")
    print("=" * 70)

    balpw_dir, stats = step1_create_balpw_labels()
    best_path = step2_train_balpw()
    results = step3_evaluate()

    print("\n" + "=" * 70)
    print("[+] BA-LPW EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
