#!/usr/bin/env python3
"""
Leave-4-Out Cross-Species Generalization Experiment

Tests whether LLMs generalize better than YOLO to unseen weed species.

Method:
  - Hold out 4 species from CottonWeedDet12: Morningglory(4), Goosegrass(3),
    Eclipta(2), Nutsedge(5) (YOLO's weakest species)
  - Train YOLO on remaining 8 species only
  - Test both YOLO and LLMs on images containing held-out species
  - Compare zero-shot detection ability on unseen species

Usage:
    python run_leave4out.py --all
    python run_leave4out.py --step 1   # Create split
    python run_leave4out.py --step 2   # Train YOLO on 8 species
    python run_leave4out.py --step 3   # Evaluate YOLO on held-out species
    python run_leave4out.py --step 4   # Compare with LLMs
    python run_leave4out.py --step 5   # Augment + retrain
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results", "leave4out")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
CW_DIR = os.path.join(DOWNLOAD_DIR, "cottonweeddet12")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")

os.makedirs(RESULT_DIR, exist_ok=True)

# CottonWeedDet12 class mapping
ALL_CLASSES = {
    0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
    4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
    8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
}

# Hold out YOLO's 4 weakest species
HOLDOUT_IDS = {2, 3, 4, 5}  # Eclipta, Goosegrass, Morningglory, Nutsedge
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}  # remaining 8

HOLDOUT_NAMES = [ALL_CLASSES[i] for i in sorted(HOLDOUT_IDS)]
TRAIN_NAMES = [ALL_CLASSES[i] for i in sorted(TRAIN_IDS)]


def compute_iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0


# ============================================================
# Step 1: Create species-split dataset
# ============================================================
def step1_create_split():
    """Split CottonWeedDet12 by species for leave-4-out experiment."""
    print("\n" + "=" * 70)
    print("STEP 1: Create Leave-4-Out dataset split")
    print(f"  Hold-out species: {HOLDOUT_NAMES}")
    print(f"  Training species: {TRAIN_NAMES}")
    print("=" * 70)

    split_dir = os.path.join(RESULT_DIR, "dataset_8species")
    holdout_dir = os.path.join(RESULT_DIR, "dataset_holdout")

    for d in [split_dir, holdout_dir]:
        for sub in ["train/images", "train/labels", "valid/images", "valid/labels",
                     "test/images", "test/labels"]:
            os.makedirs(os.path.join(d, sub), exist_ok=True)

    stats = {"train_8sp": 0, "holdout_train": 0, "holdout_test": 0}

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(CW_DIR, split, "images")
        lbl_dir = os.path.join(CW_DIR, split, "labels")

        if not os.path.isdir(img_dir):
            print(f"[!] {img_dir} not found")
            continue

        for lbl_file in sorted(os.listdir(lbl_dir)):
            if not lbl_file.endswith(".txt"):
                continue
            stem = lbl_file.replace(".txt", "")

            # Read label file, check which species are present
            with open(os.path.join(lbl_dir, lbl_file)) as f:
                lines = [l.strip() for l in f if l.strip()]

            species_in_image = set()
            for line in lines:
                parts = line.split()
                if parts:
                    species_in_image.add(int(parts[0]))

            has_holdout = bool(species_in_image & HOLDOUT_IDS)
            has_train = bool(species_in_image & TRAIN_IDS)

            # Find image file
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_file = candidate
                    break

            if not img_file:
                continue

            if has_holdout:
                # Image contains holdout species → goes to holdout set
                # Keep ALL labels (both holdout and train species) for evaluation
                dest = holdout_dir
                if split == "test":
                    stats["holdout_test"] += 1
                else:
                    stats["holdout_train"] += 1
                shutil.copy2(img_file, os.path.join(dest, split, "images", os.path.basename(img_file)))
                shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(dest, split, "labels", lbl_file))

            if has_train and not has_holdout:
                # Image ONLY has training species → goes to 8-species training set
                dest = split_dir
                stats["train_8sp"] += 1

                # Remap class IDs: keep only train species, remap to 0-7
                remap = {old: new for new, old in enumerate(sorted(TRAIN_IDS))}
                new_lines = []
                for line in lines:
                    parts = line.split()
                    cls = int(parts[0])
                    if cls in TRAIN_IDS:
                        parts[0] = str(remap[cls])
                        new_lines.append(" ".join(parts))

                shutil.copy2(img_file, os.path.join(dest, split, "images", os.path.basename(img_file)))
                with open(os.path.join(dest, split, "labels", lbl_file), "w") as f:
                    f.write("\n".join(new_lines) + "\n")

    # Create data.yaml for 8-species training
    yaml_path = os.path.join(split_dir, "data.yaml")
    remap = {old: new for new, old in enumerate(sorted(TRAIN_IDS))}
    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)]
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(split_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(split_dir, 'valid', 'images')}\n")
        f.write(f"nc: 8\n")
        f.write(f"names: {names}\n")

    print(f"\n  8-species training images: {stats['train_8sp']}")
    print(f"  Holdout test images: {stats['holdout_test']}")
    print(f"  Holdout train images (for augmentation): {stats['holdout_train']}")
    print(f"  data.yaml: {yaml_path}")

    # Save stats
    with open(os.path.join(RESULT_DIR, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return split_dir, holdout_dir


# ============================================================
# Step 2: Train YOLO on 8 species
# ============================================================
def step2_train_yolo():
    """Train YOLO on 8-species subset."""
    print("\n" + "=" * 70)
    print("STEP 2: Train YOLO on 8 species (excluding hold-out 4)")
    print("=" * 70)

    split_dir = os.path.join(RESULT_DIR, "dataset_8species")
    yaml_path = os.path.join(split_dir, "data.yaml")

    if not os.path.exists(yaml_path):
        print("[!] No data.yaml found. Run step 1 first.")
        return None

    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on {device}...")

    model = YOLO("yolo11n.pt")
    model.train(
        data=yaml_path,
        epochs=100,
        batch=-1,
        device=device,
        project=os.path.join(RESULT_DIR, "yolo_8species"),
        name="train",
        patience=20,
        verbose=True,
    )

    best_path = os.path.join(RESULT_DIR, "yolo_8species", "train", "weights", "best.pt")
    print(f"[+] Model saved to {best_path}")
    return best_path


# ============================================================
# Step 3: Evaluate YOLO on held-out species
# ============================================================
def step3_eval_yolo():
    """Test YOLO (8-species) on images with held-out species."""
    print("\n" + "=" * 70)
    print("STEP 3: Evaluate YOLO on held-out species (zero-shot)")
    print("=" * 70)

    best_path = os.path.join(RESULT_DIR, "yolo_8species", "train", "weights", "best.pt")
    holdout_dir = os.path.join(RESULT_DIR, "dataset_holdout")
    holdout_test_imgs = os.path.join(holdout_dir, "test", "images")
    holdout_test_lbls = os.path.join(holdout_dir, "test", "labels")

    if not os.path.exists(best_path):
        print("[!] No YOLO model found. Run step 2 first.")
        return None

    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(best_path)

    yolo_pred_dir = os.path.join(RESULT_DIR, "yolo_8sp_preds_holdout")
    os.makedirs(yolo_pred_dir, exist_ok=True)

    test_images = sorted([f for f in os.listdir(holdout_test_imgs)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[*] Running YOLO on {len(test_images)} holdout test images...")

    for i, img_file in enumerate(test_images):
        results = model.predict(os.path.join(holdout_test_imgs, img_file),
                                conf=0.25, device=device, save=False, verbose=False)
        stem = Path(img_file).stem
        lines = []
        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")
        with open(os.path.join(yolo_pred_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    # Evaluate (binary: any detection = weed)
    result = evaluate_binary(
        "YOLO (8-species, zero-shot on holdout)",
        yolo_pred_dir, holdout_test_lbls, holdout_ids=HOLDOUT_IDS
    )
    return result


def evaluate_binary(model_name, pred_dir, gt_dir, holdout_ids=None):
    """Binary evaluation: any weed detection counts as TP if IoU>=0.5 with any GT box."""
    total_tp, total_fp, total_fn = 0, 0, 0
    per_class = defaultdict(lambda: {"tp": 0, "fn": 0})

    for gt_file in sorted(os.listdir(gt_dir)):
        if not gt_file.endswith(".txt"):
            continue
        stem = gt_file.replace(".txt", "")

        gt_boxes = []
        with open(os.path.join(gt_dir, gt_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = [float(x) for x in parts[1:5]]
                    gt_boxes.append((cls, cx, cy, w, h))

        pred_file = os.path.join(pred_dir, f"{stem}.txt")
        pred_boxes = []
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        conf = float(parts[5]) if len(parts) >= 6 else 0.5
                        pred_boxes.append((0, cx, cy, w, h, conf))

        matched_gt = set()
        sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1][-1], reverse=True)
        tp, fp = 0, 0

        for pi, pred in sorted_preds:
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred[1:5], gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1
                per_class[gt_boxes[best_gi][0]]["tp"] += 1
            else:
                fp += 1

        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                per_class[gt[0]]["fn"] += 1

        total_tp += tp
        total_fp += fp
        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        "model": model_name,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "per_class": {},
    }

    print(f"\n  {model_name}:")
    print(f"    Overall: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"    Per-class (holdout species):")
    for cls_id in sorted(per_class.keys()):
        s = per_class[cls_id]
        cls_recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        name = ALL_CLASSES.get(cls_id, f"class_{cls_id}")
        is_holdout = cls_id in HOLDOUT_IDS if holdout_ids else False
        marker = " ← UNSEEN" if is_holdout else ""
        result["per_class"][name] = {"tp": s["tp"], "fn": s["fn"], "recall": round(cls_recall, 4)}
        print(f"      {name:20s}: recall={cls_recall:.3f} (TP={s['tp']}, FN={s['fn']}){marker}")

    return result


# ============================================================
# Step 4: Compare with LLMs on holdout species
# ============================================================
def step4_compare_llms():
    """Run evaluation of existing LLM predictions on holdout test images."""
    print("\n" + "=" * 70)
    print("STEP 4: Compare LLMs on held-out species")
    print("=" * 70)

    holdout_test_lbls = os.path.join(RESULT_DIR, "dataset_holdout", "test", "labels")
    holdout_test_imgs = os.path.join(RESULT_DIR, "dataset_holdout", "test", "images")

    if not os.path.isdir(holdout_test_lbls):
        print("[!] No holdout test labels. Run step 1 first.")
        return

    # Get list of holdout test image stems
    holdout_stems = set()
    for f in os.listdir(holdout_test_lbls):
        if f.endswith(".txt"):
            holdout_stems.add(f.replace(".txt", ""))

    print(f"[*] {len(holdout_stems)} holdout test images")

    all_results = {}

    # Evaluate YOLO (8-species)
    yolo_pred_dir = os.path.join(RESULT_DIR, "yolo_8sp_preds_holdout")
    if os.path.isdir(yolo_pred_dir):
        all_results["yolo_8species"] = evaluate_binary(
            "YOLO (8-species)", yolo_pred_dir, holdout_test_lbls, HOLDOUT_IDS)

    # Evaluate YOLO (12-species, full fine-tuned) for reference
    yolo_12sp_dir = os.path.join(BASE_DIR, "runs", "detect", "runs",
                                  "yolo11n_cottonweeddet12", "weights", "best.pt")
    if os.path.exists(yolo_12sp_dir):
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_12 = YOLO(yolo_12sp_dir)

        pred_12_dir = os.path.join(RESULT_DIR, "yolo_12sp_preds_holdout")
        os.makedirs(pred_12_dir, exist_ok=True)

        for stem in holdout_stems:
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                p = os.path.join(holdout_test_imgs, stem + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if not img_path:
                continue

            results = model_12.predict(img_path, conf=0.25, device=device, save=False, verbose=False)
            lines = []
            for r in results:
                for box in r.boxes:
                    x, y, w, h = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")
            with open(os.path.join(pred_12_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")

        all_results["yolo_12species"] = evaluate_binary(
            "YOLO (12-species, full)", pred_12_dir, holdout_test_lbls, HOLDOUT_IDS)

    # Evaluate LLMs (use their existing CottonWeedDet12 predictions, filter to holdout images)
    llm_keys = ["florence2_base", "florence2", "owlv2", "internvl2", "qwen3b", "minicpm_v45", "qwen7b"]
    for llm_key in llm_keys:
        llm_label_dir = os.path.join(LABELED_DIR, f"{llm_key}_cottonweeddet12", "detected", "labels")
        if not os.path.isdir(llm_label_dir):
            continue

        # Create filtered prediction dir with only holdout images
        filtered_dir = os.path.join(RESULT_DIR, f"{llm_key}_preds_holdout")
        os.makedirs(filtered_dir, exist_ok=True)

        for stem in holdout_stems:
            src = os.path.join(llm_label_dir, f"{stem}.txt")
            dst = os.path.join(filtered_dir, f"{stem}.txt")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                with open(dst, "w") as f:
                    f.write("")

        all_results[llm_key] = evaluate_binary(
            f"{llm_key} (zero-shot)", filtered_dir, holdout_test_lbls, HOLDOUT_IDS)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Detection on Unseen Species")
    print("=" * 70)
    print(f"{'Model':<35s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 55)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{r['model']:<35s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}")

    # Save
    out_path = os.path.join(RESULT_DIR, "leave4out_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")
    return all_results


# ============================================================
# Step 5: Augment + Retrain
# ============================================================
def step5_augment_retrain():
    """Use LLM pseudo-labels on holdout species to augment YOLO training."""
    print("\n" + "=" * 70)
    print("STEP 5: LLM-Augmented YOLO Re-training")
    print("=" * 70)

    # Create augmented dataset: 8-species training + LLM-labeled holdout images
    aug_dir = os.path.join(RESULT_DIR, "dataset_augmented")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(aug_dir, sub), exist_ok=True)

    # Copy 8-species training data
    sp8_dir = os.path.join(RESULT_DIR, "dataset_8species")
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

    # Add holdout train images with LLM pseudo-labels (use florence2_base as best LLM)
    holdout_train_imgs = os.path.join(RESULT_DIR, "dataset_holdout", "train", "images")
    florence_labels = os.path.join(LABELED_DIR, "florence2_base_cottonweeddet12", "detected", "labels")

    count_aug = 0
    if os.path.isdir(holdout_train_imgs) and os.path.isdir(florence_labels):
        for img_file in os.listdir(holdout_train_imgs):
            stem = Path(img_file).stem
            florence_lbl = os.path.join(florence_labels, f"{stem}.txt")

            if os.path.exists(florence_lbl):
                with open(florence_lbl) as f:
                    lines = [l.strip() for l in f if l.strip()]
                if lines:
                    # Remap to class 8 (new: "novel_weed_llm")
                    new_lines = []
                    for line in lines:
                        parts = line.split()
                        parts[0] = "8"
                        new_lines.append(" ".join(parts[:5]))

                    shutil.copy2(
                        os.path.join(holdout_train_imgs, img_file),
                        os.path.join(aug_dir, "train", "images", f"aug_{img_file}"))
                    with open(os.path.join(aug_dir, "train", "labels", f"aug_{stem}.txt"), "w") as f:
                        f.write("\n".join(new_lines) + "\n")
                    count_aug += 1

    print(f"  Original 8-species images: {count_orig}")
    print(f"  LLM-augmented holdout images: {count_aug}")
    print(f"  Total augmented training: {count_orig + count_aug}")

    # Create data.yaml
    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed_llm"]
    yaml_path = os.path.join(aug_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(aug_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(aug_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\n")
        f.write(f"names: {names}\n")

    # Re-train YOLO
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")
    model.train(
        data=yaml_path,
        epochs=100,
        batch=-1,
        device=device,
        project=os.path.join(RESULT_DIR, "yolo_augmented"),
        name="train",
        patience=20,
        verbose=True,
    )

    # Evaluate augmented model on holdout test
    best_path = os.path.join(RESULT_DIR, "yolo_augmented", "train", "weights", "best.pt")
    holdout_test_imgs = os.path.join(RESULT_DIR, "dataset_holdout", "test", "images")
    holdout_test_lbls = os.path.join(RESULT_DIR, "dataset_holdout", "test", "labels")

    aug_model = YOLO(best_path)
    aug_pred_dir = os.path.join(RESULT_DIR, "yolo_aug_preds_holdout")
    os.makedirs(aug_pred_dir, exist_ok=True)

    for img_file in os.listdir(holdout_test_imgs):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        results = aug_model.predict(os.path.join(holdout_test_imgs, img_file),
                                     conf=0.25, device=device, save=False, verbose=False)
        stem = Path(img_file).stem
        lines = []
        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")
        with open(os.path.join(aug_pred_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    aug_result = evaluate_binary(
        "YOLO (8sp + LLM augmented)", aug_pred_dir, holdout_test_lbls, HOLDOUT_IDS)

    # Compare
    results_path = os.path.join(RESULT_DIR, "leave4out_results.json")
    if os.path.exists(results_path):
        all_results = json.load(open(results_path))
        all_results["yolo_augmented"] = aug_result
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n  === IMPROVEMENT FROM LLM AUGMENTATION ===")
    y8_path = os.path.join(RESULT_DIR, "leave4out_results.json")
    if os.path.exists(y8_path):
        d = json.load(open(y8_path))
        if "yolo_8species" in d:
            y8 = d["yolo_8species"]
            print(f"  YOLO (8-species only):        F1={y8['f1']:.3f}")
            print(f"  YOLO (8sp + LLM augmented):   F1={aug_result['f1']:.3f}")
            print(f"  Delta:                        {aug_result['f1'] - y8['f1']:+.3f}")

    return aug_result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Leave-4-Out Cross-Species Experiment")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    if not args.all and not args.step:
        args.all = True

    if args.all or args.step == 1:
        step1_create_split()
    if args.all or args.step == 2:
        step2_train_yolo()
    if args.all or args.step == 3:
        step3_eval_yolo()
    if args.all or args.step == 4:
        step4_compare_llms()
    if args.all or args.step == 5:
        step5_augment_retrain()

    print("\n" + "=" * 70)
    print("[+] LEAVE-4-OUT EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
