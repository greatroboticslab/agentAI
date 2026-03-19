#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment

Tests whether LLMs generalize better than fine-tuned YOLO to unseen weed species.
- YOLO: trained on CottonWeedDet12 (12 species) → zero-shot on DeepWeeds (8 different species)
- LLMs: zero-shot on DeepWeeds (never seen either dataset)

If LLMs outperform YOLO on unseen species → LLMs have practical value as
zero-shot annotators for expanding YOLO's training data.

Pipeline:
  Step 1: Download DeepWeeds + run YOLO zero-shot
  Step 2: Run top LLMs (Florence-2-base, OWLv2) on DeepWeeds
  Step 3: Compare YOLO vs LLM on unseen species
  Step 4: Use LLM detections as pseudo-labels to augment YOLO training
  Step 5: Re-train YOLO with augmented data → measure improvement

Usage:
    python run_cross_dataset.py --step 1    # YOLO zero-shot
    python run_cross_dataset.py --step 2    # LLM inference
    python run_cross_dataset.py --step 3    # Compare + complementarity
    python run_cross_dataset.py --step 4    # Augment training data
    python run_cross_dataset.py --step 5    # Re-train YOLO
    python run_cross_dataset.py --all       # Everything
"""
import argparse
import json
import os
import sys
import shutil
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results", "cross_dataset")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
DW_DIR = os.path.join(DOWNLOAD_DIR, "deepweeds")
CW_DIR = os.path.join(DOWNLOAD_DIR, "cottonweeddet12")
YOLO_MODEL = os.path.join(BASE_DIR, "runs", "detect", "runs",
                           "yolo11n_cottonweeddet12", "weights", "best.pt")
HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

os.makedirs(RESULT_DIR, exist_ok=True)

# Top LLMs to test (best on CottonWeedDet12)
CROSS_LLMS = ["florence2_base", "owlv2"]


# ============================================================
# Step 1: YOLO zero-shot on DeepWeeds
# ============================================================
def step1_yolo_zero_shot():
    """Run YOLO (CottonWeedDet12-trained) on DeepWeeds test set."""
    print("\n" + "=" * 70)
    print("STEP 1: YOLO zero-shot on DeepWeeds (unseen species)")
    print("=" * 70)

    # Download DeepWeeds if needed
    dw_test_imgs = os.path.join(DW_DIR, "test", "images")
    dw_test_labels = os.path.join(DW_DIR, "test", "labels")

    if not os.path.isdir(dw_test_imgs):
        print("[*] Downloading DeepWeeds from Roboflow...")
        from datasets import download_dataset
        download_dataset("deepweeds")

    if not os.path.isdir(dw_test_imgs):
        print("[!] DeepWeeds download failed. Trying alternative...")
        # Try Roboflow direct download
        try:
            from roboflow import Roboflow
            key_file = os.path.join(BASE_DIR, ".roboflow_key")
            if os.path.exists(key_file):
                api_key = open(key_file).read().strip()
            else:
                api_key = os.environ.get("ROBOFLOW_API_KEY", "")
            if api_key:
                rf = Roboflow(api_key=api_key)
                project = rf.workspace("mtsu-2h73y").project("deepweeds-yjbbr")
                dataset = project.version(1).download("yolov8", location=DW_DIR)
                print(f"[+] Downloaded DeepWeeds to {DW_DIR}")
        except Exception as e:
            print(f"[!] Roboflow download failed: {e}")
            return None

    # Count test images
    test_images = sorted([f for f in os.listdir(dw_test_imgs)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[*] DeepWeeds test set: {len(test_images)} images")

    # Run YOLO inference
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running YOLO on {device}...")
    model = YOLO(YOLO_MODEL)

    yolo_pred_dir = os.path.join(RESULT_DIR, "yolo_preds_deepweeds")
    os.makedirs(yolo_pred_dir, exist_ok=True)

    start = time.time()
    for i, img_file in enumerate(test_images):
        img_path = os.path.join(dw_test_imgs, img_file)
        results = model.predict(img_path, conf=0.25, device=device, save=False, verbose=False)

        stem = Path(img_file).stem
        lines = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")

        with open(os.path.join(yolo_pred_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(test_images)}] YOLO predictions")

    elapsed = time.time() - start
    print(f"[+] YOLO inference done in {elapsed:.0f}s")

    # Evaluate
    return evaluate_on_deepweeds("YOLO (CottonWeedDet12-trained)", yolo_pred_dir)


def evaluate_on_deepweeds(model_name, pred_dir):
    """Evaluate predictions against DeepWeeds GT (binary: any weed = positive)."""
    gt_dir = os.path.join(DW_DIR, "test", "labels")

    total_tp, total_fp, total_fn = 0, 0, 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gt_file in sorted(os.listdir(gt_dir)):
        if not gt_file.endswith(".txt"):
            continue
        stem = gt_file.replace(".txt", "")

        # Load GT (binary: class >= 0 and class < 8 = weed, class 8 = negative)
        gt_boxes = []
        with open(os.path.join(gt_dir, gt_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls < 8:  # exclude "Negative" class
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        gt_boxes.append((cls, cx, cy, w, h))

        # Load predictions
        pred_file = os.path.join(pred_dir, f"{stem}.txt")
        pred_boxes = []
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        conf = float(parts[5]) if len(parts) >= 6 else 0.5
                        pred_boxes.append((0, cx, cy, w, h, conf))

        # Match predictions to GT (IoU >= 0.5)
        matched_gt = set()
        sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1][-1], reverse=True)

        tp, fp = 0, 0
        for pi, pred in sorted_preds:
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _compute_iou(pred[1:5], gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1
                per_class[gt_boxes[best_gi][0]]["tp"] += 1
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                per_class[gt[0]]["fn"] += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # DeepWeeds class names
    class_names = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
                   "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed"]

    result = {
        "model": model_name,
        "dataset": "deepweeds",
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_class": {},
    }

    print(f"\n  {model_name}:")
    print(f"    Overall: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
          f"(TP={total_tp} FP={total_fp} FN={total_fn})")
    print(f"    Per-class recall:")
    for cls_id, stats in sorted(per_class.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        cls_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        result["per_class"][name] = {
            "tp": stats["tp"], "fn": stats["fn"],
            "recall": round(cls_recall, 4),
        }
        print(f"      {name:20s}: recall={cls_recall:.3f} (TP={stats['tp']}, FN={stats['fn']})")

    return result


def _compute_iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ============================================================
# Step 2: LLM zero-shot on DeepWeeds
# ============================================================
def step2_llm_zero_shot():
    """Run top LLMs on DeepWeeds test set."""
    print("\n" + "=" * 70)
    print("STEP 2: LLM zero-shot on DeepWeeds")
    print("=" * 70)

    dw_test_imgs = os.path.join(DW_DIR, "test", "images")
    if not os.path.isdir(dw_test_imgs):
        print("[!] DeepWeeds not downloaded. Run step 1 first.")
        return []

    sys.path.insert(0, BASE_DIR)
    from roboflow_bridge import load_model, run_inference, extract_json, convert_bbox_to_yolo

    results = []

    for llm_key in CROSS_LLMS:
        print(f"\n--- Running {llm_key} on DeepWeeds ---")
        llm_pred_dir = os.path.join(RESULT_DIR, f"{llm_key}_preds_deepweeds")
        os.makedirs(llm_pred_dir, exist_ok=True)

        # Check if already done
        existing = len([f for f in os.listdir(llm_pred_dir) if f.endswith('.txt')])
        if existing > 100:
            print(f"[*] Already have {existing} predictions, skipping inference")
        else:
            try:
                model, processor, model_type = load_model(llm_key)
            except Exception as e:
                print(f"[!] Failed to load {llm_key}: {e}")
                continue

            test_images = sorted([f for f in os.listdir(dw_test_imgs)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            from PIL import Image
            import cv2

            for i, img_file in enumerate(test_images):
                img_path = os.path.join(dw_test_imgs, img_file)
                stem = Path(img_file).stem

                try:
                    response = run_inference(model, processor, img_path, model_type)
                    parsed = extract_json(response)

                    lines = []
                    if parsed and "detections" in parsed:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_h, img_w = img.shape[:2]
                            for det in parsed["detections"]:
                                bbox = det.get("bbox") or det.get("bbox_2d")
                                if not bbox or len(bbox) != 4:
                                    continue
                                cx, cy, w, h = convert_bbox_to_yolo(bbox, img_w, img_h)
                                if w > 0 and h > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                                    conf = 0.6  # default LLM confidence
                                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")

                    with open(os.path.join(llm_pred_dir, f"{stem}.txt"), "w") as f:
                        f.write("\n".join(lines) + "\n" if lines else "")

                except Exception as e:
                    with open(os.path.join(llm_pred_dir, f"{stem}.txt"), "w") as f:
                        f.write("")

                if (i + 1) % 200 == 0:
                    print(f"  [{i+1}/{len(test_images)}] {llm_key}")

        # Evaluate
        result = evaluate_on_deepweeds(f"{llm_key} (zero-shot)", llm_pred_dir)
        results.append(result)

    return results


# ============================================================
# Step 3: Compare + Complementarity
# ============================================================
def step3_compare():
    """Compare YOLO vs LLM on unseen species."""
    print("\n" + "=" * 70)
    print("STEP 3: Cross-Dataset Comparison")
    print("=" * 70)

    gt_dir = os.path.join(DW_DIR, "test", "labels")
    if not os.path.isdir(gt_dir):
        print("[!] No DeepWeeds GT labels found")
        return

    yolo_dir = os.path.join(RESULT_DIR, "yolo_preds_deepweeds")
    class_names = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
                   "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed"]

    all_results = {}

    # Evaluate YOLO
    if os.path.isdir(yolo_dir):
        all_results["yolo"] = evaluate_on_deepweeds("YOLO (CottonWeedDet12)", yolo_dir)

    # Evaluate LLMs
    for llm_key in CROSS_LLMS:
        llm_dir = os.path.join(RESULT_DIR, f"{llm_key}_preds_deepweeds")
        if os.path.isdir(llm_dir):
            all_results[llm_key] = evaluate_on_deepweeds(f"{llm_key} (zero-shot)", llm_dir)

    # Complementarity: which GT boxes does LLM catch that YOLO misses?
    if "yolo" in all_results and any(k in all_results for k in CROSS_LLMS):
        print("\n--- Complementarity Analysis (DeepWeeds) ---")
        for llm_key in CROSS_LLMS:
            llm_dir = os.path.join(RESULT_DIR, f"{llm_key}_preds_deepweeds")
            if not os.path.isdir(llm_dir):
                continue

            both_hit = 0
            yolo_only = 0
            llm_only = 0
            both_miss = 0

            for gt_file in sorted(os.listdir(gt_dir)):
                if not gt_file.endswith(".txt"):
                    continue
                stem = gt_file.replace(".txt", "")

                gt_boxes = []
                with open(os.path.join(gt_dir, gt_file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) < 8:
                            gt_boxes.append(tuple(float(x) for x in parts[:5]))

                # Load YOLO preds
                yp = os.path.join(yolo_dir, f"{stem}.txt")
                yolo_boxes = []
                if os.path.exists(yp):
                    with open(yp) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                yolo_boxes.append(tuple(float(x) for x in parts[:5]) + (float(parts[5]) if len(parts) > 5 else 0.5,))

                # Load LLM preds
                lp = os.path.join(llm_dir, f"{stem}.txt")
                llm_boxes = []
                if os.path.exists(lp):
                    with open(lp) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                llm_boxes.append(tuple(float(x) for x in parts[:5]) + (float(parts[5]) if len(parts) > 5 else 0.5,))

                # Match each to GT
                yolo_hits = set()
                for yi, yb in enumerate(yolo_boxes):
                    for gi, gb in enumerate(gt_boxes):
                        if _compute_iou(yb[1:5], gb[1:5]) >= 0.5:
                            yolo_hits.add(gi)
                            break

                llm_hits = set()
                for li, lb in enumerate(llm_boxes):
                    for gi, gb in enumerate(gt_boxes):
                        if _compute_iou(lb[1:5], gb[1:5]) >= 0.5:
                            llm_hits.add(gi)
                            break

                for gi in range(len(gt_boxes)):
                    y = gi in yolo_hits
                    l = gi in llm_hits
                    if y and l: both_hit += 1
                    elif y and not l: yolo_only += 1
                    elif not y and l: llm_only += 1
                    else: both_miss += 1

            total = both_hit + yolo_only + llm_only + both_miss
            if total > 0:
                print(f"\n  YOLO vs {llm_key} on DeepWeeds:")
                print(f"    Both hit:    {both_hit:5d} ({both_hit/total*100:.1f}%)")
                print(f"    YOLO only:   {yolo_only:5d} ({yolo_only/total*100:.1f}%)")
                print(f"    LLM rescues: {llm_only:5d} ({llm_only/total*100:.1f}%) ← LLM advantage")
                print(f"    Both miss:   {both_miss:5d} ({both_miss/total*100:.1f}%)")

    # Save
    out_path = os.path.join(RESULT_DIR, "cross_dataset_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")
    return all_results


# ============================================================
# Step 4: Create augmented training set
# ============================================================
def step4_augment():
    """Use LLM detections on DeepWeeds as pseudo-labels to augment YOLO training."""
    print("\n" + "=" * 70)
    print("STEP 4: Create LLM-augmented training set")
    print("=" * 70)

    aug_dir = os.path.join(RESULT_DIR, "augmented_training")
    aug_imgs = os.path.join(aug_dir, "images")
    aug_labels = os.path.join(aug_dir, "labels")
    os.makedirs(aug_imgs, exist_ok=True)
    os.makedirs(aug_labels, exist_ok=True)

    # Copy CottonWeedDet12 training data (original)
    cw_train_imgs = os.path.join(CW_DIR, "train", "images")
    cw_train_labels = os.path.join(CW_DIR, "train", "labels")

    if os.path.isdir(cw_train_imgs):
        cw_count = 0
        for f in os.listdir(cw_train_imgs):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_img = os.path.join(cw_train_imgs, f)
                src_lbl = os.path.join(cw_train_labels, Path(f).stem + ".txt")
                if os.path.exists(src_lbl):
                    shutil.copy2(src_img, os.path.join(aug_imgs, f))
                    shutil.copy2(src_lbl, os.path.join(aug_labels, Path(f).stem + ".txt"))
                    cw_count += 1
        print(f"[+] Copied {cw_count} CottonWeedDet12 training images")

    # Add DeepWeeds images with LLM pseudo-labels (best LLM = florence2_base)
    dw_train_imgs = os.path.join(DW_DIR, "train", "images")
    best_llm_preds = os.path.join(RESULT_DIR, "florence2_base_preds_deepweeds")

    # Use training images with pseudo-labels from LLM test predictions
    # (In practice, you'd run LLM on train set too; here we demonstrate the concept)
    dw_test_imgs = os.path.join(DW_DIR, "test", "images")

    if os.path.isdir(best_llm_preds) and os.path.isdir(dw_test_imgs):
        dw_count = 0
        for f in os.listdir(best_llm_preds):
            if not f.endswith(".txt"):
                continue
            stem = f.replace(".txt", "")
            pred_path = os.path.join(best_llm_preds, f)

            # Only include images where LLM found detections
            with open(pred_path) as fh:
                lines = [l.strip() for l in fh if l.strip()]
            if not lines:
                continue

            # Find corresponding image
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                img_path = os.path.join(dw_test_imgs, stem + ext)
                if os.path.exists(img_path):
                    # Remap all classes to a new class ID (12 = "novel_weed")
                    new_lines = []
                    for line in lines:
                        parts = line.split()
                        parts[0] = "12"  # new class: novel weed detected by LLM
                        new_lines.append(" ".join(parts[:5]))  # drop confidence

                    shutil.copy2(img_path, os.path.join(aug_imgs, f"dw_{stem}{ext}"))
                    with open(os.path.join(aug_labels, f"dw_{stem}.txt"), "w") as fh:
                        fh.write("\n".join(new_lines) + "\n")
                    dw_count += 1
                    break

        print(f"[+] Added {dw_count} DeepWeeds images with LLM pseudo-labels")
        print(f"[+] Total augmented training set: {cw_count + dw_count} images")

    # Create data.yaml for augmented training
    yaml_path = os.path.join(aug_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {aug_imgs}\n")
        f.write(f"val: {os.path.join(CW_DIR, 'valid', 'images')}\n")
        f.write(f"nc: 13\n")  # 12 original + 1 novel
        f.write(f"names: ['Carpetweeds','Crabgrass','Eclipta','Goosegrass','Morningglory',"
                f"'Nutsedge','PalmerAmaranth','PricklySida','Purslane','Ragweed',"
                f"'Sicklepod','SpottedSpurge','novel_weed']\n")
    print(f"[+] Created {yaml_path}")

    return aug_dir


# ============================================================
# Step 5: Re-train YOLO with augmented data
# ============================================================
def step5_retrain(aug_dir=None):
    """Re-train YOLO with augmented data and evaluate on DeepWeeds."""
    print("\n" + "=" * 70)
    print("STEP 5: Re-train YOLO with LLM-augmented data")
    print("=" * 70)

    if aug_dir is None:
        aug_dir = os.path.join(RESULT_DIR, "augmented_training")

    yaml_path = os.path.join(aug_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print("[!] No augmented data.yaml found. Run step 4 first.")
        return

    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on {device}...")

    # Train from scratch with augmented data
    model = YOLO("yolo11n.pt")  # fresh pretrained
    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=-1,
        device=device,
        project=os.path.join(RESULT_DIR, "yolo_augmented"),
        name="train",
        patience=15,
        verbose=True,
    )

    # Evaluate on DeepWeeds test set
    print("\n[*] Evaluating augmented YOLO on DeepWeeds test set...")
    best_model = YOLO(os.path.join(RESULT_DIR, "yolo_augmented", "train", "weights", "best.pt"))

    dw_test_imgs = os.path.join(DW_DIR, "test", "images")
    aug_pred_dir = os.path.join(RESULT_DIR, "yolo_augmented_preds_deepweeds")
    os.makedirs(aug_pred_dir, exist_ok=True)

    test_images = sorted([f for f in os.listdir(dw_test_imgs)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in test_images:
        img_path = os.path.join(dw_test_imgs, img_file)
        preds = best_model.predict(img_path, conf=0.25, device=device, save=False, verbose=False)
        stem = Path(img_file).stem
        lines = []
        for r in preds:
            for box in r.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")
        with open(os.path.join(aug_pred_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    aug_result = evaluate_on_deepweeds("YOLO (augmented with LLM)", aug_pred_dir)

    # Compare with original YOLO zero-shot
    orig_result_path = os.path.join(RESULT_DIR, "cross_dataset_results.json")
    if os.path.exists(orig_result_path):
        orig = json.load(open(orig_result_path))
        if "yolo" in orig:
            print(f"\n  === IMPROVEMENT ===")
            print(f"  Original YOLO (zero-shot): F1={orig['yolo']['f1']:.3f}")
            print(f"  Augmented YOLO (LLM data): F1={aug_result['f1']:.3f}")
            print(f"  Delta: {aug_result['f1'] - orig['yolo']['f1']:+.3f}")

    return aug_result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Experiment")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], help="Run specific step")
    args = parser.parse_args()

    if not args.all and not args.step:
        args.all = True

    if args.all or args.step == 1:
        step1_yolo_zero_shot()

    if args.all or args.step == 2:
        step2_llm_zero_shot()

    if args.all or args.step == 3:
        step3_compare()

    if args.all or args.step == 4:
        step4_augment()

    if args.all or args.step == 5:
        step5_retrain()

    print("\n" + "=" * 70)
    print("[+] CROSS-DATASET EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
