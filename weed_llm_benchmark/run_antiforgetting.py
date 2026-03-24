#!/usr/bin/env python3
"""
Anti-Forgetting Methods for YOLO Continual Learning

Tests multiple strategies to prevent catastrophic forgetting when adding new species:
  Method 1: Replay buffer (mix old training data into new training)
  Method 2: Frozen backbone (only train detection head on new species)
  Method 3: Progressive fine-tuning (very low lr + more epochs)
  Method 4: Combined (replay + frozen backbone + progressive lr)

All methods compared against:
  - YOLO 8sp baseline (no new species)
  - YOLO naive aug (simple addition, -2.4% forgetting)
  - YOLO BA-LPW (complete labels, -2.2% forgetting)
"""
import json, os, shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
BALPW_DIR = os.path.join(BASE_DIR, "results", "balpw")
RESULT_DIR = os.path.join(BASE_DIR, "results", "antiforgetting")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")

os.makedirs(RESULT_DIR, exist_ok=True)

ALL_CLASSES = {
    0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
    4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
    8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
}
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}


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
                pred_boxes.append((x, y, w, h, float(box.conf[0])))
        pred_boxes.sort(key=lambda x: x[4], reverse=True)
        matched = set()
        tp = 0
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched: continue
                iou = compute_iou(pb[:4], gb)
                if iou > best_iou: best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi); tp += 1
        total_tp += tp
        total_fp += len(pred_boxes) - tp
        total_fn += len(gt_boxes) - len(matched)
    prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
    rec = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"  {label}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def create_dataset_with_replay(replay_ratio=0.3):
    """Create training set: BA-LPW new images + replay buffer of old images."""
    ds_dir = os.path.join(RESULT_DIR, f"dataset_replay{int(replay_ratio*100)}")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    # BA-LPW labeled new species images
    balpw_labels = os.path.join(BALPW_DIR, "balpw_labels")
    holdout_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    count_new = 0
    if os.path.isdir(balpw_labels):
        for lbl_file in os.listdir(balpw_labels):
            if not lbl_file.endswith(".txt"): continue
            stem = lbl_file.replace(".txt", "")
            with open(os.path.join(balpw_labels, lbl_file)) as f:
                if not f.read().strip(): continue
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                img_path = os.path.join(holdout_imgs, stem + ext)
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(ds_dir, "train/images", f"new_{stem}{ext}"))
                    shutil.copy2(os.path.join(balpw_labels, lbl_file),
                                 os.path.join(ds_dir, "train/labels", f"new_{stem}.txt"))
                    count_new += 1
                    break

    # Replay buffer: sample from 8-species training data
    sp8_imgs = os.path.join(L4O_DIR, "dataset_8species", "train", "images")
    sp8_lbls = os.path.join(L4O_DIR, "dataset_8species", "train", "labels")
    import random
    random.seed(42)
    if os.path.isdir(sp8_imgs):
        all_old = [f for f in os.listdir(sp8_imgs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        n_replay = int(count_new * replay_ratio / (1 - replay_ratio)) if replay_ratio < 1 else len(all_old)
        n_replay = min(n_replay, len(all_old))
        replay = random.sample(all_old, n_replay)
        for f in replay:
            shutil.copy2(os.path.join(sp8_imgs, f), os.path.join(ds_dir, "train/images", f))
            stem = Path(f).stem
            lbl = os.path.join(sp8_lbls, f"{stem}.txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(ds_dir, "train/labels", f"{stem}.txt"))

    # Copy validation from 8sp
    sp8_valid_imgs = os.path.join(L4O_DIR, "dataset_8species", "valid", "images")
    sp8_valid_lbls = os.path.join(L4O_DIR, "dataset_8species", "valid", "labels")
    if os.path.isdir(sp8_valid_imgs):
        for f in os.listdir(sp8_valid_imgs):
            shutil.copy2(os.path.join(sp8_valid_imgs, f), os.path.join(ds_dir, "valid/images", f))
            stem = Path(f).stem
            lbl = os.path.join(sp8_valid_lbls, f"{stem}.txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(ds_dir, "valid/labels", f"{stem}.txt"))

    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed"]
    yaml_path = os.path.join(ds_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(ds_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\n")
        f.write(f"names: {names}\n")

    n_train = len([f for f in os.listdir(os.path.join(ds_dir, "train/images"))])
    print(f"  Dataset: {n_train} images (new={count_new}, replay={n_train-count_new})")
    return yaml_path


def train_method(name, yaml_path, freeze_backbone=False, lr=0.01, epochs=50):
    """Train YOLO with specific anti-forgetting configuration."""
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_8sp = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")
    model = YOLO(yolo_8sp)

    # Freeze backbone if requested (only train detection head)
    freeze_arg = 10 if freeze_backbone else 0  # freeze first 10 layers (backbone)

    print(f"\n[*] Training {name}: lr={lr}, epochs={epochs}, freeze={freeze_backbone}")
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=-1,
        device=device,
        project=os.path.join(RESULT_DIR, name),
        name="train",
        patience=15,
        lr0=lr,
        freeze=freeze_arg,
        verbose=True,
    )
    return os.path.join(RESULT_DIR, name, "train", "weights", "best.pt")


def main():
    print("=" * 70)
    print("ANTI-FORGETTING METHODS COMPARISON")
    print("=" * 70)

    holdout_test_imgs = os.path.join(L4O_DIR, "dataset_holdout", "test", "images")
    holdout_test_lbls = os.path.join(L4O_DIR, "dataset_holdout", "test", "labels")
    sp8_test_imgs = os.path.join(L4O_DIR, "dataset_8species", "test", "images")
    sp8_test_lbls = os.path.join(L4O_DIR, "dataset_8species", "test", "labels")

    results = {}

    # Method 1: Replay buffer (50% old data mixed in) + BA-LPW labels
    print("\n--- Method 1: Replay buffer (50%) + BA-LPW ---")
    yaml1 = create_dataset_with_replay(replay_ratio=0.5)
    path1 = train_method("m1_replay50", yaml1, lr=0.001, epochs=50)

    # Method 2: Frozen backbone + BA-LPW labels (only train head)
    print("\n--- Method 2: Frozen backbone + BA-LPW ---")
    yaml2 = create_dataset_with_replay(replay_ratio=0.3)
    path2 = train_method("m2_frozen", yaml2, freeze_backbone=True, lr=0.005, epochs=50)

    # Method 3: Progressive (very low lr + more epochs)
    print("\n--- Method 3: Progressive fine-tuning ---")
    yaml3 = create_dataset_with_replay(replay_ratio=0.5)
    path3 = train_method("m3_progressive", yaml3, lr=0.0005, epochs=100)

    # Method 4: Combined (replay + frozen + low lr)
    print("\n--- Method 4: Combined (replay + frozen + low lr) ---")
    yaml4 = create_dataset_with_replay(replay_ratio=0.5)
    path4 = train_method("m4_combined", yaml4, freeze_backbone=True, lr=0.0005, epochs=100)

    # Evaluate all
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    baselines = {
        "yolo_8sp": os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt"),
        "naive_aug": os.path.join(L4O_DIR, "yolo_augmented", "train", "weights", "best.pt"),
    }
    methods = {
        "m1_replay50": path1,
        "m2_frozen": path2,
        "m3_progressive": path3,
        "m4_combined": path4,
    }
    all_models = {**baselines, **methods}

    for name, path in all_models.items():
        if not os.path.exists(path):
            print(f"  [!] {name}: not found")
            continue
        print(f"\n--- {name} ---")
        results[name] = {
            "old_species": evaluate_binary(path, sp8_test_imgs, sp8_test_lbls, f"{name} (old)"),
            "new_species": evaluate_binary(path, holdout_test_imgs, holdout_test_lbls, f"{name} (new)"),
        }

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    baseline_old = results.get("yolo_8sp", {}).get("old_species", {}).get("f1", 0)
    baseline_new = results.get("yolo_8sp", {}).get("new_species", {}).get("f1", 0)

    print(f"{'Method':<25s} {'Old F1':>8s} {'New F1':>8s} {'Old Δ':>8s} {'New Δ':>8s}")
    print("-" * 60)
    for name in ["yolo_8sp", "naive_aug", "m1_replay50", "m2_frozen", "m3_progressive", "m4_combined"]:
        if name not in results: continue
        old = results[name]["old_species"]["f1"]
        new = results[name]["new_species"]["f1"]
        print(f"{name:<25s} {old:8.3f} {new:8.3f} {old-baseline_old:+8.3f} {new-baseline_new:+8.3f}")

    # Save
    out_path = os.path.join(RESULT_DIR, "antiforgetting_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
