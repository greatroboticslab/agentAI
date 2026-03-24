#!/usr/bin/env python3
"""
SAM-Enhanced LLM Labeling for YOLO Training

Uses SAM (Segment Anything) to provide precise object boundaries,
then feeds segmentation masks + original image to LLM for better labeling.

Pipeline:
  1. SAM segments all objects in holdout images → precise masks
  2. For each SAM segment, crop and send to Florence-2 for classification
  3. Florence-2 classifies: weed or not? what species?
  4. Convert SAM masks to YOLO-format bboxes with LLM labels
  5. Train YOLO with these high-quality labels
  6. Evaluate on old + new species

Comparison:
  - Previous: Florence-2 direct detection (precision=72.6%) → noisy labels
  - This: SAM segments + Florence-2 classifies → should be more precise
"""
import json, os, shutil, time, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
RESULT_DIR = os.path.join(BASE_DIR, "results", "sam_enhanced")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

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
    tp, fp, fn = 0, 0, 0
    for f in sorted(os.listdir(test_imgs)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        stem = Path(f).stem
        lp = os.path.join(test_lbls, stem + ".txt")
        if not os.path.exists(lp): continue
        gt = []
        for l in open(lp):
            p = l.strip().split()
            if len(p) >= 5: gt.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        res = model.predict(os.path.join(test_imgs, f), conf=0.25, device=device, verbose=False)
        preds = []
        for r in res:
            for b in r.boxes:
                x, y, w, h = b.xywhn[0].tolist()
                preds.append((x, y, w, h, float(b.conf[0])))
        preds.sort(key=lambda x: x[4], reverse=True)
        matched = set()
        for pb in preds:
            bi, bg = 0, -1
            for gi, gb in enumerate(gt):
                if gi in matched: continue
                iou = compute_iou(pb[:4], gb)
                if iou > bi: bi, bg = iou, gi
            if bi >= 0.5 and bg >= 0: matched.add(bg); tp += 1
            else: fp += 1
        fn += len(gt) - len(matched)
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"  {label}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def step1_sam_segment():
    """Run SAM on holdout images to get precise object masks."""
    print("\n" + "=" * 70)
    print("STEP 1: SAM Segmentation on holdout images")
    print("=" * 70)

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download SAM model if needed
    sam_checkpoint = os.path.join(HF_CACHE, "sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_checkpoint):
        print("[*] Downloading SAM ViT-B checkpoint...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
        print(f"[+] Downloaded to {sam_checkpoint}")

    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )

    holdout_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    sam_output_dir = os.path.join(RESULT_DIR, "sam_segments")
    os.makedirs(sam_output_dir, exist_ok=True)

    imgs = sorted([f for f in os.listdir(holdout_imgs)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[*] Processing {len(imgs)} images...")

    stats = {"total_images": 0, "total_segments": 0}

    for i, img_file in enumerate(imgs):
        img_path = os.path.join(holdout_imgs, img_file)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        masks = mask_generator.generate(image_rgb)

        # Convert masks to bboxes (YOLO format)
        stem = Path(img_file).stem
        segment_data = []
        for mask in masks:
            bbox = mask["bbox"]  # [x, y, w, h] in pixels
            area = mask["area"]
            # Filter: ignore very small or very large segments
            area_ratio = area / (h * w)
            if area_ratio < 0.01 or area_ratio > 0.8:
                continue
            # Convert to YOLO format [cx, cy, w, h] normalized
            cx = (bbox[0] + bbox[2] / 2) / w
            cy = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            segment_data.append({
                "bbox_yolo": [cx, cy, bw, bh],
                "bbox_pixel": bbox,
                "area_ratio": round(area_ratio, 4),
                "stability_score": round(mask["stability_score"], 4),
            })
            stats["total_segments"] += 1

        # Save segment data
        with open(os.path.join(sam_output_dir, f"{stem}.json"), "w") as f:
            json.dump(segment_data, f, indent=2)

        stats["total_images"] += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(imgs)}] {stats['total_segments']} segments so far")

    print(f"  Total: {stats['total_images']} images, {stats['total_segments']} segments")
    avg = stats['total_segments'] / max(stats['total_images'], 1)
    print(f"  Average: {avg:.1f} segments per image")
    return sam_output_dir, stats


def step2_classify_segments():
    """Use Florence-2 to classify each SAM segment as weed/not-weed."""
    print("\n" + "=" * 70)
    print("STEP 2: Florence-2 classifies SAM segments")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Florence-2-base
    model_id = "microsoft/Florence-2-base"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )

    holdout_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    sam_dir = os.path.join(RESULT_DIR, "sam_segments")
    label_dir = os.path.join(RESULT_DIR, "sam_florence_labels")
    os.makedirs(label_dir, exist_ok=True)

    stats = {"total_crops": 0, "weed_detected": 0, "not_weed": 0}

    for seg_file in sorted(os.listdir(sam_dir)):
        if not seg_file.endswith(".json"): continue
        stem = seg_file.replace(".json", "")

        # Find image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            p = os.path.join(holdout_imgs, stem + ext)
            if os.path.exists(p): img_path = p; break
        if not img_path: continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        segments = json.load(open(os.path.join(sam_dir, seg_file)))

        yolo_lines = []
        for seg in segments:
            bbox_px = seg["bbox_pixel"]
            # Crop the segment region (with padding)
            x1 = max(0, int(bbox_px[0] - bbox_px[2] * 0.1))
            y1 = max(0, int(bbox_px[1] - bbox_px[3] * 0.1))
            x2 = min(w, int(bbox_px[0] + bbox_px[2] * 1.1))
            y2 = min(h, int(bbox_px[1] + bbox_px[3] * 1.1))
            crop = image.crop((x1, y1, x2, y2))

            # Ask Florence-2: is this a weed?
            task = "<CAPTION>"
            inputs = processor(text=task, images=crop, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0].lower()

            stats["total_crops"] += 1

            # Check if caption mentions plant/weed/vegetation
            is_weed = any(word in caption for word in
                          ["weed", "plant", "grass", "leaf", "green", "vegetation",
                           "flower", "herb", "crop", "seedling", "sprout"])

            if is_weed:
                stats["weed_detected"] += 1
                cx, cy, bw, bh = seg["bbox_yolo"]
                yolo_lines.append(f"8 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                stats["not_weed"] += 1

        # Save YOLO labels
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

    print(f"  Total crops classified: {stats['total_crops']}")
    print(f"  Weed detected: {stats['weed_detected']}")
    print(f"  Not weed: {stats['not_weed']}")
    precision_est = stats['weed_detected'] / max(stats['total_crops'], 1)
    print(f"  Weed ratio: {precision_est:.1%}")
    return label_dir, stats


def step3_merge_and_train():
    """Merge SAM+Florence labels with YOLO old-species labels, then train."""
    print("\n" + "=" * 70)
    print("STEP 3: Merge labels and train YOLO")
    print("=" * 70)

    from ultralytics import YOLO
    import torch

    # Create dataset with YOLO old-species labels + SAM-enhanced new labels
    ds_dir = os.path.join(RESULT_DIR, "dataset_sam_enhanced")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    # Copy 8-species training data
    sp8_dir = os.path.join(L4O_DIR, "dataset_8species")
    count_orig = 0
    for split in ["train", "valid"]:
        src_imgs = os.path.join(sp8_dir, split, "images")
        src_lbls = os.path.join(sp8_dir, split, "labels")
        if not os.path.isdir(src_imgs): continue
        for f in os.listdir(src_imgs):
            shutil.copy2(os.path.join(src_imgs, f), os.path.join(ds_dir, split, "images", f))
            stem = Path(f).stem
            lbl = os.path.join(src_lbls, f"{stem}.txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(ds_dir, split, "labels", f"{stem}.txt"))
                count_orig += 1

    # Add holdout images with SAM+Florence labels + YOLO old-species labels
    holdout_imgs = os.path.join(L4O_DIR, "dataset_holdout", "train", "images")
    sam_labels = os.path.join(RESULT_DIR, "sam_florence_labels")
    balpw_labels = os.path.join(BASE_DIR, "results", "balpw", "balpw_labels")

    # Use BA-LPW labels for old species + SAM labels for new species
    count_enhanced = 0
    for lbl_file in os.listdir(sam_labels):
        if not lbl_file.endswith(".txt"): continue
        stem = lbl_file.replace(".txt", "")

        # SAM+Florence new species labels
        sam_lines = []
        with open(os.path.join(sam_labels, lbl_file)) as f:
            sam_lines = [l.strip() for l in f if l.strip()]

        # YOLO old species labels (from BA-LPW)
        old_lines = []
        balpw_lbl = os.path.join(balpw_labels, lbl_file)
        if os.path.exists(balpw_lbl):
            with open(balpw_lbl) as f:
                for l in f:
                    parts = l.strip().split()
                    if parts and int(parts[0]) < 8:  # old species only
                        old_lines.append(l.strip())

        merged = old_lines + sam_lines
        if not merged: continue

        # Find image
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            img_path = os.path.join(holdout_imgs, stem + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(ds_dir, "train/images", f"sam_{stem}{ext}"))
                with open(os.path.join(ds_dir, "train/labels", f"sam_{stem}.txt"), "w") as f:
                    f.write("\n".join(merged) + "\n")
                count_enhanced += 1
                break

    print(f"  Original 8sp images: {count_orig}")
    print(f"  SAM-enhanced images: {count_enhanced}")

    # data.yaml
    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed"]
    yaml_path = os.path.join(ds_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(ds_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\nnames: {names}\n")

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_8sp = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")
    model = YOLO(yolo_8sp)
    model.train(data=yaml_path, epochs=50, batch=-1, device=device,
                project=os.path.join(RESULT_DIR, "yolo_sam_enhanced"),
                name="train", patience=15, lr0=0.001, verbose=True)

    return os.path.join(RESULT_DIR, "yolo_sam_enhanced", "train", "weights", "best.pt")


def step4_evaluate():
    """Compare SAM-enhanced vs all previous methods."""
    print("\n" + "=" * 70)
    print("STEP 4: Evaluation — SAM-enhanced vs previous methods")
    print("=" * 70)

    old_imgs = os.path.join(L4O_DIR, "dataset_8species", "test", "images")
    old_lbls = os.path.join(L4O_DIR, "dataset_8species", "test", "labels")
    new_imgs = os.path.join(L4O_DIR, "dataset_holdout", "test", "images")
    new_lbls = os.path.join(L4O_DIR, "dataset_holdout", "test", "labels")

    models = {
        "yolo_8sp": os.path.join(L4O_DIR, "yolo_8species/train/weights/best.pt"),
        "naive_aug": os.path.join(L4O_DIR, "yolo_augmented/train/weights/best.pt"),
        "balpw": os.path.join(BASE_DIR, "results/balpw/yolo_balpw/train/weights/best.pt"),
        "sam_enhanced": os.path.join(RESULT_DIR, "yolo_sam_enhanced/train/weights/best.pt"),
    }

    results = {}
    baseline_old = baseline_new = 0

    print(f"\n{'Method':<20s} {'Old F1':>8s} {'New F1':>8s} {'Old D':>8s} {'New D':>8s}")
    print("-" * 55)

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"  {name}: NOT FOUND"); continue
        old = evaluate_binary(path, old_imgs, old_lbls, f"{name}(old)")
        new = evaluate_binary(path, new_imgs, new_lbls, f"{name}(new)")
        results[name] = {"old_species": old, "new_species": new}
        if name == "yolo_8sp": baseline_old, baseline_new = old["f1"], new["f1"]
        print(f"{name:<20s} {old['f1']:8.3f} {new['f1']:8.3f} "
              f"{old['f1']-baseline_old:+8.3f} {new['f1']-baseline_new:+8.3f}")

    out = os.path.join(RESULT_DIR, "sam_enhanced_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out}")


def main():
    print("=" * 70)
    print("SAM-Enhanced LLM Labeling for YOLO")
    print("=" * 70)
    step1_sam_segment()
    step2_classify_segments()
    step3_merge_and_train()
    step4_evaluate()
    print("\n[+] SAM-ENHANCED EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
