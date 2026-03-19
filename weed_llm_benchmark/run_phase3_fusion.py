#!/usr/bin/env python3
"""
Phase 3: YOLO+LLM Fusion Experiments

Runs all fusion experiments offline using existing YOLO and LLM detection results.
No GPU needed except for initial YOLO prediction generation.

Experiments:
  E1: Pairwise fusion (7 LLMs x 3 strategies)
  E2: IoU threshold sweep (top-3 LLMs x 3 strategies x 14 thresholds)
  E3: Complementarity analysis (per-image, per-class, spatial)
  E4: Improved strategies (confidence-aware, architecture-specialized)
  E5: Multi-LLM ensemble (majority vote, weighted)
  E6: Statistical significance (bootstrap CI)

Usage:
    python run_phase3_fusion.py --all
    python run_phase3_fusion.py --experiment e1
    python run_phase3_fusion.py --experiment e3 --llm florence2_base
"""
import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
GT_DIR = os.path.join(DOWNLOAD_DIR, "cottonweeddet12", "test", "labels")
IMG_DIR = os.path.join(DOWNLOAD_DIR, "cottonweeddet12", "test", "images")
YOLO_MODEL = os.path.join(BASE_DIR, "runs", "detect", "runs",
                           "yolo11n_cottonweeddet12", "weights", "best.pt")

# LLMs that produced mAP > 0 (worth fusing)
FUSION_LLMS = [
    "florence2_base",   # mAP=0.434, Prec=0.789, Rec=0.519
    "florence2",        # mAP=0.329, Prec=0.692, Rec=0.431
    "internvl2",        # mAP=0.208, Prec=0.545, Rec=0.354
    "qwen3b",           # mAP=0.196, Prec=0.333, Rec=0.249
    "minicpm_v45",      # mAP=0.192, Prec=0.407, Rec=0.340
    "owlv2",            # mAP=0.184, Prec=0.194, Rec=0.943
    "qwen7b",           # mAP=0.176, Prec=0.334, Rec=0.214
]

STRATEGIES = ["supplement", "filter", "weighted"]
DATASET = "cottonweeddet12"

os.makedirs(os.path.join(RESULT_DIR, "phase3"), exist_ok=True)


# ============================================================
# Data Loading
# ============================================================
def load_yolo_predictions(yolo_pred_dir):
    """Load YOLO per-image predictions from label files.
    Returns {image_stem: [(class_id, cx, cy, w, h, confidence), ...]}
    """
    preds = {}
    for f in sorted(os.listdir(yolo_pred_dir)):
        if not f.endswith(".txt"):
            continue
        stem = f.replace(".txt", "")
        boxes = []
        with open(os.path.join(yolo_pred_dir, f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    conf = float(parts[5]) if len(parts) >= 6 else 0.9
                    boxes.append((cls, cx, cy, w, h, conf))
        preds[stem] = boxes
    return preds


def load_llm_predictions(model_key):
    """Load LLM per-image predictions from label files.
    Returns {image_stem: [(class_id, cx, cy, w, h, confidence), ...]}
    """
    label_dir = os.path.join(LABELED_DIR, f"{model_key}_{DATASET}", "detected", "labels")
    if not os.path.isdir(label_dir):
        print(f"[!] No labels for {model_key}")
        return {}
    return load_yolo_predictions(label_dir)


def load_gt():
    """Load ground truth labels.
    Returns {image_stem: [(class_id, cx, cy, w, h), ...]}
    """
    gt = {}
    for f in sorted(os.listdir(GT_DIR)):
        if not f.endswith(".txt"):
            continue
        stem = f.replace(".txt", "")
        boxes = []
        with open(os.path.join(GT_DIR, f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls, cx, cy, w, h))
        gt[stem] = boxes
    return gt


def generate_yolo_predictions():
    """Run YOLO inference on test set, save as label files with confidence.
    Falls back to CPU if no GPU available.
    """
    yolo_pred_dir = os.path.join(RESULT_DIR, "phase3", "yolo_preds")
    if os.path.isdir(yolo_pred_dir) and len([f for f in os.listdir(yolo_pred_dir) if f.endswith('.txt')]) > 800:
        print(f"[*] YOLO predictions already exist ({len(os.listdir(yolo_pred_dir))} files)")
        return yolo_pred_dir

    print("[*] Generating YOLO predictions on test set...")
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    model = YOLO(YOLO_MODEL)

    os.makedirs(yolo_pred_dir, exist_ok=True)

    # Process images one by one for stability on CPU
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(IMG_DIR, img_file)
        stem = Path(img_file).stem
        results = model.predict(img_path, conf=0.25, device=device, save=False, verbose=False)

        lines = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")

        with open(os.path.join(yolo_pred_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(img_files)}] YOLO predictions generated")

    total = len([f for f in os.listdir(yolo_pred_dir) if f.endswith('.txt')])
    print(f"[+] Saved {total} YOLO prediction files to {yolo_pred_dir}")
    return yolo_pred_dir


# ============================================================
# IoU and Matching
# ============================================================
def compute_iou(box1, box2):
    """IoU between two YOLO-format boxes (cx, cy, w, h)."""
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


def match_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predictions to GT. Returns (tp_indices, fp_indices, fn_indices)."""
    if not pred_boxes:
        return [], [], list(range(len(gt_boxes)))
    if not gt_boxes:
        return [], list(range(len(pred_boxes))), []

    # Sort by confidence descending
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1][-1], reverse=True)
    matched_gt = set()
    tp_pred = []
    fp_pred = []

    for pred_idx, pred in sorted_preds:
        best_iou, best_gt = 0, -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred[1:5], gt[1:5])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            matched_gt.add(best_gt)
            tp_pred.append(pred_idx)
        else:
            fp_pred.append(pred_idx)

    fn_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp_pred, fp_pred, fn_gt


# ============================================================
# Fusion Strategies
# ============================================================
def fuse_single_image(yolo_boxes, llm_boxes, strategy="supplement",
                      match_iou=0.3, llm_conf_threshold=0.0):
    """Fuse YOLO and LLM detections for one image.

    Args:
        yolo_boxes: [(cls, cx, cy, w, h, conf), ...]
        llm_boxes: [(cls, cx, cy, w, h, conf), ...]
        strategy: supplement|filter|weighted
        match_iou: IoU threshold for matching YOLO↔LLM
        llm_conf_threshold: minimum LLM confidence for supplement

    Returns: fused [(cls, cx, cy, w, h, conf), ...]
    """
    if not yolo_boxes:
        if strategy == "supplement":
            return [b for b in llm_boxes if b[5] >= llm_conf_threshold]
        return []

    if not llm_boxes:
        if strategy == "filter":
            return []  # no LLM confirmation → drop all
        return list(yolo_boxes)

    # Match YOLO ↔ LLM by IoU
    yolo_matched = set()
    llm_matched = set()

    for yi, yb in enumerate(yolo_boxes):
        best_iou, best_li = 0, -1
        for li, lb in enumerate(llm_boxes):
            if li in llm_matched:
                continue
            iou = compute_iou(yb[1:5], lb[1:5])
            if iou > best_iou:
                best_iou = iou
                best_li = li
        if best_iou >= match_iou and best_li >= 0:
            yolo_matched.add(yi)
            llm_matched.add(best_li)

    fused = []

    if strategy == "supplement":
        # Keep all YOLO + add unmatched LLM detections
        fused = list(yolo_boxes)
        for li, lb in enumerate(llm_boxes):
            if li not in llm_matched and lb[5] >= llm_conf_threshold:
                fused.append(lb)

    elif strategy == "filter":
        # Keep only YOLO detections confirmed by LLM
        for yi in range(len(yolo_boxes)):
            if yi in yolo_matched:
                fused.append(yolo_boxes[yi])

    elif strategy == "weighted":
        # Matched: boost confidence. Unmatched YOLO: keep. Unmatched LLM: add with penalty.
        for yi, yb in enumerate(yolo_boxes):
            if yi in yolo_matched:
                # Boost: average of YOLO conf and 1.0 (confirmed by LLM)
                boosted_conf = min(1.0, yb[5] * 0.7 + 0.3)
                fused.append((yb[0], yb[1], yb[2], yb[3], yb[4], boosted_conf))
            else:
                # Unmatched YOLO: slight penalty
                fused.append((yb[0], yb[1], yb[2], yb[3], yb[4], yb[5] * 0.9))
        for li, lb in enumerate(llm_boxes):
            if li not in llm_matched and lb[5] >= llm_conf_threshold:
                # Unmatched LLM: add with heavy penalty
                fused.append((lb[0], lb[1], lb[2], lb[3], lb[4], lb[5] * 0.5))

    return fused


def evaluate_fusion(fused_preds, gt, iou_threshold=0.5):
    """Evaluate fused predictions against GT.
    Returns dict with mAP, precision, recall, F1.
    """
    all_tp, all_fp, all_fn = 0, 0, 0

    for stem in gt:
        pred = fused_preds.get(stem, [])
        gt_boxes = gt[stem]
        tp_idx, fp_idx, fn_idx = match_to_gt(pred, gt_boxes, iou_threshold)
        all_tp += len(tp_idx)
        all_fp += len(fp_idx)
        all_fn += len(fn_idx)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Simplified mAP (single-threshold AP ≈ precision at this IoU)
    # For proper mAP, use evaluate.py, but this is fast for sweeps
    return {
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ============================================================
# Experiment 1: Pairwise Fusion
# ============================================================
def experiment1_pairwise(yolo_preds, gt):
    """Test each LLM × each strategy. Core result table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Pairwise YOLO + Single LLM Fusion")
    print("=" * 70)

    # YOLO baseline
    yolo_eval = evaluate_fusion(yolo_preds, gt)
    print(f"\nYOLO baseline: P={yolo_eval['precision']:.3f} R={yolo_eval['recall']:.3f} "
          f"F1={yolo_eval['f1']:.3f} (TP={yolo_eval['tp']} FP={yolo_eval['fp']} FN={yolo_eval['fn']})")

    results = []
    print(f"\n{'LLM':<20s} {'Strategy':<12s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'ΔF1':>7s}")
    print("-" * 75)

    for llm_key in FUSION_LLMS:
        llm_preds = load_llm_predictions(llm_key)
        if not llm_preds:
            continue

        for strategy in STRATEGIES:
            fused = {}
            for stem in gt:
                yolo_boxes = yolo_preds.get(stem, [])
                llm_boxes = llm_preds.get(stem, [])
                fused[stem] = fuse_single_image(yolo_boxes, llm_boxes, strategy)

            ev = evaluate_fusion(fused, gt)
            delta_f1 = ev["f1"] - yolo_eval["f1"]
            results.append({
                "llm": llm_key, "strategy": strategy, "iou_threshold": 0.3,
                **ev, "delta_f1": round(delta_f1, 4),
            })

            marker = "↑" if delta_f1 > 0 else "↓" if delta_f1 < 0 else "="
            print(f"{llm_key:<20s} {strategy:<12s} {ev['precision']:6.3f} {ev['recall']:6.3f} "
                  f"{ev['f1']:6.3f} {ev['tp']:5d} {ev['fp']:5d} {ev['fn']:5d} "
                  f"{delta_f1:+7.4f} {marker}")

    # Save
    out_path = os.path.join(RESULT_DIR, "phase3", "e1_pairwise_fusion.json")
    with open(out_path, "w") as f:
        json.dump({"yolo_baseline": yolo_eval, "fusion_results": results}, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Experiment 2: IoU Threshold Sweep
# ============================================================
def experiment2_iou_sweep(yolo_preds, gt, top_llms=None):
    """Sweep IoU matching threshold for top LLMs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: IoU Threshold Sweep")
    print("=" * 70)

    if top_llms is None:
        top_llms = FUSION_LLMS[:3]  # top 3 by mAP

    thresholds = [round(t, 2) for t in np.arange(0.05, 0.75, 0.05)]
    results = []

    for llm_key in top_llms:
        llm_preds = load_llm_predictions(llm_key)
        if not llm_preds:
            continue

        print(f"\n--- {llm_key} ---")
        for strategy in STRATEGIES:
            for iou_t in thresholds:
                fused = {}
                for stem in gt:
                    yolo_boxes = yolo_preds.get(stem, [])
                    llm_boxes = llm_preds.get(stem, [])
                    fused[stem] = fuse_single_image(yolo_boxes, llm_boxes, strategy, match_iou=iou_t)

                ev = evaluate_fusion(fused, gt)
                results.append({
                    "llm": llm_key, "strategy": strategy,
                    "iou_threshold": iou_t, **ev,
                })

            # Print best IoU for this LLM+strategy
            strat_results = [r for r in results if r["llm"] == llm_key and r["strategy"] == strategy]
            best = max(strat_results, key=lambda x: x["f1"])
            print(f"  {strategy:<12s} best IoU={best['iou_threshold']:.2f} "
                  f"F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f}")

    out_path = os.path.join(RESULT_DIR, "phase3", "e2_iou_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Experiment 3: Complementarity Analysis
# ============================================================
def experiment3_complementarity(yolo_preds, gt):
    """Analyze where YOLO and LLMs disagree."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Complementarity Analysis")
    print("=" * 70)

    results = {}

    for llm_key in FUSION_LLMS:
        llm_preds = load_llm_predictions(llm_key)
        if not llm_preds:
            continue

        both_hit = 0      # YOLO TP and LLM TP on same GT box
        yolo_only = 0      # YOLO TP but LLM missed
        llm_only = 0       # LLM TP but YOLO missed
        both_miss = 0      # Neither caught it

        per_class = defaultdict(lambda: {"yolo_hit": 0, "llm_hit": 0, "both_hit": 0, "both_miss": 0})

        for stem in gt:
            gt_boxes = gt[stem]
            yolo_boxes = yolo_preds.get(stem, [])
            llm_boxes = llm_preds.get(stem, [])

            # Match YOLO to GT
            _, _, yolo_fn = match_to_gt(yolo_boxes, gt_boxes, 0.5)
            yolo_hit_gt = set(range(len(gt_boxes))) - set(yolo_fn)

            # Match LLM to GT
            _, _, llm_fn = match_to_gt(llm_boxes, gt_boxes, 0.5)
            llm_hit_gt = set(range(len(gt_boxes))) - set(llm_fn)

            for gi, gbox in enumerate(gt_boxes):
                cls = gbox[0]
                y_hit = gi in yolo_hit_gt
                l_hit = gi in llm_hit_gt

                if y_hit and l_hit:
                    both_hit += 1
                    per_class[cls]["both_hit"] += 1
                elif y_hit and not l_hit:
                    yolo_only += 1
                    per_class[cls]["yolo_hit"] += 1
                elif not y_hit and l_hit:
                    llm_only += 1
                    per_class[cls]["llm_hit"] += 1
                else:
                    both_miss += 1
                    per_class[cls]["both_miss"] += 1

        total = both_hit + yolo_only + llm_only + both_miss
        results[llm_key] = {
            "both_hit": both_hit,
            "yolo_only": yolo_only,
            "llm_only": llm_only,
            "both_miss": both_miss,
            "total_gt": total,
            "llm_rescue_rate": round(llm_only / total, 4) if total > 0 else 0,
            "per_class": dict(per_class),
        }

        print(f"\n{llm_key}:")
        print(f"  Both hit:    {both_hit:5d} ({both_hit/total*100:.1f}%)")
        print(f"  YOLO only:   {yolo_only:5d} ({yolo_only/total*100:.1f}%)")
        print(f"  LLM rescues: {llm_only:5d} ({llm_only/total*100:.1f}%) ← potential fusion gain")
        print(f"  Both miss:   {both_miss:5d} ({both_miss/total*100:.1f}%)")

    out_path = os.path.join(RESULT_DIR, "phase3", "e3_complementarity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Experiment 4: Improved Strategies
# ============================================================
def experiment4_improved(yolo_preds, gt):
    """Test advanced fusion strategies."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Improved Fusion Strategies")
    print("=" * 70)

    yolo_eval = evaluate_fusion(yolo_preds, gt)
    results = []

    # 4a: Architecture-specialized fusion
    # OWLv2 (high recall) for supplement, Florence-2-base (high precision) for filter
    owlv2_preds = load_llm_predictions("owlv2")
    florence_preds = load_llm_predictions("florence2_base")

    if owlv2_preds and florence_preds:
        print("\n4a: Architecture-specialized (OWLv2 supplement + Florence filter)")
        # Step 1: supplement with OWLv2 (catch YOLO misses)
        step1 = {}
        for stem in gt:
            yolo_boxes = yolo_preds.get(stem, [])
            owlv2_boxes = owlv2_preds.get(stem, [])
            step1[stem] = fuse_single_image(yolo_boxes, owlv2_boxes, "supplement", match_iou=0.3)

        # Step 2: filter with Florence (remove false positives)
        fused = {}
        for stem in gt:
            step1_boxes = step1.get(stem, [])
            florence_boxes = florence_preds.get(stem, [])
            fused[stem] = fuse_single_image(step1_boxes, florence_boxes, "filter", match_iou=0.2)

        ev = evaluate_fusion(fused, gt)
        delta = ev["f1"] - yolo_eval["f1"]
        results.append({"strategy": "specialized_owlv2+florence", **ev, "delta_f1": round(delta, 4)})
        print(f"  P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f} ΔF1={delta:+.4f}")

    # 4b: Confidence-aware supplement with different thresholds
    for llm_key in ["florence2_base", "owlv2"]:
        llm_preds = load_llm_predictions(llm_key)
        if not llm_preds:
            continue

        print(f"\n4b: Confidence-aware supplement ({llm_key})")
        for conf_t in [0.0, 0.3, 0.5, 0.7]:
            fused = {}
            for stem in gt:
                yolo_boxes = yolo_preds.get(stem, [])
                llm_boxes = llm_preds.get(stem, [])
                fused[stem] = fuse_single_image(yolo_boxes, llm_boxes, "supplement",
                                                 llm_conf_threshold=conf_t)

            ev = evaluate_fusion(fused, gt)
            delta = ev["f1"] - yolo_eval["f1"]
            results.append({
                "strategy": f"supplement_conf{conf_t}", "llm": llm_key,
                **ev, "delta_f1": round(delta, 4),
            })
            print(f"  conf≥{conf_t}: P={ev['precision']:.3f} R={ev['recall']:.3f} "
                  f"F1={ev['f1']:.3f} ΔF1={delta:+.4f}")

    out_path = os.path.join(RESULT_DIR, "phase3", "e4_improved_strategies.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Experiment 5: Multi-LLM Ensemble
# ============================================================
def experiment5_ensemble(yolo_preds, gt):
    """Test multi-LLM ensemble fusion."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multi-LLM Ensemble")
    print("=" * 70)

    yolo_eval = evaluate_fusion(yolo_preds, gt)
    results = []

    # Load all LLM predictions
    all_llm_preds = {}
    for llm_key in FUSION_LLMS:
        preds = load_llm_predictions(llm_key)
        if preds:
            all_llm_preds[llm_key] = preds

    print(f"Loaded {len(all_llm_preds)} LLM prediction sets")

    # For each image, collect all LLM detections and cluster by IoU
    for min_votes in [1, 2, 3]:
        fused = {}
        for stem in gt:
            yolo_boxes = yolo_preds.get(stem, [])

            # Collect all LLM detections
            all_llm_boxes = []
            for llm_key, preds in all_llm_preds.items():
                for box in preds.get(stem, []):
                    all_llm_boxes.append((*box, llm_key))

            # Cluster LLM detections by spatial overlap
            clusters = []
            used = set()
            for i, box_i in enumerate(all_llm_boxes):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                for j, box_j in enumerate(all_llm_boxes):
                    if j in used:
                        continue
                    iou = compute_iou(box_i[1:5], box_j[1:5])
                    if iou >= 0.3:
                        cluster.append(j)
                        used.add(j)
                clusters.append(cluster)

            # Keep clusters with >= min_votes unique models
            llm_consensus = []
            for cluster in clusters:
                models = set(all_llm_boxes[i][-1] for i in cluster)
                if len(models) >= min_votes:
                    # Use the box from the highest-mAP model in cluster
                    best_idx = cluster[0]
                    box = all_llm_boxes[best_idx]
                    llm_consensus.append(box[:6])

            # Supplement YOLO with consensus LLM detections
            fused[stem] = fuse_single_image(yolo_boxes, llm_consensus, "supplement")

        ev = evaluate_fusion(fused, gt)
        delta = ev["f1"] - yolo_eval["f1"]
        results.append({
            "strategy": f"ensemble_votes>={min_votes}",
            "num_llms": len(all_llm_preds),
            **ev, "delta_f1": round(delta, 4),
        })
        print(f"  min_votes={min_votes}: P={ev['precision']:.3f} R={ev['recall']:.3f} "
              f"F1={ev['f1']:.3f} ΔF1={delta:+.4f}")

    out_path = os.path.join(RESULT_DIR, "phase3", "e5_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Experiment 6: Statistical Significance
# ============================================================
def experiment6_statistics(yolo_preds, gt):
    """Bootstrap confidence intervals for key results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Statistical Significance (Bootstrap)")
    print("=" * 70)

    stems = sorted(gt.keys())
    n_bootstrap = 1000
    results = {}

    # Test YOLO alone and best fusion
    florence_preds = load_llm_predictions("florence2_base")
    if not florence_preds:
        print("[!] No Florence-2-base predictions, skipping")
        return {}

    configs = {
        "yolo_only": lambda stem: yolo_preds.get(stem, []),
        "yolo+florence_supplement": lambda stem: fuse_single_image(
            yolo_preds.get(stem, []), florence_preds.get(stem, []), "supplement"),
        "yolo+florence_filter": lambda stem: fuse_single_image(
            yolo_preds.get(stem, []), florence_preds.get(stem, []), "filter"),
    }

    for config_name, get_preds in configs.items():
        f1_samples = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(stems, size=len(stems), replace=True)
            sample_preds = {s: get_preds(s) for s in sample}
            sample_gt = {s: gt[s] for s in sample}
            ev = evaluate_fusion(sample_preds, sample_gt)
            f1_samples.append(ev["f1"])

        f1_arr = np.array(f1_samples)
        ci_low, ci_high = np.percentile(f1_arr, [2.5, 97.5])
        results[config_name] = {
            "mean_f1": round(float(f1_arr.mean()), 4),
            "std_f1": round(float(f1_arr.std()), 4),
            "ci_95_low": round(float(ci_low), 4),
            "ci_95_high": round(float(ci_high), 4),
        }
        print(f"  {config_name:<35s}: F1={f1_arr.mean():.4f} ± {f1_arr.std():.4f} "
              f"95%CI=[{ci_low:.4f}, {ci_high:.4f}]")

    out_path = os.path.join(RESULT_DIR, "phase3", "e6_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved to {out_path}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3: YOLO+LLM Fusion Experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--experiment", type=str, choices=["e1", "e2", "e3", "e4", "e5", "e6"],
                        help="Run specific experiment")
    args = parser.parse_args()

    if not args.all and not args.experiment:
        args.all = True

    # Step 0: Generate YOLO predictions
    yolo_pred_dir = generate_yolo_predictions()
    yolo_preds = load_yolo_predictions(yolo_pred_dir)
    gt = load_gt()
    print(f"\nLoaded: {len(yolo_preds)} YOLO predictions, {len(gt)} GT images")

    if args.all or args.experiment == "e1":
        experiment1_pairwise(yolo_preds, gt)

    if args.all or args.experiment == "e2":
        experiment2_iou_sweep(yolo_preds, gt)

    if args.all or args.experiment == "e3":
        experiment3_complementarity(yolo_preds, gt)

    if args.all or args.experiment == "e4":
        experiment4_improved(yolo_preds, gt)

    if args.all or args.experiment == "e5":
        experiment5_ensemble(yolo_preds, gt)

    if args.all or args.experiment == "e6":
        experiment6_statistics(yolo_preds, gt)

    print("\n" + "=" * 70)
    print("[+] ALL PHASE 3 EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
