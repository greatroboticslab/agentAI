"""
YOLO + LLM Fusion: Use LLM to correct/supplement YOLO weed detections.

This is the core module for your goal:
- YOLO11n detects weeds (fast but may miss some)
- LLM reviews the image and provides additional/corrected detections
- Fusion logic merges both results

Usage:
    python yolo_llm_fusion.py --image images/weed1.jpg --yolo-results yolo_output.json
    python yolo_llm_fusion.py --image images/weed1.jpg  # LLM-only mode
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from config import HF_CACHE, RESULT_DIR, WEED_DETECTION_PROMPT

os.environ["HF_HOME"] = HF_CACHE


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2] in percentage coords."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def load_yolo_results(yolo_path):
    """
    Load YOLO detection results.
    Expected format (from ultralytics):
    [{"class": 0, "name": "weed", "confidence": 0.85, "box": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}}]

    Or normalized format:
    [{"label": "weed", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}]
    """
    with open(yolo_path) as f:
        data = json.load(f)

    detections = []
    for d in data:
        if "box" in d:
            # Ultralytics format
            box = d["box"]
            detections.append({
                "label": d.get("name", "weed"),
                "confidence": d.get("confidence", 0),
                "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "source": "yolo",
            })
        elif "bbox" in d:
            d["source"] = "yolo"
            detections.append(d)
        else:
            detections.append(d)

    return detections


def convert_yolo_bbox_to_percent(bbox, img_w, img_h):
    """Convert pixel coordinates to percentage."""
    return [
        bbox[0] / img_w * 100,
        bbox[1] / img_h * 100,
        bbox[2] / img_w * 100,
        bbox[3] / img_h * 100,
    ]


def get_llm_detections(image_path, model_key="qwen7b"):
    """Get weed detections from LLM."""
    from test_hf_models import MODEL_SHORTCUTS, extract_json

    info = MODEL_SHORTCUTS[model_key]
    model, processor = info["loader"](info["full_name"])

    # Enhanced prompt that asks LLM to be thorough
    fusion_prompt = """Analyze this agricultural field image very carefully for weed detection.

You are being used as a second opinion after an initial detection system. Please be especially thorough:
1. Look for small weeds that might be partially hidden
2. Check edges and corners of the image
3. Identify any plants that look different from the main crop
4. Note any areas with unusual growth patterns

For each detection, provide:
- "label": "weed" or "crop"
- "species": specific species if identifiable
- "confidence": "high", "medium", or "low"
- "bbox": [x_min, y_min, x_max, y_max] as percentage (0-100) of image dimensions
- "description": brief visual description

Return ONLY valid JSON:
{
  "detections": [...],
  "scene_description": "...",
  "weed_severity": "none|low|medium|high",
  "crop_type": "..."
}"""

    response = info["inferencer"](model, processor, image_path, fusion_prompt)
    parsed = extract_json(response)

    if parsed and "detections" in parsed:
        for d in parsed["detections"]:
            d["source"] = "llm"
        return parsed["detections"]
    return []


def fuse_detections(yolo_dets, llm_dets, iou_threshold=0.3):
    """
    Fuse YOLO and LLM detections.

    Strategy:
    1. Keep all YOLO detections (they have precise bbox from trained model)
    2. For each LLM detection:
       - If it overlaps with a YOLO detection (IoU > threshold): merge metadata
       - If it doesn't overlap: add as new detection (LLM found something YOLO missed)
    3. Mark disagreements for review
    """
    fused = []
    llm_matched = set()

    # First pass: process YOLO detections and find LLM matches
    for yd in yolo_dets:
        y_bbox = yd.get("bbox", [0, 0, 0, 0])
        best_iou = 0
        best_llm_idx = -1

        for i, ld in enumerate(llm_dets):
            l_bbox = ld.get("bbox", [0, 0, 0, 0])
            iou = compute_iou(y_bbox, l_bbox)
            if iou > best_iou:
                best_iou = iou
                best_llm_idx = i

        merged = dict(yd)
        merged["source"] = "yolo"

        if best_iou > iou_threshold and best_llm_idx >= 0:
            llm_matched.add(best_llm_idx)
            llm_det = llm_dets[best_llm_idx]

            # Merge: keep YOLO bbox (more precise), add LLM species info
            merged["source"] = "yolo+llm"
            merged["iou_with_llm"] = round(best_iou, 3)
            if llm_det.get("species"):
                merged["species_llm"] = llm_det["species"]
            if llm_det.get("description"):
                merged["description_llm"] = llm_det["description"]

            # Agreement check
            yolo_label = yd.get("label", "").lower()
            llm_label = llm_det.get("label", "").lower()
            if "weed" in yolo_label and "weed" in llm_label:
                merged["agreement"] = "both_weed"
            elif "weed" not in yolo_label and "weed" not in llm_label:
                merged["agreement"] = "both_crop"
            else:
                merged["agreement"] = "disagree"
                merged["needs_review"] = True

        fused.append(merged)

    # Second pass: add LLM-only detections (YOLO missed these)
    for i, ld in enumerate(llm_dets):
        if i not in llm_matched:
            new_det = dict(ld)
            new_det["source"] = "llm_only"
            new_det["note"] = "Detected by LLM but not by YOLO - potential miss"
            fused.append(new_det)

    return fused


def fuse_dataset(yolo_dir, llm_dir, output_dir, iou_threshold=0.3, strategy="supplement"):
    """Batch fusion: fuse YOLO and LLM detections for an entire dataset.

    Args:
        yolo_dir: Directory with YOLO detection JSONs or single JSON file.
        llm_dir: Directory with LLM detection JSONs or single JSON file.
        output_dir: Output directory for fused results.
        iou_threshold: IoU threshold for matching.
        strategy: Fusion strategy - "supplement", "filter", or "weighted".

    Strategies:
        supplement: Add LLM-only detections to YOLO (improves recall)
        filter: Only keep YOLO detections confirmed by LLM (improves precision)
        weighted: Combine YOLO confidence + LLM confidence
    """
    import glob as glob_mod

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    def load_results(path):
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        results = []
        for jf in sorted(glob_mod.glob(os.path.join(path, "*.json"))):
            with open(jf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        return results

    yolo_results = load_results(yolo_dir)
    llm_results = load_results(llm_dir)

    # Build per-image lookup
    yolo_by_image = {}
    for r in yolo_results:
        stem = Path(r.get("image", "")).stem
        yolo_by_image[stem] = r.get("detections", [])

    llm_by_image = {}
    for r in llm_results:
        stem = Path(r.get("image", "")).stem
        llm_by_image[stem] = r.get("detections", [])

    all_images = sorted(set(yolo_by_image.keys()) | set(llm_by_image.keys()))
    fused_results = []
    stats = {"total": 0, "yolo_only": 0, "llm_only": 0, "both": 0, "filtered": 0}

    for img_stem in all_images:
        yolo_dets = yolo_by_image.get(img_stem, [])
        llm_dets = llm_by_image.get(img_stem, [])

        if strategy == "supplement":
            fused = fuse_detections(yolo_dets, llm_dets, iou_threshold)
        elif strategy == "filter":
            # Only keep YOLO detections that LLM also confirms
            fused = []
            for yd in yolo_dets:
                y_bbox = yd.get("bbox", [0, 0, 0, 0])
                confirmed = False
                for ld in llm_dets:
                    l_bbox = ld.get("bbox", [0, 0, 0, 0])
                    iou = compute_iou(y_bbox, l_bbox)
                    if iou > iou_threshold:
                        confirmed = True
                        break
                if confirmed:
                    fused.append(yd)
                else:
                    stats["filtered"] += 1
        elif strategy == "weighted":
            # Combine confidence scores
            fused = fuse_detections(yolo_dets, llm_dets, iou_threshold)
            for det in fused:
                yolo_conf = det.get("confidence", 0.5)
                if isinstance(yolo_conf, str):
                    yolo_conf = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(yolo_conf, 0.5)
                llm_conf = det.get("iou_with_llm", 0)
                # Weighted average: 70% YOLO, 30% LLM overlap
                det["confidence_fused"] = round(0.7 * float(yolo_conf) + 0.3 * float(llm_conf), 3)
        else:
            fused = fuse_detections(yolo_dets, llm_dets, iou_threshold)

        # Count stats
        for d in fused:
            stats["total"] += 1
            src = d.get("source", "")
            if src == "yolo":
                stats["yolo_only"] += 1
            elif src == "llm_only":
                stats["llm_only"] += 1
            elif src == "yolo+llm":
                stats["both"] += 1

        fused_results.append({
            "image": f"{img_stem}.jpg",
            "num_detections": len(fused),
            "detections": fused,
        })

    # Save fused results
    output_path = os.path.join(output_dir, "fused_results.json")
    with open(output_path, "w") as f:
        json.dump(fused_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"BATCH FUSION COMPLETE (strategy: {strategy})")
    print(f"{'='*50}")
    print(f"  Images processed:  {len(all_images)}")
    print(f"  Total detections:  {stats['total']}")
    print(f"  YOLO only:         {stats['yolo_only']}")
    print(f"  LLM only:          {stats['llm_only']}")
    print(f"  Both agree:        {stats['both']}")
    if strategy == "filter":
        print(f"  Filtered out:      {stats['filtered']}")
    print(f"  Output:            {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO + LLM fusion for weed detection")
    parser.add_argument("--image", type=str, help="Image path (single image mode)")
    parser.add_argument("--yolo-results", type=str, default=None, help="YOLO results JSON file")
    parser.add_argument("--model", type=str, default="qwen7b", help="LLM model to use")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for matching")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    # Batch mode arguments
    parser.add_argument("--batch", action="store_true", help="Batch mode: fuse entire dataset")
    parser.add_argument("--yolo-dir", type=str, help="YOLO results directory/file (batch mode)")
    parser.add_argument("--llm-dir", type=str, help="LLM results directory/file (batch mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory (batch mode)")
    parser.add_argument("--strategy", type=str, default="supplement",
                        choices=["supplement", "filter", "weighted"],
                        help="Fusion strategy (batch mode)")
    args = parser.parse_args()

    # Batch mode
    if args.batch:
        if not args.yolo_dir or not args.llm_dir:
            parser.error("Batch mode requires --yolo-dir and --llm-dir")
        output_dir = args.output_dir or os.path.join(RESULT_DIR, "fusion_batch")
        fuse_dataset(args.yolo_dir, args.llm_dir, output_dir,
                     args.iou_threshold, args.strategy)
        return

    if not args.image:
        parser.error("Single image mode requires --image")

    image_path = args.image
    img = Image.open(image_path)
    img_w, img_h = img.size
    print(f"[*] Image: {image_path} ({img_w}x{img_h})")

    # Load YOLO results
    yolo_dets = []
    if args.yolo_results:
        print(f"[*] Loading YOLO results from {args.yolo_results}")
        yolo_dets = load_yolo_results(args.yolo_results)
        # Convert pixel coords to percentage if needed
        for d in yolo_dets:
            bbox = d.get("bbox", [0, 0, 0, 0])
            if any(v > 100 for v in bbox):
                d["bbox"] = convert_yolo_bbox_to_percent(bbox, img_w, img_h)
        print(f"    Found {len(yolo_dets)} YOLO detections")
    else:
        print("[*] No YOLO results provided - running LLM-only mode")

    # Get LLM detections
    print(f"[*] Getting LLM detections with model: {args.model}")
    start = time.time()
    llm_dets = get_llm_detections(image_path, args.model)
    llm_time = time.time() - start
    print(f"    Found {len(llm_dets)} LLM detections ({llm_time:.1f}s)")

    # Fuse results
    if yolo_dets:
        print(f"[*] Fusing YOLO + LLM detections (IoU threshold: {args.iou_threshold})")
        fused = fuse_detections(yolo_dets, llm_dets, args.iou_threshold)
    else:
        fused = llm_dets

    # Summarize
    yolo_only = [d for d in fused if d.get("source") == "yolo"]
    llm_only = [d for d in fused if d.get("source") == "llm_only"]
    both = [d for d in fused if d.get("source") == "yolo+llm"]
    disagree = [d for d in fused if d.get("agreement") == "disagree"]

    print(f"\n{'='*50}")
    print("FUSION RESULTS")
    print(f"{'='*50}")
    print(f"Total detections:      {len(fused)}")
    print(f"YOLO only:             {len(yolo_only)}")
    print(f"LLM only (YOLO missed):{len(llm_only)}")
    print(f"Both agreed:           {len(both)}")
    print(f"Disagreements:         {len(disagree)}")

    if llm_only:
        print(f"\n--- LLM Found (YOLO Missed) ---")
        for d in llm_only:
            print(f"  {d.get('label')}: bbox={d.get('bbox')} species={d.get('species','?')}")

    if disagree:
        print(f"\n--- Disagreements (Need Review) ---")
        for d in disagree:
            print(f"  YOLO={d.get('label')} vs LLM species={d.get('species_llm','?')} bbox={d.get('bbox')}")

    # Save output
    output = {
        "image": os.path.basename(image_path),
        "image_size": [img_w, img_h],
        "model": args.model,
        "yolo_count": len(yolo_dets),
        "llm_count": len(llm_dets),
        "fused_count": len(fused),
        "llm_only_count": len(llm_only),
        "disagree_count": len(disagree),
        "detections": fused,
    }

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(RESULT_DIR, f"fusion_{Path(image_path).stem}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[+] Saved to {out_path}")


if __name__ == "__main__":
    main()
