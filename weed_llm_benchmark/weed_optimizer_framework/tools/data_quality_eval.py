"""
v3.0.35 — Data quality benchmark: OWLv2 vs Gemma 4 vision on cwd12 holdout.

Decides v3.0.36 architecture: assign each model the role it's actually
good at, instead of either dropping OWLv2 or fine-tuning Gemma 4 for bbox.

Four tests on cwd12 holdout (1977 imgs with hand-labeled GT bboxes):

  T1: OWLv2 baseline (current zero-shot, prompt="weed")
      → precision / recall / F1 @ IoU=0.5 vs GT bboxes
      → baseline number for everything else

  T2: Gemma 4 direct bbox output
      → ask Gemma to return [x1,y1,x2,y2] coords for every weed
      → parse + measure same metrics
      → expected POOR (VLMs weak at coord regression)

  T3: Gemma 4 image-level relevance
      → "Is this a top-down field photo of weeds/crops? yes/no"
      → all cwd12 = yes (true positive rate)
      → mix in some flagged-garbage images for negative test
      → accuracy

  T4: Gemma 4 bbox verification (the interesting one)
      → take OWLv2 candidate boxes
      → for each box: crop region + ask "Is this a weed?"
      → does Gemma correctly accept TP boxes and reject FP boxes?
      → → measures whether Gemma 4 can be a downstream filter

Run on ~200 sampled cwd12 holdout images (not full 1977 — Gemma 4 is slow;
calibration sample is enough for first decision).
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def iou_xyxy(a, b):
    """IoU between one box and one box, both (x1,y1,x2,y2)."""
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def precision_recall_f1(pred_boxes_per_img, gt_boxes_per_img, iou_thr=0.5):
    """Compute aggregate precision/recall/F1 across all images.

    pred_boxes_per_img: list of [(box, score), ...] per image
    gt_boxes_per_img: list of [box, ...] per image
    """
    tp = 0; fp = 0; fn = 0
    for preds, gts in zip(pred_boxes_per_img, gt_boxes_per_img):
        # Sort preds by score descending
        preds_sorted = sorted(preds, key=lambda p: -p[1])
        gt_matched = [False] * len(gts)
        for pbox, _ in preds_sorted:
            best_g = -1; best_iou = iou_thr
            for g_idx, gbox in enumerate(gts):
                if gt_matched[g_idx]:
                    continue
                io = iou_xyxy(pbox, gbox)
                if io >= best_iou:
                    best_iou = io
                    best_g = g_idx
            if best_g >= 0:
                tp += 1
                gt_matched[best_g] = True
            else:
                fp += 1
        fn += sum(1 for m in gt_matched if not m)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1, tp, fp, fn


def load_cwd12_holdout(cwd12_root, sample_n=None, seed=42):
    """Load cwd12 holdout (test+valid). Returns list of dicts with:
       {img_path, w, h, gt_boxes [(x1,y1,x2,y2)...], gt_classes [...]}
    """
    samples = []
    for split in ("test", "valid"):
        img_dir = Path(cwd12_root) / split / "images"
        lbl_dir = Path(cwd12_root) / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            with Image.open(img) as im:
                w, h = im.size
            lbl = lbl_dir / (img.stem + ".txt")
            gt_boxes = []
            gt_classes = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    try:
                        cls = int(p[0])
                        cx, cy, bw, bh = map(float, p[1:5])
                    except ValueError:
                        continue
                    x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w; y2 = (cy + bh/2) * h
                    gt_boxes.append((x1, y1, x2, y2))
                    gt_classes.append(cls)
            samples.append({
                "img_path": str(img), "w": w, "h": h,
                "gt_boxes": gt_boxes, "gt_classes": gt_classes,
            })
    if sample_n and sample_n < len(samples):
        random.seed(seed)
        samples = random.sample(samples, sample_n)
    return samples


# -----------------------------------------------------------------------------
# T1: OWLv2 baseline
# -----------------------------------------------------------------------------
def run_t1_owlv2_baseline(samples, conf=0.30, prompt="weed"):
    """OWLv2 zero-shot with given prompt + conf threshold."""
    print(f"\n=== T1: OWLv2 baseline (conf={conf}, prompt={prompt!r}) ===")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import torch

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).cuda().eval()

    pred_per_img = []
    gt_per_img = []
    t0 = time.time()
    for i, s in enumerate(samples):
        im = Image.open(s["img_path"]).convert("RGB")
        inputs = proc(text=[[prompt]], images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]]).to("cuda")
        # transformers renamed: post_process_object_detection -> post_process_grounded_object_detection
        # new signature requires text_labels parameter
        results = proc.post_process_grounded_object_detection(
            out, target_sizes=target_sizes, threshold=conf,
            text_labels=[[prompt]],
        )[0]
        preds = []
        for box, score in zip(results["boxes"].cpu().numpy(),
                              results["scores"].cpu().numpy()):
            preds.append((tuple(box.tolist()), float(score)))
        pred_per_img.append(preds)
        gt_per_img.append(s["gt_boxes"])
        if (i + 1) % 50 == 0:
            print(f"  T1 {i+1}/{len(samples)} ({(time.time()-t0):.0f}s)")

    p, r, f1, tp, fp, fn = precision_recall_f1(pred_per_img, gt_per_img)
    print(f"  T1 RESULT: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (TP={tp} FP={fp} FN={fn})")
    return {"test": "T1_owlv2_baseline", "conf": conf, "prompt": prompt,
            "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_images": len(samples)}


# -----------------------------------------------------------------------------
# Gemma 4 helper (via Ollama HTTP API)
# -----------------------------------------------------------------------------
def gemma4_vision_call(prompt, image_path, model="gemma4", url="http://127.0.0.1:11434"):
    """Call Ollama Gemma 4 with image. Returns response text."""
    import requests
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"model": model, "prompt": prompt, "images": [b64], "stream": False}
    r = requests.post(f"{url}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


# -----------------------------------------------------------------------------
# T2: Gemma 4 direct bbox output
# -----------------------------------------------------------------------------
def run_t2_gemma_bbox(samples):
    """Ask Gemma to output bbox coordinates."""
    print(f"\n=== T2: Gemma 4 direct bbox output ===")
    prompt = (
        "This image is a top-down field photo. Identify every weed plant "
        "visible. For each weed, output one line in the format: "
        "x1,y1,x2,y2 (pixel coordinates, image is {w}x{h}). "
        "Output ONLY the coordinate lines, nothing else."
    )

    pred_per_img = []
    gt_per_img = []
    t0 = time.time()
    for i, s in enumerate(samples):
        p_text = prompt.format(w=s["w"], h=s["h"])
        try:
            resp = gemma4_vision_call(p_text, s["img_path"])
        except Exception as e:
            print(f"  T2 {i} call failed: {e}")
            pred_per_img.append([])
            gt_per_img.append(s["gt_boxes"])
            continue
        # Parse bbox lines
        preds = []
        for line in resp.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) != 4: continue
            try:
                coords = tuple(float(x.strip()) for x in parts)
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    preds.append((coords, 1.0))  # Gemma doesn't give scores
            except ValueError:
                continue
        pred_per_img.append(preds)
        gt_per_img.append(s["gt_boxes"])
        if (i + 1) % 20 == 0:
            print(f"  T2 {i+1}/{len(samples)} ({(time.time()-t0):.0f}s)")

    p, r, f1, tp, fp, fn = precision_recall_f1(pred_per_img, gt_per_img)
    print(f"  T2 RESULT: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (TP={tp} FP={fp} FN={fn})")
    return {"test": "T2_gemma4_bbox", "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_images": len(samples)}


# -----------------------------------------------------------------------------
# T3: Gemma 4 image-level relevance (yes/no classifier)
# -----------------------------------------------------------------------------
def run_t3_gemma_relevance(positive_samples, negative_samples):
    """positive_samples = cwd12 holdout (should answer yes)
    negative_samples = list of (img_path) from flagged-garbage slugs (should answer no)
    """
    print(f"\n=== T3: Gemma 4 image relevance ({len(positive_samples)} +, "
          f"{len(negative_samples)} -) ===")
    prompt = (
        "Look at this image. Is it a top-down field photo containing weeds "
        "or crops (agricultural plants on soil)? "
        "Answer with ONLY one word: yes or no."
    )

    correct = 0; total = 0; tp = 0; tn = 0; fp_err = 0; fn_err = 0
    t0 = time.time()
    for s in positive_samples:
        try:
            resp = gemma4_vision_call(prompt, s["img_path"]).strip().lower()
        except Exception as e:
            print(f"  T3+ call failed: {e}"); continue
        ans = "yes" if "yes" in resp[:10] else "no" if "no" in resp[:10] else "?"
        total += 1
        if ans == "yes": tp += 1; correct += 1
        else: fn_err += 1
    for img_path in negative_samples:
        try:
            resp = gemma4_vision_call(prompt, img_path).strip().lower()
        except Exception as e:
            print(f"  T3- call failed: {e}"); continue
        ans = "yes" if "yes" in resp[:10] else "no" if "no" in resp[:10] else "?"
        total += 1
        if ans == "no": tn += 1; correct += 1
        else: fp_err += 1

    acc = correct / total if total > 0 else 0
    print(f"  T3 RESULT: acc={acc:.4f}  TP={tp} TN={tn} FP_err={fp_err} FN_err={fn_err}")
    print(f"  ({(time.time()-t0):.0f}s for {total} calls)")
    return {"test": "T3_gemma4_relevance", "accuracy": acc, "tp": tp, "tn": tn,
            "fp_err": fp_err, "fn_err": fn_err, "n_pos": len(positive_samples),
            "n_neg": len(negative_samples)}


# -----------------------------------------------------------------------------
# T4: Gemma 4 bbox verification (crop OWLv2 output + ask)
# -----------------------------------------------------------------------------
def run_t4_gemma_verify(samples, conf=0.30):
    """For each OWLv2 candidate box: crop + ask Gemma if it's a weed.
    Measure how well Gemma rejects FP and accepts TP.
    """
    print(f"\n=== T4: Gemma 4 bbox verification (after OWLv2 conf={conf}) ===")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import torch

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).cuda().eval()

    prompt = ("This is a cropped region from a field photo. Does it contain "
              "a weed or crop plant? Answer ONLY: yes or no.")

    n_tp_owl = 0; n_fp_owl = 0
    n_tp_kept = 0; n_fp_kept = 0  # after Gemma verify
    n_tp_dropped = 0; n_fp_dropped = 0

    tmp_dir = Path("/tmp/v3_0_35_crops"); tmp_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(samples):
        im = Image.open(s["img_path"]).convert("RGB")
        inputs = proc(text=[["weed"]], images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]]).to("cuda")
        # transformers renamed: post_process_object_detection -> post_process_grounded_object_detection
        # new signature requires text_labels parameter
        results = proc.post_process_grounded_object_detection(
            out, target_sizes=target_sizes, threshold=conf,
            text_labels=[[prompt]],
        )[0]
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()

        for j, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(s["w"], x2); y2 = min(s["h"], y2)
            if x2 <= x1 or y2 <= y1: continue

            # Is this OWLv2 prediction a TP or FP per GT?
            is_tp = any(iou_xyxy(box, gt) >= 0.5 for gt in s["gt_boxes"])
            if is_tp: n_tp_owl += 1
            else: n_fp_owl += 1

            # Crop + ask Gemma
            crop = im.crop((x1, y1, x2, y2))
            crop_path = tmp_dir / f"crop_{i}_{j}.jpg"
            crop.save(crop_path, quality=90)
            try:
                resp = gemma4_vision_call(prompt, crop_path).strip().lower()
            except Exception as e:
                continue
            finally:
                try: crop_path.unlink()
                except: pass
            keep = "yes" in resp[:10]
            if keep:
                if is_tp: n_tp_kept += 1
                else: n_fp_kept += 1
            else:
                if is_tp: n_tp_dropped += 1
                else: n_fp_dropped += 1
        if (i + 1) % 10 == 0:
            print(f"  T4 {i+1}/{len(samples)} ({(time.time()-t0):.0f}s)")

    # OWLv2 alone precision = TP / (TP+FP)
    p_owl = n_tp_owl / max(n_tp_owl + n_fp_owl, 1)
    # After Gemma verify
    p_after = n_tp_kept / max(n_tp_kept + n_fp_kept, 1)
    # Recall preservation (kept TPs / total TPs)
    r_preserved = n_tp_kept / max(n_tp_owl, 1)
    # FP rejection rate
    fp_rejected = n_fp_dropped / max(n_fp_owl, 1)
    print(f"  T4 RESULT:")
    print(f"    OWLv2 alone:  P={p_owl:.4f}  (TP={n_tp_owl}, FP={n_fp_owl})")
    print(f"    After Gemma:  P={p_after:.4f}  (kept TP={n_tp_kept}, kept FP={n_fp_kept})")
    print(f"    TP recall preserved: {r_preserved:.4f}")
    print(f"    FP rejection rate:   {fp_rejected:.4f}")
    return {"test": "T4_gemma4_verify", "n_images": len(samples), "conf": conf,
            "owlv2_alone_precision": p_owl,
            "after_gemma_precision": p_after,
            "tp_recall_preserved": r_preserved,
            "fp_rejection_rate": fp_rejected,
            "tp_owl": n_tp_owl, "fp_owl": n_fp_owl,
            "tp_kept": n_tp_kept, "fp_kept": n_fp_kept,
            "tp_dropped": n_tp_dropped, "fp_dropped": n_fp_dropped}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd12", default="downloads/cottonweeddet12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-n", type=int, default=100,
                    help="cwd12 sample size for Gemma tests (1977 max)")
    ap.add_argument("--owlv2-conf", type=float, default=0.30)
    ap.add_argument("--owlv2-prompt", default="weed")
    ap.add_argument("--tests", default="T1,T2,T3,T4",
                    help="which tests to run, comma-separated")
    ap.add_argument("--negative-dir", default=None,
                    help="dir of non-weed images for T3 negative samples "
                         "(default: pick from flagged-garbage slug if exists)")
    args = ap.parse_args()

    out_root = Path(args.out).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    tests = set(args.tests.split(","))

    print(f"=== v3.0.35 data quality benchmark ===")
    print(f"  cwd12 root: {args.cwd12}")
    print(f"  sample N (for Gemma tests): {args.sample_n}")
    print(f"  tests: {tests}")

    # Load cwd12 holdout
    print(f"\n[load] loading cwd12 holdout...")
    full_samples = load_cwd12_holdout(args.cwd12)
    print(f"  full holdout: {len(full_samples)} images")
    sampled = load_cwd12_holdout(args.cwd12, sample_n=args.sample_n, seed=42)
    print(f"  Gemma sample: {len(sampled)} images")

    results = {}

    # T1 on full holdout (OWLv2 is fast)
    if "T1" in tests:
        results["T1"] = run_t1_owlv2_baseline(
            full_samples, conf=args.owlv2_conf, prompt=args.owlv2_prompt
        )

    # T2 on sample (Gemma slow)
    if "T2" in tests:
        results["T2"] = run_t2_gemma_bbox(sampled)

    # T3 needs negatives
    if "T3" in tests:
        REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
        neg_paths = []

        # v3.0.35.2: better negative sourcing strategy. Prior version tried
        # flagged-garbage slug downloads/ paths but those datasets weren't
        # on disk (Brain dropped them at harvest, not download). Now try:
        # 1. Explicit --negative-dir if user provided
        # 2. Registry slugs whose names look "UNCATEGORIZED" (real noise:
        #    commonforms, mytwu, yonder, colo — these ARE downloaded)
        # 3. Fallback: any registry slug NOT in our TRUSTED reference set
        if args.negative_dir and os.path.isdir(args.negative_dir):
            neg_paths = sorted(str(p) for p in Path(args.negative_dir).glob("*.jpg"))[:50]
            print(f"  T3 negatives source: --negative-dir ({len(neg_paths)})")
        else:
            # Try UNCATEGORIZED slugs (downloaded but non-agricultural)
            UNCAT_KEYWORDS = ["commonforms", "mytwu", "yonder", "colo",
                              "warp", "beehive", "plantifydr", "plantdoc",
                              "plants-classification"]
            try:
                reg = json.load(open(f"{REPO}/results/framework/dataset_registry.json"))
                for slug, info in reg.get("datasets", {}).items():
                    if not isinstance(info, dict): continue
                    sl = slug.lower()
                    if not any(k in sl for k in UNCAT_KEYWORDS): continue
                    lp = info.get("local_path") or ""
                    if not lp or not os.path.isdir(lp):
                        # Also try downloads/<slug>
                        lp = f"{REPO}/downloads/{slug}"
                        if not os.path.isdir(lp): continue
                    for img in Path(lp).rglob("*"):
                        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            neg_paths.append(str(img))
                            if len(neg_paths) >= 50: break
                    if len(neg_paths) >= 50: break
            except Exception as e:
                print(f"  T3 negatives registry scan failed: {e}")
            print(f"  T3 negatives source: UNCATEGORIZED slugs ({len(neg_paths)})")

        if not neg_paths:
            print(f"  T3 WARN: no negative images found — T3 will only test positives")
        results["T3"] = run_t3_gemma_relevance(sampled, neg_paths)

    # T4 on sample (uses both OWLv2 + Gemma)
    if "T4" in tests:
        results["T4"] = run_t4_gemma_verify(sampled, conf=args.owlv2_conf)

    # Decision summary
    print("\n" + "="*60)
    print("=== DECISION SUMMARY ===")
    print("="*60)
    if "T1" in results:
        t1 = results["T1"]
        print(f"OWLv2 alone (full {t1['n_images']} imgs, conf={t1['conf']}):")
        print(f"  P={t1['precision']:.4f} R={t1['recall']:.4f} F1={t1['f1']:.4f}")
    if "T2" in results:
        t2 = results["T2"]
        print(f"Gemma 4 direct bbox ({t2['n_images']} imgs):")
        print(f"  P={t2['precision']:.4f} R={t2['recall']:.4f} F1={t2['f1']:.4f}")
        if "T1" in results:
            verdict = "BETTER than OWLv2" if t2['f1'] > results['T1']['f1'] else "WORSE than OWLv2 (expected)"
            print(f"  → {verdict}")
    if "T3" in results:
        t3 = results["T3"]
        print(f"Gemma 4 relevance: accuracy={t3['accuracy']:.4f}")
        verdict = "USEFUL as image filter" if t3['accuracy'] > 0.85 else "NOT RELIABLE"
        print(f"  → {verdict}")
    if "T4" in results:
        t4 = results["T4"]
        print(f"Gemma 4 verify:")
        print(f"  OWLv2 alone P={t4['owlv2_alone_precision']:.4f}")
        print(f"  After Gemma P={t4['after_gemma_precision']:.4f}")
        print(f"  TP preserved={t4['tp_recall_preserved']:.4f} "
              f"FP rejected={t4['fp_rejection_rate']:.4f}")
        boost = t4['after_gemma_precision'] - t4['owlv2_alone_precision']
        cost = 1 - t4['tp_recall_preserved']
        verdict = "USEFUL verifier" if boost > 0.05 and cost < 0.10 else "NOT WORTH overhead"
        print(f"  → precision +{boost:.3f}, recall cost {cost:.3f} → {verdict}")

    out_path = out_root / "v3_0_35_quality_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
