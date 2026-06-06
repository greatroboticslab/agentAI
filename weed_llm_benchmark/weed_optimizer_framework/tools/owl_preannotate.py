"""
owl_preannotate.py — OWLv2 image-conditioned bounding-box pre-annotation
for the active-learning loop.

Role in the active-learning pipeline (see
memory/project_roboflow_pipeline_plan.md):

    1. Human draws a small seed set per species  →  GREEN (gold)
    2. OWLv2 sees the green crops as VISUAL QUERIES → proposes boxes on
       target unlabeled images → writes YOLO txt with cid=0 (single-species)
       and tag `red` (model-proposed, awaiting human approval)        ← THIS
    3. Human approves/corrects red → flips to green
    4. As green grows, swap OWLv2 for a YOLO/RF-DETR fine-tuned on green
       (better than OWLv2 on fine-grained weeds — known weakness)
    5. Iterate; auto-label precision tracked as paper metric.

This module = step 2 only. Read-only on the target dir; writes YOLO txt
labels to --out-dir/<image>.txt. Does NOT upload to Roboflow (the sync
step is separate, via roboflow_sync.py species-upload with batch=red).

SECURITY: no secrets used. Read-only on inputs.

Requirements: torch, transformers (Owlv2ForObjectDetection,
Owlv2Processor), Pillow. Available in cluster `bench` env.

GPU: OWLv2-large needs ~6GB GPU. Wire via sbatch:
    run_v3_0_50_owl_preannotate.sh  (planned).

Usage:
    python -m weed_optimizer_framework.tools.owl_preannotate \\
        --target-dir <unlabeled imgs> \\
        --exemplar-config <species exemplars json> \\
        --species Goosegrass \\
        --out-dir <labels output> \\
        --conf-threshold 0.3 \\
        --max-images 100

Exemplar-config JSON format (one species per file, or aggregated):
    {
      "species": "Goosegrass",
      "exemplars": [
        {"image": "/abs/path.jpg", "bbox_yolo": [0.5, 0.5, 0.2, 0.3]},
        ...   # 3-10 examples ideal
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))

# CWD12 schema (mirrors roboflow_sync.CWD12)
CWD12 = (
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
)


def _crop_yolo_box(img, box_yolo):
    """Crop a PIL Image to the YOLO normalized bbox (cx, cy, w, h)."""
    W, H = img.size
    cx, cy, w, h = box_yolo
    x0 = max(0, int((cx - w / 2) * W))
    y0 = max(0, int((cy - h / 2) * H))
    x1 = min(W, int((cx + w / 2) * W))
    y1 = min(H, int((cy + h / 2) * H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return img.crop((x0, y0, x1, y1))


def _load_owl(model_id: str = "google/owlv2-large-patch14-ensemble"):
    """Lazy load OWLv2 (image-conditioned variant supports image queries)."""
    print(f"[owl] loading {model_id} ...")
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[owl] WARNING no GPU — inference will be slow", file=sys.stderr)
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model, device


def _nms(boxes, scores, iou_thr):
    """Greedy NMS. boxes = list of (x0,y0,x1,y1), scores = list of float.
    Returns kept indices (highest score first)."""
    idx = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
        return inter / (aa + ab - inter + 1e-9)
    while idx:
        i = idx.pop(0)
        keep.append(i)
        idx = [j for j in idx if iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def _process_targets(target_imgs, exemplar_crops, species_name: str,
                     processor, model, device, conf_threshold: float,
                     nms_iou: float = 0.5, top_k: int = 30):
    """For each target image, run OWLv2 image-conditioned detection with
    exemplar crops as visual queries. Yield (image_path, [yolo_lines]).

    v3.0.63 (button-test iter 12): Bug 3 fix — target_sizes batch dim == n_queries.
    v3.0.88 (P2): the N exemplar queries each return ~100+ boxes for the SAME
    target image → merging them gave ~600 boxes/image (precision 0.0001, useless
    for review). Now apply cross-query NMS (iou=nms_iou) + per-image top_k cap so
    proposals are actually reviewable.
    """
    import torch
    from PIL import Image
    n_queries = len(exemplar_crops)
    for tp in target_imgs:
        try:
            img = Image.open(tp).convert("RGB")
        except Exception as e:
            print(f"[owl] open fail {tp}: {e}", file=sys.stderr)
            continue
        W, H = img.size
        with torch.no_grad():
            inputs = processor(
                images=img, query_images=exemplar_crops, return_tensors="pt"
            ).to(device)
            outputs = model.image_guided_detection(**inputs)
            # target_sizes batch dim must match logits batch (== n_queries).
            target_sizes = torch.tensor(
                [img.size[::-1]] * n_queries
            ).to(device)
            results_per_query = processor.post_process_image_guided_detection(
                outputs=outputs, threshold=conf_threshold,
                target_sizes=target_sizes,
            )
        # results_per_query is a list of N dicts (one per query image). They all
        # describe the SAME target image. Collect ALL boxes, then NMS + top_k so
        # duplicates across queries collapse and the file is reviewable.
        cand_boxes = []   # (x0,y0,x1,y1) pixel
        cand_scores = []
        n_dropped = 0
        for r in results_per_query:
            for box, score in zip(r["boxes"], r["scores"]):
                x0, y0, x1, y1 = [float(v) for v in box]
                # Clamp to image (OWL post-process can return out-of-bounds).
                x0 = max(0.0, min(W, x0)); x1 = max(0.0, min(W, x1))
                y0 = max(0.0, min(H, y0)); y1 = max(0.0, min(H, y1))
                if x1 <= x0 or y1 <= y0:
                    n_dropped += 1
                    continue
                cand_boxes.append((x0, y0, x1, y1))
                cand_scores.append(float(score))
        # v3.0.88: cross-query NMS + top_k cap.
        keep = _nms(cand_boxes, cand_scores, nms_iou)
        keep = keep[:top_k] if top_k and top_k > 0 else keep
        yolo_lines = []
        for i in keep:
            x0, y0, x1, y1 = cand_boxes[i]
            cx = ((x0 + x1) / 2) / W
            cy = ((y0 + y1) / 2) / H
            w = (x1 - x0) / W
            h = (y1 - y0) / H
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}  # "
                              f"red conf={cand_scores[i]:.3f} src=owlv2")
        if n_dropped or len(cand_boxes) != len(yolo_lines):
            print(f"[owl] {tp.name}: {len(cand_boxes)} raw → {len(yolo_lines)} "
                  f"after NMS(iou={nms_iou})+top{top_k} (dropped {n_dropped} oob)")
        yield tp, yolo_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True,
                    help="dir of unlabeled images to pre-annotate")
    ap.add_argument("--exemplar-config", required=True,
                    help="JSON with {species, exemplars:[{image, bbox_yolo}, ...]}")
    ap.add_argument("--species", required=True, choices=list(CWD12),
                    help="canonical CWD12 species name (must match config)")
    ap.add_argument("--out-dir", required=True,
                    help="output dir for YOLO .txt label files (red proposals)")
    ap.add_argument("--conf-threshold", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="NMS IoU to collapse duplicate boxes across queries")
    ap.add_argument("--top-k", type=int, default=30,
                    help="max boxes kept per image after NMS (0 = unlimited)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap target images for testing (0 = all)")
    ap.add_argument("--model-id",
                    default="google/owlv2-large-patch14-ensemble")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't load model; just print plan")
    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    out_dir = Path(args.out_dir)
    cfg_path = Path(args.exemplar_config)

    if not target_dir.is_dir():
        print(f"FATAL: target-dir not found: {target_dir}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.is_file():
        print(f"FATAL: exemplar-config not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("species") != args.species:
        print(f"FATAL: config species {cfg.get('species')} != "
              f"--species {args.species}", file=sys.stderr)
        sys.exit(2)
    exemplars = cfg.get("exemplars") or []
    if not exemplars:
        print("FATAL: no exemplars in config", file=sys.stderr)
        sys.exit(2)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    target_imgs = sorted(p for p in target_dir.iterdir() if p.suffix in exts)
    if args.max_images:
        target_imgs = target_imgs[: args.max_images]

    print(f"=== OWL pre-annotate ===")
    print(f"  species:       {args.species}")
    print(f"  exemplars:     {len(exemplars)} from {cfg_path}")
    print(f"  target images: {len(target_imgs)} in {target_dir}")
    print(f"  out:           {out_dir}")
    print(f"  threshold:     {args.conf_threshold}")
    print(f"  model:         {args.model_id}")
    if args.dry_run:
        print("[dry-run] not loading model. exiting.")
        return

    # Load and crop exemplars
    from PIL import Image
    exemplar_crops = []
    for ex in exemplars:
        try:
            img = Image.open(ex["image"]).convert("RGB")
        except Exception as e:
            print(f"[exemplar] skip {ex.get('image')}: {e}", file=sys.stderr)
            continue
        c = _crop_yolo_box(img, ex["bbox_yolo"])
        if c is None:
            continue
        exemplar_crops.append(c)
    if not exemplar_crops:
        print("FATAL: 0 valid exemplar crops", file=sys.stderr)
        sys.exit(2)
    print(f"[owl] {len(exemplar_crops)} exemplar crops ready")

    processor, model, device = _load_owl(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time(); n_ok = 0; n_props = 0
    for img_path, yolo_lines in _process_targets(
        target_imgs, exemplar_crops, args.species,
        processor, model, device, args.conf_threshold,
        nms_iou=args.nms_iou, top_k=args.top_k,
    ):
        txt_path = out_dir / (img_path.stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
        n_ok += 1; n_props += len(yolo_lines)
        if n_ok % 25 == 0:
            print(f"  ... {n_ok}/{len(target_imgs)} props={n_props} "
                  f"({time.time()-t0:.0f}s)")
    print(f"DONE: {n_ok} images annotated, {n_props} proposed boxes "
          f"in {time.time()-t0:.0f}s")
    print(f"      out: {out_dir}/<img>.txt")
    print(f"      next: upload with roboflow_sync.py species-upload "
          f"--batch red")


if __name__ == "__main__":
    main()
