"""
v3.0.39 — FLUX layout-conditioned synthetic weed generator.

Per Prof. Zhang (confirmed by the user 2026-05-21): synthetic data must
also be USED FOR TRAINING the detector — emphasised repeatedly.

A naive standalone cut-paste corpus trains a detector to only ~45% mAP
(domain gap; published agriculture result). So this module uses the
strong, SOTA form instead — FLUX.1-Fill inpainting conditioned on a bbox
layout:

  1. take a REAL field background (from trusted slugs);
  2. pick K boxes + a target weed species per box;
  3. for each box, mask it and let FLUX.1-Fill inpaint a photoreal weed
     of that species into the masked region — the diffusion model closes
     the sim-to-real gap that hurts naive cut-paste;
  4. the bbox GT stays pixel-exact (we chose the boxes; the GT box is
     re-tightened to the generated plant via an Excess-Green mask);
  5. CLOSED LOOP (S3OD-style, ICLR'26): every generated image can be run
     through dinov2_object_curator / dino_label_verifier before use —
     generate -> DINO-verify -> keep only what passes.

This is the "L2 + L3" plan: layout-conditioned diffusion + DINO
verification. Output is meant as TRAINING AUGMENTATION mixed with real
data (not a standalone corpus), and is biased toward weak / under-
represented cwd12 classes — that is where synthetic data demonstrably
helps, per the few-shot / rare-class literature (DODA, Gen2Det).

Why FLUX.1-Fill (not raw text-to-image): raw text-to-image gives no
bounding boxes. Fill/inpainting respects a layout we control, so we get
realism AND exact GT in one step.

Usage:
  # generate N synthetic images (GPU; FLUX.1-Fill-dev must be available)
  python -m weed_optimizer_framework.tools.synth_diffusion generate --n 600

  # build a visual contact sheet of samples for review
  python -m weed_optimizer_framework.tools.synth_diffusion montage
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from weed_optimizer_framework.tools.synth_cutpaste import (
    REPO, SYNTH_DIR, BG_DIR, CANONICAL_12, _exg_mask, collect_backgrounds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("synth_diffusion")

DIFF_DIR    = REPO / "results" / "framework" / "synth_diffusion"
OUT_IMG_DIR = DIFF_DIR / "images"
OUT_LBL_DIR = DIFF_DIR / "labels"
MONTAGE     = DIFF_DIR / "sample_montage.jpg"

# FLUX.1-Fill-dev — the official FLUX inpainting / fill checkpoint.
# Gated on HuggingFace: the cluster needs an accepted license + HF token
# (huggingface-cli login) for the first download.
FLUX_FILL_MODEL = os.environ.get("FLUX_FILL_MODEL",
                                 "black-forest-labs/FLUX.1-Fill-dev")

# Per-species prompt fragments. cwd12's 12 cottonweed species; the prompt
# describes a single plant in a top-down agricultural photo so FLUX fills
# the masked box with an in-domain object.
SPECIES_PROMPT = {
    "Carpetweeds":    "a carpetweed plant, small green sprawling weed",
    "Crabgrass":      "a crabgrass plant, spreading grassy weed",
    "Eclipta":        "an eclipta weed plant, green leaves",
    "Goosegrass":     "a goosegrass plant, flat rosette grassy weed",
    "Morningglory":   "a morningglory weed, heart-shaped leaves vine",
    "Nutsedge":       "a nutsedge plant, upright grass-like weed",
    "PalmerAmaranth": "a palmer amaranth pigweed plant, broadleaf weed",
    "PricklySida":    "a prickly sida weed plant, broadleaf",
    "Purslane":       "a purslane plant, fleshy red-stemmed weed",
    "Ragweed":        "a ragweed plant, lobed green leaves",
    "Sicklepod":      "a sicklepod weed plant, paired oval leaflets",
    "SpottedSpurge":  "a spotted spurge plant, low mat-forming weed",
}
PROMPT_SUFFIX = ", top-down view, cotton field soil background, daylight, " \
                "photorealistic, sharp focus"

CANVAS = 768          # FLUX-Fill works well at 768/1024; 768 for throughput
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------
def _load_flux():
    """Load the FLUX.1-Fill pipeline on GPU. Returns the pipeline or exits."""
    try:
        import torch
        from diffusers import FluxFillPipeline
    except Exception as e:
        log.error(f"diffusers / torch import failed: {e}")
        log.error("install: pip install -U diffusers transformers accelerate")
        sys.exit(2)
    log.info(f"loading {FLUX_FILL_MODEL} (this downloads ~24GB on first run)...")
    try:
        pipe = FluxFillPipeline.from_pretrained(
            FLUX_FILL_MODEL, torch_dtype=torch.bfloat16)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            log.error(f"FLUX repo gated/unauthorized: {e}")
            log.error("Accept the FLUX.1-Fill-dev license on HuggingFace and "
                      "run `huggingface-cli login` on the cluster.")
        else:
            log.error(f"failed to load {FLUX_FILL_MODEL}: {e}")
        sys.exit(2)
    # Three placement modes, picked by env + visible-GPU count:
    #   FORCE_FLUX_NO_OFFLOAD=1  → place transformer/T5/VAE on cuda:0 / cuda:1
    #                              if 2+ GPUs visible (v3.0.39.3 path, full
    #                              speed, ~10s/img). If only 1 GPU visible
    #                              it falls back to pipe.to('cuda') and may OOM.
    #   default                  → enable_model_cpu_offload() — 1 GPU, idle
    #                              components on CPU, ~14GB GPU peak but
    #                              ~5min/img (v3.0.39.1 path, infeasible at scale).
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    force_no_offload = os.environ.get("FORCE_FLUX_NO_OFFLOAD") == "1"
    if force_no_offload and n_gpu >= 2:
        # Manually split heavy components across the two cards
        try:
            pipe.transformer.to("cuda:0")
            pipe.text_encoder_2.to("cuda:1")   # T5 is the other big chunk
            for name in ("text_encoder", "vae"):
                comp = getattr(pipe, name, None)
                if comp is not None:
                    comp.to("cuda:0")
            log.info(f"FLUX placed across {n_gpu} GPUs "
                     f"(transformer→cuda:0, T5→cuda:1, vae/text_encoder→cuda:0)")
        except Exception as e:
            log.warning(f"manual multi-GPU placement failed ({e}); "
                        f"falling back to pipe.to('cuda:0')")
            pipe = pipe.to("cuda:0")
    elif force_no_offload:
        log.warning(f"FORCE_FLUX_NO_OFFLOAD=1 but only {n_gpu} GPU visible; "
                    f"trying full pipe.to('cuda') — may OOM on 32GB cards.")
        try:
            pipe = pipe.to("cuda")
        except torch.cuda.OutOfMemoryError as oom:
            log.error(f"CUDA OOM: {oom}")
            log.error("Drop FORCE_FLUX_NO_OFFLOAD or request 2x GPUs.")
            sys.exit(2)
    else:
        try:
            pipe.enable_model_cpu_offload()
            log.info("FLUX loaded with model CPU offload (~14GB peak, slow).")
        except Exception as e:
            log.warning(f"enable_model_cpu_offload failed ({e}); fallback")
            pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    log.info("FLUX.1-Fill pipeline ready.")
    return pipe


def _weak_class_weights():
    """Bias species sampling toward weak cwd12 classes if a count file exists.

    results/framework/cwd12_class_counts.json (optional) maps class->instances.
    Rarer classes get proportionally higher sampling weight so synthetic
    augmentation concentrates where it actually helps.
    """
    cf = REPO / "results" / "framework" / "cwd12_class_counts.json"
    if cf.is_file():
        try:
            counts = json.load(open(cf))
            w = {c: 1.0 / max(1.0, counts.get(c, 1)) ** 0.5
                 for c in CANONICAL_12}
            s = sum(w.values())
            return {c: w[c] / s for c in CANONICAL_12}
        except Exception:
            pass
    return {c: 1.0 / len(CANONICAL_12) for c in CANONICAL_12}  # uniform


def _rect_mask(size: int, box) -> Image.Image:
    """White rectangle mask (FLUX fills white regions)."""
    m = np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = 255
    return Image.fromarray(m)


def _tighten_box(canvas: Image.Image, box, size: int):
    """Re-tighten a GT box to the generated plant via Excess-Green mask."""
    x1, y1, x2, y2 = box
    crop = np.array(canvas.crop((x1, y1, x2, y2)).convert("RGB"))
    if crop.size == 0:
        return None
    alpha = _exg_mask(crop)
    ys, xs = np.where(alpha > 96)
    if len(xs) < 8:
        bx1, by1, bx2, by2 = x1, y1, x2, y2          # fallback: mask rect
    else:
        bx1, by1 = x1 + int(xs.min()), y1 + int(ys.min())
        bx2, by2 = x1 + int(xs.max()), y1 + int(ys.max())
    cx = (bx1 + bx2) / 2 / size
    cy = (by1 + by2) / 2 / size
    bw = (bx2 - bx1) / size
    bh = (by2 - by1) / size
    return cx, cy, bw, bh


def _sample_boxes(size: int, k: int, rng: random.Random):
    """Non-trivially-overlapping random boxes for the layout."""
    boxes = []
    for _ in range(k * 4):
        if len(boxes) >= k:
            break
        s = int(rng.uniform(0.16, 0.42) * size)
        x1 = rng.randint(0, size - s)
        y1 = rng.randint(0, size - s)
        cand = (x1, y1, x1 + s, y1 + s)
        ok = True
        for b in boxes:
            ix = max(0, min(cand[2], b[2]) - max(cand[0], b[0]))
            iy = max(0, min(cand[3], b[3]) - max(cand[1], b[1]))
            if ix * iy > 0.35 * s * s:
                ok = False
                break
        if ok:
            boxes.append(cand)
    return boxes


# ----------------------------------------------------------------------------
def generate(n_images: int = 600, steps: int = 28, guidance: float = 30.0,
             objs_per_image=(1, 3), seed: int = 0,
             force_class: str | None = None):
    """Generate n FLUX-inpainted synthetic images with exact YOLO + COCO GT.

    If force_class is given (must be a CANONICAL_12 species name), EVERY
    pasted bbox is generated for that one class — used to produce a
    balanced 12-class /audit comparison set (one SLURM array task per class).
    """
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    bgs = sorted(BG_DIR.glob("*.jpg"))
    if not bgs:
        log.info("no backgrounds — collecting them now...")
        collect_backgrounds(n=300)
        bgs = sorted(BG_DIR.glob("*.jpg"))
    if not bgs:
        log.error("could not collect backgrounds (no trusted images?)")
        sys.exit(1)

    pipe = _load_flux()
    if force_class is not None:
        if force_class not in CANONICAL_12:
            log.error(f"force_class {force_class!r} not in CANONICAL_12")
            sys.exit(1)
        log.info(f"FORCE_CLASS = {force_class} — all generated bboxes will be "
                 f"this species (12-class balanced /audit set)")
        species = [force_class]
        weights = [1.0]
    else:
        species_w = _weak_class_weights()
        species = list(species_w.keys())
        weights = [species_w[c] for c in species]
    cls_id = {c: i for i, c in enumerate(CANONICAL_12)}

    rng = random.Random(seed)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c, "supercategory": "weed"}
                           for i, c in enumerate(CANONICAL_12)]}
    ann_id = 1
    t0 = time.time()
    made = 0

    # Disambiguate filenames per force_class so array tasks don't overwrite
    # each other (each writes fluxsynth_<class>_NNNNNN.jpg).
    name_prefix = f"fluxsynth_{force_class}_" if force_class else "fluxsynth_"

    # Find a starting index that doesn't collide with anything already there
    existing = sorted(OUT_IMG_DIR.glob(f"{name_prefix}*.jpg"))
    start_idx = 0
    if existing:
        try:
            last = existing[-1].stem.split("_")[-1]
            start_idx = int(last) + 1
        except Exception:
            start_idx = len(existing)

    for j in range(n_images):
        i = start_idx + j
        bg = Image.open(rng.choice(bgs)).convert("RGB").resize(
            (CANVAS, CANVAS), Image.BILINEAR)
        k = rng.randint(*objs_per_image)
        boxes = _sample_boxes(CANVAS, k, rng)
        lines, anns = [], []
        canvas = bg
        for box in boxes:
            sp = rng.choices(species, weights=weights, k=1)[0]
            prompt = SPECIES_PROMPT[sp] + PROMPT_SUFFIX
            mask = _rect_mask(CANVAS, box)
            try:
                canvas = pipe(
                    prompt=prompt,
                    image=canvas,
                    mask_image=mask,
                    height=CANVAS, width=CANVAS,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                ).images[0]
            except Exception as e:
                log.warning(f"  img {i} box {box}: FLUX fill failed: {e}")
                continue
            gt = _tighten_box(canvas, box, CANVAS)
            if gt is None:
                continue
            cx, cy, bw, bh = gt
            lines.append(f"{cls_id[sp]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            anns.append((cls_id[sp], cx, cy, bw, bh))
        if not lines:
            continue
        name = f"{name_prefix}{i:06d}"
        canvas.save(OUT_IMG_DIR / f"{name}.jpg", quality=92)
        (OUT_LBL_DIR / f"{name}.txt").write_text("\n".join(lines))
        img_id = i + 1
        coco["images"].append({"id": img_id, "file_name": f"{name}.jpg",
                               "width": CANVAS, "height": CANVAS})
        for cid, cx, cy, bw, bh in anns:
            x = (cx - bw / 2) * CANVAS
            y = (cy - bh / 2) * CANVAS
            w = bw * CANVAS
            h = bh * CANVAS
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        made += 1
        if made % 25 == 0:
            el = time.time() - t0
            log.info(f"  {made} images ({el/made:.1f}s/img, "
                     f"~{el/made*(n_images-made)/60:.0f} min left)")

    with open(DIFF_DIR / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    log.info(f"=== generate: {made} images, {ann_id-1} objects, "
             f"{(time.time()-t0)/60:.1f} min → {OUT_IMG_DIR} ===")
    _build_montage()
    return made


def _build_montage(grid: int = 6):
    """Save a grid of sample synthetic images for quick visual review."""
    imgs = sorted(OUT_IMG_DIR.glob("*.jpg"))
    if not imgs:
        log.warning("no synthetic images to montage")
        return
    pick = imgs[:: max(1, len(imgs) // (grid * grid))][:grid * grid]
    cell = 256
    sheet = Image.new("RGB", (grid * cell, grid * cell), (20, 20, 20))
    for idx, p in enumerate(pick):
        try:
            im = Image.open(p).convert("RGB").resize((cell, cell))
        except Exception:
            continue
        sheet.paste(im, ((idx % grid) * cell, (idx // grid) * cell))
    sheet.save(MONTAGE, quality=90)
    log.info(f"montage → {MONTAGE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["generate", "montage"])
    ap.add_argument("--n", type=int, default=600,
                    help="number of synthetic images to generate")
    ap.add_argument("--steps", type=int, default=28,
                    help="FLUX inference steps")
    ap.add_argument("--guidance", type=float, default=30.0,
                    help="FLUX.1-Fill guidance scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-name", default=None,
                    help="force every generated bbox to this species "
                         "(one of CANONICAL_12) — used for balanced 12-class "
                         "/audit comparison set")
    args = ap.parse_args()
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        generate(n_images=args.n, steps=args.steps,
                 guidance=args.guidance, seed=args.seed,
                 force_class=args.class_name)
    elif args.command == "montage":
        _build_montage()


if __name__ == "__main__":
    main()
