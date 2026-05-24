"""
v3.0.38-B — Object-level DINOv2 curator.

The v3.0.36 whole-image curator (dinov2_curator.py) flags off-domain
DATASETS — and it worked (trusted 0.78 vs untrusted 0.26, 51 slugs
flagged). But v3.0.37 showed the residual noise still dragging cumulative
training down is bad autolabel BOXES *inside* otherwise-OK datasets. A
whole-image filter cannot see a wrong box in a good image.

Per Prof. Zhang (2026-05-19):
  "The simple thing is to iterate all objects in the real image and
   compare all of them ... synthetic and high quality ground truth
   images have to be obtained first to have our basic ground truth."

This module does exactly that:

  Stage 1  build-object-reference
           An OBJECT reference pool — DINOv2 embeddings of object crops
           from (a) trusted real-bbox slugs' GT boxes and (b) the
           cut-paste synthetic object bank (synth_cutpaste.py). The
           synthetic crops are the "basic ground truth" Prof. Zhang asks
           for; the trusted real crops anchor the pool to real imagery.

  Stage 2  score-objects
           For each candidate slug, sample images, crop EVERY labeled
           bbox, DINOv2-embed each crop, score = mean top-K cosine to the
           object pool. Per-box scores AND a per-slug aggregate.

  Stage 3  report
           Print per-slug stats + a calibration suggestion. No
           destructive auto-flag default — same discipline as
           dinov2_curator (user reviews before anything is flagged).

DINO is strong on object boundary and can lose color (Prof. Zhang), so
comparing tightly-cropped single objects is precisely DINO's strong
regime — far better aligned than whole multi-object scenes.

Usage:
  python -m weed_optimizer_framework.tools.dinov2_object_curator build-object-reference
  python -m weed_optimizer_framework.tools.dinov2_object_curator score-objects
  python -m weed_optimizer_framework.tools.dinov2_object_curator report
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
from PIL import Image

# Reuse the validated DINOv2 helpers from the whole-image curator.
# NOTE: dinov2_curator._embed_images takes file PATHS (it calls Image.open).
# This module works with in-memory PIL crops, so it defines its own
# _embed_crops below instead of reusing _embed_images.
from weed_optimizer_framework.tools.dinov2_curator import (
    REPO, REGISTRY_PATH, TRUSTED_SLUGS,
    _load_dinov2, _normalize,
    _load_registry, _resolve_slug_local_path,
)
# Reuse object/label helpers from the synthetic generator.
from weed_optimizer_framework.tools.synth_cutpaste import (
    BANK_DIR as SYNTH_BANK_DIR, _find_label, _iter_images, IMG_EXTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("dinov2_object_curator")

CURATOR_DIR    = REPO / "results" / "framework" / "dinov2_object_curator"
OBJ_REF_POOL   = CURATOR_DIR / "object_reference_pool.npy"
OBJ_REF_META   = CURATOR_DIR / "object_reference_meta.json"
OBJ_SCORES     = CURATOR_DIR / "object_scores.json"

# How many object crops to embed per slug for the reference / scoring.
OBJECTS_PER_TRUSTED_SLUG   = 800
IMAGES_PER_CANDIDATE_SLUG  = 60
MAX_OBJECTS_PER_CANDIDATE  = 600
TOP_K_NEIGHBORS            = 30
MIN_CROP_PX                = 16

CURATOR_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Embedding — delegate to dinov2_curator._embed_pils so any backbone
# (DINOv2 / OpenCLIP / HF CLIP) works transparently via DINO_BACKBONE.
# ----------------------------------------------------------------------------
from weed_optimizer_framework.tools.dinov2_curator import _embed_pils as _embed_crops


# ----------------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------------
def _crop_boxes(img: Image.Image, label_path: Path, margin: float = 0.05):
    """Yield PIL crops for every YOLO bbox in label_path."""
    W, H = img.size
    try:
        lines = label_path.read_text().splitlines()
    except Exception:
        return
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        mx, my = bw * margin, bh * margin
        x1 = int(max(0.0, cx - bw / 2 - mx) * W)
        y1 = int(max(0.0, cy - bh / 2 - my) * H)
        x2 = int(min(1.0, cx + bw / 2 + mx) * W)
        y2 = int(min(1.0, cy + bh / 2 + my) * H)
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            continue
        yield img.crop((x1, y1, x2, y2))


def _collect_slug_crops(slug_dir: Path, max_objects: int, max_images: int,
                        seed: int = 0) -> list[Image.Image]:
    """Walk a slug dir, crop labeled bboxes, return up to max_objects crops."""
    imgs = list(_iter_images(slug_dir, cap=max_images * 6))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    crops: list[Image.Image] = []
    n_imgs = 0
    for img_path in imgs:
        if n_imgs >= max_images or len(crops) >= max_objects:
            break
        lbl = _find_label(img_path, slug_dir)
        if lbl is None:
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        n_imgs += 1
        for crop in _crop_boxes(im, lbl):
            crops.append(crop)
            if len(crops) >= max_objects:
                break
    return crops


# ----------------------------------------------------------------------------
# Stage 1: object reference pool
# ----------------------------------------------------------------------------
def build_object_reference():
    """Embed trusted-slug GT object crops + synthetic object bank crops."""
    log.info("=== Building OBJECT reference pool ===")
    # v3.0.38.1: registry shape is {"datasets": {...}, ...}; dive in.
    reg_raw = _load_registry()
    reg = reg_raw.get("datasets", reg_raw)
    model, proc = _load_dinov2()
    all_emb = []
    meta = {"sources": {}, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # (a) trusted real-bbox slugs — crop their GT boxes
    for slug in TRUSTED_SLUGS:
        slug_dir = _resolve_slug_local_path(slug, reg.get(slug, {}))
        if slug_dir is None:
            log.warning(f"  [{slug}] no local path — skipped")
            meta["sources"][slug] = {"status": "missing", "n": 0}
            continue
        crops = _collect_slug_crops(slug_dir, OBJECTS_PER_TRUSTED_SLUG,
                                    max_images=OBJECTS_PER_TRUSTED_SLUG,
                                    seed=42)
        if not crops:
            log.warning(f"  [{slug}] 0 crops")
            meta["sources"][slug] = {"status": "no_crops", "n": 0}
            continue
        emb = _embed_crops(model, proc, crops)
        all_emb.append(emb)
        meta["sources"][slug] = {"status": "ok", "n": int(emb.shape[0])}
        log.info(f"  [{slug}] {emb.shape[0]} object crops embedded")

    # (b) synthetic object bank crops (the "basic ground truth")
    if SYNTH_BANK_DIR.is_dir():
        synth_crops = []
        for p in SYNTH_BANK_DIR.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                try:
                    synth_crops.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
                if len(synth_crops) >= 4000:
                    break
        if synth_crops:
            emb = _embed_crops(model, proc, synth_crops)
            all_emb.append(emb)
            meta["sources"]["__synthetic_bank__"] = {
                "status": "ok", "n": int(emb.shape[0])}
            log.info(f"  [synthetic bank] {emb.shape[0]} crops embedded")
    else:
        log.info("  [synthetic bank] not built yet — run synth_cutpaste bank")
        meta["sources"]["__synthetic_bank__"] = {"status": "absent", "n": 0}

    if not all_emb:
        log.error("no object crops collected — cannot build pool")
        sys.exit(1)
    pool = _normalize(np.concatenate(all_emb, axis=0))
    np.save(OBJ_REF_POOL, pool)
    meta["pool_shape"] = list(pool.shape)
    with open(OBJ_REF_META, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"=== Object reference pool: {pool.shape} → {OBJ_REF_POOL} ===")
    return pool


# ----------------------------------------------------------------------------
# Stage 2: score every slug at the object level
# ----------------------------------------------------------------------------
def score_one_slug(slug, info, model, proc, ref_pool, seed=0) -> dict:
    slug_dir = _resolve_slug_local_path(slug, info)
    if slug_dir is None:
        return {"slug": slug, "status": "missing_path", "score": None}
    crops = _collect_slug_crops(slug_dir, MAX_OBJECTS_PER_CANDIDATE,
                                IMAGES_PER_CANDIDATE_SLUG, seed=seed)
    if not crops:
        return {"slug": slug, "status": "no_objects", "score": None}
    emb = _normalize(_embed_crops(model, proc, crops))
    if emb.shape[0] == 0:
        return {"slug": slug, "status": "embed_failed", "score": None}
    sims = emb @ ref_pool.T
    K = min(TOP_K_NEIGHBORS, sims.shape[1])
    per_box = np.partition(sims, -K, axis=1)[:, -K:].mean(axis=1)
    return {
        "slug": slug, "status": "ok",
        "n_objects": int(emb.shape[0]),
        "score": float(per_box.mean()),
        "score_std": float(per_box.std()),
        "score_p10": float(np.percentile(per_box, 10)),
        "score_p90": float(np.percentile(per_box, 90)),
        # fraction of boxes that look clearly off-distribution
        "frac_boxes_below_0p4": float((per_box < 0.4).mean()),
    }


def score_all_objects():
    if not OBJ_REF_POOL.exists():
        log.error("object reference pool missing — run build-object-reference")
        sys.exit(1)
    ref = np.load(OBJ_REF_POOL)
    log.info(f"Loaded object reference pool: {ref.shape}")
    # v3.0.38.1 bug fix: the registry JSON has shape
    #   {"datasets": {<slug>: {...}, ...}, "discovered": [...], ...}
    # The previous code iterated top-level keys and missed every real slug,
    # producing 0 scored slugs. Mirror dinov2_curator.score_all_slugs: pull
    # the inner "datasets" dict.
    reg_raw = _load_registry()
    reg = reg_raw.get("datasets", reg_raw)
    model, proc = _load_dinov2()
    scores = {}
    slugs = list(reg.keys())
    log.info(f"Scoring {len(slugs)} slugs at object level...")
    t0 = time.time()
    for i, slug in enumerate(slugs):
        try:
            res = score_one_slug(slug, reg[slug], model, proc, ref, seed=i)
        except Exception as e:
            log.warning(f"  [{slug}] FAIL: {e}")
            res = {"slug": slug, "status": "error", "score": None,
                   "error": str(e)}
        res["is_trusted"] = slug in TRUSTED_SLUGS
        scores[slug] = res
        sc = res.get("score")
        log.info(f"  [{i+1}/{len(slugs)}] {slug:48s} "
                 f"score={f'{sc:.4f}' if sc is not None else 'n/a':>8s} "
                 f"{'★' if res['is_trusted'] else ''}")
        if (i + 1) % 10 == 0:
            with open(OBJ_SCORES, "w") as f:
                json.dump(scores, f, indent=2)
    with open(OBJ_SCORES, "w") as f:
        json.dump(scores, f, indent=2)
    log.info(f"=== Object-scored {len(scores)} slugs in "
             f"{(time.time()-t0)/60:.1f} min → {OBJ_SCORES} ===")
    return scores


# ----------------------------------------------------------------------------
# Stage 3: report
# ----------------------------------------------------------------------------
def report_object_scores():
    if not OBJ_SCORES.exists():
        log.error("object scores missing — run score-objects first")
        sys.exit(1)
    scores = json.load(open(OBJ_SCORES))
    ranked = sorted(scores.values(), key=lambda r: r.get("score") or 0,
                    reverse=True)
    print(f"{'slug':48s} {'score':>8s} {'std':>7s} {'p10':>7s} "
          f"{'%bad_box':>9s} {'n_obj':>7s} {'trusted':>8s} {'note':>16s}")
    print("-" * 116)
    for r in ranked:
        sc = r.get("score")
        sc_s = f"{sc:.4f}" if sc is not None else "  n/a"
        std = r.get("score_std")
        std_s = f"{std:.3f}" if std is not None else "  n/a"
        p10 = r.get("score_p10")
        p10_s = f"{p10:.3f}" if p10 is not None else "  n/a"
        fb = r.get("frac_boxes_below_0p4")
        fb_s = f"{fb*100:.1f}%" if fb is not None else "  n/a"
        n = r.get("n_objects", "?")
        trust = "★ YES" if r.get("is_trusted") else ""
        note = r.get("status", "") if r.get("status") != "ok" else ""
        print(f"{r['slug']:48s} {sc_s:>8s} {std_s:>7s} {p10_s:>7s} "
              f"{fb_s:>9s} {str(n):>7s} {trust:>8s} {note:>16s}")

    trusted = [r["score"] for r in scores.values()
               if r.get("is_trusted") and r.get("score") is not None]
    untrusted = [r["score"] for r in scores.values()
                 if not r.get("is_trusted") and r.get("score") is not None]
    if trusted and untrusted:
        ts, us = np.array(trusted), np.array(untrusted)
        print()
        print("=== Calibration (object-level) ===")
        print(f"Trusted   : n={len(ts)} mean={ts.mean():.4f} "
              f"min={ts.min():.4f} 25%={np.percentile(ts,25):.4f}")
        print(f"Untrusted : n={len(us)} mean={us.mean():.4f} "
              f"max={us.max():.4f} 75%={np.percentile(us,75):.4f}")
        t25 = np.percentile(ts, 25)
        u75 = np.percentile(us, 75)
        print(f"\nSuggested object-level threshold "
              f"(midpoint of {t25:.4f} / {u75:.4f}): {(t25+u75)/2:.4f}")
        print("Review before flagging — pass scores to dinov2_curator's "
              "flag mechanism or drop high-%bad_box slugs.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build-object-reference",
                                        "score-objects", "report"])
    args = ap.parse_args()
    if args.command == "build-object-reference":
        build_object_reference()
    elif args.command == "score-objects":
        score_all_objects()
    elif args.command == "report":
        report_object_scores()


if __name__ == "__main__":
    main()
