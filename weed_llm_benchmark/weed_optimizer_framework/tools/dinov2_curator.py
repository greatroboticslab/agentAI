"""
v3.0.36 — DINOv2 feature-similarity curator for Brain-collected datasets.

Per Hongbo's collection-phase comparison direction (2026-05-16):
  "Have model compare what existing data look like vs what it collected.
   Discard if similarity too far apart."

Implementation:
  Stage 1: Build a fixed REFERENCE POOL of DINOv2 embeddings from our
           trusted real-bbox slugs (cwd12, weedsense, crop_weed_research,
           grass_weeds, weed_crop_aerial, francesco). Multi-category by
           design — pool naturally spans weed/cotton/grass/aerial.
  Stage 2: For each slug in the registry, sample N images, compute DINOv2
           embeddings, score = mean of top-K cosine similarity to pool.
  Stage 3: Print sorted scores; the user calibrates threshold before
           auto-flagging anything (no destructive default action).

Why DINOv2 (not LLM):
  - Same backbone as RFDETR -> feature space aligned with detector
  - ~50ms/image inference (vs Gemma 4 1-5s)
  - Unsupervised cosine geometry (no "rubber stamp yes" bias seen in T4)
  - Empirically validated for self-supervised image retrieval (DINOv2 paper)

Usage:
  # One-time build reference pool (~30 min GPU for ~5000 imgs)
  python -m weed_optimizer_framework.tools.dinov2_curator build-reference

  # Score every slug in registry vs pool (~1-2h for 80 slugs)
  python -m weed_optimizer_framework.tools.dinov2_curator score-all

  # Print scores sorted (cheap, after score-all has cached scores)
  python -m weed_optimizer_framework.tools.dinov2_curator report

  # Auto-flag below threshold (manual review first)
  python -m weed_optimizer_framework.tools.dinov2_curator flag --threshold 0.45
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

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("dinov2_curator")

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
)).resolve()

REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
FLAGS_PATH    = REPO / "results" / "framework" / "dataset_flags.json"
CURATOR_DIR   = REPO / "results" / "framework" / "dinov2_curator"
REF_POOL      = CURATOR_DIR / "reference_pool.npy"
REF_META      = CURATOR_DIR / "reference_meta.json"
SCORES_PATH   = CURATOR_DIR / "slug_scores.json"

# Trusted slugs that constitute the "good data" reference. Multi-category
# by design (weed + crop + aerial). These are slugs Brain has discovered
# autonomously AND we've verified via other signals (NEVER_TRAIN guardrails,
# user dashboard review, hand-labeled GT).
#
# We deliberately AVOID hand-curating new slugs here — these are all
# already-trusted in our registry. Adding more to the pool requires
# explicit human verification (else the pool itself gets contaminated).
TRUSTED_SLUGS = [
    "cottonweeddet12",        # 12 weed species top-down (5648 imgs)
    "cottonweed_sp8",         # cwd12 8-species train split (3442)
    "cottonweed_holdout",     # cwd12 4-species holdout (2206)
    "weedsense",              # large mixed weed corpus (120K)
    "crop_weed_research",     # cotton + weed bbox (4307)
    "grass_weeds",            # grass vs weeds (2490)
    "weed_crop_aerial",       # aerial weed-crop (1176)
    "francesco__weed_crop_aerial",  # aerial holdout
]

# How many images per trusted slug to put in the reference pool
SAMPLES_PER_TRUSTED_SLUG = 500

# How many images per candidate slug to embed for similarity scoring
SAMPLES_PER_CANDIDATE_SLUG = 50

# Top-K neighbors in reference pool for averaging
TOP_K_NEIGHBORS = 50

CURATOR_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _resolve_slug_local_path(slug, info):
    """Return a directory we can walk for image files, or None."""
    if not isinstance(info, dict):
        return None
    lp = info.get("local_path")
    if lp and os.path.isdir(lp):
        return Path(lp)
    # Fallback: try downloads/<slug>
    cand = REPO / "downloads" / slug
    if cand.is_dir():
        return cand
    return None


def _sample_images(slug_dir: Path, n: int, seed: int = 0) -> list[Path]:
    """Randomly sample N image files from a slug directory."""
    rng = random.Random(seed)
    all_imgs = []
    for p in slug_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            all_imgs.append(p)
            if len(all_imgs) >= n * 20:  # cap initial collect for speed
                break
    if len(all_imgs) <= n:
        return all_imgs
    return rng.sample(all_imgs, n)


def _load_dinov2():
    """Load the DINO backbone on GPU. Returns (model, processor).

    Backbone is configurable via the DINO_BACKBONE env var, so the
    fine-grained-classification roles (dino_label_verifier) can use a
    plant-specialised checkpoint while the coarse domain-FILTER role keeps
    the generic model. Any transformers-loadable DINOv2 repo works as-is;
    default is the generic facebook/dinov2-base (what v3.0.36 validated).

    Recommended plant-specialised options (drop-in once mirrored to a
    transformers-compatible repo): the PlantCLEF-2024 fine-tuned DINOv2
    ViT (1.4M plant images, 800+ species) — same architecture, plant
    features. See CHANGELOG v3.0.38 notes.
    """
    import os
    import torch
    from transformers import AutoImageProcessor, AutoModel
    backbone = os.environ.get("DINO_BACKBONE", "facebook/dinov2-base")
    log.info(f"Loading DINO backbone: {backbone}")
    proc = AutoImageProcessor.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).cuda().eval()
    log.info(f"  loaded. params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, proc


def _embed_images(model, proc, img_paths: list[Path], batch_size: int = 16) -> np.ndarray:
    """DINOv2 embed a list of images. Returns [N, 768] np array (CLS token)."""
    import torch
    embeddings = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        pils = []
        for p in batch:
            try:
                im = Image.open(p).convert("RGB")
                pils.append(im)
            except Exception as e:
                log.warning(f"  skip {p.name}: {e}")
        if not pils:
            continue
        inputs = proc(images=pils, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        # CLS token = pooled embedding [N, 768]
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)
    if not embeddings:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-9)


def build_reference_pool():
    """Stage 1: build the reference pool of trusted-slug embeddings."""
    log.info(f"=== Building reference pool from {len(TRUSTED_SLUGS)} trusted slugs ===")
    reg = _load_registry()
    ds = reg.get("datasets", {})

    model, proc = _load_dinov2()
    all_emb = []
    meta = {"slugs": {}, "samples_per_slug": SAMPLES_PER_TRUSTED_SLUG}

    for slug in TRUSTED_SLUGS:
        info = ds.get(slug, {})
        slug_dir = _resolve_slug_local_path(slug, info)
        if slug_dir is None:
            log.warning(f"  [{slug}] no local_path — skipping")
            meta["slugs"][slug] = {"status": "missing", "n_embedded": 0}
            continue
        imgs = _sample_images(slug_dir, SAMPLES_PER_TRUSTED_SLUG, seed=42)
        log.info(f"  [{slug}] sampling {len(imgs)} imgs from {slug_dir}")
        t0 = time.time()
        emb = _embed_images(model, proc, imgs)
        dt = time.time() - t0
        log.info(f"  [{slug}] embedded {emb.shape} ({dt:.0f}s, "
                 f"{emb.shape[0]/max(dt,0.01):.1f} img/s)")
        all_emb.append(emb)
        meta["slugs"][slug] = {"status": "ok", "n_embedded": emb.shape[0]}

    pool = np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 768), dtype=np.float32)
    pool = _normalize(pool)
    np.save(REF_POOL, pool)
    meta["pool_shape"] = list(pool.shape)
    meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(REF_META, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"=== Reference pool saved: {pool.shape} → {REF_POOL} ===")
    return pool


def score_one_slug(slug, info, model, proc, ref_pool: np.ndarray,
                    seed: int = 0) -> dict:
    """Embed N samples from slug, return mean top-K cosine to reference."""
    slug_dir = _resolve_slug_local_path(slug, info)
    if slug_dir is None:
        return {"slug": slug, "status": "missing_path", "score": None}
    imgs = _sample_images(slug_dir, SAMPLES_PER_CANDIDATE_SLUG, seed=seed)
    if len(imgs) == 0:
        return {"slug": slug, "status": "no_images", "score": None}
    emb = _embed_images(model, proc, imgs)
    if emb.shape[0] == 0:
        return {"slug": slug, "status": "embed_failed", "score": None}
    emb = _normalize(emb)
    # cosine = dot product after normalization
    # sims [n_slug_imgs, n_pool]
    sims = emb @ ref_pool.T
    # Per-image top-K mean
    K = min(TOP_K_NEIGHBORS, sims.shape[1])
    top_k_per_img = np.partition(sims, -K, axis=1)[:, -K:]
    per_img_score = top_k_per_img.mean(axis=1)
    score = float(per_img_score.mean())
    score_std = float(per_img_score.std())
    score_min = float(per_img_score.min())
    score_max = float(per_img_score.max())
    return {
        "slug": slug, "status": "ok",
        "n_sampled": emb.shape[0],
        "score": score, "score_std": score_std,
        "score_min": score_min, "score_max": score_max,
    }


def score_all_slugs():
    """Stage 2: score every slug in the registry vs the reference pool."""
    if not REF_POOL.exists():
        log.error(f"reference pool missing — run `build-reference` first")
        sys.exit(1)
    ref = np.load(REF_POOL)
    log.info(f"Loaded reference pool: {ref.shape}")
    reg = _load_registry()
    ds = reg.get("datasets", {})

    model, proc = _load_dinov2()
    scores = {}
    slugs = list(ds.keys())
    log.info(f"Scoring {len(slugs)} slugs (N_sample={SAMPLES_PER_CANDIDATE_SLUG} per slug)...")
    t_total = time.time()
    for i, slug in enumerate(slugs):
        info = ds[slug]
        t0 = time.time()
        try:
            res = score_one_slug(slug, info, model, proc, ref, seed=i)
        except Exception as e:
            log.warning(f"  [{slug}] FAIL: {e}")
            res = {"slug": slug, "status": "error", "score": None, "error": str(e)}
        dt = time.time() - t0
        is_trusted = slug in TRUSTED_SLUGS
        res["is_trusted"] = is_trusted
        scores[slug] = res
        sc = res.get("score")
        sc_str = f"{sc:.4f}" if sc is not None else "n/a"
        log.info(f"  [{i+1}/{len(slugs)}] {slug:50s} score={sc_str} "
                 f"({dt:.1f}s) {'★' if is_trusted else ''}")
        # Periodic save in case of failure
        if (i + 1) % 10 == 0:
            with open(SCORES_PATH, "w") as f:
                json.dump(scores, f, indent=2)

    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    log.info(f"=== Scored {len(scores)} slugs in {(time.time()-t_total)/60:.1f} min ===")
    return scores


def report_scores():
    """Stage 3: print scores sorted by similarity (high→low)."""
    if not SCORES_PATH.exists():
        log.error(f"scores missing — run `score-all` first")
        sys.exit(1)
    scores = json.load(open(SCORES_PATH))
    ranked = sorted(scores.values(),
                    key=lambda r: r.get("score") or 0,
                    reverse=True)

    print(f"{'slug':50s} {'score':>8s} {'std':>7s} {'min':>7s} {'max':>7s} {'n':>4s} {'trusted':>8s} {'note':>20s}")
    print("-" * 120)
    for r in ranked:
        sc = r.get("score")
        sc_str = f"{sc:.4f}" if sc is not None else "  n/a"
        std = r.get("score_std")
        std_str = f"{std:.3f}" if std is not None else "  n/a"
        mn = r.get("score_min")
        mn_str = f"{mn:.3f}" if mn is not None else "  n/a"
        mx = r.get("score_max")
        mx_str = f"{mx:.3f}" if mx is not None else "  n/a"
        n = r.get("n_sampled", "?")
        trusted = "★ YES" if r.get("is_trusted") else ""
        note = r.get("status", "") if r.get("status") != "ok" else ""
        print(f"{r['slug']:50s} {sc_str:>8s} {std_str:>7s} {mn_str:>7s} "
              f"{mx_str:>7s} {str(n):>4s} {trusted:>8s} {note:>20s}")

    # Calibration suggestion
    trusted_scores = [r["score"] for r in scores.values()
                      if r.get("is_trusted") and r.get("score") is not None]
    untrusted_scores = [r["score"] for r in scores.values()
                        if not r.get("is_trusted") and r.get("score") is not None]
    if trusted_scores:
        ts = np.array(trusted_scores)
        print()
        print(f"=== Calibration ===")
        print(f"Trusted slug scores: n={len(ts)} mean={ts.mean():.4f} "
              f"min={ts.min():.4f} 25%={np.percentile(ts, 25):.4f}")
        if untrusted_scores:
            us = np.array(untrusted_scores)
            print(f"Untrusted slug scores: n={len(us)} mean={us.mean():.4f} "
                  f"min={us.min():.4f} 75%={np.percentile(us, 75):.4f}")
            # Recommend threshold: midpoint of trusted-25%ile and untrusted-75%ile
            t25 = np.percentile(ts, 25)
            u75 = np.percentile(us, 75)
            print()
            print(f"Suggested threshold (midpoint of trusted-25%ile {t25:.4f} "
                  f"and untrusted-75%ile {u75:.4f}): {(t25 + u75) / 2:.4f}")


def auto_flag_low_score(threshold: float, dry_run: bool = True):
    """Stage 4: write garbage flag for slugs with score < threshold.

    NEVER touches trusted slugs (those are in the reference pool by definition).
    Logs every action; pass --no-dry-run to actually write flags.
    """
    if not SCORES_PATH.exists():
        log.error(f"scores missing — run `score-all` first")
        sys.exit(1)
    scores = json.load(open(SCORES_PATH))

    # Load existing flags (preserve manual ones)
    if FLAGS_PATH.exists():
        flags = json.load(open(FLAGS_PATH))
    else:
        flags = {}

    to_flag = []
    for slug, r in scores.items():
        if r.get("is_trusted"):
            continue  # never auto-flag trusted reference set
        sc = r.get("score")
        if sc is None:
            continue
        if sc < threshold:
            if slug in flags and flags[slug].get("flag") == "garbage":
                continue  # already flagged
            to_flag.append((slug, sc))

    print(f"Threshold: {threshold}")
    print(f"Slugs to flag as garbage ({len(to_flag)}):")
    for slug, sc in sorted(to_flag, key=lambda x: x[1]):
        print(f"  {slug:50s} score={sc:.4f}")

    if dry_run:
        print()
        print("DRY RUN — no flags written. Re-run with --no-dry-run to apply.")
        return

    for slug, sc in to_flag:
        flags[slug] = {
            "flag": "garbage",
            "reason": f"DINOv2 similarity to trusted pool ({sc:.4f}) < threshold ({threshold})",
            "ts": time.time(),
            "ts_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "auto_flagged_by": "dinov2_curator",
        }
    # Atomic write
    tmp = str(FLAGS_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(flags, f, indent=2)
    os.replace(tmp, FLAGS_PATH)
    print(f"\n✓ Wrote {len(to_flag)} new flags to {FLAGS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build-reference", "score-all",
                                          "report", "flag"])
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="for `flag` command")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="for `flag` — actually write flags")
    args = ap.parse_args()

    if args.command == "build-reference":
        build_reference_pool()
    elif args.command == "score-all":
        score_all_slugs()
    elif args.command == "report":
        report_scores()
    elif args.command == "flag":
        auto_flag_low_score(args.threshold, dry_run=not args.no_dry_run)


if __name__ == "__main__":
    main()
