"""
v3.0.38-C — DINO label verifier (the classification head).

Per Prof. Zhang's direction (2026-05-19):
  "Input synthetic and real to dino. We can add classification head on
   top of dino to differentiate."

WHY a classifier, on top of the curators:
  - dinov2_curator (v3.0.36, whole-image) catches "this DATASET isn't
    weeds/crops".
  - dinov2_object_curator (v3.0.38-B, per-bbox similarity) catches
    "this BOX doesn't look like a real weed object".
  - Neither can catch "this box IS a weed, but labelled the WRONG
    species" — a PalmerAmaranth crop tagged Crabgrass still looks like a
    weed, so it passes every similarity gate. Catching a class swap needs
    a SUPERVISED classifier: predict the species, compare to the claimed
    label. (This is the cleanlab / Confident-Learning "swapped label"
    error type, done with an independent model.)

THE HEAD:
  A single linear layer on top of FROZEN DINOv2 features — i.e. a linear
  probe, the strong-and-standard choice for frozen self-supervised
  features. Trained on object crops whose labels are GUARANTEED correct:
    - the cut-paste SYNTHETIC object bank (synth_cutpaste.py): every crop
      is filed under its true class by construction;
    - (the bank is itself built from trusted real GT boxes, so the head
      sees real weed pixels with reliable labels.)

VERIFY:
  For every registry slug whose class names map into the canonical 12
  cottonweed species: crop each labelled bbox, DINOv2-embed, the head
  predicts a species. A box is flagged ONLY when the head disagrees with
  the claimed label at HIGH confidence (Confident-Learning: act on
  confident disagreement, not on every disagreement). Per-slug
  disagreement rate is the dataset-level signal.

  Output is a report + scores; nothing is auto-flagged (review first),
  consistent with the curators.

Commands:
  python -m weed_optimizer_framework.tools.dino_label_verifier train
  python -m weed_optimizer_framework.tools.dino_label_verifier verify
  python -m weed_optimizer_framework.tools.dino_label_verifier report
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from weed_optimizer_framework.tools.dinov2_curator import (
    REPO, _load_dinov2, _load_registry, _resolve_slug_local_path,
)
from weed_optimizer_framework.tools.dinov2_object_curator import (
    _embed_crops, _crop_boxes,
)
from weed_optimizer_framework.tools.synth_cutpaste import (
    BANK_DIR as SYNTH_BANK_DIR, CANONICAL_12, IMG_EXTS,
    _find_label, _iter_images,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("dino_label_verifier")

VERIFIER_DIR = REPO / "results" / "framework" / "dino_label_verifier"
HEAD_PATH    = VERIFIER_DIR / "head.npz"           # linear head weights + classes
VERIFY_PATH  = VERIFIER_DIR / "verify_scores.json"

# Confidence above which a head/label disagreement is treated as a real
# label error (Confident-Learning style — only confident disagreements).
CONF_THRESHOLD = 0.80
# Per-slug: embed at most this many labelled boxes when verifying.
MAX_BOXES_PER_SLUG = 600
IMAGES_PER_SLUG    = 80

VERIFIER_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Canonical class-name matching
# ----------------------------------------------------------------------------
def _canon(name: str) -> str | None:
    """Map an arbitrary class-name string to one of the canonical 12, or None."""
    if not name:
        return None
    key = "".join(c for c in str(name).lower() if c.isalnum())
    for c in CANONICAL_12:
        if "".join(ch for ch in c.lower() if ch.isalnum()) == key:
            return c
    return None


# ----------------------------------------------------------------------------
# Linear head (numpy) — trained on cached DINO embeddings
# ----------------------------------------------------------------------------
def _train_linear_head(X: np.ndarray, y: np.ndarray, n_classes: int,
                       epochs: int = 300, lr: float = 0.05, l2: float = 1e-4):
    """Multinomial logistic regression by full-batch gradient descent.

    X: [N, D] (L2-normalised embeddings), y: [N] int labels. Returns (W, b).
    Kept dependency-free (numpy only) — the head is tiny, this is exact.
    """
    N, D = X.shape
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, size=(D, n_classes)).astype(np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    Y = np.eye(n_classes)[y]
    Xd = X.astype(np.float64)
    for ep in range(epochs):
        logits = Xd @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        gW = Xd.T @ (p - Y) / N + l2 * W
        gb = (p - Y).mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    return W, b


def _head_predict(X: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Return (pred_idx, max_prob, full_prob) for embeddings X."""
    logits = X.astype(np.float64) @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1), p


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def train_head():
    """Fit the linear head on synthetic-bank object crops (reliable labels)."""
    if not SYNTH_BANK_DIR.is_dir():
        log.error(f"synthetic object bank not found at {SYNTH_BANK_DIR} — "
                  f"run `synth_cutpaste bank` first")
        sys.exit(1)

    # gather crops grouped by canonical class
    per_class: dict[str, list[Path]] = {}
    for d in sorted(SYNTH_BANK_DIR.iterdir()):
        if not d.is_dir():
            continue
        canon = _canon(d.name)
        if canon is None:
            continue  # non-canonical bank class — skip (curators handle those)
        crops = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
        if crops:
            per_class.setdefault(canon, []).extend(crops)

    classes = sorted(per_class.keys())
    if len(classes) < 2:
        log.error(f"need >=2 canonical classes in the bank, got {classes}")
        sys.exit(1)
    cls_to_id = {c: i for i, c in enumerate(classes)}
    log.info(f"training classes ({len(classes)}): "
             f"{ {c: len(per_class[c]) for c in classes} }")

    model, proc = _load_dinov2()
    embs, labels = [], []
    for c in classes:
        paths = per_class[c]
        crops = []
        for p in paths:
            try:
                crops.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        e = _embed_crops(model, proc, crops)
        embs.append(_normalize(e))
        labels.extend([cls_to_id[c]] * e.shape[0])
        log.info(f"  [{c}] embedded {e.shape[0]} crops")

    X = np.concatenate(embs, axis=0)
    y = np.array(labels, dtype=np.int64)

    # 85/15 split for an honest held-out accuracy
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cut = int(0.85 * len(y))
    tr, te = idx[:cut], idx[cut:]
    W, b = _train_linear_head(X[tr], y[tr], len(classes))

    pred_te, _, _ = _head_predict(X[te], W, b)
    acc = float((pred_te == y[te]).mean()) if len(te) else float("nan")
    # per-class held-out accuracy
    per_cls_acc = {}
    for c, i in cls_to_id.items():
        m = y[te] == i
        per_cls_acc[c] = float((pred_te[m] == i).mean()) if m.any() else None

    np.savez(HEAD_PATH, W=W, b=b, classes=np.array(classes))
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_classes": len(classes), "classes": classes,
        "heldout_accuracy": acc, "per_class_heldout_accuracy": per_cls_acc,
        "embedding_dim": int(X.shape[1]),
    }
    with open(VERIFIER_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"=== head trained: held-out accuracy = {acc:.4f} "
             f"on {len(classes)} classes → {HEAD_PATH} ===")
    log.info(f"per-class held-out acc: {per_cls_acc}")
    return W, b, classes


# ----------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------
def _slug_canon_map(info: dict) -> dict[int, str] | None:
    """Map a slug's source class-id -> canonical name, via class_names."""
    names = info.get("class_names") if isinstance(info, dict) else None
    if not names:
        return None
    m = {}
    for i, n in enumerate(names):
        c = _canon(n)
        if c is not None:
            m[i] = c
    return m or None


def verify_slugs():
    """Score every registry slug for label correctness against the head."""
    if not HEAD_PATH.exists():
        log.error("head not found — run `train` first")
        sys.exit(1)
    d = np.load(HEAD_PATH, allow_pickle=True)
    W, b, classes = d["W"], d["b"], list(d["classes"])
    cls_to_id = {c: i for i, c in enumerate(classes)}
    log.info(f"loaded head: {len(classes)} classes")

    # v3.0.38.1 bug fix: registry shape is {"datasets": {<slug>: ...}, ...};
    # the previous code iterated top-level keys (datasets, discovered, ...)
    # and verified 0 real slugs. Mirror dinov2_curator: dive into "datasets".
    reg_raw = _load_registry()
    reg = reg_raw.get("datasets", reg_raw)
    model, proc = _load_dinov2()
    scores = {}

    for slug, info in reg.items():
        cmap = _slug_canon_map(info)
        if cmap is None:
            scores[slug] = {"slug": slug, "status": "unmappable_classes"}
            continue
        slug_dir = _resolve_slug_local_path(slug, info)
        if slug_dir is None:
            scores[slug] = {"slug": slug, "status": "missing_path"}
            continue

        crops, claimed = [], []
        imgs = list(_iter_images(slug_dir, cap=IMAGES_PER_SLUG * 8))
        random.Random(0).shuffle(imgs)
        n_imgs = 0
        for img_path in imgs:
            if n_imgs >= IMAGES_PER_SLUG or len(crops) >= MAX_BOXES_PER_SLUG:
                break
            lbl = _find_label(img_path, slug_dir)
            if lbl is None:
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            n_imgs += 1
            # crop boxes AND read their claimed class id (first token of line)
            try:
                lines = lbl.read_text().splitlines()
            except Exception:
                continue
            W0, H0 = im.size
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    src = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                canon = cmap.get(src)
                if canon is None or canon not in cls_to_id:
                    continue
                x1 = int(max(0.0, cx - bw / 2) * W0)
                y1 = int(max(0.0, cy - bh / 2) * H0)
                x2 = int(min(1.0, cx + bw / 2) * W0)
                y2 = int(min(1.0, cy + bh / 2) * H0)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                crops.append(im.crop((x1, y1, x2, y2)))
                claimed.append(cls_to_id[canon])
                if len(crops) >= MAX_BOXES_PER_SLUG:
                    break

        if not crops:
            scores[slug] = {"slug": slug, "status": "no_mappable_boxes"}
            continue

        emb = _normalize(_embed_crops(model, proc, crops))
        pred, conf, _ = _head_predict(emb, W, b)
        claimed = np.array(claimed[:len(pred)])
        agree = (pred == claimed)
        # confident disagreement = a likely real label error
        confident_wrong = (~agree) & (conf > CONF_THRESHOLD)
        scores[slug] = {
            "slug": slug, "status": "ok",
            "n_boxes": int(len(pred)),
            "label_agreement": float(agree.mean()),
            "confident_error_rate": float(confident_wrong.mean()),
            "mean_confidence": float(conf.mean()),
        }
        log.info(f"  [{slug:42s}] boxes={len(pred):4d} "
                 f"agree={agree.mean():.3f} "
                 f"confident_err={confident_wrong.mean():.3f}")

    with open(VERIFY_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    log.info(f"=== verified {len(scores)} slugs → {VERIFY_PATH} ===")
    return scores


def report():
    if not VERIFY_PATH.exists():
        log.error("verify scores not found — run `verify` first")
        sys.exit(1)
    scores = json.load(open(VERIFY_PATH))
    ok = [r for r in scores.values() if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("confident_error_rate", 0), reverse=True)
    print(f"{'slug':44s} {'boxes':>7s} {'agree':>8s} {'conf_err':>9s} {'mean_conf':>10s}")
    print("-" * 84)
    for r in ok:
        print(f"{r['slug']:44s} {r['n_boxes']:>7d} "
              f"{r['label_agreement']:>8.3f} {r['confident_error_rate']:>9.3f} "
              f"{r['mean_confidence']:>10.3f}")
    skipped = [r for r in scores.values() if r.get("status") != "ok"]
    print(f"\n{len(ok)} slugs verified, {len(skipped)} skipped "
          f"(unmappable classes / no boxes / missing path).")
    if ok:
        worst = ok[0]
        print(f"\nHighest confident-error-rate: {worst['slug']} "
              f"({worst['confident_error_rate']:.1%}) — review these boxes "
              f"before the dataset enters training.")
    print("\nNo dataset is auto-flagged. Review confident_error_rate, then "
          "flag via dataset_flags.json if a slug's label noise is severe.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["train", "verify", "report"])
    args = ap.parse_args()
    if args.command == "train":
        train_head()
    elif args.command == "verify":
        verify_slugs()
    elif args.command == "report":
        report()


if __name__ == "__main__":
    main()
