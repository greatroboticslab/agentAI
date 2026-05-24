"""
v3.0.38-C — Copy-paste synthetic weed image generator.

Per Prof. Zhang's collection-phase direction (2026-05-19):
  "We can generate synthetic dataset ... synthetic and high quality ground
   truth images have to be obtained first to have our basic ground truth."

The synthetic set is NOT for training the detector. It is the trusted
anchor for data CURATION: a DINO reference pool + the positive class for
a DINO classification head that separates good vs problematic collected
data (see dinov2_curator.py object-level mode, v3.0.38-B).

Approach: cut-paste synthesis (Dwibedi et al., ICCV 2017, "Cut, Paste and
Learn"). We crop real weed objects from our trusted real-bbox datasets
(exact GT boxes), then paste them onto field backgrounds at random
scale / position / rotation / flip. Because we place every object
ourselves, the bbox GT is pixel-exact — zero labeling noise.

Why cut-paste, not FLUX / Stable Diffusion (user decision 2026-05-19):
  - GT boxes are exact and free — we know where we pasted
  - Fast, CPU-only, no GPU / no model download / no quality lottery
  - Objects are REAL weed pixels, so DINO's boundary features stay valid.
    Prof. Zhang: "DINO may lose color, it's good at object boundary" —
    cut-paste preserves true object boundaries, which is what DINO uses.

Each pasted object is masked with an Excess-Green silhouette (ExG =
2G - R - B) so we paste the plant, not its rectangular bounding crop —
this gives DINO a clean boundary instead of a hard rectangle edge.

Pipeline (three subcommands):
  bank        — crop every GT bbox from trusted slugs   -> object_bank/<class>/
  backgrounds — collect field backgrounds from trusted images
  compose     — paste K objects onto backgrounds        -> images + YOLO + COCO

Usage:
  python -m weed_optimizer_framework.tools.synth_cutpaste bank
  python -m weed_optimizer_framework.tools.synth_cutpaste backgrounds
  python -m weed_optimizer_framework.tools.synth_cutpaste compose --n 2000
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("synth_cutpaste")

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()

REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
SYNTH_DIR     = REPO / "results" / "framework" / "synth_cutpaste"
BANK_DIR      = SYNTH_DIR / "object_bank"
BG_DIR        = SYNTH_DIR / "backgrounds"
OUT_IMG_DIR   = SYNTH_DIR / "images"
OUT_LBL_DIR   = SYNTH_DIR / "labels"

# Canonical 12 cottonweed classes — kept identical to mega_trainer.CANONICAL_12
# so synthetic labels share the detector's class space.
CANONICAL_12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
    "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
]

# Trusted real-bbox slugs whose GT boxes seed the object bank. Identical
# to dinov2_curator.TRUSTED_SLUGS — these are the verified anchors.
TRUSTED_SLUGS = [
    "cottonweeddet12", "cottonweed_sp8", "cottonweed_holdout",
    "weedsense", "crop_weed_research", "grass_weeds",
    "weed_crop_aerial", "francesco__weed_crop_aerial",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------
# Dataset resolution
# ----------------------------------------------------------------------------
def _load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        log.warning(f"registry not found at {REGISTRY_PATH}")
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f).get("datasets", {})


def _resolve_slug_dir(slug: str, info: dict) -> Path | None:
    """Return a directory we can walk for YOLO image/label pairs."""
    if isinstance(info, dict):
        lp = info.get("local_path")
        if lp and os.path.isdir(lp):
            return Path(lp)
    cand = REPO / "downloads" / slug
    return cand if cand.is_dir() else None


def _find_label(img_path: Path, root: Path) -> Path | None:
    """Locate the YOLO .txt label for an image (mirror or sibling layout)."""
    stem = img_path.stem
    # sibling: images/foo.jpg -> labels/foo.txt
    if img_path.parent.name == "images":
        cand = img_path.parent.parent / "labels" / (stem + ".txt")
        if cand.exists():
            return cand
    # same dir
    cand = img_path.with_suffix(".txt")
    if cand.exists():
        return cand
    # cwd12 flat layout: weedImages/foo.jpg -> annotation_YOLO_txt/foo.txt
    for lbldir in ("annotation_YOLO_txt", "labels"):
        cand = root / lbldir / (stem + ".txt")
        if cand.exists():
            return cand
    return None


def _iter_images(root: Path, cap: int = 100000):
    n = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p
            n += 1
            if n >= cap:
                return


# ----------------------------------------------------------------------------
# Excess-Green foreground mask
# ----------------------------------------------------------------------------
def _exg_mask(rgb: np.ndarray) -> np.ndarray:
    """Excess-Green plant silhouette. rgb uint8 [H,W,3] -> uint8 alpha [H,W].

    ExG = 2G - R - B. Pixels above an Otsu-ish midpoint are plant. We then
    feather the mask edge so the paste blends instead of showing a hard cut.
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    exg = 2.0 * g - r - b
    lo, hi = float(exg.min()), float(exg.max())
    if hi - lo < 1e-3:
        return np.full(rgb.shape[:2], 255, dtype=np.uint8)
    norm = (exg - lo) / (hi - lo)
    # threshold at the mean — robust enough for green-on-soil weed crops
    thr = float(norm.mean())
    mask = (norm > thr).astype(np.uint8) * 255
    return mask


# ----------------------------------------------------------------------------
# Canonical 12-class resolution (v3.0.39.4 bug fix)
# ----------------------------------------------------------------------------
# Bug found 2026-05-24: Goosegrass / Crabgrass bank crops were broadleaf
# plants (not grasses) — because build_bank trusted `info["class_names"][cid]`
# blindly, but some slugs' class_names list is mis-ordered vs the actual
# label-file class IDs (the same bug mega_trainer fixed in v3.0.25 via
# CANONICAL_12 name-matching). Side effect: the linear head was trained on
# mislabeled crops, so its 0.667 held-out accuracy is partly noise.
#
# Resolution: only accept a bank crop if its (slug, src_cid) maps to one of
# the CANONICAL_12 names via NAME-matching (fall back to the well-known
# original cwd12 ordering only for cottonweed_* slugs that lack class_names).
# Anything that doesn't map gets dropped — keeps the bank clean by design.

# cwd12's original published class ordering (5648-image release):
_CWD12_ORIG = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]


def _norm(s: str) -> str:
    """Normalise a class name for comparison (lower, strip non-alnum)."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


_CANON_NORM = {_norm(c): c for c in CANONICAL_12}


def _canon_name_for_label(slug: str, info: dict, src_cid: int) -> str | None:
    """Map (slug, src_class_id) -> a CANONICAL_12 name, or None to skip.

    1. If the slug has class_names registered, name-match into CANONICAL_12
       (case/punct-insensitive). Drops anything that doesn't match — better
       to lose a crop than to bank it under the wrong species.
    2. If class_names absent AND slug is cottonweed-related, fall back to
       the published cwd12 ordering (_CWD12_ORIG).
    3. Otherwise return None.
    """
    names = info.get("class_names") if isinstance(info, dict) else None
    if names and 0 <= src_cid < len(names):
        return _CANON_NORM.get(_norm(names[src_cid]))
    if "cottonweed" in slug.lower() and 0 <= src_cid < len(_CWD12_ORIG):
        return _CWD12_ORIG[src_cid]
    return None


# ----------------------------------------------------------------------------
# Stage 1: object bank
# ----------------------------------------------------------------------------
def build_bank(max_per_class: int = 400, margin: float = 0.06):
    """Crop every GT bbox from trusted slugs into object_bank/<class>/.

    v3.0.39.4: class names are resolved through CANONICAL_12 (via
    _canon_name_for_label). Crops whose label cannot be mapped to one of the
    12 canonical cwd12 species are DROPPED, not banked under a slug name.
    """
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    reg = _load_registry()
    per_class: dict[str, int] = {}
    skipped_unmapped: dict[str, int] = {}
    total = 0
    t0 = time.time()

    for slug in TRUSTED_SLUGS:
        slug_dir = _resolve_slug_dir(slug, reg.get(slug, {}))
        if slug_dir is None:
            log.warning(f"  [{slug}] no local path — skipped")
            continue
        info = reg.get(slug, {})
        n_slug = 0
        n_drop = 0
        for img_path in _iter_images(slug_dir, cap=4000):
            lbl = _find_label(img_path, slug_dir)
            if lbl is None:
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            W, H = im.size
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                cname = _canon_name_for_label(slug, info, cid)
                if cname is None:
                    n_drop += 1
                    continue
                if per_class.get(cname, 0) >= max_per_class:
                    continue
                # crop with margin
                mx, my = bw * margin, bh * margin
                x1 = int(max(0.0, (cx - bw / 2 - mx)) * W)
                y1 = int(max(0.0, (cy - bh / 2 - my)) * H)
                x2 = int(min(1.0, (cx + bw / 2 + mx)) * W)
                y2 = int(min(1.0, (cy + bh / 2 + my)) * H)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                cdir = BANK_DIR / cname
                cdir.mkdir(parents=True, exist_ok=True)
                idx = per_class.get(cname, 0)
                crop.save(cdir / f"{slug}_{img_path.stem}_{idx:04d}.png")
                per_class[cname] = idx + 1
                n_slug += 1
                total += 1
        skipped_unmapped[slug] = n_drop
        log.info(f"  [{slug}] cropped {n_slug} canonical objects "
                 f"(skipped {n_drop} unmappable)")

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_objects": total,
        "per_class": per_class,
        "skipped_unmapped_per_slug": skipped_unmapped,
        "max_per_class": max_per_class,
    }
    with open(SYNTH_DIR / "bank_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"=== object bank: {total} objects, {len(per_class)} classes, "
             f"{(time.time()-t0)/60:.1f} min ===")
    return meta


# ----------------------------------------------------------------------------
# Stage 2: backgrounds
# ----------------------------------------------------------------------------
def collect_backgrounds(n: int = 300, size: int = 640):
    """Sample field backgrounds from trusted images (downscaled full scenes)."""
    BG_DIR.mkdir(parents=True, exist_ok=True)
    reg = _load_registry()
    pool: list[Path] = []
    for slug in TRUSTED_SLUGS:
        slug_dir = _resolve_slug_dir(slug, reg.get(slug, {}))
        if slug_dir is None:
            continue
        pool.extend(list(_iter_images(slug_dir, cap=1500)))
    if not pool:
        log.error("no trusted images found for backgrounds")
        return 0
    rng = random.Random(42)
    rng.shuffle(pool)
    saved = 0
    for p in pool:
        if saved >= n:
            break
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        # center-crop to square, resize — a plausible field canvas
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.BILINEAR)
        # blur slightly so pasted objects pop and existing weeds defocus
        im = im.filter(ImageFilter.GaussianBlur(radius=1.2))
        im.save(BG_DIR / f"bg_{saved:04d}.jpg", quality=88)
        saved += 1
    log.info(f"=== backgrounds: {saved} saved to {BG_DIR} ===")
    return saved


# ----------------------------------------------------------------------------
# Stage 3: compose
# ----------------------------------------------------------------------------
def _paste_object(canvas: Image.Image, obj: Image.Image, rng: random.Random,
                  canvas_size: int):
    """Paste one ExG-masked object at random scale/pos/rot. Return YOLO bbox."""
    # random object scale: longest side 8%–34% of the canvas
    target = rng.uniform(0.08, 0.34) * canvas_size
    ow, oh = obj.size
    scale = target / max(ow, oh)
    nw, nh = max(8, int(ow * scale)), max(8, int(oh * scale))
    obj = obj.resize((nw, nh), Image.BILINEAR)

    if rng.random() < 0.5:
        obj = obj.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-25, 25)
    obj = obj.rotate(angle, expand=True, resample=Image.BILINEAR)

    arr = np.array(obj.convert("RGB"))
    alpha = _exg_mask(arr)
    # feather edges for a soft composite
    alpha_im = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1.5))

    nw, nh = obj.size
    if nw >= canvas_size or nh >= canvas_size:
        return None
    px = rng.randint(0, canvas_size - nw)
    py = rng.randint(0, canvas_size - nh)
    canvas.paste(obj.convert("RGB"), (px, py), alpha_im)

    # tight bbox from the alpha mask (fall back to full paste rect)
    ys, xs = np.where(np.array(alpha_im) > 32)
    if len(xs) < 4:
        bx1, by1, bx2, by2 = px, py, px + nw, py + nh
    else:
        bx1, by1 = px + int(xs.min()), py + int(ys.min())
        bx2, by2 = px + int(xs.max()), py + int(ys.max())
    cx = (bx1 + bx2) / 2 / canvas_size
    cy = (by1 + by2) / 2 / canvas_size
    bw = (bx2 - bx1) / canvas_size
    bh = (by2 - by1) / canvas_size
    return cx, cy, bw, bh


def compose(n_images: int = 2000, canvas_size: int = 640,
            objs_per_image=(1, 6), seed: int = 0):
    """Generate n synthetic images with exact YOLO + COCO ground truth."""
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    bg_files = sorted(BG_DIR.glob("*.jpg"))
    if not bg_files:
        log.error("no backgrounds — run `backgrounds` first")
        sys.exit(1)
    # object bank, keyed by class dir name
    class_dirs = [d for d in sorted(BANK_DIR.iterdir()) if d.is_dir()]
    if not class_dirs:
        log.error("empty object bank — run `bank` first")
        sys.exit(1)
    bank: dict[str, list[Path]] = {}
    for d in class_dirs:
        objs = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
        if objs:
            bank[d.name] = objs
    # synthetic label space = the class-dir names (sorted, stable)
    classnames = sorted(bank.keys())
    cls_to_id = {c: i for i, c in enumerate(classnames)}
    log.info(f"compose: {len(bg_files)} bgs, {len(classnames)} classes, "
             f"{sum(len(v) for v in bank.values())} objects")

    rng = random.Random(seed)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c, "supercategory": "weed"}
                           for c, i in cls_to_id.items()]}
    ann_id = 1
    t0 = time.time()

    for i in range(n_images):
        bg = Image.open(rng.choice(bg_files)).convert("RGB")
        bg = bg.resize((canvas_size, canvas_size), Image.BILINEAR)
        k = rng.randint(*objs_per_image)
        lines = []
        anns_this = []
        for _ in range(k):
            cname = rng.choice(classnames)
            obj = Image.open(rng.choice(bank[cname])).convert("RGB")
            res = _paste_object(bg, obj, rng, canvas_size)
            if res is None:
                continue
            cx, cy, bw, bh = res
            cid = cls_to_id[cname]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            anns_this.append((cid, cx, cy, bw, bh))
        if not lines:
            continue
        name = f"synth_{i:06d}"
        bg.save(OUT_IMG_DIR / f"{name}.jpg", quality=90)
        (OUT_LBL_DIR / f"{name}.txt").write_text("\n".join(lines))

        img_id = i + 1
        coco["images"].append({"id": img_id, "file_name": f"{name}.jpg",
                               "width": canvas_size, "height": canvas_size})
        for cid, cx, cy, bw, bh in anns_this:
            x = (cx - bw / 2) * canvas_size
            y = (cy - bh / 2) * canvas_size
            w = bw * canvas_size
            h = bh * canvas_size
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        if (i + 1) % 500 == 0:
            log.info(f"  composed {i+1}/{n_images}")

    with open(SYNTH_DIR / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    log.info(f"=== compose: {len(coco['images'])} images, "
             f"{len(coco['annotations'])} objects, "
             f"{(time.time()-t0)/60:.1f} min → {OUT_IMG_DIR} ===")
    return len(coco["images"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["bank", "backgrounds", "compose"])
    ap.add_argument("--n", type=int, default=2000,
                    help="compose: number of synthetic images; "
                         "backgrounds: number to collect")
    ap.add_argument("--max-per-class", type=int, default=400,
                    help="bank: cap objects cropped per class")
    ap.add_argument("--canvas", type=int, default=640, help="compose canvas px")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "bank":
        build_bank(max_per_class=args.max_per_class)
    elif args.command == "backgrounds":
        collect_backgrounds(n=args.n)
    elif args.command == "compose":
        compose(n_images=args.n, canvas_size=args.canvas, seed=args.seed)


if __name__ == "__main__":
    main()
