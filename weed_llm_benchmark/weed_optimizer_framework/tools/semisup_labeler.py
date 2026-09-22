"""Semi-supervised labeler -- Phase A: does the method work on data whose answers we hold?

The harvested pool carries boxes but no species: mega_trainer gives every box from
a source one md5-hashed auxiliary class, so the trainer learns "which website"
rather than "which weed". The fix under test is the semi-supervised pipeline:
crop each plant, embed it with a frozen self-supervised backbone, find structure
by clustering, let a person name only a small budget of exemplars, and propagate
those names to the rest.

Before that pipeline is allowed to relabel anything, it has to be shown to work
where the right answer is known. Phase A runs it on the cottonweeddet12 TRAIN
split -- 12 species, every box labelled by hand -- as if the labels were absent,
then scores what it recovers against the hidden truth:

  crops      one square crop per labelled box, holdout-guarded
  embed      frozen DINOv2-B/14, DINOv2-L/14 (CLS and patch-mean) and BioCLIP-2
  evaluate   (1) supervised ceiling: k-NN and a linear probe, grouped 5-fold CV
             (2) discovery: PCA -> HDBSCAN and k-means, purity / NMI / ARI
             (3) label budget: name 1-10 % of crops, propagate, score the rest;
                 random vs centroid exemplars vs exemplars + uncertainty sampling

The sealed cwd12 test+valid images (1,977) are the evaluation set of every
detector number this project reports. None may enter this module: the train split
is read directly, every source image is dHashed against the holdout and dropped
within near_dup.HOLDOUT_NEAR_DUP_BITS (6) bits of one, and the run fails if a holdout filename stem appears in the manifest.

Run (cluster, via sbatch):  python -m weed_optimizer_framework.tools.semisup_labeler all
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("semisup")

REPO = Path(os.environ.get("REPO", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
OUT = REPO / "results" / "framework" / "semisup" / "phaseA"
CWD12_TRAIN = REPO / "downloads" / "cottonweeddet12" / "train"

CROP_PX = 224
MARGIN = 0.10          # context around the box, as a fraction of its long side
MIN_BOX_PX = 16        # boxes smaller than this in either side are too small to name

# The species a cwd12 YOLO id stands for -- NOT the names in the dataset's
# data.yaml (cwd12_species.py has the evidence). _verify_names() re-derives it
# from the VGG annotations on every run.
from .cwd12_species import CWD12_SPECIES as CWD12_TRUE  # noqa: E402
VGG_DIR = REPO / "downloads" / "cottonweeddet12" / "CottonWeedDet12" / "annotation_VGG_json"


def _verify_names(split_dir, limit=400):
    """Re-derive id -> species from the VGG annotations and fail if it disagrees
    with CWD12_TRUE. Returns the number of boxes checked (0 if no VGG files)."""
    if not VGG_DIR.is_dir():
        log.warning("no VGG annotations at %s; species names unverified", VGG_DIR)
        return 0
    checked = 0
    for f in sorted(os.listdir(split_dir / "labels"))[:limit]:
        stem = f[:-4]
        jp = VGG_DIR / (stem + ".json")
        if not jp.exists():
            continue
        rec = list(json.load(open(jp)).values())[0]
        regs = rec["regions"]
        regs = [regs] if isinstance(regs, dict) and "shape_attributes" in regs else \
            (list(regs.values()) if isinstance(regs, dict) else regs)
        yl = [l.split() for l in open(split_dir / "labels" / f) if l.strip()]
        if len(regs) != len(yl):
            continue
        for r, l in zip(regs, yl):
            cw = r["region_attributes"].get("CottonWeed", {})
            sp = [k for k, v in cw.items() if v] if isinstance(cw, dict) else []
            if len(sp) == 1:
                want = CWD12_TRUE[int(float(l[0]))].lower()
                assert sp[0].lower() == want, (
                    "cwd12 id %s is %r in the VGG annotations, not %r" % (l[0], sp[0], want))
                checked += 1
    log.info("species names verified against VGG on %d boxes", checked)
    return checked


# One model load and one forward pass per backbone; each pooling of it becomes
# its own feature set. CLS is what every existing tool in this project uses;
# the patch-token mean is tried because fine-grained species differences (leaf
# margin, venation) live in local tokens that the CLS token may average away.
BACKBONES = [
    ("dinov2b", "facebook/dinov2-base", ("cls", "mean")),
    ("dinov2l", "facebook/dinov2-large", ("cls", "mean")),
    ("bioclip2", "hf-hub:imageomics/bioclip-2", ("clip",)),
]
FEATURE_SETS = [f"{b}_{p}" if p != "clip" else b for b, _n, ps in BACKBONES for p in ps]
SEEDS = (0, 1, 2)
BUDGETS = (0.01, 0.02, 0.05, 0.10)


# --------------------------------------------------------------------- crops
def _holdout_guard():
    """(NearHashIndex, stem_set) of the sealed cwd12 test+valid images.

    Reuses mega_trainer's own guard so there is one definition of "holdout" in
    the project, not two that can drift apart."""
    from .mega_trainer import _iter_holdout_images, _dhash
    from .near_dup import NearHashIndex, HOLDOUT_NEAR_DUP_BITS
    hashes, stems = NearHashIndex(), set()
    n = 0
    for p in _iter_holdout_images():
        n += 1
        stems.add(p.stem)
        h = _dhash(p)
        if h is not None:
            hashes.add(h, p.name, max_bits=HOLDOUT_NEAR_DUP_BITS)   # v3.60.0
    if n < 1900:
        raise RuntimeError("holdout guard found only %d holdout images (expected 1,977); "
                           "refusing to run without a complete guard" % n)
    log.info("holdout guard: %d images, %d distinct dHashes, copies within %d bits blocked",
             n, len(hashes), HOLDOUT_NEAR_DUP_BITS)
    return hashes, stems


def _square_box(cx, cy, w, h, W, H):
    side = max(w * W, h * H) * (1 + 2 * MARGIN)
    x0 = cx * W - side / 2
    y0 = cy * H - side / 2
    return x0, y0, side


def cmd_crops(args):
    """Write the crop manifest for the cwd12 train split.

    Crops are not written to disk as files: 6,000 small JPEGs on Lustre is the
    kind of tree this project has learned not to create, and every crop can be
    re-cut from (image, box) on demand. The manifest is the durable artifact."""
    from PIL import Image
    from .mega_trainer import _dhash
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, hstems = _holdout_guard()
    n_verified = _verify_names(CWD12_TRAIN)
    img_dir, lab_dir = CWD12_TRAIN / "images", CWD12_TRAIN / "labels"
    names = sorted(os.listdir(img_dir))          # one bounded listdir, no rglob
    rows, dropped_holdout, dropped_small = [], 0, 0
    for f in names:
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lp = lab_dir / (stem + ".txt")
        if not lp.exists():
            continue
        ip = img_dir / f
        h = _dhash(ip)
        if stem in hstems or (h is not None and hashes.find(h) is not None):
            dropped_holdout += 1
            continue
        try:
            with Image.open(ip) as im:
                W, H = im.size
        except Exception:
            continue
        for k, line in enumerate(open(lp)):
            p = line.split()
            if len(p) < 5:
                continue
            oc = int(float(p[0]))
            if not 0 <= oc < 12:
                continue
            cx, cy, w, h = map(float, p[1:5])
            if w * W < MIN_BOX_PX or h * H < MIN_BOX_PX:
                dropped_small += 1
                continue
            rows.append({"crop_id": len(rows), "image": str(ip), "stem": stem,
                         "box": k, "cx": cx, "cy": cy, "w": w, "h": h,
                         "W": W, "H": H, "label": oc})
    bad = [r for r in rows if r["stem"] in hstems]
    assert not bad, "holdout stems in the crop manifest: %s" % bad[:3]
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    counts = np.bincount([r["label"] for r in rows], minlength=12)
    info = {"crops": len(rows), "images": len({r["stem"] for r in rows}),
            "dropped_holdout_images": dropped_holdout, "dropped_small_boxes": dropped_small,
            "names_verified_boxes": n_verified,
            "per_class": {CWD12_TRUE[i]: int(c) for i, c in enumerate(counts)}}
    json.dump(info, open(OUT / "manifest_info.json", "w"), indent=1)
    log.info("crops: %s", info)
    return info


def _load_manifest():
    with open(OUT / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("crop_id", "box", "W", "H", "label"):
            r[k] = int(r[k])
        for k in ("cx", "cy", "w", "h"):
            r[k] = float(r[k])
    return rows


def _cut(im, r):
    """The square crop for manifest row r, padded with neutral grey off-frame."""
    from PIL import Image
    x0, y0, side = _square_box(r["cx"], r["cy"], r["w"], r["h"], r["W"], r["H"])
    s = int(round(side))
    canvas = Image.new("RGB", (s, s), (124, 124, 124))
    ix0, iy0 = int(round(x0)), int(round(y0))
    box = (max(0, ix0), max(0, iy0), min(r["W"], ix0 + s), min(r["H"], iy0 + s))
    if box[2] > box[0] and box[3] > box[1]:
        canvas.paste(im.crop(box), (box[0] - ix0, box[1] - iy0))
    return canvas.resize((CROP_PX, CROP_PX), Image.BICUBIC)


# --------------------------------------------------------------------- embed
def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_backbone(name, device):
    """(model, preprocess) on `device`. The same calls as dinov2_curator's loader
    -- OpenCLIP for 'hf-hub:' names, transformers AutoModel otherwise -- but not
    tied to CUDA, so a run off the cluster uses the same code path."""
    if name.startswith("hf-hub:"):
        import open_clip
        model, _train_pre, preprocess = open_clip.create_model_and_transforms(name)
        return model.to(device).eval(), preprocess
    from transformers import AutoImageProcessor, AutoModel
    return (AutoModel.from_pretrained(name).to(device).eval(),
            AutoImageProcessor.from_pretrained(name))


def _embed_batch(model, proc, pils, poolings, device):
    """Embed one batch; returns {pooling: array with exactly len(pils) rows}.

    Written here rather than reusing dinov2_curator._embed_pils, which drops an
    image it cannot convert -- every row after the first failure would then carry
    the next crop's features and the wrong label."""
    import torch
    out = {}
    with torch.no_grad():
        if poolings == ("clip",):
            t = torch.stack([proc(p) for p in pils]).to(device)
            out["clip"] = model.encode_image(t).float().cpu().numpy()
        else:
            inp = proc(images=pils, return_tensors="pt").to(device)
            hs = model(pixel_values=inp["pixel_values"]).last_hidden_state
            if "cls" in poolings:
                out["cls"] = hs[:, 0].float().cpu().numpy()
            if "mean" in poolings:
                out["mean"] = hs[:, 1:].mean(1).float().cpu().numpy()
    for v in out.values():
        assert v.shape[0] == len(pils)
    return out


def cmd_embed(args):
    import torch
    from PIL import Image
    rows = _load_manifest()
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    y = np.array([r["label"] for r in rows])
    g = np.array([r["stem"] for r in rows])
    done = {}
    device = _device()
    log.info("embedding on %s", device)
    for bname, repo_id, poolings in BACKBONES:
        tags = {p: (f"{bname}_{p}" if p != "clip" else bname) for p in poolings}
        if args.only and not (set(tags.values()) & set(args.only)):
            continue
        if all((OUT / f"emb_{tg}.npz").exists() for tg in tags.values()) and not args.force:
            log.info("%s: cached", bname)
            continue
        model, proc = _load_backbone(repo_id, device)
        t0 = time.time()
        parts = {p: {} for p in poolings}
        buf_p, buf_i = [], []

        def flush():
            res = _embed_batch(model, proc, buf_p, poolings, device)
            for p in poolings:
                for i, v in zip(buf_i, res[p]):
                    parts[p][i] = v

        for img, rs in by_img.items():
            with Image.open(img) as im:
                im = im.convert("RGB")
                for r in rs:
                    buf_p.append(_cut(im, r))
                    buf_i.append(r["crop_id"])
            if len(buf_p) >= 64:
                flush(); buf_p, buf_i = [], []
        if buf_p:
            flush()
        for p, tg in tags.items():
            X = np.stack([parts[p][i] for i in range(len(rows))])
            np.savez_compressed(OUT / f"emb_{tg}.npz", X=X.astype(np.float16), y=y, groups=g)
            done[tg] = {"dim": int(X.shape[1])}
        secs = round(time.time() - t0, 1)
        log.info("%s: %d crops in %.0fs (%.1f crops/s)", bname, len(rows), secs, len(rows) / max(secs, 1e-6))
        for tg in tags.values():
            done[tg]["seconds"] = secs
            done[tg]["device"] = device
        model = proc = None
        if device == "cuda":
            torch.cuda.empty_cache()
    json.dump(done, open(OUT / "embed_info.json", "w"), indent=1)
    return done


# ------------------------------------------------------------------ evaluate
def _norm(X):
    X = X.astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def _supervised(X, y, groups):
    """Upper bound: what the features support when every training label is known.
    Grouped (by source image, or by capture session) so crops of one group never
    straddle folds."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    res = {"knn10": [], "linear": []}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        knn = KNeighborsClassifier(n_neighbors=10, weights="distance", metric="cosine").fit(X[tr], y[tr])
        lin = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
        for k, m in (("knn10", knn), ("linear", lin)):
            p = m.predict(X[te])
            res[k].append((accuracy_score(y[te], p), balanced_accuracy_score(y[te], p)))
    out = {}
    for k, v in res.items():
        a = np.array(v)
        out[k] = {"acc": [float(a[:, 0].mean()), float(a[:, 0].std())],
                  "bal_acc": [float(a[:, 1].mean()), float(a[:, 1].std())]}
    return out


def _purity(y, lab):
    m = lab >= 0
    if not m.any():
        return 0.0
    tot = 0
    for c in np.unique(lab[m]):
        tot += np.bincount(y[m][lab[m] == c]).max()
    return tot / m.sum()


def _discovery(Z, y):
    """Structure the features show with no labels at all."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
    out = {}
    for mcs in (15, 30, 60):
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(Z)
        m = lab >= 0
        out["hdbscan_mcs%d" % mcs] = {
            "clusters": int(len(set(lab[m]))), "noise": float(1 - m.mean()),
            "purity": float(_purity(y, lab)),
            "nmi": float(nmi(y[m], lab[m])) if m.any() else 0.0,
            "ari": float(ari(y[m], lab[m])) if m.any() else 0.0}
    for k in (12, 24, 48):
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Z)
        out["kmeans_k%d" % k] = {"clusters": k, "purity": float(_purity(y, lab)),
                                 "nmi": float(nmi(y, lab)), "ari": float(ari(y, lab))}
    return out


def _propagate(Z, y_known, known_mask, n_classes=12):
    """Spread the known names to every crop. Returns (pred, per-crop margin)."""
    from sklearn.semi_supervised import LabelSpreading
    yy = np.where(known_mask, y_known, -1)
    ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=60)
    ls.fit(Z, yy)
    # classes_ holds only the names present among the known crops; a species
    # nobody named yet gets probability 0, which is the honest answer
    full = np.zeros((len(Z), n_classes))
    full[:, ls.classes_] = ls.label_distributions_
    srt = np.sort(full, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return full.argmax(1), margin


def _medoids(Z, k, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Z)
    idx = []
    for c in range(k):
        mem = np.where(km.labels_ == c)[0]
        if len(mem):
            d = np.linalg.norm(Z[mem] - km.cluster_centers_[c], axis=1)
            idx.append(int(mem[d.argmin()]))
    return np.array(sorted(set(idx)))


def _budget(Z, y, frac, seed, strategy):
    """Name `frac` of the crops (the oracle answers with the hidden truth, which
    is what a person looking at the crop would do), propagate, score the rest."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    n = len(y)
    B = max(12, int(round(frac * n)))
    rng = np.random.default_rng(seed)
    if strategy == "random":
        known = rng.choice(n, B, replace=False)
    elif strategy == "centroid":
        known = _medoids(Z, B, seed)
    elif strategy == "centroid_uncertainty":
        # half the budget on exemplars, the other half on the crops the
        # propagation is least sure about, in two rounds
        known = _medoids(Z, B // 2, seed)
        for rnd in range(2):
            mask = np.zeros(n, bool); mask[known] = True
            _, margin = _propagate(Z, y, mask)
            margin[known] = np.inf
            take = (B - len(known)) if rnd == 1 else (B - len(known)) // 2
            known = np.concatenate([known, np.argsort(margin)[:max(0, take)]])
    else:
        raise ValueError(strategy)
    mask = np.zeros(n, bool); mask[known] = True
    pred, _ = _propagate(Z, y, mask)
    rest = ~mask
    return (float(accuracy_score(y[rest], pred[rest])),
            float(balanced_accuracy_score(y[rest], pred[rest])), int(mask.sum()))


def cmd_evaluate(args):
    from sklearn.decomposition import PCA
    # --only evaluates some feature sets and keeps the others already in
    # results.json, so feature sets can be scored as their embeddings land.
    prev = OUT / "results.json"
    results = json.load(open(prev)) if (args.only and prev.exists()) else {}
    results["meta"] = {"seeds": list(SEEDS), "budgets": list(BUDGETS),
                       "manifest": json.load(open(OUT / "manifest_info.json"))}
    for tag in FEATURE_SETS:
        path = OUT / ("emb_%s.npz" % tag)
        if not path.exists() or (args.only and tag not in args.only):
            continue
        d = np.load(path, allow_pickle=True)
        X, y, groups = _norm(d["X"]), d["y"].astype(int), d["groups"]
        Z = PCA(n_components=50, random_state=0).fit_transform(X)
        Z = _norm(Z)
        t0 = time.time()
        # cwd12 was shot in sessions of consecutive frames (date_camera_person,
        # 35 in the train split), so one plant recurs in neighbouring photos.
        # Grouping by photo can put those near-twins on both sides of a fold;
        # grouping by session holds out whole field days, the stricter ceiling.
        sessions = np.array([str(g).rsplit("_", 1)[0] for g in groups])
        r = {"supervised": _supervised(X, y, groups),
             "supervised_by_session": _supervised(X, y, sessions),
             "discovery": _discovery(Z, y), "budget": {}}
        for frac in BUDGETS:
            for strat in ("random", "centroid", "centroid_uncertainty"):
                v = np.array([_budget(Z, y, frac, s, strat) for s in SEEDS])
                r["budget"]["%g/%s" % (frac, strat)] = {
                    "labelled": int(v[0, 2]),
                    "acc": [float(v[:, 0].mean()), float(v[:, 0].std())],
                    "bal_acc": [float(v[:, 1].mean()), float(v[:, 1].std())]}
        r["seconds"] = round(time.time() - t0, 1)
        results[tag] = r
        log.info("%s: linear %.3f  knn %.3f  best-5%% %s", tag,
                 r["supervised"]["linear"]["acc"][0], r["supervised"]["knn10"]["acc"][0],
                 r["budget"]["0.05/centroid_uncertainty"]["acc"])
        json.dump(results, open(OUT / "results.json", "w"), indent=1)
    return results


def cmd_all(args):
    cmd_crops(args)
    cmd_embed(args)
    cmd_evaluate(args)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="semisup_labeler")
    ap.add_argument("cmd", choices=["crops", "embed", "evaluate", "all"])
    ap.add_argument("--only", nargs="*", default=None, help="feature-set tags to embed / evaluate")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"crops": cmd_crops, "embed": cmd_embed, "evaluate": cmd_evaluate, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
