"""V3F D0-S: the residual-copy sweep aimed at the copy detector's blind spots,
crops and letterboxing (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F,
the v3 experiment family D0-D9", V3F.5 D0-S; decision 1).

    python -m weed_optimizer_framework.tools.inc2.v3fam_sweep run [--procs N]
    python -m weed_optimizer_framework.tools.inc2.v3fam_sweep verdict --pair ID --copy yes|no --read-by owner|delegate
    python -m weed_optimizer_framework.tools.inc2.v3fam_sweep status

Why. The never-train guards match a copy by dHash (the 8 flips and
rotations, 6 bits) and by the calibrated DINOv2 detector (LOCK v2,
facebook/dinov2-base:cls, cosine 0.946384), whose calibrated recall is 0.7185
for crops and 0.30 for a letterbox to 640. Base B adds 37,674 external rows
to E1-A's 6,811, so a cropped or letterboxed re-upload of a dev or test image
among them would inflate E3-B's numbers and pass both guards. The sweep looks
for such copies with pixels only; no label file is opened and nothing is
scored.

Sets (fixed by the amendment). Q = dev and test of LOCK v1 (617 + 1,977 =
2,594 images, keys "dev/<key>", "test/<key>"). R = base B's rows
(splits/v3/base_v3_weed.jsonl, sha256 18d24ead...899a) that are not
base_v2's train_core, tsw22 or tsw23 rows: its 37,674 external rows plus
base_v2's 66 NDSU rows (37,740), each read as the file base B trains on.

Procedure (V3F.5 D0-S, fixed; every number below is a constant here):
  1 trim       every image is EXIF-transposed, then from each side the lines
               (rows, then columns within the rows kept) whose grey values all
               lie within TRIM_TOL (8) of that side's median border grey and
               whose standard deviation is under TRIM_STD (2) are stripped,
               moving inward until one fails (trim_box)
  2 describe   DINOv2-base CLS under the guard's calibrated view
               (funnel.embed's: EXIF transpose, shorter edge 256, the model's
               processor). Q: 8 views of the trimmed image (VIEWS: whole,
               horizontal flip, centre crops 0.9 and 0.8, the four 0.8
               corners); R: the trimmed whole image. A reference row with no
               trimmed border reuses base v3's stored descriptor
               (splits/v3/leak/scan_desc.npz, matched to the row by its 8
               stored dHashes) if REUSE_CHECK (100) seeded rows recomputed
               all give cosine >= REUSE_COS (0.999); else every row is
               recomputed (reuse_descriptors)
  3 shortlist  per query view its SHORTLIST_TOP (5) nearest reference rows
               by cosine, kept at cosine >= SHORTLIST_COS (0.70); plus every
               pair within SHORTLIST_BITS (10) under the 8-variant dHash in
               both directions on the trimmed images (shortlist)
  4 verify     OpenCV SIFT (4,000 features) on greyscale images resized to
               long side 1,024, Lowe ratio 0.8, a RANSAC homography
               (reprojection 5 px, 2,000 iterations, confidence 0.995), each
               pair also with the reference flipped; verified at >=
               INLIERS_MIN (50) inliers (sift_inliers)
  5 calibrate  five copy families (centre crop 0.9 and 0.8, letterbox to
               640 with grey 124 padding, JPEG quality 75, horizontal flip)
               of CALIB_N (200) dev and test images ranked by
               stable_int("inc2/v3fam/sweep/calib/<key>"), made in memory and
               placed in a scratch reference index beside R (never a training
               list): recall per family; and 200 non-copy pairs (each
               calibration image with the next one in rank order from
               another session): false verifications. Recorded only: no
               threshold moves (calibrate)
  6 verdict    every verified pair gets a side-by-side montage (cosine and
               inliers on it) and a verdict by a named viewer (READ_BY):
               "copy" when the two images show the same photograph (the same
               scene, plants and capture) up to cropping, resizing, padding,
               flipping, rotation or re-encoding; "no" otherwise, which
               includes two photographs of the same plant or field taken at
               different moments. Each verdict is written once.
A true copy is a verified pair with the verdict "copy"; H = the query images
with a true copy.

Trigger (pre-registered; trigger()). The first two branches apply together:
  H holds a test image  the re-analysis job scores E3-A, E3-M, E3-B and the
                        reference on test minus H from stored arrays; BCLEAN
                        runs (E1-B's manifest minus the copies' rows, one-class
                        m640 at cold_budget, 3 seeds; L23I D0BCLEAN first in
                        step 8, L23J BCLEAN first in step 10, one test read
                        inside the cap, taken before any challenger);
  H holds a dev image   every family dev comparison (P1-P5, the Tier C gate,
                        Tier S) is decided on dev minus H (the same images for
                        every arm, bootstraps and selection correction there),
                        full dev reported beside, E3's recorded dev
                        comparisons re-read on dev minus H (reported only);
  H not empty           the copies' reference rows leave every family training
                        set before L23H writes a manifest (removal_keys, with
                        the list's sha256), D6's base_v2 staging included;
  H empty               the record says so; the paper states the yield and the
                        calibrated recall.

Records (under INC_DIR/v3fam/v1/sweep/, each written once): run.json (the
sets and their sha256s, counts per step, every shortlisted pair with its
cosine, bits and inliers, the calibration), montages/<pair>.jpg (verified
pairs), verdicts/<pair>.json, and record.json (the run, the verdicts and
viewers, H, the trigger, the removal list): written by run when nothing is
verified, else by the verdict that completes the set. L23H is not proposed
before record.json exists (complete()); v3fam_data reads it (read_record,
removal_keys). Descriptor caches (desc/*.npz) are working files.

The time stamp: run refuses unless v3fam/v1/timestamp.json exists (V3F.3.9,
V3F.12: the amendment is time-stamped before any family step). Every run keeps
the E-series trees unchanged (F.e_trees_unchanged).

What the platform registration needs (not in this module; V3F.6.2-6.4): the
delegate's verb "inc2.v3fam_sweep run" in run_inc2_build.sh (job
inc_build_v3fam_sweep, provenance and lock v3fam_sweep, HF_HUB_OFFLINE, an
import check of cv2 with SIFT_create and of transformers, this module in the
drift list), refused by stream_remote's build grammar (a person's or the
delegate's step, never a lever); its GPU-h (about 1.5) counted in the family
cap; verdicts entered on the login node; _v3fam_gate's "D0-S's record is
complete" is complete(), and the record's trigger decides whether D0BCLEAN and
BCLEAN come first in steps 8 and 10 and whether dev decisions use dev minus H
(record["trigger"]["dev_minus_h"], record["H"]).

Public API: trim_box, trimmed, view_crop, VIEWS, copy_image, CALIB_FAMILIES,
query_rows, reference_rows, prepare_set, reuse_descriptors, shortlist,
sift_inliers, verify_pairs, pipeline, calibrate, run, verdict, read_record,
complete, trigger, removal_keys, pair_id. A refusal raises SweepRefused (a
V3famRefused); the CLI prints "[inc2.v3fam_sweep] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

from ..inc import common as C
from . import base3 as B3
from . import common as C2
from . import v3fam as F

# ------------------------------------------------------------------ constants (pre-registered, V3F.5 D0-S)
NAME = "sweep"
FORMAT_RUN = "inc2-v3fam-sweep-run/1"
FORMAT_VERDICT = "inc2-v3fam-sweep-verdict/1"
FORMAT_RECORD = "inc2-v3fam-sweep-record/1"
QUERY_EXAMS = ("dev", "test")
QUERY_IMAGES = 2594                          # 617 + 1,977
BASE_V2_SOURCES = ("cottonweeddet12/train", "3seasonweeddet10/data2022", "3seasonweeddet10/data2023")
NDSU_SOURCES = ("project_agml__weed_crop_detection", "project_agml__imageweeds_aerial_weed_detection")
REFERENCE_ROWS = 37740                       # base B's 37,674 external rows + base_v2's 66 NDSU rows
BASE_B_SHA256 = "18d24eade70ba2a5c3c39061eaacd30d148fa00d5ac28fc5685479a7d104899a"
TRIM_TOL = 8                                 # grey levels from the side's median border grey
TRIM_STD = 2.0                               # a stripped line's standard deviation is under this
VIEWS = ("whole", "hflip", "centre90", "centre80", "corner80_tl", "corner80_tr", "corner80_bl", "corner80_br")
REUSE_CHECK = 100
REUSE_COS = 0.999
REUSE_SEED_TEXT = "inc2/v3fam/sweep/reuse/%s"
SHORTLIST_TOP = 5
SHORTLIST_COS = 0.70
SHORTLIST_BITS = 10
SIFT_FEATURES = 4000
SIFT_LONG = 1024
LOWE_RATIO = 0.8
RANSAC_PX = 5.0
RANSAC_ITERS = 2000
RANSAC_CONF = 0.995
INLIERS_MIN = 50
CALIB_N = 200
CALIB_NEG = 200
CALIB_SEED_TEXT = "inc2/v3fam/sweep/calib/%s"
CALIB_FAMILIES = ("centre90", "centre80", "letterbox640", "jpeg75", "hflip")
LETTERBOX = 640
GREY = 124
JPEG_QUALITY = 75
EMBEDDER = "facebook/dinov2-base:cls"        # LOCK v2's detector (its calibration's embedder)
VERDICTS = ("yes", "no")
CHUNK = 1024


class SweepRefused(F.V3famRefused):
    pass


def log(msg):
    print("[inc2.v3fam_sweep] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths
def sweep_dir():
    return F.root() / NAME


def timestamp_path():
    return F.root() / "timestamp.json"


def require_timestamp():
    """The amendment's time stamp record (V3F.12), which every family step needs first: its sha256. It must also
    name the stamp of Revisions 29-31 ({"revisions": [{"file", "sha256", ...}]} with the sha256 of
    F.REVISION_FILE), which V3F.12 has stamped before the next family step."""
    p = timestamp_path()
    doc = F._read_json(p)
    if not isinstance(doc, dict):
        raise SweepRefused("%s does not exist: the amendment is time-stamped before any family step (V3F.3.9, "
                           "V3F.12)" % p)
    revs = [r for r in doc.get("revisions") or [] if isinstance(r, dict)]
    if not any(r.get("sha256") == F.REVISION_SHA256 for r in revs):
        raise SweepRefused("%s names no stamp of %s (sha256 %s): Revisions 29-31 are time-stamped before the next "
                           "family step (V3F.12)" % (p, F.REVISION_FILE, F.REVISION_SHA256[:16]))
    return C.sha256_file(p)


def pair_id(query_key, ref_key):
    """A pair's name from its content (query key, reference key), never from positions."""
    return "p" + hashlib.sha256(("%s|%s" % (query_key, ref_key)).encode("utf-8")).hexdigest()[:12]


def list_sha256(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ pixels (no label is ever opened)
def load_rgb(path):
    """The image at path, EXIF-transposed, as RGB (the calibrated view's first step)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im0:
        im0.load()
        return ImageOps.exif_transpose(im0).convert("RGB")


def _first_fail(dev_max, sd):
    import numpy as np
    bad = (dev_max > TRIM_TOL) | (sd >= TRIM_STD)
    idx = np.flatnonzero(bad)
    return int(idx[0]) if len(idx) else len(bad)


def trim_box(img):
    """(x0, y0, x1, y1) of an image's content after the letterbox trim. Each side in turn (top, bottom, left,
    right, on the box kept so far) strips lines whose grey values all lie within TRIM_TOL of that side's median
    border grey (its outermost line's median) and whose standard deviation is under TRIM_STD, moving inward until
    one fails; the four sides repeat until none moves (nested borders of different greys). An image that would
    trim to nothing keeps its full frame."""
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if h < 2 or w < 2:
        return 0, 0, w, h

    def side(lines):
        # lines: [n, m], line 0 the outermost
        c = float(np.median(lines[0]))
        return _first_fail(np.abs(lines - c).max(axis=1), lines.std(axis=1))
    x0, y0, x1, y1 = 0, 0, w, h
    while True:
        b = g[y0:y1, x0:x1]
        t = side(b)
        bo = side(b[::-1]) if t < len(b) else 0
        if t + bo >= y1 - y0:
            return 0, 0, w, h
        ny0, ny1 = y0 + t, y1 - bo
        b = g[ny0:ny1, x0:x1]
        le = side(b.T)
        ri = side(b.T[::-1]) if le < b.shape[1] else 0
        if le + ri >= x1 - x0:
            return 0, 0, w, h
        nx0, nx1 = x0 + le, x1 - ri
        if (nx0, ny0, nx1, ny1) == (x0, y0, x1, y1):
            return x0, y0, x1, y1
        x0, y0, x1, y1 = nx0, ny0, nx1, ny1


def trimmed(img, box=None):
    box = trim_box(img) if box is None else tuple(int(v) for v in box)
    return img if box == (0, 0) + img.size else img.crop(box)


def view_crop(img, view):
    """One of the 8 query views of a (trimmed) image."""
    from PIL import Image
    W, H = img.size
    if view == "whole":
        return img
    if view == "hflip":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if view in ("centre90", "centre80"):
        f = 0.9 if view == "centre90" else 0.8
        w, h = max(1, int(round(W * f))), max(1, int(round(H * f)))
        x0, y0 = (W - w) // 2, (H - h) // 2
        return img.crop((x0, y0, x0 + w, y0 + h))
    if view.startswith("corner80_"):
        w, h = max(1, int(round(W * 0.8))), max(1, int(round(H * 0.8)))
        x0 = 0 if view.endswith(("tl", "bl")) else W - w
        y0 = 0 if view.endswith(("tl", "tr")) else H - h
        return img.crop((x0, y0, x0 + w, y0 + h))
    raise SweepRefused("no view %r (%s)" % (view, list(VIEWS)))


def copy_image(img, family):
    """A calibration copy (and D5b's catch index's) of an EXIF-transposed RGB image, in memory: centre90,
    centre80, letterbox640 (long side 640, grey 124 padding to 640 x 640, centred), jpeg75, hflip."""
    from PIL import Image
    W, H = img.size
    if family in ("centre90", "centre80", "hflip"):
        return view_crop(img, family)
    if family == "letterbox640":
        r = LETTERBOX / float(max(W, H))
        w, h = max(1, int(round(W * r))), max(1, int(round(H * r)))
        out = Image.new("RGB", (LETTERBOX, LETTERBOX), (GREY, GREY, GREY))
        out.paste(img.resize((w, h), Image.BILINEAR), ((LETTERBOX - w) // 2, (LETTERBOX - h) // 2))
        return out
    if family == "jpeg75":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        with Image.open(buf) as j:
            j.load()
            return j.convert("RGB")
    raise SweepRefused("no copy family %r (%s)" % (family, list(CALIB_FAMILIES)))


def row_image(row):
    """The EXIF-transposed RGB image a row stands for: its file, or (a calibration copy) row["make"] =
    {"source": path, "family": f} made in memory."""
    mk = row.get("make")
    if mk:
        return copy_image(load_rgb(mk["source"]), mk["family"])
    return load_rgb(row["image"])


def prepare_view(row):
    """funnel.embed's prepare hook (a module-level function, so worker processes run it): (the view image, the
    8 dHash variants of the trimmed image, {"box", "size"}). row["spec"] = {"view", "box" (optional: else the
    trim is computed)}."""
    from ..funnel import leak as L
    img = row_image(row)
    spec = row.get("spec") or {}
    box = spec.get("box")
    box = trim_box(img) if box is None else tuple(int(v) for v in box)
    t = trimmed(img, box)
    hv = L.dhash_variants(t)
    return view_crop(t, spec.get("view", "whole")), [hv[v] for v in L.VARIANTS], {"box": list(box),
                                                                                    "size": list(img.size)}


def _trim_task(row):
    """Worker: (key, box, size, 8 trimmed dHashes, 8 stored-pixel dHashes, error)."""
    try:
        from PIL import Image
        from ..funnel import leak as L
        if row.get("make"):
            img = row_image(row)
            stored = None
        else:
            from PIL import ImageOps
            with Image.open(row["image"]) as im0:
                im0.load()
                sv = L.dhash_variants(im0)
                stored = [int(sv[v]) for v in L.VARIANTS]
                img = ImageOps.exif_transpose(im0).convert("RGB")
        box = trim_box(img)
        hv = L.dhash_variants(trimmed(img, box))
        return row["key"], list(box), list(img.size), [int(hv[v]) for v in L.VARIANTS], stored, None
    except Exception as e:  # noqa: BLE001 - an image that cannot be read is recorded, never a pass
        return row["key"], None, None, None, None, "%s: %s" % (type(e).__name__, e)


# ------------------------------------------------------------------ the sets
def query_rows():
    """Q: dev and test of LOCK v1 (each manifest checked against the LOCK), pixels only: [{"key": "<exam>/<key>",
    "exam", "exam_key", "image", "sha256", "session"}] sorted by key."""
    lock = C.read_lock()
    out = []
    for exam in QUERY_EXAMS:
        C.verify_manifest_against_lock(exam, lock)
        for r in C.read_manifest(C.manifest_path(exam)):
            out.append({"key": "%s/%s" % (exam, r["key"]), "exam": exam, "exam_key": r["key"], "image": r["image"],
                        "sha256": r["sha256"], "session": r.get("session") or ""})
    return sorted(out, key=lambda r: r["key"])


def reference_rows(base_rows=None, production=True):
    """R: base B's rows that are not base_v2's train_core, tsw22 or tsw23 rows (its external rows and base_v2's
    NDSU rows), as base B trains them: [{"key", "image", "sha256", "source"}] sorted by key, and the record."""
    rec = {}
    if base_rows is None:
        summ = B3.read_summary()
        if summ is None:
            raise SweepRefused("no complete splits/v3/summary.json: base B is not built")
        arm = summ["arms"]["B"]
        path = Path(arm["manifest"])
        sha = C.sha256_file(path)
        if sha != arm["sha256"]:
            raise SweepRefused("%s hashes to %s, its summary records %s" % (path, sha[:12], arm["sha256"][:12]))
        if production and sha != BASE_B_SHA256:
            raise SweepRefused("base B's manifest is %s, the amendment's is %s" % (sha[:12], BASE_B_SHA256[:12]))
        base_rows = C.read_manifest(path)
        rec = {"manifest": str(path), "sha256": sha, "rows": len(base_rows)}
    out = [{"key": r["key"], "image": r["image"], "sha256": r["sha256"], "source": r["source"]}
           for r in base_rows if r["source"] not in BASE_V2_SOURCES]
    out.sort(key=lambda r: r["key"])
    if production and len(out) != REFERENCE_ROWS:
        raise SweepRefused("R holds %d rows, the amendment's %d" % (len(out), REFERENCE_ROWS))
    rec.update(reference_rows=len(out), ndsu_rows=sum(1 for r in out if r["source"] in NDSU_SOURCES),
               keys_sha256=list_sha256([r["key"] for r in out]))
    return out, rec



# ------------------------------------------------------------------ steps 1-2: trim, hashes, descriptors
def _pool_map(fn, items, procs):
    from ..inc import verify as V
    with V._Workers(procs) as w:
        return list(w.imap(fn, items, chunksize=8))


def trim_set(rows, procs=1):
    """{key: {"box", "size", "hashes" (trimmed), "stored" (stored pixels) }} of every row; rows that cannot be
    read are listed in the second value (fail closed: the caller refuses)."""
    got = _pool_map(_trim_task, rows, procs)
    info, bad = {}, []
    for key, box, size, hv, stored, err in got:
        if err is not None:
            bad.append([key, err])
        else:
            info[key] = {"box": box, "size": size, "hashes": hv, "stored": stored,
                         "border": box != [0, 0] + list(size)}
    return info, bad


def _normalise(X):
    import numpy as np
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def describe(rows, info, views, embedder, path=None, procs=1):
    """Normalised descriptors [len(rows) x len(views), dim] (row-major: row i's views at i * len(views) + v) of
    each row's trimmed image under each view, through funnel.embed's calibrated view. Fail closed: a row that
    cannot be described refuses."""
    import numpy as np
    from ..funnel import embed as E
    items = []
    for r in rows:
        for v in views:
            it = {"key": "%s|%s" % (r["key"], v), "image": r.get("image") or (r.get("make") or {}).get("source"),
                  "sha256": r.get("sha256"), "spec": {"view": v, "box": info[r["key"]]["box"]}}
            if r.get("make"):
                it["make"] = r["make"]
            items.append(it)
    if not items:
        return np.zeros((0, int(embedder.dim)), dtype=np.float32)
    res = E.embed_images(items, path, embedder, prepare=prepare_view, n_hashes=8, procs=procs)
    X = np.asarray(res["X"], dtype=np.float32)
    bad = [items[i]["key"] for i in np.flatnonzero(~np.isfinite(X).all(axis=1))]
    if bad:
        raise SweepRefused("%d view(s) cannot be described (e.g. %s): an image the sweep cannot compare is one it "
                           "cannot clear" % (len(bad), bad[:3]))
    return _normalise(X)


def stored_descriptors(path):
    """(H [n, 8] uint64, X normalised [n, dim]) of base v3's scan_desc.npz (the embedding scan's descriptors of
    every external candidate, keyed by row uid, with each row's 8 stored-pixel dHashes)."""
    import numpy as np
    with np.load(str(path), allow_pickle=False) as z:
        H = np.asarray(z["H"], dtype=np.uint64)
        X = np.asarray(z["X"], dtype=np.float32)
        ok = np.asarray(z["hash_ok"], dtype=bool) if "hash_ok" in z.files else np.ones(len(H), bool)
    good = ok & np.isfinite(X).all(axis=1)
    return H[good], _normalise(X[good])


def reuse_descriptors(rows, info, embedder, scan_desc, procs=1, path=None):
    """Step 2's reuse rule for R. A row with no trimmed border whose 8 stored-pixel dHashes name exactly one
    stored descriptor (or several equal ones, cosine >= REUSE_COS between them) can reuse it; REUSE_CHECK of
    those rows, ranked by stable_int(REUSE_SEED_TEXT % key), are recomputed, and the reuse is taken only if
    every one gives cosine >= REUSE_COS to its stored descriptor. Returns ({key: normalised descriptor} of the
    rows reused, record); an empty dict means every row is recomputed."""
    rec = {"file": None, "candidates": 0, "check": None, "reused": 0}
    if scan_desc is None or not Path(scan_desc).is_file():
        rec["why"] = "no stored descriptors"
        return {}, rec
    rec["file"] = {"path": str(scan_desc), "sha256": C.sha256_file(scan_desc)}
    H, X = stored_descriptors(scan_desc)
    by = collections.defaultdict(list)
    for j, h in enumerate(H.tolist()):
        by[tuple(int(x) for x in h)].append(j)
    cand = {}
    for r in rows:
        inf = info[r["key"]]
        if inf["border"] or inf["stored"] is None:
            continue
        js = by.get(tuple(int(x) for x in inf["stored"]))
        if not js:
            continue
        if len(js) > 1 and float((X[js] @ X[js[0]]).min()) < REUSE_COS:
            continue
        cand[r["key"]] = X[js[0]]
    rec["candidates"] = len(cand)
    if not cand:
        rec["why"] = "no row maps to a stored descriptor"
        return {}, rec
    pick = sorted(cand, key=lambda k: (C.stable_int(REUSE_SEED_TEXT % k), k))[:REUSE_CHECK]
    by_key = {r["key"]: r for r in rows}
    fresh = describe([by_key[k] for k in pick], info, ("whole",), embedder, path=path, procs=procs)
    cos = [float(fresh[i] @ cand[k]) for i, k in enumerate(pick)]
    rec["check"] = {"rows": len(pick), "min_cos": round(min(cos), 6), "below": sum(1 for c in cos if c < REUSE_COS),
                    "threshold": REUSE_COS}
    if any(c < REUSE_COS for c in cos):
        rec["why"] = "%d of %d checked rows below %.3f: every row is recomputed" % (rec["check"]["below"],
                                                                                  len(pick), REUSE_COS)
        return {}, rec
    rec["reused"] = len(cand)
    return cand, rec


def prepare_set(rows, views, embedder, desc_path=None, procs=1, reuse=None, info=None):
    """Steps 1-2 for one set: the trim of every row, its trimmed dHashes and its descriptors under `views`
    (reuse: {key: descriptor} for rows described elsewhere, whole view only; info: trim_set's, when the caller
    has it). Returns {"rows", "info", "X" [n x views, dim], "H" [n, 8], "counts", "views", "index"}."""
    import numpy as np
    if info is None:
        info, bad = trim_set(rows, procs=procs)
        if bad:
            raise SweepRefused("%d image(s) cannot be read (e.g. %s): the sweep fails closed" % (len(bad),
                                                                                                bad[:3]))
    reuse = reuse or {}
    todo = [r for r in rows if r["key"] not in reuse]
    Xt = describe(todo, info, views, embedder, path=desc_path, procs=procs)
    dim = Xt.shape[1] if len(Xt) else (len(next(iter(reuse.values()))) if reuse else int(embedder.dim))
    X = np.zeros((len(rows) * len(views), dim), dtype=np.float32)
    pos = {r["key"]: i for i, r in enumerate(todo)}
    for i, r in enumerate(rows):
        if r["key"] in reuse:
            X[i * len(views)] = reuse[r["key"]]
        else:
            k = pos[r["key"]]
            X[i * len(views):(i + 1) * len(views)] = Xt[k * len(views):(k + 1) * len(views)]
    H = np.asarray([info[r["key"]]["hashes"] for r in rows], dtype=np.uint64).reshape(-1, 8)
    counts = {"rows": len(rows), "views": len(rows) * len(views), "trimmed": sum(1 for r in rows
                                                                                  if info[r["key"]]["border"]),
              "described": len(todo) * len(views), "reused": len(rows) - len(todo)}
    return _index({"rows": rows, "info": info, "X": X, "H": H, "counts": counts, "views": tuple(views)})


# ------------------------------------------------------------------ step 3: shortlist
def shortlist(qset, rset, top=SHORTLIST_TOP, cos_min=SHORTLIST_COS, bits=SHORTLIST_BITS):
    """{(q, r): {"cos", "view", "bits"}}: for each query view its `top` nearest reference rows by cosine, kept at
    cosine >= cos_min (the best view's cosine recorded), plus every pair within `bits` under the 8-variant dHash
    in both directions on the trimmed images (the fewest bits recorded)."""
    import numpy as np
    nv = len(qset["views"])
    QX, RX = qset["X"], rset["X"]
    out = {}
    nr = len(rset["rows"])
    if nr and len(QX):
        k = min(int(top), nr)
        for s in range(0, len(QX), CHUNK):
            S = QX[s:s + CHUNK] @ RX.T
            tk = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for a in range(len(S)):
                qi, vi = divmod(s + a, nv)
                for j in tk[a].tolist():
                    c = float(S[a, j])
                    if c >= cos_min:
                        e = out.setdefault((qi, j), {"cos": None, "view": None, "bits": None})
                        if e["cos"] is None or c > e["cos"]:
                            e["cos"], e["view"] = round(c, 6), qset["views"][vi]
    HQ, HR = qset["H"], rset["H"]
    okq, okr = np.ones(len(HQ), bool), np.ones(len(HR), bool)
    if len(HQ) and len(HR):
        Q1, T1, D1, _v = B3.near_pairs_between(HQ, okq, HR, int(bits))
        T2, Q2, D2, _w = B3.near_pairs_between(HR, okr, HQ, int(bits))
        for q, r, d in zip(np.r_[Q1, Q2].tolist(), np.r_[T1, T2].tolist(), np.r_[D1, D2].tolist()):
            e = out.setdefault((int(q), int(r)), {"cos": None, "view": None, "bits": None})
            if e["bits"] is None or d < e["bits"]:
                e["bits"] = int(d)
    return out


# ------------------------------------------------------------------ step 4: SIFT verification
def _grey_1024(img):
    import numpy as np
    import cv2
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = g.shape
    r = SIFT_LONG / float(max(h, w))
    if r != 1.0:
        size = (max(1, int(round(w * r))), max(1, int(round(h * r))))
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return g


def _sift(g):
    import cv2
    s = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    return s.detectAndCompute(g, None)


def _inliers(fa, fb):
    import numpy as np
    import cv2
    ka, da = fa
    kb, db = fb
    if da is None or db is None or len(ka) < 4 or len(kb) < 2:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]
    if len(good) < 4:
        return 0
    src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(0)
    _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
    return 0 if mask is None else int(mask.sum())


def sift_inliers(img_q, img_r):
    """(inliers, inliers with the reference flipped) of a pair of (trimmed) images."""
    from PIL import Image
    fq = _sift(_grey_1024(img_q))
    a = _inliers(fq, _sift(_grey_1024(img_r)))
    b = _inliers(fq, _sift(_grey_1024(img_r.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
    return a, b


def _verify_task(task):
    """Worker: one query against its shortlisted references: [(ref key, inliers, flipped inliers, error)]."""
    q, qbox, refs = task
    out = []
    try:
        iq = trimmed(row_image(q), qbox)
    except Exception as e:  # noqa: BLE001 - recorded; the caller refuses
        return [(r["key"], None, None, "query %s: %s" % (type(e).__name__, e)) for r, _b in refs]
    from PIL import Image
    fq = _sift(_grey_1024(iq))
    for r, rbox in refs:
        try:
            ir = trimmed(row_image(r), rbox)
            a = _inliers(fq, _sift(_grey_1024(ir)))
            b = _inliers(fq, _sift(_grey_1024(ir.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
            out.append((r["key"], a, b, None))
        except Exception as e:  # noqa: BLE001
            out.append((r["key"], None, None, "%s: %s" % (type(e).__name__, e)))
    return out


def verify_pairs(qset, rset, cands, procs=1):
    """Step 4 over the shortlist: [{"query", "ref", "cos", "view", "bits", "inliers", "inliers_flipped",
    "verified"}] sorted by (query, ref). Fail closed: a pair whose images cannot be read refuses."""
    by_q = collections.defaultdict(list)
    for (qi, rj) in sorted(cands):
        by_q[qi].append(rj)
    tasks = []
    for qi, rjs in sorted(by_q.items()):
        q = qset["rows"][qi]
        tasks.append((q, qset["info"][q["key"]]["box"],
                      [(rset["rows"][j], rset["info"][rset["rows"][j]["key"]]["box"]) for j in rjs]))
    got = _pool_map(_verify_task, tasks, procs)
    out, bad = [], []
    for (q, _qb, refs), res in zip(tasks, got):
        qi = qset["index"][q["key"]]
        for (r, _rb), (rkey, a, b, err) in zip(refs, res):
            if err is not None:
                bad.append([q["key"], rkey, err])
                continue
            e = cands[(qi, rset["index"][rkey])]
            best = max(a, b)
            out.append({"pair": pair_id(q["key"], rkey), "query": q["key"], "ref": rkey, "cos": e["cos"],
                        "view": e["view"], "bits": e["bits"], "inliers": a, "inliers_flipped": b,
                        "verified": best >= INLIERS_MIN})
    if bad:
        raise SweepRefused("%d pair(s) cannot be verified (e.g. %s): the sweep fails closed" % (len(bad), bad[:3]))
    return sorted(out, key=lambda p: (p["query"], p["ref"]))


def _index(s):
    s["index"] = {r["key"]: i for i, r in enumerate(s["rows"])}
    return s


def pipeline(q_rows, r_rows, embedder, work_dir=None, procs=1, reuse=None, qset=None, rset=None):
    """Steps 1-4 for any query and reference sets (the sweep, its calibration, D5b's catch index): {"qset",
    "rset", "candidates", "pairs", "counts"}. qset / rset: sets prepare_set already made."""
    wd = Path(work_dir) if work_dir else None
    if qset is None:
        qset = prepare_set(q_rows, VIEWS, embedder, desc_path=(wd / "query.npz") if wd else None, procs=procs)
    if rset is None:
        rset = prepare_set(r_rows, ("whole",), embedder, desc_path=(wd / "reference.npz") if wd else None,
                           procs=procs, reuse=reuse)
    _index(qset)
    _index(rset)
    cands = shortlist(qset, rset)
    pairs = verify_pairs(qset, rset, cands, procs=procs)
    counts = {"query": qset["counts"], "reference": rset["counts"],
              "shortlist": {"by_cos": sum(1 for e in cands.values() if e["cos"] is not None),
                            "by_dhash": sum(1 for e in cands.values() if e["bits"] is not None),
                            "union": len(cands)},
              "sift_checked": len(pairs), "verified": sum(1 for p in pairs if p["verified"])}
    return {"qset": qset, "rset": rset, "candidates": cands, "pairs": pairs, "counts": counts}


# ------------------------------------------------------------------ step 5: calibration (recorded, never used)
def calibration_rows(q_rows, n=None):
    """The CALIB_N query rows ranked by stable_int(CALIB_SEED_TEXT % exam key), and their copies (one per family
    each, made in memory: never a training list)."""
    n = CALIB_N if n is None else n
    pick = sorted(q_rows, key=lambda r: (C.stable_int(CALIB_SEED_TEXT % r["exam_key"]), r["key"]))[:int(n)]
    copies = []
    for q in pick:
        for fam in CALIB_FAMILIES:
            copies.append({"key": "calib/%s/%s" % (fam, q["key"]), "make": {"source": q["image"], "family": fam},
                           "of": q["key"], "family": fam})
    return pick, copies


def negative_pairs(pick, n=None):
    """n non-copy pairs: each calibration image (in rank order) with the next one, cyclically, from another
    session (different captures)."""
    n = CALIB_NEG if n is None else n
    out = []
    for i, q in enumerate(pick):
        for k in range(1, len(pick)):
            o = pick[(i + k) % len(pick)]
            if o["session"] != q["session"] or not q["session"]:
                if o["key"] != q["key"]:
                    out.append((q, o))
                    break
        if len(out) >= n:
            break
    return out


def calibrate(q_rows, qset, rset, embedder, work_dir=None, procs=1):
    """Step 5: the calibration's recall per family (a copy found when its pair with its own original is
    shortlisted against R plus every copy and verified) and false verifications on non-copy pairs. Recorded only:
    no threshold changes (the constants above are the ones the sweep used)."""
    import numpy as np
    pick, copies = calibration_rows(q_rows)
    wd = Path(work_dir) if work_dir else None
    cset = prepare_set(copies, ("whole",), embedder, desc_path=(wd / "calib.npz") if wd else None, procs=procs)
    sub = {"rows": [r for r in qset["rows"] if r["key"] in {p["key"] for p in pick}], "views": qset["views"]}
    nv = len(qset["views"])
    keep = [qset["index"][r["key"]] for r in sub["rows"]]
    sub["X"] = np.concatenate([qset["X"][i * nv:(i + 1) * nv] for i in keep]) if keep else qset["X"][:0]
    sub["H"] = qset["H"][keep] if keep else qset["H"][:0]
    sub["info"] = qset["info"]
    sub["counts"] = {"rows": len(keep)}
    merged = {"rows": list(rset["rows"]) + copies, "views": ("whole",),
              "X": np.concatenate([rset["X"], cset["X"]]), "H": np.concatenate([rset["H"], cset["H"]]),
              "info": dict(rset["info"], **cset["info"]), "counts": {}}
    _index(sub)
    _index(merged)
    cands = shortlist(sub, merged)
    own = {}
    for c in copies:
        k = (sub["index"][c["of"]], merged["index"][c["key"]])
        if k in cands:
            own[k] = cands[k]
    pairs = verify_pairs(sub, merged, own, procs=procs)
    found = {(p["query"], p["ref"]) for p in pairs if p["verified"]}
    fam = {}
    for f in CALIB_FAMILIES:
        cs = [c for c in copies if c["family"] == f]
        sl = sum(1 for c in cs if (sub["index"][c["of"]], merged["index"][c["key"]]) in cands)
        hit = sum(1 for c in cs if (c["of"], c["key"]) in found)
        fam[f] = {"n": len(cs), "shortlisted": sl, "verified": hit, "recall": round(hit / float(len(cs)), 4)
                  if cs else None}
    negs = negative_pairs(pick)
    nres = _pool_map(_verify_task, [(a, qset["info"][a["key"]]["box"], [(b, qset["info"][b["key"]]["box"])])
                                    for a, b in negs], procs)
    nver = []
    for (a, b), res in zip(negs, nres):
        _k, ia, ib, err = res[0]
        if err is not None:
            raise SweepRefused("a calibration negative cannot be verified (%s, %s): %s" % (a["key"], b["key"], err))
        nver.append({"query": a["key"], "other": b["key"], "inliers": ia, "inliers_flipped": ib,
                     "verified": max(ia, ib) >= INLIERS_MIN})
    return {"images": len(pick), "families": fam, "keys_sha256": list_sha256([p["key"] for p in pick]),
            "seed_text": CALIB_SEED_TEXT, "negatives": {"pairs": len(nver), "false_verified":
                                                        sum(1 for v in nver if v["verified"]),
                                                        "max_inliers": max([max(v["inliers"], v["inliers_flipped"])
                                                                            for v in nver] or [0]),
                                                        "rule": "each calibration image with the next in rank "
                                                                "order from another session"},
            "thresholds_moved": False}


# ------------------------------------------------------------------ montages, records
def montage(q_row, r_row, qbox, rbox, pair, path):
    """A side-by-side JPEG of a verified pair (trimmed images, long side 640 each) with its cosine and inliers."""
    from PIL import Image, ImageDraw
    a, b = trimmed(row_image(q_row), qbox), trimmed(row_image(r_row), rbox)
    tiles = []
    for im in (a, b):
        r = 640.0 / max(im.size)
        tiles.append(im.resize((max(1, int(im.size[0] * r)), max(1, int(im.size[1] * r))), Image.BILINEAR))
    W = tiles[0].size[0] + tiles[1].size[0] + 10
    H = max(t.size[1] for t in tiles) + 40
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(tiles[0], (0, 40))
    out.paste(tiles[1], (tiles[0].size[0] + 10, 40))
    d = ImageDraw.Draw(out)
    d.text((4, 4), "%s  Q %s  |  R %s" % (pair["pair"], pair["query"], pair["ref"][:80]), fill=(0, 0, 0))
    d.text((4, 20), "cos %s  bits %s  inliers %d / flipped %d" % (pair["cos"], pair["bits"], pair["inliers"],
                                                                   pair["inliers_flipped"]), fill=(0, 0, 0))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return F.write_once(path, buf.getvalue())


def read_run():
    d = F._read_json(sweep_dir() / "run.json")
    return d if isinstance(d, dict) and d.get("format") == FORMAT_RUN else None


def read_record():
    """record.json when complete, else None."""
    d = F._read_json(sweep_dir() / "record.json")
    return d if isinstance(d, dict) and d.get("format") == FORMAT_RECORD and d.get("status") == "complete" else None


def complete():
    return read_record() is not None


def trigger(h_keys, dev_keys):
    """The pre-registered branches for H (query keys "<exam>/<key>" with a true copy) on dev's keys."""
    h_test = sorted(k for k in h_keys if k.startswith("test/"))
    h_dev = sorted(k for k in h_keys if k.startswith("dev/"))
    dev_minus = sorted(k for k in dev_keys if "dev/%s" % k not in set(h_dev))
    return {"h_empty": not h_keys, "test_branch": bool(h_test), "dev_branch": bool(h_dev),
            "reanalyse_test_minus_h": bool(h_test), "bclean": bool(h_test),
            "copies_leave_training_sets": bool(h_keys),
            "dev_decides_on": "dev minus H" if h_dev else "dev",
            "dev_minus_h": {"images": len(dev_minus), "keys_sha256": list_sha256(dev_minus)} if h_dev else None,
            "e3_reread_on_dev_minus_h": bool(h_dev), "full_dev_reported_beside": bool(h_dev),
            "h_test": h_test, "h_dev": h_dev}


def removal_keys(record=None):
    """(sorted base B keys of the reference rows in true-copy pairs, the list's sha256) of a complete record."""
    rec = record or read_record()
    if rec is None:
        raise SweepRefused("D0-S's record is not complete: no family training set is written before it (V3F.5 "
                           "D0-S)")
    keys = sorted({c["ref"] for c in rec.get("copies") or []})
    return keys, list_sha256(keys)


def _finish(run_doc, verdicts):
    """record.json from the run and every verified pair's verdict (written once)."""
    pairs = {p["pair"]: p for p in run_doc["pairs"] if p["verified"]}
    copies = [dict(pairs[pid], read_by=v["read_by"]) for pid, v in sorted(verdicts.items()) if v["copy"] == "yes"]
    h = sorted({c["query"] for c in copies})
    dev_keys = run_doc["sets"]["dev_keys"]
    rec = {"format": FORMAT_RECORD, "status": "complete", "run_sha256": C.sha256_file(sweep_dir() / "run.json"),
           "sets": {k: v for k, v in run_doc["sets"].items() if k != "dev_keys"}, "counts": run_doc["counts"],
           "calibration": run_doc["calibration"], "verified_pairs": len(pairs),
           "verdicts": {pid: {"copy": v["copy"], "read_by": v["read_by"], "utc": v["utc"]}
                        for pid, v in sorted(verdicts.items())},
           "copies": [{"pair": c["pair"], "query": c["query"], "ref": c["ref"], "cos": c["cos"], "bits": c["bits"],
                       "inliers": max(c["inliers"], c["inliers_flipped"]), "read_by": c["read_by"]} for c in copies],
           "H": h, "trigger": trigger(h, dev_keys), "decided_by": F.DECIDED_BY, "code": F.code_shas(),
           "created_utc": F._utc()}
    keys = sorted({c["ref"] for c in copies})
    rec["removal"] = {"keys": keys, "sha256": list_sha256(keys), "rows": len(keys)}
    if not h:
        rec["note"] = ("H is empty: the sweep found no crop or letterbox copy of a dev or test image among R; the "
                       "paper states the yield and the calibrated recall")
    F.write_once(sweep_dir() / "record.json", rec)
    log("D0-S record complete: %d verified pair(s), %d true cop(ies), H %d (%s)"
        % (len(pairs), len(copies), len(h), ", ".join(k for k, v in rec["trigger"].items() if v is True) or
           "no branch"))
    return rec


def run(procs=5, embedder=None, testing=False, sets=None, scan_desc=None):
    """The sweep (module docstring), the E-series trees unchanged around it (V3F.3.7): writes run.json (and
    montages) once; record.json too when no pair is verified. sets: {"q": rows, "r": rows} in testing only
    (production reads LOCK v1 and base B)."""
    with F.e_trees_unchanged():
        return _run(procs=procs, embedder=embedder, testing=testing, sets=sets, scan_desc=scan_desc)


def _run(procs=5, embedder=None, testing=False, sets=None, scan_desc=None):
    if (sweep_dir() / "run.json").exists():
        raise SweepRefused("%s exists: the sweep runs once" % (sweep_dir() / "run.json"))
    F.check_frozen()
    ts = require_timestamp()
    t0 = time.time()
    if sets is not None and not testing:
        raise SweepRefused("the sets are LOCK v1's dev and test and base B's R; another set only in testing")
    sets = sets or {}
    q_rows = sets["q"] if "q" in sets else query_rows()
    if not testing and len(q_rows) != QUERY_IMAGES:
        raise SweepRefused("Q holds %d images, the amendment's %d" % (len(q_rows), QUERY_IMAGES))
    if "r" in sets:
        r_rows, rrec = sets["r"], {"injected": True, "reference_rows": len(sets["r"]),
                                   "keys_sha256": list_sha256([r["key"] for r in sets["r"]])}
    else:
        r_rows, rrec = reference_rows(production=not testing)
    if embedder is None:
        from ..funnel import embed as FE
        model, _, pooling = EMBEDDER.rpartition(":")
        embedder = FE.LazyEmbedder(model, pooling)
    if embedder.name != EMBEDDER and not testing:
        raise SweepRefused("the sweep describes images with %s, LOCK v2's detector's embedder" % EMBEDDER)
    wd = sweep_dir() / "desc"
    log("Q %d images (%d views), R %d rows" % (len(q_rows), len(q_rows) * len(VIEWS), len(r_rows)))
    qset = prepare_set(q_rows, VIEWS, embedder, desc_path=wd / "query.npz", procs=procs)
    rinfo, bad = trim_set(r_rows, procs=procs)
    if bad:
        raise SweepRefused("%d reference image(s) cannot be read (e.g. %s)" % (len(bad), bad[:3]))
    sd = scan_desc if scan_desc is not None else (B3.out_dir() / "leak" / "scan_desc.npz")
    reuse, reuse_rec = reuse_descriptors(r_rows, rinfo, embedder, sd, procs=procs, path=None)
    rset = prepare_set(r_rows, ("whole",), embedder, desc_path=wd / "reference.npz", procs=procs, reuse=reuse,
                       info=rinfo)
    res = pipeline(q_rows, r_rows, embedder, procs=procs, qset=qset, rset=rset)
    cal = calibrate(q_rows, res["qset"], res["rset"], embedder, work_dir=wd, procs=procs)
    md = sweep_dir() / "montages"
    shots = {}
    for p in res["pairs"]:
        if p["verified"]:
            qi, rj = res["qset"]["index"][p["query"]], res["rset"]["index"][p["ref"]]
            q, r = res["qset"]["rows"][qi], res["rset"]["rows"][rj]
            shots[p["pair"]] = montage(q, r, res["qset"]["info"][q["key"]]["box"], res["rset"]["info"][r["key"]]["box"],
                                       p, md / ("%s.jpg" % p["pair"]))
    dev_keys = sorted(r["exam_key"] for r in q_rows if r.get("exam") == "dev")
    doc = {"format": FORMAT_RUN, "testing": bool(testing), "timestamp_sha256": ts,
           "sets": {"query": {"images": len(q_rows), "keys_sha256": list_sha256([r["key"] for r in q_rows]),
                              "exams": sorted({r.get("exam") for r in q_rows if r.get("exam")})},
                    "reference": rrec, "dev_keys": dev_keys},
           "parameters": {"trim": {"tol": TRIM_TOL, "std": TRIM_STD}, "views": list(VIEWS),
                          "shortlist": {"top": SHORTLIST_TOP, "cos": SHORTLIST_COS, "bits": SHORTLIST_BITS},
                          "sift": {"features": SIFT_FEATURES, "long_side": SIFT_LONG, "lowe": LOWE_RATIO,
                                   "ransac_px": RANSAC_PX, "iterations": RANSAC_ITERS, "confidence": RANSAC_CONF,
                                   "inliers_min": INLIERS_MIN}, "embedder": embedder.name},
           "reuse": reuse_rec, "counts": res["counts"], "pairs": res["pairs"], "montages": shots,
           "calibration": cal, "decided_by": F.DECIDED_BY, "code": F.code_shas(),
           "seconds": round(time.time() - t0, 1), "created_utc": F._utc()}
    F.write_once(sweep_dir() / "run.json", doc)
    n = res["counts"]["verified"]
    log("sweep run: %d pair(s) shortlisted, %d verified (montages under %s); calibration recall %s"
        % (len(res["pairs"]), n, md, {f: v["recall"] for f, v in cal["families"].items()}))
    if n == 0:
        return _finish(doc, {})
    log("%d verified pair(s) wait for a verdict: inc2.v3fam_sweep verdict --pair ID --copy yes|no --read-by "
        "owner|delegate" % n)
    return doc


def _verdicts():
    out = {}
    d = sweep_dir() / "verdicts"
    if d.is_dir():
        for p in sorted(d.glob("p*.json")):
            v = F._read_json(p)
            if isinstance(v, dict) and v.get("format") == FORMAT_VERDICT:
                out[v["pair"]] = v
    return out


def verdict(pair, copy, read_by):
    """A viewer's verdict on one verified pair, written once; the last one writes record.json."""
    who = F.read_by_value(read_by)
    if copy not in VERDICTS:
        raise SweepRefused("a verdict is %s, not %r" % ("|".join(VERDICTS), copy))
    doc = read_run()
    if doc is None:
        raise SweepRefused("no sweep run.json: run the sweep first")
    pairs = {p["pair"]: p for p in doc["pairs"] if p["verified"]}
    if pair not in pairs:
        raise SweepRefused("%s is not a verified pair of the sweep (%d verified)" % (pair, len(pairs)))
    if read_record() is not None:
        raise SweepRefused("the D0-S record is complete: a verdict changes only as a deviation (V3F.13)")
    v = {"format": FORMAT_VERDICT, "pair": pair, "copy": copy, "read_by": who, "query": pairs[pair]["query"],
         "ref": pairs[pair]["ref"], "montage": (doc.get("montages") or {}).get(pair), "utc": F._utc(),
         "criterion": "copy: the same photograph (scene, plants, capture) up to cropping, resizing, padding, "
                      "flipping, rotation or re-encoding; no: otherwise, two photographs of the same plant or field "
                      "at different moments included"}
    F.write_once(sweep_dir() / "verdicts" / ("%s.json" % pair), v)
    got = _verdicts()
    left = sorted(set(pairs) - set(got))
    if left:
        log("verdict %s: %s (%s); %d pair(s) left" % (pair, copy, who, len(left)))
        return None
    return _finish(doc, got)


def status():
    rec, doc = read_record(), read_run()
    if rec is not None:
        return {"status": "complete", "H": rec["H"], "trigger": rec["trigger"], "removal": rec["removal"]}
    if doc is None:
        return {"status": "not run"}
    pairs = [p["pair"] for p in doc["pairs"] if p["verified"]]
    got = _verdicts()
    return {"status": "verdicts pending", "verified": len(pairs), "left": sorted(set(pairs) - set(got))}


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F D0-S: the residual-copy sweep (crops and letterboxing).")
    ap.add_argument("command", choices=("run", "verdict", "status"))
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--pair", default=None)
    ap.add_argument("--copy", default=None, choices=VERDICTS)
    ap.add_argument("--read-by", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        if a.command == "run":
            run(procs=a.procs)
        elif a.command == "verdict":
            if not a.pair or not a.copy or not a.read_by:
                raise SweepRefused("verdict needs --pair, --copy yes|no and --read-by owner|delegate")
            verdict(a.pair, a.copy, a.read_by)
        else:
            print(json.dumps(status(), indent=1, sort_keys=True))
    except (F.V3famRefused, C2.Inc2Error, ValueError) as e:
        print("[inc2.v3fam_sweep] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
