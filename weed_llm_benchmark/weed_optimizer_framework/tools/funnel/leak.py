"""The copy detector of H6: does a harvested source, the base or an increment
hold an augmented copy of an evaluation image? (contract docs/FUNNEL_AUDIT.md
§3.4, §6 H6, §7, §8.7, DEC-9; runner docs/FUNNEL_AUDIT_RUNNER.md §4.13,
§5.3.3.)

Detector. Two images are a copy when the cosine of their DINOv2 whole-image
descriptors is at least the calibrated threshold, or when the dHash of any of
the eight flips and rotations of the scanned image is within dhash_bits_max
(the never-train radius, 6) bits of the evaluation image's dHash. The
descriptor view and model are embed.py's; the dHash is the project's one
dHash (inc/common.dhash: grey, 9x8 Lanczos thumbnail, horizontal gradients)
of the image as stored, so the "id" variant equals the dHash the never-train
index holds. The eight variants are computed from two thumbnails (the image
and its transpose) and flips of them, which equals transforming the image
first (Lanczos resampling commutes with flips; the transpose fixes the order
of the two resampling passes); the tests check this against the literal
definition.

Calibration, before anything is judged (contract §6 H6 [review]):
  positives  per augmentation family (flip, rot90, crop 0-20 %, brightness
             +-25 %, blur, shear, letterbox to 640, JPEG re-encoding), a
             seeded sample of min(2,000, all) reference-split images (never an
             evaluation image), each augmented once with parameters drawn
             from the family's range (seed funnel/v1/leak/pos/<family>/<key>);
             a positive is the augmented image against its original.
  threshold  the minimum over families of the 5th percentile of the positive
             cosines, so every family keeps recall >= 0.95 on its positives;
             a positive that cannot be described counts as a miss.
  negatives  (1) every pair at 7-10 dHash bits between the provenance-
             disjoint source groups the config names
             (leak.negative_source_pairs); (2) the hardest pairs: for a
             seeded sample of min(2,000, all) images of the first group (seed
             funnel/v1/leak/neg), its nearest image of the second group by
             cosine.
  gate       recall >= recall_min in every family and a false-positive rate
             <= fpr_max on both negative sets, each non-empty (the prereg's
             H6 numbers). Otherwise leak_v1.json is written with ok false and
             no scan, and LeakCalibrationError is raised.

Scans (after a passed calibration): every kept pool image, the base's
harvested images and every increment of the realloop experiment(s), each
against every evaluation image (the top 5 by cosine plus every image within
10 bits under any variant). H6(a): the sources that kept images and have a
near_eval hit in pool_summary.json (the scope, derived, never listed by
hand); a source with any copy, in any scanned set, is quarantined from
recovery as a whole, and the out-of-distribution numbers of every arm that
holds it are void, whichever evaluation split it copies. H6(b): a copy in the
base or an increment is an incident; an experiment's own copy of its base
(a manifest holding reference-split images; increments are harvested only)
is a base, not an increment. H6(c): the config's lab groups, plus every source joined to the
group of the evaluation split it copies (the split's group: leak.exam_labs
in the config, else the group naming the split or its rows' dataset, else
the reference split's group for an exam drawn from the reference dataset;
otherwise the split is reported unmapped).

Evaluation pixels never leave the cluster: leak_v1.json carries evaluation
images as keys only; their descriptors are in leak_eval_desc.npz and every
copy pair in leak_pairs_v1.csv, both cluster-only (runner §6.3).

Detector version 2 (contract §14 A2, run_v2 -> leak_v2.json; the leak verb
runs it when the prereg's amendment requires it). Version 1's calibration
measured a false-positive rate per pair, while the scan flags an image on
its best cosine over every evaluation image and H6(a) quarantined a source
on one flagged image. Version 2 rebuilds version 1's calibration (its
threshold is the floor), scores each negative image the way the scan scores
an image (its maximum cosine over the evaluation images outside its own
capture session and date, the adapter's optional capture_session naming
them), sets the threshold at a per-image false-positive rate <= fpr_max on
every constraining tier, records per-family recall there (a family below the
gate is a known limit), and decides H6(a) and H6(b) by the set rule
(set_verdict: a dHash hit, or more embedding hits than the per-image rate
predicts). It reads the descriptor files leak_v1.json records when they
still hash as recorded and never writes a file leak_v1.json names; once the
amendment is in the prereg, leak_v1.json is never rewritten (run).

Nothing here names a domain.
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import (FUNNEL_DIR, STEP1_DIR, LeakCalibrationError, LeakError, canonical_json, file_record, header,
               read_json, sha256_bytes, strip_volatile, write_csv_atomic, write_json_atomic)
from . import domain as D
from . import embed as E
from ..inc import common as C
from ..near_dup import HOLDOUT_NEAR_DUP_BITS

FAMILIES = ("flip", "rot90", "crop", "brightness", "blur", "shear", "letterbox640", "jpeg")
VARIANTS = ("id", "hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "transverse")
# The contract's ranges (§6 H6; runner §3.2). The config restates them
# (leak.families) with these keys, with "range" for a family's scalar range,
# or with a symmetric bound (SYMMETRIC_KEYS).
FAMILY_DEFAULTS = {
    "flip": {"modes": ["h", "v"]},
    "rot90": {"angles": [90, 180, 270]},
    "crop": {"frac": [0.0, 0.2]},
    "brightness": {"factor": [0.75, 1.25]},
    "blur": {"radius": [1.0, 2.0]},
    "shear": {"degrees": [-10.0, 10.0]},
    "letterbox640": {"size": 640, "fill": [0, 0, 0]},
    "jpeg": {"quality": [50, 90]},
}
RANGE_KEY = {"crop": "frac", "brightness": "factor", "blur": "radius", "shear": "degrees", "jpeg": "quality"}
# A bound b given alone: crop 0..b of each side's extent; brightness factor
# 1 - b .. 1 + b; shear -b .. +b degrees.
SYMMETRIC_KEYS = {"crop": ("max_frac", lambda b: [0.0, b]),
                  "brightness": ("max_delta", lambda b: [1.0 - b, 1.0 + b]),
                  "shear": ("max_deg", lambda b: [-b, b])}
DHASH_BITS_MAX = HOLDOUT_NEAR_DUP_BITS          # the never-train radius
CANDIDATE_BITS = 10
TOP_COS = 5
NEG_BITS = (7, 10)
POS_PER_FAMILY = 2000
NEG_HARD = 2000
LISTED_MAX = 200
RECALL_MIN = 0.95                               # contract §6 H6; the prereg's value is used when given
FPR_MAX = 0.01
INCREMENT_EXPS = ("realloop_v1",)
SEED_PREFIX = "funnel/v1/leak"
EVAL_DESC = "leak_eval_desc.npz"
OUT_NAME = "leak_v1.json"
PAIRS_NAME = "leak_pairs_v1.csv"
PAIRS_HEADER = ("set", "key", "eval_split", "eval_key", "cos", "bits", "variant", "kind")
CHUNK = 512
RULE = ("copy iff cos >= cos_threshold or min over variants of dHash bits <= dhash_bits_max")


def log(msg):
    print("[funnel.leak] %s" % msg, flush=True)


def _raw(obj):
    return getattr(obj, "raw", obj)


# ------------------------------------------------------------------ dHash
def _pack_bits(a):
    d = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in d:
        out = (out << 1) | int(b)
    return out


def dhash_variants(image):
    """{variant: dHash} of the eight flips and rotations of a PIL image, each
    equal to common.dhash of that transform of the image (as stored: no EXIF
    orientation is applied here)."""
    from PIL import Image
    L = image.convert("L")
    LT = L.transpose(Image.Transpose.TRANSPOSE)
    r = np.array(L.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    rt = np.array(LT.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return {"id": _pack_bits(r), "hflip": _pack_bits(r[:, ::-1]), "vflip": _pack_bits(r[::-1, :]),
            "rot180": _pack_bits(r[::-1, ::-1]), "transpose": _pack_bits(rt), "rot90": _pack_bits(rt[::-1, :]),
            "rot270": _pack_bits(rt[:, ::-1]), "transverse": _pack_bits(rt[::-1, ::-1])}


_M1, _M2, _M4, _H01 = (np.uint64(0x5555555555555555), np.uint64(0x3333333333333333),
                       np.uint64(0x0F0F0F0F0F0F0F0F), np.uint64(0x0101010101010101))


def popcount64(x):
    """Set bits of every uint64 in x (int64 array)."""
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return ((x * _H01) >> np.uint64(56)).astype(np.int64)


def _min_variant_bits(Hq, h_target):
    """(bits, variant index): the smallest Hamming distance between any variant
    hash of each query row (Hq [n, 8]) and the target hash(es) (scalar or [n])."""
    Hq = np.asarray(Hq, dtype=np.uint64)
    t = np.asarray(h_target, dtype=np.uint64)
    D = popcount64(Hq ^ (t[:, None] if t.ndim else t))
    return D.min(axis=1), D.argmin(axis=1)


# ---------------------------------------------------------- augmentations
def family_params(domain):
    """{family: params}: the contract's ranges, restated or narrowed by the
    config's leak.families. Every family must be named there; an unknown
    family or key, or a range outside the family's domain, refuses."""
    fams = (_raw(domain).get("leak") or {}).get("families")
    if not isinstance(fams, dict):
        raise LeakError("the domain config has no leak.families")
    unknown = sorted(set(fams) - set(FAMILIES))
    missing = [f for f in FAMILIES if f not in fams]
    if unknown or missing:
        raise LeakError("leak.families must name exactly %s (unknown %s, missing %s)"
                        % (list(FAMILIES), unknown, missing))
    return {f: _merge_params(f, fams[f]) for f in FAMILIES}


def _pair(v, what):
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
        raise LeakError("%s must be [low, high], got %r" % (what, v))
    lo, hi = float(v[0]), float(v[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise LeakError("%s: [%r, %r] is not a range" % (what, v[0], v[1]))
    return [lo, hi]


def _merge_params(family, given):
    if family not in FAMILIES:
        raise LeakError("unknown augmentation family %r (known: %s)" % (family, list(FAMILIES)))
    p = json.loads(json.dumps(FAMILY_DEFAULTS[family]))
    given = dict(given or {})
    if "range" in given:
        if family not in RANGE_KEY or RANGE_KEY[family] in given:
            raise LeakError("leak.families.%s: 'range' is not a key of this family" % family)
        given[RANGE_KEY[family]] = given.pop("range")
    sym = SYMMETRIC_KEYS.get(family)
    if sym and sym[0] in given:
        b = given.pop(sym[0])
        if RANGE_KEY[family] in given:
            raise LeakError("leak.families.%s: give %s or %s, not both" % (family, sym[0], RANGE_KEY[family]))
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not math.isfinite(b) or b < 0:
            raise LeakError("leak.families.%s.%s must be a non-negative number" % (family, sym[0]))
        given[RANGE_KEY[family]] = sym[1](float(b))
    unknown = sorted(set(given) - set(p))
    if unknown:
        accepted = sorted(p) + (["range"] if family in RANGE_KEY else []) + ([sym[0]] if sym else [])
        raise LeakError("leak.families.%s: unknown key(s) %s (accepted: %s)" % (family, unknown, accepted))
    p.update(given)
    what = "leak.families.%s" % family
    if family == "flip":
        if not p["modes"] or not set(p["modes"]) <= {"h", "v"}:
            raise LeakError("%s.modes must be a non-empty subset of ['h', 'v']" % what)
    elif family == "rot90":
        if not p["angles"] or not set(p["angles"]) <= {90, 180, 270}:
            raise LeakError("%s.angles must be a non-empty subset of [90, 180, 270]" % what)
    elif family == "crop":
        lo, hi = p["frac"] = _pair(p["frac"], what + ".frac")
        if lo < 0 or hi >= 0.5:
            raise LeakError("%s.frac must lie in [0, 0.5)" % what)
    elif family == "brightness":
        lo, hi = p["factor"] = _pair(p["factor"], what + ".factor")
        if lo <= 0:
            raise LeakError("%s.factor must be positive" % what)
    elif family == "blur":
        lo, hi = p["radius"] = _pair(p["radius"], what + ".radius")
        if lo < 0:
            raise LeakError("%s.radius must be >= 0" % what)
    elif family == "shear":
        lo, hi = p["degrees"] = _pair(p["degrees"], what + ".degrees")
        if lo <= -45 or hi >= 45:
            raise LeakError("%s.degrees must lie in (-45, 45)" % what)
    elif family == "letterbox640":
        if not isinstance(p["size"], int) or p["size"] < 16:
            raise LeakError("%s.size must be an integer >= 16" % what)
        if (not isinstance(p["fill"], (list, tuple)) or len(p["fill"]) != 3
                or not all(isinstance(c, int) and 0 <= c <= 255 for c in p["fill"])):
            raise LeakError("%s.fill must be [r, g, b] in 0..255" % what)
        p["fill"] = list(p["fill"])
    elif family == "jpeg":
        lo, hi = _pair(p["quality"], what + ".quality")
        if lo != int(lo) or hi != int(hi) or lo < 1 or hi > 100:
            raise LeakError("%s.quality must be integers in 1..100" % what)
        p["quality"] = [int(lo), int(hi)]
    return p


def _r6(x):
    return round(float(x), 6)


def augment(image, family, rng, params=None):
    """(augmented RGB image, the parameters drawn): one re-export
    augmentation of the family, its parameters drawn from rng within the
    family's range (params, else the contract's)."""
    from PIL import Image, ImageEnhance, ImageFilter
    p = _merge_params(family, params)
    im = image if image.mode == "RGB" else image.convert("RGB")
    W, H = im.size
    T = Image.Transpose
    if family == "flip":
        mode = str(p["modes"][int(rng.integers(len(p["modes"])))])
        return im.transpose(T.FLIP_LEFT_RIGHT if mode == "h" else T.FLIP_TOP_BOTTOM), {"mode": mode}
    if family == "rot90":
        angle = int(p["angles"][int(rng.integers(len(p["angles"])))])
        op = {90: T.ROTATE_90, 180: T.ROTATE_180, 270: T.ROTATE_270}[angle]
        return im.transpose(op), {"angle": angle}
    if family == "crop":
        lo, hi = p["frac"]
        fx, fy = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
        ux, uy = float(rng.random()), float(rng.random())
        cw, ch = int(round(W * fx)), int(round(H * fy))
        cw, ch = min(cw, W - 1), min(ch, H - 1)
        x0, y0 = int(round(cw * ux)), int(round(ch * uy))
        box = (x0, y0, W - (cw - x0), H - (ch - y0))
        return im.crop(box), {"fx": _r6(fx), "fy": _r6(fy), "box": list(box)}
    if family == "brightness":
        f = float(rng.uniform(*p["factor"]))
        return ImageEnhance.Brightness(im).enhance(f), {"factor": _r6(f)}
    if family == "blur":
        r = float(rng.uniform(*p["radius"]))
        return im.filter(ImageFilter.GaussianBlur(radius=r)), {"radius": _r6(r)}
    if family == "shear":
        deg = float(rng.uniform(*p["degrees"]))
        axis = "x" if rng.random() < 0.5 else "y"
        t = math.tan(math.radians(deg))
        coeffs = (1, t, -t * H / 2.0, 0, 1, 0) if axis == "x" else (1, 0, 0, t, 1, -t * W / 2.0)
        out = im.transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        return out, {"degrees": _r6(deg), "axis": axis}
    if family == "letterbox640":
        size, fill = int(p["size"]), tuple(p["fill"])
        s = size / float(max(W, H))
        nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        canvas = Image.new("RGB", (size, size), fill)
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
        return canvas, {"size": size, "scaled": [nw, nh]}
    if family == "jpeg":
        q = int(rng.integers(p["quality"][0], p["quality"][1] + 1))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        with Image.open(buf) as j:
            out = j.convert("RGB")
        return out, {"quality": q}
    raise LeakError("unknown augmentation family %r" % family)


# ------------------------------------------------- image preparation (workers)
def prepare_hashed(row):
    """(EXIF-transposed RGB view, the 8 variant dHashes of the stored image, None)."""
    from PIL import Image, ImageOps
    with Image.open(row["image"]) as im0:
        im0.load()
        hv = dhash_variants(im0)
        view = ImageOps.exif_transpose(im0).convert("RGB")
    return view, [hv[v] for v in VARIANTS], None


def prepare_augmented(row):
    """(augmented image, its 8 variant dHashes, {"family", "params"}) for a
    positive: row["spec"] = {"family", "seed_text", "params"}."""
    from PIL import Image, ImageOps
    spec = row["spec"]
    with Image.open(row["image"]) as im0:
        view = ImageOps.exif_transpose(im0).convert("RGB")
    img, used = augment(view, spec["family"], np.random.default_rng(C.stable_int(spec["seed_text"])),
                        spec.get("params"))
    hv = dhash_variants(img)
    return img, [hv[v] for v in VARIANTS], {"family": spec["family"], "params": used}


class _Store:
    """Descriptor files of one run (or, with funnel_dir None, in memory)."""

    def __init__(self, embedder, funnel_dir=None, procs=1, batch=32, force=False):
        self.embedder = embedder
        self.dir = None if funnel_dir is None else Path(funnel_dir)
        self.procs, self.batch, self.force = procs, batch, force
        self.sets = {}
        self.files = {}

    def path(self, name):
        if self.dir is None:
            return None
        return self.dir / (EVAL_DESC if name == "eval" else "emb_dinov2_images_%s.npz" % name)

    def get(self, name, rows, prepare=prepare_hashed):
        if name in self.sets:
            return self.sets[name]
        res = E.embed_images(rows, self.path(name), self.embedder, prepare=prepare, n_hashes=len(VARIANTS),
                             procs=self.procs, batch=self.batch, force=self.force)
        _finish_set(res)
        self.sets[name] = res
        if res.get("path"):
            self.files[name] = {"path": res["path"], "sha256": res["sha256"]}
        return res


def _finish_set(res):
    """Normalised descriptors, the usable mask (finite descriptor and hashes)
    and the key index of one descriptor set, in place."""
    res["Xn"] = _normalise(res["X"])
    res["ok"] = np.isfinite(res["Xn"]).all(axis=1) & np.asarray(res["hash_ok"], dtype=bool)
    res["index"] = {k: i for i, k in enumerate(res["keys"])}
    return res


def _normalise(X):
    X = np.asarray(X, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------ calibration
def threshold(positive_cos, recall_min=RECALL_MIN):
    """The minimum over families of each family's lower (1 - recall_min)
    order statistic of its positive cosines: at least recall_min of every
    family's positives have cosine >= it. A cosine that is not finite (a
    positive that could not be described) sorts below every other."""
    vals = []
    for fam in sorted(positive_cos):
        c = np.asarray(positive_cos[fam], dtype=np.float64)
        if len(c) == 0:
            raise LeakCalibrationError("family %s has no positives" % fam)
        c = np.sort(np.where(np.isfinite(c), c, -np.inf))
        k = int(math.floor((1.0 - float(recall_min)) * len(c) + 1e-9))
        vals.append(c[min(k, len(c) - 1)])
    if not vals:
        raise LeakCalibrationError("no positives")
    return float(min(vals))


def _interval(k, n):
    from . import estimate
    lo, hi = estimate.binom_interval(int(k), int(n))
    return _r6(lo), _r6(hi)


def reference_rows(domain):
    """The reference split's manifest rows, sorted by key, and its file record."""
    ref = _raw(domain)["sources"]["reference"]
    path = C.manifest_path(ref)
    if not Path(path).is_file():
        raise LeakError("the reference manifest %s is missing" % path)
    return sorted(C.read_manifest(path), key=lambda r: r["key"]), file_record(path)


def _negative_pairs_cfg(domain):
    pairs = (_raw(domain).get("leak") or {}).get("negative_source_pairs")
    if (not isinstance(pairs, list) or not pairs
            or not all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) and x for x in p)
                       and p[0] != p[1] for p in pairs)):
        raise LeakError("leak.negative_source_pairs must be a non-empty list of [source, source] pairs")
    return [tuple(p) for p in pairs]


def _group(name, domain, store, ref_rows, pool_rows):
    """(Xn, H, ok, keys) of a negative group: the reference split, or a pool source."""
    ref = _raw(domain)["sources"]["reference"]
    if name == ref:
        res = store.get("reference", ref_rows)
        idx = np.arange(len(ref_rows))
    else:
        res = store.get("pool", pool_rows)
        idx = np.array([res["index"][r["key"]] for r in pool_rows if r.get("source") == name], dtype=np.int64)
    if len(idx) == 0:
        raise LeakCalibrationError("negative group %s has no images" % name)
    return res["Xn"][idx], np.asarray(res["H"], dtype=np.uint64)[idx], res["ok"][idx], [res["keys"][i] for i in idx]


def _rule(cos, bits, theta, bits_max):
    return (np.asarray(cos) >= theta) | (np.asarray(bits) <= bits_max)


def calibrate(adapter, domain, embedder, seed_prefix=SEED_PREFIX, store=None, recall_min=RECALL_MIN,
              fpr_max=FPR_MAX, ref_rows=None, pool_rows=None, procs=1, positive_scores=None):
    """The detector's calibration (module docstring). Returns the calibration
    record; its "pairs" list holds every negative pair (for the pairs file,
    never for leak_v1.json). positive_scores, when a dict, receives each
    family's positive cosines and dHash bits ({"cos": {family: array},
    "bits": {family: array}}), which detector version 2 re-scores."""
    store = store or _Store(embedder, None, procs=procs)
    fams = family_params(domain)
    if ref_rows is None:
        ref_rows, _rec = reference_rows(domain)
    if pool_rows is None:
        pool_rows = sorted(adapter.pool_rows(), key=lambda r: r["key"])
    if not ref_rows:
        raise LeakCalibrationError("the reference split has no images")
    ref = store.get("reference", ref_rows)
    n_pos = min(POS_PER_FAMILY, len(ref_rows))
    seeds = {}
    pos_cos, pos_bits, pos_failed = {}, {}, {}
    for fam in FAMILIES:
        sp = "%s/pos/%s" % (seed_prefix, fam)
        seeds["positives/%s" % fam] = sp
        pick = np.sort(np.random.default_rng(C.stable_int(sp)).choice(len(ref_rows), n_pos, replace=False))
        rows = [{"key": "%s|%s" % (fam, ref_rows[i]["key"]), "image": ref_rows[i]["image"],
                 "sha256": ref_rows[i].get("sha256"),
                 "spec": {"family": fam, "seed_text": "%s/%s" % (sp, ref_rows[i]["key"]), "params": fams[fam]}}
                for i in pick]
        aug = store.get("pos_%s" % fam, rows, prepare=prepare_augmented)
        cos = np.einsum("ij,ij->i", aug["Xn"], ref["Xn"][pick]).astype(np.float64)
        ok = aug["ok"] & ref["ok"][pick]
        cos = np.where(ok, cos, -np.inf)
        bits, _v = _min_variant_bits(np.asarray(aug["H"], dtype=np.uint64),
                                     np.asarray(ref["H"], dtype=np.uint64)[pick, 0])
        bits = np.where(ok, bits, 65)
        pos_cos[fam], pos_bits[fam], pos_failed[fam] = cos, bits, int((~ok).sum())
    if isinstance(positive_scores, dict):
        positive_scores.update(cos=dict(pos_cos), bits=dict(pos_bits), failed=dict(pos_failed))
    theta = threshold(pos_cos, recall_min)
    finite_theta = math.isfinite(theta)
    why = []
    if not finite_theta:
        why.append("a family's threshold quantile falls on positives that could not be described")
    positives = {}
    for fam in FAMILIES:
        hits = _rule(pos_cos[fam], pos_bits[fam], theta, DHASH_BITS_MAX) & np.isfinite(pos_cos[fam])
        n, k = len(hits), int(hits.sum())
        lo, _hi = _interval(k, n)
        fin = pos_cos[fam][np.isfinite(pos_cos[fam])]
        positives[fam] = {"n": n, "hits": k, "recall": _r6(k / float(n)), "lb": lo, "params_seed": seeds["positives/%s" % fam],
                          "failed": pos_failed[fam], "params": fams[fam],
                          "cos_q05": _r6(np.quantile(fin, 0.05)) if len(fin) else None,
                          "dhash_hits": int((pos_bits[fam] <= DHASH_BITS_MAX).sum())}
        if k < recall_min * n:
            why.append("family %s: recall %.4f < %.2f" % (fam, k / float(n), recall_min))

    # negatives
    pairs_out = []
    tallies = {"pairs_7_10": [0, 0, 0], "hard": [0, 0, 0]}          # n, false hits, unscored
    by_pair = {}
    for pi, (ga, gb) in enumerate(_negative_pairs_cfg(domain)):
        Xa, Ha, oka, ka = _group(ga, domain, store, ref_rows, pool_rows)
        Xb, Hb, okb, kb = _group(gb, domain, store, ref_rows, pool_rows)
        tag = "calibration:%s|%s" % (ga, gb)
        rec = {"pairs_7_10": [0, 0, 0], "hard": [0, 0, 0]}
        # (1) every pair at 7-10 bits between the groups (id dHash)
        for s in range(0, len(Ha), CHUNK):
            D = popcount64(Ha[s:s + CHUNK, 0][:, None] ^ Hb[:, 0][None, :])
            ii, jj = np.nonzero((D >= NEG_BITS[0]) & (D <= NEG_BITS[1]))
            for i, j in zip(ii + s, jj):
                if not (oka[i] and okb[j]):
                    rec["pairs_7_10"][2] += 1
                    continue
                c = float(Xa[i] @ Xb[j])
                b, v = _min_variant_bits(Hb[j][None, :], Ha[i, 0])
                hit = bool(_rule(c, b[0], theta, DHASH_BITS_MAX)) if finite_theta else True
                rec["pairs_7_10"][0] += 1
                rec["pairs_7_10"][1] += int(hit)
                pairs_out.append({"set": tag, "key": kb[j], "eval_split": ga, "eval_key": ka[i], "cos": _r6(c),
                                  "bits": int(b[0]), "variant": VARIANTS[int(v[0])], "kind": "negative_7_10"})
        # (2) the hardest pairs: nearest second-group image of a sample of the first
        sp = "%s/neg" % seed_prefix if pi == 0 else "%s/neg/%d" % (seed_prefix, pi)
        seeds["negatives/hard/%s|%s" % (ga, gb)] = sp
        usable_a = np.flatnonzero(oka)
        usable_b = np.flatnonzero(okb)
        rec["hard"][2] += int(len(Ha) - len(usable_a))
        if len(usable_b) and len(usable_a):
            m = min(NEG_HARD, len(usable_a))
            pick = np.sort(np.random.default_rng(C.stable_int(sp)).choice(len(usable_a), m, replace=False))
            A = usable_a[pick]
            S = Xa[A] @ Xb[usable_b].T
            nn = usable_b[np.argmax(S, axis=1)]
            for i, j in zip(A, nn):
                c = float(Xa[i] @ Xb[j])
                b, v = _min_variant_bits(Hb[j][None, :], Ha[i, 0])
                hit = bool(_rule(c, b[0], theta, DHASH_BITS_MAX)) if finite_theta else True
                rec["hard"][0] += 1
                rec["hard"][1] += int(hit)
                pairs_out.append({"set": tag, "key": kb[j], "eval_split": ga, "eval_key": ka[i], "cos": _r6(c),
                                  "bits": int(b[0]), "variant": VARIANTS[int(v[0])], "kind": "negative_hard"})
        for kind in tallies:
            for t in range(3):
                tallies[kind][t] += rec[kind][t]
        by_pair["%s|%s" % (ga, gb)] = {kind: {"n": rec[kind][0], "false_hits": rec[kind][1], "unscored": rec[kind][2]}
                                       for kind in rec}
    negatives = {}
    for kind, (n, fh, uns) in tallies.items():
        if n == 0:
            negatives[kind] = {"n": 0, "false_hits": 0, "fpr": None, "ub": None, "unscored": uns}
            why.append("negative set %s is empty" % kind)
            continue
        _lo, hi = _interval(fh, n)
        negatives[kind] = {"n": n, "false_hits": fh, "fpr": _r6(fh / float(n)), "ub": hi, "unscored": uns}
        if fh > fpr_max * n:
            why.append("negative set %s: false-positive rate %.4f > %.2f" % (kind, fh / float(n), fpr_max))
    negatives["by_pair"] = by_pair
    pairs_out.sort(key=lambda r: (r["kind"], r["set"], r["key"], r["eval_key"]))
    return {"ok": not why, "why": why, "cos_threshold": _r6(theta) if finite_theta else None,
            "dhash_bits_max": DHASH_BITS_MAX, "positives": positives, "negatives": negatives,
            "recall_min": float(recall_min), "fpr_max": float(fpr_max), "seeds": seeds,
            "reference_images": len(ref_rows), "pairs": pairs_out}


def _detector_params(calibration):
    cal = calibration.get("calibration", calibration) if isinstance(calibration, dict) else None
    if not isinstance(cal, dict) or cal.get("ok") is not True or cal.get("cos_threshold") is None:
        raise LeakCalibrationError("the copy detector's calibration did not pass; nothing may be judged by it")
    return float(cal["cos_threshold"]), int(cal.get("dhash_bits_max", DHASH_BITS_MAX))


# ------------------------------------------------------------ evaluation index
class EvalIndex:
    """Descriptors and dHashes of every evaluation image (keys only; the file
    is cluster-only). embedder, when known, describes new images the same way."""

    def __init__(self, res, splits, eval_keys, embedder=None):
        self.keys = list(res["keys"])
        self.split = list(splits)
        self.eval_key = list(eval_keys)
        self.Xn = res["Xn"]
        self.H = np.asarray(res["H"], dtype=np.uint64)
        self.record = {"path": res.get("path"), "sha256": res.get("sha256")}
        self.embedder = embedder
        self.n = len(self.keys)


def eval_index(adapter, embedder, cache_path, procs=1, batch=32, force=False, store=None):
    """The EvalIndex over adapter.eval_rows() (every evaluation split), cached
    in cache_path. An evaluation image that cannot be described refuses: an
    image the detector cannot compare against is one it cannot clear.

    Without a store (a consumer such as recover, not leak.run), an existing
    cache made by another embedder or from other evaluation rows refuses
    instead of being overwritten: it is the file leak_v1.json records, and the
    calibrated threshold belongs to its descriptors."""
    er = adapter.eval_rows()
    if not isinstance(er, dict) or not er:
        raise LeakError("the adapter returned no evaluation rows")
    items, splits, ekeys = [], [], []
    for split in sorted(er):
        for r in sorted(er[split], key=lambda r: r["key"]):
            items.append({"key": "%s|%s" % (split, r["key"]), "image": r["image"], "sha256": r.get("sha256")})
            splits.append(split)
            ekeys.append(r["key"])
    if store is None:
        if (cache_path is not None and Path(cache_path).exists() and not force
                and not E.image_file_current(cache_path, E.image_ident(items, embedder, prepare_hashed,
                                                                         len(VARIANTS)))):
            raise LeakError("%s was made by another embedder or from other evaluation rows than %s gives now; "
                            "rerun leak (with --force) rather than overwrite it" % (cache_path, embedder.name))
        store = _Store(embedder, None, procs=procs, batch=batch, force=force)
        store.path = lambda name: None if cache_path is None else Path(cache_path)
    res = store.get("eval", items)
    bad = [k for k, ok in zip(res["keys"], res["ok"]) if not ok]
    if bad:
        raise LeakError("%d evaluation image(s) cannot be described (e.g. %s); the detector cannot clear "
                        "anything against them" % (len(bad), bad[:3]))
    return EvalIndex(res, splits, ekeys, embedder=embedder)


def _scan(Xn, H, ok, index, theta, bits_max, chunk=CHUNK):
    """Copies of evaluation images among the query rows: [(row, eval j, cos,
    bits, variant index)], candidates = top TOP_COS by cosine plus every eval
    image within CANDIDATE_BITS under any variant."""
    out = []
    H = np.asarray(H, dtype=np.uint64)
    He = index.H[:, 0]
    rows_ok = np.flatnonzero(ok)
    top = min(TOP_COS, index.n)
    for s in range(0, len(rows_ok), chunk):
        rows = rows_ok[s:s + chunk]
        S = Xn[rows] @ index.Xn.T
        best = np.full(S.shape, 65, dtype=np.int64)
        bestv = np.zeros(S.shape, dtype=np.int64)
        for v in range(H.shape[1]):
            D = popcount64(H[rows, v][:, None] ^ He[None, :])
            better = D < best
            best = np.where(better, D, best)
            bestv = np.where(better, v, bestv)
        cand = best <= CANDIDATE_BITS
        if top:
            tk = np.argpartition(-S, top - 1, axis=1)[:, :top]
            np.put_along_axis(cand, tk, True, axis=1)
        copy = cand & ((S >= theta) | (best <= bits_max))
        for r, j in zip(*np.nonzero(copy)):
            out.append((int(rows[r]), int(j), float(S[r, j]), int(best[r, j]), int(bestv[r, j])))
    return out


def _copy_entry(row, index, j, c, b, v):
    return {"key": row["key"], "image": str(row["image"]), "eval_split": index.split[j],
            "eval_key": index.eval_key[j], "cos": _r6(c), "bits": int(b), "variant": VARIANTS[v]}


def _check_index(calibration, eval_index):
    """With a whole leak_v1.json, the index must be the one it was calibrated
    with: the same embedder (its threshold is a cosine of that model's
    descriptors) and, when both are on disk, the same evaluation descriptor
    file. A bare calibration record carries neither and is not checked."""
    if not (isinstance(calibration, dict) and "calibration" in calibration):
        return
    want_emb = ((calibration.get("detector") or {}).get("descriptor") or {}).get("embedder")
    emb = getattr(eval_index, "embedder", None)
    if want_emb and emb is not None and emb.name != want_emb:
        raise LeakError("the evaluation index describes images with %s, but the detector was calibrated with %s"
                        % (emb.name, want_emb))
    want_sha = (calibration.get("eval_descriptors") or {}).get("sha256")
    got_sha = (getattr(eval_index, "record", None) or {}).get("sha256")
    if want_sha and got_sha and want_sha != got_sha:
        raise LeakError("the evaluation descriptors (sha256 %s) are not the ones leak_v1.json records (%s); "
                        "rerun leak" % (got_sha[:12], want_sha[:12]))


def detect(images, eval_index, calibration):
    """Every copy of an evaluation image among `images` ([{"key", "path"}],
    optionally with precomputed "desc" and "hashes"), under a passed
    calibration (leak_v1.json, or its calibration record). An image that
    cannot be described refuses (fail closed), and so does an index made by
    another embedder than the calibration's (_check_index)."""
    theta, bits_max = _detector_params(calibration)
    _check_index(calibration, eval_index)
    images = list(images)
    if not images:
        return []
    need = [im for im in images if im.get("desc") is None or im.get("hashes") is None]
    desc = {}
    if need:
        if eval_index.embedder is None:
            raise LeakError("detect needs descriptors or the index's embedder for %d image(s)" % len(need))
        rows = [{"key": im["key"], "image": str(im.get("path") or im.get("image"))} for im in need]
        res = E.embed_images(rows, None, eval_index.embedder, prepare=prepare_hashed, n_hashes=len(VARIANTS))
        Xn = _normalise(res["X"])
        for i, im in enumerate(need):
            if not (np.isfinite(Xn[i]).all() and res["hash_ok"][i]):
                raise LeakError("cannot describe %s (%s); an image the detector cannot compare is one it cannot "
                                "clear" % (im["key"], rows[i]["image"]))
            desc[im["key"]] = (Xn[i], np.asarray(res["H"][i], dtype=np.uint64))
    Xq = np.zeros((len(images), eval_index.Xn.shape[1]), dtype=np.float32)
    Hq = np.zeros((len(images), len(VARIANTS)), dtype=np.uint64)
    for i, im in enumerate(images):
        if im["key"] in desc:
            Xq[i], Hq[i] = desc[im["key"]]
        else:
            d = np.asarray(im["desc"], dtype=np.float32).reshape(-1)
            hs = list(im["hashes"])
            if d.shape != (Xq.shape[1],) or not np.isfinite(d).all() or not np.linalg.norm(d) > 0 \
                    or len(hs) != len(VARIANTS):
                raise LeakError("%s: a given descriptor or hash list is not usable (dim %s of %d, %d of %d hashes); "
                                "an image the detector cannot compare is one it cannot clear"
                                % (im["key"], d.shape, Xq.shape[1], len(hs), len(VARIANTS)))
            Xq[i] = _normalise(d[None, :])[0]
            Hq[i] = np.asarray(hs, dtype=np.uint64)
    copies = _scan(Xq, Hq, np.ones(len(images), dtype=bool), eval_index, theta, bits_max)
    rows = [{"key": im["key"], "image": str(im.get("path") or im.get("image"))} for im in images]
    out = [_copy_entry(rows[r], eval_index, j, c, b, v) for r, j, c, b, v in copies]
    return sorted(out, key=lambda e: (e["key"], e["eval_split"], e["eval_key"]))


# ------------------------------------------------------------ H6 pieces
def h6a_scope(pool_summary):
    """The sources that kept images and have at least one near_eval hit
    (pool_summary.json per_slug: kept > 0, near_eval_by_split non-empty)."""
    per = (pool_summary or {}).get("per_slug")
    if not isinstance(per, dict):
        raise LeakError("pool_summary.json has no per_slug table")
    return sorted(s for s, v in per.items()
                  if int(v.get("kept", 0) or 0) > 0
                  and sum(int(x) for x in (v.get("near_eval_by_split") or {}).values()) > 0)


def exam_labs(domain, eval_rows, ref_rows):
    """{split: lab group | None} and {split: basis}: leak.exam_labs in the
    config; else a group listing the split name, a row source, or a row
    source's dataset prefix; else the reference split's group when the split
    comes from the reference split's dataset. Ambiguous or unknown: None."""
    raw = _raw(domain)
    groups = raw["sources"].get("lab_groups") or {}
    cfg = (raw.get("leak") or {}).get("exam_labs") or {}
    ref = raw["sources"]["reference"]
    ref_group = [g for g, mem in groups.items() if ref in mem]
    ref_prefixes = {str(r.get("source") or "").split("/")[0] for r in ref_rows} - {""}
    labs, basis = {}, {}
    for split in sorted(eval_rows):
        if split in cfg:
            if cfg[split] not in groups:
                raise LeakError("leak.exam_labs.%s names %r, not a lab group" % (split, cfg[split]))
            labs[split], basis[split] = cfg[split], "config"
            continue
        srcs = {str(r.get("source") or "") for r in eval_rows[split]} - {""}
        prefixes = {s.split("/")[0] for s in srcs}
        cands = {g for g, mem in groups.items() if split in mem or srcs & set(mem) or prefixes & set(mem)}
        how = "source"
        if not cands and prefixes & ref_prefixes and len(ref_group) == 1:
            cands, how = set(ref_group), "reference_dataset"
        labs[split] = sorted(cands)[0] if len(cands) == 1 else None
        basis[split] = how if len(cands) == 1 else ("ambiguous" if cands else "unknown")
    return labs, basis


def _rows_digest(rows):
    body = [[r.get("key"), str(r.get("image")), r.get("sha256") or ""] for r in rows]
    return {"n": len(rows), "sha256": sha256_bytes(canonical_json(body).encode("utf-8"))}


def _identity(doc):
    """What makes two runs the same run: inputs, row sets, parameters, code,
    and the prereg by its core (runner §3.3: an amendment, such as the sample
    lock, leaves earlier artifacts current), the contract and the config by
    their sha256."""
    keep = ("inputs", "row_sets", "params", "code", "testing")
    out = {k: doc.get(k) for k in keep}
    out["prereg_core_sha256"] = (doc.get("prereg") or {}).get("core_sha256")
    out["contract_sha256"] = (doc.get("contract") or {}).get("sha256")
    out["domain_config_sha256"] = (doc.get("domain_config") or {}).get("sha256")
    return strip_volatile(out)


class _Inputs(object):
    """What both detector versions read: the rows of every scanned set, the
    H6 gates, the embedder the config names and the row-set digests."""


def _scan_inputs(prereg, domain, adapter, pool_summary_path=None, increment_exps=INCREMENT_EXPS):
    """An _Inputs of the scan (module docstring: the pool, the base's
    harvested images, every increment and any earlier base of the
    experiments, the evaluation rows, the reference rows and the H6(a)
    scope)."""
    si = _Inputs()
    raw = _raw(domain)
    si.ref_name = ref_name = raw["sources"]["reference"]
    si.fams = family_params(domain)
    si.neg_pairs = _negative_pairs_cfg(domain)
    h6 = ((_raw(prereg).get("hypotheses") or {}).get("H6") or {}).get("calibration") or {}
    if "recall_min_per_family" not in h6 or "fpr_max" not in h6:
        raise LeakError("the prereg's H6 calibration has no recall_min_per_family / fpr_max")
    si.recall_min, si.fpr_max = float(h6["recall_min_per_family"]), float(h6["fpr_max"])
    si.model, si.pooling = E.features_config(domain)
    si.want_name = E.embedder_name(si.model, si.pooling)
    si.ps_path = Path(pool_summary_path) if pool_summary_path else STEP1_DIR / "pool_summary.json"
    si.scope = h6a_scope(read_json(si.ps_path))
    ref_rows, si.ref_rec = reference_rows(domain)
    si.ref_rows = ref_rows
    ref_ids = ({("key", r["key"]) for r in ref_rows} | {("image", str(r["image"])) for r in ref_rows}
               | {("sha256", r.get("sha256")) for r in ref_rows if r.get("sha256")})

    def is_reference(r):
        """A reference-split image (by source, key, path or content)."""
        return (r.get("source") == ref_name or ("key", r.get("key")) in ref_ids
                or ("image", str(r.get("image"))) in ref_ids or ("sha256", r.get("sha256")) in ref_ids)

    def harvested(rows):
        """The rows that are not reference-split images."""
        return sorted((r for r in rows if not is_reference(r)), key=lambda r: r["key"])

    def by_key(rows):
        return sorted(rows, key=lambda r: r["key"])
    si.pool_rows = pool_rows = sorted(adapter.pool_rows(), key=lambda r: r["key"])
    base_raw = adapter.base_rows()
    si.base_rows = base_rows = harvested(base_raw)
    base_raw_digest = _rows_digest(by_key(base_raw))["sha256"]
    # An experiment's manifests directory also holds its copy of the base. An
    # increment is drawn from harvested images only, so a manifest holding a
    # reference-split image is a base: skipped when it is the current base row
    # for row, else scanned as a base of its own (H6(b) covers the base the
    # experiment trained on).
    increments, extra_bases, same_as_base = {}, {}, []
    for exp in increment_exps:
        for step, rows in sorted(adapter.increment_rows(exp).items()):
            name = "%s:%s" % (exp, step)
            if any(is_reference(r) for r in rows):
                if _rows_digest(by_key(rows))["sha256"] == base_raw_digest:
                    same_as_base.append(name)
                else:
                    extra_bases[name] = harvested(rows)
                continue
            increments[name] = harvested(rows)
    si.increments, si.extra_bases, si.same_as_base = increments, extra_bases, same_as_base
    er = adapter.eval_rows()
    if not isinstance(er, dict) or not er:
        raise LeakError("the adapter returned no evaluation rows")
    si.eval_rows = er
    eval_digest = _rows_digest([dict(r, key="%s|%s" % (s, r["key"])) for s in sorted(er)
                                for r in sorted(er[s], key=lambda r: r["key"])])
    row_sets = {"reference": _rows_digest(ref_rows), "pool": _rows_digest(pool_rows),
                "base": _rows_digest(base_rows), "eval": eval_digest}
    row_sets.update({"increment:%s" % k: _rows_digest(v) for k, v in increments.items()})
    row_sets.update({"base:%s" % k: _rows_digest(v) for k, v in extra_bases.items()})
    row_sets["same_as_base"] = sorted(same_as_base)
    si.row_sets = row_sets
    si.params = {"families": si.fams, "negative_source_pairs": [list(p) for p in si.neg_pairs],
                 "recall_min": si.recall_min, "fpr_max": si.fpr_max, "embedder": si.want_name, "view": E.VIEW,
                 "dhash_bits_max": DHASH_BITS_MAX, "candidate_bits": CANDIDATE_BITS, "top_cos": TOP_COS,
                 "neg_bits": list(NEG_BITS), "pos_per_family": POS_PER_FAMILY, "neg_hard": NEG_HARD,
                 "seed_prefix": SEED_PREFIX, "increment_exps": list(increment_exps)}
    si.inputs = {"pool_summary": file_record(si.ps_path), "reference_manifest": si.ref_rec}
    return si


def _locator(pool_rows):
    """locate(row) -> the row's index among pool_rows (by sha256, else by
    path), or None: base and increment rows are pool images; any that are
    not are described apart."""
    by_sha = {r.get("sha256"): i for i, r in enumerate(pool_rows) if r.get("sha256")}
    by_img = {str(r["image"]): i for i, r in enumerate(pool_rows)}

    def locate(r):
        i = by_sha.get(r.get("sha256")) if r.get("sha256") else None
        return i if i is not None else by_img.get(str(r["image"]))
    return locate


def _extra_rows(si, locate):
    """{key: row} of the base, earlier-base and increment rows that are not pool images."""
    extra_rows = {}
    for rows in ([si.base_rows] + [si.extra_bases[k] for k in sorted(si.extra_bases)]
                 + [si.increments[k] for k in sorted(si.increments)]):
        for r in rows:
            if locate(r) is None:
                extra_rows.setdefault(r["key"], r)
    return extra_rows


def _refuse_superseded_rewrite(prereg, out_path):
    """An amendment that requires a later copy detector keeps leak_v1.json as
    the first reading (contract §14 A2): never rewritten once it exists."""
    need = D.leak_detector_version(prereg)
    if need > 1:
        a = D.leak_detector_amendment(prereg) or {}
        raise LeakError("%s is the first reading that amendment %s supersedes (copy detector version %d); it is "
                        "kept unchanged, never rewritten: the leak verb writes %s" % (
                            out_path, a.get("id") or "?", need, D.LEAK_FILES.get(need, ("leak_v%d.json" % need,))[0]))


def run(prereg, domain, funnel_dir, adapter, embedder=None, procs=5, batch=32, force=False, testing=False,
        pool_summary_path=None, increment_exps=INCREMENT_EXPS):
    """leak_v1.json, leak_pairs_v1.csv and the descriptor files (module
    docstring). A rerun on the same inputs, parameters and code is a no-op
    (a failed calibration refuses again); other inputs refuse unless force.
    Once an amendment requires a later detector version, an existing
    leak_v1.json is never rewritten (_refuse_superseded_rewrite)."""
    t0 = time.time()
    funnel_dir = Path(funnel_dir)
    raw = _raw(domain)
    si = _scan_inputs(prereg, domain, adapter, pool_summary_path, increment_exps)
    ref_rows, pool_rows, base_rows = si.ref_rows, si.pool_rows, si.base_rows
    increments, extra_bases, same_as_base = si.increments, si.extra_bases, si.same_as_base
    scope, er, model, pooling = si.scope, si.eval_rows, si.model, si.pooling
    modules = (sys.modules[__name__], E, adapter)
    doc = header("leak", domain, prereg, si.inputs, modules=modules, testing=testing)
    doc.update({"row_sets": si.row_sets, "params": si.params})
    out_path = funnel_dir / OUT_NAME
    if out_path.exists():
        old = read_json(out_path)
        if _identity(old) == _identity(doc):
            if not (old.get("calibration") or {}).get("ok"):
                raise LeakCalibrationError("%s: the calibration failed (%s); nothing was scanned"
                                           % (out_path, "; ".join(old["calibration"].get("why", []))))
            log("%s is current; nothing to do" % out_path)
            return old
        _refuse_superseded_rewrite(prereg, out_path)
        if not force:
            raise LeakError("%s exists and was made from other inputs, parameters or code; rerun with --force"
                            % out_path)
    if embedder is None:
        embedder = E.LazyEmbedder(model, pooling)          # loads after the workers fork
    if embedder.name != si.want_name and not testing:
        raise LeakError("the embedder is %s, the config names %s" % (embedder.name, si.want_name))
    store = _Store(embedder, funnel_dir, procs=procs, batch=batch, force=force)
    index = eval_index(adapter, embedder, funnel_dir / EVAL_DESC, store=store)
    cal = calibrate(adapter, domain, embedder, SEED_PREFIX, store=store, recall_min=si.recall_min,
                    fpr_max=si.fpr_max, ref_rows=ref_rows, pool_rows=pool_rows)
    neg_pairs_rows = cal.pop("pairs")
    doc["seeds"] = dict(cal["seeds"])
    doc["detector"] = {"descriptor": {"model": model, "pooling": pooling, "view": E.VIEW, "embedder": embedder.name},
                       "dhash_variants": list(VARIANTS), "dhash_bits_max": DHASH_BITS_MAX,
                       "cos_threshold": cal["cos_threshold"], "candidate_bits": CANDIDATE_BITS, "top_cos": TOP_COS,
                       "rule": RULE}
    doc["calibration"] = cal
    doc["eval_descriptors"] = dict(index.record)
    doc["h6a"] = {"scope": scope}
    if not cal["ok"]:
        doc.update({"status": "calibration_failed", "scans": {}, "h6b": None, "h6c": None,
                    "descriptor_files": dict(sorted(store.files.items()))})
        doc["pairs_csv"] = _write_pairs(funnel_dir, [], neg_pairs_rows)
        doc["seconds"] = round(time.time() - t0, 1)
        write_json_atomic(out_path, doc)
        raise LeakCalibrationError("the copy detector failed its calibration (%s); %s written, nothing scanned"
                                   % ("; ".join(cal["why"]), out_path))
    theta, bits_max = _detector_params(cal)
    pool = store.get("pool", pool_rows)
    locate = _locator(pool_rows)
    extra_rows = _extra_rows(si, locate)
    extra = store.get("extra", [extra_rows[k] for k in sorted(extra_rows)]) if extra_rows else None
    copies_pairs = []
    copy_sources = set()

    def scan_set(name, rows):
        X = np.zeros((len(rows), pool["Xn"].shape[1]), dtype=np.float32)
        H = np.zeros((len(rows), len(VARIANTS)), dtype=np.uint64)
        ok = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            j = locate(r)
            src = pool if j is not None else extra
            j = j if j is not None else extra["index"][r["key"]]
            X[i], H[i], ok[i] = src["Xn"][j], src["H"][j], src["ok"][j]
        found = [_copy_entry(rows[r], index, j, c, b, v) for r, j, c, b, v in _scan(X, H, ok, index, theta, bits_max)]
        found.sort(key=lambda e: (e["key"], e["eval_split"], e["eval_key"]))
        hit_keys = {e["key"] for e in found}
        copy_sources.update(str(r.get("source")) for r in rows if r["key"] in hit_keys and r.get("source"))
        for e in found:
            copies_pairs.append(dict(e, set=name))
        n_img = len({e["key"] for e in found})
        return {"images": len(rows), "unscanned": int((~ok).sum()), "copies": n_img, "copy_pairs": len(found),
                "copy_found": n_img > 0, "splits_hit": sorted({e["eval_split"] for e in found}),
                "listed": found[:LISTED_MAX]}

    scans = {}
    by_source = {}
    for r in pool_rows:
        by_source.setdefault(r.get("source"), []).append(r)
    for src in sorted(by_source):
        scans["source:%s" % src] = scan_set("source:%s" % src, by_source[src])
    scans["base_B"] = scan_set("base_B", base_rows)
    for k in sorted(extra_bases):
        scans[k] = scan_set(k, extra_bases[k])
    for k in sorted(increments):
        scans[k] = scan_set(k, increments[k])
    with_copy = sorted(s.split(":", 1)[1] for s, v in scans.items() if s.startswith("source:") and v["copy_found"])
    # a source is quarantined as a whole when any of its images, in any scanned set, is a copy
    quarantine = sorted(set(with_copy) | copy_sources)
    # contract §6 H6(a): a copy of any never-train image, whatever its split, voids the
    # out-of-distribution numbers of every arm holding that source
    doc["h6a"] = {"scope": scope,
                  "copy_found": {s: bool(scans.get("source:%s" % s, {}).get("copy_found")) for s in scope},
                  "unscanned": {s: int(scans.get("source:%s" % s, {}).get("unscanned", 0)) for s in scope},
                  "missing_from_pool": [s for s in scope if "source:%s" % s not in scans],
                  "quarantine": quarantine,
                  "void_ood_arms_with": list(quarantine)}
    base_scans = ["base_B"] + sorted(extra_bases)
    inc_copies = {k: scans[k]["copies"] for k in sorted(increments)}
    base_copy = any(scans[k]["copy_found"] for k in base_scans)
    doc["h6b"] = {"base_copy": base_copy, "base_scans": base_scans, "same_as_base": sorted(same_as_base),
                  "increment_copies": inc_copies,
                  "incident": base_copy or any(v > 0 for v in inc_copies.values()),
                  "cleared": all(scans[k]["unscanned"] == 0 for k in base_scans + sorted(increments)),
                  "splits_hit": sorted(set().union(*[scans[k]["splits_hit"] for k in base_scans + sorted(increments)]))}
    labs, basis = exam_labs(domain, er, ref_rows)
    groups = {g: sorted(set(m)) for g, m in (raw["sources"].get("lab_groups") or {}).items()}
    added, links = {}, {}
    for s in with_copy:
        hit = scans["source:%s" % s]["splits_hit"]
        links[s] = {sp: labs.get(sp) for sp in hit}
        gs = sorted({labs[sp] for sp in hit if labs.get(sp)})
        if gs:
            added[s] = gs[0]
            for g in gs:
                if s not in groups[g]:
                    groups[g] = sorted(groups[g] + [s])
    doc["h6c"] = {"groups": groups, "added_by_h6a": added, "links": links, "exam_labs": labs,
                  "exam_lab_basis": basis, "unmapped_exams": sorted(s for s, g in labs.items() if g is None)}
    doc["scans"] = scans
    doc["status"] = "complete"
    doc["descriptor_files"] = dict(sorted(store.files.items()))
    doc["pairs_csv"] = _write_pairs(funnel_dir, copies_pairs, neg_pairs_rows)
    doc["seconds"] = round(time.time() - t0, 1)
    write_json_atomic(out_path, doc)
    log("calibration ok (cos >= %.4f); %d source(s) with a copy, base copy %s, %.0fs"
        % (theta, len(with_copy), doc["h6b"]["base_copy"], time.time() - t0))
    return doc


def _write_pairs(funnel_dir, copies, negatives, name=PAIRS_NAME, copy_kind="copy"):
    """The pairs file (cluster-only): every copy (version 2: every image hit,
    kind "hit") and the calibration's negative pairs (kinds negative_7_10 and
    negative_hard, the same rows in both versions)."""
    rows = []
    for e in sorted(copies, key=lambda e: (e["set"], e["key"], e["eval_split"], e["eval_key"])):
        rows.append([e["set"], e["key"], e["eval_split"], e["eval_key"], "%.6f" % e["cos"], e["bits"],
                     e["variant"], copy_kind])
    for e in negatives:
        rows.append([e["set"], e["key"], e["eval_split"], e["eval_key"], "%.6f" % e["cos"], e["bits"],
                     e["variant"], e["kind"]])
    path = Path(funnel_dir) / name
    sha = write_csv_atomic(path, PAIRS_HEADER, rows)
    return {"path": str(path), "sha256": sha, "rows": len(rows)}


# ================================================== detector version 2 (contract §14 A2)
# Why: leak_v1's calibration measured a false-positive rate per PAIR (a
# negative image against one partner), while the scan flags an IMAGE on its
# maximum cosine over every evaluation image and H6(a) quarantined a whole
# SOURCE on any one flagged image. Version 2 calibrates per image on
# same-domain negatives and decides a set (a source, the base, an increment)
# by a set rule. run_v2 writes leak_v2.json beside the untouched leak_v1.json.
FORMAT_V2 = "funnel-leak/2"
DETECTOR_VERSION_V2 = 2
PROTOCOL_V2 = "per_image_max_cosine/v2"
OUT_NAME_V2 = "leak_v2.json"
PAIRS_NAME_V2 = "leak_pairs_v2.csv"
NEGATIVES_NAME_V2 = "leak_negatives_v2.csv"
NEGATIVES_HEADER = ("tier", "key", "source", "cos", "eval_split", "eval_key", "bits")
EVAL_DESC_V2 = "leak_v2_eval_desc.npz"
DESC_V2 = "leak_v2_desc_%s.npz"
TIERS_V2 = ("hard", "hard_session_disjoint", "provenance_disjoint")
MIN_NEGATIVES = 100                  # a tier smaller than this cannot resolve a 1 % rate
MIN_NEGATIVES_TESTING = 5            # a synthetic world (recorded in the file)
SET_ALPHA = 0.001
UB_CONF = 0.95                       # a two-sided 95 % interval: its upper end is the one-sided 97.5 % bound
RULE_V2 = ("per image: copy iff the maximum over every evaluation image of the cosine >= cos_threshold, or the "
           "minimum over the 8 variants and every evaluation image of the dHash bits <= dhash_bits_max")
SET_RULE = ("a set of images (a source, the base, an increment) holds copies iff one of its images is within %d "
            "dHash bits of an evaluation image under a variant, or P(Binom(images scanned, p_false) >= embedding "
            "hits) < %s, p_false being the hard tier's one-sided 97.5 %% upper bound at cos_threshold"
            % (DHASH_BITS_MAX, SET_ALPHA))
THRESHOLD_RULE = ("the smallest 6-decimal cosine at which the per-image false-positive rate is <= fpr_max on the "
                  "hard tier (which must hold >= min_negatives images) and on every other tier holding >= "
                  "min_negatives images; never below version 1's threshold")


def capture_of(adapter):
    """row -> (capture session, capture date), each None when unknown: the
    adapter's optional capture_session function (the domain says which rows
    name a capture and how its date is read; the engine names neither), or
    None when the adapter offers none (no session is known, and the hard
    negative tier is then empty: the calibration refuses)."""
    fn = getattr(adapter, "capture_session", None)
    if not callable(fn):
        return None

    def of(row):
        got = fn(row)
        if not isinstance(got, (list, tuple)) or len(got) != 2:
            raise LeakError("adapter.capture_session must return (session, date), got %r" % (got,))
        s, d = got
        return (str(s) if s else None), (str(d) if d else None)
    return of


def threshold_for(scores, fpr_max):
    """The smallest 6-decimal cosine t with #(scores >= float32(t)) <=
    floor(fpr_max * n), compared in float32 as the scan compares; None for
    an empty tier."""
    c = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
    n = len(c)
    if n == 0:
        return None
    k = int(math.floor(float(fpr_max) * n + 1e-9))
    if k >= n:
        return -1.0
    t = math.floor(float(c[k]) * 1e6 + 1.0) / 1e6
    while int((c >= np.float32(t)).sum()) > k:
        t = round(t + 1e-6, 6)
    return round(t, 6)


def rate_record(scores, t):
    """{n, false_hits, fpr, ub} of per-image scores at threshold t; ub is the
    one-sided 97.5 % Clopper-Pearson upper bound (None without images)."""
    from . import estimate
    c = np.asarray(scores, dtype=np.float32)
    n = int(len(c))
    k = int((c >= np.float32(t)).sum()) if n else 0
    ub = _r6(estimate.clopper_pearson(k, n, UB_CONF)[1]) if n else None
    return {"n": n, "false_hits": k, "fpr": _r6(k / float(n)) if n else None, "ub": ub}


def set_verdict(images, hits, dhash_hits, p_false, alpha=SET_ALPHA):
    """The set rule (contract §14 A2): whether a set of `images` scanned
    images with `hits` embedding hits and `dhash_hits` dHash hits holds
    copies, against the hits its per-image false-positive rate p_false
    predicts. Without a rate, any embedding hit flags (fail closed)."""
    from . import estimate
    n, h, d = int(images), int(hits), int(dhash_hits)
    p = None if p_false is None else float(p_false)
    pv = 1.0 if (h <= 0 or p is None or n <= 0) else float(estimate.binom_upper_tail(h, n, p))
    why = []
    if d > 0:
        why.append("%d image(s) within %d dHash bits of an evaluation image under a variant" % (d, DHASH_BITS_MAX))
    if p is None and h > 0:
        why.append("no per-image false-positive rate to compare %d embedding hit(s) with (fail closed)" % h)
    elif h > 0 and pv < alpha:
        why.append("%d embedding hits of %d images, %.2f expected by chance (P = %.3g < %s)" % (h, n, n * p, pv, alpha))
    return {"images": n, "hits": h, "dhash_hits": d, "p_false": p,
            "expected_false_hits": None if p is None else round(n * p, 3), "p_value": float("%.6g" % pv),
            "alpha": alpha, "flagged": bool(why), "why": why}


def per_image(Xn, H, ok, index, q_sess=None, q_date=None, e_sess=None, e_date=None, chunk=CHUNK):
    """Per query row, as the scan scores an image: (the maximum cosine over
    the evaluation images, float32, -inf for a row that is not usable; its
    evaluation index; the minimum over the 8 variants and every evaluation
    image of the dHash bits, 65 for an unusable row; that image's index; the
    variant). With session and date codes (int arrays, -1 unknown), the
    evaluation images of the row's own session or date are left out of the
    maximum cosine (a calibration negative); the dHash minimum is over every
    evaluation image."""
    n = len(ok)
    best = np.full(n, -np.inf, dtype=np.float32)
    arg = np.full(n, -1, dtype=np.int64)
    bits = np.full(n, 65, dtype=np.int64)
    barg = np.full(n, -1, dtype=np.int64)
    bvar = np.zeros(n, dtype=np.int64)
    rows_ok = np.flatnonzero(np.asarray(ok, dtype=bool))
    H = np.asarray(H, dtype=np.uint64)
    He = index.H[:, 0]
    mask_sessions = q_sess is not None and e_sess is not None
    for s in range(0, len(rows_ok), chunk):
        rows = rows_ok[s:s + chunk]
        S = (np.asarray(Xn[rows], dtype=np.float32) @ index.Xn.T).astype(np.float32)
        if mask_sessions:
            qs, qd = np.asarray(q_sess)[rows][:, None], np.asarray(q_date)[rows][:, None]
            same = ((qs >= 0) & (qs == np.asarray(e_sess)[None, :])) | ((qd >= 0) & (qd == np.asarray(e_date)[None, :]))
            S = np.where(same, np.float32(-np.inf), S)
        best[rows] = S.max(axis=1)
        arg[rows] = S.argmax(axis=1)
        D_ = np.full(S.shape, 65, dtype=np.int64)
        V_ = np.zeros(S.shape, dtype=np.int64)
        for v in range(H.shape[1]):
            Dv = popcount64(H[rows, v][:, None] ^ He[None, :])
            better = Dv < D_
            D_ = np.where(better, Dv, D_)
            V_ = np.where(better, v, V_)
        bits[rows] = D_.min(axis=1)
        barg[rows] = D_.argmin(axis=1)
        bvar[rows] = V_[np.arange(len(rows)), barg[rows]]
    return best, arg, bits, barg, bvar


class _ReuseStore(_Store):
    """Descriptor sets for detector version 2: the file leak_v1.json records
    for a set, read (never written) when it still hashes as recorded and was
    made from exactly these rows by this embedder and preparation; else the
    set is described into this run's own file (leak_v2_desc_<set>.npz,
    leak_v2_eval_desc.npz), so no file leak_v1.json names is ever changed."""

    def __init__(self, embedder, funnel_dir, reuse=None, procs=1, batch=32, force=False):
        _Store.__init__(self, embedder, funnel_dir, procs=procs, batch=batch, force=force)
        self.reuse = dict(reuse or {})

    def path(self, name):
        if self.dir is None:
            return None
        return self.dir / (EVAL_DESC_V2 if name == "eval" else DESC_V2 % name)

    def _reusable(self, name, rows, prepare):
        rec = self.reuse.get(name) or {}
        path = rec.get("path")
        if not path:
            return None, "leak_v1.json records no file for this set"
        p = Path(path)
        if not p.is_file() and self.dir is not None and (self.dir / p.name).is_file():
            p = self.dir / p.name
        if not p.is_file():
            return None, "%s is missing" % p.name
        if C.sha256_file(p) != rec.get("sha256"):
            return None, "%s changed since leak_v1.json recorded it" % p.name
        if not E.image_file_current(p, E.image_ident(rows, self.embedder, prepare, len(VARIANTS))):
            return None, "%s was made from other rows, by another embedder or preparation" % p.name
        return E.load_images(p), None

    def get(self, name, rows, prepare=prepare_hashed):
        if name in self.sets:
            return self.sets[name]
        res, why = self._reusable(name, rows, prepare)
        if res is None:
            res = E.embed_images(rows, self.path(name), self.embedder, prepare=prepare, n_hashes=len(VARIANTS),
                                 procs=self.procs, batch=self.batch, force=self.force)
            origin = {"from": "computed", "not_reused": why}
        else:
            origin = {"from": "leak_v1"}
        _finish_set(res)
        self.sets[name] = res
        if res.get("path"):
            self.files[name] = dict({"path": res["path"], "sha256": res["sha256"]}, **origin)
        return res


def _v1_reading(v1doc, v1_path, cal1, floor):
    """The version 1 reading, as leak_v1.json recorded it (or as this run
    rebuilt its calibration when there is no file): reported as the invalid
    first reading (contract §14 A2)."""
    out = {"status": "invalid first reading (contract §14 A2): a per-pair calibration applied per image and per "
                     "source", "cos_threshold": floor}
    if v1doc is None:
        out.update(file=None, calibration={k: cal1.get(k) for k in ("ok", "why", "negatives", "positives")},
                   note="no leak_v1.json: its calibration was rebuilt here (same seeds); its scan is the version 1 "
                        "rule applied to this run's scores")
        return out
    cal = v1doc.get("calibration") or {}
    h6a, h6b = v1doc.get("h6a") or {}, v1doc.get("h6b") or {}
    q = list(h6a.get("quarantine") or [])
    n_src = len([k for k in (v1doc.get("scans") or {}) if str(k).startswith("source:")])
    out.update(file=file_record(v1_path), status_v1=v1doc.get("status"),
               calibration={"ok": cal.get("ok"), "cos_threshold": cal.get("cos_threshold"),
                            "negatives": {k: v for k, v in (cal.get("negatives") or {}).items() if k != "by_pair"},
                            "per": "pair"},
               h6a={"quarantine": q, "quarantined": len(q), "sources_scanned": n_src},
               h6b={k: h6b.get(k) for k in ("base_copy", "increment_copies", "incident", "cleared")})
    return out


def run_v2(prereg, domain, funnel_dir, adapter, embedder=None, procs=5, batch=32, force=False, testing=False,
           pool_summary_path=None, increment_exps=INCREMENT_EXPS):
    """leak_v2.json (copy detector version 2, contract §14 A2), with the
    version 1 reading beside it; leak_negatives_v2.csv and leak_pairs_v2.csv
    (cluster-only). Descriptors come from the files leak_v1.json records
    wherever they still hash as recorded (_ReuseStore). leak_v1.json and its
    files are only read. A rerun on the same inputs, parameters and code is
    a no-op (a failed calibration refuses again); other inputs refuse
    unless force."""
    t0 = time.time()
    funnel_dir = Path(funnel_dir)
    raw = _raw(domain)
    need = D.leak_detector_version(prereg)
    if need != DETECTOR_VERSION_V2:
        raise LeakError("the prereg requires copy detector version %d; run_v2 writes version %d"
                        % (need, DETECTOR_VERSION_V2))
    amendment = D.leak_detector_amendment(prereg)
    si = _scan_inputs(prereg, domain, adapter, pool_summary_path, increment_exps)
    ref_rows, pool_rows = si.ref_rows, si.pool_rows
    cap = capture_of(adapter)
    min_neg = MIN_NEGATIVES_TESTING if testing else MIN_NEGATIVES
    v1_path = funnel_dir / OUT_NAME
    v1doc = read_json(v1_path) if v1_path.is_file() else None
    reuse = dict((v1doc or {}).get("descriptor_files") or {})
    if (v1doc or {}).get("eval_descriptors"):
        reuse["eval"] = v1doc["eval_descriptors"]
    neg_groups = sorted({s for p in si.neg_pairs for s in p} - {si.ref_name})
    params = dict(si.params, detector_version=DETECTOR_VERSION_V2, protocol=PROTOCOL_V2, min_negatives=min_neg,
                  set_alpha=SET_ALPHA, ub_conf=UB_CONF, tiers=list(TIERS_V2), provenance_disjoint_sources=neg_groups,
                  capture="adapter.capture_session" if cap is not None else None)
    inputs = dict(si.inputs)
    if v1doc is not None:
        inputs["leak_v1"] = file_record(v1_path)
    modules = (sys.modules[__name__], E, adapter)
    doc = header(FORMAT_V2, domain, prereg, inputs, modules=modules, testing=testing)
    doc.update({"row_sets": si.row_sets, "params": params, "detector_version": DETECTOR_VERSION_V2,
                "amendment": ({k: amendment.get(k) for k in ("id", "date", "kind", "section", "post_hoc")}
                              if amendment else None)})
    out_path = funnel_dir / OUT_NAME_V2
    if out_path.exists():
        old = read_json(out_path)
        if _identity(old) == _identity(doc):
            if not (old.get("calibration") or {}).get("ok"):
                raise LeakCalibrationError("%s: the calibration failed (%s); nothing was scanned"
                                           % (out_path, "; ".join(old["calibration"].get("why", []))))
            log("%s is current; nothing to do" % out_path)
            return old
        if not force:
            raise LeakError("%s exists and was made from other inputs, parameters or code; rerun with --force"
                            % out_path)
    if embedder is None:
        embedder = E.LazyEmbedder(si.model, si.pooling)       # loads after the workers fork, and only if needed
    if embedder.name != si.want_name and not testing:
        raise LeakError("the embedder is %s, the config names %s" % (embedder.name, si.want_name))
    store = _ReuseStore(embedder, funnel_dir, reuse, procs=procs, batch=batch, force=force)
    index = eval_index(adapter, embedder, None, store=store)

    # ---- version 1's calibration, rebuilt: the floor, the positives and the negative pairs
    ps = {}
    cal1 = calibrate(adapter, domain, embedder, SEED_PREFIX, store=store, recall_min=si.recall_min,
                     fpr_max=si.fpr_max, ref_rows=ref_rows, pool_rows=pool_rows, positive_scores=ps)
    v1_pairs = cal1.pop("pairs")
    floor = cal1.get("cos_threshold")
    why = []
    recorded = ((v1doc or {}).get("calibration") or {}).get("cos_threshold")
    if floor is None:
        why.append("version 1's threshold cannot be rebuilt (a family's positives could not be described)")
    elif recorded is not None and abs(float(recorded) - float(floor)) > 1.5e-6:
        raise LeakCalibrationError("version 1's positives, rebuilt from its seeds and descriptor files, give "
                                   "threshold %s, but %s records %s: they are not the positives it was calibrated "
                                   "with" % (floor, v1_path, recorded))

    # ---- the evaluation side: splits, keys, session and date codes
    ev_row = {}
    for split, rows in si.eval_rows.items():
        for r in rows:
            ev_row["%s|%s" % (split, r["key"])] = r
    sess_codes, date_codes = {}, {}

    def code(table, v):
        return -1 if not v else table.setdefault(v, len(table))
    none = (lambda r: (None, None))
    capf = cap or none
    e_sd = [capf(ev_row.get(k) or {}) for k in index.keys]
    e_sess = np.array([code(sess_codes, s) for s, _d in e_sd], dtype=np.int64)
    e_date = np.array([code(date_codes, d) for _s, d in e_sd], dtype=np.int64)
    eval_sessions = {s for s, _d in e_sd if s}
    eval_dates = {d for _s, d in e_sd if d}

    # ---- negatives, per image
    ref = store.get("reference", ref_rows)
    pool = store.get("pool", pool_rows)
    tiers, tier_info, neg_rows = {}, {}, []

    def add_tier(name, res, rows, basis, sources, with_sessions):
        idx = np.array([res["index"][r["key"]] for r in rows], dtype=np.int64)
        info = {"basis": basis, "sources": sources, "candidates": len(rows)}
        if not len(idx):
            tiers[name] = np.zeros(0, dtype=np.float32)
            tier_info[name] = dict(info, excluded_dhash_copies=0, unscored=0)
            return
        ok = res["ok"][idx]
        if with_sessions:
            sd = [capf(r) for r in rows]
            qs = np.array([sess_codes.get(s, -1) if s else -1 for s, _d in sd], dtype=np.int64)
            qd = np.array([date_codes.get(d, -1) if d else -1 for _s, d in sd], dtype=np.int64)
            best, arg, bits, _ba, _bv = per_image(res["Xn"][idx], np.asarray(res["H"], dtype=np.uint64)[idx], ok,
                                                  index, qs, qd, e_sess, e_date)
        else:
            best, arg, bits, _ba, _bv = per_image(res["Xn"][idx], np.asarray(res["H"], dtype=np.uint64)[idx], ok,
                                                  index)
        keep = ok & np.isfinite(best) & (bits > DHASH_BITS_MAX)
        info.update(excluded_dhash_copies=int((ok & (bits <= DHASH_BITS_MAX)).sum()),
                    unscored=int((~ok).sum() + (ok & ~np.isfinite(best) & (bits > DHASH_BITS_MAX)).sum()))
        tiers[name] = best[keep].astype(np.float32)
        tier_info[name] = info
        for i in np.flatnonzero(keep):
            j = int(arg[i])
            neg_rows.append([name, rows[i]["key"], str(rows[i].get("source") or ""), "%.6f" % float(best[i]),
                             index.split[j], index.eval_key[j], int(bits[i])])
    hard_rows = [r for r in ref_rows if capf(r)[0]]
    add_tier("hard", ref, hard_rows, "reference images with a known capture session, scored against the evaluation "
                                     "images of other sessions and other dates", [si.ref_name], True)
    tier_info["hard"]["excluded_no_capture_session"] = len(ref_rows) - len(hard_rows)
    if cap is None:
        tier_info["hard"]["why_empty"] = "the adapter offers no capture_session: no capture session is known"
    disj = [r for r in hard_rows if capf(r)[0] not in eval_sessions and capf(r)[1] not in eval_dates]
    add_tier("hard_session_disjoint", ref, disj, "the hard images whose session and date no evaluation image shares",
             [si.ref_name], True)
    prov = [r for r in pool_rows if r.get("source") in neg_groups]
    add_tier("provenance_disjoint", pool, prov, "the non-reference groups of leak.negative_source_pairs",
             neg_groups, False)

    # ---- the threshold
    constraining = [t for t in TIERS_V2 if t == "hard" or len(tiers[t]) >= min_neg]
    t_raw = {t: threshold_for(tiers[t], si.fpr_max) for t in constraining}
    finite = [v for v in t_raw.values() if v is not None]
    theta = round(max([float(floor)] + finite), 6) if floor is not None else None
    negatives = {}
    for t in TIERS_V2:
        sc = tiers[t]
        rec_new = rate_record(sc, theta) if theta is not None else {"n": len(sc)}
        rec_old = rate_record(sc, floor) if floor is not None else {}
        negatives[t] = dict(tier_info[t], **rec_new)
        negatives[t]["at_v1_threshold"] = {k: rec_old.get(k) for k in ("false_hits", "fpr", "ub")}
        negatives[t]["constraining"] = t in constraining
        negatives[t]["tier_threshold"] = t_raw.get(t)
        if len(sc):
            negatives[t]["cos"] = {"max": _r6(sc.max()), "q99": _r6(np.quantile(sc, 0.99)),
                                   "median": _r6(np.median(sc))}
    if negatives["hard"]["n"] < max(1, min_neg):
        why.append("the hard negative tier holds %d images, fewer than %d%s" % (
            negatives["hard"]["n"], max(1, min_neg),
            " (%s)" % tier_info["hard"]["why_empty"] if tier_info["hard"].get("why_empty") else ""))
    for t in constraining:
        if theta is not None and negatives[t]["n"] and negatives[t]["false_hits"] > si.fpr_max * negatives[t]["n"] + 1e-9:
            why.append("tier %s: %d false hits of %d at %s" % (t, negatives[t]["false_hits"], negatives[t]["n"], theta))
    p_false = negatives["hard"].get("ub") if theta is not None else None

    # ---- recall per augmentation family at the new threshold (version 1's positives)
    from . import estimate
    positives, limits = {}, []
    for fam in FAMILIES:
        c, b = ps["cos"][fam], ps["bits"][fam]
        fin = np.isfinite(c)
        n = len(c)
        hits = (_rule(c, b, theta, DHASH_BITS_MAX) & fin) if theta is not None else np.zeros(n, dtype=bool)
        hits1 = (_rule(c, b, floor, DHASH_BITS_MAX) & fin) if floor is not None else np.zeros(n, dtype=bool)
        k = int(hits.sum())
        lo = estimate.clopper_pearson(k, n, UB_CONF)[0] if n else 0.0
        positives[fam] = {"n": n, "hits": k, "recall": _r6(k / float(n)) if n else None, "lb": _r6(lo),
                          "failed": int(ps["failed"][fam]), "dhash_hits": int(((b <= DHASH_BITS_MAX) & fin).sum()),
                          "cos_q05": _r6(np.quantile(c[fin], 0.05)) if fin.any() else None,
                          "params_seed": cal1["seeds"].get("positives/%s" % fam),
                          "at_v1_threshold": {"hits": int(hits1.sum()),
                                              "recall": _r6(int(hits1.sum()) / float(n)) if n else None}}
        if n and k < si.recall_min * n:
            limits.append({"family": fam, "recall": positives[fam]["recall"], "lb": positives[fam]["lb"], "n": n,
                           "why": "below the %s recall gate at the version 2 threshold: a known limit, not a refusal "
                                  "(contract §14 A2); the 8 dHash variants still catch flips, rotations and "
                                  "re-encoding" % si.recall_min})
    cal = {"ok": not why, "why": why, "protocol": PROTOCOL_V2, "detector_version": DETECTOR_VERSION_V2,
           "cos_threshold": theta, "floor": floor, "dhash_bits_max": DHASH_BITS_MAX,
           "recall_min": si.recall_min, "fpr_max": si.fpr_max, "min_negatives": min_neg, "testing": bool(testing),
           "tier_thresholds": t_raw, "constraining": constraining, "threshold_rule": THRESHOLD_RULE,
           "negatives": negatives, "positives": positives, "known_limits": limits, "p_false": p_false,
           "set_rule": {"rule": SET_RULE, "alpha": SET_ALPHA, "p_false": p_false},
           "seeds": dict(cal1["seeds"]), "v1_threshold": {"rebuilt": floor, "recorded": recorded,
                                                          "reproduced": None if recorded is None else True}}
    doc["seeds"] = dict(cal1["seeds"])
    doc["detector"] = {"version": DETECTOR_VERSION_V2,
                       "descriptor": {"model": si.model, "pooling": si.pooling, "view": E.VIEW,
                                      "embedder": embedder.name},
                       "dhash_variants": list(VARIANTS), "dhash_bits_max": DHASH_BITS_MAX, "cos_threshold": theta,
                       "rule": RULE_V2, "set_rule": SET_RULE}
    doc["calibration"] = cal
    doc["eval_descriptors"] = dict(index.record)
    doc["negatives_csv"] = _write_negatives(funnel_dir, neg_rows)
    v1r = _v1_reading(v1doc, v1_path, cal1, floor)
    if not cal["ok"]:
        doc.update({"status": "calibration_failed", "scans": {}, "h6a": {"scope": si.scope}, "h6b": None, "h6c": None,
                    "readings": {"v1": v1r, "v2": None}, "descriptor_files": dict(sorted(store.files.items()))})
        doc["pairs_csv"] = _write_pairs(funnel_dir, [], v1_pairs, name=PAIRS_NAME_V2)
        doc["seconds"] = round(time.time() - t0, 1)
        write_json_atomic(out_path, doc)
        raise LeakCalibrationError("copy detector version 2 failed its calibration (%s); %s written, nothing scanned"
                                   % ("; ".join(why), out_path))

    # ---- scans: every image scored against every evaluation image, both rules
    locate = _locator(pool_rows)
    extra_rows = _extra_rows(si, locate)
    extra = store.get("extra", [extra_rows[k] for k in sorted(extra_rows)]) if extra_rows else None
    hits_pairs = []
    # one entry per image, whatever set it is scanned in: (source, scanned, embedding hit, dHash hit, v1-rule
    # hit), keyed by where its descriptor lives (a pool image under its pool index, any other by its key)
    per_image_rec = {}
    v1scans = (v1doc or {}).get("scans") or {}

    def scan_set(name, rows):
        X = np.zeros((len(rows), pool["Xn"].shape[1]), dtype=np.float32)
        H = np.zeros((len(rows), len(VARIANTS)), dtype=np.uint64)
        ok = np.zeros(len(rows), dtype=bool)
        where = []
        for i, r in enumerate(rows):
            j = locate(r)
            where.append(("pool", j) if j is not None else ("extra", r["key"]))
            src = pool if j is not None else extra
            j = j if j is not None else extra["index"][r["key"]]
            X[i], H[i], ok[i] = src["Xn"][j], src["H"][j], src["ok"][j]
        best, arg, bits, barg, bvar = per_image(X, H, ok, index)
        emb = ok & (best >= np.float32(theta))
        dh = ok & (bits <= DHASH_BITS_MAX)
        hit = emb | dh
        v1hit = ok & ((best >= np.float32(floor)) | dh)
        listed = []
        for i in np.flatnonzero(hit):
            j = int(barg[i]) if dh[i] else int(arg[i])
            e = {"key": rows[i]["key"], "image": str(rows[i]["image"]), "eval_split": index.split[j],
                 "eval_key": index.eval_key[j], "cos": _r6(best[i]), "bits": int(bits[i]),
                 "variant": VARIANTS[int(bvar[i])], "by": "both" if (emb[i] and dh[i]) else
                 ("dhash" if dh[i] else "embedding")}
            listed.append(e)
            hits_pairs.append(dict(e, set=name))
        for i, r in enumerate(rows):
            per_image_rec.setdefault(where[i], (str(r.get("source") or ""), bool(ok[i]), bool(emb[i]), bool(dh[i]),
                                                bool(v1hit[i])))
        n_ok = int(ok.sum())
        verdict = set_verdict(n_ok, int(emb.sum()), int(dh.sum()), p_false)
        old = v1scans.get(name)
        v1 = {"copies": int(v1hit.sum()), "copy_found": bool(v1hit.any())}
        if isinstance(old, dict):
            v1["leak_v1"] = {"copies": old.get("copies"), "copy_found": old.get("copy_found")}
            v1["reproduced"] = old.get("copies") == v1["copies"]
        fin = best[np.isfinite(best)]
        return {"images": len(rows), "unscanned": int((~ok).sum()), "scanned": n_ok, "hits": int(hit.sum()),
                "embedding_hits": int(emb.sum()), "dhash_hits": int(dh.sum()),
                "expected_false_hits": verdict["expected_false_hits"], "p_value": verdict["p_value"],
                "verdict": verdict, "copy_found": verdict["flagged"],
                "splits_hit": sorted({e["eval_split"] for e in listed}),
                "max_cos": _r6(fin.max()) if len(fin) else None, "listed": listed[:LISTED_MAX], "v1_rule": v1}

    scans = {}
    by_source = {}
    for r in pool_rows:
        by_source.setdefault(r.get("source"), []).append(r)
    for src in sorted(by_source):
        scans["source:%s" % src] = scan_set("source:%s" % src, by_source[src])
    scans["base_B"] = scan_set("base_B", si.base_rows)
    for k in sorted(si.extra_bases):
        scans[k] = scan_set(k, si.extra_bases[k])
    for k in sorted(si.increments):
        scans[k] = scan_set(k, si.increments[k])

    # ---- H6(a): the set rule per source, over every scanned image of the source
    agg = {}
    for _where, (src, okk, emb, dh, v1h) in per_image_rec.items():
        a = agg.setdefault(src, [0, 0, 0, 0, 0])
        a[0] += int(okk)
        a[1] += int(emb)
        a[2] += int(dh)
        a[3] += int(v1h)
        a[4] += int(not okk)
    sources = {s: dict(set_verdict(a[0], a[1], a[2], p_false), unscanned=a[4], v1_rule_copies=a[3])
               for s, a in sorted(agg.items()) if s}
    quarantine = sorted(s for s, v in sources.items() if v["flagged"])
    doc["h6a"] = {"scope": si.scope, "rule": SET_RULE,
                  "copy_found": {s: bool((sources.get(s) or {}).get("flagged")) for s in si.scope},
                  "unscanned": {s: int(scans.get("source:%s" % s, {}).get("unscanned", 0)) for s in si.scope},
                  "missing_from_pool": [s for s in si.scope if "source:%s" % s not in scans],
                  "sources": sources, "quarantine": quarantine, "void_ood_arms_with": list(quarantine)}
    # ---- H6(b): the set rule on the base and on each increment
    base_scans = ["base_B"] + sorted(si.extra_bases)
    inc_names = sorted(si.increments)

    def brief(k):
        v = scans[k]
        return {"images": v["scanned"], "hits": v["embedding_hits"], "dhash_hits": v["dhash_hits"],
                "expected_false_hits": v["expected_false_hits"], "p_value": v["p_value"], "flagged": v["copy_found"],
                "v1_rule_copies": v["v1_rule"]["copies"]}
    base_copy = any(scans[k]["copy_found"] for k in base_scans)
    doc["h6b"] = {"rule": SET_RULE, "base_copy": base_copy, "base_scans": base_scans,
                  "same_as_base": sorted(si.same_as_base), "sets": {k: brief(k) for k in base_scans + inc_names},
                  "increment_flagged": {k: scans[k]["copy_found"] for k in inc_names},
                  "incident": base_copy or any(scans[k]["copy_found"] for k in inc_names),
                  "cleared": all(scans[k]["unscanned"] == 0 for k in base_scans + inc_names),
                  "splits_hit": sorted(set().union(*[scans[k]["splits_hit"] for k in base_scans + inc_names
                                                     if scans[k]["copy_found"]]))}
    # ---- H6(c): only flagged sources join a lab group
    labs, basis = exam_labs(domain, si.eval_rows, ref_rows)
    groups = {g: sorted(set(m)) for g, m in (raw["sources"].get("lab_groups") or {}).items()}
    added, links = {}, {}
    for s in [x for x in quarantine if "source:%s" % x in scans]:
        hit = scans["source:%s" % s]["splits_hit"]
        links[s] = {sp: labs.get(sp) for sp in hit}
        gs = sorted({labs[sp] for sp in hit if labs.get(sp)})
        if gs:
            added[s] = gs[0]
            for g in gs:
                if s not in groups[g]:
                    groups[g] = sorted(groups[g] + [s])
    doc["h6c"] = {"groups": groups, "added_by_h6a": added, "links": links, "exam_labs": labs,
                  "exam_lab_basis": basis, "unmapped_exams": sorted(s for s, g in labs.items() if g is None)}
    reproduced = [v["v1_rule"].get("reproduced") for v in scans.values() if "reproduced" in v["v1_rule"]]
    v1r["reproduced_by_this_run"] = {"threshold": cal["v1_threshold"]["reproduced"],
                                     "scans": (all(reproduced) if reproduced else None),
                                     "sets_compared": len(reproduced)}
    doc["readings"] = {
        "v1": v1r,
        "v2": {"cos_threshold": theta, "p_false": p_false, "known_limits": [x["family"] for x in limits],
               "h6a": {"quarantine": quarantine, "quarantined": len(quarantine), "sources_scanned": len(sources)},
               "h6b": {"incident": doc["h6b"]["incident"], "base_copy": base_copy,
                       "increment_flagged": doc["h6b"]["increment_flagged"]}}}
    doc["scans"] = scans
    doc["status"] = "complete"
    doc["descriptor_files"] = dict(sorted(store.files.items()))
    doc["pairs_csv"] = _write_pairs(funnel_dir, hits_pairs, v1_pairs, name=PAIRS_NAME_V2, copy_kind="hit")
    doc["seconds"] = round(time.time() - t0, 1)
    write_json_atomic(out_path, doc)
    log("version 2: cos >= %.4f (version 1: %.4f), p_false %s; %d of %d source(s) flagged (version 1: %s), base %s, "
        "%.0fs" % (theta, floor, p_false, len(quarantine), len(sources),
                   v1r.get("h6a", {}).get("quarantined", "n/a"), base_copy, time.time() - t0))
    return doc


def _write_negatives(funnel_dir, rows):
    path = Path(funnel_dir) / NEGATIVES_NAME_V2
    sha = write_csv_atomic(path, NEGATIVES_HEADER, sorted(rows))
    return {"path": str(path), "sha256": sha, "rows": len(rows)}
