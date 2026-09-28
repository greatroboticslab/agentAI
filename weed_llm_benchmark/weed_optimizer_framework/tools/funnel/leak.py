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
        res["Xn"] = _normalise(res["X"])
        res["ok"] = np.isfinite(res["Xn"]).all(axis=1) & np.asarray(res["hash_ok"], dtype=bool)
        res["index"] = {k: i for i, k in enumerate(res["keys"])}
        self.sets[name] = res
        if res.get("path"):
            self.files[name] = {"path": res["path"], "sha256": res["sha256"]}
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
              fpr_max=FPR_MAX, ref_rows=None, pool_rows=None, procs=1):
    """The detector's calibration (module docstring). Returns the calibration
    record; its "pairs" list holds every negative pair (for the pairs file,
    never for leak_v1.json)."""
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


def run(prereg, domain, funnel_dir, adapter, embedder=None, procs=5, batch=32, force=False, testing=False,
        pool_summary_path=None, increment_exps=INCREMENT_EXPS):
    """leak_v1.json, leak_pairs_v1.csv and the descriptor files (module
    docstring). A rerun on the same inputs, parameters and code is a no-op
    (a failed calibration refuses again); other inputs refuse unless force."""
    t0 = time.time()
    funnel_dir = Path(funnel_dir)
    raw = _raw(domain)
    ref_name = raw["sources"]["reference"]
    fams = family_params(domain)
    neg_pairs = _negative_pairs_cfg(domain)
    h6 = ((_raw(prereg).get("hypotheses") or {}).get("H6") or {}).get("calibration") or {}
    if "recall_min_per_family" not in h6 or "fpr_max" not in h6:
        raise LeakError("the prereg's H6 calibration has no recall_min_per_family / fpr_max")
    recall_min, fpr_max = float(h6["recall_min_per_family"]), float(h6["fpr_max"])
    model, pooling = E.features_config(domain)
    want_name = E.embedder_name(model, pooling)
    ps_path = Path(pool_summary_path) if pool_summary_path else STEP1_DIR / "pool_summary.json"
    pool_summary = read_json(ps_path)
    scope = h6a_scope(pool_summary)
    ref_rows, ref_rec = reference_rows(domain)
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
    pool_rows = sorted(adapter.pool_rows(), key=lambda r: r["key"])
    base_raw = adapter.base_rows()
    base_rows = harvested(base_raw)
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
    er = adapter.eval_rows()
    if not isinstance(er, dict) or not er:
        raise LeakError("the adapter returned no evaluation rows")
    eval_digest = _rows_digest([dict(r, key="%s|%s" % (s, r["key"])) for s in sorted(er)
                                for r in sorted(er[s], key=lambda r: r["key"])])
    row_sets = {"reference": _rows_digest(ref_rows), "pool": _rows_digest(pool_rows),
                "base": _rows_digest(base_rows), "eval": eval_digest}
    row_sets.update({"increment:%s" % k: _rows_digest(v) for k, v in increments.items()})
    row_sets.update({"base:%s" % k: _rows_digest(v) for k, v in extra_bases.items()})
    row_sets["same_as_base"] = sorted(same_as_base)
    params = {"families": fams, "negative_source_pairs": [list(p) for p in neg_pairs], "recall_min": recall_min,
              "fpr_max": fpr_max, "embedder": want_name, "view": E.VIEW, "dhash_bits_max": DHASH_BITS_MAX,
              "candidate_bits": CANDIDATE_BITS, "top_cos": TOP_COS, "neg_bits": list(NEG_BITS),
              "pos_per_family": POS_PER_FAMILY, "neg_hard": NEG_HARD, "seed_prefix": SEED_PREFIX,
              "increment_exps": list(increment_exps)}
    inputs = {"pool_summary": file_record(ps_path), "reference_manifest": ref_rec}
    modules = (sys.modules[__name__], E, adapter)
    doc = header("leak", domain, prereg, inputs, modules=modules, testing=testing)
    doc.update({"row_sets": row_sets, "params": params})
    out_path = funnel_dir / OUT_NAME
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
        embedder = E.LazyEmbedder(model, pooling)          # loads after the workers fork
    if embedder.name != want_name and not testing:
        raise LeakError("the embedder is %s, the config names %s" % (embedder.name, want_name))
    store = _Store(embedder, funnel_dir, procs=procs, batch=batch, force=force)
    index = eval_index(adapter, embedder, funnel_dir / EVAL_DESC, store=store)
    cal = calibrate(adapter, domain, embedder, SEED_PREFIX, store=store, recall_min=recall_min, fpr_max=fpr_max,
                    ref_rows=ref_rows, pool_rows=pool_rows)
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
    # base and increment rows are pool images; any that are not are described apart
    by_sha = {r.get("sha256"): i for i, r in enumerate(pool_rows) if r.get("sha256")}
    by_img = {str(r["image"]): i for i, r in enumerate(pool_rows)}

    def locate(r):
        i = by_sha.get(r.get("sha256")) if r.get("sha256") else None
        return i if i is not None else by_img.get(str(r["image"]))

    extra_rows = {}
    for rows in ([base_rows] + [extra_bases[k] for k in sorted(extra_bases)]
                 + [increments[k] for k in sorted(increments)]):
        for r in rows:
            if locate(r) is None:
                extra_rows.setdefault(r["key"], r)
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


def _write_pairs(funnel_dir, copies, negatives):
    rows = []
    for e in sorted(copies, key=lambda e: (e["set"], e["key"], e["eval_split"], e["eval_key"])):
        rows.append([e["set"], e["key"], e["eval_split"], e["eval_key"], "%.6f" % e["cos"], e["bits"],
                     e["variant"], "copy"])
    for e in negatives:
        rows.append([e["set"], e["key"], e["eval_split"], e["eval_key"], "%.6f" % e["cos"], e["bits"],
                     e["variant"], e["kind"]])
    path = Path(funnel_dir) / PAIRS_NAME
    sha = write_csv_atomic(path, PAIRS_HEADER, rows)
    return {"path": str(path), "sha256": sha, "rows": len(rows)}
