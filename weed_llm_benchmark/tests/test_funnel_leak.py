#!/usr/bin/env python3
"""funnel/leak.py: the H6 copy detector, its calibration and the scans
(contract docs/FUNNEL_AUDIT.md §3.4, §6 H6, §8.7; runner
docs/FUNNEL_AUDIT_RUNNER.md §4.13, §5.3.3).

The dHash part is checked against the project's own dHash:
  * every variant of dhash_variants equals inc/common.dhash of the PNG of
    that flip or rotation (the literal definition), on images of many sizes;
    a horizontally flipped image holds the original's dHash under "hflip";
  * popcount64 against Python's bit count.

Augmentations: each family is deterministic under its seed and draws its
parameters within the family's range; the config's keys (weed.json:
max_frac, max_delta, max_deg, radius, quality, size) give the contract's
ranges; an unknown key or a missing family refuses. threshold() keeps at
least recall_min of every family's positives.

The synthetic world: "photographs" built from a 9 x 8 grid of luminance
blocks (so dHash distances can be planted exactly) coloured with three hue
bands (a hue triple per photograph, any two triples sharing at most one hue).
The fake descriptor is a hue histogram, which survives every augmentation
family; the real dHash runs on the pixels. Sources follow weed.json
(the reference split, the MH-Weed16 source as the provenance-disjoint
negative group), with a pool source holding one augmented copy of an
evaluation image per family, a pool image 8 dHash bits from an evaluation
image with other colours, a base-B image that is a copy of a dev image and
an increment step with a copy.

Pinned on that world:
  * the calibration passes (recall >= 0.95 per family, both negative sets
    non-empty, false-positive rate 0 <= 1 %) before anything is scanned;
  * every planted copy is found, with the right evaluation key; the 8-bit
    distinct image is not; the sources with copies are quarantined whole;
  * H6(b): the base copy and the increment copy are an incident; H6(c):
    the copying source joins the lab group of the split it copies, from the
    exam rows' dataset; an exam whose dataset no group names is unmapped;
  * leak_v1.json holds evaluation images as keys only: no evaluation path,
    no pixel or descriptor array; the pairs file lists copies and negatives;
  * a rerun is a no-op; changed inputs refuse without force;
  * detect() finds a planted copy on disk, clears a clean image, refuses an
    unreadable one, and refuses to run under a failed calibration;
  * an embedder that maps everything to one point makes the negatives' false-
    positive rate exceed 1 %: leak refuses (LeakCalibrationError), writes
    leak_v1.json with ok false and scans nothing;
  * the H6(a) scope on the real local pool_summary.json is the number of
    sources the contract's §3.4 states (read from the contract text, not
    typed), and each source §3.4 lists matches exactly one scope slug;
  * H6(a): void_ood_arms_with is every source with a copy, a copy of a dev
    image included (contract §6 H6(a));
  * H6(b): the experiment's own copy of the base among its manifests is not
    an increment (same rows: skipped); an earlier, different base is scanned
    as a base;
  * the dHash half of the rule alone finds a rotated copy whose descriptor
    matches nothing;
  * detect refuses a given descriptor that is NaN or of the wrong size, and
    an evaluation index made by another embedder, or other
    evaluation descriptors than leak_v1.json records; a consumer's
    eval_index refuses to overwrite that file, and refuses an evaluation
    image it cannot describe;
  * a sample-lock amendment leaves leak_v1.json current (prereg core);
  * an empty 7-10-bit negative set fails the calibration; a row that cannot
    be described is unscanned, and the base and increments are not cleared.

Run:  python3 tests/test_funnel_leak.py
"""
import csv
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_leak_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
HERE = pathlib.Path(__file__).resolve()
try:                                  # PYTHONPATH may name a package copy (the mutation harness)
    import weed_optimizer_framework  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE.parents[1]))
REAL_REPO = HERE.parents[2]
REAL_INC = HERE.parents[1] / "results" / "framework" / "inc"
CONTRACT = REAL_REPO / "docs" / "FUNNEL_AUDIT.md"
for src, dst in ((CONTRACT, TMP / "repo" / "docs" / "FUNNEL_AUDIT.md"),
                 (REAL_INC / "funnel" / "prereg_v1.json", TMP / "inc" / "funnel" / "prereg_v1.json")):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, reason):
    print("SKIP: %s (%s)" % (name, reason))
    SKIPS.append(name)


def raises(fn, err, contains=None):
    try:
        fn()
    except err as e:
        if contains and contains not in str(e):
            print("       raised without %r: %s" % (contains, e))
            return None
        return str(e) or "raised"
    return None


try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None

REF = "train_core"
MH = "project_agml__mh_weed16_weed_detection"
TUF = "rf_tuf__weed-3434e"
CLEAN = "rf_clean__plain-set"
CWP = "rf_karthikeya-c8pvy__weed-detection-cwp10"
N_BINS = 30
BLOCK = 12


# ------------------------------------------------------------ the world
class HueEmbedder:
    """A descriptor that survives re-export augmentations: the histogram of
    hue over saturated, lit pixels (N_BINS bins), L2-normalised."""
    dim = N_BINS

    def __init__(self, name="facebook/dinov2-base:cls"):
        self.name = name
        self.calls = 0

    def __call__(self, pils):
        self.calls += 1
        out = []
        for p in pils:
            hsv = np.asarray(p.convert("HSV"), dtype=np.int64)
            m = (hsv[..., 1] > 64) & (hsv[..., 2] > 40)
            h = np.bincount((hsv[..., 0][m] * N_BINS) // 256, minlength=N_BINS).astype(np.float32)
            nrm = np.linalg.norm(h)
            out.append(h / nrm if nrm > 0 else h)
        return np.stack(out)


class ConstEmbedder(HueEmbedder):
    def __call__(self, pils):
        self.calls += 1
        return np.ones((len(pils), self.dim), dtype=np.float32)


def hue_triples(n):
    """n hue-bin triples, any two sharing at most one bin (greedy, deterministic)."""
    import itertools
    out = []
    for t in itertools.combinations(range(N_BINS), 3):
        if all(len(set(t) & set(u)) <= 1 for u in out):
            out.append(t)
            if len(out) == n:
                return out
    raise RuntimeError("not enough triples")


def grid(seed):
    """An 8 x 9 luminance grid whose horizontal neighbours differ by >= 60."""
    rng = np.random.default_rng(seed)
    levels = np.array([70, 130, 190, 250])
    g = np.zeros((8, 9), dtype=np.int64)
    for r in range(8):
        g[r, 0] = rng.choice(levels)
        for c in range(1, 9):
            g[r, c] = rng.choice([v for v in levels if abs(v - g[r, c - 1]) >= 60])
    return g


def render(g, hues):
    """RGB image: value from the block grid, hue from three horizontal bands."""
    band = np.repeat(np.repeat(np.array([0, 0, 0, 1, 1, 1, 2, 2])[:, None], 9, axis=1), BLOCK, axis=0)
    band = np.repeat(band, BLOCK, axis=1)
    H = np.array([int((h + 0.5) * 256 / N_BINS) for h in hues], dtype=np.uint8)[band]
    V = np.repeat(np.repeat(g, BLOCK, axis=0), BLOCK, axis=1).astype(np.uint8)
    S = np.full(V.shape, 150, dtype=np.uint8)
    return Image.merge("HSV", [Image.fromarray(H), Image.fromarray(S), Image.fromarray(V)]).convert("RGB")


def dh(img):
    from weed_optimizer_framework.tools.inc import common as C
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return C.dhash(buf)


def planted_near(L, g0, hues0, hues, want_bits, seed):
    """A block grid near g0 whose image (in `hues`) sits exactly want_bits
    (a set) dHash bits from the image of g0 in hues0, and > 6 bits under every
    variant."""
    h0 = dh(render(g0, hues0))
    rng = np.random.default_rng(seed)
    for _ in range(4000):
        g = g0.copy()
        for _k in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(8)), int(rng.integers(9))
            g[r, c] = int(rng.choice([70, 130, 190, 250]))
        img = render(g, hues)
        hv = L.dhash_variants(img)
        d = bin(hv["id"] ^ h0).count("1")
        if d in want_bits and min(bin(hv[v] ^ h0).count("1") for v in L.VARIANTS) > 6:
            return img, d
    raise RuntimeError("could not plant a near image")


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def mrow(path, key, source, session=""):
    from weed_optimizer_framework.tools.inc import common as C
    return {"image": str(path), "label": str(path) + ".txt", "sha256": C.sha256_file(path),
            "label_sha256": "0" * 64, "source": source, "session": session, "key": key}


def build_world(L, root):
    from weed_optimizer_framework.tools.inc import common as C
    triples = iter(hue_triples(80))
    img_dir = root / "images"
    ref_rows = []
    for i in range(20):
        g = grid(1000 + i)
        hues = next(triples)
        p = save(render(g, hues), img_dir / "ref" / ("r%02d.png" % i))
        ref_rows.append(dict(mrow(p, "core_%02d" % i, "cottonweeddet12/train", "S%d" % (i % 3)), _g=g, _h=hues))
    eval_rows, eval_meta = {}, {}
    sources = {"dev": "cottonweeddet12/valid", "test": "cottonweeddet12/test", "ood22": "3seasonweeddet10/data2022",
               "ood23": "3seasonweeddet10/data2023", "imageweeds": "project_agml__imageweeds_weed_detection"}
    for s, split in enumerate(sorted(sources)):
        eval_rows[split] = []
        for i in range(3):
            g = grid(2000 + 10 * s + i)
            hues = next(triples)
            p = save(render(g, hues), root / "eval" / split / ("e%d.png" % i))
            key = "%s_e%d" % (split, i)
            eval_rows[split].append(mrow(p, key, sources[split]))
            eval_meta[key] = (split, g, hues, p)
    pool = []
    # the provenance-disjoint negative group: 6 plain + 6 planted 7-10 bits from a reference image
    for i in range(6):
        p = save(render(grid(3000 + i), next(triples)), img_dir / "mh" / ("m%d.png" % i))
        pool.append(mrow(p, "mh_%d" % i, MH))
    for i in range(6):
        r = ref_rows[i]
        img, d = planted_near(L, r["_g"], r["_h"], next(triples), set(range(7, 11)), 4000 + i)
        pool.append(mrow(save(img, img_dir / "mh" / ("n%d.png" % i)), "mh_near_%d" % i, MH))
    # one augmented copy of an evaluation image per family, plus clean images
    ekeys = sorted(eval_meta)
    planted = {}
    for f, fam in enumerate(L.FAMILIES):
        key = ekeys[f % len(ekeys)]
        split, g, hues, p = eval_meta[key]
        with Image.open(p) as im:
            aug, params = L.augment(im.convert("RGB"), fam, np.random.default_rng(5000 + f),
                                    L.FAMILY_DEFAULTS[fam])
        path = save(aug, img_dir / "tuf" / ("copy_%s.png" % fam))
        pool.append(mrow(path, "tuf_copy_%s" % fam, TUF))
        planted["tuf_copy_%s" % fam] = (split, key)
    for i in range(2):
        pool.append(mrow(save(render(grid(6000 + i), next(triples)), img_dir / "tuf" / ("c%d.png" % i)),
                         "tuf_clean_%d" % i, TUF))
    for i in range(4):
        pool.append(mrow(save(render(grid(7000 + i), next(triples)), img_dir / "clean" / ("c%d.png" % i)),
                         "clean_%d" % i, CLEAN))
    split, g, hues, p = eval_meta["dev_e0"]
    near_img, near_bits = planted_near(L, g, hues, next(triples), {8}, 8000)
    pool.append(mrow(save(near_img, img_dir / "clean" / "near8.png"), "clean_near8", CLEAN))
    # a harvested base image that is a copy of a dev image (brightness), and clean ones
    split, g, hues, p = eval_meta["dev_e1"]
    with Image.open(p) as im:
        aug, _ = L.augment(im.convert("RGB"), "crop", np.random.default_rng(9000), L.FAMILY_DEFAULTS["crop"])
    pool.append(mrow(save(aug, img_dir / "cwp" / "copy_dev.png"), "cwp_copy_dev", CWP))
    for i in range(3):
        pool.append(mrow(save(render(grid(9100 + i), next(triples)), img_dir / "cwp" / ("c%d.png" % i)),
                         "cwp_%d" % i, CWP))
    pool.sort(key=lambda r: r["key"])
    by_key = {r["key"]: r for r in pool}
    extra = mrow(save(render(grid(9900), next(triples)), img_dir / "extra" / "x.png"), "extra_0", CWP)
    split, g, hues, p = eval_meta["test_e2"]
    with Image.open(p) as im:
        aug, _ = L.augment(im.convert("RGB"), "blur", np.random.default_rng(9500), L.FAMILY_DEFAULTS["blur"])
    extra_copy = mrow(save(aug, img_dir / "extra" / "copy_test.png"), "gone_copy_test", "rf_notinpool__x")
    base = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ref_rows] + \
        [by_key["cwp_copy_dev"], by_key["cwp_0"], by_key["cwp_1"], extra]
    ref_manifest = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ref_rows]
    # the experiment's manifests directory also holds its own copy of the base (same rows) and,
    # here, an earlier base that differs from the current one (a reference image fewer, and the
    # harvested dev copy only)
    increments = {"S1": [by_key["clean_0"], by_key["tuf_copy_jpeg"]], "S2": [by_key["clean_1"], by_key["cwp_2"]],
                  "S3": [extra_copy], "base_B": [dict(r) for r in base],
                  "base_prev": ref_manifest[1:] + [by_key["cwp_copy_dev"]]}
    C.write_manifest(C.manifest_path(REF), ref_manifest)
    per_slug = {TUF: {"kept": 8, "near_eval_by_split": {"ood23": 2}},
                CLEAN: {"kept": 5, "near_eval_by_split": {}},
                CWP: {"kept": 4, "near_eval_by_split": {"dev": 1, "test": 2}},
                MH: {"kept": 12, "near_eval_by_split": {}},
                "rf_gone__x": {"kept": 0, "near_eval_by_split": {"dev": 3}}}
    ps = C.INC_DIR / "step1" / "pool_summary.json"
    ps.parent.mkdir(parents=True, exist_ok=True)
    ps.write_text(json.dumps({"per_slug": per_slug}))

    ad = types.ModuleType("fake_leak_adapter")
    ad.__file__ = str(HERE)
    ad.pool_rows = lambda sources=None: [dict(r) for r in pool if sources is None or r["source"] in sources]
    ad.base_rows = lambda: [dict(r) for r in base]
    ad.increment_rows = lambda exp: {k: [dict(r) for r in v] for k, v in increments.items()}
    ad.eval_rows = lambda: {s: [dict(r) for r in v] for s, v in eval_rows.items()}
    return ad, {"planted": planted, "near_bits": near_bits, "eval_rows": eval_rows, "pool": pool,
                "by_key": by_key, "ps": ps, "eval_meta": eval_meta, "ref_rows": ref_manifest,
                "increments": increments}


# ------------------------------------------------------------ checks
def test_dhash(L):
    print("dHash variants")
    T = Image.Transpose
    ops = {"hflip": T.FLIP_LEFT_RIGHT, "vflip": T.FLIP_TOP_BOTTOM, "rot90": T.ROTATE_90, "rot180": T.ROTATE_180,
           "rot270": T.ROTATE_270, "transpose": T.TRANSPOSE, "transverse": T.TRANSVERSE}
    rng = np.random.default_rng(11)
    ok = True
    for t in range(60):
        w, h = int(rng.integers(9, 400)), int(rng.integers(9, 400))
        small = rng.integers(0, 256, size=(max(2, h // 20), max(2, w // 20), 3), dtype=np.uint8)
        im = Image.fromarray(small).resize((w, h), Image.BILINEAR)
        hv = L.dhash_variants(im)
        ok &= hv["id"] == dh(im)
        for v, op in ops.items():
            ok &= hv[v] == dh(im.transpose(op))
    check("every variant equals common.dhash of the PNG of that transform (60 images)", ok)
    im = render(grid(1), (0, 5, 10))
    check("a horizontally flipped image holds the original's dHash under hflip",
          L.dhash_variants(im.transpose(T.FLIP_LEFT_RIGHT))["hflip"] == L.dhash_variants(im)["id"])
    xs = np.random.default_rng(2).integers(0, 2 ** 63, size=200, dtype=np.int64).astype(np.uint64) * np.uint64(2) + \
        np.uint64(1)
    check("popcount64 equals Python's bit count",
          [int(v) for v in L.popcount64(xs)] == [bin(int(x)).count("1") for x in xs])


def test_augment(L, D):
    print("augmentations")
    im = render(grid(3), (1, 7, 13))
    ok = True
    for fam in L.FAMILIES:
        a1, p1 = L.augment(im, fam, np.random.default_rng(L.C.stable_int("x/%s" % fam)))
        a2, p2 = L.augment(im, fam, np.random.default_rng(L.C.stable_int("x/%s" % fam)))
        ok &= p1 == p2 and np.array_equal(np.asarray(a1), np.asarray(a2))
    check("each family is deterministic under its seed", ok)
    draws = {fam: [L.augment(im, fam, np.random.default_rng(s))[1] for s in range(40)] for fam in L.FAMILIES}
    check("crop removes 0-20 % of each side's extent",
          all(0 <= d["fx"] <= 0.2 and 0 <= d["fy"] <= 0.2 for d in draws["crop"]))
    check("brightness within +-25 %", all(0.75 <= d["factor"] <= 1.25 for d in draws["brightness"]))
    check("blur radius 1-2 px", all(1 <= d["radius"] <= 2 for d in draws["blur"]))
    check("shear within +-10 degrees, both axes",
          all(abs(d["degrees"]) <= 10 for d in draws["shear"]) and {d["axis"] for d in draws["shear"]} == {"x", "y"})
    check("jpeg quality 50-90", all(50 <= d["quality"] <= 90 for d in draws["jpeg"]))
    check("rot90 draws 90, 180 and 270", {d["angle"] for d in draws["rot90"]} == {90, 180, 270})
    check("flip draws both flips", {d["mode"] for d in draws["flip"]} == {"h", "v"})
    lb = L.augment(im, "letterbox640", np.random.default_rng(0))[0]
    check("letterbox to a 640 px square", lb.size == (640, 640))
    check("different seeds draw different parameters",
          len({json.dumps(d, sort_keys=True) for d in draws["crop"]}) > 30)
    weed = D.load("weed")
    fp = L.family_params(weed)
    check("weed.json's leak.families give the contract's ranges",
          fp["crop"]["frac"] == [0.0, 0.2] and fp["brightness"]["factor"] == [0.75, 1.25]
          and fp["shear"]["degrees"] == [-10.0, 10.0] and fp["blur"]["radius"] == [1.0, 2.0]
          and fp["jpeg"]["quality"] == [50, 90] and fp["letterbox640"]["size"] == 640, fp)
    bad = json.loads(json.dumps(weed.raw))
    bad["leak"]["families"]["crop"] = {"max_crop": 0.2}
    check("an unknown family key refuses",
          raises(lambda: L.family_params(bad), L.LeakError, "unknown key") is not None)
    bad2 = json.loads(json.dumps(weed.raw))
    del bad2["leak"]["families"]["jpeg"]
    check("a missing family refuses", raises(lambda: L.family_params(bad2), L.LeakError, "missing") is not None)
    fams = {"a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0], "b": list(np.linspace(0.5, 1.0, 40)) + [float("nan")]}
    th = L.threshold(fams, 0.95)
    check("threshold keeps >= 95 % of every family's positives",
          all(np.mean(np.where(np.isfinite(v), v, -1) >= th) >= 0.95 for v in map(np.array, fams.values())), th)


def contract_scope():
    """(the stated number of H6(a) sources, the §3.4 short names) from the contract text."""
    text = CONTRACT.read_text()
    m = re.search(r"§3\.4: (\d+) sources", text)
    start = text.index("Sources that kept images and have at least one near_eval hit")
    end = text.index("H6 covers all of them", start)
    names = re.findall(r"^- ([^()\n]+?) \(", text[start:end], flags=re.M)
    return int(m.group(1)), [n.strip() for n in names]


def test_scope(L):
    print("H6(a) scope on the real pool_summary.json")
    real = json.loads((REAL_INC / "step1" / "pool_summary.json").read_text())
    scope = L.h6a_scope(real)
    n, names = contract_scope()
    check("the scope has the number of sources the contract states (§3.4: %d)" % n, len(scope) == n,
          (len(scope), scope))
    match = {nm: [s for s in scope if all(w in s for w in nm.split())] for nm in names}
    check("every source §3.4 lists matches exactly one scope slug, and all differ",
          len(names) == n and all(len(v) == 1 for v in match.values())
          and len({v[0] for v in match.values()}) == n, match)
    per = real["per_slug"]
    check("the scope is the rule (kept > 0 and a near_eval hit), not a list",
          all(per[s]["kept"] > 0 and sum(per[s]["near_eval_by_split"].values()) > 0 for s in scope)
          and not [s for s, v in per.items() if s not in scope and v.get("kept", 0) > 0
                   and sum((v.get("near_eval_by_split") or {}).values()) > 0])


def test_run(L, D):
    print("run on the synthetic world")
    from weed_optimizer_framework.tools.inc import common as C
    dom = D.load("weed")
    pre = D.load_prereg(TMP / "inc" / "funnel" / "prereg_v1.json")
    ad, w = build_world(L, TMP / "world")
    check("the planted near image sits exactly 8 bits from its evaluation image", w["near_bits"] == 8)
    fdir = TMP / "inc" / "funnel"
    emb = HueEmbedder()
    doc = L.run(pre, dom, fdir, ad, embedder=emb, procs=1, testing=True)
    cal = doc["calibration"]
    check("the calibration passes before anything is scanned", cal["ok"] and doc["status"] == "complete",
          cal.get("why"))
    check("recall >= 0.95 in every family, with the seed texts pinned",
          all(v["recall"] >= 0.95 and v["n"] == 20 and v["params_seed"] == "funnel/v1/leak/pos/%s" % f
              for f, v in cal["positives"].items()) and set(cal["positives"]) == set(L.FAMILIES))
    neg = cal["negatives"]
    check("both negative sets are measured, and the false-positive rate is <= 1 %",
          neg["pairs_7_10"]["n"] >= 6 and neg["hard"]["n"] == 20 and neg["pairs_7_10"]["fpr"] <= 0.01
          and neg["hard"]["fpr"] <= 0.01, neg)
    check("the threshold is recorded in the detector", doc["detector"]["cos_threshold"] == cal["cos_threshold"]
          and 0 < cal["cos_threshold"] <= 1 and doc["detector"]["dhash_bits_max"] == 6)
    tuf = doc["scans"]["source:%s" % TUF]
    found = {e["key"]: e["eval_key"] for e in tuf["listed"]}
    check("every planted copy (one per augmentation family) is found, with its evaluation image",
          all(found.get(k) == v[1] for k, v in w["planted"].items()), (found, w["planted"]))
    check("the clean images of the copying source are not listed",
          not [k for k in found if k.startswith("tuf_clean")])
    clean = doc["scans"]["source:%s" % CLEAN]
    check("the distinct image 8 bits from an evaluation image is not a copy", not clean["copy_found"], clean)
    h6a = doc["h6a"]
    check("H6(a): the scope is derived from pool_summary.json",
          h6a["scope"] == sorted([TUF, CWP]), h6a["scope"])
    check("H6(a): every source with a copy is quarantined as a whole, in the pool or not",
          h6a["quarantine"] == sorted([TUF, CWP, "rf_notinpool__x"]) and h6a["copy_found"] == {TUF: True, CWP: True},
          h6a["quarantine"])
    h6b = doc["h6b"]
    check("H6(b): the base copy and the increment copy are an incident",
          h6b["base_copy"] and h6b["increment_copies"] == {"realloop_v1:S1": 1, "realloop_v1:S2": 0,
                                                           "realloop_v1:S3": 1}
          and h6b["incident"] and h6b["cleared"] and "dev" in h6b["splits_hit"], h6b)
    check("H6(b): the experiment's own copy of the base is not an increment (same rows: skipped)",
          h6b["same_as_base"] == ["realloop_v1:base_B"] and "realloop_v1:base_B" not in doc["scans"]
          and "realloop_v1:base_B" not in h6b["increment_copies"], h6b)
    prev = doc["scans"].get("realloop_v1:base_prev") or {}
    check("H6(b): an earlier base of the experiment is scanned as a base, its harvested images only",
          h6b["base_scans"] == ["base_B", "realloop_v1:base_prev"] and prev.get("images") == 1
          and prev.get("copy_found") is True and "realloop_v1:base_prev" not in h6b["increment_copies"], (h6b, prev))
    check("H6(a): a copy of any evaluation split, dev included, voids the ood numbers of arms holding the source",
          h6a["void_ood_arms_with"] == h6a["quarantine"] and CWP in h6a["void_ood_arms_with"]
          and "rf_notinpool__x" in h6a["void_ood_arms_with"]
          and doc["scans"]["source:%s" % CWP]["splits_hit"] == ["dev"], h6a)
    check("the base image that is not a pool image is described apart and scanned",
          doc["scans"]["base_B"]["images"] == 4 and doc["scans"]["base_B"]["unscanned"] == 0
          and "extra" in doc["descriptor_files"])
    h6c = doc["h6c"]
    tuf_splits = set(tuf["splits_hit"])
    check("H6(c): exam labs from the exam rows' datasets; an exam no group names is unmapped",
          h6c["exam_labs"]["dev"] == "LuLab" and h6c["exam_labs"]["test"] == "LuLab"
          and h6c["exam_labs"]["imageweeds"] == "NDSU" and h6c["exam_labs"]["ood22"] is None
          and h6c["unmapped_exams"] == ["ood22", "ood23"], h6c["exam_labs"])
    check("H6(c): the base-B source copying dev is in the reference lab group",
          h6c["added_by_h6a"].get(CWP) == "LuLab" and CWP in h6c["groups"]["LuLab"], h6c)
    want_tuf = sorted({h6c["exam_labs"][s] for s in tuf_splits if h6c["exam_labs"][s]})
    check("H6(c): the copying pool source joins the groups of the splits it copies",
          (h6c["added_by_h6a"].get(TUF) == want_tuf[0]) if want_tuf else TUF not in h6c["added_by_h6a"],
          (tuf_splits, h6c["added_by_h6a"]))
    text = (fdir / "leak_v1.json").read_text()
    eval_paths = [r["image"] for rows in w["eval_rows"].values() for r in rows]
    check("leak_v1.json names no evaluation image path",
          not [p for p in eval_paths if p in text] and str(TMP / "world" / "eval") not in text)
    longest = []

    def walk(o):
        if isinstance(o, list):
            if o and all(isinstance(x, (int, float)) for x in o):
                longest.append(len(o))
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
    walk(json.loads(text))
    check("leak_v1.json holds no pixel or descriptor array", max(longest or [0]) <= 8, max(longest or [0]))
    check("the header records the prereg core, the code and the inputs",
          doc["prereg"]["core_sha256"] == pre.core_sha256 and "funnel/leak.py" in doc["code"]
          and set(doc["inputs"]) == {"pool_summary", "reference_manifest"})
    with open(fdir / "leak_pairs_v1.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    kinds = {r["kind"] for r in rows}
    check("the pairs file lists the copies and the calibration negatives",
          {"copy", "negative_7_10", "negative_hard"} <= kinds
          and len([r for r in rows if r["kind"] == "copy"]) >= len(w["planted"]) + 2
          and doc["pairs_csv"]["sha256"] == C.sha256_file(fdir / "leak_pairs_v1.csv"))
    check("the evaluation descriptors are recorded by path and sha256 (the file stays on the cluster)",
          doc["eval_descriptors"]["path"].endswith("leak_eval_desc.npz")
          and doc["eval_descriptors"]["sha256"] == C.sha256_file(fdir / "leak_eval_desc.npz"))

    print("reruns")
    e2 = HueEmbedder()
    doc2 = L.run(pre, dom, fdir, ad, embedder=e2, procs=1, testing=True)
    check("a rerun on the same inputs is a no-op", e2.calls == 0 and doc2 == json.loads(text))
    pool_rows_saved = ad.pool_rows
    ad.pool_rows = lambda sources=None: [r for r in pool_rows_saved(sources) if r["key"] != "clean_1"]
    check("changed inputs refuse without force",
          raises(lambda: L.run(pre, dom, fdir, ad, embedder=HueEmbedder(), procs=1, testing=True), L.LeakError,
                 "--force") is not None)
    ad.pool_rows = pool_rows_saved
    pre_path = TMP / "inc" / "funnel" / "prereg_v1.json"
    D.append_amendment(pre_path, {"id": "T-lock", "kind": "sample_lock", "date": "2026-09-28"})
    pre_locked = D.load_prereg(pre_path)
    e3 = HueEmbedder()
    try:
        doc_locked, lock_err = L.run(pre_locked, dom, fdir, ad, embedder=e3, procs=1, testing=True), None
    except L.LeakError as e:
        doc_locked, lock_err = None, str(e)
    check("after a sample-lock amendment a rerun is still a no-op (the prereg core is unchanged)",
          lock_err is None and e3.calls == 0 and doc_locked == json.loads(text)
          and pre_locked.core_sha256 == pre.core_sha256 and pre_locked.sha256 != pre.sha256, lock_err)

    print("detect")
    eval_desc = fdir / "leak_eval_desc.npz"
    desc_sha = C.sha256_file(eval_desc)
    index = L.eval_index(ad, HueEmbedder(), eval_desc)
    by_key = w["by_key"]
    got = L.detect([{"key": "q1", "path": by_key["tuf_copy_shear"]["image"]},
                    {"key": "q2", "path": by_key["clean_2"]["image"]}], index, doc)
    check("detect finds a planted copy on disk and clears a clean image",
          [e["key"] for e in got] == ["q1"] and got[0]["eval_key"] == w["planted"]["tuf_copy_shear"][1], got)
    # the dHash half of the rule on its own: descriptors that match nothing (cosine <= 0),
    # hashes of a 90-degree rotation of an evaluation image, and of an unrelated image
    _split, _g, _h, e_path = w["eval_meta"]["ood23_e1"]
    with Image.open(e_path) as im:
        rot = im.convert("RGB").transpose(Image.Transpose.ROTATE_90)
    with Image.open(by_key["clean_3"]["image"]) as im:
        unrelated = im.convert("RGB")
    far = -np.ones(HueEmbedder.dim, dtype=np.float32)
    got_h = L.detect([{"key": "rot", "desc": far, "hashes": [L.dhash_variants(rot)[v] for v in L.VARIANTS]},
                      {"key": "other", "desc": far, "hashes": [L.dhash_variants(unrelated)[v] for v in L.VARIANTS]}],
                     index, doc)
    check("a rotated copy the descriptor misses is found by the dHash of its variants",
          [e["key"] for e in got_h] == ["rot"] and got_h[0]["eval_key"] == "ood23_e1" and got_h[0]["bits"] == 0
          and got_h[0]["variant"] != "id" and got_h[0]["cos"] < doc["detector"]["cos_threshold"], got_h)
    good_h = [L.dhash_variants(unrelated)[v] for v in L.VARIANTS]
    check("detect refuses a given descriptor that is not finite, or of the wrong size (never a silent clear)",
          raises(lambda: L.detect([{"key": "nan", "desc": np.full(HueEmbedder.dim, np.nan), "hashes": good_h}],
                                  index, doc), L.LeakError, "not usable") is not None
          and raises(lambda: L.detect([{"key": "short", "desc": np.ones(3), "hashes": good_h}], index, doc),
                     L.LeakError, "not usable") is not None
          and raises(lambda: L.detect([{"key": "few", "desc": far, "hashes": good_h[:2]}], index, doc),
                     L.LeakError, "not usable") is not None)
    other_emb = HueEmbedder(name="another/model:cls")
    idx_other = L.eval_index(ad, other_emb, None)
    check("detect refuses an evaluation index made by another embedder than the calibration's",
          raises(lambda: L.detect([{"key": "q2", "path": by_key["clean_2"]["image"]}], idx_other, doc), L.LeakError,
                 "calibrated with") is not None)
    doc_x = dict(doc, eval_descriptors=dict(doc["eval_descriptors"], sha256="0" * 64))
    check("detect refuses evaluation descriptors other than the ones leak_v1.json records",
          raises(lambda: L.detect([{"key": "q2", "path": by_key["clean_2"]["image"]}], index, doc_x), L.LeakError,
                 "records") is not None)
    check("a consumer's eval_index refuses to overwrite the recorded descriptor file with another embedder's",
          raises(lambda: L.eval_index(ad, other_emb, eval_desc), L.LeakError, "rather than overwrite") is not None
          and C.sha256_file(eval_desc) == desc_sha and doc["eval_descriptors"]["sha256"] == desc_sha)
    ad_bad = types.ModuleType("fake_leak_adapter_bad_eval")
    er_bad = {s: [dict(r) for r in rows] for s, rows in w["eval_rows"].items()}
    er_bad["test"][0]["image"] = str(TMP / "no_such_eval.png")
    ad_bad.eval_rows = lambda: er_bad
    check("eval_index refuses an evaluation image it cannot describe",
          raises(lambda: L.eval_index(ad_bad, HueEmbedder(), None), L.LeakError, "cannot be described") is not None)
    check("detect refuses an image it cannot read",
          raises(lambda: L.detect([{"key": "x", "path": str(TMP / "none.png")}], index, doc), L.LeakError,
                 "cannot describe") is not None)
    failed = dict(doc, calibration=dict(doc["calibration"], ok=False))
    check("detect refuses under a failed calibration",
          raises(lambda: L.detect([{"key": "q1", "path": by_key["clean_2"]["image"]}], index, failed),
                 L.LeakCalibrationError) is not None)

    print("a calibration that fails")
    fdir2 = TMP / "inc" / "funnel_fail"
    msg = raises(lambda: L.run(pre, dom, fdir2, ad, embedder=ConstEmbedder(), procs=1, testing=True),
                 L.LeakCalibrationError, "false-positive rate")
    doc3 = json.loads((fdir2 / "leak_v1.json").read_text()) if (fdir2 / "leak_v1.json").exists() else {}
    check("negatives above 1 % false positives refuse (LeakCalibrationError)", msg is not None, msg)
    check("... after writing leak_v1.json with ok false and no scan",
          doc3.get("calibration", {}).get("ok") is False and doc3.get("scans") == {}
          and doc3.get("status") == "calibration_failed" and doc3.get("h6b") is None)
    check("... and a rerun refuses again",
          raises(lambda: L.run(pre, dom, fdir2, ad, embedder=ConstEmbedder(), procs=1, testing=True),
                 L.LeakCalibrationError) is not None)

    print("an empty negative set")
    ref_rows, _rec = L.reference_rows(dom)
    no_near = [r for r in w["pool"] if not r["key"].startswith("mh_near_")]
    cal0 = L.calibrate(ad, dom, HueEmbedder(), store=L._Store(HueEmbedder(), None), ref_rows=ref_rows,
                       pool_rows=no_near)
    check("with no 7-10-bit negative pair the calibration fails (an unmeasured rate is no pass)",
          cal0["negatives"]["pairs_7_10"]["n"] == 0 and cal0["ok"] is False
          and "negative set pairs_7_10 is empty" in cal0["why"], cal0.get("why"))

    print("an image that cannot be scanned")
    ad4 = types.ModuleType("fake_leak_adapter_unreadable")
    for fn in ("pool_rows", "base_rows", "eval_rows"):
        setattr(ad4, fn, getattr(ad, fn))
    ad4.__file__ = ad.__file__
    gone = {"image": str(TMP / "no_such_increment.png"), "label": "x.txt", "sha256": "f" * 64,
            "label_sha256": "0" * 64, "source": CLEAN, "session": "", "key": "gone_row"}
    inc4 = dict(w["increments"], S4=[gone])
    ad4.increment_rows = lambda exp: {k: [dict(r) for r in v] for k, v in inc4.items()}
    doc4 = L.run(pre, dom, TMP / "inc" / "funnel_unscanned", ad4, embedder=HueEmbedder(), procs=1, testing=True)
    s4 = doc4["scans"]["realloop_v1:S4"]
    check("a row that cannot be described is counted unscanned, and the base and increments are not cleared",
          s4["images"] == 1 and s4["unscanned"] == 1 and not s4["copy_found"] and doc4["h6b"]["cleared"] is False
          and doc4["h6b"]["increment_copies"]["realloop_v1:S4"] == 0, (s4, doc4["h6b"]))


def _estimate_stand_in():
    """leak.py takes its interval bounds from funnel/estimate.py (G-stats). When
    that module is not in the tree yet, a test-only Wilson interval stands in,
    and the run says so; the real module is always preferred."""
    try:
        from weed_optimizer_framework.tools.funnel import estimate  # noqa: F401
        return False
    except ImportError:
        pass
    import math
    import weed_optimizer_framework.tools.funnel as F

    def binom_interval(k, n, conf=0.95):
        if n == 0:
            return 0.0, 1.0
        z, p = 1.959963984540054, k / float(n)
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, c - h), min(1.0, c + h)
    mod = types.ModuleType("weed_optimizer_framework.tools.funnel.estimate")
    mod.binom_interval = binom_interval
    sys.modules[mod.__name__] = mod
    F.estimate = mod
    print("NOTE: funnel/estimate.py is not in the tree; a test-only Wilson interval stands in for binom_interval")
    return True


def main():
    if np is None:
        skip("all", "numpy is not installed")
        return
    _estimate_stand_in()
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import leak as L
    from weed_optimizer_framework.tools.inc import common as C
    L.C = C
    test_scope(L)
    if Image is None:
        skip("dhash, augmentations and run", "PIL is not installed")
        return
    test_dhash(L)
    test_augment(L, D)
    test_run(L, D)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
