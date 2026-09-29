#!/usr/bin/env python3
"""inc2/guard.py: GuardV2, the copy guard every v2 entry point calls
(docs/CONTINUOUS_LOOP.md §3.2 "Guard", §4.2, §8 "Never-train v2").

The synthetic world is the funnel copy-detector test's: "photographs" made of
a 9 x 8 grid of luminance blocks (so dHash distances can be planted exactly)
coloured with three hue bands, a distinct hue triple per photograph (any two
share at most one hue bin). The fake descriptor is a hue histogram, which
survives every re-export augmentation; the dHash is the project's real one.

Pinned:
  * hashing: dhash_variants equals funnel.leak.dhash_variants and its "id"
    equals inc.common.dhash (the never-train index's hash) on every image;
    VARIANTS equals funnel.leak.VARIANTS; an unreadable file is None;
    variant_list accepts a dict or an 8-sequence and refuses an incomplete one;
  * every planted copy kind is caught, with its reason, in the contract's
    order (first hit wins): an exact and a 6-bit copy of an evaluation image
    (near_eval_v2), a horizontal flip and a 90-degree rotation (each more than
    6 bits away as stored: near_eval_variant, naming the variant), a copy and
    a flipped copy of a base image (base_copy), an image in the seen index
    (exact_dup), a 2-bit neighbour of an intake image (near_dup_intake; a
    5-bit one passes, 3-bit radius); a 7-10-bit neighbour of an evaluation
    image with other colours passes; an image both near evaluation and seen
    is near_eval_v2;
  * fail closed: no dHash, no variants, seven variants, or variants whose id
    differs from the dHash are unhashable; counts are kept per reason;
  * GuardV2.load reads the indexes LOCK v2 records and refuses a changed
    index, an incomplete one, a LOCK that is not v2, an index holding a split
    the LOCK does not list as evaluation, and an entry count that differs from
    the LOCK's; v2_eval_rows refuses a changed evaluation manifest;
  * the embedding check (near_eval_embed; cleared rows counted as embed_pass): the stream's own calibration
    (seed prefix stream/v1/leak) passes on this world, is written under the
    caller's directory, and a rerun is a no-op; a 15 % crop, a shear and a
    brightened crop of test images, each more than 6 bits from the original
    under all 8 variants, are refused as near_eval_embed; clean images pass;
    an unreadable image is unhashable (fail closed); a collapsing embedder
    fails the calibration, whose file then says ok false and cannot be
    loaded; load_calibration accepts a passed funnel leak_v1.json and refuses
    another embedder's, one naming no embedder, and a failed one; a scanner
    whose index uses another embedder than the calibration refuses;
  * the no-train import rule: guard.py never names mega_trainer (it hashes
    through inc.common.dhash), and ultralytics is not imported by a guard
    check;
  * guard.py is domain-free: no species, dataset or exam name in its text.

No network, no GPU.

Run:  python3 tests/test_inc2_guard.py
"""
import ast
import itertools
import json
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_guard_"))
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        if text and text not in str(e):
            print("       raised without %r: %s" % (text, e))
            return False
        return True
    return False


try:
    import numpy as np
    from PIL import Image
except ImportError as e:                       # optional heavy deps: skip with the reason
    print("SKIP: every check (numpy / PIL not importable: %s)" % e)
    sys.exit(0)

from weed_optimizer_framework.tools.inc import common as C1  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import guard as G  # noqa: E402
from weed_optimizer_framework.tools.funnel import leak as L  # noqa: E402

N_BINS = 48
BLOCK = 12
DOMAIN = ROOT / "weed_optimizer_framework" / "tools" / "funnel" / "domains" / "weed.json"
MH = "src_negative_group"


# ------------------------------------------------------------------- world
class HueEmbedder:
    """Histogram of hue over saturated, lit pixels, L2-normalised."""
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
            n = np.linalg.norm(h)
            out.append(h / n if n > 0 else h)
        return np.stack(out)


class ConstEmbedder(HueEmbedder):
    def __call__(self, pils):
        self.calls += 1
        return np.ones((len(pils), self.dim), dtype=np.float32)


def hue_triples(n):
    out = []
    for t in itertools.combinations(range(N_BINS), 3):
        if all(len(set(t) & set(u)) <= 1 for u in out):
            out.append(t)
            if len(out) == n:
                return out
    raise RuntimeError("not enough triples")


TRIPLES = iter(hue_triples(200))


def grid(seed):
    rng = np.random.default_rng(seed)
    levels = np.array([70, 130, 190, 250])
    g = np.zeros((8, 9), dtype=np.int64)
    for r in range(8):
        g[r, 0] = rng.choice(levels)
        for c in range(1, 9):
            g[r, c] = rng.choice([v for v in levels if abs(v - g[r, c - 1]) >= 60])
    return g


def render(g, hues):
    band = np.repeat(np.repeat(np.array([0, 0, 0, 1, 1, 1, 2, 2])[:, None], 9, axis=1), BLOCK, axis=0)
    band = np.repeat(band, BLOCK, axis=1)
    H = np.array([int((h + 0.5) * 256 / N_BINS) for h in hues], dtype=np.uint8)[band]
    V = np.repeat(np.repeat(g, BLOCK, axis=0), BLOCK, axis=1).astype(np.uint8)
    S = np.full(V.shape, 150, dtype=np.uint8)
    return Image.merge("HSV", [Image.fromarray(H), Image.fromarray(S), Image.fromarray(V)]).convert("RGB")


def save(img, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


def mrow(path, key, source):
    return {"image": str(path), "label": str(path) + ".txt", "sha256": C1.sha256_file(path),
            "label_sha256": "0" * 64, "source": source, "session": "", "key": key}


def neighbour(g0, hues0, hues, want, seed, all_far=False):
    """(image, bits): a grid near g0 whose id dHash is `want` (a set) bits from
    g0's image; with all_far, every variant is also > 6 bits from it."""
    h0 = L.dhash_variants(render(g0, hues0))["id"]
    rng = np.random.default_rng(seed)
    for _ in range(6000):
        g = g0.copy()
        for _k in range(int(rng.integers(1, 7))):
            g[int(rng.integers(8)), int(rng.integers(9))] = int(rng.choice([70, 130, 190, 250]))
        img = render(g, hues)
        hv = L.dhash_variants(img)
        d = bits(hv["id"], h0)
        if d in want and (not all_far or min(bits(hv[v], h0) for v in L.VARIANTS) > 6):
            return img, d
    raise RuntimeError("could not plant a neighbour")


def far_augment(src, family, seed, params=None):
    """An augmented copy of the image at src whose 8 variants are all > 6 bits
    from the original's dHash (so only the embedding detector can find it)."""
    with Image.open(src) as im:
        base = im.convert("RGB")
    h0 = L.dhash_variants(base)["id"]
    for s in range(seed, seed + 400):
        img, used = L.augment(base, family, np.random.default_rng(s), params)
        hv = L.dhash_variants(img)
        if min(bits(hv[v], h0) for v in L.VARIANTS) > 6:
            return img, used
    raise RuntimeError("no far %s copy" % family)


class World:
    def __init__(self):
        root = TMP / "w"
        self.eval_rows = {"dev": [], "test": [], "imageweeds": []}
        self.meta = {}
        for s, split in enumerate(("dev", "test", "imageweeds")):
            for i in range(4):
                g, h = grid(200 + 10 * s + i), next(TRIPLES)
                p = save(render(g, h), root / "eval" / split / ("e%d.png" % i))
                key = "%s__e%d" % (split, i)
                self.eval_rows[split].append(mrow(p, key, split))
                self.meta[key] = (g, h, p)
        self.ref_rows = []
        for i in range(24):
            g, h = grid(1000 + i), next(TRIPLES)
            p = save(render(g, h), root / "ref" / ("r%02d.png" % i))
            self.ref_rows.append(dict(mrow(p, "train_core__r%02d" % i, "ref"), _g=g, _h=h))
        self.pool_rows = []
        for i in range(6):
            p = save(render(grid(3000 + i), next(TRIPLES)), root / "neg" / ("m%d.png" % i))
            self.pool_rows.append(mrow(p, "neg_%d" % i, MH))
        for i in range(6):
            r = self.ref_rows[i]
            img, _d = neighbour(r["_g"], r["_h"], next(TRIPLES), set(range(7, 11)), 4000 + i, all_far=True)
            self.pool_rows.append(mrow(save(img, root / "neg" / ("n%d.png" % i)), "neg_near_%d" % i, MH))
        self.ref_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.ref_rows]
        self.root = root

    def entries(self):
        out = []
        for split, rows in self.eval_rows.items():
            for r in rows:
                out.append([C1.dhash(r["image"]), split, r["key"]])
        return out


def domain_raw():
    raw = json.loads(DOMAIN.read_text())
    raw = json.loads(json.dumps(raw))
    raw["leak"]["negative_source_pairs"] = [["train_core", MH]]
    raw["sources"]["reference"] = "train_core"
    return raw


# ------------------------------------------------------------------ checks
def test_hashing(w):
    print("hashing")
    check("VARIANTS equals funnel.leak.VARIANTS", G.VARIANTS == L.VARIANTS)
    ok = True
    paths = [r["image"] for rows in w.eval_rows.values() for r in rows] + [r["image"] for r in w.ref_rows]
    rng = np.random.default_rng(3)
    for i in range(20):
        a = rng.integers(0, 256, size=(int(rng.integers(3, 30)), int(rng.integers(3, 30)), 3), dtype=np.uint8)
        paths.append(str(save(Image.fromarray(a).resize((int(rng.integers(9, 300)), int(rng.integers(9, 300)))),
                              TMP / "rand" / ("x%d.jpg" % i))))
    for p in paths:
        h, v = G.image_hashes(p)
        with Image.open(p) as im:
            want = {k: int(x) for k, x in L.dhash_variants(im).items()}
        ok &= h == C1.dhash(p) and v == want and v["id"] == h
    check("image_hashes: inc.common.dhash, and variants equal to funnel.leak's with id == dHash (%d images)"
          % len(paths), ok)
    bad = TMP / "rand" / "broken.png"
    bad.write_bytes(b"not an image")
    check("an unreadable file hashes to (None, None)", G.image_hashes(bad) == (None, None)
          and G.dhash_variants(bad) is None)
    v = G.dhash_variants(paths[0])
    check("variant_list takes a dict or an 8-sequence in VARIANTS order",
          G.variant_list(v) == [(n, v[n]) for n in G.VARIANTS]
          and G.variant_list([v[n] for n in G.VARIANTS]) == G.variant_list(v))
    v7 = dict(v)
    del v7["transverse"]
    check("... and refuses seven, a None, or a value outside 64 bits",
          G.variant_list(v7) is None and G.variant_list([1] * 7) is None
          and G.variant_list([1] * 7 + [None]) is None and G.variant_list([2 ** 64] + [1] * 7) is None)


def test_check(w):
    print("GuardV2.check: every planted copy kind")
    guard = G.GuardV2(w.entries(), [[C1.dhash(r["image"]), "train_core", r["key"]] for r in w.ref_rows[:12]])
    d = w.root / "planted"
    d.mkdir(parents=True, exist_ok=True)
    T = Image.Transpose
    g, h, p = w.meta["test__e0"]
    exact = d / "exact.png"
    shutil.copyfile(p, exact)
    near_img, near_bits = neighbour(g, h, h, {3, 4, 5, 6}, 11)
    near = save(near_img, d / "near.png")
    with Image.open(p) as im:
        hflip = save(im.transpose(T.FLIP_LEFT_RIGHT), d / "hflip.png")
    g1, h1, p1 = w.meta["dev__e1"]
    with Image.open(p1) as im:
        rot = save(im.transpose(T.ROTATE_90), d / "rot90.png")
    far_img, far_bits = neighbour(g, h, next(TRIPLES), set(range(7, 11)), 12, all_far=True)
    far = save(far_img, d / "far.png")
    r0 = w.ref_rows[3]
    base_copy = d / "base_copy.png"
    shutil.copyfile(r0["image"], base_copy)
    with Image.open(w.ref_rows[4]["image"]) as im:
        base_flip = save(im.transpose(T.FLIP_TOP_BOTTOM), d / "base_vflip.png")
    fresh = save(render(grid(9000), next(TRIPLES)), d / "fresh.png")
    g2, hues2 = grid(9001), next(TRIPLES)
    fresh2 = save(render(g2, hues2), d / "fresh2.png")
    got = {}
    for name, path in (("exact", exact), ("near", near), ("hflip", hflip), ("rot", rot), ("far", far),
                       ("base_copy", base_copy), ("base_flip", base_flip), ("fresh", fresh)):
        got[name] = guard.check_path(path)
    check("an exact copy of a test image: near_eval_v2, 0 bits",
          got["exact"][:2] == ("near_eval_v2", {"split": "test", "key": "test__e0", "bits": 0, "variant": "id"}),
          got["exact"][:2])
    check("a %d-bit copy: near_eval_v2 at %d bits" % (near_bits, near_bits),
          got["near"][0] == "near_eval_v2" and got["near"][1]["bits"] == near_bits, got["near"][:2])
    hv = G.dhash_variants(hflip)
    check("a horizontal flip (%d bits as stored): near_eval_variant, variant hflip"
          % bits(hv["id"], C1.dhash(p)),
          bits(hv["id"], C1.dhash(p)) > 6 and got["hflip"][0] == "near_eval_variant"
          and got["hflip"][1]["variant"] == "hflip" and got["hflip"][1]["key"] == "test__e0", got["hflip"][:2])
    check("a 90-degree rotation of a dev image: near_eval_variant, a rotation variant",
          got["rot"][0] == "near_eval_variant" and got["rot"][1]["variant"] in ("rot90", "rot270")
          and got["rot"][1]["key"] == "dev__e1", got["rot"][:2])
    check("a %d-bit neighbour with other colours passes (> 6 under every variant)" % far_bits,
          got["far"][:2] == (None, None), got["far"][:2])
    check("a copy of a base image: base_copy", got["base_copy"][0] == "base_copy"
          and got["base_copy"][1]["key"] == r0["key"] and got["base_copy"][1]["variant"] == "id", got["base_copy"][:2])
    check("a flipped copy of a base image: base_copy, variant vflip", got["base_flip"][0] == "base_copy"
          and got["base_flip"][1]["variant"] == "vflip", got["base_flip"][:2])
    check("a fresh image passes", got["fresh"][:2] == (None, None))
    hf, vf = got["fresh"][2]
    guard.add_seen(hf, "step1:seen_a")
    check("the same image once in the seen index: exact_dup",
          guard.check(hf, vf) == ("exact_dup", {"owner": "step1:seen_a", "bits": 0}))
    h2, _v2 = G.image_hashes(fresh2)
    guard.add_intake(h2, "intake/b0001:x")
    img2, b2 = neighbour(g2, hues2, hues2, {1, 2}, 31)
    img5, b5 = neighbour(g2, hues2, hues2, {5}, 32)
    r2 = guard.check(*G.image_hashes(save(img2, d / "intake2.png")))
    r5 = guard.check(*G.image_hashes(save(img5, d / "intake5.png")))
    check("a %d-bit neighbour of an intake image: near_dup_intake" % b2,
          r2 == ("near_dup_intake", {"owner": "intake/b0001:x", "bits": b2}), r2)
    check("a 5-bit neighbour passes the 3-bit intake radius", r5 == (None, None), r5)
    he, ve = G.image_hashes(exact)
    guard.add_seen(he, "step1:seen_eval")
    check("near evaluation and seen: near_eval_v2 (the first check wins)", guard.check(he, ve)[0] == "near_eval_v2")
    check("no dHash: unhashable", guard.check(None, ve)[0] == "unhashable")
    check("no variants: unhashable", guard.check(he, None)[0] == "unhashable")
    v7 = dict(ve)
    del v7["rot90"]
    check("seven variants: unhashable", guard.check(he, v7)[0] == "unhashable")
    check("variants of other pixels (id != dHash): unhashable",
          guard.check(he, G.dhash_variants(fresh))[1]["why"].startswith("the variants' id"))
    check("counts per reason are kept",
          guard.counts["near_eval_v2"] >= 3 and guard.counts["unhashable"] == 4 and guard.counts["pass"] >= 3
          and guard.counts["exact_dup"] == 1 and guard.counts["near_dup_intake"] == 1, dict(guard.counts))
    rec = guard.index_record()
    check("index_record: sizes and radii", rec["eval_entries"] == 12 and rec["base_entries"] == 12
          and rec["seen"] == 2 and rec["intake"] == 1 and rec["eval_bits"] == 6 and rec["intake_bits"] == 3, rec)
    return guard


def write_lock_dir(w, d, base_entries, complete=True, eval_splits=("dev", "test", "imageweeds"), omit=()):
    """A LOCK v2 directory; `omit` drops those (split, key) entries (or every
    entry of a split named alone) from the never-train index, the LOCK's
    counts following the index."""
    d.mkdir(parents=True, exist_ok=True)
    ents = [e for e in w.entries() if e[1] not in omit and (e[1], e[2]) not in omit]
    nt = {"entries": ents, "bits": 6, "complete": complete, "min_expected": len(ents) + (0 if complete else 1)}
    bc = {"entries": base_entries, "bits": 6, "complete": True, "min_expected": len(base_entries)}
    nt_sha = C2.write_json_atomic(d / "nevertrain_dhash.json", nt)
    bc_sha = C2.write_json_atomic(d / "base_copies_dhash.json", bc)
    mans = {}
    for split, rows in w.eval_rows.items():
        mans[split] = C1.write_manifest(d / ("%s.jsonl" % split), rows)
    lock = {"splits_version": "v2", "manifests": mans, "eval_splits": list(eval_splits),
            "nevertrain_sha256": nt_sha, "nevertrain_entries": len(ents),
            "base_copies_sha256": bc_sha, "base_copies_entries": len(base_entries)}
    C2.write_json_atomic(d / "LOCK.json", lock)
    return lock


def test_load(w):
    print("GuardV2.load")
    base = [[C1.dhash(r["image"]), "train_core", r["key"]] for r in w.ref_rows]
    d = TMP / "lockdir"
    write_lock_dir(w, d, base)
    g = G.GuardV2.load(d / "LOCK.json")
    check("loads the indexes LOCK v2 records", g.n_eval == 12 and g.n_base == 24
          and g.record["nevertrain_sha256"] == C1.sha256_file(d / "nevertrain_dhash.json"))
    check("the loaded guard refuses a test image and a base image",
          g.check_path(w.eval_rows["test"][1]["image"])[0] == "near_eval_v2"
          and g.check_path(w.ref_rows[20]["image"])[0] == "base_copy")
    rows = G.v2_eval_rows(d / "LOCK.json")
    check("v2_eval_rows reads the LOCK's evaluation manifests", sorted(rows) == ["dev", "imageweeds", "test"]
          and len(rows["test"]) == 4)
    nt = json.loads((d / "nevertrain_dhash.json").read_text())
    nt["entries"] = nt["entries"][:-1]
    nt["min_expected"] = len(nt["entries"])
    C2.write_json_atomic(d / "nevertrain_dhash.json", nt)
    check("a changed never-train index refuses", raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError,
                                                         "changed since LOCK v2"))
    write_lock_dir(w, d, base, complete=False)
    check("an incomplete index refuses", raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError,
                                                "not complete"))
    write_lock_dir(w, d, base, eval_splits=("dev", "test"))
    check("an index holding a split the LOCK does not list refuses",
          raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError, "imageweeds"))
    lk = write_lock_dir(w, d, base)
    lk["base_copies_entries"] = 99
    C2.write_json_atomic(d / "LOCK.json", lk)
    check("an entry count other than the LOCK's refuses",
          raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError, "LOCK v2 says 99"))
    lk["splits_version"] = "v1"
    C2.write_json_atomic(d / "LOCK.json", lk)
    check("a LOCK that is not v2 refuses", raises(lambda: G.GuardV2.load(d / "LOCK.json"), C2.Inc2Error,
                                                  "not a v2 LOCK"))
    write_lock_dir(w, d, base, eval_splits=("dev", "test"), omit=("imageweeds",))
    check("a LOCK and index that leave out an evaluation split (imageweeds) refuse, consistent as they are",
          raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError, "covers exactly"))
    write_lock_dir(w, d, base, omit=(("test", "test__e2"),))
    check("an index missing one evaluation image (counts consistent with the LOCK) refuses",
          raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError, "1 missing"))
    write_lock_dir(w, d, base)
    (d / "imageweeds.jsonl").unlink()
    check("an evaluation manifest missing beside the LOCK refuses",
          raises(lambda: G.GuardV2.load(d / "LOCK.json"), G.GuardError, "does not match LOCK v2"))
    lk = write_lock_dir(w, d, base)
    rows = C1.read_manifest(d / "test.jsonl")
    rows[0]["session"] = "x"
    C1.write_manifest(d / "test.jsonl", rows)
    check("v2_eval_rows refuses a changed evaluation manifest",
          raises(lambda: G.v2_eval_rows(d / "LOCK.json"), G.GuardError, "does not match"))
    check("no LOCK: load refuses", raises(lambda: G.GuardV2.load(TMP / "nolock" / "LOCK.json"), C2.Inc2Error))


def test_embed(w, guard):
    print("near_eval_embed: the calibrated embedding detector")
    emb = HueEmbedder()
    out = TMP / "leakcal"
    cal = G.calibrate(out, domain_raw(), emb, w.ref_rows, w.pool_rows, 0.95, 0.01)
    doc = json.loads((out / G.CAL_NAME).read_text())
    check("the calibration passes and is written under the caller's directory (%s)" % G.CAL_NAME,
          doc["calibration"]["ok"] and cal["cos_threshold"] == doc["calibration"]["cos_threshold"]
          and doc["format"] == G.CAL_FORMAT, doc["calibration"].get("why"))
    check("its seeds use the stream's prefix stream/v1/leak",
          all(v.startswith("stream/v1/leak") for v in doc["seeds"].values()) and doc["seeds"], doc["seeds"])
    check("both negative sets are non-empty with no false hit",
          doc["calibration"]["negatives"]["pairs_7_10"]["n"] > 0 and doc["calibration"]["negatives"]["hard"]["n"] > 0
          and doc["calibration"]["negatives"]["pairs_7_10"]["false_hits"] == 0, doc["calibration"]["negatives"])
    calls = emb.calls
    again = G.calibrate(out, domain_raw(), emb, w.ref_rows, w.pool_rows, 0.95, 0.01)
    check("a rerun with the same identity is a no-op", emb.calls == calls and again["cos_threshold"] == cal["cos_threshold"])
    check("nothing is written outside the caller's directory (no funnel directory)",
          not (TMP / "inc" / "funnel").exists())
    index = G.eval_index(w.eval_rows, emb, out / "eval_desc.npz")
    scanner = G.EmbedScanner(index, cal)
    d = w.root / "embed"
    crop, used_c = far_augment(w.meta["test__e1"][2], "crop", 100, {"frac": [0.15, 0.15]})
    shear, used_s = far_augment(w.meta["test__e2"][2], "shear", 200)
    bright_src = save(render(*w.meta["test__e3"][:2]), d / "b_src.png")
    with Image.open(bright_src) as im:
        from PIL import ImageEnhance
        brightened = ImageEnhance.Brightness(im.convert("RGB")).enhance(1.25)
    bpath = save(brightened, d / "bright_full.png")
    bcrop, _u = far_augment(bpath, "crop", 300, {"frac": [0.15, 0.15]})
    rows = [mrow(save(crop, d / "crop15.png"), "crop15", "x"), mrow(save(shear, d / "shear.png"), "shear", "x"),
            mrow(save(bcrop, d / "bright_crop.png"), "bright_crop", "x"),
            mrow(save(render(grid(7000), next(TRIPLES)), d / "clean0.png"), "clean0", "x"),
            mrow(save(render(grid(7001), next(TRIPLES)), d / "clean1.png"), "clean1", "x")]
    broken = d / "broken.png"
    broken.write_bytes(b"\x89PNG broken")
    rows.append({"key": "broken", "image": str(broken), "sha256": None})
    far = {r["key"]: min(bits(G.dhash_variants(r["image"])[v], C1.dhash(w.meta[t][2])) for v in G.VARIANTS)
           for r, t in zip(rows[:3], ("test__e1", "test__e2", "test__e3"))}
    check("the planted copies are > 6 bits from their original under all 8 variants %s" % far,
          all(v > 6 for v in far.values()))
    res = guard.check_embed(rows, scanner, desc_path=d / "desc.npz")
    check("a 15 % crop of a test image: near_eval_embed", res.get("crop15", (None,))[0] == "near_eval_embed"
          and res["crop15"][1]["eval_key"] == "test__e1" and res["crop15"][1]["eval_split"] == "test", res.get("crop15"))
    check("a sheared copy: near_eval_embed", res.get("shear", (None,))[0] == "near_eval_embed"
          and res["shear"][1]["eval_key"] == "test__e2", res.get("shear"))
    check("a brightened crop: near_eval_embed", res.get("bright_crop", (None,))[0] == "near_eval_embed"
          and res["bright_crop"][1]["eval_key"] == "test__e3", res.get("bright_crop"))
    check("clean images pass", "clean0" not in res and "clean1" not in res, res)
    check("check_embed counts refusals by reason and cleared rows as embed_pass",
          guard.counts["near_eval_embed"] == 3 and guard.counts["embed_pass"] == 2, dict(guard.counts))
    check("an image that cannot be described is unhashable (fail closed)",
          res.get("broken", (None,))[0] == "unhashable" and res["broken"][1]["stage"] == "embed", res.get("broken"))
    check("the copy entries carry evaluation keys, never evaluation paths",
          not any(str(v).endswith(".png") and "/eval/" in str(v)
                  for k in res for v in (res[k][1] or {}).values()), res)
    check("scanner.record names the calibration and the evaluation descriptors",
          scanner.record()["calibration"]["sha256"] == C1.sha256_file(out / G.CAL_NAME)
          and scanner.record()["eval_images"] == 12)

    # funnel-format calibration, reused
    fdoc = {"format": "funnel-leak/1", "status": "complete",
            "detector": {"descriptor": {"embedder": emb.name}},
            "calibration": dict(doc["calibration"])}
    fpath = TMP / "funnel_leak_v1.json"
    C2.write_json_atomic(fpath, fdoc)
    fc = G.load_calibration(fpath, emb.name)
    check("load_calibration accepts a passed funnel leak_v1.json", fc["cos_threshold"] == cal["cos_threshold"]
          and fc["format"] == "funnel-leak/1")
    check("... and refuses it for another embedder",
          raises(lambda: G.load_calibration(fpath, "other:cls"), G.GuardError, "calibrated with"))
    fdoc["detector"] = {}
    C2.write_json_atomic(fpath, fdoc)
    check("... and one that names no embedder", raises(lambda: G.load_calibration(fpath), G.GuardError, "embedder"))
    fdoc["detector"] = {"descriptor": {"embedder": emb.name}}
    good = json.loads(json.dumps(fdoc["calibration"]))
    for what, edit, text in (
            ("a cosine threshold above 1 (the embedding half could never fire)",
             lambda c: c.update(cos_threshold=2.0), "not a cosine"),
            ("a family without positives", lambda c: c["positives"].pop("shear"), "family shear"),
            ("a family below its recall gate", lambda c: c["positives"]["crop"].update(hits=0), "family crop"),
            ("gates looser than H6's", lambda c: c.update(recall_min=0.5), "looser"),
            ("an empty negative set", lambda c: c["negatives"]["hard"].update(n=0), "negative set hard"),
            ("a dHash radius other than 6", lambda c: c.update(dhash_bits_max=0), "never-train radius")):
        cal_x = json.loads(json.dumps(good))
        edit(cal_x)
        fdoc["calibration"] = cal_x
        C2.write_json_atomic(fpath, fdoc)
        check("load_calibration refuses a record that says ok but shows %s" % what,
              cal_x["ok"] is True and raises(lambda: G.load_calibration(fpath, emb.name), G.GuardError, text))
    fdoc["calibration"] = dict(good, ok=False)
    C2.write_json_atomic(fpath, fdoc)
    check("... and a failed one", raises(lambda: G.load_calibration(fpath), G.GuardError, "no passed calibration"))
    other = HueEmbedder("other-model:cls")
    idx2 = G.eval_index(w.eval_rows, other, None)
    check("a scanner whose index uses another embedder than the calibration refuses",
          raises(lambda: G.EmbedScanner(idx2, cal), G.GuardError, "calibration's embedder"))

    const = ConstEmbedder()
    out2 = TMP / "leakcal_const"
    check("a collapsing embedder fails the calibration (false positives)",
          raises(lambda: G.calibrate(out2, domain_raw(), const, w.ref_rows, w.pool_rows, 0.95, 0.01), G.GuardError,
                 "failed its calibration"))
    d2 = json.loads((out2 / G.CAL_NAME).read_text())
    check("... its file says ok false, and it cannot be loaded",
          d2["calibration"]["ok"] is False and raises(lambda: G.load_calibration(out2 / G.CAL_NAME), G.GuardError))
    check("... and a rerun refuses again without recomputing",
          raises(lambda: G.calibrate(out2, domain_raw(), const, w.ref_rows, w.pool_rows, 0.95, 0.01), G.GuardError,
                 "the calibration failed"))


def test_rules():
    print("import and domain rules")
    src = (ROOT / "weed_optimizer_framework" / "tools" / "inc2" / "guard.py").read_text()
    refs = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            refs += [n for n in names if "mega_trainer" in n]
        elif isinstance(node, ast.Name) and "mega_trainer" in node.id:
            refs.append(node.id)
        elif isinstance(node, ast.Attribute) and "mega_trainer" in node.attr:
            refs.append(node.attr)
    check("guard.py's code never imports or names mega_trainer (it hashes through inc.common.dhash)", not refs, refs)
    check("GuardV2 hashes with inc.common.dhash", G.C2.dhash is C1.dhash)
    check("a guard check does not import ultralytics", "ultralytics" not in sys.modules)
    from weed_optimizer_framework.tools.cwd12_species import CWD12_SPECIES
    low = src.lower()
    words = [s.lower() for s in CWD12_SPECIES] + ["cwd12", "imageweeds", "3season", "otherplant", "cottonweed",
                                                    "weed", "lulab", "ndsu"]
    found = [w for w in words if w in low]
    check("guard.py is domain-free (no species, dataset or exam name)", not found, found)


def main():
    w = World()
    test_hashing(w)
    guard = test_check(w)
    test_load(w)
    test_embed(w, guard)
    test_rules()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), FAILURES))
        return 1
    print("all inc2.guard checks passed%s" % (" (%d skipped)" % len(SKIPS) if SKIPS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
