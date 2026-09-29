#!/usr/bin/env python3
"""inc2/splits.py and run_inc2_splits.sh: splits v2 end to end on a synthetic
world (docs/CONTINUOUS_LOOP.md §4.1-4.2, D-A, L-5, §9 group A acceptance).

The world: a small cwd12 (10 train sessions, valid, test), 3SeasonWeedDet10
data2022 / data2023 and ImageWeeds, built into splits v1 by the real
inc.splits build and lock; then Step 1's base_selected.jsonl (harvested
images of cwp10, vanpe and two other sources), select_summary.json,
pool.jsonl (the calibration's negative group) and the funnel's card index.
Every image is a block-grid "photograph" in its own hue triple, so dHash
distances can be planted exactly and a hue-histogram embedder stands in for
DINOv2 (the funnel copy-detector test's world).

Pinned:
  * the real-data pins equal the local artifacts (v1 LOCK shas, the scorer,
    base_selected's sha256, 617 + 1,977 + 3,208 = 5,802 never-train entries,
    1,915 / 1,784 tsw rows, cwp10 544 + vanpe 268 = 812 L-5 images), and a
    build without --testing refuses the synthetic world on them, writing
    nothing;
  * a copy of a test image (a 90-degree rotation) in the part of base B that
    base v2 keeps refuses the build (R4), writing nothing;
  * decision L-8: the world's train_core holds a transverse copy of a test
    image (v1, which compared the stored dHash only, kept it). The build
    drops it instead of refusing: listed in train_core_variant_drops.jsonl
    with its match, recorded in summary.json as an incident, absent from
    train_core.jsonl (v1's bytes minus that line), base_v2, the base-copy
    index and the provenance; lock records the list's sha256 and the
    derivation; GuardV2, the v2 NeverTrainGuard and inc2.train.guard_rows
    refuse its bytes re-listed under another key. More drops than the cap
    (max(1, 0.5 % of train_core)), or a stored-dHash hit, refuse the build
    (R4), writing nothing; lock refuses a list changed after the build,
    and a forged list naming a clean row even with a matching summary;
    verify catches a changed list after lock;
  * build: the three byte copies are v1's bytes (the v1 LOCK shas); the
    never-train index holds dev + test + imageweeds, incomplete (refused)
    until lock; tsw keys tsw2x__<stem>, source 3seasonweeddet10/data202x,
    session = stem minus the frame number, labels byte copies (same sha256);
    no training row carries an exam key; drops, each recorded: a flipped test
    image (near_eval_variant), a rotated imageweeds image, the ood23 near
    duplicate of an ood22 row (the tsw22 row is kept), every row of a dev
    session and (L-9(a)) the row sharing a session with test, counted; L-5
    drops every cwp10 and vanpe image (listed in l5_excluded.jsonl), and a
    flipped test image among them is recorded as an H6(b) incident without
    refusing; a harvested near copy of a train_core image is dropped; base_v2
    is the union of the parts, pairwise disjoint; licences resolve from the
    card index, the funnel config and the Zenodo record, an unresolved one is
    research_only, and an owner table resolves train_core;
  * lock refuses before the embedding scan; the scan (the stream's own
    calibration, since the funnel's leak_v1.json there failed) finds the 15 %
    crop of a test image in tsw22 and the sheared test image in base B's kept
    part, both > 6 bits under every variant, the flipped / rotated copies,
    and the 2022 capture of a test scene (kept under L-9(a));
    it covers every candidate (all v1 ood rows, all of base B), not the
    build's manifests, and carries evaluation keys only;
    lock then refuses (flagged rows), and a second build refuses (R4, the
    harvested copy); without that image the second build drops the tsw crop
    (near_eval_embed); a scan reusing a passed funnel leak_v1.json records it;
  * lock: LOCK v2 records the manifests, the index shas and counts, the
    scorer, derived_from (the v1 LOCK, identical byte copies, train_core as
    v1 minus the L-8 list), the L-8 list's sha256 and the H6
    status; the indexes are complete; every file under splits/v2 is 0444;
    verify is [] for v2 and v1, and no v1 file changed; GuardV2.load works and
    refuses a flipped test image, a base copy and the dropped ood23 duplicate,
    and passes a fresh image; build, scan and lock refuse once locked; verify
    catches a changed label and a writable file;
  * decision L-9 (the second real build, job 47259471, refused on scenes):
    the world adds two train_core frames of a test scene (cosine 0.95 under
    the hue embedder, another capture date), a 2022 tsw22 capture of a test
    scene in its own capture session (cosine 1.0, far by dHash) and a
    harvested photograph of a test scene (0.85). Every tsw and base B
    candidate's copy rule is in copy_rules.jsonl and the provenance file:
    tsw rows of a dev or test capture session are dropped (the test-session
    row is no longer kept), other tsw rows with a capture session are exempt
    from the embedding threshold (the flagged 2022 capture is kept), rows
    without one are judged like base B (the 15 % crop is dropped). The scan
    writes the v2 calibration: per-image hard negatives raise the threshold
    above the train_core scene frames (their false hits at the scan's
    threshold recorded), the base's positives give back its threshold,
    rates and bounds per tier, recall per family, evaluation keys only.
    With the incident's funnel calibration (0.80, no seeds) as the base: the
    sheared test image in base B still refuses at the v2 threshold; without
    it the build passes, keeping the harvested scene (recorded, not applied)
    and the funnel's matches against ood23 (not_v2_split), test and
    imageweeds below the v2 threshold (below_v2_threshold); a scan whose
    calibration file changed is stale (build scans again), and
    build --skip-scan on a stale scan (no v2 calibration) reproduces the
    incident, except the ood23 match. lock refuses another v2 calibration
    than the build's, a changed copy_rules.jsonl and a provenance copy_rule
    that is not the re-derived one; LOCK v2 records the three files'
    sha256 and the v2 threshold; verify catches each file changed after lock;
  * the CLI end to end: build --testing, lock, verify, summary;
  * run_inc2_splits.sh: bash -n, GPU-shared, verbs, refusals (unknown verb,
    --testing, scan without a GPU, outer/nested drift), a dry run per verb,
    no git reset, rsync or copy of the package.

No network, no GPU, no Slurm.

Run:  python3 tests/test_inc2_splits.py
"""
import ast
import itertools
import json
import os
import pathlib
import random
import shutil
import stat
import subprocess
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_splits_"))
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REAL_INC = ROOT / "results" / "framework" / "inc"
SCRIPT = ROOT / "run_inc2_splits.sh"

FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:2000]))
        FAILURES.append(name)


def skip(name, reason):
    print("SKIP: %s (%s)" % (name, reason))
    SKIPS.append(name)


try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print("SKIP: every check (numpy / PIL not importable: %s)" % e)
    sys.exit(0)

from weed_optimizer_framework.tools.inc import common as C1  # noqa: E402
from weed_optimizer_framework.tools.inc import splits as S1  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import embed_calibration as EC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import guard as G  # noqa: E402
from weed_optimizer_framework.tools.inc2 import splits as S2  # noqa: E402
from weed_optimizer_framework.tools.funnel import leak as L  # noqa: E402

assert C1.INC_DIR == TMP / "inc" and C2.SPLITS_DIR == TMP / "inc" / "splits" / "v2", "fake INC_DIR not in effect"


def raises(fn, text="", exc=C2.Inc2Error):
    try:
        fn()
    except exc as e:
        return text in str(e), str(e)
    return False, "no error"


REPO = TMP / "repo"
INC = TMP / "inc"
SCORER = TMP / "fake_scorer.py"
SCORER.write_text("# the scorer the synthetic v1 LOCK records\n")
SESSIONS = [12, 6, 5, 5, 4, 4, 3, 3, 2, 2]
N_BINS = 48
BLOCK = 12
CWP, VANPE = S2.L5_SOURCES
WCD = "project_agml__weed_crop_detection"
IWA = "project_agml__imageweeds_aerial_weed_detection"
MH = "project_agml__mh_weed16_weed_detection"
IMG = {}
VARIANT_STEM = "20210701_FakeCam_S0_%d" % (SESSIONS[0] + 1)      # the L-8 image: a transverse test copy
SCENE_STEMS = tuple("20210701_FakeCam_S0_%d" % (SESSIONS[0] + k) for k in (2, 3))   # L-9(c) hard negatives
Y22_STEM = "20220715_FakeCam_Y22_1"             # L-9(a): a tsw22 capture of a test scene, another session


# ------------------------------------------------------------------- world
class HueEmbedder:
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


MADE = []


def fake_make_embedder(_raw):
    """Stands in for inc2.splits._make_embedder (DINOv2 from the HF cache):
    every embedder a build or scan makes for itself is recorded."""
    e = HueEmbedder()
    MADE.append(e)
    return e


S2._make_embedder = fake_make_embedder


def hue_triples(n):
    out = []
    for t in itertools.combinations(range(N_BINS), 3):
        if all(len(set(t) & set(u)) <= 1 for u in out):
            out.append(t)
            if len(out) == n:
                return out
    raise RuntimeError("not enough triples")


TRIPLES = iter(hue_triples(260))


def grid(seed):
    rng = np.random.default_rng(seed)
    levels = np.array([70, 130, 190, 250])
    g = np.zeros((8, 9), dtype=np.int64)
    for r in range(8):
        g[r, 0] = rng.choice(levels)
        for c in range(1, 9):
            g[r, c] = rng.choice([v for v in levels if abs(v - g[r, c - 1]) >= 60])
    return g


def render(g, hues, bands=(0, 0, 0, 1, 1, 1, 2, 2)):
    """The block-grid photograph: row band i painted in hue hues[bands[i]],
    brightness from the grid. The hue histogram (HueEmbedder) depends on the
    bands only, the dHash on the grid and the hues."""
    band = np.repeat(np.repeat(np.array(bands)[:, None], 9, axis=1), BLOCK, axis=0)
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


def photo(path, seed):
    g, h = grid(seed), next(TRIPLES)
    save(render(g, h), path)
    IMG[str(path)] = (g, h)
    return path


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


def neighbour(g0, hues0, hues, want, seed, all_far=False):
    h0 = L.dhash_variants(render(g0, hues0))["id"]
    rng = np.random.default_rng(seed)
    for _ in range(6000):
        g = g0.copy()
        for _k in range(int(rng.integers(1, 7))):
            g[int(rng.integers(8)), int(rng.integers(9))] = int(rng.choice([70, 130, 190, 250]))
        img = render(g, hues)
        hv = L.dhash_variants(img)
        if bits(hv["id"], h0) in want and (not all_far or min(bits(hv[v], h0) for v in L.VARIANTS) > 6):
            return img
    raise RuntimeError("could not plant a neighbour")


def scene_of(src, bands, extra_hue, seed):
    """A "same scene" photograph of the image at src (L-9): its hues in the
    given row bands (plus extra_hue as a fourth), a fresh brightness grid, so
    its hue histogram is close to src's (cosine by the bands: (3,3,1,1) ->
    0.954, (2,2,2,2) -> 0.853, the same bands -> 1.0) while its dHash is
    more than 6 bits from src's under every variant: not a copy."""
    _g0, h0 = IMG[str(src)]
    t0 = L.dhash_variants(render(_g0, h0))["id"]
    hues = tuple(h0) + (extra_hue,)
    for s_ in range(seed, seed + 400):
        img = render(grid(s_), hues, bands)
        hv = L.dhash_variants(img)
        if min(bits(hv[v], t0) for v in L.VARIANTS) > 6:
            return img
    raise RuntimeError("no far scene image")


SCENE_TRAIN = (0, 0, 0, 1, 1, 1, 2, 3)          # (3,3,1,1): cosine 0.954 with the test image
SCENE_BASE = (0, 0, 1, 1, 2, 2, 3, 3)           # (2,2,2,2): cosine 0.853


def far_augment(src, family, seed, params=None):
    with Image.open(src) as im:
        base = im.convert("RGB")
    h0 = L.dhash_variants(base)["id"]
    for s in range(seed, seed + 400):
        img, _used = L.augment(base, family, np.random.default_rng(s), params)
        hv = L.dhash_variants(img)
        if min(bits(hv[v], h0) for v in L.VARIANTS) > 6:
            return img
    raise RuntimeError("no far %s copy" % family)


def transpose(src, op):
    with Image.open(src) as im:
        return im.transpose(op)


def yolo(path, lines):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(v) for v in ln) + "\n" for ln in lines))


def voc(img_path, objects):
    with Image.open(img_path) as im:
        w, h = im.size
    objs = "".join("<object><name>%s</name><bndbox><xmin>%d</xmin><ymin>%d</ymin><xmax>%d</xmax><ymax>%d</ymax>"
                   "</bndbox></object>" % ((n,) + tuple(b)) for n, b in objects)
    pathlib.Path(img_path).with_suffix(".xml").write_text(
        "<annotation><size><width>%d</width><height>%d</height><depth>3</depth></size>%s</annotation>" % (w, h, objs))


def cwd12_root():
    return REPO / "downloads" / "cottonweeddet12"


def test_image(n):
    return cwd12_root() / "test" / "images" / ("20210801_FakeCam_TEST_%d.png" % n)


def make_v1_world():
    root = cwd12_root()
    g, seed = 0, 100
    for s, n in enumerate(SESSIONS):
        for f in range(n):
            stem = "20210701_FakeCam_S%d_%d" % (s, f + 1)
            photo(root / "train" / "images" / (stem + ".png"), seed)
            seed += 1
            yolo(root / "train" / "labels" / (stem + ".txt"),
                 [((g + k) % 12, 0.2 + 0.25 * j, 0.5, 0.1, 0.1) for j, k in enumerate((0, 4, 8))])
            g += 1
    for split, n in (("valid", 4), ("test", 4)):
        for f in range(n):
            stem = "20210801_FakeCam_%s_%d" % (split.upper(), f + 1)
            photo(root / split / "images" / (stem + ".png"), seed)
            seed += 1
            yolo(root / split / "labels" / (stem + ".txt"), [(f % 12, 0.5, 0.5, 0.2, 0.2)])
    # L-8: a transverse copy of a test image among the frames of train session S0 (too large to become dev)
    p = save(transpose(test_image(1), Image.Transpose.TRANSVERSE), root / "train" / "images" / (VARIANT_STEM + ".png"))
    yolo(root / "train" / "labels" / (VARIANT_STEM + ".txt"), [(5, 0.5, 0.5, 0.2, 0.2)])
    # L-9(c): two frames of S0 (another capture date than test) that photograph test image 3's scene: the hard
    # negatives whose cosine (0.954) the v2 calibration must rise above
    extra = next(TRIPLES)[0]
    for i, stem in enumerate(SCENE_STEMS):
        save(scene_of(test_image(3), SCENE_TRAIN, extra, 3000 + 50 * i), root / "train" / "images" / (stem + ".png"))
        yolo(root / "train" / "labels" / (stem + ".txt"), [(7, 0.5, 0.5, 0.2, 0.2)])
    ts = REPO / "downloads" / "3seasonweeddet10"
    d22, d23 = ts / "data2022" / "fieldA", ts / "data2023"
    for s in range(len(SESSIONS)):                 # one frame of every train session (dev ones included)
        p = photo(d22 / ("20210701_FakeCam_S%d_901.png" % s), 500 + s)
        voc(p, [("Waterhemp", (10, 10, 50, 50))])
    p = photo(d22 / "20210801_FakeCam_TEST_901.png", 520)          # shares a session with test
    voc(p, [("Palmer Amaranth", (10, 10, 50, 50)), ("Lambsquarters", (60, 20, 100, 60))])
    # L-9(a): a 2022 capture of test image 1's scene (its hues, cosine 1.0 under the hue embedder, far by dHash):
    # the scan flags it at any threshold; its capture session is not a dev or test one, so it is kept
    g1, h1 = IMG[str(test_image(1))]
    p = save(scene_of(test_image(1), (0, 0, 0, 1, 1, 1, 2, 2), h1[0], 3200), d22 / (Y22_STEM + ".png"))
    voc(p, [("Carpetweed", (10, 10, 50, 50))])
    for n in ("field22_a", "field22_b"):
        p = photo(d22 / (n + ".png"), 530 + len(n) + (n == "field22_b"))
        voc(p, [("Carpetweed", (10, 10, 50, 50))])
    p = save(transpose(test_image(1), Image.Transpose.FLIP_LEFT_RIGHT), d22 / "field22_hflip_test.png")
    voc(p, [("Goosegrass", (10, 10, 50, 50))])
    p = save(far_augment(test_image(2), "crop", 600, {"frac": [0.15, 0.15]}), d22 / "field22_crop_test.png")
    voc(p, [("Goosegrass", (10, 10, 40, 40))])
    ga, ha = IMG[str(d22 / "field22_a.png")]
    p = save(neighbour(ga, ha, ha, {1, 2, 3}, 700), d23 / "field23_dup22.png")
    voc(p, [("Sicklepod", (10, 10, 50, 50))])
    for n in ("field23_a", "field23_b"):
        p = photo(d23 / (n + ".png"), 710 + (n == "field23_b"))
        voc(p, [("Lambsquarters", (10, 10, 50, 50)), ("Morningglory", (60, 20, 90, 50))])
    iw = REPO / "datasets" / S1.IMAGEWEEDS_SLUG
    for i in range(5):
        photo(iw / "images" / ("iw_%d.png" % i), 800 + i)
        yolo(iw / "labels" / ("iw_%d.txt" % i), [(i, 0.5, 0.5, 0.2, 0.2)])
    p = save(transpose(iw / "images" / "iw_0.png", Image.Transpose.ROTATE_90), d23 / "field23_rot_iw.png")
    voc(p, [("Sicklepod", (10, 10, 50, 50))])
    S1.build(frac_range=(0.13, 0.18), min_dev_boxes=1)
    S1.lock(scorer_path=SCORER, accept_counts=True)


BASE, EXTRA = {}, {}


def harvested(table, slug, name, img, boxes):
    p = save(img, REPO / "datasets" / slug / "train" / "images" / (name + ".png"))
    key = "%s__train__%s" % (slug, name)
    lbl = INC / "step1" / "labels" / (key + ".txt")
    yolo(lbl, boxes)
    table[name] = {"image": str(p), "label": str(lbl), "sha256": C1.sha256_file(p),
                   "label_sha256": C1.sha256_file(lbl), "source": slug, "session": "", "key": key}


def v1_core():
    """v1 train_core without the planted L-8 image and the L-9 scene frames, sorted by key."""
    return sorted((r for r in C1.read_manifest(C1.manifest_path("train_core"))
                   if not r["key"].endswith((VARIANT_STEM,) + SCENE_STEMS)), key=lambda r: r["key"])


def variant_row():
    return next(r for r in C1.read_manifest(C1.manifest_path("train_core")) if r["key"].endswith(VARIANT_STEM))


def make_step1():
    core = v1_core()
    seed = 900
    for slug, names in ((CWP, ("c0", "c1")), (VANPE, ("v0", "v1")), (WCD, ("w0", "w1", "w2")), (IWA, ("a0", "a1"))):
        for n in names:
            harvested(BASE, slug, n, render(grid(seed), next(TRIPLES)), [(0, 0.5, 0.5, 0.2, 0.2), (12, .2, .2, .1, .1)])
            seed += 1
    harvested(BASE, CWP, "hflip_test3", transpose(test_image(3), Image.Transpose.FLIP_LEFT_RIGHT),
              [(3, 0.5, 0.5, 0.2, 0.2)])
    harvested(BASE, CWP, "broken", render(grid(990), next(TRIPLES)), [(3, 0.5, 0.5, 0.2, 0.2)])
    bad = pathlib.Path(BASE["broken"]["image"])
    bad.write_bytes(b"\x89PNG truncated")                  # an L-5 image nobody can read
    BASE["broken"]["sha256"] = C1.sha256_file(bad)
    g0, h0 = IMG[core[5]["image"]]
    harvested(BASE, WCD, "near_core", neighbour(g0, h0, h0, {1, 2}, 950), [(1, 0.5, 0.5, 0.2, 0.2)])
    # L-9(c): a harvested photograph of test image 3's scene (cosine 0.853, far by dHash): not a copy; flagged
    # at a low per-pair threshold, cleared at the v2 one (below the scene frames of train_core at 0.954)
    harvested(BASE, WCD, "scene_test3", scene_of(test_image(3), SCENE_BASE, next(TRIPLES)[0], 3100),
              [(4, 0.5, 0.5, 0.2, 0.2)])
    harvested(EXTRA, WCD, "rot_test4", transpose(test_image(4), Image.Transpose.ROTATE_90), [(2, .5, .5, .2, .2)])
    harvested(EXTRA, WCD, "shear_test4", far_augment(test_image(4), "shear", 960), [(2, .5, .5, .2, .2)])
    pool = []
    for i in range(6):
        p = save(render(grid(1100 + i), next(TRIPLES)), REPO / "datasets" / MH / ("m%d.png" % i))
        pool.append((p, "%s__m%d" % (MH, i)))
    for i in range(6):
        g, h = IMG[core[i]["image"]]
        p = save(neighbour(g, h, next(TRIPLES), set(range(7, 11)), 1200 + i, all_far=True),
                 REPO / "datasets" / MH / ("n%d.png" % i))
        pool.append((p, "%s__n%d" % (MH, i)))
    rows = []
    for p, key in pool:
        lbl = INC / "step1" / "labels" / (key + ".txt")
        yolo(lbl, [(12, 0.5, 0.5, 0.2, 0.2)])
        rows.append({"image": str(p), "label": str(lbl), "sha256": C1.sha256_file(p),
                     "label_sha256": C1.sha256_file(lbl), "source": MH, "session": "", "key": key})
    C1.write_manifest(INC / "step1" / "pool.jsonl", rows)
    per_slug = {CWP: {"kept": 3, "cwd12_copies": 2, "near_eval_by_split": {"test": 1}},
                VANPE: {"kept": 2, "cwd12_copies": 1, "near_eval_by_split": {"dev": 1}},
                WCD: {"kept": 5, "cwd12_copies": 0, "near_eval_by_split": {}},
                IWA: {"kept": 2, "cwd12_copies": 0, "near_eval_by_split": {}}}
    C2.write_json_atomic(INC / "step1" / "pool_summary.json", {"per_slug": per_slug})
    C2.write_json_atomic(INC / "funnel" / "cards" / "index.json",
                         {"cards": {IWA: [{"what": "card", "licence": None, "url": "u0"},
                                          {"what": "card", "licence": "cc-by-4.0", "url": "https://hf/card"}]}})
    C2.write_json_atomic(S2.default_tsw_record(), {"metadata": {"license": {"id": "cc-by-4.0"}}})


def write_base_selected(extra=()):
    rows = [BASE[k] for k in sorted(BASE)] + [EXTRA[k] for k in extra]
    sha = C1.write_manifest(S2.base_selected_path(), rows)
    C2.write_json_atomic(S2.select_summary_path(), {"outputs": {"base_selected.jsonl": {"sha256": sha,
                                                                                        "rows": len(rows)}}})
    return sha


def lock_v1_core_sha():
    return C1.sha256_file(C1.manifest_path("train_core"))


def tree_shas(root):
    return {p: C1.sha256_file(p) for p in S2._walk_files(root)}


# ------------------------------------------------------------------ checks
def test_real_pins():
    print("the real-data pins against the local artifacts")
    lk = REAL_INC / "splits" / "v1" / "LOCK.json"
    if not lk.is_file():
        skip("real pins", "%s is not here" % lk)
        return
    lock = json.loads(lk.read_text())
    pins = S2.REAL_PINS
    check("the v1 manifest pins are prefixes of the local v1 LOCK",
          all(lock["manifests"][s].startswith(p) for s, p in pins["v1_manifests"].items()), lock["manifests"])
    check("the scorer pin is the local v1 LOCK's", lock["scorer_sha256"].startswith(pins["v1_scorer"]))
    summ = json.loads((REAL_INC / "splits" / "v1" / "summary.json").read_text())
    n = {s: summ["splits"][s]["images"] for s in summ["splits"]}
    check("the row pins are the local v1 summary's", all(n[s] == pins["rows"][s] for s in n), (n, pins["rows"]))
    check("never-train v2 = dev + test + imageweeds = %d" % pins["nevertrain_entries"],
          n["dev"] + n["test"] + n["imageweeds"] == pins["nevertrain_entries"] == 5802)
    sel = REAL_INC / "step1" / "select_summary.json"
    if not sel.is_file():
        skip("base_selected pins", "%s is not here" % sel)
        return
    ss = json.loads(sel.read_text())
    rec = ss["outputs"]["base_selected.jsonl"]
    check("base_selected's sha256 and rows are the pins", rec["sha256"].startswith(pins["base_selected"])
          and rec["rows"] == pins["rows"]["base_selected"])
    per = ss["sources"]["selected"]
    l5 = sum(per.get(s, 0) for s in S2.L5_SOURCES)
    check("L-5 removes cwp10 %d + vanpe %d = %d of base B's 878, leaving %d"
          % (per.get(CWP, 0), per.get(VANPE, 0), l5, sum(per.values()) - l5),
          l5 == pins["l5_excluded"] == 812 and sum(per.values()) == 878)


def test_units():
    print("licence rule and keys")
    cases = {"CC BY 4.0": False, "cc-by-4.0": False, "CC BY-SA 4.0": False, "CC0": False, "MIT": False,
             "cc-by-nc-sa-4.0": True, "CC BY-NC 4.0": True, "Non-commercial use only": True,
             "unresolved": True, None: True, "": True, "Other (see README)": True,
             "CC BY 4.0 (Mendeley 10.17632/mthv4ppwyw.2; AgML card cc-by-4.0)": False,
             "CC BY Non Commercial 4.0": True, "CC BY 4.0, research use only": True,
             "Attribution-NonCommercial-ShareAlike": True, "CC BY 4.0 (academic use)": True}
    got = {k: S2.research_only(k) for k in cases}
    check("research_only: permissive -> False; NC, unresolved, unknown -> True", got == cases, got)
    check("tsw_key maps ood22__x -> tsw22__x (the v1 disambiguation suffix kept)",
          S2.tsw_key("tsw22", "ood22__img_1__ab12cd34") == "tsw22__img_1__ab12cd34"
          and raises(lambda: S2.tsw_key("tsw23", "ood22__x"), "lacks the prefix")[0])
    groups = {"LuLab": ["train_core", "project_agml__three_season_weed_detection"], "NDSU": [IWA]}
    check("lab groups: train_core, tsw (through AgML's copy of the release), a slug; unknown -> None",
          S2.lab_group(groups, "cottonweeddet12/train", "train_core") == "LuLab"
          and S2.lab_group(groups, "3seasonweeddet10/data2022", "tsw22") == "LuLab"
          and S2.lab_group(groups, IWA, "base_b") == "NDSU" and S2.lab_group(groups, WCD, "base_b") is None)



def test_near_both_directions():
    print("near copies of an earlier part, both directions (job 47260765)")
    rnd = random.Random(47260765)

    def far(n=8):                        # hashes >= 20 bits from 0 and from each other's seeds
        out = []
        while len(out) < n:
            h = rnd.getrandbits(64)
            if bin(h).count("1") >= 20:
                out.append(h)
        return out
    y = far(1)[0]
    x_vars = [0] + far(2) + [y ^ 0b111] + far(4)          # X's rot90 is 3 bits from Y's dHash
    x = {"key": "train_core__x", "dhash": 0, "variants": x_vars}
    yv = [y] + far(7)
    yrow = {"key": "tsw23__y", "dhash": y, "variants": yv}
    one_way = [h for h in yv if bin(h ^ x["dhash"]).count("1") <= S2.BITS]
    idx = S2.EarlierIndex()
    idx.add(x, "train_core")
    hit = S2._near_earlier(idx, yrow)
    rows = [dict(x, part="train_core", image="a", sha256="a"), dict(yrow, part="tsw23", image="b", sha256="b")]
    check("a pair only the reverse direction finds: none of Y's variants is near X's dHash, X's rot90 is",
          one_way == [] and hit == (("train_core", "train_core__x"), 3, "earlier:rot90"), (one_way, hit))
    check("disjoint_problems flags that pair, and not the base without Y",
          len(S2.disjoint_problems(rows)) == 1 and S2.disjoint_problems(rows[:1]) == [],
          S2.disjoint_problems(rows))
    z = {"key": "base_b__z", "dhash": 1, "variants": [1] + far(7)}
    check("the forward direction still finds a row whose dHash is near an earlier dHash",
          S2._near_earlier(idx, z) == (("train_core", "train_core__x"), 1, "id"))
    w = {"key": "base_b__w", "dhash": far(1)[0], "variants": far(8)}
    check("a row near nothing is not a hit", S2._near_earlier(idx, w) is None)


def test_writer_lock():
    print("the writer lock: a holder whose Slurm job has ended is taken over")
    fake = TMP / "squeue_bin"
    fake.mkdir(exist_ok=True)
    lockp = S2.writer_lock_path()
    lockp.parent.mkdir(parents=True, exist_ok=True)
    old_path = os.environ.get("PATH", "")
    results = {}
    try:
        for state, script in (("ended", "echo 'slurm_load_jobs error: Invalid job id specified' >&2; exit 1"),
                              ("running", "echo RUNNING"), ("no squeue", None)):
            sq = fake / "squeue"
            if sq.exists():
                sq.unlink()
            if script is not None:
                sq.write_text("#!/bin/sh\n%s\n" % script)
                os.chmod(sq, 0o755)
            os.environ["PATH"] = str(fake)
            lockp.write_text("123 some-other-node 2026-09-28T00:00:00Z 4242\n")
            try:
                with S2.writer():
                    results[state] = "taken"
            except C2.Inc2Error as e:
                results[state] = "refused: %s" % e
    finally:
        os.environ["PATH"] = old_path
        if lockp.exists():
            lockp.unlink()
    check("a lock held by an ended job on another node is taken over (and released after)",
          results.get("ended") == "taken", results)
    check("a lock held by a running job, or one squeue cannot judge, refuses",
          str(results.get("running")).startswith("refused") and str(results.get("no squeue")).startswith("refused"),
          results)


def test_pins_refuse():
    print("a build without --testing refuses the synthetic world")
    write_base_selected()
    ok, msg = raises(lambda: S2.build(scorer_path=SCORER, procs=1), "real-data pins")
    check("refused on the real-data pins", ok, msg[:400])
    check("... naming the v1 LOCK shas and the counts", "v1 LOCK sha256 of dev" in msg and "row count" in msg
          and "never-train entries" in msg, msg[:600])
    check("... and nothing was written", not C2.SPLITS_DIR.exists() or not any(C2.SPLITS_DIR.iterdir()))


def test_refuse_included_copy():
    print("a copy of a test image in base B's kept part refuses the build")
    write_base_selected(["rot_test4"])
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="skip"), "R4 INCIDENT")
    check("refused: R4 incident (funnel H6(b))", ok, msg[:300])
    check("... naming the rotated image and its reason", "rot_test4" in msg and "near_eval_variant" in msg, msg[:600])
    check("... nothing written", not S2.summary_path().exists() and not C2.v2_manifest_path("base_v2").exists())


def test_train_core_variant_fixture():
    print("the L-8 fixture: v1 kept a transverse copy of a test image in train_core")
    r = variant_row()
    t1 = C1.dhash(test_image(1))
    hv = G.dhash_variants(r["image"])
    check("v1 lists it in train_core (its stored dHash is %d bits from the test image, beyond v1's 6)"
          % bits(hv["id"], t1), bits(hv["id"], t1) > 6 and r["key"].startswith("train_core__"))
    check("... and its transverse variant is the test image (0 bits)", bits(hv["transverse"], t1) == 0)


def test_refuse_train_core_variant():
    print("L-8's limits: more drops than the cap, or a stored-dHash hit, refuse the build")
    write_base_selected([])
    core = v1_core()
    victim, t = core[2], C1.dhash(test_image(2))
    orig = G.image_hashes

    def planted(path):
        h, v = orig(path)
        if str(path) == victim["image"] and v is not None:
            v = dict(v, rot90=t)                 # its 90-degree rotation is a test image too
        return h, v
    G.image_hashes = planted
    try:
        ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="skip"), "R4 INCIDENT")
    finally:
        G.image_hashes = orig
    n_core = len(C1.read_manifest(C1.manifest_path("train_core")))
    check("two variant hits in a %d-row train_core exceed the cap %d: refused (R4), naming both"
          % (n_core, S2.variant_drop_cap(n_core)),
          ok and S2.variant_drop_cap(n_core) == 1 and "more than the 1" in msg and victim["key"] in msg
          and VARIANT_STEM in msg and "near_eval_variant" in msg, msg[:500])
    check("... nothing written", not S2.summary_path().exists() and not C2.v2_manifest_path("base_v2").exists()
          and not C2.TRAIN_CORE_VARIANT_DROPS.exists())
    check("the cap is max(1, floor(0.5 % of train_core)): 15 of the real 3,049, 1 of 40",
          S2.variant_drop_cap(3049) == 15 and S2.variant_drop_cap(40) == 1 and S2.variant_drop_cap(400) == 2)

    def stored(path):
        h, v = orig(path)
        if str(path) == victim["image"] and v is not None:
            h = C1.dhash(test_image(3))
            v = dict(v, id=h)                    # its stored dHash is a test image: v1 would have caught it
        return h, v
    G.image_hashes = stored
    try:
        ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="skip"), "R4 INCIDENT")
    finally:
        G.image_hashes = orig
    check("a train_core hit on the stored dHash (v1 and this build disagree; L-8 covers flips and rotations "
          "only) refuses (R4)", ok and "stored dHash" in msg and victim["key"] in msg and "near_eval_v2" in msg,
          msg[:400])
    check("... nothing written", not S2.summary_path().exists() and not C2.v2_manifest_path("base_v2").exists())


def test_first_build(v1):
    print("build (CLI, --testing), with the sheared test image in base B")
    write_base_selected(["shear_test4"])
    rc = S2.main(["build", "--testing", "--skip-scan", "--scorer", str(SCORER), "--procs", "1"])
    check("build --testing --skip-scan exits 0", rc == 0)
    check("... and writes no embed_scan.json (the scan was skipped)", not S2.embed_scan_path().exists())
    s = S2.read_summary()
    lock1 = C1.read_lock()
    for split in C2.BYTE_COPIES:
        check("%s is a byte copy of v1 (the v1 LOCK sha)" % split,
              C2.v2_manifest_path(split).read_bytes() == C1.manifest_path(split).read_bytes()
              and C1.sha256_file(C2.v2_manifest_path(split)) == lock1["manifests"][split])
    vr = variant_row()
    v1_lines = C1.manifest_path("train_core").read_bytes().splitlines(keepends=True)
    kept_lines = [ln for ln in v1_lines if json.loads(ln)["key"] != vr["key"]]
    check("L-8: train_core.jsonl is v1's bytes minus the transverse test copy's line (every other line identical, "
          "in order), not a byte copy", C2.v2_manifest_path("train_core").read_bytes() == b"".join(kept_lines)
          and len(kept_lines) == len(v1_lines) - 1
          and C1.sha256_file(C2.v2_manifest_path("train_core")) != lock1["manifests"]["train_core"]
          and s["train_core"]["identical"] is False and s["train_core"]["rows"] == len(v1_lines) - 1
          and s["manifests"]["train_core"]["rows"] == len(v1_lines) - 1, s["train_core"])
    drops = C1.read_manifest(C2.TRAIN_CORE_VARIANT_DROPS)
    vd = s["train_core_variant_drops"]
    check("... listed in train_core_variant_drops.jsonl: its v1 row, dHash and 8 variants, near_eval_variant, the "
          "test image it matches (transverse, 0 bits), decided by L-8",
          len(drops) == 1 and all(drops[0][k] == vr[k] for k in C1.MANIFEST_KEYS)
          and drops[0]["reason"] == "near_eval_variant" and drops[0]["match"]["split"] == "test"
          and drops[0]["match"]["key"].endswith("TEST_1") and drops[0]["match"]["variant"] == "transverse"
          and drops[0]["match"]["bits"] == 0 and drops[0]["decided_by"].startswith("L-8")
          and len(drops[0]["variants"]) == 8 and drops[0]["variants"][0] == drops[0]["dhash"], drops)
    check("... recorded in summary.json as an incident (count, cap, keys, matches, the list's sha256, the note on "
          "B0, B and realloop_v1), not refused", vd["incident"] is True and vd["count"] == 1 and vd["cap"] == 1
          and vd["keys"] == [vr["key"]] and vd["matches"][0]["match"]["variant"] == "transverse"
          and vd["sha256"] == C1.sha256_file(C2.TRAIN_CORE_VARIANT_DROPS) and "not re-run" in vd["v1_results"]
          and "1 of %d" % len(v1_lines) in vd["v1_results"], vd)
    nt = json.loads(C2.NEVER_TRAIN_INDEX.read_text())
    n_eval = sum(len(C1.read_manifest(C1.manifest_path(x))) for x in C2.EVAL_SPLITS)
    check("the never-train index holds dev + test + imageweeds (%d), 6 bits, v1's hashes" % n_eval,
          len(nt["entries"]) == n_eval and nt["bits"] == 6
          and {s_ for _h, s_, _k in nt["entries"]} == set(C2.EVAL_SPLITS))
    check("... incomplete until lock, so every loader refuses it",
          nt["complete"] is False and raises(C2.nevertrain_v2, "expected", RuntimeError)[0])
    dev_sessions = set(json.loads((INC / "splits" / "v1" / "summary.json").read_text())["dev"]["sessions"])
    n_dev = sum(1 for x in range(len(SESSIONS)) if "20210701_FakeCam_S%d" % x in dev_sessions)
    t22, t23 = s["tsw"]["tsw22"], s["tsw"]["tsw23"]
    check("tsw22 drops: the flipped test image (near_eval_variant), the %d dev-session row(s) and (L-9(a)) the "
          "test-session row" % n_dev,
          t22["dropped"] == {"near_eval_variant": 1, "dev_session": n_dev, "test_session": 1} and n_dev >= 1,
          t22["dropped"])
    check("tsw22 keeps the rest, the embedding copy included (no scan yet): %d of %d"
          % (t22["kept"], t22["v1_rows"]), t22["kept"] == t22["v1_rows"] - 2 - n_dev)
    check("tsw23 drops: the ood22 near duplicate (the tsw22 row kept) and the rotated imageweeds image",
          t23["dropped"] == {"near_tsw22": 1, "near_eval_variant": 1}, t23["dropped"])
    ex = {e["key"]: e for e in t23["examples"]}
    check("... each recorded with what it matched", ex["tsw23__field23_dup22"]["match"]["with"] == "tsw22__field22_a"
          and ex["tsw23__field23_rot_iw"]["match"]["split"] == "imageweeds"
          and ex["tsw23__field23_rot_iw"]["match"]["variant"] in ("rot90", "rot270"), t23["examples"])
    ov = t22["session_overlap"]
    check("L-9(a): the row sharing a capture session with test is dropped (test_session) and counted (test 1); "
          "rows sharing one with train_core are kept (%d)" % ov["train_core"],
          ov["test"] == 1 and ov["dev"] == n_dev and ov["train_core"] == len(SESSIONS) - n_dev
          and "tsw22__20210801_FakeCam_TEST_901" not in {r["key"] for r in C1.read_manifest(
              C2.v2_manifest_path("tsw22"))}
          and {e["key"]: e["reason"] for e in t22["examples"]}.get("tsw22__20210801_FakeCam_TEST_901")
          == "test_session", ov)
    rows22 = C1.read_manifest(C2.v2_manifest_path("tsw22"))
    v1_22 = {S2.tsw_key("tsw22", r["key"]): r for r in C1.read_manifest(C1.manifest_path("ood22"))}
    check("tsw rows: key tsw22__<stem>, source 3seasonweeddet10/data2022, session = stem minus frame number",
          all(r["key"].startswith("tsw22__") and r["source"] == "3seasonweeddet10/data2022" for r in rows22)
          and next(r for r in rows22 if r["key"].endswith(Y22_STEM))["session"] == "20220715_FakeCam_Y22"
          and next(r for r in rows22 if r["key"].endswith("field22_a"))["session"] == "field22_a")
    rules = {x["key"]: x for x in C1.read_manifest(S2.copy_rules_path())}
    cand = ({S2.tsw_key(t, r["key"]) for t, v in S2.TSW.items() for r in C1.read_manifest(C1.manifest_path(v))}
            | {r["key"] for r in C1.read_manifest(S2.base_selected_path())})
    want_rule = {}
    for k in cand:
        if k.startswith(("tsw22__", "tsw23__")):
            sess_ = S1.session_of(k.split("__", 1)[1])
            want_rule[k] = ("tsw_eval_session" if (sess_ in dev_sessions or sess_ == "20210801_FakeCam_TEST")
                            else "tsw_provenance" if sess_[:1].isdigit() else "tsw_embed_v2")
        else:
            want_rule[k] = "l5_excluded" if k.startswith((CWP, VANPE)) else "base_b_embed_v2"
    check("L-9: copy_rules.jsonl names the rule of every tsw and base B candidate (%s)"
          % dict(__import__("collections").Counter(x["rule"] for x in rules.values())),
          set(rules) == cand - {BASE["hflip_test3"]["key"]} | {BASE["hflip_test3"]["key"]}
          and all(rules[k]["rule"] == want_rule[k] for k in rules) and set(rules) == cand
          and all(x["decided_by"].startswith("L-9") for x in rules.values())
          and s["l9"]["copy_rules"]["sha256"] == C1.sha256_file(S2.copy_rules_path()),
          sorted((k, rules.get(k, {}).get("rule"), v) for k, v in want_rule.items()
                 if rules.get(k, {}).get("rule") != v)[:5])
    check("... tsw rows of a capture session outside dev and test are tsw_provenance (the embedding threshold does "
          "not apply), a row without one tsw_embed_v2, dev / test sessions tsw_eval_session (dropped)",
          rules["tsw22__" + Y22_STEM]["rule"] == "tsw_provenance"
          and rules["tsw22__" + Y22_STEM]["embed_threshold_applies"] is False
          and rules["tsw22__field22_crop_test"]["rule"] == "tsw_embed_v2"
          and rules["tsw22__20210801_FakeCam_TEST_901"]["decision"] == "dropped"
          and rules["tsw22__20210801_FakeCam_TEST_901"]["reason"] == "test_session")
    prov_ = {p_["key"]: p_ for p_ in C1.read_manifest(C2.BASE_PROVENANCE)}
    check("... and the provenance file carries each kept row's copy rule (train_core: train_core)",
          all(p_["copy_rule"] == ("train_core" if p_["part"] == "train_core" else rules[k]["rule"])
              for k, p_ in prov_.items()))
    check("without a scan the v2 calibration is not applied (recorded)",
          s["embed_v2"]["applied"] is False and s["l9"]["v2_threshold"] is None, s["embed_v2"])
    check("tsw labels are byte copies of the v1 converted files under v2/labels/tsw22 (same sha256)",
          all(pathlib.Path(r["label"]).parent == S2.label_dir("tsw22")
              and pathlib.Path(r["label"]).read_bytes() == pathlib.Path(v1_22[r["key"]]["label"]).read_bytes()
              and r["label_sha256"] == v1_22[r["key"]]["label_sha256"] and r["sha256"] == v1_22[r["key"]]["sha256"]
              for r in rows22))
    exam_prefixes = tuple("%s__" % e for e in ("dev", "test", "imageweeds", "ood22", "ood23"))
    base = C1.read_manifest(C2.v2_manifest_path("base_v2"))
    check("no training row carries an exam key",
          not [r["key"] for m in ("tsw22", "tsw23", "base_v2", "train_core")
               for r in C1.read_manifest(C2.v2_manifest_path(m)) if r["key"].startswith(exam_prefixes)])
    b = s["base_b"]
    check("L-5: every cwp10 and vanpe image leaves base v2 (%s)" % b["l5"]["per_source"],
          b["l5"]["per_source"] == {CWP: 4, VANPE: 2} and b["l5"]["excluded"] == 6
          and not [r for r in base if r["source"] in S2.L5_SOURCES])
    l5 = C1.read_manifest(C2.L5_EXCLUDED)
    check("... listed in l5_excluded.jsonl with the decision", len(l5) == 6
          and all(r["decided_by"].startswith("L-5") for r in l5))
    check("an unreadable L-5 image is recorded as unchecked, not as a copy",
          [u["key"] for u in b["h6b"]["unchecked_l5_part"]] == [BASE["broken"]["key"]]
          and b["h6b"]["unchecked_l5_part"][0]["reason"] == "unhashable", b["h6b"]["unchecked_l5_part"])
    check("the flipped test image among them is an H6(b) incident, recorded, not refused",
          b["h6b"]["incident_l5_part"] is True and [i["key"] for i in b["h6b"]["incident"]]
          == [BASE["hflip_test3"]["key"]] and b["h6b"]["incident"][0]["reason"] == "near_eval_variant",
          b["h6b"])
    check("the harvested near copy of a train_core image is dropped (near_train_core)",
          b["dropped"] == {"near_train_core": 1} and b["kept"] == 7, b)
    parts = s["base_v2"]["parts"]
    check("base_v2 = train_core (minus the L-8 drop) + tsw22 + tsw23 + base B's kept part (%s)" % parts,
          len(base) == sum(parts.values()) and parts["train_core"] == len(C1.read_manifest(C1.manifest_path(
              "train_core"))) - 1 and parts["tsw22"] == t22["kept"] and parts["tsw23"] == t23["kept"]
          and parts["base_b"] == 7)
    bc_keys = {k for _h, _p, k in json.loads(C2.BASE_COPIES_INDEX.read_text())["entries"]}
    check("the L-8 image is in no v2 training manifest, not in the base-copy index nor the provenance, by key or "
          "by bytes", not [m for m in C2.V2_TRAIN_MANIFESTS for r in C1.read_manifest(C2.v2_manifest_path(m))
                           if r["key"] == vr["key"] or r["sha256"] == vr["sha256"]]
          and vr["key"] not in bc_keys
          and vr["key"] not in {p["key"] for p in C1.read_manifest(C2.BASE_PROVENANCE)})
    prov = C1.read_manifest(C2.BASE_PROVENANCE)
    check("... pairwise disjoint across parts (key, path, sha256, 6-bit dHash with variants)",
          S2.disjoint_problems([dict(p) for p in prov]) == [] and len(prov) == len(base))
    bc = json.loads(C2.BASE_COPIES_INDEX.read_text())
    check("the base-copy index lists every base v2 image, incomplete until lock",
          sorted(k for _h, _p, k in bc["entries"]) == sorted(r["key"] for r in base) and bc["complete"] is False)
    lic = s["licences"]["per_source"]
    check("licences: the card index (%s), the funnel config (%s), the Zenodo record (%s)"
          % (lic[IWA]["licence"], lic[WCD]["licence"][:10], lic["3seasonweeddet10/data2022"]["licence"]),
          lic[IWA]["licence"] == "cc-by-4.0" and "cards index" in lic[IWA]["basis"]
          and lic[WCD]["licence"].startswith("CC BY 4.0") and "domain config" in lic[WCD]["basis"]
          and lic["3seasonweeddet10/data2022"]["basis"].startswith("Zenodo record 14861516")
          and not any(v["research_only"] for k, v in lic.items() if k != "cottonweeddet12/train"))
    check("an unresolved licence (train_core here) is research_only, never silently",
          lic["cottonweeddet12/train"]["licence"] == "unresolved" and lic["cottonweeddet12/train"]["research_only"]
          and s["licences"]["unresolved_sources"] == ["cottonweeddet12/train"]
          and s["licences"]["research_only_rows"] == parts["train_core"] and "exemption" in s["licences"])
    check("provenance rows carry part, dHash, 8 variants, licence and lab group",
          all(len(p["variants"]) == 8 and p["variants"][0] == p["dhash"] for p in prov)
          and {p["lab_group"] for p in prov if p["part"] in ("train_core", "tsw22")} == {"LuLab"}
          and {p["lab_group"] for p in prov if p["source"] == IWA} == {"NDSU"})
    check("summary records every input by sha256 (v1 LOCK, v1 index, base_selected, the domain config...)",
          all(k in s["inputs"] and len(s["inputs"][k]["sha256"]) == 64 for k in
              ("v1_lock", "v1_nevertrain", "v1_summary", "base_selected", "select_summary", "funnel_domain",
               "cards_index", "tsw_zenodo_record", "scorer")) and s["testing"] is True)
    check("v1 is untouched by the build", tree_shas(INC / "splits" / "v1") == v1)


def test_scan_and_refusals():
    print("lock before the scan, the scan, and what it flags")
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "embed_scan.json missing")
    check("lock refuses before the embedding scan", ok, msg[:300])
    C2.write_json_atomic(S2.funnel_leak_path(), {"format": "funnel-leak/1", "status": "calibration_failed",
                                                 "detector": {"descriptor": {"embedder": "facebook/dinov2-base:cls"}},
                                                 "calibration": {"ok": False, "why": ["planted failure"],
                                                                 "cos_threshold": None}})
    emb = HueEmbedder()
    doc = S2.scan(embedder=emb, procs=1, testing=True)
    cal = json.loads((S2.leak_dir() / G.CAL_NAME).read_text())
    check("a failed funnel leak_v1.json is not reused: the stream's own calibration runs and passes",
          doc["detector"]["calibration_source"] == "inc2" and cal["calibration"]["ok"]
          and all(v.startswith("stream/v1/leak") for v in cal["seeds"].values()), doc["detector"])
    hit_keys = {(h["set"], h["key"]) for h in doc["hits"]}
    want = {("tsw22", "tsw22__field22_crop_test"), ("base_b", EXTRA["shear_test4"]["key"]),
            ("tsw22", "tsw22__field22_hflip_test"), ("tsw23", "tsw23__field23_rot_iw"),
            ("base_b", BASE["hflip_test3"]["key"]), ("tsw22", "tsw22__" + Y22_STEM)}
    check("found: the 15 % crop of a test image in tsw22 and the sheared test image in base B's kept part "
          "(embedding), the flipped / rotated copies (the detector's dHash half), and the 2022 capture of a test "
          "scene (a scene, not a copy: L-9(a) keeps it); nothing else",
          hit_keys == want, sorted(hit_keys ^ want))
    ec = json.loads(S2.embed_calibration_path().read_text())
    cal2 = ec["calibration"]
    hard = cal2["negatives"]["hard"]
    check("the scan wrote the v2 calibration (L-9(c)) on the scan's own calibration: per-image hard negatives, the "
          "two train_core frames of test image 3's scene are false hits at the scan's threshold %.4f and not at the "
          "v2 one %.4f (raised above them, never below the base)" % (cal2["floor"], cal2["cos_threshold"]),
          ec["format"] == EC.FORMAT and cal2["ok"] and cal2["base"]["source"] == "inc2"
          and cal2["cos_threshold"] >= cal2["floor"] == doc["detector"]["cos_threshold"]
          and hard["at_base"]["false_hits"] == 2 and hard["false_hits"] == 0 and hard["n"] >= 30
          and hard["excluded_dhash_copies"] == 1 and cal2["base"]["positives"] == "reproduced from its seeds"
          and cal2["base"]["recomputed_threshold"] == cal2["floor"], (cal2["floor"], cal2["cos_threshold"], hard))
    check("... every family's recall recorded at the new threshold and at the base's; the one-sided 97.5 % upper "
          "bounds reported per tier", all(set(v) >= {"recall", "lb", "at_base"} for v in cal2["positives"].values())
          and all(cal2["negatives"][t]["ub"] is not None for t in ("hard", "provenance_disjoint"))
          and cal2["negatives"]["easy"]["n"] == 0 and cal2["negatives"]["easy"].get("absent_from_pool"))
    neg_text = (C2.SPLITS_DIR / EC.NEGATIVES_NAME).read_text()
    check("... its negatives file names evaluation keys, never an evaluation image path",
          "test__20210801_FakeCam_TEST_3" in neg_text and not [p_ for x in C2.EVAL_SPLITS
                                                              for p_ in (r["image"] for r in C1.read_manifest(
                                                                  C1.manifest_path(x))) if p_ in neg_text])
    check("the unreadable L-5 image is listed as unscannable, nothing else",
          doc["unscannable"] == [{"set": "base_b", "key": BASE["broken"]["key"]}], doc["unscannable"])
    far = {}
    for k, img, src in (("crop", str(REPO / "downloads" / "3seasonweeddet10" / "data2022" / "fieldA" /
                                     "field22_crop_test.png"), test_image(2)),
                        ("shear", EXTRA["shear_test4"]["image"], test_image(4))):
        far[k] = min(bits(G.dhash_variants(img)[v], C1.dhash(src)) for v in G.VARIANTS)
    check("... the crop and the shear more than 6 bits from their original under all 8 variants %s" % far,
          all(v > 6 for v in far.values()))
    n22 = len(C1.read_manifest(C1.manifest_path("ood22")))
    check("the scan covers every candidate, not the build's manifests: all %d ood22 rows under tsw keys, "
          "all of base B" % n22,
          len(doc["scanned"]["tsw22"]) == n22 and len(doc["scanned"]["tsw23"]) == 4
          and len(doc["scanned"]["base_b"]) == len(BASE) + 1 and doc["testing"] is True)
    text = S2.embed_scan_path().read_text()
    eval_paths = [r["image"] for x in C2.EVAL_SPLITS for r in C1.read_manifest(C1.manifest_path(x))]
    check("embed_scan.json carries evaluation keys, never an evaluation image path",
          not [p for p in eval_paths if p in text] and all("eval_key" in h for h in doc["hits"]))
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "flagged by the embedding scan")
    check("lock refuses: base v2 rows are flagged", ok, msg[:300])
    made = len(MADE)
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1), "R4 INCIDENT")
    check("a second build refuses: the harvested copy is an R4 incident", ok and "shear_test4" in msg, msg[:400])
    check("... using the scan that covers its candidates (no second scan)", len(MADE) == made, len(MADE))


def funnel_doc(own, listed=(), copies=None, pairs=None):
    """A funnel leak_v1.json (status complete) whose base_B scan lists `listed`."""
    listed = list(listed)
    doc = {"format": "funnel-leak/1", "status": "complete",
           "h6b": {"base_copy": bool(copies if copies is not None else listed), "incident": bool(listed)},
           "detector": {"descriptor": {"embedder": "facebook/dinov2-base:cls"}},
           "calibration": dict(own["calibration"]),
           "scans": {"base_B": {"images": len(BASE), "copies": len(listed) if copies is None else copies,
                                "listed": listed}}}
    if pairs is not None:
        doc["pairs_csv"] = pairs
    return doc


def listed_entry(row, split="test", key="test__x", cos=0.99, bits_=30):
    return {"key": row["key"], "image": row["image"], "eval_split": split, "eval_key": key, "cos": cos,
            "bits": bits_, "variant": "id"}


INCIDENT_THETA = 0.80


def incident_funnel_doc(listed):
    """The funnel's leak_v1.json of the real incident (job 47259471), in this
    world: complete, a passed calibration record at a per-pair threshold
    (0.80) below the world's same-scene cosines (0.853 harvested, 0.954
    train_core), no seeds recorded; its base_B list holds the given entries."""
    cal = {"ok": True, "why": [], "cos_threshold": INCIDENT_THETA, "dhash_bits_max": 6,
           "recall_min": L.RECALL_MIN, "fpr_max": L.FPR_MAX,
           "positives": {f: {"n": 40, "hits": 40} for f in L.FAMILIES},
           "negatives": {"pairs_7_10": {"n": 200, "false_hits": 0}, "hard": {"n": 200, "false_hits": 1}}}
    return {"format": "funnel-leak/1", "status": "complete", "h6b": {"base_copy": bool(listed), "incident": False},
            "detector": {"descriptor": {"embedder": "facebook/dinov2-base:cls"}}, "calibration": cal,
            "scans": {"base_B": {"images": len(BASE), "copies": len({e["key"] for e in listed}), "listed": listed}}}


def test_final_build_and_lock(v1):
    print("the platform's sequence: build (it scans by itself), lock")
    write_base_selected([])
    os.unlink(S2.embed_scan_path())
    made = len(MADE)
    rc = S2.main(["build", "--testing", "--scorer", str(SCORER), "--procs", "3"])
    s = S2.read_summary()
    check("build with no embed_scan.json runs the embedding scan itself (L23 is build then lock; no scan verb)",
          rc == 0 and len(MADE) == made + 1 and S2.embed_scan_path().is_file() and s["embed_scan"]["applied"],
          (rc, len(MADE) - made))
    check("... and applies it: the tsw22 crop is dropped (near_eval_embed)",
          s["tsw"]["tsw22"]["dropped"].get("near_eval_embed") == 1, s["tsw"]["tsw22"]["dropped"])
    doc = json.loads(S2.embed_scan_path().read_text())
    check("... the scan covers the current candidates (base B without the sheared image)",
          len(doc["scanned"]["base_b"]) == len(BASE) and doc["detector"]["calibration_source"] == "inc2")
    made = len(MADE)
    S2.build(testing=True, scorer_path=SCORER, procs=1)
    check("a build whose candidates the scan covers does not scan again", len(MADE) == made)
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="bogus"), "scan_mode")
    check("an unknown scan_mode refuses", ok, msg)

    print("the funnel's leak_v1.json: base B copies it lists")
    own = json.loads((S2.leak_dir() / G.CAL_NAME).read_text())
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, [listed_entry(BASE["w0"])]))
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1), "R4 INCIDENT")
    check("a kept base B image the funnel lists as a copy refuses the build (R4)",
          ok and BASE["w0"]["key"] in msg and S2.FUNNEL_REASON in msg, msg[:400])
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, [], copies=1,
                                                           pairs={"path": str(TMP / "no_pairs.csv"), "sha256": None}))
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1), "fail closed")
    check("a truncated listing whose pairs file is missing refuses (fail closed)", ok, msg[:300])
    pairs = TMP / "leak_pairs_v1.csv"
    pairs.write_text("set,key,eval_split,eval_key,cos,bits,variant,kind\n"
                     "base_B,%s,test,test__x,0.990000,30,id,copy\n"
                     "calibration:a|b,zz,a,b,0.100000,8,id,negative_7_10\n" % BASE["w1"]["key"])
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, [], copies=1,
                                                           pairs={"path": str(pairs), "sha256": C1.sha256_file(pairs)}))
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1), "R4 INCIDENT")
    check("... and one whose pairs file names a kept base B image refuses (R4)", ok and BASE["w1"]["key"] in msg,
          msg[:300])
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, [listed_entry(BASE["c0"])]))
    S2.build(testing=True, scorer_path=SCORER, procs=1)
    s = S2.read_summary()
    check("a copy the funnel lists in the L-5 part is recorded as the incident, not refused",
          BASE["c0"]["key"] in {i["key"] for i in s["base_b"]["h6b"]["incident"]}
          and s["funnel_leak_v1"]["state"] == "complete", s["base_b"]["h6b"]["incident"])
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, [listed_entry(BASE["w2"])]))
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "funnel H6(b)")
    check("lock refuses when leak_v1.json lists a base v2 image, even after a clean build", ok, msg[:300])

    print("the incident of job 47259471 (L-9): the funnel's per-pair threshold flags scenes, not copies")
    ood23_key = sorted(r["key"] for r in C1.read_manifest(C1.manifest_path("ood23")))[0]
    inc_listed = [listed_entry(BASE["w0"], "ood23", ood23_key, 0.84, 22),          # an exam in v1, training in v2
                  listed_entry(BASE["w1"], "test", "test__20210801_FakeCam_TEST_2", 0.83, 27),
                  listed_entry(BASE["a0"], "imageweeds", "imageweeds__iw_1", 0.828, 27)]
    C2.write_json_atomic(S2.funnel_leak_path(), incident_funnel_doc(inc_listed))
    write_base_selected(["shear_test4"])
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1), "R4 INCIDENT")
    check("with the incident's calibration as the base, the sheared test image in base B's kept part (an augmented "
          "copy) still refuses the build, judged at the v2 threshold",
          ok and EXTRA["shear_test4"]["key"] in msg and "near_eval_embed" in msg and "v2 threshold" in msg, msg[:500])
    ec = json.loads(S2.embed_calibration_path().read_text())["calibration"]
    scene_cos = {}
    emb_ = HueEmbedder()
    with Image.open(test_image(3)) as im3, Image.open(BASE["scene_test3"]["image"]) as imb:
        e3, eb = emb_([im3.convert("RGB")])[0], emb_([imb.convert("RGB")])[0]
    scene_cos["base_b"] = float(e3 @ eb)
    check("the v2 calibration on the incident's base: floor %.2f, raised to %.4f, above the harvested scene "
          "(cos %.3f) and the train_core scene frames (their hard-tier false hits: %d at %.2f, 0 at v2); the base's "
          "positives are drawn with this stream's seeds (it records none)"
          % (INCIDENT_THETA, ec["cos_threshold"], scene_cos["base_b"], ec["negatives"]["hard"]["at_base"]["false_hits"],
             INCIDENT_THETA),
          ec["floor"] == INCIDENT_THETA and ec["base"]["source"] == "funnel_leak_v1"
          and ec["cos_threshold"] > max(scene_cos["base_b"], 0.95) and ec["negatives"]["hard"]["false_hits"] == 0
          and ec["negatives"]["hard"]["at_base"]["false_hits"] >= 2 and ec["negatives"]["hard"]["at_base"]["fpr"] > 0.01
          and "stream/v1/leak" in ec["base"]["positives"], ec)
    write_base_selected([])

    print("without the harvested copy: scan (funnel threshold), build, lock")
    doc = S2.scan(embedder=HueEmbedder(), procs=1, testing=True)
    check("a passed funnel leak_v1.json is reused and recorded",
          doc["detector"]["calibration_source"] == "funnel_leak_v1"
          and doc["detector"]["calibration"]["path"] == str(S2.funnel_leak_path()), doc["detector"])
    flagged = {h["key"] for h in doc["hits"]}
    check("the scan at the incident's threshold flags the harvested scene of a test image and the tsw22 capture "
          "of one (scenes, not copies)", BASE["scene_test3"]["key"] in flagged and "tsw22__" + Y22_STEM in flagged,
          sorted(flagged))
    fdoc = json.loads(S2.funnel_leak_path().read_text())
    C2.write_json_atomic(S2.funnel_leak_path(), dict(fdoc, note="rewritten by a funnel rerun"))
    made = len(MADE)
    S2.build(testing=True, scorer_path=SCORER, procs=1)
    new_sha = C1.sha256_file(S2.funnel_leak_path())
    check("a scan whose calibration file changed since it ran is stale: build scans again (embed_scan.json now "
          "records the new file) and makes the v2 calibration again on it",
          len(MADE) == made + 1
          and json.loads(S2.embed_scan_path().read_text())["detector"]["calibration"]["sha256"] == new_sha
          and json.loads(S2.embed_calibration_path().read_text())["calibration"]["base"]["file"]["sha256"] == new_sha,
          len(MADE) - made)
    C2.write_json_atomic(S2.funnel_leak_path(), dict(fdoc, note="rewritten again"))
    ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="skip"), "R4 INCIDENT")
    check("... while build --skip-scan applies the stale scan at the scan's own threshold, without a v2 calibration: "
          "the incident recurs (the harvested scene and the funnel's test and imageweeds matches refuse, R4), but "
          "not through the ood23 match (L-9(b) needs no calibration)",
          ok and "no v2 calibration" in msg and BASE["scene_test3"]["key"] in msg and BASE["w1"]["key"] in msg
          and BASE["a0"]["key"] in msg and BASE["w0"]["key"] not in msg and len(MADE) == made + 1, msg[:600])
    C2.write_json_atomic(S2.funnel_leak_path(), fdoc)
    doc = S2.scan(embedder=HueEmbedder(), procs=1, testing=True)
    table = TMP / "licences.json"
    table.write_text(json.dumps({"sources": {"cottonweeddet12": {"licence": "CC BY 4.0",
                                                                  "evidence": "a person's reading of the record"}}}))
    made = len(MADE)
    rc = S2.main(["build", "--testing", "--scorer", str(SCORER), "--procs", "1", "--licences", str(table)])
    s = S2.read_summary()
    check("an owner licence table resolves train_core (recorded with its evidence)",
          rc == 0 and s["licences"]["per_source"]["cottonweeddet12/train"]["licence"] == "CC BY 4.0"
          and "owner table" in s["licences"]["per_source"]["cottonweeddet12/train"]["basis"]
          and s["licences"]["research_only_rows"] == 0 and "licences" in s["inputs"] and len(MADE) == made)
    rules = {x["key"]: x for x in C1.read_manifest(S2.copy_rules_path())}
    in_base = {r["key"] for r in C1.read_manifest(C2.v2_manifest_path("base_v2"))}
    sc = rules[BASE["scene_test3"]["key"]]
    check("L-9(c): the harvested scene of a test image, flagged at the incident's threshold (cos %.3f), is below "
          "the v2 threshold: recorded, not applied, kept (the build no longer refuses on it)"
          % sc["embed"]["base_hit"]["cos"],
          sc["rule"] == "base_b_embed_v2" and sc["decision"] == "kept" and sc["embed"]["base_hit"]
          and not sc["embed"]["applied_hit"] and BASE["scene_test3"]["key"] in in_base, sc)
    y = rules["tsw22__" + Y22_STEM]
    check("L-9(a): the tsw22 capture of a test scene (another capture session) is kept under tsw_provenance, "
          "though the scan flags it at both thresholds (cos %.3f)" % y["embed"]["base_hit"]["cos"],
          y["rule"] == "tsw_provenance" and y["decision"] == "kept" and y["embed"]["applied_hit"]
          and y["embed_threshold_applies"] is False and "tsw22__" + Y22_STEM in in_base, y)
    fv = {k: [e["verdict"] for e in rules[BASE[k]["key"]].get("funnel") or []] for k in ("w0", "w1", "a0")}
    check("L-9(b): the funnel's matches are read: against ood23 not a v2 leak (not_v2_split), against test and "
          "imageweeds below the v2 threshold (below_v2_threshold); none applied, all three images kept",
          fv == {"w0": ["not_v2_split"], "w1": ["below_v2_threshold"], "a0": ["below_v2_threshold"]}
          and all(BASE[k]["key"] in in_base for k in ("w0", "w1", "a0"))
          and s["l9"]["base_b"]["funnel_images"] == {"applied": 0, "not_v2_split": 1, "below_v2_threshold": 2}, fv)
    ev2 = s["embed_v2"]
    check("summary.json records the v2 calibration applied (its base, floor, per-tier rates at both thresholds, "
          "recall, known limits) and per source the hits against the count the per-image rate predicts",
          ev2["applied"] and ev2["base"]["cos_threshold"] == INCIDENT_THETA and ev2["cos_threshold"] > 0.95
          and ev2["negatives"]["hard"]["at_base"]["false_hits"] >= 2 and "per_source" in ev2
          and all("expected_false_hits" in v for v in ev2["per_source"].values())
          and s["l9"]["tsw22"]["flagged_exempt_kept"] >= 1, ev2.get("negatives"))
    check("... and funnel_leak_v1 records how many of its matches are against each split",
          s["funnel_leak_v1"]["entries_by_split"] == {"imageweeds": 1, "ood23": 1, "test": 1},
          s["funnel_leak_v1"])
    scan_bytes = S2.embed_scan_path().read_bytes()
    doc = json.loads(scan_bytes)
    in_base = {r["key"] for r in C1.read_manifest(C2.v2_manifest_path("base_v2"))}
    i = next(j for j, (k, _sha) in enumerate(doc["scanned"]["base_b"]) if k in in_base)
    dropped_key = doc["scanned"]["base_b"].pop(i)[0]
    C2.write_json_atomic(S2.embed_scan_path(), doc)
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "were not scanned")
    check("lock refuses when embed_scan.json leaves a base v2 row unscanned", ok and dropped_key in msg, msg[:300])
    S2.embed_scan_path().write_bytes(scan_bytes)

    print("lock re-checks the L-9 copy rules and the v2 calibration")
    cpath, rpath, ppath = S2.embed_calibration_path(), S2.copy_rules_path(), C2.BASE_PROVENANCE
    c_bytes, r_bytes, p_bytes = cpath.read_bytes(), rpath.read_bytes(), ppath.read_bytes()
    cdoc = json.loads(c_bytes)
    cdoc["calibration"]["cos_threshold"] = cdoc["calibration"]["floor"]      # still a record that passes its gates
    C2.write_json_atomic(cpath, cdoc)
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "cannot lock")
    check("lock refuses a v2 calibration other than the one the build applied (here its threshold lowered to the "
          "floor)", ok and "the build applied another v2 calibration" in msg and not C2.LOCK_PATH.exists(), msg[:400])
    cpath.write_bytes(c_bytes)
    rpath.write_bytes(r_bytes + b"\n")
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "cannot lock")
    check("lock refuses copy_rules.jsonl changed since the build", ok and "copy_rules.jsonl is missing or changed" in msg,
          msg[:400])
    rpath.write_bytes(r_bytes)
    prov_rows = C1.read_manifest(ppath)
    j = next(i for i, p_ in enumerate(prov_rows) if p_["part"] == "base_b")
    prov_rows[j]["copy_rule"] = "tsw_provenance"
    C2.write_jsonl_atomic(ppath, prov_rows)
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "cannot lock")
    check("lock re-derives every row's copy rule: a base B row the provenance file calls tsw_provenance (exempt from "
          "the embedding threshold) refuses", ok and "do not carry the copy rule" in msg, msg[:400])
    ppath.write_bytes(p_bytes)

    print("lock re-checks the L-8 list")
    dpath, spath = C2.TRAIN_CORE_VARIANT_DROPS, S2.summary_path()
    d_bytes, s_bytes = dpath.read_bytes(), spath.read_bytes()
    dpath.write_bytes(d_bytes + b"\n")
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "changed since the build")
    check("lock refuses an L-8 list changed after the build", ok and C2.VARIANT_DROPS_NAME in msg, msg[:300])
    dpath.write_bytes(d_bytes)
    clean = v1_core()[0]
    h, v = G.image_hashes(clean["image"])
    forged = dict({k: clean[k] for k in C1.MANIFEST_KEYS}, dhash=h, variants=[x for _n, x in G.variant_list(v)],
                  reason="near_eval_variant", match={"split": "test", "key": "test__x", "bits": 0,
                                                     "variant": "transverse"}, decided_by=S2.L8_DECISION)
    fsha = C2.write_jsonl_atomic(dpath, [forged])
    summ = json.loads(s_bytes)
    summ["train_core_variant_drops"].update(sha256=fsha, keys=[clean["key"]])
    C2.write_json_atomic(spath, summ)
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "cannot lock")
    check("lock refuses a forged list naming a clean train_core row, even with a matching summary: the guard does "
          "not refuse the row, and v2's train_core is not v1's minus it",
          ok and "is not refused as near_eval_variant" in msg and "not v1's train_core minus" in msg
          and not C2.LOCK_PATH.exists(), msg[:600])
    dpath.write_bytes(d_bytes)
    spath.write_bytes(s_bytes)
    rc = S2.main(["lock", "--scorer", str(SCORER), "--procs", "3"])
    check("lock without --testing refuses a --testing build (the platform cannot seal a synthetic world)",
          rc == 2 and not C2.LOCK_PATH.exists())
    rc = S2.main(["lock", "--testing", "--scorer", str(SCORER), "--procs", "3"])
    check("lock --testing exits 0", rc == 0)
    lk = C2.read_lock_v2()
    check("LOCK v2 records the seven manifests", set(lk["manifests"]) == set(C2.V2_MANIFESTS)
          and all(lk["manifests"][x] == C1.sha256_file(C2.v2_manifest_path(x)) for x in C2.V2_MANIFESTS))
    check("... derived_from: the v1 LOCK, byte copies identical",
          lk["derived_from"]["v1_lock_sha256"] == C1.sha256_file(C1.LOCK_PATH)
          and set(lk["derived_from"]["identical"]) == set(C2.BYTE_COPIES) == {"dev", "test", "imageweeds"}
          and all(lk["manifests"][x] == C1.read_lock()["manifests"][x] for x in C2.BYTE_COPIES))
    der = lk["derived_from"]["train_core"]
    check("... train_core derived as v1 minus the L-8 list, whose sha256 LOCK v2 records with its keys",
          der["v1_sha256"] == C1.read_lock()["manifests"]["train_core"] == lock_v1_core_sha()
          and der["sha256"] == lk["manifests"]["train_core"] != der["v1_sha256"]
          and der["minus_sha256"] == lk[C2.VARIANT_DROPS_LOCK_KEY] == C1.sha256_file(C2.TRAIN_CORE_VARIANT_DROPS)
          and der["dropped"] == 1 and lk["train_core_variant_drops"]["keys"] == [variant_row()["key"]]
          and lk["train_core_variant_drops"]["incident"] is True, (der, lk.get("train_core_variant_drops")))
    check("... the index shas and counts, the scorer, eval splits",
          lk["nevertrain_sha256"] == C1.sha256_file(C2.NEVER_TRAIN_INDEX)
          and lk["base_copies_sha256"] == C1.sha256_file(C2.BASE_COPIES_INDEX)
          and lk["scorer_sha256"] == C1.read_lock()["scorer_sha256"] and lk["eval_splits"] == list(C2.EVAL_SPLITS))
    check("... the provenance file both as provenance_sha256 and as provenance {sha256} (the stream reads the latter)",
          lk["provenance_sha256"] == lk["provenance"]["sha256"] == C1.sha256_file(C2.BASE_PROVENANCE))
    check("... the H6 status: the scan (funnel threshold, 0 copies in base v2), the L-5 incident, leak_v1",
          lk["h6"]["base_b"]["embed_scan"]["copies_in_base_v2"] == 0
          and lk["h6"]["base_b"]["embed_scan"]["calibration_source"] == "funnel_leak_v1"
          and lk["h6"]["base_b"]["incident_in_l5_part"] is True
          and lk["h6"]["base_b"]["funnel_leak_v1"]["status"] == "complete" and lk["testing"] is True
          and lk["h6_status"]["funnel_leak_v1"] == "complete", lk["h6"])
    ec2 = lk.get("embed_calibration_v2") or {}
    check("... L-9: the v2 calibration (sha256, threshold above the incident's 0.80, its base), its negatives file "
          "and copy_rules.jsonl by sha256; the H6 block carries the v2 threshold",
          lk[EC.LOCK_KEY] == C1.sha256_file(S2.embed_calibration_path()) == ec2["sha256"]
          and ec2["cos_threshold"] > 0.95 and ec2["base"]["source"] == "funnel_leak_v1"
          and ec2["base"]["cos_threshold"] == INCIDENT_THETA
          and lk["embed_calibration_v2_negatives_sha256"] == C1.sha256_file(C2.SPLITS_DIR / EC.NEGATIVES_NAME)
          and lk["copy_rules_sha256"] == C1.sha256_file(S2.copy_rules_path())
          and lk["h6"]["base_b"]["embed_scan"]["cos_threshold_v2"] == ec2["cos_threshold"]
          and lk["h6_status"]["embed_v2_threshold"] == ec2["cos_threshold"] and lk["l9"]["rules"]["tsw22"], ec2)
    check("... and EC.locked reads it back through LOCK v2", EC.locked(C2.LOCK_PATH)[0]["cos_threshold"]
          == ec2["cos_threshold"])
    nt = json.loads(C2.NEVER_TRAIN_INDEX.read_text())
    check("both indexes are complete now", nt["complete"] is True and nt["min_expected"] == len(nt["entries"])
          and json.loads(C2.BASE_COPIES_INDEX.read_text())["complete"] is True)
    files = S2._walk_files(C2.SPLITS_DIR)
    writable = [f for f in files if os.stat(f).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)]
    check("every file under splits/v2 is 0444 (%d files)" % len(files), not writable, writable[:5])
    probs = S2.verify(scorer_path=SCORER, procs=3)
    check("verify is [] for v2 and v1 (files hashed by 3 worker processes)", probs == [], probs[:5])
    check("v1 verify is still [] and no v1 file changed",
          S1.verify(scorer_path=SCORER) == [] and tree_shas(INC / "splits" / "v1") == v1)


def test_after_lock():
    print("after lock: the guard, refusals, tamper checks")
    g = G.GuardV2.load(C2.LOCK_PATH)
    d22 = REPO / "downloads" / "3seasonweeddet10" / "data2022" / "fieldA"
    d23 = REPO / "downloads" / "3seasonweeddet10" / "data2023"
    core = C1.read_manifest(C2.v2_manifest_path("train_core"))
    r_flip = g.check_path(d22 / "field22_hflip_test.png")
    r_dup = g.check_path(d23 / "field23_dup22.png")
    r_core = g.check_path(core[0]["image"])
    r_w = g.check_path(BASE["w0"]["image"])
    r_fresh = g.check_path(save(render(grid(5555), next(TRIPLES)), TMP / "fresh.png"))
    check("GuardV2.load: a flipped test image is near_eval_variant", r_flip[0] == "near_eval_variant"
          and r_flip[1]["split"] == "test", r_flip[:2])
    check("... the dropped ood23 duplicate is a base copy (of its tsw22 twin)",
          r_dup[0] == "base_copy" and r_dup[1]["key"] == "tsw22__field22_a", r_dup[:2])
    check("... a train_core image and a kept harvested image are base copies",
          r_core[0] == "base_copy" and r_w[0] == "base_copy")
    check("... a fresh image passes", r_fresh[:2] == (None, None))
    nt2 = C2.nevertrain_v2()
    check("nevertrain_v2 loads the locked index",
          nt2.n == len(json.loads(C2.NEVER_TRAIN_INDEX.read_text())["entries"]))
    flip = d22 / "field22_hflip_test.png"
    v1_hits, _u = C1.NeverTrainGuard.check(nt2, [flip])
    v2_hits, v2_un = nt2.check([flip])
    check("nevertrain_v2's check refuses the flipped test image that the v1 (stored dHash only) check passes",
          v1_hits == [] and [h[1] for h in v2_hits] == ["test"] and not v2_un, (v1_hits, v2_hits))
    check("... and its assert_trainable raises on it, passing a fresh image",
          raises(lambda: nt2.assert_trainable([flip]), "flips and rotations", RuntimeError)[0]
          and nt2.assert_trainable([TMP / "fresh.png"]) is True)
    test_after_lock_variant_drop(g, nt2)
    prov = C1.read_manifest(C2.BASE_PROVENANCE)
    ents = [(int(h), sp, k) for h, sp, k in json.loads(C2.NEVER_TRAIN_INDEX.read_text())["entries"]]
    bad = dict(prov[0], variants=[prov[0]["dhash"], C1.dhash(test_image(2))] + prov[0]["variants"][2:])
    check("lock's own never-train re-check passes the locked base and names a row whose recorded flip is a "
          "test image", S2._never_train_problems(prov, ents) == []
          and "refused by the never-train guard" in " ".join(S2._never_train_problems([bad], ents)))
    check("build refuses once locked", raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1),
                                              "locked")[0])
    check("scan refuses once locked", raises(lambda: S2.scan(embedder=HueEmbedder(), procs=1, testing=True),
                                             "locked")[0])
    check("a second lock refuses", raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True),
                                          "new version")[0])
    lbl = pathlib.Path(C1.read_manifest(C2.v2_manifest_path("tsw22"))[0]["label"])
    orig = lbl.read_bytes()
    os.chmod(lbl, 0o644)
    lbl.write_bytes(orig + b"0 0.5 0.5 0.1 0.1\n")
    probs = S2.verify(scorer_path=SCORER, check_v1=False, procs=1)
    check("verify catches a changed label and a writable file",
          any("changed" in p and str(lbl) in p for p in probs) and any("writable" in p for p in probs), probs[:4])
    lbl.write_bytes(orig)
    os.chmod(lbl, 0o444)
    check("verify passes again once restored", S2.verify(scorer_path=SCORER, check_v1=False, procs=1) == [])
    for f_ in (S2.embed_calibration_path(), S2.copy_rules_path(), C2.SPLITS_DIR / EC.NEGATIVES_NAME):
        orig = f_.read_bytes()
        os.chmod(f_, 0o644)
        f_.write_bytes(orig + b" ")
        probs = S2.verify(scorer_path=SCORER, check_v1=False, procs=1)
        e_ = raises(lambda: EC.locked(C2.LOCK_PATH), "hashes to", C2.Inc2Error)
        f_.write_bytes(orig)
        os.chmod(f_, 0o444)
        check("verify catches %s changed after lock%s" % (f_.name, " (and EC.locked refuses it)"
                                                          if f_.name == EC.NAME else ""),
              any("changed since it was locked" in p and f_.name in p for p in probs)
              and (f_.name != EC.NAME or e_[0]), probs[:3])
    check("verify passes again once restored", S2.verify(scorer_path=SCORER, check_v1=False, procs=1) == [])
    check("CLI verify and summary exit 0",
          S2.main(["verify", "--scorer", str(SCORER), "--procs", "1"]) == 0 and S2.main(["summary"]) == 0)
    check("the writer lock is released", not S2.writer_lock_path().exists())


def test_after_lock_variant_drop(g, nt2):
    print("after lock: the L-8 image is refused under any key")
    vr = variant_row()
    r = g.check_path(vr["image"])
    check("GuardV2.load refuses the dropped train_core image: near_eval_variant, the test image, transverse",
          r[0] == "near_eval_variant" and r[1]["split"] == "test" and r[1]["variant"] == "transverse", r[:2])
    hits, un = nt2.check([vr["image"]])
    check("... the v2 NeverTrainGuard (inc2.common) too, and its assert_trainable raises",
          [h[1] for h in hits] == ["test"] and not un
          and raises(lambda: nt2.assert_trainable([vr["image"]]), "flips and rotations", RuntimeError)[0])
    from weed_optimizer_framework.tools.inc2 import train as T
    check("inc2.train reads the L-8 list LOCK v2 records (%s)" % T.VARIANT_DROPS_NAME,
          T.VARIANT_DROPS_NAME == C2.VARIANT_DROPS_NAME and T.VARIANT_DROPS_LOCK_KEY == C2.VARIANT_DROPS_LOCK_KEY)
    base = C1.read_manifest(C2.v2_manifest_path("base_v2"))
    relisted = dict({k: vr[k] for k in C1.MANIFEST_KEYS}, key="elsewhere__relisted_0001", source="some_reupload")
    m = TMP / "relisted.jsonl"
    C1.write_manifest(m, base + [relisted])
    rows, dh, _info = T.check_manifest(m)
    try:
        T.guard_rows(rows, dh, production=False)
        e = None
    except T.RunError as x:
        e = x
    gr = getattr(e, "guard", {}) or {}
    check("inc2.train.guard_rows refuses its bytes re-listed under another key and source: the L-8 list "
          "(train_core_variant_drop) and GuardV2 (near_eval_variant)",
          e is not None and e.stage == "guard" and gr.get("reasons", {}).get("train_core_variant_drop") == 1
          and gr.get("reasons", {}).get("near_eval_variant") == 1 and gr.get("refused") == 2
          and (gr.get("train_core_variant_drops") or {}).get("keys") == [vr["key"]], (e, gr.get("reasons")))
    rows, dh, _info = T.check_manifest(C2.v2_manifest_path("base_v2"))
    rec = T.guard_rows(rows, dh, production=False)
    check("... while base_v2 passes, the L-8 list recorded in the guard record",
          rec["refused"] == 0 and rec["train_core_variant_drops"]["sha256"]
          == C1.sha256_file(C2.TRAIN_CORE_VARIANT_DROPS), rec.get("train_core_variant_drops"))
    try:
        T.guard_rows(rows, dh, production=True)
        e = None
    except T.RunError as x:
        e = x
    check("... (a production run refuses this testing LOCK)", e is not None and "testing build" in str(e), e)
    d = C2.TRAIN_CORE_VARIANT_DROPS
    orig = d.read_bytes()
    os.chmod(d, 0o644)
    d.write_bytes(orig + b"\n")
    try:
        probs = S2.verify(scorer_path=SCORER, check_v1=False, procs=1)
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    finally:
        d.write_bytes(orig)
        os.chmod(d, 0o444)
    check("verify catches an L-8 list changed after lock, and inc2.train stops on it",
          any("changed since it was locked" in p and C2.VARIANT_DROPS_NAME in p for p in probs)
          and e is not None and e.stage == "guard" and C2.VARIANT_DROPS_NAME in str(e), (probs[:3], e))
    lk = json.loads(C2.LOCK_PATH.read_text())
    try:
        os.chmod(C2.LOCK_PATH, 0o644)
        C2.write_json_atomic(C2.LOCK_PATH, dict((k, v) for k, v in lk.items() if k != C2.VARIANT_DROPS_LOCK_KEY))
        rows_, rec_ = C2.read_variant_drops(production=False)
        e1 = raises(lambda: C2.read_variant_drops(production=True), "records no")[0]
    finally:
        C2.write_json_atomic(C2.LOCK_PATH, lk)
        os.chmod(C2.LOCK_PATH, 0o444)
    check("a LOCK that records no L-8 list: read_variant_drops refuses in production, gives none when testing",
          e1 and rows_ == [] and rec_["sha256"] is None)
    check("verify passes again once restored", S2.verify(scorer_path=SCORER, check_v1=False, procs=1) == [])


def test_job_script():
    print("run_inc2_splits.sh")
    text = SCRIPT.read_text()
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    check("bash -n passes", r.returncode == 0, r.stderr)
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    sbatch = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    check("GPU-shared with one V100, never RM-shared",
          "#SBATCH --partition=GPU-shared" in sbatch and "#SBATCH --gres=gpu:v100-32:1" in sbatch
          and not any("RM" in ln for ln in sbatch) and "RM-shared" not in code and "-p RM" not in code, sbatch)
    check("no git reset / checkout, no rsync, no copy of the package",
          not any(w in code for w in ("git reset", "git checkout", "rsync", "cp -r", "cp -a", "cp ")), code[:200])
    mods = [ln for ln in text.split("MODULES=(", 1)[1].split(")", 1)[0].split()]
    need = {"tools/inc2/common.py", "tools/inc2/guard.py", "tools/inc2/splits.py", "tools/inc/splits.py",
            "tools/funnel/leak.py", "tools/funnel/embed.py", "tools/mega_trainer.py", "tools/funnel/domain.py",
            "tools/funnel/domains/weed.json", "tools/inc2/embed_calibration.py", "tools/funnel/estimate.py"}
    check("the drift check covers the inc2, splits, funnel leak/embed and dHash modules", need <= set(mods),
          sorted(need - set(mods)))
    pkg = ROOT / "weed_optimizer_framework"
    check("every module the drift check names exists", all((pkg / m).is_file() for m in mods),
          [m for m in mods if not (pkg / m).is_file()])
    repo = TMP / "jobrepo"
    (repo / "results" / "framework" / "inc" / "logs").mkdir(parents=True)
    os.symlink(ROOT, repo / "weed_llm_benchmark")
    for m in mods:
        dst = repo / "weed_optimizer_framework" / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pkg / m, dst)
    fake = TMP / "fakebin"
    fake.mkdir()
    os.symlink(sys.executable, fake / "python")
    conda = TMP / "conda.sh"
    conda.write_text("conda() { return 0; }\n")

    def job(args, gpu=True, env=None):
        (fake / "nvidia-smi").write_text("#!/bin/sh\n%s\n" % ("echo 'Tesla V100-SXM2-32GB, 32768 MiB'" if gpu
                                                               else "exit 9"))
        os.chmod(fake / "nvidia-smi", 0o755)
        e = dict(os.environ, PATH="%s:%s" % (fake, os.environ["PATH"]), INC2_SPLITS_REPO=str(repo),
                 INC2_SPLITS_CONDA_SH=str(conda), INC2_SPLITS_DRY_RUN="1")
        e.update(env or {})
        return subprocess.run(["bash", str(SCRIPT)] + list(args), capture_output=True, text=True, env=e, timeout=300)

    import importlib.util
    missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    for verb in ("build", "lock", "verify", "summary"):
        # build imports torch and transformers for its scan; without them here, dry-run it with --skip-scan
        r = job(([verb, "--procs", "5"] + (["--skip-scan"] if missing else [])) if verb == "build" else [verb])
        check("dry run of %s: exit 0 and the command" % verb, r.returncode == 0 and
              ("DRY RUN: python -u -m weed_optimizer_framework.tools.inc2.splits %s" % verb) in r.stdout,
              r.stdout[-400:] + r.stderr[-400:])
    if missing:
        skip("scan dry run", "%s not installed here (the job imports them for scan)" % missing)
    else:
        r = job(["scan"])
        check("dry run of scan with a GPU: exit 0", r.returncode == 0 and "DRY RUN" in r.stdout, r.stderr[-300:])
    r = job(["scan"], gpu=False)
    check("scan without a GPU refuses (exit 2)", r.returncode == 2 and "no GPU" in r.stderr, r.stderr[-300:])
    r = job(["build"], gpu=False)
    check("build without a GPU refuses (exit 2): it runs the embedding scan",
          r.returncode == 2 and "no GPU" in r.stderr, r.stderr[-300:])
    r = job(["build", "--skip-scan"], gpu=False)
    check("... build --skip-scan and lock run without one",
          r.returncode == 0 and "DRY RUN" in r.stdout and job(["lock"], gpu=False).returncode == 0,
          r.stdout[-300:] + r.stderr[-300:])
    r = job(["lock", "--testing"])
    check("lock --testing refuses (exit 2)", r.returncode == 2 and "synthetic world" in r.stderr, r.stderr)
    r = job(["bogus"])
    check("an unknown verb refuses (exit 2)", r.returncode == 2 and "usage" in r.stderr)
    r = job([])
    check("no verb refuses (exit 2)", r.returncode == 2)
    r = job(["build", "--testing"])
    check("--testing refuses (exit 2)", r.returncode == 2 and "synthetic world" in r.stderr, r.stderr)
    target = repo / "weed_optimizer_framework" / "tools" / "inc2" / "guard.py"
    target.write_text(target.read_text() + "\n# drift\n")
    r = job(["build"])
    check("an outer module that differs from the nested copy refuses (exit 1)",
          r.returncode == 1 and "DIFFERS" in r.stdout and "differ from the nested copy" in r.stderr,
          r.stdout[-300:] + r.stderr[-300:])
    r = job(["build"], env={"INC2_SPLITS_ALLOW_DRIFT": "1"})
    check("... unless INC2_SPLITS_ALLOW_DRIFT=1 (logged)", r.returncode == 0 and "WARNING" in r.stdout)


def test_module_rules():
    print("module rules")
    src = (ROOT / "weed_optimizer_framework" / "tools" / "inc2" / "splits.py").read_text()
    writes = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("write"):
            if node.args and "C1." in ast.unparse(node.args[0]) and "SPLITS_DIR" in ast.unparse(node.args[0]):
                writes.append(node.lineno)
    check("splits.py writes nothing under a v1 path (the v1 tree is compared byte for byte above)", not writes,
          writes)


def main():
    test_real_pins()
    test_units()
    test_near_both_directions()
    test_writer_lock()
    make_v1_world()
    v1 = tree_shas(INC / "splits" / "v1")
    check("the synthetic v1 world verifies", S1.verify(scorer_path=SCORER) == [])
    make_step1()
    test_pins_refuse()
    test_refuse_included_copy()
    test_train_core_variant_fixture()
    test_refuse_train_core_variant()
    test_first_build(v1)
    test_scan_and_refusals()
    test_final_build_and_lock(v1)
    test_after_lock()
    test_job_script()
    test_module_rules()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), FAILURES))
        return 1
    print("all inc2.splits checks passed%s" % (" (%d skipped)" % len(SKIPS) if SKIPS else ""))
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
