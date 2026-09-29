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
  * build: the four byte copies are v1's bytes (the v1 LOCK shas); the
    never-train index holds dev + test + imageweeds, incomplete (refused)
    until lock; tsw keys tsw2x__<stem>, source 3seasonweeddet10/data202x,
    session = stem minus the frame number, labels byte copies (same sha256);
    no training row carries an exam key; drops, each recorded: a flipped test
    image (near_eval_variant), a rotated imageweeds image, the ood23 near
    duplicate of an ood22 row (the tsw22 row is kept), every row of a dev
    session; the row sharing a session with test is kept and counted; L-5
    drops every cwp10 and vanpe image (listed in l5_excluded.jsonl), and a
    flipped test image among them is recorded as an H6(b) incident without
    refusing; a harvested near copy of a train_core image is dropped; base_v2
    is the union of the parts, pairwise disjoint; licences resolve from the
    card index, the funnel config and the Zenodo record, an unresolved one is
    research_only, and an owner table resolves train_core;
  * lock refuses before the embedding scan; the scan (the stream's own
    calibration, since the funnel's leak_v1.json there failed) finds the 15 %
    crop of a test image in tsw22 and the sheared test image in base B's kept
    part, both > 6 bits under every variant, and the flipped / rotated copies;
    it covers every candidate (all v1 ood rows, all of base B), not the
    build's manifests, and carries evaluation keys only;
    lock then refuses (flagged rows), and a second build refuses (R4, the
    harvested copy); without that image the second build drops the tsw crop
    (near_eval_embed); a scan reusing a passed funnel leak_v1.json records it;
  * lock: LOCK v2 records the manifests, the index shas and counts, the
    scorer, derived_from (the v1 LOCK, identical byte copies) and the H6
    status; the indexes are complete; every file under splits/v2 is 0444;
    verify is [] for v2 and v1, and no v1 file changed; GuardV2.load works and
    refuses a flipped test image, a base copy and the dropped ood23 duplicate,
    and passes a fresh image; build, scan and lock refuse once locked; verify
    catches a changed label and a writable file;
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
    ts = REPO / "downloads" / "3seasonweeddet10"
    d22, d23 = ts / "data2022" / "fieldA", ts / "data2023"
    for s in range(len(SESSIONS)):                 # one frame of every train session (dev ones included)
        p = photo(d22 / ("20210701_FakeCam_S%d_901.png" % s), 500 + s)
        voc(p, [("Waterhemp", (10, 10, 50, 50))])
    p = photo(d22 / "20210801_FakeCam_TEST_901.png", 520)          # shares a session with test
    voc(p, [("Palmer Amaranth", (10, 10, 50, 50)), ("Lambsquarters", (60, 20, 100, 60))])
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


def make_step1():
    core = sorted(C1.read_manifest(C1.manifest_path("train_core")), key=lambda r: r["key"])
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


def test_refuse_train_core_variant():
    print("a flipped / rotated evaluation copy inside train_core refuses the build")
    write_base_selected([])
    core = sorted(C1.read_manifest(C1.manifest_path("train_core")), key=lambda r: r["key"])
    victim, t = core[2], C1.dhash(test_image(1))
    orig = G.image_hashes

    def planted(path):
        h, v = orig(path)
        if str(path) == victim["image"] and v is not None:
            v = dict(v, rot90=t)                 # its 90-degree rotation is a test image
        return h, v
    G.image_hashes = planted
    try:
        ok, msg = raises(lambda: S2.build(testing=True, scorer_path=SCORER, procs=1, scan_mode="skip"), "R4 INCIDENT")
    finally:
        G.image_hashes = orig
    check("refused: R4 incident naming the train_core image and near_eval_variant (base v2 holds all of "
          "train_core; v1 checked the stored dHash only)",
          ok and victim["key"] in msg and "near_eval_variant" in msg and "train_core" in msg, msg[:400])
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
    check("tsw22 drops: the flipped test image (near_eval_variant) and the %d dev-session row(s)" % n_dev,
          t22["dropped"] == {"near_eval_variant": 1, "dev_session": n_dev} and n_dev >= 1, t22["dropped"])
    check("tsw22 keeps the rest, the embedding copy included (no scan yet): %d of %d"
          % (t22["kept"], t22["v1_rows"]), t22["kept"] == t22["v1_rows"] - 1 - n_dev)
    check("tsw23 drops: the ood22 near duplicate (the tsw22 row kept) and the rotated imageweeds image",
          t23["dropped"] == {"near_tsw22": 1, "near_eval_variant": 1}, t23["dropped"])
    ex = {e["key"]: e for e in t23["examples"]}
    check("... each recorded with what it matched", ex["tsw23__field23_dup22"]["match"]["with"] == "tsw22__field22_a"
          and ex["tsw23__field23_rot_iw"]["match"]["split"] == "imageweeds"
          and ex["tsw23__field23_rot_iw"]["match"]["variant"] in ("rot90", "rot270"), t23["examples"])
    ov = t22["session_overlap"]
    check("the row sharing a session with test is kept and counted (test 1); train_core sessions %d"
          % ov["train_core"], ov["test"] == 1 and ov["dev"] == n_dev and ov["train_core"] == len(SESSIONS) - n_dev
          and "tsw22__20210801_FakeCam_TEST_901" in {r["key"] for r in C1.read_manifest(C2.v2_manifest_path("tsw22"))},
          ov)
    rows22 = C1.read_manifest(C2.v2_manifest_path("tsw22"))
    v1_22 = {S2.tsw_key("tsw22", r["key"]): r for r in C1.read_manifest(C1.manifest_path("ood22"))}
    check("tsw rows: key tsw22__<stem>, source 3seasonweeddet10/data2022, session = stem minus frame number",
          all(r["key"].startswith("tsw22__") and r["source"] == "3seasonweeddet10/data2022" for r in rows22)
          and next(r for r in rows22 if r["key"].endswith("TEST_901"))["session"] == "20210801_FakeCam_TEST"
          and next(r for r in rows22 if r["key"].endswith("field22_a"))["session"] == "field22_a")
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
          b["dropped"] == {"near_train_core": 1} and b["kept"] == 6, b)
    parts = s["base_v2"]["parts"]
    check("base_v2 = train_core + tsw22 + tsw23 + base B's kept part (%s)" % parts,
          len(base) == sum(parts.values()) and parts["train_core"] == len(C1.read_manifest(C1.manifest_path(
              "train_core"))) and parts["tsw22"] == t22["kept"] and parts["tsw23"] == t23["kept"]
          and parts["base_b"] == 6)
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
            ("base_b", BASE["hflip_test3"]["key"])}
    check("found: the 15 % crop of a test image in tsw22 and the sheared test image in base B's kept part "
          "(embedding), the flipped / rotated copies (the detector's dHash half), nothing else",
          hit_keys == want, sorted(hit_keys ^ want))
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


def listed_entry(row, split="test", key="test__x"):
    return {"key": row["key"], "image": row["image"], "eval_split": split, "eval_key": key, "cos": 0.99,
            "bits": 30, "variant": "id"}


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
    C2.write_json_atomic(S2.funnel_leak_path(), funnel_doc(own, []))

    print("without the harvested copy: scan (funnel threshold), build, lock")
    doc = S2.scan(embedder=HueEmbedder(), procs=1, testing=True)
    check("a passed funnel leak_v1.json is reused and recorded",
          doc["detector"]["calibration_source"] == "funnel_leak_v1"
          and doc["detector"]["calibration"]["path"] == str(S2.funnel_leak_path()), doc["detector"])
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
    scan_bytes = S2.embed_scan_path().read_bytes()
    doc = json.loads(scan_bytes)
    in_base = {r["key"] for r in C1.read_manifest(C2.v2_manifest_path("base_v2"))}
    i = next(j for j, (k, _sha) in enumerate(doc["scanned"]["base_b"]) if k in in_base)
    dropped_key = doc["scanned"]["base_b"].pop(i)[0]
    C2.write_json_atomic(S2.embed_scan_path(), doc)
    ok, msg = raises(lambda: S2.lock(scorer_path=SCORER, procs=1, testing=True), "were not scanned")
    check("lock refuses when embed_scan.json leaves a base v2 row unscanned", ok and dropped_key in msg, msg[:300])
    S2.embed_scan_path().write_bytes(scan_bytes)
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
          and set(lk["derived_from"]["identical"]) == set(C2.BYTE_COPIES)
          and all(lk["manifests"][x] == C1.read_lock()["manifests"][x] for x in C2.BYTE_COPIES))
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
    check("CLI verify and summary exit 0",
          S2.main(["verify", "--scorer", str(SCORER), "--procs", "1"]) == 0 and S2.main(["summary"]) == 0)
    check("the writer lock is released", not S2.writer_lock_path().exists())


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
            "tools/funnel/domains/weed.json"}
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
    test_writer_lock()
    make_v1_world()
    v1 = tree_shas(INC / "splits" / "v1")
    check("the synthetic v1 world verifies", S1.verify(scorer_path=SCORER) == [])
    make_step1()
    test_pins_refuse()
    test_refuse_included_copy()
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
