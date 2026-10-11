#!/usr/bin/env python3
"""V3F D0-S: the residual-copy sweep (inc2/v3fam_sweep.py; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11):
V3F, the v3 experiment family D0-D9", V3F.5 D0-S).

The synthetic world of tests/test_inc2_train.py (LOCK v1 with dev and test) and synthetic textured "soil"
images on which OpenCV's SIFT finds structure. A stand-in embedder (quadrant colour means, no model) replaces
DINOv2.

What is pinned:
- the constants of the procedure (views, shortlist, SIFT, 50 inliers, the calibration's 200 images and five
  families, the trim's 8 grey levels and std 2, the reuse rule 100 rows at 0.999, Q 2,594 and R 37,740, base
  B's sha256, LOCK v2's embedder);
- the trim on synthetic letterboxed images (trimmed to the content within 1 px: top and bottom, left and right,
  four sides of different greys, a slightly noisy border), an image without a border kept whole, a border with
  a gradient (std >= 2) kept, a uniform image kept whole;
- each query view's geometry, and the five calibration copy families;
- the shortlist rule: top 5 per view at cosine >= 0.70 (0.69 left out, a sixth row left out), and dHash pairs
  within 10 bits in either direction (a reverse-only pair kept with its bits, 11 bits left out);
- SIFT verification: a planted crop and a letterbox copy verified (>= 50 inliers), a horizontal flip found by
  the flipped pass only, two different soil images not verified;
- the reuse rule: stored descriptors matched by the 8 stored dHashes are reused when every checked row gives
  cosine >= 0.999; one checked row below refuses the reuse (every row recomputed); a row with a border is never
  reused;
- the trigger's branches: H in test only, dev only, both, empty;
- end to end in testing: run refuses without the time stamp; a letterboxed copy of a dev image and a crop of a
  test image in R are verified with montages; calibration recall per family and false verifications recorded,
  no threshold changed; run.json written once; a verdict needs owner|delegate and yes|no, is written once,
  and the last one writes record.json (H, the trigger, the removal list and its sha256); no verdict after it;
  with nothing verified, run writes record.json itself (H empty);
- no label file is opened by the sweep on LOCK v1's dev and test (query_rows), only pixels;
- the CLI: status prints JSON, a verdict without its arguments exits 2.

Run:  python3 tests/test_inc2_v3fam_sweep.py
"""
import builtins
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO first)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_sweep as SW  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = W.patched
TMP = W.TMP
IMG = TMP / "sweep_imgs"


class StubEmbedder(object):
    """Deterministic 16-dim features: per-channel means of the four quadrants and the whole's std, and a constant
    (no model, no download)."""
    name = "stub-quadrant16"
    dim = 16

    def __call__(self, pils):
        out = []
        for p in pils:
            a = np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0
            h, w = a.shape[:2]
            q = [a[:h // 2, :w // 2], a[:h // 2, w // 2:], a[h // 2:, :w // 2], a[h // 2:, w // 2:]]
            out.append(np.concatenate([x.mean(axis=(0, 1)) for x in q] + [a.std(axis=(0, 1)), np.ones(1)]))
        return np.stack(out).astype(np.float32)


EMB = StubEmbedder()


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except F.V3famRefused as e:
        return e
    return None


def soil(seed, W_=480, H=360):
    """A textured 'soil' image with blobs and strokes (SIFT finds structure in it)."""
    rng = np.random.RandomState(seed)
    low = rng.randint(40, 200, (H // 12, W_ // 12, 3)).astype(np.uint8)
    im = Image.fromarray(low).resize((W_, H), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    for _ in range(60):
        x, y, r = rng.randint(0, W_), rng.randint(0, H), rng.randint(3, 18)
        c = tuple(int(v) for v in rng.randint(0, 255, 3))
        if rng.rand() < 0.5:
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        else:
            d.line([x, y, x + rng.randint(-40, 40), y + rng.randint(-40, 40)], fill=c, width=int(rng.randint(1, 4)))
    return im


def save(im, name, fmt="PNG"):
    p = IMG / name
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, format=fmt, quality=95) if fmt == "JPEG" else im.save(p, format=fmt)
    return str(p)


def pad(im, top=0, bottom=0, left=0, right=0, grey=(124, 124, 124), noise=0.0, seed=0):
    W_, H = im.size
    out = Image.new("RGB", (W_ + left + right, H + top + bottom), grey)
    if noise:
        a = np.asarray(out, dtype=np.float32)
        a += np.random.RandomState(seed).uniform(-noise, noise, a.shape)
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out.paste(im, (left, top))
    return out


def near(a, b, tol=1):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


# ================================================================== UNITS
def u_constants():
    print("the pre-registered constants of D0-S")
    check("views: whole, hflip, centre crops 0.9 and 0.8, the four 0.8 corners (8)",
          SW.VIEWS == ("whole", "hflip", "centre90", "centre80", "corner80_tl", "corner80_tr", "corner80_bl",
                       "corner80_br"))
    check("shortlist top 5 at cosine >= 0.70, plus dHash within 10 bits both directions",
          (SW.SHORTLIST_TOP, SW.SHORTLIST_COS, SW.SHORTLIST_BITS) == (5, 0.70, 10))
    check("SIFT 4,000 features, long side 1,024, Lowe 0.8, RANSAC 5 px / 2,000 iterations / 0.995, verified at 50",
          (SW.SIFT_FEATURES, SW.SIFT_LONG, SW.LOWE_RATIO, SW.RANSAC_PX, SW.RANSAC_ITERS, SW.RANSAC_CONF,
           SW.INLIERS_MIN) == (4000, 1024, 0.8, 5.0, 2000, 0.995, 50))
    check("calibration: 200 images, five families, its seed text; 200 negatives",
          SW.CALIB_N == 200 and SW.CALIB_NEG == 200 and SW.CALIB_SEED_TEXT == "inc2/v3fam/sweep/calib/%s"
          and SW.CALIB_FAMILIES == ("centre90", "centre80", "letterbox640", "jpeg75", "hflip")
          and SW.GREY == 124 and SW.JPEG_QUALITY == 75 and SW.LETTERBOX == 640)
    check("trim 8 grey levels, std under 2; reuse 100 rows at cosine 0.999",
          (SW.TRIM_TOL, SW.TRIM_STD, SW.REUSE_CHECK, SW.REUSE_COS) == (8, 2.0, 100, 0.999))
    check("Q = dev + test (2,594), R = 37,740 rows of base B 18d24ead...899a; LOCK v2's embedder",
          SW.QUERY_EXAMS == ("dev", "test") and SW.QUERY_IMAGES == 2594 and SW.REFERENCE_ROWS == 37740
          and SW.BASE_B_SHA256 == "18d24eade70ba2a5c3c39061eaacd30d148fa00d5ac28fc5685479a7d104899a"
          and SW.EMBEDDER == "facebook/dinov2-base:cls"
          and SW.BASE_V2_SOURCES == ("cottonweeddet12/train", "3seasonweeddet10/data2022",
                                     "3seasonweeddet10/data2023"))
    check("the six E3-frozen files hash as V3F.1 records (the family's precondition)",
          F.FROZEN_SHA256 == {
              "twostage": "8fe2edc0e0d2a00d07182c3746407281a112c9e50c76662bc9f48a558b328ddc",
              "scorer": "18c008371cd2c021104d034d06b837564016f4509e8a969c2a3f850b09dd2de4",
              "scorer_sidecar": "de7aacda903780b32944866be0b4daa5c6b1f76e2c86965f7a99d85f443d5e52",
              "scorer_native": "ba9eeb928a2b5a0756ae8ac0144b54df2fdfb3570fd011f66b4a8022039d6b7d",
              "verify": "602eba3c935f375179d6f922dce28bbf89006054e5bcec5894f0c9cc7b87aa10",
              "semisup_labeler": "03b28c50699b30a32720e383ff2b713a2638926650007acb92a367d1f450458c"}
          and F.check_frozen() == F.FROZEN_SHA256)


def u_trim():
    print("the letterbox trim")
    c = soil(1, 200, 150)
    cases = {"top and bottom": (pad(c, top=40, bottom=40), (0, 40, 200, 190)),
             "left and right, black": (pad(c, left=30, right=17, grey=(0, 0, 0)), (30, 0, 230, 150)),
             "four sides": (pad(pad(c, top=11, bottom=5, grey=(255, 255, 255)), left=9, right=23, grey=(20, 20, 20)),
                            (9, 11, 209, 161)),
             "a noisy border (within 8, std < 2)": (pad(c, top=25, bottom=25, noise=1.5, seed=3), (0, 25, 200, 175))}
    for name, (im, want) in cases.items():
        got = SW.trim_box(im)
        check("%s: trimmed to the content within 1 px (%s, want %s)" % (name, got, want), near(got, want), got)
    check("an image without a border is kept whole", SW.trim_box(c) == (0, 0, 200, 150), SW.trim_box(c))
    g = np.tile(np.linspace(60, 200, 200, dtype=np.float32), (30, 1))
    grad = Image.fromarray(np.stack([g] * 3, -1).astype(np.uint8))
    im = Image.new("RGB", (200, 180))
    im.paste(grad, (0, 0))
    im.paste(c, (0, 30))
    check("a border with a gradient (std >= 2) is not trimmed", SW.trim_box(im)[1] == 0, SW.trim_box(im))
    flat = Image.new("RGB", (50, 40), (90, 90, 90))
    check("a uniform image keeps its full frame", SW.trim_box(flat) == (0, 0, 50, 40), SW.trim_box(flat))


def u_views_and_copies():
    print("the query views and the calibration's copy families")
    im = soil(2, 100, 80)
    sizes = {v: SW.view_crop(im, v).size for v in SW.VIEWS}
    check("view sizes: whole 100x80, flip 100x80, centre90 90x72, centre80 and the corners 80x64",
          sizes == {"whole": (100, 80), "hflip": (100, 80), "centre90": (90, 72), "centre80": (80, 64),
                    "corner80_tl": (80, 64), "corner80_tr": (80, 64), "corner80_bl": (80, 64), "corner80_br": (80, 64)},
          sizes)
    a = np.asarray(im)
    check("centre90 is the centre (5, 4); corner80_br is the bottom-right block; hflip mirrors",
          (np.asarray(SW.view_crop(im, "centre90")) == a[4:76, 5:95]).all()
          and (np.asarray(SW.view_crop(im, "corner80_br")) == a[16:80, 20:100]).all()
          and (np.asarray(SW.view_crop(im, "hflip")) == a[:, ::-1]).all())
    big = soil(3, 480, 360)
    lb = SW.copy_image(big, "letterbox640")
    j = SW.copy_image(big, "jpeg75")
    check("letterbox640 is 640 x 640 with grey 124 padding around a 640 x 480 content (trim recovers it)",
          lb.size == (640, 640) and lb.getpixel((5, 5)) == (124, 124, 124) and near(SW.trim_box(lb), (0, 80, 640, 560)),
          (lb.size, SW.trim_box(lb)))
    check("jpeg75 re-encodes (same size, other pixels); centre80 and hflip are the views of that name",
          j.size == big.size and not (np.asarray(j) == np.asarray(big)).all()
          and SW.copy_image(big, "centre80").size == (384, 288)
          and (np.asarray(SW.copy_image(big, "hflip")) == np.asarray(big)[:, ::-1]).all())
    check("an unknown view or family refuses", refused(SW.view_crop, im, "zoom") is not None
          and refused(SW.copy_image, im, "blur") is not None)


def _set(X, H, views=("whole",)):
    rows = [{"key": "r%d" % i} for i in range(len(H))]
    s = {"rows": rows, "views": tuple(views), "X": np.asarray(X, np.float32), "H": np.asarray(H, np.uint64),
         "info": {}, "counts": {}}
    return SW._index(s)


def u_shortlist():
    print("the shortlist rule")
    d = 8
    q = np.zeros((1, d), np.float32)
    q[0, 0] = 1.0

    def vec(c):
        v = np.zeros(d, np.float32)
        v[0], v[1] = c, np.sqrt(max(0.0, 1 - c * c))
        return v
    cos = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.69]
    R = np.stack([vec(c) for c in cos])
    rng = np.random.RandomState(0)
    Hr = rng.randint(0, 2 ** 63, (len(cos), 8), dtype=np.int64).astype(np.uint64)
    Hq = rng.randint(0, 2 ** 63, (1, 8), dtype=np.int64).astype(np.uint64)
    qs = _set(q, Hq, views=("whole",))
    rs = _set(R, Hr)
    got = SW.shortlist(qs, rs)
    check("top 5 by cosine at >= 0.70: the five nearest kept, the sixth (0.75) and 0.69 left out",
          sorted(got) == [(0, j) for j in range(5)] and all(got[(0, j)]["cos"] == round(cos[j], 6) for j in range(5)),
          got)
    # a reverse-only dHash pair: the reference's rot90 variant equals the query's dHash with 9 bits flipped
    R2 = np.stack([vec(0.1)] * 3)
    Hr2 = rng.randint(0, 2 ** 63, (3, 8), dtype=np.int64).astype(np.uint64)
    Hr2[0, 3] = Hq[0, 0] ^ np.uint64(0x1FF)            # 9 bits, the reverse direction
    Hr2[1, 3] = Hq[0, 0] ^ np.uint64(0x7FF)            # 11 bits
    got2 = SW.shortlist(_set(q, Hq), _set(R2, Hr2))
    check("a pair within 10 bits in the reverse direction only is shortlisted with its bits; 11 bits is not",
          (0, 0) in got2 and got2[(0, 0)]["bits"] == 9 and got2[(0, 0)]["cos"] is None and (0, 1) not in got2, got2)


def u_sift():
    print("SIFT verification")
    a, b = soil(11), soil(12)
    crop = a.crop((48, 36, 432, 324))
    lb = SW.trimmed(SW.copy_image(a, "letterbox640"))
    i_crop, i_lb, i_other = SW.sift_inliers(a, crop), SW.sift_inliers(a, lb), SW.sift_inliers(a, b)
    i_flip = SW.sift_inliers(a, a.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    check("a planted 0.8 crop is verified (%s inliers)" % (i_crop,), max(i_crop) >= SW.INLIERS_MIN, i_crop)
    check("a letterboxed copy, trimmed, is verified (%s)" % (i_lb,), max(i_lb) >= SW.INLIERS_MIN, i_lb)
    check("two different soil images are not verified (%s)" % (i_other,), max(i_other) < SW.INLIERS_MIN, i_other)
    check("a horizontal flip is found by the flipped pass only (%s)" % (i_flip,),
          i_flip[1] >= SW.INLIERS_MIN and i_flip[0] < SW.INLIERS_MIN, i_flip)


def u_reuse():
    print("the reuse rule for stored descriptors")
    rows, imgs = [], []
    for i in range(4):
        im = soil(20 + i, 120, 90)
        rows.append({"key": "ext_%d" % i, "image": save(im, "reuse/ext_%d.png" % i), "sha256": None})
        imgs.append(im)
    bordered = pad(soil(30, 120, 90), top=20, bottom=20)
    rows.append({"key": "ext_border", "image": save(bordered, "reuse/ext_border.png"), "sha256": None})
    info, bad = SW.trim_set(rows)
    X = SW.describe(rows, info, ("whole",), EMB)
    H = np.asarray([info[r["key"]]["stored"] for r in rows], np.uint64)
    path = TMP / "scan_desc.npz"
    np.savez(path, keys=np.asarray(["u%d" % i for i in range(len(rows))]), X=X.astype(np.float16), H=H,
             hash_ok=np.ones(len(rows), bool), meta=np.asarray("{}"), extra=np.asarray(["null"] * len(rows)))
    got, rec = SW.reuse_descriptors(rows, info, EMB, path)
    check("rows without a border whose 8 stored dHashes name a stored descriptor are reused (%d), every checked row "
          "at cosine >= 0.999; the bordered row is not a candidate" % rec["reused"],
          not bad and rec["reused"] == 4 and "ext_border" not in got and rec["check"]["below"] == 0, rec)
    X2 = X.copy()
    X2[2] = X2[2][::-1]
    np.savez(path, keys=np.asarray(["u%d" % i for i in range(len(rows))]), X=X2.astype(np.float16), H=H,
             hash_ok=np.ones(len(rows), bool), meta=np.asarray("{}"), extra=np.asarray(["null"] * len(rows)))
    got2, rec2 = SW.reuse_descriptors(rows, info, EMB, path)
    check("one checked row below 0.999 refuses the reuse: every row is recomputed",
          got2 == {} and rec2["check"]["below"] >= 1 and "recomputed" in rec2["why"], rec2)
    with patched(SW, "REUSE_CHECK", 1):
        order = sorted([r["key"] for r in rows[:4]], key=lambda k: (C.stable_int(SW.REUSE_SEED_TEXT % k), k))
        checked = rec["check"]["rows"]
        _g3, rec3 = SW.reuse_descriptors(rows, info, EMB, path)
    check("the checked rows are the first by stable_int(inc2/v3fam/sweep/reuse/<key>) (all 4 here; 1 when the cap "
          "is 1, %s first)" % order[0], checked == 4 and rec3["check"]["rows"] == 1, (checked, rec3))
    got4, rec4 = SW.reuse_descriptors(rows, info, EMB, TMP / "missing.npz")
    check("no stored file: nothing reused", got4 == {} and rec4["why"] == "no stored descriptors", rec4)


def u_trigger():
    print("the trigger's branches")
    dev = ["d1", "d2", "d3"]
    t = SW.trigger(["test/t9"], dev)
    check("H in test only: the re-analysis and BCLEAN, copies leave training, dev decides on dev",
          t["test_branch"] and t["bclean"] and t["reanalyse_test_minus_h"] and not t["dev_branch"]
          and t["dev_decides_on"] == "dev" and t["copies_leave_training_sets"] and t["dev_minus_h"] is None, t)
    t = SW.trigger(["dev/d2"], dev)
    check("H in dev only: decided on dev minus H (2 images), E3 re-read there, full dev beside; no BCLEAN",
          t["dev_branch"] and not t["bclean"] and t["dev_decides_on"] == "dev minus H"
          and t["dev_minus_h"]["images"] == 2 and t["e3_reread_on_dev_minus_h"] and t["full_dev_reported_beside"]
          and t["dev_minus_h"]["keys_sha256"] == SW.list_sha256(["d1", "d3"]), t)
    t = SW.trigger(["dev/d1", "test/t1"], dev)
    check("H in both: both branches apply together", t["dev_branch"] and t["test_branch"] and t["bclean"]
          and t["dev_decides_on"] == "dev minus H", t)
    t = SW.trigger([], dev)
    check("H empty: no branch, the record says so", t["h_empty"] and not any(
        t[k] for k in ("test_branch", "dev_branch", "bclean", "copies_leave_training_sets")), t)


UNITS = ("u_constants", "u_trim", "u_views_and_copies", "u_shortlist", "u_sift", "u_reuse", "u_trigger")


# ================================================================== WORLD
def q_row(exam, i, im):
    key = "%s__20210812_iPhoneSE_YL_%d" % (exam, i)
    p = save(im, "q/%s.jpg" % key, fmt="JPEG")
    return {"key": "%s/%s" % (exam, key), "exam": exam, "exam_key": key, "image": p, "sha256": C.sha256_file(p),
            "session": "s%d" % (i % 2)}


def r_row(key, im):
    p = save(im, "r/%s.png" % key)
    return {"key": key, "image": p, "sha256": C.sha256_file(p), "source": "rf_x__y"}


@contextlib.contextmanager
def aside(path):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".aside")
    if path.exists():
        os.replace(str(path), str(tmp))
    try:
        yield
    finally:
        if path.exists():
            shutil.rmtree(str(path))
        if tmp.exists():
            os.replace(str(tmp), str(path))


def w_run():
    print("the sweep end to end (testing; stand-in embedder)")
    d0, d1, d2, t0, t1 = soil(100), soil(101), soil(102), soil(103), soil(104)
    q = [q_row("dev", 0, d0), q_row("dev", 1, d1), q_row("dev", 2, d2), q_row("test", 0, t0), q_row("test", 1, t1)]
    r = [r_row("rf_x__y__lb_dev0", SW.copy_image(d0.convert("RGB"), "letterbox640")),
         r_row("rf_x__y__crop_test1", t1.crop((48, 36, 432, 324))),
         r_row("rf_x__y__other_a", soil(200)), r_row("rf_x__y__other_b", soil(201))]
    sets = {"q": q, "r": r}
    ts = F.root() / "timestamp.json"
    if ts.exists():
        ts.unlink()
    e = refused(SW.run, procs=1, embedder=EMB, testing=True, sets=sets)
    check("run refuses before the amendment's time stamp exists", e is not None and "time-stamped" in str(e), e)
    ts.parent.mkdir(parents=True, exist_ok=True)
    ts.write_text(json.dumps({"ots": "V3F_2026-10-11.md.ots", "sha256": "0" * 64}))
    check("injected sets only in testing", refused(SW.run, procs=1, embedder=EMB, testing=False, sets=sets) is not None)
    consts = {k: getattr(SW, k) for k in ("SHORTLIST_TOP", "SHORTLIST_COS", "SHORTLIST_BITS", "INLIERS_MIN",
                                          "TRIM_TOL", "TRIM_STD")}
    with patched(SW, "CALIB_N", 3), patched(SW, "CALIB_NEG", 3):
        doc = SW.run(procs=1, embedder=EMB, testing=True, sets=sets)
    pairs = {(p["query"], p["ref"]): p for p in doc["pairs"]}
    ver = sorted(k for k, p in pairs.items() if p["verified"])
    check("the letterboxed copy of dev 0 and the crop of test 1 are verified, nothing else (%s)" % ver,
          ver == [("dev/dev__20210812_iPhoneSE_YL_0", "rf_x__y__lb_dev0"),
                  ("test/test__20210812_iPhoneSE_YL_1", "rf_x__y__crop_test1")], [(k, p["inliers"]) for k, p in pairs.items()])
    md = SW.sweep_dir() / "montages"
    check("each verified pair has its montage", sorted(doc["montages"]) == sorted(pairs[k]["pair"] for k in ver)
          and all((md / ("%s.jpg" % p)).is_file() for p in doc["montages"]), doc["montages"])
    cal = doc["calibration"]
    check("calibration: recall per family over 3 images, false verifications on non-copy pairs, no threshold moved",
          set(cal["families"]) == set(SW.CALIB_FAMILIES) and all(v["n"] == 3 for v in cal["families"].values())
          and cal["negatives"]["pairs"] == 3 and cal["thresholds_moved"] is False
          and {k: getattr(SW, k) for k in consts} == consts, cal)
    check("calibration found the crop and letterbox families it planted (%s)"
          % {f: v["recall"] for f, v in cal["families"].items()},
          cal["families"]["centre90"]["recall"] == 1.0 and cal["families"]["letterbox640"]["recall"] == 1.0
          and cal["negatives"]["false_verified"] == 0, cal)
    check("record.json waits for the verdicts", SW.read_record() is None and SW.status()["status"] ==
          "verdicts pending" and len(SW.status()["left"]) == 2, SW.status())
    check("run.json is written once", refused(SW.run, procs=1, embedder=EMB, testing=True, sets=sets) is not None)
    pd = pairs[ver[0]]["pair"]
    pt = pairs[ver[1]]["pair"]
    unver = [p["pair"] for p in doc["pairs"] if not p["verified"]] or ["p000000000000"]
    check("a verdict needs owner|delegate and yes|no; an unverified pair refuses",
          refused(SW.verdict, pd, "yes", "someone") is not None and refused(SW.verdict, pd, "maybe", "owner")
          is not None and refused(SW.verdict, unver[0], "yes", "delegate") is not None)
    SW.verdict(pd, "yes", "delegate")
    check("a verdict is written once", refused(SW.verdict, pd, "no", "owner") is not None)
    rec = SW.verdict(pt, "no", "owner")
    check("the last verdict writes record.json: H = the dev image, the dev branch only, the copy's reference row "
          "on the removal list with its sha256",
          rec is not None and rec["H"] == ["dev/dev__20210812_iPhoneSE_YL_0"] and rec["trigger"]["dev_branch"]
          and not rec["trigger"]["test_branch"] and rec["removal"]["keys"] == ["rf_x__y__lb_dev0"]
          and rec["removal"]["sha256"] == SW.list_sha256(["rf_x__y__lb_dev0"])
          and rec["verdicts"][pd]["read_by"] == "Claude (delegate under the 2026-09-30 owner grant)"
          and rec["verdicts"][pt]["read_by"] == "Harry He (owner)" and SW.complete(), rec)
    check("removal_keys reads the complete record", SW.removal_keys() == (["rf_x__y__lb_dev0"],
                                                                         SW.list_sha256(["rf_x__y__lb_dev0"])))
    check("no verdict after the record (a change is a deviation)", refused(SW.verdict, pt, "yes", "owner")
          is not None)
    with aside(SW.sweep_dir()):
        with patched(SW, "CALIB_N", 2), patched(SW, "CALIB_NEG", 2):
            SW.run(procs=1, embedder=EMB, testing=True, sets={"q": q[1:3], "r": r[2:]})
        rec0 = SW.read_record()
        check("with nothing verified run writes record.json itself: H empty, no branch",
              rec0 is not None and rec0["H"] == [] and rec0["trigger"]["h_empty"] and rec0["removal"]["keys"] == []
              and "note" in rec0, rec0)
    with aside(SW.sweep_dir()):
        check("removal_keys refuses without a complete record (L23H writes nothing before it)",
              refused(SW.removal_keys) is not None)


def w_no_label():
    print("no label file is opened: LOCK v1's dev and test through query_rows, pixels only")
    world = W.WORLD_ROWS
    labels = {str(pathlib.Path(r["label"]).resolve()) for s in ("dev", "test") for r in world[s]}
    images = {str(pathlib.Path(r["image"]).resolve()) for s in ("dev", "test") for r in world[s]}
    opened = []
    real_open, real_pil = builtins.open, Image.open

    def spy_open(f, *a, **k):
        try:
            opened.append(str(pathlib.Path(f).resolve()))
        except TypeError:
            pass
        return real_open(f, *a, **k)

    def spy_pil(f, *a, **k):
        if isinstance(f, (str, pathlib.Path)):
            opened.append(str(pathlib.Path(f).resolve()))
        return real_pil(f, *a, **k)
    r = [r_row("rf_x__y__nl_a", soil(300, 96, 96)), r_row("rf_x__y__nl_b", soil(301, 96, 96))]
    with aside(SW.sweep_dir()):
        builtins.open, Image.open = spy_open, spy_pil
        try:
            with patched(SW, "CALIB_N", 2), patched(SW, "CALIB_NEG", 2):
                SW.run(procs=1, embedder=EMB, testing=True, sets={"r": r})
        finally:
            builtins.open, Image.open = real_open, real_pil
        doc = SW.read_run()
    hit = sorted(set(opened) & labels)
    check("the sweep read dev's and test's pixels (%d of %d images) and no label file (%d opened)"
          % (len(set(opened) & images), len(images), len(hit)), not hit and (set(opened) & images) == images, hit[:3])
    check("Q is LOCK v1's dev and test (keys 'dev/...', 'test/...')", doc["sets"]["query"]["images"] ==
          len(world["dev"]) + len(world["test"]) and doc["sets"]["query"]["exams"] == ["dev", "test"], doc["sets"])


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(SW.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def cli(*a):
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_sweep"] + list(a),
                           env=env, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    rc, out, _e = cli("status")
    rc2, _o2, err2 = cli("verdict", "--pair", "p1")
    check("status prints the record's state (exit 0); a verdict without --copy and --read-by exits 2",
          rc == 0 and json.loads(out)["status"] == "complete" and rc2 == 2 and "ERROR" in err2, (rc, out[:200],
                                                                                             rc2, err2[-200:]))


WORLD_TESTS = ("w_run", "w_no_label", "w_cli")


def main():
    t0 = time.time()
    try:
        W.WORLD_ROWS = W.build_world()
        for name in UNITS + WORLD_TESTS:
            globals()[name]()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
