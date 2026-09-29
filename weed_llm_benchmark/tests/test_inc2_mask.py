#!/usr/bin/env python3
"""inc2/mask.py: per-box admission with masking (D-B) and mask_except
(docs/CONTINUOUS_LOOP.md §3.3 stage 6).

Pinned:
  * decide(): the v1 image rule gives "whole"; otherwise at least one VERIFIED
    target box gives "masked", with kept = VERIFIED targets + OtherPlant
    OTHER_OK/SMALL and masked = every other box (target CONFLICT, UNKNOWN,
    FAILED, SMALL; OtherPlant CONFLICT, FAILED and the out-of-fold UNKNOWN;
    every unmapped id 13); a masked box at IoU >= 0.5 with a kept box gives
    "refused_overlap"; no verified target box gives "not_admitted"; bad input
    raises.
  * mask_except(): kept pixels are unchanged, filled pixels equal the image's
    mean RGB (rounded, taken before the fill), a partly overlapping kept box
    keeps its pixels, masked_area_frac and the visible share of each masked
    box are exact; an overlap at IoU >= 0.5 is refused; without overlap the
    PNG is byte-identical to funnel.recover.mask (name, sha256, bytes), also
    for an EXIF-rotated original (both write the EXIF-transposed image); the
    file is content-addressed and written once; an unreadable image raises.

No network, no GPU. Run:  python3 tests/test_inc2_mask.py
"""
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_mask_test_"))
os.environ.setdefault("REPO", str(TMP / "repo"))
os.environ.setdefault("INC_DIR", str(TMP / "inc"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import mask as MK  # noqa: E402
from weed_optimizer_framework.tools.funnel import recover as R  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


def picture(path, W=97, H=61, seed=3, exif_orientation=None):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    im = Image.fromarray(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif_orientation is None:
        im.save(path)
    else:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        im.save(path, format="JPEG", quality=95, exif=ex.tobytes())
    return path


def test_decide():
    print("decide")
    T, O, U = 3, V.OTHER, MK.UNMAPPED_ID
    far = [(0.1 + 0.2 * i, 0.5, 0.08, 0.1) for i in range(5)]
    d = MK.decide([T, O], [V.VERIFIED, V.OTHER_OK], far[:2])
    check("the v1 image rule admits -> whole, every box kept", d["admission"] == MK.WHOLE and d["keep"] == [0, 1]
          and d["mask"] == [] and d["image_verdict"] == V.ADMITTED, d)
    d = MK.decide([T, O], [V.VERIFIED, V.SMALL], far[:2])
    check("an OtherPlant box too small to embed does not block whole", d["admission"] == MK.WHOLE, d)
    labels = [T, 5, 7, 2, 11, O, O, O, O, U]
    verdicts = [V.VERIFIED, V.CONFLICT, V.UNKNOWN, V.FAILED, V.SMALL, V.OTHER_OK, V.SMALL, V.CONFLICT, V.FAILED,
                MK.UNMAPPED]
    boxes = [(0.05 + 0.1 * i, 0.5, 0.04, 0.04) for i in range(len(labels))]
    d = MK.decide(labels, verdicts, boxes)
    check("masked: kept = verified target + OtherPlant other_ok/small; the rest masked (target conflict, unknown, "
          "failed, small; OtherPlant conflict, failed; unmapped)",
          d["admission"] == MK.MASKED and d["keep"] == [0, 5, 6] and d["mask"] == [1, 2, 3, 4, 7, 8, 9], d)
    d = MK.decide([T, O], [V.VERIFIED, V.UNKNOWN], far[:2])
    check("an OtherPlant box judged unknown (out-of-fold, no model) is masked, not kept",
          d["admission"] == MK.MASKED and d["mask"] == [1], d)
    d = MK.decide([T, 5], [V.VERIFIED, V.CONFLICT], [(0.3, 0.5, 0.2, 0.3), (0.31, 0.51, 0.2, 0.3)])
    check("a masked box at IoU >= 0.5 with a kept box -> refused_overlap, with the pair and its IoU",
          d["admission"] == MK.REFUSED_OVERLAP and d["overlaps"][0][:2] == [1, 0] and d["overlaps"][0][2] >= 0.5, d)
    d = MK.decide([T, 5], [V.VERIFIED, V.CONFLICT], [(0.3, 0.5, 0.2, 0.3), (0.42, 0.5, 0.2, 0.3)])
    check("a partial overlap below 0.5 is masked, not refused", d["admission"] == MK.MASKED
          and MK.iou((0.3, 0.5, 0.2, 0.3), (0.42, 0.5, 0.2, 0.3)) < 0.5, d)
    d = MK.decide([T, O], [V.CONFLICT, V.OTHER_OK], far[:2])
    check("no verified target box -> not_admitted (conflict image)", d["admission"] == MK.NOT_ADMITTED
          and d["image_verdict"] == V.CONFLICT and d["keep"] == [], d)
    d = MK.decide([O, O], [V.OTHER_OK, V.CONFLICT], far[:2])
    check("an OtherPlant-only image with a conflict -> not_admitted", d["admission"] == MK.NOT_ADMITTED, d)
    d = MK.decide([U], [MK.UNMAPPED], far[:1])
    check("an unmapped box alone -> not_admitted (never whole)", d["admission"] == MK.NOT_ADMITTED, d)
    check("mismatched lengths and ids outside 0..13 raise",
          raises(lambda: MK.decide([1], [], []), ValueError) and raises(lambda: MK.decide([14], ["x"], [far[0]]),
                                                                        ValueError))
    check("iou: identical 1, disjoint 0, with or without a class id",
          abs(MK.iou((0.5, 0.5, 0.2, 0.2), (3, 0.5, 0.5, 0.2, 0.2)) - 1.0) < 1e-12
          and MK.iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0)


def test_mask_except():
    print("mask_except")
    img = picture(TMP / "src" / "a.png")
    a = np.asarray(Image.open(img).convert("RGB"))
    H, W = a.shape[:2]
    mean = np.rint(a.reshape(-1, 3).astype(np.float64).mean(0)).astype(np.uint8)
    mboxes = [(1, 0.3, 0.4, 0.2, 0.3), (0, 0.8, 0.7, 0.15, 0.2)]
    kboxes = [(2, 0.36, 0.45, 0.2, 0.3)]                 # overlaps the first masked box, IoU < 0.5
    check("fixture: the kept box overlaps the first masked box below 0.5",
          0 < MK.iou(mboxes[0], kboxes[0]) < 0.5, MK.iou(mboxes[0], kboxes[0]))
    rec = MK.mask_except(img, mboxes, kboxes, TMP / "out", key="k1")
    b = np.asarray(Image.open(rec["path"]).convert("RGB"))
    fill = np.zeros((H, W), bool)
    for m in mboxes:
        x0, y0, x1, y1 = MK.pixel_rect(m, W, H)
        fill[y0:y1, x0:x1] = True
    kept = np.zeros((H, W), bool)
    x0, y0, x1, y1 = MK.pixel_rect(kboxes[0], W, H)
    kept[y0:y1, x0:x1] = True
    region = fill & ~kept
    check("masked pixels equal the image's mean colour", (b[region] == mean).all())
    check("every other pixel is unchanged (the kept box's pixels included)", (b[~region] == a[~region]).all())
    check("the kept pixels inside a masked rectangle stay visible", (b[fill & kept] == a[fill & kept]).all()
          and (fill & kept).any())
    check("masked_area_frac is the filled share of the frame",
          rec["masked_area_frac"] == round(region.sum() / float(W * H), 6) and rec["masked_px"] == int(region.sum()))
    x0, y0, x1, y1 = MK.pixel_rect(mboxes[0], W, H)
    want = round(float(kept[y0:y1, x0:x1].sum()) / ((x1 - x0) * (y1 - y0)), 6)
    check("visible_in_kept: the share of each masked box inside kept boxes", rec["visible_in_kept"] == [want, 0.0]
          and want > 0, rec["visible_in_kept"])
    check("the record names the mean colour and the frame", rec["mean_rgb"] == mean.tolist()
          and (rec["W"], rec["H"]) == (W, H))
    check("the file is <key>.<sha16>.png and hashes to its record",
          pathlib.Path(rec["path"]).name == "k1.%s.png" % rec["sha256"][:16]
          and V.C.sha256_file(rec["path"]) == rec["sha256"])
    mtime = os.stat(rec["path"]).st_mtime_ns
    rec2 = MK.mask_except(img, mboxes, kboxes, TMP / "out", key="k1")
    check("written once: the same call returns the same file untouched", rec2 == rec
          and os.stat(rec["path"]).st_mtime_ns == mtime)
    check("PNG, lossless", Image.open(rec["path"]).format == "PNG")

    print("overlap refused")
    check("a masked box at IoU >= 0.5 with a kept box is refused",
          raises(lambda: MK.mask_except(img, [(0.3, 0.4, 0.2, 0.3)], [(0.31, 0.41, 0.2, 0.3)], TMP / "out2", "k2"),
                 MK.MaskError, "overlaps") and not (TMP / "out2").exists())

    print("byte-identical to recover.mask without overlap")
    kb = [(4, 0.2, 0.8, 0.1, 0.1)]
    rec3 = MK.mask_except(img, mboxes, kb, TMP / "mine", key="k3")
    p, sha = R.mask(img, mboxes, TMP / "theirs", key="k3")
    check("same bytes, same sha256, same name", pathlib.Path(rec3["path"]).read_bytes() == p.read_bytes()
          and rec3["sha256"] == sha and pathlib.Path(rec3["path"]).name == p.name)
    rec4 = MK.mask_except(img, mboxes, [], TMP / "mine", key="k4")
    p4, _s = R.mask(img, [m[1:] for m in mboxes], TMP / "theirs", key="k4")
    check("no kept boxes at all: identical too (boxes with or without a class id)",
          pathlib.Path(rec4["path"]).read_bytes() == p4.read_bytes())
    edge = [(0.0, 0.0, 0.33, 0.21), (0.999, 0.999, 0.4, 0.4), (0.51234, 0.49876, 0.0331, 0.0777)]
    rec5 = MK.mask_except(img, edge, [], TMP / "mine", key="k5")
    p5, _s = R.mask(img, edge, TMP / "theirs", key="k5")
    check("boxes off the frame and fractional edges round the same way",
          pathlib.Path(rec5["path"]).read_bytes() == p5.read_bytes())
    rot = picture(TMP / "src" / "rot.jpg", W=80, H=50, seed=5, exif_orientation=6)
    rec6 = MK.mask_except(rot, mboxes, kb, TMP / "mine", key="k6")
    p6, _s = R.mask(rot, mboxes, TMP / "theirs", key="k6")
    check("an EXIF-rotated original: both write the transposed image (50 x 80), byte-identical",
          pathlib.Path(rec6["path"]).read_bytes() == p6.read_bytes() and (rec6["W"], rec6["H"]) == (50, 80))

    print("unreadable")
    bad = TMP / "src" / "bad.png"
    bad.write_bytes(b"not an image")
    check("an unreadable image raises MaskError", raises(lambda: MK.mask_except(bad, mboxes, [], TMP / "out3", "k7"),
                                                         MK.MaskError, "cannot open"))


def main():
    try:
        test_decide()
        test_mask_except()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
