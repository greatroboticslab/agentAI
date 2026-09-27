#!/usr/bin/env python3
"""INC Step 0.1: the splits every later INC number is measured on.

What is pinned, and why:
  * dev is made of whole capture sessions and meets the protocol's rules
    (about 15% of images, every species >= 30 boxes) and nothing more: the
    defaults carry no keep-in-train rule, which the protocol does not contain
    and which made the real cwd12 fail. The search is exact, so it is checked
    against brute force on random instances, feasible ones and ones that are
    infeasible for a reason other than a floor above a cap; ties go to the
    smallest sorted tuple of session names. When nothing is feasible the error
    names the contradiction and lists the relaxations, the protocol as written
    first. On the real cwd12 (when present) the defaults admit a dev.
  * train_core loses a re-encoded copy of a test image; an exam image that is
    a cwd12 train photograph leaves the exam instead
    (dHash within 6 bits), with the reason logged, and nothing else.
  * exam labels are converted into the INC class space and written under
    SPLITS_DIR, not next to the source: hand-checked boxes (XML, the json
    fallback, an EXIF-rotated image annotated as shown), Lambsquarters ->
    OtherPlant (and --other-plant adds names to it), ImageWeeds ragweed ->
    Ragweed and everything else -> OtherPlant, clipping and degenerate-box
    counts, and an unknown class name is a hard error. A COCO json that lists
    only other images is an error, not every box in the file.
  * an annotation that may be on another frame than the trainer sees is
    dropped: a transposed size, a transposing orientation with no size (json)
    or a square image, and, once a split shows raw-frame annotations, every
    EXIF-oriented image.
  * an exam never quietly shrinks: every image drop reason and degenerate
    boxes are capped (an images-first unpack stops the build), ImageWeeds
    images without a label file are background and capped too, and lock
    compares the images KEPT with the protocol's count.
  * a never-train index built while an exam is missing, or before lock
    accepted the counts, is refused by NeverTrainGuard.load and so by the
    merge guard; lock makes it final, and then it flags the planted test copy.
  * a missing exam source is skipped by build and blocks lock.
  * a rebuild that fails leaves every locked file as it was (labels are
    staged), so verify still passes.
  * keys are unique across all splits; exam dirs are read-only and never
    rebuilt silently; lock refuses to overwrite itself without --relock (and
    keeps the old one when relocking); verify catches a tampered manifest and
    a tampered image.

Everything runs in a fake REPO under a temp dir (REPO and INC_DIR are set
before the INC modules are imported), offline, on CPU, in seconds (plus a few
more for the real-cwd12 dev search when that dataset is on disk).

Run:  python3 tests/test_inc_splits.py
"""
import collections
import itertools
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_splits_test_"))
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools import mega_trainer as M  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import splits as S  # noqa: E402

assert C.REPO == TMP / "repo" and C.INC_DIR == TMP / "inc", "fake REPO not in effect"

FAILURES = []
RNG = np.random.default_rng(0)
DEV_ARGS = dict(frac_range=(0.13, 0.18), min_dev_boxes=2)
SESSION_SIZES = [30, 12, 11, 10, 9, 8, 8, 7, 6, 5]
SCORER = TMP / "fake_scorer.py"
REAL_CWD12 = ROOT.parent / "downloads" / "cottonweeddet12"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, text="", exc=S.SplitError):
    try:
        fn()
    except exc as e:
        return text in str(e), str(e)
    return False, "no error"


# ------------------------------------------------------------ fake data
def photo(path, w=96, h=72, exif_orientation=None):
    """A smooth random picture: distinct pictures are ~32 dHash bits apart,
    and a re-encoded copy stays within a few bits."""
    small = RNG.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    im = Image.fromarray(small).resize((w, h), Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"quality": 92}
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    im.save(path, **kw)


def reencoded_copy(src, dst):
    with Image.open(src) as im:
        im.convert("RGB").resize((im.width - 6, im.height - 4), Image.BILINEAR).save(dst, quality=60)


def yolo(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(v) for v in ln) + "\n" for ln in lines))


def voc(path, size, objects):
    objs = "".join(
        "<object><name>%s</name><bndbox><xmin>%s</xmin><ymin>%s</ymin><xmax>%s</xmax>"
        "<ymax>%s</ymax></bndbox></object>" % ((n,) + tuple(b)) for n, b in objects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<annotation><size><width>%d</width><height>%d</height><depth>3</depth>"
                    "</size>%s</annotation>" % (size[0], size[1], objs))


def vgg(path, image_name, x, y, w, h, cls="Carpetweed"):
    path.write_text(json.dumps({image_name + "123": {
        "filename": image_name, "size": 123, "file_attributes": {},
        "regions": [{"shape_attributes": {"name": "rect", "x": x, "y": y, "width": w, "height": h},
                     "region_attributes": {"CottonWeed": {cls: True}}}]}}))


def make_cwd12(repo):
    root = repo / "downloads" / "cottonweeddet12"
    g = 0
    for s, n in enumerate(SESSION_SIZES):
        for f in range(n):
            stem = "20210701_FakeCam_S%d_%d" % (s, f + 1)
            photo(root / "train" / "images" / (stem + ".jpg"))
            yolo(root / "train" / "labels" / (stem + ".txt"),
                 [((g + k) % 12, 0.2 + 0.05 * k, 0.5, 0.1, 0.1) for k in (0, 4, 8)])
            g += 1
    for split, n in (("valid", 8), ("test", 6)):
        for f in range(n):
            stem = "20210801_FakeCam_%s_%d" % (split.upper(), f + 1)
            photo(root / split / "images" / (stem + ".jpg"))
            yolo(root / split / "labels" / (stem + ".txt"), [(f % 12, 0.5, 0.5, 0.2, 0.2)])
    return root


def make_data2022(root):
    """8 images: 4 kept, 4 dropped (one per reason, 4 <= the always-ok 5)."""
    a, b = root / "fieldA", root / "fieldB"
    photo(a / "img22_0.jpg", 100, 80)
    voc(a / "img22_0.xml", (100, 80), [("Palmer Amaranth", (10, 20, 50, 60)),
                                       ("Lambsquarters", (60, 10, 90, 40))])
    photo(a / "img22_1.jpg", 100, 80)
    voc(a / "img22_1.xml", (100, 80), [("Carpetweed", (80, 60, 130, 95)),     # clipped
                                       ("Carpetweed", (30, 30, 30, 50)),      # degenerate
                                       ("Goosegrass", (5, 5, 25, 25))])
    photo(a / "img22_2.jpg", 100, 80)                                         # json only (VGG)
    vgg(a / "img22_2.json", "img22_2.jpg", 20, 8, 40, 32)
    photo(a / "img22_3.jpg", 100, 80)                                         # no annotation
    photo(a / "img22_5.jpg", 100, 80)                                         # annotated rotated
    voc(a / "img22_5.xml", (80, 100), [("Carpetweed", (8, 10, 40, 50))])
    photo(a / "img22_6.jpg", 100, 80, exif_orientation=3)                     # size agrees; the split
    voc(a / "img22_6.xml", (100, 80), [("Carpetweed", (60, 10, 95, 30))])     # has raw-frame evidence
    photo(a / "img22_7.jpg", 100, 80, exif_orientation=6)                     # json only, transposing
    vgg(a / "img22_7.json", "img22_7.jpg", 60, 10, 35, 20)
    photo(b / "img22_0.jpg", 100, 80)                                         # same stem as fieldA
    voc(b / "img22_0.xml", (100, 80), [("Waterhemp", (40, 40, 60, 60))])
    photo(b / "img22_8.jpg", 100, 80)                   # test_build plants a copy in cwd12 train
    voc(b / "img22_8.xml", (100, 80), [("Goosegrass", (20, 20, 50, 50))])


def make_data2023(root):
    """5 images, all kept: no raw-frame evidence here, so EXIF images stay."""
    for i in range(3):
        photo(root / ("img23_%d.jpg" % i), 120, 90)
        voc(root / ("img23_%d.xml" % i), (120, 90), [("Sicklepod", (10, 10, 60, 50))])
    photo(root / "img23_r6.jpg", 120, 90, exif_orientation=6)                 # shown as 90 x 120
    voc(root / "img23_r6.xml", (90, 120), [("Morningglory", (8, 10, 40, 50)),
                                           ("Spotted spurge", (0, 0, 20, 20))])
    photo(root / "img23_r3.jpg", 120, 90, exif_orientation=3)
    voc(root / "img23_r3.xml", (120, 90), [("Carpetweed", (60, 10, 96, 40))])


def make_imageweeds(root):
    for i in range(6):                     # iw_<id> carries ImageWeeds class id; iw_5 is unlabelled
        photo(root / "images" / ("iw_%d.jpg" % i), 100, 100)
        if i < 5:
            yolo(root / "labels" / ("iw_%d.txt" % i), [(i, 0.5, 0.5, 0.2, 0.2)])
    yolo(root / "labels" / "iw_0.txt", [(0, 0.5, 0.5, 0.2, 0.2), (3, 0.95, 0.5, 0.2, 0.2)])


# ------------------------------------------------------------ dev search
def brute_dev(sizes, boxes, lo, hi, target, floor, upper):
    names = sorted(sizes)
    total = sum(sizes.values())
    best = None
    for r in range(len(names) + 1):
        for combo in itertools.combinations(names, r):
            n = sum(sizes[s] for s in combo)
            if not lo * total - 1e-9 <= n <= hi * total + 1e-9:
                continue
            bx = [sum(boxes[s][c] for s in combo) for c in range(12)]
            if any(bx[c] < floor or bx[c] > upper[c] for c in range(12)):
                continue
            key = (abs(n - target * total), combo)
            best = key if best is None or key < best else best
    return best


def random_agreement(seed, n, min_boxes, keep, frac_range, target):
    """(agree, feasible, infeasible with every floor <= its cap) over n random
    instances, exact search vs brute force."""
    rnd = random.Random(seed)
    agree = feasible = hard_infeasible = 0
    for _ in range(n):
        names = ["s%02d" % i for i in range(11)]
        sizes = {s: rnd.randint(1, 30) for s in names}
        boxes = {s: [rnd.randint(0, 4) for _ in range(12)] for s in names}
        totals = [sum(boxes[s][c] for s in names) for c in range(12)]
        upper = [t - int(np.ceil(keep * t - 1e-9)) for t in totals]
        want = brute_dev(sizes, boxes, frac_range[0], frac_range[1], target, min_boxes, upper)
        try:
            got = tuple(S.choose_dev(sizes, boxes, target_frac=target, frac_range=frac_range,
                                     min_dev_boxes=min_boxes, min_keep_frac=keep,
                                     explain=False)["sessions"])
        except S.SplitError:
            got = None
        agree += (want is None and got is None) or (want is not None and want[1] == got)
        feasible += want is not None
        hard_infeasible += want is None and all(min_boxes <= u for u in upper)
    return agree, feasible, hard_infeasible


def test_choose_dev():
    print("dev search")
    check("the defaults are the protocol's: >= 30 boxes, 15% target, no keep-in-train rule",
          (S.DEV_MIN_BOXES, S.DEV_TARGET_FRAC, S.DEV_MIN_KEEP_FRAC) == (30, 0.15, 0.0)
          and S.DEV_RELAXATIONS[0] == (30, 0.0, False))

    one = [1] * 12
    sizes = {"e": 55, "d": 5, "c": 10, "b": 15, "a": 15}
    got = S.choose_dev(sizes, {s: one for s in sizes}, min_dev_boxes=0, frac_range=(0.10, 0.20))
    check("ties go to the smallest sorted tuple of session names ({a} over {b}, {c,d})",
          got["sessions"] == ["a"] and got["images"] == 15, got["sessions"])

    for label, args in (("keep rule 0.6", dict(seed=1, n=40, min_boxes=4, keep=0.6,
                                               frac_range=(0.13, 0.25), target=0.2)),
                        ("protocol shape", dict(seed=2, n=40, min_boxes=5, keep=0.0,
                                                frac_range=(0.13, 0.18), target=0.15))):
        agree, feasible, hard = random_agreement(**args)
        check("exact search equals brute force, %s (%d/%d agree; %d feasible, %d infeasible "
              "with every floor <= its cap)" % (label, agree, args["n"], feasible, hard),
              agree == args["n"] and feasible >= 5 and hard >= 5)

    # cwd12's shape: a rare species (81 boxes) can put 30 in dev, but not while
    # 70% of it stays in train (at most 24 may go).
    rare = {"x": 10, "y": 10, "z": 80}
    bx = {"x": [30] * 11 + [24], "y": [30] * 11 + [26], "z": [100] * 11 + [31]}
    got = S.choose_dev(rare, bx, frac_range=(0.05, 0.25))
    check("the defaults admit a dev where a 70% keep rule would not (x+y, 50 rare boxes)",
          got["sessions"] == ["x", "y"] and got["boxes"]["CutleafGroundcherry"] == 50, got["sessions"])
    ok, msg = raises(lambda: S.choose_dev(rare, bx, frac_range=(0.05, 0.25), min_keep_frac=0.7),
                     "CutleafGroundcherry")
    check("an opted-in keep rule no dev can meet fails, naming the species", ok, msg[:200])
    check("... and lists the relaxations, the protocol as written first and feasible",
          "Relaxations" in msg and "0.00 (the protocol as written): feasible" in msg, msg[-400:])
    got = S.choose_dev(rare, bx, frac_range=(0.05, 0.25), min_keep_frac=0.7, cap_rare=True)
    check("cap_rare lowers only the rare species' floor, and records it",
          got["search"]["capped"] == {"CutleafGroundcherry": 24} and got["sessions"] == ["x"],
          got["search"]["capped"])


def test_real_cwd12_dev():
    if not (REAL_CWD12 / "train" / "labels").is_dir():
        print("  skip the real cwd12 dev search: %s is not here" % REAL_CWD12)
        return
    print("dev search on the real cwd12 train (%s)" % REAL_CWD12)
    saved = S.cwd12_dir
    S.cwd12_dir = lambda: REAL_CWD12
    try:
        train = S.list_cwd12("train")
    finally:
        S.cwd12_dir = saved
    by = collections.defaultdict(list)
    for r in train:
        by[r["session"]].append(r)
    got = S.choose_dev({s: len(v) for s, v in by.items()},
                       {s: S.species_counts(b for r in v for b in r["boxes"]) for s, v in by.items()})
    low = min(got["boxes"].values())
    check("the protocol's defaults admit a dev on the real cwd12: %d sessions, %d of %d images "
          "(%.1f%%), min species boxes %d" % (len(got["sessions"]), got["images"], len(train),
                                              100 * got["frac"], low),
          low >= 30 and 0.13 <= got["frac"] <= 0.18 and set(got["sessions"]) <= set(by))


# ------------------------------------------------------------ converters
def test_converters():
    print("exam converters")
    src = TMP / "exif_src"
    photo(src / "ok.jpg", 100, 80)
    voc(src / "ok.xml", (100, 80), [("Carpetweed", (10, 10, 50, 50))])
    photo(src / "r6.jpg", 100, 80, exif_orientation=6)        # annotated on the stored pixels
    voc(src / "r6.xml", (100, 80), [("Carpetweed", (60, 10, 95, 30))])
    photo(src / "r3.jpg", 100, 80, exif_orientation=3)        # same, but a 180-degree turn keeps the size
    voc(src / "r3.xml", (100, 80), [("Carpetweed", (60, 10, 95, 30))])
    photo(src / "j6.jpg", 100, 80, exif_orientation=6)        # json only: no size to compare
    vgg(src / "j6.json", "j6.jpg", 60, 10, 35, 20)
    photo(src / "sq6.jpg", 80, 80, exif_orientation=6)       # square: a turn keeps the size
    voc(src / "sq6.xml", (80, 80), [("Carpetweed", (10, 10, 30, 30))])
    rows, st = S.convert_three_season(src, "ood22")
    check("raw-frame annotations: the transposed size is dropped as rotated; orientation 3 in "
          "that split, json-only orientation 6 and a square orientation 6 as unverifiable",
          [r["rel"] for r in rows] == ["ok.jpg"]
          and dict(st["dropped_images"]) == {"annotation_frame_rotated": 1,
                                              "annotation_frame_unverifiable": 3},
          ([r["rel"] for r in rows], dict(st["dropped_images"])))

    iwx = TMP / "iw_exif"
    photo(iwx / "images" / "a.jpg", 64, 64)
    yolo(iwx / "labels" / "a.txt", [(3, 0.5, 0.5, 0.2, 0.2)])
    photo(iwx / "images" / "b.jpg", 64, 48, exif_orientation=6)
    yolo(iwx / "labels" / "b.txt", [(3, 0.5, 0.5, 0.2, 0.2)])
    photo(iwx / "images" / "c.jpg", 64, 48, exif_orientation=6)   # background: nothing to misplace
    rows, st, _ = S.convert_imageweeds(iwx)
    check("ImageWeeds: an EXIF-oriented image with boxes is unverifiable (YOLO has no size); "
          "one without boxes is kept as background",
          sorted(r["rel"] for r in rows) == ["a.jpg", "c.jpg"]
          and dict(st["dropped_images"]) == {"annotation_frame_unverifiable": 1}
          and dict(st["background_images"]) == {"no_label_file": 1},
          (sorted(r["rel"] for r in rows), dict(st["dropped_images"])))

    unpack = TMP / "unpacking"
    for i in range(10):
        photo(unpack / ("a_%d.jpg" % i), 100, 80)
        if i < 3:
            voc(unpack / ("a_%d.xml" % i), (100, 80), [("Carpetweed", (10, 10, 50, 50))])
    ok, msg = raises(lambda: S.convert_three_season(unpack, "ood22"), "no_annotation")
    check("7 of 10 images without an annotation (an images-first unpack) stop the conversion",
          ok and "--max-drop-share" in msg, msg[:200])
    rows, st = S.convert_three_season(unpack, "ood22", max_drop_share=1.0)
    check("... unless --max-drop-share admits them after inspection",
          len(rows) == 3 and dict(st["dropped_images"]) == {"no_annotation": 7})
    iwu = TMP / "iw_unpacking"
    for i in range(10):
        photo(iwu / "images" / ("x_%d.jpg" % i), 64, 64)
        if i < 3:
            yolo(iwu / "labels" / ("x_%d.txt" % i), [(3, 0.5, 0.5, 0.2, 0.2)])
    ok, msg = raises(lambda: S.convert_imageweeds(iwu), "no_label_file")
    check("ImageWeeds: 7 of 10 images without a label file stop the conversion", ok, msg[:200])
    deg = TMP / "degenerate"
    photo(deg / "d.jpg", 100, 80)
    voc(deg / "d.xml", (100, 80), [("Carpetweed", (10, 10, 10, 50))] * 6 + [("Carpetweed", (10, 10, 50, 50))])
    ok, msg = raises(lambda: S.convert_three_season(deg, "ood22"), "boxes degenerate")
    check("more degenerate boxes than the cap stop the conversion", ok, msg[:200])

    coco = TMP / "coco.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [1, 1, 5, 5]},
                        {"image_id": 2, "category_id": 7, "bbox": [2, 2, 6, 6]}],
        "categories": [{"id": 7, "name": "Carpetweed"}]}))
    ok, msg = raises(lambda: S._json_boxes(coco, "c.jpg"), "none is c.jpg")
    check("a COCO json listing only other images is an error, not every box in the file", ok, msg[:200])
    check("... and the matching image gets only its own boxes",
          S._json_boxes(coco, "b.jpg")[0] == [("Carpetweed", 2.0, 2.0, 8.0, 8.0)])

    check("--other-plant adds to Lambsquarters instead of replacing it",
          S.other_plant_names(["Nutsedge"]) == ("Lambsquarters", "Nutsedge")
          and S.other_plant_names(["lambsquarters", "Nutsedge", "nutsedge"]) == ("Lambsquarters", "Nutsedge")
          and S.other_plant_names(None) == ("Lambsquarters",))
    bad = TMP / "unknown_src"
    photo(bad / "u.jpg", 50, 50)
    voc(bad / "u.xml", (50, 50), [("Nutsedge", (1, 1, 20, 20)), ("Carpetweed", (1, 1, 20, 20))])
    ok, msg = raises(lambda: S.convert_three_season(bad, "ood22"), "Nutsedge")
    check("an unknown class name is a hard error that names it", ok, msg[:200])
    check("... unless it is declared OtherPlant",
          S.convert_three_season(bad, "ood22", S.other_plant_names(["Nutsedge"]))[0][0]["boxes"][0][0] == 12)
    lbl = TMP / "bad_cls.txt"
    lbl.write_text("12 0.5 0.5 0.1 0.1\n")
    check("a cwd12 label with a class outside 0..11 is refused",
          raises(lambda: S._strict_yolo(lbl, 11), "outside 0..11")[0])

    e = [[1, "dev", "dev__a"], [2, "test", "test__b"]]
    full = S.index_record(e, [], {}, 1)
    part = S.index_record(e, ["ood23"], {}, 1)
    short = S.index_record(e, [], {"ood22": {"expected": 9, "found": 9, "kept": 8}}, 1)
    check("an index is complete only with nothing missing and every count as the protocol says",
          full["complete"] and full["min_expected"] == 2
          and not part["complete"] and part["min_expected"] > 2
          and not short["complete"] and short["min_expected"] > 2)


# ------------------------------------------------------------ build / lock / verify
def rows_of(split):
    return C.read_manifest(C.manifest_path(split))


def label_of(split, image_suffix):
    hit = [r for r in rows_of(split) if r["image"].endswith(image_suffix)]
    assert len(hit) == 1, (split, image_suffix, len(hit))
    return hit[0], C.read_yolo(hit[0]["label"])


def close(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and all(abs(u - v) < 1e-5 for u, v in zip(x[1:], y[1:]))
                                    for x, y in zip(a, b))


def index_file():
    return json.loads(C.NEVER_TRAIN_INDEX.read_text())


def test_build_lock_verify():
    repo = C.REPO
    cwd = make_cwd12(repo)
    d22 = repo / "downloads" / "3seasonweeddet10" / "data2022"
    make_data2022(d22)
    make_imageweeds(repo / "datasets" / S.IMAGEWEEDS_SLUG)
    # planted: a re-encoded test image and a re-encoded exam image in the big train session
    planted_test = cwd / "train" / "images" / "20210701_FakeCam_S0_901.jpg"
    reencoded_copy(cwd / "test" / "images" / "20210801_FakeCam_TEST_1.jpg", planted_test)
    yolo(cwd / "train" / "labels" / "20210701_FakeCam_S0_901.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    planted_exam = cwd / "train" / "images" / "20210701_FakeCam_S0_902.jpg"
    reencoded_copy(d22 / "fieldB" / "img22_8.jpg", planted_exam)
    yolo(cwd / "train" / "labels" / "20210701_FakeCam_S0_902.txt", [(1, 0.5, 0.5, 0.2, 0.2)])
    SCORER.write_text("# stand-in for inc/scorer.py\n")

    print("build with data2023 missing")
    summary = S.build(**DEV_ARGS)
    check("a missing exam source is skipped and recorded", summary["missing"] == ["ood23"],
          summary["missing"])
    check("no manifest is written for the missing split", not C.manifest_path("ood23").exists())
    ok, msg = raises(lambda: S.lock(scorer_path=SCORER), "ood23")
    check("lock refuses while an exam split is missing", ok, msg[:200])
    check("... and writes no LOCK.json", not C.LOCK_PATH.exists())
    idx = index_file()
    check("the partial never-train index says so (complete false, ood23 missing, "
          "min_expected above its %d entries)" % len(idx["entries"]),
          idx["complete"] is False and idx["missing"] == ["ood23"]
          and idx["min_expected"] > len(idx["entries"]), {k: idx[k] for k in idx if k != "entries"})
    ok, msg = raises(C.NeverTrainGuard.load, "expected >=", RuntimeError)
    check("NeverTrainGuard.load refuses the partial index", ok, msg[:200])
    ok, msg = raises(lambda: M._inc_train_guard(True), "expected >=", RuntimeError)
    check("... so the merge guard (mega_trainer._inc_train_guard) aborts instead of using it",
          ok, msg[:200])

    # --- dev and train_core
    print("dev and train_core")
    dev, core = rows_of("dev"), rows_of("train_core")
    train_sessions = {"20210701_FakeCam_S%d" % i: n for i, n in enumerate(SESSION_SIZES)}
    train_sessions["20210701_FakeCam_S0"] += 2
    dev_sessions = {r["session"] for r in dev}
    check("dev is whole sessions (every image of a dev session is in dev)",
          all(sum(1 for r in dev if r["session"] == s) == train_sessions[s] for s in dev_sessions))
    check("no dev session appears in train_core", not dev_sessions & {r["session"] for r in core})
    check("dev sessions recorded in summary.json",
          sorted(dev_sessions) == summary["dev"]["sessions"], summary["dev"]["sessions"])
    total = sum(train_sessions.values())
    frac = len(dev) / total
    check("dev share within 13-18%% (%.3f)" % frac, 0.13 <= frac <= 0.18)
    dev_counts = S.species_counts(b for r in dev for b in C.read_yolo(r["label"]))
    check("every species has >= min boxes in dev", min(dev_counts) >= DEV_ARGS["min_dev_boxes"],
          dev_counts)
    check("the fixture's 2-box floor is recorded as a deviation from the protocol's 30",
          any("30" in d for d in summary["dev"]["protocol_deviations"])
          and not any("keep" in d for d in summary["dev"]["protocol_deviations"]),
          summary["dev"]["protocol_deviations"])
    core_images = {pathlib.Path(r["image"]).name for r in core}
    check("the re-encoded test image is not in train_core", planted_test.name not in core_images)
    check("an exam image that is a cwd12 train photograph leaves the exam (counted) ...",
          summary["exam_cwd12_copies"]["ood22"]["dropped"] == 1
          and not any(r["image"].endswith("fieldB/img22_8.jpg") for r in rows_of("ood22")),
          summary["exam_cwd12_copies"])
    check("... and the train photograph stays in train_core", planted_exam.name in core_images)
    check("drops are logged by reason, and only the planted test copy is dropped",
          summary["dropped"]["train_core"] == {"near_test": 1}, summary["dropped"]["train_core"])
    check("train_core + dev + dropped = all train images",
          len(core) + len(dev) + 1 == total, (len(core), len(dev), total))
    check("dev/test labels are cwd12's own files", all("/downloads/cottonweeddet12/" in r["label"]
                                                        for s in ("dev", "test") for r in rows_of(s)))
    check("test = valid + test (14)", len(rows_of("test")) == 14)

    # --- exam conversion
    print("exam labels")
    r0, lb0 = label_of("ood22", "fieldA/img22_0.jpg")
    check("converted labels live under SPLITS_DIR/labels/ood22",
          r0["label"].startswith(str(C.SPLITS_DIR / "labels" / "ood22")), r0["label"])
    check("Palmer Amaranth box: (10,20,50,60) in 100x80 -> 8 0.3 0.5 0.4 0.5; Lambsquarters -> 12",
          close(lb0, [(8, 0.3, 0.5, 0.4, 0.5), (12, 0.75, 0.3125, 0.3, 0.375)]), lb0)
    _, lb1 = label_of("ood22", "img22_1.jpg")
    check("an out-of-image box is clipped, a zero-width box dropped",
          close(lb1, [(4, 0.9, 0.875, 0.2, 0.25), (10, 0.15, 0.1875, 0.2, 0.25)]), lb1)
    _, lb2 = label_of("ood22", "img22_2.jpg")
    check("json fallback (VGG rect x,y,width,height) when there is no XML",
          close(lb2, [(4, 0.4, 0.3, 0.4, 0.4)]), lb2)
    dropped = summary["dropped"]["ood22"]
    check("no annotation, a rotated-frame annotation, and two unverifiable frames are dropped "
          "and counted", dropped["images"] == {"no_annotation": 1, "annotation_frame_rotated": 1,
                                               "annotation_frame_unverifiable": 2}, dropped)
    check("degenerate and clipped boxes are counted",
          dropped["boxes"] == {"degenerate": 1} and summary["conversion"]["ood22"]["clipped_boxes"] == 1,
          (dropped, summary["conversion"]["ood22"]["clipped_boxes"]))
    ood_keys = [r["key"] for r in rows_of("ood22")]
    check("colliding stems (fieldA/img22_0, fieldB/img22_0) get distinct stable keys",
          sum(k.startswith("ood22__img22_0__") for k in ood_keys) == 2, ood_keys)

    _, iw0 = label_of("imageweeds", "iw_0.jpg")
    check("ImageWeeds horseweed -> OtherPlant, ragweed -> Ragweed (5), box clipped at the edge",
          close(iw0, [(12, 0.5, 0.5, 0.2, 0.2), (5, 0.925, 0.5, 0.15, 0.2)]), iw0)
    iw_cls = {pathlib.Path(r["image"]).name: [b[0] for b in C.read_yolo(r["label"])]
              for r in rows_of("imageweeds")}
    check("ImageWeeds kochia, corn, redrootpigweed -> OtherPlant; ragweed -> 5",
          [iw_cls["iw_%d.jpg" % i] for i in range(1, 5)] == [[12], [12], [5], [12]], iw_cls)
    check("an ImageWeeds image without a label file is kept as background (empty label), counted",
          iw_cls.get("iw_5.jpg") == [] and summary["dropped"]["imageweeds"]["images"] == {}
          and summary["dropped"]["imageweeds"]["kept_as_background"] == {"no_label_file": 1},
          summary["dropped"]["imageweeds"])
    check("the ImageWeeds class order is recorded as an unverified assumption",
          "UNVERIFIED" in summary["imageweeds_class_order"]["status"]
          and summary["imageweeds_class_order"]["names"][3] == "ragweed")

    # --- second build, every source present, through the CLI
    print("rebuild with data2023 present (CLI, --other-plant Nutsedge)")
    make_data2023(repo / "downloads" / "3seasonweeddet10" / "data2023")
    # The fixture's own source counts stand in for the protocol's: ood22 finds
    # 9 images and keeps 4, which must still count as a difference.
    S.EXPECTED_IMAGES = {"test": 14, "ood22": 9, "ood23": 5, "imageweeds": 6}
    rc = S.main(["build", "--dev-min-boxes", "2", "--other-plant", "Nutsedge"])
    summary = S.read_summary()
    check("the CLI build succeeds, nothing missing now", rc == 0 and summary["missing"] == [])
    check("--other-plant Nutsedge kept Lambsquarters (its boxes still map to OtherPlant)",
          summary["other_plant_names"] == ["Lambsquarters", "Nutsedge"])
    check("existing exam dirs are kept, the new one created",
          summary["exams"] == {"dev": "kept", "test": "kept", "ood22": "kept",
                               "ood23": "created", "imageweeds": "kept"}, summary["exams"])
    check("the count check compares the images KEPT (ood22: found 9 = expected, kept 4)",
          summary["count_warnings"] == {"ood22": {"expected": 9, "found": 9, "kept": 4}},
          summary["count_warnings"])
    _, lb6 = label_of("ood23", "img23_r6.jpg")
    check("EXIF-rotated image annotated as shown: normalised by the 90x120 the trainer sees",
          close(lb6, [(1, 24 / 90, 0.25, 32 / 90, 40 / 120), (3, 10 / 90, 10 / 120, 20 / 90, 20 / 120)]), lb6)
    _, lb3 = label_of("ood23", "img23_r3.jpg")
    check("an orientation-3 image is kept where the split shows no raw-frame annotation",
          close(lb3, [(4, 0.65, 25 / 90, 0.3, 30 / 90)]), lb3)
    all_keys = [r["key"] for s in S.ALL_SPLITS for r in rows_of(s)]
    check("keys are unique across every split (%d)" % len(all_keys), len(all_keys) == len(set(all_keys)))
    idx = index_file()
    eval_keys = {(s, r["key"]) for s in C.EVAL_SPLITS for r in rows_of(s)}
    check("the never-train index holds every evaluation image, at 6 bits",
          {(s, k) for _h, s, k in idx["entries"]} == eval_keys and idx["bits"] == 6)
    check("... but is not final while a count differs, and the guard refuses it",
          idx["complete"] is False and idx["missing"] == []
          and raises(C.NeverTrainGuard.load, "expected >=", RuntimeError)[0])
    exam = C.EXAMS_DIR / "test"
    modes = [os.stat(exam / "labels" / n).st_mode for n in os.listdir(exam / "labels")]
    check("exam label files and data.yaml are read-only",
          modes and all(not m & 0o222 for m in modes)
          and not os.stat(exam / "data.yaml").st_mode & 0o222)
    check("exam images link to the manifest images",
          all(os.path.islink(exam / "images" / n) for n in os.listdir(exam / "images")))

    victim = C.EXAMS_DIR / "ood22" / "labels" / sorted(os.listdir(C.EXAMS_DIR / "ood22" / "labels"))[0]
    saved = victim.read_bytes()
    os.chmod(victim, 0o644)
    victim.write_text("0 0.5 0.5 0.1 0.1\n")
    ok, msg = raises(lambda: S.build(other_plant_names=S.other_plant_names(["Nutsedge"]), **DEV_ARGS),
                     "differs")
    check("an exam dir that differs from the manifest is an error, never rebuilt", ok, msg[:200])
    victim.write_bytes(saved)
    os.chmod(victim, 0o444)

    print("lock")
    ok, msg = raises(lambda: S.lock(scorer_path=SCORER), "image counts differ")
    check("an exam count that differs from the protocol blocks the lock", ok, msg[:200])
    check("... and leaves the index not final", index_file()["complete"] is False)
    lk = S.lock(scorer_path=SCORER, accept_counts=True)
    idx = index_file()
    check("lock --accept-counts makes the index final (min_expected = entries, accepted counts kept)",
          idx["complete"] is True and idx["min_expected"] == len(idx["entries"]) == len(eval_keys)
          and set(idx["accepted_count_warnings"]) == {"ood22"} and "note" not in idx)
    check("LOCK.json records every manifest, the final index and the scorer",
          set(lk["manifests"]) == set(S.ALL_SPLITS) and lk["scorer_sha256"] == C.sha256_file(SCORER)
          and lk["nevertrain_sha256"] == C.sha256_file(C.NEVER_TRAIN_INDEX)
          and lk["splits_version"] == C.SPLITS_VERSION and "created_utc" in lk)
    guard = C.NeverTrainGuard.load()
    hits, _ = guard.check([str(planted_test), str(planted_exam)])
    check("the final index loads, flags the planted test copy and not the train photograph "
          "whose exam twin left the exam", [h[1] for h in hits] == ["test"], hits)
    g2, src = M._inc_train_guard(True)
    check("... and the merge guard uses it", src == "nevertrain_index" and g2.n == len(eval_keys))
    check("lock refuses to overwrite itself",
          raises(lambda: S.lock(scorer_path=SCORER, accept_counts=True), "--relock")[0])
    check("build refuses once locked", raises(lambda: S.build(**DEV_ARGS), "locked")[0])
    S.lock(scorer_path=SCORER, accept_counts=True, relock=True)
    archived = [n for n in os.listdir(C.SPLITS_DIR) if n.startswith("LOCK.") and n != "LOCK.json"]
    log_lines = (C.SPLITS_DIR / S.LOCK_LOG_NAME).read_text().splitlines()
    check("--relock keeps the previous lock and logs the diff",
          len(archived) == 1 and len(log_lines) == 2 and "diff" in json.loads(log_lines[1]),
          (archived, len(log_lines)))

    print("verify")
    S.default_scorer_path = lambda: SCORER
    check("verify passes on the intact splits", S.verify() == [], S.verify()[:3])
    check("the CLI verify exits 0", S.main(["verify"]) == 0)

    man = C.manifest_path("dev")
    saved = man.read_bytes()
    os.chmod(man, 0o644)
    lines = saved.decode().splitlines(True)
    row = json.loads(lines[0])
    row["label_sha256"] = "0" * 64
    man.write_text(json.dumps(row, sort_keys=True) + "\n" + "".join(lines[1:]))
    probs = S.verify()
    check("a tampered manifest is caught (hash vs LOCK and row vs file)",
          any("changed since it was locked" in p for p in probs)
          and any("label" in p and "changed" in p for p in probs), probs[:3])
    check("the CLI verify exits non-zero on it", S.main(["verify"]) == 1)
    man.write_bytes(saved)
    os.chmod(man, 0o444)

    img = pathlib.Path(rows_of("test")[0]["image"])
    saved = img.read_bytes()
    img.write_bytes(saved + b"\0")
    probs = S.verify()
    check("a tampered image is caught", any(str(img) in p and "changed" in p for p in probs), probs[:3])
    img.write_bytes(saved)
    check("verify passes again once restored", S.verify() == [])

    print("a rebuild that fails")
    xml = d22 / "fieldA" / "img22_0.xml"
    orig = xml.read_text()
    xml.write_text(orig.replace("<xmax>50</xmax>", "<xmax>55</xmax>"))
    lbl_dir = S.converted_label_dir("ood22")
    before = {n: (lbl_dir / n).read_bytes() for n in os.listdir(lbl_dir)}
    ok, msg = raises(lambda: S.build(rebuild_locked=True, other_plant_names=S.other_plant_names(["Nutsedge"]),
                                     **DEV_ARGS), "differs")
    check("a --rebuild-locked whose exam labels changed fails on the exam dir", ok, msg[:200])
    check("... leaves the converted labels byte for byte as locked, and no staging dir",
          {n: (lbl_dir / n).read_bytes() for n in os.listdir(lbl_dir)} == before
          and not S.staging_dir().exists())
    check("... so verify still passes, before the source is even restored", S.verify() == [], S.verify()[:3])
    xml.write_text(orig)

    gone = repo / "downloads" / "3seasonweeddet10" / "data2023"
    gone.rename(gone.with_name("data2023.away"))
    ok, msg = raises(lambda: S.build(rebuild_locked=True, **DEV_ARGS), "source is gone")
    check("a source that vanishes after its manifest was written is an error", ok, msg[:200])
    gone.with_name("data2023.away").rename(gone)

    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.splits", "summary"],
                         cwd=str(ROOT), capture_output=True, text=True, env=dict(os.environ))
    check("the CLI summary runs and reports the lock and the index",
          out.returncode == 0 and "LOCK.json: present" in out.stdout
          and "never-train index" in out.stdout, out.stdout[-300:] + out.stderr[-300:])


def main():
    try:
        test_choose_dev()
        test_real_cwd12_dev()
        test_converters()
        test_build_lock_verify()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
