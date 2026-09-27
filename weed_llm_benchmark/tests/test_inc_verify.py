#!/usr/bin/env python3
"""INC Step 1: the box verifier (labeler Phase B).

What is pinned, and why:
  * the class join: trainer slot -> INC id is the exact inverse of
    cwd12_species.CWD12_ID_TO_SLOT for every id, aux slots become OtherPlant,
    and a harvested class list joins by the species it names (aliases and
    legacy-looking names included; a whole legacy list is read as legacy
    labels; no names at all -> every box OtherPlant).
  * the pre-v3.60.0 join, rebuilt for calibration: legacy-label string match,
    names deleted in a "cottonweed dataset", cottonweed_holdout through its
    four-name list (the documented wrong joins: Purslane -> Palmer amaranth,
    Ragweed -> Sicklepod, zig-zag Crabgrass / Nutsedge -> Morning glory /
    Ragweed, Waterhemp deleted).
  * the OtherPlant training names: genus names and variants of cwd12 species
    ("Amaranthus sp.", "pigweed", "spurge", ...) are kept out.
  * joint thresholds: a probe that ranks every held-out crop first verifies
    >= 95 % of every species (the AND of two separate 95 % tests gave ~0.93).
  * pool: skips NEVER_TRAIN, user-flagged, autolabelled, missing, oddly laid
    out and non-bbox slugs, each counted; reads the two leave-4-out copies for
    their train_core photographs only; refuses a never-train index that
    differs from LOCK.json and a pre-v3.60.0 mega_trainer; drops a re-encoded
    copy of a dev image (never-train index), keeps copies of train_core photos
    apart as cwd12_copies.jsonl, drops exact duplicates across slugs,
    unhashable images, unlabelled and unmapped images; label files are named
    by their content and never rewritten; source labels are never touched; a
    re-run reuses the hash cache and gives the same manifest.
  * verdicts on hand-made probabilities, and the image admission rule.
  * embed: shards and chunks resume after a kill, finished shards are skipped,
    an incomplete shard set is refused, rows are aligned to crop ids and a crop
    whose image cannot be read is a NaN row that shifts nothing.
  * calibrate (a) box-matches copies to their train_core original and scores
    them under the current join (a planted wrong label) and the old join
    (wrong and deleted boxes known); calibrate (b) catches seeded label swaps,
    spread over the species and the same on every run.
  * admit: verified.jsonl holds exactly the images the rule admits; conflicts
    go to conflicts.csv and the crop sheet; the OtherPlant training crops are
    judged out of fold; another shard set than the verifier's is refused.
  * stale inputs: after `pool` alone re-runs on a changed registry, embed, fit,
    calibrate and admit refuse, and the labels an earlier verified.jsonl names
    are unchanged; the full chain then finds the new conflicts.

Everything runs in a fake REPO under a temp dir (REPO and INC_DIR are set
before the INC modules are imported), offline, on CPU, without open_clip: a
fake embedder reads the colour painted in each box, so its features cluster
by the TRUE class whatever the label says.

Run:  python3 tests/test_inc_verify.py
"""
import csv
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zlib

# Worker processes (spawn) re-import this file; they must reuse the same TMP.
TMP = pathlib.Path(os.environ.get("INC_VERIFY_TEST_TMP") or tempfile.mkdtemp(prefix="inc_verify_test_"))
os.environ["INC_VERIFY_TEST_TMP"] = str(TMP)
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
from weed_optimizer_framework.tools import mega_trainer as M  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.semisup_labeler import _cut  # noqa: E402

assert C.REPO == TMP / "repo" and C.INC_DIR == TMP / "inc", "fake REPO not in effect"

FAILURES = []
RNG = np.random.default_rng(int(os.environ.get("INC_VERIFY_TEST_SEED", "7")))   # fixture seed
REPO = C.REPO
DS = REPO / "datasets"
IMG_W, IMG_H = 160, 120

# 13 box colours, one per INC class, >= 128 apart (a JPEG cannot confuse them)
COLORS = [p for p in itertools.product((0, 128, 255), repeat=3)
          if p not in ((128, 128, 128), (0, 0, 0), (255, 255, 255))][:13]

A_NAMES = ["Waterhemp", "Morningglory", "Carpetweeds", "Amaranthus palmeri", "Crabgrass",
           "Lambsquarters", "Ragweed", "Goosegrass", "Purslane", "Spotted spurge", "Eclipta",
           "Prickly sida", "Sicklepod", "Cutleaf groundcherry"]
A_INC = [0, 1, 4, 8, 12, 12, 5, 10, 2, 3, 6, 7, 9, 11]
A_SRC = {}                                   # INC species id -> a_species source id
for _i, _c in enumerate(A_INC):
    A_SRC.setdefault(_c, _i)
LAMBS = 5                                    # a_species source id of an OtherPlant name
B_NAMES = ["corn", "Giant ragweed", "weed", "pigweed"]


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# ------------------------------------------------------------ fake data
HASHES = []


def _bits(a, b):
    return bin(a ^ b).count("1")


def paint(path, boxes, texture, fmt):
    """A smooth random picture with each box (cls, cx, cy, w, h) filled with its
    class colour; texture adds per-pixel noise inside the boxes. Regenerated
    until it is > 12 dHash bits from every picture made so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(50):
        small = RNG.integers(40, 216, size=(6, 8, 3), dtype=np.uint8)
        arr = np.array(Image.fromarray(small).resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.int16)
        for c, cx, cy, w, h in boxes:
            x0, x1 = int(round((cx - w / 2) * IMG_W)), int(round((cx + w / 2) * IMG_W))
            y0, y1 = int(round((cy - h / 2) * IMG_H)), int(round((cy + h / 2) * IMG_H))
            patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.int16) + np.array(COLORS[c], dtype=np.int16)
            if texture:
                patch += RNG.integers(-14, 15, size=patch.shape).astype(np.int16)
            arr[y0:y1, x0:x1] = patch
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if fmt == "jpg":
            im.save(path, quality=92)
        else:
            im.save(path)
        h = C.dhash(path)
        if all(_bits(h, o) > 12 for o in HASHES):
            HASHES.append(h)
            return h
    raise AssertionError("could not make a distinct fixture picture")


def yolo(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(v) for v in ln) + "\n" for ln in lines))


def layout3(classes, jitter=True):
    """Three boxes side by side, (cls, cx, cy, w, h)."""
    out = []
    for k, c in enumerate(classes):
        dx = float(RNG.uniform(-0.02, 0.02)) if jitter else 0.0
        dy = float(RNG.uniform(-0.08, 0.08)) if jitter else 0.0
        out.append((c, round(0.18 + 0.32 * k + dx, 4), round(0.5 + dy, 4), 0.22, 0.32))
    return out


def reencoded_copy(src, dst):
    """A re-export: slightly resized, JPEG q60 (a few dHash bits away)."""
    with Image.open(src) as im:
        im.convert("RGB").resize((im.width - 6, im.height - 4), Image.BILINEAR).save(dst, quality=60)


def shift_copy(src, dst, px=4):
    """A re-export cropped by px on the left and resized back (JPEG q92)."""
    with Image.open(src) as im:
        im.convert("RGB").crop((px, 0, im.width, im.height)).resize(im.size, Image.BILINEAR).save(
            dst, quality=92)


def clean_copy(src, dst):
    """A cleaner re-export of the same photograph (blur, JPEG q95)."""
    with Image.open(src) as im:
        im.convert("RGB").filter(ImageFilter.GaussianBlur(1.2)).save(dst, quality=95)


def manifest_row(split, stem, img, lbl, session, source):
    return {"image": str(img), "label": str(lbl), "sha256": C.sha256_file(img),
            "label_sha256": C.sha256_file(lbl), "source": source, "session": session,
            "key": "%s__%s" % (split, stem)}


def make_splits():
    """cwd12 train_core (10 sessions x 4 photos x 3 boxes, every session holds
    all 12 species), dev (2 sessions), test (3 photos), the never-train index."""
    root = REPO / "downloads" / "cottonweeddet12"
    rows = {"train_core": [], "dev": [], "test": []}
    boxes_of = {}
    for split, sessions, frames in (("train_core", ["20210701_Cam_S%d" % s for s in range(10)], 4),
                                    ("dev", ["20210702_Cam_D%d" % d for d in range(2)], 3),
                                    ("test", ["20210703_Cam_T0"], 3)):
        sub = "valid" if split == "test" else "train"
        for s, sess in enumerate(sessions):
            for f in range(frames):
                stem = "%s_%d" % (sess, f + 1)
                boxes = layout3([(s + 3 * f + k) % 12 for k in range(3)])
                img = root / sub / "images" / (stem + ".jpg")
                lbl = root / sub / "labels" / (stem + ".txt")
                paint(img, boxes, texture=True, fmt="jpg")
                yolo(lbl, boxes)
                boxes_of[str(img)] = boxes
                rows[split].append(manifest_row(split, stem, img, lbl, sess if split != "test" else "",
                                                "cottonweeddet12/%s" % sub))
    for split, rs in rows.items():
        C.write_manifest(C.manifest_path(split), rs)
    entries = [[C.dhash(r["image"]), split, r["key"]] for split in ("dev", "test") for r in rows[split]]
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries, "min_expected": len(entries)}))
    return rows, boxes_of


def make_pool(rows, boxes_of):
    """Three harvested datasets plus one registry entry per skip reason.
    Returns what the test expects to find."""
    exp = {"copies": {}}
    a, b, c = DS / "a_species", DS / "b_other", DS / "c_nonames"
    core = rows["train_core"]
    # a_species: flat layout, every cwd12 species named (incl. legacy-looking names)
    for i in range(6):
        sp = [(2 * i) % 12, (2 * i + 1) % 12]
        bx = layout3(sp + [12])
        paint(a / "images" / ("a_ok_%d.png" % i), bx, texture=False, fmt="png")
        yolo(a / "labels" / ("a_ok_%d.txt" % i),
             [(A_SRC[x[0]] if x[0] < 12 else LAMBS,) + x[1:] for x in bx])
    bx = layout3([2, 12, 12])                     # painted Purslane, labelled Waterhemp
    paint(a / "images" / "a_bad.png", bx, texture=False, fmt="png")
    yolo(a / "labels" / "a_bad.txt", [(A_SRC[0],) + bx[0][1:], (4,) + bx[1][1:], (LAMBS,) + bx[2][1:]])
    bx = [(3, 0.3, 0.5, 0.22, 0.32), (7, 0.8, 0.8, 0.03, 0.04)]    # second box 5 x 5 px
    paint(a / "images" / "a_small.png", bx, texture=False, fmt="png")
    yolo(a / "labels" / "a_small.txt", [(A_SRC[b0[0]],) + b0[1:] for b0 in bx])
    # a re-encoded copy of a dev photograph
    dev0 = rows["dev"][0]
    reencoded_copy(dev0["image"], a / "images" / "a_evalcopy.jpg")
    yolo(a / "labels" / "a_evalcopy.txt", [(A_SRC[0], 0.5, 0.5, 0.2, 0.2)])
    exp["evalcopy_bits"] = _bits(C.dhash(a / "images" / "a_evalcopy.jpg"), C.dhash(dev0["image"]))
    # copies of train_core photographs: two labelled right, one with a planted wrong box
    for name, src_row, wrong in (("a_copy_ok1", core[0], False), ("a_copy_ok2", core[5], False),
                                 ("a_copy_planted", core[9], True)):
        clean_copy(src_row["image"], a / "images" / (name + ".jpg"))
        truth = boxes_of[src_row["image"]]
        lab = []
        for k, t in enumerate(truth):
            cls = (t[0] + 5) % 12 if (wrong and k == 0) else t[0]
            lab.append((A_SRC[cls],) + tuple(t[1:]))
        yolo(a / "labels" / (name + ".txt"), lab)
        exp["copies"]["a_species__" + name] = src_row["key"]
    exp["planted_truth"] = boxes_of[core[9]["image"]][0][0]
    # images the pool must drop
    paint(a / "images" / "a_nolabel.png", layout3([0, 1, 2]), texture=False, fmt="png")
    paint(a / "images" / "a_empty.png", layout3([0, 1, 2]), texture=False, fmt="png")
    yolo(a / "labels" / "a_empty.txt", [])
    (a / "images" / "a_corrupt.jpg").write_bytes(b"not a picture at all")
    yolo(a / "labels" / "a_corrupt.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    paint(a / "images" / "a_unmapped.png", layout3([0, 1, 2]), texture=False, fmt="png")
    yolo(a / "labels" / "a_unmapped.txt", [(99, 0.5, 0.5, 0.2, 0.2)])

    # b_other: train/valid layout, no cwd12 species among its names
    for i in range(4):
        split = "train" if i < 3 else "valid"
        bx = layout3([12, 12, 12])
        paint(b / split / "images" / ("b_ok_%d.png" % i), bx, texture=False, fmt="png")
        yolo(b / split / "labels" / ("b_ok_%d.txt" % i),
             [((3 if i == 3 and k == 0 else k % 3),) + x[1:] for k, x in enumerate(bx)])
    bx = layout3([9, 12, 12])                     # painted Sicklepod, labelled "weed"
    paint(b / "train" / "images" / "b_conflict.png", bx, texture=False, fmt="png")
    yolo(b / "train" / "labels" / "b_conflict.txt", [(2 if k == 0 else 0,) + x[1:] for k, x in enumerate(bx)])
    shutil.copyfile(a / "images" / "a_ok_0.png", b / "train" / "images" / "b_dup.png")
    yolo(b / "train" / "labels" / "b_dup.txt", [(0, 0.5, 0.5, 0.2, 0.2)])

    # c_nonames: no class names at all -> every box OtherPlant
    for i in range(2):
        bx = layout3([12, 12, 12])
        paint(c / "images" / ("c_ok_%d.png" % i), bx, texture=False, fmt="png")
        yolo(c / "labels" / ("c_ok_%d.txt" % i), [(0,) + x[1:] for x in bx])
    clean_copy(core[14]["image"], c / "images" / "c_copy.jpg")
    yolo(c / "labels" / "c_copy.txt", [(k,) + tuple(t[1:]) for k, t in enumerate(boxes_of[core[14]["image"]])])
    exp["copies"]["c_nonames__c_copy"] = core[14]["key"]
    # a re-export 4-5 dHash bits from its train_core original (a 4 px crop):
    # past the general 3-bit radius, inside the 6 bits a train_core copy gets
    for ci in [i for i in range(len(core)) if i not in (0, 4, 5, 9, 14, 24, 29, 33)]:
        shift_copy(core[ci]["image"], c / "images" / "c_copy_shift.jpg")
        bits = _bits(C.dhash(c / "images" / "c_copy_shift.jpg"), C.dhash(core[ci]["image"]))
        if bits in (4, 5):
            break
    else:
        raise AssertionError("no train_core picture gives a 4-5 bit shifted copy")
    yolo(c / "labels" / "c_copy_shift.txt", [(0,) + tuple(t[1:]) for t in boxes_of[core[ci]["image"]]])
    exp["copies"]["c_nonames__c_copy_shift"] = core[ci]["key"]
    exp["shift_bits"] = bits
    exp["copy_bits"] = {k: _bits(C.dhash(p), C.dhash(q)) for k, p, q in (
        ("a_copy_ok1", a / "images" / "a_copy_ok1.jpg", core[0]["image"]),
        ("a_copy_ok2", a / "images" / "a_copy_ok2.jpg", core[5]["image"]),
        ("a_copy_planted", a / "images" / "a_copy_planted.jpg", core[9]["image"]),
        ("c_copy", c / "images" / "c_copy.jpg", core[14]["image"]))}

    # the leave-4-out copies: byte copies of train_core photographs (labels in
    # their own id spaces), a dev photograph, and a picture that is not cwd12
    hc, sp = DS / "holdoutcopy", DS / "sp8copy"
    exp["calib"] = {}
    for root, slug, cores, to_local in ((hc, "cottonweed_holdout", (4, 33), lambda c: c),
                                        (sp, "cottonweed_sp8", (24, 29), lambda c: S.CWD12_ID_TO_SLOT[c])):
        for ci in cores:
            r = core[ci]
            stem = pathlib.Path(r["image"]).stem
            (root / "train" / "images").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["image"], root / "train" / "images" / (stem + ".jpg"))
            truth = boxes_of[r["image"]]
            assert all(to_local(t[0]) < (12 if slug == "cottonweed_holdout" else 8) for t in truth)
            yolo(root / "train" / "labels" / (stem + ".txt"), [(to_local(t[0]),) + tuple(t[1:]) for t in truth])
            exp["calib"]["%s__train__%s" % (slug, stem)] = (r["key"], [t[0] for t in truth])
        paint(root / "train" / "images" / "notcwd.png", layout3([12, 12, 12]), texture=False, fmt="png")
        yolo(root / "train" / "labels" / "notcwd.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    dev1 = rows["dev"][1]
    (hc / "valid" / "images").mkdir(parents=True)
    shutil.copyfile(dev1["image"], hc / "valid" / "images" / "devcopy.jpg")
    yolo(hc / "valid" / "labels" / "devcopy.txt", C.read_yolo(dev1["label"]))

    # one entry per skip reason
    for slug in ("d_flagged", "e_autolabel", "h_classification"):
        paint(DS / slug / "images" / "x.png", layout3([12, 12, 12]), texture=False, fmt="png")
        yolo(DS / slug / "labels" / "x.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    (DS / "g_weird").mkdir(parents=True)
    shutil.copyfile(a / "images" / "a_ok_1.png", DS / "g_weird" / "loose.png")
    reg = {"datasets": {
        "a_species": {"local_path": str(a), "annotation": "bbox", "class_names": A_NAMES},
        "b_other": {"local_path": str(b), "annotation": "yolo", "class_names": B_NAMES},
        "c_nonames": {"local_path": str(c), "annotation": "bbox", "class_names": []},
        "cottonweeddet12": {"local_path": str(REPO / "downloads" / "cottonweeddet12"),
                            "annotation": "bbox", "class_names": []},
        "cottonweed_sp8": {"local_path": str(DS / "sp8copy"), "annotation": "bbox",
                           "class_names": list(S.CWD12_ID_SPACE["cottonweed_sp8"])},
        "cottonweed_holdout": {"local_path": str(DS / "holdoutcopy"), "annotation": "bbox",
                               "class_names": list(S.CWD12_SPECIES)},
        "d_flagged": {"local_path": str(DS / "d_flagged"), "annotation": "bbox", "class_names": ["corn"]},
        "e_autolabel": {"local_path": str(DS / "e_autolabel"), "annotation": "yolo_autolabel"},
        "f_missing": {"local_path": str(DS / "f_missing"), "annotation": "bbox", "class_names": ["corn"]},
        "g_weird": {"local_path": str(DS / "g_weird"), "annotation": "bbox", "class_names": ["corn"]},
        "h_classification": {"local_path": str(DS / "h_classification"), "annotation": "classification"},
    }}
    fw = REPO / "results" / "framework"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "dataset_registry.json").write_text(json.dumps(reg))
    (fw / "dataset_flags.json").write_text(json.dumps({"d_flagged": {"flag": "garbage", "reason": "t"}}))
    return exp


# ------------------------------------------------------------ fake embedder
def fake_feature(pil):
    """Features that cluster by the colour painted in the box (the TRUE class):
    a one-hot of the nearest class colour, plus noise that grows with the
    texture inside the box (a flat box sits at its class centre)."""
    a = np.asarray(pil.convert("RGB"), dtype=np.float64)
    ctr = a[100:124, 100:124].reshape(-1, 3)
    k = int(np.argmin(((np.array(COLORS, dtype=np.float64) - ctr.mean(0)) ** 2).sum(1)))
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(a[100:124, 100:124]).tobytes()))
    v = np.zeros(24)
    v[k] = 1.0
    return v + rng.normal(0, 1, 24) * 0.012 * float(ctr.std(0).mean())


class Kill(BaseException):
    """A job killed mid-shard (not an Exception: the embed code must not catch it)."""


class FakeEmbedder:
    name = "fake-colour"
    dim = 24

    def __init__(self, kill_after=None):
        self.calls = 0
        self.crops = 0
        self.kill_after = kill_after

    def __call__(self, pils):
        if self.kill_after is not None and self.calls >= self.kill_after:
            raise Kill()
        self.calls += 1
        self.crops += len(pils)
        return np.stack([fake_feature(p) for p in pils])


def args(*argv):
    return V.parse_args(list(argv))


# ------------------------------------------------------------ unit tests
def legacy_class_map(slug, info):
    """A pre-v3.60.0 style mega_trainer join (by legacy slot label)."""
    return ({i: M.CANONICAL_12_NAMES.index(n) if n in M.CANONICAL_12_NAMES else 50
             for i, n in enumerate(info.get("class_names") or [])}, [])


def test_join():
    print("class join")
    ok = all(V.inc_id_of_slot(S.CWD12_ID_TO_SLOT[i]) == i for i in range(12))
    check("slot -> INC id inverts CWD12_ID_TO_SLOT for every cwd12 id", ok)
    check("each slot's species is the INC id's species",
          all(M.CANONICAL_12_SPECIES[s] == C.CLASS_NAMES[V.inc_id_of_slot(s)] for s in range(12)))
    check("aux slots 12-99 are OtherPlant", all(V.inc_id_of_slot(s) == 12 for s in range(12, 100)))
    names = list(S.CWD12_SPECIES)
    RNG.shuffle(names)
    to_inc, _n, wild = V.class_join("x_all12", {"class_names": names, "annotation": "bbox"})
    check("a dataset naming all 12 species joins each name to its INC id",
          not wild and all(to_inc[i] == C.CLASS_NAMES.index(n) for i, n in enumerate(names)), to_inc)
    to_inc, _n, _w = V.class_join("a_species", {"class_names": A_NAMES, "annotation": "bbox"})
    check("aliases and legacy-looking names join by species (Carpetweeds -> Carpetweed, "
          "Crabgrass -> OtherPlant)", [to_inc[i] for i in range(len(A_NAMES))] == A_INC, to_inc)
    to_inc, _n, _w = V.class_join("x_legacy", {"class_names": list(S.CWD12_LEGACY_LABELS),
                                               "annotation": "bbox"})
    check("a whole legacy list in cwd12 order is read as legacy labels (id i -> INC i)",
          to_inc == {i: i for i in range(12)}, to_inc)
    to_inc, _n, _w = V.class_join("x_slotlegacy", {"class_names": list(S.TRAINER_SLOT_LEGACY),
                                                   "annotation": "bbox"})
    check("a whole legacy list in slot order maps slot i to its species' INC id",
          to_inc == {i: S.CWD12_SPECIES.index(S.TRAINER_SLOT_SPECIES[i]) for i in range(12)}, to_inc)
    _t, _n, wild = V.class_join("x_none", {"class_names": [], "annotation": "bbox"})
    check("no class names -> wildcard (every box OtherPlant)", wild)
    to_inc, src, _w = V.class_join("x_dict", {"class_names": {"0": "Waterhemp", "2": "corn"},
                                              "annotation": "bbox"})
    check("class_names as {id: name} keeps the ids", to_inc == {0: 0, 1: 12, 2: 12} and src[2] == "corn",
          (to_inc, src))

    p = TMP / "src_label.txt"
    p.write_text("3 0.5 0.5 0.2 0.2\n"
                 "1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"      # polygon -> its box
                 "2 0.95 0.5 0.2 0.2\n"                     # runs off the right edge
                 "4 0.5 0.5 0 0.2\n"                        # degenerate
                 "x 0.5 0.5 0.2 0.2\n"                      # malformed
                 "5 0.5 0.5 0.2\n")                         # too few columns
    boxes, bad, clipped = V.read_source_label(p)
    check("source labels: box, polygon -> box, clipped box; 3 bad lines",
          len(boxes) == 3 and bad == 3 and clipped == 1
          and np.allclose(boxes[1][1:], (0.2, 0.25, 0.2, 0.3))
          and np.allclose(boxes[2][1:], (0.925, 0.5, 0.15, 0.2)), (boxes, bad, clipped))

    for name, want in (("corn", True), ("Giant ragweed", True), ("Lambsquarters", True),
                       ("Crabgrass", True), ("weed", False), ("Weeds", False), ("", False),
                       ("Waterhemp", False), ("class3", False), ("leaf blight", False),
                       ("Broadleaf weed", False), ("aphid", False)):
        check("named_other_plant(%r) is %s" % (name, want), V.named_other_plant(name) == want)
    related = ["amaranth", "Amaranthus", "Amaranthus sp.", "pigweed", "Redroot pigweed", "Palmer",
               "palmer amaranth seedling", "waterhemp_seedling", "Waterhemp-Palmer", "ragweed sp",
               "morning glory sp", "Ipomoea sp", "spurge", "Euphorbia", "Sida", "Physalis",
               "groundcherry", "Ambrosia", "Senna", "Cassia tora", "Mollugo", "Portulaca",
               "horse purslane", "Eleusine", "goosegrass seedling", "sicklepod_young", "false daisy seedling"]
    bad = [n for n in related if V.other_name_status(n) != "cwd12_related"]
    check("genus names and variants of cwd12 species never feed the OtherPlant sample", not bad, bad)
    keys = [k for k in S._ALIASES]
    check("every cwd12 alias key holds a CWD12_RELATED_TOKENS token",
          all(any(t in k for t in V.CWD12_RELATED_TOKENS) for k in keys),
          [k for k in keys if not any(t in k for t in V.CWD12_RELATED_TOKENS)])

    print("the pre-v3.60.0 join")
    sp = C.CLASS_NAMES.index
    cwid15 = sorted(list(S.CWD12_LEGACY_LABELS) + ["SpurredAnoda", "Swinecress", "Waterhemp"])
    m, d = V.old_join("rf_zig-zag", cwid15)
    got = {n: m[i] for i, n in enumerate(cwid15)}
    check("zig-zag (CottonWeedID15 names): Crabgrass -> MorningGlory, Nutsedge -> Ragweed, "
          "Purslane -> PalmerAmaranth, Ragweed -> Sicklepod; Waterhemp and the rest deleted",
          got["Crabgrass"] == sp("MorningGlory") and got["Nutsedge"] == sp("Ragweed")
          and got["Purslane"] == sp("PalmerAmaranth") and got["Ragweed"] == sp("Sicklepod")
          and got["Waterhemp"] is None and got["SpurredAnoda"] is None and d is None, got)
    m, d = V.old_join("x_three", ["Purslane", "Ragweed", "Waterhemp", "corn"])
    check("fewer than four legacy names: they join by label, the others become OtherPlant",
          m == {0: sp("PalmerAmaranth"), 1: sp("Sicklepod"), 2: 12, 3: 12} and d is None, m)
    m, d = V.old_join("cottonweed_holdout", list(S.CWD12_SPECIES))
    check("cottonweed_holdout through its old four-name list: ids 0-3 wrong, 4-11 deleted",
          [m.get(i, d) for i in range(12)] == [sp("Purslane"), sp("SpottedSpurge"), sp("Carpetweed"),
                                               sp("Ragweed")] + [None] * 8, m)
    m, d = V.old_join("cottonweed_sp8", list(S.CWD12_ID_SPACE["cottonweed_sp8"]))
    check("cottonweed_sp8 through its old legacy list: local id i -> slot i, right",
          [m.get(i, d) for i in range(8)] == [C.CLASS_NAMES.index(n) for n in S.TRAINER_SLOT_SPECIES[:8]], m)
    m, d = V.old_join("x_none", [])
    check("no names: every box OtherPlant", m == {} and d == 12)
    m, d = V.old_join("x_dict", {"0": "Purslane", "1": "Waterhemp"})
    check("{id: name} was read by its keys: OtherPlant", m == {0: 12, 1: 12} and d is None, m)

    print("join version")
    try:
        V._check_join_version()
        ok = True
    except V.VerifyError:
        ok = False
    check("the current mega_trainer passes the join check", ok)
    real = M._build_canonical_class_map
    M._build_canonical_class_map = legacy_class_map
    try:
        V._check_join_version()
        refused = False
    except V.VerifyError as e:
        refused = "not the v3.60.0 species join" in str(e)
    finally:
        M._build_canonical_class_map = real
    check("a pre-v3.60.0 (legacy-label) mega_trainer join is refused", refused)


def test_thresholds():
    print("joint thresholds")
    rng = np.random.default_rng(3)
    d = 48
    centres = rng.normal(0, 1, (13, d))
    Xc, yc, gc = [], [], []
    for s in range(12):
        for k in range(12):
            n = int(rng.integers(8, 16))
            Xc.append(centres[k] + rng.normal(0, 1.0, (n, d)))
            yc += [k] * n
            gc += ["S%d" % s] * n
    Xc, yc, gc = V._norm(np.concatenate(Xc)), np.array(yc), np.array(gc)
    Xo = V._norm(centres[12] + rng.normal(0, 1.0, (400, d)))
    go = np.array(["slug%d" % (i % 6) for i in range(400)])
    ver = V.fit_verifier(Xc, yc, gc, Xo, go, seed=0)
    pc = ver.info["per_class"]
    top1 = [pc[n]["cv_top1"] for n in C.CLASS_NAMES[:12]]
    rec = [pc[n]["cv_confirmed"] for n in C.CLASS_NAMES[:12]]
    check("fixture: the probe ranks (almost) every held-out species crop first", min(top1) >= 0.97, top1)
    check("every species is verified on >= 95 %% of its held-out crops (got %.3f-%.3f)"
          % (min(rec), max(rec)), min(rec) >= 0.95, rec)
    check("and the thresholds are not looser than they need be (<= 0.975)", max(rec) <= 0.975, rec)
    check("the achieved recall is reported", ver.info["cv_recall_species_min"] == min(rec)
          and all(pc[n]["thresholds"]["target_reachable"] for n in C.CLASS_NAMES[:12]))
    check("out-of-fold OtherPlant predictions are kept for admit",
          ver.oof_other["P"].shape == (400, 13) and ver.oof_other["usable"].all())
    # one class: every crop at p = 1; cosines spread; 90 % ranked first
    p = np.ones(200)
    cos = np.linspace(0.5, 0.99, 200)
    top = np.arange(200) >= 20
    tau, sig, info = V.joint_thresholds(p, cos, top, 0)
    check("a class ranked first only 90 % of the time is not loosened past 99 % of those "
          "(recall 179/200, flagged)", info["target_reachable"] is False
          and info["cond_pass_target"] == 0.99 and info["recall_cv"] == 0.895
          and sig == cos[21] and tau == 1.0, info)
    tau, sig, info = V.joint_thresholds(p[:3], cos[:3], top[:3] | True, 0)
    check("fewer than MIN_CAL crops: never confident", tau == np.inf and sig == np.inf)


def test_verdicts():
    print("verdicts")
    tau = np.full(13, 0.5)
    sig = np.full(13, 0.8)
    tau[3] = 0.95
    rows = [  # label, argmax, p, cosine at argmax, expected
        (0, 0, 0.9, 0.9, V.VERIFIED),
        (0, 0, 0.4, 0.9, V.UNKNOWN),          # probability under tau
        (0, 0, 0.9, 0.7, V.UNKNOWN),          # cosine under sigma
        (0, 5, 0.9, 0.9, V.CONFLICT),
        (0, 5, 0.9, 0.5, V.UNKNOWN),
        (0, 3, 0.9, 0.9, V.UNKNOWN),          # tau of the PREDICTED class (0.95) applies
        (0, 12, 0.9, 0.1, V.CONFLICT),        # OtherPlant: probability test only
        (0, 12, 0.4, 0.9, V.UNKNOWN),
        (12, 12, 0.9, 0.9, V.OTHER_OK),
        (12, 5, 0.9, 0.9, V.CONFLICT),
        (12, 5, 0.9, 0.5, V.OTHER_OK),
        (12, 5, 0.3, 0.9, V.OTHER_OK),
        (7, 7, 0.5, 0.8, V.VERIFIED),         # thresholds are inclusive
    ]
    P = np.full((len(rows) + 1, 13), 0.0)
    cos = np.full((len(rows) + 1, 13), 0.0)
    for i, (_l, j, p, c, _e) in enumerate(rows):
        P[i] = (1 - p) / 12
        P[i, j] = p
        cos[i, j] = c
    P[-1] = np.nan
    labels = [r[0] for r in rows] + [0]
    got, j, pj, cj = V.verdicts(labels, P, cos, tau, sig)
    want = [r[4] for r in rows] + [V.FAILED]
    for i, (g, w) in enumerate(zip(got, want)):
        check("verdict row %d %s" % (i, rows[i][:4] if i < len(rows) else "NaN probabilities"),
              g == w, "got %s want %s" % (g, w))
    check("argmax and its p / cosine are returned", j[3] == 5 and abs(pj[3] - 0.9) < 1e-12
          and abs(cj[3] - 0.9) < 1e-12)
    tau_inf = tau.copy()
    tau_inf[0] = np.inf
    check("an uncalibrated class (tau inf) is never verified",
          V.verdicts([0], P[:1], cos[:1], tau_inf, sig)[0][0] == V.UNKNOWN)

    print("image rule")
    for labels, vs, want in (([0, 12], [V.VERIFIED, V.OTHER_OK], V.ADMITTED),
                             ([0, 12], [V.VERIFIED, V.SMALL], V.ADMITTED),
                             ([12], [V.OTHER_OK], V.ADMITTED),
                             ([0, 1], [V.VERIFIED, V.SMALL], V.UNKNOWN),
                             ([0, 12], [V.UNKNOWN, V.OTHER_OK], V.UNKNOWN),
                             ([0], [V.FAILED], V.UNKNOWN),
                             ([12], [V.FAILED], V.UNKNOWN),
                             ([0, 12], [V.VERIFIED, V.CONFLICT], V.CONFLICT),
                             ([0, 1], [V.CONFLICT, V.UNKNOWN], V.CONFLICT)):
        check("image %s %s -> %s" % (labels, vs, want), V.image_verdict(labels, vs) == want,
              V.image_verdict(labels, vs))


# ------------------------------------------------------------ pipeline
def test_pipeline():
    rows, boxes_of = make_splits()
    exp = make_pool(rows, boxes_of)
    check("fixture: the dev re-encode is within 6 bits", exp["evalcopy_bits"] <= 6, exp["evalcopy_bits"])
    check("fixture: the train_core copies are within 3 bits",
          all(b <= 3 for b in exp["copy_bits"].values()), exp["copy_bits"])
    check("fixture: the shifted copy is 4-5 bits from its original", exp["shift_bits"] in (4, 5))
    a_src_before = {p.name: p.read_bytes() for p in (DS / "a_species" / "labels").iterdir()}

    print("pool")
    real = M._build_canonical_class_map
    M._build_canonical_class_map = legacy_class_map
    try:
        V.cmd_pool(args("pool", "--procs", "1"))
        refused = False
    except V.VerifyError as e:
        refused = "not the v3.60.0 species join" in str(e)
    finally:
        M._build_canonical_class_map = real
    check("pool refuses to run through a pre-v3.60.0 mega_trainer join", refused and not V.POOL.exists())
    lock = {"manifests": {"train_core": C.sha256_file(C.manifest_path("train_core"))},
            "nevertrain_sha256": "0" * 64}
    C.LOCK_PATH.write_text(json.dumps(lock))
    try:
        V.cmd_pool(args("pool", "--procs", "1"))
        refused = False
    except V.VerifyError as e:
        refused = "never-train index" in str(e) and "locked" in str(e)
    check("a never-train index that differs from LOCK.json is refused", refused)
    lock["nevertrain_sha256"] = C.sha256_file(C.NEVER_TRAIN_INDEX)
    C.LOCK_PATH.write_text(json.dumps(lock))
    summ = V.cmd_pool(args("pool", "--procs", "2"))
    check("the locked never-train index is checked and recorded",
          summ["lock"]["nevertrain_locked"] is True and summ["never_train_index"]["matches_lock"] is True
          and summ["lock"]["locked"] is True, summ["lock"])
    sk = summ["skipped"]
    for reason, slug in (("never_train", "cottonweeddet12"),
                         ("user_flag_garbage", "d_flagged"), ("autolabel", "e_autolabel"),
                         ("missing_dir", "f_missing"), ("unknown_layout", "g_weird"),
                         ("annotation_not_bbox", "h_classification")):
        check("slug %s skipped as %s" % (slug, reason), sk.get(reason) == [slug], sk)
    ps = summ["per_slug"]
    check("the leave-4-out copies are read for calibration copies only",
          summ["calibration_only_slugs"] == ["cottonweed_holdout", "cottonweed_sp8"]
          and ps["cottonweed_holdout"]["kept"] == 0 and ps["cottonweed_sp8"]["kept"] == 0
          and ps["cottonweed_holdout"]["dropped"] == {"calibration_only_not_train_core": 1, "cwd12_copy": 2,
                                                      "near_eval": 1}
          and ps["cottonweed_sp8"]["dropped"] == {"calibration_only_not_train_core": 1, "cwd12_copy": 2},
          (ps["cottonweed_holdout"]["dropped"], ps["cottonweed_sp8"]["dropped"]))
    check("a_species drops: near_eval, 3 copies, no_label, no_boxes, unhashable, unmapped",
          ps["a_species"]["dropped"] == {"cwd12_copy": 3, "near_eval": 1, "no_boxes": 1, "no_label": 1,
                                         "unhashable": 1, "unmapped_class": 1}, ps["a_species"]["dropped"])
    check("the eval near-copy is attributed to dev", ps["a_species"]["near_eval_by_split"] == {"dev": 1},
          ps["a_species"]["near_eval_by_split"])
    check("b_other: the byte copy of an a_species image is an exact duplicate",
          ps["b_other"]["dropped"] == {"exact_dup": 1} and ps["b_other"]["layout"] == ["train", "valid"],
          ps["b_other"])
    check("c_nonames: its train_core copies are kept apart, the 4-5 bit one included",
          ps["c_nonames"]["dropped"] == {"cwd12_copy": 2}, ps["c_nonames"]["dropped"])
    check("the join is recorded per slug",
          ps["a_species"]["join"]["2"] == ["Carpetweeds", "Carpetweed"]
          and ps["c_nonames"]["join"] == {"*": [None, "OtherPlant"]}, ps["a_species"]["join"])
    check("the old join is recorded per slug",
          ps["a_species"]["old_join"]["0"] == "deleted" and ps["a_species"]["old_join"]["2"] == "Waterhemp"
          and ps["cottonweed_holdout"]["old_join"]["0"] == "Purslane"
          and ps["cottonweed_holdout"]["old_join"]["*"] == "deleted"
          and ps["c_nonames"]["old_join"] == {"*": "OtherPlant"}, ps["a_species"]["old_join"])
    pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    want_keys = ({"a_species__a_ok_%d" % i for i in range(6)} | {"a_species__a_bad", "a_species__a_small"}
                 | {"b_other__train__b_ok_%d" % i for i in range(3)} | {"b_other__valid__b_ok_3"}
                 | {"b_other__train__b_conflict"} | {"c_nonames__c_ok_0", "c_nonames__c_ok_1"})
    check("pool.jsonl holds exactly the images that survive", set(pool) == want_keys,
          sorted(set(pool) ^ want_keys))
    copies = {r["key"]: r for r in C.read_manifest(V.COPIES)}
    want_copies = dict(exp["copies"], **{k: v[0] for k, v in exp["calib"].items()})
    check("cwd12_copies.jsonl pairs each copy with its train_core original",
          {k: r["train_core_key"] for k, r in copies.items()} == want_copies, sorted(copies))
    check("the leave-4-out copies are flagged calibration-only; the harvested ones are not",
          all(r["calibration_only"] == (k in exp["calib"]) for k, r in copies.items()))
    ho = [k for k in exp["calib"] if k.startswith("cottonweed_holdout")]
    s8 = [k for k in exp["calib"] if k.startswith("cottonweed_sp8")]
    four = [C.CLASS_NAMES.index(n) for n in ("Purslane", "SpottedSpurge", "Carpetweed", "Ragweed")]
    check("holdout copies: current labels are the truth; old labels through the four-name list",
          all([b[0] for b in C.read_yolo(copies[k]["label"])] == exp["calib"][k][1]
              and copies[k]["old_join"] == [four[t] if t < 4 else None for t in exp["calib"][k][1]]
              for k in ho), [(copies[k]["old_join"], exp["calib"][k][1]) for k in ho])
    check("sp8 copies: current and old labels are both the truth",
          all([b[0] for b in C.read_yolo(copies[k]["label"])] == exp["calib"][k][1] == copies[k]["old_join"]
              for k in s8), [(copies[k]["old_join"], exp["calib"][k][1]) for k in s8])
    check("no pool image comes from a leave-4-out copy", not any(k.startswith("cottonweed_") for k in pool))
    lb = C.read_yolo(pool["a_species__a_ok_0"]["label"])
    check("converted labels are in INC ids (species + OtherPlant for Lambsquarters)",
          [b[0] for b in lb] == [0, 1, 12], lb)
    check("wildcard boxes are OtherPlant",
          {b[0] for b in C.read_yolo(pool["c_nonames__c_ok_0"]["label"])} == {12})
    check("labels are written under step1/labels/<slug>/",
          pathlib.Path(pool["b_other__train__b_ok_0"]["label"]).parent == V.LABELS_DIR / "b_other")
    check("label files are named by their content (<key>.<sha256[:16]>.txt)",
          all(pathlib.Path(r["label"]).name == "%s.%s.txt" % (k, r["label_sha256"][:16])
              for k, r in list(pool.items()) + list(copies.items())))
    check("label_sha256 is the written file's sha256",
          all(C.sha256_file(r["label"]) == r["label_sha256"] for r in list(pool.values()) + list(copies.values())))
    check("no temporary label file is left behind",
          not [p for p in V.LABELS_DIR.rglob("*") if ".tmp" in p.name])
    check("source label files are untouched",
          {p.name: p.read_bytes() for p in (DS / "a_species" / "labels").iterdir()} == a_src_before)
    guard = C.NeverTrainGuard.load()
    check("every pool image clears the never-train guard",
          guard.check([r["image"] for r in pool.values()]) == ([], []))
    check("per-class box counts cover every kept box",
          summ["boxes"] == sum(len(C.read_yolo(r["label"])) for r in pool.values())
          == sum(summ["boxes_per_class"].values()))

    calls = []
    orig_probe = V._probe_image
    V._probe_image = lambda p: calls.append(p) or orig_probe(p)
    try:
        sha = C.sha256_file(V.POOL)
        V.cmd_pool(args("pool", "--procs", "1"))
    finally:
        V._probe_image = orig_probe
    check("a re-run hashes nothing new (cache) and writes the same pool.jsonl",
          calls == [str(DS / "a_species" / "images" / "a_corrupt.jpg")] and C.sha256_file(V.POOL) == sha,
          calls)
    check("a re-run writes no label file (same content, same name)",
          json.load(open(V.POOL_SUMMARY))["label_files_written"] == 0)

    print("crops")
    info = V.cmd_crops(args("crops"))
    n_pool_boxes = sum(len(C.read_yolo(r["label"])) for r in pool.values())
    check("crops: 120 train_core, 27 copy, pool crops; one small box left out",
          info["per_set"] == {"core": 120, "copy": 27, "pool": n_pool_boxes - 1}
          and info["dropped_small_boxes"] == {"pool": 1}, info)
    with open(V.CROPS_SKIPPED, newline="") as fh:
        left = list(csv.reader(fh))
    check("crops_skipped.csv names the small box", left == [list(V.SKIPPED_FIELDS),
                                                            ["pool", "a_species__a_small", "1", "small"]], left)
    check("crops_info binds crops.csv to its inputs",
          info["inputs"] == {"pool_sha256": C.sha256_file(V.POOL), "pool_meta_sha256": C.sha256_file(V.POOL_META),
                             "copies_sha256": C.sha256_file(V.COPIES),
                             "train_core_sha256": C.sha256_file(C.manifest_path("train_core"))}, info["inputs"])
    crops = V.Crops()
    core = crops.where("core")
    check("core crops carry cwd12 ids and sessions as groups",
          crops.label[core].max() < 12 and crops.group[core[0]].startswith("20210701_Cam_S"))
    pool_c = crops.where("pool")
    check("pool crops are grouped by slug and keep the source name",
          crops.group[pool_c[0]] == crops.source[pool_c[0]]
          and "Lambsquarters" in {crops.src_name[i] for i in pool_c})

    print("embed")
    # an image that fails after the crops were made: its crops must be NaN rows
    broken = pool["b_other__train__b_ok_1"]["image"]
    saved = pathlib.Path(broken).read_bytes()
    pathlib.Path(broken).write_bytes(b"broken")
    killer = FakeEmbedder(kill_after=4)
    try:
        V.cmd_embed(args("embed", "--shard", "0", "--nshards", "2", "--procs", "1",
                         "--chunk-images", "3", "--batch", "4"), embedder=killer)
        killed = False
    except Kill:
        killed = True
    parts = sorted(p.name for p in V.EMB_DIR.iterdir() if ".part" in p.name)
    check("a job killed mid-shard leaves its finished chunks", killed and parts == [
        "emb_s000_of_002.part00000.npz"], parts)
    fe = FakeEmbedder()
    V.cmd_embed(args("embed", "--shard", "0", "--nshards", "2", "--procs", "1",
                     "--chunk-images", "3", "--batch", "4"), embedder=fe)
    images = crops.images()
    shard0 = sum(len(cs) for _img, cs in images[:len(images) // 2])
    check("the re-run embeds only what is missing", fe.crops == shard0 - 9, (fe.crops, shard0))
    check("finished shard: chunks folded into one file",
          sorted(p.name for p in V.EMB_DIR.iterdir()) == ["emb_s000_of_002.npz"])
    fe2 = FakeEmbedder()
    V.cmd_embed(args("embed", "--shard", "0", "--nshards", "2", "--chunk-images", "3", "--procs", "1"),
                embedder=fe2)
    check("a finished shard is skipped", fe2.calls == 0)
    try:
        V.load_embeddings(crops)
        refused = False
    except V.VerifyError as e:
        refused = "missing shards [1]" in str(e)
    check("an incomplete shard set is refused", refused)
    V.cmd_embed(args("embed", "--shard", "1", "--nshards", "2", "--procs", "2", "--batch", "5"),
                embedder=FakeEmbedder())
    pathlib.Path(broken).write_bytes(saved)
    X, emb = V.load_embeddings(crops)
    nan_rows = np.flatnonzero(~np.isfinite(X).all(1))
    check("the unreadable image's crops are the only NaN rows",
          sorted(crops.image[i] for i in nan_rows) == [broken] * 3 and emb["failed_crops"] == 3,
          (nan_rows, emb))
    ok = True
    nan_set = set(nan_rows.tolist())
    for i in range(crops.n):
        if i in nan_set:
            continue
        with Image.open(crops.image[i]) as im:
            im = im.convert("RGB")
            r = crops.row(i)
            f = fake_feature(_cut(im, dict(r, W=im.size[0], H=im.size[1])))
        if not np.allclose(X[i].astype(np.float32), f, atol=2e-3):
            ok = False
            print("    row %d misaligned" % i)
            break
    check("every row holds its own crop's features (no shift after the failure)", ok)

    print("fit")
    fi = V.cmd_fit(args("fit"))
    th = json.load(open(V.VERIFIER_DIR / "thresholds.json"))
    check("every species and OtherPlant has finite thresholds",
          all(th["tau_p"][n] is not None and th["sigma"][n] is not None for n in C.CLASS_NAMES), th)
    os_ = fi["other_sample"]
    check("OtherPlant sample: named non-cwd12 plants only (no 'weed', no nameless boxes, no 'pigweed')",
          set(os_["eligible_names"]) == {"Lambsquarters", "Crabgrass", "corn", "Giant ragweed"}
          and os_["excluded_names_by_reason"].get("cwd12_related") == {"pigweed": 1}
          and "weed" in os_["excluded_names_by_reason"]["generic"]
          and "(no name)" in os_["excluded_names_by_reason"]["no_name"], os_)
    check("cv top-1 on species is high with class-clustered features",
          fi["cv_top1_species"] >= 0.95, fi["cv_top1_species"])
    check("the verifier records the inputs and the shard set it was fitted on",
          fi["inputs"] == json.load(open(V.CROPS_INFO))["inputs"] and fi["embeddings"]["nshards"] == 2,
          (fi["inputs"], fi["embeddings"]))

    print("calibrate")
    cal = V.cmd_calibrate(args("calibrate"))
    a = cal["cwd12_copies"]
    check("(a) every copy box is matched to its original", a["matched_boxes"] == 27
          and a["unmatched_boxes"] == {} and a["matched_without_fold"] == 0, a["unmatched_boxes"])
    cj = a["current_join"]
    check("(a) current join: 20 of 27 right (planted box + nameless copies wrong; leave-4-out copies right)",
          cj["overall"]["label_correct_rate"] == round(20 / 27, 4)
          and cj["per_slug"]["a_species"]["label_correct_rate"] == round(8 / 9, 4)
          and cj["per_slug"]["c_nonames"]["label_correct_rate"] == 0.0
          and cj["per_slug"]["cottonweed_holdout"]["label_correct_rate"] == 1.0
          and cj["per_slug"]["cottonweed_sp8"]["label_correct_rate"] == 1.0, cj["per_slug"])
    # a wrong label is a conflict only where the probe is confident on the true
    # class (95 % of true crops by construction), so a miss can happen; a right
    # label is never a conflict and a wrong one is never verified
    check("(a) current join: wrong labels are conflicts (>= 5 of 7), no right one is; verified boxes are right",
          cj["overall"]["wrong_judged"] == 7 and cj["overall"]["conflict_recall_on_wrong"] >= 0.7
          and cj["overall"]["false_conflict_rate_on_correct"] == 0.0
          and cj["overall"]["verified_precision"] == 1.0, cj["overall"])
    ow = a["old_wrong_joins"]
    check("(a) old joins: right / wrong / deleted boxes per slug are known",
          ow["right_wrong_deleted"] == {"a_species": [0, 7, 2], "c_nonames": [0, 6, 0],
                                        "cottonweed_holdout": [0, 5, 1], "cottonweed_sp8": [6, 0, 0]},
          ow["right_wrong_deleted"])
    check("(a) old joins: wrong joins are conflicts (>= 15 of 18), no right one is; verified boxes are right",
          ow["overall"]["wrong_judged"] == 18 and ow["overall"]["conflict_recall_on_wrong"] >= 0.8
          and ow["overall"]["false_conflict_rate_on_correct"] == 0.0
          and ow["overall"]["verified_precision"] == 1.0, ow["overall"])
    check("(a) the optimistic final-model numbers are reported separately",
          "overall" in cj["final_model_optimistic"] and "overall" in ow["final_model_optimistic"])
    b = cal["swaps"]
    check("(b) 20% swaps per held-out fold", len(b["folds"]) == 5
          and all(f["swapped"] == round(0.2 * f["boxes"]) for f in b["folds"]), b["folds"])
    check("(b) the swaps are spread over the species (>= 9 true and >= 9 new species of 12)",
          len(b["swapped_true_species"]) >= 9 and len(b["swapped_to_species"]) >= 9,
          (b["swapped_true_species"], b["swapped_to_species"]))
    sw_test = V._swap_labels(np.arange(12) % 12, np.arange(12), 0)
    check("(b) a swapped label is never the true one", (sw_test[1] != sw_test[0] % 12).all())
    # a swapped crop is a conflict only when the probe is confident on the true class;
    # over 25 swaps the spread is ~0.06, so the bound leaves room for other builds
    check("(b) swaps are caught: conflict recall >= 0.6, false conflict <= 0.05, verified precision >= 0.95",
          b["overall"]["conflict_recall_on_wrong"] >= 0.6
          and b["overall"]["false_conflict_rate_on_correct"] <= 0.05
          and b["overall"]["verified_precision"] >= 0.95, b["overall"])
    cal2 = V.cmd_calibrate(args("calibrate"))
    check("(b) the swaps are the same on every run",
          [f["swap_sha256"] for f in cal2["swaps"]["folds"]] == [f["swap_sha256"] for f in b["folds"]]
          and cal2["swaps"]["overall"] == b["overall"])

    print("admit")
    summ = V.cmd_admit(args("admit"))
    admitted = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    want = ({"a_species__a_ok_%d" % i for i in range(6)} | {"b_other__train__b_ok_0", "b_other__train__b_ok_2",
                                                              "b_other__valid__b_ok_3"}
            | {"c_nonames__c_ok_0", "c_nonames__c_ok_1"})
    check("verified.jsonl: clean images admitted; conflict, small-box and unreadable images not",
          admitted == want, sorted(admitted ^ want))
    with open(V.CONFLICTS, newline="") as fh:
        conf = list(csv.DictReader(fh))
    got = {(r["key"], r["label_name"], r["pred_name"]) for r in conf}
    check("conflicts.csv: Waterhemp-labelled Purslane and weed-labelled Sicklepod",
          got == {("a_species__a_bad", "Waterhemp", "Purslane"),
                  ("b_other__train__b_conflict", "OtherPlant", "Sicklepod")}, got)
    check("conflict rows carry p, cosine and a sheet index",
          all(float(r["p"]) > 0 and r["cosine"] != "" and r["sheet_index"] != "" for r in conf))
    with Image.open(V.SHEET) as im:
        check("the crop sheet holds the conflicts", im.size == (1280, 158), im.size)
    iv = summ["images"]
    check("image verdict counts", iv == {"admitted": 11, "conflict": 2, "unknown": 2}, iv)
    check("the small-box image is counted as blocked by its small box",
          summ["images_unknown_only_for_small_species_boxes"] == 1 and summ["boxes_small_not_embedded"] == 1)
    check("per-slug and per-species summaries",
          summ["per_slug"]["a_species"]["images"]["admitted"] == 6
          and summ["per_species"]["Purslane"]["boxes"].get("verified", 0) >= 1, summ["per_slug"])
    pv = np.load(V.POOL_VERDICTS)
    codes = list(V.VERDICT_CODES)
    by_box = {(crops.key[c], int(crops.box[c])): codes[v] for c, v in zip(pv["crop_id"], pv["verdict"])}
    meta = {m["key"]: m for m in C.read_manifest(V.POOL_META)}
    rule = {k for k, m in meta.items()
            if V.image_verdict([b[0] for b in m["boxes"]],
                               [by_box.get((k, i), V.SMALL) for i in range(len(m["boxes"]))]) == V.ADMITTED}
    check("verified.jsonl is exactly what the rule admits from the per-box verdicts", rule == admitted)
    ver = V.Verifier.load()
    oid = ver.arrays["other_ids"]
    vo = V.verdicts(np.full(len(oid), 12), ver.arrays["other_oof_P"], ver.arrays["other_oof_cos"],
                    ver.tau_p, ver.sigma)[0]
    got_o = [codes[int(x)] for x in pv["verdict"][np.searchsorted(pv["crop_id"], oid)]]
    check("the OtherPlant training crops are judged by the out-of-fold models",
          summ["otherplant_training_boxes_judged_out_of_fold"] == len(oid) == 15
          and got_o == list(vo), (got_o, list(vo)))
    # plant an out-of-fold call of Sicklepod on one of them: admit must follow it,
    # although the final probe (which trained on that crop) says OtherPlant
    npz = V.VERIFIER_DIR / "verifier.npz"
    saved_npz = npz.read_bytes()
    d = dict(np.load(npz))
    meta_s = d.pop("meta")
    d["other_oof_P"] = d["other_oof_P"].copy()
    d["other_oof_cos"] = d["other_oof_cos"].copy()
    d["other_oof_P"][0] = 0.0
    d["other_oof_P"][0, 9] = 1.0
    d["other_oof_cos"][0, 9] = 1.0
    np.savez(npz, meta=meta_s, **d)
    final_says = V.Verifier.load().judge_features(np.array([12]), crops_x(oid[:1]))[0][0]
    try:
        V.cmd_admit(args("admit"))
        k0 = crops.key[int(oid[0])]
        with open(V.CONFLICTS, newline="") as fh:
            conf2 = {(r["key"], r["pred_name"]) for r in csv.DictReader(fh)}
        adm2 = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
        check("an out-of-fold conflict on a training crop blocks its image (the final probe says %s)"
              % final_says, final_says == V.OTHER_OK and (k0, "Sicklepod") in conf2 and k0 not in adm2,
              (k0, conf2))
    finally:
        npz.write_bytes(saved_npz)
        V.cmd_admit(args("admit"))

    print("cli")
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.verify", "fit",
                          "--nshards", "3"], cwd=str(ROOT), capture_output=True, text=True, env=env)
    check("the CLI fails cleanly without a complete shard set",
          out.returncode == 2 and "no complete, current set" in out.stdout, out.stdout[-300:] + out.stderr[-300:])
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.verify", "embed",
                          "--shard", "2", "--nshards", "2"], cwd=str(ROOT), capture_output=True, text=True, env=env)
    check("the CLI refuses a shard outside --nshards", out.returncode == 2 and "--shard" in out.stderr)

    print("all")
    res = V.cmd_all(args("all", "--procs", "1", "--nshards", "1"), embedder=FakeEmbedder())
    check("`all` runs every stage end to end", res["images"].get("admitted", 0) >= 10
          and (V.EMB_DIR / "emb_s000_of_001.npz").exists(), res["images"])
    for cmd in ("admit", "calibrate"):
        try:
            V.COMMANDS[cmd](args(cmd, "--nshards", "2"))
            refused = False
        except V.VerifyError as e:
            refused = "not the ones the verifier was fitted on" in str(e) and "nshards" in str(e)
        check("%s refuses another shard set than the verifier's (nshards 2 vs 1)" % cmd, refused)
    test_stale(exp)


def crops_x(ids):
    """Features of crop ids from the current shard set."""
    X, _info = V.load_embeddings(V.Crops())
    return X[np.asarray(ids, dtype=np.int64)]


def test_stale(exp):
    print("stale inputs")
    first = C.read_manifest(V.VERIFIED_MANIFEST)
    # a new slug: Sicklepod plants labelled "corn"; and a_species' class list
    # changes (Waterhemp <-> Morningglory), after the verifier was fitted
    z = DS / "z_new"
    for i in range(3):
        bx = layout3([9, 9, 9])
        paint(z / "images" / ("z_%d.png" % i), bx, texture=False, fmt="png")
        yolo(z / "labels" / ("z_%d.txt" % i), [(0,) + b[1:] for b in bx])
    regp = REPO / "results" / "framework" / "dataset_registry.json"
    reg = json.loads(regp.read_text())
    names = reg["datasets"]["a_species"]["class_names"]
    names[0], names[1] = names[1], names[0]
    reg["datasets"]["z_new"] = {"local_path": str(z), "annotation": "bbox", "class_names": ["corn"]}
    regp.write_text(json.dumps(reg))
    V.cmd_pool(args("pool", "--procs", "1"))
    check("the labels an earlier verified.jsonl names are unchanged by a new pool",
          all(C.sha256_file(r["label"]) == r["label_sha256"] for r in first))
    new_pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    old0 = {r["key"]: r for r in first}["a_species__a_ok_0"]
    check("a relabelled image gets a new label file; the old one stays",
          new_pool["a_species__a_ok_0"]["label"] != old0["label"]
          and [b[0] for b in C.read_yolo(new_pool["a_species__a_ok_0"]["label"])] == [1, 0, 12]
          and [b[0] for b in C.read_yolo(old0["label"])] == [0, 1, 12])
    for name, fn in (("embed", lambda: V.cmd_embed(args("embed", "--nshards", "1", "--procs", "1"),
                                                   embedder=FakeEmbedder())),
                     ("fit", lambda: V.cmd_fit(args("fit"))),
                     ("calibrate", lambda: V.cmd_calibrate(args("calibrate"))),
                     ("admit", lambda: V.cmd_admit(args("admit")))):
        try:
            fn()
            refused = False
        except V.VerifyError as e:
            refused = "pool" in str(e) and "changed after `verify crops`" in str(e)
        check("after `pool` alone, %s refuses the stale crops.csv" % name, refused)
    check("admit left verified.jsonl as it was", [r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)]
          == [r["key"] for r in first])
    # a crops.csv that lost a pool box row is refused by admit too
    V.cmd_crops(args("crops"))
    V.cmd_embed(args("embed", "--nshards", "1", "--procs", "1"), embedder=FakeEmbedder())
    V.cmd_fit(args("fit"))
    summ = V.cmd_admit(args("admit"))
    adm = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    check("the full chain on the new pool: the mislabelled new slug and the relabelled images are "
          "conflicts, not admitted",
          not any(k.startswith("z_new") for k in adm) and "a_species__a_ok_0" not in adm
          and summ["per_slug"]["z_new"]["images"] == {"conflict": 3}, (sorted(adm), summ["per_slug"].get("z_new")))
    rows = list(csv.reader(open(V.CROPS_SKIPPED, newline="")))
    with open(V.CROPS_SKIPPED, "w", newline="") as fh:
        csv.writer(fh).writerows(rows[:1])
    info = json.load(open(V.CROPS_INFO))
    info["skipped_sha256"] = C.sha256_file(V.CROPS_SKIPPED)
    V._write_json(V.CROPS_INFO, info)
    try:
        V.cmd_admit(args("admit"))
        refused = False
    except V.VerifyError as e:
        refused = "no crop row" in str(e)
    check("a pool box with neither a crop row nor a skipped row is refused, not taken as small", refused)


def main():
    try:
        test_join()
        test_thresholds()
        test_verdicts()
        test_pipeline()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
