#!/usr/bin/env python3
"""inc2/step1_stream.py: incremental Step 1 with per-box admission
(docs/CONTINUOUS_LOOP.md §3.3, §3.9, §6.7, §8; group C of §9).

The world is test_inc_verify's synthetic world (its fixture functions, a
fake REPO and INC_DIR under a temp dir, a fake embedder that reads the colour
painted in each box), extended with: an ImageWeeds split, a 3SeasonWeedDet10-
like expert set "tsw22" (in the v1 never-train index as ood22, as in v1) and a
harvested copy of it, a slug of vetoed images (a verified target box beside a
conflict, and one beside an overlapping conflict), and a pair of near
duplicates in two slugs. v1 Step 1 (verify pool, crops, embed, fit, admit) and
select build run on it; splits v2 is written in inc2.guard's format, and the
real GuardV2 is used when inc2.guard (group A) loads it, else a double of the
contract's interface (the test says which).

Pinned (acceptance of group C):
  * equivalence: from empty, with the image rule, the stream's pool keys,
    label sha256s, box verdicts and admitted set equal verify pool + crops +
    admit;
  * split invariance: two batches give the same effective queue as one, ids
    (batch, group numbers, paths) aside, and the same near-duplicate groups;
  * guards: a byte copy of dev (near_eval_v2), a flipped test image and a
    rotated dev image (near_eval_variant), a copy of a base image (base_copy),
    a copy of an expert base image (known truth), an exact duplicate with the
    same labels (exact_dup) and with other labels (held_join_conflict: both
    held, a rejoin queued), cropped / sheared / brightened copies of dev and
    test more than 6 bits away under all 8 variants (near_eval_embed), and
    the rotation variant of a masked PNG made from an EXIF-rotated copy of a
    dev image; a registry slug absent from the v1 pool and one whose join
    changed are refused; a v1 slug with evaluation near-copies yields rows
    held h6_scan, and rows without a licence are held licence;
  * D28-v2 (amendment 2026-10-03): each dHash copy of an evaluation image
    carries its pair cosine with the image GuardV2 matched (the copy
    scanner's descriptors; the byte copy 1.0), batch.json records them
    (eval_hits), status.json folds them per source and the autopilot's D28
    reads the fold; without a scanner index, or for a match that names no
    evaluation image, a hit is left unweighed with the reason. A batch whose
    batch.json does not weigh its hits (made before the amendment) is
    weighed again by eval-hits into step1_stream/eval_hits/<batch>.json
    (b0000 from admission.jsonl, the v1 pool and dHash cache through
    GuardV2), batch.json untouched; status.json lists it as due until then,
    folds a sidecar made from the committed batch.json that weighed exactly
    its hits, and refuses any other (a sidecar records the sha256 of the
    batch.json bytes read; a batch.json changed since commit gets none);
    a hit is weighed against every evaluation image within the radius, not
    only the guard's match; the CLI exits 2 when a batch gets no sidecar,
    and eval-hits --intake X alone weighs no Step 1 batch; run_inc2_stream.sh
    accepts eval-hits and hashes the collector package for it;
  * pins: an altered verifier.npz, another embedder, another v2 index and a
    canary drift each refuse before anything is written;
  * state: global crop ids are contiguous and disjoint, a committed batch is
    a no-op when rerun, a killed batch resumes and reuses its finished chunks,
    the ledger chain verifies and a broken one is refused;
  * b0000: every candidate is accounted for (whole + masked + refused_overlap
    + not_admitted), the counts equal an independently computed census, the
    recovered boxes per species are <= the census's; a census that does not
    match refuses and commits nothing; masked rows hold funnel_F9, base B's
    images are never queued, realloop draws are tagged prior;
  * holds: licence, funnel_F9 (domain dev refused for good, the rest
    released) and h6_scan (the funnel's calibration serves same-lab rows at
    once, the stream's own only after the deadline) are served; intake rows
    (INC ids with unmapped 13) are masked per box and provenance-cleared
    sources are not held; a person's licence override resolves an intake
    row (research_only unless the record says false and its text names a
    known licence that does not restrict use), and an override of any other
    type or without a licence text refuses; rejoin relabels from a recorded resolution and
    supersedes the old rows; the status schema; the CLI; the job script.

No network, no GPU. Run:  python3 tests/test_inc2_step1_stream.py
"""
import collections
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
os.environ["INC_VERIFY_TEST_TMP"] = tempfile.mkdtemp(prefix="inc2_stream_test_")
sys.path.insert(0, str(HERE))
import test_inc_verify as TV  # noqa: E402  (sets REPO and INC_DIR under the temp dir)

import numpy as np  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import select as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.funnel import leak as L  # noqa: E402
from weed_optimizer_framework.tools.inc2 import mask as MK  # noqa: E402
from weed_optimizer_framework.tools.inc2 import step1_stream as SS  # noqa: E402
from weed_optimizer_framework.tools.near_dup import NearHashIndex  # noqa: E402

TMP = TV.TMP
ROOT = HERE.parent
REPO, DS = TV.REPO, TV.DS
FAILURES = []
NOTES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:600]))
        FAILURES.append(name)


def raises(fn, exc=SS.StreamError, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e) or ("expected %r in %r" % (text, str(e)))
    return False


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


# ---------------------------------------------------------------- the world
TSW = []                     # expert rows of the 3SeasonWeedDet10-like set


def make_world():
    rows, boxes_of = TV.make_splits()
    exp = TV.make_pool(rows, boxes_of)
    a_src = TV.A_SRC
    # ImageWeeds (an evaluation split of v2)
    iw = []
    for i in range(2):
        img = REPO / "downloads" / "imageweeds" / ("iw_%d.jpg" % i)
        lbl = img.with_suffix(".txt")
        bx = TV.layout3([12, 12, 12])
        TV.paint(img, bx, texture=True, fmt="jpg")
        TV.yolo(lbl, bx)
        iw.append(TV.manifest_row("imageweeds", "iw_%d" % i, img, lbl, "", "imageweeds"))
    C.write_manifest(C.manifest_path("imageweeds"), iw)
    rows["imageweeds"] = iw
    # tsw22: expert-labelled images (base in v2; an ood exam in v1)
    for i in range(4):
        img = REPO / "downloads" / "tsw" / "data2022" / ("tsw_%d.jpg" % i)
        lbl = img.with_suffix(".txt")
        bx = TV.layout3([(3 * i + k) % 12 for k in range(3)])
        TV.paint(img, bx, texture=True, fmt="jpg")
        TV.yolo(lbl, bx)
        r = TV.manifest_row("tsw22", "tsw_%d" % i, img, lbl, "S%d" % i, "3seasonweeddet10/data2022")
        r["key"] = "tsw22__tsw_%d" % i
        TSW.append(r)
    # a harvested copy of two tsw images (one with a wrong first box)
    t = DS / "t_tswcopy"
    _mk(t / "images")
    for i, wrong in ((0, False), (1, True)):
        TV.clean_copy(TSW[i]["image"], t / "images" / ("tcopy_%d.jpg" % i))
        truth = C.read_yolo(TSW[i]["label"])
        TV.yolo(t / "labels" / ("tcopy_%d.txt" % i),
                [(a_src[((b[0] + 4) % 12) if (wrong and k == 0) else b[0]],) + tuple(b[1:])
                 for k, b in enumerate(truth)])
    # vetoed images: a verified target beside a conflict; one beside an overlapping conflict
    k = DS / "k_mixed"
    bx = TV.layout3([0, 8, 12])
    TV.paint(k / "images" / "k_veto1.png", bx, texture=False, fmt="png")
    TV.yolo(k / "labels" / "k_veto1.txt", [(a_src[0],) + bx[0][1:], (TV.LAMBS,) + bx[1][1:], (TV.LAMBS,) + bx[2][1:]])
    bx = TV.layout3([8, 2, 12])
    TV.paint(k / "images" / "k_veto2.png", bx, texture=False, fmt="png")
    TV.yolo(k / "labels" / "k_veto2.txt", [(a_src[8],) + bx[0][1:], (a_src[0],) + bx[1][1:], (TV.LAMBS,) + bx[2][1:]])
    bx = [(10, 0.3, 0.5, 0.22, 0.32), (10, 0.33, 0.52, 0.22, 0.32), (12, 0.8, 0.5, 0.15, 0.2)]
    TV.paint(k / "images" / "k_overlap.png", bx, texture=False, fmt="png")
    TV.yolo(k / "labels" / "k_overlap.txt", [(a_src[10],) + bx[0][1:], (TV.LAMBS,) + bx[1][1:],
                                            (TV.LAMBS,) + bx[2][1:]])
    # a near-duplicate pair split over two slugs (3 bits or less, not exact)
    bx = TV.layout3([1, 12, 12])
    TV.paint(DS / "a_near" / "images" / "near_a.png", bx, texture=False, fmt="png")
    TV.yolo(DS / "a_near" / "labels" / "near_a.txt", [(a_src[1],) + bx[0][1:]] + [(TV.LAMBS,) + b[1:] for b in bx[1:]])
    src, dst = DS / "a_near" / "images" / "near_a.png", _mk(DS / "n_near" / "images") / "near_b.jpg"
    d = None
    for q, dw in ((95, 0), (85, 0), (70, 0), (95, 4), (90, 6), (80, 8), (60, 6)):
        with Image.open(src) as im:
            im = im.convert("RGB")
            if dw:
                im = im.resize((im.width - dw, im.height - dw // 2), Image.BILINEAR)
            im.save(dst, quality=q)
        d = bits(C.dhash(src), C.dhash(dst))
        if 0 < d <= 3:
            break
    TV.yolo(DS / "n_near" / "labels" / "near_b.txt", [(a_src[1],) + bx[0][1:]] + [(TV.LAMBS,) + b[1:]
                                                                                   for b in bx[1:]])
    exp["near_bits"] = d
    regp = REPO / "results" / "framework" / "dataset_registry.json"
    reg = json.loads(regp.read_text())
    for slug in ("t_tswcopy", "k_mixed", "a_near", "n_near"):
        reg["datasets"][slug] = {"local_path": str(DS / slug), "annotation": "bbox", "class_names": TV.A_NAMES}
    regp.write_text(json.dumps(reg))
    # the v1 never-train index: dev, test, imageweeds and the tsw images as ood22 (as v1 had them)
    entries = [[C.dhash(r["image"]), s, r["key"]] for s in ("dev", "test", "imageweeds") for r in rows[s]]
    entries += [[C.dhash(r["image"]), "ood22", r["key"]] for r in TSW]
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries, "min_expected": len(entries)}))
    C.LOCK_PATH.write_text(json.dumps({"manifests": {"train_core": C.sha256_file(C.manifest_path("train_core"))},
                                       "nevertrain_sha256": C.sha256_file(C.NEVER_TRAIN_INDEX)}))
    return rows, boxes_of, exp


def _mk(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_v1():
    V.cmd_pool(TV.args("pool", "--procs", "1"))
    V.cmd_crops(TV.args("crops"))
    V.cmd_embed(TV.args("embed", "--nshards", "1", "--procs", "1"), embedder=TV.FakeEmbedder())
    V.cmd_fit(TV.args("fit"))
    V.cmd_admit(TV.args("admit"))
    return S.build(base_frac=0.6, seed=0, workers=1)


def make_splits_v2(rows, name="v2", with_selected=True):
    """splits/<name> in inc2.guard's format: byte copies of the v1 manifests,
    tsw22, base_v2 (train_core + tsw22 + base_selected, or without
    base_selected for the equivalence test, whose v1 base is train_core only),
    the never-train and base-copy indexes (complete, 6 bits) and LOCK.json."""
    d = C.INC_DIR / "splits" / name
    d.mkdir(parents=True, exist_ok=True)
    man = {}
    for s in ("dev", "test", "imageweeds", "train_core"):
        shutil.copyfile(C.manifest_path(s), d / ("%s.jsonl" % s))
    C.write_manifest(d / "tsw22.jsonl", TSW)
    sel = C.read_manifest(V.STEP1 / S.BASE_SELECTED) if with_selected else []
    base = C.read_manifest(C.manifest_path("train_core")) + TSW + sel
    C.write_manifest(d / "base_v2.jsonl", base)
    for s in ("dev", "test", "imageweeds", "train_core", "tsw22", "base_v2"):
        man[s] = C.sha256_file(d / ("%s.jsonl" % s))
    nt = [[C.dhash(r["image"]), s, r["key"]] for s in ("dev", "test", "imageweeds") for r in rows[s]]
    part = {r["key"]: "base_selected" for r in sel}
    part.update({r["key"]: "tsw22" for r in TSW})
    bc = [[C.dhash(r["image"]), part.get(r["key"], "train_core"), r["key"]] for r in base]
    (d / "nevertrain_dhash.json").write_text(json.dumps({"entries": nt, "complete": True, "bits": 6,
                                                         "min_expected": len(nt), "splits": ["dev", "test", "imageweeds"]}))
    (d / "base_copies_dhash.json").write_text(json.dumps({"entries": bc, "complete": True, "bits": 6,
                                                          "min_expected": len(bc)}))
    lock = {"splits_version": "v2", "eval_splits": ["dev", "test", "imageweeds"], "testing": True,
            "nevertrain_sha256": C.sha256_file(d / "nevertrain_dhash.json"),
            "base_copies_sha256": C.sha256_file(d / "base_copies_dhash.json"), "manifests": man,
            "nevertrain_entries": len(nt), "base_copies_entries": len(bc)}
    (d / "LOCK.json").write_text(json.dumps(lock))
    return d / "LOCK.json", sel


# ------------------------------------------------------------ the guard
class GuardDouble:
    """The contract's GuardV2 interface over the same LOCK files (used only
    when inc2.guard cannot be loaded)."""

    def __init__(self, lock_path):
        d = pathlib.Path(lock_path).parent
        self.ev, self.base = NearHashIndex(), NearHashIndex()
        for h, s, k in json.loads((d / "nevertrain_dhash.json").read_text())["entries"]:
            self.ev.add(int(h), (s, k), max_bits=6)
        for h, p, k in json.loads((d / "base_copies_dhash.json").read_text())["entries"]:
            self.base.add(int(h), (p, k), max_bits=6)

    def check(self, h, variants):
        m = self.ev.find(int(h))
        if m is not None:
            return "near_eval_v2", {"split": m[0][0], "key": m[0][1], "bits": m[1]}
        for name, v in variants.items():
            m = self.ev.find(int(v))
            if m is not None:
                return "near_eval_variant", {"split": m[0][0], "key": m[0][1], "bits": m[1], "variant": name}
        for name, v in variants.items():
            m = self.base.find(int(v))
            if m is not None:
                return "base_copy", {"part": m[0][0], "key": m[0][1], "bits": m[1], "variant": name}
        return None, None


def double_variants(path):
    try:
        with Image.open(path) as im:
            im.load()
            return {k: int(v) for k, v in L.dhash_variants(im).items()}
    except Exception:
        return None


def make_guards(lock):
    try:
        g = SS.load_guards(lock)
        NOTES.append("GuardV2: the real inc2.guard (group A) over the test LOCK v2")
        return g
    except SS.StreamError as e:
        NOTES.append("GuardV2: inc2.guard not usable here (%s); the contract double is used" % str(e)[:200])
        return SS.Guards(GuardDouble(lock), double_variants, record={"lock": str(lock), "double": True})


# ------------------------------------------------------- the copy detector
class HueEmbedder:
    """A whole-image descriptor that survives crops, shears and brightness
    changes: hue histograms (16 bins) of lit, saturated pixels on a 3 x 3
    grid, L2-normalised (funnel's test descriptor, made spatial so that
    unrelated pictures of the same palette stay apart)."""
    bins, grid = 16, 3
    dim = bins * grid * grid

    def __init__(self, name="facebook/dinov2-base:cls"):
        self.name = name

    def __call__(self, pils):
        out = []
        for p in pils:
            hsv = np.asarray(p.convert("RGB").convert("HSV"), dtype=np.int64)
            H, W = hsv.shape[:2]
            parts = []
            for gy in range(self.grid):
                for gx in range(self.grid):
                    c = hsv[gy * H // self.grid:(gy + 1) * H // self.grid, gx * W // self.grid:(gx + 1) * W // self.grid]
                    m = (c[..., 1] > 40) & (c[..., 2] > 30)
                    parts.append(np.bincount((c[..., 0][m] * self.bins) // 256, minlength=self.bins))
            h = np.concatenate(parts).astype(np.float32)
            n = np.linalg.norm(h)
            out.append(h / n if n > 0 else h)
        return np.stack(out)


def augment(src, dst, kind):
    with Image.open(src) as im0:
        im = im0.convert("RGB")
    W, H = im.size
    if kind == "crop":
        c = int(round(0.15 * W)), int(round(0.15 * H))
        im = im.crop((c[0], c[1], W, H)).resize((W, H), Image.BILINEAR)
    elif kind == "shear":
        im = im.transform((W, H), Image.AFFINE, (1, 0.35, -0.175 * H, 0, 1, 0), resample=Image.BILINEAR)
    elif kind == "bright":
        im = ImageEnhance.Brightness(im).enhance(1.25).crop((6, 4, W, H)).resize((W, H))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=90)


def min_variant_bits(path, targets):
    v = double_variants(path)
    return min(bits(x, t) for x in v.values() for t in targets)


def passed_record(threshold):
    """A calibration record that shows it passed (inc2.guard.calibration_problems
    finds nothing): every augmentation family at recall >= 0.95, both negative
    sets at <= 1 % false positives, H6's gates, the never-train dHash radius."""
    return {"ok": True, "why": [], "cos_threshold": threshold, "dhash_bits_max": 6,
            "recall_min": L.RECALL_MIN, "fpr_max": L.FPR_MAX,
            "positives": {f: {"n": 40, "hits": 40} for f in L.FAMILIES},
            "negatives": {"pairs_7_10": {"n": 200, "false_hits": 0}, "hard": {"n": 200, "false_hits": 1}}}


def write_calibration(path, threshold, name="facebook/dinov2-base:cls", record=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"calibration": record if record is not None else passed_record(threshold),
                                "detector": {"descriptor": {"embedder": name}}}))


class Kill(BaseException):
    pass


class KillEmbedder(TV.FakeEmbedder):
    def __call__(self, pils):
        if self.kill_after is not None and self.calls >= self.kill_after:
            raise Kill()
        return super().__call__(pils)


class DriftEmbedder(TV.FakeEmbedder):
    def __call__(self, pils):
        X = super().__call__(pils)
        return X + np.random.default_rng(0).normal(0, 0.05, X.shape)


class OtherEmbedder(TV.FakeEmbedder):
    name = "another-model"


def queue_view(rows, drop=("batch", "group", "hold_deadline", "stream_pins_sha", "label", "image", "input",
                           "released", "unmasked_image", "admitted_utc")):
    return {r["key"]: {k: v for k, v in r.items() if k not in drop} for r in rows}


def partition(rows):
    by = collections.defaultdict(set)
    for r in rows:
        by[r["group"]].add(r["key"])
    return sorted(sorted(g) for g in by.values())


def v1_box_verdicts():
    crops = V.Crops()
    pv, _codes = S.read_pool_verdicts(V.POOL_VERDICTS, crops)
    lookup = V._box_verdict_lookup(crops, "pool", lambda i: V.VERDICT_CODES[int(pv[i])])
    meta = {m["key"]: m for m in V._read_jsonl(V.POOL_META)}
    return {(k, b): lookup(k, b)[0] for k, m in meta.items() for b in range(len(m["boxes"]))}, meta


def census_from_v1():
    """The funnel census's veto numbers (adapters/inc_step1: verified target
    boxes per class, admitted or lost to the image rule), computed here
    independently of step1_stream."""
    verdicts, meta = v1_box_verdicts()
    admitted = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    per = collections.defaultdict(lambda: {"verified": 0, "admitted": 0})
    lost, imgs = 0, set()
    for k, m in meta.items():
        for b, box in enumerate(m["boxes"]):
            if int(box[0]) < 12 and verdicts[(k, b)] == V.VERIFIED:
                per[C.CLASS_NAMES[int(box[0])]]["verified"] += 1
                if k in admitted:
                    per[C.CLASS_NAMES[int(box[0])]]["admitted"] += 1
                else:
                    lost += 1
                    imgs.add(k)
    return {"veto": {"lost_boxes": lost, "images": len(imgs), "per_class": {k: dict(v) for k, v in per.items()}}}, imgs


def batch_rows(layout, bid, name="ingest.jsonl"):
    return SS._read_jsonl(layout.batch_dir(bid) / name)


def by_key(rows):
    return {r["key"]: r for r in rows if r.get("key")}


# ----------------------------------------------------------- equivalence
def test_equivalence(lock, guards):
    print("equivalence with verify pool + crops + admit (image rule, from empty)")
    lay = SS.Layout(TMP / "stream_eq")
    SS.bootstrap(lay, lock, testing=True, seed_v1=False, procs=1)
    doc = SS.run_batch(lay, "registry", lambda: SS.list_registry(lay), TV.FakeEmbedder(), guards, None, rule="image",
                       procs=1, kind=SS.KIND_REGISTRY)
    bid = doc["batch"]
    rows = batch_rows(lay, bid)
    v1_pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    pool = {r["key"]: r for r in rows if r["decision"] == "pool"}
    check("the same pool keys", set(pool) == set(v1_pool), sorted(set(pool) ^ set(v1_pool)))
    check("the same label sha256 per key",
          all(pool[k]["label_sha256_full"] == v1_pool[k]["label_sha256"] for k in set(pool) & set(v1_pool)))
    v1v, _meta = v1_box_verdicts()
    adm = {r["key"]: r for r in batch_rows(lay, bid, "admission.jsonl")}
    mine = {(k, b): v for k, r in adm.items() for b, v in enumerate(r["verdicts"])}
    check("the same verdict for every pool box", mine == v1v and len(mine) > 20,
          sorted((k, mine.get(k), v1v.get(k)) for k in set(mine) | set(v1v) if mine.get(k) != v1v.get(k))[:5])
    admitted = {k for k, r in adm.items() if r["admission"] == MK.WHOLE}
    want = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    check("the same admitted set (%d images)" % len(want), admitted == want, sorted(admitted ^ want))
    check("the image rule never masks", not any(r["admission"] == MK.MASKED for r in adm.values()))
    q = {r["key"]: r for r in SS.load_queue(lay)}
    check("every admitted image is queued, whole, with its own label sha256",
          set(q) == admitted and all(q[k]["label_sha256"] == v1_pool[k]["label_sha256"] for k in q))
    ps = json.loads(V.POOL_SUMMARY.read_text())
    dec = collections.Counter(r["decision"] for r in rows)
    v1d = collections.Counter()
    for st in ps["per_slug"].values():
        if not st.get("calibration_only"):          # the leave-4-out copies are never pool data in either
            v1d.update(st["dropped"])
    check("the drops map one to one: near_eval -> near_eval_v2 (the two tsw copies are base copies in v2), "
          "cwd12_copy -> base copy (known truth), exact_dup -> exact_dup or held_join_conflict, unhashable, "
          "no_label, no_boxes, unmapped_class",
          dec["near_eval_v2"] + dec["near_eval_variant"] == v1d["near_eval"] - 2
          and dec["knowntruth"] + dec["base_copy"] == v1d["cwd12_copy"] + 2 and dec["held_join_conflict"] == 1
          and dec["exact_dup"] + dec["held_join_conflict"] == v1d["exact_dup"]
          and all(dec[k] == v1d[k] for k in ("unhashable", "no_label", "no_boxes", "unmapped_class")),
          (dict(dec), dict(v1d)))
    kt = doc["knowntruth"]
    check("known truth: the planted wrong label on a train_core copy is not verified, the base copies are measured "
          "per set (train_core, tsw22)", set(kt["per_set"]) == {"train_core", "tsw22"}
          and kt["overall"]["verified_precision"] == 1.0 and kt["overall"]["label_correct_rate"] < 1.0, kt)
    return lay


# -------------------------------------------------------- split invariance
def test_split_invariance(lock, guards):
    print("split invariance: two batches give the same queue as one")
    one, two = SS.Layout(TMP / "stream_one"), SS.Layout(TMP / "stream_two")
    for lay in (one, two):
        SS.bootstrap(lay, lock, testing=True, seed_v1=False, procs=1)
    SS.run_batch(one, "registry", lambda: SS.list_registry(one), TV.FakeEmbedder(), guards, None, procs=1)
    ps = json.loads(V.POOL_SUMMARY.read_text())
    slugs = sorted(s for s, v in ps["per_slug"].items() if not v.get("calibration_only"))
    first, second = slugs[:2], slugs[2:]
    check("fixture: the near-duplicate pair and the conflicting exact duplicate straddle the split",
          "a_near" in first and "n_near" in second and "a_species" in first and "b_other" in second, (first, second))
    for part in (first, second):
        SS.run_batch(two, "registry:%s" % ",".join(part), lambda p=part: SS.list_registry(two, p), TV.FakeEmbedder(),
                     guards, None, procs=1)
    q1, q2 = SS.load_queue(one), SS.load_queue(two)
    v1, v2 = queue_view(q1), queue_view(q2)
    diff = sorted(k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k))
    check("the same effective queue rows (%d), batch ids, group numbers and paths aside" % len(v1), not diff and v1,
          [(k, {f: (v1.get(k, {}).get(f), v2.get(k, {}).get(f)) for f in set(v1.get(k, {})) | set(v2.get(k, {}))
                if v1.get(k, {}).get(f) != v2.get(k, {}).get(f)}) for k in diff[:3]])
    check("the same near-duplicate groups", partition(q1) == partition(q2), (partition(q1), partition(q2)))
    near = [r for r in q2 if r["key"] in ("a_near__near_a", "n_near__near_b")]
    check("the near-duplicate pair shares one group across the two batches",
          len(near) == 2 and near[0]["group"] == near[1]["group"], [(r["key"], r["group"]) for r in near])
    held = [r["key"] for r in q2 if "join_conflict" in r["holds"]]
    check("the conflicting exact duplicate's twin, queued by the first batch, is held by the second (an event)",
          held == ["a_species__a_ok_0"], held)
    ev = SS._read_jsonl(two.events)
    check("and a rejoin is queued for the duplicate", any(e["event"] == "rejoin_needed" and e["key"] == "b_other__train__b_dup"
                                                          for e in ev), ev[:3])
    st = SS.verify_state(two)
    check("the two-batch state verifies (ledger chain, contiguous crop ids)", st == [], st)
    led = SS.verify_ledger(two.ledger)
    pins = SS.read_pins(two)
    check("global crop ids: the first batch starts at v1's count, the second right after",
          led[0]["crop_offset"] == pins["crop_offset"] == V.Crops().n
          and led[1]["crop_offset"] == led[0]["crop_offset"] + led[0]["crops"] and led[1]["crops"] > 0,
          [(e["crop_offset"], e["crops"]) for e in led])
    return one, two


def test_groups_unit():
    print("near-duplicate groups (union-find)")
    g = SS.Groups({})
    a, _m = g.assign(0b0)
    b, _m = g.assign(0b111111 << 20)             # far from a
    c, merged = g.assign((0b1) | (0b11 << 20))    # 1 bit from a... and 4 from b: joins a only
    check("a new hash joins the group it is near", c == a and not merged)
    d, merged = g.assign((0b111111 << 20) | (1 << 30))     # 1 bit from b, 6 from c
    check("another joins b", d == b and not merged, (a, b, d, merged))
    x = d
    data = g.to_data()
    g2 = SS.Groups(data)
    check("the index round-trips", g2.to_data() == data and g2.find(x) == g.find(x))
    g3 = SS.Groups({})
    p, _ = g3.assign(0)
    q, _ = g3.assign(0b1111 << 8)                 # 4 bits: separate
    r, merged = g3.assign(0b11 << 8)              # 2 bits from both: merges them
    check("a hash near two groups merges them into the smaller id, and says so",
          r == min(p, q) and merged == [max(p, q)] and g3.find(q) == g3.find(p))


def test_intake_licence_unit():
    print("an intake row's licence under a person's override")
    row = {"key": "src_x__a", "source": "src_x", "licence": "unresolved", "licence_class": "unresolved",
           "research_only": False}

    def state(ov, **lrec):
        return SS.intake_licence_state(row, {"licence": dict({"id": "unresolved", "class": "unresolved",
                                                              "research_only": False}, override=ov, **lrec)})
    owner = {"id": "research-only", "class": "research_only", "research_only": True, "decided_by": "human:owner",
             "decided_utc": "2026-09-30", "reason": "research use only"}
    check("the collector's record of a person's decision resolves the row: its id, research_only as decided",
          state(owner) == ("research-only (person override)", True), state(owner))
    check("a restriction in the override's text keeps research_only (collect.licence.restricted), whatever the "
          "record's flag says", state(dict(owner, id="CC BY 4.0, academic use only", research_only=False))[1] is True
          and state("CC BY 4.0, research use only")[1] is True and state("Non-Commercial terms")[1] is True)
    check("a permissive override text is not research-only; a non-commercial one is (as before)",
          state("cc-by-4.0") == ("cc-by-4.0 (person override)", False)
          and state({"id": "cc-by-nc-4.0", "decided_by": "owner"})[1] is True, (state("cc-by-4.0"),))
    check("an override that is neither a licence text nor a decision record refuses (StreamError, fail closed)",
          raises(lambda: state(["research-only"]), text="neither a licence text") is True
          and raises(lambda: state(5), text="neither a licence text") is True)
    check("a decision record without a licence text refuses (the row's own 'unresolved' is never read as one)",
          raises(lambda: state({"decided_by": "human:x", "research_only": False}), text="no licence text") is True
          and raises(lambda: state(dict(owner, id="  ")), text="no licence text") is True)
    check("fail closed: an override whose text names no known licence is research-only whatever its record says; "
          "a record without research_only false is research-only; a permissive record with research_only false "
          "is not",
          state(dict(owner, id="unknown", research_only=False)) == ("unknown (person override)", True)
          and state("unknown")[1] is True and state("Custom terms, see card")[1] is True
          and state({"id": "cc-by-4.0", "decided_by": "human:x"})[1] is True
          and state(dict(owner, id="cc-by-4.0", research_only=False)) == ("cc-by-4.0 (person override)", False),
          (state(dict(owner, id="unknown", research_only=False)), state({"id": "cc-by-4.0", "decided_by": "human:x"})))



# ---------------------------------------------------------------- b0000
def test_backfill(lock, guards, sel):
    print("b0000: D-B over the v1 pool, reconciled with the census")
    lay = SS.Layout(TMP / "stream_main")
    SS.bootstrap(lay, lock, testing=True, seed_v1=True, procs=1)
    pins = SS.read_pins(lay)
    seen = SS.Index(lay, "seen", {}).data
    check("bootstrap seeds the seen index from the v1 pool, and records the pins",
          len(seen) == len(C.read_manifest(V.POOL)) and pins["splits"]["lock_sha256"] == C.sha256_file(lock)
          and pins["crop_offset"] == V.Crops().n and pins["embedder"]["name"] == "fake-colour"
          and set(pins["verifier"]["files"]) >= {"probe.joblib", "verifier.npz", "oof_keys.json"})
    check("bootstrap runs once", raises(lambda: SS.bootstrap(lay, lock, testing=True), text="runs once"))
    ref = SS.Reference(lay.reference)
    check("the reference pack's l1/l2 centres are select's clusters (same seed, same features)",
          ref.meta["select_clusters_agreement"]["same_l1_l2"] == ref.meta["select_clusters_agreement"]["images"] > 0,
          ref.meta["select_clusters_agreement"])
    census, cand = census_from_v1()
    check("fixture: four vetoed images (the small-box one, two with a conflict, one overlapping)",
          len(cand) == 4 and {"k_mixed__k_veto1", "k_mixed__k_veto2", "k_mixed__k_overlap",
                              "a_species__a_small"} == cand, sorted(cand))
    inc = C.read_manifest(V.STEP1 / S.POOL)
    rl = C.INC_DIR / "realloop_v1"
    drawn = sorted(r["key"] for r in inc)[:2]
    msha = C.write_manifest(rl / "manifests" / "inc_02.jsonl", [r for r in inc if r["key"] in drawn])
    (rl / "exp.json").write_text(json.dumps({"steps": [{"name": "V2", "manifest": "/cluster/only/inc_02.jsonl",
                                                        "select_manifest": "inc_02.jsonl", "manifest_sha256": msha}]}))
    (rl / "report.json").write_text(json.dumps({"steps": [{"step": "V2", "chains": {
        "full": {"verdict": "REJECT", "attribution": {"blame": "recipe"}}}}]}))
    bad = json.loads(json.dumps(census))
    bad["veto"]["images"] += 1
    (TMP / "census_bad.json").write_text(json.dumps(bad))
    check("a census that does not match refuses, and nothing is committed",
          raises(lambda: SS.backfill(lay, guards, None, TMP / "census_bad.json"), text="does not reconcile")
          and not lay.queue.exists() and SS.verify_ledger(lay.ledger) == []
          and json.loads((lay.batch_dir("b0000") / "reconciliation.json").read_text())["checks"][
              "images_match_census"] is False)
    check("without a census it refuses unless told to record the check as not done",
          raises(lambda: SS.backfill(lay, guards, None, TMP / "no_census.json"), text="census_v1.json"))
    cp = copy_layout(lay, "stream_b0fail")
    check("a b0000 that did not reconcile blocks no other verb (an admit runs; the backfill waits for a person)",
          SS._in_progress(cp) is None and SS.run_batch(cp, "registry:c_nonames",
                                                        lambda: SS.list_registry(cp, ["c_nonames"]),
                                                        TV.FakeEmbedder(), guards, None, procs=1) is None
          and "b0000" in SS.write_status(cp)["batches"]["in_progress"])
    (C.INC_DIR / "funnel").mkdir(parents=True, exist_ok=True)
    (C.INC_DIR / "funnel" / "census_v1.json").write_text(json.dumps(census))
    l5p = lock.parent / "l5_excluded.jsonl"
    l5p.write_text(json.dumps({"key": "not_a_base_image", "source": "x"}) + "\n")
    check("an L-5 list naming a key outside base B refuses", raises(lambda: SS.backfill(lay, guards),
                                                                      text="not base B images"))
    l5p.write_text(json.dumps({"key": sel[0]["key"], "source": sel[0]["source"]}) + "\n")
    doc = SS.backfill(lay, guards)
    rec = doc["reconciliation"]
    check("every candidate accounted for: 0 whole + 3 masked + 1 refused_overlap + 0 not admitted = 4",
          (rec["whole"], rec["masked"], rec["refused_overlap"], rec["not_admitted"], rec["candidates"]) == (0, 3, 1, 0, 4)
          and rec["ok"], rec)
    check("the census checks pass: images, lost boxes, lost per class; recovered <= census per species",
          all(rec["checks"].values()) and set(rec["checks"]) >= {"images_match_census", "boxes_match_census",
                                                                 "recovered_le_census", "lost_per_class_match_census"}
          and all(n <= census["veto"]["per_class"][k]["verified"] - census["veto"]["per_class"][k]["admitted"]
                  for k, n in rec["recovered_per_class"].items()) and rec["recovered_boxes"] == 3, rec)
    check("the contract's 457 is recorded, not asserted", rec["contract_images"] == {
        "contract": 457, "got": 4, "recorded_not_asserted": True})
    q = {r["key"]: r for r in SS.load_queue(lay)}
    inc_keys = {r["key"] for r in inc}
    masked = {k for k, r in q.items() if r["admission"] == MK.MASKED}
    check("the queue: the v1 increment pool (whole) and the three masked rows",
          set(q) == inc_keys | masked and masked == cand - {"k_mixed__k_overlap"}
          and all(q[k]["admission"] == MK.WHOLE for k in inc_keys), sorted(set(q) ^ (inc_keys | masked)))
    check("base B's images are never queued, and are counted (in base_v2 / dropped by L-5)",
          not ({r["key"] for r in sel} & set(q)) and doc["base"]["base_b"] == len(sel)
          and doc["base"]["in_base_v2"] == len(sel) and doc["base"]["dropped_l5"] == 1
          and doc["base"]["l5_excluded"]["path"] == str(l5p), doc["base"])
    hum = [r for r in SS._read_jsonl(lay.human) if r["batch"] == "b0000"]
    check("the overlapping one goes to the human queue", [r["key"] for r in hum] == ["k_mixed__k_overlap"]
          and hum[0]["reason"] == "refused_overlap")
    today21 = SS._date(21)
    check("masked rows hold funnel_F9 with a 21-day deadline (P8, §6.7)",
          all("funnel_F9" in q[k]["holds"] and q[k]["hold_deadline"]["funnel_F9"] == today21 for k in masked)
          and not any("funnel_F9" in q[k]["holds"] for k in inc_keys))
    check("without a licence and without a copy scan every row is held (licence, h6_scan)",
          all({"licence", "h6_scan"} <= set(r["holds"]) for r in q.values()) and not any(
              r["eligible_step1"] for r in q.values()))
    check("a v1 slug with evaluation near-copies is LuLab and its rows hold h6_scan as lab evidence",
          all(r["lab_group"] == "LuLab" and r["h6_reason"] == "lab_evidence" for r in q.values()
              if r["source"] == "a_species")
          and all(r["h6_reason"] == "not_scanned" for r in q.values() if r["source"] in ("b_other", "k_mixed")))
    check("realloop_v1's draws are tagged prior", all(q[k]["prior"] == "realloop_v1:V2:REJECT(recipe)"
                                                      for k in drawn if k in q)
          and sum(1 for r in q.values() if r["prior"]) == len([k for k in drawn if k in q]))
    k1 = q["k_mixed__k_veto1"]
    lab = C.read_yolo(k1["label"])
    a = np.asarray(Image.open(k1["image"]).convert("RGB"))
    orig = np.asarray(Image.open(k1["unmasked_image"]).convert("RGB"))
    H, W = a.shape[:2]
    mean = np.rint(orig.reshape(-1, 3).astype(np.float64).mean(0)).astype(np.uint8)
    mbox = [b for b in json.loads(json.dumps(V._read_jsonl(V.POOL_META)))
            if b["key"] == "k_mixed__k_veto1"][0]["boxes"][1]
    x0, y0, x1, y1 = MK.pixel_rect(mbox, W, H)
    check("a masked row: its label holds only the kept boxes, its image is the masked PNG (mean colour in the "
          "masked box), its sha256 and dHashes recorded",
          [b[0] for b in lab] == [0, 12] and (a[y0:y1, x0:x1] == mean).all() and k1["n_masked"] == 1
          and C.sha256_file(k1["image"]) == k1["sha256"] and k1["dhash_masked"] is not None
          and 0 < k1["masked_area_frac"] < 0.2 and k1["species_boxes"][0] == 1, (lab, k1))
    check("queue rows carry every documented field, the manifest fields and the pins sha",
          all(set(SS.QUEUE_KEYS) <= set(r) and set(C.MANIFEST_KEYS) <= set(r)
              and r["stream_pins_sha"] == C.sha256_file(lay.stream_json) and r["verifier"] == "v1"
              and len(r["species_boxes"]) == 12 for r in q.values()),
          sorted(set(SS.QUEUE_KEYS) - set(next(iter(q.values())))))
    n = len(SS.verify_ledger(lay.ledger))
    again = SS.backfill(lay, guards)
    check("a committed b0000 is a no-op", again["batch"] == "b0000" and len(SS.verify_ledger(lay.ledger)) == n)
    check("the state verifies", SS.verify_state(lay) == [], SS.verify_state(lay))
    return lay


# ----------------------------------------------------- planted guard cases
def fresh(path, classes, labels_src, texture=False):
    bx = TV.layout3(classes)
    TV.paint(path, bx, texture=texture, fmt=path.suffix.lstrip("."))
    TV.yolo(path.parent.parent / "labels" / (path.stem + ".txt"),
            [(s,) + b[1:] for s, b in zip(labels_src, bx)])
    return bx


def a_label(inc_boxes):
    """INC boxes in a_species' source ids."""
    return [(TV.A_SRC[b[0]] if b[0] < 12 else TV.LAMBS,) + tuple(b[1:]) for b in inc_boxes]


def test_registry_planted(lay, guards, rows, sel):
    print("registry batch over new paths of v1 slugs: the planted guard cases")
    a = DS / "a_species"
    img, lbl = a / "images", a / "labels"
    dev, test = rows["dev"], rows["test"]
    one = [(TV.A_SRC[0], 0.5, 0.5, 0.2, 0.2)]
    shutil.copyfile(dev[0]["image"], img / "p_devbyte.jpg")
    TV.yolo(lbl / "p_devbyte.txt", one)
    Image.open(test[0]["image"]).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(img / "p_testflip.png")
    TV.yolo(lbl / "p_testflip.txt", one)
    Image.open(dev[2]["image"]).transpose(Image.Transpose.ROTATE_90).save(img / "p_devrot.png")
    TV.yolo(lbl / "p_devrot.txt", one)
    TV.clean_copy(sel[0]["image"], img / "p_basecopy.jpg")
    TV.yolo(lbl / "p_basecopy.txt", a_label(C.read_yolo(sel[0]["label"])))
    TV.clean_copy(TSW[2]["image"], img / "p_tswcopy.jpg")
    TV.yolo(lbl / "p_tswcopy.txt", a_label(C.read_yolo(TSW[2]["label"])))
    q = {r["key"]: r for r in SS.load_queue(lay)}
    whole = sorted(k for k, r in q.items() if r["admission"] == MK.WHOLE and r["source"] != "a_species")
    t1, t2 = q[whole[0]], q[whole[1]]
    shutil.copyfile(t1["image"], img / "p_dupsame.png")
    TV.yolo(lbl / "p_dupsame.txt", a_label(C.read_yolo(t1["label"])))
    shutil.copyfile(t2["image"], img / "p_dupdiff.png")
    TV.yolo(lbl / "p_dupdiff.txt", a_label([(9,) + tuple(b[1:]) for b in C.read_yolo(t2["label"])]))
    eval_hashes = [C.dhash(r["image"]) for s in ("dev", "test", "imageweeds") for r in rows[s]]
    aug = {}
    for kind, src in (("crop", dev[1]), ("shear", test[1]), ("bright", test[2])):
        p = img / ("p_aug_%s.jpg" % kind)
        augment(src["image"], p, kind)
        TV.yolo(lbl / ("p_aug_%s.txt" % kind), one)
        aug[kind] = (p, src)
    fresh(img / "p_new_ok.png", [3, 12, 12], [TV.A_SRC[3], TV.LAMBS, TV.LAMBS])
    fresh(img / "p_new_veto.png", [6, 8, 12], [TV.A_SRC[6], TV.LAMBS, TV.LAMBS])
    far = {k: min_variant_bits(p, eval_hashes) for k, (p, _s) in aug.items()}
    check("fixture: the augmented copies are more than 6 bits from every evaluation image under all 8 variants",
          all(v > 6 for v in far.values()), far)
    # the copy detector: a hue descriptor, a threshold between copies and non-copies
    from weed_optimizer_framework.tools.funnel import embed as FE

    def descs(paths):                        # the detector's own view (funnel.embed, EXIF transposed, shorter edge 256)
        res = FE.embed_images([{"key": str(i), "image": str(p)} for i, p in enumerate(paths)], None, HueEmbedder(),
                              prepare=L.prepare_hashed, n_hashes=len(L.VARIANTS))
        X = np.asarray(res["X"], dtype=np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    A, O = descs([p for p, _s in aug.values()]), descs([s["image"] for _p, s in aug.values()])
    cos_copy = [float(x) for x in (A * O).sum(axis=1)]
    pool_imgs = [r["image"] for r in C.read_manifest(V.POOL)] + [str(img / "p_new_ok.png"), str(img / "p_new_veto.png")]
    eval_imgs = [r["image"] for s in ("dev", "test", "imageweeds") for r in rows[s]]
    cos_other = float(max((descs(pool_imgs) @ descs(eval_imgs).T).max(), 0))
    theta = round((min(cos_copy) + cos_other) / 2, 4)
    check("fixture: the descriptor separates the copies (min %.3f) from every pool image (max %.3f)"
          % (min(cos_copy), cos_other), min(cos_copy) > cos_other + 0.02, (cos_copy, cos_other))
    # a record that says ok is not trusted: a threshold that can never fire (above 1) with loose gates, a
    # family below its recall gate, or no embedder named -> not used, the rows stay held, and it is recorded
    flp = C.INC_DIR / "funnel" / "leak_v1.json"
    loose = dict(passed_record(7.0), recall_min=0.5, fpr_max=0.5)
    weak = passed_record(theta)
    weak["positives"]["shear"] = {"n": 40, "hits": 20}
    for why, rec, name in (("threshold 7.0, loose gates", loose, "facebook/dinov2-base:cls"),
                           ("a family below its recall gate", weak, "facebook/dinov2-base:cls"),
                           ("no embedder named", passed_record(theta), None)):
        write_calibration(flp, theta, name=name, record=rec)
        bad_sc = SS.load_scanner(lay, funnel_path=flp, own_path=TMP / "none.json", embedder=HueEmbedder(), procs=1)
        check("a calibration that says ok but does not show it passed is not used (%s); the refusal is recorded"
              % why, not bad_sc.available and bad_sc.which(False, True) is None
              and str(flp) in bad_sc.record().get("rejected", {}), bad_sc.record())
    write_calibration(flp, theta)
    scanner = SS.load_scanner(lay, embedder=HueEmbedder(), procs=1)
    junk = TMP / "junk" / "not_an_image.jpg"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"\xff\xd8\xff not a jpeg")
    res = scanner.scan([{"key": "junk", "path": junk}, {"key": "fine", "path": C.read_manifest(V.POOL)[0]["image"]}],
                       "funnel")
    check("an image the copy detector cannot describe is refused on its own (unhashable_embed), never failing the "
          "batch", (res["junk"] or {}).get("unscannable") and SS.scan_reason(res["junk"]) == "unhashable_embed"
          and res["fine"] is None, res)
    check("the funnel's passed leak_v1.json is reused by sha256", scanner.cal["funnel"] is not None
          and scanner.cal["funnel"]["sha256"] == C.sha256_file(C.INC_DIR / "funnel" / "leak_v1.json")
          and scanner.cal["own"] is None)
    pre = copy_layout(lay, "stream_pre_d28v2")    # the same state, to replay this batch as one made before D28-v2
    doc = SS.run_batch(lay, "registry:a_species", lambda: SS.list_registry(lay, ["a_species"]), TV.FakeEmbedder(),
                       guards, scanner, procs=1)
    rows_all = batch_rows(lay, doc["batch"])
    rows_b = by_key(rows_all)
    dec = {r["stem"]: r["decision"] for r in rows_all}
    want = {"p_devbyte": "near_eval_v2", "p_testflip": "near_eval_variant", "p_devrot": "near_eval_variant",
            "p_basecopy": "base_copy", "p_tswcopy": "knowntruth", "p_dupsame": "exact_dup",
            "p_dupdiff": "held_join_conflict", "p_aug_crop": "near_eval_embed", "p_aug_shear": "near_eval_embed",
            "p_aug_bright": "near_eval_embed", "p_new_ok": "pool", "p_new_veto": "pool"}
    check("every planted case is caught by its guard, the clean images pass",
          {k: dec.get(k) for k in want} == want,
          {k: (dec.get(k), v) for k, v in want.items() if dec.get(k) != v})
    check("the embedding hit names the evaluation image it copies",
          rows_b["a_species__p_aug_crop"]["guard_match"]["eval_key"] == dev[1]["key"]
          and rows_b["a_species__p_aug_shear"]["guard_match"]["eval_key"] == test[1]["key"],
          rows_b["a_species__p_aug_crop"]["guard_match"])
    test_eval_hit_cosines(lay, doc, rows_all, scanner, theta, guards)
    test_eval_hit_sidecars(pre, guards, scanner, theta, rows)
    pre = {r["stem"]: r["decision"] for r in rows_all if r["stem"] not in want}
    check("the v1 images of the slug are not read again; the four v1 dropped before hashing (not in the v1 dHash "
          "cache) are read once, dropped the same way, and marked processed",
          pre == {"a_corrupt": "unhashable", "a_empty": "no_boxes", "a_nolabel": "no_label",
                  "a_unmapped": "unmapped_class"} and not any(r["stem"].startswith("a_ok") for r in rows_all)
          and SS.list_registry(lay, ["a_species"])[0] == [], pre)
    check("a refused near-eval image gets no key (as in v1)", all(r["key"] is None for r in rows_all
                                                                 if r["decision"].startswith("near_eval_v")))
    qq = {r["key"]: r for r in SS.load_queue(lay)}
    check("the duplicate with other labels holds its queued twin (join_conflict) and queues a rejoin",
          "join_conflict" in qq[t2["key"]]["holds"] and "join_conflict" not in qq[t1["key"]]["holds"]
          and any(e["event"] == "rejoin_needed" and e["twin"] == t2["key"] for e in SS._read_jsonl(lay.events)))
    new_ok, new_veto = qq["a_species__p_new_ok"], qq["a_species__p_new_veto"]
    check("scanned by the funnel's calibration, a same-lab row is not held h6_scan; the licence hold stays",
          new_ok["holds"] == ["licence"] and new_ok["scanned"] == "funnel" and new_ok["admission"] == MK.WHOLE)
    check("a fresh image with a verified target and a conflict is masked in a registry batch",
          new_veto["admission"] == MK.MASKED and new_veto["species_boxes"][6] == 1 and new_veto["n_masked"] == 1)
    kt = doc["knowntruth"]
    check("the batch's known truth: the tsw copy box-matched to its expert labels", "tsw22" in kt["per_set"]
          and kt["per_set"]["tsw22"]["boxes"] == 3 and kt["overall"]["verified_precision_wilson_lb"] is not None, kt)
    return scanner, theta


def test_eval_hit_cosines(lay, doc, rows_all, scanner, theta, guards):
    """D28-v2 (docs/CONTINUOUS_LOOP.md, amendment 2026-10-03): the admit job
    weighs every dHash copy of an evaluation image by its pair cosine with the
    evaluation image GuardV2 matched, from the copy scanner's own descriptors;
    batch.json keeps the record, status.json folds it per source, and the
    autopilot's D28 reads it."""
    print("D28-v2: the dHash hits' pair cosines (the admit job)")
    from weed_optimizer_framework.tools.inc2 import embed_calibration as EC
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    check("step1_stream's restated v2 calibration format is inc2.embed_calibration's",
          SS.V2_CAL_FORMAT == EC.FORMAT, SS.V2_CAL_FORMAT)
    stems = {r["stem"]: r for r in rows_all}
    hits = {s: stems[s] for s in ("p_devbyte", "p_testflip", "p_devrot")}
    check("every row GuardV2 refused as a dHash copy of an evaluation image carries its pair cosine with the image "
          "it matched (the byte copy of dev: 1.0)", all(r.get("pair_cos") is not None for r in hits.values())
          and hits["p_devbyte"]["pair_cos"] == 1.0 and all(r["decision"] in EH.DHASH_HIT_REASONS
                                                           for r in hits.values()),
          {s: (r["decision"], r.get("pair_cos"), r.get("guard_match")) for s, r in hits.items()})
    rec = doc.get("eval_hits") or {}
    row = (rec.get("per_source") or {}).get("a_species") or {}
    check("batch.json records them (eval_hits: 3 hits, 3 weighed, the calibration's threshold and embedder; no "
          "evaluation key in the record)", rec.get("format") == EH.FORMAT and rec.get("hits") == 3
          and rec.get("scored") == 3 and row.get("pair_cos") == sorted((r["pair_cos"] for r in hits.values()),
                                                                       reverse=True)
          and rec.get("copy_threshold") == theta and rec.get("embedder") == scanner.index.embedder.name
          and "dev__" not in json.dumps(rec) and "test__" not in json.dumps(rec), rec)
    st = SS.write_status(lay)
    ps = st["per_source"].get("a_species") or {}
    n_dh = sum(int(ps.get("decision:%s" % k, 0)) for k in EH.DHASH_HIT_REASONS)
    check("status.json folds them per source: eval_hits_scored, eval_hit_pair_cos (descending), "
          "eval_hit_copy_threshold; the schema still checks", ps.get("eval_hits_scored", 0) >= 3
          and len(ps.get("eval_hit_pair_cos") or []) == ps.get("eval_hits_scored")
          and ps["eval_hit_pair_cos"] == sorted(ps["eval_hit_pair_cos"], reverse=True)
          and ps.get("eval_hit_copy_threshold") == theta and SS.check_status(st) == [], ps)
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
        from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS
        ev = E.from_texts({"step1_stream/status.json": json.dumps(st)}, "x", context={"sid": "none"})
        d = DS.by_id(DS.detect(ev, LS.load_domain("weed"), LS.load_thresholds(), only=("D28",)))["D28"]
        leak = next((h for h in (d.get("detail") or {}).get("leaks") or [] if h["source"] == "a_species"), {})
        dv = (leak.get("verdict") or {}).get("dhash") or {}
        check("the autopilot's D28 reads the fold: a_species leaks %s" % (
            "by the byte copy at or above the copy threshold" if len(ps["eval_hit_pair_cos"]) >= n_dh else
            "by the one-hit rule (an earlier batch left %d hit(s) unweighed)" % (n_dh - len(ps["eval_hit_pair_cos"]))),
              d.get("fired") and dv and ((dv.get("copy_hits") or 0) >= 1 if len(ps["eval_hit_pair_cos"]) >= n_dh
                                         else dv.get("fail_closed")), (dv, d.get("summary")))
    except ImportError as e:
        NOTES.append("D28 interop skipped: the autopilot's stream diagnoses do not load (%s)" % e)
    # no index: nothing weighed, the reason recorded (fail closed downstream); a match without its key likewise
    plain = [dict(r, pair_cos=None) for r in hits.values()]
    rec0 = SS.score_eval_hits(plain, SS.CopyScanner())
    check("without a copy scanner index nothing is weighed: every hit unscored with the reason, the record says why",
          rec0["hits"] == 3 and rec0["scored"] == 0 and "no copy scanner index" in (rec0.get("why") or "")
          and all(r["pair_cos"] is None and "no copy scanner index" in r.get("pair_cos_why", "") for r in plain), rec0)
    odd = [dict(hits["p_devbyte"], guard_match={"planted": "no evaluation key"}, pair_cos=None)]
    rec1 = SS.score_eval_hits(odd, scanner)
    check("a match that names no evaluation image is unscored (never a guess)", rec1["scored"] == 0
          and odd[0]["pair_cos"] is None and "neither" in odd[0].get("pair_cos_why", ""), (rec1, odd))
    # the pair: a hit is weighed against every evaluation image within the never-train radius, not only the one the
    # guard names (GuardV2 returns the first match it finds): with the guard's match planted as the evaluation image
    # least like the byte copy of dev, the copy still keeps 1.0 from the dev original within 0 bits of it
    idx = scanner.index
    Xn = np.asarray(idx.Xn, dtype=np.float32)
    pos = {(str(sp), str(k)): j for j, (sp, k) in enumerate(zip(idx.split, idx.eval_key))}
    m0 = hits["p_devbyte"]["guard_match"]
    j0 = pos[(str(m0["split"]), str(m0["key"]))]
    far = min((j for j in range(len(idx.eval_key)) if j != j0), key=lambda j: float(Xn[j] @ Xn[j0]))
    planted = {"split": str(idx.split[far]), "key": str(idx.eval_key[far])}
    alone = [dict(hits["p_devbyte"], guard_match=planted, pair_cos=None)]
    every = [dict(hits["p_devbyte"], guard_match=planted, pair_cos=None)]
    SS.score_eval_hits(alone, scanner)
    rec2 = SS.score_eval_hits(every, scanner, guards=guards)
    check("a hit whose guard match is an unrelated evaluation image (pair cos %s alone) keeps 1.0 from the "
          "evaluation image it copies within the radius (pair_cos_best names it): the guard's match is not weighed "
          "alone" % alone[0].get("pair_cos"), alone[0].get("pair_cos") is not None and alone[0]["pair_cos"] < theta
          and every[0].get("pair_cos") == 1.0 and rec2["scored"] == 1
          and every[0].get("pair_cos_best") == [str(m0["split"]), str(m0["key"])], (alone[0].get("pair_cos"), every))


def _d28(st, extra=None):
    """The autopilot's D28 over a status.json (None when group F does not load)."""
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
        from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS
    except ImportError as e:
        NOTES.append("D28 interop skipped: the autopilot's stream diagnoses do not load (%s)" % e)
        return None
    texts = {"step1_stream/status.json": json.dumps(st)}
    texts.update(extra or {})
    ev = E.from_texts(texts, "x", context={"sid": "none"})
    return DS.by_id(DS.detect(ev, LS.load_domain("weed"), LS.load_thresholds(), only=("D28",)))["D28"]


def test_eval_hit_sidecars(pre, guards, scanner, theta, rows):
    """D28-v2: a committed batch whose batch.json does not weigh its dHash hits
    (made before the amendment; here, the same registry batch admitted without
    a copy scanner) is weighed again by step1_stream eval-hits into
    step1_stream/eval_hits/<batch>.json; batch.json is never rewritten (the
    ledger hash-locks it); write_status folds the sidecar per source, and D28
    judges the source by it. b0000 (no ingest.jsonl) is re-derived from
    admission.jsonl, the v1 pool manifest and dHash cache through GuardV2."""
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    print("D28-v2: the sidecars of Step 1 batches committed before the amendment (step1_stream eval-hits)")
    doc = SS.run_batch(pre, "registry:a_species", lambda: SS.list_registry(pre, ["a_species"]), TV.FakeEmbedder(),
                       guards, None, procs=1)
    bid = doc["batch"]
    led = {e["batch"]: e for e in SS.verify_ledger(pre.ledger)}
    st = SS.write_status(pre)
    ps = st["per_source"]["a_species"]
    n_dh = sum(int(ps.get("decision:%s" % k, 0)) for k in EH.DHASH_HIT_REASONS)
    check("fixture: the batch counts dHash hits that its own record does not weigh: status.json lists it as due for a "
          "sidecar, and D28 reads its source by the one-hit rule", (doc.get("eval_hits") or {}).get("scored") == 0
          and n_dh >= 3 and st["eval_hits"]["due"] == [bid] and SS.check_status(st) == [], (st["eval_hits"], n_dh))
    d = _d28(st)
    if d is not None:
        dv = next(((h.get("verdict") or {}).get("dhash") or {} for h in (d.get("detail") or {}).get("leaks") or []
                   if h["source"] == "a_species"), {})
        check("  (D28: fail closed)", d.get("fired") and dv.get("fail_closed"), d.get("summary"))
    sha_before = C.sha256_file(pre.batch_dir(bid) / "batch.json")
    blind = SS.eval_hits(pre, guards, SS.CopyScanner(), intakes=False)
    check("without a copy scanner index no sidecar is written (it would weigh nothing and use up the batch's "
          "attempt): the batch is 'failed' and stays due", blind["step1"].get(bid, {}).get("status") == "failed"
          and not SS.eval_hits_sidecar_path(pre, bid).exists()
          and json.loads(pre.status.read_text())["eval_hits"]["due"] == [bid], blind["step1"])
    out = SS.eval_hits(pre, guards, scanner, intakes=False)
    side = json.loads(SS.eval_hits_sidecar_path(pre, bid).read_text())
    dev_byte = next((p for p in side["pairs"] if "p_devbyte" in p["key"]), {})
    check("eval-hits writes step1_stream/eval_hits/<batch>.json: every hit re-derived (its ingest row, the image's "
          "sha256, GuardV2's decision again) and weighed by the copy scanner (the byte copy of dev: 1.0)",
          out["step1"].get(bid, {}).get("status") == "written" and side["format"] == EH.SIDECAR_FORMAT
          and side["batch"] == bid and side["batch_json_sha256"] == led[bid]["batch_json_sha256"]
          and side["eval_hits"]["hits"] == n_dh and side["eval_hits"]["scored"] == n_dh
          and dev_byte.get("pair_cos") == 1.0 and (dev_byte.get("match") or {}).get("split") == "dev"
          and side["eval_hits"]["copy_threshold"] == theta, (out, side["eval_hits"]))
    check("  batch.json is untouched (the ledger's hash still holds) and the state verifies",
          C.sha256_file(pre.batch_dir(bid) / "batch.json") == sha_before == led[bid]["batch_json_sha256"]
          and SS.verify_state(pre) == [], SS.verify_state(pre))
    st2 = json.loads(pre.status.read_text())
    ps2 = st2["per_source"]["a_species"]
    check("status.json folds the sidecar in the batch's place: nothing due, the sidecar used, every hit of a_species "
          "with its pair cosine, one_time.eval_hits stamped", st2["eval_hits"]["due"] == []
          and st2["eval_hits"]["sidecars"][bid]["used"] is True and len(ps2.get("eval_hit_pair_cos") or []) == n_dh
          and 1.0 in ps2["eval_hit_pair_cos"] and ps2.get("eval_hit_copy_threshold") == theta
          and isinstance(st2["one_time"].get("eval_hits"), str) and SS.check_status(st2) == [], st2["eval_hits"])
    d = _d28(st2)
    if d is not None:
        dv = next(((h.get("verdict") or {}).get("dhash") or {} for h in (d.get("detail") or {}).get("leaks") or []
                   if h["source"] == "a_species"), {})
        check("the autopilot's D28 now judges a_species by the pair cosines: a leak by the byte copy at or above the "
              "copy threshold, not by the fallback", d.get("fired") and (dv.get("copy_hits") or 0) >= 1
              and not dv.get("fail_closed"), (dv, d.get("summary")))
    again = SS.eval_hits(pre, guards, scanner, intakes=False)
    check("a second run leaves the sidecar alone (one attempt per batch; --force writes it again)",
          again["step1"].get(bid, {}).get("status") == "exists", again["step1"])
    sp = SS.eval_hits_sidecar_path(pre, bid)
    keep = sp.read_text()
    sp.write_text(json.dumps(dict(json.loads(keep), batch_json_sha256="0" * 64)))
    st3 = SS.write_status(pre)
    check("a sidecar made from another batch.json than the one the ledger commits is not used: its hits fall back "
          "to the one-hit rule", st3["eval_hits"]["sidecars"][bid]["used"] is False
          and "another batch.json" in st3["eval_hits"]["sidecars"][bid]["why"]
          and st3["per_source"]["a_species"].get("eval_hit_pair_cos") == [], st3["eval_hits"])
    bad = json.loads(keep)
    bad["eval_hits"]["per_source"]["a_species"]["hits"] = n_dh + 1
    sp.write_text(json.dumps(bad))
    st3 = SS.write_status(pre)
    check("  nor one that weighed another number of hits than the batch counted", not st3["eval_hits"]["sidecars"][
        bid]["used"] and "weighed" in st3["eval_hits"]["sidecars"][bid]["why"], st3["eval_hits"])
    sp.write_text(keep)
    SS.write_status(pre)
    # the sidecar records the sha256 of the batch.json bytes the job read, never the ledger's: a batch.json changed
    # since commit gets no sidecar (it would never be folded), and a sidecar made from one is not folded
    bj = pre.batch_dir(bid) / "batch.json"
    bj_keep = bj.read_bytes()
    bj.write_bytes(bj_keep + b"\n")
    try:
        chg = SS.eval_hits(pre, guards, scanner, bids=[bid], intakes=False, force=True)
        untouched = sp.read_text() == keep
        sha_chg = C.sha256_file(bj)
        side_chg = SS.step1_eval_hits(pre, bid, json.loads(bj.read_bytes()), sha_chg, guards, scanner)
    finally:
        bj.write_bytes(bj_keep)
    check("eval-hits gives no sidecar to a batch whose batch.json no longer hashes to the sha256 the ledger "
          "commits (it would never be folded): the batch is 'failed' and its sidecar is left as it was",
          chg["step1"].get(bid, {}).get("status") == "failed" and "ledger commits" in chg["step1"][bid].get("why", "")
          and untouched, chg["step1"])
    st4 = SS.write_status(pre)
    check("  a sidecar made from a changed batch.json records the sha256 of the bytes read, and write_status does "
          "not fold it", side_chg["batch_json_sha256"] == sha_chg != led[bid]["batch_json_sha256"]
          and st4["eval_hits"]["sidecars"][bid]["used"] is False
          and "another batch.json" in st4["eval_hits"]["sidecars"][bid]["why"], st4["eval_hits"])
    sp.write_text(keep)
    SS.write_status(pre)
    # the CLI's scope: --intake X alone weighs that intake batch and no Step 1 batch (the help text's promise)

    def scope(*argv):
        return SS.eval_hits_scope(SS.build_parser().parse_args(["eval-hits"] + list(argv)))
    check("the CLI eval-hits --intake X (no --batch-id) weighs only that intake batch, no Step 1 batch; --batch-id "
          "only those Step 1 batches, no intake batch; both, both; neither, every batch that needs it",
          scope("--intake", "i1") == ([], ["i1"]) and scope("--batch-id", "b1") == (["b1"], False)
          and scope("--intake", "i1", "--batch-id", "b1", "--batch-id", "b2") == (["b1", "b2"], ["i1"])
          and scope() == (None, None), [scope("--intake", "i1"), scope("--batch-id", "b1"), scope()])
    none = SS.eval_hits(pre, guards, scanner, bids=[], intakes=False, force=True)
    check("  and eval_hits with bids [] touches no Step 1 batch (force or not)", none["step1"] == {}
          and sp.read_text() == keep, none["step1"])
    # b0000: no ingest.jsonl; its hits come back through admission.jsonl, the v1 pool and dHash cache, and GuardV2
    b0 = pre.batch_dir("b0000")
    adm = b0 / "admission.jsonl"
    adm_keep = adm.read_bytes()
    vf = SS.v1_files()
    pool = C.read_manifest(vf["pool"])
    meta = V._read_jsonl(vf["pool_meta"])
    dev0 = rows["dev"][0]
    copy_img = TMP / "eh_v1" / "v1_devcopy.jpg"
    copy_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dev0["image"], copy_img)
    tmpl, mtmpl = pool[0], next(m for m in meta if m["key"] == pool[0]["key"])
    pool2 = pool + [dict(tmpl, key="a_species__v1_devcopy", image=str(copy_img), sha256=C.sha256_file(copy_img),
                         source="a_species")]
    meta2 = meta + [dict(mtmpl, key="a_species__v1_devcopy", dhash=int(C.dhash(copy_img)))]
    (TMP / "eh_v1" / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool2))
    (TMP / "eh_v1" / "pool_meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in meta2))
    with open(adm, "a") as fh:
        fh.write(json.dumps({"key": "a_species__v1_devcopy", "source": "a_species", "refusal": "near_eval_v2"}) + "\n")
        fh.write(json.dumps({"key": pool[1]["key"], "source": pool[1]["source"],
                             "refusal": "near_eval_variant"}) + "\n")
    real = SS.v1_files
    SS.v1_files = lambda: dict(vf, pool=TMP / "eh_v1" / "pool.jsonl", pool_meta=TMP / "eh_v1" / "pool_meta.jsonl")
    try:
        d0 = dict(json.loads((b0 / "batch.json").read_text()),
                  per_source_decisions={"a_species": {"near_eval_v2": 1}, pool[1]["source"]: {"near_eval_variant": 1}})
        side0 = SS.step1_eval_hits(pre, "b0000", d0, "f" * 64, guards, scanner)
    finally:
        SS.v1_files = real
        adm.write_bytes(adm_keep)
    p0 = {p["key"]: p for p in side0["pairs"]}
    hit0, odd0 = p0.get("v1|a_species__v1_devcopy") or {}, p0.get("v1|%s" % pool[1]["key"]) or {}
    check("b0000's sidecar: a v1 pool image that copies a dev image is re-derived from admission.jsonl, the v1 pool "
          "manifest and dHash cache, decided again by GuardV2 (near_eval_v2, the dev key) and weighed (1.0)",
          hit0.get("pair_cos") == 1.0 and (hit0.get("match") or {}).get("key") == dev0["key"]
          and side0["batch"] == "b0000" and side0["batch_kind"] == SS.KIND_BACKFILL, hit0)
    check("  a row GuardV2 no longer refuses as recorded is left unweighed with the reason (never guessed)",
          odd0.get("pair_cos") is None and "GuardV2 now decides" in (odd0.get("pair_cos_why") or ""), odd0)
    SS.eval_hits_sidecar_path(pre, "b0000").unlink()


def test_exif_masked(guards, rows, lay):
    print("the masked copy of an EXIF-rotated original is caught through a rotation variant")
    dev = rows["dev"][4]
    src = TMP / "exif" / "rotated.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    ex = Image.Exif()
    ex[0x0112] = 6
    with Image.open(dev["image"]) as im:
        im.convert("RGB").save(src, format="JPEG", quality=95, exif=ex.tobytes())
    reason_u, _m, _h = guards.check(src)
    rec = MK.mask_except(src, [(0.05, 0.05, 0.06, 0.06)], [], TMP / "exif" / "out", key="rot")
    reason, match, h = guards.check(rec["path"])
    hd = C.dhash(dev["image"])
    check("fixture: the stored pixels are the dev image's (unmasked: near_eval_v2), the masked PNG is transposed "
          "(its own dHash > 6 bits away)", reason_u == "near_eval_v2" and bits(h, hd) > 6, (reason_u, bits(h, hd)))
    check("the masked PNG is refused through a rotation variant", reason == "near_eval_variant"
          and match.get("variant") in ("rot90", "rot270", "transpose", "transverse"), (reason, match))


def test_refusals(lay):
    print("registry refusals")
    regp = REPO / "results" / "framework" / "dataset_registry.json"
    saved = regp.read_text()
    reg = json.loads(saved)
    fresh(DS / "z_new" / "images" / "z.png", [0, 12, 12], [0, 1, 1])
    reg["datasets"]["z_new"] = {"local_path": str(DS / "z_new"), "annotation": "bbox", "class_names": ["Waterhemp", "corn"]}
    reg["datasets"]["c_nonames"]["class_names"] = ["corn"]
    regp.write_text(json.dumps(reg))
    try:
        items, ref, _l = SS.list_registry(lay)
    finally:
        regp.write_text(saved)
    check("a slug absent from the v1 pool is refused (it must come through collect intake)",
          ref.get("z_new") == "not_in_v1_pool" and not any(i["source"] == "z_new" for i in items), ref)
    check("a slug whose class join changed since v1 is refused (a rejoin is needed)",
          ref.get("c_nonames") == "join_changed", ref)
    real = TV.M._build_canonical_class_map
    TV.M._build_canonical_class_map = TV.legacy_class_map
    try:
        check("a pre-v3.60.0 (legacy-label) mega_trainer join refuses the listing (a stale package copy)",
              raises(lambda: SS.list_registry(lay), text="not the v3.60.0 species join"))
    finally:
        TV.M._build_canonical_class_map = real
    check("the leave-4-out copies and never-train slugs are never read",
          ref.get("cottonweed_sp8") == "calibration_only" and ref.get("cottonweeddet12") == "not_in_v1_pool", ref)


# ------------------------------------------------------------------ holds
def test_holds(lay, scanner, theta):
    print("holds: licence, h6_scan (deadline), funnel_F9 (domain dev)")
    cards = C.INC_DIR / "funnel" / "cards" / "index.json"
    cards.parent.mkdir(parents=True, exist_ok=True)
    cards.write_text(json.dumps({"cards": {"b_other": [{"status": 200, "licence": "CC BY-NC 4.0", "url": "u",
                                                        "fetched_utc": "t"}],
                                           "k_mixed": [{"status": 200, "licence": "CC BY 4.0", "url": "u",
                                                        "fetched_utc": "t"}],
                                           "c_nonames": [{"status": 200, "licence": "Other (specified in description)",
                                                          "url": "https://www.kaggle.com/datasets/mit/cc-by-x",
                                                          "fetched_utc": "t"}]}}))
    SS.serve_holds(lay, SS.CopyScanner(), kinds=["h6_scan"])
    q = {r["key"]: r for r in SS.load_queue(lay)}
    check("serve-holds --hold h6_scan serves that hold only: a licence the cards now record is not released by it",
          all("licence" in r["holds"] for r in q.values() if r["source"] == "b_other"))
    SS.serve_holds(lay, SS.CopyScanner())
    q = {r["key"]: r for r in SS.load_queue(lay)}
    check("licence: released for the source the funnel's cards now license (NC -> research_only); a card that "
          "names no licence ('Other (specified in description)', its URL naming 'mit' and 'cc-by') keeps it held",
          all("licence" not in r["holds"] and r["licence"].startswith("CC BY-NC") and r["research_only"]
              for r in q.values() if r["source"] == "b_other")
          and all("licence" in r["holds"] for r in q.values() if r["source"] == "c_nonames"))
    sfx = " (fetched by L11a from https://github.com/mit-lab/cc-by-weeds, 2026-09-01)"
    got = {t: SS.licence_state(t) for t in ("unknown" + sfx, "Other (specified in description)" + sfx, "Private",
                                            "All Rights Reserved", "", None, "see description", "custom terms",
                                            "Weed dataset terms v2" + sfx, "All rights reserved (MIT Press)",
                                            "Other (attribution required, see description)",
                                            "CC BY 4.0" + sfx, "CC BY-NC-SA 4.0", "CC BY-ND 4.0", "CC0: Public Domain",
                                            "MIT", "ODbL", "https://creativecommons.org/licenses/by-nc/4.0/")}
    held = [t for t, (lic, _ro) in got.items() if lic is None]
    check("licence text: unknown, other, private, all rights reserved, empty and unrecognised texts stay held "
          "(P6), whatever the card URL says; CC, CC0, MIT and ODbL resolve; NC and ND are research-only",
          set(held) == {"unknown" + sfx, "Other (specified in description)" + sfx, "Private", "All Rights Reserved",
                        "", None, "see description", "custom terms", "Weed dataset terms v2" + sfx,
                        "All rights reserved (MIT Press)", "Other (attribution required, see description)"}
          and [got[t][1] for t in ("CC BY 4.0" + sfx, "CC BY-NC-SA 4.0", "CC BY-ND 4.0", "CC0: Public Domain", "MIT",
                                   "ODbL", "https://creativecommons.org/licenses/by-nc/4.0/")]
          == [False, True, True, False, False, False, True], got)
    # the stream's own calibration only: same-lab rows wait for the deadline
    ownp = C.INC_DIR / "splits" / "v2" / "leak" / "leak_calibration.json"
    write_calibration(ownp, theta)
    own = SS.load_scanner(lay, funnel_path=TMP / "none.json", embedder=HueEmbedder(), procs=1)
    check("the stream's own calibration is found where inc2.splits writes it", own.cal["own"] is not None
          and own.cal["funnel"] is None and own.cal["own"]["path"] == str(ownp))
    SS.serve_holds(lay, own)
    q = {r["key"]: r for r in SS.load_queue(lay) if r["batch"] == "b0000"}
    check("before the deadline, the stream's own detector releases the rows that are not same-lab only",
          all("h6_scan" not in r["holds"] for r in q.values() if r["h6_reason"] == "not_scanned")
          and all("h6_scan" in r["holds"] for r in q.values() if r["h6_reason"] == "lab_evidence")
          and any(r["h6_reason"] == "lab_evidence" for r in q.values()))
    st = SS.write_status(lay)
    check("status: no hold is past its deadline yet", st["holds_past_deadline"] == {}, st["holds_past_deadline"])
    real = SS.CLOCK
    SS.CLOCK = lambda: real() + datetime.timedelta(days=22)
    try:
        st = SS.write_status(lay)
        check("22 days on, the funnel_F9 holds are past their deadline and listed for a person (R3)",
              st["holds_past_deadline"].get("funnel_F9") == 3, st["holds_past_deadline"])
        SS.serve_holds(lay, own)
        q = {r["key"]: r for r in SS.load_queue(lay) if r["batch"] == "b0000"}
        check("after the deadline, the stream's own detector serves the same-lab rows too",
              all("h6_scan" not in r["holds"] for r in q.values()))
        check("a funnel_F9 hold past its deadline is never released by the platform",
              all("funnel_F9" in r["holds"] for r in q.values() if r["admission"] == MK.MASKED))
    finally:
        SS.CLOCK = real
    ddp = C.INC_DIR / "step1_r1" / "domain_dev.jsonl"
    pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    C.write_manifest(ddp, [pool["k_mixed__k_veto1"]])
    SS.serve_holds(lay, SS.CopyScanner())
    q = {r["key"]: r for r in SS.load_queue(lay)}
    check("F9 wrote domain_dev.jsonl: its image is refused for good, the other masked rows are released",
          q["k_mixed__k_veto1"]["refused"] == "domain_dev"
          and all("funnel_F9" not in q[k]["holds"] for k in ("k_mixed__k_veto2", "a_species__a_small")))
    led = SS.verify_ledger(lay.root / "ledger" / "holds.jsonl")
    check("every serve run is on its own hash-chained ledger", len(led) == 5 and led[-1]["refused"] == {"domain_dev": 1},
          [e.get("refused") for e in led])
    ok = sorted(r["key"] for r in q.values() if r["eligible_step1"])
    check("rows with no hold left, evidenced and holding a verified target box are eligible (the released masked "
          "row); OtherPlant-only rows never are", "k_mixed__k_veto2" in ok and all(
              not q[k]["holds"] and not q[k]["refused"] and sum(q[k]["species_boxes"]) for k in ok)
          and not any(q[k]["kind"] == "other" for k in ok), ok)


# ------------------------------------------------------------------ pins
def test_pins(lay, lock, guards):
    print("pins refuse before anything is written")
    a = DS / "c_nonames"
    fresh(a / "images" / "c_pin.png", [12, 12, 12], [0, 0, 0])

    def state():
        return (SS._batch_ids(lay), lay.queue.read_bytes(), lay.events.read_bytes())
    before = state()
    run = lambda emb: SS.run_batch(lay, "registry:c_nonames", lambda: SS.list_registry(lay, ["c_nonames"]),  # noqa
                                   emb, guards, None, procs=1)
    p = lay.verifier_dir / "verifier.npz"
    data = p.read_bytes()
    p.write_bytes(data[:-3] + bytes([data[-3] ^ 1]) + data[-2:])
    try:
        check("an altered verifier.npz refuses", raises(lambda: run(TV.FakeEmbedder()), text="frozen verifier file"))
    finally:
        p.write_bytes(data)
    check("another embedder refuses", raises(lambda: run(OtherEmbedder()), text="is not the pinned"))
    nt = lock.parent / "nevertrain_dhash.json"
    nts = nt.read_text()
    nt.write_text(nts + " ")
    try:
        check("a changed v2 never-train index refuses", raises(lambda: run(TV.FakeEmbedder()), text="changed since the v2 lock"))
    finally:
        nt.write_text(nts)
    ls = lock.read_text()
    lk = json.loads(ls)
    nt.write_text(nts.replace('"complete": true', '"complete": true, "note": "relocked"'))
    lk["nevertrain_sha256"] = C.sha256_file(nt)
    lock.write_text(json.dumps(lk))
    try:
        check("a re-locked v2 index (another sha in a new LOCK) refuses: a new splits version is a new stream version",
              raises(lambda: run(TV.FakeEmbedder()), text="changed since bootstrap"))
    finally:
        lock.write_text(ls)
        nt.write_text(nts)
    check("canary drift refuses", raises(lambda: run(DriftEmbedder()), text="canary drift"))
    check("... and nothing was written by any of them", state() == before)
    doc = run(TV.FakeEmbedder())
    check("with every pin intact the batch runs", doc is not None and doc["canary"]["min_cos"] >= 0.999
          and doc["canary"]["verdicts_same"] == doc["canary"]["n"], doc and doc.get("canary"))


# ------------------------------------------------------------------ state
def test_state(lay, guards):
    print("state: rerun no-op, kill and resume, contiguous crop ids, ledger chain")
    n = len(SS.verify_ledger(lay.ledger))
    qb = lay.queue.read_bytes()
    again = SS.run_batch(lay, "registry:c_nonames", lambda: SS.list_registry(lay, ["c_nonames"]), TV.FakeEmbedder(),
                         guards, None, procs=1)
    check("rerunning a committed batch's argv is a no-op", again is None and len(SS.verify_ledger(lay.ledger)) == n
          and lay.queue.read_bytes() == qb)
    last = SS.verify_ledger(lay.ledger)[-1]["batch"]
    SS.finish_commit(lay, last)
    check("finishing a committed batch again changes nothing", len(SS.verify_ledger(lay.ledger)) == n
          and lay.queue.read_bytes() == qb)
    a = DS / "c_nonames"
    for i in range(3):
        fresh(a / "images" / ("c_kill_%d.png" % i), [12, 12, 12], [0, 0, 0])
    spec = "registry:c_nonames"
    fn = lambda: SS.list_registry(lay, ["c_nonames"])  # noqa: E731
    killed = False
    try:
        SS.run_batch(lay, spec, fn, KillEmbedder(kill_after=2), guards, None, procs=1, chunk=1)   # canary, 1 chunk
    except Kill:
        killed = True
    ip = SS._in_progress(lay)
    check("a batch killed while embedding is left in progress (no batch.json)", killed and ip is not None
          and not (lay.batch_dir(ip) / "batch.json").exists() and len(SS.verify_ledger(lay.ledger)) == n)
    check("another input is refused while it is in progress",
          raises(lambda: SS.run_batch(lay, "registry:b_other", lambda: SS.list_registry(lay, ["b_other"]),
                                      TV.FakeEmbedder(), guards, None, procs=1), text="is in progress"))
    fe = TV.FakeEmbedder()
    doc = SS.run_batch(lay, spec, fn, fe, guards, None, procs=1, chunk=1)
    total = sum(1 for r in batch_rows(lay, doc["batch"]) if r["decision"] == "pool")
    check("rerunning its argv resumes it: the finished chunk is reused (%d of %d images embedded again)"
          % (fe.calls - 1, total), doc["batch"] == ip and 0 < fe.calls - 1 < total, (fe.calls, total))
    probs = SS.verify_state(lay)
    led = SS.verify_ledger(lay.ledger)
    rngs = [(e["crop_offset"], e["crop_offset"] + e["crops"]) for e in led]
    check("global crop ids are contiguous and disjoint over every batch", probs == []
          and all(rngs[i][1] == rngs[i + 1][0] for i in range(len(rngs) - 1)) and rngs[0][0] == V.Crops().n,
          (probs, rngs))
    qp = TMP / "queue_copy.jsonl"
    qp.write_bytes(lay.queue.read_bytes() + b'{"key": "half')
    n_rows = len(SS._read_jsonl(lay.queue))
    check("a reader skips a last line whose append is still in progress", len(SS._read_jsonl(qp)) == n_rows)
    SS._append_jsonl(qp, [{"key": "whole"}])
    check("the next append drops the partial line first", [r["key"] for r in SS._read_jsonl(qp)][-1] == "whole"
          and len(SS._read_jsonl(qp)) == n_rows + 1 and qp.read_bytes().endswith(b'{"key":"whole"}\n'))
    cp = TMP / "ledger_copy.jsonl"
    data = bytearray(lay.ledger.read_bytes())
    i = data.index(b'"crops":') + 8
    data[i:i + 1] = b"9" if data[i:i + 1] != b"9" else b"8"
    cp.write_bytes(bytes(data))
    check("an edited ledger breaks the chain and is refused", raises(lambda: SS.verify_ledger(cp), text="edited")
          or raises(lambda: SS.verify_ledger(cp), text="prev_sha256"))


def test_writer_lock(lay):
    print("one writer across nodes: flock plus an O_EXCL owner file")
    import errno
    import fcntl
    import socket
    owner = lay.root / ".lock.owner"

    def plant(**kw):
        doc = dict({"token": "x", "host": "othernode", "pid": 4242, "job": None, "t": __import__("time").time(),
                    "utc": "t"}, **kw)
        owner.write_text(json.dumps(doc))

    def acquire():
        with SS.writer_lock(lay):
            return True
    check("a writer leaves no owner file behind", acquire() and not owner.exists())
    plant()
    check("a live writer on another node (its owner file) refuses: node-local flock alone would let both write",
          raises(acquire, text="names a live writer") and owner.exists())
    plant(t=__import__("time").time() - 10 * 3600)
    check("an owner file older than any job runs is taken over", acquire() and not owner.exists())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    plant(host=socket.gethostname(), pid=p.pid)
    check("an owner file whose process on this host has exited is taken over", acquire() and not owner.exists())
    fake = TMP / "fakebin"
    fake.mkdir(exist_ok=True)
    sq = fake / "squeue"
    path0 = os.environ["PATH"]
    os.environ["PATH"] = "%s:%s" % (fake, path0)
    try:
        sq.write_text("#!/bin/sh\nexit 0\n")
        sq.chmod(0o755)
        plant(job="4242")
        check("an owner whose Slurm job squeue no longer lists is taken over", acquire() and not owner.exists())
        sq.write_text("#!/bin/sh\necho RUNNING\n")
        plant(job="4242")
        check("... and one squeue lists as RUNNING refuses", raises(acquire, text="names a live writer"))
    finally:
        os.environ["PATH"] = path0
        owner.unlink()
    real = fcntl.flock

    def no_flock(fd, op):
        if op & fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, "No locks available")
        return real(fd, op)
    fcntl.flock = no_flock
    try:
        with SS.writer_lock(lay):
            inner = raises(acquire, text="names a live writer")
        check("a mount without flock (ENOLCK): the writer runs, and the owner file alone refuses a second one",
              inner and not owner.exists())
    finally:
        fcntl.flock = real
    with SS.writer_lock(lay):
        inner = raises(acquire, text="held by another writer")
    check("with flock, a second writer on the node is refused by the lock itself", inner)


# ------------------------------------------------------------- knowntruth
def test_knowntruth(lay, guards):
    print("knowntruth: the frozen verifier on base copies of expert labels (R1)")
    cp = copy_layout(lay, "stream_kt_empty")
    check("fixture: no knowntruth has run yet", SS.write_status(cp)["one_time"]["knowntruth"] is None)
    none = SS.run_knowntruth(cp, TV.FakeEmbedder(), guards, sets=("tsw23",), procs=1)
    check("a knowntruth run with nothing to measure commits no batch but is recorded as run (one_time), so the "
          "platform does not submit it again", none is None and isinstance(SS.write_status(cp)["one_time"]["knowntruth"],
                                                                            str))
    qb = lay.queue.read_bytes()
    doc = SS.run_knowntruth(lay, TV.FakeEmbedder(), guards, sets=("tsw22",), procs=1)
    kt = doc["knowntruth"]
    rows = batch_rows(lay, doc["batch"])
    check("the two harvested copies of tsw images (v1 near-eval as ood22) are read and measured; nothing is queued",
          sorted(r["stem"] for r in rows) == ["tcopy_0", "tcopy_1"] and all(r["decision"] == "knowntruth" for r in rows)
          and doc["queue_rows"] == 0 and lay.queue.read_bytes() == qb, [(r["stem"], r["decision"]) for r in rows])
    t = kt["per_set"]["tsw22"]
    check("tsw22: 6 boxes matched, 5 labels right, the wrong one not verified; the Wilson bound is recorded",
          t["boxes"] == 6 and t["label_correct_rate"] == round(5 / 6, 4) and t["verified_precision"] == 1.0
          and t["verified_precision_wilson_lb"] is not None and t["verified_precision_wilson_lb"] < 1.0, t)
    check("a rerun measures nothing new", SS.run_knowntruth(lay, TV.FakeEmbedder(), guards, sets=("tsw22",)) is None)
    st = SS.write_status(lay)
    check("status carries the reading per batch; fewer than 30 matched verified boxes never fire the refit trigger",
          doc["batch"] in st["knowntruth"] and st["refit_triggers"]["precision_below"] == [], st["refit_triggers"])
    lbs = [SS.wilson_lb(k, n) for k, n in ((30, 30), (100, 100), (99, 100), (0, 0))]
    check("Wilson lower bound (z 1.96): 30/30 -> 0.8865, 100/100 -> 0.9630, 99/100 -> 0.9455, none without boxes",
          lbs[:3] == [0.886483, 0.963005, 0.945512] and lbs[3] is None, lbs)
    ps = [SS.refit_precision_p(k, n) for k, n in ((284, 284), (277, 284), (276, 284), (None, 50), (5, 0))]
    check("refit trigger is a binomial test against 0.99: 284/284 and missing counts -> 1.0, 7 errors in 284 -> "
          "0.025 (silent), 8 -> 0.0084 (fires)",
          ps[0] == 1.0 and ps[3] == 1.0 and ps[4] == 1.0 and ps[1] > SS.REFIT_ALPHA > ps[2]
          and abs(ps[2] - 0.008444) < 1e-5, ps)


def test_empty_batch(lay, guards):
    print("a batch whose only new path is dropped before hashing")
    img = DS / "c_nonames" / "images"
    TV.paint(img / "c_nolabel_new.png", TV.layout3([12, 12, 12]), texture=False, fmt="png")
    doc = SS.run_batch(lay, "registry:c_nonames", lambda: SS.list_registry(lay, ["c_nonames"]), TV.FakeEmbedder(),
                       guards, None, procs=1)
    check("it commits with no crops and no queue rows, and the state verifies",
          doc["crops"] == 0 and doc["queue_rows"] == 0 and doc["decisions"] == {"no_label": 1}
          and SS.verify_state(lay) == [], (doc and doc.get("decisions"),
                                           [(r["stem"], r["decision"]) for r in batch_rows(lay, doc["batch"])]))


# ----------------------------------------------------------------- rejoin
def test_rejoin(lay, guards):
    print("rejoin from a recorded resolution")
    check("no recorded resolution: refused", raises(lambda: SS.rejoin(lay, "b_other", guards), text="no recorded"))
    nsp = C.INC_DIR / "funnel" / "name_status_v2.json"
    nsp.write_text(json.dumps({"names": [{"source": "b_other", "src_id": "2", "name": "weed",
                                          "status_v2": "target_synonym", "via": "override",
                                          "taxon": "Senna obtusifolia"},
                                         {"source": "b_other", "src_id": "3", "name": "pigweed",
                                          "status_v2": "target_synonym", "via": "join", "taxon": "Amaranthus"}]}))
    before = {r["key"]: r for r in SS.load_queue(lay)}
    doc = SS.rejoin(lay, "b_other", guards)
    q = {r["key"]: r for r in SS.load_queue(lay)}
    new = q.get("b_other__train__b_conflict__rj1")
    check("only the resolution recover itself applies is used (via override or scientific): id 2 -> Sicklepod",
          doc["map"] == {"2": 9}, doc["map"])
    check("the image whose 'weed' box was a Sicklepod conflict is now admitted whole as Sicklepod, superseding the "
          "old key", new is not None and new["admission"] == MK.WHOLE and new["species_boxes"][9] == 1
          and new["supersedes"] == "b_other__train__b_conflict", new)
    olds = [k for k, r in before.items() if r["source"] == "b_other" and not r.get("supersedes")]
    check("every old queued row of the slug that the map relabels is refused as superseded",
          olds and all(q[k]["refused"] == "superseded:rejoin_v1" for k in olds
                       if any(s[0] == 2 for s in json.loads(json.dumps(
                           [m for m in V._read_jsonl(V.POOL_META) if m["key"] == k][0]["src"])))), olds)
    ov = SS.Index(lay, "overrides", {}).data
    check("the override is versioned in the index", ov["b_other"]["version"] == 1 and ov["b_other"]["map"] == {"2": 9})
    check("the same resolution again is a no-op", SS.rejoin(lay, "b_other", guards) is None)
    fresh(DS / "b_other" / "train" / "images" / "b_new.png", [9, 12, 12], [2, 0, 0])
    items, _r, _l = SS.list_registry(lay, ["b_other"])
    it = [i for i in items if i["stem"] == "b_new"]
    check("later registry batches of the slug read it through the override", it and it[0]["boxes"][0][0] == 9
          and it[0]["join_version"] == 1, it)


class RefuseGuard:
    """GuardV2 with one planted refusal: the dHash of a chosen image is
    refused as a rotated copy of a test image (the check v1 never ran)."""

    def __init__(self, inner, bad):
        self.inner, self.bad = inner, set(int(h) for h in bad)

    def check(self, h, variants=None):
        if int(h) in self.bad:
            return "near_eval_variant", {"split": "test", "key": "planted", "bits": 0, "variant": "rot90"}
        return self.inner.check(h, variants)


def copy_layout(lay, name):
    dst = TMP / name
    shutil.copytree(lay.root, dst)
    return SS.Layout(dst, C.INC_DIR)


def live_by_origin(layout):
    out = collections.defaultdict(list)
    for r in SS.load_queue(layout):
        if not r["refused"]:
            out[r.get("supersedes") or r["key"]].append(r["key"])
    return out


def test_rejoin_guards(lay, guards, sel):
    print("rejoin: base B never queued, GuardV2 on every relabelled image, one live row per image, holds carried")
    nsp = C.INC_DIR / "funnel" / "name_status_v2.json"
    saved = nsp.read_text()
    meta = {m["key"]: m for m in V._read_jsonl(V.POOL_META)}
    base_b = {r["key"] for r in sel}
    try:
        # (1) a_species id 5 (Lambsquarters, OtherPlant) -> Sicklepod: every base B image of the slug has the id
        lay2 = copy_layout(lay, "stream_rj")
        ns = json.loads(saved)
        ns["names"].append({"source": "a_species", "src_id": "5", "name": "Lambsquarters", "status_v2": "target_synonym",
                            "via": "override", "taxon": "Senna obtusifolia"})
        nsp.write_text(json.dumps(ns))
        bad = meta["a_species__a_bad"]
        g2 = SS.Guards(RefuseGuard(guards.guard, [bad["dhash"]]), guards.variants_fn, record=dict(guards.record))
        doc = SS.rejoin(lay2, "a_species", g2)
        q = {r["key"]: r for r in SS.load_queue(lay2)}
        adm = {r["supersedes"]: r for r in batch_rows(lay2, doc["batch"], "admission.jsonl")}
        base_here = sorted(k for k, m in meta.items() if m["source"] == "a_species" and k in base_b)
        check("fixture: base B holds a_species images that the map relabels", len(base_here) >= 2, base_here)
        check("base B's images (base_v2 and the L-5 drops) are never relabelled into the queue",
              not any(r.get("supersedes") in base_b for r in q.values()) and not (set(adm) & base_b)
              and doc["base_b_left_out"] == len(base_here), (doc["base_b_left_out"], sorted(set(adm) & base_b)))
        check("GuardV2 runs on every relabelled unmasked image: the planted rotated test copy is refused, not queued",
              adm["a_species__a_bad"]["refusal"] == "near_eval_variant"
              and adm["a_species__a_bad"]["admission"] == MK.NOT_ADMITTED
              and "a_species__a_bad__rj1" not in q, adm.get("a_species__a_bad"))
        rj = [r for r in q.values() if r["key"].endswith("__rj1") and not r["refused"]]
        rv1 = [r for r in rj if r["input"].startswith("rejoin_v1:v1")]
        check("rows relabelled from the v1 pool hold funnel_F9 (P8) and h6_scan (a new row is scanned again); "
              "rows relabelled from the stream's own batches hold h6_scan only",
              rv1 and all({"funnel_F9", "h6_scan"} <= set(r["holds"]) for r in rv1)
              and all("funnel_F9" not in r["holds"] and "h6_scan" in r["holds"] for r in rj if r not in rv1),
              [(r["key"], r["input"], r["holds"]) for r in rj])
        # (2) a second resolution for b_other: every rejoin_v1 row is superseded, one live row per image
        lay3 = copy_layout(lay, "stream_rj2")
        before = live_by_origin(lay3)
        check("fixture: after rejoin v1 of b_other, each image has one live row",
              all(len(v) == 1 for v in before.values()) and any(k.endswith("__rj1") for v in before.values() for k in v))
        ns = json.loads(saved)
        ns["names"].append({"source": "b_other", "src_id": "3", "name": "pigweed", "status_v2": "target_synonym",
                            "via": "override", "taxon": "Amaranthus palmeri"})
        nsp.write_text(json.dumps(ns))
        doc2 = SS.rejoin(lay3, "b_other", guards)
        after = live_by_origin(lay3)
        q3 = {r["key"]: r for r in SS.load_queue(lay3)}
        check("rejoin v2 supersedes the rejoin v1 rows too: no image is queued twice",
              doc2["version"] == 2 and all(len(v) <= 1 for v in after.values())
              and all(q3[k]["refused"] == "superseded:rejoin_v2" for v in before.values() for k in v
                      if k.endswith("__rj1") and (k[:-5] + "__rj2") in q3),
              {k: v for k, v in after.items() if len(v) > 1})
        # (3) a rejoin killed half-way is resumed by its own argv and blocks nothing else in the meantime
        lay4 = copy_layout(lay, "stream_rj3")
        ns = json.loads(saved)
        ns["names"].append({"source": "b_other", "src_id": "1", "name": "Giant ragweed", "status_v2": "target_synonym",
                            "via": "override", "taxon": "Ambrosia artemisiifolia"})
        nsp.write_text(json.dumps(ns))
        real = SS._place_rows

        def boom(*a, **k):
            raise Kill()
        SS._place_rows = boom
        try:
            SS.rejoin(lay4, "b_other", guards)
            killed = False
        except Kill:
            killed = True
        finally:
            SS._place_rows = real
        ip = SS._in_progress(lay4)
        check("a rejoin killed half-way is left in progress under its own plan", killed and ip is not None
              and json.loads((lay4.batch_dir(ip) / "plan.json").read_text())["spec"] == "rejoin:b_other")
        check("... another input is told to rerun that argv",
              raises(lambda: SS.run_batch(lay4, "registry:c_nonames", lambda: SS.list_registry(lay4, ["c_nonames"]),
                                          TV.FakeEmbedder(), guards, None, procs=1), text="rerun its own argv"))
        doc4 = SS.rejoin(lay4, "b_other", guards)
        check("... and rerunning the rejoin finishes it in the same batch", doc4["batch"] == ip
              and SS._in_progress(lay4) is None and SS.verify_state(lay4) == [], SS.verify_state(lay4))
        (lay4.batches / "b0090").mkdir()
        check("a batch directory a job left before planning (no plan.json) blocks nothing",
              SS._in_progress(lay4) is None and SS.run_batch(lay4, "registry:c_nonames",
                                                              lambda: SS.list_registry(lay4, ["c_nonames"]),
                                                              TV.FakeEmbedder(), guards, None, procs=1) is None)
    finally:
        nsp.write_text(saved)


# ----------------------------------------------------------------- intake
def test_intake(lay, guards):
    print("an intake batch: INC ids, unmapped boxes masked, provenance-cleared sources, checksums")
    d = C.INC_DIR / "intake" / "tb1"
    rows = []

    def add(name, classes, labels, source, licence="CC BY 4.0", sha=None, cap="tray1"):
        p = d / "images" / (name + ".png")
        bx = TV.layout3(classes)
        TV.paint(p, bx, texture=False, fmt="png")
        lp = d / "labels" / (name + ".txt")
        TV.yolo(lp, [(c,) + b[1:] for c, b in zip(labels, bx)])
        rows.append({"key": "%s__%s" % (source, name), "image": str(p), "sha256": sha or C.sha256_file(p),
                     "label": str(lp), "label_sha256": C.sha256_file(lp), "source": source, "licence": licence,
                     "research_only": False, "lab_group": None, "capture_group": cap, "dhash": C.dhash(p)})
    add("i_ok", [0, 12, 12], [0, 12, 12], "src_rf")
    add("i_unmapped", [3, 12, 5], [3, 12, 13], "src_rf")
    add("i_badsha", [4, 12, 12], [4, 12, 12], "src_rf", sha="0" * 64)
    add("i_cleared", [2, 12, 12], [2, 12, 12], "src_tum", cap="tray7")
    add("i_nolic", [7, 12, 12], [7, 12, 12], "src_tum", licence="unresolved")
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "sources.json").write_text(json.dumps({"sources": {"src_rf": {"provenance_cleared": False},
                                                            "src_tum": {"provenance_cleared": True,
                                                                        "lab_group": "TUM"}}}))
    check("an incomplete intake batch (no summary.json) is refused",
          raises(lambda: SS.list_intake(lay, "tb1"), text="not complete"))
    (d / "summary.json").write_text(json.dumps({"format": "collect-intake-summary/1"}))
    ev = SS.Index(lay, "evidence", {})
    doc = SS.run_batch(lay, "intake:tb1", lambda: SS.list_intake(lay, "tb1"), TV.FakeEmbedder(), guards, None,
                       procs=1, kind=SS.KIND_INTAKE)
    dec = {r["intake_key"]: r["decision"] for r in batch_rows(lay, doc["batch"])}
    check("a row whose image does not hash to its declared sha256 is refused", dec["src_rf__i_badsha"] == "sha_mismatch")
    q = {r["key"]: r for r in SS.load_queue(lay) if r["batch"] == doc["batch"]}
    um = q.get("src_rf__i_unmapped")
    check("an unmapped box (13) is masked and never reaches the training label",
          um is not None and um["admission"] == MK.MASKED and [b[0] for b in C.read_yolo(um["label"])] == [3, 12], um)
    check("intake keys are the queue keys; capture groups are carried", q["src_tum__i_cleared"]["capture_group"] == "tray7"
          and set(q) == {"src_rf__i_ok", "src_rf__i_unmapped", "src_tum__i_cleared", "src_tum__i_nolic"}, sorted(q))
    check("a provenance-cleared source is not held h6_scan; the others are (no scan ran)",
          "h6_scan" not in q["src_tum__i_cleared"]["holds"] and "h6_scan" in q["src_rf__i_ok"]["holds"])
    check("the intake licence is the row's: recorded, or held when unresolved",
          q["src_tum__i_cleared"]["licence"] == "CC BY 4.0" and q["src_tum__i_nolic"]["holds"] == ["licence"])
    ev2 = SS.Index(lay, "evidence", {}).data
    check("the evidence index counts the new source's verified boxes; its rows are evidenced",
          ev2.get("src_tum", 0) >= 2 and not ev.data.get("src_tum") and q["src_tum__i_cleared"]["evidenced"])
    check("the intake batch is fully processed (a rerun has nothing new)",
          SS.run_batch(lay, "intake:tb1", lambda: SS.list_intake(lay, "tb1"), TV.FakeEmbedder(), guards, None,
                       procs=1) is None)
    # near_consumed: an image within 3 bits (not exact) of one a stream has consumed never returns as new data
    cons = C.INC_DIR / "stream" / "s_test" / "consumed.jsonl"
    cons.parent.mkdir(parents=True, exist_ok=True)
    cons.write_text(json.dumps({"key": "src_rf__i_ok", "increment": "i001", "disposition": "accepted"}) + "\n")
    d5 = C.INC_DIR / "intake" / "tb5"
    src_img = pathlib.Path(q["src_rf__i_ok"]["image"])
    dst = d5 / "images" / "c_near.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    nb = None
    for qual, dw in ((95, 0), (85, 0), (70, 0), (95, 4), (90, 6), (80, 8), (60, 6), (50, 10)):
        with Image.open(src_img) as im:
            im = im.convert("RGB")
            if dw:
                im = im.resize((im.width - dw, im.height - dw // 2), Image.BILINEAR)
            im.save(dst, quality=qual)
        nb = bits(C.dhash(src_img), C.dhash(dst))
        if 0 < nb <= 3:
            break
    lp = d5 / "labels" / "c_near.txt"
    lp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(q["src_rf__i_ok"]["label"], lp)
    (d5 / "manifest.jsonl").write_text(json.dumps({
        "key": "src_rf__c_near", "image": str(dst), "sha256": C.sha256_file(dst), "label": str(lp),
        "label_sha256": C.sha256_file(lp), "source": "src_rf", "licence": "CC BY 4.0", "research_only": False,
        "lab_group": None, "capture_group": "tray1", "dhash": C.dhash(dst)}) + "\n")
    (d5 / "summary.json").write_text(json.dumps({"format": "collect-intake-summary/1"}))
    try:
        doc5 = SS.run_batch(lay, "intake:tb5", lambda: SS.list_intake(lay, "tb5"), TV.FakeEmbedder(), guards, None,
                            procs=1, kind=SS.KIND_INTAKE)
    finally:
        cons.unlink()
    dec5 = {r["intake_key"]: r["decision"] for r in batch_rows(lay, doc5["batch"])}
    check("fixture: the re-encoded copy is 1-3 bits from the consumed image", 0 < nb <= 3, nb)
    check("an image within 3 bits of one a stream consumed is refused (near_consumed), never queued again",
          dec5 == {"src_rf__c_near": "near_consumed"} and not any(r["key"] == "src_rf__c_near"
                                                                   for r in SS.load_queue(lay)), dec5)
    # the collector's own verdicts: its one-source sources.json, its licence class and a person's override
    d, rows = C.INC_DIR / "intake" / "tb2", []
    add("k_over", [11, 12, 12], [11, 12, 12], "src_kg", licence="other:terms")
    add("k_claims", [6, 12, 12], [6, 12, 12], "src_kg", licence="other:terms")
    add("u_unres", [8, 12, 12], [8, 12, 12], "src_other", licence="other:foo")
    add("u_refused", [8, 12, 3], [8, 12, 12], "src_other", licence="cc-by-4.0")
    for r in rows:
        claims = r["key"].endswith("k_claims")
        r.update(licence_class="unresolved", holds=["licence"] if claims else ["h6_scan", "licence"],
                 hold_until="licence" if claims else "h6_scan", provenance_cleared=claims)
    rows[-1]["licence_class"] = "refused"
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "sources.json").write_text(json.dumps({
        "source_id": "src_kg", "provenance_cleared": False, "lab_group": None,
        "licence": {"id": "other:terms", "class": "unresolved", "research_only": False,
                    "override": {"id": "cc-by-nc-4.0", "decided_by": "owner"}}}))
    (d / "summary.json").write_text(json.dumps({"format": "collect-intake-summary/1"}))
    doc = SS.run_batch(lay, "intake:tb2", lambda: SS.list_intake(lay, "tb2"), TV.FakeEmbedder(), guards, None,
                       procs=1, kind=SS.KIND_INTAKE)
    q = {r["key"]: r for r in SS.load_queue(lay) if r["batch"] == doc["batch"]}
    check("the collector's licence verdict decides: an unresolved or refused class without a person's override "
          "stays held, whatever the id says (cc-by-4.0 included); the override recorded in its one-source sources.json resolves (NC -> research_only)",
          "licence" in q["src_other__u_unres"]["holds"] and "licence" in q["src_other__u_refused"]["holds"]
          and "licence" not in q["src_kg__k_over"]["holds"]
          and "person override" in q["src_kg__k_over"]["licence"] and q["src_kg__k_over"]["research_only"], q)
    check("a row that says provenance-cleared while its source says not is not cleared (held h6_scan)",
          "h6_scan" in q["src_kg__k_claims"]["holds"], q["src_kg__k_claims"]["holds"])
    # an image the mask step cannot open refuses itself, not the batch (a failing batch blocks every input)
    d, rows = C.INC_DIR / "intake" / "tb3", []
    add("m_fail", [9, 12, 4], [9, 12, 13], "src_rf2")
    add("m_ok", [7, 12, 3], [7, 12, 13], "src_rf2")
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps({"format": "collect-intake-summary/1"}))
    real = MK.mask_except

    def flaky(image, *a, **k):
        if "m_fail" in str(image):
            raise MK.MaskError("cannot open %s to mask it (OSError: truncated)" % image)
        return real(image, *a, **k)
    MK.mask_except = flaky
    try:
        doc = SS.run_batch(lay, "intake:tb3", lambda: SS.list_intake(lay, "tb3"), TV.FakeEmbedder(), guards, None,
                           procs=1, kind=SS.KIND_INTAKE)
    finally:
        MK.mask_except = real
    adm = {r["key"]: r for r in batch_rows(lay, doc["batch"], "admission.jsonl")}
    q = {r["key"]: r for r in SS.load_queue(lay) if r["batch"] == doc["batch"]}
    check("the batch commits; the image that cannot be masked is refused (mask_failed), the other is queued masked",
          adm["src_rf2__m_fail"]["refusal"] == "mask_failed" and "src_rf2__m_fail" not in q
          and q.get("src_rf2__m_ok", {}).get("admission") == MK.MASKED, (adm.get("src_rf2__m_fail"), sorted(q)))

    # the masked copy passes GuardV2 inside the pipeline (a guard that refuses every masked PNG)
    d, rows = C.INC_DIR / "intake" / "tb4", []
    add("m_copy", [10, 12, 2], [10, 12, 13], "src_rf3")
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps({"format": "collect-intake-summary/1"}))

    class MaskedRefuser(SS.Guards):
        def check(self, path, h=None):
            if "images_masked" in str(path):
                return "near_eval_variant", {"planted": "masked copy"}, 0
            return super().check(path, h)
    g4 = MaskedRefuser(guards.guard, guards.variants_fn, record=dict(guards.record))
    doc = SS.run_batch(lay, "intake:tb4", lambda: SS.list_intake(lay, "tb4"), TV.FakeEmbedder(), g4, None,
                       procs=1, kind=SS.KIND_INTAKE)
    adm = {r["key"]: r for r in batch_rows(lay, doc["batch"], "admission.jsonl")}
    check("a masked copy that GuardV2 refuses is not queued (masked_<reason>), though its unmasked image passed",
          adm["src_rf3__m_copy"]["refusal"] == "masked_near_eval_variant"
          and not any(r["key"] == "src_rf3__m_copy" for r in SS.load_queue(lay)), adm.get("src_rf3__m_copy"))


# ----------------------------------------------------------- status and CLI
def test_status_cli(lay, guards):
    print("status schema and the CLI")
    doc = SS.run_batch(lay, "registry", lambda: SS.list_registry(lay), TV.FakeEmbedder(), guards, None, procs=1)
    st = SS.write_status(lay)
    check("a full registry listing picks up the one new path left (b_other's, read through its override)",
          doc is not None and doc["items"] == 1 and st["pending"]["registry_new_paths"].get("b_other") == 0,
          (doc and doc.get("items"), st["pending"]))
    check("status.json has the schema", SS.check_status(st) == [], SS.check_status(st))
    check("status lists the refused registry slugs and the rejoins needed",
          st["pending"]["registry_refused"].get("cottonweed_sp8") == "calibration_only"
          and "a_species" in st["pending"]["rejoin_needed"], st["pending"])
    check("status: masked and refused-overlap counts, holds, known truth per batch, versions, crop ids",
          st["admission"]["masked"] >= 4 and st["admission"]["refused_overlap"] >= 1 and st["knowntruth"]
          and st["versions"]["verifier"] == "v1" and st["crop_ids"]["next"] > st["crop_ids"]["v1"]
          and st["batches"]["committed"][0] == "b0000" and "refit_triggers" in st, st["admission"])
    check("status: one_time names when bootstrap, backfill and knowntruth ran (the autopilot's rollout reads it)",
          all(isinstance(st["one_time"].get(k), str) for k in ("bootstrap", "backfill", "knowntruth")), st["one_time"])
    ps = st["per_source"]["a_species"]
    check("status per source: images_seen, near_eval_embed (copy-scan hits at ingest and later), never-train "
          "refusals and target_boxes_admitted, as D28 and the yield read them",
          ps["near_eval_embed"] >= 3 and ps["images_seen"] >= ps["decision:near_eval_embed"] + ps["decision:pool"]
          and ps["never_train_refused"] >= 6 and isinstance(ps["target_boxes_admitted"], int)
          and all({"images_seen", "near_eval_embed", "target_boxes_admitted"} <= set(v)
                  for v in st["per_source"].values()), ps)
    victim = next(r for r in SS.load_queue(lay) if not r["refused"] and r["source"] == "src_tum")
    SS._append_jsonl(lay.events, [{"event": "refuse", "batch": "serve-test", "key": victim["key"],
                                   "reason": "near_eval_embed", "utc": SS._utc()}])
    st2 = SS.write_status(lay)
    check("a copy-scan refusal made after admission (serve-holds) counts for its source's near_eval_embed",
          st2["per_source"]["src_tum"]["near_eval_embed"] == st["per_source"].get("src_tum", {}).get("near_eval_embed",
                                                                                                        0) + 1
          and st2["per_source"]["src_tum"]["refused:near_eval_embed"] == 1, st2["per_source"]["src_tum"])
    bad = dict(st)
    bad.pop("queue")
    check("a status without its queue section fails the schema", SS.check_status(bad) == ["missing queue"])
    bad = dict(st, one_time={"bootstrap": "t"})
    check("a status without the one-time record fails the schema", "one_time.backfill missing" in SS.check_status(bad))
    env = dict(os.environ)
    for verb in ("verify", "status"):
        out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.step1_stream", verb,
                              "--state-dir", str(lay.root)], cwd=str(ROOT), capture_output=True, text=True, env=env)
        check("the CLI `%s` runs (exit 0)" % verb, out.returncode == 0, out.stdout[-300:] + out.stderr[-500:])
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.step1_stream", "admit",
                          "--state-dir", str(lay.root), "--intake", "x", "--registry"], cwd=str(ROOT),
                         capture_output=True, text=True, env=env)
    check("the CLI refuses admit with both --intake and --registry (exit 2)", out.returncode == 2
          and "exactly one" in out.stdout, out.stdout[-300:] + out.stderr[-300:])
    n_led = len(SS.verify_ledger(lay.root / "ledger" / "holds.jsonl"))
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.step1_stream", "scan-holds",
                          "--state-dir", str(lay.root), "--hold", "licence", "--funnel-leak", str(TMP / "none.json"),
                          "--leak", str(TMP / "none.json")], cwd=str(ROOT), capture_output=True, text=True, env=env)
    led = SS.verify_ledger(lay.root / "ledger" / "holds.jsonl")
    check("the CLI accepts scan-holds --hold KIND (the verb and flag the autopilot submits) and serves only that hold",
          out.returncode == 0 and len(led) == n_led + 1 and led[-1]["kinds"] == ["licence"]
          and led[-1]["scanned"] == {}, out.stdout[-300:] + out.stderr[-500:])
    other = TMP / "other_lock" / "LOCK.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SS.read_pins(lay)["splits"]["lock"], other)
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.step1_stream", "admit",
                          "--state-dir", str(lay.root), "--intake", "tb1", "--lock", str(other)], cwd=str(ROOT),
                         capture_output=True, text=True, env=env)
    check("the CLI refuses a --lock other than the pinned splits v2 LOCK (exit 2): the guard is the pinned one",
          out.returncode == 2 and "is not the pinned splits v2 LOCK" in out.stdout, out.stdout[-300:] + out.stderr[-300:])
    # eval-hits (L17's D28-v2 verb): a batch it cannot give a sidecar fails the job, never a silent no-op
    fake = C.INC_DIR / "intake" / "i9999_evalhits_fake"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "summary.json").write_text(json.dumps({"source": "fake_src", "batch": fake.name, "images": 10,
                                                   "guard": {"near_eval_v2": 1},
                                                   "decisions": {"file": "decisions.jsonl", "sha256": "0" * 64}}))
    argv = [sys.executable, "-m", "weed_optimizer_framework.tools.inc2.step1_stream", "eval-hits", "--state-dir",
            str(lay.root), "--funnel-leak", str(TMP / "none.json"), "--leak", str(TMP / "none.json")]
    out = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, env=env)
    check("the CLI eval-hits tries every intake batch whose hits no record weighs, and exits 2 when one gets no "
          "sidecar (here its decisions are missing), naming it", out.returncode == 2
          and "no sidecar for intake i9999_evalhits_fake" in out.stdout and not (fake / "eval_hits.json").exists(),
          out.stdout[-600:] + out.stderr[-300:])
    shutil.rmtree(fake)
    out = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, env=env)
    st = json.loads(lay.status.read_text())
    check("  with nothing left to weigh it exits 0 and stamps one_time.eval_hits", out.returncode == 0
          and isinstance(st["one_time"].get("eval_hits"), str) and st["eval_hits"]["due"] == [],
          out.stdout[-300:] + out.stderr[-300:])


# ------------------------------------------------------------- job script
def test_job_script():
    print("run_inc2_stream.sh")
    script = ROOT / "run_inc2_stream.sh"
    text = script.read_text()
    check("bash -n passes", subprocess.run(["bash", "-n", str(script)]).returncode == 0)
    check("GPU-shared with one V100, never RM-shared", "#SBATCH --partition=GPU-shared" in text
          and "#SBATCH --gres=gpu:v100-32:1" in text and "RM-shared\n" not in text.split("set -uo")[1])
    outp = [ln for ln in text.splitlines() if ln.startswith("#SBATCH --output=")]
    check("the Slurm log goes to results/framework/inc/logs (created before any stream job), never under "
          "step1_stream/, which does not exist before the first bootstrap",
          len(outp) == 1 and outp[0].endswith("/results/framework/inc/logs/%x_%j.out"), outp)
    check("no checkout reset, no nested-to-outer copy, no Roboflow sync, no training",
          "git reset" not in text and "rsync" not in text and "cp -" not in text and "roboflow" not in text.lower()
          and "mega_trainer.py" in text and "inc.train" not in text)
    repo = TMP / "jobrepo"
    (repo / "results").mkdir(parents=True, exist_ok=True)
    os.symlink(ROOT, repo / "weed_llm_benchmark")
    outer = repo / "weed_optimizer_framework"
    for m in ("tools/inc/verify.py", "tools/inc/select.py", "tools/funnel/recover.py", "tools/funnel/leak.py",
              "tools/cwd12_species.py", "tools/near_dup.py", "tools/semisup_labeler.py", "tools/mega_trainer.py"):
        (outer / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "weed_optimizer_framework" / m, outer / m)
    bin_ = TMP / "jobbin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "python").symlink_to(sys.executable)
    ns = bin_ / "nvidia-smi"
    ns.write_text("#!/bin/sh\necho 'Tesla V100-SXM2-32GB, 32768'\n")
    ns.chmod(0o755)
    conda = TMP / "conda.sh"
    conda.write_text("conda() { return 0; }\n")
    stub = TMP / "jobstub"
    for mod in ("open_clip", "transformers", "torch"):
        try:
            __import__(mod)
        except ImportError:
            (stub / mod).mkdir(parents=True, exist_ok=True)
            (stub / mod / "__init__.py").write_text("")
            NOTES.append("job script test: %s is not installed here; a stub satisfies the import check" % mod)
    env = dict(os.environ, INC2_STREAM_REPO=str(repo), INC2_STREAM_CONDA_SH=str(conda), INC2_STREAM_DRY_RUN="1",
               PATH="%s:%s" % (bin_, os.environ["PATH"]), PYTHONPATH=str(stub))
    env.pop("SLURM_ARRAY_TASK_ID", None)

    def job(*argv, extra=None):
        return subprocess.run(["bash", str(script)] + list(argv), capture_output=True, text=True,
                              env=dict(env, **(extra or {})))
    out = job("admit", "--intake", "b7")
    check("a dry run of admit: exit 0, every module logged, the command unchanged", out.returncode == 0
          and "DRY RUN: python -u -m weed_optimizer_framework.tools.inc2.step1_stream admit --intake b7" in out.stdout
          and "module tools/inc/verify.py:" in out.stdout and "= outer" in out.stdout
          and "module tools/inc2/step1_stream.py:" in out.stdout, out.stdout[-800:] + out.stderr[-400:])
    check("the collector's licence rule an override is read with (collect/__init__.py, collect/licence.py) is "
          "hashed into the log", "module tools/collect/__init__.py:" in out.stdout
          and "module tools/collect/licence.py:" in out.stdout, out.stdout[-800:])
    (outer / "tools" / "collect").mkdir(parents=True, exist_ok=True)
    (outer / "tools" / "collect" / "licence.py").write_text("# stale\n")
    out_c = job("admit", "--intake", "b7")
    (outer / "tools" / "collect" / "licence.py").unlink()
    check("an outer collect/licence.py that differs from the nested copy refuses (exit 2)",
          out_c.returncode == 2 and "module tools/collect/licence.py: nested" in out_c.stdout
          and "differs from the nested" in out_c.stderr, out_c.stdout[-400:] + out_c.stderr[-300:])
    check("an unknown verb and no verb refuse (exit 2)", job("pool").returncode == 2 and job().returncode == 2)
    out = job("eval-hits")
    check("eval-hits (L17's D28-v2 verb) runs; it hashes every file of the collector package into the log "
          "(collect.intake.rescore_eval_hits re-derives intake hit images) and needs torch and transformers",
          out.returncode == 0 and "DRY RUN: python -u -m weed_optimizer_framework.tools.inc2.step1_stream eval-hits"
          in out.stdout and "module tools/collect/intake.py:" in out.stdout
          and "module tools/collect/normalize.py:" in out.stdout and "torch, transformers" in out.stdout,
          out.stdout[-600:] + out.stderr[-300:])
    out_a = job("admit", "--intake", "b7")
    check("  (only eval-hits: admit does not hash the collector's intake module)",
          "module tools/collect/intake.py:" not in out_a.stdout, out_a.stdout[-300:])
    (outer / "tools" / "collect" / "intake.py").write_text("# stale\n")
    out_c = job("eval-hits")
    (outer / "tools" / "collect" / "intake.py").unlink()
    check("  an outer collect/intake.py that differs from the nested copy refuses eval-hits (exit 2)",
          out_c.returncode == 2 and "differs from the nested" in out_c.stderr,
          out_c.stdout[-300:] + out_c.stderr[-300:])
    out = job("scan-holds", "--hold", "h6_scan")
    check("scan-holds --hold h6_scan (the autopilot's L17 form) runs", out.returncode == 0
          and "DRY RUN: python -u -m weed_optimizer_framework.tools.inc2.step1_stream scan-holds --hold h6_scan"
          in out.stdout, out.stdout[-400:] + out.stderr[-300:])
    check("an array job refuses", job("status", extra={"SLURM_ARRAY_TASK_ID": "1"}).returncode == 2)
    ns.write_text("#!/bin/sh\nexit 1\n")
    out = job("admit", "--registry")
    check("admit without a GPU refuses (exit 2)", out.returncode == 2 and "GPU" in out.stderr, out.stderr[-300:])
    check("status runs without a GPU", job("status").returncode == 0)
    ns.write_text("#!/bin/sh\necho 'Tesla V100-SXM2-32GB, 32768'\n")
    (outer / "tools" / "near_dup.py").write_text("# stale\n")
    out = job("backfill")
    check("an outer library module that differs from the nested copy refuses (exit 2)",
          out.returncode == 2 and "differs from the nested" in out.stderr, out.stdout[-400:] + out.stderr[-300:])
    (outer / "tools" / "near_dup.py").unlink()
    out = job("backfill")
    check("a missing outer library module refuses too", out.returncode == 2 and "NO OUTER COPY" in out.stdout)
    shutil.copyfile(ROOT / "weed_optimizer_framework" / "tools" / "near_dup.py", outer / "tools" / "near_dup.py")
    (repo / "run_inc2_stream.sh").write_text("#!/bin/bash\n# old\n")
    out = job("status")
    check("an outer run_inc2_stream.sh that differs from the nested copy refuses", out.returncode == 2
          and "differs from" in out.stderr)
    shutil.copyfile(script, repo / "run_inc2_stream.sh")
    check("an identical outer copy runs", job("status").returncode == 0)


def main():
    t0 = datetime.datetime.now()
    try:
        rows, boxes_of, exp = make_world()
        check("fixture: the near-duplicate pair is 1-3 bits apart", 0 < exp["near_bits"] <= 3, exp["near_bits"])
        run_v1()
        lock_eq, _sel = make_splits_v2(rows, "v2eq", with_selected=False)
        lock, sel = make_splits_v2(rows)
        guards_eq = make_guards(lock_eq)
        guards = make_guards(lock)
        test_groups_unit()
        test_intake_licence_unit()
        test_equivalence(lock_eq, guards_eq)
        test_split_invariance(lock_eq, guards_eq)
        lay = test_backfill(lock, guards, sel)
        scanner, theta = test_registry_planted(lay, guards, rows, sel)
        test_exif_masked(guards, rows, lay)
        test_refusals(lay)
        test_holds(lay, scanner, theta)
        test_pins(lay, lock, guards)
        test_state(lay, guards)
        test_writer_lock(lay)
        test_knowntruth(lay, guards)
        test_empty_batch(lay, guards)
        test_rejoin(lay, guards)
        test_rejoin_guards(lay, guards, sel)
        test_intake(lay, guards)
        test_status_cli(lay, guards)
        test_job_script()
    finally:
        for n in sorted(set(NOTES)):
            print("NOTE: %s" % n)
        if not os.environ.get("INC2_STREAM_KEEP"):
            shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s) (%.0fs)" % (len(FAILURES), (datetime.datetime.now() - t0).total_seconds()))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
