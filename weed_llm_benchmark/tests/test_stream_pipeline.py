#!/usr/bin/env python3
"""The continuous loop end to end on one synthetic world, through the CLIs
(docs/CONTINUOUS_LOOP.md §3, §4.2, §5, §6, §8, §10; the integration of groups
A-F).

One world, one INC_DIR, the real code of every group in order. The autopilot
(inc_autopilot stream mode: the real campaign.tick -> StreamRun, executor,
policy and approvals on test_stream_ap_world's simulated cluster) proposes
every lever from the evidence the real code wrote, and each proposal is
executed by the real module it names:

  v1   the pinned inc.splits build + lock and v1 Step 1 (verify pool / crops
       / embed / fit / admit, select build): the precondition, as on the cluster;
  R0   L23 (inc2.splits build, which runs the embedding copy scan itself with
       the funnel's passed leak_v1.json; then lock), each filed for a person
       and approved as the owner; L23B (inc2.baseline build: B_v2, the
       canary, both capacity arms, B0 u tsw) on the pinned driver with a
       FakeBackend; LV verdicts; L25 (Stage A); LI (inc2.stream init); LA
       (choose-arm); L28 (Stage C feasibility) and LC (compare); then, R0
       complete, L23B of the measurement arms m832, s1024, y26l640, y26m640
       and l640 (the stream's arm unchanged); once each is done, L23N (inc2.baseline rescore-native:
       its finals at its own imgsz), once per arm;
  R1   L17 bootstrap / knowntruth / backfill (inc2.step1_stream): batch b0000
       holds a masked veto image (funnel_F9), refuses a mirrored test image
       (GuardV2) and a 15 % dev crop (the embedding scan), sends an overlap to
       the human queue; D28 quarantines the leaking source (L24);
  R2   a person resolves licences (inc2.stream release); the cut of exactly
       M; L18 (segment build on the pinned driver; every spec through a
       synthetic executor that checks each training manifest with the real
       inc2.train); L19 commit with Protocol v3 (inc2.gate3): the planted bad
       increment REJECTed as data and quarantined, the good one ACCEPTed;
  MAINT L20 milestone 1 (5 v 5 on dev: helps); a second segment the chain
       accepts; L20 milestone 2 (hurts) -> L21 rollback -> L27 bisect, one arm
       per suspect increment: the harmful one quarantined, the good one
       returned; inc2.stream_report --stream;
  P8   the funnel's F9 file and serve-holds release the masked row's
       funnel_F9 hold; the next snapshot refreshes the stream summary;
  DATA two annotation-index sources through the real collector (fetch on a
       fake network, intake against the real LOCK v2): the autopilot admits
       the clean one's batch (L17 admit --intake, real step1_stream) and D28
       quarantines the one holding a mirrored dev image before admission;
       that batch's own pair cosines then removed (a batch committed before
       D28-v2), DR0 proposes L17 eval-hits, the real step1_stream eval-hits
       weighs the hit again from the staging blob into
       intake/<batch>/eval_hits.json, the snapshot ships it and D28 judges
       the source by it;
  deploy deploy/deploy_funnel.sh --dry-run ships every stream module, job
       script, test and fixture the replay gate needs.

Invariant checked throughout: no dev, test or ImageWeeds image (by key,
sha256, or dHash within 6 bits under any of the 8 flips and rotations)
reaches any training manifest the driver hands the executor, and every one
of those manifests passes the real inc2.train.check_manifest + guard_rows.

Honest scope (every deviation from the platform is named here):
  * The models are not trained: the pinned driver's FakeBackend runs every
    spec through a synthetic executor that writes scores and Protocol v3
    sidecars from what the manifest holds (harv_good helps, harv_bad hurts,
    harv_sneaky helps the warm chain but hurts a cold model).
  * Embedders are stand-ins: v1 Step 1 and step1_stream use test_inc_verify's
    colour embedder; the copy scan uses a hue-histogram descriptor, with a
    threshold set from this world (the funnel's leak_v1.json is written as a
    passed calibration, as the funnel's leak step would). The descriptor is
    summed over the image's 8 flips and rotations, so a mirrored copy scores
    1.0 against its original, as DINOv2's nearly does: D28-v2 weighs each
    dHash hit by that pair cosine (b0000's mirrored test image, the intake's
    mirrored dev image).
  * capacity-verdict refuses test-mode scores and has no --testing: the
    same function is called with testing_ok. The canary is judged against
    b0_v1's real runs and Stage A (pilot_v4) needs pilot_v3's real bins,
    neither of which a synthetic world has: their verdict files are written
    in inc2.baseline's and inc2.pilot4's formats by the autopilot world
    (their builds, L23B canary_v2 and L25, still run).
  * The platform's argv never passes --testing; the jobs of this world add
    it where a synthetic world needs it (inc2.splits, inc2.baseline build,
    step1_stream bootstrap, inc2.stream init).
  * Licences: the owner releases licence holds by keys (inc2.stream release,
    decided by a person), which is how this test sets exactly M eligible
    images per source.
  * The network probe's placement.json is the world's; collect fetch runs
    in-process on a fake network, with the disk-headroom rule patched; the
    L4 label audit is proposed but refused by the policy's path patterns,
    which pin the cluster's INC_DIR.
  * The measurement arms' native-resolution rescore (L23N) runs the real
    inc2.baseline rescore-native, which refuses this world's stand-in
    weights (they cannot be loaded as a detector): the platform's failure
    path is what this world exercises (a card, the stream runs on, never
    proposed again). The scoring itself runs on real CPU passes in
    tests/test_inc2_native.py.
  * sbatch, squeue, sacct, ssh, the allocation and quota reads are the
    simulated cluster's (test_stream_ap_world); every INC_DIR file the
    autopilot reads is written by the real code of groups A-E, and a done
    experiment's report by the pinned inc.report the snapshot runs.

No network, no GPU. Run:  python3 tests/test_stream_pipeline.py
(STREAM_PIPELINE_KEEP=1 keeps the world; STREAM_PIPELINE_COMMANDS=1 prints every command the platform proposed
and every login-node verb it ran, in order.)
"""
import collections
import datetime
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
import traceback

HERE = pathlib.Path(__file__).resolve().parent
TMP = pathlib.Path(tempfile.mkdtemp(prefix="stream_pipeline_"))
os.environ["INC_VERIFY_TEST_TMP"] = str(TMP)
os.environ["INC_SCORER_TESTING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["COLLECT_REGISTRY"] = str(TMP / "repo" / "results" / "framework" / "dataset_registry.json")
for _k in ("KAGGLE_API_TOKEN", "ROBOFLOW_API_KEY", "HF_TOKEN", "GITHUB_TOKEN", "SLURM_JOB_ID", "FUNNEL_GBIF_RECORD"):
    os.environ.pop(_k, None)
for _k in ("INC_JOB_SCRIPT", "INCAP_DECIDED_BY", "CLUSTER_SSH"):
    os.environ.pop(_k, None)
sys.path.insert(0, str(HERE))
import test_inc_verify as TV  # noqa: E402  (sets REPO = TMP/repo and INC_DIR = TMP/inc before the INC modules)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools import cwd12_species as CS  # noqa: E402
from weed_optimizer_framework.tools.funnel import embed as FE  # noqa: E402
from weed_optimizer_framework.tools.funnel import leak as L  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import select as SEL  # noqa: E402
from weed_optimizer_framework.tools.inc import splits as S1  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import splits as S2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import step1_stream as SS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream as ST  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream_report as SR2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T2  # noqa: E402
import funnel_prereg as FPR  # noqa: E402

assert C.INC_DIR == TMP / "inc" and C.REPO == TMP / "repo", "the temporary REPO / INC_DIR are not in effect"

REPO, INC = C.REPO, C.INC_DIR
ROOT_PKG = HERE.parent                     # weed_llm_benchmark/ of this checkout
FAILURES = []
NOTES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)
    return bool(cond)


def stage(title):
    print("\n== %s" % title)


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


def rj(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def jl(path):
    p = pathlib.Path(path)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------- stand-ins
class HueEmbedder:
    """The copy detector's descriptor (a stand-in for DINOv2 whole-image
    features): hue histograms of lit, saturated pixels on a 3 x 3 grid,
    summed over the image's 8 flips and rotations, L2-normalised; it survives
    crops, shears and brightness changes, and a flipped or rotated copy gets
    the descriptor of its original, as DINOv2's nearly does (true flip and
    rotation copies score >= 0.896 in pair cosine, D28-v2)."""
    bins, grid = 16, 3
    dim = bins * grid * grid
    D4 = (None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90,
          Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270, Image.Transpose.TRANSPOSE,
          Image.Transpose.TRANSVERSE)

    def __init__(self, model="facebook/dinov2-base", pooling="cls"):
        self.model_name, self.pooling = str(model), pooling
        self.name = FE.embedder_name(self.model_name, pooling)

    def _grid(self, im):
        hsv = np.asarray(im.convert("HSV"), dtype=np.int64)
        H, W = hsv.shape[:2]
        parts = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                c = hsv[gy * H // self.grid:(gy + 1) * H // self.grid, gx * W // self.grid:(gx + 1) * W // self.grid]
                m = (c[..., 1] > 40) & (c[..., 2] > 30)
                parts.append(np.bincount((c[..., 0][m] * self.bins) // 256, minlength=self.bins))
        return np.concatenate(parts).astype(np.float32)

    def __call__(self, pils):
        out = []
        for p in pils:
            rgb = p.convert("RGB")
            h = sum(self._grid(rgb if t is None else rgb.transpose(t)) for t in self.D4)
            n = np.linalg.norm(h)
            out.append(h / n if n > 0 else h)
        return np.stack(out)


# every copy-scan descriptor (inc2.splits' scan, step1_stream's CopyScanner) is made through
# funnel.embed.LazyEmbedder: the stand-in replaces it in this process only
FE.LazyEmbedder = HueEmbedder
# step1_stream admit / knowntruth embed crops with verify.BioclipEmbedder: the v1 world's colour embedder
V.BioclipEmbedder = lambda *a, **k: TV.FakeEmbedder()


# ------------------------------------------------------------------- world
IMG_W, IMG_H = TV.IMG_W, TV.IMG_H
SESSIONS = [12, 6, 5, 5, 4, 4, 3, 3, 2, 2]          # cwd12 train sessions (dev = whole sessions of 13-18 %)
SPECIES = list(CS.CWD12_SPECIES)
A_SRC, LAMBS = TV.A_SRC, TV.LAMBS
N_PER_SOURCE = 14
SOURCES = {"harv_good": "good", "harv_bad": "bad", "harv_sneaky": "sneaky"}
WORLD = {"boxes": {}, "copies": {}}


def paint(path, boxes, texture=True, fmt="jpg"):
    return TV.paint(pathlib.Path(path), boxes, texture=texture, fmt=fmt)


def voc(img_path, objects):
    """Pascal VOC next to the image: objects = [(name, (cls, cx, cy, w, h))]."""
    objs = ""
    for name, b in objects:
        _c, cx, cy, w, h = b
        x0, x1 = int(round((cx - w / 2) * IMG_W)), int(round((cx + w / 2) * IMG_W))
        y0, y1 = int(round((cy - h / 2) * IMG_H)), int(round((cy + h / 2) * IMG_H))
        objs += ("<object><name>%s</name><bndbox><xmin>%d</xmin><ymin>%d</ymin><xmax>%d</xmax><ymax>%d</ymax>"
                 "</bndbox></object>" % (name, x0, y0, x1, y1))
    pathlib.Path(img_path).with_suffix(".xml").write_text(
        "<annotation><size><width>%d</width><height>%d</height><depth>3</depth></size>%s</annotation>"
        % (IMG_W, IMG_H, objs))


def make_v1_sources():
    """cwd12 (train sessions, valid, test), 3SeasonWeedDet10 data2022 / data2023
    (VOC) and ImageWeeds, laid out as inc.splits reads them."""
    cw = REPO / "downloads" / "cottonweeddet12"
    g = 0
    for s, n in enumerate(SESSIONS):
        for f in range(n):
            stem = "20210701_Cam_S%d_%d" % (s, f + 1)
            bx = TV.layout3([(g + 4 * k) % 12 for k in range(3)])
            paint(cw / "train" / "images" / (stem + ".jpg"), bx)
            TV.yolo(cw / "train" / "labels" / (stem + ".txt"), bx)
            WORLD["boxes"][stem] = bx
            g += 1
    for sub, n, tag in (("valid", 3, "V"), ("test", 3, "T")):
        for f in range(n):
            stem = "20210801_Cam_%s_%d" % (tag, f + 1)
            bx = TV.layout3([(g + 4 * k) % 12 for k in range(3)])
            paint(cw / sub / "images" / (stem + ".jpg"), bx)
            TV.yolo(cw / sub / "labels" / (stem + ".txt"), bx)
            WORLD["boxes"][stem] = bx
            g += 1
    ts = REPO / "downloads" / "3seasonweeddet10"
    for season, d, sess, n in (("2022", ts / "data2022" / "fieldA", "a", 5), ("2022", ts / "data2022" / "fieldB", "b", 5),
                               ("2023", ts / "data2023", "c", 6)):
        for i in range(n):
            stem = "field%s%s_%d" % (season, sess, i + 1)
            bx = TV.layout3([(g + 4 * k) % 12 for k in range(2)] + [12])
            p = d / (stem + ".jpg")
            paint(p, bx)
            voc(p, [(SPECIES[b[0]], b) for b in bx[:2]] + [("Lambsquarters", bx[2])])
            g += 1
    iw = REPO / "datasets" / S1.IMAGEWEEDS_SLUG
    for i in range(4):
        bx = TV.layout3([12, 12, 12])
        paint(iw / "images" / ("iw_%d.jpg" % i), bx)
        TV.yolo(iw / "labels" / ("iw_%d.txt" % i), [(3 if k == 0 else 0,) + tuple(b[1:]) for k, b in enumerate(bx)])


def harvested(slug, name, boxes, labels, texture=False, fmt="png", image=None):
    """One image of a harvested slug (flat layout, a_species' class names)."""
    d = REPO / "datasets" / slug
    p = d / "images" / ("%s.%s" % (name, fmt))
    if image is None:
        paint(p, boxes, texture=texture, fmt=fmt)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        image.save(p)
    TV.yolo(d / "labels" / ("%s.txt" % name), labels)
    return p


def src_label(cls):
    return A_SRC[cls] if cls < 12 else LAMBS


def make_harvest():
    """The v1 registry: three target sources of N_PER_SOURCE verified images
    (two species boxes and one OtherPlant box each), an OtherPlant-only
    source, and a veto source: a verified target box beside a conflict (the
    masked veto image), one beside an overlapping conflict (refused_overlap),
    a mirrored copy of a test image and a 15 % crop of a dev image, each with
    one conflict box (so v1 vetoes them and b0000 must refuse them)."""
    k = 0
    for slug in SOURCES:
        for i in range(N_PER_SOURCE):
            sp = [(k + 5 * i) % 12, (k + 5 * i + 3) % 12]
            bx = TV.layout3(sp + [12])
            harvested(slug, "%s_%02d" % (slug.split("_")[1], i), bx, [(src_label(b[0]),) + tuple(b[1:]) for b in bx])
        k += 1
    for i in range(6):
        bx = TV.layout3([12, 12, 12])
        harvested("harv_other", "other_%02d" % i, bx, [(LAMBS,) + tuple(b[1:]) for b in bx])
    # the masked veto image: Waterhemp verified, a Purslane-painted box labelled Lambsquarters (conflict)
    bx = TV.layout3([0, 8, 12])
    harvested("harv_veto", "veto_mask", bx, [(A_SRC[0],) + bx[0][1:], (LAMBS,) + bx[1][1:], (LAMBS,) + bx[2][1:]])
    # a conflict overlapping a verified box (IoU >= 0.5): refused_overlap, the human queue
    bx = [(10, 0.3, 0.5, 0.22, 0.32), (10, 0.33, 0.52, 0.22, 0.32), (12, 0.8, 0.5, 0.15, 0.2)]
    harvested("harv_veto", "veto_overlap", bx, [(A_SRC[10],) + bx[0][1:], (LAMBS,) + bx[1][1:], (LAMBS,) + bx[2][1:]])
    # planted evaluation copies (conflicted so that v1 vetoes them and b0000 re-reads them)
    cw = REPO / "downloads" / "cottonweeddet12"
    t_stem = "20210801_Cam_T_1"
    with Image.open(cw / "test" / "images" / (t_stem + ".jpg")) as im:
        flip = im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    tb = [(b[0], round(1.0 - b[1], 4)) + tuple(b[2:]) for b in WORLD["boxes"][t_stem]]
    harvested("harv_veto", "copy_hflip_test", None,
              [(A_SRC[tb[0][0]],) + tb[0][1:], (LAMBS,) + tb[1][1:]], image=flip)
    WORLD["copies"]["harv_veto__copy_hflip_test"] = "test"
    WORLD["copy_sources"] = {"test": cw / "test" / "images" / (t_stem + ".jpg")}


def register():
    ds = REPO / "datasets"
    reg = {"datasets": {s: {"local_path": str(ds / s), "annotation": "bbox", "class_names": TV.A_NAMES}
                        for s in list(SOURCES) + ["harv_other", "harv_veto"]}}
    fw = REPO / "results" / "framework"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "dataset_registry.json").write_text(json.dumps(reg))
    (fw / "dataset_flags.json").write_text(json.dumps({}))


def plant_dev_crop_copy():
    """A 15 % crop of a dev image (known only after the v1 build chose dev),
    more than 6 bits from it under all 8 variants: only the embedding scan can
    see it."""
    dev = C.read_manifest(C.manifest_path("dev"))
    for r in dev:
        with Image.open(r["image"]) as im0:
            im = im0.convert("RGB")
        W, H = im.size
        c = int(round(0.15 * W)), int(round(0.15 * H))
        cp = im.crop((c[0], c[1], W, H)).resize((W, H), Image.BILINEAR)
        hv = L.dhash_variants(cp)
        h0 = L.dhash_variants(im)["id"]
        if min(bits(hv[v], h0) for v in L.VARIANTS) <= 6:
            continue
        stem = pathlib.Path(r["image"]).stem
        bx = WORLD["boxes"][stem]
        nb = []
        for b in bx:
            cx, cy = (b[1] - 0.15) / 0.85, (b[2] - 0.15) / 0.85
            w, h = b[3] / 0.85, b[4] / 0.85
            if 0.1 < cx < 0.9 and 0.1 < cy < 0.9:
                nb.append((b[0], round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)))
        if len(nb) < 2:
            continue
        harvested("harv_veto", "copy_crop_dev", None,
                  [(A_SRC[nb[0][0]],) + nb[0][1:], (LAMBS,) + nb[1][1:]], image=cp)
        WORLD["copies"]["harv_veto__copy_crop_dev"] = "dev"
        WORLD["copy_sources"]["dev"] = pathlib.Path(r["image"])
        return True
    return False


def census_from_v1():
    """The funnel census's veto numbers (verified target boxes per class,
    admitted or lost to the image rule), computed here from the v1 files,
    independently of step1_stream (as the funnel's census adapter does)."""
    crops = V.Crops()
    pv, _codes = SEL.read_pool_verdicts(V.POOL_VERDICTS, crops)
    lookup = V._box_verdict_lookup(crops, "pool", lambda i: V.VERDICT_CODES[int(pv[i])])
    meta = {m["key"]: m for m in V._read_jsonl(V.POOL_META)}
    admitted = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    per = collections.defaultdict(lambda: {"verified": 0, "admitted": 0})
    lost, imgs = 0, set()
    for k, m in meta.items():
        for b, box in enumerate(m["boxes"]):
            if int(box[0]) < 12 and lookup(k, b)[0] == V.VERIFIED:
                per[C.CLASS_NAMES[int(box[0])]]["verified"] += 1
                if k in admitted:
                    per[C.CLASS_NAMES[int(box[0])]]["admitted"] += 1
                else:
                    lost += 1
                    imgs.add(k)
    return {"veto": {"lost_boxes": lost, "images": len(imgs), "per_class": {k: dict(v) for k, v in per.items()}}}, imgs


def passed_record(threshold):
    """A calibration record that shows it passed (inc2.guard.calibration_problems
    finds nothing): every augmentation family at recall 1, both negative sets
    within the false-positive gate, H6's gates, the never-train radius."""
    return {"ok": True, "why": [], "cos_threshold": threshold, "dhash_bits_max": 6,
            "recall_min": L.RECALL_MIN, "fpr_max": L.FPR_MAX,
            "positives": {f: {"n": 40, "hits": 40} for f in L.FAMILIES},
            "negatives": {"pairs_7_10": {"n": 200, "false_hits": 0}, "hard": {"n": 200, "false_hits": 1}}}


# ------------------------------------------------------------------ stages
def stage_v1():
    stage("v1: the pinned inc.splits build + lock, then v1 Step 1 (verify, select) -- the precondition")
    make_v1_sources()
    rc_b = S1.main(["build", "--dev-min-boxes", "1"])
    rc_l = S1.main(["lock", "--accept-counts"])
    check("inc.splits build and lock (v1) on the synthetic cwd12 / 3SeasonWeedDet10 / ImageWeeds",
          rc_b == 0 and rc_l == 0 and C.LOCK_PATH.is_file(), (rc_b, rc_l))
    make_harvest()
    check("fixture: a 15 % crop of a dev image, > 6 bits from it under every variant, is planted",
          plant_dev_crop_copy(), WORLD["copies"])
    register()
    V.cmd_pool(TV.args("pool", "--procs", "1"))
    V.cmd_crops(TV.args("crops"))
    V.cmd_embed(TV.args("embed", "--nshards", "1", "--procs", "1"), embedder=TV.FakeEmbedder())
    V.cmd_fit(TV.args("fit"))
    V.cmd_admit(TV.args("admit"))
    SEL.build(base_frac=0.1, seed=0, workers=1)
    admitted = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    base_b = C.read_manifest(V.STEP1 / SEL.BASE_SELECTED)
    WORLD["base_b_keys"] = {r["key"] for r in base_b}
    per = collections.Counter(k.split("__")[0] for k in admitted)
    check("v1 Step 1 admits the three target sources' images whole (%s) and vetoes the veto source" % dict(per),
          all(per.get(s, 0) >= N_PER_SOURCE - 1 for s in SOURCES) and not any(k.startswith("harv_veto") for k in admitted),
          dict(per))
    check("v1 select put %d harvested images into base B" % len(base_b), 0 < len(base_b) <= 12, len(base_b))


def write_funnel_inputs():
    """What the funnel campaign has written on the cluster, read only by the
    stream: the card index (licences; harv_other only, so the target sources
    wait for a person's licence decision), the census (veto numbers), and a
    complete leak_v1.json whose calibration passed, with a threshold set from
    this world's descriptors (the planted dev crop above it, every unrelated
    pair below)."""
    fdir = INC / "funnel"
    (fdir / "cards").mkdir(parents=True, exist_ok=True)
    (fdir / "cards" / "index.json").write_text(json.dumps({"cards": {"harv_other": [
        {"what": "card", "status": 200, "licence": "cc-by-4.0", "url": "https://example.org/card",
         "fetched_utc": "2026-09-20T00:00:00Z"}]}}))
    census, imgs = census_from_v1()
    (fdir / "census_v1.json").write_text(json.dumps(census))
    WORLD["veto_images"] = sorted(imgs)
    emb = HueEmbedder()
    ev = [r["image"] for s in ("dev", "test", "imageweeds") for r in C.read_manifest(C.manifest_path(s))]

    def desc(paths):
        pils = []
        for p in paths:
            with Image.open(p) as im:
                pils.append(im.convert("RGB"))
        return emb(pils)
    E = desc(ev)
    cand = [str(p) for p in sorted((REPO / "datasets").glob("harv_*/images/*"))
            if not p.name.startswith("copy_")]          # the planted copies (the mirrored test image scores 1.0)
    cand += [r["image"] for s in ("ood22", "ood23") for r in C.read_manifest(C.manifest_path(s))]
    other = float((desc(cand) @ E.T).max())
    crop = REPO / "datasets" / "harv_veto" / "images" / "copy_crop_dev.png"
    orig = str(WORLD["copy_sources"]["dev"])
    cos_copy = float((desc([crop]) @ desc([orig]).T)[0, 0])
    theta = round((cos_copy + other) / 2, 4)
    check("fixture: the descriptor separates the planted dev crop (cos %.4f) from every unrelated pair (max %.4f); "
          "threshold %.4f" % (cos_copy, other, theta), cos_copy > other, (cos_copy, other))
    doc = {"format": "funnel-leak/1", "status": "complete", "h6b": {"base_copy": False, "incident": False},
           "detector": {"descriptor": {"embedder": emb.name}}, "calibration": passed_record(theta),
           "scans": {"base_B": {"images": len(WORLD["base_b_keys"]), "copies": 0, "listed": []}}}
    (fdir / "leak_v1.json").write_text(json.dumps(doc))
    WORLD["theta"] = theta


def approve_l23(w, verb):
    """Tick until the autopilot files L23 <verb> for a person (the D-A command needs one approval, 6.3), approve
    it as the owner, and tick until its job has run. Returns the job's exit status or None."""
    from weed_optimizer_framework.tools.brain import approvals as AP
    item = None
    for _ in range(12):
        w.tick()
        w.process_jobs()
        pend = [p for p in AP.pending("weed", root=str(w.lab)) if p.get("action") == "inc_splits_build"
                and '"%s"' % verb in json.dumps(p.get("params") or {})]
        if pend:
            item = pend[0]
            break
    if item is None:
        return None, None
    AP.decide("weed", item["id"], "approve", W.OWNER, "the D-A splits %s (test owner)" % verb, w.clock(), root=str(w.lab))
    n0 = len(w.jobs)
    w.run_until(lambda: any(a and a[:1] == ["inc2.splits"] and verb in a for n, a, rc in w.jobs[n0:]), max_ticks=12,
                note="L23 %s" % verb)
    done = [rc for n, a, rc in w.jobs[n0:] if a and a[:1] == ["inc2.splits"] and verb in a]
    return item, (done[-1] if done else None)


def stage_splits_v2(w):
    stage("R0: the autopilot files L23 (inc2.splits build, then lock) for a person; approved, its jobs run the real "
          "inc2.splits (the build scans by itself)")
    write_funnel_inputs()
    # the network probe runs on a compute node; its placement.json (group D's format) is the world's
    w.placement({"hf": "pass", "ftp": "pass", "weedai": "pass", "kaggle": "pass", "roboflow": "pass",
                 "mendeley_zenodo": "pass"})
    item, rc = approve_l23(w, "build")
    summ = rj(S2.summary_path(), {})
    scan = rj(S2.embed_scan_path(), {})
    check("L23 build was filed for a person (R3, never run by the envelope), approved, and its job ran inc2.splits "
          "build: exit 0; the embedding scan ran inside it and reused the funnel's passed leak_v1.json",
          item is not None and rc == 0 and summ.get("testing") and scan.get("status") == "complete"
          and (scan.get("detector") or {}).get("calibration_source") == "funnel_leak_v1",
          (item and item.get("action"), rc, scan.get("detector")))
    item, rc = approve_l23(w, "lock")
    lock = rj(C2.LOCK_PATH, {})
    check("L23 lock, likewise: LOCK v2 records the v2 manifests, both indexes and the scorer; every file is read-only",
          item is not None and rc == 0 and lock.get("splits_version") == "v2" and set(lock.get("manifests") or {}) >=
          {"train_core", "tsw22", "tsw23", "base_v2", "dev", "test", "imageweeds"}
          and all(not os.access(str(p), os.W_OK) for p in C2.SPLITS_DIR.glob("*.jsonl")), (rc, sorted(lock)))
    base = C.read_manifest(C2.SPLITS_DIR / "base_v2.jsonl")
    ev_keys = {r["key"] for s in ("dev", "test", "imageweeds") for r in C.read_manifest(C.manifest_path(s))}
    check("base_v2 = train_core + tsw22 + tsw23 + base B's kept part, no evaluation key (%d images)" % len(base),
          len(base) > 0 and not ({r["key"] for r in base} & ev_keys), len(base))
    WORLD["base_v2"] = base
    WORLD["M"] = int(math.ceil(0.10 * len(base)))
    NOTES.append("base_v2 %d images -> M = ceil(0.10 x |base_v2|) = %d" % (len(base), WORLD["M"]))
    check("M = %d leaves room for %d images per target source" % (WORLD["M"], N_PER_SOURCE),
          2 <= WORLD["M"] <= N_PER_SOURCE - 6, WORLD["M"])

# --------------------------------------------------------------- executor
NOISE = {0: 0.0, 1: 0.001, 2: -0.001, 3: 0.0005, 4: -0.0005}
KIND_OF_SOURCE = {"harv_good": "good", "harv_bad": "bad", "harv_sneaky": "sneaky"}
# kind -> (cand effect, null effect) of an incremental step; and each image's cold effect (x 1/M)
STEP_EFFECT = {"good": (0.02, 0.0), "bad": (-0.06, 0.0), "sneaky": (0.02, 0.0), "expert": (0.02, 0.0),
               "neutral": (0.0, 0.0)}
COLD_EFFECT = {"good": 0.01, "bad": -0.03, "sneaky": -0.05}
FACTOR = {"dev": 1.0, "imageweeds": 0.8, "test": 1.05}
SE_DEFAULT = 0.004


def kind_of_row(r):
    if r.get("source") in KIND_OF_SOURCE:
        return KIND_OF_SOURCE[r["source"]]
    if str(r.get("source", "")).startswith("3seasonweeddet10/"):
        return "expert"
    return "neutral"


class EvalIndex:
    """An independent never-train check (not GuardV2): every dev, test and
    ImageWeeds image of the v1 manifests by key, sha256, and dHash within 6
    bits under any of the 8 flips and rotations."""

    def __init__(self):
        self.keys, self.shas, self.hashes = set(), set(), []
        for s in ("dev", "test", "imageweeds"):
            for r in C.read_manifest(C.manifest_path(s)):
                self.keys.add(r["key"])
                self.shas.add(r["sha256"])
                with Image.open(r["image"]) as im:
                    self.hashes.append((s, r["key"], int(L.dhash_variants(im)["id"])))

    def hits(self, rows):
        out = []
        for r in rows:
            if r["key"] in self.keys or r.get("sha256") in self.shas:
                out.append((r["key"], "key/sha256"))
                continue
            with Image.open(r["image"]) as im:
                hv = L.dhash_variants(im)
            for s, k, h in self.hashes:
                b = min(bits(v, h) for v in hv.values())
                if b <= 6:
                    out.append((r["key"], "%s %s at %d bits" % (s, k, b)))
                    break
        return out


class Executor:
    """What run_inc2_job.sh runs, stood in for: every spec is validated by the
    real inc2.train.validate_spec, every training manifest by the real
    inc2.train.check_manifest + guard_rows and by an independent never-train
    check; scores and Protocol v3 sidecars are then written in the scorer's
    format from what the manifest holds."""

    def __init__(self):
        self.runs, self.manifests, self.problems = [], {}, []
        self.index = None

    def __call__(self, spec):
        out = pathlib.Path(spec["out_dir"])
        rid, kind = spec["run_id"], spec["kind"]
        self.runs.append((spec["exp"], rid))
        try:
            T2.validate_spec(spec, out / "spec.json")
        except T2.RunError as e:
            self.problems.append("spec %s/%s refused by inc2.train.validate_spec: %s" % (spec["exp"], rid, e))
            return "FAILED"
        if spec.get("train_manifest"):
            self._check_manifest(spec)
        if (out / "run.json").exists():
            (out / "run.json").unlink()
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if kind == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s %s" % (spec["exp"], rid)).encode())
        for exam in spec["exams"]:
            v, pc = self.values(spec, exam)
            self._score(out / "scores" / ("%s.json" % exam), exam, v, pc, w)
            if exam == "dev" and kind in ("base", "union", "cand", "soup"):
                self._sidecar(out / "scores" / "dev.sidecar.json", v, w)
        (out / "run.json").write_text(json.dumps({"status": "done", "attempt": 1, "seconds": 100.0, "error": None,
                                                  "weights_sha256": C.sha256_file(w)}))
        return "COMPLETED"

    def _check_manifest(self, spec):
        path = str(pathlib.Path(spec["train_manifest"]).resolve())
        if path in self.manifests:
            return
        rec = {"exp": spec["exp"], "run": spec["run_id"], "rows": 0, "guard": None, "hits": []}
        try:
            rows, dh, _info = T2.check_manifest(path)
            rec["rows"] = len(rows)
            rec["guard"] = T2.guard_rows(rows, dh, production=False)
        except T2.RunError as e:
            self.problems.append("training manifest %s (%s) refused by inc2.train: %s" % (path, spec["run_id"], e))
            rows = C.read_manifest(path)
        if self.index is None:
            self.index = EvalIndex()
        rec["hits"] = self.index.hits(rows)
        if rec["hits"]:
            self.problems.append("training manifest %s holds evaluation images: %s" % (path, rec["hits"][:5]))
        self.manifests[path] = rec

    @staticmethod
    def parent(weights):
        s = json.loads((pathlib.Path(weights).parent.parent / "scores" / "dev.json").read_text())
        return s["map50_95"], s["per_class"]

    @staticmethod
    def step_kind(spec):
        defn = rj(D.Paths(spec["exp"]).exp_json, {})
        tag = spec["run_id"].split("__")[1]
        for st in defn.get("steps") or []:
            if tag == st["name"] or tag.endswith("_" + st["name"]):
                kinds = collections.Counter(kind_of_row(r) for r in C.read_manifest(st["manifest"]))
                return kinds.most_common(1)[0][0]
        raise RuntimeError("run %s names no step of %s" % (spec["run_id"], spec["exp"]))

    def values(self, spec, exam):
        kind = spec["kind"]
        M = WORLD["M"]
        if kind in ("base", "union"):
            rows = C.read_manifest(spec["train_manifest"])
            eff = sum(COLD_EFFECT.get(kind_of_row(r), 0.0) for r in rows) / float(M)
            v = 0.40 + eff + NOISE[spec["recipe"]["seed"]]
            pc = {s: v for s in SPECIES}
        elif kind in ("cand", "null"):
            pv, ppc = self.parent(spec["init"])
            ce, ne = STEP_EFFECT[self.step_kind(spec)]
            e = ce if kind == "cand" else ne
            n = NOISE[spec["recipe"]["seed"]]
            v = pv + e + n
            pc = {s: ppc[s] + e + n for s in SPECIES}
        elif kind == "soup":
            vals = [self.parent(x) for x in spec["soup_of"]]
            v = float(np.mean([a for a, _ in vals])) + 0.001
            pc = {s: float(np.mean([b[s] for _, b in vals])) + 0.001 for s in SPECIES}
        else:
            pv, ppc = self.parent(spec["init"])
            v = pv * FACTOR[exam]
            pc = {s: ppc[s] * FACTOR[exam] for s in SPECIES}
        return v, pc

    @staticmethod
    def _score(path, exam, v, pc, weights):
        import hashlib
        n_gt = {s: 40 for s in SPECIES}
        n_gt["OtherPlant"] = 0 if exam in ("dev", "test") else 25
        per_class = dict(pc)
        if n_gt["OtherPlant"]:
            per_class["OtherPlant"] = 0.5 * v
        doc = {"exam": exam, "scorer_sha256": "TEST-" + "5" * 64, "production": False,
               "deviations": ["LOCK.json not checked"],
               "manifest_sha256": hashlib.sha256(("m/" + exam).encode()).hexdigest(),
               "key_order_sha256": hashlib.sha256(("k/" + exam).encode()).hexdigest(),
               "weights_sha256": C.sha256_file(weights), "n_images": 20,
               "map50_95": v, "map50": min(1.0, v + 0.2), "agnostic_map50_95": v + 0.1,
               "agnostic_map50": min(1.0, v + 0.3), "per_class": per_class, "n_gt": n_gt,
               "image_correct": "1" * 20, "species_map50_95": v, "species_map50": v + 0.2}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))

    @staticmethod
    def _sidecar(path, v, weights):
        import hashlib
        stamps = {"exam": "dev", "scorer_sha256": "TEST-" + "5" * 64,
                  "manifest_sha256": hashlib.sha256(b"m/dev").hexdigest(),
                  "key_order_sha256": hashlib.sha256(b"k/dev").hexdigest(), "n_images": 20}
        per = {s: {"se": SE_DEFAULT, "n_valid": 1000, "n_gt": 40, "ap": v, "mean": v, "p2_5": v - 0.01,
                   "p97_5": v + 0.01} for s in SPECIES}
        path.write_text(json.dumps({"format": "inc2-scorer-sidecar/1", "exam": "dev",
                                    "weights_sha256": C.sha256_file(weights),
                                    "score": {"stamps": stamps, "production": False},
                                    "species_se": {"seed_text": "inc2/v3/species_se", "resamples": 1000,
                                                   "per_species": per}}))


EX = Executor()
FB = D.FakeBackend(runner=EX)
D.SlurmBackend = lambda *a, **k: FB          # every driver the CLIs make in this process uses the FakeBackend


def drive(exp, max_iter=60):
    st = {}
    for _ in range(max_iter):
        ran = FB.run_pending(EX)
        before = len(FB.submissions)
        D.Driver(exp, backend=FB, quiet=True).advance()
        st = rj(D.Paths(exp).state, {})
        if st.get("done") or (ran == 0 and len(FB.submissions) == before):
            break
    return st


# ------------------------------------------------- the stream's dependencies
class JobScriptEnv:
    """run_inc2_build.sh and run_inc2_job.sh export INC_JOB_SCRIPT as the v2
    executor's script; the in-process CLIs get the same."""

    def __enter__(self):
        self.old = os.environ.get("INC_JOB_SCRIPT")
        os.environ["INC_JOB_SCRIPT"] = str(B2.job_script_path())

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("INC_JOB_SCRIPT", None)
        else:
            os.environ["INC_JOB_SCRIPT"] = self.old


def baseline_runner(argv):
    with JobScriptEnv():
        return B2.main(list(argv))


def secondary_runner(exp, weights, source):
    return B2.secondary(exp, weights, source=source)["argv"]


def submitter(argv):
    lst = pathlib.Path(argv[-2])
    for ln in lst.read_text().splitlines():
        if ln.strip():
            EX(json.loads(pathlib.Path(ln.strip()).read_text()))
    return "777"


DEPS = ST.Deps(backend=FB, baseline_runner=baseline_runner, secondary_runner=secondary_runner, submitter=submitter)


def call(fn, argv, **kw):
    """(exit status, stdout, stderr) of a CLI main run in this process."""
    import contextlib
    import io
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = fn(list(argv), **kw)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
    o, e_ = out.getvalue(), err.getvalue()
    sys.stdout.write(o[-3000:])
    sys.stdout.write(e_[-3000:])
    return int(rc or 0), o, e_



# ------------------------------------------------- the autopilot on this world
import test_stream_ap_world as W  # noqa: E402  (the simulated cluster: sbatch, squeue, sacct, ssh, projects, quota)

REAL_REPORT = W.R.report                       # the pinned report the snapshot runs (before any world patches it)

BUILD_SCRIPTS = ("run_inc2_build.sh", "run_inc2_stream.sh", "run_inc_collect.sh")


class PipelineWorld(W.World):
    """test_stream_ap_world's simulated cluster, with every INC_DIR file and
    every verb the real code of groups A-E: the snapshot reads this test's
    INC_DIR; a login-node verb runs the real CLI in this process; an sbatch'd
    build or Step 1 job is run by the real module it names (process_jobs);
    an advance drives the pinned driver on the FakeBackend."""

    def __init__(self):
        super().__init__("pipeline")
        self.inc = INC                         # the cluster's INC_DIR is this world's
        self.done_submits = 0
        self.jobs = []                         # (job name, argv tail, exit status)
        self.lever_log = []

    # -- login-node verbs (stream_remote._run)
    def _run(self, argv, timeout=120, stdin=None, env=None):
        self.runs.append(list(argv))
        mod = argv[2] if len(argv) > 2 else ""
        rest = list(argv[3:])
        verb = rest[0] if rest else None
        who = (env or {}).get("INCAP_DECIDED_BY")
        if mod.endswith("inc2.stream"):
            old = os.environ.get("INCAP_DECIDED_BY")
            if who:
                os.environ["INCAP_DECIDED_BY"] = who
            try:
                return call(ST.main, rest, deps=DEPS, clock=self.clock)
            finally:
                if old is None:
                    os.environ.pop("INCAP_DECIDED_BY", None)
                else:
                    os.environ["INCAP_DECIDED_BY"] = old
        if mod.endswith("inc2.baseline") and verb == "capacity-verdict":
            # the CLI refuses test-mode scores and has no --testing: the same function with testing_ok
            dec, _rep = B2.capacity_verdict(testing_ok=True)
            return 0, json.dumps({"chosen_arm": dec["chosen_arm"]}), ""
        if mod.endswith("inc2.baseline") and verb == "canary-verdict":
            # the canary reproduces b0_v1's real runs, which a synthetic world does not have: the verdict file is
            # written in inc2.baseline's format by the world (named in the module docstring)
            self.canary_file(passed=True)
            return 0, "{}", ""
        if mod.endswith("inc2.pilot4") and verb == "verdict":
            self.stage_a_file(status="READY", recipes=["r0", "x1a"])
            return 0, "{}", ""
        return 1, "", "the pipeline world has no login-node verb %r %r" % (mod, verb)

    # -- an advance of the pinned driver (remote.advance)
    def _advance(self, exp, backend=None):
        self.advances.append({"exp": exp, "job_script": os.environ.get("INC_JOB_SCRIPT")})
        n0 = len(FB.submissions)
        st = drive(exp)
        rec = W.R.base_record("advance")
        rec["exp"] = exp
        rec["result"] = {"locked": False, "passes": 1, "submitted": len(FB.submissions) - n0, "job_ids": [],
                         "done": bool(st.get("done")), "lines": []}
        return rec

    # -- the snapshot's report of a done experiment: the real one (the pinned inc.report)
    def _report(self, exp):
        return REAL_REPORT(exp)

    def finish_lab(self):
        """A lab job (L15 discovery) ends: this world's discovery finds no source."""
        pending = [l for l in self.runner.launched if l["job"] not in self.runner.results]
        if pending:
            self.candidates([], recall={"known_items": 0, "found": 0})
            self.runner.finish()

    def process_jobs(self):
        """Run every sbatch'd job the autopilot submitted since the last call,
        by the real module its job script would run; then end the job in the
        simulated sacct (COMPLETED on exit 0, else FAILED)."""
        while self.done_submits < len(self.submits):
            sub = self.submits[self.done_submits]
            self.done_submits += 1
            argv = [str(a).replace(W.M.CLUSTER_INC_DIR, str(INC)) for a in sub["argv"]]
            idx = next((i for i, a in enumerate(argv) if os.path.basename(a) in BUILD_SCRIPTS), None)
            if idx is None:
                self.jobs.append((sub["name"], argv, None))
                continue
            script, args = os.path.basename(argv[idx]), argv[idx + 1:]
            rc = None
            if script == "run_inc2_build.sh":
                mod, rest = args[0].rsplit(".", 1)[-1], args[1:]
                if mod == "baseline":
                    rc = baseline_runner(rest + ["--testing"])
                    exp = rest[rest.index("--exp") + 1]
                    drive(exp)
                elif mod == "pilot4":
                    # Stage A needs pilot_v3's real bins: the world's experiment record (named in the docstring)
                    self.experiment("pilot_v4", typ="chain", done=True)
                    rc = 0
                elif mod == "stream":
                    extra = ["--testing"] if rest and rest[0] == "init" else []
                    rc, out, _err = call(ST.main, rest + extra, deps=DEPS, clock=self.clock)
                    for ln in out.splitlines():
                        if ln.startswith("[inc2.stream] built experiment "):
                            drive(ln.split()[-1])
                elif mod == "splits":
                    rc = S2.main(rest + ["--testing", "--procs", "1"])
                elif mod == "base3":
                    rc = B3.main(rest + ["--testing"])
            elif script == "run_inc2_stream.sh":
                extra = ["--testing"] if args and args[0] == "bootstrap" else []
                rc = SS.main(args + extra + ["--procs", "1"])
            if rc is None:
                self.jobs.append((sub["name"], args, None))       # a collect job: never runs in this world
                continue
            self.jobs.append((sub["name"], args, rc))
            self.job_done(sub["name"], state="COMPLETED" if rc == 0 else "FAILED")

    def executed(self):
        return [(e.get("lever"), e.get("lane")) for e in self.events("executed")]

    def run_until(self, pred, max_ticks=60, note=""):
        for _ in range(max_ticks):
            self.tick()
            self.process_jobs()
            self.finish_lab()
            if pred():
                return True
        print("      (run_until %s: gave up after %d ticks; executed %s)" % (note, max_ticks, self.executed()[-8:]))
        st = self.state() or {}
        for ln, x in sorted((st.get("lanes") or {}).items()):
            print("      lane %s: %s" % (ln, json.dumps({k: x.get(k) for k in ("phase", "hold", "busy", "item")},
                                                       default=str)[:600]))
        print("      last error: %s" % json.dumps(st.get("last_error"), default=str)[:800])
        return False

    def lever_count(self, lever):
        return sum(1 for lv, _ln in self.executed() if lv == lever)


def set_clocks(w):
    """The stream and step1_stream read the simulated clock (admission times,
    ages, the ledger), so that the autopilot's day counts are the world's."""
    SS.CLOCK = lambda: datetime.datetime.fromtimestamp(w.clock(), tz=datetime.timezone.utc)


def stream_run(w, argv, who=None):
    old = os.environ.get("INCAP_DECIDED_BY")
    try:
        if who:
            os.environ["INCAP_DECIDED_BY"] = who
        return call(ST.main, argv, deps=DEPS, clock=w.clock)
    finally:
        if old is None:
            os.environ.pop("INCAP_DECIDED_BY", None)
        else:
            os.environ["INCAP_DECIDED_BY"] = old



def stage_r0(w):
    stage("R0 + R1 by the autopilot: baselines, verdicts, Stage A, the stream, its arm, Stage C; Step 1's one-time jobs")
    want = ["L23", "L23"] + ["L23B"] * 5 + ["LV", "LV", "L25", "LV", "LI", "LA", "L28", "LC"]

    def r0_done():
        maint = [lv for lv, ln in w.executed() if ln == "MAINT"]
        data = [lv for lv, ln in w.executed() if ln == "DATA"]
        return maint[:len(want)] == want and data.count("L17") >= 3
    ok = w.run_until(r0_done, max_ticks=120, note="R0")
    maint = [lv for lv, ln in w.executed() if ln == "MAINT"]
    check("the autopilot proposed and ran R0 in order, one MAINT item at a time: %s" % maint,
          ok and maint[:len(want)] == want, w.executed())
    verbs = [a[0] for n, a, rc in w.jobs if a and a[0] in ("bootstrap", "knowntruth", "backfill")]
    check("... and Step 1's one-time jobs in the DATA lane, in order (L17 bootstrap, knowntruth, backfill), each exit 0",
          verbs[:3] == ["bootstrap", "knowntruth", "backfill"]
          and all(rc == 0 for n, a, rc in w.jobs if a and a[0] in ("bootstrap", "knowntruth", "backfill")),
          [(n, a[:2], rc) for n, a, rc in w.jobs])
    data = [lv for lv, ln in w.executed() if ln == "DATA"]
    check("R1 before collection: the DATA lane ran Step 1's three one-time jobs before any discovery or fetch: %s"
          % data[:5], data[:3] == ["L17", "L17", "L17"], data)
    fails = [(n, a[:3], rc) for n, a, rc in w.jobs if rc not in (0, None)]
    check("every job the autopilot submitted ran to exit 0 on the real code", not fails, fails)
    bv2 = rj(D.Paths("b_v2").exp_json, {})
    st = rj(D.Paths("b_v2").state, {})
    check("B_v2 (milestone 0): inc2.baseline built 5 cold seeds on base_v2 (role b_v2, finals dev/imageweeds/test), "
          "run to done on the pinned driver", bv2.get("seeds") == [0, 1, 2, 3, 4] and bv2.get("role") == "b_v2"
          and bv2.get("final_exams") == ["dev", "imageweeds", "test"] and st.get("done") is True,
          {k: bv2.get(k) for k in ("seeds", "role", "final_exams")})
    cap = rj(INC / "capacity" / "capacity_v1.json", {})
    check("the capacity decision (L-4, dev only) keeps n640: no arm beats it by 2 pooled sd",
          cap.get("chosen_arm") == "n640" and cap.get("chosen_exp") == "b_v2", {k: cap.get(k) for k in
                                                                                ("chosen_arm", "qualifying")})
    led = jl(ST.StreamPaths(w.sid).ledger)
    ev = [e["event"] for e in led]
    fz = [e for e in led if e["event"] == "feasibility" and e.get("phase") == "read"]
    check("the stream exists (init, prospective record, arm adopted) and Stage C read M as feasible",
          ev[:2] == ["init", "prospective"] and "arm" in ev and fz and (fz[-1].get("result") or {}).get("m_feasible"),
          (ev, fz and fz[-1].get("result")))
    summ = rj(ST.StreamPaths(w.sid).summary, {})
    check("the stream's M is ceil(0.10 x |base_v2|) = %d" % WORLD["M"], summ.get("M") == WORLD["M"], summ.get("M"))


def queue_rows():
    return SS.load_queue(SS.Layout())


def stage_b0000(w):
    stage("R1: batch b0000 (per-box admission over the v1 pool), its guards and holds")
    status = rj(SS.Layout().status, {})
    check("step1_stream/status.json: bootstrap, knowntruth and backfill have run",
          all((status.get("one_time") or {}).get(k) for k in ("bootstrap", "knowntruth", "backfill")),
          status.get("one_time"))
    rows = queue_rows()
    rows = rows[0] if isinstance(rows, tuple) else rows
    by_key = {r["key"]: r for r in rows}
    print("      queue: %d rows; admissions %s; holds %s" % (len(rows), dict(collections.Counter(
        r.get("admission") for r in rows)), dict(collections.Counter(tuple(r.get("holds") or ()) for r in rows))))
    WORLD["queue0"] = rows
    masked = [r for r in rows if r.get("admission") == "masked"]
    mk = [r for r in masked if r["key"].startswith("harv_veto__veto_mask")]
    check("the masked veto image is queued as 'masked', holding funnel_F9 (P8)",
          len(mk) == 1 and "funnel_F9" in (mk[0].get("holds") or []), [(r["key"], r.get("holds")) for r in masked])
    if mk:
        r = mk[0]
        with Image.open(r["image"]) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.int64)
        with Image.open(r["unmasked_image"]) as im:
            u = np.asarray(im.convert("RGB"), dtype=np.int64)
        lab = C.read_yolo(r["label"])
        H, Wd = a.shape[:2]
        kept_ok = True
        for b in lab:
            x0, x1 = int((b[1] - b[3] / 2) * Wd) + 1, int((b[1] + b[3] / 2) * Wd) - 1
            y0, y1 = int((b[2] - b[4] / 2) * H) + 1, int((b[2] + b[4] / 2) * H) - 1
            kept_ok &= bool((a[y0:y1, x0:x1] == u[y0:y1, x0:x1]).all())
        diff = (a != u).any(-1)
        mean = u.reshape(-1, 3).mean(0).round().astype(np.int64)
        check("its PNG keeps the kept boxes' pixels and fills the conflict box with the image's mean colour "
              "(masked_area_frac %.3f)" % float(r.get("masked_area_frac") or 0),
              kept_ok and diff.any() and (np.abs(a[diff] - mean).max() <= 1) and len(lab) == 2
              and r["image"].endswith(".png"), (kept_ok, int(diff.sum())))
    planted = [k for k in by_key if "copy_hflip_test" in k or "copy_crop_dev" in k]
    adm = {r["key"]: r for r in jl(SS.Layout().batch_dir("b0000") / "admission.jsonl")}
    ref = {k: (adm.get(k) or {}).get("refusal") for k in WORLD["copies"]}
    check("the planted evaluation copies never reach the queue: b0000 refuses the mirrored test image "
          "(near_eval_variant, GuardV2) and the 15 %% dev crop (near_eval_embed, the embedding scan): %s" % ref,
          not planted and ref.get("harv_veto__copy_hflip_test") == "near_eval_variant"
          and ref.get("harv_veto__copy_crop_dev") == "near_eval_embed", (planted, ref))
    hq = status.get("human_queue") or {}
    check("the overlap veto image goes to the human queue (refused_overlap)",
          (status.get("admission") or {}).get("refused_overlap", 0) >= 1 and hq.get("rows", 0) >= 1,
          (status.get("admission"), hq))
    base_keys = WORLD["base_b_keys"]
    check("base B's images are never queued", not (set(by_key) & base_keys), sorted(set(by_key) & base_keys)[:5])
    tgt = [r for r in rows if r["source"] in SOURCES]
    lic = [r for r in tgt if "licence" in (r.get("holds") or [])]
    check("every target-source row waits for a licence decision (no card licence; P6), none for the copy scan "
          "(the funnel's calibration served it inside backfill)", tgt and len(lic) == len(tgt)
          and not any("h6_scan" in (r.get("holds") or []) for r in tgt), collections.Counter(
              tuple(r.get("holds") or ()) for r in tgt))
    per = collections.Counter(r["source"] for r in tgt if r.get("admission") == "whole")
    check("each target source keeps >= M whole-admitted rows after base B: %s" % dict(per),
          all(per.get(s, 0) >= WORLD["M"] for s in SOURCES), dict(per))
    w.tick()
    w.process_jobs()
    held = summary(w).get("held") or {}
    n_lic = sum(1 for r in rows if "licence" in (r.get("holds") or []))
    check("queue_summary.json counts the %d licence-held rows the snapshot ships to the autopilot" % n_lic,
          (held.get("licence") or {}).get("rows") == n_lic, held)


def release_keys(w, source, n, why):
    """A person resolves the licence of n images of one source (inc2.stream
    release --hold licence --keys, decided by a person)."""
    done = WORLD.setdefault("released", set())      # step1_stream's view keeps the hold; the stream records the release
    rows = [r for r in queue_rows() if r["source"] == source and r.get("admission") == "whole"
            and "licence" in (r.get("holds") or []) and r["key"] not in done]
    keys = sorted(r["key"] for r in rows)[:n]
    done.update(keys)
    kf = TMP / ("release_%s.txt" % source)
    kf.write_text("".join(k + "\n" for k in keys))
    rc, _o, err = stream_run(w, ["release", "--stream", w.sid, "--hold", "licence", "--keys", str(kf),
                                 "--reason", why], who=W.OWNER)
    return rc, keys, err



def summary(w):
    return rj(ST.StreamPaths(w.sid).summary, {})


def inc_sources(w, inc):
    return collections.Counter(r["source"] for r in C.read_manifest(ST.StreamPaths(w.sid).inc_manifest(inc)))


def stage_segment1(w):
    stage("R2: a person resolves the licences of M good and M bad images; the autopilot cuts, builds, runs and "
          "commits segment 1")
    M = WORLD["M"]
    for src in ("harv_good", "harv_bad"):
        rc, keys, err = release_keys(w, src, M, "licence confirmed by the owner (test)")
        check("inc2.stream release --hold licence --keys (a person): %d %s rows" % (len(keys), src),
              rc == 0 and len(keys) == M, (rc, err[-300:]))
    q = summary(w)
    check("queue_summary.json after the releases: Q = 2M = %d eligible target images, two sources" % (2 * M),
          (q.get("eligible") or {}).get("images") == 2 * M
          and set((q.get("eligible") or {}).get("by_source") or {}) == {"harv_good", "harv_bad"}, q.get("eligible"))
    rc, out, _e = stream_run(w, ["cut", "--stream", w.sid])
    plan = json.loads(out[out.index("{"):]) if "{" in out else {}
    check("inc2.stream cut (the dry run) would cut exactly M images from one source",
          rc == 0 and [p["images"] for p in plan.get("would_cut", [])] == [M]
          and len(plan["would_cut"][0]["sources"]) == 1, plan.get("would_cut"))
    w.advance(8 * W.DAY)                       # the oldest eligible row is 8 days old: D22's staleness rule
    ok = w.run_until(lambda: w.lever_count("L19") >= 1, max_ticks=40, note="segment 1")
    seg = "%s_s001" % w.sid
    tr = [lv for lv, ln in w.executed() if ln == "TRAIN"]
    check("the autopilot ran L18 (cut + build of %s) and then L19 (commit) in the TRAIN lane" % seg,
          ok and tr[:2] == ["L18", "L19"], w.executed()[-6:])
    l18 = [e for e in w.events("proposed") if e.get("lever") == "L18"]
    check("L18's argv: build --stream SID --k 2 --exp SID_s001 (K = Q // M)",
          l18 and l18[0]["argv"][-6:] == ["--stream", w.sid, "--k", "2", "--exp", seg], l18 and l18[0]["argv"])
    defn = rj(D.Paths(seg).exp_json, {})
    try:
        D.validate_definition(json.loads(json.dumps(defn)))
        D.check_definition_data(defn)
        valid = True
    except D.DriverError as e:
        valid = str(e)
    check("the segment's exp.json passes the pinned validate_definition and check_definition_data; chain, full "
          "replay, net gate, truth on, finals dev + imageweeds",
          valid is True and defn.get("type") == "chain" and defn.get("replay_mode") == "full"
          and defn.get("gate") == {"flips_mode": "net"} and defn.get("truth") is True
          and defn.get("final_exams") == ["dev", "imageweeds"], valid)
    steps = [st["name"] for st in defn.get("steps") or []]
    kinds = {i: inc_sources(w, i) for i in steps}
    check("two increments of exactly M, each from one source: %s" % {i: dict(k) for i, k in kinds.items()},
          len(steps) == 2 and all(sum(k.values()) == M and len(k) == 1 for k in kinds.values()), kinds)
    led = jl(ST.StreamPaths(w.sid).ledger)
    com = [e for e in led if e["event"] == "commit" and e.get("exp") == seg]
    disp = (com[-1].get("dispositions") if com else {}) or {}
    good = next((i for i, k in kinds.items() if "harv_good" in k), None)
    bad = next((i for i, k in kinds.items() if "harv_bad" in k), None)
    check("commit (Protocol v3, inc2.gate3): the good increment ACCEPTed, the planted bad one REJECTed as data",
          disp.get(good) == "accepted" and disp.get(bad) == "data", disp)
    g3 = rj(INC / seg / "gate3.json", {})
    check("... and the commit records gate3.json by sha256", com and (com[-1].get("gate") or {}).get("gate3", {})
          .get("sha256") == C.sha256_file(INC / seg / "gate3.json") and g3.get("steps"), com and com[-1].get("gate"))
    st_ = ST.Stream(w.sid, deps=DEPS, clock=w.clock, quiet=True)
    f = st_.load()
    pool = f.current_pool()
    prow = C.read_manifest(pool["path"])
    bad_keys = set(f.inc_rows(bad)) if bad else set()
    qkeys = {x["key"] for x in f.quarantine}
    check("P_1 = P_0 + the good increment (the planted bad increment rolled back: never in the pool)",
          pool["name"] == "P_1" and len(prow) == len(WORLD["base_v2"]) + M
          and not (bad_keys & {r["key"] for r in prow}), (pool["name"], len(prow)))
    check("the bad increment's images are quarantined by dHash (never cut again)", bad_keys and bad_keys <= qkeys,
          (len(bad_keys), len(qkeys)))
    WORLD["inc"] = {"good": good, "bad": bad}
    WORLD["P1"] = pool["name"]



def ledger_events(w, event, **match):
    return [e for e in jl(ST.StreamPaths(w.sid).ledger) if e["event"] == event
            and all(e.get(k) == v for k, v in match.items())]


def stage_measure_arms(w):
    stage("MAINT: the measurement arms (m832, s1024; y26l640, y26m640, l640), proposed by the platform once R0 is "
          "complete; the stream's arm stays the capacity decision's")
    exps = {b["exp"]: (b["arm"], RC.ARMS[b["arm"]]["imgsz"], RC.ARMS[b["arm"]].get("batch", RC.COMMON["batch"]))
            for b in w.dom["baselines"]["items"] if b.get("measure") and not b.get("requires")}
    check("the domain's measurement arms: m832 (832, batch 16), s1024 (1024, 32), y26l640, y26m640, l640 (640, 32)",
          list(exps.items()) == [("b_v2_m832", ("m832", 832, 16)), ("b_v2_s1024", ("s1024", 1024, 32)),
                                 ("b_v2_y26l640", ("y26l640", 640, 32)), ("b_v2_y26m640", ("y26m640", 640, 32)),
                                 ("b_v2_l640", ("l640", 640, 32))], exps)
    ok = w.run_until(lambda: all(rj(D.Paths(e).state, {}).get("done") for e in exps), max_ticks=80,
                     note="measurement arms")
    pro = [e for e in w.events("proposed") if e.get("lane") == "MAINT"]
    idx = {e.get("child_exp"): i for i, e in enumerate(pro) if e.get("lever") == "L23B"}
    lc = next((i for i, e in enumerate(pro) if e.get("lever") == "LC"), None)
    check("the platform proposed each measurement arm once, as L23B (--arm ID --role capacity), after R0's last "
          "item (LC reading Stage C)", ok and all(sum(1 for e in pro if e.get("child_exp") == x) == 1 for x in exps)
          and lc is not None and all(idx.get(x, -1) > lc for x in exps)
          and all(pro[idx[x]]["argv"][-4:] == ["--arm", a[0], "--role", "capacity"] for x, a in exps.items()),
          [(e.get("lever"), e.get("child_exp")) for e in pro])
    good = {}
    for x, (aid, imgsz, batch) in exps.items():
        d = rj(D.Paths(x).exp_json, {})
        good[x] = (d.get("role") == "capacity" and (d.get("arm") or {}).get("id") == aid
                   and d.get("seeds") == [0, 1, 2] and d.get("final_exams") == ["dev", "imageweeds"]
                   and (d.get("base") or {}).get("recipe", {}).get("imgsz") == imgsz
                   and (d.get("base") or {}).get("recipe", {}).get("batch") == batch
                   and rj(D.Paths(x).state, {}).get("done") is True
                   and not list(D.Paths(x).root.glob("runs/*/scores/test.json")))
    check("inc2.baseline built them on base_v2 (role capacity, 3 seeds, finals dev/imageweeds: no test outside a "
          "milestone, P10; the arm's imgsz and batch) and the pinned driver ran them to done", all(good.values()), good)
    cap = rj(INC / "capacity" / "capacity_v1.json", {})
    arms = [e for e in ledger_events(w, "arm")]
    check("the stream's arm is unchanged: one arm line (the capacity decision's %s); the decision names neither "
          "measurement arm" % cap.get("chosen_arm"),
          len(arms) == 1 and (arms[0].get("arm") or {}).get("id") == cap.get("chosen_arm") == "n640"
          and not any(x in json.dumps(cap) for x in exps), [(e.get("arm") or {}).get("id") for e in arms])


def stage_native_rescore(w):
    stage("MAINT: each done measurement arm's native-resolution rescore (L23N), once; its failure is a card")
    exps = tuple(b["exp"] for b in w.dom["baselines"]["items"] if b.get("measure") and b.get("native") is not False)

    def ended():
        return sum(1 for n, _a, rc in w.jobs if n.startswith("inc_build_native_") and rc is not None) >= len(exps)
    ok = ended() or w.run_until(ended, max_ticks=30, note="native rescore")
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23N"]
    ref = w.dom["capacity"]["native"]["reference_exp"]
    check("the platform proposed each done measurement arm's rescore once, as L23N (rescore-native --exp E "
          "--reference %s), each one run_inc2_build.sh job under its own name" % ref,
          ok and sorted((e.get("argv") or [])[-3] for e in pro) == sorted(exps)
          and all((e.get("argv") or [])[-5:] == ["rescore-native", "--exp", (e.get("argv") or [])[-3], "--reference",
                                                 ref] for e in pro)
          and sorted(n for n, _a, _rc in w.jobs if n.startswith("inc_build_native_"))
          == sorted("inc_build_native_%s" % x for x in exps), [(e.get("argv") or [])[-5:] for e in pro])
    runs = [(n, rc) for n, _a, rc in w.jobs if n.startswith("inc_build_native_")]
    written = [str(p) for x in exps for p in D.Paths(x).root.glob("runs/*/scores/*@*")]
    check("the real rescore-native refused this world's stand-in weights (exit 1: they cannot be loaded as a "
          "detector), writing no native score", runs and all(rc == 1 for _n, rc in runs) and not written,
          (runs, written))
    st = w.state()
    cards = [c["title"] for c in st.get("cards") or [] if "Native-resolution rescore" in c.get("title", "")]
    held = [e for e in w.events("lane_held") if "L23N" in str(e.get("hold"))]
    check("each failure is a card, never a pause or a held lane, and stays failed (not proposed again)",
          sorted(cards) == sorted("Native-resolution rescore of %s failed (L23N)" % x for x in exps)
          and w.config().get("enabled") is True and not held and len(pro) == len(exps)
          and all(((st.get("stage") or {}).get("r0") or {}).get("native_%s" % b["id"]) == "failed"
                  for b in w.dom["baselines"]["items"] if b.get("measure") and b.get("native") is not False),
          (cards, held, len(pro)))


def stage_e1_base3(w):
    stage("MAINT: E1 (2026-10-03): its first arm requires splits v3, so the platform proposes the base v3 build "
          "(L23V) once; this world has no dataset registry, so the real inc2.base3 build refuses: a card, the "
          "stream runs on, and no E1 arm is built")
    e1 = [b for b in w.dom["baselines"]["items"] if b.get("requires") == "base3"]

    def ended():
        return any(n == "inc_build_base3_v3" and rc is not None for n, _a, rc in w.jobs)
    ok = ended() or w.run_until(ended, max_ticks=30, note="base v3 build")
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23V"]
    runs = [(n, a, rc) for n, a, rc in w.jobs if n == "inc_build_base3_v3"]
    check("the platform proposed the base v3 build once (L23V inc2.base3 build --stream %s), one run_inc2_build.sh "
          "job under its own name, and the real build refused (exit 1: no dataset registry in this world)" % w.sid,
          ok and len(pro) == 1 and (pro[0].get("argv") or [])[-3:] == ["build", "--stream", w.sid]
          and len(runs) == 1 and runs[0][2] == 1 and not (INC / "splits" / "v3" / "summary.json").exists(),
          ([(e.get("argv") or [])[-4:] for e in pro], runs))
    w.run_until(lambda: False, max_ticks=3, note="settle")
    st = w.state()
    cards = [c["title"] for c in st.get("cards") or [] if "Base v3 build" in c.get("title", "")]
    held = [e for e in w.events("lane_held") if "L23V" in str(e.get("hold"))]
    check("its failure is one card, never a pause or a held lane; it stays failed (not proposed again) and neither "
          "E1 arm is built",
          cards == ["Base v3 build (splits v3, E1) failed (L23V)"] and w.config().get("enabled") is True and not held
          and len([e for e in w.events("proposed") if e.get("lever") == "L23V"]) == 1
          and ((st.get("stage") or {}).get("r0") or {}).get("base3") == "failed"
          and not [e for e in w.events("proposed") if e.get("lever") == "L23B" and e.get("child_exp")
                   in [b["exp"] for b in e1]], (cards, held, ((st.get("stage") or {}).get("r0") or {}).get("base3")))


def d28_now(w):
    """The autopilot's last D28 diagnosis (the campaign's diagnoses.json)."""
    doc = rj(W.S.StreamPaths(str(w.lab), w.domain).diagnoses(W.NAME), {}) or {}
    return next((d for d in doc.get("diagnoses") or [] if d.get("id") == "D28"), {})


def stage_leak_and_audit(w):
    stage("the leaking veto source (D28 -> L24)")
    q = summary(w)
    stop = [lv for lv, ln in w.executed() if ln == "STOP"]
    check("D28: the veto source (2 of its 4 images refused as evaluation copies) was quarantined by L24 once the "
          "stream existed, with no failed L24 before it", "harv_veto" in (q.get("quarantined_sources") or {})
          and stop.count("L24") >= 1 and not any(e.get("lever") == "L24" for e in w.events("failed")),
          (q.get("quarantined_sources"), stop, [e.get("reason") for e in w.events("failed")][-3:]))
    row = (rj(INC / "step1_stream" / "status.json", {}).get("per_source") or {}).get("harv_veto") or {}
    leak = next((h for h in (d28_now(w).get("detail") or {}).get("leaks") or [] if h.get("source") == "harv_veto"),
                {})
    dv = (leak.get("verdict") or {}).get("dhash") or {}
    check("D28-v2: b0000 weighed the mirrored test image by its pair cosine (the copy scanner's descriptors), "
          "status.json folds it per source, and D28 reads it: a hit at or above the copy threshold is a leak",
          row.get("decision:near_eval_variant") == 1 and row.get("eval_hits_scored") == 1
          and row.get("eval_hit_pair_cos") and row["eval_hit_pair_cos"][0] >= row.get("eval_hit_copy_threshold")
          and dv.get("copy_hits") == 1 and not dv.get("fail_closed") and dv.get("verdict") == "leak",
          (row, dv))


def stage_milestone1(w):
    stage("MAINT: milestone 1 (L20, 5 cold seeds on P_1) compared 5 v 5 on dev with milestone 0 (LC)")
    n20 = w.lever_count("L20")
    w.advance(31 * W.DAY)                      # 30 days since the first ACCEPT: D24's day trigger
    ok = w.run_until(lambda: w.lever_count("L20") > n20 and ledger_events(w, "milestone", phase="compare"),
                     max_ticks=40, note="milestone 1")
    l20 = [e for e in w.events("proposed") if e.get("lever") == "L20"]
    check("D24 (30 days since the first ACCEPT, counted past the last stream write) -> L20; then LC compares it",
          ok and l20 and "D24" in (l20[-1].get("trigger") or []), [e.get("trigger") for e in l20])
    m1 = "%s_m001" % w.sid
    defn = rj(D.Paths(m1).exp_json, {})
    check("milestone 1 is an inc2.baseline build: 5 seeds on P_1, role milestone, finals dev/imageweeds/test",
          defn.get("seeds") == [0, 1, 2, 3, 4] and defn.get("role") == "milestone"
          and defn.get("final_exams") == ["dev", "imageweeds", "test"]
          and pathlib.Path(defn["base"]["source_manifest"]).name.startswith("P_1."),
          {k: defn.get(k) for k in ("seeds", "role", "final_exams")})
    cmp_ = ledger_events(w, "milestone", phase="compare", exp=m1)
    c = cmp_[-1] if cmp_ else {}
    check("5 v 5 on dev against milestone 0 (b_v2): helps, no rollback (p %s)" % c.get("perm_p"),
          c.get("verdict") == "helps" and c.get("compared_with") == "b_v2" and not c.get("rollback_recommended"), c)
    inc_rec = rj(ST.StreamPaths(w.sid).milestone_dir(1) / "incumbent.json", {})
    check("the chain incumbent's secondary scores were written by inc2.baseline secondary and run",
          inc_rec.get("status") == "submitted" and D.Paths(m1).score(ST.SECONDARY_RUN, "test").is_file(), inc_rec)


def stage_segment2(w):
    stage("R2: a person resolves M more good and M 'sneaky' licences; segment 2 accepts both increments")
    M = WORLD["M"]
    for src in ("harv_good", "harv_sneaky"):
        rc, keys, err = release_keys(w, src, M, "licence confirmed by the owner (test)")
        check("the release of %d more %s rows" % (len(keys), src), rc == 0 and len(keys) == M, err[-300:])
    n19 = w.lever_count("L19")
    ok = w.run_until(lambda: w.lever_count("L19") > n19, max_ticks=40, note="segment 2")
    seg = "%s_s002" % w.sid
    com = ledger_events(w, "commit", exp=seg)
    disp = com[-1].get("dispositions") if com else {}
    steps = [st["name"] for st in rj(D.Paths(seg).exp_json, {}).get("steps") or []]
    srcs = {i: inc_sources(w, i) for i in steps}
    check("the autopilot cut, built and committed %s: two single-source increments of M (good, sneaky), both "
          "ACCEPTed by the chain" % seg, ok and sorted(tuple(k) for k in srcs.values()) ==
          [("harv_good",), ("harv_sneaky",)] and all(sum(k.values()) == M for k in srcs.values())
          and disp == {i: "accepted" for i in steps}, (srcs, disp))
    WORLD["inc"]["good2"] = next((i for i, k in srcs.items() if "harv_good" in k), None)
    WORLD["inc"]["sneaky"] = next((i for i, k in srcs.items() if "harv_sneaky" in k), None)
    pool = ST.Stream(w.sid, deps=DEPS, clock=w.clock, quiet=True).load().current_pool()
    check("P_2 = P_1 + both increments", pool["name"] == "P_2"
          and len(C.read_manifest(pool["path"])) == len(WORLD["base_v2"]) + 3 * M, pool)


def stage_milestone2_rollback(w):
    stage("MAINT: milestone 2 hurts (5 v 5) -> rollback to P_1 (L21) -> bisect (L27, one arm per suspect "
          "increment) -> the sneaky increment quarantined, the good one returned")
    w.advance(31 * W.DAY)
    ok = w.run_until(lambda: len(ledger_events(w, "bisect", phase="decide")) >= 1 and all(
        i in {k for e in ledger_events(w, "bisect", phase="decide") for k in (e.get("decisions") or {})}
        for i in (WORLD["inc"]["sneaky"], WORLD["inc"]["good2"])), max_ticks=80, note="milestone 2")
    m2 = "%s_m002" % w.sid
    c = (ledger_events(w, "milestone", phase="compare", exp=m2) or [{}])[-1]
    check("milestone 2 vs milestone 1 on dev: hurts (one-sided permutation p %s <= 0.025, lower mean), rollback "
          "recommended to P_1" % c.get("perm_p"), c.get("verdict") == "hurts" and c.get("rollback_recommended")
          and c.get("to_pool") == "P_1" and c.get("compared_with") == "%s_m001" % w.sid, c)
    # the measurement arms' builds and rescores (L23B, L23N; E1's L23V, L23E: record only, R0 long complete) may
    # take the idle MAINT lane between them
    maint = [lv for lv, ln in w.executed() if ln == "MAINT" and lv not in ("L23B", "L23N", "L23V", "L23E")]
    tail_ = maint[maint.index("L21") - 2:] if "L21" in maint else maint
    lcs = [e for e in w.events("proposed") if e.get("lever") == "LC"]
    check("the autopilot ran L20, LC, then L21 (D25), L27 (bisect) and LC (compare --exp on a bisect arm), and no "
          "milestone while the rollback was pending: %s" % tail_, ok and tail_ == ["L20", "LC", "L21", "L27", "LC"]
          and lcs and lcs[-1]["argv"][-1].startswith("%s_b" % w.sid), (maint, lcs and lcs[-1]["argv"][-3:]))
    rb = (ledger_events(w, "rollback") or [{}])[-1]
    inc = WORLD["inc"]
    check("rollback: the pool pointer is P_1 again; both increments accepted since milestone 1 are suspect",
          rb.get("to") == "P_1" and set(rb.get("suspect") or []) == {inc["sneaky"], inc["good2"]}, rb)
    bis = ledger_events(w, "bisect", phase="build")
    arms = (bis[-1].get("arms") if bis else {}) or {}
    check("bisect built one cold arm per suspect increment (%s), each run by the autopilot's advances"
          % sorted(arms.values()), len(arms) == 2 and all(rj(D.Paths(e).state, {}).get("done") for e in arms.values()),
          arms)
    f = ST.Stream(w.sid, deps=DEPS, clock=w.clock, quiet=True).load()
    dec = {}
    for e in ledger_events(w, "bisect", phase="decide"):
        dec.update(e.get("decisions") or {})
    sk, gk = set(f.inc_rows(inc["sneaky"])), set(f.inc_rows(inc["good2"]))
    check("bisect: the sneaky arm hurts against milestone 1's seeds -> quarantined; the good arm helps -> returned "
          "to the queue", dec.get(inc["sneaky"]) == "hurts" and dec.get(inc["good2"]) == "helps"
          and all(f.keys[k]["status"] == "quarantined" for k in sk)
          and all(f.keys[k]["status"] == "returned_bisect" for k in gk) and f.current_pool()["name"] == "P_1",
          (dec, f.current_pool()["name"]))
    q = summary(w)
    check("queue_summary: X4 is not raised (the bisection separated the suspect increments); the returned good "
          "images are eligible again", (q.get("x4") or {}).get("raised") is False
          and ((q.get("eligible") or {}).get("by_source") or {}).get("harv_good", 0) >= WORLD["M"],
          (q.get("x4"), q.get("eligible")))


def stage_report(w):
    stage("reports: inc2.stream_report --stream")
    rc, _o, err = call(SR2.main, ["--stream", w.sid])
    rep = rj(ST.StreamPaths(w.sid).report_json, {})
    tl = [(t.get("increment"), t.get("disposition")) for t in rep.get("timeline") or []]
    inc = WORLD["inc"]
    check("the stream report's timeline: bad data, good accepted, sneaky accepted then suspect/quarantined",
          rc == 0 and (inc["bad"], "data") in tl and (inc["good"], "accepted") in tl
          and any(i == inc["sneaky"] for i, _d in tl), (rc, tl, err[-300:]))
    tlm = {t.get("increment"): t for t in rep.get("timeline") or []}
    seg_rep = rj(INC / ("%s_s001" % w.sid) / "report.json", {})
    check("segment 1's report.json is still the stream's (its commit reading), not replaced by the snapshot's "
          "regeneration with the pinned report", (seg_rep.get("stream_commit") or {}).get("chosen") == "r0", sorted(seg_rep)[:12])
    check("... the sneaky increment reads 'accepted, then bisect_hurts' (its rollback and bisect are reported)",
          (tlm.get(inc["sneaky"]) or {}).get("status") == "bisect_hurts"
          and ((rep.get("yield") or {}).get("stream") or {}).get("harv_sneaky", {}).get("then_bisect_hurts") ==
          WORLD["M"], (tlm.get(inc["sneaky"]), (rep.get("yield") or {}).get("stream")))
    h = rep.get("headline") or {}
    md = ST.StreamPaths(w.sid).report_md
    text = md.read_text() if md.is_file() else ""
    check("its headline is the latest milestone's test (m002) with the gap to 0.90, and says the stream rolled back "
          "from it to P_1", h.get("milestone") == "%s_m002" % w.sid and h.get("gap_to_target") is not None
          and h.get("rolled_back") is True and h.get("current_pool") == "P_1" and "rolled back" in text, h)


def stage_f9(w):
    stage("P8: the funnel's F9 writes domain_dev.jsonl; serve-holds releases the masked veto row's funnel_F9 hold")
    mk = [r for r in queue_rows() if r["key"].startswith("harv_veto__veto_mask")]
    dd = INC / "step1_r1" / "domain_dev.jsonl"
    dd.parent.mkdir(parents=True, exist_ok=True)
    dd.write_text(json.dumps({"key": "harv_other__other_00"}) + "\n")
    before = (summary(w).get("held") or {}).get("funnel_F9") or {}
    rc = SS.main(["serve-holds", "--hold", "funnel_F9", "--procs", "1"])
    mk2 = [r for r in queue_rows() if r["key"].startswith("harv_veto__veto_mask")]
    check("serve-holds --hold funnel_F9 (the L17 scan-holds form): the masked row keeps only its licence hold",
          rc == 0 and mk and "funnel_F9" in (mk[0].get("holds") or []) and mk2
          and "funnel_F9" not in (mk2[0].get("holds") or []), (rc, mk2 and mk2[0].get("holds")))
    n0 = len(w.runs)
    for _ in range(4):                          # a tick that submits takes no snapshot: the next snapshot refreshes
        w.tick()
        w.process_jobs()
        refresh = [r for r in w.runs[n0:] if len(r) > 3 and r[2].endswith("inc2.stream") and r[3] == "summary"]
        if refresh:
            break
    after = (summary(w).get("held") or {}).get("funnel_F9") or {}
    check("the next snapshot refreshed the stream summary, which no stream verb had rewritten since Step 1 changed "
          "the queue (stream_remote.refresh_summary): funnel_F9 rows %s -> %s" % (before.get("rows"), after.get("rows")),
          len(refresh) == 1 and before.get("rows") == 1 and not after.get("rows"), (len(refresh), before, after))



CLEAN_REF = "7a1b2c3d-0000-4000-8000-00000000000a"
LEAK_REF = "7a1b2c3d-0000-4000-8000-00000000000b"


def weedai_source(cfg, CW, ref, n, dev_flip=False):
    """A fetched annotation-index (weedai) source of n painted images (Palmer
    amaranth and Lambsquarters boxes, WeedCOCO), plus a mirrored dev image
    when dev_flip; fetched by the real collect.fetch on a fake network.
    Returns the source id."""
    from weed_optimizer_framework.tools.collect import fetch as CF
    files, anns, imgs = {}, [], []
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}, {"id": 2, "name": "weed: chenopodium album"}]
    tmpd = TMP / ("weedai_%s" % ref[-1])
    for i in range(n):
        bx = TV.layout3([8, 12, 8])
        p = tmpd / ("p%d.png" % i)
        paint(p, bx, texture=False, fmt="png")
        files["images/p%d.png" % i] = p.read_bytes()
        imgs.append({"id": i, "file_name": "p%d.png" % i, "width": IMG_W, "height": IMG_H})
        for k, b in enumerate(bx):
            x, y = (b[1] - b[3] / 2) * IMG_W, (b[2] - b[4] / 2) * IMG_H
            anns.append({"id": 10 * i + k, "image_id": i, "category_id": 1 if b[0] == 8 else 2,
                         "bbox": [round(x, 2), round(y, 2), round(b[3] * IMG_W, 2), round(b[4] * IMG_H, 2)]})
    if dev_flip:
        dev = C.read_manifest(C.manifest_path("dev"))[0]
        with Image.open(dev["image"]) as im:
            flip = im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        fp = tmpd / "dev_flip.png"
        flip.save(fp)
        files["images/p%d.png" % n] = fp.read_bytes()
        imgs.append({"id": n, "file_name": "p%d.png" % n, "width": IMG_W, "height": IMG_H})
        anns.append({"id": 999, "image_id": n, "category_id": 1, "bbox": [10, 10, 30, 30]})
    files["weedcoco.json"] = json.dumps(dict(CW.coco_doc(imgs, cats, anns), agcontexts=[])).encode()
    z = CW.zip_bytes(files)
    base = cfg.provider("weedai")["base_url"]
    info = {"metadata": {"name": "Plants of farm %s" % ref[-1], "description": "boxes",
                         "license": "https://creativecommons.org/licenses/by/4.0/"},
            "agcontexts": [{"n_images": len(imgs), "category_statistics": {
                c["name"]: {"image_count": n, "bounding_box_count": 2 * n} for c in cats}}], "head_version": 1}
    net = CW.make_net({("GET", base + "/api/upload_info/" + ref): (200, info),
                       ("GET", base + "/code/download/%s.zip" % ref): (200, z, {"Content-Length": str(len(z))})})
    sid = "weedai_" + ref
    cands = TMP / ("lab_candidates_%s.json" % ref[-1])
    cands.write_text(json.dumps({"format": "collect-candidates/1", "candidates": [
        {"source_id": sid, "provider": "weedai", "ref": ref, "title": "Plants of farm %s" % ref[-1]}]}))
    CF.disk_free = lambda p: 10 ** 13          # the machine's free space is not the subject here (as test_collect_fetch)
    CF.fetch(cfg, sid, candidates_path=cands, net=net)
    return sid


def intake_cli(sid):
    """collect intake --source SID (the CLI); returns (exit status, batch, {file: decision reason})."""
    from weed_optimizer_framework.tools.collect import __main__ as CM
    rc, out, err = call(CM.main, ["intake", "--source", sid, "--testing"])
    batch = None
    for ln in out.splitlines():
        if ln.startswith("[collect] intake: "):
            batch = (json.loads(ln.split(": ", 1)[1]) or {}).get("batch")
    dec = {d.get("rel", "").split("/")[-1]: d.get("reason")
           for d in jl(INC / "intake" / str(batch) / "decisions.jsonl") if d.get("kind") == "image"}
    return rc, batch, dec


def stage_intake(w):
    stage("DATA: collected sources through the real collector (fetch, intake against LOCK v2), then the autopilot: "
          "L17 admits the clean source's batch; D28 quarantines the leaking one (L24) before any admission")
    import test_collect_world as CW
    import funnel_world as FWD
    fdir = INC / "funnel"
    FPR.write_pre_draw(fdir / "prereg_v1.json", HERE.parent / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json")
    (REPO / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE.parent.parent / "docs" / "FUNNEL_AUDIT.md", REPO / "docs" / "FUNNEL_AUDIT.md")
    FWD.build_taxonomy_cache(fdir, list(CW.CACHE_NAMES))
    cfg = CW.config()
    clean = weedai_source(cfg, CW, CLEAN_REF, 5)
    leak = weedai_source(cfg, CW, LEAK_REF, 5, dev_flip=True)
    check("collect fetch (the weedai provider on a fake network) staged both sources, content-addressed",
          all((INC / "intake" / "staging" / s_ / "fetch.json").is_file() for s_ in (clean, leak)))
    rc_c, batch_c, dec_c = intake_cli(clean)
    rc_l, batch_l, dec_l = intake_cli(leak)
    WORLD["intake_leak"] = (str(batch_l), leak)
    check("collect intake against the real LOCK v2 (real GuardV2): the clean source keeps its 5 images; in the "
          "other the mirrored dev image is refused (near_eval_variant) and 5 are kept",
          rc_c == 0 and rc_l == 0 and list(dec_c.values()).count("kept") == 5
          and dec_l.get("p5.png") == "near_eval_variant" and list(dec_l.values()).count("kept") == 5, (dec_c, dec_l))
    leak_i = rj(INC / "intake" / str(batch_l) / "summary.json", {}).get("source_leak") or {}
    check("the intake summary's source_leak (D28's input) holds the evaluation share 1/6",
          abs((leak_i.get("eval_share") or 0) - 1 / 6.0) < 1e-3 and leak_i.get("fires") is True, leak_i)
    eh_l = rj(INC / "intake" / str(batch_l) / "summary.json", {}).get("eval_hits") or {}
    eh_c = rj(INC / "intake" / str(batch_c) / "summary.json", {}).get("eval_hits") or {}
    row = (eh_l.get("per_source") or {}).get(leak) or {}
    hit = [d for d in jl(INC / "intake" / str(batch_l) / "decisions.jsonl") if d.get("reason") == "near_eval_variant"]
    check("D28-v2 at intake: the mirrored dev image was weighed before its copy was removed (pair cosine %s with "
          "the dev image it matched, by the v2 calibration's embedder; copy threshold %s); the clean batch records "
          "no hit" % (row.get("pair_cos"), eh_l.get("copy_threshold")),
          eh_l.get("hits") == 1 and eh_l.get("scored") == 1 and row.get("pair_cos") == [1.0]
          and eh_l.get("copy_threshold")
          == (rj(C2.LOCK_PATH, {}).get("embed_calibration_v2") or {}).get("cos_threshold")
          and len(hit) == 1 and hit[0].get("pair_cos") == 1.0 and (hit[0].get("match") or {}).get("split") == "dev"
          and not list((INC / "intake" / str(batch_l) / "images").glob("*p5.*"))
          and eh_c.get("hits") == 0 and eh_c.get("per_source") == {}, (eh_l, hit, eh_c))
    n0 = len(w.jobs)
    ok = w.run_until(lambda: any(a and a[0] == "admit" for n, a, rc_ in w.jobs[n0:]) and
                     leak in (summary(w).get("quarantined_sources") or {}), max_ticks=30, note="intake")
    adm = [(a, rc_) for n, a, rc_ in w.jobs[n0:] if a and a[0] == "admit"]
    check("the autopilot's DPIPE ran L17 admit --intake %s (the real step1_stream admit) and nothing for the "
          "leaking source; exit 0" % batch_c, ok and len(adm) == 1 and adm[0][0][:3] == ["admit", "--intake", batch_c]
          and adm[0][1] == 0, adm)
    rows = [r for r in queue_rows() if r.get("source") == clean]
    check("its rows are queued: whole-admitted, licensed by the collector's verdict, the copy scan served in the "
          "admit job (no hold left)", len(rows) == 5 and all(r.get("admission") == "whole" and not (r.get("holds") or [])
                                                             and r.get("licence") for r in rows),
          [(r["key"], r.get("admission"), r.get("holds"), r.get("licence")) for r in rows])
    for _ in range(4):                          # the next snapshot refreshes the summary after the admit batch
        w.tick()
        w.process_jobs()
        q = summary(w)
        if ((q.get("eligible") or {}).get("by_source") or {}).get(clean) == 5:
            break
    stop = [e for e in w.events("proposed") if e.get("lever") == "L24" and leak in " ".join(e.get("argv") or [])]
    check("D28 read the collector's source_leak and L24 quarantined the leaking source; the clean source's rows "
          "are eligible supply", stop and leak in (q.get("quarantined_sources") or {})
          and ((q.get("eligible") or {}).get("by_source") or {}).get(clean) == 5,
          (len(stop), q.get("quarantined_sources"), (q.get("eligible") or {}).get("by_source")))
    d = d28_now(w)
    dv = next((((h.get("verdict") or {}).get("dhash") or {}) for h in (d.get("detail") or {}).get("leaks") or []
               if h.get("source") == leak), {})
    check("D28-v2 judged the leaking source by the pair cosine its intake summary records (a hit at or above the "
          "copy threshold, not the bare dHash hit), and its summary states hits, confirmed hits, max pair cos, P and "
          "the verdict", dv.get("copy_hits") == 1 and not dv.get("fail_closed")
          and "%s (1 dHash hit(s) in 6 images, 1 confirmed (pair cos >= 0.8), max pair cos 1.000, P = " % leak
          in d.get("summary", "") and "-> leak" in d.get("summary", ""), (dv, d.get("summary")))


def stage_eval_hits(w):
    stage("D28-v2's sidecars: an intake batch committed before the amendment (no pair cosines) is weighed again "
          "by the platform (DR0 -> L17 eval-hits, the real step1_stream eval-hits) and D28 judges its source by them")
    batch_l, leak = WORLD["intake_leak"]
    sp = INC / "intake" / batch_l / "summary.json"
    keep = sp.read_bytes()
    sm = json.loads(keep)
    sp.write_text(json.dumps({k: v for k, v in sm.items() if k != "eval_hits"}, indent=1, sort_keys=True))
    n0 = len(w.jobs)
    ok = w.run_until(lambda: any(a and a[0] == "eval-hits" for n, a, rc_ in w.jobs[n0:]), max_ticks=20,
                     note="eval-hits")
    eh = [(a, rc_) for n, a, rc_ in w.jobs[n0:] if a and a[0] == "eval-hits"]
    side = rj(INC / "intake" / batch_l / "eval_hits.json", {})
    row = ((side.get("eval_hits") or {}).get("per_source") or {}).get(leak) or {}
    pro = [e for e in w.events("proposed") if e.get("lever") == "L17" and "eval-hits" in (e.get("argv") or [])]
    check("with the batch's own record gone, DR0 proposed L17 eval-hits and the platform ran the real step1_stream "
          "eval-hits once (exit 0): intake/<batch>/eval_hits.json weighs the mirrored dev image again from the "
          "staging blob (pair cos 1.0), summary.json and decisions.jsonl untouched",
          ok and len(eh) == 1 and eh[0][1] == 0 and len(pro) == 1 and side.get("batch") == batch_l
          and row.get("pair_cos") == [1.0] and json.loads(sp.read_text()).get("eval_hits") is None,
          (eh, [e.get("argv") for e in pro], side.get("eval_hits")))
    for _ in range(3):
        w.tick()
        w.process_jobs()
    d = d28_now(w)
    dv = next((((h.get("verdict") or {}).get("dhash") or {}) for h in (d.get("detail") or {}).get("leaks") or []
               if h.get("source") == leak), {})
    again = [a for n, a, rc_ in w.jobs[n0:] if a and a[0] == "eval-hits"]
    check("the snapshot ships the sidecar and D28 judges the source by it (a copy at the copy threshold, not the "
          "fail-closed fallback), citing it; eval-hits is not proposed again", dv.get("copy_hits") == 1
          and not dv.get("fail_closed") and len(again) == 1
          and any(c.get("artifact") == "intake/%s/eval_hits.json" % batch_l for c in d.get("cites") or []),
          (dv, len(again), [c.get("artifact") for c in d.get("cites") or []][-4:]))
    sp.write_bytes(keep)


def stage_invariants(w):
    stage("invariants: no dev / test / ImageWeeds image in any training manifest; every one passed inc2.train")
    l4 = [e for e in w.events("proposed") if e.get("lever") == "L4"]
    inc = WORLD["inc"]
    audited = " ".join(" ".join(e.get("argv") or []) for e in l4)
    check("D31 sent the data-disposed increment (%s) and the truth-'hurts' one (%s) to the L4 label audit (the "
          "policy's path patterns pin the cluster's INC_DIR, so here the audit itself is refused)"
          % (inc["bad"], inc["sneaky"]), "%s=" % inc["bad"] in audited and "%s=" % inc["sneaky"] in audited,
          [e.get("argv") for e in l4])
    check("the executor ran %d specs; every spec passed inc2.train.validate_spec and every training manifest "
          "check_manifest + guard_rows" % len(EX.runs), EX.runs and not EX.problems, EX.problems[:5])
    hits = {p: r["hits"] for p, r in EX.manifests.items() if r["hits"]}
    check("the independent never-train check (key, sha256, dHash within 6 bits under the 8 flips and rotations) "
          "finds no evaluation image in any of the %d training manifests" % len(EX.manifests),
          len(EX.manifests) >= 10 and not hits, hits)
    idx = EX.index or EvalIndex()
    sp = ST.StreamPaths(w.sid)
    files = sorted(sp.pool.glob("P_*.jsonl")) + sorted(sp.increments.glob("*.jsonl"))
    bad = {str(p.name): idx.hits(C.read_manifest(p)) for p in files}
    check("... nor in any pool P_s or increment the stream wrote (%d files)" % len(files),
          files and not any(bad.values()), {k: v for k, v in bad.items() if v})
    NOTES.append("training manifests checked: %d, of experiments %s" % (
        len(EX.manifests), ", ".join(sorted({r["exp"] for r in EX.manifests.values()}))))



def stage_deploy():
    stage("deploy: deploy/deploy_funnel.sh ships everything the stream and its replay gate need")
    import re
    import subprocess
    from weed_optimizer_framework.tools.inc_autopilot import executor as XE
    from weed_optimizer_framework.tools.inc_autopilot import stream_remote as SRM
    script = ROOT_PKG / "deploy" / "deploy_funnel.sh"
    syn = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    dry = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True, cwd=str(ROOT_PKG))
    check("bash -n and --dry-run (which copies nothing and needs no ssh) exit 0",
          syn.returncode == 0 and dry.returncode == 0, (syn.stderr[-300:], dry.stderr[-300:]))
    pkg = [ln.split(": ", 1)[1] for ln in dry.stdout.splitlines() if ln.startswith("package: ")]
    pre = [ln.split(": ", 1)[1] for ln in dry.stdout.splitlines() if ln.startswith("pre-flight: ")]

    def shipped(rel):
        return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in pkg)
    need = ["weed_optimizer_framework/tools/inc2", "weed_optimizer_framework/tools/collect",
            "run_inc2_build.sh", "run_inc2_job.sh", "run_inc2_splits.sh", "run_inc2_stream.sh", "run_inc_collect.sh"]
    need += ["weed_optimizer_framework/%s" % m for m in SRM.module_hashes()]
    missing = [n for n in need if not shipped(n)]
    check("every stream module (stream_remote.module_hashes, S23) and the five stream job scripts are shipped",
          not missing, missing)
    rs = sorted(set(XE.REPLAY_SCRIPTS.values()))
    check("every replay script the gate runs (executor.REPLAY_SCRIPTS) is shipped and in the local pre-flight, "
          "with the stream pipeline", all(shipped(t) for t in rs) and set(rs) <= set(pre)
          and "tests/test_stream_pipeline.py" in pre, (rs, pre))
    imp = re.compile(r"^\s*import (test_[a-z0-9_]+)|^\s*from (test_[a-z0-9_]+) import", re.M)
    closure, todo = set(), [t for t in pkg if t.startswith("tests/test_") and t.endswith(".py")]
    while todo:
        t = todo.pop()
        if t in closure:
            continue
        closure.add(t)
        for m in imp.finditer((ROOT_PKG / t).read_text()):
            todo.append("tests/%s.py" % (m.group(1) or m.group(2)))
    miss = sorted(t for t in closure if not shipped(t))
    check("every test module a shipped test imports is shipped too (%d test files)" % len(closure), not miss, miss)
    fixtures = {"tests/fixtures/collect", "tests/fixtures/inc_replay", "tests/fixtures/funnel"}
    check("the fixture directories the stream, collector and replay tests read are shipped",
          all(shipped(f) for f in fixtures), [f for f in fixtures if not shipped(f)])


def main_stages():
    stage_deploy()
    stage_v1()
    w = PipelineWorld()
    set_clocks(w)
    stage_splits_v2(w)
    stage_r0(w)
    stage_b0000(w)
    stage_segment1(w)
    stage_measure_arms(w)
    stage_leak_and_audit(w)
    stage_milestone1(w)
    stage_segment2(w)
    stage_milestone2_rollback(w)
    stage_report(w)
    stage_f9(w)
    stage_intake(w)
    stage_eval_hits(w)
    stage_native_rescore(w)
    stage_e1_base3(w)
    stage_invariants(w)
    if os.environ.get("STREAM_PIPELINE_COMMANDS"):
        # the commands the platform ran, in order: sbatch argv (stream_remote.stream_submit) and login-node verbs
        for e in w.events("proposed"):
            print("CMD %s %s %s" % (e.get("lane"), e.get("lever"), " ".join(str(a) for a in (e.get("argv") or []))))
        for r in w.runs:
            print("LOGIN %s" % " ".join(str(a) for a in r[2:]))
        for sub in w.submits:
            print("SBATCH %s" % " ".join(str(a) for a in sub["argv"]))



def main():
    t0 = datetime.datetime.now()
    try:
        main_stages()
    except Exception as e:  # noqa: BLE001 - a crashed stage is a failure, with its trace
        check("the pipeline ran without raising", False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                          traceback.format_exc()[-3000:]))
    finally:
        for n in NOTES:
            print("NOTE: %s" % n)
        if not os.environ.get("STREAM_PIPELINE_KEEP"):
            shutil.rmtree(TMP, ignore_errors=True)
            shutil.rmtree(W.TMP, ignore_errors=True)
        else:
            print("kept %s" % TMP)
    print("\n%d failure(s) (%.0fs)" % (len(FAILURES), (datetime.datetime.now() - t0).total_seconds()))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
