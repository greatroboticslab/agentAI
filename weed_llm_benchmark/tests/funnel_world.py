"""A small Step 1 world for the funnel tests (runner §5.2.7), built by running
Step 1's own code (inc/verify.py pool, crops, embed, fit, calibrate, admit;
inc/select.py build) on painted pictures, so every file is in verify's exact
format and consistent with every other.

    import funnel_world as FW
    world = FW.build_world(TMP)          # TMP/inc is INC_DIR, TMP/repo is REPO

The caller sets os.environ["INC_DIR"] = TMP/inc and os.environ["REPO"] =
TMP/repo BEFORE importing any weed_optimizer_framework module (common.py reads
them at import). build_world refuses otherwise.

Pictures are 160 x 120 px. Each box is painted in the colour of its TRUE class
(13 colours, one per INC class); the fake embedder reads the colour in the
middle of the crop, so the verifier's features cluster by the true class
whatever a source's label says. The sources use the real slugs of the weed
domain config, so domains/weed.json applies as it does to the real pool.

Planted cases (World.planted):
  vetoed          a verified Waterhemp box of an authoritative source in an image
                  that a Blackbean box painted as Palmer amaranth vetoes
  veto_noevidence an other_ok box in a vetoed image of the no-name source
                  (no verified box anywhere: S10 and S12 both fail)
  exact_dup       a Morning glory box whose twin (same bytes) the alphabetically
                  earlier source labels "weed" (H5a)
  near_eval       re-encoded copies of a dev and a test picture in two sources
                  outside the reference lab (G5 pairs)
  copy            a clean copy of a train_core picture in a reference-lab source
  small_box       a 5 x 5 px box (never embedded)
  hidden_target   a numeric-named box painted Sicklepod
  sources         no_name, numeric, named_other, authoritative, generic_role
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import pathlib
import shutil
import time
import zlib

IMG_W, IMG_H = 160, 120
COLORS = [p for p in itertools.product((0, 128, 255), repeat=3)
          if p not in ((128, 128, 128), (0, 0, 0), (255, 255, 255))][:13]
ROOT = pathlib.Path(__file__).resolve().parents[1]

# class ids (INC): 0 Waterhemp 1 MorningGlory 2 Purslane 3 SpottedSpurge 4 Carpetweed 5 Ragweed
# 6 Eclipta 7 PricklySida 8 PalmerAmaranth 9 Sicklepod 10 Goosegrass 11 CutleafGroundcherry 12 OtherPlant
LATVIA = "project_agml__crop_weed_detection_latvia"
GREENHOUSE = "project_agml__greenhouse_crop_weed_detection"
MH = "project_agml__mh_weed16_weed_detection"
THREE = "project_agml__three_season_weed_detection"
WEEDCROP = "project_agml__weed_crop_detection"
CWP10 = "rf_karthikeya-c8pvy__weed-detection-cwp10"
LEOPARD = "rf_leopard-ai__weed-detection-8h4kx"
TUF = "rf_tuf__weed-3434e"
PERADENIYA = "rf_university-of-peradeniya__weed-detection-ekr8i"
BQDOK = "rf_weed-tnf9e__weed-bqdok"
VANPE = "rf_zig-zag-lnodr__weed-detection-vanpe"
HOLDOUT = "cottonweed_holdout"
SOURCE_NAMES = {
    LATVIA: ["crop", "weed"],
    GREENHOUSE: ["Blackbean", "Palmer Amaranth", "Ragweed", "Redroot Pigweed", "Waterhemp"],
    MH: [str(i) for i in range(15)],
    WEEDCROP: ["Blackbean", "Kochia", "Ragweed", "Redroot Pigweed", "Waterhemp"],
    CWP10: ["sicklepod", "spottedspurge", "swinecress", "waterhemp"],
    LEOPARD: ["crop", "weed"],
    TUF: [],
    PERADENIYA: ["crop", "weed"],
    BQDOK: ["BroWeed", "Maize", "NarWeed"],
    VANPE: ["Carpet weed", "Crabgrass", "Morning glory", "Nutsedge"],
}


class WorldUnavailable(RuntimeError):
    """An optional dependency the world needs is absent (the test skips)."""


class World(object):
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.repo = self.root / "repo"
        self.inc_dir = self.root / "inc"
        self.step1 = self.inc_dir / "step1"
        self.funnel_dir = self.inc_dir / "funnel"
        self.sources = {}
        self.planted = {}
        self.never_train = None
        self.registry = None
        self.images = {}
        self.taxonomy_cache = None


def _bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


class _Painter(object):
    def __init__(self, seed, C):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(seed)
        self.C = C
        self.hashes = []

    def paint(self, path, boxes, texture=False, fmt="png"):
        """A smooth random picture, each (cls, cx, cy, w, h) box filled with its
        class colour; regenerated until it is > 12 dHash bits from every
        picture made so far."""
        from PIL import Image
        np = self.np
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(60):
            small = self.rng.integers(40, 216, size=(6, 8, 3), dtype=np.uint8)
            arr = np.array(Image.fromarray(small).resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.int16)
            for c, cx, cy, w, h in boxes:
                x0, x1 = int(round((cx - w / 2) * IMG_W)), int(round((cx + w / 2) * IMG_W))
                y0, y1 = int(round((cy - h / 2) * IMG_H)), int(round((cy + h / 2) * IMG_H))
                patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.int16) + np.array(COLORS[c], dtype=np.int16)
                if texture:
                    patch += self.rng.integers(-14, 15, size=patch.shape).astype(np.int16)
                arr[y0:y1, x0:x1] = patch
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if fmt == "jpg":
                im.save(path, quality=92)
            else:
                im.save(path)
            h = self.C.dhash(path)
            if all(_bits(h, o) > 12 for o in self.hashes):
                self.hashes.append(h)
                return h
        raise AssertionError("could not make a distinct picture")

    def layout3(self, classes):
        out = []
        for k, c in enumerate(classes):
            dx = float(self.rng.uniform(-0.02, 0.02))
            dy = float(self.rng.uniform(-0.08, 0.08))
            out.append((c, round(0.18 + 0.32 * k + dx, 4), round(0.5 + dy, 4), 0.22, 0.32))
        return out


def _yolo(path, lines):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(v) for v in ln) + "\n" for ln in lines))


def fake_feature(pil):
    import numpy as np
    a = np.asarray(pil.convert("RGB"), dtype=np.float64)
    ctr = a[100:124, 100:124].reshape(-1, 3)
    k = int(np.argmin(((np.array(COLORS, dtype=np.float64) - ctr.mean(0)) ** 2).sum(1)))
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(a[100:124, 100:124]).tobytes()))
    v = np.zeros(24)
    v[k] = 1.0
    return v + rng.normal(0, 1, 24) * 0.012 * float(ctr.std(0).mean())


class FakeEmbedder(object):
    name = "fake-colour"
    dim = 24

    def __call__(self, pils):
        import numpy as np
        return np.stack([fake_feature(p) for p in pils])


def _manifest_row(C, split, stem, img, lbl, session, source):
    return {"image": str(img), "label": str(lbl), "sha256": C.sha256_file(img), "label_sha256": C.sha256_file(lbl),
            "source": source, "session": session, "key": "%s__%s" % (split, stem)}


class PlantContext(object):
    """What a plant hook of build_world gets: the world so far, the painter,
    add(slug, stem, painted, labelled, fmt) of the named sources, src_id(slug,
    name), the split rows and their painted boxes, and the datasets root."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_world(root, seed=0, with_probe=True, n_images=60, plant=None, core_frames=4):
    """Build the world under root (root/inc = INC_DIR, root/repo = REPO) and
    return a World. with_probe=False stops after verify pool and crops (no
    sklearn needed). n_images scales the filler images of the named sources.
    plant(ctx), when given, is called with a PlantContext after the built-in
    sources are painted and before the registry is written and Step 1 runs,
    so a test can add pictures to the named sources (tests/
    test_funnel_pipeline.py plants the copy detector's calibration negatives
    and an augmented evaluation copy); the world is unchanged without it.
    core_frames is the number of train_core pictures per capture session (10
    sessions); a hook may add registry entries through ctx.registry_extra."""
    root = pathlib.Path(root).resolve()
    need = ["numpy", "PIL"] + (["sklearn", "joblib"] if with_probe else [])
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        raise WorldUnavailable("%s missing" % ", ".join(missing))
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.inc import verify as V
    from weed_optimizer_framework.tools import cwd12_species as S
    if pathlib.Path(C.INC_DIR).resolve() != root / "inc" or pathlib.Path(C.REPO).resolve() != root / "repo":
        raise RuntimeError("set INC_DIR=%s and REPO=%s before importing the INC modules (now %s, %s)"
                           % (root / "inc", root / "repo", C.INC_DIR, C.REPO))
    W = World(root)
    P = _Painter(seed, C)
    repo, DS = W.repo, W.repo / "datasets"
    (repo / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT.parent / "docs" / "FUNNEL_AUDIT.md", repo / "docs" / "FUNNEL_AUDIT.md")
    W.funnel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json",
                    W.funnel_dir / "prereg_v1.json")

    # ---------------------------------------------------------------- splits
    cwd = repo / "downloads" / "cottonweeddet12"
    rows = {"train_core": [], "dev": [], "test": []}
    boxes_of = {}
    for split, sessions, frames in (("train_core", ["20210701_Cam_S%d" % s for s in range(10)], int(core_frames)),
                                    ("dev", ["20210702_Cam_D%d" % d for d in range(2)], 3),
                                    ("test", ["20210703_Cam_T0"], 3)):
        sub = "valid" if split == "test" else "train"
        for s, sess in enumerate(sessions):
            for f in range(frames):
                stem = "%s_%s_%d" % (split, sess, f + 1)
                bx = P.layout3([(s + 3 * f + k) % 12 for k in range(3)])
                img, lbl = cwd / sub / "images" / (stem + ".jpg"), cwd / sub / "labels" / (stem + ".txt")
                P.paint(img, bx, texture=True, fmt="jpg")
                _yolo(lbl, bx)
                boxes_of[str(img)] = bx
                rows[split].append(_manifest_row(C, split, stem, img, lbl, sess if split != "test" else "",
                                                 "cottonweeddet12/%s" % sub))
    for split, rs in rows.items():
        C.write_manifest(C.manifest_path(split), rs)
    for split in ("ood22", "ood23", "imageweeds"):
        C.write_manifest(C.manifest_path(split), [])
    entries = [[C.dhash(r["image"]), split, r["key"]] for split in ("dev", "test") for r in rows[split]]
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries, "min_expected": len(entries)}))
    C.LOCK_PATH.write_text(json.dumps({"manifests": {"train_core": C.sha256_file(C.manifest_path("train_core"))},
                                       "nevertrain_sha256": C.sha256_file(C.NEVER_TRAIN_INDEX)}))
    W.never_train = C.NEVER_TRAIN_INDEX
    core = rows["train_core"]

    # ---------------------------------------------------------------- sources
    def src_id(slug, name):
        return SOURCE_NAMES[slug].index(name)

    def add(slug, stem, painted, labelled, fmt="png"):
        """painted: [(true cls, cx, cy, w, h)]; labelled: source class id per box."""
        d = DS / slug
        img = d / "images" / ("%s.%s" % (stem, fmt))
        h = P.paint(img, painted, texture=False, fmt=fmt)
        _yolo(d / "labels" / (stem + ".txt"), [(sid,) + tuple(b[1:]) for sid, b in zip(labelled, painted)])
        W.images["%s|%s" % (slug, stem)] = img
        return img, h

    n_fill = max(2, int(n_images) // 12)
    # authoritative NDSU sources: target boxes painted right, some painted as a relative
    for i in range(n_fill):
        add(WEEDCROP, "wc_%02d" % i, P.layout3([0, 12, 5]),
            [src_id(WEEDCROP, "Waterhemp"), src_id(WEEDCROP, "Blackbean"), src_id(WEEDCROP, "Ragweed")])
        add(GREENHOUSE, "gh_%02d" % i, P.layout3([8, 12, 0]),
            [src_id(GREENHOUSE, "Palmer Amaranth"), src_id(GREENHOUSE, "Redroot Pigweed"),
             src_id(GREENHOUSE, "Waterhemp")])
    # the vetoed image: a verified Waterhemp, a Blackbean painted Palmer amaranth
    vet_img, _ = add(WEEDCROP, "wc_veto", P.layout3([0, 8, 12]),
                     [src_id(WEEDCROP, "Waterhemp"), src_id(WEEDCROP, "Blackbean"), src_id(WEEDCROP, "Kochia")])
    # a Ragweed label on a Redroot pigweed plant (a rejected target-labelled box)
    add(WEEDCROP, "wc_wrong", P.layout3([12, 12, 12]),
        [src_id(WEEDCROP, "Ragweed"), src_id(WEEDCROP, "Redroot Pigweed"), src_id(WEEDCROP, "Kochia")])
    # a small box
    small_img, _ = add(WEEDCROP, "wc_small", [(0, 0.3, 0.5, 0.22, 0.32), (12, 0.8, 0.8, 0.03, 0.04)],
                       [src_id(WEEDCROP, "Waterhemp"), src_id(WEEDCROP, "Kochia")])
    # numeric source: id 12 is Sicklepod (hidden target), the rest other plants
    for i in range(n_fill):
        add(MH, "mh_%02d" % i, P.layout3([12, 12, 12]), [2, 1, 4])
    hid_img, _ = add(MH, "mh_hidden", P.layout3([9, 12, 12]), [12, 2, 1])
    # no-name source (wildcard): other plants; one image with a hidden Palmer amaranth
    for i in range(n_fill):
        add(TUF, "tuf_%02d" % i, P.layout3([12, 12, 12]), [0, 0, 0])
    tuf_veto, _ = add(TUF, "tuf_veto", P.layout3([12, 8, 12]), [0, 0, 0])
    # named-other source: BroWeed, Maize, NarWeed
    for i in range(n_fill):
        add(BQDOK, "bq_%02d" % i, P.layout3([12, 12, 12]),
            [src_id(BQDOK, "BroWeed"), src_id(BQDOK, "Maize"), src_id(BQDOK, "NarWeed")])
    # role + generic names
    for i in range(n_fill):
        add(LEOPARD, "lp_%02d" % i, P.layout3([12, 12, 12]),
            [src_id(LEOPARD, "crop"), src_id(LEOPARD, "weed"), src_id(LEOPARD, "crop")])
    # the reference lab's re-export: species named, one clean copy of a train_core picture
    for i in range(n_fill):
        add(CWP10, "cw_%02d" % i, P.layout3([9, 3, 12]),
            [src_id(CWP10, "sicklepod"), src_id(CWP10, "spottedspurge"), src_id(CWP10, "swinecress")])
    copy_row = core[5]
    d = DS / CWP10
    from PIL import Image, ImageFilter
    with Image.open(copy_row["image"]) as im:
        (d / "images").mkdir(parents=True, exist_ok=True)
        im.convert("RGB").filter(ImageFilter.GaussianBlur(1.2)).save(d / "images" / "cw_copy.jpg", quality=95)
    name_of = {0: "waterhemp", 9: "sicklepod", 3: "spottedspurge"}
    truth = boxes_of[copy_row["image"]]
    lab = [(src_id(CWP10, name_of[t[0]]) if t[0] in name_of else src_id(CWP10, "swinecress"),) + tuple(t[1:])
           for t in truth]
    lab.append((src_id(CWP10, "swinecress"), 0.5, 0.12, 0.2, 0.2))     # no train_core twin box
    _yolo(d / "labels" / "cw_copy.txt", lab)
    # exact duplicate: latvia (earlier slug) labels a Morning glory "weed"; vanpe names it
    dup_img, _ = add(LATVIA, "lv_dup", P.layout3([1, 12, 12]),
                     [src_id(LATVIA, "weed"), src_id(LATVIA, "crop"), src_id(LATVIA, "crop")])
    for i in range(n_fill):
        add(LATVIA, "lv_%02d" % i, P.layout3([12, 12, 12]),
            [src_id(LATVIA, "crop"), src_id(LATVIA, "weed"), src_id(LATVIA, "crop")])
    (DS / VANPE / "images").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dup_img, DS / VANPE / "images" / "vp_dup.png")
    dup_boxes = [ln.split() for ln in (DS / LATVIA / "labels" / "lv_dup.txt").read_text().splitlines()]
    _yolo(DS / VANPE / "labels" / "vp_dup.txt",
          [(src_id(VANPE, "Morning glory"),) + tuple(dup_boxes[0][1:]),
           (src_id(VANPE, "Crabgrass"),) + tuple(dup_boxes[1][1:]),
           (src_id(VANPE, "Nutsedge"),) + tuple(dup_boxes[2][1:])])
    for i in range(n_fill):
        add(VANPE, "vp_%02d" % i, P.layout3([1, 4, 12]),
            [src_id(VANPE, "Morning glory"), src_id(VANPE, "Carpet weed"), src_id(VANPE, "Nutsedge")])
    # near_eval: re-encoded dev and test pictures in two sources outside the reference lab
    for slug, ev, stem in ((TUF, rows["dev"][0], "tuf_evalcopy"), (PERADENIYA, rows["test"][0], "pe_evalcopy")):
        (DS / slug / "images").mkdir(parents=True, exist_ok=True)
        with Image.open(ev["image"]) as im:
            im.convert("RGB").resize((im.width - 6, im.height - 4), Image.BILINEAR).save(
                DS / slug / "images" / (stem + ".jpg"), quality=60)
        _yolo(DS / slug / "labels" / (stem + ".txt"), [(0, 0.5, 0.5, 0.2, 0.2)])
    for i in range(n_fill):
        add(PERADENIYA, "pe_%02d" % i, P.layout3([12, 12, 12]),
            [src_id(PERADENIYA, "weed"), src_id(PERADENIYA, "crop"), src_id(PERADENIYA, "weed")])
    add(PERADENIYA, "pe_nobox", [(12, 0.5, 0.5, 0.2, 0.2)], [0])
    (DS / PERADENIYA / "labels" / "pe_nobox.txt").write_text("")
    # the leave-4-out holdout copy (calibration only) and the three_season copies (KT3)
    for slug, idx in ((HOLDOUT, (4, 33)), (THREE, (8, 24))):
        dd = DS / slug
        for ci in idx:
            r = core[ci]
            stem = pathlib.Path(r["image"]).stem
            (dd / "train" / "images").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["image"], dd / "train" / "images" / (stem + ".jpg"))
            _yolo(dd / "train" / "labels" / (stem + ".txt"), boxes_of[r["image"]])
    registry_extra = {}
    if plant is not None:
        plant(PlantContext(world=W, painter=P, add=add, src_id=src_id, rows=rows, boxes_of=boxes_of,
                           datasets=DS, core=core, C=C, registry_extra=registry_extra))
    # skipped registry entries
    (DS / "q_quarantined" / "images").mkdir(parents=True, exist_ok=True)
    reg = {"datasets": {
        s: {"local_path": str(DS / s), "annotation": "bbox", "class_names": list(n)}
        for s, n in SOURCE_NAMES.items()}}
    reg["datasets"][HOLDOUT] = {"local_path": str(DS / HOLDOUT), "annotation": "bbox",
                                "class_names": list(S.CWD12_SPECIES)}
    reg["datasets"][THREE] = {"local_path": str(DS / THREE), "annotation": "bbox",
                              "class_names": list(S.CWD12_SPECIES)}
    reg["datasets"]["cottonweeddet12"] = {"local_path": str(cwd), "annotation": "bbox", "class_names": []}
    reg["datasets"]["q_quarantined"] = {"local_path": str(DS / "q_quarantined"), "annotation": "bbox",
                                        "class_names": ["weed"], "status": "quarantined"}
    for slug, entry in sorted(registry_extra.items()):
        if slug in reg["datasets"]:
            raise AssertionError("a plant hook may add registry entries, not replace %s" % slug)
        reg["datasets"][slug] = dict(entry)
    fw = repo / "results" / "framework"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "dataset_registry.json").write_text(json.dumps(reg))
    (fw / "dataset_flags.json").write_text(json.dumps({}))
    W.registry = fw / "dataset_registry.json"

    # ---------------------------------------------------------------- Step 1
    def args(*argv):
        return V.parse_args(list(argv))
    V.cmd_pool(args("pool", "--procs", "1"))
    V.cmd_crops(args("crops"))
    if with_probe:
        V.cmd_embed(args("embed", "--nshards", "1", "--procs", "1"), embedder=FakeEmbedder())
        V.cmd_fit(args("fit"))
        V.cmd_calibrate(args("calibrate"))
        V.cmd_admit(args("admit"))
        from weed_optimizer_framework.tools.inc import select as SEL
        SEL.build(workers=1)

    # ---------------------------------------------------------------- record
    key = lambda slug, stem, split="": "%s__%s%s" % (V._sanitise(slug), split + "__" if split else "", stem)
    W.sources = {"no_name": TUF, "numeric": MH, "named_other": BQDOK, "authoritative": WEEDCROP,
                 "generic_role": LEOPARD, "reference_lab": CWP10, "all": sorted(SOURCE_NAMES)}
    W.planted = {
        "vetoed": {"key": key(WEEDCROP, "wc_veto"), "box": 0, "blocker_box": 1},
        "veto_noevidence": {"key": key(TUF, "tuf_veto"), "box": 0, "blocker_box": 1},
        "exact_dup": {"dropped": "d:%s|images/vp_dup.png" % VANPE, "kept_key": key(LATVIA, "lv_dup")},
        "near_eval": {"d:%s|images/tuf_evalcopy.jpg" % TUF: ("dev", rows["dev"][0]["key"]),
                      "d:%s|images/pe_evalcopy.jpg" % PERADENIYA: ("test", rows["test"][0]["key"])},
        "copy": {"id": "d:%s|images/cw_copy.jpg" % CWP10, "train_core_key": copy_row["key"],
                 "key": key(CWP10, "cw_copy"), "boxes": len(lab), "unmatched_box": len(lab) - 1},
        "no_boxes": "d:%s|images/pe_nobox.png" % PERADENIYA,
        "small_box": {"key": key(WEEDCROP, "wc_small"), "box": 1},
        "hidden_target": {"key": key(MH, "mh_hidden"), "box": 0},
        "wrong_ragweed": {"key": key(WEEDCROP, "wc_wrong"), "box": 0},
        "holdout_copies": [core[i]["key"] for i in (4, 33)],
        "three_season_copies": [core[i]["key"] for i in (8, 24)],
    }
    return W


# ------------------------------------------------------------------ fixtures
def gbif_recording():
    """The recorded GBIF answers (tests/fixtures/funnel/gbif_recording.json)."""
    p = ROOT / "tests" / "fixtures" / "funnel" / "gbif_recording.json"
    with open(p) as fh:
        return json.load(fh)


def replay_transport(recording=None, calls=None):
    """transport(url, params) replaying the recording; a request it does not
    hold answers 404 (and is appended to calls when given)."""
    rec = recording or gbif_recording()
    table = {}
    for k, v in rec["calls"].items():
        table[(v["url"], json.dumps(v["params"], sort_keys=True))] = v

    def transport(url, params):
        if calls is not None:
            calls.append((url, dict(params)))
        v = table.get((url, json.dumps(dict(params), sort_keys=True)))
        if v is None and os.environ.get("FUNNEL_GBIF_RECORD") == "1" and recording is None:
            v = _record_call(url, params)
            table[(url, json.dumps(dict(params), sort_keys=True))] = v
        if v is None:
            return 404, b"{}", {}
        return v["status"], json.dumps(v["body"], sort_keys=True).encode("utf-8"), {}
    return transport


def _record_call(url, params):
    """Only with FUNNEL_GBIF_RECORD=1 (never in a normal test run): fetch a
    request the recording lacks from the live authority and append it to
    tests/fixtures/funnel/gbif_recording.json, so that a config change (a new
    override taxon) can be recorded once and then replayed offline."""
    import hashlib
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode(sorted(dict(params).items()))
    with urllib.request.urlopen(url + ("?" + q if q else ""), timeout=30) as r:
        status, raw = r.status, r.read()
    body = json.loads(raw.decode("utf-8"))
    p = ROOT / "tests" / "fixtures" / "funnel" / "gbif_recording.json"
    rec = json.loads(p.read_text())
    k = "%s?%s" % (url, json.dumps(dict(params), sort_keys=True))
    v = {"body": body, "original_sha256": hashlib.sha256(raw).hexdigest(), "params": dict(params),
         "status": status, "url": url}
    rec["calls"][k] = v
    rec.setdefault("added", []).append({"calls": [k], "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "why": "FUNNEL_GBIF_RECORD: a request the recording lacked"})
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return v


def build_taxonomy_cache(world_or_dir, names, domain="weed"):
    """taxonomy_cache.json in the funnel dir, built by taxonomy.build_cache
    through the replayed GBIF recording (offline)."""
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import taxonomy as T
    fd = world_or_dir.funnel_dir if isinstance(world_or_dir, World) else pathlib.Path(world_or_dir)
    pre = D.load_prereg(fd / "prereg_v1.json")
    out = fd / "taxonomy_cache.json"
    T.build_cache(names, D.load(domain), replay_transport(), out, prereg=pre, testing=True)
    return out
