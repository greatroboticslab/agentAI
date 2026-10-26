#!/usr/bin/env python3
"""V3F deviation 6 (2026-10-25; docs/prereg/V3F_dev6_2026-10-25.md): D1's passes on ood22 and ood23 under the loader
fix (inc2.v3fam_loader), its pre-flight, identity_ood23.json moved aside, score:D1 reopened under deviation 6's marker,
and the second continuation waiting for D1's record since then. Record-level checks on the cluster's own files
(tests/fixtures/v3fam_dev6, each verified by sha256 against its MANIFEST.json) and Ultralytics' own validation
dataloader (no model, no scoring pass), so tests/test_v3fam_dev6_mutants.py runs this file against each mutant.

- the fixture: the first 4,559 bytes of ood23__20230804_HTRC_iPhone12_WY_2.jpg (every marker segment through the start
  of scan; the full file's sha256 adc328fb... is the manifest's) and its label (322981fd...): PIL reads MPO,
  orientation 6, stored 4032x3024; the loader decodes the full file's frame, 4032 x 3024 (h x w);
- the geometry: that frame through Ultralytics' validation dataloader (rect batches of 32 at 640, stride 32) and E3's
  own ground-truth round trip (twostage.e3_validator_class(...)[0].e3_gt_check, unedited): refused with 0.109094, as on
  the cluster (jobs 49820090, 49821128); under the loader fix it comes back within 1e-6. In a batch with 27 landscape
  photos and variants of the same header (MPO tagged 1, 3, 8; JPEG tagged 1, 6, 8): without the fix the MPO photos
  tagged 6 and 8 refuse, with it none, and the fix changes exactly those two sizes;
- the pre-flight: on a world of such files (ood22 without an MPO photo tagged 6 or 8) it records every image's sizes,
  frame and letterbox scale, the changed images (ood23 two, ood22 none), the defect's extent without the fix, and is
  written once; a fix that changes nothing refuses, writing nothing; a changed manifest refuses the record;
- D1's score job: every pass on ood22 and ood23 under the fix and checked against the record before it is committed
  (each carries loader_shape_fix), dev without it; ood23's identity taken under the fix and noted, an identity without
  its note refused unless created since deviation 6's reopening (then moved aside and retaken); an identity whose
  observer check refuses moved aside before the refusal; a pass whose label cache never called exif_size refused; a
  kept ood23 pass without the note, or taken against another pre-flight record, refused; the record refused once the
  code it pins changed; D1's score record carrying the caveat on plain ood23 numbers; the fix never left installed;
- the record superseded: identity_ood23.json (3c297647..., the cluster's bytes) moved aside byte for byte into
  equivalence/superseded_2026-10-25/ with a manifest naming deviation 6; identity_ood22.json left; reported superseded,
  never stale, while in place;
- the verdict's guard: with D1's provenance as it stands (attempts started 2026-10-25T15:20:48Z and 16:32:50Z) and
  --ticker-failed score:D1 it refuses, naming deviation 6 and D1, writing nothing; a failure record of score:D1 written
  at 16:39:35Z is not read; with an attempt since 2026-10-25T17:00:00Z, or D1's complete score record, it is ready;
  DEV6_REOPEN_UTC equals the domain block's before_utc;
- the ticker: on its live items of 2026-10-25 the block reopens score:D1 once and nothing else, its spend kept
  (1.0684 + 0.0928 GPU-h), its ids retired; L23L then waits for D1 and L23J D1 is proposed; never without the file;
- a refusal of the loader check, or of E3's ground-truth and frame checks (the geometry refusal of 2026-10-25 among
  them), is the platform's and deterministic: held for a person at the first refusal, never counted, never released
  by the records' state, proposed again only by a later deviation's reopen block; an E3 refusal resting on the arm's
  own boxes stays the arm's;
- the deploy: the deviation's file shipped and pinned by sha256, kept byte for byte in docs/CONTINUOUS_LOOP.md; the code
  the records pin unchanged (inc2/v3fam.py 3d776dae...); the job script lists the new module; a deploy whose head does
  not descend from the deployed head refused before anything is copied.

Run:  python3 tests/test_v3fam_dev6.py
"""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_v3fam_dev5 as D5  # noqa: E402  (sets INC_DIR / REPO in a temporary directory first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_trained as VTR  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_verdict as VR  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as S  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream_remote as SR  # noqa: E402

try:
    from weed_optimizer_framework.tools.inc2 import v3fam_loader as LD  # noqa: E402
except ImportError:                              # the tree before deviation 6
    LD = None
try:
    from weed_optimizer_framework.tools.inc2 import v3fam_supersede as SUP  # noqa: E402
except ImportError:
    SUP = None

VT = D5.VT
check = VT.check
FAILURES = VT.FAILURES
DOM, FAM, GIT, PKG = D5.DOM, D5.FAM, D5.GIT, D5.PKG
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "v3fam_dev6"
DEV6_FILE = "docs/prereg/V3F_dev6_2026-10-25.md"
DEV6_MARKER = "deviation 6 (2026-10-25)"
DEV6_UTC = "2026-10-25T17:00:00Z"
WY2 = "ood23__20230804_HTRC_iPhone12_WY_2"
HEAD = WY2 + ".head.bin"
OOD23_SHA = "3c29764783b8303b0659cfe75ee49b966dff4334394477fba24b2245045ad3ea"
OOD22_SHA = "aebd81bc4cae8a31e96342f3b6b55824252968e17d8be8c803a2a1d71ef3eef6"
B6C4AEF = D5.B6C4AEF
REFUSAL6 = "deviation 6's loader check refuses"
E6 = b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"    # the header's EXIF orientation entry (IFD0, big-endian), 6
GT_GEOMETRY = ("ood23__20230804_HTRC_iPhone12_WY_2: ground-truth box 0 comes back through the geometry 0.109094 from "
               "its label (> 0.001): the map to the original image is wrong")


def sha(p):
    """sha256 of a file, None when it does not exist (a check on the tree before deviation 6 fails, never crashes)."""
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def man():
    return json.loads((FIX / "MANIFEST.json").read_text())


def head_bytes():
    return (FIX / HEAD).read_bytes()


def variant(tag=6, mpf=True):
    """The fixture's header as a decodable file (an end-of-image marker appended), its EXIF orientation set to tag and,
    with mpf False, its MPF segment left out (PIL then reads JPEG)."""
    b = head_bytes()
    if tag != 6:
        b = b.replace(E6, E6[:8] + struct.pack(">H", int(tag)) + b"\x00\x00")
    if not mpf:
        i = b.index(b"MPF\x00") - 4
        n = struct.unpack(">H", b[i + 2:i + 4])[0]
        b = b[:i] + b[i + 2 + n:]
    return b + b"\xff\xd9"


def label_bytes():
    return (FIX / (WY2 + ".txt")).read_bytes()


@contextlib.contextmanager
def fix_or_nothing(exam="ood23"):
    """The loader fix where this tree has it; nothing on the tree before deviation 6 (whose behaviour is then tested)."""
    if LD is not None and hasattr(LD, "loader_fix"):
        with LD.loader_fix(exam) as obs:
            yield obs
    else:
        yield None


def ul_exif_size():
    import ultralytics.data.utils as U
    return U.exif_size


# ------------------------------------------------------------------ the fixture
def t_fixture():
    print("the fixture: the failing ood23 frame's header and label, as on the cluster")
    from PIL import Image
    m = man()
    ok = all(sha(FIX / n) == d["sha256"] for n, d in m["files"].items())
    check("every fixture file hashes as its MANIFEST.json records (copied from the cluster and the lab on 2026-10-25); "
          "the header's source is the manifest's image (full sha256 adc328fb..., 10,368,508 bytes), the label the "
          "manifest's label_sha256", ok and m["files"][HEAD]["full_file_sha256"] == m["manifest_row"]["sha256"]
          == "adc328fb98e351efe942027b093e7bdee3bf2efc57f41174d45c3f3d19a9f7e7"
          and sha(FIX / (WY2 + ".txt")) == m["manifest_row"]["label_sha256"] and len(head_bytes()) == 4559
          and sha(FIX / "identity_ood23.json") == OOD23_SHA and sha(FIX / "identity_ood22.json") == OOD22_SHA,
          {n: sha(FIX / n)[:12] for n in m["files"]})
    p = VT.TMP / "fixture_wy2.jpg"
    p.write_bytes(variant())
    with Image.open(str(p)) as im:
        fmt, tag, size = im.format, im.getexif().get(274), im.size
    import cv2
    from ultralytics.utils.patches import imread
    a = imread(str(p), flags=cv2.IMREAD_COLOR)
    check("PIL reads it as MPO, orientation 6, stored 4032x3024 (W x H); the loader (ultralytics.utils.patches.imread) "
          "decodes the frame of the full file, 4032 x 3024 (h x w); the label is the source XML's box in that frame",
          fmt == "MPO" and tag == 6 and size == (4032, 3024) and a is not None and a.shape[:2] == (4032, 3024)
          and label_bytes() == b"12 0.545470 0.459821 0.195437 0.257937\n", (fmt, tag, size, None if a is None
                                                                            else a.shape, label_bytes()))
    check("Ultralytics' own exif_size (the installed release) keeps the stored size for it: the defect is there",
          tuple(ul_exif_size()(Image.open(str(p)))) == (4032, 3024))


# ------------------------------------------------------------------ the geometry
def make_view(root, files):
    """images/ and labels/ as the locked scorer lays out an exam (inc.scorer.build_view): {key: bytes}."""
    shutil.rmtree(str(root), ignore_errors=True)
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for k, b in files.items():
        (root / "images" / ("%s.jpg" % k)).write_bytes(b)
        (root / "labels" / ("%s.txt" % k)).write_bytes(label_bytes())
    return root


def gt_scan(view):
    """Every image of a view through Ultralytics' validation dataloader (the validator's own build: rect batches of 32
    at 640, stride 32, pad 0.5) and E3's unedited ground-truth round trip: {key: (ok, diff or message)}."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator
    E3V, _base = TS.e3_validator_class({"mode": "identity", "nms": "locked"})

    class Probe(object):
        e3_gt_maxdiff = 0.0
    args = get_cfg(overrides={"imgsz": 640, "batch": 32, "rect": True, "mode": "val", "task": "detect", "workers": 0,
                              "plots": False, "half": False})
    v = DetectionValidator(save_dir=view.parent / ("runs_%s" % view.name), args=args)
    v.device = torch.device("cpu")
    v.data = {"names": {i: n for i, n in enumerate(C.CLASS_NAMES)}, "nc": C.NC, "channels": 3}
    v.stride = 32
    out = {}
    for batch in v.get_dataloader(str(view / "images"), 32):
        batch = v.preprocess(batch)
        for si in range(len(batch["im_file"])):
            pb = v._prepare_batch(si, batch)
            pb["im_file"] = batch["im_file"][si]
            key = pathlib.Path(pb["im_file"]).stem
            h0, w0 = int(pb["ori_shape"][0]), int(pb["ori_shape"][1])
            pr = Probe()
            try:
                E3V.e3_gt_check(pr, pb, key, w0, h0)
                out[key] = (True, pr.e3_gt_maxdiff)
            except TS.E3Refused as e:
                out[key] = (False, str(e))
    return out


def t_geometry():
    print("the geometry: the failing frame through Ultralytics' validation dataloader and E3's round trip")
    root = VT.TMP / "geometry"
    view = make_view(root / "wy2", {WY2: variant()})
    plain = gt_scan(view)
    view_f = make_view(root / "wy2_fixed", {WY2: variant()})
    with fix_or_nothing() as obs:
        fixed = gt_scan(view_f)
    check("without the fix E3's round trip refuses the frame with the cluster's own message (0.109094 = 0.2 x the box's "
          "cx 0.545470: the letterbox's further 0.8, unrecorded in ratio_pad)", plain[WY2][0] is False
          and GT_GEOMETRY in plain[WY2][1], plain)
    check("under deviation 6's loader fix the same frame comes back within 1e-6 (the fix saw it once and changed its "
          "size)", fixed[WY2][0] is True and fixed[WY2][1] <= 1e-6 and obs is not None
          and obs.calls.get(WY2) == 1 and obs.changed.get(WY2) == 1, (fixed, obs and dict(obs.calls)))
    files = {WY2: variant(6, True), "ood23__MPO8": variant(8, True), "ood23__MPO3": variant(3, True),
             "ood23__MPO1": variant(1, True), "ood23__JPEG6": variant(6, False), "ood23__JPEG8": variant(8, False),
             "ood23__JPEG1": variant(1, False)}
    files.update({"ood23__L%02d" % i: variant(1, False) for i in range(27)})
    plain_b = gt_scan(make_view(root / "batch", files))
    with fix_or_nothing() as obs_b:
        fixed_b = gt_scan(make_view(root / "batch_fixed", files))
    bad_plain = sorted(k for k, (ok, _d) in plain_b.items() if not ok)
    bad_fixed = sorted(k for k, (ok, _d) in fixed_b.items() if not ok)
    check("a batch of 34 (27 landscape photos and the header's variants, so that the 32 images Ultralytics' cache "
          "holds as landscape fill the first rect batch): without the fix exactly the MPO photos tagged 6 and 8 "
          "refuse; under it none, every round trip within 1e-6", bad_plain == sorted([WY2, "ood23__MPO8"])
          and bad_fixed == [] and max(d for ok, d in fixed_b.values() if ok) <= 1e-6, (bad_plain, bad_fixed))
    check("  the fix changed exactly those two sizes (JPEG tagged 6 and 8 already swapped by Ultralytics, MPO tagged 1 "
          "and 3 untouched), each image seen once, nothing changed that is not MPO tagged 6 or 8",
          obs_b is not None and sorted(obs_b.changed) == sorted([WY2, "ood23__MPO8"])
          and set(obs_b.calls) == set(files) and set(obs_b.calls.values()) == {1} and not obs_b.bad,
          obs_b and (sorted(obs_b.changed), obs_b.bad))
    keys = list(files)
    e_set = refused_by(obs_b.check, keys, [WY2], 1, "the batch") if obs_b is not None else None
    e_n = refused_by(obs_b.check, keys, [WY2, "ood23__MPO8"], 2, "the batch") if obs_b is not None else None
    e_k = refused_by(obs_b.check, keys + ["ood23__absent"], [WY2, "ood23__MPO8"], 1, "the batch") \
        if obs_b is not None else None
    check("  the observer's check refuses a changed set other than the record's, a build count other than the "
          "pass's, and an exam image it never saw", e_set is not None and "the record lists 1" in str(e_set)
          and e_n is not None and "not seen exactly 2" in str(e_n) and e_k is not None and "ood23__absent" in str(e_k),
          (e_set, e_n, e_k))
    check("  the fix is removed on exit (Ultralytics' own exif_size again) and is never nested",
          not getattr(ul_exif_size(), "_v3fam_dev6", False) and LD is not None and nested_refused(), ul_exif_size())


def nested_refused():
    try:
        with LD.loader_fix("ood23"):
            with LD.loader_fix("ood23"):
                return False
    except F.V3famRefused as e:
        return REFUSAL6 in str(e) and not getattr(ul_exif_size(), "_v3fam_dev6", False)
    return False


# ------------------------------------------------------------------ the pre-flight
OOD23_FILES = {WY2: (6, True), "ood23__MPO8": (8, True), "ood23__MPO3": (3, True), "ood23__JPEG6": (6, False),
               "ood23__JPEG1": (1, False)}
OOD23_FILES.update({"ood23__L%02d" % i: (1, False) for i in range(28)})
OOD22_FILES = {"ood22__MPO1": (1, True), "ood22__JPEG6": (6, False), "ood22__JPEG8": (8, False),
               "ood22__JPEG1": (1, False)}


def make_exam(exam, spec):
    """A materialised exam in this INC_DIR (manifest, exams/v1/<exam>/{images, labels, data.yaml}) of header variants."""
    ed = C.EXAMS_DIR / exam
    shutil.rmtree(str(ed), ignore_errors=True)
    (ed / "images").mkdir(parents=True)
    (ed / "labels").mkdir()
    src = C.INC_DIR / "src" / exam
    shutil.rmtree(str(src), ignore_errors=True)
    src.mkdir(parents=True)
    rows = []
    for k, (tag, mpf) in sorted(spec.items()):
        p = src / ("%s.jpg" % k)
        p.write_bytes(variant(tag, mpf))
        os.symlink(str(p), str(ed / "images" / ("%s.jpg" % k)))
        (ed / "labels" / ("%s.txt" % k)).write_bytes(label_bytes())
        rows.append({"key": k, "image": str(p), "label": str(ed / "labels" / ("%s.txt" % k)), "sha256": sha(p),
                     "label_sha256": sha(ed / "labels" / ("%s.txt" % k)), "session": "", "source": "fixture"})
    with open(str(ed / "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n" % (ed, C.NC))
        for n in C.CLASS_NAMES:
            fh.write("  - %s\n" % n)
    mp = C.manifest_path(exam)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return rows


def ood_world():
    make_exam("ood23", OOD23_FILES)
    make_exam("ood22", OOD22_FILES)


def t_preflight():
    print("the pre-flight: every ood22 and ood23 frame, cached size and letterbox, before D1's passes")
    D5.clean()
    ood_world()
    if LD is None or not hasattr(LD, "preflight"):
        check("inc2.v3fam_loader.preflight exists (deviation 6's pre-flight)", False)
        return
    rec = LD.preflight()
    e23, e22 = rec["exams"]["ood23"], rec["exams"]["ood22"]
    want = sorted([WY2, "ood23__MPO8"])
    check("written once under v3fam/v1/exif/loader_shapes.json: ood23's changed images are exactly its MPO photos "
          "tagged 6 and 8, ood22's none; under the fix every cached size is the loader's frame and every letterbox "
          "scale 1", LD.record_path().is_file() and e23["changed"] == want and e22["changed"] == []
          and e23["fixed_equals_loader"] is True and e22["fixed_equals_loader"] is True
          and e23["scale_fixed_min"] == 1.0 and e22["scale_fixed_min"] == 1.0
          and e23["images"] == len(OOD23_FILES) and e22["images"] == len(OOD22_FILES), (e23.get("changed"),
                                                                                          e22.get("changed")))
    rows = {r[0]: r for r in e23["rows"]}
    check("  per image: format, tag, stored, cached, fixed and loader sizes and both scales (the failing frame: MPO 6, "
          "stored 4032x3024, cached 4032x3024, fixed 3024x4032, loader 3024x4032; scale 0.8 without the fix, 1 with "
          "it); the defect's extent recorded beside (2 images shrunk without the fix)",
          rows[WY2][1:] == ["MPO", 6, 4032, 3024, 4032, 3024, 3024, 4032, 3024, 4032, 0.8, 1.0]
          and e23["unfixed_shrunk"] == 2 and e22["unfixed_shrunk"] == 0, (rows.get(WY2), e23.get("unfixed_shrunk")))
    before = sha(LD.record_path())
    again = LD.preflight()
    check("  a second call reads the record back (written once, unchanged)", sha(LD.record_path()) == before
          and again["exams"]["ood23"]["changed"] == want)
    mp = C.manifest_path("ood22")
    keep = mp.read_bytes()
    mp.write_bytes(keep + b"\n")
    e = refused_by(LD.require_record)
    mp.write_bytes(keep)
    check("  a manifest changed since: the record is refused (the loader check's refusal)", e is not None
          and REFUSAL6 in str(e) and "ood22's manifest changed" in str(e), e)
    rb = LD.record_path().read_bytes()
    alt = json.loads(rb)
    alt["code"]["v3fam_loader"] = "0" * 64
    LD.record_path().write_text(json.dumps(alt))
    e_code = refused_by(LD.require_record)
    alt["code"] = dict(rec["code"], twostage="0" * 64)
    LD.record_path().write_text(json.dumps(alt))
    e_code2 = refused_by(LD.require_record)
    LD.record_path().write_bytes(rb)
    check("  the record pins the code that took it: refused once inc2/v3fam_loader.py, or a file inc2.v3fam.code_shas "
          "names, is other than the record's; read back once they are", e_code is not None and REFUSAL6 in str(e_code)
          and "taken by other code" in str(e_code) and e_code2 is not None and "taken by other code" in str(e_code2)
          and refused_by(LD.require_record) is None and (rec.get("code") or {}).get("v3fam_loader")
          == sha(pathlib.Path(LD.__file__).resolve()), (str(e_code)[:200], str(e_code2)[:200]))
    os.rename(str(LD.record_path()), str(LD.record_path()) + ".kept")
    real = LD.fixed_size
    LD.fixed_size = lambda orig, img: tuple(orig(img))
    try:
        e2 = refused_by(LD.preflight)
    finally:
        LD.fixed_size = real
    check("  a fix that changes nothing (Ultralytics' own size): refused (the fixed size is not the loader's frame, the "
          "letterbox still shrinks), nothing written", e2 is not None and REFUSAL6 in str(e2)
          and "not the loader's frame" in str(e2) and "still shrunk" in str(e2) and not LD.record_path().exists(),
          str(e2)[:300])
    os.rename(str(LD.record_path()) + ".kept", str(LD.record_path()))


def refused_by(fn, *a, **k):
    try:
        fn(*a, **k)
    except F.V3famRefused as e:
        return e
    return None


# ------------------------------------------------------------------ D1's score job
class _NoWorkers(object):
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def emulate_build(exam):
    """What the locked scorer's model.val does with exif_size: one label cache of the exam's images (PIL open, verify,
    ultralytics.data.utils.exif_size as installed now)."""
    import ultralytics.data.utils as U
    from PIL import Image
    for p in sorted((C.EXAMS_DIR / exam / "images").iterdir()):
        with Image.open(str(p)) as im:
            im.verify()
            U.exif_size(im)


def t_score_d1():
    print("D1's score job: ood22 and ood23 under the fix, checked before each commit; dev without it")
    if LD is None or not LD.record_path().is_file():
        check("the pre-flight record exists for D1's job (t_preflight)", False)
        return
    saved = {}

    def patch(mod, name, val):
        saved.setdefault((mod, name), getattr(mod, name))
        setattr(mod, name, val)
    log = {"passes": [], "commits": [], "identity_builds": {}}
    mode = {"skip_build": None, "identity_builds": 2, "created": None}

    def fake_pass(weights, exam, cfg, arm, **kw):
        log["passes"].append((exam, cfg.get("mode"), bool(getattr(ul_exif_size(), "_v3fam_dev6", False))))
        if exam in ("ood22", "ood23") and mode["skip_build"] != exam:
            emulate_build(exam)
        return {"result": {"species_map50_95": 0.1, "agnostic_map50_95": 0.2}}

    def fake_identity(exam, arm, kw):
        p = F.equiv_dir() / ("identity_%s.json" % exam)
        if not p.exists():
            log["identity_builds"][exam] = bool(getattr(ul_exif_size(), "_v3fam_dev6", False))
            for _ in range(mode["identity_builds"]):
                emulate_build(exam)
            F.write_once(p, {"format": F.FMT_EQUIV, "item": "identity", "status": "passed", "exam": exam,
                             "created_utc": mode["created"] or F._utc()})
        return {}

    def fake_commit(arm, dirname, seed, exam, ps, cfg, names, classifier=None, extra=None, **k):
        log["commits"].append((dirname, seed, exam, dict(extra or {})))
        return {"file": "arms/%s/s%d/%s.json" % (dirname, seed, exam), "sha256": "x", "status": "written"}
    kept = {}
    patch(VTR, "sources", lambda b: {s: {"exp": b, "seed": s, "weights": "/w/%s_%d.pt" % (b, s)} for s in F.SEEDS})
    patch(F, "exif_preflight_passed", lambda: True)
    patch(VTR, "d1_classifier", lambda testing=None: ({"W": None}, {}, {"pin_sha256": "p"}))
    patch(F, "require_equivalence", lambda names: {})
    patch(F, "_family_pass", fake_pass)
    patch(SYS, "require_or_make_identity", fake_identity)
    patch(SYS, "commit", fake_commit)
    patch(SYS, "existing", lambda dirname, s, exam, variant="", ens=None: kept.get((dirname, s, exam)))
    patch(SYS, "score_record", lambda arm, dev, eqs, **k: {"arm": arm, "extra": k.get("extra")})
    patch(TS, "_workers", _NoWorkers)
    patch(VTR.V, "BioclipEmbedder", lambda *a, **k: object())
    try:
        shutil.rmtree(str(F.equiv_dir()), ignore_errors=True)
        F.equiv_dir().mkdir(parents=True, exist_ok=True)
        (F.equiv_dir() / "identity_ood22.json").write_bytes((FIX / "identity_ood22.json").read_bytes())
        for n in ("loader_identity_ood22.json", "loader_identity_ood23.json"):
            if (F.exif_dir() / n).exists():
                os.unlink(str(F.exif_dir() / n))
        rec = VTR.score_d1(reported=False)
        ood = [p for p in log["passes"] if p[0] in ("ood22", "ood23")]
        dev = [p for p in log["passes"] if p[0] == "dev"]
        check("every pass on ood22 and ood23 ran under the loader fix (15 each), every dev pass without it (15)",
              len(ood) == 30 and all(p[2] for p in ood) and len(dev) == 15 and not any(p[2] for p in dev),
              log["passes"][:4])
        c23 = [c for c in log["commits"] if c[2] == "ood23"]
        c22 = [c for c in log["commits"] if c[2] == "ood22"]
        cdev = [c for c in log["commits"] if c[2] == "dev"]
        check("  each ood pass committed with loader_shape_fix (deviation 6, one build, ood23's two changed images, "
              "ood22's none) after its check; dev's without", len(c23) == 15 and len(c22) == 15
              and all((c[3].get("loader_shape_fix") or {}).get("deviation") == DEV6_FILE
                      and c[3]["loader_shape_fix"]["changed"] == 2 and c[3]["loader_shape_fix"]["builds"] == 1
                      for c in c23)
              and all((c[3].get("loader_shape_fix") or {}).get("changed") == 0 for c in c22)
              and not any("loader_shape_fix" in c[3] for c in cdev), (c23[:1], cdev[:1]))
        note = F._read_json(F.exif_dir() / "loader_identity_ood23.json") or {}
        check("  ood23's identity reproduction (missing) taken under the fix, both builds observed, and noted by its "
              "sha256; ood22's (in place, the fix changes none of its images) kept without a note",
              log["identity_builds"].get("ood23") is True and "ood22" not in log["identity_builds"]
              and note.get("identity_sha256") == sha(F.equiv_dir() / "identity_ood23.json") and note.get("builds") == 2
              and not (F.exif_dir() / "loader_identity_ood22.json").exists(), (log["identity_builds"], note))
        check("  D1's score record names the deviation, the record and the changed counts; the fix is not left "
              "installed", ((rec or {}).get("extra") or {}).get("loader_shape_fix", {}).get("changed")
              == {"ood22": 0, "ood23": 2} and not getattr(ul_exif_size(), "_v3fam_dev6", False), rec)
        lsf = ((rec or {}).get("extra") or {}).get("loader_shape_fix") or {}
        check("  and carries the caveat on plain ood23 numbers (b0_v1's 0.1861 / 0.1236 taken without the fix) with the "
              "record's count of images the letterbox shrank without it", "b0_v1" in str(lsf.get("note"))
              and "0.1861" in str(lsf.get("note")) and "0.1236" in str(lsf.get("note"))
              and lsf.get("unfixed_shrunk") == {"ood22": 0, "ood23": 2}, lsf)
        # an identity record in place without its note, on an exam the fix changes
        os.unlink(str(F.exif_dir() / "loader_identity_ood23.json"))
        p23 = F.equiv_dir() / "identity_ood23.json"
        old_rec = p23.read_bytes()
        p23.write_text(json.dumps(dict(json.loads(old_rec), created_utc="2026-10-25T15:31:06.816709Z")))
        e = refused_by(VTR.score_d1, reported=False)
        check("an ood23 identity record in place that its note does not name, created before deviation 6's reopening: "
              "refused (the loader check's refusal), left in place", e is not None and REFUSAL6 in str(e)
              and "was not taken under the loader fix" in str(e) and p23.is_file(), e)
        os.unlink(str(p23))
        # deviation 6's identity order: a record taken under the fix whose observer check refuses is moved aside
        ud = F.equiv_dir() / "superseded_dev6_unchecked"
        mode["identity_builds"] = 1
        e_id = refused_by(VTR.score_d1, reported=False)
        mode["identity_builds"] = 2
        moved = sorted(ud.glob("identity_ood23.*.json")) if ud.is_dir() else []
        mf = F._read_json(ud / "MANIFEST.json") or {}
        check("an identity whose observer check refuses (one build, not two): the refusal is the loader check's, the "
              "record it wrote is moved aside byte for byte into equivalence/superseded_dev6_unchecked/ (named by its "
              "sha256, in the folder's manifest), nothing left in place, no note", e_id is not None
              and REFUSAL6 in str(e_id) and "identity reproduction" in str(e_id) and not p23.exists()
              and len(moved) == 1 and moved[0].name == "identity_ood23.%s.json" % sha(moved[0])[:16]
              and ((mf.get("entries") or {}).get(moved[0].name) or {}).get("sha256") == sha(moved[0])
              and not (F.exif_dir() / "loader_identity_ood23.json").exists(), (str(e_id)[:200], moved))
        # a job cut off between the record and its note: created since the reopening, no note
        mode["created"] = "2026-10-25T18:00:00.000001Z"
        SYS.require_or_make_identity("ood23", "D1", {})
        mode["created"] = None
        cut = sha(p23)
        rec_cut = VTR.score_d1(reported=False)
        note = F._read_json(F.exif_dir() / "loader_identity_ood23.json") or {}
        moved = sorted(ud.glob("identity_ood23.*.json"))
        check("an identity created since deviation 6's reopening with no note (cut off before it): moved aside the same "
              "way, taken again under the fix and noted; the job completes", rec_cut is not None
              and (ud / ("identity_ood23.%s.json" % cut[:16])).is_file() and len(moved) == 2
              and sha(p23) != cut and note.get("identity_sha256") == sha(p23), (cut[:12], moved, note))
        os.unlink(str(F.exif_dir() / "loader_identity_ood23.json"))
        os.unlink(str(p23))
        mode["skip_build"] = "ood23"
        log["passes"].clear()
        e2 = refused_by(VTR.score_d1, reported=False)
        mode["skip_build"] = None
        check("a pass whose label cache never called exif_size (a cache read back from disk): refused before anything "
              "is committed", e2 is not None and REFUSAL6 in str(e2) and "it saw 0 of" in str(e2), e2)
        doc = F.root() / "arms" / "D1A1" / "s0" / "ood23.json"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(json.dumps({"format": F.FMT_SCORE, "exam": "ood23"}))
        kept[("D1A1", 0, "ood23")] = {"file": "arms/D1A1/s0/ood23.json", "sha256": "k", "status": "kept"}
        e3 = refused_by(VTR.score_d1, reported=False)
        kept.clear()
        check("a kept ood23 pass committed without the loader fix: refused", e3 is not None and REFUSAL6 in str(e3)
              and "committed without the loader fix" in str(e3), e3)
        doc.write_text(json.dumps({"format": F.FMT_SCORE, "exam": "ood23", "loader_shape_fix": {
            "deviation": DEV6_FILE, "record_sha256": "0" * 64}}))
        kept[("D1A1", 0, "ood23")] = {"file": "arms/D1A1/s0/ood23.json", "sha256": "k", "status": "kept"}
        e4 = refused_by(VTR.score_d1, reported=False)
        doc.write_text(json.dumps({"format": F.FMT_SCORE, "exam": "ood23", "loader_shape_fix": {
            "deviation": DEV6_FILE, "record_sha256": sha(LD.record_path())}}))
        e5 = refused_by(VTR.score_d1, reported=False)
        kept.clear()
        check("a kept ood23 pass taken against another pre-flight record: refused; one taken against the record in "
              "place: kept", e4 is not None and REFUSAL6 in str(e4) and "another pre-flight record" in str(e4)
              and e5 is None, (e4, e5))
        check("  after each refusal the fix is not left installed", not getattr(ul_exif_size(), "_v3fam_dev6", False))
    finally:
        for (mod, name), val in saved.items():
            setattr(mod, name, val)


# ------------------------------------------------------------------ the record superseded
def t_supersede():
    print("identity_ood23.json of 2026-10-25 moved aside byte for byte")
    D5.clean()
    if SUP is None:
        check("inc2.v3fam_supersede exists", False)
        return
    eq = F.equiv_dir()
    eq.mkdir(parents=True, exist_ok=True)
    for n in ("identity_ood23.json", "identity_ood22.json"):
        (eq / n).write_bytes((FIX / n).read_bytes())
    snap = SR.v3fam_equivalence(C.INC_DIR)
    rec23 = next((r for r in snap["records"] if r["name"] == "identity_ood23.json"), {})
    check("in place, the snapshot reports identity_ood23.json superseded (by sha256), never stale; identity_ood22.json "
          "neither", rec23.get("superseded") is True and "identity_ood23.json" not in snap["stale"]
          and "identity_ood22.json" not in snap["superseded"] and "identity_ood23.json" in snap["superseded"], rec23)
    out = SUP.apply(C.INC_DIR, repo=GIT, job="t", groups=(SUP.GROUP_SCORES,))
    dst = eq / "superseded_2026-10-25" / "identity_ood23.json"
    mf = F._read_json(eq / "superseded_2026-10-25" / "MANIFEST.json") or {}
    ent = (mf.get("entries") or {}).get("identity_ood23.json") or {}
    check("the score jobs' group moves it into equivalence/superseded_2026-10-25/ (the same bytes, absent at its path), "
          "its manifest naming the sha256, deviation 6's file and its sha256 in the tree; identity_ood22.json stays",
          "v3fam/v1/equivalence/identity_ood23.json" in out["moved"] and dst.is_file() and sha(dst) == OOD23_SHA
          and not (eq / "identity_ood23.json").exists() and ent.get("sha256") == OOD23_SHA
          and ent.get("deviation") == DEV6_FILE and ent.get("deviation_sha256") == (sha(GIT / DEV6_FILE)
                                                                                    if (GIT / DEV6_FILE).is_file()
                                                                                    else "missing")
          and sha(eq / "identity_ood22.json") == OOD22_SHA, (out["moved"], ent))
    again = SUP.apply(C.INC_DIR, repo=GIT, job="t", groups=(SUP.GROUP_SCORES,))
    (eq / "identity_ood23.json").write_text('{"retaken": true}\n')
    other = SUP.apply(C.INC_DIR, repo=GIT, job="t", groups=(SUP.GROUP_SCORES,))
    check("  a second run changes nothing; the identity D1's job takes again (other bytes) is left in place",
          again["moved"] == [] and other["moved"] == [] and (eq / "identity_ood23.json").read_text()
          == '{"retaken": true}\n' and sha(dst) == OOD23_SHA, (again["moved"], other["moved"]))
    D5.clean()


# ------------------------------------------------------------------ the verdict's guard
def d1_world():
    """The cluster's records as of 2026-10-25 18:00Z: the verdicts of 2026-10-13 and 2026-10-22, the six score arms of
    deviation 5 but D1 complete (D6's too here, to isolate D1), D1's provenance as on the cluster."""
    cap = D5.oct22_world()
    shutil.rmtree(str(F.root() / "failed"), ignore_errors=True)
    for a in ("D8", "D4F", "D4N", "D4R15", "D5B", "D6"):
        TS._write_json(SYS.score_record_path(a), {"format": SYS.FMT_SCORE_RECORD, "arm": a, "status": "complete",
                                                 "exam": "dev", "dev": []})
    pdir = C.INC_DIR / "_campaign" / "provenance"
    (pdir / "v3fam_score_d1.json").write_bytes((FIX / "provenance_v3fam_score_d1.json").read_bytes())
    return cap


def t_verdict():
    print("the verdict's guard: D1 needs a record since deviation 6's reopening")
    cap = d1_world()
    before = {p.name: p.read_bytes() for p in cap.glob("*.json")}
    blk = next((b for b in SR.v3fam_reopen_blocks(DOM) if b.get("marker") == DEV6_MARKER), None)
    utc = getattr(VR, "DEV6_REOPEN_UTC", None)
    prov = json.loads((FIX / "provenance_v3fam_score_d1.json").read_text())
    starts = [a["started_utc"] for a in prov["attempts"]]
    check("the cluster guard's reopening equals the domain block's before_utc (2026-10-25T17:00:00Z): after D1's "
          "second failure (ticker 16:39:35Z) and both attempts of 2026-10-25 (15:20:48Z, 16:32:50Z), before any "
          "deploy", blk is not None and utc == blk.get("before_utc") == DEV6_UTC and utc > "2026-10-25T16:39:35Z"
          and all(s < utc for s in starts) and starts[-2:] == ["2026-10-25T15:20:48Z", "2026-10-25T16:32:50Z"],
          (utc, blk and blk.get("before_utc"), starts))
    ok_f, why_f = VR.verdict_ready(failed=["score:D1"])
    ok_n, why_n = VR.verdict_ready()
    check("with D1's attempts of 2026-10-25 only and --ticker-failed score:D1 (what L23L would pass): not ready, naming "
          "deviation 6, its file and D1", not ok_f and "deviation 6" in why_f and DEV6_FILE in why_f and "D1" in why_f
          and "D8" not in why_f, why_f)
    check("  and without it: not ready, D1 not scored since deviation 6 reopened it", not ok_n
          and "not scored since deviation 6 reopened it" in why_n and "D1" in why_n, why_n)
    ff = VR.failed_file("score:D1")
    ff.parent.mkdir(parents=True, exist_ok=True)
    ff.write_text(json.dumps({"format": "inc2-v3fam-failed/1", "item": "score:D1", "status": "failed",
                              "written_utc": "2026-10-25T16:39:35.000000Z"}))
    st_early = VR.arm_status("D1")
    ff.write_text(json.dumps({"format": "inc2-v3fam-failed/1", "item": "score:D1", "status": "failed",
                              "written_utc": "2026-10-25T18:10:00.000000Z"}))
    st_late = VR.arm_status("D1")
    os.unlink(str(ff))
    check("  a failure record of score:D1 written before the reopening (16:39:35Z) is not read; one written since is",
          st_early == "missing" and st_late == "failed (score:D1 failed twice)", (st_early, st_late))
    e = refused_by(VR.verdict, ticker_failed=["score:D1"])
    check("  the verb refuses with --ticker-failed score:D1, naming deviation 6: no second continuation written, the "
          "earlier verdicts unchanged byte for byte, no failure record written", e is not None and "deviation 6" in str(e)
          and not (cap / "v3fam_v1_cont2.json").exists() and not (cap / "v3fam_v1_cont2_external.json").exists()
          and all((cap / n).read_bytes() == b for n, b in before.items()) and not ff.exists(), e)
    D5.attempt_since("v3fam_score_d1", utc="2026-10-25T18:00:00Z")
    ok_a, why_a = VR.verdict_ready(failed=["score:D1"])
    ok_a0, why_a0 = VR.verdict_ready()
    check("  with a D1 job since the reopening (its provenance), the ticker's failure ends D1: ready; without the "
          "failure named still not ready", ok_a and not ok_a0 and "D1" in why_a0, (why_a, why_a0))
    d1_world()
    TS._write_json(SYS.score_record_path("D1"), {"format": SYS.FMT_SCORE_RECORD, "arm": "D1", "status": "complete",
                                                "exam": "dev", "dev": []})
    ok_c, why_c = VR.verdict_ready()
    check("  D1's complete score record: ready (nothing named)", ok_c, why_c)
    cont = getattr(VR, "continuation_check", None)
    src = open(VR.__file__).read()
    check("  the second continuation names deviation 6 among its deviations",
          cont is not None and "deviations=[DEVIATION_4, DEVIATION_5, DEVIATION_6]" in src)
    D5.clean()


# ------------------------------------------------------------------ the ticker
def live_state():
    d = json.loads((FIX / "state_v3fam_2026-10-25.json").read_text())
    return {"v3fam": {"items": d["items"]}, "failed_ids": list(d["failed_ids"]), "declined": []}


def ticker_arts():
    a = D5.arts()
    for arm in ("D8", "D4F", "D4N", "D4R15", "D5B"):
        a["capacity/v3fam_score_%s.json" % arm.lower()] = {"status": "complete"}
    return a


def t_reopen():
    print("deviation 6's reopening on the ticker's live items of 2026-10-25")
    root = VT.TMP / "reopen6"
    blk = next((b for b in SR.v3fam_reopen_blocks(DOM) if b.get("marker") == DEV6_MARKER), None)
    loop = (GIT / "docs" / "CONTINUOUS_LOOP.md").read_bytes()
    check("the domain's deviation 6 block: score:D1 only, before 2026-10-25T17:00:00Z, its file shipped with this "
          "sha256 and kept byte for byte in docs/CONTINUOUS_LOOP.md; the last of the reopen blocks", blk is not None
          and blk["items"] == ["score:D1"] and blk["before_utc"] == DEV6_UTC and blk["file"] == DEV6_FILE
          and (GIT / DEV6_FILE).is_file() and sha(GIT / DEV6_FILE) == blk["sha256"]
          and (GIT / DEV6_FILE).read_bytes() in loop and SR.v3fam_reopen_blocks(DOM)[-1] is blk
          and FAM["verdict_record"] == "capacity/v3fam_v1_cont2.json", blk and {k: blk[k] for k in blk if k != "why"})
    dep = ("docs/prereg/V3F_dev4_2026-10-20.md", D5.DEV5_FILE, DEV6_FILE)
    st0 = live_state()
    before = json.loads(json.dumps(st0["v3fam"]["items"]))
    f = D5.Fam(root, st0, ticker_arts(), deploy=dep)
    f.utc = "2026-10-26T00:09:35Z"
    got = f._v3fam_reopen()
    it = f.st["v3fam"]["items"]
    d1 = it["score:D1"]
    check("deployed: score:D1 (failed twice on 2026-10-25, last started 16:29:35Z) reopened once and nothing else: "
          "missing, fails 0, deviation 6's marker, its history keeping deviation 5's reopening and the failed record",
          got == ["score:D1"] and d1["state"] == "missing" and d1["fails"] == 0 and d1["reopened"] == DEV6_MARKER
          and [h.get("marker") for h in d1["reopen_history"]] == [D5.DEV5_MARKER, DEV6_MARKER]
          and d1["reopen_history"][-1]["state"] == "failed" and d1["reopen_history"][-1]["fails"] == 2
          and all(it[k] == before[k] for k in it if k != "score:D1"), (got, d1.get("state"), d1.get("reopened")))
    check("  every attempt stays counted against the family cap (prior_gpuh 1.0684 + 0.0928 = 1.1612); the ids of "
          "2026-10-25 retired (620adf36..., f6ba1f35...) beside the earlier ones; one card",
          abs(float(d1.get("prior_gpuh") or 0) - 1.1612) < 1e-9
          and {"620adf36840ee82121a88537d40ac43f", "f6ba1f35efd7a9f6c1e20375310f5c1a",
               "8dca3a02bdcc3f20c0a5cbbb84ebf13e", "59a6a52876f3b3f6203893caab88acbc"} <= set(f.st["failed_ids"])
          and len([c for c in f.cards if DEV6_MARKER in c["title"]]) == 1, (d1.get("prior_gpuh"), f.cards))
    again = f._v3fam_reopen()
    fs = f._v3fam_stage(D5.BUILT)
    ready, pending = DS.v3fam_verdict_ready(FAM, fs)
    plan, _waits = DS.v3fam_plan(FAM, fs)
    check("  once per marker; then L23L waits for D1 (pending) and L23J D1 is proposed on MAINT", again == []
          and not ready and "D1" in pending and plan.get("MAINT") == {"lever": "L23J", "arm": "D1", "item": "score:D1"},
          (again, pending, plan))
    p, _c = f._v3fam_params("L23L", {}, "inc2", {})
    check("  L23L's argv no longer carries --ticker-failed score:D1", "score:D1" not in json.dumps(p), p)
    none = D5.Fam(root / "none", live_state(), ticker_arts(), deploy=dep[:2])
    late_st = live_state()
    late_st["v3fam"]["items"]["score:D1"]["started_utc"] = "2026-10-25T17:30:00Z"
    late = D5.Fam(root / "late", late_st, ticker_arts(), deploy=dep)
    check("  never reopened while the deployed tree lacks the deviation's file, nor an attempt started after "
          "2026-10-25T17:00:00Z", none._v3fam_reopen() == [] and late._v3fam_reopen() == []
          and none.st["v3fam"]["items"]["score:D1"]["state"] == "failed"
          and late.st["v3fam"]["items"]["score:D1"]["state"] == "failed")


def t_refusal():
    print("a refusal of the loader check, or of E3's ground-truth and frame checks, is the platform's: held for a person")
    tb = ('  File "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/weed_optimizer_framework/tools/inc2/'
          'v3fam_trained.py", line 455, in score_d1\n    x["loader_shape_fix"] = obs.check(...)\n'
          'weed_optimizer_framework.tools.inc2.v3fam.V3famRefused: %s: D1 A1 seed 0 on ood23: it saw 0 of the exam\'s '
          '1784 images' % REFUSAL6)
    geo = json.loads((FIX / "provenance_v3fam_score_d1.json").read_text())["attempts"][-1]["refusal"]
    arm_own = ("job(s) 7 ended FAILED; refused: weed_optimizer_framework.tools.inc2.v3fam.V3famRefused: E3's path "
               "refuses: the two-stage pass recorded no stage-1 boxes for exam image ood23__x")
    rx = S.V3FAM_PLATFORM_REFUSAL_RE
    px = getattr(S, "V3FAM_PERSON_REFUSAL_RE", None)
    check("the loader check's refusal and E3's ground-truth geometry refusal of 2026-10-25 (the cluster's provenance) "
          "are preconditions of the platform's own making, both deterministic (held for a person); an E3 refusal that "
          "rests on the arm's boxes is neither", rx.search(tb) is not None and GT_GEOMETRY in geo
          and rx.search(geo) is not None and px is not None and px.search(tb) is not None
          and px.search(geo) is not None and rx.search(arm_own) is None and px.search(arm_own) is None, geo[-160:])
    dep = ("docs/prereg/V3F_dev4_2026-10-20.md", D5.DEV5_FILE, DEV6_FILE)

    def d1_job():
        # D1's score job after deviation 6's deploy (started after the block's before_utc: never reopened by it)
        return dict(D5.running("D1"), started_utc="2026-10-26T00:10:00Z", reopened=DEV6_MARKER)
    f = D5.Fam(VT.TMP / "refusal6", {"v3fam": {"items": {"score:D1": d1_job()}}, "declined": []},
               ticker_arts(), deploy=dep)
    f._v3fam_failed("MAINT", D5.item("D1", "r1"), "job(s) 5 ended FAILED; refused: %s" % tb, True)
    r = f.st["v3fam"]["items"]["score:D1"]
    card = f.cards[-1] if f.cards else {}
    check("  the loader check: held for a person at the first refusal, no fails counted, one card naming the loader "
          "check (not the reproduction records' remedy)", r["state"] == DS.V3FAM_HELD_PLATFORM and not r.get("fails")
          and r.get("held_for_person") is True and "held for a person" in card.get("title", "")
          and "loader_shapes.json" in card.get("detail", "") and "restore the code they pin" not in card.get("detail", "")
          and len(f.cards) == 1, (r.get("state"), card))
    fs = f._v3fam_stage(D5.BUILT)
    plan, waits = DS.v3fam_plan(FAM, fs)
    p23l, _c = f._v3fam_params("L23L", {}, "inc2", {})
    check("  never released although the snapshot reports the records current: the stage names it held for a person, "
          "no L23J D1 is proposed, the planner's wait says why, L23L's argv names no D1 failure",
          (fs.get("score") or {}).get("D1") == DS.V3FAM_HELD_PLATFORM and fs.get("held_person") == ["score:D1"]
          and r["state"] == DS.V3FAM_HELD_PLATFORM and (plan.get("MAINT") or {}).get("arm") != "D1"
          and "held for a person" in str(waits.get("score:D1")) and "score:D1" not in json.dumps(p23l)
          and not [e for e in f.ledger if e.get("event") == "v3fam_released"], (fs.get("score", {}).get("D1"),
                                                                                waits.get("score:D1")))
    g = D5.Fam(VT.TMP / "refusal6g", {"v3fam": {"items": {"score:D1": d1_job()}}, "declined": []},
               ticker_arts(), deploy=dep)
    g._v3fam_failed("MAINT", D5.item("D1", "r2"), "job(s) 49821128 ended FAILED; refused: %s" % geo, True)
    rg = g.st["v3fam"]["items"]["score:D1"]
    cg = g.cards[-1] if g.cards else {}
    check("  E3's geometry refusal (as on 2026-10-25): held for a person, never counted as D1's failure, its card "
          "naming E3's ground-truth and frame checks", rg["state"] == DS.V3FAM_HELD_PLATFORM and not rg.get("fails")
          and rg.get("held_for_person") is True and "ground-truth and frame checks" in cg.get("detail", "")
          and not [e for e in g.ledger if e.get("event") == "failed"], (rg.get("state"), cg.get("title")))
    blk = next((b for b in SR.v3fam_reopen_blocks(DOM) if b.get("marker") == DEV6_MARKER), None) or {}
    g.utc = "2026-10-27T01:00:00Z"
    later = dict(blk, marker="a later deviation (test)", before_utc="2026-10-27T00:00:00Z")
    got = g._v3fam_reopen_block(later)
    check("  a later recorded deviation's reopen block proposes it again: missing, the person hold cleared",
          got == ["score:D1"] and rg["state"] == "missing" and "held_for_person" not in rg, (got, rg.get("state")))
    h = D5.Fam(VT.TMP / "refusal6h", {"v3fam": {"items": {"score:D1": d1_job()}}, "declined": []},
               ticker_arts(), deploy=dep)
    h._v3fam_failed("MAINT", D5.item("D1", "r3"), arm_own, True)
    rh = h.st["v3fam"]["items"]["score:D1"]
    check("  an E3 refusal that rests on the arm's own boxes stays the arm's failure (failed once)",
          rh["state"] == "failed_once" and rh.get("fails") == 1 and not rh.get("held_for_person"), rh.get("state"))


# ------------------------------------------------------------------ the deploy
def t_deploy():
    print("the deploy")
    sh = (PKG / "deploy" / "deploy_funnel.sh").read_text()
    roots = sh.split("ROOT_PATHS=(", 1)[1].split(")", 1)[0].split()
    ok, lines = SR.v3fam_deploy_check(DOM, GIT, roots)
    code = F.code_shas()
    check("deploy_funnel.sh ships the deviation's file among its git-root paths and the deploy check passes with no "
          "flag; the code every reproduction record pins is unchanged (inc2/v3fam.py 3d776dae..., the domain's "
          "equivalence_code)", ok and DEV6_FILE in roots and code.get("v3fam") == B6C4AEF
          and code == FAM["equivalence_code"], lines)
    mods = (PKG / "run_inc2_build.sh").read_text().split("MODULES=(", 1)[1].split(")", 1)[0]
    check("  the job script's module list (outer and nested copies compared) names inc2/v3fam_loader.py",
          "tools/inc2/v3fam_loader.py" in mods.split())
    t_descends(sh)


def t_descends(sh):
    print("a deploy whose head does not descend from the deployed head is refused")
    import subprocess
    fn = getattr(SR, "deploy_descends", None)
    if fn is None:
        check("stream_remote.deploy_descends exists", False)
        return
    g = VT.TMP / "descends_git"
    shutil.rmtree(str(g), ignore_errors=True)
    g.mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")

    def git(*a):
        return subprocess.run(["git", "-C", str(g)] + list(a), capture_output=True, text=True, env=env,
                              check=True).stdout.strip()
    git("init", "-q")
    shas = []
    for n in ("a", "b"):
        (g / n).write_text(n)
        git("add", n)
        git("commit", "-q", "-m", n)
        shas.append(git("rev-parse", "HEAD"))
    git("checkout", "-q", "-b", "old", shas[0])
    (g / "c").write_text("c")
    git("add", "c")
    git("commit", "-q", "-m", "c")
    old_head = git("rev-parse", "HEAD")
    rec = lambda h: json.dumps({"git_head": h, "script": "deploy/deploy_funnel.sh"})  # noqa: E731
    ok_same = fn(g, rec(old_head))[0]
    git("checkout", "-q", shas[1])
    ok_desc = fn(g, rec(shas[0]))[0] and fn(g, rec(shas[1]))[0]
    bad_old, why_old = fn(g, rec(old_head))
    bad_unknown, why_unknown = fn(g, rec("f" * 40))
    first = fn(g, "")[0] and fn(g, "not json")[0]
    check("stream_remote.deploy_descends: a head that holds the deployed commit passes; one cut before it (a branch "
          "from an older main) is refused, as is a deployed head this checkout does not know; no record passes",
          ok_same and ok_desc and not bad_old and "does not descend" in why_old and not bad_unknown
          and "not in this checkout" in why_unknown and first, (why_old, why_unknown))
    pre = sh.split('echo "== 0) pre-flight', 1)[0]
    check("  deploy_funnel.sh runs it on the lab's stream_deployed.json before its pre-flight and any copy, refusing "
          "(exit 3) unless --allow-older-head; --dry-run stays without network", "S.deploy_descends(" in pre
          and "stream_deployed.json" in pre and 'if [ "$ALLOW_OLDER" != 1 ]; then' in pre
          and 'echo "refused: $DESCENDS; nothing was copied" >&2\n    exit 3' in pre
          and pre.index("S.deploy_descends(") > pre.index('if [ "$DRY" = 1 ]; then'), None)


def main():
    try:
        for fn in (t_fixture, t_geometry, t_preflight, t_score_d1, t_supersede, t_verdict, t_reopen, t_refusal,
                   t_deploy):
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-1500:]))
    finally:
        shutil.rmtree(str(VT.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
