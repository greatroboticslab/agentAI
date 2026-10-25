"""V3F deviation 6 (2026-10-25; docs/prereg/V3F_dev6_2026-10-25.md): the image size Ultralytics' label cache
records for a photo PIL reads as MPO, during D1's passes on ood22 and ood23, and the pre-flight that proves every
exam image's frame and letterbox before those passes.

    python -m weed_optimizer_framework.tools.inc2.v3fam_loader status       (read-only: the record, if written)
    python -m weed_optimizer_framework.tools.inc2.v3fam_loader preflight    (D1's score job runs it first; once)

The defect. Ultralytics 8.4.37 (the locked scorer's pin) builds each validation dataset's label cache with
ultralytics.data.utils.exif_size, which swaps width and height for EXIF orientation 6 or 8 only when PIL names the
file's format "JPEG" (the same gate is in 8.4.129). An iPhone photo carrying an MPF segment is named "MPO" by PIL, so
for such a photo tagged 6 or 8 the cache records the stored (landscape) size while the loader (cv2, which applies the
EXIF orientation) returns the portrait frame. The validator sorts its rect batches by the cached aspect ratio, puts
these images in landscape batches, and LetterBox(scaleup=False) shrinks each by a further 0.8 or 0.9 that ratio_pad
does not record. Ultralytics' own metrics stay self-consistent (they match in letterbox space); E3's inverse
geometry (twostage.to_original) maps every box back to that factor times its place, and E3's ground-truth round
trip refuses. 289 of ood23's 1,784 images are such photos (257 tagged 6, 32 tagged 8); ood22 has none; dev,
ImageWeeds, ooddev_v1 and the probe allow only tags none, 0 and 1. D1's score jobs 49820090 and 49821128
(2026-10-25) were refused on ood23__20230804_HTRC_iPhone12_WY_2 (0.109094 = 0.2 x its box's cx 0.545470).

The fix (loader_fix(exam), D1's passes on ood22 and ood23 only). While the context is open,
ultralytics.data.utils.exif_size is Ultralytics' own function followed by one step: when PIL names the file MPO,
Ultralytics' function returned the stored size unchanged and the EXIF orientation is 6 or 8, width and height are
swapped, as Ultralytics does for a JPEG. Its only caller in the validation path is verify_image_label, which
YOLODataset.cache_labels runs in threads of this process before any DataLoader worker starts; the locked scorer
validates a fresh copy of the exam each call, so no cache file written earlier is read. The original is restored on
exit. Every call is observed (Observer): after a pass the observer must have seen every exam image once per
locked-scorer call, the images whose size it changed must be exactly those the pre-flight record lists, and every
one of them must be MPO tagged 6 or 8; else the pass refuses before anything is committed.

The pre-flight (preflight(), v3fam/v1/exif/loader_shapes.json, written once). For every image of ood22 and ood23
(the locked scorer's exam check first: every image and label hashing as its manifest records), on a copy of the
exam laid out as the locked scorer lays it out (inc.scorer.build_view): the format and EXIF orientation PIL reads,
the stored size, the size Ultralytics' exif_size records, the size under the fix, and the frame the loader decodes
(ultralytics.utils.patches.imread, BaseDataset.load_image's call, every image decoded). Then Ultralytics' own
validation dataset of the exam (DetectionValidator.build_dataset: rect batches of 32 at 640, stride 32, pad 0.5),
built without and with the fix, gives each image's batch shape, and from it the scale LetterBox(scaleup=False)
applies after load_image's resize; ratio_pad records the resize and the padding only, so a scale under 1 is a box
E3's geometry maps to the wrong place. It refuses, writing nothing, unless under the fix the cached size equals the
loader's frame for every image, every changed image is MPO tagged 6 or 8 (and every such image is changed), and the
scale is 1 for every image; the scales without the fix are recorded beside (the defect's extent). The EXIF frame
pre-flight of V3F.3.2 (inc2.v3fam.exif_preflight, kept) proved the loader's frame on one image per tag; this record
proves the cached size and the letterbox on every image D1 reads under the fix.

The identity reproduction of an exam whose record lists changed images is taken under the fix too (identity()):
both of its locked-scorer calls observed, and a note (v3fam/v1/exif/loader_identity_<exam>.json, written once) names
its sha256; an identity record in place without that note is refused for such an exam. The record of 2026-10-25
(identity_ood23.json, sha256 3c297647..., taken without the fix) is moved aside by inc2.v3fam_supersede first.

Every refusal here starts with REFUSAL ("deviation 6's loader check refuses"): a precondition of the platform's own
making, which the ticker holds instead of counting as the arm's failure (inc_autopilot.stream,
V3FAM_PLATFORM_REFUSAL_RE).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from . import v3fam as F

DEVIATION_6 = "docs/prereg/V3F_dev6_2026-10-25.md"
FORMAT = "inc2-v3fam-loader-shapes/1"
FMT_IDENTITY = "inc2-v3fam-loader-identity/1"
RECORD = "loader_shapes.json"                # under v3fam/v1/exif/
IDENTITY_NOTE = "loader_identity_%s.json"    # under v3fam/v1/exif/
EXAMS = ("ood22", "ood23")                   # D1's deciding exams, the only ones that read tags 3, 6 and 8
FIX_FORMATS = ("MPO",)                       # what the fix adds to Ultralytics' own JPEG gate
SWAP_TAGS = (6, 8)                           # EXIF orientations that swap width and height (Ultralytics' set)
ORIENTATION = 274
IMGSZ = S.IMGSZ                              # 640: the locked scorer's
BATCH = S.BATCH                              # 32: the locked scorer's (Model.val sets rect=True)
STRIDE = 32                                  # the detectors' largest stride (YOLO11)
REFUSAL = "deviation 6's loader check refuses"
DECODE_THREADS = 8


def log(msg):
    print("[inc2.v3fam_loader] %s" % msg, flush=True)


def refuse(msg):
    raise F.V3famRefused("%s: %s" % (REFUSAL, msg))


def record_path():
    return F.exif_dir() / RECORD


def note_path(exam):
    return F.exif_dir() / (IDENTITY_NOTE % exam)


def _keys_sha(keys):
    return C.sha256_text("\n".join(sorted(keys)))


def _ul_utils():
    import ultralytics.data.utils as U
    return U


# ------------------------------------------------------------------ the fix
def _tag(img):
    try:
        exif = img.getexif()
        return exif.get(ORIENTATION, None) if exif else None
    except Exception:                        # noqa: BLE001 - as Ultralytics: an unreadable EXIF block is no tag
        return None


def fixed_size(orig, img):
    """(W, H) under the fix: Ultralytics' exif_size (orig), then for a file PIL names MPO whose size it returned
    unchanged and whose EXIF orientation is 6 or 8, width and height swapped, as Ultralytics swaps them for a JPEG."""
    s = tuple(orig(img))
    if getattr(img, "format", None) in FIX_FORMATS and s == tuple(img.size) and _tag(img) in SWAP_TAGS:
        return (s[1], s[0])
    return s


class Observer(object):
    """Every exif_size call made while the fix is installed: per image key (the file's stem, which the locked
    scorer's exam layout names after the manifest key), how often it was called, how often the fix changed the size,
    and any change on an image that is not MPO tagged 6 or 8 (never, by construction; checked all the same)."""

    def __init__(self, exam):
        self.exam = exam
        self.lock = threading.Lock()
        self.calls = collections.Counter()
        self.changed = collections.Counter()
        self.bad = []
        self.unnamed = 0

    def see(self, img, before, after):
        name = getattr(img, "filename", None)
        key = os.path.splitext(os.path.basename(str(name)))[0] if name else None
        with self.lock:
            if not key:
                self.unnamed += 1
                return
            self.calls[key] += 1
            if tuple(after) != tuple(before):
                self.changed[key] += 1
                if getattr(img, "format", None) not in FIX_FORMATS or _tag(img) not in SWAP_TAGS:
                    self.bad.append((key, getattr(img, "format", None), _tag(img)))

    def check(self, keys, expected, builds, what):
        """Refuse unless the fix saw each of keys `builds` times (one label cache per locked-scorer call), changed
        exactly the images in expected (each `builds` times) and nothing else. Returns the record a pass carries."""
        keys, expected = set(keys), set(expected)
        probs = []
        if self.unnamed:
            probs.append("%d call(s) on an image without a file name" % self.unnamed)
        seen = set(self.calls)
        if seen != keys:
            probs.append("it saw %d of the exam's %d images (missing %s, extra %s)" % (
                len(seen & keys), len(keys), sorted(keys - seen)[:3], sorted(seen - keys)[:3]))
        off = sorted(k for k in seen & keys if self.calls[k] != builds)
        if off:
            probs.append("%d image(s) not seen exactly %d time(s) (%s)" % (len(off), builds, off[:3]))
        if set(self.changed) != expected:
            probs.append("it changed %d image(s), the record lists %d (not listed %s, not changed %s)" % (
                len(self.changed), len(expected), sorted(set(self.changed) - expected)[:3],
                sorted(expected - set(self.changed))[:3]))
        offc = sorted(k for k in set(self.changed) & expected if self.changed[k] != builds)
        if offc:
            probs.append("%d listed image(s) not changed in each of the %d build(s) (%s)" % (len(offc), builds,
                                                                                           offc[:3]))
        if self.bad:
            probs.append("it changed image(s) that are not MPO tagged 6 or 8: %s" % self.bad[:3])
        if probs:
            refuse("%s on %s: %s" % (what, self.exam, "; ".join(probs)))
        return {"deviation": DEVIATION_6, "record": "v3fam/%s/exif/%s" % (F.NAME, RECORD),
                "record_sha256": F._sha(record_path()), "builds": int(builds), "images": len(keys),
                "changed": len(expected), "changed_sha256": _keys_sha(expected)}


@contextlib.contextmanager
def loader_fix(exam):
    """ultralytics.data.utils.exif_size under the fix for the block (D1's passes on ood22 and ood23 only), every call
    observed; the original restored on exit. Yields the Observer. Never nested, never for another exam."""
    if exam not in EXAMS:
        refuse("the loader fix is for %s only, not %s" % (list(EXAMS), exam))
    U = _ul_utils()
    orig = U.exif_size
    if getattr(orig, "_v3fam_dev6", False):
        refuse("the loader fix is installed already: it is never nested")
    obs = Observer(exam)

    def exif_size(img):
        before = tuple(orig(img))
        after = fixed_size(lambda _im: before, img)
        obs.see(img, before, after)
        return after
    exif_size._v3fam_dev6 = True
    exif_size.__wrapped__ = orig
    U.exif_size = exif_size
    try:
        yield obs
    finally:
        U.exif_size = orig


@contextlib.contextmanager
def deciding(exam):
    """loader_fix(exam) on ood22 and ood23, nothing on any other exam (dev, ooddev_v1). Yields the Observer or None."""
    if exam in EXAMS:
        with loader_fix(exam) as obs:
            yield obs
    else:
        yield None


def installed():
    """True while the fix is installed (a test reads it inside a pass)."""
    return bool(getattr(_ul_utils().exif_size, "_v3fam_dev6", False))


# ------------------------------------------------------------------ the pre-flight record
def _versions():
    out = {}
    for mod, attr in (("ultralytics", "__version__"), ("PIL", "__version__"), ("cv2", "__version__"),
                      ("numpy", "__version__")):
        try:
            out[mod] = getattr(__import__(mod), attr)
        except Exception:                    # noqa: BLE001 - recorded as missing
            out[mod] = None
    return out


def _header(path, orig):
    """(format, tag, stored (W, H), Ultralytics' cached (W, H), the fix's (W, H)), as verify_image_label reads it:
    PIL open, verify, exif_size."""
    from PIL import Image
    with Image.open(path) as im:
        im.verify()
        return (im.format, _tag(im), tuple(im.size), tuple(orig(im)), tuple(fixed_size(orig, im)))


def _loader_hw(path):
    import cv2
    from ultralytics.utils.patches import imread
    a = imread(str(path), flags=cv2.IMREAD_COLOR)
    return None if a is None else (int(a.shape[0]), int(a.shape[1]))


def _val_dataset(images_dir, save_dir):
    """Ultralytics' own validation dataset of a view, as the validator builds it for the locked scorer's model.val
    (rect batches of BATCH at IMGSZ, stride STRIDE, pad 0.5): its sorted image files, batch index and batch shapes."""
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator
    args = get_cfg(overrides={"imgsz": IMGSZ, "batch": BATCH, "rect": True, "mode": "val", "task": "detect",
                              "workers": 0, "plots": False, "save_json": False, "save_txt": False})
    v = DetectionValidator(save_dir=Path(save_dir), args=args)
    v.data = {"names": {i: n for i, n in enumerate(C.CLASS_NAMES)}, "nc": C.NC, "channels": 3}
    v.stride = STRIDE
    ds = v.build_dataset(str(images_dir), mode="val", batch=BATCH)
    if not getattr(ds, "rect", False) or getattr(ds, "batch_shapes", None) is None:
        refuse("Ultralytics' validation dataset has no rect batches (rect=%r)" % getattr(ds, "rect", None))
    return ds


def scales(ds, loader_hw):
    """{key: the scale LetterBox(scaleup=False) applies after load_image's resize} for each image of a validation
    dataset (BaseDataset.load_image: the long side to IMGSZ; LetterBox: min(batch shape / resized shape), at most 1;
    ratio_pad records the resize and the padding, never this scale)."""
    out = {}
    for i, f in enumerate(ds.im_files):
        key = Path(f).stem
        h0, w0 = loader_hw[key]
        r = IMGSZ / max(h0, w0)
        w, h = (min(math.ceil(w0 * r), IMGSZ), min(math.ceil(h0 * r), IMGSZ)) if r != 1 else (w0, h0)
        bh, bw = (int(x) for x in ds.batch_shapes[int(ds.batch[i])])
        out[key] = min(1.0, bh / float(h), bw / float(w))
    return out


def measure(exam, threads=None):
    """One exam's measurements (module docstring): {manifest_sha256, rows (per image), changed, scale_unfixed,
    scale_fixed, observer}. Read-only for the exam; its views are temporary copies laid out as the locked scorer's."""
    U = _ul_utils()
    orig = U.exif_size
    if getattr(orig, "_v3fam_dev6", False):
        refuse("the loader fix is installed while the pre-flight measures Ultralytics' own sizes")
    mp = C.manifest_path(exam)
    if not mp.is_file():
        refuse("no manifest for %s at %s" % (exam, mp))
    rows = sorted(C.read_manifest(mp), key=lambda r: r["key"])
    try:
        files, labels = S.check_exam(exam, rows)
    except S.ScorerRefused as e:
        refuse("the locked scorer's exam check on %s: %s" % (exam, e))
    out = {"manifest_sha256": C.sha256_file(mp), "images": len(rows)}
    with tempfile.TemporaryDirectory(prefix="inc2_v3fam_loader_") as tmp:
        # two copies of the exam's layout: Ultralytics writes labels.cache beside each and reads it back on the next
        # build of the same directory (keyed by file sizes and paths), so a build under the fix in the directory of a
        # build without it would read the unfixed sizes and never call exif_size (the locked scorer validates a fresh
        # copy at each call for the same reason)
        view, view1 = Path(tmp) / "view", Path(tmp) / "view_fixed"
        S.build_view(view, exam, rows, files, labels)
        S.build_view(view1, exam, rows, files, labels)
        paths = {r["key"]: view / "images" / files[r["key"]] for r in rows}
        head = {k: _header(p, orig) for k, p in paths.items()}
        n = max(1, min(int(threads or DECODE_THREADS), os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=n) as ex:
            hw = dict(zip(list(paths), ex.map(_loader_hw, list(paths.values()))))
        unread = sorted(k for k, v in hw.items() if v is None)
        if unread:
            refuse("the loader read nothing for %d image(s) of %s (%s)" % (len(unread), exam, unread[:3]))
        ds0 = _val_dataset(view / "images", Path(tmp) / "runs0")
        s0 = scales(ds0, hw)
        del ds0
        with loader_fix(exam) as obs:
            ds1 = _val_dataset(view1 / "images", Path(tmp) / "runs1")
        s1 = scales(ds1, hw)
        del ds1
    if set(s0) != set(paths) or set(s1) != set(paths):
        refuse("Ultralytics' validation dataset of %s holds %d / %d of its %d images" % (exam, len(s0), len(s1),
                                                                                    len(paths)))
    table, changed = [], []
    for k in sorted(paths):
        fmt, tag, raw, cache, fixed = head[k]
        h, w = hw[k]
        table.append([k, fmt, tag, raw[0], raw[1], cache[0], cache[1], fixed[0], fixed[1], w, h,
                      round(s0[k], 6), round(s1[k], 6)])
        if fixed != cache:
            changed.append(k)
    out.update(rows=table, changed=changed, scale_unfixed=s0, scale_fixed=s1, observer=obs)
    return out


def _problems(exam, m):
    """What the record refuses on (module docstring)."""
    probs = []
    by = {r[0]: r for r in m["rows"]}
    off = [k for k, r in by.items() if (r[7], r[8]) != (r[9], r[10])]
    if off:
        probs.append("%d image(s) whose size under the fix is not the loader's frame (%s)" % (len(off), [
            "%s: fixed %dx%d, loader %dx%d" % (k, by[k][7], by[k][8], by[k][9], by[k][10]) for k in sorted(off)[:3]]))
    mpo = sorted(k for k, r in by.items() if r[1] in FIX_FORMATS and r[2] in SWAP_TAGS)
    if sorted(m["changed"]) != mpo:
        probs.append("the fix changed %d image(s), %d are MPO tagged 6 or 8 (%s)" % (
            len(m["changed"]), len(mpo), sorted(set(m["changed"]) ^ set(mpo))[:3]))
    shrunk = sorted(k for k, s in m["scale_fixed"].items() if s < 1.0)
    if shrunk:
        probs.append("%d image(s) still shrunk by the letterbox under the fix (%s)" % (len(shrunk), shrunk[:3]))
    try:
        m["observer"].check([r[0] for r in m["rows"]], m["changed"], 1, "the pre-flight's dataset")
    except F.V3famRefused as e:
        probs.append(str(e)[len(REFUSAL) + 2:])
    return ["%s: %s" % (exam, p) for p in probs]


def preflight(exams=EXAMS, threads=None):
    """The pre-flight record (module docstring), written once; the one in place checked and returned otherwise
    (require_record). Returns the record."""
    p = record_path()
    if p.exists():
        return require_record()
    t0 = time.time()
    per, probs = {}, []
    for exam in exams:
        m = measure(exam, threads=threads)
        probs += _problems(exam, m)
        s0 = m["scale_unfixed"]
        shrunk0 = sorted(k for k, s in s0.items() if s < 1.0)
        tags = collections.Counter("%s/%s" % (r[1], r[2]) for r in m["rows"])
        per[exam] = {"manifest_sha256": m["manifest_sha256"], "images": m["images"],
                     "changed": sorted(m["changed"]), "changed_n": len(m["changed"]),
                     "changed_sha256": _keys_sha(m["changed"]), "format_tag": dict(sorted(tags.items())),
                     "fixed_equals_loader": all((r[7], r[8]) == (r[9], r[10]) for r in m["rows"]),
                     "scale_fixed_min": min(m["scale_fixed"].values()) if m["rows"] else None,
                     "unfixed_shrunk": len(shrunk0), "unfixed_shrunk_sha256": _keys_sha(shrunk0),
                     "unfixed_scales": dict(sorted(collections.Counter(
                         str(round(s, 4)) for s in s0.values() if s < 1.0).items())),
                     "columns": ["key", "format", "tag", "stored_w", "stored_h", "cache_w", "cache_h", "fixed_w",
                                 "fixed_h", "loader_w", "loader_h", "scale_unfixed", "scale_fixed"],
                     "rows": m["rows"]}
        log("%s: %d images, the fix changes %d (MPO tagged 6 or 8); without it %d shrunk by the letterbox, with it "
            "%d" % (exam, m["images"], len(m["changed"]), len(shrunk0),
                    sum(1 for s in m["scale_fixed"].values() if s < 1.0)))
    if probs:
        refuse("the pre-flight (%s) found: %s" % (record_path(), "; ".join(probs)[:1500]))
    doc = {"format": FORMAT, "deviation": DEVIATION_6, "exams": per,
           "batches": {"imgsz": IMGSZ, "batch": BATCH, "stride": STRIDE, "pad": 0.5, "rect": True,
                       "how": "inc.scorer.build_view, then ultralytics DetectionValidator.build_dataset"},
           "versions": _versions(), "decided_by": F.DECIDED_BY,
           "code": dict(F.code_shas(), v3fam_loader=C.sha256_file(Path(__file__).resolve())),
           "job": os.environ.get("SLURM_JOB_ID") or None, "seconds": round(time.time() - t0, 3),
           "written_utc": F._utc()}
    F.write_once(p, doc)
    return require_record()


def require_record():
    """The pre-flight record, checked: written by this deviation, every exam's manifest unchanged, taken with the
    Ultralytics that runs now, the fix's size equal to the loader's frame and the letterbox's scale 1 for every
    image. Refused otherwise."""
    p = record_path()
    d = F._read_json(p)
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        refuse("no pre-flight record at %s (inc2.v3fam_loader preflight)" % p)
    probs = []
    now = _versions().get("ultralytics")
    if (d.get("versions") or {}).get("ultralytics") != now:
        probs.append("taken with Ultralytics %s, this is %s" % ((d.get("versions") or {}).get("ultralytics"), now))
    for exam in EXAMS:
        e = (d.get("exams") or {}).get(exam) or {}
        mp = C.manifest_path(exam)
        if e.get("manifest_sha256") != (C.sha256_file(mp) if mp.is_file() else None):
            probs.append("%s's manifest changed since" % exam)
        if e.get("fixed_equals_loader") is not True or e.get("scale_fixed_min") not in (1, 1.0) \
                or e.get("changed_sha256") != _keys_sha(e.get("changed") or []):
            probs.append("%s's record does not show every frame proved" % exam)
    if probs:
        refuse("%s: %s" % (p, "; ".join(probs)))
    return d


def changed_of(rec, exam):
    return set(((rec.get("exams") or {}).get(exam) or {}).get("changed") or [])


def exam_keys(exam):
    return [r["key"] for r in C.read_manifest(C.manifest_path(exam))]


# ------------------------------------------------------------------ what D1's job calls
def identity(exam, arm, kw, rec):
    """The identity reproduction a D1 pass on exam rests on (V3F.3.2 item 3; v3fam_systems.require_or_make_identity),
    taken if missing under the fix, both locked-scorer calls observed, and noted once (note_path) by its sha256. When
    the record lists images the fix changes on exam, an identity record in place must be the one its note names."""
    from . import v3fam_systems as SYS
    name = "identity_%s.json" % exam
    p = F.equiv_dir() / name
    changed = changed_of(rec, exam)
    if not p.exists():
        with loader_fix(exam) as obs:
            SYS.require_or_make_identity(exam, arm, kw)
        got = obs.check(exam_keys(exam), changed, 2, "%s's identity reproduction" % exam)
        F.write_once(note_path(exam), dict(got, format=FMT_IDENTITY, exam=exam, identity=name,
                                           identity_sha256=F._sha(p), written_utc=F._utc()))
    elif changed:
        n = F._read_json(note_path(exam))
        if not isinstance(n, dict) or n.get("format") != FMT_IDENTITY or n.get("identity_sha256") != F._sha(p):
            refuse("%s (sha256 %s) was not taken under the loader fix, and the fix changes %d image(s) of %s: it is "
                   "moved aside (inc2.v3fam_supersede) and retaken" % (p, str(F._sha(p))[:12], len(changed), exam))
    return SYS.require_or_make_identity(exam, arm, kw)


def check_kept(doc, exam, rec, what):
    """A pass kept from an earlier attempt on an exam whose record lists changed images must carry the fix's note."""
    if exam in EXAMS and changed_of(rec, exam):
        if ((doc or {}).get("loader_shape_fix") or {}).get("deviation") != DEVIATION_6:
            refuse("%s on %s was committed without the loader fix, which changes %d of its images" % (
                what, exam, len(changed_of(rec, exam))))


def summary(rec):
    """What D1's score record carries of this deviation."""
    return {"deviation": DEVIATION_6, "record": "v3fam/%s/exif/%s" % (F.NAME, RECORD),
            "record_sha256": F._sha(record_path()),
            "changed": {e: len(changed_of(rec, e)) for e in EXAMS},
            "identity_notes": {e: F._sha(note_path(e)) for e in EXAMS if note_path(e).exists()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F deviation 6: the loader shapes pre-flight on ood22 and ood23.")
    ap.add_argument("command", choices=("status", "preflight"))
    a = ap.parse_args(argv)
    try:
        if a.command == "status":
            d = F._read_json(record_path())
            print(json.dumps({k: v for k, v in (d or {}).items() if k != "exams"} if isinstance(d, dict) else None,
                             indent=1, sort_keys=True, default=str))
        else:
            d = preflight()
            print(json.dumps({e: {k: v for k, v in x.items() if k not in ("rows", "changed")}
                              for e, x in d["exams"].items()}, indent=1, sort_keys=True))
    except F.V3famRefused as e:
        print("[inc2.v3fam_loader] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
