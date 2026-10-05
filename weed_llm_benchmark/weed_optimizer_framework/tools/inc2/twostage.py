"""E3: two-stage species detection, a one-class box detector and then a
BioCLIP-2 crop species classifier (docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-05): E3, two-stage species detection (pre-registered)").

    python -m weed_optimizer_framework.tools.inc2.twostage fit-classifier
    python -m weed_optimizer_framework.tools.inc2.twostage dev-gt
    python -m weed_optimizer_framework.tools.inc2.twostage equivalence
    python -m weed_optimizer_framework.tools.inc2.twostage score-arm --arm M|A|B [--seeds 0,1,2]     (L23F)
    python -m weed_optimizer_framework.tools.inc2.twostage verdict [--out-dir D]                      (L23G)
    python -m weed_optimizer_framework.tools.inc2.twostage test-read --arm X                          (a person)
    python -m weed_optimizer_framework.tools.inc2.twostage score-test --arm X                         (its job)
    python -m weed_optimizer_framework.tools.inc2.twostage test-report --arm X [--out-dir D]
    python -m weed_optimizer_framework.tools.inc2.twostage restore-classifier --from DIR
    python -m weed_optimizer_framework.tools.inc2.twostage sensitivity                                (optional)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

The arms (stage 1, no retraining): seed s of arm X reads the final EMA
weights of an existing base run, <exp>/runs/base__s<s>/weights/final.pt
(ARMS: M b_v2_m640, A e1_a_m640, B e1_b_m640), under the locked scorer's
own inference settings (640 px, its conf, IoU and max_det). E3-A and E3-B
(one-class detectors) take the locked scorer's own NMS; E3-M takes
class-agnostic NMS (NMS), which collapses b_v2_m640's twelve classes to
one box set. Rows at identical coordinates are then reduced to their most
confident one (inc/scorer.py one_per_box). A box set under the locked NMS
is the one its run's recorded protocol agnostic dev score was computed on,
so E3-A's and E3-B's agnostic AP must reproduce that score (STAGE1_TOL);
E3-M's agnostic-NMS box set has no recorded score to reproduce. E3-M's
boxes under the locked NMS are a reported reading (REPORTED_NMS).

Stage 2: each stage-1 box is mapped back to the original image (the
inverse of the validator's letterbox, per axis), cut as inc.audit cuts a
crop (inc/verify.py _cut_task, semisup_labeler._cut), embedded by BioCLIP-2
(verify.BioclipEmbedder), L2-normalised (verify._norm) and classified by
one multinomial logistic regression over the 13 INC classes, fitted once
on base_v2's ground-truth boxes (fit_classifier) and pinned
(capacity/e3_classifier_pin.json). Each box is emitted as its top-3
classes with score q x p(c | crop), within the locked scorer's own limits
(score above its conf, at most max_det rows per image).

How the predictions are scored: the locked scorer (inc.scorer.score, every
check unchanged) runs with a subclass of its own validator, through the
scorer sidecar's capturing subclass (inc2.scorer_sidecar), that replaces
each image's predictions after NMS and before Ultralytics' metric update
(update_metrics). Ultralytics' matching and ap_per_class and the scorer's
per-class, species and agnostic definitions are computed on them,
unchanged. The same path fed a run's own predictions (identity mode) must
reproduce b_v2_m640's recorded protocol dev scores (equivalence), or no E3
score is taken. Every pass checks that the ground-truth boxes survive the
geometry (GT_TOL) and that each image's EXIF orientation is absent, 0 or
1 (OK_ORIENTATIONS) and its size Ultralytics' ori_shape. A training image
of the classifier may also carry 3, 6 or 8 (FIT_ORIENTATIONS): it is cut,
as every crop is, in the EXIF-transposed frame, which is its labels'.

Layout (INC_DIR/twostage/e3_v1, written once each unless said):
  classifier/classifier.json, classifier.npz   the fit (weights, prior, the fold classifiers)
  classifier/train_crops.csv, train_emb.npz    the training boxes and their fp32 features
  classifier/dev_gt.json                       top-1 / top-3 on dev ground-truth crops (reported)
  equivalence.json                             the identity check, with the code it ran
  arms/<X>/s<k>/<exam>.json, .images.npz, .stage1.npz   E3 scores (dev decides; imageweeds reported)
  arms/M/s<k>/dev.locked_nms.*                 E3-M's boxes under the locked NMS (reported)
  arms/<X>/reported.json                       the reported passes' outcome (overwritten)
  test/<X>/e3_test_read.json, test/<X>/s<k>/test.*, test/reference/s<k>/test.identity.*
  sensitivity.json                             the fold classifiers on seed 0 (optional, reported)
and in INC_DIR/capacity: e3_classifier_pin.json, e3_score_<X>.json (L23F's
record, dev only, overwritten), e3_v1.json (the verdict, dev only),
e3_v1_report.{json,md} (people), e3_rescore.json (L23G's record),
e3_test_<X>.{json,md} (people).

Every E3 score file says production false and is stamped scorer_sha256
"E3-" + the locked scorer's sha256 ("TEST-E3-" in test mode), so no gate,
milestone or capacity decision can take it for a protocol score.

A refusal raises E3Refused; the CLI prints "[inc2.twostage] ERROR: ..." and
exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..inc import audit as AU
from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from .. import semisup_labeler as SL
from . import base3 as B3
from . import baseline as B
from . import common as C2
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import train as T

# ------------------------------------------------------------------ constants (pre-registered)
E3_NAME = "e3_v1"
ARMS = {"M": "b_v2_m640", "A": "e1_a_m640", "B": "e1_b_m640"}
ARM_ORDER = ("M", "A", "B")                 # the proposal order, and the tie order of the choice
NMS = {"M": "agnostic", "A": "locked", "B": "locked"}   # the deciding stage-1 NMS (E3-M: class-agnostic)
REPORTED_NMS = {"M": "locked"}              # E3-M's boxes under the locked scorer's own NMS: reported, never deciding
# the arms whose deciding box set is the one their run's recorded protocol agnostic dev score was computed on
STAGE1_COMPARED = tuple(a for a in ARM_ORDER if NMS[a] == "locked")
REFERENCE = "b_v2_m640"
SEEDS = (0, 1, 2)
EXAM = "dev"
REPORT_EXAMS = ("imageweeds",)
IMGSZ = 640
TOP_K = 3
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CV_FOLDS = 5
MAX_ITER = V.PROBE_MAX_ITER
CLASS_WEIGHT = None
CLF_SEED_TEXT = "inc2/e3/classifier"
SPECIES_SEED_TEXT = "inc2/e3/species_se"
ATTR_SEED_TEXT = "inc2/e3/attribution_se"
RESAMPLES = 1000
EQ_TOL = SC.MAX_SCORE_DIFF                  # the identity path against the recorded protocol score
STAGE1_TOL = SC.MAX_SCORE_DIFF              # a stage-1 box set's agnostic AP against the recorded one
STAGE1_EXACT = 1e-12                        # identity mode: the stage-1 helper against the scorer's own agnostic
GT_TOL = 1e-3                               # the ground-truth round trip through the geometry, normalised units
PROBA_TOL = 1e-6
TIE_TOL = 1e-12                             # the CV argmin and the choice: equal within this
REPORTED_SPECIES = B.NATIVE_TARGET_SPECIES
PROTOCOL_CHECK_MIN, PROTOCOL_CHECK_MEDIAN_COS = 100, 0.99
GEOM_TOL = AU.GEOM_TOL                      # the crop-protocol check's box match (inc.audit's)
ORIENTATION_TAG = 0x0112
# EXIF orientation tags read: none and 1 are no rotation, and so is the invalid 0 (PIL's exif_transpose and
# OpenCV, Ultralytics' reader, both leave it unrotated). A pass reads only these: its crops and Ultralytics' frame
# must agree, which the size check sees for 5-8 but not for 3. A training image of the classifier may also carry 3,
# 6 or 8: it is cut in the EXIF-transposed frame (verify._cut_task), the frame its labels are in (read on base_v2's
# tagged images, 2026-10-05) and the frame Ultralytics trained b_v2_m640 in; any other tag refuses.
OK_ORIENTATIONS = (None, 0, 1)
FIT_ORIENTATIONS = (None, 0, 1, 3, 6, 8)
STAMP = "E3-"
E3_STAMPS = ("manifest_sha256", "key_order_sha256", "n_images", "locked_scorer_sha256", "ultralytics_version")
SETTINGS = B.NATIVE_SETTINGS
TEST_STAMPS = ("manifest_sha256", "key_order_sha256")
EMBED_BATCH = V.BATCH
PROCS = V.PROCS

FMT_CLASSIFIER = "inc2-e3-classifier/1"
FMT_PIN = "inc2-e3-classifier-pin/1"
FMT_DEV_GT = "inc2-e3-dev-gt/1"
FMT_EQUIVALENCE = "inc2-e3-equivalence/1"
FMT_SCORE = "inc2-e3-score/1"
FMT_SCORE_RECORD = "inc2-e3-score-record/1"
FMT_VERDICT = "inc2-e3-verdict/1"
FMT_RESCORE = "inc2-e3-rescore/1"
FMT_TEST_READ = "inc2-e3-test-read/1"
FMT_TEST_REPORT = "inc2-e3-test-report/1"
FMT_SENSITIVITY = "inc2-e3-sensitivity/1"

DECIDED_BY = ("docs/CONTINUOUS_LOOP.md, Amendment (2026-10-05): E3, two-stage species detection (pre-registered)")
RULE = ("for each arm (M: b_v2_m640's boxes under class-agnostic NMS, A: E1-A's, B: E1-B's under the locked scorer's "
        "NMS; each reduced by one_per_box, named by one BioCLIP-2 logistic-regression classifier fitted once on "
        "base_v2's ground-truth boxes, each box emitted as its top 3 classes at q x p within the locked conf and "
        "max_det), on seeds 0, 1, 2, "
        "D = mean(arm's dev species_map50_95, scored by the locked scorer's own matching and AP) - mean(b_v2_m640's "
        "native dev files at 640); the arm qualifies when D > 2 x pooled sd (sqrt((sd_arm^2 + sd_ref^2) / 2), sample "
        "sd) AND D > the paired image-bootstrap SE of D (inc2.baseline native_bootstrap, 1,000 resamples of the dev "
        "images under stable_int('inc2/e3/species_se'), one draw for every run); the largest qualifying D is E3's "
        "choice, a tie goes to M, then A, then B; record only: nothing switches; the sealed test is read once per "
        "qualifying arm, after this verdict, by a person (inc2.twostage test-read); the chosen arm's is E3's headline")
ATTR_RULE = ("D_data = mean(E3-B) - mean(E3-A) on dev species_map50_95, its SE by the same construction under "
             "stable_int('inc2/e3/attribution_se'); E3's box gain is credited to base v3's data when D_data > 2 x "
             "pooled sd AND D_data > SE; record only: it changes neither qualification nor the choice")
DECISION_KEYS = ("status", "qualifying", "chosen", "testing_allowed")
ARM_DECISION_KEYS = ("status", "seeds", "dev", "reference_dev", "diff", "pooled_sd", "se_diff", "conditions",
                     "qualifies")
ATTR_DECISION_KEYS = ("status", "diff", "pooled_sd", "se_diff", "conditions", "credited_to_data")
CLASSIFIER_FILES = ("classifier.json", "classifier.npz", "train_crops.csv", "train_emb.npz")
TRAIN_CROP_FIELDS = ("crop_id", "key", "image", "box", "label", "session", "cx", "cy", "w", "h")
CODE_FILES = {"twostage": __file__, "scorer": S.__file__, "scorer_sidecar": SC.__file__,
              "scorer_native": SN.__file__, "verify": V.__file__, "semisup_labeler": SL.__file__}

_TEST_TOKENS = []                           # score_test's tokens: the only callers that may score test


class E3Refused(RuntimeError):
    """What was asked is not E3's pre-registered measurement."""


def log(msg):
    print("[inc2.twostage] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths and files
def root():
    return C.INC_DIR / "twostage" / E3_NAME


def capacity_dir(out_dir=None):
    return Path(out_dir) if out_dir else C.INC_DIR / "capacity"


def classifier_dir():
    return root() / "classifier"


def pin_path():
    return C.INC_DIR / "capacity" / "e3_classifier_pin.json"


def arm_dir(arm, seed):
    return root() / "arms" / arm / ("s%d" % int(seed))


def score_record_path(arm, out_dir=None):
    return capacity_dir(out_dir) / ("e3_score_%s.json" % arm)


def verdict_path(out_dir=None):
    return capacity_dir(out_dir) / ("%s.json" % E3_NAME)


def rescore_path(out_dir=None):
    return capacity_dir(out_dir) / "e3_rescore.json"


def test_dir(arm):
    return root() / "test" / arm


def reported_variant(arm):
    """The file variant of an arm's reported NMS reading (E3-M: locked_nms)."""
    return "%s_nms" % REPORTED_NMS[arm]


def file_names(exam, variant=""):
    """(json, images npz, stage-1 npz) names of an E3 score."""
    stem = "%s%s" % (exam, ".%s" % variant if variant else "")
    return "%s.json" % stem, "%s.images.npz" % stem, "%s.stage1.npz" % stem


def _utc():
    """UTC with microseconds: the fit-before-dev-read order is compared on these."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _sha(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path, obj):
    """Atomic overwrite (records that are rewritten: a score record, a pending verdict, reports)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        tmp.write_text(_dump(obj))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


def _write_once(path, data):
    """Write bytes (or a JSON object) to path only if it does not exist: a temp file hard-linked into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.%d.tmp" % (path.name, os.getpid(), time.monotonic_ns()))
    try:
        if isinstance(data, (bytes, bytearray)):
            tmp.write_bytes(bytes(data))
        else:
            tmp.write_text(_dump(data))
        try:
            os.link(str(tmp), str(path))
        except FileExistsError:
            raise E3Refused("%s exists: it is written once and never overwritten" % path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


def _npz_bytes(arrays):
    import numpy as np
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _code_shas():
    return {k: C.sha256_file(Path(v).resolve()) for k, v in sorted(CODE_FILES.items())}


def _versions():
    out = {}
    for mod in ("sklearn", "numpy", "torch", "open_clip", "ultralytics"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                    # noqa: BLE001 - a module that is not installed is recorded as such
            out[mod] = None
    return out


def _locked(scorer_sha):
    s = str(scorer_sha or "")
    return s[len(S.TEST_PREFIX):] if s.startswith(S.TEST_PREFIX) else s


def stamp(locked_sha, test):
    return "%s%s%s" % (S.TEST_PREFIX if test else "", STAMP, locked_sha)


# ------------------------------------------------------------------ geometry
def to_original(xyxy, ratio_pad, ori_shape):
    """Letterboxed xyxy -> original-image pixels: the inverse of the validator's
    letterbox per axis (Ultralytics' ratio_pad: ((gain_h, gain_w), (pad_x,
    pad_y))), clipped to the image (ori_shape (h, w)). float64 [n, 4]."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    try:
        (gh, gw), (px, py) = ratio_pad
        gh, gw, px, py = float(gh), float(gw), float(px), float(py)
        h0, w0 = float(ori_shape[0]), float(ori_shape[1])
    except (TypeError, ValueError) as e:
        raise RuntimeError("ratio_pad %r / ori_shape %r are not ((gain_h, gain_w), (pad_x, pad_y)) and (h, w): the "
                           "validator's letterbox record changed (%s); port inc2/twostage.py" % (ratio_pad, ori_shape, e))
    if gh <= 0 or gw <= 0:
        raise RuntimeError("a non-positive letterbox gain %r" % (ratio_pad,))
    out = np.empty_like(a)
    out[:, [0, 2]] = np.clip((a[:, [0, 2]] - px) / gw, 0.0, w0)
    out[:, [1, 3]] = np.clip((a[:, [1, 3]] - py) / gh, 0.0, h0)
    return out


def normalise(orig, w0, h0):
    """Original-pixel xyxy -> normalised (cx, cy, w, h), float64 [n, 4]."""
    import numpy as np
    a = np.asarray(orig, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(a)
    out[:, 0] = (a[:, 0] + a[:, 2]) / 2.0 / w0
    out[:, 1] = (a[:, 1] + a[:, 3]) / 2.0 / h0
    out[:, 2] = (a[:, 2] - a[:, 0]) / w0
    out[:, 3] = (a[:, 3] - a[:, 1]) / h0
    return out


def clip_label(cx, cy, w, h):
    """A label's normalised box clipped to the image, as the geometry clips a mapped box."""
    x1, y1 = min(max(cx - w / 2.0, 0.0), 1.0), min(max(cy - h / 2.0, 0.0), 1.0)
    x2, y2 = min(max(cx + w / 2.0, 0.0), 1.0), min(max(cy + h / 2.0, 0.0), 1.0)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def degenerate(cx, cy, w, h, w0, h0):
    """True when the box's square crop side rounds below 1 px (semisup_labeler._cut cannot cut it)."""
    _x0, _y0, side = SL._square_box(cx, cy, w, h, w0, h0)
    return int(round(side)) < 1


def small(w, h, w0, h0):
    return w * w0 < SL.MIN_BOX_PX or h * h0 < SL.MIN_BOX_PX


def header(path):
    """(EXIF orientation tag or None, (W, H) as the image is seen after EXIF transposition), from the header: tags
    5-8 swap W and H, as PIL's exif_transpose does; 0 and the others do not."""
    from PIL import Image
    with Image.open(path) as im:
        try:
            ori = im.getexif().get(ORIENTATION_TAG)
        except Exception:                    # noqa: BLE001 - an unreadable EXIF block reads as no tag
            ori = None
        w, h = im.size
    return ori, ((h, w) if ori in (5, 6, 7, 8) else (w, h))


def _e3_cut(task):
    """Worker: (image, crop ids, [crops] or None, error, (orientation, (W, H)) or None). The header is read
    first; the crops are inc/verify.py's _cut_task's."""
    image, rows = task
    try:
        hd = header(image)
    except Exception as e:                   # noqa: BLE001 - reported to the caller, which refuses
        return image, [r["crop_id"] for r in rows], None, "%s: %s" % (type(e).__name__, e), None
    if not rows:
        return image, [], [], None, hd
    img, ids, arrays, err = V._cut_task((image, rows))
    return img, ids, arrays, err, hd


# ------------------------------------------------------------------ the classifier's probabilities and emission
def proba(X, W, b, classes):
    """softmax(X W' + b) over the fitted classes, as [n, 13] (a class the fit never saw: 0)."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    Z = X @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
    Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    Pk = E / E.sum(axis=1, keepdims=True)
    P = np.zeros((len(X), C.NC), dtype=np.float64)
    P[:, np.asarray(classes, dtype=np.int64)] = Pk
    return P


def emit(q, P, k, conf_min=S.CONF, max_det=300):
    """The rows a set of boxes is emitted as: each box's top k classes by P
    (equal p: the lower class id first) at score float32(q x p); rows whose
    score is not above conf_min dropped; at most max_det rows kept, by score
    (ties: the lower box index, then the lower class id). Returns (box index,
    class, score float32, stats), in emission order (box, then rank)."""
    import numpy as np
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = len(q)
    st = {"rows_candidate": n * k, "dropped_conf": 0, "dropped_cap": 0, "capped": 0}
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), st
    P = np.asarray(P, dtype=np.float64).reshape(n, -1)
    order = np.argsort(-P, axis=1, kind="stable")[:, :k]
    box = np.repeat(np.arange(n, dtype=np.int64), order.shape[1])
    cls = order.reshape(-1).astype(np.int64)
    score = (q[box] * P[box, cls]).astype(np.float32)
    keep = np.flatnonzero(score > np.float32(conf_min))
    st["dropped_conf"] = int(len(score) - len(keep))
    if len(keep) > max_det:
        o = np.lexsort((cls[keep], box[keep], -score[keep].astype(np.float64)))
        sel = np.sort(keep[o[:max_det]])
        st["dropped_cap"] = int(len(keep) - max_det)
        st["capped"] = 1
        keep = sel
    return box[keep], cls[keep], score[keep], st


def onehot(cls):
    import numpy as np
    c = np.rint(np.asarray(cls, dtype=np.float64)).astype(np.int64).reshape(-1)
    P = np.zeros((len(c), C.NC), dtype=np.float64)
    P[np.arange(len(c)), c] = 1.0
    return P


def _rows_pred(boxes, extra, box_idx, cls_ids, score32, cls_dtype):
    """The emitted rows as the per-image prediction dict Ultralytics' update_metrics reads: bboxes the boxes'
    own rows (same dtype), conf float32 (never fp16), cls in the NMS output's dtype, extra the boxes' rows."""
    import numpy as np
    import torch
    dev = boxes.device
    rep = torch.as_tensor(np.asarray(box_idx, dtype=np.int64), dtype=torch.long, device=dev)
    out = {"bboxes": boxes[rep],
           "conf": torch.as_tensor(np.asarray(score32, dtype=np.float32), dtype=torch.float32, device=dev),
           "cls": torch.as_tensor(np.asarray(cls_ids, dtype=np.float64), device=dev).to(cls_dtype)}
    out["extra"] = extra[rep] if extra is not None else boxes.new_zeros((len(rep), 0))
    return out


# ------------------------------------------------------------------ the validator
def e3_validator_class(cfg):
    """(E3 validator class, the scorer's own validator it is built on). cfg:
    mode identity | two_stage | gt, nms locked | agnostic, W, b, classes,
    prior (two_stage, gt), embedder, workers (verify._Workers, entered),
    top_k, max_det, keep_crops (tests). The class is built on the scorer's
    validator as it is before anything is installed (exactly one E3 layer)."""
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils.metrics import box_iou

    base = S.validator_class()
    if any(k.__dict__.get("_e3_layer") for k in base.__mro__):
        raise RuntimeError("the scorer's validator is already an E3 validator: an earlier pass did not restore it")
    missing = [n for n in ("postprocess", "update_metrics", "_prepare_batch", "_process_batch", "match_predictions",
                           "init_metrics") if not callable(getattr(base, n, None))]
    if missing or "agnostic_nms" not in DEFAULT_CFG_DICT:
        raise RuntimeError("ultralytics %s: DetectionValidator has no %s%s, which inc2/twostage.py hooks; port it "
                           "to this version" % (ultralytics.__version__, missing,
                                                "" if "agnostic_nms" in DEFAULT_CFG_DICT else " (or no agnostic_nms)"))
    mode, nms = cfg.get("mode"), cfg.get("nms", "locked")
    if mode not in ("identity", "two_stage", "gt") or nms not in ("locked", "agnostic"):
        raise RuntimeError("E3 validator mode %r nms %r" % (mode, nms))
    sc = SC.sidecar_validator_class()
    if sc.__mro__[1] is not base:
        raise RuntimeError("the scorer sidecar's validator is not built on the scorer's own")

    class E3Validator(sc):
        _e3_layer = True

        def inc_reset(self):
            super().inc_reset()
            self.e3_s1_tp, self.e3_s1_q, self.e3_s1_ngt = [], [], 0
            self.e3_images = {}
            self.e3_stats = collections.Counter()
            self.e3_orientations = collections.Counter()
            self.e3_gt = []
            self.e3_crops = {} if cfg.get("keep_crops") else None
            self.e3_gt_maxdiff = 0.0
            self.e3_embed_s = 0.0
            self.e3_cut_s = 0.0

        def postprocess(self, preds):
            if nms != "agnostic":
                return super().postprocess(preds)
            old = self.args.agnostic_nms
            self.args.agnostic_nms = True
            try:
                return super().postprocess(preds)
            finally:
                self.args.agnostic_nms = old

        def update_metrics(self, preds, batch):
            pbs = [self._prepare_batch(si, batch) for si in range(len(preds))]
            return super().update_metrics(self.e3_batch(preds, pbs), batch)

        # -------------------------------------------------------- one validator batch
        def e3_batch(self, preds, pbs):
            items = []
            for pred, pb in zip(preds, pbs):
                for d, keys in ((pred, ("bboxes", "conf", "cls")),
                                (pb, ("bboxes", "cls", "im_file", "ori_shape", "ratio_pad"))):
                    if not all(k in d for k in keys):
                        raise RuntimeError("ultralytics %s: per-image dict has keys %s, inc2/twostage.py needs %s"
                                           % (ultralytics.__version__, sorted(d), keys))
                items.append(self.e3_stage1(pred, pb))
            if mode in ("two_stage", "gt"):
                self.e3_classify(items)
            out = []
            for it in items:
                pred = it["pred"]
                if mode == "two_stage":
                    q, P, k, boxes = it["q"], it["P"], int(cfg.get("top_k") or TOP_K), it["s1_boxes"]
                    extra = pred.get("extra")
                    extra = extra[it["idx"]] if extra is not None else None
                else:
                    q, P, k, boxes = (pred["conf"].float().cpu().numpy().reshape(-1), onehot(
                        pred["cls"].float().cpu().numpy()), 1, pred["bboxes"])
                    extra = pred.get("extra")
                bi, ci, sc_, st = emit(q, P, k, conf_min=S.CONF, max_det=int(cfg.get("max_det") or self.args.max_det))
                for name, v in st.items():
                    self.e3_stats[name] += int(v)
                self.e3_stats["rows"] += int(len(bi))
                if mode == "two_stage":
                    it["rec"]["emitted"] = int(len(bi))
                    self.e3_images[it["key"]] = it["rec"]
                out.append(_rows_pred(boxes, extra, bi, ci, sc_, pred["cls"].dtype))
            return out

        def e3_stage1(self, pred, pb):
            """The stage-1 box set (one_per_box on the NMS output) and its class-collapsed matches, by the
            scorer's own per-image hook's lines."""
            boxes, conf = pred["bboxes"], pred["conf"]
            conf_np = conf.float().cpu().numpy().reshape(-1)
            keep = S.one_per_box(boxes.float().cpu().numpy(), conf_np)
            idx = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
            s1 = boxes[idx]
            q = conf[idx].float().cpu().numpy().reshape(-1)
            gc = pb["cls"]
            if len(gc) and len(keep):
                iou = box_iou(pb["bboxes"], s1)
                zeros = torch.zeros(len(keep), device=boxes.device)
                tp = self.match_predictions(zeros, torch.zeros_like(gc), iou).cpu().numpy()
            else:
                tp = np.zeros((len(keep), self.niou), dtype=bool)
            self.e3_s1_tp.append(tp)
            self.e3_s1_q.append(q)
            self.e3_s1_ngt += int(len(gc))
            self.e3_stats["stage1_boxes"] += int(len(keep))
            return {"key": Path(pb["im_file"]).stem, "pred": pred, "pb": pb, "idx": idx, "s1_boxes": s1, "q": q}

        def e3_gt_check(self, pb, key, w0, h0):
            """The ground-truth boxes through the geometry against the exam label's (the view's verified bytes):
            each within GT_TOL of a label box of its class, one to one. Returns their normalised boxes."""
            gc = np.rint(pb["cls"].float().cpu().numpy().reshape(-1)).astype(np.int64)
            lab = Path(pb["im_file"]).parent.parent / "labels" / ("%s.txt" % key)
            try:
                rows = sorted(set(T.read_label_strict(lab)))
            except (OSError, ValueError) as e:
                raise E3Refused("cannot read the exam label of %s for the geometry check: %s" % (key, e))
            if len(rows) != len(gc):
                raise E3Refused("%s: %d ground-truth boxes in the batch, %d distinct label rows" % (key, len(gc),
                                                                                                  len(rows)))
            if not len(gc):
                return np.zeros((0, 4)), gc
            norm = normalise(to_original(pb["bboxes"].float().cpu().numpy(), pb["ratio_pad"], pb["ori_shape"]),
                             w0, h0)
            lab_c = np.array([r[0] for r in rows], dtype=np.int64)
            lab_n = np.array([clip_label(*r[1:]) for r in rows], dtype=np.float64)
            used = np.zeros(len(rows), dtype=bool)
            for i in range(len(gc)):
                cand = np.flatnonzero(~used & (lab_c == gc[i]))
                if not len(cand):
                    raise E3Refused("%s: ground-truth box %d (class %d) has no label box of its class" % (key, i, gc[i]))
                d = np.abs(lab_n[cand] - norm[i]).max(axis=1)
                j = int(np.argmin(d))
                if not d[j] <= GT_TOL:
                    raise E3Refused("%s: ground-truth box %d comes back through the geometry %.6f from its label "
                                    "(> %g): the map to the original image is wrong" % (key, i, float(d[j]), GT_TOL))
                used[cand[j]] = True
                self.e3_gt_maxdiff = max(self.e3_gt_maxdiff, float(d[j]))
            return norm, gc

        def e3_classify(self, items):
            """Cut, embed and classify every box of the batch (two_stage: the stage-1 boxes; gt: the ground-truth
            boxes of at least MIN_BOX_PX)."""
            tasks, slots = [], []
            for it in items:
                pb = it["pb"]
                h0, w0 = int(pb["ori_shape"][0]), int(pb["ori_shape"][1])
                it["size"] = (w0, h0)
                gt_norm, gt_cls = self.e3_gt_check(pb, it["key"], w0, h0)
                rows = []
                if mode == "two_stage":
                    n = len(it["q"])
                    nb = normalise(to_original(it["s1_boxes"].float().cpu().numpy(), pb["ratio_pad"], pb["ori_shape"]),
                                   w0, h0) if n else np.zeros((0, 4))
                    deg = np.array([degenerate(*nb[j], w0, h0) for j in range(n)], dtype=bool)
                    sm = np.array([small(nb[j, 2], nb[j, 3], w0, h0) for j in range(n)], dtype=bool)
                    rows = [{"crop_id": j, "cx": float(nb[j, 0]), "cy": float(nb[j, 1]), "w": float(nb[j, 2]),
                             "h": float(nb[j, 3])} for j in range(n) if not deg[j]]
                    it.update(norm=nb, deg=deg, small=sm)
                    self.e3_stats["degenerate"] += int(deg.sum())
                    self.e3_stats["small"] += int(sm.sum())
                else:
                    keep = [j for j in range(len(gt_cls)) if not small(gt_norm[j, 2], gt_norm[j, 3], w0, h0)]
                    self.e3_stats["gt_small_left_out"] += int(len(gt_cls) - len(keep))
                    rows = [{"crop_id": j, "cx": float(gt_norm[j, 0]), "cy": float(gt_norm[j, 1]),
                             "w": float(gt_norm[j, 2]), "h": float(gt_norm[j, 3])} for j in keep]
                    it.update(gt_cls=gt_cls, gt_keep=keep, gt_norm=gt_norm)
                tasks.append((str(pb["im_file"]), rows))
                slots.append(it)
            from PIL import Image
            by_item = collections.defaultdict(dict)
            buf, owner = [], []

            def flush():
                # one embedder batch: features, L2-normalised, classified; memory stays one batch of crops
                t1 = time.time()
                F = V._embed_safely(cfg["embedder"], buf, self.e3_stats)
                if not np.isfinite(F).all():
                    raise E3Refused("the embedder returned a non-finite feature for %d crop(s)"
                                    % int((~np.isfinite(F).all(axis=1)).sum()))
                X = V._norm(F)
                P = proba(X, cfg["W"], cfg["b"], cfg["classes"])
                for (it_, c_), p_, x_ in zip(owner, P, X):
                    by_item[id(it_)][int(c_)] = (p_, x_)
                self.e3_stats["crops"] += len(buf)
                self.e3_embed_s += time.time() - t1
                del buf[:], owner[:]
            t0 = time.time()
            e0 = self.e3_embed_s
            for it, (image, ids, arrays, err, hd) in zip(slots, cfg["workers"].imap(_e3_cut, tasks, chunksize=1)):
                if hd is None or err is not None:
                    raise E3Refused("cannot cut the crops of %s: %s (every exam image was hashed: a defect)"
                                    % (image, err))
                ori, (w, h) = hd
                self.e3_orientations[str(ori)] += 1
                if ori not in OK_ORIENTATIONS:
                    raise E3Refused("%s has EXIF orientation %r: a pass reads only an absent tag, 0 or 1 (the crop "
                                    "and Ultralytics' frame could differ)" % (image, ori))
                if (w, h) != it["size"]:
                    raise E3Refused("%s is %dx%d after EXIF transposition, Ultralytics read it as %dx%d (ori_shape)"
                                    % (image, w, h, it["size"][0], it["size"][1]))
                for c, a in zip(ids, arrays):
                    buf.append(Image.fromarray(a))
                    owner.append((it, c))
                    if self.e3_crops is not None:
                        self.e3_crops[(it["key"], int(c))] = a
                    if len(buf) >= EMBED_BATCH:
                        flush()
            if buf:
                flush()
            self.e3_cut_s += (time.time() - t0) - (self.e3_embed_s - e0)
            prior = np.asarray(cfg["prior"], dtype=np.float64)
            dim = int(cfg["embedder"].dim)
            for it in slots:
                got_ = by_item.get(id(it), {})
                if mode == "two_stage":
                    n = len(it["q"])
                    Pi = np.tile(prior, (n, 1)) if n else np.zeros((0, C.NC))
                    Xi = np.zeros((n, dim), dtype=np.float32)
                    for j, (p, x) in got_.items():
                        Pi[j], Xi[j] = p, x
                    it["P"] = Pi
                    top = np.argsort(-Pi, axis=1, kind="stable")[:, :TOP_K] if n else np.zeros((0, TOP_K), np.int64)
                    it["rec"] = {"xyxy": to_original(it["s1_boxes"].float().cpu().numpy(), it["pb"]["ratio_pad"],
                                                     it["pb"]["ori_shape"]) if n else np.zeros((0, 4)),
                                 "q": np.asarray(it["q"], dtype=np.float64), "top_cls": top.astype(np.int16),
                                 "top_p": np.take_along_axis(Pi, top, axis=1).astype(np.float32) if n
                                 else np.zeros((0, TOP_K), np.float32),
                                 "small": it["small"], "degenerate": it["deg"], "feat": Xi.astype(np.float16)}
                else:
                    for j in it["gt_keep"]:
                        p, _x = got_[int(j)]
                        order = np.argsort(-p, kind="stable")
                        self.e3_gt.append((int(it["gt_cls"][j]), int(order[0]), [int(x) for x in order[:3]]))

    layers = [k for k in E3Validator.__mro__ if k.__dict__.get("_e3_layer")]
    if len(layers) != 1 or E3Validator.__mro__.count(base) != 1:
        raise RuntimeError("the E3 validator holds %d E3 layers over the scorer's validator" % len(layers))
    return E3Validator, base


def _stage1_arrays(v, keys):
    """The per-box stage-1 record of a two-stage pass, in exam key order (stage1.npz)."""
    import numpy as np
    parts = collections.defaultdict(list)
    for i, k in enumerate(keys):
        r = v.e3_images.get(k)
        if r is None:
            raise E3Refused("the two-stage pass recorded no stage-1 boxes for exam image %s" % k)
        n = len(r["q"])
        parts["img"].append(np.full(n, i, dtype=np.int64))
        for f in ("xyxy", "q", "top_cls", "top_p", "small", "degenerate", "feat"):
            parts[f].append(r[f])
    out = {"keys": np.asarray(list(keys))}
    for f, v_ in parts.items():
        out[f] = np.concatenate(v_, 0)
    return out


def _score_pass(weights, exam, cfg, lock_check=True, batch=S.BATCH, device=None, test_token=None):
    """One locked-scorer pass of `weights` on `exam` with the E3 validator
    installed (module docstring). Returns {result, arrays (exam key order),
    stage1 (two_stage), stage1_agnostic (AP50-95, AP50), stats, gt, crops}."""
    import numpy as np
    if exam not in ("dev", "imageweeds", "test"):
        raise E3Refused("exam %r: E3 reads dev (deciding), imageweeds (reported) and, after the verdict, test" % exam)
    if exam == "test" and not any(test_token is t for t in _TEST_TOKENS):
        raise E3Refused("test is read only by score-test, after the verdict and a person's test-read")
    if cfg.get("mode") in ("two_stage", "gt") and cfg.get("embedder") is None:
        raise E3Refused("a two-stage pass needs the embedder")
    cls, base = e3_validator_class(cfg)
    prev = S._VALIDATOR
    del SC._CAPTURE[:]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="inc2_e3_") as tmp:
        S._VALIDATOR = cls
        try:
            res = S.score(weights, exam, Path(tmp) / "score.json", lock_check=lock_check, imgsz=IMGSZ, batch=batch,
                          device=device)
        except S.ScorerRefused as e:
            raise E3Refused("the locked scorer refuses: %s" % e)
        finally:
            S._VALIDATOR = prev if prev is not None else base
    caps = [x for x in SC._CAPTURE if isinstance(x, cls)]
    del SC._CAPTURE[:]
    if not caps:
        raise E3Refused("the scorer ran no E3 validator")
    v = caps[-1]
    res = {k: val for k, val in res.items() if k != "out"}
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path(exam)))
    if C.sha256_text("\n".join(keys)) != res["key_order_sha256"]:
        raise E3Refused("the exam's key order does not hash to the score's key_order_sha256")
    try:
        recompute, restricted = SC.check_capture(res["per_class"], v.sc_images)
        arrays = SC.flatten(v.sc_images, keys)
    except SC.SidecarError as e:
        raise E3Refused("the captured per-image arrays do not reproduce the score: %s" % e)
    if v.e3_s1_ngt != int(res["n_boxes"]):
        raise E3Refused("the stage-1 helper counted %d GT boxes, the scorer %d" % (v.e3_s1_ngt, res["n_boxes"]))
    tp = np.concatenate(v.e3_s1_tp, 0) if v.e3_s1_tp else np.zeros((0, SC.NIOU), dtype=bool)
    q = np.concatenate(v.e3_s1_q, 0) if v.e3_s1_q else np.zeros(0)
    s1 = S.collapsed_ap(tp, q, v.e3_s1_ngt)
    stats = {k: int(val) for k, val in sorted(v.e3_stats.items())}
    stats.update(orientations=dict(sorted(v.e3_orientations.items())), gt_roundtrip_max_diff=v.e3_gt_maxdiff,
                 embed_seconds=round(v.e3_embed_s, 3), cut_seconds=round(v.e3_cut_s, 3),
                 full_recompute_max_abs_diff=recompute, species_restricted_max_abs_diff=restricted)
    out = {"result": res, "arrays": arrays, "stage1_agnostic": s1, "stats": stats, "gt": list(v.e3_gt),
           "crops": v.e3_crops, "seconds": round(time.time() - t0, 3), "validator": v}
    out["stage1"] = _stage1_arrays(v, keys) if cfg.get("mode") == "two_stage" else None
    return out


# ------------------------------------------------------------------ the source runs
def _source(arm, seed, testing_ok=False):
    """What arm-seed reads (module docstring): the base run's weights (a
    regular file hashing as its run.json says), its final run carrying the
    same weights, the recorded protocol dev score of that final run, and the
    experiment's research-only flag (missing counts as true)."""
    if arm not in ARMS:
        raise E3Refused("arm %r is not one of E3's %s" % (arm, list(ARM_ORDER)))
    exp, seed = ARMS[arm], int(seed)
    root_ = C.INC_DIR / exp
    defn = _read_json(root_ / "exp.json")
    if not isinstance(defn, dict):
        raise E3Refused("%s has no exp.json" % exp)
    if seed not in [int(s) for s in defn.get("seeds") or []]:
        raise E3Refused("%s has no seed %d" % (exp, seed))
    rid, frid = "base__s%d" % seed, "final__base__s%d" % seed
    rj = _read_json(root_ / "runs" / rid / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise E3Refused("%s/%s is not done" % (exp, rid))
    w = root_ / "runs" / rid / "weights" / "final.pt"
    if w.is_symlink() or not w.is_file():
        raise E3Refused("%s is missing or a symlink: a stage-1 detector is its base run's own file" % w)
    wsha = C.sha256_file(w)
    if wsha != rj.get("weights_sha256"):
        raise E3Refused("%s hashes to %s, its run.json records %s" % (w, wsha[:12], str(rj.get("weights_sha256"))[:12]))
    fj = _read_json(root_ / "runs" / frid / "run.json")
    if not isinstance(fj, dict) or fj.get("status") != "done" or fj.get("weights_sha256") != wsha:
        raise E3Refused("%s/%s is not done or does not carry %s's weights" % (exp, frid, rid))
    rp = root_ / "runs" / frid / "scores" / "dev.json"
    rec = _read_json(rp)
    if not isinstance(rec, dict) or rec.get("exam") != "dev":
        raise E3Refused("%s/%s has no recorded protocol dev score" % (exp, frid))
    if rec.get("production") is not True and not testing_ok:
        raise E3Refused("%s is not a production score" % rp)
    if rec.get("weights_sha256") != wsha:
        raise E3Refused("%s names weights %s, not the base run's" % (rp, str(rec.get("weights_sha256"))[:12]))
    ro = defn.get("research_only")
    flag = ro.get("flag") if isinstance(ro, dict) else ro
    return {"arm": arm, "exp": exp, "seed": seed, "run_id": rid, "final_run_id": frid, "weights": str(w.resolve()),
            "weights_sha256": wsha, "base_run_json_sha256": _sha(root_ / "runs" / rid / "run.json"),
            "final_run_json_sha256": _sha(root_ / "runs" / frid / "run.json"),
            "recorded": {"file": "%s/%s/scores/dev.json" % (exp, frid), "sha256": _sha(rp),
                         "species_map50_95": rec.get("species_map50_95"),
                         "agnostic_map50_95": rec.get("agnostic_map50_95"), "production": rec.get("production")},
            "recorded_doc": rec, "research_only": True if flag is None else bool(flag)}


def _testing_ok():
    return S.testing()


# ------------------------------------------------------------------ the classifier
class _Table(object):
    """verify._embed_images' crops interface over the training boxes (W and H come from the image)."""

    def __init__(self, rows):
        self.rows = rows

    def row(self, i):
        r = self.rows[i]
        return {"crop_id": int(i), "cx": r["cx"], "cy": r["cy"], "w": r["w"], "h": r["h"], "W": 0, "H": 0}


def _parse_verified(label_path, want_sha, tmpdir):
    """The boxes of a label file parsed once from bytes that hash to want_sha (inc2.train read_label_strict)."""
    with open(label_path, "rb") as fh:
        data = fh.read()
    if hashlib.sha256(data).hexdigest() != want_sha:
        raise E3Refused("%s does not hash to its manifest's label_sha256" % label_path)
    p = Path(tmpdir) / "label.txt"
    p.write_bytes(data)
    try:
        return T.read_label_strict(p)
    except ValueError as e:
        raise E3Refused("%s: %s" % (label_path, e))


def all_boxes(rows):
    """[(key, box index, label, cx, cy, w, h)] of every box of the rows' verified labels, and the sha256 of the
    sorted lines key\\tbox\\tlabel (train_boxes_sha256)."""
    out = []
    with tempfile.TemporaryDirectory(prefix="inc2_e3_lab_") as tmp:
        for r in rows:
            for b, box in enumerate(_parse_verified(r["label"], r["label_sha256"], tmp)):
                out.append((r["key"], b, int(box[0])) + tuple(float(x) for x in box[1:]))
    lines = sorted("%s\t%d\t%d" % (k, b, c) for k, b, c, *_ in out)
    return out, hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _headers(paths):
    """{path: (orientation, (W, H))} of every image, read from headers on threads."""
    def one(p):
        try:
            return p, header(p), None
        except Exception as e:               # noqa: BLE001 - reported, then refused
            return p, None, "%s: %s" % (type(e).__name__, e)
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(one, list(dict.fromkeys(paths))))
    bad = [(p, e) for p, h, e in got if h is None]
    if bad:
        raise E3Refused("%d image header(s) cannot be read: %s" % (len(bad), bad[:3]))
    return {p: h for p, h, _e in got}


def _exists_after_fit():
    """What exists only after the classifier was fitted and used (a fit is refused while any does)."""
    found = [p for p in (root() / "arms", classifier_dir() / "dev_gt.json", root() / "equivalence.json",
                         root() / "test", pin_path(), verdict_path(), rescore_path())
             if p.exists()]
    found += sorted((C.INC_DIR / "capacity").glob("e3_score_*.json")) if (C.INC_DIR / "capacity").is_dir() else []
    return [str(p) for p in found]


def _prior_test_hits(rows):
    prior, rec = B3.prior_test_lists()
    shas, paths = set(), set()
    for p in prior:
        for k in ("original_sha256", "sha256", "train_sha256"):
            if p.get(k):
                shas.add(str(p[k]))
        for k in ("image", "train_image", "original_image"):
            if p.get(k):
                paths.add(str(p[k]))
    hits = [r["key"] for r in rows if str(r.get("sha256")) in shas or str(r.get("image")) in paths]
    return hits, rec


def _reference_manifest_check(sha, testing):
    """The training manifest is the one b_v2_m640 trained on: its exp.json and (production: every) base run
    record that sha256. Returns the record."""
    defn = _read_json(C.INC_DIR / REFERENCE / "exp.json") or {}
    exp_sha = (defn.get("base") or {}).get("manifest_sha256")
    runs = {}
    for s in SEEDS:
        rj = _read_json(C.INC_DIR / REFERENCE / "runs" / ("base__s%d" % s) / "run.json") or {}
        runs["base__s%d" % s] = rj.get("train_manifest_sha256")
    probs = []
    if exp_sha != sha:
        probs.append("its exp.json records %s" % str(exp_sha)[:12])
    bad = {k: str(v)[:12] for k, v in runs.items() if v != sha and (v is not None or not testing)}
    if bad:
        probs.append("its base runs record %s" % bad)
    if probs:
        raise E3Refused("base_v2 (%s) is not the manifest %s trained on: %s" % (sha[:12], REFERENCE, "; ".join(probs)))
    return {"exp_manifest_sha256": exp_sha, "base_runs_train_manifest_sha256": runs}


def protocol_check(boxes_rows, X, testing, crops=None, emb=None):
    """The crop-protocol check (amendment): fresh normalised features of base_v2's cwd12 train boxes against
    Step 1's stored ones of the same boxes (crops.csv core rows, matched by key with the same image sha256, box
    index and geometry within GEOM_TOL). crops/emb are injected by tests; otherwise read and checked fresh."""
    import numpy as np
    rec = {"min_matched": PROTOCOL_CHECK_MIN, "min_median_cosine": PROTOCOL_CHECK_MEDIAN_COS, "geom_tol": GEOM_TOL}
    try:
        if crops is None:
            crops = V.Crops()
            rec["crops_info"] = {k: V.check_fresh(crops).get(k) for k in ("crops_sha256", "built_utc")}
            emb, info = V.load_embeddings(crops)
            rec["embeddings"] = info
        core_sha = {r["key"]: r.get("sha256") for r in C.read_manifest(C.manifest_path("train_core"))} \
            if C.manifest_path("train_core").is_file() else {}
    except (V.VerifyError, OSError, ValueError) as e:
        if testing:
            return dict(rec, compared=False, why="no current Step 1 crops or embeddings (%s)" % e)
        raise E3Refused("the crop-protocol check cannot read Step 1's crops and embeddings: %s" % e)
    where = {}
    for i in crops.where("core"):
        where.setdefault((crops.key[i], int(crops.box[i])), []).append(int(i))
    cos = []
    for j, r in enumerate(boxes_rows):
        ids = where.get((r["key"], int(r["box"])))
        if not ids or core_sha.get(r["key"]) != r.get("sha256"):
            continue
        for i in ids:
            if max(abs(crops.cx[i] - r["cx"]), abs(crops.cy[i] - r["cy"]), abs(crops.w[i] - r["w"]),
                   abs(crops.h[i] - r["h"])) <= GEOM_TOL:
                s = np.asarray(emb[i], dtype=np.float32)
                if np.isfinite(s).all():
                    s = V._norm(s[None, :])[0]
                    cos.append(float(np.dot(s, X[j])))
                break
    rec["matched"] = len(cos)
    if cos:
        c = np.asarray(cos)
        rec.update(min=float(c.min()), median=float(np.median(c)), share_ge_0_999=float((c >= 0.999).mean()))
    if len(cos) < PROTOCOL_CHECK_MIN:
        if testing:
            return dict(rec, compared=False, why="%d boxes matched (< %d)" % (len(cos), PROTOCOL_CHECK_MIN))
        raise E3Refused("the crop-protocol check matched %d boxes to Step 1's stored crops (< %d): it cannot vouch "
                        "for the crop protocol" % (len(cos), PROTOCOL_CHECK_MIN))
    rec["compared"] = True
    if rec["median"] < PROTOCOL_CHECK_MEDIAN_COS:
        if testing:
            return dict(rec, passed=False)
        raise E3Refused("the fresh crops' features differ from Step 1's stored ones (median cosine %.4f < %g): not "
                        "inc.audit's crop protocol" % (rec["median"], PROTOCOL_CHECK_MEDIAN_COS))
    return dict(rec, passed=True)


def _fit_lr(X, y, c):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=float(c), max_iter=MAX_ITER, class_weight=CLASS_WEIGHT,
                                 random_state=C.stable_int(CLF_SEED_TEXT)).fit(X, y)
    return clf, int(np.max(clf.n_iter_)) < MAX_ITER


def _clf_proba(clf, X):
    """predict_proba as [n, 13] (a class missing from the fit: 0)."""
    import numpy as np
    P = np.zeros((len(X), C.NC), dtype=np.float64)
    P[:, clf.classes_.astype(np.int64)] = clf.predict_proba(X)
    return P


def cross_validate(X, y, groups):
    """C by GroupKFold cross-validation (amendment): per C the pooled held-out multinomial log-loss, a C with a
    fold fit that does not converge left out; the smallest loss, ties to the smaller C. Returns (chosen C, record,
    {fold: model at the chosen C})."""
    import numpy as np
    from sklearn.metrics import log_loss
    folds = V._assign_folds(groups, CV_FOLDS, C.stable_int(CLF_SEED_TEXT), random_ok=False)
    nf = int(folds.max()) + 1 if len(folds) else 0
    counts = {}
    for f in range(nf):
        tr, te = folds != f, folds == f
        counts[f] = {"train": {C.CLASS_NAMES[k]: int((y[tr] == k).sum()) for k in range(C.NC)},
                     "held_out": {C.CLASS_NAMES[k]: int((y[te] == k).sum()) for k in range(C.NC)},
                     "groups": int(len(set(np.asarray(groups)[te].tolist())))}
    per_c, models = [], {}
    for c in C_GRID:
        P = np.zeros((len(y), C.NC), dtype=np.float64)
        fold_rec, ok, fm = [], True, {}
        for f in range(nf):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            try:
                clf, conv = _fit_lr(X[tr], y[tr], c)
            except ValueError as e:
                fold_rec.append({"fold": f, "error": str(e)[:200]})
                ok = False
                continue
            P[te] = _clf_proba(clf, X[te])
            ll = log_loss(y[te], P[te], labels=list(range(C.NC))) if len(te) else None
            fold_rec.append({"fold": f, "converged": conv, "n_iter": int(np.max(clf.n_iter_)),
                             "log_loss": _finite(ll), "classes": [int(k) for k in clf.classes_]})
            ok = ok and conv
            fm[f] = clf
        pooled = log_loss(y, P, labels=list(range(C.NC))) if ok else None
        per_c.append({"C": c, "eligible": ok, "pooled_log_loss": _finite(pooled), "folds": fold_rec})
        models[c] = fm
    elig = [r for r in per_c if r["eligible"] and r["pooled_log_loss"] is not None]
    if not elig:
        raise E3Refused("no C of %s had every fold fit converge: the classifier cannot be fixed by its rule"
                        % list(C_GRID))
    best = None
    for r in elig:                           # ascending C: a later C replaces only when strictly better
        if best is None or r["pooled_log_loss"] < best["pooled_log_loss"] - TIE_TOL:
            best = r
    rec = {"folds": nf, "grouped_by": "session (GroupKFold, verify._assign_folds)", "grid": list(C_GRID),
           "criterion": "pooled held-out multinomial log-loss", "per_c": per_c, "chosen_C": best["C"],
           "left_out": [r["C"] for r in per_c if not r["eligible"]], "fold_class_counts": counts,
           "fold_sizes": {f: int((folds == f).sum()) for f in range(nf)}}
    return best["C"], rec, models[best["C"]]


def fit_classifier(embedder=None, testing=None, workers=None):
    """Fit E3's classifier once (amendment, Stage 2) and pin it. Refuses
    while the pin, the classifier or anything made after it exists."""
    import numpy as np
    testing = _testing_ok() if testing is None else bool(testing)
    d = classifier_dir()
    after = _exists_after_fit()
    if (d / "classifier.json").exists() or after:
        raise E3Refused("a classifier was fitted already or E3 has read dev (%s): E3's classifier is fitted once, "
                        "before any E3 dev score exists, and never refitted; a moved-aside E3 directory is restored "
                        "with restore-classifier" % ", ".join(after or [str(d / "classifier.json")]))
    t0 = time.time()
    try:
        path = C2.v2_manifest_path("base_v2")
        msha = C2.verify_manifest_against_lock_v2("base_v2")
        lock = C2.read_lock_v2()
    except C2.Inc2Error as e:
        raise E3Refused("LOCK v2's base_v2: %s" % e)
    ref = _reference_manifest_check(msha, testing)
    try:
        rows, dhashes, summ = T.check_manifest(path)
        guard = T.guard_rows(rows, dhashes, production=not testing)
    except T.RunError as e:
        raise E3Refused("base_v2 refused at %s: %s" % (e.stage, e))
    if summ.get("train_manifest_sha256") != msha:
        raise E3Refused("base_v2 changed while it was checked")
    hits, prior_rec = _prior_test_hits(rows)
    if hits:
        raise E3Refused("%d base_v2 row(s) are in a test v1 list or its companions (e.g. %s): never trained"
                        % (len(hits), hits[:3]))
    boxes, train_boxes_sha = all_boxes(rows)
    hd = _headers([r["image"] for r in rows])
    ors = collections.Counter(str(h[0]) for h in hd.values())
    bad = [(p, h[0]) for p, h in hd.items() if h[0] not in FIT_ORIENTATIONS]
    if bad:
        raise E3Refused("%d base_v2 image(s) carry an EXIF orientation other than none, 0, 1, 3, 6 or 8 (e.g. %s)"
                        % (len(bad), bad[:3]))
    by_key = {r["key"]: r for r in rows}
    train, n_small = [], collections.Counter()
    ors_boxes = collections.Counter()
    for key, b, lab, cx, cy, w, h in boxes:
        r = by_key[key]
        W0, H0 = hd[r["image"]][1]
        if small(w, h, W0, H0):
            n_small[C.CLASS_NAMES[lab]] += 1
            continue
        ors_boxes[str(hd[r["image"]][0])] += 1
        train.append({"key": key, "image": r["image"], "box": b, "label": lab, "session": r.get("session") or "",
                      "cx": cx, "cy": cy, "w": w, "h": h, "sha256": r["sha256"]})
    if not train:
        raise E3Refused("base_v2 holds no box of at least %d px" % SL.MIN_BOX_PX)
    # crop ids 0..n-1 with each image's boxes contiguous (verify._embed_images cuts an image's crops together)
    train = [train[i] for _img, ids in _group(train) for i in ids]
    images = [(img, ids) for img, ids in _group(train)]
    if embedder is None:
        embedder = V.BioclipEmbedder()
    t1 = time.time()
    with _workers(workers, procs=1 if testing else PROCS) as wk:
        ids, F, stats = V._embed_images(images, _Table(train), embedder, workers=wk, batch=EMBED_BATCH)
    if list(ids) != list(range(len(train))) or not np.isfinite(F).all():
        raise E3Refused("%d training crop(s) could not be cut or embedded (%s): every image was hashed, so this is "
                        "a defect" % (int((~np.isfinite(F).all(axis=1)).sum()), dict(stats)))
    embed_s = time.time() - t1
    F = np.asarray(F, dtype=np.float32)
    X32 = V._norm(F)
    X = X32.astype(np.float64)
    y = np.array([t["label"] for t in train], dtype=np.int64)
    groups = np.array([t["session"] for t in train])
    pcheck = protocol_check(train, X32, testing)
    chosen, cv, fold_models = cross_validate(X, y, groups)
    clf, conv = _fit_lr(X, y, chosen)
    if not conv:
        raise E3Refused("the final fit at C=%g did not converge in %d iterations" % (chosen, MAX_ITER))
    W, b, classes = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64), clf.classes_.astype(np.int64)
    if W.shape[0] != len(classes):
        raise E3Refused("the fit's coefficients are not one row per class (%s for %d classes)" % (W.shape,
                                                                                                 len(classes)))
    Pn, Ps = proba(X, W, b, classes), _clf_proba(clf, X)
    pdiff = float(np.abs(Pn - Ps).max()) if len(X) else 0.0
    if not pdiff <= PROBA_TOL:
        raise E3Refused("softmax(X W' + b) differs from scikit-learn's predict_proba by %.3g (> %g)"
                        % (pdiff, PROBA_TOL))
    counts = np.bincount(y, minlength=C.NC).astype(np.float64)
    prior = counts / counts.sum()
    arrays = {"W": W, "b": b, "classes": classes, "prior": prior}
    for f, m in sorted(fold_models.items()):
        arrays["fold%d_W" % f] = m.coef_.astype(np.float64)
        arrays["fold%d_b" % f] = m.intercept_.astype(np.float64)
        arrays["fold%d_classes" % f] = m.classes_.astype(np.int64)
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(TRAIN_CROP_FIELDS)
    for i, t in enumerate(train):
        wr.writerow([i, t["key"], t["image"], t["box"], t["label"], t["session"], repr(t["cx"]), repr(t["cy"]),
                     repr(t["w"]), repr(t["h"])])
    crops_bytes = buf.getvalue().encode("utf-8")
    emb_bytes = _npz_bytes({"crop_id": np.arange(len(train), dtype=np.int64), "F": F})
    npz_bytes = _npz_bytes(arrays)
    d.mkdir(parents=True, exist_ok=True)
    # the record names what it was fitted on; the files are written once, the record last, then the pin
    after = _exists_after_fit()
    if after:
        raise E3Refused("%s appeared while the classifier was fitted" % after)
    sh = {"train_crops_sha256": _write_once(d / "train_crops.csv", crops_bytes),
          "train_emb_sha256": _write_once(d / "train_emb.npz", emb_bytes),
          "npz_sha256": _write_once(d / "classifier.npz", npz_bytes)}
    top1 = float((np.argmax(Pn, axis=1) == y).mean())
    rec = {"format": FMT_CLASSIFIER, "name": E3_NAME, "pre_registered": DECIDED_BY,
           "manifest": {"split": "base_v2", "path": str(path), "sha256": msha},
           "lock_v2_sha256": _sha(T.v2_lock_path()), "index_sha256": lock.get("nevertrain_sha256"),
           "reference_manifest": ref,
           "guard": {k: guard.get(k) for k in ("lock_sha256", "index_sha256", "checked", "refused",
                                               "crosscheck_hits", "reasons")},
           "test_v1": {"hits": len(hits), "files": prior_rec.get("files"), "rows": prior_rec.get("rows")},
           "images": len(rows), "orientations": dict(sorted(ors.items())),
           "orientation_train_boxes": dict(sorted(ors_boxes.items())),
           "orientation_rule": "tags 3, 6 and 8 read through exif_transpose (verify._cut_task: the labels' frame); "
                               "0 and none as no rotation; any other tag refuses",
           "boxes": len(boxes), "train_boxes": len(train), "train_boxes_sha256": train_boxes_sha,
           "skipped_small": {"total": int(sum(n_small.values())), "per_class": dict(sorted(n_small.items()))},
           "per_class": {C.CLASS_NAMES[k_]: int(counts[k_]) for k_ in range(C.NC)},
           "sessions": int(len(set(groups.tolist()))),
           "files": dict(sh, train_crops="train_crops.csv", train_emb="train_emb.npz", npz="classifier.npz"),
           "embedder": {"name": getattr(embedder, "name", V.EMBEDDER_NAME), "dim": int(F.shape[1]),
                        "embed_seconds": round(embed_s, 3), "stats": {k_: int(v) for k_, v in dict(stats).items()}},
           "crop_protocol": {"crop_px": SL.CROP_PX, "margin": SL.MARGIN, "min_box_px": SL.MIN_BOX_PX,
                             "pad_grey": [124, 124, 124], "resize": "bicubic", "exif_transpose": True,
                             "semisup_labeler_sha256": C.sha256_file(Path(SL.__file__).resolve()),
                             "verify_sha256": C.sha256_file(Path(V.__file__).resolve())},
           "crop_protocol_check": pcheck,
           "cv": cv,
           "fit": {"C": chosen, "solver": "lbfgs", "max_iter": MAX_ITER, "class_weight": CLASS_WEIGHT,
                   "random_state": C.stable_int(CLF_SEED_TEXT), "seed_text": CLF_SEED_TEXT,
                   "n_iter": int(np.max(clf.n_iter_)), "converged": True, "classes": [int(c) for c in classes],
                   "proba_max_abs_diff": pdiff, "proba_tol": PROBA_TOL, "train_top1": top1,
                   "features": "L2-normalised fp32 BioCLIP-2 features (verify._norm), fitted in float64"},
           "prior": [float(x) for x in prior],
           "versions": _versions(),
           "code": {"twostage": C.sha256_file(Path(__file__).resolve()),
                    "verify": C.sha256_file(Path(V.__file__).resolve()),
                    "semisup_labeler": C.sha256_file(Path(SL.__file__).resolve()),
                    "inc2_train": C.sha256_file(Path(T.__file__).resolve()),
                    "inc2_guard": _sha(Path(T.__file__).resolve().with_name("guard.py")),
                    "inc2_base3": C.sha256_file(Path(B3.__file__).resolve())},
           "research_only": B.research_only_record(rows),
           "testing": testing, "seconds": round(time.time() - t0, 3), "written_utc": _utc()}
    jsha = _write_once(d / "classifier.json", rec)
    _write_once(pin_path(), {"format": FMT_PIN, "json_sha256": jsha, "npz_sha256": sh["npz_sha256"],
                             "train_crops_sha256": sh["train_crops_sha256"], "train_emb_sha256": sh["train_emb_sha256"],
                             "train_boxes_sha256": train_boxes_sha, "written_utc": rec["written_utc"],
                             "root": str(root()), "note": "E3's classifier is fitted once; this pin is never "
                                                          "rewritten; a refit is refused while it exists"})
    log("classifier: %d training boxes (%d small left out), C=%g by CV, train top-1 %.4f, %.0fs"
        % (len(train), sum(n_small.values()), chosen, top1, time.time() - t0))
    return rec


def _group(train):
    """[(image, [indices])] of train, in order (each image's boxes contiguous)."""
    out, pos = [], {}
    for i, t in enumerate(train):
        j = pos.get(t["image"])
        if j is None:
            pos[t["image"]] = len(out)
            out.append((t["image"], [i]))
        else:
            out[j][1].append(i)
    return out


def read_pin():
    p = _read_json(pin_path())
    if not isinstance(p, dict) or p.get("format") != FMT_PIN:
        raise E3Refused("%s is missing: E3's classifier has not been fitted (or its pin was removed)" % pin_path())
    return p


def load_classifier(fold=None):
    """(W, b, classes, prior, record, {json_sha256, npz_sha256}) of the pinned classifier; fold f: that CV fold's
    model at the chosen C. Refuses files that do not hash as the pin records."""
    import numpy as np
    pin = read_pin()
    d = classifier_dir()
    jsha, nsha = _sha(d / "classifier.json"), _sha(d / "classifier.npz")
    if jsha is None:
        raise E3Refused("%s is missing: restore it from the pin (restore-classifier --from DIR); E3's classifier is "
                        "never refitted" % (d / "classifier.json"))
    if jsha != pin.get("json_sha256") or nsha != pin.get("npz_sha256"):
        raise E3Refused("the classifier in %s does not hash as its pin %s records" % (d, pin_path()))
    rec = _read_json(d / "classifier.json")
    if (rec.get("files") or {}).get("npz_sha256") != nsha:
        raise E3Refused("classifier.npz does not hash as classifier.json records")
    with np.load(str(d / "classifier.npz"), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if fold is None:
        W, b, classes = a["W"], a["b"], a["classes"]
    else:
        if "fold%d_W" % fold not in a:
            raise E3Refused("the classifier holds no fold %d" % fold)
        W, b, classes = a["fold%d_W" % fold], a["fold%d_b" % fold], a["fold%d_classes" % fold]
    return W, b, classes, a["prior"], rec, {"json_sha256": jsha, "npz_sha256": nsha}


def restore_classifier(src):
    """Copy the pinned classifier's files from a moved-aside E3 directory (or its classifier/) into this one,
    checking every sha256 against the pin. Never refits."""
    pin = read_pin()
    src = Path(src)
    if (src / "classifier").is_dir():
        src = src / "classifier"
    d = classifier_dir()
    if (d / "classifier.json").exists():
        raise E3Refused("%s exists: nothing to restore" % (d / "classifier.json"))
    want = {"classifier.json": pin.get("json_sha256"), "classifier.npz": pin.get("npz_sha256"),
            "train_crops.csv": pin.get("train_crops_sha256"), "train_emb.npz": pin.get("train_emb_sha256")}
    for name, sha in want.items():
        if _sha(src / name) != sha:
            raise E3Refused("%s does not hash as the pin records (%s): not E3's classifier" % (src / name,
                                                                                              str(sha)[:12]))
    d.mkdir(parents=True, exist_ok=True)
    for name in CLASSIFIER_FILES:
        _write_once(d / name, (src / name).read_bytes())
    log("classifier restored from %s (sha256s as pinned)" % src)
    return load_classifier()[4]


def _clf_cfg(fold=None):
    W, b, classes, prior, rec, shas = load_classifier(fold)
    return {"W": W, "b": b, "classes": classes, "prior": prior}, rec, shas


# ------------------------------------------------------------------ dev ground-truth accuracy
def dev_gt_accuracy(embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, keep_crops=False):
    """The classifier on dev's ground-truth crops (reported): one pass of E3-M seed 0's detector on dev in mode
    gt (its predictions emitted unchanged), the ground-truth boxes cut through the same geometry code as the
    detected ones. Writes classifier/dev_gt.json once. keep_crops (tests) returns the crops too."""
    import numpy as np
    out = classifier_dir() / "dev_gt.json"
    if out.exists():
        raise E3Refused("%s exists: written once" % out)
    ccfg, _rec, shas = _clf_cfg()
    src = _source("M", SEEDS[0], testing_ok=_testing_ok())
    if embedder is None:
        embedder = V.BioclipEmbedder()
    with _workers(workers) as wk:
        cfg = dict(ccfg, mode="gt", nms="locked", embedder=embedder, workers=wk, keep_crops=keep_crops)
        ps = _score_pass(src["weights"], EXAM, cfg, lock_check=lock_check, batch=batch, device=device)
    gt = ps["gt"]
    conf = np.zeros((C.NC, C.NC), dtype=np.int64)
    per = {}
    for t, p1, p3 in gt:
        conf[t, p1] += 1
        e = per.setdefault(C.CLASS_NAMES[t], {"n": 0, "top1": 0, "top3": 0})
        e["n"] += 1
        e["top1"] += int(p1 == t)
        e["top3"] += int(t in p3)
    n = len(gt)
    rec = {"format": FMT_DEV_GT, "exam": EXAM, "classifier": shas, "detector": {k: src[k] for k in (
        "exp", "run_id", "weights_sha256")}, "n": n,
           "gt_small_left_out": int(ps["stats"].get("gt_small_left_out", 0)),
           "top1": (sum(1 for t, p1, _ in gt if p1 == t) / n) if n else None,
           "top3": (sum(1 for t, _p, p3 in gt if t in p3) / n) if n else None,
           "per_class": {k: dict(v, top1=v["top1"] / v["n"], top3=v["top3"] / v["n"]) for k, v in sorted(per.items())},
           "confusion": conf.tolist(), "gt_roundtrip_max_diff": ps["stats"].get("gt_roundtrip_max_diff"),
           "pass": {"species_map50_95": ps["result"].get("species_map50_95"),
                    "vs_recorded": _finite(abs(float(ps["result"]["species_map50_95"])
                                               - float(src["recorded"]["species_map50_95"])))
                    if src["recorded"].get("species_map50_95") is not None else None,
                    "production": ps["result"].get("production")},
           "code": _code_shas(), "created_utc": _utc(), "note": "reported only: the classifier on dev's "
                                                                  "ground-truth crops, never deciding"}
    _write_once(out, rec)
    log("dev ground truth: top-1 %s, top-3 %s on %d crops" % (rec["top1"], rec["top3"], n))
    if keep_crops:
        rec = dict(rec, crops=ps["crops"])
    return rec


class _workers(object):
    """A verify._Workers pool to enter, or the caller's (already entered)."""

    def __init__(self, given, procs=None):
        self.given = given
        self.own = None
        self.procs = procs

    def __enter__(self):
        if self.given is not None:
            return self.given
        self.own = V._Workers(self.procs if self.procs is not None else (1 if _testing_ok() else PROCS))
        return self.own.__enter__()

    def __exit__(self, *exc):
        if self.own is not None:
            return self.own.__exit__(*exc)
        return False


# ------------------------------------------------------------------ equivalence
def equivalence(lock_check=True, batch=S.BATCH, device=None, workers=None):
    """The identity check (amendment): each b_v2_m640 run's own predictions through E3's path reproduce its
    recorded protocol dev score within EQ_TOL, and the stage-1 helper the scorer's own agnostic AP. Writes
    equivalence.json once, only when every seed passes."""
    out = root() / "equivalence.json"
    if out.exists():
        raise E3Refused("%s exists: written once" % out)
    testing = _testing_ok()
    per, fails = {}, []
    with _workers(workers) as wk:
        for s in SEEDS:
            src = _source("M", s, testing_ok=testing)
            ps = _score_pass(src["weights"], EXAM, {"mode": "identity", "nms": "locked", "workers": wk},
                             lock_check=lock_check, batch=batch, device=device)
            res = ps["result"]
            d = dict(res, locked_scorer_sha256=_locked(res["scorer_sha256"]))
            r = {"weights_sha256": src["weights_sha256"], "recorded": {k: src["recorded"][k] for k in (
                "file", "sha256", "species_map50_95", "agnostic_map50_95", "production")},
                 "identity": {k: res.get(k) for k in ("map50_95", "species_map50_95", "agnostic_map50_95",
                                                      "production")}}
            try:
                r["vs_protocol_score"] = SN.compare_with_protocol(d, src["recorded_doc"], testing=not res["production"])
            except SN.NativeRefused as e:
                r["vs_protocol_score"] = {"compared": False, "refused": str(e)}
                fails.append("seed %d: %s" % (s, e))
            sd = abs(float(res["species_map50_95"]) - float(src["recorded_doc"]["species_map50_95"]))
            r["species_diff"] = sd
            if not sd <= EQ_TOL:
                fails.append("seed %d: species mean differs by %.6f (> %g)" % (s, sd, EQ_TOL))
            if r["vs_protocol_score"].get("compared") is not True:
                fails.append("seed %d: not compared with the recorded protocol score (%s)"
                             % (s, r["vs_protocol_score"].get("why") or r["vs_protocol_score"].get("refused")))
            s1d = abs(float(ps["stage1_agnostic"][0]) - float(res["agnostic_map50_95"]))
            r["stage1_vs_identity_agnostic"] = s1d
            if not s1d <= STAGE1_EXACT:
                fails.append("seed %d: the stage-1 helper's agnostic AP differs from the scorer's own by %.3g"
                             % (s, s1d))
            nat = _read_json(C.INC_DIR / REFERENCE / "runs" / src["final_run_id"] / "scores" / "dev@640.json")
            r["vs_native_dev640_agnostic"] = _finite(abs(float(nat["agnostic_map50_95"]) - float(
                res["agnostic_map50_95"]))) if isinstance(nat, dict) and nat.get("agnostic_map50_95") is not None \
                else None
            r["emitted"] = {k: ps["stats"].get(k, 0) for k in ("rows", "dropped_conf", "dropped_cap", "capped")}
            per[str(s)] = r
    if fails:
        raise E3Refused("the identity path does not reproduce b_v2_m640's recorded dev scores: %s; no E3 score is "
                        "taken" % "; ".join(fails))
    import torch
    import ultralytics
    rec = {"format": FMT_EQUIVALENCE, "exam": EXAM, "reference": REFERENCE, "seeds": list(SEEDS), "per_seed": per,
           "all_passed": True, "tolerance": EQ_TOL, "stage1_exact": STAGE1_EXACT, "code": _code_shas(),
           "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                        "open_clip": _versions().get("open_clip")},
           "settings": {"imgsz": IMGSZ, "batch": batch, "nms": "locked", "mode": "identity"},
           "testing": testing, "created_utc": _utc()}
    _write_once(out, rec)
    log("equivalence: the identity path reproduces b_v2_m640's dev scores on seeds %s" % list(SEEDS))
    return rec


def check_equivalence():
    """equivalence.json (all passed) and its sha256; refuses when the code changed since it ran."""
    p = root() / "equivalence.json"
    eq = _read_json(p)
    if not isinstance(eq, dict) or eq.get("format") != FMT_EQUIVALENCE or eq.get("all_passed") is not True:
        raise E3Refused("%s is missing or did not pass: no E3 score is taken without the equivalence check" % p)
    now = _code_shas()
    if eq.get("code") != now:
        diff = sorted(k for k in set(now) | set(eq.get("code") or {}) if now.get(k) != (eq.get("code") or {}).get(k))
        raise E3Refused("the code changed since the equivalence check ran (%s): move twostage/%s aside, restore its "
                        "classifier (restore-classifier) and rerun E3" % (", ".join(diff), E3_NAME))
    return eq, _sha(p)


# ------------------------------------------------------------------ an E3 score file
def _e3_doc(arm, seed, exam, src, ps, cfg, clf_shas, eq_sha, ro_clf, variant=""):
    res = ps["result"]
    locked = _locked(res["scorer_sha256"])
    test = not res.get("production")
    doc = dict(res)
    doc.update({"format": FMT_SCORE, "name": E3_NAME, "arm": arm, "seed": int(seed), "exp": src["exp"],
                "variant": variant or None, "scorer_sha256": stamp(locked, test), "locked_scorer_sha256": locked,
                "production": False, "e3_production": not test, "imgsz": IMGSZ,
                "settings": dict(res["settings"], nms=cfg["nms"], agnostic_nms=cfg["nms"] == "agnostic",
                                 mode=cfg["mode"], top_k=int(cfg.get("top_k") or TOP_K)),
                "stage1": {"exp": src["exp"], "run_id": src["run_id"], "weights": src["weights"],
                           "weights_sha256": src["weights_sha256"], "base_run_json_sha256": src["base_run_json_sha256"],
                           "final_run_json_sha256": src["final_run_json_sha256"], "nms": cfg["nms"],
                           "n_boxes": int(ps["stats"].get("stage1_boxes", 0)),
                           "agnostic_map50_95": ps["stage1_agnostic"][0], "agnostic_map50": ps["stage1_agnostic"][1]},
                "stage2": {"classifier_json_sha256": clf_shas["json_sha256"],
                           "classifier_npz_sha256": clf_shas["npz_sha256"],
                           "embedder": getattr(cfg.get("embedder"), "name", None), "top_k": int(cfg.get("top_k") or TOP_K),
                           "n_crops": int(ps["stats"].get("crops", 0)), "n_small": int(ps["stats"].get("small", 0)),
                           "n_degenerate": int(ps["stats"].get("degenerate", 0)),
                           "embed_seconds": ps["stats"].get("embed_seconds")},
                "emitted": {k: int(ps["stats"].get(k, 0)) for k in ("rows", "rows_candidate", "dropped_conf",
                                                                     "dropped_cap", "capped")},
                "checks": {k: ps["stats"].get(k) for k in ("orientations", "gt_roundtrip_max_diff",
                                                           "full_recompute_max_abs_diff",
                                                           "species_restricted_max_abs_diff")},
                "equivalence_sha256": eq_sha,
                "research_only": {"flag": bool(src["research_only"] or ro_clf), "detector": src["research_only"],
                                  "classifier": bool(ro_clf)},
                "code": _code_shas(), "created_utc": _utc(), "seconds": ps["seconds"]})
    return doc


def _commit_file(d, exam, doc, arrays, stage1, variant=""):
    """Write an E3 score once: the stage-1 npz and the images npz, then the JSON (inc2.scorer_native's commit)."""
    jn, nn, sn = file_names(exam, variant)
    js, npz, s1 = d / jn, d / nn, d / sn
    if js.exists() or js.is_symlink():
        raise E3Refused("%s exists: an E3 score is written once" % js)
    d.mkdir(parents=True, exist_ok=True)
    staged = npz.with_name(".%s.%d.%d.staged" % (npz.name, os.getpid(), time.monotonic_ns()))
    try:
        if stage1 is not None:
            s1sha = SC.save_npz(s1, stage1)
            doc["stage1_images"] = {"name": s1.name, "sha256": s1sha, "n_boxes": int(len(stage1["q"]))}
        npz_sha = SC.save_npz(staged, arrays)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise
    doc["images"] = {"name": npz.name, "sha256": npz_sha, "n_images": int(len(arrays["keys"])),
                     "n_predictions": int(len(arrays["conf"])), "n_targets": int(len(arrays["target_cls"]))}
    try:
        SN._commit(js, npz, staged, doc)
    except SN.NativeRefused as e:
        raise E3Refused(str(e))
    return js


# ------------------------------------------------------------------ L23F: score an arm
def score_arm(arm, seeds=SEEDS, embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None,
              reported=True):
    """E3's scoring job of one arm (lever L23F; module docstring): the
    classifier (fitted once if missing), the dev ground-truth accuracy and
    the equivalence check first when missing; then every dev pass of the
    arm (written once), capacity/e3_score_<arm>.json (dev only), then the
    reported passes (E3-M's locked-NMS reading, ImageWeeds), whose
    failure is recorded and does not fail the job."""
    if arm not in ARMS:
        raise E3Refused("--arm %r is not one of E3's arms %s" % (arm, list(ARM_ORDER)))
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise E3Refused("seeds %s are not a subset of E3's %s" % (seeds, list(SEEDS)))
    testing = _testing_ok()
    t0 = time.time()
    with _workers(workers) as wk:
        if not (classifier_dir() / "classifier.json").exists():
            if pin_path().exists():
                raise E3Refused("the classifier is missing from %s but its pin %s exists: restore it "
                                "(restore-classifier --from DIR); E3's classifier is never refitted"
                                % (classifier_dir(), pin_path()))
            if embedder is None:
                embedder = V.BioclipEmbedder()
            fit_classifier(embedder=embedder, testing=testing, workers=wk)
        ccfg, crec, shas = _clf_cfg()
        if not (classifier_dir() / "dev_gt.json").exists():
            if embedder is None:
                embedder = V.BioclipEmbedder()
            dev_gt_accuracy(embedder=embedder, workers=wk, lock_check=lock_check, batch=batch, device=device)
        if not (root() / "equivalence.json").exists():
            equivalence(lock_check=lock_check, batch=batch, device=device, workers=wk)
        _eq, eq_sha = check_equivalence()
        srcs = {s: _source(arm, s, testing_ok=testing) for s in seeds}       # every source before any pass
        ro_clf = bool((crec.get("research_only") or {}).get("flag", True))
        files = []
        for s in seeds:
            d = arm_dir(arm, s)
            js = d / file_names(EXAM)[0]
            status = "kept"
            if not js.exists():
                if embedder is None:
                    embedder = V.BioclipEmbedder()
                cfg = dict(ccfg, mode="two_stage", nms=NMS[arm], embedder=embedder, workers=wk, top_k=TOP_K)
                ps = _score_pass(srcs[s]["weights"], EXAM, cfg, lock_check=lock_check, batch=batch, device=device)
                doc = _e3_doc(arm, s, EXAM, srcs[s], ps, cfg, shas, eq_sha, ro_clf)
                rec_ag = srcs[s]["recorded"].get("agnostic_map50_95")
                diff = abs(float(ps["stage1_agnostic"][0]) - float(rec_ag)) if rec_ag is not None else None
                if arm in STAGE1_COMPARED:
                    if diff is None or not diff <= STAGE1_TOL:
                        raise E3Refused("%s seed %d: the stage-1 box set's agnostic AP %.6f is not the run's "
                                        "recorded %s (tolerance %g): not the box set E3 pre-registered"
                                        % (arm, s, ps["stage1_agnostic"][0], rec_ag, STAGE1_TOL))
                    doc["stage1"]["vs_recorded_agnostic"] = {"compared": True, "recorded": float(rec_ag),
                                                            "max_abs_diff": diff, "tolerance": STAGE1_TOL,
                                                            "file": srcs[s]["recorded"]["file"],
                                                            "sha256": srcs[s]["recorded"]["sha256"]}
                else:
                    doc["stage1"]["vs_recorded_agnostic"] = {
                        "compared": False, "why": "%s NMS: a box set no recorded score was computed on" % NMS[arm],
                        "recorded": _finite(rec_ag), "abs_diff": _finite(diff), "file": srcs[s]["recorded"]["file"]}
                _commit_file(d, EXAM, doc, ps["arrays"], ps["stage1"])
                status = "written"
                log("E3-%s seed %d on dev: species %.4f, stage-1 agnostic %.4f (recorded %s, %s), %d crops, %.0fs"
                    % (arm, s, doc["species_map50_95"] or 0.0, ps["stage1_agnostic"][0], rec_ag,
                       "compared" if arm in STAGE1_COMPARED else "%s NMS, not compared" % NMS[arm],
                       doc["stage2"]["n_crops"], ps["seconds"]))
            dd = _read_json(js) or {}
            files.append({"seed": s, "file": "arms/%s/s%d/%s" % (arm, s, js.name), "sha256": _sha(js),
                          "images_sha256": (dd.get("images") or {}).get("sha256"), "status": status,
                          "species_map50_95": dd.get("species_map50_95"),
                          "stage1_agnostic_map50_95": (dd.get("stage1") or {}).get("agnostic_map50_95")})
        rec = {"format": FMT_SCORE_RECORD, "status": "complete", "arm": arm, "exp": ARMS[arm], "exam": EXAM,
               "seeds": seeds, "dev": files, "classifier": shas, "equivalence_sha256": eq_sha,
               "reported": {"status": "pending" if reported else "not run"}, "written_utc": _utc(),
               "note": "dev only; names relative to twostage/%s and sha256s, no path; the reported passes are in "
                       "arms/%s/reported.json" % (E3_NAME, arm)}
        _write_json(score_record_path(arm), rec)
        if reported:
            rep = _reported_passes(arm, seeds, srcs, ccfg, shas, eq_sha, ro_clf, embedder, wk, lock_check, batch,
                                   device)
            rec["reported"] = {"status": "complete", "passes": len(rep), "failed": sum(
                1 for x in rep if x["status"] == "failed")}
            rec["written_utc"] = _utc()
            _write_json(score_record_path(arm), rec)
    log("E3-%s scored on dev (%s), %.0fs" % (arm, ", ".join("s%d %s" % (f["seed"], f["status"]) for f in files),
                                              time.time() - t0))
    return rec


def _reported_passes(arm, seeds, srcs, ccfg, shas, eq_sha, ro_clf, embedder, wk, lock_check, batch, device):
    """E3-M's locked-NMS dev reading and every arm's ImageWeeds passes, each written once; a failure is
    recorded (arms/<arm>/reported.json) and never raised."""
    out = []
    plan = []
    if arm in REPORTED_NMS:
        plan += [(s, EXAM, REPORTED_NMS[arm], reported_variant(arm)) for s in seeds]
    plan += [(s, exam, NMS[arm], "") for exam in REPORT_EXAMS for s in seeds]
    for s, exam, nms, variant in plan:
        d = arm_dir(arm, s)
        js = d / file_names(exam, variant)[0]
        item = {"seed": s, "exam_file": "arms/%s/s%d/%s" % (arm, s, js.name), "nms": nms}
        if js.exists():
            out.append(dict(item, status="kept", sha256=_sha(js)))
            continue
        try:
            if embedder is None:
                embedder = V.BioclipEmbedder()
            cfg = dict(ccfg, mode="two_stage", nms=nms, embedder=embedder, workers=wk, top_k=TOP_K)
            ps = _score_pass(srcs[s]["weights"], exam, cfg, lock_check=lock_check, batch=batch, device=device)
            doc = _e3_doc(arm, s, exam, srcs[s], ps, cfg, shas, eq_sha, ro_clf, variant=variant)
            rec_doc = _read_json(C.INC_DIR / srcs[s]["exp"] / "runs" / srcs[s]["final_run_id"] / "scores"
                                 / ("%s.json" % exam))
            rec_ag = _finite(rec_doc.get("agnostic_map50_95")) if isinstance(rec_doc, dict) else None
            doc["stage1"]["vs_recorded_agnostic"] = {
                "compared": False, "why": "a reported reading: recorded beside, never checked",
                "recorded": rec_ag, "abs_diff": _finite(abs(float(ps["stage1_agnostic"][0]) - rec_ag))
                if rec_ag is not None else None}
            _commit_file(d, exam, doc, ps["arrays"], ps["stage1"], variant=variant)
            out.append(dict(item, status="written", sha256=_sha(js)))
        except Exception as e:               # noqa: BLE001 - a reported pass never fails the job
            log("WARNING: E3-%s seed %d reported pass %s failed: %s: %s" % (arm, s, js.name, type(e).__name__, e))
            out.append(dict(item, status="failed", error="%s: %s" % (type(e).__name__, str(e)[:400])))
    _write_json(root() / "arms" / arm / "reported.json", {"arm": arm, "passes": out, "written_utc": _utc()})
    return out


# ------------------------------------------------------------------ L23G: the verdict
def _e3_files(arm, testing_ok, clf_shas, eq, pin):
    """({seed: (doc, arrays, input)}, [missing]) of an arm's dev files, refusing every departure (amendment)."""
    out, missing = {}, []
    first_stamps = None
    for s in SEEDS:
        d = arm_dir(arm, s)
        js = d / file_names(EXAM)[0]
        doc = _read_json(js)
        if doc is None:
            missing.append("arms/%s/s%d/%s" % (arm, s, js.name))
            continue
        probs = []
        if doc.get("format") != FMT_SCORE or doc.get("exam") != EXAM or doc.get("imgsz") != IMGSZ \
                or doc.get("variant") or doc.get("arm") != arm:
            probs.append("not E3-%s's dev score (format, exam, imgsz, variant, arm)" % arm)
        if doc.get("e3_production") is not True and not testing_ok:
            probs.append("a test-mode score (%s)" % "; ".join(doc.get("deviations") or []))
        if doc.get("seed") != s:
            probs.append("its seed field is %r, its directory s%d" % (doc.get("seed"), s))
        s1 = doc.get("stage1") or {}
        src = _source(arm, s, testing_ok=testing_ok)
        if s1.get("weights_sha256") != src["weights_sha256"] or doc.get("weights_sha256") != src["weights_sha256"]:
            probs.append("its stage-1 weights are not %s base__s%d's (and final__base__s%d's)" % (ARMS[arm], s, s))
        if arm == "M":
            fj = _read_json(C.INC_DIR / REFERENCE / "runs" / ("final__base__s%d" % s) / "run.json") or {}
            if s1.get("weights_sha256") != fj.get("weights_sha256"):
                probs.append("E3-M seed %d's stage-1 weights are not the reference's seed-%d final weights" % (s, s))
        if s1.get("nms") != NMS[arm] or (doc.get("settings") or {}).get("nms") != NMS[arm]:
            probs.append("its NMS is %r, not %r" % (s1.get("nms"), NMS[arm]))
        if arm in STAGE1_COMPARED and (s1.get("vs_recorded_agnostic") or {}).get("compared") is not True:
            probs.append("its stage-1 agnostic AP was not checked against the recorded one")
        st2 = doc.get("stage2") or {}
        if st2.get("classifier_npz_sha256") != clf_shas["npz_sha256"] \
                or st2.get("classifier_json_sha256") != clf_shas["json_sha256"]:
            probs.append("it was scored with another classifier than the pinned one")
        if doc.get("equivalence_sha256") != eq["sha256"] or doc.get("code") != eq["doc"].get("code"):
            probs.append("it was scored by other code, or under another equivalence check, than equivalence.json's")
        if str(doc.get("created_utc") or "") <= str(pin.get("written_utc") or "~"):
            probs.append("it was not taken after the classifier was fitted (pin %s)" % pin.get("written_utc"))
        npz = d / file_names(EXAM)[1]
        img = doc.get("images") or {}
        if _sha(npz) is None or _sha(npz) != img.get("sha256"):
            probs.append("%s does not hash as recorded" % npz.name)
        if probs:
            raise E3Refused("arms/%s/s%d/%s: %s" % (arm, s, js.name, "; ".join(probs)))
        arrays = SC.load_npz(npz)
        if C.sha256_text("\n".join(str(k) for k in arrays["keys"])) != doc.get("key_order_sha256"):
            raise E3Refused("%s is not in the key order its score records" % npz)
        st = _stamps(doc)
        if first_stamps is None:
            first_stamps = st
        elif st != first_stamps:
            raise E3Refused("arms/%s/s%d was scored on another exam, scorer or settings than s%d (%s differ)"
                            % (arm, s, SEEDS[0], sorted(k for k in st if st[k] != first_stamps[k])))
        out[s] = (doc, arrays, {"seed": s, "file": "arms/%s/s%d/%s" % (arm, s, js.name), "sha256": _sha(js),
                                "images_sha256": img["sha256"], "weights_sha256": s1["weights_sha256"],
                                "classifier_npz_sha256": st2["classifier_npz_sha256"]})
    return out, missing


def _stamps(doc):
    out = {k: doc.get(k) for k in E3_STAMPS}
    out.update(("settings.%s" % k, (doc.get("settings") or {}).get(k)) for k in SETTINGS)
    return out


def _reference_files(e2v, testing_ok):
    """b_v2_m640's native dev@640 files and arrays, each exactly as capacity/e2_v1.json lists it."""
    listed = {(r.get("run_id"), r.get("sha256")) for a in (e2v.get("arms") or {}).values() if isinstance(a, dict)
              for r in a.get("reference_inputs") or [] if isinstance(r, dict)}
    try:
        files, missing = B._native_files(REFERENCE, list(SEEDS), IMGSZ, testing_ok)
    except B.BaselineError as e:
        raise E3Refused("the reference's native dev files: %s" % e)
    if missing:
        raise E3Refused("the reference's native dev files %s are missing (E2's verdict read them)" % missing)
    out = {}
    for s in SEEDS:
        doc, arrays, inp = files[s]
        if (inp["run_id"], inp["sha256"]) not in listed:
            raise E3Refused("%s %s's dev@640.json (%s) is not the file capacity/e2_v1.json read"
                            % (REFERENCE, inp["run_id"], str(inp["sha256"])[:12]))
        fj = _read_json(C.INC_DIR / REFERENCE / "runs" / inp["run_id"] / "run.json") or {}
        if doc.get("weights_sha256") != fj.get("weights_sha256"):
            raise E3Refused("%s %s's native file names weights other than its final run's" % (REFERENCE, inp["run_id"]))
        if not testing_ok and ((doc.get("vs_protocol_score") or {}).get("compared") is not True
                               or doc.get("native_production") is not True):
            raise E3Refused("%s %s's native file was not checked against its production protocol score"
                            % (REFERENCE, inp["run_id"]))
        out[s] = (doc, arrays, dict(inp, weights_sha256=doc.get("weights_sha256")))
    return out


def _train_boxes_check(crec):
    """base_v2's current labels give the classifier's train_boxes_sha256, and every training crop row is a base_v2
    box with its label (classifier.json's train_crops.csv, by sha256)."""
    try:
        msha = C2.verify_manifest_against_lock_v2("base_v2")
    except C2.Inc2Error as e:
        raise E3Refused("LOCK v2's base_v2: %s" % e)
    if (crec.get("manifest") or {}).get("sha256") != msha:
        raise E3Refused("the classifier was fitted on manifest %s, not LOCK v2's base_v2 %s"
                        % (str((crec.get("manifest") or {}).get("sha256"))[:12], msha[:12]))
    rows = C.read_manifest(C2.v2_manifest_path("base_v2"))
    boxes, sha = all_boxes(rows)
    if sha != crec.get("train_boxes_sha256"):
        raise E3Refused("base_v2's labels give train_boxes_sha256 %s, the classifier records %s: it was not fitted "
                        "on base_v2's boxes" % (sha[:12], str(crec.get("train_boxes_sha256"))[:12]))
    p = classifier_dir() / "train_crops.csv"
    if _sha(p) != (crec.get("files") or {}).get("train_crops_sha256"):
        raise E3Refused("train_crops.csv does not hash as classifier.json records")
    have = {(k, b, c) for k, b, c, *_ in boxes}
    with open(p, newline="") as fh:
        rd = csv.reader(fh)
        if tuple(next(rd)) != TRAIN_CROP_FIELDS:
            raise E3Refused("train_crops.csv: unexpected columns")
        n = 0
        for row in rd:
            n += 1
            if (row[1], int(row[3]), int(row[4])) not in have:
                raise E3Refused("training crop %s box %s (label %s) is not a base_v2 box with that label"
                                % (row[1], row[3], row[4]))
    if n != crec.get("train_boxes") or n + int((crec.get("skipped_small") or {}).get("total") or 0) != len(boxes):
        raise E3Refused("train_crops.csv holds %d crops; the classifier records %s (and %s small) of %d boxes"
                        % (n, crec.get("train_boxes"), (crec.get("skipped_small") or {}).get("total"), len(boxes)))
    return {"manifest_sha256": msha, "train_boxes_sha256": sha, "train_crops": n}


def _pair(a_files, b_files, seed_text, resamples):
    """D, pooled sd, SE and the conditions of a - b over SEEDS (inc2.baseline native_bootstrap, unchanged)."""
    av = [float(a_files[s][0]["species_map50_95"]) for s in SEEDS]
    bv = [float(b_files[s][0]["species_map50_95"]) for s in SEEDS]
    ma, sa = B._mean_sd(av)
    mb, sb = B._mean_sd(bv)
    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2.0)
    diff = ma - mb
    boot = B.native_bootstrap([a_files[s][1] for s in SEEDS], [b_files[s][1] for s in SEEDS], resamples=resamples,
                              seed_text=seed_text)
    se = boot["se"]
    conds = {"above_2_pooled_sd": diff > 2.0 * pooled, "above_se": se is not None and diff > se}
    return {"seeds": list(SEEDS), "dev": av, "mean": ma, "sd": sa, "other_dev": bv, "other_mean": mb, "other_sd": sb,
            "diff": diff, "pooled_sd": pooled, "two_pooled_sd": 2.0 * pooled, "se_diff": se,
            "n_valid": boot["n_valid"], "conditions": conds, "per_species_se": boot["per_species"]}


def e3_gate(testing_ok=False):
    """What the verdict (and the test read) rests on besides the arm files: E2's decided verdict under its
    parameters, the pinned classifier, its training boxes, the equivalence check. Returns a dict."""
    try:
        vp2, v2 = B._e2_verdict_doc(reference=REFERENCE, testing_ok=testing_ok)
    except B.BaselineError as e:
        raise E3Refused("E2's verdict: %s" % e)
    pin = read_pin()
    _W, _b, _cls, _prior, crec, shas = load_classifier()
    g = crec.get("guard") or {}
    if g.get("refused") != 0 or g.get("crosscheck_hits") != 0 or (crec.get("test_v1") or {}).get("hits") != 0:
        raise E3Refused("the classifier's guard record is not clean (refused %r, cross-check %r, test v1 hits %r)"
                        % (g.get("refused"), g.get("crosscheck_hits"), (crec.get("test_v1") or {}).get("hits")))
    if crec.get("written_utc") != pin.get("written_utc") or crec.get("train_boxes_sha256") != pin.get(
            "train_boxes_sha256"):
        raise E3Refused("classifier.json and its pin disagree on the fit's time or training boxes")
    tb = _train_boxes_check(crec)
    _reference_manifest_check(tb["manifest_sha256"], testing_ok)
    eq_doc, eq_sha = check_equivalence()
    dg = _read_json(classifier_dir() / "dev_gt.json")
    if isinstance(dg, dict) and str(dg.get("created_utc") or "") <= str(pin.get("written_utc")):
        raise E3Refused("dev_gt.json was not taken after the classifier was fitted")
    return {"e2_path": vp2, "e2": v2, "e2_sha256": _sha(vp2), "pin": pin, "classifier": crec, "clf_shas": shas,
            "train_boxes": tb, "eq": {"doc": eq_doc, "sha256": eq_sha}, "dev_gt": dg}


def e3_decision(testing_ok=False, resamples=RESAMPLES):
    """The pre-registered rule on E3's dev files and the reference's (dev only)."""
    g = e3_gate(testing_ok)
    ref = _reference_files(g["e2"], testing_ok)
    arms, pending, qualifying, loaded = {}, [], [], {}
    ref_stamps = None
    for s in SEEDS:
        st = _stamps(ref[s][0])
        if ref_stamps is None:
            ref_stamps = st
        elif st != ref_stamps:
            raise E3Refused("the reference's native files differ in exam, scorer or settings")
    for arm in ARM_ORDER:
        files, missing = _e3_files(arm, testing_ok, g["clf_shas"], g["eq"], g["pin"])
        if missing:
            arms[arm] = {"status": "pending", "missing": missing}
            pending.append(arm)
            continue
        st = _stamps(files[SEEDS[0]][0])
        if st != ref_stamps:
            raise E3Refused("E3-%s's files and the reference's were scored on another exam, scorer or settings (%s "
                            "differ)" % (arm, sorted(k for k in st if st[k] != ref_stamps[k])))
        loaded[arm] = files
        p = _pair(files, ref, SPECIES_SEED_TEXT, resamples)
        q = all(p["conditions"].values())
        species = {}
        for name in REPORTED_SPECIES:
            pa = [(files[s][0].get("per_class") or {}).get(name) for s in SEEDS]
            pr = [(ref[s][0].get("per_class") or {}).get(name) for s in SEEDS]
            if any(x is None for x in pa + pr):
                species[name] = {"arm_mean": None, "ref_mean": None, "diff": None, "se": None}
                continue
            am, rm = statistics.fmean(float(x) for x in pa), statistics.fmean(float(x) for x in pr)
            species[name] = {"arm_mean": am, "ref_mean": rm, "diff": am - rm,
                             "se": p["per_species_se"].get(name, {}).get("se")}
        s1 = [(files[s][0].get("stage1") or {}).get("agnostic_map50_95") for s in SEEDS]
        m1, sd1 = B._mean_sd([x for x in s1 if x is not None])
        arms[arm] = {"status": "decided", "exp": ARMS[arm], "nms": NMS[arm], "seeds": list(SEEDS),
                     "dev": p["dev"], "mean": p["mean"], "sd": p["sd"], "reference_dev": p["other_dev"],
                     "reference_mean": p["other_mean"], "reference_sd": p["other_sd"], "diff": p["diff"],
                     "pooled_sd": p["pooled_sd"], "two_pooled_sd": p["two_pooled_sd"], "se_diff": p["se_diff"],
                     "n_valid": p["n_valid"], "conditions": p["conditions"], "qualifies": q,
                     "stamps": st, "inputs": [files[s][2] for s in SEEDS],
                     "reference_inputs": [ref[s][2] for s in SEEDS],
                     "reported": {"stage1_agnostic": {"values": s1, "mean": m1, "sd": sd1},
                                  "e3_agnostic": [files[s][0].get("agnostic_map50_95") for s in SEEDS],
                                  "species": species,
                                  "research_only": any(bool((files[s][0].get("research_only") or {}).get("flag", True))
                                                       for s in SEEDS)}}
        if arm == "M":
            arms[arm]["reported"]["paired_per_seed"] = {
                "diffs": [p["dev"][i] - p["other_dev"][i] for i in range(len(SEEDS))],
                "note": "E3-M seed s uses the reference's seed-s detector: paired differences, reported only"}
            ag = []
            for s in SEEDS:
                x = _read_json(arm_dir("M", s) / file_names(EXAM, reported_variant("M"))[0])
                ag.append({"seed": s, "species_map50_95": x.get("species_map50_95"),
                           "stage1_agnostic_map50_95": (x.get("stage1") or {}).get("agnostic_map50_95")}
                          if isinstance(x, dict) else {"seed": s, "missing": True})
            arms[arm]["reported"]["%s_reading" % reported_variant("M")] = ag
        if q:
            qualifying.append(arm)
    status = "decided" if not pending else "pending"
    chosen = None
    if status == "decided" and qualifying:
        best = max(arms[k]["diff"] for k in qualifying)
        chosen = next(k for k in ARM_ORDER if k in qualifying and abs(arms[k]["diff"] - best) <= TIE_TOL)
    attribution = {"status": "pending"}
    if "A" in loaded and "B" in loaded:
        p = _pair(loaded["B"], loaded["A"], ATTR_SEED_TEXT, resamples)
        attribution = {"status": "decided", "comparison": "E3-B - E3-A", "rule": ATTR_RULE, "diff": p["diff"],
                       "pooled_sd": p["pooled_sd"], "two_pooled_sd": p["two_pooled_sd"], "se_diff": p["se_diff"],
                       "n_valid": p["n_valid"], "conditions": p["conditions"],
                       "credited_to_data": all(p["conditions"].values()),
                       "bootstrap": {"seed_text": ATTR_SEED_TEXT, "seed": C.stable_int(ATTR_SEED_TEXT),
                                     "resamples": int(resamples)}}
    dg = g["dev_gt"] if isinstance(g["dev_gt"], dict) else {}
    ref_proto = [_read_json(C.INC_DIR / REFERENCE / "runs" / ("final__base__s%d" % s) / "scores" / "dev.json")
                 for s in SEEDS]
    pm, psd = B._mean_sd([float(x["species_map50_95"]) for x in ref_proto
                          if isinstance(x, dict) and x.get("species_map50_95") is not None])
    return {"format": FMT_VERDICT, "name": E3_NAME, "rule": RULE, "pre_registered": DECIDED_BY, "status": status,
            "exam": EXAM, "imgsz": IMGSZ, "reference": {"exp": REFERENCE, "seeds": list(SEEDS),
                                                        "protocol_dev": {"mean": pm, "sd": psd}},
            "arms": arms, "qualifying": [k for k in ARM_ORDER if k in qualifying], "chosen": chosen,
            "pending": pending, "attribution": attribution,
            "bootstrap": {"seed_text": SPECIES_SEED_TEXT, "seed": C.stable_int(SPECIES_SEED_TEXT),
                          "resamples": int(resamples), "paired": "one draw of the dev images for every run", "ddof": 1,
                          "statistic": "the 12-class mean AP50-95 over the species with a GT box in the resample, "
                                       "averaged over seeds per arm; the arm minus the reference"},
            "e2_verdict": {"name": g["e2_path"].name, "sha256": g["e2_sha256"], "status": g["e2"].get("status")},
            "classifier": dict(g["clf_shas"], C=(g["classifier"].get("fit") or {}).get("C"),
                               train_boxes=g["classifier"].get("train_boxes"),
                               train_boxes_sha256=g["classifier"].get("train_boxes_sha256"),
                               research_only=(g["classifier"].get("research_only") or {}).get("flag")),
            "equivalence_sha256": g["eq"]["sha256"],
            "reported": {"dev_gt": {k: dg.get(k) for k in ("n", "top1", "top3", "per_class")} if dg else None},
            "testing_allowed": bool(testing_ok),
            "on_decision": "record only: nothing switches; the autopilot raises one card; a person reads each "
                           "qualifying arm's sealed test once (inc2.twostage test-read); the chosen arm's is E3's "
                           "headline",
            "note": "dev only; ImageWeeds is in the report, for people"}


def canonical(d):
    """The decision of an E3 verdict document: what a recomputation must reproduce for a decided file to be kept."""
    d = d or {}
    out = {k: d.get(k) for k in DECISION_KEYS}
    out["bootstrap"] = {k: (d.get("bootstrap") or {}).get(k) for k in ("seed_text", "resamples")}
    arms = {}
    for k, a in sorted((d.get("arms") or {}).items()):
        a = a if isinstance(a, dict) else {}
        x = {f: a.get(f) for f in ARM_DECISION_KEYS}
        x["inputs"] = [[r.get(g) for g in ("seed", "sha256", "images_sha256", "weights_sha256",
                                           "classifier_npz_sha256")] for r in a.get("inputs") or []]
        x["reference_inputs"] = [[r.get(g) for g in ("run_id", "sha256", "images_sha256", "weights_sha256")]
                                 for r in a.get("reference_inputs") or []]
        arms[k] = x
    out["arms"] = arms
    at = d.get("attribution") or {}
    out["attribution"] = {k: at.get(k) for k in ATTR_DECISION_KEYS}
    out["attribution"]["bootstrap"] = {k: (at.get("bootstrap") or {}).get(k) for k in ("seed_text", "resamples")}
    out["e2_verdict_sha256"] = (d.get("e2_verdict") or {}).get("sha256")
    out["classifier"] = {k: (d.get("classifier") or {}).get(k) for k in ("json_sha256", "npz_sha256")}
    out["equivalence_sha256"] = d.get("equivalence_sha256")
    return json.loads(json.dumps(out, sort_keys=True))


def e3_report(decision, testing_ok=False):
    """For people: ImageWeeds 12-class and agnostic of every arm (its E3 files) and of the reference (its finals'
    protocol scores), mean +- sd, with the files read."""
    rows = {}
    for arm in ARM_ORDER:
        tw, ag, s1, read = [], [], [], []
        for s in SEEDS:
            js = arm_dir(arm, s) / file_names("imageweeds")[0]
            d = _read_json(js)
            if isinstance(d, dict) and d.get("format") == FMT_SCORE and (d.get("e3_production") or testing_ok):
                tw.append(d.get("species_map50_95"))
                ag.append(d.get("agnostic_map50_95"))
                s1.append((d.get("stage1") or {}).get("agnostic_map50_95"))
                read.append({"file": "arms/%s/s%d/%s" % (arm, s, js.name), "sha256": _sha(js)})
        rows["E3-%s" % arm] = _agg(tw, ag, read, s1)
    tw, ag, read = [], [], []
    for s in SEEDS:
        p = C.INC_DIR / REFERENCE / "runs" / ("final__base__s%d" % s) / "scores" / "imageweeds.json"
        d = _read_json(p)
        if isinstance(d, dict) and (d.get("production") is True or testing_ok):
            tw.append(d.get("species_map50_95"))
            ag.append(d.get("agnostic_map50_95"))
            read.append({"file": "%s/final__base__s%d/scores/imageweeds.json" % (REFERENCE, s), "sha256": _sha(p)})
    rows[REFERENCE] = _agg(tw, ag, read)
    return {"format": FMT_VERDICT + "-report", "status": decision.get("status"), "qualifying": decision.get(
        "qualifying"), "chosen": decision.get("chosen"), "imageweeds": rows,
            "note": "for people: ImageWeeds of each arm and of b_v2_m640; the decision reads dev only"}


def _agg(tw, ag, read, s1=None):
    tw = [float(x) for x in tw if x is not None]
    ag = [float(x) for x in ag if x is not None]
    m1, s1_ = B._mean_sd(tw)
    m2, s2 = B._mean_sd(ag)
    out = {"n": len(tw), "twelve": {"mean": m1, "sd": s1_}, "agnostic": {"mean": m2, "sd": s2}, "read": read}
    if s1 is not None:
        m3, s3 = B._mean_sd([float(x) for x in s1 if x is not None])
        out["stage1_agnostic"] = {"mean": m3, "sd": s3}
    return out


def _md(report, decision):
    def f(x):
        return "-" if x is None else "%.4f" % x
    lines = ["# E3: two-stage species detection (dev decides)", "",
             "| Arm | D (dev) | 2 x pooled sd | SE | qualifies |", "|---|---|---|---|---|"]
    for k in ARM_ORDER:
        a = (decision.get("arms") or {}).get(k) or {}
        if a.get("status") == "decided":
            lines.append("| E3-%s | %s | %s | %s | %s |" % (k, f(a["diff"]), f(a["two_pooled_sd"]), f(a["se_diff"]),
                                                          "yes" if a["qualifies"] else "no"))
        else:
            lines.append("| E3-%s | pending | | | |" % k)
    at = decision.get("attribution") or {}
    if at.get("status") == "decided":
        lines += ["", "Attribution E3-B - E3-A: D_data %s, 2 x pooled sd %s, SE %s -> %s" % (
            f(at["diff"]), f(at["two_pooled_sd"]), f(at["se_diff"]),
            "credited to base v3's data" if at["credited_to_data"] else "not credited")]
    lines += ["", "Chosen: %s" % ("E3-%s" % decision["chosen"] if decision.get("chosen") else "none"), "",
              "ImageWeeds (for people):", "", "| | 12-class mean +- sd | agnostic mean +- sd | n |", "|---|---|---|---|"]
    for k, r in report["imageweeds"].items():
        lines.append("| %s | %s +- %s | %s +- %s | %d |" % (k, f(r["twelve"]["mean"]), f(r["twelve"]["sd"]),
                                                          f(r["agnostic"]["mean"]), f(r["agnostic"]["sd"]), r["n"]))
    return "\n".join(lines) + "\n"


def verdict(out_dir=None, write=True, testing_ok=False, resamples=RESAMPLES):
    """capacity/e3_v1.json (dev only) and its report. A decided file is kept byte for byte when a recomputation's
    decision agrees, refused when it differs; a pending file is overwritten."""
    decision = e3_decision(testing_ok=testing_ok, resamples=resamples)
    decision["generated_utc"] = _utc()
    report = e3_report(decision, testing_ok=testing_ok)
    report["generated_utc"] = decision["generated_utc"]
    if write:
        d = capacity_dir(out_dir)
        out = d / ("%s.json" % E3_NAME)
        decision["out"] = str(out)
        old = _read_json(out)
        if isinstance(old, dict) and old.get("status") == "decided":
            a, b = canonical(old), canonical(decision)
            if a != b:
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                raise E3Refused("%s holds a decided verdict this recomputation does not reproduce (%s differ): a "
                                "decided E3 verdict is never rewritten; a person moves it aside" % (out, diff))
            decision["kept"] = True
        else:
            _write_json(out, {k: v for k, v in decision.items() if k not in ("kept", "out")})
        report["decision_sha256"] = _sha(out)
        _write_json(d / ("%s_report.json" % E3_NAME), report)
        md = d / ("%s_report.md" % E3_NAME)
        tmp = md.with_name(".%s.tmp" % md.name)
        tmp.write_text(_md(report, decision))
        os.replace(tmp, md)
    log("E3 verdict: %s; %s" % (decision["status"], "; ".join(
        "E3-%s D %.4f vs 2 pooled sd %.4f, SE %s -> %s" % (k, a["diff"], a["two_pooled_sd"],
                                                           "-" if a["se_diff"] is None else "%.4f" % a["se_diff"],
                                                           "qualifies" if a["qualifies"] else "does not qualify")
        if a.get("status") == "decided" else "E3-%s pending" % k for k, a in sorted(decision["arms"].items()))))
    return decision, report


def rescore_verdict(out_dir=None, testing_ok=None, resamples=RESAMPLES):
    """L23G: the verdict, then capacity/e3_rescore.json (complete: the score records' and the verdict's names and
    sha256s, no path). Refuses when the verdict is still pending."""
    testing_ok = _testing_ok() if testing_ok is None else testing_ok
    decision, _rep = verdict(out_dir=out_dir, testing_ok=testing_ok, resamples=resamples)
    if decision.get("status") != "decided":
        raise E3Refused("E3's verdict is pending: %s" % {k: a.get("missing") for k, a in decision["arms"].items()
                                                        if a.get("status") != "decided"})
    vp = verdict_path(out_dir)
    recs = []
    for arm in ARM_ORDER:
        p = score_record_path(arm)
        r = _read_json(p) or {}
        recs.append({"arm": arm, "name": p.name, "sha256": _sha(p), "status": r.get("status")})
    rec = {"format": FMT_RESCORE, "status": "complete", "exam": EXAM, "score_records": recs,
           "verdict": {"name": vp.name, "sha256": _sha(vp), "status": decision["status"],
                       "qualifying": decision["qualifying"], "chosen": decision["chosen"],
                       "credited_to_data": (decision.get("attribution") or {}).get("credited_to_data")},
           "written_utc": _utc(), "note": "dev only; names and sha256s, no path"}
    _write_json(rescore_path(out_dir), rec)
    log("E3 rescore record complete; qualifying %s, chosen %s" % (decision["qualifying"] or "none",
                                                                  decision["chosen"]))
    return rec


# ------------------------------------------------------------------ the test read (a person's)
def _verdict_doc(path=None, testing_ok=False):
    vp = Path(path) if path else verdict_path()
    v = _read_json(vp)
    if not isinstance(v, dict) or v.get("format") != FMT_VERDICT or v.get("status") != "decided":
        raise E3Refused("%s is not a decided E3 verdict: test is read only after it" % vp)
    bs = v.get("bootstrap") or {}
    probs = []
    if v.get("rule") != RULE:
        probs.append("its rule is not E3's pre-registered rule")
    if (v.get("reference") or {}).get("exp") != REFERENCE:
        probs.append("its reference is not %s" % REFERENCE)
    if bs.get("seed_text") != SPECIES_SEED_TEXT or bs.get("resamples") != RESAMPLES:
        probs.append("its bootstrap (%r, %r) is not the pre-registered %r, %d" % (bs.get("seed_text"),
                                                                                bs.get("resamples"),
                                                                                SPECIES_SEED_TEXT, RESAMPLES))
    if v.get("testing_allowed") is not False and not testing_ok:
        probs.append("it admitted test-mode files")
    if probs:
        raise E3Refused("%s was not decided under E3's pre-registered parameters (%s): no sealed test is read on it"
                        % (vp, "; ".join(probs)))
    return vp, v


def _build_script():
    return C.REPO / "weed_llm_benchmark" / "run_inc2_build.sh"


def test_read(arm, verdict_file=None):
    """Prepare the one read of a qualifying arm's sealed test (amendment): pin the verdict, the classifier and the
    stage-1 weights by sha256 and write the job's argv. Checks everything before writing anything."""
    if arm not in ARMS:
        raise E3Refused("--arm %r is not one of E3's arms" % (arm,))
    vp, v = _verdict_doc(verdict_file)
    if arm not in (v.get("qualifying") or []):
        raise E3Refused("E3-%s did not qualify (qualifying: %s): its test is not read (pre-registered)"
                        % (arm, v.get("qualifying") or "none"))
    td = test_dir(arm)
    held = [str(p) for p in [td / "e3_test_read.json"] + sorted(td.glob("s*/test.json")) if p.exists()]
    if held:
        prev = _read_json(td / "e3_test_read.json") or {}
        raise E3Refused("E3-%s's test read was prepared already (%s; argv %s): it is read once"
                        % (arm, ", ".join(held), prev.get("argv")))
    _W, _b, _c, _p, _rec, shas = load_classifier()
    vc = v.get("classifier") or {}
    if shas["json_sha256"] != vc.get("json_sha256") or shas["npz_sha256"] != vc.get("npz_sha256"):
        raise E3Refused("the classifier no longer hashes as the verdict recorded")
    inputs = ((v.get("arms") or {}).get(arm) or {}).get("inputs") or []
    weights = {}
    for inp in inputs:
        s = int(inp["seed"])
        src = _source(arm, s, testing_ok=True)
        if src["weights_sha256"] != inp.get("weights_sha256"):
            raise E3Refused("E3-%s seed %d's stage-1 weights hash to %s; the verdict read %s"
                            % (arm, s, src["weights_sha256"][:12], str(inp.get("weights_sha256"))[:12]))
        weights[str(s)] = {"exp": src["exp"], "run_id": src["run_id"], "weights_sha256": src["weights_sha256"]}
    if sorted(int(s) for s in weights) != list(SEEDS):
        raise E3Refused("the verdict names seeds %s of E3-%s, not %s" % (sorted(weights), arm, list(SEEDS)))
    logs = C.INC_DIR / "logs"
    argv = ["sbatch", "--parsable", "--job-name=inc_build_e3test_%s" % arm.lower(),
            "--output=%s" % (logs / "%x_%j.out"), str(_build_script()), "inc2.twostage", "score-test", "--arm", arm]
    rec = {"format": FMT_TEST_READ, "arm": arm, "verdict": vp.name, "verdict_sha256": _sha(vp),
           "verdict_diff": ((v.get("arms") or {}).get(arm) or {}).get("diff"), "chosen": v.get("chosen"),
           "headline": arm == v.get("chosen"), "classifier": shas, "weights": weights, "argv": argv,
           "written_utc": _utc(), "note": "the one read of this qualifying arm's sealed test, after the dev verdict; a "
                                          "person submits the argv; the scores stay off the platform's evidence"}
    logs.mkdir(parents=True, exist_ok=True)
    _write_once(td / "e3_test_read.json", rec)
    return rec


def score_test(arm, embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, verdict_file=None):
    """The person's test job: b_v2_m640's three recorded test scores reproduced through the identity path first
    (or refused before any E3 test score exists), then the arm's three seeds on test, each written once."""
    td = test_dir(arm)
    rr = _read_json(td / "e3_test_read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_TEST_READ or rr.get("arm") != arm:
        raise E3Refused("E3-%s has no prepared test read (test-read --arm %s first)" % (arm, arm))
    vp, _v = _verdict_doc(verdict_file)
    if _sha(vp) != rr.get("verdict_sha256"):
        raise E3Refused("the verdict changed since the test read was prepared")
    ccfg, _rec, shas = _clf_cfg()
    if shas != rr.get("classifier"):
        raise E3Refused("the classifier changed since the test read was prepared")
    srcs = {}
    for s in SEEDS:
        src = _source(arm, s, testing_ok=True)
        if src["weights_sha256"] != ((rr.get("weights") or {}).get(str(s)) or {}).get("weights_sha256"):
            raise E3Refused("E3-%s seed %d's weights changed since the test read was prepared" % (arm, s))
        srcs[s] = src
    done = [str(p) for p in sorted(td.glob("s*/test.json"))]
    if done:
        raise E3Refused("E3-%s's test was scored already (%s): read once" % (arm, done))
    token = object()
    _TEST_TOKENS.append(token)
    out = {"reference": [], "arm": []}
    try:
        with _workers(workers) as wk:
            for s in SEEDS:
                d = root() / "test" / "reference" / ("s%d" % s)
                js = d / "test.identity.json"
                if js.exists():
                    x = _read_json(js) or {}
                    if (x.get("vs_protocol_score") or {}).get("compared") is not True:
                        raise E3Refused("%s does not record a passed comparison" % js)
                    out["reference"].append({"seed": s, "status": "kept", "sha256": _sha(js)})
                    continue
                rsrc = _source("M", s, testing_ok=True)
                recorded = _read_json(C.INC_DIR / REFERENCE / "runs" / rsrc["final_run_id"] / "scores" / "test.json")
                ps = _score_pass(rsrc["weights"], "test", {"mode": "identity", "nms": "locked", "workers": wk},
                                 lock_check=lock_check, batch=batch, device=device, test_token=token)
                res = ps["result"]
                dd = dict(res, locked_scorer_sha256=_locked(res["scorer_sha256"]))
                try:
                    cmp_ = SN.compare_with_protocol(dd, recorded, testing=not res["production"])
                except SN.NativeRefused as e:
                    raise E3Refused("b_v2_m640 seed %d's test score is not reproduced by the identity path (%s): no "
                                    "E3 test score is taken" % (s, e))
                sd = abs(float(res["species_map50_95"]) - float((recorded or {}).get("species_map50_95")))
                if cmp_.get("compared") is not True or not sd <= EQ_TOL:
                    raise E3Refused("b_v2_m640 seed %d's test score is not reproduced (compared %s, species diff %.6f)"
                                    % (s, cmp_.get("compared"), sd))
                doc = dict(res, format=FMT_SCORE, name=E3_NAME, arm="reference", seed=s, variant="identity",
                           scorer_sha256=stamp(_locked(res["scorer_sha256"]), not res["production"]),
                           locked_scorer_sha256=_locked(res["scorer_sha256"]), production=False,
                           e3_production=bool(res["production"]), vs_protocol_score=cmp_, species_diff=sd,
                           code=_code_shas(), created_utc=_utc())
                _commit_identity(d, doc, ps["arrays"])
                out["reference"].append({"seed": s, "status": "written", "sha256": _sha(js)})
            if embedder is None:
                embedder = V.BioclipEmbedder()
            ro_clf = bool((_rec.get("research_only") or {}).get("flag", True))
            for s in SEEDS:
                cfg = dict(ccfg, mode="two_stage", nms=NMS[arm], embedder=embedder, workers=wk, top_k=TOP_K)
                ps = _score_pass(srcs[s]["weights"], "test", cfg, lock_check=lock_check, batch=batch, device=device,
                                 test_token=token)
                doc = _e3_doc(arm, s, "test", srcs[s], ps, cfg, shas, None, ro_clf)
                doc["stage1"]["vs_recorded_agnostic"] = {"compared": False, "why": "test: reported"}
                doc["test_read_sha256"] = _sha(td / "e3_test_read.json")
                _commit_file(td / ("s%d" % s), "test", doc, ps["arrays"], ps["stage1"])
                out["arm"].append({"seed": s, "species_map50_95": doc.get("species_map50_95")})
    finally:
        _TEST_TOKENS.remove(token)
    log("E3-%s test scored on seeds %s (test-report --arm %s)" % (arm, list(SEEDS), arm))
    return out


def _commit_identity(d, doc, arrays):
    js = d / "test.identity.json"
    npz = d / "test.identity.images.npz"
    if js.exists():
        raise E3Refused("%s exists: written once" % js)
    d.mkdir(parents=True, exist_ok=True)
    staged = npz.with_name(".%s.%d.%d.staged" % (npz.name, os.getpid(), time.monotonic_ns()))
    doc["images"] = {"name": npz.name, "sha256": SC.save_npz(staged, arrays)}
    try:
        SN._commit(js, npz, staged, doc)
    except SN.NativeRefused as e:
        raise E3Refused(str(e))


def test_report(arm, out_dir=None, testing_ok=False, verdict_file=None):
    """The arm's test read, for people: 12-class and agnostic mean +- sd against b_v2_m640's final test files,
    the gap to 0.90, the headline; pending while a score is missing. Writes capacity/e3_test_<arm>.{json,md}."""
    vp, v = _verdict_doc(verdict_file, testing_ok=testing_ok)
    if arm not in (v.get("qualifying") or []):
        raise E3Refused("E3-%s did not qualify: its test was not read" % arm)
    td = test_dir(arm)
    rr = _read_json(td / "e3_test_read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_TEST_READ or rr.get("arm") != arm:
        raise E3Refused("E3-%s has no test read prepared by test-read: its scores are not reported" % arm)
    if rr.get("verdict_sha256") != _sha(vp):
        raise E3Refused("the verdict changed since the test read was prepared")
    missing, arm_rows, ref_rows, s1, stamps, inputs = [], [], [], [], {}, []

    def stamp_of(d, what):
        st = {"locked_scorer_sha256": d.get("locked_scorer_sha256") or _locked(d.get("scorer_sha256"))}
        st.update((k, d.get(k)) for k in TEST_STAMPS)
        if not stamps:
            stamps.update(st)
        elif st != stamps:
            raise E3Refused("%s was scored on another scorer, test manifest or key order than the report's first "
                            "file (%s differ)" % (what, sorted(k for k in st if st[k] != stamps[k])))
    for s in SEEDS:
        p = td / ("s%d" % s) / "test.json"
        d = _read_json(p)
        w = ((rr.get("weights") or {}).get(str(s)) or {}).get("weights_sha256")
        if d is None:
            missing.append("E3-%s s%d" % (arm, s))
        else:
            if d.get("format") != FMT_SCORE or d.get("exam") != "test" or d.get("seed") != s:
                raise E3Refused("%s is not E3-%s seed %d's test score" % (p, arm, s))
            if d.get("e3_production") is not True and not testing_ok:
                raise E3Refused("%s is a test-mode score" % p)
            if (d.get("stage1") or {}).get("weights_sha256") != w:
                raise E3Refused("%s names stage-1 weights other than the ones its test read was prepared on" % p)
            if (d.get("stage2") or {}).get("classifier_npz_sha256") != (rr.get("classifier") or {}).get(
                    "npz_sha256") or (d.get("stage2") or {}).get("classifier_json_sha256") != (
                    rr.get("classifier") or {}).get("json_sha256"):
                raise E3Refused("%s was scored with another classifier than the read's" % p)
            stamp_of(d, p)
            arm_rows.append(d)
            s1.append((d.get("stage1") or {}).get("agnostic_map50_95"))
        inputs.append({"seed": s, "weights_sha256": w, "test_sha256": _sha(p) if d else None})
        rp = C.INC_DIR / REFERENCE / "runs" / ("final__base__s%d" % s) / "scores" / "test.json"
        r = _read_json(rp)
        if r is None:
            missing.append("%s final__base__s%d" % (REFERENCE, s))
            continue
        if r.get("exam") != "test" or (r.get("production") is not True and not testing_ok):
            raise E3Refused("%s is not a production test score" % rp)
        fj = _read_json(C.INC_DIR / REFERENCE / "runs" / ("final__base__s%d" % s) / "run.json") or {}
        if r.get("weights_sha256") != fj.get("weights_sha256"):
            raise E3Refused("%s names weights other than its final run's" % rp)
        stamp_of(r, rp)
        ref_rows.append(r)

    def agg(rows_):
        return {"twelve": B._e2_mean([x.get("species_map50_95") for x in rows_]),
                "agnostic": B._e2_mean([x.get("agnostic_map50_95") for x in rows_])}
    a, r = agg(arm_rows), agg(ref_rows)
    am, rm = a["twelve"]["mean"], r["twelve"]["mean"]
    rep = {"format": FMT_TEST_REPORT, "status": "pending" if missing else "complete", "arm": arm,
           "verdict_sha256": _sha(vp), "chosen": v.get("chosen"), "headline": arm == v.get("chosen"),
           "missing": missing, "arm_scores": a, "stage1_agnostic": B._e2_mean(s1), "inputs": inputs,
           "read_record_sha256": _sha(td / "e3_test_read.json"), "stamps": dict(stamps) or None,
           "reference": {"exp": REFERENCE, "scores": r},
           "d_test": {"twelve": (am - rm) if None not in (am, rm) else None,
                      "agnostic": (a["agnostic"]["mean"] - r["agnostic"]["mean"])
                      if None not in (a["agnostic"]["mean"], r["agnostic"]["mean"]) else None},
           "target_test_map50_95": B.TARGET_TEST,
           "gap_to_target": {"arm": (B.TARGET_TEST - am) if am is not None else None,
                             "reference": (B.TARGET_TEST - rm) if rm is not None else None},
           "testing_allowed": bool(testing_ok), "written_utc": _utc(),
           "note": "the one read of a qualifying arm's sealed test, for people; never the platform's evidence; E3's "
                   "headline test number is the chosen arm's"}
    d = capacity_dir(out_dir)
    _write_json(d / ("e3_test_%s.json" % arm), rep)

    def f(x):
        return "-" if x is None else "%.4f" % x
    md = ["# E3-%s: the sealed test (one read, after the dev verdict)" % arm, "",
          "| | 12-class mean +- sd | agnostic mean +- sd | n |", "|---|---|---|---|",
          "| E3-%s | %s +- %s | %s +- %s | %d |" % (arm, f(am), f(a["twelve"]["sd"]), f(a["agnostic"]["mean"]),
                                                 f(a["agnostic"]["sd"]), a["twelve"]["n"]),
          "| %s | %s +- %s | %s +- %s | %d |" % (REFERENCE, f(rm), f(r["twelve"]["sd"]), f(r["agnostic"]["mean"]),
                                               f(r["agnostic"]["sd"]), r["twelve"]["n"]), "",
          "Gap to %.2f: E3-%s %s, %s %s. Status: %s%s." % (B.TARGET_TEST, arm, f(rep["gap_to_target"]["arm"]),
                                                         REFERENCE, f(rep["gap_to_target"]["reference"]),
                                                         rep["status"], " (missing %s)" % ", ".join(missing)
                                                         if missing else ""), "",
          ("E3-%s is the verdict's choice: this is E3's headline test number." % arm) if arm == v.get("chosen") else
          ("E3-%s qualified, but the verdict chose E3-%s: E3's headline test number is E3-%s's."
           % (arm, v.get("chosen"), v.get("chosen")))]
    p = d / ("e3_test_%s.md" % arm)
    tmp = p.with_name(".%s.tmp" % p.name)
    tmp.write_text("\n".join(md) + "\n")
    os.replace(tmp, p)
    log("E3-%s test: %s (12-class %s vs %s; gap to %.2f %s)" % (arm, rep["status"], f(am), f(rm), B.TARGET_TEST,
                                                                f(rep["gap_to_target"]["arm"])))
    return rep


# ------------------------------------------------------------------ the optional sensitivity (reported)
def sensitivity(embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, folds=None):
    """After the verdict: the cross-validation fold classifiers at the chosen C applied to seed 0 of each arm on
    dev; their spread is reported (sensitivity.json), never deciding."""
    _vp, v = _verdict_doc(testing_ok=_testing_ok())
    out = root() / "sensitivity.json"
    if out.exists():
        raise E3Refused("%s exists: written once" % out)
    _W, _b, _c, _p, crec, shas = load_classifier()
    nf = int((crec.get("cv") or {}).get("folds") or 0)
    folds = list(range(nf)) if folds is None else list(folds)
    if embedder is None:
        embedder = V.BioclipEmbedder()
    res = {}
    with _workers(workers) as wk:
        for arm in ARM_ORDER:
            src = _source(arm, SEEDS[0], testing_ok=_testing_ok())
            vals = []
            for f in folds:
                ccfg, _r, _s = _clf_cfg(fold=f)
                cfg = dict(ccfg, mode="two_stage", nms=NMS[arm], embedder=embedder, workers=wk, top_k=TOP_K)
                ps = _score_pass(src["weights"], EXAM, cfg, lock_check=lock_check, batch=batch, device=device)
                vals.append(ps["result"].get("species_map50_95"))
            m, sd = B._mean_sd([x for x in vals if x is not None])
            full = ((v.get("arms") or {}).get(arm) or {}).get("dev") or [None]
            res[arm] = {"seed": SEEDS[0], "folds": folds, "values": vals, "mean": m, "sd": sd,
                        "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                        "full_classifier": full[0]}
    rec = {"format": FMT_SENSITIVITY, "exam": EXAM, "classifier": shas, "arms": res, "created_utc": _utc(),
           "note": "reported only: the fold classifiers at the chosen C on seed 0; never deciding"}
    _write_once(out, rec)
    return rec


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="E3: two-stage species detection (one-class boxes + a BioCLIP-2 crop "
                                             "classifier), dev decides.")
    ap.add_argument("command", choices=("fit-classifier", "dev-gt", "equivalence", "score-arm", "verdict",
                                        "test-read", "score-test", "test-report", "restore-classifier",
                                        "sensitivity"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seeds", default=None, help="score-arm: comma-separated, a subset of 0,1,2")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--from", dest="from_dir", default=None, help="restore-classifier: a moved-aside E3 directory")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.twostage] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    kw = {"lock_check": not a.no_lock_check_for_tests, "batch": a.batch or S.BATCH, "device": a.device}
    needs_arm = a.command in ("score-arm", "test-read", "score-test", "test-report")
    try:
        if needs_arm and a.arm not in ARMS:
            raise E3Refused("%s needs --arm M, A or B" % a.command)
        if not needs_arm and a.arm is not None:
            raise E3Refused("%s takes no --arm" % a.command)
        if a.command == "fit-classifier":
            fit_classifier()
        elif a.command == "dev-gt":
            dev_gt_accuracy(**kw)
        elif a.command == "equivalence":
            equivalence(**kw)
        elif a.command == "score-arm":
            seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else list(SEEDS)
            score_arm(a.arm, seeds=seeds, **kw)
        elif a.command == "verdict":
            rescore_verdict(out_dir=a.out_dir)
        elif a.command == "test-read":
            print(json.dumps(test_read(a.arm)["argv"]))
        elif a.command == "score-test":
            score_test(a.arm, **kw)
        elif a.command == "test-report":
            test_report(a.arm, out_dir=a.out_dir)
        elif a.command == "restore-classifier":
            if not a.from_dir:
                raise E3Refused("restore-classifier needs --from DIR")
            restore_classifier(a.from_dir)
        else:
            sensitivity(**kw)
    except (E3Refused, B.BaselineError, T.RunError, SN.NativeRefused, S.ScorerRefused, SC.SidecarError,
            V.VerifyError, C2.Inc2Error, B3.Base3Error, ValueError) as e:
        print("[inc2.twostage] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
