"""The locked scorer of the INC protocol, and the only producer of its metrics
(docs/INCREMENTAL_PROTOCOL.md, "Scorer").

    python -m weed_optimizer_framework.tools.inc.scorer --weights W --exam E --out OUT_JSON

Every accept/reject decision is made from numbers this file wrote, so it refuses
to score unless what it is about to measure is what was locked:
  (a) the exam's manifest hashes to its LOCK.json entry;
  (b) this file hashes to LOCK.json's scorer_sha256;
  (c) the materialised exam dir EXAMS_DIR/<exam> holds exactly the manifest's
      keys, and every label and every image matches its manifest sha256 (all
      of them: a sample would let an altered photo through for most weights);
  (d) the settings are the protocol's: imgsz 640, batch 32 (rect batching makes
      the letterbox shape depend on it), fp16 on a CUDA device, and Ultralytics
      PINNED_ULTRALYTICS. Every AP here is Ultralytics' ap_per_class, whose
      definition changed between 8.4.22 and 8.4.37 (on the same predictions
      AP50-95 0.145 against 0.131), so the release is pinned in this file,
      where LOCK.json's hash of it covers the pin: upgrading Ultralytics means
      editing the pin, relocking and re-scoring, never mixing the two;
  (e) the weights are a plain Ultralytics detector in the INC class space. A
      LoRA run's last.pt still holds ConvLoRA adapters, which Ultralytics' val
      cannot fuse; the file scored for a LoRA run is the merged checkpoint
      inc.lora writes beside it (last_merged.pt).
A refusal exits 2. The output path is cleared before anything else, so neither
a refusal nor a crash can leave an earlier run's score.json there to be
collected as this run's.

Tests set INC_SCORER_TESTING=1, which allows a score that departs from (a), (b)
or (d): --no-lock-check-for-tests, another imgsz or batch, CPU fp32, another
Ultralytics. Such a score says production=false, lists how it departs under
"deviations", and is stamped scorer_sha256 = "TEST-" + sha256, so it never
shares the stamp the gate compares with a protocol score. Without the variable
the same call is refused.

One Ultralytics validation pass (conf 0.001, iou 0.7) gives:
  - map50_95, map50 and per_class AP50-95 (per_class_ap50: AP50) over the
    classes that have a GT box, and n_gt for all 13: what a plain model.val
    reports on the same data with the same settings;
  - species_map50_95 / species_map50: the same average over the cwd12 ids 0-11
    only (OtherPlant left out), the protocol's species score;
  - agnostic_map50_95 / agnostic_map50: the same predictions and GT with every
    class collapsed into one, so that localisation and naming can be told
    apart. The validator's NMS is multi-label: a box above conf for several
    classes is emitted once per class, at identical coordinates. Collapsed,
    those copies are one detection, so only the most confident is kept;
    counted as extra detections they would make a model that hedges between
    two species on a well-placed box lose agnostic AP, and a label problem
    would be attributed to localisation;
  - image_correct: one bit per exam image in key order (key_order_sha256), 1
    when the predictions at conf >= 0.25, one per box as above (so each box
    carries its most confident class), and the GT boxes pair up one-to-one at
    IoU >= 0.5 with the right class (greedy by confidence), for the gate's flip
    count.

Ultralytics validates a temporary copy of the exam's layout (images symlinked to
the exam's, verified label bytes copied), for two reasons. It writes labels.cache
next to the labels dir and later trusts any cache whose key is only file sizes
and paths, so a relabelled exam of the same size would be scored with the old
labels. And it re-saves in place every JPEG that does not end in an EOI marker,
which through the exam's symlinks would rewrite the source photograph; those
JPEGs are copied into the temporary dir instead of linked.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import common as C

IMGSZ, BATCH, CONF, IOU = 640, 32, 0.001, 0.7
CORRECT_CONF, CORRECT_IOU = 0.25, 0.5
PINNED_ULTRALYTICS = "8.4.37"       # the cluster's release; see (d) in the module docstring
TEST_ENV = "INC_SCORER_TESTING"
TEST_PREFIX = "TEST-"
HASH_WORKERS = 8                    # exam images are hashed in parallel: Lustre reads dominate
# inc.lora.MERGED_NAME; not imported, because lora.py checks trainer internals at import
MERGED_LORA_NAME = "last_merged.pt"


class ScorerRefused(RuntimeError):
    """What was about to be measured is not what was locked."""


# ------------------------------------------------------------ metric helpers
def _iou_matrix(a, b):
    import numpy as np
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def one_per_box(xyxy, conf):
    """Indices (ascending) of the predictions left when rows with identical
    coordinates are reduced to their most confident one (the first of equal
    confidences). Exact for multi-label NMS output, which repeats a box once per
    class; boxes that merely overlap are left alone."""
    import numpy as np
    b = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    c = np.asarray(conf, dtype=float).reshape(-1)
    if len(c) != len(b):
        raise ValueError("%d boxes but %d confidences" % (len(b), len(c)))
    if len(c) < 2:
        return np.arange(len(c))
    order = np.argsort(-c, kind="stable")
    first = np.unique(b[order], axis=0, return_index=True)[1]
    return np.sort(order[first])


def image_correct(gt_cls, gt_xyxy, pred_cls, pred_xyxy, pred_conf,
                  conf=CORRECT_CONF, iou=CORRECT_IOU):
    """1 if, keeping predictions with confidence >= conf, every GT box is matched
    by a prediction of its class at IoU >= iou and no prediction is left over;
    else 0. Predictions claim GT boxes in descending confidence, each taking the
    unmatched same-class GT box it overlaps most."""
    import numpy as np
    gc = np.rint(np.asarray(gt_cls, dtype=float).reshape(-1)).astype(int)
    gb = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    pf = np.asarray(pred_conf, dtype=float).reshape(-1)
    keep = pf >= conf
    pc = np.rint(np.asarray(pred_cls, dtype=float).reshape(-1)[keep]).astype(int)
    pb = np.asarray(pred_xyxy, dtype=float).reshape(-1, 4)[keep]
    pf = pf[keep]
    if len(pc) != len(gc):
        return 0
    if not len(pc):
        return 1
    ious = _iou_matrix(pb, gb)
    taken = np.zeros(len(gc), dtype=bool)
    for p in np.argsort(-pf, kind="stable"):
        ok = ~taken & (gc == pc[p]) & (ious[p] >= iou)
        if not ok.any():
            return 0
        taken[int(np.argmax(np.where(ok, ious[p], -1.0)))] = True
    return 1


def collapsed_ap(tp, conf, n_gt):
    """(AP50-95, AP50) of predictions whose class-collapsed match matrix is tp
    (N x 10, Ultralytics' IoU thresholds) against n_gt GT boxes, with
    Ultralytics' own AP computation."""
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class
    conf = np.asarray(conf, dtype=float).reshape(-1)
    # No GT, or a model that predicts nothing: AP is 0. A collapsed model must be
    # scored (and then rejected by the gate), not crash the scorer.
    if n_gt == 0 or len(conf) == 0:
        return 0.0, 0.0
    tp = np.asarray(tp, dtype=bool).reshape(len(conf), -1)
    ap = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))[5]
    if ap.shape != (1, tp.shape[1]):
        raise RuntimeError("ap_per_class returned AP of shape %s, expected (1, %d)"
                           % (ap.shape, tp.shape[1]))
    return float(ap.mean()), float(ap[:, 0].mean())


# ------------------------------------------------------------- validator hook
_VALIDATOR = None


def validator_class():
    """A DetectionValidator that, from the predictions it already scores, also
    keeps each image's class-collapsed match matrix and correctness bit."""
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import box_iou

    missing = [n for n in ("_process_batch", "match_predictions", "init_metrics")
               if not callable(getattr(DetectionValidator, n, None))]
    if missing:
        raise RuntimeError("ultralytics %s: DetectionValidator has no %s, which inc/scorer.py "
                           "hooks; port the scorer to this version"
                           % (ultralytics.__version__, missing))

    class IncValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.inc_reset()

        def inc_reset(self):
            self.inc_tp, self.inc_conf, self.inc_n_gt, self.inc_bits = [], [], 0, {}

        # Called once per image with the predictions after NMS and the GT, both
        # as xyxy in the letterboxed input frame (IoU is unchanged by letterboxing).
        def _process_batch(self, preds, batch):
            out = super()._process_batch(preds, batch)
            for d, keys in ((preds, ("bboxes", "conf", "cls")), (batch, ("bboxes", "cls", "im_file"))):
                if not all(k in d for k in keys):
                    raise RuntimeError("ultralytics %s: per-image dict has keys %s, the scorer "
                                       "needs %s" % (ultralytics.__version__, sorted(d), keys))
            pbox, gc = preds["bboxes"], batch["cls"]
            keep = one_per_box(pbox.float().cpu().numpy(), preds["conf"].float().cpu().numpy())
            idx = torch.as_tensor(keep, dtype=torch.long, device=pbox.device)
            pbox = pbox[idx]
            conf = preds["conf"][idx].float().cpu().numpy()
            if len(gc) and len(keep):
                iou = box_iou(batch["bboxes"], pbox)
                zeros = torch.zeros(len(keep), device=pbox.device)
                tp = self.match_predictions(zeros, torch.zeros_like(gc), iou).cpu().numpy()
            else:
                tp = np.zeros((len(keep), self.niou), dtype=bool)
            self.inc_tp.append(tp)
            self.inc_conf.append(conf)
            self.inc_n_gt += int(len(gc))
            key = Path(batch["im_file"]).stem
            if key in self.inc_bits:
                raise RuntimeError("exam image %s was scored twice" % key)
            self.inc_bits[key] = image_correct(
                gc.float().cpu().numpy(), batch["bboxes"].float().cpu().numpy(),
                preds["cls"][idx].float().cpu().numpy(), pbox.float().cpu().numpy(), conf)
            return out

        def inc_results(self):
            tp = (np.concatenate(self.inc_tp, 0) if self.inc_tp
                  else np.zeros((0, self.niou), dtype=bool))
            conf = np.concatenate(self.inc_conf, 0) if self.inc_conf else np.zeros(0)
            m, m50 = collapsed_ap(tp, conf, self.inc_n_gt)
            return {"agnostic_map50_95": m, "agnostic_map50": m50,
                    "n_gt": self.inc_n_gt, "bits": dict(self.inc_bits)}

    _VALIDATOR = IncValidator
    return IncValidator


# ------------------------------------------------------------------ checks
def _refuse(msg, items=()):
    items = list(items)
    if items:
        msg = "%s: %d, e.g. %s" % (msg, len(items), items[:3])
    raise ScorerRefused(msg)


def testing():
    return os.environ.get(TEST_ENV) == "1"


def deviations(lock_check, imgsz, batch, half, ultralytics_version, device=None):
    """Every way a score taken with these settings departs from a protocol
    score; [] for a protocol score."""
    out = []
    if not lock_check:
        out.append("LOCK.json not checked")
    if imgsz != IMGSZ:
        out.append("imgsz %s, not %d" % (imgsz, IMGSZ))
    if batch != BATCH:
        out.append("batch %s, not %d" % (batch, BATCH))
    if not half:
        out.append("fp32 on device %s, not fp16 on a CUDA device" % (device,))
    if ultralytics_version != PINNED_ULTRALYTICS:
        out.append("ultralytics %s, not the pinned %s" % (ultralytics_version, PINNED_ULTRALYTICS))
    return out


def check_lock(exam, lock_check=True):
    """(manifest sha256, scorer sha256), after comparing both with LOCK.json."""
    manifest = C.manifest_path(exam)
    if not manifest.is_file():
        _refuse("no manifest for exam %r at %s" % (exam, manifest))
    scorer_sha = C.sha256_file(Path(__file__).resolve())
    if not lock_check:
        return C.sha256_file(manifest), scorer_sha
    try:
        lock = C.read_lock()
        manifest_sha = C.verify_manifest_against_lock(exam, lock)
        want = lock["scorer_sha256"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        raise ScorerRefused("lock check failed for %r: %s: %s" % (exam, type(e).__name__, e))
    if scorer_sha != want:
        _refuse("scorer.py changed since it was locked (%s != %s)" % (scorer_sha[:12], str(want)[:12]))
    return manifest_sha, scorer_sha


def _hash_one(path):
    try:
        return C.sha256_file(path), None
    except OSError as e:
        return None, e


def check_exam(exam, rows):
    """Refuse unless EXAMS_DIR/<exam> is the manifest's exam. Returns the image
    file name of every key and the verified bytes of every label."""
    exam_dir = C.EXAMS_DIR / exam
    img_dir, lab_dir = exam_dir / "images", exam_dir / "labels"
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        _refuse("duplicate keys in the %s manifest" % exam)
    if not img_dir.is_dir() or not lab_dir.is_dir():
        _refuse("exam %r is not materialised at %s" % (exam, exam_dir))
    import yaml
    try:
        with open(exam_dir / "data.yaml") as fh:
            data = yaml.safe_load(fh)
        names = data["names"]
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        if int(data["nc"]) != C.NC or names != C.CLASS_NAMES:
            _refuse("%s/data.yaml has nc=%s names=%s, not the INC class space"
                    % (exam_dir, data["nc"], names))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        _refuse("cannot read %s/data.yaml: %s" % (exam_dir, e))

    files = {}
    for name in os.listdir(img_dir):
        stem = os.path.splitext(name)[0]
        if stem in files:
            _refuse("two images for key %s in %s: %s, %s" % (stem, img_dir, files[stem], name))
        files[stem] = name
    lab_names = os.listdir(lab_dir)
    lab_keys = {n[:-4] for n in lab_names if n.endswith(".txt")}
    want = set(keys)
    _check_same(img_dir, want, set(files))
    _check_same(lab_dir, want, lab_keys)
    stray = [n for n in lab_names if not n.endswith(".txt")]
    if stray:
        _refuse("files other than labels in %s" % lab_dir, sorted(stray))

    labels, bad = {}, []
    for r in rows:
        with open(lab_dir / (r["key"] + ".txt"), "rb") as fh:
            labels[r["key"]] = fh.read()
        if hashlib.sha256(labels[r["key"]]).hexdigest() != r["label_sha256"]:
            bad.append(r["key"])
    if bad:
        _refuse("exam labels that differ from the manifest", bad)

    with ThreadPoolExecutor(max_workers=max(1, min(HASH_WORKERS, len(rows)))) as ex:
        hashed = list(ex.map(_hash_one, [img_dir / files[r["key"]] for r in rows]))
    bad = [r["key"] if err is None else "%s (%s)" % (r["key"], err)
           for r, (h, err) in zip(rows, hashed) if h != r["sha256"]]
    if bad:
        _refuse("exam images that differ from the manifest", bad)
    return files, labels


def _check_same(where, want, got):
    if got != want:
        _refuse("%s does not hold exactly the manifest's keys: %d missing %s, %d extra %s"
                % (where, len(want - got), sorted(want - got)[:3],
                   len(got - want), sorted(got - want)[:3]))


# ------------------------------------------------------------------ running
def _jpeg_without_eoi(path):
    with open(path, "rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return False
        fh.seek(-2, os.SEEK_END)
        return fh.read(2) != b"\xff\xd9"


def build_view(view, exam, rows, files, labels):
    """The directory Ultralytics validates (see the module docstring). Returns
    (data.yaml path, keys of JPEGs copied rather than linked)."""
    src_dir = C.EXAMS_DIR / exam / "images"
    (view / "images").mkdir(parents=True)
    (view / "labels").mkdir()
    copied = []
    for r in rows:
        name = files[r["key"]]
        try:
            needs_copy = _jpeg_without_eoi(src_dir / name)
        except OSError as e:
            _refuse("cannot read exam image %s: %s" % (src_dir / name, e))
        if needs_copy:
            shutil.copyfile(src_dir / name, view / "images" / name)
            copied.append(r["key"])
        else:
            os.symlink(src_dir / name, view / "images" / name)
        with open(view / "labels" / (r["key"] + ".txt"), "wb") as fh:
            fh.write(labels[r["key"]])
    yaml_path = view / "data.yaml"
    with open(yaml_path, "w") as fh:
        fh.write("path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n" % (view, C.NC))
        for n in C.CLASS_NAMES:
            fh.write("  - %s\n" % n)
    return yaml_path, copied


def load_model(weights):
    """YOLO(weights). A checkpoint naming a module that cannot be imported makes
    Ultralytics pip-install a package of that name; a scorer never installs
    anything."""
    import ultralytics.utils.checks as checks
    from ultralytics import YOLO
    checks.AUTOINSTALL = False
    return YOLO(str(weights))


def _check_model(model, weights):
    names = model.names
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if model.task != "detect" or names != C.CLASS_NAMES:
        _refuse("%s is a %s model with classes %s, not the INC class space %s"
                % (weights, model.task, names[:4] + (["..."] if len(names) > 4 else []),
                   C.CLASS_NAMES[:4] + ["..."]))
    foreign = sorted({"%s.%s" % (type(m).__module__, type(m).__name__) for m in model.model.modules()
                      if not type(m).__module__.startswith(("torch.", "ultralytics."))})
    if foreign:
        _refuse("%s holds modules that are neither torch nor Ultralytics (%s). A LoRA run's "
                "last.pt keeps its ConvLoRA adapters, which val cannot fuse; score the merged "
                "checkpoint inc.lora writes beside it, %s"
                % (weights, ", ".join(foreign), MERGED_LORA_NAME))


def _device(device):
    import torch
    if device in (None, ""):
        device = "0" if torch.cuda.is_available() else "cpu"
    device = str(device)
    half = torch.cuda.is_available() and device.lower() not in ("cpu", "mps")
    return device, half


def _precision_kwargs(half):
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    if "half" in DEFAULT_CFG_DICT:
        return {"half": half}
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if half else None}
    raise RuntimeError("ultralytics %s has neither a 'half' nor a 'quantize' setting"
                       % ultralytics.__version__)


def resolve_out(out):
    """--out is a .json path, or a directory that receives score.json."""
    out = Path(out)
    if out.is_dir() or out.suffix.lower() != ".json":
        out = out / "score.json"
    return out.resolve()


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def score(weights, exam, out_json, lock_check=True, imgsz=IMGSZ, batch=BATCH, device=None):
    """Score weights on exam and write out_json; returns the written dict.
    Raises ScorerRefused when a check fails; out_json is removed first, so a
    refusal or an error leaves no score there."""
    t0 = time.time()
    out_json = resolve_out(out_json)
    if out_json.is_file() or out_json.is_symlink():
        out_json.unlink()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", str(exam)):
        _refuse("bad exam name %r" % (exam,))
    weights = Path(weights).resolve()
    if not weights.is_file():
        _refuse("weights not found: %s" % weights)

    import torch
    import ultralytics
    device, half = _device(device)
    devs = deviations(lock_check, imgsz, batch, half, ultralytics.__version__, device)
    if devs and not testing():
        _refuse("not a protocol score (%s); only tests score that way, with %s=1, and their "
                "scores are stamped %s" % ("; ".join(devs), TEST_ENV, TEST_PREFIX))

    weights_sha = C.sha256_file(weights)
    manifest_sha, scorer_sha = check_lock(exam, lock_check)
    rows = sorted(C.read_manifest(C.manifest_path(exam)), key=lambda r: r["key"])
    if not rows:
        _refuse("the %s manifest lists no images" % exam)
    files, labels = check_exam(exam, rows)

    model = load_model(weights)
    _check_model(model, weights)
    made = []

    def make_validator(args=None, _callbacks=None):
        made.append(validator_class()(args=args, _callbacks=_callbacks))
        return made[-1]

    with tempfile.TemporaryDirectory(prefix="inc_scorer_") as tmp:
        view_yaml, copied = build_view(Path(tmp) / "view", exam, rows, files, labels)
        model.val(validator=make_validator, data=str(view_yaml), imgsz=imgsz, batch=batch,
                  conf=CONF, iou=IOU, device=device, plots=False, save_json=False,
                  save_txt=False, verbose=False, project=str(Path(tmp) / "runs"), name="val",
                  exist_ok=True, **_precision_kwargs(half))
    v = made[-1]
    if bool(v.args.half) != half:
        raise RuntimeError("asked Ultralytics for half=%s on %s and it validated with half=%s"
                           % (half, device, v.args.half))
    metrics, inc = v.metrics, v.inc_results()

    keys = [r["key"] for r in rows]
    if set(inc["bits"]) != set(keys):
        _refuse("Ultralytics scored %d of the %d exam images (corrupt image or label?)"
                % (len(inc["bits"]), len(keys)), sorted(set(keys) - set(inc["bits"])))
    nt = metrics.nt_per_class
    n_boxes = int(nt.sum()) if nt is not None else 0
    if n_boxes != inc["n_gt"]:
        raise RuntimeError("GT box counts disagree: Ultralytics %d, collapsed %d" % (n_boxes, inc["n_gt"]))

    box = metrics.box
    per_class, per_class_ap50 = {}, {}
    for i, c in enumerate(box.ap_class_index):
        per_class[C.CLASS_NAMES[int(c)]] = float(box.ap[i])
        per_class_ap50[C.CLASS_NAMES[int(c)]] = float(box.ap50[i])
    n_gt = {n: (int(nt[i]) if nt is not None else 0) for i, n in enumerate(C.CLASS_NAMES)}
    species = [n for n in per_class if C.CLASS_NAMES.index(n) != C.OTHER_PLANT]

    result = {
        "exam": exam,
        "exam_dir": str(C.EXAMS_DIR / exam),
        "manifest_sha256": manifest_sha,
        "scorer_sha256": (TEST_PREFIX + scorer_sha) if devs else scorer_sha,
        "production": not devs,
        "deviations": devs,
        "lock_checked": bool(lock_check),
        "weights": str(weights),
        "weights_sha256": weights_sha,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "per_class": per_class,
        "per_class_ap50": per_class_ap50,
        "n_gt": n_gt,
        "species_map50_95": (sum(per_class[n] for n in species) / len(species)) if species else None,
        "species_map50": (sum(per_class_ap50[n] for n in species) / len(species)) if species else None,
        "agnostic_map50_95": inc["agnostic_map50_95"],
        "agnostic_map50": inc["agnostic_map50"],
        "image_correct": "".join(str(inc["bits"][k]) for k in keys),
        "key_order_sha256": C.sha256_text("\n".join(keys)),
        "n_images_correct": sum(inc["bits"].values()),
        "n_images": len(keys),
        "n_boxes": n_boxes,
        "labels_checked": len(labels),
        "images_checked": len(rows),
        "jpegs_copied_not_linked": len(copied),
        "settings": {"imgsz": imgsz, "batch": batch, "conf": CONF, "iou": IOU, "half": bool(v.args.half),
                     "rect": bool(v.args.rect), "max_det": int(v.args.max_det),
                     "image_correct_conf": CORRECT_CONF, "image_correct_iou": CORRECT_IOU},
        "protocol_settings": imgsz == IMGSZ and batch == BATCH and bool(v.args.half),
        "device": str(v.device),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "seconds": round(time.time() - t0, 3),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json(out_json, result)
    result["out"] = str(out_json)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="The INC scorer: the only producer of INC metrics.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--exam", required=True, help="split name with a locked manifest, e.g. dev")
    ap.add_argument("--out", required=True, help="score.json path, or a directory to hold score.json")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help="protocol: %d; anything else only with %s=1" % (IMGSZ, TEST_ENV))
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="protocol: %d; anything else only with %s=1" % (BATCH, TEST_ENV))
    ap.add_argument("--device", default=None,
                    help="default: GPU 0 if there is one (the protocol needs one), else cpu")
    ap.add_argument("--no-lock-check-for-tests", action="store_true",
                    help="skip the LOCK.json comparisons; only with %s=1, and the score is "
                         "stamped %s" % (TEST_ENV, TEST_PREFIX))
    a = ap.parse_args(argv)
    # before Ultralytics is imported: the scorer never installs or fetches anything
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        r = score(a.weights, a.exam, a.out, lock_check=not a.no_lock_check_for_tests,
                  imgsz=a.imgsz, batch=a.batch, device=a.device)
    except ScorerRefused as e:
        print("scorer: REFUSED: %s" % e, file=sys.stderr)
        return 2
    print("scorer: %s on %s: mAP50-95 %.4f, agnostic %.4f, %d/%d images correct, %.0fs -> %s%s"
          % (Path(a.weights).name, a.exam, r["map50_95"], r["agnostic_map50_95"],
             r["n_images_correct"], r["n_images"], r["seconds"], r["out"],
             "" if r["production"] else " (TEST score: %s)" % "; ".join(r["deviations"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
