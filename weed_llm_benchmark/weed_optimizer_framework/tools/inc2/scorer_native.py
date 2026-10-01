"""Native-resolution scores of the measurement arms (docs/CONTINUOUS_LOOP.md,
group B, "Amendment (2026-10-01): the measurement arms read at their own
resolution (pre-registered)").

    python -m weed_optimizer_framework.tools.inc2.scorer_native --exp E --run final__base__s0 --exam dev
        [--batch N --device D --no-lock-check-for-tests]   (test mode only)

Why. The locked scorer (inc/scorer.py, pinned: its sha256 is in LOCK.json)
infers at 640 px and refuses any other size outside test mode. The
measurement arms (inc2.recipes.MEASURE_ARMS: m832, s1024) train at 832 and
1024 px; whether the small prostrate weeds gain from more pixels at
inference needs a score at the arm's own size. This module is that second,
separately named scorer. It never edits or replaces the locked one. The
box-quality arms (l640, y26m640, y26l640, 2026-10-01) train at 640 px: they
are read here at 640 on dev only, as the reference arm is, so the verdict
has their per-image arrays; their score must reproduce the run's recorded
protocol score.

What it reuses, unchanged (the locked scorer as a library): check_lock (the
exam manifest and scorer.py against LOCK.json), check_exam (every image and
label of the materialised exam against its manifest), load_model and
_check_model (the INC class space, no foreign modules), build_view (the
temporary exam layout Ultralytics validates), the validator class with its
per-image hook, _device and _precision_kwargs (fp16 on a CUDA device), the
settings CONF 0.001, IOU 0.7, BATCH 32 (rect batching), Ultralytics' default
max_det, PINNED_ULTRALYTICS, and the metric definitions (per_class over the
classes with a GT box, species_map50_95 over cwd12 ids 0-11, the agnostic
AP, image_correct in key order). The validator is the scorer sidecar's
subclass of the scorer's own (inc2.scorer_sidecar.sidecar_validator_class),
so the per-image detections are captured in the same pass; they must
reproduce the score's per_class exactly (scorer_sidecar.check_capture).
What changes is imgsz, and only imgsz: every other departure from the
protocol (no LOCK check, another batch, fp32, another Ultralytics) is refused
outside test mode (INC_SCORER_TESTING=1), as the locked scorer refuses it.

What it refuses (NativeRefused, exit 2), before anything is written:
  * the exam test, at every size (P10: test is read only at milestones, and
    nothing scores test at a size other than 640); any exam but dev and
    imageweeds;
  * a size other than an allowed one: a measurement arm's training imgsz,
    or 640 for the reference arm (m640, REFERENCE_ARM), which is read on
    dev only (REFERENCE_EXAMS): no exam but dev is scored at 640;
  * (score_run) an experiment that is not a baseline, an arm that is
    neither a measurement arm nor the reference arm, a size other than the
    arm's own (the reference arm: 640), the reference arm on an exam other
    than dev, a run that is not one of the experiment's final runs
    (final__base__s<k>) or is not done, weights that do not hash as its
    run.json records, an exam outside its final exams;
  * an output path that is not scores/<exam>@<imgsz>.json, or one that
    already exists: a native score is written once, and a run's 640 px
    scores (scores/<exam>.json) are never touched;
  * at 640 (the reference), a score that does not reproduce the run's
    recorded protocol score within MAX_REFERENCE_DIFF (the sidecar's
    tolerance), when that score was taken with the same settings; in
    production a recorded score taken with other settings refuses too.

Outputs, beside the run's other scores, committed under an exclusive lock
file (scores/.<exam>@<imgsz>.commit, held for the two renames only): the npz
first, renamed into place only while no JSON is there; the JSON last,
hard-linked into place only if absent. So a JSON always names the npz beside
it, and a second writer of the same score is refused without replacing
either file (its npz is discarded):
  scores/<exam>@<imgsz>.json         format inc2-native-score/1: the locked
                                     scorer's result fields (metrics, stamps,
                                     settings) at this imgsz; production
                                     false (never a protocol score);
                                     native_production (every setting but
                                     imgsz the protocol's, the LOCK checked);
                                     scorer_sha256 stamped NATIVE<imgsz>- (and
                                     TEST- in test mode), so no gate, milestone
                                     or capacity decision can take it for a
                                     protocol score; locked_scorer_sha256 and
                                     scorer_native_sha256; the npz's sha256;
  scores/<exam>@<imgsz>.images.npz   the per-image arrays in exam key order
                                     (keys, pred_img, tp, conf, pred_cls,
                                     target_img, target_cls), as the sidecar
                                     writes them.
Exit codes: 0 written, 2 refused, 1 any other error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from . import recipes as RC
from . import scorer_sidecar as SC

FORMAT = "inc2-native-score/1"
PROTOCOL = "inc2-native-imgsz/1"
EXAMS = ("dev", "imageweeds")               # P10: never test
REFUSED_EXAMS = ("test",)
REFERENCE_ARM = "m640"                      # the comparison arm, scored at the protocol's 640 px
REFERENCE_EXAMS = ("dev",)                  # what the reference arm (and anything at 640) is read on
STAMP_PREFIX = "NATIVE%d-"                  # before the locked scorer's sha256 (and after TEST- in test mode)
NPZ_SUFFIX = ".images.npz"
COMMIT_SUFFIX = ".commit"                   # scores/.<exam>@<imgsz>.commit: held while a score is put in place
COMMIT_WAIT_S = 60.0                        # how long a writer waits for another's commit (two renames)
MAX_REFERENCE_DIFF = SC.MAX_SCORE_DIFF      # the reference at 640 against its recorded protocol score
FINAL_RUN_RE = re.compile(r"final__base__s(\d{1,2})\Z")
COMPARED_SETTINGS = ("imgsz", "batch", "conf", "iou", "half")


class NativeRefused(RuntimeError):
    """What was asked is not a pre-registered native-resolution score."""


def log(msg):
    print("[inc2.scorer_native] %s" % msg, flush=True)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_once(path, obj):
    """Write path atomically, only if it does not exist: a temp file hard-linked
    into place, so two writers can never replace each other's score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise NativeRefused("%s appeared while this score was taken: a native score is written once and never "
                                "overwritten" % path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _commit(js, npz, staged, obj):
    """Put one native score in place: its npz (written at `staged`, renamed
    to npz) and then its JSON (obj, hard-linked only if absent), under the
    score's lock file (scores/.<exam>@<imgsz>.commit, created exclusively and
    held for the two renames only). An npz is replaced only while no JSON
    names it (a leftover of an attempt that wrote no JSON), so a JSON always
    names the npz beside it; a second writer of the same score is refused and
    replaces neither file. `staged` is removed whatever happens."""
    js, npz, staged = Path(js), Path(npz), Path(staged)
    lock = js.with_name(".%s%s" % (js.stem, COMMIT_SUFFIX))
    try:
        deadline = time.time() + COMMIT_WAIT_S
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise NativeRefused("%s is held: another native pass of this score is putting it in place (or "
                                        "one died doing so; a person removes the file)" % lock)
                time.sleep(0.2)
        try:
            os.write(fd, ("pid %d %s\n" % (os.getpid(), _utc())).encode("ascii"))
            if js.exists() or js.is_symlink():
                raise NativeRefused("%s appeared while this score was taken: a native score is written once and "
                                    "never overwritten" % js)
            os.replace(staged, npz)
            _write_once(js, obj)
        finally:
            os.close(fd)
            os.unlink(str(lock))
    finally:
        if staged.exists():
            staged.unlink()


# ------------------------------------------------------------------ rules
def check_exam(exam):
    """Refuse test (at every size) and any exam but dev and imageweeds."""
    if exam in REFUSED_EXAMS:
        raise NativeRefused("the exam %r is never scored by the native scorer: test is read only at milestones, by "
                            "the locked scorer at 640 px (P10)" % exam)
    if exam not in EXAMS:
        raise NativeRefused("exam %r: the native scorer scores only %s" % (exam, list(EXAMS)))
    return exam


def native_imgsz(arm):
    """The size an arm is read at: a measurement arm's training imgsz, or
    640 for the reference arm. Any other arm is refused: the grid's arms are
    read by the locked scorer alone."""
    aid = RC.arm_id(arm)
    if aid in RC.MEASURE_ARMS:
        return int(RC.ARMS[aid]["imgsz"])
    if aid == REFERENCE_ARM:
        return int(S.IMGSZ)
    raise NativeRefused("arm %s is neither a measurement arm %s nor the reference arm %s: the native scorer reads "
                        "nothing else" % (aid, list(RC.MEASURE_ARMS), REFERENCE_ARM))


def allowed_sizes():
    return sorted({native_imgsz(a) for a in RC.MEASURE_ARMS} | {native_imgsz(REFERENCE_ARM)})


def score_name(exam, imgsz):
    """scores/<exam>@<imgsz>.json: never a protocol score's name (<exam>.json)."""
    return "%s@%d.json" % (exam, int(imgsz))


def paths_for(scores_dir, exam, imgsz):
    """(json, npz) of a native score in a run's scores directory."""
    d = Path(scores_dir)
    return d / score_name(exam, imgsz), d / ("%s@%d%s" % (exam, int(imgsz), NPZ_SUFFIX))


def stamp(scorer_sha, imgsz, test):
    return "%s%s%s" % (S.TEST_PREFIX if test else "", STAMP_PREFIX % int(imgsz), scorer_sha)


def check_out(out_json, exam, imgsz):
    """(json path, npz path), refusing any other name than
    scores/<exam>@<imgsz>.json and an existing score."""
    out_json = Path(out_json).resolve()
    if out_json.name == "%s.json" % exam:
        raise NativeRefused("%s is the run's protocol score: the native scorer never writes it" % out_json)
    if out_json.name != score_name(exam, imgsz) or out_json.parent.name != "scores":
        raise NativeRefused("a native score is written as scores/%s, not %s" % (score_name(exam, imgsz), out_json))
    js, npz = paths_for(out_json.parent, exam, imgsz)
    if js.exists() or js.is_symlink():
        raise NativeRefused("%s already exists: a native score is written once and never overwritten" % js)
    return js, npz


def compare_with_protocol(native, recorded, testing):
    """At 640 (the reference arm): the native score against the run's
    recorded protocol score. Returns the record; raises NativeRefused when
    the two differ beyond MAX_REFERENCE_DIFF, or (production) when the
    recorded score was not taken with the protocol's settings."""
    if not isinstance(recorded, dict):
        if testing:
            return {"compared": False, "why": "no recorded protocol score"}
        raise NativeRefused("the run has no recorded protocol score to reproduce at 640 px")
    rs, ns = recorded.get("settings") or {}, native.get("settings") or {}
    same = [k for k in COMPARED_SETTINGS if rs.get(k) != ns.get(k)]
    if same:
        if testing:
            return {"compared": False, "why": "settings differ (%s)" % ", ".join(same)}
        raise NativeRefused("the run's recorded score was not taken with these settings (%s): it cannot vouch for "
                            "the reference at 640 px" % ", ".join(same))
    for k in ("exam", "manifest_sha256", "key_order_sha256", "n_images", "weights_sha256", "n_gt"):
        if recorded.get(k) != native.get(k):
            raise NativeRefused("the native score at 640 px has %s %r, the recorded protocol score %r"
                                % (k, str(native.get(k))[:40], str(recorded.get(k))[:40]))
    rsha = str(recorded.get("scorer_sha256") or "")
    if (rsha[len(S.TEST_PREFIX):] if rsha.startswith(S.TEST_PREFIX) else rsha) != native.get("locked_scorer_sha256"):
        raise NativeRefused("the recorded protocol score was taken by another scorer (%s) than the locked one (%s)"
                            % (rsha[:16], str(native.get("locked_scorer_sha256"))[:12]))
    if set(recorded.get("per_class") or {}) != set(native.get("per_class") or {}):
        raise NativeRefused("per_class covers other classes in the native score than in the recorded one")
    diffs = [abs(float(recorded["map50_95"]) - float(native["map50_95"]))]
    diffs += [abs(float(recorded["per_class"][c]) - float(native["per_class"][c])) for c in recorded["per_class"]]
    worst = max(diffs)
    if not math.isfinite(worst) or worst > MAX_REFERENCE_DIFF:
        raise NativeRefused("the native score at 640 px differs from the run's recorded protocol score by %.6f (> %g): "
                            "it is not the locked scorer's reading" % (worst, MAX_REFERENCE_DIFF))
    return {"compared": True, "max_abs_diff": worst, "tolerance": MAX_REFERENCE_DIFF}


# ------------------------------------------------------------------ scoring
def score(weights, exam, out_json, imgsz, lock_check=True, batch=S.BATCH, device=None, recorded=None,
          extra=None):
    """Score weights on exam at imgsz with the locked scorer's code and
    settings (module docstring); write scores/<exam>@<imgsz>.json and its
    npz. `recorded`: the run's protocol score (dict), checked at 640.
    Returns the written record. Raises NativeRefused."""
    t0 = time.time()
    check_exam(exam)
    imgsz = int(imgsz)
    if imgsz not in allowed_sizes():
        raise NativeRefused("imgsz %d is not a pre-registered native size %s" % (imgsz, allowed_sizes()))
    if imgsz == int(S.IMGSZ) and exam not in REFERENCE_EXAMS:
        raise NativeRefused("at %d px the native scorer reads only the reference arm, on %s: a %s score at 640 px is "
                            "the locked scorer's" % (imgsz, list(REFERENCE_EXAMS), exam))
    js, npz = check_out(out_json, exam, imgsz)
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise NativeRefused("weights not found: %s" % weights)

    import torch
    import ultralytics
    device, half = S._device(device)
    # every departure but imgsz (the locked scorer's own list, with imgsz at the protocol's)
    other = S.deviations(lock_check, S.IMGSZ, batch, half, ultralytics.__version__, device)
    if other and not S.testing():
        raise NativeRefused("not a native score (%s); only tests score that way, with %s=1"
                            % ("; ".join(other), S.TEST_ENV))
    devs = S.deviations(lock_check, imgsz, batch, half, ultralytics.__version__, device)
    try:
        weights_sha = C.sha256_file(weights)
        manifest_sha, scorer_sha = S.check_lock(exam, lock_check)
        rows = sorted(C.read_manifest(C.manifest_path(exam)), key=lambda r: r["key"])
        if not rows:
            raise NativeRefused("the %s manifest lists no images" % exam)
        files, labels = S.check_exam(exam, rows)
    except S.ScorerRefused as e:
        raise NativeRefused("the locked scorer's checks refuse: %s" % e)
    try:
        model = S.load_model(weights)
    except Exception as e:                      # noqa: BLE001 - an unloadable checkpoint scores nothing
        raise NativeRefused("%s cannot be loaded as a detector (%s: %s)" % (weights, type(e).__name__, str(e)[:200]))
    try:
        S._check_model(model, weights)
    except S.ScorerRefused as e:
        raise NativeRefused("the locked scorer's checks refuse: %s" % e)
    made = []
    cls = SC.sidecar_validator_class()

    def make_validator(args=None, _callbacks=None):
        made.append(cls(args=args, _callbacks=_callbacks))
        return made[-1]

    with tempfile.TemporaryDirectory(prefix="inc2_native_") as tmp:
        try:
            view_yaml, copied = S.build_view(Path(tmp) / "view", exam, rows, files, labels)
        except S.ScorerRefused as e:
            raise NativeRefused("the locked scorer's checks refuse: %s" % e)
        model.val(validator=make_validator, data=str(view_yaml), imgsz=imgsz, batch=batch, conf=S.CONF, iou=S.IOU,
                  device=device, plots=False, save_json=False, save_txt=False, verbose=False,
                  project=str(Path(tmp) / "runs"), name="val", exist_ok=True, **S._precision_kwargs(half))
    v = made[-1]
    del SC._CAPTURE[:]                          # the sidecar's capture list: this pass keeps its own validator
    if bool(v.args.half) != half:
        raise RuntimeError("asked Ultralytics for half=%s on %s and it validated with half=%s"
                           % (half, device, v.args.half))
    if int(v.args.imgsz) != imgsz:
        raise NativeRefused("asked Ultralytics for imgsz %d and it validated at %s" % (imgsz, v.args.imgsz))
    metrics, inc = v.metrics, v.inc_results()

    keys = [r["key"] for r in rows]
    if set(inc["bits"]) != set(keys):
        raise NativeRefused("Ultralytics scored %d of the %d exam images (corrupt image or label?)"
                            % (len(inc["bits"]), len(keys)))
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

    # the per-image arrays: they must reproduce this pass's per_class exactly (the sidecar's check)
    images = dict(v.sc_images)
    try:
        recompute, restricted = SC.check_capture(per_class, images)
        arrays = SC.flatten(images, keys)
    except SC.SidecarError as e:
        raise NativeRefused("the captured per-image arrays do not reproduce the score: %s" % e)
    test = bool(other)
    result = {
        "format": FORMAT, "protocol": PROTOCOL, "exam": exam, "exam_dir": str(C.EXAMS_DIR / exam),
        "imgsz": imgsz, "protocol_imgsz": int(S.IMGSZ),
        "manifest_sha256": manifest_sha,
        "scorer_sha256": stamp(scorer_sha, imgsz, test),
        "locked_scorer_sha256": scorer_sha,
        "scorer_native_sha256": C.sha256_file(Path(__file__).resolve()),
        "scorer_sidecar_sha256": C.sha256_file(Path(SC.__file__).resolve()),
        "production": False,
        "native_production": not other,
        "deviations": devs, "other_deviations": other,
        "lock_checked": bool(lock_check),
        "weights": str(weights), "weights_sha256": weights_sha,
        "map50_95": float(box.map), "map50": float(box.map50),
        "per_class": per_class, "per_class_ap50": per_class_ap50, "n_gt": n_gt,
        "species_map50_95": (sum(per_class[n] for n in species) / len(species)) if species else None,
        "species_map50": (sum(per_class_ap50[n] for n in species) / len(species)) if species else None,
        "agnostic_map50_95": inc["agnostic_map50_95"], "agnostic_map50": inc["agnostic_map50"],
        "image_correct": "".join(str(inc["bits"][k]) for k in keys),
        "key_order_sha256": C.sha256_text("\n".join(keys)),
        "n_images_correct": sum(inc["bits"].values()), "n_images": len(keys), "n_boxes": n_boxes,
        "labels_checked": len(labels), "images_checked": len(rows), "jpegs_copied_not_linked": len(copied),
        "settings": {"imgsz": imgsz, "batch": batch, "conf": S.CONF, "iou": S.IOU, "half": bool(v.args.half),
                     "rect": bool(v.args.rect), "max_det": int(v.args.max_det),
                     "image_correct_conf": S.CORRECT_CONF, "image_correct_iou": S.CORRECT_IOU},
        "consistency": {"full_recompute_max_abs_diff": recompute, "species_restricted_max_abs_diff": restricted},
        "device": str(v.device), "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
    }
    if imgsz == S.IMGSZ:
        result["vs_protocol_score"] = compare_with_protocol(result, recorded, testing=test)
    if extra:
        result.update(extra)
    staged = npz.with_name(".%s.%d.%d.staged" % (npz.name, os.getpid(), time.monotonic_ns()))
    try:
        npz_sha = SC.save_npz(staged, arrays)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise
    result["images"] = {"path": str(npz), "sha256": npz_sha, "n_images": len(keys),
                        "n_predictions": int(len(arrays["conf"])), "n_targets": int(len(arrays["target_cls"]))}
    result["seconds"] = round(time.time() - t0, 3)
    result["created_utc"] = _utc()
    _commit(js, npz, staged, result)
    result["out"] = str(js)
    log("%s on %s at %d px: species mAP50-95 %.4f (%s), %.0fs -> %s"
        % (weights.name, exam, imgsz, result["species_map50_95"] or 0.0,
           "native protocol" if not other else "TEST: %s" % "; ".join(other), result["seconds"], js))
    return result


# ------------------------------------------------------------------ runs
def run_dir(exp, run_id):
    return C.INC_DIR / exp / "runs" / run_id


def load_arm(exp):
    """(exp.json, arm id, native imgsz) of a baseline experiment whose arm
    the native scorer reads; refuses anything else."""
    defn = _read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(defn, dict) or defn.get("type") != "baseline":
        raise NativeRefused("%s is not a baseline experiment: native scores are taken of the measurement arms' "
                            "finals (and the reference's) only" % exp)
    if not isinstance(defn.get("arm"), dict):
        raise NativeRefused("%s pins no arm" % exp)
    try:
        aid = RC.arm_id(defn["arm"])
    except RC.RecipeError as e:
        raise NativeRefused(str(e))
    return defn, aid, native_imgsz(aid)


def final_run(exp, run_id, defn):
    """(run.json, weights path) of a done final run of the experiment."""
    m = FINAL_RUN_RE.match(str(run_id))
    if not m or int(m.group(1)) not in [int(s) for s in defn.get("seeds") or []]:
        raise NativeRefused("%s is not one of %s's final runs (final__base__s<k>, k in %s)"
                            % (run_id, exp, defn.get("seeds")))
    rd = run_dir(exp, run_id)
    rj = _read_json(rd / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise NativeRefused("%s of %s is not done" % (run_id, exp))
    w = rd / "weights" / "final.pt"
    if not w.is_file():
        raise NativeRefused("%s has no weights/final.pt" % rd)
    if C.sha256_file(w) != rj.get("weights_sha256"):
        raise NativeRefused("%s does not hash to the weights_sha256 run.json records" % w)
    return rj, w


def testing_settings(defn):
    """A testing experiment's scoring settings (exp.json "testing": batch,
    device, lock_check), which need INC_SCORER_TESTING=1; {} otherwise."""
    t = defn.get("testing")
    if not t:
        return {}
    if not S.testing():
        raise NativeRefused("%s is a testing experiment: it is scored only with %s=1" % (defn.get("exp"), S.TEST_ENV))
    t = t if isinstance(t, dict) else {}
    return {"batch": t.get("batch") or S.BATCH, "device": t.get("device") or "cpu",
            "lock_check": t.get("lock_check") is not False}


def score_run(exp, run_id, exam, batch=None, device=None, lock_check=None):
    """Score one final run of a measurement arm at its training imgsz (or
    of the reference arm at 640) on exam; returns the written record."""
    check_exam(exam)
    defn, aid, imgsz = load_arm(exp)
    if aid == REFERENCE_ARM and exam not in REFERENCE_EXAMS:
        raise NativeRefused("the reference arm %s is read on %s only (pre-registered); its %s score is the locked "
                            "scorer's" % (aid, list(REFERENCE_EXAMS), exam))
    if exam not in (defn.get("final_exams") or []):
        raise NativeRefused("%s's final exams are %s, not %s" % (exp, defn.get("final_exams"), exam))
    rj, weights = final_run(exp, run_id, defn)
    ts = testing_settings(defn)
    batch = batch if batch is not None else ts.get("batch", S.BATCH)
    device = device if device is not None else ts.get("device")
    lock_check = lock_check if lock_check is not None else ts.get("lock_check", True)
    scores = run_dir(exp, run_id) / "scores"
    proto = scores / ("%s.json" % exam)
    before = C.sha256_file(proto) if proto.is_file() else None
    recorded = _read_json(proto) if proto.is_file() else None
    extra = {"exp": exp, "run_id": run_id, "arm": aid, "trained_imgsz": int(RC.ARMS[aid]["imgsz"]),
             "reference_arm": aid == REFERENCE_ARM,
             "protocol_score": {"file": proto.name, "sha256": before,
                                "species_map50_95": (recorded or {}).get("species_map50_95"),
                                "production": (recorded or {}).get("production")}}
    js, _npz = paths_for(scores, exam, imgsz)
    rec = score(weights, exam, js, imgsz, lock_check=lock_check, batch=batch, device=device, recorded=recorded,
                extra=extra)
    after = C.sha256_file(proto) if proto.is_file() else None
    if after != before:
        raise RuntimeError("%s changed while the native score was taken" % proto)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="A measurement arm's final run scored at its own imgsz (never test).")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True, help="a final run, final__base__s<k>")
    ap.add_argument("--exam", required=True, help="one of %s (test is refused)" % ", ".join(EXAMS))
    ap.add_argument("--batch", type=int, default=None, help="protocol: %d; anything else only with %s=1"
                                                            % (S.BATCH, S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        score_run(a.exp, a.run, a.exam, batch=a.batch, device=a.device,
                  lock_check=False if a.no_lock_check_for_tests else None)
    except NativeRefused as e:
        print("[inc2.scorer_native] REFUSED: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
