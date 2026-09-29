"""The scorer sidecar: per-image, per-class AP inputs of one score, and the
image-bootstrap standard error of every species' dev AP that Protocol v3's
species guard reads (docs/CONTINUOUS_LOOP.md §2.6 L-3; inc2/gate3.py).

    python -m weed_optimizer_framework.tools.inc2.scorer_sidecar --weights W --exam dev
        --score RUN/scores/dev.json --out RUN/scores/dev.sidecar.json
        [--imgsz N --batch N --device D --no-lock-check-for-tests]   (test mode only)

Why a sidecar. The locked scorer (inc/scorer.py, pinned: its sha256 is in
LOCK.json) writes per-class AP over the whole exam and one correctness bit per
image, but not the per-image, per-class detections behind the AP, so an image
bootstrap of a species' AP cannot be computed from score.json. The scorer is
never edited. This module calls it, unchanged, as a library:
scorer.score(weights, exam, ...) runs every one of its checks (the LOCK, the
exam materialisation, the model, the protocol settings) and its one
Ultralytics validation pass. The only addition is a validator subclass of the
scorer's own (scorer.validator_class()), installed for this process through
the scorer's validator cache: it calls the scorer's per-image hook first and
then keeps, per exam image, what Ultralytics' DetectionValidator feeds its
metrics (DetMetrics.update_stats): tp (N x 10, class-aware matches at IoU
0.50:0.95), conf, pred_cls and target_cls. Their concatenation is exactly
DetMetrics.process's input, so ap_per_class on it gives the score's own
per_class AP; this is checked on every run (full_recompute).

The sidecar's own score is a second pass on the same weights; it must carry
the recorded score's stamps (exam, scorer_sha256, manifest_sha256,
key_order_sha256, n_images, n_gt, weights_sha256, production) and its
per_class AP within MAX_SCORE_DIFF of the recorded one (the same predictions
on the same device; the tolerance absorbs cuDNN non-determinism only).
Anything else refuses (exit 2), writing nothing.

The bootstrap (pre-registered, L-3): RESAMPLES = 1,000 resamples of the exam's
images with replacement, drawn once for all species with
numpy.random.default_rng(stable_int("inc2/v3/species_se")) as an array
[RESAMPLES, n_images] of image indices. For species s and resample b, every
detection of class s of an image drawn k times counts k times, and so do its
GT boxes; AP50-95 is Ultralytics' ap_per_class on those arrays (the scorer's
AP function). A resample without a GT box of s is skipped for s and counted
(n_valid). SE_s = the sample sd (ddof 1) of the n_valid values.

Outputs (atomic: temp file, then rename):
  <out>                      JSON, format inc2-scorer-sidecar/1: the recorded
                             score (path, sha256, stamps), the sidecar score's
                             metrics, the consistency checks, the npz (path,
                             sha256), species_se {seed_text, seed, resamples,
                             per_species {s: {ap, se, n_valid, n_gt, mean,
                             p2_5, p97_5}}}, the code (sha256 of this file and
                             of scorer.py), Ultralytics' version;
  <out minus .sidecar.json>.images.npz
                             keys, pred_img, tp, conf, pred_cls, target_img,
                             target_cls (flat arrays; *_img index keys).
Exit codes: 0 written, 2 refused (the scorer refused, or a consistency check
failed), 1 any other error.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S

FORMAT = "inc2-scorer-sidecar/1"
SEED_TEXT = "inc2/v3/species_se"
RESAMPLES = 1000
SPECIES = tuple(C.CLASS_NAMES[:C.OTHER_PLANT])
MAX_SCORE_DIFF = 0.002            # per_class / map50_95, sidecar pass vs the recorded score
MAX_RECOMPUTE_DIFF = 1e-9         # full-sample ap_per_class vs the sidecar score's own per_class
MAX_RESTRICTED_DIFF = 0.01        # per-species (class-restricted) full-sample AP vs the full call
JSON_SUFFIX = ".sidecar.json"
NPZ_SUFFIX = ".images.npz"
STAMPS = ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "n_images", "weights_sha256")


class SidecarError(RuntimeError):
    """The sidecar cannot vouch for what it would write."""


def log(msg):
    print("[inc2.scorer_sidecar] %s" % msg, flush=True)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def paths_for(out_json):
    """(json path, npz path) of a sidecar: <x>.sidecar.json and <x>.images.npz."""
    out_json = Path(out_json)
    name = out_json.name
    stem = name[:-len(JSON_SUFFIX)] if name.endswith(JSON_SUFFIX) else out_json.stem
    return out_json, out_json.with_name(stem + NPZ_SUFFIX)


def default_out(score_json):
    """scores/<exam>.json -> scores/<exam>.sidecar.json"""
    p = Path(score_json)
    return p.with_name(p.stem + JSON_SUFFIX)


def _write_json(path, obj):
    path = Path(path)
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


# ----------------------------------------------------------------- capture
_CAPTURE = []
NIOU = 10                          # Ultralytics' IoU thresholds 0.50:0.95


def _as_tp(tp, n_pred):
    """tp as a bool array [n_pred, NIOU] (an image without predictions gives
    a [0, NIOU] array whatever shape it arrived in)."""
    import numpy as np
    tp = np.asarray(tp, dtype=bool)
    if n_pred == 0:
        return np.zeros((0, tp.shape[1] if tp.ndim == 2 and tp.shape[1] else NIOU), dtype=bool)
    if tp.ndim != 2:
        tp = tp.reshape(n_pred, -1)
    if tp.shape[0] != n_pred:
        raise RuntimeError("tp has %d rows for %d predictions" % (tp.shape[0], n_pred))
    return tp


def sidecar_validator_class():
    """A subclass of the scorer's own validator that also keeps each image's
    metric inputs (module docstring)."""
    base = S.validator_class()

    class SidecarValidator(base):
        def inc_reset(self):
            super().inc_reset()
            self.sc_images = {}
            _CAPTURE.append(self)

        def _process_batch(self, preds, batch):
            out = super()._process_batch(preds, batch)
            key = Path(batch["im_file"]).stem
            if key in self.sc_images:
                raise RuntimeError("exam image %s was captured twice" % key)
            tp = _as_tp(out["tp"], len(preds["cls"]))
            self.sc_images[key] = {
                "tp": tp,
                "conf": preds["conf"].float().cpu().numpy().reshape(-1),
                "pred_cls": preds["cls"].float().cpu().numpy().reshape(-1),
                "target_cls": batch["cls"].float().cpu().numpy().reshape(-1)}
            return out

    return SidecarValidator


def capture_score(weights, exam, lock_check=True, imgsz=S.IMGSZ, batch=S.BATCH, device=None):
    """(the scorer's result dict, {key: per-image arrays}) from one
    scorer.score call with the capturing validator installed for it."""
    base = S.validator_class()
    cls = sidecar_validator_class()
    prev = S._VALIDATOR
    del _CAPTURE[:]
    with tempfile.TemporaryDirectory(prefix="inc2_sidecar_") as tmp:
        S._VALIDATOR = cls
        try:
            res = S.score(weights, exam, Path(tmp) / "score.json", lock_check=lock_check, imgsz=imgsz,
                          batch=batch, device=device)
        finally:
            S._VALIDATOR = prev if prev is not None else base
    if not _CAPTURE:
        raise SidecarError("the scorer ran no capturing validator")
    return res, dict(_CAPTURE[-1].sc_images)


# ------------------------------------------------------------------ arrays
def flatten(images, keys):
    """Flat arrays in key order: {keys, pred_img, tp, conf, pred_cls,
    target_img, target_cls}."""
    import numpy as np
    missing = [k for k in keys if k not in images]
    if missing or len(images) != len(keys):
        raise SidecarError("captured %d images for %d exam keys (missing %s)" % (len(images), len(keys),
                                                                                missing[:3]))
    tps, confs, pcls, pimg, tcls, timg = [], [], [], [], [], []
    niou = None
    for i, k in enumerate(keys):
        d = images[k]
        tp = _as_tp(d["tp"], len(np.asarray(d["conf"]).reshape(-1)))
        if niou is None and tp.shape[1]:
            niou = tp.shape[1]
        tps.append(tp)
        confs.append(np.asarray(d["conf"], dtype=np.float64).reshape(-1))
        pcls.append(np.asarray(d["pred_cls"], dtype=np.float64).reshape(-1))
        pimg.append(np.full(len(confs[-1]), i, dtype=np.int64))
        tcls.append(np.asarray(d["target_cls"], dtype=np.float64).reshape(-1))
        timg.append(np.full(len(tcls[-1]), i, dtype=np.int64))
    niou = niou or 10
    tps = [t if t.shape[1] == niou else np.zeros((len(t), niou), dtype=bool) for t in tps]
    out = {"keys": np.asarray(list(keys)),
           "tp": np.concatenate(tps, 0) if tps else np.zeros((0, niou), dtype=bool),
           "conf": np.concatenate(confs) if confs else np.zeros(0),
           "pred_cls": np.concatenate(pcls) if pcls else np.zeros(0),
           "pred_img": np.concatenate(pimg) if pimg else np.zeros(0, dtype=np.int64),
           "target_cls": np.concatenate(tcls) if tcls else np.zeros(0),
           "target_img": np.concatenate(timg) if timg else np.zeros(0, dtype=np.int64)}
    if not (len(out["tp"]) == len(out["conf"]) == len(out["pred_cls"]) == len(out["pred_img"])):
        raise SidecarError("prediction arrays of unequal length")
    return out


def save_npz(path, arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp.npz" % (path.name, os.getpid()))
    try:
        np.savez_compressed(str(tmp), **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


def load_npz(path):
    import numpy as np
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _ap_per_class():
    from ultralytics.utils.metrics import ap_per_class
    return ap_per_class


def full_per_class(arrays, ap_fn=None):
    """{class name: AP50-95} over the whole exam, computed exactly as
    Ultralytics' DetMetrics.process does (one ap_per_class call on the
    concatenated arrays)."""
    import numpy as np
    ap_fn = ap_fn or _ap_per_class()
    if not len(arrays["target_cls"]):
        return {}
    res = ap_fn(arrays["tp"], arrays["conf"], arrays["pred_cls"], arrays["target_cls"])
    ap, classes = res[5], res[6]
    return {C.CLASS_NAMES[int(c)]: float(np.asarray(ap[i]).mean()) for i, c in enumerate(classes)}


def species_arrays(arrays, s_id):
    """(tp, conf, pred_img, gt per image) restricted to class s_id."""
    import numpy as np
    n = len(arrays["keys"])
    sel = np.rint(arrays["pred_cls"]).astype(np.int64) == s_id
    gt = np.bincount(arrays["target_img"][np.rint(arrays["target_cls"]).astype(np.int64) == s_id],
                     minlength=n).astype(np.int64)
    return arrays["tp"][sel], arrays["conf"][sel], arrays["pred_img"][sel], gt


def class_ap(tp, conf, n_gt, s_id, ap_fn):
    """AP50-95 of one class from its detections and GT count, with
    ap_per_class (0.0 when there are GT boxes and no detection)."""
    import numpy as np
    if n_gt <= 0:
        return None
    if not len(conf):
        return 0.0
    res = ap_fn(tp, conf, np.full(len(conf), float(s_id)), np.full(int(n_gt), float(s_id)))
    ap = np.asarray(res[5])
    return float(ap[0].mean()) if ap.size else 0.0


def resample_indices(n_images, resamples=RESAMPLES, seed_text=SEED_TEXT):
    import numpy as np
    rng = np.random.default_rng(C.stable_int(seed_text))
    return rng.integers(0, n_images, size=(resamples, n_images))


def bootstrap_species_se(arrays, resamples=RESAMPLES, seed_text=SEED_TEXT, ap_fn=None, species=SPECIES):
    """{species: {ap, se, n_valid, n_gt, mean, p2_5, p97_5}} (module
    docstring). ap is the class-restricted full-sample AP."""
    import numpy as np
    ap_fn = ap_fn or _ap_per_class()
    n = len(arrays["keys"])
    if n < 2:
        raise SidecarError("an image bootstrap needs at least 2 exam images, got %d" % n)
    idx = resample_indices(n, resamples, seed_text)
    counts = np.stack([np.bincount(row, minlength=n) for row in idx])      # [resamples, n_images]
    out = {}
    for name in species:
        s_id = C.CLASS_NAMES.index(name)
        tp, conf, pimg, gt = species_arrays(arrays, s_id)
        full = class_ap(tp, conf, int(gt.sum()), s_id, ap_fn)
        vals = []
        for b in range(resamples):
            m = counts[b]
            n_gt = int((m * gt).sum())
            if n_gt <= 0:
                continue
            rep = m[pimg]
            vals.append(class_ap(np.repeat(tp, rep, axis=0), np.repeat(conf, rep), n_gt, s_id, ap_fn))
        v = np.asarray(vals, dtype=np.float64)
        se = float(v.std(ddof=1)) if len(v) >= 2 else None
        out[name] = {"ap": full, "se": se, "n_valid": int(len(v)), "n_gt": int(gt.sum()),
                     "mean": float(v.mean()) if len(v) else None,
                     "p2_5": float(np.percentile(v, 2.5)) if len(v) else None,
                     "p97_5": float(np.percentile(v, 97.5)) if len(v) else None}
    return out


# ------------------------------------------------------------------- checks
def compare_scores(recorded, sidecar):
    """Raise unless the sidecar's own score is the recorded score's (stamps
    equal, per-class AP within MAX_SCORE_DIFF). Returns the largest diff."""
    for k in STAMPS + ("n_gt", "production"):
        if recorded.get(k) != sidecar.get(k):
            raise SidecarError("the sidecar's score has %s %r, the recorded score %r: not the same weights, "
                               "exam or scorer" % (k, str(sidecar.get(k))[:40], str(recorded.get(k))[:40]))
    if set(recorded.get("per_class") or {}) != set(sidecar.get("per_class") or {}):
        raise SidecarError("per_class covers %s in the sidecar's score, %s in the recorded one"
                           % (sorted(sidecar.get("per_class") or {}), sorted(recorded.get("per_class") or {})))
    diffs = [abs(float(recorded["map50_95"]) - float(sidecar["map50_95"]))]
    diffs += [abs(float(recorded["per_class"][c]) - float(sidecar["per_class"][c])) for c in recorded["per_class"]]
    worst = max(diffs)
    if not math.isfinite(worst) or worst > MAX_SCORE_DIFF:
        raise SidecarError("the sidecar's score differs from the recorded one by %.6f (> %g) in mAP or a class's "
                           "AP: the predictions are not the recorded score's" % (worst, MAX_SCORE_DIFF))
    return worst


def check_capture(per_class, images, ap_fn=None):
    """check_recompute on the arrays concatenated in the order the validator
    saw the images (images' insertion order), the order DetMetrics
    concatenates its stats in. ap_per_class sorts with np.argsort(-conf), which
    is not stable: predictions with equal confidence (common at half precision)
    are ranked by input order, so the same arrays in another image order give a
    slightly different AP (0.0007-0.0012 on the real dev exam, pilot_v4,
    2026-09-29). Only this order reproduces the score exactly."""
    return check_recompute(per_class, flatten(images, list(images)), ap_fn)


def check_recompute(per_class, arrays, ap_fn=None):
    """Raise unless ap_per_class on the captured arrays gives the score's
    per_class exactly. Returns (largest diff, largest class-restricted diff)."""
    ap_fn = ap_fn or _ap_per_class()
    full = full_per_class(arrays, ap_fn)
    if set(full) != set(per_class):
        raise SidecarError("the captured arrays cover classes %s, the score %s" % (sorted(full), sorted(per_class)))
    worst = max([abs(full[c] - float(per_class[c])) for c in per_class] or [0.0])
    if worst > MAX_RECOMPUTE_DIFF:
        raise SidecarError("ap_per_class on the captured per-image arrays differs from the score's per_class by "
                           "%.3g: they are not the inputs of that score" % worst)
    restricted = 0.0
    for name in SPECIES:
        if name not in per_class:
            continue
        s_id = C.CLASS_NAMES.index(name)
        tp, conf, _, gt = species_arrays(arrays, s_id)
        restricted = max(restricted, abs(class_ap(tp, conf, int(gt.sum()), s_id, ap_fn) - full[name]))
    if restricted > MAX_RESTRICTED_DIFF:
        raise SidecarError("a species' AP from its own detections differs from the full call by %.4f" % restricted)
    return worst, restricted


# --------------------------------------------------------------------- run
def run(weights, exam, score_json, out_json=None, lock_check=True, imgsz=S.IMGSZ, batch=S.BATCH, device=None,
        resamples=RESAMPLES, seed_text=SEED_TEXT):
    """Write the sidecar of score_json (module docstring); returns its record."""
    score_json = Path(score_json).resolve()
    out_json, npz = paths_for(Path(out_json or default_out(score_json)).resolve())
    for p in (out_json, npz):
        if p.is_file() or p.is_symlink():
            p.unlink()
    with open(score_json, "rb") as fh:
        raw = fh.read()
    recorded = json.loads(raw.decode("utf-8"))
    if recorded.get("exam") != exam:
        raise SidecarError("%s is a score on %r, not %r" % (score_json, recorded.get("exam"), exam))
    weights = Path(weights).resolve()
    if C.sha256_file(weights) != recorded.get("weights_sha256"):
        raise SidecarError("%s does not hash to the weights_sha256 %s records" % (weights, score_json))

    res, images = capture_score(weights, exam, lock_check=lock_check, imgsz=imgsz, batch=batch, device=device)
    worst = compare_scores(recorded, res)
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path(exam)))
    if C.sha256_text("\n".join(keys)) != res["key_order_sha256"]:
        raise SidecarError("the exam's key order does not hash to the score's key_order_sha256")
    arrays = flatten(images, keys)        # the npz and the bootstrap: exam key order
    ap_fn = _ap_per_class()
    recompute, restricted = check_capture(res["per_class"], images, ap_fn)
    se = bootstrap_species_se(arrays, resamples=resamples, seed_text=seed_text, ap_fn=ap_fn)
    npz_sha = save_npz(npz, arrays)
    import ultralytics
    rec = {
        "format": FORMAT, "exam": exam, "created_utc": _utc(),
        "weights": str(weights), "weights_sha256": res["weights_sha256"],
        "score": {"path": str(score_json), "sha256": hashlib.sha256(raw).hexdigest(),
                  "stamps": {k: recorded.get(k) for k in STAMPS}, "production": recorded.get("production"),
                  "n_gt": recorded.get("n_gt")},
        "sidecar_score": {k: res.get(k) for k in ("map50_95", "map50", "per_class", "production", "deviations",
                                                   "scorer_sha256", "settings", "device")},
        "consistency": {"max_abs_diff_vs_recorded": worst, "max_score_diff": MAX_SCORE_DIFF,
                        "full_recompute_max_abs_diff": recompute, "max_recompute_diff": MAX_RECOMPUTE_DIFF,
                        "full_recompute_order": "capture (the validator's image order; the npz is in exam key "
                                                "order, and ap_per_class ranks equal confidences by input order)",
                        "species_restricted_max_abs_diff": restricted},
        "images": {"path": str(npz), "sha256": npz_sha, "n_images": len(keys),
                   "n_predictions": int(len(arrays["conf"])), "n_targets": int(len(arrays["target_cls"]))},
        "species_se": {"seed_text": seed_text, "seed": C.stable_int(seed_text), "resamples": int(resamples),
                       "statistic": "AP50-95 (ap_per_class, mean over IoU 0.50:0.95)", "ddof": 1,
                       "per_species": se},
        "code": {"scorer_sidecar": C.sha256_file(Path(__file__).resolve()),
                 "scorer": C.sha256_file(Path(S.__file__).resolve())},
        "ultralytics_version": ultralytics.__version__,
    }
    _write_json(out_json, rec)
    log("%s on %s: SE per species %s -> %s" % (weights.name, exam,
                                              {s: (round(v["se"], 4) if v["se"] is not None else None)
                                               for s, v in se.items()}, out_json))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-image AP inputs and the species SE of one locked score.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--score", required=True, help="the recorded score.json of these weights on this exam")
    ap.add_argument("--out", default=None, help="default: <score stem>.sidecar.json beside --score")
    ap.add_argument("--imgsz", type=int, default=S.IMGSZ)
    ap.add_argument("--batch", type=int, default=S.BATCH)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        run(a.weights, a.exam, a.score, a.out, lock_check=not a.no_lock_check_for_tests, imgsz=a.imgsz,
            batch=a.batch, device=a.device)
    except (S.ScorerRefused, SidecarError) as e:
        print("[inc2.scorer_sidecar] REFUSED: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
