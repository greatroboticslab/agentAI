"""Class-agnostic per-image arrays of a baseline's final dev scores, for E1's
verdict (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-03): E1, weed-box base
v3 (pre-registered)").

    python -m weed_optimizer_framework.tools.inc2.scorer_agnostic --exp E --run final__base__s0
        [--batch N --device D --no-lock-check-for-tests]   (test mode only)

Why. E1's models are one-class weed detectors (every box INC class 12), so on
cwd12 dev (ground truth in classes 0-11) their 12-class AP is 0 by
construction and the comparison is the class-agnostic mAP50-95 the locked
scorer already reports (agnostic_map50_95). A paired image bootstrap of the
difference between two arms needs that AP per image, and the locked scorer
keeps only the concatenation. The sidecar (inc2.scorer_sidecar) keeps
class-aware matches, which are all false for a class-12 model on dev.

How. The locked scorer (inc/scorer.py, pinned) is used as a library, as the
sidecar and the native scorer use it: scorer.score() runs every one of its
checks (the LOCK, the exam materialisation, the model, the protocol settings
at 640 px) and its one Ultralytics validation pass. The only addition is a
subclass of the scorer's own validator that, after the scorer's per-image
hook has computed the class-collapsed match matrix, keeps it per image: the
image's key, its collapsed tp (N x 10), conf and number of GT boxes. Their
concatenation in the validator's order is exactly what the scorer's
collapsed_ap read, and that is checked on every pass (MAX_RECOMPUTE_DIFF).

What it refuses (AgnosticRefused, exit 2), before anything is written: any
exam but dev; an experiment that is not a baseline; a run that is not one of
its final runs (final__base__s<k>) or is not done; weights that do not hash
as run.json records; a recorded dev score that is missing, not a production
score (outside test mode) or on other weights; a pass whose stamps
(scorer, manifest, key order, images, GT boxes, weights) differ from the
recorded score's, or whose agnostic AP differs from it by more than
MAX_SCORE_DIFF (cuDNN non-determinism only); arrays that do not reproduce
the pass's agnostic AP; tie-broken arrays (inc2.scorer_sidecar.tie_break,
exam key order: the bootstrap's input) more than MAX_TIE_DIFF from the
recorded score; an output that exists (written once).

Outputs, beside the run's other scores:
  scores/dev.agnostic.json   format inc2-agnostic-score/1: the recorded
                             score's path, sha256 and stamps, the pass's
                             agnostic AP, the checks, the npz's sha256,
                             production false (never a protocol score)
  scores/dev.agnostic.npz    keys, pred_img, tp, conf, n_gt (per image), in
                             exam key order
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from . import scorer_sidecar as SC

FORMAT = "inc2-agnostic-score/1"
EXAMS = ("dev",)
JSON_NAME = "%s.agnostic.json"
NPZ_NAME = "%s.agnostic.npz"
MAX_SCORE_DIFF = SC.MAX_SCORE_DIFF        # the pass against the recorded protocol score (cuDNN only)
MAX_RECOMPUTE_DIFF = 1e-9                 # collapsed_ap on the captured arrays against the pass's own
# The tie-broken arrays in exam key order against the recorded score: they differ from the validator's order only in
# how equal confidences rank (ap_per_class's sort is not stable; up to 0.0103 for a species' AP on a real dev exam,
# inc2.scorer_sidecar.tie_break). The exact identity check is MAX_RECOMPUTE_DIFF in capture order; this bound only
# catches arrays that are not the score's, without refusing a real run on its ties.
MAX_TIE_DIFF = 0.02
FINAL_RUN_RE = re.compile(r"final__base__s(\d{1,2})\Z")
STAMPS = ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "n_images", "weights_sha256")


class AgnosticRefused(RuntimeError):
    """What was asked is not a pre-registered agnostic rescore."""


def log(msg):
    print("[inc2.scorer_agnostic] %s" % msg, flush=True)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def paths_for(scores_dir, exam="dev"):
    d = Path(scores_dir)
    return d / (JSON_NAME % exam), d / (NPZ_NAME % exam)


# ----------------------------------------------------------------- capture
_CAPTURE = []


def validator_class():
    """The locked scorer's validator, keeping each image's collapsed matches."""
    base = S.validator_class()

    class AgnosticValidator(base):
        def inc_reset(self):
            super().inc_reset()
            self.ag_images = {}
            self.ag_order = []
            _CAPTURE.append(self)

        def _process_batch(self, preds, batch):
            n_before = len(self.inc_tp)
            out = super()._process_batch(preds, batch)
            if len(self.inc_tp) != n_before + 1:
                raise RuntimeError("the scorer's per-image hook did not add one image")
            key = Path(batch["im_file"]).stem
            if key in self.ag_images:
                raise RuntimeError("exam image %s was captured twice" % key)
            self.ag_images[key] = {"tp": SC._as_tp(self.inc_tp[-1], len(self.inc_conf[-1])),
                                   "conf": self.inc_conf[-1].reshape(-1), "n_gt": int(len(batch["cls"]))}
            self.ag_order.append(key)
            return out

    return AgnosticValidator


def capture(weights, exam, lock_check=True, imgsz=S.IMGSZ, batch=S.BATCH, device=None):
    """(the scorer's result, {key: arrays}, capture order) of one scorer.score
    call with the capturing validator installed for it."""
    base = S.validator_class()
    cls = validator_class()
    prev = S._VALIDATOR
    del _CAPTURE[:]
    with tempfile.TemporaryDirectory(prefix="inc2_agnostic_") as tmp:
        S._VALIDATOR = cls
        try:
            res = S.score(weights, exam, Path(tmp) / "score.json", lock_check=lock_check, imgsz=imgsz, batch=batch,
                          device=device)
        finally:
            S._VALIDATOR = prev if prev is not None else base
    if not _CAPTURE:
        raise AgnosticRefused("the scorer ran no capturing validator")
    v = _CAPTURE[-1]
    return res, dict(v.ag_images), list(v.ag_order)


def flatten(images, keys):
    """{keys, tp, conf, pred_img, n_gt} in the given key order."""
    import numpy as np
    missing = [k for k in keys if k not in images]
    if missing or len(images) != len(keys):
        raise AgnosticRefused("captured %d images for %d exam keys (missing %s)" % (len(images), len(keys),
                                                                                   missing[:3]))
    tps, confs, pimg, ngt = [], [], [], []
    for i, k in enumerate(keys):
        d = images[k]
        conf = np.asarray(d["conf"], dtype=np.float64).reshape(-1)
        tps.append(SC._as_tp(d["tp"], len(conf)))
        confs.append(conf)
        pimg.append(np.full(len(conf), i, dtype=np.int64))
        ngt.append(int(d["n_gt"]))
    niou = next((t.shape[1] for t in tps if t.shape[1]), SC.NIOU)
    tps = [t if t.shape[1] == niou else np.zeros((len(t), niou), dtype=bool) for t in tps]
    return {"keys": np.asarray(list(keys)), "tp": np.concatenate(tps, 0) if tps else np.zeros((0, niou), bool),
            "conf": np.concatenate(confs) if confs else np.zeros(0),
            "pred_img": np.concatenate(pimg) if pimg else np.zeros(0, dtype=np.int64),
            "n_gt": np.asarray(ngt, dtype=np.int64)}


def collapsed(arrays):
    """(AP50-95, AP50) of the arrays, by the locked scorer's collapsed_ap."""
    return S.collapsed_ap(arrays["tp"], arrays["conf"], int(arrays["n_gt"].sum()))


def save_npz(path, arrays):
    return SC.save_npz(path, arrays)


def load_npz(path):
    return SC.load_npz(path)


# --------------------------------------------------------------------- run
def final_run(exp, run_id):
    """(exp.json, run.json, seed) of a done final run of a baseline experiment."""
    root = C.INC_DIR / exp
    defn = _read_json(root / "exp.json")
    if not isinstance(defn, dict) or defn.get("type") != "baseline":
        raise AgnosticRefused("%s is not a baseline experiment" % exp)
    m = FINAL_RUN_RE.match(str(run_id))
    if not m or int(m.group(1)) not in [int(s) for s in defn.get("seeds") or []]:
        raise AgnosticRefused("%s is not one of %s's final runs" % (run_id, exp))
    rj = _read_json(root / "runs" / run_id / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise AgnosticRefused("%s/%s is not done" % (exp, run_id))
    return defn, rj, int(m.group(1))


def score_run(exp, run_id, exam="dev", batch=None, device=None, lock_check=None):
    """Write scores/<exam>.agnostic.{json,npz} of a done final run (module
    docstring); returns the record (the existing one when written already)."""
    if exam not in EXAMS:
        raise AgnosticRefused("exam %r: the agnostic rescore reads %s only" % (exam, list(EXAMS)))
    defn, rj, _seed = final_run(exp, run_id)
    scores = C.INC_DIR / exp / "runs" / run_id / "scores"
    js, npz = paths_for(scores, exam)
    if js.is_file():
        rec = _read_json(js)
        if not isinstance(rec, dict) or rec.get("format") != FORMAT:
            raise AgnosticRefused("%s exists and is not an agnostic score" % js)
        return dict(rec, status="kept")
    if defn.get("testing") and not S.testing():
        raise AgnosticRefused("%s is a testing experiment: it is scored only with %s=1" % (exp, S.TEST_ENV))
    testing = bool(defn.get("testing"))
    rec_path = scores / ("%s.json" % exam)
    recorded = _read_json(rec_path)
    if not isinstance(recorded, dict) or recorded.get("exam") != exam:
        raise AgnosticRefused("%s has no recorded %s score" % (run_id, exam))
    if recorded.get("production") is not True and not testing:
        raise AgnosticRefused("%s is not a production score" % rec_path)
    weights = Path(rj.get("weights") or (C.INC_DIR / exp / "runs" / run_id / "weights" / "final.pt")).resolve()
    if not weights.is_file() or C.sha256_file(weights) != rj.get("weights_sha256") \
            or recorded.get("weights_sha256") != rj.get("weights_sha256"):
        raise AgnosticRefused("%s's weights do not hash as run.json and its score record" % run_id)
    if testing:
        # the settings inc2.train scored this run with (scorer_command): exp.json's testing object
        st = defn["testing"] if isinstance(defn.get("testing"), dict) else {}
        kw = {"imgsz": int(st.get("imgsz") or S.IMGSZ), "batch": int(batch or st.get("batch") or S.BATCH),
              "device": device or st.get("device") or "cpu",
              "lock_check": (st.get("lock_check") is not False) if lock_check is None else lock_check}
    else:
        kw = {"batch": int(batch or S.BATCH), "device": device, "lock_check": True if lock_check is None
              else lock_check}
    res, images, order = capture(weights, exam, **kw)
    for k in STAMPS + ("n_gt", "production"):
        if res.get(k) != recorded.get(k):
            raise AgnosticRefused("the pass has %s %r, the recorded score %r: not the same weights, exam or scorer"
                                  % (k, str(res.get(k))[:40], str(recorded.get(k))[:40]))
    rec_ag = float(recorded["agnostic_map50_95"])
    diff = abs(float(res["agnostic_map50_95"]) - rec_ag)
    if diff > MAX_SCORE_DIFF:
        raise AgnosticRefused("the pass's agnostic AP differs from the recorded one by %.6f (> %g)"
                              % (diff, MAX_SCORE_DIFF))
    cap = collapsed(flatten(images, order))
    rd = abs(cap[0] - float(res["agnostic_map50_95"]))
    if rd > MAX_RECOMPUTE_DIFF:
        raise AgnosticRefused("collapsed_ap on the captured arrays differs from the pass by %.3g: they are not its "
                              "inputs" % rd)
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path(exam)))
    if C.sha256_text("\n".join(keys)) != res["key_order_sha256"]:
        raise AgnosticRefused("the exam's key order does not hash to the score's key_order_sha256")
    arrays = flatten(images, keys)
    tb = collapsed(SC.tie_break(arrays))
    tdiff = abs(tb[0] - rec_ag)
    if tdiff > MAX_TIE_DIFF:
        raise AgnosticRefused("the tie-broken arrays give %.6f, %.6f from the recorded agnostic AP (> %g)"
                              % (tb[0], tdiff, MAX_TIE_DIFF))
    import ultralytics
    npz_sha = save_npz(npz, arrays)
    rec = {"format": FORMAT, "exp": exp, "run_id": run_id, "exam": exam, "created_utc": _utc(), "production": False,
           "agnostic_production": not testing and kw["lock_check"] is True and recorded.get("production") is True,
           "recorded": {"name": rec_path.name, "sha256": C.sha256_file(rec_path),
                        "stamps": {k: recorded.get(k) for k in STAMPS}, "agnostic_map50_95": rec_ag,
                        "agnostic_map50": recorded.get("agnostic_map50")},
           "agnostic_map50_95": float(res["agnostic_map50_95"]), "agnostic_map50": float(res["agnostic_map50"]),
           "tie_broken_agnostic_map50_95": tb[0], "n_gt": int(arrays["n_gt"].sum()), "n_images": len(keys),
           "key_order_sha256": res["key_order_sha256"],
           "checks": {"vs_recorded": diff, "max_score_diff": MAX_SCORE_DIFF, "recompute": rd,
                      "max_recompute_diff": MAX_RECOMPUTE_DIFF, "tie_broken_vs_recorded": tdiff,
                      "max_tie_diff": MAX_TIE_DIFF},
           "images": {"name": npz.name, "sha256": npz_sha, "n_predictions": int(len(arrays["conf"]))},
           "code": {"scorer_agnostic": C.sha256_file(Path(__file__).resolve()),
                    "scorer": C.sha256_file(Path(S.__file__).resolve())},
           "ultralytics_version": ultralytics.__version__}
    tmp = js.with_name(".%s.%d.tmp" % (js.name, os.getpid()))
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    try:
        os.link(tmp, js)                       # written once: never replaces an existing record
    except FileExistsError:
        raise AgnosticRefused("%s appeared while this pass ran: written once" % js)
    finally:
        tmp.unlink()
    log("%s %s on %s: agnostic %.4f (recorded %.4f) -> %s" % (exp, run_id, exam, rec["agnostic_map50_95"], rec_ag, js))
    return dict(rec, status="written")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Class-agnostic per-image arrays of a baseline's final dev score.")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--exam", default="dev")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        score_run(a.exp, a.run, a.exam, batch=a.batch, device=a.device,
                  lock_check=False if a.no_lock_check_for_tests else None)
    except (AgnosticRefused, S.ScorerRefused) as e:
        print("[inc2.scorer_agnostic] REFUSED: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
