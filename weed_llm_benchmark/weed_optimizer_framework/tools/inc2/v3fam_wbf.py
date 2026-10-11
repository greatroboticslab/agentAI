"""V3F D3: the seed ensemble of E3-B's stage-1 boxes by weighted boxes fusion,
one fixed setting, no grid (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11):
V3F, the v3 experiment family D0-D9 (pre-registered as one family)", V3F.5
D3; decision 9).

    python -m weed_optimizer_framework.tools.inc2.v3fam_wbf score      (= inc2.v3fam score --arm D3, L23J)
    python -m weed_optimizer_framework.tools.inc2.v3fam_wbf impl       (the ensemble_boxes record; pre-flight)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

The fusion (fixed now). Per exam image, the three seeds' stage-1 rows of the
family's own E3-B records (E3RS's arms/E3B/s<k>/<exam>.stage1.npz: one_per_box
on the locked NMS's output, xyxy in original pixels, q), normalised by the
image's size as it is seen after EXIF transposition, go through
ensemble_boxes.weighted_boxes_fusion 1.0.9 (its file's sha256 pinned,
ENSEMBLE_BOXES_WBF_SHA256) with iou_thr 0.55, skip_box_thr 0.001, conf_type
avg, weights [1, 1, 1] and one label; the fused boxes are denormalised and at
most 300 per image are kept by score (ties: WBF's order). Exactly seeds 0, 1
and 2 are fused; another seed's record refuses.

Scoring. The fused boxes replace stage 1 in the family inject_boxes pass
(inc2.v3fam): E1-B seed 0's weights drive the validator scaffold, E3's pinned
BioCLIP-2 classifier names every fused box and E3's emission emits it, q =
the fused score. Dev decides (P4 and Tier C, one system: arms/D3/ens/dev.json
against E3RS's E3-B arrays); ImageWeeds is fused the same way from E3RS's
E3-B ImageWeeds records (reported); ooddev_v1 from E3RS's E1-B records in the
zoo's root, injected as rows of the one-class detectors' class through the
inject_rows pass in a subprocess with INC_DIR there (agnostic, reported).

Records: v3fam/v1/arms/D3/ens/<exam>.{json, images.npz, stage1.npz | rows.npz}
(written once; the fusion's settings, the library record, the inputs' sha256s
and the fusion counts in each JSON); capacity/v3fam_score_d3.json (dev only);
v3fam/v1/reported/D3.json.

A refusal raises inc2.v3fam.V3famRefused; the CLI prints "[inc2.v3fam_wbf]
ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import common as C2
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_systems as SYS

# ------------------------------------------------------------------ constants (pre-registered, V3F.5 D3)
ARM = "D3"
DIR = "D3"
ENS = "ens"                                  # one system: no seed
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001
WBF_CONF_TYPE = "avg"
WBF_WEIGHTS = (1, 1, 1)                      # seeds 0, 1, 2
WBF_LABEL = 0                                # one label
MAX_FUSED = 300                              # fused boxes per image, by score
NORM_EPS = 1e-9                              # a normalised coordinate this far outside [0, 1] refuses
ENSEMBLE_BOXES_VERSION = "1.0.9"
ENSEMBLE_BOXES_WBF_FILE = "ensemble_boxes_wbf.py"
ENSEMBLE_BOXES_WBF_SHA256 = "6189c54cf9582d1f7afadcb1ceb751c39e86e8e2ae7a35b7f0f44db1c2fe4efb"   # the 1.0.9 wheel's
SCAFFOLD = ("B", 0)                          # E1-B seed 0's weights drive the validator scaffold
ONE_CLASS_ID = C.OTHER_PLANT                 # the class E1's one-class detectors predict (ooddev_v1 rows)
INPUT_DIRS = {"dev": SYS.E3_DIRS["B"], "imageweeds": SYS.E3_DIRS["B"], SYS.OODDEV: SYS.E1_DIRS["B"]}
REPORTED_EXAMS = ("imageweeds", SYS.OODDEV)
FMT_FUSION = "inc2-v3fam-wbf/1"
SETTINGS = {"iou_thr": WBF_IOU_THR, "skip_box_thr": WBF_SKIP_BOX_THR, "conf_type": WBF_CONF_TYPE,
            "weights": list(WBF_WEIGHTS), "labels": "one label", "max_fused_per_image": MAX_FUSED,
            "normalisation": "xyxy / (W, H) of the EXIF-transposed image", "seeds": list(F.SEEDS)}


def log(msg):
    print("[inc2.v3fam_wbf] %s" % msg, flush=True)


# ------------------------------------------------------------------ the library
def wbf_impl():
    """(ensemble_boxes.weighted_boxes_fusion, its record), refusing another version or another file."""
    try:
        import importlib.metadata as md
        import ensemble_boxes
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as e:
        raise F.V3famRefused("ensemble_boxes is not importable (%s): D3 fuses with ensemble_boxes %s"
                             % (e, ENSEMBLE_BOXES_VERSION))
    ver = None
    for dist in ("ensemble-boxes", "ensemble_boxes"):
        try:
            ver = md.version(dist)
            break
        except md.PackageNotFoundError:
            continue
    path = Path(ensemble_boxes.__file__).resolve().parent / ENSEMBLE_BOXES_WBF_FILE
    sha = C.sha256_file(path) if path.is_file() else None
    if ver != ENSEMBLE_BOXES_VERSION or sha != ENSEMBLE_BOXES_WBF_SHA256 \
            or weighted_boxes_fusion.__module__ != "ensemble_boxes.ensemble_boxes_wbf":
        raise F.V3famRefused("ensemble_boxes %s with %s sha256 %s: D3 is pre-registered on %s (sha256 %s)"
                             % (ver, ENSEMBLE_BOXES_WBF_FILE, str(sha)[:12], ENSEMBLE_BOXES_VERSION,
                                ENSEMBLE_BOXES_WBF_SHA256[:12]))
    return weighted_boxes_fusion, {"package": "ensemble_boxes", "version": ver, "file": str(path), "sha256": sha}


# ------------------------------------------------------------------ the fusion
def normalise(xyxy, w, h):
    """Original-pixel xyxy -> [0, 1] by the image's (W, H). float64 [n, 4]."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return a / np.asarray([w, h, w, h], dtype=np.float64)


def denormalise(xyxy, w, h):
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return a * np.asarray([w, h, w, h], dtype=np.float64)


def fuse_image(seed_rows, w, h, fn):
    """One image's fused boxes: seed_rows [{xyxy (original px), q}] for seeds 0, 1, 2 (exactly three), (W, H) the
    image's size. Returns (xyxy float64 [n, 4] original px, score float64 [n], counts), at most MAX_FUSED rows,
    in score order."""
    import numpy as np
    if len(seed_rows) != len(WBF_WEIGHTS):
        raise F.V3famRefused("D3 fuses exactly %d stage-1 seeds, not %d" % (len(WBF_WEIGHTS), len(seed_rows)))
    w, h = float(w), float(h)
    if not (w > 0 and h > 0):
        raise F.V3famRefused("an image of size %r x %r" % (w, h))
    boxes, scores, labels = [], [], []
    st = {"rows_in": 0, "zero_area": 0, "below_skip": 0}
    for r in seed_rows:
        xy = np.asarray(r["xyxy"], dtype=np.float64).reshape(-1, 4)
        q = np.asarray(r["q"], dtype=np.float64).reshape(-1)
        if len(xy) != len(q) or not (np.isfinite(xy).all() and np.isfinite(q).all()):
            raise F.V3famRefused("a stage-1 record is not n finite boxes with n finite scores")
        nb = normalise(xy, w, h)
        if len(nb) and (nb.min() < -NORM_EPS or nb.max() > 1.0 + NORM_EPS):
            raise F.V3famRefused("a stage-1 box lies outside its %dx%d image (normalised %.6f..%.6f): the records "
                                 "are clipped to the image" % (w, h, nb.min(), nb.max()))
        nb = np.clip(nb, 0.0, 1.0)
        live = q >= WBF_SKIP_BOX_THR
        st["rows_in"] += int(len(q))
        st["below_skip"] += int((~live).sum())
        st["zero_area"] += int(((nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1]) == 0.0)[live].sum())
        boxes.append(nb)
        scores.append(q)
        labels.append(np.full(len(q), WBF_LABEL, dtype=np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # its zero-area notices: counted above
        fb, fs, _fl = fn(boxes, scores, labels, weights=list(WBF_WEIGHTS), iou_thr=WBF_IOU_THR,
                         skip_box_thr=WBF_SKIP_BOX_THR, conf_type=WBF_CONF_TYPE)
    fb = np.asarray(fb, dtype=np.float64).reshape(-1, 4)
    fs = np.asarray(fs, dtype=np.float64).reshape(-1)
    if len(fb) != len(fs) or not (np.isfinite(fb).all() and np.isfinite(fs).all()):
        raise F.V3famRefused("the fusion returned %d boxes for %d scores, or a non-finite value" % (len(fb), len(fs)))
    order = np.argsort(-fs, kind="stable")
    st["fused"] = int(len(fs))
    st["capped"] = int(max(0, len(order) - MAX_FUSED))
    order = order[:MAX_FUSED]
    st["kept"] = int(len(order))
    return denormalise(fb[order], w, h), fs[order], st


def fuse_exam(stage1s, sizes, fn):
    """Every exam image's fused boxes from the three seeds' stage-1 records (keys, img ascending, xyxy, q; one
    exam key order) and {key: (W, H)}. Returns ({key: {xyxy, conf}} for every key, counts)."""
    import numpy as np
    if len(stage1s) != len(WBF_WEIGHTS):
        raise F.V3famRefused("D3 fuses exactly %d stage-1 seeds, not %d" % (len(WBF_WEIGHTS), len(stage1s)))
    keys = [str(k) for k in stage1s[0]["keys"]]
    for a in stage1s[1:]:
        if [str(k) for k in a["keys"]] != keys:
            raise F.V3famRefused("the seeds' stage-1 records are not in one exam key order")
    if sorted(sizes) != sorted(keys):
        raise F.V3famRefused("the image sizes cover %d keys, the records %d" % (len(sizes), len(keys)))
    cuts = []
    for a in stage1s:
        img = np.asarray(a["img"], dtype=np.int64)
        if len(img) and (np.any(np.diff(img) < 0) or img[0] < 0 or img[-1] >= len(keys)):
            raise F.V3famRefused("a stage-1 record is not grouped by image in key order")
        cuts.append(np.searchsorted(img, np.arange(len(keys) + 1)))
    out, tot = {}, {"images": len(keys), "rows_in": 0, "zero_area": 0, "below_skip": 0, "fused": 0, "capped": 0,
                    "kept": 0, "images_capped": 0, "max_kept_per_image": 0}
    for i, k in enumerate(keys):
        rows = [{"xyxy": np.asarray(a["xyxy"])[c[i]:c[i + 1]], "q": np.asarray(a["q"])[c[i]:c[i + 1]]}
                for a, c in zip(stage1s, cuts)]
        w, h = sizes[k]
        xy, sc, st = fuse_image(rows, w, h, fn)
        out[k] = {"xyxy": xy, "conf": sc.astype(np.float32)}
        for f in ("rows_in", "zero_area", "below_skip", "fused", "capped", "kept"):
            tot[f] += st[f]
        tot["images_capped"] += int(st["capped"] > 0)
        tot["max_kept_per_image"] = max(tot["max_kept_per_image"], st["kept"])
    return out, tot


def as_rows(fused, cls_id=ONE_CLASS_ID):
    """Fused boxes as inject_rows rows of one class (ooddev_v1: agnostic only)."""
    import numpy as np
    return {k: {"xyxy": v["xyxy"], "conf": v["conf"], "cls": np.full(len(v["conf"]), float(cls_id))}
            for k, v in fused.items()}


# ------------------------------------------------------------------ the inputs (E3RS's records)
def load_inputs(exam):
    """The three seeds' stage-1 records D3 fuses on exam (E3RS's E3-B records; ooddev_v1: E1-B's), each checked
    against the sha256 its score JSON records. A record of any other seed refuses (exactly 0, 1, 2)."""
    if exam not in INPUT_DIRS:
        raise F.V3famRefused("D3 reads %s, not %s" % (sorted(INPUT_DIRS), exam))
    d = INPUT_DIRS[exam]
    base = F.root() / "arms" / d
    jn = F.file_names(exam)[0]
    have = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / jn).exists()) if base.is_dir() else []
    want = sorted("s%d" % s for s in F.SEEDS)
    if have != want:
        raise F.V3famRefused("D3 fuses exactly %s's seeds %s on %s; arms/%s holds %s (E3RS writes them)"
                             % (d, list(F.SEEDS), exam, d, have))
    arrays, recs = [], []
    for s in F.SEEDS:
        doc, a = SYS.family_side(d, s, exam, "stage1_images")
        if doc.get("arm") != "E3RS" or doc.get("dir") != d or doc.get("seed") != s or doc.get("exam") != exam:
            raise F.V3famRefused("arms/%s/s%d/%s is not E3RS's %s record" % (d, s, jn, d))
        arrays.append(a)
        recs.append({"seed": s, "file": SYS.rel(F.arm_dir(d, s) / jn), "sha256": SYS._sha(F.arm_dir(d, s) / jn),
                     "stage1_sha256": doc["stage1_images"]["sha256"], "n_boxes": int(len(a["q"]))})
    return arrays, recs


def fused_for(exam, fn, impl):
    """({key: fused boxes}, the fusion record) of an exam."""
    stage1s, inputs = load_inputs(exam)
    sizes = SYS.image_sizes(SYS.exam_rows(exam))
    fused, counts = fuse_exam(stage1s, sizes, fn)
    return fused, {"format": FMT_FUSION, "settings": dict(SETTINGS), "library": impl, "inputs": inputs,
                   "counts": counts, "input_dir": INPUT_DIRS[exam]}


# ------------------------------------------------------------------ the score job (L23J --arm D3)
def score_d3(embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, reported=True):
    """D3's scoring job: the fused boxes on dev through the family inject_boxes pass (deciding: P4, Tier C), then
    ImageWeeds (inject_boxes) and ooddev_v1 (inject_rows, in the zoo's root), reported. Written once each."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    fn, impl = wbf_impl()
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        names = F.equivalence_names("dev", e3_arms=("B",))
        eqs = F.require_equivalence(names)
        scaffold = SYS.e3_source(*SCAFFOLD)
        ccfg, _crec, shas = TS._clf_cfg()
        with TS._workers(workers) as wk:
            got = SYS.existing(DIR, None, "dev", ens=ENS)
            if got is None:
                fused, frec = fused_for("dev", fn, impl)
                embedder = embedder or V.BioclipEmbedder()
                cfg = dict(ccfg, mode="inject_boxes", nms="locked", embedder=embedder, workers=wk, top_k=TS.TOP_K,
                           inject=fused)
                ps = F._family_pass(scaffold["weights"], "dev", cfg, ARM, **kw)
                got = SYS.commit(ARM, DIR, None, "dev", ps, cfg, names, classifier=shas, ens=ENS,
                                 extra={"fusion": frec, "scaffold": SYS.source_record(scaffold),
                                        "one_system": True})
                log("D3 on dev: species %.4f, fused stage-1 agnostic %.4f (%d fused boxes on %d images, %d capped)"
                    % (ps["result"]["species_map50_95"] or 0.0, ps["stage1_agnostic"][0], frec["counts"]["kept"],
                       frec["counts"]["images"], frec["counts"]["capped"]))
            rec = SYS.score_record(ARM, [got], eqs, before=before,
                                   reported={"status": "pending" if reported else "not run"},
                                   extra={"one_system": True, "library": impl})
            if reported:
                passes = []
                for exam in REPORTED_EXAMS:
                    item = {"dir": DIR, "seed": None, "exam": exam}
                    g = SYS.existing(DIR, None, exam, ens=ENS)
                    if g is not None:
                        passes.append(dict(item, file=g["file"], sha256=g["sha256"], status="kept"))
                        continue

                    def one(exam=exam):
                        fused, frec = fused_for(exam, fn, impl)
                        ex = {"fusion": frec, "scaffold": SYS.source_record(scaffold), "one_system": True}
                        if exam in F.ROOT_EXAMS:
                            ex_names = F.equivalence_names(exam)
                            F.require_equivalence(ex_names)
                            cfg = {"mode": "inject_rows", "nms": "locked", "inject": as_rows(fused)}
                            ps = SYS.in_root(scaffold["weights"], exam, cfg, ARM, kw)
                            ex["statistic"] = "agnostic (weed boxes only); rows of class %d" % ONE_CLASS_ID
                            return SYS.commit(ARM, DIR, None, exam, ps, cfg, ex_names, ens=ENS, extra=ex)
                        ex_names = F.equivalence_names(exam, e3_arms=("B",))
                        F.require_equivalence(ex_names)
                        emb = embedder or V.BioclipEmbedder()
                        cfg = dict(ccfg, mode="inject_boxes", nms="locked", embedder=emb, workers=wk,
                                   top_k=TS.TOP_K, inject=fused)
                        ps = F._family_pass(scaffold["weights"], exam, cfg, ARM, **kw)
                        return SYS.commit(ARM, DIR, None, exam, ps, cfg, ex_names, classifier=shas, ens=ENS,
                                          extra=ex)
                    SYS.reported_pass(passes, item, one)
                rec = SYS.score_record(ARM, [got], eqs, before=before, reported=SYS.write_reported(ARM, passes),
                                       extra={"one_system": True, "library": impl})
    log("D3 scored (%.0fs)" % (time.time() - t0))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F D3: WBF of E3-B's three stage-1 seeds, one fixed setting.")
    ap.add_argument("command", choices=("score", "impl"))
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam_wbf] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    try:
        if a.command == "impl":
            print(json.dumps(wbf_impl()[1], indent=1, sort_keys=True))
        else:
            score_d3(lock_check=not a.no_lock_check_for_tests, batch=a.batch or S.BATCH, device=a.device)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError,
            C2.Inc2Error, ValueError) as e:
        print("[inc2.v3fam_wbf] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
