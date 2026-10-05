#!/usr/bin/env python3
"""E3: two-stage species detection (inc2/twostage.py; docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-05): E3, two-stage species detection (pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 exams materialised, the
v1 LOCK with scorer.py's sha256, LOCK v2 marked testing). The three stage-1
sources are fixtures: b_v2_m640 (exp.json naming base_v2's sha256, its base
runs' training manifest, three done base runs whose weights are real
13-class checkpoints, final runs scored on dev, ImageWeeds and test by the
locked scorer, and native dev@640 files that a decided capacity/e2_v1.json
lists), e1_a_m640 and e1_b_m640 (the same without test). A deterministic
stub embedder (per-crop channel statistics, 16 dims) replaces BioCLIP-2;
open_clip is never imported. Every pass is a real Ultralytics CPU pass in
test mode (INC_SCORER_TESTING=1).

What is pinned:
- the constants (arms, order, class-agnostic NMS for E3-M and the locked
  NMS for E3-A and E3-B, E3-M's locked-NMS reading reported, the EXIF
  orientations read, seeds, seed texts, 1,000 resamples, top 3, the C grid,
  the tolerances);
- emit (top 3, ties to the lower id, q x p float32, the conf and max_det
  limits, OtherPlant kept), the geometry (the per-axis inverse, the
  ground-truth round trip), the prior of a degenerate box;
- fit-classifier: the manifest checks and the guard called, every refusal
  writing nothing (another manifest, a planted dev image, a test v1 row,
  an evaluation manifest, a fit after a dev read, a second fit, a pin
  present), deterministic, the CV argmin with ties to the smaller C, a
  non-converged C left out, sessions never split, probabilities as
  scikit-learn's, no dev or test file opened, the crop-protocol check's
  refusals; the pin and restore-classifier;
- the crops: equal to verify._cut_task's, an EXIF-orientation tag
  refused in a pass (0 read as none), training images tagged 6, 3 and 0
  fitted with crops equal to verify._cut_task's and cut in their labels'
  frame (another tag refused), the dev ground-truth crops cut through the
  same geometry;
- the validator layering (two passes in one process, the restore after an
  exception, exactly one E3 layer);
- identity and equivalence (a real pass reproduces the locked scorer's score
  exactly, the stage-1 helper its agnostic AP), score-arm end to end (files
  written once, the dev record before the reported passes, a failed
  reported pass recorded, every refusal), the verdict on real and synthetic
  files (each condition alone, pooled sd, the largest D, ties M > A > B,
  pending, every refusal, the attribution's own draw, dev only, kept /
  refused / overwritten), the test read, score-test and test-report, and
  the CLI.

Run:  python3 tests/test_inc2_e3.py
"""
import builtins
import contextlib
import copy
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_native as N  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_native as SN  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402

FAILURES = W.FAILURES
check = W.check
KW = {"batch": 32, "device": "cpu", "lock_check": True}
REF = TS.REFERENCE


class StubEmbedder(object):
    """Deterministic 16-dim features of a crop: per-channel means of its four quadrants and of the whole, and a
    constant (no model, no download)."""
    name = "stub-hist16"
    dim = 16

    def __init__(self):
        self.calls = 0

    def __call__(self, pils):
        self.calls += 1
        out = []
        for p in pils:
            a = np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0
            h, w = a.shape[:2]
            q = [a[:h // 2, :w // 2], a[:h // 2, w // 2:], a[h // 2:, :w // 2], a[h // 2:, w // 2:]]
            v = [x.mean(axis=(0, 1)) for x in q] + [a.std(axis=(0, 1))]
            out.append(np.concatenate(v + [np.ones(1, np.float32)]))
        return np.stack(out).astype(np.float32)


EMB = StubEmbedder()


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (TS.E3Refused, B.BaselineError, T.RunError, SN.NativeRefused, S.ScorerRefused) as e:
        return e
    return None


def tree(root):
    root = pathlib.Path(root)
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


@contextlib.contextmanager
def moved(path):
    """Move a file or directory aside for the block, then put it back."""
    path = pathlib.Path(path)
    aside = path.with_name(path.name + ".aside")
    had = path.exists() or path.is_symlink()
    if had:
        os.replace(str(path), str(aside))
    try:
        yield
    finally:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(str(path))
        elif path.exists() or path.is_symlink():
            path.unlink()
        if had:
            os.replace(str(aside), str(path))


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def base_v2_sha():
    return W.sha(W.v2_dir() / "base_v2.jsonl")


# ------------------------------------------------------------------ the oracle detector
ORACLE = {"on": False, "degenerate": None, "p_right": 0.75}


def _install_oracle():
    """Ultralytics' DetectionValidator with its NMS output replaced by deterministic, GT-derived predictions of
    the weights being scored (seeded by their sha256): most GT boxes found with jitter, the right class 75 % of
    the time, a second row at identical coordinates with another class (multi-label), two false positives. The
    real NMS still runs first. ORACLE['on'] switches it; ORACLE['degenerate'] = key plants a sub-pixel box there.
    So every locked-scorer pass (the fixtures' recorded scores and E3's) reads non-trivial AP."""
    import hashlib
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator as DV
    pre0, post0 = DV.preprocess, DV.postprocess

    def pre(self, batch):
        batch = pre0(self, batch)
        self._oracle_batch = batch
        return batch

    def post(self, preds):
        real = post0(self, preds)
        if not ORACLE["on"]:
            return real
        b = self._oracle_batch
        if not hasattr(self, "_oracle_seed"):
            self._oracle_seed = int(W.sha(pathlib.Path(str(self.args.model)).resolve())[:8], 16)
        out = []
        for si in range(len(real)):
            pb = DV._prepare_batch(self, si, b)
            key = pathlib.Path(pb["im_file"]).stem
            rng = np.random.default_rng([self._oracle_seed, int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)])
            hh, ww = int(pb["imgsz"][0]), int(pb["imgsz"][1])
            rows = []
            for box, c in zip(pb["bboxes"].float().cpu().numpy(), pb["cls"].float().cpu().numpy()):
                if rng.random() < 0.1:
                    continue
                w, h = box[2] - box[0], box[3] - box[1]
                bb = np.clip(box + rng.normal(0, 0.04, 4) * np.array([w, h, w, h]), 0, [ww, hh, ww, hh])
                conf = 0.3 + 0.65 * rng.random()
                cls = int(c) if rng.random() < ORACLE["p_right"] else int(rng.integers(0, 13))
                rows.append((bb, conf, cls))
                rows.append((bb.copy(), conf * 0.5, int((cls + 1 + rng.integers(0, 12)) % 13)))
            for _ in range(2):
                x1, y1 = rng.random() * ww * 0.7, rng.random() * hh * 0.7
                bw, bh = (0.08 + 0.15 * rng.random()) * ww, (0.08 + 0.15 * rng.random()) * hh
                rows.append((np.array([x1, y1, x1 + bw, y1 + bh]), 0.02 + 0.2 * rng.random(), int(rng.integers(0, 13))))
            if ORACLE["degenerate"] == key:
                rows.append((np.array([10.0, 10.0, 10.2, 10.2]), 0.9, 3))
            rows.sort(key=lambda r: -r[1])
            if self.args.agnostic_nms or self.args.single_cls:
                seen, kept = set(), []
                for r in rows:
                    t = tuple(np.round(r[0], 6))
                    if t not in seen:
                        seen.add(t)
                        kept.append(r)
                rows = kept
            dev = real[si]["bboxes"].device
            out.append({"bboxes": torch.tensor(np.array([r[0] for r in rows], dtype=np.float32).reshape(-1, 4),
                                               device=dev),
                        "conf": torch.tensor([r[1] for r in rows], dtype=torch.float32, device=dev),
                        "cls": torch.tensor([float(r[2]) for r in rows], dtype=torch.float32, device=dev),
                        "extra": torch.zeros((len(rows), 0), device=dev)})
        return out
    DV.preprocess, DV.postprocess = pre, post


@contextlib.contextmanager
def no_oracle():
    old = ORACLE["on"]
    ORACLE["on"] = False
    try:
        yield
    finally:
        ORACLE["on"] = old


# ------------------------------------------------------------------ fixtures
def make_source(exp, wseed0, exams=("dev", "imageweeds"), research_only=True, manifest_sha=None, native=False):
    """An experiment of three done base runs (real checkpoints) and final runs scored by the locked scorer on
    `exams` (CPU, test mode); native: dev@640 files too (inc2.scorer_native)."""
    root = C.INC_DIR / exp
    if root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True)
    defn = {"exp": exp, "type": "baseline", "seeds": [0, 1, 2], "final_exams": list(exams),
            "research_only": {"flag": research_only}, "testing": {"batch": 32, "device": "cpu"}}
    if manifest_sha:
        defn["base"] = {"manifest_sha256": manifest_sha}
    (root / "exp.json").write_text(json.dumps(defn))
    for s in (0, 1, 2):
        rd = N.make_final(exp, s, wseed0 + s, protocol=False)
        w = root / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
        brj = {"status": "done", "testing": False, "weights_sha256": W.sha(w)}
        if manifest_sha:
            brj["train_manifest_sha256"] = manifest_sha
        (root / "runs" / ("base__s%d" % s) / "run.json").write_text(json.dumps(brj))
        for exam in exams:
            S.score(rd / "weights" / "final.pt", exam, rd / "scores" / ("%s.json" % exam), lock_check=True,
                    imgsz=640, batch=32, device="cpu")
        if native:
            rec = json.loads((rd / "scores" / "dev.json").read_text())
            js, _npz = SN.paths_for(rd / "scores", "dev", 640)
            SN.score(rd / "weights" / "final.pt", "dev", js, 640, lock_check=True, batch=32, device="cpu",
                     recorded=rec, extra={"exp": exp, "run_id": "final__base__s%d" % s, "arm": "m640"})
    return root


def write_e2_verdict(**over):
    refs = []
    for s in (0, 1, 2):
        js = C.INC_DIR / REF / "runs" / ("final__base__s%d" % s) / "scores" / "dev@640.json"
        d = json.loads(js.read_text())
        refs.append({"exp": REF, "run_id": "final__base__s%d" % s, "score": js.name, "sha256": W.sha(js),
                     "images_sha256": d["images"]["sha256"], "weights_sha256": d["weights_sha256"]})
    doc = {"format": B.E2_FORMAT, "name": B.E2_NAME, "rule": B.E2_RULE, "status": "decided",
           "reference": {"exp": REF, "arm": "m640", "seeds": [0, 1, 2]},
           "arms": {"W": {"status": "decided", "reference_inputs": refs, "qualifies": False},
                    "S": {"status": "decided", "reference_inputs": refs, "qualifies": False}},
           "qualifying": [], "chosen": None, "testing_allowed": False,
           "bootstrap": {"seed_text": B.E2_SEED_TEXT, "resamples": B.E2_RESAMPLES}}
    doc.update(over)
    p = C.INC_DIR / "capacity" / "e2_v1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def setup_world():
    _install_oracle()
    ORACLE["on"] = True
    msha = base_v2_sha()
    make_source(REF, 70, exams=("dev", "imageweeds", "test"), research_only=True, manifest_sha=msha, native=True)
    make_source("e1_a_m640", 80)
    make_source("e1_b_m640", 90)
    write_e2_verdict()


def reset_e3():
    """Remove everything E3 wrote (its directory, the pin, its capacity files)."""
    shutil.rmtree(str(TS.root().parent), ignore_errors=True)
    cap = C.INC_DIR / "capacity"
    for p in list(cap.glob("e3_*")) if cap.is_dir() else []:
        p.unlink()




@contextlib.contextmanager
def tamper(path, fn):
    """Rewrite a JSON file with fn(doc) for the block; its bytes are restored afterwards."""
    path = pathlib.Path(path)
    raw = path.read_bytes()
    d = json.loads(raw)
    fn(d)
    path.write_text(json.dumps(d))
    try:
        yield
    finally:
        path.write_bytes(raw)


@contextlib.contextmanager
def spy(mod, name, calls):
    fn = getattr(mod, name)

    def wrapped(*a, **k):
        calls.append(name)
        return fn(*a, **k)
    with patched(mod, name, wrapped):
        yield


def nothing_written():
    return not TS.root().parent.exists() and not TS.pin_path().exists() \
        and not list((C.INC_DIR / "capacity").glob("e3_*"))


def src_weights(arm, s):
    return C.INC_DIR / TS.ARMS[arm] / "runs" / ("base__s%d" % s) / "weights" / "final.pt"


# ------------------------------------------------------------------ 1. constants and units
def test_constants():
    print("E3's constants (pre-registered)")
    check("arms M b_v2_m640, A e1_a_m640, B e1_b_m640 in the order M, A, B; class-agnostic NMS for E3-M, the locked "
          "NMS for E3-A and E3-B (whose stage-1 AP is compared with the recorded one); E3-M's locked-NMS reading "
          "reported only; reference b_v2_m640; seeds 0-2; dev decides, imageweeds reported",
          TS.ARMS == {"M": "b_v2_m640", "A": "e1_a_m640", "B": "e1_b_m640"} and TS.ARM_ORDER == ("M", "A", "B")
          and TS.NMS == {"M": "agnostic", "A": "locked", "B": "locked"} and TS.REPORTED_NMS == {"M": "locked"}
          and TS.STAGE1_COMPARED == ("A", "B") and TS.reported_variant("M") == "locked_nms"
          and TS.REFERENCE == "b_v2_m640" and TS.SEEDS == (0, 1, 2) and TS.EXAM == "dev"
          and TS.REPORT_EXAMS == ("imageweeds",) and TS.IMGSZ == 640)
    check("seed texts inc2/e3/classifier, inc2/e3/species_se, inc2/e3/attribution_se; 1,000 resamples; top 3; the "
          "C grid 0.01..100; 5 folds; max_iter 3000; no class weights; the tolerances 0.002 (= the sidecar's), "
          "GT 1e-3, proba 1e-6; the crop-protocol check 100 boxes, median 0.99",
          TS.CLF_SEED_TEXT == "inc2/e3/classifier" and TS.SPECIES_SEED_TEXT == "inc2/e3/species_se"
          and TS.ATTR_SEED_TEXT == "inc2/e3/attribution_se" and TS.RESAMPLES == 1000 and TS.TOP_K == 3
          and TS.C_GRID == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0) and TS.CV_FOLDS == 5
          and TS.MAX_ITER == 3000 and TS.CLASS_WEIGHT is None and TS.EQ_TOL == SC.MAX_SCORE_DIFF == 0.002
          and TS.STAGE1_TOL == 0.002 and TS.GT_TOL == 1e-3 and TS.PROBA_TOL == 1e-6
          and (TS.PROTOCOL_CHECK_MIN, TS.PROTOCOL_CHECK_MEDIAN_COS) == (100, 0.99)
          and TS.REPORTED_SPECIES == ("Carpetweed", "SpottedSpurge", "Purslane"))
    check("EXIF orientations: a pass reads none, 0 and 1; a training image also 3, 6 and 8",
          TS.OK_ORIENTATIONS == (None, 0, 1) and TS.FIT_ORIENTATIONS == (None, 0, 1, 3, 6, 8))


def test_units():
    print("emit, the geometry, the probabilities")
    import torch
    P = np.zeros((4, 13))
    P[0, [2, 4, 7]] = [0.4, 0.4, 0.2]            # a tie between 2 and 4
    P[1, 12] = 0.9                               # OtherPlant first
    P[1, 0] = 0.1
    P[2, 5] = 1.0
    P[3, 1] = 1.0
    q = np.array([0.9, 0.5, 0.0015, 0.8], dtype=np.float32)
    bi, ci, sc, st = TS.emit(q, P, 3)
    by = {(int(b), int(c)): float(x) for b, c, x in zip(bi, ci, sc)}
    check("top 3 per box, equal p to the lower class id first, in emission order (box, then rank)",
          list(ci[bi == 0]) == [2, 4, 7] and list(bi) == sorted(bi), (bi, ci))
    check("the score is float32(q x p) and never fp16; OtherPlant is emitted", sc.dtype == np.float32
          and by[(0, 2)] == float(np.float32(np.float64(np.float32(0.9)) * 0.4)) and (1, 12) in by, (sc.dtype, by))
    check("rows whose score is not above the locked conf 0.001 are dropped (q 0.0015 x p 0 and x 1 kept only "
          "above it), and counted", (2, 5) in by and not any(b == 2 and c != 5 for b, c in by)
          and st["dropped_conf"] == sum(1 for r in range(4) for c in np.argsort(-P[r], kind="stable")[:3]
                                        if np.float32(q[r] * P[r, c]) <= np.float32(0.001)), st)
    Q = np.full(200, 0.5)
    PP = np.tile(np.eye(13)[0] * 0.5 + np.eye(13)[1] * 0.3 + np.eye(13)[2] * 0.2, (200, 1))
    bi, ci, sc, st = TS.emit(Q, PP, 3, max_det=300)
    check("at most max_det (300) rows per image, by score, ties to the lower box index then the lower class id; "
          "capped and counted", len(bi) == 300 and st["dropped_cap"] == 300 and st["capped"] == 1
          and set(ci) == {0, 1} and list(bi[ci == 1]) == list(range(100)) and list(bi[ci == 0]) == list(range(200)),
          (len(bi), st))
    conf = np.array([0.9, 0.7, 0.0011, 0.5], dtype=np.float32)
    cls = np.array([3, 12, 0, 3], dtype=np.float32)
    bi, ci, sc, st = TS.emit(conf, TS.onehot(cls), 1)
    check("identity: every row emitted as itself (k 1, one-hot), in order, its conf unchanged",
          list(bi) == [0, 1, 2, 3] and list(ci) == [3, 12, 0, 3] and np.array_equal(sc, conf)
          and st["dropped_conf"] == 0 and st["dropped_cap"] == 0, (bi, ci, sc))
    boxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float16)
    rp = TS._rows_pred(boxes, None, [1, 1, 0], [2, 3, 4], np.array([0.5, 0.25, 0.75], dtype=np.float32),
                       torch.float16)
    check("_rows_pred: the boxes' own rows (dtype kept), conf float32 never fp16, cls in the NMS dtype, extra [n, 0]",
          rp["conf"].dtype == torch.float32 and rp["bboxes"].dtype == torch.float16
          and rp["bboxes"].tolist() == [[5, 6, 7, 8], [5, 6, 7, 8], [1, 2, 3, 4]] and rp["cls"].dtype == torch.float16
          and tuple(rp["extra"].shape) == (3, 0), rp)
    orig = np.array([[10.0, 20.0, 150.0, 90.0], [0.0, 0.0, 200.0, 100.0]])
    rpad = ((0.5, 0.25), (3, 7))                 # gain_h, gain_w; pad_x, pad_y
    lb = np.column_stack([orig[:, 0] * 0.25 + 3, orig[:, 1] * 0.5 + 7, orig[:, 2] * 0.25 + 3, orig[:, 3] * 0.5 + 7])
    back = TS.to_original(lb, rpad, (100, 200))
    check("to_original inverts the letterbox per axis (gain 0.25 in x, 0.5 in y, pads 3 and 7) and clips",
          np.allclose(back, orig, atol=1e-9)
          and TS.to_original([[0, 0, 1000, 1000]], rpad, (100, 200)).tolist() == [[0.0, 0.0, 200.0, 100.0]], back)
    n = TS.normalise(orig, 200, 100)
    check("normalise gives (cx, cy, w, h); clip_label clips a label box to the image",
          np.allclose(n[0], [0.4, 0.55, 0.7, 0.7]) and np.allclose(TS.clip_label(0.95, 0.5, 0.2, 0.2),
                                                                   (0.925, 0.5, 0.15, 0.2)),
          (n, TS.clip_label(0.95, 0.5, 0.2, 0.2)))
    check("degenerate: a square side that rounds below 1 px; small: under 16 px on a side",
          TS.degenerate(0.5, 0.5, 1e-4, 1e-4, 96, 96) and not TS.degenerate(0.5, 0.5, 0.05, 0.05, 96, 96)
          and TS.small(0.1, 0.5, 96, 96) and not TS.small(0.2, 0.2, 96, 96))
    X = np.random.default_rng(0).normal(size=(5, 4))
    Wm = np.random.default_rng(1).normal(size=(3, 4))
    bb = np.array([0.1, -0.2, 0.3])
    Pp = TS.proba(X, Wm, bb, [0, 5, 12])
    check("proba: softmax over the fitted classes, 0 for a class the fit never saw, rows summing to 1",
          np.allclose(Pp.sum(1), 1) and (Pp[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]] == 0).all() and Pp.shape == (5, 13))


# ------------------------------------------------------------------ 2. fit-classifier: refusals (nothing written)
@contextlib.contextmanager
def swap_base_v2(rows, lock=True):
    """base_v2 replaced by `rows` with LOCK v2 (unless lock is False) and b_v2_m640's records naming its sha256
    (restored afterwards)."""
    d = W.v2_dir()
    keep = {p: p.read_bytes() for p in [d / "base_v2.jsonl", d / "LOCK.json", C.INC_DIR / REF / "exp.json"]
            + [C.INC_DIR / REF / "runs" / ("base__s%d" % s) / "run.json" for s in (0, 1, 2)]}
    try:
        sha = C.write_manifest(d / "base_v2.jsonl", rows)
        if lock:
            lk = json.loads((d / "LOCK.json").read_text())
            lk["manifests"]["base_v2"] = sha
            (d / "LOCK.json").write_text(json.dumps(lk, sort_keys=True))
        ex = json.loads((C.INC_DIR / REF / "exp.json").read_text())
        ex["base"]["manifest_sha256"] = sha
        (C.INC_DIR / REF / "exp.json").write_text(json.dumps(ex))
        for s in (0, 1, 2):
            p = C.INC_DIR / REF / "runs" / ("base__s%d" % s) / "run.json"
            rj = json.loads(p.read_text())
            rj["train_manifest_sha256"] = sha
            p.write_text(json.dumps(rj))
        yield sha
    finally:
        for p, b in keep.items():
            p.write_bytes(b)


def test_fit_refusals():
    print("fit-classifier: refusals, each writing nothing")
    rows = C.read_manifest(W.v2_dir() / "base_v2.jsonl")
    out = {}
    bp = W.v2_dir() / "base_v2.jsonl"
    raw = bp.read_bytes()
    try:
        bp.write_bytes(raw + b"\n")
        out["base_v2 not as LOCK v2 records"] = refused(TS.fit_classifier, embedder=EMB)
    finally:
        bp.write_bytes(raw)
    with swap_base_v2(rows[:-1], lock=False):
        out["base_v2 changed after LOCK v2 (b_v2_m640's records naming the new bytes)"] = refused(
            TS.fit_classifier, embedder=EMB)
    with tamper(C.INC_DIR / REF / "exp.json", lambda d: d["base"].update(manifest_sha256="0" * 64)):
        out["b_v2_m640 trained another manifest"] = refused(TS.fit_classifier, embedder=EMB)
    dev = C.read_manifest(C.manifest_path("dev"))[0]
    planted = W.planted("e3_dev_copy", dev["image"], "exact")
    with swap_base_v2(rows + [W.row_for("e3_dev_copy", planted, [(1, 0.5, 0.5, 0.3, 0.3)], "x", "s9")]):
        out["a dev image planted in base_v2 (the guard)"] = refused(TS.fit_classifier, embedder=EMB)
    for sub in ("test_v1", "test_v1_companions"):
        p = C.INC_DIR / "splits" / "v3" / sub / "listed.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        C.write_manifest(p, [dict(rows[3])])
        try:
            out["a base_v2 row in a %s list" % sub] = refused(TS.fit_classifier, embedder=EMB)
        finally:
            shutil.rmtree(str(C.INC_DIR / "splits" / "v3"))
    calls = []
    devm = C.manifest_path("dev")
    with patched(C2, "v2_manifest_path", lambda split: devm), \
            patched(C2, "verify_manifest_against_lock_v2", lambda split: W.sha(devm)), \
            patched(TS, "_reference_manifest_check", lambda sha, testing: {}), spy(T, "check_manifest", calls):
        out["an evaluation manifest (check_manifest)"] = refused(TS.fit_classifier, embedder=EMB)
    ok = all(e is not None for e in out.values()) and nothing_written()
    check("refused, writing nothing: %s" % "; ".join(out), ok and calls == ["check_manifest"],
          {k: str(v)[:160] for k, v in out.items()})
    check("  the guard refusal names the never-train guard; the test v1 refusal the list",
          "guard" in str(out["a dev image planted in base_v2 (the guard)"])
          and "test v1" in str(out["a base_v2 row in a test_v1 list"])
          and "test v1" in str(out["a base_v2 row in a test_v1_companions list"])
          and "is the v1 dev split" in str(out["an evaluation manifest (check_manifest)"])
          and "locked" in str(out["base_v2 changed after LOCK v2 (b_v2_m640's records naming the new bytes)"]),
          {k: str(v)[:200] for k, v in out.items()})
    after = {}
    for what, path in (("an arms directory", TS.root() / "arms" / "M"),
                       ("dev_gt.json", TS.classifier_dir() / "dev_gt.json"),
                       ("equivalence.json", TS.root() / "equivalence.json"),
                       ("capacity/e3_score_M.json", TS.score_record_path("M")),
                       ("the pin", TS.pin_path())):
        path.parent.mkdir(parents=True, exist_ok=True)
        if what == "an arms directory":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.write_text("{}")
        after[what] = refused(TS.fit_classifier, embedder=EMB)
        reset_e3()
    check("a fit is refused once E3 has read dev or a pin exists (%s), and wrote nothing" % ", ".join(after),
          all(e is not None and "fitted once" in str(e) for e in after.values()) and nothing_written(),
          {k: str(v)[:120] for k, v in after.items()})
    flag = {}
    cv0, fit0 = TS.cross_validate, TS._fit_lr

    def cv(*a, **k):
        r = cv0(*a, **k)
        flag["after_cv"] = True
        return r

    def fit(X, y, c):
        clf, conv = fit0(X, y, c)
        return clf, (False if flag.get("after_cv") else conv)
    with patched(TS, "cross_validate", cv), patched(TS, "_fit_lr", fit):
        e = refused(TS.fit_classifier, embedder=EMB)
    check("a final fit that does not converge is refused, writing nothing", e is not None and "converge" in str(e)
          and nothing_written(), e)


def test_cv_units():
    print("the cross-validation rule")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6))
    y = np.array([i % 3 for i in range(60)])
    X[np.arange(60), y] += 2.0
    groups = np.array(["g%d" % (i % 7) for i in range(60)])
    c, rec, models = TS.cross_validate(X, y, groups)
    losses = {r["C"]: r["pooled_log_loss"] for r in rec["per_c"] if r["eligible"]}
    best = min(losses.values())
    check("C is the argmin of the pooled held-out log-loss over the grid (%g)" % c,
          c == min(k for k, v in losses.items() if v <= best + 1e-12) and rec["folds"] == 5
          and set(models) == set(range(5)), losses)
    folds = V._assign_folds(groups, 5, 0, random_ok=False)
    check("no session in two folds (GroupKFold over the sessions)",
          all(len(set(folds[groups == g].tolist())) == 1 for g in set(groups)))
    fit0 = TS._fit_lr

    def same(Xa, ya, cc):
        return fit0(Xa, ya, 1.0)
    with patched(TS, "_fit_lr", same):
        c2, rec2, _m = TS.cross_validate(X, y, groups)
    check("equal losses: the smaller C (%g)" % c2, c2 == 0.01, [r["pooled_log_loss"] for r in rec2["per_c"]])

    def no_conv(Xa, ya, cc):
        clf, conv = fit0(Xa, ya, cc)
        return clf, (False if cc == c else conv)
    with patched(TS, "_fit_lr", no_conv):
        c3, rec3, _m = TS.cross_validate(X, y, groups)
    check("a C with a fold fit that does not converge is left out of the choice and recorded (%g left out, %g chosen)"
          % (c, c3), c3 != c and c in rec3["left_out"], rec3["left_out"])
    yy = y.copy()
    yy[groups == "g0"] = 11
    _c, rec4, _m = TS.cross_validate(X, yy, groups)
    f0 = [f for f, cnt in rec4["fold_class_counts"].items() if cnt["held_out"]["Goosegrass"] or
          cnt["held_out"]["CutleafGroundcherry"]]
    check("per-fold class counts recorded; a class whose only session is held out has 0 training boxes in that fold",
          all(rec4["fold_class_counts"][f]["train"]["CutleafGroundcherry"] == 0 for f in f0) and f0, f0)


class FakeCrops(object):
    def __init__(self, rows):
        self.key = [r["key"] for r in rows]
        self.box = np.array([r["box"] for r in rows])
        for f in ("cx", "cy", "w", "h"):
            setattr(self, f, np.array([r[f] for r in rows]))

    def where(self, name):
        return np.arange(len(self.key)) if name == "core" else np.zeros(0, dtype=np.int64)


def test_protocol_check():
    print("the crop-protocol check")
    core = C.read_manifest(C.manifest_path("train_core"))[0]
    rows = [{"key": core["key"], "box": b, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "sha256": core["sha256"]}
            for b in range(120)]
    rng = np.random.default_rng(5)
    X = V._norm(rng.normal(size=(120, 16)))
    same = X.astype(np.float16)
    other = rng.normal(size=(120, 16)).astype(np.float16)
    ok = TS.protocol_check(rows, X, False, crops=FakeCrops(rows), emb=same)
    e1 = refused(TS.protocol_check, rows, X, False, crops=FakeCrops(rows), emb=other)
    e2 = refused(TS.protocol_check, rows[:50], X[:50], False, crops=FakeCrops(rows[:50]), emb=same[:50])
    moved_ = [dict(r, cx=0.51) for r in rows]
    e3 = refused(TS.protocol_check, rows, X, False, crops=FakeCrops(moved_), emb=same)
    bad_sha = [dict(r, sha256="0" * 64) for r in rows]
    e4 = refused(TS.protocol_check, bad_sha, X, False, crops=FakeCrops(rows), emb=same)
    t = TS.protocol_check(rows[:50], X[:50], True, crops=FakeCrops(rows[:50]), emb=same[:50])
    check("production: 120 boxes matched by key, image sha256, box and geometry with the same features pass "
          "(median %.4f); other features (median < 0.99), fewer than 100 matched, a geometry off by 0.01, another "
          "image sha256 are refused; in a test world it records 'not compared'" % ok["median"],
          ok["passed"] and ok["matched"] == 120 and ok["median"] > 0.999 and all(x is not None for x in (e1, e2, e3, e4))
          and "median" in str(e1) and "matched 50" in str(e2) and "matched 0" in str(e3) and "matched 0" in str(e4)
          and t["compared"] is False, (ok, e1, e2, e3, e4, t))


# ------------------------------------------------------------------ 3. the fit, the pin, restore
EVAL_MANIFESTS = None
PATCH_BOX = (10, 60, 49, 109)                # the patch in the seen (EXIF-transposed) 80 x 120 frame, inclusive


def exif_image(key, tag, colour):
    """A training image stored with EXIF orientation `tag` whose EXIF-transposed frame (80 x 120, the frame its
    label is in) holds a `colour` patch at PATCH_BOX; its stored frame is that frame turned back. Returns (path,
    label boxes)."""
    from PIL import Image
    seen = Image.new("RGB", (80, 120), (120, 110, 90))
    seen.paste(colour, (PATCH_BOX[0], PATCH_BOX[1], PATCH_BOX[2] + 1, PATCH_BOX[3] + 1))
    back = {0: None, 2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
            6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}[tag]
    stored = seen.transpose(back) if back is not None else seen
    ex = Image.Exif()
    ex[0x0112] = tag
    p = W.TMP / "exif_train" / ("%s.jpg" % key)
    p.parent.mkdir(parents=True, exist_ok=True)
    stored.save(p, quality=95, exif=ex)
    x0, y0, x1, y1 = PATCH_BOX
    return p, [(2, (x0 + x1 + 1) / 2.0 / 80, (y0 + y1 + 1) / 2.0 / 120, (x1 - x0 + 1) / 80.0, (y1 - y0 + 1) / 120.0)]


def _centre(a):
    h, w = a.shape[:2]
    return a[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).astype(np.float64).mean(axis=0)


def test_fit_exif():
    print("fit-classifier: training images tagged 6, 3 and 0 (base_v2 holds 401, 156 and 54) are cut in their "
          "labels' frame, as verify._cut_task cuts them; another tag is refused")
    from PIL import Image, ImageOps
    rows = C.read_manifest(W.v2_dir() / "base_v2.jsonl")
    colour = (200, 30, 40)
    tagged = {}
    for tag in (6, 3, 0):
        img, boxes = exif_image("tsw23__exif%d" % tag, tag, colour)
        tagged[tag] = W.row_for("tsw23__exif%d" % tag, img, boxes, "3seasonweeddet10/data2023", "s_exif%d" % tag)
    with Image.open(tagged[6]["image"]) as im:
        seen6 = ImageOps.exif_transpose(im)
        fixture_ok = seen6.size == (80, 120) and im.size == (120, 80) \
            and np.abs(np.asarray(seen6.convert("RGB"))[85, 30].astype(int) - colour).max() < 40

    class Capture(StubEmbedder):
        def __init__(self):
            StubEmbedder.__init__(self)
            self.crops = []

        def __call__(self, pils):
            self.crops += [np.asarray(p) for p in pils]
            return StubEmbedder.__call__(self, pils)
    cap = Capture()
    with swap_base_v2(rows + [tagged[t] for t in (6, 3, 0)]):
        e0 = refused(TS.fit_classifier, embedder=cap)
        rec = json.loads((TS.classifier_dir() / "classifier.json").read_text()) if e0 is None else {}
        crops_csv = list(__import__("csv").DictReader(open(TS.classifier_dir() / "train_crops.csv"))) \
            if e0 is None else []
    reset_e3()
    check("a base_v2 holding images tagged 6, 3 and 0 is fitted, not refused", e0 is None, e0)
    by_key = {r["key"]: int(r["crop_id"]) for r in crops_csv}
    res = {}
    for tag, row in tagged.items():
        cx, cy, w, h = T.read_label_strict(pathlib.Path(row["label"]))[0][1:]
        want = V._cut_task((row["image"], [{"crop_id": 0, "cx": cx, "cy": cy, "w": w, "h": h}]))[2][0]
        got = cap.crops[by_key[row["key"]]] if row["key"] in by_key else None
        with Image.open(row["image"]) as im:
            stored = im.convert("RGB")
        raw = np.asarray(TS.SL._cut(stored, {"cx": cx, "cy": cy, "w": w, "h": h, "W": stored.size[0],
                                             "H": stored.size[1]}))
        res[tag] = {"same_as_cut_task": got is not None and np.array_equal(got, want),
                    "on_patch": got is not None and float(np.abs(_centre(got) - colour).max()) < 40,
                    "stored_frame_off_patch": float(np.abs(_centre(raw) - colour).max()) > 60 if tag else None}
    check("the fixtures: tag 6's stored frame is 120 x 80, its EXIF-transposed frame 80 x 120 with the patch under "
          "its label", fixture_ok)
    check("fitted with the tagged images: each one's crop is verify._cut_task's array, bit for bit, and its centre is "
          "the patch under its label (the EXIF-transposed frame); for tags 6 and 3 the stored frame cut at the same "
          "box is not",
          all(r["same_as_cut_task"] and r["on_patch"] for r in res.values())
          and res[6]["stored_frame_off_patch"] and res[3]["stored_frame_off_patch"], res)
    check("  classifier.json counts the images and training boxes per tag",
          e0 is None and rec["orientations"].get("6") == 1 and rec["orientations"].get("3") == 1
          and rec["orientations"].get("0") == 1 and rec["orientation_train_boxes"].get("6") == 1
          and rec["orientation_train_boxes"].get("3") == 1 and rec["orientation_train_boxes"].get("0") == 1
          and rec["orientations"].get("None") == len(rows) and "exif_transpose" in rec["orientation_rule"],
          (rec.get("orientations"), rec.get("orientation_train_boxes")))
    img2, boxes2 = exif_image("tsw23__exif2", 2, colour)
    with swap_base_v2(rows + [W.row_for("tsw23__exif2", img2, boxes2, "3seasonweeddet10/data2023", "s_exif2")]):
        e = refused(TS.fit_classifier, embedder=EMB)
    check("a training image with another tag (2, a mirror) refuses the fit, writing nothing",
          e is not None and "EXIF orientation" in str(e) and nothing_written(), e)


def test_fit():
    print("fit-classifier: the fit and its record")
    calls, opened, read = [], [], []
    real_open = builtins.open
    real_read = C.read_manifest

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_read(path):
        read.append(str(path))
        return real_read(path)
    with spy(T, "check_manifest", calls), spy(T, "guard_rows", calls), \
            spy(C2, "verify_manifest_against_lock_v2", calls), spy(TS.B3, "prior_test_lists", calls), \
            patched(builtins, "open", spy_open), patched(C, "read_manifest", spy_read):
        rec = TS.fit_classifier(embedder=EMB)
    evals = {str(p.resolve()) for _v, _s, p in T.eval_manifest_paths()}
    check("the fit ran check_manifest, guard_rows, LOCK v2's base_v2 check and the test v1 lists",
          set(calls) == {"check_manifest", "guard_rows", "verify_manifest_against_lock_v2", "prior_test_lists"},
          calls)
    check("no exam image or label was opened and no dev, test or imageweeds manifest was read (row by row)",
          not [p for p in opened if p.startswith(str(C.EXAMS_DIR))]
          and not [p for p in read if str(pathlib.Path(p).resolve()) in evals],
          ([p for p in opened if p.startswith(str(C.EXAMS_DIR))][:3], read))
    d = TS.classifier_dir()
    pin = json.loads(TS.pin_path().read_text())
    z = np.load(str(d / "classifier.npz"))
    emb = np.load(str(d / "train_emb.npz"))
    check("classifier.json: base_v2 by sha256 (LOCK v2 and b_v2_m640's), a clean guard, no test v1 hit, every box "
          "counted, the crop protocol, the CV block, the fit, the code; the pin names its files",
          rec["manifest"]["sha256"] == base_v2_sha() == rec["reference_manifest"]["exp_manifest_sha256"]
          and rec["guard"]["refused"] == 0 and rec["guard"]["crosscheck_hits"] == 0 and rec["test_v1"]["hits"] == 0
          and rec["boxes"] == rec["train_boxes"] + rec["skipped_small"]["total"]
          and rec["crop_protocol"]["crop_px"] == 224 and rec["crop_protocol"]["min_box_px"] == 16
          and rec["cv"]["chosen_C"] == rec["fit"]["C"] and rec["fit"]["converged"]
          and rec["fit"]["proba_max_abs_diff"] <= 1e-6 and rec["crop_protocol_check"]["compared"] is False
          and pin["json_sha256"] == W.sha(d / "classifier.json") and pin["npz_sha256"] == W.sha(d / "classifier.npz")
          and pin["train_boxes_sha256"] == rec["train_boxes_sha256"] and pin["written_utc"] == rec["written_utc"]
          and z["W"].shape == (len(z["classes"]), 16) and emb["F"].dtype == np.float32
          and emb["F"].shape == (rec["train_boxes"], 16), rec.get("cv", {}).get("chosen_C"))
    from sklearn.linear_model import LogisticRegression
    Xs = V._norm(emb["F"]).astype(np.float64)
    y = np.array([int(r["label"]) for r in __import__("csv").DictReader(open(d / "train_crops.csv"))])
    clf = LogisticRegression(C=rec["fit"]["C"], max_iter=3000).fit(Xs, y)
    check("the stored fp32 features reproduce the fit (refit coefficients within 1e-6 of the stored ones)",
          np.abs(clf.coef_ - z["W"]).max() < 1e-6 and np.abs(clf.intercept_ - z["b"]).max() < 1e-6,
          np.abs(clf.coef_ - z["W"]).max())
    npz0 = W.sha(d / "classifier.npz")
    reset_e3()
    rec2 = TS.fit_classifier(embedder=EMB)
    check("deterministic: a second fit in a fresh directory gives the same npz sha256 and C",
          W.sha(d / "classifier.npz") == npz0 and rec2["fit"]["C"] == rec["fit"]["C"]
          and rec2["train_boxes_sha256"] == rec["train_boxes_sha256"])
    e = refused(TS.fit_classifier, embedder=EMB)
    check("a second fit is refused (fitted once)", e is not None and "fitted once" in str(e), e)
    return rec2


def test_pin_restore():
    print("the pin: a moved-aside E3 directory is restored, never refitted")
    r = TS.root()
    aside = r.with_name(r.name + ".moved")
    os.replace(str(r), str(aside))
    try:
        e1 = refused(TS.fit_classifier, embedder=EMB)
        e2 = refused(TS.score_arm, "M", embedder=EMB, **KW)
        e3 = refused(TS.load_classifier)
        bad = aside.with_name("bad_copy")
        shutil.copytree(str(aside), str(bad))
        (bad / "classifier" / "classifier.npz").write_bytes(b"x" + (bad / "classifier" / "classifier.npz").read_bytes())
        e4 = refused(TS.restore_classifier, bad)
        shutil.rmtree(str(bad))
        TS.restore_classifier(aside)
        ok = TS.load_classifier()[5]["npz_sha256"] == json.loads(TS.pin_path().read_text())["npz_sha256"]
    finally:
        shutil.rmtree(str(r), ignore_errors=True)
        os.replace(str(aside), str(r))
    check("E3's directory moved aside: a refit is refused (the pin), score-arm refuses (restore it), the classifier "
          "cannot be loaded, a copy that does not hash as pinned is refused; restore-classifier copies the pinned "
          "files back", all(x is not None for x in (e1, e2, e3, e4)) and "fitted once" in str(e1)
          and "restore" in str(e2) and "pin" in str(e4) and ok, (e1, e2, e3, e4))
    d = TS.classifier_dir()
    with tamper(d / "classifier.json", lambda x: x.update(train_boxes=1)):
        e = refused(TS.load_classifier)
    check("a classifier.json that does not hash as pinned is refused", e is not None and "pin" in str(e), e)


# ------------------------------------------------------------------ 4. crops and headers
def test_crops():
    print("crops: verify._cut_task's, the EXIF header")
    from PIL import Image
    row = C.read_manifest(C.manifest_path("train_core"))[0]
    rows = [{"crop_id": 0, "cx": 0.4, "cy": 0.5, "w": 0.3, "h": 0.2}, {"crop_id": 1, "cx": 0.02, "cy": 0.98,
                                                                         "w": 0.3, "h": 0.4}]
    a = TS._e3_cut((row["image"], rows))
    b = V._cut_task((row["image"], rows))
    check("a crop is verify._cut_task's array, bit for bit (one inside, one off-frame and padded grey)",
          a[3] is None and all(np.array_equal(x, y) for x, y in zip(a[2], b[2])) and a[4] == (None, (96, 96))
          and len(a[2]) == 2 and a[2][0].shape == (224, 224, 3), a[3])
    p = W.TMP / "exif6.jpg"
    im = Image.new("RGB", (60, 40), (10, 200, 30))
    ex = Image.Exif()
    ex[0x0112] = 6
    im.save(p, exif=ex)
    check("header: an EXIF orientation 6 image reads as tag 6 and its transposed size (40 x 60)",
          TS.header(p) == (6, (40, 60)) and TS.header(row["image"])[0] is None, TS.header(p))


# ------------------------------------------------------------------ 5. dev ground truth (the same geometry)
def test_dev_gt():
    print("dev ground truth: the classifier on dev's GT crops, cut through the detected boxes' geometry")
    rec = TS.dev_gt_accuracy(embedder=EMB, keep_crops=True, **KW)
    crops = rec.pop("crops")
    same, n = True, 0
    for r in C.read_manifest(C.manifest_path("dev")):
        lab = C.EXAMS_DIR / "dev" / "labels" / ("%s.txt" % r["key"])
        boxes = T.read_label_strict(lab)
        rows = [{"crop_id": j, "cx": b[1], "cy": b[2], "w": b[3], "h": b[4]} for j, b in enumerate(boxes)
                if not TS.small(b[3], b[4], 96, 96)]
        if not rows:
            continue
        img = C.EXAMS_DIR / "dev" / "images" / pathlib.Path(r["image"]).name
        _i, ids, arrs, _e = V._cut_task((str(img), rows))
        for c, a in zip(ids, arrs):
            n += 1
            same = same and (r["key"], c) in crops and np.array_equal(crops[(r["key"], c)], a)
    on = json.loads((TS.classifier_dir() / "dev_gt.json").read_text())
    check("its %d crops equal verify._cut_task's of the exam labels' boxes, bit for bit; top-1/top-3, the "
          "confusion and the round trip recorded; the pass reproduces the recorded dev score" % n,
          same and n == rec["n"] > 0 and rec["gt_roundtrip_max_diff"] < 1e-5 and len(on["confusion"]) == 13
          and 0 <= on["top1"] <= on["top3"] <= 1 and on["pass"]["vs_recorded"] == 0.0, (n, rec["n"], on["pass"]))
    e = refused(TS.dev_gt_accuracy, embedder=EMB, **KW)
    check("written once", e is not None and "once" in str(e), e)


# ------------------------------------------------------------------ 6. the validator: layering and identity
def two_stage_cfg(nms="locked", embedder=EMB, **over):
    ccfg, _rec, _shas = TS._clf_cfg()
    return dict(ccfg, mode="two_stage", nms=nms, embedder=embedder, workers=V._Workers(1), top_k=3, **over)


def test_layering_and_identity():
    print("the validator: one E3 layer, restored after every pass; identity reproduces the locked scorer exactly")
    base = S.validator_class()
    w = src_weights("M", 0)
    ps = TS._score_pass(w, "dev", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
    ok1 = S._VALIDATOR is base
    ps2 = TS._score_pass(w, "dev", two_stage_cfg(), **KW)
    ok2 = S._VALIDATOR is base and ps2["stage1"] is not None

    class Boom(StubEmbedder):
        def __call__(self, pils):
            raise RuntimeError("boom")
    e = refused(TS._score_pass, w, "dev", two_stage_cfg(embedder=Boom()), **KW)
    ok3 = S._VALIDATOR is base
    ps3 = TS._score_pass(w, "dev", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
    check("two passes in one process (identity, then two-stage), a pass whose embedder fails (refused, the "
          "validator restored), then another pass: the scorer's validator is the same class after each",
          ok1 and ok2 and ok3 and e is not None and "non-finite" in str(e)
          and ps3["result"]["species_map50_95"] == ps["result"]["species_map50_95"], (ok1, ok2, ok3, e))
    cls, b2 = TS.e3_validator_class({"mode": "identity", "nms": "locked"})
    layers = [k for k in cls.__mro__ if k.__dict__.get("_e3_layer")]
    with patched(S, "_VALIDATOR", cls):
        e2 = None
        try:
            TS.e3_validator_class({"mode": "identity", "nms": "locked"})
        except RuntimeError as x:
            e2 = x
    check("exactly one E3 layer over the scorer's own validator (via the sidecar's); building one over an E3 "
          "validator is refused", len(layers) == 1 and cls.__mro__.count(b2) == 1 and b2 is base
          and cls.__mro__[1].__mro__[1] is base and e2 is not None, (layers, e2))
    calls = []
    with spy(TS, "emit", calls), spy(TS, "_rows_pred", calls):
        TS._score_pass(w, "dev", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
    n_dev = len(C.read_manifest(C.manifest_path("dev")))
    check("identity mode emits every image's rows through emit and _rows_pred (the two-stage path's own code), once "
          "per image", calls.count("emit") == n_dev and calls.count("_rows_pred") == n_dev, (calls.count("emit"),
                                                                                             calls.count("_rows_pred")))
    for label, ctx in (("the oracle's predictions", contextlib.nullcontext()), ("an untrained model's real NMS "
                                                                               "output", no_oracle())):
        with ctx:
            ref = S.score(w, "dev", W.TMP / "e3_ref_score.json", **KW)
            ps = TS._score_pass(w, "dev", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
        r = ps["result"]
        same = all(r[k] == ref[k] for k in ("map50_95", "map50", "per_class", "species_map50_95",
                                            "agnostic_map50_95", "image_correct", "n_gt", "scorer_sha256"))
        check("identity on %s: the locked scorer's score exactly (mAP %.4f, species %.4f, agnostic %.4f), the "
              "stage-1 helper its agnostic AP exactly, nothing dropped" % (label, r["map50_95"],
                                                                           r["species_map50_95"] or 0,
                                                                           r["agnostic_map50_95"]),
              same and ps["stage1_agnostic"][0] == r["agnostic_map50_95"] and ps["stats"]["dropped_conf"] == 0
              and ps["stats"]["dropped_cap"] == 0, (r["map50_95"], ref["map50_95"], ps["stage1_agnostic"]))
    rec = json.loads((C.INC_DIR / REF / "runs" / "final__base__s0" / "scores" / "dev.json").read_text())
    check("  the oracle's recorded dev scores are not trivial (agnostic %.4f, species %.4f)"
          % (rec["agnostic_map50_95"], rec["species_map50_95"]),
          rec["agnostic_map50_95"] > 0.3 and rec["species_map50_95"] > 0.1)


def test_nms_and_cap():
    print("the locked NMS against agnostic NMS, and the max_det cap (an untrained model's real NMS output)")
    w = src_weights("M", 1)
    with no_oracle():
        a = TS._score_pass(w, "dev", two_stage_cfg(), **KW)
        b = TS._score_pass(w, "dev", two_stage_cfg(nms="agnostic"), **KW)
    per_img = np.bincount(a["arrays"]["pred_img"], minlength=len(a["arrays"]["keys"]))
    check("agnostic NMS reads another box set than the locked NMS (%d vs %d stage-1 boxes): E3-M's deciding boxes "
          "are the agnostic ones, its locked ones a reported reading" % (b["stats"]["stage1_boxes"],
                                                                        a["stats"]["stage1_boxes"]),
          a["stats"]["stage1_boxes"] != b["stats"]["stage1_boxes"], (a["stats"], b["stats"]))
    check("at most 300 rows per image after top 3 (%d images capped, %d rows dropped by the cap, %d by conf)"
          % (a["stats"]["capped"], a["stats"]["dropped_cap"], a["stats"]["dropped_conf"]),
          per_img.max() <= 300 and a["stats"]["capped"] > 0 and a["stats"]["dropped_cap"] > 0
          and a["result"]["settings"]["max_det"] == 300, (per_img.max(), a["stats"]))


def test_pass_refusals():
    print("a two-stage pass refuses: an EXIF orientation, another size, a geometry that loses the GT boxes")
    w = src_weights("A", 0)
    h0 = TS.header
    out = {}
    with patched(TS, "header", lambda p: (6, h0(p)[1])):
        out["EXIF orientation 6"] = refused(TS._score_pass, w, "dev", two_stage_cfg(), **KW)
    with patched(TS, "header", lambda p: (None, (95, 96))):
        out["a size other than ori_shape"] = refused(TS._score_pass, w, "dev", two_stage_cfg(), **KW)
    t0 = TS.to_original
    with patched(TS, "to_original", lambda xy, rp, os_: t0(xy, rp, os_) + 2.0):
        out["a map to the original image 2 px off"] = refused(TS._score_pass, w, "dev", two_stage_cfg(), **KW)
    with patched(TS, "header", lambda p: (0, h0(p)[1])):
        ps0 = TS._score_pass(w, "dev", two_stage_cfg(), **KW)
    n_dev = len(C.read_manifest(C.manifest_path("dev")))
    check("EXIF orientation 0 (invalid) reads as no rotation in a pass (%d images)" % n_dev,
          ps0["stats"]["orientations"] == {"0": n_dev}, ps0["stats"]["orientations"])
    e = refused(TS._score_pass, w, "test", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
    check("refused: %s; and test outside score-test" % "; ".join(out),
          "EXIF orientation" in str(out["EXIF orientation 6"]) and "ori_shape" in str(out["a size other than ori_shape"])
          and "geometry" in str(out["a map to the original image 2 px off"]) and e is not None
          and "score-test" in str(e) and S._VALIDATOR is S.validator_class(), {k: str(v)[:150] for k, v in out.items()})
    with patched(TS, "_TEST_TOKENS", [object()]):
        e = refused(TS._score_pass, w, "test", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)},
                    test_token=object(), **KW)
    check("  a token that is not score-test's own is refused", e is not None, e)
    ORACLE["degenerate"] = C.read_manifest(C.manifest_path("dev"))[2]["key"]
    try:
        ps = TS._score_pass(w, "dev", two_stage_cfg(), **KW)
    finally:
        ORACLE["degenerate"] = None
    s1 = ps["stage1"]
    i = int(np.flatnonzero(s1["degenerate"])[0]) if s1["degenerate"].any() else None
    prior = TS.load_classifier()[3]
    check("a sub-pixel stage-1 box is not cut: it takes the training class prior (counted degenerate)",
          ps["stats"]["degenerate"] == 1 and i is not None
          and np.allclose(s1["top_p"][i], np.sort(prior)[::-1][:3], atol=1e-6), (ps["stats"].get("degenerate"), i))


# ------------------------------------------------------------------ 7. equivalence
def test_equivalence():
    print("equivalence: b_v2_m640's own predictions through E3's path")
    rec_dev = C.INC_DIR / REF / "runs" / "final__base__s1" / "scores" / "dev.json"
    with tamper(rec_dev, lambda d: d.update(map50_95=d["map50_95"] + 0.01)):
        e = refused(TS.equivalence, **KW)
    check("a recorded dev score the identity path does not reproduce (0.01 off): refused, nothing written",
          e is not None and "seed 1" in str(e) and not (TS.root() / "equivalence.json").exists(), e)
    with tamper(rec_dev, lambda d: d.update(species_map50_95=d["species_map50_95"] + 0.003)):
        e = refused(TS.equivalence, **KW)
    check("  a recorded species mean 0.003 off (the mAP and every class as reproduced): refused",
          e is not None and "species mean" in str(e) and not (TS.root() / "equivalence.json").exists(), e)
    rec = TS.equivalence(**KW)
    check("written once, every seed compared with its recorded protocol dev score (diff 0 on the CPU), the stage-1 "
          "helper equal to the scorer's agnostic AP, the code's sha256s recorded",
          rec["all_passed"] and all(r["vs_protocol_score"]["compared"] and r["vs_protocol_score"]["max_abs_diff"] == 0
                                    and r["species_diff"] == 0 and r["stage1_vs_identity_agnostic"] == 0
                                    for r in rec["per_seed"].values())
          and set(rec["code"]) == {"twostage", "scorer", "scorer_sidecar", "scorer_native", "verify", "semisup_labeler"},
          rec["per_seed"])
    e = refused(TS.equivalence, **KW)
    with patched(TS, "_code_shas", lambda: dict(rec["code"], twostage="0" * 64)):
        e2 = refused(TS.check_equivalence)
        e3 = refused(TS.score_arm, "A", embedder=EMB, **KW)
    check("a second run is refused (written once); code other than the one it ran refuses score-arm before any pass",
          e is not None and e2 is not None and "move twostage" in str(e2) and e3 is not None
          and not (TS.root() / "arms").exists(), (e, e2, e3))
    with tamper(TS.root() / "equivalence.json", lambda d: d.update(all_passed=False)):
        e4 = refused(TS.check_equivalence)
        e5 = refused(TS.score_arm, "A", embedder=EMB, **KW)
    check("  an equivalence record that did not pass refuses score-arm before any pass", e4 is not None
          and e5 is not None and "did not pass" in str(e4) and not (TS.root() / "arms").exists(), (e4, e5))


# ------------------------------------------------------------------ 8. score-arm
def test_score_arm():
    print("score-arm: refusals before any pass, then M, A, B end to end")
    out = {}
    rj = C.INC_DIR / "e1_a_m640" / "runs" / "base__s2" / "run.json"
    with tamper(rj, lambda d: d.update(status="running")):
        out["a source run not done"] = refused(TS.score_arm, "A", embedder=EMB, **KW)
    w = src_weights("A", 1)
    raw = w.read_bytes()
    try:
        w.write_bytes(raw + b"x")
        out["weights that do not hash"] = refused(TS.score_arm, "A", embedder=EMB, **KW)
    finally:
        w.write_bytes(raw)
    fj = C.INC_DIR / "e1_a_m640" / "runs" / "final__base__s0" / "run.json"
    with tamper(fj, lambda d: d.update(weights_sha256="0" * 64)):
        out["final weights that differ"] = refused(TS.score_arm, "A", embedder=EMB, **KW)
    rd = C.INC_DIR / "e1_a_m640" / "runs" / "final__base__s0" / "scores" / "dev.json"
    with tamper(rd, lambda d: d.update(agnostic_map50_95=d["agnostic_map50_95"] + 0.003)):
        out["a stage-1 agnostic AP 0.003 from the recorded one"] = refused(TS.score_arm, "A", embedder=EMB, **KW)
    check("refused, writing no score of E3-A: %s" % "; ".join(out),
          all(e is not None for e in out.values()) and not (TS.root() / "arms" / "A").exists()
          and not TS.score_record_path("A").exists() and "stage-1" in str(out[
              "a stage-1 agnostic AP 0.003 from the recorded one"]), {k: str(v)[:150] for k, v in out.items()})
    e = refused(TS.score_arm, "X", embedder=EMB, **KW)
    e2 = refused(TS.score_arm, "M", seeds=[0, 3], embedder=EMB, **KW)
    check("another arm, or a seed outside 0-2, is refused", e is not None and e2 is not None)
    t0 = time.time()
    recm = TS.score_arm("M", embedder=EMB, **KW)
    files = tree(TS.root() / "arms" / "M")
    check("E3-M (%.0fs): three dev files with their arrays and stage-1 records, the locked-NMS reading and ImageWeeds "
          "reported, the record complete (dev only, names and sha256s, no path)" % (time.time() - t0),
          recm["status"] == "complete" and [f["status"] for f in recm["dev"]] == ["written"] * 3
          and all("s%d/dev.json" % s in files and "s%d/dev.stage1.npz" % s in files
                  and "s%d/dev.locked_nms.json" % s in files and "s%d/imageweeds.json" % s in files for s in (0, 1, 2))
          and recm["reported"] == {"status": "complete", "passes": 6, "failed": 0}
          and not BP.dev_leaks(recm) and str(C.INC_DIR) not in json.dumps(recm), (recm, files))
    d0 = json.loads((TS.arm_dir("M", 0) / "dev.json").read_text())
    check("an E3-M dev file: stamped E3- (TEST- here), production false, its stage-1 weights the base run's, "
          "class-agnostic NMS (settings agnostic_nms true), its stage-1 agnostic AP recorded beside and not compared, "
          "the classifier's sha256s, the emission's counts",
          d0["scorer_sha256"].startswith("TEST-E3-") and d0["production"] is False and d0["e3_production"] is False
          and d0["stage1"]["weights_sha256"] == W.sha(src_weights("M", 0)) and d0["stage1"]["nms"] == "agnostic"
          and d0["stage1"]["vs_recorded_agnostic"]["compared"] is False
          and "agnostic NMS" in d0["stage1"]["vs_recorded_agnostic"]["why"]
          and d0["stage2"]["classifier_npz_sha256"] == TS.load_classifier()[5]["npz_sha256"]
          and d0["emitted"]["rows"] > 0 and d0["settings"]["nms"] == "agnostic"
          and d0["settings"]["agnostic_nms"] is True and d0["settings"]["top_k"] == 3, d0)
    lk = json.loads((TS.arm_dir("M", 0) / "dev.locked_nms.json").read_text())
    check("  the locked-NMS reading (reported): variant locked_nms, settings agnostic_nms false, its stage-1 box set "
          "the one the recorded agnostic dev score was computed on (its AP the recorded one exactly, recorded beside, "
          "never checked)",
          lk["variant"] == "locked_nms" and lk["settings"]["agnostic_nms"] is False and lk["stage1"]["nms"] == "locked"
          and lk["stage1"]["vs_recorded_agnostic"]["compared"] is False
          and lk["stage1"]["vs_recorded_agnostic"]["abs_diff"] == 0, lk["stage1"])
    s0 = W.sha(TS.arm_dir("M", 0) / "dev.json")
    rec2 = TS.score_arm("M", embedder=EMB, **KW)
    check("a second run keeps every file (written once)", [f["status"] for f in rec2["dev"]] == ["kept"] * 3
          and W.sha(TS.arm_dir("M", 0) / "dev.json") == s0)
    real = TS._score_pass

    def fail_iw(weights, exam, cfg, **k):
        if exam == "imageweeds":
            raise TS.E3Refused("planted ImageWeeds failure")
        return real(weights, exam, cfg, **k)
    with patched(TS, "_score_pass", fail_iw):
        reca = TS.score_arm("A", embedder=EMB, **KW)
    rep = json.loads((TS.root() / "arms" / "A" / "reported.json").read_text())
    check("E3-A with every ImageWeeds pass failing: the dev files and the record complete, the failures recorded "
          "(3), the job not failed", reca["status"] == "complete" and [f["status"] for f in reca["dev"]] == ["written"] * 3
          and reca["reported"]["failed"] == 3 and [p["status"] for p in rep["passes"]] == ["failed"] * 3,
          (reca, rep))
    reca2 = TS.score_arm("A", embedder=EMB, **KW)
    check("  a rerun writes the reported passes, keeps the dev files", reca2["reported"]["failed"] == 0
          and [f["status"] for f in reca2["dev"]] == ["kept"] * 3)
    TS.score_arm("B", embedder=EMB, **KW)
    check("E3-B scored", TS.score_record_path("B").exists())
    a0 = json.loads((TS.arm_dir("A", 0) / "dev.json").read_text())
    check("an E3-A dev file: the locked NMS (settings agnostic_nms false), its stage-1 agnostic AP compared with the "
          "recorded one (diff 0 on the CPU)", a0["stage1"]["nms"] == "locked" and a0["settings"]["agnostic_nms"] is False
          and a0["stage1"]["vs_recorded_agnostic"]["compared"] is True
          and a0["stage1"]["vs_recorded_agnostic"]["max_abs_diff"] == 0, a0["stage1"])


# ------------------------------------------------------------------ 9. the verdict on the real files
def test_verdict_real():
    print("the verdict on the real CPU files")
    out = C.INC_DIR / "cap_e3"
    opened = []
    real_open = builtins.open

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)
    planted = []
    for s in (0, 1, 2):
        p = TS.arm_dir("B", s) / "test.json"
        p.write_text(json.dumps({"species_map50_95": 0.99}))
        planted.append(p)
    try:
        with patched(builtins, "open", spy_open):
            d, rep = TS.verdict(out_dir=out, testing_ok=True, resamples=30)
    finally:
        for p in planted:
            p.unlink()
    check("decided on the dev files: every arm decided, D, pooled sd, SE, the conditions; the attribution B - A under "
          "its own seed text; the reference's files the ones capacity/e2_v1.json lists",
          d["status"] == "decided" and all(d["arms"][k]["status"] == "decided" for k in "MAB")
          and d["attribution"]["status"] == "decided"
          and d["attribution"]["bootstrap"]["seed_text"] == "inc2/e3/attribution_se"
          and d["bootstrap"]["seed_text"] == "inc2/e3/species_se"
          and all(len(d["arms"][k]["inputs"]) == 3 for k in "MAB"), d.get("arms"))
    check("each arm's inputs and the reference's are paired by seed (seed s of the arm with final__base__s<s>), and "
          "E3-M seed s's stage-1 weights are the reference's seed-s weights",
          all([r["seed"] for r in d["arms"][k]["inputs"]] == [0, 1, 2]
              and [r["run_id"] for r in d["arms"][k]["reference_inputs"]] == ["final__base__s%d" % i for i in (0, 1, 2)]
              for k in "MAB")
          and [r["weights_sha256"] for r in d["arms"]["M"]["inputs"]]
          == [r["weights_sha256"] for r in d["arms"]["M"]["reference_inputs"]], d["arms"]["M"]["inputs"])
    check("the planted arms/*/s*/test.json (which would reverse any choice) were never opened; no exam test or "
          "imageweeds score was read by the decision",
          not [p for p in opened if p.endswith("/test.json") or "/test/" in p or p.endswith("scores/test.json")],
          [p for p in opened if "test" in p][:5])
    m = d["arms"]["M"]
    check("reported beside: each arm's stage-1 agnostic mean, E3-M's paired per-seed differences and its "
          "locked-NMS reading, the three species with their SE, the dev ground-truth accuracy, the research-only "
          "flags", m["reported"]["stage1_agnostic"]["mean"] is not None
          and len(m["reported"]["paired_per_seed"]["diffs"]) == 3
          and all("species_map50_95" in x for x in m["reported"]["locked_nms_reading"])
          and set(m["reported"]["species"]) == {"Carpetweed", "SpottedSpurge", "Purslane"}
          and d["reported"]["dev_gt"]["n"] > 0 and m["reported"]["research_only"] is True, m["reported"])
    vj = json.loads((out / "e3_v1.json").read_text())
    check("capacity/e3_v1.json is dev only (brain_plan.dev_leaks) and names no path; the report holds ImageWeeds",
          BP.dev_leaks(vj) == [] and str(C.INC_DIR) not in json.dumps(vj)
          and all(rep["imageweeds"]["E3-%s" % k]["n"] == 3 for k in "MAB") and rep["imageweeds"][REF]["n"] == 3,
          BP.dev_leaks(vj)[:5])
    v0 = W.sha(out / "e3_v1.json")
    d2, _r = TS.verdict(out_dir=out, testing_ok=True, resamples=30)
    e = refused(TS.verdict, out_dir=out, testing_ok=True, resamples=20)
    check("a recomputation that agrees keeps the decided file byte for byte; other resamples are refused",
          d2.get("kept") and W.sha(out / "e3_v1.json") == v0 and e is not None and "never rewritten" in str(e), e)
    e = refused(TS.e3_decision, testing_ok=False, resamples=10)
    check("production refuses the test-mode files", e is not None and "test-mode" in str(e), e)
    return out


def _arm_file(arm, s):
    return TS.arm_dir(arm, s) / "dev.json"


def test_verdict_refusals():
    print("the verdict's refusals (each on one changed record, restored after)")
    out = {}
    f = _arm_file("A", 1)

    def run():
        return refused(TS.e3_decision, testing_ok=True, resamples=10)
    for what, fn in (("another format", lambda d: d.update(format="x")),
                     ("another exam", lambda d: d.update(exam="imageweeds")),
                     ("another imgsz", lambda d: d.update(imgsz=832)),
                     ("the seed field not the directory's", lambda d: d.update(seed=2)),
                     ("stage-1 weights not the base run's", lambda d: d["stage1"].update(weights_sha256="0" * 64)),
                     ("agnostic NMS", lambda d: d["stage1"].update(nms="agnostic")),
                     ("the stage-1 agnostic AP not compared", lambda d: d["stage1"].update(
                         vs_recorded_agnostic={"compared": False})),
                     ("another classifier", lambda d: d["stage2"].update(classifier_npz_sha256="0" * 64)),
                     ("other code than the equivalence check's", lambda d: d["code"].update(twostage="0" * 64)),
                     ("another equivalence check", lambda d: d.update(equivalence_sha256="0" * 64)),
                     ("taken before the classifier was fitted", lambda d: d.update(created_utc="2020-01-01T00:00:00Z")),
                     ("another settings stamp", lambda d: d["settings"].update(iou=0.6)),
                     ("another key order", lambda d: d.update(key_order_sha256="0" * 64))):
        with tamper(f, fn):
            out[what] = run()
    with tamper(_arm_file("M", 1), lambda d: (d["stage1"].update(nms="locked"), d["settings"].update(nms="locked"))):
        out["E3-M under the locked NMS (its deciding NMS is class-agnostic)"] = run()
    npz = TS.arm_dir("A", 1) / "dev.images.npz"
    raw = npz.read_bytes()
    try:
        npz.write_bytes(raw + b"0")
        out["an npz that does not hash"] = run()
    finally:
        npz.write_bytes(raw)
    msrc = W.sha(src_weights("A", 0))
    with tamper(_arm_file("M", 0), lambda d: (d["stage1"].update(weights_sha256=msrc), d.update(weights_sha256=msrc))):
        out["E3-M's stage-1 weights not the reference's seed weights"] = run()
    with tamper(C.INC_DIR / "capacity" / "e2_v1.json", lambda d: d.update(status="pending")):
        out["E2's verdict not decided"] = run()
    with tamper(C.INC_DIR / "capacity" / "e2_v1.json", lambda d: d["bootstrap"].update(resamples=200)):
        out["E2's verdict under other parameters"] = run()
    with tamper(C.INC_DIR / "capacity" / "e2_v1.json", lambda d: d["arms"]["W"].update(reference_inputs=[]) or
                d["arms"]["S"].update(reference_inputs=[])):
        out["a reference file E2's verdict did not read"] = run()
    with tamper(TS.pin_path(), lambda d: d.update(npz_sha256="0" * 64)):
        out["a classifier other than the pin's"] = run()
    real_all = TS.all_boxes
    with patched(TS, "all_boxes", lambda rows: (real_all(rows)[0], "0" * 64)):
        out["base_v2's labels no longer give the training boxes"] = run()
    real_load = TS.load_classifier

    def dirty(fold=None):
        r = list(real_load(fold))
        r[4] = copy.deepcopy(r[4])
        r[4]["guard"]["refused"] = 1
        return tuple(r)
    with patched(TS, "load_classifier", dirty):
        out["a guard record that is not clean"] = run()
    with tamper(TS.root() / "equivalence.json", lambda d: d.update(all_passed=False)):
        out["an equivalence check that did not pass"] = run()
    with moved(TS.root() / "equivalence.json"):
        out["no equivalence check"] = run()
    with tamper(TS.classifier_dir() / "dev_gt.json", lambda d: d.update(created_utc="2020-01-01T00:00:00Z")):
        out["dev ground truth read before the fit"] = run()
    bad = [k for k, e in out.items() if e is None]
    check("refused: %s" % "; ".join(out), not bad, (bad, {k: str(v)[:120] for k, v in out.items()}))
    with moved(TS.arm_dir("B", 2) / "dev.json"):
        d = TS.e3_decision(testing_ok=True, resamples=10)
    check("a missing file: that arm pending, the verdict pending, no choice, the attribution pending",
          d["status"] == "pending" and d["arms"]["B"]["status"] == "pending" and d["chosen"] is None
          and d["attribution"]["status"] == "pending" and d["arms"]["A"]["status"] == "decided", d["arms"]["B"])


# ------------------------------------------------------------------ 10. the rule on synthetic files
def synth_root(tag, vals, arrays=None):
    """A copy of E3's directory (the classifier, the equivalence check, dev ground truth) whose arm dev files carry
    the given species values and per-image arrays (the reference's GT, predictions perturbed): TS.root is pointed
    at it by the caller."""
    src = TS.root()
    dst = src.with_name("%s_%s" % (src.name, tag))
    if dst.exists():
        shutil.rmtree(str(dst))
    shutil.copytree(str(src / "classifier"), str(dst / "classifier"))
    shutil.copy(str(src / "equivalence.json"), str(dst / "equivalence.json"))
    ref = [SC.load_npz(C.INC_DIR / REF / "runs" / ("final__base__s%d" % s) / "scores" / "dev@640.images.npz")
           for s in (0, 1, 2)]
    for arm in "MAB":
        for s in (0, 1, 2):
            doc = json.loads((src / "arms" / arm / ("s%d" % s) / "dev.json").read_text())
            a = (arrays or {}).get((arm, s))
            if a is None:
                a = dict(ref[s])
                rng = np.random.default_rng(C.stable_int("%s/%s/%d" % (tag, arm, s)))
                a["conf"] = np.clip(a["conf"] + rng.normal(0, 0.05, len(a["conf"])), 0.002, 1)
            doc["species_map50_95"] = vals[arm][s]
            d = dst / "arms" / arm / ("s%d" % s)
            d.mkdir(parents=True, exist_ok=True)
            doc["images"]["sha256"] = SC.save_npz(d / "dev.images.npz", a)
            doc["key_order_sha256"] = C.sha256_text("\n".join(str(k) for k in a["keys"]))
            doc["created_utc"] = TS._utc()
            (d / "dev.json").write_text(json.dumps(doc))
    return dst


@contextlib.contextmanager
def at_root(path):
    with patched(TS, "root", lambda: pathlib.Path(path)):
        yield


def ref_vals():
    return [json.loads((C.INC_DIR / REF / "runs" / ("final__base__s%d" % s) / "scores" / "dev@640.json").read_text())[
        "species_map50_95"] for s in (0, 1, 2)]


def _boot(se):
    def fn(a, b, resamples=1000, seed_text=None, **k):
        return {"se": se, "n_valid": resamples, "per_species": {n: {"se": 0.01, "n_valid": resamples}
                                                                for n in SC.SPECIES}}
    return fn


def test_rule():
    print("the rule on synthetic E3 files (the reference's real files; the bootstrap's SE set per world)")
    r = ref_vals()
    sd_r = float(np.std(r, ddof=1))
    big = 3 * sd_r + 0.02

    def shift(dv, spread=(0.0, 0.0, 0.0)):
        return [r[i] + dv + spread[i] for i in range(3)]
    tiny = (0.0005, -0.0005, 0.0)
    worlds = {
        "M and B qualify, B larger": ({"M": shift(big, tiny), "A": shift(0), "B": shift(big + 0.03, tiny)}, 0.001),
        "D above SE, not 2 pooled sd": ({"M": shift(0.05, (big + 0.1, -big - 0.1, 0.0)), "A": shift(0),
                                         "B": shift(0)}, 0.001),
        "D above 2 pooled sd, not SE": ({"M": shift(big, tiny), "A": shift(0), "B": shift(0)}, big + 1.0),
        "a tie of D between M and B": ({"M": shift(big, tiny), "A": shift(0), "B": shift(big, tiny)}, 0.001),
        "A and B tie": ({"M": shift(0), "A": shift(big, tiny), "B": shift(big, tiny)}, 0.001),
        "B far above A, not above the reference": ({"M": shift(0), "A": shift(-big - 0.05, tiny), "B": shift(0, tiny)},
                                                   0.001)}
    got = {}
    for name, (vals, se) in worlds.items():
        d = synth_root("r%d" % len(got), vals)
        with at_root(d), patched(B, "native_bootstrap", _boot(se)):
            got[name] = TS.e3_decision(testing_ok=True, resamples=40)
        shutil.rmtree(str(d))
    g = got["M and B qualify, B larger"]
    vm = worlds["M and B qualify, B larger"][0]["M"]
    check("both conditions: M and B qualify, A does not; the largest D (B) is chosen; pooled sd is "
          "sqrt((sd_arm^2 + sd_ref^2) / 2)", g["qualifying"] == ["M", "B"] and g["chosen"] == "B"
          and abs(g["arms"]["M"]["pooled_sd"] - np.sqrt((np.std(vm, ddof=1) ** 2 + sd_r ** 2) / 2)) < 1e-12,
          {k: (a["diff"], a["two_pooled_sd"], a["se_diff"], a["qualifies"]) for k, a in g["arms"].items()})
    a1 = got["D above SE, not 2 pooled sd"]["arms"]["M"]
    a2 = got["D above 2 pooled sd, not SE"]["arms"]["M"]
    check("each condition alone does not qualify (D > SE only; D > 2 pooled sd only)",
          a1["conditions"] == {"above_2_pooled_sd": False, "above_se": True} and not a1["qualifies"]
          and a2["conditions"] == {"above_2_pooled_sd": True, "above_se": False} and not a2["qualifies"],
          (a1["conditions"], a2["conditions"]))
    check("ties go to M, then A, then B", got["a tie of D between M and B"]["chosen"] == "M"
          and got["A and B tie"]["chosen"] == "A", (got["a tie of D between M and B"]["chosen"],
                                                    got["A and B tie"]["chosen"]))
    gb = got["B far above A, not above the reference"]
    check("the attribution is recorded beside and never qualifies or chooses an arm (B credited over A, yet B does "
          "not qualify against the reference)",
          all("credited_to_data" in x["attribution"] for x in got.values())
          and got["A and B tie"]["attribution"]["diff"] == 0.0 and got["A and B tie"]["qualifying"] == ["A", "B"]
          and gb["attribution"]["credited_to_data"] is True and gb["qualifying"] == [] and gb["chosen"] is None,
          (gb["attribution"], gb["qualifying"]))
    d = synth_root("attr", worlds["M and B qualify, B larger"][0])
    with at_root(d):
        da = TS.e3_decision(testing_ok=True, resamples=40)
    arrs = {arm: [SC.load_npz(d / "arms" / arm / ("s%d" % s) / "dev.images.npz") for s in (0, 1, 2)] for arm in "AB"}
    shutil.rmtree(str(d))
    at = da["attribution"]
    ind = N.independent_se(arrs["B"], arrs["A"], 40, "inc2/e3/attribution_se")
    sp = B.native_bootstrap(arrs["B"], arrs["A"], resamples=40, seed_text="inc2/e3/species_se")["se"]
    check("the attribution's SE (%.6f) equals an independent recomputation under inc2/e3/attribution_se (%.6f) and "
          "differs from the species draw's (%.6f)" % (at["se_diff"], ind, sp),
          abs(at["se_diff"] - ind) < 1e-9 and abs(at["se_diff"] - sp) > 1e-12, (at["se_diff"], ind, sp))
    se_ind = N.independent_se([SC.load_npz(TS.arm_dir("M", s) / "dev.images.npz") for s in (0, 1, 2)],
                              [SC.load_npz(C.INC_DIR / REF / "runs" / ("final__base__s%d" % s) / "scores"
                                           / "dev@640.images.npz") for s in (0, 1, 2)], 30, "inc2/e3/species_se")
    real = json.loads((C.INC_DIR / "cap_e3" / "e3_v1.json").read_text())
    check("on the real files, E3-M's SE (%.6f) equals an independent recomputation under inc2/e3/species_se (%.6f)"
          % (real["arms"]["M"]["se_diff"], se_ind), abs(real["arms"]["M"]["se_diff"] - se_ind) < 1e-9)
    check("each decision is dev only", all(BP.dev_leaks(x) == [] for x in got.values()))


def test_verdict_file():
    print("verdict files: kept, refused, overwritten; the L23G record")
    out = C.INC_DIR / "cap_e3_files"
    with moved(TS.arm_dir("B", 0) / "dev.json"):
        d1, _ = TS.verdict(out_dir=out, testing_ok=True, resamples=20)
        e = refused(TS.rescore_verdict, out_dir=out, testing_ok=True, resamples=20)
    s1 = json.loads((out / "e3_v1.json").read_text())["status"]
    d2, _ = TS.verdict(out_dir=out, testing_ok=True, resamples=20)
    check("a pending verdict is written, rescore_verdict refuses while pending (no record), and a decided one "
          "overwrites the pending file", d1["status"] == "pending" and s1 == "pending" and e is not None
          and "pending" in str(e) and d2["status"] == "decided" and not (out / "e3_rescore.json").exists(), e)
    rec = TS.rescore_verdict(out_dir=out, testing_ok=True, resamples=20)
    check("rescore_verdict (L23G): capacity/e3_rescore.json complete with the score records' and the verdict's names "
          "and sha256s, dev only, no path", rec["status"] == "complete"
          and rec["verdict"]["sha256"] == W.sha(out / "e3_v1.json") and len(rec["score_records"]) == 3
          and all(x["sha256"] and x["status"] == "complete" for x in rec["score_records"])
          and not BP.dev_leaks(rec) and str(C.INC_DIR) not in json.dumps(rec), rec)
    with tamper(out / "e3_v1.json", lambda d: d["arms"]["A"].update(diff=1.0)):
        e = refused(TS.verdict, out_dir=out, testing_ok=True, resamples=20)
    check("a decided file whose decision differs is refused (a person moves it aside)", e is not None, e)
    with tamper(out / "e3_v1.json", lambda d: d.update(testing_allowed=False)):
        e = refused(TS.verdict, out_dir=out, testing_ok=True, resamples=20)
    check("  one decided without test-mode files admitted is refused by a recomputation that admits them",
          e is not None, e)
    with tamper(out / "e3_v1.json", lambda d: d["bootstrap"].update(resamples=1000)):
        e = refused(TS.verdict, out_dir=out, testing_ok=True, resamples=20)
    with tamper(out / "e3_v1.json", lambda d: d["attribution"]["bootstrap"].update(seed_text="inc2/e3/species_se")):
        e2 = refused(TS.verdict, out_dir=out, testing_ok=True, resamples=20)
    check("  one whose decision is the same but recorded under other bootstrap parameters (1,000 resamples; the "
          "attribution under the species seed text) is refused", e is not None and e2 is not None, (e, e2))
    check("the evidence allow-lists capacity/e3_v1.json, e3_rescore.json, e3_score_<X>.json; never the report, the "
          "test read, the pin or E3's directory",
          E.allowed("capacity/e3_v1.json") and E.allowed("capacity/e3_rescore.json")
          and all(E.allowed("capacity/e3_score_%s.json" % k) for k in "MAB") and not E.allowed("capacity/e3_score_X.json")
          and not E.allowed("capacity/e3_v1_report.json") and not E.allowed("capacity/e3_test_M.json")
          and not E.allowed("capacity/e3_classifier_pin.json") and not E.allowed("twostage/exp.json")
          and "twostage" in E.RESERVED_DIRS)


# ------------------------------------------------------------------ 11. the test read
def fabricate_verdict(qualifying=("A", "B"), chosen="A", **over):
    """capacity/e3_v1.json as decided under the pre-registered parameters, on the real files' decision, with the
    given arms qualifying."""
    d = json.loads((C.INC_DIR / "cap_e3" / "e3_v1.json").read_text())
    for k in "MAB":
        d["arms"][k]["qualifies"] = k in qualifying
    d.update(qualifying=list(qualifying), chosen=chosen, testing_allowed=False)
    d["bootstrap"]["resamples"] = 1000
    d.update(over)
    TS.verdict_path().write_text(json.dumps(d))
    return TS.verdict_path()


def test_test_read():
    print("the test read: once per qualifying arm, after the verdict; score-test; test-report")
    e0 = refused(TS.test_read, "A")
    vp = fabricate_verdict()
    with tamper(vp, lambda d: d["bootstrap"].update(resamples=30)):
        e1 = refused(TS.test_read, "A")
    with tamper(vp, lambda d: d.update(testing_allowed=True)):
        e2 = refused(TS.test_read, "A")
    with tamper(vp, lambda d: d.update(rule="another")):
        e3 = refused(TS.test_read, "A")
    e4 = refused(TS.test_read, "M")
    with tamper(vp, lambda d: d["classifier"].update(npz_sha256="0" * 64)):
        e5 = refused(TS.test_read, "A")
    with tamper(vp, lambda d: d["arms"]["A"]["inputs"][1].update(weights_sha256="0" * 64)):
        e6 = refused(TS.test_read, "A")
    e7 = refused(TS.score_test, "A", embedder=EMB, **KW)
    with tamper(vp, lambda d: d.update(status="pending")):
        e8 = refused(TS.test_read, "A")
    with tamper(vp, lambda d: d.update(format="inc2-e2-verdict/1")):
        e9 = refused(TS.test_read, "A")
    check("refused, writing nothing: before the verdict, under other resamples, test-mode files admitted, another "
          "rule, a non-qualifying arm, another classifier, changed weights, a pending verdict, another format; "
          "score-test without a read",
          all(x is not None for x in (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9)) and not (TS.root() / "test").exists()
          and "did not qualify" in str(e4) and "test-read" in str(e7), (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9))
    rec = TS.test_read("A")
    argv = rec["argv"]
    check("test-read --arm A: the record pins the verdict, the classifier and the three weights; one "
          "run_inc2_build.sh argv (inc2.twostage score-test --arm A, job inc_build_e3test_a); the headline",
          rec["headline"] and rec["verdict_sha256"] == W.sha(vp) and sorted(rec["weights"]) == ["0", "1", "2"]
          and argv[:3] == ["sbatch", "--parsable", "--job-name=inc_build_e3test_a"]
          and argv[-4:] == ["inc2.twostage", "score-test", "--arm", "A"] and argv[-5].endswith("run_inc2_build.sh"),
          rec)
    e = refused(TS.test_read, "A")
    check("  a second read is refused", e is not None and "once" in str(e), e)
    pending = TS.test_report("A", testing_ok=True)
    check("test-report before the job: pending", pending["status"] == "pending" and len(pending["missing"]) == 3,
          pending["missing"])
    rt = C.INC_DIR / REF / "runs" / "final__base__s1" / "scores" / "test.json"
    with tamper(rt, lambda d: d.update(map50_95=d["map50_95"] + 0.01)):
        e = refused(TS.score_test, "A", embedder=EMB, **KW)
    check("score-test refuses before any E3 test score when b_v2_m640's recorded test score is not reproduced by the "
          "identity path", e is not None and "not reproduced" in str(e)
          and not list((TS.root() / "test" / "A").glob("s*/test.json")), e)
    out = TS.score_test("A", embedder=EMB, **KW)
    t0 = json.loads((TS.test_dir("A") / "s0" / "test.json").read_text())
    check("score-test: the reference's three identity checks, then E3-A's three test scores, written once",
          len(out["arm"]) == 3 and len(out["reference"]) == 3
          and all((TS.root() / "test" / "reference" / ("s%d" % s) / "test.identity.json").exists() for s in (0, 1, 2))
          and t0["exam"] == "test" and t0["stage1"]["weights_sha256"] == rec["weights"]["0"]["weights_sha256"], out)
    e = refused(TS.score_test, "A", embedder=EMB, **KW)
    check("  a second score-test is refused", e is not None and "once" in str(e), e)
    with moved(TS.test_dir("A") / "e3_test_read.json"):
        e = refused(TS.test_read, "A")
        again = (TS.test_dir("A") / "e3_test_read.json").exists()
    check("  with the read record gone, a new read is refused while E3-A's test scores exist (none written)",
          e is not None and "prepared already" in str(e) and not again, e)
    rep = TS.test_report("A", testing_ok=True)
    check("test-report: complete, 12-class and agnostic mean +- sd against b_v2_m640's test files, the gap to 0.90, "
          "the headline; capacity/e3_test_A.json is not on the evidence", rep["status"] == "complete"
          and rep["headline"] and rep["arm_scores"]["twelve"]["n"] == 3 and rep["reference"]["scores"]["twelve"]["n"] == 3
          and abs(rep["gap_to_target"]["arm"] - (0.90 - rep["arm_scores"]["twelve"]["mean"])) < 1e-12
          and not E.allowed("capacity/e3_test_A.json"), rep)
    e = refused(TS.test_report, "A")
    with tamper(TS.test_dir("A") / "s1" / "test.json", lambda d: d["stage1"].update(weights_sha256="0" * 64)):
        e2 = refused(TS.test_report, "A", testing_ok=True)
    with tamper(TS.test_dir("A") / "s2" / "test.json", lambda d: d["stage2"].update(classifier_npz_sha256="0" * 64)):
        e3 = refused(TS.test_report, "A", testing_ok=True)
    with tamper(TS.test_dir("A") / "s2" / "test.json", lambda d: d.update(manifest_sha256="0" * 64)):
        e4 = refused(TS.test_report, "A", testing_ok=True)
    check("test-report refuses test-mode scores in production, a score from other weights, another classifier, "
          "another test manifest", all(x is not None for x in (e, e2, e3, e4)), (e, e2, e3, e4))
    rb = TS.test_read("B")
    TS.score_test("B", embedder=EMB, **KW)
    repb = TS.test_report("B", testing_ok=True)
    check("E3-B (qualifying, not chosen): read once, the reference identities reused, reported as not the headline",
          not rb["headline"] and repb["status"] == "complete" and not repb["headline"]
          and "chose E3-A" in (C.INC_DIR / "capacity" / "e3_test_B.md").read_text(), repb)


def test_sensitivity():
    print("the optional sensitivity (fold classifiers on seed 0), reported only")
    rec = TS.sensitivity(embedder=EMB, folds=[0, 1], **KW)
    check("written once, per arm the two fold classifiers' values and their spread beside the full classifier's",
          all(len(rec["arms"][k]["values"]) == 2 and rec["arms"][k]["full_classifier"] is not None for k in "MAB")
          and refused(TS.sensitivity, embedder=EMB, folds=[0], **KW) is not None, rec["arms"])


# ------------------------------------------------------------------ 12. the CLI
def cli(*args, env=None):
    import io as _io
    err = _io.StringIO()
    old = os.environ.get(S.TEST_ENV)
    if env is not None:
        os.environ.pop(S.TEST_ENV, None)
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(_io.StringIO()):
            try:
                rc = TS.main(list(args))
            except SystemExit as e:
                rc = e.code
    finally:
        if old is not None:
            os.environ[S.TEST_ENV] = old
    return rc, err.getvalue()


def test_cli():
    print("the CLI")
    cases = [(("score-arm",), "needs --arm"), (("verdict", "--arm", "M"), "takes no --arm"),
             (("score-arm", "--arm", "Z"), "needs --arm"), (("test-read", "--arm", "M"), "did not qualify"),
             (("score-test", "--arm", "M"), "no prepared test read"), (("fit-classifier",), "fitted once"),
             (("restore-classifier",), "needs --from"), (("equivalence",), "written once")]
    res = [(a, cli(*a)) for a, _w in cases]
    check("refusals exit 2 with the [inc2.twostage] ERROR line: %s" % "; ".join(" ".join(a) for a, _ in cases),
          all(rc == 2 and "[inc2.twostage] ERROR:" in err and w in err for (a, (rc, err)), (_a, w) in zip(res, cases)),
          [(a, rc, err[-160:]) for a, (rc, err) in res])
    rc, err = cli("score-arm", "--arm", "M", "--batch", "8", env="unset")
    check("--batch outside test mode exits 2", rc == 2 and "need" in err, (rc, err))
    rc, _err = cli("test-report", "--arm", "A")
    check("test-report --arm A runs (production: refuses the test-mode scores, exit 2)", rc == 2, rc)
    rc, _err = cli("nonsense")
    check("an unknown verb is a usage error (exit 2)", rc == 2)


SEQUENCE = ("test_constants", "test_units", "test_cv_units", "test_protocol_check", "test_fit_refusals",
            "test_fit_exif", "test_fit",
            "test_pin_restore", "test_crops", "test_dev_gt", "test_layering_and_identity", "test_equivalence",
            "test_score_arm", "test_pass_refusals", "test_nms_and_cap", "test_verdict_real", "test_verdict_refusals",
            "test_rule", "test_verdict_file", "test_test_read", "test_sensitivity", "test_cli")


def main():
    """Every test in SEQUENCE (each builds on the files the earlier ones wrote); E3_UPTO=<name> stops after it."""
    t0 = time.time()
    upto = os.environ.get("E3_UPTO")
    try:
        W.build_world()
        setup_world()
        for name in SEQUENCE:
            globals()[name]()
            if name == upto:
                break
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
