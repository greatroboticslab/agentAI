#!/usr/bin/env python3
"""The measurement arms read at their own resolution (inc2/scorer_native.py,
inc2.baseline rescore-native and native-verdict; docs/CONTINUOUS_LOOP.md,
group B, "Amendment (2026-10-01): the measurement arms read at their own
resolution (pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 exams materialised, the
v1 LOCK with scorer.py's sha256); scores are real Ultralytics passes on the
CPU in test mode (INC_SCORER_TESTING=1), so every score is stamped TEST-.

What is pinned:
- refusals, before anything is written: the exam test (at every size, and
  through the CLI: exit 2), any exam but dev and imageweeds; a size that is
  not pre-registered; an arm that is neither a measurement arm nor the
  reference (n640, s640); an experiment that is not a baseline; a run that
  is not a final run, not done, or whose weights do not hash as recorded;
  an output that is the protocol score (<exam>.json) or not <exam>@<imgsz>;
  an existing native score (written once); outside test mode, any
  departure but imgsz (fp32 on the CPU here), and a testing experiment;
  weights that cannot be loaded; any exam but dev at 640, and the
  reference arm on imageweeds (score_run and the CLI: exit 2);
- putting a score in place: an npz left without its JSON is replaced; a
  second writer of the same score is refused and replaces neither file;
  a held lock refuses after the wait, leaving the lock to its holder;
- rescore-native and native-verdict set YOLO_OFFLINE and YOLO_AUTOINSTALL
  before they run, and importing inc2.baseline loads no Ultralytics;
- a real pass of a measurement arm at 832: scores/dev@832.json and its npz
  beside the 640 scores, format inc2-native-score/1, production false, the
  stamp TEST-NATIVE832-<the LOCK's scorer sha256>, the locked and native
  scorers' sha256, imgsz 832 and the locked scorer's conf, iou and max_det,
  the per-image arrays reproducing per_class exactly; the run's protocol
  score untouched;
- the reference at 640: the native pass reproduces the pinned scorer's own
  score of the same weights exactly (per_class, image_correct, key order):
  only imgsz differs between the two scorers; a recorded score that differs
  beyond 0.002 refuses, writing nothing;
- rescore-native: every missing native score of the arm (dev, imageweeds)
  and the reference's dev at 640, then native_rescore.json (complete) and
  the verdict; native_rescore.json lists the dev files only, with no path;
  a second arm's rescore keeps the first arm's decision byte for byte; a
  production verdict refuses kept test-mode scores (a non-testing
  experiment, or test mode off); a second run writes nothing; no protocol
  score changes; a
  grid arm and an unfinished final run are refused before anything is
  scored; nothing opens, hashes or scores the test exam (every exam name,
  manifest and opened path recorded);
- the verdict rule on synthetic numbers: qualifies with all three
  conditions; each condition failing alone (2 pooled sd, the bootstrap SE,
  no target species improved) does not qualify; pooled sd is
  sqrt((sd_arm^2 + sd_ref^2) / 2) (unequal sds: a D between it and the
  other formulas qualifies); a missing file is pending; every stamp and
  setting mismatching alone (within an arm, or arm against reference),
  a test-mode score in a production verdict, a tampered npz,
  a grid arm and a reference that is not m640 refuse; the decision file
  holds no non-dev exam key and no score path (the autopilot's scrub drops
  nothing), and the CLI writes it;
- the paired bootstrap: deterministic under stable_int("inc2/native/diff_se"),
  1,000 resamples by default, identical runs give SE 0, and its SE equals an
  independent recomputation (one ap_per_class call per run and resample on
  the drawn images' concatenated arrays).

Run:  python3 tests/test_inc2_native.py
"""
import builtins
import contextlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_native as SN  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402

FAILURES = W.FAILURES
check = W.check
CPU8 = {"batch": 8, "device": "cpu"}
CPU32 = {"batch": 32, "device": "cpu"}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (SN.NativeRefused, B.BaselineError) as e:
        return e
    return None


def named_checkpoint(path, seed=0):
    """W.cold_checkpoint with the INC class names, so the scorer accepts it untrained."""
    import torch
    W.cold_checkpoint(path, seed=seed)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    ck["model"].names = {i: n for i, n in enumerate(C.CLASS_NAMES)}
    torch.save(ck, str(path))
    return path


def make_exp(exp, arm, seeds=(0, 1, 2), testing=None, final_exams=None, typ="baseline"):
    root = C.INC_DIR / exp
    root.mkdir(parents=True, exist_ok=True)
    defn = {"exp": exp, "type": typ, "seeds": list(seeds),
            "final_exams": list(final_exams or (["dev", "imageweeds"] if RC.arm_id(arm) in RC.MEASURE_ARMS
                                                else ["dev", "imageweeds", "test"])),
            "testing": testing if testing is not None else CPU8}
    defn.update(RC.stamp(RC.resolve_arm(arm, require_weights=False)))
    (root / "exp.json").write_text(json.dumps(defn))
    return root


def make_final(exp, seed, wseed, protocol=True, done=True):
    """runs/final__base__s<seed>: final.pt a symlink to the base run's weights (as inc2.train links it), its
    run.json, and (protocol) its dev score by the pinned scorer at 640, batch 32 (test mode, on the CPU)."""
    root = C.INC_DIR / exp / "runs"
    base = named_checkpoint(root / ("base__s%d" % seed) / "weights" / "final.pt", seed=wseed)
    rd = root / ("final__base__s%d" % seed)
    (rd / "weights").mkdir(parents=True, exist_ok=True)
    fin = rd / "weights" / "final.pt"
    if fin.exists() or fin.is_symlink():
        fin.unlink()
    os.symlink(str(base), str(fin))
    (rd / "run.json").write_text(json.dumps({"status": "done" if done else "running",
                                             "weights_sha256": W.sha(base)}))
    if protocol:
        S.score(fin, "dev", rd / "scores" / "dev.json", lock_check=True, imgsz=640, batch=32, device="cpu")
    return rd


def tree_shas(root):
    return {str(p): W.sha(p) for p in sorted(pathlib.Path(root).rglob("*")) if p.is_file() and not p.is_symlink()}


def copy_arm(src, dst, testing=None):
    """Another experiment of src's arm whose final runs carry src's weights and native scores (renamed to it), its
    rescore record removed: rescore-native keeps every file and scores nothing. testing: exp.json's setting."""
    root = C.INC_DIR / dst
    shutil.copytree(str(C.INC_DIR / src), str(root), symlinks=True)
    defn = json.loads((root / "exp.json").read_text())
    defn["exp"] = dst
    if testing is not None:
        defn["testing"] = testing
    (root / "exp.json").write_text(json.dumps(defn))
    for p in root.rglob("*@*.json"):
        x = json.loads(p.read_text())
        x["exp"] = dst
        p.write_text(json.dumps(x))
    (root / B.NATIVE_RECORD).unlink()
    return root


@contextlib.contextmanager
def no_testing_env():
    old = os.environ.pop(S.TEST_ENV, None)
    try:
        yield
    finally:
        if old is not None:
            os.environ[S.TEST_ENV] = old


@contextlib.contextmanager
def reads():
    """Every exam name the scorers' checks and manifests are asked for, and every path opened."""
    seen = {"exams": [], "paths": []}
    orig = (C.manifest_path, S.check_exam, S.check_lock, builtins.open)

    def manifest_path(exam, *a, **k):
        seen["exams"].append(exam)
        return orig[0](exam, *a, **k)

    def check_exam(exam, rows):
        seen["exams"].append(exam)
        return orig[1](exam, rows)

    def check_lock(exam, lock_check=True):
        seen["exams"].append(exam)
        return orig[2](exam, lock_check)

    def opener(file, *a, **k):
        if isinstance(file, (str, os.PathLike)):
            seen["paths"].append(str(file))
        return orig[3](file, *a, **k)
    C.manifest_path, S.check_exam, S.check_lock, builtins.open = manifest_path, check_exam, check_lock, opener
    try:
        yield seen
    finally:
        C.manifest_path, S.check_exam, S.check_lock, builtins.open = orig


def touches_test(seen):
    tdir = str(C.EXAMS_DIR / "test")
    bad = [e for e in seen["exams"] if e == "test"]
    bad += [p for p in seen["paths"] if p.startswith(tdir) or os.path.basename(p).startswith("test.")
            or os.path.basename(p).startswith("test@")]
    return bad


# ------------------------------------------------------------------ refusals
def test_refusals():
    print("refusals (nothing written)")
    e = refused(SN.check_exam, "test")
    check("the exam test is refused at every size (P10)", e is not None and "P10" in str(e), e)
    check("... and every exam but dev and imageweeds", refused(SN.check_exam, "ood22") is not None
          and SN.check_exam("dev") == "dev" and SN.check_exam("imageweeds") == "imageweeds")
    check("the sizes: a measurement arm's training imgsz (m832 832, s1024 1024), the reference arm m640 at 640",
          SN.native_imgsz("m832") == 832 and SN.native_imgsz("s1024") == 1024 and SN.native_imgsz("m640") == 640
          and SN.allowed_sizes() == [640, 832, 1024] and SN.REFERENCE_ARM == "m640")
    for arm in ("n640", "s640"):
        e = refused(SN.native_imgsz, arm)
        check("a grid arm other than the reference (%s) is never read by the native scorer" % arm, e is not None, e)
    w = named_checkpoint(W.TMP / "nat_any.pt", seed=7)
    out = W.TMP / "nat_out" / "scores"
    out.mkdir(parents=True, exist_ok=True)
    for what, args, frag in (("the exam test", ("test", out / "test@832.json", 832), "P10"),
                             ("the protocol score's name", ("dev", out / "dev.json", 640), "protocol score"),
                             ("a name that is not the size's", ("dev", out / "dev@832.json", 1024), "written as"),
                             ("a size that is not pre-registered", ("dev", out / "dev@704.json", 704), "pre-registered"),
                             ("imageweeds at 640 (the reference's size, read on dev only)",
                              ("imageweeds", out / "imageweeds@640.json", 640), "reads only the reference arm"),
                             ("a path outside a run's scores dir", ("dev", W.TMP / "dev@832.json", 832), "written as")):
        e = refused(SN.score, w, args[0], args[1], args[2], device="cpu")
        check("score() refuses %s, writing nothing" % what, e is not None and frag in str(e) and not any(out.iterdir())
              and not (W.TMP / "dev@832.json").exists(), e)
    real = SN.allowed_sizes
    SN.allowed_sizes = lambda: [830]
    try:
        e = refused(SN.score, w, "dev", out / "dev@830.json", 830, device="cpu")
    finally:
        SN.allowed_sizes = real
    check("a size Ultralytics validates at another (830 -> 832, its stride) is refused, writing nothing",
          e is not None and "validated at 832" in str(e) and not any(out.iterdir()), e)
    p = out / "once.json"
    p.write_text("{}")
    e = refused(SN._write_once, p, {"x": 1})
    check("the JSON is created only if absent (a hard link): an existing file is never replaced",
          e is not None and p.read_text() == "{}" and sorted(x.name for x in out.iterdir()) == ["once.json"], e)
    p.unlink()
    with no_testing_env():
        e = refused(SN.score, w, "dev", out / "dev@832.json", 832, device="cpu")
    check("outside test mode every departure but imgsz refuses (fp32 on the CPU, another Ultralytics)",
          e is not None and "not a native score" in str(e) and "fp32" in str(e) and not any(out.iterdir()), e)
    bad = W.TMP / "nat_bad" / "final.pt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"w" * 64)
    e = refused(SN.score, bad, "dev", out / "dev@832.json", 832, device="cpu")
    check("weights that cannot be loaded are refused (not a crash)", e is not None and "cannot be loaded" in str(e), e)

    make_exp("nat_n640", "n640")
    make_final("nat_n640", 0, 11, protocol=False)
    e = refused(SN.score_run, "nat_n640", "final__base__s0", "dev")
    check("score_run refuses a grid arm's experiment (n640)", e is not None and "neither a measurement" in str(e), e)
    make_exp("nat_chain", "m832", typ="chain")
    e = refused(SN.score_run, "nat_chain", "final__base__s0", "dev")
    check("... and an experiment that is not a baseline", e is not None and "not a baseline" in str(e), e)
    make_exp("nat_r", "m832")
    make_final("nat_r", 0, 12, protocol=False)
    make_final("nat_r", 1, 13, protocol=False, done=False)
    for what, run, exam, frag in (("a base run", "base__s0", "dev", "final runs"),
                                  ("a seed the experiment does not have", "final__base__s7", "dev", "final runs"),
                                  ("a run that is not done", "final__base__s1", "dev", "not done"),
                                  ("an exam outside its final exams", "final__base__s0", "test", "test")):
        e = refused(SN.score_run, "nat_r", run, exam)
        check("score_run refuses %s" % what, e is not None and frag in str(e), e)
    make_exp("nat_devonly", "m832", final_exams=["dev"])
    make_final("nat_devonly", 0, 12, protocol=False)
    e = refused(SN.score_run, "nat_devonly", "final__base__s0", "imageweeds")
    check("score_run refuses an exam outside the experiment's final exams (imageweeds of a dev-only arm)",
          e is not None and "final exams" in str(e), e)
    rj = C.INC_DIR / "nat_r" / "runs" / "final__base__s0" / "run.json"
    good = rj.read_text()
    rj.write_text(json.dumps({"status": "done", "weights_sha256": "0" * 64}))
    e = refused(SN.score_run, "nat_r", "final__base__s0", "dev")
    rj.write_text(good)
    check("... and weights that do not hash as run.json records", e is not None and "does not hash" in str(e), e)
    with no_testing_env():
        e = refused(SN.score_run, "nat_r", "final__base__s0", "dev")
    check("... and a testing experiment outside test mode", e is not None and "testing experiment" in str(e), e)
    check("none of these wrote a native score", not list((C.INC_DIR / "nat_r").rglob("*@*")))
    env = dict(os.environ, PYTHONPATH=str(W.ROOT))
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.scorer_native", "--exp", "nat_r",
                        "--run", "final__base__s0", "--exam", "test"], cwd=str(W.ROOT), env=env,
                       capture_output=True, text=True)
    check("the CLI refuses --exam test (exit 2, REFUSED), writing nothing",
          r.returncode == 2 and "REFUSED" in r.stderr and not list((C.INC_DIR / "nat_r").rglob("*@*")),
          (r.returncode, r.stderr[-300:]))


def test_commit():
    print("putting a native score in place: the npz only while no JSON names it, the JSON once")
    d = W.TMP / "nat_commit" / "scores"
    d.mkdir(parents=True, exist_ok=True)
    js, npz = SN.paths_for(d, "dev", 832)
    a_arr, b_arr = synth_arrays(seed=3), synth_arrays(seed=4)

    def staged(arrays, tag):
        p = d / (".staged_%s" % tag)
        return p, SC.save_npz(p, arrays)
    npz.write_bytes(b"left by an attempt that wrote no JSON")
    sa, sha_a = staged(a_arr, "a")
    sb, sha_b = staged(b_arr, "b")             # a second pass of the same score, staged before the first commits
    SN._commit(js, npz, sa, {"images": {"sha256": sha_a}})
    rec = json.loads(js.read_text())
    check("the first writer replaces an npz left without its JSON, then links its JSON: the JSON names the npz beside it",
          W.sha(npz) == sha_a == rec["images"]["sha256"] and not sa.exists(), (W.sha(npz), sha_a))
    j0 = js.read_bytes()
    e = refused(SN._commit, js, npz, sb, {"images": {"sha256": sha_b}})
    check("a second writer of the same score is refused and replaces neither file (its npz discarded)",
          e is not None and "written once" in str(e) and sha_b != sha_a and W.sha(npz) == sha_a
          and js.read_bytes() == j0 and not sb.exists(), e)
    check("  and the score's lock file is gone after both",
          not [p.name for p in d.iterdir() if p.name.endswith(SN.COMMIT_SUFFIX)], sorted(p.name for p in d.iterdir()))
    js2, npz2 = SN.paths_for(d, "imageweeds", 832)
    lock = js2.with_name("." + js2.stem + SN.COMMIT_SUFFIX)
    lock.write_text("pid 1\n")
    sc, sha_c = staged(a_arr, "c")
    wait = SN.COMMIT_WAIT_S
    SN.COMMIT_WAIT_S = 0.3
    try:
        e = refused(SN._commit, js2, npz2, sc, {"images": {"sha256": sha_c}})
    finally:
        SN.COMMIT_WAIT_S = wait
    check("while another writer holds the score's lock nothing is put in place: refused after the wait, its npz "
          "discarded, the lock left to its holder",
          e is not None and "is held" in str(e) and not js2.exists() and not npz2.exists() and not sc.exists()
          and lock.read_text() == "pid 1\n", e)


def test_offline_env():
    print("rescore-native and native-verdict run offline, as the scorers' own CLIs")
    env = dict(os.environ, PYTHONPATH=str(W.ROOT))
    r = subprocess.run([sys.executable, "-c", "import sys; from weed_optimizer_framework.tools.inc2 import baseline; "
                        "sys.exit(1 if 'ultralytics' in sys.modules else 0)"], cwd=str(W.ROOT), env=env,
                       capture_output=True, text=True)
    keys = ("YOLO_OFFLINE", "YOLO_AUTOINSTALL")
    seen, saved = {}, {k: os.environ.pop(k, None) for k in keys}
    real = (B.rescore_native, B.native_verdict)
    B.rescore_native = lambda *a, **k: seen.setdefault("rescore-native", {x: os.environ.get(x) for x in keys})
    B.native_verdict = lambda *a, **k: seen.setdefault("native-verdict", {x: os.environ.get(x) for x in keys})
    try:
        rcs = [B.main(["rescore-native", "--exp", "nat_any"]), B.main(["native-verdict"])]
    finally:
        B.rescore_native, B.native_verdict = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    want = {"YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false"}
    check("both verbs set YOLO_OFFLINE=true and YOLO_AUTOINSTALL=false before they run, and importing inc2.baseline "
          "does not import Ultralytics (so the setting precedes it)",
          r.returncode == 0 and rcs == [0, 0] and seen == {"rescore-native": want, "native-verdict": want},
          (r.returncode, r.stderr[-200:], rcs, seen))


# ------------------------------------------------------------------ real passes
def test_real_scores():
    print("real CPU passes: a measurement arm at 832, the reference at 640")
    make_exp("nat_m832", "m832", testing=CPU32)
    make_exp("nat_m640", "m640", testing=CPU32)
    for s in (0, 1, 2):
        make_final("nat_m832", s, 20 + s)
        make_final("nat_m640", s, 30 + s)
    rd = C.INC_DIR / "nat_m832" / "runs" / "final__base__s0"
    proto = rd / "scores" / "dev.json"
    before = W.sha(proto)
    (rd / "scores" / "dev@832.images.npz").write_bytes(b"left by an attempt that wrote no JSON")
    rec = SN.score_run("nat_m832", "final__base__s0", "dev")
    js, npz = rd / "scores" / "dev@832.json", rd / "scores" / "dev@832.images.npz"
    d = json.loads(js.read_text())
    lock = json.loads(C.LOCK_PATH.read_text())
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path("dev")))
    check("scores/dev@832.json and its npz are written beside the 640 scores; the protocol score is untouched",
          js.is_file() and npz.is_file() and W.sha(proto) == before and rec["out"] == str(js.resolve())
          and d["protocol_score"]["sha256"] == before, sorted(p.name for p in (rd / "scores").iterdir()))
    check("its format, protocol and stamps: production false, test mode, TEST-NATIVE832-<the LOCK's scorer sha256>",
          d["format"] == SN.FORMAT == "inc2-native-score/1" and d["protocol"] == SN.PROTOCOL
          and d["production"] is False and d["native_production"] is False
          and d["scorer_sha256"] == "TEST-NATIVE832-" + lock["scorer_sha256"]
          and d["locked_scorer_sha256"] == lock["scorer_sha256"]
          and d["scorer_native_sha256"] == W.sha(pathlib.Path(SN.__file__).resolve()), d["scorer_sha256"])
    st = d["settings"]
    check("imgsz 832 and every other setting the locked scorer's (conf %s, iou %s, max_det %s, batch %s)"
          % (st["conf"], st["iou"], st["max_det"], st["batch"]),
          d["imgsz"] == st["imgsz"] == 832 and st["conf"] == S.CONF and st["iou"] == S.IOU and st["max_det"] == 300
          and st["batch"] == 32 and d["trained_imgsz"] == 832 and d["arm"] == "m832"
          and "imgsz 832, not 640" in d["deviations"] and not any("imgsz" in x for x in d["other_deviations"]), st)
    arr = SC.load_npz(npz)
    check("the per-image arrays are in exam key order, hash as recorded and reproduce per_class exactly",
          [str(k) for k in arr["keys"]] == keys and d["images"]["sha256"] == W.sha(npz)
          and d["key_order_sha256"] == C.sha256_text("\n".join(keys))
          and d["consistency"]["full_recompute_max_abs_diff"] <= 1e-9
          and set(SC.full_per_class(arr)) == set(d["per_class"]), d["consistency"])
    check("  (an npz left without its JSON was replaced; no lock or staged file is left in scores/)",
          not [p.name for p in (rd / "scores").iterdir() if p.name.startswith(".")],
          sorted(p.name for p in (rd / "scores").iterdir()))
    b0 = js.read_bytes()
    e = refused(SN.score_run, "nat_m832", "final__base__s0", "dev")
    check("a native score is written once: a second pass is refused before scoring and the file is unchanged",
          e is not None and "already exists" in str(e) and js.read_bytes() == b0, e)
    # the reference at 640: the same weights scored by the pinned scorer and by the native scorer
    rr = C.INC_DIR / "nat_m640" / "runs" / "final__base__s0" / "scores"
    pr = json.loads((rr / "dev.json").read_text())
    nr = SN.score_run("nat_m640", "final__base__s0", "dev")
    same = [k for k in ("per_class", "map50_95", "species_map50_95", "image_correct", "key_order_sha256",
                        "manifest_sha256", "n_gt", "agnostic_map50_95") if pr.get(k) != nr.get(k)]
    check("at 640 the native pass reproduces the pinned scorer's own score of the same weights exactly: only imgsz "
          "differs between the two scorers (%s)" % (same or "per_class, image_correct, key order equal"),
          not same and nr["vs_protocol_score"]["compared"] is True and nr["vs_protocol_score"]["max_abs_diff"] == 0.0
          and nr["scorer_sha256"] == "TEST-NATIVE640-" + lock["scorer_sha256"], (same, nr.get("vs_protocol_score")))
    e = refused(SN.score_run, "nat_m640", "final__base__s1", "imageweeds")
    env = dict(os.environ, PYTHONPATH=str(W.ROOT))
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.scorer_native", "--exp", "nat_m640",
                        "--run", "final__base__s1", "--exam", "imageweeds"], cwd=str(W.ROOT), env=env,
                       capture_output=True, text=True)
    check("the reference arm is read on dev only: score_run and the CLI (exit 2, REFUSED) refuse its imageweeds, "
          "though imageweeds is among its final exams, writing nothing",
          e is not None and "read on ['dev'] only" in str(e) and r.returncode == 2 and "REFUSED" in r.stderr
          and not list((C.INC_DIR / "nat_m640").rglob("imageweeds@*")), (e, r.returncode, r.stderr[-300:]))
    make_exp("nat_m640b", "m640", testing=CPU32)
    make_final("nat_m640b", 0, 30)
    pb = C.INC_DIR / "nat_m640b" / "runs" / "final__base__s0" / "scores" / "dev.json"
    x = json.loads(pb.read_text())
    x["per_class"] = {k: v + 0.01 for k, v in x["per_class"].items()}
    pb.write_text(json.dumps(x))
    e = refused(SN.score_run, "nat_m640b", "final__base__s0", "dev")
    check("a recorded protocol score the native pass at 640 does not reproduce (+0.01) is refused, writing nothing",
          e is not None and "differs from the run's recorded protocol score" in str(e)
          and not (pb.parent / "dev@640.json").exists() and not (pb.parent / "dev@640.images.npz").exists(), e)
    base = dict(nr)
    for what, rec_, testing, frag in (
            ("other settings (batch 16), in production", dict(pr, settings=dict(pr["settings"], batch=16)), False,
             "not taken with these settings"),
            ("other weights", dict(pr, weights_sha256="0" * 64), True, "weights_sha256"),
            ("another scorer", dict(pr, scorer_sha256="TEST-" + "0" * 64), True, "another scorer"),
            ("no recorded score, in production", None, False, "no recorded protocol score")):
        e = refused(SN.compare_with_protocol, base, rec_, testing)
        check("the 640 reproduction refuses a recorded score with %s" % what, e is not None and frag in str(e), e)
    got = SN.compare_with_protocol(base, dict(pr, settings=dict(pr["settings"], batch=16)), True)
    check("  (test mode records other settings as not compared, rather than refusing)",
          got.get("compared") is False and "batch" in got.get("why", ""), got)


def test_rescore():
    print("rescore-native: the arm's finals at its imgsz, the reference's dev at 640, then the verdict")
    make_exp("nat_ref", "m640", testing=CPU32)
    for s in (0, 1, 2):
        make_final("nat_ref", s, 30 + s)       # the same weights as nat_m640: a separate reference
    shas0 = tree_shas(C.INC_DIR / "nat_m832")
    shas0.update(tree_shas(C.INC_DIR / "nat_ref"))
    out = C.INC_DIR / "cap_native"
    with reads() as seen:
        rec = B.rescore_native("nat_m832", reference="nat_ref", out_dir=out, resamples=100)
    sc = rec["scores"]
    check("every missing native score is written (dev, imageweeds at 832; s0's dev kept), the reference's dev at 640",
          sorted(sc) == ["dev", "imageweeds"] and [r["status"] for r in sc["dev"]] == ["kept", "written", "written"]
          and [r["status"] for r in sc["imageweeds"]] == ["written"] * 3
          and all(r["score"] == "dev@832.json" for r in sc["dev"])
          and [r["score"] for r in rec["reference"]["scores"]["dev"]] == ["dev@640.json"] * 3
          and rec["status"] == "complete" and (C.INC_DIR / "nat_m832" / "native_rescore.json").is_file(), rec)
    changed = [p for p, h in shas0.items() if pathlib.Path(p).is_file() and W.sha(p) != h]
    check("no file that existed before changed (every 640 px score, run.json, weights)", not changed, changed)
    rf = json.loads((C.INC_DIR / "nat_m832" / B.NATIVE_RECORD).read_text())
    check("native_rescore.json (the platform reads it) lists the dev files only, by name, with no path: the "
          "autopilot's scrub drops nothing",
          rf["status"] == "complete" and sorted(rf["scores"]) == ["dev"] and len(rf["scores"]["dev"]) == 3
          and rf["n_native_scores"] == 6 and "out" not in rf and not E.scrub(rf)[1] and not E.leaks(rf)
          and not [x for x in walk_values(rf) if isinstance(x, str) and (x.startswith("/") or x in ("test", "imageweeds")
                                                                       or x.startswith(("test@", "imageweeds@")))],
          (sorted(rf), E.scrub(rf)[1]))
    bad = touches_test(seen)
    check("nothing asked for, opened or scored the test exam (%d exam requests, %d paths opened)"
          % (len(seen["exams"]), len(seen["paths"])), not bad and "dev" in seen["exams"], bad[:5])
    allf = [p.name for e in ("nat_m832", "nat_ref") for p in (C.INC_DIR / e).rglob("*@*.json")]
    check("no native score of test exists (and the reference, whose finals include test, got dev only)",
          not [n for n in allf if n.startswith("test")] and not [n for n in allf if n.startswith("imageweeds@640")],
          sorted(allf))
    dec = json.loads((out / "native_v1.json").read_text())
    a = dec["arms"]["nat_m832"]
    check("the verdict is recorded beside capacity_v1.json: the arm decided on its 3 seeds against the reference",
          dec["format"] == B.NATIVE_VERDICT_FORMAT and a["status"] == "decided" and a["seeds"] == [0, 1, 2]
          and a["imgsz"] == 832 and a["reference_imgsz"] == 640 and dec["bootstrap"]["resamples"] == 100
          and dec["testing_allowed"] is True and isinstance(a["qualifies"], bool)
          and (out / "native_v1_report.json").is_file() and (out / "native_v1_report.md").is_file(), a.get("status"))
    _scrub, dropped = E.scrub(dec)
    check("the decision holds no non-dev exam key and no score path: the autopilot's scrub drops nothing",
          not dropped and not E.leaks(dec), dropped)
    rep = json.loads((out / "native_v1_report.json").read_text())
    check("the report (for people) has dev and imageweeds at 832 and at 640, never test",
          sorted(rep["arms"]["nat_m832"]["exams"]) == ["dev", "imageweeds"]
          and rep["arms"]["nat_m832"]["exams"]["imageweeds"]["native"]["n"] == 3
          and "test" not in rep["arms"]["nat_ref"]["exams"], rep["arms"]["nat_m832"]["exams"])
    first = json.dumps(dec["arms"]["nat_m832"], sort_keys=True)
    copy_arm("nat_m832", "nat_m832b")
    rb = B.rescore_native("nat_m832b", reference="nat_ref", out_dir=out, resamples=100)
    dec2 = json.loads((out / "native_v1.json").read_text())
    check("a second arm's rescore (its own job) keeps the first arm's decision: native_v1.json lists both, the first "
          "byte for byte", sorted(dec2["arms"]) == ["nat_m832", "nat_m832b"]
          and json.dumps(dec2["arms"]["nat_m832"], sort_keys=True) == first
          and dec2["arms"]["nat_m832b"]["status"] == "decided"
          and all(r["status"] == "kept" for v in rb["scores"].values() for r in v), sorted(dec2["arms"]))
    snap = tree_shas(C.INC_DIR / "nat_m832")
    rec2 = B.rescore_native("nat_m832", reference="nat_ref", out_dir=out, verdict=False)
    check("a second rescore writes nothing: every score kept, every file byte for byte",
          all(r["status"] == "kept" for v in rec2["scores"].values() for r in v)
          and {k: v for k, v in tree_shas(C.INC_DIR / "nat_m832").items() if not k.endswith(B.NATIVE_RECORD)}
          == {k: v for k, v in snap.items() if not k.endswith(B.NATIVE_RECORD)})
    e = refused(B.rescore_native, "nat_n640", reference="nat_ref", out_dir=out)
    e2 = refused(B.rescore_native, "nat_ref", reference="nat_ref", out_dir=out)
    check("rescore-native refuses a grid arm (n640, and the reference arm m640 itself)",
          e is not None and "measurement arm" in str(e) and e2 is not None and "only a measurement arm" in str(e2),
          (e, e2))
    before = sorted(p.name for p in (C.INC_DIR / "nat_r").rglob("*@*"))
    e = refused(B.rescore_native, "nat_r", reference="nat_ref", out_dir=out)
    check("... and an arm with a final run not done, before anything is scored",
          e is not None and "not done" in str(e) and sorted(p.name for p in (C.INC_DIR / "nat_r").rglob("*@*"))
          == before, e)
    e = refused(B.rescore_native, "nat_m832", reference="nat_m832", out_dir=out)
    check("... and a reference that is not the m640 arm", e is not None and "not the reference arm" in str(e), e)
    make_exp("nat_ref_s", "m640", seeds=(5, 6), testing=CPU32)
    rec_path = C.INC_DIR / "nat_m832" / B.NATIVE_RECORD
    rec0 = rec_path.read_bytes()
    e = refused(B.rescore_native, "nat_m832", reference="nat_ref_s", out_dir=out)
    check("... and a reference sharing fewer than 2 seeds with the arm, before anything is written",
          e is not None and "the comparison needs at least 2" in str(e) and rec_path.read_bytes() == rec0, e)
    # the production gate: test-mode native scores (kept, nothing scored) never enter a production verdict
    copy_arm("nat_m832", "nat_prod", testing=False)
    out_p = C.INC_DIR / "cap_native_prod"
    e = refused(B.rescore_native, "nat_prod", reference="nat_ref", out_dir=out_p, resamples=20)
    with no_testing_env():
        e2 = refused(B.rescore_native, "nat_m832", reference="nat_ref", out_dir=out_p, resamples=20)
    check("a production verdict never reads a kept test-mode native score: a non-testing experiment (with %s=1) and "
          "a testing one (without it) both refuse, writing no decision" % S.TEST_ENV,
          e is not None and "test-mode" in str(e) and e2 is not None and "test-mode" in str(e2)
          and not (out_p / "native_v1.json").exists(), (e, e2))


# ------------------------------------------------------------------ the rule
def synth_arrays(n=24, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    keys = ["k%03d" % i for i in range(n)]
    gt = np.random.default_rng(99)
    images = {}
    for i, k in enumerate(keys):
        tc = gt.integers(0, 12, int(gt.integers(1, 4))).astype(float)     # the same GT for every run (one exam)
        npred = int(rng.integers(3, 15))
        conf = rng.random(npred)
        images[k] = {"tp": rng.random((npred, 10)) < (conf[:, None] * 0.7), "conf": conf,
                     "pred_cls": rng.integers(0, 12, npred).astype(float), "target_cls": tc}
    return SC.flatten(images, keys)


def fake_native(exp, arm, vals, arrays, per_class=None, seeds=(0, 1, 2), settings=None, production=True,
                skip=()):
    """A baseline whose final runs carry native dev scores (the format scorer_native writes) with the
    given species_map50_95 and per-image arrays."""
    make_exp(exp, arm, seeds=seeds)
    imgsz = SN.native_imgsz(arm)
    for i, s in enumerate(seeds):
        if s in skip:
            continue
        scores = C.INC_DIR / exp / "runs" / ("final__base__s%d" % s) / "scores"
        js, npz = SN.paths_for(scores, "dev", imgsz)
        a = arrays[i] if isinstance(arrays, list) else arrays
        sha = SC.save_npz(npz, a)
        pc = dict((per_class or {}).get(s) or {n: 0.80 for n in SC.SPECIES})
        st = dict({"imgsz": imgsz, "batch": 32, "conf": 0.001, "iou": 0.7, "half": True, "rect": True,
                   "max_det": 300, "image_correct_conf": 0.25, "image_correct_iou": 0.5}, **(settings or {}))
        js.write_text(json.dumps({
            "format": SN.FORMAT, "exam": "dev", "imgsz": imgsz, "exp": exp, "run_id": "final__base__s%d" % s,
            "species_map50_95": vals[i], "per_class": pc, "native_production": production,
            "other_deviations": [] if production else ["fp32 on device cpu"],
            "manifest_sha256": "m" * 64, "key_order_sha256": C.sha256_text("\n".join(str(k) for k in a["keys"])),
            "n_images": len(a["keys"]), "locked_scorer_sha256": "s" * 64, "ultralytics_version": "8.4.37",
            "settings": st, "images": {"path": str(npz), "sha256": sha}}))


def walk_values(x):
    """Every dict key and scalar of a JSON document."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from walk_values(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk_values(v)
    else:
        yield x


def independent_se(arm_list, ref_list, resamples, seed_text):
    """SE of D recomputed without the module's bootstrap: per run and
    resample one ap_per_class call on the drawn images' concatenated
    (tie-broken) arrays, the 12-class mean over the species it returns."""
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class
    n = len(arm_list[0]["keys"])
    idx = np.random.default_rng(C.stable_int(seed_text)).integers(0, n, size=(resamples, n))

    def per_image(a):
        tb = SC.tie_break(a)
        return [(a["tp"][a["pred_img"] == i], tb["conf"][a["pred_img"] == i], a["pred_cls"][a["pred_img"] == i],
                 a["target_cls"][a["target_img"] == i]) for i in range(n)]
    runs = [per_image(a) for a in list(arm_list) + list(ref_list)]
    k, diffs = len(arm_list), []
    for b in range(resamples):
        tw = []
        for r in runs:
            parts = [r[i] for i in idx[b]]
            res = ap_per_class(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                               np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))
            ap, classes = res[5], res[6]
            vals = [float(ap[j].mean()) for j, c in enumerate(classes) if int(c) != C.OTHER_PLANT]
            tw.append(sum(vals) / len(vals))
        diffs.append(sum(tw[:k]) / k - sum(tw[k:]) / (len(tw) - k))
    return float(np.std(np.asarray(diffs), ddof=1))


def test_rule():
    print("the pre-registered rule on synthetic numbers")
    check("pre-registered constants: reference b_v2_m640, targets Carpetweed / SpottedSpurge / Purslane, 1,000 "
          "resamples under stable_int('inc2/native/diff_se')",
          B.NATIVE_REFERENCE == "b_v2_m640" and B.NATIVE_TARGET_SPECIES == ("Carpetweed", "SpottedSpurge", "Purslane")
          and B.NATIVE_RESAMPLES == 1000 and B.NATIVE_SEED_TEXT == "inc2/native/diff_se"
          and B.NATIVE_ARMS == ("b_v2_m832", "b_v2_s1024"))
    ref = synth_arrays(seed=1)
    other = [synth_arrays(seed=10 + i) for i in range(3)]
    fake_native("v_ref", "m640", [0.850, 0.851, 0.849], ref)
    up = {s: dict({n: 0.80 for n in SC.SPECIES}, Carpetweed=0.82) for s in (0, 1, 2)}
    down = {s: dict({n: 0.83 for n in SC.SPECIES}, Carpetweed=0.79, SpottedSpurge=0.79, Purslane=0.80)
            for s in (0, 1, 2)}
    fake_native("v_q", "m832", [0.861, 0.862, 0.860], ref, up)
    fake_native("v_sd", "m832", [0.880, 0.845, 0.865], ref, up)
    fake_native("v_se", "s1024", [0.861, 0.862, 0.860], other, up)
    fake_native("v_sp", "m832", [0.861, 0.862, 0.860], ref, down)
    fake_native("v_pend", "m832", [0.861, 0.862, 0.860], ref, up, skip=(2,))
    dec = B.native_decision(["v_q", "v_sd", "v_se", "v_sp", "v_pend", "v_none"], reference="v_ref", resamples=200)
    A = dec["arms"]
    q = A["v_q"]
    check("qualifies: D %.4f > 2 pooled sd %.4f and > SE %.4f (identical arrays), Carpetweed improved"
          % (q["diff"], q["two_pooled_sd"], q["se_diff"]),
          q["qualifies"] and all(q["conditions"].values()) and q["se_diff"] == 0.0
          and abs(q["diff"] - 0.011) < 1e-9 and q["improved_targets"] == ["Carpetweed"] and dec["qualifying"] == ["v_q"],
          q["conditions"])
    for e, cond in (("v_sd", "above_2_pooled_sd"), ("v_se", "above_se"), ("v_sp", "target_species_improved")):
        c = A[e]["conditions"]
        check("%s fails %s alone and does not qualify (D %.4f, 2 pooled sd %.4f, SE %.4f, improved %s)"
              % (e, cond, A[e]["diff"], A[e]["two_pooled_sd"], A[e]["se_diff"], A[e]["improved_targets"]),
              not A[e]["qualifies"] and c[cond] is False and all(v for k, v in c.items() if k != cond), c)
    off = [e for e in ("v_q", "v_sd", "v_se", "v_sp")
           if abs(A[e]["pooled_sd"] - math.sqrt((statistics.stdev(A[e]["dev"]) ** 2
                                                 + statistics.stdev(A[e]["reference_dev"]) ** 2) / 2.0)) > 1e-12
           or A[e]["two_pooled_sd"] != 2.0 * A[e]["pooled_sd"]]
    check("pooled sd is sqrt((sd_arm^2 + sd_ref^2) / 2) of the seeds' sample sds, in every decided case", not off, off)
    fake_native("v_ref2", "m640", [0.850, 0.8505, 0.8495], ref)
    fake_native("v_pool", "m832", [0.854, 0.860, 0.866], ref, up)
    pa = B.native_decision(["v_pool"], reference="v_ref2", resamples=20)["arms"]["v_pool"]
    sa, sr = statistics.stdev(pa["dev"]), statistics.stdev(pa["reference_dev"])
    check("unequal sds (arm %.4f, reference %.4f): D %.4f > 2 pooled sd %.4f qualifies, though D is below 2 x the "
          "larger sd and 2 x sqrt(sd_arm^2 + sd_ref^2)" % (sa, sr, pa["diff"], pa["two_pooled_sd"]),
          abs(pa["pooled_sd"] - math.sqrt((sa ** 2 + sr ** 2) / 2.0)) < 1e-12 and pa["qualifies"]
          and pa["conditions"]["above_2_pooled_sd"] and pa["diff"] < 2.0 * max(sa, sr)
          and pa["diff"] < 2.0 * math.sqrt(sa ** 2 + sr ** 2), pa["conditions"])
    check("an arm missing a native dev score is pending (named), one not built is pending",
          A["v_pend"]["status"] == "pending" and A["v_pend"]["missing"] == ["v_pend/final__base__s2"]
          and A["v_none"] == {"status": "pending", "why": "not built"} and dec["pending"] == ["v_pend", "v_none"],
          (A["v_pend"], A["v_none"]))
    fake_native("v_mix", "m832", [0.861, 0.862, 0.860], ref, up)
    p = C.INC_DIR / "v_mix" / "runs" / "final__base__s1" / "scores" / "dev@832.json"
    x = json.loads(p.read_text())
    x["settings"]["batch"] = 16
    p.write_text(json.dumps(x))
    e = refused(B.native_decision, ["v_mix"], reference="v_ref", resamples=20)
    check("a native file with another setting than the comparison's refuses", e is not None and "batch" in str(e), e)
    fake_native("v_st", "m832", [0.861, 0.862, 0.860], ref, up)
    fake_native("v_st_ref", "m640", [0.850, 0.851, 0.849], ref)
    ok0 = B.native_decision(["v_st"], reference="v_st_ref", resamples=5, testing_ok=True)["arms"]["v_st"]["status"]
    # the pre-registered list, written out (not read from the module, so a field dropped there is missed here)
    fields = [(None, k) for k in ("manifest_sha256", "key_order_sha256", "n_images", "locked_scorer_sha256",
                                  "ultralytics_version", "native_production")]
    fields += [("settings", k) for k in ("batch", "conf", "iou", "half", "rect", "max_det", "image_correct_conf",
                                         "image_correct_iou")]
    missed = []
    for who, e_, seeds in (("one arm file", "v_st", (1,)), ("the reference's files", "v_st_ref", (0, 1, 2))):
        size = 832 if e_ == "v_st" else 640
        for parent, k in fields:
            files = [SN.paths_for(C.INC_DIR / e_ / "runs" / ("final__base__s%d" % s_) / "scores", "dev", size)
                     for s_ in seeds]
            saved = {p_: p_.read_bytes() for pair in files for p_ in pair}
            for js_, npz_ in files:
                x = json.loads(js_.read_text())
                if k == "key_order_sha256":     # arrays in another order that hash and key as recorded
                    arr = SC.load_npz(npz_)
                    rev = dict(arr, keys=arr["keys"][::-1])
                    x["images"]["sha256"] = SC.save_npz(npz_, rev)
                    x[k] = C.sha256_text("\n".join(str(t) for t in rev["keys"]))
                else:
                    tgt = x[parent] if parent else x
                    v = tgt[k]
                    tgt[k] = (not v) if isinstance(v, bool) else (v + 1) if isinstance(v, (int, float)) else v + "x"
                js_.write_text(json.dumps(x))
            e = refused(B.native_decision, ["v_st"], reference="v_st_ref", resamples=5, testing_ok=True)
            name = "%s.%s" % (parent, k) if parent else k
            if e is None or "(%s differ)" % name not in str(e):
                missed.append((who, name, str(e)[:160]))
            for p_, b_ in saved.items():
                p_.write_bytes(b_)
    check("every stamp and setting of the comparison is compared (%d fields), within an arm and between the arm and "
          "the reference: each mismatch alone refuses, naming it" % len(fields), ok0 == "decided" and not missed,
          missed[:4])
    fake_native("v_tm", "m832", [0.861, 0.862, 0.860], ref, up, production=False)
    fake_native("v_ref_tm", "m640", [0.850, 0.851, 0.849], ref, production=False)
    e = refused(B.native_decision, ["v_tm"], reference="v_ref_tm", resamples=20)
    ok = B.native_decision(["v_tm"], reference="v_ref_tm", resamples=20, testing_ok=True)
    mixed = refused(B.native_decision, ["v_tm"], reference="v_ref", resamples=20, testing_ok=True)
    check("a test-mode score refuses a production verdict; only testing_ok (tests) reads it, and never mixed with "
          "native-protocol scores",
          e is not None and "test-mode" in str(e) and ok["arms"]["v_tm"]["status"] == "decided"
          and mixed is not None and "native_production" in str(mixed), (e, mixed))
    fake_native("v_npz", "m832", [0.861, 0.862, 0.860], ref, up)
    npz = C.INC_DIR / "v_npz" / "runs" / "final__base__s0" / "scores" / "dev@832.images.npz"
    SC.save_npz(npz, other[0])
    e = refused(B.native_decision, ["v_npz"], reference="v_ref", resamples=20)
    check("an npz that does not hash as its score records refuses", e is not None and "does not hash" in str(e), e)
    fake_native("v_run", "m832", [0.861, 0.862, 0.860], ref, up)
    p1 = C.INC_DIR / "v_run" / "runs" / "final__base__s1" / "scores" / "dev@832.json"
    x = json.loads(p1.read_text())
    x["run_id"] = "final__base__s0"
    p1.write_text(json.dumps(x))
    e = refused(B.native_decision, ["v_run"], reference="v_ref", resamples=20)
    check("a native file that names another run (or exam, size, experiment) refuses", e is not None
          and "is not v_run final__base__s1's native dev score" in str(e), e)
    fake_native("v_keys", "m832", [0.861, 0.862, 0.860], ref, up)
    p2 = C.INC_DIR / "v_keys" / "runs" / "final__base__s0" / "scores"
    x = json.loads((p2 / "dev@832.json").read_text())
    x["images"]["sha256"] = SC.save_npz(p2 / "dev@832.images.npz", dict(ref, keys=ref["keys"][::-1]))
    (p2 / "dev@832.json").write_text(json.dumps(x))
    e = refused(B.native_decision, ["v_keys"], reference="v_ref", resamples=20)
    check("per-image arrays not in the exam key order the score records refuse (even when they hash as recorded)",
          e is not None and "not in the exam key order" in str(e), e)
    fake_native("v_seeds", "m832", [0.861, 0.862], [ref, ref], up, seeds=(0, 5))
    e = refused(B.native_decision, ["v_seeds"], reference="v_ref", resamples=20)
    check("an arm sharing fewer than 2 seeds with the reference refuses", e is not None and "share seeds" in str(e), e)
    gt = dict(ref, target_cls=ref["target_cls"][::-1].copy())
    e = refused(B.native_bootstrap, [gt], [ref], resamples=5)
    check("runs whose GT boxes differ refuse the bootstrap (one exam)", e is not None and "GT boxes differ" in str(e), e)
    e = refused(B.native_decision, ["v_ref"], reference="v_ref", resamples=20)
    check("the reference arm (a grid arm) is not an arm of the rule", e is not None and "measurement arm" in str(e), e)
    e = refused(B.native_decision, ["v_q"], reference="v_q", resamples=20)
    check("a reference that is not the m640 arm refuses", e is not None and "not the reference arm" in str(e), e)

    print("the paired bootstrap")
    b1 = B.native_bootstrap(other, [ref, ref, ref], resamples=40)
    b2 = B.native_bootstrap(other, [ref, ref, ref], resamples=40)
    b3 = B.native_bootstrap(other, [ref, ref, ref], resamples=40, seed_text="another")
    check("deterministic under its seed text, and another seed text draws other resamples",
          b1 == b2 and b1["se"] != b3["se"] and b1["n_valid"] == 40, (b1["se"], b3["se"]))
    ind = independent_se(other, [ref, ref, ref], 40, B.NATIVE_SEED_TEXT)
    check("its SE equals an independent recomputation (%.6f vs %.6f)" % (b1["se"], ind), abs(b1["se"] - ind) < 1e-9,
          (b1["se"], ind))
    z = B.native_bootstrap([ref, ref], [ref, ref], resamples=30)
    check("identical runs on both sides: SE 0, every species' SE 0", z["se"] == 0.0
          and all(v["se"] in (0.0, None) for v in z["per_species"].values()), z["se"])
    shuffled = dict(ref, keys=ref["keys"][::-1])
    e = refused(B.native_bootstrap, [shuffled], [ref], resamples=5)
    check("runs in another key order refuse (paired by exam image)", e is not None and "key order" in str(e), e)

    out = C.INC_DIR / "cap_rule"
    rc = B.main(["native-verdict", "--arms", "v_q,v_sp", "--reference", "v_ref", "--out-dir", str(out)])
    dec = json.loads((out / "native_v1.json").read_text())
    check("the CLI (native-verdict, the pre-registered 1,000 resamples) writes the decision: v_q qualifies, v_sp not",
          rc == 0 and dec["qualifying"] == ["v_q"] and dec["bootstrap"]["resamples"] == 1000
          and dec["arms"]["v_sp"]["qualifies"] is False, dec.get("qualifying"))
    check("... dev only: no non-dev exam key, no score path, nothing the autopilot's scrub would drop",
          not E.scrub(dec)[1] and not E.leaks(dec)
          and not [x for x in walk_values(dec) if x in ("test", "imageweeds")
                   or (isinstance(x, str) and x.startswith(("test@", "test.", "imageweeds@", "imageweeds.")))],
          E.scrub(dec)[1])


def main():
    t0 = time.time()
    try:
        W.build_world()
        test_refusals()
        test_commit()
        test_offline_env()
        test_real_scores()
        test_rescore()
        test_rule()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
