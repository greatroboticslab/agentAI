#!/usr/bin/env python3
"""E1's recipe, its agnostic rescore, its verdict and its test read
(inc2/recipes.py cold_budget, inc2/train.py, inc2/scorer_agnostic.py,
inc2.baseline rescore-agnostic / agnostic-verdict / e1-test-read;
docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-03): E1, weed-box base v3
(pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 exams materialised, the
v1 LOCK with scorer.py's sha256, LOCK v2 marked testing); scores are real
Ultralytics passes on the CPU in test mode.

What is pinned:
- cold_budget: epochs = round_half_up(1.2M / N), warmup_epochs = round(640 /
  ceil(N / 32), 6) (Ultralytics then warms up for exactly 640 iterations at
  every N from 1,000 to 200,000), close_mosaic = max(1, round_half_up(0.1 x
  epochs)), a half rounds up; every other key m640's cold recipe; only m640
  has one; deviations() and match() accept it for kind base only and only
  when named with the base's N; a union or an incremental run naming it, or
  another N, departs;
- inc2.train: a production base run of an exp.json that names cold_budget
  (with its base's n_images and the pre-registered budget record) passes
  stage recipe and stops at device (no CUDA); a recipe one epoch off, a
  budget record that differs, a chain naming it or a union run naming it are
  refused at stage recipe; the pinned driver accepts the recipe (float
  warmup) in a definition and in a base run's spec;
- inc2.train.guard_verdicts gives guard_rows' record and refusals per row;
- the agnostic scorer: refuses every exam but dev, a run that is not a done
  final run of a baseline, weights that do not hash as recorded, a recorded
  score whose agnostic AP is off by more than 0.002; its arrays reproduce the
  pass's agnostic AP (capture order) and the recorded one (tie-broken, key
  order); written once (kept on a second call); nothing touches test;
- the bootstrap: deterministic, identical runs give SE 0, and it equals an
  independent recomputation;
- the verdict: the rule on synthetic numbers (qualifies with both
  conditions, fails with each alone, pooled sd = sqrt((sd_B^2 + sd_A^2)/2));
  a missing file is pending; arms built from other summaries, or the arms
  swapped, refuse; rescore-agnostic writes agnostic_rescore.json (complete,
  no path) and capacity/e1_v1.json (dev only: the autopilot's scrub drops
  nothing);
- e1-test-read: refused without a decided verdict; once decided, one
  kind-final spec per seed on test (the v2 executor accepts it) and the
  run_inc2_job.sh argv for a person; refused once a test score exists and
  when the verdict changed.

Run:  python3 tests/test_inc2_e1.py
"""
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_agnostic as SA  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402

FAILURES = W.FAILURES
check = W.check
CPU32 = {"batch": 32, "device": "cpu"}
SUMMARY_SHA = "5" * 64


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (SA.AgnosticRefused, B.BaselineError, RC.RecipeError, D.DriverError) as e:
        return e
    return None


# ------------------------------------------------------------------ recipe
def test_recipe():
    print("cold_budget")
    ok = []
    for n in list(range(1000, 200001, 997)) + [6811, 30000, 34400, 23500]:
        r = RC.cold_budget("m640", n)
        nb = int(math.ceil(n / 32.0))
        ok.append(max(int(round(r["warmup_epochs"] * nb)), 100) == 640
                  and r["epochs"] == int(math.floor(1.2e6 / n + 0.5))
                  and r["close_mosaic"] == max(1, int(math.floor(0.1 * r["epochs"] + 0.5))))
    check("Ultralytics warms up for exactly 640 iterations at every N (1,000-200,000), epochs and close_mosaic by "
          "the formula", all(ok), ok.count(False))
    a, b = RC.cold_budget("m640", 6811), RC.cold_budget("m640", 30000)
    check("E1-A at 6,811 images: 176 epochs, warmup 3.004695, close_mosaic 18; E1-B at 30,000: 40, 0.682303, 4",
          (a["epochs"], a["warmup_epochs"], a["close_mosaic"]) == (176, 3.004695, 18)
          and (b["epochs"], b["warmup_epochs"], b["close_mosaic"]) == (40, 0.682303, 4), (a, b))
    check("a half rounds up (Python's round would give the even neighbour): 1.2M / 480,000 = 2.5 -> 3 epochs; "
          "0.1 x 45 = 4.5 -> close_mosaic 5",
          RC.cold_budget("m640", 480000)["epochs"] == 3 and RC.budget_close_mosaic(45) == 5 and round(4.5) == 4)
    same = {k: v for k, v in a.items() if k not in ("epochs", "warmup_epochs", "close_mosaic")}
    cold = {k: v for k, v in RC.cold("m640").items() if k not in ("epochs", "warmup_epochs", "close_mosaic")}
    check("every other key is m640's cold recipe", same == cold, (same, cold))
    check("only m640 has one (E1 is pre-registered on YOLO11m at 640)", refused(RC.cold_budget, "n640", 6811)
          is not None and refused(RC.cold_budget, "l640", 6811) is not None)
    check("deviations: none for a base run named cold_budget with its own N; the cold table's three keys without the "
          "name", RC.deviations("base", a, "m640", "cold_budget", 6811) == []
          and len(RC.deviations("base", a, "m640")) == 3)
    check("a union or an incremental run naming it departs; so does another N, and a recipe one epoch off",
          RC.deviations("union", a, "m640", "cold_budget", 6811) != []
          and RC.deviations("cand", a, "m640", "cold_budget", 6811) != []
          and RC.deviations("base", a, "m640", "cold_budget", 7000) != []
          and RC.deviations("base", dict(a, epochs=177), "m640", "cold_budget", 6811) != [])
    check("match names it, and nothing else does", RC.match("base", a, "m640", "cold_budget", 6811) == "cold_budget"
          and RC.match("base", a, "m640") is None and RC.match("base", RC.cold("m640"), "m640") == "cold")
    rec = RC.budget_record("m640", 6811)
    check("the budget record exp.json carries checks against the constants",
          RC.check_budget_record(rec, "m640", 6811) == [] and RC.check_budget_record(dict(rec, image_epochs=2e6),
                                                                                       "m640", 6811) != []
          and RC.check_budget_record(None, "m640", 6811) != [])
    c = RC.budget_cost(6811, [0, 1, 2], ["dev", "imageweeds"])
    check("budget_cost: 4.73 h of training per run (1.2M x 14.2 ms) whatever N, under D26's 6.4 h line",
          abs(1.2e6 * 14.2 / 3.6e6 - 4.733) < 1e-3 and c["walltime"]["over_d26_line"] == [False, False]
          and RC.budget_cost(34000, [0, 1, 2], ["dev"])["per_run_gpu_h"][0] == c["per_run_gpu_h"][0])
    spec = {"exp": "x", "run_id": "base__s0", "kind": "base", "init": "yolo11m.pt", "train_manifest": "/a.jsonl",
            "recipe": dict(a, seed=0), "exams": ["dev"], "out_dir": "/x"}
    D.validate_recipe(dict(a), "base")
    D.validate_spec(spec)
    check("the pinned driver accepts the recipe (a float warmup_epochs) in a definition and in a base run's spec", True)


def write_summary(manifest_a, status="complete"):
    """A splits v3 summary.json recording manifest_a as E1's arm A (and a stand-in arm B); its sha256."""
    from weed_optimizer_framework.tools.inc2 import base3 as B3
    sp = B3.out_dir() / B3.SUMMARY
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"format": B3.FORMAT, "status": status,
                              "arms": {"A": {"manifest": str(manifest_a), "sha256": W.sha(manifest_a)},
                                       "B": {"manifest": "/nowhere.jsonl", "sha256": "b" * 64}}}))
    return W.sha(sp)


def write_exp(exp, n, typ="baseline", recipe_name=RC.BUDGET_NAME, budget=True, arm="m640", manifest=None, e1=True):
    root = C.INC_DIR / exp
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rec = RC.resolve_arm(arm, repo=C.REPO)
    defn = {"exp": exp, "type": typ, "base": {"n_images": n}}
    if manifest is not None:
        defn["base"]["manifest_sha256"] = W.sha(manifest)
    if e1 is True:
        from weed_optimizer_framework.tools.inc2 import base3 as B3
        sp = B3.out_dir() / B3.SUMMARY
        defn["e1"] = {"arm": "A", "summary_sha256": W.sha(sp) if sp.is_file() else None}
    elif isinstance(e1, dict):
        defn["e1"] = e1
    defn.update(RC.stamp(rec))
    if recipe_name:
        defn["recipe_name"] = recipe_name
    if budget:
        defn["budget"] = RC.budget_record(arm, n) if budget is True else budget
    (root / "exp.json").write_text(json.dumps(defn))


def test_train_recipe():
    print("inc2.train: a production base run of cold_budget")
    (C.REPO / "yolo11m.pt").write_bytes(os.urandom(4096))
    base_m = str(W.v2_dir() / "base_v2.jsonl")
    n = len(C.read_manifest(base_m))
    import torch

    def go(exp, run_id, kind, recipe, **kw):
        p = W.spec(run_id, kind, exp=exp, init="yolo11m.pt", train_manifest=base_m, recipe=recipe, **kw)
        rc = W.run(p)
        return rc, W.run_json(p)
    write_summary(base_m)
    write_exp("e1t_prod", n, manifest=base_m)
    rc, rj = go("e1t_prod", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    if torch.cuda.is_available():
        print("  NOTE a CUDA device is present; the production device refusal is not exercised")
    else:
        check("exp.json names cold_budget with its base's N and the pre-registered budget: the recipe passes stage "
              "recipe and stops at device (no CUDA here); run.json names cold_budget",
              rc == 1 and rj.get("stage") == "device" and rj.get("recipe_name") == "cold_budget"
              and rj.get("protocol_recipe") is True, (rj.get("stage"), (rj.get("error") or "")[-300:]))
    rc, rj = go("e1t_prod", "base__s1", "base", dict(RC.cold_budget("m640", n), epochs=177, seed=1))
    check("one epoch off: refused at stage recipe", rc == 1 and rj.get("stage") == "recipe", rj.get("stage"))
    rc, rj = go("e1t_prod", "base__s2", "base", dict(RC.cold("m640"), seed=2))
    check("the cold table under an exp.json that names cold_budget: refused", rc == 1 and rj.get("stage") == "recipe",
          rj.get("stage"))
    write_exp("e1t_bud", n, budget=dict(RC.budget_record("m640", n), image_epochs=2400000), manifest=base_m)
    rc, rj = go("e1t_bud", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    check("a budget record that is not the pre-registered one: refused at stage recipe",
          rc == 1 and rj.get("stage") == "recipe" and "budget" in (rj.get("error") or ""),
          (rj.get("error") or "")[-300:])
    write_exp("e1t_chain", n, typ="chain", manifest=base_m)
    rc, rj = go("e1t_chain", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    check("a chain naming cold_budget: refused (baseline experiments only)", rc == 1 and rj.get("stage") == "recipe",
          rj.get("stage"))
    write_exp("e1t_union", n, manifest=base_m)
    rc, rj = go("e1t_union", "union__s0", "union", dict(RC.cold_budget("m640", n), seed=0))
    check("a union run under an exp.json naming cold_budget: refused (base runs only)",
          rc == 1 and rj.get("stage") == "recipe", rj.get("stage"))
    # cold_budget is E1's: the base manifest must be an arm of the complete splits v3 summary the e1 record names
    out = {}
    write_exp("e1t_noe1", n, manifest=base_m, e1=False)
    out["no e1 record"] = go("e1t_noe1", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    other = W.TMP / "e1_other.jsonl"
    C.write_manifest(other, C.read_manifest(base_m))
    other.write_text(other.read_text() + "\n")
    write_exp("e1t_other", n, manifest=other)
    out["a base manifest the summary does not record"] = go("e1t_other", "base__s0", "base",
                                                            dict(RC.cold_budget("m640", n), seed=0))
    write_exp("e1t_arm", n, manifest=base_m, e1={"arm": "B", "summary_sha256": write_summary(base_m)})
    out["the other arm"] = go("e1t_arm", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    write_exp("e1t_moved", n, manifest=base_m)
    write_summary(base_m, status="over_walltime")
    out["an over_walltime summary"] = go("e1t_moved", "base__s0", "base", dict(RC.cold_budget("m640", n), seed=0))
    write_exp("e1t_changed", n, manifest=base_m, e1={"arm": "A", "summary_sha256": "0" * 64})
    write_summary(base_m)
    out["a summary that changed since the build"] = go("e1t_changed", "base__s0", "base",
                                                       dict(RC.cold_budget("m640", n), seed=0))
    check("cold_budget only for an E1 arm: refused at stage recipe with %s" % ", ".join(out),
          all(rc_ == 1 and rj_.get("stage") == "recipe" for rc_, rj_ in out.values()),
          {k: (rj_.get("stage"), (rj_.get("error") or "")[-200:]) for k, (rc_, rj_) in out.items()})


def test_guard_verdicts(Wd):
    print("inc2.train.guard_verdicts")
    rows = list(Wd["base_v2"][:6])
    copy = W.planted("gv_copy", Wd["dev"][1]["image"], "hflip")
    rows.append(W.row_for("gv_copy", copy, [(0, 0.5, 0.5, 0.2, 0.2)], "x", ""))
    dh = {r["image"]: C.dhash(r["image"]) for r in rows}
    per, rec, refused_, cc = T.guard_verdicts(rows, dh, production=False)
    try:
        T.guard_rows(rows, dh, production=False)
        err = None
    except T.RunError as e:
        err = e
    check("guard_verdicts: the hflip dev copy refused, the base rows counted as base copies, per row; guard_rows "
          "refuses on the same record",
          per["gv_copy"]["refused"] and not any(per[r["key"]]["refused"] for r in rows[:6])
          and all("base_copy" in per[r["key"]]["reasons"] for r in rows[:6]) and err is not None
          and err.guard["reasons"] == rec["reasons"] and len(per["gv_copy"]["variants"]) == 8, (per["gv_copy"], rec))


# ------------------------------------------------------------- the scorer
def named_checkpoint(path, seed=0):
    import torch
    W.cold_checkpoint(path, seed=seed)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    ck["model"].names = {i: n for i, n in enumerate(C.CLASS_NAMES)}
    torch.save(ck, str(path))
    return path


def make_arm(exp, e1_arm, seeds=(0, 1, 2), wseed0=20, summary=SUMMARY_SHA, protocol=True, testing=CPU32):
    root = C.INC_DIR / exp
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    defn = {"exp": exp, "type": "baseline", "role": "baseline", "seeds": list(seeds),
            "final_exams": ["dev", "imageweeds"], "testing": testing, "recipe_name": RC.BUDGET_NAME,
            "base": {"n_images": 100 if e1_arm == "A" else 300},
            "e1": {"arm": e1_arm, "summary_sha256": summary, "manifests": {"A": "a" * 64, "B": "b" * 64}}}
    defn.update(RC.stamp(RC.resolve_arm("m640", require_weights=False)))
    (root / "exp.json").write_text(json.dumps(defn))
    for i, s in enumerate(seeds):
        base = named_checkpoint(root / "runs" / ("base__s%d" % s) / "weights" / "final.pt", seed=wseed0 + i)
        rd = root / "runs" / ("final__base__s%d" % s)
        (rd / "weights").mkdir(parents=True, exist_ok=True)
        os.symlink(str(base), str(rd / "weights" / "final.pt"))
        (rd / "run.json").write_text(json.dumps({"status": "done", "weights_sha256": W.sha(base),
                                                 "weights": str(rd / "weights" / "final.pt")}))
        if protocol:
            S.score(rd / "weights" / "final.pt", "dev", rd / "scores" / "dev.json", lock_check=True, imgsz=640,
                    batch=32, device="cpu")
    return root


def test_scorer():
    print("the agnostic scorer (inc2.scorer_agnostic)")
    root = make_arm("e1t_sa", "B", seeds=(0, 1))
    e = refused(SA.score_run, "e1t_sa", "final__base__s0", "test")
    e2 = refused(SA.score_run, "e1t_sa", "final__base__s0", "imageweeds")
    check("every exam but dev is refused (test, imageweeds)", e is not None and e2 is not None, (e, e2))
    for what, run, frag in (("a base run", "base__s0", "final runs"), ("a seed it does not have", "final__base__s7",
                                                                     "final runs")):
        e = refused(SA.score_run, "e1t_sa", run, "dev")
        check("refused: %s" % what, e is not None and frag in str(e), e)
    rj = root / "runs" / "final__base__s1" / "run.json"
    good = rj.read_text()
    rj.write_text(json.dumps(dict(json.loads(good), status="running")))
    e = refused(SA.score_run, "e1t_sa", "final__base__s1", "dev")
    rj.write_text(json.dumps(dict(json.loads(good), weights_sha256="0" * 64)))
    e2 = refused(SA.score_run, "e1t_sa", "final__base__s1", "dev")
    rj.write_text(good)
    check("refused: a run not done, weights that do not hash as recorded", e is not None and e2 is not None, (e, e2))
    sp = root / "runs" / "final__base__s1" / "scores" / "dev.json"
    keep = sp.read_text()
    sp.write_text(json.dumps(dict(json.loads(keep), agnostic_map50_95=json.loads(keep)["agnostic_map50_95"] + 0.01)))
    e = refused(SA.score_run, "e1t_sa", "final__base__s1", "dev")
    sp.write_text(keep)
    check("a recorded agnostic AP 0.01 off the pass: refused, nothing written", e is not None and "differs" in str(e)
          and not (sp.parent / "dev.agnostic.json").exists(), e)
    seen = []
    real = C.manifest_path
    C.manifest_path = lambda exam, *a, **k: (seen.append(exam), real(exam, *a, **k))[1]
    try:
        rec = SA.score_run("e1t_sa", "final__base__s0", "dev")
    finally:
        C.manifest_path = real
    js, npz = SA.paths_for(root / "runs" / "final__base__s0" / "scores", "dev")
    arr = SA.load_npz(npz)
    recd = json.loads((root / "runs" / "final__base__s0" / "scores" / "dev.json").read_text())
    check("written: dev.agnostic.json and its npz; the arrays (exam key order, tie-broken) reproduce the recorded "
          "agnostic AP %.4f; production false; nothing asked for test" % recd["agnostic_map50_95"],
          rec["status"] == "written" and js.is_file() and W.sha(npz) == rec["images"]["sha256"]
          and abs(SA.collapsed(SC.tie_break(arr))[0] - recd["agnostic_map50_95"]) <= SA.MAX_TIE_DIFF
          and rec["checks"]["recompute"] <= SA.MAX_RECOMPUTE_DIFF and rec["production"] is False
          and rec["recorded"]["agnostic_map50_95"] == recd["agnostic_map50_95"]
          and list(arr["keys"]) == sorted(r["key"] for r in C.read_manifest(C.manifest_path("dev")))
          and "test" not in seen and len(arr["n_gt"]) == len(arr["keys"])
          and int(arr["n_gt"].sum()) == sum(int(x) for x in recd["n_gt"].values()) == rec["n_gt"]
          and len(arr["conf"]) > 0
          and arr["tp"].shape == (len(arr["conf"]), 10), (rec.get("checks"), seen))
    j0 = js.read_bytes()
    again = SA.score_run("e1t_sa", "final__base__s0", "dev")
    check("written once: a second call keeps the file", again["status"] == "kept" and js.read_bytes() == j0)


# ---------------------------------------------------------- the bootstrap
def synth(n=24, seed=0, gt_seed=99):
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(gt_seed)
    images = {}
    keys = ["k%03d" % i for i in range(n)]
    for k in keys:
        npred = int(rng.integers(2, 12))
        conf = rng.random(npred)
        images[k] = {"tp": rng.random((npred, 10)) < conf[:, None] * 0.8, "conf": conf,
                     "n_gt": int(g.integers(1, 4))}
    return SA.flatten(images, keys)


def test_bootstrap():
    print("the paired image bootstrap of D")
    a = [synth(seed=1), synth(seed=2)]
    b = [synth(seed=3), synth(seed=4)]
    r1 = B.agnostic_bootstrap(b, a, resamples=60)
    r2 = B.agnostic_bootstrap(b, a, resamples=60)
    same = B.agnostic_bootstrap(a, a, resamples=30)
    check("deterministic under stable_int('inc2/e1/agnostic_se'); identical runs give SE 0",
          r1 == r2 and r1["se"] > 0 and same["se"] == 0.0, (r1, same))
    idx = SC.resample_indices(24, 60, B.E1_SEED_TEXT)
    diffs = []
    for row in idx:
        vals = []
        for arr in b + a:
            tb = SC.tie_break(arr)
            tps, confs, n_gt = [], [], 0
            for i in row:
                sel = tb["pred_img"] == i
                tps.append(tb["tp"][sel])
                confs.append(tb["conf"][sel])
                n_gt += int(tb["n_gt"][i])
            vals.append(S.collapsed_ap(np.concatenate(tps), np.concatenate(confs), n_gt)[0])
        diffs.append(statistics.fmean(vals[:2]) - statistics.fmean(vals[2:]))
    check("its SE equals an independent recomputation (the drawn images' arrays concatenated)",
          abs(float(np.std(diffs, ddof=1)) - r1["se"]) < 1e-12, (np.std(diffs, ddof=1), r1["se"]))
    bad = dict(a[1], n_gt=a[1]["n_gt"] + 1)
    check("runs of another exam (other GT counts) refuse", refused(B.agnostic_bootstrap, b, [a[0], bad], 10)
          is not None)


def fake_agnostic(exp, e1_arm, vals, arrays, summary=SUMMARY_SHA, skip=(), production=True):
    """An E1 arm whose final runs carry agnostic dev files with the given recorded values and arrays."""
    seeds = tuple(range(len(vals)))
    make_arm(exp, e1_arm, seeds=seeds, summary=summary, protocol=False)
    for s, (v, arr) in enumerate(zip(vals, arrays)):
        if s in skip:
            continue
        sc = C.INC_DIR / exp / "runs" / ("final__base__s%d" % s) / "scores"
        js, npz = SA.paths_for(sc, "dev")
        sha = SA.save_npz(npz, arr)
        js.write_text(json.dumps({"format": SA.FORMAT, "exp": exp, "run_id": "final__base__s%d" % s, "exam": "dev",
                                  "agnostic_production": production, "images": {"sha256": sha},
                                  "key_order_sha256": C.sha256_text("\n".join(str(k) for k in arr["keys"])),
                                  "n_gt": int(arr["n_gt"].sum()),
                                  "recorded": {"agnostic_map50_95": v,
                                               "stamps": {"exam": "dev", "scorer_sha256": "s", "manifest_sha256": "m",
                                                          "key_order_sha256": "k", "n_images": 24}}}))


def test_verdict():
    print("the E1 verdict (dev only)")
    arrs = [synth(seed=i) for i in range(3)]
    out = W.TMP / "e1_cap"
    cases = {"qualifies": ([0.86, 0.87, 0.88], [0.80, 0.81, 0.82], True),
             "under 2 pooled sd": ([0.83, 0.85, 0.87], [0.82, 0.83, 0.84], False)}
    res = {}
    for name, (bv, av, want) in cases.items():
        fake_agnostic("e1v_b", "B", bv, arrs)
        fake_agnostic("e1v_a", "A", av, [synth(seed=10 + i) for i in range(3)])
        d, rep = B.e1_verdict("e1v_b", "e1v_a", out_dir=out, testing_ok=True, resamples=40)
        sb, sa = statistics.stdev(bv), statistics.stdev(av)
        res[name] = (d["qualifies"], want, abs(d["pooled_sd"] - math.sqrt((sb ** 2 + sa ** 2) / 2)) < 1e-12,
                     d["conditions"])
    check("qualifies when D > 2 pooled sd and D > SE (pooled sd = sqrt((sd_B^2 + sd_A^2) / 2)); not under 2 pooled sd",
          all(q == w and p for q, w, p, _c in res.values()), res)
    old = B.agnostic_bootstrap
    B.agnostic_bootstrap = lambda *a, **k: {"se": 1.0, "n_valid": 40}
    try:
        fake_agnostic("e1v_b", "B", [0.86, 0.87, 0.88], arrs)
        d, _r = B.e1_verdict("e1v_b", "e1v_a", out_dir=out, testing_ok=True, resamples=40)
    finally:
        B.agnostic_bootstrap = old
    check("  and not when D <= SE alone (SE 1.0)", d["conditions"]["above_2_pooled_sd"]
          and not d["conditions"]["above_se"] and not d["qualifies"], d["conditions"])
    B.agnostic_bootstrap = lambda *a, **k: {"se": 0.001, "n_valid": 40}
    try:
        fake_agnostic("e1v_b", "B", [0.84, 0.85, 0.86], arrs)
        fake_agnostic("e1v_a", "A", [0.825, 0.835, 0.845], [synth(seed=10 + i) for i in range(3)])
        d, _r = B.e1_verdict("e1v_b", "e1v_a", out_dir=out, testing_ok=True, resamples=40)
    finally:
        B.agnostic_bootstrap = old
    check("  and not when D is over SE (0.001) and over 1 pooled sd but not over 2 (D %.4f, pooled sd %.4f)"
          % (d["diff"], d["pooled_sd"]), d["conditions"] == {"above_2_pooled_sd": False, "above_se": True}
          and not d["qualifies"] and d["pooled_sd"] < d["diff"] < 2 * d["pooled_sd"], d["conditions"])
    from weed_optimizer_framework.tools.inc_autopilot import remote as R
    doc = json.loads((out / "e1_v1.json").read_text())
    check("capacity/e1_v1.json: dev only (the autopilot's scrub drops nothing), the rule and the bootstrap recorded",
          not R.non_dev_keys(doc) and doc["rule"] == B.E1_RULE
          and doc["bootstrap"]["seed_text"] == "inc2/e1/agnostic_se"
          and doc["exam"] == "dev" and doc["status"] == "decided", R.non_dev_keys(doc))
    fake_agnostic("e1v_b", "B", [0.86, 0.87, 0.88], arrs, skip=(2,))
    d, _r = B.e1_verdict("e1v_b", "e1v_a", out_dir=out, testing_ok=True, resamples=10)
    check("a missing agnostic file: pending", d["status"] == "pending" and d["missing"] == ["e1v_b/final__base__s2"], d)
    fake_agnostic("e1v_b", "B", [0.86, 0.87, 0.88], arrs, summary="6" * 64)
    check("arms built from different splits v3 summaries refuse",
          refused(B.e1_verdict, "e1v_b", "e1v_a", out_dir=out, testing_ok=True) is not None)
    fake_agnostic("e1v_b", "B", [0.86, 0.87, 0.88], arrs)
    check("the arms swapped (--exp E1-A) refuse", refused(B.e1_verdict, "e1v_a", "e1v_b", out_dir=out) is not None)
    fake_agnostic("e1v_b", "B", [0.86, 0.87, 0.88], arrs, production=False)
    check("a test-mode agnostic file in a production verdict refuses",
          refused(B.e1_verdict, "e1v_b", "e1v_a", out_dir=out, testing_ok=False) is not None)


def test_rescore_and_test_read():
    print("rescore-agnostic end to end (real passes), then the test read")
    make_arm("e1r_a", "A", seeds=(0, 1), wseed0=30)
    make_arm("e1r_b", "B", seeds=(0, 1), wseed0=40)
    cap = C.INC_DIR / "capacity"
    if cap.exists():
        shutil.rmtree(cap)
    e = refused(B.e1_test_read, ["e1r_b"])
    check("e1-test-read before the verdict: refused", e is not None and "verdict" in str(e), e)
    rec = B.rescore_agnostic("e1r_b", "e1r_a", resamples=20)
    doc = json.loads((C.INC_DIR / "e1r_b" / B.AGNOSTIC_RECORD).read_text())
    v = json.loads((cap / "e1_v1.json").read_text())
    check("rescore-agnostic scores every final run of both arms, writes agnostic_rescore.json (complete, names and "
          "sha256s, no path) and the verdict (decided)",
          doc["status"] == "complete" and sorted(doc["scores"]) == ["e1r_a", "e1r_b"]
          and all(len(x) == 2 and all(r["sha256"] and "/" not in r["score"] for r in x) for x in doc["scores"].values())
          and "/" not in json.dumps(doc["scores"]) and v["status"] == "decided" and rec["status"] == "complete", doc)
    second = B.rescore_agnostic("e1r_b", "e1r_a", resamples=20)
    check("a second run scores nothing (every file kept)",
          all(r["status"] == "kept" for x in second["scores"].values() for r in x))
    res = B.e1_test_read(["e1r_a", "e1r_b"])
    specs = [json.loads(pathlib.Path(p).read_text()) for r in res.values() for p in r["specs"]]
    for sp in specs:
        T.validate_spec(sp, pathlib.Path(sp["out_dir"]) / "spec.json")
    check("e1-test-read once decided: one kind-final spec per seed on test from the base weights (the v2 executor "
          "accepts each), and run_inc2_job.sh's argv for a person; nothing submitted",
          len(specs) == 4 and all(sp["kind"] == "final" and sp["exams"] == ["test"] for sp in specs)
          and all(r["argv"][0] == "sbatch" and r["argv"][-3].endswith("run_inc2_job.sh") for r in res.values())
          and all(r["verdict_sha256"] == W.sha(cap / "e1_v1.json") for r in res.values()), res)
    sc = C.INC_DIR / "e1r_b" / "runs" / "e1test__s0" / "scores"
    sc.mkdir(parents=True, exist_ok=True)
    (sc / "test.json").write_text("{}")
    e = refused(B.e1_test_read, ["e1r_b"])
    check("once a test score exists: refused (test is read once per arm)", e is not None and "once" in str(e), e)
    (sc / "test.json").unlink()
    v["diff"] = 0.5
    (cap / "e1_v1.json").write_text(json.dumps(v))
    e = refused(B.e1_test_read, ["e1r_b"])
    check("a verdict that changed since the read was prepared: refused", e is not None and "changed" in str(e), e)


def main():
    t0 = time.time()
    Wd = W.build_world()
    test_recipe()
    test_train_recipe()
    test_guard_verdicts(Wd)
    test_scorer()
    test_bootstrap()
    test_verdict()
    test_rescore_and_test_read()
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    shutil.rmtree(W.TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
