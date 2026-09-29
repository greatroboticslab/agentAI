#!/usr/bin/env python3
"""Protocol v3's species guard (inc2/gate3.py, decision L-3) and the scorer
sidecar that feeds it (inc2/scorer_sidecar.py).

What is pinned:
- the sidecar's arrays: flattening keeps images without predictions; one
  ap_per_class call on them is the score's per_class; a class-restricted AP
  equals the full call's; the bootstrap is deterministic under
  stable_int("inc2/v3/species_se"), 1,000 resamples, and a species without a
  GT box has no SE; compare_scores and check_recompute refuse other weights,
  other stamps, a per_class beyond the tolerance and arrays that do not
  reproduce the score;
- a real CPU sidecar (the pinned scorer called as a library, test mode) on a
  synthetic dev reproduces its own score's per_class exactly and the recorded
  score's within the tolerance, restores the scorer's validator, leaves
  scorer.py's sha256 as LOCK.json records it, and is refused (exit 2) against
  the score of other weights;
- gate3.decide re-derives the recorded decision through the pinned gate
  (verify), refuses a tampered score file or a recorded decision the gate
  does not reproduce, and refuses a sidecar of other weights, other stamps,
  another seed or resample count, a missing or non-finite SE, or a test-mode
  sidecar for a production decision;
- only the species tolerance changes: with SE_s = pinned threshold_s / 1.96
  every verdict of a battery equals the pinned one; P_data, P_recipe and the
  regression and flips guards are the recorded ones; tol_s = max(0.03, 1.96
  SE_s); blame follows the new verdict;
- on realloop_v1's recorded ledger (the local artifact), ledger-only: with
  small SEs every verdict is the pinned REJECT; with SE(PricklySida) = 0.025
  only V4 (species-only failure on PricklySida, P_data 1.00) becomes ACCEPT;
- truth3 replaces the truth arm's species tolerance the same way;
- the sidecar's SE is the sample sd (ddof 1) of the per-resample APs, each
  computed independently here from the drawn images' own arrays;
- decide_experiment writes gate3.json with the sha256 of every input, records
  a step without a sidecar as unavailable: a pinned ACCEPT commits as HOLD
  (never an accept on a rule that was not applied), a pinned REJECT stands,
  with the pinned failed guards recorded; the CLI decides and shows;
- truth3 binds the sidecar to the without arm's first run with and without
  verification.

Run:  python3 tests/test_inc2_gate3.py
"""
import contextlib
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO to its own temporary dirs first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import gate3 as G3  # noqa: E402

FAILURES = W.FAILURES
check = W.check
REAL_INC = W.ROOT / "results" / "framework" / "inc"
CFG = G.GateConfig(require_production=False, flips_mode="net")


def have_ultralytics():
    try:
        import ultralytics  # noqa: F401
        from ultralytics.utils.metrics import ap_per_class  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        print("  NOTE ultralytics is not importable (%s): the sidecar tests that need ap_per_class are skipped" % e)
        return False


# -------------------------------------------------------------- synthetic
def synth_images(n=60, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    images, keys = {}, ["k%03d" % i for i in range(n)]
    for i, k in enumerate(keys):
        ng = int(rng.integers(1, 4))
        tc = rng.integers(0, 12, ng).astype(float)
        npred = 0 if i == 3 else int(rng.integers(5, 40))
        pc = rng.integers(0, 12, npred).astype(float)
        conf = rng.random(npred)
        tp = rng.random((npred, 10)) < (conf[:, None] * 0.6)
        images[k] = {"tp": tp, "conf": conf, "pred_cls": pc, "target_cls": tc}
    return images, keys


def score(value, weights, species=None, bits=None, n=20, production=False):
    per = {s: (species or {}).get(s, value) for s in G.SPECIES}
    return {"exam": "dev", "scorer_sha256": ("" if production else "TEST-") + "s" * 16, "manifest_sha256": "m" * 16,
            "key_order_sha256": "k" * 16, "weights_sha256": weights, "n_images": n, "map50_95": value,
            "map50": value + 0.1, "agnostic_map50_95": value + 0.05, "agnostic_map50": value + 0.15,
            "per_class": per, "n_gt": {c: (40 if c != "OtherPlant" else 0) for c in C.CLASS_NAMES},
            "image_correct": bits or ("1" * 12 + "0" * (n - 12)), "production": production}


def sidecar_for(weights, se, stamps=None, production=False, seed_text="inc2/v3/species_se", resamples=1000):
    per = {s: {"se": se.get(s, 0.005) if isinstance(se, dict) else se, "n_valid": resamples, "n_gt": 40, "ap": 0.5}
           for s in G.SPECIES}
    st = stamps or {"exam": "dev", "scorer_sha256": "TEST-" + "s" * 16, "manifest_sha256": "m" * 16,
                    "key_order_sha256": "k" * 16, "n_images": 20}
    return {"format": G3.SIDECAR_FORMAT, "exam": "dev", "weights_sha256": weights,
            "score": {"stamps": st, "production": production},
            "species_se": {"seed_text": seed_text, "resamples": resamples, "per_species": per}}


def write_score(root, rid, s):
    p = root / "runs" / rid / "scores" / "dev.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, sort_keys=True))
    return {"run_id": rid, "path": str(p), "sha256": C.sha256_file(p)}


def make_step(root, chain, k, inc_v, cand_vs, null_vs, cand_species=None, null_species=None, inc_rid="base__s0",
              step=None, cfg=CFG):
    """A ledger gate entry as the pinned driver writes it, with its score files."""
    inc = score(inc_v, "w_%s_inc_%d" % (chain, k))
    cands = [score(v, "w_%s_%d_c%d" % (chain, k, i), cand_species) for i, v in enumerate(cand_vs)]
    nulls = [score(v, "w_%s_%d_n%d" % (chain, k, i), null_species) for i, v in enumerate(null_vs)]
    d = G.decide(inc, cands, nulls, cfg)
    ins = {"inc": write_score(root, inc_rid, inc),
           "cand": [write_score(root, "%s__s%02d__cand__s%d" % (chain, k, i), s) for i, s in enumerate(cands)],
           "null": [write_score(root, "%s__s%02d__null__s%d" % (chain, k, i), s) for i, s in enumerate(nulls)]}
    return {"id": "gate/%s/%d" % (chain, k), "type": "gate", "chain": chain, "k": k, "step": step or "D%d" % k,
            "tag": "s%02d_D%d" % (k, k), "clean": False, "decision": d.to_dict(), "inputs": ins,
            "incumbent_before": {"run_id": inc_rid}}


def species_values(base, drops):
    return {s: base - drops.get(s, 0.0) for s in G.SPECIES}


# ------------------------------------------------------------------ tests
def test_sidecar_units():
    print("sidecar arrays and bootstrap")
    import numpy as np
    from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC
    from ultralytics.utils.metrics import ap_per_class
    images, keys = synth_images()
    arr = SC.flatten(images, keys)
    check("flatten: one row per prediction and target, the image without predictions kept",
          len(arr["keys"]) == 60 and len(arr["tp"]) == sum(len(images[k]["conf"]) for k in keys)
          and arr["tp"].shape[1] == 10 and 3 not in set(arr["pred_img"].tolist())
          and 3 in set(arr["target_img"].tolist()))
    res = ap_per_class(arr["tp"], arr["conf"], arr["pred_cls"], arr["target_cls"])
    want = {C.CLASS_NAMES[int(c)]: float(res[5][i].mean()) for i, c in enumerate(res[6])}
    full = SC.full_per_class(arr)
    check("full_per_class is one ap_per_class call on the arrays", full == want)
    worst, restricted = SC.check_recompute(want, arr)
    check("check_recompute: exact, and a class's AP from its own detections equals the full call's (%.2g)" % restricted,
          worst == 0.0 and restricted <= 1e-9)
    try:
        SC.check_recompute(dict(want, Waterhemp=want["Waterhemp"] + 0.01), arr)
        e = None
    except SC.SidecarError as x:
        e = x
    check("check_recompute refuses a per_class the arrays do not give", e is not None, e)
    # equal confidences (half precision): ap_per_class ranks ties by input order, so only the validator's own
    # image order reproduces its score (pilot_v4, 2026-09-29: 0.0007-0.0012 off in exam key order)
    import numpy as np
    rng = np.random.default_rng(20260929)
    tied = {}
    for i in rng.permutation(80):                           # the validator's order is not the key order
        n = int(rng.integers(5, 40))
        conf = (np.round(rng.random(n) * 20) / 20).astype(np.float32)   # 21 distinct values: many ties
        tied["img_%03d" % i] = {"tp": rng.random((n, 10)) < (0.2 + 0.7 * conf[:, None]), "conf": conf,
                                "pred_cls": rng.integers(0, 12, n).astype(np.float32),
                                "target_cls": rng.integers(0, 12, int(rng.integers(1, 12))).astype(np.float32)}
    score_pc = SC.full_per_class(SC.flatten(tied, list(tied)))   # what DetMetrics reports
    key_order = SC.flatten(tied, sorted(tied))
    off = max(abs(SC.full_per_class(key_order)[c] - score_pc[c]) for c in score_pc)
    try:
        SC.check_recompute(score_pc, key_order)
        e_key = None
    except SC.SidecarError as x:
        e_key = x
    check("with tied confidences the exam key order is off the score (%.2g) and check_recompute refuses it; "
          "check_capture, in the validator's order, reproduces it exactly" % off,
          off > 1e-9 and e_key is not None and SC.check_capture(score_pc, tied)[0] == 0.0, (off, e_key))
    # ties also split a class-restricted call from the full call (0.0103 on a real dev exam, pilot_v4): both now run
    # on tie_break's one order, and agree exactly
    tb = SC.tie_break(key_order)
    kept = np.all(np.diff(key_order["conf"][np.argsort(-tb["conf"], kind="stable")]) <= 0)
    check("tie_break: every confidence distinct, no prediction crosses another's value, deterministic",
          kept and len(np.unique(tb["conf"])) == len(tb["conf"])
          and np.array_equal(tb["conf"], SC.tie_break(key_order)["conf"]))
    _w, restricted_tied = SC.check_capture(score_pc, tied)
    check("on tied arrays the class-restricted AP equals the full call's exactly (both tie-broken)",
          restricted_tied == 0.0, restricted_tied)
    shift = SC.tie_shift(score_pc, key_order)
    check("tie_shift records how far the tie-broken AP sits from the score (%.2g), within the tie noise" % shift,
          0.0 <= shift < 0.05, shift)
    t0 = time.time()
    a = SC.bootstrap_species_se(arr, resamples=200)
    b = SC.bootstrap_species_se(arr, resamples=200)
    c = SC.bootstrap_species_se(arr, resamples=200, seed_text="another")
    check("the bootstrap is deterministic under its seed text, and another seed gives other values (%.1fs)"
          % (time.time() - t0), a == b and a != c)
    idx = SC.resample_indices(60, 5)
    rng = np.random.default_rng(C.stable_int("inc2/v3/species_se"))
    check("resamples are default_rng(stable_int('inc2/v3/species_se')).integers over the image indices, "
          "1,000 by default", (idx == rng.integers(0, 60, size=(5, 60))).all() and SC.RESAMPLES == 1000
          and SC.SEED_TEXT == "inc2/v3/species_se")
    ok = all(v["se"] is not None and v["se"] > 0 and v["n_valid"] >= 150 for v in a.values())
    check("every species with boxes gets a positive SE", ok, {s: (v["se"], v["n_valid"]) for s, v in a.items()})
    only = {k: dict(v, target_cls=np.where(v["target_cls"] == 7, 0.0, v["target_cls"])) for k, v in images.items()}
    arr2 = SC.flatten(only, keys)
    se2 = SC.bootstrap_species_se(arr2, resamples=50)
    check("a species without a GT box has no SE (n_valid 0)", se2["PricklySida"]["se"] is None
          and se2["PricklySida"]["n_valid"] == 0)
    from ultralytics.utils.metrics import ap_per_class as apc
    n_res = 120
    draws = SC.resample_indices(len(keys), n_res)
    for name in ("Waterhemp", "PricklySida"):
        sid = C.CLASS_NAMES.index(name)
        vals = []
        for b in range(n_res):
            tp_l, cf_l, ngt = [], [], 0
            for i in draws[b]:
                im = images[keys[i]]
                sel = np.rint(im["pred_cls"]).astype(int) == sid
                tp_l.append(np.asarray(im["tp"], dtype=bool)[sel])
                cf_l.append(np.asarray(im["conf"])[sel])
                ngt += int((np.rint(im["target_cls"]).astype(int) == sid).sum())
            if ngt == 0:
                continue
            tp_b, cf_b = np.concatenate(tp_l, 0), np.concatenate(cf_l)
            if not len(cf_b):
                vals.append(0.0)
                continue
            r = apc(tp_b, cf_b, np.full(len(cf_b), float(sid)), np.full(ngt, float(sid)))
            vals.append(float(np.asarray(r[5])[0].mean()))
        want_se = float(np.std(vals, ddof=1))
        got = SC.bootstrap_species_se(arr, resamples=n_res)[name]
        check("SE(%s) is the sample sd (ddof 1) of %d independently recomputed resample APs (%.6f)"
              % (name, len(vals), want_se), abs(got["se"] - want_se) <= 1e-9 and got["n_valid"] == len(vals),
              (got["se"], want_se, got["n_valid"], len(vals)))
    rec = {"exam": "dev", "scorer_sha256": "a", "manifest_sha256": "b", "key_order_sha256": "c", "n_images": 3,
           "weights_sha256": "w", "n_gt": {"x": 1}, "production": True, "map50_95": 0.5,
           "per_class": {"Waterhemp": 0.5}}
    check("compare_scores accepts the same score", SC.compare_scores(rec, dict(rec)) == 0.0)
    for what, other in (("other weights", dict(rec, weights_sha256="v")), ("another scorer", dict(rec, scorer_sha256="z")),
                        ("per_class beyond the tolerance", dict(rec, per_class={"Waterhemp": 0.51})),
                        ("another production flag", dict(rec, production=False))):
        try:
            SC.compare_scores(rec, other)
            e = None
        except SC.SidecarError as x:
            e = x
        check("compare_scores refuses %s" % what, e is not None, e)


def named_checkpoint(path, seed=0):
    """W.cold_checkpoint with the INC class names, so the scorer accepts it
    without training (a trained run gets them from data.yaml)."""
    import torch
    W.cold_checkpoint(path, seed=seed)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    ck["model"].names = {i: n for i, n in enumerate(C.CLASS_NAMES)}
    torch.save(ck, str(path))
    return path


def test_sidecar_real():
    print("a real sidecar on the CPU (the pinned scorer as a library)")
    from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC
    W.build_world()
    weights = named_checkpoint(W.TMP / "named.pt")
    out = W.TMP / "sc"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(W.ROOT))
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.scorer", "--weights", str(weights),
                        "--exam", "dev", "--out", str(out / "dev.json"), "--imgsz", "64", "--batch", "8",
                        "--device", "cpu"], cwd=str(W.ROOT), env=env, capture_output=True, text=True)
    check("the pinned scorer scores the checkpoint (test mode)", r.returncode == 0, r.stderr[-500:])
    before = S._VALIDATOR
    rec = SC.run(weights, "dev", out / "dev.json", out / "dev.sidecar.json", imgsz=64, batch=8, device="cpu",
                 resamples=100)
    js, npz = SC.paths_for(out / "dev.sidecar.json")
    check("the sidecar writes its JSON and npz, reproducing its own score exactly and the recorded one",
          js.is_file() and npz.is_file() and rec["consistency"]["full_recompute_max_abs_diff"] <= 1e-9
          and rec["consistency"]["max_abs_diff_vs_recorded"] <= SC.MAX_SCORE_DIFF
          and rec["images"]["sha256"] == C.sha256_file(npz), rec["consistency"])
    arrays = SC.load_npz(npz)
    check("the npz holds the flat arrays in key order", list(arrays["keys"]) == sorted(r["key"] for r in C.read_manifest(
        C.manifest_path("dev"))) and arrays["tp"].shape[1] == 10)
    check("the scorer's validator is restored after the capture", S._VALIDATOR is before or S._VALIDATOR is not None
          and S._VALIDATOR.__name__ != "SidecarValidator")
    lock = json.loads(C.LOCK_PATH.read_text())
    check("scorer.py still hashes to the LOCK's scorer_sha256 (the file is never edited)",
          C.sha256_file(pathlib.Path(S.__file__).resolve()) == lock["scorer_sha256"])
    other = named_checkpoint(W.TMP / "other.pt", seed=5)
    r2 = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.scorer_sidecar", "--weights",
                         str(other), "--exam", "dev", "--score", str(out / "dev.json"), "--out",
                         str(out / "x.sidecar.json"), "--imgsz", "64", "--batch", "8", "--device", "cpu"],
                        cwd=str(W.ROOT), env=env, capture_output=True, text=True)
    check("a sidecar of other weights than the score's is refused (exit 2), writing nothing",
          r2.returncode == 2 and not (out / "x.sidecar.json").exists() and "REFUSED" in r2.stderr,
          (r2.returncode, r2.stderr[-300:]))


def test_decide_rules():
    print("gate3.decide: verification, sidecar checks, the species tolerance only")
    root = W.C.INC_DIR / "g3_exp"
    base_sp = species_values(0.60, {})
    e1 = make_step(root, "r0", 1, 0.60, [0.62, 0.63, 0.625], [0.60, 0.605, 0.598],
                   cand_species=species_values(0.62, {"PricklySida": 0.07}), null_species=base_sp)
    d = e1["decision"]
    check("fixture: the pinned gate REJECTs on the species guard alone (PricklySida)",
          d["verdict"] == "REJECT" and d["guards"]["species"]["failed"] == ["PricklySida"]
          and d["guards"]["regression"]["passed"] and d["guards"]["flips"]["passed"], d["reason"])
    inc_w = d["weights"]["inc"]
    v_small = G3.decide(e1, sidecar_for(inc_w, 0.005), verify=True)
    v_big = G3.decide(e1, sidecar_for(inc_w, {"PricklySida": 0.03}), verify=True)
    check("the drop is -0.05: with SE 0.005 the tolerance is the 0.03 floor and the verdict stays REJECT; with "
          "SE(PricklySida) 0.03 it is 0.0588 and the step is ACCEPTed (P_data %.2f)" % d["p_data"],
          v_small["verdict"] == "REJECT" and not v_small["changed"]
          and abs(v_small["species_tolerance"]["PricklySida"] - 0.03) < 1e-12
          and v_big["verdict"] == "ACCEPT" and v_big["changed"] and v_big["pinned_verdict"] == "REJECT"
          and abs(v_big["species_tolerance"]["PricklySida"] - 1.96 * 0.03) < 1e-12, (v_small["reason"], v_big["reason"]))
    same = all(v_big[k] == d[k] for k in ("p_data", "p_recipe", "inc", "cand_mean", "null_mean", "null_sd"))
    check("P_data, P_recipe and the regression and flips guards are the recorded ones; blame follows the verdict",
          same and v_big["guards"]["regression"] == d["guards"]["regression"]
          and v_big["guards"]["flips"] == d["guards"]["flips"] and v_big["attribution"]["blame"] is None
          and v_small["attribution"]["blame"] == d["attribution"]["blame"]
          and v_big["inputs"]["verified"] is True and v_big["failed_guards"] == [])

    p = pathlib.Path(e1["inputs"]["cand"][0]["path"])
    saved = p.read_bytes()
    p.write_bytes(saved + b" ")
    try:
        try:
            G3.decide(e1, sidecar_for(inc_w, 0.02))
            e = None
        except G3.Gate3Error as x:
            e = x
    finally:
        p.write_bytes(saved)
    check("verify: a score file that no longer hashes as recorded is refused", e is not None and "no longer" in str(e), e)
    forged = json.loads(json.dumps(e1))
    forged["decision"]["p_data"] = 0.5
    try:
        G3.decide(forged, sidecar_for(inc_w, 0.02))
        e = None
    except G3.Gate3Error as x:
        e = x
    check("verify: a recorded decision the pinned gate does not reproduce is refused",
          e is not None and "does not reproduce" in str(e), e)
    bad = {"other weights": sidecar_for("elsewhere", 0.02),
           "other stamps": sidecar_for(inc_w, 0.02, stamps={"exam": "dev", "scorer_sha256": "other",
                                                          "manifest_sha256": "m" * 16, "key_order_sha256": "k" * 16,
                                                          "n_images": 20}),
           "another seed": sidecar_for(inc_w, 0.02, seed_text="x"),
           "another resample count": sidecar_for(inc_w, 0.02, resamples=500),
           "a missing SE": dict(sidecar_for(inc_w, 0.02), species_se={"seed_text": "inc2/v3/species_se",
                                                                      "resamples": 1000, "per_species": {}}),
           "a non-finite SE": sidecar_for(inc_w, {"Eclipta": float("nan")}),
           "another exam": dict(sidecar_for(inc_w, 0.02), exam="test"),
           "not a sidecar": {"format": "x"}}
    for what, sc in bad.items():
        try:
            G3.decide(e1, sc, verify=False)
            e = None
        except G3.Gate3Error as x:
            e = x
        check("the sidecar is refused: %s" % what, e is not None, e)
    prod = json.loads(json.dumps(e1))
    prod["decision"]["config"]["require_production"] = True
    try:
        G3.species_tolerances(sidecar_for(inc_w, 0.02), weights_sha256=inc_w, require_production=True)
        e = None
    except G3.Gate3Error as x:
        e = x
    check("a test-mode sidecar cannot serve a production decision", e is not None and "test-mode" in str(e), e)

    battery = [(0.60, [0.62, 0.63, 0.625], [0.60, 0.605, 0.598], {"PricklySida": 0.07}, {}),
               (0.60, [0.62, 0.63, 0.625], [0.60, 0.605, 0.598], {"Eclipta": 0.02}, {}),
               (0.60, [0.55, 0.56, 0.555], [0.60, 0.605, 0.598], {}, {}),
               (0.60, [0.60, 0.603, 0.601], [0.60, 0.605, 0.598], {"Sicklepod": 0.04, "Goosegrass": 0.05}, {}),
               (0.60, [0.61, 0.612, 0.609], [0.60, 0.605, 0.598], {}, {"Ragweed": -0.03})]
    agree = []
    for i, (inc_v, cv, nv, cdrop, ndrop) in enumerate(battery):
        e = make_step(root, "bat", i + 1, inc_v, cv, nv, cand_species=species_values(cv[0], cdrop),
                      null_species=species_values(nv[0], ndrop), inc_rid="bat_inc_%d" % i)
        per = e["decision"]["guards"]["species"]["per_species"]
        sc = sidecar_for(e["decision"]["weights"]["inc"], {s: per[s]["threshold"] / 1.96 for s in G.SPECIES})
        v = G3.decide(e, sc)
        agree.append((v["verdict"] == e["decision"]["verdict"], v["guards"]["species"]["failed"]
                      == e["decision"]["guards"]["species"]["failed"]))
    check("with SE_s = pinned threshold_s / 1.96 every verdict and species failure of a battery is the pinned one",
          all(a and b for a, b in agree), agree)


def test_truth3():
    print("truth3")
    root = W.C.INC_DIR / "g3_truth"
    base_sp = species_values(0.60, {})
    w = [W_ for W_ in (score(v, "tw%d" % i, species_values(v, {"PricklySida": 0.06})) for i, v in
                       enumerate([0.62, 0.63, 0.625]))]
    wo = [score(v, "two%d" % i, base_sp) for i, v in enumerate([0.60, 0.605, 0.598])]
    det = G.truth_detail(w, wo, CFG)
    entry = {"id": "truth/1", "type": "truth", "k": 1, "step": "D1", "tag": "s01_D1", "clean": False, "detail": det,
             "inputs": {"with": [write_score(root, "truth__s01_D1__union__s%d" % i, s) for i, s in enumerate(w)],
                        "without": [write_score(root, "base__s%d" % i, s) for i, s in enumerate(wo)]}}
    check("fixture: the pinned truth rule says hurts on the species guard alone", det["verdict"] == "hurts"
          and det["species"]["failed"] == ["PricklySida"] and det["p"] == 1.0)
    t_small = G3.truth3(entry, sidecar_for("two0", 0.005), CFG)
    t_big = G3.truth3(entry, sidecar_for("two0", {"PricklySida": 0.035}), CFG)
    check("truth3: SE 0.005 keeps hurts; SE(PricklySida) 0.035 (tol 0.0686) gives helps at P 1.0",
          t_small["verdict"] == "hurts" and t_big["verdict"] == "helps" and t_big["changed"]
          and t_big["tolerance_from"] == "base__s0")
    for verify in (True, False):
        try:
            G3.truth3(entry, sidecar_for("elsewhere", 0.035), CFG, verify=verify)
            e = None
        except G3.Gate3Error as x:
            e = x
        check("truth3 (verify %s) refuses a sidecar that is not the without arm's first run" % verify,
              e is not None and "weights" in str(e), e)


def test_realloop_v1():
    print("realloop_v1's recorded ledger (ledger only)")
    led_path = REAL_INC / "realloop_v1" / "ledger.jsonl"
    if not led_path.is_file():
        print("  NOTE %s is not here; skipped" % led_path)
        return
    entries = [json.loads(ln) for ln in led_path.read_text().splitlines() if ln.strip()]
    gates = [e for e in entries if e.get("type") == "gate"]

    def run_with(se_p):
        out = []
        for e in gates:
            sc = sidecar_for(e["decision"]["weights"]["inc"], {"PricklySida": se_p}, stamps=e["decision"]["stamps"],
                             production=True)
            out.append(G3.decide(e, sc, verify=False))
        return out

    small, big = run_with(0.005), run_with(0.025)
    check("SE 0.005 everywhere: the six v3 verdicts are the six pinned REJECTs",
          [v["verdict"] for v in small] == [e["decision"]["verdict"] for e in gates] == ["REJECT"] * 6)
    changed = [(v["step"], v["verdict"]) for v in big if v["changed"]]
    check("SE(PricklySida) 0.025 (tol 0.049): only V4 (species-only on PricklySida, P_data 1.00) is ACCEPTed; "
          "V1-V3 and UNVERIFIED still fail the regression guard, OTHER_HEAVY the flips guard",
          changed == [("V4", "ACCEPT")] and [v["failed_guards"] for v in big][4] == ["flips"]
          and all("regression" in v["failed_guards"] for v in big[:4]), [(v["step"], v["failed_guards"]) for v in big])


def test_experiment():
    print("decide_experiment and the CLI")
    exp = "g3_seg"
    root = W.C.INC_DIR / exp
    base_sp = species_values(0.60, {})
    e1 = make_step(root, "r0", 1, 0.60, [0.62, 0.63, 0.625], [0.60, 0.605, 0.598],
                   cand_species=species_values(0.62, {"PricklySida": 0.07}), null_species=base_sp, inc_rid="base__s0")
    e2 = make_step(root, "r0", 2, 0.60, [0.61, 0.612, 0.609], [0.60, 0.605, 0.598], inc_rid="r0_inc_2",
                   cand_species=species_values(0.61, {}), null_species=base_sp)
    e3 = make_step(root, "r0", 3, 0.60, [0.62, 0.63, 0.625], [0.60, 0.605, 0.598], inc_rid="r0_inc_3",
                   cand_species=species_values(0.62, {"PricklySida": 0.07}), null_species=base_sp)
    pin = {"id": "gate_pin/0", "type": "gate_pin", "config": dataclasses.asdict(CFG), "block": {"flips_mode": "net"}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "exp.json").write_text(json.dumps({"exp": exp, "testing": True}))
    (root / "ledger.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in (pin, e1, e2, e3)))
    sc_path = root / "runs" / "base__s0" / "scores" / "dev.sidecar.json"
    sc_path.write_text(json.dumps(sidecar_for(e1["decision"]["weights"]["inc"], {"PricklySida": 0.03})))
    doc = G3.decide_experiment(exp)
    s1, s2, s3 = doc["steps"]
    check("step 1 (sidecar present): v3 ACCEPT over the pinned REJECT, its sidecar hashed",
          s1["v3_applied"] and s1["commit_verdict"] == "ACCEPT" and s1["pinned_verdict"] == "REJECT"
          and s1["inputs"]["sidecar"]["sha256"] == C.sha256_file(sc_path), s1.get("reason"))
    check("step 2 (no sidecar, pinned ACCEPT): unavailable, and it commits as HOLD, never as an ACCEPT on a rule "
          "that was not applied", e2["decision"]["verdict"] == "ACCEPT" and not s2["v3_applied"]
          and s2["status"] == "unavailable" and s2["commit_verdict"] == "HOLD" and s2["pinned_verdict"] == "ACCEPT"
          and "no sidecar" in s2["reason"] and s2["failed_guards"] == [] and s2["p_data_le_p_reject"] is False, s2)
    check("step 3 (no sidecar, pinned species-only REJECT): unavailable, the REJECT stands with its pinned failed "
          "guards", e3["decision"]["verdict"] == "REJECT" and not s3["v3_applied"] and s3["commit_verdict"] == "REJECT"
          and s3["failed_guards"] == ["species"], s3)
    out = root / "gate3.json"
    on_disk = json.loads(out.read_text())
    check("gate3.json is written with the ledger's and exp.json's sha256 and the summary",
          on_disk["inputs"]["ledger"]["sha256"] == C.sha256_file(root / "ledger.jsonl")
          and on_disk["inputs"]["exp_json"]["sha256"] == C.sha256_file(root / "exp.json")
          and on_disk["summary"]["changed"] == ["gate3/r0/1"]
          and on_disk["summary"]["unavailable"] == ["gate3/r0/2", "gate3/r0/3"]
          and on_disk["summary"]["unavailable_held"] == ["gate3/r0/2"]
          and on_disk["config"]["gate"]["flips_mode"] == "net")
    env = dict(os.environ, PYTHONPATH=str(W.ROOT))
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.gate3", "decide", "--exp", exp,
                        "--out", str(root / "g3_cli.json")], cwd=str(W.ROOT), env=env, capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.gate3", "show", "--exp", "nope"],
                        cwd=str(W.ROOT), env=env, capture_output=True, text=True)
    check("the CLI decides (exit 0, file written) and refuses an experiment without a ledger (exit 1)",
          r.returncode == 0 and (root / "g3_cli.json").is_file() and "gate3/r0/1" in r.stdout and r2.returncode == 1,
          (r.returncode, r.stderr[-300:], r2.returncode))


def main():
    t0 = time.time()
    try:
        if have_ultralytics():
            test_sidecar_units()
            test_sidecar_real()
        test_decide_rules()
        test_truth3()
        test_realloop_v1()
        test_experiment()
    finally:
        import shutil
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
