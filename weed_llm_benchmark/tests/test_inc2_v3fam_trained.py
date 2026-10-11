#!/usr/bin/env python3
"""V3F's trained arms (inc2/v3fam_trained.py) and the edits V3F.6.1 makes to inc2.recipes, inc2.train and
inc2.baseline (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F, the v3 experiment family D0-D9", V3F.5 D1, D4,
D5b, D0-S's BCLEAN; V3F.6.1).

Two groups. UNITS need no scoring pass (V3FAM_ONLY=units runs only them):
- inc2.recipes' v3fam set: the build arms, their experiments and detectors; cold_budget at the amendment's
  counts (F and R15 78 epochs, warmup 1.325052, close_mosaic 8; N 41, 0.707182, 4; A'1 389, 6.597938, 39; B'1 29,
  0.502355, 3) and the cold table for the 12-class arms; another arm refused;
- build_kwargs (what inc2.baseline build --v3fam takes): the arm's manifest from the complete family summary by
  sha256, seeds 0-2, role baseline, finals dev and ImageWeeds, exp.json's v3fam record; refused: another --exp,
  other seeds, another role, finals with test, a manifest that no longer hashes, an unknown arm; D5b's three
  builds take dev as their only final (Revision 30), and their sources never open a run's protocol dev score;
  baseline's --v3fam refuses --manifest, --union, --arm and --e2;
  a family manifest (under splits/v3fam, or by its sha256 elsewhere) is claimed, so it trains through --v3fam
  only;
- inc2.train.v3fam_problems: a budget arm's record against the summary (another arm, a changed summary, another
  manifest), and experiment_budget taking it for an exp.json with a v3fam record, e1_problems otherwise;
- the D1 classifier on E3's stored features of the A' rows: fitted and pinned once, equal to a fresh fit of
  inc2.v3fam_embed.fit_lr_protocol on the same rows, a crop outside A' never used, a refit refused, a tampered
  file refused at load;
- inc2.v3fam score's dispatch (every score arm to its module).
WORLD runs real Ultralytics CPU passes in the synthetic world of tests/test_inc2_v3fam_systems.py (E3's records,
E3RS's reproductions, a zoo root with ooddev_v1, ood22 and ood23 materialised):
- inc2.baseline build --v3fam D4F, D1A1, D1B1 and D1A12 (definition only: the driver's init is not run here): the
  family's manifest, cold_budget recomputed from its count under the v3fam record (inc2.train's own check), the
  cold table for D1A12;
- score_d4 (D4F): three dev files through the family two-stage pass with E3's pinned classifier, each stage-1 box
  set equal to its run's recorded agnostic dev AP, ImageWeeds and ooddev_v1 reported, the platform's record dev
  only, a rerun keeping every file, a definition on another manifest refused;
- score_d1: refused before the EXIF pre-flight; then the D1 classifier pinned, ood22's and ood23's identity
  reproductions, A'1 and B'1 in stage1 mode with P7's arrays, TS-A' and TS-B' named by the D1 classifier, A'12
  identity, on ood22, ood23 and dev; ooddev_v1 reported; the record dev only.

Run:  python3 tests/test_inc2_v3fam_trained.py      (V3FAM_ONLY=units: the units only)
"""
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_e3 as E3T  # noqa: E402
import test_inc2_native as N  # noqa: E402
import test_inc2_v3fam as FT  # noqa: E402
import test_inc2_v3fam_systems as SYT  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_data as VD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_embed as EMB  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_trained as VT  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = E3T.patched
tamper = E3T.tamper
KW = FT.KW
EMBD = E3T.EMB
WORLD = SYT.WORLD


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, B.BaselineError, RC.RecipeError) as e:
        return e
    return None


def write_summary(arms):
    """splits/v3fam/summary.json, complete, with {arm: rows} written as the family's manifests (row bytes as
    given). Returns {arm: (path, sha256, n)}."""
    out, recs = {}, {}
    d = VD.out_dir()
    d.mkdir(parents=True, exist_ok=True)
    for arm, rows in arms.items():
        p = d / VD.ARMS[arm]["file"] if arm in VD.ARMS else d / ("%s.jsonl" % arm.lower())
        sha = C.write_manifest(p, rows)
        recs[arm] = {"manifest": str(p), "sha256": sha, "images": len(rows)}
        out[arm] = (p, sha, len(rows))
    sp = d / VD.SUMMARY
    if sp.exists():
        sp.unlink()
    sp.write_text(json.dumps({"format": VD.FORMAT, "status": "complete", "arms": recs}))
    return out


# ================================================================== UNITS
def u_recipes():
    print("inc2.recipes' v3fam set (V3F.3.8)")
    want = {("D4F", 15432): (78, 1.325052, 8), ("D4R15", 15432): (78, 1.325052, 8), ("D4N", 28953): (41, 0.707182, 4),
            ("D1A1", 3088): (389, 6.597938, 39), ("D1B1", 40762): (29, 0.502355, 3)}
    got = {k: (RC.v3fam_recipe(*k)["epochs"], RC.v3fam_recipe(*k)["warmup_epochs"],
               RC.v3fam_recipe(*k)["close_mosaic"]) for k in want}
    check("cold_budget at the amendment's counts (F, R15, N, A'1, B'1)", got == want, got)
    check("the 12-class arms train the cold table (D1A12, D5B*, D8M m640, D8S s640); BCLEAN cold_budget",
          all(RC.v3fam_recipe(a, 6811) == RC.cold("m640") for a in ("D1A12", "D5B0", "D5B25", "D5B100", "D8M"))
          and RC.v3fam_recipe("D8S", 3048) == RC.cold("s640") and RC.v3fam_recipe_name("D0BCLEAN") == RC.BUDGET_NAME
          and RC.v3fam_recipe("D0BCLEAN", 44470) == RC.cold_budget("m640", 44470))
    check("the twelve build arms, their experiments (v3_*) and detectors; another arm refused",
          sorted(RC.V3FAM_ARMS) == sorted(F.BUILD_ARMS) and all(v.startswith("v3_") for v in RC.V3FAM_EXPS.values())
          and refused(RC.v3fam_recipe, "D4X", 10) is not None and refused(RC.v3fam_recipe_name, "D8") is not None)


def u_build_kwargs():
    print("build_kwargs: what inc2.baseline build --v3fam takes")
    rows = WORLD["rows"]["base_v2"]
    m = write_summary({"D4F": rows[:-1], "D1A12": [r for r in rows if r["key"].startswith("tc")]})
    kw = VT.build_kwargs("D4F")
    rec = kw["extra"]["v3fam"]
    check("D4F: v3_d4_f_m640 on the summary's manifest by sha256, m640, seeds 0-2, role baseline, finals dev and "
          "ImageWeeds, the v3fam record (the summary's sha256, cold_budget)",
          kw["exp"] == "v3_d4_f_m640" and kw["manifest"] == str(m["D4F"][0]) and kw["arm"] == "m640"
          and kw["seeds"] == [0, 1, 2] and kw["role"] == "baseline" and kw["final_exams"] == ["dev", "imageweeds"]
          and rec["manifest_sha256"] == m["D4F"][1] and rec["images"] == m["D4F"][2] and rec["recipe"] == "cold_budget"
          and rec["summary_sha256"] == W.sha(VD.out_dir() / VD.SUMMARY), rec)
    errs = {"another --exp": refused(VT.build_kwargs, "D4F", exp="v3_d4_n_m640"),
            "seeds 0": refused(VT.build_kwargs, "D4F", seeds="0"),
            "role capacity": refused(VT.build_kwargs, "D4F", role="capacity"),
            "finals with test": refused(VT.build_kwargs, "D4F", final_exams=["dev", "imageweeds", "test"]),
            "an arm the summary lacks": refused(VT.build_kwargs, "D4N"),
            "an unknown arm": refused(VT.build_kwargs, "D4X")}
    def gone(arm):
        raise VD.DataRefused("%s no longer hashes as the family summary records" % arm)
    with patched(VD, "arm_manifest", gone):
        errs["a manifest that no longer hashes"] = refused(VT.build_kwargs, "D4F")
    check("refused: %s" % ", ".join(errs), all(v is not None for v in errs.values()), errs)
    m5 = write_summary({"D4F": rows[:-1], "D5B25": rows[:-2]})
    k5 = VT.build_kwargs("D5B25")
    e5 = refused(VT.build_kwargs, "D5B25", final_exams=["dev", "imageweeds"])
    check("D5b's builds take dev as their only final (Revision 30: the pinned driver's minimum; the family never reads "
          "those dev scores): v3_d5b_25_m640 on the summary's manifest, the cold table; ImageWeeds among its finals "
          "refused; D4F's finals stay dev and ImageWeeds",
          k5["final_exams"] == ["dev"] and k5["exp"] == "v3_d5b_25_m640" and k5["manifest"] == str(m5["D5B25"][0])
          and k5["extra"]["v3fam"]["recipe"] == "cold" and e5 is not None and VT.finals_of("D4F") == ("dev", "imageweeds")
          and all(VT.finals_of(a) == ("dev",) for a in VT.D5B_BUILDS), (k5["final_exams"], e5))
    m = write_summary({"D4F": rows[:-1], "D1A12": [r for r in rows if r["key"].startswith("tc")]})
    bad = {}
    for flag, kw2 in (("--manifest", {"manifest": "/x.jsonl"}), ("--union", {"union": ["/x.jsonl"]}),
                      ("--arm", {"arm": "m640"}), ("--e2", {"e2": "W"})):
        bad[flag] = refused(B.v3fam_args, "D4F", "v3_d4_f_m640", **kw2)
    bad["a v3fam record in extra"] = refused(B.v3fam_args, "D4F", "v3_d4_f_m640", extra={"v3fam": {}})
    check("inc2.baseline build --v3fam takes no %s" % ", ".join(bad), all(v is not None for v in bad.values()), bad)
    copy = W.TMP / "elsewhere" / "copy.jsonl"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(m["D4F"][0]), str(copy))
    check("a family manifest is claimed (under splits/v3fam, or its bytes anywhere by the summary's sha256), so it "
          "trains through --v3fam only; base_v2 is not",
          B.v3fam_claim(m["D4F"][0], m["D4F"][1]) and B.v3fam_claim(copy, m["D4F"][1])
          and B.v3fam_claim(C2_base_v2(), W.sha(C2_base_v2())) is None)


def u_d5b_sources():
    print("D5b's sources never open a run's protocol dev score (Revision 30)")
    exp = "v3_src_check"
    root_ = C.INC_DIR / exp
    shutil.rmtree(str(root_), ignore_errors=True)
    w = root_ / "runs" / "base__s0" / "weights" / "final.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"weights")
    sha = W.sha(w)
    (root_ / "exp.json").write_text(json.dumps({"exp": exp, "seeds": [0, 1, 2]}))
    for rid in ("base__s0", "final__base__s0"):
        (root_ / "runs" / rid).mkdir(parents=True, exist_ok=True)
        (root_ / "runs" / rid / "run.json").write_text(json.dumps({"status": "done", "weights_sha256": sha}))
    dev = root_ / "runs" / "final__base__s0" / "scores" / "dev.json"
    dev.parent.mkdir(parents=True)
    dev.write_text("{not json")
    import builtins
    opened = []
    real_open = builtins.open

    def spy(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)
    with patched(builtins, "open", spy):
        try:
            src = SYS.run_source(exp, 0, testing_ok=True, read_dev=False)
        except F.V3famRefused as ex:             # a source that reads the dev score refuses this unreadable one
            src = {"refused": str(ex)}
    e = refused(SYS.run_source, exp, 0, testing_ok=True)
    check("read_dev False: the base run's weights and the final run's record checked, scores/dev.json never opened, "
          "no dev number returned; the default reads it (and refuses this unreadable one)",
          src.get("weights_sha256") == sha and "recorded" not in src and "recorded_doc" not in src
          and not any(x.endswith("scores/dev.json") for x in opened) and e is not None
          and "recorded" not in SYS.source_record(src), (sorted(src), e))
    shutil.rmtree(str(root_))
    seen = {}

    def rs(exp_, s, testing_ok=False, read_dev=True):
        seen[(exp_, s)] = read_dev
        return {}
    with patched(VT, "check_definition", lambda build, testing_ok=False: {}), patched(SYS, "run_source", rs):
        VT.sources("D5B25")
        VT.sources("D4F")
    check("  D5b's build sources take read_dev off, every other build's on",
          all(seen[("v3_d5b_25_m640", s)] is False and seen[("v3_d4_f_m640", s)] is True for s in F.SEEDS), seen)


def C2_base_v2():
    from weed_optimizer_framework.tools.inc2 import common as C2
    return C2.v2_manifest_path("base_v2")


def u_train_problems():
    print("inc2.train.v3fam_problems (V3F.6.1)")
    m = write_summary({"D4F": WORLD["rows"]["base_v2"][:-1]})
    rec = VT.build_record("D4F", *m["D4F"])
    data = {"type": "baseline", "base": {"manifest_sha256": m["D4F"][1], "n_images": m["D4F"][2]}, "v3fam": rec}
    check("a budget arm's record against the complete summary: no problem", T.v3fam_problems(data) == [],
          T.v3fam_problems(data))
    p1 = T.v3fam_problems(dict(data, v3fam=dict(rec, arm="D1A12")))
    p2 = T.v3fam_problems(dict(data, base={"manifest_sha256": "0" * 64}))
    p3 = T.v3fam_problems(dict(data, v3fam=dict(rec, summary_sha256="1" * 64)))
    check("problems: a 12-class arm naming cold_budget, another base manifest, a summary changed since the build",
          p1 and p2 and p3, (p1, p2, p3))
    exp = "v3_d4_f_m640"
    d = C.INC_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    arm = RC.resolve_arm("m640", require_weights=False)
    full = dict(data, exp=exp, recipe_name=RC.BUDGET_NAME, budget=RC.budget_record("m640", m["D4F"][2]))
    (d / "exp.json").write_text(json.dumps(full))
    name, n, probs = T.experiment_budget(exp, arm)
    (d / "exp.json").write_text(json.dumps({k: v for k, v in full.items() if k != "v3fam"}))
    _n2, _x, probs2 = T.experiment_budget(exp, arm)
    check("experiment_budget: cold_budget under a v3fam record checked by v3fam_problems (none here); without it, "
          "E1's check refuses it", name == RC.BUDGET_NAME and n == m["D4F"][2] and probs == []
          and any("E1 record" in x for x in probs2), (probs, probs2))
    shutil.rmtree(str(d))


def u_d1_classifier():
    print("the D1 classifier on E3's stored features of the A' rows")
    rng = np.random.default_rng(5)
    rows, a_keys = [], []
    for i in range(13 * 14):
        k = "tc_%03d" % (i // 2) if i % 3 else "tsw22__f_%03d" % i
        rows.append({"crop_id": i, "key": k, "label": i % 13, "session": "s%d" % (i % 7)})
        if k.startswith("tc"):
            a_keys.append(k)
    feats = rng.normal(size=(len(rows), 16)).astype(np.float32)
    feats[np.arange(len(rows)), [r["label"] for r in rows]] += 3.0
    with patched(EMB, "e3_train_features", lambda: (feats, rows, {"train_emb": {"sha256": "e" * 64}})), \
            patched(VT, "a_prime_keys", lambda: sorted(set(a_keys))):
        shutil.rmtree(str(VT.d1_dir()), ignore_errors=True)
        if VT.d1_pin_path().exists():
            VT.d1_pin_path().unlink()
        cfg, rec, shas = VT.fit_d1_classifier(testing=True)
        sel = [i for i, r in enumerate(rows) if r["key"] in set(a_keys)]
        arrays, _r = EMB.fit_lr_protocol(V._norm(feats[sel]), np.array([rows[i]["label"] for i in sel]),
                                         np.array([rows[i]["session"] for i in sel]))
        check("fitted on the A' rows' stored features only (%d of %d crops), equal to a fresh fit of E3's protocol "
              "on the same rows; pinned at capacity/v3fam_d1_classifier_pin.json" % (len(sel), len(rows)),
              rec["a_prime"]["crops"] == len(sel) and np.array_equal(cfg["W"], arrays["W"])
              and np.array_equal(cfg["b"], arrays["b"]) and VT.d1_pin_path().exists()
              and shas["pin_sha256"] == W.sha(VT.d1_pin_path()), rec["a_prime"])
        e1 = refused(VT.fit_d1_classifier, testing=True)
        check("  fitted once: a refit refuses", e1 is not None, e1)
        cfg2, _r2, _s2 = VT.d1_classifier()
        check("  d1_classifier loads the pinned one", np.array_equal(cfg2["W"], cfg["W"]))
        npz = VT.d1_dir() / "classifier.npz"
        raw = npz.read_bytes()
        npz.write_bytes(raw + b"x")
        e2 = refused(VT.load_d1_classifier)
        npz.write_bytes(raw)
        check("  a classifier file that no longer hashes as its pin refuses", e2 is not None, e2)
    shutil.rmtree(str(VT.d1_dir()), ignore_errors=True)
    VT.d1_pin_path().unlink()


def u_dispatch():
    print("inc2.v3fam score's dispatch")
    mods = {a: F.score_module(a)[0].__name__.rsplit(".", 1)[1] for a in F.SCORE_ARMS}
    check("every score arm goes to its module: E3RS, D3, D6, D8 v3fam_systems; D2's nine and D9 v3fam_embed; D1, "
          "D4F, D4N, D4R15, D5B, BCLEAN v3fam_trained",
          all(mods[a] == "v3fam_systems" for a in ("E3RS", "D3", "D6", "D8"))
          and all(mods[a] == "v3fam_embed" for a in F.SCORE_ARMS if a.startswith("D2") or a == "D9")
          and all(mods[a] == "v3fam_trained" for a in VT.TRAINED_ARMS), mods)
    check("  an arm outside the family's list refuses", refused(F.score_module, "D7") is not None)
    e = refused(VT.score_arm, "D5B")
    check("  D5B's scores refuse while its builds are not built (its definitions missing)", e is not None, e)


def u_stage1_check():
    print("a one-class detector's stage-1 box set against its run's recorded agnostic dev AP")
    src = {"recorded": {"agnostic_map50_95": 0.61, "file": "x/final__base__s0/scores/dev.json", "sha256": "a" * 64}}
    ok = VT.stage1_vs_recorded({"stage1_agnostic": (0.61 + 0.5 * TS.STAGE1_TOL, 0.8)}, src, "D4F seed 0 dev")
    e1 = refused(VT.stage1_vs_recorded, {"stage1_agnostic": (0.61 + 2 * TS.STAGE1_TOL, 0.8)}, src, "D4F seed 0 dev")
    e2 = refused(VT.stage1_vs_recorded, {"stage1_agnostic": (0.61, 0.8)}, {"recorded": {}}, "D4F seed 0 dev")
    check("within twostage.STAGE1_TOL: compared and recorded; beyond it, or with no recorded score: refused",
          ok["compared"] is True and ok["max_abs_diff"] <= TS.STAGE1_TOL and e1 is not None and e2 is not None,
          (ok, e1, e2))


def u_d1_preflight():
    print("D1 reads ood23's tags 3, 6 and 8 only after the EXIF frame pre-flight")
    with patched(VT, "sources", lambda build, testing=None: {s: {} for s in F.SEEDS}), \
            patched(F, "exif_preflight_passed", lambda: False):
        e = refused(VT.score_d1, embedder=EMBD, **KW)
    check("score_d1 refuses before the pre-flight passed (its sources checked first)",
          e is not None and "pre-flight" in str(e), e)


UNITS = ("u_recipes", "u_dispatch", "u_stage1_check", "u_d1_preflight", "u_d5b_sources")
WORLD_UNITS = ("u_build_kwargs", "u_train_problems", "u_d1_classifier")


# ================================================================== WORLD
def make_runs(exp, wseed0, exams=("dev", "imageweeds")):
    """The runs a family build's jobs would leave: three done base runs (real checkpoints) and final runs scored on
    its finals by the locked scorer (CPU, test mode)."""
    for s in F.SEEDS:
        rd = N.make_final(exp, s, wseed0 + s, protocol=False)
        w = C.INC_DIR / exp / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
        (C.INC_DIR / exp / "runs" / ("base__s%d" % s) / "run.json").write_text(json.dumps(
            {"status": "done", "testing": True, "weights_sha256": W.sha(w)}))
        for exam in exams:
            S.score(rd / "weights" / "final.pt", exam, rd / "scores" / ("%s.json" % exam), lock_check=True,
                    imgsz=640, batch=32, device="cpu")


def w_builds():
    print("inc2.baseline build --v3fam (definitions only)")
    (C.REPO / "yolo11m.pt").write_bytes(os.urandom(2048))
    rows = WORLD["rows"]["base_v2"]
    a_rows = [r for r in rows if r["key"].startswith("tc")]
    m = write_summary({"D4F": rows[:-1], "D1B1": rows[:-2], "D1A1": a_rows, "D1A12": a_rows, "D5B25": rows[:-3]})
    WORLD["fam_manifests"] = m
    defs = {}
    for arm in ("D4F", "D1A1", "D1B1", "D1A12", "D5B25"):
        exp = RC.V3FAM_EXPS[arm]
        shutil.rmtree(str(C.INC_DIR / exp), ignore_errors=True)
        defn, _summ = B.build_definition(exp, v3fam=arm, testing=True)
        (C.INC_DIR / exp / "exp.json").write_text(json.dumps(defn))
        defs[arm] = defn
    d = defs["D4F"]
    check("D4F: the family's manifest by sha256, cold_budget recomputed from its count (%d rows: %d epochs) under the "
          "v3fam record, finals dev and ImageWeeds, role baseline, inc2.train's own check passing"
          % (m["D4F"][2], RC.budget_epochs(m["D4F"][2])),
          d["base"]["manifest_sha256"] == m["D4F"][1] and d["recipe_name"] == RC.BUDGET_NAME
          and d["base"]["recipe"] == RC.cold_budget("m640", m["D4F"][2])
          and d["budget"] == RC.budget_record("m640", m["D4F"][2]) and d["final_exams"] == ["dev", "imageweeds"]
          and d["v3fam"]["arm"] == "D4F" and T.v3fam_problems(d) == [], d.get("v3fam"))
    check("D1A12: the cold table, no budget record; D1A1 and D1B1 cold_budget",
          defs["D1A12"]["base"]["recipe"] == RC.cold("m640") and "budget" not in defs["D1A12"]
          and defs["D1A1"]["recipe_name"] == RC.BUDGET_NAME and defs["D1B1"]["recipe_name"] == RC.BUDGET_NAME)
    d5 = defs["D5B25"]
    check("D5B25 builds through the pinned driver's definition (Revision 30): finals dev only, the cold table, role "
          "baseline, the family's manifest; its definition is the family's (check_definition)",
          d5["final_exams"] == ["dev"] and d5["base"]["recipe"] == RC.cold("m640") and d5["role"] == "baseline"
          and d5["base"]["manifest_sha256"] == m["D5B25"][1] and VT.check_definition("D5B25", testing_ok=True),
          {k: d5.get(k) for k in ("final_exams", "role")})
    e = refused(B.build_definition, "plain_copy", manifest=str(m["D4F"][0]), role="baseline", arm="m640",
                testing=True)
    check("  the family's manifest through a plain build (no --v3fam) refuses", e is not None and "V3F" in str(e), e)
    for arm, w0 in (("D4F", 500), ("D1A1", 510), ("D1B1", 520), ("D1A12", 530)):
        make_runs(RC.V3FAM_EXPS[arm], w0)
    check("each experiment's definition is the family's (check_definition)",
          all(VT.check_definition(a, testing_ok=True) for a in ("D4F", "D1A1", "D1B1", "D1A12")))


def w_d4():
    print("score_d4: TS-F through the family two-stage pass")
    t0 = time.time()
    rec = VT.score_d4("D4F", embedder=EMBD, **KW)
    files = [json.loads((F.arm_dir("D4F", s) / "dev.json").read_text()) for s in F.SEEDS]
    check("three dev files (%.0fs): E3's pinned classifier, E3-B's NMS, each stage-1 box set equal to its run's "
          "recorded agnostic dev AP" % (time.time() - t0),
          len(rec["dev"]) == 3 and all(f["settings"]["mode"] == "two_stage" and f["settings"]["nms"] == VT.NMS
                                       and f["stage1_vs_recorded"]["compared"] is True
                                       and f["stage2"]["classifier"] == TS._clf_cfg()[2] for f in files),
          [f.get("stage1_vs_recorded") for f in files])
    rep = json.loads(SYS.reported_path("D4F").read_text())
    check("reported: ImageWeeds two-stage and ooddev_v1 stage1 in the zoo's root, 3 each, written",
          sorted((p["exam"], p["seed"]) for p in rep["passes"] if p["status"] == "written")
          == sorted((e, s) for e in ("imageweeds", "ooddev_v1") for s in F.SEEDS),
          [p for p in rep["passes"] if p["status"] != "written"])
    cap = json.loads(SYS.score_record_path("D4F").read_text())
    check("capacity/v3fam_score_d4f.json: complete, dev files only, the build and its experiment named",
          cap["status"] == "complete" and all(f["exam"] == "dev" for f in cap["dev"]) and cap["build"] == "D4F"
          and cap["exp"] == "v3_d4_f_m640", cap.get("reported"))
    rec2 = VT.score_d4("D4F", embedder=EMBD, **KW)
    check("  a rerun keeps every file", all(f["status"] == "kept" for f in rec2["dev"]))
    ej = C.INC_DIR / "v3_d4_f_m640" / "exp.json"
    with tamper(ej, lambda d: d["base"].update(manifest_sha256="0" * 64)):
        e = refused(VT.score_d4, "D4F", embedder=EMBD, **KW)
    check("  a definition on another manifest refuses", e is not None, e)


def w_d1():
    print("score_d1: the season hold-out")
    for p in (F.exif_dir() / "verdict.json", F.exif_dir() / "preflight.json"):
        if p.exists():
            p.unlink()
    e = refused(VT.score_d1, embedder=EMBD, **KW)
    check("refused before the EXIF frame pre-flight passed", e is not None and "pre-flight" in str(e), e)
    F.exif_dir().mkdir(parents=True, exist_ok=True)
    (F.exif_dir() / "preflight.json").write_text(json.dumps({"format": F.FMT_EXIF, "numeric_passed": True}))
    (F.exif_dir() / "verdict.json").write_text(json.dumps({"format": F.FMT_EXIF_VERDICT, "frame": "yes",
                                                           "preflight_sha256": W.sha(F.exif_dir() /
                                                                                     "preflight.json")}))
    t0 = time.time()
    rec = VT.score_d1(embedder=EMBD, **KW)
    have = {(d, s, e) for d in VT.D1_DIRS.values() for s in F.SEEDS for e in VT.D1_EXAMS
            if (F.arm_dir(d, s) / ("%s.json" % e)).exists()}
    check("every arm on ood22, ood23 and dev (%d files, %.0fs); the D1 classifier pinned; ood22's and ood23's "
          "identity reproductions" % (len(have), time.time() - t0),
          len(have) == 5 * 3 * 3 and VT.d1_pin_path().exists()
          and all((F.equiv_dir() / ("identity_%s.json" % x)).exists() for x in ("ood22", "ood23")), len(have))
    a1 = json.loads((F.arm_dir("D1A1", 0) / "ood23.json").read_text())
    tsb = json.loads((F.arm_dir("TSB", 0) / "ood22.json").read_text())
    a12 = json.loads((F.arm_dir("D1A12", 0) / "ood22.json").read_text())
    check("A'1 in stage1 mode with P7's arrays; TS-B' named by the D1 classifier; A'12 identity",
          a1["settings"]["mode"] == "stage1" and (F.arm_dir("D1A1", 0) / a1["p7"]["name"]).exists()
          and tsb["settings"]["mode"] == "two_stage" and tsb["d1_classifier"]["pin_sha256"] == W.sha(VT.d1_pin_path())
          and a12["settings"]["mode"] == "identity", (a1["settings"], tsb.get("d1_classifier")))
    cap = json.loads(SYS.score_record_path("D1").read_text())
    check("capacity/v3fam_score_d1.json: dev files only (15), the ood files counted as reported",
          cap["status"] == "complete" and len(cap["dev"]) == 15 and all(f["exam"] == "dev" for f in cap["dev"]),
          cap.get("reported"))


WORLD_TESTS = ("w_builds", "w_d4", "w_d1")


def main():
    t0 = time.time()
    only = os.environ.get("V3FAM_ONLY")
    try:
        for name in UNITS:
            globals()[name]()
        WORLD["rows"] = W.build_world()
        if only != "units":
            SYT.w_setup()
            SYT.w_e3rs()
        for name in WORLD_UNITS + (() if only == "units" else WORLD_TESTS):
            try:
                globals()[name]()
            except Exception as e:               # noqa: BLE001 - one group's crash fails it, not the file
                import traceback
                check("%s ran" % name, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2000:]))
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
