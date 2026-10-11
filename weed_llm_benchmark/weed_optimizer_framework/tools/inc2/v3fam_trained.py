"""V3F's trained arms: the build arguments of inc2.baseline build --v3fam ARM
(lever L23I) and the scoring jobs of the detectors the family trains (lever
L23J): D4's field-only, no-SIU and size-matched random box detectors and
D0-S's BCLEAN as two-stage arms with E3's pinned classifier, D1's season
hold-out on ood22 and ood23, and D5b's probe doses (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered
as one family)", V3F.5 D0-S, D1, D4, D5b; V3F.6.1, V3F.6.2).

    python -m weed_optimizer_framework.tools.inc2.v3fam_trained score --arm D1|D4F|D4N|D4R15|D5B|BCLEAN   (L23J)
    python -m weed_optimizer_framework.tools.inc2.v3fam_trained build-args --v3fam ARM       (read-only)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

The platform's verbs are inc2.v3fam score --arm X (which hands these arms to
score_arm here) and inc2.baseline build --exp E --v3fam ARM --seeds 0,1,2
--role baseline (which takes its arguments from build_kwargs).

Builds (build_kwargs, V3F.6.1). Each of the twelve build arms has its
pre-registered experiment (inc2.recipes.V3FAM_EXPS) and detector; the
manifest is the arm's in the complete splits/v3fam/summary.json, hashing as
the summary records (inc2.v3fam_data.arm_manifest; D8's is LOCK v2's
train_core, inc2.v3fam_systems.d8_build_kwargs); seeds 0-2; role baseline;
finals dev and ImageWeeds (D1, D4, D8, BCLEAN). exp.json carries a v3fam
record (format inc2-v3fam-build/1: the arm, the manifest and the summary by
sha256, the recipe's name), which inc2.train.v3fam_problems checks for the
arms at equal compute. D5b's three builds have the finals dev only
(D5B_FINALS; Revision 30 of the amendment): the pinned driver (inc/driver.py,
unchanged) requires dev among a definition's finals and scores dev in every
base run, so D5b's runs carry the driver's own protocol dev scores; the
family never reads, compares or reports them (their sources here never open
a D5b run's scores/dev.json; D5b is scored on the probe only), and they are
counted among the family's dev passes. No ImageWeeds, no test.

D4 and BCLEAN (score_d4; V3F.5 D4, D0-S): each seed's base run (the final
run's weights, its recorded protocol dev score) through the family
two-stage pass with E3's pinned classifier and E3-B's NMS (the locked one:
these are one-class detectors like E1-B), its stage-1 box set's agnostic AP
equal to the run's recorded agnostic dev AP within twostage.STAGE1_TOL, or
refused; dev decides (P2, P3, Tier C; BCLEAN's rule against b_v2_m640).
Reported: ImageWeeds two-stage (agnostic and Ragweed AP); ooddev_v1 in the
zoo's root in stage1 mode, the stage-1 boxes' agnostic AP (D4 only: BCLEAN
does not read ooddev_v1, V3F.3.1).

D1 (score_d1; V3F.5 D1, decision 7). First the D1 classifier, fitted and
pinned once (capacity/v3fam_d1_classifier_pin.json) before any of D1's
two-stage passes: E3's stored training features (train_emb.npz, hashing as
E3's pin) of the crops whose key is an A' row (the keys of D1A12's
manifest, which D1A1's must equal), E3's protocol (twostage.cross_validate,
twostage._fit_lr through inc2.v3fam_embed.fit_lr_protocol). Then, after the
EXIF frame pre-flight has passed (inc2.v3fam.exif_preflight_passed: ood23's
tags 3, 6 and 8 are read only then) and ood22's and ood23's identity
reproductions (taken if missing), every seed of A'1 and B'1 in stage1 mode
(their box sets and P7's ignore-region arrays), TS-A' and TS-B' (their boxes
named by the D1 classifier) and A'12 (identity) on ood22, ood23 and dev
(dev reported only); and ooddev_v1 for A'1, B'1 (stage1) and A'12 (identity)
in the zoo's root, reported. ImageWeeds is the builds' plain finals only.

D5b (score_d5b; V3F.5 D5b): each dose's three base runs through the
identity pass on the probe, in the probe's own root (a subprocess with
INC_DIR there), after the probe's identity reproduction.

Where everything goes (V3F.5 "Outputs of every scored arm"):
  v3fam/v1/arms/<dir>/s<k>/<exam>.{json, images.npz, stage1.npz, rows.npz, p7.npz}
      dirs: D4F, D4N, D4R15, BCLEAN (two-stage); D1A1, D1B1 (stage1), TSA, TSB (two-stage, the D1 classifier),
      D1A12 (identity); D5B0, D5B25, D5B100 (identity, the probe)
  v3fam/v1/d1/classifier/{classifier.json, classifier.npz}, capacity/v3fam_d1_classifier_pin.json
  v3fam/v1/reported/<ARM>.json        the non-deciding and non-dev passes' outcome (names, sha256s, status)
  capacity/v3fam_score_<arm>.json     the platform's record: dev files, sha256s, status and counts
Every family score is written once (inc2.v3fam.commit_pass); a rerun keeps a
complete file and takes only what is missing. Every job hashes twostage/e3_v1
and capacity/e{1,2,3}_* before and after and fails if either changed.

A refusal raises inc2.v3fam.V3famRefused; the CLI prints
"[inc2.v3fam_trained] ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import common as C2
from . import recipes as RC
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_systems as SYS

# ------------------------------------------------------------------ constants (pre-registered)
TRAINED_ARMS = ("D1", "D4F", "D4N", "D4R15", "D5B", "BCLEAN")     # the L23J arms this module scores
TWO_STAGE_BUILD = {"D4F": "D4F", "D4N": "D4N", "D4R15": "D4R15", "BCLEAN": "D0BCLEAN"}
D1_BUILDS = ("D1A1", "D1B1", "D1A12")
D5B_BUILDS = ("D5B0", "D5B25", "D5B100")
SCORE_BUILDS = {"D1": D1_BUILDS, "D4F": ("D4F",), "D4N": ("D4N",), "D4R15": ("D4R15",), "D5B": D5B_BUILDS,
                "BCLEAN": ("D0BCLEAN",)}
FINALS = ("dev", "imageweeds")               # D1, D4, D8 and BCLEAN (V3F.6.1)
D5B_FINALS = ("dev",)                        # D5b: the pinned driver's minimum, never read by the family (Revision 30)
ROLE = "baseline"
NMS = TS.NMS["B"]                            # one-class detectors' boxes, taken as E3-B's (locked NMS)
NMS_REF = "B"
D4_REPORTED = {"D4F": ("imageweeds", SYS.OODDEV), "D4N": ("imageweeds", SYS.OODDEV),
               "D4R15": ("imageweeds", SYS.OODDEV), "BCLEAN": ("imageweeds",)}
D1_DIRS = {"A1": "D1A1", "B1": "D1B1", "A12": "D1A12", "TSA": "TSA", "TSB": "TSB"}
D1_DECIDING_EXAMS = ("ood22", "ood23")
D1_EXAMS = D1_DECIDING_EXAMS + ("dev",)
D1_PIN = "v3fam_d1_classifier_pin.json"      # under capacity/
FMT_BUILD = SYS.FMT_D8                       # "inc2-v3fam-build/1"
FMT_D1_CLASSIFIER = "inc2-v3fam-d1-classifier/1"
FMT_D1_PIN = "inc2-v3fam-d1-classifier-pin/1"


def log(msg):
    print("[inc2.v3fam_trained] %s" % msg, flush=True)


def _testing():
    return TS._testing_ok()


# ------------------------------------------------------------------ builds (L23I)
def build_record(arm, path, sha, n):
    """exp.json's v3fam record of a family build (inc2.train.v3fam_problems reads it)."""
    from . import v3fam_data as VD
    sp = VD.out_dir() / VD.SUMMARY
    return {"format": FMT_BUILD, "family": F.NAME, "arm": arm, "experiment": "D0" if arm == "D0BCLEAN" else arm[:2],
            "manifest": str(path), "manifest_sha256": sha, "images": int(n), "summary": str(sp),
            "summary_sha256": C.sha256_file(sp) if sp.is_file() else None, "recipe": RC.v3fam_recipe_name(arm),
            "decided_by": F.DECIDED_BY}


def finals_of(arm):
    """A build arm's finals: dev and ImageWeeds (D1, D4, BCLEAN), dev only for D5b (Revision 30)."""
    return D5B_FINALS if arm in D5B_BUILDS else FINALS


def build_kwargs(arm, exp=None, seeds=None, role=None, final_exams=None):
    """inc2.baseline build's keyword arguments for V3F's build arm `arm` (L23I: --exp <the arm's> --v3fam ARM
    --seeds 0,1,2 --role baseline): the arm's experiment, its manifest from the family summary by sha256, its
    detector, seeds 0-2, role baseline, its finals (finals_of), exp.json's v3fam record. Anything else refuses."""
    from . import v3fam_data as VD
    try:
        RC.v3fam_arm(arm)
    except RC.RecipeError as e:
        raise F.V3famRefused(str(e))
    if arm in SYS.D8_ARMS:
        return SYS.d8_build_kwargs(arm, exp=exp, seeds=seeds, role=role, final_exams=final_exams)
    want = RC.V3FAM_EXPS[arm]
    if exp is not None and exp != want:
        raise F.V3famRefused("--v3fam %s builds %s, not %s" % (arm, want, exp))
    if seeds is not None and [int(s) for s in (seeds.split(",") if isinstance(seeds, str) else seeds)] \
            != list(F.SEEDS):
        raise F.V3famRefused("V3F's builds train seeds %s" % list(F.SEEDS))
    if role not in (None, ROLE):
        raise F.V3famRefused("V3F's builds are role %s, not %s" % (ROLE, role))
    if final_exams is not None and list(final_exams) != list(finals_of(arm)):
        raise F.V3famRefused("%s's finals are %s (no test: the family's reads are its own, V3F.3.6)"
                             % (arm, list(finals_of(arm))))
    path, sha, n = VD.arm_manifest(arm)
    return {"exp": want, "manifest": str(path), "seeds": list(F.SEEDS), "arm": RC.V3FAM_DETECTOR[arm],
            "final_exams": list(finals_of(arm)), "role": ROLE, "extra": {"v3fam": build_record(arm, path, sha, n)}}


def check_definition(arm, testing_ok=False):
    """Refuse unless the experiment of build arm `arm` was built as pre-registered (its manifest the summary's
    by sha256, its detector, role, finals, seeds, recipe and v3fam record). Returns the definition."""
    from . import v3fam_data as VD
    if arm in SYS.D8_ARMS:
        return SYS.check_d8_definition(arm, testing_ok=testing_ok)
    exp = RC.V3FAM_EXPS[RC.v3fam_arm(arm)]
    defn = TS._read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(defn, dict):
        raise F.V3famRefused("%s has no exp.json: %s's build has not run" % (exp, arm))
    try:
        _path, sha, n = VD.arm_manifest(arm)
    except F.V3famRefused as e:
        raise F.V3famRefused("%s: %s" % (exp, e))
    probs = []
    base = defn.get("base") or {}
    if base.get("manifest_sha256") != sha or base.get("n_images") != n:
        probs.append("its base manifest is not the family summary's %s" % arm)
    det = RC.V3FAM_DETECTOR[arm]
    if (defn.get("arm") or {}).get("id") != det:
        probs.append("arm %r, not %s" % ((defn.get("arm") or {}).get("id"), det))
    if defn.get("role") != ROLE:
        probs.append("role %r" % defn.get("role"))
    if list(defn.get("final_exams") or []) != list(finals_of(arm)):
        probs.append("finals %s" % defn.get("final_exams"))
    if [int(s) for s in defn.get("seeds") or []] != list(F.SEEDS):
        probs.append("seeds %s" % defn.get("seeds"))
    if base.get("recipe") != RC.v3fam_recipe(arm, n):
        probs.append("not the recipe %s of %d images" % (RC.v3fam_recipe_name(arm), n))
    if arm in RC.V3FAM_BUDGET_ARMS and (defn.get("recipe_name") != RC.BUDGET_NAME
                                        or RC.check_budget_record(defn.get("budget"), det, n)):
        probs.append("no %s record of %d images" % (RC.BUDGET_NAME, n))
    v = defn.get("v3fam") or {}
    if v.get("arm") != arm or v.get("manifest_sha256") != sha:
        probs.append("no v3fam record naming %s" % arm)
    if defn.get("testing") and not testing_ok:
        probs.append("a testing build")
    if probs:
        raise F.V3famRefused("%s is not %s's pre-registered build: %s" % (exp, arm, "; ".join(probs)))
    return defn


def sources(build, testing=None):
    """{seed: run_source} of a build arm's experiment, every seed's base run done (checked first). D5b's never open
    a run's scores/dev.json (Revision 30: the family does not read D5b's dev scores)."""
    testing = _testing() if testing is None else testing
    check_definition(build, testing_ok=testing)
    return {s: SYS.run_source(RC.V3FAM_EXPS[build], s, testing_ok=testing, read_dev=build not in D5B_BUILDS)
            for s in F.SEEDS}


def stage1_vs_recorded(ps, src, what):
    """A one-class detector's stage-1 box set in a two-stage pass against its run's recorded agnostic dev AP
    (E3's check for its one-class arms, twostage.STAGE1_TOL): refused beyond it."""
    rec_ag = (src.get("recorded") or {}).get("agnostic_map50_95")
    got = ps["stage1_agnostic"][0]
    diff = abs(float(got) - float(rec_ag)) if rec_ag is not None and got is not None else None
    if diff is None or not diff <= TS.STAGE1_TOL:
        raise F.V3famRefused("%s: the stage-1 box set's agnostic AP %s is not the run's recorded %s (tolerance %g)"
                             % (what, got, rec_ag, TS.STAGE1_TOL))
    return {"compared": True, "recorded": float(rec_ag), "max_abs_diff": diff, "tolerance": TS.STAGE1_TOL,
            "file": src["recorded"]["file"], "sha256": src["recorded"]["sha256"]}


# ------------------------------------------------------------------ D4 and BCLEAN (L23J)
def score_d4(arm, embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, reported=True):
    """D4F, D4N, D4R15 or BCLEAN (module docstring): each seed's detector named by E3's pinned classifier on dev
    (deciding), then the reported passes. capacity/v3fam_score_<arm>.json (dev only)."""
    if arm not in TWO_STAGE_BUILD:
        raise F.V3famRefused("%r is not a two-stage trained arm %s" % (arm, sorted(TWO_STAGE_BUILD)))
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    build = TWO_STAGE_BUILD[arm]
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        srcs = sources(build)
        ccfg, _crec, shas = TS._clf_cfg()
        names = F.equivalence_names("dev", e3_arms=(NMS_REF,))
        eqs = F.require_equivalence(names)
        dev_files = []
        with TS._workers(workers) as wk:
            for s in F.SEEDS:
                got = SYS.existing(arm, s, "dev")
                if got is not None:
                    dev_files.append(got)
                    continue
                embedder = embedder or V.BioclipEmbedder()
                cfg = SYS._two_stage_cfg(ccfg, NMS_REF, embedder, wk)
                ps = F._family_pass(srcs[s]["weights"], "dev", cfg, arm, **kw)
                s1 = stage1_vs_recorded(ps, srcs[s], "%s seed %d dev" % (arm, s))
                dev_files.append(SYS.commit(arm, arm, s, "dev", ps, cfg, names, classifier=shas,
                                            extra={"source": SYS.source_record(srcs[s]), "build": build, "nms": NMS,
                                                   "stage1_vs_recorded": s1}))
                log("%s seed %d on dev: species %.4f, stage-1 agnostic %.4f" % (
                    arm, s, ps["result"]["species_map50_95"] or 0.0, ps["stage1_agnostic"][0] or 0.0))
            rec = SYS.score_record(arm, dev_files, eqs, before=before,
                                   reported={"status": "pending" if reported else "not run"},
                                   extra={"build": build, "exp": RC.V3FAM_EXPS[build]})
            if reported:
                passes = []
                for exam in D4_REPORTED[arm]:
                    try:
                        SYS.require_or_make_identity(exam, arm, kw)
                        ok = True
                    except Exception as e:   # noqa: BLE001 - a reported exam never fails the job
                        ok = False
                        passes.append({"exam": exam, "item": "identity_%s.json" % exam, "status": "failed",
                                       "error": "%s: %s" % (type(e).__name__, str(e)[:400])})
                    ex_names = F.equivalence_names(exam, e3_arms=(NMS_REF,) if exam != SYS.OODDEV else ())
                    for s in F.SEEDS:
                        item = {"dir": arm, "seed": s, "exam": exam}
                        if not ok:
                            passes.append(dict(item, status="failed", error="the %s identity reproduction failed"
                                               % exam))
                            continue
                        got = SYS.existing(arm, s, exam)
                        if got is not None:
                            passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
                            continue

                        def one(s=s, exam=exam, ex_names=ex_names):
                            src = srcs[s]
                            if exam in F.ROOT_EXAMS:
                                cfg = {"mode": "stage1", "nms": NMS}
                                ps = SYS.in_root(src["weights"], exam, cfg, arm, kw)
                                return SYS.commit(arm, arm, s, exam, ps, cfg, ex_names,
                                                  extra={"source": SYS.source_record(src), "build": build,
                                                         "statistic": "agnostic of the stage-1 boxes"})
                            emb = embedder or V.BioclipEmbedder()
                            cfg = SYS._two_stage_cfg(ccfg, NMS_REF, emb, wk)
                            ps = F._family_pass(src["weights"], exam, cfg, arm, **kw)
                            return SYS.commit(arm, arm, s, exam, ps, cfg, ex_names, classifier=shas,
                                              extra={"source": SYS.source_record(src), "build": build, "nms": NMS})
                        SYS.reported_pass(passes, item, one)
                rec = SYS.score_record(arm, dev_files, eqs, before=before, reported=SYS.write_reported(arm, passes),
                                       extra={"build": build, "exp": RC.V3FAM_EXPS[build]})
    log("%s: %d dev files, %.0fs" % (arm, len(dev_files), time.time() - t0))
    return rec


# ------------------------------------------------------------------ D1 (L23J)
def d1_dir():
    return F.root() / "d1" / "classifier"


def d1_pin_path():
    return C.INC_DIR / "capacity" / D1_PIN


def a_prime_keys():
    """A''s keys (D1A12's manifest, D1A1's equal by key), from the complete family summary by sha256."""
    from . import v3fam_data as VD
    keys = {}
    for arm in ("D1A12", "D1A1"):
        path, _sha, _n = VD.arm_manifest(arm)
        keys[arm] = sorted(r["key"] for r in C.read_manifest(path))
    if keys["D1A12"] != keys["D1A1"]:
        raise F.V3famRefused("A''s 12-class and one-class manifests hold different keys")
    return keys["D1A12"]


def _d1_after_fit():
    return [str(p) for d in (D1_DIRS["TSA"], D1_DIRS["TSB"]) for p in sorted((F.root() / "arms" / d).glob("s*/*.json"))]


def load_d1_classifier():
    """(classifier cfg, record, {pin_sha256, json_sha256, npz_sha256}) of the D1 classifier, its files hashing as
    its pin records."""
    import numpy as np
    pin = TS._read_json(d1_pin_path())
    if not isinstance(pin, dict) or pin.get("format") != FMT_D1_PIN:
        raise F.V3famRefused("%s is missing: the D1 classifier is fitted and pinned once, by D1's job" % d1_pin_path())
    d = d1_dir()
    got = {"json_sha256": TS._sha(d / "classifier.json"), "npz_sha256": TS._sha(d / "classifier.npz")}
    bad = [k for k, v in got.items() if v is None or v != pin.get(k)]
    if bad:
        raise F.V3famRefused("the D1 classifier's %s do not hash as its pin records: it is fitted once" % bad)
    rec = TS._read_json(d / "classifier.json")
    with np.load(str(d / "classifier.npz"), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    return ({"W": a["W"], "b": a["b"], "classes": a["classes"], "prior": a["prior"]}, rec,
            dict(got, pin_sha256=TS._sha(d1_pin_path())))


def fit_d1_classifier(testing=None):
    """The D1 classifier, fitted and pinned once (V3F.5 D1): E3's stored training features of the crops whose key
    is an A' row, E3's protocol. Refused once it exists or once a two-stage D1 pass exists."""
    import numpy as np
    from . import v3fam_embed as EMB
    testing = _testing() if testing is None else testing
    d = d1_dir()
    if (d / "classifier.json").exists() or d1_pin_path().exists() or _d1_after_fit():
        raise F.V3famRefused("the D1 classifier was fitted already or a D1 two-stage pass exists: fitted once")
    t0 = time.time()
    keys = set(a_prime_keys())
    Fm, rows, frec = EMB.e3_train_features()
    sel = [i for i, r in enumerate(rows) if r["key"] in keys]
    if not sel:
        raise F.V3famRefused("no training crop of E3's classifier lies on an A' row")
    used = {rows[i]["key"] for i in sel}
    X32 = V._norm(np.asarray(Fm, dtype=np.float32)[sel])
    y = np.array([rows[i]["label"] for i in sel], dtype=np.int64)
    groups = np.array([rows[i]["session"] for i in sel])
    arrays, rec = EMB.fit_lr_protocol(X32, y, groups)
    sh = F.write_once(d / "classifier.npz", EMB._npz_bytes(arrays))
    doc = {"format": FMT_D1_CLASSIFIER, "family": F.NAME, "decided_by": F.DECIDED_BY,
           "features": {k: frec.get(k) for k in ("train_crops", "train_emb", "e3_classifier", "e3_pin_sha256",
                                                  "base_v2_sha256")},
           "a_prime": {"rows": len(keys), "rows_with_crops": len(used), "crops": len(sel),
                       "crop_ids_sha256": C.sha256_text("\n".join(str(rows[i]["crop_id"]) for i in sel))},
           "files": {"npz": "classifier.npz", "npz_sha256": sh},
           "protocol": "E3's: twostage.cross_validate (5-fold GroupKFold by session, pooled log-loss, ties to the "
                       "smaller C) and twostage._fit_lr (lbfgs, max_iter %d, no class weights), on E3's stored "
                       "BioCLIP-2 features" % TS.MAX_ITER,
           "limitation": "OtherPlant crops are NDSU's only, and most PalmerAmaranth and Purslane crops are gone with "
                         "tsw22/tsw23 (V3F.5 D1, O10)",
           "testing": bool(testing), "code": dict(F.code_shas(), v3fam_trained=C.sha256_file(__file__)),
           "seconds": round(time.time() - t0, 3), "written_utc": TS._utc()}
    doc.update(rec)
    jsha = F.write_once(d / "classifier.json", doc)
    F.write_once(d1_pin_path(), {"format": FMT_D1_PIN, "json_sha256": jsha, "npz_sha256": sh,
                                 "train_emb_sha256": (frec.get("train_emb") or {}).get("sha256"),
                                 "written_utc": doc["written_utc"],
                                 "note": "fitted once, before any of D1's two-stage passes; never rewritten"})
    log("D1 classifier: %d crops on %d A' rows, C=%g, train top-1 %.4f" % (len(sel), len(used), rec["fit"]["C"],
                                                                         rec["fit"]["train_top1"]))
    return load_d1_classifier()


def d1_classifier(testing=None):
    if d1_pin_path().exists():
        return load_d1_classifier()
    return fit_d1_classifier(testing=testing)


def score_d1(embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, reported=True):
    """D1's scoring job (module docstring): the D1 classifier, the pre-flight and the reproductions, every arm on
    ood22, ood23 and dev, ooddev_v1 reported. capacity/v3fam_score_d1.json (dev only)."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        srcs = {b: sources(b) for b in D1_BUILDS}
        if not F.exif_preflight_passed():
            raise F.V3famRefused("the EXIF frame pre-flight has not passed (V3F.3.2): D1 reads ood23's tags 3, 6 and 8 "
                                 "only after it (inc2.v3fam exif-preflight, then exif-verdict by a named viewer)")
        dcfg, _drec, dshas = d1_classifier()
        dev_names = F.equivalence_names("dev")
        eqs = F.require_equivalence(dev_names)
        for exam in D1_DECIDING_EXAMS:
            SYS.require_or_make_identity(exam, "D1", kw)
        plan = [("A1", "D1A1", "stage1"), ("B1", "D1B1", "stage1"), ("TSA", "D1A1", "two_stage"),
                ("TSB", "D1B1", "two_stage"), ("A12", "D1A12", "identity")]
        dev_files, passes = [], []
        with TS._workers(workers) as wk:
            for exam in D1_EXAMS:
                names = F.equivalence_names(exam)
                for key, build, mode in plan:
                    dirname = D1_DIRS[key]
                    for s in F.SEEDS:
                        got = SYS.existing(dirname, s, exam)
                        if got is None:
                            src = srcs[build][s]
                            if mode == "two_stage":
                                embedder = embedder or V.BioclipEmbedder()
                                cfg = dict(dcfg, mode="two_stage", nms=NMS, embedder=embedder, workers=wk,
                                           top_k=TS.TOP_K)
                                clf = dshas
                            else:
                                cfg, clf = {"mode": mode, "nms": NMS if mode == "stage1" else "locked"}, None
                            ps = F._family_pass(src["weights"], exam, cfg, "D1", **kw)
                            got = SYS.commit("D1", dirname, s, exam, ps, cfg, names, classifier=clf,
                                             extra={"source": SYS.source_record(src), "build": build,
                                                    "d1": key, "d1_classifier": clf})
                            log("D1 %s seed %d on %s: species %s, agnostic %s" % (
                                key, s, exam, ps["result"].get("species_map50_95"),
                                ps["result"].get("agnostic_map50_95")))
                        if exam == "dev":
                            dev_files.append(got)
                        else:
                            passes.append({"dir": dirname, "seed": s, "exam": exam, "file": got["file"],
                                           "sha256": got["sha256"], "status": got["status"], "deciding": True})
            rec = SYS.score_record("D1", dev_files, eqs, before=before,
                                   reported={"status": "pending" if reported else "not run"},
                                   extra={"d1_classifier": dshas, "deciding_exams": len(D1_DECIDING_EXAMS)})
            if reported:
                try:
                    SYS.require_or_make_identity(SYS.OODDEV, "D1", kw)
                    od_ok = True
                except Exception as e:       # noqa: BLE001 - a reported exam never fails the job
                    od_ok = False
                    passes.append({"exam": SYS.OODDEV, "item": "identity_%s.json" % SYS.OODDEV, "status": "failed",
                                   "error": "%s: %s" % (type(e).__name__, str(e)[:400])})
                od_names = F.equivalence_names(SYS.OODDEV)
                for key, build, mode in [x for x in plan if x[2] != "two_stage"]:
                    dirname = D1_DIRS[key]
                    for s in F.SEEDS:
                        item = {"dir": dirname, "seed": s, "exam": SYS.OODDEV}
                        if not od_ok:
                            passes.append(dict(item, status="failed", error="the ooddev_v1 identity reproduction "
                                                                           "failed"))
                            continue
                        got = SYS.existing(dirname, s, SYS.OODDEV)
                        if got is not None:
                            passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
                            continue

                        def one(build=build, mode=mode, dirname=dirname, s=s, key=key):
                            src = srcs[build][s]
                            cfg = {"mode": mode, "nms": NMS if mode == "stage1" else "locked"}
                            ps = SYS.in_root(src["weights"], SYS.OODDEV, cfg, "D1", kw)
                            return SYS.commit("D1", dirname, s, SYS.OODDEV, ps, cfg, od_names,
                                              extra={"source": SYS.source_record(src), "build": build, "d1": key,
                                                     "statistic": "agnostic (weed boxes only)"})
                        SYS.reported_pass(passes, item, one)
            rec = SYS.score_record("D1", dev_files, eqs, before=before, reported=SYS.write_reported("D1", passes),
                                   extra={"d1_classifier": dshas, "deciding_exams": len(D1_DECIDING_EXAMS)})
    log("D1: %d dev files, %d other passes, %.0fs" % (len(dev_files), len(passes), time.time() - t0))
    return rec


# ------------------------------------------------------------------ D5b (L23J)
def score_d5b(lock_check=True, batch=S.BATCH, device=None):
    """D5b's scoring job: each dose's three base runs through the identity pass on the probe in its own root,
    after the probe's identity reproduction; nothing else is read. capacity/v3fam_score_d5b.json (no dev file)."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        srcs = {b: sources(b) for b in D5B_BUILDS}
        eqs = SYS.require_or_make_identity("probe", "D5B", kw)
        names = F.equivalence_names("probe")
        passes = []
        cfg = {"mode": "identity", "nms": "locked"}
        for build in D5B_BUILDS:
            for s in F.SEEDS:
                got = SYS.existing(build, s, "probe")
                if got is None:
                    src = srcs[build][s]
                    ps = SYS.in_root(src["weights"], "probe", cfg, "D5B", kw)
                    got = SYS.commit("D5B", build, s, "probe", ps, cfg, names,
                                     extra={"source": SYS.source_record(src), "build": build})
                passes.append({"dir": build, "seed": s, "exam": "probe", "file": got["file"], "sha256": got["sha256"],
                               "status": got["status"], "deciding": True})
        rec = SYS.score_record("D5B", [], eqs, before=before, reported=SYS.write_reported("D5B", passes),
                               extra={"note_d5b": "D5b reads the probe only: no dev file (V3F.5 D5b)"})
    log("D5B: %d probe passes, %.0fs" % (len(passes), time.time() - t0))
    return rec


# ------------------------------------------------------------------ dispatch
def score_arm(arm, embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, reported=True):
    """inc2.v3fam score --arm X for this module's arms (D1, D4F, D4N, D4R15, D5B, BCLEAN)."""
    if arm not in TRAINED_ARMS:
        raise F.V3famRefused("--arm %r is not one of %s" % (arm, list(TRAINED_ARMS)))
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    if arm == "D1":
        return score_d1(embedder=embedder, workers=workers, reported=reported, **kw)
    if arm == "D5B":
        return score_d5b(**kw)
    return score_d4(arm, embedder=embedder, workers=workers, reported=reported, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F's trained arms: D1, D4F, D4N, D4R15, D5B and BCLEAN (scores) and "
                                             "the arguments of inc2.baseline build --v3fam.")
    ap.add_argument("command", choices=("score", "build-args"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--v3fam", default=None, help="build-args: the build arm")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam_trained] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    kw = {"lock_check": not a.no_lock_check_for_tests, "batch": a.batch or S.BATCH, "device": a.device}
    try:
        if a.command == "score":
            score_arm(a.arm, **kw)
        else:
            print(json.dumps(build_kwargs(a.v3fam), indent=1, sort_keys=True, default=str))
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError,
            C2.Inc2Error, ValueError) as e:
        print("[inc2.v3fam_trained] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
