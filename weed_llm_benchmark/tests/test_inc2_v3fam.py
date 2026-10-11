#!/usr/bin/env python3
"""V3F core: the family validator, the family pass, the reproductions and the
statistics (inc2/v3fam.py; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11):
V3F, the v3 experiment family D0-D9 (pre-registered as one family)").

Two groups. UNITS need no scoring pass (V3FAM_ONLY=units runs only them):
- the constants (seeds, 1,000 resamples, SESOI 0.005, the 90 % interval, C =
  14 and its bar 2.690110, Holm's sets, the read cap and order, read_by, the
  260 GPU-h cap, the exam allow-lists, the arm lists) and the six E3-frozen
  sha256s of V3F.1 (asserted literally; a changed file refuses every verb),
  the amendment file's sha256;
- the session rule (the examples of V3F.3.1, the 66 iOS keys one session, an
  unknown form refused), the image bootstrap equal to inc2.baseline's
  native_bootstrap and agnostic_bootstrap exactly, the session bootstrap on a
  hand-checked case, the selection correction (its identity, ties to M, the
  plain D never deciding), the E-rule (each condition alone, the one-system
  rule), combined z (E3-B's recorded 3.87), the outcome classes and flags,
  Holm per exam (the worked check of V3F.3.4, a not-run primary keeping m),
  the C = 14 gate counting a not-run arm (and 0.05/13 not used), the read order
  and cap with BCLEAN's priority, read_by, the family cap (the worked example
  of V3F.7, running items at their estimate, never retried), the ignore-region
  statistic on a hand-computed 5-box case, subsets and the dev-session stratum,
  the stage-1 comparison, the reproduction branches (1e-6 / 0.002), write
  once, the E-series trees, the EXIF frame check and the pre-flight records.
WORLD runs real Ultralytics CPU passes in the synthetic world of
tests/test_inc2_e3.py (b_v2_m640, e1_a_m640, e1_b_m640 fixtures, the oracle
detector, the stub embedder; E3's own score-arm writes E3's records first):
- the validator layering (one family layer over exactly one E3 layer, the
  scorer's validator restored after a pass and after a refusal);
- identity equal to the locked scorer exactly; two_stage equal to E3's own
  pass exactly; stage1 with P7's arrays; inject_rows reproducing identity;
  inject_boxes reproducing two_stage; oracle; view640's cut hook;
- every exam allow-list, the test token, the roots, the mode refusals, the
  EXIF tags (6 refused on dev, on ood22 before the pre-flight; read after it,
  its true tag recorded);
- the four reproductions written once (1e-6 passed, a 0.001 deviation recorded,
  0.01 refused writing nothing), item 3 on imageweeds, ood22 and the probe root
  (a subprocess with INC_DIR there), require_equivalence, commit_pass written
  once (features on dev only), the E-series trees unchanged by family passes,
  the CLI.

Run:  python3 tests/test_inc2_v3fam.py          (V3FAM_ONLY=units: the units only)
"""
import collections
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
import test_inc2_e3 as E3T  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = E3T.patched
moved = E3T.moved
tamper = E3T.tamper
KW = {"batch": 32, "device": "cpu", "lock_check": True}
EMB = E3T.EMB
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORLD = {}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused) as e:
        return e
    return None


def safe(fn, *a, **k):
    """fn's value, or the exception it raised (so a check fails rather than the file crashing)."""
    try:
        return fn(*a, **k)
    except Exception as e:                       # noqa: BLE001 - reported by the check that reads it
        return e


# ------------------------------------------------------------------ synthetic per-image arrays
def synth(seed, n=24, k=4, species=(0, 1, 2, 3), gt_seed=99):
    """Species arrays of one run on an exam of n images (two GT boxes each, one GT draw for every run); confs
    rounded to 0.01 so ties occur."""
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(gt_seed)
    keys = np.asarray(["k%03d" % i for i in range(n)])
    pimg = np.repeat(np.arange(n), k)
    tp = rng.random((n * k, 10)) < np.linspace(0.85, 0.2, 10)
    conf = np.round(rng.random(n * k), 2)
    pcls = rng.choice(np.asarray(species, dtype=float), n * k)
    timg = np.repeat(np.arange(n), 2)
    tcls = g.choice(np.asarray(species, dtype=float), 2 * n)
    return {"keys": keys, "tp": tp, "conf": conf, "pred_cls": pcls, "pred_img": pimg, "target_img": timg,
            "target_cls": tcls}


def agnostic_of(a):
    n = len(a["keys"])
    return {"keys": a["keys"], "tp": a["tp"], "conf": a["conf"], "pred_img": a["pred_img"],
            "n_gt": np.bincount(a["target_img"], minlength=n)}


# ================================================================== UNITS
def u_constants():
    print("the family's constants (pre-registered)")
    check("seeds 0-2, 1,000 resamples, SESOI 0.005, one-sided alpha 0.05, the 90 % interval's z the exact "
          "Phi^-1(0.95) (the text's 1.645)", F.SEEDS == (0, 1, 2) and F.RESAMPLES == 1000 and F.SESOI == 0.005
          and F.ALPHA == 0.05 and abs(F.Z90 - 1.6448536269514722) < 1e-12 and round(F.Z90, 3) == 1.645, F.Z90)
    check("C = 14 registered challengers (the nine D2 alternatives, D3, TS-F, TS-N, TS-R15, RF-DETR-L), bar "
          "Phi^-1(1 - 0.05/14) = 2.690110", F.C_CHALLENGERS == 14 and len(set(F.CHALLENGERS)) == 14
          and F.CHALLENGERS[:9] == ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1",
                                    "D2BC2640", "D2FT")
          and set(F.CHALLENGERS[9:]) == {"D3", "D4F", "D4N", "D4R15", "D6"}
          and "%.6f" % F.CHALLENGER_BAR == "2.690110", F.CHALLENGER_BAR)
    check("nine primaries; Holm per exam: dev P1-P5, ood22 P6, ood23 P7, probe P8-P9; P2, P3, P5 selection-"
          "corrected; P6-P9 the larger SE; P4 one system",
          sorted(F.PRIMARIES) == ["P%d" % i for i in range(1, 10)]
          and F.HOLM_SETS == {"dev": ("P1", "P2", "P3", "P4", "P5"), "ood22": ("P6",), "ood23": ("P7",),
                              "probe": ("P8", "P9")}
          and all(F.PRIMARIES[p]["exam"] == e for e, ps in F.HOLM_SETS.items() for p in ps)
          and F.SELECTION_PRIMARIES == ("P2", "P3", "P5") and F.SESSION_PRIMARIES == ("P6", "P7", "P8", "P9")
          and F.ONE_SYSTEM_PRIMARIES == ("P4",) and F.SELECTION_ARMS == ("M", "A", "B") and F.SELECTED == "B")
    check("reads: at most 5, challengers 3 (2 once BCLEAN reads), references D6 then D8-m; read_by exactly the "
          "owner's or the delegate's name; the family cap 260 GPU-h",
          F.READ_CAP == 5 and F.CHALLENGER_READS == 3 and F.CHALLENGER_READS_BCLEAN == 2
          and F.REFERENCE_READS == ("D6", "D8M")
          and F.READ_BY == {"owner": "Harry He (owner)",
                            "delegate": "Claude (delegate under the 2026-09-30 owner grant)"}
          and F.DECIDED_BY_ENV == {"delegate": "assistant:claude-delegate"} and F.FAMILY_CAP_GPUH == 260.0
          and F.NOT_RUN_CAP == "not run (family cap)")
    check("the exam allow-lists of V3F.3.1: D5b reads only the probe, D1 alone ood22/ood23, test no arm (a token)",
          F.EXAM_READERS["probe"] == ("D5B",) and F.EXAM_READERS["ood22"] == ("D1",)
          and F.EXAM_READERS["ood23"] == ("D1",) and "D5B" not in F.EXAM_READERS["dev"]
          and "D2" not in F.EXAM_READERS["imageweeds"] and "D9" not in F.EXAM_READERS["imageweeds"]
          and "D5B" not in F.EXAM_READERS["ooddev_v1"] and F.EXAM_READERS["test"] == ()
          and F.ROOT_EXAMS == ("ooddev_v1", "probe") and F.OOD_EXAMS == ("ood22", "ood23"))
    check("the L23J arms (20) and the L23I builds (12); the modes; the tolerances 1e-6 / 0.002, stage-1 1e-3 px and "
          "1e-4", len(F.SCORE_ARMS) == 20 and F.SCORE_ARMS[0] == "E3RS" and len(F.BUILD_ARMS) == 12
          and F.MODES == ("identity", "two_stage", "gt", "stage1", "inject_boxes", "inject_rows", "oracle", "view640")
          and F.REPRO_TOL == 1e-6 and F.DEVIATION_TOL == 0.002 and F.STAGE1_XYXY_TOL == 1e-3
          and F.STAGE1_Q_TOL == 1e-4)
    check("DECIDED_BY names the amendment, its commit and its sha256",
          "55cc2d9" in F.DECIDED_BY and F.PREREG_SHA256 in F.DECIDED_BY and "V3F" in F.DECIDED_BY
          and F.PREREG_COMMIT.startswith("55cc2d9"))
    p = REPO_ROOT / F.PREREG_FILE
    check("the amendment's byte copy %s hashes to PREREG_SHA256" % F.PREREG_FILE,
          p.is_file() and C.sha256_file(p) == F.PREREG_SHA256, p)


def u_frozen():
    print("the six E3-frozen files (V3F.1)")
    want = {"twostage": ("8fe2edc0", "8ddc"), "scorer": ("18c00837", "2de4"), "scorer_sidecar": ("de7aacda", "5e52"),
            "scorer_native": ("ba9eeb92", "6b7d"), "verify": ("602eba3c", "aa10"),
            "semisup_labeler": ("03b28c50", "458c")}
    check("FROZEN_SHA256 holds V3F.1's six (prefix and suffix as the amendment cites them)",
          set(F.FROZEN_SHA256) == set(want) and all(F.FROZEN_SHA256[k].startswith(a) and F.FROZEN_SHA256[k].endswith(b)
                                                    for k, (a, b) in want.items()))
    now = {k: C.sha256_file(pathlib.Path(v).resolve()) for k, v in TS.CODE_FILES.items()}
    check("the six files on disk hash as FROZEN_SHA256 and check_frozen passes", now == F.FROZEN_SHA256
          and F.check_frozen() == now, {k: (now[k][:12], F.FROZEN_SHA256.get(k, "")[:12]) for k in now})
    with patched(TS, "_code_shas", lambda: dict(now, scorer="0" * 64)):
        e = refused(F.check_frozen)
        e2 = refused(F._family_pass, "w.pt", "dev", {"mode": "identity"}, "E3RS")
        e3 = refused(F.require_equivalence, ["dev_identity.json"])
    check("a changed frozen file refuses check_frozen, the family pass and require_equivalence", e is not None
          and "scorer" in str(e) and e2 is not None and e3 is not None, (e, e2, e3))
    cs = F.code_shas()
    check("code_shas: the six and this module", set(cs) == set(F.FROZEN_SHA256) | {"v3fam"}
          and cs["v3fam"] == C.sha256_file(pathlib.Path(F.__file__).resolve()))


def u_sessions():
    print("the session rule (V3F.3.1)")
    ok = {("ood22", "ood22__20220618_IphoneSE_YL_1"): "20220618_IphoneSE_YL",
          ("ood22", "ood22__20220711_123456789_iOS"): "20220711_iOS",
          ("ood23", "ood23__20230720_HTRC_EDMUND_MP_867"): "20230720_HTRC_EDMUND_MP",
          ("ood23", "ood23__20230629_HTRC_EDMUND_MP_12"): "20230629_HTRC_EDMUND_MP"}
    got = {k: safe(F.session_of, k[0], k[1]) for k in ok}
    check("ood keys: the exam prefix stripped, a trailing _<digits> dropped, <YYYYMMDD>_<9 digits>_iOS -> "
          "<YYYYMMDD>_iOS", got == ok, got)
    ios = ["ood22__20220711_%09d_iOS" % (100000000 + i) for i in range(66)]
    check("the 66 iOS keys of one day are one session", {str(safe(F.session_of, "ood22", k)) for k in ios}
          == {"20220711_iOS"})
    bad = {"another form": ("ood22", "ood22__IMG"), "no prefix": ("ood22", "20220618_IphoneSE_YL_1"),
           "the other exam's prefix": ("ood23", "ood22__20220618_IphoneSE_YL_1"),
           "iOS with 8 digits": ("ood22", "ood22__20220711_12345678_iOS_x"),
           "an exam without a rule": ("imageweeds", "iw_001"), "dev without its field": ("dev", "x")}
    out = {k: refused(F.session_of, *v) for k, v in bad.items()}
    check("refused: %s" % ", ".join(out), all(e is not None for e in out.values()), out)
    dev = ["20210903_iPhone11Pro_BT_148", "20210903_iPhone11Pro_BT_7", "20210812_NIKOND3300_YL_3"]
    check("the key rule applied to dev keys gives their sessions; dev, test and probe read the manifest's field",
          [F.key_session(k) for k in dev] == ["20210903_iPhone11Pro_BT", "20210903_iPhone11Pro_BT",
                                              "20210812_NIKOND3300_YL"]
          and F.session_of("dev", "x", "s7") == "s7" and F.session_of("probe", "y", "S") == "S"
          and F.sessions_for("dev", [{"key": "b", "session": "s2"}, {"key": "a", "session": "s1"}]) == ["s1", "s2"])
    rows_t = [{"key": "t%d" % i, "session": "s%d" % (i % 5)} for i in range(10)]
    rows_d = [{"key": "d%d" % i, "session": "s%d" % i} for i in (0, 3)]
    check("the dev-session stratum of test: the test keys from dev's sessions",
          F.dev_session_stratum(rows_t, rows_d) == ["t0", "t3", "t5", "t8"])


def u_bootstrap_exact():
    print("the image bootstrap reproduces inc2.baseline's native_bootstrap and agnostic_bootstrap exactly")
    X = [synth(s) for s in (1, 2, 3)]
    Y = [synth(s) for s in (4, 5, 6)]
    t0 = time.time()
    st = F.seed_text("P1", "image")
    nb = B.native_bootstrap(X, Y, resamples=1000, seed_text=st)
    fb = F.image_bootstrap(X, Y, st, resamples=1000)
    check("species: the same SE to the bit (%.10f) and n_valid (%d), 1,000 resamples, seed text %s (%.0fs)"
          % (fb["se"], fb["n_valid"], st, time.time() - t0), fb["se"] == nb["se"] and fb["n_valid"] == nb["n_valid"]
          and st == "inc2/v3fam/P1/image", (fb["se"], nb["se"]))
    check("  it returns every resample's difference (mean over X runs - mean over Y runs)",
          len(fb["diffs"]) == fb["n_valid"] and abs(F._sd(fb["diffs"]) - fb["se"]) == 0
          and fb["diffs"][0] == statistics.fmean(fb["Vx"][0]) - statistics.fmean(fb["Vy"][0]))
    XA, YA = [agnostic_of(a) for a in X], [agnostic_of(a) for a in Y]
    na = B.agnostic_bootstrap(XA, YA, resamples=1000, seed_text=st)
    fa = F.image_bootstrap(XA, YA, st, resamples=1000, statistic="agnostic")
    check("agnostic: the same SE to the bit as agnostic_bootstrap (%.10f)" % fa["se"], fa["se"] == na["se"]
          and fa["n_valid"] == na["n_valid"], (fa["se"], na["se"]))
    bad = dict(Y[0], target_cls=np.roll(Y[0]["target_cls"], 1))
    e = refused(F.image_bootstrap, X, [bad] + Y[1:], st, resamples=10)
    e2 = refused(F.image_bootstrap, X, [dict(Y[0], keys=Y[0]["keys"][::-1])] + Y[1:], st, resamples=10)
    check("runs scored on another exam (other GT) or in another key order are refused", e is not None
          and e2 is not None, (e, e2))
    m = F.statistic_value(X[0])
    tb = SC.tie_break(X[0])
    hand = statistics.fmean(SC.class_ap(*SC.species_arrays(tb, s)[:2], int(SC.species_arrays(tb, s)[3].sum()), s,
                                        SC._ap_per_class()) for s in (0, 1, 2, 3))
    check("statistic_value: the species mean of the tie-broken arrays (every image once)", m == hand, (m, hand))


def u_session_bootstrap():
    print("the session bootstrap (decision 6) on a hand-checked case")
    sessions = ["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6
    cnt = F.session_counts(sessions, "inc2/v3fam/T/session", resamples=50)
    rng = np.random.default_rng(C.stable_int("inc2/v3fam/T/session"))
    draws = rng.integers(0, 4, size=(50, 4))
    hand = np.stack([np.repeat(np.bincount(r, minlength=4), 6) for r in draws])
    check("multiplicities: the sorted sessions drawn with replacement by default_rng(stable_int(seed text)), every "
          "image of a session taken with its session's count", np.array_equal(cnt, hand)
          and (cnt.sum(1) == 24).all(), cnt[:2])
    X = [synth(s) for s in (11, 12, 13)]
    Y = [dict(x) for x in X]
    for y in Y:                                  # Y differs from X only on session d's images (18..23)
        sel = y["pred_img"] >= 18
        conf = y["conf"].copy()
        conf[sel] = np.round(conf[sel] * 0.5, 3)
        y["conf"] = conf
    res = F.session_bootstrap(X, Y, sessions, "inc2/v3fam/T/session", resamples=200)
    rng = np.random.default_rng(C.stable_int("inc2/v3fam/T/session"))
    d_drawn = rng.integers(0, 4, size=(200, 4))
    no_d = [r for r in range(200) if 3 not in d_drawn[r]]
    zero = all(res["diffs"][res["kept"].index(r)] == 0.0 for r in no_d if r in res["kept"])
    check("X and Y differ only on one session: every resample that does not draw it gives D = 0 exactly (%d of 200), "
          "and the session SE is positive (%.4f)" % (len(no_d), res["se"]), zero and len(no_d) > 0 and res["se"] > 0,
          (len(no_d), res["se"]))
    check("  fewer than 2 sessions, or a session list of another length, is refused",
          refused(F.session_counts, ["a"] * 5, "x") is not None
          and refused(F.session_bootstrap, X, Y, sessions[:-1], "x") is not None)
    xv = [F.statistic_value(a) for a in X]
    yv = [F.statistic_value(a) for a in Y]
    p8 = F.comparison("P8", xv, yv, X, Y, sessions=sessions, se_rule="max", resamples=200)
    p1 = F.comparison("P1", xv, yv, X, Y, sessions=sessions, se_rule="image", resamples=200, paired=True)
    check("P6-P9 decide on the larger of the image SE (%.4f) and the session SE (%.4f), seed texts "
          "inc2/v3fam/<id>/image and /session; on dev the session SE is reported beside, never deciding"
          % (p8["image"]["se"], p8["session"]["se"]),
          p8["decision"]["se"] == max(p8["image"]["se"], p8["session"]["se"]) and p8["session"]["deciding"]
          and p8["session"]["seed_text"] == "inc2/v3fam/P8/session" and p8["image"]["seed_text"] ==
          "inc2/v3fam/P8/image" and p8["session"]["n_sessions"] == 4 and p1["decision"]["se"] == p1["image"]["se"]
          and p1["session"]["deciding"] is False and len(p1["paired_per_seed"]) == 3, (p8["decision"], p1["session"]))
    Z = [dict(x) for x in X]
    for z_ in Z:                                 # Z loses every true positive of sessions c and d: a session-level effect
        tp = z_["tp"].copy()
        tp[z_["pred_img"] >= 12] = False
        z_["tp"] = tp
    zv = [F.statistic_value(a) for a in Z]
    p8s = F.comparison("P8", xv, zv, X, Z, sessions=sessions, se_rule="max", resamples=200)
    check("  where the effect lives in whole sessions the session SE (%.4f) exceeds the image SE (%.4f) and decides"
          % (p8s["session"]["se"], p8s["image"]["se"]), p8s["session"]["se"] > 1.5 * p8s["image"]["se"]
          and p8s["decision"]["se"] == p8s["session"]["se"], p8s["decision"])
    check("  a comparison deciding on the larger SE without sessions is refused",
          refused(F.comparison, "P6", xv, yv, X, Y, se_rule="max", resamples=20) is not None)
    keep = [str(k) for k in X[0]["keys"] if int(str(k)[1:]) % 4 != 1]
    ph = F.comparison("P1", xv, yv, X, Y, sessions=sessions, keep=keep, resamples=50)
    sub = [F.statistic_value(F.subset_arrays(a, keep)) for a in X]
    check("dev minus H (D0-S's dev branch): every array restricted to the kept images, the full-set values "
          "recomputed from them, the bootstrap over those images only",
          ph["n_images"] == len(keep) and ph["x_values"] == sub and ph["restricted_to"]["n_images"] == len(keep)
          and ph["session"]["n_sessions"] == 4, (ph["n_images"], ph["restricted_to"]))


def u_selection():
    print("the selection correction of P2, P3 and P5 (decision 5)")
    Y = [synth(s) for s in (31, 32, 33)]
    good = [synth(s) for s in (34, 35, 36)]
    for a in good:                               # E3-B clearly best (about 0.99 against 0.57): a true positive at
        hit = a["tp"][:, 0]                      # 0.50 is one at every threshold and ranks above every miss
        a["tp"] = np.repeat(hit[:, None], 10, axis=1)
        a["conf"] = np.where(hit, 0.5 + 0.5 * a["conf"], 0.5 * a["conf"])
    M = [synth(s) for s in (37, 38, 39)]
    A = [synth(s) for s in (40, 41, 42)]
    vals = {k: [F.statistic_value(a) for a in runs] for k, runs in (("M", M), ("A", A), ("B", good))}
    yv = [F.statistic_value(a) for a in Y]
    d = statistics.fmean(vals["B"]) - statistics.fmean(yv)
    counts = F.image_counts(len(Y[0]["keys"]), F.seed_text("P2", "image"), 300)
    sc = F.selection_correct({"M": M, "A": A, "B": good}, vals, Y, counts, d, Vy=None)
    plain = F._bootstrap(good, Y, counts, "species", SC.SPECIES, None)
    own = statistics.fmean(plain["diffs"]) - (sc["full_set"]["B"] - sc["full_set"]["Y"])
    check("B always chosen: share 1, SE_sel = the plain image SE of B - Y exactly, and the correction is the "
          "bootstrap's own bias of B - Y (mean of the resample differences - the full-set difference)",
          sc["share_chosen"]["B"] == 1.0 and sc["se_sel"] == plain["se"] and abs(sc["bias"] - own) < 1e-12,
          (sc["share_chosen"], sc["bias"], own, sc["se_sel"], plain["se"]))
    hand = statistics.fmean(t - (sc["full_set"][c] - sc["full_set"]["Y"]) for t, c in zip(sc["T"], sc["chosen"]))
    check("  D_sel = D - mean_r[T_r - (S_c(r) - S_Y)], by hand from the returned draws", abs(sc["bias"] - hand) < 1e-15
          and abs(sc["diff_sel"] - (d - hand)) < 1e-15, (sc["bias"], hand))
    same = F.selection_correct({"M": good, "A": good, "B": good}, vals, Y, counts, d)
    check("three equal arms: every tie goes to M (E3's order), the same draws and bias as B alone",
          same["share_chosen"]["M"] == 1.0 and same["T"] == sc["T"] and same["bias"] == sc["bias"],
          same["share_chosen"])
    noisy = {"M": [synth(s) for s in (51, 52, 53)], "A": [synth(s) for s in (54, 55, 56)],
             "B": [synth(s) for s in (57, 58, 59)]}
    nv = {k: [F.statistic_value(a) for a in r] for k, r in noisy.items()}
    dn = statistics.fmean(nv["B"]) - statistics.fmean(yv)
    rec = F.comparison("P2", nv["B"], yv, noisy["B"], Y, se_rule="selection", resamples=200,
                       selection={"runs": noisy, "values": nv})
    hand = statistics.fmean(t - (rec["selection"]["full_set"][c] - rec["selection"]["full_set"]["Y"])
                            for t, c in zip(rec["selection"]["T"], rec["selection"]["chosen"]))
    check("comparisons of exchangeable arms: the choice in the resamples is not always B (%s), the bias is the hand "
          "value, and the rule decides on D_sel and SE_sel, the plain D and SE reported beside"
          % {k: round(v, 3) for k, v in rec["selection"]["share_chosen"].items()},
          rec["selection"]["share_chosen"]["B"] < 1.0 and abs(rec["selection"]["bias"] - hand) < 1e-15
          and rec["decision"]["diff"] == rec["selection"]["diff_sel"] != rec["plain"]["diff"] == dn
          and rec["decision"]["se"] == rec["selection"]["se_sel"] and rec["plain"]["se"] == rec["image"]["se"],
          (rec["decision"]["diff"], rec["plain"]))
    e = refused(F.comparison, "P2", nv["A"], yv, noisy["A"], Y, se_rule="selection", resamples=20,
                selection={"runs": noisy, "values": nv})
    e2 = refused(F.comparison, "P2", nv["B"], yv, noisy["B"], Y, se_rule="selection", resamples=20)
    check("  X other than E3-B's arrays, or no E3 arrays at all, is refused", e is not None and e2 is not None, (e, e2))


def u_rule():
    print("the E-rule, combined z and the outcome classes (V3F.3.3)")
    x, y = [0.8850, 0.8853, 0.8874], [0.85465, 0.84976, 0.85282]
    r = F.decide(0.0335, 0.0085, x, y)
    sx, sy = statistics.stdev(x), statistics.stdev(y)
    check("E3-B's recorded D +0.0335, SE 0.0085: qualifies, z %.2f (V3F.3.4: 3.87), sigma^2 = SE^2 + (sd_X^2 + "
          "sd_Y^2) / 3, pooled sd = sqrt((sd_X^2 + sd_Y^2) / 2) with the sample sd" % r["z"], r["qualifies"]
          and abs(r["z"] - 3.87) < 0.01 and abs(r["sigma"] ** 2 - (0.0085 ** 2 + (sx ** 2 + sy ** 2) / 3)) < 1e-15
          and abs(r["pooled_sd"] - math.sqrt((sx ** 2 + sy ** 2) / 2)) < 1e-15 and r["outcome"] == "qualifies", r)
    big_sd = F.decide(0.01, 0.002, [0.80, 0.82, 0.84], [0.80, 0.81, 0.82])
    big_se = F.decide(0.01, 0.02, [0.8, 0.8001, 0.8002], [0.79, 0.7901, 0.7902])
    check("each condition alone fails: D above SE but not above 2 pooled sd; D above 2 pooled sd but not above SE",
          big_sd["conditions"] == {"above_2_sd": False, "above_se": True} and not big_sd["qualifies"]
          and big_se["conditions"] == {"above_2_sd": True, "above_se": False} and not big_se["qualifies"],
          (big_sd["conditions"], big_se["conditions"]))
    one = F.decide(0.004, 0.001, [0.889], [0.8850, 0.8853, 0.8874], one_system=True)
    sb = statistics.stdev([0.8850, 0.8853, 0.8874])
    check("P4's one system: D > 2 x sd(E3-B), the seed term 2 sd(E3-B)^2 / 3; a two-sided value list is refused",
          one["two_sd"] == 2 * sb and abs(one["seed_term"] - 2 * sb ** 2 / 3) < 1e-18 and one["pooled_sd"] is None
          and refused(F.decide, 0.004, 0.001, [0.88, 0.89, 0.9], [0.88, 0.88, 0.88], one_system=True) is not None
          and refused(F.decide, 0.004, 0.001, [0.88], [0.88, 0.88, 0.88]) is not None, one)
    tiny = F.decide(0.0012, 0.0004, [0.9, 0.9001, 0.9002], [0.8989, 0.8988, 0.899])
    eq = F.decide(0.0005, 0.0015, [0.9, 0.9001, 0.9002], [0.8995, 0.8996, 0.8997])
    inc = F.decide(0.003, 0.004, [0.9, 0.91, 0.92], [0.9, 0.9, 0.9])
    rev = F.decide(-0.03, 0.004, [0.8, 0.801, 0.802], [0.83, 0.831, 0.832])
    check("outcomes: qualifies with the flag below SESOI (interval inside (-0.005, 0.005)); equivalent; inconclusive; "
          "inconclusive reversed (the whole interval below 0)",
          tiny["outcome"] == "qualifies" and tiny["flags"] == ["below SESOI"]
          and eq["outcome"] == "equivalent" and eq["flags"] == []
          and inc["outcome"] == "inconclusive" and inc["flags"] == []
          and rev["outcome"] == "inconclusive" and rev["flags"] == ["reversed"],
          (tiny["outcome"], tiny["flags"], eq["outcome"], inc["outcome"], rev["outcome"], rev["flags"]))
    s = 0.005 / F.Z90
    edge = F.outcome(False, 0.0, s * 1.000001)
    inside = F.outcome(False, 0.0, s * 0.999999)
    check("the interval against (-0.005, 0.005): just outside is inconclusive, just inside equivalent",
          edge[0] == "inconclusive" and inside[0] == "equivalent", (edge, inside))
    check("a missing SE is refused", refused(F.decide, 0.01, None, x, y) is not None)


def u_holm():
    print("Holm per exam (decision 3)")
    z = {"P1": 3.06, "P2": 3.87, "P3": 2.92, "P4": 1.43, "P5": 2.97}
    pr = {p: {"status": "decided", "qualifies": True, "z": v} for p, v in z.items()}
    h = safe(F.holm_overlay, pr)
    h = h if isinstance(h, dict) else collections.defaultdict(lambda: {"family_confirmed": None, "bar": 0, "rank": 0})
    check("V3F.3.4's worked check (3.06, 3.87, 2.92, 1.43, 2.97 as one dev set): all confirmed but 1.43; bars "
          "2.326, 2.241, 2.128, 1.960, 1.645 in order of z",
          [p for p in ("P1", "P2", "P3", "P4", "P5") if h[p]["family_confirmed"]] == ["P1", "P2", "P3", "P5"]
          and [round(h[p]["bar"], 3) for p in ("P2", "P1", "P5", "P3", "P4")] == [2.326, 2.241, 2.128, 1.96, 1.645],
          {p: (h[p]["rank"], round(h[p]["bar"], 3)) for p in h})
    pr2 = {"P1": {"status": "decided", "qualifies": True, "z": 2.2},
           "P2": {"status": "decided", "qualifies": True, "z": 2.15},
           "P3": {"status": "decided", "qualifies": True, "z": 1.0},
           "P4": {"status": "decided", "qualifies": False, "z": 4.0}}
    h2 = safe(F.holm_overlay, pr2)
    h2 = h2 if isinstance(h2, dict) else collections.defaultdict(lambda: {"survives": None, "z": 0, "bar": 1})
    check("step-down stops at the first failure (2.2 < 2.241 at rank 2: 2.15 is not confirmed though above its own "
          "bar 2.128); a primary surviving Holm without qualifying is not confirmed; P5 not run keeps m = 5 and is "
          "never confirmed",
          h2["P4"]["survives"] and not h2["P4"]["family_confirmed"] and not h2["P1"]["survives"]
          and not h2["P2"]["survives"] and h2["P2"]["z"] > h2["P2"]["bar"] and not h2["P2"]["family_confirmed"]
          and h2["P5"]["status"] == "not run" and not h2["P5"]["family_confirmed"] and h2["P1"]["m"] == 5
          and h2["P5"]["rank"] == 5, h2)
    h3 = safe(F.holm_overlay, {"P8": {"status": "decided", "qualifies": True, "z": 1.8},
                         "P9": {"status": "decided", "qualifies": True, "z": 2.0},
                         "P6": {"status": "decided", "qualifies": True, "z": 1.7},
                         "P7": {"status": "failed", "qualifies": True, "z": 9.0}})
    h3 = h3 if isinstance(h3, dict) else collections.defaultdict(lambda: {"family_confirmed": None, "bar": 0, "m": 0})
    check("probe m = 2 (1.960 then 1.645): 2.0 then 1.8 both confirmed; ood22 m = 1 (1.645); a failed P7 is never "
          "confirmed whatever its z", h3["P9"]["family_confirmed"] and h3["P8"]["family_confirmed"]
          and round(h3["P9"]["bar"], 3) == 1.96 and h3["P6"]["family_confirmed"] and h3["P6"]["m"] == 1
          and not h3["P7"]["family_confirmed"], h3)


def u_gate_and_reads():
    print("Tier C (C = 14) and the sealed-test reads")
    g = F.challenger_gate({"D2CLIP": {"status": "decided", "qualifies": True, "z": 2.70, "diff": 0.004},
                           "D4F": {"status": "decided", "qualifies": True, "z": 2.68, "diff": 0.009},
                           "D6": {"status": "decided", "qualifies": False, "z": 3.5, "diff": 0.02},
                           "D3": {"status": "not run", "qualifies": None, "z": None, "diff": None}})
    bar13 = statistics.NormalDist().inv_cdf(1 - 0.05 / 13)
    check("C stays 14 with 3 scored and 11 not run; z 2.70 opens, 2.68 does not (it would under 0.05/13 = %.4f); a "
          "non-qualifying arm never opens" % bar13,
          g["C"] == 14 and g["bar"] == F.CHALLENGER_BAR and g["opening"] == ["D2CLIP"]
          and 2.68 > bar13 and g["arms"]["D2LAION"]["status"] == "not run" and not g["arms"]["D6"]["opens_read"], g)
    check("an unregistered challenger is refused", refused(F.challenger_gate, {"D2X": {}}) is not None)
    four = F.challenger_gate({a: {"status": "decided", "qualifies": True, "z": 5.0, "diff": d} for a, d in
                              (("D2CLIP", 0.01), ("D4F", 0.03), ("D3", 0.02), ("D4N", 0.005))})
    o = F.read_order(four)
    ob = F.read_order(four, bclean_triggered=True)
    check("read order: D6, D8-m, then the challengers largest D first, at most 3; with BCLEAN: BCLEAN first and 2 "
          "challengers; never more than 5",
          [x["arm"] for x in o] == ["D6", "D8M", "D4F", "D3", "D2CLIP"]
          and [x["arm"] for x in ob] == ["BCLEAN", "D6", "D8M", "D4F", "D3"], (o, ob))
    no6 = F.read_order(four, reference_available=("D8M",))
    check("  D6 not run: its reference read is not in the order", [x["arm"] for x in no6][:2] == ["D8M", "D4F"])
    with6 = F.challenger_gate({a: {"status": "decided", "qualifies": True, "z": 5.0, "diff": d} for a, d in
                               (("D6", 0.05), ("D4F", 0.03), ("D3", 0.02), ("D2CLIP", 0.01))})
    o6 = F.read_order(with6)
    o6b = F.read_order(with6, bclean_triggered=True)
    check("  D6 opening as a challenger takes no challenger place (its read is the reference read): D6, D8-m, then "
          "three other challengers; with BCLEAN two; no arm twice; the third challenger's read is allowed",
          with6["opening"][0] == "D6" and [x["arm"] for x in o6] == ["D6", "D8M", "D4F", "D3", "D2CLIP"]
          and [x["kind"] for x in o6] == ["reference", "reference", "challenger", "challenger", "challenger"]
          and [x["arm"] for x in o6b] == ["BCLEAN", "D6", "D8M", "D4F", "D3"]
          and F.check_read("D2CLIP", o6, {"D6": "read", "D8M": "read", "D4F": "read", "D3": "read"})["position"]
          == 5 and F.check_read("D4F", o6, {"D6": "read", "D8M": "read"})["position"] == 3, (o6, o6b))
    o6n = F.read_order(with6, reference_available=("D8M",))
    check("  and when D6's reference read is not in the order, D6 still takes no challenger place",
          [x["arm"] for x in o6n] == ["D8M", "D4F", "D3", "D2CLIP"], o6n)
    out = {"a challenger before the references": refused(F.check_read, "D4F", o, {}),
           "an arm that opens no read": refused(F.check_read, "D4N", o, {"D6": "read", "D8M": "read"}),
           "a second read": refused(F.check_read, "D6", o, {"D6": "read"})}
    ok = F.check_read("D4F", o, {"D6": "read", "D8M": F.NOT_RUN_CAP})
    check("check_read refuses %s; a read follows earlier reads taken or recorded not run" % ", ".join(out),
          all(e is not None for e in out.values()) and ok["position"] == 3 and ok["kind"] == "challenger", out)
    check("read_by: exactly 'Harry He (owner)' or 'Claude (delegate under the 2026-09-30 owner grant)'; anything "
          "else refused", F.read_by_value("owner") == "Harry He (owner)"
          and F.read_by_value("delegate") == "Claude (delegate under the 2026-09-30 owner grant)"
          and refused(F.read_by_value, "Harry") is not None and refused(F.read_by_value, None) is not None)


def u_cap():
    print("the family cap of 260 GPU-h (decision 25)")
    shutil.rmtree(str(F.cap_dir()), ignore_errors=True)
    items = [{"item": "rest", "state": "ended", "estimate": 130.0, "settled": 128.4},
             {"item": "D6", "state": "running", "estimate": 108.0, "settled": None},
             {"item": "BCLEAN", "state": "ended", "estimate": 15.0, "settled": 14.6},
             {"item": "D4F", "state": "ended", "estimate": 7.4, "settled": None}]
    base = F.family_gpuh(items)
    check("running items count at their estimate, ended ones at their settled GPU-h (or estimate until settled): "
          "%.1f" % base, abs(base - 258.4) < 1e-9, base)
    reads = items + [{"item": "read_%s" % a, "state": "ended", "estimate": 0.5, "settled": 0.5}
                     for a in ("BCLEAN", "D6", "D8M")]
    e = refused(F.check_cap, reads, "read_D4F", 0.5)
    rec = json.loads((F.cap_dir() / "read_D4F.json").read_text()) if (F.cap_dir() / "read_D4F.json").exists() else {}
    ok = F.check_cap(reads, "reanalysis", 0.1)
    check("V3F.7's corner: at 259.9 a challenger read (0.5) is refused and recorded 'not run (family cap)'; the "
          "re-analysis (0.1) still fits at 260.0", isinstance(e, F.V3famCapRefused) and rec.get("status")
          == F.NOT_RUN_CAP and abs(ok["after_gpuh"] - 260.0) < 1e-9, (e, rec, ok))
    e2 = refused(F.check_cap, items[:1], "read_D4F", 0.01)
    check("  a refused item is never retried, even when it would now fit", isinstance(e2, F.V3famCapRefused)
          and "never retried" in str(e2), e2)
    shutil.rmtree(str(F.cap_dir()), ignore_errors=True)
    tenths = [{"item": "base", "state": "ended", "estimate": 259.7, "settled": None}] + [
        {"item": "r%d" % i, "state": "ended", "estimate": 0.1, "settled": None} for i in range(2)]
    fits = safe(F.check_cap, tenths, "last_tenth", 0.1, record=False)
    check("  an item that fits to the decimal (259.7 + 0.1 + 0.1, then 0.1: 260.00000000000006 in floats) is not "
          "refused for float noise", isinstance(fits, dict) and abs(fits["after_gpuh"] - 260.0) < 1e-9, fits)
    run = [{"item": "x", "state": "running", "estimate": 259.0, "settled": None}]
    e3 = refused(F.check_cap, run, "y", 2.0, record=False)
    check("  a running item's estimate counts (259 running + 2 refused); an unknown state is refused",
          isinstance(e3, F.V3famCapRefused) and refused(F.family_gpuh, [{"item": "z", "state": "odd"}]) is not None)
    shutil.rmtree(str(F.cap_dir()), ignore_errors=True)


def u_p7_hand():
    print("P7: target-species agnostic AP with OtherPlant boxes as ignore regions, by hand (V3F.6.5)")
    import torch
    # GT: targets T1 [10,10,50,50] (3), T2 [100,100,140,140] (0), T3 [400,400,440,440] (5);
    #     OtherPlant O1 [200,200,240,240], O2 [400,400,440,444] (IoU with T3 0.909)
    gt = torch.tensor([[10, 10, 50, 50], [100, 100, 140, 140], [400, 400, 440, 440], [200, 200, 240, 240],
                       [400, 400, 440, 444]], dtype=torch.float32)
    gc = torch.tensor([3, 0, 5, 12, 12], dtype=torch.float32)
    # p0 on T1 (IoU 1), p1 on T2 (IoU 1600/2080 = 0.769), p2 on O1 (IoU 1, no target), p3 alone, p4 on T3 (IoU 1;
    # 1600/1760 = 0.909 with O2); no IoU sits on the threshold grid 0.50:0.05:0.95
    boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 140, 152], [200, 200, 240, 240], [300, 300, 340, 340],
                          [400, 400, 440, 440]], dtype=torch.float32)
    conf = np.array([0.9, 0.8, 0.95, 0.7, 0.5])
    r = F.ignore_region_matches(gt, gc, boxes, F.default_match())
    tp_t = np.zeros((5, 10), bool)
    tp_t[0], tp_t[4] = True, True
    tp_t[1, :6] = True                           # IoU 0.769: thresholds 0.50..0.75
    check("target reading by hand: TP rows p0 (all 10), p1 (0.50-0.75), p4 (all 10); p2 (on an OtherPlant box, no "
          "target) ignored; p3 a kept false positive; p4 not ignored though it overlaps O2 (a TP at 0.50); 3 target "
          "boxes", np.array_equal(r["tp_t"], tp_t) and r["ign_t"].tolist() == [False, False, True, False, False]
          and r["n_t"] == 3, (r["tp_t"].astype(int), r["ign_t"]))
    tp_o = np.zeros((5, 10), bool)
    tp_o[2] = True
    tp_o[4, :9] = True                           # p4 against O2: IoU 0.909, thresholds 0.50..0.90
    check("the symmetric OtherPlant reading by hand: TPs p2 (all), p4 (0.50-0.90); p0 and p1 (on targets) ignored; "
          "2 OtherPlant boxes", np.array_equal(r["tp_o"], tp_o)
          and r["ign_o"].tolist() == [True, True, False, False, False] and r["n_o"] == 2,
          (r["tp_o"].astype(int), r["ign_o"]))
    v = F.p7_value(r["tp_t"], conf, r["ign_t"], r["n_t"])
    keep = [0, 1, 3, 4]
    hand = S.collapsed_ap(tp_t[keep], conf[keep], 3)
    check("AP50-95 = collapsed_ap on the kept rows (p0, p1, p3, p4) with n_gt = 3 (%.6f)" % v[0], v == hand, (v, hand))
    gt2 = torch.tensor([[10, 10, 50, 50], [200, 200, 240, 240]], dtype=torch.float32)
    gc2 = torch.tensor([1, 12], dtype=torch.float32)
    b2 = torch.tensor([[200, 200, 240, 240], [10, 10, 50, 50]], dtype=torch.float32)
    r2 = F.ignore_region_matches(gt2, gc2, b2, F.default_match())
    with_ign = F.p7_value(r2["tp_t"], [0.95, 0.9], r2["ign_t"], 1)[0]
    without = F.p7_value(r2["tp_t"], [0.95, 0.9], np.zeros(2, bool), 1)[0]
    # by hand (Ultralytics' 101-point interpolation): one TP ranked first, recall 1 at precision 1 -> 0.99 + 0.01/2
    check("a second case by hand: a more confident box on an OtherPlant plant is ignored, so the one true positive "
          "ranks first and AP50-95 = 0.995 (Ultralytics' interpolation); without the ignore rule it is %.4f" % without,
          abs(with_ign - 0.995) < 1e-9 and without < 0.6, (with_ign, without))
    e = F.ignore_region_matches(gt2[:0], gc2[:0], b2, F.default_match())
    check("  an image without GT: no TP, nothing ignored, n 0", not e["tp_t"].any() and not e["ign_t"].any()
          and e["n_t"] == 0 and e["n_o"] == 0)


def u_subsets_and_compare():
    print("subsets, the stage-1 comparison, recorded-score comparisons, reproduction branches")
    a = synth(7)
    keep = [str(k) for k in a["keys"][::3]]
    s = F.subset_arrays(a, keep)
    built = {"keys": np.asarray(keep)}
    idx = [int(k[1:]) for k in keep]
    p = np.isin(a["pred_img"], idx)
    t = np.isin(a["target_img"], idx)
    remap = {old: new for new, old in enumerate(idx)}
    built.update(tp=a["tp"][p], conf=a["conf"][p], pred_cls=a["pred_cls"][p],
                 pred_img=np.asarray([remap[i] for i in a["pred_img"][p]]), target_cls=a["target_cls"][t],
                 target_img=np.asarray([remap[i] for i in a["target_img"][t]]))
    check("subset_arrays keeps the images' rows, re-indexed; its statistic equals the hand-built subset's",
          all(np.array_equal(s[k], built[k]) for k in built) and F.statistic_value(s) == F.statistic_value(built))
    check("  a key that is not an exam image is refused", refused(F.subset_arrays, a, ["nope"]) is not None)
    st = {"keys": np.asarray(["a", "b"]), "img": np.asarray([0, 0, 1]),
          "xyxy": np.asarray([[1.0, 2, 3, 4], [5, 6, 7, 8], [0, 0, 9, 9]]), "q": np.asarray([0.9, 0.5, 0.3])}
    near = dict(st, xyxy=st["xyxy"] + 0.0009, q=st["q"] + 0.00009)
    ok = F.stage1_compare(near, st)
    out = {"xyxy 0.002 px off": refused(F.stage1_compare, dict(st, xyxy=st["xyxy"] + 0.002), st),
           "q 0.0002 off": refused(F.stage1_compare, dict(st, q=st["q"] + 0.0002), st),
           "a row more": refused(F.stage1_compare, dict(st, img=np.asarray([0, 1, 1])), st),
           "another key order": refused(F.stage1_compare, dict(st, keys=np.asarray(["b", "a"])), st)}
    check("stage-1 boxes within 1e-3 px and q within 1e-4 reproduce; refused: %s" % ", ".join(out),
          ok["rows"] == 3 and all(e is not None for e in out.values()), out)
    rec = {"exam": "dev", "manifest_sha256": "m", "key_order_sha256": "k", "n_images": 3, "weights_sha256": "w",
           "n_gt": {"Ragweed": 2}, "scorer_sha256": "TEST-abc", "map50_95": 0.5, "species_map50_95": 0.4,
           "per_class": {"Ragweed": 0.4, "OtherPlant": 0.6}, "agnostic_map50_95": 0.7}
    new = dict(rec, map50_95=0.5000005, per_class={"Ragweed": 0.401, "OtherPlant": 0.6})
    c = F.compare_with_recorded(new, rec, "x")
    e3doc = dict(rec, scorer_sha256="TEST-E3-abc", locked_scorer_sha256="abc")
    check("compare_with_recorded: mAP, species mean and every class (worst 0.001 here); an E3-stamped record's locked "
          "scorer matches", abs(c["max_abs_diff"] - 0.001) < 1e-12 and abs(c["map50_95"] - 5e-7) < 1e-12
          and F.compare_with_recorded(rec, e3doc, "y")["max_abs_diff"] == 0, c)
    bad = {"other weights": dict(rec, weights_sha256="v"), "another scorer": dict(rec, scorer_sha256="def"),
           "other classes": dict(rec, per_class={"Ragweed": 0.4}), "other GT": dict(rec, n_gt={"Ragweed": 3})}
    out = {k: refused(F.compare_with_recorded, v, rec, "x") for k, v in bad.items()}
    check("  refused: %s" % ", ".join(out), all(e is not None for e in out.values()), out)
    check("reproduction branches: <= 1e-6 passed, <= 0.002 deviation, beyond or NaN refused",
          [F.repro_status(x) for x in (0.0, 1e-6, 2e-6, 0.002, 0.0021, float("nan"), None)]
          == ["passed", "passed", "deviation", "deviation", "refused", "refused", "refused"])
    check("  a deviation continues; a refusal raises", F._status_of([0.0, 0.001], "x") == "deviation"
          and F._status_of([0.0, 1e-7], "x") == "passed" and refused(F._status_of, [0.0, 0.003], "x") is not None)
    rows = {"keys": np.asarray(["a", "b", "c"]), "img": np.asarray([0, 0, 2]),
            "xyxy": np.arange(12.0).reshape(3, 4), "conf": np.asarray([0.9, 0.8, 0.7]), "cls": np.asarray([1, 2, 3.0])}
    inj = F.inject_from_rows(rows)
    check("inject_from_rows: every key, possibly with no row, each image's rows in order; rows not grouped by "
          "image are refused", sorted(inj) == ["a", "b", "c"] and inj["a"]["conf"].tolist() == [0.9, 0.8]
          and len(inj["b"]["conf"]) == 0 and inj["c"]["cls"].tolist() == [3.0]
          and refused(F.inject_from_rows, dict(rows, img=np.asarray([0, 2, 0]))) is not None)
    rp = ((0.5, 0.25), (3.0, 7.0))
    o = np.asarray([[10.0, 20.0, 150.0, 90.0], [-4.0, 0.0, 210.0, 101.0]])
    check("to_letterbox and from_letterbox are exact inverses (unclipped), and from_letterbox agrees with E3's "
          "to_original inside the image", np.allclose(F.from_letterbox(F.to_letterbox(o, rp), rp), o, atol=1e-12)
          and np.allclose(F.from_letterbox(F.to_letterbox(o[:1], rp), rp), TS.to_original(F.to_letterbox(o[:1], rp),
                                                                                         rp, (100, 200))))


def u_write_once_and_trees():
    print("write once; the E-series trees (V3F.3.7)")
    p = F.root() / "unit" / "x.json"
    shutil.rmtree(str(p.parent), ignore_errors=True)
    F.write_once(p, {"a": 1})
    e = refused(F.write_once, p, {"a": 2})
    check("a record is written once; a second write is refused and leaves it", e is not None
          and json.loads(p.read_text()) == {"a": 1}, e)
    shutil.rmtree(str(p.parent), ignore_errors=True)
    tw = C.INC_DIR / "twostage" / TS.E3_NAME
    cap = C.INC_DIR / "capacity"
    made = not tw.exists()
    if made:
        (tw / "arms").mkdir(parents=True, exist_ok=True)
        (tw / "arms" / "x.json").write_text("{}")
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "e3_unit.json").write_text("{}")
    try:
        with F.e_trees_unchanged() as before:
            (tw / "sensitivity.json").write_text("{}")
            (cap / "v3fam_unit.json").write_text("{}")
        ok1 = "capacity/e3_unit.json" in before

        def touch_e3():
            with F.e_trees_unchanged():
                (cap / "e3_unit.json").write_text('{"x": 1}')

        def add_e2():
            with F.e_trees_unchanged():
                (cap / "e2_unit.json").write_text("{}")
        e1, e2 = refused(touch_e3), refused(add_e2)
        check("D9a's sensitivity.json and capacity/v3fam_* may change; a changed capacity/e3_* or a new e2_* fails "
              "the job", ok1 and e1 is not None and e2 is not None, (e1, e2))
    finally:
        for q in ("e3_unit.json", "e2_unit.json", "v3fam_unit.json"):
            if (cap / q).exists():
                (cap / q).unlink()
        if (tw / "sensitivity.json").exists():
            (tw / "sensitivity.json").unlink()
        if made:
            shutil.rmtree(str(tw.parent))


def tagged_jpeg(path, tag, size=(60, 40), seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    a[: size[1] // 4, : size[0] // 3] = 255
    im = Image.fromarray(a)
    ex = im.getexif()
    if tag is not None:
        ex[0x0112] = tag
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95, exif=ex)
    return path


def u_exif():
    print("the EXIF frame pre-flight (V3F.3.2)")
    import cv2
    import ultralytics.utils.patches as UP
    d = W.TMP / "exif"
    imgs = {t: tagged_jpeg(d / ("t%s.jpg" % t), t) for t in (3, 6, 8, 0)}
    got = {t: F.exif_frame_check(p) for t, p in imgs.items()}
    check("tags 3, 6, 8 and 0: Ultralytics' loader frame equals exif_transpose's (mean abs diff %s)"
          % {t: g.get("mad") for t, g in got.items()}, all(g["ok"] for g in got.values())
          and got[6]["loader_shape"] == [60, 40], got)
    imread0 = UP.imread

    def ignoring(f, flags=cv2.IMREAD_COLOR):
        return imread0(f, flags=flags | cv2.IMREAD_IGNORE_ORIENTATION)
    with patched(UP, "imread", ignoring):
        bad6, bad3 = F.exif_frame_check(imgs[6]), F.exif_frame_check(imgs[3])
    check("a loader that does not turn the image fails: tag 6 by its shape, tag 3 by its pixels",
          not bad6["ok"] and bad6["mad"] is None and not bad3["ok"] and bad3["mad"] > F.EXIF_MAD_TOL, (bad6, bad3))
    rows = []
    for i, t in enumerate((None, 3, 6, 8, 0, 1)):
        p = tagged_jpeg(d / ("r%d.jpg" % i), t, seed=i)
        rows.append(W.row_for("exif_%d" % i, p, [(1, 0.3, 0.3, 0.2, 0.2)], "tsw23"))
    shutil.rmtree(str(F.exif_dir()), ignore_errors=True)
    with patched(UP, "imread", ignoring):
        pre_bad = F.exif_preflight(rows=rows)
    F.exif_verdict("yes", "delegate")
    failed_numeric = F.exif_preflight_passed()
    shutil.rmtree(str(F.exif_dir()))
    pre = F.exif_preflight(rows=rows)
    before = F.exif_preflight_passed()
    e = refused(F.exif_preflight, rows=rows)
    ev = refused(F.exif_verdict, "maybe", "delegate")
    ev2 = refused(F.exif_verdict, "yes", "someone")
    v = F.exif_verdict("yes", "delegate")
    after = F.exif_preflight_passed()
    e3 = refused(F.exif_verdict, "no", "owner")
    check("the pre-flight picks one copy of each of tags 3, 6, 8 and 0, writes its record and montage once, and "
          "passes only after a named viewer's yes naming it; a numeric failure never passes; the verdict is written "
          "once", pre["numeric_passed"] and sorted(pre["checks"]) == ["0", "3", "6", "8"]
          and not pre_bad["numeric_passed"] and not failed_numeric and not before and after
          and v["read_by"] == F.READ_BY["delegate"] and (F.exif_dir() / "montage.jpg").is_file()
          and all(x is not None for x in (e, ev, ev2, e3)), (pre["checks"], e, ev, ev2, e3))
    check("orientations_for: tags 3, 6, 8 on ood22 and ood23 after the pre-flight only",
          F.orientations_for("ood22") == TS.FIT_ORIENTATIONS and F.orientations_for("dev") == TS.OK_ORIENTATIONS
          and F.orientations_for("imageweeds") == TS.OK_ORIENTATIONS)
    with moved(F.exif_dir()):
        check("  without the pre-flight, ood22 reads none, 0 and 1", F.orientations_for("ood22") == TS.OK_ORIENTATIONS)
    rows_missing = [r for r in rows if TS.header(r["image"])[0] != 8]
    with moved(F.exif_dir()):
        e4 = refused(F.exif_preflight, rows=rows_missing)
    check("  a base_v2 without a tag-8 copy is refused", e4 is not None and "8" in str(e4), e4)


def u_allow_lists():
    print("the exam allow-lists and the test token (V3F.3.1, V3F.3.6)")
    out = {"D5B on dev": ("D5B", "dev"), "D2LAION on imageweeds": ("D2LAION", "imageweeds"),
           "D4F on ood22": ("D4F", "ood22"), "E3RS on ood23": ("E3RS", "ood23"), "D3 on the probe": ("D3", "probe"),
           "D5B on ooddev_v1": ("D5B", "ooddev_v1"), "D9 on imageweeds": ("D9", "imageweeds"),
           "an unknown exam": ("E3RS", "train_core"), "an unknown arm": ("D7", "dev"), "test without a token": ("D6",
                                                                                                         "test")}
    got = {k: refused(F.check_exam_allowed, *v) for k, v in out.items()}
    with F.test_read_token() as tok:
        ok_t = F.check_exam_allowed("D6", "test", tok)
        foreign = refused(F.check_exam_allowed, "D6", "test", object())
    check("refused: %s; test with score-test's token only (a foreign token refused, the token gone after)"
          % ", ".join(got), all(e is not None for e in got.values()) and ok_t and foreign is not None
          and F._TEST_TOKENS == [], {k: str(v)[:80] for k, v in got.items()})
    check("allowed: E3RS dev, imageweeds, ooddev_v1; D1 ood22/ood23/dev; D5B probe; D2 arms dev",
          all(F.check_exam_allowed(a, e) for a, e in (("E3RS", "dev"), ("E3RS", "imageweeds"), ("E3RS", "ooddev_v1"),
                                                      ("D1", "ood22"), ("D1", "ood23"), ("D1", "dev"), ("D5B", "probe"),
                                                      ("D2BC2640", "dev"), ("D4R15", "ooddev_v1"))))
    check("roots: ooddev_v1 in the zoo's root, the probe in v3fam/v1/probe_root",
          F.root_of("ooddev_v1") == C.INC_DIR / "_zoo" / "v1" / "root"
          and F.root_of("probe") == C.INC_DIR / "v3fam" / "v1" / "probe_root" and F.root_of("dev") == C.INC_DIR)


UNITS = ("u_constants", "u_frozen", "u_sessions", "u_bootstrap_exact", "u_session_bootstrap", "u_selection", "u_rule",
         "u_holm", "u_gate_and_reads", "u_cap", "u_p7_hand", "u_subsets_and_compare", "u_write_once_and_trees",
         "u_allow_lists", "u_exif")


# ================================================================== WORLD
def ident_cfg(**over):
    return dict({"mode": "identity", "nms": "locked"}, **over)


def ts_cfg(**over):
    c = E3T.two_stage_cfg()
    c.update(over)
    return c


def w_setup():
    print("the world: E3's fixtures, E3's own records (score-arm M, A, B), ood exams with OtherPlant boxes, a probe "
          "root")
    E3T.setup_world()
    t0 = time.time()
    for arm in ("M", "A", "B"):
        TS.score_arm(arm, embedder=EMB, reported=False, **KW)
    check("E3's records exist (classifier pinned, equivalence, dev files of M, A, B) (%.0fs)" % (time.time() - t0),
          all((TS.arm_dir(a, s) / "dev.json").exists() for a in "MAB" for s in (0, 1, 2)))
    # ood22 as built; ood23 rebuilt with OtherPlant boxes on and beside target boxes (the LOCK updated)
    lock = json.loads(C.LOCK_PATH.read_text())
    rows23 = []
    for i in range(4):
        img = W.TMP / "src" / "o23x" / ("ood23x_%03d.jpg" % i)
        W.make_image(img, 900 + i)
        boxes = [(3, 0.3, 0.3, 0.3, 0.3), (12, 0.32, 0.3, 0.3, 0.34), (12, 0.75, 0.75, 0.25, 0.25),
                 (i % 12, 0.7, 0.25, 0.2, 0.2)]
        rows23.append(W.row_for("ood23x_%03d" % i, img, boxes, "ood23"))
    lock["manifests"]["ood23"] = C.write_manifest(C.manifest_path("ood23"), rows23)
    C.LOCK_PATH.write_text(json.dumps(lock))
    for ex, rows in (("ood22", WORLD["rows"]["ood22"]), ("ood23", rows23)):
        if (C.EXAMS_DIR / ex).exists():
            shutil.rmtree(str(C.EXAMS_DIR / ex))
        C.materialise(rows, C.EXAMS_DIR / ex)
    pr = F.root_of("probe")
    prow = WORLD["rows"]["train_core"][:5]
    (pr / "splits" / "v1").mkdir(parents=True, exist_ok=True)
    psha = C.write_manifest(pr / "splits" / "v1" / "probe.jsonl", prow)
    (pr / "splits" / "v1" / "LOCK.json").write_text(json.dumps({"manifests": {"probe": psha}, "scorer_sha256":
                                                                W.sha(pathlib.Path(S.__file__).resolve())}))
    C.materialise(prow, pr / "exams" / "v1" / "probe")
    check("ood22 and ood23 materialised under LOCK v1; the probe root holds its exam and its LOCK",
          (C.EXAMS_DIR / "ood23" / "data.yaml").is_file() and (pr / "exams" / "v1" / "probe" / "data.yaml").is_file())


def w_layering_identity():
    print("the family validator over E3's; identity equal to the locked scorer exactly")
    base = S.validator_class()
    cls, b2 = F.family_validator_class(ident_cfg())
    fam = [k for k in cls.__mro__ if k.__dict__.get("_v3fam_layer")]
    e3 = [k for k in cls.__mro__ if k.__dict__.get("_e3_layer")]
    check("one family layer over exactly one E3 layer over the sidecar's validator over the scorer's own",
          len(fam) == 1 and len(e3) == 1 and b2 is base and cls.__mro__.count(base) == 1
          and cls.__mro__[1] is e3[0] and cls.__mro__[2].__mro__[1] is base, cls.__mro__[:4])
    w = E3T.src_weights("M", 0)
    ref = S.score(w, "dev", W.TMP / "v3f_ref.json", **KW)
    ps = F._family_pass(w, "dev", ident_cfg(), "E3RS", **KW)
    e3p = TS._score_pass(w, "dev", {"mode": "identity", "nms": "locked", "workers": V._Workers(1)}, **KW)
    r = ps["result"]
    same = all(r[k] == ref[k] == e3p["result"][k] for k in ("map50_95", "map50", "per_class", "species_map50_95",
                                                            "agnostic_map50_95", "image_correct", "n_gt"))
    check("identity: the locked scorer's score exactly (mAP %.4f, species %.4f) and E3's identity pass exactly; the "
          "stage-1 helper the scorer's agnostic AP exactly; the validator restored" % (r["map50_95"],
                                                                                       r["species_map50_95"] or 0),
          same and ps["stage1_agnostic"][0] == r["agnostic_map50_95"] and S._VALIDATOR is base
          and SC.MAX_RECOMPUTE_DIFF >= ps["stats"]["full_recompute_max_abs_diff"], (r["map50_95"], ref["map50_95"]))
    n_dev = len(C.read_manifest(C.manifest_path("dev")))
    check("  every image's EXIF tag read by the family (none here), the rows recorded in original pixels",
          ps["stats"]["orientations"] == {"None": n_dev} and ps["rows"] is not None
          and len(ps["rows"]["conf"]) == len(ps["arrays"]["conf"]), ps["stats"]["orientations"])
    e = refused(F._family_pass, w, "dev", {"mode": "nonsense"}, "E3RS", **KW)
    check("  a refused pass leaves the scorer's validator as it was", e is not None and S._VALIDATOR is base)
    WORLD["ident_M0"] = ps


def w_modes():
    print("two_stage, stage1, inject_rows, inject_boxes, oracle, view640")
    w = E3T.src_weights("A", 0)
    e3p = TS._score_pass(w, "dev", ts_cfg(), **KW)
    fp = F._family_pass(w, "dev", ts_cfg(), "E3RS", **KW)
    same = all(fp["result"][k] == e3p["result"][k] for k in ("map50_95", "per_class", "species_map50_95",
                                                             "agnostic_map50_95"))
    s1same = all(np.array_equal(fp["stage1"][k], e3p["stage1"][k]) for k in e3p["stage1"])
    check("two_stage: E3's own pass exactly (species %.4f), its stage-1 record equal field by field"
          % fp["result"]["species_map50_95"], same and s1same)
    WORLD["ts_A0"] = fp
    ident = F._family_pass(w, "dev", ident_cfg(), "E3RS", **KW)
    inj = F.inject_from_rows(ident["rows"])
    ir = F._family_pass(w, "dev", ident_cfg(mode="inject_rows", inject=inj), "D6", **KW)
    check("inject_rows: the identity pass's own rows in original pixels, injected, reproduce it exactly; the GT round "
          "trip checked (%.2g)" % ir["stats"]["gt_roundtrip_max_diff"],
          all(ir["result"][k] == ident["result"][k] for k in ("map50_95", "per_class", "agnostic_map50_95",
                                                               "image_correct"))
          and ir["stats"]["injected_rows"] == len(ident["rows"]["conf"]) and ir["stats"]["gt_roundtrip_max_diff"] <= 1e-3)
    boxes = {k: {"xyxy": v["xyxy"], "conf": v["conf"]} for k, v in inj.items()}
    ib = F._family_pass(w, "dev", ts_cfg(mode="inject_boxes", inject=boxes), "D3", **KW)
    check("inject_boxes: the detector's own boxes injected reproduce its two-stage pass exactly (one_per_box, E3's "
          "naming and emission)", all(ib["result"][k] == fp["result"][k] for k in ("map50_95", "per_class",
                                                                                   "species_map50_95"))
          and np.array_equal(ib["stage1"]["q"], fp["stage1"]["q"]))
    t0_ = TS.to_original
    with patched(TS, "to_original", lambda xy, rp, os_: t0_(xy, rp, os_) + 2.0):
        g1 = refused(F._family_pass, w, "dev", ident_cfg(mode="stage1"), "D4F", **KW)
        g2 = refused(F._family_pass, w, "dev", ident_cfg(mode="inject_rows", inject=inj), "D6", **KW)
    check("a map to the original image 2 px off refuses a stage1 and an inject_rows pass (E3's ground-truth round "
          "trip)", g1 is not None and "geometry" in str(g1) and g2 is not None and "geometry" in str(g2), (g1, g2))
    s1 = F._family_pass(w, "ood23", ident_cfg(mode="stage1"), "D1", **KW)
    p7 = s1["p7_full"]
    st = s1["p7"]["t"]
    tie = F.statistic_value(st, "agnostic")
    check("stage1 on ood23: the box set's rows and P7's arrays; %d target rows ignored; the full value in the "
          "validator's order (%.4f), the tie-broken arrays' within 0.01 (%.4f); the target GT count"
          % (p7["t"]["ignored_rows"], p7["t"]["ignore"][0], tie),
          s1["stage1"] is not None and p7["t"]["ignored_rows"] > 0 and p7["t"]["n_gt"] == 8 and p7["o"]["n_gt"] == 8
          and abs(tie - p7["t"]["ignore"][0]) < 0.01 and int(st["n_gt"].sum()) == 8 and set(s1["p7"]) == {"t", "o"},
          p7)
    WORLD["stage1_ood23"] = s1
    orc = F._family_pass(w, "dev", ts_cfg(mode="oracle"), "E3RS", **KW)
    og = orc["stage1"]["oracle_gt"]
    hit = og >= 0
    check("oracle: %d of %d stage-1 boxes named by their GT class (p 1, emitted once at q), the others E3's top 3"
          % (int(hit.sum()), len(og)), hit.any() and (orc["stage1"]["top_cls"][hit, 0] == og[hit]).all()
          and np.allclose(orc["stage1"]["top_p"][hit, 0], 1.0) and np.allclose(orc["stage1"]["top_p"][hit, 1:], 0.0)
          and orc["stats"]["oracle_named"] == int(hit.sum()),
          (orc["result"]["species_map50_95"], fp["result"]["species_map50_95"]))
    dg = json.loads((TS.classifier_dir() / "dev_gt.json").read_text())
    g = F._family_pass(E3T.src_weights("M", 0), "dev", ts_cfg(mode="gt"), "E3RS", **KW)
    top1 = sum(1 for t, p1, _ in g["gt"] if p1 == t) / float(len(g["gt"]))
    check("gt: the classifier on dev's ground-truth crops through the family pass, E3's dev_gt.json exactly (top-1 "
          "%.4f on %d crops), the predictions emitted unchanged" % (top1, len(g["gt"])),
          len(g["gt"]) == dg["n"] and top1 == dg["top1"] and g["stage1"] is None
          and g["result"]["map50_95"] == WORLD["ident_M0"]["result"]["map50_95"], (top1, dg["top1"]))
    CUT_CALLS.clear()
    v6 = F._family_pass(w, "dev", ts_cfg(mode="view640", cut=counting_cut), "D2BC2640", **KW)
    check("view640: the arm's cut function cuts every image's crops (here E3's own cut: the two-stage pass exactly)",
          len(CUT_CALLS) == len(C.read_manifest(C.manifest_path("dev")))
          and all(v6["result"][k] == fp["result"][k] for k in ("per_class", "species_map50_95")), len(CUT_CALLS))
    out = {"view640 without a cut": refused(F._family_pass, w, "dev", ts_cfg(mode="view640"), "D2BC2640", **KW),
           "two_stage without the embedder": refused(F._family_pass, w, "dev", ts_cfg(embedder=None), "E3RS", **KW),
           "inject_rows without rows": refused(F._family_pass, w, "dev", ident_cfg(mode="inject_rows"), "D6", **KW),
           "inject_rows missing an image": refused(F._family_pass, w, "dev", ident_cfg(
               mode="inject_rows", inject={k: v for k, v in list(inj.items())[1:]}), "D6", **KW),
           "inject_rows naming an image outside the exam": refused(F._family_pass, w, "dev", ident_cfg(
               mode="inject_rows", inject=dict(inj, not_an_exam_image=next(iter(inj.values())))), "D6", **KW),
           "injected rows of class 13": refused(F._family_pass, w, "dev", ident_cfg(
               mode="inject_rows", inject={k: dict(v, cls=np.full(len(v["conf"]), 13.0)) for k, v in inj.items()}),
               "D6", **KW),
           "an unknown mode": refused(F._family_pass, w, "dev", {"mode": "x"}, "E3RS", **KW)}
    check("refused: %s" % ", ".join(out), all(e is not None for e in out.values()) and S._VALIDATOR is
          S.validator_class(), {k: str(v)[:100] for k, v in out.items()})


CUT_CALLS = []


def counting_cut(task):
    """A view640 cut for the test: E3's own cut, counted (in-process workers)."""
    CUT_CALLS.append(task[0])
    return TS._e3_cut(task)


def w_exams_and_exif():
    print("the exams a pass may read; EXIF tags 3, 6, 8 only on ood22 / ood23 after the pre-flight")
    w = E3T.src_weights("M", 0)
    out = {"D5B on dev": refused(F._family_pass, w, "dev", ident_cfg(), "D5B", **KW),
           "D2LAION on imageweeds": refused(F._family_pass, w, "imageweeds", ident_cfg(), "D2LAION", **KW),
           "D4F on ood22": refused(F._family_pass, w, "ood22", ident_cfg(), "D4F", **KW),
           "test without the token": refused(F._family_pass, w, "test", ident_cfg(), "D6", **KW),
           "the probe in this process": refused(F._family_pass, w, "probe", ident_cfg(), "D5B", **KW),
           "ooddev_v1 in this process": refused(F._family_pass, w, "ooddev_v1", ident_cfg(), "E3RS", **KW),
           "the probe root asked for a two-stage pass": refused(F.family_pass_in_root, w, "probe", ts_cfg(), "D5B",
                                                                W.TMP / "out_x")}
    check("refused before any score: %s" % ", ".join(out), all(e is not None for e in out.values()),
          {k: str(v)[:100] for k, v in out.items()})
    with F.test_read_token() as tok:
        t = F._family_pass(w, "test", ident_cfg(), "D6", test_token=tok, **KW)
    check("test with score-test's token", t["result"]["exam"] == "test")
    h0 = TS.header
    n22 = len(C.read_manifest(C.manifest_path("ood22")))
    with patched(TS, "header", lambda p: (6, h0(p)[1])):
        e_dev = refused(F._family_pass, w, "dev", ident_cfg(), "E3RS", **KW)
        with moved(F.exif_dir()):
            e_pre = refused(F._family_pass, w, "ood22", ident_cfg(), "D1", **KW)
        ok = F.exif_preflight_passed()
        ps = F._family_pass(w, "ood22", ts_cfg(), "D1", **KW)
    check("tag 6 refused on dev and on ood22 before the pre-flight; after it a two-stage ood22 pass reads it: the "
          "true tag recorded, E3's check handed an untagged frame (%s)" % ps["stats"]["frame_adapted"],
          e_dev is not None and "orientation" in str(e_dev) and e_pre is not None and ok
          and ps["stats"]["orientations"] == {"6": n22} and ps["stats"]["frame_adapted"] == {"6": n22}
          and ps["stats"]["e3_orientations"] == {"None": n22}, (e_dev, e_pre, ps["stats"]["orientations"]))
    with patched(TS, "header", lambda p: (5, h0(p)[1])):
        e5 = refused(F._family_pass, w, "ood22", ident_cfg(), "D1", **KW)
    check("  tag 5 is never read", e5 is not None, e5)


def w_equivalence():
    print("the four reproductions (V3F.3.2), each written once")
    rec_dev = C.INC_DIR / E3T.REF / "runs" / "final__base__s1" / "scores" / "dev.json"
    d = json.loads(rec_dev.read_text())
    c0 = sorted(d["per_class"])[0]
    with tamper(rec_dev, lambda x: x["per_class"].update({c0: x["per_class"][c0] + 0.01})):
        e = refused(F.equivalence_dev_identity, **KW)
    check("item 1: a recorded class AP 0.01 off is refused, nothing written", e is not None
          and not (F.equiv_dir() / "dev_identity.json").exists(), e)
    with tamper(rec_dev, lambda x: x["per_class"].update({c0: x["per_class"][c0] + 0.001})):
        dv = F.equivalence_dev_identity(**KW)
    check("  0.001 off: recorded as a deviation (status deviation, the job continues)", dv["status"] == "deviation"
          and dv["per"]["1"]["status"] == "deviation" and dv["per"]["0"]["status"] == "passed", dv["per"]["1"])
    (F.equiv_dir() / "dev_identity.json").unlink()
    r1 = F.equivalence_dev_identity(**KW)
    check("item 1: b_v2_m640 seeds 0-2 reproduce their recorded dev scores (diff 0 on the CPU); written once",
          r1["status"] == "passed" and all(v["diff"]["max_abs_diff"] == 0 for v in r1["per"].values())
          and refused(F.equivalence_dev_identity, **KW) is not None, r1["per"])
    t2 = {}
    for arm in ("M", "A", "B"):
        t2[arm] = F.equivalence_two_stage(arm, EMB, **KW)
    check("item 2: E3-M, E3-A, E3-B reproduce E3's recorded dev files exactly and their stored stage-1 box sets (xyxy "
          "%.2g px, q %.2g)" % (max(v["stage1"]["xyxy_max_abs_diff_px"] for r in t2.values()
                                    for v in r["record"]["per"].values()),
                                max(v["stage1"]["q_max_abs_diff"] for r in t2.values() for v in r["record"]["per"].values())),
          all(r["record"]["status"] == "passed" and len(r["passes"]) == 3 for r in t2.values())
          and all(v["stage1"]["top_cls_equal"] for r in t2.values() for v in r["record"]["per"].values()))
    (F.equiv_dir() / "two_stage_B.json").unlink()
    load0 = SC.load_npz

    def shifted(p):
        x = load0(p)
        if str(p).endswith("dev.stage1.npz"):
            x["q"] = x["q"] + 2e-4
        return x
    with patched(SC, "load_npz", shifted):
        e2 = refused(F.equivalence_two_stage, "B", EMB, **KW)
    check("  a stored stage-1 record whose q differs by 2e-4 refuses the pass, nothing written", e2 is not None
          and "stage-1" in str(e2) and not (F.equiv_dir() / "two_stage_B.json").exists(), e2)
    F.equivalence_two_stage("B", EMB, **KW)
    r3 = F.equivalence_identity("imageweeds", "E3RS", **KW)
    r3b = F.equivalence_identity("ood22", "D1", **KW)
    e3 = refused(F.equivalence_identity, "ood23", "E3RS", **KW)
    check("item 3: b_v2_m640 s0 identity = the plain locked score in the same job on imageweeds and ood22 (exactly), "
          "the plain score recorded only there; an arm that does not read the exam is refused",
          r3["status"] == r3b["status"] == "passed" and r3["per"]["0"]["diff"]["max_abs_diff"] == 0
          and r3["per"]["0"]["plain"]["map50_95"] is not None and e3 is not None, (r3["per"], e3))
    t0 = time.time()
    r3p = F.equivalence_identity("probe", "D5B", **KW)
    pr = F.root_of("probe")
    stray = [p for p in pr.rglob("*") if p.is_file() and not str(p.relative_to(pr)).startswith(("splits", "exams"))]
    check("  the probe in its own root: a subprocess with INC_DIR there takes both scores (%.0fs); nothing written in "
          "the root" % (time.time() - t0), r3p["status"] == "passed" and r3p["exam"] == "probe"
          and r3p["per"]["0"]["plain"]["n_images"] == 5 and not stray, (r3p["per"], stray))
    r4 = F.equivalence_injection(**KW)
    check("item 4: b_v2_m640 s0's dev rows, injected, reproduce its recorded dev score (diff %.2g)"
          % r4["per"]["0"]["diff"]["max_abs_diff"], r4["status"] == "passed"
          and r4["per"]["0"]["vs_identity"]["max_abs_diff"] == 0, r4["per"])
    names = F.equivalence_names("imageweeds", e3_arms=("B",))
    got = F.require_equivalence(names)
    check("require_equivalence: the records of an imageweeds score resting on E3-B's boxes", set(got) == set(names)
          and all(v["status"] == "passed" for v in got.values()), got)
    cs0 = F.code_shas
    with patched(F, "code_shas", lambda: dict(cs0(), v3fam="0" * 64)):
        e5 = refused(F.require_equivalence, names)
    e6 = refused(F.require_equivalence, ["identity_ood23.json"])
    check("  refused: code other than the one that ran; a missing record", e5 is not None and "code changed" in str(e5)
          and e6 is not None, (e5, e6))
    WORLD["two_stage"] = t2


def w_commit_and_trees():
    print("commit_pass (written once) and the E-series trees")
    ps = WORLD["ts_A0"]
    eqs = F.require_equivalence(F.equivalence_names("dev", e3_arms=("A",)))
    doc = F.family_doc("E3RS", 0, "dev", ps, ts_cfg(), classifier=TS.load_classifier()[5], equivalence=eqs)
    d = F.arm_dir("E3RS_unit", 0)
    js = F.commit_pass(d, "dev", doc, ps)
    rec = json.loads(js.read_text())
    s1 = SC.load_npz(d / F.file_names("dev")[2])
    check("a family score: stamped V3F- (never a protocol score), the mode, its npz files by sha256, features kept "
          "on dev", rec["scorer_sha256"].startswith("TEST-V3F-") and rec["production"] is False
          and rec["settings"]["mode"] == "two_stage" and rec["images"]["sha256"] == W.sha(d / rec["images"]["name"])
          and rec["stage1_images"]["sha256"] == W.sha(d / rec["stage1_images"]["name"]) and "feat" in s1
          and rec["decided_by"] == F.DECIDED_BY and rec["stage2"]["classifier"]["npz_sha256"]
          and set(rec["equivalence"]) == {"dev_identity.json", "injection.json", "two_stage_A.json"}
          and rec["family_deviations"] == [], sorted(rec)[:10])
    e = refused(F.commit_pass, d, "dev", dict(doc), ps)
    check("  written once: a second commit is refused", e is not None, e)
    s1o = WORLD["stage1_ood23"]
    d2 = F.arm_dir("D1_unit", 0)
    js2 = F.commit_pass(d2, "ood23", F.family_doc("D1", 0, "ood23", s1o, ident_cfg(mode="stage1")), s1o)
    r2 = json.loads(js2.read_text())
    p7 = SC.load_npz(d2 / r2["p7"]["name"])
    check("  a stage1 score keeps its rows and P7's arrays (target and OtherPlant readings) beside it",
          "rows" in r2 and "t_tp" in p7 and "o_n_gt" in p7 and r2["p7"]["sha256"] == W.sha(d2 / r2["p7"]["name"]))
    ts = WORLD["two_stage"]["B"]["passes"][0]
    ts_iw = F._family_pass(E3T.src_weights("B", 0), "imageweeds", ts_cfg(), "D4F", **KW)
    d3 = F.arm_dir("D4F_unit", 0)
    js3 = F.commit_pass(d3, "imageweeds", F.family_doc("D4F", 0, "imageweeds", ts_iw, ts_cfg()), ts_iw)
    s13 = SC.load_npz(d3 / json.loads(js3.read_text())["stage1_images"]["name"])
    check("  an imageweeds stage-1 record is written without per-box features", "feat" not in s13
          and "feat" in ts["stage1"])
    with F.e_trees_unchanged() as before:
        F._family_pass(E3T.src_weights("M", 1), "dev", ident_cfg(), "E3RS", **KW)
    check("family passes leave twostage/e3_v1 and capacity/e{1,2,3}_* unchanged (%d files hashed)" % len(before),
          len(before) > 10)
    shutil.rmtree(str(F.root() / "arms"), ignore_errors=True)


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(F.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def cli(*a):
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam"] + list(a), env=env,
                           capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    rc, out, _ = cli("frozen")
    check("frozen prints the six sha256s (exit 0)", rc == 0 and json.loads(out) == F.FROZEN_SHA256, (rc, out[:200]))
    rc, _o, err = cli("equivalence")
    rc2, _o2, err2 = cli("equivalence", "--item", "identity", "--exam", "ood23", "--arm", "E3RS")
    rc3, _o3, _e3 = cli("nonsense")
    check("usage errors and refusals exit 2 with [inc2.v3fam] ERROR", rc == 2 and "--item" in err and rc2 == 2
          and "ERROR" in err2 and rc3 == 2, (rc, err[-200:], rc2, err2[-200:], rc3))


WORLD_TESTS = ("w_setup", "w_layering_identity", "w_modes", "w_exams_and_exif", "w_equivalence", "w_commit_and_trees",
               "w_cli")


def main():
    """UNITS, then (unless V3FAM_ONLY=units) WORLD_TESTS in order (each builds on the files the earlier ones
    wrote); V3FAM_UPTO=<name> stops after it."""
    t0 = time.time()
    only = os.environ.get("V3FAM_ONLY")
    upto = os.environ.get("V3FAM_UPTO")
    try:
        WORLD["rows"] = W.build_world()
        seq = UNITS + (() if only == "units" else WORLD_TESTS)
        for name in seq:
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
