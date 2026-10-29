#!/usr/bin/env python3
"""DS1's core (inc2/ds1.py; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-29): DS1, scaling verified training data at
the weed level", DS.7, DS.8, DS.9.4, DS.10, DS.11; docs/prereg/DS1_2026-10-29.md).

UNITS (no scoring pass; DS1_ONLY=units runs only them):
- the registration (the prereg's sha256 equals the committed file's; the time-stamp record and its check);
- the constants: Bonferroni within each exam (P1-P3 on MD-S0 z > 2.128, P4 on ood23 z > 1.645), SESOI 0.005, three
  seeds, the family cap 260 GPU-h and the phase reserve's line 174.2, the read cap 4, the dates;
- the statistics: DS1's seed texts (inc2/ds1/<cid>/image | group | session), the image SE equal to inc2.v3fam's
  bootstrap on the same arrays and counts, the cluster bootstrap, max(image, cluster) deciding, the 1,771-image
  restriction recomputing the full-set values; 'confirmed' = qualifies and z over its exam's bar ('qualifies, not
  confirmed' otherwise); N1 (lower 90 % end above -0.005), H1 (z < -1.645), N1's power about 0.3 at sigma 0.0044,
  the MDEs (ood23 about 0.040); the substitution and its date; the fixed outcome sentences;
- the reads: the order (S0x, then S2x or S1ux, the challenger only on a confirmed P1 or P2 with the larger D, then
  the cwd12 test), the cap of 4, out of order refused, a read taken once, read_by exactly the owner's or the
  delegate's name; the 0.90 reading (reaches / clears with SE = max(image, group));
- the family cap and the phase reserve; the storage floors (257.8 GB and 425.6 GB on the 6.836 TiB quota, phase 1's
  60 GB under V3F's floor);
- RA on the cluster's two segment reports (fixtures, sha256-checked): every step's chain and truth D, the seed-only
  z (inc0001 -2.5, inc0003 -4.7, inc0004 -1.4, inc0005 -0.4), the readings (a resolved loss below / above SESOI,
  not evidence of harm, naming not placement), written once with its .md;
- LN's statistic and readings; the MT-W rule (1 in 60: Wilson 0.089; 2: 0.114; a whole source passes once
  corrected); the exam allow-list and the sealed exams; the write isolation (trees_unchanged);
- the verdicts on synthetic committed arrays (a loader stub): P1 and P2 at v1, a not-run arm keeping its place,
  v2's P3 on MD-S0, P4 and its 1,771-image reading in the external record, N1 and H1 on dev, the substitution,
  the platform record free of exam-named keys and written once, the read order inside it.
WORLD (real Ultralytics CPU passes in test_inc2_e3.py's world): DS1's validator layer over V3F's over E3's; a DS1
pass's weed arrays give exactly the stage-1 agnostic AP, which equals V3F's family pass's for the same weights; an
MD root written by inc2.ds1_data scored in a subprocess with the plain locked score beside (identity = plain).

Run:  python3 tests/test_inc2_ds1.py        (DS1_ONLY=units: the units only)
"""
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
UNITS_ONLY = os.environ.get("DS1_ONLY") == "units"
if UNITS_ONLY:
    TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_"))
    os.environ["INC_DIR"] = str(TMP / "inc")
    os.environ["REPO"] = str(TMP / "repo")
    os.environ["INC_SCORER_TESTING"] = "1"
    sys.path.insert(0, str(HERE.parent))
else:
    sys.path.insert(0, str(HERE))
    import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
    import test_inc2_e3 as E3T  # noqa: E402
    TMP = W.TMP

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402

FAILURES = []
FIX = HERE / "fixtures" / "ds1"
REPORTS = {"weed_stream_v1_fork1364_s001_report.json":
           "fa4856e3a744b9f2d42bf5add629ca6bf45902d8aaffd4cec4b2f1a3f4258b98",
           "weed_stream_v1_fork1364_s002_report.json":
           "1bf148e77e1cd16585abc19bc66d3e87a0f3c29c00236b8754121608b9ed70a0"}
RES = 40                                       # bootstrap resamples in the tests (DS1's are 1,000)


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
        return e
    return None


# ------------------------------------------------------------------ synthetic arrays
def agn_arrays(n_img, seed, quality, keys=None):
    """Class-collapsed per-image arrays {keys, tp, conf, pred_img, n_gt}: each image 3 GT boxes, 4 predictions, a
    prediction a true positive at the first k thresholds with probability `quality`."""
    rng = np.random.default_rng(seed)
    keys = keys or ["k%04d" % i for i in range(n_img)]
    tp, conf, img = [], [], []
    for i in range(n_img):
        for _j in range(4):
            good = rng.random() < quality
            k = int(rng.integers(5, 11)) if good else 0
            row = np.zeros(10, dtype=bool)
            row[:k] = True
            tp.append(row)
            conf.append(float(rng.random() * (0.5 + 0.5 * good)))
            img.append(i)
    return {"keys": np.asarray(keys), "tp": np.asarray(tp), "conf": np.asarray(conf), "pred_img": np.asarray(img),
            "n_gt": np.full(n_img, 3, dtype=np.int64)}


def runs(n, base_seed, quality):
    return [agn_arrays(n, base_seed + s, quality) for s in range(3)]


# ------------------------------------------------------------------ units
def t_registration():
    print("the registration and the time stamp")
    f = HERE.parents[1] / D.PREREG_FILE
    check("the prereg file hashes to the registered sha256 (3b63933b...)",
          hashlib.sha256(f.read_bytes()).hexdigest() == D.PREREG_SHA256, D.PREREG_SHA256)
    check("its commit and the stamp commit are named", D.PREREG_COMMIT.startswith("b60b8ee") and
          D.STAMP_COMMIT.startswith("972b3eb"))
    check("no DS1 job before the time-stamp record (require_timestamp refuses)", refused(D.require_timestamp))
    pr = TMP / "proofs"
    pr.mkdir(parents=True, exist_ok=True)
    (pr / "a.ots").write_bytes(b"proof-a")
    (pr / "b.ots").write_bytes(b"proof-b")
    real = [str(HERE.parents[1] / "docs" / "prereg" / n) for n in ("DS1_2026-10-29.md.ots",
                                                                    "DS1_2026-10-29.stamp.txt.ots")]
    e0 = refused(D.timestamp_record, str(pr / "a.ots"), str(pr / "b.ots"))
    e1 = refused(D.timestamp_record, real[1], real[1])
    e2 = refused(D.timestamp_record, real[0], real[0])
    check("proofs that do not stamp the prereg (a.ots), the stamp file's proof given as the prereg's, or the "
          "prereg's as the stamp file's are refused", e0 is not None and e1 is not None and e2 is not None,
          (e0, e1, e2))
    check("  the committed proofs' headers stamp the prereg's sha256 and the stamp file's",
          D.ots_digest(real[0]) == D.PREREG_SHA256 and D.ots_digest(real[1]) == C.sha256_file(real[1][:-4]))
    e = refused(D.timestamp_record, real[0], real[1], sha256="0" * 64)
    check("a record naming another sha256 is refused", e is not None, e)
    rec = D.timestamp_record(real[0], real[1])
    check("the record names the prereg's sha256 and both proofs (what each stamps); timestamp_ok and "
          "require_timestamp pass", D.timestamp_ok() and D.require_timestamp() == C.sha256_file(D.timestamp_path())
          and rec["proofs"][0]["stamps"] == D.PREREG_SHA256, rec.get("proofs"))
    check("  and it is written once", refused(D.timestamp_record, real[0], real[1]))


def t_constants():
    print("the constants of DS.7 and DS.11")
    check("Bonferroni within each exam: MD-S0 m = 3 (z > 2.128), ood23 m = 1 (z > 1.645)",
          abs(D.THRESHOLD["md_s0"] - 2.128) < 5e-4 and abs(D.THRESHOLD["ood23"] - 1.645) < 5e-4, D.THRESHOLD)
    check("SESOI 0.005, three seeds, 1,000 resamples", D.SESOI == 0.005 and tuple(D.SEEDS) == (0, 1, 2)
          and D.RESAMPLES == 1000)
    check("the family cap 260 GPU-h, the phase reserve 85.8, phase 1's line 174.2", D.FAMILY_CAP_GPUH == 260.0 and
          abs(D.PHASE1_CEILING_GPUH - 174.2) < 1e-9)
    check("at most 4 sealed reads; S2's freeze date 2026-12-05, the reads by 2026-12-20",
          D.READ_CAP == 4 and D.S2_FREEZE_BY == "2026-12-05" and D.READS_BY == "2026-12-20")
    check("the primaries: P1 S1c over S0, P2 S1u over S1c (MD-S0, phase 1); P3 S2x over S0x (MD-S0), P4 S2x over S0x "
          "(ood23) at phase 2", [(p, D.PRIMARIES[p]["x"], D.PRIMARIES[p]["y"], D.PRIMARIES[p]["exam"],
                                  D.PRIMARIES[p]["phase"]) for p in ("P1", "P2", "P3", "P4")] ==
          [("P1", "S1C", "S0", "md_s0", 1), ("P2", "S1U", "S1C", "md_s0", 1), ("P3", "S2X", "S0X", "md_s0", 2),
           ("P4", "S2X", "S0X", "ood23", 2)])
    check("ood23's 13 disputed images are BD_26 ... BD_38", len(D.OOD23_DISPUTED) == 13 and
          D.OOD23_DISPUTED[0].endswith("BD_26") and D.OOD23_DISPUTED[-1].endswith("BD_38"))


def t_statistics():
    print("the statistics (V3F's array functions, DS1's seed texts)")
    x, y = runs(60, 10, 0.75), runs(60, 20, 0.6)
    groups = ["g%02d" % (i // 3) for i in range(60)]
    xv = [F.statistic_value(a, "agnostic") for a in x]
    yv = [F.statistic_value(a, "agnostic") for a in y]
    rec = D.comparison("P1/md_s0", xv, yv, x, y, "agnostic", groups, "group", "max", resamples=RES)
    st_i = "inc2/ds1/P1/md_s0/image"
    ref = F._bootstrap(x, y, F.image_counts(60, st_i, RES), "agnostic", SC.SPECIES, None)
    check("the image SE is inc2.v3fam's paired image bootstrap under the seed text inc2/ds1/<cid>/image",
          rec["image"]["seed_text"] == st_i and abs(rec["image"]["se"] - ref["se"]) < 1e-12, (rec["image"], ref["se"]))
    refc = F._bootstrap(x, y, F.session_counts(groups, "inc2/ds1/P1/md_s0/group", RES), "agnostic", SC.SPECIES,
                        None)
    check("the capture-group bootstrap under inc2/ds1/<cid>/group", rec["cluster"]["seed_text"] ==
          "inc2/ds1/P1/md_s0/group" and abs(rec["cluster"]["se"] - refc["se"]) < 1e-12 and
          rec["cluster"]["n_clusters"] == 20)
    check("max(image, group) decides", abs(rec["decision"]["se"] - max(ref["se"], refc["se"])) < 1e-12)
    d = statistics.fmean(xv) - statistics.fmean(yv)
    sig = math.sqrt(rec["decision"]["se"] ** 2 + (statistics.stdev(xv) ** 2 + statistics.stdev(yv) ** 2) / 3)
    check("D and the combined z: z = D / sqrt(SE^2 + (sd_X^2 + sd_Y^2) / 3)", abs(rec["diff"] - d) < 1e-12 and
          abs(rec["decision"]["z"] - d / sig) < 1e-9)
    keep = ["k%04d" % i for i in range(60) if i % 10 != 3]
    rk = D.comparison("P4/ood23_1771", xv, yv, x, y, "agnostic", groups, "session", "max", keep=keep, resamples=RES)
    check("a restriction (ood23's 1,771) recomputes the full-set values on the kept images",
          rk["n_images"] == 54 and abs(rk["x_values"][0] - F.statistic_value(F.subset_arrays(x[0], keep), "agnostic"))
          < 1e-12 and rk["restricted_to"]["n_images"] == 54)
    check("a comparison with two seeds a side is refused", refused(D.comparison, "c", xv[:2], yv[:2], x[:2], y[:2],
                                                                   "agnostic", resamples=RES))
    check("se_rule max without clusters is refused", refused(D.comparison, "c", xv, yv, x, y, "agnostic",
                                                             se_rule="max", resamples=RES))
    # confirmed
    mk = lambda q, z, lo, hi, o="inconclusive": {"decision": {"qualifies": q, "z": z, "interval_90": [lo, hi],  # noqa
                                                              "outcome": o, "diff": (lo + hi) / 2}}
    check("confirmed: qualifies and z over the exam's bar (MD-S0 2.13, ood23 1.645)",
          D.confirmed(mk(True, 2.2, 0.001, 0.02), "md_s0") and not D.confirmed(mk(True, 2.0, 0.001, 0.02), "md_s0")
          and D.confirmed(mk(True, 1.7, 0.001, 0.02), "ood23") and not D.confirmed(mk(False, 3.0, 0.001, 0.02),
                                                                                   "md_s0"))
    check("  a primary that qualifies under its bar reads 'qualifies, not confirmed'",
          D.primary_reading("P1", mk(True, 2.0, 0.001, 0.02, "qualifies"))["reading"] == "qualifies, not confirmed")
    check("N1: non-inferior when the lower 90 % end is above -0.005", D.n1_reading(mk(False, 0.1, -0.0049, 0.004))[
        "non_inferior"] and not D.n1_reading(mk(False, 0.1, -0.0051, 0.004))["non_inferior"])
    check("H1 fires at z < -1.645", D.h1_reading(mk(False, -1.7, -0.02, -0.001))["fires"] and
          not D.h1_reading(mk(False, -1.6, -0.02, 0.001))["fires"])
    check("N1's power about 0.3 at sigma 0.0044 and a true difference of 0", abs(D.n1_power(0.0044) - 0.305) < 0.01,
          D.n1_power(0.0044))
    check("the MDEs: (2.128 + 0.842) sigma on MD-S0; ood23 at V3F P7's sigma about 0.040",
          abs(D.mde(0.01, D.THRESHOLD["md_s0"]) - 0.0297) < 1e-4 and abs(D.mde(D.P7_SIGMA, D.THRESHOLD["ood23"]) -
                                                                         0.040) < 0.001)
    check("the substitution: S2 not frozen after 2026-12-05 (and not before, nor when frozen)",
          D.substitution_due("2026-12-06", False) and not D.substitution_due("2026-12-05", False) and
          not D.substitution_due("2026-12-30", True) and D.phase2_arms(True) == ("S1U", "S1UX"))
    s1 = D.sentence("P1", mk(True, 2.5, 0.004, 0.02, "qualifies"), {"images": 61234, "photos": 50000, "groups": 900})
    s2 = D.sentence("P3", mk(False, 0.1, -0.004, 0.004, "equivalent"), k=3)
    s4 = D.sentence("P4", mk(False, 0.5, -0.03, 0.05))
    check("the fixed sentences (DS.7.4): confirmed, equivalent, inconclusive, P4 with ood23's caveat",
          s1.startswith("Adding the in-hand recovered images (61234 images") and "raised" in s1 and
          "less than +-0.005" in s2 and "3 times" in s2 and "cannot tell" in s4 and "1,771" in s4, (s1, s2, s4))


def t_reads():
    print("the sealed reads (DS.7.5) and the 0.90 wording")
    v1 = {"status": "decided", "primaries": {"P1": {"confirmed": True, "diff": 0.004},
                                             "P2": {"confirmed": True, "diff": 0.006}}}
    v2 = {"status": "decided"}
    o = D.read_order(v1, v2)
    check("the order: S0x then S2x on MT-W, the confirmed primary with the larger D (P2: S1u), then S2x on the cwd12 "
          "test; at most 4", [(r["arm"], r["exam"], r["kind"]) for r in o] == [
              ("S0X", "mtw", "reference"), ("S2X", "mtw", "candidate"), ("S1U", "mtw", "challenger"),
              ("S2X", "cwd12", "cwd12")], o)
    o2 = D.read_order({"status": "decided", "primaries": {"P1": {"confirmed": False, "diff": 0.01}}}, v2, True)
    check("no confirmed primary: no challenger read; the substitution reads S1ux", [r["arm"] for r in o2] ==
          ["S0X", "S1UX", "S1UX"], o2)
    check("before verdict v2 no read is in the order", D.read_order(v1, None) ==
          [{"arm": "S1U", "exam": "mtw", "kind": "challenger", "primary": "P2"}])
    check("a read out of turn is refused", refused(D.check_read, "S2X", "mtw", o, {}))
    check("  a read not in the order is refused", refused(D.check_read, "S1C", "mtw", o, {}))
    check("  a read taken once", refused(D.check_read, "S0X", "mtw", o, {("S0X", "mtw"): "read"}))
    check("  in order it passes with its position", D.check_read("S2X", "mtw", o, {("S0X", "mtw"): "read"})[
        "position"] == 2)
    check("read_by is exactly the owner's or the delegate's name", D.read_by_value("owner") == "Harry He (owner)" and
          D.read_by_value("delegate") == "the delegate under the 2026-09-30 owner grant" and
          refused(D.read_by_value, "Harry"))
    r = D.reaches_clears([0.905, 0.91, 0.908], 0.002)
    lo = statistics.fmean([0.905, 0.91, 0.908]) - D.Z90 * math.sqrt(0.002 ** 2 + statistics.stdev(
        [0.905, 0.91, 0.908]) ** 2 / 3)
    check("reaches: mean >= 0.9000; clears: mean - 1.645 sqrt(SE^2 + sd^2/3) >= 0.9000", r["reaches"] and
          abs(r["lower_90"] - lo) < 1e-12 and r["clears"] == (lo >= 0.9))
    r2 = D.reaches_clears([0.902, 0.899, 0.903], 0.004)
    check("  a mean over 0.90 whose interval is not clears nothing", r2["reaches"] and not r2["clears"])


def t_cap_storage():
    print("the family cap, the phase reserve and the storage floors")
    check("within the cap", D.cap_check(100.0, 13.2, 1)[0])
    check("a phase-1 item past 174.2 before verdict v2 is refused (the phase reserve)",
          not D.cap_check(170.0, 13.2, 1)[0] and D.cap_check(170.0, 13.2, 1, v2_done=True)[0] and
          D.cap_check(170.0, 13.2, 1, substituted=True)[0] and D.cap_check(170.0, 13.2, 2)[0])
    check("past 260 is refused at any phase", not D.cap_check(250.0, 13.2, 2)[0])
    q = 6.836 * 1099.511627776
    fl = D.floors(q)
    check("the floors on the 6.836 TiB quota: V3F's 257.8 GB, lane DATA's 425.6 GB",
          abs(fl["v3f"] - 257.8) < 0.2 and abs(fl["data"] - 425.6) < 0.2, fl)
    check("a phase-1 item within its 60 GB keeps V3F's floor (611 GB free, 40 GB item: fine)",
          D.storage_short(611.0, q, 40.0, phase1_bytes_gb=10.0) == "")
    check("  past phase 1's 60 GB it keeps lane DATA's floor (460 GB free, 40 GB item: short)",
          D.storage_short(460.0, q, 40.0, phase1_bytes_gb=30.0).startswith("storage short") and
          D.storage_short(460.0, q, 40.0, phase1_bytes_gb=10.0) == "")
    check("  an item adding no bytes keeps V3F's floor", D.storage_short(250.0, q, 0.0).startswith("storage short")
          and D.storage_short(300.0, q, 0.0) == "")


def t_ra():
    print("RA on the stream's two segment reports (fixtures)")
    paths = []
    for n, sha in REPORTS.items():
        p = FIX / n
        check("fixture %s hashes as the cluster's copy (%s...)" % (n, sha[:8]), C.sha256_file(p) == sha)
        paths.append(str(p))
    rec = D.ra(paths, out_dir=TMP / "ra")
    by = {s["step"]: s for s in rec["steps"]}
    z = {k: by[k]["chains"]["r0"]["chain"]["seed_z"] for k in by}
    check("five steps: inc0001-inc0005", sorted(by) == ["inc0001", "inc0002", "inc0003", "inc0004", "inc0005"])
    check("the seed-only z: inc0001 -2.5, inc0003 -4.7, inc0004 -1.4, inc0005 -0.4 (DS.1's table)",
          round(z["inc0001"], 1) == -2.5 and round(z["inc0003"], 1) == -4.7 and round(z["inc0004"], 1) == -1.4 and
          round(z["inc0005"], 1) == -0.4, z)
    c1, c3, c4 = (by[k]["chains"]["r0"] for k in ("inc0001", "inc0003", "inc0004"))
    check("chain D: -0.0037, -0.0063, -0.0026", [round(c["chain"]["diff"], 4) for c in (c1, c3, c4)] ==
          [-0.0037, -0.0063, -0.0026])
    check("readings: inc0001 a resolved 12-class loss below SESOI and naming not placement; inc0003 above SESOI; "
          "inc0004 not evidence of harm at the chain's resolution",
          "a resolved 12-class loss, below SESOI" in c1["reading"] and "naming, not placement (cause C)" in
          c1["reading"] and "a resolved 12-class loss, above SESOI" in c3["reading"] and
          "not evidence of harm at the chain's resolution" in c4["reading"], (c1["reading"], c3["reading"],
                                                                                c4["reading"]))
    check("the dispositions as recorded (data; inc0002 and inc0005 hold)", c1["disposition"] == "data" and
          by["inc0002"]["chains"]["r0"]["disposition"] == "hold" and by["inc0005"]["chains"]["r0"]["disposition"] ==
          "hold")
    check("truth D: -0.0024, -0.0013, -0.0060 (inc0001, inc0003, inc0004)", [round(by[k]["truth"]["diff"], 4) for k in
                                                                            ("inc0001", "inc0003", "inc0004")] ==
          [-0.0024, -0.0013, -0.006])
    check("the shared null's one-sd shift is reported", "shared_null_shift" in c1["chain"])
    check("written once with its .md, every report named by sha256", (TMP / "ra" / "ds1_ra_gate.json").is_file() and
          (TMP / "ra" / "ds1_ra_gate.md").is_file() and refused(D.ra, paths, out_dir=TMP / "ra") and
          sorted(f["sha256"] for f in rec["reports"]) == sorted(REPORTS.values()))


def t_ln_mtw():
    print("LN's statistic and the MT-W check rule")
    def ims(n, wrong_every, miss=0):
        return [{"marks": ["not_plant" if wrong_every and i % wrong_every == 0 else "correct", "correct"],
                 "missing": miss} for i in range(n)]
    check("the error share: (wrong + missing) / (boxes + missing)", abs(D.ln_error_share(
        [{"marks": ["correct", "loose"], "missing": 1}]) - 2 / 3.0) < 1e-12)
    st = D.ln_statistic({"siu_increments": ims(200, 2), "inc0005": ims(100, 0), "train_core": ims(250, 0)},
                        resamples=200)
    check("a group with a quarter of its boxes wrong against a clean reference: supported",
          st["differences"]["siu_increments"]["reading"] == "supported", st["differences"])
    check("  a clean group: not supported", st["differences"]["inc0005"]["reading"] == "not supported")
    st2 = D.ln_statistic({"siu_increments": ims(20, 8), "train_core": ims(20, 0)}, resamples=200)
    check("  a small, noisy sample: unresolved", st2["differences"]["siu_increments"]["reading"] == "unresolved",
          st2["differences"])
    check("Wilson's 95 % upper bound: 1 in 60 is 0.089, 2 in 60 is 0.114", round(D.wilson_upper(1, 60), 3) == 0.089
          and round(D.wilson_upper(2, 60), 3) == 0.114, (D.wilson_upper(1, 60), D.wilson_upper(2, 60)))
    check("a source passes with at most 1 image in error of 60, fails with 2",
          D.mtw_source_verdict(60, 400, 1)["verdict"] == "pass" and D.mtw_source_verdict(60, 400, 2)["verdict"] ==
          "fail")
    check("a source of fewer than 60 MT-W images is checked whole and passes once its errors are corrected",
          D.mtw_source_verdict(35, 35, 3)["verdict"] == "fail" and D.mtw_source_verdict(35, 35, 3, corrected=True)[
              "verdict"] == "pass" and D.mtw_source_verdict(35, 35, 0)["verdict"] == "pass")
    check("  a sample short of 60 for a larger source is refused", refused(D.mtw_source_verdict, 40, 400, 0))


def t_allow_isolation():
    print("the exam allow-list, the sealed exams and the write isolation")
    check("a K arm reads dev, MD-S0 and ImageWeeds, never ood23", D.exam_allowed("KD", "dev") and
          refused(D.exam_allowed, "KD", "ood23"))
    check("an S arm reads ood23 and the MD-N exams", D.exam_allowed("S1C", "ood23") and
          D.exam_allowed("S1C", "mdn_rf_x"))
    check("S1c-R reads MD-S0 and dev only", refused(D.exam_allowed, "S1CR", "ood22"))
    check("the sealed exams only with score-test's token", refused(D.exam_allowed, "S0X", "test") and
          refused(D.exam_allowed, "S0X", "mtw_v2"))
    with D.test_token() as tok:
        check("  with the token they are allowed", D.exam_allowed("S2X", "test", tok))
    check("an unknown arm is refused", refused(D.exam_allowed, "S9", "dev"))
    check("an MD exam is scored only in its own root (ds1_pass refuses it in this process)",
          refused(D.ds1_pass, "w.pt", "md_s0", {"mode": "stage1"}, "S0"))
    check("  a sealed root exam without a prepared read is refused", refused(D.ds1_pass, "w.pt", "test_v1",
                                                                            {"mode": "stage1"}, "S0X"))
    cap = C.INC_DIR / "capacity"
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "e3_v1.json").write_text("{}")
    e = None
    try:
        with D.trees_unchanged():
            (cap / "e3_v1.json").write_text('{"x": 1}')
    except D.Ds1Refused as ex:
        e = ex
    check("a DS1 job that changes an E-series record fails (trees_unchanged)", e is not None, e)
    try:
        with D.trees_unchanged():
            (cap / "ds1_power.json").write_text("{}")
        ok = True
    except D.Ds1Refused:
        ok = False
    check("  a write under capacity/ds1_* is DS1's own", ok)
    for p in ("e3_v1.json", "ds1_power.json"):
        (cap / p).unlink()


def t_verdict():
    print("the verdicts on committed arrays (a loader stub)")
    n = 50
    md_keys = ["md%03d" % i for i in range(n)]
    oo_keys = ["ood23__2023080%d_iPhone12_WY_%d" % (i % 5, i) for i in range(n - 13)] + list(D.OOD23_DISPUTED)
    dev_keys = ["dev%03d" % i for i in range(n)]
    gpath = D.splits_dir() / "md_v1" / "groups.jsonl"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("".join(json.dumps({"exam": "md_s0", "key": k, "group": "g%d" % (i // 2)}) + "\n"
                             for i, k in enumerate(md_keys)))
    devp = C.manifest_path("dev")
    devp.parent.mkdir(parents=True, exist_ok=True)
    devp.write_text("".join(json.dumps({"image": "/x/%s.jpg" % k, "label": "/x/%s.txt" % k, "sha256": "0" * 64,
                                        "label_sha256": "0" * 64, "source": "cwd12", "session": "s%d" % (i % 8),
                                        "key": k}) + "\n" for i, k in enumerate(dev_keys)))
    qual = {"S0": 0.55, "S1C": 0.72, "S1U": 0.70, "S1CR": 0.6, "KD": 0.5, "KT": 0.5, "KF": 0.5, "K0": 0.5,
            "S0X": 0.6, "S2X": 0.75, "S2": 0.6, "S1UX": 0.7}

    def species(keys, seed, q):
        a = agn_arrays(len(keys), seed, q, keys)
        rng = np.random.default_rng(seed + 1)
        gt = np.random.default_rng(7).integers(0, 12, 3 * len(keys))      # one exam: the same GT for every run
        out = {"keys": a["keys"], "tp": a["tp"], "conf": a["conf"], "pred_img": a["pred_img"],
               "pred_cls": rng.integers(0, 12, len(a["conf"])), "target_img": np.repeat(np.arange(len(keys)), 3),
               "target_cls": gt}
        return out

    def loader(arm, exam, kind):
        out = []
        for s in range(3):
            seed = 1000 * (list(qual).index(arm) + 1) + s
            if kind == "species":
                a = species(dev_keys, seed, qual[arm])
                out.append(({"species_map50_95": F.statistic_value(a, "species")}, a))
            else:
                a = agn_arrays(n, seed, qual[arm], md_keys if exam.startswith("md") else oo_keys)
                out.append(({}, a))
        return out
    st = {a: "done" for a in qual}
    st["KD"] = "not run (supply under N_K)"
    D.RESAMPLES = RES                            # the bootstraps' size in the tests (DS1's are 1,000)
    doc, ext = D.verdict(1, st, loader=loader, write=True)
    p1, p2 = doc["primaries"]["P1"], doc["primaries"]["P2"]
    check("verdict v1 decides P1 (S1c over S0) and P2 (S1u over S1c) on MD-S0 with max(image, group)",
          p1["status"] == "decided" and p2["status"] == "decided" and p1["comparison"]["se_rule"] == "max" and
          p1["comparison"]["cluster"]["kind"] == "group" and p1["x"] == "S1C" and p1["y"] == "S0")
    check("  each reads its bar 2.128 and says confirmed when it qualifies over it", abs(p1["threshold"] - 2.128) <
          5e-4 and p1["confirmed"] == (p1["qualifies"] and p1["z"] > p1["threshold"]))
    check("  a Tier S comparison of an arm not run keeps its place, 'not run'", doc["tier_s"]["KT_over_KD/dev"][
        "status"] == "not run")
    check("  Tier S prints its MDE", "mde" in doc["tier_s"]["KF_over_KT/dev"], doc["tier_s"]["KF_over_KT/dev"])
    cleaned, dropped = R.dev_only(doc)
    check("  the platform record holds no exam-named key (the evidence scrub leaves it whole)", not dropped, dropped)
    check("  written once (capacity/ds1_v1.json and its external record)", refused(D.verdict, 1, st, loader=loader))
    doc2, ext2 = D.verdict(2, st, loader=loader, on_date="2026-11-30", s2_frozen=True, write=False)
    p3, p4 = doc2["primaries"]["P3"], ext2["primaries"]["P4"]
    check("verdict v2 decides P3 (S2x over S0x on MD-S0); P4 on ood23 goes to the external record",
          p3["status"] == "decided" and p3["x"] == "S2X" and "P4" not in doc2["primaries"] and
          p4["status"] == "decided" and p4["comparison"]["cluster"]["kind"] == "session")
    check("  P4's 1,771-image reading beside it", p4["sensitivity_1771"]["n_images"] == n - 13)
    check("  N1 and H1 on dev (species, image SE)", doc2["n1"] is not None and doc2["h1"] is not None and
          doc2["n1"]["comparison"]["se_rule"] == "image")
    check("  the read order inside the record (S0x, S2x, then the cwd12 test)", [r["arm"] for r in doc2["reads"][
        "order"]][:2] == ["S0X", "S2X"])
    doc3, _e3 = D.verdict(2, st, loader=loader, on_date="2026-12-06", s2_frozen=False, write=False)
    check("the substitution after 2026-12-05 with S2 not frozen: P3 and N1 on S1ux, named",
          doc3["primaries"]["P3"]["x"] == "S1UX" and doc3["substitution"]["applied"] and
          doc3["substitution"]["arms"] == ["S1U", "S1UX"])
    st2 = dict(st, S2X="failed")
    doc4, _e4 = D.verdict(2, st2, loader=loader, on_date="2026-11-30", write=False)
    check("a primary whose arm failed is 'not run' and keeps its α share", doc4["primaries"]["P3"]["status"] ==
          "not run")
    w1, w2 = D.mean_se(runs(30, 5, 0.7), {"k%04d" % i: "g%d" % (i // 3) for i in range(30)}, "mtw/S0X", RES)
    check("the 3-seed mean's SE (the 0.90 interval): image and max(image, group)", w2 >= w1 > 0, (w1, w2))
    m = D.merge_weed([agn_arrays(3, 1, 0.5, ["a", "b", "c"]), agn_arrays(2, 2, 0.5, ["d", "e"])])
    check("MT-W's parts pooled (keys distinct); a shared key refused", len(m["keys"]) == 5 and int(m["pred_img"].max())
          == 4 and refused(D.merge_weed, [agn_arrays(1, 1, 0.5, ["a"]), agn_arrays(1, 2, 0.5, ["a"])]))


def t_test_read():
    print("test-read: the order the verdicts fix, read_by, written once")
    cap = C.INC_DIR / "capacity"
    v1 = {"status": "decided", "primaries": {"P1": {"confirmed": False, "diff": 0.001}}}
    v2 = {"status": "decided", "substitution": {"applied": False}}
    e = refused(D.test_read, "S2X", "mtw", "owner", v1=v1, v2=v2)
    check("out of turn refused (S0x first)", e is not None, e)
    doc, argv = D.test_read("S0X", "mtw", "delegate", v1=v1, v2=v2)
    check("S0x on MT-W: read.json with read_by and the order; the argv runs score-test", doc["read_by"] ==
          D.DELEGATE and argv[-5:] == ["score-test", "--arm", "S0X", "--exam", "mtw"] and
          (D.reads_dir() / "1_S0X_mtw" / "read.json").is_file())
    check("  a second read of it refused", refused(D.test_read, "S0X", "mtw", "owner", v1=v1, v2=v2))
    check("  the next in order passes", D.test_read("S2X", "mtw", "owner", v1=v1, v2=v2)[0]["position"] == 2)
    check("  a read_by other than the two names refused", refused(D.test_read, "S2X", "cwd12", "harry", v1=v1, v2=v2))
    shutil.rmtree(str(D.reads_dir()), ignore_errors=True)


# ------------------------------------------------------------------ the world (CPU passes)
KW = {"batch": 32, "device": "cpu", "lock_check": True}


def w_passes():
    print("the world: DS1's pass on E3's fixtures (CPU)")
    from weed_optimizer_framework.tools.inc2 import ds1_data as DD
    E3T.setup_world()
    cls, base = D.ds1_validator_class({"mode": "stage1", "nms": "locked"})
    ds1l = [k for k in cls.__mro__ if k.__dict__.get("_ds1_layer")]
    fam = [k for k in cls.__mro__ if k.__dict__.get("_v3fam_layer")]
    e3 = [k for k in cls.__mro__ if k.__dict__.get("_e3_layer")]
    check("one DS1 layer over V3F's family layer over E3's", len(ds1l) == 1 and len(fam) == 1 and len(e3) == 1 and
          cls.__mro__[1] is fam[0])
    w = E3T.src_weights("B", 0)
    t0 = time.time()
    ps = D.ds1_pass(w, "dev", {"mode": "stage1", "nms": "locked"}, "K0", **KW)
    wa = ps["weed"]
    ap = F.S.collapsed_ap(wa["tp"], wa["conf"], int(wa["n_gt"].sum()))
    check("the weed arrays give exactly the pass's stage-1 agnostic AP (%.0fs)" % (time.time() - t0),
          abs(ap[0] - ps["stage1_agnostic"][0]) < 1e-12 and len(wa["keys"]) == len(ps["arrays"]["keys"]))
    fp = F._family_pass(w, "dev", {"mode": "stage1", "nms": "locked"}, "D4F", **KW)
    check("  equal to V3F's family pass of the same weights (stage-1 agnostic AP, the scorer's result)",
          abs(fp["stage1_agnostic"][0] - ps["stage1_agnostic"][0]) < 1e-12 and
          fp["result"]["map50_95"] == ps["result"]["map50_95"])
    rows = [dict(r, key="md_%s" % r["key"]) for r in W.WORLD["rows"]["dev"][:6]]
    root = TMP / "md_root_test"
    rr = DD.write_exam_root("md_s0", rows, root, {r["key"]: "g%d" % i for i, r in enumerate(rows)})
    out = D.pass_in_root(w, "md_s0", "identity", "POWER", TMP / "md_out", root_dir=root, plain=True, **KW)
    cmp_ = F.compare_with_recorded(out["result"], out["plain"], "MD identity")
    check("an MD root (inc2.ds1_data.write_exam_root) scored in a subprocess: identity equals the plain locked score",
          rr["images"] == 6 and cmp_["max_abs_diff"] <= 1e-6, cmp_)
    out2 = D.pass_in_root(w, "md_s0", "stage1", "POWER", TMP / "md_out2", root_dir=root, **KW)
    a2 = out2["weed"]
    check("  its stage1 pass returns the weed arrays of the root's images",
          a2 is not None and len(a2["keys"]) == 6 and abs(F.S.collapsed_ap(a2["tp"], a2["conf"], int(a2["n_gt"].sum()))[
              0] - out2["stage1_agnostic"][0]) < 1e-12)


def main():
    tests = [t_registration, t_constants, t_statistics, t_reads, t_cap_storage, t_ra, t_ln_mtw, t_allow_isolation,
             t_verdict, t_test_read]
    if not UNITS_ONLY:
        tests.append(w_passes)
    for fn in tests:
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
