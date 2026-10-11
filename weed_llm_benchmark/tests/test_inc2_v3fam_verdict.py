#!/usr/bin/env python3
"""V3F's verdict and its sealed-test reads (inc2/v3fam_verdict.py, the inc2.v3fam verbs verdict, test-read,
score-test, test-report, reanalyse-test, spend; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F, the v3
experiment family D0-D9", V3F.3.3-V3F.3.6, V3F.6.4).

A synthetic world in a temporary INC_DIR: the exams' manifests (dev's 8 sessions, ood22 and ood23 by the key
rule, the probe in its own root, test with dev's sessions among its 34), and committed family scores (the per-image
arrays of inc2.v3fam.commit_pass) for every side of every pre-declared comparison. No scoring pass runs: what is
pinned is how the verdict reads the committed scores and how the reads are guarded.

- the verdict: every primary computed by inc2.v3fam.comparison on the committed arrays (P1's image SE, P2's
  selection correction, P4's one-system rule, P6's larger of the image and session SEs over ood22's sessions, P7
  on the ignore-region arrays, P8 and P9 over the probe's seven species), equal to the same call made directly;
  Holm per exam; the Tier C gate with C = 14 counting the arms not run; the read order (D6 then D8-m when scored,
  BCLEAN first when D0-S triggered it); a side whose arm was not run is "not run" with its reason and keeps its
  place; D0-S's dev branch decides on dev minus H; the platform's record dev only (the evidence scrub leaves it
  whole) and written once and last, the external records beside;
- the reads: refused before the verdict, for an arm out of the order or out of turn, for a read_by other than the
  owner's or the delegate's exact name, a second time, and at the family cap (recorded "not run (family cap)",
  never retried); read.json written once with the argv; score-test refused unless INCAP_DECIDED_BY names the
  person read_by names (decision 20), when the verdict or the sources changed, or a second time; test-report's
  stratum of the test images from dev's sessions and the gap to 0.90; the re-analysis of E3's stored test arrays
  (O4's stratum, test minus H), once, with the same submitter guard;
- the family's GPU-h from the cluster's records (provenance attempts, running at their estimates; a read prepared
  and not yet submitted at its estimate; the asking job's own running attempt left out; the family's experiments'
  reports);
- C_D3 is P4's record (one seed text); the report shows D and D_sel with the plain D and E3-B's share beside; the
  dev pass count by kind; Tier S's array readings (stage-1 agnostic AP matched again to dev's labels and checked
  against the recorded value, D1's species subsets, D5b's per-copy-family inflation); D0-S's dev branch with every
  dev comparison's full-dev numbers beside and E3's comparisons re-read on dev minus H under E3's seed texts; the
  declaration of D0-S's removals; the ticker's cap refusals passed in the verdict's argv; the re-analysis only
  after the reads.

Run:  python3 tests/test_inc2_v3fam_verdict.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_v3fam_verdict_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_verdict as VR  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as EV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402

FAILURES = []
RESAMPLES = 20                                  # the bootstraps' size in the tests (the family's is 1,000)


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except F.V3famRefused as e:
        return e
    return None


# ------------------------------------------------------------------ the world
DEV_SESSIONS = ["2021081%d_iPhoneSE_YL" % i for i in range(8)]


def keys_for(exam, n):
    if exam == "dev":
        return ["dev_%03d" % i for i in range(n)]
    if exam == "ood22":
        return ["ood22__2022061%d_IphoneSE_YL_%d" % (i % 5, i) for i in range(n)]
    if exam == "ood23":
        return ["ood23__2023072%d_HTRC_EDMUND_MP_%d" % (i % 4, i) for i in range(n)]
    if exam == "probe":
        return ["train_core__probe_%03d" % i for i in range(n)]
    if exam == "test":
        return ["test_%03d" % i for i in range(n)]
    raise ValueError(exam)


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["key"])))


SIZES = {"dev": 24, "ood22": 20, "ood23": 16, "probe": 20, "test": 30}


def build_manifests():
    for exam, n in SIZES.items():
        rows = []
        for i, k in enumerate(keys_for(exam, n)):
            sess = ""
            if exam == "dev":
                sess = DEV_SESSIONS[i % 8]
            elif exam == "probe":
                sess = "2021090%d_iPhoneSE_YL" % (i % 5)
            elif exam == "test":
                sess = DEV_SESSIONS[i % 8] if i < 10 else "2021070%d_CanonEOS_EO" % (i % 6)
            rows.append({"key": k, "image": "/x/%s/%s.jpg" % (exam, k), "label": "/x/%s/%s.txt" % (exam, k),
                         "sha256": "%064x" % (abs(hash(k)) % (1 << 64)), "session": sess, "source": "s"})
        p = F.root_of(exam) / "splits" / "v1" / ("%s.jsonl" % exam) if exam == "probe" else C.manifest_path(exam)
        write_manifest(p, rows)


def synth(exam, seed, shift=0.0, species=(0, 1, 2, 3)):
    """Species arrays of one run on an exam (two GT boxes per image, one GT draw per exam; shift moves the TP rate)."""
    n = SIZES[exam]
    k = 4
    rng = np.random.default_rng(1000 * seed + 7 + int(round(shift * 1000)) + 500)
    g = np.random.default_rng(99 + n)
    keys = np.asarray(sorted(keys_for(exam, n)))
    pimg = np.repeat(np.arange(n), k)
    tp = rng.random((n * k, 10)) < np.clip(np.linspace(0.85, 0.2, 10) + shift, 0, 1)
    conf = np.round(rng.random(n * k), 2)
    pcls = rng.choice(np.asarray(species, dtype=float), n * k)
    timg = np.repeat(np.arange(n), 2)
    tcls = g.choice(np.asarray(species, dtype=float), 2 * n)
    return {"keys": keys, "tp": tp, "conf": conf, "pred_cls": pcls, "pred_img": pimg, "target_img": timg,
            "target_cls": tcls}


def p7_arrays(exam, seed, shift=0.0):
    a = synth(exam, seed, shift)
    n = len(a["keys"])
    return {"keys": a["keys"], "tp": a["tp"], "conf": a["conf"], "pred_img": a["pred_img"],
            "n_gt": np.bincount(a["target_img"], minlength=n)}


def put(dirname, exam, seed, shift=0.0, ens=None, variant="", arm="X", p7=False, species=(0, 1, 2, 3)):
    arrays = synth(exam, seed if seed is not None else 9, shift, species)
    d = SYS.system_dir(dirname, seed, ens)
    sp = F.statistic_value(arrays, "species")
    doc = {"format": F.FMT_SCORE, "family": F.NAME, "arm": arm, "dir": dirname, "seed": seed, "exam": exam,
           "variant": variant or None, "species_map50_95": sp, "agnostic_map50_95": sp - 0.01,
           "per_class": {C.CLASS_NAMES[c]: sp for c in species}, "stage1": {"agnostic_map50_95": sp - 0.02}}
    ps = {"arrays": arrays}
    if p7:
        pa = p7_arrays(exam, seed, shift)
        ps["p7"] = {"t": pa, "o": pa}
    F.commit_pass(d, exam, doc, ps, variant=variant)


SHIFT = {"E3M": 0.02, "E3A": 0.01, "E3B": 0.06, "REF": 0.0, "D2LAION": 0.03, "D2CLIP": 0.0, "D2DINOL": 0.01,
         "D2DINOB": 0.0, "D2SIGLIP": 0.02, "D2RN50": -0.02, "D2BC1": 0.0, "D2BC2640": 0.04, "D2FT": 0.0,
         "D4F": 0.04, "D4N": 0.07, "D4R15": 0.02, "D6": 0.03, "D8S": -0.05, "D8M": -0.03}


def build_scores(skip=()):
    for name, sh in SHIFT.items():
        if name in skip:
            continue
        for s in F.SEEDS:
            put(name, "dev", s, sh, arm=VR.SIDES[name][3])
    put("D3", "dev", None, 0.07, ens="ens", arm="D3")
    for s in F.SEEDS:
        put("TSB", "ood22", s, 0.05, arm="D1")
        put("TSA", "ood22", s, 0.02, arm="D1")
        put("D1A12", "ood22", s, 0.0, arm="D1")
        put("TSB", "ood23", s, 0.03, arm="D1")
        put("TSA", "ood23", s, 0.01, arm="D1")
        put("D1A12", "ood23", s, 0.0, arm="D1")
        put("D1B1", "ood23", s, 0.04, arm="D1", p7=True)
        put("D1A1", "ood23", s, 0.0, arm="D1", p7=True)
        if "D5B" not in skip:
            for name, sh in (("D5B0", 0.0), ("D5B25", 0.02), ("D5B100", 0.08)):
                put(name, "probe", s, sh, arm="D5B", species=(0, 1, 2, 3))
    for arm in ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640",
                "D2FT", "D3", "D4F", "D4N", "D4R15", "D6", "D8", "D1"):
        if arm in skip:
            continue
        TS._write_json(SYS.score_record_path(arm), {"format": SYS.FMT_SCORE_RECORD, "arm": arm, "status": "complete",
                                                   "exam": "dev", "dev": []})


def reset():
    for d in (F.root(), C.INC_DIR / "capacity"):
        shutil.rmtree(str(d), ignore_errors=True)


# ------------------------------------------------------------------ tests
def t_verdict():
    print("the verdict on committed family scores")
    reset()
    build_manifests()
    build_scores(skip=("D5B",))
    old = F.RESAMPLES
    F.RESAMPLES = RESAMPLES
    try:
        VR.F.RESAMPLES = RESAMPLES
        # the comparisons take RESAMPLES through inc2.v3fam.comparison's default: run with a small size here
        import functools
        real = F.comparison
        F.comparison = functools.partial(real, resamples=RESAMPLES)
        try:
            doc = VR.verdict()
        finally:
            F.comparison = real
    finally:
        F.RESAMPLES = old
    p = doc["primaries"]
    check("capacity/v3fam_v1.json decided, dev only: P1-P5, Holm on dev, the Tier C gate (C = 14, bar 2.690110), "
          "the read order, Tier S on dev", doc["status"] == "decided" and sorted(p) == ["P1", "P2", "P3", "P4", "P5"]
          and sorted(doc["holm"]) == ["P1", "P2", "P3", "P4", "P5"] and doc["challengers"]["C"] == 14
          and abs(doc["challengers"]["bar"] - 2.690110) < 1e-6 and doc["exam"] == "dev"
          and all(v.get("exam") == "dev" for v in doc["tier_s"].values()), sorted(doc))
    xr = [a for _d, a in SYS.family_runs("E3B", "dev")]
    yr = [a for _d, a in SYS.family_runs("D2LAION", "dev")]
    want = F.comparison("P1", [d["species_map50_95"] for d, _a in SYS.family_runs("E3B", "dev")],
                        [d["species_map50_95"] for d, _a in SYS.family_runs("D2LAION", "dev")], xr, yr,
                        sessions=VR.sessions("dev"), resamples=RESAMPLES, paired=True)
    check("P1 equals inc2.v3fam.comparison on the same committed arrays (D, the image SE, the session SE beside, z, "
          "the outcome), paired per seed beside",
          p["P1"]["status"] == "decided" and abs(p["P1"]["diff"] - want["diff"]) < 1e-12
          and abs(p["P1"]["decision"]["z"] - want["decision"]["z"]) < 1e-9
          and p["P1"]["session"]["n_sessions"] == 8 and p["P1"]["session"]["deciding"] is False
          and len(p["P1"]["paired_per_seed"]) == 3, (p["P1"].get("diff"), want["diff"]))
    check("P2, P3, P5 selection-corrected (E3's choice redone in each resample; the plain D beside), P4 one system",
          all(p[k]["selection"] and "share_B_chosen" in p[k]["plain"] and p[k]["se_rule"] == "selection"
              for k in ("P2", "P3", "P5")) and p["P4"]["decision"]["one_system"] is True, [p[k].get("status")
                                                                                          for k in p])
    ext = json.loads((C.INC_DIR / "capacity" / VR.EXTERNAL).read_text())
    e = ext["primaries"]
    check("the external record: P6 on ood22 with the larger of the image and session SEs (5 sessions by the key "
          "rule), P7 on ood23's ignore-region arrays (agnostic), P8 and P9 'not run' (D5b's records missing) with "
          "their place in the probe's Holm set",
          e["P6"]["status"] == "decided" and e["P6"]["se_rule"] == "max" and e["P6"]["session"]["n_sessions"] == 5
          and e["P6"]["decision"]["se"] == max(e["P6"]["image"]["se"], e["P6"]["session"]["se"])
          and e["P7"]["status"] == "decided" and e["P7"]["statistic"] == "agnostic"
          and e["P8"]["status"] == "not run" and e["P9"]["status"] == "not run"
          and ext["holm"]["P8"]["m"] == 2 and ext["holm"]["P8"]["family_confirmed"] is False,
          {k: (v.get("status"), v.get("why")) for k, v in e.items()})
    gate = doc["challengers"]
    check("Tier C: fourteen arms, D9 not among them, D3 one system; every scored challenger decided; the order of "
          "the reads: D6 then D8-m (references, scored), then the challengers that open a read",
          sorted(gate["arms"]) == sorted(F.CHALLENGERS) and gate["comparisons"]["D3"]["decision"]["one_system"]
          and [o["arm"] for o in doc["reads"]["order"]][:2] == ["D6", "D8M"]
          and all(o["kind"] == "challenger" for o in doc["reads"]["order"][2:]), doc["reads"])
    check("the platform's record passes the evidence scrub whole (no non-dev exam key) and the allow-list",
          R.dev_only(doc)[0] == doc and EV.allowed("capacity/v3fam_v1.json"), None)
    e2 = refused(VR.verdict)
    check("the verdict is written once (a second run refuses)", e2 is not None, e2)
    rep = json.loads((C.INC_DIR / "capacity" / (VR.REPORT + ".json")).read_text())
    md = (C.INC_DIR / "capacity" / (VR.REPORT + ".md")).read_text()
    check("the report: every comparison with D, 2 sd, both SEs, z, the interval, the outcome; the family table; "
          "the dev pass count by kind",
          len(rep["comparisons"]) == len(VR.COMPARISONS) + len(F.CHALLENGERS) + len(VR.TIER_S) + len(VR.TIER_S_ARRAYS)
          and rep["dev_passes"] == 3 * len(SHIFT) + 1 == rep["dev_passes_by_kind"]["family_scores"]
          and "| P1 | dev |" in md and rep["table"], (rep["dev_passes"], rep.get("dev_passes_by_kind")))
    r2, c3, p4 = rep["comparisons"]["P2"], rep["comparisons"]["C_D3"], rep["comparisons"]["P4"]
    check("the report's P2: D the plain D, D_sel the deciding one (z from D_sel and SE_sel), the plain SE and E3-B's "
          "share of the resamples beside; the markdown row shows both",
          r2["plain"]["diff"] == p["P2"]["plain"]["diff"] and r2["decision"]["diff"] == p["P2"]["selection"]["diff_sel"]
          and r2["selection"]["se_sel"] == p["P2"]["selection"]["se_sel"] and "share_B_chosen" in r2["plain"]
          and "| D | D_sel |" in md and "Selection-corrected" in md
          and ("| P2 | dev | E3B | D4F | %.4f | %.4f |" % (r2["plain"]["diff"], r2["decision"]["diff"])) in md,
          {k: r2.get(k) for k in ("plain", "selection")})
    check("C_D3 is P4's record (V3F.5 D3: the same comparison): same D, SE, seed text and z; no second bootstrap",
          c3["same_as"] == "P4" and c3["decision"] == p4["decision"] and c3["image"] == p4["image"]
          and doc["challengers"]["comparisons"]["D3"]["image"]["seed_text"] == "inc2/v3fam/P4/image", c3.get("image"))
    ts = json.loads((C.INC_DIR / "capacity" / VR.EXTERNAL).read_text())["tier_s"]
    check("Tier S's array readings: D1's species subsets on ood22 (8 species) and ood23 (3) decided with the larger "
          "SE; the stage-1 readings and D5b's not run here, with their reasons (no labels, no D5b scores)",
          ts["S_tsb_over_a12_ood22_8sp"]["status"] == "decided"
          and ts["S_tsb_over_a12_ood22_8sp"]["species"] == list(VR.OOD22_SPECIES_100)
          and ts["S_tsb_over_a12_ood23_3sp"]["status"] == "decided" and ts["S_tsb_over_a12_ood23_3sp"]["se_rule"] == "max"
          and doc["tier_s"]["S_s1_tsf_over_e1a"]["status"] == "not run"
          and ts["S_d5b_copy_hflip"]["status"] == "not run" and "S_d3_stage1_over_e3b" in doc["tier_s"],
          {k: (v.get("status"), str(v.get("why"))[:80]) for k, v in ts.items()})
    return doc


def t_not_run_and_dev_minus_h():
    print("a side not run, and D0-S's dev branch")
    reset()
    build_scores(skip=("D2LAION", "D6"))
    TS._write_json(SYS.score_record_path("D2LAION"), {"format": SYS.FMT_SCORE_RECORD, "arm": "D2LAION",
                                                     "status": "not run", "reason": "not run (download)"})
    sw = F.root() / "sweep" / "record.json"
    sw.parent.mkdir(parents=True, exist_ok=True)
    h = ["dev/dev_000", "dev/dev_005"]
    sw.write_text(json.dumps({"format": "inc2-v3fam-sweep-record/1", "status": "complete", "H": h + ["test/test_001"],
                              "trigger": {"h_dev": h, "dev_branch": True, "h_test": ["test/test_001"],
                                          "test_branch": True, "bclean": True}}))
    TS._write_json(SYS.score_record_path("BCLEAN"), {"format": SYS.FMT_SCORE_RECORD, "arm": "BCLEAN",
                                                    "status": "complete", "exam": "dev", "dev": []})
    import functools
    real = F.comparison
    F.comparison = functools.partial(real, resamples=RESAMPLES)
    try:
        doc = VR.verdict()
    finally:
        F.comparison = real
    p = doc["primaries"]
    check("P1 (its control D2LAION not run: download) and P5 (D6 missing) are 'not run' with the reason; Holm keeps "
          "m = 5; the gate counts C = 14; D6 reads nothing (not scored); BCLEAN's read first (D0-S triggered it), "
          "then D8-m, at most two challengers after them",
          p["P1"]["status"] == "not run" and "download" in p["P1"]["why"] and p["P5"]["status"] == "not run"
          and doc["holm"]["P1"]["m"] == 5 and doc["challengers"]["C"] == 14
          and doc["challengers"]["arms"]["D2LAION"]["status"] == "not run"
          and [o["arm"] for o in doc["reads"]["order"]][:2] == ["BCLEAN", "D8M"]
          and doc["reads"]["bclean_triggered"] is True
          and len([o for o in doc["reads"]["order"] if o["kind"] == "challenger"]) <= 2, (p["P1"].get("why"),
                                                                                       doc["reads"]))
    check("D0-S's dev branch: decided on dev minus H (22 of 24 images), the full dev beside every dev comparison "
          "(P1-P5, Tier C, Tier S on dev)",
          doc["decided_on"] == "dev minus H" and doc["dev_minus_h"]["n_images"] == 22
          and p["P2"]["n_images"] == 22 and doc["full_dev_beside"]["P2"]["status"] == "decided"
          and all("C_%s" % a in doc["full_dev_beside"] for a in F.CHALLENGERS)
          and doc["full_dev_beside"]["C_D4F"]["status"] == "decided"
          and doc["full_dev_beside"]["S_ref_over_d8m"]["status"] == "decided"
          and doc["full_dev_beside"]["C_D3"]["same_as"] == "P4"
          and doc["challengers"]["comparisons"]["D4F"]["n_images"] == 22, sorted(doc["full_dev_beside"])[:6])
    e3 = doc["e3_reread"]
    xr = [a for _d, a in SYS.family_runs("E3B", "dev")]
    yr = [a for _d, a in SYS.family_runs("REF", "dev")]
    keep = VR.dev_keep(VR.d0s_trigger())
    want = F.comparison("E3_B_over_ref", [d["species_map50_95"] for d, _a in SYS.family_runs("E3B", "dev")],
                        [d["species_map50_95"] for d, _a in SYS.family_runs("REF", "dev")], xr, yr,
                        sessions=VR.sessions("dev"), resamples=RESAMPLES, keep=keep,
                        image_seed_text=TS.SPECIES_SEED_TEXT)
    check("E3's recorded dev comparisons re-read on dev minus H from E3RS's arrays (E3-M, E3-A, E3-B over the "
          "reference under E3's species seed text, E3-B over E3-A under its attribution text), the full dev beside, "
          "reported only", sorted(e3["dev_minus_h"]) == ["E3_A_over_ref", "E3_B_over_A", "E3_B_over_ref",
                                                         "E3_M_over_ref"]
          and e3["dev_minus_h"]["E3_B_over_ref"]["n_images"] == 22
          and abs(e3["dev_minus_h"]["E3_B_over_ref"]["diff"] - want["diff"]) < 1e-12
          and e3["dev_minus_h"]["E3_B_over_ref"]["image"]["seed_text"] == "inc2/e3/species_se"
          and e3["dev_minus_h"]["E3_B_over_A"]["image"]["seed_text"] == "inc2/e3/attribution_se"
          and e3["full_dev"]["E3_M_over_ref"]["status"] == "decided"
          and "E3_B_over_ref" not in doc["primaries"], sorted(e3.get("dev_minus_h") or {}))


def write_verdict(order=(("D6", "reference"), ("D8M", "reference"), ("D4N", "challenger"))):
    vp = VR.verdict_path()
    vp.parent.mkdir(parents=True, exist_ok=True)
    if vp.exists():
        vp.unlink()
    vp.write_text(json.dumps({"format": VR.FMT_VERDICT, "status": "decided", "exam": "dev",
                              "reads": {"order": [{"arm": a, "kind": k} for a, k in order], "cap": 5}}))


def t_reads():
    print("the sealed-test reads: the read path's guards")
    reset()
    srcs = {"D6": {"mode": "inject_rows", "seeds": [{"seed": s, "sha256": "%d" % s * 64} for s in F.SEEDS]},
            "D8M": {"mode": "identity", "seeds": [{"seed": s} for s in F.SEEDS]},
            "D4N": {"mode": "two_stage", "seeds": []}}
    real = VR.read_sources
    VR.read_sources = lambda arm: dict(srcs[arm])
    try:
        e0 = refused(VR.test_read, "D6", "delegate")
        write_verdict()
        e1 = refused(VR.test_read, "D6", "someone")
        e2 = refused(VR.test_read, "D8M", "delegate")
        e3 = refused(VR.test_read, "D5B", "owner")
        check("test-read refuses before the verdict, a read_by other than owner or delegate, an arm out of turn "
              "(D8-m before D6), an arm with no read", all(x is not None for x in (e0, e1, e2, e3)), (e0, e1, e2, e3))
        rec = VR.test_read("D6", "delegate")
        rj = json.loads((VR.test_dir("D6") / "read.json").read_text())
        check("D6's read prepared: read.json once with read_by exactly the delegate's name, the kind and position, "
              "the verdict and the sources pinned, the one argv (run_inc2_build.sh inc2.v3fam score-test --arm D6)",
              rj["read_by"] == "Claude (delegate under the 2026-09-30 owner grant)" and rj["kind"] == "reference"
              and rj["position"] == 1 and rj["verdict_sha256"] == F._sha(VR.verdict_path())
              and rj["argv"][-4:] == ["inc2.v3fam", "score-test", "--arm", "D6"]
              and rj["incap_decided_by"] == "assistant:claude-delegate" and rec["sources"] == srcs["D6"], rj)
        e4 = refused(VR.test_read, "D6", "owner")
        check("  a second read of D6 refuses", e4 is not None, e4)
        rec2 = VR.test_read("D8M", "owner")
        check("  then D8-m, read by the owner ('Harry He (owner)')", rec2["read_by"] == "Harry He (owner)"
              and rec2["incap_decided_by"].startswith("the owner's account"), rec2["read_by"])
        # the family cap: provenance of family jobs near 260 GPU-h
        prov = C.INC_DIR / "_campaign" / "provenance" / "v3fam_score_d9.json"
        prov.parent.mkdir(parents=True, exist_ok=True)
        prov.write_text(json.dumps({"attempts": [{"started_utc": "2026-10-12T00:00:00Z",
                                                  "finished_utc": "2026-10-22T19:42:00Z"}]}))
        sp = VR.spend()
        e5 = None
        try:
            VR.test_read("D4N", "delegate")
        except F.V3famCapRefused as ex:
            e5 = ex
        capf = F.cap_dir() / "test_read_D4N.json"
        check("the family cap: with %.2f GPU-h settled (a job's wall time), D4N's read (0.5) would pass 260: refused "
              "'not run (family cap)', recorded, no read.json" % sp["gpuh"],
              e5 is not None and capf.exists() and json.loads(capf.read_text())["status"] == F.NOT_RUN_CAP
              and not (VR.test_dir("D4N") / "read.json").exists(), (sp["gpuh"], e5))
        e6 = refused(VR.test_read, "D4N", "delegate")
        check("  and never retried: a second test-read of D4N refuses (it was not run, family cap)",
              e6 is not None and "family cap" in str(e6), e6)
        prov.unlink()
    finally:
        VR.read_sources = real


def t_submitter_and_score_test():
    print("score-test's guards: the submitter, the verdict and the sources")
    for key, env, ok in (("delegate", "assistant:claude-delegate", True), ("delegate", "human:owner@x", False),
                         ("owner", "human:owner@x", True), ("owner", "assistant:claude-delegate", False),
                         ("owner", "human:", False), ("owner", "", False), ("delegate", None, False)):
        e = refused(VR.check_submitter, key, {"INCAP_DECIDED_BY": env} if env is not None else {})
        check("read_by %s, INCAP_DECIDED_BY %r: %s" % (key, env, "accepted" if ok else "refused"),
              (e is None) == ok, e)
    real = VR.read_sources
    VR.read_sources = lambda arm: {"mode": "identity", "seeds": [{"seed": 0}]}
    try:
        e1 = refused(VR.score_test, "D4F", env={"INCAP_DECIDED_BY": "assistant:claude-delegate"})
        e2 = refused(VR.score_test, "D6", env={"INCAP_DECIDED_BY": "human:someone"})
        check("score-test refuses an arm with no prepared read, and a submitter other than the one read_by names "
              "(D6's read is the delegate's)", e1 is not None and e2 is not None and "INCAP_DECIDED_BY" in str(e2),
              (e1, e2))
        VR.read_sources = lambda arm: {"mode": "identity", "seeds": [{"seed": 0}], "changed": True}
        e3 = refused(VR.score_test, "D6", env={"INCAP_DECIDED_BY": "assistant:claude-delegate"})
        check("  and sources changed since the read was prepared", e3 is not None and "changed" in str(e3), e3)
    finally:
        VR.read_sources = real
    write_verdict(order=(("D6", "reference"), ("D8M", "reference"), ("D4F", "challenger")))
    e4 = refused(VR.score_test, "D6", env={"INCAP_DECIDED_BY": "assistant:claude-delegate"})
    check("  and a verdict changed since the read was prepared", e4 is not None, e4)


def t_report_and_retest():
    print("test-report's stratum and gap; the re-analysis")
    build_manifests()
    for s in F.SEEDS:
        for d, sh in (("D6", 0.03), (VR.REF_TEST_DIR, 0.0)):
            arrays = synth("test", s, sh)
            doc = {"format": F.FMT_SCORE, "exam": "test", "seed": s, "species_map50_95": F.statistic_value(arrays),
                   "agnostic_map50_95": 0.9}
            F.commit_pass(VR.test_dir(d) / ("s%d" % s), "test", doc, {"arrays": arrays})
    rep = VR.test_report("D6")
    strat = F.dev_session_stratum(SYS.exam_rows("test"), SYS.exam_rows("dev"))
    check("test-report: mean +- sd, the gap to 0.90, the stratum of the test images from dev's sessions (%d of %d) "
          "from the same arrays, read_by and the kind" % (len(strat), SIZES["test"]),
          rep["arm_test"]["stratum"]["n_images"] == len(strat) == 10
          and abs(rep["arm_test"]["gap_to_target"] - (0.90 - rep["arm_test"]["mean"])) < 1e-12
          and rep["read_by"] == "Claude (delegate under the 2026-09-30 owner grant)" and rep["kind"] == "reference"
          and rep["reference_test"]["mean"] is not None
          and (C.INC_DIR / "capacity" / "v3fam_test_d6.md").is_file(), rep["arm_test"])
    for sub in ("A", "M", "B", "reference"):
        for s in F.SEEDS:
            arrays = synth("test", s, 0.01)
            d = TS.root() / "test" / sub / ("s%d" % s)
            d.mkdir(parents=True, exist_ok=True)
            jn = "test.identity.json" if sub == "reference" else "test.json"
            sha = SC.save_npz(d / "x.images.npz", arrays)
            (d / jn).write_text(json.dumps({"images": {"name": "x.images.npz", "sha256": sha}}))
    e0 = refused(VR.reanalyse_test, "delegate", env={"INCAP_DECIDED_BY": "human:x"})
    write_verdict(order=(("D6", "reference"), ("D8M", "reference"), ("D4F", "challenger")))
    ep = refused(VR.reanalyse_test, "delegate", env={"INCAP_DECIDED_BY": "assistant:claude-delegate"})
    check("the re-analysis is the last of the reads: refused while D4F's read (third in the order) is neither "
          "prepared nor recorded not run, writing nothing", ep is not None and "D4F" in str(ep)
          and not (C.INC_DIR / "capacity" / VR.RETEST).exists(), ep)
    F.write_once(F.cap_dir() / "test_read_D4F.json", {"format": F.FMT_CAP, "status": F.NOT_RUN_CAP})
    pdir = C.INC_DIR / "_campaign" / "provenance"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "v3fam_retest.json").write_text(json.dumps({"attempts": [{"job_id": "777",
                                                                      "started_utc": "2026-10-12T00:00:00Z"}]}))
    near = 260.0 - 0.1 - 2 * VR.READ_GPUH - 0.05            # D6's and D8-m's prepared reads count at 0.5 each
    (pdir / "v3fam_score_d9.json").write_text(json.dumps({"attempts": [
        {"started_utc": "2026-10-12T00:00:00Z", "finished_utc": "2026-10-12T00:00:00Z"}]}))
    real = VR._hours
    VR._hours = lambda a, b: near if (a, b) == ("2026-10-12T00:00:00Z", "2026-10-12T00:00:00Z") else real(a, b)
    try:
        rec = VR.reanalyse_test("delegate", env={"INCAP_DECIDED_BY": "assistant:claude-delegate",
                                                 "SLURM_JOB_ID": "777"})
    finally:
        VR._hours = real
    e1 = refused(VR.reanalyse_test, "delegate", env={"INCAP_DECIDED_BY": "assistant:claude-delegate"})
    check("the re-analysis: refused for another submitter; once every read is prepared or recorded not run, O4's "
          "stratum of E3-A, E3-M, E3-B and the reference from E3's stored arrays (no inference), its own running "
          "attempt not counted twice against the cap (%.2f + 0.1 fits); once" % (near + 2 * VR.READ_GPUH),
          e0 is not None and rec["stratum_images"] == 10 and rec["reads_before"] == ["D6", "D8M", "D4F"]
          and all(rec["arms"][k]["status"] == "ok" for k in ("A", "M", "B", "reference")) and e1 is not None,
          (e0, rec.get("arms", {}).get("A")))
    for n in ("v3fam_retest.json", "v3fam_score_d9.json"):
        (pdir / n).unlink()


def real_dev():
    """Real dev images (64 x 48 JPEGs) and labels (two boxes each), the dev manifest rewritten to them."""
    from PIL import Image
    d = TMP / "real_dev"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, k in enumerate(keys_for("dev", SIZES["dev"])):
        ip, lp = d / "images" / ("%s.jpg" % k), d / "labels" / ("%s.txt" % k)
        Image.new("RGB", (64, 48), ((7 * i) % 255, 90, 40)).save(str(ip))
        lp.write_text("0 0.25 0.5 0.2 0.4\n1 0.7 0.5 0.3 0.5\n")
        rows.append({"key": k, "image": str(ip), "label": str(lp), "label_sha256": C.sha256_file(lp),
                     "sha256": "%064x" % i, "session": DEV_SESSIONS[i % 8], "source": "s"})
    write_manifest(C.manifest_path("dev"), rows)
    VR._GT.clear()
    return rows


GT_PX = np.asarray([[9.6, 14.4, 22.4, 33.6], [35.2, 12.0, 54.4, 36.0]])


def put_s1(dirname, seed, jitter, offset=0.0, ens=None, n_boxes=None):
    """A dev score whose stage-1 record holds, per image, both GT boxes moved by `jitter` px and one false box;
    its recorded stage-1 agnostic AP the validator's way (Ultralytics' match, collapsed_ap on the rows as they
    are), plus `offset`."""
    import torch
    from ultralytics.utils.metrics import box_iou
    from weed_optimizer_framework.tools.inc import scorer as S
    keys = sorted(keys_for("dev", SIZES["dev"]))
    rng = np.random.default_rng(31 + int(1000 * jitter) + (seed or 0))
    xyxy, q, img, tps = [], [], [], []
    match = F.default_match()
    for i, _k in enumerate(keys):
        b = np.vstack([GT_PX + rng.normal(0, jitter, GT_PX.shape), [[0.0, 0.0, 6.0, 6.0]]])
        c = np.round(rng.random(3), 3)
        xyxy.append(b)
        q.append(c)
        img.append(np.full(3, i, dtype=np.int64))
        tps.append(np.asarray(match(box_iou(torch.as_tensor(GT_PX), torch.as_tensor(b))), dtype=bool))
    s1 = {"keys": np.asarray(keys), "img": np.concatenate(img), "xyxy": np.concatenate(xyxy), "q": np.concatenate(q)}
    rec = S.collapsed_ap(np.concatenate(tps, 0), np.concatenate(q), 2 * len(keys))[0]
    d = SYS.system_dir(dirname, seed, ens)
    shutil.rmtree(str(d), ignore_errors=True)
    arrays = synth("dev", seed if seed is not None else 9, 0.0)
    doc = {"format": F.FMT_SCORE, "family": F.NAME, "dir": dirname, "seed": seed, "exam": "dev", "variant": None,
           "species_map50_95": F.statistic_value(arrays, "species"), "stage1": {"agnostic_map50_95": rec + offset},
           "n_boxes": 2 * len(keys) if n_boxes is None else n_boxes}
    F.commit_pass(d, "dev", doc, {"arrays": arrays, "stage1": s1})
    return rec


def t_arrays_and_cap_argv():
    print("Tier S's array readings, D0-S's declaration, the ticker's cap refusals")
    import functools
    reset()
    build_manifests()
    build_scores()
    real_dev()
    rec = {}
    for name, jit in (("D4F", 1.0), ("E3A", 3.0), ("E3B", 2.0)):
        rec[name] = [put_s1(name, s, jit + 0.2 * s) for s in F.SEEDS]
    rec["D3"] = [put_s1("D3", None, 0.8, ens="ens")]
    real = F.comparison
    F.comparison = functools.partial(real, resamples=RESAMPLES)
    try:
        c1 = VR.compare("S_s1_tsf_over_e1a", "dev", "D4F", "E3A", "image", "stage1")
        c2 = VR.compare("S_d3_stage1_over_e3b", "dev", "D3", "E3B", "one_system", "stage1")
        put_s1("E3A", 0, 3.0, offset=0.01)
        c3 = VR.compare("S_s1_tsf_over_e1a", "dev", "D4F", "E3A", "image", "stage1")
        put_s1("E3B", 2, 2.4, n_boxes=5)
        c3b = VR.compare("S_s1_tsf_over_e1b", "dev", "D4F", "E3B", "image", "stage1")
        e_rm = refused(VR.copy_keys, "hflip")
        plan = [[k, i, VR.COPY_FAMILIES[i % 4]] for i, k in enumerate(sorted(keys_for("probe", SIZES["probe"])))]
        drop = "v3fam_d5b_copy__%s" % plan[3][0]
        from weed_optimizer_framework.tools.inc2 import v3fam_data as VD
        TS._write_json(VD._summary_path(), {"format": VD.FORMAT, "status": "complete",
                                            "copies": {"plan": plan, "dropped": [drop]},
                                            "d0s": {"removal_keys": ["b1", "b2", "nd1"], "removal_sha256": "ab" * 32,
                                                    "d6_staging_removals": ["nd1"]}})
        c4 = VR.run_spec(("S_d5b_copy_hflip", "probe", "D5B100", "D5B0", "max", "copy:hflip", None, None), None)
        c5 = VR.run_spec(("S_d5b_copy_jpeg75", "probe", "D5B100", "D5B0", "max", "copy:jpeg75", None, None), None)
    finally:
        F.comparison = real
    check("stage-1 agnostic readings: each side's committed stage-1 boxes matched again to dev's labels equal the "
          "recorded value (within %g); D4's detector against E1-A's boxes decided with the image rule; D3's fused "
          "set against E3-B's seeds as one system, each seed's difference beside" % VR.STAGE1_RECOMPUTE_TOL,
          c1["status"] == "decided" and c1["statistic"] == "agnostic"
          and max(abs(a - b) for a, b in zip(c1["x_values"], rec["D4F"])) <= VR.STAGE1_RECOMPUTE_TOL
          and c1["stage1_recompute"]["x"]["max_abs_diff"] <= VR.STAGE1_RECOMPUTE_TOL
          and c2["status"] == "decided" and c2["decision"]["one_system"] and len(c2["x_minus_each_y"]) == 3,
          (c1.get("why"), c2.get("why"), c1.get("x_values"), rec["D4F"]))
    check("  a recorded stage-1 value 0.01 off what its boxes give, or a score that counted other GT boxes than the "
          "labels hold: the reading is not run, with the reason", c3["status"] == "not run" and "apart" in str(c3["why"])
          and c3b["status"] == "not run" and "labels hold" in str(c3b["why"]), (c3.get("why"), c3b.get("why")))
    fam_keys = [k for k, _i, f in plan if f == "hflip" and "v3fam_d5b_copy__%s" % k != drop]
    check("D5b per copy family: refused without the family data summary; then dose 1.0 over dose 0 on the probe "
          "images whose kept copy is of that family (hflip: %d of 5, one copy dropped), the larger SE" % len(fam_keys),
          e_rm is not None and c4["status"] == "decided" and c4["n_images"] == len(fam_keys) == 4
          and c4["subset"]["n_images"] == 4 and c4["se_rule"] == "max" and c5["n_images"] == 5,
          (e_rm, c4.get("why"), c4.get("n_images"), c5.get("n_images")))
    rm = VR.d0s_removals()
    d1 = VR.declared("E3B", "D4F", rm)
    d2 = VR.declared("D2LAION", "E3B", rm)
    d3 = VR.declared("REF", "D8M", rm)
    d4 = VR.declared("D6", "REF", dict(rm, base_v2_rows=0))
    d5 = VR.declared("D6", "REF", rm)
    check("D0-S's removals declared beside: a family-trained side lost the rows, the comparator was not retrained; "
          "E3's crops keep the base_v2 rows' crops (Revision 31); nothing for two comparators off E3's crops, nor for "
          "a base_v2-built side (D6) when no removed row is base_v2's",
          rm == {"rows": 3, "sha256": "ab" * 32, "base_v2_rows": 1} and "TS-F" in d1 and "not retrained" in d1
          and "Revision 31" in d2 and d3 is None and d4 is None and "RF-DETR-L" in (d5 or "")
          and VR.declared("E3B", "D4F", None) is None, (rm, d1, d2, d3, d4, d5))
    got = VR.record_ticker_refusals(["build:D4F", "score:D2CLIP", ""])
    e_bad = refused(VR.record_ticker_refusals, ["score:../x"])
    sd = VR.side("D4F", "dev")
    check("the ticker's cap refusals passed in the verdict's argv: written once under v3fam/v1/cap (an existing one "
          "kept), so a refused build's score arm and a refused score arm read 'not run (family cap)'; a name that is "
          "not a family item refuses", got == ["build:D4F", "score:D2CLIP"] and VR.cap_refused("build:D4F")
          and VR.arm_status("D4F") == F.NOT_RUN_CAP and VR.arm_status("D2CLIP") == F.NOT_RUN_CAP
          and VR.arm_status("D4N") == "complete" and VR.record_ticker_refusals(["build:D4F"]) == ["build:D4F"]
          and e_bad is not None, (got, VR.arm_status("D4F"), sd.get("why"), e_bad))
    VR.record_ticker_refusals(["data"])
    check("  a refused data build makes every arm that scores a build or D6 'not run (family cap)', never E3RS or D2",
          VR.arm_status("D1") == F.NOT_RUN_CAP and VR.arm_status("D6") == F.NOT_RUN_CAP
          and VR.arm_status("E3RS") == "complete" and VR.arm_status("D2LAION") == "complete",
          {a: VR.arm_status(a) for a in ("D1", "D6", "E3RS", "D2LAION")})
    base = VR.dev_passes()["by_kind"]
    F.write_once(F.equiv_dir() / "dev_identity.json", {"per": {"0": {}, "1": {}, "2": {}}})
    F.write_once(F.equiv_dir() / "injection.json", {"per": {"0": {}}})
    for k in ("LAION", "CLIP"):
        F.write_once(F.root() / "d2" / k / "dev_gt.json", {"top1": 0.9})
    F.write_once(F.root() / "d2" / "FT" / "dev_gt.s0.json", {"top1": 0.9})
    TS._write_json(TS.root() / "sensitivity.json", {"arms": {a: {"values": [0.1] * 5} for a in ("M", "A", "B")}})
    for exp in ("v3_d5b_0_m640", "v3_d4_f_m640"):
        for rid in ("base__s0", "final__base__s0"):
            p_ = C.INC_DIR / exp / "runs" / rid / "scores" / "dev.json"
            p_.parent.mkdir(parents=True, exist_ok=True)
            p_.write_text("{}")
    dp = VR.dev_passes()
    check("the dev pass count by kind (V3F.3.5): the committed family scores, D2's ground-truth passes (3), D9a's "
          "fold passes (15), the reproductions (dev identity 3, injection 2) and the builds' protocol dev scores, "
          "D5b's included (4)", dp["by_kind"] == dict(base, d2_gt=3, d9a=15, reproductions=5, builds_protocol=4)
          and dp["total"] == sum(dp["by_kind"].values()), dp)
    (TS.root() / "sensitivity.json").unlink()
    for exp in ("v3_d5b_0_m640", "v3_d4_f_m640"):
        shutil.rmtree(str(C.INC_DIR / exp))
    VD._summary_path().unlink()
    shutil.rmtree(str(F.cap_dir()), ignore_errors=True)
    build_manifests()
    VR._GT.clear()


def t_spend():
    print("the family's GPU-h from the cluster's records")
    pdir = C.INC_DIR / "_campaign" / "provenance"
    shutil.rmtree(str(pdir), ignore_errors=True)
    pdir.mkdir(parents=True)
    (pdir / "v3fam_score_e3rs.json").write_text(json.dumps({"attempts": [
        {"started_utc": "2026-10-12T00:00:00Z", "finished_utc": "2026-10-12T03:00:00Z"}]}))
    (pdir / "v3fam_v1.json").write_text(json.dumps({"attempts": [{"started_utc": "2026-10-12T04:00:00Z"}]}))
    (pdir / "e3_score_m.json").write_text(json.dumps({"attempts": [
        {"started_utc": "2026-10-12T00:00:00Z", "finished_utc": "2026-10-13T00:00:00Z"}]}))
    shutil.rmtree(str(F.root() / "test"), ignore_errors=True)
    sp = VR.spend()
    check("spend: a family job's finished attempt at its wall time (3 h), a running one at its estimate (the verdict's "
          "0.5), never a non-family job (E3's)", abs(sp["gpuh"] - 3.5) < 1e-9 and len(sp["items"]) == 2, sp)
    ps = VR.person_spend()
    check("  the person steps alone: none here", ps["gpuh"] == 0.0 and ps["items"] == [], ps)
    (pdir / "v3fam_score_e3rs.json").write_text(json.dumps({"attempts": [
        {"started_utc": "2026-10-12T00:00:00Z", "finished_utc": "2026-10-12T03:00:00Z"},
        {"started_utc": "2026-10-12T04:00:00Z", "finished_utc": "2026-10-12T05:00:00Z"}]}))
    sp2 = VR.spend()
    check("  a retried job: both attempts count (3 h + 1 h), the first never replaced by the second",
          abs(sp2["gpuh"] - 4.5) < 1e-9, sp2)
    for arm in ("D6", "D8M"):
        F.write_once(VR.test_dir(arm) / "read.json", {"format": VR.FMT_READ, "arm": arm})
    sp3 = VR.spend()
    (pdir / "v3famtest_d6.json").write_text(json.dumps({"attempts": [{"started_utc": "2026-10-12T06:00:00Z"}]}))
    sp4 = VR.spend()
    ps4 = VR.person_spend()
    sp5 = VR.spend_items(exclude_job="91")
    (pdir / "v3fam_v1.json").write_text(json.dumps({"attempts": [{"job_id": "91",
                                                                  "started_utc": "2026-10-12T04:00:00Z"}]}))
    sp6 = VR.spend_items(exclude_job="91")
    check("  reads prepared by test-read and not yet submitted count at their estimate (0.5 each); once a read's "
          "score-test job has an attempt, that attempt counts instead (never both); the asking job's own running "
          "attempt is left out", abs(sp3["gpuh"] - 5.5) < 1e-9 and abs(sp4["gpuh"] - 5.5) < 1e-9
          and sum(1 for i in sp4["items"] if i["item"].endswith(":prepared")) == 1
          and abs(ps4["gpuh"] - 1.0) < 1e-9 and len(sp6) == len(sp5) - 1, (sp3["gpuh"], sp4["gpuh"], ps4))
    shutil.rmtree(str(F.root() / "test"), ignore_errors=True)


def main():
    for fn in (t_verdict, t_not_run_and_dev_minus_h, t_reads, t_submitter_and_score_test, t_report_and_retest,
               t_arrays_and_cap_argv, t_spend):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            import traceback
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:]))
    print("\n%d failure(s)" % len(FAILURES))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
