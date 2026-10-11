"""V3F's verdict and its sealed-test reads (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered
as one family)", V3F.3.3-V3F.3.6, V3F.5, V3F.6.1).

    python -m weed_optimizer_framework.tools.inc2.v3fam verdict                                  (L23L; a job)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-read --arm X --read-by owner|delegate  (login node)
    sbatch run_inc2_build.sh inc2.v3fam score-test --arm X        (the person test-read names; INCAP_DECIDED_BY)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-report --arm X
    sbatch run_inc2_build.sh inc2.v3fam reanalyse-test --read-by owner|delegate   (a person's; INCAP_DECIDED_BY)
    python -m weed_optimizer_framework.tools.inc2.v3fam spend      (the family's GPU-h as the cluster's records give it)

The verdict (L23L, once every arm's record is complete or failed). Every
side of every comparison is a set of committed family scores
(inc2.v3fam_systems.family_runs: each JSON and its per-image arrays checked
against the sha256 it records), so both sides went through the one family
path (decision 17). Each pre-declared comparison (COMPARISONS) is computed
by inc2.v3fam.comparison: D (D_sel for P2, P3, P5), 2 pooled sd, the image
SE, the session SE (dev's 8 sessions reported beside; deciding as the larger
of the two for P6-P9), z, the 90 % interval and the outcome class; a side
whose arm was not run, failed or was refused at the family cap is "not run"
with its reason, and keeps its place (Holm's m, C = 14). Then Holm per exam
(inc2.v3fam.holm_overlay), the Tier C gate (challenger_gate: plain D over
E3-B, z > Phi^-1(1 - 0.05/14)) and the order of the sealed-test reads
(read_order: BCLEAN if D0-S triggered it, the reference reads D6 then D8-m
when their arms were scored, the challengers that open a read). When D0-S
found a dev image with a true copy, every dev comparison is decided on dev
minus H (the full dev beside).

Records (V3F.3.5): capacity/v3fam_v1.json, the platform's (dev only: P1-P5,
Holm on dev, Tier C, the read order, Tier S on dev, the dev pass count),
written once and last; capacity/v3fam_v1_external.json (P6-P9, Holm on
ood22, ood23 and the probe, Tier S on those exams); capacity/v3fam_ext_v1.json
(ImageWeeds agnostic and Ragweed AP, ooddev_v1 agnostic, per arm, reported);
capacity/v3fam_v1_report.{json,md} (the family table). The platform's
evidence admits only the first.

Sealed-test reads (V3F.3.6). test-read checks the verdict, that the arm is
next in the read order (check_read; a read refused at the family cap is
recorded "not run (family cap)" and not retried), the family cap
(check_cap over the family's GPU-h as the cluster's records give it: spend
items), pins the verdict, the arm's weights and classifier by sha256, and
writes v3fam/v1/test/<X>/read.json once with read_by exactly "Harry He
(owner)" or "Claude (delegate under the 2026-09-30 owner grant)" and the one
argv. score-test refuses unless INCAP_DECIDED_BY names that person (the
owner's account, human:..., or assistant:claude-delegate for the delegate),
then reproduces b_v2_m640's three recorded test scores through the family
identity path (1e-6, 0.002 as a recorded deviation, else refused before any
family test score), then scores the arm once (written once; a second run
refuses). test-report states the gap to 0.90, the stratum of the test images
from dev's sessions (dev_session_stratum), read_by and the kind of read
(capacity/v3fam_test_<x>.{json,md}). reanalyse-test (no inference) reads E3's
stored test arrays: O4's stratum for E3-A, E3-M, E3-B and the reference, and
test minus H when D0-S found test copies (capacity/v3fam_retest.json).

A refusal raises inc2.v3fam.V3famRefused (V3famCapRefused at the family cap).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import recipes as RC
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_systems as SYS

# ------------------------------------------------------------------ constants (pre-registered)
VERDICT = "v3fam_v1.json"                    # under capacity/: the platform's record (dev only)
EXTERNAL = "v3fam_v1_external.json"
EXT = "v3fam_ext_v1.json"
REPORT = "v3fam_v1_report"
RETEST = "v3fam_retest.json"
TEST_RECORD = "v3fam_test_%s"                # .json / .md, lower-case arm
FMT_VERDICT = "inc2-v3fam-verdict/1"
FMT_EXTERNAL = "inc2-v3fam-external/1"
FMT_EXT = "inc2-v3fam-ext/1"
FMT_REPORT = "inc2-v3fam-report/1"
FMT_READ = "inc2-v3fam-test-read/1"
FMT_TEST_REPORT = "inc2-v3fam-test-report/1"
FMT_RETEST = "inc2-v3fam-retest/1"
TARGET = 0.90
READ_GPUH = 0.5                              # one test read (V3F.7 step 12)
RETEST_GPUH = 0.1
BCLEAN_SEED_TEXT = "inc2/v3fam/bclean/species_se"     # BCLEAN's rule against b_v2_m640 (V3F.5 D0-S)
DECIDED_BY_OWNER_PREFIX = "human:"
DELEGATE_ACCOUNT = F.DECIDED_BY_ENV["delegate"]
REF_TEST_DIR = "REF"

# The sides (V3F.5): {name: (arms/<dir>, variant, one-system ens or None, the score arm whose record it is)}
SIDES = {"E3M": ("E3M", "", None, "E3RS"), "E3A": ("E3A", "", None, "E3RS"), "E3B": ("E3B", "", None, "E3RS"),
         "REF": ("REF", "", None, "E3RS"), "ORACLE": ("E3B", SYS.ORACLE_VARIANT, None, "E3RS"),
         "D3": ("D3", "", "ens", "D3"), "D6": ("D6", "", None, "D6"), "D8S": ("D8S", "", None, "D8"),
         "D8M": ("D8M", "", None, "D8"), "BCLEAN": ("BCLEAN", "", None, "BCLEAN"),
         "D1A1": ("D1A1", "", None, "D1"), "D1B1": ("D1B1", "", None, "D1"), "D1A12": ("D1A12", "", None, "D1"),
         "TSA": ("TSA", "", None, "D1"), "TSB": ("TSB", "", None, "D1"),
         "D5B0": ("D5B0", "", None, "D5B"), "D5B25": ("D5B25", "", None, "D5B"), "D5B100": ("D5B100", "", None, "D5B")}
for _a in ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
           "D4F", "D4N", "D4R15"):
    SIDES[_a] = (_a, "", None, _a)
LABELS = {"E3M": "E3-M", "E3A": "E3-A", "E3B": "E3-B", "REF": "b_v2_m640 (YOLO11m reference)",
          "ORACLE": "E3-B's boxes, perfect naming (O5)", "D2LAION": "LAION-2B CLIP ViT-L/14",
          "D2CLIP": "OpenAI CLIP ViT-L/14", "D2DINOL": "DINOv2-L", "D2DINOB": "DINOv2-B", "D2SIGLIP": "SigLIP2",
          "D2RN50": "ResNet-50", "D2BC1": "BioCLIP-1", "D2BC2640": "BioCLIP-2@640", "D2FT": "FT-ResNet-50",
          "D3": "WBF of E3-B's three stage-1 seeds", "D4F": "TS-F", "D4N": "TS-N", "D4R15": "TS-R15",
          "D6": "RF-DETR-L 12-class", "D8S": "D8-s (YOLO11s train_core)", "D8M": "D8-m (YOLO11m train_core)",
          "BCLEAN": "BCLEAN", "D1A1": "A'1", "D1B1": "B'1", "D1A12": "A'12", "TSA": "TS-A'", "TSB": "TS-B'",
          "D5B0": "dose 0", "D5B25": "dose 0.25", "D5B100": "dose 1.0"}
# the pre-declared comparisons: (id, exam, X, Y, se rule, statistic, tier); statistic species | p7 | probe
COMPARISONS = (
    ("P1", "dev", "E3B", "D2LAION", "image", "species", "P"),
    ("P2", "dev", "E3B", "D4F", "selection", "species", "P"),
    ("P3", "dev", "E3B", "D4N", "selection", "species", "P"),
    ("P4", "dev", "D3", "E3B", "one_system", "species", "P"),
    ("P5", "dev", "E3B", "D6", "selection", "species", "P"),
    ("P6", "ood22", "TSB", "D1A12", "max", "species", "P"),
    ("P7", "ood23", "D1B1", "D1A1", "max", "p7", "P"),
    ("P8", "probe", "D5B100", "D5B0", "max", "probe", "P"),
    ("P9", "probe", "D5B25", "D5B0", "max", "probe", "P"))
CHALLENGER_SIDES = {a: a for a in F.CHALLENGERS}
TIER_S = (
    ("S_bc2_over_bc2640", "dev", "E3B", "D2BC2640", "image"),
    ("S_ft_over_rn50", "dev", "D2FT", "D2RN50", "image"),
    ("S_bc2_over_clip", "dev", "E3B", "D2CLIP", "image"),
    ("S_laion_over_clip", "dev", "D2LAION", "D2CLIP", "image"),
    ("S_dinol_over_clip", "dev", "D2DINOL", "D2CLIP", "image"),
    ("S_bc1_over_clip", "dev", "D2BC1", "D2CLIP", "image"),
    ("S_tsf_over_tsr15", "dev", "D4F", "D4R15", "image"),
    ("S_e3b_over_tsr15", "dev", "E3B", "D4R15", "image"),
    ("S_tsf_over_e3a", "dev", "D4F", "E3A", "image"),
    ("S_e3a_over_d6", "dev", "E3A", "D6", "image"),
    ("S_d6_over_ref", "dev", "D6", "REF", "image"),
    ("S_ref_over_d8m", "dev", "REF", "D8M", "image"),
    ("S_d8m_over_d8s", "dev", "D8M", "D8S", "image"),
    ("S_bclean_over_ref", "dev", "BCLEAN", "REF", "image"),
    ("S_bclean_over_e3b", "dev", "BCLEAN", "E3B", "image"),
    ("S_tsa_over_a12_ood22", "ood22", "TSA", "D1A12", "max"),
    ("S_tsb_over_tsa_ood22", "ood22", "TSB", "TSA", "max"))
PAIRED = ("P1",) + tuple("C_%s" % a for a in F.CHALLENGERS if a.startswith("D2")) + (
    "S_bc2_over_bc2640", "S_ft_over_rn50", "S_bc2_over_clip", "S_laion_over_clip", "S_dinol_over_clip",
    "S_bc1_over_clip")
REPORTED_EXAMS = ("imageweeds", SYS.OODDEV)
FAMILY_ESTIMATES = {"v3fam_data": 2.0, "v3fam_sweep": 1.5, "v3fam_v1": 0.5, "v3fam_detr_smoke": 0.5,
                    "v3fam_detr_s": 24.0, "v3fam_retest": RETEST_GPUH, "v3famtest_": READ_GPUH}
SCORE_ESTIMATE_GPUH = 1.5                    # a family score job without a price of its own here
PERSON_JOBS = ("v3fam_sweep", "v3famtest_", "v3fam_retest")


def log(msg):
    print("[inc2.v3fam] %s" % msg, flush=True)


def cap_path(name):
    return C.INC_DIR / "capacity" / name


def verdict_path():
    return cap_path(VERDICT)


def test_dir(arm):
    return F.root() / "test" / arm


# ------------------------------------------------------------------ the sides
def score_record(arm):
    """The platform's record of a score arm (capacity/v3fam_score_<arm>.json), or None."""
    d = F._read_json(SYS.score_record_path(arm))
    return d if isinstance(d, dict) else None


def cap_refused(item):
    return (F.cap_dir() / ("%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(item)))).exists()


def arm_status(arm):
    """complete | not run (<why>) | failed | missing, from the arm's score record and the family cap's records."""
    if cap_refused("score_%s" % arm) or cap_refused(arm):
        return F.NOT_RUN_CAP
    r = score_record(arm)
    if r is None:
        return "missing"
    st = r.get("status")
    if st == "complete":
        return "complete"
    if st == "not run":
        return "not run (%s)" % (r.get("reason") or "recorded")
    return str(st or "failed")


def side(name, exam):
    """{"status": "ok", "docs", "runs", "species"} of a side's committed scores on exam, or {"status", "why"}."""
    dirname, variant, ens, arm = SIDES[name]
    try:
        got = SYS.family_runs(dirname, exam, variant, ens=ens)
    except F.V3famRefused as e:
        st = arm_status(arm)
        return {"status": "not run" if st != "complete" else "failed", "arm": arm,
                "why": "%s: %s" % (st, str(e)[:300])}
    docs = [d for d, _a in got]
    for d in docs:
        if d.get("exam") != exam or (d.get("variant") or "") != (variant or ""):
            return {"status": "failed", "arm": arm, "why": "%s's file is not a %s score" % (name, exam)}
    return {"status": "ok", "arm": arm, "docs": docs, "runs": [a for _d, a in got],
            "species": [d.get("species_map50_95") for d in docs]}


def p7_runs(name):
    """P7's per-image arrays of a stage1-mode side on ood23 (the target reading, ignored rows left out: the p7 npz
    the score JSON names) and its per-seed values, the collapsed AP50-95 of those arrays (V3F.3.2)."""
    dirname, _v, _e, arm = SIDES[name]
    runs = []
    for s in F.SEEDS:
        _doc, z = SYS.family_side(dirname, s, "ood23", "p7")
        runs.append({"keys": z["t_keys"], "tp": z["t_tp"], "conf": z["t_conf"], "pred_img": z["t_pred_img"],
                     "n_gt": z["t_n_gt"]})
    return runs, [F.statistic_value(a, "agnostic") for a in runs]


def sessions(exam):
    """One session per image in the arrays' key order (V3F.3.1's rule; the probe in its own root)."""
    rows = SYS.exam_rows(exam)
    return F.sessions_for(exam, rows)


def d0s_trigger():
    """D0-S's trigger from its complete record ({} when there is none)."""
    from . import v3fam_sweep as SW
    rec = SW.read_record()
    return (rec or {}).get("trigger") or {}


def dev_keep(trig):
    """The dev keys a dev comparison is decided on: every dev key but those with a true copy (D0-S's dev branch),
    or None when no dev image has one."""
    h = [str(k)[len("dev/"):] for k in trig.get("h_dev") or []]
    if not h:
        return None
    return [r["key"] for r in SYS.exam_rows("dev") if r["key"] not in set(h)]


# ------------------------------------------------------------------ one comparison
def compare(cid, exam, xname, yname, rule, statistic="species", selection=None, keep=None, seed_text=None):
    """One pre-declared comparison on committed family scores (module docstring); {"id", "status": decided |
    not run, ...} with inc2.v3fam.comparison's record when decided."""
    base = {"id": cid, "exam": exam, "x": xname, "y": yname, "x_label": LABELS.get(xname),
            "y_label": LABELS.get(yname), "se_rule": rule}
    try:
        if statistic == "p7":
            xr, xv = p7_runs(xname)
            yr, yv = p7_runs(yname)
            stat, species = "agnostic", SC.SPECIES
        else:
            xs, ys = side(xname, exam), side(yname, exam)
            for nm, sd in ((xname, xs), (yname, ys)):
                if sd["status"] != "ok":
                    return dict(base, status="not run", why="%s: %s" % (nm, sd.get("why")))
            xr, yr = xs["runs"], ys["runs"]
            if statistic == "probe":
                stat, species = "species", F.PROBE_SPECIES
                xv = [F.statistic_value(a, "species", species) for a in xr]
                yv = [F.statistic_value(a, "species", species) for a in yr]
            else:
                stat, species = "species", SC.SPECIES
                xv, yv = xs["species"], ys["species"]
        ses = sessions(exam) if exam in ("dev", "ood22", "ood23", "probe") else None
        sel = None
        if rule == "selection":
            sel = {"runs": {}, "values": {}}
            for a in F.SELECTION_ARMS:
                sd = side("E3%s" % a, exam)
                if sd["status"] != "ok":
                    return dict(base, status="not run", why="the selection correction needs E3-%s: %s"
                                % (a, sd.get("why")))
                sel["runs"][a], sel["values"][a] = sd["runs"], sd["species"]
        rec = F.comparison(cid, xv, yv, xr, yr, statistic=stat, species=species, sessions=ses,
                           se_rule={"one_system": "image"}.get(rule, rule), one_system=rule == "one_system",
                           selection=sel, keep=keep if exam == "dev" else None, paired=cid in PAIRED,
                           image_seed_text=seed_text)
    except F.V3famRefused as e:
        return dict(base, status="not run", why=str(e)[:400])
    for k in ("x_values", "y_values"):
        rec[k] = [float(v) for v in rec[k]]
    sel_rec = rec.get("selection")
    if sel_rec:
        rec["selection"] = {k: v for k, v in sel_rec.items() if k not in ("T", "chosen")}
    return dict(base, status="decided", **rec)


def summary(rec):
    """What Holm, the gate and the report read of a comparison."""
    if rec.get("status") != "decided":
        return {"status": rec.get("status") or "not run", "why": rec.get("why")}
    d = rec["decision"]
    return {"status": "decided", "qualifies": d["qualifies"], "z": d["z"], "diff": d["diff"],
            "outcome": d["outcome"], "flags": d["flags"], "interval_90": d["interval_90"]}


# ------------------------------------------------------------------ the verdict (L23L)
def decide_family(keep=None):
    """Every pre-declared comparison of the family on its committed scores; Holm, the gate, the read order."""
    trig = d0s_trigger()
    keep = dev_keep(trig) if keep is None else keep
    out = {}
    for cid, exam, x, y, rule, stat, _tier in COMPARISONS:
        out[cid] = compare(cid, exam, x, y, rule, stat, keep=keep)
    for a in F.CHALLENGERS:
        cid = "C_%s" % a
        out[cid] = compare(cid, "dev", CHALLENGER_SIDES[a], "E3B", "one_system" if a == "D3" else "image",
                           keep=keep)
    for cid, exam, x, y, rule in TIER_S:
        st = BCLEAN_SEED_TEXT if cid == "S_bclean_over_ref" else None
        out[cid] = compare(cid, exam, x, y, rule, keep=keep, seed_text=st)
    holm = F.holm_overlay({p: summary(out[p]) for p in F.PRIMARIES})
    gate = F.challenger_gate({a: summary(out["C_%s" % a]) for a in F.CHALLENGERS})
    ref_avail = tuple(a for a in F.REFERENCE_READS if arm_status("D8" if a == "D8M" else a) == "complete")
    bclean = bool(trig.get("bclean")) and arm_status("BCLEAN") == "complete"
    order = F.read_order(gate, bclean_triggered=bclean, reference_available=ref_avail)
    full = None
    if keep is not None:
        full = {cid: summary(compare(cid, exam, x, y, rule, stat)) for cid, exam, x, y, rule, stat, _t in
                COMPARISONS if exam == "dev"}
    return {"comparisons": out, "holm": holm, "gate": gate, "order": order, "trigger": trig,
            "decided_on": "dev minus H" if keep is not None else "dev",
            "dev_minus_h": None if keep is None else {"n_images": len(keep),
                                                     "keys_sha256": C.sha256_text("\n".join(sorted(keep)))},
            "full_dev_beside": full}


def dev_pass_count():
    """Every committed family dev score (a dev pass of the family; V3F.3.5's read count)."""
    base = F.root() / "arms"
    if not base.is_dir():
        return 0
    return sum(1 for p in base.glob("*/*/dev*.json") if p.is_file())


def _strip(rec):
    """A comparison record without its large members (the per-resample values)."""
    return {k: v for k, v in rec.items() if k not in ("Vx", "Vy", "diffs", "kept")}


def ext_record():
    """ImageWeeds (agnostic, Ragweed AP) and ooddev_v1 (agnostic) of every side that has them, reported."""
    out = {}
    for name in sorted(SIDES):
        for exam in REPORTED_EXAMS:
            sd = side(name, exam)
            if sd["status"] != "ok":
                continue
            vals = {"agnostic": [d.get("agnostic_map50_95") for d in sd["docs"]],
                    "stage1_agnostic": [(d.get("stage1") or {}).get("agnostic_map50_95") for d in sd["docs"]]}
            if exam == "imageweeds":
                vals["ragweed_ap50_95"] = [(d.get("per_class") or {}).get("Ragweed") for d in sd["docs"]]
            row = {}
            for k, v in vals.items():
                v = [float(x) for x in v if x is not None]
                if v:
                    m, s = F._mean_sd(v)
                    row[k] = {"values": v, "mean": m, "sd": s}
            out.setdefault(name, {})[exam] = row
    return out


def _table():
    """The family table: every side, every exam it holds, every seed (species, agnostic, stage-1 agnostic)."""
    rows = []
    for name in sorted(SIDES):
        for exam in ("dev", "ood22", "ood23", "probe") + REPORTED_EXAMS:
            sd = side(name, exam)
            if sd["status"] != "ok":
                continue
            sp = [d.get("species_map50_95") for d in sd["docs"]]
            rows.append({"side": name, "label": LABELS.get(name), "exam": exam, "seeds": [d.get("seed") for d in
                                                                                          sd["docs"]],
                         "species": sp, "species_mean": F._mean_sd([x for x in sp if x is not None])[0],
                         "agnostic": [d.get("agnostic_map50_95") for d in sd["docs"]],
                         "stage1_agnostic": [(d.get("stage1") or {}).get("agnostic_map50_95") for d in sd["docs"]]})
    return rows


def _md(rep):
    def f(x):
        return "-" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))
    lines = ["# V3F family report (%s)" % rep["generated_utc"], "",
             "Decided on: %s. Dev passes of the family: %d." % (rep["decided_on"], rep["dev_passes"]), "",
             "| id | exam | X | Y | D | 2 sd | SE image | SE session | z | 90 %% interval | outcome | confirmed / "
             "opens |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, r in rep["comparisons"].items():
        if r.get("status") != "decided":
            lines.append("| %s | %s | %s | %s | not run: %s |  |  |  |  |  |  |  |" % (
                cid, r.get("exam"), r.get("x"), r.get("y"), str(r.get("why"))[:80]))
            continue
        d = r["decision"]
        extra = ""
        if cid in rep["holm"]:
            extra = "family-confirmed" if rep["holm"][cid]["family_confirmed"] else "not confirmed"
        if cid.startswith("C_"):
            extra = "opens a read" if (rep["gate"]["arms"].get(cid[2:]) or {}).get("opens_read") else "no read"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | [%s, %s] | %s%s | %s |" % (
            cid, r["exam"], r["x"], r["y"], f(d["diff"]), f(d["two_sd"]), f((r.get("image") or {}).get("se")),
            f((r.get("session") or {}).get("se")), f(d["z"]), f(d["interval_90"][0]), f(d["interval_90"][1]),
            d["outcome"], (" (%s)" % ", ".join(d["flags"])) if d["flags"] else "", extra))
    lines += ["", "Reads in order: %s." % (", ".join("%s (%s)" % (o["arm"], o["kind"]) for o in rep["order"])
                                         or "none"), "",
              "| side | exam | species (seeds) | mean | agnostic | stage-1 agnostic |", "|---|---|---|---|---|---|"]
    for t in rep["table"]:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (t["label"], t["exam"], ", ".join(f(x) for x in t["species"]),
                                                          f(t["species_mean"]), ", ".join(f(x) for x in t["agnostic"]),
                                                          ", ".join(f(x) for x in t["stage1_agnostic"])))
    return "\n".join(lines) + "\n"


def verdict():
    """L23L (module docstring): the family's comparisons on its committed scores; the non-dev records first, the
    platform's dev record (capacity/v3fam_v1.json) once and last."""
    F.check_frozen()
    vp = verdict_path()
    if vp.exists():
        raise F.V3famRefused("%s exists: the family's verdict is written once" % vp)
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        arms = {a: arm_status(a) for a in F.SCORE_ARMS}
        fam = decide_family()
        comps = {k: _strip(v) for k, v in fam["comparisons"].items()}
        holm = fam["holm"]
        dev_ids = [c[0] for c in COMPARISONS if c[1] == "dev"]
        ext_ids = [c[0] for c in COMPARISONS if c[1] != "dev"]
        common = {"family": F.NAME, "decided_by": F.DECIDED_BY, "code": dict(F.code_shas(),
                                                                          v3fam_verdict=C.sha256_file(__file__)),
                  "e_trees": {"files": len(before), "sha256": C.sha256_text(json.dumps(before, sort_keys=True))},
                  "written_utc": F._utc()}
        ext = dict(common, format=FMT_EXTERNAL, status="decided",
                   primaries={p: comps[p] for p in ext_ids}, holm={p: holm[p] for p in ext_ids},
                   tier_s={c[0]: comps[c[0]] for c in TIER_S if c[1] != "dev"},
                   note="P6-P9 and the non-dev Tier S readings: never on the platform's evidence (V3F.3.5)")
        TS._write_json(cap_path(EXTERNAL), ext)
        TS._write_json(cap_path(EXT), dict(common, format=FMT_EXT, arms=ext_record(),
                                           note="ImageWeeds Ragweed AP50-95 (common ragweed, 2,010 boxes; other "
                                                "ImageWeeds species scored only as OtherPlant) and agnostic; "
                                                "ooddev_v1 agnostic; reported only"))
        rep = dict(common, format=FMT_REPORT, decided_on=fam["decided_on"], dev_passes=dev_pass_count(),
                   comparisons={k: dict(summary(v), exam=v.get("exam"), x=v.get("x"), y=v.get("y"),
                                        decision=v.get("decision"), image=v.get("image"), session=v.get("session"),
                                        why=v.get("why")) for k, v in comps.items()},
                   holm=holm, gate=fam["gate"], order=fam["order"], arms=arms, table=_table(),
                   generated_utc=F._utc())
        TS._write_json(cap_path(REPORT + ".json"), rep)
        cap_path(REPORT + ".md").write_text(_md(rep))
        doc = dict(common, format=FMT_VERDICT, status="decided", exam="dev", decided_on=fam["decided_on"],
                   dev_minus_h=fam["dev_minus_h"], full_dev_beside=fam["full_dev_beside"],
                   primaries={p: comps[p] for p in dev_ids}, holm={p: holm[p] for p in dev_ids},
                   challengers=dict(fam["gate"], comparisons={a: comps["C_%s" % a] for a in F.CHALLENGERS}),
                   reads={"order": fam["order"], "cap": F.READ_CAP,
                          "bclean_triggered": any(o["kind"] == "bclean" for o in fam["order"])},
                   tier_s={c[0]: comps[c[0]] for c in TIER_S if c[1] == "dev"},
                   arms=arms, dev_passes=dev_pass_count(),
                   sesoi=F.SESOI, rule="V3F.3.3: D > 2 pooled sd and D > SE; combined z; 90 % interval; outcome "
                                       "classes; Holm per exam; Tier C bar Phi^-1(1 - 0.05/14)",
                   note="dev only; the external primaries in capacity/%s; nothing switches" % EXTERNAL)
        F.write_once(vp, doc)
    log("verdict: %s; reads %s (%.0fs)" % (", ".join("%s %s" % (p, (doc["primaries"].get(p) or {}).get(
        "decision", {}).get("outcome", doc["primaries"].get(p, {}).get("status"))) for p in dev_ids),
        [o["arm"] for o in fam["order"]], time.time() - t0))
    return doc


def read_verdict():
    d = F._read_json(verdict_path())
    if not isinstance(d, dict) or d.get("format") != FMT_VERDICT or d.get("status") != "decided":
        raise F.V3famRefused("the family's verdict (%s) is not decided: no sealed-test read before it (V3F.3.6)"
                             % verdict_path())
    return d


# ------------------------------------------------------------------ the family's GPU-h (decision 25)
def _hours(a, b):
    try:
        t0 = datetime.datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.datetime.strptime(b, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None
    return max(0.0, (t1 - t0).total_seconds() / 3600.0)


def _estimate(name):
    for k, v in FAMILY_ESTIMATES.items():
        if name == k or (k.endswith("_") and name.startswith(k)) or (k == "v3fam_detr_s" and name.startswith(k)):
            return float(v)
    return SCORE_ESTIMATE_GPUH


def family_job(name):
    return name.startswith("v3fam") or name in RC.V3FAM_EXPS.values()


def spend_items(names=None):
    """The family's items as the cluster's records give them (inc2.v3fam.family_gpuh's input): every attempt of
    every family job's provenance (one V100 per job: finished wall time, or its estimate while running), and
    every family experiment's training runs (report.json's GPU-h once done, else exp.json's high estimate)."""
    out = []
    pdir = C.INC_DIR / "_campaign" / "provenance"
    for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        name = p.stem
        if not family_job(name) or (names is not None and not any(name.startswith(n) for n in names)):
            continue
        rec = F._read_json(p) or {}
        for i, att in enumerate(rec.get("attempts") or []):
            h = _hours(att.get("started_utc"), att.get("finished_utc"))
            out.append({"item": "%s#%d" % (name, i), "state": "ended" if att.get("finished_utc") else "running",
                        "estimate": _estimate(name), "settled": h})
    if names is None:
        for exp in RC.V3FAM_EXPS.values():
            defn = F._read_json(C.INC_DIR / exp / "exp.json")
            if not isinstance(defn, dict):
                continue
            rep = F._read_json(C.INC_DIR / exp / "report.json") or {}
            est = float(((defn.get("cost_estimate") or {}).get("total_gpu_h") or [0, 0])[-1] or 0)
            if rep.get("done"):
                gh = rep.get("gpu_hours") or {}
                got = sum(float((v or {}).get("hours") or 0) for v in gh.values() if isinstance(v, dict))
                out.append({"item": "%s:runs" % exp, "state": "ended", "estimate": est, "settled": got})
            else:
                out.append({"item": "%s:runs" % exp, "state": "running", "estimate": est})
    return out


def spend():
    items = spend_items()
    return {"items": items, "gpuh": F.family_gpuh(items), "cap_gpuh": F.FAMILY_CAP_GPUH}


def person_spend():
    """The person steps' GPU-h (the sweep, the test reads, the re-analysis): what the ticker adds to its own."""
    items = spend_items(PERSON_JOBS)
    return {"items": items, "gpuh": F.family_gpuh(items)}


# ------------------------------------------------------------------ sealed-test reads (V3F.3.6)
def reads_done():
    """{arm: "read" | not run (family cap)} of the family's reads so far."""
    out = {}
    base = F.root() / "test"
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if (d / "read.json").exists():
                out[d.name] = "read"
    for p in sorted(F.cap_dir().glob("test_read_*.json")) if F.cap_dir().is_dir() else []:
        out.setdefault(p.stem[len("test_read_"):], F.NOT_RUN_CAP)
    return out


def read_sources(arm):
    """What a read of `arm` scores, pinned by sha256: its runs or exports, and its classifier."""
    if arm in F.REFERENCE_READS:
        return SYS.reference_sources(arm)
    if arm in ("D4F", "D4N", "D4R15", "BCLEAN"):
        from . import v3fam_trained as VT
        build = VT.TWO_STAGE_BUILD[arm]
        srcs = VT.sources(build)
        return {"mode": "two_stage", "build": build, "seeds": [SYS.source_record(srcs[s]) for s in F.SEEDS],
                "classifier": TS._clf_cfg()[2]}
    if arm == "D3":
        from . import v3fam_wbf as WB
        return {"mode": "inject_boxes", "stage1": [SYS.source_record(SYS.e3_source("B", s)) for s in F.SEEDS],
                "library": WB.wbf_impl()[1], "classifier": TS._clf_cfg()[2]}
    if arm in F.CHALLENGERS and arm.startswith("D2"):
        from . import v3fam_embed as EMB
        cl = F._read_json(EMB.pin_path(EMB.ARM_KEY[arm]))
        return {"mode": "view640" if EMB.ARM_CUT[arm] == "view640" else "two_stage",
                "stage1": [SYS.source_record(SYS.e3_source("B", s)) for s in F.SEEDS],
                "classifier_pin_sha256": TS._sha(EMB.pin_path(EMB.ARM_KEY[arm])), "pin": cl}
    raise F.V3famRefused("%s has no sealed-test read (V3F.3.6)" % arm)


def test_read(arm, read_by):
    """Prepare one sealed-test read (module docstring): checked before anything is written."""
    who = F.read_by_value(read_by)
    v = read_verdict()
    order = list((v.get("reads") or {}).get("order") or [])
    done = reads_done()
    pos = F.check_read(arm, order, done)
    td = test_dir(arm)
    if (td / "read.json").exists() or sorted(td.glob("s*/test.json")) or sorted(td.glob("*/test.json")):
        raise F.V3famRefused("%s's read was prepared or taken already: a read is taken once" % arm)
    try:
        before = F.check_cap(spend_items(), "test_read_%s" % arm, READ_GPUH)
    except F.V3famCapRefused:
        log("%s: %s" % (arm, F.NOT_RUN_CAP))
        raise
    srcs = read_sources(arm)
    logs = C.INC_DIR / "logs"
    argv = ["sbatch", "--parsable", "--job-name=inc_build_v3famtest_%s" % arm.lower(),
            "--output=%s" % (logs / "%x_%j.out"), str(TS._build_script()), "inc2.v3fam", "score-test", "--arm", arm]
    env = DELEGATE_ACCOUNT if read_by == "delegate" else "the owner's account (%s...)" % DECIDED_BY_OWNER_PREFIX
    rec = {"format": FMT_READ, "arm": arm, "kind": pos["kind"], "position": pos["position"], "read_by": who,
           "read_by_key": read_by, "incap_decided_by": env, "verdict": verdict_path().name,
           "verdict_sha256": F._sha(verdict_path()), "sources": srcs, "argv": argv, "cap": before,
           "decided_by": F.DECIDED_BY, "written_utc": F._utc(),
           "note": "submitted by the person named in read_by with INCAP_DECIDED_BY naming them; the scores stay off "
                   "the platform's evidence (V3F.3.6)"}
    logs.mkdir(parents=True, exist_ok=True)
    F.write_once(td / "read.json", json.loads(json.dumps(rec, default=str)))
    return rec


def check_submitter(read_by_key, env=None):
    """The read-path guard (decision 20): INCAP_DECIDED_BY must name the person read_by names (the owner's
    account, human:..., or the delegate's assistant:claude-delegate). Returns the value."""
    got = (os.environ if env is None else env).get("INCAP_DECIDED_BY") or ""
    if read_by_key == "delegate":
        ok = got == DELEGATE_ACCOUNT
    elif read_by_key == "owner":
        ok = got.startswith(DECIDED_BY_OWNER_PREFIX) and len(got) > len(DECIDED_BY_OWNER_PREFIX)
    else:
        ok = False
    if not ok:
        raise F.V3famRefused("INCAP_DECIDED_BY is %r, not the person read_by names (%s): a read submitted by anyone "
                             "else is not a read under the amendment (V3F.3.6)" % (got, read_by_key))
    return got


def _ref_test(token, kw, wk):
    """b_v2_m640's three recorded test scores reproduced through the family identity path (V3F.3.6 item 3);
    written once under v3fam/v1/test/REF/s<k>, kept for a later read."""
    out = []
    names = F.equivalence_names("test")
    for s in F.SEEDS:
        d = test_dir(REF_TEST_DIR) / ("s%d" % s)
        if (d / "test.json").exists():
            doc = F._read_json(d / "test.json") or {}
            if doc.get("repro_status") not in ("passed", "deviation"):
                raise F.V3famRefused("%s does not record a reproduced reference test score" % (d / "test.json"))
            out.append({"seed": s, "status": "kept", "repro": doc.get("repro_status")})
            continue
        src = SYS.e3_source("M", s)
        recorded = F._read_json(C.INC_DIR / TS.REFERENCE / "runs" / src["final_run_id"] / "scores" / "test.json")
        if not isinstance(recorded, dict):
            raise F.V3famRefused("b_v2_m640 seed %d has no recorded test score" % s)
        cfg = {"mode": "identity", "nms": "locked", "workers": wk}
        ps = F._family_pass(src["weights"], "test", cfg, "E3RS", test_token=token, **kw)
        cmp_ = F.compare_with_recorded(ps["result"], recorded, "b_v2_m640 seed %d test" % s)
        st = F.repro_status(cmp_["max_abs_diff"])
        if st == "refused":
            raise F.V3famRefused("b_v2_m640 seed %d's test score is not reproduced (%.4g > %g): no family test "
                                 "score is taken" % (s, cmp_["max_abs_diff"], F.DEVIATION_TOL))
        doc = F.family_doc("REF", s, "test", ps, cfg, equivalence=F.require_equivalence(names),
                           extra={"source": SYS.source_record(src), "vs_recorded": cmp_, "repro_status": st})
        F.commit_pass(d, "test", doc, ps)
        out.append({"seed": s, "status": "written", "repro": st})
    return out


def _test_passes(arm, token, kw, wk, embedder=None):
    """[(seed or None, ps, cfg, extra, classifier)] of arm's test passes (the read's inference)."""
    out = []
    if arm == "D8M":
        for srec in SYS.reference_sources(arm)["seeds"]:
            src = SYS.run_source(SYS.D8_ARMS[arm]["exp"], srec["seed"], testing_ok=TS._testing_ok())
            cfg = {"mode": "identity", "nms": "locked"}
            out.append((srec["seed"], F._family_pass(src["weights"], "test", cfg, "D8", test_token=token, **kw), cfg,
                        {"source": SYS.source_record(src)}, None))
        return out
    if arm == "D6":
        from . import v3fam_detr as DT
        backend = DT.default_backend(None)
        scaffold = SYS.e3_source("M", 0)
        for s in F.SEEDS:
            fr = DT.final_record(s)
            model = backend.load_model(fr["path"])
            rows, st = DT.predict_exam(backend, model, "test")
            cfg = {"mode": "inject_rows", "nms": "locked", "inject": rows}
            out.append((s, F._family_pass(scaffold["weights"], "test", cfg, "D6", test_token=token, **kw), cfg,
                        {"detr": {"final": {k: v for k, v in fr.items() if k != "path"}, "predictions": st},
                         "scaffold": SYS.source_record(scaffold)}, None))
        return out
    ccfg, _crec, shas = TS._clf_cfg()
    if arm in ("D4F", "D4N", "D4R15", "BCLEAN"):
        from . import v3fam_trained as VT
        srcs = VT.sources(VT.TWO_STAGE_BUILD[arm])
        embedder = embedder or V.BioclipEmbedder()
        for s in F.SEEDS:
            cfg = SYS._two_stage_cfg(ccfg, "B", embedder, wk)
            out.append((s, F._family_pass(srcs[s]["weights"], "test", cfg, arm, test_token=token, **kw), cfg,
                        {"source": SYS.source_record(srcs[s])}, shas))
        return out
    if arm == "D3":
        from . import v3fam_wbf as WB
        fn, impl = WB.wbf_impl()
        embedder = embedder or V.BioclipEmbedder()
        stage1s = []
        for s in F.SEEDS:
            cfg = SYS._two_stage_cfg(ccfg, "B", embedder, wk)
            stage1s.append(F._family_pass(SYS.e3_source("B", s)["weights"], "test", cfg, "D3", test_token=token,
                                          **kw)["stage1"])
        fused, counts = WB.fuse_exam(stage1s, SYS.image_sizes(SYS.exam_rows("test")), fn)
        scaffold = SYS.e3_source(*WB.SCAFFOLD)
        cfg = dict(ccfg, mode="inject_boxes", nms="locked", embedder=embedder, workers=wk, top_k=TS.TOP_K,
                   inject=fused)
        out.append((None, F._family_pass(scaffold["weights"], "test", cfg, "D3", test_token=token, **kw), cfg,
                    {"fusion": {"library": impl, "counts": counts}, "one_system": True}, shas))
        return out
    if arm.startswith("D2"):
        from . import v3fam_embed as EMB
        bb = EMB.ARM_BACKBONE[arm]
        for s in F.SEEDS:
            if arm == EMB.FT_ARM:
                ck = EMB.checkpoint(bb)
                emb, finfo = EMB._ft_embedder(s, ck)
                rows, _r = EMB.training_crops(images=False)
                import numpy as np
                counts = np.bincount([r["label"] for r in rows], minlength=C.NC).astype(np.float64)
                cfg = dict(EMB.ft_classifier_cfg(counts / counts.sum()), mode="two_stage", nms=TS.NMS["B"],
                           embedder=emb, workers=wk, top_k=TS.TOP_K)
                clf = {"key": "FT", "pin_sha256": finfo["pin_sha256"], "weights_sha256": finfo["sha256"]}
            else:
                emb, _ck = EMB.load_embedder(bb)
                dcfg, _rec, clf = EMB.load_d2_classifier(EMB.ARM_KEY[arm])
                mode = "view640" if EMB.ARM_CUT[arm] == "view640" else "two_stage"
                cfg = dict(dcfg, mode=mode, nms=TS.NMS["B"], embedder=emb, workers=wk, top_k=TS.TOP_K)
                if mode == "view640":
                    cfg["cut"] = EMB.e3_cut640
            src = SYS.e3_source("B", s)
            out.append((s, F._family_pass(src["weights"], "test", cfg, arm, test_token=token, **kw), cfg,
                        {"stage1_source": SYS.source_record(src)}, clf))
        return out
    raise F.V3famRefused("%s has no sealed-test read" % arm)


def score_test(arm, embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, env=None):
    """The read's job (module docstring): the submitter checked, the reference reproduced, the arm scored once."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    td = test_dir(arm)
    rr = F._read_json(td / "read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_READ or rr.get("arm") != arm:
        raise F.V3famRefused("%s has no prepared read (test-read --arm %s --read-by ... first)" % (arm, arm))
    submitter = check_submitter(rr.get("read_by_key"), env)
    read_verdict()
    if F._sha(verdict_path()) != rr.get("verdict_sha256"):
        raise F.V3famRefused("the verdict changed since the read was prepared")
    now = json.loads(json.dumps(read_sources(arm), default=str))
    if now != rr.get("sources"):
        raise F.V3famRefused("%s's weights or classifier changed since the read was prepared" % arm)
    done = sorted(str(p) for p in td.glob("*/test.json"))
    if done:
        raise F.V3famRefused("%s's test was scored already (%s): read once" % (arm, done))
    names = F.equivalence_names("test", e3_arms=("B",) if arm not in ("D6", "D8M") else ())
    t0 = time.time()
    out = {"reference": [], "arm": []}
    with F.e_trees_unchanged():
        with F.test_read_token() as token:
            with TS._workers(workers) as wk:
                out["reference"] = _ref_test(token, kw, wk)
                eqs = F.require_equivalence(names)
                for s, ps, cfg, extra, clf in _test_passes(arm, token, kw, wk, embedder=embedder):
                    x = dict(extra, read={"read_by": rr["read_by"], "incap_decided_by": submitter,
                                          "read_sha256": F._sha(td / "read.json"), "kind": rr["kind"]})
                    doc = F.family_doc(arm, s, "test", ps, cfg, equivalence=eqs, classifier=clf, extra=x)
                    F.commit_pass(td / ("s%d" % s if s is not None else "ens"), "test", doc, ps)
                    out["arm"].append({"seed": s, "species_map50_95": doc.get("species_map50_95")})
    log("%s's sealed test scored (%s), %.0fs; test-report --arm %s" % (arm, out["arm"], time.time() - t0, arm))
    return out


def _arrays_mean(runs, keys=None):
    vals = []
    for a in runs:
        b = F.subset_arrays(a, keys) if keys is not None else a
        vals.append(F.statistic_value(b, "species"))
    return vals


def test_report(arm):
    """The read's report, for people (capacity/v3fam_test_<x>.{json,md}): the arm's and the reference's test
    species mean +- sd, agnostic, the gap to 0.90, the stratum of the test images from dev's sessions."""
    rr = F._read_json(test_dir(arm) / "read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_READ:
        raise F.V3famRefused("%s has no prepared read" % arm)
    base = test_dir(arm)
    got, ref = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        doc = F._read_json(d / "test.json")
        if isinstance(doc, dict):
            got.append((doc, SC.load_npz(d / doc["images"]["name"])))
    for s in F.SEEDS:
        d = test_dir(REF_TEST_DIR) / ("s%d" % s)
        doc = F._read_json(d / "test.json")
        if isinstance(doc, dict):
            ref.append((doc, SC.load_npz(d / doc["images"]["name"])))
    if not got:
        raise F.V3famRefused("%s's test has not been scored (score-test)" % arm)
    stratum = F.dev_session_stratum(SYS.exam_rows("test"), SYS.exam_rows("dev"))

    def side_of(pairs):
        if not pairs:
            return None
        sp = [float(d["species_map50_95"]) for d, _a in pairs]
        ag = [float(d.get("agnostic_map50_95")) for d, _a in pairs if d.get("agnostic_map50_95") is not None]
        st = _arrays_mean([a for _d, a in pairs], stratum)
        m, s = F._mean_sd(sp)
        return {"species": sp, "mean": m, "sd": s, "gap_to_target": TARGET - m, "agnostic": ag,
                "agnostic_mean": F._mean_sd(ag)[0] if ag else None,
                "stratum": {"n_images": len(stratum), "species": st, "mean": F._mean_sd(st)[0]}}
    rep = {"format": FMT_TEST_REPORT, "arm": arm, "label": LABELS.get(arm, arm), "kind": rr.get("kind"),
           "read_by": rr.get("read_by"), "target": TARGET, "arm_test": side_of(got), "reference_test": side_of(ref),
           "stratum_rule": "the test images whose session is one of dev's (V3F.3.6; 329 of 1,977 on 2026-10-11)",
           "note": "descriptive; a reference or challenger test number is a separate labelled row, never the "
                   "headline (V3F.3.6)", "written_utc": F._utc()}
    p = cap_path(TEST_RECORD % arm.lower())
    TS._write_json(p.with_suffix(".json"), rep)
    a = rep["arm_test"]
    p.with_suffix(".md").write_text(
        "# %s's sealed test (%s read, read by %s)\n\n12-class mAP50-95 %.4f +- %.4f (gap to 0.90: %.4f); the %d test "
        "images from dev's sessions: %.4f. Reference b_v2_m640: %s.\n" % (
            rep["label"], rep["kind"], rep["read_by"], a["mean"], a["sd"] or 0.0, a["gap_to_target"],
            a["stratum"]["n_images"], a["stratum"]["mean"] or 0.0,
            "%.4f" % rep["reference_test"]["mean"] if rep["reference_test"] else "not scored"))
    return rep


def reanalyse_test(read_by, env=None):
    """O4's re-analysis of stored test arrays (no inference; module docstring): capacity/v3fam_retest.json once."""
    who = F.read_by_value(read_by)
    submitter = check_submitter(read_by, env)
    read_verdict()
    out = cap_path(RETEST)
    if out.exists():
        raise F.V3famRefused("%s exists: the re-analysis runs once" % out)
    F.check_cap(spend_items(), "reanalyse_test", RETEST_GPUH)
    stratum = F.dev_session_stratum(SYS.exam_rows("test"), SYS.exam_rows("dev"))
    trig = d0s_trigger()
    h = [str(k)[len("test/"):] for k in trig.get("h_test") or []]
    minus = [r["key"] for r in SYS.exam_rows("test") if r["key"] not in set(h)] if h else None
    rows = {}
    for name, sub in (("A", "A"), ("M", "M"), ("B", "B"), ("reference", "reference")):
        pairs = []
        for s in F.SEEDS:
            d = TS.root() / "test" / sub / ("s%d" % s)
            jn = "test.identity.json" if sub == "reference" else "test.json"
            doc = F._read_json(d / jn)
            if not isinstance(doc, dict):
                continue
            pairs.append(SC.load_npz(d / doc["images"]["name"]))
        if len(pairs) != len(F.SEEDS):
            rows[name] = {"status": "missing"}
            continue
        st = _arrays_mean(pairs, stratum)
        rows[name] = {"status": "ok", "stratum": {"species": st, "mean": F._mean_sd(st)[0]}}
        if minus is not None:
            tm = _arrays_mean(pairs, minus)
            rows[name]["test_minus_h"] = {"species": tm, "mean": F._mean_sd(tm)[0]}
    rec = {"format": FMT_RETEST, "read_by": who, "incap_decided_by": submitter, "stratum_images": len(stratum),
           "test_minus_h_images": None if minus is None else len(minus), "arms": rows,
           "decided_by": F.DECIDED_BY, "written_utc": F._utc(),
           "note": "re-analysis of E3's stored test arrays, no inference; descriptive (V3F.3.6)"}
    F.write_once(out, rec)
    return rec
