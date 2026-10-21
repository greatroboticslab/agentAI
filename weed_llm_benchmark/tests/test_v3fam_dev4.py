#!/usr/bin/env python3
"""V3F deviation 4 (2026-10-20; docs/prereg/V3F_dev4_2026-10-20.md): the guards that keep the family's verdict from
being decided before every arm's record is complete or failed (V3F.7 step 11), the reopening of the items the
deviation names, and the read path to the continuation verdict. Fast, record-level checks only (no scoring pass, no
ticker run), so tests/test_v3fam_dev4_mutants.py can run this file against each of its mutants.

- the platform's guard (diagnose_stream.v3fam_verdict_ready and v3fam_plan): the state the ticker held on
  2026-10-13 (the data build failed twice, its dependants 'not run (the family's data build failed)', every score
  before the builds done) proposes no L23L and waits naming the data build and deviation 4; the data build done and
  every item ended proposes L23L; a pending build, a running D6 seed, an item 'not run (the family's data build
  failed)' alone, each holds it; the registered not-run reasons (the family cap, D6's walltime, an arm's own second
  failure) do not;
- the reopening (stream.StreamRun._v3fam_reopen): the data build (failed) and the verdict (done) of 2026-10-12/13
  reopened once the deployed tree holds the deviation's file with its sha256: state missing, the spend so far moved
  into prior_gpuh, every earlier proposal id in failed_ids, the earlier record kept, the verdict's card mark moved;
  once per marker; never without the file or with another sha256, never an item started after the deviation, never
  a running item;
- the verb's guard (inc2.v3fam_verdict.verdict_ready, verdict): refused without the family's data summary and while
  an arm never started (no record, no attempt of its score job, its builds or D6's jobs), writing nothing; BCLEAN
  excused only without D0-S's trigger; an item refused at the family cap excused; an item the ticker recorded
  failed after its own second failure (--ticker-failed) ends the arms it ends, also when its job failed before
  writing a provenance attempt, and is written once under v3fam/v1/failed; the data build's failure is refused there;
- the two guards agree: a build that failed twice before its provenance attempt (cancelled while pending, NODE_FAIL
  at launch) and its score arm derived 'not run': the ticker proposes L23L with --ticker-failed naming the build,
  and the verb is then ready; every dependency the domain's score_after names is one the verb's ending_items counts;
- L23L's record-only follow (a submission whose outcome is unknown) names the domain's verdict record without
  raising (it read self.dom inside a static method: NameError on every tick);
- the read path: with the verdict of 2026-10-13 in place (its sha256) the family's verdict is the continuation;
  read_verdict refuses until it is decided;
- the continuation's reproduction check: equal decisions pass; a D 1e-6 off, another outcome, or a primary no longer
  family-confirmed refuses.

Run:  python3 tests/test_v3fam_dev4.py
"""
import hashlib
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_v3fam_verdict as VT  # noqa: E402  (sets INC_DIR / REPO in a temporary directory first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_data as VD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_verdict as VR  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as S  # noqa: E402

FAILURES = VT.FAILURES
check = VT.check
refused = VT.refused
FAM = LS.load_domain("weed")["v3fam"]
NOT_RUN_DATA = "not run (the family's data build failed)"


# ------------------------------------------------------------------ the platform's guard
def stage(data="done", score=None, build=None, detr=None, verdict="missing"):
    """A /stage/v3fam: every score arm done, every build done (D0BCLEAN not needed), D6's jobs done, unless given."""
    sc = {a: "done" for a in FAM["pre_build_scores"] + FAM["post_build_scores"]}
    sc["BCLEAN"] = "not needed"
    sc.update(score or {})
    b = {a: "done" for a in FAM["builds"]}
    b["D0BCLEAN"] = "not needed"
    b.update(build or {})
    d = {"smoke": "done", "train": "done"}
    d.update(detr or {})
    return {"data": data, "score": sc, "build": b, "detr": d, "verdict": verdict}


def t_platform_guard():
    print("the platform's guard: L23L only once every arm's record is complete or failed")
    # the ticker's state on 2026-10-13 (stream.py _v3fam_stage after L23H's second failure)
    post = {a: "not run (a dependency ended without its effect)" for a in FAM["post_build_scores"]}
    post["BCLEAN"] = "not needed"
    oct13 = stage(data="failed", score=post, build={a: NOT_RUN_DATA for a in FAM["builds"] if a != "D0BCLEAN"},
                  detr={"smoke": NOT_RUN_DATA, "train": NOT_RUN_DATA})
    plan, waits = DS.v3fam_plan(FAM, oct13)
    ready, pending = DS.v3fam_verdict_ready(FAM, oct13)
    check("the state of 2026-10-13 (the data build failed twice, its builds and D6 'not run (the family's data build "
          "failed)', every score before the builds done): no L23L; the verdict waits, naming the data build and "
          "deviation 4", "MAINT" not in plan and not ready and pending[0] == "data (failed)"
          and "build:D4F" in pending and "detr:smoke" in pending and "deviation 4" in waits.get("verdict", "")
          and "data (failed)" in waits.get("verdict", ""), (plan, pending, waits))
    plan, _w = DS.v3fam_plan(FAM, stage())
    check("the data build done and every item ended: L23L proposed", plan.get("MAINT") == {"lever": "L23L",
                                                                                        "item": "verdict"}, plan)
    cases = {
        "a build not yet run (missing)": stage(build={"D4N": "missing"}),
        "a build running": stage(build={"D1B1": "running"}),
        "D6's seeds running": stage(detr={"train": "running"}),
        "a score arm failed once (its retry due)": stage(score={"D1": "failed_once"}),
        "the data build done but one build 'not run (the family's data build failed)'":
            stage(build={"D5B0": NOT_RUN_DATA}),
        "the data build done but D6's smoke 'not run (the family's data build failed)'":
            stage(detr={"smoke": NOT_RUN_DATA, "train": NOT_RUN_DATA}),
        "the data build missing (reopened), everything else ended": stage(data="missing"),
        "the data build failed once": stage(data="failed_once")}
    held = {k: (DS.v3fam_plan(FAM, v)[0].get("MAINT") or {}).get("lever") for k, v in cases.items()}
    check("each holds the verdict: %s" % "; ".join(cases), all(v != "L23L" for v in held.values()), held)
    ok_cases = {
        "the data build refused at the family cap": stage(data="not run (family cap)"),
        "D6 not run at its walltime": stage(detr={"smoke": "not run (walltime)", "train": "not run (walltime)"},
                                            score={"D6": "not run (a dependency ended without its effect)"}),
        "an arm failed after its own second failure": stage(score={"D2LAION": "failed"}),
        "a build failed after its own second failure": stage(build={"D4F": "failed"},
                                                             score={"D4F": "not run (a dependency ended without "
                                                                           "its effect)"}),
        "an arm refused at the family cap": stage(score={"D9": "not run (family cap)"}),
        "a download not run": stage(score={"D2SIGLIP": "not run (download)"})}
    got = {k: (DS.v3fam_plan(FAM, v)[0].get("MAINT") or {}).get("lever") for k, v in ok_cases.items()}
    check("the registered not-run reasons do not hold it (V3F.3.5): %s" % "; ".join(ok_cases),
          all(v == "L23L" for v in got.values()), got)
    plan, _w = DS.v3fam_plan(FAM, stage(verdict="done"))
    plan2, _w = DS.v3fam_plan(FAM, stage(verdict="running"))
    check("a verdict done or running is not proposed again", "MAINT" not in plan and "MAINT" not in plan2,
          (plan, plan2))


# ------------------------------------------------------------------ the reopening
class Paths(object):
    def __init__(self, root):
        self.lab_inc = root / "lab" / "results" / "framework" / "inc"
        self.ledger = root / "ledger.jsonl"


class Fam(object):
    """S.StreamRun's own family methods on a bare state."""
    _v3fam = S.StreamRun._v3fam
    _v3fam_fold = staticmethod(S.StreamRun._v3fam_fold)
    _v3fam_reopen = S.StreamRun._v3fam_reopen
    _v3fam_deployed = S.StreamRun._v3fam_deployed
    _v3fam_params = S.StreamRun._v3fam_params
    _failed_ids = S.StreamRun._failed_ids

    def __init__(self, root, reopen, st):
        self.paths = Paths(root)
        self.dom = {"v3fam": dict(FAM, reopen=reopen)}
        self.st = st
        self.utc = "2026-10-21T00:00:00Z"
        self.cards, self.ledger = [], []

    def _ledger(self, event, **kw):
        self.ledger.append(dict(kw, event=event))

    def _card(self, kind, title, detail="", lever=None, trigger=None):
        self.cards.append(title)


def oct_state():
    """The family items as the ticker held them on 2026-10-20 (the live state's data and verdict items)."""
    return {"failed_ids": ["5220f1e01ef8a5808f1372e0d48b87d0"], "declined": ["669bc97b17bfc538aebf22ca53b4e4d7"],
            "v3fam": {"verdict_card": "2026-10-13T13:14:58Z", "items": {
                "data": {"attempts": [{"gpuh": 0.5592, "how": "settled", "proposal": "5220f1e01ef8a5808f1372e0d48b87d0"}],
                         "estimate": 2.0, "fails": 2, "lever": "L23H", "prior_gpuh": 0.5592,
                         "proposal": "669bc97b17bfc538aebf22ca53b4e4d7", "settled": 0.2328, "spend": "settled",
                         "started_utc": "2026-10-12T12:24:57Z", "state": "failed"},
                "verdict": {"estimate": 0.5, "lever": "L23L", "proposal": "d6a9917c5aa1d222848525288c2e138c",
                            "settled": 0.66, "spend": "settled", "started_utc": "2026-10-13T12:04:58Z",
                            "state": "done"},
                "score:D9": {"estimate": 4.4, "lever": "L23J", "proposal": "p9", "settled": 3.9678, "spend": "settled",
                             "started_utc": "2026-10-13T07:44:58Z", "state": "done"}}}}


def t_reopen():
    print("the reopening of the items deviation 4 names")
    root = VT.TMP / "reopen"
    doc = root / "docs" / "prereg" / "V3F_dev4.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("deviation 4\n")
    sha = hashlib.sha256(doc.read_bytes()).hexdigest()
    ro = {"marker": "deviation 4 (2026-10-20)", "file": "docs/prereg/V3F_dev4.md", "sha256": sha,
          "items": ["data", "verdict"], "before_utc": "2026-10-20T00:00:00Z", "why": "test"}
    f = Fam(root, ro, oct_state())
    got = f._v3fam_reopen()
    it = f.st["v3fam"]["items"]
    d, v = it["data"], it["verdict"]
    check("the data build (failed twice) and the verdict (done) of 2026-10-12/13 reopened: state missing, fails 0, "
          "the marker recorded, the earlier record kept", got == ["data", "verdict"] and d["state"] == "missing"
          and v["state"] == "missing" and d["fails"] == 0 and d["reopened"] == ro["marker"]
          and d["reopen_history"][0]["state"] == "failed" and d["reopen_history"][0]["fails"] == 2
          and v["reopen_history"][0]["state"] == "done", (got, d.get("state"), v.get("state")))
    check("  every attempt stays counted against the family cap: data's prior_gpuh 0.5592 + 0.2328, the verdict's "
          "0.66; nothing settled or open is left", abs(d["prior_gpuh"] - 0.792) < 1e-9
          and abs(v["prior_gpuh"] - 0.66) < 1e-9 and d.get("spend") is None and v.get("settled") is None
          and len(d["attempts"]) == 2, (d.get("prior_gpuh"), v.get("prior_gpuh")))
    check("  the earlier proposal ids go to failed_ids (a new proposal takes a new id), the verdict's card mark moves "
          "into its history, one card and one ledger entry",
          set(f.st["failed_ids"]) >= {"5220f1e01ef8a5808f1372e0d48b87d0", "669bc97b17bfc538aebf22ca53b4e4d7",
                                      "d6a9917c5aa1d222848525288c2e138c"}
          and "verdict_card" not in f.st["v3fam"] and v["reopen_history"][0]["verdict_card"] == "2026-10-13T13:14:58Z"
          and len(f.cards) == 1 and [x["event"] for x in f.ledger] == ["v3fam_reopened"]
          and it["score:D9"]["state"] == "done", (f.st["failed_ids"], f.cards))
    again = f._v3fam_reopen()
    d["state"] = "failed"                     # the reopened build fails twice again: not reopened a second time
    third = f._v3fam_reopen()
    check("  once per marker: a second tick reopens nothing, nor a reopened item that failed again",
          again == [] and third == [] and it["data"]["state"] == "failed" and len(f.cards) == 1, (again, third))
    res = {}
    for tag, kw in (("its file missing", {"file": "docs/prereg/none.md"}), ("another sha256", {"sha256": "0" * 64}),
                    ("no marker", {"marker": None})):
        g = Fam(root, dict(ro, **kw), oct_state())
        res[tag] = (g._v3fam_reopen(), g.st["v3fam"]["items"]["data"]["state"])
    late = oct_state()
    late["v3fam"]["items"]["data"]["started_utc"] = "2026-10-20T03:00:00Z"
    g = Fam(root, ro, late)
    res["an item started after the deviation"] = (g._v3fam_reopen(), g.st["v3fam"]["items"]["data"]["state"])
    run = oct_state()
    run["v3fam"]["items"]["data"]["state"] = "running"
    g = Fam(root, ro, run)
    res["an item running"] = (g._v3fam_reopen(), g.st["v3fam"]["items"]["data"]["state"])
    check("never reopened: %s" % "; ".join(res), res["its file missing"] == ([], "failed")
          and res["another sha256"] == ([], "failed") and res["no marker"] == ([], "failed")
          and res["an item started after the deviation"] == (["verdict"], "failed")
          and res["an item running"] == (["verdict"], "running"), res)
    check("the domain names deviation 4's reopening: the data build and the verdict, before 2026-10-20, the "
          "continuation as the verdict record", FAM["reopen"]["items"] == ["data", "verdict"]
          and FAM["reopen"]["before_utc"] == "2026-10-20T00:00:00Z"
          and FAM["reopen"]["file"] == "docs/prereg/V3F_dev4_2026-10-20.md"
          and FAM["verdict_record"] == "capacity/v3fam_v1_cont.json"
          and FAM["partial_verdict_record"] == "capacity/v3fam_v1.json", FAM.get("reopen"))
    dev4 = pathlib.Path(__file__).resolve().parents[2] / FAM["reopen"]["file"]
    check("  and the sha256 it records is the committed deviation file's",
          dev4.is_file() and hashlib.sha256(dev4.read_bytes()).hexdigest() == FAM["reopen"]["sha256"], str(dev4))


# ------------------------------------------------------------------ the verb's guard and the read path
def provenance(name, attempts=1):
    p = C.INC_DIR / "_campaign" / "provenance" / ("%s.json" % name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"attempts": [{"job_id": "1", "started_utc": "2026-10-12T00:00:00Z",
                                           "finished_utc": "2026-10-12T00:00:00Z"}] * attempts}))
    return p


def records(arms):
    for a in arms:
        TS._write_json(SYS.score_record_path(a), {"format": SYS.FMT_SCORE_RECORD, "arm": a, "status": "complete",
                                                 "exam": "dev", "dev": []})


def clean():
    for d in (F.root(), C.INC_DIR / "capacity", C.INC_DIR / "_campaign", C.INC_DIR / "splits" / "v3fam"):
        shutil.rmtree(str(d), ignore_errors=True)


def t_verb_guard():
    print("the verb's guard: no verdict before every arm's record is complete or failed")
    clean()
    first = ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
             "D3", "D9")
    records(first)
    ok0, why0 = VR.verdict_ready()
    e0 = refused(VR.verdict)
    check("the records of 2026-10-13 (the twelve arms scored before the builds; no data summary; D1, D4F, D4N, D4R15, "
          "D5B, D6, D8 never started; BCLEAN not triggered): not ready, naming the summary and the seven arms; the "
          "verb refuses and writes nothing",
          not ok0 and "summary" in why0 and "never started: D1, D4F, D4N, D4R15, D5B, D6, D8" in why0
          and e0 is not None and "every arm's record is complete or failed" in str(e0)
          and not (C.INC_DIR / "capacity" / VR.VERDICT).exists()
          and not (C.INC_DIR / "capacity" / VR.EXTERNAL).exists() and not F.cap_dir().exists(), (why0, e0))
    TS._write_json(VD._summary_path(), {"format": VD.FORMAT, "status": "complete", "arms": {}})
    ok1, why1 = VR.verdict_ready()
    made = [provenance("v3_d4_f_m640"), provenance("v3fam_score_d4n"), provenance("v3fam_detr_s1"),
            provenance("v3_d8_s640_tc"), provenance("v3fam_score_d5b"), provenance("v3fam_score_d4r15")]
    ok2, why2 = VR.verdict_ready()
    check("with the summary, an arm is started by an attempt of its score job, of a build it scores or of D6's jobs: "
          "only D1 left", not ok1 and "summary" not in why1 and not ok2 and why2 == "never started: D1",
          (why1, why2))
    ok3, why3 = VR.verdict_ready(cap_refused=["build:D1B1"])
    ok4, _w4 = VR.verdict_ready(cap_refused=["score:D1"])
    provenance("v3_d1_a12_m640")
    ok5, why5 = VR.verdict_ready()
    check("an item refused at the family cap (a build D1 scores, or D1's own score) excuses it; then an attempt of "
          "one of D1's builds: ready", ok3 and ok4 and ok5 and why5 == "", (why3, why5))
    (C.INC_DIR / "_campaign" / "provenance" / "v3_d1_a12_m640.json").unlink()
    got = {tag: VR.verdict_ready(failed=f) for tag, f in (
        ("a build D1 scores failed twice (no attempt recorded)", ["build:D1B1"]),
        ("D1's own score job failed twice", ["score:D1"]),
        ("E3RS, which every arm is scored after, failed twice", ["score:E3RS"]),
        ("another arm's build failed twice", ["build:D4F"]))}
    check("an item the ticker recorded failed after its own second failure ends the arms it ends, its job attempted "
          "or not: %s" % "; ".join(got), got["a build D1 scores failed twice (no attempt recorded)"][0]
          and got["D1's own score job failed twice"][0] and got["E3RS, which every arm is scored after, failed twice"][0]
          and got["another arm's build failed twice"] == (False, "never started: D1"), got)
    bad = {tag: refused(VR.verdict_ready, failed=f) for tag, f in (("the data build", ["data"]),
                                                                   ("the verdict", ["verdict"]),
                                                                   ("not a family item", ["build:d1b1;rm"]))}
    check("  --ticker-failed never names the data build (its failure ends nothing: the summary is needed) or the "
          "verdict", all(e is not None and "--ticker-failed" in str(e) for e in bad.values()),
          {k: str(v)[:100] for k, v in bad.items()})
    names = VR.record_ticker_failures(["build:D1B1"])
    rec = json.loads(VR.failed_file("build:D1B1").read_text())
    st, (ok_r, why_r) = VR.arm_status("D1"), VR.verdict_ready()
    VR.record_ticker_failures(["build:D1B1"])
    check("  written once under v3fam/v1/failed (a second call keeps it); the arm is then reported failed with its "
          "reason, and the verb is ready without the argv", names == ["build:D1B1"] and rec["status"] == "failed"
          and rec["item"] == "build:D1B1" and st == "failed (build:D1B1 failed twice)" and ok_r
          and json.loads(VR.failed_file("build:D1B1").read_text()) == rec, (st, why_r, rec))
    shutil.rmtree(str(VR.failed_dir()))
    provenance("v3_d1_a12_m640")
    sw = F.root() / "sweep" / "record.json"
    sw.parent.mkdir(parents=True, exist_ok=True)
    sw.write_text(json.dumps({"format": "inc2-v3fam-sweep-record/1", "status": "complete", "H": ["test/t1"],
                              "trigger": {"bclean": True, "test_branch": True}}))
    trig = VR.d0s_trigger()
    ok6, why6 = VR.verdict_ready()
    provenance("v3_d0_bclean_m640")
    ok7, _w7 = VR.verdict_ready()
    check("BCLEAN is excused only without D0-S's trigger: triggered and never started, not ready; its build tried, "
          "ready", bool(trig.get("bclean")) and not ok6 and why6 == "never started: BCLEAN" and ok7, (trig, why6))
    sw.unlink()
    (C.INC_DIR / "splits" / "v3fam" / "summary.json").unlink()
    ok8, why8 = VR.verdict_ready(cap_refused=["data"])
    VR.record_ticker_refusals(["data"])
    ok9, _w9 = VR.verdict_ready()
    check("a data build refused at the family cap (in the argv, or its record written) stands for the summary",
          ok8 and ok9, why8)
    clean()


def t_guards_agree():
    print("the ticker's and the verb's guards agree on a job that failed twice before its provenance attempt")
    clean()
    first = ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
             "D3", "D9")
    records(first)
    TS._write_json(VD._summary_path(), {"format": VD.FORMAT, "status": "complete", "arms": {}})
    for n in ("v3fam_score_d4n", "v3fam_score_d4r15", "v3fam_score_d1", "v3fam_score_d5b", "v3fam_score_d6",
              "v3fam_score_d8"):
        provenance(n)
    # the ticker: build:D4F recorded failed after its second failure (both attempts ended before 'prov start'), so
    # its score arm is derived 'not run (a dependency ended without its effect)'; every other item done
    fs = stage(build={"D4F": "failed"}, score={"D4F": "not run (a dependency ended without its effect)"})
    t_ready, t_pending = DS.v3fam_verdict_ready(FAM, fs)
    root = VT.TMP / "agree"
    items = {"build:D4F": {"state": "failed", "fails": 2, "lever": "L23I"},
             "build:D4N": {"state": "done", "lever": "L23I"}, "data": {"state": "done", "lever": "L23H"},
             "score:D9": {"state": "not run (family cap)", "lever": "L23J"}}
    f = Fam(root, None, {"v3fam": {"items": items}})
    params, _child = f._v3fam_params("L23L", {}, "inc2", {})
    argv_failed = [x for x in (params.get("ticker_failed") or "").split(",") if x]
    v_before = VR.verdict_ready(cap_refused=[x for x in (params.get("cap_refused") or "").split(",") if x])
    v_ready, v_why = VR.verdict_ready(cap_refused=[x for x in (params.get("cap_refused") or "").split(",") if x],
                                      failed=argv_failed)
    check("the ticker proposes L23L with --ticker-failed naming the failed build (never the data build), and the verb "
          "given it is ready (without it: 'never started: D4F', the disagreement that would fail L23L twice)",
          t_ready and not t_pending and params.get("ticker_failed") == "build:D4F"
          and params.get("cap_refused") == "score:D9" and v_before == (False, "never started: D4F") and v_ready,
          (t_pending, params, v_before, v_why))
    f2 = Fam(root, None, {"v3fam": {"items": dict(items, data={"state": "failed", "lever": "L23H"})}})
    p2, _c = f2._v3fam_params("L23L", {}, "inc2", {})
    check("  a failed data build is never passed as a failure", p2.get("ticker_failed") == "build:D4F", p2)
    gaps = {}
    for arm, deps in (FAM.get("score_after") or {}).items():
        want = {"score:%s" % d if not (d.startswith("build:") or d.startswith("detr:")) else d
                for d in deps if d != "exif"}
        miss = want - set(VR.ending_items(arm))
        if miss:
            gaps[arm] = sorted(miss)
    check("every dependency the domain's score_after names (but the EXIF pre-flight, which only waits) is one whose "
          "failure the verb counts for that arm (ending_items)", not gaps and set(FAM["score_after"]) <= set(
              F.SCORE_ARMS), gaps)
    clean()


def t_record_only_follow():
    print("L23L's record-only follow (a submission whose outcome is unknown)")
    root = VT.TMP / "follow"
    f = Fam(root, None, {"v3fam": {"items": {}}})
    f._record_only_follow = S.StreamRun._record_only_follow.__get__(f)
    got, err = {}, None
    try:
        for lv, prm in (("L23L", {}), ("L23H", {}), ("L23J", {"arm": "D4F"})):
            got[lv] = f._record_only_follow({"lever": lv, "proposal": {"params": prm}})
    except Exception as e:                       # noqa: BLE001  (NameError: self in a static method)
        err = e
    check("L23L follows the domain's verdict record (the continuation), L23H and L23J theirs, without raising",
          err is None and got.get("L23L") == ("inc_build_v3fam_v1", "capacity/v3fam_v1_cont.json")
          and got.get("L23H") == ("inc_build_v3fam_data", "splits/v3fam/summary.json")
          and got.get("L23J") == ("inc_build_v3fam_score_d4f", "capacity/v3fam_score_d4f.json"),
          (got, "%s: %s" % (type(err).__name__, err) if err else None))


def t_read_path():
    print("the read path: after the verdict of 2026-10-13, the reads follow its continuation")
    clean()
    vp = VR.verdict_path()
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(json.dumps({"format": VR.FMT_VERDICT, "status": "decided", "exam": "dev",
                              "reads": {"order": [], "cap": 5}}))
    real = VR.PARTIAL_VERDICT_SHA256
    try:
        check("another verdict in place: the family's verdict is capacity/v3fam_v1.json, read as decided",
              VR.family_verdict_path() == vp and VR.read_verdict()["status"] == "decided")
        VR.PARTIAL_VERDICT_SHA256 = F._sha(vp)
        cp = C.INC_DIR / "capacity" / VR.CONTINUATION["verdict"]
        e = refused(VR.read_verdict)
        check("the verdict of 2026-10-13 in place (its sha256): the family's verdict is the continuation; "
              "read_verdict refuses until it is decided, naming deviation 4",
              VR.family_verdict_path() == cp and e is not None and "deviation 4" in str(e) and VR.is_partial(), e)
        cp.write_text(json.dumps({"format": VR.FMT_VERDICT, "status": "decided", "exam": "dev",
                                  "reads": {"order": [{"arm": "D6", "kind": "reference"}], "cap": 5}}))
        got = VR.read_verdict()
        check("  once decided, read_verdict gives the continuation (its read order)",
              [o["arm"] for o in got["reads"]["order"]] == ["D6"], got.get("reads"))
        e2 = refused(VR.verdict)
        check("  and the verb refuses a second continuation (written once)", e2 is not None
              and "written once" in str(e2), e2)
        check("the pinned sha256 is the record of job 49107774 (2026-10-13 12:47Z) as read from the cluster",
              real == "304bad59f15487bfd3965d2f1e9384658daac58c5971494118ecaad4fcfaa7d9"
              and VR.CONTINUATION["verdict"] == "v3fam_v1_cont.json")
    finally:
        VR.PARTIAL_VERDICT_SHA256 = real
        clean()


def dec(diff, z, lo, hi, outcome, q):
    return {"status": "decided", "decision": {"diff": diff, "z": z, "interval_90": [lo, hi], "outcome": outcome,
                                              "qualifies": q}}


def t_reproduction():
    print("the continuation reproduces every comparison the verdict of 2026-10-13 decided")
    clean()
    old = {"primaries": {"P1": dec(0.0757, 7.77, 0.0597, 0.0918, "qualifies", True),
                         "P2": {"status": "not run", "why": "missing"},
                         "P4": dec(-0.0042, -1.51, -0.0087, 0.0004, "inconclusive", False)},
           "holm": {"P1": {"family_confirmed": True}, "P4": {"family_confirmed": False}},
           "tier_s": {"S_x": dec(0.01, 2.0, 0.0, 0.02, "qualifies", True)},
           "challengers": {"comparisons": {"D2CLIP": dec(-0.08, -7.8, -0.1, -0.06, "worse", False)}}}
    vp = VR.verdict_path()
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(json.dumps(old))
    new = json.loads(json.dumps(old))
    new["primaries"]["P2"] = dec(0.02, 3.0, 0.01, 0.03, "qualifies", True)
    new["holm"]["P2"] = {"family_confirmed": True}
    c = VR.continuation_check(new, {})
    check("equal decisions (P2 decided now): passes, listing what it reproduced",
          c["reproduced"] == ["C_D2CLIP", "P1", "P4", "S_x"] and c["family_confirmed_kept"] == ["P1"], c)
    res = {}
    for tag, path, val in (("P1's D 1e-6 off", ("primaries", "P1", "diff"), 0.0757 + 1e-6),
                           ("P4 another outcome", ("primaries", "P4", "outcome"), "equivalent"),
                           ("a challenger's interval 1e-6 off", ("challengers", "D2CLIP", "interval_90"),
                            [-0.1, -0.06 + 1e-6]),
                           ("a Tier S comparison no longer decided", ("tier_s", "S_x", None), None)):
        m = json.loads(json.dumps(new))
        if path[0] == "challengers":
            m["challengers"]["comparisons"][path[1]]["decision"][path[2]] = val
        elif path[2] is None:
            m[path[0]][path[1]] = {"status": "not run", "why": "x"}
        else:
            m[path[0]][path[1]]["decision"][path[2]] = val
        res[tag] = refused(VR.continuation_check, m, {})
    m = json.loads(json.dumps(new))
    m["holm"]["P1"]["family_confirmed"] = False
    res["P1 no longer family-confirmed"] = refused(VR.continuation_check, m, {})
    check("refused: %s" % "; ".join(res), all(e is not None and "does not reproduce" in str(e) for e in res.values()),
          {k: str(v)[:120] for k, v in res.items()})
    clean()


def main():
    try:
        for fn in (t_platform_guard, t_reopen, t_verb_guard, t_guards_agree, t_record_only_follow, t_read_path,
                   t_reproduction):
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-1500:]))
    finally:
        shutil.rmtree(str(VT.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
