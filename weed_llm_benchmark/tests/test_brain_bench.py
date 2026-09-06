#!/usr/bin/env python3
"""WP2 benchmark-harness tests — the scorer, on a synthetic corpus, no model.

Everything here runs offline: six hand-built cases in a temporary directory
(four incidents, two healthy controls) and `bench.FakeModel`, which is a
keyword matcher, not a model. No cluster, no Mongo, no provider, no sleeping.

What the story protects, in the words of the failures it comes from:
  * A0 is the scripted scheduler and its detection is a definition in code, not
    a description: the right step recorded `failed` AND the domain paused
    within one tick. On 2026-08-29 the first TIMEOUT was recorded failed and
    the identical command was resubmitted, so A0 detected nothing until a
    second 12 h job had burnt. A test that let A0 "detect" the first failure
    would hand every later arm a baseline that never existed.
  * Every number is a margin over A0+ and every margin carries a minimum
    detectable difference. At n = 4 incidents nothing here is a result, and the
    harness has to say so itself rather than leave it to the slide.
  * A quote that does not resolve into the bundle is not evidence. L1 sees
    status fields only, so when it does fire it cites nothing that resolves —
    that separation is the point of the information ablation and is asserted,
    not assumed.
  * `--reproduce` must re-score committed verdicts to the same numbers with no
    model call, or the published table cannot be checked by anyone.

Run:  python -m pytest tests/test_brain_bench.py
 (or) python tests/test_brain_bench.py
"""
import json
import math
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import bench  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


def ck_eq(name, got, want, tol=1e-9):
    if isinstance(want, float) and isinstance(got, (int, float)) and got is not None:
        cond = abs(float(got) - want) <= tol
    else:
        cond = got == want
    print(("  ok   " if cond else "  FAIL ") + name +
          ("" if cond else "   got %r want %r" % (got, want)))
    if not cond:
        _fails.append(name)


# ============================================================ synthetic corpus

# Two consecutive train jobs at the 12 h walltime, a filter step that recorded
# `done` while its gate did nothing, a design-class case with no artifact, and
# two healthy controls — one of which mentions an older timeout in passing,
# which is what a false alarm looks like.
C1_LINES = [
    [4120, "                24/60      12.4G      1.021      0.734      0.912        139        640"],
    [4125, "[rounds] weed: train job 44727703 projected 104400s over walltime 43200s - signal walltime_bound"],
    [4127, "slurmstepd: error: *** JOB 44727703 ON v012 CANCELLED AT 2026-08-29T05:53:00 DUE TO TIME LIMIT ***"],
]
C2_LINES = [
    [3005, "epoch 16/60 complete elapsed 2701.5s eta_total 162090s walltime 43200s"],
    [3009, "pool iterations per epoch 8583 previous round 6246 growth 37 percent"],
    [3011, "slurmstepd: error: *** JOB 44767709 ON v015 CANCELLED AT 2026-08-29T17:53:00 DUE TO TIME LIMIT ***"],
]
C3_LINES = [
    [812, "merge: kept 8583 of 8583 candidates from tier curated (slug_scores.json missing)"],
    [815, "merge complete writing dataset yaml with 12 classes and 8583 images"],
]
H1_LINES = [
    [990, "epoch 60/60 complete elapsed 601.2s eta_total 36072s walltime 43200s"],
    [992, "results saved to results/framework/mega_iterrnd2_train_44100001/train"],
]
H2_LINES = [
    [140, "note: the previous attempt was CANCELLED AT 2026-08-20T02:00:00 DUE TO TIME LIMIT; this run completed"],
    [142, "epoch 60/60 complete elapsed 640.0s eta_total 38400s walltime 43200s"],
]


def _out_tail(job, lines):
    return {"artifact_id": "job_%s.out" % job, "path": "/ocean/x/job_%s.out" % job,
            "sha256": "0" * 64, "lines": [list(x) for x in lines]}


def _ledger(round_no, step, status, at, detail, pause=None, tick_s=None,
            started_at=None):
    entry = {"status": status, "actor": "round-scheduler", "at": at,
             "detail": detail, "job": "x"}
    if started_at:
        entry["attempts"] = [{"status": "running", "actor": "round-scheduler",
                              "at": started_at, "job": "x"}]
    sec = {"domain": "weed",
           "rounds": [{"round_num": round_no, "steps": {step: entry}}]}
    if pause:
        sec["pause"] = pause
    if tick_s:
        sec["tick_s"] = tick_s
    return sec


def _case1():
    bundle = {
        "bundle_id": "b1", "sha256": "", "domain": "weed", "round": 4,
        "step": "train", "built_ts": "2026-08-29T06:00:00+00:00",
        "sections": {
            "ledger": _ledger(4, "train", "failed", "2026-08-29T05:53:00+00:00",
                              "job state TIMEOUT after 12h walltime on the merged "
                              "pool (signal walltime_bound)",
                              started_at="2026-08-28T17:53:00+00:00"),
            "sacct": [{"JobID": "44727703", "State": "TIMEOUT", "Elapsed": "12:00:07"}],
            "out_tail": _out_tail("44727703", C1_LINES),
            "results_csv": {"rows": 24, "best": 0.581, "last": 0.577,
                            "epoch_time_s": 1740.2},
            "strategy": {"epochs": 60, "tier": "curated", "time_h": None,
                         "iter_name": "rnd4_train_44727703"},
            "signals": [{"signal": "walltime_bound", "severity": "crit",
                         "value": {"eta_total_s": 104400, "walltime_s": 43200},
                         "ts": "2026-08-29T05:40:00+00:00",
                         "evidence": [{"artifact_id": "job_44727703.out",
                                       "line": 4125, "quote": C1_LINES[1][1]}]}],
        },
        "token_estimate": 900, "caps": {},
    }
    truth = {
        "case_id": "inc_walltime_first", "date": "2026-08-29", "incident": True,
        "class": "operational", "signals_expected": ["walltime_bound"],
        "load_bearing_lines": [{"artifact": "job_44727703.out", "line": 4125}],
        "acceptable_corrections": [{"action": "set_round_param",
                                    "params_range": {"epochs": [1, 30]},
                                    "risk": "R1"}],
        "escalation_expected": "tier1", "provenance": "raw",
        "labels": {"pre_registered": {"incident": True}, "adjudicated": {}},
        "notes": "first of the two 12 h TIMEOUTs; recorded failed, no pause",
    }
    return bundle, truth


def _case2():
    bundle = {
        "bundle_id": "b2", "sha256": "", "domain": "weed", "round": 5,
        "step": "train", "built_ts": "2026-08-29T18:00:00+00:00",
        "sections": {
            "ledger": _ledger(5, "train", "failed", "2026-08-29T17:53:00+00:00",
                              "job state TIMEOUT after 12h walltime on the merged pool",
                              pause={"reason": "stop-loss: 2 consecutive step failures",
                                     "at": "2026-08-29T17:53:40+00:00"}),
            "sacct": [{"JobID": "44767709", "State": "TIMEOUT", "Elapsed": "12:00:04"}],
            "out_tail": _out_tail("44767709", C2_LINES),
            "results_csv": {"rows": 16, "best": 0.562, "last": 0.559,
                            "epoch_time_s": 2701.5},
            "strategy": {"epochs": 60, "tier": "curated", "time_h": None},
            "signals": [
                {"signal": "walltime_bound", "severity": "crit",
                 "value": {"eta_total_s": 162090, "walltime_s": 43200},
                 "ts": "2026-08-29T17:20:00+00:00",
                 "evidence": [{"artifact_id": "job_44767709.out", "line": 3011,
                               "quote": C2_LINES[2][1]}]},
                {"signal": "pool_growth", "severity": "warn", "value": 0.37,
                 "evidence": [{"artifact_id": "job_44767709.out", "line": 3009,
                               "quote": C2_LINES[1][1]}]}],
        },
        "token_estimate": 950, "caps": {},
    }
    truth = {
        "case_id": "inc_walltime_second", "date": "2026-08-29", "incident": True,
        "class": "operational", "signals_expected": ["walltime_bound", "pool_growth"],
        "load_bearing_lines": [{"artifact": "job_44767709.out", "line": 3011}],
        "acceptable_corrections": [{"action": "set_round_param",
                                    "params_range": {"epochs": [1, 30]},
                                    "risk": "R1"}],
        "escalation_expected": "human", "provenance": "raw",
        "incident_ts": "2026-08-29T17:00:00+00:00",
        "labels": {"pre_registered": {"incident": True}, "adjudicated": {}},
        "notes": "second TIMEOUT; stop-loss paused the domain in the same tick",
    }
    return bundle, truth


def _case3():
    bundle = {
        "bundle_id": "b3", "sha256": "", "domain": "weed", "round": 6,
        "step": "filter", "built_ts": "2026-08-20T09:00:00+00:00",
        "sections": {
            "ledger": _ledger(6, "filter", "done", "2026-08-20T08:59:00+00:00",
                              "filter step completed"),
            "sacct": [{"JobID": "45010101", "State": "COMPLETED", "Elapsed": "00:31:02"}],
            "out_tail": _out_tail("45010101", C3_LINES),
            "slug_scores": {"n": 0, "unscored": 41, "median": None},
            "strategy": {"tier": "curated", "min_dino_score": 0.50},
            "signals": [{"signal": "gate_noop", "severity": "warn",
                         "value": {"kept": 8583, "raw": 8583},
                         "evidence": [{"artifact_id": "job_45010101.out",
                                       "line": 812, "quote": C3_LINES[0][1]}]}],
        },
        "token_estimate": 600, "caps": {},
    }
    truth = {
        "case_id": "inc_silent_gate", "date": "2026-08-20", "incident": True,
        "class": "config", "signals_expected": ["gate_noop"],
        "load_bearing_lines": [{"artifact": "job_45010101.out", "line": 812}],
        "acceptable_corrections": [{"action": "requeue_step",
                                    "params_range": {"step": {"in": ["filter"]}},
                                    "risk": "R2"}],
        "escalation_expected": "tier1", "provenance": "raw",
        "labels": {"pre_registered": {"incident": True}, "adjudicated": {}},
        "notes": "curated tier merged every raw candidate; the step recorded done",
    }
    return bundle, truth


def _case4():
    bundle = {
        "bundle_id": "b4", "sha256": "", "domain": "weed", "round": 7,
        "step": "train", "built_ts": "2026-08-10T12:00:00+00:00",
        "sections": {
            "ledger": _ledger(7, "train", "done", "2026-08-10T11:00:00+00:00",
                              "train step completed"),
            "sacct": [{"JobID": "44900001", "State": "COMPLETED", "Elapsed": "09:12:00"}],
            "strategy": {"epochs": 60, "recipe": "mamba-t", "fresh_start": True},
            "signals": [],
        },
        "token_estimate": 200, "caps": {},
    }
    truth = {
        "case_id": "inc_design_confound", "date": "2026-08-10", "incident": True,
        "class": "design", "signals_expected": [],
        "load_bearing_lines": [], "acceptable_corrections": [],
        "escalation_expected": "tier2", "provenance": "record-only",
        "labels": {"pre_registered": {"incident": True}, "adjudicated": {}},
        "notes": "architecture compared against a differently initialised control",
    }
    return bundle, truth


def _healthy(cid, job, lines, round_no, signals):
    bundle = {
        "bundle_id": "h" + cid, "sha256": "", "domain": "weed", "round": round_no,
        "step": "train", "built_ts": "2026-08-15T12:00:00+00:00",
        "sections": {
            "ledger": _ledger(round_no, "train", "done", "2026-08-15T11:59:00+00:00",
                              "train step completed"),
            "sacct": [{"JobID": job, "State": "COMPLETED", "Elapsed": "10:02:11"}],
            "out_tail": _out_tail(job, lines),
            "results_csv": {"rows": 60, "best": 0.601, "last": 0.598,
                            "epoch_time_s": 601.2},
            "strategy": {"epochs": 60, "tier": "curated"},
            "signals": signals,
        },
        "token_estimate": 500, "caps": {},
    }
    truth = {
        "case_id": cid, "date": "2026-08-15", "incident": False,
        "class": "operational", "signals_expected": [], "load_bearing_lines": [],
        "acceptable_corrections": [], "escalation_expected": "none",
        "provenance": "raw",
        "labels": {"pre_registered": {"incident": False}, "adjudicated": {}},
        "notes": "healthy control, re-audited against every signal",
    }
    return bundle, truth


def build_corpus(root):
    """Write the six cases, one raw artifact and a self-hashed split.json."""
    root = pathlib.Path(root)
    cases = [("inc_walltime_first",) + _case1(),
             ("inc_walltime_second",) + _case2(),
             ("inc_silent_gate",) + _case3(),
             ("inc_design_confound",) + _case4(),
             ("healthy_round2_train",) + _healthy("healthy_round2_train",
                                                  "44100001", H1_LINES, 2, []),
             ("healthy_noisy_log",) + _healthy(
                 "healthy_noisy_log", "44200002", H2_LINES, 3,
                 [{"signal": "source_degraded", "severity": "info",
                   "value": {"source": "github"},
                   "evidence": [{"artifact_id": "job_44200002.out", "line": 140,
                                 "quote": H2_LINES[0][1]}]}])]
    for cid, bundle, truth in cases:
        d = root / "cases" / cid
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(d / "bundle.json", "w") as f:
            json.dump(bundle, f, indent=2, sort_keys=True)
        with open(d / "truth.json", "w") as f:
            json.dump(truth, f, indent=2, sort_keys=True)
    # One real line-numbered artifact, for the retrieval round: 300 lines whose
    # numbers are absolute, exactly as corpus.py writes them.
    art = root / "cases" / "inc_walltime_first" / "artifacts" / "job_44727703.out"
    with open(art, "w") as f:
        for i in range(1, 301):
            text = "training line %d of the archived job output" % i
            if i == 288:
                text = "retrieved-only detail: dataloader workers 8 pin_memory True"
            f.write("%6d\t%s\n" % (i, text))
    split = {"dev": ["inc_design_confound"],
             "test": ["inc_walltime_first", "inc_walltime_second",
                      "inc_silent_gate", "healthy_round2_train",
                      "healthy_noisy_log"],
             "rule": "dev = incidents dated <= 2026-08-25 kept out of reported "
                     "numbers; test = everything else"}
    split["sha256"] = bench._sha256(split)
    with open(root / "split.json", "w") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    return root


TMP = pathlib.Path(tempfile.mkdtemp(prefix="bench_test_"))
ROOT = build_corpus(TMP / "supervision_bench")
OUT = TMP / "out"


# ============================================================ A0's definition

def a0b(entry, pause=None, tick_s=None, ledger=True, round_no=5, step="train"):
    b = {"round": round_no, "step": step, "sections": {}}
    if ledger:
        sec = {"rounds": [{"round_num": round_no, "steps": {step: entry}}]}
        if pause:
            sec["pause"] = pause
        if tick_s:
            sec["tick_s"] = tick_s
        b["sections"]["ledger"] = sec
    return b


T0 = "2026-08-29T17:53:00+00:00"
FAILED = {"status": "failed", "actor": "round-scheduler", "at": T0,
          "detail": "job state TIMEOUT"}
PAUSE_40 = {"reason": "stop-loss: 2 consecutive step failures",
            "at": "2026-08-29T17:53:40+00:00"}

print("\n-- A0: the scripted scheduler's own definition --")
d = bench.a0_decision(a0b(FAILED, pause=PAUSE_40))
ck("A0 fires on failed + pause inside one tick", d["fired"] is True)
ck_eq("A0 records the gap it measured", d["dt_s"], 40.0)

d = bench.a0_decision(a0b(FAILED))
ck("A0 misses a first failure with no pause (the 08-29 behaviour)", d["fired"] is False)
ck("A0 says why it missed", "not recorded paused" in d["reason"])

d = bench.a0_decision(a0b({"status": "done", "at": T0}, pause=PAUSE_40))
ck("A0 misses a step recorded done (the silent-wrongness class)", d["fired"] is False)
ck("A0 names the status it saw", "recorded done" in d["reason"])

d = bench.a0_decision(a0b(FAILED, pause={"reason": "stop-loss",
                                         "at": "2026-08-29T18:03:00+00:00"}))
ck("A0 misses a pause later than one tick", d["fired"] is False)
ck("A0 quotes the tick it compared against", "more than one tick" in d["reason"])

d = bench.a0_decision(a0b(FAILED, pause={"reason": "stop-loss",
                                         "at": "2026-08-29T17:58:00+00:00"},
                          tick_s=600))
ck("A0 honours a tick_s carried by the bundle", d["fired"] is True)

d = bench.a0_decision(a0b(FAILED, pause={"reason": "an earlier stop-loss",
                                         "at": "2026-08-29T16:53:00+00:00"}))
ck("A0 rejects a pause that precedes the failure", d["fired"] is False)
ck("A0 says the pause belongs elsewhere", "earlier incident" in d["reason"])

d = bench.a0_decision({"round": 5, "step": "train", "sections": {}})
ck("A0 is undecidable with no ledger section", d["undecidable"] is True and not d["fired"])

d = bench.a0_decision(a0b({"status": "failed", "at": None}, pause=PAUSE_40))
ck("A0 is undecidable when the failure has no timestamp", d["undecidable"] is True)

head_running = {"status": "running", "at": "2026-08-29T17:55:00+00:00",
                "attempts": [dict(FAILED)]}
d = bench.a0_decision(a0b(head_running, pause=PAUSE_40))
ck("A0 reads the failure out of attempts[] under a newer head", d["fired"] is True)

superseded = {"status": "skipped", "at": T0,
              "detail": "superseded: cancelled by the supervisor"}
d = bench.a0_decision(a0b(superseded, pause=PAUSE_40))
ck("A0 does not read a superseded cancel as a failure", d["fired"] is False)


# ============================================================ statistics

print("\n-- Wilson intervals against published values --")
w = bench.wilson(0, 10)
ck_eq("wilson 0/10 lower", round(w["lo"], 4), 0.0)
ck_eq("wilson 0/10 upper", round(w["hi"], 4), 0.2775)
w = bench.wilson(1, 10)
ck_eq("wilson 1/10 lower", round(w["lo"], 4), 0.0179)
ck_eq("wilson 1/10 upper", round(w["hi"], 4), 0.4042)
w = bench.wilson(5, 10)
ck_eq("wilson 5/10 lower", round(w["lo"], 4), 0.2366)
ck_eq("wilson 5/10 upper", round(w["hi"], 4), 0.7634)
w = bench.wilson(10, 10)
ck_eq("wilson 10/10 lower", round(w["lo"], 4), 0.7225)
ck_eq("wilson 10/10 upper", round(w["hi"], 4), 1.0)
ck("wilson n=0 has no point estimate", bench.wilson(0, 0)["p"] is None)
ck("wilson clamps k above n", bench.wilson(99, 10)["k"] == 10)

print("\n-- minimum detectable difference --")
m58 = bench.mdd_proportion(58, 0.5)
ck("MDD at n=58, p0=0.5 is the textbook 0.25 for 0.50 vs 0.75",
   0.24 < m58 < 0.26)
ck_eq("power at that MDD is the requested 0.80",
      round(bench.power_two_proportions(58, 0.5, 0.5 + m58), 2), 0.80)
ck("power below the MDD is under 0.80",
   bench.power_two_proportions(58, 0.5, 0.5 + m58 - 0.03) < 0.80)
m12 = bench.mdd_proportion(12, 0.5)
ck("MDD at n=12 is 0.478 — a dozen cases resolve nothing smaller",
   abs(m12 - 0.478) < 0.005)
ck("MDD shrinks as n grows", bench.mdd_proportion(200, 0.5) < m58 < m12)
ck("MDD is None when nothing is detectable at that n",
   bench.mdd_proportion(4, 0.75) is None)
ck("MDD is None for an empty arm", bench.mdd_proportion(0, 0.5) is None)
mm = bench.mdd_mean(30, 2.0)
ck("MDD on means scales with sd/sqrt(n)",
   abs(mm - (1.959963984540054 + 0.8416212335729141) * 2.0 * math.sqrt(2.0 / 30)) < 1e-9)
ck("MDD on means is None without a spread", bench.mdd_mean(30, 0.0) is None)

print("\n-- margin arithmetic and the MDD attached to it --")


def fake_agg(arm, k, n):
    st = bench.wilson(k, n)
    st.update(unit="case", subset="incident", better="higher", kind="prop")
    return {"arm": arm, "model": "", "counts": {},
            "groups": {"all": {"detection_recall": st}}}


rows = bench.compare({"A0p": fake_agg("A0p", 30, 60), "L3": fake_agg("L3", 36, 60)})
row = [r for r in rows if r["metric"] == "detection_recall"][0]
ck_eq("margin over A0+ is the difference of the two rates", round(row["delta"], 6), 0.1)
ck("a 10-point margin at n=60 is below the MDD", row["below_mdd"] is True)
ck("the margin's CI brackets the margin",
   row["delta_ci"][0] < row["delta"] < row["delta_ci"][1])
rows = bench.compare({"A0p": fake_agg("A0p", 30, 60), "L3": fake_agg("L3", 48, 60)})
row = [r for r in rows if r["metric"] == "detection_recall"][0]
ck_eq("a 30-point margin at n=60", round(row["delta"], 6), 0.3)
ck("a 30-point margin at n=60 clears the MDD", row["below_mdd"] is False)
report = bench.render_report({"comparisons": rows, "arms": [], "per_arm": {},
                              "split": {}, "run_id": "t"})
ck("the report prints an MDD column", "mdd" in report)
rows_small = bench.compare({"A0p": fake_agg("A0p", 30, 60), "L3": fake_agg("L3", 33, 60)})
report_small = bench.render_report({"comparisons": rows_small, "arms": [],
                                    "per_arm": {}, "split": {}, "run_id": "t"})
ck("the report marks a margin under the MDD as not a result",
   "below MDD - not a result" in report_small)


# ============================================================ corpus and split

print("\n-- corpus, split and their hashes --")
CORPUS = bench.load_corpus(ROOT)
ck_eq("six cases load", len(CORPUS["cases"]), 6)
ck("no load errors", CORPUS["errors"] == [])
ck("corpus hash is stable across loads",
   bench.load_corpus(ROOT)["hash"] == CORPUS["hash"])
SPLIT = bench.load_split(ROOT)
ck("split.json certifies itself", SPLIT["sha_ok"] is True)
known, missing = bench.select_cases(CORPUS, SPLIT, "test")
ck_eq("test split names five cases", len(known), 5)
ck("the dev case is not in the test split", "inc_design_confound" not in known)
ck("nothing in the split is missing from the corpus", missing == [])

_bad = json.loads((ROOT / "split.json").read_text())
_bad["test"] = _bad["test"] + ["inc_design_confound"]
(ROOT / "split_bad.json").write_text(json.dumps(_bad))
shutil.copy(ROOT / "split.json", ROOT / "split_good.json")
shutil.copy(ROOT / "split_bad.json", ROOT / "split.json")
_check = bench.load_split(ROOT)
ck("an edited split fails its own hash", _check["sha_ok"] is False)
ck("an edited split reports dev/test overlap",
   any("overlap" in e for e in _check["errors"]))
shutil.copy(ROOT / "split_good.json", ROOT / "split.json")

CASE1 = bench.load_case(ROOT / "cases" / "inc_walltime_first")
ck("a case carries its artifacts index", "job_44727703.out" in CASE1["artifacts"])
_broken = bench.load_case(ROOT / "cases" / "does_not_exist")
ck("a missing case reports instead of raising",
   _broken["ok"] is False and _broken["errors"])


# ============================================================ the arms end to end

print("\n-- arms over the corpus (FakeModel, no provider) --")
FAKE = bench.FakeModel(
    triggers={"TIME LIMIT": "walltime_bound"},
    corrections=[{"action": "set_round_param", "params": {"epochs": 45},
                  "risk": "R1", "reason": "cap the epoch budget", "quote": ""}],
    escalate="tier1", su_per_call=0.05)
RES = bench.run(root=ROOT, arms=["A0", "A0p", "A0pp", "L1", "L2", "L3"],
                split_name="all", repeats=3, model_fn=FAKE, model="fake",
                num_ctx=32768, out_dir=OUT)


def stat(arm, metric, group="all"):
    return ((RES["per_arm"].get(arm) or {}).get("groups") or {}).get(group, {}).get(metric, {})


def margin(arm, metric, group="all"):
    for r in RES["comparisons"]:
        if r["arm"] == arm and r["metric"] == metric and r["group"] == group:
            return r
    return {}


ck("the run produced no errors", RES["errors"] == [])
ck_eq("four incidents in the detection subset", stat("A0", "detection_recall")["n"], 4)
ck_eq("A0 detects one of four — only the failure that also paused",
      stat("A0", "detection_recall")["k"], 1)
ck_eq("A0+ detects three of four (the design case has no signal)",
      stat("A0p", "detection_recall")["k"], 3)
ck_eq("A0++ detects what A0+ detects", stat("A0pp", "detection_recall")["k"], 3)
ck_eq("L1 (status fields only) detects none of them",
      stat("L1", "detection_recall")["k"], 0)
ck_eq("L2 (raw excerpts, signals hidden) detects two",
      stat("L2", "detection_recall")["k"], 2)
ck_eq("L3 (signals + excerpts) detects two", stat("L3", "detection_recall")["k"], 2)

ck_eq("two healthy controls in the false-alarm subset",
      stat("A0p", "false_alarm_rate")["n"], 2)
ck_eq("A0+ raises no false alarm — the info-severity signal is below the bar",
      stat("A0p", "false_alarm_rate")["k"], 0)
ck_eq("A0 raises no false alarm", stat("A0", "false_alarm_rate")["k"], 0)
ck_eq("L3 fires on the healthy log that mentions an older timeout",
      stat("L3", "false_alarm_rate")["k"], 1)

ck_eq("three cases are in the R1/R2-correctable subset",
      stat("A0p", "correction_correct")["n"], 3)
ck_eq("A0+ corrects the two walltime cases and not the gate",
      stat("A0p", "correction_correct")["k"], 2)
ck_eq("L3's out-of-range epochs correct nothing",
      stat("L3", "correction_correct")["k"], 0)

ck_eq("three incidents carry load-bearing lines",
      stat("A0p", "evidence_hit_rate")["n"], 3)
ck_eq("A0+ quotes a load-bearing line in all three",
      stat("A0p", "evidence_hit_rate")["k"], 3)
ck_eq("L3 quotes the load-bearing line in one of three",
      stat("L3", "evidence_hit_rate")["k"], 1)

ck_eq("A0 cites nothing, so citation validity has no denominator",
      stat("A0", "citation_validity")["n"], 0)
ck_eq("A0 findings without a quote are counted instead",
      RES["per_arm"]["A0"]["counts"]["findings_without_quote"], 1.0)
ck_eq("A0+ quotes four signal evidence lines", stat("A0p", "citation_validity")["n"], 4)
ck_eq("all four resolve", stat("A0p", "citation_validity")["k"], 4)
ck_eq("L3 quotes three lines", stat("L3", "citation_validity")["n"], 3)
ck_eq("all three resolve", stat("L3", "citation_validity")["k"], 3)

ck_eq("escalation: A0 is right only where nothing was expected",
      stat("A0", "escalation_ok")["k"], 2)
ck_eq("escalation: A0+ has no channel and matches the same two",
      stat("A0p", "escalation_ok")["k"], 2)
ck_eq("escalation: A0++ pages a person and adds the stop-loss case",
      stat("A0pp", "escalation_ok")["k"], 3)

ck_eq("deterministic arms cost no service units",
      stat("A0p", "su_per_review")["mean"], 0.0)
ck_eq("the model arm's SU comes from the callable",
      round(stat("L3", "su_per_review")["mean"], 6), 0.05)

print("\n-- margins over A0+ --")
ck_eq("A0 detection margin over A0+", round(margin("A0", "detection_recall")["delta"], 6), -0.5)
ck_eq("L1 detection margin over A0+", round(margin("L1", "detection_recall")["delta"], 6), -0.75)
ck_eq("L3 detection margin over A0+", round(margin("L3", "detection_recall")["delta"], 6), -0.25)
ck_eq("L3 false-alarm margin over A0+", round(margin("L3", "false_alarm_rate")["delta"], 6), 0.5)
ck_eq("L3 evidence margin over A0+",
      round(margin("L3", "evidence_hit_rate")["delta"], 6), round(1 / 3 - 1.0, 6))
ck("no detection/evidence margin at this corpus size is presented as a result",
   all(r.get("below_mdd") in (True, None) for r in RES["comparisons"]
       if r.get("delta") is not None and r.get("kind") in ("prop", "ratio")))
ck("the prompt-size difference between arms is above its own MDD — a real "
   "measured difference, unlike every rate here",
   [r for r in RES["comparisons"] if r["arm"] == "L3" and r["group"] == "all"
    and r["metric"] == "tokens_in"][0]["below_mdd"] is False)
ck("a cost difference measured without noise is not buried under an MDD",
   margin("L3", "su_per_review")["mdd"] == 0.0
   and margin("L3", "su_per_review")["below_mdd"] is False)
ck("the baseline is A0+", RES["baseline"] == "A0p")
ck("A0+ is not compared against itself",
   not any(r["arm"] == "A0p" for r in RES["comparisons"]))

print("\n-- notification latency belongs only to the arm that has a channel --")
ck("A0 declares no notification channel",
   RES["per_arm"]["A0"]["notify_latency_s"]["n"] == 0
   and all(r["notify_latency_s"] is None for r in RES["matrix"] if r["arm"] == "A0"))
ck("A0+ has no notification latency either",
   RES["per_arm"]["A0p"]["notify_latency_s"]["n"] == 0)
ck("A0++ reports a replayed latency for the two cases that carry an onset",
   RES["per_arm"]["A0pp"]["notify_latency_s"]["n"] == 2)
_lat = [r for r in RES["matrix"]
        if r["arm"] == "A0pp" and r["case_id"] == "inc_walltime_second"][0]
ck_eq("measured from the onset the truth states, plus the alarm's own delay",
      _lat["notify_latency_s"], 20 * 60 + bench.ALARM_SURFACE_S)
_notes = {r["case_id"]: (r["meta"] or {}).get("notify_latency_note")
          for r in bench.load_verdicts(OUT / "verdicts")[0]
          if r["arm"] == "A0pp" and r["repeat"] == 0}
ck("every latency states which zero point it was measured from",
   "truth.incident_ts" in _notes["inc_walltime_second"]
   and "step's start" in _notes["inc_walltime_first"])
ck("a case with neither onset says so instead of reporting a zero",
   "no onset" in _notes["inc_silent_gate"])


# ============================================================ citation scoring

print("\n-- citations and evidence, scored directly --")


def rec(case, findings, corrections=None, verdict="issue", escalate="none", meta=None):
    return {"case_id": case["case_id"], "arm": "X", "repeat": 0, "model": "",
            "verdict": {"verdict": verdict, "findings": findings,
                        "corrections": corrections or [],
                        "escalate": {"to": escalate, "reason": ""},
                        "confidence": 0.9},
            "meta": meta or {}, "bundle_sha256": case["bundle_sha256"]}


def find(quote, severity="crit", signal="walltime_bound"):
    return [{"signal": signal, "quote": quote, "diagnosis": "d", "severity": severity}]


s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[2][1])))
ck("a verbatim line resolves", s["citations_valid"] == 1 and s["citations_total"] == 1)
ck("the incident is detected", s["detected"] is True)
ck("a grounded detection needs a resolving quote", s["detected_grounded"] is True)
ck("a quote that is not load-bearing is no evidence hit", s["evidence_hit"] is False)

s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[1][1])))
ck("quoting the load-bearing line is an evidence hit", s["evidence_hit"] is True)
ck_eq("and it counts one of the case's load-bearing lines", s["evidence_hits"], 1)
ck_eq("out of the one the truth names", s["evidence_total"], 1)

s = bench.score_record(CASE1, rec(CASE1, find("DUE TO TIME LIMIT")))
ck("a quote under the 20-character rule does not resolve",
   s["citations_valid"] == 0 and s["citations_total"] == 1)
ck("an ungrounded detection is still a detection, and is marked",
   s["detected"] is True and s["detected_grounded"] is False)

s = bench.score_record(CASE1, rec(CASE1, find(
    "the training job was cancelled because the reviewer asked for it")))
ck("a fabricated quote does not resolve", s["citations_valid"] == 0)

s = bench.score_record(CASE1, rec(CASE1, find("")))
ck("a finding with no quote is counted apart from the citation ratio",
   s["findings_without_quote"] == 1 and s["citations_total"] == 0)

s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[2][1], severity="info")))
ck("an info-severity finding is below the pre-registered detection bar",
   s["fired"] is False and s["detected"] is False)
ck("but it is not silence either — it is recorded as unsupported",
   s["fired_unsupported"] is True)

s = bench.score_record(CASE1, rec(CASE1, []))
ck("`issue` with no finding is not a detection",
   s["fired"] is False and s["fired_unsupported"] is True)

s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[2][1]), verdict="ok"))
ck("findings under an `ok` verdict do not fire", s["fired"] is False)

HEALTHY = bench.load_case(ROOT / "cases" / "healthy_noisy_log")
s = bench.score_record(HEALTHY, rec(HEALTHY, find(H2_LINES[0][1])))
ck("firing on a healthy control is a false alarm, not a detection",
   s["false_alarm"] is True and s["detected"] is False)
s = bench.score_record(HEALTHY, rec(HEALTHY, [], verdict="ok"))
ck("staying quiet on a healthy control is no false alarm", s["false_alarm"] is False)

print("\n-- escalation appropriateness --")
s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[2][1]), escalate="tier1"))
ck("escalating exactly as the truth expects", s["escalation_ok"] is True)
s = bench.score_record(CASE1, rec(CASE1, find(C1_LINES[2][1]), escalate="human"))
ck("over-escalating is not appropriate either", s["escalation_ok"] is False)


# ============================================================ corrections

print("\n-- correction correctness on the R1/R2 subset --")
ck("a two-number list is a range", bench._in_range(30, [1, 30]) is True)
ck("and it excludes what falls outside", bench._in_range(45, [1, 30]) is False)
ck("min/max form", bench._in_range(12, {"min": 1, "max": 30}) is True)
ck("explicit range form", bench._in_range(12, {"range": [1, 30]}) is True)
ck("membership form", bench._in_range("filter", {"in": ["filter"]}) is True)
ck("membership form rejects a stranger", bench._in_range("train", {"in": ["filter"]}) is False)
ck("a scalar spec is equality", bench._in_range("curated", "curated") is True)
ck("a non-numeric value is not in a numeric range", bench._in_range("many", [1, 30]) is False)

acceptable = [{"action": "set_round_param", "params_range": {"epochs": [1, 30]},
               "risk": "R1"}]
ok, note = bench.correction_match(
    [{"action": "set_round_param", "params": {"epochs": 30}}], acceptable)
ck("in-range correction matches", ok is True)
ok, note = bench.correction_match(
    [{"action": "set_round_param", "params": {"epochs": 45}}], acceptable)
ck("out-of-range correction does not", ok is False and "out of range" in note)
ok, note = bench.correction_match(
    [{"action": "quarantine_slug", "params": {"slug": "x"}}], acceptable)
ck("a different action does not", ok is False and "not in the acceptable set" in note)
ok, note = bench.correction_match(
    [{"action": "set_round_param", "params": {"time_h": 8}}], acceptable)
ck("a proposal missing the constrained parameter does not",
   ok is False and "missing epochs" in note)
ok, note = bench.correction_match([], acceptable)
ck("no proposal is not a correction", ok is False and "no correction proposed" in note)
ok, note = bench.correction_match(
    [{"action": "set_round_param", "params": {"epochs": 30, "imgsz": 512}}], acceptable)
ck("an unconstrained extra parameter is allowed and named",
   ok is True and "unconstrained" in note)

resolved, unresolved = bench._resolve_correction_params({}, {"epochs": "half"})
ck("a symbolic correction with nothing to halve stays symbolic and is flagged",
   resolved == {"epochs": "half"} and unresolved == ["epochs"])
r = bench.arm_signals(CASE1, {}, notify=False)
ck_eq("A0+ halves the epochs the step actually ran with",
      r["verdict"]["corrections"][0]["params"]["epochs"], 30)
ck("and cites the line the signal fired on",
   r["verdict"]["corrections"][0]["quote"] == C1_LINES[1][1])


# ============================================================ what each arm sees

print("\n-- information ablation: the views are actually different --")
V1 = bench.view_for_arm(CASE1, "L1")
P1 = bench.render_prompt(V1)
ck("L1 has no raw artifact section", "out_tail" not in V1["sections"])
ck("L1 has no signals", "signals" not in V1["sections"])
ck("L1 cannot see the artifact line", "DUE TO TIME LIMIT" not in P1)
ck("L1 does see the job's own status field", "State: TIMEOUT" in P1)
ck("L1 does see the ledger step status", "steps.train.status: failed" in P1)

V2 = bench.view_for_arm(CASE1, "L2")
P2 = bench.render_prompt(V2)
V3 = bench.view_for_arm(CASE1, "L3")
P3 = bench.render_prompt(V3)
ck("L2 drops the signals section", V2["note"]["removed"] == ["signals"])
ck("L2's prompt has no signals section", "== signals ==" not in P2)
ck("L3's prompt does", "== signals ==" in P3)
ck("L2 still carries the raw lines", "DUE TO TIME LIMIT" in P2)
ck("L2 scrubs the covering signal's name where it leaks into other text",
   V2["note"]["redacted_values"] >= 1 and "[redacted: signal name]" in P2)
ck("L2 does not scrub a load-bearing line, and says how often that happened",
   V2["note"]["skipped_load_bearing"] == 1)
ck("so the signal name survives exactly on the line the truth needs",
   P2.count("walltime_bound") == 1)

print("\n-- L4's one bounded retrieval round --")
FAKE4 = bench.FakeModel(triggers={"retrieved-only detail": "stale_artifact"},
                        retrieve=["job_44727703.out", "not_a_file.out"])
arts, refused = bench.retrieve_artifacts(CASE1, ["job_44727703.out", "nope.out"])
ck_eq("retrieval reads the tail of the artifact", len(arts[0]["lines"]), 200)
ck_eq("with absolute line numbers, not offsets", arts[0]["lines"][0][0], 101)
ck("a name outside the whitelist is refused", refused == ["nope.out"])

R4 = bench.arm_model(CASE1, {"model_fn": FAKE4, "model": "fake", "num_ctx": 32768}, "L4")
ck_eq("L4 makes exactly two calls: ask, retrieve, answer", FAKE4.calls, 2)
ck_eq("and records the one round it was allowed", R4["meta"]["retrieval_rounds"], 1)
ck("the refused name is recorded", "not_a_file.out" in R4["meta"]["retrieval_refused"])
S4 = bench.score_record(CASE1, R4)
ck("the answer after retrieval fires", S4["fired"] is True)
ck("and its quote resolves into the retrieved lines", S4["citations_valid"] == 1)

FAKE3 = bench.FakeModel(triggers={"retrieved-only detail": "stale_artifact"},
                        retrieve=["job_44727703.out"])
R3 = bench.arm_model(CASE1, {"model_fn": FAKE3, "model": "fake", "num_ctx": 32768}, "L3")
ck_eq("L3 gets no retrieval round at all", FAKE3.calls, 1)
ck("so it never sees the line that was trimmed out of the bundle",
   bench.score_record(CASE1, R3)["fired"] is False)


# ============================================================ failure states

print("\n-- the harness reports its own failures instead of averaging them in --")
info = bench.signals_for({"sections": {}})
if bench._wp3("signals") is None:
    ck("with no signals module the arm is degraded, with the reason named",
       info["degraded"] is True and "not available" in info["reason"])
else:
    ck("the WP3 signals module is used when it is present",
       info["source"].startswith("module"))
ck("a bundle that froze its signals uses those",
   bench.signals_for(CASE1["bundle"])["source"] == "bundle")

_deg_case = {"case_id": "deg", "dir": "", "artifacts": {}, "ok": True,
             "bundle_sha256": "", "truth_sha256": "",
             "bundle": {"round": 1, "step": "train", "sections": {}},
             "truth": {"incident": True, "class": "operational"}}
_deg = bench.arm_signals(_deg_case, {}, notify=False)
if bench._wp3("signals") is None:
    ck("a case with no signals anywhere is marked degraded, not clean",
       _deg["meta"]["degraded"] is True)


def boom(prompt, model_id, num_ctx):
    raise RuntimeError("provider down")


RB = bench.arm_model(CASE1, {"model_fn": boom, "model": "m", "num_ctx": 4096}, "L3")
ck("a provider that raises is recorded, not propagated",
   "provider down" in RB["meta"]["model_error"])
ck("and scores as no detection", bench.score_record(CASE1, RB)["detected"] is False)

RN = bench.arm_model(CASE1, {"model_fn": None, "model": "", "num_ctx": 4096}, "L3")
ck("no callable is a recorded skip", "no model callable" in RN["meta"]["skipped"])

RO = bench.arm_model(CASE1, {"model_fn": bench.FakeModel(), "model": "m",
                             "num_ctx": 1}, "L3")
ck("a prompt that will not fit num_ctx is flagged before the call",
   RO["meta"]["context_overflow"] is True)

FENCED = bench.FakeModel(text_wrapper=lambda t: "Here you go:\n```json\n%s\n```\n" % t)
RF = bench.arm_model(CASE1, {"model_fn": FENCED, "model": "m", "num_ctx": 4096}, "L3")
ck("JSON wrapped in prose and a fence still parses",
   RF["verdict"]["verdict"] == "issue" and not RF["meta"].get("parse_error"))
RJ = bench.arm_model(CASE1, {"model_fn": lambda p, m, n: {"text": "I cannot help"},
                             "model": "m", "num_ctx": 4096}, "L3")
ck("a reply with no JSON is a parse error and no detection",
   RJ["meta"]["parse_error"] and RJ["verdict"]["verdict"] == "ok")

nv = bench.normalise_verdict({"verdict": "ISSUE", "findings": [{"quote": "q"}],
                              "escalate": {"to": "bogus"}, "confidence": 4})
ck("an unknown escalation target becomes none", nv["escalate"]["to"] == "none")
ck("a missing severity is info, not a detection", nv["findings"][0]["severity"] == "info")
ck("confidence is clamped", nv["confidence"] == 1.0)
ck("a non-object reply is an empty verdict",
   bench.normalise_verdict(None)["verdict"] == "ok")


# ============================================================ --reproduce

print("\n-- --reproduce: the same numbers from the committed verdicts --")
CALLS_BEFORE = FAKE.calls
REP = bench.reproduce(root=ROOT, verdicts_dir=OUT / "verdicts", out_dir=OUT,
                      split_name="all", write=False)
ck_eq("every committed verdict is re-read", REP["verdicts_read"], 6 * 6 * 3)
ck("no model was called", FAKE.calls == CALLS_BEFORE)
ck("the per-arm statistics are identical",
   bench._canonical(REP["per_arm"]) == bench._canonical(RES["per_arm"]))
ck("and so are the margins over A0+",
   bench._canonical(REP["comparisons"]) == bench._canonical(RES["comparisons"]))
ck("the corpus hash is carried through", REP["corpus_hash"] == RES["corpus_hash"])
ck("as is the rubric hash", REP["rubric_sha256"] == RES["rubric_sha256"])

ROOT2 = TMP / "tampered"
shutil.copytree(ROOT, ROOT2)
_b = json.loads((ROOT2 / "cases" / "inc_walltime_first" / "bundle.json").read_text())
_b["sections"]["out_tail"]["lines"].append([9999, "a line nobody reviewed"])
(ROOT2 / "cases" / "inc_walltime_first" / "bundle.json").write_text(json.dumps(_b))
REP2 = bench.reproduce(root=ROOT2, verdicts_dir=OUT / "verdicts", out_dir=ROOT2,
                       split_name="all", write=False)
ck("a verdict is not re-scored against a bundle it never saw",
   any("was produced against bundle" in w for w in REP2["warnings"]))
ck("and the case drops out of the count rather than changing the number",
   REP2["per_arm"]["A0p"]["groups"]["all"]["detection_recall"]["n"] == 3)


# ============================================================ files and CLI

print("\n-- artifacts on disk and the command line --")
ck("verdicts are written per arm and repeat",
   (OUT / "verdicts" / "A0p_fake" / "inc_walltime_first_r2.json").exists())
ck("a per-arm results file is written",
   (OUT / "results" / "A0p_fake.json").exists())
_r = json.loads((OUT / "results" / "A0p_fake.json").read_text())
ck("every results file carries the corpus and rubric hashes it was scored under",
   _r["corpus_hash"] == RES["corpus_hash"] and _r["rubric_sha256"] == bench.rubric_sha256())
ck("and the split it ran on", _r["split"]["name"] == "all")

import contextlib  # noqa: E402
import io  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_list = bench.main(["list", "--root", str(ROOT)])
    rc_run = bench.main(["run", "--root", str(ROOT), "--arms", "A0,A0p",
                         "--split", "test", "--repeats", "1", "--no-write"])
    rc_rep = bench.main(["--reproduce", "--root", str(ROOT),
                         "--verdicts", str(OUT / "verdicts")])
    rc_bad = bench.main(["run", "--root", str(ROOT), "--arms", "NOPE"])
    rc_none = bench.main(["list", "--root", str(TMP / "no_such_corpus")])
text = buf.getvalue()
ck("list exits clean on a good corpus", rc_list == 0)
ck("run exits clean", rc_run == 0)
ck("--reproduce is accepted as a flag, before the options", rc_rep == 0)
ck("an unknown arm is refused", rc_bad == 2)
ck("a missing corpus reports instead of raising", rc_none == 1)
ck("the report names the baseline every arm is a margin over", "margins over A0p" in text)
ck("the report prints the MDD beside every comparison", "mdd" in text)
ck("the report shows the split's self-hash check", "self-hash ok" in text)

print("\n-- two models in one verdicts tree are not one arm --")
OUT2 = TMP / "out2"
bench.run(root=ROOT, arms=["A0p", "L3"], split_name="all", repeats=1,
          model_fn=bench.FakeModel(), model="mA", out_dir=OUT2)
bench.run(root=ROOT, arms=["L3"], split_name="all", repeats=1,
          model_fn=bench.FakeModel(triggers={"epoch 16/60": "pool_growth"}),
          model="mB", out_dir=OUT2)
MIX = bench.reproduce(root=ROOT, verdicts_dir=OUT2 / "verdicts", split_name="all")
ck("one arm run with two models is aggregated as two",
   "L3@mA" in MIX["per_arm"] and "L3@mB" in MIX["per_arm"])
ck("and the merge that did not happen is explained",
   any("aggregated separately" in w for w in MIX["warnings"]))
ck("the deterministic arm, which has no model, stays one arm",
   "A0p" in MIX["per_arm"] and not any(a.startswith("A0p@") for a in MIX["per_arm"]))
ck("both model arms are still margins over the same baseline",
   {r["baseline"] for r in MIX["comparisons"]} == {"A0p"})

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1)
shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASS")
