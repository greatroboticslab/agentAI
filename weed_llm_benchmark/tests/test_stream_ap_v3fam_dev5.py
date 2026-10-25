#!/usr/bin/env python3
"""V3F deviation 5 (2026-10-24; docs/prereg/V3F_dev5_2026-10-24.md) end to end on the stream ticker (campaign.tick,
the executor, the real stream_remote verbs on a fake cluster: tests/test_stream_ap_world.py).

- the live state of 2026-10-24 with deviation 5 deployed: the six arms D8, D4F, D4N, D4R15, D1 and D5B recorded
  failed twice (their score jobs refused by the platform's stale records on 2026-10-21), D6's smoke 'not run
  (walltime)' (the record of job 49542653, byte for byte), the continuation of 2026-10-22 decided, the failure
  records and identity_probe.json (2026-10-21) on the cluster: the six arms and the verdict reopened once, then the
  six scores in V3F.7's order (D8 first), then L23L once with no --ticker-failed, writing the second continuation
  (capacity/v3fam_v1_cont2.json); the earlier verdicts unchanged; one card for it; no family item failed; D6 stays
  'not run';
- a deploy that changes the code the reproduction records pin (deviation 4's kind of change) and D8's score job
  refused by them ("the code changed since dev_identity.json ran", in its provenance only): D8 is not failed (no
  fails, nothing declined, its GPU-h kept), held with one card naming the remedy; no other score is proposed while
  the records are stale; once they are current D8 is released and proposed again under a new id, and the family
  runs to its verdict with no --ticker-failed and no item failed;
- D28 on the live state of 2026-10-24 (tests/fixtures/v3fam_dev5/d28_live_2026-10-24.json: the sources the stream
  quarantines while the ticker holds them 'quarantine pending', lane STOP's stall record, the declined L24 ids): the
  next ticks resync them, propose, run and decline no L24, and end the stall (lane_stall_ended).

Run:  python3 tests/test_stream_ap_v3fam_dev5.py        (DEV5_WORLD=live,code,d28 runs only those)
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_units as U  # noqa: E402  (the world and the family helpers)
import test_stream_ap_world as W  # noqa: E402

from test_stream_ap_world import NAME, S, SR, check  # noqa: E402

FAILURES = W.FAILURES
TESTS = pathlib.Path(__file__).resolve().parent
FIX = TESTS / "fixtures" / "v3fam_dev5"
SIX = ("D8", "D4F", "D4N", "D4R15", "D1", "D5B")
DEV4_V3FAM = "c5e923dcad8f576cdb9761a0433d5c85241def4a87a0a0081063aa57a50e5e78"
REFUSAL = ('  File "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/weed_optimizer_framework/tools/inc2/'
           'v3fam.py", line 1289, in require_equivalence\n    raise V3famRefused("the code changed since %s ran (%s): '
           'it is moved aside and rerun, as a deviation"\nweed_optimizer_framework.tools.inc2.v3fam.V3famRefused: the '
           'code changed since dev_identity.json ran (v3fam): it is moved aside and rerun, as a deviation')


def prov(w, name, job_id, refusal):
    """The family job's provenance (run_inc2_build.sh's record): one attempt, ended build_failed with its refusal."""
    p = w.inc / "_campaign" / "provenance" / ("%s.json" % name)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(p.read_text()) if p.exists() else {"format": "inc_autopilot.provenance/1", "attempts": []}
    rec["attempts"].append({"job_id": str(job_id), "status": "build_failed", "build_rc": 1, "refusal": refusal,
                            "started_utc": W.utc(w.t[0]), "finished_utc": W.utc(w.t[0])})
    p.write_text(json.dumps(rec))


def queued(w, name):
    return next((j for j in w.squeue if j["name"] == name), None)


def state_write(w, st):
    S.StreamPaths(str(w.lab), "weed").state(NAME).write_text(json.dumps(st))


def items(w):
    return (w.state().get("v3fam") or {}).get("items") or {}


def run_to_d8(w, smoke_record=None):
    """Tick the family until D8's score job is queued (every earlier item done; D6's smoke written as given)."""
    d8 = U.V3FAM_SCORE_JOB % "d8"
    for _ in range(300):
        w.tick(1)
        if queued(w, "inc_detr_v3fam_smoke") is not None:
            w.job_done("inc_detr_v3fam_smoke")
            p = w.inc / "v3fam" / "v1" / "d6" / "smoke" / "smoke.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            if smoke_record is not None:
                p.write_bytes(smoke_record)
            else:
                w.v3fam_detr()
        U._fam_cluster(w, skip=("inc_detr_v3fam_smoke", d8))
        if queued(w, d8) is not None:
            return queued(w, d8)
    return None


def t_live():
    print("the live state of 2026-10-24 with deviation 5 deployed")
    w = U._v3fam_world("dev5_live")
    d8 = U.V3FAM_SCORE_JOB % "d8"
    j = run_to_d8(w, smoke_record=(FIX / "d6" / "smoke.json").read_bytes())
    # what the ticker of 2026-10-21/22 made of the refusals (the code before this deviation): the six arms failed
    # twice and declined, the continuation of 2026-10-22 decided; on the cluster the verdicts of 2026-10-13 and
    # 2026-10-22, the six failure records and identity_probe.json, each byte for byte
    cap = w.inc / "capacity"
    cap.mkdir(parents=True, exist_ok=True)
    for n in ("v3fam_v1.json", "v3fam_v1_external.json", "v3fam_v1_cont.json", "v3fam_v1_cont_external.json"):
        shutil.copyfile(str(FIX / "capacity" / n), str(cap / n))
    for p in (FIX / "failed").glob("*.json"):
        dst = w.inc / "v3fam" / "v1" / "failed" / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(p), str(dst))
    shutil.copyfile(str(FIX / "equivalence" / "identity_probe.json"),
                    str(w.inc / "v3fam" / "v1" / "equivalence" / "identity_probe.json"))
    w.job_done(d8, state="FAILED", elapsed=14.0)
    w.tick(1)
    st = json.loads(S.StreamPaths(str(w.lab), "weed").state(NAME).read_text())
    its = st["v3fam"]["items"]
    for i, a in enumerate(SIX):
        its["score:%s" % a] = {"state": "failed", "fails": 2, "lever": "L23J", "proposal": "dead%02d" % i,
                               "started_utc": "2026-10-21T2%d:00:33Z" % i, "spend": "settled", "settled": 0.08,
                               "prior_gpuh": 0.07, "estimate": 1.0}
        st["declined"] = list(st.get("declined") or []) + ["dead%02d" % i]
    its["verdict"] = {"state": "done", "lever": "L23L", "proposal": "54e3fd24ffbf024da91c6201a33d6b6d",
                      "started_utc": "2026-10-22T00:20:33Z", "spend": "settled", "settled": 0.5778, "prior_gpuh": 0.66}
    st["v3fam"]["verdict_card"] = "2026-10-22T01:30:34Z"
    state_write(w, st)
    before = {n: (cap / n).read_bytes() for n in ("v3fam_v1.json", "v3fam_v1_cont.json")}
    w.tick(2)
    quiet = [e for e in w.events("proposed") if e.get("lever") in S.V3FAM_LEVERS]
    n0 = len(w.events("proposed"))
    # the deploy ships deviation 5's file with its code (one commit): the next tick is the platform's
    w.v3fam_deploy_dev5()
    U._fam_run(w, ticks=3, skip=(d8,))
    its2 = items(w)
    first = [(e.get("lever"), U._fam_arg(e)) for e in w.events("proposed")[n0:] if e.get("lever") in S.V3FAM_LEVERS]
    rcards = [c for c in w.state().get("cards") or [] if "deviation 5" in c["title"] and "proposed again" in c["title"]]
    check("before the deploy nothing more is proposed (the family idle, as since 2026-10-22); deployed: the six arms "
          "and the verdict reopened once (one card), their spend kept in prior_gpuh; L23J D8 first",
          j is not None and len([e for e in quiet if e.get("lever") == "L23J" and U._fam_arg(e) == "D8"]) == 1
          and all(its2["score:%s" % a].get("reopened") == "deviation 5 (2026-10-24)" for a in SIX)
          and abs(float(its2["score:D8"].get("prior_gpuh") or 0) - 0.15) < 1e-9 and len(rcards) == 1
          and first[:1] == [("L23J", "D8")], (first, [(a, its2["score:%s" % a].get("reopened")) for a in SIX]))
    done = U._fam_run(w, ticks=300, until=lambda: items(w).get("verdict", {}).get("state") == "done"
                      and (cap / "v3fam_v1_cont2.json").exists())
    pro = [(e.get("lever"), U._fam_arg(e)) for e in w.events("proposed")[n0:] if e.get("lever") in S.V3FAM_LEVERS]
    l23j = [a for lv, a in pro if lv == "L23J"]
    l23l = [e for e in w.events("proposed")[n0:] if e.get("lever") == "L23L"]
    w.tick(3)
    vc = [c for c in w.state().get("cards") or [] if "second continuation verdict is decided" in c["title"]]
    st_end = items(w)
    check("then the six arms' scores in V3F.7's order and L23L once with no --ticker-failed: the second continuation "
          "written, the verdicts of 2026-10-13 and 2026-10-22 unchanged byte for byte, one card for it naming both; no "
          "family item failed; D6 stays not run (walltime), never proposed",
          done and l23j == list(SIX) and len(l23l) == 1 and "--ticker-failed" not in (l23l[0].get("argv") or [])
          and all((cap / n).read_bytes() == b for n, b in before.items()) and len(vc) == 1
          and "2026-10-22" in vc[0]["detail"] and not [k for k, v in st_end.items() if v.get("state") == "failed"]
          and ("L23K", "0,1,2") not in pro and "score:D6" not in st_end, (done, pro, len(vc)))


def t_code():
    print("a deploy that changes the code the records pin; D8's score job refused by them")
    w = U._v3fam_world("dev5_code")
    d8 = U.V3FAM_SCORE_JOB % "d8"
    j = run_to_d8(w)
    first = [e.get("proposal_id") for e in U._fam_pro(w, "L23J") if U._fam_arg(e) == "D8"]
    old = dict(w.v3fam_code_now(), v3fam=DEV4_V3FAM)
    w.v3fam_equiv_records(code=old)                   # the records now pin other code than the deployed one
    prov(w, "v3fam_score_d8", j["id"], REFUSAL)
    w.job_done(d8, state="FAILED", elapsed=14.0)
    U._fam_run(w, ticks=4, skip=(d8,))
    rec = items(w).get("score:D8") or {}
    pro = [(e.get("lever"), U._fam_arg(e)) for e in U._fam_pro(w)]
    after = pro[pro.index(("L23J", "D8")) + 1:]
    cards = [c for c in w.state().get("cards") or []]
    titles = [c["title"] for c in cards]
    check("D8's refusal is no failure: no fails, not declined, its GPU-h (14 s from sacct) kept in prior_gpuh, its id "
          "retired, held; one card naming the remedy and one for the stale records; no score proposed after it",
          j is not None and len(first) == 1 and not rec.get("fails") and rec.get("state") == "held (platform)"
          and rec.get("platform_refusals") == 1 and first[0] not in (w.state().get("declined") or [])
          and first[0] in (w.state().get("failed_ids") or [])
          and abs(float(rec.get("prior_gpuh") or 0) - 14.0 / 3600) < 1e-4
          and not [x for x in after if x[0] == "L23J"]
          and any("held, not the arm's failure" in t for t in titles)
          and any("family scores wait" in t for t in titles)
          and not [e for e in w.events("failed") if e.get("lever") == "L23J"], (rec, after, titles[-3:]))
    w.v3fam_equiv_records()                           # current again (a recorded deviation restored the code)
    done = U._fam_run(w, ticks=300, until=lambda: items(w).get("verdict", {}).get("state") == "done")
    d8_ids = [e.get("proposal_id") for e in U._fam_pro(w, "L23J") if U._fam_arg(e) == "D8"]
    l23l = U._fam_pro(w, "L23L")
    check("once the records are current: D8 released and proposed again under a new id, then the rest of the family, "
          "and L23L once with no --ticker-failed; no family item failed",
          done and len(set(d8_ids)) == 2 and len(l23l) == 1 and "--ticker-failed" not in (l23l[0].get("argv") or [])
          and not [k for k, v in items(w).items() if v.get("state") == "failed"]
          and items(w).get("score:D8", {}).get("state") == "done"
          and [e for e in w.events("v3fam_released")], (done, d8_ids))


def t_d28():
    print("D28: the live state of 2026-10-24, lane STOP stalled on quarantines already in effect")
    live = json.loads((FIX / "d28_live_2026-10-24.json").read_text())
    w = W.World("dev5_d28")
    w.ready_r0()
    w.tick(2)
    for s in live["d28_leaks"]:
        rec = live["sources"][s]
        w.intake(rec["batch"], s, images=600, eval_share=0.06, reasons={"near_eval_v2": 20, "near_eval_variant": 14})
    qrec = {}
    for s, q in sorted(live["quarantined_sources"].items(), key=lambda kv: kv[1]["seq"]):
        if s in live["sources"]:
            ev = w.stream_event("quarantine", scope="source", source=s, cite=q["cite"])
            qrec[s] = {"cite": q["cite"], "seq": ev["seq"], "utc": q["utc"]}
        else:
            qrec[s] = dict(q)
    w.queue(0, extra={"quarantined_sources": qrec})
    w.tick(2)                                        # D28 fires on the three leaks from here on, as live
    st = json.loads(S.StreamPaths(str(w.lab), "weed").state(NAME).read_text())
    for s, rec in live["sources"].items():
        st.setdefault("sources", {})[s] = json.loads(json.dumps(rec))       # verbatim from the lab's state.json
    st["stalls"] = json.loads(json.dumps(live["stalls"]))
    st["declined"] = list(st.get("declined") or []) + live["declined_l24"]
    state_write(w, st)
    n0 = len(w.ledger())
    fired = []
    for _ in range(3):
        w.tick(1)
        dg = json.loads((w.lab / "results" / "framework" / "_brain" / "weed" / "inc" / "campaigns" / NAME /
                         "diagnoses.json").read_text())
        fired.append(any(d.get("id") == "D28" and d.get("fired") for d in dg.get("diagnoses") or []))
    led = w.ledger()[n0:]
    st2 = w.state()
    l24 = [e for e in led if e.get("lever") == "L24" and e.get("event") in ("proposed", "executed", "not_taken",
                                                                             "recovered")]
    check("D28 still fires on the three leaks on every tick (its finding stands), the sources resynced to quarantined "
          "from the stream's ledger (one event each), no L24 proposed, run or declined again, lane STOP's stall of "
          "2026-10-07 ended (lane_stall_ended)", all(fired)
          and all(st2["sources"][s]["status"] == "quarantined" for s in live["sources"])
          and len([e for e in led if e.get("event") == "source_quarantine_resynced"]) == len(live["sources"])
          and not l24 and "STOP" not in (st2.get("stalls") or {})
          and any(e.get("event") == "lane_stall_ended" and e.get("lane") == "STOP" for e in led),
          ([(e.get("event"), e.get("lever")) for e in led][-10:], fired, l24,
           {s: st2["sources"][s].get("status") for s in live["sources"]}))


def main():
    want = [x for x in os.environ.get("DEV5_WORLD", "").split(",") if x]
    try:
        for fn in (t_live, t_code, t_d28):
            if want and fn.__name__[2:] not in want:
                continue
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-2000:]))
    finally:
        shutil.rmtree(str(W.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
