#!/usr/bin/env python3
"""A source whose fetch is complete on the cluster and whose intake failed is
intaken again, never fetched again (2026-10-28; zenodo_10762138,
2SeasonWeedDet8). End to end on the stream ticker (campaign.tick, the
executor, the real stream_remote verbs on the fake cluster of
tests/test_stream_ap_world.py), from the lab's records
(tests/fixtures/intake_via/live_2026-10-28.json).

What happened: the lab fetched the source (L16L, 35.8 GB, 9.2 h), synced it
(L16S), and its intake job 49959269 crashed in collect.normalize.read_via.
_failed made the source a candidate again; D20 then proposed a second lab
fetch (L16L), which would have reached 71.7 GB against the 50 GB per-source
limit and was filed for a person (ap-1793187344-83572915). The DATA lane
waited on it; a person's denial would have closed the source (DENIED_CLOSES).

  chain   the same sequence from the start: after the intake's failure the
          source is 'fetched' (one source_status event with the reason), L16I
          is proposed again under a new id and submitted, no fetch is proposed,
          filed or launched, nothing is filed for a person; a second and third
          failure close it after 3 (bounded, counted on the source; the third
          does not say the intake runs again); a staging
          the intake finds gone (not_fetched) makes it a candidate again, as
          before; a re-intake that succeeds makes it intaken.
  live    the lab's state as read on 2026-10-28 (the source a candidate with 1
          failure, the L16I step failure, the DATA lane holding the filed
          L16L, its pending approval in the approvals log): on the next tick
          the source is 'fetched' again, the approval is withdrawn by its
          requester (status withdrawn, never denied), the lane item is
          withdrawn with the reason and the lane freed, and L16I is proposed
          and submitted under a new id; the source is not closed, no card, no
          stall, no denial; D20 never proposes a fetch of it again. The
          re-intake then ends as it will on the real files: the collector
          closes the source (never_train, the config's entry for
          2SeasonWeedDet8: almost every image is a copy of cwd12 or of the
          ood22 exam), and DATA proposes the next source's fetch.
  units   DS.fetch_on_cluster (complete, synced after the fetch, the staging
          not lost), the D20 pre-check's refusal, a new fetch clears the sync.

Run:  python3 tests/test_stream_ap_intake_retry.py      (RETRY=chain,live runs only those)
"""
import copy
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import AP, S, World, check  # noqa: E402

DS = W.DS
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "intake_via" / "live_2026-10-28.json"
LIVE = json.loads(FIX.read_text())
SRC = "zenodo_10762138"
CAND = {"id": SRC, "provider": "zenodo", "licence": "CC-BY-NC-4.0", "target_classes": ["PricklySida", "Sicklepod"],
        "bytes": 35829729400, "images": 6664, "expected_target_boxes": 10848}
NEXT = {"id": "hf_next_source", "provider": "hf", "licence": "CC-BY-4.0", "target_classes": ["Waterhemp"],
        "bytes": 900000000, "images": 1200, "expected_target_boxes": 3000}
# the collector's line for the re-intake on the real files (collect.intake's never-fetch gate, the config's entry of
# 2SeasonWeedDet8, Zenodo 10762138)
CLOSE_LINE = ("[collect] intake refused: never_train: zenodo_10762138: the config lists zenodo:10762138 as never "
              "fetched (2SeasonWeedDet8: its 2021 part is derived from CottonWeedDet12, its 2022 part holds every "
              "ood22 exam image)")


def world(tag):
    w = World(tag, floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "zenodo": "fail"})          # zenodo placed on the lab, as on 2026-10-27
    w.candidates([dict(CAND)])
    return w


def src(w):
    return (((w.state() or {}).get("sources") or {}).get(SRC)) or {}


def intakes(w):
    return [s for s in w.submits if "run_inc_collect.sh" in " ".join(s["argv"]) and "intake" in s["argv"]
            and SRC in s["argv"]]


def lab_fetches(w):
    return [x for x in w.runner.launched if " fetch " in " %s " % " ".join(x["argv"]) and SRC in x["argv"]]


def fetch_props(w, n0=0):
    return [e for e in w.ledger()[n0:] if e.get("event") in ("proposed", "filed")
            and e.get("lever") in ("L16", "L16L", "L16R", "L16RL") and SRC in json.dumps(e)]


def write_state(w, st):
    S.StreamPaths(str(w.lab), w.domain).state(W.NAME).write_text(json.dumps(st))


def finish_fetch(w):
    """The lab fetch ends complete, as the lab's closing line said on 2026-10-27."""
    job = lab_fetches(w)[-1]["job"]
    p = S.StreamPaths(str(w.lab), w.domain).lab_inc / "intake" / "sources.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "a") as fh:
        for ev in ({"event": "fetch_started"}, {"event": "fetched", "status": "fetched", "bytes": 35829729400,
                                                "complete": True, "remaining": 0, "files": 4}):
            fh.write(json.dumps(dict(ev, source=SRC, ts=W.utc(w.clock()), format="collect-source-event/1",
                                     slurm_job_id=None), sort_keys=True) + "\n")
    w.runner.results[job] = {"ok": True, "rc": 0, "seconds": 33159.08,
                             "tail": '[collect] fetch: {"bytes": 35829729400, "complete": true, "remaining": 0, '
                                     '"status": "fetched"}'}


def crash_intake(w, refusal=None):
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal=refusal)


def no_person_items(w):
    return [a for a in w.approvals().values() if SRC in json.dumps(a.get("params") or {})
            and a.get("status") == "pending"]


def t_chain():
    print("chain: fetched on the lab, synced, the intake crashed -> intaken again, never fetched again")
    w = world("ir_chain")
    w.tick(3)
    check("set-up: the lab fetch (L16L) of %s is launched" % SRC, len(lab_fetches(w)) == 1,
          [x["argv"] for x in w.runner.launched])
    if not lab_fetches(w):
        return
    finish_fetch(w)
    w.tick(4)
    s = src(w)
    check("set-up: fetched, complete, synced at its own time (synced_utc), and the intake submitted",
          s.get("synced") and s.get("synced_utc") and s.get("partial") is False and len(intakes(w)) == 1,
          (s, [x["argv"] for x in w.submits]))
    if len(intakes(w)) != 1:
        return
    n0 = len(w.ledger())
    crash_intake(w)
    w.tick(3)
    s = src(w)
    led = w.ledger()[n0:]
    failed = [e for e in led if e.get("event") == "failed" and e.get("lever") == "L16I"]
    st_ev = [e for e in led if e.get("event") == "source_status" and e.get("source") == SRC
             and e.get("status") == "fetched"]
    check("the crash is a failure counted on the source (1), the source 'fetched' again with the reason",
          failed and s.get("failures") == 1 and st_ev and "never a new fetch" in str(st_ev[0].get("reasons")),
          (s, failed[-1:], st_ev))
    pro = [e for e in w.events("proposed") if e.get("lever") == "L16I"]
    check("L16I proposed again under a new id and submitted; no fetch proposed, filed or launched; nothing for a "
          "person", len(intakes(w)) == 2 and len({e.get("proposal_id") for e in pro}) == 2
          and not fetch_props(w, n0) and len(lab_fetches(w)) == 1 and not no_person_items(w),
          ([e.get("proposal_id") for e in pro], fetch_props(w, n0), no_person_items(w)))
    check("no card, no lane hold, no stall", not [c for c in (w.state() or {}).get("cards") or []
                                                  if SRC in json.dumps(c)]
          and not w.lane("DATA").get("hold") and not [e for e in led if e.get("event") == "lane_stalled"],
          ((w.state() or {}).get("cards"), w.lane("DATA")))
    # bounded: two more failures close it after 3, never a fetch
    led3 = []
    for k in (2, 3):
        nk = len(w.ledger())
        crash_intake(w)
        w.tick(3)
        led3 = w.ledger()[nk:]
    s = src(w)
    again3 = [e for e in led3 if e.get("event") == "source_status" and e.get("source") == SRC
              and e.get("status") == "fetched"]
    check("the third failure closes it and says only that: no 'fetched ... the intake runs again' event that tick",
          not again3 and any(e.get("event") == "source_closed" and e.get("source") == SRC for e in led3),
          [(e.get("event"), e.get("status"), (e.get("reasons") or [""])[0][:80]) for e in led3])
    check("a deterministic crash ends bounded: closed after 3 failed intakes ('3 failed attempts'), 3 intakes in "
          "all, never a new fetch", s.get("status") == "closed" and s.get("failures") == 3
          and "3 failed attempts" in str(s.get("closed_reason")) and len(intakes(w)) == 3
          and not fetch_props(w, n0) and len(lab_fetches(w)) == 1, (s, len(intakes(w))))

    print("chain: a staging the intake finds gone makes the source a candidate again (as before)")
    w2 = world("ir_gone")
    w2.tick(3)
    if lab_fetches(w2):
        finish_fetch(w2)
        w2.tick(4)
        crash_intake(w2, refusal="[collect] intake refused: not_fetched: no fetch record at /x/staging/%s/fetch.json: "
                                 "fetch the source first (lever L16)" % SRC)
        w2.tick(1)
        s = src(w2)
        check("not_fetched: staging_lost_utc set, the source a candidate (D20 may fetch it anew), one failure",
              s.get("status") in ("candidate", "fetching") and s.get("staging_lost_utc") and s.get("failures") == 1
              and not DS.fetch_on_cluster(s), s)

    print("chain: a re-intake that succeeds")
    w3 = world("ir_ok")
    w3.tick(3)
    if lab_fetches(w3):
        finish_fetch(w3)
        w3.tick(4)
        crash_intake(w3)
        w3.tick(3)
        w3.job_done("inc_stream_collect_intake_%s" % SRC)
        w3.tick(2)
        s = src(w3)
        check("the second intake completes: intaken, 1 failure on record, 2 intakes, 1 fetch",
              s.get("status") in ("intaken", "admitted") and s.get("failures") == 1 and len(intakes(w3)) == 2
              and len(lab_fetches(w3)) == 1, (s, len(intakes(w3))))


def plant_live(w):
    """The lab's records of 2026-10-28 in this world: the source, its L16I step failure and attempts, the DATA
    lane's filed L16L and its pending approval (the request row as the lab's approvals log holds it)."""
    st = w.state()
    st["sources"][SRC] = copy.deepcopy(LIVE["source"])
    st.setdefault("step_failures", {}).update(copy.deepcopy(LIVE["step_failures"]))
    st.setdefault("attempts", {}).update(copy.deepcopy(LIVE["attempts"]))
    lane = copy.deepcopy(LIVE["lane_DATA"])
    st["lanes"]["DATA"] = lane
    write_state(w, st)
    req = [r for r in LIVE["approval_rows"] if r.get("kind") == "request"][0]
    p = AP.path(w.domain, root=str(w.lab))
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(req, sort_keys=True) + "\n")
    return req


def t_live():
    print("live: the lab's state of 2026-10-28 (the filed second fetch) after this change is deployed")
    rows = {r["line"]: r["row"] for r in LIVE["ledger_rows"]}
    check("fixture: the lab's sequence (L16L, L16S, L16I executed and failed, D20's L16L filed and waiting)",
          [rows[n].get("event") for n in (2858, 2893, 2900, 2901, 2903, 3155, 3156, 3158)]
          == ["proposed", "proposed", "proposed", "executed", "failed", "proposed", "filed", "waiting"]
          and rows[3156]["approval_id"] == LIVE["lane_DATA"]["item"]["approval_id"]
          and LIVE["source"]["status"] == "candidate" and LIVE["source"]["fetch_remaining"] == 0, None)
    w = world("ir_live")
    w.tick(1)                                     # the first snapshot
    req = plant_live(w)
    aid = req["id"]
    pid_fetch = LIVE["lane_DATA"]["item"]["proposal"]["id"]
    pid_intake = LIVE["step_failures"]["L16I:02a8ed6176ea"]["proposal_id"]
    check("set-up: the planted approval is pending, requested by the autopilot, for the lane's L16L",
          (w.approvals().get(aid) or {}).get("status") == "pending"
          and (w.lane("DATA").get("item") or {}).get("approval_id") == aid, w.approvals().get(aid))
    n0 = len(w.ledger())
    w.tick(2)
    led = w.ledger()[n0:]
    s = src(w)
    a = w.approvals().get(aid) or {}
    check("the source is 'fetched' again with its reason (a source_status event by the stream)",
          s.get("status") in ("fetched", "intaken") and any(e.get("event") == "source_status" and e.get("source") == SRC
                                                            and e.get("status") == "fetched" for e in led), (s, led[:4]))
    check("the pending approval is withdrawn by its requester with the reason (status withdrawn, not denied)",
          a.get("status") == "withdrawn" and a.get("withdrawn_by") == W.AUTO and not a.get("decided_by")
          and "complete and on the cluster" in str(a.get("withdrawal_reason")), a)
    wd = [e for e in led if e.get("event") == "withdrawn" and e.get("proposal_id") == pid_fetch]
    check("the lane item is withdrawn (ledger 'withdrawn', approval_withdrawn true) and the lane freed of it",
          wd and wd[0].get("approval_withdrawn") is True and (w.lane("DATA").get("item") or {}).get("proposal", {})
          .get("id") != pid_fetch, (wd, w.lane("DATA").get("item")))
    pro = [e for e in led if e.get("event") == "proposed" and e.get("lever") == "L16I"]
    check("L16I is proposed under a new id (not the failed one) and submitted", pro and pro[0].get("proposal_id")
          != pid_intake and len(intakes(w)) == 1, ([e.get("proposal_id") for e in pro], [x["argv"] for x in w.submits]))
    check("no source closed, no denial, no card, no stall, the failures still 1",
          s.get("status") != "closed" and int(s.get("failures") or 0) == 1
          and not [e for e in led if e.get("event") in ("source_closed", "denied", "lane_stalled", "lane_held")]
          and not [c for c in (w.state() or {}).get("cards") or [] if SRC in json.dumps(c)],
          ([e.get("event") for e in led], (w.state() or {}).get("cards")))
    w.tick(3)
    check("D20 never proposes a fetch of it again; nothing pending for a person", not fetch_props(w, n0)
          and not lab_fetches(w) and not no_person_items(w), (fetch_props(w, n0), no_person_items(w)))
    # the re-intake's outcome on the real files: the collector closes the source (the config's never-fetch entry of
    # 2SeasonWeedDet8; a copy_candidate refusal is the same close), and DATA takes the next source
    w.candidates([dict(CAND), dict(NEXT)])
    n2 = len(w.ledger())
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal=CLOSE_LINE)
    w.tick(3)
    led = w.ledger()[n2:]
    s = src(w)
    closed = [e for e in led if e.get("event") == "source_closed" and e.get("source") == SRC]
    check("the re-intake's never_train refusal closes the source (by the collector, its code), not a failure",
          s.get("status") == "closed" and closed and closed[0].get("code") == "never_train"
          and int(s.get("failures") or 0) == 1 and "never fetched" in str(s.get("closed_reason")),
          (s, closed, [e.get("event") for e in led]))
    nxt = [e for e in led if e.get("event") in ("proposed", "filed") and NEXT["id"] in json.dumps(e)]
    check("DATA moves on: the next source's fetch is proposed; no lane hold, no stall, nothing for a person",
          nxt and not w.lane("DATA").get("hold") and not [e for e in led if e.get("event") == "lane_stalled"]
          and not no_person_items(w) and not fetch_props(w, n2),
          ([(e.get("event"), e.get("lever")) for e in led], w.lane("DATA")))
    # the platform's own path also reads a withdrawn approval of a filed item (a tick that withdrew it and did not
    # write its state): the item leaves its lane as withdrawn, never as a denial
    w2 = world("ir_live2")
    w2.tick(1)
    req2 = plant_live(w2)
    got = AP.withdraw(w2.domain, req2["id"], W.AUTO, "test: withdrawn before the state was written", w2.clock(),
                      root=str(w2.lab))
    st = w2.state()
    st["sources"][SRC]["status"] = "fetched"                # as the lost tick left it
    write_state(w2, st)
    n1 = len(w2.ledger())
    w2.tick(2)
    led2 = w2.ledger()[n1:]
    check("a filed item whose approval was withdrawn leaves the lane as withdrawn; the source is not closed",
          got.get("ok") and any(e.get("event") == "withdrawn" and e.get("approval_id") == req2["id"] for e in led2)
          and src(w2).get("status") != "closed" and not [e for e in led2 if e.get("event") == "denied"],
          [e.get("event") for e in led2])


def t_units():
    print("units: fetch_on_cluster, the pre-check, a new fetch clears the sync")
    base = {"fetched_utc": "2026-10-27T20:45:42Z", "partial": False, "fetch_remaining": 0, "placement": "lab",
            "synced": True}
    cases = [("the live record", dict(LIVE["source"]), True),
             ("synced, no synced_utc (recorded before the field)", dict(base), True),
             ("synced after the fetch", dict(base, synced_utc="2026-10-28T00:05:42Z"), True),
             ("a sync recorded before this fetch", dict(base, synced_utc="2026-10-20T00:00:00Z"), False),
             ("not synced", dict(base, synced=False), False),
             ("partial", dict(base, partial=True), False),
             ("files remaining", dict(base, fetch_remaining=3), False),
             ("completeness never recorded", {k: v for k, v in base.items() if k != "partial"}, False),
             ("never fetched", {k: v for k, v in base.items() if k != "fetched_utc"}, False),
             ("fetched on the cluster", dict(base, placement="cluster", synced=None), True),
             ("its staging found gone", dict(base, staging_lost_utc="2026-10-28T03:05:42Z"), False)]
    for what, s, want in cases:
        check("fetch_on_cluster: %s -> %s" % (what, want), DS.fetch_on_cluster(s) is want, s)

    class V(object):
        dom = {}

        def c(self, ptr):
            return None
    ref, _rev = DS.precheck(V(), {"id": SRC, "status": "candidate", "intake_retry": True, "licence": "CC BY 4.0",
                                  "target_classes": ["Sicklepod"]})
    check("D20's pre-check refuses to fetch a source to be intaken again", any("never a new fetch" in r for r in ref),
          ref)
    ref, _rev = DS.precheck(V(), {"id": SRC, "status": "candidate", "intake_retry": False, "licence": "CC BY 4.0",
                                  "target_classes": ["Sicklepod"]})
    check("  and not one without", not ref, ref)
    import inspect
    src_on = inspect.getsource(S.StreamCampaign._on_started) if hasattr(S, "StreamCampaign") else ""
    if not src_on:
        cls = next(v for v in vars(S).values() if isinstance(v, type) and hasattr(v, "_on_started"))
        src_on = inspect.getsource(cls._on_started)
    check("a fetch's start clears the sync of the fetch before it (a new fetch record is synced again)",
          's.pop("synced", None)' in src_on and 's.pop("synced_utc", None)' in src_on)


def main():
    want = [x for x in os.environ.get("RETRY", "").split(",") if x]
    try:
        for fn in (t_units, t_chain, t_live):
            if want and fn.__name__[2:] not in want:
                continue
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-2500:]))
    finally:
        shutil.rmtree(str(W.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(W.FAILURES))
    for x in W.FAILURES:
        print("  FAILED: %s" % x)
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
