#!/usr/bin/env python3
"""Runs a lane lost, declined proposals and stalled lanes (2026-10-03..05).

Live, 2026-10-03 07:44Z: the DATA lane executed an L16S candidate sync of
rf_a-programlama__ag-programlama (lab job
sync_rf_a_programlama__ag_programlama_c2ad2f813a, ok) and the dashboard
restarted before the tick wrote its state. The state came back without the
running item. At 15:36Z D20 proposed the same sync again (a deterministic id),
the executor refused it as already run, and the 'recovered' branch declined it
on the belief that its effect shows in the snapshots; an L16S's effect
(candidate_synced) lives only in the ticker's state. From then on the declined
id was skipped every tick with no event and no card, and from 2026-10-04 09:27Z
the DATA lane stood idle. The source's estimate was 0 target boxes ('Trypophobia').

Pinned here, in the stream world (real ticker, executor and lab hooks; a fake
lab runner whose sync jobs succeed):
  * the 10-03 sequence replayed (executed record and ok lab result, the state
    rolled back): the next tick takes the run back from the execution log
    before anything is proposed, its lab result is read, candidate_synced is
    set, and the next proposal is the cluster fetch (L16) under a new id; the
    sync is never launched twice;
  * the same, with the run older than the state's last write (no lost-run
    scan): the re-proposal refused as already run follows the recorded lab
    job instead of being declined;
  * the live condition (the id already declined): it is taken back on the
    next tick, not skipped;
  * a run a lane already followed to its end is never taken back: a second
    discovery (L15) with the same classes runs under a new id (its result is
    never the first one's read again), and on a state written before L15 got
    a new attempt when it finished, the repeat refused as already run (or
    declined so) is proposed again under a new id; a run taken back once is
    never taken back twice;
  * the first tick on a state written before ended_ids existed (a deploy)
    takes back no run the lanes followed (an L24 quarantine is done once,
    whatever the executor's clock says), and still takes back one they lost;
  * a declined proposal with no run to take back writes exactly one
    not_taken, and after watchdog stall_ticks ticks a card names the lane;
  * D20 never proposes a candidate whose estimate is 0 target boxes: with only
    such candidates it takes the discovery path (L15), and D29 counts none of
    them open; a candidate whose zero comes from pending class names gets L26
    (never a fetch); a positive candidate is ranked as before.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import DAY, LS, NAME, S, X, World, check  # noqa: E402

SRC = "rf_ag_prog"
CAND = {"id": SRC, "provider": "roboflow", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
        "bytes": 1e9, "images": 300, "expected_target_boxes": 600}


def world(tag, cands=None):
    w = World(tag, floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(c) for c in (cands or [CAND])])
    return w


def paths(w):
    return S.StreamPaths(str(w.lab), w.domain)


def syncs(w):
    return [x for x in w.runner.launched if "lab-sync" in x["argv"] and "--file" in x["argv"]]


def lost_tick(w, drop_record=False):
    """Tick until the DATA lane executes the candidate sync, then write back the
    state from before that tick (the restart that lost it). With drop_record
    the execution log loses that tick's lines too (a run that never happened).
    Returns the sync's proposal id and its execution record."""
    p = paths(w)
    log = w.xctx.exec_log
    for _ in range(8):
        before = p.state(NAME).read_text() if p.state(NAME).exists() else None
        size = log.stat().st_size if log.exists() else 0
        w.tick(1)
        if syncs(w):
            break
    else:
        raise AssertionError("no candidate sync was launched")
    rec = [r for r in w.executions() if r.get("lever") == "L16S" and r.get("status") == "executed"][-1]
    p.state(NAME).write_text(before)
    if drop_record:
        with open(str(log), "r+b") as fh:
            fh.truncate(size)
    return rec["proposal_id"], rec


def edit_state(w, fn):
    p = paths(w).state(NAME)
    st = json.loads(p.read_text())
    fn(st)
    p.write_text(json.dumps(st))


def src_state(w):
    return ((w.state() or {}).get("sources") or {}).get(SRC) or {}


def fetch_after(w, pid):
    return [e for e in w.events("proposed") if e.get("lever") == "L16" and e.get("proposal_id") != pid]


def t_replay_lost_tick():
    print("the 10-03 sequence: an executed L16S lost with the tick's state is taken back from the execution log")
    w = world("lost_tick")
    pid, rec = lost_tick(w)
    check("the rolled-back state holds no candidate sync (the restart lost it)",
          not (w.lane("DATA").get("item") or {}) and not src_state(w).get("candidate_synced"), w.lane("DATA"))
    n_ev = len(w.ledger())
    w.tick(1)
    new = w.ledger()[n_ev:]
    ad = [e for e in new if e.get("event") == "adopted_run" and e.get("proposal_id") == pid]
    it = w.lane("DATA").get("item") or {}
    check("the next tick takes the run back first: adopted_run with its lab job, the item running, nothing "
          "proposed or refused as already run", ad and ad[0].get("lab_job") == rec["remote"]["payload"]["lab_job"]
          and it.get("status") == "running" and (it.get("proposal") or {}).get("id") == pid
          and not [e for e in new if e.get("event") in ("recovered", "proposed")], [e.get("event") for e in new])
    w.tick(3)
    st = w.state()
    check("its lab result is read: item_done, candidate_synced set, never declined",
          any(e.get("proposal_id") == pid for e in w.events("item_done")) and src_state(w).get("candidate_synced")
          and pid not in (st.get("declined") or []), src_state(w))
    check("  the sync was launched once, never again", len(syncs(w)) == 1, [x["job"] for x in syncs(w)])
    check("  the next proposal is the cluster fetch (L16) under a new id", fetch_after(w, pid),
          [(e.get("lever"), e.get("proposal_id")) for e in w.events("proposed")])
    check("  the run is adopted once (adopted_runs), and no other run of the campaign was adopted",
          [e.get("proposal_id") for e in w.events("adopted_run")] == [pid], w.events("adopted_run"))


def t_recovered():
    print("a lab item refused as already run follows its recorded lab job (the run older than the last state write)")
    w = world("recovered_lab")
    pid, _rec = lost_tick(w)
    edit_state(w, lambda st: st.update(updated_utc=W.utc(w.t[0] + 60)))
    w.tick(1)
    rec_ev = [e for e in w.events("recovered") if e.get("lever") == "L16S"]
    ad = [e for e in w.events("adopted_run") if e.get("proposal_id") == pid]
    check("the re-proposal is refused as already run, and the lane takes the run back by its lab job (not declined)",
          rec_ev and ad and pid not in (w.state().get("declined") or []) and "already ran" in str(ad[0].get("reasons")),
          (rec_ev, ad))
    w.tick(3)
    check("  its result is read: candidate_synced, then L16 under a new id; launched once",
          src_state(w).get("candidate_synced") and fetch_after(w, pid) and len(syncs(w)) == 1,
          (src_state(w), len(syncs(w))))
    # no lab job recorded: declined, with a card
    w2 = world("recovered_nojob")
    pid2, _ = lost_tick(w2, drop_record=True)
    run = W.S.StreamRun(NAME, w2.config(), W.C.Paths(str(w2.lab)), W.C._SshBudget(None), w2.clock, w2.log, w2.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w2.state(), LS.load_domain("weed"), LS.load_thresholds()
    it = {"proposal": {"id": pid2, "policy_action": "inc_stream_sync", "follow": "lab", "params": {"source": SRC},
                       "trigger": ["D20"]}, "lever": "L16S", "status": "proposed"}
    run.st["lanes"]["DATA"]["item"] = it
    run._on_result("DATA", it, {"status": "refused", "reasons": ["proposal %s already ran (started at x); a proposal "
                                                                 "runs once" % pid2]})
    check("  with no lab job recorded for the run, it is declined with a card (never silently)",
          pid2 in run.st["declined"] and run.st["lanes"]["DATA"]["item"] is None
          and any(c.get("kind") == "platform" and "already ran" in c["title"] for c in run.st["cards"]),
          run.st["cards"][-1:])
    # a run taken back once (adopted_runs) and not followed since: never taken back a second time
    w3 = world("recovered_twice")
    pid3, rec3 = lost_tick(w3)
    run = W.S.StreamRun(NAME, w3.config(), W.C.Paths(str(w3.lab)), W.C._SshBudget(None), w3.clock, w3.log, w3.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w3.state(), LS.load_domain("weed"), LS.load_thresholds()
    run.st["adopted_runs"] = [rec3["run_id"]]
    it = {"proposal": {"id": pid3, "policy_action": "inc_stream_sync", "follow": "lab", "params": {"source": SRC},
                       "trigger": ["D20"]}, "lever": "L16S", "status": "proposed"}
    run.st["lanes"]["DATA"]["item"] = it
    run._on_result("DATA", it, {"status": "refused", "reasons": ["proposal %s already ran (executed at x); a proposal "
                                                                 "runs once" % pid3]})
    check("  a run taken back once already is not taken back again: declined with a card",
          pid3 in run.st["declined"] and run.st["lanes"]["DATA"]["item"] is None
          and run.st["adopted_runs"] == [rec3["run_id"]]
          and any("taken back once already" in c.get("detail", "") for c in run.st["cards"]), run.st["cards"][-1:])


def t_live_declined():
    print("the live condition: the run's id already declined -> taken back on the next tick, not skipped")
    w = world("live_declined")
    pid, _rec = lost_tick(w)
    edit_state(w, lambda st: st.update(updated_utc=W.utc(w.t[0] + 60), declined=list(st.get("declined") or []) + [pid]))
    n_ev = len(w.ledger())
    w.tick(1)
    new = w.ledger()[n_ev:]
    check("the next tick adopts the declined proposal's run (adopted_run), no silent skip",
          any(e.get("event") == "adopted_run" and e.get("proposal_id") == pid for e in new)
          and (w.lane("DATA").get("item") or {}).get("status") == "running", [e.get("event") for e in new])
    w.tick(3)
    check("  candidate_synced is set and the cluster fetch (L16) is proposed under a new id; the sync ran once",
          src_state(w).get("candidate_synced") and fetch_after(w, pid) and len(syncs(w)) == 1, src_state(w))


def t_declined_watchdog():
    print("a declined proposal with no run to take back: one not_taken, then the watchdog's card")
    n = int(LS.t(LS.load_thresholds(), "watchdog", "stall_ticks"))
    check("the watchdog's tick count is declared (stream_thresholds.json watchdog stall_ticks, with its reason)",
          n >= 2 and LS.load_thresholds()["watchdog"]["stall_ticks"].get("why"), n)
    w = world("declined_stall")
    pid, _rec = lost_tick(w, drop_record=True)
    edit_state(w, lambda st: st.update(declined=list(st.get("declined") or []) + [pid]))
    w.tick(n - 1)
    nt = [e for e in w.events("not_taken") if e.get("proposal_id") == pid]
    stall = [c for c in (w.state().get("cards") or []) if c.get("kind") == "stall"]
    check("the declined skip writes exactly one not_taken over %d ticks, naming the proposal" % (n - 1),
          len(nt) == 1 and "declined" in str(nt[0].get("reasons")) and not w.lane("DATA").get("item"), nt)
    check("  no card before %d ticks" % n, not stall, stall)
    w.tick(1)
    stall = [c for c in (w.state().get("cards") or []) if c.get("kind") == "stall"]
    check("after %d ticks a card names the stalled lane, the diagnosis, the lever and the reason" % n,
          len(stall) == 1 and "Lane DATA stalled" in stall[0]["title"] and "D20" in stall[0]["title"]
          and pid in stall[0]["detail"] and w.events("lane_stalled"), stall)
    w.tick(3)
    check("  once per stall (no second card), and still one not_taken",
          len([c for c in w.state().get("cards") or [] if c.get("kind") == "stall"]) == 1
          and len([e for e in w.events("not_taken") if e.get("proposal_id") == pid]) == 1)
    edit_state(w, lambda st: st.update(declined=[x for x in st.get("declined") or [] if x != pid]))
    w.tick(1)
    check("  a lane that takes an item again ends the stall (lane_stall_ended)",
          w.events("lane_stall_ended") and "DATA" not in (w.state().get("stalls") or {}),
          (w.state().get("stalls"), w.lane("DATA")))


def _d(w):
    return {d["id"]: d for d in json.loads(paths(w).diagnoses(NAME).read_text())["diagnoses"]}


def d20_said(w):
    """Every summary D20 fired with (the campaign ledger's diagnosed events)."""
    return [f.get("summary") for e in w.events("diagnosed") for f in e.get("fired") or [] if f.get("id") == "D20"]


def t_zero_estimate():
    print("D20 never proposes a candidate whose estimate is 0 target boxes")
    zero = {"id": "rf_zero", "provider": "roboflow", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
            "bytes": 1e8, "images": 288, "expected_target_boxes": 0.0}
    w = world("zero_only", [zero])
    w.tick(3)
    on_zero = [e for e in w.events("proposed") if (e.get("argv") or []) and "rf_zero" in " ".join(e["argv"])] + \
        [x for x in w.runner.launched if "rf_zero" in x["argv"]]
    check("only zero-estimate candidates: no L16 (nor its L16S) on them; the discovery path instead (L15, never run)",
          not on_zero and any("no open candidate (1 with a zero estimate)" in str(x) and "L15" in str(x)
                              for x in d20_said(w))
          and any(e.get("lever") == "L15" for e in w.events("proposed")), (d20_said(w), on_zero))
    # D29: a recent empty discovery and only zero-estimate candidates -> collection exhausted (WAIT_DATA)
    w2 = world("zero_d29", [zero])
    w2.tick(1)
    edit_state(w2, lambda st: st.update(discover={"last_utc": W.utc(w2.t[0] - DAY), "found_new": 0, "empty_runs": 1,
                                                  "runs": 2}))
    w2.tick(2)
    d29 = _d(w2)["D29"]
    check("  D29 counts no zero-estimate candidate open: a recent empty discovery -> WAIT_DATA",
          d29["fired"] and w2.lane("DATA").get("phase") == "WAIT_DATA", (d29["summary"], w2.lane("DATA")))
    # names pending: the zero is unknown -> L26, never a fetch
    pend = dict(zero, id="rf_pending", target_classes=[], names_unresolved=True)
    w3 = world("zero_names", [pend])
    w3.tick(3)
    names = [x for x in w3.runner.launched if "names" in x["argv"] and "rf_pending" in x["argv"]]
    fetch = [x for x in w3.runner.launched if "rf_pending" in x["argv"] and ("lab-sync" in x["argv"]
                                                                              or "fetch" in x["argv"])]
    check("  a candidate whose zero comes from pending class names gets L26 (names), never a fetch",
          names and not fetch, [x["argv"][-6:] for x in w3.runner.launched])
    w3.runner.finish(ok=True)
    w3.tick(3)
    fetch = [x for x in w3.runner.launched if "rf_pending" in x["argv"] and ("lab-sync" in x["argv"]
                                                                             or "fetch" in x["argv"])]
    check("  once its names are resolved its estimate is still 0: not fetched (the next L15 estimates it again)",
          (w3.state()["sources"].get("rf_pending") or {}).get("names_resolved") and not fetch
          and "zero estimate" in str(d20_said(w3)[-1]), (d20_said(w3), fetch))
    # a positive candidate ranks as before, the zero one never
    w4 = world("zero_and_positive", [zero, dict(CAND)])
    w4.tick(2)
    d20 = _d(w4)["D20"]
    check("  with a positive candidate beside it: L16 on the positive one, the zero one not ranked",
          (d20["detail"].get("propose") or {}).get("source") == SRC
          and [r["source"] for r in d20["detail"]["ranked"]] == [SRC]
          and d20["detail"]["zero_estimate"]["sources"] == ["rf_zero"], d20["detail"].get("ranked"))


def discoveries(w):
    return [x for x in w.runner.launched if "plan" in x["argv"]]


def l15_world(tag):
    """No candidate at all: D20 takes the discovery path (L15); the first
    discovery runs, finds nothing, and the DATA lane waits (D29, WAIT_DATA).
    Returns the world and the first L15's proposal id."""
    w = World(tag, floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([])
    for _ in range(6):
        w.tick(1)
        if discoveries(w):
            break
    pid = [e for e in w.events("proposed") if e.get("lever") == "L15"][-1]["proposal_id"]
    w.runner.finish(ok=True)
    w.tick(2)
    return w, pid


def l15_ids(w):
    return [e.get("proposal_id") for e in w.events("proposed") if e.get("lever") == "L15"]


def t_l15_again():
    print("a second discovery (L15) with the same classes runs under a new id; the first run is never read again")
    w, pid = l15_world("l15_again")
    d = w.state()["discover"]
    check("the first discovery ran once and found nothing: runs 1, DATA waits (WAIT_DATA)",
          len(discoveries(w)) == 1 and d.get("runs") == 1 and w.lane("DATA").get("phase") == "WAIT_DATA",
          (d, w.lane("DATA")))
    w.advance(8 * DAY)
    w.tick(2)
    ids = l15_ids(w)
    st = w.state()
    check("after the wait D20 calls for L15 with the same classes: proposed under a new id and launched again; no "
          "run taken back, nothing refused as already run",
          len(ids) == 2 and ids[1] != pid and len(discoveries(w)) == 2 and not w.events("adopted_run")
          and not w.events("recovered") and (st["discover"] or {}).get("runs") == 1,
          (ids, len(discoveries(w)), st["discover"]))
    w.runner.finish(ok=True)
    w.tick(2)
    d = w.state()["discover"]
    check("  its own result ends it: runs 2, empty_runs 2, one card for each wait at most",
          d.get("runs") == 2 and d.get("empty_runs") == 2 and len(discoveries(w)) == 2, d)


def _repeat_old_state(w, pid, declined=False):
    """The state as the code before this fix left it after an L15: no new
    attempt for its step (the repeat gets the same id), and with `declined`
    the repeat already refused as run and declined (main's 'recovered')."""
    def fn(st):
        st["attempts"] = {k: v for k, v in (st.get("attempts") or {}).items() if not k.startswith("L15:")}
        if declined:
            st["declined"] = list(st.get("declined") or []) + [pid]
    edit_state(w, fn)


def t_followed_not_taken_back():
    print("a repeat of a run a lane followed is proposed again under a new id, never taken back")
    for declined in (False, True):
        tag = "declined" if declined else "refused"
        w, pid = l15_world("followed_%s" % tag)
        _repeat_old_state(w, pid, declined)
        w.advance(8 * DAY)
        w.tick(1)
        st = w.state()
        nt = [e for e in w.events("not_taken") if e.get("proposal_id") == pid]
        check("%s: the repeat of L15 (same id, its run followed to item_done) is not taken back: no adopted_run, "
              "the first run's result not read again (runs 1), not declined anew, no card" % tag,
              not w.events("adopted_run") and (st["discover"] or {}).get("runs") == 1
              and (declined or pid not in (st.get("declined") or []))
              and not [c for c in st.get("cards") or [] if "already ran" in c.get("title", "")]
              and nt and "new id" in str(nt[-1].get("reasons")), (w.events("adopted_run"), st["discover"], nt))
        w.tick(2)
        ids = l15_ids(w)
        check("  %s: the next tick proposes L15 under a new id and launches it" % tag,
              ids[-1] != pid and len(discoveries(w)) == 2, (ids, len(discoveries(w))))


def _lagged(lag):
    """X.Context with the executor's clock `lag` seconds ahead of the ticker's
    (in production a run's record is written after the tick started)."""
    orig = X.Context

    class Lagged(orig):
        def __init__(self, *a, **kw):
            c = kw.get("clock")
            if callable(c):
                kw["clock"] = lambda c=c: c() + lag
            orig.__init__(self, *a, **kw)
    return orig, Lagged


def t_first_tick_after_deploy():
    print("the first tick on a state written before ended_ids existed takes back no run the lanes followed")
    for lag in (0.0, 20.0):
        orig, lagged = _lagged(lag)
        X.Context = lagged
        try:
            w = World("deploy_lag%d" % lag)
            w.ready_r0()
            w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
            w.tick(2)
            w.segment(1, [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"]),
                          ("I3", "REJECT", 0.1, ["regression"], "hurts", ["src_b"]),
                          ("I4", "REJECT", 0.1, ["regression"], "hurts", ["src_b"])], done=True)
            for _ in range(10):
                w.tick(1)
                if [e for e in w.events("item_done") if e.get("lever") == "L24"]:
                    break
            done = [e for e in w.events("item_done") if e.get("lever") == "L24"]
            qu = ((w.state().get("sources") or {}).get("src_b") or {}).get("quarantined_utc")
            rec = [r for r in w.executions() if r.get("lever") == "L24" and r.get("status") == "executed"]
            check("lag %ds: L24 quarantined src_b and was done in its tick (the record's ts %s, the tick %s)"
                  % (lag, rec and rec[-1].get("ts"), qu), len(done) == 1 and qu and rec, done)
            edit_state(w, lambda st: st.pop("ended_ids", None))
            w.tick(1)
            st = w.state()
            check("  lag %ds: the next tick on that state without ended_ids takes back nothing: one item_done of L24, "
                  "quarantined_utc unchanged, ended_ids taken from the campaign ledger" % lag,
                  not w.events("adopted_run")
                  and len([e for e in w.events("item_done") if e.get("lever") == "L24"]) == 1
                  and ((st.get("sources") or {}).get("src_b") or {}).get("quarantined_utc") == qu
                  and done[0]["proposal_id"] in (st.get("ended_ids") or []),
                  (w.events("adopted_run"), st.get("ended_ids")))
        finally:
            X.Context = orig
    # a run the lanes lost on such a state is still taken back
    w = world("deploy_lost")
    pid, _rec = lost_tick(w)
    edit_state(w, lambda st: st.pop("ended_ids", None))
    w.tick(1)
    check("  a run lost on a state without ended_ids is still taken back (adopted_run, running)",
          [e.get("proposal_id") for e in w.events("adopted_run")] == [pid]
          and (w.lane("DATA").get("item") or {}).get("status") == "running", w.events("adopted_run"))


def main():
    for fn in (t_replay_lost_tick, t_recovered, t_live_declined, t_declined_watchdog, t_zero_estimate,
               t_l15_again, t_followed_not_taken_back, t_first_tick_after_deploy):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
