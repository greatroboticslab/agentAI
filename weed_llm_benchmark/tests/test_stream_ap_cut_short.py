#!/usr/bin/env python3
"""A segment cut refused for want of data is no failure of TRAIN (2026-10-05).

Live: at 04:53Z D31 called for L24 on zenodo_15808623 and D22 for L18 in the
same tick. The quarantine ran at once; L18 was filed (a replay pass for other
code) and submitted at 05:15Z on evidence taken after the quarantine, which
showed Q 319 against M 1,364. The cutter refused ('[inc2.stream] ERROR: nothing
cut for weed_stream_v1_fork1364_s002: short') and the ticker counted TRAIN's
failed step (lane fails 1, a step failure; a second would have held TRAIN).
The refusal line was not even read: run_inc2_build.sh writes a stream build's
provenance as stream_<sid>.json and the ticker looked it up under the segment's
name.

Pinned here, in the stream world:
  * a 'short' refusal, found under provenance/stream_<sid>.json by its job id,
    while the queue holds Q < M: not_taken, a new id for the next cut, the
    build's estimate released, no lane failure, no step failure; twice in a
    row, no hold;
  * the same refusal while Q >= M: the cutter and D22 disagree, a failure as
    before and a card;
  * L18 is not proposed in the tick an L24 is (a quarantine changes Q), nor
    before a snapshot shows the quarantine, and with the queue fallen below M
    no cut is ever submitted;
  * a filed L18 whose D22 no longer calls for a cut (read with the TRAIN lane
    idle) is not submitted; approved and not run, its approval is closed and
    it is withdrawn, and the next cut gets a new id;
  * while its approval is pending (only a person decides it), or could not be
    closed (a person started it meanwhile), it is kept filed, never
    withdrawn: a person's run of it from the INC page is followed by TRAIN,
    never orphaned (live, L18 is filed pending after every deploy until the
    replay gate passes, as at 05:03Z on 10-05), and under the envelope the
    same approval runs once the queue calls for the cut again.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import DAY, NAME, X, World, check  # noqa: E402


def train_world(tag, Q=None, **kw):
    w = World(tag, **kw)
    w.ready_r0()
    w.queue(4 * w.M if Q is None else Q, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    return w


def builds(w):
    return [x for x in w.submits if "run_inc2_build.sh" in " ".join(x["argv"]) and "build" in x["argv"]
            and "inc2.stream" in x["argv"]]


def l18(w):
    return [e for e in w.events("proposed") if e.get("lever") == "L18"]


def refuse_short(w, why="short"):
    """The cutter's refusal as run_inc2_build.sh records it: provenance/stream_<sid>.json (one record per
    stream; its last attempt is this build's job) and the job FAILED."""
    job = builds(w)[-1]
    exp = job["name"][len("inc_build_"):]
    jid = next(j["id"] for j in w.squeue if j["name"] == job["name"])
    p = w.inc / "_campaign" / "provenance" / ("stream_%s.json" % w.sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(p.read_text()) if p.exists() else {"format": "inc_autopilot.provenance/1",
                                                        "exp": "stream_%s" % w.sid, "attempts": []}
    rec["attempts"].append({"job_id": jid, "status": "build_failed", "build_rc": 1, "started_utc": W.utc(w.t[0]),
                            "refusal": "[inc2.stream] ERROR: nothing cut for %s: %s" % (exp, why)})
    p.write_text(json.dumps(rec))
    w.job_done(job["name"], state="FAILED")
    return exp


def t_short_below_m():
    print("a 'short' refusal while Q < M: not taken, never a failure; twice, no hold")
    w = train_world("short_below_m")
    w.tick(2)
    first = l18(w)
    check("Q >= 4M: L18 proposed and its build submitted", len(first) == 1 and len(builds(w)) == 1,
          [x["name"] for x in w.submits])
    exp = refuse_short(w)
    w.queue(w.M // 4, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(2)
    st = w.state()
    nt = [e for e in w.events("not_taken") if e.get("proposal_id") == first[0]["proposal_id"]]
    lane = w.lane("TRAIN")
    bud = X.budget_now(dict(w.config(), name=NAME), w.xctx)
    check("the refusal is read from provenance/stream_<sid>.json by its job id, and recorded not_taken with Q < M",
          nt and "nothing cut for %s: short" % exp in str(nt[0].get("reasons")) and nt[0].get("Q") == w.M // 4
          and nt[0].get("M") == w.M, nt)
    check("  no failure: the lane's failure count and the step's untouched, no hold, the item cleared",
          not int(lane.get("fails") or 0) and not lane.get("hold") and lane.get("item") is None
          and not [k for k in st.get("step_failures") or {} if k.startswith("L18:")]
          and not w.events("failed"), (lane, st.get("step_failures")))
    check("  its id retired (failed_ids) and its build's estimate released",
          first[0]["proposal_id"] in st.get("failed_ids")
          and not [x for x in bud["committed"] if x.get("child_exp") == exp], bud["committed"])
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(2)
    second = l18(w)
    check("the queue refills: the next cut is proposed under a new id and submitted",
          len(second) == 2 and second[1]["proposal_id"] != first[0]["proposal_id"] and len(builds(w)) == 2,
          [e.get("proposal_id") for e in second])
    refuse_short(w, "short_after_guard")
    w.queue(w.M // 4, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(2)
    lane = w.lane("TRAIN")
    check("  refused short again (short_after_guard) with Q < M: still no failure and no hold after two",
          len([e for e in w.events("not_taken") if e.get("lever") == "L18" and e.get("Q") is not None]) == 2
          and not int(lane.get("fails") or 0) and not lane.get("hold") and not (w.state().get("paused")), lane)


def t_short_q_ge_m():
    print("a 'short' refusal while Q >= M: the cutter and D22 disagree -> a failure and a card")
    w = train_world("short_ge_m")
    w.tick(2)
    exp = refuse_short(w)
    w.tick(2)
    st = w.state()
    cards = [c for c in st.get("cards") or [] if "refused it short while the queue holds" in c["title"]]
    fl = [e for e in w.events("failed") if e.get("lever") == "L18"]
    check("counted as TRAIN's failed step (its refusal line read) and a card names the disagreement",
          fl and "nothing cut for %s: short" % exp in str(fl[0].get("reasons")) and int(w.lane("TRAIN").get("fails")
                                                                                      or 0) == 1
          and len(cards) == 1 and exp in cards[0]["title"], (fl, cards, w.lane("TRAIN")))


def t_l24_same_tick():
    print("L18 is not proposed while a quarantine (L24) is pending or unseen, so it never cuts on a stale Q")
    w = train_world("l24_same_tick")
    w.tick(2)
    exp1 = "%s_s001" % w.sid
    rows = [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"]),
            ("I3", "REJECT", 0.1, ["regression"], "hurts", ["src_b"]),
            ("I4", "REJECT", 0.1, ["regression"], "hurts", ["src_b"])]
    w.segment(1, rows, done=True)
    tick_of = {}
    for i in range(10):
        n = len(w.ledger())
        w.tick(1)
        for e in w.ledger()[n:]:
            if e.get("event") in ("proposed", "executed") and e.get("lever") in ("L24", "L18"):
                tick_of.setdefault((e["event"], e["lever"]), i)
        if ("executed", "L24") in tick_of:
            break
    t24 = tick_of.get(("proposed", "L24"))
    same = [e for e in w.events("proposed") if e.get("lever") == "L18" and e.get("child_exp") != exp1]
    check("D31 calls for L24 on src_b; in that tick no second segment is proposed (the cut waits for it)",
          t24 is not None and not same, (tick_of, [e.get("child_exp") for e in same]))
    # the quarantine took src_b's rows out of the queue: Q falls below M
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(6)
    waits = [e for e in w.events("not_taken") if e.get("lever") == "L18"
             and "quarantine" in str(e.get("reasons"))]
    check("  after it, L18 waits for a snapshot that shows the quarantine, then the queue reads Q < M: no second "
          "cut is ever proposed or submitted", waits and not [e for e in w.events("proposed") if e.get("lever") == "L18"
                                                              and e.get("child_exp") != exp1]
          and len(builds(w)) == 1, ([e.get("reasons") for e in waits], [x["name"] for x in builds(w)]))


def person_page(w):
    """The INC page's run of an approval: executor.execute_approved as a person, outside the ticker."""
    cfg = w.config()
    camp = {"name": NAME, "autonomy": cfg.get("autonomy") or "off", "autonomy_granted_by": cfg.get("autonomy_granted_by"),
            "envelope_su": cfg.get("envelope_su"), "daily_cap_su": cfg.get("daily_cap_su"), "paused_reason": None}
    return camp, X.Context(slurm_sh=w, lab_repo=str(w.lab), resources=W.RES, clock=w.clock, domain=w.domain)


def person_runs(w, aid, approve=True):
    if approve:
        W.AP.decide(w.domain, aid, "approve", W.OWNER, "test: a person cuts", w.clock(), root=str(w.lab))
    camp, page = person_page(w)
    return X.execute_approved(aid, camp, page, invoked_by=W.OWNER)


def filed_cut(w):
    it = w.lane("TRAIN").get("item") or {}
    return it, (it.get("proposal") or {}).get("id"), it.get("approval_id")


def held_events(w, pid):
    return [e for e in w.events("waiting") if e.get("proposal_id") == pid and "kept filed" in str(e.get("reasons"))]


def held_cards(w):
    return [c for c in w.state().get("cards") or [] if "not submitted: the queue no longer calls for it" in c["title"]]


def t_filed_pending_held():
    print("a filed L18 whose D22 no longer calls for a cut, its approval pending: kept filed, never withdrawn, and a "
          "person's run of it is followed")
    w = train_world("filed_held", autonomy="off")
    w.tick(2)
    it, pid, aid = filed_cut(w)
    check("autonomy off: the cut (R3) is filed for a person", it.get("lever") == "L18" and it.get("status") == "filed"
          and aid, it.get("status"))
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    st = w.state()
    it1, pid1, aid1 = filed_cut(w)
    held, cards = held_events(w, pid), held_cards(w)
    check("the queue falls below M: not submitted, and not withdrawn while its approval is pending (only a person "
          "decides it): the item stays filed with its approval, its id not retired",
          not w.events("withdrawn") and not builds(w) and it1.get("status") == "filed" and pid1 == pid
          and aid1 == aid and pid not in (st.get("failed_ids") or []) and w.approvals()[aid]["status"] == "pending",
          (w.events("withdrawn"), it1.get("status"), st.get("failed_ids")))
    check("  one waiting event and one card, naming the approval, over three ticks",
          len(held) == 1 and "D22 no longer calls for a cut" in str(held[0].get("reasons"))
          and held[0].get("approval_id") == aid and len(cards) == 1 and aid in cards[0]["title"], (held, cards))
    # a person approves the pending cut and runs it from the INC page
    ran = person_runs(w, aid)
    w.tick(3)
    it2, pid2, _a = filed_cut(w)
    check("  a person approves and runs it from the INC page: the TRAIN lane follows that run (executed_elsewhere, "
          "running, the same id), and no second cut is proposed beside it",
          ran.get("status") == "executed" and it2.get("status") == "running" and pid2 == pid
          and [e for e in w.events("executed_elsewhere") if e.get("approval_id") == aid]
          and len(l18(w)) == 1 and len(builds(w)) == 1, (ran.get("status"), it2.get("status"), len(l18(w))))
    refuse_short(w)
    w.tick(2)
    nt = [e for e in w.events("not_taken") if e.get("proposal_id") == pid]
    lane = w.lane("TRAIN")
    check("  followed to its end: the cutter refuses it short with Q < M, recorded not_taken, no lane failure",
          nt and lane.get("item") is None and not int(lane.get("fails") or 0) and not w.events("failed"), (nt, lane))


def t_filed_approved_closed():
    print("a filed L18 whose D22 no longer calls for a cut, approved and not run: its approval closed, then withdrawn")
    w = train_world("filed_closed", autonomy="off")
    w.tick(2)
    _it, pid, aid = filed_cut(w)
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    W.AP.decide(w.domain, aid, "approve", W.OWNER, "test: a person approves the cut", w.clock(), root=str(w.lab))
    w.tick(2)
    wd = [e for e in w.events("withdrawn") if e.get("proposal_id") == pid]
    st = w.state()
    ex = (w.approvals().get(aid) or {}).get("execution") or {}
    check("approved and not yet run: its approval closed unsubmitted (no job), then the item withdrawn, never "
          "submitted, its id retired (failed_ids, not declined), no failure",
          wd and wd[0].get("approval_closed") is True and "D22 no longer calls for a cut" in str(wd[0].get("reasons"))
          and ex.get("phase") == "failed" and (ex.get("outcome") or {}).get("status") == "not_submitted"
          and not builds(w) and w.lane("TRAIN").get("item") is None and pid in st.get("failed_ids")
          and pid not in (st.get("declined") or []) and not int(w.lane("TRAIN").get("fails") or 0), (wd, ex))
    ran = person_runs(w, aid, approve=False)
    check("  a person's Run of the closed approval from the INC page is refused (one approval runs once)",
          ran.get("status") == "refused" and "already executed" in str(ran.get("reasons")) and not builds(w),
          (ran.get("status"), ran.get("reasons")))
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    it2, pid2, _a = filed_cut(w)
    check("  the queue refills: the same cut is proposed again under a new id (never skipped as declined)",
          it2.get("lever") == "L18" and pid2 not in (None, pid), it2.get("status"))


def t_filed_run_by_person():
    print("a filed L18 a person approved and ran is followed, never withdrawn, however the queue reads after")
    wp = train_world("filed_run_by_person", autonomy="off")
    wp.tick(2)
    itp, pidp, aidp = filed_cut(wp)
    # the person cuts after a snapshot already shows the queue below M (the item is still filed then)
    wp.queue(wp.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(wp.t[0] - 2 * DAY))
    wp.tick(1)
    ran = person_runs(wp, aidp)
    wp.tick(2)
    itp2, pidp2, _a = filed_cut(wp)
    check("a filed cut a person ran from the INC page is followed (executed_elsewhere, running), never withdrawn",
          ran.get("status") == "executed" and itp2.get("status") == "running" and pidp2 == pidp
          and not wp.events("withdrawn") and wp.events("executed_elsewhere"), (ran.get("status"), itp2.get("status")))
    # the person runs it while the ticker reads it: approved and not run when _ready reads the approvals, started by
    # the person before the ticker closes it; the close fails, and the item is kept for its run to be followed
    w = train_world("filed_close_race", autonomy="off")
    w.tick(2)
    _it, pid, aid = filed_cut(w)
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    W.AP.decide(w.domain, aid, "approve", W.OWNER, "test: a person cuts", w.clock(), root=str(w.lab))
    orig = W.S.StreamRun._cut_still_called
    got = {}

    def racing(self):
        if not got:
            got["ran"] = person_runs(w, aid, approve=False)
        return orig(self)
    W.S.StreamRun._cut_still_called = racing
    try:
        w.tick(1)
    finally:
        W.S.StreamRun._cut_still_called = orig
    held = [e for e in w.events("waiting") if e.get("proposal_id") == pid and "could not be closed" in str(e.get("reasons"))]
    check("  the person starts it between the ticker's read and its close: the close fails and the item is kept "
          "filed, not withdrawn", (got.get("ran") or {}).get("status") == "executed" and held
          and not w.events("withdrawn") and pid not in (w.state().get("failed_ids") or []),
          ((got.get("ran") or {}).get("status"), w.events("withdrawn")))
    w.tick(2)
    it2, pid2, _a = filed_cut(w)
    check("  its run is followed on the next tick (executed_elsewhere, running), one build, no second cut",
          it2.get("status") == "running" and pid2 == pid and w.events("executed_elsewhere")
          and len(builds(w)) == 1 and len(l18(w)) == 1, (it2.get("status"), len(builds(w))))


def t_envelope_held():
    print("under the envelope, an L18 filed while the replay gate is stale and then not called for: kept filed, and "
          "the same approval runs once the queue calls for the cut again")
    w = train_world("envelope_held", replay=False)
    w.tick(2)
    it, pid, aid = filed_cut(w)
    check("the replay gate is stale: the cut (R3) is filed, its approval pending",
          it.get("lever") == "L18" and it.get("status") == "filed" and aid
          and w.approvals()[aid]["status"] == "pending", it.get("status"))
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    _it1, pid1, aid1 = filed_cut(w)
    check("  the queue falls below M: kept filed (not withdrawn, not submitted), one card",
          pid1 == pid and aid1 == aid and not w.events("withdrawn") and not builds(w) and len(held_cards(w)) == 1,
          (pid1, w.events("withdrawn")))
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.replay_pass()
    w.tick(3)
    it2, pid2, aid2 = filed_cut(w)
    a = w.approvals().get(aid) or {}
    check("  the queue refills and the replay passes: the same approval is granted under the envelope and runs, one "
          "build, no second cut and no second approval",
          it2.get("status") == "running" and pid2 == pid and aid2 == aid and a.get("status") == "approved"
          and a.get("decision_basis") == "envelope" and len(builds(w)) == 1 and len(l18(w)) == 1
          and len([x for x in w.approvals().values() if (x.get("context") or {}).get("lever") == "L18"]) == 1,
          (it2.get("status"), a.get("status"), len(builds(w)), len(l18(w))))


def main():
    for fn in (t_short_below_m, t_short_q_ge_m, t_l24_same_tick, t_filed_pending_held, t_filed_approved_closed,
               t_filed_run_by_person, t_envelope_held):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
