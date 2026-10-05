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
    idle) is withdrawn before submission, and the next cut gets a new id.
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


def t_filed_withdrawn():
    print("a filed L18 whose D22 no longer calls for a cut is withdrawn before submission")
    w = train_world("filed_withdrawn", autonomy="off")
    w.tick(2)
    it = w.lane("TRAIN").get("item") or {}
    pid = (it.get("proposal") or {}).get("id")
    check("autonomy off: the cut (R3) is filed for a person", it.get("lever") == "L18" and it.get("status") == "filed",
          it.get("status"))
    w.queue(w.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    wd = [e for e in w.events("withdrawn") if e.get("proposal_id") == pid]
    st = w.state()
    check("the queue falls below M: the filed L18 is withdrawn (D22 read with the lane idle no longer calls for it), "
          "never submitted, its id retired (failed_ids, not declined), no failure",
          wd and "D22 no longer calls for a cut" in str(wd[0].get("reasons")) and not builds(w)
          and w.lane("TRAIN").get("item") is None and pid in st.get("failed_ids")
          and pid not in (st.get("declined") or []) and not int(w.lane("TRAIN").get("fails") or 0), (wd, w.lane("TRAIN")))
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    it2 = w.lane("TRAIN").get("item") or {}
    check("  the queue refills: the same cut is proposed again under a new id (never skipped as declined)",
          it2.get("lever") == "L18" and (it2.get("proposal") or {}).get("id") not in (None, pid), it2.get("status"))
    # a filed cut a person approved and ran from the INC page is followed, never withdrawn, whatever Q reads after
    wp = train_world("filed_run_by_person", autonomy="off")
    wp.tick(2)
    itp = wp.lane("TRAIN").get("item") or {}
    # the person cuts after a snapshot already shows the queue below M (the item is still filed then)
    wp.queue(wp.M // 2, boxes={"Purslane": 90}, oldest_utc=W.utc(wp.t[0] - 2 * DAY))
    wp.tick(1)
    W.AP.decide(wp.domain, itp.get("approval_id"), "approve", W.OWNER, "test: a person cuts", wp.clock(),
                root=str(wp.lab))
    cfg = wp.config()
    camp = {"name": NAME, "autonomy": cfg.get("autonomy") or "off", "autonomy_granted_by": cfg.get("autonomy_granted_by"),
            "envelope_su": cfg.get("envelope_su"), "daily_cap_su": cfg.get("daily_cap_su"), "paused_reason": None}
    page = X.Context(slurm_sh=wp, lab_repo=str(wp.lab), resources=W.RES, clock=wp.clock, domain=wp.domain)
    ran = X.execute_approved(itp.get("approval_id"), camp, page, invoked_by=W.OWNER)
    wp.tick(2)
    itp2 = wp.lane("TRAIN").get("item") or {}
    check("  a filed cut a person ran from the INC page is followed (executed_elsewhere, running), never withdrawn",
          ran.get("status") == "executed" and itp2.get("status") == "running"
          and (itp2.get("proposal") or {}).get("id") == (itp.get("proposal") or {}).get("id")
          and not wp.events("withdrawn") and wp.events("executed_elsewhere"), (ran.get("status"), itp2.get("status")))


def main():
    for fn in (t_short_below_m, t_short_q_ge_m, t_l24_same_tick, t_filed_withdrawn):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
