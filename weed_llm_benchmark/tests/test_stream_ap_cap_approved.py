#!/usr/bin/env python3
"""An item filed past today's SU cap and then approved waits for the cap; it
never fails.

Live, 2026-10-01: the box-quality measurement arms are priced 16.7-18.7 SU
each and the stream's daily cap is 120 SU, so an arm proposed late in a UTC
day is filed for a person ("exceeds the SU left under today's cap"). Approved
before the day turned, executor.execute_approved asked the policy gate first;
the gate escalated the same shortfall to an approval the item already had, the
stream read that refusal as a failed step, cleared the lane and filed the arm
again under a new approval id, each tick (two in a row hold the MAINT lane).
execute_approved now checks the campaign's caps first: an approval never lifts
them, and a refusal that names today's cap is one the stream waits on.

Since the 2026-10-04 amendment (docs/CONTINUOUS_LOOP.md 6.6) no daily cap has a
default; the mechanism stays for a campaign that declares one, so this world
declares the 120 SU the stream had then.

Pinned, in the stream world of test_stream_ap_world:
  * with no daily cap declared, no arm is filed: each runs within the envelope;
  * an arm past today's (declared) cap is filed for a person, not run;
  * approved, it waits: the same approval stays open on the lane, no failed
    step and no refusal are recorded, nothing is filed again;
  * when the UTC day turns it runs, once, under that approval;
  * a second arm filed past the cap and left alone (no person) runs by the
    envelope after the turn, as before.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_units as U  # noqa: E402
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import AP, OWNER, check  # noqa: E402

DAY = 86400.0


def to_cap(tag, daily_cap_su=120.0):
    """The measure world with the arms built in order until one is filed past today's cap
    (the campaign declares `daily_cap_su`; None declares none, the default)."""
    w = U._measure_world(tag)
    if daily_cap_su is not None:
        w.set_config(daily_cap_su=daily_cap_su)
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    w.tick(3)
    for b in [x for x in w.dom["baselines"]["items"] if x.get("measure")]:
        it = w.lane("MAINT").get("item") or {}
        if it.get("lever") == "L23B" and it.get("status") == "filed":
            return w, it, b
        w.experiment(b["exp"], done=False)
        w.job_done("inc_build_%s" % b["exp"])
        w.tick(3)
    it = w.lane("MAINT").get("item") or {}
    return w, it, None


def l23b(w, event):
    return [e for e in w.events(event) if e.get("lever") == "L23B"]


def test_approved_waits():
    print("an arm filed past today's cap and approved waits for the cap, then runs under that approval")
    w, it, b = to_cap("cap_approved")
    filed = l23b(w, "filed")
    check("an arm past today's cap is filed for a person, not run",
          it.get("status") == "filed" and filed and "today's cap" in str(filed[-1].get("reasons")),
          (it.get("status"), filed[-1:] if filed else None))
    aid = it.get("approval_id")
    n_filed = len(filed)
    res = AP.decide(w.domain, aid, "approve", OWNER, "test: past today's cap", w.clock(), root=str(w.lab))
    check("  a person approves it", isinstance(res, dict) and res.get("ok"), res)
    w.tick(3)
    it2 = w.lane("MAINT").get("item") or {}
    check("approved, it waits: the same approval stays open on the lane; no failed step, no refusal, no new filing",
          it2.get("approval_id") == aid and it2.get("status") == "filed" and not int(w.lane("MAINT").get("fails") or 0)
          and not l23b(w, "refused") and not l23b(w, "failed") and len(l23b(w, "filed")) == n_filed
          and (AP.state(w.domain, root=str(w.lab)).get(aid) or {}).get("execution") is None,
          (it2.get("status"), it2.get("approval_id"), [e.get("reasons") for e in l23b(w, "refused")]))
    waits = [e for e in w.events("waiting") if e.get("lever") == "L23B"]
    check("  the wait names today's cap", waits and "today's cap" in str(waits[-1].get("reasons")), waits[-1:])
    w.advance(DAY)
    w.tick(2)
    ex = l23b(w, "executed")
    check("when the UTC day turns it runs, once, under that approval",
          ex and ex[-1].get("approval_id") == aid and ex[-1].get("child_exp") == (b or {}).get("exp")
          and [e.get("child_exp") for e in ex].count((b or {}).get("exp")) == 1,
          [(e.get("child_exp"), e.get("approval_id")) for e in ex])


def test_unapproved_runs_by_envelope():
    print("an arm filed past today's cap and left alone runs by the envelope once the day turns")
    w, it, b = to_cap("cap_alone")
    check("an arm past today's cap is filed for a person", it.get("status") == "filed", it.get("status"))
    w.tick(2)
    check("  it waits while today's cap holds", not [e for e in l23b(w, "executed")
                                                   if e.get("child_exp") == (b or {}).get("exp")])
    w.advance(DAY)
    w.tick(2)
    ex = [e for e in l23b(w, "executed") if e.get("child_exp") == (b or {}).get("exp")]
    check("after the turn it runs within the envelope (no person)", len(ex) == 1 and ex[0].get("basis") == "envelope",
          [(e.get("child_exp"), e.get("basis")) for e in ex])


def test_no_cap_no_filing():
    print("with no daily cap declared (the default since 2026-10-04) no arm is filed: each runs by the envelope")
    w, it, b = to_cap("cap_none", daily_cap_su=None)
    ex = l23b(w, "executed")
    check("no arm is filed for a person; every measurement arm ran within the envelope",
          b is None and not l23b(w, "filed") and len(ex) == len([x for x in w.dom["baselines"]["items"]
                                                                  if x.get("measure") and not x.get("requires")])
          and all(e.get("basis") == "envelope" for e in ex),
          (it.get("status"), [(e.get("child_exp"), e.get("basis")) for e in ex]))


def main():
    test_no_cap_no_filing()
    test_approved_waits()
    test_unapproved_runs_by_envelope()
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
