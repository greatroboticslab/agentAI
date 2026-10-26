#!/usr/bin/env python3
"""A lane never stalls on work already done, a diagnosis reads its lever's result, and one bad source never stops
collection (2026-10-26; docs/CONTINUOUS_LOOP.md, amendment 2026-10-26). End to end on the stream ticker
(campaign.tick, the executor, the real stream_remote verbs on the fake cluster of tests/test_stream_ap_world.py), from
the lab's records (tests/fixtures/platform_stalls/).

  l4      the live D31 case: s001's three data-blamed increments of one source, L4's label audit run and done after
          the segment was frozen, its id filed as declined and lane MAINT carded stalled (the lab's state of
          2026-10-26: no done_ids, no audits_run). The next ticks: no L4 proposed, recovered or declined again, the
          stall ended (lane_stall_ended) and never carded again; the frozen record taken again with the audit (the old
          one moved aside, never deleted); D31 reads the audit of 2026-10-05 byte for byte (0 of 4,092 boxes judged,
          image_not_in_crops) as 'not judged', one card, the increments rejected and the source quarantined as they are.
  judged  the same with an audit that judged every box above its baseline: X3's card, no L4 again.
  done    a diagnosis that does not read its lever's result (D31 as it was) calling for the L4 that ran: one
          done_recalled event and one card, never run, declined or stalled on again.
  pairs   the same guard for DHOLD -> L17 scan-holds, DCMP -> LC, D23 -> L19 and D25 -> L21 (unit level).
  lift    a source quarantined on D31's word and lifted by a person (inc2.stream unquarantine): no L24 again; a
          person's denial of D31's L24 likewise.
  units   _audit_result's readings, the byte plan, the collector refusal's classes.
  palmer  the 2026-10-07 sequence: the top-ranked lab source refused over_byte_plan twice (the collector's result as
          the lab recorded it): planned again once at the per-source cap, then closed with its reason; lane DATA
          neither failed nor held; the next candidate proposed on the next tick.
  systemic two download failures still hold lane DATA (the stop-loss for a systemic failure stands).
  release the lab's DATA state of 2026-10-26 (held since 2026-10-07): a person's release (stream release) frees the
          lane; gh_geezacoleman__palmer_detection is proposed under a new id (its third attempt), and refused again
          it is closed with its reason while the lane takes the next source.

Run:  python3 tests/test_stream_ap_stalls.py          (STALLS=l4,palmer runs only those)
"""
import copy
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402

from test_stream_ap_world import DAY, NAME, OWNER, S, check  # noqa: E402

DS = W.DS
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "platform_stalls"
LIVE_L4 = json.loads((FIX / "l4_live_2026-10-26.json").read_text())
LIVE_DATA = json.loads((FIX / "data_live_2026-10-26.json").read_text())
AUDIT_BYTES = (FIX / LIVE_L4["audit"]["file"]).read_bytes()
GIB10 = 10737418240
CAP = 50 * 10 ** 9


def state_write(w, st):
    S.StreamPaths(str(w.lab), "weed").state(NAME).write_text(json.dumps(st))


def led_since(w, n0):
    return w.ledger()[n0:]


def diag(w, did):
    p = w.lab / "results" / "framework" / "_brain" / "weed" / "inc" / "campaigns" / NAME / "diagnoses.json"
    doc = json.loads(p.read_text()) if p.exists() else {}
    return next((d for d in doc.get("diagnoses") or [] if d.get("id") == did), {})


# ----------------------------------------------------------------- the L4 case
# s001's commit of 2026-10-05: inc0001 data (P_data 0.0, truth neutral), inc0002 hold, inc0003 data (the regression
# guard failed), inc0004 data (truth hurts, P_data 0.22); one source for all four
ROWS = [("inc0001", "REJECT", 0.0, [], "neutral", ["zenodo_15808623"]),
        ("inc0002", "HOLD", 0.5, [], "neutral", ["zenodo_15808623"]),
        ("inc0003", "REJECT", 0.0, ["regression"], "neutral", ["zenodo_15808623"]),
        ("inc0004", "REJECT", 0.22, [], "hurts", ["zenodo_15808623"])]


def l4_world(tag):
    """The stream with s001 committed (the live dispositions), D31's L4 and L24 run and L4's job done; returns the
    world, the segment and L4's proposal id."""
    w = W.World(tag)
    w.ready_r0()
    w.queue(0)
    w.tick()
    exp = w.segment(1, ROWS)
    w._commit(exp)
    for _ in range(10):
        w.tick()
        if [e for e in w.events("executed") if e.get("lever") == "L4"]:
            break
    l4 = [e for e in w.events("proposed") if e.get("lever") == "L4"]
    w.job_done("inc_audit")
    pid = l4[0]["proposal_id"] if l4 else None
    for _ in range(6):
        w.tick()
        if [e for e in w.events("item_done") if e.get("proposal_id") == pid]:
            break
    w.tick()
    return w, exp, pid


def live_overlay(w, pid):
    """The lab's state of 2026-10-26 on the world: L4's id filed as declined (the executor refused it as already run on
    2026-10-05), lane MAINT carded stalled, and none of this change's records (done_ids, audits_run, refreeze)."""
    st = w.state()
    for k in ("done_ids", "audits_run", "refreeze"):
        st.pop(k, None)
    st["declined"] = [x for x in LIVE_L4["declined"] if x != LIVE_L4["proposal_id"]] + [pid]
    st["stalls"] = {"MAINT": dict(LIVE_L4["stalls"]["MAINT"], last=LIVE_L4["stalls"]["MAINT"]["last"].replace(
        LIVE_L4["proposal_id"], pid))}
    state_write(w, st)


def write_audit(w, exp, audit_bytes):
    p = w.inc / exp / "audit" / "label_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(audit_bytes)


def t_l4():
    print("l4: the live D31 case (s001's label audit ran on 2026-10-05; lane MAINT carded stalled 6 times)")
    w, exp, pid = l4_world("stl_l4")
    st = w.state()
    check("set-up: L4 ran on s001's data-blamed increments (inc0001, inc0003, inc0004) and its item ended done, after "
          "s001 was frozen", pid and any(e.get("event") == "item_done" and e.get("proposal_id") == pid
                                        for e in w.ledger()) and exp in (st.get("frozen") or {}),
          (pid, sorted(st.get("frozen") or {})))
    fz_path = S.StreamPaths(str(w.lab), "weed").frozen(NAME, exp)
    fz_before = fz_path.read_bytes()
    check("set-up: the frozen record of s001 carries no audit",
          b"audit/label_audit.json" not in fz_before.split(b'"missing"')[0], len(fz_before))
    live_overlay(w, pid)
    write_audit(w, exp, AUDIT_BYTES)            # the audit job's record, as the cluster holds it since 05:01:53Z
    n0 = len(w.ledger())
    w.tick(16)                                  # beyond the watchdog's 12 ticks
    led = led_since(w, n0)
    st = w.state()
    l4 = [e for e in led if e.get("lever") == "L4" and e.get("event") in ("proposed", "executed", "recovered",
                                                                         "not_taken")]
    check("no L4 proposed, run, recovered or declined again over 16 ticks", not l4,
          [(e.get("event"), (e.get("reasons") or [""])[0][:120]) for e in l4])
    check("lane MAINT's stall ended (lane_stall_ended) and no lane_stalled since", any(
        e.get("event") == "lane_stall_ended" and e.get("lane") == "MAINT" for e in led)
        and not [e for e in led if e.get("event") == "lane_stalled"] and "MAINT" not in (st.get("stalls") or {}),
        ([(e.get("event"), e.get("lane")) for e in led if "stall" in str(e.get("event"))], st.get("stalls")))
    aside = sorted((fz_path.parent / "superseded").glob("%s.*.json" % exp))
    check("the frozen record taken again with the audit; the old one moved aside byte for byte (nothing deleted)",
          (st.get("refreeze") or {}).get(exp, {}).get("state") == "done" and len(aside) == 1
          and aside[0].read_bytes() == fz_before
          and b"%s/audit/label_audit.json" % exp.encode() in fz_path.read_bytes(),
          (st.get("refreeze"), [a.name for a in aside]))
    d31 = diag(w, "D31")
    res = ((d31.get("detail") or {}).get("audits") or {}).get(exp) or {}
    steps = res.get("steps") or {}
    check("D31 reads the audit of 2026-10-05: not judged, 0 of 1,364 boxes on each of inc0001, inc0003, inc0004 "
          "(image_not_in_crops), and calls for no L4", res.get("verdict") == "not_judged"
          and sorted(steps) == ["inc0001", "inc0003", "inc0004"]
          and all(s["judged"] == 0 and s["boxes"] == 1364 and s["not_judged_reasons"] == {"image_not_in_crops": 1364}
                  for s in steps.values()) and "L4" not in (d31.get("levers") or [])
          and "OP_CARD" in (d31.get("levers") or []), (d31.get("levers"), d31.get("summary"), res))
    cards = [e for e in led if e.get("event") == "card" and "D31" in str(e.get("title"))]
    check("one card says so (not judged, the rejections stand, a person decides any re-admission)",
          len(cards) == 1 and "did not judge" in cards[0].get("detail", "") and "re-admission" in cards[0]["detail"],
          [c.get("title") for c in cards])
    srcs = st.get("sources") or {}
    check("the source stays quarantined (L24 ran once, cited D31), nothing more to collect from it",
          (srcs.get("zenodo_15808623") or {}).get("status") == "quarantined"
          and len([e for e in w.ledger() if e.get("lever") == "L24" and e.get("event") == "executed"]) == 1,
          srcs.get("zenodo_15808623"))
    check("the increments stay rejected (the commit's dispositions unchanged)",
          [e.get("dispositions") for e in w.stream_ledger() if e.get("event") == "commit"][-1]
          == {"inc0001": "data", "inc0002": "hold", "inc0003": "data", "inc0004": "data"})
    check("the live record: L4's audit of 2026-10-05 is the fixture's (sha256 %s...)" % LIVE_L4["audit"]["sha256"][:12],
          __import__("hashlib").sha256(AUDIT_BYTES).hexdigest() == LIVE_L4["audit"]["sha256"])


def t_judged():
    print("judged: the audit judged every box and finds the labels above its baseline -> X3, no L4 again")
    w, exp, pid = l4_world("stl_judged")
    aud = json.loads(AUDIT_BYTES)
    for s in ("inc0001", "inc0003", "inc0004"):
        sp = aud["audits"][s]["species"]
        sp.update(boxes_judged=1364, above_baseline=(s == "inc0003"), conflict=200, conflict_rate=0.147)
    write_audit(w, exp, json.dumps(aud).encode())
    n0 = len(w.ledger())
    w.tick(6)
    led = led_since(w, n0)
    d31 = diag(w, "D31")
    res = ((d31.get("detail") or {}).get("audits") or {}).get(exp) or {}
    check("D31: labels the cause on inc0003 -> X3; no L4", res.get("verdict") == "labels" and res.get("labels")
          == ["inc0003"] and "X3" in (d31.get("levers") or []) and "L4" not in (d31.get("levers") or []),
          (d31.get("levers"), res.get("verdict")))
    check("X3's card raised; no L4 proposed again", any(e.get("event") == "card" and str(e.get("title")).startswith(
        "X3") for e in led) and not [e for e in led if e.get("lever") == "L4" and e.get("event") == "proposed"])
    aud2 = json.loads(AUDIT_BYTES)
    for s in ("inc0001", "inc0003", "inc0004"):
        aud2["audits"][s]["species"].update(boxes_judged=1300, above_baseline=False)
    write_audit(w, exp, json.dumps(aud2).encode())
    st = w.state()
    st["refreeze"][exp]["state"] = "pending"     # taken again, as a re-audit would be
    state_write(w, st)
    w.tick(3)
    res = (((diag(w, "D31").get("detail") or {}).get("audits") or {}).get(exp)) or {}
    check("judged on 1,300 of 1,364 boxes and not above: labels not the cause, nothing called for",
          res.get("verdict") == "not_labels" and not (set(diag(w, "D31").get("levers") or []) & {"L4", "X3"}),
          (res.get("verdict"), diag(w, "D31").get("levers")))


def t_done():
    print("done: D31 as it was (L4 on every tick, its result never read): the done id is recalled once, never a stall")
    orig = DS.d31

    def d31_old(v, prior=None, include_prior=False):
        d = orig(v, prior, include_prior)
        det = d.get("detail") or {}
        if d.get("fired"):
            exps = {}
            for b in det.get("blamed") or []:
                exps.setdefault(b["exp"], []).append(b["step"])
            det["propose"] = [{"lever": "L4", "exp": e, "steps": sorted(set(s))} for e, s in sorted(exps.items())] + [
                p for p in det.get("propose") or [] if p.get("lever") != "L4"]
            d["levers"] = ["L4"] + [x for x in d.get("levers") or [] if x != "L4"]
        return d
    DS.d31 = d31_old
    try:
        w, exp, pid = l4_world("stl_done")
        # the lab's form of 2026-10-26: the id filed as declined, no done_ids (found by the ledger's item_done)
        live_overlay(w, pid)
        n0 = len(w.ledger())
        w.tick(16)
        led = led_since(w, n0)
        st = w.state()
    finally:
        DS.d31 = orig
    rec = [e for e in led if e.get("done_recalled")]
    check("one done_recalled event naming L4's id and D31, and one card", len(rec) == 1 and rec[0].get(
        "proposal_id") == pid and rec[0].get("trigger") == ["D31"] and len(
        [e for e in led if e.get("event") == "card" and "still calls for it" in str(e.get("title"))]) == 1,
        [(e.get("event"), e.get("proposal_id")) for e in rec])
    check("never run, recovered or declined again; the stall of the lab's state ended, no stall card", not [
        e for e in led if e.get("lever") == "L4" and e.get("event") in ("proposed", "executed", "recovered")]
        and not [e for e in led if e.get("lever") == "L4" and "declined earlier" in str(e.get("reasons"))]
        and not [e for e in led if e.get("event") == "lane_stalled"]
        and any(e.get("event") == "lane_stall_ended" and e.get("lane") == "MAINT" for e in led),
        [(e.get("event"), e.get("lever")) for e in led if e.get("lever") == "L4" or "stall" in str(e.get("event"))])
    check("the id is recorded done (done_ids)", (st.get("done_ids") or {}).get(pid, {}).get("lever") == "L4",
          (st.get("done_ids") or {}).get(pid))


def t_pairs():
    print("pairs: the same guard for DHOLD -> L17 scan-holds, DCMP -> LC, D23 -> L19, D25 -> L21")
    w = W.World("stl_pairs")
    w.ready_r0()
    w.queue(0)
    w.tick(2)
    run = W_run(w)
    run._build_evidence()
    run.diags = []
    sid = w.sid
    cases = [("STOP", "L17", {"lever": "L17", "hold": "funnel_F9", "verb": "scan-holds"}, "DHOLD"),
             ("MAINT", "LC", {"lever": "LC", "exp": "%s_c001" % sid}, "DCMP"),
             ("TRAIN", "L19", {"lever": "L19", "exp": "%s_s001" % sid}, "D23"),
             ("STOP", "L21", {"lever": "L21", "to": "P_0"}, "D25")]
    for ln, lever, pr, did in cases:
        lane_of = S.LANE_OF.get(lever)
        d = {"id": did, "fired": True, "cites": []}
        p = run._proposal(lever, pr, did, d)
        if p is None:
            check("%s -> %s renders" % (did, lever), False, pr)
            continue
        run.st = w.state()
        run._note_done(p["id"], {"utc": W.utc(w.t[0] - DAY), "lever": p.get("lever"), "exp": p.get("child_exp")})
        for k in [x for x in (run.st.get("lanes") or {})]:
            run.st["lanes"][k]["item"] = None
        run.st["stalls"] = {lane_of: {"since_utc": W.utc(w.t[0] - DAY), "ticks": 20, "carded": W.utc(w.t[0]),
                                      "levers": [lever], "triggers": [did], "last": "x"}}
        run._intents = (lambda a=(lane_of, lever, pr, did, d): [a])
        n0 = len(w.ledger())
        run._materialise()
        led = led_since(w, n0)
        # the executor refusing the done id as already run (a lane that held it): done, never declined
        run.st["lanes"][lane_of]["item"] = {"proposal": p, "lever": p.get("lever") or lever, "status": "proposed",
                                            "trigger": did, "diagnoses": [d]}
        run._on_result(lane_of, run.st["lanes"][lane_of]["item"], {"status": "refused", "reasons": [
            "proposal %s already ran (executed at %s); a proposal runs once" % (p["id"], W.utc(w.t[0] - DAY))]})
        check("%s -> %s: refused as already run, its item done: cleared, not declined" % (did, lever),
              run.st["lanes"][lane_of].get("item") is None and p["id"] not in (run.st.get("declined") or []),
              run.st.get("declined"))
        check("%s -> %s (lane %s): done id recalled once, not proposed, no stall, the stall ended" % (
            did, lever, lane_of), run.st["lanes"][lane_of].get("item") is None and not run._stalls.get(lane_of)
            and any(e.get("done_recalled") and e.get("proposal_id") == p["id"] for e in led)
            and any(e.get("event") == "lane_stall_ended" and e.get("lane") == lane_of for e in led)
            and lane_of not in (run.st.get("stalls") or {}),
            [(e.get("event"), e.get("lever")) for e in led])


def W_run(w):
    paths = W.C.Paths(str(w.lab))
    run = S.StreamRun(NAME, w.config(), paths, W.C._SshBudget(None), w.clock, w.log, w.hooks, lambda: W.RES, None,
                      None, None)
    run.st = w.state()
    run.dom = W.LS.load_domain("weed")
    run.th = W.LS.load_thresholds()
    return run


def t_lift():
    print("lift: a D31 quarantine lifted by a person is not taken again; a denied one is not proposed again")
    w = W.World("stl_lift")
    w.ready_r0()
    w.queue(0)
    w.tick()
    e1 = w.segment(1, [("J1", "REJECT", 0.0, [], "neutral", ["src_q"]), ("J2", "REJECT", 0.0, [], "neutral",
                                                                                  ["src_q"])])
    w._commit(e1)
    w.tick(3)
    w.job_done("inc_audit")
    w.tick(3)
    q = [e for e in w.stream_ledger() if e.get("event") == "quarantine" and e.get("source") == "src_q"]
    check("set-up: src_q blamed twice, quarantined by L24 cited D31", q and q[-1].get("cite") == "D31",
          [e.get("event") for e in w.stream_ledger()])
    w.queue(0, extra={"quarantined_sources": {"src_q": {"cite": "D31", "seq": q[-1]["seq"], "utc": q[-1]["utc"]}}}
            if q else None)
    w.tick(2)
    w.stream_event("unquarantine", source="src_q", by=OWNER)
    w.queue(0, extra={"quarantined_sources": {}})
    st = w.state()
    st["sources"]["src_q"]["status"] = "admitted"       # a person's lift: the source collectable again
    state_write(w, st)
    n0 = len(w.ledger())
    w.tick(14)
    led = led_since(w, n0)
    l24 = [e for e in led if e.get("lever") == "L24" and e.get("event") in ("proposed", "executed", "recovered",
                                                                           "not_taken")]
    d31 = diag(w, "D31")
    check("after the person's lift: D31 calls for no L24 on src_q (the blames before the lift not counted) and names "
          "the lift; no L24 again, no stall", not l24 and not [e for e in led if e.get("event") == "lane_stalled"]
          and "src_q" in ((d31.get("detail") or {}).get("lifted") or {})
          and not [x for x in (d31.get("detail") or {}).get("propose") or [] if x.get("lever") == "L24"],
          ([(e.get("event"), e.get("lever")) for e in l24], d31.get("summary")))
    # two blames committed after the lift count again: D31 calls for L24 on src_q; the ticker's own record of the
    # source still says quarantined (a person's lift changes the stream's ledger only), so nothing runs and nothing
    # stalls, and the id would be a new one (the lift starts a new attempt)
    e2 = w.segment(2, [("J3", "REJECT", 0.0, [], "neutral", ["src_q"]), ("J4", "REJECT", 0.0, [], "neutral",
                                                                                  ["src_q"])])
    w._commit(e2)
    n1 = len(w.ledger())
    w.tick(6)
    d31 = diag(w, "D31")
    run = W_run(w)
    run._build_evidence()
    st = run.st
    st["sources"]["src_q"]["status"] = "admitted"
    p = run._proposal("L24", {"lever": "L24", "source": "src_q", "cite": "D31"}, "D31", {"cites": []})
    first = [e for e in w.ledger() if e.get("lever") == "L24" and e.get("event") == "proposed"]
    check("two blames committed after the lift: D31 calls for L24 on src_q again; no stall; rendered (were the "
          "ticker's record lifted too) under a new id", any(x.get("lever") == "L24" and x.get("source") == "src_q"
                                                           for x in (d31.get("detail") or {}).get("propose") or [])
          and not [e for e in led_since(w, n1) if e.get("event") == "lane_stalled"]
          and p is not None and first and p["id"] != first[0]["proposal_id"],
          (d31.get("summary"), p and p["id"], [e.get("proposal_id") for e in first]))
    # a denial (shadow mode: L24 filed for a person, denied) keeps the source
    run = W_run(w)
    run._build_evidence()
    it = {"lever": "L24", "proposal": {"params": {"source": "src_k", "cite": "D31"}}, "approval_id": "a1"}
    run._person_denied(it, {"decided_by": OWNER})
    check("a person's denial of D31's L24 keeps the source (blame_kept_by)",
          run.st["sources"]["src_k"].get("blame_kept_by") == OWNER, run.st["sources"].get("src_k"))
    run.st["sources"]["src_k"]["status"] = "admitted"
    p = run._proposal("L24", {"lever": "L24", "source": "src_k", "cite": "D31"}, "D31", {"cites": []})
    check("and D31's L24 on it renders nothing", p is None, p)


class FakeEv(object):
    def __init__(self, docs):
        self.docs = docs

    def json(self, name):
        return self.docs.get(name)

    def cite(self, name, ptr):
        return {"artifact": name, "pointer": ptr}


def t_units():
    print("units: the audit's readings, the byte plan, the collector refusal's classes")
    aud = json.loads(AUDIT_BYTES)
    ev = FakeEv({"x/audit/label_audit.json": aud})
    got = DS._audit_result(None, ev, "x", ["inc0001", "inc0003", "inc0004"], 0.5)
    check("the live audit: not judged (0 of 1,364 on each step); above_baseline false read as no finding",
          got and got[0]["verdict"] == "not_judged" and all(s["above_baseline"] is None for s in got[0]["steps"].values()),
          got and got[0])
    check("an audit missing a blamed step is no result", DS._audit_result(None, ev, "x", ["inc0001", "inc0002"], 0.5)
          is None)
    check("no audit in the evidence: no result", DS._audit_result(None, FakeEv({}), "x", ["inc0001"], 0.5) is None)
    a2 = copy.deepcopy(aud)
    a2["audits"]["inc0001"]["species"].update(boxes_judged=681, above_baseline=True)
    r = DS._audit_result(None, FakeEv({"x/audit/label_audit.json": a2}), "x", ["inc0001"], 0.5)[0]
    check("681 of 1,364 judged (under half): not judged, its above_baseline not read", r["verdict"] == "not_judged", r)
    a2["audits"]["inc0001"]["species"].update(boxes_judged=682)
    r = DS._audit_result(None, FakeEv({"x/audit/label_audit.json": a2}), "x", ["inc0001"], 0.5)[0]
    check("682 of 1,364 judged and above: labels", r["verdict"] == "labels" and r["labels"] == ["inc0001"], r)
    pilot = FakeEv({"audit/x_audit.json": a2})
    check("the pilot's root audit is read too", DS._audit_result(None, pilot, "x", ["inc0001"], 0.5)[0]["artifact"]
          == "audit/x_audit.json")
    w = W.World("stl_units")
    w.tick()
    run = W_run(w)
    run.st.setdefault("sources", {})
    check("byte plan: the estimate first (64,145,408 bytes, as on 2026-10-07)", run._byte_plan("a", 64145408)
          == 64145408)
    check("byte plan: a 60 GB estimate plans the 50 GB per-source cap", run._byte_plan("a", 60e9) == CAP)
    check("byte plan: no estimate plans one 10 GiB shard", run._byte_plan("a", None) == GIB10)
    run.st["sources"]["a"] = {"byte_plan": 39 * 10 ** 9}
    check("byte plan: a source a refusal raised plans the raised plan", run._byte_plan("a", 64145408) == 39 * 10 ** 9)
    it = {"lever": "L16L", "proposal": {"params": {"source": "a", "max_bytes": GIB10}}}
    res = json.loads(LIVE_DATA["lab_jobs"][0]["result"]["tail"].split("[collect] fetch: ", 1)[1])
    res10 = json.loads(json.dumps(res).replace("64145408", str(GIB10)))
    check("over_byte_plan at the plan: replan", (run._collector_refusal(it, 2, res10) or {}).get("kind") == "replan")
    check("over_byte_plan at a cap below the plan (the lab's disk, the envelope): the lane's failure",
          run._collector_refusal(it, 2, res) is None)
    check("download_failed, listing_failed, credentials_missing, byte_envelope: the lane's failure", all(
        run._collector_refusal(it, 2, {"status": st_, "code": c, "action": a, "risk": r_}) is None for st_, c, a, r_ in (
            ("refused", "download_failed", "refuse", None), ("refused", "listing_failed", "refuse", None),
            ("held", "credentials_missing", "hold", "R3"), ("held", "byte_envelope", "hold", "R3"))))
    check("never_train (close), no_data_files: close; licence_unresolved: a person", (
        run._collector_refusal(it, 2, {"status": "closed", "code": "never_train", "action": "close"}) or {}).get(
        "kind") == "close" and (run._collector_refusal(it, 2, {"status": "refused", "code": "no_data_files",
                                                              "action": "refuse"}) or {}).get("kind") == "close"
        and (run._collector_refusal(it, 2, {"status": "held", "code": "licence_unresolved", "action": "hold",
                                            "risk": "R3"}) or {}).get("kind") == "person")
    check("a crash (rc 1), a sync or an admit: the lane's failure", run._collector_refusal(it, 1, res10) is None
          and run._collector_refusal(dict(it, lever="L16S"), 2, res10) is None
          and run._collector_refusal(dict(it, lever="L17"), 2, res10) is None)


# ----------------------------------------------------------------- DATA
PALMER = "gh_geezacoleman__palmer_detection"
NEXT = "weedai_0935d5a0"


def data_world(tag):
    w = W.World(tag)
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "ftp": "fail", "weedai": "pass"})
    # the ranking of 2026-10-26 (D20 replay on the lab): palmer first (500 placeholder boxes over 0.064 GB), then
    # weedai_0935d5a0 (Palmer amaranth, Cornell; 995 images, 1.32 GB)
    w.candidates([{"id": PALMER, "provider": "github", "licence": "MIT", "target_classes": ["PalmerAmaranth"],
                   "bytes": 64145408, "expected_target_boxes": 500},
                  {"id": NEXT, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PalmerAmaranth"],
                   "bytes": 1.32e9, "images": 995, "expected_target_boxes": 1410}])
    return w


def palmer_jobs(w):
    return [x for x in w.runner.launched if "fetch" in x["argv"] and PALMER in x["argv"]]


def refuse(w, launched, plan):
    """The collector's refusal as the lab recorded it on 2026-10-07 (rc 2, its closing JSON in the tail), with the
    byte cap the refused attempt was planned at."""
    res = copy.deepcopy(LIVE_DATA["lab_jobs"][0]["result"])
    res = json.loads(json.dumps(res).replace("64145408", str(plan)))
    w.runner.results[launched["job"]] = {k: res[k] for k in ("ok", "rc", "seconds", "stderr_tail", "tail")}
    # and its lab collector ledger, as the lab's recorded it: started, then failed with nothing fetched
    p = w.lab / "results" / "framework" / "inc" / "intake" / "sources.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "a") as fh:
        for ev in ({"event": "fetch_started", "provider": "github", "ref": "geezacoleman/Palmer-detection"},
                   {"event": "fetch_failed", "provider": "github", "ref": "geezacoleman/Palmer-detection",
                    "reason": "over_byte_plan", "bytes_partial": 0, "seconds": 4.9,
                    "detail": "github Palmer-detection-main.zip: more than the %d-byte cap" % plan}):
            fh.write(json.dumps(dict(ev, source=PALMER, ts=W.utc(w.t[0]), format="collect-source-event/1"),
                                sort_keys=True) + "\n")


def plan_of(launched):
    a = launched["argv"]
    return int(a[a.index("--max-bytes") + 1])


def t_palmer():
    print("palmer: the 2026-10-07 sequence, the collector's refusal is the source's outcome")
    w = data_world("stl_palmer")
    w.tick(3)
    jobs = palmer_jobs(w)
    check("palmer ranks first and is fetched on the lab at its estimate, 64,145,408 bytes (as on 2026-10-07)",
          len(jobs) == 1 and plan_of(jobs[0]) == 64145408, [(j["job"], plan_of(j)) for j in jobs])
    if not jobs:
        return
    refuse(w, jobs[0], plan_of(jobs[0]))
    n0 = len(w.ledger())
    w.tick(3)
    jobs = palmer_jobs(w)
    st = w.state()
    check("refused over_byte_plan (the lab's result of 2026-10-07 byte for byte): not a failure (lane fails 0, no "
          "step failure, not held), planned again at the 50 GB the per-source limit leaves, under a new id",
          st["lanes"]["DATA"].get("fails") == 0 and not st["lanes"]["DATA"].get("hold")
          and not [k for k in st.get("step_failures") or {} if k.startswith("L16")]
          and len(jobs) == 2 and plan_of(jobs[1]) == CAP and jobs[1]["job"] != jobs[0]["job"]
          and any(e.get("event") == "source_replanned" for e in led_since(w, n0)),
          (st["lanes"]["DATA"], [(j["job"], plan_of(j)) for j in jobs]))
    if len(jobs) < 2:
        return
    refuse(w, jobs[1], plan_of(jobs[1]))
    n1 = len(w.ledger())
    w.tick(4)
    st = w.state()
    led = led_since(w, n1)
    src = st["sources"].get(PALMER) or {}
    check("refused again at the cap: palmer closed with its reason (how a person may approve more)",
          src.get("status") == "closed" and "over_byte_plan" in str(src.get("closed_reason"))
          and "approved_bytes" in str(src.get("closed_reason")), src)
    check("lane DATA neither failed nor held; palmer in the zero-yield run (one source, under the 3-source fuse)",
          st["lanes"]["DATA"].get("fails") == 0 and not st["lanes"]["DATA"].get("hold")
          and [r.get("source") for r in st.get("zero_run") or []] == [PALMER], (st["lanes"]["DATA"], st.get("zero_run")))
    nxt = [e for e in led if e.get("event") == "proposed" and e.get("lane") == "DATA"]
    check("the next candidate (%s) is proposed" % NEXT, any(NEXT in " ".join(e.get("argv") or []) for e in nxt),
          [(e.get("lever"), e.get("argv")) for e in nxt])
    check("no stop-loss card, no lane_held", not [e for e in w.ledger() if e.get("event") == "lane_held"])


def t_systemic():
    print("systemic: two download failures still hold lane DATA")
    w = data_world("stl_systemic")
    w.tick(3)
    for k in range(2):
        jobs = palmer_jobs(w)
        if len(jobs) <= k:
            check("attempt %d launched" % (k + 1), False, [j["job"] for j in jobs])
            return
        res = {"ok": False, "rc": 2, "seconds": 3.0,
               "stderr_tail": "[collect] fetch refused: download_failed: github: HTTP 502\n",
               "tail": '[collect] fetch: {"action": "refuse", "code": "download_failed", "detail": "github: HTTP 502", '
                       '"status": "refused", "format": "collect-result/1", "verb": "fetch"}\n'}
        w.runner.results[jobs[k]["job"]] = res
        w.tick(3)
    st = w.state()
    check("download_failed twice: lane DATA held by its stop-loss (2 consecutive failed steps)",
          str(st["lanes"]["DATA"].get("hold") or "").startswith("stop-loss: 2 consecutive failed steps")
          and st["sources"][PALMER].get("failures") == 2, (st["lanes"]["DATA"], st["sources"].get(PALMER)))


def t_release():
    print("release: the lab's DATA state of 2026-10-26, released by a person after the fix is deployed")
    w = data_world("stl_release")
    w.tick(1)                                   # the first tick only observes
    st = w.state()
    st["lanes"]["DATA"].update({k: v for k, v in LIVE_DATA["lane_DATA"].items() if k in ("hold", "hold_utc", "fails")})
    st["lanes"]["DATA"]["hold_utc"] = W.utc(w.t[0])        # after the configure stamp of the world
    st["sources"][PALMER] = copy.deepcopy(LIVE_DATA["source"])
    st["step_failures"] = {k: v for k, v in LIVE_DATA["step_failures"].items()}
    state_write(w, st)
    w.tick(2)
    check("held: nothing proposed on DATA", not w.state()["lanes"]["DATA"].get("item")
          and w.state()["lanes"]["DATA"].get("hold"))
    out = S.release(NAME, OWNER, cfg_hooks=w.hooks, lab_repo=str(w.lab), clock=w.clock)
    n0 = len(w.ledger())
    w.tick(3)
    led = led_since(w, n0)
    st = w.state()
    jobs = palmer_jobs(w)
    check("the release frees lane DATA (lane_released, fails 0, its step failures cleared)",
          out.get("held_lanes", {}).get("DATA") and any(e.get("event") == "lane_released" and e.get("lane") == "DATA"
                                                        for e in led)
          and not st["lanes"]["DATA"].get("hold") and not [k for k, v in (st.get("step_failures") or {}).items()
                                                           if (v or {}).get("lane") == "DATA"], st["lanes"]["DATA"])
    check("palmer proposed under a new id at its estimate (its third attempt)", jobs and plan_of(jobs[-1])
          == 64145408 and st["sources"][PALMER].get("attempts") == 3
          and not [e for e in led if e.get("proposal_id") == LIVE_DATA["step_failures"]["L16L:01d67f6e8755"][
              "proposal_id"]], ([(j["job"], plan_of(j)) for j in jobs], st["sources"].get(PALMER)))
    if not jobs:
        return
    refuse(w, jobs[-1], plan_of(jobs[-1]))
    n1 = len(w.ledger())
    w.tick(3)
    st = w.state()
    led = led_since(w, n1)
    check("refused again (its zip is 84,379,800 bytes): its attempts used up, palmer is closed with its reason; lane "
          "DATA neither failed nor held, and the next candidate (%s) is proposed" % NEXT,
          (st["sources"].get(PALMER) or {}).get("status") == "closed"
          and "approved_bytes" in str((st["sources"].get(PALMER) or {}).get("closed_reason"))
          and not st["lanes"]["DATA"].get("hold") and int(st["lanes"]["DATA"].get("fails") or 0) == 0
          and any(NEXT in " ".join(e.get("argv") or []) for e in led if e.get("event") == "proposed"),
          (st["sources"].get(PALMER), st["lanes"]["DATA"].get("hold"),
           [(e.get("event"), e.get("lever")) for e in led][-8:]))


def main():
    want = [x for x in os.environ.get("STALLS", "").split(",") if x]
    try:
        for fn in (t_units, t_l4, t_judged, t_done, t_pairs, t_lift, t_palmer, t_systemic, t_release):
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
