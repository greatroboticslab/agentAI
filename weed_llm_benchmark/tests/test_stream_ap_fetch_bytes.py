#!/usr/bin/env python3
"""The stream autopilot's fetch byte limits count the bytes a fetch actually
fetched, not the max_bytes it requested (docs/CONTINUOUS_LOOP.md 6.6, live
incident of 2026-10-01).

executor.stream_limits summed the REQUESTED max_bytes of every past fetch
attempt, whatever its outcome. On 2026-10-01 it refused the next lab fetch
(L16L) of mediatum_1717366 on all three byte caps ("source mediatum_1717366
would reach 71.5 GB (limit 50 GB unless a person approves)", "today's
fetches would reach 121.5 GB (limit 20 GB)", "the campaign's fetches would
reach 121.5 GB (collect_gb_envelope 60 GB)"), while what was on disk was
CottonWeedDet3's 5.59 GB (requested 50 GB), 19 MB of mediatum's failed lab
fetch (requested 10.7 GB) and nothing of its cluster review refused before
any download (requested 50 GB). Once the envelope read exceeded, every later
fetch waited for a person.

Pinned, in the stream world of test_stream_ap_world:
  * live: that history, written as the live platform holds it (a state with
    no record of fetch ends), counts 5.59 GB + 19 MB; no cap is reached and
    the next L16L of mediatum runs on the lab without a person;
  * in flight: a fetch that has not ended reserves its requested max_bytes;
    once it ends, the reservation gives way to what it fetched;
  * unknown: an ended fetch whose fetched bytes cannot be determined counts
    its requested max_bytes and the reason says so -- the lab's ledger
    unreadable, a cluster fold without fetched events, and a cluster fetch
    whose end the snapshot that folded the ledger already showed (its ledger
    may predate the end) until the next snapshot;
  * caps: the caps still hold on real bytes -- per source, per day by the time
    the bytes were fetched (a fetch requested 26 h ago whose bytes landed 2 h
    ago counts today), and over the campaign -- and the ticker files the
    next fetch for a person;
  * killed: a fetch killed before the collector recorded an end (the lab
    runner's timeout, the cluster's walltime: its ledger holds a
    fetch_started and nothing after) counts its requested max_bytes;
  * _failed records a fetch's end (partial bytes count, nothing reserved);
    a source fetched on both machines counts both; a snapshot without the
    fold keeps the last one; a cluster ledger not read whole (a torn line,
    a read stopped at the cap) gives a fold without fetched facts.
"""
import json
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import AP, S, SR, X, World, check  # noqa: E402

GB = 1e9
HOUR = 3600.0
COT = "kg_yuzhenlu__cottonweeddet3"
MED = "mediatum_1717366"
NEW = "hf_new_source"
# the L16 family's fetches, and those the lab runs (executor.FETCH_ACTIONS, LAB_FETCH_ACTIONS)
FETCH = ("inc_stream_collect", "inc_stream_collect_lab", "inc_stream_collect_review", "inc_stream_collect_review_lab")
LAB_FETCH = ("inc_stream_collect_lab", "inc_stream_collect_review_lab")
MED_CAND = {"id": MED, "provider": "mediatum", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
            "bytes": 10.8e9, "expected_target_boxes": 900}


# ------------------------------------------------------------------ the world
def write_state(w, st):
    S.StreamPaths(str(w.lab), w.domain).state(W.NAME).write_text(json.dumps(st))


def base_world(tag, daily=20.0, envelope=60.0):
    """Past R0 with an empty queue, the live campaign's byte caps, mediatum
    placed on the lab, no candidate, and a discovery that just ran (no L15)."""
    w = World(tag)
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "kaggle": "pass", "mediatum": "fail"})
    w.set_config(collect_gb_daily=daily, collect_gb_envelope=envelope)
    w.candidates([])
    w.tick()                                      # the first snapshot
    st = w.state()
    st["discover"] = {"last_utc": W.utc(w.clock()), "runs": 1, "found_new": 1, "empty_runs": 0}
    write_state(w, st)
    return w


def run_record(w, pid, action, lever, src, max_bytes, epoch, job=None):
    """One executed fetch in the execution log, as the executor writes it."""
    params = {"source": src, "max_bytes": int(max_bytes)}
    if action in LAB_FETCH:
        params["out"] = str(S.StreamPaths(str(w.lab), w.domain).staging()) + "/"
    rec = {"ts": W.utc(epoch), "epoch": epoch, "campaign": W.NAME, "actor": W.AUTO, "authorized_as": W.AUTO,
           "action": action, "risk": "R3" if lever in ("L16R", "L16RL") else "R2", "params": params,
           "meta_params": {}, "status": "executed", "ok": True, "reasons": [], "lever": lever, "trigger": ["D20"],
           "proposal_id": pid, "est_su": 0.0, "charged": True, "job_ids": [job] if job else [],
           "remote": {"ok": True, "known": True, "started": True, "may_have_run": True, "rc": 0,
                      "payload": {"ok": True, "detached": True, "lab_job": "fetch_%s" % pid} if not job else {},
                      "error": ""}, "run_id": uuid.uuid4().hex}
    assert X._log(w.xctx, rec), "set-up: the execution log was not written"


def ledger_event(w, event, lever, pid, epoch, **kw):
    """The ticker's campaign ledger entry for an item's end (item_done, failed)."""
    S._append(S.StreamPaths(str(w.lab), w.domain).ledger,
              dict({"utc": W.utc(epoch), "campaign": W.NAME, "event": event, "mode": "stream", "lane": "DATA",
                    "lever": lever, "proposal_id": pid, "decided_by": W.AUTO}, **kw))


def lab_ledger(w, rows):
    """Events of the lab collector's own ledger (collect.state on the lab INC_DIR)."""
    p = S.StreamPaths(str(w.lab), w.domain).lab_inc / "intake" / "sources.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "a") as fh:
        for r in rows:
            fh.write(json.dumps(dict({"format": "collect-source-event/1", "slurm_job_id": None}, **r),
                                sort_keys=True) + "\n")


def live_history(w):
    """The live record of 2026-09-30 / 10-01, as the platform held it before
    the change: three ended fetch attempts, their ends in the campaign ledger,
    the cluster's and the lab's collector ledgers, and a state written before
    fetch ends were recorded."""
    now = w.clock()
    t_cot, t_lab, t_rev = now - 20 * HOUR, now - 3 * HOUR, now - 1 * HOUR
    # CottonWeedDet3 on the cluster: requested 50 GB, fetched 5.59 GB, complete
    run_record(w, "p_cot", "inc_stream_collect", "L16", COT, 50 * GB, t_cot, job="47300001")
    w.sources([{"source": COT, "event": "fetch_started", "ts": W.utc(t_cot + 60), "slurm_job_id": "47300001"},
               {"source": COT, "event": "fetched", "status": "fetched", "ts": W.utc(t_cot + 110),
                "bytes": 5590000000, "complete": True, "remaining": 0, "slurm_job_id": "47300001"}])
    ledger_event(w, "item_done", "L16", "p_cot", t_cot + 600)
    # mediatum on the lab: requested 10.7 GB, gt.csv (19 MB) on disk, then download_failed
    run_record(w, "p_med_lab", "inc_stream_collect_lab", "L16L", MED, 10.7 * GB, t_lab)
    lab_ledger(w, [{"source": MED, "event": "fetch_started", "ts": W.utc(t_lab + 30)},
                   {"source": MED, "event": "fetch_failed", "ts": W.utc(t_lab + 900), "reason": "download_failed",
                    "bytes_partial": 19000000},
                   {"source": MED, "event": "fetched", "status": "fetched", "ts": W.utc(t_lab + 900),
                    "bytes": 19000000, "complete": False, "files": 1}])
    ledger_event(w, "failed", "L16L", "p_med_lab", t_lab + 1200, reasons=["the lab process ended 2"])
    # its source review on the cluster: requested 50 GB, refused not_placed_on_cluster before any download
    run_record(w, "p_med_rev", "inc_stream_collect_review", "L16R", MED, 50 * GB, t_rev, job="47302914")
    ledger_event(w, "failed", "L16R", "p_med_rev", t_rev + 1200,
                 reasons=["job(s) 47302914 ended FAILED; refused: not_placed_on_cluster"])
    st = w.state()
    st.pop("fetch_ends", None)
    st.pop("cluster_fetched", None)
    st["sources"][COT] = {"status": "admitted", "attempts": 1, "placement": "cluster", "max_bytes": 50000000000}
    st["sources"][MED] = {"status": "candidate", "attempts": 2, "failures": 2, "placement": "lab",
                          "max_bytes": 50000000000}
    write_state(w, st)


def live_world(tag, **kw):
    w = base_world(tag, **kw)
    live_history(w)
    w.tick()                      # the next snapshot folds the cluster's ledger, after every end
    return w


def stream_camp(w):
    """The campaign as the ticker hands it to the executor (StreamRun.camp),
    read from the state the last tick wrote."""
    sr = S.StreamRun.__new__(S.StreamRun)
    sr.name, sr.cfg, sr.st, sr.dom = W.NAME, w.config(), w.state(), w.dom
    sr.paths = S.StreamPaths(str(w.lab), w.domain)
    sr._artifact = lambda rel: None
    sr._limit_counts = lambda: {}
    return sr.camp


def limits(w, src, gb, lab=False):
    act = "inc_stream_collect_lab" if lab else "inc_stream_collect"
    params = {"source": src, "max_bytes": int(gb * GB)}
    if lab:
        params["out"] = "/lab/x/intake/staging/"
    return X.stream_limits(w.xctx, stream_camp(w), "L16", {"action": act, "params": params})


def tick_until(w, cond, n=6):
    for _ in range(n):
        if cond():
            return True
        w.tick()
    return cond()


def has(why, text):
    return any(text in x for x in why)


def lab_fetches(w, src):
    return [x for x in w.runner.launched if "fetch" in x["argv"] and src in x["argv"]]


def fetches_pending_person(w, src):
    return [a for a in w.approvals().values() if a.get("status") == "pending"
            and a.get("action") in FETCH and (a.get("params") or {}).get("source") == src]


# ------------------------------------------------------------------ the cases
def t_live():
    print("the live history counts what was fetched: 5.59 GB + 19 MB, not 110.7 GB requested")
    w = live_world("fb_live")
    st = w.state()
    assert len([r for r in w.executions() if r.get("action") in FETCH]) == 3 \
        and not (stream_camp(w).get("in_flight") or {}).get("L16"), \
        "set-up: three ended fetches in the execution log (50 + 10.7 + 50 GB requested), none running"
    check("the ends the platform recorded before the change are taken from the campaign ledger",
          set(st.get("fetch_ends") or {}) == {"p_cot", "p_med_lab", "p_med_rev"}, st.get("fetch_ends"))
    cf = st.get("cluster_fetched") or {}
    check("the snapshot's fold of the cluster ledger carries CottonWeedDet3's fetched event (5.59 GB), "
          "and nothing of mediatum", [b for _t, b in (cf.get("sources") or {}).get(COT) or []] == [5.59e9]
          and MED not in (cf.get("sources") or {}), cf)
    why = limits(w, MED, 10.8, lab=True)
    check("the next L16L of mediatum (10.8 GB) reaches no cap: source 10.8, today 16.4, campaign 16.4 of 50/20/60 GB",
          why == [], why)
    why = limits(w, MED, 49.99, lab=True)
    check("  the 19 MB on disk count for the source: a 49.99 GB request of it would reach 50.0 GB; the 10.7 GB "
          "and 50 GB requests count nothing",
          has(why, "source %s would reach 50.0 GB (limit 50 GB unless a person approves)" % MED)
          and not has(why, "max_bytes"), why)
    why = limits(w, NEW, 15)
    check("  today's fetches are what was fetched in the last 24 h: 5.59 + 0.019 + 15 = 20.6 GB",
          has(why, "today's fetches would reach 20.6 GB (limit 20 GB)") and not has(why, "max_bytes"), why)
    why = limits(w, NEW, 55)
    check("  the campaign's fetches are what was fetched: 5.59 + 0.019 + 55 = 60.6 GB",
          has(why, "the campaign's fetches would reach 60.6 GB (collect_gb_envelope 60 GB)")
          and not has(why, "max_bytes"), why)
    w.candidates([dict(MED_CAND)])
    w.tick()
    it = w.lane("DATA").get("item") or {}
    lf = lab_fetches(w, MED)
    check("the next L16L of mediatum runs without a person: launched on the lab under the autopilot's own "
          "authority, nothing filed",
          it.get("lever") == "L16L" and it.get("status") == "running" and len(lf) == 1
          and "--max-bytes" in lf[0]["argv"] and lf[0]["argv"][lf[0]["argv"].index("--max-bytes") + 1] == "10800000000"
          and not fetches_pending_person(w, MED)
          and not [e for e in w.events("filed") if e.get("lever") in ("L16", "L16L")],
          (it.get("lever"), it.get("status"), [x["argv"] for x in lf], w.events("filed")[-1:]))
    return w


def t_in_flight():
    print("a fetch in flight reserves its requested max_bytes; once it ends, what it fetched counts")
    w = live_world("fb_flight")
    w.candidates([dict(MED_CAND)])
    w.tick()
    it = w.lane("DATA").get("item") or {}
    if it.get("status") == "filed":
        # what is checked here is the reservation, whoever started the fetch
        AP.decide(w.domain, it.get("approval_id"), "approve", W.OWNER, "test: a person starts the fetch",
                  w.clock(), root=str(w.lab))
        w.tick()
        it = w.lane("DATA").get("item") or {}
    assert it.get("lever") == "L16L" and it.get("status") == "running", ("set-up: the L16L is not running", it)
    pid = it["proposal"]["id"]
    why = limits(w, NEW, 5)
    check("while mediatum's L16L runs, its 10.8 GB are reserved: today 5.59 + 0.019 + 10.8 + 5 = 21.4 GB, "
          "and the reason says so",
          has(why, "today's fetches would reach 21.4 GB (limit 20 GB); counted at their requested max_bytes: "
                   "10.8 GB requested by 1 fetch(es) not yet seen to end"), why)
    why = limits(w, MED, 39.2, lab=True)
    check("  and for the source: 0.019 + 10.8 + 39.2 = 50.0 GB > 50 GB",
          has(why, "source %s would reach 50.0 GB" % MED), why)
    # it ends: 3 GB fetched, complete
    lab_ledger(w, [{"source": MED, "event": "fetch_started", "ts": W.utc(w.clock() - 300)},
                   {"source": MED, "event": "fetched", "status": "fetched", "ts": W.utc(w.clock() - 60),
                    "bytes": 3000000000, "complete": True, "remaining": 0, "files": 40}])
    w.runner.finish(ok=True, rc=0, tail='[collect] fetch: {"bytes": 3000000000, "complete": true, "remaining": 0, '
                                         '"status": "fetched"}')
    w.tick()
    st = w.state()
    check("set-up: the lab run ended and the lane recorded its end",
          (st["sources"].get(MED) or {}).get("status") == "fetched" and pid in (st.get("fetch_ends") or {}),
          (st["sources"].get(MED), st.get("fetch_ends")))
    why = limits(w, NEW, 5)
    check("once it ended, the reservation is released: today 5.59 + 0.019 + 3.0 + 5 = 13.6 GB, within 20 GB",
          why == [], why)
    why = limits(w, NEW, 11.5)
    check("  its 3.0 GB count instead: 5.59 + 0.019 + 3.0 + 11.5 = 20.1 GB, nothing at max_bytes",
          has(why, "today's fetches would reach 20.1 GB (limit 20 GB)") and not has(why, "max_bytes"), why)


def t_unknown_lab():
    print("an ended lab fetch whose ledger cannot be read counts its requested max_bytes, and says so")
    w = base_world("fb_unk_lab")
    live_history(w)
    p = S.StreamPaths(str(w.lab), w.domain).lab_inc / "intake" / "sources.jsonl"
    with open(str(p), "a") as fh:
        fh.write('{"source": "%s", "event": "fetched", "bytes": 1\n' % MED)        # a torn line
    w.tick()
    why = limits(w, NEW, 5)
    check("mediatum's L16L counts its 10.7 GB request: today 5.59 + 10.7 + 5 = 21.3 GB, named in the reason",
          has(why, "today's fetches would reach 21.3 GB (limit 20 GB); counted at their requested max_bytes: "
                   "10.7 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)
    w.candidates([dict(MED_CAND)])
    w.tick()
    it = w.lane("DATA").get("item") or {}
    a = w.approvals().get(it.get("approval_id")) or {}
    check("  the next L16L of mediatum is filed for a person, the reason naming the unknown bytes",
          it.get("lever") == "L16L" and it.get("status") == "filed" and not lab_fetches(w, MED)
          and "fetched bytes not known" in str(a.get("reason")) + str([e.get("reasons") for e in w.events("filed")]),
          (it.get("lever"), it.get("status"), w.events("filed")[-1:]))


def t_unknown_cluster_fold():
    print("an ended cluster fetch counts its requested max_bytes when the fold carries no fetched events")
    real = SR._fold_sources

    def old_fold(rows, **kw):                     # a stream_remote before the change
        out = real(rows, **kw)
        for row in out.values():
            if isinstance(row, dict):
                row.pop("fetched_events", None)
                row.pop("open_fetches", None)
        return out
    SR._fold_sources = old_fold
    try:
        w = live_world("fb_unk_fold")
    finally:
        SR._fold_sources = real
    cf = w.state().get("cluster_fetched") or {}
    check("set-up: a snapshot shipped that fold after every end, and it carries no fetched facts",
          cf.get("utc") and cf.get("sources") is None
          and str(cf["utc"]) > max(str(v) for v in (w.state().get("fetch_ends") or {"": ""}).values()), cf)
    why = limits(w, NEW, 5)
    check("both cluster attempts count their requests, named in the reason: such a fold cannot say that mediatum's "
          "review fetched nothing either; today 50 + 50 + 0.019 + 5 = 105.0 GB",
          has(why, "today's fetches would reach 105.0 GB (limit 20 GB); counted at their requested max_bytes: "
                   "100.0 GB requested by 2 fetch(es) ended, fetched bytes not known"), why)


def t_unknown_until_snapshot():
    print("a cluster fetch whose end the folding snapshot showed counts its max_bytes until the next snapshot")
    w = base_world("fb_unk_snap", daily=2.5)
    w.candidates([{"id": NEW, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 2e9, "expected_target_boxes": 900}])
    tick_until(w, lambda: (w.lane("DATA").get("item") or {}).get("lever") == "L16"
               and (w.lane("DATA").get("item") or {}).get("status") == "running")    # after its candidate sync (L16S)
    it = w.lane("DATA").get("item") or {}
    assert it.get("lever") == "L16" and it.get("status") == "running" and it.get("job_ids"), \
        ("set-up: the L16 was not submitted", it.get("lever"), it.get("status"))
    pid, job = it["proposal"]["id"], it["job_ids"][0]
    w.sources([{"source": NEW, "event": "fetch_started", "slurm_job_id": job},
               {"source": NEW, "event": "fetched", "status": "fetched", "bytes": 1500000000, "complete": True,
                "remaining": 0, "slurm_job_id": job}])
    w.job_done("inc_stream_collect_")
    for _ in range(4):
        w.tick()
        if pid in (w.state().get("fetch_ends") or {}):
            break
    st = w.state()
    end, cf = (st.get("fetch_ends") or {}).get(pid), st.get("cluster_fetched") or {}
    check("set-up: the snapshot that showed the job ended also folded the ledger (the same tick)",
          end and cf.get("utc") == end, (end, cf.get("utc")))
    why = limits(w, "other", 1.2)
    check("that fold may predate the end: the fetch counts its 2.0 GB request (2.0 + 1.2 = 3.2 GB), named in the reason",
          has(why, "today's fetches would reach 3.2 GB (limit 2.5 GB); counted at their requested max_bytes: "
                   "2.0 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)
    for _ in range(4):
        w.tick()
        if str((w.state().get("cluster_fetched") or {}).get("utc")) > str(end):
            break
    why = limits(w, "other", 1.2)
    check("  after the next snapshot its 1.5 GB count (1.5 + 1.2 = 2.7 GB), nothing at max_bytes",
          has(why, "today's fetches would reach 2.7 GB (limit 2.5 GB)") and not has(why, "max_bytes"), why)


def t_caps():
    print("the caps still hold on the bytes actually fetched")
    w = base_world("fb_caps")
    now = w.clock()
    big = "hf_big"
    # two fetches of one source: 30 GB landed 30 h ago; 18 GB requested 26 h ago that landed 2 h ago
    run_record(w, "p_big1", "inc_stream_collect", "L16", big, 50 * GB, now - 40 * HOUR, job="47310001")
    run_record(w, "p_big2", "inc_stream_collect", "L16", big, 20 * GB, now - 26 * HOUR, job="47310002")
    w.sources([{"source": big, "event": "fetched", "status": "fetched", "ts": W.utc(now - 30 * HOUR),
                "bytes": 30000000000, "complete": False, "remaining": 9, "slurm_job_id": "47310001"},
               {"source": big, "event": "fetched", "status": "fetched", "ts": W.utc(now - 2 * HOUR),
                "bytes": 18000000000, "complete": False, "remaining": 2, "slurm_job_id": "47310002"}])
    ledger_event(w, "item_done", "L16", "p_big1", now - 30 * HOUR + 600)
    ledger_event(w, "item_done", "L16", "p_big2", now - 2 * HOUR + 600)
    st = w.state()
    st.pop("fetch_ends", None)
    st["sources"][big] = {"status": "admitted", "attempts": 2, "placement": "cluster"}
    write_state(w, st)
    w.tick()
    why = limits(w, big, 5)
    check("per source: 30 + 18 fetched + 5 = 53.0 GB > 50 GB",
          has(why, "source %s would reach 53.0 GB (limit 50 GB unless a person approves)" % big), why)
    why = limits(w, NEW, 5)
    check("per day, by the time the bytes were fetched: the 18 GB that landed 2 h ago count today (its request is "
          "26 h old), the 30 GB of 30 h ago do not: 18 + 5 = 23.0 GB > 20 GB",
          has(why, "today's fetches would reach 23.0 GB (limit 20 GB)") and not has(why, "source %s" % NEW), why)
    why = limits(w, NEW, 15)
    check("over the campaign: 48 + 15 = 63.0 GB > 60 GB",
          has(why, "the campaign's fetches would reach 63.0 GB (collect_gb_envelope 60 GB)"), why)
    w.candidates([{"id": NEW, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 5e9, "expected_target_boxes": 900}])
    tick_until(w, lambda: (w.lane("DATA").get("item") or {}).get("lever") == "L16")   # after its candidate sync
    it = w.lane("DATA").get("item") or {}
    a = w.approvals().get(it.get("approval_id")) or {}
    check("the ticker files the next fetch (5 GB) for a person on the day's cap, and submits nothing",
          it.get("lever") == "L16" and it.get("status") == "filed" and a.get("status") == "pending"
          and not [s for s in w.submits if NEW in s["argv"]]
          and "today's fetches would reach 23.0 GB" in json.dumps(w.events("filed")[-1:]),
          (it.get("lever"), it.get("status"), w.events("filed")[-1:]))


def start_med_lab(w):
    """The next L16L of mediatum (10.8 GB), running on the lab; its proposal id."""
    w.candidates([dict(MED_CAND)])
    w.tick()
    it = w.lane("DATA").get("item") or {}
    assert it.get("lever") == "L16L" and it.get("status") == "running" and lab_fetches(w, MED), \
        ("set-up: the L16L of mediatum is not running", it.get("lever"), it.get("status"))
    return it["proposal"]["id"]


def t_killed_lab():
    print("a lab fetch killed before it recorded an end counts its requested max_bytes")
    w = live_world("fb_killed_lab")
    pid = start_med_lab(w)
    # the collector began (finished tray files sit in staging), then the lab runner's timeout killed it:
    # its ledger holds the start and nothing after
    lab_ledger(w, [{"source": MED, "event": "fetch_started", "ts": W.utc(w.clock() - 60)}])
    w.runner.finish(ok=False, rc=None, error="TimeoutExpired: Command '...' timed out after 21600 seconds")
    w.tick()
    st = w.state()
    assert pid in (st.get("fetch_ends") or {}) and not (w.lane("DATA").get("item") or {}).get("status") == "running", \
        ("set-up: the lane did not record the killed run's end", st.get("fetch_ends"))
    why = limits(w, NEW, 5)
    check("its 10.8 GB count, named in the reason; the earlier failed attempt still counts its 19 MB: "
          "today 5.59 + 0.019 + 10.8 + 5 = 21.4 GB",
          has(why, "today's fetches would reach 21.4 GB (limit 20 GB); counted at their requested max_bytes: "
                   "10.8 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)
    why = limits(w, MED, 39.2, lab=True)
    check("  and for the source: 0.019 + 10.8 + 39.2 = 50.0 GB > 50 GB",
          has(why, "source %s would reach 50.0 GB" % MED), why)
    # a later attempt that closes leaves the killed one counted (its blobs are still in staging)
    lab_ledger(w, [{"source": MED, "event": "fetch_started", "ts": W.utc(w.clock() + 30)},
                   {"source": MED, "event": "fetched", "status": "fetched", "ts": W.utc(w.clock() + 90),
                    "bytes": 1000000000, "complete": True, "remaining": 0, "files": 3}])
    run_record(w, "p_med_lab3", "inc_stream_collect_lab", "L16L", MED, 5 * GB, w.clock())
    ledger_event(w, "item_done", "L16L", "p_med_lab3", w.clock() + 120)
    st = w.state()
    st["fetch_ends"]["p_med_lab3"] = W.utc(w.clock() + 120)
    write_state(w, st)
    why = limits(w, NEW, 5)
    check("  a later closed attempt counts its own 1.0 GB, the killed one still its 10.8 GB: "
          "5.59 + 0.019 + 10.8 + 1.0 + 5 = 22.4 GB",
          has(why, "today's fetches would reach 22.4 GB (limit 20 GB); counted at their requested max_bytes: "
                   "10.8 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)


def t_killed_cluster():
    print("a cluster fetch that hit its walltime before it recorded an end counts its requested max_bytes")
    w = base_world("fb_killed_cl", daily=2.5)
    w.candidates([{"id": NEW, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 2e9, "expected_target_boxes": 900}])
    tick_until(w, lambda: (w.lane("DATA").get("item") or {}).get("lever") == "L16"
               and (w.lane("DATA").get("item") or {}).get("status") == "running")
    it = w.lane("DATA").get("item") or {}
    assert it.get("lever") == "L16" and it.get("status") == "running" and it.get("job_ids"), \
        ("set-up: the L16 was not submitted", it.get("lever"), it.get("status"))
    pid, job = it["proposal"]["id"], it["job_ids"][0]
    w.sources([{"source": NEW, "event": "fetch_started", "slurm_job_id": job}])
    w.job_done("inc_stream_collect_", state="TIMEOUT")
    end = None
    for _ in range(6):
        w.tick()
        st = w.state()
        end = (st.get("fetch_ends") or {}).get(pid)
        if end and str((st.get("cluster_fetched") or {}).get("utc")) > str(end):
            break
    st = w.state()
    cf = st.get("cluster_fetched") or {}
    check("set-up: the lane recorded the end, and a later snapshot folded the ledger: its start is open",
          end and str(cf.get("utc")) > str(end) and (cf.get("open") or {}).get(NEW)
          and (cf.get("sources") or {}).get(NEW) == [], (end, cf))
    why = limits(w, "other", 1.2)
    check("its 2.0 GB count after that snapshot too (2.0 + 1.2 = 3.2 GB), named in the reason",
          has(why, "today's fetches would reach 3.2 GB (limit 2.5 GB); counted at their requested max_bytes: "
                   "2.0 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)


def t_failed_end():
    print("a fetch that fails after the state records ends counts what it finished, and reserves nothing")
    w = live_world("fb_failed_end")
    assert isinstance(w.state().get("fetch_ends"), dict), "set-up: the state records fetch ends"
    pid = start_med_lab(w)
    w.candidates([])
    lab_ledger(w, [{"source": MED, "event": "fetch_started", "ts": W.utc(w.clock() - 60)},
                   {"source": MED, "event": "fetch_failed", "ts": W.utc(w.clock() - 30), "reason": "download_failed",
                    "bytes_partial": 2000000000},
                   {"source": MED, "event": "fetched", "status": "fetched", "ts": W.utc(w.clock() - 30),
                    "bytes": 2000000000, "complete": False, "files": 30}])
    w.runner.finish(ok=False, rc=2, stderr_tail="refused: download_failed")
    w.tick()
    st = w.state()
    check("set-up: the lane failed the run and recorded its end",
          pid in (st.get("fetch_ends") or {}) and not (w.lane("DATA").get("item") or {}).get("status") == "running"
          and [e for e in w.events("failed") if e.get("proposal_id") == pid], st.get("fetch_ends"))
    why = limits(w, NEW, 12.5)
    check("its 2.0 GB count, at nothing requested: 5.59 + 0.019 + 2.0 + 12.5 = 20.1 GB",
          has(why, "today's fetches would reach 20.1 GB (limit 20 GB)") and not has(why, "max_bytes"), why)


def t_two_machines():
    print("a source fetched on both machines counts both machines' bytes")
    w = base_world("fb_two")
    now = w.clock()
    two = "hf_two"
    run_record(w, "p_two_cl", "inc_stream_collect", "L16", two, 20 * GB, now - 10 * HOUR, job="47320001")
    w.sources([{"source": two, "event": "fetch_started", "ts": W.utc(now - 10 * HOUR + 60), "slurm_job_id": "47320001"},
               {"source": two, "event": "fetched", "status": "fetched", "ts": W.utc(now - 9 * HOUR),
                "bytes": 4000000000, "complete": False, "remaining": 5, "slurm_job_id": "47320001"}])
    ledger_event(w, "item_done", "L16", "p_two_cl", now - 9 * HOUR + 600)
    run_record(w, "p_two_lab", "inc_stream_collect_lab", "L16L", two, 20 * GB, now - 5 * HOUR)
    lab_ledger(w, [{"source": two, "event": "fetch_started", "ts": W.utc(now - 5 * HOUR + 30)},
                   {"source": two, "event": "fetched", "status": "fetched", "ts": W.utc(now - 4 * HOUR),
                    "bytes": 3000000000, "complete": True, "remaining": 0, "files": 5}])
    ledger_event(w, "item_done", "L16L", "p_two_lab", now - 4 * HOUR + 600)
    st = w.state()
    st.pop("fetch_ends", None)
    write_state(w, st)
    w.tick()
    why = limits(w, NEW, 13.5)
    check("the cluster's 4.0 GB and the lab's 3.0 GB both count today: 4.0 + 3.0 + 13.5 = 20.5 GB",
          has(why, "today's fetches would reach 20.5 GB (limit 20 GB)") and not has(why, "max_bytes"), why)
    why = limits(w, two, 43.5)
    check("  and for the source: 4.0 + 3.0 + 43.5 = 50.5 GB",
          has(why, "source %s would reach 50.5 GB" % two) and not has(why, "max_bytes"), why)


def t_fold_missing():
    print("a snapshot without the cluster ledger's fold keeps the last fold: an end after it stays unknown")
    w = base_world("fb_fold_missing", daily=2.5)
    w.sources([{"source": "other_src", "event": "candidate"}])      # a cluster ledger the snapshots fold
    w.candidates([{"id": NEW, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 2e9, "expected_target_boxes": 900}])
    tick_until(w, lambda: (w.lane("DATA").get("item") or {}).get("lever") == "L16"
               and (w.lane("DATA").get("item") or {}).get("status") == "running")
    it = w.lane("DATA").get("item") or {}
    assert it.get("lever") == "L16" and it.get("status") == "running" and it.get("job_ids"), \
        ("set-up: the L16 was not submitted", it.get("lever"), it.get("status"))
    pid, job = it["proposal"]["id"], it["job_ids"][0]
    w.sources([{"source": NEW, "event": "fetch_started", "slurm_job_id": job},
               {"source": NEW, "event": "fetched", "status": "fetched", "bytes": 1500000000, "complete": True,
                "remaining": 0, "slurm_job_id": job}])
    before = (w.state().get("cluster_fetched") or {}).get("utc")
    led = w.inc / "intake" / "sources.jsonl"
    led.rename(led.with_name("sources.jsonl.away"))                   # unreadable from here on: no fold shipped
    w.job_done("inc_stream_collect_")
    for _ in range(4):
        w.tick()
    st = w.state()
    end, cf = (st.get("fetch_ends") or {}).get(pid), st.get("cluster_fetched") or {}
    check("set-up: the lane recorded the end, and no snapshot after it folded the ledger",
          end and before and cf.get("utc") == before and str(before) < str(end), (end, before, cf.get("utc")))
    why = limits(w, "other", 1.2)
    check("the fetch counts its 2.0 GB request (2.0 + 1.2 = 3.2 GB), named in the reason",
          has(why, "today's fetches would reach 3.2 GB (limit 2.5 GB); counted at their requested max_bytes: "
                   "2.0 GB requested by 1 fetch(es) ended, fetched bytes not known"), why)


def fold_shipped_without_facts(w):
    st = w.state()
    cf = st.get("cluster_fetched") or {}
    return bool(cf.get("utc")) and cf.get("sources") is None and isinstance((st.get("sources") or {}).get(COT), dict) \
        and str(cf["utc"]) > max(str(v) for v in (st.get("fetch_ends") or {"": ""}).values())


def t_cluster_ledger_incomplete():
    print("a cluster ledger the snapshot cannot read whole gives a fold without fetched facts")
    w = base_world("fb_cl_torn")
    live_history(w)
    with open(str(w.inc / "intake" / "sources.jsonl"), "a") as fh:
        fh.write('{"source": "%s", "event": "fetched", "bytes": 1\n' % MED)        # a torn line
    w.tick()
    check("set-up: a snapshot after every end shipped the fold, without fetched facts",
          fold_shipped_without_facts(w), w.state().get("cluster_fetched"))
    why = limits(w, NEW, 5)
    check("a torn line: both cluster attempts count their requests: 50 + 50 + 0.019 + 5 = 105.0 GB",
          has(why, "today's fetches would reach 105.0 GB (limit 20 GB); counted at their requested max_bytes: "
                   "100.0 GB requested by 2 fetch(es) ended, fetched bytes not known"), why)
    w._activate()
    arts = SR.stream_summary(w.sid)
    check("  the snapshot says why", any("1 unparsed line(s)" in n and "no fetched bytes" in n
                                         for n in arts.get("notes") or []), arts.get("notes"))
    # a ledger past the read cap: only its first lines are read
    w2 = base_world("fb_cl_cap")
    live_history(w2)
    real = SR._jsonl

    def capped(path, cap=SR.MAX_LEDGER_ROWS, info=None):
        return real(path, cap=1 if str(path).endswith("sources.jsonl") else cap, info=info)
    SR._jsonl = capped
    try:
        w2.tick()
        w2._activate()
        notes = SR.stream_summary(w2.sid).get("notes") or []
    finally:
        SR._jsonl = real
    check("set-up: a snapshot after every end shipped the fold, without fetched facts",
          fold_shipped_without_facts(w2), w2.state().get("cluster_fetched"))
    why = limits(w2, NEW, 5)
    check("a read stopped at the cap: both cluster attempts count their requests (105.0 GB), and the snapshot says so",
          has(why, "today's fetches would reach 105.0 GB (limit 20 GB); counted at their requested max_bytes: "
                   "100.0 GB requested by 2 fetch(es) ended, fetched bytes not known")
          and any("read stopped at" in n for n in notes), (why, notes))
    fold = SR._fold_sources([{"source": "a", "event": "fetch_started", "ts": "2026-10-01T00:00:00Z"},
                             {"source": "a", "event": "fetch_started", "ts": "2026-10-01T01:00:00Z"},
                             {"source": "a", "event": "fetch_failed", "ts": "2026-10-01T01:10:00Z"},
                             {"source": "a", "event": "fetched", "ts": "2026-10-01T01:10:00Z", "bytes": 7},
                             {"source": "b", "event": "fetch_started", "ts": "2026-10-01T02:00:00Z"},
                             {"source": "b", "event": "held", "ts": "2026-10-01T02:01:00Z"},
                             {"source": "c", "event": "fetch_started", "ts": "2026-10-01T03:00:00Z"}])
    check("the fold: a start followed by the next start is open, a start a closing event follows is not",
          fold["a"]["open_fetches"] == ["2026-10-01T00:00:00Z"]
          and fold["a"]["fetched_events"] == [["2026-10-01T01:10:00Z", 7]]
          and fold["b"]["open_fetches"] == [] and fold["c"]["open_fetches"] == ["2026-10-01T03:00:00Z"],
          {k: (v.get("open_fetches"), v.get("fetched_events")) for k, v in fold.items()})


def main():
    W.run_case("live", t_live)
    W.run_case("in_flight", t_in_flight)
    W.run_case("unknown_lab", t_unknown_lab)
    W.run_case("unknown_fold", t_unknown_cluster_fold)
    W.run_case("unknown_until_snapshot", t_unknown_until_snapshot)
    W.run_case("caps", t_caps)
    W.run_case("killed_lab", t_killed_lab)
    W.run_case("killed_cluster", t_killed_cluster)
    W.run_case("failed_end", t_failed_end)
    W.run_case("two_machines", t_two_machines)
    W.run_case("fold_missing", t_fold_missing)
    W.run_case("cluster_ledger_incomplete", t_cluster_ledger_incomplete)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
