#!/usr/bin/env python3
"""Continuation intake shards in the stream (docs/CONTINUOUS_LOOP.md,
amendment 2026-10-03).

Found live, 2026-10-03: zenodo_15808623 (230,899 boxed images in one 49.7 GB
zip) committed one intake batch of 40,000 (collect budgets.intake_max_images)
and deferred 190,899. The collector now takes them as continuation shards of
the same fetch record (collect.intake step 1b, tests/test_collect_intake.py);
the stream has to ask for them.

Pinned, in the stream world (the snapshot's intake summaries, the cluster
jobs):
  * intake with images deferred -> L17 admit of its batch -> the source is
    shard_pending (not admitted), recorded; DPIPE proposes L16I again under a
    new id; its batch is admitted; with nothing deferred the source is
    admitted. No failure is counted, no stop-loss step, no new fetch (D20),
    and the yield (zero-yield, D21) is judged once, on the complete source;
  * the order: while DR0's DATA item (L17 eval-hits) is due the next shard
    waits and eval-hits runs first; while an E1 arm waits for base v3 (L23V
    due or running) the next shard waits and L23V is proposed; meanwhile
    D29 never calls collection exhausted;
  * a continuation intake that failed: the source's failure and the lane's
    step are counted as for any job, and the source waits for its shard
    again (never a candidate D20 would fetch anew); the retry has a new id;
  * a source the stream admitted before shards existed (its summary records
    yield.images_deferred, no shard block) waits for its next shard after the
    next snapshot, and D21 never judges it on a part of its images;
  * a continuation intake that made no new batch: with the collector's ledger
    showing nothing deferred the source is complete (admitted, and D21 judges
    it); with images still deferred it made no progress and is closed with a
    card, never proposed again in a loop;
  * decisions rest on what a snapshot shows, never on what it lacks (review
    of 2026-10-03): snapshots whose summaries failed to read after a
    continuation intake leave its shard to be admitted once they are read
    again (no orphaned batch); an admit that completes in such a snapshot for
    a source recorded before shards existed is decided once the summary is
    back, and D21 does not judge it meanwhile;
  * a detour of the status never loses the shards: the collector's hold
    during a shard, then its release, and a person's reopening of a source
    closed part-way through them, all end in shard_pending, never a fetch;
  * failures count per shard (reset when a shard is admitted); a source
    closed after admitting target boxes is not a zero-yield source, and a
    stale zero-yield entry of it leaves the run at its first admitted boxes;
  * a shard that left as many images deferred as the one before it made no
    progress: closed with a card, no further intake;
  * the order with D20: while eval-hits is due no new source is fetched
    either (D20 waits for DR0's L17 item, as for R1), so eval-hits, then the
    shard, then a new source; a base v3 build that ended without a complete
    summary (unconfirmed) does not hold the shards.

Run:  python3 tests/test_stream_ap_shards.py
"""
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import DS, S, SR, World, check  # noqa: E402

SRC = "src_s"
CAND = {"id": SRC, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
        "bytes": 1e9, "images": 500, "expected_target_boxes": 800}


def shard_summary(w, batch, source, n, left, images=400, earlier=(), legacy=False):
    """collect.intake's summary.json of one shard: rows, the yield and, unless
    legacy (a batch committed before shards existed), the shard record."""
    w.intake(batch, source, images=images)
    p = w.inc / "intake" / batch / "summary.json"
    doc = json.loads(p.read_text())
    doc["yield"]["images_deferred"] = left
    if not legacy:
        doc["shard"] = {"n": n, "deferred_remaining": left, "earlier_batches": list(earlier), "candidates": images + left,
                        "tree": "reused" if n > 1 else "extracted"}
    p.write_text(json.dumps(doc))


def src(w, s=SRC):
    return (((w.state() or {}).get("sources") or {}).get(s)) or {}


def intakes(w, s=SRC):
    return [x for x in w.submits if "run_inc_collect.sh" in " ".join(x["argv"]) and "intake" in x["argv"]
            and s in x["argv"]]


def fetches(w, s=SRC):
    return [x for x in w.submits if "fetch" in x["argv"] and s in x["argv"]]


def admits(w):
    return [x["argv"][-1] for x in w.submits if "run_inc2_stream.sh" in " ".join(x["argv"]) and "admit" in x["argv"]
            and "--intake" in x["argv"]]


def eval_hits(w):
    return [x for x in w.submits if "run_inc2_stream.sh" in " ".join(x["argv"]) and "eval-hits" in x["argv"]]


def quiet(w):
    """No failure, no stop-loss, no hold, no pause."""
    st = w.state() or {}
    return (not [e for e in w.events("failed") if e.get("lever") in ("L16I", "L17")]
            and int(w.lane("DATA").get("fails") or 0) == 0 and not w.lane("DATA").get("hold")
            and not st.get("paused") and int(src(w).get("failures") or 0) == 0)


def tick(w, n=1):
    """Ticks in which every lab job ends at once (D20's discovery takes the free
    DATA lane while a shard waits: no open candidate)."""
    for _ in range(n):
        w.runner.finish(ok=True)
        w.tick(1)


def to_first_admit(w, left=150):
    """The source fetched, its first batch intaken with `left` deferred, and admitted."""
    w.queue(0)
    w.candidates([dict(CAND)])
    w.tick(3)
    w.job_done("inc_stream_collect_fetch_%s" % SRC)
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0001_%s" % SRC, SRC, 1, left)
    w.tick(2)
    w.job_done("inc_stream_admit_admit_i0001_%s" % SRC)
    w.tick(1)


def test_flow():
    print("intake with deferred images -> admit -> next shard -> admit -> complete")
    w = World("shards_flow", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w)
    s = src(w)
    ev = [e for e in w.events("source_status") if e.get("source") == SRC and e.get("status") == "shard_pending"]
    check("the first batch admitted with 150 images deferred: shard_pending, recorded, not admitted",
          s.get("status") == "shard_pending" and s.get("admitted_batches") == ["i0001_%s" % SRC]
          and s.get("intake_deferred") == 150 and ev and ev[-1].get("deferred") == 150, (s, ev[-1:]))
    check("  its yield is not judged on a part of the source (no zero-yield record yet)",
          not s.get("yield_recorded"), s)
    w.tick(2)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L16I"]
    check("DPIPE proposes L16I again, the next shard, under a new id", len(intakes(w)) == 2 and len(pro) == 2
          and len({e.get("proposal_id") for e in pro}) == 2, [e.get("proposal_id") for e in pro])
    check("  D20 never fetches the source again", len(fetches(w)) == 1, [x["argv"] for x in fetches(w)])
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0002_%s" % SRC, SRC, 2, 0, images=150, earlier=["i0001_%s" % SRC])
    w.tick(2)
    check("the shard's batch is admitted next", admits(w) == ["i0001_%s" % SRC, "i0002_%s" % SRC], admits(w))
    w.job_done("inc_stream_admit_admit_i0002_%s" % SRC)
    w.step1_status(per_source={SRC: {"target_boxes_admitted": 900}})
    w.tick(2)
    s = src(w)
    check("nothing deferred: the source is admitted, its yield recorded once (900 boxes, the whole source)",
          s.get("status") == "admitted" and s.get("intake_deferred") == 0 and s.get("yield_recorded")
          and s.get("admitted_batches") == ["i0001_%s" % SRC, "i0002_%s" % SRC]
          and s.get("admitted_target_boxes") == 900, s)
    check("  no failure counted, no stop-loss step, no hold", quiet(w), (w.lane("DATA"), s))
    w.tick(3)
    check("  and no further intake of the source", len(intakes(w)) == 2 and len(fetches(w)) == 1,
          [x["argv"] for x in w.submits[-4:]])


def test_order_eval_hits():
    print("the next shard waits while DR0's eval-hits is due")
    w = World("shards_evalhits", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    tick(w, 3)
    w.job_done("inc_stream_collect_fetch_%s" % SRC)
    tick(w, 2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0001_%s" % SRC, SRC, 1, 150)
    # a Step 1 batch whose dHash hits no record weighs: DR0 proposes L17 eval-hits on the DATA lane
    w.step1_status(extra={"eval_hits": {"due": ["s1b0007"]}})
    tick(w, 2)
    check("the first batch's admit (DPIPE) still comes before eval-hits, as before",
          admits(w) == ["i0001_%s" % SRC] and not eval_hits(w), (admits(w), len(eval_hits(w))))
    w.job_done("inc_stream_admit_admit_i0001_%s" % SRC)
    tick(w, 4)
    dp = DS.by_id(json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(W.NAME).read_text())["diagnoses"])
    check("shard_pending, eval-hits due: L17 eval-hits takes the DATA lane, not the next shard",
          src(w).get("status") == "shard_pending" and len(eval_hits(w)) == 1 and len(intakes(w)) == 1
          and "eval-hits" in (dp.get("DPIPE") or {}).get("summary", ""), ((dp.get("DPIPE") or {}).get("summary"),
                                                                           len(eval_hits(w)), len(intakes(w))))
    w.job_done("inc_stream_admit_evalhits")
    w.step1_status(extra={"eval_hits": {"due": []}})
    tick(w, 3)
    check("  eval-hits done: then the next shard", len(intakes(w)) == 2 and quiet(w), len(intakes(w)))


def _e1_world(tag):
    """R0 complete; E1's base v3 (splits v3) and its arms not built."""
    w = World(tag, floors=(5.0, 10.0))
    w.ready_r0()
    (w.inc / "splits" / "v3" / "summary.json").unlink()
    for b in w.dom["baselines"]["items"]:
        if b.get("requires") == "base3":
            shutil.rmtree(str(w.inc / b["exp"]))
    return w


def test_order_base3():
    print("the next shard waits while an E1 arm waits for base v3")
    w = _e1_world("shards_base3")
    to_first_admit(w)
    tick(w, 2)
    v23 = [e for e in w.events("proposed") if e.get("lever") == "L23V"]
    dp = DS.by_id(json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(W.NAME).read_text())["diagnoses"])
    check("shard_pending while base v3 is not built: L23V is proposed, the next shard waits",
          src(w).get("status") == "shard_pending" and len(v23) == 1 and len(intakes(w)) == 1
          and "base v3" in (dp.get("DPIPE") or {}).get("summary", ""),
          ((dp.get("DPIPE") or {}).get("summary"), len(v23), len(intakes(w))))
    tick(w, 2)
    check("  still waiting while L23V runs, and no new fetch of the waiting source (D20)", len(intakes(w)) == 1
          and w.state()["stage"]["r0"].get("base3") == "running" and len(fetches(w)) == 1,
          (w.state()["stage"]["r0"].get("base3"), len(fetches(w))))
    dp = DS.by_id(json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(W.NAME).read_text())["diagnoses"])
    check("  and collection is not 'exhausted' meanwhile (D29 silent: a source waits for its next shard), "
          "though discovery ran and found nothing new", w.state()["discover"].get("runs")
          and not (dp.get("D29") or {}).get("fired") and "next intake shard" in (dp.get("D29") or {}).get("summary", ""),
          ((dp.get("D29") or {}).get("summary"), w.state()["discover"]))
    w.job_done("inc_build_base3_v3")
    w.base3_summary()
    tick(w, 3)
    check("base v3 built: the next shard runs (E1's arms are built from splits v3 alongside)",
          len(intakes(w)) == 2 and quiet(w), len(intakes(w)))


def test_failure():
    print("a continuation intake that failed waits for its shard again")
    w = World("shards_fail", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w)
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal="[collect] intake: out of memory")
    w.tick(1)
    s = src(w)
    check("the failure is counted (the source's and the lane's), and the source waits for its shard again, "
          "never a candidate", s.get("status") == "shard_pending" and int(s.get("failures") or 0) == 1
          and int(w.lane("DATA").get("fails") or 0) == 1, (s, w.lane("DATA")))
    w.tick(2)
    pro = [e.get("proposal_id") for e in w.events("proposed") if e.get("lever") == "L16I"]
    check("  the shard is proposed again under a new id; no new fetch", len(intakes(w)) == 3
          and len(set(pro)) == 3 and len(fetches(w)) == 1, pro)


def test_legacy():
    print("a source admitted before shards existed waits for its next shard")
    w = World("shards_legacy", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    shard_summary(w, "i0004_src_l", "src_l", 1, 190899, images=39958, legacy=True)
    w.tick(1)                                  # the snapshot holds its summary.json, as the live one has for hours
    paths = S.StreamPaths(str(w.lab), "weed")
    st = json.loads(paths.state(W.NAME).read_text())
    # what the stream recorded before: admitted, yield judged on the first batch, 40 GB fetched
    st["sources"]["src_l"] = {"status": "admitted", "batch": "i0004_src_l", "batches": ["i0004_src_l"],
                              "admit_done_utc": W.utc(w.t[0]), "yield_recorded": True, "bytes": 4.97e10,
                              "su": 2.0, "admitted_target_boxes": 50}
    paths.state(W.NAME).write_text(json.dumps(st))
    w.step1_status(per_source={"src_l": {"target_boxes_admitted": 50}})
    w.tick(1)
    d21 = DS.by_id(json.loads(paths.diagnoses(W.NAME).read_text())["diagnoses"]).get("D21") or {}
    check("D21 does not judge it on its first batch (1 box per GB would close it)", not d21.get("fired")
          and src(w, "src_l").get("status") != "closed", (d21.get("summary"), src(w, "src_l")))
    tick(w, 4)
    s = src(w, "src_l")
    check("after a snapshot: its batch admitted, 190,899 deferred, shard_pending; the next shard is proposed",
          s.get("admitted_batches") == ["i0004_src_l"] and s.get("intake_deferred") == 190899
          and len(intakes(w, "src_l")) == 1, (s, [x["argv"] for x in w.submits]))


def diag(w):
    return DS.by_id(json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(W.NAME).read_text())["diagnoses"])


def put_state(w, fn):
    """The stream's state edited between ticks (what an earlier code wrote, or a counter set)."""
    paths = S.StreamPaths(str(w.lab), "weed")
    st = json.loads(paths.state(W.NAME).read_text())
    fn(st)
    paths.state(W.NAME).write_text(json.dumps(st))


def cards(w, word):
    return [c for c in w.state().get("cards") or [] if word in str(c.get("title"))]


def test_no_new_batch():
    print("a continuation intake that made no new batch: complete, or no progress")
    for left, tag in ((0, "noop"), (150, "noop_left")):
        w = World("shards_%s" % tag, floors=(5.0, 10.0))
        w.ready_r0()
        to_first_admit(w)
        w.tick(2)
        # the collector's ledger: the source's one batch, and what its last intake left deferred
        w.sources([{"source": SRC, "event": "intaken", "batch": "i0001_%s" % SRC,
                    "yield": {"images_deferred": left}}])
        w.job_done("inc_stream_collect_intake_%s" % SRC)
        w.tick(4)
        s = src(w)
        if not left:
            ev = [e for e in w.events("source_status") if e.get("source") == SRC and e.get("status") == "admitted"]
            check("the collector's ledger shows nothing deferred and no new batch: admitted, recorded, no further "
                  "intake", s.get("status") == "admitted" and s.get("shards_done") == "i0001_%s" % SRC
                  and s.get("intake_deferred") == 0 and ev and len(intakes(w)) == 2 and quiet(w),
                  (s, ev[-1:], len(intakes(w))))

            # the collector's ledger: 50 GB fetched for 10 admitted target boxes
            w.sources([{"source": SRC, "event": "fetched", "bytes": 50000000000}])
            w.step1_status(per_source={SRC: {"target_boxes_admitted": 10}})
            tick(w, 3)
            check("  D21 judges the complete source (10 boxes from 50 GB: below the floors, closed)",
                  src(w).get("yield_recorded") and src(w).get("status") == "closed"
                  and "boxes/GB" in str(src(w).get("closed_reason")), src(w))
        else:
            check("the collector's ledger shows 150 images deferred and no new batch: the shard made no progress; "
                  "closed with a card, no further intake", s.get("status") == "closed" and cards(w, "no progress")
                  and len(intakes(w)) == 2 and len(fetches(w)) == 1, (s, len(intakes(w))))


def test_summary_gap():
    print("snapshots whose summaries failed after a continuation intake: its shard is admitted once they read again")
    w = World("shards_gap", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w)
    w.tick(2)
    real = SR.stream_summary

    def boom(*a, **k):
        raise OSError("transient Lustre error (injected)")
    SR.stream_summary = boom
    try:
        shard_summary(w, "i0002_%s" % SRC, SRC, 2, 0, images=150, earlier=["i0001_%s" % SRC])
        w.sources([{"source": SRC, "event": "intaken", "batch": "i0001_%s" % SRC, "yield": {"images_deferred": 150}},
                   {"source": SRC, "event": "intaken", "batch": "i0002_%s" % SRC, "yield": {"images_deferred": 0}}])
        w.job_done("inc_stream_collect_intake_%s" % SRC)
        tick(w, 4)
        s = src(w)
        check("while the summaries do not read, nothing is decided on their absence (intaken, not 'admitted')",
              s.get("status") == "intaken" and not s.get("shards_done"), s)
    finally:
        SR.stream_summary = real
    # (a snapshot without the summary shows no placement either: DR0 proposed the network probe, which ends)
    w.job_done("inc_stream_collect_probe")
    tick(w, 2)
    check("  read again: the shard's batch is admitted (no orphaned batch)",
          admits(w) == ["i0001_%s" % SRC, "i0002_%s" % SRC], admits(w))
    w.job_done("inc_stream_admit_admit_i0002_%s" % SRC)
    tick(w, 2)
    s = src(w)
    check("  and the source is complete", s.get("status") == "admitted" and s.get("intake_deferred") == 0
          and s.get("admitted_batches") == ["i0001_%s" % SRC, "i0002_%s" % SRC] and quiet(w), s)


def test_legacy_gap():
    print("an admit of a source recorded before shards existed, completing while the summaries do not read")
    w = World("shards_legacy_gap", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    w.tick(3)
    w.job_done("inc_stream_collect_fetch_%s" % SRC)
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0001_%s" % SRC, SRC, 1, 190899, images=400, legacy=True)
    w.tick(2)
    check("(the admit of its first batch is submitted)", admits(w) == ["i0001_%s" % SRC], admits(w))

    def as_before(st):
        # what the stream recorded before shards existed: no deferred count, no admitted batches, no shard log
        s = st["sources"][SRC]
        for k in ("intake_deferred", "intake_shard", "shard_log", "admitted_batches"):
            s.pop(k, None)
        s["bytes"] = 4.97e10
    put_state(w, as_before)
    w.step1_status(per_source={SRC: {"target_boxes_admitted": 0}})
    real = SR.stream_summary

    def boom(*a, **k):
        raise OSError("transient Lustre error (injected)")
    SR.stream_summary = boom
    try:
        w.job_done("inc_stream_admit_admit_i0001_%s" % SRC)
        tick(w, 3)
        s = src(w)
        d21 = diag(w).get("D21") or {}
        check("its admit completes in a snapshot without its summary: admitted as before, its shards undecided, and "
              "D21 does not judge it (0 boxes from 49.7 GB would close it)",
              s.get("status") == "admitted" and not isinstance(s.get("admitted_batches"), list)
              and not d21.get("fired"), (s, d21.get("summary")))
    finally:
        SR.stream_summary = real
    w.job_done("inc_stream_collect_probe")
    tick(w, 3)
    s = src(w)
    check("  the summary read again: 190,899 deferred, shard_pending, the next shard proposed (no new fetch)",
          s.get("status") in ("shard_pending", "intaken") and s.get("admitted_batches") == ["i0001_%s" % SRC]
          and len(intakes(w)) == 2 and len(fetches(w)) == 1, (s, len(intakes(w))))


def test_detours():
    print("the collector's hold and release during a shard, and a person's reopening, end in shard_pending")
    w = World("shards_hold", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w, left=300)
    w.tick(2)
    w.sources([{"source": SRC, "event": "held", "reason": "copy_scan", "codes": ["copy_scan"]}])
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal="[collect] intake: held (copy_scan)")
    w.tick(2)
    check("the collector holds the source during its shard: held (its hold is kept, not overwritten)",
          src(w).get("status") == "held" and len(intakes(w)) == 2, src(w))
    w.advance(60)
    w.sources([{"source": SRC, "event": "released", "reason": "a person cleared the copy scan"}])
    tick(w, 3)
    s = src(w)
    check("  released: shard_pending, its next shard proposed under a new id; never fetched again",
          s.get("status") in ("shard_pending", "intaken") and len(intakes(w)) == 3 and len(fetches(w)) == 1
          and len({e.get("proposal_id") for e in w.events("proposed") if e.get("lever") == "L16I"}) == 3,
          (s, len(intakes(w)), len(fetches(w))))
    w2 = World("shards_reopen", floors=(5.0, 10.0))
    w2.ready_r0()
    to_first_admit(w2, left=300)

    def closed(st):
        st["sources"][SRC].update(status="closed", closed_reason="3 failed attempts (7.5)", failures=3,
                                  closed_utc=W.utc(w2.t[0]))
    put_state(w2, closed)
    w2.advance(60)
    S.configure_stream(W.NAME, W.OWNER, cfg_hooks=w2.hooks, lab_repo=str(w2.lab), reopen_source=SRC,
                       reopen_why="node failures fixed", clock=w2.clock)
    tick(w2, 3)
    s = src(w2)
    check("a person reopens a source closed part-way through its shards: its shards resume (a new id), no fetch",
          s.get("status") in ("shard_pending", "intaken") and int(s.get("failures") or 0) == 0
          and len(intakes(w2)) == 2 and len(fetches(w2)) == 1, (s, len(intakes(w2)), len(fetches(w2))))
    # whatever its status says, a source part-way through its shards is never a candidate D20 fetches
    w3 = World("shards_guard", floors=(5.0, 10.0))
    w3.ready_r0()
    to_first_admit(w3, left=300)
    w3.job_done("inc_stream_collect_intake_%s" % SRC)

    def candidate(st):
        st["sources"][SRC]["status"] = "candidate"
        st["lanes"]["DATA"].update(item=None, phase="IDLE")
    put_state(w3, candidate)
    tick(w3, 3)
    d20 = diag(w3).get("D20") or {}
    check("  a source part-way through its shards whose status reads 'candidate': D20 refuses it (precheck), no "
          "fetch", len(fetches(w3)) == 1 and "L16 on %s" % SRC not in str(d20.get("summary")),
          (d20.get("summary"), len(fetches(w3))))


def test_failures_per_shard():
    print("failures count per shard; a source that admitted boxes is not a zero-yield failure")
    w = World("shards_failcount", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w, left=300)
    w.step1_status(per_source={SRC: {"target_boxes_admitted": 600}})
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal="[collect] intake: node fail")
    w.tick(3)
    check("(shard 2 failed once, then proposed again)", int(src(w).get("failures") or 0) == 1
          and len(intakes(w)) == 3, src(w))
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0002_%s" % SRC, SRC, 2, 150, images=150, earlier=["i0001_%s" % SRC])
    w.tick(2)
    w.job_done("inc_stream_admit_admit_i0002_%s" % SRC)
    w.tick(2)
    check("shard 2 admitted: the source's failure count restarts (0), shard 3 is next",
          src(w).get("status") == "shard_pending" and int(src(w).get("failures") or 0) == 0, src(w))
    w.tick(1)

    def two(st):
        # two failures in this shard, and two zero-yield sources just before it in the run
        st["sources"][SRC]["failures"] = 2
        st["zero_run"] = [{"source": z, "how": "zero", "reasons": {}, "utc": W.utc(w.t[0])} for z in ("z1", "z2")]
    put_state(w, two)
    w.job_done("inc_stream_collect_intake_%s" % SRC, state="FAILED", refusal="[collect] intake: node fail")
    w.tick(2)
    st = w.state()
    check("a third failure in one shard closes it, but a source whose shards admitted 600 target boxes is not "
          "counted as a zero-yield failure (a third one would hold the DATA lane)", src(w).get("status") == "closed"
          and not [r for r in st.get("zero_run") or [] if r.get("source") == SRC]
          and "zero admitted" not in str(w.lane("DATA").get("hold")), (src(w), st.get("zero_run"), w.lane("DATA")))


def test_zero_run_stale():
    print("a stale zero-yield entry of the source leaves the run at its first admitted boxes")
    w = World("shards_zero_run", floors=(5.0, 10.0))
    w.ready_r0()
    w.tick(1)

    def stale(st):
        st["zero_run"] = [{"source": SRC, "how": "failed", "reasons": {}, "utc": W.utc(w.t[0])}]
    put_state(w, stale)
    to_first_admit(w, left=300)
    w.step1_status(per_source={SRC: {"target_boxes_admitted": 600}})
    w.tick(2)
    check("its first shard admitted 600 target boxes: its 'failed' entry leaves the zero-yield run (the run "
          "would otherwise hold the DATA lane two sources sooner, for as long as the shards take)",
          src(w).get("status") in ("shard_pending", "intaken")
          and not [r for r in w.state().get("zero_run") or [] if r.get("source") == SRC]
          and w.events("zero_run_cleared"), w.state().get("zero_run"))


def test_no_progress():
    print("a shard that left as many images deferred as the one before it")
    w = World("shards_stall", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w)
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0002_%s" % SRC, SRC, 2, 150, images=0, earlier=["i0001_%s" % SRC])
    w.tick(2)
    w.job_done("inc_stream_admit_admit_i0002_%s" % SRC)
    w.tick(4)
    s = src(w)
    check("no progress: closed with a card, no third intake, no fetch", s.get("status") == "closed"
          and cards(w, "no progress") and len(intakes(w)) == 2 and len(fetches(w)) == 1, (s, len(intakes(w))))


def test_order_d20():
    print("while eval-hits is due no new source is fetched either; then the shard; then a new source")
    w = World("shards_d20", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    tick(w, 3)
    w.job_done("inc_stream_collect_fetch_%s" % SRC)
    tick(w, 2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0001_%s" % SRC, SRC, 1, 150)
    w.step1_status(extra={"eval_hits": {"due": ["s1b0007"]}})
    tick(w, 2)
    w.candidates([dict(CAND), dict(CAND, id="src_b", expected_target_boxes=500)])
    w.job_done("inc_stream_admit_admit_i0001_%s" % SRC)
    tick(w, 4)
    check("Q low and an open candidate, eval-hits due: eval-hits takes the DATA lane, neither a fetch (D20 waits "
          "for DR0's L17 item) nor the shard", len(eval_hits(w)) == 1 and not fetches(w, "src_b")
          and len(intakes(w)) == 1, (len(eval_hits(w)), len(fetches(w, "src_b")), len(intakes(w))))
    w.job_done("inc_stream_admit_evalhits")
    w.step1_status(extra={"eval_hits": {"due": []}})
    tick(w, 2)
    check("  eval-hits done: the shard comes before the new source", len(intakes(w)) == 2
          and not fetches(w, "src_b"), (len(intakes(w)), len(fetches(w, "src_b"))))
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    shard_summary(w, "i0002_%s" % SRC, SRC, 2, 0, images=150, earlier=["i0001_%s" % SRC])
    tick(w, 2)
    w.job_done("inc_stream_admit_admit_i0002_%s" % SRC)
    tick(w, 3)
    check("  the source complete: then the new source is fetched", src(w).get("status") == "admitted"
          and len(fetches(w, "src_b")) == 1 and quiet(w), (src(w).get("status"), len(fetches(w, "src_b"))))


def test_base3_unconfirmed():
    print("a base v3 build that ended without a complete summary does not hold the shards")
    w = _e1_world("shards_base3_unconfirmed")
    to_first_admit(w)
    tick(w, 3)
    check("(L23V proposed and running; the shard waits)", w.state()["stage"]["r0"].get("base3") == "running"
          and len(intakes(w)) == 1, w.state()["stage"]["r0"].get("base3"))
    w.job_done("inc_build_base3_v3")
    tick(w, 3)
    check("the build ended without a complete summary (unconfirmed): the next shard runs, it does not wait for "
          "a build nobody proposes again", w.state()["stage"].get("base3") == "unconfirmed"
          or len(intakes(w)) == 2, (w.state()["stage"].get("base3"), len(intakes(w))))
    check("  the next shard is proposed", len(intakes(w)) == 2, (len(intakes(w)),
                                                                  (diag(w).get("DPIPE") or {}).get("summary")))


def test_units():
    print("deferred_left and D21 on rows of the stream's state, without a world")
    from test_stream_ap_world import E, LS
    dom = LS.load_domain("weed")
    th = LS.load_thresholds()                       # D21's floors: 5 boxes per GB, 10 per SU

    def view(sources, arts=None):
        ctx = {"now_utc": "2026-10-03T00:00:00Z", "sid": dom["sid"], "sources": sources}
        return DS.View(E.from_texts({k: json.dumps(v) for k, v in (arts or {}).items()}, dom["sid"], context=ctx),
                       dom, th)
    row = {"status": "admitted", "yield_recorded": True, "batch": "i0004_x", "batches": ["i0004_x"],
           "bytes": 4.97e10, "admitted_target_boxes": 0}
    summ = {"source": "x", "batch": "i0004_x", "yield": {"images_deferred": 190899}}
    check("deferred_left: unknown (None) without the state's count or the summary; read from the summary; 0 once "
          "the shards are done (whatever the count), and 0 for a source with no intake batch",
          DS.deferred_left(view({"x": row}), row) is None
          and DS.deferred_left(view({"x": row}, {"intake/i0004_x/summary.json": summ}), row) == 190899
          and DS.deferred_left(view({}), dict(row, intake_deferred=150, shards_done="i0004_x")) == 0
          and DS.deferred_left(view({}), {"status": "admitted"}) == 0)
    d21 = DS.d21(view({"x": row}))
    check("D21 does not judge a source whose deferred count is unknown (0 boxes from 49.7 GB would close it)",
          not d21.get("fired"), d21.get("summary"))
    d21 = DS.d21(view({"x": dict(row, intake_deferred=150, shards_done="i0004_x")}))
    check("  and judges one whose shards are done (closed: 0 boxes per GB)", d21.get("fired")
          and "x (" in d21.get("summary", ""), d21.get("summary"))


def main():
    for fn in (test_flow, test_order_eval_hits, test_order_base3, test_failure, test_legacy, test_no_new_batch,
               test_summary_gap, test_legacy_gap, test_detours, test_failures_per_shard, test_zero_run_stale,
               test_no_progress, test_order_d20, test_base3_unconfirmed, test_units):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
