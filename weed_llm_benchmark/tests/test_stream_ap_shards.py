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
  * a continuation intake that made no new batch ends the source (admitted)
    at the next snapshot, never a loop.

Run:  python3 tests/test_stream_ap_shards.py
"""
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import DS, S, World, check  # noqa: E402

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


def test_no_new_batch():
    print("a continuation intake that made no new batch ends the source")
    w = World("shards_noop", floors=(5.0, 10.0))
    w.ready_r0()
    to_first_admit(w)
    w.tick(2)
    w.job_done("inc_stream_collect_intake_%s" % SRC)
    w.tick(4)
    s = src(w)
    ev = [e for e in w.events("source_status") if e.get("source") == SRC and e.get("status") == "admitted"]
    check("no new batch at the next snapshot: admitted, recorded, no further intake", s.get("status") == "admitted"
          and s.get("shards_done") == "i0001_%s" % SRC and ev and len(intakes(w)) == 2 and quiet(w),
          (s, ev[-1:], len(intakes(w))))


def main():
    for fn in (test_flow, test_order_eval_hits, test_order_base3, test_failure, test_legacy, test_no_new_batch):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
