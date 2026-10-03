#!/usr/bin/env python3
"""An intake refused for class names runs the names round, never a failure.

Found before it happened, 2026-10-03: zenodo_15808623 (SIU Weed Growth
Stage, 203,567 images) names its 174 classes "<EPPO>_week_<n>". With the EPPO
prefix (collect.classmap) its four target species map, but 12 non-target
binomials are not in the lab's names layer, so the cluster intake refuses
with names_pending (collect.intake, exit 2). The job ended FAILED: the stream
charged it as a failed step, retried the same refusal, and the second failure
in a row held the DATA lane (stop-loss); the collector's 'held' made the
source 'held', which nothing proposes L26 for. The intake now starts the
names round instead.

Pinned, in the stream world (the snapshot's fold of the collector ledger,
the lab runner, the cluster jobs):
  * the snapshot's fold carries a source's pending names and their time, and
    a later event of the source clears them;
  * an L16I job that ended on names_pending is neither the source's failure
    nor a step toward the lane's stop-loss; the source is names_pending with
    the names, recorded (intake_names_pending), and the intake's id retired;
  * DPIPE proposes L26 for it, whose lab hook writes the names to the lab's
    intake/work/<source>/pending_names.json (what `collect names` reads);
    then L16S of the names layer (--names); then the source is fetched again
    and L16I runs under a new id;
  * D20 never fetches a names_pending source again;
  * still pending after NAMES_ROUNDS_MAX rounds: held, with a card for a
    person, and no further L26.

Run:  python3 tests/test_stream_ap_names_round.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import S, World, check  # noqa: E402

NAMES = ["Setaria faberi", "Abutilon theophrasti"]
CAND = {"id": "src_n", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
        "bytes": 1e9, "images": 500, "expected_target_boxes": 800}


def intakes(w):
    return [s for s in w.submits if "run_inc_collect.sh" in " ".join(s["argv"]) and "intake" in s["argv"]
            and "src_n" in s["argv"]]


def lab(w, word):
    return [x for x in w.runner.launched if word in " ".join(x["argv"])]


def src(w):
    return (((w.state() or {}).get("sources") or {}).get("src_n")) or {}


def test_fold():
    print("the snapshot's fold carries the pending names until a later event of the source")
    from weed_optimizer_framework.tools.inc_autopilot import stream_remote as SR
    rows = [{"source": "a", "event": "fetched", "ts": "2026-10-03T01:00:00Z", "bytes": 5},
            {"source": "a", "event": "held", "ts": "2026-10-03T02:00:00Z", "reason": "names_pending",
             "codes": ["names_pending"], "stage": "intake", "names": NAMES},
            {"source": "b", "event": "held", "ts": "2026-10-03T02:00:00Z", "reason": "names_pending", "stage": "intake",
             "names": ["X y"]},
            {"source": "b", "event": "intaken", "ts": "2026-10-03T03:00:00Z", "batch": "i0001_b"},
            {"source": "c", "event": "held", "ts": "2026-10-03T02:00:00Z", "reason": "licence_unresolved"},
            {"source": "d", "event": "held", "ts": "2026-10-03T02:00:00Z", "reason": "names_pending", "stage": "fetch"}]
    f = SR._fold_sources(rows)
    check("a held names_pending event of an intake: its names and time", f["a"]["pending_names"] == NAMES
          and f["a"]["pending_ts"] == "2026-10-03T02:00:00Z" and f["a"]["status"] == "held", f["a"])
    check("  cleared by a later intake of the source; absent for another hold and for a fetch's names hold",
          f["b"]["pending_names"] is None and f["b"]["pending_ts"] is None and f["c"]["pending_names"] is None
          and f["d"]["pending_names"] is None, (f["b"], f["c"], f["d"]))


def test_round():
    print("in the stream world, an intake refused for class names runs the names round")
    w = World("names_round", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    w.tick(3)
    w.job_done("inc_stream_collect_fetch_src_n")
    w.tick(2)
    check("the fetch is done and the intake submitted", len(intakes(w)) == 1, [s["argv"] for s in w.submits])
    w.sources([{"source": "src_n", "event": "held", "reason": "names_pending", "codes": ["names_pending"],
                "stage": "intake", "names": NAMES}])
    w.job_done("inc_stream_collect_intake_src_n", state="FAILED",
               refusal="[collect] intake refused: 2 class name(s) of src_n are not in the taxonomy cache or the "
                       "names layer; run `collect names --source src_n` (lever L26) first")
    w.tick(1)
    s = src(w)
    ev = [e for e in w.events("intake_names_pending") if e.get("source") == "src_n"]
    failed = [e for e in w.events("failed") if e.get("lever") == "L16I"]
    check("the intake ended on names_pending: recorded, not a failure", ev and not failed
          and ev[-1].get("names") == NAMES, (ev[-1:], failed[-1:]))
    check("  the source is names_pending with its names; no failure, no stop-loss step, no hold",
          s.get("status") == "names_pending" and s.get("pending_names") == NAMES
          and int(s.get("failures") or 0) == 0 and int(w.lane("DATA").get("fails") or 0) == 0
          and not w.lane("DATA").get("hold") and not (w.state() or {}).get("paused"), (s, w.lane("DATA")))
    w.tick(2)
    l26 = lab(w, "collect names")
    check("DPIPE -> L26 on the lab for that source", len(l26) == 1 and "src_n" in l26[0]["argv"],
          [x["argv"] for x in w.runner.launched])
    pf = w.lab / "results" / "framework" / "inc" / "intake" / "work" / "src_n" / "pending_names.json"
    doc = json.loads(pf.read_text()) if pf.is_file() else {}
    check("  its lab hook wrote the names where `collect names` reads them", doc.get("names") == NAMES
          and doc.get("source") == "src_n", (str(pf), doc))
    check("  D20 does not fetch the source again", len([s for s in w.submits if "fetch" in s["argv"]
                                                        and "src_n" in s["argv"]]) == 1)
    w.runner.finish(ok=True)
    w.tick(3)
    s = src(w)
    syncs = [x for x in lab(w, "lab-sync") if "--names" in x["argv"] and "src_n" in x["argv"]]
    check("L26 done -> L16S of the names layer (--names), then the source is fetched again",
          len(syncs) == 1 and s.get("names_rounds") == 1 and s.get("names_synced"), (s, [x["argv"] for x in syncs]))
    check("  and the intake runs again under a new id", len(intakes(w)) == 2, [s_["argv"] for s_ in w.submits])
    pro = [e for e in w.events("proposed") if e.get("lever") == "L16I"]
    check("  (two L16I proposals with different ids)", len({e.get("proposal_id") or e.get("id") for e in pro}) >= 2,
          [e.get("proposal_id") for e in pro])
    # a second refusal: still pending after the round -> a person
    w.advance(60)
    w.sources([{"source": "src_n", "event": "held", "reason": "names_pending", "codes": ["names_pending"],
                "stage": "intake", "names": NAMES[:1]}])
    w.job_done("inc_stream_collect_intake_src_n", state="FAILED", refusal="[collect] intake refused: names")
    w.tick(3)
    s = src(w)
    cards = [c for c in (w.state() or {}).get("cards") or [] if "unresolved after L26" in (c.get("title") or "")]
    check("still pending after the round: held with a card, and no second L26",
          s.get("status") == "held" and cards and len(lab(w, "collect names")) == 1
          and int(s.get("failures") or 0) == 0, (s, cards[-1:], len(lab(w, "collect names"))))


def test_review_cases():
    print("the round's edge cases")
    w = World("names_round_edges", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    w.tick(3)
    w.job_done("inc_stream_collect_fetch_src_n")
    w.tick(2)
    # the job's refusal line arrives before the collector's ledger is folded
    w.job_done("inc_stream_collect_intake_src_n", state="FAILED",
               refusal="[collect] intake refused: 2 class name(s) of src_n are not in the taxonomy cache or the "
                       "names layer; run `collect names --source src_n` (lever L26) first")
    w.tick(1)
    s = src(w)
    check("a refusal line without the fold: not a failure; the source waits in the round, no L26 without names",
          s.get("status") == "names_pending" and int(s.get("failures") or 0) == 0
          and int(w.lane("DATA").get("fails") or 0) == 0 and not lab(w, "collect names"), (s, w.lane("DATA")))
    w.sources([{"source": "src_n", "event": "held", "reason": "names_pending", "codes": ["names_pending"],
                "stage": "intake", "names": NAMES}])
    w.tick(1)                      # the fold brings the names; the free lane took D20's discovery (L15) meanwhile
    w.runner.finish(ok=True)       # which ends
    w.tick(2)
    check("  the fold brings the names: L26", src(w).get("pending_names") == NAMES and len(lab(w, "collect names")) == 1,
          src(w))
    # an L26 the authority answered only in part runs again before the round counts
    w.runner.finish(ok=True, tail='[collect] names: {"errors": 1, "names": 1, "source": "src_n", "status": "partial"}')
    w.tick(2)
    s = src(w)
    check("a partial L26 runs again under a new id; the round is not counted yet",
          len(lab(w, "collect names")) == 2 and not s.get("names_rounds") and s.get("names_partial") == 1
          and w.events("names_partial"), (s, len(lab(w, "collect names"))))
    w.runner.finish(ok=True, tail='[collect] names: {"errors": 0, "names": 2, "source": "src_n", "status": "done"}')
    w.tick(3)
    check("  a complete one counts the round and the intake runs again", src(w).get("names_rounds") == 1
          and len(intakes(w)) == 2, src(w))
    # still pending: held; then a person maps the names and reopens it
    w.advance(60)
    w.sources([{"source": "src_n", "event": "held", "reason": "names_pending", "codes": ["names_pending"],
                "stage": "intake", "names": NAMES[:1]}])
    w.job_done("inc_stream_collect_intake_src_n", state="FAILED", refusal="[collect] intake refused: (lever L26)")
    w.tick(2)
    check("held after its round", src(w).get("status") == "held" and src(w).get("names_held"), src(w))
    w.advance(60)
    S.configure_stream(W.NAME, W.OWNER, cfg_hooks=w.hooks, lab_repo=str(w.lab), reopen_source="src_n",
                       reopen_why="a card table maps its names", clock=w.clock)
    w.tick(3)
    s = src(w)
    check("a person's reopen of a source held after its names round: fetched with its rounds reset, intake again",
          s.get("names_rounds") == 0 and not s.get("names_held") and len(intakes(w)) == 3
          and [e for e in w.events("source_reopened") if e.get("source") == "src_n"], (s, len(intakes(w))))


def test_prefetch_l26_ids():
    print("a source whose names L26 resolved before its fetch: the round's L26 and L16S get new ids")
    w = World("names_round_ids", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.candidates([dict(CAND)])
    w.tick(3)
    paths = S.StreamPaths(str(w.lab), w.domain)
    st = S._read_json(paths.state(W.NAME))
    # what a pre-fetch round left: its L26 and L16S --names ran under attempt 0
    for lever, params in (("L26", {"source": "src_n", "out": str(paths.names_dir()) + "/"}),
                          ("L16S", {"source": "src_n", "names": 1})):
        p = S.LS.proposal(W.NAME, lever, params, attempt=0)
        st.setdefault("prefetch_ids", []).append(p["id"])
    S._write_json(paths.state(W.NAME), st)
    w.job_done("inc_stream_collect_fetch_src_n")
    w.tick(2)
    w.sources([{"source": "src_n", "event": "held", "reason": "names_pending", "codes": ["names_pending"],
                "stage": "intake", "names": NAMES}])
    w.job_done("inc_stream_collect_intake_src_n", state="FAILED", refusal="x (lever L26) first")
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L26"]
    ids = (S._read_json(paths.state(W.NAME)) or {}).get("prefetch_ids") or []
    check("the round's L26 has an id no earlier L26 of the source had", pro and pro[-1].get("proposal_id") not in ids
          and pro[-1].get("attempt", 1) >= 1, ([e.get("proposal_id") for e in pro], ids))


def main():
    test_fold()
    test_round()
    test_review_cases()
    test_prefetch_l26_ids()
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
