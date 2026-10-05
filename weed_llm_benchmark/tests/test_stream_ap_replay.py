#!/usr/bin/env python3
"""Stream-mode replay and scenario cases S1-S29 (docs/CONTINUOUS_LOOP.md 6.8;
S29 reproduces the live incident of 2026-09-28: a failed job's step proposed
again under its old id).

Each case prints "case <id>: pass|fail"; executor.run_replay_tests records
them per case (executor.STREAM_REPLAY_CASES), and a failing case fails the
whole replay record, so it blocks envelope autonomy for every campaign.

The world (tests/test_stream_ap_world.py) runs the real ticker
(campaign.tick -> stream.StreamRun), the real executor, policy and approvals,
and the real remote.py / stream_remote.py verbs on a temporary INC_DIR; only
the cluster's own commands are faked. No network, no GPU.

Honest scope: S6 and S19 replay recorded ledgers (realloop_v1, pilot_v3); the
dispositions they assert follow the guard rule of 3.5 as written (see the
build notes of docs/CONTINUOUS_LOOP.md for the two places the contract's own
worked example of S6 disagrees with that rule). Every other case is a
synthetic scenario written with the contract in view: a reproduction of the
contract, not evidence that the loop works on real data.

Run:  python3 tests/test_stream_ap_replay.py [S1 S4 ...]
"""
import copy
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import (AP, B, C, DS, E, LS, M, NAME, OWNER, R, S, SR, TICK, DAY, X, World, check,  # noqa: E402
                                  commit_fields, gate, ledger_text, seg_ledger, truth, run_case)

FIX = W.FIX
PKG = W.PKG_ROOT
MOD = "weed_optimizer_framework.tools."
KNOWN = [{"id": "mfwd_porol", "provider": "ftp", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
          "bytes": 5.4e9, "images": 4435, "target_boxes": 9400, "lab_group": "TUM", "recall": True},
         {"id": "pags8", "provider": "weedai", "licence": "MIT", "target_classes": ["PalmerAmaranth"],
          "bytes": 2e9, "images": 1228, "target_boxes": 5026, "lab_group": "TAMU"},
         {"id": "cottonweeddet3", "provider": "kaggle", "licence": "CC BY 4.0",
          "target_classes": ["Carpetweed", "MorningGlory", "PalmerAmaranth"], "bytes": 5.18e9, "images": 848,
          "target_boxes": 1532, "lab_group": "LuLab"}]


def data_world(tag, **kw):
    """A world past R0 and R1 (lock, baselines, Stage A and C, placement,
    Step 1's one-time jobs) with an empty queue: the DATA lane's turn."""
    w = World(tag, **kw)
    w.ready_r0()
    w.queue(0)
    return w


def lane_items(w):
    return {ln: (w.lane(ln).get("item") or {}) for ln in S.LANES}


def fetches(w, source=None):
    return [s for s in w.submits if "run_inc_collect.sh" in " ".join(s["argv"]) and "fetch" in s["argv"]
            and (source is None or source in s["argv"])]


# ----------------------------------------------------------------------- S1
def s1():
    w = data_world("s1")
    w.candidates([
        {"id": "src_top", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
         "bytes": 1e9, "images": 500, "expected_target_boxes": 800},
        {"id": "src_second", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
         "bytes": 1e9, "images": 500, "expected_target_boxes": 700},
        {"id": "cottonweeddet12", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Waterhemp"],
         "bytes": 1e8, "images": 5000, "expected_target_boxes": 99999},
        {"id": "src_evallab", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
         "bytes": 1e8, "images": 900, "expected_target_boxes": 9000, "lab_group": "LuLab"},
        {"id": "src_imagelevel", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Waterhemp"],
         "bytes": 1e8, "images": 5000, "expected_target_boxes": 50000, "annotation_type": "image-level"},
        {"id": "src_quarantined", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Sicklepod"],
         "bytes": 1e8, "images": 5000, "expected_target_boxes": 70000}])
    w.tick()                                      # the first snapshot
    st = w.state()
    st["sources"]["src_quarantined"] = {"status": "quarantined"}
    pathlib.Path(S.StreamPaths(str(w.lab), "weed").state(NAME)).write_text(json.dumps(st))
    w.tick()
    ds = json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(NAME).read_text())["diagnoses"]
    d20 = next(d for d in ds if d["id"] == "D20")
    check("D20 fires on the empty queue (Q 0 < 2M) and names L16", d20["fired"] and d20["levers"] == ["L16"],
          d20["summary"])
    syn = [x for x in w.runner.launched if "lab-sync" in x["argv"] and "--file" in x["argv"]]
    rel = "intake/candidates_sync/src_top.json"
    rec = S.StreamPaths(str(w.lab), "weed").lab_inc / rel
    check("a discovered source fetched on the cluster: the lab first pushes its candidate record (L16S, no ssh)",
          syn and syn[0]["argv"][syn[0]["argv"].index("--file") + 1] == rel and rec.is_file()
          and json.loads(rec.read_text())["candidates"][0]["id"] == "src_top", [x["argv"][-4:] for x in syn])
    w.tick()
    f = fetches(w)
    check("exactly one L16 went to the cluster, on the top-ranked source", len(f) == 1 and "src_top" in f[0]["argv"],
          [x["argv"] for x in f])
    want = ["sbatch", "-p", "GPU-shared", "run_inc_collect.sh", "fetch", "--source", "src_top", "--max-bytes",
            "1000000000", "--candidates", LS.inc_path(rel)]
    ex = [e for e in w.executions() if e.get("lever") == "L16" and e.get("status") == "executed"]
    check("its argv is byte-equal to the documented command, risk R2",
          ex and ex[-1]["argv"] == want[3:] and ex[-1]["risk"] == "R2"
          and w.events("proposed")[-1]["argv"] == want, (ex and ex[-1]["argv"], w.events("proposed")[-1]["argv"]))
    check("  and the job names GPU-shared (-p GPU-shared)", f and f[0]["argv"][3:5] == ["-p", "GPU-shared"], f)
    from weed_optimizer_framework.tools.collect import __main__ as CM
    ns = CM.build_parser().parse_args(want[4:])
    check("  the collector's own parser reads it (fetch --candidates)", ns.candidates == LS.inc_path(rel), ns)
    never = [x for x in w.executions() if (x.get("params") or {}).get("source") in
             ("cottonweeddet12", "src_quarantined")]
    check("never a NEVER_TRAIN or quarantined source (no proposal, no filing)", not never, never)
    rev = (w.state() or {}).get("source_reviews") or {}
    check("the evaluation-lab and image-level sources are filed R3 for a person, never collected automatically",
          set(rev) == {"src_evallab", "src_imagelevel"} and all(
              w.approvals()[r["approval_id"]]["risk"] == "R3" and w.approvals()[r["approval_id"]]["status"] == "pending"
              for r in rev.values()), rev)
    check("no image-level or evaluation-lab source reached the cluster",
          not [x for x in w.submits if "src_evallab" in x["argv"] or "src_imagelevel" in x["argv"]])
    check("every tick made at most one ssh", w.max_calls() <= 1, w.max_calls())
    # a person approves the evaluation-lab review: it is adopted into the DATA
    # lane when the lane is free and run through the approval (never lost)
    aid = rev["src_evallab"]["approval_id"]
    AP.decide("weed", aid, "approve", OWNER, "the copy scan is calibrated", w.clock(), root=str(w.lab))
    w.job_done("inc_stream_collect_fetch_src_top")
    w.tick(3)
    check("an approved review (L16R) is adopted into the DATA lane and run through its approval",
          fetches(w, "src_evallab") and (w.approvals()[aid].get("execution") or {}).get("phase") == "done",
          ([x["argv"] for x in fetches(w)], w.approvals()[aid].get("execution")))
    check("  its fetch is followed like any L16 (attempt counted, source fetching)",
          (w.state()["sources"].get("src_evallab") or {}).get("attempts") == 1, w.state()["sources"].get("src_evallab"))


# ----------------------------------------------------------------------- S1b
def s1b():
    w = data_world("s1b", collect_config={"format": "collect-domain/1", "known_items": KNOWN +
                                           [{"id": "mh_weed16", "provider": "mendeley_zenodo", "licence": "CC BY 4.0",
                                             "target_classes": ["MorningGlory"], "bytes": 3e9, "images": 2000}],
                                           "placement": {"lab_only": ["github"]}})
    w.placement({"ftp": "pass"})
    # the empty candidates file: D20 has only the known items to fetch; they are
    # all in the queue's way, so leave them out of the deficit by marking them held
    w.tick()
    st = w.state()
    for k in ("mfwd_porol", "pags8", "cottonweeddet3", "mh_weed16"):
        st["sources"][k] = {"status": "held"}
    st["discover"] = {}
    S.StreamPaths(str(w.lab), "weed").state(NAME).write_text(json.dumps(st))
    w.tick()
    l15 = [x for x in w.runner.launched if "plan" in x["argv"]]
    check("with no open candidate and no discovery yet, D20 starts L15 on the lab (detached), for the deficit classes",
          len(l15) == 1 and "--classes" in l15[0]["argv"]
          and "PricklySida" in l15[0]["argv"][l15[0]["argv"].index("--classes") + 1], l15)
    lab_inc = str(S.StreamPaths(str(w.lab), "weed").lab_inc)
    check("  and names the lab's INC tree (--inc-dir): the ticker's environment has no INC_DIR, and the default "
          "is the cluster's /ocean path (2026-09-29, PermissionError at intake_lock)",
          l15 and "--inc-dir" in l15[0]["argv"] and l15[0]["argv"][l15[0]["argv"].index("--inc-dir") + 1] == lab_inc,
          l15 and l15[0]["argv"])
    check("  L15 used no ssh: the tick's one call was the snapshot",
          w.verbs[-1][0] == "stream-snapshot" and w.max_calls() <= 1)
    # the recorded provider responses reproduce four of the five known items
    w.candidates([dict(k, known_item=False) for k in KNOWN] +
                 [{"id": "new_rf_sida", "provider": "roboflow", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
                   "bytes": 5e8, "images": 300, "expected_target_boxes": 200}])
    w.runner.finish(ok=True)
    w.tick()
    rec = ((w.state() or {}).get("discover") or {}).get("recall") or {}
    check("recall against the owner's known items is reported (as H12): 3 of 4 found, mh_weed16 missed",
          rec.get("known") == 4 and rec.get("found") == 3 and rec.get("missed") == ["mh_weed16"], rec)
    check("  the miss raises a discovery-defect card", any(c.get("kind") == "discovery" for c in w.state()["cards"]))
    run = _run_of(w)
    got = {c["id"] for c in run._candidates()}
    check("  and the missed known item stays fetchable (it is a candidate of L16)", "mh_weed16" in got, got)
    d = (w.state() or {}).get("discover") or {}
    check("the discovery record: last run, new candidates found", d.get("found_new") >= 1 and d.get("last_utc"), d)
    # the collector's own recall (candidates.json `recall`, match rules) is the one read when present
    w.candidates([dict(k, known_item=False) for k in KNOWN],
                 recall={"known_items": 4, "found": [{"id": k["id"]} for k in KNOWN[:2]],
                         "missed": [{"id": "cottonweeddet3"}, {"id": "mh_weed16"}]})
    run._check_recall()
    rec = run.st["discover"]["recall"]
    check("  the collector's recall block, when present, is the one reported (2 of 4, by its match rules)",
          rec.get("basis") == "collector" and rec.get("found") == 2 and rec.get("missed") == ["cottonweeddet3",
                                                                                               "mh_weed16"], rec)


def _run_of(w):
    """A StreamRun on the world's last state, for reading its derived views."""
    paths = C.Paths(str(w.lab))
    ssh = C._SshBudget(None)
    raw = w.config()
    run = S.StreamRun(NAME, raw, paths, ssh, w.clock, w.log, w.hooks, lambda: W.RES, None, None, None)
    run.st = w.state()
    run.dom = LS.load_domain("weed")
    run.th = LS.load_thresholds()
    return run


# ----------------------------------------------------------------------- S2
def s2():
    w = data_world("s2", floors=(50.0, 10.0))
    w.candidates([{"id": "low", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 3e9, "images": 900, "expected_target_boxes": 9000},
                  {"id": "high", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 3e9, "images": 900, "expected_target_boxes": 100}])
    # both fetches stopped at a byte cap (collect.fetch: complete false, files
    # remaining): the next shard is collected once the admitted yield is seen
    w.sources([{"source": "low", "event": "fetch_started"}, {"source": "high", "event": "fetch_started"},
               {"source": "low", "event": "fetched", "bytes": 3e9, "seconds": 18000.0, "su": 5.0,
                "complete": False, "remaining": 4},
               {"source": "high", "event": "fetched", "bytes": 3e9, "seconds": 18000.0, "su": 5.0,
                "complete": False, "remaining": 4}])
    w.step1_status(per_source={"low": {"target_boxes_admitted": 30}, "high": {"target_boxes_admitted": 900}})
    w.tick()
    st = w.state()
    check("the collector's incomplete fetch marks each source partial (its next shard is collectable)",
          all(st["sources"][k].get("partial") is True for k in ("low", "high")), st["sources"])
    w0 = data_world("s2pre", floors=(50.0, 10.0))
    w0.sources([{"source": "mid", "event": "fetch_started"},
                {"source": "mid", "event": "fetched", "bytes": 3e9, "seconds": 18000.0, "su": 5.0, "complete": True}])
    w0.step1_status(per_source={"mid": {"target_boxes_admitted": 30}})
    w0.tick(3)
    check("  and D21 judges no source before its admission is observed (3 GB fetched, 30 boxes admitted so far, "
          "admission not yet seen: not closed)", (w0.state()["sources"].get("mid") or {}).get("status") != "closed"
          and not _diag(w0)["D21"]["fired"], (w0.state()["sources"].get("mid"), _diag(w0)["D21"]["summary"]))
    for k in ("low", "high"):
        # what the L17 admit's completion records (stream._done); the fold then
        # records the admitted yield (yield_recorded) from step1_stream's status
        st["sources"][k].update(status="admitted", admit_done_utc=W.utc(w.t[0]))
    S.StreamPaths(str(w.lab), "weed").state(NAME).write_text(json.dumps(st))
    w.tick(2)
    st = w.state()
    check("D21 closes the source whose yield is below the floor after 3 GB (>= min(2 GB, the source))",
          st["sources"]["low"]["status"] == "closed" and "floor" in st["sources"]["low"].get("closed_reason", ""),
          st["sources"]["low"])
    check("the source above the floor stays open", st["sources"]["high"]["status"] != "closed", st["sources"]["high"])
    check("no further L16 for the closed source", not fetches(w, "low"), [x["argv"] for x in fetches(w)])
    w.tick(3)
    check("  while the source above the floor is still collected (its next shard)", fetches(w, "high")
          and not fetches(w, "low"), [x["argv"] for x in fetches(w)])


# ----------------------------------------------------------------------- S3
def s3():
    w = data_world("s3")
    w.candidates([{"id": "src_a", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
                   "bytes": 1e9, "images": 500, "expected_target_boxes": 800},
                  {"id": "src_b", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "images": 500, "expected_target_boxes": 100}])
    w.tick(3)
    check("L16 on src_a runs", len(fetches(w, "src_a")) == 1)
    w.job_done("inc_stream_collect_fetch_src_a")
    w.tick(2)
    intakes = [s for s in w.submits if "intake" in s["argv"]]
    check("collect done -> intake (L16I) for that source only", len(intakes) == 1 and "src_a" in intakes[0]["argv"],
          [s["argv"] for s in w.submits])
    w.job_done("inc_stream_collect_intake_src_a")
    w.intake("b0001", "src_a", images=480)
    w.tick(2)
    admits = [s for s in w.submits if "run_inc2_stream.sh" in " ".join(s["argv"])]
    check("intake done -> L17 admit of that source's batch only (b0001)",
          len(admits) == 1 and admits[0]["argv"][-3:] == ["admit", "--intake", "b0001"], [s["argv"] for s in admits])
    pro = [e for e in w.events("proposed") if e.get("lever") == "L17"]
    check("  priced from the batch's size (480 images) and carrying the pinned verifier in its provenance",
          pro and pro[-1].get("est_gpu_hours") is not None, pro[-1:] if pro else None)
    # a verifier or LOCK mismatch: the admit refuses (step1_stream's pin check, exit 2), and is not retried
    w.job_done("inc_stream_admit_admit_b0001", state="FAILED",
               refusal="[step1_stream] REFUSED: the frozen verifier file step1_stream/verifier/verifier.joblib changed "
                       "(pinned 3fa2c1d0e9b8)")
    w.tick(3)
    admits2 = [s for s in w.submits if "run_inc2_stream.sh" in " ".join(s["argv"])]
    check("a verifier or LOCK mismatch is refused for good (retry: false): the admit is not submitted again",
          len(admits2) == 1, [s["argv"] for s in admits2])
    check("  and a card goes to a person (X11)", any(c.get("lever") == "X11" or "verifier" in (c.get("detail") or "")
                                                    for c in w.state()["cards"]), w.state()["cards"][-3:])


# ----------------------------------------------------------------------- S4
def train_world(tag, Q=None, **kw):
    w = World(tag, **kw)
    w.ready_r0()
    w.queue(4 * w.M if Q is None else Q, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    return w


def l18_argv(w):
    return [e for e in w.events("proposed") if e.get("lever") == "L18"]


def s4():
    w = train_world("s4")
    w.tick(2)
    pro = l18_argv(w)
    seg1 = "%s_s001" % w.sid
    want = ["python", "-m", "weed_optimizer_framework.tools.inc2.stream", "build", "--stream", w.sid, "--k", "4",
            "--exp", seg1]
    check("Q >= 4M -> exactly one L18, K = 4, argv byte-equal, naming the summary's next segment (the Stage B arms "
          "r0 and x1a are the stream's own, set at init)", len(pro) == 1 and pro[0]["argv"] == want,
          [p["argv"] for p in pro])
    sub = [x for x in w.submits if "run_inc2_build.sh" in " ".join(x["argv"])]
    check("  it ran within the envelope (R3, autonomy envelope, replay pass): one build job on GPU-shared",
          len(sub) == 1 and sub[0]["argv"][3:5] == ["-p", "GPU-shared"]
          and sub[0]["argv"][-8:] == ["inc2.stream", "build", "--stream", w.sid, "--k", "4", "--exp", seg1],
          [x["argv"] for x in sub])
    check("  its job is named for the segment it builds (inc_build_<sid>_s001) and carries the provenance",
          sub and sub[0]["name"] == "inc_build_%s_s001" % w.sid and sub[0]["env"].get("INCAP_CHILD_EXP")
          == "%s_s001" % w.sid and sub[0]["env"].get("INCAP_TRIGGER") == "D22", sub and sub[0]["env"])
    req = SR.parse_submit("build", ["inc2.stream"] + want[3:])
    check("the cluster's grammar reads it back (stream_remote.parse_submit: stream build, k 4, --exp)",
          req["params"] == {"stream": w.sid, "k": "4", "exp": seg1}, req)
    try:
        import importlib
        mod = importlib.import_module("weed_optimizer_framework.tools.inc2.stream")
        parser = getattr(mod, "build_parser", None) or getattr(mod, "parser", None)
        if callable(parser):
            ns = parser().parse_args(want[3:])
            check("inc2.stream's own argparse reads it back", getattr(ns, "k", None) in (4, "4")
                  and getattr(ns, "exp", None) == seg1, ns)
        else:
            print("      (inc2.stream has no build_parser(): checked against the contract grammar only)")
    except ImportError:
        print("      (inc2.stream is not in this checkout (group E): checked against the contract grammar only)")
    ex = [e for e in w.executions() if e.get("lever") == "L18" and e.get("status") == "executed"]
    check("priced: a positive est_gpu_hours charged to the envelope (segment K=4, the two Stage B recipes, truth on)",
          ex and ex[-1]["est_su"] and ex[-1]["est_su"] > 30, ex and ex[-1]["est_su"])
    it = next(e for e in w.events("proposed") if e.get("lever") == "L18")
    st = w.state()
    item = (st["lanes"]["TRAIN"].get("item") or {}).get("proposal") or {}
    check("base = P_{s-1}: the proposal records the pool it builds on, with its sha256",
          item.get("base_pool", {}).get("current") == "P_0" and item.get("base_pool", {}).get("sha256") == "p" * 64,
          item.get("base_pool"))
    ps = w.events("prospective_stream")
    check("it passes the prospective guard: the stream record (M, K, truth, thresholds, recipe rule, gate) went to "
          "the ledger before the L18", ps and w.ledger().index(ps[0]) < w.ledger().index(it)
          and pathlib.Path(ps[0]["path"]).is_file() and ps[0]["rules_version"] == DS.rules_version(), ps)


# ----------------------------------------------------------------------- S5
def s5():
    w = train_world("s5")
    w.tick(2)
    exp = "%s_s001" % w.sid
    rows = [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"]),
            ("I3", "REJECT", 0.1, ["regression"], "hurts", ["src_b"]),
            ("I4", "REJECT", 0.6, ["regression"], "neutral", ["src_c"]),
            ("I5", "HOLD", 0.5, [], "neutral", ["src_d"])]
    w.segment(1, rows, done=False)
    w.tick(2)
    check("an unfinished segment is never committed (no L19 while it runs)",
          not [e for e in w.events("proposed") if e.get("lever") == "L19"], w.events("proposed")[-2:])
    ex = [e for e in w.executions() if e.get("lever") == "L18" and e.get("status") == "executed"]
    jid = (ex[-1].get("job_ids") or [None])[0] if ex else None
    keys = [str(e.get("job")) for e in B.su_ledger._read_deduped("weed", w.xctx.su_base_dir)]
    check("the segment's build job itself (one V100 on GPU-shared while it builds) is settled from sacct once it "
          "ended (6.6)", jid and "inc:job%s:sacct" % jid in keys, (jid, keys))
    w.segment(1, rows, done=True)
    w.tick(4)
    l19 = [e for e in w.events("proposed") if e.get("lever") == "L19"]
    check("the finished segment -> L19 commit, argv exact",
          l19 and l19[0]["argv"] == ["python", "-m", "weed_optimizer_framework.tools.inc2.stream", "commit", "--exp", exp],
          l19)
    check("  run on the login node (stream-run), once", len([r for r in w.runs if "commit" in r]) == 1, w.runs)
    com = [e for e in _stream_ledger(w) if e["event"] == "commit"]
    disp = (com[-1].get("dispositions") if com else None)
    check("dispositions by the guard rule (inc2.stream's commit): I2 accepted, I3 data (quarantine), I4 recipe and "
          "I5 HOLD (returned once)", disp == {"I2": "accepted", "I3": "data", "I4": "recipe", "I5": "hold"}, disp)
    run = _run_of(w)
    run._build_evidence()
    v = DS.View(run.ev, run.dom, run.th)
    rows_read = DS.commit_rows(v, *DS.stream_commits(v)[-1])
    check("  the diagnoses read those dispositions from the commit line, cited at their address in the stream ledger",
          {r["step"]: r["disposition"] for r in rows_read} == disp
          and all(c["artifact"] == "stream/%s/ledger.jsonl" % w.sid for r in rows_read for c in r["cites"]),
          [(r["step"], r["disposition"]) for r in rows_read])
    st = w.state()
    seg = next(s for s in st["segments"] if s["exp"] == exp)
    check("the segment is committed; P_1 = P_0 + I2 (the commit line's accepted)",
          seg.get("committed") and com and com[-1]["accepted"] == ["I2"], seg)
    l4 = [e for e in w.events("proposed") if e.get("lever") == "L4"]
    check("the data-disposed I3 is audited (L4 on I3 only)", l4 and "I3=" in " ".join(l4[0]["argv"])
          and "I4=" not in " ".join(l4[0]["argv"]), l4 and l4[0]["argv"])


def _stream_ledger(w):
    p = w.inc / "stream" / w.sid / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# ----------------------------------------------------------------------- S6
def s6():
    man = json.loads((FIX / "funnel" / "MANIFEST.json").read_text())
    dom = LS.load_domain("weed")
    pin = man["files"]["realloop_v1/ledger.jsonl"]["sha256"]
    check("realloop_v1 is the funnel's pinned copy, reused by sha256 (not duplicated)",
          dom["prior"]["sha256"] == pin and not (FIX / "realloop_v1").exists(), (dom["prior"]["sha256"], pin))
    prior = DS.prior_evidence(dom)
    rows = DS.steps(prior, prior.exp)
    disp = [(r["step"], r["disposition"]) for r in rows]
    check("dispositions by the guard rule: V1 data; V2 species; UNVERIFIED species (P_data 0.11 but truth helps); "
          "V3 species; OTHER_HEAVY flips; V4 species",
          disp == [("V1", "data"), ("V2", "species"), ("UNVERIFIED", "species"), ("V3", "species"),
                   ("OTHER_HEAVY", "flips"), ("V4", "species")], disp)
    ctx = {"now_utc": "2026-10-01T00:00:00Z", "sid": dom["sid"], "M": dom["increment"]["M"],
           "lanes": {ln: {"phase": "IDLE", "busy": False} for ln in S.LANES}, "queue": {"Q": 100},
           "stage": {"lock": True, "step1_stream": {"bootstrap": "2026-09-29T00:00:00Z",   # R1 ran (contract 10)
                                                    "knowntruth": "2026-09-29T00:00:00Z",
                                                    "backfill": "2026-09-29T00:00:00Z"}},
           "discover": {"last_utc": "2026-09-30T00:00:00Z", "found_new": 2},
           "candidates": [{"id": "src_x", "licence": "CC BY 4.0", "target_classes": ["Purslane"], "bytes": 1e9,
                           "expected_target_boxes": 500}], "sources": {}}
    ev = E.from_texts({}, dom["sid"], context=ctx)
    th = LS.load_thresholds()
    by = DS.by_id(DS.detect(ev, dom, th, prior=prior, include_prior_d31=True))
    check("D30 does not fire: no REJECT is regression-only (recipe-caused)", not by["D30"]["fired"], by["D30"]["summary"])
    d31 = by["D31"]
    blamed = [b["step"] for b in (d31.get("detail") or {}).get("blamed") or []]
    check("D31 fires with L4 first: on V1 (disposition data) and on the truth arm's 'hurts' (OTHER_HEAVY, V4)",
          d31["fired"] and d31["levers"][0] == "L4" and blamed == ["V1", "OTHER_HEAVY", "V4"], (blamed, d31["summary"]))
    check("D33 names PricklySida (5 of 6 REJECTs)", by["D33"]["fired"] and by["D33"]["detail"]["species"]
          == ["PricklySida"] and by["D33"]["detail"]["counts"]["PricklySida"] == 5, by["D33"]["summary"])
    check("the DATA lane still proposes L16 when the queue is low", by["D20"]["fired"] and by["D20"]["levers"] == ["L16"],
          by["D20"]["summary"])
    th2 = LS.load_thresholds()
    by2 = DS.by_id(DS.detect(ev, dom, th2, prior=prior))
    check("in production the prior feeds D30 and D33 only (no L4 is ever proposed on realloop_v1)",
          not by2["D31"]["fired"] and by2["D33"]["fired"], by2["D31"]["summary"])
    ctx2 = dict(ctx, stage=dict(ctx["stage"], stage_a_ready=True))
    by3 = DS.by_id(DS.detect(E.from_texts({}, dom["sid"], context=ctx2), dom, th, prior=prior))
    check("once Stage A is READY with no segment decided, D30 and D33 read nothing (silent)",
          not by3["D30"]["fired"] and not by3["D33"]["fired"], (by3["D30"]["summary"], by3["D33"]["summary"]))


# ----------------------------------------------------------------------- S7
def s7():
    w = train_world("s7", Q=0)
    w.tick()
    e1 = w.segment(1, [("I1", "REJECT", 0.0, ["regression"], "hurts", ["src_x"]),
                       ("I2", "ACCEPT", 0.9, [], "helps", ["src_y"])])
    w._commit(e1)
    w.tick(3)
    check("a data blame (P_data 0.0 from source X, disposed 'data' by the commit) -> L4 on that increment first, "
          "no L24 yet", [e["lever"] for e in w.events("proposed") if e.get("lever") in ("L4", "L24")] == ["L4"],
          [e["lever"] for e in w.events("proposed")])
    e2 = w.segment(2, [("J1", "REJECT", 0.1, ["species"], "neutral", ["src_x"])])
    w._commit(e2)
    w.tick(4)
    q = [r for r in w.runs if "quarantine" in r]
    check("X blamed twice -> L24 on X, cited D31 (a login-node verb)", q and "src_x" in q[0] and "D31" in q[0], w.runs)
    st = w.state()
    check("  and X is closed: quarantined (L24) and, blamed twice, closed by D21",
          st["sources"]["src_x"]["status"] in ("quarantined", "closed") and st["sources"]["src_x"].get("blames") == 2,
          st["sources"].get("src_x"))
    w.candidates([{"id": "src_x", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 9e5}])
    w.tick(3)
    check("  X is never collected again", not fetches(w, "src_x"), [f["argv"] for f in fetches(w)])


# ----------------------------------------------------------------------- S8
def s8():
    w = train_world("s8", Q=0)
    w.tick()
    s1 = w.segment(1, [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"])])
    w._commit(s1)
    m1 = w.milestone_built(done=True)
    w.compare_plan[m1] = {"verdict": "hurts", "perm_p": 0.004, "new_mean_dev": 0.781, "old_mean_dev": 0.812}
    w.tick(3)
    lc = [e for e in w.events("proposed") if e.get("lever") == "LC"]
    check("the finished milestone is compared on dev by the stream (LC: compare --exp, a login-node verb)",
          lc and lc[0]["argv"][-4:] == [MOD + "inc2.stream", "compare", "--exp", m1] and "DCMP" in lc[0]["trigger"],
          [e.get("argv") for e in lc])
    w.tick(3)
    ds = _diag(w)
    check("the recorded 5 v 5 'hurts' (p 0.004 <= 0.025, lower mean) with its recommended rollback -> D25 with L21 "
          "to P_c (milestone 0's pool P_0)",
          (ds["D25"]["fired"] and ds["D25"]["detail"].get("to") == "P_0" and ds["D25"]["levers"][0] == "L21")
          or any(e.get("lever") == "L21" for e in w.events("proposed")), ds["D25"]["summary"])
    rb = [r for r in w.runs if "rollback" in r]
    check("L21 ran to P_0 within the envelope (the rollback the last 'hurts' milestone recommended)",
          rb and rb[0][-2:] == ["--to", "P_0"], w.runs)
    check("  TRAIN held while it stands", any(e.get("event") == "lane_hold" and e.get("lane") == "TRAIN"
                                              for e in w.events("lane_hold")), w.events("lane_hold"))
    check("  card X4 raised (leave-one-source-out), the platform bisects (L27 next)",
          any(c.get("lever") == "X4" for c in w.state()["cards"]), w.state()["cards"][-3:])
    w.tick(3)
    l27 = [e for e in w.events("proposed") if e.get("lever") == "L27"]
    check("  and L27 bisects the suspect increments against milestone 0's seeds, its first arm <sid>_b001",
          l27 and l27[0]["argv"][-4:] == ["--stream", w.sid, "--from", "P_0"] and l27[0].get("child_exp")
          == "%s_b001" % w.sid, [(e.get("argv"), e.get("child_exp")) for e in l27])
    arms = w.bisect_built(done=True) if [e for e in w.stream_ledger() if e.get("event") == "rollback"] else {}
    w.tick(4)
    dec = [e for e in _stream_ledger(w) if e.get("event") == "bisect" and e.get("phase") == "decide"]
    check("  each finished arm is decided by the stream (LC compare --exp <arm>)",
          dec and set(dec[-1]["decisions"]) == set(arms), dec)
    # a recommendation the recorded comparison does not support goes to a person
    w3 = train_world("s8c", Q=0)
    w3.tick()
    w3._commit(w3.segment(1, [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"])]))
    m3 = w3.milestone_built(done=True)
    w3.compare_plan[m3] = {"verdict": "hurts", "perm_p": 0.2, "new_mean_dev": 0.80, "old_mean_dev": 0.81}
    w3.tick(6)
    d3 = _diag(w3)["D25"]
    check("a recommended rollback whose recorded p (0.2) misses the pre-registered 0.025: no L21, a person reads it",
          d3["fired"] and "OP_ESCALATE" in d3["levers"] and not [r for r in w3.runs if "rollback" in r], d3["summary"])
    # the boundary check
    w2 = train_world("s8b", Q=0)
    w2.tick()
    for n in (1, 2):
        w2._commit(w2.segment(n, [("I%d" % n, "ACCEPT", 0.9, [], "helps", ["a"])]))
    w2.es["boundary"] = {"segment": "%s_s002" % w2.sid, "previous": "%s_s001" % w2.sid, "new_mean": 0.791,
                         "old_mean": 0.813, "old_sd": 0.001, "fires": True}
    w2._summary()
    w2.tick(3)
    l20 = [e for e in w2.events("proposed") if e.get("lever") == "L20"]
    check("a failed boundary check (mean(new) < mean(old) - 2 sd(old), from the summary's means) -> L20 at once, "
          "TRAIN holds", l20 and "D25" in l20[0]["trigger"] and any(e.get("lane") == "TRAIN"
                                                                    for e in w2.events("lane_hold")),
          (l20, w2.events("lane_hold")))
    check("  the milestone it builds is the summary's next (<sid>_m001)",
          l20 and l20[0].get("child_exp") == "%s_m001" % w2.sid, l20 and l20[0].get("child_exp"))


# ----------------------------------------------------------------------- S9
def _write_state(w, st):
    S.StreamPaths(str(w.lab), w.domain).state(NAME).write_text(json.dumps(st))


def _diag(w):
    return {d["id"]: d for d in json.loads(S.StreamPaths(str(w.lab), w.domain).diagnoses(NAME).read_text())["diagnoses"]}


def _never_complete(w):
    st = w.state() or {}
    phases = [(st.get("lanes") or {}).get(ln, {}).get("phase") for ln in S.ALL_LANES]
    return "COMPLETE" not in phases and not w.config().get("completed_by") and st.get("phase") != "COMPLETE"


def s9():
    w = data_world("s9a")
    w.candidates([{"id": "src_ok", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 500}])
    w.tick(3)
    check("an empty queue with a candidate and budget left: L16 runs, nothing completes",
          fetches(w, "src_ok") and _never_complete(w), [f["argv"] for f in fetches(w)])
    # exhausted: no candidate, a discovery a day ago found nothing new
    w2 = data_world("s9b")
    w2.tick()
    st = w2.state()
    st["discover"] = {"last_utc": W.utc(w2.t[0] - DAY), "found_new": 0, "empty_runs": 1, "runs": 3}
    _write_state(w2, st)
    w2.tick(2)
    ln = w2.lane("DATA")
    check("exhausted: D29 moves DATA to WAIT_DATA (7 days, first empty run) with a card, never COMPLETE",
          ln.get("phase") == "WAIT_DATA" and ln.get("until_utc") == W.utc(S._secs(ln.get("until_utc")))
          and any(c.get("kind") == "wait_data" for c in w2.state()["cards"]) and _never_complete(w2), ln)
    check("  and discovery (L15) does not run inside the wait",
          not [x for x in w2.runner.launched if "plan" in x["argv"]], w2.runner.launched)
    w2.advance(8 * DAY)
    w2.tick(2)
    check("after the wait, discovery runs again (L15, detached)", [x for x in w2.runner.launched if "plan" in x["argv"]],
          w2.runner.launched)
    st = w2.state()
    st["lanes"]["DATA"].update(item=None, phase="IDLE")
    st["discover"] = {"last_utc": W.utc(w2.t[0] - 3600), "found_new": 0, "empty_runs": 2}
    _write_state(w2, st)
    w2.tick(2)
    until = S._secs(w2.lane("DATA").get("until_utc")) or 0
    check("  a second empty discovery backs off to 14 days", 13 * DAY < until - (w2.t[0] - 2 * TICK) < 15 * DAY,
          (until - w2.t[0]) / DAY)
    # budget out -> PAUSE, never COMPLETE
    w3 = data_world("s9c")
    B.su_ledger.record({"domain": "weed", "job": "inc:old:base", "step": "inc:%s" % NAME, "actor": AUTO_ACTOR,
                        "gpu_count": 1, "gpu_type": "v100-32", "elapsed_s": 1000 * 3600.0,
                        "su": B.su_ledger.su_for("v100-32", 1, 1000 * 3600.0), "ts": W.utc(w3.t[0])},
                       base_dir=w3.xctx.su_base_dir)
    w3.tick(2)
    check("the envelope exhausted -> PAUSE (budget_exhausted, D10S) with a card, not COMPLETE",
          "budget_exhausted" in str(w3.config().get("paused_reason")) and _never_complete(w3),
          w3.config().get("paused_reason"))
    # the allocation's end date
    w4 = data_world("s9d")
    w4.projects[0]["end_date"] = "2026-10-20"
    w4.tick(3)
    check("30 days before the allocation ends: the renewal card (X14), no pause",
          any(c.get("lever") == "X14" for c in w4.state()["cards"]) and not w4.config().get("paused_reason"),
          (w4.state()["cards"][-3:], w4.config().get("paused_reason")))
    w4.projects[0]["end_date"] = "2026-09-30"
    w4.tick(3)
    check("at the end date: PAUSE allocation_ended (resumes on renewal without a rebuild)",
          "allocation_ended" in str(w4.config().get("paused_reason")) and _never_complete(w4),
          w4.config().get("paused_reason"))


AUTO_ACTOR = W.AUTO


# ----------------------------------------------------------------------- S10
def s10():
    w = data_world("s10a")
    w.set_config(alloc_reserve_su=12000.0)
    w.tick(3)
    check("D27: the allocation balance minus what is committed is under the reserve -> PAUSE",
          "allocation_reserve" in str(w.config().get("paused_reason")), w.config().get("paused_reason"))
    wr = data_world("s10r")
    wr.set_config(alloc_reserve_su=10000.0)
    B.su_ledger.record({"domain": "weed", "job": "inc:spent_exp:base", "step": "inc:weed_inc_v1", "actor": AUTO_ACTOR,
                        "gpu_count": 1, "gpu_type": "v100-32", "elapsed_s": 600.0 * 3600.0,
                        "su": B.su_ledger.su_for("v100-32", 1, 600.0 * 3600.0), "ts": W.utc(wr.t[0])},
                       base_dir=wr.xctx.su_base_dir)
    wr.tick(3)
    check("D27: SU already spent is not taken off the balance a second time (balance 10529 - committed 0 >= reserve "
          "10000 with 600 SU spent): no pause", not wr.config().get("paused_reason"), wr.config().get("paused_reason"))
    wh = data_world("s10h")
    B.su_ledger.record({"domain": "weed", "job": "rndtrain_46000", "step": "train", "actor": "round-scheduler",
                        "gpu_count": 1, "gpu_type": "v100-32", "elapsed_s": 1600.0 * 3600.0,
                        "su": B.su_ledger.su_for("v100-32", 1, 1600.0 * 3600.0), "ts": W.utc(wh.t[0] - 30 * DAY)},
                       base_dir=wh.xctx.su_base_dir)
    wh.tick(3)
    check("the domain's round history (1,600 SU of pre-INC training) already exhausts its 1,500 SU envelope -> "
          "PAUSE before the stream spends anything, card X14 (6.6 review)",
          "domain_envelope_exhausted" in str(wh.config().get("paused_reason"))
          and any(c.get("lever") == "X14" for c in wh.state()["cards"]), wh.config().get("paused_reason"))
    wq = data_world("s10q")
    wq.quota_rec = {"ok": True, "used_gb": 6850.0, "quota_gb": 7000.0, "free_gb": 150.0}
    wq.tick(3)
    check("D27: /ocean with less than 3 % free (150 of 7,000 GB) -> PAUSE (quota)",
          "quota" in str(wq.config().get("paused_reason")), wq.config().get("paused_reason"))
    for tag, cfg, ledger_other, needle in (("day", {"daily_cap_su": 5.0}, 0.0, "today's cap"),
                                           ("month", {"window_cap_su": 10.0}, 0.0, "this month's window"),
                                           ("domain", {}, 1490.0, "of the domain's")):
        w2 = train_world("s10" + tag)
        if cfg:
            w2.set_config(**cfg)
        if ledger_other:
            B.su_ledger.record({"domain": "weed", "job": "inc:other_exp:base", "step": "inc:weed_inc_v1",
                                "actor": AUTO_ACTOR, "gpu_count": 1, "gpu_type": "v100-32",
                                "elapsed_s": ledger_other * 3600.0,
                                "su": B.su_ledger.su_for("v100-32", 1, ledger_other * 3600.0), "ts": W.utc(w2.t[0])},
                               base_dir=w2.xctx.su_base_dir)
        w2.tick(3)
        ex = [e for e in w2.executions() if e.get("lever") == "L18"]
        why = " ".join(" ".join(e.get("reasons") or []) for e in ex)
        check("the %s cap refuses the L18 build (it waits; the campaign does not pause)" % tag,
              not [e for e in ex if e.get("status") == "executed"] and needle in why
              and not w2.config().get("paused_reason"), (why[:300], w2.config().get("paused_reason")))


# ----------------------------------------------------------------------- S11
def s11():
    w = train_world("s11")
    w.queue(4 * w.M, boxes={"Purslane": 900}, pool_images=45000)
    w.tick(3)
    d = _diag(w)["D26"]
    check("D26: a cold run at 45,000 + M images projects to >= 0.8 x its 8 h limit", d["fired"] and "X15" in d["levers"],
          d["summary"])
    check("  no build: no L18", not l18_argv(w), [e.get("lever") for e in w.events("proposed")])
    check("  card X15", any(c.get("lever") == "X15" for c in w.state()["cards"]))
    check("  TRAIN and MAINT held", w.lane("TRAIN").get("diag_hold") and w.lane("MAINT").get("diag_hold"),
          (w.lane("TRAIN"), w.lane("MAINT")))
    # a 12,500-image pool on the s640 arm (est. cost factor 2): an x1b run (50 epochs) projects past 0.8 x its 3 h
    # limit, an x1a run (30 epochs) and the cold run (100 epochs, 8 h limit) stay under their lines
    for tag, recipes, fires in (("r0x1a", ("r0", "x1a"), False), ("r0x1b", ("r0", "x1b"), True)):
        w2 = World("s11" + tag)
        w2.capacity_choice, w2.stage_a_recipes = "s640", recipes
        w2.ready_r0()
        w2.queue(4 * w2.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w2.t[0] - 2 * DAY), pool_images=12500)
        w2.tick(3)
        d = _diag(w2)["D26"]
        check("D26 on the s640 arm with Stage B arms %s: %s (it prices the recipes the stream runs, not every "
              "recipe of the domain)" % ("+".join(recipes), "holds" if fires else "no hold"),
              bool(d["fired"]) is fires and bool(l18_argv(w2)) is (not fires), (d["summary"], l18_argv(w2)))


# ----------------------------------------------------------------------- S12
def _scenario_s12(w):
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY),
            extra={"test": {"images": 1977, "map50_95": 0.85}, "imageweeds": {"n": 3208}})
    w.candidates([{"id": "tsw22_extra", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 500}])
    w.tick(2)
    rows = [("I2", "ACCEPT", 0.9, [], "helps", ["tsw22"]), ("I3", "REJECT", 0.1, ["regression"], "hurts", ["tsw23"])]
    fin = [w.final_row("chain r0: final incumbent", 0.812, 0.0, 1, test_mean=0.861)]
    exp = w.segment(1, rows, final=fin, dev={"base__s0": 0.81, "base__s1": 0.811, "base__s2": 0.812})
    w._w("%s/runs/base__s0/scores/test.json" % exp, {"exam": "test", "map50_95": 0.86})
    w.stream_event("milestone", phase="note", exp="m0", test={"mean": 0.854, "sd": 0.007}, dev={"mean": 0.81})
    w.tick(4)


def _norm(w, obj):
    return json.loads(json.dumps(obj).replace(str(w.dir), "<W>"))


def s12():
    a, b = train_world("s12a"), train_world("s12b", perturbed=True)
    b.perturbed = True
    for w in (a, b):
        _scenario_s12(w)
    da = [(e.get("fired"), e.get("summary"), e.get("digest_sha256")) for e in a.events("diagnosed")]
    db = [(e.get("fired"), e.get("summary"), e.get("digest_sha256")) for e in b.events("diagnosed")]
    check("every test and non-decision exam value perturbed: identical diagnoses and evidence digests",
          da == db and da, (da[-1:], db[-1:]))
    pa = _norm(a, [(e.get("lever"), e.get("argv")) for e in a.events("proposed")])
    pb = _norm(b, [(e.get("lever"), e.get("argv")) for e in b.events("proposed")])
    check("  identical proposals and argv", pa == pb and pa, (pa, pb))
    xa = _norm(a, [(e.get("action"), e.get("status"), e.get("argv")) for e in a.executions()])
    xb = _norm(b, [(e.get("action"), e.get("status"), e.get("argv")) for e in b.executions()])
    check("  identical executions", xa == xb, (xa[-2:], xb[-2:]))
    ev = S.StreamPaths(str(b.lab), "weed")
    run = _run_of(b)
    run._build_evidence()
    check("the evidence holds no non-decision exam (D14's leak check) and the tsw rows only as training sources",
          not run.ev.leaks() and "tsw22" in json.dumps(run.ev.artifacts), run.ev.leaks()[:5])
    check("the test score file of a run is never read", "scores/test.json" not in json.dumps(run.ev.loader_record()))


# ----------------------------------------------------------------------- S13
def _domain_free_files():
    base = PKG / "weed_optimizer_framework" / "tools" / "inc_autopilot"
    return [base / n for n in ("stream.py", "stream_remote.py", "diagnose_stream.py", "levers_stream.py")]


def _stream_terms():
    sys.path.insert(0, str(PKG / "tests"))
    import test_funnel_domain_free as DF
    t = DF.terms()
    sub, tok = set(t["substring"]), set(t["token"])
    for p in (PKG / "weed_optimizer_framework" / "tools" / "inc_autopilot" / "stream_domains").glob("*.json"):
        d = json.loads(p.read_text())
        tok.add(str(d.get("domain")).lower())
        tok |= {str(k).lower() for k in ((d.get("species_priority") or {}).get("value") or {})}
        tok |= {str(k).lower() for k in ((d.get("eval_lab_groups") or {}).get("value") or [])}
        tok.add(str((d.get("stage_c") or {}).get("holdout")).lower())
    return DF, {"substring": sorted(sub), "token": sorted(tok)}


def _scan(files, DF, t):
    out = []
    toks = set(t["token"])
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            n = DF.normalise(line)
            out += [(p.name, i, x) for x in t["substring"] if x in n]
            out += [(p.name, i, x) for x in DF.TOKEN_RE.findall(n) if x in toks]
    return out


VEH_STREAM = {
    "format": "inc-autopilot/stream-domain/1", "domain": "vehicles", "sid": "veh_stream_v1", "protocol_package": "inc2",
    "funnel_domain": str(FIX / "funnel" / "vehicles" / "vehicles.json"), "collect_config": "collect/domains/vehicles.json",
    "never_train": {"source": None, "attr": "VEH_NEVER"}, "metric": "map50_95",
    "increment": {"M": 100, "K_max": 4, "base_images": 1000},
    "species_priority": {"value": {"bus": 1, "car": 3, "truck": 2}}, "eval_lab_groups": {"value": ["RoadLab"]},
    "baselines": {"items": [{"id": "vb", "exp": "veh_b0", "manifest": "splits/v2/base_v2.jsonl", "seeds": "0,1,2,3,4",
                             "arm": "n640", "role": "b_v2", "required": True}]},
    "capacity": {"default_arm": "n640", "verdict": "capacity-verdict",
                 "arms": {"n640": {"cost_factor": 1.0, "exp": "veh_b0"}}},
    "stage_a": {"exp": "veh_pilot4", "from_exp": "veh_pilot3", "recipes": "x1a", "record": "stage_a.json"},
    "stage_c": {"holdout": "fleet22"},
    "cost": {"cold_ms_per_image_epoch": 7.0, "incr_ms_per_image_epoch": 7.4, "cold_epochs": 100,
             "chain_epochs": {"r0": 30, "x1a": 30, "x1b": 50}, "seeds": 3, "milestone_seeds": 5, "finals_hours": 0.8,
             "build_job_hours": 4.0, "collect_job_hours": 4.0, "intake_job_hours": 1.0, "admit_fixed_hours": 0.5,
             "admit_seconds_per_1000_images": 75, "probe_hours": 0.5, "splits_hours": 4.0},
    "walltime": {"cold_h": 8.0, "incremental_h": 3.0, "build_h": 4.0},
    "allocation": {"end_date": "2027-06-30", "resource_pattern": "GPU"}, "decisions": [], "prior": {}}
VEH_NEVER = {"banned_roadcam_set"}
VEH_TRAINER = "# the vehicles trainer's never-train slugs, parsed (never imported)\nVEH_NEVER = {'banned_roadcam_set'}\n"


def s13():
    DF, t = _stream_terms()
    files = _domain_free_files()
    hits = _scan(files, DF, t)
    check("the stream code greps clean of every domain term (%d files, the funnel's method)" % len(files),
          not hits, hits[:10])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="s13_", dir=str(W.TMP)))
    planted = tmp / "planted.py"
    planted.write_text("# D33 names PricklySida\nX = 'imageweeds'\n")
    got = {x for _f, _i, x in _scan([planted], DF, t)}
    check("  a planted species name and exam name are flagged", {"pricklysida", "imageweeds"} <= got, got)
    # the second domain runs S1-S5 on its own config
    cfgp = tmp / "vehicles_stream.json"
    trainer = tmp / "vehicles_trainer.py"
    trainer.write_text(VEH_TRAINER)
    cfgp.write_text(json.dumps(dict(VEH_STREAM, never_train={"source": str(trainer), "attr": "VEH_NEVER"})))
    w = World("s13v", domain="vehicles", stream_domain=str(cfgp),
              collect_config={"known_items": [], "placement": {"lab_only": []}})
    w.stage_a_recipes = ["r0"]                      # no Stage A survivor there: r0 alone
    w.ready_r0()
    w.placement({"hf": "pass"})
    w.queue(0)
    w.candidates([{"id": "fleetcam_2025", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["bus"],
                   "bytes": 1e9, "expected_target_boxes": 900},
                  {"id": "banned_roadcam_set", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["car"],
                   "bytes": 1e8, "expected_target_boxes": 90000}])
    w.tick(3)
    f = fetches(w)
    check("vehicles S1: L16 on the top vehicles candidate, never its own never-train set",
          len(f) == 1 and "fleetcam_2025" in f[0]["argv"], [x["argv"] for x in f])
    check("  its ledger and approvals live under _brain/vehicles/inc",
          (w.lab / "results" / "framework" / "_brain" / "vehicles" / "inc" / "inc_campaign.jsonl").is_file()
          and not (w.lab / "results" / "framework" / "_brain" / "weed" / "inc" / "inc_campaign.jsonl").is_file())
    w.job_done("inc_stream_collect_fetch_fleetcam")
    w.tick(2)
    w.job_done("inc_stream_collect_intake_fleetcam")
    w.intake("vb0001", "fleetcam_2025", images=300)
    w.tick(2)
    check("vehicles S3: intake, then the admit of its batch",
          any(s["argv"][-3:] == ["admit", "--intake", "vb0001"] for s in w.submits), [s["argv"][-4:] for s in w.submits])
    w.queue(4 * 100, boxes={"bus": 300}, pool_images=1000)
    w.tick(3)
    b = [x for x in w.submits if "run_inc2_build.sh" in " ".join(x["argv"])]
    check("vehicles S4: Q >= 4M (M 100) -> L18 K 4 on veh_stream_v1",
          b and b[-1]["argv"][-8:] == ["inc2.stream", "build", "--stream", "veh_stream_v1", "--k", "4", "--exp",
                                       "veh_stream_v1_s001"], [x["argv"][-8:] for x in b])
    rowsv = [("V1", "ACCEPT", 0.9, [], "helps", ["fleetcam_2025"])]
    w.segment(1, rowsv, final=[dict(w.final_row("chain r0: final incumbent", 0.7, 0.0, 1), exams={
        "dev": {"twelve": {"mean": 0.7}}, "night_exam": {"twelve": {"mean": 0.3}}})])
    w.tick(4)
    check("vehicles S5: the finished segment is committed (L19)", any("commit" in r for r in w.runs), w.runs)
    run = _run_of_domain(w, "vehicles")
    run._build_evidence()
    check("  the vehicles exams (night_exam, rain_exam) never enter its evidence",
          "night_exam" not in json.dumps(run.ev.artifacts) and not run.ev.leaks(), run.ev.leaks()[:3])


def _run_of_domain(w, domain):
    paths = C.Paths(str(w.lab))
    run = S.StreamRun(NAME, w.config(), paths, C._SshBudget(None), w.clock, w.log, w.hooks, lambda: W.RES, None,
                      None, None)
    run.st = w.state()
    run.dom = LS.load_domain((w.config().get("stream") or {}).get("stream_domain"))
    run.th = LS.load_thresholds()
    return run


# ----------------------------------------------------------------------- S14
def s14():
    w = data_world("s14")
    w.tick()
    st = w.state()
    st["sources"]["leaky"] = {"status": "intaken", "batch": "b0009", "attempts": 1}
    _write_state(w, st)
    w.tick()
    check("no batch is admitted before its intake summary is observed",
          not [s for s in w.submits if "b0009" in s["argv"]], [s["argv"][-3:] for s in w.submits])
    w.intake("b0009", "leaky", images=100, eval_share=0.06, base_share=0.01,
             reasons={"near_eval_v2": 4, "near_eval_variant": 2, "base_copy": 1})
    # D28-v2: a large source whose two dHash hits the intake weighed at pair cos 0.12 and 0.31 (chance)
    w.intake("b0010", "chancy", images=3000, eval_share=0.0007, reasons={"near_eval_variant": 2},
             pair_cos=[0.12, 0.31])
    w.tick(3)
    d = _diag(w)["D28"]
    check("D28: the source's dHash copies of evaluation images, with no pair cosine recorded (the intake "
          "summary's guard counts; D28-v2's fail-closed rule) -> L24 and a card",
          d["fired"] and "L24" in d["levers"], d["summary"])
    q = [r for r in w.runs if "quarantine" in r]
    check("  L24 ran on the source, cited D28", q and "leaky" in q[0] and "D28" in q[0], w.runs)
    check("  D28-v2: the source whose two hits were weighed at pair cos 0.12 and 0.31 is judged chance and not "
          "quarantined", not any("chancy" in r for r in w.runs) and "chancy: 2 dHash hit(s) in 3002 images, "
          "0 confirmed" in d["summary"] and not any(h["source"] == "chancy" for h in d["detail"]["leaks"]),
          (d["summary"], w.runs))
    check("  a leak card", any(c.get("kind") in ("leak", "escalation") for c in w.state()["cards"]))
    check("  and its images never reach the queue: its batch is never admitted",
          not [s for s in w.submits if "b0009" in s["argv"]], [s["argv"][-3:] for s in w.submits])


# ----------------------------------------------------------------------- S15
def s15():
    w = data_world("s15")
    cands = [{"id": "src%02d" % i, "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
              "bytes": 1e9, "expected_target_boxes": 100 + i} for i in range(12)]
    w.candidates(cands)
    w.tick()
    for day in range(10):
        w.tick(2)
        w.job_done("inc_stream_collect_fetch_")
        w.tick()
        st = w.state()
        # the fetched source's pipeline ends at once (so the lane takes the next one)
        for k, v in st["sources"].items():
            if v.get("status") == "fetched":
                v["status"] = "closed"
        st["lanes"]["DATA"].update(item=None, phase="IDLE")
        _write_state(w, st)
        w.advance(DAY)
    n = len({f["argv"][f["argv"].index("--source") + 1] for f in fetches(w)})
    check("10 collects on different sources do not pause the campaign", n >= 10 and not w.config().get("paused_reason"),
          (n, w.config().get("paused_reason")))
    w2 = data_world("s15b")
    w2.candidates(cands[:1])
    w2.tick()
    st = w2.state()
    st["sources"]["src00"] = {"status": "candidate", "attempts": 3}
    _write_state(w2, st)
    w2.tick(2)
    check("a 4th attempt on one source pauses (stop-loss)", "4th collection attempt" in str(w2.config().get(
        "paused_reason")), w2.config().get("paused_reason"))
    w3 = data_world("s15c")
    w3.candidates(cands[:1])
    w3.tick()
    st = w3.state()
    st["sources"]["src00"] = {"status": "admitted", "partial": True, "attempts": 3, "yield_recorded": True,
                              "admit_done_utc": W.utc(w3.t[0])}
    _write_state(w3, st)
    w3.tick(2)
    rev = w3.state().get("source_reviews") or {}
    check("a sharded source whose 3 attempts are used, shards remaining: filed for a person (L16R), no pause and no "
          "4th automatic attempt", "src00" in rev and not w3.config().get("paused_reason") and not fetches(w3),
          (rev, w3.config().get("paused_reason")))
    # a day's job allowance counts jobs: lab -> cluster syncs (no job, no download) do not use it. Since
    # the 2026-10-04 amendment stream_levers.json declares no per-day count; the mechanism stays for one
    # that is declared, so the check declares six jobs a day here
    now = w3.clock()
    ex = [{"campaign": NAME, "lever": "L16S", "action": "inc_stream_sync", "status": "executed", "epoch": now - 60,
           "params": {"source": "s%d" % i}} for i in range(8)]
    jobs = [{"campaign": NAME, "lever": "L16", "action": "inc_stream_collect", "status": "executed", "epoch": now - 60,
             "params": {"source": "j%d" % i, "max_bytes": 1000}} for i in range(6)]
    req16 = {"action": "inc_stream_collect", "params": {"source": "new", "max_bytes": 1000}}
    orig, orig_lim = X.executions, LS.limits
    try:
        X.executions = lambda ctx=None: ex + jobs
        why_none = X.stream_limits(w3.xctx, {"name": NAME}, "L16", req16)
        LS.limits = lambda lid, menu=None: dict(orig_lim(lid, menu), jobs_per_day=6)
        X.executions = lambda ctx=None: ex
        why = X.stream_limits(w3.xctx, {"name": NAME}, "L16", req16)
        X.executions = lambda ctx=None: ex + jobs
        why6 = X.stream_limits(w3.xctx, {"name": NAME}, "L16", req16)
    finally:
        X.executions, LS.limits = orig, orig_lim
    check("  no per-day job count by default: eight syncs and six fetch jobs today leave a 7th fetch free", not why_none,
          why_none)
    check("  eight syncs today leave a declared six-jobs-a-day allowance untouched", not why, why)
    check("  six fetch jobs today use it up", any("in the last 24 h (limit 6)" in x for x in why6), why6)


# ----------------------------------------------------------------------- S16
def s16():
    w = train_world("s16")
    w.tick(1)
    ctx = X.Context(slurm_sh=w, resources=W.RES, clock=w.clock, lab_repo=str(w.lab), domain="weed",
                    diagnoses=[{"id": "D22", "fired": True, "levers": ["L18"],
                                "cites": [{"artifact": "campaign/context.json", "pointer": "/M", "value": w.M}]},
                               {"id": "D28", "fired": True, "levers": ["L24"],
                                "cites": [{"artifact": "campaign/context.json", "pointer": "/M", "value": w.M}]}])
    cite = [{"artifact": "campaign/context.json", "pointer": "/M", "value": w.M}]
    base = {"name": NAME, "mode": "stream", "autonomy": "envelope", "autonomy_granted_by": OWNER, "envelope_su": 1000.0,
            "daily_cap_su": 120.0, "window_cap_su": 350.0, "data_autonomy": "on", "last_milestone_pool": "P_0"}

    def req(lid, params, trigger=("D22",), est=None, n=0):
        w.advance(5)                  # one approval per request and instant (approvals.propose's id)
        if lid == "L18":
            params = dict(params, exp="%s_s00%d" % (w.sid, n + 1))
        p = LS.proposal(NAME, lid, params, trigger=list(trigger), cites=cite, est=est, attempt=n,
                        child_exp=params.get("exp") if lid == "L18" else None)
        p["lever"] = lid
        return p
    w._activate()
    col = {"source": "g1", "max_bytes": 1000}
    r = X.submit(req("L16", col, ("D20",), 4.0), campaign=dict(base, data_autonomy="off"), ctx=ctx)
    check("R2 data lever (L16), data_autonomy off: filed for a person, nothing ran", r["status"] == "filed", r["reasons"])
    r = X.submit(req("L16", dict(col, source="g2"), ("D20",), 4.0), campaign=base, ctx=ctx)
    check("R2 data lever, data_autonomy on, a stream replay pass, floors set: direct", r["status"] == "executed",
          r["reasons"])
    X.record_replay_result("pass", {k: v for k, v in W.all_pass_cases().items() if not k.startswith("S")}, ctx=ctx)
    r = X.submit(req("L16", dict(col, source="g3"), ("D20",), 4.0), campaign=base, ctx=ctx)
    check("  without the stream cases in the replay pass: filed", r["status"] == "filed"
          and any("stream" in x for x in r["reasons"]), r["reasons"])
    r = X.submit(req("L18", {"pkg": "inc2", "stream": w.sid, "k": 2}, est=20.0, n=1), campaign=base, ctx=ctx)
    check("  and an R3 build waits too (the envelope needs the stream replay)", r["status"] == "filed", r["reasons"])
    w.replay_pass()
    r = X.submit(req("L18", {"pkg": "inc2", "stream": w.sid, "k": 2}, est=20.0, n=2), campaign=base, ctx=ctx)
    check("R3 L18 with autonomy envelope, a replay pass and a fired D22 citing it: runs within the envelope",
          r["status"] == "executed" and r["basis"] == "envelope", r["reasons"])
    r = X.submit(req("L18", {"pkg": "inc2", "stream": w.sid, "k": 2}, est=20.0, n=3),
                 campaign=dict(base, autonomy="off"), ctx=ctx)
    check("R3 L18 with autonomy off: filed for a person", r["status"] == "filed", r["reasons"])
    r = X.submit(req("L23", {"pkg": "inc2", "verb": "build"}, ("DR0",), 4.0), campaign=base, ctx=ctx)
    check("R3 L23 (the splits build): a person, even within the envelope", r["status"] == "filed", r["reasons"])
    r = X.submit(req("LH", {"pkg": "inc2", "stream": w.sid, "hold": "funnel_F9"}, ("DHOLD",)), campaign=base, ctx=ctx)
    check("R3 LH (a funnel_F9 release): a person", r["status"] == "filed", r["reasons"])
    r = X.submit(dict(req("L18", {"pkg": "inc2", "stream": w.sid, "k": 2}, est=20.0, n=4), risk="R4"), campaign=base,
                 ctx=ctx)
    check("an R4 request is refused (a card only)", r["status"] == "refused" and "R4" in " ".join(r["reasons"]),
          r["reasons"])
    r = X.submit(req("L18", {"pkg": "inc2", "stream": w.sid, "k": 2}, est=20.0, n=5), actor="tier2:somemodel",
                 campaign=base, ctx=ctx)
    check("a brain (tier2) may not request a stream build", r["status"] == "refused", r["reasons"])
    r = X.submit(req("L24", {"pkg": "inc2", "source": "g2", "cite": "D31"}, ("D31",)), campaign=base, ctx=ctx)
    check("L24 without a firing D28 or D31: filed for a person", r["status"] == "filed", r["reasons"])
    r = X.submit(req("L24", {"pkg": "inc2", "source": "g2", "cite": "D28"}, ("D28",)), campaign=base, ctx=ctx)
    check("L24 citing a firing D28: direct (R2, gated)", r["status"] == "executed", r["reasons"])
    r = X.submit(req("LP", {}, ("DR0",), 0.5), campaign=dict(base, data_autonomy="off"), ctx=ctx)
    check("R0 (the network probe): direct whatever data_autonomy says", r["status"] == "executed", r["reasons"])
    w.segment(1, [("I1", "ACCEPT", 0.9, [], "helps", ["a"])])
    r = X.submit(req("L19", {"pkg": "inc2", "exp": "%s_s001" % w.sid}, ("D23",)), campaign=base, ctx=ctx)
    check("R1 (the commit): direct", r["status"] == "executed", r["reasons"])
    r = X.submit(req("L16", dict(col, source="g9"), ("D20",), 4.0), actor=OWNER, campaign=dict(base, data_autonomy="off"),
                 ctx=ctx)
    check("a person may run the R2 data lever directly", r["status"] == "executed", r["reasons"])
    ws = data_world("s16s", data_autonomy="off")
    ws.candidates([{"id": "src_s", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["PricklySida"],
                    "bytes": 1e9, "expected_target_boxes": 800, "known_item": True}])
    ws.tick(4)
    it = ws.lane("DATA").get("item") or {}
    check("shadow mode: the L16 is filed for a person and nothing is fetched",
          it.get("lever") == "L16" and it.get("status") == "filed" and not fetches(ws), (it, fetches(ws)))
    ws.set_config(data_autonomy="on")
    ws.tick(3)
    check("  once a person sets data_autonomy on, the filed L16 is re-checked and runs (the lane does not wait "
          "for ever on the old filing)", len(fetches(ws, "src_s")) == 1, [x["argv"] for x in fetches(ws)])


# ----------------------------------------------------------------------- S17
def s17():
    w = train_world("s17")
    w.es.update(accepted_since=4, segments_since=1, first_accept_utc=W.utc(w.t[0] - DAY))
    w.queue(4 * w.M, boxes={"Purslane": 100}, oldest_utc=W.utc(w.t[0] - 2 * DAY), consumed={"Purslane": 900})
    w.candidates([{"id": "src_p", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 500}])
    busy = []
    for i in range(12):
        w.tick()
        st = w.state()
        busy.append(sum(1 for ln in S.LANES if (st["lanes"][ln].get("item") or {}).get("status") == "running"))
        for n in ("inc_stream_collect_fetch_src_p",):
            if any(j["name"] == n for j in w.squeue) and i > 6:
                w.job_done(n)
    check("every tick of the three-lane run made at most one ssh", w.max_calls() <= 1, w.max_calls())
    lv = {e.get("lever") for e in w.events("executed")}
    check("the three lanes each ran an item: TRAIN L18, DATA L16, MAINT L20 (D24: 4 accepted since the last milestone)",
          {"L18", "L16", "L20"} <= lv, lv)
    l20 = [e for e in w.events("proposed") if e.get("lever") == "L20"]
    check("  L20 came from D24", l20 and l20[0]["trigger"] == ["D24"], l20)
    check("  three lanes were in flight at once", max(busy) >= 3, busy)


# ----------------------------------------------------------------------- S18
def s18():
    w = train_world("s18")
    w.tick(2)
    exp = w.segment(1, [("I1", "ACCEPT", 0.9, [], None, ["a"])], done=False)
    w.tick(3)
    adv = [a for a in w.advances if a["exp"] == exp]
    check("the ticker advances its v2 segment with INC_JOB_SCRIPT = run_inc2_job.sh",
          adv and all(a["job_script"] == str(w.scripts / "run_inc2_job.sh") for a in adv), adv)
    check("  and never with the v1 script", not [a for a in w.advances if str(a["job_script"]).endswith("run_inc_job.sh")])
    (w.scripts / "run_inc2_job.sh").unlink()
    n = len(w.advances)
    w.tick(2)
    check("without run_inc2_job.sh nothing is advanced (fail closed; the v1 executor is never used)",
          len(w.advances) == n, w.advances[n:])
    snap = json.loads(S.StreamPaths(str(w.lab), "weed").latest(NAME).read_text())
    a = ((snap.get("experiments") or {}).get(exp) or {}).get("advance") or {}
    check("  the snapshot says why", a.get("ok") is False and "run_inc2_job.sh" in str(a.get("error")), a)


# ----------------------------------------------------------------------- S19
def s19():
    ev = E.load_dir(FIX, "pilot_v3", exps=["pilot_v3"])
    rows = DS.steps(ev, "pilot_v3", "full")
    disp = {r["step"]: r["disposition"] for r in rows}
    check("pilot_v3's full chain: the planted Bswap (P_data 0.00, blame 'recipe' in the ledger) is disposed 'data'",
          disp.get("Bswap") == "data" and next(r for r in rows if r["step"] == "Bswap")["blame"] == "recipe", disp)
    check("  and Breal is disposed 'species' (returned once by the commit)", disp.get("Breal") == "species", disp)
    for ln, e in ev.ledger("pilot_v3"):
        if e.get("type") == "gate":
            a = e["decision"].setdefault("attribution", {})
            a["blame"] = {"recipe": "data", "data": "recipe", None: "data"}.get(a.get("blame"), "data")
    rows2 = DS.steps(ev, "pilot_v3", "full")
    check("the disposition never reads attribution.blame (every blame flipped: the same dispositions)",
          {r["step"]: r["disposition"] for r in rows2} == disp, {r["step"]: r["disposition"] for r in rows2})
    ctx = {"now_utc": "2026-10-01T00:00:00Z", "sid": "x", "M": 682,
           "lanes": {ln: {"phase": "IDLE"} for ln in S.LANES}, "stage": {"stage_a_ready": False}}
    full = [x for x in (FIX / "pilot_v3" / "ledger.jsonl").read_text().splitlines()
            if x.strip() and (json.loads(x).get("type") != "gate" or json.loads(x).get("chain") == "full")]
    prior = E.from_texts({"pilot_v3/ledger.jsonl": "\n".join(full) + "\n"}, "pilot_v3")
    by = DS.by_id(DS.detect(E.from_texts({}, "x", context=ctx), LS.load_domain("weed"), LS.load_thresholds(),
                            prior=prior, include_prior_d31=True))
    blamed = [b["step"] for b in by["D31"]["detail"].get("blamed") or []]
    check("read as the pinned prior: D31 blames Bswap (data) and proposes L4", "Bswap" in blamed and "L4" in by["D31"]["levers"],
          (blamed, by["D31"]["summary"]))
    check("  D33 names the species both REJECTs failed on (PalmerAmaranth, 2 of 2)",
          by["D33"]["fired"] and "PalmerAmaranth" in by["D33"]["detail"]["species"], by["D33"]["summary"])


# ----------------------------------------------------------------------- S20
def _zero_run(w, srcs):
    st = w.state()
    for sname in srcs:
        st["sources"][sname] = {"status": "admitted", "batch": "b_%s" % sname, "admit_done_utc": W.utc(w.t[0]),
                                "attempts": 1}
    _write_state(w, st)
    per = {sname: {"target_boxes_admitted": 0} for sname in srcs}
    old = json.loads((w.inc / "step1_stream" / "status.json").read_text())
    old["per_source"].update(per)
    w._w("step1_stream/status.json", old)
    for sname in srcs:
        w.intake("b_%s" % sname, sname, images=0, target_boxes=0, reasons={"no_target_class": 40, "near_eval_v2": 10})
    w.tick(2)


def s20():
    w = data_world("s20")
    w.tick()
    _zero_run(w, ["z1", "z2"])
    check("two consecutive zero-yield sources do not hold the DATA lane", not w.lane("DATA").get("hold"),
          w.lane("DATA"))
    _zero_run(w, ["z3"])
    ln = w.lane("DATA")
    check("the third holds it (the silent '+0' rounds of 2026-08/09 cannot recur)",
          "3 consecutive sources" in str(ln.get("hold")), ln)
    card = [c for c in w.state()["cards"] if c.get("kind") == "stop_loss"]
    check("  with a card listing each source's decision reasons",
          card and all(x in card[-1]["detail"] for x in ("z1", "z2", "z3", "no_target_class 40")),
          card and card[-1]["detail"])
    w.tick(2)
    check("  the hold stays while nobody acts", "3 consecutive sources" in str(w.lane("DATA").get("hold")),
          w.lane("DATA"))
    S.configure_stream(NAME, OWNER, enable=True, cfg_hooks=w.hooks, lab_repo=str(w.lab), clock=w.clock)
    w.tick(2)
    check("  a person's resume (stream enable) releases the held lane, with no pause needed",
          not w.lane("DATA").get("hold") and not w.config().get("paused_reason")
          and any(e.get("lane") == "DATA" for e in w.events("lane_released")), w.lane("DATA"))
    w2 = data_world("s20b")
    w2.tick()
    _zero_run(w2, ["y1", "y2"])
    st = w2.state()
    st["sources"]["y3"] = {"status": "admitted", "batch": "b_y3", "admit_done_utc": W.utc(w2.t[0])}
    _write_state(w2, st)
    old = json.loads((w2.inc / "step1_stream" / "status.json").read_text())
    old["per_source"]["y3"] = {"target_boxes_admitted": 57}
    w2._w("step1_stream/status.json", old)
    w2.tick(2)
    _zero_run(w2, ["y4"])
    check("a source with yield in between resets the run", not w2.lane("DATA").get("hold"), w2.lane("DATA"))


# ----------------------------------------------------------------------- S21
def s21():
    w = data_world("s21")
    w.candidates([{"id": "src_q", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 500}])
    w.qos = True
    w.tick(4)
    sub = fetches(w)
    check("the collect job names GPU-shared, never RM-shared", sub and all("GPU-shared" in x["argv"] and
                                                                         "RM-shared" not in x["argv"] for x in sub),
          [x["argv"] for x in sub])
    check("  an 'Invalid qos' refusal is submitted once, never retried there", len(sub) == 1, len(sub))
    ev = w.events("platform_defect")
    check("  recorded as a platform defect (error_kind qos) with a card", ev and ev[0].get("error_kind") == "qos"
          and any(c.get("kind") == "platform" for c in w.state()["cards"]), ev)
    src = (w.state()["sources"].get("src_q") or {})
    ex = [e for e in w.executions() if e.get("lever") == "L16"]
    check("  not a failed attempt of the source: no attempt counted, nothing charged",
          not src.get("attempts") and not src.get("failures") and not any(e.get("charged") for e in ex), (src, ex[-1:]))


# ----------------------------------------------------------------------- S22
class _SlowRunner(S.LabRunner):
    def launch(self, job, argv, timeout=0):
        return S.LabRunner.launch(self, job, [sys.executable, "-c", "import time; time.sleep(1.5)"], timeout=60)


def s22():
    w = data_world("s22")
    w.placement({"hf": "pass", "ftp": "fail"})
    w.candidates([{"id": "gh_src", "provider": "github", "licence": "MIT", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 900},
                  {"id": "ftp_src", "provider": "ftp", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 100}])
    runner = _SlowRunner(w.dir / "labjobs")
    S.LAB_RUNNER = lambda run: runner
    try:
        w.tick()
        t0 = time.time()
        w.tick()
        took = time.time() - t0
        st = w.state()
        it = st["lanes"]["DATA"].get("item") or {}
        check("a provider not passed in placement.json (github: lab-only) gets the lab hook, never an sbatch",
              it.get("lever") == "L16L" and not fetches(w), (it.get("lever"), [x["argv"] for x in fetches(w)]))
        check("  the lab fetch runs detached: the tick returned in %.2f s while it runs" % took,
              took < 1.2 and it.get("status") == "running", (took, it.get("status")))
        res = None
        for _ in range(60):
            res = runner.poll(it.get("lab_job"))
            if res:
                break
            time.sleep(0.1)
        check("  its result file appears when the process ends", res and res.get("ok"), res)
        w.tick()
        st = w.state()
        check("  a later tick folds it: the source is fetched, the sync (L16S) comes next on the lab",
              st["sources"]["gh_src"]["status"] == "fetched" and (st["lanes"]["DATA"].get("item") or {}).get("lever")
              == "L16S", (st["sources"].get("gh_src"), (st["lanes"]["DATA"].get("item") or {}).get("lever")))
    finally:
        S.LAB_RUNNER = lambda run, w=w: w.runner
    # a source whose class names L26 resolved on the lab, collected on the
    # cluster: the collector there reads the names layer offline, so the lab
    # pushes it (L16S --names) before the fetch
    wn = data_world("s22n")
    wn.placement({"hf": "pass"})
    wn.candidates([{"id": "hf_names", "provider": "hf", "licence": "CC BY 4.0", "target_classes": [],
                    "bytes": 1e9, "expected_target_boxes": 900, "names_pending": ["Sida spinosa L."],
                    "decision": {"status": "pending_names"}}])
    wn.tick(2)
    l26 = [x for x in wn.runner.launched if "names" in x["argv"] and "collect" in " ".join(x["argv"])]
    check("pending names: L26 resolves them on the lab first (detached), in the lab's INC tree",
          l26 and all(x["argv"][x["argv"].index("--inc-dir") + 1] == str(S.StreamPaths(str(wn.lab), "weed").lab_inc)
                      for x in l26 if "--inc-dir" in x["argv"]) and all("--inc-dir" in x["argv"] for x in l26),
          [x["argv"] for x in wn.runner.launched])
    wn.runner.finish()
    wn.tick(4)
    syn = [x for x in wn.runner.launched if "lab-sync" in x["argv"]]
    order = [("names" if "--names" in x["argv"] else "candidate" if "--file" in x["argv"] else "staging") for x in syn]
    check("  then the names layer is pushed to the cluster (L16S --names) before the cluster fetch",
          order[:1] == ["names"] and fetches(wn, "hf_names")
          and wn.state()["sources"]["hf_names"].get("names_synced"), (order, [x["argv"] for x in fetches(wn)]))


# ----------------------------------------------------------------------- S23
def s23():
    w = train_world("s23")
    lab = SR.module_hashes()
    bad = dict(lab)
    k = sorted(bad)[0]
    bad[k] = "0" * 64
    w.code_override = bad
    w.tick(2)
    st = w.state()
    check("the snapshot returns the cluster's module hashes; a mismatch with the lab's is recorded",
          st.get("drift") == [k], st.get("drift"))
    check("  with a card", any(c.get("kind") == "drift" for c in st["cards"]))
    w.tick(3)
    check("  and every stream submission is refused while it stands (only snapshots)",
          not w.submits and all(v[0] == "stream-snapshot" for v in w.verbs), [v[0] for v in w.verbs])
    w.code_override = None
    w.tick(3)
    check("the modules agree again: the drift clears and the build runs", not w.state().get("drift")
          and any("run_inc2_build.sh" in " ".join(x["argv"]) for x in w.submits), [x["argv"][-4:] for x in w.submits])


# ----------------------------------------------------------------------- S24
def s24():
    w = train_world("s24")
    w.tick(2)
    exp = "%s_s001" % w.sid
    why = ("nothing cut for %s: no exact fill of M = %d from %d eligible images in 9 units: a capture group larger "
           "than the remainder" % (exp, w.M, 4 * w.M))
    # run_inc2_build.sh writes every inc2.stream build's provenance as stream_<sid>.json (its last attempt is
    # this build's job), never under the segment's name
    w._w("_campaign/provenance/stream_%s.json" % w.sid, {"exp": "stream_%s" % w.sid, "attempts": [
        {"job_id": str(w.next_job), "status": "build_failed", "started_utc": "x", "refusal": "[inc2.stream] ERROR: " + why}]})
    w.job_done("inc_build_%s" % exp, state="FAILED")
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY),
            probe={"exact_fill": False, "why": "no exact fill of M = %d" % w.M})
    w.tick(4)
    d = _diag(w)["D22"]
    check("a cutter that holds Q >= M and cannot fill raises D22 with the refusal (never a silent wait)",
          d["fired"] and "cannot fill" in d["summary"] and "no exact fill" in d["summary"]
          and "OP_ESCALATE" in d["levers"], d["summary"])
    check("  a person reads it: an escalation card; TRAIN holds", any(c.get("kind") == "escalation" for c in
                                                                     w.state()["cards"])
          and "D22" in str(w.lane("TRAIN").get("diag_hold")), (w.lane("TRAIN"), w.state()["cards"][-2:]))
    check("  and the build is not submitted again meanwhile",
          len([x for x in w.submits if "run_inc2_build.sh" in " ".join(x["argv"])]) == 1)
    w.queue(5 * w.M, boxes={"Purslane": 1100}, oldest_utc=W.utc(w.t[0] - 2 * DAY), probe={"exact_fill": True})
    w.tick(3)
    d = _diag(w)["D22"]
    check("once the cutter's probe fills exactly M again, the old refusal is superseded: TRAIN is not held and "
          "L18 is proposed again", d["fired"] and "L18" in d["levers"] and "D22" not in str(w.lane("TRAIN").get("diag_hold"))
          and len(l18_argv(w)) >= 2, (d["summary"], w.lane("TRAIN")))


# ----------------------------------------------------------------------- S25
def s25():
    w = data_world("s25")
    w.step1_status(per_source={"rf_reupload": {"images_seen": 200, "near_eval_embed": 14}})
    w.candidates([{"id": "rf_sida_reupload", "provider": "roboflow", "licence": "CC BY 4.0",
                   "target_classes": ["PricklySida", "Sicklepod", "Purslane"], "bytes": 1e8,
                   "expected_target_boxes": 5000, "lab_group": "LuLab"}])
    w.tick(3)
    d = _diag(w)["D28"]
    check("augmented copies caught by near_eval_embed (7 %% of the source's images) -> D28 -> L24 and a card",
          d["fired"] and any(h["source"] == "rf_reupload" for h in d["detail"]["leaks"]), d["summary"])
    check("  L24 ran on it", any("rf_reupload" in r for r in w.runs), w.runs)
    rev = w.state().get("source_reviews") or {}
    check("a re-upload declaring PricklySida (a presumed LuLab derivative) is never collected automatically: "
          "a person decides (held for the copy scan)", "rf_sida_reupload" in rev and not fetches(w, "rf_sida_reupload"),
          rev)
    wa = train_world("s25a")
    wa.step1_status(per_source={"rf_leak": {"images_seen": 200, "near_eval_embed": 14}})
    wa.tick(4)
    q_at = next((k for k, r in enumerate(wa.runs) if "quarantine" in r and "rf_leak" in r), None)
    seg = l18_argv(wa)
    check("with Q >= 4M and a leaking source, the quarantine (L24) runs and no segment is cut before it",
          q_at is not None and seg and wa.ledger().index(seg[0]) > next(
              k for k, e in enumerate(wa.ledger()) if e.get("event") == "executed" and e.get("lever") == "L24"),
          (wa.runs, [e.get("lever") for e in wa.events("proposed")]))
    ws = train_world("s25s", data_autonomy="off")
    ws.step1_status(per_source={"rf_leak": {"images_seen": 200, "near_eval_embed": 14}})
    ws.tick(4)
    stop = ws.lane("STOP").get("item") or {}
    check("shadow mode: the leak's quarantine waits for a person, and TRAIN holds meanwhile (nothing is cut while "
          "the leaking source's rows are still eligible)", stop.get("lever") == "L24" and stop.get("status") == "filed"
          and not l18_argv(ws) and "D28" in str(ws.lane("TRAIN").get("diag_hold")), (stop, ws.lane("TRAIN")))
    AP.decide("weed", stop["approval_id"], "deny", OWNER, "not a copy source", ws.clock(), root=str(ws.lab))
    ws.tick(3)
    check("  a person keeping the source (denying the quarantine) lifts the hold: the segment is cut",
          l18_argv(ws) and (ws.state()["sources"].get("rf_leak") or {}).get("leak_kept_by") == OWNER,
          (ws.lane("TRAIN"), ws.state()["sources"].get("rf_leak")))


# ----------------------------------------------------------------------- S26
def s26():
    w = data_world("s26")
    w.queue(0, held={"funnel_F9": {"rows": 457, "past_deadline": 457}, "h6_scan": {"rows": 120, "past_deadline": 120}})
    w.tick(2)
    sc = [s for s in w.submits if "scan-holds" in s["argv"]]
    check("an h6_scan hold past its deadline is served by the stream's own copy detector (L17 scan-holds)",
          sc and sc[0]["argv"][-3:] == ["scan-holds", "--hold", "h6_scan"], [s["argv"][-3:] for s in w.submits])
    w.job_done("inc_stream_admit_scanholds")
    w.queue(0, held={"funnel_F9": {"rows": 457, "past_deadline": 457}})
    w.tick(3)
    lh = [e for e in w.events("filed") if e.get("lever") == "LH"]
    check("a funnel_F9 hold past its deadline becomes an R3 item for a person (release or keep)",
          lh and w.approvals()[lh[0]["approval_id"]]["risk"] == "R3", lh)
    check("  neither waits silently (both are in the ledger)", sc and lh)
    check("  the release waits for a person without holding the DATA lane",
          (w.lane("DATA").get("item") or {}).get("lever") != "LH", w.lane("DATA"))
    AP.decide("weed", lh[0]["approval_id"], "approve", OWNER, "release them", w.clock(), root=str(w.lab))
    w.runner.finish()                               # the discovery that holds the DATA lane ends
    w.tick(3)
    check("  approved, it is adopted into the DATA lane and run (inc2.stream release --hold funnel_F9)",
          any("release" in r and "funnel_F9" in r for r in w.runs), w.runs)
    w3 = data_world("s26c")
    w3.queue(0, held={"funnel_F9": {"rows": 457, "past_deadline": 457}})
    w3.candidates([{"id": "src_c", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"],
                    "bytes": 1e9, "expected_target_boxes": 500, "known_item": True}])
    w3.tick(4)
    check("while a funnel_F9 release waits for a person, collection goes on (L16 on the open candidate)",
          fetches(w3, "src_c") and [e for e in w3.events("filed") if e.get("lever") == "LH"],
          ([x["argv"] for x in fetches(w3)], w3.lane("DATA")))


# ----------------------------------------------------------------------- S27
def s27():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="s27_", dir=str(W.TMP)))
    old_root = X.CODE_ROOT
    try:
        for rel in X.GOVERNANCE_FILES:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_text("v1\n")
        X.CODE_ROOT = tmp
        h1 = X.code_hash()
        (tmp / "weed_optimizer_framework/tools/collect/domains/weed.json").write_text('{"known_items": []}\n')
        check("a change to collect/domains/weed.json changes the code hash, so it voids the replay pass",
              X.code_hash() != h1 and "weed_optimizer_framework/tools/collect/domains/weed.json" in X.GOVERNANCE_FILES)
    finally:
        X.CODE_ROOT = old_root
    ap = PKG / "weed_optimizer_framework" / "tools" / "inc_autopilot"
    watched = [ap / "stream_thresholds.json", ap / "thresholds.json", ap / "stream_levers.json", ap / "levers.json",
               ap / "stream_domains" / "weed.json"]
    before = {str(p): p.read_bytes() for p in watched}
    w = data_world("s27")
    before[str(w.collect_cfg)] = w.collect_cfg.read_bytes()
    w.candidates([{"id": "g", "provider": "hf", "licence": "CC BY 4.0", "target_classes": ["Purslane"], "bytes": 1e9,
                   "expected_target_boxes": 50}])
    w.tick(4)
    after = {k: pathlib.Path(k).read_bytes() for k in before}
    check("the platform never writes the collector's config, floor_gb / floor_su, thresholds.json or the stream "
          "policy files", after == before, [k for k in before if before[k] != after[k]])
    src = "".join(p.read_text() for p in _domain_free_files())
    check("  and no stream module opens them for writing", not re.search(
        r"(stream_thresholds|thresholds|levers|stream_levers)\.json[^\n]*['\"]w", src))


# ----------------------------------------------------------------------- S29
def _maint(w):
    it = w.lane("MAINT").get("item") or {}
    return it, (it.get("proposal") or {}).get("id")


def s29():
    """The live incident of 2026-09-28 (a stream campaign): the MAINT lane's
    L23 splits build ran after a person approved it and its job ended FAILED;
    the lane ticker proposed L23 again under the SAME proposal id (the attempt
    was counted under one key and read under another: the params with and
    without the price), so the executor refused it as a changed proposal under
    a filed id and the lane was held. Now: a new id on every re-proposal,
    filed for a person again, nothing refused; the step is proposed again at
    most stop_loss step_retries times, then its lane waits for a person, whose
    release (stream release) frees it."""
    w = World("s29")
    w.step1_status()
    w.placement({"hf": "pass"})
    w.tick(2)
    it, first = _maint(w)
    check("no LOCK: L23 (the splits build) is proposed in MAINT and filed for a person (R3)",
          it.get("lever") == "L23" and it.get("status") == "filed"
          and w.approvals()[it["approval_id"]]["status"] == "pending", it)
    AP.decide("weed", it["approval_id"], "approve", OWNER, "run the splits build", w.clock(), root=str(w.lab))
    w.tick(2)
    it, _ = _maint(w)
    check("  approved, it runs as a job", it.get("status") == "running" and it.get("job_ids"), it)
    w.job_done("inc_build_splits", "FAILED")
    w.tick(2)
    it, second = _maint(w)
    refused = [r for r in w.executions() if r.get("lever") == "L23" and r.get("status") == "refused"]
    check("its job ended FAILED: L23 is proposed again under a NEW proposal id",
          it.get("lever") == "L23" and second and second != first, (first, second))
    check("  and filed for a person again (a pending approval of the new id), not refused as a changed proposal",
          it.get("status") == "filed" and w.approvals()[it["approval_id"]]["status"] == "pending"
          and (w.approvals()[it["approval_id"]].get("context") or {}).get("proposal_id") == second
          and not refused, (it.get("status"), [r.get("reasons") for r in refused]))
    st = w.state()
    key = S.step_key("L23", it["proposal"]["params"])
    check("  the lane is not held; the failed id is recorded, and the step's failed runs are 1",
          not w.lane("MAINT").get("hold") and first in st["failed_ids"]
          and st["step_failures"][key]["n"] == 1, (w.lane("MAINT").get("hold"), st.get("step_failures")))
    AP.decide("weed", it["approval_id"], "approve", OWNER, "again", w.clock(), root=str(w.lab))
    w.tick(2)
    w.job_done("inc_build_splits", "FAILED")
    w.tick(2)
    hold = w.lane("MAINT").get("hold")
    check("a second consecutive failure holds MAINT (contract 6.6, 2 consecutive failed steps), with its card",
          "2 consecutive failed steps" in str(hold)
          and any(c.get("title") == "Lane MAINT held" for c in w.state()["cards"]), hold)
    # the live campaign's state was written before failed_ids existed: the
    # ids of its failed items are taken from the campaign ledger instead
    st = w.state()
    for k in ("failed_ids", "attempts", "step_failures"):
        st.pop(k, None)
    _write_state(w, st)
    failed = sorted({e.get("proposal_id") for e in w.events("failed") if e.get("lever") == "L23"})
    out = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.stream",
                          "--config", str(w.cfg), "--lab-repo", str(w.lab), "release", "--name", NAME,
                          "--by", OWNER], capture_output=True, text=True, cwd=str(PKG), timeout=300,
                         env=dict(os.environ, PYTHONPATH=str(PKG)))
    rel = json.loads(out.stdout or "{}") if out.returncode == 0 else {}
    check("a person releases the lane with the documented CLI (stream release --name N --by human:<email>)",
          out.returncode == 0 and rel.get("ok") and "MAINT" in (rel.get("held_lanes") or {}),
          (out.returncode, out.stdout[-300:], out.stderr[-600:]))
    # the CLI stamps the resume with the wall clock, and this world's clock runs
    # ahead of it: the stamp that frees the lane here is the world's, through
    # the function the CLI calls
    S.release(NAME, OWNER, cfg_hooks=w.hooks, lab_repo=str(w.lab), clock=w.clock)
    w.tick(2)
    it, third = _maint(w)
    check("  the next tick frees MAINT (lane_released) and L23 is filed again under an id no failed item had "
          "(read from the ledger for a state written before the record existed)",
          not w.lane("MAINT").get("hold") and any(e.get("lane") == "MAINT" for e in w.events("lane_released"))
          and it.get("status") == "filed" and third and third not in failed and len(failed) == 2,
          (w.lane("MAINT").get("hold"), third, failed))
    # the step bound: a step whose failures were not consecutive (another item
    # of the lane succeeded in between) is proposed again at most step_retries
    # times; the state below is that of its third failed run
    retries = int(LS.t(LS.load_thresholds(), "stop_loss", "step_retries"))
    st = w.state()
    st["lanes"]["MAINT"].update(item=None, phase="IDLE", fails=0)
    st["failed_ids"] = list(st.get("failed_ids") or []) + [third]
    st["step_failures"] = {key: {"n": retries + 1, "lever": "L23", "lane": "MAINT", "last": "job(s) 9 ended FAILED",
                                 "utc": W.utc(w.t[0]), "proposal_id": third}}
    _write_state(w, st)
    w.tick(2)
    hold = w.lane("MAINT").get("hold")
    check("after %d failed runs of one step it is not proposed again: MAINT waits for a person, the hold naming "
          "the release command" % (retries + 1),
          not w.lane("MAINT").get("item") and "L23 failed %d times" % (retries + 1) in str(hold)
          and "stream release --name %s" % NAME in str(hold), (hold, _maint(w)[0].get("status")))
    S.release(NAME, OWNER, cfg_hooks=w.hooks, lab_repo=str(w.lab), clock=w.clock)
    w.tick(2)
    it, fourth = _maint(w)
    check("  a person's release clears the lane's step counts: L23 is filed again under a fresh id",
          not w.lane("MAINT").get("hold") and it.get("status") == "filed" and fourth not in (first, second, third)
          and not w.state().get("step_failures"), (w.lane("MAINT").get("hold"), fourth))
    check("every tick made at most one ssh", w.max_calls() <= 1, w.max_calls())


# ----------------------------------------------------------------------- S28
def s28():
    sys.path.insert(0, str(PKG / "tests"))
    import test_round_scheduler_stream_guard as G
    res = G.run_checks(check)
    check("the round scheduler, dashboard hook and Roboflow sync checks ran", res, res)


# ----------------------------------------------------------------------- prospective
def s_prospective():
    w = train_world("sp")
    w.tick(2)
    ps = w.events("prospective_stream")
    check("before the first L18: one prospective stream record, its sha256 in the ledger", len(ps) == 1, ps)
    rec = json.loads(pathlib.Path(ps[0]["path"]).read_text())
    check("  it carries M, K, the truth policy, the thresholds' sha256, the recipe rule, the gate block and the "
          "rules version", rec["M"] == w.M and rec["K_max"] == 4 and rec["truth"] and len(rec["thresholds_sha256"]) == 64
          and rec["recipe_rule"]["stage_a"]["recipes"] == "r0,x1a" and rec["gate"]["flips_mode"] == "net"
          and rec["rules_version"] == DS.rules_version(), rec)
    it = [e for e in w.events("proposed") if e.get("lever") == "L18"][0]
    check("  written before the L18 proposal", w.ledger().index(ps[0]) < w.ledger().index(it))
    w.tick(3)
    check("  and written once per rules version", len(w.events("prospective_stream")) == 1)
    # D8S -> L22 (a fork doubles M under a new stream version) -> the new
    # stream's first L18 gets its own prospective record (M changed)
    wf = train_world("spf")
    wf.tick(2)
    rows = [("H1", "HOLD", 0.5, [], "neutral", ["a"]), ("H2", "HOLD", 0.5, [], "neutral", ["a"])]
    wf.segment(1, rows, done=True)
    wf.tick(6)
    forks = [x for x in wf.submits if "fork" in x["argv"]]
    check("underpowered steps (P_data in (0.25, 0.75), |cand - null| < 2 sd) -> D8S -> L22 forks with M doubled",
          forks and forks[0]["argv"][forks[0]["argv"].index("--m") + 1] == str(2 * wf.M),
          ([x["argv"][-6:] for x in wf.submits], [e.get("lever") for e in wf.events("proposed")]))
    # what inc2.stream fork writes before its job ends: the old summary names
    # the new stream, which exists with its own ledger (init) and summary
    old_sid, new_sid = wf.sid, wf.sid + "_fork%d" % (2 * wf.M)
    wf.es["forked_to"] = new_sid
    wf._summary()
    wf.sid, wf.M = new_sid, 2 * wf.M
    wf.es.update(forked_to=None, next_seg=1, uncommitted_done=[], segments={})
    wf.stream_init(("r0", "x1a"))
    wf.queue(4 * wf.M, boxes={"Purslane": 900}, oldest_utc=W.utc(wf.t[0] - 2 * DAY))
    wf.job_done("inc_build_stream_fork")
    wf.tick(2)
    check("  the fork is adopted in the fold that sees its job end: the campaign runs the new stream version, and no "
          "second L22 was proposed on the old stream's evidence", (wf.config().get("stream") or {}).get("sid") == new_sid
          and len([e for e in wf.events("proposed") if e.get("lever") == "L22"]) == 1
          and not wf.lane("TRAIN").get("fails"), (wf.config().get("stream"), wf.lane("TRAIN")))
    fa = [e for e in wf.ledger() if e.get("event") == "fork_adopted"]
    dn = [e for e in wf.ledger() if e.get("event") == "item_done" and e.get("lever") == "L22"]
    check("  adopted in the very fold that finished the L22 item", fa and dn and fa[0].get("tick") == dn[0].get("tick"),
          (fa and fa[0].get("tick"), dn and dn[0].get("tick")))
    # a fork whose summary names the new stream only one snapshot later: the
    # finished L22 is not proposed again on the evidence that predates it
    wd = train_world("spd")
    wd.tick(2)
    wd.segment(1, rows, done=True)
    wd.tick(6)
    wd.job_done("inc_build_stream_fork")
    wd.tick(3)
    check("  a finished L22 is never proposed again on evidence taken before its effect showed (no second L22, no "
          "failed step)", len([e for e in wd.events("proposed") if e.get("lever") == "L22"]) == 1
          and not wd.lane("TRAIN").get("fails"), ([e.get("lever") for e in wd.events("proposed")], wd.lane("TRAIN")))
    wf.advance(DAY)                                # (was: the daily L18 limit, removed 2026-10-04)
    wf.tick(6)
    check("  the forked stream adopts the capacity decision itself (LA: its init carries no arm line)",
          any("choose-arm" in r and new_sid in r for r in wf.runs), [r[3:] for r in wf.runs])
    ps = wf.events("prospective_stream")
    check("  and before the new stream's first L18 a prospective record of its own (its sid, M doubled)",
          len(ps) == 2 and ps[-1]["M"] == wf.M and new_sid in ps[-1]["path"]
          and any(e.get("child_exp") == "%s_s001" % new_sid for e in l18_argv(wf)),
          ([(e.get("M"), e.get("path")) for e in ps], [e.get("child_exp") for e in l18_argv(wf)]))


def _step(w, lever, simulate=None, ticks=8):
    """Tick until lever `lever` executes once more; then simulate its effect on
    the cluster (what the job or the other group's verb writes). Returns the
    proposal ledger line, or None."""
    n0 = len([e for e in w.events("executed") if e.get("lever") == lever])
    for _ in range(ticks):
        w.tick()
        ex = [e for e in w.events("executed") if e.get("lever") == lever]
        if len(ex) > n0:
            if simulate:
                simulate()
            w.advance(DAY)                         # a new day (no daily cap binds since 2026-10-04 anyway)
            return [e for e in w.events("proposed") if e.get("lever") == lever][-1]
    return None


def s_r0():
    """The rollout (contract 10 R0, R0b, R2's prerequisites) driven by the
    platform itself, step by step, against the other groups' real argv
    grammars: baselines, the verdicts, Stage A, the stream's creation, the
    arm, Stage C, the measurement arms and their rescores, E1 (2026-10-03:
    base v3's build L23V, its two arms L23B --role baseline, their agnostic
    rescore L23E), E2 (2026-10-04: once E1's verdict qualifies E1-B, its six
    single-seed builds L23B --e2 W|S), E2-C (2026-10-04, later: its three
    builds L23B --e2 C), E2's rescore and verdict L23C, then E2-C's
    attribution L23D, E3 (2026-10-05: its three arms' scoring jobs L23F
    in the order M, A, B, then its verdict L23G), the model-zoo audit
    (L23Z, once, last), then the first segment."""
    w = World("r0")
    w.lock()
    w.step1_status()
    w.placement({"hf": "pass"})
    inc = M.CLUSTER_INC_DIR
    tail = lambda pr, n: (pr or {}).get("argv", [])[-n:]  # noqa: E731
    got = []
    measure = [b for b in w.dom["baselines"]["items"] if b.get("measure") and not b.get("requires")]
    e1 = [b for b in w.dom["baselines"]["items"] if b.get("requires") == "base3"]
    for b in [x for x in w.dom["baselines"]["items"] if not x.get("measure")]:
        pr = _step(w, "L23B", lambda b=b: (w.experiment(b["exp"], final=[w.final_row("base", 0.81, 0.002, 3)]),
                                           w.job_done("inc_build_%s" % b["exp"])))
        got.append(pr)
    want_bv2 = ["build", "--exp", "b_v2", "--manifest", "%s/splits/v2/base_v2.jsonl" % inc, "--seeds", "0,1,2,3,4",
                "--arm", "n640", "--role", "b_v2"]
    check("L23B builds B_v2 first, in inc2.baseline's grammar (--arm, --role)", tail(got[0], 11) == want_bv2,
          tail(got[0], 11))
    check("  then the canary, the capacity arms (--arm s640 / m640, role capacity), and B0 u tsw (--union)",
          [(x or {}).get("child_exp") for x in got] == ["b_v2", "canary_v2", "b_v2_s640", "b_v2_m640", "b0_tsw_v2"]
          and tail(got[2], 4) == ["--arm", "s640", "--role", "capacity"]
          and "--union" in (got[4] or {}).get("argv", []) and "%s/splits/v2/tsw23.jsonl" % inc in
          (got[4] or {}).get("argv", [])[(got[4] or {}).get("argv", []).index("--union") + 1],
          [(x or {}).get("child_exp") for x in got])
    sub = [x for x in w.submits if "inc2.baseline" in x["argv"]]
    req = SR.parse_submit("build", sub[2]["argv"][sub[2]["argv"].index("inc2.baseline"):]) if len(sub) > 2 else {}
    check("  the cluster's grammar reads the capacity arm back", req.get("params", {}).get("arm") == "s640", req)
    pr = _step(w, "LV")
    check("LV records the canary's verdict (inc2.baseline canary-verdict --exp canary_v2, a login-node verb)",
          tail(pr, 4) == [MOD + "inc2.baseline", "canary-verdict", "--exp", "canary_v2"]
          and (w.inc / "canary_v2" / "canary.json").is_file(), tail(pr, 4))
    pr = _step(w, "LV")
    check("LV records the capacity decision (capacity-verdict) once every arm is done",
          tail(pr, 2) == [MOD + "inc2.baseline", "capacity-verdict"] and (w.inc / "capacity" / "capacity_v1.json").is_file(),
          tail(pr, 2))
    pr = _step(w, "L25", lambda: (w.experiment("pilot_v4", typ="chain"), w.job_done("inc_build_pilot_v4")))
    check("L25 builds Stage A (Protocol v3 accepted by a person)",
          tail(pr, 6) == ["--exp", "pilot_v4", "--from", "pilot_v3", "--recipes", "x1a,x1b"], tail(pr, 6))
    pr = _step(w, "LV")
    check("LV records Stage A's verdict (inc2.pilot4 verdict --exp pilot_v4)",
          tail(pr, 4) == [MOD + "inc2.pilot4", "verdict", "--exp", "pilot_v4"], tail(pr, 4))
    pr = _step(w, "LI", lambda: (w.job_done("inc_build_stream_init"), w.stream_init(("r0", "x1a"))))
    check("LI creates the stream with Stage A's recorded recipes as its Stage B arms",
          tail(pr, 6) == [MOD + "inc2.stream", "init", "--stream", w.sid, "--stage-b", "r0,x1a"], tail(pr, 6))
    isub = [x for x in w.submits if "init" in x["argv"] and "inc2.stream" in x["argv"]]
    check("  as a build job on GPU-shared (run_inc2_build.sh inc2.stream init)",
          isub and isub[0]["argv"][3:5] == ["-p", "GPU-shared"], [x["argv"] for x in isub])
    pr = _step(w, "LA")
    check("LA adopts the capacity decision (inc2.stream choose-arm)",
          tail(pr, 4) == [MOD + "inc2.stream", "choose-arm", "--stream", w.sid]
          and any(e.get("event") == "arm" for e in w.stream_ledger()), tail(pr, 4))
    pr = _step(w, "L28", lambda: w.stage_c_built(done=True))
    check("L28 builds Stage C with the stream's own M",
          tail(pr, 6) == ["--stream", w.sid, "--holdout", "tsw22", "--m", str(w.M)], tail(pr, 6))
    pr = _step(w, "LC")
    check("LC reads Stage C (inc2.stream compare --exp <sid>_c001)",
          tail(pr, 4) == [MOD + "inc2.stream", "compare", "--exp", "%s_c001" % w.sid]
          and any(e.get("event") == "feasibility" and e.get("phase") == "read" for e in w.stream_ledger()), tail(pr, 4))
    arm0 = [e for e in w.stream_ledger() if e.get("event") == "arm"]
    cap0 = (w.inc / "capacity" / "capacity_v1.json").read_bytes()
    mgot = []
    for b in measure:
        mgot.append(_step(w, "L23B", lambda b=b: (w.experiment(b["exp"], final=[w.final_row("base", 0.83, 0.002, 3)]),
                                                  w.job_done("inc_build_%s" % b["exp"]))))
    mex = [e.get("basis") for e in w.events("executed")
           if e.get("lever") == "L23B" and e.get("child_exp") in [b["exp"] for b in measure]]
    check("R0 complete: the measurement arms, each once, as L23B in inc2.baseline's grammar (--arm m832 / s1024 / "
          "y26l640 / y26m640 / l640, role capacity), within the envelope (no person asked; a day apart, so the "
          "daily cap does not bind)",
          [(x or {}).get("child_exp") for x in mgot] == [b["exp"] for b in measure]
          == ["b_v2_m832", "b_v2_s1024", "b_v2_y26l640", "b_v2_y26m640", "b_v2_l640"]
          and all(tail(x, 4) == ["--arm", b["arm"], "--role", "capacity"] for x, b in zip(mgot, measure))
          and mex == ["envelope"] * len(measure),
          ([(x or {}).get("argv", [])[-6:] for x in mgot], mex))
    sub = [x for x in w.submits if "inc2.baseline" in x["argv"] and "b_v2_m832" in x["argv"]]
    req = SR.parse_submit("build", sub[0]["argv"][sub[0]["argv"].index("inc2.baseline"):]) if sub else {}
    check("  the cluster's grammar reads the measurement arm back", req.get("params", {}).get("arm") == "m832", req)
    # 2026-10-03: E1. Its first arm is next and requires splits v3: the base v3 build (L23V) once, then the two arms
    pv = _step(w, "L23V", lambda: (w.job_done("inc_build_base3_v3"), w.base3_summary()))
    vex = [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23V"]
    vsub = [x for x in w.submits if "inc2.base3" in x["argv"]]
    vreq = SR.parse_submit("build", vsub[0]["argv"][vsub[0]["argv"].index("inc2.base3"):]) if vsub else {}
    check("E1: the base v3 build next (L23V inc2.base3 build --stream SID), once, within the envelope, one "
          "run_inc2_build.sh job under its own name, read back by the cluster's grammar",
          tail(pv, 4) == [MOD + "inc2.base3", "build", "--stream", w.sid] and vex == ["envelope"]
          and [x["name"] for x in vsub] == ["inc_build_base3_v3"] and vreq.get("params") == {"stream": w.sid},
          (tail(pv, 4), vex, [x["name"] for x in vsub]))
    egot = []
    for b in e1:
        egot.append(_step(w, "L23B", lambda b=b: (w.experiment(b["exp"], final=[w.final_row("base", 0.83, 0.002, 3)]),
                                                  w.job_done("inc_build_%s" % b["exp"]))))
    esub = [x for x in w.submits if "inc2.baseline" in x["argv"] and "e1_a_m640" in x["argv"]]
    ereq = SR.parse_submit("build", esub[0]["argv"][esub[0]["argv"].index("inc2.baseline"):]) if esub else {}
    check("  then E1-A and E1-B, each once, as L23B on splits v3 (--arm m640 --role baseline), within the envelope; "
          "the cluster's grammar reads role baseline back",
          [(x or {}).get("child_exp") for x in egot] == ["e1_a_m640", "e1_b_m640"]
          and [tail(x, 8) for x in egot] == [["--manifest", "%s/%s" % (inc, b["manifest"]), "--seeds", "0,1,2",
                                              "--arm", "m640", "--role", "baseline"] for b in e1]
          and ereq.get("params", {}).get("role") == "baseline",
          ([tail(x, 8) for x in egot], ereq.get("params")))
    # 2026-10-01: each done measurement arm is read at its own resolution, once (L23N), within the envelope
    ngot = []
    for b in measure:
        ngot.append(_step(w, "L23N", lambda b=b: (w.job_done("inc_build_native_%s" % b["exp"]),
                                                  w.native_record(b["exp"]))))
    nex = [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23N"]
    ref = w.dom["capacity"]["native"]["reference_exp"]
    check("the measurement arms done: each is rescored at its own resolution once, as L23N (inc2.baseline "
          "rescore-native --exp E --reference %s), within the envelope, citing only the lock and its own state" % ref,
          [tail(x, 4) for x in ngot] == [["--exp", b["exp"], "--reference", ref] for b in measure]
          and nex == ["envelope"] * len(measure)
          and [sorted(c.get("pointer") for c in (x or {}).get("cites") or []) for x in ngot]
          == [sorted(["/stage/lock", "/stage/exp_status/%s" % b["exp"], "/stage/native/%s" % b["id"]])
              for b in measure], ([tail(x, 6) for x in ngot], nex))
    nsub = [x["name"] for x in w.submits if "rescore-native" in x["argv"]]
    check("  each one GPU job of run_inc2_build.sh under its own name (never the arm's build job's)",
          nsub == ["inc_build_native_%s" % b["exp"] for b in measure], nsub)
    pe = _step(w, "L23E", lambda: (w.job_done("inc_build_agnostic_e1_b_m640"), w.agnostic_record("e1_b_m640")))
    check("E1's arms done: their agnostic rescore and E1's verdict, once (L23E rescore-agnostic --exp e1_b_m640 "
          "--reference e1_a_m640), within the envelope; no L23N for them",
          tail(pe, 5) == ["rescore-agnostic", "--exp", "e1_b_m640", "--reference", "e1_a_m640"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23E"] == ["envelope"]
          and not [e for e in w.events("proposed") if e.get("lever") == "L23N"
                   and (e.get("argv") or [])[-3] in ("e1_a_m640", "e1_b_m640")], tail(pe, 5))
    check("  and the stream's arm stays the capacity decision's: no new arm line, no LA, capacity_v1.json unchanged",
          [e for e in w.stream_ledger() if e.get("event") == "arm"] == arm0
          and (w.inc / "capacity" / "capacity_v1.json").read_bytes() == cap0
          and [e.get("lever") for e in w.events("proposed")].count("LA") == 1, arm0)
    # 2026-10-04: E2. E1's verdict qualifies E1-B (inc2.baseline wrote capacity/e1_v1.json in L23E's job): E2's six
    # single-seed builds, one at a time, then its rescore and verdict (L23C), each once, within the envelope
    w.e1_verdict()
    e2 = [b for b in w.dom["baselines"]["items"] if b.get("requires") == "e1" and b.get("e2") in ("W", "S")]
    e2c = [b for b in w.dom["baselines"]["items"] if b.get("requires") == "e1" and b.get("e2") == "C"]
    e2got = []
    for b in e2:
        e2got.append(_step(w, "L23B", lambda b=b: (w.experiment(b["exp"], final=[w.final_row("base", 0.85, 0.002, 1)]),
                                                   w.job_done("inc_build_%s" % b["exp"]))))
    e2ex = [e.get("basis") for e in w.events("executed")
            if e.get("lever") == "L23B" and e.get("child_exp") in [b["exp"] for b in e2]]
    e2sub = [x for x in w.submits if "--e2" in x["argv"]]
    e2req = SR.parse_submit("build", e2sub[1]["argv"][e2sub[1]["argv"].index("inc2.baseline"):]) \
        if len(e2sub) > 1 else {}
    check("E2: E1-B qualified, so its six builds next, each once, as L23B (--seeds k --arm m640 --role baseline "
          "--e2 W|S), in the order W0, S0, W1, S1, W2, S2, within the envelope, each one build job named after its "
          "experiment; the cluster's grammar reads --e2 back",
          [(x or {}).get("child_exp") for x in e2got] == ["e2_w_m640_seed0", "e2_s_m640_seed0", "e2_w_m640_seed1",
                                                          "e2_s_m640_seed1", "e2_w_m640_seed2", "e2_s_m640_seed2"]
          and [tail(x, 8) for x in e2got] == [["--seeds", b["seeds"], "--arm", "m640", "--role", "baseline", "--e2",
                                               b["e2"]] for b in e2]
          and e2ex == ["envelope"] * 6 and [x["name"] for x in e2sub] == ["inc_build_%s" % b["exp"] for b in e2]
          and e2req.get("params", {}).get("e2") == "S", ([tail(x, 8) for x in e2got], e2ex))
    # 2026-10-04, later: E2-C. E1-A is done: E2-C's three builds after E2's six, each once, within the envelope
    cgot = []
    for b in e2c:
        cgot.append(_step(w, "L23B", lambda b=b: (w.experiment(b["exp"], final=[w.final_row("base", 0.85, 0.002, 1)]),
                                                  w.job_done("inc_build_%s" % b["exp"]))))
    csub = [x for x in w.submits if "--e2" in x["argv"] and "C" in x["argv"]]
    creq = SR.parse_submit("build", csub[0]["argv"][csub[0]["argv"].index("inc2.baseline"):]) if csub else {}
    check("E2-C: then its three builds, each once, as L23B (--seeds k --arm m640 --role baseline --e2 C), in the order "
          "C0, C1, C2, within the envelope, each one build job named after its experiment; the grammar reads --e2 C",
          [(x or {}).get("child_exp") for x in cgot] == ["e2_c_m640_seed0", "e2_c_m640_seed1", "e2_c_m640_seed2"]
          and [tail(x, 8) for x in cgot] == [["--seeds", b["seeds"], "--arm", "m640", "--role", "baseline", "--e2", "C"]
                                             for b in e2c]
          and [x["name"] for x in csub] == ["inc_build_%s" % b["exp"] for b in e2c]
          and creq.get("params", {}).get("e2") == "C", ([tail(x, 8) for x in cgot], creq.get("params")))
    pc = _step(w, "L23C", lambda: (w.job_done("inc_build_e2_v1"), w.e2_records()))
    check("  then, all six and b_v2_m640 done, L23C once (inc2.baseline rescore-e2), within the envelope, one job "
          "named inc_build_e2_v1; no L23N for any E2 experiment",
          tail(pc, 2) == [MOD + "inc2.baseline", "rescore-e2"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23C"] == ["envelope"]
          and [x["name"] for x in w.submits if x["argv"][-1] == "rescore-e2"] == ["inc_build_e2_v1"]
          and not [e for e in w.events("proposed") if e.get("lever") == "L23N"
                   and str((e.get("argv") or [])[-3:]).count("e2_")], tail(pc, 2))
    e2v = (w.inc / "capacity" / "e2_v1.json").read_bytes()
    pd = _step(w, "L23D", lambda: (w.job_done("inc_build_e2_attr_v1"), w.e2_attr_records()))
    check("  then, E2's verdict recorded and E2-W's and E2-C's runs done, L23D once (inc2.baseline rescore-e2-attr), "
          "within the envelope, one job named inc_build_e2_attr_v1; capacity/e2_v1.json unchanged",
          tail(pd, 2) == [MOD + "inc2.baseline", "rescore-e2-attr"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23D"] == ["envelope"]
          and [x["name"] for x in w.submits if "rescore-e2-attr" in x["argv"]] == ["inc_build_e2_attr_v1"]
          and (w.inc / "capacity" / "e2_v1.json").read_bytes() == e2v, tail(pd, 2))
    # 2026-10-05: E3. E2's verdict decided and E2-C's attribution recorded: one scoring job per arm, M, A, B, then
    # E3's verdict, each once, within the envelope
    pf = []
    for arm in ("M", "A", "B"):
        pf.append(_step(w, "L23F", lambda arm=arm: (w.job_done("inc_build_e3_score_%s" % arm.lower()),
                                                    w.e3_records(arms=(arm,), rescore=False))))
    check("E3: then L23F three times (inc2.twostage score-arm --arm M, A, B), in order, within the envelope, each one "
          "job named inc_build_e3_score_<arm>; the grammar reads --arm",
          [tail(x, 4) for x in pf] == [[MOD + "inc2.twostage", "score-arm", "--arm", k] for k in "MAB"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23F"] == ["envelope"] * 3
          and [x["name"] for x in w.submits if "score-arm" in x["argv"]] == ["inc_build_e3_score_%s" % k
                                                                             for k in "mab"], [tail(x, 4) for x in pf])
    pg = _step(w, "L23G", lambda: (w.job_done("inc_build_e3_v1"), w.e3_records(arms=(), verdict=True)))
    check("  then L23G once (inc2.twostage verdict), within the envelope, one job named inc_build_e3_v1; capacity/"
          "e2_v1.json unchanged", tail(pg, 2) == [MOD + "inc2.twostage", "verdict"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23G"] == ["envelope"]
          and [x["name"] for x in w.submits if x["argv"][-1] == "verdict"] == ["inc_build_e3_v1"]
          and (w.inc / "capacity" / "e2_v1.json").read_bytes() == e2v, tail(pg, 2))
    # 2026-10-04: the model-zoo audit (Amendment Z1). E2's verdict and E2-C's attribution recorded
    # (capacity/e2_rescore.json and capacity/e2_attr_rescore.json complete) and E3's jobs proposed first (its block
    # precedes the zoo's in DR0): L23Z once, last, within the envelope; its chain followed by its five job ids
    pz = _step(w, "L23Z", lambda: w.zoo_finish("complete"))
    check("  then, E2's verdict and E2-C's attribution recorded, E3's verdict recorded and no zoo record, L23Z once "
          "(bash run_inc2_zoo.sh submit --version v1 --shards-a 32 --shards-c 16 --concurrency 4 --max-gpu-hours "
          "40), within the envelope, one submission",
          tail(pz, 13) == ["bash", "run_inc2_zoo.sh", "submit", "--version", "v1", "--shards-a", "32", "--shards-c",
                           "16", "--concurrency", "4", "--max-gpu-hours", "40"]
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23Z"] == ["envelope"]
          and len(w.zoo_submits) == 1, (tail(pz, 13), len(w.zoo_submits)))
    w.tick(2)
    check("  once both exist, no measurement arm is proposed again, nor its rescore",
          [e.get("child_exp") for e in w.events("proposed") if e.get("lever") == "L23B"].count("b_v2_m832") == 1
          and [e.get("child_exp") for e in w.events("proposed") if e.get("lever") == "L23B"].count("b_v2_s1024") == 1
          and len([e for e in w.events("proposed") if e.get("lever") == "L23N"]) == len(measure))
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * DAY))
    pr = _step(w, "L18")
    check("R0 READY: the first segment is cut (L18)", pr is not None and pr.get("child_exp") == "%s_s001" % w.sid,
          [e.get("lever") for e in w.events("proposed")][-5:])
    lv = [e.get("lever") for e in w.events("executed") if e.get("lane") == "MAINT"]
    check("the whole sequence ran by the platform, in order, one MAINT item at a time, each once",
          lv == ["L23B"] * 5 + ["LV", "LV", "L25", "LV", "LI", "LA", "L28", "LC"] + ["L23B"] * len(measure)
          + ["L23V"] + ["L23B"] * len(e1) + ["L23N"] * len(measure) + ["L23E"] + ["L23B"] * len(e2)
          + ["L23B"] * len(e2c) + ["L23C", "L23D"] + ["L23F"] * 3 + ["L23G", "L23Z"], lv)
    zc = [c for c in w.state()["cards"] if c.get("lever") == "L23Z"]
    check("  the zoo's chain ended complete: /stage/zoo done, one research card (counts and the report's path, no "
          "metric), never proposed again",
          len(zc) == 1 and zc[0]["kind"] == "research" and "_zoo/v1/report.md" in zc[0]["detail"]
          and "descriptive" in zc[0]["detail"] and len([e for e in w.events("proposed") if e.get("lever") == "L23Z"])
          == 1, zc)


def _commits_ev(segments, ctx):
    """Evidence holding a stream ledger of commit lines (inc2.stream's form)."""
    events = [dict(commit_fields(n, "x_s%03d" % n, rows), event="commit") for n, rows in enumerate(segments, 1)]
    return E.from_texts({"stream/x/ledger.jsonl": ledger_text(events, "x")}, "x", context=ctx)


def s6_extra():
    """D30, D32 and D33 on synthetic committed segments (their positive cases)."""
    dom, th = LS.load_domain("weed"), LS.load_thresholds()
    ctx = {"now_utc": "2026-10-01T00:00:00Z", "sid": "x", "M": 682,
           "lanes": {ln: {"phase": "IDLE"} for ln in S.LANES}, "stage": {"stage_a_ready": True}}
    ev = _commits_ev([[("A1", "REJECT", 0.6, ["regression"], "neutral", ["s"]),
                       ("A2", "REJECT", 0.5, ["regression"], "neutral", ["s"]),
                       ("A3", "REJECT", 0.1, ["species"], "neutral", ["s"])]], ctx)
    by = DS.by_id(DS.detect(ev, dom, th))
    check("D30 fires when half the REJECTs are recipe-caused (2 of 3: only the regression guard failed): TRAIN holds, "
          "cards X1 and X13", by["D30"]["fired"] and by["D30"]["levers"] == ["X1", "X13"]
          and by["D30"]["detail"]["hold"] == "TRAIN", by["D30"]["summary"])
    check("  its cites are the commit line's steps in the stream ledger",
          all(c["artifact"] == "stream/x/ledger.jsonl" and "/steps/r0/" in c["pointer"] for c in by["D30"]["cites"]),
          by["D30"]["cites"][:2])
    segs = [[("B%d%d" % (i, j), "REJECT", 0.4, ["species"], "neutral", ["s"], ["Carpetweed"]) for j in range(3)]
            for i in range(4)]
    by2 = DS.by_id(DS.detect(_commits_ev(segs, ctx), dom, th))
    check("D32 fires on 0 ACCEPT in the last 12 decided increments with D30 silent: L18 holds, card X17, DATA at "
          "half cadence", by2["D32"]["fired"] and by2["D32"]["detail"].get("half_cadence") and not by2["D30"]["fired"],
          by2["D32"]["summary"])
    check("D33 names the species failing the guard in every REJECT of the last segment",
          by2["D33"]["fired"] and by2["D33"]["detail"]["species"] == ["Carpetweed"], by2["D33"]["summary"])
    rows3 = [("C%d" % j, "ACCEPT" if j == 5 else "REJECT", 0.4, [] if j == 5 else ["species"], "neutral", ["s"])
             for j in range(12)]
    by3 = DS.by_id(DS.detect(_commits_ev([rows3], ctx), dom, th))
    check("  one ACCEPT among the 12 keeps it silent", not by3["D32"]["fired"], by3["D32"]["summary"])
    fire = [("F%d" % j, "REJECT", 0.4, ["species"], "neutral", ["s"], ["Carpetweed"]) for j in range(2)]
    calm = [("G%d" % j, "ACCEPT", 0.9, [], "helps", ["s"]) for j in range(2)]
    d5 = DS.by_id(DS.detect(_commits_ev([fire, calm, fire], ctx), dom, th))["D33"]
    check("D33's L18 hold needs 2 consecutive segments: fired, silent, fired -> fires, no hold",
          d5["fired"] and d5["detail"]["consecutive"] == 1 and d5["detail"]["hold"] is None, d5["detail"])
    d6 = DS.by_id(DS.detect(_commits_ev([calm, fire, fire], ctx), dom, th))["D33"]
    check("  fired on the last 2 segments -> TRAIN holds", d6["detail"]["consecutive"] == 2
          and d6["detail"]["hold"] == "TRAIN", d6["detail"])
    ctx7 = dict(ctx, d33_history=[{"exp": "realloop_v1", "fired": True, "species": ["PricklySida"]}])
    d7 = DS.by_id(DS.detect(_commits_ev([fire], ctx7), dom, th))["D33"]
    check("  the pinned prior is not a stream segment: after it fired, the first segment's D33 does not hold TRAIN",
          d7["fired"] and d7["detail"]["hold"] is None and d7["detail"]["consecutive"] == 1, d7["detail"])
    stale = [dict(commit_fields(1, "x_s001", rows3), event="commit", stale_base=True)]
    by4 = DS.by_id(DS.detect(E.from_texts({"stream/x/ledger.jsonl": ledger_text(stale, "x")}, "x", context=ctx),
                             dom, th))
    check("  a commit on a stale base (a rollback ran meanwhile) decides nothing", not by4["D33"]["fired"]
          and not by4["D32"]["fired"], (by4["D33"]["summary"], by4["D32"]["summary"]))


CASES = [("S1", s1), ("S1b", s1b), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", lambda: (s6(), s6_extra())),
         ("S7", s7), ("S8", s8), ("S9", s9), ("S10", s10), ("S11", s11), ("S12", s12), ("S13", s13), ("S14", s14),
         ("S15", s15), ("S16", s16), ("S17", s17), ("S18", s18), ("S19", s19), ("S20", s20), ("S21", s21),
         ("S22", s22), ("S23", s23), ("S24", s24), ("S25", s25), ("S26", s26), ("S27", s27), ("S28", s28),
         ("S29", s29),
         ("stream_prospective", s_prospective), ("stream_r0", s_r0)]


def main(argv=None):
    ids = list(sys.argv[1:] if argv is None else argv)
    for cid, fn in CASES:
        if ids and cid not in ids:
            continue
        run_case(cid, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
