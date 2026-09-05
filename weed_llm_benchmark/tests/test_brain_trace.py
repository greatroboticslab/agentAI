#!/usr/bin/env python3
"""Unit tests for the hash-chained artifact trace (no cluster / Mongo needed).

Covers what the 2026-08-29 walltime TIMEOUTs need from a trace and what the
supervision layer needs to trust one:
  * append/read round trip, and the record `append` returns
  * the sha chain links record N to record N-1
  * verify() catches a record edited after the fact
  * latest(kind=...) picks the last record of one kind
  * a torn trailing line (writer killed mid-write) hides nothing from read()
  * append() on an unwritable path returns {} instead of killing the job

Run:  python3 tests/test_brain_trace.py
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import trace  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


_tmp = tempfile.mkdtemp(prefix="brain_trace_")


# ---- append / read round trip --------------------------------------------
p = os.path.join(_tmp, "deep", "r1_train_1234.jsonl")
r1 = trace.append(p, {"kind": "start", "domain": "weed", "round": 1, "step": "train",
                      "job_id": "1234"})
r2 = trace.append(p, {"kind": "epoch", "domain": "weed", "round": 1, "step": "train",
                      "job_id": "1234", "epoch": 1, "map50_95": 0.41})
r3 = trace.append(p, {"kind": "epoch", "domain": "weed", "round": 1, "step": "train",
                      "job_id": "1234", "epoch": 2, "map50_95": 0.47})

ck("append creates parent directories", os.path.exists(p))
ck("append returns the written record", r2.get("epoch") == 1 and r2.get("kind") == "epoch")
ck("append stamps a float ts", isinstance(r1.get("ts"), float))
ck("caller-supplied ts is kept",
   trace.append(p, {"kind": "decision", "ts": 1000.5})["ts"] == 1000.5)

recs = trace.read(p)
ck("read returns every record, oldest first", len(recs) == 4 and recs[0]["kind"] == "start")
ck("read round-trips payload fields", recs[2]["map50_95"] == 0.47)
ck("read(limit=2) returns the last two", [r["kind"] for r in trace.read(p, limit=2)]
   == ["epoch", "decision"])
ck("read(limit=0) returns nothing", trace.read(p, limit=0) == [])
ck("read of a missing file is empty", trace.read(os.path.join(_tmp, "nope.jsonl")) == [])


# ---- sha chain ------------------------------------------------------------
ck("first record has empty sha_prev", r1["sha_prev"] == "")
ck("second record links to the first", r2["sha_prev"] == r1["sha"])
ck("third record links to the second", r3["sha_prev"] == r2["sha"])
ck("sha excludes itself",
   r2["sha"] == trace._sha_of({k: v for k, v in r2.items() if k != "sha"}))
ck("verify passes on an untouched file", trace.verify(p) == {"ok": True, "records": 4,
                                                             "broken_at": None})
ck("verify of a missing file is ok with 0 records",
   trace.verify(os.path.join(_tmp, "nope.jsonl")) == {"ok": True, "records": 0,
                                                      "broken_at": None})

# A second process appending later must continue the same chain.
p2 = os.path.join(_tmp, "reopen.jsonl")
trace.append(p2, {"kind": "epoch", "epoch": 1})
a = trace.append(p2, {"kind": "epoch", "epoch": 2})
ck("chain continues across a reopen", trace.verify(p2)["ok"] and a["sha_prev"] != "")

# The tail read must not start a second chain when a record is longer than the
# 4 KB window.
p3 = os.path.join(_tmp, "fat.jsonl")
trace.append(p3, {"kind": "report", "text": "x" * 9000})
fat = trace.append(p3, {"kind": "epoch", "epoch": 1})
ck("record larger than the tail window still links", fat["sha_prev"] != "")
ck("verify passes past a record larger than the tail window", trace.verify(p3)["ok"])


# ---- verify() catches an edited middle record -----------------------------
p4 = os.path.join(_tmp, "tampered.jsonl")
for i in (1, 2, 3):
    trace.append(p4, {"kind": "epoch", "epoch": i, "map50_95": 0.1 * i})
lines = open(p4).read().splitlines()
mid = json.loads(lines[1])
mid["map50_95"] = 0.99            # the silent-wrongness case: plausible, edited
lines[1] = json.dumps(mid, sort_keys=True, separators=(",", ":"))
open(p4, "w").write("\n".join(lines) + "\n")
v = trace.verify(p4)
ck("verify fails on an edited middle record", v["ok"] is False)
ck("verify points at the edited record", v["broken_at"] == 2)
ck("verify reports the records that held up", v["records"] == 1)

# Deleting a record breaks the link of the one that follows it.
p5 = os.path.join(_tmp, "deleted.jsonl")
for i in (1, 2, 3):
    trace.append(p5, {"kind": "epoch", "epoch": i})
lines = open(p5).read().splitlines()
open(p5, "w").write("\n".join([lines[0], lines[2]]) + "\n")
v5 = trace.verify(p5)
ck("verify fails when a record is removed", v5["ok"] is False and v5["broken_at"] == 2)


# ---- latest(kind=...) -----------------------------------------------------
ck("latest returns the last record", trace.latest(p)["kind"] == "decision")
ck("latest(kind) filters", trace.latest(p, kind="epoch")["epoch"] == 2)
ck("latest(kind) with no match is None", trace.latest(p, kind="candidate") is None)
ck("latest of a missing file is None", trace.latest(os.path.join(_tmp, "nope.jsonl")) is None)


# ---- torn trailing line (writer killed mid-write) -------------------------
p6 = os.path.join(_tmp, "torn.jsonl")
trace.append(p6, {"kind": "epoch", "epoch": 1})
last = trace.append(p6, {"kind": "epoch", "epoch": 2})
with open(p6, "a") as f:
    f.write('{"kind": "epoch", "epoch": 3, "map5')   # killed at the walltime
ck("read skips a torn trailing line", [r["epoch"] for r in trace.read(p6)] == [1, 2])
ck("verify tolerates a torn trailing line", trace.verify(p6)["ok"] is True)
resumed = trace.append(p6, {"kind": "end", "status": "done"})
ck("append links across a torn line", resumed["sha_prev"] == last["sha"])
ck("read still sees every whole record after a torn line", len(trace.read(p6)) == 3)


# ---- append() never raises ------------------------------------------------
unwritable = os.path.join(p, "cannot", "exist.jsonl")   # parent is a regular file
ck("append on an unwritable path returns {}", trace.append(unwritable, {"kind": "epoch"}) == {})
ck("append of a non-dict record returns {}", trace.append(p, ["not", "a", "record"]) == {})
ck("append failure wrote nothing", len(trace.read(p)) == 4)
ck("read of an unreadable path is empty", trace.read(unwritable) == [])


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
