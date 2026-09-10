#!/usr/bin/env python3
"""The heartbeat keeps a history now, and the history must not break the tick.

`scheduler_status.json` is a snapshot: every write erases the last one, so it can
say the loop is alive and never say when it stopped. These pin the archive that
sits beside it — one compact line per tick — and, more importantly, that a broken
archive is not allowed to look like a missing heartbeat, which is the one thing
the alarm treats as crit.

Run:  python tests/test_scheduler_archive.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import round_scheduler as RS  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


ROOT = tempfile.mkdtemp(prefix="sched_archive_")
_saved_root = RS._repo_root
RS._repo_root = lambda: ROOT
JSONL = pathlib.Path(ROOT) / "results" / "framework" / "scheduler_status.jsonl"
SNAP = pathlib.Path(ROOT) / "results" / "framework" / "scheduler_status.json"


def payload(**over):
    p = {"ts": 1789050000.25, "tick_s": 120, "tick_duration_s": 1.25,
         "mongo_ok": True, "config_ok": True, "config_domains_seen": 1,
         "domains": {"weed": {"enabled": True, "paused_reason": "", "job": "45672628",
                              "step": "train", "fails": 0, "rounds_today": 2}}}
    p.update(over)
    return p


print("one line per tick")
RS._archive_heartbeat(payload())
RS._archive_heartbeat(payload(ts=1789050120.5))
lines = JSONL.read_text().strip().splitlines()
ck("two calls append two lines", len(lines) == 2)
row = json.loads(lines[0])
ck("the line parses", isinstance(row, dict))
ck("and carries the tick's own timestamp", row["ts"] == 1789050000.2 or row["ts"] == 1789050000.3)
ck("and whether Mongo was reachable", row["mongo"] is True)
ck("and whether the config could be read", row["cfg"] is True)
ck("and what each domain was doing", row["d"]["weed"]["step"] == "train")
ck("and the job it was waiting on", row["d"]["weed"]["job"] == "45672628")
ck("and how many rounds it had advanced today", row["d"]["weed"]["today"] == 2)
RS._archive_heartbeat(payload(domains={"weed": {"enabled": False,
                                                "paused_reason": "stop-loss: two failures",
                                                "fails": 2}}))
last = json.loads(JSONL.read_text().strip().splitlines()[-1])
ck("a paused domain reads as paused, without carrying the whole reason",
   last["d"]["weed"]["paused"] is True and "reason" not in last["d"]["weed"])
ck("and a disabled domain reads as off", last["d"]["weed"]["on"] is False)
ck("the line stays compact", len(JSONL.read_text().splitlines()[-1]) < 200)

print("\nthe history is bounded")
_cap = RS._ARCHIVE_MAX_BYTES
try:
    RS._ARCHIVE_MAX_BYTES = 200
    before = len(JSONL.read_text().strip().splitlines())
    RS._archive_heartbeat(payload())      # over the cap: rotates, then writes 1
    RS._archive_heartbeat(payload())      # under it again: appends
    rotated = pathlib.Path(str(JSONL) + ".1")
    ck("it rotates once past the cap", rotated.exists())
    ck("and the live file holds only what was written after the rotation",
       len(JSONL.read_text().strip().splitlines()) == 2)
    ck("and the previous generation is kept whole, not deleted",
       len(rotated.read_text().strip().splitlines()) == before)
finally:
    RS._ARCHIVE_MAX_BYTES = _cap

print("\nit is not allowed to break the tick")
JSONL.unlink(missing_ok=True)
# A directory where the file should be: append will raise IsADirectoryError.
JSONL.mkdir(parents=True, exist_ok=True)
raised = False
try:
    RS._archive_heartbeat(payload())
except Exception:
    raised = True
ck("an unwritable archive raises out of _archive_heartbeat", raised)

RS._LEDGER.update({"ok": True, "last_error_ts": None})
RS._CFG_READ.update({"ok": True, "error": "", "present": True, "n_domains": 1})
cfg = {"domains": {"weed": {"enabled": True}}}
SNAP.unlink(missing_ok=True)
RS._heartbeat(cfg, 1.5)
ck("but _heartbeat still writes the snapshot the alarm reads", SNAP.exists())
snap = json.loads(SNAP.read_text())
ck("and the snapshot is complete, not truncated",
   snap.get("config_ok") is True and "domains" in snap and snap.get("tick_s"))

shutil.rmtree(JSONL, ignore_errors=True)
RS._heartbeat(cfg, 1.5)
ck("with the path usable again the archive resumes",
   JSONL.exists() and len(JSONL.read_text().strip().splitlines()) == 1)

RS._repo_root = _saved_root
shutil.rmtree(ROOT, ignore_errors=True)

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
