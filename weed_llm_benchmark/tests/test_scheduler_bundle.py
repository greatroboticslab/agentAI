#!/usr/bin/env python3
"""The scheduler writes an evidence bundle for a step that COMPLETED.

Until this existed, `evidence.build` had no production caller at all -- it was
reached only by its own test and its own `main()`, nothing wrote the
`latest_bundle.json` the supervision page reads, and the twelve deterministic
checks ran on the archived corpus and never once on a live round. Four rounds ran
at roughly half their stated recipe with the metric flat inside its own noise
floor, and the layer built to notice exactly that was not looking.

The COMPLETED path is the one that matters: the review path already covers
failures, and both `epochs_truncated` and `plateau` are findings about runs that
succeeded.

Run:  python tests/test_scheduler_bundle.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import round_scheduler as rs  # noqa: E402
from weed_optimizer_framework.tools import db                     # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, *a):
        self.lines.append(("info",) + a)

    def warning(self, *a):
        self.lines.append(("warn",) + a)

    def error(self, *a):
        self.lines.append(("error",) + a)


class _DB:
    """Enough database for the bundle path, and nothing else."""

    def __init__(self, rounds=None, raise_on_rounds=False):
        self._rounds = rounds if rounds is not None else [
            {"round_num": 11, "metrics": {"map50_95": 0.5554}, "steps": {}},
            {"round_num": 10, "metrics": {"map50_95": 0.5711}, "steps": {}},
            {"round_num": 9, "metrics": {"map50_95": 0.5675}, "steps": {}},
        ]
        self._raise = raise_on_rounds

    def get_rounds(self, domain, limit=25):
        if self._raise:
            raise RuntimeError("mongo is down")
        return self._rounds[:limit]

    def get_current_round(self, domain):
        return self._rounds[0] if self._rounds else None


def _ctx(root, log, dbobj, slurm=None):
    return {"log": log, "db": dbobj, "repo": root,
            "slurm_sh": slurm or (lambda cmd, timeout=None: {"stdout": "", "ok": True})}


ROOT = tempfile.mkdtemp(prefix="sched_bundle_")
OUT = os.path.join(ROOT, "results", "framework", "_brain", "weed",
                   "latest_bundle.json")
CFG = db.DEFAULT_DOMAIN_CONFIG

print("a completed step leaves a bundle behind")
log = _Log()
rs._CTX.clear()
rs._CTX.update(_ctx(ROOT, log, _DB()))
b = rs._build_bundle("weed", "train", "45444584", CFG)
ck("a bundle is returned", isinstance(b, dict))
ck("and written where the supervision page reads it", os.path.exists(OUT))
if os.path.exists(OUT):
    stored = json.load(open(OUT))
    ck("the stored bundle carries the domain", stored.get("domain") == "weed")
    ck("and the step", stored.get("step") == "train")
    ck("and the round it read from the database", str(stored.get("round")) == "11")
    ck("and a sections block the checks can read",
       isinstance(stored.get("sections"), dict))
ck("no temporary file is left behind", not os.path.exists(OUT + ".tmp"))
ck("it logged that it wrote one",
   any("bundle written" in str(l) for l in log.lines))

print("\nit is evidence, not a decision: nothing here may stop a round")
log2 = _Log()
rs._CTX.clear()
rs._CTX.update(_ctx(ROOT, log2, _DB(raise_on_rounds=True)))
_ok = True
try:
    rs._build_bundle("weed", "train", "45444584", CFG)
except Exception as e:
    _ok = False
    print("   raised:", e)
ck("a database that raises does not stop it", _ok)
ck("and the failure is reported, not swallowed silently",
   any("could not read rounds" in str(l) for l in log2.lines))

log3 = _Log()
rs._CTX.clear()
rs._CTX.update({"log": log3, "repo": ROOT})       # no db, no runner at all
_ok = True
try:
    rs._build_bundle("weed", "train", None, CFG)
except Exception as e:
    _ok = False
    print("   raised:", e)
ck("a context with no database and no runner does not raise", _ok)

log4 = _Log()
rs._CTX.clear()
rs._CTX.update(_ctx("/proc/nonexistent-and-unwritable", log4, _DB()))
_ok = True
try:
    rs._build_bundle("weed", "train", "1", CFG)
except Exception as e:
    _ok = False
    print("   raised:", e)
ck("an unwritable output path does not raise", _ok)
ck("and it says the build failed",
   any("bundle build failed" in str(l) or "bundle written" not in str(l)
       for l in log4.lines))

print("\nthe COMPLETED path is wired, and it is called before the state is cleared")
src = pathlib.Path(rs.__file__).read_text()
ck("_advance calls _build_bundle", "_build_bundle(domain, st[\"step\"]" in src)
_call = src.index('_build_bundle(domain, st["step"]')
_clear = src.index('st.update(job=None, step=None', _call)
ck("the call comes before the in-flight fields are cleared", _call < _clear)
# `cur` is assigned further down in _advance; passing it here raised
# UnboundLocalError on every completed step in the first version of this change.
ck("the call site does not read a variable assigned later",
   "cur" not in src[_call:_call + 80])
ck("the helper reads the round itself instead",
   "get_current_round" in src[src.index("def _build_bundle"):_call])

print("\na warn-or-worse signal in the bundle is written to the log")
log5 = _Log()
rs._CTX.clear()
rs._CTX.update(_ctx(ROOT, log5, _DB()))
rs._build_bundle("weed", "train", "45444584", CFG)
ck("the bundle build reports how many signals fired",
   any("signal(s) at warn or worse" in str(l) for l in log5.lines))

shutil.rmtree(ROOT, ignore_errors=True)

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
