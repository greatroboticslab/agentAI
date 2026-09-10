#!/usr/bin/env python3
"""Tests for the liveness alarm — which had none.

The module every page load consults, and the one thing standing between a
stopped campaign and six quiet days, was the only check system in this tree with
no test at all. These pin the behaviour it already had, and the branch added
after it was found to have the same blind spot it exists to close: `_cfg()`
swallowed every exception and returned no domains, the heartbeat reported no
domains, and the verdict read that as "idle by configuration" — green. A
truncated config file and a deliberately idle loop painted the identical page.

Run:  python tests/test_scheduler_health.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import scheduler_health as H  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


NOW = 1789000000.0
ROOT = tempfile.mkdtemp(prefix="sched_health_")


def verdict(**over):
    """A verdict over a heartbeat built from a healthy baseline plus overrides."""
    hb = {"ts": NOW, "tick_s": 120.0, "tick_duration_s": 1.2, "mongo_ok": True,
          "config_ok": True, "config_present": True, "config_error": "",
          "config_domains_seen": 1,
          "domains": {"weed": {"enabled": True, "paused_reason": "", "job": "1",
                               "step": "train", "fails": 0, "review": None,
                               "rounds_today": 1, "unknown_ticks": 0}}}
    for k, v in over.items():
        if k == "domain":
            hb["domains"]["weed"].update(v)
        elif v is _ABSENT:
            hb.pop(k, None)
        else:
            hb[k] = v
    d = os.path.join(ROOT, "results", "framework")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "scheduler_status.json"), "w") as fh:
        json.dump(hb, fh)
    H._CTX.update({"repo": ROOT})
    H._CACHE.update({"at": 0.0, "val": None})
    return H.verdict(now=NOW)


class _Absent:
    pass


_ABSENT = _Absent()

print("the behaviour it already had")
ck("a healthy loop is ok", verdict()["level"] == "ok")
ck("and says how many domains are advancing",
   "advancing" in verdict()["reason"])
ck("a stale heartbeat is crit", verdict(ts=NOW - 10000)["level"] == "crit")
ck("mongo down is crit", verdict(mongo_ok=False)["level"] == "crit")
ck("a paused domain is crit",
   verdict(domain={"paused_reason": "stop-loss: two failures"})["level"] == "crit")
ck("and the pause reason is carried, not summarised away",
   "stop-loss" in verdict(domain={"paused_reason": "stop-loss: two failures"})["reason"])
ck("a slow tick is warn, not crit",
   verdict(tick_duration_s=H.TICK_WARN_S + 5)["level"] == "warn")
ck("an idle loop is ok, not red",
   verdict(domain={"enabled": False})["level"] == "ok")

print("\nno heartbeat at all is crit, because absence is the alarm")
shutil.rmtree(os.path.join(ROOT, "results"), ignore_errors=True)
H._CTX.update({"repo": ROOT})
H._CACHE.update({"at": 0.0, "val": None})
_none = H.verdict(now=NOW)
ck("a missing heartbeat is crit", _none["level"] == "crit")
ck("and it is not reported as ok", _none["ok"] is False)

print("\nthe blind spot this alarm had about itself")
# _cfg() swallowed everything and returned {"domains": {}}; the verdict then
# said "no domain is enabled — the loop is idle by configuration". Green.
_bad = verdict(config_ok=False, config_error="JSONDecodeError: Expecting value",
               domains={})
ck("an unreadable configuration is crit", _bad["level"] == "crit")
ck("and it says the enabled state is unknown, not that the loop is idle",
   "unknown" in _bad["reason"] and "idle by configuration" not in _bad["reason"])
ck("and it carries the parse error so it can be fixed",
   "JSONDecodeError" in _bad["reason"])

_fresh = verdict(config_ok=True, config_present=False, domains={})
ck("a deploy that has never been configured is ok, not crit",
   _fresh["level"] == "ok")
ck("and says so in its own words",
   "no scheduler configuration exists yet" in _fresh["reason"])

# A heartbeat written before the field existed must keep its old behaviour. The
# same rule mongo_ok already follows: absent is not a new red across every
# machine that has not restarted yet.
_old = verdict(config_ok=_ABSENT, config_present=_ABSENT, domains={})
ck("a pre-upgrade heartbeat is unchanged", _old["level"] == "ok")
ck("and reads as the idle loop it used to",
   "idle by configuration" in _old["reason"])
ck("the field is reported as absent rather than assumed true",
   _old["config_ok"] is None)

print("\nit fails closed on its own failure")
_saved = H._status_path
try:
    H._status_path = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    H._CACHE.update({"at": 0.0, "val": None})
    _self = H._cached()
    ck("a module that cannot check itself does not report ok",
       _self.get("ok") is not True)
finally:
    H._status_path = _saved
    H._CACHE.update({"at": 0.0, "val": None})

shutil.rmtree(ROOT, ignore_errors=True)

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
