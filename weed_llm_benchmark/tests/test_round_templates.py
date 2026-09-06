#!/usr/bin/env python3
"""WP1 step-template tests (no Mongo / cluster / network needed).

The scheduler's sbatch lines moved out of the `_WEED_STEPS` literals into the
per-domain config (`steps` templates + `round_params`) so a correction can change
the next submitted command without a code change. That is only safe if the
templates, rendered with their DEFAULT round_params, come out byte-identical to
the strings the loop has been submitting: one stray character inside a --export
list is a silently different experiment, and the whole campaign is comparisons
between runs that must differ only where we say they differ.

Covered:
  * db.DEFAULT_DOMAIN_CONFIG collect/filter templates == the v3.24 literals
  * the train template renders with the fields the scheduler supplies, and its
    time cap sits inside its own walltime (the 2026-08-29 TIMEOUT fix)
  * round_scheduler._render_local produces the same bytes as a bare .format
  * round_scheduler._step_command falls back to _WEED_STEPS, and says so once,
    when a domain config carries no steps block

Run:  python -m pytest tests/test_round_templates.py   (or) python tests/test_round_templates.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db  # noqa: E402
from weed_optimizer_framework.tools import round_scheduler as rs  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, level, msg, *a):
        self.lines.append((level, (msg % a) if a else str(msg)))

    def info(self, m, *a):
        self._add("info", m, *a)

    def warning(self, m, *a):
        self._add("warning", m, *a)

    def error(self, m, *a):
        self._add("error", m, *a)


D = db.DEFAULT_DOMAIN_CONFIG
STEPS = D.get("steps") if isinstance(D.get("steps"), dict) else {}
PARAMS = D.get("round_params") if isinstance(D.get("round_params"), dict) else {}

ck("DEFAULT_DOMAIN_CONFIG carries step templates", bool(STEPS))
ck("DEFAULT_DOMAIN_CONFIG carries round_params", bool(PARAMS))


def _render(step, **extra):
    """Render one template the way the scheduler does: round_params first, then
    the per-submission fields."""
    fields = dict(PARAMS)
    fields.update({"round": 7, "step": step, "domain": "weed",
                   "iter_name": "rnd7_" + step, "trace": ""})
    fields.update(extra)
    return STEPS[step].format(**fields)


# ---- byte equality with the literals the loop submitted before v3.25.0 ----
if STEPS and PARAMS:
    for step in ("collect", "filter"):
        ck("template present for step '%s'" % step, step in STEPS)

    if "collect" in STEPS:
        got = _render("collect")
        want = rs._WEED_STEPS["collect"][0]
        ck("collect renders byte-identically to the _WEED_STEPS literal", got == want)
        if got != want:
            print("      want: %r\n      got : %r" % (want, got))

    if "filter" in STEPS:
        got = _render("filter")
        want = rs._WEED_STEPS["filter"][0]
        ck("filter renders byte-identically to the _WEED_STEPS literal", got == want)
        if got != want:
            print("      want: %r\n      got : %r" % (want, got))

    # ---- the train template is the one that CHANGES, on purpose -----------
    if "train" in STEPS:
        train = _render("train")
        ck("train renders without a missing field", bool(train))
        for token in ("TIER=", "MIN_DINO_SCORE=", "TRAIN_EPOCHS=", "TRAIN_TIME_H=",
                      "ITER_NAME=", "BRAIN_DOMAIN=", "run_m1_merged_seeds.sh"):
            ck("train line carries %s" % token, token in train)
        ck("train line carries the round's iteration name", "rnd7_train" in train)

    # ---- the parameter that makes the 08-29 recipe finish -----------------
    try:
        cap = float(PARAMS.get("train_time_cap_h"))
        wall = float(PARAMS.get("train_time_h"))
        ck("train_time_cap_h is inside train_time_h (a capped run still saves "
           "best.pt before the wall)", 0 < cap < wall)
    except (TypeError, ValueError):
        ck("train_time_cap_h / train_time_h are numeric", False)
    ck("default epochs is the sealed 60", int(PARAMS.get("epochs", 0)) == 60)
    ck("default imgsz is the sealed 640", int(PARAMS.get("imgsz", 0)) == 640)
    ck("default patience is the sealed 20", int(PARAMS.get("patience", 0)) == 20)

    # ---- the scheduler's own renderer agrees with a bare .format ----------
    rs._CTX["log"] = _Log()
    fields = dict(PARAMS)
    fields.update({"round": 7, "step": "collect", "domain": "weed",
                   "iter_name": "rnd7_collect", "trace": ""})
    ck("_render_local matches str.format on the collect template",
       rs._render_local("weed", "collect", STEPS["collect"], fields)
       == _render("collect"))
    ck("_render_local reports a template with an unknown field instead of "
       "submitting it", rs._render_local("weed", "collect", "sbatch {nope}",
                                         fields) == "")


# ---- the scheduler's real path: db config + db.render_step_command --------
class _RealTemplateDB:
    """The real renderer and the real defaults, without touching Mongo."""

    @staticmethod
    def get_domain_config(domain):
        import copy
        return copy.deepcopy(db.DEFAULT_DOMAIN_CONFIG)

    render_step_command = staticmethod(db.render_step_command)


rs._CTX["log"] = _Log()
rs._CTX["db"] = _RealTemplateDB()
rs._FALLBACK_LOGGED.clear()

cmd, params, used_fallback = rs._step_command("weed", "collect", 7)
ck("_step_command renders collect through db.render_step_command, byte-identical",
   cmd == rs._WEED_STEPS["collect"][0] and used_fallback is False)
cmd, params, used_fallback = rs._step_command("weed", "filter", 7)
ck("_step_command renders filter through db.render_step_command, byte-identical",
   cmd == rs._WEED_STEPS["filter"][0] and used_fallback is False)

cmd, params, used_fallback = rs._step_command("weed", "train", 7)
ck("_step_command renders train from the config, not the fallback literal",
   used_fallback is False and "TRAIN_TIME_H=" in cmd)
ck("train submission carries the round's iteration name",
   "ITER_NAME=rnd7_train" in cmd)
ck("train submission leaves the trace path to the job",
   "BRAIN_TRACE= " in cmd or cmd.rstrip().endswith("BRAIN_TRACE="))
ck("train submission records the parameters it rendered with",
   params.get("epochs") == 60 and params.get("iter_name") == "rnd7_train")


# ---- fallback path: a domain config with no steps block ------------------
class _NoStepsDB:
    @staticmethod
    def get_domain_config(domain):
        return {"round_params": {}}


_log = _Log()
rs._CTX["log"] = _log
rs._CTX["db"] = _NoStepsDB()
rs._FALLBACK_LOGGED.clear()

cmd, params, used_fallback = rs._step_command("weed", "collect", 7)
ck("no template -> the built-in collect literal is used",
   cmd == rs._WEED_STEPS["collect"][0] and used_fallback is True)
ck("fallback carries the per-submission fields",
   params.get("iter_name") == "rnd7_collect" and params.get("round") == 7)
ck("fallback is reported once", sum(1 for lvl, m in _log.lines
                                    if "built-in fallback command" in m) == 1)
rs._step_command("weed", "collect", 8)
ck("fallback is not reported again for the same step",
   sum(1 for lvl, m in _log.lines if "built-in fallback command" in m) == 1)

cmd, _p, used_fallback = rs._step_command("weed", "nosuchstep", 7)
ck("an unwired step yields no command", cmd is None and used_fallback is True)


# ---- the policy gate authorises what the command actually carries (v3.29.1) --
# The loop is an actor with real permissions on a shared allocation, so every
# submission passes the same gate the web path uses. The parameters authorised
# must be the ones the template substitutes: handing the whole round dict to the
# gate made all three steps refuse, because a bound is declared per action and
# `epochs` arriving with a collect submission is an undeclared parameter reaching
# a command line. This is the second time a validation added to this path would
# have silently stopped the campaign, so it is asserted rather than reasoned about.
from weed_optimizer_framework.tools.brain import policy as _policy  # noqa: E402

_D = db.DEFAULT_DOMAIN_CONFIG
ck("collect renders only its own two fields",
   db.step_fields(_D, "collect") == ["collect_time_h", "max_new"])
ck("filter renders no parameters at all", db.step_fields(_D, "filter") == [])
ck("train renders the recipe fields",
   set(db.step_fields(_D, "train")) >= {"epochs", "train_time_h", "train_time_cap_h",
                                        "imgsz", "patience", "tier", "min_dino_score"})
ck("an unknown step renders nothing rather than guessing",
   db.step_fields(_D, "nosuchstep") == [])

for _step in ("collect", "filter", "train"):
    _sub = {k: v for k, v in _D["round_params"].items()
            if k in db.step_fields(_D, _step)}
    _d = _policy.authorize("round-scheduler", "round_" + _step, _sub, None, None)
    ck("the live %s step passes the gate" % _step,
       _d["allowed"] and not _d["needs_approval"])

_bad = {k: v for k, v in _D["round_params"].items()
        if k in db.step_fields(_D, "train")}
_bad["epochs"] = 100000
ck("an out-of-bounds epoch count is refused",
   not _policy.authorize("round-scheduler", "round_train", _bad, None, None)["allowed"])

_extra = {k: v for k, v in _D["round_params"].items()
          if k in db.step_fields(_D, "collect")}
_extra["epochs"] = 60
ck("a parameter the collect template never renders is refused",
   not _policy.authorize("round-scheduler", "round_collect", _extra, None, None)["allowed"])

# An earlier block left a stub wired; the gate asks the CURRENT database for
# the fields, so point it at the real one for this check.
rs._CTX["db"] = db
ck("the scheduler filters params down to the rendered set",
   rs._gate_params(_D, "collect", _D["round_params"]) ==
   {"collect_time_h": 10, "max_new": 3})
ck("refused is a status the ledger can record, distinct from failed and skipped",
   "refused" in db._ROUND_STATUSES)


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
