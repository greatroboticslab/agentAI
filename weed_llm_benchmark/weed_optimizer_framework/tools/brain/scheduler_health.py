"""Round-scheduler liveness alarm — a stopped unattended loop must not look
like a quiet one.

Why this exists
---------------
On 2026-08-29 17:53 the weed scheduler recorded `step train FAILED (TIMEOUT)`
twice, tripped its stop-loss and paused the domain, exactly as designed. The
pause then held for six days. Nothing on the platform said so: the project page
still rendered the last completed round, the tick kept running with the domain
disabled, and the only trace of the outage was one line in the process journal.
A loop that has stopped working is indistinguishable from a loop with nothing
to do — unless a page says which one it is.

The shape of the check is `sync_health`'s, for the same reason: an alarm that
waits to be *told* about a failure cannot see this class of outage. The
scheduler must keep proving it is alive, and both the absence of that proof and
a proof that says "paused" are the alarm. A dead thread, a crashed worker
process, a wedged tick and a stop-loss pause therefore reach the same verdict,
which is the point — the scheduler is a thread inside the single uvicorn worker,
so from outside there is nothing to distinguish the thread dying from the box
dying, and both stop rounds just as completely.

Severity split (v3.25.0)
------------------------
  crit  no heartbeat, a heartbeat older than 3x the scheduler's own tick, a
        heartbeat reporting `mongo_ok` false, or any domain carrying a
        `paused_reason`. The first three mean rounds have stopped or are running
        unrecorded; the last means they stopped on purpose.
  warn  a step waiting on a supervisor review longer than the review timeout, a
        tick slower than TICK_WARN_S, or this module failing its own check.
        Rounds still advance (or may be advancing unseen); these are the leading
        indicators of the crit above, and a slow tick delays every failure
        detection queued behind it.
  ok    everything else, explicitly including "no domain is enabled". An
        operator-chosen idle loop is a normal state and must not sit red: an
        alarm that is always red is one people learn to scroll past, which is
        the same failure as having no alarm, reached from the other side.

Surfaces
--------
  GET /api/health/scheduler   the verdict as JSON
  banner_html()               a page-top bar, injected on every HTML page when
                              the verdict is not ok
  python -m weed_optimizer_framework.tools.brain.scheduler_health
                              same verdict on stdout; exit 1 when alarming, so a
                              cron or a supervision tick can gate on it.

The heartbeat is `results/framework/scheduler_status.json`, rewritten by the
scheduler on every tick. Cheapness matters: banner_html() runs inside the global
HTML middleware, so the verdict is memoised for CACHE_TTL seconds and never
shells out.
"""
import html
import json
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
_CTX = {}
_CACHE = {"at": 0.0, "val": None}


def _env_float(name, default):
    """A mistyped threshold must not raise at import: this module is imported
    from the site-wide HTML middleware, where an ImportError is swallowed and
    the alarm would then be silently absent — the exact failure it exists for."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None if default is None else float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None if default is None else float(default)


# 3x the scheduler's 120 s tick: one missed tick is a slow squeue, three is a
# stopped loop. The live tick_s in the heartbeat is preferred so a domain moved
# to a slower tick does not read as dead; 360 s covers a heartbeat written
# without it. An explicit environment value overrides both.
_STALE_ENV = _env_float("SCHED_STALE_AFTER_S", None)
STALE_AFTER_S = _STALE_ENV if _STALE_ENV else 360.0
# The scheduler's own review timeout (90 min) — past it the step is waiting on a
# verdict that is not coming, and the deterministic correction should have run.
REVIEW_WAIT_S = _env_float("SCHED_REVIEW_WAIT_S", 5400)
# A tick is one squeue plus a ledger read. Past this it is doing something else.
TICK_WARN_S = _env_float("SCHED_TICK_WARN_S", 90)
CACHE_TTL = 15.0


def _status_path():
    repo = _CTX.get("repo") or os.environ.get("REPO_ROOT") or os.path.expanduser(
        "~/weed_llm_benchmark")
    return os.path.join(str(repo), "results", "framework", "scheduler_status.json")


def _human(sec):
    if sec is None:
        return "never"
    sec = int(sec)
    if sec < 90:
        return "%ds" % sec
    if sec < 5400:
        return "%dm" % (sec // 60)
    if sec < 172800:
        return "%dh %dm" % (sec // 3600, (sec % 3600) // 60)
    return "%dd %dh" % (sec // 86400, (sec % 86400) // 3600)


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _review_wait(review, now):
    """Seconds a step has been awaiting a supervisor verdict, or None.

    Only computable when the heartbeat carries a queue timestamp. A review entry
    without one is reported as awaiting with an unknown age rather than timed
    from an invented start — an alarm built on a made-up clock is worse than no
    alarm on that step.
    """
    if not isinstance(review, dict):
        return None
    if str(review.get("status") or "") != "awaiting":
        return None
    for key in ("queued_at", "awaiting_since", "since", "at", "ts"):
        v = _num(review.get(key), 0.0)
        if v > 0:
            return max(0.0, now - v)
    return None


def verdict(now=None):
    """Scheduler verdict. Never raises — a broken alarm must not break the page.

    A missing or unreadable heartbeat is `crit`, not `ok`: this alarm exists
    because a stopped loop was silent, and defaulting to ok would rebuild
    exactly that silence one level up.
    """
    now = now or time.time()
    st, read_err = {}, None
    try:
        with open(_status_path()) as fh:
            st = json.load(fh) or {}
    except FileNotFoundError:
        read_err = ("no heartbeat file — the round scheduler has not ticked since "
                    "this alarm was installed")
    except Exception as e:
        read_err = "heartbeat unreadable (%s)" % str(e)[:80]
    if not isinstance(st, dict):
        st, read_err = {}, "heartbeat is not a JSON object"

    ts = _num(st.get("ts"), 0.0)
    age = (now - ts) if ts > 0 else None
    tick_s = _num(st.get("tick_s"), 0.0)
    tick_dur = _num(st.get("tick_duration_s"), 0.0)
    # v3.25.0: the scheduler reports whether its round-ledger writes are landing.
    # A ticking loop whose provenance writes fail leaves no record of what ran —
    # the same blind spot as a stopped loop, reached while jobs keep burning SU.
    # Absent (a pre-v3.25.0 heartbeat) is unknown, not healthy-and-not-checked.
    mongo_ok = st.get("mongo_ok")
    mongo_ok = None if mongo_ok is None else bool(mongo_ok)
    mongo_err_ts = _num(st.get("mongo_last_error_ts"), 0.0)
    stale_after = STALE_AFTER_S
    if not _STALE_ENV and tick_s > 0:
        stale_after = 3 * tick_s

    domains = st.get("domains")
    domains = domains if isinstance(domains, dict) else {}
    rows, paused, enabled, overdue = [], [], [], []
    for name, d in sorted(domains.items()):
        d = d if isinstance(d, dict) else {}
        reason = str(d.get("paused_reason") or "").strip()
        review = d.get("review") if isinstance(d.get("review"), dict) else None
        wait = _review_wait(review, now)
        step = d.get("step")
        rows.append({
            "domain": str(name), "enabled": bool(d.get("enabled")),
            "paused_reason": reason, "step": step, "job": d.get("job"),
            "fails": int(_num(d.get("fails"), 0)),
            "rounds_today": int(_num(d.get("rounds_today"), 0)),
            "review_status": (str(review.get("status") or "") if review else None),
            "review_step": (review.get("step") if review else None),
            "review_wait_s": wait,
        })
        if reason:
            paused.append("%s (%s)" % (name, reason[:120]))
        if d.get("enabled"):
            enabled.append(str(name))
        if wait is not None and wait > REVIEW_WAIT_S:
            overdue.append("%s/%s has been awaiting a review for %s"
                           % (name, (review.get("step") or step or "?"), _human(wait)))

    if read_err:
        level, why = "crit", read_err
    elif age is None:
        level, why = "crit", "the heartbeat carries no timestamp — the tick cannot be dated"
    elif age > stale_after:
        level, why = "crit", ("no scheduler tick for %s (one is expected every %s)"
                              % (_human(age), _human(tick_s or 120)))
    elif mongo_ok is False:
        _since = (" (failing for %s)" % _human(max(0.0, now - mongo_err_ts))
                  if mongo_err_ts > 0 else "")
        level, why = "crit", ("the loop is running but its ledger writes are "
                              "failing%s — rounds are advancing unrecorded" % _since)
    elif paused:
        level, why = "crit", "paused — " + "; ".join(paused)
    elif overdue:
        level, why = "warn", "; ".join(overdue)
    elif tick_dur > TICK_WARN_S:
        level, why = "warn", ("the last tick took %s — every failure detection queues "
                              "behind it" % _human(tick_dur))
    elif not enabled:
        level, why = "ok", "no domain is enabled — the loop is idle by configuration"
    else:
        level, why = "ok", ("%d domain(s) advancing, last tick %s ago"
                            % (len(enabled), _human(age)))

    # A pause reported alongside a dead heartbeat is the load-bearing half of the
    # message: it says the loop stopped on purpose and needs a decision, not a
    # restart. Never let the staleness line swallow it.
    if paused and level == "crit" and not why.startswith("paused"):
        why += "; paused — " + "; ".join(paused)

    return {
        "level": level, "ok": level == "ok", "reason": why,
        "heartbeat_ts": ts or None, "heartbeat_age_s": age,
        "heartbeat_age_human": _human(age),
        "tick_s": tick_s or None, "tick_duration_s": tick_dur or None,
        "mongo_ok": mongo_ok, "mongo_last_error_ts": mongo_err_ts or None,
        "domains": rows, "enabled_domains": enabled,
        "paused_domains": paused, "reviews_overdue": overdue,
        "thresholds": {"stale_after_s": stale_after,
                       "review_wait_s": REVIEW_WAIT_S,
                       "tick_warn_s": TICK_WARN_S},
        # relative, not absolute: this payload is served unauthenticated
        "heartbeat": "results/framework/scheduler_status.json",
        "checked_ts": now,
    }


def _cached():
    now = time.time()
    if _CACHE["val"] is None or now - _CACHE["at"] > CACHE_TTL:
        try:
            _CACHE["val"] = verdict(now)
        except Exception as e:                                  # never break a page
            # Fails CLOSED (v3.25.0). This used to report level ok, so an alarm
            # that could not run its own check painted no bar and the page looked
            # healthy — the same silence that hid a six-day pause, rebuilt one
            # level up. An alarm that cannot see must say it cannot see.
            _CACHE["val"] = {"level": "warn", "ok": False,
                             "reason": "scheduler alarm self-check failed (%s) — "
                                       "the state of the loop is unknown from "
                                       "this page" % str(e)[:80]}
        _CACHE["at"] = now
    return _CACHE["val"]


# Deliberately not position:sticky. The stale-data alarm already holds the
# pinned top:0 slot on every page, and two bars pinned to the same offset overlap
# on scroll — whichever renders later hides the other, so one alarm would erase
# the other exactly when both are true. In normal flow the two stack instead.
# The pinned slot stays with the sync bar because it qualifies the content being
# read at that moment; this bar states a site-wide condition and is restated at
# the top of every page (v3.25.0).
_BAR = ('<div id="_schedalarm" role="status" style="position:relative;z-index:9999;'
        'display:flex;gap:.6rem;align-items:center;justify-content:center;flex-wrap:wrap;'
        'padding:.5rem .9rem;font:600 13px/1.4 system-ui,-apple-system,sans-serif;'
        'background:%s;color:%s;border-bottom:1px solid rgba(0,0,0,.15)">'
        '<span>%s</span><span style="font-weight:400;opacity:.9">%s</span>'
        '<a href="/api/health/scheduler" style="color:inherit;text-decoration:underline;'
        'font-weight:400">details</a></div>')


def banner_html():
    """Page-top bar, or '' when the loop is healthy. Injected site-wide."""
    v = _cached()
    if v.get("ok"):
        return ""
    crit = v.get("level") == "crit"
    bg, fg = ("#7f1d1d", "#fff") if crit else ("#fef3c7", "#78350f")
    head = ("ROUNDS ARE NOT ADVANCING — the scheduler is paused or silent"
            if crit else "Scheduler needs attention")
    # The reason carries a job's own failure text (a stop-loss detail, an sbatch
    # error tail); escaped because this string is spliced into every HTML page.
    return _BAR % (bg, fg, head, html.escape(str(v.get("reason", ""))))


@router.get("/api/health/scheduler")
def health_scheduler(request: Request):
    v = verdict()
    return JSONResponse(v, status_code=200 if v["ok"] else 503)


def mount(app, ctx: dict):
    _CTX.update(ctx)
    app.include_router(router)


if __name__ == "__main__":
    import sys
    v = verdict()
    print(json.dumps(v, indent=1))
    sys.exit(0 if v["ok"] else 1)
