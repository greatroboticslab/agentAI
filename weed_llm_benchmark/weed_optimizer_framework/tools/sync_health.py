"""Sync freshness alarm — makes a stalled cluster->lab sync impossible to mistake
for a quiet period.

Why this exists
---------------
Twice now the `weed-sync` oneshot has hung on a blocked ssh and simply *stayed*
activating. systemd will not start a second run while the first lives, so the
timer stopped firing, nothing failed, nothing logged, and the platform served
three-week-old data while looking completely healthy. The second occurrence was
found by hand, 22 days after the last successful sync.

The lesson is about the shape of the check, not the bug: an alarm that waits to
be *told* about a failure cannot see this class of outage, because a wedged
process reports nothing. So the alarm here is inverted — the sync must keep
proving it is alive, and the absence of that proof is itself the alarm. A
missing heartbeat file, a hung box, a disabled timer, a killed run and a
crashed run all produce the same verdict, which is the point.

What it reports is *data freshness* — the age of the last successful sync —
because that is what a person looking at the dashboard actually needs to know.
The cause (hung / failed / never ran) rides along as detail.

Freshness is measured on the METADATA leg (registry pulled, local paths
rewritten, Mongo re-mirrored), not on the whole run. That leg is everything the
dashboard actually serves; the image trees behind it are hundreds of gigabytes
and can legitimately need several cycles to catch up after an outage. Alarming on
the full run would leave the banner red through every one of those cycles, and an
alarm that is always red is one people learn to scroll past — which is the same
failure as having no alarm, arrived at from the other direction. Image-tree lag is
still reported, as detail rather than as an alarm.

Surfaces
--------
  GET /api/health/sync      the verdict as JSON
  banner_html()             a page-top bar, injected on every HTML page when
                            the verdict is not ok
  python -m weed_optimizer_framework.tools.sync_health
                            same verdict on stdout; exit 1 when alarming, so a
                            cron or a supervision tick can gate on it.

Cheapness matters: banner_html() runs inside the global HTML middleware, so the
verdict is memoised for CACHE_TTL seconds and never shells out.
"""
import json
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
_CTX = {}
_CACHE = {"at": 0.0, "val": None}

# The timer runs every 30 min. WARN at 6 consecutive misses, CRIT at a full day
# of missed cycles — early enough to catch a wedge the same working day, late
# enough that a slow multi-GB dataset pull does not cry wolf.
WARN_AFTER_S = float(os.environ.get("SYNC_WARN_AFTER_S", 3 * 3600))
CRIT_AFTER_S = float(os.environ.get("SYNC_CRIT_AFTER_S", 12 * 3600))
# A single run has never legitimately exceeded ~40 min. Past this it is wedged.
HUNG_AFTER_S = float(os.environ.get("SYNC_HUNG_AFTER_S", 2 * 3600))
CACHE_TTL = 30.0


def _status_path():
    repo = _CTX.get("repo") or os.environ.get("REPO_ROOT") or os.path.expanduser(
        "~/weed_llm_benchmark")
    return os.path.join(str(repo), "results", "framework", "sync_status.json")


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


def verdict(now=None):
    """Freshness verdict. Never raises — a broken alarm must not break the page.

    A missing or unreadable heartbeat is `crit`, not `ok`: this alarm exists
    because silence was indistinguishable from health, and defaulting to ok
    would rebuild exactly that.
    """
    now = now or time.time()
    st, read_err = {}, None
    try:
        with open(_status_path()) as fh:
            st = json.load(fh) or {}
    except FileNotFoundError:
        read_err = "no heartbeat file — the sync has not reported since this alarm was installed"
    except Exception as e:
        read_err = "heartbeat unreadable (%s)" % str(e)[:80]

    # Metadata freshness drives the verdict; full-run freshness is detail.
    last_full = st.get("last_success_ts")
    last_ok = st.get("last_meta_success_ts") or last_full
    age = (now - last_ok) if last_ok else None
    full_age = (now - last_full) if last_full else None
    started = st.get("run_started_ts")
    in_flight = bool(st.get("in_flight"))
    in_flight_for = (now - started) if (in_flight and started) else None

    if read_err:
        level, why = "crit", read_err
    elif age is None:
        level, why = "crit", "no successful sync has ever been recorded"
    elif age > CRIT_AFTER_S:
        level, why = "crit", "registry last refreshed %s ago" % _human(age)
    elif age > WARN_AFTER_S:
        level, why = "warn", "registry last refreshed %s ago" % _human(age)
    else:
        level, why = "ok", "registry last refreshed %s ago" % _human(age)

    # A run wedged in flight is worth naming even while the data is still fresh
    # enough to pass — it is the leading indicator of the outage above.
    if in_flight_for and in_flight_for > HUNG_AFTER_S:
        why += "; a run has been in flight for %s and is likely wedged" % _human(in_flight_for)
        if level == "ok":
            level = "warn"

    # Image trees lagging far behind the metadata is worth saying out loud, but
    # it is not the same outage: the pages still render correct, current records.
    if full_age and age is not None and full_age - age > CRIT_AFTER_S:
        why += ("; image trees are %s behind and still catching up"
                % _human(full_age))

    return {
        "level": level, "ok": level == "ok", "reason": why,
        "last_success_ts": last_ok, "last_success_age_s": age,
        "last_full_sync_ts": last_full, "last_full_sync_age_s": full_age,
        "last_full_sync_human": _human(full_age),
        "last_success_human": _human(age),
        "last_outcome": st.get("last_outcome"), "last_detail": st.get("detail"),
        "stage": st.get("stage"), "in_flight": in_flight,
        "in_flight_for_s": in_flight_for,
        "run_started_ts": started,
        "thresholds": {"warn_after_s": WARN_AFTER_S, "crit_after_s": CRIT_AFTER_S,
                       "hung_after_s": HUNG_AFTER_S},
        # relative, not absolute: this payload is served unauthenticated
        "heartbeat": "results/framework/sync_status.json",
        "checked_ts": now,
    }


def _cached():
    now = time.time()
    if _CACHE["val"] is None or now - _CACHE["at"] > CACHE_TTL:
        try:
            _CACHE["val"] = verdict(now)
        except Exception as e:                                  # never break a page
            _CACHE["val"] = {"level": "ok", "ok": True,
                             "reason": "alarm self-check failed: %s" % str(e)[:80]}
        _CACHE["at"] = now
    return _CACHE["val"]


_BAR = ('<div id="_syncalarm" role="status" style="position:sticky;top:0;z-index:9999;'
        'display:flex;gap:.6rem;align-items:center;justify-content:center;flex-wrap:wrap;'
        'padding:.5rem .9rem;font:600 13px/1.4 system-ui,-apple-system,sans-serif;'
        'background:%s;color:%s;border-bottom:1px solid rgba(0,0,0,.15)">'
        '<span>%s</span><span style="font-weight:400;opacity:.9">%s</span>'
        '<a href="/api/health/sync" style="color:inherit;text-decoration:underline;'
        'font-weight:400">details</a></div>')


def banner_html():
    """Page-top bar, or '' when the data is fresh. Injected site-wide."""
    v = _cached()
    if v.get("ok"):
        return ""
    crit = v.get("level") == "crit"
    bg, fg = ("#7f1d1d", "#fff") if crit else ("#fef3c7", "#78350f")
    head = ("DATA MAY BE STALE — the registry has not refreshed"
            if crit else "Cluster sync is behind")
    return _BAR % (bg, fg, head, v.get("reason", ""))


@router.get("/api/health/sync")
def health_sync(request: Request):
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
