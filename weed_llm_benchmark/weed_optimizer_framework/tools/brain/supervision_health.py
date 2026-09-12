"""Supervision-layer alarm — a reviewer that is wired wrong must not look correct.

Why this exists
---------------
`scheduler_health` answers "is the loop running". This answers the question that
sat unanswered while nine campaign reviews were produced by a 4.7 GB model on the
lab's RTX 3060 between 2026-09-04 and 09-11: **is the layer that watches the loop
itself in a state whose output may be reported?**

Every one of those nine records carried a model name, an endpoint, `mode: shadow`
and `applied: false`, and looked exactly like a correctly-wired shadow reviewer.
`applied: false` answers "did the loop act on it". Nothing answered "may this be
quoted", and nothing noticed that the newest completed step had no review after
it, or that a benchmark arm standing on 78 of 149 cases was being read as though
it stood on 149.

So this module checks five things that were each, at some point in this project's
own record, silently wrong:

  1. the reviewer's placement -- where the verdicts are actually coming from;
  2. review freshness -- a step finished and no review followed;
  3. rubric drift -- committed verdicts answering a question the rubric no
     longer asks;
  4. arm completeness -- which benchmark arms are partial, and by how much;
  5. score staleness -- verdicts written after the last time anything scored
     them, so the published numbers describe a corpus that has moved.

Severity split
--------------
  crit  the reviewer is enabled and its verdicts are being produced somewhere
        the placement rule forbids, or a completed step has had no review for
        longer than the review timeout. Both mean the supervision layer is
        reporting something it should not, or nothing at all.
  warn  partial arms, rubric drift, stale scores, or this module failing its own
        check. The numbers still exist; they are just not what a reader would
        assume.
  ok    everything else, explicitly including "the reviewer is switched off".
        An operator-chosen silence is a normal state. An alarm that is always
        red is one people learn to scroll past, which is the same failure as
        having no alarm, reached from the other side.

Nothing here calls a model. It reads committed artifacts only, so it is safe to
run on every request and cheap enough to run on a tick.
"""
import json
import os
import time

MAX_ARMS = 40
REVIEW_STALE_S = float(os.environ.get("SUPERVISION_REVIEW_STALE_S", 5400))


def _read(path):
    try:
        with open(str(path), "rb") as fh:
            return json.loads(fh.read().decode("utf-8")), None
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return None, "%s: %s" % (os.path.basename(str(path)), e)


def _newest(d, suffix=".json"):
    """The newest file in a directory, by mtime, or (None, None)."""
    try:
        names = [n for n in os.listdir(str(d)) if n.endswith(suffix)]
    except Exception:
        return None, None
    best, best_ts = None, None
    for n in names:
        try:
            ts = os.path.getmtime(os.path.join(str(d), n))
        except Exception:
            continue
        if best_ts is None or ts > best_ts:
            best, best_ts = n, ts
    return best, best_ts


def _human(sec):
    if sec is None:
        return "unknown"
    sec = int(sec)
    if sec < 90:
        return "%ds" % sec
    if sec < 5400:
        return "%dm" % (sec // 60)
    if sec < 172800:
        return "%dh" % (sec // 3600)
    return "%dd" % (sec // 86400)


def review_placement(rec):
    """Where a committed review came from, and whether it may be reported.

    Reads the fields `round_scheduler._review_authority` writes. A record from
    before those fields existed reports `authoritative: False` with a reason
    saying so, which is the correct reading of it: nothing on that record
    establishes where it was produced.
    """
    if not isinstance(rec, dict):
        return {"place": "", "authoritative": False,
                "why": "no review record to read"}
    if "authoritative" not in rec:
        return {"place": rec.get("place") or "unknown", "authoritative": False,
                "why": "record predates the placement fields, so nothing on it "
                       "says where the verdict was produced",
                "model": rec.get("model") or "", "endpoint": rec.get("endpoint") or ""}
    return {"place": rec.get("place") or "unknown",
            "authoritative": bool(rec.get("authoritative")),
            "why": rec.get("not_authoritative_why") or "",
            "model": rec.get("model") or "", "endpoint": rec.get("endpoint") or "",
            "tier": rec.get("tier") or ""}


def verdict(brain_dir=None, bench_root=None, now=None, last_step_ts=None,
            review_enabled=None):
    """The supervision layer's state, as {level, reason, checks}.

    `last_step_ts` is when the scheduler last finished a step; pass it and the
    freshness check can see "a step completed and nothing reviewed it", which is
    the failure that ran for two weeks and was found by hand.
    """
    now = float(now if now is not None else time.time())
    checks, errors = [], []
    crit, warn = [], []

    # --- 1. placement ------------------------------------------------------
    latest_rec, latest_ts = None, None
    if brain_dir:
        latest_rec, err = _read(os.path.join(str(brain_dir), "latest_review.json"))
        if err:
            errors.append(err)
        _, latest_ts = _newest(os.path.join(str(brain_dir), "reviews"))
        if latest_rec and latest_rec.get("ts"):
            latest_ts = float(latest_rec["ts"])
    place = review_placement(latest_rec)
    if review_enabled is False:
        checks.append({"check": "placement", "level": "ok",
                       "detail": "the reviewer is switched off; no verdict is being produced"})
    elif latest_rec is None:
        checks.append({"check": "placement", "level": "ok",
                       "detail": "no review has been written in this domain yet"})
    elif not place["authoritative"]:
        msg = ("reviews are coming from %s (%s at %s) and may not be reported as "
               "results: %s" % (place.get("tier") or "an unknown tier",
                                place.get("model") or "?",
                                place.get("endpoint") or "?", place["why"]))
        crit.append(msg)
        checks.append({"check": "placement", "level": "crit", "detail": msg,
                       **place})
    else:
        checks.append({"check": "placement", "level": "ok",
                       "detail": "reviews are produced on the cluster by %s"
                                 % (place.get("model") or "?"), **place})

    # --- 2. freshness ------------------------------------------------------
    if review_enabled is False or last_step_ts is None:
        checks.append({"check": "freshness", "level": "ok",
                       "detail": "no completed step to review, or the reviewer is off"})
    else:
        age = None if latest_ts is None else now - float(latest_ts)
        gap = float(last_step_ts) - float(latest_ts or 0)
        if latest_ts is None or gap > REVIEW_STALE_S:
            msg = ("a step finished %s ago and the newest review is %s old: the "
                   "reviewer is enabled but nothing followed that step"
                   % (_human(now - float(last_step_ts)), _human(age)))
            crit.append(msg)
            checks.append({"check": "freshness", "level": "crit", "detail": msg,
                           "latest_review_ts": latest_ts,
                           "last_step_ts": last_step_ts})
        else:
            checks.append({"check": "freshness", "level": "ok",
                           "detail": "newest review is %s old" % _human(age),
                           "latest_review_ts": latest_ts})

    # --- 3, 4, 5. the benchmark corpus ------------------------------------
    if bench_root:
        checks.extend(_bench_checks(str(bench_root), warn, errors))

    if errors:
        warn.append("this check could not read %d artifact(s)" % len(errors))
    level = "crit" if crit else ("warn" if warn else "ok")
    reason = (crit or warn or ["the supervision layer is wired and current"])[0]
    return {"level": level, "ok": level == "ok", "reason": reason,
            "crit": crit, "warn": warn, "checks": checks, "errors": errors,
            "ts": now, "review_stale_s": REVIEW_STALE_S}


def _bench_checks(root, warn, errors):
    """Rubric drift, arm completeness and score staleness, from committed files."""
    out = []
    split, err = _read(os.path.join(root, "split.json"))
    if err:
        errors.append(err)
    n_dev = len((split or {}).get("dev") or []) or None

    vdir = os.path.join(root, "verdicts")
    if not os.path.isdir(vdir):
        # The benchmark corpus lives on the cluster; the lab box does not carry
        # it. "0 arms, all complete" would be vacuously green and is exactly the
        # kind of sentence this module exists to stop being written.
        return [{"check": "arm_completeness", "level": "ok",
                 "detail": "no benchmark corpus on this machine (%s); nothing to "
                           "check here, and nothing here is evidence that the "
                           "arms elsewhere are complete" % vdir,
                 "arms": [], "split_n": n_dev, "present": False}]
    arms, newest_verdict_ts = [], None
    rubrics = {}
    try:
        names = sorted(os.listdir(vdir))
    except Exception:
        names = []
    for name in names[:MAX_ARMS]:
        d = os.path.join(vdir, name)
        if not os.path.isdir(d):
            continue
        try:
            files = [f for f in os.listdir(d) if f.endswith(".json")]
        except Exception:
            continue
        _, ts = _newest(d)
        if ts and (newest_verdict_ts is None or ts > newest_verdict_ts):
            newest_verdict_ts = ts
        rec = None
        if files:
            rec, err = _read(os.path.join(d, sorted(files)[0]))
            if err:
                errors.append(err)
        rub = (rec or {}).get("rubric_sha256") or ""
        rubrics.setdefault(rub, []).append(name)
        arms.append({"arm": name, "verdicts": len(files),
                     "complete": (n_dev is None or len(files) >= n_dev),
                     "rubric_sha256": rub})

    partial = [a for a in arms if not a["complete"]]
    if partial:
        msg = ("%d benchmark arm(s) are partial and must not be read as complete: %s"
               % (len(partial), ", ".join("%s %d of %s" % (a["arm"], a["verdicts"],
                                                           n_dev) for a in partial[:6])))
        warn.append(msg)
        out.append({"check": "arm_completeness", "level": "warn", "detail": msg,
                    "arms": arms, "split_n": n_dev})
    else:
        out.append({"check": "arm_completeness", "level": "ok",
                    "detail": "%d arm(s), all at the full split" % len(arms),
                    "arms": arms, "split_n": n_dev})

    # A verdict carries the sha of the rubric it was answered under. More than
    # one sha in the directory means two arms answered different questions and
    # the scorer is being asked to compare them.
    rubrics.pop("", None)
    if len(rubrics) > 1:
        msg = ("committed verdicts carry %d different rubric hashes, so not every "
               "arm answered the same question: %s" % (
                   len(rubrics),
                   "; ".join("%s.. %d arm(s)" % (k[:8], len(v))
                             for k, v in list(rubrics.items())[:4])))
        warn.append(msg)
        out.append({"check": "rubric_drift", "level": "warn", "detail": msg,
                    "rubrics": {k: v for k, v in list(rubrics.items())[:4]}})
    else:
        out.append({"check": "rubric_drift", "level": "ok",
                    "detail": "every arm answered under one rubric"})

    # If any verdict is newer than the newest scoring run, the published numbers
    # describe a corpus that has since moved.
    _, score_ts = _newest(os.path.join(root, "results"))
    for cand in ("supervision_rescore", "rescore"):
        _, t2 = _newest(os.path.join(os.path.dirname(root), cand, "results"))
        if t2 and (score_ts is None or t2 > score_ts):
            score_ts = t2
    if newest_verdict_ts and score_ts and newest_verdict_ts > score_ts + 60:
        msg = ("verdicts were written %s after the last scoring run, so the "
               "published numbers are stale"
               % _human(newest_verdict_ts - score_ts))
        warn.append(msg)
        out.append({"check": "score_staleness", "level": "warn", "detail": msg,
                    "newest_verdict_ts": newest_verdict_ts, "scored_ts": score_ts})
    else:
        out.append({"check": "score_staleness", "level": "ok",
                    "detail": "the last scoring run is at or after the newest verdict",
                    "newest_verdict_ts": newest_verdict_ts, "scored_ts": score_ts})
    return out


# ---------------------------------------------------------------- HTTP surface
_CTX = {}


def _repo():
    return str(_CTX.get("repo") or ".")


def live_verdict(domain="weed", now=None):
    """The verdict for a running platform, reading the paths this repo uses.

    Pulls `last_step_ts` and whether the reviewer is enabled off the scheduler's
    own heartbeat rather than re-deriving them, so the two alarms cannot disagree
    about whether a step finished.
    """
    fw = os.path.join(_repo(), "results", "framework")
    brain_dir = os.path.join(fw, "_brain", str(domain))
    bench_root = os.path.join(fw, "supervision_bench")
    last_step_ts, enabled = None, None
    st, _err = _read(os.path.join(fw, "scheduler_status.json"))
    # The heartbeat writes `domains` as a MAPPING of domain -> state, while the
    # /api/health/scheduler response renders it as a LIST of rows carrying a
    # "domain" key. Both shapes are real and this reads either: the first
    # version of this function assumed the list, iterated the mapping's keys and
    # tried to call .get() on a string. An alarm whose own reader crashes is
    # worse than no alarm, so it also never raises out of here.
    doms = (st or {}).get("domains")
    row = None
    if isinstance(doms, dict):
        row = doms.get(domain)
    elif isinstance(doms, list):
        row = next((d for d in doms
                    if isinstance(d, dict) and d.get("domain") == domain), None)
    if isinstance(row, dict):
        enabled = bool(row.get("enabled"))
        for k in ("last_step_done_ts", "step_done_ts", "last_done_ts"):
            v = row.get(k)
            if isinstance(v, (int, float)) and v:
                last_step_ts = float(v)
                break
    if enabled is False:
        # A disabled domain finishes no steps, so freshness has nothing to say.
        last_step_ts = None
    return verdict(brain_dir=brain_dir, bench_root=bench_root, now=now,
                   last_step_ts=last_step_ts,
                   review_enabled=(None if enabled is None else enabled))


def mount(app, ctx: dict):
    """Register GET /api/health/supervision. Never raises into the app."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
    _CTX.update(ctx or {})
    router = APIRouter()

    @router.get("/api/health/supervision")
    def health_supervision():
        try:
            v = live_verdict()
        except Exception as e:      # an alarm must not be the thing that breaks
            v = {"level": "warn", "ok": False,
                 "reason": "the supervision alarm failed its own check: %s" % e,
                 "crit": [], "warn": [], "checks": [], "errors": [str(e)]}
        return JSONResponse(v, status_code=200 if v.get("ok") else 503)

    app.include_router(router)
