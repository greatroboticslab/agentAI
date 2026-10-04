"""SU envelope of one INC campaign, and the SU-ledger writer for finished experiments.

docs/INC_AUTOPILOT.md, section d: the executor passes a real `budget_state` to
`policy.authorize`, so the budget escalation that never ran in production
finally does, and the campaign-envelope rule has a balance to check against.

The envelope
------------
A campaign draws on a sub-envelope of the domain's `budget.su_envelope`
(db.py DEFAULT_DOMAIN_CONFIG: 1500 SU). The sub-envelope is the campaign's
`envelope_su`, 300 SU when the campaign does not set one, and never more than
the domain envelope. A daily cap applies only when one is declared: the
campaign's `daily_cap_su`, else the domain's `budget.daily_cap` when the
domain declares one, and never more than a domain cap that is declared.

No time-based throttle by default (docs/CONTINUOUS_LOOP.md 6.6, amendment
2026-10-04, decided by the owner). db.DEFAULT_DOMAIN_CONFIG carried
`budget.daily_cap` 120 and the stream campaign's defaults a 120 SU daily cap
and a 350 SU monthly window. On 2026-10-04 the cluster sat idle about 9 h
while the stream's next segment was filed for a person only because "116.2 SU
exceeds the 86.52 SU left under today's cap of 120" (the campaign had set
180; the code default cut it to 120) and L18 had run once in the last 24 h.
A cap that only delays healthy work protects nothing the fuses below do not,
so none has a default: an absent daily cap or window is no cap (no refusal,
no reason, reported as none). A cap a campaign or a domain declares still
applies. The fuses stay: the lifetime envelopes (the campaign's and the
domain's, against a runaway bug), the executor's in-flight and per-source
limits, D27's disk headroom, the stop-losses and the allocation's end date.

What counts against it
----------------------
* **Spent**: SU the ledger holds for the campaign, one entry per experiment
  unit, written by `record_report_spend` from `report.gpu_hours` once an
  experiment is done (step `inc:<campaign>`, job `inc:<exp>:<unit>`), and
  one for the job that built it (`record_build_spend`, job `inc:<exp>:build`),
  which the report's gpu_hours do not hold.
* **Committed**: the estimate of every execution the executor charged
  (`charged: true` in its execution log) whose experiment has no spend in the
  ledger yet. A build's estimate is released when the spend of the
  experiment it builds (its own `params.exp`, whatever `child_exp` the
  record states) is recorded. An action with no report of its own
  (relevance, audit) keeps its walltime estimate for good, which overstates
  rather than understates.
* **Today**: estimates charged since 00:00 UTC, reported always and checked
  only against a declared daily cap.

The executor writes a `started` record (charged) before it runs anything and
an outcome record with the same `run_id` after; `fold` keeps the last record
of each run. A run whose outcome was never written (the lab stopped between
the ssh and the log, or the log write failed) stays charged: it may have run.

Unknown is never 0 here either: a campaign with no envelope reports
`remaining_su: None`, and `fits()` refuses an estimate it cannot check.

The writer reads only `exp`, `done`, `done_utc` and `gpu_hours` from a report,
never `final` or any exam, because budget code is decision code.
"""
from __future__ import annotations

import datetime
import re

from . import model as M
from ..brain import su_ledger

DEFAULT_SUB_ENVELOPE_SU = 300.0
GPU_TYPE = "v100-32"            # run_inc_job.sh: --gres=gpu:v100-32:1
STEP_PREFIX = "inc:"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# The only report keys the writer may read (budget code is decision code, docs
# section d: no test, ood or imageweeds value may reach a decision).
REPORT_KEYS = ("exp", "done", "done_utc", "gpu_hours")


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def campaign_step(name):
    """The su_ledger `step` every entry of campaign `name` is filed under."""
    if not _NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r does not match %s" % (name, _NAME_RE.pattern))
    return STEP_PREFIX + str(name)


def domain_budget(given=None):
    """(budget block, source): the caller's, else db.py's default, else empty."""
    if isinstance(given, dict) and given:
        return dict(given), "caller"
    try:
        from .. import db
        return dict(db.DEFAULT_DOMAIN_CONFIG.get("budget") or {}), "db.DEFAULT_DOMAIN_CONFIG"
    except Exception:
        return {}, "unavailable"


def envelope(campaign, given_budget=None):
    """The campaign's sub-envelope and daily cap, with where each came from.

    The daily cap is the campaign's `daily_cap_su`, else the domain's declared
    `daily_cap`; with neither it is None ("none"): no daily cap (amendment
    2026-10-04: db.py no longer declares one by default)."""
    c = campaign if isinstance(campaign, dict) else {}
    dom, dom_src = domain_budget(given_budget)
    dom_env = _num(dom.get("su_envelope", dom.get("envelope")))
    reasons = []
    want = _num(c.get("envelope_su"))
    src = "campaign.envelope_su"
    if want is None:
        want, src = DEFAULT_SUB_ENVELOPE_SU, "default sub-envelope"
    if want < 0:
        reasons.append("campaign envelope_su %r is negative; treated as 0" % want)
        want = 0.0
    env = want
    if dom_env is not None and dom_env < want:
        env = dom_env
        reasons.append("capped at the domain su_envelope %.6g (%s)" % (dom_env, dom_src))
    dom_daily = _num(dom.get("daily_cap"))
    daily = _num(c.get("daily_cap_su"))
    want_daily = daily
    daily_src = "campaign.daily_cap_su"
    if daily is None:
        daily = dom_daily
        daily_src = ("domain budget.daily_cap (%s)" % dom_src if daily is not None
                     else "none (no daily cap is declared)")
    elif daily < 0:
        reasons.append("campaign daily_cap_su %r is negative; treated as 0" % daily)
        daily = 0.0
    if daily is not None and dom_daily is not None and dom_daily < daily:
        reasons.append("daily cap capped at the domain daily_cap %.6g (%s)" % (dom_daily, dom_src))
        daily = dom_daily
    return {"envelope_su": env, "requested_envelope_su": want, "domain_envelope_su": dom_env,
            "daily_cap_su": daily, "requested_daily_cap_su": want_daily,
            "domain_daily_cap_su": dom_daily,
            "sources": {"envelope_su": src, "domain_envelope_su": dom_src,
                        "daily_cap_su": daily_src},
            "reasons": reasons}


def _exp_of_job(job):
    parts = str(job or "").split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "inc" else None


_SACCT_JOB_RE = re.compile(r"^inc:job([0-9]+(?:_[0-9]+)?):sacct$")


def spent(name, domain=M.DOMAIN, base_dir=None):
    """SU the ledger holds for campaign `name`, the experiments it settles, and
    the jobs settled from sacct (stream mode: record_job_spend)."""
    step = campaign_step(name)
    # su_ledger folds each (job, step) to its last line; its public readers do
    # not filter by step, so the fold and its aggregate are reused directly.
    entries = [e for e in su_ledger._read_deduped(domain, base_dir) if str(e.get("step")) == step]
    agg = su_ledger._aggregate(entries)
    settled = sorted({x for x in (_exp_of_job(e.get("job")) for e in entries) if x})
    jobs = sorted({m.group(1) for m in (_SACCT_JOB_RE.match(str(e.get("job") or "")) for e in entries) if m})
    return {"su": agg["su"], "n_entries": agg["n_entries"], "n_unknown": agg["n_unknown"],
            "unknown_su_jobs": agg["unknown_su_jobs"], "settled_exps": settled, "settled_jobs": jobs}


def fold(executions):
    """One record per executor run, in the order the runs started.

    Records sharing a `run_id` are one run: its `started` record and its
    outcome; the last one wins. A run with no outcome keeps its `started`
    record, which is charged. Records without a `run_id` stand alone.
    Folding a folded list changes nothing.
    """
    out, where = [], {}
    for rec in executions or ():
        if not isinstance(rec, dict):
            continue
        rid = rec.get("run_id")
        if rid:
            if rid in where:
                out[where[rid]] = rec
                continue
            where[rid] = len(out)
        out.append(rec)
    return out


def _charged(executions, name):
    for rec in fold(executions):
        if rec.get("campaign") != name or not rec.get("charged"):
            continue
        est = _num(rec.get("est_su"))
        if est is None or est <= 0:
            continue
        yield rec, est


def child_of(rec):
    """The experiment whose recorded spend releases this record's estimate.

    A build's is the experiment it builds (`params.exp`), never a `child_exp`
    the caller stated: naming a settled experiment there would otherwise
    release the estimate before anything was spent.
    """
    if str(rec.get("action") or "").startswith("inc_build_"):
        exp = (rec.get("params") or {}).get("exp")
        if not exp and rec.get("action") in STREAM_CHILD_ACTIONS:
            # a stream build names no --exp; the executor checked its child_exp
            # is one of its stream's experiments (executor._resolve)
            exp = rec.get("child_exp")
        return str(exp) if exp else None
    return rec.get("child_exp")


# Stream builds (docs/CONTINUOUS_LOOP.md 6.3) whose experiment is the request's
# checked child_exp, and the stream jobs settled from sacct (record_job_spend).
STREAM_CHILD_ACTIONS = ("inc_build_segment", "inc_build_consolidation")


def committed(executions, name, settled_exps=(), settled_jobs=()):
    """Estimates charged by the executor whose experiment has no recorded spend
    (and, in stream mode, whose jobs sacct has not settled)."""
    settled = set(settled_exps or ())
    jobs = set(str(j) for j in settled_jobs or ())
    su, items = 0.0, []
    for rec, est in _charged(executions, name):
        child = child_of(rec)
        if child and child in settled:
            continue
        ids = [str(j) for j in rec.get("job_ids") or []]
        # a job settled from sacct releases its estimate, unless it is a build
        # whose experiment's report spend releases it (a build job with no
        # experiment, such as a stream fork, is released by its sacct)
        if jobs and ids and (not str(rec.get("action") or "").startswith("inc_build_") or not child) \
                and all(j in jobs for j in ids):
            continue
        su += est
        items.append({"action": rec.get("action"), "child_exp": child, "est_su": est,
                      "approval_id": rec.get("approval_id"), "ts": rec.get("ts")})
    return {"su": round(su, 6), "items": items}


def _day_start(now):
    d = datetime.datetime.fromtimestamp(float(now), datetime.timezone.utc)
    return datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc).timestamp()


def today(executions, name, now):
    """Estimates charged since 00:00 UTC of `now`'s day."""
    start = _day_start(now)
    su = 0.0
    for rec, est in _charged(executions, name):
        t = _num(rec.get("epoch"))
        if t is not None and start <= t < start + 86400.0:
            su += est
    return round(su, 6)


def state(campaign, executions=(), given_budget=None, now=None, domain=M.DOMAIN, base_dir=None):
    """Everything the executor needs to charge one more action to the campaign."""
    c = campaign if isinstance(campaign, dict) else {}
    name = c.get("name")
    env = envelope(c, given_budget)
    sp = spent(name, domain, base_dir)
    cm = committed(executions, name, sp["settled_exps"], sp.get("settled_jobs"))
    td = today(executions, name, now if now is not None else datetime.datetime.now(
        datetime.timezone.utc).timestamp())
    remaining = None
    if env["envelope_su"] is not None:
        remaining = round(env["envelope_su"] - sp["su"] - cm["su"], 6)
    daily_remaining = None
    if env["daily_cap_su"] is not None:
        daily_remaining = round(env["daily_cap_su"] - td, 6)
    known = [v for v in (remaining, daily_remaining) if v is not None]
    out = {"campaign": name, "envelope_su": env["envelope_su"],
           "domain_envelope_su": env["domain_envelope_su"],
           "daily_cap_su": env["daily_cap_su"], "sources": env["sources"],
           "spent_su": sp["su"], "committed_su": cm["su"], "committed": cm["items"],
           "today_su": td, "remaining_su": remaining, "daily_remaining_su": daily_remaining,
           "su_remaining": min(known) if known else None,
           "unknown_su_jobs": sp["unknown_su_jobs"], "settled_exps": sp["settled_exps"],
           "reasons": env["reasons"]}
    if c.get("mode") == "stream":
        out.update(stream_windows(c, executions, env, sp, cm, now, domain, base_dir))
        known = [v for v in (out["remaining_su"], out["daily_remaining_su"], out["window_remaining_su"],
                             out["domain_remaining_su"]) if v is not None]
        out["su_remaining"] = min(known) if known else None
    return out


# --- stream mode (docs/CONTINUOUS_LOOP.md 6.6, L-2) --------------------------------------
def _month_start(now):
    d = datetime.datetime.fromtimestamp(float(now), datetime.timezone.utc)
    return datetime.datetime(d.year, d.month, 1, tzinfo=datetime.timezone.utc).timestamp()


def _ts_epoch(ts):
    try:
        return datetime.datetime.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def stream_windows(campaign, executions, env, sp, cm, now, domain=M.DOMAIN, base_dir=None):
    """The stream campaign's monthly window and the domain's cross-campaign cap.

    Window: the SU the ledger holds for this campaign with a timestamp in the
    current calendar month (UTC), plus the estimates charged this month that
    are still committed, against the campaign's `window_cap_su` when it
    declares one (L-2's 350 was its default until the 2026-10-04 amendment;
    an absent window is none and refuses nothing). Domain: every
    campaign's `inc:*` ledger steps (the funnel's and weed_inc_v1's included)
    plus every campaign's committed estimates in the domain's execution log,
    against the domain envelope (db.py su_envelope 1500, or the live domain's
    budget block the ticker passes). Unknown is never 0: a window with no cap
    reports None and fits() refuses nothing on it."""
    now = now if now is not None else datetime.datetime.now(datetime.timezone.utc).timestamp()
    c = campaign if isinstance(campaign, dict) else {}
    start = _month_start(now)
    step = campaign_step(c.get("name"))
    entries = su_ledger._read_deduped(domain, base_dir)
    mine = [e for e in entries if str(e.get("step")) == step
            and (_ts_epoch(e.get("ts")) or 0.0) >= start]
    window_spent = su_ledger._aggregate(mine)["su"]
    window_committed = 0.0
    for it in cm.get("items") or []:
        t = _ts_epoch(it.get("ts"))
        if t is not None and t >= start:
            window_committed += float(it.get("est_su") or 0.0)
    cap = _num(c.get("window_cap_su"))
    win_used = round(window_spent + window_committed, 6)
    inc_entries = [e for e in entries if str(e.get("step") or "").startswith(STEP_PREFIX)]
    dom_spent = su_ledger._aggregate(inc_entries)["su"]
    names = sorted({r.get("campaign") for r in fold(executions) if r.get("campaign")})
    dom_committed = 0.0
    for n in names:
        try:
            s_n = spent(n, domain, base_dir)
        except ValueError:
            continue
        dom_committed += committed(executions, n, s_n["settled_exps"], s_n.get("settled_jobs"))["su"]
    dom_cap = env.get("domain_envelope_su")
    dom_used = round(dom_spent + dom_committed, 6)
    return {"window": "month", "window_start_utc": datetime.datetime.fromtimestamp(
                start, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_cap_su": cap, "window_su": win_used,
            "window_remaining_su": None if cap is None else round(cap - win_used, 6),
            "domain_inc_su": dom_used, "domain_committed_su": round(dom_committed, 6), "domain_cap_su": dom_cap,
            "domain_remaining_su": None if dom_cap is None else round(dom_cap - dom_used, 6)}


def budget_state(st):
    """The dict `policy.authorize` reads (`su_remaining`), or None with no campaign."""
    if not isinstance(st, dict):
        return None
    return {"su_remaining": st.get("su_remaining"), "campaign": st.get("campaign"),
            "remaining_su": st.get("remaining_su"),
            "daily_remaining_su": st.get("daily_remaining_su")}


def fits(st, est_su, need_daily=False):
    """(ok, [reasons]): may `est_su` more be charged to this campaign now?

    The envelope is always checked (unknown refuses). A daily cap and a
    monthly window are checked when declared; an absent one is no cap.
    `need_daily` (the envelope rule's caller) once made an undeclared daily
    cap or window a refusal; since the 2026-10-04 amendment (no time-based
    throttle by default) it changes nothing, and it is still accepted so a
    caller written before the amendment keeps working.
    """
    del need_daily
    reasons = []
    est = _num(est_su)
    if est is None:
        return False, ["the SU estimate is unknown, so it cannot be checked against the envelope"]
    if not isinstance(st, dict):
        return False, ["no campaign budget state to charge against"]
    rem = st.get("remaining_su")
    if rem is None:
        reasons.append("the campaign envelope is unknown")
    elif est > rem:
        reasons.append("estimated %.4g SU exceeds the %.4g SU left in the campaign envelope "
                       "(%.4g spent, %.4g committed of %.4g)"
                       % (est, rem, st.get("spent_su") or 0.0, st.get("committed_su") or 0.0,
                          st.get("envelope_su") or 0.0))
    daily = st.get("daily_remaining_su")
    if daily is not None and est > daily:
        reasons.append("estimated %.4g SU exceeds the %.4g SU left under today's cap of %.4g"
                       % (est, daily, st.get("daily_cap_su") or 0.0))
    # stream mode only (the keys exist only for a stream campaign): the monthly
    # window and the domain's cross-campaign cap
    if "window_remaining_su" in st:
        win = st.get("window_remaining_su")
        if win is not None and est > win:
            reasons.append("estimated %.4g SU exceeds the %.4g SU left in this month's window of %.4g"
                           % (est, win, st.get("window_cap_su") or 0.0))
    if "domain_remaining_su" in st:
        dom = st.get("domain_remaining_su")
        if dom is not None and est > dom:
            reasons.append("estimated %.4g SU exceeds the %.4g SU left of the domain's %.4g across every campaign"
                           % (est, dom, st.get("domain_cap_su") or 0.0))
    return (not reasons), reasons


def record_report_spend(report, campaign_name, actor=M.AUTOPILOT_ACTOR, domain=M.DOMAIN,
                        base_dir=None, gpu_type=GPU_TYPE):
    """Write one su_ledger entry per unit of a finished experiment's report.

    V100 at the su_rates.json rate (1.0 SU per GPU-hour). The hours are the sum
    of run.json seconds the report computed, not sacct; each entry says so. A
    second call for the same report updates the same (job, step) keys and is
    never billed twice.
    """
    rep = report if isinstance(report, dict) else {}
    got = {k: rep.get(k) for k in REPORT_KEYS}
    step = campaign_step(campaign_name)
    exp = got["exp"]
    if not _NAME_RE.match(str(exp or "")):
        return {"ok": False, "reason": "the report names no valid experiment (%r)" % (exp,)}
    if got["done"] is not True:
        return {"ok": False, "reason": "experiment %s is not done; its spend is recorded only "
                                       "once, from the final report" % exp}
    hours = got["gpu_hours"] if isinstance(got["gpu_hours"], dict) else {}
    if not hours:
        return {"ok": False, "reason": "report of %s carries no gpu_hours" % exp}
    recorded, total, updated, gaps = [], 0.0, 0, []
    for unit in sorted(hours):
        row = hours[unit] if isinstance(hours[unit], dict) else {}
        h = _num(row.get("hours"))
        runs = row.get("runs")
        if h is None or h < 0:
            gaps.append(unit)
            elapsed = None
        else:
            elapsed = h * 3600.0
        su = su_ledger.su_for(gpu_type, 1, elapsed, estimated=False)
        su["source"] = "report.gpu_hours"
        su["reason"] = ("report.json gpu_hours[%s]: %s runs, summed from run.json seconds, not "
                        "sacct; %s" % (unit, runs, su.get("reason") or ""))
        r = su_ledger.record({"domain": domain, "job": "inc:%s:%s" % (exp, unit), "step": step,
                              "actor": actor, "gpu_count": 1, "gpu_type": gpu_type,
                              "elapsed_s": elapsed, "su": su, "ts": got["done_utc"]},
                             base_dir=base_dir)
        recorded.append(r["key"])
        updated += 1 if r.get("updated") else 0
        if su.get("value") is not None:
            total += su["value"]
    out = {"ok": True, "exp": exp, "step": step, "recorded": recorded, "su": round(total, 6),
           "updated": updated}
    if gaps:
        out["unknown_units"] = gaps
    return out


BUILD_UNIT = "build"


def record_build_spend(exp, campaign_name, hours, source, ts=None, measured=False,
                       actor=M.AUTOPILOT_ACTOR, domain=M.DOMAIN, base_dir=None, gpu_type=GPU_TYPE):
    """Write the su_ledger entry of the job that built experiment `exp`
    (run_inc_build.sh: one V100 on GPU-shared), job `inc:<exp>:build`.

    A build's estimate is the experiment's runs plus this job
    (levers.with_build_job); the report's gpu_hours hold only the runs, so
    without this entry the job's SU would be released from `committed` with
    the rest of the estimate and never reach `spent`. `hours` is the job's
    elapsed time from its provenance record when `measured`, else the job's
    walltime limit, an upper bound; `source` says which. A second call for the
    same experiment updates the same (job, step) key."""
    if not _NAME_RE.match(str(exp or "")):
        return {"ok": False, "reason": "no valid experiment name (%r)" % (exp,)}
    h = _num(hours)
    if h is None or h < 0:
        return {"ok": False, "reason": "the build job's hours %r are not a number >= 0" % (hours,)}
    step = campaign_step(campaign_name)
    su = su_ledger.su_for(gpu_type, 1, h * 3600.0, estimated=not measured)
    su["source"] = "build job: %s" % source
    su["reason"] = ("the job that built %s (run_inc_build.sh, one V100 on GPU-shared): %.4g h, %s; %s"
                    % (exp, h, "measured from its provenance record, not sacct" if measured
                       else "its walltime limit, an upper bound", su.get("reason") or ""))
    try:
        r = su_ledger.record({"domain": domain, "job": "inc:%s:%s" % (exp, BUILD_UNIT), "step": step,
                              "actor": actor, "gpu_count": 1, "gpu_type": gpu_type,
                              "elapsed_s": h * 3600.0, "estimated": not measured, "su": su, "ts": ts},
                             base_dir=base_dir)
    except (OSError, ValueError) as e:
        return {"ok": False, "reason": "the SU ledger could not be written: %s" % e}
    return {"ok": True, "exp": exp, "step": step, "key": r.get("key"), "su": su.get("value"),
            "hours": h, "measured": bool(measured), "updated": bool(r.get("updated"))}


def record_job_spend(job_id, campaign_name, sacct_job, action=None, actor=M.AUTOPILOT_ACTOR, domain=M.DOMAIN,
                     base_dir=None, ts=None):
    """Settle a stream job from sacct (docs/CONTINUOUS_LOOP.md 6.6: 'Each L16,
    L17, L18 and L20 job's estimate is settled from sacct'): one su_ledger entry
    job `inc:job<ID>:sacct` under the campaign's step, priced by su_ledger.su_for
    from the sacct row's GPU family, count and elapsed time. `committed` then
    releases the estimate of the execution whose job ids are all settled. A
    build's estimate is released by its experiment's report spend instead (its
    runs are later jobs), so builds are not settled here."""
    if not re.match(r"^[0-9]+(_[0-9]+)?$", str(job_id or "")):
        return {"ok": False, "reason": "no job id (%r)" % (job_id,)}
    row = sacct_job if isinstance(sacct_job, dict) else {}
    el = _num(row.get("elapsed_s"))
    gpu_type = row.get("gpu_type") or GPU_TYPE
    gpu_count = int(row.get("gpu_count") or 1)
    su = su_ledger.su_for(gpu_type, gpu_count, el, estimated=False)
    su["source"] = "sacct"
    su["reason"] = ("job %s (%s): sacct %s, %s s on %d x %s; %s"
                    % (job_id, action or "stream job", row.get("state"), el, gpu_count, gpu_type, su.get("reason") or ""))
    try:
        r = su_ledger.record({"domain": domain, "job": "inc:job%s:sacct" % job_id, "step": campaign_step(campaign_name),
                              "actor": actor, "gpu_count": gpu_count, "gpu_type": gpu_type, "elapsed_s": el,
                              "su": su, "ts": ts}, base_dir=base_dir)
    except (OSError, ValueError) as e:
        return {"ok": False, "reason": "the SU ledger could not be written: %s" % e}
    return {"ok": True, "job": job_id, "su": su.get("value"), "key": r.get("key")}


def partition_rates(path=None):
    """{partition: {"billing", ...}} of su_rates.json's `partitions` block
    (docs/CONTINUOUS_LOOP.md 6.6: an explicit rate for every partition the
    loop uses, so no job is priced 'unknown'); {} when there is none."""
    import json as _json
    from pathlib import Path as _P
    p = _P(path) if path else _P(su_ledger.__file__).with_name("su_rates.json")
    try:
        raw = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in ((raw or {}).get("partitions") or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
    return out
