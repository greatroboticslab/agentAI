"""SU envelope of one INC campaign, and the SU-ledger writer for finished experiments.

docs/INC_AUTOPILOT.md, section d: the executor passes a real `budget_state` to
`policy.authorize`, so the budget escalation that never ran in production
finally does, and the campaign-envelope rule has a balance to check against.

The envelope
------------
A campaign draws on a sub-envelope of the domain's `budget.su_envelope`
(db.py DEFAULT_DOMAIN_CONFIG: 1500 SU). The sub-envelope is the campaign's
`envelope_su`, 300 SU when the campaign does not set one, and never more than
the domain envelope. A daily cap applies as well: the campaign's `daily_cap_su`,
or the domain's `budget.daily_cap` (120 SU) when the campaign does not set one,
and never more than the domain's daily cap.

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
* **Today**: estimates charged since 00:00 UTC, against the daily cap.

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
    """The campaign's sub-envelope and daily cap, with where each came from."""
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
        daily_src = "domain budget.daily_cap (%s)" % dom_src if daily is not None else "none"
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


def spent(name, domain=M.DOMAIN, base_dir=None):
    """SU the ledger holds for campaign `name`, and the experiments it settles."""
    step = campaign_step(name)
    # su_ledger folds each (job, step) to its last line; its public readers do
    # not filter by step, so the fold and its aggregate are reused directly.
    entries = [e for e in su_ledger._read_deduped(domain, base_dir) if str(e.get("step")) == step]
    agg = su_ledger._aggregate(entries)
    settled = sorted({x for x in (_exp_of_job(e.get("job")) for e in entries) if x})
    return {"su": agg["su"], "n_entries": agg["n_entries"], "n_unknown": agg["n_unknown"],
            "unknown_su_jobs": agg["unknown_su_jobs"], "settled_exps": settled}


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
        return str(exp) if exp else None
    return rec.get("child_exp")


def committed(executions, name, settled_exps=()):
    """Estimates charged by the executor whose experiment has no recorded spend."""
    settled = set(settled_exps or ())
    su, items = 0.0, []
    for rec, est in _charged(executions, name):
        child = child_of(rec)
        if child and child in settled:
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
    cm = committed(executions, name, sp["settled_exps"])
    td = today(executions, name, now if now is not None else datetime.datetime.now(
        datetime.timezone.utc).timestamp())
    remaining = None
    if env["envelope_su"] is not None:
        remaining = round(env["envelope_su"] - sp["su"] - cm["su"], 6)
    daily_remaining = None
    if env["daily_cap_su"] is not None:
        daily_remaining = round(env["daily_cap_su"] - td, 6)
    known = [v for v in (remaining, daily_remaining) if v is not None]
    return {"campaign": name, "envelope_su": env["envelope_su"],
            "domain_envelope_su": env["domain_envelope_su"],
            "daily_cap_su": env["daily_cap_su"], "sources": env["sources"],
            "spent_su": sp["su"], "committed_su": cm["su"], "committed": cm["items"],
            "today_su": td, "remaining_su": remaining, "daily_remaining_su": daily_remaining,
            "su_remaining": min(known) if known else None,
            "unknown_su_jobs": sp["unknown_su_jobs"], "settled_exps": sp["settled_exps"],
            "reasons": env["reasons"]}


def budget_state(st):
    """The dict `policy.authorize` reads (`su_remaining`), or None with no campaign."""
    if not isinstance(st, dict):
        return None
    return {"su_remaining": st.get("su_remaining"), "campaign": st.get("campaign"),
            "remaining_su": st.get("remaining_su"),
            "daily_remaining_su": st.get("daily_remaining_su")}


def fits(st, est_su, need_daily=False):
    """(ok, [reasons]): may `est_su` more be charged to this campaign now?

    `need_daily` makes an undeclared daily cap a refusal (the envelope rule);
    otherwise a cap that is declared still applies and an absent one does not.
    """
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
    if daily is None:
        if need_daily:
            reasons.append("no daily cap is declared for this campaign")
    elif est > daily:
        reasons.append("estimated %.4g SU exceeds the %.4g SU left under today's cap of %.4g"
                       % (est, daily, st.get("daily_cap_su") or 0.0))
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
