"""Read-only HTTP surface over the supervision layer (v3.31.0).

Why this exists
---------------
Every part of the layer -- what the checks found, what was corrected and by
whom, what a tier escalated, what an experiment proposed, what it cost -- has to
be readable from the platform itself, by a person with a browser and no agent
session anywhere in the picture. A supervision layer whose state can only be
inspected by someone running commands in a terminal is not a platform feature;
it is a private tool with a web page next to it.

These routes are deliberately read-only. Writes go through the paths that
already own them: corrections through the single-writer channel on the scheduler
thread, actions through the policy gate in `api_cluster_action`. A second write
path here would be a second writer, which is precisely what the correction
channel refuses.

Honesty rule
------------
An empty answer and a missing store are different facts and are reported
differently. `{"available": false, "reason": ...}` means the layer cannot see
that state at all; `{"available": true, "rows": []}` means it looked and there
was nothing. Collapsing the two would let a broken mount read on the page as a
quiet campaign, which is the same failure -- a silence mistaken for health --
that the whole layer was built after.

Mounted by `dashboard_server` alongside the scheduler alarm; the auth middleware
covers these paths because they are not in `_AUTH_EXEMPT_PATHS`.
"""

import json
import os
import time

from fastapi import APIRouter

router = APIRouter()
_CTX = {}

MAX_ROWS = 200


def _log():
    lg = _CTX.get("log")
    return lg if lg is not None else _NullLog()


class _NullLog(object):
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _repo():
    return str(_CTX.get("repo") or os.getcwd())


def _domain(raw):
    """A domain id reduced to the characters a domain id may contain.

    These ids reach filesystem paths below, so the sanitising happens here once
    rather than at each use.
    """
    out = "".join(c for c in str(raw or "").strip().lower()
                  if c.isalnum() or c in "_-")[:40]
    return out


def _unavailable(reason, **extra):
    out = {"available": False, "reason": reason, "rows": []}
    out.update(extra)
    return out


def _brain_dir(domain):
    return os.path.join(_repo(), "results", "framework", "_brain", domain)


# --- corrections -------------------------------------------------------------

@router.get("/api/brain/{domain}/corrections")
def api_brain_corrections(domain: str):
    """The correction chain, with the result of recomputing it.

    `chain_ok` is reported next to the rows rather than gating them: a reader
    looking at a tampered chain needs to see both what it says and that it
    cannot be trusted, and returning nothing would hide the evidence of the
    tampering along with the tampering.
    """
    dom = _domain(domain)
    try:
        from . import corrections as C
    except Exception as e:
        return _unavailable("correction channel unavailable: %s" % e)
    try:
        rep = C.verify(dom, root=_repo())
    except Exception as e:
        return _unavailable("chain verify failed: %s" % e)
    rows = []
    try:
        rows = C.effective(dom, 10 ** 9, root=_repo()) or []
    except Exception as e:
        _log().warning("[brain-api] effective() failed for %s: %s" % (dom, e))
    return {"available": True, "domain": dom,
            "chain_ok": bool(rep.get("ok")),
            "chain_length": rep.get("length"),
            "first_bad_seq": rep.get("first_bad_seq"),
            "chain_reason": rep.get("reason"),
            "rows": rows[-MAX_ROWS:], "truncated": len(rows) > MAX_ROWS}


# --- spend -------------------------------------------------------------------

@router.get("/api/brain/{domain}/su")
def api_brain_su(domain: str):
    """Service-unit spend for a domain, with the envelope it is drawn against."""
    dom = _domain(domain)
    try:
        from . import su_ledger as L
    except Exception as e:
        return _unavailable("SU ledger unavailable: %s" % e)
    # `total` on a domain with no ledger returns zero spend, which is a true
    # statement about an empty file and a false one about a missing file. Saying
    # "0 SU" when nothing was ever recorded is the same conflation this module's
    # honesty rule exists to prevent, so the file is checked first.
    ledger = os.path.join(_brain_dir(dom), "su_ledger.jsonl")
    if not os.path.exists(ledger):
        return _unavailable("no SU ledger has been written for this domain yet",
                            path=ledger)
    out = {"available": True, "domain": dom, "rows": [], "ledger_path": ledger}
    try:
        out["total"] = L.total(dom)
        out["by_actor"] = L.by_actor(dom)
        out["by_step"] = L.by_step(dom)
    except Exception as e:
        return _unavailable("SU ledger read failed: %s" % e)
    try:
        cfg = _CTX["db"].get_domain_config(dom)
        budget = cfg.get("budget") or {}
        out["budget"] = budget
        out["remaining"] = L.remaining(dom, budget)
    except Exception as e:
        # The spend is a fact even when the envelope cannot be read; saying so
        # beats withholding the number that was measured.
        out["budget_error"] = str(e)
    return out


# --- what the checks say now -------------------------------------------------

@router.get("/api/brain/{domain}/signals")
def api_brain_signals(domain: str):
    """The deterministic checks over the most recent evidence bundle on disk.

    A bundle is written by a review; when none exists yet this reports that,
    rather than an empty list of findings that would read as "nothing wrong".
    """
    dom = _domain(domain)
    path = os.path.join(_brain_dir(dom), "latest_bundle.json")
    if not os.path.exists(path):
        return _unavailable("no evidence bundle has been built for this domain yet",
                            path=path)
    try:
        with open(path, "rb") as fh:
            bundle = json.loads(fh.read().decode("utf-8"))
        from . import signals as S
        rows = S.detect_all(bundle)
    except Exception as e:
        return _unavailable("signals could not be read: %s" % e)
    fired = [r for r in rows if r.get("severity") in ("warn", "crit")]
    unknown = [r for r in rows if r.get("severity") == "unknown"]
    return {"available": True, "domain": dom, "rows": rows,
            "n_fired": len(fired), "n_unknown": len(unknown),
            "bundle_sha256": bundle.get("sha256"),
            "built_ts": bundle.get("built_ts")}


# --- plans and experiments ---------------------------------------------------

@router.get("/api/brain/{domain}/plans")
def api_brain_plans(domain: str):
    """Every plan version, newest first, each carrying whether it was simulated."""
    dom = _domain(domain)
    d = os.path.join(_brain_dir(dom), "plans")
    if not os.path.isdir(d):
        return _unavailable("no plans have been written for this domain yet", path=d)
    rows = []
    try:
        for name in sorted(os.listdir(d), reverse=True)[:MAX_ROWS]:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, name), "rb") as fh:
                    plan = json.loads(fh.read().decode("utf-8"))
            except Exception as e:
                rows.append({"file": name, "error": str(e)})
                continue
            rows.append({"file": name, "version": plan.get("version"),
                         "backend": plan.get("backend"),
                         "simulated": bool(plan.get("simulated")),
                         "refused": bool(plan.get("refused")),
                         "reason": plan.get("reason"),
                         "n_experiments": len(plan.get("ordered_experiments") or []),
                         "hypotheses": plan.get("hypotheses") or []})
    except Exception as e:
        return _unavailable("plans could not be listed: %s" % e)
    return {"available": True, "domain": dom, "rows": rows}


@router.get("/api/brain/{domain}/experiments")
def api_brain_experiments(domain: str):
    """Proposed experiments and, where one has run, its result and verdict."""
    dom = _domain(domain)
    path = os.path.join(_brain_dir(dom), "experiments.jsonl")
    if not os.path.exists(path):
        return _unavailable("no experiments have been proposed for this domain yet",
                            path=path)
    rows = []
    try:
        with open(path, "rb") as fh:
            for line in fh.read().decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # A torn last line is what a walltime-killed writer leaves;
                    # it is not a reason to report the whole file as missing.
                    continue
    except Exception as e:
        return _unavailable("experiments could not be read: %s" % e)
    return {"available": True, "domain": dom, "rows": rows[-MAX_ROWS:],
            "truncated": len(rows) > MAX_ROWS}


# --- who is wired to which tier ----------------------------------------------

@router.get("/api/brain/{domain}/roles")
def api_brain_roles(domain: str):
    """Which model is wired to each tier, and the policy in force.

    An empty tier id means that tier is not wired. It is reported as `null`
    rather than filled in from a default, because a missing deployment must
    never silently promote another model into a review role.
    """
    dom = _domain(domain)
    try:
        cfg = _CTX["db"].get_domain_config(dom)
    except Exception as e:
        return _unavailable("domain config unavailable: %s" % e)
    brain = cfg.get("brain") or {}
    tiers = {k: (v or None) for k, v in (brain.get("tiers") or {}).items()}
    return {"available": True, "domain": dom,
            "policy": brain.get("policy"), "tiers": tiers,
            "review_timeout_min": brain.get("review_timeout_min"),
            "periodic_audit_every": brain.get("periodic_audit_every"),
            "budget": cfg.get("budget") or {},
            "noise_floor": cfg.get("noise_floor") or {},
            "lever_menu": [{"id": l.get("id"), "question": l.get("question"),
                            "control": l.get("control"), "risk": l.get("risk"),
                            "est_su": l.get("est_su")}
                           for l in (cfg.get("lever_menu") or [])]}


# --- the actions a tier may take ---------------------------------------------

@router.get("/api/brain/{domain}/policy")
def api_brain_policy(domain: str):
    """The action catalogue as the gate sees it, per risk tier.

    This is the page a person reads to answer "what is this thing allowed to do
    to my cluster account", which is the question the layer has to be able to
    answer about itself.
    """
    _ = _domain(domain)
    try:
        from . import policy as P
        table = P._table()
    except Exception as e:
        return _unavailable("policy catalogue unavailable: %s" % e)
    rows = []
    for action, row in sorted((table.get("actions") or {}).items()):
        rows.append({"action": action, "risk": row.get("risk"),
                     "reversible": row.get("reversible"),
                     "allowed_tiers": row.get("allowed_tiers"),
                     "description": row.get("description"),
                     "param_bounds": sorted((row.get("param_bounds") or {}).keys())})
    return {"available": True, "rows": rows, "n_actions": len(rows),
            "errors": list(table.get("errors") or [])}


# --- one merged, time-ordered view -------------------------------------------

@router.get("/api/brain/{domain}/timeline")
def api_brain_timeline(domain: str):
    """Corrections, plans and experiments in one time-ordered list.

    Each entry keeps the name of the source it came from, so a reader can always
    get back to the underlying record rather than trusting a merged summary.
    """
    dom = _domain(domain)
    events, missing = [], []
    for name, fn, ts_key in (("corrections", api_brain_corrections, "ts"),
                             ("experiments", api_brain_experiments, "ts")):
        try:
            block = fn(dom)
        except Exception as e:
            missing.append("%s: %s" % (name, e))
            continue
        if not block.get("available"):
            missing.append("%s: %s" % (name, block.get("reason")))
            continue
        for row in block.get("rows") or []:
            if isinstance(row, dict):
                events.append({"source": name, "ts": row.get(ts_key), "row": row})
    events.sort(key=lambda e: (e["ts"] is None, e["ts"] or 0))
    return {"available": True, "domain": dom, "events": events[-MAX_ROWS:],
            "sources_unavailable": missing, "checked_ts": time.time()}


def mount(app, ctx: dict):
    _CTX.update(ctx)
    app.include_router(router)
