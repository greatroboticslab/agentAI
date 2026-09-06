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

from fastapi import APIRouter, Request

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


# --- approvals: the one place a person rules on a queued request -------------
#
# This is the exception to the read-only rule above, and it is the exception
# because it is the opposite of a second writer: an approval is a PERSON
# deciding, and there is nowhere else on the platform for them to do it. The
# single-writer rule protects the correction chain from a second automated
# author; it was never about keeping people out of their own governance queue.

@router.get("/api/brain/{domain}/approvals")
def api_brain_approvals(domain: str):
    """Everything that has been queued, with what a person decided."""
    dom = _domain(domain)
    try:
        from . import approvals as AP
    except Exception as e:
        return _unavailable("approval queue unavailable: %s" % e)
    log_path = AP.path(dom, root=_repo())
    if not os.path.exists(log_path):
        return _unavailable("nothing has been queued for approval in this domain yet",
                            path=log_path)
    try:
        items = AP.state(dom, root=_repo())
    except Exception as e:
        return _unavailable("approval queue could not be read: %s" % e)
    rows = sorted(items.values(), key=lambda r: (r.get("ts") or 0))
    return {"available": True, "domain": dom, "rows": rows[-MAX_ROWS:],
            "n_pending": sum(1 for r in rows if r.get("status") == "pending")}


@router.post("/api/brain/{domain}/approvals/{item_id}")
def api_brain_approvals_decide(domain: str, item_id: str, payload: dict = None,
                               request: Request = None):
    """Approve or deny one queued request.

    The actor comes from the signed session, never from the request body: a
    decision whose author the body could name is a decision anyone can attribute
    to anyone. If this module cannot identify the person -- no identity hook
    wired, or the caller may not manage this agent -- it refuses. Failing open
    here would let an unauthenticated call approve an R3 action.
    """
    dom = _domain(domain)
    body = payload if isinstance(payload, dict) else {}
    who = _CTX.get("actor_of")
    can = _CTX.get("can_manage")
    if not callable(who):
        return {"ok": False, "reason": "no identity hook is wired; refusing to "
                                       "record a decision with no author"}
    try:
        actor = str(who(request) or "")
    except Exception as e:
        return {"ok": False, "reason": "could not identify the caller: %s" % e}
    if not actor:
        return {"ok": False, "reason": "could not identify the caller"}
    if callable(can):
        try:
            allowed = bool(can(actor, dom))
        except Exception as e:
            return {"ok": False, "reason": "permission check failed: %s" % e}
        if not allowed:
            return {"ok": False, "reason": "you do not manage this agent"}
    try:
        from . import approvals as AP
    except Exception as e:
        return {"ok": False, "reason": "approval queue unavailable: %s" % e}
    res = AP.decide(dom, str(item_id), str(body.get("decision") or ""),
                    "human:" + actor, str(body.get("reason") or ""),
                    time.time(), root=_repo())
    try:
        log = _CTX.get("log_action")
        if callable(log):
            log("brain_approval", {"ok": bool(res.get("ok")), "domain": dom,
                                   "item": item_id, "actor": actor,
                                   "decision": body.get("decision"),
                                   "msg": res.get("reason") or ""})
    except Exception:
        pass
    return res


# --- the page ----------------------------------------------------------------

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supervision &middot; __DOMAIN__</title>
<style>
 body{margin:0;padding:1rem;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 h1{margin:.2rem 0 .1rem}
 .sub{opacity:.7;font-size:.85rem;margin-bottom:1rem}
 .card{border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:.9rem 1rem;
       margin:0 0 1rem;background:rgba(255,255,255,.03)}
 .card h3{margin:0 0 .5rem;font-size:1rem}
 /* align-items:start stops one long card from stretching every other card in
    its row to the same height: the action catalogue is 35 rows and left the
    rest of the page as tall empty boxes. */
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;
       align-items:start}
 table{width:100%;border-collapse:collapse;font-size:.85rem}
 td,th{text-align:left;padding:.28rem .5rem;border-bottom:1px solid rgba(255,255,255,.08);
       vertical-align:top}
 th{opacity:.65;font-weight:600}
 code{font-size:.8rem;word-break:break-all}
 .pill{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.72rem;
       border:1px solid currentColor}
 .crit{color:#ff6b6b}.warn{color:#ffb454}.info{color:#7fb3ff}
 .unknown{opacity:.55}.ok{color:#5fd08a}
 .missing{opacity:.6;font-style:italic}
 /* A long table scrolls inside its own card rather than growing the page.
    Both axes: wide rows scroll sideways, long ones scroll down. */
 .scroll{overflow-x:auto;max-height:22rem;overflow-y:auto}
 /* Deliberately NOT white-space:nowrap. These cells carry prose -- a lever's
    control, a check's reason -- and forcing them onto one line pushed the
    later columns out of the card entirely, which reads as a broken table
    rather than as a scrollable one. */
 .scroll th{position:sticky;top:0;background:#0e1626}
 /* Identifiers here are long and have no spaces -- action ids, actor ids,
    artifact names -- so without this they refuse to wrap and push the later
    columns out of the card. The governance columns (who asked, who decided)
    are the ones that were disappearing. */
 td{overflow-wrap:anywhere}
</style></head><body>
<h1>Supervision &mdash; __DOMAIN__</h1>
<div class="sub">Everything this layer knows about the campaign, read from the
platform. Nothing on this page can change anything: corrections are written only
by the scheduler's single-writer channel, and actions only through the policy
gate.</div>
<div class="grid" id="grid"></div>
<script>
const DOMAIN = "__DOMAIN__";
const SECTIONS = [
  ["signals",     "Deterministic checks"],
  ["corrections", "Corrections applied"],
  ["su",          "Compute spent"],
  ["roles",       "Tiers and levers"],
  ["plans",       "Plans"],
  ["experiments", "Experiments"],
  ["approvals",   "Waiting on a person"],
  ["policy",      "What this layer may do"],
  ["timeline",    "Timeline"]
];
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function sev(s){return '<span class="pill '+esc(s)+'">'+esc(s)+'</span>';}
function table(head, rows){
  if(!rows.length) return '<div class="missing">nothing recorded yet</div>';
  return '<div class="scroll"><table><tr>'+head.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr>'
    + rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('') + '</table></div>';
}
function render(key, data){
  // A store the layer cannot see is said so, never drawn as an empty one: an
  // empty table where a missing store belongs is exactly the silence this
  // layer exists to break.
  if(!data.available) return '<div class="missing">Not available &mdash; '+esc(data.reason)+'</div>';
  if(key==="signals"){
    const rows = (data.rows||[]).map(r=>[sev(r.severity), esc(r.signal),
      esc(r.reason), (r.evidence&&r.evidence[0]) ?
        '<code>'+esc(r.evidence[0].artifact_id)+':'+esc(r.evidence[0].line)+'</code>' : '']);
    return '<div class="sub">'+data.n_fired+' firing, '+data.n_unknown+
           ' could not run</div>'+table(["severity","check","why","evidence"], rows);
  }
  if(key==="corrections"){
    const head = '<div class="sub">chain '+(data.chain_ok?'<span class="ok">verified</span>'
      :'<span class="crit">BROKEN at seq '+esc(data.first_bad_seq)+'</span>')+
      ', '+esc(data.chain_length)+' record(s)</div>';
    return head + table(["seq","author","kind","target","reason"],
      (data.rows||[]).map(r=>[esc(r.seq), esc(r.author), esc(r.kind),
        esc(r.target&&r.target.key), esc(r.reason)]));
  }
  if(key==="su"){
    const t = data.total||{};
    return table(["measure","value"], [
      ["spent (SU)", esc(t.su)], ["entries", esc(t.n_entries)],
      ["unknown cost", esc(t.n_unknown)],
      ["envelope", esc((data.budget||{}).su_envelope)],
      ["remaining", esc(JSON.stringify(data.remaining||{}))]]);
  }
  if(key==="roles"){
    const tiers = Object.entries(data.tiers||{}).map(([k,v])=>[esc(k),
      v?esc(v):'<span class="missing">not wired</span>']);
    const levers = (data.lever_menu||[]).map(l=>[esc(l.id), esc(l.control), esc(l.risk)]);
    return '<div class="sub">policy: '+esc(data.policy)+'</div>'
      + table(["tier","model"], tiers)
      + '<div class="sub" style="margin-top:.7rem">Sealed levers an experiment may change</div>'
      + table(["lever","control","risk"], levers);
  }
  if(key==="plans"){
    return table(["version","backend","simulated","experiments"],
      (data.rows||[]).map(r=>[esc(r.version), esc(r.backend),
        r.simulated?'<span class="warn">simulated</span>':'no', esc(r.n_experiments)]));
  }
  if(key==="experiments"){
    return table(["lever","status","verdict","n"],
      (data.rows||[]).map(r=>[esc(r.lever||r.id), esc(r.status),
        esc(r.verdict), esc(r.n)]));
  }
  if(key==="approvals"){
    return '<div class="sub">'+esc(data.n_pending)+' waiting</div>' + table(
      ["status","action","risk","asked by","decided by"],
      (data.rows||[]).map(r=>[
        r.status==="pending" ? '<span class="warn">pending</span>' : esc(r.status),
        esc(r.action), esc(r.risk), esc(r.requested_by),
        r.decided_by ? esc(r.decided_by) : '<span class="missing">—</span>']));
  }
  if(key==="policy"){
    return '<div class="sub">'+esc(data.n_actions)+' action(s) in the catalogue</div>'
      + table(["action","risk","reversible","who may ask"],
        (data.rows||[]).map(r=>[esc(r.action), esc(r.risk),
          r.reversible?'yes':'<span class="warn">no</span>',
          esc((r.allowed_tiers||[]).join(", "))]));
  }
  if(key==="timeline"){
    const miss = (data.sources_unavailable||[]).length
      ? '<div class="missing">not shown: '+esc(data.sources_unavailable.join("; "))+'</div>' : '';
    return miss + table(["when","source","what"],
      (data.events||[]).map(e=>[esc(e.ts), esc(e.source),
        esc(JSON.stringify(e.row).slice(0,160))]));
  }
  return '<pre>'+esc(JSON.stringify(data,null,1)).slice(0,2000)+'</pre>';
}
const grid = document.getElementById("grid");
SECTIONS.forEach(([key,title])=>{
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = '<h3>'+title+'</h3><div id="s_'+key+'">loading&hellip;</div>';
  grid.appendChild(el);
  fetch("/api/brain/"+encodeURIComponent(DOMAIN)+"/"+key)
    .then(r=>r.json())
    .then(d=>{document.getElementById("s_"+key).innerHTML = render(key,d);})
    .catch(e=>{document.getElementById("s_"+key).innerHTML =
      '<div class="crit">could not load: '+esc(e)+'</div>';});
});
</script></body></html>"""


@router.get("/supervision/{domain}")
def page_supervision(domain: str):
    """One page showing everything the layer knows, for a browser and nobody else.

    The whole checklist behind this work is that the demo can be driven from a
    browser with no agent session anywhere: every tier runs as a platform
    service, every result is a route, and this is the page those routes feed.
    It is read-only for the same reason the routes are.
    """
    from fastapi.responses import HTMLResponse
    dom = _domain(domain)
    return HTMLResponse(_PAGE.replace("__DOMAIN__", dom))


def mount(app, ctx: dict):
    _CTX.update(ctx)
    app.include_router(router)
