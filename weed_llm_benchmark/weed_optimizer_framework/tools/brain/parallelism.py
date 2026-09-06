"""How many units to run in parallel, and whether to run at all (v3.28.0).

Why this exists
---------------
The supervision layer is allowed to spend a shared GPU allocation. Deciding
`one agent or several, and how many` is therefore a resource and evidence
decision before it is a performance one, and it is the decision most likely to
be made badly by a model asked for a number with no ceiling.

The rule this module implements is the same one the rest of the layer uses: an
invariant computes the ceiling, the model chooses inside it. A caller may ask
for eight trainers; if the remaining SU envelope supports one, it gets one, and
the decision says which constraint bound it. Nothing here consults a model, and
nothing a model returns can raise a cap.

Two of the caps are not optimisations and must not be read as such:

* **A claim runs at the seed count or not at all.** A round-to-round delta is
  reportable only as mean+/-std over the campaign's three seeds; the sealed
  `noise_floor` was derived that way. When the caps allow fewer seeds than a
  claim needs, `plan()` returns `feasible: false` rather than a smaller number.
  Returning 1 would produce a run that looks like a result and is not one --
  the reduction is not a cheaper experiment, it is a different and unpublishable
  one.
* **The collector is fixed at one.** It writes the shared dataset registry;
  two collectors racing on it is the single-writer violation the correction
  channel exists to detect. Parallelism there buys no data and risks the
  campaign's bookkeeping.

Fail-closed behaviour
---------------------
Every cap has a named source. A resource fact that is missing does not widen a
cap: the source contributes the role floor and is listed in `unknown`, so a
decision made without knowing the queue depth is visibly a decision made without
knowing the queue depth. `plan()` does not raise; an internal error yields
`feasible: false` naming the failure.

Every number lives in `parallelism_rules.json` next to the reason it has that
value, read at call time and cached on mtime. There are no compiled fallbacks,
for the reason `signals.py` states about thresholds: two copies of a governing
number drift, and a drifted cap silently changes what the layer may do.

Decision record
---------------
    {"n_parallel": int,          # 0 when not feasible
     "feasible": bool,
     "role": str,
     "purpose": "routine" | "claim" | "exploration",
     "requested": int,
     "binding": str | None,      # the source that set the ceiling
     "caps": [{"source", "cap", "reason"}, ...],
     "slots": [{"index", "role", "seed", "est_su"}, ...],
     "refusals": [str, ...],     # requests denied, each with its reason
     "unknown": [str, ...],      # facts that were missing, each named
     "rationale": str,           # one English sentence a human can act on
     "citation": {...} | None,
     "est_su_total": float | None}

`ledger_row()` turns that into the append-only row the attribution ledger keeps,
so a later analysis can ask what the layer decided and what it cost, and
`plan()` itself stays free of wall-clock reads -- the caller stamps the time, so
the same inputs always produce the same decision.
"""

import json
import os
import sys

RULES_FILENAME = "parallelism_rules.json"
PURPOSES = ("routine", "claim", "exploration")

_RULES_CACHE = {"path": None, "mtime": None, "data": None}


# --- rule loading ------------------------------------------------------------

def rules_path():
    """Location of the rule table; overridable for tests and for a second domain."""
    return os.environ.get("WEED_PARALLELISM_RULES") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), RULES_FILENAME)


def load_rules(path=None):
    """Rule table as a dict, cached on mtime. Returns {} when it cannot be read.

    An unreadable table is not a reason to invent defaults: every caller below
    treats a missing key as an unknown that collapses to the floor.
    """
    p = path or rules_path()
    try:
        st = os.stat(p)
    except Exception:
        return {}
    c = _RULES_CACHE
    if c["path"] == p and c["mtime"] == st.st_mtime and c["data"] is not None:
        return c["data"]
    try:
        with open(p, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    c.update({"path": p, "mtime": st.st_mtime, "data": data})
    return data


def _value(rules, key):
    """A declared rule value, or None when the table does not carry it.

    Values are stored as {"value": ..., "reason": ...} so the reason travels
    with the number and cannot be edited away separately.
    """
    entry = (rules or {}).get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return None


def _reason(rules, key, default=""):
    entry = (rules or {}).get(key)
    if isinstance(entry, dict):
        return str(entry.get("reason") or default)
    return default


def role_rule(role, rules=None):
    """The declared rule for one role, or None when the table has no such role."""
    r = (load_rules() if rules is None else rules).get("roles") or {}
    entry = r.get(str(role))
    return entry if isinstance(entry, dict) else None


def explain(role, rules=None):
    """Human-readable account of one role's ceiling and preconditions."""
    entry = role_rule(role, rules)
    if not entry:
        return {"role": role, "known": False,
                "reason": "no rule is declared for this role"}
    return {"role": role, "known": True,
            "floor": entry.get("floor"), "ceiling": entry.get("ceiling"),
            "requires": list(entry.get("requires") or []),
            "mutates": list(entry.get("mutates") or []),
            "reason": entry.get("reason") or ""}


# --- caps --------------------------------------------------------------------

def _as_number(v):
    """A finite number, or None. Strings that look numeric are NOT accepted.

    A cap computed from a string that arrived from a model's JSON is a cap
    computed from unvalidated input; refusing the coercion keeps the failure
    visible as an unknown instead of as a plausible number.
    """
    if isinstance(v, bool) or v is None:
        return None
    if not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _cap_su(resources, proposal, rules, floor):
    margin = _as_number(_value(rules, "su_safety_margin"))
    remaining = _as_number((resources or {}).get("su_remaining"))
    per_unit = _as_number((proposal or {}).get("est_su_per_unit"))
    if margin is None:
        return None, "rule su_safety_margin is not declared", None
    if remaining is None:
        return None, "su_remaining is unknown", None
    if per_unit is None or per_unit <= 0:
        return None, "est_su_per_unit is unknown", None
    usable = remaining * (1.0 - margin)
    cap = int(usable // per_unit)
    return max(cap, 0), _reason(rules, "su_safety_margin"), usable


def _cap_queue(resources, rules, floor):
    bands = _value(rules, "queue_depth_bands")
    pending = _as_number((resources or {}).get("queue_pending"))
    if not isinstance(bands, list) or not bands:
        return None, "rule queue_depth_bands is not declared", None
    if pending is None:
        return None, "queue_pending is unknown", None
    for band in bands:
        if not isinstance(band, dict):
            continue
        mx = band.get("max_pending")
        cap = _as_number(band.get("cap"))
        if cap is None:
            continue
        if mx is None or pending <= _as_number(mx):
            return int(cap), _reason(rules, "queue_depth_bands"), pending
    return None, "no queue depth band matched", pending


def _cap_disk(resources, rules, floor):
    per_unit = _as_number(_value(rules, "min_free_gb_per_unit"))
    free_gb = _as_number((resources or {}).get("free_gb"))
    if per_unit is None or per_unit <= 0:
        return None, "rule min_free_gb_per_unit is not declared", None
    if free_gb is None:
        return None, "free_gb is unknown", None
    return max(int(free_gb // per_unit), 0), _reason(rules, "min_free_gb_per_unit"), free_gb


def _cap_role(role, resources, rules):
    """Ceiling from the role's own rule, lowered to the floor when a precondition fails.

    A precondition is a property of the platform, not a preference: `train`
    above one unit needs job-scoped artefacts, because without them two
    concurrent trainers write metric files that a single glob matches and the
    second run reads as a second sample of the first.
    """
    entry = role_rule(role, rules)
    if not entry:
        return None, "no rule is declared for role %r" % (role,), []
    floor = _as_number(entry.get("floor"))
    ceiling = _as_number(entry.get("ceiling"))
    if floor is None or ceiling is None:
        return None, "role %r declares no floor/ceiling" % (role,), []
    unmet = []
    res = resources or {}
    for req in (entry.get("requires") or []):
        if res.get(req) is not True:
            unmet.append(str(req))
    if unmet:
        return int(floor), ("preconditions not satisfied: %s" % ", ".join(unmet)), unmet
    return int(ceiling), str(entry.get("reason") or ""), []


def caps(role, proposal, resources, rules=None):
    """Every cap with its source and reason, most restrictive first.

    A source whose input is missing reports `cap: None`; callers treat that as
    the role floor and name it, so an absent fact narrows the decision instead
    of silently widening it.
    """
    rules = load_rules() if rules is None else rules
    entry = role_rule(role, rules) or {}
    floor = _as_number(entry.get("floor"))
    floor = 1 if floor is None else int(floor)

    out = []
    cap, why, _ = _cap_role(role, resources, rules)
    out.append({"source": "role_rule", "cap": cap, "reason": why})
    cap, why, _ = _cap_su(resources, proposal, rules, floor)
    out.append({"source": "su_envelope", "cap": cap, "reason": why})
    cap, why, _ = _cap_queue(resources, rules, floor)
    out.append({"source": "queue_depth", "cap": cap, "reason": why})
    cap, why, _ = _cap_disk(resources, rules, floor)
    out.append({"source": "quota_headroom", "cap": cap, "reason": why})

    out.sort(key=lambda c: (c["cap"] is not None,
                            c["cap"] if c["cap"] is not None else 0))
    return out


# --- the decision ------------------------------------------------------------

def _citation_ok(citation):
    """A citation is a signal name or plan step plus a verbatim quote.

    The quote is the part that matters: an uncited request to spend the shared
    allocation is indistinguishable from a guess, which is the same reason the
    correction channel refuses an LLM-authored change with no quote.
    """
    if not isinstance(citation, dict):
        return False
    has_source = bool(str(citation.get("signal") or "").strip()) or \
        bool(str(citation.get("plan_step") or "").strip())
    has_quote = bool(str(citation.get("quote") or "").strip())
    return has_source and has_quote


def plan(domain, proposal, resources, config=None, rules=None):
    """Decide how many units may run, or refuse, with the binding constraint named.

    `proposal` is what a tier proposed: {role, purpose, n_requested, citation,
    est_su_per_unit, requested_by, reason}. `resources` is what the platform
    measured. Nothing here reads the clock or the network, so the same inputs
    always yield the same decision and a decision can be replayed.
    """
    try:
        return _plan(domain, proposal, resources, config, rules)
    except Exception as e:                       # never raise into a scheduler tick
        # The handler must survive a proposal that is not a dict at all. An
        # earlier version read fields off `proposal` here and raised a second
        # time, which would have turned a malformed request into a crashed tick.
        safe = proposal if isinstance(proposal, dict) else {}
        return {"n_parallel": 0, "feasible": False, "role": str(safe.get("role")),
                "purpose": str(safe.get("purpose") or "routine"),
                "requested": 0, "binding": None, "caps": [], "slots": [],
                "refusals": ["planning failed: %s" % (e,)], "unknown": [],
                "rationale": "No parallelism was planned because the planner itself failed.",
                "citation": None, "est_su_total": None, "domain": domain}


def _plan(domain, proposal, resources, config, rules):
    rules = load_rules() if rules is None else rules
    proposal = proposal or {}
    resources = resources or {}

    role = str(proposal.get("role") or "").strip()
    purpose = str(proposal.get("purpose") or "routine").strip()
    requested_raw = _as_number(proposal.get("n_requested"))
    requested = int(requested_raw) if requested_raw is not None and requested_raw >= 0 else None

    refusals, unknown = [], []
    entry = role_rule(role, rules)
    if entry is None:
        return {"n_parallel": 0, "feasible": False, "role": role, "purpose": purpose,
                "requested": requested or 0, "binding": "role_rule", "caps": [], "slots": [],
                "refusals": ["role %r has no declared rule, so no unit may run" % (role,)],
                "unknown": [], "citation": None, "est_su_total": None, "domain": domain,
                "rationale": "Refused: %r is not a role this layer is allowed to schedule." % (role,)}
    if purpose not in PURPOSES:
        refusals.append("purpose %r is not one of %s; treated as routine"
                        % (purpose, "/".join(PURPOSES)))
        purpose = "routine"

    floor = int(_as_number(entry.get("floor")) or 1)
    computed = caps(role, proposal, resources, rules)
    for c in computed:
        if c["cap"] is None:
            unknown.append("%s: %s" % (c["source"], c["reason"]))

    # An unknown source contributes the floor. This is the fail-closed step: it
    # is why a decision taken with no queue reading can never exceed one unit.
    effective = [(c["source"], floor if c["cap"] is None else int(c["cap"])) for c in computed]
    allowed = min(v for _, v in effective)
    binding = min(effective, key=lambda kv: kv[1])[0]

    if requested is None:
        refusals.append("n_requested was missing or not a number; treated as the floor")
        requested = floor

    cite_above = _as_number(_value(rules, "citation_required_above"))
    citation = proposal.get("citation") if _citation_ok(proposal.get("citation")) else None
    if cite_above is not None and requested > cite_above and citation is None:
        refusals.append("request for %d units was reduced to %d: more than %d units needs a "
                        "signal or plan step and a verbatim quote, and none was supplied"
                        % (requested, int(cite_above), int(cite_above)))
        requested = int(cite_above)

    seeds_needed = _as_number(_value(rules, "seeds_for_claim"))
    if purpose == "claim":
        if seeds_needed is None:
            return _refuse(domain, role, purpose, requested, binding, computed, refusals,
                           unknown, citation,
                           "rule seeds_for_claim is not declared, so a claim cannot be sized")
        need = int(seeds_needed)
        if allowed < need:
            return _refuse(
                domain, role, purpose, requested, binding, computed, refusals, unknown, citation,
                "A claim needs %d seeds and the caps allow %d; %s is the binding constraint. "
                "Running fewer seeds would produce a run that reads as a result and is not one."
                % (need, allowed, binding))
        n = need
    else:
        n = max(min(requested, allowed), 0)

    if n <= 0:
        return _refuse(domain, role, purpose, requested, binding, computed, refusals, unknown,
                       citation,
                       "No unit may run: %s allows %d." % (binding, allowed))

    per_unit = _as_number(proposal.get("est_su_per_unit"))
    seeds = list(proposal.get("seeds") or [])
    slots = []
    for i in range(n):
        slots.append({"index": i + 1, "role": role,
                      "seed": seeds[i] if i < len(seeds) else None,
                      "est_su": per_unit})
    est_total = round(per_unit * n, 3) if per_unit is not None else None

    if n == 1:
        rationale = ("One unit: %s. %s allows %d." %
                     ("a claim would need more, but this is a routine step"
                      if purpose != "claim" else "the seed count is one",
                      binding, allowed))
        if entry.get("ceiling") == 1:
            rationale = "One unit, always, for %s: %s" % (role, entry.get("reason") or "")
    else:
        rationale = ("%d units in parallel for %s (%s): %s is the binding constraint at %d."
                     % (n, role, purpose, binding, allowed))

    return {"n_parallel": n, "feasible": True, "role": role, "purpose": purpose,
            "requested": requested, "binding": binding, "caps": computed, "slots": slots,
            "refusals": refusals, "unknown": unknown, "rationale": rationale,
            "citation": citation, "est_su_total": est_total, "domain": domain}


def _refuse(domain, role, purpose, requested, binding, computed, refusals, unknown,
            citation, why):
    refusals = list(refusals) + [why]
    return {"n_parallel": 0, "feasible": False, "role": role, "purpose": purpose,
            "requested": requested, "binding": binding, "caps": computed, "slots": [],
            "refusals": refusals, "unknown": unknown, "citation": citation,
            "est_su_total": None, "domain": domain, "rationale": why}


def ledger_row(decision, actor, ts, review_id=None):
    """The append-only attribution row for one parallelism decision.

    `ts` is supplied by the caller: keeping the clock out of `plan()` is what
    lets a decision be replayed from its inputs and compared against what was
    actually submitted.
    """
    d = decision or {}
    return {"kind": "parallelism", "ts": ts, "actor": actor, "review_id": review_id,
            "domain": d.get("domain"), "role": d.get("role"), "purpose": d.get("purpose"),
            "requested": d.get("requested"), "granted": d.get("n_parallel"),
            "feasible": d.get("feasible"), "binding": d.get("binding"),
            "est_su_total": d.get("est_su_total"),
            "refusals": list(d.get("refusals") or []),
            "unknown": list(d.get("unknown") or []),
            "cited": bool(d.get("citation")), "rationale": d.get("rationale")}


# --- CLI ---------------------------------------------------------------------

def _main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        # A hand-rolled dispatcher still owes a person `--help`; printing
        # "unknown command '--help'" is how a working tool looks broken.
        print("usage: parallelism.py {plan|caps|explain|rules} [json]", file=sys.stderr)
        return 0 if len(argv) > 1 else 2
    cmd = argv[1]
    if cmd == "rules":
        print(json.dumps(load_rules(), indent=2, sort_keys=True))
        return 0
    if cmd == "explain":
        role = argv[2] if len(argv) > 2 else ""
        print(json.dumps(explain(role), indent=2, sort_keys=True))
        return 0
    payload = json.loads(argv[2]) if len(argv) > 2 else json.loads(sys.stdin.read() or "{}")
    if cmd == "caps":
        print(json.dumps(caps(payload.get("proposal", {}).get("role", ""),
                              payload.get("proposal", {}),
                              payload.get("resources", {})), indent=2, sort_keys=True))
        return 0
    if cmd == "plan":
        print(json.dumps(plan(payload.get("domain"), payload.get("proposal", {}),
                              payload.get("resources", {}), payload.get("config")),
                         indent=2, sort_keys=True))
        return 0
    print("unknown command %r" % (cmd,), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
