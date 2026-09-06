"""When a step's own tier is no longer the right authority to decide it (WP6).

Why this exists
----------------
The tiers below this module (the scripted loop, tier-0, tier-1) each have a
ceiling on what they may decide, the same way `policy.py` gives each actor
tier a ceiling on what it may *do*. This module is the other half of that
choke point: given everything one step or round produced, it says which
authority the step's outcome now belongs to -- nobody stays, everybody
escalates, or the ceiling has already been reached and a human has to look.

The escalation authority is an invariant over the step's own artifacts, the
same discipline `signals.py`'s docstring states for signals: a model's own
confidence never decides whether something needs a bigger model or a person,
a rule over what the step actually produced does. `decide()` therefore reads
`context` -- deterministic signals, corrections already on the chain, a
review verdict, resource facts -- and never asks anything for an opinion.

Three trigger families, each escalating from where the previous one stopped:

* **E1 (tier-0 -> tier-1).** Any signal at warn or above; every step end (a
  cheap review that runs whether or not anything else fired); every Nth step
  (a periodic audit on a fixed cadence, so a drift that never crosses a
  single-step threshold is still looked at on a schedule).
* **E2 (tier-1 -> tier-2).** The same signal recurring after two corrections
  already targeted it (tier-1's authority over that parameter has been spent
  twice without closing the issue); a plateau of several rounds that also
  carries a recipe change (the lever pulled did not move anything -- a design
  question, not a parameter one); a correction whose class is `code` or
  `design` (no agent tier may make either kind of edit); a review confident
  below half on a finding already at warn or above; a budget breach (a
  resource fact, not a judgment call).
* **E3 (-> human).** Any R3/R4 action (`policy.py` already refuses these to
  every agent tier; this is that refusal reaching the escalation record too);
  the stop-loss; `ownership_violation`; daily citation validity under 70
  percent. E3 also holds tier-1 autonomy while the deterministic signals keep
  running -- see `held` below, and the module docstring's second half for why
  that split matters.

Context (the one argument to `decide()`)
-----------------------------------------
A plain dict. Every key below is read defensively: absent entirely is
reported as `unknown` naming the key; present but the wrong shape is also
`unknown` (never coerced into a number that looks plausible); present and
valid is checked against the rule. `signals` entries not present in the list
at all are read the way `signals.detect()` itself is meant to be read: a
check that ran and found nothing (`ok`) is simply absent, which is why an
absent `budget` or `ownership_violation` entry is treated as checked-clean
rather than unknown -- the ambiguity that would create (never checked vs.
checked and clean) is the one this module refuses everywhere else, and here
it is resolved by relying on `signals.py`'s own convention rather than
inventing a second one.

    domain, round, step        identity, carried into the ledger row only.
    ts                          caller-supplied timestamp, never read here.
    signals                     this step's `signals.detect()` output: a list
                                 of {signal, severity, value, reason, ...}.
    step_end                    True when this call is the step's cheap
                                 review point.
    domain_config.brain.periodic_audit_every
                                 a domain's own cadence override; falls back
                                 to the file default when absent.
    corrections_history          the full past `brain_corrections` record
                                 (WP5 shape, `digest.py`'s docstring), each
                                 optionally carrying the `signal` it targeted.
    new_corrections              corrections applied for *this* round/step
                                 only, same shape -- what `E2.correction_class`
                                 classifies by `kind`.
    plateau.rounds, .recipe_changed
                                 how many rounds the plateau has spanned and
                                 whether a recipe change falls inside it.
    review.confidence, .findings
                                 the tier-0/tier-1 verdict for this step
                                 (`TIERED_SUPERVISION_PLAN` verdict shape:
                                 `{verdict, findings[{signal, quote,
                                 diagnosis, severity}], corrections, escalate,
                                 confidence}`); `findings[].severity` is
                                 checked against warn, `confidence` against
                                 the floor.
    actions                     actions proposed or taken this step, each
                                 `{"risk": "R3"}` or `{"action": <policy
                                 action id>}` (resolved through
                                 `policy.risk_of`, never re-derived here).
    stop_loss                   `True`/`False`, or `{"active": bool,
                                 "reason": str}`.
    citation_validity            a percent (0-100), or `{"pct": ...}`, or
                                 `{"valid": n, "total": n}`.

Every threshold this module compares against lives in `escalation_rules.json`
next to the reason it has that value -- the same discipline `parallelism.py`
uses and for the same reason: a constant compiled into this file and a copy
in the JSON would drift, and a drifted threshold silently changes what may
escalate. There is no compiled fallback: a rule this file does not declare
makes the corresponding trigger `unknown` and names the missing key, rather
than falling back to a number nobody can see.

Fail-closed and fail-loud
--------------------------
An unknown input never fires a trigger and is never reported as if the check
ran clean -- those are different facts and conflating them is exactly the
failure this whole campaign is a response to (see `signals.py`'s docstring).
`decide()` itself never raises: an internal defect returns `escalate_to:
"human"` naming the failure, because a broken escalator has to fail toward a
person, not toward silence -- the same reasoning behind escalating every
stop-loss trip on principle rather than trusting a downstream reader to
notice a paused domain on their own.

`decide()` reads no clock. Every timestamp comes from `context["ts"]`, so the
same context always produces the same decision and a decision already made
can be replayed and checked, not merely trusted.

Decision record
----------------
    {"escalate_to": "none" | "tier1" | "tier2" | "human",
     "triggers": [{"rule", "family", "status": "fired" | "unknown",
                   "destination", "reason", "missing", "evidence"}, ...],
     "held": {"tier1_autonomy": bool, "signals": bool, "reason": str | None},
     "rationale": str,
     "ledger_row": {...}}

Every trigger that fired is listed, even when a higher one also fired and
decided `escalate_to` -- a single reported destination that hides which of
several rules fired is not auditable. An `unknown` entry names what was
missing or unusable in `missing`, in addition to a full sentence in `reason`,
so a caller can filter on the short name without parsing prose.

CLI (a context file in, the decision out):
    python -m weed_optimizer_framework.tools.brain.escalation decide <context.json>
    python -m weed_optimizer_framework.tools.brain.escalation rules
    python -m weed_optimizer_framework.tools.brain.escalation explain [<rule_id>]
"""
import json
import os

try:                                     # package import (the normal path)
    from . import policy
    from . import signals
except ImportError:                      # direct execution from this directory
    import policy                        # type: ignore
    import signals                       # type: ignore

TOOL_VERSION = "wp6-escalation/1"

RULES_FILENAME = "escalation_rules.json"

# Reused rather than restated: two copies of the severity order, or of the
# risk-tier list, is exactly the drift `signals.py` and `policy.py` both
# warn about for their own constants.
_SEVERITIES = signals.SEVERITIES
_SEV_INDEX = {s: i for i, s in enumerate(_SEVERITIES)}
_RISKS = policy.RISKS

DESTINATIONS = ("none", "tier1", "tier2", "human")
_DEST_RANK = {d: i for i, d in enumerate(DESTINATIONS)}

_ESCALATING_CORRECTION_KINDS = ("code", "design")

_RULES_CACHE = {"path": None, "mtime": None, "data": None}


# --- rule loading (same shape as parallelism.py) -----------------------------
def rules_path():
    """Location of the rule table; overridable for tests and for a second domain."""
    return os.environ.get("WEED_ESCALATION_RULES") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), RULES_FILENAME)


def load_rules(path=None):
    """Rule table as a dict, cached on mtime. Returns {} when it cannot be read.

    An unreadable table is not a reason to invent defaults: every trigger
    below treats a missing key as unknown, never as a compiled-in number.
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


def _threshold(rules, key):
    """(value, reason) for a declared threshold, or (None, None) when absent
    or malformed. `key` missing from the file is the disable path every
    trigger below routes through -- there is no second, compiled-in value."""
    entry = (rules or {}).get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry.get("value"), str(entry.get("reason") or "")
    return None, None


# --- hostile-input coercion ---------------------------------------------------
def _as_finite_number(v):
    """A finite number, or None. Mirrors `parallelism.py`'s own coercion for
    the same reason: a string that looks numeric, a bool, NaN or infinity are
    all shapes a model's JSON can produce, and trusting any of them turns an
    `unknown` into a plausible-looking number nobody actually measured."""
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


def _as_bool_strict(v):
    """`True`/`False` only. Anything else (1, "true", None, a list) is not a
    boolean this module will read as one -- a hostile value must never be
    coerced into firing, or into the false sense that it was checked clean."""
    if v is True or v is False:
        return v
    return None


def _sev_rank(sev):
    """Severity order position, or a rank below every real severity for a
    value that is not one of them -- so a garbage severity can never satisfy
    an `>= warn` comparison by accident."""
    return _SEV_INDEX.get(sev, -99)


def _find_signal(sigs, name):
    """The first entry in a signals list whose `signal` field is `name`, or
    None. Absence is a different fact from severity `ok`, but both are read
    the same way here: `signals.detect()` omits `ok` findings by convention,
    so "not in this list" and "in this list at severity ok" both mean the
    check ran and found nothing to report."""
    for item in sigs:
        if isinstance(item, dict) and item.get("signal") == name:
            return item
    return None


def _resolve_pct(raw):
    """A 0-100 percent from a number, `{"pct"|"percent"|"value": n}`, or
    `{"valid": n, "total": n}`. None for anything else, including a total of
    zero (a ratio with no denominator is not a percent of anything)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return _as_finite_number(raw)
    if isinstance(raw, dict):
        for key in ("pct", "percent", "value"):
            if key in raw:
                return _as_finite_number(raw.get(key))
        valid = _as_finite_number(raw.get("valid"))
        total = _as_finite_number(raw.get("total"))
        if valid is not None and total is not None and total > 0:
            return (valid / total) * 100.0
        return None
    return None


# --- rule catalogue (for explain()/rules(), and the family/destination map) --
_RULE_DEFS = {
    "E1.signal_warn": {
        "family": "E1", "destination": "tier1", "threshold": None,
        "explain": (
            "Any signal at warn or above is a graded finding already, not a "
            "guess: signals.py only reaches that severity from a check that "
            "can cite an artifact address. Leaving it to tier-0 alone would "
            "make a cheap model the last word on something a deterministic "
            "check already flagged, so it goes to tier-1 for a second read."),
    },
    "E1.step_end": {
        "family": "E1", "destination": "tier1", "threshold": None,
        "explain": (
            "A cheap review at every step end costs one call and catches a "
            "divergence before the next step compounds it. It runs whether "
            "or not anything else fired, which is why it is its own rule "
            "rather than folded into the signal-triggered one above."),
    },
    "E1.periodic_audit": {
        "family": "E1", "destination": "tier1",
        "threshold": "periodic_audit_every_steps",
        "explain": (
            "A periodic audit runs on a fixed cadence regardless of whether "
            "a signal or the step-end review found anything, so a slow "
            "drift that never crosses a single-step threshold is still "
            "looked at on a schedule instead of only after it is already a "
            "problem."),
    },
    "E2.recurring_signal": {
        "family": "E2", "destination": "tier2", "threshold": "recurring_after_corrections",
        "explain": (
            "A signal that recurs after two corrections already targeted it "
            "means tier-1's authority over that parameter has been spent "
            "twice without closing the issue -- the pattern a third tier-1 "
            "edit is unlikely to fix, which is what tier-2 exists for."),
    },
    "E2.plateau": {
        "family": "E2", "destination": "tier2", "threshold": "plateau_min_rounds",
        "explain": (
            "A plateau is only escalated when it also carries a recipe "
            "change: a flat metric under an unchanged recipe is not new "
            "information (the plateau signal already says so on its own), "
            "but a flat metric right after a recipe change means the lever "
            "pulled did not move anything -- a design question, not a "
            "parameter one."),
    },
    "E2.correction_class": {
        "family": "E2", "destination": "tier2", "threshold": None,
        "explain": (
            "code and design are two of the correction classes that mean an "
            "edit is beyond a bounded parameter change; no agent tier may "
            "make either kind of change, so a correction of that class is "
            "handed to tier-2 rather than applied by tier-1."),
    },
    "E3.correction_needs_a_person": {
        "family": "E3", "destination": "human", "threshold": None,
        "explain": (
            "The companion to E2.correction_class, and the reason both fire "
            "together. Tier-2 is the right place to ASK for a code or design "
            "change, because that is a planning question; it is not a place "
            "that can MAKE one, because editing code and redesigning an "
            "experiment are human-only actions under the risk tiers. The "
            "benchmark's own scoring rubric records the destination for every "
            "code, design and plan correction in the campaign record as "
            "`human`, and it is right: in each of those cases a person had to "
            "act. Firing only the tier-2 rule would leave the module and the "
            "rubric disagreeing about the same case, which is the kind of "
            "quiet divergence between two statements of one fact that this "
            "layer has already had to repair once."),
    },
    "E2.low_confidence": {
        "family": "E2", "destination": "tier2", "threshold": "confidence_floor",
        "explain": (
            "A review that scores its own finding below the confidence "
            "floor, on a finding already at warn or above, is telling the "
            "layer it is unsure about something the layer already knows is "
            "real. Tier-2 resolves that gap cheaper than an unsure tier-1 "
            "verdict standing as the record."),
    },
    "E2.budget_breach": {
        "family": "E2", "destination": "tier2", "threshold": None,
        "explain": (
            "A round or campaign that has already spent past its own "
            "budget signal is a resource fact, not a judgment call; the "
            "correction it needs (slow down, stop, or re-plan the "
            "campaign) is a tier-2 planning decision, not a tier-1 "
            "parameter edit."),
    },
    "E3.r3_r4_action": {
        "family": "E3", "destination": "human", "threshold": None,
        "explain": (
            "R3 (external side effect, SU-heavy, or design-changing) and R4 "
            "(destructive or irreversible) are the two risk tiers policy.py "
            "already refuses to every agent tier outright. This rule makes "
            "sure the escalation record says so too, rather than only the "
            "authorization gate."),
    },
    "E3.stop_loss": {
        "family": "E3", "destination": "human", "threshold": None,
        "explain": (
            "The stop-loss held a domain silently for six days once already "
            "(2026-08-29, jobs 44727703/44767709) because nothing told a "
            "person it had fired. Escalating to human on every trip, every "
            "time, is the fix for that specific failure, not a new policy "
            "invented for this module."),
    },
    "E3.ownership_violation": {
        "family": "E3", "destination": "human", "threshold": None,
        "explain": (
            "A registry or corrections mirror that no longer matches its "
            "ledger copy means the single-writer convention was broken by "
            "something running under the shared account. Detection alone "
            "cannot tell whether that was benign, so it is a human's fact "
            "to investigate, never an agent tier's to clear on its own "
            "re-read of the same mirror."),
    },
    "E3.citation_validity": {
        "family": "E3", "destination": "human",
        "threshold": "citation_validity_floor_pct",
        "explain": (
            "Below the daily floor, a tier's citations can no longer be "
            "trusted at face value on any of that day's verdicts. The fix "
            "is a person reviewing what that tier has been asserting, not "
            "another automated tier grading the same unreliable citations."),
    },
}


def explain(rule_id):
    """The one-paragraph rationale for one escalation rule. Never raises."""
    key = str(rule_id or "").strip()
    d = _RULE_DEFS.get(key)
    if d is None:
        return {"rule": key, "known": False,
                "reason": "%r is not one of the escalation rules this module "
                          "computes (%s)" % (key, ", ".join(sorted(_RULE_DEFS)))}
    out = {"rule": key, "known": True, "family": d["family"],
           "destination": d["destination"], "explain": d["explain"]}
    thr = d.get("threshold")
    if thr:
        value, reason = _threshold(load_rules(), thr)
        out["threshold_key"] = thr
        out["threshold_value"] = value
        out["threshold_reason"] = reason
    return out


def rules():
    """Effective thresholds and their reasons, for a results file or the CLI."""
    raw = load_rules()
    out = {"tool_version": TOOL_VERSION, "file": rules_path(),
           "values": {}, "reasons": {}, "errors": []}
    if not raw:
        out["errors"].append("rules file missing or unreadable at %s" % rules_path())
        return out
    for key, entry in raw.items():
        if str(key).startswith("_"):
            continue
        if isinstance(entry, dict) and "value" in entry:
            out["values"][key] = entry["value"]
            out["reasons"][key] = str(entry.get("reason") or "")
            if not entry.get("reason"):
                out["errors"].append("%s has no 'reason' declared" % key)
        else:
            out["errors"].append("%s is not a {value, reason} entry" % key)
    return out


# --- trigger record builders --------------------------------------------------
def _fired(rule, reason, evidence=None):
    d = _RULE_DEFS[rule]
    return {"rule": rule, "family": d["family"], "status": "fired",
            "destination": d["destination"], "reason": str(reason),
            "missing": None, "evidence": evidence if evidence is not None else {}}


def _unknown(rule, missing, detail):
    d = _RULE_DEFS[rule]
    return {"rule": rule, "family": d["family"], "status": "unknown",
            "destination": None, "reason": str(detail), "missing": str(missing),
            "evidence": None}


def _field(context, key):
    """(present, value). `context` is already known to be a dict by the time
    any check function runs -- `_decide` validates that once, up front."""
    if key in context:
        return True, context[key]
    return False, None


# --- E1 --------------------------------------------------------------------
def _check_signal_warn(context, rules_table):
    rule = "E1.signal_warn"
    present, raw = _field(context, "signals")
    if not present:
        return [_unknown(rule, "signals", "context carries no 'signals' list "
                         "(this step's signals.detect() output)")]
    if not isinstance(raw, list):
        return [_unknown(rule, "signals", "context['signals'] is not a list")]
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name, sev = item.get("signal"), item.get("severity")
        if not isinstance(name, str) or not name or sev not in _SEVERITIES:
            continue
        if _sev_rank(sev) >= _sev_rank("warn"):
            out.append(_fired(rule, "signal %r fired at %s: %s"
                              % (name, sev, str(item.get("reason") or "")[:200]),
                              evidence={"signal": name, "severity": sev,
                                       "value": item.get("value")}))
    return out


def _check_step_end(context, rules_table):
    rule = "E1.step_end"
    present, raw = _field(context, "step_end")
    if not present:
        return [_unknown(rule, "step_end", "context carries no 'step_end' flag")]
    flag = _as_bool_strict(raw)
    if flag is None:
        return [_unknown(rule, "step_end", "context['step_end'] is not a plain boolean")]
    if flag:
        return [_fired(rule, "this call is at step end, the cheap review point",
                       evidence={"step_end": True})]
    return []


def _resolve_periodic_audit_every(context, rules_table):
    """(n, source) on success, or (None, missing_key) naming what is absent.

    The domain's own cadence, when given, wins over the file default -- the
    file default only fills in domains that have not set one, per WP6's own
    wording ('N from the domain's brain.periodic_audit_every')."""
    dc = context.get("domain_config")
    if isinstance(dc, dict):
        brain_cfg = dc.get("brain")
        if isinstance(brain_cfg, dict):
            override = _as_finite_number(brain_cfg.get("periodic_audit_every"))
            if override is not None and override > 0:
                return int(override), "domain_config.brain.periodic_audit_every"
    value, _reason = _threshold(rules_table, "periodic_audit_every_steps")
    n = _as_finite_number(value)
    if n is None or n <= 0:
        return None, "periodic_audit_every_steps"
    return int(n), "escalation_rules.json"


def _check_periodic_audit(context, rules_table):
    rule = "E1.periodic_audit"
    n, source_or_missing = _resolve_periodic_audit_every(context, rules_table)
    if n is None:
        return [_unknown(rule, source_or_missing,
                         "threshold %r is not declared in %s and no "
                         "domain_config.brain.periodic_audit_every override "
                         "was given" % (source_or_missing,
                                       os.path.basename(rules_path())))]
    present, raw = _field(context, "step")
    if not present:
        return [_unknown(rule, "step", "context carries no 'step' number")]
    step = _as_finite_number(raw)
    if step is None or step < 0 or step != int(step):
        return [_unknown(rule, "step", "context['step'] is not a non-negative integer")]
    step = int(step)
    if step > 0 and step % n == 0:
        return [_fired(rule, "step %d is a multiple of the periodic-audit "
                             "cadence (every %d steps, from %s)"
                             % (step, n, source_or_missing),
                       evidence={"step": step, "every": n, "source": source_or_missing})]
    return []


# --- E2 --------------------------------------------------------------------
def _check_recurring_signal(context, rules_table):
    rule = "E2.recurring_signal"
    value, _reason = _threshold(rules_table, "recurring_after_corrections")
    need = _as_finite_number(value)
    if need is None:
        return [_unknown(rule, "recurring_after_corrections",
                         "threshold 'recurring_after_corrections' is not "
                         "declared in %s" % os.path.basename(rules_path()))]
    need = int(need)

    have_sig, sigs = _field(context, "signals")
    have_hist, hist = _field(context, "corrections_history")
    missing = [k for k, present in
              (("signals", have_sig), ("corrections_history", have_hist)) if not present]
    if missing:
        return [_unknown(rule, "+".join(missing),
                         "context carries no %s" % " or ".join(repr(m) for m in missing))]
    if not isinstance(sigs, list) or not isinstance(hist, list):
        return [_unknown(rule, "signals+corrections_history",
                         "context['signals'] and/or context['corrections_history'] "
                         "is not a list")]

    counts = {}
    for rec in hist:
        if not isinstance(rec, dict):
            continue
        sig_name = rec.get("signal")
        if isinstance(sig_name, str) and sig_name:
            counts[sig_name] = counts.get(sig_name, 0) + 1

    out = []
    for item in sigs:
        if not isinstance(item, dict):
            continue
        name, sev = item.get("signal"), item.get("severity")
        if not isinstance(name, str) or not name or sev not in _SEVERITIES:
            continue
        if sev in ("ok", "unknown"):
            continue                       # not a real recurrence of the signal
        n_prior = counts.get(name, 0)
        if n_prior >= need:
            out.append(_fired(rule, "signal %r recurred after %d prior applied "
                                    "correction(s) already targeted it (>= %d)"
                                    % (name, n_prior, need),
                              evidence={"signal": name, "prior_corrections": n_prior}))
    return out


def _check_plateau(context, rules_table):
    rule = "E2.plateau"
    value, _reason = _threshold(rules_table, "plateau_min_rounds")
    need = _as_finite_number(value)
    if need is None:
        return [_unknown(rule, "plateau_min_rounds",
                         "threshold 'plateau_min_rounds' is not declared in %s"
                         % os.path.basename(rules_path()))]
    need = int(need)

    present, raw = _field(context, "plateau")
    if not present:
        return [_unknown(rule, "plateau", "context carries no 'plateau' block")]
    if not isinstance(raw, dict):
        return [_unknown(rule, "plateau", "context['plateau'] is not a dict")]
    rounds = _as_finite_number(raw.get("rounds"))
    recipe_changed = _as_bool_strict(raw.get("recipe_changed"))
    if rounds is None:
        return [_unknown(rule, "plateau.rounds",
                         "context['plateau']['rounds'] is missing or not a number")]
    if recipe_changed is None:
        return [_unknown(rule, "plateau.recipe_changed",
                         "context['plateau']['recipe_changed'] is missing or "
                         "not a plain boolean")]
    if rounds >= need and recipe_changed:
        return [_fired(rule, "a plateau spanning %s round(s) (>= %d) includes "
                             "a recipe change" % (raw.get("rounds"), need),
                       evidence={"rounds": rounds, "recipe_changed": True})]
    return []


def _check_correction_class(context, rules_table):
    rule = "E2.correction_class"
    present, raw = _field(context, "new_corrections")
    if not present:
        return [_unknown(rule, "new_corrections",
                         "context carries no 'new_corrections' list (the "
                         "corrections applied for this round/step)")]
    if not isinstance(raw, list):
        return [_unknown(rule, "new_corrections",
                         "context['new_corrections'] is not a list")]
    out = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if isinstance(kind, str) and kind in _ESCALATING_CORRECTION_KINDS:
            target = rec.get("target")
            key = target.get("key") if isinstance(target, dict) else None
            detail = ("a %r-class correction was applied (seq=%s, target=%s)"
                      % (kind, rec.get("seq"), key))
            evidence = {"kind": kind, "seq": rec.get("seq"), "target": target}
            out.append(_fired(rule, detail, evidence=evidence))
            # Tier-2 is asked for the change; a person has to make it. Both are
            # true of the same correction, so both are reported and the higher
            # destination wins.
            out.append(_fired("E3.correction_needs_a_person", detail,
                              evidence=evidence))
    return out


def _check_low_confidence(context, rules_table):
    rule = "E2.low_confidence"
    value, _reason = _threshold(rules_table, "confidence_floor")
    floor = _as_finite_number(value)
    if floor is None:
        return [_unknown(rule, "confidence_floor",
                         "threshold 'confidence_floor' is not declared in %s"
                         % os.path.basename(rules_path()))]

    present, raw = _field(context, "review")
    if not present:
        return [_unknown(rule, "review", "context carries no 'review' block "
                         "(verdict confidence and findings)")]
    if not isinstance(raw, dict):
        return [_unknown(rule, "review", "context['review'] is not a dict")]
    conf = _as_finite_number(raw.get("confidence"))
    findings = raw.get("findings")
    if conf is None:
        return [_unknown(rule, "review.confidence",
                         "context['review']['confidence'] is missing or not a number")]
    if not isinstance(findings, list):
        return [_unknown(rule, "review.findings",
                         "context['review']['findings'] is missing or not a list")]
    if conf >= floor:
        return []
    out = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        sev = f.get("severity")
        if sev in _SEVERITIES and _sev_rank(sev) >= _sev_rank("warn"):
            label = str(f.get("signal") or f.get("diagnosis") or "")[:120]
            out.append(_fired(rule, "confidence %.2f is below the %.2f floor "
                                    "on a %s finding (%r)"
                                    % (conf, floor, sev, label),
                              evidence={"confidence": conf, "signal": f.get("signal"),
                                       "severity": sev}))
    return out


def _check_budget_breach(context, rules_table):
    rule = "E2.budget_breach"
    present, raw = _field(context, "signals")
    if not present:
        return [_unknown(rule, "signals", "context carries no 'signals' list")]
    if not isinstance(raw, list):
        return [_unknown(rule, "signals", "context['signals'] is not a list")]
    item = _find_signal(raw, "budget")
    if item is None:
        return []                          # absent == checked clean, signals.py's own convention
    sev = item.get("severity")
    if sev not in _SEVERITIES:
        return [_unknown(rule, "budget", "the 'budget' signal entry carries "
                         "an unrecognised severity")]
    if sev == "unknown":
        return [_unknown(rule, "budget", "the 'budget' signal itself could "
                         "not be computed: %s" % str(item.get("reason") or "")[:200])]
    if sev in ("warn", "crit"):
        return [_fired(rule, "the budget signal fired at %s: %s"
                            % (sev, str(item.get("reason") or "")[:200]),
                      evidence={"severity": sev, "value": item.get("value")})]
    return []


# --- E3 --------------------------------------------------------------------
def _check_r3_r4_action(context, rules_table):
    rule = "E3.r3_r4_action"
    present, raw = _field(context, "actions")
    if not present:
        return [_unknown(rule, "actions", "context carries no 'actions' "
                         "list for this step")]
    if not isinstance(raw, list):
        return [_unknown(rule, "actions", "context['actions'] is not a list")]
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        risk = item.get("risk")
        if risk not in _RISKS:
            action_id = item.get("action")
            risk = None
            if action_id is not None:
                try:
                    risk = policy.risk_of(action_id)
                except Exception:
                    risk = None
        if risk in ("R3", "R4"):
            out.append(_fired(rule, "action %r is risk %s"
                                    % (item.get("action") or item.get("risk"), risk),
                              evidence={"action": item.get("action"), "risk": risk}))
    return out


def _check_stop_loss(context, rules_table):
    rule = "E3.stop_loss"
    present, raw = _field(context, "stop_loss")
    if not present:
        return [_unknown(rule, "stop_loss", "context carries no 'stop_loss' status")]
    active, reason_text = None, ""
    if raw is True or raw is False:
        active = raw
    elif isinstance(raw, dict):
        a = raw.get("active")
        if a is True or a is False:
            active = a
            reason_text = str(raw.get("reason") or "")
    if active is None:
        return [_unknown(rule, "stop_loss", "context['stop_loss'] is neither "
                         "a plain boolean nor a dict with a boolean 'active' key")]
    if active:
        return [_fired(rule, "the domain's stop-loss is tripped%s"
                            % ((": " + reason_text) if reason_text else ""),
                      evidence={"active": True, "reason": reason_text or None})]
    return []


def _check_ownership_violation(context, rules_table):
    rule = "E3.ownership_violation"
    present, raw = _field(context, "signals")
    if not present:
        return [_unknown(rule, "signals", "context carries no 'signals' list")]
    if not isinstance(raw, list):
        return [_unknown(rule, "signals", "context['signals'] is not a list")]
    item = _find_signal(raw, "ownership_violation")
    if item is None:
        return []
    sev = item.get("severity")
    if sev not in _SEVERITIES:
        return [_unknown(rule, "ownership_violation", "the "
                         "'ownership_violation' signal entry carries an "
                         "unrecognised severity")]
    if sev == "unknown":
        return [_unknown(rule, "ownership_violation", "the "
                         "'ownership_violation' signal itself could not be "
                         "computed: %s" % str(item.get("reason") or "")[:200])]
    if sev == "ok":
        return []
    return [_fired(rule, "the ownership_violation signal fired at %s: %s"
                        % (sev, str(item.get("reason") or "")[:200]),
                  evidence={"severity": sev, "value": item.get("value")})]


def _check_citation_validity(context, rules_table):
    rule = "E3.citation_validity"
    value, _reason = _threshold(rules_table, "citation_validity_floor_pct")
    floor = _as_finite_number(value)
    if floor is None:
        return [_unknown(rule, "citation_validity_floor_pct",
                         "threshold 'citation_validity_floor_pct' is not "
                         "declared in %s" % os.path.basename(rules_path()))]
    present, raw = _field(context, "citation_validity")
    if not present:
        return [_unknown(rule, "citation_validity",
                         "context carries no 'citation_validity' figure")]
    pct = _resolve_pct(raw)
    if pct is None:
        return [_unknown(rule, "citation_validity",
                         "context['citation_validity'] has an unusable shape "
                         "(want a number, {'pct': ...}, or "
                         "{'valid': ..., 'total': ...})")]
    if pct < floor:
        return [_fired(rule, "daily citation validity is %.1f%%, below the "
                            "%.1f%% floor" % (pct, floor),
                      evidence={"pct": pct})]
    return []


_CHECKS = (
    _check_signal_warn, _check_step_end, _check_periodic_audit,
    _check_recurring_signal, _check_plateau, _check_correction_class,
    _check_low_confidence, _check_budget_breach,
    _check_r3_r4_action, _check_stop_loss, _check_ownership_violation,
    _check_citation_validity,
)


# --- the decision --------------------------------------------------------------
def _rationale(escalate_to, fired, triggers):
    if escalate_to == "none":
        unknown_n = sum(1 for t in triggers if t["status"] == "unknown")
        if unknown_n:
            return ("No escalation: every checkable trigger is clean, though "
                    "%d could not be checked and are reported as unknown, "
                    "not as clean." % unknown_n)
        return "No escalation: every checked trigger is clean."
    names = ", ".join(sorted(set(t["rule"] for t in fired)))
    return "Escalated to %s: %d trigger(s) fired (%s)." % (escalate_to, len(fired), names)


def _decide(context):
    if not isinstance(context, dict):
        raise TypeError("context must be a dict, got %s" % type(context).__name__)

    rules_table = load_rules()
    triggers = []
    for check in _CHECKS:
        triggers.extend(check(context, rules_table))

    fired = [t for t in triggers if t["status"] == "fired"]
    escalate_to = (max(fired, key=lambda t: _DEST_RANK[t["destination"]])["destination"]
                  if fired else "none")

    e3_fired = [t for t in fired if t["family"] == "E3"]
    if e3_fired:
        names = ", ".join(sorted(set(t["rule"] for t in e3_fired)))
        held = {"tier1_autonomy": True, "signals": False,
                "reason": ("E3 fired (%s): tier-1 autonomy is paused pending "
                          "a human review; the deterministic signals keep "
                          "running, because a supervisor that goes quiet at "
                          "the moment its authority is suspended is how an "
                          "incident stops being observed." % names)}
    else:
        held = {"tier1_autonomy": False, "signals": False, "reason": None}

    ledger_row = {
        "kind": "escalation",
        "ts": context.get("ts"),
        "domain": context.get("domain"),
        "round": context.get("round"),
        "step": context.get("step"),
        "escalate_to": escalate_to,
        "fired": [t["rule"] for t in fired],
        "unknown": [t["rule"] for t in triggers if t["status"] == "unknown"],
        "held": held["tier1_autonomy"],
    }

    return {"escalate_to": escalate_to, "triggers": triggers, "held": held,
            "rationale": _rationale(escalate_to, fired, triggers),
            "ledger_row": ledger_row}


def decide(context):
    """The escalation decision for one step/round's `context`. Never raises.

    An internal defect (a check function's own bug, a context so malformed
    the top-level type check rejects it) still has to produce a decision --
    the fallback below escalates to human and holds tier-1 autonomy, because
    a broken escalator has to fail toward a person, not toward silence."""
    try:
        return _decide(context)
    except Exception as exc:
        safe = context if isinstance(context, dict) else {}
        reason = ("the escalation function itself failed (%s: %s); "
                 "escalating to a person rather than staying silent"
                 % (type(exc).__name__, str(exc)[:200]))
        return {
            "escalate_to": "human",
            "triggers": [],
            "held": {"tier1_autonomy": True, "signals": False, "reason": reason},
            "rationale": "Escalated to human: %s" % reason,
            "ledger_row": {"kind": "escalation", "ts": safe.get("ts"),
                          "domain": safe.get("domain"), "round": safe.get("round"),
                          "step": safe.get("step"), "escalate_to": "human",
                          "fired": [], "unknown": [], "held": True,
                          "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])},
        }


# --- CLI -----------------------------------------------------------------------
def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None):
    """`decide <context.json>` | `rules` | `explain [<rule_id>]`.

    Exit 1 when the decision escalates past `none`, so a shell step can gate
    on it; exit 0 for no escalation and for the other commands.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="escalation", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    dec = sub.add_parser("decide", help="the escalation decision for a context JSON file")
    dec.add_argument("context")
    sub.add_parser("rules", help="effective thresholds and their reasons")
    exp = sub.add_parser("explain", help="the rationale for one rule, or all")
    exp.add_argument("rule", nargs="?")
    args = ap.parse_args(argv)

    if args.cmd == "rules":
        _dump(rules())
        return 0
    if args.cmd == "explain":
        _dump(explain(args.rule) if args.rule else [explain(r) for r in sorted(_RULE_DEFS)])
        return 0
    if args.cmd != "decide":
        ap.print_help()
        return 0
    try:
        with open(args.context, "r", encoding="utf-8") as fh:
            context = json.load(fh)
    except Exception as exc:
        print("cannot read %s: %s" % (args.context, exc))
        return 2
    result = decide(context)
    _dump(result)
    return 0 if result["escalate_to"] == "none" else 1


if __name__ == "__main__":
    raise SystemExit(main())
